"""
AI Research Agents - Main Entry Point
Provides easy access to the research agent system
"""

from .research_agents.base_agent import BaseResearchAgent
from .research_agents.it_agent import ITResearchAgent
from .research_agents.pharma_agent import PharmaResearchAgent
from .research_agents.selector import AgentSelector
from .research_agents.models import ResearchPlan, ResearchReport, AgentState

# Import typing modules
from typing import List, Dict, Any, TypedDict

# Import Django settings
from django.conf import settings

__all__ = [
    'BaseResearchAgent',
    'ITResearchAgent',
    'PharmaResearchAgent',
    'AgentSelector',
    'ResearchPlan',
    'ResearchReport',
    'AgentState'
]

# LangChain imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import Pinecone as LangchainPinecone
from langchain_tavily import TavilySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate

from pydantic import BaseModel, Field

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Initialize Pinecone with new API
from pinecone import Pinecone as PineconeClient

# Define the API key directly to avoid issues
pinecone_api_key = "pcsk_6d1bNh_Ez7hr1V9BCki23dipaUVvD5gpFYztCftysGCqeLuPh53AsK1eUMesjEHyv39KWB"

class ResearchPlan(BaseModel):
    """Structure for research planning"""
    main_topics: List[str] = Field(description="Main topics to research")
    sub_topics: List[str] = Field(description="Sub-topics for each main topic")
    research_questions: List[str] = Field(description="Specific questions to answer")
    expected_sources: List[str] = Field(description="Types of sources to consult")

class ResearchReport(BaseModel):
    """Structure for research reports"""
    content: str = Field(description="Complete research report in markdown format")
    sources: List[str] = Field(description="Sources consulted")

class AgentState(TypedDict):
    """State management for LangGraph agents"""
    topic: str
    plan: ResearchPlan
    documents: List[Document]
    analysis: Dict[str, Any]
    report: ResearchReport
    agent_type: str
    current_step: str
    error: str

class BaseResearchAgent:
    """Base class for all research agents using LangGraph"""
    
    def __init__(self):
        # Initialize LLM and embeddings
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.7,
            openai_api_key=settings.OPENAI_API_KEY
        )
        
        # Create custom embeddings adapter for 512 dimensions
        class CustomDimensionEmbeddings:
            """Adapter class to force 512 dimensions for OpenAI embeddings"""
            
            def __init__(self, api_key):
                self.model = "text-embedding-3-small (custom 512-dim adapter)"
                self.api_key = api_key
                self._original_embeddings = OpenAIEmbeddings(
                    openai_api_key=api_key,
                    model="text-embedding-ada-002"  # This model works reliably
                )
            
            def embed_query(self, text):
                # Get the original embedding
                original_embedding = self._original_embeddings.embed_query(text)
                
                # Use the first 512 dimensions only
                truncated_embedding = original_embedding[:512]
                return truncated_embedding
            
            def embed_documents(self, documents):
                # Get the original embeddings
                original_embeddings = self._original_embeddings.embed_documents(documents)
                
                # Truncate each embedding to 512 dimensions
                truncated_embeddings = [emb[:512] for emb in original_embeddings]
                return truncated_embeddings
                
            # Pass through any other attribute access to the original embeddings
            def __getattr__(self, name):
                return getattr(self._original_embeddings, name)
        
        # Use our custom embeddings adapter
        self.embeddings = CustomDimensionEmbeddings(settings.OPENAI_API_KEY)
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        
        # Initialize Tavily Search with explicit api_key parameter
        self.search_tool = TavilySearch(tavily_api_key=settings.TAVILY_API_KEY)
        
        # Initialize vector store
        self.vector_store = None
        self.pinecone_client = None
        self.setup_vector_store()
        
        # Initialize LangGraph
        self.memory = MemorySaver()
        self.graph = self._create_agent_graph()
    
    def setup_vector_store(self):
        """Setup Pinecone vector store"""
        try:
            # Set environment variable for Pinecone API key
            import os
            os.environ["PINECONE_API_KEY"] = pinecone_api_key
            
            # Initialize Pinecone client with direct API key
            pc = PineconeClient(
                api_key=pinecone_api_key
            )
            
            # Check if index exists, create if not
            index_name = settings.PINECONE_INDEX_NAME
            existing_indexes = [index.name for index in pc.list_indexes()]
            
            if index_name not in existing_indexes:
                print(f"Creating Pinecone index: {index_name}")
                # Get the dimension from the embeddings object
                test_embedding = self.embeddings.embed_query("test")
                embedding_dimension = len(test_embedding)
                
                pc.create_index(
                    name=index_name,
                    dimension=embedding_dimension,  # Use actual embedding dimension
                    metric="cosine",
                    spec={
                        "serverless": {
                            "cloud": "aws",
                            "region": "us-east-1"
                        }
                    }
                )
            
            # Create vector store using the updated API
            self.vector_store = LangchainPinecone.from_existing_index(
                index_name=settings.PINECONE_INDEX_NAME,
                embedding=self.embeddings,
                text_key="text"
            )
            
            print(f"Successfully connected to Pinecone index: {settings.PINECONE_INDEX_NAME}")
        except Exception as e:
            print(f"Error setting up vector store: {e}")
            self.vector_store = None
    
    def _create_agent_graph(self) -> StateGraph:
        """Create the LangGraph workflow for the agent"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("plan_research", self._plan_research_node)
        workflow.add_node("gather_information", self._gather_information_node)
        workflow.add_node("analyze_information", self._analyze_information_node)
        workflow.add_node("generate_report", self._generate_report_node)
        
        # Add edges
        workflow.add_edge("plan_research", "gather_information")
        workflow.add_edge("gather_information", "analyze_information")
        workflow.add_edge("analyze_information", "generate_report")
        workflow.add_edge("generate_report", END)
        
        # Set entry point
        workflow.set_entry_point("plan_research")
        
        return workflow.compile(checkpointer=self.memory)
    
    def _plan_research_node(self, state: AgentState) -> AgentState:
        """Node for planning research"""
        try:
            state["current_step"] = "Planning research..."
            print(f"Planning research for: {state['topic']}")
            
            plan = self.plan_research(state["topic"])
            state["plan"] = plan
            state["current_step"] = "Research planned successfully"
            
        except Exception as e:
            state["error"] = f"Planning failed: {str(e)}"
            state["current_step"] = "Planning failed"
        
        return state
    
    def _gather_information_node(self, state: AgentState) -> AgentState:
        """Gather information from both web search and knowledge base"""
        try:
            print("Gathering comprehensive information...")
            
            # Get relevant documents from knowledge base (RAG)
            kb_docs = []
            if self.vector_store:
                try:
                    # Determine sector for targeted retrieval
                    sector = "IT" if "IT" in state["agent_type"] else "Pharma"
                    kb_docs = self.vector_store.similarity_search(
                        state["topic"], 
                        k=10,  # Get more documents for comprehensive coverage
                        filter={"sector": sector}
                    )
                    print(f"Retrieved {len(kb_docs)} relevant documents from knowledge base")
                except Exception as e:
                    print(f"Knowledge base retrieval error: {e}")
            
            # Perform web search
            web_results = []
            try:
                search_query = f"{state['topic']} {sector if 'sector' in locals() else ''}"
                web_results = self.search_tool.invoke({
                    "query": search_query,
                    "max_results": 25  # Ensure parameter is properly applied
                })
                print(f"Retrieved {len(web_results)} web search results")
            except Exception as e:
                print(f"Web search error: {e}")
            
            # Combine both sources
            all_documents = []
            
            # Add knowledge base documents
            for doc in kb_docs:
                all_documents.append({
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "Knowledge Base"),
                    "type": "knowledge_base",
                    "sector": doc.metadata.get("sector", "Unknown")
                })
            
            # Add web search results
            for result in web_results:
                # Handle different types of results (dict or string)
                if isinstance(result, dict):
                    all_documents.append({
                        "content": result.get("content", ""),
                        "source": result.get("url", "Web Search"),
                        "type": "web_search",
                        "sector": "web"
                    })
                else:
                    # Handle string or other types
                    all_documents.append({
                        "content": str(result),
                        "source": "Web Search",
                        "type": "web_search",
                        "sector": "web"
                    })
            
            print(f"Total documents gathered: {len(all_documents)}")
            
            # Update state
            state["documents"] = all_documents
            state["current_step"] = "Information gathered successfully"
            
            return state
            
        except Exception as e:
            state["error"] = f"Error gathering information: {str(e)}"
            state["current_step"] = "Failed to gather information"
            return state
    
    def _analyze_information_node(self, state: AgentState) -> AgentState:
        """Analyze gathered information from both knowledge base and web search"""
        try:
            print("Analyzing information...")
            
            # Separate knowledge base and web search documents
            kb_docs = [doc for doc in state["documents"] if doc.get("type") == "knowledge_base"]
            web_docs = [doc for doc in state["documents"] if doc.get("type") == "web_search"]
            
            print(f"Analyzing {len(kb_docs)} knowledge base documents and {len(web_docs)} web search results")
            
            # Create comprehensive analysis prompt
            analysis_prompt = f"""
            Analyze the following information about: {state['topic']}
            
            Knowledge Base Documents ({len(kb_docs)}):
            {chr(10).join([f"- {str(doc['content'])[:200]}... (Source: {str(doc['source'])})" for doc in kb_docs[:5]])}
            
            Web Search Results ({len(web_docs)}):
            {chr(10).join([f"- {str(doc['content'])[:200]}... (Source: {str(doc['source'])})" for doc in web_docs[:5]])}
            
            Please provide a comprehensive analysis that:
            1. Identifies key insights from both knowledge base and web search
            2. Highlights sector-specific trends and developments
            3. Notes any contradictions or complementary information between sources
            4. Provides context and relevance to the research topic
            
            Analysis:
            """
            
            # Get analysis from LLM
            analysis_response = self.llm.invoke(analysis_prompt)
            
            state["analysis"] = {
                "kb_documents_analyzed": len(kb_docs),
                "web_results_analyzed": len(web_docs),
                "key_insights": analysis_response.content,
                "sources_used": [doc["source"] for doc in state["documents"]]
            }
            
            state["current_step"] = "Analysis completed successfully"
            print("Analysis completed successfully")
            
            return state
            
        except Exception as e:
            state["error"] = f"Analysis failed: {str(e)}"
            state["current_step"] = "Analysis failed"
            return state
    
    def _generate_report_node(self, state: AgentState) -> AgentState:
        """Generate comprehensive report using both knowledge base and web search insights"""
        try:
            print("Generating comprehensive report...")
            
            # Create enhanced report generation prompt
            report_prompt = f"""
            Generate a comprehensive, detailed research report on: {state['topic']}
            
            Research Plan: {state.get('plan', 'Not available')}
            
            Analysis Summary:
            - Knowledge Base Documents Analyzed: {state['analysis'].get('kb_documents_analyzed', 0)}
            - Web Search Results Analyzed: {state['analysis'].get('web_results_analyzed', 0)}
            - Key Insights: {state['analysis'].get('key_insights', 'Not available')}
            
            Requirements:
            1. Generate a comprehensive, well-structured markdown report (1000-1500 words)
            2. Incorporate insights from both knowledge base documents and web search
            3. Use proper markdown formatting: headers (# ## ###), bold (**text**), italic (*text*), lists (- item), blockquotes (> text), code blocks (```code```)
            4. Ensure exactly one blank line between sections
            5. Make the report dynamic and flowing, not rigidly structured
            6. Include relevant examples, trends, and insights
            7. Provide actionable insights and future outlook
            
            Report:
            """
            
            # Generate report using LLM
            report_response = self.llm.invoke(report_prompt)
            
            # Create ResearchReport object
            report = ResearchReport(
                content=report_response.content,
                sources=state['analysis'].get('sources_used', [])
            )
            
            state["report"] = report
            state["current_step"] = "Report generated successfully"
            print("Comprehensive report generated successfully")
            
            return state
            
        except Exception as e:
            state["error"] = f"Report generation failed: {str(e)}"
            state["current_step"] = "Report generation failed"
            return state
    
    def plan_research(self, topic: str) -> ResearchPlan:
        """Create a research plan for the given topic"""
        planning_prompt = ChatPromptTemplate.from_template("""
        Create a comprehensive research plan for the topic: {topic}
        
        Focus on:
        1. Main topics to research
        2. Sub-topics for each main topic
        3. Specific research questions
        4. Types of sources to consult
        
        Make it structured and actionable.
        """)
        
        # Use the new RunnableSequence approach instead of deprecated LLMChain
        chain = planning_prompt | self.llm
        result = chain.invoke({"topic": topic})
        
        # Parse the result into ResearchPlan structure
        # For simplicity, we'll create a basic structure
        return ResearchPlan(
            main_topics=[topic],
            sub_topics=[f"Analysis of {topic}"],
            research_questions=[f"What are the key aspects of {topic}?"],
            expected_sources=["Web search", "Knowledge base"]
        )
    
    def gather_information(self, topic: str, plan: ResearchPlan) -> List[Document]:
        """Gather information from multiple sources"""
        documents = []
        
        # 1. Web search for current information
        try:
            web_results = self.search_tool.invoke({
                "query": topic,
                "max_results": 25  # Ensure parameter is properly applied as a dictionary
            })
            for result in web_results:
                if isinstance(result, dict):
                    # Handle dictionary result format
                    doc = Document(
                        page_content=result.get("content", ""),
                        metadata={
                            "source": result.get("url", "Web Search"),
                            "title": result.get("title", "Search Result"),
                            "type": "web_search"
                        }
                    )
                else:
                    # Handle string or other result types
                    doc = Document(
                        page_content=str(result),
                        metadata={
                            "source": "Web Search",
                            "title": "Search Result",
                            "type": "web_search"
                        }
                    )
                documents.append(doc)
        except Exception as e:
            print(f"Web search error: {e}")
        
        # 2. Query knowledge base (vector store)
        if self.vector_store:
            try:
                knowledge_results = self.vector_store.similarity_search(topic, k=5)
                documents.extend(knowledge_results)
            except Exception as e:
                print(f"Knowledge base search error: {e}")
        
        return documents
    
    def analyze_information(self, documents: List[Document], topic: str) -> Dict[str, Any]:
        """Analyze gathered information"""
        analysis_prompt = ChatPromptTemplate.from_template("""
        Analyze the following information about {topic}:
        
        {documents}
        
        Provide:
        1. Key findings
        2. Important insights
        3. Trends or patterns
        4. Areas of uncertainty
        
        Make it clear and well-structured.
        """)
        
        # Combine document content
        combined_content = "\n\n".join([doc.page_content for doc in documents])
        
        # Use the new RunnableSequence approach instead of deprecated LLMChain
        chain = analysis_prompt | self.llm
        analysis_result = chain.invoke({
            "topic": topic,
            "documents": combined_content
        })
        
        return {
            "analysis": analysis_result.content,
            "sources": [doc.metadata.get("source", "Unknown") for doc in documents],
            "document_count": len(documents)
        }
    
    def generate_report(self, topic: str, plan: ResearchPlan, analysis: Dict[str, Any]) -> ResearchReport:
        """Generate final research report as free-flowing markdown"""
        report_prompt = ChatPromptTemplate.from_template("""
        Generate a comprehensive, detailed research report for: {topic}
        
        Research Plan: {plan}
        Analysis: {analysis}
        
        Create a professional, well-structured report in markdown format. 
        The report should flow naturally and include:
        
        - A compelling introduction and overview
        - Detailed analysis with proper markdown formatting
        - Key insights and findings
        - Technical details and implications
        - Future trends and recommendations
        - Conclusion
        
        IMPORTANT FORMATTING REQUIREMENTS:
        - Use ### for section headers (no extra spaces before or after)
        - Use **bold** for emphasis on key terms and concepts
        - Use *italic* for important terms and definitions
        - Use bullet points (- or *) for lists with proper indentation
        - Use numbered lists (1. 2. 3.) where appropriate
        - Use > for important quotes or highlights
        - For code examples, use proper markdown code blocks with ```python and ```
        - Ensure NO extra blank lines between sections - use exactly ONE blank line between sections
        - Keep consistent spacing throughout the document
        - Make lists compact and well-formatted
        
        CONTENT REQUIREMENTS:
        - Make the report comprehensive and detailed (1000-1500 words)
        - Avoid any placeholder text or generic sections
        - Ensure smooth transitions between sections
        - Include specific examples and technical details
        - Make the content actionable and informative
        
        FORMAT EXAMPLE:
        ### Section Title
        Content here with proper formatting.
        
        ### Next Section
        More content with lists:
        - First item
        - Second item
        - Third item
        
        Code example:
        ```python
        def example():
            return "properly formatted"
        ```
        
        Generate a clean, professional report that follows these formatting guidelines exactly.
        """)
        
        # Use the new RunnableSequence approach instead of deprecated LLMChain
        chain = report_prompt | self.llm
        report_result = chain.invoke({
            "topic": topic,
            "plan": str(plan),
            "analysis": str(analysis)
        })
        
        return ResearchReport(
            content=report_result,
            sources=analysis.get("sources", [])
        )
    
    def research(self, topic: str) -> ResearchReport:
        """Main research method using LangGraph workflow"""
        print(f"Starting comprehensive research on: {topic}")
        
        # Initialize state
        initial_state = AgentState(
            topic=topic,
            plan=None,
            documents=[],
            analysis={},
            report=None,
            agent_type=self.__class__.__name__,
            current_step="Initializing...",
            error=""
        )
        
        # Execute the graph with proper checkpoint configuration
        try:
            config = {
                "configurable": {
                    "thread_id": f"research_{hash(topic)}",
                    "checkpoint_ns": "research_agent",
                    "checkpoint_id": f"research_{hash(topic)}_{id(self)}"
                }
            }
            
            final_state = self.graph.invoke(initial_state, config=config)
            
            if final_state.get("error"):
                raise Exception(final_state["error"])
            
            print("Comprehensive research completed!")
            return final_state["report"]
            
        except Exception as e:
            print(f"Research workflow failed: {e}")
            # Fallback to basic report
            return ResearchReport(
                content=f"Research failed: {str(e)}",
                sources=[]
            )

class ITResearchAgent(BaseResearchAgent):
    """Specialized agent for IT research using LangGraph"""
    
    def __init__(self):
        super().__init__()
        self.domain = "IT"
        self.specialized_sources = [
            "github.com", "stackoverflow.com", "arxiv.org", 
            "ieee.org", "techcrunch.com", "wired.com"
        ]
    
    def plan_research(self, topic: str) -> ResearchPlan:
        """IT-specific research planning"""
        it_planning_prompt = ChatPromptTemplate.from_template("""
        Create a comprehensive IT research plan for: {topic}
        
        Focus on:
        1. Technical aspects
        2. Implementation details
        3. Best practices
        4. Security considerations
        5. Performance implications
        6. Industry standards
        
        Make it technically focused and practical.
        """)
        
        # Use the new RunnableSequence approach instead of deprecated LLMChain
        chain = it_planning_prompt | self.llm
        result = chain.invoke({"topic": topic})
        
        return ResearchPlan(
            main_topics=[f"Technical analysis of {topic}"],
            sub_topics=["Implementation", "Security", "Performance", "Standards"],
            research_questions=[f"What are the technical requirements for {topic}?"],
            expected_sources=["Technical documentation", "Code repositories", "Industry standards"]
        )
    
    def analyze_information(self, documents: List[Document], topic: str) -> Dict[str, Any]:
        """IT-specific information analysis"""
        it_analysis_prompt = ChatPromptTemplate.from_template("""
        Analyze the following IT information about {topic}:
        
        {documents}
        
        Focus on:
        1. Technical implementation details
        2. Security considerations
        3. Performance implications
        4. Best practices
        5. Industry standards
        6. Code examples or patterns
        
        Make it technically accurate and actionable.
        """)
        
        combined_content = "\n\n".join([doc.page_content for doc in documents])
        # Use the new RunnableSequence approach instead of deprecated LLMChain
        chain = it_analysis_prompt | self.llm
        analysis_result = chain.invoke({
            "topic": topic,
            "documents": combined_content
        })
        
        return {
            "analysis": analysis_result.content,
            "sources": [doc.metadata.get("source", "Unknown") for doc in documents],
            "document_count": len(documents),
            "domain": "IT"
        }

class PharmaResearchAgent(BaseResearchAgent):
    """Specialized agent for pharmaceutical research using LangGraph"""
    
    def __init__(self):
        super().__init__()
        self.domain = "Pharmaceutical"
        self.specialized_sources = [
            "pubmed.ncbi.nlm.nih.gov", "clinicaltrials.gov", 
            "who.int", "nih.gov", "fda.gov"
        ]
    
    def plan_research(self, topic: str) -> ResearchPlan:
        """Pharma-specific research planning"""
        pharma_planning_prompt = ChatPromptTemplate.from_template("""
        Create a comprehensive pharmaceutical research plan for: {topic}
        
        Focus on:
        1. Medical/clinical aspects
        2. Drug development stages
        3. Clinical trial information
        4. Safety and efficacy
        5. Regulatory considerations
        6. Treatment protocols
        
        Make it medically accurate and research-focused.
        """)
        
        # Use the new RunnableSequence approach instead of deprecated LLMChain
        chain = pharma_planning_prompt | self.llm
        result = chain.invoke({"topic": topic})
        
        return ResearchPlan(
            main_topics=[f"Medical research on {topic}"],
            sub_topics=["Clinical trials", "Safety", "Efficacy", "Regulations"],
            research_questions=[f"What are the medical implications of {topic}?"],
            expected_sources=["Medical journals", "Clinical trials", "Regulatory documents"]
        )
    
    def analyze_information(self, documents: List[Document], topic: str) -> Dict[str, Any]:
        """Pharma-specific information analysis"""
        pharma_analysis_prompt = ChatPromptTemplate.from_template("""
        Analyze the following pharmaceutical information about {topic}:
        
        {documents}
        
        Focus on:
        1. Clinical evidence
        2. Safety data
        3. Efficacy results
        4. Regulatory status
        5. Treatment guidelines
        6. Research methodology
        
        Make it medically accurate and evidence-based.
        """)
        
        combined_content = "\n\n".join([doc.page_content for doc in documents])
        # Use the new RunnableSequence approach instead of deprecated LLMChain
        chain = pharma_analysis_prompt | self.llm
        analysis_result = chain.invoke({
            "topic": topic,
            "documents": combined_content
        })
        
        return {
            "analysis": analysis_result.content,
            "sources": [doc.metadata.get("source", "Unknown") for doc in documents],
            "document_count": len(documents),
            "domain": "Pharmaceutical"
        }

class AgentSelector:
    """Automatically selects the appropriate agent based on topic"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.7,
            openai_api_key=settings.OPENAI_API_KEY
        )
    
    def select_agent(self, topic: str) -> str:
        """Determine which agent to use based on topic"""
        selection_prompt = ChatPromptTemplate.from_template("""
        Analyze this research topic and determine which specialized agent should handle it:
        
        Topic: {topic}
        
        Choose between:
        - IT Research Agent: For software, technology, programming, cybersecurity, cloud computing, etc.
        - Pharma Research Agent: For drugs, medical research, clinical trials, healthcare, treatments, etc.
        
        Respond with only: "IT" or "Pharma"
        """)
        
        # Use the new RunnableSequence approach instead of deprecated LLMChain
        chain = selection_prompt | self.llm
        result = chain.invoke({"topic": topic})
        
        # Clean the result - access content from AIMessage
        result_text = result.content.strip().lower()
        if "it" in result_text or "technology" in result_text:
            return "IT"
        elif "pharma" in result_text or "medical" in result_text:
            return "Pharma"
        else:
            # Default to IT for unclear topics
            return "IT"
