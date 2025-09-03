"""
Base Research Agent Implementation
Contains the core functionality shared by all research agents
"""

from typing import List, Dict, Any, TypedDict
from django.conf import settings

# LangChain imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import Pinecone as LangchainPinecone
from langchain_tavily import TavilySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Local imports
from .models import ResearchPlan, ResearchReport, AgentState

# Initialize Pinecone with new API
from pinecone import Pinecone as PineconeClient

# LangChain imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import Pinecone as LangchainPinecone
from langchain_tavily import TavilySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Pinecone
from pinecone import Pinecone as PineconeClient

from .models import ResearchPlan, ResearchReport, AgentState

class BaseResearchAgent:
    """Base class for all research agents using LangGraph"""
    
    def __init__(self):
        # Initialize LLM and embeddings
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.7,
            openai_api_key=settings.OPENAI_API_KEY
        )
        # Use the same embedding model logic as document_loader
        try:
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=settings.OPENAI_API_KEY,
                model="text-embedding-3-small"
            )
        except Exception as e:
            print(f"Warning: Could not initialize with text-embedding-3-small: {e}")
            print("Trying with text-embedding-ada-002 (1536 dimensions) as fallback...")
            
            # Fallback to the older model that we know works
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=settings.OPENAI_API_KEY,
                model="text-embedding-ada-002"
            )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        
        # Initialize Tavily Search with explicit api_key parameter and advanced search settings
        self.search_tool = TavilySearch(
            tavily_api_key=settings.TAVILY_API_KEY,
            search_depth="advanced",  # Use advanced search for more comprehensive results
            k=25  # Set default k to 25 (max_results can override this)
        )
        
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
            pc = PineconeClient(api_key=settings.PINECONE_API_KEY)
            
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
            
            kb_docs = []
            if self.vector_store:
                try:
                    sector = "IT" if "IT" in state["agent_type"] else "Pharma"
                    kb_docs = self.vector_store.similarity_search(
                        state["topic"], 
                        k=10,
                        filter={"sector": sector}
                    )
                    print(f"Retrieved {len(kb_docs)} relevant documents from knowledge base")
                except Exception as e:
                    print(f"Knowledge base retrieval error: {e}")
            
            web_results = []
            try:
                search_query = f"{state['topic']} {sector if 'sector' in locals() else ''}"
                # Force max_results to 25 with other search parameters
                web_results = self.search_tool.invoke({
                    "query": search_query,
                    "max_results": 25,
                    "search_depth": "advanced",
                    "include_domains": [],
                    "exclude_domains": []
                })
                print(f"Retrieved {len(web_results)} web search results")
                
                # Add a check to debug Tavily results limitation
                if len(web_results) < 25:
                    print(f"Warning: Tavily returned fewer results than requested ({len(web_results)})")
                    print("Making a second search request to supplement results...")
                    # Try a slightly modified query to get additional results
                    alt_query = f"{state['topic']} latest research {sector if 'sector' in locals() else ''}"
                    additional_results = self.search_tool.invoke({
                        "query": alt_query,
                        "max_results": 25 - len(web_results),
                        "search_depth": "advanced"
                    })
                    
                    # Add only unique results based on URL
                    existing_urls = [r.get("url", "") for r in web_results if isinstance(r, dict)]
                    for result in additional_results:
                        if isinstance(result, dict) and result.get("url") not in existing_urls:
                            web_results.append(result)
                            existing_urls.append(result.get("url", ""))
                    
                    print(f"After supplementing: {len(web_results)} total web search results")
            except Exception as e:
                print(f"Web search error: {e}")
            
            all_documents = []
            
            for doc in kb_docs:
                all_documents.append({
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "Knowledge Base"),
                    "type": "knowledge_base",
                    "sector": doc.metadata.get("sector", "Unknown")
                })
            
            for result in web_results:
                if isinstance(result, dict):
                    all_documents.append({
                        "content": result.get("content", ""),
                        "source": result.get("url", "Web Search"),
                        "type": "web_search",
                        "sector": "web"
                    })
                else:
                    all_documents.append({
                        "content": str(result),
                        "source": "Web Search",
                        "type": "web_search",
                        "sector": "web"
                    })
            
            print(f"Total documents gathered: {len(all_documents)}")
            state["documents"] = all_documents
            state["current_step"] = "Information gathered successfully"
            
            return state
            
        except Exception as e:
            state["error"] = f"Error gathering information: {str(e)}"
            state["current_step"] = "Failed to gather information"
            return state

    def _analyze_information_node(self, state: AgentState) -> AgentState:
        """Analyze gathered information"""
        try:
            print("Analyzing information...")
            
            kb_docs = [doc for doc in state["documents"] if doc.get("type") == "knowledge_base"]
            web_docs = [doc for doc in state["documents"] if doc.get("type") == "web_search"]
            
            print(f"Analyzing {len(kb_docs)} knowledge base documents and {len(web_docs)} web search results")
            
            # Create a very concise sources summary for display
            def get_concise_sources_summary():
                """Create a very brief summary of sources used"""
                kb_unique = list(set([clean_source_display(doc['source']) for doc in kb_docs]))
                web_unique = list(set([clean_source_display(doc['source']) for doc in web_docs]))
                
                kb_summary = ", ".join(kb_unique[:5])  # Show max 5 unique KB sources
                web_summary = f"{len(web_docs)} web sources" if web_docs else "No web sources"
                
                return f"KB: {kb_summary} | {web_summary}"
            
            print(f"Sources: {get_concise_sources_summary()}")
            
            # Create a much more concise analysis prompt with minimal source info
            def clean_source_display(source):
                """Clean source for display - show only filename, not full path"""
                if isinstance(source, str):
                    if source.startswith("C:\\") or source.startswith("/"):
                        # Extract just the filename
                        filename = source.split("\\")[-1].split("/")[-1]
                        # Remove .pdf extension for cleaner display
                        return filename.replace(".pdf", "")
                    elif source == "Web Search":
                        return "Web"
                    else:
                        return source[:30] + "..." if len(source) > 30 else source
                return str(source)[:30] + "..." if len(str(source)) > 30 else str(source)
            
            # Show only 2-3 documents with very short content previews
            kb_preview = chr(10).join([f"- {str(doc['content'])[:50]}... ({clean_source_display(doc['source'])})" for doc in kb_docs[:3]])
            web_preview = chr(10).join([f"- {str(doc['content'])[:50]}... ({clean_source_display(doc['source'])})" for doc in web_docs[:2]])
            
            analysis_prompt = f"""
            Analyze the following information about: {state['topic']}
            
            Knowledge Base Documents ({len(kb_docs)}):
            {kb_preview}
            
            Web Search Results ({len(web_docs)}):
            {web_preview}
            """
            
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
        """Generate final research report"""
        try:
            print("Generating comprehensive report...")
            
            report_prompt = f"""
            Generate a comprehensive, detailed research report on: {state['topic']}
            Based on the following analysis:
            {state['analysis'].get('key_insights', 'No analysis available')}
            
            Requirements:
            1. Generate a comprehensive, well-structured markdown report (800-1200 words)
            2. Use proper markdown formatting: headers (# ## ###), bold (**text**), italic (*text*), lists (- item)
            3. Make the report dynamic and flowing, not rigidly structured
            4. Include relevant examples, trends, and insights
            5. Provide actionable insights and future outlook
            6. Include a concise "Sources" section at the end with key references
            
            Report:
            """
            
            report_response = self.llm.invoke(report_prompt)
            
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
        """)
        
        chain = planning_prompt | self.llm
        result = chain.invoke({"topic": topic})
        
        return ResearchPlan(
            main_topics=[topic],
            sub_topics=[f"Analysis of {topic}"],
            research_questions=[f"What are the key aspects of {topic}?"],
            expected_sources=["Web search", "Knowledge base"]
        )

    def research(self, topic: str) -> ResearchReport:
        """Main research method using LangGraph workflow"""
        print(f"Starting comprehensive research on: {topic}")
        
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
            return ResearchReport(
                content=f"Research failed: {str(e)}",
                sources=[]
            )
