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
        """Create the LangGraph workflow for the agent with loops for iterative research"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("plan_research", self._plan_research_node)
        workflow.add_node("gather_information", self._gather_information_node)
        workflow.add_node("analyze_information", self._analyze_information_node)
        workflow.add_node("validate_information", self._validate_information_node)
        workflow.add_node("refine_research", self._refine_research_node)
        workflow.add_node("generate_report", self._generate_report_node)
        workflow.add_node("review_report", self._review_report_node)
        workflow.add_node("finalize_report", self._finalize_report_node)
        
        # Add edges with conditional logic
        workflow.add_edge("plan_research", "gather_information")
        workflow.add_edge("gather_information", "analyze_information")
        workflow.add_edge("analyze_information", "validate_information")
        
        # Conditional edge: validate -> refine or continue
        workflow.add_conditional_edges(
            "validate_information",
            self._should_refine_research,
            {
                "refine": "refine_research",
                "continue": "generate_report"
            }
        )
        
        # Loop back to gather more information if refinement needed
        workflow.add_edge("refine_research", "gather_information")
        
        workflow.add_edge("generate_report", "review_report")
        
        # Conditional edge: review -> regenerate or finalize
        workflow.add_conditional_edges(
            "review_report",
            self._should_regenerate_report,
            {
                "regenerate": "generate_report",
                "finalize": "finalize_report"
            }
        )
        
        workflow.add_edge("finalize_report", END)
        
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
            iteration_count = state.get("iteration_count", 0)
            is_refinement = iteration_count > 0
            
            if is_refinement:
                print(f"Gathering additional information (Refinement {iteration_count})...")
            else:
                print("Gathering comprehensive information...")
            
            # Determine search queries
            base_query = state["topic"]
            search_queries = [base_query]
            
            if is_refinement and state.get("refinement_queries"):
                # Add refinement queries for targeted search
                search_queries.extend(state["refinement_queries"])
                print(f"Using {len(search_queries)} search queries including refinements")
            
            # Gather from knowledge base
            kb_docs = []
            if self.vector_store:
                try:
                    sector = "IT" if "IT" in state["agent_type"] else "Pharma"
                    
                    # Search with multiple queries if refining
                    for query in search_queries[:2]:  # Limit to prevent too many KB calls
                        docs = self.vector_store.similarity_search(
                            query, 
                            k=8 if is_refinement else 10,
                            filter={"sector": sector}
                        )
                        
                        # Add unique documents only
                        existing_content = [doc["content"] for doc in kb_docs]
                        for doc in docs:
                            if doc.page_content not in existing_content:
                                kb_docs.append(doc)
                    
                    print(f"Retrieved {len(kb_docs)} relevant documents from knowledge base")
                except Exception as e:
                    print(f"Knowledge base retrieval error: {e}")
            
            # Gather from web search
            web_results = []
            try:
                sector = "IT" if "IT" in state["agent_type"] else "Pharma"
                
                for i, query in enumerate(search_queries):
                    if i >= 3:  # Limit web searches to prevent rate limiting
                        break
                        
                    search_query = f"{query} {sector}"
                    results_per_query = 15 if i == 0 else 8  # More results for main query
                    
                    try:
                        results = self.search_tool.invoke({
                            "query": search_query,
                            "max_results": results_per_query,
                            "search_depth": "advanced",
                            "include_domains": [],
                            "exclude_domains": []
                        })
                        
                        # Add unique results only
                        existing_urls = [r.get("url", "") for r in web_results if isinstance(r, dict)]
                        for result in results:
                            if isinstance(result, dict) and result.get("url") not in existing_urls:
                                web_results.append(result)
                                existing_urls.append(result.get("url", ""))
                        
                        print(f"Query {i+1}: Retrieved {len(results)} results")
                        
                    except Exception as e:
                        print(f"Web search error for query {i+1}: {e}")
                
                print(f"Total unique web search results: {len(web_results)}")
                
            except Exception as e:
                print(f"Web search error: {e}")
            
            # Combine with existing documents if this is a refinement
            all_documents = []
            if is_refinement and state.get("documents"):
                all_documents = state["documents"].copy()
                print(f"Starting with {len(all_documents)} existing documents")
            
            # Add new KB documents
            existing_content = [doc.get("content", "") for doc in all_documents]
            for doc in kb_docs:
                if doc.page_content not in existing_content:
                    all_documents.append({
                        "content": doc.page_content,
                        "source": doc.metadata.get("source", "Knowledge Base"),
                        "type": "knowledge_base",
                        "sector": doc.metadata.get("sector", "Unknown")
                    })
                    existing_content.append(doc.page_content)
            
            # Add new web results
            for result in web_results:
                content = ""
                if isinstance(result, dict):
                    content = result.get("content", "")
                    if content and content not in existing_content:
                        all_documents.append({
                            "content": content,
                            "source": result.get("url", "Web Search"),
                            "type": "web_search",
                            "sector": "web"
                        })
                        existing_content.append(content)
                else:
                    content = str(result)
                    if content and content not in existing_content:
                        all_documents.append({
                            "content": content,
                            "source": "Web Search",
                            "type": "web_search",
                            "sector": "web"
                        })
                        existing_content.append(content)
            
            print(f"Total documents gathered: {len(all_documents)}")
            state["documents"] = all_documents
            
            if is_refinement:
                state["current_step"] = f"Additional information gathered (Refinement {iteration_count})"
            else:
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

    def _validate_information_node(self, state: AgentState) -> AgentState:
        """Validate the quality and completeness of gathered information"""
        try:
            print("Validating information quality...")
            
            # Check information quality metrics
            total_docs = len(state["documents"])
            kb_docs = len([doc for doc in state["documents"] if doc.get("type") == "knowledge_base"])
            web_docs = len([doc for doc in state["documents"] if doc.get("type") == "web_search"])
            
            # Quality validation prompt
            validation_prompt = f"""
            Evaluate the quality and completeness of research information for topic: {state['topic']}
            
            Information gathered:
            - Total documents: {total_docs}
            - Knowledge base documents: {kb_docs}
            - Web search results: {web_docs}
            
            Analysis insights: {state['analysis'].get('key_insights', 'No analysis available')[:500]}...
            
            Rate the information quality on these criteria (1-10 scale):
            1. Completeness: Are all important aspects covered?
            2. Relevance: How relevant is the information to the topic?
            3. Currency: How recent and up-to-date is the information?
            4. Diversity: Are multiple perspectives represented?
            5. Depth: Is there sufficient detail for comprehensive analysis?
            
            Provide scores and brief justification. If average score < 7, recommend refinement.
            
            Format your response as:
            SCORES: Completeness=X, Relevance=X, Currency=X, Diversity=X, Depth=X
            AVERAGE: X.X
            RECOMMENDATION: CONTINUE/REFINE
            REASONING: [brief explanation]
            """
            
            validation_response = self.llm.invoke(validation_prompt)
            validation_content = validation_response.content
            
            # Parse the validation response
            try:
                average_score = float([line for line in validation_content.split('\n') if 'AVERAGE:' in line][0].split('AVERAGE:')[1].strip())
                recommendation = [line for line in validation_content.split('\n') if 'RECOMMENDATION:' in line][0].split('RECOMMENDATION:')[1].strip()
            except:
                # Fallback if parsing fails
                average_score = 6.0
                recommendation = "CONTINUE"
            
            state["validation"] = {
                "quality_score": average_score,
                "recommendation": recommendation,
                "details": validation_content,
                "iteration_count": state.get("iteration_count", 0)
            }
            
            state["current_step"] = f"Information validated (Score: {average_score}/10)"
            print(f"Validation completed. Quality score: {average_score}/10")
            
            return state
            
        except Exception as e:
            state["error"] = f"Validation failed: {str(e)}"
            state["current_step"] = "Validation failed"
            # Default to continue if validation fails
            state["validation"] = {
                "quality_score": 7.0,
                "recommendation": "CONTINUE",
                "details": f"Validation error: {str(e)}",
                "iteration_count": state.get("iteration_count", 0)
            }
            return state

    def _refine_research_node(self, state: AgentState) -> AgentState:
        """Refine research based on validation feedback"""
        try:
            iteration_count = state.get("iteration_count", 0) + 1
            state["iteration_count"] = iteration_count
            
            print(f"Refining research (Iteration {iteration_count})...")
            
            if iteration_count >= 3:  # Prevent infinite loops
                print("Maximum refinement iterations reached. Proceeding with current information.")
                state["current_step"] = "Maximum refinements reached - proceeding"
                return state
            
            # Analyze what needs improvement based on validation
            refinement_prompt = f"""
            Based on the validation feedback, identify specific gaps in research for: {state['topic']}
            
            Current validation details:
            {state['validation'].get('details', 'No validation details')}
            
            Current documents count: {len(state["documents"])}
            
            Suggest 3-5 specific search queries or focus areas to improve research quality:
            1. [Specific query/area]
            2. [Specific query/area]
            3. [Specific query/area]
            
            Focus on areas that scored lowest in validation.
            """
            
            refinement_response = self.llm.invoke(refinement_prompt)
            
            # Extract search queries from response
            refinement_queries = []
            for line in refinement_response.content.split('\n'):
                if line.strip() and (line.strip().startswith('1.') or line.strip().startswith('2.') or 
                                   line.strip().startswith('3.') or line.strip().startswith('4.') or 
                                   line.strip().startswith('5.')):
                    query = line.split('.', 1)[1].strip() if '.' in line else line.strip()
                    refinement_queries.append(query)
            
            state["refinement_queries"] = refinement_queries[:3]  # Limit to 3 queries
            state["current_step"] = f"Research refinement planned (Iteration {iteration_count})"
            
            print(f"Refinement queries identified: {len(refinement_queries)}")
            return state
            
        except Exception as e:
            state["error"] = f"Research refinement failed: {str(e)}"
            state["current_step"] = "Refinement failed"
            return state

    def _review_report_node(self, state: AgentState) -> AgentState:
        """Review the generated report for quality and completeness"""
        try:
            print("Reviewing generated report...")
            
            if not state.get("report") or not state["report"].content:
                state["review"] = {
                    "quality_score": 3.0,
                    "recommendation": "REGENERATE",
                    "feedback": "No report content generated"
                }
                return state
            
            report_content = state["report"].content
            word_count = len(report_content.split())
            
            review_prompt = f"""
            Review this research report for quality and completeness:
            
            Topic: {state['topic']}
            Report length: {word_count} words
            
            Report content (first 800 chars):
            {report_content[:800]}...
            
            Evaluate the report on these criteria (1-10 scale):
            1. Structure: Is the report well-organized with clear sections?
            2. Content Quality: Is the information accurate and relevant?
            3. Completeness: Does it cover all important aspects of the topic?
            4. Clarity: Is it well-written and easy to understand?
            5. Length: Is it appropriately detailed (target: 800-1200 words)?
            
            Provide scores and brief feedback. If average score < 7, recommend regeneration.
            
            Format your response as:
            SCORES: Structure=X, Content=X, Completeness=X, Clarity=X, Length=X
            AVERAGE: X.X
            RECOMMENDATION: FINALIZE/REGENERATE
            FEEDBACK: [specific improvement suggestions]
            """
            
            review_response = self.llm.invoke(review_prompt)
            review_content = review_response.content
            
            # Parse the review response
            try:
                average_score = float([line for line in review_content.split('\n') if 'AVERAGE:' in line][0].split('AVERAGE:')[1].strip())
                recommendation = [line for line in review_content.split('\n') if 'RECOMMENDATION:' in line][0].split('RECOMMENDATION:')[1].strip()
            except:
                # Fallback if parsing fails
                average_score = 7.0
                recommendation = "FINALIZE"
            
            state["review"] = {
                "quality_score": average_score,
                "recommendation": recommendation,
                "feedback": review_content,
                "regeneration_count": state.get("regeneration_count", 0)
            }
            
            state["current_step"] = f"Report reviewed (Score: {average_score}/10)"
            print(f"Report review completed. Quality score: {average_score}/10")
            
            return state
            
        except Exception as e:
            state["error"] = f"Report review failed: {str(e)}"
            state["current_step"] = "Review failed"
            # Default to finalize if review fails
            state["review"] = {
                "quality_score": 7.0,
                "recommendation": "FINALIZE",
                "feedback": f"Review error: {str(e)}",
                "regeneration_count": state.get("regeneration_count", 0)
            }
            return state

    def _finalize_report_node(self, state: AgentState) -> AgentState:
        """Finalize the report with metadata and summary"""
        try:
            print("Finalizing report...")
            
            if state.get("report"):
                # Add metadata to the report
                final_content = state["report"].content
                
                # Add research metadata
                metadata_section = f"""

---

## Research Metadata

**Research Iterations:** {state.get('iteration_count', 0)}
**Information Quality Score:** {state.get('validation', {}).get('quality_score', 'N/A')}/10
**Report Quality Score:** {state.get('review', {}).get('quality_score', 'N/A')}/10
**Documents Analyzed:** {len(state.get('documents', []))}
**Knowledge Base Sources:** {len([d for d in state.get('documents', []) if d.get('type') == 'knowledge_base'])}
**Web Sources:** {len([d for d in state.get('documents', []) if d.get('type') == 'web_search'])}

*Generated by AI Research Agent with iterative quality validation*
"""
                
                final_content += metadata_section
                
                # Update the report with final content
                state["report"] = ResearchReport(
                    content=final_content,
                    sources=state["report"].sources
                )
            
            state["current_step"] = "Report finalized successfully"
            print("Report finalized with metadata")
            
            return state
            
        except Exception as e:
            state["error"] = f"Report finalization failed: {str(e)}"
            state["current_step"] = "Finalization failed"
            return state

    def _should_refine_research(self, state: AgentState) -> str:
        """Conditional logic to determine if research should be refined"""
        validation = state.get("validation", {})
        quality_score = validation.get("quality_score", 10.0)
        recommendation = validation.get("recommendation", "CONTINUE")
        iteration_count = state.get("iteration_count", 0)
        
        # Refine if quality is low and we haven't exceeded max iterations
        if quality_score < 7.0 and recommendation == "REFINE" and iteration_count < 3:
            return "refine"
        else:
            return "continue"

    def _should_regenerate_report(self, state: AgentState) -> str:
        """Conditional logic to determine if report should be regenerated"""
        review = state.get("review", {})
        quality_score = review.get("quality_score", 10.0)
        recommendation = review.get("recommendation", "FINALIZE")
        regeneration_count = state.get("regeneration_count", 0)
        
        # Regenerate if quality is low and we haven't exceeded max regenerations
        if quality_score < 7.0 and recommendation == "REGENERATE" and regeneration_count < 2:
            state["regeneration_count"] = regeneration_count + 1
            return "regenerate"
        else:
            return "finalize"

    def _generate_report_node(self, state: AgentState) -> AgentState:
        """Generate final research report"""
        try:
            regeneration_count = state.get("regeneration_count", 0)
            is_regeneration = regeneration_count > 0
            
            if is_regeneration:
                print(f"Regenerating report (Attempt {regeneration_count + 1})...")
            else:
                print("Generating comprehensive report...")
            
            # Enhanced report prompt with feedback incorporation
            report_prompt = f"""
            Generate a comprehensive, detailed research report on: {state['topic']}
            Based on the following analysis:
            {state['analysis'].get('key_insights', 'No analysis available')}
            
            Documents analyzed: {len(state.get('documents', []))}
            Knowledge base sources: {len([d for d in state.get('documents', []) if d.get('type') == 'knowledge_base'])}
            Web sources: {len([d for d in state.get('documents', []) if d.get('type') == 'web_search'])}
            """
            
            # Add feedback from previous review if regenerating
            if is_regeneration and state.get("review", {}).get("feedback"):
                report_prompt += f"""
                
                IMPORTANT - Address these specific feedback points from the previous report review:
                {state['review']['feedback']}
                
                Focus on improving the areas that scored lowest in the previous review.
                """
            
            report_prompt += """
            
            Requirements:
            1. Generate a comprehensive, well-structured markdown report (800-1200 words)
            2. Use proper markdown formatting: headers (# ## ###), bold (**text**), italic (*text*), lists (- item)
            3. Make the report dynamic and flowing, not rigidly structured
            4. Include relevant examples, trends, and insights
            5. Provide actionable insights and future outlook
            6. Include a concise "Sources" section at the end with key references
            7. Ensure all important aspects of the topic are covered
            8. Write clearly and professionally
            
            Report:
            """
            
            report_response = self.llm.invoke(report_prompt)
            
            report = ResearchReport(
                content=report_response.content,
                sources=state['analysis'].get('sources_used', [])
            )
            
            state["report"] = report
            
            if is_regeneration:
                state["current_step"] = f"Report regenerated (Attempt {regeneration_count + 1})"
                print(f"Report regenerated successfully (Attempt {regeneration_count + 1})")
            else:
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
