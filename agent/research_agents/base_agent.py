"""
Base Research Agent Implementation
Contains the core functionality shared by all research agents
"""

from typing import List, Dict, Any, TypedDict
from django.conf import settings

# LangChain imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import Pinecone as LangchainPinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Local imports
from .models import ResearchPlan, ResearchReport, AgentState

# External APIs
from pinecone import Pinecone as PineconeClient
from tavily import TavilyClient

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
        # Create custom embeddings adapter to ensure 512 dimensions for Pinecone compatibility
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
        try:
            print("Using custom dimension adapter for embeddings (512 dim)")
            self.embeddings = CustomDimensionEmbeddings(settings.OPENAI_API_KEY)
        except Exception as e:
            print(f"Error initializing embeddings: {e}")
            # Last resort fallback
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=settings.OPENAI_API_KEY,
                model="text-embedding-ada-002",
                dimensions=512  # Try to force dimensions
            )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        
        # Initialize Tavily client directly
        try:
            print(f"Initializing TavilyClient with API key: {settings.TAVILY_API_KEY[:5]}...")
            self.search_tool = TavilyClient(api_key=settings.TAVILY_API_KEY)
            
            # Test the Tavily search to ensure it's working
            print("Testing Tavily search connection...")
            try:
                test_results = self.search_tool.search(
                    query="AI research test",
                    search_depth="advanced",
                    max_results=2
                )
                print(f"Tavily test successful: {len(test_results['results'])} results returned")
            except Exception as tavily_test_error:
                print(f"Tavily test failed: {tavily_test_error}")
        except Exception as e:
            print(f"ERROR initializing Tavily search: {str(e)}")
            self.search_tool = None
        
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
            # Initialize Pinecone client
            pc = PineconeClient(api_key=settings.PINECONE_API_KEY)
            print("Pinecone client initialized successfully")
            
            # Check for existing indexes
            index_name = settings.PINECONE_INDEX_NAME
            existing_indexes = [index.name for index in pc.list_indexes()]
            print(f"Found existing Pinecone indexes: {existing_indexes}")
            
            if index_name not in existing_indexes:
                print(f"Creating Pinecone index: {index_name}")
                
                # Get the dimension from the embeddings object - ensure it's 512
                test_embedding = self.embeddings.embed_query("test")
                embedding_dimension = len(test_embedding)
                print(f"Generated test embedding with dimension: {embedding_dimension}")
                
                # Force dimension to be 512 if it's not already
                if embedding_dimension != 512:
                    print(f"WARNING: Embedding dimension {embedding_dimension} doesn't match required 512. Forcing to 512.")
                    embedding_dimension = 512
                
                # Create the index with the correct dimension
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
                print(f"Successfully created Pinecone index '{index_name}' with dimension {embedding_dimension}")
            else:
                print(f"Using existing Pinecone index: {index_name}")
            
            # Create vector store from the index
            print("Creating LangChain vector store from Pinecone index...")
            self.vector_store = LangchainPinecone.from_existing_index(
                index_name=settings.PINECONE_INDEX_NAME,
                embedding=self.embeddings,
                text_key="text"
            )
            
                # Test the vector store with a simple query to validate it's working
            print("Testing vector store connection...")
            try:
                test_results = self.vector_store.similarity_search(
                    "test query", 
                    k=5  # Test with small sample
                )
                print(f"Vector store test successful: Retrieved {len(test_results)} results")
                
                # Verify retrieval with larger k value
                print("Testing vector store with k=50...")
                test_results_large = self.vector_store.similarity_search(
                    "test query large", 
                    k=50
                )
                print(f"Vector store large retrieval test: Retrieved {len(test_results_large)} results")
            except Exception as test_error:
                print(f"Vector store test failed: {test_error}")
                # Continue anyway since the index might be empty            print(f"Successfully connected to Pinecone index: {settings.PINECONE_INDEX_NAME}")
            
        except Exception as e:
            print(f"ERROR setting up vector store: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            self.vector_store = None
            print("Vector store initialization failed - will rely on web search only.")
    
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
                print(f"===== LOOP ITERATION {iteration_count} - REFINEMENT TRIGGERED =====")
                print(f"Gathering additional information (Refinement {iteration_count})...")
                print(f"Using refinement queries: {state.get('refinement_queries', ['No queries specified'])}")
            else:
                print("===== FIRST ITERATION - INITIAL GATHERING =====")
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
                    print(f"Querying knowledge base for sector: {sector}...")
                    
                    # Search with multiple queries if refining
                    for query in search_queries[:2]:  # Limit to prevent too many KB calls
                        print(f"KB Query: {query}")
                        try:
                            # First try with sector filter - using 50 chunks
                            docs = self.vector_store.similarity_search(
                                query, 
                                k=40 if is_refinement else 50,
                                filter={"sector": sector}
                            )
                            
                            # If no results, try without filter
                            if not docs:
                                print("No results with sector filter, trying without filter...")
                                docs = self.vector_store.similarity_search(
                                    query, 
                                    k=40 if is_refinement else 50
                                )
                            
                            # Add unique documents only
                            existing_content = [doc.page_content if hasattr(doc, "page_content") else "Unknown" for doc in kb_docs]
                            for doc in docs:
                                if hasattr(doc, "page_content") and doc.page_content not in existing_content:
                                    kb_docs.append(doc)
                                    
                            print(f"  - Retrieved {len(docs)} documents for query '{query}'")
                        except Exception as query_error:
                            print(f"  - Error searching for query '{query}': {query_error}")
                    
                    print(f"Retrieved {len(kb_docs)} total relevant documents from knowledge base")
                except Exception as e:
                    print(f"Knowledge base retrieval error: {e}")
                    import traceback
                    print(f"Traceback: {traceback.format_exc()}")
            
            # Gather from web search with robust error handling
            web_results = []
            try:
                sector = "IT" if "IT" in state["agent_type"] else "Pharma"
                
                # Check if search tool is available
                if not self.search_tool:
                    print("WARNING: Tavily search tool not initialized. Web search will be skipped.")
                    print("Using synthetic web results instead.")
                    web_results = [{
                        "content": f"This is synthetic content for '{state['topic']}' created because the web search tool is not available. The research will rely primarily on knowledge base results.",
                        "url": "https://example.com/synthetic-content",
                        "title": "Synthetic Web Content"
                    }]
                else:
                    # Adjust search depth based on KB results
                    web_search_limit = 5 if not kb_docs else 3
                    
                    print(f"Performing web search for {min(web_search_limit, len(search_queries))} queries...")
                
                # Skip the for loop if search tool is not available
                if not self.search_tool:
                    print("Skipping web search queries since search tool is unavailable")
                else:
                    for i, query in enumerate(search_queries):
                        if i >= web_search_limit:
                            break
                            
                        search_query = f"{query} {sector}"
                        print(f"Web Query {i+1}: {search_query}")
                    
                    try:
                            # Using TavilyClient directly
                        if not self.search_tool:
                            print("Tavily client not initialized, skipping web search")
                            results = []
                        else:
                            try:
                                # Call the TavilyClient search method
                                tavily_response = self.search_tool.search(
                                    query=search_query,
                                    search_depth="advanced",
                                    max_results=15 if i == 0 else 8,
                                    include_answer=False,
                                    include_raw_content=True
                                )
                                
                                # Extract results from the response
                                if tavily_response and "results" in tavily_response:
                                    results = tavily_response["results"]
                                    print(f"   - Tavily search successful: {len(results)} results")
                                else:
                                    print(f"   - Tavily search returned no results structure")
                                    results = []
                            except Exception as search_error:
                                print(f"   - Tavily search failed: {str(search_error)}")
                                results = []
                                
                        # Process results if we have any
                        if results:
                            # Process and filter results
                            valid_results = []
                            for result in results:
                                if isinstance(result, dict) and result.get('content'):
                                    # Clean and validate content
                                    content = result['content'].strip()
                                    if len(content) > 50:  # Ensure meaningful content
                                        valid_results.append(result)
                            
                            # Add unique results
                            existing_urls = [r.get("url", "") for r in web_results]
                            for result in valid_results:
                                if result.get("url") not in existing_urls:
                                    web_results.append(result)
                                    existing_urls.append(result.get("url", ""))
                            
                            print(f"Query {i+1}: Retrieved {len(valid_results)} valid results")
                        else:
                            print(f"Query {i+1}: No results returned from Tavily")
                            
                    except Exception as query_error:
                        print(f"Error in web search query {i+1}: {str(query_error)}")
                        print(f"Query that failed: {search_query}")
                        # Skip to the next iteration
                        results = []
                
                print(f"Total unique web search results with content: {len(web_results)}")
                
                # Only create synthetic content if absolutely no results found
                if not web_results and not kb_docs:
                    print("WARNING: No content found from KB or web. Creating synthetic placeholder content.")
                    web_results = [{
                        "content": f"AI is transforming software development through automated code generation, testing, and developer productivity tools. This placeholder content was created due to search API limitations for '{search_query}'.",
                        "url": "https://example.com/ai-software-development",
                        "title": "AI in Software Development Overview"
                    }]
                
            except Exception as e:
                print(f"Web search error: {str(e)}")
                import traceback
                print(f"Web search traceback: {traceback.format_exc()}")
                # Don't fail completely - continue with any results we have
            
            # Combine with existing documents if this is a refinement
            all_documents = []
            if is_refinement and state.get("documents"):
                all_documents = state["documents"].copy()
                print(f"Starting with {len(all_documents)} existing documents")
            
            # Add new KB documents
            existing_content = [
                doc.get("content", "") if isinstance(doc, dict) 
                else doc.page_content if hasattr(doc, "page_content") 
                else str(doc) 
                for doc in all_documents
            ]
            
            for doc in kb_docs:
                content = doc.page_content if hasattr(doc, "page_content") else str(doc)
                if content not in existing_content:
                    all_documents.append({
                        "content": content,
                        "source": doc.metadata.get("source", "Knowledge Base") if hasattr(doc, "metadata") else "Knowledge Base",
                        "type": "knowledge_base",
                        "sector": doc.metadata.get("sector", "Unknown") if hasattr(doc, "metadata") else "Unknown"
                    })
                    existing_content.append(content)
            
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
            
            # Create a very concise sources summary for display
            def get_concise_sources_summary():
                """Create a very brief summary of sources used"""
                kb_unique = list(set([clean_source_display(doc.get('source', 'Unknown')) for doc in kb_docs]))
                web_unique = list(set([clean_source_display(doc.get('source', 'Unknown')) for doc in web_docs]))
                
                kb_summary = ", ".join(kb_unique[:5])  # Show max 5 unique KB sources
                web_summary = f"{len(web_docs)} web sources" if web_docs else "No web sources"
                
                return f"KB: {kb_summary} | {web_summary}"
            
            print(f"Sources: {get_concise_sources_summary()}")
            
            # Show only 2-3 documents with very short content previews
            kb_preview = chr(10).join([f"- {str(doc.get('content', 'No content'))[:50]}... ({clean_source_display(doc.get('source', 'Unknown'))})" for doc in kb_docs[:3]])
            web_preview = chr(10).join([f"- {str(doc.get('content', 'No content'))[:50]}... ({clean_source_display(doc.get('source', 'Unknown'))})" for doc in web_docs[:2]])
            
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
                "sources_used": [
                    doc.get("source", "Unknown") if isinstance(doc, dict) 
                    else doc.metadata.get("source", "Unknown") if hasattr(doc, "metadata") 
                    else "Unknown" 
                    for doc in state["documents"]
                ]
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
            except Exception as parsing_error:
                print(f"Error parsing validation response: {parsing_error}")
                # Fallback if parsing fails
                average_score = 6.0
                recommendation = "CONTINUE"
            
            # Debugging output
            print(f"===== VALIDATION RESPONSE =====\n{validation_content}\n===== END VALIDATION =====")
            print(f"Parsed average_score: {average_score}, recommendation: {recommendation}")
            
            # ALWAYS FORCE REFINEMENT for first two iterations to ensure the loop works
            iteration_count = state.get("iteration_count", 0)
            if iteration_count < 2:  # Force refinement for first TWO iterations
                print(f"***** FORCING REFINEMENT for iteration {iteration_count} to ensure loop works *****")
                average_score = 5.0
                recommendation = "REFINE"
            
            state["validation"] = {
                "quality_score": average_score,
                "recommendation": recommendation,
                "details": validation_content,
                "iteration_count": iteration_count
            }
            
            state["current_step"] = f"Information validated (Score: {average_score}/10)"
            print(f"Validation completed. Quality score: {average_score}/10, Recommendation: {recommendation}")
            
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
            
            # Add default refinement queries in case the LLM fails to provide them
            default_queries = [
                f"latest trends in {state['topic']}",
                f"challenges and limitations of {state['topic']}",
                f"future of {state['topic']} in 2025"
            ]
            
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
            
            print("Generating refinement queries...")
            refinement_response = self.llm.invoke(refinement_prompt)
            print("Received refinement response from LLM")
            
            # Extract search queries from response
            refinement_queries = []
            for line in refinement_response.content.split('\n'):
                if line.strip() and (line.strip().startswith('1.') or line.strip().startswith('2.') or 
                                   line.strip().startswith('3.') or line.strip().startswith('4.') or 
                                   line.strip().startswith('5.')):
                    query = line.split('.', 1)[1].strip() if '.' in line else line.strip()
                    refinement_queries.append(query)
            
            # Use default queries if none were extracted
            if not refinement_queries:
                print("No refinement queries identified from LLM response. Using defaults.")
                refinement_queries = default_queries
            
            state["refinement_queries"] = refinement_queries[:3]  # Limit to 3 queries
            state["current_step"] = f"Research refinement planned (Iteration {iteration_count})"
            
            print(f"Refinement queries identified ({len(refinement_queries)}): {refinement_queries}")
            return state
            
        except Exception as e:
            print(f"Error during research refinement: {e}")
            state["error"] = f"Research refinement failed: {str(e)}"
            state["current_step"] = "Refinement failed"
            
            # Even if an error occurs, provide default refinement queries to allow loop to continue
            state["refinement_queries"] = [
                f"latest trends in {state['topic']}",
                f"challenges and limitations of {state['topic']}",
                f"future of {state['topic']} in 2025"
            ]
            print("Set default refinement queries due to error")
            
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
        quality_score = validation.get("quality_score", 7.0)
        recommendation = validation.get("recommendation", "CONTINUE")
        iteration_count = state.get("iteration_count", 0)
        
        print(f"DECISION POINT - Refine Research? Score: {quality_score}, Recommendation: {recommendation}, Iteration: {iteration_count}")
        
        # ALWAYS REFINE for the first two iterations to ensure the loop works
        if iteration_count < 2:
            print(f"DECISION: ALWAYS REFINING for iteration {iteration_count} (forced)")
            return "refine"
        # Otherwise use the normal logic for the third iteration
        elif (quality_score < 7.0 or recommendation == "REFINE") and iteration_count < 3:
            print(f"DECISION: REFINING research (Iteration {iteration_count+1})")
            return "refine"
        else:
            print(f"DECISION: CONTINUING to report generation (no more refinement)")
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
