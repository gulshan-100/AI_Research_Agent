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
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate

# For web search
from tavily import TavilyClient

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

# Keep imports and clean up class duplication - import directly from research_agents

# Import directly from research_agents (note that these are already imported at the top)
# from .research_agents.base_agent import BaseResearchAgent
# from .research_agents.it_agent import ITResearchAgent
# from .research_agents.pharma_agent import PharmaResearchAgent

# Only keep the AgentSelector
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
