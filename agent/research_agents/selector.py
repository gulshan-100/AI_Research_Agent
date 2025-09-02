"""
Agent Selector Implementation
Handles automatic selection of appropriate research agent based on topic
"""

from django.conf import settings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Local imports
from .base_agent import BaseResearchAgent
from .models import ResearchPlan

class AgentSelector:
    """Automatically selects the appropriate agent based on topic"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4",
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
        
        chain = selection_prompt | self.llm
        result = chain.invoke({"topic": topic})
        
        result_text = result.content.strip().lower()
        if "it" in result_text or "technology" in result_text:
            return "IT"
        elif "pharma" in result_text or "medical" in result_text:
            return "Pharma"
        else:
            return "IT"  # Default to IT for unclear topics
