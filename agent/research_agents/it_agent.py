"""
IT Research Agent Implementation
Specialized agent for IT and technology research
"""

from typing import List, Dict, Any
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document

# Local imports
from .base_agent import BaseResearchAgent
from .models import ResearchPlan

from .base_agent import BaseResearchAgent
from .models import ResearchPlan

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
        """)
        
        chain = it_planning_prompt | self.llm
        result = chain.invoke({"topic": topic})
        
        return ResearchPlan(
            main_topics=[f"Technical analysis of {topic}"],
            sub_topics=["Implementation", "Security", "Performance", "Standards"],
            research_questions=[f"What are the technical requirements for {topic}?"],
            expected_sources=["Technical documentation", "Code repositories", "Industry standards"]
        )
