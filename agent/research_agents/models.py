"""
Models for Research Agents
Contains Pydantic models used across different agent implementations
"""

from typing import List, Dict, Any, TypedDict
from pydantic import BaseModel, Field
from langchain.schema import Document

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

class AgentState(Dict):
    """State management for LangGraph agents"""
    topic: str
    plan: ResearchPlan
    documents: List[Any]
    analysis: Dict[str, Any]
    report: ResearchReport
    agent_type: str
    current_step: str
    error: str
