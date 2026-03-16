"""
Test script for the enhanced research agent with looping capabilities
"""

import os
import sys
import time
import django

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'AI_Research_Agent.settings')
django.setup()

def test_research_loops():
    """Test the research agent's looping capabilities"""
    try:
        from agent.research_agents.it_agent import ITResearchAgent
        from agent.research_agents.pharma_agent import PharmaResearchAgent
        from agent.research_agents.selector import AgentSelector
        
        print("🔄 Testing Research Agent Looping Capabilities")
        
        # Use a simple topic that should trigger research loops
        topic = "Recent advancements in quantum computing and their impact on AI"
        
        # Use agent selector to determine the right agent
        selector = AgentSelector()
        selected_agent_type = selector.select_agent(topic)
        print(f"📊 Selected agent type: {selected_agent_type}")
        
        # Create the appropriate agent
        if selected_agent_type == "IT":
            agent = ITResearchAgent()
        else:
            agent = PharmaResearchAgent()
        
        print(f"\n🔍 Starting iterative research on: {topic}")
        print("This may take several minutes as multiple research iterations will occur...")
        
        # Track time to measure total research duration
        start_time = time.time()
        
        # Run the research
        report = agent.research(topic)
        
        # Calculate duration
        duration = time.time() - start_time
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        
        print(f"\n✅ Research completed in {minutes}m {seconds}s")
        
        # Print report summary
        print("\n📑 Research Report Summary:")
        print(f"Title: {topic}")
        print(f"Content length: {len(report.content)} characters")
        print(f"Number of sources: {len(report.sources)}")
        
        # Save the report
        report_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research_report.md")
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(f"# Research Report: {topic}\n\n")
            f.write(report.content)
            f.write("\n\n## Sources\n")
            for source in report.sources:
                f.write(f"- {source}\n")
                
        print(f"\n📄 Full report saved to: {report_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Research loop test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_research_loops()
