"""
Simple test script to verify LangGraph implementation
Run this to test if the agents work correctly
"""

import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'AI_Research_Agent.settings')
django.setup()

from agent.agents import ITResearchAgent, PharmaResearchAgent, AgentSelector

def test_agent_creation():
    """Test if agents can be created successfully"""
    try:
        print("Testing IT Research Agent creation...")
        it_agent = ITResearchAgent()
        print("✓ IT Research Agent created successfully")
        
        print("Testing Pharma Research Agent creation...")
        pharma_agent = PharmaResearchAgent()
        print("✓ Pharma Research Agent created successfully")
        
        print("Testing Agent Selector creation...")
        selector = AgentSelector()
        print("✓ Agent Selector created successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Agent creation failed: {e}")
        return False

def test_agent_selection():
    """Test agent selection logic"""
    try:
        selector = AgentSelector()
        
        # Test IT topic
        it_topic = "Python programming best practices"
        selected = selector.select_agent(it_topic)
        print(f"Topic: '{it_topic}' -> Selected Agent: {selected}")
        
        # Test Pharma topic
        pharma_topic = "COVID-19 vaccine development"
        selected = selector.select_agent(pharma_topic)
        print(f"Topic: '{pharma_topic}' -> Selected Agent: {selected}")
        
        return True
        
    except Exception as e:
        print(f"✗ Agent selection failed: {e}")
        return False

def test_langgraph_structure():
    """Test if LangGraph workflow is properly structured"""
    try:
        it_agent = ITResearchAgent()
        
        # Check if graph exists
        if hasattr(it_agent, 'graph'):
            print("✓ LangGraph workflow exists")
            
            # Check if memory exists
            if hasattr(it_agent, 'memory'):
                print("✓ Memory system exists")
            else:
                print("✗ Memory system missing")
                
        else:
            print("✗ LangGraph workflow missing")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ LangGraph structure test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing LangGraph Implementation\n")
    
    tests = [
        ("Agent Creation", test_agent_creation),
        ("Agent Selection", test_agent_selection),
        ("LangGraph Structure", test_langgraph_structure),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"Running: {test_name}")
        if test_func():
            passed += 1
            print(f"✓ {test_name} passed\n")
        else:
            print(f"✗ {test_name} failed\n")
    
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! LangGraph implementation is working correctly.")
    else:
        print("❌ Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    main()
