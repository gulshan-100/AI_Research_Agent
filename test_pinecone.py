"""
Simple test script to verify Pinecone setup
"""

import os
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'AI_Research_Agent.settings')
django.setup()

from agent.agents import BaseResearchAgent

def test_pinecone_setup():
    """Test if Pinecone setup works correctly"""
    try:
        print("Testing Pinecone setup...")
        
        # Create a base agent to test Pinecone setup
        agent = BaseResearchAgent()
        
        if agent.vector_store:
            print("✓ Pinecone vector store setup successful")
            return True
        else:
            print("✗ Pinecone vector store setup failed")
            return False
            
    except Exception as e:
        print(f"✗ Pinecone setup error: {e}")
        return False

def test_langgraph_creation():
    """Test if LangGraph workflow can be created"""
    try:
        print("Testing LangGraph workflow creation...")
        
        agent = BaseResearchAgent()
        
        if hasattr(agent, 'graph') and agent.graph:
            print("✓ LangGraph workflow created successfully")
            return True
        else:
            print("✗ LangGraph workflow creation failed")
            return False
            
    except Exception as e:
        print(f"✗ LangGraph creation error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Pinecone and LangGraph Setup\n")
    
    tests = [
        ("Pinecone Setup", test_pinecone_setup),
        ("LangGraph Creation", test_langgraph_creation),
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
        print("🎉 All tests passed! Setup is working correctly.")
    else:
        print("❌ Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    main()
