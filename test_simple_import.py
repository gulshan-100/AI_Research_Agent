"""
Simple test to verify imports work
"""

import os
import sys
import django

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'AI_Research_Agent.settings')
django.setup()

def test_imports():
    """Test if all imports work correctly"""
    try:
        print("Testing imports...")
        
        # Test agent imports
        from agent.agents import ITResearchAgent, PharmaResearchAgent, AgentSelector
        print("✓ Agent imports successful")
        
        # Test document loader imports
        from agent.document_loader import DocumentProcessor
        print("✓ Document loader imports successful")
        
        # Test utility imports
        from agent.utils import populate_knowledge_base, check_knowledge_base_status
        print("✓ Utility imports successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_agent_creation():
    """Test if agents can be created"""
    try:
        print("\nTesting agent creation...")
        
        from agent.agents import ITResearchAgent, PharmaResearchAgent
        
        # Test IT agent creation
        it_agent = ITResearchAgent()
        print("✓ IT Research Agent created successfully")
        
        # Test Pharma agent creation
        pharma_agent = PharmaResearchAgent()
        print("✓ Pharma Research Agent created successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Agent creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🧪 Testing System Imports and Agent Creation\n")
    
    tests = [
        ("Import Test", test_imports),
        ("Agent Creation Test", test_agent_creation),
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
        print("🎉 All tests passed! System is ready for use.")
        print("\nNext steps:")
        print("1. Populate knowledge base: python manage.py populate_kb")
        print("2. Test research functionality")
    else:
        print("❌ Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    main()
