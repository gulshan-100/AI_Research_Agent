"""
Test script for document loading and RAG system
"""

import os
import sys
import django

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'AI_Research_Agent.settings')
django.setup()

def test_document_processor():
    """Test the document processor"""
    try:
        print("Testing document processor...")
        
        # Import after Django setup
        from agent.document_loader import DocumentProcessor
        
        processor = DocumentProcessor()
        print("✓ DocumentProcessor created successfully")
        
        # Test loading documents from IT folder
        print("\nTesting IT documents loading...")
        it_docs = processor.load_documents_from_folder("docs/IT_docs", "IT")
        print(f"✓ Loaded {len(it_docs)} IT document chunks")
        
        # Test loading documents from Pharma folder
        print("\nTesting Pharma documents loading...")
        pharma_docs = processor.load_documents_from_folder("docs/pharma_docs", "Pharma")
        print(f"✓ Loaded {len(pharma_docs)} Pharma document chunks")
        
        # Test document metadata
        if it_docs:
            print(f"\nSample IT document metadata:")
            print(f"  Source: {it_docs[0].metadata.get('source', 'N/A')}")
            print(f"  Sector: {it_docs[0].metadata.get('sector', 'N/A')}")
            print(f"  Content preview: {it_docs[0].page_content[:100]}...")
        
        if pharma_docs:
            print(f"\nSample Pharma document metadata:")
            print(f"  Source: {pharma_docs[0].metadata.get('source', 'N/A')}")
            print(f"  Sector: {pharma_docs[0].metadata.get('sector', 'N/A')}")
            print(f"  Content preview: {pharma_docs[0].page_content[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ Document processor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_knowledge_base_setup():
    """Test knowledge base setup (without actually populating)"""
    try:
        print("\nTesting knowledge base setup...")
        
        from agent.document_loader import DocumentProcessor
        
        processor = DocumentProcessor()
        
        # Test Pinecone connection
        from pinecone import Pinecone
        from django.conf import settings
        
        pinecone_client = Pinecone(api_key=settings.PINECONE_API_KEY)
        
        if settings.PINECONE_INDEX_NAME in pinecone_client.list_indexes().names():
            print(f"✓ Pinecone index '{settings.PINECONE_INDEX_NAME}' exists")
        else:
            print(f"⚠ Pinecone index '{settings.PINECONE_INDEX_NAME}' does not exist (will be created)")
        
        print("✓ Knowledge base setup test passed")
        return True
        
    except Exception as e:
        print(f"✗ Knowledge base setup test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Document Loading and RAG System\n")
    
    tests = [
        ("Document Processor", test_document_processor),
        ("Knowledge Base Setup", test_knowledge_base_setup),
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
        print("🎉 All tests passed! Document loading system is working correctly.")
        print("\nNext steps:")
        print("1. Run: python manage.py populate_kb")
        print("2. Test the research agents with RAG enabled")
    else:
        print("❌ Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    main()
