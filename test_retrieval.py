"""
Test retrieving information from the vector database
"""

import os
import sys
import django

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'AI_Research_Agent.settings')
django.setup()

def test_retrieval():
    """Test retrieving information from the vector database"""
    try:
        from agent.document_loader import DocumentProcessor
        
        processor = DocumentProcessor()
        
        # Test queries for IT sector
        it_queries = [
            "What are the major trends in the Indian IT industry?",
            "What is the size of Indian IT market?",
            "What are the challenges faced by Indian IT companies?",
            "How has COVID-19 impacted the IT sector in India?",
            "What are the future growth areas for Indian IT companies?"
        ]
        
        # Test queries for pharma sector
        pharma_queries = [
            "What are the major trends in the Indian pharmaceutical industry?",
            "What is the size of Indian pharmaceutical market?",
            "What are the challenges faced by Indian pharmaceutical companies?",
            "How has COVID-19 impacted the pharmaceutical sector in India?",
            "What are the future growth areas for Indian pharmaceutical companies?"
        ]
        
        print("🔍 Testing IT Sector Queries...")
        for query in it_queries:
            print(f"\nQuery: {query}")
            docs = processor.get_relevant_documents(query, sector="IT", k=3)
            
            if docs:
                print(f"✓ Found {len(docs)} relevant documents")
                for i, doc in enumerate(docs, 1):
                    print(f"\nResult {i}:")
                    print(f"Source: {doc.metadata.get('filename', 'Unknown')}, Page: {doc.metadata.get('page', 'Unknown')}")
                    print(f"Content snippet: {doc.page_content[:200]}...")
            else:
                print("✗ No relevant documents found")
        
        print("\n🔍 Testing Pharma Sector Queries...")
        for query in pharma_queries:
            print(f"\nQuery: {query}")
            docs = processor.get_relevant_documents(query, sector="Pharma", k=3)
            
            if docs:
                print(f"✓ Found {len(docs)} relevant documents")
                for i, doc in enumerate(docs, 1):
                    print(f"\nResult {i}:")
                    print(f"Source: {doc.metadata.get('filename', 'Unknown')}, Page: {doc.metadata.get('page', 'Unknown')}")
                    print(f"Content snippet: {doc.page_content[:200]}...")
            else:
                print("✗ No relevant documents found")
                
        return True
        
    except Exception as e:
        print(f"❌ Retrieval test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run retrieval test"""
    print("🧪 Testing Vector Database Retrieval\n")
    
    success = test_retrieval()
    
    if success:
        print("\n🎉 Vector database retrieval is working correctly!")
        print("\nYou can now generate reports using the research agents.")
    else:
        print("\n❌ Vector database retrieval has issues. Please check the implementation.")

if __name__ == "__main__":
    main()
