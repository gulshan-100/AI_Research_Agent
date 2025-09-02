"""
Simple script to populate the knowledge base with documents
"""

import os
import sys
import django

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'AI_Research_Agent.settings')
django.setup()

def main():
    """Populate the knowledge base with documents"""
    try:
        print("🚀 Starting Knowledge Base Population...")
        
        # Import the document processor
        from agent.document_loader import DocumentProcessor
        
        # Create processor
        processor = DocumentProcessor()
        print("✓ Document processor created")
        
        # Populate vector store
        processor.populate_vector_store(force_recreate=True)
        
        print("🎉 Knowledge base population completed successfully!")
        print("\nYour documents are now available for RAG!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
