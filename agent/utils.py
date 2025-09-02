"""
Utility functions for the AI Research Agent
"""

from django.conf import settings
from agent.document_loader import setup_knowledge_base

def populate_knowledge_base(force_recreate: bool = False):
    """
    Populate the knowledge base with documents from the docs folder
    
    Args:
        force_recreate (bool): If True, clear existing data before populating
    """
    try:
        print("Starting knowledge base population...")
        setup_knowledge_base(force_recreate=force_recreate)
        print("Knowledge base population completed successfully!")
        return True
    except Exception as e:
        print(f"Error populating knowledge base: {e}")
        return False

def check_knowledge_base_status():
    """
    Check the status of the knowledge base
    
    Returns:
        dict: Status information about the knowledge base
    """
    try:
        from pinecone import Pinecone as PineconeClient
        
        pinecone_client = PineconeClient(api_key=settings.PINECONE_API_KEY)
        
        # Check if index exists
        index_exists = settings.PINECONE_INDEX_NAME in pinecone_client.list_indexes().names()
        
        if index_exists:
            # Get index stats
            index = pinecone_client.Index(settings.PINECONE_INDEX_NAME)
            stats = index.describe_index_stats()
            
            return {
                "status": "active",
                "index_name": settings.PINECONE_INDEX_NAME,
                "total_vector_count": stats.get("total_vector_count", 0),
                "dimension": stats.get("dimension", 0),
                "metric": stats.get("metric", "unknown")
            }
        else:
            return {
                "status": "not_created",
                "index_name": settings.PINECONE_INDEX_NAME,
                "message": "Index does not exist yet"
            }
            
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }
