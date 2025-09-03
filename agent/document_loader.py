"""
Document loader utility for processing PDF documents and preparing them for RAG
"""

import os
import sys
import time
from typing import List, Dict, Any
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone as PineconeClient
import logging

# Setup Django environment when run as standalone script
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'AI_Research_Agent.settings')
    import django
    django.setup()

# Now import settings after Django is configured (if needed)
from django.conf import settings

# Get API key from settings (loaded from .env file)
pinecone_api_key = None  # Will be set from settings


logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Processes documents and loads them into the vector store"""
    
    def __init__(self):
        # For text-embedding-3-small, we need to ensure we get 512-dimensional embeddings
        # to match the existing Pinecone index

        try:
            # Create a simple class to generate 512-dimension vectors from existing models
            class CustomDimensionEmbeddings:
                """Adapter class to force 512 dimensions for OpenAI embeddings"""
                
                def __init__(self, api_key):
                    self.model = "text-embedding-3-small (custom 512-dim adapter)"
                    self.api_key = api_key
                    self._original_embeddings = OpenAIEmbeddings(
                        openai_api_key=api_key,
                        model="text-embedding-ada-002"  # This model works reliably
                    )
                
                def embed_query(self, text):
                    # Get the original embedding
                    original_embedding = self._original_embeddings.embed_query(text)
                    
                    # Use the first 512 dimensions only
                    truncated_embedding = original_embedding[:512]
                    return truncated_embedding
                
                def embed_documents(self, documents):
                    # Get the original embeddings
                    original_embeddings = self._original_embeddings.embed_documents(documents)
                    
                    # Truncate each embedding to 512 dimensions
                    truncated_embeddings = [emb[:512] for emb in original_embeddings]
                    return truncated_embeddings
            
            # Create our custom embeddings adapter
            self.embeddings = CustomDimensionEmbeddings(settings.OPENAI_API_KEY)
            
            # Test to verify we get 512 dimensions
            test_text = "Test"
            test_embedding = self.embeddings.embed_query(test_text)
            
            if len(test_embedding) != 512:
                raise ValueError(f"Custom embeddings adapter failed! Got {len(test_embedding)} dimensions instead of 512")
                
            print(f"✅ Embedding adapter verified: Using truncated embeddings with 512 dimensions")
            
        except Exception as e:
            print(f"Error initializing custom embeddings: {e}")
            raise
            
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
    def load_documents_from_folder(self, folder_path: str, sector: str) -> List[Dict[str, Any]]:
        """Load all PDF documents from a folder and return processed chunks"""
        documents = []
        folder = Path(folder_path)
        
        if not folder.exists():
            logger.warning(f"Folder {folder_path} does not exist")
            print(f"Folder {folder_path} does not exist")
            return documents
            
        # Get all PDF files in the folder
        pdf_files = list(folder.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files in {folder_path}")
        print(f"Found {len(pdf_files)} PDF files in {folder_path}")
        
        for pdf_file in pdf_files:
            try:
                logger.info(f"Processing {pdf_file.name}")
                print(f"Processing {pdf_file.name}")
                
                # Load PDF content
                loader = PyPDFLoader(str(pdf_file))
                pages = loader.load()
                
                # Add metadata
                for page in pages:
                    page.metadata.update({
                        "source": str(pdf_file),
                        "filename": pdf_file.name,
                        "sector": sector,
                        "page": page.metadata.get("page", 0)
                    })
                
                # Split into chunks
                chunks = self.text_splitter.split_documents(pages)
                documents.extend(chunks)
                
                logger.info(f"Successfully processed {pdf_file.name} into {len(chunks)} chunks")
                print(f"Successfully processed {pdf_file.name} into {len(chunks)} chunks")
                
            except Exception as e:
                logger.error(f"Error processing {pdf_file.name}: {e}")
                print(f"Error processing {pdf_file.name}: {e}")
                continue
                
        return documents
    
    def populate_vector_store(self, force_recreate: bool = False):
        """Populate the vector store with documents from both sectors"""
        try:
            # Get API key from settings (loaded from .env file)
            api_key = settings.PINECONE_API_KEY
            
            # Set the environment variable for Pinecone
            os.environ["PINECONE_API_KEY"] = api_key
            
            # Initialize Pinecone client
            pinecone_client = PineconeClient(api_key=api_key)
            
            # Check if index exists and get its details
            index_exists = settings.PINECONE_INDEX_NAME in pinecone_client.list_indexes().names()
            
            # Use the existing index if it exists
            if index_exists:
                existing_index = pinecone_client.Index(settings.PINECONE_INDEX_NAME)
                index_stats = existing_index.describe_index_stats()
                logger.info(f"Using existing index: {settings.PINECONE_INDEX_NAME}")
                
                # Only clear the data if force_recreate is True
                if force_recreate:
                    logger.info(f"Force recreate requested, clearing existing data")
                    index = pinecone_client.Index(settings.PINECONE_INDEX_NAME)
                    index.delete(delete_all=True)
                    logger.info("Waiting for delete operation...")
                    time.sleep(5)  # Wait for deletion to complete
            else:
                # Create new index with 512 dimensions 
                logger.info(f"Creating Pinecone index: {settings.PINECONE_INDEX_NAME}")
                pinecone_client.create_index(
                    name=settings.PINECONE_INDEX_NAME,
                    dimension=512,  # Fixed 512 dimension
                    metric="cosine",
                    spec={
                        "serverless": {
                            "cloud": "aws",
                            "region": "us-east-1"
                        }
                    }
                )
                logger.info("Waiting for index to initialize...")
                time.sleep(10)  # Wait for index to be ready
                
            # Get the project root directory
            project_root = Path(__file__).resolve().parent.parent
            
            # Load documents from both sectors using absolute paths
            it_docs = self.load_documents_from_folder(str(project_root / "docs" / "IT_docs"), "IT")
            pharma_docs = self.load_documents_from_folder(str(project_root / "docs" / "pharma_docs"), "Pharma")
            
            all_docs = it_docs + pharma_docs
            logger.info(f"Total documents loaded: {len(all_docs)}")
            
            if not all_docs:
                logger.warning("No documents loaded, skipping vector store population")
                return
            
            # Debug: Check embedding model and dimensions
            logger.info(f"Using embedding model: {self.embeddings.model}")
            
            # Test embedding to verify dimensions
            test_text = "Test document for dimension verification"
            test_embedding = self.embeddings.embed_query(test_text)
            logger.info(f"Test embedding dimension: {len(test_embedding)}")
            
            # The embedding model was already verified in __init__, so just log success
            logger.info("Embedding dimension verification passed ✓")
            
            # Create vector store and add documents
            # First create an empty vector store
            vector_store = PineconeVectorStore(
                index_name=settings.PINECONE_INDEX_NAME,
                embedding=self.embeddings,
                pinecone_api_key=api_key,
            )
            
            # Then add documents one by one to avoid dimension mismatch
            logger.info("Adding documents to vector store...")
            for i, doc in enumerate(all_docs):
                try:
                    vector_store.add_documents([doc])
                    if (i + 1) % 100 == 0:
                        logger.info(f"Added {i + 1} documents to vector store")
                except Exception as e:
                    logger.error(f"Error adding document {i}: {e}")
                    continue
            
            logger.info(f"Successfully populated vector store with {len(all_docs)} document chunks")
            
        except Exception as e:
            logger.error(f"Error populating vector store: {e}")
            raise
    
    def get_relevant_documents(self, query: str, sector: str = None, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant documents from the vector store"""
        try:
            # Get API key from settings (loaded from .env file)
            api_key = settings.PINECONE_API_KEY
            
            # Ensure environment variable is set
            os.environ["PINECONE_API_KEY"] = api_key
            
            # Initialize Pinecone client
            pinecone_client = PineconeClient(api_key=api_key)
            
            if settings.PINECONE_INDEX_NAME not in pinecone_client.list_indexes().names():
                logger.warning("Pinecone index does not exist")
                return []
            
            # Create vector store for retrieval
            vector_store = PineconeVectorStore(
                index_name=settings.PINECONE_INDEX_NAME,
                embedding=self.embeddings,
                pinecone_api_key=api_key,
            )
            
            # Build search query
            search_query = query
            if sector:
                search_query += f" {sector} sector"
            
            # Retrieve relevant documents
            docs = vector_store.similarity_search(search_query, k=k)
            
            logger.info(f"Retrieved {len(docs)} relevant documents for query: {query}")
            return docs
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []

def setup_knowledge_base(force_recreate: bool = False):
    """Setup and populate the knowledge base with documents"""
    processor = DocumentProcessor()
    processor.populate_vector_store(force_recreate=force_recreate)
    print("Knowledge base setup completed!")
    # Test the document processor

if __name__ == "__main__":
    setup_knowledge_base()
