#!/usr/bin/env python3
"""
Test script to verify Pinecone API key configuration
"""

import os
import sys
import django

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'AI_Research_Agent.settings')
django.setup()

def test_pinecone_config():
    """Test if Pinecone configuration is working"""
    try:
        from django.conf import settings
        from pinecone import Pinecone
        
        print("🔍 Testing Pinecone Configuration...")
        print(f"PINECONE_API_KEY: {'✓ Set' if hasattr(settings, 'PINECONE_API_KEY') and settings.PINECONE_API_KEY else '✗ Missing'}")
        print(f"PINECONE_ENVIRONMENT: {'✓ Set' if hasattr(settings, 'PINECONE_ENVIRONMENT') and settings.PINECONE_ENVIRONMENT else '✗ Missing'}")
        print(f"PINECONE_INDEX_NAME: {'✓ Set' if hasattr(settings, 'PINECONE_INDEX_NAME') and settings.PINECONE_INDEX_NAME else '✗ Missing'}")
        
        if not hasattr(settings, 'PINECONE_API_KEY') or not settings.PINECONE_API_KEY:
            print("❌ PINECONE_API_KEY not found in settings")
            return False
            
        # Test Pinecone client creation
        print("\n🔍 Testing Pinecone Client Creation...")
        pinecone_client = Pinecone(api_key=settings.PINECONE_API_KEY)
        print("✓ Pinecone client created successfully")
        
        # Test listing indexes
        print("\n🔍 Testing Index Listing...")
        indexes = pinecone_client.list_indexes()
        print(f"✓ Found {len(indexes.names())} indexes: {list(indexes.names())}")
        
        # Check if our index exists
        if hasattr(settings, 'PINECONE_INDEX_NAME') and settings.PINECONE_INDEX_NAME:
            if settings.PINECONE_INDEX_NAME in indexes.names():
                print(f"✓ Index '{settings.PINECONE_INDEX_NAME}' exists")
            else:
                print(f"⚠ Index '{settings.PINECONE_INDEX_NAME}' does not exist (will be created)")
        
        print("\n✅ Pinecone configuration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Pinecone configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_agent_creation():
    """Test if we can create a research agent"""
    try:
        print("\n🔍 Testing Agent Creation...")
        from agent.agents import BaseResearchAgent
        
        agent = BaseResearchAgent()
        
        if agent.vector_store:
            print("✓ Agent created successfully with vector store")
            return True
        else:
            print("⚠ Agent created but vector store is None")
            return False
            
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Pinecone Configuration Tests...\n")
    
    # Test Pinecone configuration
    config_ok = test_pinecone_config()
    
    if config_ok:
        # Test agent creation
        agent_ok = test_agent_creation()
        
        if agent_ok:
            print("\n🎉 All tests passed! Pinecone is configured correctly.")
        else:
            print("\n⚠ Pinecone config is OK but agent creation has issues.")
    else:
        print("\n❌ Pinecone configuration failed. Please check your settings.")
    
    return config_ok

if __name__ == "__main__":
    main()


