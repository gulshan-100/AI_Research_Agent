"""
Test Django settings loading
"""

import os
import sys
import django

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'AI_Research_Agent.settings')
django.setup()

def test_settings():
    """Test if Django settings are loading correctly"""
    try:
        from django.conf import settings
        
        print("🔍 Testing Django Settings...")
        print(f"OPENAI_API_KEY: {'✓ Set' if hasattr(settings, 'OPENAI_API_KEY') and settings.OPENAI_API_KEY else '✗ Missing'}")
        print(f"TAVILY_API_KEY: {'✓ Set' if hasattr(settings, 'TAVILY_API_KEY') and settings.TAVILY_API_KEY else '✗ Missing'}")
        print(f"PINECONE_API_KEY: {'✓ Set' if hasattr(settings, 'PINECONE_API_KEY') and settings.PINECONE_API_KEY else '✗ Missing'}")
        print(f"PINECONE_INDEX_NAME: {'✓ Set' if hasattr(settings, 'PINECONE_INDEX_NAME') and settings.PINECONE_INDEX_NAME else '✗ Missing'}")
        
        # Check actual values (masked for security)
        if hasattr(settings, 'PINECONE_API_KEY') and settings.PINECONE_API_KEY:
            print(f"PINECONE_API_KEY value: {settings.PINECONE_API_KEY[:10]}...")
        if hasattr(settings, 'PINECONE_INDEX_NAME') and settings.PINECONE_INDEX_NAME:
            print(f"PINECONE_INDEX_NAME value: {settings.PINECONE_INDEX_NAME}")
            
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pinecone_connection():
    """Test Pinecone connection"""
    try:
        from django.conf import settings
        from pinecone import Pinecone
        
        print("\n🔍 Testing Pinecone Connection...")
        
        if not hasattr(settings, 'PINECONE_API_KEY') or not settings.PINECONE_API_KEY:
            print("❌ PINECONE_API_KEY not found in settings")
            return False
            
        # Test Pinecone client creation
        pinecone_client = Pinecone(api_key=settings.PINECONE_API_KEY)
        print("✓ Pinecone client created successfully")
        
        # Test listing indexes
        indexes = pinecone_client.list_indexes()
        print(f"✓ Found {len(indexes.names())} indexes: {list(indexes.names())}")
        
        # Check if our index exists
        if hasattr(settings, 'PINECONE_INDEX_NAME') and settings.PINECONE_INDEX_NAME:
            if settings.PINECONE_INDEX_NAME in indexes.names():
                print(f"✓ Index '{settings.PINECONE_INDEX_NAME}' exists")
            else:
                print(f"⚠ Index '{settings.PINECONE_INDEX_NAME}' does not exist (will be created)")
        
        return True
        
    except Exception as e:
        print(f"❌ Pinecone connection error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Django Settings and Pinecone Connection\n")
    
    tests = [
        ("Settings Test", test_settings),
        ("Pinecone Connection Test", test_pinecone_connection),
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
        print("🎉 All tests passed! Settings are loading correctly.")
    else:
        print("❌ Some tests failed. Please check the settings configuration.")

if __name__ == "__main__":
    main()
