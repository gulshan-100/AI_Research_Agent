#!/usr/bin/env python3
"""
Test script to verify that .env file is loaded correctly and API keys are accessible
"""

import os
import sys

# Setup Django environment
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'AI_Research_Agent.settings')

try:
    import django
    django.setup()
    
    from django.conf import settings
    
    print("🔑 Testing Environment Variable Loading")
    print("=" * 40)
    
    # Test if .env file is loaded
    print("📁 Checking if .env file exists...")
    if os.path.exists('.env'):
        print("✅ .env file found")
        
        # Read .env file to show what's in it
        with open('.env', 'r') as f:
            env_content = f.read()
        print("📄 .env file contents:")
        print("-" * 30)
        for line in env_content.strip().split('\n'):
            if line.strip() and not line.startswith('#'):
                key = line.split('=')[0]
                value = line.split('=')[1] if '=' in line else ''
                masked_value = value[:10] + "..." if len(value) > 10 else value
                print(f"  {key}: {masked_value}")
        print("-" * 30)
    else:
        print("❌ .env file not found")
        print("💡 Create a .env file with your API keys")
    
    print("\n🔍 Testing API Key Access from Settings...")
    
    # Test OpenAI API Key
    openai_key = getattr(settings, 'OPENAI_API_KEY', None)
    if openai_key:
        print(f"✅ OPENAI_API_KEY: {openai_key[:10]}...")
    else:
        print("❌ OPENAI_API_KEY: Not found")
    
    # Test Tavily API Key
    tavily_key = getattr(settings, 'TAVILY_API_KEY', None)
    if tavily_key:
        print(f"✅ TAVILY_API_KEY: {tavily_key[:10]}...")
    else:
        print("❌ TAVILY_API_KEY: Not found")
    
    # Test Pinecone API Key
    pinecone_key = getattr(settings, 'PINECONE_API_KEY', None)
    if pinecone_key:
        print(f"✅ PINECONE_API_KEY: {pinecone_key[:10]}...")
    else:
        print("❌ PINECONE_API_KEY: Not found")
    
    # Test LangSmith API Key
    langsmith_key = getattr(settings, 'LANGSMITH_API_KEY', None)
    if langsmith_key:
        print(f"✅ LANGSMITH_API_KEY: {langsmith_key[:10]}...")
    else:
        print("❌ LANGSMITH_API_KEY: Not found")
    
    print("\n🎯 Summary:")
    api_keys = [openai_key, tavily_key, pinecone_key, langsmith_key]
    valid_keys = [key for key in api_keys if key]
    
    if len(valid_keys) == 4:
        print("✅ All API keys are loaded successfully!")
        print("🚀 Your application should work with the .env configuration")
    else:
        print(f"⚠️  {len(valid_keys)} out of 4 API keys are loaded")
        print("🔧 Check your .env file and ensure all required keys are set")
    
except Exception as e:
    print(f"❌ Error testing environment: {e}")
    print("🔧 Make sure Django is properly configured")
