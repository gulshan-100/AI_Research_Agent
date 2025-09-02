"""
Generate a comprehensive report from the vector database
"""

import os
import sys
import django
from pathlib import Path

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'AI_Research_Agent.settings')
django.setup()

def generate_report(topic, sector, output_file):
    """Generate a comprehensive report on the given topic for a specific sector"""
    try:
        from agent.document_loader import DocumentProcessor
        from langchain_openai import ChatOpenAI
        from langchain.chains import LLMChain
        from langchain.prompts import PromptTemplate
        from django.conf import settings
        
        print(f"🔍 Generating report on '{topic}' for the {sector} sector...")
        
        # Get relevant documents
        processor = DocumentProcessor()
        docs = processor.get_relevant_documents(topic, sector=sector, k=10)
        
        if not docs:
            print("❌ No relevant documents found. Cannot generate report.")
            return False
        
        print(f"✓ Found {len(docs)} relevant documents")
        
        # Extract information from the documents
        context = ""
        for i, doc in enumerate(docs):
            source = doc.metadata.get('filename', 'Unknown')
            page = doc.metadata.get('page', 'Unknown')
            content = doc.page_content
            
            # Add to context
            context += f"\n\nSource {i+1}: {source} (Page {page})\n{content}"
        
        # Create prompt template
        template = """
        You are an expert research analyst specializing in the {sector} sector in India.
        Your task is to create a comprehensive report on the topic: {topic}
        
        Use ONLY the following sources to create your report:
        
        {context}
        
        Generate a professional, well-structured report with the following sections:
        1. Executive Summary
        2. Introduction and Background
        3. Current Market Landscape
        4. Key Trends and Developments
        5. Challenges and Opportunities
        6. Future Outlook
        7. Recommendations
        8. References (cite the sources properly)
        
        Make the report fact-based, data-driven and insightful. Include relevant statistics, market sizes, growth rates, and key players when available from the sources.
        Format the report in proper markdown with headings, subheadings, bullet points, and emphasis where appropriate.
        """
        
        prompt = PromptTemplate(
            input_variables=["sector", "topic", "context"],
            template=template
        )
        
        # Initialize LLM
        llm = ChatOpenAI(
            temperature=0.2,
            model="gpt-4o",
            openai_api_key=settings.OPENAI_API_KEY
        )
        
        # Create chain
        chain = LLMChain(llm=llm, prompt=prompt)
        
        # Run chain
        print(f"🤖 Generating report... (this may take a few minutes)")
        result = chain.run(sector=sector, topic=topic, context=context)
        
        # Save to file
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result)
        
        print(f"✅ Report generated successfully and saved to: {output_file}")
        return True
        
    except Exception as e:
        print(f"❌ Report generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Generate reports for both sectors"""
    print("📊 Report Generation Tool\n")
    
    # Create reports directory
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    
    # Generate IT sector report
    it_topic = "Current State and Future Outlook of the Indian IT Industry"
    it_file = reports_dir / "indian_it_industry_report.md"
    it_success = generate_report(it_topic, "IT", it_file)
    
    # Generate Pharma sector report
    pharma_topic = "Current State and Future Outlook of the Indian Pharmaceutical Industry"
    pharma_file = reports_dir / "indian_pharma_industry_report.md"
    pharma_success = generate_report(pharma_topic, "Pharma", pharma_file)
    
    if it_success and pharma_success:
        print("\n🎉 Both reports generated successfully!")
    elif it_success:
        print("\n⚠️ Only IT sector report generated successfully.")
    elif pharma_success:
        print("\n⚠️ Only Pharma sector report generated successfully.")
    else:
        print("\n❌ Report generation failed for both sectors.")

if __name__ == "__main__":
    main()
