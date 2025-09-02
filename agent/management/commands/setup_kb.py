"""
Django management command to setup the knowledge base
Usage: python manage.py setup_kb
"""

from django.core.management.base import BaseCommand
from django.conf import settings
from agent.utils import populate_knowledge_base, get_knowledge_base_stats

class Command(BaseCommand):
    help = 'Setup the AI Research Agent knowledge base with sample documents'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--clear',
            action='store_true',
            help='Clear existing knowledge base before populating',
        )
    
    def handle(self, *args, **options):
        self.stdout.write(
            self.style.SUCCESS('Starting knowledge base setup...')
        )
        
        # Check if API keys are configured
        if not all([
            settings.OPENAI_API_KEY != "your-openai-api-key-here",
            settings.TAVILY_API_KEY != "your-tavily-api-key-here",
            settings.PINECONE_API_KEY != "your-pinecone-api-key-here"
        ]):
            self.stdout.write(
                self.style.ERROR(
                    'Please configure your API keys in settings.py first!'
                )
            )
            return
        
        # Get current stats
        current_stats = get_knowledge_base_stats()
        self.stdout.write(f"Current knowledge base status: {current_stats['status']}")
        
        if options['clear']:
            self.stdout.write('Clearing existing knowledge base...')
            # Clear functionality would go here if needed
        
        # Populate knowledge base
        self.stdout.write('Populating knowledge base with sample documents...')
        success = populate_knowledge_base()
        
        if success:
            # Get updated stats
            new_stats = get_knowledge_base_stats()
            self.stdout.write(
                self.style.SUCCESS(
                    f'Knowledge base setup completed successfully!'
                )
            )
            self.stdout.write(f"Documents in knowledge base: {new_stats.get('document_count', 0)}")
        else:
            self.stdout.write(
                self.style.ERROR('Failed to setup knowledge base')
            )
