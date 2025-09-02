"""
Django management command to populate the knowledge base with documents
"""

from django.core.management.base import BaseCommand
from agent.document_loader import setup_knowledge_base

class Command(BaseCommand):
    help = 'Populate the knowledge base with documents from the docs folder'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force recreate the knowledge base (clear existing data)',
        )
    
    def handle(self, *args, **options):
        force = options['force']
        
        if force:
            self.stdout.write(
                self.style.WARNING('Force recreate requested. This will clear existing data.')
            )
        
        try:
            self.stdout.write('Starting knowledge base population...')
            setup_knowledge_base(force_recreate=force)
            self.stdout.write(
                self.style.SUCCESS('Knowledge base population completed successfully!')
            )
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Error populating knowledge base: {e}')
            )
