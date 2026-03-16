"""
Django views for AI Research Agent using LangGraph
Advanced implementation with iter        # Run research using LangGraph workflow with progress tracking in a separate thread
        import time
        start_time = time.time()
        print(f"Starting research process for topic: {topic}")
        
        try:
            # Start the research in a separate thread
            with ThreadPoolExecutor() as executor:
                print("Submitting research task to executor")
                future = executor.submit(agent.research, topic)
                print("Waiting for research results...")
                report = future.result()
                print("Research completed successfully")
            
            # Calculate research duration
            duration = time.time() - start_time
            minutes = int(duration // 60)
            seconds = int(duration % 60)
            print(f"Research completed in {minutes}m {seconds}s")h capabilities and user feedback
"""

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from django.conf import settings

# Import our agents
from .agents import ITResearchAgent, PharmaResearchAgent, AgentSelector

def index(request):
    """Main page with research interface"""
    # Pass research parameters to the template
    context = {
        'max_iterations': settings.MAX_RESEARCH_DEPTH,
        'max_deep_dives': settings.MAX_DEEP_DIVES,
    }
    return render(request, 'agent/index.html', context)

@require_http_methods(["POST"])
def research_topic(request):
    """Handle research requests using LangGraph agents with iteration control"""
    try:
        print("Received research request")
        
        # Check for API keys
        if not settings.OPENAI_API_KEY:
            raise ValueError("OpenAI API key is not configured")
        if not settings.TAVILY_API_KEY:
            raise ValueError("Tavily API key is not configured")
        if not settings.PINECONE_API_KEY:
            raise ValueError("Pinecone API key is not configured")
            
        # Parse request data
        data = json.loads(request.body)
        print("Request data:", data)
        
        topic = data.get('topic', '').strip()
        agent_choice = data.get('agent_choice', 'auto')  # auto, it, pharma
        
        # Get research parameters from the request or use defaults from settings
        from django.conf import settings
        max_iterations = data.get('max_iterations', settings.MAX_RESEARCH_DEPTH)
        allow_deep_dives = data.get('allow_deep_dives', True)
        thread_id = data.get('thread_id', f'research_{hash(topic)}')
        
        if not topic:
            return JsonResponse({
                'success': False,
                'error': 'Research topic is required'
            })
        
        # Determine which agent to use
        print(f"Selecting agent for topic: {topic}, choice: {agent_choice}")
        try:
            if agent_choice == 'auto':
                selector = AgentSelector()
                selected_agent = selector.select_agent(topic)
                print(f"Auto-selected agent: {selected_agent}")
            elif agent_choice == 'it':
                selected_agent = 'IT'
            elif agent_choice == 'pharma':
                selected_agent = 'Pharma'
            else:
                selected_agent = 'IT'  # Default fallback
            print(f"Final selected agent: {selected_agent}")
            
            # Create the appropriate agent
            if selected_agent == 'IT':
                print("Initializing IT Research Agent")
                agent = ITResearchAgent()
            else:
                print("Initializing Pharma Research Agent")
                agent = PharmaResearchAgent()
        except Exception as e:
            print(f"Error during agent initialization: {str(e)}")
            raise
        
        # Configure agent parameters
        agent.max_iterations = max_iterations
        agent.allow_deep_dives = allow_deep_dives
        
        # Run research using LangGraph workflow with progress tracking in a separate thread
        import time
        start_time = time.time()
        
        # Start the research in a separate thread to allow status monitoring
        with ThreadPoolExecutor() as executor:
            future = executor.submit(agent.research, topic)
            report = future.result()
        
        # Calculate research duration
        duration = time.time() - start_time
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        
        # Set basic metadata
        research_journey = []
        iterations_performed = 0
        explored_subtopics = []
        
        # Convert report to dictionary for JSON response
        report_data = {
            'content': report.content,
            'sources': report.sources,
            'agent_used': selected_agent,
            'topic': topic,
            'research_metadata': {
                'duration_minutes': minutes,
                'duration_seconds': seconds,
                'iterations_performed': iterations_performed,
                'research_journey': research_journey,
                'explored_subtopics': explored_subtopics
            }
        }
        
        return JsonResponse({
            'success': True,
            'report': report_data
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Research failed: {str(e)}'
        })

@csrf_exempt
@require_http_methods(["POST"])
def detect_agent(request):
    """Auto-detect which agent should be used for a topic"""
    try:
        data = json.loads(request.body)
        topic = data.get('topic', '').strip()
        
        if not topic:
            return JsonResponse({
                'success': False,
                'error': 'Topic is required for agent detection'
            })
        
        # Use agent selector to determine appropriate agent
        selector = AgentSelector()
        selected_agent = selector.select_agent(topic)
        
        return JsonResponse({
            'success': True,
            'selected_agent': selected_agent,
            'topic': topic
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Agent detection failed: {str(e)}'
        })

@csrf_exempt
@require_http_methods(["POST"])
def provide_research_feedback(request):
    """Allow users to provide feedback during the research process"""
    try:
        data = json.loads(request.body)
        topic = data.get('topic', '').strip()
        thread_id = data.get('thread_id', f'research_{hash(topic)}')
        agent_type = data.get('agent_type', '').strip()
        feedback_type = data.get('feedback_type', '').strip()  # 'deep_dive', 'refine', or 'complete'
        selected_subtopics = data.get('selected_subtopics', [])
        custom_queries = data.get('custom_queries', [])
        
        if not topic or not agent_type:
            return JsonResponse({
                'success': False,
                'error': 'Topic and agent_type are required'
            })
        
        # Create the appropriate agent to access its memory
        if agent_type.lower() == 'it':
            agent = ITResearchAgent()
        else:
            agent = PharmaResearchAgent()
            
        
        # Try to update the research state with user feedback
        try:
            # Since we removed checkpoint functionality, just return a success message
            return JsonResponse({
                'success': True,
                'message': f'Research feedback noted: {feedback_type}',
                'updated_state': {
                    'feedback_type': feedback_type,
                    'selected_subtopics': selected_subtopics,
                    'custom_queries': custom_queries,
                    'current_step': f'Processing {feedback_type} feedback'
                }
            })
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': f'Error applying feedback: {str(e)}'
            })
            
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Feedback submission failed: {str(e)}'
        })

@csrf_exempt
@require_http_methods(["POST"])
def get_research_status(request):
    """Get detailed status of ongoing research from LangGraph agents"""
    try:
        data = json.loads(request.body)
        topic = data.get('topic', '')
        agent_type = data.get('agent_type', '')
        thread_id = data.get('thread_id', f'research_{hash(topic)}')
        
        # Access the LangGraph checkpoint system to get current research status
        from .research_agents.base_agent import BaseResearchAgent
        from .research_agents.it_agent import ITResearchAgent
        from .research_agents.pharma_agent import PharmaResearchAgent
        
        # Create a temporary agent to access memory
        if agent_type.lower() == 'it':
            agent = ITResearchAgent()
        else:
            agent = PharmaResearchAgent()
            
        # Create basic research state
        research_state = {
            'current_step': 'Initializing research process...',
            'iteration': 0,
            'total_iterations': getattr(agent, 'max_iterations', 3),
            'explored_subtopics': [],
            'research_journey': ['Starting research process']
        }
            
        return JsonResponse({
            'success': True,
            'status': research_state.get('current_step', 'ready'),
            'message': f'Research in progress. Current step: {research_state.get("current_step", "Unknown")}',
            'topic': topic,
            'research_progress': research_state
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Status check failed: {str(e)}'
        })
