"""
Django views for AI Research Agent using LangGraph
Simple and easy to understand implementation with advanced agent capabilities
"""

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Import our agents
from .agents import ITResearchAgent, PharmaResearchAgent, AgentSelector

def index(request):
    """Main page with research interface"""
    return render(request, 'agent/index.html')

@csrf_exempt
@require_http_methods(["POST"])
def research_topic(request):
    """Handle research requests using LangGraph agents"""
    try:
        # Parse request data
        data = json.loads(request.body)
        topic = data.get('topic', '').strip()
        agent_choice = data.get('agent_choice', 'auto')  # auto, it, pharma
        
        if not topic:
            return JsonResponse({
                'success': False,
                'error': 'Research topic is required'
            })
        
        # Determine which agent to use
        if agent_choice == 'auto':
            selector = AgentSelector()
            selected_agent = selector.select_agent(topic)
        elif agent_choice == 'it':
            selected_agent = 'IT'
        elif agent_choice == 'pharma':
            selected_agent = 'Pharma'
        else:
            selected_agent = 'IT'  # Default fallback
        
        # Create the appropriate agent
        if selected_agent == 'IT':
            agent = ITResearchAgent()
        else:
            agent = PharmaResearchAgent()
        
        # Run research using LangGraph workflow
        with ThreadPoolExecutor() as executor:
            future = executor.submit(agent.research, topic)
            report = future.result()
        
        # Convert report to dictionary for JSON response
        report_data = {
            'content': report.content,
            'sources': report.sources,
            'agent_used': selected_agent,
            'topic': topic
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
def get_research_status(request):
    """Get status of ongoing research from LangGraph agents"""
    try:
        data = json.loads(request.body)
        topic = data.get('topic', '')
        agent_type = data.get('agent_type', '')
        
        # This would typically connect to LangGraph's checkpoint system
        # For now, return a basic status
        return JsonResponse({
            'success': True,
            'status': 'ready',
            'message': f'Research system is ready. Agent type: {agent_type}',
            'topic': topic
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Status check failed: {str(e)}'
        })
