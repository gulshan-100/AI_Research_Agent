"""
URL patterns for AI Research Agent app
"""

from django.urls import path
from . import views

app_name = 'agent'

urlpatterns = [
    path('', views.index, name='index'),
    path('research/', views.research_topic, name='research_topic'),
    path('detect-agent/', views.detect_agent, name='detect_agent'),
    path('status/', views.get_research_status, name='status'),
]
