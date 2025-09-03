# AI Research Agent

A specialized AI research application with two specialized agents: **IT Research Agent** and **Pharmaceutical Research Agent**. Built with Django, LangChain, LangGraph, and LangSmith.

# Interface Sscreenshot

<img width="1317" height="837" alt="Screenshot 2025-09-03 133518" src="https://github.com/user-attachments/assets/43ada846-eb44-4ce3-b392-91a5d9717ff7" />

# Demo Video 

https://github.com/user-attachments/assets/e757c302-c325-4979-a8aa-d48d20f3bd4c

## Features

- 🤖 **Auto-Detection**: Automatically selects the best agent for your research topic
- 💻 **IT Research Agent**: Specialized in technology, software, cybersecurity, and IT topics
- 🏥 **Pharma Research Agent**: Specialized in medical research, drug development, and clinical trials
- 🔍 **Advanced Research**: Combines RAG (Retrieval-Augmented Generation) with web search
- 📊 **Comprehensive Reports**: Generates detailed research reports with findings and recommendations
- 🎨 **Modern UI**: Beautiful, responsive interface built with Bootstrap

## Architecture

```
Frontend (HTML/CSS/JS + Bootstrap)
           ↓
    Django Backend
           ↓
    AI Agent System
           ↓
    ┌─────────────────┐
    │   RAG System    │  ← Pinecone Vector DB
    │  (Knowledge)    │
    └─────────────────┘
           ↓
    ┌─────────────────┐
    │  Web Search     │  ← Tavily API
    │  (Real-time)    │
    └─────────────────┘
           ↓
    ┌─────────────────┐
    │  LangChain +    │
    │  LangGraph      │
    └─────────────────┘
```

## Prerequisites

- Python 3.8+
- Django 4.1+
- OpenAI API key
- Tavily API key
- Pinecone API key

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/gulshan-100/AI_Research_Agent.git
   cd AI_Research_Agent
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API keys**
   Edit `AI_Research_Agent/settings.py` and update:
   ```python
   OPENAI_API_KEY = "your-openai-api-key-here"
   TAVILY_API_KEY = "your-tavily-api-key-here"
   PINECONE_API_KEY = "your-pinecone-api-key-here"
   ```

4. **Run the application**
   ```bash
   python manage.py runserver
   ```

5. **Open your browser**
   Navigate to `http://127.0.0.1:8000/`

## Usage

### 1. Agent Selection
- **Auto Detect**: Let AI choose the best agent for your topic
- **IT Agent**: Force use of IT research agent
- **Pharma Agent**: Force use of pharmaceutical research agent

### 2. Research Process
1. Enter your research topic
2. Select agent (or use auto-detect)
3. Click "Start Research"
4. Wait for the AI to complete research
5. Review the comprehensive report

### 3. Report Structure
- **Executive Summary**: Brief overview of findings
- **Methodology**: How research was conducted
- **Key Findings**: Main discoveries
- **Analysis**: Detailed analysis
- **Recommendations**: Actionable insights
- **Sources**: References and sources used

## API Endpoints

- `GET /` - Main interface
- `POST /research/` - Submit research request
- `POST /detect-agent/` - Auto-detect appropriate agent
- `GET /status/` - System status

## Research Topics Examples

### IT Research Agent
- "Cloud computing security best practices"
- "Python web development frameworks"
- "Cybersecurity threats in 2024"
- "Machine learning deployment strategies"
- "DevOps implementation guide"

### Pharmaceutical Research Agent
- "COVID-19 vaccine development process"
- "Clinical trial phases for new drugs"
- "FDA drug approval timeline"
- "Drug interaction with common medications"
- "Clinical trial recruitment strategies"

## Technical Details

### Backend Technologies
- **Django**: Web framework
- **LangChain**: AI orchestration
- **LangGraph**: Workflow management
- **LangSmith**: Monitoring and debugging
- **Pinecone**: Vector database
- **OpenAI**: Language model
- **Tavily**: Web search

### Frontend Technologies
- **HTML5**: Structure
- **CSS3**: Styling with gradients and animations
- **JavaScript**: Interactivity
- **Bootstrap 5**: Responsive design
- **Font Awesome**: Icons

### AI Components
- **Base Research Agent**: Common functionality
- **IT Research Agent**: Technology specialization
- **Pharma Research Agent**: Medical specialization
- **Agent Selector**: Intelligent agent selection

