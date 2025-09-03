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
   git clone <repository-url>
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
   PINECONE_ENVIRONMENT = "your-pinecone-environment-here"
   ```

4. **Run migrations**
   ```bash
   python manage.py migrate
   ```

5. **Setup knowledge base**
   ```bash
   python manage.py setup_kb
   ```

6. **Run the application**
   ```bash
   python manage.py runserver
   ```

7. **Open your browser**
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

## Customization

### Adding New Agents
1. Create a new agent class inheriting from `BaseResearchAgent`
2. Implement specialized methods for your domain
3. Add agent selection logic in `AgentSelector`
4. Update the frontend interface

### Modifying Research Process
1. Edit the research workflow in agent classes
2. Modify prompts for different research phases
3. Adjust the RAG system parameters
4. Customize report generation

## Troubleshooting

### Common Issues

1. **API Key Errors**
   - Ensure all API keys are correctly configured
   - Check API key permissions and quotas

2. **Pinecone Connection Issues**
   - Verify Pinecone environment and index name
   - Check network connectivity

3. **Research Failures**
   - Check console for error messages
   - Verify topic relevance to agent specialization
   - Ensure sufficient API credits

### Debug Mode
Enable Django debug mode in settings for detailed error messages:
```python
DEBUG = True
```

## Development

### Project Structure
```
AI_Research_Agent/
├── AI_Research_Agent/          # Main project
│   ├── settings.py             # Configuration
│   ├── urls.py                 # Main URLs
│   └── wsgi.py                 # WSGI config
├── agent/                      # Agent app
│   ├── agents.py               # AI agent classes
│   ├── views.py                # Django views
│   ├── urls.py                 # App URLs
│   ├── utils.py                # Utility functions
│   └── templates/              # HTML templates
├── requirements.txt             # Python dependencies
└── README.md                   # This file
```

### Running Tests
```bash
python manage.py test
```

### Code Style
- Follow PEP 8 guidelines
- Use descriptive variable names
- Add docstrings to functions and classes
- Keep functions simple and focused

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is for educational purposes. Please ensure compliance with all API terms of service.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review Django and LangChain documentation
3. Check API provider documentation
4. Create an issue in the repository

## Future Enhancements

- [ ] User authentication system
- [ ] Research history and saving
- [ ] Export reports to PDF/DOCX
- [ ] Collaborative research features
- [ ] Additional specialized agents
- [ ] Advanced analytics dashboard
- [ ] API rate limiting and caching
- [ ] Multi-language support

---

**Note**: This is an assignment project demonstrating AI agent implementation. Ensure you have proper API access and credits before running the application.
