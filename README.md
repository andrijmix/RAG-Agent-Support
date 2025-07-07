# Anzara Loan Assistant - RAG Agent with Multi-Agent Routing

A sophisticated chatbot system that intelligently routes user queries between FAQ retrieval and web search capabilities. Built with LangChain, LangGraph, and OpenAI GPT models.

## 🌟 Features

- **Intelligent Query Routing**: Automatically determines whether to use FAQ retrieval or web search based on user queries
- **RAG-based FAQ System**: Retrieval-Augmented Generation for answering questions about Anzara loan services
- **Web Search Integration**: DuckDuckGo search for general questions outside the loan domain
- **Multi-Agent Architecture**: Supervisor agent coordinates between specialized agents
- **Vector-based Similarity Search**: FAISS vector store for efficient FAQ retrieval

## 🏗️ Architecture

```
[Start]
   |
[Supervisor] ──▶ [FAQ Agent] ──▶ [END]
     │
     └────▶ [WEB Search Agent] ──▶ [END]
```

- **Supervisor Agent**: Routes queries to appropriate specialized agents
- **FAQ Agent**: Handles Anzara loan-related questions using RAG
- **Web Search Agent**: Processes general queries using web search

## 📋 Prerequisites

- Python 3.8+
- OpenAI API key
- Internet connection for web search functionality

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd anzara-loan-assistant
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_PROJECT=anzara-supervisor
   ```

## 🎯 Usage

### Running the Main Application

```bash
python langgraph_faq_web_router.py
```

This starts the interactive chat interface where you can ask questions about:
- Anzara loan services (routed to FAQ agent)
- General topics like weather, news, facts (routed to web search)

### Testing Individual Components

**FAQ Agent Only:**
```bash
python qa_agent.py
```

**Vector Store Testing:**
```bash
python vectorstore_utils.py
```

**RAG Utilities Testing:**
```bash
python rag_utils.py
```

## 💡 Example Queries

### FAQ Agent (Anzara-related)
- "What is an online loan?"
- "How does Anzara operate?"
- "What is the Anzara loan limit?"
- "What are the qualifications for loan eligibility?"

### Web Search Agent (General queries)
- "What's the weather today?"
- "Latest news about artificial intelligence"
- "Who won the latest football match?"

## 📁 Project Structure

```
├── langgraph_faq_web_router.py  # Main application with multi-agent routing
├── qa_agent.py                  # FAQ retrieval agent
├── faq_data.py                  # FAQ content database
├── rag_utils.py                 # RAG processing utilities
├── vectorstore_utils.py         # Vector store management
├── main.py                      # Basic setup testing
├── requirements.txt             # Python dependencies
├── architecture.txt             # System architecture diagram
├── .env                         # Environment variables (create this)
└── .gitignore                   # Git ignore rules
```

## 🔧 Configuration

### FAQ Content
Edit `faq_data.py` to modify or add FAQ entries. The system automatically processes Q&A pairs for vector storage.

### Model Configuration
- Default model: `gpt-3.5-turbo`
- Temperature: 0 (deterministic responses)
- Max search results: 3

### Vector Store Settings
- Embeddings: OpenAI embeddings
- Vector store: FAISS (CPU optimized)
- Similarity search: Top-k retrieval

## 🧪 Testing

The system includes built-in testing capabilities:

1. **Interactive Testing**: Run the main application and test with various queries
2. **Component Testing**: Test individual modules using their `if __name__ == "__main__"` blocks
3. **Error Handling**: Comprehensive error handling for API failures and invalid inputs

## 🔍 How It Works

1. **User Input**: User submits a query through the chat interface
2. **Supervisor Decision**: LLM-powered supervisor analyzes the query and routes it to the appropriate agent
3. **Agent Processing**: 
   - FAQ Agent: Retrieves relevant FAQ entries using vector similarity search
   - Web Search Agent: Searches the web using DuckDuckGo and formats results
4. **Response Generation**: Selected agent generates a comprehensive response
5. **Output**: Formatted answer returned to user with routing information

## 🛠️ Dependencies

- `langchain-community`: LangChain community integrations
- `langchain-openai`: OpenAI models integration
- `langgraph`: Graph-based agent orchestration
- `python-dotenv`: Environment variable management
- `duckduckgo-search`: Web search functionality
- `faiss-cpu`: Vector similarity search

## 📊 Monitoring

The system includes LangChain tracing for monitoring:
- Query routing decisions
- Agent performance metrics
- Response quality tracking

## 🚨 Error Handling

- API rate limiting protection
- Graceful degradation for search failures
- Input validation and sanitization
- Comprehensive logging for debugging

## 🔒 Security

- API keys stored in environment variables
- No sensitive data in version control
- Input sanitization for user queries

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the error logs for detailed information
2. Verify API key configuration
3. Ensure all dependencies are installed
4. Review the FAQ content format

## 🔮 Future Enhancements

- Support for multiple languages
- Advanced conversation memory
- Custom embedding models
- API endpoint exposure
- Database integration for FAQ management
- Analytics dashboard
- User authentication system