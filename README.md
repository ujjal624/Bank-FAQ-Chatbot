# HDFC Bank FAQ Chatbot

An intelligent FAQ chatbot for HDFC Bank using Retrieval-Augmented Generation (RAG) with Gemma 3 27B model. The chatbot provides accurate answers to banking-related queries with advanced features like guardrails, chat history, and query tracking.

## Features

### Core Functionality
- **RAG-based Question Answering**: Uses semantic search to retrieve top 10 most relevant FAQ chunks
- **Gemma 3 27B Integration**: Powered by Google's Gemma 3 27B instruction-tuned model via Gemini API
- **Semantic Search**: Uses sentence transformers for intelligent similarity matching
- **Context-Aware Responses**: Provides answers based on official HDFC Bank FAQ knowledge base

### Advanced Features
1. **Chat History**: Maintains last 10 interactions as context for better conversational flow
2. **Relevance Guardrails**: Detects and politely declines irrelevant (non-banking) queries
3. **Unknown Query Tracking**: Automatically logs queries with low similarity scores for expert review
4. **Comprehensive Logging**: All interactions and system events are logged to `chatbot.log`

### User Interface
- Modern, responsive web interface built with HTML, CSS, and JavaScript
- Real-time chat experience with typing indicators
- Mobile-friendly design
- Clean, professional banking-themed UI

## Project Structure

```
FAQ-chatbot-HDFCBank/
├── app.py                  # Main Flask application with routing and logic
├── gemini_client.py        # LLM client for Gemma 3 27B via Gemini API
├── rag_engine.py          # RAG engine with similarity search
├── faq.json               # FAQ knowledge base (2000+ banking FAQs)
├── Pipfile                # Dependency management (Pipenv)
├── Pipfile.lock           # Locked dependencies
├── requirements.txt       # Alternative dependency file
├── .gitignore            # Git ignore rules
├── chatbot.log           # Application logs (generated)
├── unknown_queries.json  # Tracked unknown queries (generated)
├── README.md             # This file
├── templates/
│   └── index.html        # Main frontend template
└── static/
    ├── style.css         # Application styling
    └── script.js         # Frontend JavaScript
```

## Installation

### Prerequisites
- Python 3.13 (or compatible version)
- Pipenv (recommended) or pip
- Google Gemini API key with access to Gemma 3 27B model

### Step 1: Clone or Download the Project

```bash
cd /path/to/FAQ-chatbot-HDFCBank
```

### Step 2: Set Up Environment Variables

Create a `.env` file in the project root and add your Gemini API key:

```bash
GEMINI_API_KEY=your_actual_api_key_here
```

**Note**: Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

### Step 3: Install Dependencies

#### Option A: Using Pipenv (Recommended)

```bash
# Install Pipenv if not already installed
pip install pipenv

# Install dependencies
pipenv install

# Activate virtual environment
pipenv shell
```

#### Option B: Using pip

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install flask python-dotenv google-generativeai sentence-transformers numpy scikit-learn requests torch
```

## Usage

### Running the Application

1. Make sure you're in the project directory and have activated your virtual environment

2. Start the Flask server:
   ```bash
   python app.py
   ```

3. Open your web browser and navigate to:
   ```
   http://localhost:5002
   ```

4. Start chatting! Ask questions like:
   - "How can I reset my Internet Banking password?"
   - "What are the different types of fund transfers available?"
   - "How do I activate my user ID?"
   - "Can I link multiple accounts to one user ID?"

### Example Queries

**Banking Queries (Handled):**
- "How do I enable my HDFC Bank relationships online?"
- "What are the credit card service requests I can make?"
- "Tell me about bill payment options"
- "How can I transfer funds to another bank?"
- "How do I change my password?"
- "What information is required for NEFT?"

**Irrelevant Queries (Will Be Declined Politely):**
- "What's the weather today?"
- "Tell me a joke"
- "Who won the cricket match?"

## API Endpoints

### `GET /`
Renders the main chat interface

### `POST /chat`
Handles chat requests

**Request Body:**
```json
{
  "query": "Your question here"
}
```

**Response:**
```json
{
  "response": "Answer from the chatbot",
  "relevant": true,
  "top_similarity": 0.85
}
```

Or if query is irrelevant:
```json
{
  "response": "Polite decline message",
  "relevant": false
}
```

### `POST /clear_history`
Clears the chat history for the current session

**Response:**
```json
{
  "status": "success"
}
```

### `GET /unknown_queries`
Returns list of queries that weren't well-covered by the FAQ database

**Response:**
```json
{
  "unknown_queries": [
    {
      "query": "Question that wasn't answered well",
      "timestamp": "2025-10-08T12:34:56"
    }
  ]
}
```

### `GET /health`
Health check endpoint

**Response:**
```json
{
  "status": "healthy",
  "rag_chunks": 2000,
  "model": {
    "model_name": "gemma-3-27b-it",
    "provider": "Google Gemini API",
    "type": "Gemma 3 27b Instruction Tuned"
  }
}
```

## How It Works

### RAG Pipeline

1. **Query Reception**: User submits a question through the web interface

2. **Semantic Search**: Query is embedded using sentence transformers and compared against FAQ embeddings with chat history context

3. **Relevance Check**: LLM determines if the query is banking-related
   - ✅ Relevant → Continue to next step
   - ❌ Irrelevant → Politely decline and suggest banking topics

4. **Retrieval**: Top 10 most similar FAQ chunks are retrieved based on cosine similarity

5. **Context Formation**: Retrieved FAQs are formatted as context for the LLM

6. **Answer Generation**: Gemma 3 27B generates a natural language answer using:
   - Retrieved FAQ context
   - Last 10 chat interactions (chat history)
   - Query text

7. **Unknown Query Tracking**: If similarity score < 0.5, query is logged for expert review

### Guardrails

The chatbot implements multiple layers of guardrails:

1. **Domain Relevance**: Ensures queries are related to banking/HDFC Bank services
2. **Context-Aware**: Uses chat history to maintain conversation context and improve retrieval
3. **Quality Tracking**: Logs low-confidence answers for human review

## Monitoring and Maintenance

### Viewing Logs

All application activity is logged to `chatbot.log`:

```bash
tail -f chatbot.log
```

### Reviewing Unknown Queries

Access the unknown queries endpoint to see questions that need expert attention:

```bash
curl http://localhost:5002/unknown_queries
```

Or view the `unknown_queries.json` file directly:

```bash
cat unknown_queries.json
```

### Updating FAQ Knowledge Base

To add new FAQs, edit `faq.json` and restart the application. The FAQ file is a JSON array with the following format:

```json
[
  {
    "question": "Your new question?",
    "answer": "Your detailed answer here.",
    "found_duplicate": false
  }
]
```

**Note**: The `"found_duplicate"` field is optional and ignored by the system. Only `"question"` and `"answer"` fields are used.

## Configuration

### Model Settings

The LLM generation parameters are configured in `gemini_client.py`:

- **Temperature**: 0.7 (controls randomness)
- **Top P**: 0.9 (nucleus sampling)
- **Top K**: 40 (top-k sampling)
- **Max Output Tokens**: 1000

### RAG Settings

RAG parameters are in `rag_engine.py`:

- **Embedding Model**: `all-MiniLM-L6-v2` (sentence-transformers)
- **Top K Retrieval**: 10 chunks
- **Similarity Threshold**: 0.5 (for unknown query tracking)

## Troubleshooting

### Issue: "GEMINI_API_KEY not found"

**Solution**: Make sure you've created a `.env` file with your API key:
```bash
cp .env.example .env
# Edit .env and add your actual API key
```

### Issue: Model not found or access denied

**Solution**: Verify your API key has access to Gemma 3 27B model. You may need to request access through Google AI Studio.

### Issue: Slow response times

**Solution**: 
- The first query may be slow due to model initialization
- Check your internet connection
- Consider using a lighter embedding model if needed

### Issue: Port 5002 already in use

**Solution**: Change the port in `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5003)  # Use different port
```

## Development

### Running in Development Mode

The application runs in debug mode by default on port 5002. To disable debug mode:

```python
# In app.py
app.run(debug=False, host='0.0.0.0', port=5002)
```

### Adding New Features

The modular structure makes it easy to extend:

- **New guardrails**: Add functions in `app.py` similar to `check_relevance()`
- **Different LLM**: Modify `gemini_client.py` to use another model
- **Enhanced RAG**: Update `rag_engine.py` to add reranking, filtering, etc.
- **UI improvements**: Edit files in `templates/` and `static/`

## Technology Stack

- **Backend**: Flask (Python web framework)
- **LLM**: Gemma 3 27B via Google Gemini API
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Operations**: NumPy, scikit-learn
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Logging**: Python logging module

## Security Notes

- Never commit your `.env` file or API keys to version control
- The `.gitignore` file is configured to exclude sensitive files
- Use environment variables for all secrets
- Consider adding rate limiting for production deployments

## License

This project is for educational and demonstration purposes.

## Support

For issues related to:
- **HDFC Bank services**: Contact HDFC Bank customer care
- **Technical issues**: Check the troubleshooting section above
- **API access**: Visit [Google AI Studio](https://makersuite.google.com)

## Contributors

Built with ❤️ for intelligent banking customer support.

