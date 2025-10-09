import os
import json
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session
from gemini_client import GeminiClient
from rag_engine import RAGEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management

# Initialize RAG engine and LLM client
logger.info("Starting FAQ Chatbot application...")
rag_engine = RAGEngine('faq.json')
llm_client = GeminiClient()
logger.info("Application initialized successfully")

# Storage for unknown queries (queries not answered by FAQ)
UNKNOWN_QUERIES_FILE = 'unknown_queries.json'

def load_unknown_queries():
    """Load unknown queries from file"""
    if os.path.exists(UNKNOWN_QUERIES_FILE):
        try:
            with open(UNKNOWN_QUERIES_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_unknown_query(query, timestamp):
    """Save an unknown query for expert review"""
    unknown_queries = load_unknown_queries()
    unknown_queries.append({
        'query': query,
        'timestamp': timestamp
    })
    with open(UNKNOWN_QUERIES_FILE, 'w') as f:
        json.dump(unknown_queries, f, indent=2)
    logger.info(f"Saved unknown query: {query}")

def check_relevance(query, retrieved_chunks, chat_history):
    """
    Check if query is relevant to banking using LLM with context from FAQ and chat history
    
    Args:
        query (str): User query
        retrieved_chunks (list): Retrieved FAQ chunks from similarity search
        chat_history (list): Previous chat interactions
        
    Returns:
        tuple: (is_relevant, reason)
    """
    logger.info(f"Checking relevance for query: '{query}'")
    
    # Format context from retrieved chunks
    context = rag_engine.format_context_for_llm(retrieved_chunks) if retrieved_chunks else "No relevant FAQ context found."
    
    prompt = f"""You are a banking domain expert. Determine if the following question is related to banking, financial services, or HDFC Bank services.

Question: "{query}"

Available FAQ Context:
{context}

Based on the question, the available FAQ context, and the conversation history - only take history when the question needs some context from the conversation history, 
respond with ONLY "RELEVANT" or "IRRELEVANT" followed by a brief reason.
Consider that follow-up questions might reference previous context from the conversation.

Examples:
- "What is my account balance?" -> RELEVANT: This is about banking account services.
- "How do I make a credit card payment?" -> RELEVANT: This is about banking payment services.
- "What is the weather today?" -> IRRELEVANT: This is about weather, not banking.
- "Tell me a joke" -> IRRELEVANT: This is not related to banking services.
- "when did prime minister sleep last time?" -> IRRELEVANT: This is not related to banking services.
- "How to make payment for my loan?" -> RELEVANT: This is about banking payment services.
- "what is the capital of india?" -> IRRELEVANT: This is not related to banking services.

Your response:"""
    
    response = llm_client.query(prompt, history=chat_history[-14:] if chat_history else None)
    
    is_relevant = "RELEVANT" in response.upper() and "IRRELEVANT" not in response.split('\n')[0].upper()
    
    logger.info(f"Relevance check result: {'RELEVANT' if is_relevant else 'IRRELEVANT'}")
    
    return is_relevant, response

def determine_unknown_query(query, retrieved_chunks, chat_history):
    """
    Use LLM to determine if there's sufficient information in the RAG context to answer the query
    
    Args:
        query (str): User query
        retrieved_chunks (list): Retrieved FAQ chunks from RAG
        chat_history (list): Previous chat interactions
        
    Returns:
        tuple: (is_unknown, response)
            - is_unknown (bool): True if query cannot be answered with available info
            - response (str): The response to send to the user
    """
    logger.info(f"Determining if query has sufficient information: '{query[:50]}...'")
    
    # # Check if the query is well-covered by the retrieved FAQs
    # top_similarity = retrieved_chunks[0]['similarity_score'] if retrieved_chunks else 0
    
    # Format context from retrieved chunks
    context = rag_engine.format_context_for_llm(retrieved_chunks) if retrieved_chunks else "No relevant FAQ context found."
    
    # print("How context is being formatted: ", context)
    
    # Build evaluation prompt
    evaluation_prompt = f"""You are an intelligent assistant evaluating whether you have sufficient information to answer a user's query.

Available FAQ Knowledge Base:
{context}

User's Question: {query}

TASK:
Carefully analyze whether the provided FAQ knowledge base contains sufficient and relevant information to answer the user's question accurately.

EVALUATION CRITERIA:
1. Does the FAQ knowledge base contain specific information that directly addresses the user's question?
2. Is the information complete enough to provide a helpful and accurate answer?
3. Consider the conversation history if the question builds on previous context.

OUTPUT FORMAT:
You MUST respond in this exact format:

STATUS: [SUFFICIENT or INSUFFICIENT]
REASONING: [Brief explanation of your decision]

If STATUS is SUFFICIENT, you will be asked to answer the question in the next step.
If STATUS is INSUFFICIENT, provide a polite response explaining that the information is not available.

Examples:
- If FAQ has clear answer: STATUS: SUFFICIENT
- If FAQ is vague or missing key details: STATUS: INSUFFICIENT
- If FAQ is completely unrelated: STATUS: INSUFFICIENT

Your evaluation:"""

    evaluation_response = llm_client.query(evaluation_prompt, history=chat_history[-20:] if chat_history else None)
    
    logger.info(f"LLM evaluation result: {evaluation_response[:80]}...")
    
    # Parse the response to determine if it's unknown
    is_unknown = "STATUS: INSUFFICIENT" in evaluation_response.upper() or "INSUFFICIENT" in evaluation_response.split('\n')[0].upper()
    
    if is_unknown:
        logger.warning(f"LLM determined insufficient information to answer query")
        save_unknown_query(query, datetime.now().isoformat())
        
        # Return predetermined polite response without using LLM
        response = ("Thank you for your question. I apologize, but I don't have specific information about this in my current knowledge base. "
                   "Your query has been saved and will be forwarded to our subject matter experts for review. "
                   "They will get back to you with a detailed response. "
                   "In the meantime, if you have any other banking-related questions, I'd be happy to help!")
        
        logger.info("Returned predetermined response for unknown query")
        return True, response
    else:
        logger.info("LLM determined sufficient information available")
        return False, None


def generate_answer(query, retrieved_chunks, chat_history):
    """
    Generate answer using LLM with RAG context and chat history
    
    Args:
        query (str): User query
        retrieved_chunks (list): Retrieved FAQ chunks
        chat_history (list): Last 20 chat interactions
        
    Returns:
        str: Generated answer
    """
    logger.info(f"Generating answer for query: '{query[:50]}...'")
    
    # Format context from retrieved chunks
    context = rag_engine.format_context_for_llm(retrieved_chunks)
    
    print("How context is being formatted: ", context)
    # Build the prompt
    prompt = f"""You are a helpful HDFC Bank customer service assistant. Answer the user's question based on the provided FAQ knowledge base.

{context}

INSTRUCTIONS:
1. Use the FAQ information above to answer the question accurately and concisely.
2. If the information is directly available in the FAQs, provide a clear answer.
3. Do not mention the FAQ number in the answer.
4. If you're not completely sure the FAQs cover the user's specific question, acknowledge this and provide the closest relevant information available.
5. Keep your response natural and conversational.
6. Answer the question in such a way that it can be directly told to a customer.
7. Keep the formatting simple.


User's Question: {query}

Your Answer:"""
    # Generate response with chat history
    response = llm_client.query(prompt, history=chat_history[-20:] if chat_history else None)
    
    logger.info(f"Generated answer (length: {len(response)} chars)")
    
    return response

@app.route('/')
def index():
    """Render main chat interface"""
    logger.info("Rendering main page")
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat requests"""
    try:
        data = request.json
        user_query = data.get('query', '').strip()
        
        if not user_query:
            return jsonify({'error': 'Empty query'}), 400
        
        logger.info(f"Received chat request: '{user_query}'")
        
        # Get or initialize chat history from session
        if 'chat_history' not in session:
            session['chat_history'] = []
        
        chat_history = session['chat_history']
        
        # Step 1: Retrieve relevant FAQs using RAG first with chat history context
        retrieved_chunks, final_query = rag_engine.retrieve_similar_chunks(user_query, top_k=10, chat_history=chat_history[-20:] if chat_history else None, num_chat_pairs=10)
        # print("RAG Response: ", retrieved_chunks)
        
        # Step 2: Check relevance with context and chat history
        is_relevant, relevance_response = check_relevance(user_query, retrieved_chunks, chat_history)
        
        if not is_relevant:
            logger.info("Query is irrelevant to banking - declining politely")
            response = "I'm sorry, but I'm specifically designed to help with HDFC Bank and banking-related questions. Your question appears to be outside my area of expertise. Is there anything related to banking or HDFC Bank services that I can help you with?"
            
            # Add to chat history
            chat_history.append({'role': 'user', 'content': user_query})
            chat_history.append({'role': 'assistant', 'content': response})
            session['chat_history'] = chat_history[-20:]  # Keep last 20 messages (10 interactions)
            
            return jsonify({
                'response': response,
                'relevant': False
            })
        
        # Step 3: Determine if we have sufficient information to answer (LLM-based evaluation)
        is_unknown, unknown_response = determine_unknown_query(final_query, retrieved_chunks, chat_history)
        
        if is_unknown:
            logger.info("Query identified as unknown - insufficient information to answer")
            answer = unknown_response
        else:
            # Step 4: Generate answer using LLM (we have sufficient information)
            answer = generate_answer(final_query, retrieved_chunks, chat_history)
            logger.info("Answer generated successfully for the query")
        
        # Add to chat history
        chat_history.append({'role': 'user', 'content': user_query})
        chat_history.append({'role': 'assistant', 'content': answer})
        print("\n")
        print("CHAT HISTORY: ", chat_history)
        print("\n")
        session['chat_history'] = chat_history[-20:]  # Keep last 20 messages (10 interactions)
        
        logger.info(f" Chat history length: {len(chat_history)}")
        
        return jsonify({
            'response': answer,
            'relevant': True,
            'top_similarity': retrieved_chunks[0]['similarity_score'] if retrieved_chunks else 0
        })
        
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}", exc_info=True)
        return jsonify({'error': 'An error occurred processing your request'}), 500

@app.route('/clear_history', methods=['POST'])
def clear_history():
    """Clear chat history"""
    logger.info("Clearing chat history")
    session['chat_history'] = []
    return jsonify({'status': 'success'})

@app.route('/unknown_queries', methods=['GET'])
def get_unknown_queries():
    """Get list of unknown queries for expert review"""
    logger.info("Fetching unknown queries")
    unknown_queries = load_unknown_queries()
    return jsonify({'unknown_queries': unknown_queries})

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'rag_chunks': len(rag_engine.chunks),
        'model': llm_client.get_model_info()
    })

if __name__ == '__main__':
    logger.info("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5002)

