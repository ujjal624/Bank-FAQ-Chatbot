import os
import json
import logging
from datetime import datetime
from flask import Flask, Response, render_template, request, jsonify, session
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

def generate_answer(query, retrieved_chunks, chat_history):
    """
    Generate answer using LLM with RAG context and chat history
    
    Args:
        query (str): User query
        retrieved_chunks (list): Retrieved FAQ chunks
        chat_history (list): Last 10 chat interactions
        
    Returns:
        str: Generated answer
    """
    logger.info(f"Generating answer for query: '{query[:50]}...'")
    
    # Check if the query is well-covered by the retrieved FAQs
    top_similarity = retrieved_chunks[0]['similarity_score'] if retrieved_chunks else 0
    # Format context from retrieved chunks
    context = rag_engine.format_context_for_llm(retrieved_chunks)
    
    print("How context is being formatted: ", context)
    # If similarity is low, treat as unknown query
    if top_similarity < 0.5:
        logger.warning(f"Low similarity score ({top_similarity:.2f}) - treating as unknown query")
        save_unknown_query(query, datetime.now().isoformat())
        
        # Return predetermined polite response without using LLM
        response = ("Thank you for your question. I apologize, but I don't have specific information about this in my current knowledge base. "
                   "Your query has been saved and will be forwarded to our subject matter experts for review. "
                   "They will get back to you with a detailed response. "
                   "In the meantime, if you have any other banking-related questions, I'd be happy to help!")
        
        logger.info("Returned predetermined response for unknown query")
        return response
    
    # # Format context from retrieved chunks
    # context = rag_engine.format_context_for_llm(retrieved_chunks)
    
    # print("How context is being formatted: ", context)
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

def split_combined_query(user_query):
    """Use LLM to split a combined query into a list of individual queries."""
    logger.info(f"Analyzing for combined query: '{user_query}'")
    
    prompt = f"""You are a query analysis expert. Your task is to determine if a user's query is a single question or a combined question containing multiple distinct topics.

- If it is a single question, return the original query.
- If it is a combined question, split it into a comma-separated list of individual, self-contained questions.

Examples:
- User Query: "What is the interest rate on a personal loan?"
- Your Response: "What is the interest rate on a personal loan?"

- User Query: "How do I reset my internet banking password and what are the forex card charges?"
- Your Response: "How do I reset my internet banking password, what are the forex card charges?"

- User Query: "Tell me about credit cards and also the process for applying for a new debit card"
- Your Response: "Tell me about credit cards, what is the process for applying for a new debit card"

User Query: "{user_query}"
Your Response:"""
    
    response = llm_client.query(prompt)
    logger.info(f"LLM analysis result for combined query: '{response}'")
    return [q.strip() for q in response.split(',')]


@app.route('/')
def index():
    """Render main chat interface"""
    logger.info("Rendering main page")
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat requests with streaming response"""
    try:
        data = request.json
        user_query = data.get('query', '').strip()
        
        if not user_query:
            return jsonify({'error': 'Empty query'}), 400
        
        logger.info(f"Received chat request: '{user_query}'")
        
        chat_history = data.get('history', [])

        # --- Streaming Logic ---
        def stream_response_generator():
            # This generator function now handles the core logic and streams the final response.
            
            # Step 1: Split combined queries
            sub_queries = split_combined_query(user_query)
            
            all_answers = []
            all_retrieved_chunks_for_prompt = [] # For single-query context
            final_query_for_single_prompt = user_query

            # Step 2: Process each sub-query
            for sub_query in sub_queries:
                if not sub_query: continue

                retrieved_chunks, final_query = rag_engine.retrieve_similar_chunks(
                    sub_query, top_k=5, chat_history=chat_history, num_chat_pairs=5
                )
                
                # Store chunks for single-query prompt context
                if len(sub_queries) == 1:
                    all_retrieved_chunks_for_prompt = retrieved_chunks
                    final_query_for_single_prompt = final_query

                is_relevant, _ = check_relevance(sub_query, retrieved_chunks, chat_history)
                
                if is_relevant:
                    answer = generate_answer(final_query, retrieved_chunks, chat_history)
                    all_answers.append(answer)
                else:
                    all_answers.append(f"Regarding '{sub_query}', I can only assist with banking-related topics.")

            # Step 3: Determine the final prompt for streaming
            final_prompt = ""
            if len(all_answers) > 1:
                qa_pairs_str = ""
                for i, (question, answer) in enumerate(zip(sub_queries, all_answers), 1):
                    qa_pairs_str += f"Question {i}: {question}\nAnswer {i}: {answer}\n\n"
                
                final_prompt = f"""You are a helpful AI assistant. Synthesize a final, conversational response based on the user's original query and the internal Q&A processing.

Original User Query: \"{user_query}\"

Processed Questions and Answers:
{qa_pairs_str}

Your Final, Synthesized Response:"""
            
            elif all_answers:
                # Re-create the prompt for the single answer to stream it properly
                context = rag_engine.format_context_for_llm(all_retrieved_chunks_for_prompt)
                final_prompt = f"""You are a helpful HDFC Bank customer service assistant. Answer the user's question based on the provided FAQ knowledge base.

{context}

User's Question: {final_query_for_single_prompt}

Your Answer:"""
            else:
                yield "I'm sorry, I couldn't find a relevant answer. Please try rephrasing."
                return

            # Step 4: Stream the final response from the determined prompt
            for chunk in llm_client.stream_query(final_prompt, chat_history):
                yield chunk

        return Response(stream_response_generator(), mimetype='text/plain')

    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}", exc_info=True)
        return jsonify({'error': 'An error occurred processing your request'}), 500


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

