import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from gemini_client import GeminiClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGEngine:
    """
    Retrieval-Augmented Generation (RAG) engine for FAQ chatbot
    Uses sentence embeddings for similarity search
    """
    
    def __init__(self, faq_file='faq.json'):
        """
        Initialize RAG engine with FAQ data
        
        Args:
            faq_file (str): Path to FAQ JSON file
        """
        logger.info("Initializing RAG engine...")
        
        # Load FAQ data
        self.faqs = self._load_faqs(faq_file)
        logger.info(f"Loaded {len(self.faqs)} FAQs from {faq_file}")
        
        # Initialize sentence transformer model for embeddings
        logger.info("Loading sentence transformer model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Sentence transformer model loaded successfully")
        
        # Create chunks (one chunk per Q&A pair)
        self.chunks = self._create_chunks()
        logger.info(f"Created {len(self.chunks)} chunks from FAQs")
        
        # Generate embeddings for all chunks
        self.chunk_embeddings = self._generate_embeddings()
        logger.info("Generated embeddings for all chunks")

        # Initialize Gemini client for query expansion
        self.gemini_client = GeminiClient()
        self.EXPAND_QUERY_PROMPT = """You are an expert query expander for a banking domain search engine. Your task is to rewrite a user's query to be more descriptive for a semantic search. Focus on the core concepts.

For a given user query, you should:
1.  Identify the core subject.
2.  If it's an acronym, Keep both the acronym and the expanded form.
3.  Remove all unnecessary words. For example, if the user query is "how to make payment for my loan?", the expanded query should be "make payment for loan".

Examples:
- "how to make payment for my loan?" -> "make payment for loan"
- "what is CDF" -> "CDF - cardholder dispute form?"

**User Query:**
`{query}`

**Expanded Search Query:**"""

    
    def _load_faqs(self, faq_file):
        """
        Load FAQs from JSON file (expects direct array format)
        
        Args:
            faq_file (str): Path to FAQ JSON file
            
        Returns:
            list: List of FAQ dictionaries
        """
        try:
            with open(faq_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data
        except Exception as e:
            logger.error(f"Error loading FAQs from {faq_file}: {e}")
            return []
    
    def _create_chunks(self):
        """
        Create text chunks from FAQs (one chunk per Q&A pair)
        
        Returns:
            list: List of chunk dictionaries with text and metadata
        """
        chunks = []
        for idx, faq in enumerate(self.faqs):
            # Combine question and answer into a single chunk
            chunk_text = f"Question: {faq['question']}\nAnswer: {faq['answer']}"
            chunks.append({
                'id': idx,
                'text': chunk_text,
                'question': faq['question'],
                'answer': faq['answer']
            })
        return chunks
    
    def _generate_embeddings(self):
        """
        Generate embeddings for all chunks
        
        Returns:
            np.ndarray: Array of embeddings
        """
        texts = [chunk['text'] for chunk in self.chunks]
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings

    def _expand_query(self, query):
        """
        Expand the user query for better search results.
        """
        # Heuristic to avoid expanding long queries
        if len(query.split()) > 3:
            return query

        prompt = self.EXPAND_QUERY_PROMPT.format(query=query)
        expanded_query = self.gemini_client.query(prompt)
        return expanded_query.strip()
    
    def retrieve_similar_chunks(self, query, top_k=10, chat_history=None, num_chat_pairs=10):
        """
        Retrieve top-k most similar chunks to the query with optional chat history context
        
        Args:
            query (str): User query
            top_k (int): Number of similar chunks to retrieve
            chat_history (list): Optional list of previous chat messages for context
            
        Returns:
            list: List of most similar chunks with similarity scores
        """
        logger.info(f"Retrieving top {top_k} similar chunks for query: '{query[:50]}...'")

        # Expand the query for better retrieval
        expanded_query = self._expand_query(query)
        logger.info(f"Expanded query: '{expanded_query}'")
        
        # Build enhanced query with relevant chat history context
        enhanced_query = expanded_query
        if chat_history and len(chat_history) > 0:
            history_messages = [msg['content'] for msg in chat_history[-2*num_chat_pairs:]]
            
            # Embed the current query and the history messages
            all_texts_to_embed = [expanded_query] + history_messages
            all_embeddings = self.model.encode(all_texts_to_embed)
            query_embedding = all_embeddings[0]
            history_embeddings = all_embeddings[1:]
            
            # Calculate similarity scores
            similarities = cosine_similarity([query_embedding], history_embeddings)[0]
            
            # Filter for relevant messages
            relevant_history = []
            similarity_threshold = 0.5 # This can be tuned
            for i, score in enumerate(similarities):
                if score > similarity_threshold:
                    relevant_history.append(history_messages[i])
            
            if relevant_history:
                history_context = " ".join(relevant_history)
                enhanced_query = f"Query: {expanded_query} Chat History: {history_context} "
                logger.info(f"Enhanced query with {len(relevant_history)} relevant messages from chat history")
            else:
                logger.info("No relevant chat history found to enhance the query.")

        # Generate embedding for the final enhanced query
        query_embedding_for_rag = self.model.encode([enhanced_query])[0]
        
        # Calculate cosine similarity between query and all chunks
        similarities = cosine_similarity(
            [query_embedding_for_rag],
            self.chunk_embeddings
        )[0]
        
        # Get top-k most similar chunks
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Prepare results
        results = []
        for idx in top_indices:
            results.append({
                'chunk': self.chunks[idx],
                'similarity_score': float(similarities[idx])
            })
        
        logger.info(f"Retrieved {len(results)} chunks. Top similarity score: {results[0]['similarity_score']:.4f}")
        
        return results
    
    def format_context_for_llm(self, retrieved_chunks):
        """
        Format retrieved chunks as context for LLM
        
        Args:
            retrieved_chunks (list): List of retrieved chunks with scores
            
        Returns:
            str: Formatted context string
        """
        context = "Here are the most relevant FAQs from the knowledge base:\n\n"
        
        for i, item in enumerate(retrieved_chunks, 1):
            chunk = item['chunk']
            score = item['similarity_score']
            context += f"[FAQ {i}] (Relevance: {score:.2f})\n"
            context += f"Q: {chunk['question']}\n"
            context += f"A: {chunk['answer']}\n\n"
        
        return context
    
    def get_all_questions(self):
        """
        Get all FAQ questions for reference
        
        Returns:
            list: List of all FAQ questions
        """
        return [faq['question'] for faq in self.faqs]

