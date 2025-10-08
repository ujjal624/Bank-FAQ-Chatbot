import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging

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
        
        # Build enhanced query with chat history context
        enhanced_query = query
        if chat_history and len(chat_history) > 0:
            # Include recent conversation context to improve retrieval
            history_context = " ".join([msg['content'] for msg in chat_history[-2*num_chat_pairs:]])  # Last 10 pairs
            enhanced_query = f"{history_context} {query}"
            logger.info(f"Enhanced query with {len(chat_history[-2*num_chat_pairs:])} messages from chat history")
        
        # Generate embedding for query
        query_embedding = self.model.encode([enhanced_query])[0]
        
        # Calculate cosine similarity between query and all chunks
        similarities = cosine_similarity(
            [query_embedding],
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

