import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import re
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

        # Initialize BM25 index for sparse retrieval
        self.bm25_index = self._initialize_bm25()
        logger.info("Initialized BM25 index for hybrid retrieval")

        # Initialize Gemini client for query expansion
        self.gemini_client = GeminiClient()
        self.EXPAND_QUERY_PROMPT = """You are an expert query expander for a banking domain search engine. Your task is to rewrite a user's query to be more descriptive for a semantic search. Focus on the core concepts.

For a given user query, you should:
1.  Identify the core subject.
2.  If it's an acronym, expand it.
3.  Remove all unnecessary words. For example, if the user query is "how to make payment for my loan?", the expanded query should be "make payment for loan".

Examples:
- "how to make payment for my loan?" -> "make payment for loan"
- "what is CDF" -> "cardholder dispute form?"

**User Query:**
`{query}`

**Expanded Search Query:**"""
        self.REWRITE_QUERY_PROMPT = """You are an expert query rewriter. Your task is to take a conversation history and a new user query and rewrite the new query to be self-contained. The user's new query might be a follow-up question that relies on the context of the conversation. You must resolve pronouns and add the necessary context from the history.

**Conversation History:**
```
{chat_history}
```

**User's New Query:**
`{query}`

**Rewritten, Self-Contained Query:**"""

    
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
    
    def _tokenize(self, text):
        """
        Tokenize text for BM25 (simple tokenization)
        
        Args:
            text (str): Text to tokenize
            
        Returns:
            list: List of tokens
        """
        # Convert to lowercase and split by whitespace and punctuation
        text = text.lower()
        # Keep alphanumeric and spaces, replace everything else with space
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        tokens = text.split()
        return tokens
    
    def _initialize_bm25(self):
        """
        Initialize BM25 index from chunks
        
        Returns:
            BM25Okapi: BM25 index
        """
        # Tokenize all chunk texts
        tokenized_corpus = [self._tokenize(chunk['text']) for chunk in self.chunks]
        # Create BM25 index
        bm25 = BM25Okapi(tokenized_corpus)
        return bm25

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

    def _rewrite_query_with_history(self, query, chat_history):
        if not chat_history:
            return query

        # Format the history
        formatted_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])

        prompt = self.REWRITE_QUERY_PROMPT.format(
            chat_history=formatted_history,
            query=query
        )
        
        rewritten_query = self.gemini_client.query(prompt)
        logger.info(f"Rewritten query with history: '{rewritten_query.strip()}'")
        return rewritten_query.strip()
    
    def _hybrid_retrieval(self, query, top_k=10, bm25_weight=0.5, embedding_weight=0.5):
        """
        Perform hybrid retrieval combining BM25 and embedding-based search using Reciprocal Rank Fusion (RRF)
        
        Args:
            query (str): Search query
            top_k (int): Number of results to return
            bm25_weight (float): Weight for BM25 scores (default 0.5)
            embedding_weight (float): Weight for embedding scores (default 0.5)
            
        Returns:
            list: List of top-k chunks with hybrid scores
        """
        # 1. BM25 Retrieval (sparse)
        tokenized_query = self._tokenize(query)
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        
        # Normalize BM25 scores to [0, 1]
        if np.max(bm25_scores) > 0:
            bm25_scores_normalized = bm25_scores / np.max(bm25_scores)
        else:
            bm25_scores_normalized = bm25_scores
        
        # 2. Embedding-based Retrieval (dense)
        query_embedding = self.model.encode([query])[0]
        embedding_similarities = cosine_similarity(
            [query_embedding],
            self.chunk_embeddings
        )[0]
        
        # Embedding similarities are already in [0, 1] range for cosine similarity
        
        # 3. Combine scores using weighted linear combination
        # Both scores are normalized, so we can combine them directly
        hybrid_scores = (bm25_weight * bm25_scores_normalized) + (embedding_weight * embedding_similarities)
        
        # 4. Get top-k results
        top_indices = np.argsort(hybrid_scores)[-top_k:][::-1]
        
        # 5. Prepare results with detailed scores
        results = []
        for idx in top_indices:
            results.append({
                'chunk': self.chunks[idx],
                'similarity_score': float(hybrid_scores[idx]),
                'bm25_score': float(bm25_scores_normalized[idx]),
                'embedding_score': float(embedding_similarities[idx])
            })
        
        logger.info(f"Hybrid retrieval complete. Top score: {results[0]['similarity_score']:.4f} "
                   f"(BM25: {results[0]['bm25_score']:.4f}, Embedding: {results[0]['embedding_score']:.4f})")
        
        return results
    
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
        logger.info(f"Original query: '{query[:50]}...'")

        # 1. Rewrite the query using chat history to make it self-contained
        history_for_rewrite = None
        if chat_history:
            history_for_rewrite = chat_history[-2*num_chat_pairs:]
        rewritten_query = self._rewrite_query_with_history(query, history_for_rewrite)
        
        # 2. Expand the (potentially rewritten) query for better retrieval
        expanded_query = self._expand_query(rewritten_query)
        logger.info(f"Final query for retrieval: '{expanded_query}'")
        
        # 3. Perform hybrid retrieval (combining BM25 and embeddings)
        results = self._hybrid_retrieval(
            query=expanded_query,
            top_k=top_k,
            bm25_weight=0.4,      # BM25 weight for keyword matching
            embedding_weight=0.6   # Embedding weight for semantic similarity
        )
        
        logger.info(f"Retrieved {len(results)} chunks using hybrid retrieval. "
                   f"Top hybrid score: {results[0]['similarity_score']:.4f}")
        
        return results, expanded_query
    
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

