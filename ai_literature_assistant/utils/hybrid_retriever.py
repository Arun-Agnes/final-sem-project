# utils/hybrid_retriever.py
"""
Hybrid Search System combining Semantic and Keyword search
Uses BM25 for keyword matching and Sentence Transformers for semantic similarity
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder


class HybridRetriever:
    """Combines BM25 keyword search with semantic embedding search"""
    
    def __init__(self, embedding_model: str = "all-mpnet-base-v2"):
        """
        Initialize hybrid retriever
        
        Args:
            embedding_model: Name of sentence transformer model
        """
        self.embed_model = SentenceTransformer(embedding_model)
        self.bm25 = None
        self.chunks = []
        self.chunk_embeddings = None
        self.chunk_metadata = []
        
    def index_chunks(self, chunks: List[str], metadata: List[Dict] = None):
        """
        Index chunks for hybrid search
        
        Args:
            chunks: List of text chunks
            metadata: Optional metadata for each chunk
        """
        self.chunks = chunks
        self.chunk_metadata = metadata if metadata else [{} for _ in chunks]
        
        # Prepare BM25 (keyword search)
        tokenized_chunks = [chunk.lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)
        
        # Prepare embeddings (semantic search)
        print(f"[HYBRID] Encoding {len(chunks)} chunks...")
        self.chunk_embeddings = self.embed_model.encode(
            chunks,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        print("[HYBRID] Indexing complete")
        
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.5,
        doc_filter: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Hybrid retrieval combining BM25 and semantic search
        
        Args:
            query: Search query
            top_k: Number of results to return
            alpha: Weight for semantic search (0-1). 1 = pure semantic, 0 = pure keyword
            doc_filter: Optional list of document IDs to filter by
            
        Returns:
            List of dicts with chunk text, score, and metadata
        """
        if not self.chunks or self.bm25 is None or self.chunk_embeddings is None:
            raise ValueError("No chunks indexed. Call index_chunks() first.")
        
        # Apply document filter if provided
        if doc_filter:
            valid_indices = [
                i for i, meta in enumerate(self.chunk_metadata)
                if meta.get('paper_id') in doc_filter
            ]
            if not valid_indices:
                return []
        else:
            valid_indices = list(range(len(self.chunks)))
        
        # BM25 scores (keyword)
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Semantic scores
        query_embedding = self.embed_model.encode(query, convert_to_numpy=True)
        semantic_scores = np.dot(self.chunk_embeddings, query_embedding)
        
        # Normalize scores to 0-1 range
        def normalize(scores):
            min_s, max_s = scores.min(), scores.max()
            if max_s - min_s == 0:
                return np.zeros_like(scores)
            return (scores - min_s) / (max_s - min_s)
        
        bm25_scores_norm = normalize(bm25_scores)
        semantic_scores_norm = normalize(semantic_scores)
        
        # Hybrid score: weighted combination
        hybrid_scores = alpha * semantic_scores_norm + (1 - alpha) * bm25_scores_norm
        
        # Filter and get top-k
        filtered_scores = [(i, hybrid_scores[i]) for i in valid_indices]
        filtered_scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in filtered_scores[:top_k]]
        
        # Build results
        results = []
        for idx in top_indices:
            result = {
                'text': self.chunks[idx],
                'score': float(hybrid_scores[idx]),
                'bm25_score': float(bm25_scores_norm[idx]),
                'semantic_score': float(semantic_scores_norm[idx]),
                'metadata': self.chunk_metadata[idx],
                'index': idx
            }
            results.append(result)
        
        return results


class ReRanker:
    """Re-ranks retrieved results using cross-encoder for better accuracy"""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize re-ranker
        
        Args:
            model_name: Cross-encoder model name
        """
        self.model = CrossEncoder(model_name)
        
    def rerank(
        self,
        query: str,
        results: List[Dict],
        top_k: int = 5
    ) -> List[Dict]:
        """
        Re-rank results using cross-encoder
        
        Args:
            query: Original search query
            results: List of result dicts with 'text' field
            top_k: Number of top results to return
            
        Returns:
            Re-ranked results with updated scores
        """
        if not results:
            return []
        
        # Extract texts
        texts = [r['text'] for r in results]
        
        # Create query-text pairs
        pairs = [[query, text] for text in texts]
        
        # Get cross-encoder scores
        scores = self.model.predict(pairs, show_progress_bar=False)
        
        # Update results with new scores
        for i, result in enumerate(results):
            result['rerank_score'] = float(scores[i])
            result['original_score'] = result.get('score', 0)
        
        # Sort by rerank score
        reranked = sorted(results, key=lambda x: x['rerank_score'], reverse=True)
        
        return reranked[:top_k]


class QueryExpander:
    """Expands queries for better retrieval coverage"""
    
    @staticmethod
    def expand_with_synonyms(query: str) -> List[str]:
        """
        Simple query expansion with common research paper terms
        
        Args:
            query: Original query
            
        Returns:
            List of expanded queries
        """
        queries = [query]
        
        # Common expansions for research queries
        expansions = {
            'method': ['methodology', 'approach', 'technique'],
            'result': ['findings', 'outcome', 'conclusion'],
            'model': ['architecture', 'framework', 'system'],
            'dataset': ['data', 'corpus', 'benchmark'],
            'performance': ['accuracy', 'results', 'metrics'],
            'limitation': ['weakness', 'drawback', 'constraint']
        }
        
        query_lower = query.lower()
        for key, synonyms in expansions.items():
            if key in query_lower:
                for synonym in synonyms:
                    expanded = query_lower.replace(key, synonym)
                    if expanded != query_lower:
                        queries.append(expanded)
        
        return list(set(queries))[:3]  # Return up to 3 unique queries
