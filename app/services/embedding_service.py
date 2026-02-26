"""Embedding service for generating vector embeddings"""

import json
from typing import List, Optional
import numpy as np
from functools import lru_cache

from app.config import settings


class EmbeddingService:
    """Service for generating text embeddings using OpenAI or local models"""
    
    _instance: Optional["EmbeddingService"] = None
    _model = None
    _openai_client = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @property
    def is_openai(self) -> bool:
        return settings.embedding_provider == "openai"
    
    @property
    def openai_client(self):
        if self._openai_client is None and self.is_openai:
            from openai import OpenAI
            self._openai_client = OpenAI(api_key=settings.openai_api_key)
            print(f"OpenAI client initialized for model: {settings.embedding_model}")
        return self._openai_client
    
    @property
    def local_model(self):
        if self._model is None and not self.is_openai:
            from sentence_transformers import SentenceTransformer
            print(f"Loading local embedding model: {settings.embedding_model}")
            self._model = SentenceTransformer(settings.embedding_model)
            print(f"Model loaded. Embedding dimension: {self._model.get_sentence_embedding_dimension()}")
        return self._model
    
    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        if self.is_openai:
            response = self.openai_client.embeddings.create(
                model=settings.embedding_model,
                input=text
            )
            return response.data[0].embedding
        else:
            embedding = self.local_model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        if not texts:
            return []
        
        if self.is_openai:
            # OpenAI supports batch embedding
            response = self.openai_client.embeddings.create(
                model=settings.embedding_model,
                input=texts
            )
            # Sort by index to maintain order
            sorted_data = sorted(response.data, key=lambda x: x.index)
            return [item.embedding for item in sorted_data]
        else:
            embeddings = self.local_model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()
    
    def embed_to_json(self, text: str) -> str:
        """Generate embedding and return as JSON string (for SQLite storage)"""
        embedding = self.embed(text)
        return json.dumps(embedding)
    
    @staticmethod
    def embedding_from_json(json_str: str) -> List[float]:
        """Parse embedding from JSON string"""
        return json.loads(json_str)
    
    @staticmethod
    def cosine_similarity(embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings.
        
        DEPRECATED for memory search: Use pgvector's built-in cosine distance operator
        (<=> in SQL) instead. This method is kept for non-pgvector code paths (e.g., SQLite).
        """
        a = np.array(embedding1)
        b = np.array(embedding2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    def search_similar(
        self,
        query_embedding: List[float],
        candidate_embeddings: List[tuple[str, List[float]]],
        top_k: int = 10,
        min_similarity: float = 0.0
    ) -> List[tuple[str, float]]:
        """
        Search for similar embeddings (in-memory search for SQLite).
        Returns list of (id, similarity_score) tuples.
        """
        query = np.array(query_embedding)
        results = []
        
        for item_id, embedding in candidate_embeddings:
            candidate = np.array(embedding)
            similarity = float(np.dot(query, candidate) / (np.linalg.norm(query) * np.linalg.norm(candidate)))
            if similarity >= min_similarity:
                results.append((item_id, similarity))
        
        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


@lru_cache()
def get_embedding_service() -> EmbeddingService:
    """Get the singleton embedding service"""
    return EmbeddingService()
