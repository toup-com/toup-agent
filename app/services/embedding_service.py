"""Embedding service for generating vector embeddings.

Supports OpenAI and local (sentence-transformers) models.
Auto-falls back to local model when OpenAI key is unavailable.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import numpy as np
from functools import lru_cache

from app.config import settings

logger = logging.getLogger(__name__)

# Default local model — small (80MB), fast, 384 dims
LOCAL_DEFAULT_MODEL = "all-MiniLM-L6-v2"


class EmbeddingService:
    """Service for generating text embeddings using OpenAI or local models"""

    _instance: Optional["EmbeddingService"] = None
    _model = None
    _openai_client = None
    _resolved_provider: Optional[str] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def reset_provider_cache(cls) -> None:
        """Drop the resolved provider/client/model so the NEXT access
        re-resolves against the current settings.

        MUST be called wherever bind-critical settings change at runtime
        (/admin/bind, tunnel config_update). Pool containers boot in LOBBY
        mode (llm_mode="manual", no toup_token): the agent_main boot
        pre-load touches `provider` at ~25% boot, which resolves and caches
        "local" — and bind can only arrive AFTER boot finishes (the bridge
        waits for lobby health `ready`). Without this reset, every freshly
        claimed or blue-green-recreated pool container keeps provider
        "local" for its whole process lifetime after bind flips
        llm_mode/toup_token, so every memory write degrades to an
        unembedded row (dedup off, vector recall blind) — the silent
        follow-up to canary 533354ce.

        Note the properties assign `self._resolved_provider = ...`, which
        shadows the class attribute on the SINGLETON'S instance dict — so
        clear the instance dict too, not just the class slots. Singleton
        identity is preserved on purpose: long-lived holders
        (MemoryService.embedding_service, MemoryDedupService, …) re-resolve
        on their next use instead of keeping a stale instance.
        """
        cls._resolved_provider = None
        cls._openai_client = None
        cls._model = None
        inst = cls._instance
        if inst is not None:
            for attr in ("_resolved_provider", "_openai_client", "_model"):
                inst.__dict__.pop(attr, None)

    @staticmethod
    def _proxy_embeddings_active() -> bool:
        """True when embeddings should route through the platform proxy
        (bundle mode + flag) instead of a raw OpenAI key in the container.
        Lets embeddings work with NO OPENAI_API_KEY present — the toup_token
        + proxy carry the call. See hardening-runbook.md Step 1."""
        return bool(
            getattr(settings, "embeddings_via_proxy", False)
            and getattr(settings, "llm_mode", "") == "bundle"
            and getattr(settings, "toup_token", "")
        )

    @property
    def provider(self) -> str:
        """Resolve actual provider: use OpenAI if a key exists OR the proxy
        path is active; otherwise fall back to local."""
        if self._resolved_provider is None:
            if settings.embedding_provider == "openai" and (
                settings.openai_api_key or self._proxy_embeddings_active()
            ):
                self._resolved_provider = "openai"
            else:
                self._resolved_provider = "local"
                if settings.embedding_provider == "openai":
                    logger.warning(
                        "OPENAI_API_KEY not set and proxy path inactive — "
                        "falling back to local embedding model"
                    )
        return self._resolved_provider

    @property
    def is_openai(self) -> bool:
        return self.provider == "openai"

    @property
    def model_name(self) -> str:
        if self.is_openai:
            return settings.embedding_model
        # For local: use configured model or default
        if settings.embedding_provider == "local" and settings.embedding_model != "text-embedding-3-small":
            return settings.embedding_model
        return LOCAL_DEFAULT_MODEL

    @property
    def dimension(self) -> int:
        """Return the embedding dimension for the active model."""
        if self.is_openai:
            return settings.embedding_dimension
        # For local models, query the model itself
        model = self.local_model
        if model:
            return model.get_sentence_embedding_dimension()
        return 384  # default for all-MiniLM-L6-v2

    @property
    def openai_client(self):
        if self._openai_client is None and self.is_openai:
            from openai import OpenAI
            if self._proxy_embeddings_active():
                # Route through the platform proxy with the scoped TOUP_TOKEN —
                # no raw provider key needed in the container. The SDK POSTs to
                # {base_url}/embeddings → /llm/openai/v1/embeddings.
                base = f"{settings.platform_api_url.rstrip('/')}/llm/openai/v1"
                self._openai_client = OpenAI(api_key=settings.toup_token, base_url=base)
                logger.info("OpenAI embedding client via platform proxy: %s", self.model_name)
            else:
                self._openai_client = OpenAI(api_key=settings.openai_api_key)
                logger.info(f"OpenAI embedding client initialized: {self.model_name}")
        return self._openai_client

    @property
    def local_model(self):
        if self._model is None and not self.is_openai:
            from sentence_transformers import SentenceTransformer
            name = self.model_name
            logger.info(f"Loading local embedding model: {name}")
            self._model = SentenceTransformer(name)
            logger.info(f"Model loaded. Dimension: {self._model.get_sentence_embedding_dimension()}")
        return self._model

    def _get_openai_client(self, api_key: Optional[str] = None):
        """Get OpenAI client, using per-user key override if provided."""
        if api_key:
            from openai import OpenAI
            return OpenAI(api_key=api_key)
        return self.openai_client

    def embed(self, text: str, api_key: Optional[str] = None) -> List[float]:
        """Generate embedding for a single text (sync — blocks event loop for OpenAI).

        For async code paths (e.g., chat message handling), prefer embed_async() instead.
        """
        if self.is_openai:
            client = self._get_openai_client(api_key)
            response = client.embeddings.create(model=self.model_name, input=text)
            return response.data[0].embedding
        else:
            embedding = self.local_model.encode(text, convert_to_numpy=True)
            return embedding.tolist()

    async def embed_async(self, text: str, api_key: Optional[str] = None) -> List[float]:
        """Generate embedding without blocking the event loop.

        - Local models: runs in a thread executor (~5ms, avoids GIL contention)
        - OpenAI: runs in a thread executor to avoid blocking on sync HTTP call

        Use this in any async code path where latency matters (chat, search).
        """
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.embed, text, api_key)

    def embed_batch(self, texts: List[str], api_key: Optional[str] = None) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []
        if self.is_openai:
            client = self._get_openai_client(api_key)
            response = client.embeddings.create(model=self.model_name, input=texts)
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


# ── Degrade observability ────────────────────────────────────────────
# Memory writes degrade to UNEMBEDDED rows when the vector cannot be
# computed (memory_service.create_memory / memory_dedup_service.
# _embed_or_none). The per-write signal is deliberately a quiet WARNING —
# a memory write must never fail because of the vector — but the degrade
# also swallows deterministic failures (proxy 429 monthly-budget, 401/403
# token problems), which can silently strip vectors from EVERY write for
# the rest of a billing month. This counter makes that countable:
# /agent/health surfaces it so fleet-wide unembedded writes are observable
# in one poll instead of buried in container logs. `last_error` carries the
# exception type, so auth/budget failures (RateLimitError/
# AuthenticationError/PermissionDeniedError) are distinguishable from the
# expected slim-image ImportError. Recovery for accumulated NULL-embedding
# rows = `python -m app.scripts.backfill_embeddings` (re-embeds them).
_embed_degrade_stats: Dict[str, Any] = {
    # Rows actually stored with embedding=None (create_memory's degrade —
    # the only site that persists an unembedded row, so this is an exact
    # row count, not an attempt count).
    "degraded_writes": 0,
    # Times the dedup pre-embed failed (dedup skipped for that write; the
    # row itself is still counted above iff create_memory's retry also
    # failed).
    "dedup_skips": 0,
    "last_error": None,
    "last_at": None,
}


def record_embed_degrade(error: BaseException, *, site: str = "write") -> None:
    """Count one embed degrade. site="write" → a row was stored without a
    vector; site="dedup" → the dedup pre-embed failed (dedup skipped)."""
    key = "degraded_writes" if site == "write" else "dedup_skips"
    _embed_degrade_stats[key] += 1
    _embed_degrade_stats["last_error"] = f"{type(error).__name__}: {error}"[:300]
    _embed_degrade_stats["last_at"] = datetime.now(timezone.utc).isoformat()


def embed_degrade_stats() -> Dict[str, Any]:
    """Snapshot of the degrade counter for /agent/health."""
    return dict(_embed_degrade_stats)
