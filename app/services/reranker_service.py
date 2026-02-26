"""
Cross-Encoder Re-ranker Service (Phase 6).

Improves retrieval precision by re-scoring (query, document) pairs
with a cross-encoder model after initial bi-encoder (embedding) retrieval.

Pipeline:
  bi-encoder (fast, top-N) → RRF fusion → cross-encoder (accurate, top-K)

Supported backends:
  1. Cohere Rerank API (primary) — cheapest, purpose-built, ~$0.0001/query
  2. GPT-4o-mini scoring (fallback) — available if OpenAI key is set
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cohere Rerank via raw HTTP (avoids heavy cohere SDK dependency)
# ---------------------------------------------------------------------------
COHERE_RERANK_URL = "https://api.cohere.com/v2/rerank"
COHERE_DEFAULT_MODEL = "rerank-v3.5"


class RerankerService:
    """
    Cross-encoder re-ranker for improving retrieval precision.

    Bi-encoders (embedding search) are fast but approximate — they encode
    query and document *independently*.  Cross-encoders process the (query,
    document) pair *together*, enabling much higher accuracy but at the cost
    of speed (O(N) inference vs O(1) lookup).

    Typical usage:
        reranker = get_reranker_service()
        reranked = await reranker.rerank(query, candidates, top_k=15)
    """

    def __init__(
        self,
        cohere_api_key: Optional[str] = None,
        cohere_model: str = COHERE_DEFAULT_MODEL,
        openai_api_key: Optional[str] = None,
        timeout: float = 10.0,
    ):
        self.cohere_api_key = cohere_api_key
        self.cohere_model = cohere_model
        self.openai_api_key = openai_api_key
        self.timeout = timeout
        self._http: Optional[httpx.AsyncClient] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 15,
    ) -> List[Dict[str, Any]]:
        """
        Re-rank *candidates* for *query* and return the top-K results.

        Each candidate dict MUST have at least an ``"id"`` and ``"content"``
        key.  The returned list preserves the original dict structure with
        an added ``"rerank_score"`` field.

        Falls back gracefully:
          Cohere → GPT-4o-mini → passthrough (no re-ranking)
        """
        if not candidates:
            return []

        if len(candidates) <= top_k:
            # Nothing to prune — skip the API call
            for c in candidates:
                c["rerank_score"] = c.get("final_score", 0.0)
            return candidates

        t0 = time.monotonic()

        # --- Try Cohere first ---
        if self.cohere_api_key:
            try:
                result = await self._cohere_rerank(query, candidates, top_k)
                elapsed = (time.monotonic() - t0) * 1000
                logger.info(
                    f"[RERANKER] Cohere re-ranked {len(candidates)}→{len(result)} "
                    f"in {elapsed:.0f}ms"
                )
                return result
            except Exception as exc:
                logger.warning(f"[RERANKER] Cohere failed, falling back: {exc}")

        # --- Fallback: GPT-4o-mini scoring ---
        if self.openai_api_key:
            try:
                result = await self._llm_rerank(query, candidates, top_k)
                elapsed = (time.monotonic() - t0) * 1000
                logger.info(
                    f"[RERANKER] LLM re-ranked {len(candidates)}→{len(result)} "
                    f"in {elapsed:.0f}ms"
                )
                return result
            except Exception as exc:
                logger.warning(f"[RERANKER] LLM fallback failed: {exc}")

        # --- Final fallback: passthrough ---
        logger.info("[RERANKER] No re-ranker available, returning top-K by score")
        for c in candidates:
            c["rerank_score"] = c.get("final_score", 0.0)
        return candidates[:top_k]

    # ------------------------------------------------------------------
    # Cohere Rerank v2 API
    # ------------------------------------------------------------------

    async def _cohere_rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Call Cohere Rerank v2 API and return re-ranked candidates."""
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=self.timeout)

        documents = []
        for c in candidates:
            # Build a compact text representation for re-ranking
            content = c.get("content", "")
            summary = c.get("summary", "")
            category = c.get("category", "")
            text = content
            if summary and summary != content:
                text = f"[{category}] {summary}\n{content}"
            elif category:
                text = f"[{category}] {content}"
            documents.append(text)

        payload = {
            "model": self.cohere_model,
            "query": query,
            "documents": documents,
            "top_n": top_k,
            "return_documents": False,
        }

        resp = await self._http.post(
            COHERE_RERANK_URL,
            json=payload,
            headers={
                "Authorization": f"Bearer {self.cohere_api_key}",
                "Content-Type": "application/json",
            },
        )
        resp.raise_for_status()
        data = resp.json()

        # data["results"] = [{"index": 3, "relevance_score": 0.98}, ...]
        reranked: List[Dict[str, Any]] = []
        for item in data.get("results", []):
            idx = item["index"]
            score = item["relevance_score"]
            candidate = candidates[idx].copy()
            candidate["rerank_score"] = score
            reranked.append(candidate)

        # Already sorted by Cohere, but ensure order
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]

    # ------------------------------------------------------------------
    # GPT-4o-mini fallback re-ranker
    # ------------------------------------------------------------------

    async def _llm_rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        Score each (query, document) pair with GPT-4o-mini.

        We batch all candidates into a single prompt asking for relevance
        scores 0-10 for each, keeping costs low (~$0.0005 per call).
        """
        from app.services.llm_service import get_llm_service

        llm = get_llm_service()

        # Build numbered document list
        doc_lines = []
        for i, c in enumerate(candidates):
            content = c.get("content", "")[:300]  # Truncate for token budget
            category = c.get("category", "")
            doc_lines.append(f"[{i}] ({category}) {content}")

        docs_text = "\n".join(doc_lines)

        prompt = f"""Score how relevant each document is to the query.
Return a JSON object with "scores": a list of objects with "index" (int) and "score" (float 0-10, 10=perfectly relevant, 0=irrelevant).

Query: "{query}"

Documents:
{docs_text}

Return ONLY valid JSON: {{"scores": [{{"index": 0, "score": 7.5}}, ...]}}"""

        try:
            response = await llm.complete_with_json(
                messages=[{"role": "user", "content": prompt}],
                model="gpt-4o-mini",
                temperature=0.0,
                max_tokens=2000,
            )

            result = json.loads(response.content)
            score_map: Dict[int, float] = {}
            for entry in result.get("scores", []):
                idx = int(entry.get("index", -1))
                sc = float(entry.get("score", 0.0))
                if 0 <= idx < len(candidates):
                    score_map[idx] = sc / 10.0  # Normalize to 0-1

            # Apply scores
            scored: List[Dict[str, Any]] = []
            for i, c in enumerate(candidates):
                cand = c.copy()
                cand["rerank_score"] = score_map.get(i, 0.0)
                scored.append(cand)

            scored.sort(key=lambda x: x["rerank_score"], reverse=True)
            return scored[:top_k]

        except Exception as exc:
            logger.warning(f"[RERANKER] LLM scoring parse error: {exc}")
            raise

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def close(self):
        """Close the HTTP client."""
        if self._http:
            await self._http.aclose()
            self._http = None


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_reranker_service: Optional[RerankerService] = None


def get_reranker_service() -> RerankerService:
    """Get or create the reranker service singleton."""
    global _reranker_service
    if _reranker_service is None:
        from app.config import settings

        _reranker_service = RerankerService(
            cohere_api_key=getattr(settings, "cohere_api_key", None),
            cohere_model=getattr(settings, "reranker_model", COHERE_DEFAULT_MODEL),
            openai_api_key=settings.openai_api_key,
        )
    return _reranker_service
