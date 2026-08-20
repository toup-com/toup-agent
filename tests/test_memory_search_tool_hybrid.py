"""hybrid_search degrades gracefully — the engine, not the tool.

W1.5 routed the `memory_search` TOOL through hybrid_search so it shared
the RRF engine auto-recall used every turn. Memory v3 (§3.2) re-points the
tool at memory FILES, with hybrid_search surviving only as the
document/media leg (§3.4) — so the tool-wiring half of this file moved to
`tests/test_memory_tools_v3.py`, where the assertions are about files and
file attribution rather than row ids and similarity scores.

What stays here is the part that was never about the tool: hybrid_search
must still answer when the query embedding fails. The embed call happens
BEFORE the per-strategy gather isolation, so it needs its own guard —
without it, an embedding outage takes keyword and graph down with it, and
document recall goes silent rather than degraded.
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch



# ----------------------------------------------------------------
# hybrid_search vector-arm isolation (embedding failure)
# ----------------------------------------------------------------

class TestHybridSearchEmbeddingFailure:

    async def test_keyword_and_graph_survive_embedding_failure(self):
        """If embed_async raises, vector is skipped but keyword/graph still return."""
        from app.services.memory_service import MemoryService

        mock_db = AsyncMock()
        service = MemoryService(mock_db)
        service.embedding_service = MagicMock()
        service.embedding_service.embed_async = AsyncMock(
            side_effect=RuntimeError("embedding provider down")
        )

        mock_memories = []
        for mem_id in ("mem_2", "mem_3"):
            m = MagicMock()
            m.id = mem_id
            m.content = f"Content for {mem_id}"
            m.summary = f"Summary {mem_id}"
            m.brain_type = "user"
            m.category = "knowledge"
            m.memory_type = "fact"
            m.importance = 0.5
            m.confidence = 0.8
            m.strength = 0.9
            m.emotional_salience = 0.3
            m.created_at = datetime(2026, 7, 1)
            m.updated_at = datetime(2026, 7, 1)
            m.last_accessed_at = None
            m.access_count = 1
            m.source_type = "conversation"
            mock_memories.append(m)

        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = mock_memories
        mock_result.scalars.return_value = mock_scalars
        # Only the Memory fetch hits the DB — the similarity query is
        # skipped when there is no query embedding. (Later reinforcement
        # calls exhaust the side_effect; that path is try/except'd.)
        mock_db.execute = AsyncMock(side_effect=[mock_result])

        with patch.object(service, "_vector_search") as mock_vec, \
             patch.object(service, "_keyword_search", return_value=[("mem_2", 0.8)]), \
             patch.object(service, "_graph_search", return_value=[("mem_3", 1.0)]):
            results = await service.hybrid_search(
                user_id="test_user",
                query="notes about the redesign",
                limit=5,
                strategies=["vector", "keyword", "graph"],
            )

        mock_vec.assert_not_called()
        result_ids = {r["id"] for r in results}
        assert result_ids == {"mem_2", "mem_3"}
        # No embedding → similarity falls back to 0 for every result
        assert all(r["similarity_score"] == 0.0 for r in results)
