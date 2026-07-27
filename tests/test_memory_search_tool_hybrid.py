"""
W1.5 — memory_search tool routes through hybrid_search.

The agent's explicit memory_search tool used to call
search_memories_by_embedding (pure dense cosine — no keyword/graph arms,
no strength floor). It now goes through MemoryService.hybrid_search, the
same RRF engine auto-recall uses every turn. These tests pin:

  1. the tool calls hybrid_search with the right args (filters preserved),
  2. the rendered output keeps the ``id=`` line memory_delete targets (A6-6),
  3. hybrid_search itself degrades gracefully when the query embedding
     fails — keyword/graph results still come back (the embed call happens
     BEFORE the per-strategy gather isolation, so it needs its own guard).
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from app.agent.tool_executor import ToolExecutor


def _mem(mem_id: str, content: str, sim: float = 0.85, category: str = "knowledge") -> dict:
    """A result dict in the shape hybrid_search returns."""
    return {
        "id": mem_id,
        "content": content,
        "summary": content[:50],
        "brain_type": "user",
        "category": category,
        "memory_type": "fact",
        "importance": 0.5,
        "confidence": 0.8,
        "strength": 0.9,
        "created_at": "2026-07-01T00:00:00",
        "updated_at": "2026-07-01T00:00:00",
        "last_accessed_at": None,
        "access_count": 1,
        "source_type": "conversation",
        "similarity_score": sim,
        "final_score": sim,
        "retrieval_strategies": ["vector", "keyword", "graph"],
    }


@pytest.fixture
def executor(tmp_path):
    return ToolExecutor(workspace=str(tmp_path))


# ----------------------------------------------------------------
# 1. Tool → hybrid_search wiring
# ----------------------------------------------------------------

class TestMemorySearchToolHybrid:

    async def test_routes_through_hybrid_search(self, executor):
        executor.set_user_id("test_user")
        hybrid = AsyncMock(return_value=[
            _mem("mem_1", "Alice runs the design team"),
            _mem("mem_2", "Alice prefers async standups", sim=0.72),
        ])
        with patch("app.services.memory_service.MemoryService.hybrid_search", hybrid):
            out = await executor._tool_memory_search(
                {"query": "what do you remember about Alice", "limit": 3}
            )

        hybrid.assert_awaited_once()
        kwargs = hybrid.await_args.kwargs
        assert kwargs["user_id"] == "test_user"
        assert kwargs["query"] == "what do you remember about Alice"
        assert kwargs["limit"] == 3
        assert kwargs["min_similarity"] == 0.1
        assert kwargs["brain_types"] is None
        assert kwargs["strategies"] == ["vector", "keyword", "graph"]

        assert "Alice runs the design team" in out
        assert "Alice prefers async standups" in out

    async def test_brain_type_filter_preserved(self, executor):
        executor.set_user_id("test_user")
        hybrid = AsyncMock(return_value=[])
        with patch("app.services.memory_service.MemoryService.hybrid_search", hybrid):
            await executor._tool_memory_search(
                {"query": "project notes", "brain_type": "business"}
            )
        assert hybrid.await_args.kwargs["brain_types"] == ["business"]

    async def test_default_limit_is_five(self, executor):
        executor.set_user_id("test_user")
        hybrid = AsyncMock(return_value=[])
        with patch("app.services.memory_service.MemoryService.hybrid_search", hybrid):
            await executor._tool_memory_search({"query": "anything"})
        assert hybrid.await_args.kwargs["limit"] == 5

    async def test_id_line_intact_for_memory_delete(self, executor):
        """memory_delete targets results by the id= token in the rendering (A6-6)."""
        executor.set_user_id("test_user")
        hybrid = AsyncMock(return_value=[_mem("mem_abc123", "Bob's birthday is in May", sim=0.87)])
        with patch("app.services.memory_service.MemoryService.hybrid_search", hybrid):
            out = await executor._tool_memory_search({"query": "Bob birthday"})
        assert "id=mem_abc123" in out
        assert out.startswith("1. [knowledge] (sim=0.87, id=mem_abc123) ")

    async def test_empty_results(self, executor):
        executor.set_user_id("test_user")
        hybrid = AsyncMock(return_value=[])
        with patch("app.services.memory_service.MemoryService.hybrid_search", hybrid):
            out = await executor._tool_memory_search({"query": "nothing stored"})
        assert out == "No memories found."

    async def test_missing_query_errors(self, executor):
        out = await executor._tool_memory_search({})
        assert out == "ERROR: query is required"

    async def test_service_failure_returns_error_string(self, executor):
        executor.set_user_id("test_user")
        hybrid = AsyncMock(side_effect=RuntimeError("db unavailable"))
        with patch("app.services.memory_service.MemoryService.hybrid_search", hybrid):
            out = await executor._tool_memory_search({"query": "anything"})
        assert out.startswith("ERROR:")


# ----------------------------------------------------------------
# 2. hybrid_search vector-arm isolation (embedding failure)
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
