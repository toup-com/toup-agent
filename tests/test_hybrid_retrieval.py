"""
Tests for the Hybrid Retrieval Engine — RRF fusion, entity extraction, strategies.

These tests validate the core logic WITHOUT needing a database connection.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime


# ----------------------------------------------------------------
# 1. Reciprocal Rank Fusion (RRF)
# ----------------------------------------------------------------

class TestReciprocalRankFusion:
    """Test the static RRF method."""

    def _get_rrf(self):
        from app.services.memory_service import MemoryService
        return MemoryService.reciprocal_rank_fusion

    def test_single_list(self):
        rrf = self._get_rrf()
        ranked = [("a", 0.9), ("b", 0.7), ("c", 0.5)]
        result = rrf([ranked], k=60)
        # First item should have highest score: 1/(60+1)
        assert result[0][0] == "a"
        assert result[1][0] == "b"
        assert result[2][0] == "c"

    def test_two_lists_same_order(self):
        rrf = self._get_rrf()
        list1 = [("a", 0.9), ("b", 0.7), ("c", 0.5)]
        list2 = [("a", 0.8), ("b", 0.6), ("c", 0.4)]
        result = rrf([list1, list2], k=60)
        # 'a' appears rank 1 in both → highest RRF score
        assert result[0][0] == "a"
        assert result[1][0] == "b"
        assert result[2][0] == "c"

    def test_two_lists_different_order(self):
        rrf = self._get_rrf()
        list1 = [("a", 0.9), ("b", 0.7)]
        list2 = [("b", 0.9), ("c", 0.7)]
        result = rrf([list1, list2], k=60)
        ids = [doc_id for doc_id, _ in result]
        # 'b' appears in both lists (rank 2 + rank 1) → should have highest score
        assert ids[0] == "b"
        # 'a' rank 1 in list1 only, 'c' rank 2 in list2 only → same score
        assert set(ids[1:]) == {"a", "c"}

    def test_empty_lists(self):
        rrf = self._get_rrf()
        result = rrf([], k=60)
        assert result == []

    def test_disjoint_lists(self):
        rrf = self._get_rrf()
        list1 = [("a", 0.9)]
        list2 = [("b", 0.9)]
        list3 = [("c", 0.9)]
        result = rrf([list1, list2, list3], k=60)
        # All have same RRF score (1/(60+1) each)
        assert len(result) == 3
        scores = [s for _, s in result]
        assert scores[0] == scores[1] == scores[2]

    def test_k_parameter(self):
        rrf = self._get_rrf()
        ranked = [("a", 0.9), ("b", 0.7)]
        result_k1 = rrf([ranked], k=1)
        result_k60 = rrf([ranked], k=60)
        # With k=1: rank 1 score = 1/(1+1)=0.5, rank 2 = 1/(1+2)=0.333
        # With k=60: rank 1 score = 1/61≈0.0164, rank 2 = 1/62≈0.0161
        # Higher k → less difference between ranks
        diff_k1 = result_k1[0][1] - result_k1[1][1]
        diff_k60 = result_k60[0][1] - result_k60[1][1]
        assert diff_k1 > diff_k60


# ----------------------------------------------------------------
# 2. Entity Name Extraction
# ----------------------------------------------------------------

class TestEntityNameExtraction:
    """Test the _extract_entity_names static method."""

    def _extract(self, query: str):
        from app.services.memory_service import MemoryService
        return MemoryService._extract_entity_names(query)

    def test_capitalized_names(self):
        names = self._extract("Tell me about Alice and Bob")
        assert "Alice" in names
        assert "Bob" in names

    def test_multi_word_names(self):
        names = self._extract("What do you know about John Smith?")
        assert "John Smith" in names

    def test_quoted_strings(self):
        names = self._extract('Find memories about "Project Apollo"')
        assert "Project Apollo" in names

    def test_single_quoted_strings(self):
        names = self._extract("Tell me about 'Google Cloud'")
        assert "Google Cloud" in names

    def test_skips_common_words(self):
        names = self._extract("What is the weather today?")
        # "What" should be filtered out
        assert "What" not in names

    def test_empty_query(self):
        names = self._extract("")
        assert names == []

    def test_no_entities(self):
        names = self._extract("how are you doing today?")
        assert len(names) == 0

    def test_mixed_case(self):
        names = self._extract("I talked to Maria about Python")
        assert "Maria" in names
        assert "Python" in names

    def test_first_word_entity(self):
        names = self._extract("Nariman likes coffee")
        assert "Nariman" in names

    def test_organization_names(self):
        names = self._extract("Does anyone work at Google or Microsoft?")
        assert "Google" in names
        assert "Microsoft" in names


# ----------------------------------------------------------------
# 3. Hybrid Search Integration (mocked DB)
# ----------------------------------------------------------------

class TestHybridSearchIntegration:
    """Test hybrid_search orchestration with mocked strategies."""

    @pytest.mark.asyncio
    async def test_hybrid_search_combines_strategies(self):
        """Verify RRF fusion happens when multiple strategies return results."""
        from app.services.memory_service import MemoryService

        # Create a mock db session
        mock_db = AsyncMock()
        service = MemoryService(mock_db)

        # Mock the embedding service
        service.embedding_service = MagicMock()
        service.embedding_service.embed.return_value = [0.1] * 1536

        # Mock individual strategies
        with patch.object(service, '_vector_search', return_value=[
            ("mem_1", 0.9), ("mem_2", 0.7), ("mem_3", 0.5),
        ]) as mock_vec, \
        patch.object(service, '_keyword_search', return_value=[
            ("mem_2", 0.8), ("mem_4", 0.6),
        ]) as mock_kw, \
        patch.object(service, '_graph_search', return_value=[
            ("mem_3", 1.0), ("mem_5", 0.5),
        ]) as mock_graph:

            # Mock the DB fetch for the actual Memory objects
            mock_memories = []
            for i in range(1, 6):
                m = MagicMock()
                m.id = f"mem_{i}"
                m.content = f"Memory content {i}"
                m.summary = f"Summary {i}"
                m.brain_type = "user"
                m.category = "knowledge"
                m.memory_type = "fact"
                m.importance = 0.5
                m.confidence = 0.8
                m.strength = 0.9
                m.emotional_salience = 0.3
                m.created_at = datetime(2026, 2, 10)
                m.updated_at = datetime(2026, 2, 10)
                m.last_accessed_at = None
                m.access_count = 1
                m.source_type = "conversation"
                m.embedding = [0.1] * 1536
                mock_memories.append(m)

            # Mock the db.execute for Memory fetch
            mock_result = MagicMock()
            mock_scalars = MagicMock()
            mock_scalars.all.return_value = mock_memories
            mock_result.scalars.return_value = mock_scalars
            
            # Mock similarity query
            mock_sim_result = MagicMock()
            sim_rows = [MagicMock(id=f"mem_{i}", sim=0.8 - i * 0.1) for i in range(1, 6)]
            mock_sim_result.all.return_value = sim_rows

            mock_db.execute = AsyncMock(side_effect=[mock_result, mock_sim_result])

            results = await service.hybrid_search(
                user_id="test_user",
                query="test query about Alice",
                limit=5,
                strategies=["vector", "keyword", "graph"],
            )

            # All three strategies should have been called
            mock_vec.assert_called_once()
            mock_kw.assert_called_once()
            mock_graph.assert_called_once()

            # mem_2 and mem_3 appear in 2 strategies each → should rank higher
            # (exact ordering depends on scoring weights)
            assert len(results) <= 5
            result_ids = [r["id"] for r in results]
            # mem_2 is in vector + keyword → boosted by RRF
            assert "mem_2" in result_ids
            # mem_3 is in vector + graph → boosted by RRF
            assert "mem_3" in result_ids


# ----------------------------------------------------------------
# 4. Schema Validation
# ----------------------------------------------------------------

class TestSchemaUpdates:
    """Test that the MemorySearchRequest schema has the strategies field."""

    def test_strategies_field_exists(self):
        from app.schemas import MemorySearchRequest
        req = MemorySearchRequest(query="test", strategies=["vector", "keyword"])
        assert req.strategies == ["vector", "keyword"]

    def test_strategies_default_none(self):
        from app.schemas import MemorySearchRequest
        req = MemorySearchRequest(query="test")
        assert req.strategies is None

    def test_strategies_empty_list(self):
        from app.schemas import MemorySearchRequest
        req = MemorySearchRequest(query="test", strategies=[])
        assert req.strategies == []


# ----------------------------------------------------------------
# 5. Model Validation
# ----------------------------------------------------------------

class TestEntityRelationshipModel:
    """Test that the EntityRelationship model is properly defined."""

    def test_model_exists(self):
        from app.db.models import EntityRelationship
        assert EntityRelationship.__tablename__ == "entity_relationships"

    def test_model_columns(self):
        from app.db.models import EntityRelationship
        columns = {c.name for c in EntityRelationship.__table__.columns}
        expected = {
            "id", "user_id", "source_entity_id", "target_entity_id",
            "relationship_type", "relationship_label", "properties_json",
            "confidence", "mention_count", "first_seen_at", "last_seen_at",
            "created_at", "updated_at",
        }
        assert expected.issubset(columns)

    def test_relationship_label_column(self):
        from app.db.models import EntityRelationship
        columns = {c.name for c in EntityRelationship.__table__.columns}
        assert "relationship_label" in columns


# ----------------------------------------------------------------
# 6. Entity name_search tsvector column
# ----------------------------------------------------------------

class TestEntityNameSearch:
    """Test that Entity model has name_search tsvector column."""

    def test_name_search_column_exists(self):
        from app.db.models import Entity
        # name_search is a Column (not mapped_column), so check __table__
        columns = {c.name for c in Entity.__table__.columns}
        assert "name_search" in columns


# ----------------------------------------------------------------
# 7. Graph API schemas
# ----------------------------------------------------------------

class TestGraphSchemas:
    """Test that graph-related Pydantic schemas are properly defined."""

    def test_entity_brief_response(self):
        from app.schemas import EntityBriefResponse
        e = EntityBriefResponse(id="1", name="Alice", entity_type="person")
        assert e.name == "Alice"

    def test_entity_relationship_response(self):
        from app.schemas import EntityRelationshipResponse
        r = EntityRelationshipResponse(
            id="1",
            relationship_type="works_at",
            relationship_label="Alice works at Google",
            confidence=0.9,
            mention_count=3,
        )
        assert r.relationship_type == "works_at"
        assert r.relationship_label == "Alice works at Google"

    def test_graph_traversal_request(self):
        from app.schemas import GraphTraversalRequest
        req = GraphTraversalRequest(entity_names=["Alice"], max_depth=2)
        assert req.max_depth == 2
        assert req.entity_names == ["Alice"]

    def test_graph_traversal_request_ids(self):
        from app.schemas import GraphTraversalRequest
        req = GraphTraversalRequest(entity_ids=["abc-123"], max_depth=3, limit=20)
        assert req.entity_ids == ["abc-123"]
        assert req.limit == 20

    def test_graph_traversal_response(self):
        from app.schemas import GraphTraversalResponse, EntityBriefResponse, GraphTraversalNode
        resp = GraphTraversalResponse(
            seed_entities=[EntityBriefResponse(id="1", name="Alice", entity_type="person")],
            nodes=[GraphTraversalNode(
                entity_id="2", entity_name="Google", entity_type="organization",
                depth=1, relationship_type="works_at",
            )],
            total_entities=2,
            total_relationships=1,
        )
        assert resp.total_entities == 2
        assert resp.nodes[0].depth == 1

    def test_graph_exploration_response(self):
        from app.schemas import GraphExplorationResponse
        resp = GraphExplorationResponse(
            entities=[], relationships=[],
            total_entities=0, total_relationships=0,
        )
        assert resp.total_entities == 0


# ----------------------------------------------------------------
# 8. traverse_entity_graph (mocked)
# ----------------------------------------------------------------

class TestTraverseEntityGraph:
    """Test the traverse_entity_graph method with mocked DB."""

    @pytest.mark.asyncio
    async def test_traverse_returns_nodes(self):
        """traverse_entity_graph should return graph nodes with depth info."""
        from app.services.memory_service import MemoryService
        from unittest.mock import AsyncMock, MagicMock, PropertyMock

        db = AsyncMock()
        service = MemoryService.__new__(MemoryService)
        service.db = db
        service.embedding_service = MagicMock()

        # Mock the CTE query result
        mock_rows = [
            MagicMock(
                entity_id="e1", entity_name="Alice", entity_type="person",
                depth=0, relationship_type=None, relationship_label=None,
                from_entity_id=None, from_entity_name=None,
            ),
            MagicMock(
                entity_id="e2", entity_name="Google", entity_type="organization",
                depth=1, relationship_type="works_at",
                relationship_label="Alice works at Google",
                from_entity_id="e1", from_entity_name="Alice",
            ),
        ]
        mock_result = MagicMock()
        mock_result.fetchall.return_value = mock_rows
        db.execute.return_value = mock_result

        result = await service.traverse_entity_graph(
            user_id="user-1",
            seed_entity_ids=["e1"],
            max_depth=2,
        )

        assert len(result) == 2
        assert result[0]["entity_name"] == "Alice"
        assert result[0]["depth"] == 0
        assert result[1]["entity_name"] == "Google"
        assert result[1]["depth"] == 1
        assert result[1]["relationship_type"] == "works_at"


# ----------------------------------------------------------------
# 9. get_entity_relationships / get_entities (mocked)
# ----------------------------------------------------------------

class TestGraphQueries:
    """Test get_entity_relationships and get_entities service methods."""

    @pytest.mark.asyncio
    async def test_get_entities_returns_list(self):
        from app.services.memory_service import MemoryService
        from unittest.mock import AsyncMock, MagicMock
        from datetime import datetime

        db = AsyncMock()
        service = MemoryService.__new__(MemoryService)
        service.db = db
        service.embedding_service = MagicMock()

        mock_entity = MagicMock()
        mock_entity.id = "e1"
        mock_entity.name = "Alice"
        mock_entity.entity_type = "person"
        mock_entity.description = "A colleague"
        mock_entity.mention_count = 5
        mock_entity.first_seen_at = datetime(2026, 1, 1)
        mock_entity.last_seen_at = datetime(2026, 2, 1)
        mock_entity.attributes_json = None

        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [mock_entity]
        mock_result.scalars.return_value = mock_scalars
        db.execute.return_value = mock_result

        entities = await service.get_entities(user_id="user-1")

        assert len(entities) == 1
        assert entities[0]["name"] == "Alice"
        assert entities[0]["entity_type"] == "person"


class TestPromptBuilderBudgets:
    """Test that prompt builder token budgets are increased."""

    def test_increased_budgets(self):
        from app.services.prompt_builder import PromptBuilder
        pb = PromptBuilder()
        assert pb.MAX_SYSTEM_TOKENS >= 12000
        assert pb.MAX_MEMORY_TOKENS >= 6000
        assert pb.MAX_IDENTITY_TOKENS >= 4000
