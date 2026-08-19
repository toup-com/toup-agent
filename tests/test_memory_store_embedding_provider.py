"""memory_store embedding-provider regression tests (canary 533354ce, 2026-07-28).

The memory_store tool on agent images errored with
`No module named 'sentence_transformers'` DESPITE EMBEDDING_PROVIDER=openai:
bundle-mode containers are keyless (no OPENAI_API_KEY — chat rides TOUP_TOKEN
through the platform proxy), and EmbeddingService.provider resolves "openai"
only with a raw key OR the proxy escape hatch (EMBEDDINGS_VIA_PROXY). With the
flag unset, provider silently fell back to "local" and the write path
(smart_create_memory → embed_async → local_model) hit the deliberately
unshipped sentence-transformers import. memory_create (the platform-side MCP
tool) kept working, masking the breakage.

This file locks in:

(a) the agent image pins EMBEDDINGS_VIA_PROXY=true next to
    EMBEDDING_PROVIDER=openai, so provider resolution actually lands on
    "openai" in keyless bundle containers
(b) with the proxy path active, memory_store's write path runs the REAL
    EmbeddingService end-to-end against a stubbed OpenAI client with no
    sentence_transformers import reachable — this test environment does not
    ship the package (CI's targeted pip list omits it on purpose), and
    sys.modules is poisoned belt-and-braces for dev machines that have it
(c) when the provider DOES resolve "local" and sentence-transformers is
    absent, the write path degrades gracefully — memory stored without a
    vector + a [MEMORY] warning — instead of erroring the tool
"""

from __future__ import annotations

import logging
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import pytest_asyncio

from app.schemas import BrainType, MemoryCreate, MemoryLevel, MemoryType

_BACKEND = Path(__file__).resolve().parent.parent


# ── fakes / helpers ──────────────────────────────────────────────────


def _column_dimension() -> int:
    """The memories.embedding Vector column bakes settings.embedding_dimension
    at import time — the fake must emit vectors of THAT dimension or the
    insert fails on a dim mismatch (which would test the fixture, not the fix)."""
    from app.db.models import Memory

    col = Memory.__table__.c.get("embedding")
    if col is not None:
        dim = getattr(col.type, "dim", None)
        if dim:
            return dim
    from app.config import settings

    return settings.embedding_dimension


class _FakeOpenAIEmbeddings:
    """Stub for the OpenAI SDK's `client.embeddings` namespace."""

    def __init__(self, dimension: int):
        self.calls: list[dict] = []
        self._dimension = dimension

    def create(self, model, input):
        self.calls.append({"model": model, "input": input})
        texts = input if isinstance(input, list) else [input]
        return SimpleNamespace(
            data=[
                SimpleNamespace(embedding=[0.01] * self._dimension, index=i)
                for i in range(len(texts))
            ]
        )


class _FakeOpenAIClient:
    def __init__(self, dimension: int):
        self.embeddings = _FakeOpenAIEmbeddings(dimension)


def _mem_create(content: str) -> MemoryCreate:
    return MemoryCreate(
        content=content,
        summary=content[:50],
        brain_type=BrainType.USER,
        category="preferences",
        memory_type=MemoryType.FACT,
        importance=0.5,
        confidence=0.9,
        memory_level=MemoryLevel.EPISODIC,
        source_type="agent_tool",
    )


async def _ensure_memory_tables():
    """memories/memory_events/brain_stats/entities are AGENT_ONLY — CI runs
    RUN_MODE=platform, so conftest's init_db excludes them. Create the tables
    the smart_create_memory path touches directly (checkfirst → no-op under
    monolith/agent runs)."""
    from app.db.database import engine
    from app.db.models import BrainStats, Entity, Memory, MemoryEvent, memory_relationships

    async with engine.begin() as conn:
        for table in (
            Memory.__table__,
            MemoryEvent.__table__,
            BrainStats.__table__,
            Entity.__table__,
            memory_relationships,
        ):
            await conn.run_sync(table.create, checkfirst=True)


# ── fixtures ─────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_embedding_singleton():
    """EmbeddingService caches provider/client/model as CLASS attributes and
    get_embedding_service is lru_cached — reset before AND after each test so
    this file neither inherits nor leaks resolved state (a stale
    _resolved_provider is exactly the failure mode under test)."""
    from app.services import embedding_service as _es
    from app.services.embedding_service import EmbeddingService, get_embedding_service

    def _reset():
        EmbeddingService._instance = None
        EmbeddingService._model = None
        EmbeddingService._openai_client = None
        EmbeddingService._resolved_provider = None
        get_embedding_service.cache_clear()
        _es._embed_degrade_stats.update(
            {"degraded_writes": 0, "dedup_skips": 0, "last_error": None, "last_at": None}
        )

    _reset()
    yield
    _reset()


@pytest_asyncio.fixture
async def memory_user():
    """Create a User row. Returns the id."""
    from app.db import User, async_session_maker

    user_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(
            id=user_id,
            email=f"mem-{user_id[:8]}@example.com",
            hashed_password="x",
            name="Memory Test",
        ))
        await db.commit()
    return user_id


def _keyless_bundle_settings(monkeypatch, *, via_proxy: bool):
    """The agent-container reality: bundle mode, TOUP_TOKEN set, NO raw
    OpenAI key. `via_proxy` is the Dockerfile.agent contract under test."""
    monkeypatch.setattr("app.services.embedding_service.settings.embedding_provider", "openai", raising=False)
    monkeypatch.setattr("app.services.embedding_service.settings.embedding_model", "text-embedding-3-small", raising=False)
    monkeypatch.setattr("app.services.embedding_service.settings.openai_api_key", "", raising=False)
    monkeypatch.setattr("app.services.embedding_service.settings.llm_mode", "bundle", raising=False)
    monkeypatch.setattr("app.services.embedding_service.settings.toup_token", "toup_tok_test", raising=False)
    monkeypatch.setattr("app.services.embedding_service.settings.embeddings_via_proxy", via_proxy, raising=False)


# ── (a) Dockerfile pin ───────────────────────────────────────────────


def test_agent_image_pins_embeddings_via_proxy():
    """EMBEDDING_PROVIDER=openai without the proxy flag resolves "local" in
    keyless bundle containers — the pair must ship together or memory_store
    regresses to the sentence_transformers ImportError."""
    df = (_BACKEND / "Dockerfile.agent").read_text()
    assert "EMBEDDING_PROVIDER=openai" in df
    assert "EMBEDDINGS_VIA_PROXY=true" in df


# ── (b) openai/proxy write path — no sentence_transformers reachable ─


@pytest.mark.asyncio
async def test_memory_store_write_path_uses_openai_embeddings(monkeypatch, memory_user):
    import app.services.memory_dedup_service as mds
    from app.services.embedding_service import EmbeddingService

    await _ensure_memory_tables()
    _keyless_bundle_settings(monkeypatch, via_proxy=True)

    # Stub the OpenAI client BEFORE first use — the openai_client property
    # only constructs a real SDK client when the cached slot is None.
    fake_client = _FakeOpenAIClient(_column_dimension())
    monkeypatch.setattr(EmbeddingService, "_openai_client", fake_client, raising=False)

    # Belt-and-braces for dev machines that DO have the package: a None
    # entry in sys.modules makes any `import sentence_transformers` raise.
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)

    # Dedup adjudication must never fire on an empty DB — keep the LLM inert.
    monkeypatch.setattr(mds, "get_llm_service", lambda: MagicMock())

    from app.db.database import async_session_maker
    async with async_session_maker() as db:
        dedup = mds.MemoryDedupService(db)
        assert dedup.embedding_service.provider == "openai"
        memory, action = await dedup.smart_create_memory(
            new_memory=_mem_create("The user's favorite espresso is a double ristretto"),
            user_id=memory_user,
        )

    assert action == "created"
    assert fake_client.embeddings.calls, "OpenAI embeddings client was never called"
    assert fake_client.embeddings.calls[0]["model"] == "text-embedding-3-small"
    assert memory.embedding_json is not None


# ── (c) local-mode absence degrades gracefully ───────────────────────


@pytest.mark.asyncio
async def test_memory_store_degrades_without_sentence_transformers(monkeypatch, memory_user, caplog):
    """Pre-fix container reality: EMBEDDING_PROVIDER=openai but keyless and
    proxy flag OFF → provider resolves "local" → sentence-transformers is not
    shipped. The write must degrade to an unembedded memory + [MEMORY]
    warning, not surface a tool ERROR."""
    import app.services.memory_dedup_service as mds

    await _ensure_memory_tables()
    _keyless_bundle_settings(monkeypatch, via_proxy=False)
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    monkeypatch.setattr(mds, "get_llm_service", lambda: MagicMock())

    from app.db.database import async_session_maker
    with caplog.at_level(logging.WARNING):
        async with async_session_maker() as db:
            dedup = mds.MemoryDedupService(db)
            assert dedup.embedding_service.provider == "local"
            memory, action = await dedup.smart_create_memory(
                new_memory=_mem_create("User plays tennis on Sundays"),
                user_id=memory_user,
            )

    assert action == "created"
    assert memory.embedding_json is None
    assert any("[MEMORY]" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_batch_write_path_degrades_without_sentence_transformers(monkeypatch, memory_user, caplog):
    """smart_create_memories (automatic extraction's storage path) shares the
    degrade: every memory in the batch lands unembedded instead of the whole
    extraction erroring."""
    import app.services.memory_dedup_service as mds

    await _ensure_memory_tables()
    _keyless_bundle_settings(monkeypatch, via_proxy=False)
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    monkeypatch.setattr(mds, "get_llm_service", lambda: MagicMock())

    from app.db.database import async_session_maker
    with caplog.at_level(logging.WARNING):
        async with async_session_maker() as db:
            dedup = mds.MemoryDedupService(db)
            results = await dedup.smart_create_memories(
                new_memories=[
                    _mem_create("User's dog is named Max"),
                    _mem_create("User prefers window seats on flights"),
                ],
                user_id=memory_user,
            )

    assert [action for _, action in results] == ["created", "created"]
    assert all(memory.embedding_json is None for memory, _ in results)


async def test_batch_write_path_survives_a_missing_memory_files_table(
    monkeypatch, memory_user, caplog
):
    """CI regression (2026-08-19): under RUN_MODE=platform the memories table
    exists but the AGENT_ONLY memory_files table does not, so file routing
    fails for every item. Routing runs on its own session and latches off
    after the first failure — the degrade must cost ONLY the file stamp.
    The original defect rolled back the SHARED session instead, expiring the
    batch's earlier committed instances; the caller then read
    DetachedInstanceError off a memory it had just been handed."""
    import app.services.memory_dedup_service as mds

    await _ensure_memory_tables()
    _keyless_bundle_settings(monkeypatch, via_proxy=False)
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    monkeypatch.setattr(mds, "get_llm_service", lambda: MagicMock())

    from sqlalchemy import text as sql_text

    from app.db.database import async_session_maker, engine

    async with engine.begin() as conn:
        await conn.execute(sql_text("DROP TABLE IF EXISTS memory_files"))

    with caplog.at_level(logging.WARNING):
        async with async_session_maker() as db:
            dedup = mds.MemoryDedupService(db)
            results = await dedup.smart_create_memories(
                new_memories=[
                    _mem_create("User's cat is named Vesper"),
                    _mem_create("User is learning Spanish on Tuesdays"),
                ],
                user_id=memory_user,
            )

    # The reads below run AFTER the session closed — the exact CI symptom.
    assert [action for _, action in results] == ["created", "created"]
    for memory, _ in results:
        assert memory.content
        assert memory.embedding_json is None
        assert memory.file_slug is None  # stored unfiled, not dropped

    unavailable = [
        r for r in caplog.records if "routing unavailable" in r.getMessage()
    ]
    assert len(unavailable) == 1, (
        "routing must latch off after the first failure, "
        f"got {len(unavailable)} warnings"
    )


# ── (d) bind must re-resolve the provider cached during LOBBY boot ───


def test_bind_reset_reresolves_lobby_cached_provider(monkeypatch):
    """Pool spares and blue-green recreates boot in LOBBY mode: no TOUP_TOKEN,
    llm_mode defaults "manual", and agent_main's boot pre-load touches
    `provider` at ~25% boot — resolving and caching "local" on the singleton
    even with EMBEDDINGS_VIA_PROXY=true baked into the image
    (_proxy_embeddings_active needs bundle+token too). /admin/bind can only
    fire after lobby health reports ready, so it ALWAYS loses that race; it
    mutates llm_mode/toup_token on the live settings but the cache would
    otherwise pin "local" for the whole process lifetime.
    reset_provider_cache() (called by /admin/bind and tunnel
    _reload_settings) must make held references re-resolve to "openai"."""
    from app.services.embedding_service import EmbeddingService, get_embedding_service

    # Lobby-boot reality: image pins provider=openai + via_proxy, but the
    # container is keyless AND identity-less (manual mode, no token).
    monkeypatch.setattr("app.services.embedding_service.settings.embedding_provider", "openai", raising=False)
    monkeypatch.setattr("app.services.embedding_service.settings.embedding_model", "text-embedding-3-small", raising=False)
    monkeypatch.setattr("app.services.embedding_service.settings.openai_api_key", "", raising=False)
    monkeypatch.setattr("app.services.embedding_service.settings.llm_mode", "manual", raising=False)
    monkeypatch.setattr("app.services.embedding_service.settings.toup_token", "", raising=False)
    monkeypatch.setattr("app.services.embedding_service.settings.embeddings_via_proxy", True, raising=False)

    svc = get_embedding_service()  # held reference, like MemoryService.embedding_service
    assert svc.provider == "local"  # boot pre-load resolution

    # /admin/bind: runtime_identity.apply_to_settings flips these in place.
    monkeypatch.setattr("app.services.embedding_service.settings.llm_mode", "bundle", raising=False)
    monkeypatch.setattr("app.services.embedding_service.settings.toup_token", "toup_tok_test", raising=False)

    # The trap under test: settings mutation alone does NOT re-resolve —
    # the property cached its answer on the singleton's instance dict.
    assert svc.provider == "local"

    EmbeddingService.reset_provider_cache()

    # Held references AND fresh lookups now land on the proxy client path.
    assert svc.provider == "openai"
    assert get_embedding_service().provider == "openai"


def test_reset_provider_cache_clears_instance_shadowing_attrs():
    """The provider/openai_client/local_model properties assign `self._…`,
    shadowing the class attributes on the singleton's __dict__ — a reset that
    only nulls the class slots would be a no-op. Guard the instance-dict
    clearing explicitly."""
    from app.services.embedding_service import EmbeddingService, get_embedding_service

    svc = get_embedding_service()
    svc._resolved_provider = "local"
    svc._openai_client = object()
    assert "_resolved_provider" in svc.__dict__

    EmbeddingService.reset_provider_cache()

    assert "_resolved_provider" not in svc.__dict__
    assert "_openai_client" not in svc.__dict__
    assert "_model" not in svc.__dict__
    assert EmbeddingService._resolved_provider is None


def test_bind_and_reload_settings_call_reset_provider_cache():
    """Source pins for the two runtime settings-change sites: /admin/bind
    (pool claim / blue-green recreate) and tunnel_client._reload_settings
    (config_update). Either one silently dropping the reset re-opens the
    forever-local defect."""
    bind_src = (_BACKEND / "app" / "api" / "admin_pool.py").read_text()
    assert "reset_provider_cache" in bind_src

    tunnel_src = (_BACKEND / "app" / "agent" / "tunnel_client.py").read_text()
    assert "reset_provider_cache" in tunnel_src


# ── (e) degraded writes are countable, not just log-grep-able ────────


@pytest.mark.asyncio
async def test_degraded_writes_increment_health_counter(monkeypatch, memory_user):
    """The graceful degrade swallows deterministic failures too (proxy 429
    monthly-budget, 401/403) — every write for the rest of the month lands
    unembedded with only a log WARNING. The degrade must therefore bump a
    countable, structured counter that /agent/health surfaces."""
    import app.services.memory_dedup_service as mds
    from app.services.embedding_service import embed_degrade_stats

    await _ensure_memory_tables()
    _keyless_bundle_settings(monkeypatch, via_proxy=False)
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    monkeypatch.setattr(mds, "get_llm_service", lambda: MagicMock())

    assert embed_degrade_stats()["degraded_writes"] == 0

    from app.db.database import async_session_maker
    async with async_session_maker() as db:
        dedup = mds.MemoryDedupService(db)
        memory, action = await dedup.smart_create_memory(
            new_memory=_mem_create("User is allergic to shellfish"),
            user_id=memory_user,
        )

    assert action == "created"
    assert memory.embedding_json is None

    stats = embed_degrade_stats()
    # Exactly one row was stored without a vector...
    assert stats["degraded_writes"] == 1
    # ...and the dedup pre-embed degrade was counted at its own key.
    assert stats["dedup_skips"] == 1
    # last_error carries the exception type/message so auth/budget failures
    # (RateLimitError etc.) are distinguishable from the slim-image ImportError.
    assert stats["last_error"] and "sentence_transformers" in stats["last_error"]
    assert stats["last_at"] is not None


def test_agent_health_surfaces_embed_degrade_stats():
    """/agent/health must expose the embeddings block — provider + degrade
    counters — so fleet-wide unembedded writes are observable in one poll."""
    src = (_BACKEND / "agent_main.py").read_text()
    assert "embed_degrade_stats" in src
    assert '"embeddings"' in src
