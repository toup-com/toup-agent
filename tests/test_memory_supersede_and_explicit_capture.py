"""Memory-defect fix unit (2026-07-29) — D-mem-A + D-mem-C + D-eval-mem pins.

Production evidence (canary 533354ce, memory-quality baseline 0.33/1.0,
docs/audits/2026-07-remediation.md):

D-mem-A  memory_store("...kestrel-dbf7") restated with "...kestrel-13b4"
         answered "Memory reinforced" — the dedup AUTO_DUPLICATE shortcut
         (similarity >= 0.90) reinforced the OLD row and discarded the
         conflicting new value unconditionally. Users could not change a
         stored fact. Fix: a cheap value-conflict guard on the shortcut
         routes conflicting pairs to the LLM adjudicator, whose
         contradiction_update verdict lands on the EXISTING supersede path
         (memories.superseded_by + is_active=False, #375 mechanism).
         Flag: settings.memory_supersede_on_conflict (default ON).

D-mem-C  realistic "Please remember: ..." phrasing captured only 3/8
         explicitly-requested facts — capture rides on background
         extraction, and token-like payloads ("memqa-...-lock-ece9") died
         in the extractor's code-snippet/length/importance noise filters.
         Fix: shared is_explicit_remember_request predicate feeding
         (1) a memory-intent boost in classify_query_intent (flag:
         settings.memory_tools_on_remember, default ON),
         (2) a trivial-turn-gate bypass in _extract_memories, and
         (3) an explicit_save_requested hint into
         extract_memories_with_llm that mandates verbatim capture and
         relaxes the noise filters for that call only.

D-eval-mem  behavioral_suite.py's memory scenario left its codeword rows
         behind after every run; the stale near-duplicates are what armed
         the D-mem-A collisions. The suite now sweeps `%kestrel-%` rows
         post-run via memory_quality_suite.cleanup_marker_rows.

Fixture style mirrors tests/test_extraction_fanout.py (scripted LLM +
fake embeddings, no network) and tests/test_memory_taxonomy_and_ttl.py
(local sqlite engine with just the ORM tables these paths touch).
"""

from __future__ import annotations

import importlib.util
import json
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import settings
from app.schemas import (
    BrainType,
    MemoryCategory,
    MemoryLevel,
    MemoryType,
    MemoryCreate,
    MemoryUpdate,
)
from app.services.memory_extractor import ExtractedMemory, MemoryExtractor
from app.services.query_classifier import is_explicit_remember_request


# ── shared fakes (mirrors test_extraction_fanout.py) ─────────────────


class _RecordingLLM:
    """complete_with_json fake that records every call."""

    def __init__(self, content: str = '{"action": "new", "reason": "x"}',
                 error: Exception | None = None):
        self.calls: list = []
        self._content = content
        self._error = error

    async def complete_with_json(self, messages, model=None, **kwargs):
        self.calls.append({"messages": messages, "model": model})
        if self._error:
            raise self._error
        return SimpleNamespace(content=self._content)


def _column_dimension() -> int:
    """memories.embedding bakes settings.embedding_dimension at import time —
    fake vectors must match it or the insert fails on a dim mismatch (which
    would test the fixture, not the fix)."""
    from app.db.models import Memory

    col = Memory.__table__.c.get("embedding")
    dim = getattr(col.type, "dim", None) if col is not None else None
    return int(dim) if dim else 1536


class _FakeEmbeddings:
    """Returns ONE fixed vector for every text — forces similarity 1.0
    between any two contents, which is exactly the D-mem-A arming condition
    (near-identical sentences differing only in a value token)."""

    def __init__(self, dim: int | None = None):
        self.vector = [0.1] * (dim or _column_dimension())
        self.async_calls = 0

    async def embed_async(self, text, api_key=None):
        self.async_calls += 1
        return list(self.vector)

    def embed(self, text, api_key=None):  # pragma: no cover - failure path
        raise AssertionError("sync embed() must not run on the async storage path (W1.4e)")

    @staticmethod
    def cosine_similarity(v1, v2):
        """Real cosine — the sqlite fallback search path scores through the
        embedding service, and identical fake vectors must read as 1.0."""
        dot = sum(a * b for a, b in zip(v1, v2))
        n1 = sum(a * a for a in v1) ** 0.5
        n2 = sum(b * b for b in v2) ** 0.5
        return dot / (n1 * n2) if n1 and n2 else 0.0


def _make_dedup(monkeypatch, llm=None, emb=None):
    import app.services.memory_dedup_service as mds

    llm = llm or _RecordingLLM()
    emb = emb or _FakeEmbeddings(dim=8)
    monkeypatch.setattr(mds, "get_llm_service", lambda: llm)
    monkeypatch.setattr(mds, "get_embedding_service", lambda: emb)
    dedup = mds.MemoryDedupService(db=MagicMock())
    dedup.memory_service = AsyncMock()
    return dedup, llm, emb


def _mem_create(content: str, importance: float = 0.7) -> MemoryCreate:
    return MemoryCreate(
        content=content,
        summary=content[:50],
        brain_type=BrainType.USER,
        category="possessions",
        memory_type=MemoryType.FACT,
        importance=importance,
        confidence=0.9,
        memory_level=MemoryLevel.EPISODIC,
        source_type="conversation",
    )


async def _memory_session():
    """Local sqlite session with just the tables these tests touch — same
    pattern (and reason) as test_memory_taxonomy_and_ttl.py: `memories` is
    AGENT_ONLY, so a platform-profile conftest init_db() never creates it."""
    from app.db.models import BrainStats, Entity, Memory, MemoryEvent, memory_relationships
    from app.db.models.base import Base
    from app.db.models.user import User as _U

    engine = create_async_engine(
        "sqlite+aiosqlite://", connect_args={"check_same_thread": False}
    )
    async with engine.begin() as conn:
        await conn.run_sync(
            Base.metadata.create_all,
            tables=[
                _U.__table__,
                Memory.__table__,
                MemoryEvent.__table__,
                # create_memory's _update_brain_stats / entity linking touch
                # these — same table set as test_memory_store_embedding_provider
                BrainStats.__table__,
                Entity.__table__,
                memory_relationships,
            ],
        )
    return engine, async_sessionmaker(engine, expire_on_commit=False)


# The exact production shape: same sentence, different value token.
_OLD_FACT = "My storage locker passphrase is kestrel-dbf7"
_NEW_FACT = "My storage locker passphrase is kestrel-13b4"


# ── D-mem-A (1): the value-conflict guard itself ─────────────────────


def test_conflicts_on_value_detects_the_kestrel_pair(monkeypatch):
    dedup, _, _ = _make_dedup(monkeypatch)
    assert dedup._conflicts_on_value(_OLD_FACT, _NEW_FACT) is True


def test_conflicts_on_value_ignores_identical_and_rewordings(monkeypatch):
    dedup, _, _ = _make_dedup(monkeypatch)
    # Verbatim restatement — true duplicate, must keep the cheap shortcut.
    assert dedup._conflicts_on_value(_NEW_FACT, _NEW_FACT) is False
    # Paraphrase with low token overlap — a rewording, not a value swap.
    assert dedup._conflicts_on_value(
        "I love pizza", "Pizza is my favorite food"
    ) is False
    # Strict superset — added detail, not a conflicting value.
    assert dedup._conflicts_on_value(
        "I play basketball", "I play basketball on Saturdays"
    ) is False
    # Disjoint texts — different facts entirely; the middle-band/LLM owns
    # those, the >=0.90 shortcut should not burn a call on them.
    assert dedup._conflicts_on_value("x", "y") is False


async def test_decide_action_conflict_at_high_similarity_consults_llm(monkeypatch):
    dedup, llm, _ = _make_dedup(
        monkeypatch,
        llm=_RecordingLLM(content='{"action": "contradiction_update", "reason": "value changed"}'),
    )
    decision = await dedup._decide_action(
        similarity=0.97, existing_content=_OLD_FACT, new_content=_NEW_FACT
    )
    assert decision["action"] == "contradiction_update"
    assert len(llm.calls) == 1  # adjudicated, not auto-"duplicate"


async def test_decide_action_true_duplicate_keeps_the_no_llm_shortcut(monkeypatch):
    dedup, llm, _ = _make_dedup(monkeypatch)
    decision = await dedup._decide_action(
        similarity=0.97, existing_content=_NEW_FACT, new_content=_NEW_FACT
    )
    assert decision["action"] == "duplicate"
    assert llm.calls == []  # reinforcement of a real restatement is unchanged


async def test_decide_action_kill_switch_restores_old_write_semantics(monkeypatch):
    dedup, llm, _ = _make_dedup(monkeypatch)
    monkeypatch.setattr(settings, "memory_supersede_on_conflict", False)
    decision = await dedup._decide_action(
        similarity=0.97, existing_content=_OLD_FACT, new_content=_NEW_FACT
    )
    assert decision["action"] == "duplicate"
    assert llm.calls == []


async def test_batch_path_routes_the_same_conflict_through_adjudication(monkeypatch):
    """smart_create_memories' auto-band (>=0.90) shares _decide_action, so a
    conflicting item inside a batch must reach the supersede path too."""
    dedup, llm, emb = _make_dedup(
        monkeypatch,
        llm=_RecordingLLM(content='{"action": "contradiction_update", "reason": "r"}'),
    )
    dedup.memory_service.search_memories_by_embedding = AsyncMock(
        return_value=[{"id": "m1", "similarity_score": 0.95, "content": _OLD_FACT}]
    )
    superseding = MagicMock(id="new-id", content=_NEW_FACT)
    dedup._supersede_with_new = AsyncMock(return_value=superseding)

    results = await dedup.smart_create_memories(
        new_memories=[_mem_create(_NEW_FACT)], user_id="u1"
    )
    assert [action for _, action in results] == ["contradiction_updated"]
    dedup._supersede_with_new.assert_awaited_once()
    assert dedup._supersede_with_new.call_args.kwargs["old_memory_id"] == "m1"
    assert len(llm.calls) == 1


# ── D-mem-A (2): end-to-end supersede on a real (sqlite) DB ──────────


async def test_supersede_e2e_new_row_active_old_row_superseded(monkeypatch):
    """The full production sequence: store fact → restate with a different
    value → old row superseded (superseded_by + is_active=False), new value
    active and retrievable → verbatim restatement of the NEW value still
    reinforces (no LLM) → update_memory on the returned id succeeds."""
    import app.services.memory_dedup_service as mds
    import app.services.memory_service as ms_mod
    from sqlalchemy import select
    from app.db.models import Memory, User

    llm = _RecordingLLM(content='{"action": "contradiction_update", "reason": "value changed"}')
    emb = _FakeEmbeddings()
    monkeypatch.setattr(mds, "get_llm_service", lambda: llm)
    monkeypatch.setattr(mds, "get_embedding_service", lambda: emb)
    monkeypatch.setattr(ms_mod, "get_embedding_service", lambda: emb)

    user_id = str(uuid.uuid4())
    engine, Session = await _memory_session()
    try:
        async with Session() as db:
            db.add(User(id=user_id, email=f"sup-{user_id[:8]}@example.com", hashed_password="x"))
            await db.commit()

            dedup = mds.MemoryDedupService(db=db)

            # 1. store the original fact
            old_mem, action = await dedup.smart_create_memory(
                new_memory=_mem_create(_OLD_FACT), user_id=user_id
            )
            assert action == "created"
            old_id = old_mem.id

            # 2. restate with a conflicting value — sim 1.0 (same fake
            #    vector), so pre-fix this was swallowed as "reinforced"
            new_mem, action = await dedup.smart_create_memory(
                new_memory=_mem_create(_NEW_FACT), user_id=user_id
            )
            assert action == "contradiction_updated"
            assert new_mem.content == _NEW_FACT
            assert new_mem.id != old_id
            assert len(llm.calls) == 1

            old_row = (await db.execute(
                select(Memory).where(Memory.id == old_id)
            )).scalar_one()
            assert old_row.is_active is False
            assert old_row.superseded_by == new_mem.id
            # #375 mechanism: superseded rows are archived, never deleted
            assert old_row.is_deleted is False

            # 3. the superseded row is invisible to dedup search, so a
            #    verbatim restatement of the NEW value reinforces the NEW
            #    row via the no-LLM shortcut
            again, action = await dedup.smart_create_memory(
                new_memory=_mem_create(_NEW_FACT), user_id=user_id
            )
            assert action == "reinforced"
            assert again.id == new_mem.id
            assert len(llm.calls) == 1  # no further adjudication

            # 4. D-mem-B companion (same-DB half): the id the reinforce path
            #    returned must be updatable — the shared update_memory lookup
            #    is strictly more permissive than the reinforce candidate
            #    filters. (The cross-DB half — MCP/REST routing to the tenant
            #    — is #379's, pinned in test_memory_taxonomy_and_ttl.py.)
            svc = ms_mod.MemoryService(db)
            updated = await svc.update_memory(
                again.id, user_id,
                MemoryUpdate(content="My storage locker passphrase is kestrel-ffff"),
            )
            assert updated is not None
            assert updated.content.endswith("kestrel-ffff")
    finally:
        await engine.dispose()


async def test_plain_create_dedup_never_reinforces_a_superseded_row(monkeypatch):
    """The REST POST / MCP memory_create path (create_memory(deduplicate=True)
    → _find_similar_memory) filtered on user/is_deleted/brain only — a freshly
    superseded row could be matched and 'reinforced' back into service,
    re-arming D-mem-A one write later. Pins the is_active filter."""
    import app.services.memory_service as ms_mod
    from sqlalchemy import select
    from app.db.models import Memory, User

    emb = _FakeEmbeddings()
    monkeypatch.setattr(ms_mod, "get_embedding_service", lambda: emb)

    user_id = str(uuid.uuid4())
    new_id = str(uuid.uuid4())
    superseded_id = str(uuid.uuid4())
    engine, Session = await _memory_session()
    try:
        async with Session() as db:
            db.add(User(id=user_id, email=f"arc-{user_id[:8]}@example.com", hashed_password="x"))
            db.add(Memory(
                id=superseded_id, user_id=user_id, content=_OLD_FACT,
                category="possessions", memory_type="fact", brain_type="user",
                is_active=False, superseded_by=new_id,
                embedding_json=json.dumps(emb.vector),
            ))
            await db.commit()

            svc = ms_mod.MemoryService(db)
            created = await svc.create_memory(
                user_id=user_id,
                memory_data=_mem_create(_OLD_FACT),
                deduplicate=True,
                embedding=list(emb.vector),
            )
            # Must be a NEW row — never the archived one resurrected/reinforced
            assert created.id != superseded_id
            archived = (await db.execute(
                select(Memory).where(Memory.id == superseded_id)
            )).scalar_one()
            assert archived.is_active is False
            assert archived.access_count in (0, None)  # not "reinforced"
    finally:
        await engine.dispose()


# ── D-mem-B companion: update_memory consistency + degrade ───────────


async def test_update_memory_syncs_canonical_content_and_clears_stale_summary(monkeypatch):
    """Dedup adjudicates against `canonical_content or content` and the card
    renders `summary || content` — an update that left them stale kept both
    future dedup verdicts and the UI anchored to the pre-update text."""
    import app.services.memory_service as ms_mod
    from app.db.models import Memory, User

    emb = _FakeEmbeddings()
    monkeypatch.setattr(ms_mod, "get_embedding_service", lambda: emb)

    user_id = str(uuid.uuid4())
    mem_id = str(uuid.uuid4())
    engine, Session = await _memory_session()
    try:
        async with Session() as db:
            db.add(User(id=user_id, email=f"upd-{user_id[:8]}@example.com", hashed_password="x"))
            db.add(Memory(
                id=mem_id, user_id=user_id, content=_OLD_FACT,
                canonical_content=_OLD_FACT, summary=_OLD_FACT[:20],
                category="possessions", memory_type="fact", brain_type="user",
            ))
            await db.commit()

            svc = ms_mod.MemoryService(db)
            updated = await svc.update_memory(
                mem_id, user_id, MemoryUpdate(content=_NEW_FACT)
            )
            assert updated.content == _NEW_FACT
            assert updated.canonical_content == _NEW_FACT
            assert updated.summary is None  # stale abbreviation cleared
    finally:
        await engine.dispose()


async def test_update_memory_accepts_plain_string_category(monkeypatch):
    """MemoryUpdate.category is Optional[str]; the old `.value` access raised
    AttributeError on EVERY category-carrying update (both REST PATCH and the
    MCP memory_update no-tenant fallback). Values normalize through the single
    taxonomy — aliases degrade instead of 500."""
    import app.services.memory_service as ms_mod
    from app.db.models import Memory, User
    from app.memory_taxonomy import normalize_category

    emb = _FakeEmbeddings()
    monkeypatch.setattr(ms_mod, "get_embedding_service", lambda: emb)

    user_id = str(uuid.uuid4())
    mem_id = str(uuid.uuid4())
    engine, Session = await _memory_session()
    try:
        async with Session() as db:
            db.add(User(id=user_id, email=f"cat-{user_id[:8]}@example.com", hashed_password="x"))
            db.add(Memory(
                id=mem_id, user_id=user_id, content=_OLD_FACT,
                category="other", memory_type="fact", brain_type="user",
            ))
            await db.commit()

            svc = ms_mod.MemoryService(db)
            updated = await svc.update_memory(
                mem_id, user_id, MemoryUpdate(category="preferences")
            )
            assert updated is not None
            assert updated.category == "preferences"

            # Unknown value degrades through normalize_category, never raises
            updated = await svc.update_memory(
                mem_id, user_id, MemoryUpdate(category="no-such-category")
            )
            assert updated.category == normalize_category("no-such-category", brain_type="user")
    finally:
        await engine.dispose()


async def test_update_memory_degrades_when_embedding_fails(monkeypatch):
    """W1.5 mirror: update_memory was the ONE write path still calling the
    sync embed() with no try/except — loop-blocking, and a hard 500 on agent
    images without sentence-transformers. A vector failure must degrade to a
    content-only update.

    2026-07-31 (F4): the degrade must also CLEAR the vector. Keeping the old
    one leaves the row indexed under text it no longer contains — vector
    search surfaces it for its pre-update meaning and misses the new one — and
    scripts/backfill_embeddings only re-embeds rows whose embedding IS NULL,
    so nothing could ever repair it."""
    import app.services.memory_service as ms_mod
    from app.db.models import Memory, User

    class _ExplodingEmbeddings:
        async def embed_async(self, text, api_key=None):
            raise ImportError("No module named 'sentence_transformers'")

        def embed(self, text, api_key=None):  # pragma: no cover
            raise AssertionError("sync embed() must not run on the update path")

    monkeypatch.setattr(ms_mod, "get_embedding_service", lambda: _ExplodingEmbeddings())

    user_id = str(uuid.uuid4())
    mem_id = str(uuid.uuid4())
    old_vec = json.dumps([0.2] * 8)
    engine, Session = await _memory_session()
    try:
        async with Session() as db:
            db.add(User(id=user_id, email=f"deg-{user_id[:8]}@example.com", hashed_password="x"))
            db.add(Memory(
                id=mem_id, user_id=user_id, content=_OLD_FACT,
                category="possessions", memory_type="fact", brain_type="user",
                embedding_json=old_vec,
            ))
            await db.commit()

            svc = ms_mod.MemoryService(db)
            updated = await svc.update_memory(
                mem_id, user_id, MemoryUpdate(content=_NEW_FACT)
            )
            assert updated is not None  # no 500
            assert updated.content == _NEW_FACT
            # Stale vector cleared, NOT kept — and NULL is exactly what the
            # backfill job looks for.
            assert updated.embedding_json is None
            assert getattr(updated, "embedding", None) is None
    finally:
        await engine.dispose()


# ── D-mem-C (1): the shared explicit-remember predicate ──────────────

# The four memory_quality_suite sentence shapes that measured 5/8 dropped
# (real token shapes, not the eval's live nonces).
_HARNESS_SENTENCES = [
    "Please remember this: my storage locker passphrase is memqa-11112222-lock-ece9.",
    "For my records: the watering code for the community garden is memqa-33334444-lock-9c15. Please remember it.",
    "Note this down: the nickname of my touring bicycle is memqa-55556666-slot-ab12.",
    "Please remember: my assigned parking spot code is memqa-77778888-slot-9c15.",
]


def test_predicate_matches_every_harness_sentence():
    for sentence in _HARNESS_SENTENCES:
        assert is_explicit_remember_request(sentence) is True, sentence
    for phrasing in [
        "Save this: my wifi password is hunter2",
        "note down: locker 44 is mine",
        "don't forget: the gate code changed to 9911",
    ]:
        assert is_explicit_remember_request(phrasing) is True, phrasing


def test_predicate_rejects_reminders_distractors_and_empties():
    # "remember to <verb>" is reminder phrasing — routines territory, not a
    # fact-save; matching it would bypass the trivial gate on every reminder.
    assert is_explicit_remember_request("remember to water the ferns tomorrow") is False
    assert is_explicit_remember_request("please remember to call my mother") is False
    # Distractor seeds from the harness carry no remember phrasing.
    assert is_explicit_remember_request("her desk door code is memqa-9999-lock-0000") is False
    assert is_explicit_remember_request("what's the weather today") is False
    assert is_explicit_remember_request("") is False
    assert is_explicit_remember_request(None) is False


# ── D-mem-C (2): intent boost in classify_query_intent ───────────────


def test_remember_sentences_classify_as_memory_intent():
    """Pre-fix, 'code'-carrying payloads tied memory 2 vs code 2 and lost on
    the priority order — the save ask got the CODE prompt sections. The boost
    must land every explicit save in memory intent, with the memory tools
    exposed. (#371 TOOLS_DOCGEN precedent: keyword-collision routing fix.)"""
    from app.agent.query_intent import classify_query_intent

    for sentence in _HARNESS_SENTENCES:
        intent = classify_query_intent(sentence)
        assert intent.category == "memory", (sentence, intent.category)
        assert "memory_store" in intent.tool_names
        assert "memory_search" in intent.tool_names


def test_remember_boost_kill_switch_restores_old_scoring(monkeypatch):
    from app.agent.query_intent import classify_query_intent

    monkeypatch.setattr(settings, "memory_tools_on_remember", False)
    # "…parking spot CODE is…" ties memory 2 / code 2 → code wins on the
    # tie-break priority. That IS the old (buggy) routing — the kill switch
    # must reproduce it exactly.
    intent = classify_query_intent(_HARNESS_SENTENCES[3])
    assert intent.category == "code"


def test_memory_tools_stay_always_included_for_plain_questions():
    """Tool EXPOSURE was never the D-mem-C gap — memory_store/memory_search
    are in _ALWAYS_INCLUDED_TOOLS (A6-3) and must stay reachable on tool-less
    intents; the boost only moves prompt sections and the stable-prefix
    allowed_tools restriction."""
    from app.agent.query_intent import (
        _ALWAYS_INCLUDED_TOOLS,
        classify_query_intent,
        filter_tools_by_intent,
    )

    assert {"memory_store", "memory_search"} <= _ALWAYS_INCLUDED_TOOLS

    tools = [{"name": n} for n in ("memory_store", "memory_search", "run_python")]
    intent = classify_query_intent("what's the capital of France?")
    kept = {t["name"] for t in filter_tools_by_intent(tools, intent)}
    assert {"memory_store", "memory_search"} <= kept


# ── D-mem-C (3): trivial-gate bypass in _extract_memories ────────────


class _FakeExtractor:
    def __init__(self, memories=None):
        self._memories = memories or []
        self.extract_calls: list[dict] = []
        self.last_extraction_outcome = "ok"

    async def extract_memories_with_llm(self, **kwargs):
        self.extract_calls.append(kwargs)
        return self._memories

    async def extract_relationships_with_llm(self, **kwargs):
        return []


class _FakeDedupService:
    def __init__(self, db, api_key=None):
        pass

    async def smart_create_memories(self, new_memories, user_id):
        return [
            (SimpleNamespace(id=f"m{i}", content=nm.content), "created")
            for i, nm in enumerate(new_memories)
        ]


def _make_runner():
    from app.agent.agent_runner import AgentRunner

    runner = AgentRunner.__new__(AgentRunner)  # skip heavy __init__
    runner._last_extraction_ok = "-"
    return runner


def _patch_extraction_pipeline(monkeypatch, extractor):
    import app.services.memory_extractor as me_mod
    import app.services.memory_dedup_service as mds_mod

    monkeypatch.setattr(me_mod, "get_memory_extractor", lambda: extractor)
    monkeypatch.setattr(mds_mod, "MemoryDedupService", _FakeDedupService)


async def test_explicit_save_bypasses_the_trivial_gate(monkeypatch):
    """A short remember-ask can classify trivial (<= 8 words); pre-fix the
    W1.4c gate dropped extraction for the turn and the fact was gone."""
    extractor = _FakeExtractor()
    _patch_extraction_pipeline(monkeypatch, extractor)
    runner = _make_runner()
    count = await runner._extract_memories(
        db=MagicMock(), user_id="u1",
        user_message="Remember this: gate code 4417",
        assistant_response="Saved.",
        query_was_trivial=True,
    )
    assert count == 0  # extractor returned nothing — but it WAS consulted
    assert len(extractor.extract_calls) == 1
    assert extractor.extract_calls[0]["explicit_save_requested"] is True


async def test_trivial_gate_still_skips_ordinary_chatter(monkeypatch):
    extractor = _FakeExtractor()
    _patch_extraction_pipeline(monkeypatch, extractor)
    runner = _make_runner()
    count = await runner._extract_memories(
        db=MagicMock(), user_id="u1", user_message="thanks",
        assistant_response="np", query_was_trivial=True,
    )
    assert count == 0
    assert extractor.extract_calls == []  # W1.4c gate unchanged


async def test_non_trivial_turns_thread_the_hint_honestly(monkeypatch):
    extractor = _FakeExtractor()
    _patch_extraction_pipeline(monkeypatch, extractor)
    runner = _make_runner()
    await runner._extract_memories(
        db=MagicMock(), user_id="u1",
        user_message="I moved to Berlin last month and started a new job",
        assistant_response="noted",
    )
    assert extractor.extract_calls[0]["explicit_save_requested"] is False


# ── D-mem-C (4): the extractor hint + relaxed noise filters ──────────


def _extractor_with_llm(monkeypatch, llm):
    import app.services.llm_service as llm_service

    monkeypatch.setattr(llm_service, "get_llm_service", lambda: llm)
    return MemoryExtractor()


async def test_explicit_save_block_present_only_when_requested(monkeypatch):
    llm = _RecordingLLM(content='{"memories": []}')
    extractor = _extractor_with_llm(monkeypatch, llm)

    await extractor.extract_memories_with_llm(
        user_message="Please remember: my locker code is memqa-1234",
        assistant_response="Saved.",
        explicit_save_requested=True,
    )
    hinted = llm.calls[0]["messages"][0]["content"]
    assert "EXPLICITLY REQUESTED SAVE" in hinted
    assert "VERBATIM" in hinted

    await extractor.extract_memories_with_llm(
        user_message="I like green tea", assistant_response="noted",
    )
    base = llm.calls[1]["messages"][0]["content"]
    # The hint must not leak into ordinary extractions — the base prompt
    # stays stable for every other call.
    assert "EXPLICITLY REQUESTED SAVE" not in base


async def test_explicit_save_keeps_token_like_and_low_importance_facts(monkeypatch):
    """The two filters that measurably ate explicit saves: the <4-words gate
    (token-like output: 'Parking code: memqa-x') and the importance<0.3
    floor. Both stay ON for ordinary turns."""
    payload = json.dumps({"memories": [
        {
            "content": "Parking code: memqa-77778888-slot-9c15",  # 3 words
            "category": "possessions", "memory_type": "fact",
            "importance": 0.9, "confidence": 0.9,
        },
        {
            "content": "The user's touring bicycle nickname is memqa-55556666-slot-ab12",
            "category": "possessions", "memory_type": "fact",
            "importance": 0.2,  # below the 0.3 floor
            "confidence": 0.9,
        },
    ]})

    llm = _RecordingLLM(content=payload)
    extractor = _extractor_with_llm(monkeypatch, llm)
    kept = await extractor.extract_memories_with_llm(
        user_message="Please remember both codes",
        assistant_response="Saved.",
        explicit_save_requested=True,
    )
    assert [m.content for m in kept] == [
        "Parking code: memqa-77778888-slot-9c15",
        "The user's touring bicycle nickname is memqa-55556666-slot-ab12",
    ]

    llm2 = _RecordingLLM(content=payload)
    extractor2 = _extractor_with_llm(monkeypatch, llm2)
    dropped = await extractor2.extract_memories_with_llm(
        user_message="chatting about parking", assistant_response="ok",
    )
    assert dropped == []  # ordinary turns: both noise filters still apply


# ── flags: built vs enabled contract ─────────────────────────────────


def test_flag_defaults_are_on_with_kill_switches():
    assert settings.memory_supersede_on_conflict is True
    assert settings.memory_tools_on_remember is True


# ── D-eval-mem: behavioral_suite sweeps its own rows ─────────────────

_SUITE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "eval" / "behavioral_suite.py"
_spec = importlib.util.spec_from_file_location("behavioral_suite_dmem", _SUITE_PATH)
bs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bs)


def test_cleanup_pattern_matches_the_scenario_token_shape():
    assert bs.MEMORY_FACT_PREFIX == "kestrel"
    assert bs.MEMORY_CLEANUP_PATTERN == "%kestrel-%"
    # The scenario token is built FROM the prefix constant, so the sweep
    # pattern can never drift away from what scenario 2 actually seeds.
    src = _SUITE_PATH.read_text()
    assert 'f"{MEMORY_FACT_PREFIX}-{secrets.token_hex(2)}"' in src


def test_cleanup_delegates_to_the_quality_suite_sweep(monkeypatch):
    """cleanup_memory_rows must reuse memory_quality_suite.cleanup_marker_rows
    — the FK-safe delete order (children before parent, superseded_by unlink,
    ::text[] casts) lives there and is pinned by test_memory_quality_scoring."""
    recorded = {}

    def _fake_sweep(db_url, pattern):
        recorded["args"] = (db_url, pattern)
        return {"pattern": pattern, "matched": 3, "statements": {}}

    fake_mod = SimpleNamespace(cleanup_marker_rows=_fake_sweep)
    monkeypatch.setitem(sys.modules, "memory_quality_suite", fake_mod)

    result = bs.cleanup_memory_rows("postgresql://x/y")
    assert recorded["args"] == ("postgresql://x/y", "%kestrel-%")
    assert result["matched"] == 3


def test_suite_wires_the_sweep_and_the_keep_memories_escape_hatch():
    """Source pins (main() builds its parser inline, and run_suite drives six
    live chat turns — neither is separable for a functional call here):
    the sweep is wired into run_suite, its failure is recorded rather than
    raised (the canary cron alerts on scenario failures / exit codes, and a
    cleanup hiccup must not page anyone), and --keep-memories skips it."""
    src = _SUITE_PATH.read_text()
    assert 'results["memory_cleanup"]' in src
    assert "cleanup_memory_rows(os.environ.get" in src
    # sweep failure → recorded under memory_cleanup, never a raise
    assert 'results["memory_cleanup"] = {"error"' in src
    # escape hatch declared and consulted
    assert "--keep-memories" in src
    assert 'getattr(args, "keep_memories", False)' in src


# ── Post-review integrity round (2026-07-31) ─────────────────────────
#
# An adversarial data-integrity review of this PR confirmed six defects
# that the first pass left behind. Each test below fails without its fix.


def test_conflicts_on_value_catches_typo_corrections_under_containment():
    """F2. The guard used to abstain whenever _is_same_information said the
    two texts were 'the same' — and that helper calls one string containing
    the other with a length delta < 20 chars identical. A corrected value is
    exactly that shape, so every one-character fix ("991" → "9911") was sent
    straight back to the reinforce path that swallows it."""
    from app.services.memory_service import conflicts_on_value

    assert conflicts_on_value("My gate code is 991", "My gate code is 9911")
    assert conflicts_on_value("My gate code is 9911", "My gate code is 991")
    assert conflicts_on_value(
        "My wifi password is hunter2", "My wifi password is hunter22"
    )
    assert conflicts_on_value(_OLD_FACT, "My storage locker passphrase is kestrel-dbf")


def test_conflicts_on_value_still_merges_added_detail():
    """The same relaxation must NOT turn 'added detail' into a conflict —
    a strict token subset stays on the no-LLM duplicate shortcut."""
    from app.services.memory_service import conflicts_on_value

    assert not conflicts_on_value(
        "I play basketball", "I play basketball on Saturdays"
    )
    assert not conflicts_on_value("I love pizza", "I love pizza")
    # Different topics share too few tokens to be a value conflict.
    assert not conflicts_on_value("I love pizza", "I have a dog named Max")


async def test_rest_create_surface_keeps_the_correction(monkeypatch):
    """F1 (the one the review called P0-class). The guard shipped only on
    MemoryDedupService._decide_action — the memory_store tool + background
    extraction. MemoryService.create_memory(deduplicate=True) has its OWN
    copy of the shortcut, and that is the one POST /api/memories and #379's
    MCP memory_create actually run: _find_similar_memory(0.9) →
    _reinforce_memory, which never reads the new content. So the exact
    production repro still discarded the user's correction on that surface."""
    import app.services.memory_service as ms_mod
    from sqlalchemy import select
    from app.db.models import Memory, User

    monkeypatch.setattr(ms_mod, "get_embedding_service", lambda: _FakeEmbeddings())

    user_id = str(uuid.uuid4())
    engine, Session = await _memory_session()
    try:
        async with Session() as db:
            db.add(User(id=user_id, email=f"rest-{user_id[:8]}@example.com", hashed_password="x"))
            await db.commit()
            svc = ms_mod.MemoryService(db)

            first = await svc.create_memory(user_id=user_id, memory_data=_mem_create(_OLD_FACT))
            second = await svc.create_memory(user_id=user_id, memory_data=_mem_create(_NEW_FACT))

            # Pre-fix: second is first (reinforced), and the new value is gone.
            assert second.id != first.id, "correction was swallowed by reinforcement"
            assert second.content == _NEW_FACT

            rows = (await db.execute(
                select(Memory).where(Memory.user_id == user_id, Memory.is_active == True)
            )).scalars().all()
            contents = {r.content for r in rows}
            assert _NEW_FACT in contents, "the corrected value must be retrievable"


    finally:
        await engine.dispose()


async def test_rest_create_surface_still_reinforces_true_duplicates(monkeypatch):
    """The F1 guard must not turn ordinary duplicate writes into row spam:
    a verbatim restatement still reinforces exactly as before."""
    import app.services.memory_service as ms_mod
    from app.db.models import User

    monkeypatch.setattr(ms_mod, "get_embedding_service", lambda: _FakeEmbeddings())

    user_id = str(uuid.uuid4())
    engine, Session = await _memory_session()
    try:
        async with Session() as db:
            db.add(User(id=user_id, email=f"dup-{user_id[:8]}@example.com", hashed_password="x"))
            await db.commit()
            svc = ms_mod.MemoryService(db)

            first = await svc.create_memory(user_id=user_id, memory_data=_mem_create(_OLD_FACT))
            again = await svc.create_memory(user_id=user_id, memory_data=_mem_create(_OLD_FACT))

            assert again.id == first.id, "verbatim duplicate must still reinforce"
    finally:
        await engine.dispose()


async def test_rest_create_kill_switch_restores_old_semantics(monkeypatch):
    """F1 under settings.memory_supersede_on_conflict=False: byte-for-byte
    the pre-fix behaviour (correction swallowed), so the flag is a real
    kill switch on this surface too."""
    import app.services.memory_service as ms_mod
    from app.db.models import User

    monkeypatch.setattr(ms_mod, "get_embedding_service", lambda: _FakeEmbeddings())
    monkeypatch.setattr(settings, "memory_supersede_on_conflict", False, raising=False)

    user_id = str(uuid.uuid4())
    engine, Session = await _memory_session()
    try:
        async with Session() as db:
            db.add(User(id=user_id, email=f"ks-{user_id[:8]}@example.com", hashed_password="x"))
            await db.commit()
            svc = ms_mod.MemoryService(db)

            first = await svc.create_memory(user_id=user_id, memory_data=_mem_create(_OLD_FACT))
            second = await svc.create_memory(user_id=user_id, memory_data=_mem_create(_NEW_FACT))

            assert second.id == first.id, "kill switch must restore reinforcement"
    finally:
        await engine.dispose()


async def test_two_different_doors_reach_the_adjudicator_and_both_survive(monkeypatch):
    """F3. Two DIFFERENT instances of the same kind of fact ("front door
    code" vs "garage door code") share enough tokens to trip the value-
    conflict guard, so the pair now reaches the LLM adjudicator. The
    adjudicator must be able to answer "new" — and when it does, BOTH rows
    stay active. A contradiction_update here would retire a fact the user
    never changed, with no undo.

    The prompt rule that steers this is asserted behaviourally: the pair is
    adjudicated (not auto-duplicated), and a "new" verdict keeps both."""
    import app.services.memory_dedup_service as mds
    import app.services.memory_service as ms_mod
    from sqlalchemy import select
    from app.db.models import Memory, User

    front = "My front door code is 1234"
    garage = "My garage door code is 9876"

    llm = _RecordingLLM(content='{"action": "new", "reason": "different doors"}')
    emb = _FakeEmbeddings()
    monkeypatch.setattr(mds, "get_llm_service", lambda: llm)
    monkeypatch.setattr(mds, "get_embedding_service", lambda: emb)
    monkeypatch.setattr(ms_mod, "get_embedding_service", lambda: emb)

    user_id = str(uuid.uuid4())
    engine, Session = await _memory_session()
    try:
        async with Session() as db:
            db.add(User(id=user_id, email=f"door-{user_id[:8]}@example.com", hashed_password="x"))
            await db.commit()
            dedup = mds.MemoryDedupService(db=db)

            await dedup.smart_create_memory(new_memory=_mem_create(front), user_id=user_id)
            _, action = await dedup.smart_create_memory(
                new_memory=_mem_create(garage), user_id=user_id
            )

            assert len(llm.calls) == 1, "same-kind/different-instance pair must be adjudicated"
            assert action == "created"

            active = (await db.execute(
                select(Memory).where(Memory.user_id == user_id, Memory.is_active == True)
            )).scalars().all()
            assert {m.content for m in active} == {front, garage}, "both doors must survive"

    # Both adjudication prompts must carry the different-instance rule, or
    # the model has nothing to distinguish these pairs by.
    finally:
        await engine.dispose()


async def test_update_on_a_superseded_row_lands_on_the_active_head(monkeypatch):
    """F5. update_memory → get_memory filters is_deleted ONLY. After dedup
    supersedes A with B, an update against A's id (the model routinely holds
    an id from an earlier turn or search) returned 200 while every retrieval
    path — which filters is_active — kept serving B's old value. The edit was
    invisible forever, with no surface that could show the user it had gone
    nowhere."""
    import app.services.memory_dedup_service as mds
    import app.services.memory_service as ms_mod
    from sqlalchemy import select
    from app.db.models import Memory, User

    llm = _RecordingLLM(content='{"action": "contradiction_update", "reason": "value changed"}')
    emb = _FakeEmbeddings()
    monkeypatch.setattr(mds, "get_llm_service", lambda: llm)
    monkeypatch.setattr(mds, "get_embedding_service", lambda: emb)
    monkeypatch.setattr(ms_mod, "get_embedding_service", lambda: emb)

    user_id = str(uuid.uuid4())
    engine, Session = await _memory_session()
    try:
        async with Session() as db:
            db.add(User(id=user_id, email=f"head-{user_id[:8]}@example.com", hashed_password="x"))
            await db.commit()
            dedup = mds.MemoryDedupService(db=db)

            old_mem, _ = await dedup.smart_create_memory(
                new_memory=_mem_create(_OLD_FACT), user_id=user_id
            )
            new_mem, action = await dedup.smart_create_memory(
                new_memory=_mem_create(_NEW_FACT), user_id=user_id
            )
            assert action == "contradiction_updated"

            # The stale id an earlier turn handed the model.
            svc = ms_mod.MemoryService(db)
            corrected = "My storage locker passphrase is kestrel-9999"
            updated = await svc.update_memory(old_mem.id, user_id, MemoryUpdate(content=corrected))

            assert updated is not None
            assert updated.id == new_mem.id, "edit must land on the active head, not the tombstone"

            head = (await db.execute(
                select(Memory).where(Memory.id == new_mem.id)
            )).scalar_one()
            assert head.content == corrected
            assert head.is_active is True
    finally:
        await engine.dispose()


async def test_resolve_active_head_survives_a_supersede_cycle(monkeypatch):
    """F5 guard: superseded_by is a plain column with no acyclicity
    guarantee. A self-reference or an A→B→A pair must not spin the write
    path forever — it degrades to updating in place."""
    import app.services.memory_service as ms_mod
    from app.db.models import Memory, User

    monkeypatch.setattr(ms_mod, "get_embedding_service", lambda: _FakeEmbeddings())

    user_id = str(uuid.uuid4())
    engine, Session = await _memory_session()
    try:
        async with Session() as db:
            db.add(User(id=user_id, email=f"cyc-{user_id[:8]}@example.com", hashed_password="x"))
            await db.commit()
            svc = ms_mod.MemoryService(db)

            a = await svc.create_memory(user_id=user_id, memory_data=_mem_create(_OLD_FACT))
            b = await svc.create_memory(
                user_id=user_id, memory_data=_mem_create(_NEW_FACT), deduplicate=False
            )
            a_row = await svc.get_memory(a.id, user_id)
            b_row = await svc.get_memory(b.id, user_id)
            a_row.is_active, a_row.superseded_by = False, b.id
            b_row.is_active, b_row.superseded_by = False, a.id  # cycle
            await db.commit()

            updated = await svc.update_memory(
                a.id, user_id, MemoryUpdate(content="cycle-safe write")
            )
            assert updated is not None  # terminated, no hang, no exception
    finally:
        await engine.dispose()


async def test_supersede_is_atomic_when_the_deactivation_fails(monkeypatch):
    """F6. _supersede_with_new used to run TWO transactions: create_memory
    committed the replacement, then the old row was deactivated and committed
    separately. A crash in between left BOTH conflicting values active and
    retrievable — precisely the state supersede exists to prevent. The pair
    must now be all-or-nothing."""
    import app.services.memory_dedup_service as mds
    import app.services.memory_service as ms_mod
    from sqlalchemy import select
    from app.db.models import Memory, User

    llm = _RecordingLLM(content='{"action": "contradiction_update", "reason": "value changed"}')
    emb = _FakeEmbeddings()
    monkeypatch.setattr(mds, "get_llm_service", lambda: llm)
    monkeypatch.setattr(mds, "get_embedding_service", lambda: emb)
    monkeypatch.setattr(ms_mod, "get_embedding_service", lambda: emb)

    user_id = str(uuid.uuid4())
    engine, Session = await _memory_session()
    try:
        async with Session() as db:
            db.add(User(id=user_id, email=f"atom-{user_id[:8]}@example.com", hashed_password="x"))
            await db.commit()
            dedup = mds.MemoryDedupService(db=db)

            old_mem, _ = await dedup.smart_create_memory(
                new_memory=_mem_create(_OLD_FACT), user_id=user_id
            )

            boom = RuntimeError("connection reset between the two commits")
            real_commit = db.commit
            state = {"armed": True}

            async def _explode():
                if state["armed"]:
                    state["armed"] = False
                    raise boom
                await real_commit()

            monkeypatch.setattr(db, "commit", _explode)
            with pytest.raises(RuntimeError):
                await dedup._supersede_with_new(
                    old_memory_id=old_mem.id,
                    new_memory_data=_mem_create(_NEW_FACT),
                    user_id=user_id,
                    reason="value changed",
                    new_embedding=emb.vector,
                )
            monkeypatch.undo()

            active = (await db.execute(
                select(Memory).where(Memory.user_id == user_id, Memory.is_active == True)
            )).scalars().all()
            contents = {m.content for m in active}
            assert _NEW_FACT not in contents, "replacement must not survive a failed supersede"
            assert contents == {_OLD_FACT}, "exactly one value stays active"
    finally:
        await engine.dispose()


async def test_supersede_loser_does_not_leave_a_stray_duplicate(monkeypatch):
    """F6, concurrent half. Two writers restating the same changed fact both
    see the old row active; the optimistic `UPDATE … WHERE is_active = true`
    lets exactly one win, and the loser rolls its replacement back instead of
    leaving a second active copy of the same value behind."""
    import app.services.memory_dedup_service as mds
    import app.services.memory_service as ms_mod
    from sqlalchemy import select
    from app.db.models import Memory, User

    llm = _RecordingLLM(content='{"action": "contradiction_update", "reason": "value changed"}')
    emb = _FakeEmbeddings()
    monkeypatch.setattr(mds, "get_llm_service", lambda: llm)
    monkeypatch.setattr(mds, "get_embedding_service", lambda: emb)
    monkeypatch.setattr(ms_mod, "get_embedding_service", lambda: emb)

    user_id = str(uuid.uuid4())
    engine, Session = await _memory_session()
    try:
        async with Session() as db:
            db.add(User(id=user_id, email=f"race-{user_id[:8]}@example.com", hashed_password="x"))
            await db.commit()
            dedup = mds.MemoryDedupService(db=db)

            old_mem, _ = await dedup.smart_create_memory(
                new_memory=_mem_create(_OLD_FACT), user_id=user_id
            )
            winner = await dedup._supersede_with_new(
                old_memory_id=old_mem.id,
                new_memory_data=_mem_create(_NEW_FACT),
                user_id=user_id,
                reason="value changed",
                new_embedding=emb.vector,
            )

            # Second writer arrives holding the same (now stale) old id.
            with pytest.raises(mds.SupersedeRaceError):
                await dedup._supersede_with_new(
                    old_memory_id=old_mem.id,
                    new_memory_data=_mem_create("My storage locker passphrase is kestrel-aaaa"),
                    user_id=user_id,
                    reason="value changed again",
                    new_embedding=emb.vector,
                )

            active = (await db.execute(
                select(Memory).where(Memory.user_id == user_id, Memory.is_active == True)
            )).scalars().all()
            assert [m.id for m in active] == [winner.id], "loser must leave nothing behind"
    finally:
        await engine.dispose()
