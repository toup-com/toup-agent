"""W1.4 — extraction fan-out shape tests.

The background memory pipeline (fact extraction, relationship extraction,
dedup adjudication, dedup merge) used to make one LLM call per memory with
NO model pin — LLMService.default_model resolves to settings.agent_model
(the premium chat model) whenever an Anthropic key is present. This file
locks in:

(a) explicit model pin (settings.memory_extraction_model) at all call sites
(b) relationship extraction gated on extraction returning > 0 memories
(c) extraction skipped entirely on trivial turns (bool threaded, not re-classified)
(d) dedup adjudication batched into ONE call + similarity auto-thresholds
(e) the content embedding computed once and threaded through storage
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.config import settings
from app.schemas import BrainType, MemoryCategory, MemoryLevel, MemoryType, MemoryCreate
from app.services.memory_extractor import ExtractedMemory, MemoryExtractor


# ── fakes ────────────────────────────────────────────────────────────


class _RecordingLLM:
    """complete_with_json fake that records the model kwarg per call."""

    def __init__(self, content: str = '{"action": "new", "reason": "x"}', error: Exception | None = None):
        self.calls: list = []
        self._content = content
        self._error = error

    async def complete_with_json(self, messages, model=None, **kwargs):
        self.calls.append({"messages": messages, "model": model})
        if self._error:
            raise self._error
        return SimpleNamespace(content=self._content)


class _FakeEmbeddings:
    """embed_async fake; the sync embed() asserts it is never used (W1.4e)."""

    def __init__(self):
        self.vector = [0.1] * 8
        self.async_calls = 0

    async def embed_async(self, text, api_key=None):
        self.async_calls += 1
        return list(self.vector)

    def embed(self, text, api_key=None):  # pragma: no cover - failure path
        raise AssertionError("sync embed() must not run on the async storage path (W1.4e)")


def _make_dedup(monkeypatch, llm=None, emb=None):
    import app.services.memory_dedup_service as mds

    llm = llm or _RecordingLLM()
    emb = emb or _FakeEmbeddings()
    monkeypatch.setattr(mds, "get_llm_service", lambda: llm)
    monkeypatch.setattr(mds, "get_embedding_service", lambda: emb)
    dedup = mds.MemoryDedupService(db=MagicMock())
    dedup.memory_service = AsyncMock()
    return dedup, llm, emb


def _mem_create(content: str, importance: float = 0.5) -> MemoryCreate:
    return MemoryCreate(
        content=content,
        summary=content[:50],
        brain_type=BrainType.USER,
        category="preferences",
        memory_type=MemoryType.FACT,
        importance=importance,
        confidence=0.9,
        memory_level=MemoryLevel.EPISODIC,
        source_type="conversation",
    )


def _patch_llm_singleton(monkeypatch, llm):
    import app.services.llm_service as llm_service

    monkeypatch.setattr(llm_service, "get_llm_service", lambda: llm)


# ── (a) model pin at each call site ──────────────────────────────────


async def test_fact_extraction_pins_model(monkeypatch):
    llm = _RecordingLLM(content='{"memories": []}')
    _patch_llm_singleton(monkeypatch, llm)
    extractor = MemoryExtractor()
    await extractor.extract_memories_with_llm(user_message="hi", assistant_response="hello")
    assert len(llm.calls) == 1
    assert llm.calls[0]["model"] == settings.memory_extraction_model


async def test_relationship_extraction_pins_model(monkeypatch):
    llm = _RecordingLLM(content='{"relationships": []}')
    _patch_llm_singleton(monkeypatch, llm)
    extractor = MemoryExtractor()
    await extractor.extract_relationships_with_llm(user_message="hi", assistant_response="hello")
    assert len(llm.calls) == 1
    assert llm.calls[0]["model"] == settings.memory_extraction_model


async def test_dedup_adjudication_pins_model(monkeypatch):
    dedup, llm, _ = _make_dedup(
        monkeypatch, llm=_RecordingLLM(content='{"action": "merge", "reason": "r"}')
    )
    decision = await dedup._llm_decide_action(existing_content="I love pizza", new_content="I love pepperoni pizza")
    assert decision["action"] == "merge"
    assert len(llm.calls) == 1
    assert llm.calls[0]["model"] == settings.memory_extraction_model


async def test_dedup_merge_pins_model(monkeypatch):
    dedup, llm, _ = _make_dedup(
        monkeypatch,
        llm=_RecordingLLM(content='{"merged_content": "merged", "change_summary": "s"}'),
    )
    merged, summary = await dedup._llm_merge_contents(existing_content="a b c", new_content="d e f")
    assert merged == "merged"
    assert len(llm.calls) == 1
    assert llm.calls[0]["model"] == settings.memory_extraction_model


# ── (d) auto thresholds skip the LLM entirely ────────────────────────


async def test_decide_action_auto_duplicate_at_high_similarity(monkeypatch):
    dedup, llm, _ = _make_dedup(monkeypatch)
    decision = await dedup._decide_action(similarity=0.95, existing_content="x", new_content="y")
    assert decision["action"] == "duplicate"
    assert llm.calls == []  # no LLM call


async def test_decide_action_auto_new_below_low_similarity(monkeypatch):
    dedup, llm, _ = _make_dedup(monkeypatch)
    decision = await dedup._decide_action(similarity=0.45, existing_content="x", new_content="y")
    assert decision["action"] == "new"
    assert llm.calls == []


async def test_decide_action_middle_band_consults_llm(monkeypatch):
    dedup, llm, _ = _make_dedup(
        monkeypatch, llm=_RecordingLLM(content='{"action": "merge", "reason": "r"}')
    )
    decision = await dedup._decide_action(similarity=0.70, existing_content="x", new_content="y")
    assert decision["action"] == "merge"
    assert len(llm.calls) == 1


# ── (d) batched adjudication shape ───────────────────────────────────


async def test_batch_adjudication_single_call_aligned_verdicts(monkeypatch):
    dedup, llm, _ = _make_dedup(
        monkeypatch,
        llm=_RecordingLLM(
            content='{"decisions": ['
            '{"index": 0, "action": "merge", "reason": "r0"}, '
            '{"index": 1, "action": "contradiction_update", "reason": "r1"}]}'
        ),
    )
    decisions = await dedup._llm_decide_actions_batch([("e0", "n0"), ("e1", "n1")])
    assert len(llm.calls) == 1  # ONE call for both pairs
    assert llm.calls[0]["model"] == settings.memory_extraction_model
    assert [d["action"] for d in decisions] == ["merge", "contradiction_update"]
    # Both pairs appear in the single prompt
    prompt = llm.calls[0]["messages"][0]["content"]
    assert "PAIR 0" in prompt and "PAIR 1" in prompt


async def test_batch_adjudication_invalid_entries_degrade_to_heuristic(monkeypatch):
    dedup, llm, _ = _make_dedup(
        monkeypatch,
        llm=_RecordingLLM(
            # index 1 missing, index 0 has a bogus action, index 7 out of range
            content='{"decisions": ['
            '{"index": 0, "action": "explode", "reason": "r0"}, '
            '{"index": 7, "action": "merge", "reason": "oob"}]}'
        ),
    )
    decisions = await dedup._llm_decide_actions_batch(
        [("different topic entirely", "another thing"), ("same text here ok", "same text here ok")]
    )
    assert decisions[0]["action"] == "new"  # bogus action coerced
    assert decisions[1]["action"] == "duplicate"  # missing entry -> text heuristic


async def test_batch_adjudication_llm_failure_falls_back_per_pair(monkeypatch):
    dedup, llm, _ = _make_dedup(monkeypatch, llm=_RecordingLLM(error=RuntimeError("boom")))
    decisions = await dedup._llm_decide_actions_batch(
        [("same text here ok", "same text here ok"), ("apples and oranges", "quantum physics")]
    )
    assert decisions[0]["action"] == "duplicate"
    assert decisions[1]["action"] == "new"


async def test_batch_adjudication_single_pair_delegates_to_single_prompt(monkeypatch):
    dedup, llm, _ = _make_dedup(
        monkeypatch, llm=_RecordingLLM(content='{"action": "duplicate", "reason": "r"}')
    )
    decisions = await dedup._llm_decide_actions_batch([("e", "n")])
    assert len(decisions) == 1
    assert decisions[0]["action"] == "duplicate"
    assert len(llm.calls) == 1
    assert llm.calls[0]["model"] == settings.memory_extraction_model


# ── (d)+(e) smart_create_memories end-to-end shape ───────────────────


async def test_smart_create_memories_batches_and_threads_embedding(monkeypatch):
    llm = _RecordingLLM(
        content='{"decisions": ['
        '{"index": 0, "action": "merge", "reason": "r0"}, '
        '{"index": 1, "action": "new", "reason": "r1"}]}'
    )
    dedup, llm, emb = _make_dedup(monkeypatch, llm=llm)

    # 5 memories: no-candidate, auto-duplicate (0.95), auto-new (0.45),
    # and two ambiguous (0.70, 0.72) that must share ONE adjudication call.
    search_results = [
        [],
        [{"id": "m1", "similarity_score": 0.95, "content": "existing 1"}],
        [{"id": "m2", "similarity_score": 0.45, "content": "existing 2"}],
        [{"id": "m3", "similarity_score": 0.70, "content": "existing 3"}],
        [{"id": "m4", "similarity_score": 0.72, "content": "existing 4"}],
    ]
    dedup.memory_service.search_memories_by_embedding = AsyncMock(side_effect=search_results)
    created = MagicMock(id="new-id", content="created content")
    dedup.memory_service.create_memory = AsyncMock(return_value=created)

    reinforced = MagicMock(id="m1", content="existing 1")
    dedup._get_memory_by_id = AsyncMock(return_value=reinforced)
    dedup._reinforce_existing_memory = AsyncMock(return_value=reinforced)
    merged = MagicMock(id="m3", content="merged content")
    dedup._merge_memories = AsyncMock(return_value=merged)

    memories = [_mem_create(f"fact number {i} about the user") for i in range(5)]
    results = await dedup.smart_create_memories(new_memories=memories, user_id="u1")

    assert [action for _, action in results] == [
        "created", "reinforced", "created", "merged", "created",
    ]
    # ONE adjudication call total, for the two ambiguous pairs only
    assert len(llm.calls) == 1
    prompt = llm.calls[0]["messages"][0]["content"]
    assert "existing 3" in prompt and "existing 4" in prompt
    assert "existing 1" not in prompt and "existing 2" not in prompt

    # (e) content embedded exactly once per memory, threaded into storage
    assert emb.async_calls == 5
    for call in dedup.memory_service.create_memory.call_args_list:
        assert call.kwargs["embedding"] == emb.vector
        assert call.kwargs["deduplicate"] is False


async def test_smart_create_memory_single_threads_embedding(monkeypatch):
    dedup, llm, emb = _make_dedup(monkeypatch)
    dedup.memory_service.search_memories_by_embedding = AsyncMock(return_value=[])
    created = MagicMock(id="new-id", content="c")
    dedup.memory_service.create_memory = AsyncMock(return_value=created)

    memory, action = await dedup.smart_create_memory(
        new_memory=_mem_create("the user plays tennis on Sundays"), user_id="u1"
    )
    assert action == "created"
    assert emb.async_calls == 1
    assert dedup.memory_service.create_memory.call_args.kwargs["embedding"] == emb.vector
    assert llm.calls == []


async def test_merge_threads_embedding_into_merge_memory(monkeypatch):
    dedup, llm, emb = _make_dedup(monkeypatch)
    existing = MagicMock(canonical_content="the user works at Google", content="the user works at Google")
    dedup._get_memory_by_id = AsyncMock(return_value=existing)
    dedup._llm_merge_contents = AsyncMock(return_value=("the user works at Google as an engineer", "added role"))
    # expires_at is set explicitly for the same reason _FakeDedupService below
    # mirrors its input's fields: the stand-in must carry every attribute the
    # caller reads. _merge_memories now reconciles the expiry lease (a transient
    # memory restated as durable must not keep its horizon), and a bare
    # MagicMock returns a truthy mock for `.expires_at`, which sends it down the
    # promote branch and into `await self.db.commit()` on a mock session.
    # A real merge target has expires_at None or a datetime.
    updated = MagicMock(importance=0.9, expires_at=None)
    dedup.memory_service.merge_memory = AsyncMock(return_value=updated)

    result = await dedup._merge_memories(
        existing_memory_id="m1",
        new_content="the user is an engineer at Google",
        new_memory_data=_mem_create("the user is an engineer at Google"),
        user_id="u1",
    )
    assert result is updated
    assert emb.async_calls == 1  # merged content embedded once, async
    assert dedup.memory_service.merge_memory.call_args.kwargs["new_embedding"] == emb.vector


# ── (b) relationship gate + (c) trivial skip in _extract_memories ────


class _FakeExtractor:
    def __init__(self, memories):
        self._memories = memories
        self.extract_calls = 0
        self.relationship_calls = 0
        self.last_extraction_outcome = "ok"

    async def extract_memories_with_llm(self, **kwargs):
        self.extract_calls += 1
        return self._memories

    async def extract_relationships_with_llm(self, **kwargs):
        self.relationship_calls += 1
        return []


class _FakeDedupService:
    """Stands in for MemoryDedupService, returning stored-memory stand-ins.

    The stand-in must carry every field the caller reads off a stored memory,
    not just the ones this file cares about. It originally returned only
    `id` and `content`, and `_extract_memories` reads `.category` too — in its
    `describe_memory(...)` LOG line. The AttributeError propagated out of the
    store loop, the whole extraction was recorded as failed, and the test saw
    0 memories with no hint that a log statement was the cause.

    Mirroring the input's fields keeps that from recurring: whatever the
    caller reads, it is here, because the real pipeline puts it there.
    """

    #: action every stored memory is reported with. Class-level so a test can
    #: swap it without rebuilding the fake — the relationship mirror keys off
    #: this and nothing else.
    action = "created"

    def __init__(self, db, api_key=None):
        pass

    async def smart_create_memories(self, new_memories, user_id, *, person_names=None):
        # `person_names` mirrors the real contract (file routing, rebuild
        # 2026-08); the fake ignores it — routing is dedup's concern, and
        # this file guards the fan-out logic around it.
        return [
            (
                SimpleNamespace(
                    id=f"m{i}",
                    content=nm.content,
                    summary=getattr(nm, "summary", None),
                    category=getattr(nm, "category", None),
                    memory_type=getattr(nm, "memory_type", None),
                    importance=getattr(nm, "importance", None),
                ),
                type(self).action,
            )
            for i, nm in enumerate(new_memories)
        ]


def _make_runner():
    from app.agent.agent_runner import AgentRunner

    runner = AgentRunner.__new__(AgentRunner)  # skip heavy __init__
    runner._last_extraction_ok = "-"
    return runner


def _extracted(content: str) -> ExtractedMemory:
    return ExtractedMemory(
        content=content,
        summary=content[:50],
        category=MemoryCategory.PREFERENCES,
        memory_type=MemoryType.FACT,
        importance=0.7,
        confidence=0.9,
        entities=[],
        tags=[],
        metadata={"brain_type": "user"},
    )


def _patch_extraction_pipeline(monkeypatch, extractor):
    import app.services.memory_extractor as me_mod
    import app.services.memory_dedup_service as mds_mod

    monkeypatch.setattr(me_mod, "get_memory_extractor", lambda: extractor)
    monkeypatch.setattr(mds_mod, "MemoryDedupService", _FakeDedupService)


async def test_relationships_skipped_when_no_memories_extracted(monkeypatch):
    extractor = _FakeExtractor(memories=[])
    _patch_extraction_pipeline(monkeypatch, extractor)
    runner = _make_runner()
    count = await runner._extract_memories(
        db=MagicMock(), user_id="u1", user_message="hello", assistant_response="hi",
    )
    assert count == 0
    assert extractor.extract_calls == 1
    assert extractor.relationship_calls == 0  # W1.4b gate


async def test_relationships_run_when_memories_extracted(monkeypatch):
    extractor = _FakeExtractor(memories=[_extracted("the user's dog is named Max")])
    _patch_extraction_pipeline(monkeypatch, extractor)
    runner = _make_runner()
    count = await runner._extract_memories(
        db=MagicMock(), user_id="u1", user_message="my dog is Max", assistant_response="noted",
    )
    assert count == 1
    assert extractor.relationship_calls == 1


@pytest.mark.parametrize("action", ["merged", "reinforced", "skipped"])
async def test_relationships_skipped_when_nothing_new_was_recorded(monkeypatch, action):
    """A recall turn must not mint graph rows restating what memory just said.

    Measured in production (canary tenant, 2026-08-05). The user stated a fact
    in one session; in a DIFFERENT session they asked a question the agent
    answered from memory. Extraction behaved correctly and merged that turn
    into the existing row — creating nothing — and the relationship mirror,
    which sees none of that, wrote a brand-new memory restating the answer.

    Every recall turn was therefore a junk generator, and the junk is itself
    retrievable, so it compounds. The gate is "did this turn record a NEW
    fact", not "did it store anything": merged, reinforced and skipped all
    mean the graph already knows.
    """
    monkeypatch.setattr(_FakeDedupService, "action", action)
    extractor = _FakeExtractor(memories=[_extracted("the user only buys salmon cat food")])
    _patch_extraction_pipeline(monkeypatch, extractor)
    runner = _make_runner()

    count = await runner._extract_memories(
        db=MagicMock(), user_id="u1",
        user_message="I'm at the pet store, what should I pick up?",
        assistant_response="Salmon-based food, no chicken — for Vesper.",
    )

    assert count == 1, "the memory itself is still stored/merged as before"
    assert extractor.extract_calls == 1
    assert extractor.relationship_calls == 0, (
        f"action={action!r} recorded no new fact, so there is no new edge to "
        "mirror — firing the mirror here is what wrote 'USER buys salmon food'"
    )


async def test_trivial_turn_skips_extraction_entirely(monkeypatch):
    extractor = _FakeExtractor(memories=[_extracted("should never be stored")])
    _patch_extraction_pipeline(monkeypatch, extractor)
    runner = _make_runner()
    count = await runner._extract_memories(
        db=MagicMock(), user_id="u1", user_message="thanks", assistant_response="np",
        query_was_trivial=True,
    )
    assert count == 0
    assert extractor.extract_calls == 0  # no extraction LLM call
    assert extractor.relationship_calls == 0  # no relationship LLM call


async def test_trivial_skip_flag_off_restores_old_behavior(monkeypatch):
    extractor = _FakeExtractor(memories=[])
    _patch_extraction_pipeline(monkeypatch, extractor)
    monkeypatch.setattr(settings, "skip_extraction_for_trivial_queries", False)
    runner = _make_runner()
    await runner._extract_memories(
        db=MagicMock(), user_id="u1", user_message="thanks", assistant_response="np",
        query_was_trivial=True,
    )
    assert extractor.extract_calls == 1  # gate disabled -> extraction fires
