"""Dedup/contradiction adjudication must see EVERY candidate, not just the top one.

`MemoryDedupService._find_candidates` fetches up to MAX_ADJUDICATED_CANDIDATES
rows and keeps the ones at or above CANDIDATE_THRESHOLD (0.40). Both entry
points then sorted that list and threw away everything except `candidates[0]`:

    smart_create_memory    top_match = candidates[0]
    smart_create_memories  top_match = candidates[0]

There is no separate contradiction detector — `_decide_action` returns one of
duplicate|merge|contradiction_update|new for a single (existing, new) pair, and
`_apply_decision` acts on it. So a stale memory that contradicts the incoming
fact but ranked 2nd was never compared against it at all: the new fact was
stored beside it and both retrieved from then on.

These tests are pure-mock (no DB, no pgvector), so they run in CI's sqlite
sweep as well as under Postgres.
"""

from __future__ import annotations

import re
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.schemas import BrainType, MemoryCreate, MemoryLevel, MemoryType
from app.services.memory_dedup_service import (
    AUTO_DUPLICATE_THRESHOLD,
    AUTO_NEW_THRESHOLD,
    MAX_ADJUDICATED_CANDIDATES,
    MemoryDedupService,
)

# pytest.ini sets asyncio_mode = auto — async tests need no marker.


# ── the fixture facts ────────────────────────────────────────────────
#
# This is the bug's own output used as the fixture, which is why it is
# realistic: "planning is on Monday" was stored; "planning is on Thursday"
# arrived later, was adjudicated against some third row, and was stored
# alongside Monday instead of superseding it. Now Thursday is restated.

INCOMING = "The user's weekly planning session is on Thursday"
PARAPHRASE = "Thursday is when the user does their weekly planning"  # near-dup
STALE = "The user's weekly planning session is on Monday"           # contradiction
UNRELATED = "The user's weekly planning session notes live in Notion"


class _ScriptedAdjudicator:
    """complete_with_json fake that answers per PAIR, keyed on the EXISTING
    memory text found in the prompt. It records every call so a test can pin
    the per-write call count."""

    def __init__(self, verdicts: dict[str, str]):
        self.verdicts = verdicts
        self.calls: list[dict] = []

    def _verdict(self, existing: str) -> str:
        assert existing in self.verdicts, f"unscripted pair: {existing!r}"
        return self.verdicts[existing]

    async def complete_with_json(self, messages, model=None, **kwargs):
        prompt = messages[0]["content"]
        self.calls.append({"prompt": prompt, "model": model})

        batch = re.findall(r'PAIR (\d+):\nEXISTING MEMORY:\n"([^"]*)"', prompt)
        if batch:
            decisions = [
                {"index": int(i), "action": self._verdict(existing), "reason": "scripted"}
                for i, existing in batch
            ]
            import json
            return SimpleNamespace(content=json.dumps({"decisions": decisions}))

        single = re.search(r'EXISTING MEMORY:\n"([^"]*)"', prompt)
        assert single, "prompt matched neither the batch nor the single-pair shape"
        import json
        return SimpleNamespace(
            content=json.dumps({"action": self._verdict(single.group(1)), "reason": "scripted"})
        )

    @property
    def pair_count(self) -> int:
        """Total (existing, new) pairs sent across all calls."""
        total = 0
        for call in self.calls:
            batch = re.findall(r'PAIR (\d+):\nEXISTING MEMORY:\n"([^"]*)"', call["prompt"])
            total += len(batch) if batch else 1
        return total


class _FakeEmbeddings:
    def __init__(self):
        self.vector = [0.1] * 8

    async def embed_async(self, text, api_key=None):
        return list(self.vector)


def _make_dedup(monkeypatch, verdicts: dict[str, str] | None = None):
    import app.services.memory_dedup_service as mds

    llm = _ScriptedAdjudicator(verdicts or {})
    monkeypatch.setattr(mds, "get_llm_service", lambda: llm)
    monkeypatch.setattr(mds, "get_embedding_service", _FakeEmbeddings)
    dedup = mds.MemoryDedupService(db=MagicMock())
    dedup.memory_service = AsyncMock()

    # Outcome seams — each records which candidate id it was aimed at.
    dedup.memory_service.create_memory = AsyncMock(
        return_value=MagicMock(id="created-row", content=INCOMING)
    )
    dedup._get_memory_by_id = AsyncMock(
        side_effect=lambda mid: MagicMock(id=mid, content="row " + mid)
    )
    dedup._reinforce_existing_memory = AsyncMock(
        side_effect=lambda memory, new_data: memory
    )
    dedup._supersede_with_new = AsyncMock(
        return_value=MagicMock(id="replacement-row", content=INCOMING)
    )
    dedup._merge_memories = AsyncMock(return_value=MagicMock(id="merged-row"))
    return dedup, llm


def _mem(content: str) -> MemoryCreate:
    return MemoryCreate(
        content=content,
        summary=content[:50],
        brain_type=BrainType.USER,
        category="preferences",
        memory_type=MemoryType.FACT,
        importance=0.5,
        confidence=0.9,
        memory_level=MemoryLevel.EPISODIC,
        source_type="conversation",
    )


def _candidate(mid: str, score: float, content: str) -> dict:
    return {"id": mid, "similarity_score": score, "content": content}


def _search(dedup, *result_sets):
    dedup.memory_service.search_memories_by_embedding = AsyncMock(
        side_effect=[list(rs) for rs in result_sets]
    )


# ── HEADLINE: the rank-2 contradiction is adjudicated and superseded ──


async def test_rank_two_contradiction_supersedes_when_rank_one_is_unrelated(monkeypatch):
    """The plain case: rank 1 embeds close but is about something else, rank 2
    is the stale fact this write invalidates. Before the fix only rank 1 was
    ever adjudicated, its verdict was "new", and the Monday row survived."""
    dedup, llm = _make_dedup(monkeypatch, {UNRELATED: "new", STALE: "contradiction_update"})
    _search(
        dedup,
        [_candidate("rank-1-unrelated", 0.74, UNRELATED),
         _candidate("rank-2-stale", 0.61, STALE)],
    )

    memory, action = await dedup.smart_create_memory(new_memory=_mem(INCOMING), user_id="u1")

    assert action == "contradiction_updated", (
        "the rank-2 contradiction was never adjudicated; the stale fact stays retrievable"
    )
    assert dedup._supersede_with_new.await_args.kwargs["old_memory_id"] == "rank-2-stale"
    assert memory.id == "replacement-row"


async def test_contradiction_outranks_a_duplicate_verdict_on_a_nearer_row(monkeypatch):
    """The precedence call, pinned. Rank 1 is a true paraphrase of the incoming
    fact ("duplicate"); rank 2 is the stale value ("contradiction_update").
    Deciding on rank 1 alone reinforces the paraphrase, drops the incoming
    fact, and leaves the Monday row ACTIVE — two answers to one question,
    forever. Contradiction wins: the stale row is retired (reversibly) and the
    fact is stored."""
    dedup, llm = _make_dedup(
        monkeypatch, {PARAPHRASE: "duplicate", STALE: "contradiction_update"}
    )
    _search(
        dedup,
        [_candidate("rank-1-paraphrase", 0.78, PARAPHRASE),
         _candidate("rank-2-stale", 0.62, STALE)],
    )

    _, action = await dedup.smart_create_memory(new_memory=_mem(INCOMING), user_id="u1")

    assert action == "contradiction_updated"
    assert dedup._supersede_with_new.await_args.kwargs["old_memory_id"] == "rank-2-stale"
    assert dedup._reinforce_existing_memory.await_count == 0


async def test_batch_path_also_supersedes_a_rank_two_contradiction(monkeypatch):
    """smart_create_memories is the path production takes — agent_runner calls
    it on every turn, while smart_create_memory is only reached from the
    memory_store tool. A fix that lands on one of these and not the other is
    this file's whole history, so the batch path is pinned separately."""
    dedup, llm = _make_dedup(monkeypatch, {UNRELATED: "new", STALE: "contradiction_update"})
    _search(
        dedup,
        [_candidate("rank-1-unrelated", 0.74, UNRELATED),
         _candidate("rank-2-stale", 0.61, STALE)],
    )

    results = await dedup.smart_create_memories(new_memories=[_mem(INCOMING)], user_id="u1")

    assert [action for _, action in results] == ["contradiction_updated"]
    assert dedup._supersede_with_new.await_args.kwargs["old_memory_id"] == "rank-2-stale"


# ── ANTI-VACUITY CONTROLS: must stay GREEN under the mutation ─────────


async def test_control_plain_duplicate_at_rank_one_is_still_a_duplicate(monkeypatch):
    """Adjudicating every candidate must not turn ordinary writes into
    supersedes. Rank 1 says duplicate, rank 2 says new -> reinforce rank 1,
    exactly as before the change."""
    dedup, llm = _make_dedup(monkeypatch, {PARAPHRASE: "duplicate", UNRELATED: "new"})
    _search(
        dedup,
        [_candidate("rank-1-paraphrase", 0.78, PARAPHRASE),
         _candidate("rank-2-unrelated", 0.55, UNRELATED)],
    )

    memory, action = await dedup.smart_create_memory(new_memory=_mem(INCOMING), user_id="u1")

    assert action == "reinforced"
    assert memory.id == "rank-1-paraphrase"
    assert dedup._supersede_with_new.await_count == 0
    assert dedup.memory_service.create_memory.await_count == 0


async def test_control_auto_duplicate_shortcut_still_skips_the_llm(monkeypatch):
    """A single >= AUTO_DUPLICATE_THRESHOLD candidate is reinforced with no
    adjudication call at all — the W1.4d cost shortcut is untouched."""
    dedup, llm = _make_dedup(monkeypatch)
    _search(dedup, [_candidate("verbatim", 0.95, PARAPHRASE)])

    memory, action = await dedup.smart_create_memory(new_memory=_mem(INCOMING), user_id="u1")

    assert action == "reinforced"
    assert memory.id == "verbatim"
    assert llm.calls == []


async def test_control_all_candidates_new_still_creates(monkeypatch):
    """Every candidate says "new" -> a plain create, as before."""
    dedup, llm = _make_dedup(monkeypatch, {UNRELATED: "new", PARAPHRASE: "new"})
    _search(
        dedup,
        [_candidate("a", 0.74, UNRELATED), _candidate("b", 0.66, PARAPHRASE)],
    )

    memory, action = await dedup.smart_create_memory(new_memory=_mem(INCOMING), user_id="u1")

    assert action == "created"
    assert memory.id == "created-row"
    assert dedup._supersede_with_new.await_count == 0


# ── COST: adjudication is batched, so N candidates != N calls ─────────


async def test_four_ambiguous_candidates_cost_exactly_one_llm_call(monkeypatch):
    """The whole point of using _llm_decide_actions_batch: widening the
    comparison from 1 candidate to N must not multiply the call count."""
    contents = [f"The user's weekly planning note number {i}" for i in range(4)]
    dedup, llm = _make_dedup(monkeypatch, {c: "new" for c in contents})
    _search(
        dedup,
        [_candidate(f"c{i}", 0.80 - 0.05 * i, c) for i, c in enumerate(contents)],
    )

    await dedup.smart_create_memory(new_memory=_mem(INCOMING), user_id="u1")

    assert len(llm.calls) == 1, "one adjudication call per write, regardless of candidate count"
    assert llm.pair_count == 4
    assert all(f"PAIR {i}" in llm.calls[0]["prompt"] for i in range(4))


async def test_candidates_outside_the_ambiguous_band_never_reach_the_llm(monkeypatch):
    """Deterministic verdicts still cost nothing: >= AUTO_DUPLICATE_THRESHOLD
    and < AUTO_NEW_THRESHOLD are settled by the thresholds at every rank, not
    just at rank 1. This write made 0 calls before the change and makes 0
    after."""
    dedup, llm = _make_dedup(monkeypatch)
    _search(
        dedup,
        [_candidate("auto-dup", AUTO_DUPLICATE_THRESHOLD + 0.02, PARAPHRASE),
         _candidate("auto-new", AUTO_NEW_THRESHOLD - 0.05, UNRELATED)],
    )

    _, action = await dedup.smart_create_memory(new_memory=_mem(INCOMING), user_id="u1")

    assert action == "reinforced"
    assert llm.calls == []


async def test_a_second_ambiguous_candidate_costs_one_call_not_two(monkeypatch):
    """The only class of write whose call count rises: rank 1 auto-resolves
    (0 calls before) but rank 2 is ambiguous. It costs ONE call, not one per
    candidate — and it is the case the fix exists for."""
    dedup, llm = _make_dedup(monkeypatch, {STALE: "contradiction_update"})
    _search(
        dedup,
        [_candidate("auto-dup", 0.93, PARAPHRASE),
         _candidate("rank-2-stale", 0.62, STALE)],
    )

    _, action = await dedup.smart_create_memory(new_memory=_mem(INCOMING), user_id="u1")

    assert len(llm.calls) == 1
    assert llm.pair_count == 1  # only the ambiguous candidate was sent
    assert action == "contradiction_updated"


async def test_batch_path_keeps_one_call_for_every_memory_and_candidate(monkeypatch):
    """3 memories x 2 ambiguous candidates = 6 pairs, still ONE request for the
    whole turn. Before the change that turn also cost one call (3 pairs); the
    prompt grows, the call count does not."""
    existing = [f"The user's weekly planning note number {i}" for i in range(6)]
    dedup, llm = _make_dedup(monkeypatch, {c: "new" for c in existing})
    _search(
        dedup,
        [_candidate("a0", 0.80, existing[0]), _candidate("a1", 0.70, existing[1])],
        [_candidate("b0", 0.80, existing[2]), _candidate("b1", 0.70, existing[3])],
        [_candidate("c0", 0.80, existing[4]), _candidate("c1", 0.70, existing[5])],
    )

    memories = [_mem(f"The user's planning fact number {i} is settled") for i in range(3)]
    results = await dedup.smart_create_memories(new_memories=memories, user_id="u1")

    assert [action for _, action in results] == ["created", "created", "created"]
    assert len(llm.calls) == 1, "one adjudication call for the whole turn"
    assert llm.pair_count == 6


async def test_adjudication_is_capped_at_max_adjudicated_candidates(monkeypatch):
    """The fan-out bound is explicit, not accidental: however many rows the
    search returns, at most MAX_ADJUDICATED_CANDIDATES pairs are adjudicated."""
    contents = [f"The user's weekly planning note number {i}" for i in range(9)]
    dedup, llm = _make_dedup(monkeypatch, {c: "new" for c in contents})
    _search(
        dedup,
        [_candidate(f"c{i}", 0.85 - 0.01 * i, c) for i, c in enumerate(contents)],
    )

    await dedup.smart_create_memory(new_memory=_mem(INCOMING), user_id="u1")

    assert len(llm.calls) == 1
    assert llm.pair_count == MAX_ADJUDICATED_CANDIDATES


# ── the precedence table itself ───────────────────────────────────────


@pytest.mark.parametrize(
    "verdicts,expected_action,expected_id",
    [
        # contradiction beats everything
        (["duplicate", "contradiction_update"], "contradiction_update", "b"),
        (["merge", "contradiction_update"], "contradiction_update", "b"),
        (["new", "contradiction_update"], "contradiction_update", "b"),
        # merge beats duplicate and new
        (["duplicate", "merge"], "merge", "b"),
        (["new", "merge"], "merge", "b"),
        # duplicate beats new
        (["new", "duplicate"], "duplicate", "b"),
        # ties resolve to the NEARER row (the list arrives similarity-sorted)
        (["duplicate", "duplicate"], "duplicate", "a"),
        (["new", "new"], "new", "a"),
        (["contradiction_update", "contradiction_update"], "contradiction_update", "a"),
    ],
)
def test_precedence_contradiction_gt_merge_gt_duplicate_gt_new(
    verdicts, expected_action, expected_id
):
    candidates = [
        {"id": "a", "similarity_score": 0.80, "content": PARAPHRASE},
        {"id": "b", "similarity_score": 0.60, "content": STALE},
    ]
    decisions = [{"action": v, "reason": ""} for v in verdicts]

    decision, winner = MemoryDedupService._select_decision(candidates, decisions)

    assert decision["action"] == expected_action
    assert winner["id"] == expected_id


def test_select_decision_treats_an_unanswered_candidate_as_new():
    """A None verdict (the adjudicator returned nothing for that pair) must
    never be read as a contradiction."""
    candidates = [
        {"id": "a", "similarity_score": 0.80, "content": PARAPHRASE},
        {"id": "b", "similarity_score": 0.60, "content": STALE},
    ]
    decision, winner = MemoryDedupService._select_decision(
        candidates, [None, {"action": "duplicate", "reason": ""}]
    )
    assert decision["action"] == "duplicate"
    assert winner["id"] == "b"
