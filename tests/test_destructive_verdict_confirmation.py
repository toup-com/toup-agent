"""A verdict that makes a fact unretrievable must be confirmed on its own.

WHAT THIS PINS (measured 2026-08-08)
------------------------------------
`memverify`'s concurrency test caught 1 of 20 rapid-fire writes vanishing with
no exception raised. The mechanism is not a lock, a transaction or a race in
the storage layer — it is adjudication:

  * every candidate above CANDIDATE_THRESHOLD for one write is adjudicated in a
    SINGLE batched LLM call, and
  * `_select_decision` applies the highest-PRECEDENCE verdict among them,

so one wrong verdict out of N decides a destructive action. Measured against
gpt-4o-mini on facts sharing a sentence template and differing only in a
two-digit number ("project number 04 … vireo-04", cosine 0.91-0.92), 40 trials
per cell — what matters is WHERE the incoming fact sits among its candidates:

    incoming 04 vs {05, 03}   straddled    │ 15/40 and 28/40 (two runs)
    incoming 04 vs {03, 02}   both below   │  0/40
    incoming 02 vs {00, 01}   sequential   │  0/40

Straddled, the model reads the incoming fact as project 05's value having
CHANGED and answers contradiction_update, retiring the existing row. Widening
the batch dilutes it (same pair inside 3 candidates 4/40, inside 5 1/40, asked
alone 0/40), so the misfire needs a straddle AND few candidates — which is
what racing writes produce and sequential writes never do.

Bringing the batched prompt's rubric to parity with the single-pair prompt
moved the straddled cell 28% -> 22%, i.e. it was not the prompt: what is
reliable is asking about ONE pair at a time.

Hence: batched adjudication stays the cheap pre-filter, and any verdict that
would make a fact stop being retrievable is re-asked alone before it is applied.

These tests are deterministic — the adjudicator is scripted, so they assert the
PROTOCOL (what is re-asked, and what happens on agreement vs disagreement), not
a model's accuracy. The measured rates above are re-checked against the real
model by tests/memverify/test_f_dedup.py.
"""

from __future__ import annotations

import json
import re
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytestmark = pytest.mark.asyncio

# Two facts of the same KIND about DIFFERENT instances. Nothing in the text
# says they conflict; a wrong contradiction_update retires the first one.
FACT_A = "The user's project number 03 is codenamed vireo-03."
FACT_B = "The user's project number 04 is codenamed vireo-04."

# A genuine value change on the SAME thing — the control. If a fix stops
# superseding altogether to make the tests above pass, this one fails.
DOOR_OLD = "The user's front door code is 1234."
DOOR_NEW = "The user changed the front door code to 9876."


class _ScriptedAdjudicator:
    """Answers the batched and single-pair prompts INDEPENDENTLY.

    That split is the whole point: the batched call is what misfires in
    production and the single-pair call is what corrects it, so a fake that
    could not disagree with itself would make every test here vacuous.
    """

    def __init__(self, batch_action: str, single_action: str):
        self.batch_action = batch_action
        self.single_action = single_action
        self.calls: list[dict] = []

    async def complete_with_json(self, messages, model=None, **kwargs):
        prompt = messages[0]["content"]
        pairs = re.findall(r'PAIR (\d+):\nEXISTING MEMORY:\n"([^"]*)"', prompt)
        self.calls.append({
            "prompt": prompt,
            "shape": "batch" if pairs else "single",
            "pairs": len(pairs) or 1,
        })
        if pairs:
            return SimpleNamespace(content=json.dumps({
                "decisions": [
                    {"index": int(i), "action": self.batch_action, "reason": "scripted"}
                    for i, _ in pairs
                ]
            }))
        return SimpleNamespace(
            content=json.dumps({"action": self.single_action, "reason": "scripted"})
        )

    @property
    def shapes(self) -> list[str]:
        return [c["shape"] for c in self.calls]


class _FakeEmbeddings:
    """Every text embeds identically -> similarity 1.0, so the >=0.90 branch is
    always taken and the value/subject guards decide whether the LLM is asked.
    Keeps the tests about adjudication rather than about embedding behaviour."""

    def embed(self, text, api_key=None):
        return [0.1] * 1536

    async def embed_async(self, text, api_key=None):
        return [0.1] * 1536


def _mem(content: str):
    from app.schemas import BrainType, MemoryCreate, MemoryLevel, MemoryType

    return MemoryCreate(
        content=content,
        brain_type=BrainType.USER,
        category="identity",
        memory_type=MemoryType("fact"),
        importance=0.8,
        confidence=0.9,
        memory_level=MemoryLevel.EPISODIC,
        emotional_salience=0.5,
        source_type="agent_tool",
    )


def _dedup(monkeypatch, llm, candidates):
    """A dedup service whose candidate search is fixed and whose LLM is
    scripted. `_lock_user_writes` is stubbed: advisory locks need Postgres and
    this file is about adjudication."""
    import app.services.memory_dedup_service as mds

    monkeypatch.setattr(mds, "get_llm_service", lambda: llm)
    monkeypatch.setattr(mds, "get_embedding_service", lambda: _FakeEmbeddings())

    svc = mds.MemoryDedupService.__new__(mds.MemoryDedupService)
    svc.db = None
    svc.api_key = None
    svc.llm_service = llm
    svc.embedding_service = _FakeEmbeddings()
    svc.memory_service = SimpleNamespace(
        search_memories_by_embedding=AsyncMock(return_value=list(candidates)),
        create_memory=AsyncMock(
            side_effect=lambda **kw: SimpleNamespace(
                id=str(uuid.uuid4()), content=kw["memory_data"].content
            )
        ),
    )
    svc._lock_user_writes = AsyncMock(return_value=None)
    svc._supersede_with_new = AsyncMock(
        side_effect=lambda **kw: SimpleNamespace(
            id="superseding", content=kw["new_memory_data"].content
        )
    )
    svc._get_memory_by_id = AsyncMock(
        return_value=SimpleNamespace(id="m1", content=FACT_A)
    )
    svc._reinforce_existing_memory = AsyncMock(
        return_value=SimpleNamespace(id="m1", content=FACT_A)
    )
    return svc


# Which band a candidate sits in decides whether the LLM is consulted at all,
# so every test states it rather than inheriting a default:
#
#   >= AUTO_DUPLICATE_THRESHOLD (0.90) — the shortcut answers "duplicate"
#       WITHOUT the LLM unless conflicts_on_value or mentions_different_subjects
#       denies it. The vireo pairs measure 0.91-0.92 and DO conflict on a value
#       token, so they are escalated: that is the production shape.
#   AUTO_NEW_THRESHOLD (0.50) .. 0.90 — always adjudicated.
#   < 0.50 — "new" without the LLM.
AMBIGUOUS = 0.75
NEAR_IDENTICAL = 0.95


def _candidates(*contents, similarity=NEAR_IDENTICAL):
    return [
        {"id": f"m{i + 1}", "content": c, "similarity_score": similarity}
        for i, c in enumerate(contents)
    ]


# ── The refusal ──────────────────────────────────────────────────────────

async def test_batched_contradiction_refused_when_re_asked_alone(monkeypatch):
    """The production failure, reproduced deterministically.

    The batch says contradiction_update (as gpt-4o-mini did on 14 of 80 pairs
    at k=2); asked about the one pair alone it says new. The row must survive
    and the incoming fact must be stored — both facts retrievable."""
    llm = _ScriptedAdjudicator(batch_action="contradiction_update", single_action="new")
    svc = _dedup(monkeypatch, llm, _candidates(FACT_A, "The user's project number 02 is codenamed vireo-02."))

    memory, action = await svc.smart_create_memory(new_memory=_mem(FACT_B), user_id="u1")

    assert action == "created", "a refused contradiction must still STORE the incoming fact"
    assert memory.content == FACT_B
    svc._supersede_with_new.assert_not_awaited(), "no row may be retired on a refused verdict"
    assert llm.shapes == ["batch", "single"], "the winning pair was re-asked on its own"


async def test_batched_duplicate_refused_when_re_asked_alone(monkeypatch):
    """The other destructive verdict. `duplicate` discards the INCOMING text
    (the existing row is reinforced instead), so it is gated identically."""
    llm = _ScriptedAdjudicator(batch_action="duplicate", single_action="new")
    svc = _dedup(monkeypatch, llm, _candidates(FACT_A, "The user's project number 02 is codenamed vireo-02."))

    memory, action = await svc.smart_create_memory(new_memory=_mem(FACT_B), user_id="u1")

    assert action == "created"
    assert memory.content == FACT_B
    svc._reinforce_existing_memory.assert_not_awaited()


# ── The controls: the gate must not disable the outcomes it guards ───────

async def test_agreed_contradiction_still_supersedes(monkeypatch):
    """Without this, "never supersede" would pass every test above.

    A real value change on the SAME thing: both the batch and the re-ask say
    contradiction_update, so the old row IS retired.

    Adjudicated from the AMBIGUOUS band, which is where a rephrased value
    change actually lands — "…code is 1234" vs "changed the … code to 9876"
    shares too few tokens for conflicts_on_value (its floor is 0.5 overlap),
    so at >=0.90 it would take the shortcut and never reach the model.
    """
    llm = _ScriptedAdjudicator(
        batch_action="contradiction_update", single_action="contradiction_update"
    )
    svc = _dedup(
        monkeypatch, llm,
        _candidates(DOOR_OLD, "The user's dog is called Kesh.", similarity=AMBIGUOUS),
    )

    _, action = await svc.smart_create_memory(new_memory=_mem(DOOR_NEW), user_id="u1")

    assert action == "contradiction_updated"
    svc._supersede_with_new.assert_awaited_once()
    assert svc._supersede_with_new.call_args.kwargs["old_memory_id"] == "m1"
    assert llm.shapes == ["batch", "single"]


async def test_agreed_duplicate_still_reinforces(monkeypatch):
    """The same control for the dedup half — a confirmed duplicate still
    reinforces, so the gate has not turned dedup off and started accumulating
    near-identical rows."""
    llm = _ScriptedAdjudicator(batch_action="duplicate", single_action="duplicate")
    svc = _dedup(monkeypatch, llm, _candidates(FACT_A, "The user's dog is called Kesh."))

    _, action = await svc.smart_create_memory(new_memory=_mem(FACT_B), user_id="u1")

    assert action == "reinforced"
    svc._reinforce_existing_memory.assert_awaited_once()


async def test_merge_is_not_confirmed(monkeypatch):
    """`merge` composes both texts into one row rather than dropping either,
    so it is not in _DESTRUCTIVE_ACTIONS and pays no extra call. Pinned so the
    gate cannot quietly widen onto every verdict."""
    llm = _ScriptedAdjudicator(batch_action="merge", single_action="new")
    svc = _dedup(
        monkeypatch, llm,
        _candidates(FACT_A, "The user's dog is called Kesh.", similarity=AMBIGUOUS),
    )
    svc._merge_memories = AsyncMock(
        return_value=SimpleNamespace(id="m1", content=FACT_A + " " + FACT_B)
    )

    _, action = await svc.smart_create_memory(new_memory=_mem(FACT_B), user_id="u1")

    assert action == "merged"
    assert llm.shapes == ["batch"], "merge must not pay for a confirmation"


# ── Cost containment ─────────────────────────────────────────────────────

async def test_threshold_duplicate_is_never_re_asked(monkeypatch):
    """The >=0.90 shortcut is a similarity measurement past both guards, not a
    model opinion. Re-asking it would spend a call on the commonest path AND
    let a sampled "new" split a true paraphrase into two rows."""
    llm = _ScriptedAdjudicator(batch_action="new", single_action="new")
    # Identical text -> conflicts_on_value False, different_subjects False, so
    # _decide_action returns the threshold verdict with no LLM call at all.
    svc = _dedup(monkeypatch, llm, _candidates(FACT_A))

    _, action = await svc.smart_create_memory(new_memory=_mem(FACT_A), user_id="u1")

    assert action == "reinforced"
    assert llm.calls == [], "the threshold shortcut must not consult the LLM at all"


async def test_new_verdict_costs_no_confirmation(monkeypatch):
    """The overwhelmingly common outcome pays nothing extra."""
    llm = _ScriptedAdjudicator(batch_action="new", single_action="new")
    svc = _dedup(monkeypatch, llm, _candidates(FACT_A, "The user's project number 02 is codenamed vireo-02."))

    _, action = await svc.smart_create_memory(new_memory=_mem(FACT_B), user_id="u1")

    assert action == "created"
    assert llm.shapes == ["batch"]


async def test_confirmation_asks_about_the_winning_pair(monkeypatch):
    """The re-ask must carry the row that would actually be destroyed. Asking
    about some other candidate would be a confirmation in name only."""
    llm = _ScriptedAdjudicator(batch_action="contradiction_update", single_action="new")
    other = "The user's project number 02 is codenamed vireo-02."
    svc = _dedup(monkeypatch, llm, _candidates(FACT_A, other))
    # FACT_A is the top match by similarity order
    svc._get_memory_by_id = AsyncMock(return_value=SimpleNamespace(id="m1", content=FACT_A))

    await svc.smart_create_memory(new_memory=_mem(FACT_B), user_id="u1")

    confirm = llm.calls[-1]["prompt"]
    assert FACT_A in confirm and FACT_B in confirm
    assert confirm.count("EXISTING MEMORY") == 1, "confirmation must be a single pair"


# ── Failure of the adjudicator must not authorise deletion ───────────────

async def test_provider_failure_during_confirmation_keeps_both_facts(monkeypatch):
    """Provider down on the confirmation call.

    This does NOT reach _confirm_destructive_verdict's except branch:
    _llm_decide_action catches its own provider errors and degrades to
    _heuristic_decision, which can only answer duplicate (strict containment)
    or new — never contradiction_update. So the verdict is refused by the
    heuristic. Asserted explicitly because the safety here comes from a
    catch two layers down, and a future refactor that lets the error
    propagate must land on the other test below rather than on nothing.
    """

    class _BrokenOnSingle(_ScriptedAdjudicator):
        async def complete_with_json(self, messages, model=None, **kwargs):
            prompt = messages[0]["content"]
            if "PAIR 0:" not in prompt:
                raise RuntimeError("provider down")
            return await super().complete_with_json(messages, model=model, **kwargs)

    llm = _BrokenOnSingle(batch_action="contradiction_update", single_action="new")
    svc = _dedup(monkeypatch, llm, _candidates(FACT_A, "The user's project number 02 is codenamed vireo-02."))

    memory, action = await svc.smart_create_memory(new_memory=_mem(FACT_B), user_id="u1")

    assert action == "created"
    assert memory.content == FACT_B
    svc._supersede_with_new.assert_not_awaited()


async def test_confirmation_raising_keeps_both_facts(monkeypatch):
    """An outage must not become a licence to retire rows.

    Here the confirmation itself raises, which is the case
    _confirm_destructive_verdict handles directly. Written separately because
    the test above passes even with that handler deleted — the heuristic
    absorbs the error first — so without this one the handler would be
    unverified.
    """
    llm = _ScriptedAdjudicator(batch_action="contradiction_update", single_action="new")
    svc = _dedup(monkeypatch, llm, _candidates(FACT_A, "The user's project number 02 is codenamed vireo-02."))
    svc._llm_decide_action = AsyncMock(side_effect=RuntimeError("provider down"))

    memory, action = await svc.smart_create_memory(new_memory=_mem(FACT_B), user_id="u1")

    assert action == "created"
    assert memory.content == FACT_B
    svc._supersede_with_new.assert_not_awaited()


# ── The discard is recorded ──────────────────────────────────────────────

async def test_reinforce_records_the_discarded_incoming_text():
    """A confirmed `duplicate` still throws the incoming wording away. It must
    at least be traceable: without this, a fact that went missing left no
    evidence of whether it was recognised or absorbed."""
    import app.services.memory_dedup_service as mds

    svc = mds.MemoryDedupService.__new__(mds.MemoryDedupService)
    svc.db = SimpleNamespace(commit=AsyncMock(), refresh=AsyncMock())

    existing = SimpleNamespace(
        id="m1", content=FACT_A, canonical_content=None, strength=0.5,
        importance=0.5, confidence=0.5, expires_at=None, access_count=0,
        history_json=None, last_reinforced_at=None, updated_at=None,
    )
    await svc._reinforce_existing_memory(existing, _mem(FACT_B))

    history = json.loads(existing.history_json)
    assert history[-1]["action"] == "reinforced"
    assert history[-1]["discarded_content"] == FACT_B, (
        "the text that was thrown away has to be recoverable"
    )
    assert history[-1]["content"] == FACT_A, "and `content` still means the survivor"


async def test_reinforce_omits_the_field_for_a_verbatim_restatement():
    """No key when nothing was actually discarded — otherwise every ordinary
    restatement doubles the row's history for no information."""
    import app.services.memory_dedup_service as mds

    svc = mds.MemoryDedupService.__new__(mds.MemoryDedupService)
    svc.db = SimpleNamespace(commit=AsyncMock(), refresh=AsyncMock())

    existing = SimpleNamespace(
        id="m1", content=FACT_A, canonical_content=None, strength=0.5,
        importance=0.5, confidence=0.5, expires_at=None, access_count=0,
        history_json=None, last_reinforced_at=None, updated_at=None,
    )
    await svc._reinforce_existing_memory(existing, _mem(FACT_A))

    assert "discarded_content" not in json.loads(existing.history_json)[-1]
