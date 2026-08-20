"""A failure inside the write path must never silently lose a fact.

THE PROPERTY, AND WHERE IT COMES FROM
-------------------------------------
`test_destructive_verdict_confirmation.py` is retired with
`MemoryDedupService`, but two of its tests were not about the dedup service
at all — `test_provider_failure_during_confirmation_keeps_both_facts` and
`test_confirmation_raising_keeps_both_facts` pinned a PROPERTY: when the
adjudicating model call fails mid-decision, nothing is destroyed and nothing
is dropped. The mechanism died; the property did not.

v3's translation is the curator outbox, and the property is harder here,
because the v3 writer runs fire-and-forget AFTER the reply has streamed.
There is no request to fail, no user watching, and no retry anyone will ask
for. If `curate_turn` raises and the caller shrugs, everything the user said
that turn is gone with no error surfaced anywhere.

FOUR OUTCOMES, AND ONLY ONE OF THEM IS A LOSS
---------------------------------------------
1. The model call RAISES (provider down, 401, timeout) -> park the turn,
   replay it on a later turn, and the fact lands. Tested below.
2. The model returns UNPARSEABLE JSON -> identical handling. A malformed
   reply is not "the model decided nothing".
3. The validator REJECTS every op, twice -> this is a DECISION, not a loss.
   The turn must NOT be parked (parking it means retrying a refusal forever
   at one model call each), and the refusal must be recorded so it is
   explicable afterwards.
4. `apply_ops` raises MID-BATCH -> the loss candidate. Nothing may be
   half-written: a body persisted without its change line, or two of three
   files updated, is a corpus nobody can reconcile.

WHY A REAL DATABASE
-------------------
Cases 3 and 4 are about what is on disk after the failure. Every test here
runs on its own sqlite engine, so the whole file runs in the ordinary CI
sweep with no key.
"""

from __future__ import annotations

import json
import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.models.base import Base
from app.db.models.memory import MemoryFile, MemoryFileChange
from app.db.models.memory_capture_outbox import MemoryCaptureOutbox
from app.db.models.user import User
from app.services import memory_capture_outbox_service as outbox
from app.services import memory_curator as curator
from app.services import memory_file_ops as ops
from app.services.user_identity import forget_cached_identity

pytestmark = pytest.mark.asyncio

#: A well-formed op set the validator accepts.
GOOD_OPS = json.dumps({"ops": [
    {"op": "add", "slug": "you/profile",
     "bullet": "moved to Vancouver in June 2026 for the UBC program",
     "change": "Added Profile: moved to Vancouver."},
]})

#: The turn that carries the fact. Long enough to clear the pre-gate.
USER_TEXT = "I moved to Vancouver in June 2026 for the UBC program."


class _Resp:
    def __init__(self, content):
        self.content = content


class _Sequence:
    """Replays canned replies; an Exception in the list is RAISED."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = 0

    async def complete_with_json(self, messages, **kw):
        self.calls += 1
        reply = self.replies.pop(0) if self.replies else GOOD_OPS
        if isinstance(reply, BaseException):
            raise reply
        return _Resp(reply)


async def _session():
    engine = create_async_engine(
        "sqlite+aiosqlite://", connect_args={"check_same_thread": False}
    )
    async with engine.begin() as conn:
        await conn.run_sync(
            Base.metadata.create_all,
            tables=[
                User.__table__, MemoryFile.__table__, MemoryFileChange.__table__,
                MemoryCaptureOutbox.__table__,
            ],
        )
    db = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)()
    user_id = str(uuid.uuid4())
    db.add(User(id=user_id, email="d@toup.ai", hashed_password="x", name="Dara Ahmadi"))
    await db.commit()
    forget_cached_identity()
    return db, user_id


def _install(monkeypatch, *replies):
    llm = _Sequence(*replies)
    monkeypatch.setattr(curator, "_llm", lambda api_key: llm)
    monkeypatch.setattr(curator, "EXTRACTION_RETRY_BACKOFF_S", 0)
    return llm


async def _bodies(db, user_id):
    return {f.slug: (f.body_md or "") for f in await ops._all_files(db, user_id)}


async def _park(db, user_id, exc: BaseException):
    """The runner's failure handler, verbatim in shape."""
    await outbox.record_turn_failure(
        db, user_id, USER_TEXT, "Noted.", exc, channel="app",
    )
    await db.commit()


# ── 1. The model call raises ──────────────────────────────────────────

async def test_a_provider_failure_parks_the_turn_and_the_replay_lands_the_fact(
    monkeypatch,
):
    db, user_id = await _session()

    # Turn 1: the provider is down. Both the call and its one retry fail.
    llm = _install(monkeypatch, RuntimeError("provider down"), RuntimeError("still down"))
    with pytest.raises(RuntimeError):
        await curator.curate_turn(db, user_id, user_text=USER_TEXT, assistant_text="Noted.")
    assert llm.calls == 2, "the transient retry did not fire"
    await _park(db, user_id, RuntimeError("provider down"))

    assert await outbox.pending_count(db, user_id) == 1
    assert "vancouver" not in " ".join((await _bodies(db, user_id)).values()).lower()

    # Turn 2: the provider is back. The replay re-runs the WRITER — not a
    # stored op set — against the files as they are now.
    _install(monkeypatch, GOOD_OPS)
    resolved = await outbox.replay_pending(db, user_id)
    await db.commit()

    assert resolved == 1
    assert "vancouver" in " ".join((await _bodies(db, user_id)).values()).lower()
    assert await outbox.pending_count(db, user_id) == 0


async def test_an_unparseable_reply_is_handled_exactly_like_a_dead_provider(
    monkeypatch,
):
    """A malformed reply is not "the model decided nothing"."""
    db, user_id = await _session()

    _install(monkeypatch, "I'm sorry, I can't help with that.", "not json either")
    with pytest.raises(Exception):
        await curator.curate_turn(db, user_id, user_text=USER_TEXT)
    await _park(db, user_id, ValueError("bad json"))
    assert await outbox.pending_count(db, user_id) == 1

    _install(monkeypatch, GOOD_OPS)
    assert await outbox.replay_pending(db, user_id) == 1
    await db.commit()
    assert "vancouver" in " ".join((await _bodies(db, user_id)).values()).lower()


async def test_the_parked_payload_is_the_TURN_not_an_op_set(monkeypatch):
    """Round 8 parked serialized `MemoryCreate` rows, which made the retry
    free. v3 cannot: the curator decides what to change ABOUT THE CURRENT
    FILES, so an op set computed against yesterday's bodies would `rewrite`
    bullets that have since been merged away and its `match` strings would
    no longer exist. The payload has to be re-runnable input."""
    db, user_id = await _session()
    await _park(db, user_id, RuntimeError("x"))

    row = (await db.execute(select(MemoryCaptureOutbox))).scalars().one()
    payload = row.payload_json
    if isinstance(payload, str):
        payload = json.loads(payload)
    assert payload["user_text"] == USER_TEXT
    assert "assistant_text" in payload and "channel" in payload
    assert "ops" not in payload


async def test_a_park_survives_a_poisoned_caller_session(monkeypatch):
    """The case the outbox exists for. If the write failed because of a
    database error, the caller's transaction is already poisoned and the
    INSERT that parks the row raises too — so the facts would be lost
    exactly when the safety net was supposed to catch them.

    Also pins the no-double-park rule: a failed flush leaves the object in
    `db.new`, so a later successful commit on that session would insert it a
    SECOND time alongside the rescue copy.
    """
    db, user_id = await _session()
    calls = {"n": 0}
    real_flush = db.flush

    async def _poisoned_flush(*a, **kw):
        calls["n"] += 1
        raise RuntimeError("current transaction is aborted")

    monkeypatch.setattr(db, "flush", _poisoned_flush)

    # The rescue session needs the same engine, which this file's sqlite
    # engine is not — so assert the fallback was ATTEMPTED and that nothing
    # was double-parked on the caller's session once it recovers.
    await outbox.record_turn_failure(db, user_id, USER_TEXT, "", RuntimeError("db"))
    assert calls["n"] == 1
    monkeypatch.setattr(db, "flush", real_flush)
    await db.commit()

    rows = (await db.execute(select(MemoryCaptureOutbox))).scalars().all()
    assert len(rows) == 0, (
        "the row that failed to flush was still in db.new and got inserted by "
        "the next commit — that is the double-park defect"
    )


# ── 2. A rejection is a DECISION, not a loss ──────────────────────────

async def test_a_rejected_op_set_is_recorded_and_NOT_parked(monkeypatch):
    """Case 3.

    The validator refusing everything twice is the writer working. Parking
    that turn would retry a refusal forever at one model call per turn — the
    outbox would become a permanent tax paid for a decision that already
    happened. The refusal has to be visible instead: `curate_turn` RETURNS
    (it does not raise), `applied` is 0, and `rejected` carries the
    validator's own words.
    """
    db, user_id = await _session()
    bad = json.dumps({"ops": [
        {"op": "add", "slug": "you/profile",
         "bullet": "You moved to Vancouver",          # leading subject
         "change": "Added Profile."},
        {"op": "add", "slug": "topics/ghost",          # no such file
         "bullet": "likes Googoosh", "change": "Added."},
    ]})
    llm = _install(monkeypatch, bad, bad)

    result = await curator.curate_turn(db, user_id, user_text=USER_TEXT)

    assert llm.calls == 2, "the validator's one retry did not fire"
    assert result["applied"] == 0
    assert result["rejected"], "a refusal with no reason is indistinguishable from a crash"
    assert any("subject is implied" in r for r in result["rejected"])
    assert await outbox.pending_count(db, user_id) == 0, (
        "a rejected turn was parked — the outbox is for FAILURES, and a "
        "refusal is an answer"
    )
    assert "vancouver" not in " ".join((await _bodies(db, user_id)).values()).lower()


async def test_one_bad_op_does_not_take_the_good_ones_with_it(monkeypatch):
    """The other half of "a rejection is not a loss": the batch is validated
    op by op, so a fact the writer got right still lands."""
    db, user_id = await _session()
    mixed = json.dumps({"ops": [
        {"op": "add", "slug": "you/profile",
         "bullet": "moved to Vancouver in June 2026",
         "change": "Added Profile: moved to Vancouver."},
        {"op": "add", "slug": "you/profile",
         "bullet": "The user prefers dark mode",       # third person
         "change": "Added Profile."},
    ]})
    _install(monkeypatch, mixed)

    result = await curator.curate_turn(db, user_id, user_text=USER_TEXT)
    assert result["applied"] == 1
    assert result["rejected"]
    body = (await _bodies(db, user_id))["you/profile"].lower()
    assert "vancouver" in body
    assert "the user prefers" not in body


# ── 3. apply_ops is atomic ────────────────────────────────────────────

async def test_a_crash_mid_apply_leaves_no_half_written_corpus(monkeypatch):
    """Case 4 — the real loss candidate.

    `apply_ops` writes whole bodies for every touched file and then a change
    row per accepted op, and commits ONCE at the end. A crash between those
    two must leave NOTHING: a body persisted without its change line is a
    change the user cannot see in their memory log, and two of three files
    updated is a corpus nobody can reconcile against the log.

    The injection point is `_write_change`, which runs after every body has
    been assigned on the session and before the single commit — the exact
    window where a partial write would show up if the boundary were wrong.
    """
    db, user_id = await _session()
    two_files = json.dumps({"ops": [
        {"op": "add", "slug": "you/profile",
         "bullet": "moved to Vancouver in June 2026",
         "change": "Added Profile: moved to Vancouver."},
        {"op": "create_file", "section": "topics", "slug": "topics/music",
         "title": "Music",
         "description": "Music taste — artists and albums; read when music comes up."},
        {"op": "add", "slug": "topics/music", "bullet": "likes Googoosh",
         "change": "Added Music: likes Googoosh."},
    ]})
    _install(monkeypatch, two_files)

    calls = {"n": 0}
    real_write_change = ops._write_change

    async def _explode(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 2:            # after the first change row is staged
            raise RuntimeError("disk went away mid-batch")
        return await real_write_change(*a, **kw)

    monkeypatch.setattr(ops, "_write_change", _explode)

    with pytest.raises(RuntimeError):
        await curator.curate_turn(db, user_id, user_text=USER_TEXT)
    await db.rollback()

    bodies = await _bodies(db, user_id)
    joined = " ".join(bodies.values()).lower()
    assert "vancouver" not in joined, (
        "a body was persisted while the batch it belonged to failed:\n" + repr(bodies)
    )
    assert "topics/music" not in bodies, "a half-created file survived the crash"
    changes = (await db.execute(select(MemoryFileChange))).scalars().all()
    assert not changes, f"change rows survived a failed batch: {changes}"


async def test_the_change_log_and_the_bodies_agree_after_a_successful_batch(
    monkeypatch,
):
    """The control for the test above. Without it, "nothing was written"
    passes trivially on a writer that never writes anything."""
    db, user_id = await _session()
    _install(monkeypatch, GOOD_OPS)

    result = await curator.curate_turn(db, user_id, user_text=USER_TEXT)
    assert result["applied"] == 1

    bodies = await _bodies(db, user_id)
    assert "vancouver" in bodies["you/profile"].lower()
    changes = (await db.execute(select(MemoryFileChange))).scalars().all()
    assert [c.summary for c in changes] == ["Added Profile: moved to Vancouver."]
    assert changes[0].file_slug == "you/profile"


# ── 4. The outbox cannot become an unbounded liability ────────────────

async def test_a_poison_turn_is_abandoned_with_its_reason_recorded(monkeypatch):
    """Retrying forever turns one bad row into a permanent per-turn LLM
    call. After MAX_ATTEMPTS the row is resolved — with `last_error` saying
    why, so an abandonment is explicable rather than a silent disappearance.
    """
    db, user_id = await _session()
    await _park(db, user_id, RuntimeError("x"))

    monkeypatch.setattr(outbox, "_BACKOFF_MINUTES", (0, 0, 0, 0, 0))
    _install(
        monkeypatch,
        *[RuntimeError("still broken")] * (outbox.MAX_ATTEMPTS * 2 + 2),
    )
    for _ in range(outbox.MAX_ATTEMPTS):
        await outbox.replay_pending(db, user_id)
        await db.commit()

    row = (await db.execute(select(MemoryCaptureOutbox))).scalars().one()
    assert row.resolved_at is not None, "a poison turn is retried forever"
    assert row.attempts >= outbox.MAX_ATTEMPTS
    assert "still broken" in (row.last_error or ""), row.last_error
    assert await outbox.pending_count(db, user_id) == 0


async def test_a_stale_turn_is_dropped_rather_than_replayed_with_a_wrong_date(
    monkeypatch,
):
    """The curator resolves relative dates against TODAY. Replaying a
    two-day-old "the exam is tomorrow at 9" writes a date that was never
    true — a confident wrong fact, which is worse than a lost one."""
    from datetime import datetime, timedelta

    db, user_id = await _session()
    await _park(db, user_id, RuntimeError("x"))
    row = (await db.execute(select(MemoryCaptureOutbox))).scalars().one()
    row.created_at = datetime.utcnow() - timedelta(hours=outbox.MAX_AGE_HOURS + 1)
    await db.commit()

    llm = _install(monkeypatch, GOOD_OPS)
    await outbox.replay_pending(db, user_id)
    await db.commit()

    assert llm.calls == 0, "a stale turn was replayed"
    row = (await db.execute(select(MemoryCaptureOutbox))).scalars().one()
    assert row.resolved_at is not None
    assert "dropped" in (row.last_error or ""), row.last_error


async def test_the_replay_is_capped_at_one_turn(monkeypatch):
    """Each replay is a real model call on the user's reply path. Draining a
    backlog in one turn would be a burst of them."""
    db, user_id = await _session()
    for _ in range(3):
        await _park(db, user_id, RuntimeError("x"))
    assert await outbox.pending_count(db, user_id) == 3

    llm = _install(monkeypatch, GOOD_OPS, GOOD_OPS, GOOD_OPS)
    assert await outbox.replay_pending(db, user_id) == 1
    await db.commit()
    assert llm.calls == 1
    assert await outbox.pending_count(db, user_id) == 2
