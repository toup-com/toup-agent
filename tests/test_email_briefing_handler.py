"""Gate 2 tests: email briefing handler + runner retry/nudge integration.

All MCP and LLM dependencies are mocked. No real Gmail or Haiku calls.
The tests exercise:
  - Bootstrap vs steady-state query window selection
  - Empty-result path (posts "no new emails", no watermark advance)
  - Success path (posts summary, watermark advances)
  - Reauth detection from MCP `kind=reauth_required` envelope
  - Runner retry loop: 3 attempts with tiny delays
  - Failure nudge after retry exhaustion
  - Reauth nudge gated by per-day idempotency (only one row per day)
  - Watermark advances on success only, never on failure
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

import pytest
import pytest_asyncio
from sqlalchemy import MetaData, Table, inspect, select


# ── Helpers ──────────────────────────────────────────────────────────


@pytest_asyncio.fixture(autouse=True)
async def _ensure_routine_write_tables(_reset_database):
    """`init_db()` skips messages / conversations / day_chats under SQLite
    because they declare a `pgvector` embedding column SQLite can't
    render. Routine tests need to write rows to those tables. We
    recreate them here with vector columns coerced to LargeBinary so
    SQLAlchemy's INSERTs (which still include those columns because the
    ORM sees pgvector at Python-import time) line up with the on-disk
    schema. Tests don't exercise vector behaviour — coercion is fine.

    Depends on `_reset_database` (from conftest) so this runs AFTER it
    has built every other table the test session needs."""
    from sqlalchemy import Column, LargeBinary

    from app.db.database import engine
    from app.db.models.base import Base

    async with engine.connect() as conn:
        existing = set(await conn.run_sync(lambda c: inspect(c).get_table_names()))

    needed = ("day_chats", "conversations", "messages")
    if all(n in existing for n in needed):
        return

    md = MetaData()
    for t_name in needed:
        if t_name in existing:
            continue
        orig = Base.metadata.tables.get(t_name)
        if orig is None:
            continue
        cols = []
        for c in orig.columns:
            if "vector" in str(c.type).lower():
                # Replace with a SQLite-friendly opaque blob — INSERT will
                # write NULL since the ORM has nothing to send.
                cols.append(Column(c.name, LargeBinary, nullable=True))
            else:
                cols.append(c._copy())
        Table(t_name, md, *cols)
    async with engine.begin() as conn:
        await conn.run_sync(md.create_all, checkfirst=True)


async def _make_user(timezone: str = "UTC") -> str:
    from app.db import async_session_maker
    from app.db.models import User

    user_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(
            id=user_id,
            email=f"{user_id}@routine-test.local",
            hashed_password="x",
            name="T",
            timezone=timezone,
        ))
        await db.commit()
    return user_id


async def _make_routine(
    user_id: str,
    *,
    kind: str = "email_briefing",
    last_state: Optional[dict] = None,
    config: Optional[dict] = None,
) -> str:
    from app.db import async_session_maker
    from app.db.models import Routine

    rid = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(Routine(
            id=rid, user_id=user_id, kind=kind, enabled=True,
            schedule_cron_local="0 7 * * *", last_status="never_run",
            config_json=config, last_state_json=last_state,
        ))
        await db.commit()
    return rid


class _Result:
    """Minimal shape that imitates fastmcp.CallToolResult — the handler
    reads `.structured_content` (or camelCase variant)."""
    def __init__(self, envelope: dict):
        self.structured_content = envelope


class _FakeMCP:
    """Test double for fastmcp.Client. Each entry in `responses` is a
    (tool_name, expected_args_subset, envelope) tuple consumed in order
    via call_tool. If `expected_args_subset` is None, args are ignored.

    Implements async-context manager so the handler's
    `async with mcp_client:` pattern (mirrors tool_executor.py:381)
    works against this double too."""

    def __init__(self, responses: list[tuple[str, Optional[dict], dict]]):
        self._responses = list(responses)
        self.calls: list[tuple[str, dict]] = []
        self.context_entries = 0

    async def __aenter__(self):
        self.context_entries += 1
        return self

    async def __aexit__(self, *exc):
        return False

    async def call_tool(self, name: str, args: dict) -> _Result:
        self.calls.append((name, dict(args)))
        if not self._responses:
            raise RuntimeError(f"_FakeMCP: no more responses queued (call: {name})")
        exp_name, exp_args, envelope = self._responses.pop(0)
        assert name == exp_name, f"expected {exp_name}, got {name}"
        if exp_args is not None:
            for k, v in exp_args.items():
                assert args.get(k) == v, f"arg {k}: expected {v}, got {args.get(k)}"
        return _Result(envelope)


def _ok(payload: dict) -> dict:
    """Build a `kind=ok` envelope mirroring connector_mcp.py."""
    return {"kind": "ok", "content": json.dumps(payload)}


def _reauth() -> dict:
    return {
        "kind": "reauth_required",
        "reauth_url": "https://toup.ai/agent/integrations",
        "message": "[reauth_required] Reconnect at /agent/integrations and try again.",
    }


def _list_messages_ok(ids: list[str], result_size: Optional[int] = None) -> dict:
    return _ok({
        "messages": [{"id": i, "threadId": i + "_t"} for i in ids],
        "result_size": result_size if result_size is not None else len(ids),
    })


def _get_message_ok(mid: str, *, from_addr: str = "boss@acme.com",
                    subject: str = "Quick question", body: str = "Can you review?",
                    internal_date_ms: int = 1715300000000) -> dict:
    return _ok({
        "id": mid,
        "threadId": mid + "_t",
        "headers": {
            "From": from_addr, "Subject": subject,
            "Date": "Sun, 11 May 2026 09:00:00 -0400",
        },
        "snippet": body[:80],
        "body": body,
        "internalDate": internal_date_ms,
    })


async def _fake_llm(**kwargs) -> Optional[str]:
    """Default fake LLM — returns a predictable summary."""
    return "**1 new email overnight, 1 needs a reply.**\n- ⚑ Quick question from boss@acme.com"


async def _failing_llm(**kwargs) -> Optional[str]:
    return None  # simulate timeout/auth fail


class _RecordingWriter:
    """Test writer that captures what would have been written."""
    def __init__(self):
        self.calls: list[dict] = []

    async def __call__(self, db, *, user_id, content, source, **kwargs):
        msg_id = str(uuid.uuid4())
        self.calls.append({
            "user_id": user_id, "content": content, "source": source, **kwargs,
        })
        return msg_id, "fake_day_chat_id"


# ── Handler tests ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_handler_bootstrap_window_uses_newer_than_1d():
    """No watermark → query is `newer_than:1d`."""
    from app.agent.routines.email_briefing_handler import EmailBriefingHandler
    from app.db import async_session_maker
    from app.db.models import Routine

    user_id = await _make_user("UTC")
    rid = await _make_routine(user_id, last_state=None)
    async with async_session_maker() as db:
        routine = await db.get(Routine, rid)

    mcp = _FakeMCP([
        ("gmail__list_messages", {"q": "newer_than:1d", "max_results": 50}, _list_messages_ok([])),
    ])
    writer = _RecordingWriter()
    handler = EmailBriefingHandler(mcp_client=mcp, llm_fn=_fake_llm, writer=writer)

    async with async_session_maker() as db:
        result = await handler.execute(routine, None, db)

    assert result.status == "success"
    assert result.emails_fetched == 0
    assert mcp.calls[0][1]["q"] == "newer_than:1d"
    # Empty result still posts a Message ("No new emails since the last briefing.")
    assert len(writer.calls) == 1
    assert writer.calls[0]["content"] == "No new emails since the last briefing."
    # Watermark MUST NOT advance — no new max(internal_date) seen.
    assert result.new_watermark is None


@pytest.mark.asyncio
async def test_handler_steady_state_window_uses_hours_since_watermark():
    """Watermark set 6h ago → query is roughly `newer_than:7h` (6h + 1h buffer)."""
    from app.agent.routines.email_briefing_handler import EmailBriefingHandler
    from app.db import async_session_maker
    from app.db.models import Routine

    user_id = await _make_user("UTC")
    # Pure-ms math to dodge the naive-utcnow().timestamp() trap (same fix
    # the handler now uses internally).
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    six_hours_ago_ms = now_ms - 6 * 3_600_000
    rid = await _make_routine(user_id, last_state={"last_processed_internal_date": six_hours_ago_ms})
    async with async_session_maker() as db:
        routine = await db.get(Routine, rid)

    mcp = _FakeMCP([
        ("gmail__list_messages", None, _list_messages_ok([])),
    ])
    handler = EmailBriefingHandler(mcp_client=mcp, llm_fn=_fake_llm, writer=_RecordingWriter())

    async with async_session_maker() as db:
        await handler.execute(routine, None, db)

    q = mcp.calls[0][1]["q"]
    assert q.startswith("newer_than:") and q.endswith("h"), f"q={q}"
    hours = int(q.split(":")[1][:-1])
    assert 6 <= hours <= 8, f"expected ~7h window, got {hours}h"


@pytest.mark.asyncio
async def test_handler_steady_state_caps_at_one_week():
    """Watermark from 30 days ago → fall back to bootstrap window."""
    from app.agent.routines.email_briefing_handler import EmailBriefingHandler
    from app.db import async_session_maker
    from app.db.models import Routine

    user_id = await _make_user("UTC")
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    month_ago_ms = now_ms - 30 * 24 * 3_600_000
    rid = await _make_routine(user_id, last_state={"last_processed_internal_date": month_ago_ms})
    async with async_session_maker() as db:
        routine = await db.get(Routine, rid)

    mcp = _FakeMCP([("gmail__list_messages", None, _list_messages_ok([]))])
    handler = EmailBriefingHandler(mcp_client=mcp, llm_fn=_fake_llm, writer=_RecordingWriter())

    async with async_session_maker() as db:
        await handler.execute(routine, None, db)

    assert mcp.calls[0][1]["q"] == "newer_than:1d"


@pytest.mark.asyncio
async def test_handler_success_path_writes_summary_and_advances_watermark():
    """list → get_message × N → LLM → write Message → watermark advance."""
    from app.agent.routines.email_briefing_handler import EmailBriefingHandler
    from app.db import async_session_maker
    from app.db.models import Routine

    user_id = await _make_user("UTC")
    rid = await _make_routine(user_id)
    async with async_session_maker() as db:
        routine = await db.get(Routine, rid)

    mcp = _FakeMCP([
        ("gmail__list_messages", None, _list_messages_ok(["m1", "m2"])),
        ("gmail__get_message", {"message_id": "m1"}, _get_message_ok("m1", internal_date_ms=1715300000000)),
        ("gmail__get_message", {"message_id": "m2"}, _get_message_ok("m2", internal_date_ms=1715400000000)),
    ])
    writer = _RecordingWriter()
    handler = EmailBriefingHandler(mcp_client=mcp, llm_fn=_fake_llm, writer=writer)

    async with async_session_maker() as db:
        result = await handler.execute(routine, None, db)

    assert result.status == "success"
    assert result.emails_fetched == 2
    assert result.summary_message_id is not None
    assert result.new_watermark == {
        "last_processed_internal_date": 1715400000000,
        "last_processed_at": result.new_watermark["last_processed_at"],  # tautology — just verify the field is present
    }
    assert len(writer.calls) == 1
    assert "1 new email overnight" in writer.calls[0]["content"]
    assert writer.calls[0]["source"] == "email_briefing"


@pytest.mark.asyncio
async def test_handler_reauth_short_circuits_before_get_message():
    """A `kind=reauth_required` on list_messages → status=skipped_reauth,
    NO get_message calls, NO Message written by the handler."""
    from app.agent.routines.email_briefing_handler import EmailBriefingHandler
    from app.db import async_session_maker
    from app.db.models import Routine

    user_id = await _make_user("UTC")
    rid = await _make_routine(user_id)
    async with async_session_maker() as db:
        routine = await db.get(Routine, rid)

    mcp = _FakeMCP([("gmail__list_messages", None, _reauth())])
    writer = _RecordingWriter()
    handler = EmailBriefingHandler(mcp_client=mcp, llm_fn=_fake_llm, writer=writer)

    async with async_session_maker() as db:
        result = await handler.execute(routine, None, db)

    assert result.status == "skipped_reauth"
    assert result.error_class == "reauth_required"
    assert len(mcp.calls) == 1  # only list_messages, no follow-up get_message
    assert len(writer.calls) == 0  # nudge is the runner's job, not the handler's


@pytest.mark.asyncio
async def test_handler_llm_returns_none_is_failed():
    """LLM timeout / parse error → status=failed, error_class=llm_returned_none."""
    from app.agent.routines.email_briefing_handler import EmailBriefingHandler
    from app.db import async_session_maker
    from app.db.models import Routine

    user_id = await _make_user("UTC")
    rid = await _make_routine(user_id)
    async with async_session_maker() as db:
        routine = await db.get(Routine, rid)

    mcp = _FakeMCP([
        ("gmail__list_messages", None, _list_messages_ok(["m1"])),
        ("gmail__get_message", None, _get_message_ok("m1")),
    ])
    handler = EmailBriefingHandler(mcp_client=mcp, llm_fn=_failing_llm, writer=_RecordingWriter())

    async with async_session_maker() as db:
        result = await handler.execute(routine, None, db)

    assert result.status == "failed"
    assert result.error_class == "llm_returned_none"


@pytest.mark.asyncio
async def test_handler_individual_get_message_failure_is_skipped_not_fatal():
    """One bad message → handler continues with the rest. Better to ship
    a briefing with N-1 emails than to fail the whole window."""
    from app.agent.routines.email_briefing_handler import EmailBriefingHandler
    from app.db import async_session_maker
    from app.db.models import Routine

    user_id = await _make_user("UTC")
    rid = await _make_routine(user_id)
    async with async_session_maker() as db:
        routine = await db.get(Routine, rid)

    mcp = _FakeMCP([
        ("gmail__list_messages", None, _list_messages_ok(["m1", "m2"])),
        ("gmail__get_message", {"message_id": "m1"}, {"kind": "tool_error", "message": "transient"}),
        ("gmail__get_message", {"message_id": "m2"}, _get_message_ok("m2")),
    ])
    handler = EmailBriefingHandler(mcp_client=mcp, llm_fn=_fake_llm, writer=_RecordingWriter())

    async with async_session_maker() as db:
        result = await handler.execute(routine, None, db)

    assert result.status == "success"
    assert result.emails_fetched == 2  # total seen in window
    # internal: only 1 email made it through to the LLM (the other was skipped)


# ── Runner retry / nudge tests ───────────────────────────────────────


def _set_kind_handler(kind: str, handler):
    """Install a kind handler for the duration of a test, returning the
    previous handler so it can be restored."""
    from app.agent.routines.registry import KIND_HANDLERS
    prev = KIND_HANDLERS.get(kind)
    KIND_HANDLERS[kind] = handler
    return prev


def _restore_kind_handler(kind: str, prev):
    from app.agent.routines.registry import KIND_HANDLERS
    if prev is None:
        KIND_HANDLERS.pop(kind, None)
    else:
        KIND_HANDLERS[kind] = prev


class _ScriptedHandler:
    """Returns successive RoutineResults from a list. After exhausting
    the list, repeats the last entry — so tests can assert "fails 3x
    then succeeds" or "fails forever"."""
    def __init__(self, kind: str, results: list):
        self.kind = kind
        self._results = list(results)
        self.call_count = 0

    async def execute(self, routine, run, db):
        self.call_count += 1
        idx = min(self.call_count - 1, len(self._results) - 1)
        return self._results[idx]


@pytest.mark.asyncio
async def test_runner_retries_three_times_then_writes_failure_nudge():
    """3 failed attempts → failure nudge posted, run row finalized as
    failed, watermark untouched."""
    from app.agent.routines import RoutineRunner
    from app.agent.routines.base_handler import RoutineResult
    from app.db import async_session_maker
    from app.db.models import Routine, RoutineRun, Message

    # Enable email_briefing via the flag for this test so _load_enabled
    # would register the routine. (We call _fire directly so registration
    # doesn't matter, but be defensive.)
    from app.config import settings
    settings.routines_email_briefing_enabled = True

    user_id = await _make_user("UTC")
    rid = await _make_routine(user_id, last_state={"last_processed_internal_date": 1715200000000})

    scripted = _ScriptedHandler("email_briefing", [
        RoutineResult(status="failed", error_class="transient_500", error_detail="503"),
        RoutineResult(status="failed", error_class="transient_500", error_detail="503"),
        RoutineResult(status="failed", error_class="transient_500", error_detail="503"),
    ])
    prev = _set_kind_handler("email_briefing", scripted)
    rr = RoutineRunner(retry_delays=(0.01, 0.01, 0.01))
    try:
        await rr._fire(rid)
        assert scripted.call_count == 3

        async with async_session_maker() as db:
            run = (await db.execute(select(RoutineRun).where(RoutineRun.routine_id == rid))).scalar_one()
            assert run.status == "failed"
            assert run.error_class == "transient_500"
            assert run.attempt == 3
            assert run.summary_message_id is not None  # nudge message

            # Watermark must NOT advance on failure.
            routine = await db.get(Routine, rid)
            assert routine.last_state_json == {"last_processed_internal_date": 1715200000000}
            assert routine.last_status == "failed"

            # The failure nudge Message exists.
            msg = await db.get(Message, run.summary_message_id)
            assert msg is not None
            assert msg.channel == "routine"
            assert msg.source == "email_briefing"
            assert "Couldn't reach Gmail" in msg.content
    finally:
        _restore_kind_handler("email_briefing", prev)
        settings.routines_email_briefing_enabled = False


@pytest.mark.asyncio
async def test_runner_retry_succeeds_on_second_attempt_advances_watermark():
    """Fail once, succeed → watermark advances, status=success, no nudge."""
    from app.agent.routines import RoutineRunner
    from app.agent.routines.base_handler import RoutineResult
    from app.db import async_session_maker
    from app.db.models import Routine, RoutineRun

    from app.config import settings
    settings.routines_email_briefing_enabled = True

    user_id = await _make_user("UTC")
    rid = await _make_routine(user_id, last_state=None)

    new_watermark = {"last_processed_internal_date": 9999999999000}
    scripted = _ScriptedHandler("email_briefing", [
        RoutineResult(status="failed", error_class="transient", error_detail="x"),
        RoutineResult(
            status="success", emails_fetched=3,
            summary_message_id="fake-msg-id",
            new_watermark=new_watermark,
        ),
    ])
    prev = _set_kind_handler("email_briefing", scripted)
    rr = RoutineRunner(retry_delays=(0.01, 0.01, 0.01))
    try:
        await rr._fire(rid)
        assert scripted.call_count == 2

        async with async_session_maker() as db:
            run = (await db.execute(select(RoutineRun).where(RoutineRun.routine_id == rid))).scalar_one()
            assert run.status == "success"
            assert run.emails_fetched == 3
            assert run.attempt == 2
            assert run.summary_message_id == "fake-msg-id"

            routine = await db.get(Routine, rid)
            assert routine.last_state_json == new_watermark
            assert routine.last_status == "success"
    finally:
        _restore_kind_handler("email_briefing", prev)
        settings.routines_email_briefing_enabled = False


@pytest.mark.asyncio
async def test_runner_skipped_reauth_writes_nudge_no_retry():
    """skipped_reauth short-circuits — no retry, nudge posted, attempt=1."""
    from app.agent.routines import RoutineRunner
    from app.agent.routines.base_handler import RoutineResult
    from app.db import async_session_maker
    from app.db.models import RoutineRun, Message

    from app.config import settings
    settings.routines_email_briefing_enabled = True

    user_id = await _make_user("UTC")
    rid = await _make_routine(user_id)

    scripted = _ScriptedHandler("email_briefing", [
        RoutineResult(status="skipped_reauth", error_class="reauth_required",
                      error_detail="vault status=reauth_required"),
    ])
    prev = _set_kind_handler("email_briefing", scripted)
    rr = RoutineRunner(retry_delays=(0.01, 0.01, 0.01))
    try:
        await rr._fire(rid)
        assert scripted.call_count == 1  # no retry

        async with async_session_maker() as db:
            run = (await db.execute(select(RoutineRun).where(RoutineRun.routine_id == rid))).scalar_one()
            assert run.status == "skipped_reauth"
            assert run.attempt == 1
            assert run.summary_message_id is not None
            msg = await db.get(Message, run.summary_message_id)
            assert "Reconnect Gmail" in msg.content
            assert msg.source == "email_briefing"
            assert msg.channel == "routine"
    finally:
        _restore_kind_handler("email_briefing", prev)
        settings.routines_email_briefing_enabled = False


@pytest.mark.asyncio
async def test_runner_nudge_gated_by_per_day_idempotency():
    """A second fire on the same local_date hits the UNIQUE and exits
    silently — so no second nudge, regardless of status."""
    from app.agent.routines import RoutineRunner
    from app.agent.routines.base_handler import RoutineResult
    from app.db import async_session_maker
    from app.db.models import RoutineRun, Message

    from app.config import settings
    settings.routines_email_briefing_enabled = True

    user_id = await _make_user("UTC")
    rid = await _make_routine(user_id)

    scripted = _ScriptedHandler("email_briefing", [
        RoutineResult(status="skipped_reauth", error_class="reauth_required", error_detail="x"),
    ])
    prev = _set_kind_handler("email_briefing", scripted)
    rr = RoutineRunner(retry_delays=(0.01, 0.01, 0.01))
    try:
        await rr._fire(rid)
        await rr._fire(rid)  # second fire same local_date

        async with async_session_maker() as db:
            runs = (await db.execute(select(RoutineRun).where(RoutineRun.routine_id == rid))).scalars().all()
            assert len(runs) == 1  # idempotency held

            # Only one nudge Message was written.
            msgs = (await db.execute(
                select(Message).where(Message.source == "email_briefing", Message.channel == "routine")
            )).scalars().all()
            assert len(msgs) == 1
    finally:
        _restore_kind_handler("email_briefing", prev)
        settings.routines_email_briefing_enabled = False
