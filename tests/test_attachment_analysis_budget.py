"""Budget-aware document analysis: preflight, honest stop, retry, resume (spec B7, fix pass v3).

Incident 2026-09-28: a spent monthly model budget turned a six-page handout
into "(0/6 6 pages processed)" plus six "Could not analyze this page." lines,
a retry failed the same way, and nothing tried again after the reset. Every
name, date, zone and amount below is synthetic.

Lane: platform
Runs in the platform sweep (RUN_MODE=platform) and in the monolith lane without
a COVERAGE_DEBT entry: the AGENT_ONLY tables it touches are created by the
`_reset_database` override below (pattern: tests/test_job_runner.py). Tests
that need the production (agent-DB) checkpoint path patch `AA._agent_mode`
and `voice_tasks._agent_claims_allowed` instead of relying on RUN_MODE.
"""
from __future__ import annotations

import asyncio
import copy
import io
import json
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone

import httpx
import pytest
import pytest_asyncio
from openai import APIStatusError, AsyncOpenAI, RateLimitError
from sqlalchemy import select, update

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "fixtures"))
import attachments as fx  # noqa: E402

from app.agent import attachment_analysis as AA  # noqa: E402
from app.api import chat_attachments as CA  # noqa: E402
from app.config import settings  # noqa: E402
from app.services import bundle_client  # noqa: E402
from app.services.budget_refusal import REASON  # noqa: E402
from app.services.file_storage import get_storage_backend  # noqa: E402

USER = "budget-user-0001"
AID = "b" * 32
TASK = "Summarize every page"
FA_TASK = "هر صفحه را خلاصه کن"
NAME = "Handout.pdf"
SESSION, SESSION_B = "budget-session-0001", "budget-session-0002"
ANCHOR_1, ANCHOR_2 = "budget-anchor-0001", "budget-anchor-0002"
TZ = "America/Chicago"                          # a synthetic user zone
PERIOD_START = "2026-09-20T20:45:00+00:00"
PERIOD_END = "2026-10-20T20:45:00+00:00"        # a synthetic window end, as iso_utc writes it
NAIVE_PERIOD_END = "2026-10-20T20:45:00"        # the same instant from an offset-less proxy
PAST_END = "2026-09-01T00:00:00+00:00"          # a reset that has already passed
FROZEN = datetime(2026, 9, 29, 12, 0, tzinfo=timezone.utc)

INCIDENT_SNAPSHOT = (
    "Handout.pdf — not read yet (0 of 6 pages)\n\n"
    "My monthly AI budget is used up, so I haven’t read any of this file. "
    "It resets on October 20, and I’ll try again automatically then. "
    "Your credits aren’t affected."
)
# The proxy's typed 429 detail (spec v3 S2: no spend or budget on the wire).
TYPED_DETAIL = {
    "error": REASON, "message": "Monthly openai budget exceeded", "provider": "openai",
    "period_start": PERIOD_START, "period_end": NAIVE_PERIOD_END,
}


# ── fixtures ─────────────────────────────────────────────────────────


@pytest_asyncio.fixture(autouse=True)
async def _reset_database():
    """Overrides conftest's: init_db() under RUN_MODE=platform skips AGENT_ONLY tables."""
    from app.db import drop_db, init_db
    from app.db.database import engine
    from app.db.models import AttachmentAnalysisJob, Conversation, DayChat, Message

    await init_db()
    async with engine.begin() as conn:
        for table in (DayChat.__table__, Conversation.__table__,
                      Message.__table__, AttachmentAnalysisJob.__table__):
            await conn.run_sync(lambda c, t=table: t.create(c, checkfirst=True))
    yield
    await drop_db()
    await engine.dispose()


@pytest.fixture(autouse=True)
def storage(tmp_path, monkeypatch):
    from app.agent._user_tz_cache import invalidate_cached_user_tz_with_day_chat

    fx.local_storage(tmp_path)                                   # as test_attachment_analysis.py
    AA._active.clear()
    AA._start_locks.clear()
    AA._verdict_waits.clear()                                    # every test reuses the same job keys
    invalidate_cached_user_tz_with_day_chat(USER)
    monkeypatch.setattr(AA, "_work_gate", asyncio.Semaphore(1))  # fresh per event loop
    monkeypatch.setattr(AA, "_agent_mode", lambda: False)        # file checkpoints unless a test opts in
    monkeypatch.setattr(AA, "PAGE_MODEL_CONCURRENCY", 1)         # deterministic attempt order
    monkeypatch.setattr("app.agent.voice_tasks._agent_claims_allowed", lambda: True)
    yield
    AA._active.clear()
    AA._start_locks.clear()
    AA._verdict_waits.clear()
    invalidate_cached_user_tz_with_day_chat(USER)
    from app.services import file_storage
    file_storage._backend = None


class _Preflight:
    """Stands in for `_budget_preflight` (unit-tested separately below). The
    default is what a check that fails open returns: not blocked, not known."""

    def __init__(self) -> None:
        self.verdict = {"blocked": False, "known": False, "period_end": None, "remaining_cents": None}
        self.calls = 0

    def block(self, period_end=PERIOD_END) -> None:
        """The platform's own verdict: spent."""
        self.verdict = {"blocked": True, "known": True, "period_end": period_end,
                        "remaining_cents": -12.5}

    def block_unknown(self, period_end=PERIOD_END) -> None:
        """An older platform (no openai_blocked) whose remaining figure is spent."""
        self.verdict = {"blocked": True, "known": False, "period_end": period_end,
                        "remaining_cents": -12.5}

    def admit(self) -> None:
        self.verdict = {"blocked": False, "known": True, "period_end": PERIOD_END,
                        "remaining_cents": 640.0}

    async def __call__(self) -> dict:
        self.calls += 1
        return dict(self.verdict)


@pytest.fixture
def preflight(monkeypatch):
    fake = _Preflight()
    monkeypatch.setattr(AA, "_budget_preflight", fake)
    return fake


class _FrozenDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        return FROZEN.astimezone(tz) if tz is not None else FROZEN.replace(tzinfo=None)

    @classmethod
    def utcnow(cls):
        return FROZEN.replace(tzinfo=None)


@pytest.fixture
def frozen(monkeypatch):
    """The module's clock at a fixed moment, so a reset date in delivered
    copy never goes stale. The DB lease clock stays real."""
    monkeypatch.setattr(AA, "datetime", _FrozenDateTime)
    return FROZEN


@pytest.fixture
def broadcasts(monkeypatch):
    from app.api import ws_chat

    seen: list[dict] = []

    async def capture(uid, event, exclude=None):
        seen.append(event)
        return 1

    monkeypatch.setattr(ws_chat, "broadcast_to_user", capture)
    return seen


@pytest.fixture
def bundle_mode(monkeypatch):
    monkeypatch.setattr(settings, "llm_mode", "bundle")
    monkeypatch.setattr(settings, "toup_token", "synthetic-token")
    # The normalizer does not re-run on assignment: give the full /api value.
    monkeypatch.setattr(settings, "platform_api_url", "http://platform.test/api")


# ── helpers ──────────────────────────────────────────────────────────


def budget_429(period_end=NAIVE_PERIOD_END, *, header=True, typed=True):
    """What openai 2.53.0 raises for the proxy's refusal (spec A2, v3 S2): a
    real 429 whose ``.body`` is the JSON and ``str()`` a python repr."""
    req = httpx.Request("POST", "http://platform.test/api/llm/openai/v1/chat/completions")
    body = ({"detail": dict(TYPED_DETAIL, period_end=period_end)} if typed
            else {"detail": "Monthly openai budget exceeded"})
    headers = {"Retry-After": "604800", "x-should-retry": "false"}
    if header:
        headers["X-Toup-Reason"] = REASON
    resp = httpx.Response(429, request=req, json=body, headers=headers)
    return RateLimitError(f"Error code: 429 - {body}", response=resp, body=body)


def credits_402(reason="insufficient_message_credits"):
    """The proxy's credit pre-flight refusal (llm_proxy proxy_chat)."""
    req = httpx.Request("POST", "http://platform.test/api/llm/openai/v1/chat/completions")
    body = {"detail": {"error": "out_of_credits", "reason": reason,
                       "bucket": "message", "balance_after": "0"}}
    resp = httpx.Response(402, request=req, json=body)
    return APIStatusError(f"Error code: 402 - {body}", response=resp, body=body)


class BudgetModel:
    """FakeModel twin (tests/test_attachment_analysis.py) that records every unit SENT."""

    def __init__(self, *, refuse=(), refuse_stage=None, period_end=NAIVE_PERIOD_END):
        self.calls: list[int] = []
        self.stages: list[str] = []
        self.refuse = set(refuse)
        self.refuse_stage = refuse_stage
        self.period_end = period_end

    async def page(self, task, number, count, native_text, image):
        self.calls.append(number)
        await asyncio.sleep(0)
        if number in self.refuse:
            raise budget_429(self.period_end)
        return f"Page {number}: evidence"

    async def text_chunk(self, task, number, count, text):
        self.calls.append(number)
        return f"Part {number}: evidence"

    async def _stage(self, stage, text):
        self.stages.append(stage)
        if self.refuse_stage == stage:
            raise budget_429(self.period_end)
        return text

    async def section(self, task, first, last, pages, unit_kind):
        return await self._stage("section", f"Section {first}-{last}")

    async def volume(self, task, first, last, sections, unit_kind):
        return await self._stage("volume", f"Volume {first}-{last}")

    async def overview(self, task, count, volumes, failed, unit_kind, selected_range=None):
        return await self._stage("overview", f"Overview of {count}")

    async def close(self):
        pass


def _pdf(monkeypatch, model, pages=6):
    """A stored PDF of ``pages`` pages whose model is ``model``; returns the
    list of model clients built (the preflight must stop the job before one)."""
    made: list[int] = []
    monkeypatch.setattr(AA, "inspect_pdf", lambda data: pages)
    monkeypatch.setattr(AA, "_native_texts", lambda data, count: [""] * count)
    monkeypatch.setattr(AA, "render_page", lambda data, index: b"webp")
    monkeypatch.setattr(AA, "_make_model", lambda: made.append(1) or model)
    return made


async def _stored():
    key = f"{USER}/{AID}.original"
    await get_storage_backend().put(key, b"%PDF-fake")
    rec = {"attachment_id": AID, "name": NAME, "mime": "application/pdf",
           "attachment": {"storage_path": key, "attachment_id": AID},
           "ingest": {"status": "truncated"}}
    await CA._write_json(CA._record_key(USER, AID), rec)
    return rec


DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def _synthetic_text(parts: int) -> str:
    """Synthetic prose that splits into exactly ``parts`` parts. Numbered, so
    a DOCX of it stays under the ingest layer's zip-ratio guard."""
    size = (parts - 1) * AA.TEXT_CHUNK_CHARS + 500
    sentences, total = [], 0
    while total < size:
        sentences.append(f"Synthetic sentence {len(sentences) + 1} of a budget test. ")
        total += len(sentences[-1])
    return "".join(sentences)[:size]


async def _stored_text(name: str, mime: str, parts: int):
    """A stored text document (DOCX or plain text) of ``parts`` parts."""
    text = _synthetic_text(parts)
    data = fx.docx_doc(text) if mime == DOCX else text.encode("utf-8")
    key = f"{USER}/{AID}.original"
    await get_storage_backend().put(key, data)
    rec = {"attachment_id": AID, "name": name, "mime": mime,
           "attachment": {"storage_path": key, "attachment_id": AID}, "ingest": {"status": "ok"}}
    await CA._write_json(CA._record_key(USER, AID), rec)
    return rec


def _agent_db(monkeypatch):
    """The production path: job state lives in attachment_analysis_jobs."""
    monkeypatch.setattr(AA, "_agent_mode", lambda: True)


async def _seed_user(tz=TZ):
    from app.db.database import async_session_maker
    from app.db.models import User

    async with async_session_maker() as db:
        db.add(User(id=USER, email="budget-user@example.test", hashed_password="x", timezone=tz))
        await db.commit()


async def _seed_chat(*anchors, days_ago=0):
    """The user's conversation, a day chat ``days_ago`` days back and the
    assistant replies (anchors) that a delivery files itself under."""
    from app.db.database import async_session_maker
    from app.db.models import Conversation, DayChat, Message

    day_id = f"budget-day-{days_ago:04d}"
    async with async_session_maker() as db:
        db.add(DayChat(id=day_id, user_id=USER, timezone=TZ,
                       local_date=date.today() - timedelta(days=days_ago)))
        # Messages reference day_chats by a bare FK (no relationship), so the
        # unit of work may order their INSERT first; Postgres enforces the FK
        # where SQLite does not. Flush the day chat before anything points at it.
        await db.flush()
        if await db.get(Conversation, SESSION) is None:
            db.add(Conversation(id=SESSION, user_id=USER, day_chat_id=day_id, channel="app"))
        for anchor in anchors:
            db.add(Message(id=anchor, conversation_id=SESSION, day_chat_id=day_id,
                           role="assistant", channel="app",
                           content="I’m analyzing the full file. I’ll post the result in this chat."))
        await db.commit()
    return day_id


async def _seed_conversation(session_id, day_id="budget-day-0000"):
    """A second chat of the same user (the day chat must already exist)."""
    from app.db.database import async_session_maker
    from app.db.models import Conversation

    async with async_session_maker() as db:
        db.add(Conversation(id=session_id, user_id=USER, day_chat_id=day_id, channel="app"))
        await db.commit()


def _capture_delivery(monkeypatch, tz_name=TZ):
    """Every message a delivery would post, rendered as `_deliver_to_target` does."""
    out: list[list[str]] = []

    async def fake(state, target):
        out.append(AA._delivery_parts(state, tz_name=tz_name,
                                      target_session=target.get("session_id")))
        return True

    monkeypatch.setattr(AA, "_deliver_to_target", fake)
    return out


def _capture_notify(monkeypatch):
    from app.services import agent_notify_client

    pushes: list[dict] = []

    async def fake_notify(**kwargs):
        pushes.append(kwargs)
        return "outbox-row"

    monkeypatch.setattr(agent_notify_client, "notify", fake_notify)
    return pushes


def _analysis_tasks() -> list[asyncio.Task]:
    current = asyncio.current_task()
    found = []
    for task in asyncio.all_tasks():
        if task is current or task.done():
            continue
        code = getattr(task.get_coro(), "cr_code", None)
        if code is not None and os.path.basename(code.co_filename) == "attachment_analysis.py":
            found.append(task)
    return found


async def _settle(timeout: float = 20.0) -> None:
    """Wait for every run and delivery the module scheduled, fire-and-forget
    ones included, and surface any that crashed."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while tasks := _analysis_tasks():
        remaining = deadline - loop.time()
        assert remaining > 0, f"analysis work still running: {tasks}"
        done, _pending = await asyncio.wait(tasks, timeout=remaining)
        for task in done:
            if not task.cancelled() and task.exception() is not None:
                raise task.exception()


async def _analysis_messages() -> dict:
    from app.db.database import async_session_maker
    from app.db.models import Message

    async with async_session_maker() as db:
        rows = (await db.execute(select(Message).where(
            Message.source == "attachment_analysis"))).scalars().all()
    return {m.id: m for m in rows}


def _message_id(analysis_id, attempt, part, session=SESSION):
    return AA._delivery_message_id({"user_id": USER, "attachment_id": AID,
                                    "analysis_id": analysis_id, "delivery_attempt": attempt},
                                   session, part)


async def _row(analysis_id):
    from app.db.database import async_session_maker
    from app.db.models import AttachmentAnalysisJob

    async with async_session_maker() as db:
        return await db.get(AttachmentAnalysisJob, AA._job_id(USER, AID, analysis_id))


async def _day_of(day_chat_id):
    from app.db.database import async_session_maker
    from app.db.models import DayChat

    async with async_session_maker() as db:
        row = await db.get(DayChat, day_chat_id)
        return row.local_date if row else None


def _stopped(analysis_id, *, read=(), pages=6, blocked_until=None, first_failed_at=None,
             auto_resume_count=0, targets=None, task=TASK):
    """A delivered budget stop, as run_analysis leaves it."""
    state = AA.new_state(USER, AID, analysis_id, task, NAME, "page", pages,
                         SESSION, "app", ANCHOR_1)
    state["pages"] = [{"page_number": n, "status": "completed", "summary": f"Page {n}: evidence"}
                      for n in read]
    state.update(
        status="partial" if read else "failed", stage="complete",
        error={"code": "analysis_budget_exceeded", "message": AA._STOP_MESSAGES["analysis_budget_exceeded"]},
        blocked_until=AA.iso_utc(blocked_until), units_attempted=len(state["pages"]),
        auto_resume_count=auto_resume_count,
        first_failed_at=(first_failed_at or datetime.now(timezone.utc)).isoformat(),
        delivered=True,
    )
    for target in state["delivery_targets"]:
        target["delivered"] = True
    if targets is not None:
        state["delivery_targets"] = targets
    return state


def _legacy_stop(analysis_id):
    """A budget stop the previous image recorded: no blocked_until, no
    first_failed_at or units_attempted, its old sentence."""
    legacy = _stopped(analysis_id)
    for field in ("blocked_until", "units_attempted", "auto_resume_count", "first_failed_at"):
        legacy.pop(field)
    legacy["error"] = {"code": "analysis_budget_exceeded", "message": (
        "The monthly analysis model budget has been reached. Retry after service access is restored.")}
    return legacy


async def _finish_row(analysis_id, *, delivered=True):
    """What a finished job's checkpoint does to the row (revision + 1).
    ``delivered=False`` is the worker's own terminal write: it delivers right
    after (in one process, once the start lock is free)."""
    from app.db.database import async_session_maker
    from app.db.models import AttachmentAnalysisJob

    async with async_session_maker() as db:
        row = await db.get(AttachmentAnalysisJob, AA._job_id(USER, AID, analysis_id))
        done = copy.deepcopy(dict(row.state_json))
        done.update(status="completed", stage="complete", overview="Overview of 6", delivered=delivered,
                    pages=[{"page_number": n, "status": "completed", "summary": f"Page {n}: evidence"}
                           for n in range(1, 7)])
        done["delivery_targets"] = [dict(t, delivered=delivered) for t in done["delivery_targets"]]
        row.state_json = done
        row.status = "completed"
        row.delivered = delivered
        row.revision = int(row.revision or 0) + 1
        await db.commit()


class _OtherProcessLocks(dict):
    """Another replica's start locks: no asyncio lock is shared across processes."""

    def setdefault(self, key, default=None):
        return asyncio.Lock()


def _pass_clock(monkeypatch):
    """Later reconcile passes run ``hours`` after the database's now (the
    pass clock only; leases keep the database's). Returns the setter."""
    offset = [timedelta(0)]
    real = AA._auto_resume_candidates

    async def shifted(db):
        now, candidates = await real(db)
        return now + offset[0], candidates

    monkeypatch.setattr(AA, "_auto_resume_candidates", shifted)

    def later(*, hours):
        offset[0] = timedelta(hours=hours)

    return later


def _first_parts(messages: dict) -> dict:
    """Part 0 of each delivery, by conversation."""
    return {m.conversation_id: m.content for m in messages.values()
            if json.loads(m.metadata_json or "{}").get("analysis_part") == 0}


# ── the incident, end to end ─────────────────────────────────────────


async def test_incident_spent_budget_reads_nothing_and_posts_one_honest_message(
        monkeypatch, preflight, frozen, broadcasts):
    """Screenshots 1-2 showed '(0/6 6 pages processed)' and six 'Could not
    analyze this page.' lines. Now: no model call at all, one message."""
    _agent_db(monkeypatch)
    model = BudgetModel()
    made = _pdf(monkeypatch, model)
    preflight.block(PERIOD_END)
    await _seed_user()
    await _seed_chat(ANCHOR_1)

    state = await AA.start_analysis(USER, AID, await _stored(), TASK, session_id=SESSION,
                                    channel="app", anchor_message_id=ANCHOR_1)
    await _settle()

    final = await AA.load_state_async(USER, AID, state["analysis_id"])
    assert model.calls == [] and made == []            # the model client is never even built
    assert preflight.calls == 1
    assert final["status"] == "failed"
    assert final["error"]["code"] == "analysis_budget_exceeded"
    assert final["blocked_until"] == PERIOD_END
    assert final["units_attempted"] == 0 and final["pages"] == []
    assert final["first_failed_at"]
    assert final["delivered"] is True and final["delivery_attempt"] == 0

    messages = await _analysis_messages()
    assert list(messages) == [_message_id(state["analysis_id"], 0, 0)]
    assert [m.content for m in messages.values()] == [INCIDENT_SNAPSHOT]
    assert [event["content"] for event in broadcasts] == [INCIDENT_SNAPSHOT]
    row = await _row(state["analysis_id"])            # fencing keys never reach the stored JSON
    assert not [key for key in row.state_json if key.startswith("_claim")]
    assert row.claim_token is None and row.status == "failed"


async def test_wire_spent_budget_is_asked_once_and_a_refusal_is_sent_once(
        monkeypatch, bundle_mode, frozen):
    """Real AnalysisModel and openai SDK over one mock platform: when blocked
    the only request is GET /llm/usage (and the job is stopped before it
    exists); when a stale gate admits, the proxy's typed refusal of the first
    page stops the job after exactly one POST."""
    wire: list[tuple[str, str]] = []
    usage = {"openai_blocked": True}

    def platform(request: httpx.Request) -> httpx.Response:
        wire.append((request.method, request.url.path))
        if request.url.path == "/api/llm/usage":
            return httpx.Response(200, json={
                "openai_monthly_cents": 1012.5, "openai_budget_cents": 1000,
                "openai_remaining_cents": -12.5, "openai_blocked": usage["openai_blocked"],
                "budget_exempt": False, "period_start": PERIOD_START, "period_end": PERIOD_END,
            })
        return httpx.Response(429, json={"detail": TYPED_DETAIL}, headers={
            "X-Toup-Reason": REASON, "x-should-retry": "false", "Retry-After": "604800"})

    monkeypatch.setattr(bundle_client, "_proxy_http_client", lambda timeout=120.0: httpx.AsyncClient(
        transport=httpx.MockTransport(platform), timeout=timeout))
    monkeypatch.setattr(AA, "inspect_pdf", lambda data: 6)
    monkeypatch.setattr(AA, "_native_texts", lambda data, count: [""] * count)
    monkeypatch.setattr(AA, "render_page", lambda data, index: b"webp")
    delivered = _capture_delivery(monkeypatch)
    rec = await _stored()

    state = await AA.start_analysis(USER, AID, rec, TASK, session_id=SESSION, channel="app")
    await _settle()
    assert wire == [("GET", "/api/llm/usage")]
    assert state["status"] == "failed" and state["blocked_until"] == PERIOD_END
    assert delivered == []                              # the caller's reply is the message

    wire.clear()
    usage["openai_blocked"] = False
    await AA.start_analysis(USER, AID, rec, TASK, retry_failed=True, session_id=SESSION, channel="app")
    await _settle()
    assert wire == [("GET", "/api/llm/usage"), ("GET", "/api/llm/usage"),
                    ("POST", "/api/llm/openai/v1/chat/completions")]
    final = AA.load_state(USER, AID, state["analysis_id"])
    assert final["status"] == "failed" and final["units_attempted"] == 0
    assert final["blocked_until"] == PERIOD_END        # from the refusal body, normalised
    assert final["pages"] == []                        # a refused call is not a failed page
    assert delivered == [[INCIDENT_SNAPSHOT]]


# ── a new job while the budget is spent (fix pass item 14) ──────────


async def test_a_new_job_while_the_platform_says_spent_is_stopped_before_it_runs(
        monkeypatch, bundle_mode, preflight, frozen, broadcasts):
    """The kickoff used to say "I’m analyzing the full file" seconds before
    the run's own check posted "not read yet". Now the stop is recorded when
    the job is created, and the caller's reply is the only message."""
    _agent_db(monkeypatch)
    model = BudgetModel()
    made = _pdf(monkeypatch, model)
    preflight.block(PERIOD_END)
    await _seed_user()
    await _seed_chat(ANCHOR_1, ANCHOR_2)
    rec = await _stored()

    kickoff = dict(retry_failed=True, session_id=SESSION, channel="app",
                   redeliver_when_blocked=False)       # agent_runner's kwargs
    state = await AA.start_analysis(USER, AID, rec, TASK, anchor_message_id=ANCHOR_1, **kickoff)
    assert AA._active == {}                             # nothing queued, nothing running
    await _settle()
    assert preflight.calls == 1 and model.calls == [] and made == []
    assert state["status"] == "failed" and state["error"]["code"] == "analysis_budget_exceeded"
    assert AA.blocked_reply_text(state, persian=False, tz_name=TZ) == (
        "I can’t read this file yet: my monthly AI budget is used up. It resets on October 20, "
        "and I’ll try again automatically then. Your credits aren’t affected.")
    row = await _row(state["analysis_id"])
    stored = row.state_json
    assert row.status == "failed" and row.delivered is True and row.claim_token is None
    assert stored["blocked_until"] == PERIOD_END and stored["units_attempted"] == 0
    assert stored["first_failed_at"] and stored["pages"] == [] and stored["delivery_attempt"] == 0
    assert [t["delivered"] for t in stored["delivery_targets"]] == [True]
    assert await _analysis_messages() == {} and broadcasts == []

    # Asked again while still spent: one answer per request, nothing posted.
    again = await AA.start_analysis(USER, AID, rec, TASK, anchor_message_id=ANCHOR_2, **kickoff)
    await _settle()
    assert AA.blocked_reply_text(again, persian=False, tz_name=TZ)
    assert await _analysis_messages() == {} and model.calls == [] and broadcasts == []


@pytest.mark.parametrize("verdict", ["older_platform_spent", "fail_open"])
async def test_a_new_job_is_stopped_up_front_only_on_the_platforms_own_verdict(
        monkeypatch, bundle_mode, preflight, frozen, verdict):
    """Without a boolean openai_blocked the start keeps today's behaviour:
    the job runs, and its own check (or the proxy) decides."""
    model = BudgetModel()
    _pdf(monkeypatch, model)
    delivered = _capture_delivery(monkeypatch)
    if verdict == "older_platform_spent":
        preflight.block_unknown(PERIOD_END)

    state = await AA.start_analysis(USER, AID, await _stored(), TASK, session_id=SESSION,
                                    channel="app")
    assert state["status"] == "queued"
    await _settle()
    final = AA.load_state(USER, AID, state["analysis_id"])
    assert preflight.calls == 2                         # at the start, then by the run
    if verdict == "older_platform_spent":
        assert model.calls == [] and final["status"] == "failed"
        assert delivered == [[INCIDENT_SNAPSHOT]]
    else:
        assert sorted(model.calls) == [1, 2, 3, 4, 5, 6] and final["status"] == "completed"


async def test_the_analyze_tool_answers_a_new_blocked_job_with_guidance(
        monkeypatch, bundle_mode, preflight, frozen, tmp_path):
    from app.agent.tool_executor import ToolExecutor

    model = BudgetModel()
    _pdf(monkeypatch, model)
    preflight.block(PERIOD_END)
    await _stored()
    executor = ToolExecutor(workspace=str(tmp_path))
    executor.set_user_id(USER)

    result = json.loads(await executor.execute("analyze_attachment", {"attachment_id": AID, "task": TASK}))
    await _settle()
    assert result["status"] == "failed" and result["units_completed"] == 0
    assert result["guidance"].startswith(
        "The monthly AI budget is used up (resets on October 20). Nothing more was read.")
    assert "running in the background" not in result["guidance"]
    assert result["error"]["message"] == "The AI budget is used up, so reading stopped. Credits are not affected."
    assert model.calls == [] and preflight.calls == 1 and AA._active == {}


# ── a text document stopped before it runs (follow-up F3.1) ─────────


_KICKOFF = dict(retry_failed=True, session_id=SESSION, channel="app",
                redeliver_when_blocked=False)                # agent_runner's kwargs


@pytest.mark.parametrize("name,mime", [("Notes.docx", DOCX), ("Notes.txt", "text/plain")],
                         ids=["docx", "txt"])
async def test_a_text_document_stopped_at_creation_is_counted_and_resumed_like_a_pdf(
        monkeypatch, bundle_mode, preflight, broadcasts, name, mime):
    """A text document asked for while the platform says the budget is spent
    never reaches the worker that splits it into parts. It used to be stored
    with no part count, which read as "too large": the first reconcile pass
    retired it and it was never resumed, while a PDF stopped the same way
    was. Its parts are now counted when the stop is recorded (locally, no
    model call), so the reply promises the retry and the first pass the
    platform admits after the reset resumes it, once, to completion."""
    from app.services.budget_refusal import reset_when_phrase

    _agent_db(monkeypatch)
    model, made = BudgetModel(), []
    monkeypatch.setattr(AA, "_make_model", lambda: made.append(1) or model)
    pushes = _capture_notify(monkeypatch)
    later = _pass_clock(monkeypatch)
    await _seed_user()
    await _seed_chat(ANCHOR_1)
    rec = await _stored_text(name, mime, parts=3)
    reset = AA.iso_utc(datetime.now(timezone.utc) + timedelta(days=20))
    preflight.block(reset)

    state = await AA.start_analysis(USER, AID, rec, "Summarize this document",
                                    anchor_message_id=ANCHOR_1, **_KICKOFF)
    await _settle()
    analysis_id = state["analysis_id"]
    assert AA._active == {} and model.calls == [] and made == [] and preflight.calls == 1
    row = await _row(analysis_id)
    stored = row.state_json
    assert row.status == "failed" and stored["unit_kind"] == "chunk"
    assert (stored["page_count"], stored["selected_end"]) == (3, 3)      # counted, not read
    assert stored["pages"] == [] and stored["units_attempted"] == 0 and stored["blocked_until"] == reset
    assert AA._pending_model_calls(stored) == 3 + 1 + 1 + 1               # parts, section, part, overview
    when = reset_when_phrase(reset, TZ, datetime.now(timezone.utc))
    assert AA.blocked_reply_text(state, persian=False, tz_name=TZ) == (
        f"I can’t read this file yet: my monthly AI budget is used up. It resets on {when}, "
        "and I’ll try again automatically then. Your credits aren’t affected.")
    assert AA._delivery_parts(state, tz_name=TZ) == [
        f"{name} — not read yet (0 of 3 parts)\n\n"
        f"My monthly AI budget is used up, so I haven’t read any of this file. It resets on {when}, "
        "and I’ll try again automatically then. Your credits aren’t affected."]
    assert AA.blocked_guidance(state).endswith("It will be retried automatically after the reset.")

    await AA.reconcile_local_analyses()                # before the reset: nothing to do
    await _settle()
    assert (await _row(analysis_id)).revision == row.revision and preflight.calls == 1

    later(hours=20 * 24 + 1)                           # after the reset; the platform admits
    preflight.admit()
    await AA.reconcile_local_analyses()
    await _settle()
    final = await AA.load_state_async(USER, AID, analysis_id)
    assert final["status"] == "completed", final.get("error")
    assert final["auto_resume_count"] == 1 and "auto_resume_retired" not in final
    assert model.calls == [1, 2, 3] and model.stages == ["section", "volume", "overview"]
    assert len(made) == 1 and preflight.calls == 3     # the pass asked once, then the run
    assert _first_parts(await _analysis_messages()) == {SESSION: (
        f"Earlier you asked me to read {name}. My AI budget has reset, so here is the result.\n\n"
        f"Analysis of {name} (3 parts)\n\nOverview of 3")}
    assert [(p["title"], p["body"]) for p in pushes] == [
        ("Your file is ready", f"I finished reading {name}.")]

    await AA.reconcile_local_analyses()                # resumed exactly once
    await _settle()
    assert model.calls == [1, 2, 3] and preflight.calls == 3 and len(pushes) == 1


async def test_a_text_document_too_large_to_resume_says_so_and_retires(
        monkeypatch, bundle_mode, preflight):
    """Counted at the stop, 36 parts need 41 model calls (36 parts, 3
    sections, a part synthesis, the overview), more than one automatic
    resume may spend: nothing is promised, the copy says what was not read,
    and the reconciler retires it as too_large for that true reason."""
    from app.services.budget_refusal import reset_when_phrase

    _agent_db(monkeypatch)
    model, made = BudgetModel(), []
    monkeypatch.setattr(AA, "_make_model", lambda: made.append(1) or model)
    await _seed_user()
    await _seed_chat(ANCHOR_1)
    rec = await _stored_text("Notes.docx", DOCX, parts=36)
    reset = AA.iso_utc(datetime.now(timezone.utc) + timedelta(days=20))
    preflight.block(reset)

    state = await AA.start_analysis(USER, AID, rec, "Summarize this document",
                                    anchor_message_id=ANCHOR_1, **_KICKOFF)
    await _settle()
    stored = (await _row(state["analysis_id"])).state_json
    assert (stored["page_count"], stored["selected_end"]) == (36, 36)
    assert AA._pending_model_calls(stored) == 36 + 3 + 1 + 1 == AA.AUTO_RESUME_MAX_CALLS + 1
    when = reset_when_phrase(reset, TZ, datetime.now(timezone.utc))
    assert AA.blocked_reply_text(state, persian=False, tz_name=TZ) == (
        f"I can’t read this file yet: my monthly AI budget is used up. It resets on {when}; "
        "ask me again after that. Your credits aren’t affected.")
    assert AA._delivery_parts(state, tz_name=TZ) == [
        "Notes.docx — not read (0 of 36 parts)\n\n"
        f"My monthly AI budget is used up, so I haven’t read any of this file. It resets on {when}; "
        "ask me again after that. Your credits aren’t affected."]
    assert AA.blocked_guidance(state).endswith(
        "It will not be retried automatically; the user can ask again after the reset.")

    preflight.admit()
    await AA.reconcile_local_analyses()
    await _settle()
    row = await _row(state["analysis_id"])
    assert row.status == "failed" and row.state_json["auto_resume_retired"] == "too_large"
    assert row.state_json["auto_resume_count"] == AA.AUTO_RESUME_LIMIT
    assert model.calls == [] and made == [] and preflight.calls == 1 and await _analysis_messages() == {}


async def test_a_text_document_whose_parts_could_not_be_counted_is_left_for_the_user(
        monkeypatch, bundle_mode, preflight, broadcasts):
    """If the text cannot be extracted when the stop is recorded (here once,
    as if the process ran short of memory), the stop keeps no part count, as
    before, and nothing is promised. An unknown count is not "too large": the
    reconciler leaves the job exactly as it is (no platform or model call,
    no retirement), and the user's own request after the reset reads it. The
    first pass used to retire it as too_large."""
    from app.services.budget_refusal import reset_when_phrase

    _agent_db(monkeypatch)
    model = BudgetModel()
    monkeypatch.setattr(AA, "_make_model", lambda: model)
    later = _pass_clock(monkeypatch)
    await _seed_user()
    await _seed_chat(ANCHOR_1, ANCHOR_2)
    rec = await _stored_text("Notes.docx", DOCX, parts=2)
    real_units, failed = getattr(AA, "_text_units", None), []

    def extraction_fails_once(data, mime):
        if not failed:
            failed.append(mime)
            raise MemoryError("synthetic")
        return real_units(data, mime)

    monkeypatch.setattr(AA, "_text_units", extraction_fails_once, raising=False)
    reset = AA.iso_utc(datetime.now(timezone.utc) + timedelta(days=20))
    preflight.block(reset)
    task = "Summarize this document"

    state = await AA.start_analysis(USER, AID, rec, task, anchor_message_id=ANCHOR_1, **_KICKOFF)
    await _settle()
    analysis_id = state["analysis_id"]
    before = await _row(analysis_id)
    assert before.status == "failed" and before.state_json["page_count"] == 0   # stopped, not counted
    when = reset_when_phrase(reset, TZ, datetime.now(timezone.utc))
    assert AA.blocked_reply_text(state, persian=False, tz_name=TZ) == (
        f"I can’t read this file yet: my monthly AI budget is used up. It resets on {when}; "
        "ask me again after that. Your credits aren’t affected.")

    later(hours=20 * 24 + 1)                           # after the reset; the platform admits
    preflight.admit()
    for _ in range(2):
        await AA.reconcile_local_analyses()
        await _settle()
    row = await _row(analysis_id)
    assert row.revision == before.revision and row.state_json == before.state_json
    assert "auto_resume_retired" not in row.state_json
    assert preflight.calls == 1 and model.calls == [] and await _analysis_messages() == {}
    assert AA._resume_limits(row.state_json, datetime.now(timezone.utc)) == (False, "size_unknown")

    # The user asks again after the reset: the run counts the parts and reads them.
    await AA.start_analysis(USER, AID, rec, task, anchor_message_id=ANCHOR_2, **_KICKOFF)
    await _settle()
    final = await AA.load_state_async(USER, AID, analysis_id)
    assert final["status"] == "completed" and final["page_count"] == 2
    assert model.calls == [1, 2] and failed == [DOCX]


# ── how a stop is recorded and told ──────────────────────────────────


async def test_mid_run_refusal_with_three_workers_names_the_unread_gap(monkeypatch, preflight, frozen):
    """Page 3 is refused while page 4 is still in flight: pages 1, 2 and 4
    are kept and delivered; 3, 5 and 6 are 'not read', never 'failed'."""

    class GapModel(BudgetModel):
        def __init__(self):
            super().__init__()
            self.four_started = asyncio.Event()
            self.refused = asyncio.Event()

        async def page(self, task, number, count, native_text, image):
            self.calls.append(number)
            if number == 3:
                await asyncio.wait_for(self.four_started.wait(), 5)
                self.refused.set()
                raise budget_429()
            if number == 4:
                self.four_started.set()
            if number in (2, 4):
                await asyncio.wait_for(self.refused.wait(), 5)   # in flight across the refusal
            return f"Page {number}: evidence"

    monkeypatch.setattr(AA, "PAGE_MODEL_CONCURRENCY", 3)
    model = GapModel()
    _pdf(monkeypatch, model)
    delivered = _capture_delivery(monkeypatch)

    state = await AA.start_analysis(USER, AID, await _stored(), TASK, session_id=SESSION, channel="app")
    await _settle()

    final = AA.load_state(USER, AID, state["analysis_id"])
    assert sorted(model.calls) == [1, 2, 3, 4]         # 5 and 6 were never sent
    assert final["status"] == "partial"
    assert sorted(p["page_number"] for p in final["pages"]) == [1, 2, 4]
    assert {p["status"] for p in final["pages"]} == {"completed"}
    assert final["units_attempted"] == 3               # recorded results, not refused calls
    assert final["blocked_until"] == PERIOD_END        # from the refusal body, not Retry-After
    assert delivered == [[
        "Analysis of Handout.pdf (3 of 6 pages read)\n\n"
        "My monthly AI budget ran out partway through, so I read pages 1–2 and 4, "
        "but not pages 3 and 5–6. It resets on October 20, and I’ll try the unread pages "
        "again automatically then. Your credits aren’t affected.\n\n"
        "Page-by-page details for pages 1–2 and 4 follow.",
        "Handout.pdf — pages 1–6 of 6\n\n"
        "### Page 1\nPage 1: evidence\n\n"
        "### Page 2\nPage 2: evidence\n\n"
        "Page 3 was not read.\n\n"
        "### Page 4\nPage 4: evidence\n\n"
        "Pages 5–6 were not read.",
    ]]
    assert "Could not analyze" not in "".join(delivered[0])


async def test_stop_during_synthesis_keeps_every_page_and_retry_rebills_none(
        monkeypatch, preflight, frozen):
    model = BudgetModel(refuse_stage="section")
    _pdf(monkeypatch, model)
    delivered = _capture_delivery(monkeypatch)
    rec = await _stored()

    state = await AA.start_analysis(USER, AID, rec, TASK, session_id=SESSION, channel="app")
    await _settle()
    final = AA.load_state(USER, AID, state["analysis_id"])
    assert final["status"] == "partial" and final["units_attempted"] == 6
    assert final["error"]["code"] == "analysis_budget_exceeded"
    assert final["blocked_until"] == PERIOD_END
    [parts] = delivered
    assert parts[0] == (
        "Analysis of Handout.pdf (6 pages)\n\n"
        "I read all 6 pages, but the monthly AI budget ran out before I could write the "
        "combined summary. It resets on October 20, and I’ll try again automatically "
        "then. Your credits aren’t affected.\n\n"
        "Page-by-page details for pages 1–6 follow.")
    assert len(parts) == 2
    assert all(f"### Page {n}\nPage {n}: evidence" in parts[1] for n in range(1, 7))
    assert "not read" not in parts[1] and "Could not analyze" not in "".join(parts)
    assert model.stages == ["section"]
    assert AA.blocked_reply_text(final, persian=False, tz_name=TZ).startswith(
        "I can’t finish this file yet: my monthly AI budget is used up.")

    # The budget is back: the retry writes the summary without re-sending a page.
    model.refuse_stage = None
    preflight.admit()
    await AA.start_analysis(USER, AID, rec, TASK, retry_failed=True, session_id=SESSION, channel="app")
    await _settle()
    final = AA.load_state(USER, AID, state["analysis_id"])
    assert final["status"] == "completed" and final["overview"] == "Overview of 6"
    assert sorted(model.calls) == [1, 2, 3, 4, 5, 6]   # each page sent once
    assert model.stages == ["section", "section", "volume", "overview"]


async def test_a_budget_stop_stops_every_worker(monkeypatch, preflight, frozen):
    monkeypatch.setattr(AA, "PAGE_MODEL_CONCURRENCY", 3)
    model = BudgetModel(refuse=range(1, 7))
    _pdf(monkeypatch, model)
    delivered = _capture_delivery(monkeypatch)

    state = await AA.start_analysis(USER, AID, await _stored(), TASK, session_id=SESSION, channel="app")
    await _settle()

    final = AA.load_state(USER, AID, state["analysis_id"])
    assert 1 <= len(model.calls) <= 3                  # no unit dequeued after the stop
    assert final["status"] == "failed" and final["pages"] == []
    assert final["units_attempted"] == 0
    assert delivered == [[INCIDENT_SNAPSHOT]]          # refused calls read nothing


@pytest.mark.parametrize("source", ["refusal", "preflight"])
async def test_a_reset_already_past_is_never_stored_and_undated_copy_says_ask_later(
        monkeypatch, preflight, frozen, source):
    """A past period_end (from the refusal body or the platform's check) is
    not a reset to promise or wait for; without one the copy names no
    period and asks the user to try again later."""
    model = BudgetModel(refuse=range(1, 7), period_end=PAST_END)
    _pdf(monkeypatch, model)
    delivered = _capture_delivery(monkeypatch)
    if source == "preflight":
        preflight.block(PAST_END)

    state = await AA.start_analysis(USER, AID, await _stored(), TASK, session_id=SESSION, channel="app")
    await _settle()
    final = AA.load_state(USER, AID, state["analysis_id"])
    assert final["status"] == "failed" and "blocked_until" not in final
    assert delivered == [[
        "Handout.pdf — not read (0 of 6 pages)\n\n"
        "My AI budget is used up, so I haven’t read any of this file. Ask me again later. "
        "Your credits aren’t affected."]]
    assert final["error"]["message"] == "The AI budget is used up, so reading stopped. Credits are not affected."
    assert AA.blocked_reply_text(final, persian=False) == (
        "I can’t read this file right now: my AI budget is used up. Your credits aren’t affected.")
    assert AA.blocked_guidance(final) == (
        "The AI budget is used up and no reset date is known. Nothing more was read. "
        "Tell the user that; do not describe unread pages and do not call "
        "read_attachment_analysis for them. It will not be retried automatically; "
        "the user can ask again later.")


def test_undated_persian_copy_names_no_period_and_says_ask_later():
    state = _stopped(AA.analysis_id_for(FA_TASK), read=(1, 2), task=FA_TASK)
    assert AA._delivery_parts(state, tz_name=TZ, now=FROZEN)[0].split("\n\n")[1] == (
        "بودجهٔ هوش مصنوعی من وسط کار تمام شد؛ صفحه‌های ۱ تا ۲ را خواندم، ولی صفحه‌های ۳ تا ۶ "
        "را نه. بعداً دوباره از من بخواهید. این موضوع روی اعتبار شما اثری ندارد.")
    assert "ماهانه" not in AA.blocked_reply_text(state, persian=True, now=FROZEN)


async def test_a_credit_stop_keeps_the_pages_read_and_is_never_resumed_automatically(
        monkeypatch, preflight, frozen):
    class CreditModel(BudgetModel):
        async def page(self, task, number, count, native_text, image):
            self.calls.append(number)
            if number >= 3:
                raise credits_402()
            return f"Page {number}: evidence"

    model = CreditModel()
    _pdf(monkeypatch, model)
    delivered = _capture_delivery(monkeypatch)
    state = await AA.start_analysis(USER, AID, await _stored(), TASK, session_id=SESSION, channel="app")
    await _settle()

    final = AA.load_state(USER, AID, state["analysis_id"])
    assert model.calls == [1, 2, 3]
    assert final["status"] == "partial" and final["error"]["code"] == "analysis_credits_exhausted"
    assert final.get("blocked_until") is None
    assert delivered == [[
        "Analysis of Handout.pdf (2 of 6 pages read)\n\n"
        "You’re out of Toup credits for now, so I haven’t read pages 3–6. "
        "Open Credits to top up or upgrade, then ask me again.\n\n"
        "Page-by-page details for pages 1–2 follow.",
        "Handout.pdf — pages 1–6 of 6\n\n"
        "### Page 1\nPage 1: evidence\n\n### Page 2\nPage 2: evidence\n\n"
        "Pages 3–6 were not read.",
    ]]
    assert AA.blocked_reply_text(final, persian=False) is None

    await AA.reconcile_local_analyses()               # topping up is the user's move
    await _settle()
    assert model.calls == [1, 2, 3] and preflight.calls == 1


@pytest.mark.parametrize("reason,message,body", [
    ("daily_cap_exceeded", "Today’s message limit is reached, so reading stopped.",
     "You’ve reached today’s message limit, so I haven’t read pages 3–6. "
     "Ask me again after it resets."),
    ("email_not_verified", "The account’s email address is not verified yet, so reading stopped.",
     "Please verify your email so I can read pages 3–6."),
    ("insufficient_message_credits", "The account is out of Toup credits, so reading stopped.",
     "You’re out of Toup credits for now, so I haven’t read pages 3–6. "
     "Open Credits to top up or upgrade, then ask me again."),
], ids=["daily_cap", "email_not_verified", "empty_balance"])
async def test_a_credit_stop_says_what_the_refusal_means(
        monkeypatch, preflight, frozen, reason, message, body):
    """The free tier's daily cap and an unverified email are not an empty
    balance: neither may be told "you’re out of Toup credits"."""
    class CreditModel(BudgetModel):
        async def page(self, task, number, count, native_text, image):
            self.calls.append(number)
            if number >= 3:
                raise credits_402(reason)
            return f"Page {number}: evidence"

    _pdf(monkeypatch, CreditModel())
    delivered = _capture_delivery(monkeypatch)
    state = await AA.start_analysis(USER, AID, await _stored(), TASK, session_id=SESSION, channel="app")
    await _settle()

    final = AA.load_state(USER, AID, state["analysis_id"])
    assert final["error"] == {"code": "analysis_credits_exhausted", "message": message}
    assert final["credit_reason"] == reason
    assert delivered[0][0] == (f"Analysis of Handout.pdf (2 of 6 pages read)\n\n{body}\n\n"
                               "Page-by-page details for pages 1–2 follow.")


def test_credit_stop_copy_for_a_file_not_read_at_all():
    state = _stopped("c" * 20)
    state["error"] = {"code": "analysis_credits_exhausted", "message": "x"}
    state.pop("blocked_until")
    state["credit_reason"] = "email_not_verified"
    assert AA._delivery_parts(state, now=FROZEN) == [
        "Handout.pdf — not read (0 of 6 pages)\n\nPlease verify your email so I can read this file."]
    state["credit_reason"] = "daily_cap_exceeded"
    assert AA._delivery_parts(state, now=FROZEN) == [
        "Handout.pdf — not read (0 of 6 pages)\n\n"
        "You’ve reached today’s message limit, so I haven’t read any of this file. "
        "Ask me again after it resets."]
    state["task"] = FA_TASK
    assert AA._delivery_parts(state, now=FROZEN)[0].split("\n\n")[1] == (
        "به سقف پیام‌های امروزتان رسیده‌اید، برای همین هنوز هیچ بخشی از این فایل را نخوانده‌ام. "
        "وقتی این سقف تمدید شد، دوباره از من بخواهید.")
    state["credit_reason"] = "email_not_verified"
    assert AA._delivery_parts(state, now=FROZEN)[0].split("\n\n")[1] == (
        "لطفاً ایمیلتان را تأیید کنید تا بتوانم این فایل را بخوانم.")


def test_an_ordinary_unit_failure_still_says_could_not_analyze():
    """Non-service failures keep today's layout (test_attachment_analysis.py)."""
    state = AA.new_state(USER, AID, "f" * 20, TASK, NAME, "page", 6, SESSION, "app", None, True)
    state.update(
        status="partial", stage="complete", overview="Overview of 6",
        pages=[{"page_number": n, "status": "completed", "summary": f"Page {n}: evidence"}
               for n in (1, 2, 4, 5, 6)] + [
            {"page_number": 3, "status": "failed", "error": "unit_analysis_failed"}],
        error={"code": "units_failed", "message": "I couldn’t read page 3."},
    )
    parts = AA._delivery_parts(state)
    assert parts[0].startswith("Analysis of Handout.pdf (5 of 6 pages read)\n\nOverview of 6")
    assert "Coverage is incomplete." in parts[0]
    assert "### Page 3\nCould not analyze this page." in parts[1]
    assert "budget" not in "".join(parts)

    state.update(status="failed", overview=None,
                 pages=[{"page_number": n, "status": "failed", "error": "unit_analysis_failed"}
                        for n in range(1, 7)],
                 error={"code": "units_failed", "message": "No document units could be analyzed."})
    parts = AA._delivery_parts(state)
    assert parts[0].startswith("Handout.pdf — not read (0 of 6 pages)\n\n"
                               "I couldn’t read pages 1–6.")
    assert parts[1].count("Could not analyze this page.") == 6


def test_a_failed_page_is_never_blamed_on_the_budget_or_promised_a_retry():
    """Page 2 failed on its own. The automatic resume re-sends only pages
    never read, so only those are promised, and page 2 is told apart."""
    state = _stopped("d" * 20, read=(1,), blocked_until=PERIOD_END, first_failed_at=FROZEN)
    state["pages"].append({"page_number": 2, "status": "failed", "error": "unit_analysis_failed"})
    parts = AA._delivery_parts(state, tz_name=TZ, now=FROZEN)
    assert parts == [
        "Analysis of Handout.pdf (1 of 6 pages read)\n\n"
        "My monthly AI budget ran out partway through, so I read page 1, but not pages 3–6. "
        "I couldn’t read page 2. It resets on October 20, and I’ll try the unread pages again "
        "automatically then. Your credits aren’t affected.\n\n"
        "Page-by-page details for page 1 follow.",
        "Handout.pdf — pages 1–6 of 6\n\n### Page 1\nPage 1: evidence\n\n"
        "### Page 2\nCould not analyze this page.\n\nPages 3–6 were not read.",
    ]
    assert AA.blocked_reply_text(state, persian=False, tz_name=TZ, now=FROZEN) == (
        "I can’t read the rest of this file yet: my monthly AI budget is used up. It resets on "
        "October 20, and I’ll try the unread pages again automatically then. "
        "Your credits aren’t affected.")
    resumed = copy.deepcopy(state)                      # what the reconciler really re-sends
    AA._requeue_failed_units(resumed, include_failed_units=False)
    assert [(p["page_number"], p["status"]) for p in resumed["pages"]] == [
        (1, "completed"), (2, "failed")]

    fa = copy.deepcopy(state)
    fa["task"] = FA_TASK
    assert AA._delivery_parts(fa, tz_name=TZ, now=FROZEN)[0].split("\n\n")[1] == (
        "بودجهٔ ماهانهٔ هوش مصنوعی من وسط کار تمام شد؛ صفحهٔ ۱ را خواندم، ولی صفحه‌های ۳ تا ۶ "
        "را نه. نتوانستم صفحهٔ ۲ را بخوانم. این بودجه ۲۰ اکتبر تمدید می‌شود و آن موقع "
        "صفحه‌های خوانده‌نشده را به‌طور خودکار دوباره امتحان می‌کنم. این موضوع روی اعتبار شما "
        "اثری ندارد.")

    # Every page has a record and the stop hit the summary: nothing unread to promise.
    whole = _stopped("d" * 20, read=range(1, 7), blocked_until=PERIOD_END, first_failed_at=FROZEN)
    whole["pages"][5] = {"page_number": 6, "status": "failed", "error": "unit_analysis_failed"}
    assert AA._delivery_parts(whole, tz_name=TZ, now=FROZEN)[0] == (
        "Analysis of Handout.pdf (5 of 6 pages read)\n\n"
        "I read pages 1–5, but the monthly AI budget ran out before I could write the combined "
        "summary. I couldn’t read page 6. It resets on October 20, and I’ll try again "
        "automatically then. Your credits aren’t affected.\n\n"
        "Page-by-page details for pages 1–5 follow.")


def test_a_delivery_made_after_the_reset_says_the_budget_has_reset():
    """A stop delivered late (the agent was down across the reset) must not
    say the budget "is used up"; the title stays "not read"."""
    after = datetime(2026, 10, 21, 12, 0, tzinfo=timezone.utc)
    state = _stopped("a" * 20, blocked_until=PERIOD_END, first_failed_at=FROZEN)
    assert AA._delivery_parts(state, tz_name=TZ, now=after) == [
        "Handout.pdf — not read (0 of 6 pages)\n\n"
        "My AI budget was used up when I tried. It has reset, so I’ll read it again shortly. "
        "Your credits aren’t affected."]
    state["auto_resume_count"] = AA.AUTO_RESUME_LIMIT   # no automatic resume left
    assert AA._delivery_parts(state, tz_name=TZ, now=after) == [
        "Handout.pdf — not read (0 of 6 pages)\n\n"
        "My AI budget was used up when I tried. It has reset, so you can ask me again now. "
        "Your credits aren’t affected."]
    fa = _stopped("a" * 20, blocked_until=PERIOD_END, first_failed_at=FROZEN, task=FA_TASK)
    assert AA._delivery_parts(fa, tz_name=TZ, now=after)[0].split("\n\n")[1] == (
        "وقتی امتحان کردم، بودجهٔ هوش مصنوعی من تمام شده بود. این بودجه تمدید شده است و "
        "به‌زودی دوباره آن را می‌خوانم. این موضوع روی اعتبار شما اثری ندارد.")


async def test_persian_request_gets_persian_title_body_and_not_read_lines(monkeypatch, preflight, frozen):
    model = BudgetModel()
    _pdf(monkeypatch, model)
    delivered = _capture_delivery(monkeypatch)
    preflight.block(PERIOD_END)

    await AA.start_analysis(USER, AID, await _stored(), FA_TASK, session_id=SESSION, channel="app")
    await _settle()
    assert delivered == [[
        "فایل «Handout.pdf» هنوز خوانده نشده است (۰ از ۶ صفحه)\n\n"
        "بودجهٔ ماهانهٔ هوش مصنوعی من تمام شده است، برای همین هنوز هیچ بخشی از این فایل را "
        "نخوانده‌ام. این بودجه ۲۰ اکتبر تمدید می‌شود و آن موقع به‌طور خودکار دوباره امتحان "
        "می‌کنم. این موضوع روی اعتبار شما اثری ندارد."
    ]]

    partial = _stopped(AA.analysis_id_for(FA_TASK), read=(1, 2), blocked_until=PERIOD_END,
                       first_failed_at=FROZEN, task=FA_TASK)
    parts = AA._delivery_parts(partial, tz_name=TZ)
    assert parts == [
        "تحلیل فایل «Handout.pdf» (۲ از ۶ صفحه خوانده شد)\n\n"
        "بودجهٔ ماهانهٔ هوش مصنوعی من وسط کار تمام شد؛ صفحه‌های ۱ تا ۲ را خواندم، ولی "
        "صفحه‌های ۳ تا ۶ را نه. این بودجه ۲۰ اکتبر تمدید می‌شود و آن موقع صفحه‌های "
        "خوانده‌نشده را به‌طور خودکار دوباره امتحان می‌کنم. این موضوع روی اعتبار شما اثری ندارد.\n\n"
        "جزئیات صفحه‌های ۱ تا ۲ در ادامه می‌آید.",
        "فایل «Handout.pdf» — صفحه‌های ۱ تا ۶ از ۶\n\n"
        "### صفحهٔ ۱\nPage 1: evidence\n\n"
        "### صفحهٔ ۲\nPage 2: evidence\n\n"
        "صفحه‌های ۳ تا ۶ خوانده نشد.",
    ]
    # Every Persian line starts with a Persian word, so both clients lay it out RTL.
    for line in parts[0].split("\n\n") + [parts[1].split("\n\n")[0], parts[1].split("\n\n")[-1]]:
        assert AA._PERSIAN_RE.match(line), line


def _blocked(*, blocked_until=PERIOD_END, targets=True, auto_resume_count=0, status="failed",
             read=()):
    state = _stopped("e" * 20, read=read, blocked_until=blocked_until, first_failed_at=FROZEN,
                     auto_resume_count=auto_resume_count, targets=None if targets else [])
    state["status"] = status
    return state


@pytest.mark.parametrize("state_kw,persian,expected", [
    ({}, False,
     "I can’t read this file yet: my monthly AI budget is used up. It resets on October 20, "
     "and I’ll try again automatically then. Your credits aren’t affected."),
    ({"targets": False}, False,       # nowhere to post a result: no automatic retry promised
     "I can’t read this file yet: my monthly AI budget is used up. It resets on October 20; "
     "ask me again after that. Your credits aren’t affected."),
    ({"auto_resume_count": 2}, False,  # automatic retries used up
     "I can’t read this file yet: my monthly AI budget is used up. It resets on October 20; "
     "ask me again after that. Your credits aren’t affected."),
    ({"blocked_until": None}, False,
     "I can’t read this file right now: my AI budget is used up. Your credits aren’t affected."),
    ({"blocked_until": PAST_END}, False,   # a past reset is never promised
     "I can’t read this file right now: my AI budget is used up. Your credits aren’t affected."),
    ({"read": (1, 2), "status": "partial"}, False,
     "I can’t read the rest of this file yet: my monthly AI budget is used up. It resets on "
     "October 20, and I’ll try the unread pages again automatically then. "
     "Your credits aren’t affected."),
    ({"read": range(1, 7), "status": "partial"}, False,   # only the combined summary is missing
     "I can’t finish this file yet: my monthly AI budget is used up. It resets on October 20, "
     "and I’ll try again automatically then. Your credits aren’t affected."),
    ({}, True,
     "فعلاً نمی‌توانم این فایل را بخوانم؛ بودجهٔ ماهانهٔ هوش مصنوعی من تمام شده است. این بودجه "
     "۲۰ اکتبر تمدید می‌شود و آن موقع به‌طور خودکار دوباره امتحان می‌کنم. این موضوع روی اعتبار "
     "شما اثری ندارد."),
    ({"targets": False}, True,
     "فعلاً نمی‌توانم این فایل را بخوانم؛ بودجهٔ ماهانهٔ هوش مصنوعی من تمام شده است. این بودجه "
     "۲۰ اکتبر تمدید می‌شود؛ بعد از آن دوباره از من بخواهید. این موضوع روی اعتبار شما اثری ندارد."),
    ({"blocked_until": None}, True,
     "فعلاً نمی‌توانم این فایل را بخوانم؛ بودجهٔ هوش مصنوعی من تمام شده است. این موضوع روی اعتبار "
     "شما اثری ندارد."),
    ({"read": (1, 2), "status": "partial"}, True,
     "فعلاً نمی‌توانم بقیهٔ این فایل را بخوانم؛ بودجهٔ ماهانهٔ هوش مصنوعی من تمام شده است. این "
     "بودجه ۲۰ اکتبر تمدید می‌شود و آن موقع صفحه‌های خوانده‌نشده را به‌طور خودکار دوباره امتحان "
     "می‌کنم. این موضوع روی اعتبار شما اثری ندارد."),
], ids=["en_auto", "en_no_destination", "en_retries_used", "en_no_date", "en_past_reset",
        "en_partial", "en_summary_only", "fa_auto", "fa_no_destination", "fa_no_date", "fa_partial"])
def test_blocked_reply_text_is_the_kickoff_reply(state_kw, persian, expected):
    assert AA.blocked_reply_text(_blocked(**state_kw), persian=persian, now=FROZEN) == expected


def test_blocked_reply_text_names_the_hour_inside_48_hours_and_ignores_other_states():
    soon = datetime(2026, 10, 19, 21, 0, tzinfo=timezone.utc)
    assert AA.blocked_reply_text(_blocked(), persian=False, tz_name=TZ, now=soon) == (
        "I can’t read this file yet: my monthly AI budget is used up. It resets on October 20 "
        "at 3:45 PM, and I’ll try again automatically then. Your credits aren’t affected.")
    assert AA.blocked_reply_text(_blocked(status="partial"), persian=False, now=FROZEN)
    assert AA.blocked_reply_text(_blocked(status="queued"), persian=False, now=FROZEN) is None
    done = _blocked()
    done.update(status="completed", error=None)
    assert AA.blocked_reply_text(done, persian=False, now=FROZEN) is None
    other = _blocked()
    other["error"] = {"code": "analysis_credits_exhausted", "message": "x"}
    assert AA.blocked_reply_text(other, persian=True, now=FROZEN) is None
    assert AA.blocked_reply_text(None, persian=False) is None


async def test_tools_tell_the_model_what_was_not_read_while_the_budget_is_spent(
        monkeypatch, preflight, frozen, tmp_path):
    from app.agent.tool_executor import ToolExecutor

    model = BudgetModel()
    _pdf(monkeypatch, model)
    preflight.block(PERIOD_END)
    await _stored()
    executor = ToolExecutor(workspace=str(tmp_path))
    executor.set_user_id(USER)

    first = json.loads(await executor.execute("analyze_attachment", {"attachment_id": AID, "task": TASK}))
    assert first["status"] == "queued"
    assert first["guidance"].startswith("Analysis is running in the background.")
    await _settle()

    read = json.loads(await executor.execute("read_attachment_analysis", {
        "attachment_id": AID, "analysis_id": first["analysis_id"]}))
    assert read["status"] == "failed" and read["units"] == []
    assert read["guidance"].startswith(
        "The monthly AI budget is used up (resets on October 20). Nothing more was read. "
        "Tell the user that and whether it will be retried automatically; do not describe "
        "unread pages and do not call read_attachment_analysis for them.")

    again = json.loads(await executor.execute("analyze_attachment", {
        "attachment_id": AID, "task": TASK, "retry_failed": True}))
    assert again["status"] == "failed"
    assert again["guidance"] == read["guidance"]
    assert "The result is stored" not in again["guidance"]
    assert model.calls == []

    def too_large(record):
        raise ValueError("file_too_large")

    monkeypatch.setattr(AA, "read_original_bytes", too_large)
    refused = await executor.execute("analyze_attachment", {
        "attachment_id": AID, "task": "Summarize the whole handout"})
    assert refused == "ERROR: The stored file exceeds the 25 MB analysis limit."


# ── explicit retries (spec B4) ───────────────────────────────────────


async def test_retry_while_still_blocked_reposts_only_the_status_and_reads_nothing(
        monkeypatch, preflight, frozen, broadcasts):
    _agent_db(monkeypatch)
    model = BudgetModel(refuse={3, 4, 5, 6})
    made = _pdf(monkeypatch, model)
    await _seed_user()
    yesterday = await _seed_chat(ANCHOR_1, days_ago=1)
    today = await _seed_chat(ANCHOR_2, days_ago=0)
    rec = await _stored()

    state = await AA.start_analysis(USER, AID, rec, TASK, session_id=SESSION, channel="app",
                                    anchor_message_id=ANCHOR_1)
    await _settle()
    analysis_id = state["analysis_id"]
    assert (await AA.load_state_async(USER, AID, analysis_id))["status"] == "partial"
    assert model.calls == [1, 2, 3] and len(made) == 1

    preflight.block(NAIVE_PERIOD_END)                  # still spent (an offset-less platform)
    retried = await AA.start_analysis(USER, AID, rec, TASK, retry_failed=True, session_id=SESSION,
                                      channel="app", anchor_message_id=ANCHOR_2)
    assert (USER, AID, analysis_id) not in AA._active   # nothing requeued
    assert retried["status"] == "partial" and retried["delivery_attempt"] == 1
    await _settle()

    assert model.calls == [1, 2, 3] and len(made) == 1  # the model was not asked again
    assert preflight.calls == 2
    final = await AA.load_state_async(USER, AID, analysis_id)
    assert final["delivered"] is True and final["delivery_attempt"] == 1
    assert final["auto_resume_count"] == 0 and final["blocked_until"] == PERIOD_END
    assert [p["page_number"] for p in final["pages"]] == [1, 2]

    messages = await _analysis_messages()
    attempt0 = {_message_id(analysis_id, 0, 0), _message_id(analysis_id, 0, 1)}
    attempt1 = _message_id(analysis_id, 1, 0)
    assert set(messages) == attempt0 | {attempt1}      # attempts [0, 1]; attempt 1 = status only
    status = (
        "Analysis of Handout.pdf (2 of 6 pages read)\n\n"
        "My monthly AI budget ran out partway through, so I read pages 1–2, but not pages 3–6. "
        "It resets on October 20, and I’ll try the unread pages again automatically then. "
        "Your credits aren’t affected.")
    assert messages[attempt1].content == status
    assert messages[_message_id(analysis_id, 0, 0)].content == (
        status + "\n\nPage-by-page details for pages 1–2 follow.")
    # Filed under the turn that asked again, not the first request's day.
    assert {messages[m].day_chat_id for m in attempt0} == {yesterday}
    assert messages[attempt1].day_chat_id == today
    assert len(broadcasts) == 3


async def test_kickoff_retry_while_blocked_answers_itself_and_posts_nothing(
        monkeypatch, preflight, frozen, broadcasts):
    """agent_runner's kickoff passes redeliver_when_blocked=False: its reply
    (blocked_reply_text) is the message, so nothing is re-posted."""
    _agent_db(monkeypatch)
    model = BudgetModel()
    _pdf(monkeypatch, model)
    preflight.block(PERIOD_END)
    await _seed_user()
    await _seed_chat(ANCHOR_1, ANCHOR_2)
    rec = await _stored()
    state = await AA.start_analysis(USER, AID, rec, TASK, session_id=SESSION, channel="app",
                                    anchor_message_id=ANCHOR_1)
    await _settle()

    kicked = await AA.start_analysis(USER, AID, rec, TASK, retry_failed=True, session_id=SESSION,
                                     channel="app", anchor_message_id=ANCHOR_2,
                                     redeliver_when_blocked=False)
    await _settle()
    assert kicked["status"] == "failed"
    assert AA.blocked_reply_text(kicked, persian=False) == (
        "I can’t read this file yet: my monthly AI budget is used up. It resets on October 20, "
        "and I’ll try again automatically then. Your credits aren’t affected.")
    assert list(await _analysis_messages()) == [_message_id(state["analysis_id"], 0, 0)]
    row = await _row(state["analysis_id"])
    assert row.delivered is True and row.state_json["delivery_attempt"] == 0
    assert model.calls == [] and len(broadcasts) == 1


async def test_a_retry_while_blocked_never_keeps_a_reset_that_has_passed(monkeypatch, preflight, frozen):
    _pdf(monkeypatch, BudgetModel())
    _capture_delivery(monkeypatch)
    rec = await _stored()
    analysis_id = AA.analysis_id_for(TASK)
    await AA.save_state(_stopped(analysis_id, blocked_until="2026-09-20T00:00:00+00:00",
                                 first_failed_at=FROZEN))

    preflight.block("2026-09-25T00:00:00+00:00")        # still spent; its date is stale too
    state = await AA.start_analysis(USER, AID, rec, TASK, retry_failed=True, session_id=SESSION,
                                    channel="app")
    await _settle()
    assert state["status"] == "failed" and "blocked_until" not in state
    assert "blocked_until" not in AA.load_state(USER, AID, analysis_id)

    preflight.block(PERIOD_END)
    state = await AA.start_analysis(USER, AID, rec, TASK, retry_failed=True, session_id=SESSION,
                                    channel="app")
    await _settle()
    assert state["blocked_until"] == PERIOD_END


async def test_asking_for_page_details_after_the_job_finished_reposts_them(
        monkeypatch, preflight, broadcasts):
    """A finished job's include_unit_details bump used to be dropped silently
    in agent mode (the no-claim save refused any attempt change)."""
    _agent_db(monkeypatch)
    await _seed_user()
    await _seed_chat(ANCHOR_1, ANCHOR_2)
    rec = await _stored()
    task = "Summarize this file"
    analysis_id = AA.analysis_id_for(task)
    state = AA.new_state(USER, AID, analysis_id, task, NAME, "page", 6, SESSION, "app", ANCHOR_1)
    state.update(status="completed", stage="complete", overview="Overview of 6", delivered=True,
                 pages=[{"page_number": n, "status": "completed", "summary": f"Page {n}: evidence"}
                        for n in range(1, 7)])
    state["delivery_targets"][0]["delivered"] = True
    await AA.save_state(state)

    await AA.start_analysis(USER, AID, rec, task, include_unit_details=True, session_id=SESSION,
                            channel="app", anchor_message_id=ANCHOR_2)
    await _settle()

    row = await _row(analysis_id)
    assert row.state_json["include_unit_details"] is True
    assert row.state_json["delivery_attempt"] == 1 and row.delivered is True
    messages = await _analysis_messages()
    assert set(messages) == {_message_id(analysis_id, 1, 0), _message_id(analysis_id, 1, 1)}
    assert "### Page 6\nPage 6: evidence" in messages[_message_id(analysis_id, 1, 1)].content
    assert preflight.calls == 0


# ── automatic recovery after the reset (spec B5) ─────────────────────


async def test_reconciler_resumes_once_after_the_reset_and_rebills_nothing(
        monkeypatch, preflight, broadcasts):
    _agent_db(monkeypatch)
    model = BudgetModel()
    _pdf(monkeypatch, model)
    pushes = _capture_notify(monkeypatch)
    await _seed_user()
    old_day = await _seed_chat(ANCHOR_1, days_ago=20)
    await _stored()
    analysis_id = AA.analysis_id_for(TASK)
    now = datetime.now(timezone.utc)
    await AA.save_state(_stopped(analysis_id, read=(1, 2),
                                 blocked_until=now - timedelta(minutes=3),
                                 first_failed_at=now - timedelta(days=20)))
    preflight.admit()

    # Two overlapping passes (two replicas' loops, or a slow pass and the next tick).
    await asyncio.gather(AA.reconcile_local_analyses(), AA.reconcile_local_analyses())
    await _settle()

    final = await AA.load_state_async(USER, AID, analysis_id)
    assert final["status"] == "completed", final.get("error")
    assert sorted(model.calls) == [3, 4, 5, 6]         # pages 1-2 were already paid for
    assert final["auto_resume_count"] == 1 and final["resumed_automatically"] is True
    assert final["resumed_attempt"] == 1 and final["resumed_sessions"] == [SESSION]
    assert final["delivery_attempt"] == 1 and final["delivered"] is True
    assert preflight.calls == 2                        # one pass asked, then the run itself

    messages = await _analysis_messages()
    assert set(messages) == {_message_id(analysis_id, 1, 0), _message_id(analysis_id, 1, 1)}
    lead = messages[_message_id(analysis_id, 1, 0)]
    assert lead.content == (
        "Earlier you asked me to read Handout.pdf. My AI budget has reset, so here "
        "is the result.\n\n"
        "Analysis of Handout.pdf (6 pages)\n\nOverview of 6\n\n"
        "Numbered details follow in batches.")
    # Delivered into today's chat, not the day of the request three weeks ago.
    from app.agent.day_chat_resolver import resolve_local_date
    today = {resolve_local_date(datetime.now(timezone.utc) + timedelta(minutes=m), TZ)[0]
             for m in (-5, 0)}
    assert {m.day_chat_id for m in messages.values()} != {old_day}
    days = [await _day_of(m.day_chat_id) for m in messages.values()]
    assert days and all(day in today for day in days), days
    assert [(p["event_kind"], p["title"], p["body"]) for p in pushes] == [
        ("mission_completed", "Your file is ready", "I finished reading Handout.pdf.")]

    # Nothing is left to resume: another pass asks neither the platform nor the model.
    await AA.reconcile_local_analyses()
    await _settle()
    assert sorted(model.calls) == [3, 4, 5, 6] and preflight.calls == 2 and len(pushes) == 1


async def test_a_resume_that_runs_out_again_says_so_and_pushes_what_it_read(
        monkeypatch, preflight, broadcasts):
    from app.services.budget_refusal import reset_when_phrase

    _agent_db(monkeypatch)
    now = datetime.now(timezone.utc)
    next_reset = AA.iso_utc(now + timedelta(days=30))

    class RunsOutAgain(BudgetModel):
        async def page(self, task, number, count, native_text, image):
            self.calls.append(number)
            if number >= 4:
                raise budget_429(next_reset)
            return f"Page {number}: evidence"

    model = RunsOutAgain()
    _pdf(monkeypatch, model)
    pushes = _capture_notify(monkeypatch)
    await _seed_user()
    await _seed_chat(ANCHOR_1, days_ago=2)
    await _stored()
    analysis_id = AA.analysis_id_for(TASK)
    await AA.save_state(_stopped(analysis_id, read=(1, 2), blocked_until=now - timedelta(minutes=3),
                                 first_failed_at=now - timedelta(days=2)))
    preflight.admit()

    await AA.reconcile_local_analyses()
    await _settle()

    final = await AA.load_state_async(USER, AID, analysis_id)
    assert model.calls == [3, 4]
    assert final["status"] == "partial" and final["auto_resume_count"] == 1
    assert final["blocked_until"] == next_reset
    when = reset_when_phrase(next_reset, TZ, datetime.now(timezone.utc))
    messages = await _analysis_messages()
    assert messages[_message_id(analysis_id, 1, 0)].content == (
        "Earlier you asked me to read Handout.pdf. My AI budget has reset, so here is the result.\n\n"
        "Analysis of Handout.pdf (3 of 6 pages read)\n\n"
        "My monthly AI budget ran out partway through, so I read pages 1–3, but not pages 4–6. "
        f"It resets on {when}, and I’ll try the unread pages again automatically then. "
        "Your credits aren’t affected.\n\n"
        "Page-by-page details for pages 1–3 follow.")
    # "Your file is ready" would be false; what it read is true.
    assert [(p["title"], p["body"]) for p in pushes] == [
        ("Your file is partly ready", "I read 3 of 6 pages of Handout.pdf.")]


async def test_a_resume_that_reads_nothing_says_it_tried_and_pushes_nothing(
        monkeypatch, preflight, broadcasts):
    _agent_db(monkeypatch)
    monkeypatch.setattr(AA, "MODEL_ATTEMPTS", 1)

    class Broken(BudgetModel):
        async def page(self, task, number, count, native_text, image):
            self.calls.append(number)
            raise RuntimeError("empty_model_response")

    model = Broken()
    _pdf(monkeypatch, model)
    pushes = _capture_notify(monkeypatch)
    await _seed_user()
    await _seed_chat(ANCHOR_1, days_ago=2)
    await _stored()
    analysis_id = AA.analysis_id_for(TASK)
    now = datetime.now(timezone.utc)
    await AA.save_state(_stopped(analysis_id, blocked_until=now - timedelta(minutes=3),
                                 first_failed_at=now - timedelta(days=2)))
    preflight.admit()

    await AA.reconcile_local_analyses()
    await _settle()
    final = await AA.load_state_async(USER, AID, analysis_id)
    assert final["status"] == "failed" and final["error"]["code"] == "units_failed"
    first = (await _analysis_messages())[_message_id(analysis_id, 1, 0)].content
    assert first.startswith("Earlier you asked me to read Handout.pdf, so I tried again.\n\n"
                            "Handout.pdf — not read (0 of 6 pages)")
    assert "has reset" not in first
    assert pushes == []


@pytest.mark.parametrize("task,refused_from,spent_before_the_run,lead,push", [
    (TASK, 4, False, "Earlier you asked me to read Handout.pdf, so I tried again.", None),
    (TASK, None, True, "Earlier you asked me to read Handout.pdf, so I tried again.", None),
    (FA_TASK, 4, False, "قبلاً خواسته بودید «Handout.pdf» را بخوانم؛ برای همین دوباره امتحان کردم.", None),
    (FA_TASK, 5, False,
     "قبلاً خواسته بودید «Handout.pdf» را بخوانم. بودجهٔ هوش مصنوعی‌ام تمدید شده است؛ این هم نتیجه.",
     ("Your file is partly ready", "I read 4 of 6 pages of Handout.pdf.")),
], ids=["en_refused_again_at_once", "en_spent_again_before_the_run", "fa_refused_again_at_once",
        "fa_one_new_page"])
async def test_a_resume_claims_a_result_only_for_what_it_completed_itself(
        monkeypatch, preflight, broadcasts, task, refused_from, spent_before_the_run, lead, push):
    """Pages 1-3 were read (and posted) before the stop. A resume that is
    refused again before it completes anything new has no new result: it
    says it tried again and sends no push. It used to say "has reset, so
    here is the result" and push "partly ready" for the three pages the user
    already had. One page more is a result, in Persian as in English."""
    _agent_db(monkeypatch)
    now = datetime.now(timezone.utc)
    next_reset = AA.iso_utc(now + timedelta(days=30))

    class RefusedAgain(BudgetModel):
        async def page(self, task, number, count, native_text, image):
            self.calls.append(number)
            if refused_from is not None and number >= refused_from:
                raise budget_429(next_reset)
            return f"Page {number}: evidence"

    class SpentAgainAfterTheResumeAsked(_Preflight):
        """The pass's verdict admits; the window is spent again before the run asks."""

        async def __call__(self):
            verdict = await super().__call__()
            if spent_before_the_run:
                self.block(next_reset)
            return verdict

    model = RefusedAgain()
    _pdf(monkeypatch, model)
    asks = SpentAgainAfterTheResumeAsked()
    asks.admit()
    monkeypatch.setattr(AA, "_budget_preflight", asks)
    pushes = _capture_notify(monkeypatch)
    await _seed_user()
    await _seed_chat(ANCHOR_1, days_ago=2)
    await _stored()
    analysis_id = AA.analysis_id_for(task)
    await AA.save_state(_stopped(analysis_id, read=(1, 2, 3), blocked_until=now - timedelta(minutes=3),
                                 first_failed_at=now - timedelta(days=2), task=task))

    await AA.reconcile_local_analyses()
    await _settle()
    final = await AA.load_state_async(USER, AID, analysis_id)
    assert final["status"] == "partial" and final["error"]["code"] == "analysis_budget_exceeded"
    assert model.calls == ([] if refused_from is None else list(range(4, refused_from + 1)))
    first = (await _analysis_messages())[_message_id(analysis_id, 1, 0)].content
    assert first.split("\n\n")[0] == lead
    assert [(p["title"], p["body"]) for p in pushes] == ([push] if push else [])
    assert final["auto_resume_count"] == 1 and final["resumed_baseline"] == 3
    assert final["blocked_until"] == next_reset and asks.calls == 2


async def test_the_resume_lead_in_goes_only_to_the_chats_that_waited(
        monkeypatch, preflight, broadcasts):
    _agent_db(monkeypatch)
    model = BudgetModel()
    _pdf(monkeypatch, model)
    pushes = _capture_notify(monkeypatch)
    await _seed_user()
    await _seed_chat(ANCHOR_1, ANCHOR_2)
    await _seed_conversation(SESSION_B)
    rec = await _stored()
    task = "Summarize this file"
    analysis_id = AA.analysis_id_for(task)
    now = datetime.now(timezone.utc)
    await AA.save_state(_stopped(analysis_id, read=(1, 2), blocked_until=now - timedelta(minutes=3),
                                 first_failed_at=now - timedelta(days=3), task=task))
    preflight.admit()
    await AA.reconcile_local_analyses()
    await _settle()
    assert (await AA.load_state_async(USER, AID, analysis_id))["status"] == "completed"

    # Today, in another chat, the user asks for the same file again (the kickoff's kwargs).
    await AA.start_analysis(USER, AID, rec, task, retry_failed=True, session_id=SESSION_B,
                            channel="app", anchor_message_id=None, request_identity=task,
                            redeliver_when_blocked=False)
    await _settle()
    first = _first_parts(await _analysis_messages())
    assert first[SESSION].startswith(
        "Earlier you asked me to read Handout.pdf. My AI budget has reset, so here is the result.")
    assert first[SESSION_B] == "Analysis of Handout.pdf (6 pages)\n\nOverview of 6"
    assert len(pushes) == 1

    # Page details asked for later are a new attempt: no lead-in there either.
    await AA.start_analysis(USER, AID, rec, task, include_unit_details=True, session_id=SESSION,
                            channel="app", anchor_message_id=ANCHOR_2)
    await _settle()
    later = (await _analysis_messages())[_message_id(analysis_id, 2, 0)]
    assert later.content.startswith("Analysis of Handout.pdf (6 pages)\n\nOverview of 6")
    assert len(pushes) == 1


async def test_local_checkpoint_reconciler_resumes_the_same_way(monkeypatch, preflight):
    """Dev/monolith parity: the file-checkpoint branch uses the same helper.
    An automatic resume re-sends only pages never read: page 2 failed on
    its own (and may already have been billed three times), so it waits
    for the user to ask again."""
    model = BudgetModel()
    _pdf(monkeypatch, model)
    delivered = _capture_delivery(monkeypatch)
    await _seed_user()
    await _seed_chat(ANCHOR_1)
    await _stored()
    analysis_id = AA.analysis_id_for(TASK)
    now = datetime.now(timezone.utc)
    stopped = _stopped(analysis_id, read=(1,), blocked_until=now - timedelta(minutes=3))
    stopped["pages"].append({"page_number": 2, "status": "failed", "error": "unit_analysis_failed"})
    await AA.save_state(stopped)
    preflight.admit()

    await AA.reconcile_local_analyses()
    await _settle()
    final = AA.load_state(USER, AID, analysis_id)
    assert sorted(model.calls) == [3, 4, 5, 6]
    assert final["status"] == "partial" and final["auto_resume_count"] == 1
    assert delivered[0][0].startswith(
        "Earlier you asked me to read Handout.pdf. My AI budget has reset")
    assert "### Page 2\nCould not analyze this page." in delivered[0][1]

    await AA.reconcile_local_analyses()
    await _settle()
    assert sorted(model.calls) == [3, 4, 5, 6]


async def test_two_replicas_racing_to_resume_the_same_job_resume_it_once(monkeypatch):
    """No asyncio lock is shared across processes: the revision
    compare-and-swap alone decides, and the loser neither runs nor counts."""
    _agent_db(monkeypatch)
    await _seed_user()
    await _seed_chat(ANCHOR_1)
    await _stored()
    analysis_id = AA.analysis_id_for(TASK)
    await AA.save_state(_stopped(analysis_id, read=(1, 2),
                                 blocked_until=datetime.now(timezone.utc) - timedelta(minutes=3)))
    snapshot = await AA.load_state_async(USER, AID, analysis_id)

    class _PerProcessLocks(dict):
        def setdefault(self, key, default=None):
            return asyncio.Lock()

    arrived, asked = asyncio.Event(), []

    async def both_ask_first():            # both replicas have read the row before either commits
        asked.append(1)
        if len(asked) == 2:
            arrived.set()
        await asyncio.wait_for(arrived.wait(), 5)
        return {"blocked": False, "known": True, "period_end": None, "remaining_cents": 500.0}

    started: list[tuple] = []
    monkeypatch.setattr(AA, "_start_locks", _PerProcessLocks())
    monkeypatch.setattr(AA, "_budget_preflight", both_ask_first)
    monkeypatch.setattr(AA, "ensure_running", lambda *key: started.append(key))

    now = datetime.now(timezone.utc)
    results = await asyncio.gather(
        AA._auto_resume_one(copy.deepcopy(snapshot), now, AA._pass_verdict()),
        AA._auto_resume_one(copy.deepcopy(snapshot), now, AA._pass_verdict()))
    assert sorted(results) == [False, True]
    assert started == [(USER, AID, analysis_id)]
    row = await _row(analysis_id)
    assert row.status == "queued"
    assert row.state_json["auto_resume_count"] == 1 and row.state_json["delivery_attempt"] == 1


async def test_stop_recorded_before_blocked_until_existed_is_resumed_once(
        monkeypatch, preflight, broadcasts):
    """A stop the previous image wrote a few hours before the deploy:
    analysis_budget_exceeded, no blocked_until, no first_failed_at."""
    _agent_db(monkeypatch)
    model = BudgetModel()
    _pdf(monkeypatch, model)
    _capture_notify(monkeypatch)
    await _seed_user()
    await _seed_chat(ANCHOR_1)
    await _stored()
    analysis_id = AA.analysis_id_for(TASK)
    await AA.save_state(_legacy_stop(analysis_id))
    preflight.admit()

    await AA.reconcile_local_analyses()
    await _settle()
    final = await AA.load_state_async(USER, AID, analysis_id)
    assert final["status"] == "completed" and final["auto_resume_count"] == 1
    assert sorted(model.calls) == [1, 2, 3, 4, 5, 6]

    await AA.reconcile_local_analyses()
    await _settle()
    assert sorted(model.calls) == [1, 2, 3, 4, 5, 6]


async def test_a_fail_open_verdict_never_uses_up_the_one_time_backfill(
        monkeypatch, preflight, broadcasts):
    """The fleet rolled before the platform: /llm/usage has no openai_blocked,
    so there is no verdict. The legacy stop stays exactly as it was (and
    inside its backfill window) until the platform's own verdict admits it;
    between asks it waits (in memory), so a pass right after asks nothing."""
    _agent_db(monkeypatch)
    model = BudgetModel()
    _pdf(monkeypatch, model)
    _capture_notify(monkeypatch)
    later = _pass_clock(monkeypatch)
    await _seed_user()
    await _seed_chat(ANCHOR_1)
    await _stored()
    analysis_id = AA.analysis_id_for(TASK)
    await AA.save_state(_legacy_stop(analysis_id))
    before = await _row(analysis_id)

    await AA.reconcile_local_analyses()                # a check that failed open
    await _settle()
    await AA.reconcile_local_analyses()                # the next pass, right away: not asked again
    await _settle()
    assert preflight.calls == 1
    later(hours=1)
    preflight.block_unknown(None)                      # an older platform's "spent"
    await AA.reconcile_local_analyses()
    await _settle()
    row = await _row(analysis_id)
    assert row.revision == before.revision and row.state_json == before.state_json
    assert preflight.calls == 2 and model.calls == [] and await _analysis_messages() == {}

    later(hours=2)
    preflight.admit()
    await AA.reconcile_local_analyses()
    await _settle()
    final = await AA.load_state_async(USER, AID, analysis_id)
    assert final["status"] == "completed" and final["auto_resume_count"] == 1
    assert sorted(model.calls) == [1, 2, 3, 4, 5, 6]


async def test_no_platform_verdict_backs_off_per_job_until_a_verdict_starts_it_over(
        monkeypatch, preflight):
    """Without the platform's own verdict (fail-open) a due job was looked at
    on every 60-second pass for up to 7 or 35 days. Now it is asked about
    again after 1, 2, 4 ... at most 60 minutes, with no I/O in between and no
    row write; a known verdict starts the wait over."""
    _agent_db(monkeypatch)
    await _seed_user()
    await _seed_chat(ANCHOR_1)
    await _stored()
    started: list[tuple] = []
    monkeypatch.setattr(AA, "ensure_running", lambda *key: started.append(key))
    reads: list[tuple] = []
    real_load = AA.load_state_async

    async def counted_load(*key):
        reads.append(key)
        return await real_load(*key)

    monkeypatch.setattr(AA, "load_state_async", counted_load)
    analysis_id = AA.analysis_id_for(TASK)
    t0 = datetime.now(timezone.utc)
    await AA.save_state(_stopped(analysis_id, read=(1, 2), blocked_until=t0 - timedelta(minutes=3)))
    before = await _row(analysis_id)
    asked: list[int] = []

    async def passes(start, end):
        """One reconcile pass a minute, from ``start`` to ``end`` seconds after t0."""
        snapshot = dict((await _row(analysis_id)).state_json)
        for at in range(start, end, 60):
            calls = preflight.calls
            await AA._auto_resume_one(copy.deepcopy(snapshot), t0 + timedelta(seconds=at),
                                      AA._pass_verdict())
            if preflight.calls > calls:
                asked.append(at)

    reads.clear()
    await passes(0, 7200)                              # two hours with no verdict: 7 asks, not 120
    assert asked == [0, 60, 180, 420, 900, 1860, 3780]
    assert len(reads) == len(asked)                    # a waiting job costs no read at all
    row = await _row(analysis_id)
    assert row.revision == before.revision and row.state_json == before.state_json
    assert started == []

    reset = t0 + timedelta(days=30)                    # a verdict: still spent, a later reset
    preflight.block(AA.iso_utc(reset))
    await passes(7380, 7440)
    assert asked[-1] == 7380
    assert (await _row(analysis_id)).state_json["blocked_until"] == AA.iso_utc(reset)

    preflight.verdict = {"blocked": False, "known": False, "period_end": None, "remaining_cents": None}
    after = int((reset - t0).total_seconds()) + AA.AUTO_RESUME_GRACE_SECONDS + 60
    await passes(after, after + 300)                   # no verdict again: the wait starts over
    assert asked[-3:] == [after, after + 60, after + 180]

    preflight.admit()
    await passes(after + 300, after + 480)
    assert asked[-1] == after + 420 and started == [(USER, AID, analysis_id)]


async def test_reconciler_pass_while_still_blocked_only_moves_the_reset_date(monkeypatch, preflight):
    _agent_db(monkeypatch)
    model = BudgetModel()
    _pdf(monkeypatch, model)
    await _seed_user()
    await _seed_chat(ANCHOR_1)
    await _stored()
    analysis_id = AA.analysis_id_for(TASK)
    now = datetime.now(timezone.utc)
    await AA.save_state(_stopped(analysis_id, read=(1, 2), blocked_until=now - timedelta(minutes=3)))
    before = await _row(analysis_id)
    later = AA.iso_utc(now + timedelta(days=30))
    preflight.block(later)                             # e.g. a replica still counting last month

    await AA.reconcile_local_analyses()
    await _settle()
    row = await _row(analysis_id)
    assert row.state_json["blocked_until"] == later
    assert row.status == "partial" and row.revision == before.revision + 1
    assert row.state_json["auto_resume_count"] == 0    # a refused resume costs no attempt
    assert row.state_json["delivery_attempt"] == 0     # and posts no message
    assert model.calls == [] and await _analysis_messages() == {}

    await AA.reconcile_local_analyses()                # now waiting for the new date
    await _settle()
    assert preflight.calls == 1 and (await _row(analysis_id)).revision == row.revision


@pytest.mark.parametrize("period_end", [None, PAST_END], ids=["no_end", "end_already_past"])
async def test_spent_with_no_reset_date_is_checked_a_day_later_never_every_pass(
        monkeypatch, preflight, period_end):
    """E.g. the rolling window switched off: the platform says spent and
    names no end (or one already past, which is no reset to wait for). A
    dated stop waits a day between checks; a one-time backfill is
    answered, and retires."""
    _agent_db(monkeypatch)
    model = BudgetModel()
    _pdf(monkeypatch, model)
    await _seed_user()
    await _seed_chat(ANCHOR_1)
    await _stored()
    now = datetime.now(timezone.utc)
    dated, legacy = AA.analysis_id_for(TASK), "9" * 20
    await AA.save_state(_stopped(dated, read=(1, 2), blocked_until=now - timedelta(minutes=3)))
    await AA.save_state(_legacy_stop(legacy))
    preflight.block(period_end)

    await AA.reconcile_local_analyses()
    await _settle()
    row = await _row(dated)
    recheck = AA.parse_utc(row.state_json["next_resume_check_at"])
    assert timedelta(hours=23) < recheck - now < timedelta(hours=25)
    assert row.state_json["blocked_until"] == AA.iso_utc(now - timedelta(minutes=3))
    assert row.status == "partial" and row.state_json["auto_resume_count"] == 0
    assert row.state_json["delivery_attempt"] == 0
    retired = (await _row(legacy)).state_json
    assert retired["auto_resume_retired"] == "no_reset_date"
    assert retired["auto_resume_count"] == AA.AUTO_RESUME_LIMIT
    assert preflight.calls == 1 and model.calls == []

    for _ in range(3):
        await AA.reconcile_local_analyses()
        await _settle()
    assert preflight.calls == 1                         # nothing asks again for a day


@pytest.mark.parametrize("task,first", [
    (TASK,
     "Analysis of Handout.pdf (2 of 6 pages read)\n\n"
     "My AI budget ran out partway through, so I read pages 1–2, but not pages 3–6. "
     "Ask me again later. Your credits aren’t affected.\n\n"
     "Page-by-page details for pages 1–2 follow."),
    (FA_TASK,
     "تحلیل فایل «Handout.pdf» (۲ از ۶ صفحه خوانده شد)\n\n"
     "بودجهٔ هوش مصنوعی من وسط کار تمام شد؛ صفحه‌های ۱ تا ۲ را خواندم، ولی صفحه‌های ۳ تا ۶ "
     "را نه. بعداً دوباره از من بخواهید. این موضوع روی اعتبار شما اثری ندارد.\n\n"
     "جزئیات صفحه‌های ۱ تا ۲ در ادامه می‌آید."),
], ids=["en", "fa"])
async def test_still_spent_after_the_reset_with_no_new_date_is_told_undated(
        monkeypatch, preflight, broadcasts, task, first):
    """(F3.3) The reset a stop recorded has passed, but the platform still
    says spent and names no new reset: the reconciler keeps the past date
    and asks again a day later (next_resume_check_at). Until then the copy
    is the undated "used up" one. A delivery made then (here: one deferred
    until this pass) used to say "It has reset, so I’ll read it again
    shortly", and the tools "it has reset since … read again automatically
    shortly"."""
    _agent_db(monkeypatch)
    model = BudgetModel()
    _pdf(monkeypatch, model)
    await _seed_user()
    await _seed_chat(ANCHOR_1)
    await _stored()
    now = datetime.now(timezone.utc)
    analysis_id = AA.analysis_id_for(task)
    stopped = _stopped(analysis_id, read=(1, 2), blocked_until=now - timedelta(minutes=3), task=task)
    stopped["delivered"] = False                        # its delivery was deferred until this pass
    stopped["delivery_targets"][0]["delivered"] = False
    await AA.save_state(stopped)
    preflight.block(None)                               # still spent, no end named

    await AA.reconcile_local_analyses()
    await _settle()
    row = await _row(analysis_id)
    recheck = AA.parse_utc(row.state_json["next_resume_check_at"])
    assert recheck - now > timedelta(hours=23)          # the next check is a day away
    assert row.state_json["blocked_until"] == AA.iso_utc(now - timedelta(minutes=3))
    assert row.delivered is True and preflight.calls == 1 and model.calls == []
    posted = _first_parts(await _analysis_messages())
    assert posted == {SESSION: first}
    for word in ("shortly", "has reset", "به‌زودی", "تمدید شده"):
        assert word not in posted[SESSION]
    assert AA.blocked_guidance(row.state_json) == (
        "The AI budget is used up and no reset date is known. Nothing more was read. Tell the "
        "user that; do not describe unread pages and do not call read_attachment_analysis for "
        "them. The user can ask again later.")


@pytest.mark.parametrize("case", [
    "two_resumes_used", "reset_in_the_future", "inside_the_grace", "over_40_unread",
    "synthesis_over_40_calls", "no_destination", "conversation_deleted", "original_missing",
    "original_file_missing", "first_stop_36_days_ago", "no_reset_date_after_a_resume",
    "new_stop_without_a_date",
])
async def test_reconciler_leaves_ineligible_budget_stops_alone(monkeypatch, preflight, case):
    _agent_db(monkeypatch)
    model = BudgetModel()
    pages = {"over_40_unread": 60, "synthesis_over_40_calls": 38}.get(case, 6)
    _pdf(monkeypatch, model, pages=pages)
    await _seed_user()
    if case != "conversation_deleted":
        await _seed_chat(ANCHOR_1)
    if case != "original_missing":
        rec = await _stored()
    if case == "original_file_missing":                # the record is there, the stored file is not
        await get_storage_backend().delete(rec["attachment"]["storage_path"])
    now = datetime.now(timezone.utc)
    kw: dict = {"read": (1, 2), "blocked_until": now - timedelta(minutes=3), "pages": pages}
    retired = {
        "over_40_unread": "too_large", "synthesis_over_40_calls": "too_large",
        "no_destination": "no_destination", "conversation_deleted": "no_destination",
        "original_missing": "original_missing", "original_file_missing": "original_missing",
        "first_stop_36_days_ago": "too_old",
        "no_reset_date_after_a_resume": "no_reset_date", "new_stop_without_a_date": "no_reset_date",
    }.get(case)
    if case == "two_resumes_used":
        kw["auto_resume_count"] = 2
    elif case == "reset_in_the_future":
        kw["blocked_until"] = now + timedelta(hours=1)
    elif case == "inside_the_grace":
        kw["blocked_until"] = now - timedelta(seconds=60)
    elif case == "no_destination":
        kw["targets"] = [{"session_id": SESSION, "channel": "app", "anchor_message_id": ANCHOR_1,
                          "delivered": True, "delivery_failed": "destination_unavailable"}]
    elif case == "first_stop_36_days_ago":
        kw["first_failed_at"] = now - timedelta(days=36)
    elif case == "no_reset_date_after_a_resume":
        kw.update(blocked_until=None, auto_resume_count=1)
    elif case == "new_stop_without_a_date":       # recorded by this image: no one-time backfill
        kw["blocked_until"] = None
    analysis_id = AA.analysis_id_for(TASK)
    await AA.save_state(_stopped(analysis_id, **kw))
    preflight.admit()                              # the budget is back; the job's limits decide

    await AA.reconcile_local_analyses()
    await _settle()

    row = await _row(analysis_id)
    assert row.status == "partial" and row.state_json["delivery_attempt"] == 0
    assert (USER, AID, analysis_id) not in AA._active
    assert model.calls == [] and preflight.calls == 0  # no platform call for a job that cannot resume
    assert await _analysis_messages() == {}
    assert row.state_json.get("auto_resume_retired") == retired
    if retired:                                        # out of later scans for good
        assert row.state_json["auto_resume_count"] == AA.AUTO_RESUME_LIMIT


@pytest.mark.parametrize("fault", ["torn_record", "storage_error"])
async def test_an_original_that_cannot_be_read_is_not_taken_for_gone(
        monkeypatch, preflight, broadcasts, fault):
    """Only a confirmed absence retires a job for good. A record that exists
    but cannot be read (a torn write) or a storage error leaves the row
    exactly as it was for this pass (no platform or model call), and the
    next pass that can read the original resumes it."""
    _agent_db(monkeypatch)
    model = BudgetModel()
    _pdf(monkeypatch, model)
    _capture_notify(monkeypatch)
    await _seed_user()
    await _seed_chat(ANCHOR_1)
    rec = await _stored()
    analysis_id = AA.analysis_id_for(TASK)
    await AA.save_state(_stopped(analysis_id, read=(1, 2),
                                 blocked_until=datetime.now(timezone.utc) - timedelta(minutes=3)))
    before = await _row(analysis_id)
    preflight.admit()
    backend = get_storage_backend()
    readable = backend.exists
    if fault == "torn_record":
        await backend.put(CA._record_key(USER, AID), b'{"attachment_id": "')
    else:
        def unavailable(key):
            raise OSError("storage unavailable")

        monkeypatch.setattr(backend, "exists", unavailable)

    for _ in range(2):
        await AA.reconcile_local_analyses()
        await _settle()
    row = await _row(analysis_id)
    assert row.revision == before.revision and row.state_json == before.state_json
    assert "auto_resume_retired" not in row.state_json
    assert preflight.calls == 0 and model.calls == [] and await _analysis_messages() == {}

    if fault == "torn_record":                         # readable again
        await CA._write_json(CA._record_key(USER, AID), rec)
    else:
        monkeypatch.setattr(backend, "exists", readable)
    await AA.reconcile_local_analyses()
    await _settle()
    final = await AA.load_state_async(USER, AID, analysis_id)
    assert final["status"] == "completed" and final["auto_resume_count"] == 1
    assert sorted(model.calls) == [3, 4, 5, 6]


async def test_one_resume_per_tenant_per_pass_and_none_while_one_is_queued(monkeypatch, preflight):
    """Every stop of a tenant becomes due at the same reset. Each resume may
    spend the new window, so one per pass, and none while one is running."""
    _agent_db(monkeypatch)
    await _seed_user()
    await _seed_chat(ANCHOR_1)
    await _stored()
    started: list[tuple] = []
    monkeypatch.setattr(AA, "ensure_running", lambda *key: started.append(key))
    now = datetime.now(timezone.utc)
    jobs = ("1" * 20, "2" * 20)
    for analysis_id in jobs:
        await AA.save_state(_stopped(analysis_id, read=(1, 2), blocked_until=now - timedelta(minutes=3)))
    preflight.admit()

    await AA.reconcile_local_analyses()
    [resumed] = {key[2] for key in started}
    [waiting] = [analysis_id for analysis_id in jobs if analysis_id != resumed]
    assert (await _row(resumed)).status == "queued" and preflight.calls == 1
    untouched = await _row(waiting)
    assert untouched.status == "partial" and untouched.state_json["auto_resume_count"] == 0

    await AA.reconcile_local_analyses()                # the tenant is busy: the other waits
    assert {key[2] for key in started} == {resumed}
    assert (await _row(waiting)).revision == untouched.revision and preflight.calls == 1


async def test_an_automatic_resume_builds_only_missing_syntheses(monkeypatch, preflight, broadcasts):
    """A section that failed (and may have been billed three times) is kept
    like a failed page; the resume makes only the part and overview calls."""
    _agent_db(monkeypatch)
    model = BudgetModel()
    _pdf(monkeypatch, model)
    _capture_notify(monkeypatch)
    await _seed_user()
    await _seed_chat(ANCHOR_1)
    await _stored()
    analysis_id = AA.analysis_id_for(TASK)
    stopped = _stopped(analysis_id, read=range(1, 7),
                       blocked_until=datetime.now(timezone.utc) - timedelta(minutes=3))
    stopped["sections"] = [{"first_page": 1, "last_page": 6, "error": "section_analysis_failed"}]
    await AA.save_state(stopped)
    assert AA._pending_model_calls(stopped) == 2
    preflight.admit()

    await AA.reconcile_local_analyses()
    await _settle()
    final = await AA.load_state_async(USER, AID, analysis_id)
    assert model.calls == [] and model.stages == ["volume", "overview"]
    assert final["sections"] == stopped["sections"]
    assert final["status"] == "partial" and final["auto_resume_count"] == 1


def test_pending_model_calls_count_what_a_resume_would_spend():
    assert AA._pending_model_calls(_stopped("p" * 20)) == 6 + 1 + 1 + 1
    assert AA._pending_model_calls(_stopped("p" * 20, read=range(1, 7))) == 3
    assert AA._pending_model_calls(_stopped("p" * 20, pages=3)) == 3   # answered from its pages
    at_limit = _stopped("p" * 20, pages=35)                              # 35 + 3 sections + 1 + 1
    assert AA._pending_model_calls(at_limit) == AA.AUTO_RESUME_MAX_CALLS == 40
    assert AA._resume_limits(at_limit, FROZEN) == (True, "")
    over = _stopped("p" * 20, pages=36)
    assert AA._pending_model_calls(over) == 41
    assert AA._resume_limits(over, FROZEN) == (False, "too_large")
    assert AA._pending_model_calls(_stopped("p" * 20, read=range(1, 131), pages=130)) == 11 + 2 + 1
    parts = _stopped("p" * 20, pages=3)
    parts["unit_kind"] = "chunk"                         # 3 parts + a section, a part synthesis, the overview
    assert AA._pending_model_calls(parts) == 6 and AA._resume_limits(parts, FROZEN) == (True, "")
    unknown = _stopped("p" * 20, pages=0)                # a text file whose parts could not be counted
    unknown["unit_kind"] = "chunk"
    assert AA._pending_model_calls(unknown) is None      # unknown, never "too large" (F3.1)
    assert AA._resume_limits(unknown, FROZEN) == (False, "size_unknown")
    unknown["first_failed_at"] = (FROZEN - timedelta(days=36)).isoformat()
    assert AA._resume_limits(unknown, FROZEN) == (False, "too_old")   # a lasting limit still retires


async def test_the_resume_scan_finds_budget_stops_behind_older_failures(monkeypatch):
    """Rows are never deleted: a LIMIT over every failed row would starve."""
    from app.db.database import async_session_maker
    from app.db.models import AttachmentAnalysisJob

    _agent_db(monkeypatch)
    await _seed_user()
    now = datetime.now(timezone.utc)
    for i in range(25):                                # ordinary failures
        other = _stopped(f"{i:020d}", blocked_until=now - timedelta(minutes=3))
        other["error"] = {"code": "units_failed", "message": "No document units could be analyzed."}
        await AA.save_state(other)
    for i in range(25, 45):                            # automatic retries used up
        await AA.save_state(_stopped(f"{i:020d}", blocked_until=now - timedelta(minutes=3),
                                     auto_resume_count=2))
    due = _stopped(AA.analysis_id_for(TASK), read=(1, 2), blocked_until=now - timedelta(minutes=3))
    await AA.save_state(due)
    async with async_session_maker() as db:
        await db.execute(update(AttachmentAnalysisJob).where(
            AttachmentAnalysisJob.id != AA._job_id(USER, AID, due["analysis_id"]),
        ).values(updated_at=datetime.utcnow() - timedelta(days=1)))
        await db.commit()
        _now, candidates = await AA._auto_resume_candidates(db)
    assert [snapshot["analysis_id"] for snapshot, _updated in candidates] == [due["analysis_id"]]


# ── state-write fencing (spec B6) ────────────────────────────────────


async def test_a_stale_requeue_cannot_roll_back_a_finished_job(monkeypatch, preflight):
    _agent_db(monkeypatch)
    model = BudgetModel(refuse={3, 4, 5, 6})
    _pdf(monkeypatch, model)
    await _seed_user()
    rec = await _stored()
    state = await AA.start_analysis(USER, AID, rec, TASK)
    await _settle()
    analysis_id = state["analysis_id"]
    stale = await AA.load_state_async(USER, AID, analysis_id)   # partial, pages 1-2
    assert stale["status"] == "partial"

    model.refuse = set()
    await AA.start_analysis(USER, AID, rec, TASK, retry_failed=True)
    await _settle()
    assert (await AA.load_state_async(USER, AID, analysis_id))["status"] == "completed"
    sent = sorted(model.calls)
    assert sent == [1, 2, 3, 3, 4, 5, 6]               # page 3's first call was refused, not billed

    # A requeue computed from the old snapshot (a slow reconciler, another replica):
    AA._requeue_failed_units(stale, include_failed_units=True)
    assert await AA._fenced_commit(copy.deepcopy(stale)) is False   # compare-and-swap refuses it
    await AA.save_state(stale)                         # the unfenced save adopts the row instead
    assert stale["status"] == "completed"
    AA.ensure_running(USER, AID, analysis_id)
    await _settle()

    row = await _row(analysis_id)
    assert row.status == "completed" and row.state_json["overview"] == "Overview of 6"
    assert len(row.state_json["pages"]) == 6
    assert sorted(model.calls) == sent                 # no page sent again


async def test_finished_job_saves_accept_only_the_next_delivery_attempt(monkeypatch):
    _agent_db(monkeypatch)
    await _seed_user()
    analysis_id = AA.analysis_id_for(TASK)
    await AA.save_state(_stopped(analysis_id, read=(1, 2), blocked_until=PERIOD_END))
    loaded = await AA.load_state_async(USER, AID, analysis_id)

    redelivery = copy.deepcopy(loaded)                 # monotonic: attempt 0 -> 1 lands
    AA._bump_attempt(redelivery)
    await AA.save_state(redelivery)
    row = await _row(analysis_id)
    assert row.state_json["delivery_attempt"] == 1 and row.delivered is False

    stale = copy.deepcopy(loaded)                      # a deliverer still on attempt 0
    stale["delivery_targets"][0]["delivery_failed"] = "destination_unavailable"
    await AA.save_state(stale)
    assert stale["delivery_attempt"] == 1              # it now holds the row instead
    row = await _row(analysis_id)
    assert row.state_json["delivery_attempt"] == 1
    assert "delivery_failed" not in row.state_json["delivery_targets"][0]

    jump = copy.deepcopy(loaded)                       # skipping an attempt is refused too
    jump["delivery_attempt"] = 3
    await AA.save_state(jump)
    assert (await _row(analysis_id)).state_json["delivery_attempt"] == 1

    # The row is still partial, but a requeue computed from the old revision
    # (another replica's snapshot) must not land on it.
    stale_requeue = copy.deepcopy(loaded)
    AA._requeue_failed_units(stale_requeue, include_failed_units=True)
    assert await AA._fenced_commit(stale_requeue) is False
    assert (await _row(analysis_id)).status == "partial"

    requeued = await AA.load_state_async(USER, AID, analysis_id)
    AA._requeue_failed_units(requeued, include_failed_units=True)
    assert await AA._fenced_commit(requeued) is True
    finished = copy.deepcopy(loaded)                   # a finished write onto the requeued row
    finished["delivered"] = True
    await AA.save_state(finished)
    row = await _row(analysis_id)
    assert row.status == "queued" and finished["status"] == "queued"


async def test_the_next_attempt_from_a_stale_revision_is_adopted_not_written(monkeypatch):
    """attempt + 1 is accepted only from the latest revision: a snapshot
    read before the reconciler retired the job must not roll that back."""
    _agent_db(monkeypatch)
    await _seed_user()
    analysis_id = AA.analysis_id_for(TASK)
    await AA.save_state(_stopped(analysis_id, read=(1, 2), blocked_until=PERIOD_END))
    loaded = await AA.load_state_async(USER, AID, analysis_id)

    retired = copy.deepcopy(loaded)                    # same attempt, a newer revision
    retired.update(auto_resume_count=AA.AUTO_RESUME_LIMIT, auto_resume_retired="too_old")
    assert await AA._fenced_commit(retired) is True

    stale = copy.deepcopy(loaded)
    AA._bump_attempt(stale)                            # attempt 1, computed from the old revision
    await AA.save_state(stale)
    row = await _row(analysis_id)
    assert row.state_json["delivery_attempt"] == 0 and row.revision == retired["_claim_revision"]
    assert row.state_json["auto_resume_retired"] == "too_old"
    assert stale["delivery_attempt"] == 0 and stale["auto_resume_retired"] == "too_old"   # adopted


@pytest.mark.parametrize("delivered", [True, False],
                         ids=["row_already_delivered", "row_not_yet_delivered"])
async def test_a_chat_that_joins_as_the_job_finishes_still_gets_the_result(
        monkeypatch, preflight, broadcasts, delivered):
    """The job's terminal write lands between a second chat's read and its
    checkpoint. The finished row keeps it (never rolled back) and posts to it.
    ``row_not_yet_delivered`` is the worker's real terminal write: the join
    used to be appended in place to the list SQLAlchemy had loaded, so the
    UPDATE left state_json out and chat B was dropped."""
    _agent_db(monkeypatch)
    await _seed_user()
    await _seed_chat(ANCHOR_1)
    await _seed_conversation(SESSION_B)
    rec = await _stored()
    analysis_id = AA.analysis_id_for(TASK)
    running = AA.new_state(USER, AID, analysis_id, TASK, NAME, "page", 6, SESSION, "app", ANCHOR_1)
    running.update(status="running", stage="units")
    await AA.save_state(running)
    monkeypatch.setattr(AA, "ensure_running", lambda *key: None)   # the worker is elsewhere

    real_load = AA.load_state_async
    raced: list[str] = []

    async def the_worker_finishes_right_after_this_read(*key):
        snapshot = await real_load(*key)
        if not raced:
            raced.append(snapshot["status"])
            await _finish_row(analysis_id, delivered=delivered)
        return snapshot

    monkeypatch.setattr(AA, "load_state_async", the_worker_finishes_right_after_this_read)
    state = await AA.start_analysis(USER, AID, rec, TASK, retry_failed=True, session_id=SESSION_B,
                                    channel="app", anchor_message_id=None, request_identity=TASK,
                                    redeliver_when_blocked=False)
    joined = await _row(analysis_id)                   # the join itself reached the database
    assert [t["session_id"] for t in joined.state_json["delivery_targets"]] == [SESSION, SESSION_B]
    await _settle()
    assert raced == ["running"] and state["status"] == "completed"
    assert [t["session_id"] for t in state["delivery_targets"]] == [SESSION, SESSION_B]
    row = await _row(analysis_id)
    assert [t["session_id"] for t in row.state_json["delivery_targets"]] == [SESSION, SESSION_B]
    assert all(t["delivered"] for t in row.state_json["delivery_targets"])
    assert row.status == "completed" and row.state_json["overview"] == "Overview of 6"
    assert len(row.state_json["pages"]) == 6 and row.delivered is True
    first = _first_parts(await _analysis_messages())
    # A finished row that was already delivered had told the first chat.
    assert sorted(first) == ([SESSION_B] if delivered else [SESSION, SESSION_B])
    assert first[SESSION_B].startswith("Analysis of Handout.pdf (6 pages)\n\nOverview of 6")


class _GatedOverview(BudgetModel):
    """Holds the overview call until the test opens the gate."""

    def __init__(self):
        super().__init__()
        self.in_overview = asyncio.Event()
        self.gate = asyncio.Event()

    async def overview(self, task, count, volumes, failed, unit_kind, selected_range=None):
        self.in_overview.set()
        await asyncio.wait_for(self.gate.wait(), 10)
        return await super().overview(task, count, volumes, failed, unit_kind, selected_range)


async def test_a_chat_that_joins_while_the_worker_finishes_gets_the_result_end_to_end(
        monkeypatch, preflight, broadcasts):
    """The real worker in this process. Chat B's kickoff reads the running
    job while the worker waits on its overview; the worker then commits its
    terminal checkpoint (not delivered yet: its delivery waits for the start
    lock B's kickoff holds), and B's join lands on the finished row. The
    kickoff tells B the result will be posted there, so B must get it."""
    _agent_db(monkeypatch)
    model = _GatedOverview()
    _pdf(monkeypatch, model)
    await _seed_user()
    await _seed_chat(ANCHOR_1)
    await _seed_conversation(SESSION_B)
    rec = await _stored()
    analysis_id = AA.analysis_id_for(TASK)
    kickoff = dict(retry_failed=True, channel="app", request_identity=TASK, redeliver_when_blocked=False)

    first = await AA.start_analysis(USER, AID, rec, TASK, session_id=SESSION, anchor_message_id=ANCHOR_1,
                                    **kickoff)
    assert first["status"] == "queued"
    await asyncio.wait_for(model.in_overview.wait(), 10)

    real_load = AA.load_state_async
    raced: list = []

    async def read_then_let_the_worker_finish(*key):
        snapshot = await real_load(*key)
        if not raced:
            raced.append(snapshot["status"])
            model.gate.set()
            loop = asyncio.get_running_loop()
            deadline = loop.time() + 10
            while (await _row(analysis_id)).status != "completed":   # the terminal checkpoint
                assert loop.time() < deadline, "the worker never finished"
                await asyncio.sleep(0.01)
            raced.append((await _row(analysis_id)).delivered)
        return snapshot

    monkeypatch.setattr(AA, "load_state_async", read_then_let_the_worker_finish)
    joined = await AA.start_analysis(USER, AID, rec, TASK, session_id=SESSION_B, anchor_message_id=None,
                                     **kickoff)
    monkeypatch.setattr(AA, "load_state_async", real_load)
    assert raced == ["running", False]                 # finished, not delivered yet
    # agent_runner: "I found the completed analysis and will post it in this chat."
    assert joined["status"] == "completed" and not joined["delivered"]
    assert [t["session_id"] for t in joined["delivery_targets"]] == [SESSION, SESSION_B]
    await _settle()

    row = await _row(analysis_id)
    assert [t["session_id"] for t in row.state_json["delivery_targets"]] == [SESSION, SESSION_B]
    assert all(t["delivered"] for t in row.state_json["delivery_targets"]) and row.delivered is True
    parts = _first_parts(await _analysis_messages())
    assert sorted(parts) == [SESSION, SESSION_B]
    assert parts[SESSION_B].startswith("Analysis of Handout.pdf (6 pages)\n\nOverview of 6")
    assert sorted(model.calls) == [1, 2, 3, 4, 5, 6] and model.stages.count("overview") == 1


async def test_a_chat_that_joins_as_another_replica_resumes_the_job_gets_the_result(
        monkeypatch, preflight, broadcasts):
    """The mirror across processes: chat B's kickoff reads the stopped job,
    another replica's automatic resume requeues it (fenced on the revision),
    then B's join checkpoint lands on the queued row. B's snapshot is stale,
    but B's join is not: the requeued run posts to B as well (without the
    "earlier you asked" lead-in, which belongs to the chat that waited)."""
    _agent_db(monkeypatch)
    model = BudgetModel()
    _pdf(monkeypatch, model)
    pushes = _capture_notify(monkeypatch)
    await _seed_user()
    await _seed_chat(ANCHOR_1)
    await _seed_conversation(SESSION_B)
    rec = await _stored()
    analysis_id = AA.analysis_id_for(TASK)
    await AA.save_state(_stopped(analysis_id, read=(1, 2),
                                 blocked_until=datetime.now(timezone.utc) - timedelta(minutes=3)))
    preflight.admit()
    run = AA.ensure_running
    monkeypatch.setattr(AA, "ensure_running", lambda *key: None)   # nobody runs it yet
    real_load = AA.load_state_async
    raced: list = []

    async def read_then_another_replica_resumes(*key):
        snapshot = await real_load(*key)
        if not raced:
            raced.append(snapshot["status"])
            with pytest.MonkeyPatch.context() as other_replica:
                other_replica.setattr(AA, "_start_locks", _OtherProcessLocks())
                raced.append(await AA._auto_resume_one(
                    copy.deepcopy(snapshot), datetime.now(timezone.utc), AA._pass_verdict()))
        return snapshot

    monkeypatch.setattr(AA, "load_state_async", read_then_another_replica_resumes)
    state = await AA.start_analysis(USER, AID, rec, TASK, retry_failed=True, session_id=SESSION_B,
                                    channel="app", anchor_message_id=None, request_identity=TASK,
                                    redeliver_when_blocked=False)
    monkeypatch.setattr(AA, "load_state_async", real_load)
    assert raced == ["partial", True]
    assert state["status"] == "queued"                 # the kickoff says it is reading the file
    assert [t["session_id"] for t in state["delivery_targets"]] == [SESSION, SESSION_B]
    row = await _row(analysis_id)
    assert row.status == "queued" and row.state_json["auto_resume_count"] == 1
    assert [t["session_id"] for t in row.state_json["delivery_targets"]] == [SESSION, SESSION_B]
    assert row.state_json["resumed_sessions"] == [SESSION]

    run(USER, AID, analysis_id)                        # the requeued run, on either replica
    await _settle()
    parts = _first_parts(await _analysis_messages())
    assert sorted(parts) == [SESSION, SESSION_B]
    assert parts[SESSION].startswith(
        "Earlier you asked me to read Handout.pdf. My AI budget has reset, so here is the result.")
    assert parts[SESSION_B].startswith("Analysis of Handout.pdf (6 pages)\n\nOverview of 6")
    assert sorted(model.calls) == [3, 4, 5, 6] and len(pushes) == 1


@pytest.mark.parametrize("delivered", [True, False],
                         ids=["row_already_delivered", "row_not_yet_delivered"])
async def test_a_details_request_that_races_the_finish_posts_the_details_once(
        monkeypatch, preflight, broadcasts, delivered):
    """(F3.2) A page-details request reads the running job; another replica
    finishes it (in ``row_already_delivered`` it also posts the result,
    without details) before the request's checkpoint lands. The details are
    posted once, in a new delivery attempt either way, and the user's next
    page-by-page request posts nothing more. The flag used to be switched on
    over the posted result with no re-delivery, so neither request posted any
    details (cross-process only: in one process the delivery waits for the
    request's start lock)."""
    _agent_db(monkeypatch)
    await _seed_user()
    await _seed_chat(ANCHOR_1, ANCHOR_2)
    rec = await _stored()
    task = "Summarize this file"                       # no page details asked for at first
    analysis_id = AA.analysis_id_for(task)
    running = AA.new_state(USER, AID, analysis_id, task, NAME, "page", 6, SESSION, "app", ANCHOR_1)
    running.update(status="running", stage="units")
    assert running["include_unit_details"] is False
    await AA.save_state(running)
    monkeypatch.setattr(AA, "ensure_running", lambda *key: None)   # the worker is on another replica

    real_load = AA.load_state_async
    raced: list[str] = []

    async def another_replica_finishes_right_after_this_read(*key):
        snapshot = await real_load(*key)
        if not raced:
            raced.append(snapshot["status"])
            await _finish_row(analysis_id, delivered=delivered)
        return snapshot

    monkeypatch.setattr(AA, "load_state_async", another_replica_finishes_right_after_this_read)
    details = dict(include_unit_details=True, session_id=SESSION, channel="app",
                   anchor_message_id=ANCHOR_2)
    state = await AA.start_analysis(USER, AID, rec, task, **details)
    monkeypatch.setattr(AA, "load_state_async", real_load)
    await _settle()
    assert raced == ["running"] and state["status"] == "completed"
    attempt = 1                                        # newly asked details: a new attempt
    posted = await _analysis_messages()
    assert set(posted) == {_message_id(analysis_id, attempt, 0), _message_id(analysis_id, attempt, 1)}
    assert posted[_message_id(analysis_id, attempt, 0)].content == (
        "Analysis of Handout.pdf (6 pages)\n\nOverview of 6\n\nNumbered details follow in batches.")
    assert "### Page 6\nPage 6: evidence" in posted[_message_id(analysis_id, attempt, 1)].content
    row = await _row(analysis_id)
    assert row.state_json["include_unit_details"] is True and row.delivered is True
    assert row.state_json["delivery_attempt"] == attempt

    # The user asks for the page-by-page details again: already posted, nothing more.
    await AA.start_analysis(USER, AID, rec, task, **details)
    await _settle()
    assert set(await _analysis_messages()) == set(posted)
    assert (await _row(analysis_id)).state_json["delivery_attempt"] == attempt
    assert preflight.calls == 0


async def test_details_that_land_while_another_replica_is_posting_are_still_posted(
        monkeypatch, preflight, broadcasts):
    """(F3.2, cross-replica) Replica X has loaded the finished row — no page
    details on it — and is inside ``_deliver_to_target``, before its first
    delivery write, when a page-details request's checkpoint lands. Nothing
    is marked delivered yet. The merge used to only switch the flag on: X
    then posted the overview without details and marked the row delivered,
    the request's own delivery found it delivered and stopped, and the
    user's next page-by-page request found the flag on — the details were
    never posted. Now the merge starts a new attempt: X stops at its next
    checkpoint, and the details are posted exactly once. Prove it on
    Postgres too (the SQLite test engine shares one connection)."""
    _agent_db(monkeypatch)
    await _seed_user()
    await _seed_chat(ANCHOR_1, ANCHOR_2)
    rec = await _stored()
    task = "Summarize this file"                       # no page details asked for at first
    analysis_id = AA.analysis_id_for(task)
    running = AA.new_state(USER, AID, analysis_id, task, NAME, "page", 6, SESSION, "app", ANCHOR_1)
    running.update(status="running", stage="units")
    assert running["include_unit_details"] is False
    await AA.save_state(running)
    monkeypatch.setattr(AA, "ensure_running", lambda *key: None)   # the worker is on another replica

    real_load, real_target, real_locks = AA.load_state_async, AA._deliver_to_target, AA._start_locks
    x_inside, x_gate, x_done = asyncio.Event(), asyncio.Event(), asyncio.Event()
    replica_x: dict = {}

    async def x_waits_before_its_first_write(state, target):
        if asyncio.current_task() is replica_x.get("task"):
            x_inside.set()
            await asyncio.wait_for(x_gate.wait(), 10)
        return await real_target(state, target)

    async def x_delivers(snapshot):
        try:
            await AA.deliver_analysis(snapshot)
        finally:
            x_done.set()

    async def this_replica_reads_after_x(*key):        # the request's later reads
        if asyncio.current_task() is not replica_x.get("task"):
            await asyncio.wait_for(x_done.wait(), 10)
        return await real_load(*key)

    async def read_then_x_finishes_and_starts_posting(*key):
        snapshot = await real_load(*key)
        if "task" not in replica_x:
            await _finish_row(analysis_id, delivered=False)
            finished = await real_load(*key)            # what replica X loaded: no details
            assert finished["status"] == "completed" and not finished["include_unit_details"]
            monkeypatch.setattr(AA, "_start_locks", _OtherProcessLocks())
            replica_x["task"] = asyncio.create_task(x_delivers(finished))
            await asyncio.wait_for(x_inside.wait(), 10)
            monkeypatch.setattr(AA, "_start_locks", real_locks)
            monkeypatch.setattr(AA, "load_state_async", this_replica_reads_after_x)
        return snapshot

    monkeypatch.setattr(AA, "_deliver_to_target", x_waits_before_its_first_write)
    monkeypatch.setattr(AA, "load_state_async", read_then_x_finishes_and_starts_posting)
    details = dict(include_unit_details=True, session_id=SESSION, channel="app",
                   anchor_message_id=ANCHOR_2)
    state = await AA.start_analysis(USER, AID, rec, task, **details)
    merged = (await _row(analysis_id)).state_json
    assert merged["include_unit_details"] is True
    assert merged["delivery_attempt"] == 1              # the merge started the details attempt
    assert state["status"] == "completed"
    x_gate.set()
    await _settle()
    monkeypatch.setattr(AA, "load_state_async", real_load)
    monkeypatch.setattr(AA, "_deliver_to_target", real_target)

    posted = await _analysis_messages()
    # Replica X may have posted its overview (attempt 0, no details) before
    # it saw the new attempt; it posted no details, and nothing else.
    assert set(posted) - {_message_id(analysis_id, 0, 0)} == {
        _message_id(analysis_id, 1, 0), _message_id(analysis_id, 1, 1)}
    assert "### Page 6\nPage 6: evidence" in posted[_message_id(analysis_id, 1, 1)].content
    row = await _row(analysis_id)
    assert row.state_json["include_unit_details"] is True and row.delivered is True
    assert row.state_json["delivery_attempt"] == 1

    # The user asks for the page-by-page details again: already posted, nothing more.
    await AA.start_analysis(USER, AID, rec, task, **details)
    await _settle()
    assert set(await _analysis_messages()) == set(posted)
    assert (await _row(analysis_id)).state_json["delivery_attempt"] == 1
    assert preflight.calls == 0


# ── the pieces ───────────────────────────────────────────────────────


def _usage_transport(monkeypatch, handler):
    seen: list = []

    def factory(timeout=120.0):
        seen.append(timeout)
        return httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=timeout)

    monkeypatch.setattr(bundle_client, "_proxy_http_client", factory)
    return seen


_USAGE = {"openai_monthly_cents": 1012.5, "openai_budget_cents": 1000,
          "anthropic_monthly_cents": 0.0, "anthropic_daily_cents": 0.0,
          "anthropic_budget_cents": 0, "anthropic_daily_cap_cents": 0,
          "period_start": PERIOD_START, "period_end": PERIOD_END}


def _json(body):
    return lambda request: httpx.Response(200, json=body)


def _raise_connect(request):
    raise httpx.ConnectError("platform unreachable", request=request)


@pytest.mark.parametrize("reply,expected", [
    (_json(dict(_USAGE, openai_blocked=True, budget_exempt=False, openai_remaining_cents=-12.5)),
     {"blocked": True, "known": True, "period_end": PERIOD_END, "remaining_cents": -12.5}),
    (_json(dict(_USAGE, openai_blocked=True, openai_remaining_cents=0, period_end=NAIVE_PERIOD_END)),
     {"blocked": True, "known": True, "period_end": PERIOD_END, "remaining_cents": 0.0}),
    (_json(dict(_USAGE, openai_blocked=False, openai_remaining_cents=640.0)),
     {"blocked": False, "known": True, "period_end": PERIOD_END, "remaining_cents": 640.0}),
    (_json(dict(_USAGE, openai_blocked=False, budget_exempt=True, openai_remaining_cents=1e12)),
     {"blocked": False, "known": True, "period_end": PERIOD_END, "remaining_cents": 1e12}),
    # A platform without openai_blocked: its remaining figure decides, but it is no verdict...
    (_json(dict(_USAGE, openai_remaining_cents=0)),
     {"blocked": True, "known": False, "period_end": PERIOD_END, "remaining_cents": 0.0}),
    # ...never for an exempt admin...
    (_json(dict(_USAGE, openai_remaining_cents=-5, admin_unlimited=True)),
     {"blocked": False, "known": False, "period_end": PERIOD_END, "remaining_cents": -5.0}),
    # ...and a platform with neither never blocks (budget - spend is not the gate).
    (_json(_USAGE), {"blocked": False, "known": False, "period_end": PERIOD_END, "remaining_cents": None}),
], ids=["blocked", "offset_less_period", "admitted", "exempt", "older_platform_spent",
        "older_platform_admin", "oldest_platform"])
async def test_budget_preflight_reads_the_platform_verdict(monkeypatch, bundle_mode, reply, expected):
    requests: list[httpx.Request] = []

    def handler(request):
        requests.append(request)
        return reply(request)

    timeouts = _usage_transport(monkeypatch, handler)
    assert await AA._budget_preflight() == expected
    [request] = requests
    assert request.method == "GET" and str(request.url) == "http://platform.test/api/llm/usage"
    assert request.headers["authorization"] == "Bearer synthetic-token"
    assert timeouts == [AA.BUDGET_PREFLIGHT_TIMEOUT]


@pytest.mark.parametrize("handler", [
    _raise_connect,
    lambda request: httpx.Response(500, json={"detail": "Internal Server Error"}),
    lambda request: httpx.Response(200, text="<!doctype html><html><body>Toup</body></html>",
                                   headers={"content-type": "text/html"}),
    _json([1, 2, 3]),
], ids=["connect_error", "server_error", "spa_html", "not_an_object"])
async def test_budget_preflight_fails_open(monkeypatch, bundle_mode, handler):
    """The proxy's own gate still refuses a spent budget; a broken check never
    blocks, and is never taken for the platform's verdict."""
    _usage_transport(monkeypatch, handler)
    assert await AA._budget_preflight() == {
        "blocked": False, "known": False, "period_end": None, "remaining_cents": None}


@pytest.mark.parametrize("mode,token", [("manual", "synthetic-token"), ("bundle", "")])
async def test_budget_preflight_outside_bundle_mode_makes_no_request(monkeypatch, mode, token):
    seen = _usage_transport(monkeypatch, lambda request: httpx.Response(500))
    monkeypatch.setattr(settings, "llm_mode", mode)
    monkeypatch.setattr(settings, "toup_token", token)
    verdict = await AA._budget_preflight()
    assert verdict["blocked"] is False and verdict["known"] is False
    assert seen == []


async def test_retry_takes_the_reset_from_the_refusal_body_and_never_retries_it():
    calls: list[str] = []

    def refusing(error):
        async def call():
            calls.append("sent")
            raise error
        return call

    with pytest.raises(AA.SummaryBudgetExceeded) as typed:
        await AA._retry(refusing(budget_429(NAIVE_PERIOD_END)))
    assert typed.value.period_end == PERIOD_END and calls == ["sent"]

    with pytest.raises(AA.SummaryBudgetExceeded) as legacy:
        await AA._retry(refusing(budget_429(header=False, typed=False)))
    assert legacy.value.period_end is None             # the old proxy sends no date

    with pytest.raises(AA.SummaryBudgetExceeded) as undated:
        await AA._retry(refusing(budget_429(None)))    # Retry-After (604800) is not a reset date
    assert undated.value.period_end is None
    assert calls == ["sent"] * 3


async def test_a_credit_refusal_carries_its_reason():
    async def refused():
        raise credits_402("daily_cap_exceeded")

    with pytest.raises(AA.SummaryCreditsExhausted) as caught:
        await AA._retry(refused)
    assert caught.value.reason == "daily_cap_exceeded"


async def test_a_stopped_job_ends_a_retry_backoff_early():
    calls: list[int] = []

    async def flaky():
        calls.append(1)
        raise RuntimeError("upstream hiccup")

    started = time.monotonic()
    with pytest.raises(AA._RetryStopped):
        await AA._retry(flaky, stop=lambda: True)
    assert calls == [1] and time.monotonic() - started < 0.5


async def _sdk_error(status, payload, headers=None):
    """The exception the pinned openai SDK really raises for this response."""
    client = AsyncOpenAI(
        api_key="synthetic-token", base_url="http://platform.test/api/llm/openai/v1", max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(status, json=payload, headers=headers or {}))),
    )
    try:
        await client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "page 1"}])
    except APIStatusError as exc:
        return exc
    finally:
        await client.close()
    raise AssertionError("the mock platform did not refuse the call")


@pytest.mark.parametrize("payload,headers,expected", [
    ({"detail": "Too Many Requests"}, {"X-Toup-Reason": REASON}, "budget"),
    ({"detail": TYPED_DETAIL}, {}, "budget"),
    ({"detail": "Monthly OpenAI budget exceeded"}, {}, "budget"),
    ({"detail": "Rate limit exceeded for this tenant token."}, {"Retry-After": "12"}, None),
    ({"error": {"message": "Rate limit reached for gpt-4o", "type": "requests",
                "code": "rate_limit_exceeded"}}, {}, None),
    ({"detail": json.dumps({"error": {"message": "You exceeded your current quota",
                                      "type": "insufficient_quota", "code": "insufficient_quota"}})},
     {}, "service_quota"),
], ids=["header_only", "body_only", "legacy_sentence", "per_minute_limiter", "upstream_rate_limit",
        "upstream_quota"])
async def test_proxy_limit_reason_recognises_the_budget_refusal_however_it_arrives(
        payload, headers, expected):
    exc = await _sdk_error(429, payload, headers)
    assert AA._proxy_limit_reason(exc) == expected


# ── image tools (fix pass item 15) ───────────────────────────────────


@pytest.fixture
def image_rig(monkeypatch, tmp_path):
    """A ToolExecutor whose image path reaches the OpenAI call with every
    advisory hop off, and whose OpenAI calls the proxy refuses on budget."""
    from app.agent._user_tz_cache import set_cached_user_tz
    from app.agent.tool_executor import ToolExecutor

    for name, value in (("image_provider", "openai"), ("image_gen_enabled", True),
                        ("image_edit_enabled", True), ("image_grounding_enabled", False),
                        ("image_source_describe_enabled", False),
                        ("image_gen_model", "gpt-image-2"), ("image_gen_fallback_model", "gpt-image-1")):
        monkeypatch.setattr(settings, name, value, raising=False)
    monkeypatch.setattr(bundle_client, "make_openai_client", lambda **kwargs: object())
    executor = ToolExecutor(workspace=str(tmp_path))
    executor.set_user_id(USER)
    set_cached_user_tz(USER, TZ)
    sent: list[str] = []

    async def refused(client, model, *args):
        sent.append(model)
        raise budget_429(PERIOD_END)

    async def no_expansion(system, instruction):
        return ""

    monkeypatch.setattr(executor, "_openai_generate_image", refused)
    monkeypatch.setattr(executor, "_openai_edit_image", refused)
    monkeypatch.setattr(executor, "_expand_scene", no_expansion)
    return executor, sent


async def test_image_tools_answer_a_budget_refusal_with_the_job_sentence(image_rig):
    """The raw 429 text (the proxy's typed detail) never reaches the model,
    and the fallback model, gated by the same budget, is not asked."""
    from PIL import Image
    from app.services import budget_refusal as br

    executor, sent = image_rig
    expected = "ERROR: " + br.job_sentence(br.budget_refusal_detail(budget_429(PERIOD_END)), TZ)

    generated = await executor._tool_generate_image({"prompt": "A lighthouse at dusk"})
    assert generated == expected and sent == ["gpt-image-2"]

    source = io.BytesIO()
    Image.new("RGB", (16, 16), (10, 20, 30)).save(source, format="PNG")
    workspace = executor._get_user_workspace()
    os.makedirs(workspace, exist_ok=True)
    with open(os.path.join(workspace, "source.png"), "wb") as fh:
        fh.write(source.getvalue())
    sent.clear()
    edited = await executor._tool_edit_image({"prompt": "Make the sky orange", "image": "source.png"})
    assert edited == expected and sent == ["gpt-image-2"]

    for text in (generated, edited):
        assert "Error code" not in text and "429" not in text and "2026-" not in text
        assert REASON not in text
