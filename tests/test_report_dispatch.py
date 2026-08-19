"""Screenshot reports → the Admin thread — the support loop, end to end.

A user files a report from the app (``POST /support/issues`` then the
screenshot upload). It has always gone to email; now it ALSO opens as a
message in the user's Admin thread, so it lands in Conversations with its
screenshot and severity and the operator answers it there. This file proves
that loop the way the brief states it:

  user sends report → appears in Conversations with screenshot + severity
  → admin replies → reply arrives in the user's chat (persistent card, Reply
  action) and their Admin thread → user answers → answer shows in the admin
  thread.

Everything here runs RUN_MODE=platform: the whole loop is platform-side —
support intake, the thread, the dispatch row, the notification row. The one
tenant hop (the chat card) is the fan-out's ordinary agent hop, asserted on
the recorded call the way every other dispatch test does.

What each test defends:

  1. Intake opens ONE report row: kind='report', the reporter's severity,
     the parsed context (screen / app / build / device / OS / platform), a
     uuid5 id — and the note as the body, nothing prefixed. Email still goes.
  2. The screenshot upload (a SEPARATE request) finds that row and hangs the
     picture on it; the admin fetches it scoped by user, the owner fetches it,
     a stranger gets 404; the same bytes twice is one picture. An issue with
     no report row (pre-feature, or filed by another path) is a no-op.
  3. A second report is a second CARD in the same conversation — one row in
     the list (D3), report_count 2, badge = loudest open severity.
  4. Answering an open report is a persistent dispatch to that user: the
     thread row (uuid5, pre-written, never duplicated by the fan-out), the
     announcement push, and the agent hop carrying `REPORT_REPLY_TITLE`. It
     closes the report — the next reply is thread-only, as before — and the
     list badge falls from open to answered. Kill switch off ⇒ thread-only.
  5. The user's side of the loop: their inbox shows the answer unread; their
     reply is the next inbound row in the admin thread.
  6. A thread that could not be opened never fails intake; a report deleted
     for everyone keeps its kind and severity but loses its context and
     picture; the open/answered predicate ignores tombstoned reports and
     counts a deleted operator turn as an answer.
"""

from __future__ import annotations

import asyncio
import json
import types
import uuid
from datetime import datetime, timedelta

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select

from app.api.admin.dispatch import router as dispatch_router
from app.api.auth import get_current_user
from app.api.notices import router as notices_router
from app.api.support import router as support_router
from app.config import settings
from app.db import get_db, async_session_maker
from app.db.models import (
    AdminDispatch,
    AdminDispatchTarget,
    AdminThreadAttachment,
    AdminThreadMessage,
    AgentConfig,
    NotificationQueue,
    ProductEvent,
    PE_DISPATCH_CREATED,
    PE_REPORT_FILED,
    THREAD_IN,
    THREAD_KIND_REPORT,
    THREAD_OUT,
    User,
)
from app.services import report_thread
from app.support import repository as repo


# ── RUN_MODE precondition (same stance as test_admin_dispatch.py) ─────
def _require_platform_run_mode() -> None:
    mode = (settings.run_mode or "").strip().lower()
    if mode != "platform":
        raise RuntimeError(
            f"test_report_dispatch.py needs RUN_MODE=platform (got {mode!r}).\n"
            "    RUN_MODE=platform pytest tests/test_report_dispatch.py"
        )


_require_platform_run_mode()


# ── Harness ───────────────────────────────────────────────────────

def _principal(uid: str, role: str = "user", email: str = "u@x.com"):
    return types.SimpleNamespace(id=uid, role=role, email=email, name="U")


class _Spawns:
    """Records what the support router hands to `pipeline.spawn` — the
    diagnosis pipeline (an LLM call) and the card email — and closes the
    coroutines instead of running them. The test asserts the email was
    STILL spawned: reports go to both destinations."""

    def __init__(self):
        self.names: list[str] = []

    def __call__(self, coro):
        self.names.append(getattr(coro, "__name__", repr(coro)))
        coro.close()


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class _AgentSpy:
    def __init__(self):
        self.calls: list[dict] = []

    async def post(self, url, *, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers or {}, "json": json, "timeout": timeout})
        return _FakeResponse(201, {"message_id": f"msg-{len(self.calls)}", "day_chat_id": None, "ws_count": 0})


@pytest_asyncio.fixture
async def loop(monkeypatch):
    """The three routers under test on one app, the support kill switch on,
    the off-request spawns recorded, the fan-out spawn recorded (run by hand),
    and the agent HTTP client replaced by a spy."""
    monkeypatch.setattr(settings, "support_agent_enabled", True, raising=False)
    monkeypatch.setattr(settings, "support_notify_enabled", False, raising=False)

    spawns = _Spawns()
    monkeypatch.setattr("app.api.support.pipeline.spawn", spawns)

    fanouts: list[str] = []

    async def _record_spawn(dispatch_id: str) -> None:
        fanouts.append(dispatch_id)

    monkeypatch.setattr("app.api.admin.dispatch.spawn_dispatch_fanout", _record_spawn)

    spy = _AgentSpy()
    from app.services import agent_http
    monkeypatch.setattr(agent_http, "get_agent_http_client", lambda: spy)

    app = FastAPI()
    app.include_router(support_router, prefix=settings.api_prefix)
    app.include_router(dispatch_router, prefix=settings.api_prefix)
    app.include_router(notices_router, prefix=settings.api_prefix)

    async def _override_db():
        async with async_session_maker() as db:
            yield db

    app.dependency_overrides[get_db] = _override_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://t") as c:
        yield types.SimpleNamespace(client=c, app=app, spawns=spawns, fanouts=fanouts, spy=spy)


async def _mk_user(*, role: str = "user") -> str:
    from app.services.auth_service import get_password_hash

    user_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(
            id=user_id,
            email=f"rp-{uuid.uuid4().hex[:10]}@example.com",
            hashed_password=get_password_hash("x" * 12),
            name="Report Test",
            role=role,
            timezone="America/Toronto",
            created_at=datetime.utcnow(),
            notification_preferences={
                "quiet_hours": {"enabled": False, "start": "22:00", "end": "08:00"},
            },
        ))
        await db.commit()
    return user_id


async def _mk_agent(user_id: str) -> str:
    key = f"tk-{uuid.uuid4().hex}"
    async with async_session_maker() as db:
        db.add(AgentConfig(user_id=user_id, agent_api_key=key,
                           agent_url="https://agent.example", deploy_status="active"))
        await db.commit()
    return key


def _as(loop, uid: str, role: str = "user"):
    loop.app.dependency_overrides[get_current_user] = lambda: _principal(uid, role=role)


CONTEXT = "Screen: Chat\nApp: 1.2.0 (40)\nDevice: iPhone 15 Pro · iOS 18.5\nPlatform: ios"
PNG = ("shot.png", b"\x89PNG\r\n\x1a\n" + b"x" * 64, "image/png")


async def _file_report(loop, uid: str, *, note="The send button does nothing", severity="high",
                       repro=CONTEXT, channel="mobile") -> str:
    _as(loop, uid)
    r = await loop.client.post(f"{settings.api_prefix}/support/issues", json={
        "raw_report": note, "channel": channel, "repro_info": repro, "severity": severity,
    })
    assert r.status_code == 200, r.text
    return r.json()["id"]


async def _thread_rows(uid: str) -> list[AdminThreadMessage]:
    async with async_session_maker() as db:
        return list((await db.execute(
            select(AdminThreadMessage)
            .where(AdminThreadMessage.user_id == uid)
            .order_by(AdminThreadMessage.created_at.asc())
        )).scalars().all())


async def _events(event: str) -> list[ProductEvent]:
    async with async_session_maker() as db:
        return list((await db.execute(
            select(ProductEvent).where(ProductEvent.event == event)
            .order_by(ProductEvent.created_at.asc())
        )).scalars().all())


async def _run_fanout(dispatch_id: str) -> dict:
    from app.services.admin_dispatch_worker import run_dispatch_fanout
    return await run_dispatch_fanout(dispatch_id)


# ── 1. intake opens the report row ────────────────────────────────

@pytest.mark.asyncio
async def test_intake_opens_one_report_row_with_severity_context_and_a_uuid5_id(loop):
    uid = await _mk_user()
    issue_id = await _file_report(loop, uid, note="  Radio stops when I lock the phone  ", severity="critical")

    rows = await _thread_rows(uid)
    assert len(rows) == 1, "one report → one row, and nothing else in the thread"
    m = rows[0]
    assert m.id == report_thread.report_message_id(issue_id), "the id is a uuid5 of the issue"
    assert m.direction == THREAD_IN and m.kind == THREAD_KIND_REPORT
    assert m.severity == "critical"
    assert m.body == "Radio stops when I lock the phone", "the body is the note and ONLY the note"
    assert m.dispatch_id is None and m.author_admin_id is None and m.sender_name is None
    assert m.admin_read_at is None, "a fresh report is unread on the operator's side"

    rj = m.report_json
    assert rj["support_issue_id"] == issue_id and rj["channel"] == "mobile"
    ctx = rj["context"]
    assert ctx["screen"] == "Chat" and ctx["app_version"] == "1.2.0" and ctx["build"] == "40"
    assert ctx["device"] == "iPhone 15 Pro" and ctx["os"] == "iOS 18.5" and ctx["platform"] == "ios"
    assert ctx["raw"] == CONTEXT, "the device's own words survive the parser"

    # Both destinations: the thread row above AND the email spawn (recorded,
    # not run). The diagnosis pipeline is the other spawn.
    assert "_notify_admin_new_card" in loop.spawns.names, loop.spawns.names
    assert "run_diagnosis_pipeline" in loop.spawns.names

    # The funnel start, once, keyed on the issue.
    ev = await _events(PE_REPORT_FILED)
    mine = [e for e in ev if e.user_id == uid]
    assert len(mine) == 1
    assert mine[0].payload_json["severity"] == "critical" and mine[0].payload_json["support_issue_id"] == issue_id
    assert mine[0].payload_json["channel"] == "mobile"
    assert "Radio" not in json.dumps(mine[0].payload_json), "length and labels, never the note"

    # …and it is on the operator's list, badged, unread, open.
    admin = await _mk_user(role="admin")
    _as(loop, admin, role="admin")
    r = await loop.client.get(f"{settings.api_prefix}/admin/dispatch/threads")
    assert r.status_code == 200
    row = next(t for t in r.json()["threads"] if t["user_id"] == uid)
    assert row["report_severity"] == "critical" and row["report_open"] is True
    assert row["report_count"] == 1 and row["unread_in"] == 1
    assert row["last_direction"] == "in" and row["last_body"] == "Radio stops when I lock the phone"

    # The thread GET carries the same fields per message, plus the open report.
    r = await loop.client.get(f"{settings.api_prefix}/admin/dispatch/threads/{uid}")
    assert r.status_code == 200
    body = r.json()
    assert body["open_report"] == {"id": m.id, "severity": "critical", "count": 1, "ids": [m.id]}
    assert body["report_count"] == 1
    msg = body["messages"][0]
    assert msg["kind"] == "report" and msg["severity"] == "critical"
    assert msg["report"]["support_issue_id"] == issue_id
    assert msg["report"]["context"]["build"] == "40"
    assert msg["attachments"] == []


@pytest.mark.asyncio
async def test_a_structured_context_block_wins_over_the_parsed_lines(loop):
    uid = await _mk_user()
    _as(loop, uid)
    r = await loop.client.post(f"{settings.api_prefix}/support/issues", json={
        "raw_report": "web report", "channel": "web", "repro_info": CONTEXT, "severity": "low",
        "context": {"screen": "Settings › Billing", "platform": "Web", "bogus": "dropped",
                    "build": "x" * 500},
    })
    assert r.status_code == 200, r.text
    m = (await _thread_rows(uid))[0]
    ctx = m.report_json["context"]
    assert ctx["screen"] == "Settings › Billing"
    assert ctx["platform"] == "web", "platform is normalised to lower case"
    assert ctx["app_version"] == "1.2.0", "unspecified fields still come from the parsed lines"
    assert len(ctx["build"]) == 200, "every value is capped"
    assert "bogus" not in ctx


@pytest.mark.asyncio
async def test_a_thread_that_cannot_open_never_fails_intake(loop, monkeypatch):
    async def _boom(*a, **k):
        raise RuntimeError("platform db hiccup")

    monkeypatch.setattr(report_thread, "open_report_in_thread", _boom)
    uid = await _mk_user()
    issue_id = await _file_report(loop, uid)
    async with async_session_maker() as db:
        assert await repo.get_issue(db, issue_id) is not None, "the card is filed"
    assert await _thread_rows(uid) == []
    assert "_notify_admin_new_card" in loop.spawns.names, "and the email still goes"


# ── 2. the screenshot ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_the_screenshot_upload_finds_the_report_row_and_both_sides_can_fetch_it(loop):
    uid = await _mk_user()
    issue_id = await _file_report(loop, uid)
    _as(loop, uid)
    up = await loop.client.post(f"{settings.api_prefix}/support/issues/{issue_id}/attachment",
                                files={"file": PNG})
    assert up.status_code == 200, up.text

    row_id = report_thread.report_message_id(issue_id)
    async with async_session_maker() as db:
        atts = list((await db.execute(
            select(AdminThreadAttachment).where(AdminThreadAttachment.message_id == row_id)
        )).scalars().all())
    assert len(atts) == 1
    att = atts[0]
    assert att.mime_type == "image/png" and att.size_bytes == len(PNG[1]) and att.data == PNG[1]
    assert att.uploaded_by_user_id == uid and att.sha256

    # Same bytes again (a client retry) — still ONE picture on the row.
    up2 = await loop.client.post(f"{settings.api_prefix}/support/issues/{issue_id}/attachment",
                                 files={"file": PNG})
    assert up2.status_code == 200
    async with async_session_maker() as db:
        n = (await db.execute(
            select(func.count()).select_from(AdminThreadAttachment)
            .where(AdminThreadAttachment.message_id == row_id)
        )).scalar_one()
    assert n == 1

    # The operator sees it described on the message, and fetches the bytes
    # scoped by the user whose thread it is.
    admin = await _mk_user(role="admin")
    _as(loop, admin, role="admin")
    r = await loop.client.get(f"{settings.api_prefix}/admin/dispatch/threads/{uid}")
    msg = r.json()["messages"][0]
    assert msg["kind"] == "report"
    assert [a["id"] for a in msg["attachments"]] == [att.id]
    assert msg["attachments"][0]["mime_type"] == "image/png"
    got = await loop.client.get(
        f"{settings.api_prefix}/admin/dispatch/threads/{uid}/attachments/{att.id}")
    assert got.status_code == 200 and got.content == PNG[1]
    assert got.headers["content-type"].startswith("image/png")
    other = await _mk_user()
    wrong_scope = await loop.client.get(
        f"{settings.api_prefix}/admin/dispatch/threads/{other}/attachments/{att.id}")
    assert wrong_scope.status_code == 404, "the join is the authorisation, even for an admin"

    # The reporter fetches their own picture through their inbox route…
    _as(loop, uid)
    mine = await loop.client.get(f"{settings.api_prefix}/notices/thread/attachments/{att.id}")
    assert mine.status_code == 200 and mine.content == PNG[1]
    r = await loop.client.get(f"{settings.api_prefix}/notices/thread")
    assert r.json()["messages"][0]["attachments"][0]["id"] == att.id
    # …and a stranger cannot.
    _as(loop, other)
    assert (await loop.client.get(
        f"{settings.api_prefix}/notices/thread/attachments/{att.id}")).status_code == 404


@pytest.mark.asyncio
async def test_a_screenshot_for_an_issue_with_no_report_row_is_a_no_op(loop):
    """An issue that predates the feature, or was filed by a path that never
    opened a thread: the support copy is stored and the upload succeeds, and
    nothing is written to the thread tables."""
    uid = await _mk_user()
    async with async_session_maker() as db:
        issue = await repo.create_issue(db, raw_report="old-style", channel="mobile",
                                        reporter_user_id=uid)
        issue_id = issue.id
    _as(loop, uid)
    up = await loop.client.post(f"{settings.api_prefix}/support/issues/{issue_id}/attachment",
                                files={"file": PNG})
    assert up.status_code == 200, up.text
    async with async_session_maker() as db:
        supp = await repo.list_attachments(db, issue_id)
        stray = (await db.execute(
            select(func.count()).select_from(AdminThreadAttachment)
            .where(AdminThreadAttachment.message_id == report_thread.report_message_id(issue_id))
        )).scalar_one()
    assert len(supp) == 1, "the support copy is unaffected"
    assert stray == 0 and await _thread_rows(uid) == []


# ── 3. a second report ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_second_report_is_a_second_card_in_the_same_conversation(loop):
    uid = await _mk_user()
    first = await _file_report(loop, uid, note="first — low", severity="low")
    second = await _file_report(loop, uid, note="second — critical", severity="critical")
    assert first != second

    rows = await _thread_rows(uid)
    assert [r.kind for r in rows] == ["report", "report"]
    assert [r.body for r in rows] == ["first — low", "second — critical"]
    assert len({r.id for r in rows}) == 2

    admin = await _mk_user(role="admin")
    _as(loop, admin, role="admin")
    r = await loop.client.get(f"{settings.api_prefix}/admin/dispatch/threads")
    mine = [t for t in r.json()["threads"] if t["user_id"] == uid]
    assert len(mine) == 1, "D3: one conversation per user, never a second row"
    assert mine[0]["report_count"] == 2 and mine[0]["unread_in"] == 2
    assert mine[0]["report_severity"] == "critical" and mine[0]["report_open"] is True

    # Loudest OPEN, not latest: file critical then low, and the badge stays critical.
    uid2 = await _mk_user()
    await _file_report(loop, uid2, note="crit", severity="critical")
    await _file_report(loop, uid2, note="low", severity="low")
    _as(loop, admin, role="admin")
    r = await loop.client.get(f"{settings.api_prefix}/admin/dispatch/threads")
    row = next(t for t in r.json()["threads"] if t["user_id"] == uid2)
    assert row["report_severity"] == "critical" and row["report_open"] is True


# ── 4 + 5. the loop: answer → card + thread + push; user answers back ──

@pytest.mark.asyncio
async def test_answering_a_report_lands_a_persistent_card_in_their_chat_and_closes_the_report(loop):
    uid = await _mk_user()
    api_key = await _mk_agent(uid)
    issue_id = await _file_report(loop, uid, note="Voice mode is silent", severity="high")
    _as(loop, uid)
    assert (await loop.client.post(f"{settings.api_prefix}/support/issues/{issue_id}/attachment",
                                   files={"file": PNG})).status_code == 200

    admin = await _mk_user(role="admin")
    _as(loop, admin, role="admin")
    # The operator opens the conversation (the client marks it read on focus)…
    r = await loop.client.get(f"{settings.api_prefix}/admin/dispatch/threads/{uid}")
    assert r.json()["unread_in"] == 1 and r.json()["open_report"]["severity"] == "high"
    assert r.json()["messages"][0]["attachments"], "…with the screenshot on the report"
    assert (await loop.client.post(
        f"{settings.api_prefix}/admin/dispatch/threads/{uid}/read")).status_code == 204
    # …and answers.
    r = await loop.client.post(f"{settings.api_prefix}/admin/dispatch/threads/{uid}",
                               json={"body": "Thanks — which speaker are you on?"})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["in_chat"] is True and body["dispatch_id"]
    assert body["answered_report_id"] == report_thread.report_message_id(issue_id)
    dispatch_id = body["dispatch_id"]
    assert body["message"]["direction"] == "out" and body["message"]["dispatch_id"] == dispatch_id
    assert body["message"]["body"] == "Thanks — which speaker are you on?"
    assert body["message"]["sender_name"] == settings.admin_dispatch_sender_name
    assert loop.fanouts == [dispatch_id], "the answer was handed to the ordinary fan-out"

    # The dispatch row is the persistent, single-user shape the fan-out knows.
    async with async_session_maker() as db:
        d = await db.get(AdminDispatch, dispatch_id)
        assert d.mode == "persistent" and d.audience == "user" and d.target_user_id == uid
        assert d.title == report_thread.REPORT_REPLY_TITLE and d.body == "Thanks — which speaker are you on?"
        assert d.urgent is False and d.created_by_user_id == admin

    # Run the fan-out the route only spawned. Nothing is duplicated: the thread
    # row pre-written by the route IS the fan-out's uuid5 row.
    summary = await _run_fanout(dispatch_id)
    assert summary["chat_delivered_count"] == 1 and summary["failed_count"] == 0, summary
    assert summary["status"] == "sent"
    rows = await _thread_rows(uid)
    outs = [m for m in rows if m.direction == THREAD_OUT]
    assert len(outs) == 1 and outs[0].dispatch_id == dispatch_id
    assert outs[0].id == str(uuid.uuid5(uuid.NAMESPACE_URL, f"admin-thread:{dispatch_id}:{uid}"))

    # The card in their chat: the agent hop carried the report-answer title,
    # persistent mode (⇒ Reply action), the operator's words, the X-Agent-Key.
    hops = [c for c in loop.spy.calls if c["json"].get("dispatch_id") == dispatch_id]
    assert len(hops) == 1, loop.spy.calls
    hop = hops[0]
    assert hop["url"].endswith("/api/internal/admin-notice")
    assert hop["headers"]["X-Agent-Key"] == api_key
    assert hop["json"]["mode"] == "persistent"
    assert hop["json"]["title"] == report_thread.REPORT_REPLY_TITLE
    assert hop["json"]["body"] == "Thanks — which speaker are you on?"
    assert hop["json"]["user_id"] == uid

    # ONE push, on the announcement lane, deep-linked to the Admin thread —
    # the fan-out's own; the route did NOT also enqueue a reply notification.
    async with async_session_maker() as db:
        nq = list((await db.execute(
            select(NotificationQueue).where(NotificationQueue.user_id == uid)
        )).scalars().all())
        tgt = (await db.execute(
            select(AdminDispatchTarget).where(AdminDispatchTarget.dispatch_id == dispatch_id)
        )).scalar_one()
    assert len(nq) == 1, [n.idempotency_key for n in nq]
    assert nq[0].event_kind == "announcement"
    assert nq[0].idempotency_key == f"admin-dispatch:{dispatch_id}:{uid}"
    assert nq[0].data_json["deep_link"] == f"toup://notices?mission=admin:{dispatch_id}"
    assert nq[0].data_json["mode"] == "persistent"
    assert tgt.chat_status == "delivered" and tgt.state == "done"

    # The funnel: a dispatch_created that says it answers a report.
    created = [e for e in await _events(PE_DISPATCH_CREATED) if e.entity_id == dispatch_id]
    assert len(created) == 1
    assert created[0].payload_json["report_reply"] is True
    assert created[0].payload_json["report_severity"] == "high"

    # The report is now ANSWERED: badge muted, and the thread says so.
    _as(loop, admin, role="admin")
    r = await loop.client.get(f"{settings.api_prefix}/admin/dispatch/threads")
    row = next(t for t in r.json()["threads"] if t["user_id"] == uid)
    assert row["report_open"] is False and row["report_severity"] == "high"
    assert row["last_direction"] == "out"
    r = await loop.client.get(f"{settings.api_prefix}/admin/dispatch/threads/{uid}")
    assert r.json()["open_report"] is None

    # ── The user's side ──
    _as(loop, uid)
    r = await loop.client.get(f"{settings.api_prefix}/notices/thread")
    inbox = r.json()
    assert inbox["unread"] == 1
    kinds = [(m["direction"], m.get("kind")) for m in inbox["messages"]]
    assert kinds == [("in", "report"), ("out", None)]
    assert inbox["messages"][0]["severity"] == "high"
    assert len(inbox["messages"][0]["attachments"]) == 1
    assert inbox["messages"][1]["body"] == "Thanks — which speaker are you on?"
    st = await loop.client.get(f"{settings.api_prefix}/notices/state")
    assert st.json()["has_thread"] is True and st.json()["thread_unread"] == 1

    # The user answers (the Reply action lands here).
    r = await loop.client.post(f"{settings.api_prefix}/notices/thread",
                               json={"body": "AirPods Pro, connected fine"})
    assert r.status_code == 201, r.text

    # …and it shows in the admin thread as the next inbound row, unread.
    _as(loop, admin, role="admin")
    r = await loop.client.get(f"{settings.api_prefix}/admin/dispatch/threads/{uid}")
    msgs = r.json()["messages"]
    assert [(m["direction"], m.get("kind")) for m in msgs] == [
        ("in", "report"), ("out", None), ("in", None),
    ]
    assert msgs[-1]["body"] == "AirPods Pro, connected fine"
    assert r.json()["unread_in"] == 1, "the report was read above; the user's answer is the one unread row"
    _as(loop, admin, role="admin")
    r = await loop.client.get(f"{settings.api_prefix}/admin/dispatch/threads")
    row = next(t for t in r.json()["threads"] if t["user_id"] == uid)
    assert row["unread_in"] == 1 and row["last_direction"] == "in"
    assert row["last_body"] == "AirPods Pro, connected fine"


@pytest.mark.asyncio
async def test_a_follow_up_after_the_answer_is_thread_only_as_before(loop):
    """The card is for the ANSWER. Once the report is answered, replies inside
    the conversation are ordinary thread replies: no dispatch row, no agent
    hop, the reply notification the thread has always sent."""
    uid = await _mk_user()
    await _mk_agent(uid)
    await _file_report(loop, uid, severity="medium")
    admin = await _mk_user(role="admin")
    _as(loop, admin, role="admin")

    first = await loop.client.post(f"{settings.api_prefix}/admin/dispatch/threads/{uid}",
                                   json={"body": "Looking into it."})
    assert first.json()["in_chat"] is True
    d1 = first.json()["dispatch_id"]
    await _run_fanout(d1)

    second = await loop.client.post(f"{settings.api_prefix}/admin/dispatch/threads/{uid}",
                                    json={"body": "Found it — fix ships tomorrow."})
    assert second.status_code == 201
    assert second.json()["in_chat"] is False and second.json()["dispatch_id"] is None
    assert loop.fanouts == [d1], "no second dispatch"
    async with async_session_maker() as db:
        n_disp = (await db.execute(
            select(func.count()).select_from(AdminDispatch).where(AdminDispatch.target_user_id == uid)
        )).scalar_one()
        nq = list((await db.execute(
            select(NotificationQueue.idempotency_key).where(NotificationQueue.user_id == uid)
        )).scalars().all())
    assert n_disp == 1
    msg_id = second.json()["message"]["id"]
    assert f"admin-thread:{msg_id}" in nq, nq
    outs = [m for m in await _thread_rows(uid) if m.direction == THREAD_OUT]
    assert [m.body for m in outs] == ["Looking into it.", "Found it — fix ships tomorrow."]
    assert outs[1].dispatch_id is None

    # A NEW report re-opens the loop: the next answer cards again.
    await _file_report(loop, uid, note="another one", severity="low")
    _as(loop, admin, role="admin")
    third = await loop.client.post(f"{settings.api_prefix}/admin/dispatch/threads/{uid}",
                                   json={"body": "On it."})
    assert third.json()["in_chat"] is True and third.json()["dispatch_id"] != d1


@pytest.mark.asyncio
async def test_the_kill_switch_falls_back_to_a_thread_only_answer(loop, monkeypatch):
    monkeypatch.setattr(settings, "admin_dispatch_enabled", False, raising=False)
    uid = await _mk_user()
    await _file_report(loop, uid)
    admin = await _mk_user(role="admin")
    _as(loop, admin, role="admin")
    r = await loop.client.post(f"{settings.api_prefix}/admin/dispatch/threads/{uid}",
                               json={"body": "Seen."})
    assert r.status_code == 201, r.text
    assert r.json()["in_chat"] is False and r.json()["dispatch_id"] is None
    assert loop.fanouts == []
    outs = [m for m in await _thread_rows(uid) if m.direction == THREAD_OUT]
    assert len(outs) == 1 and outs[0].dispatch_id is None


@pytest.mark.asyncio
async def test_a_dead_fanout_still_leaves_the_answer_readable_in_the_thread(loop):
    """The route writes the thread row BEFORE handing off. If the fan-out
    never runs (worker died, replica redeployed), the answer is in the thread
    and the dispatch is honestly `queued` in Sent — never a card that was
    promised and a thread that has nothing."""
    uid = await _mk_user()
    await _file_report(loop, uid)
    admin = await _mk_user(role="admin")
    _as(loop, admin, role="admin")
    r = await loop.client.post(f"{settings.api_prefix}/admin/dispatch/threads/{uid}",
                               json={"body": "We see it."})
    dispatch_id = r.json()["dispatch_id"]
    # deliberately: no _run_fanout
    outs = [m for m in await _thread_rows(uid) if m.direction == THREAD_OUT]
    assert len(outs) == 1 and outs[0].body == "We see it." and outs[0].dispatch_id == dispatch_id
    async with async_session_maker() as db:
        d = await db.get(AdminDispatch, dispatch_id)
    assert d.status == "queued"
    _as(loop, uid)
    r = await loop.client.get(f"{settings.api_prefix}/notices/thread")
    assert [m["body"] for m in r.json()["messages"] if m["direction"] == "out"] == ["We see it."]


# ── 6. deletion, tombstones, and the open/answered predicate ──────

@pytest.mark.asyncio
async def test_a_report_deleted_for_everyone_keeps_kind_and_severity_and_loses_context_and_picture(loop):
    uid = await _mk_user()
    issue_id = await _file_report(loop, uid, severity="critical")
    _as(loop, uid)
    await loop.client.post(f"{settings.api_prefix}/support/issues/{issue_id}/attachment", files={"file": PNG})
    row_id = report_thread.report_message_id(issue_id)

    admin = await _mk_user(role="admin")
    _as(loop, admin, role="admin")
    r = await loop.client.delete(
        f"{settings.api_prefix}/admin/dispatch/threads/{uid}/messages/{row_id}?scope=everyone")
    assert r.status_code == 200, r.text
    m = r.json()["message"]
    assert m["kind"] == "report" and m["severity"] == "critical"
    assert m["report"] is None and m["attachments"] == []
    async with async_session_maker() as db:
        n = (await db.execute(
            select(func.count()).select_from(AdminThreadAttachment)
            .where(AdminThreadAttachment.message_id == row_id)
        )).scalar_one()
    assert n == 0, "093's purge ran"

    # A late upload onto the tombstone writes nothing.
    _as(loop, uid)
    assert (await loop.client.post(f"{settings.api_prefix}/support/issues/{issue_id}/attachment",
                                   files={"file": PNG})).status_code == 200
    async with async_session_maker() as db:
        n = (await db.execute(
            select(func.count()).select_from(AdminThreadAttachment)
            .where(AdminThreadAttachment.message_id == row_id)
        )).scalar_one()
    assert n == 0

    # A tombstoned report is not an open report.
    _as(loop, admin, role="admin")
    r = await loop.client.get(f"{settings.api_prefix}/admin/dispatch/threads")
    row = next(t for t in r.json()["threads"] if t["user_id"] == uid)
    assert row["report_severity"] is None and row["report_open"] is False and row["report_count"] == 0


@pytest.mark.asyncio
async def test_report_state_open_means_no_operator_turn_after_it():
    """The one predicate the badge and the delivery share, at the unit level:
    open ⇔ no `out` row after the report; a deleted operator turn still
    counts as a turn; a hidden-from-user report still counts as a report;
    among open reports the loudest wins and among equals the newest."""
    uid = await _mk_user()
    t0 = datetime.utcnow() - timedelta(minutes=10)
    async with async_session_maker() as db:
        def rep(i, sev, **kw):
            return AdminThreadMessage(
                id=str(uuid.uuid4()), user_id=uid, direction=THREAD_IN, body=f"r{i}",
                kind=THREAD_KIND_REPORT, severity=sev, report_json={},
                created_at=t0 + timedelta(minutes=i), **kw,
            )
        low = rep(1, "low")
        crit_hidden = rep(2, "critical", hidden_from_user_at=datetime.utcnow())
        db.add_all([low, crit_hidden])
        await db.commit()

        state = (await report_thread.report_state_for_users(db, [uid]))[uid]
        assert state.open_count == 2 and state.severity == "critical"
        assert state.open_report_id == crit_hidden.id, "hidden from the user is still the operator's queue"

        # The operator answers — even a turn later deleted for everyone.
        db.add(AdminThreadMessage(
            id=str(uuid.uuid4()), user_id=uid, direction=THREAD_OUT, body="This message was deleted.",
            created_at=t0 + timedelta(minutes=3), deleted_at=datetime.utcnow(),
        ))
        await db.commit()
        state = (await report_thread.report_state_for_users(db, [uid]))[uid]
        assert state.open_count == 0 and state.open_report_id is None
        assert state.severity == "critical" and state.latest_severity == "critical"
        assert state.report_count == 2

        # A new report after the answer re-opens; a tombstoned one does not count.
        again = rep(4, "medium")
        gone = rep(5, "critical", deleted_at=datetime.utcnow())
        db.add_all([again, gone])
        await db.commit()
        state = (await report_thread.report_state_for_users(db, [uid]))[uid]
        assert state.open_count == 1 and state.open_report_id == again.id
        assert state.severity == "medium" and state.report_count == 3

        # Ties: two open highs → the newest names the badge.
        h1, h2 = rep(6, "high"), rep(7, "high")
        db.add_all([h1, h2])
        await db.commit()
        state = (await report_thread.report_state_for_users(db, [uid]))[uid]
        assert state.severity == "high" and state.open_report_id == h2.id and state.open_count == 3
        assert state.open_report_ids == (again.id, h1.id, h2.id), "every open one, oldest first"

        # A BROADCAST's thread row is not an answer to anyone's report: the
        # persistent "maintenance tonight" to everyone must not close the
        # queue and send the operator's real answer thread-only.
        bcast = AdminDispatch(
            id=str(uuid.uuid4()), created_by_user_id=None, mode="persistent", audience="all",
            target_user_id=None, sender_name="Toup", title="Maintenance", body="Tonight 02:00 UTC.",
            status="sent", created_at=t0 + timedelta(minutes=8),
        )
        db.add(bcast)
        db.add(AdminThreadMessage(
            id=str(uuid.uuid4()), user_id=uid, direction=THREAD_OUT, body="Tonight 02:00 UTC.",
            dispatch_id=bcast.id, sender_name="Toup", created_at=t0 + timedelta(minutes=8),
        ))
        await db.commit()
        state = (await report_thread.report_state_for_users(db, [uid]))[uid]
        assert state.open_count == 3 and state.open_report_id == h2.id, "the broadcast answered nothing"

        # …but a dispatch to THIS user alone does (that is what an answer IS).
        direct = AdminDispatch(
            id=str(uuid.uuid4()), created_by_user_id=None, mode="persistent", audience="user",
            target_user_id=uid, sender_name="Toup", title=report_thread.REPORT_REPLY_TITLE,
            body="On it.", status="sent", created_at=t0 + timedelta(minutes=9),
        )
        db.add(direct)
        db.add(AdminThreadMessage(
            id=str(uuid.uuid4()), user_id=uid, direction=THREAD_OUT, body="On it.",
            dispatch_id=direct.id, sender_name="Toup", created_at=t0 + timedelta(minutes=9),
        ))
        await db.commit()
        state = (await report_thread.report_state_for_users(db, [uid]))[uid]
        assert state.open_count == 0 and state.severity == "high"

    # A user with no reports at all is simply absent.
    other = await _mk_user()
    async with async_session_maker() as db:
        assert other not in await report_thread.report_state_for_users(db, [other])
        assert (await report_thread.open_report_for_user(db, other)) == report_thread.EMPTY_REPORT_STATE


def test_context_parser_is_forgiving():
    p = report_thread.parse_report_context
    assert p("App: 1.2.0 (40)")["build"] == "40"
    assert p("App: 1.2.0")["app_version"] == "1.2.0" and p("App: 1.2.0")["build"] is None
    assert p("Device: Pixel 8 - Android 15")["os"] == "Android 15"
    assert p("junk line without colon\nScreen: Chat")["screen"] == "Chat"
    assert p("Platform: iOS")["platform"] == "ios"
    long = "Screen: " + "x" * 1000
    assert len(p(long)["screen"]) == 200 and len(p(long)["raw"]) <= 2000
    assert p("")["raw"] is None and p(None)["screen"] is None
