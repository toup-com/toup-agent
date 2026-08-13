"""Admin Dispatch — the AGENT half (contract §5, §10 item 7).

RUN_MODE=agent is load-bearing for this file: every assertion here touches
`messages` / `conversations` / `day_chats`, which are AGENT_ONLY. Under
RUN_MODE=platform they fail with "no such table", which reads as a defect
and is really a mis-invocation. See tests/COVERAGE_DEBT.txt.

    cd backend && RUN_MODE=agent USER_ID=test-user AGENT_API_KEY=test-agent-key \
      ENVIRONMENT=test PYTHONPATH=. \
      DATABASE_URL="sqlite+aiosqlite:///file::memory:?cache=shared&uri=true" \
      JWT_SECRET=test-jwt-secret-for-ci ENCRYPTION_KEY=test-32-byte-encryption-key--x12 \
      ./.venv-test/bin/python -m pytest tests/test_admin_notice_agent.py -q -p no:cacheprovider

What it locks, in order of how expensive each was to learn:

1. **The reload path.** The notice Message carries `day_chat_id = NULL`
   (D3 — that NULL is what keeps it out of the agent's context), but
   `GET /api/day-chats/{date}/messages`'s fast path selects
   `WHERE Message.day_chat_id = dc.id`. Without the OR arm that also
   accepts `(Conversation.channel == 'admin' AND
   Conversation.day_chat_id == dc.id)` the notice is written, broadcast,
   rendered live — and GONE on reload. That is the single most important
   test in this file, and its twin: the OR arm must not duplicate an
   ordinary same-day row.
2. **The agent must not see it.** `load_day_context` is the other side of
   the same NULL. If the notice ever appears there the agent starts
   answering for the operator.
3. **Idempotency by PK.** The Message id is a uuid5 of (dispatch, user),
   so a worker retry collides instead of giving the user a second copy.
4. **The frame carries NO `channel` key** (D7). The web client drops a
   frame stamped `channel:'admin'` silently, so the wrong shape here is
   invisible everywhere else.
"""

from __future__ import annotations

import json
import uuid
from datetime import date as Date, datetime, timedelta

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from app.config import settings


AGENT_KEY = "test-agent-key"
OWNER_ID = "test-user"


# ── fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def agent_identity(monkeypatch):
    """Pin the container's identity on `settings` instead of reading it from
    the environment.

    The endpoint's second and third gates compare against
    `settings.agent_api_key` and `settings.user_id`. Sourcing those from env
    would make the file pass or fail on the invocation's `USER_ID=` /
    `AGENT_API_KEY=` — and the agent-mode steps in
    `.github/workflows/test-backend.yml` set neither, so a missing env var
    would surface as a bogus 401 rather than as a defect. Setting them here
    keeps the assertions about the CODE.

    `run_mode` is NOT forced: RUN_MODE=agent stays a real precondition of
    this file, because `init_db()` builds `messages`/`conversations`/
    `day_chats` only under it. A platform-mode run still fails loudly with
    "no such table" rather than passing vacuously.
    """
    monkeypatch.setattr(settings, "agent_api_key", AGENT_KEY)
    monkeypatch.setattr(settings, "user_id", OWNER_ID)


@pytest_asyncio.fixture
async def owner_id(agent_identity) -> str:
    """The container's one user — it must BE `settings.user_id`, because the
    endpoint's third gate compares the envelope against exactly that."""
    from app.db import User, async_session_maker

    uid = settings.user_id
    assert uid, "settings.user_id must be set — the agent gate compares against it"
    async with async_session_maker() as db:
        existing = (
            await db.execute(select(User).where(User.id == uid))
        ).scalar_one_or_none()
        if existing is None:
            db.add(User(
                id=uid,
                email=f"admin-notice-{uuid.uuid4().hex[:8]}@example.com",
                hashed_password="x",
                name="Notice Owner",
            ))
            await db.commit()
    return uid


@pytest_asyncio.fixture
async def api_client(owner_id: str) -> AsyncClient:
    """Minimal app carrying exactly the two routers this file exercises:
    the agent-side write endpoint and the history reader the phone/web
    reload through. Built here rather than reusing conftest's
    `_build_test_app` because that one is the platform billing shape and
    mounts neither."""
    from fastapi import FastAPI
    from app.api.admin_notice import router as admin_notice_router
    from app.api.day_chats import router as day_chats_router

    app = FastAPI()
    app.include_router(admin_notice_router, prefix=settings.api_prefix)
    app.include_router(day_chats_router, prefix=settings.api_prefix)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://agent.test") as ac:
        yield ac


@pytest.fixture
def captured_frames(monkeypatch) -> list[dict]:
    """Capture every WS frame the writer emits.

    `broadcast_admin_notice` does `from app.api.ws_chat import
    broadcast_to_user` INSIDE the function, so patching the attribute on
    the ws_chat module is what the call actually resolves.
    """
    from app.api import ws_chat

    frames: list[dict] = []

    async def _fake_broadcast(user_id: str, event: dict) -> int:
        frames.append({"user_id": user_id, "event": event})
        return 2

    monkeypatch.setattr(ws_chat, "broadcast_to_user", _fake_broadcast)
    return frames


def _notice(dispatch_id: str, *, mode: str = "persistent", title: str = "Scheduled maintenance",
            sender: str = "Toup", sent_at: str = "2026-08-13T09:00:00") -> dict:
    from app.agent.admin_message_writer import build_admin_notice_payload

    return build_admin_notice_payload(
        dispatch_id=dispatch_id, mode=mode, title=title,
        sender_name=sender, sent_at=sent_at,
    )


async def _write(user_id: str, dispatch_id: str, *, content: str = "We are upgrading tonight.",
                 mode: str = "persistent") -> tuple[str, dict]:
    """Run the writer against its own session, like the endpoint does."""
    from app.agent.admin_message_writer import write_admin_notice
    from app.db import async_session_maker

    notice = _notice(dispatch_id, mode=mode)
    async with async_session_maker() as db:
        msg_id, day_chat_id = await write_admin_notice(
            db, user_id=user_id, content=content, notice=notice,
        )
    assert day_chat_id is None, (
        "the writer must always answer day_chat_id=None — the endpoint echoes "
        "it so the platform can see the NULL is intentional (D3)"
    )
    return msg_id, notice


async def _load_message(msg_id: str):
    from app.db import async_session_maker
    from app.db.models import Message

    async with async_session_maker() as db:
        return (
            await db.execute(select(Message).where(Message.id == msg_id))
        ).scalar_one_or_none()


async def _load_conversation(conv_id: str):
    from app.db import async_session_maker
    from app.db.models import Conversation

    async with async_session_maker() as db:
        return (
            await db.execute(select(Conversation).where(Conversation.id == conv_id))
        ).scalar_one_or_none()


async def _today_day_chat_id(user_id: str) -> str:
    from app.db import async_session_maker
    from app.db.message_helpers import resolve_day_chat_id_for_now

    async with async_session_maker() as db:
        dc_id = await resolve_day_chat_id_for_now(db, user_id)
        await db.commit()
    assert dc_id, "the day chat for today must resolve"
    return dc_id


async def _write_ordinary_message(user_id: str, day_chat_id: str, content: str) -> str:
    """An ordinary same-day web message — the control row for both the
    reload test (must come back exactly ONCE) and the context test (must
    be visible to the agent)."""
    from app.db import async_session_maker
    from app.db.models import Conversation, Message

    conv_id = str(uuid.uuid4())
    msg_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(Conversation(
            id=conv_id, user_id=user_id, day_chat_id=day_chat_id,
            channel="web", title="Web",
        ))
        db.add(Message(
            id=msg_id, conversation_id=conv_id, day_chat_id=day_chat_id,
            role="user", content=content, channel="web",
        ))
        await db.commit()
    return msg_id


# ── 1. the written row (contract §5 + §7) ─────────────────────────────


@pytest.mark.asyncio
async def test_writer_row_matches_contract(owner_id):
    """role/channel/source/day_chat_id and the persisted payload, plus the
    Conversation that carries the day the Message does not."""
    dispatch_id = str(uuid.uuid4())
    day_chat_id = await _today_day_chat_id(owner_id)

    msg_id, notice = await _write(owner_id, dispatch_id, mode="once")

    msg = await _load_message(msg_id)
    assert msg is not None, "the notice row must exist"
    assert msg.role == "assistant"
    assert msg.channel == "admin"
    assert msg.source == "admin_dispatch"
    assert msg.day_chat_id is None, (
        "D3: NULL day_chat_id is the whole mechanism that keeps the operator's "
        "message out of load_day_context. A non-NULL value here means the agent "
        "will start answering for the operator."
    )

    persisted = json.loads(msg.metadata_json)["admin_notice"]
    assert persisted == {
        "dispatch_id": dispatch_id,
        "mode": "once",
        "title": "Scheduled maintenance",
        "sender_name": "Toup",
        "sent_at": "2026-08-13T09:00:00",
    }, "AdminNoticePayload is §7 verbatim — both clients read these exact keys"
    assert "content" not in persisted and "body" not in persisted, (
        "the prose lives in Message.content and is deliberately not duplicated"
    )

    conv = await _load_conversation(msg.conversation_id)
    assert conv is not None
    assert conv.channel == "admin"
    assert conv.day_chat_id == day_chat_id, (
        "the CONVERSATION carries today's day chat — it is what the reload "
        "path's OR arm reads the day off"
    )


@pytest.mark.asyncio
async def test_message_id_is_the_uuid5_of_dispatch_and_user(owner_id):
    """The PK is the idempotency key AND the retract handle: the platform
    holds only the dispatch id and must be able to name the row."""
    from app.agent.admin_message_writer import admin_notice_message_id

    dispatch_id = str(uuid.uuid4())
    msg_id, _ = await _write(owner_id, dispatch_id)
    assert msg_id == str(
        uuid.uuid5(uuid.NAMESPACE_URL, f"admin-notice:{dispatch_id}:{owner_id}")
    )
    assert msg_id == admin_notice_message_id(dispatch_id, owner_id)


# ── 2. Reading A dedupe: one admin Conversation per user per day ──────


@pytest.mark.asyncio
async def test_two_dispatches_same_day_share_one_conversation(owner_id):
    await _today_day_chat_id(owner_id)
    id_a, _ = await _write(owner_id, str(uuid.uuid4()), content="first")
    id_b, _ = await _write(owner_id, str(uuid.uuid4()), content="second")

    a, b = await _load_message(id_a), await _load_message(id_b)
    assert a.id != b.id, "two distinct dispatches are two distinct rows"
    assert a.conversation_id == b.conversation_id, (
        "'admin' is in SYSTEM_CHANNELS so repeated dispatches on one day append "
        "to ONE Conversation (Reading A). A second conversation here means the "
        "sidebar grows a thread per announcement."
    )


@pytest.mark.asyncio
async def test_dispatches_on_different_days_get_different_conversations(
    owner_id, monkeypatch,
):
    """Reading A dedupes on (user, day_chat, channel), so a new day is a new
    admin Conversation. Yesterday is reached by freezing the resolver's
    `utc_now` — the writer imports the helper at call time, so patching the
    module attribute is what it actually resolves."""
    from app.db import message_helpers

    today_dc = await _today_day_chat_id(owner_id)
    id_today, _ = await _write(owner_id, str(uuid.uuid4()), content="today")

    real = message_helpers.resolve_day_chat_id_for_now

    async def _yesterday(db, user_id, tz_override=None, utc_now=None):
        return await real(
            db, user_id, tz_override=tz_override,
            utc_now=datetime.utcnow() - timedelta(days=1),
        )

    monkeypatch.setattr(message_helpers, "resolve_day_chat_id_for_now", _yesterday)
    id_yesterday, _ = await _write(owner_id, str(uuid.uuid4()), content="yesterday")

    a = await _load_message(id_today)
    b = await _load_message(id_yesterday)
    assert a.conversation_id != b.conversation_id, (
        "a dispatch on another day must not append to today's admin thread"
    )
    conv_yesterday = await _load_conversation(b.conversation_id)
    assert conv_yesterday.day_chat_id not in (None, today_dc)


# ── 3. idempotency (contract §5) ──────────────────────────────────────


@pytest.mark.asyncio
async def test_second_write_returns_same_id_and_leaves_one_row(owner_id):
    """The fan-out worker retries a target on any transport failure, and a
    broadcast re-run replays every target. A second copy of an announcement
    in the user's chat is the failure this uuid5 PK exists to prevent."""
    from app.db import async_session_maker
    from app.db.models import Message

    dispatch_id = str(uuid.uuid4())
    first, _ = await _write(owner_id, dispatch_id, content="only once")
    second, _ = await _write(owner_id, dispatch_id, content="only once")
    assert first == second

    async with async_session_maker() as db:
        rows = (
            await db.execute(select(Message).where(Message.id == first))
        ).scalars().all()
        all_admin = (
            await db.execute(
                select(Message).where(Message.source == "admin_dispatch")
            )
        ).scalars().all()
    assert len(rows) == 1
    assert len(all_admin) == 1, f"expected exactly one notice row, got {len(all_admin)}"


# ── 4. retract (contract §5, §10 item 7) ──────────────────────────────


@pytest.mark.asyncio
async def test_retract_deletes_exactly_that_row_and_is_idempotent(owner_id):
    from app.agent.admin_message_writer import delete_admin_notice
    from app.db import async_session_maker
    from app.db.models import Message

    keep_dispatch = str(uuid.uuid4())
    drop_dispatch = str(uuid.uuid4())
    keep_id, _ = await _write(owner_id, keep_dispatch, content="stays")
    drop_id, _ = await _write(owner_id, drop_dispatch, content="goes", mode="once")

    async with async_session_maker() as db:
        deleted = await delete_admin_notice(
            db, user_id=owner_id, dispatch_id=drop_dispatch,
        )
    assert deleted == 1

    assert await _load_message(drop_id) is None
    assert await _load_message(keep_id) is not None, (
        "the retract must name exactly one row — the PK encodes (dispatch, user)"
    )

    async with async_session_maker() as db:
        again = await delete_admin_notice(
            db, user_id=owner_id, dispatch_id=drop_dispatch,
        )
    assert again == 0, (
        "zero rows is SUCCESS — the platform replays a read receipt whose "
        "retract already landed"
    )

    async with async_session_maker() as db:
        remaining = (
            await db.execute(
                select(Message).where(Message.source == "admin_dispatch")
            )
        ).scalars().all()
    assert [m.id for m in remaining] == [keep_id]


# ── 5. the endpoint's three gates + response shape (contract §5) ──────


def _body(user_id: str, dispatch_id: str, **over) -> dict:
    payload = {
        "user_id": user_id,
        "dispatch_id": dispatch_id,
        "mode": "persistent",
        "title": "Scheduled maintenance",
        "body": "We are upgrading tonight.",
        "sender_name": "Toup",
        "sent_at": "2026-08-13T09:00:00",
    }
    payload.update(over)
    return payload


@pytest.mark.asyncio
async def test_endpoint_404_when_not_agent_mode(api_client, owner_id, monkeypatch):
    """The routes are mounted only by agent_main, but the gate means a
    platform deploy can never answer for them either."""
    monkeypatch.setattr(settings, "run_mode", "platform")
    r = await api_client.post(
        f"{settings.api_prefix}/internal/admin-notice",
        json=_body(owner_id, str(uuid.uuid4())),
        headers={"X-Agent-Key": AGENT_KEY},
    )
    assert r.status_code == 404, r.text


@pytest.mark.asyncio
async def test_endpoint_rejects_missing_and_wrong_agent_key(api_client, owner_id):
    """AgentAPIKeyMiddleware FALLS OPEN when agent_api_key is unset, so this
    route runs its own compare. Both misses must be 401/403, never a write."""
    from app.db import async_session_maker
    from app.db.models import Message

    url = f"{settings.api_prefix}/internal/admin-notice"

    r_missing = await api_client.post(url, json=_body(owner_id, str(uuid.uuid4())))
    assert r_missing.status_code in (401, 403), r_missing.text

    r_wrong = await api_client.post(
        url,
        json=_body(owner_id, str(uuid.uuid4())),
        headers={"X-Agent-Key": "not-the-key"},
    )
    assert r_wrong.status_code in (401, 403), r_wrong.text

    async with async_session_maker() as db:
        rows = (
            await db.execute(
                select(Message).where(Message.source == "admin_dispatch")
            )
        ).scalars().all()
    assert rows == [], "a rejected call must not have written anything"


@pytest.mark.asyncio
async def test_endpoint_403_when_envelope_user_is_not_the_container_owner(
    api_client, owner_id,
):
    """`get_current_user`'s agent branch auto-creates a stub User for a
    mis-routed envelope, so a platform routing bug would otherwise land
    silently in the wrong tenant."""
    from app.db import async_session_maker
    from app.db.models import Message

    r = await api_client.post(
        f"{settings.api_prefix}/internal/admin-notice",
        json=_body(str(uuid.uuid4()), str(uuid.uuid4())),
        headers={"X-Agent-Key": AGENT_KEY},
    )
    assert r.status_code == 403, r.text

    async with async_session_maker() as db:
        rows = (
            await db.execute(
                select(Message).where(Message.source == "admin_dispatch")
            )
        ).scalars().all()
    assert rows == []


@pytest.mark.asyncio
async def test_endpoint_201_response_shape(api_client, owner_id, captured_frames):
    from app.agent.admin_message_writer import admin_notice_message_id

    dispatch_id = str(uuid.uuid4())
    r = await api_client.post(
        f"{settings.api_prefix}/internal/admin-notice",
        json=_body(owner_id, dispatch_id),
        headers={"X-Agent-Key": AGENT_KEY},
    )
    assert r.status_code == 201, r.text
    data = r.json()
    assert set(data) == {"message_id", "day_chat_id", "ws_count"}, data
    assert data["message_id"] == admin_notice_message_id(dispatch_id, owner_id)
    assert data["day_chat_id"] is None
    assert data["ws_count"] == 2, "the endpoint reports the live connection count"


@pytest.mark.asyncio
async def test_endpoint_retract_response_shape(api_client, owner_id, captured_frames):
    dispatch_id = str(uuid.uuid4())
    await api_client.post(
        f"{settings.api_prefix}/internal/admin-notice",
        json=_body(owner_id, dispatch_id, mode="once"),
        headers={"X-Agent-Key": AGENT_KEY},
    )
    r = await api_client.post(
        f"{settings.api_prefix}/internal/admin-notice/retract",
        json={"user_id": owner_id, "dispatch_id": dispatch_id},
        headers={"X-Agent-Key": AGENT_KEY},
    )
    assert r.status_code == 200, r.text
    assert r.json() == {"deleted": 1, "ws_count": 2}

    r_again = await api_client.post(
        f"{settings.api_prefix}/internal/admin-notice/retract",
        json={"user_id": owner_id, "dispatch_id": dispatch_id},
        headers={"X-Agent-Key": AGENT_KEY},
    )
    assert r_again.status_code == 200
    assert r_again.json()["deleted"] == 0


@pytest.mark.asyncio
async def test_retract_endpoint_runs_the_same_gates(api_client, owner_id):
    r = await api_client.post(
        f"{settings.api_prefix}/internal/admin-notice/retract",
        json={"user_id": owner_id, "dispatch_id": str(uuid.uuid4())},
        headers={"X-Agent-Key": "not-the-key"},
    )
    assert r.status_code in (401, 403), r.text


# ── 6. the WS frames (contract D7 + §7) ───────────────────────────────


@pytest.mark.asyncio
async def test_notice_frame_has_no_channel_key_and_matches_contract(
    api_client, owner_id, captured_frames,
):
    """D7: mobile accepts `!channel || channel==='app' || type==='message'`,
    the web `!channel || channel==='web'`. A frame stamped channel:'admin'
    is SILENTLY dropped by the browser — absent is the one value both
    filters pass, and nothing else in the stack can catch the wrong one."""
    from app.agent.admin_message_writer import admin_notice_message_id

    dispatch_id = str(uuid.uuid4())
    r = await api_client.post(
        f"{settings.api_prefix}/internal/admin-notice",
        json=_body(owner_id, dispatch_id),
        headers={"X-Agent-Key": AGENT_KEY},
    )
    assert r.status_code == 201, r.text

    assert len(captured_frames) == 1, captured_frames
    sent = captured_frames[0]
    assert sent["user_id"] == owner_id
    event = sent["event"]

    assert "channel" not in event, (
        f"D7 violated — the frame carries a channel key: {event!r}. The web "
        "client drops it and the card never appears in the browser."
    )
    assert set(event) == {"type", "id", "created_at", "content", "notice"}, event
    assert event["type"] == "admin_notice"
    assert event["id"] == admin_notice_message_id(dispatch_id, owner_id)
    assert event["content"] == "We are upgrading tonight."
    assert isinstance(event["created_at"], str) and event["created_at"]
    assert event["notice"] == {
        "dispatch_id": dispatch_id,
        "mode": "persistent",
        "title": "Scheduled maintenance",
        "sender_name": "Toup",
        "sent_at": "2026-08-13T09:00:00",
    }


@pytest.mark.asyncio
async def test_retract_frame_has_no_channel_key_and_matches_contract(
    api_client, owner_id, captured_frames,
):
    dispatch_id = str(uuid.uuid4())
    await api_client.post(
        f"{settings.api_prefix}/internal/admin-notice",
        json=_body(owner_id, dispatch_id, mode="once"),
        headers={"X-Agent-Key": AGENT_KEY},
    )
    captured_frames.clear()

    await api_client.post(
        f"{settings.api_prefix}/internal/admin-notice/retract",
        json={"user_id": owner_id, "dispatch_id": dispatch_id},
        headers={"X-Agent-Key": AGENT_KEY},
    )
    assert len(captured_frames) == 1, captured_frames
    event = captured_frames[0]["event"]
    assert "channel" not in event, f"D7 violated on the retract frame: {event!r}"
    assert event == {"type": "admin_notice_retract", "dispatch_id": dispatch_id}


@pytest.mark.asyncio
async def test_frame_carries_the_id_that_was_persisted(
    api_client, owner_id, captured_frames,
):
    """A client that takes the live frame and a client that hydrates from
    history must be looking at the same row, or a retract lands on nothing."""
    dispatch_id = str(uuid.uuid4())
    r = await api_client.post(
        f"{settings.api_prefix}/internal/admin-notice",
        json=_body(owner_id, dispatch_id),
        headers={"X-Agent-Key": AGENT_KEY},
    )
    msg = await _load_message(r.json()["message_id"])
    assert msg is not None
    assert captured_frames[0]["event"]["id"] == msg.id
    assert captured_frames[0]["event"]["notice"] == json.loads(
        msg.metadata_json
    )["admin_notice"]


# ── 7. THE RELOAD PATH — the regression this file exists for ──────────


@pytest.mark.asyncio
async def test_notice_survives_reload_through_day_chats_messages(
    api_client, owner_id, captured_frames,
):
    """`GET /api/day-chats/{today}/messages` is what every client fetches.
    Its fast path selects `WHERE Message.day_chat_id = dc.id`, and the
    notice's is deliberately NULL (D3) — so without the OR arm that reads
    the day off the admin CONVERSATION, the notice is written, broadcast,
    rendered live, and then GONE on reload."""
    day_chat_id = await _today_day_chat_id(owner_id)
    ordinary_id = await _write_ordinary_message(owner_id, day_chat_id, "hello agent")

    dispatch_id = str(uuid.uuid4())
    r = await api_client.post(
        f"{settings.api_prefix}/internal/admin-notice",
        json=_body(owner_id, dispatch_id),
        headers={"X-Agent-Key": AGENT_KEY},
    )
    assert r.status_code == 201, r.text
    notice_msg_id = r.json()["message_id"]

    today = Date.today().isoformat()
    resp = await api_client.get(
        f"{settings.api_prefix}/day-chats/{today}/messages",
        headers={"X-Agent-Key": AGENT_KEY},
    )
    assert resp.status_code == 200, resp.text
    rows = resp.json()
    by_id = {row["id"]: row for row in rows}

    assert notice_msg_id in by_id, (
        "the notice vanished on reload — the day-chats fast path must accept "
        "(Conversation.channel == 'admin' AND Conversation.day_chat_id == dc.id) "
        "as a second arm, because Message.day_chat_id is NULL by design (D3)"
    )
    row = by_id[notice_msg_id]
    assert row["admin_notice"] == {
        "dispatch_id": dispatch_id,
        "mode": "persistent",
        "title": "Scheduled maintenance",
        "sender_name": "Toup",
        "sent_at": "2026-08-13T09:00:00",
    }, "the serializer must emit the §7 payload, or the card renders as a bare agent bubble"
    assert row["role"] == "assistant"
    assert row["channel"] == "admin"
    assert row["content"] == "We are upgrading tonight."

    ids = [row["id"] for row in rows]
    assert ids.count(ordinary_id) == 1, (
        f"the new OR arm duplicated an ordinary same-day row: {ids}"
    )
    assert ids.count(notice_msg_id) == 1, f"the notice came back twice: {ids}"
    assert by_id[ordinary_id]["admin_notice"] is None


@pytest.mark.asyncio
async def test_or_arm_is_scoped_to_the_admin_channel(owner_id, api_client):
    """The new OR arm must stay `channel == 'admin' AND day_chat_id == dc.id`,
    not a bare `Conversation.day_chat_id == dc.id`.

    The duplicate-row assertion in the test above cannot catch a widened arm
    — Message→Conversation is a 1:1 join, so an ordinary row satisfying both
    arms still returns once. This is what catches it: a NULL-day_chat_id row
    on a NON-admin conversation of the same day (degraded resolution leaves
    these behind) is invisible to the fast path today, and a widened arm
    would start serving it.
    """
    from app.db import async_session_maker
    from app.db.models import Conversation, Message

    day_chat_id = await _today_day_chat_id(owner_id)
    stray_conv = str(uuid.uuid4())
    stray_msg = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(Conversation(
            id=stray_conv, user_id=owner_id, day_chat_id=day_chat_id,
            channel="web", title="Web",
        ))
        db.add(Message(
            id=stray_msg, conversation_id=stray_conv, day_chat_id=None,
            role="user", content="stray row with no day pointer", channel="web",
        ))
        await db.commit()

    resp = await api_client.get(
        f"{settings.api_prefix}/day-chats/{Date.today().isoformat()}/messages",
        headers={"X-Agent-Key": AGENT_KEY},
    )
    assert resp.status_code == 200, resp.text
    ids = [row["id"] for row in resp.json()]
    assert stray_msg not in ids, (
        "the admin OR arm has been widened past channel=='admin' — it is now "
        "serving unrelated NULL-day_chat_id rows off the conversation's day"
    )


@pytest.mark.asyncio
async def test_retracted_notice_is_gone_from_the_reload_path(
    api_client, owner_id, captured_frames,
):
    """D4: `once` is a real lifecycle. After the retract a reload must never
    re-serve the card."""
    day_chat_id = await _today_day_chat_id(owner_id)
    ordinary_id = await _write_ordinary_message(owner_id, day_chat_id, "still here")

    dispatch_id = str(uuid.uuid4())
    r = await api_client.post(
        f"{settings.api_prefix}/internal/admin-notice",
        json=_body(owner_id, dispatch_id, mode="once"),
        headers={"X-Agent-Key": AGENT_KEY},
    )
    notice_msg_id = r.json()["message_id"]
    await api_client.post(
        f"{settings.api_prefix}/internal/admin-notice/retract",
        json={"user_id": owner_id, "dispatch_id": dispatch_id},
        headers={"X-Agent-Key": AGENT_KEY},
    )

    resp = await api_client.get(
        f"{settings.api_prefix}/day-chats/{Date.today().isoformat()}/messages",
        headers={"X-Agent-Key": AGENT_KEY},
    )
    assert resp.status_code == 200, resp.text
    ids = [row["id"] for row in resp.json()]
    assert notice_msg_id not in ids
    assert ordinary_id in ids, "the retract must not take the day's other rows with it"


# ── 8. the agent must NOT see it (D3) ─────────────────────────────────


@pytest.mark.asyncio
async def test_agent_context_cannot_see_the_notice(owner_id, monkeypatch):
    """`load_day_context` selects `WHERE Message.day_chat_id = :id`. The
    notice's NULL is what makes it structurally invisible there — the
    operator is talking to the user, and an agent that reads the message
    starts answering for the operator."""
    from app.agent.day_context_loader import load_day_context
    from app.db import async_session_maker

    day_chat_id = await _today_day_chat_id(owner_id)
    await _write_ordinary_message(owner_id, day_chat_id, "ORDINARY-SENTINEL")
    await _write(
        owner_id, str(uuid.uuid4()),
        content="NOTICE-SENTINEL operator announcement body",
    )

    async with async_session_maker() as db:
        ctx = await load_day_context(db, day_chat_id)

    blob = json.dumps(ctx["messages"]) + json.dumps(ctx["raw_messages"])
    assert "ORDINARY-SENTINEL" in blob, (
        "sanity: the control message must be in the agent's context, or this "
        "test would pass for the wrong reason"
    )
    assert "NOTICE-SENTINEL" not in blob, (
        "the agent can see an operator notice — D3 is broken. Check that "
        "Message.day_chat_id is still NULL on the notice row and that "
        "load_day_context has not grown a Conversation-based arm."
    )
    assert not any(
        (m.get("channel") == "admin") or (m.get("source") == "admin_dispatch")
        for m in ctx["raw_messages"]
    ), ctx["raw_messages"]
