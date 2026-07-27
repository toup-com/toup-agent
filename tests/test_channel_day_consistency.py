"""Multichannel day-consistency (audit W1.7).

Locks four defects:
    (a) VOICE LOCAL-TZ ROLLOVER — the realtime relay reuses a voice
        session by the user's LOCAL calendar day (not UTC date), and
        POST /sessions/{id}/messages stamps day_chat_id from the
        CURRENT time (not the session's stale creation-day pointer)
        plus Message.channel.
    (b) ws_browser Message/Conversation rows carry day_chat_id.
    (c) /api/chat sessions route through the canonical day resolver
        (channel 'api' is in the system-channel unique index).
    (d) Autopilot ticks (ephemeral runs) create NO Conversation row.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import select


LA = "America/Los_Angeles"


async def _mk_user(tz: str | None = None) -> str:
    from app.db import User, async_session_maker

    user_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(
            id=user_id,
            email=f"dayc-{uuid.uuid4().hex[:10]}@example.com",
            hashed_password="x",
            name="Day Consistency",
            timezone=tz,
        ))
        await db.commit()
    return user_id


async def _mk_day_chat(user_id: str, local_date: date, tz: str) -> str:
    from app.db import async_session_maker
    from app.db.models import DayChat

    dc_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(DayChat(
            id=dc_id, user_id=user_id, local_date=local_date, timezone=tz,
        ))
        await db.commit()
    return dc_id


# ──────────────────────────────────────────────────────────────────────
# (a1) Voice session reuse rolls at LOCAL midnight, not UTC
# ──────────────────────────────────────────────────────────────────────

def _freeze_ws_realtime_now(monkeypatch, frozen_utc: datetime):
    """Patch ws_realtime's module-level `datetime` so now() is frozen.
    Subclass keeps fromisoformat/strftime working for the helpers."""
    from app.api import ws_realtime

    class _Frozen(datetime):
        @classmethod
        def now(cls, tz=None):  # noqa: D102
            return frozen_utc.astimezone(tz) if tz else frozen_utc.replace(tzinfo=None)

    monkeypatch.setattr(ws_realtime, "datetime", _Frozen)


@pytest.mark.asyncio
async def test_voice_session_reused_across_utc_midnight_same_local_day(monkeypatch):
    """LA evening: 18:00 local Jul 26 = 01:00 UTC Jul 27. The session
    started 16:00 local (23:00 UTC Jul 26) — SAME local day, different
    UTC dates. Old UTC-date check churned a new session here."""
    from app.api import ws_realtime

    user_id = await _mk_user(tz=LA)
    sid = str(uuid.uuid4())

    _freeze_ws_realtime_now(
        monkeypatch, datetime(2026, 7, 27, 1, 0, 0, tzinfo=timezone.utc),
    )

    async def fake_get_vps_info(_uid):
        return ("http://agent.test", "key")

    calls: list[tuple] = []

    async def fake_vps_api(url, key, method, path, params=None, json_body=None):
        calls.append((method, path))
        if method == "GET":
            return {"id": sid, "started_at": "2026-07-26T23:00:00"}
        return {"id": "SHOULD-NOT-CREATE"}

    monkeypatch.setattr(ws_realtime, "_get_vps_info", fake_get_vps_info)
    monkeypatch.setattr(ws_realtime, "_vps_api", fake_vps_api)

    result = await ws_realtime._get_or_create_voice_session(user_id, sid)
    assert result == sid, "same local day must REUSE the session"
    assert ("POST", "/api/sessions") not in calls


@pytest.mark.asyncio
async def test_voice_session_rolls_at_local_midnight_before_utc(monkeypatch):
    """Tokyo, 01:00 local Jul 27 = 16:00 UTC Jul 26. The session started
    22:00 local Jul 26 (13:00 UTC) — same UTC date, but the LOCAL day
    has rolled. Old check reused it, stranding post-midnight transcripts
    in yesterday's session until UTC midnight."""
    from app.api import ws_realtime

    user_id = await _mk_user(tz="Asia/Tokyo")
    old_sid = str(uuid.uuid4())
    new_sid = str(uuid.uuid4())

    _freeze_ws_realtime_now(
        monkeypatch, datetime(2026, 7, 26, 16, 0, 0, tzinfo=timezone.utc),
    )

    async def fake_get_vps_info(_uid):
        return ("http://agent.test", "key")

    async def fake_vps_api(url, key, method, path, params=None, json_body=None):
        if method == "GET":
            return {"id": old_sid, "started_at": "2026-07-26T13:00:00"}
        return {"id": new_sid}

    monkeypatch.setattr(ws_realtime, "_get_vps_info", fake_get_vps_info)
    monkeypatch.setattr(ws_realtime, "_vps_api", fake_vps_api)

    result = await ws_realtime._get_or_create_voice_session(user_id, old_sid)
    assert result == new_sid, "new LOCAL day must mint a new session"


# ──────────────────────────────────────────────────────────────────────
# (a2) POST /sessions/{id}/messages stamps day_chat per-NOW + channel
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_session_message_lands_in_todays_local_day_chat():
    """A voice session created yesterday (stale day_chat_id) held open
    past local midnight: the saved message must land in TODAY's local
    day chat, not the session's creation day, and carry channel."""
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from app.api.auth import get_current_user
    from app.api.sessions import router as sessions_router
    from app.db import User, async_session_maker
    from app.db.models import Conversation, DayChat, Message

    user_id = await _mk_user(tz=LA)
    today_la = datetime.now(ZoneInfo(LA)).date()
    stale_dc = await _mk_day_chat(user_id, today_la - timedelta(days=1), LA)

    conv_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(Conversation(
            id=conv_id, user_id=user_id, channel="voice",
            is_active=True, day_chat_id=stale_dc,
        ))
        await db.commit()

    async def override_current_user():
        async with async_session_maker() as db:
            return (await db.execute(
                select(User).where(User.id == user_id)
            )).scalar_one()

    app = FastAPI()
    app.include_router(sessions_router, prefix="/api")
    app.dependency_overrides[get_current_user] = override_current_user

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://agent") as ac:
        resp = await ac.post(
            f"/api/sessions/{conv_id}/messages",
            params={"role": "user", "content": "post-midnight voice turn"},
        )
        assert resp.status_code == 201, resp.text

    async with async_session_maker() as db:
        msg = (await db.execute(
            select(Message).where(Message.conversation_id == conv_id)
        )).scalar_one()
        today_dc = (await db.execute(
            select(DayChat).where(
                DayChat.user_id == user_id,
                DayChat.local_date == today_la,
            )
        )).scalar_one_or_none()

    assert msg.day_chat_id is not None
    assert msg.day_chat_id != stale_dc, (
        "message must NOT inherit the session's stale creation-day stamp"
    )
    assert today_dc is not None and msg.day_chat_id == today_dc.id
    assert msg.channel == "voice"


# ──────────────────────────────────────────────────────────────────────
# (b) ws_browser rows carry day_chat_id
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_browser_agent_reply_is_day_stamped(monkeypatch):
    from app.api import ws_browser
    from app.db import async_session_maker
    from app.db.models import Conversation, Message

    user_id = await _mk_user(tz=LA)
    conv_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(Conversation(
            id=conv_id, user_id=user_id, title="Browser",
            channel="web", is_active=True,
        ))
        await db.commit()

    class _FakeWS:
        async def send_json(self, data, **kwargs):
            return None

    async def fake_inner(websocket, page, tab_manager, browser_mod,
                         user_message, conversation, overlay,
                         model_override=None):
        await websocket.send_json(
            {"type": "agent_message", "content": "browsed it"}
        )

    monkeypatch.setattr(ws_browser, "_run_browser_agent_inner", fake_inner)

    await ws_browser._run_browser_agent(
        _FakeWS(), None, None, None, "find a laptop", [], None,
        session_id=conv_id, user_id_for_save=user_id,
    )

    async with async_session_maker() as db:
        msg = (await db.execute(
            select(Message).where(Message.conversation_id == conv_id)
        )).scalar_one()

    assert msg.role == "assistant"
    assert msg.day_chat_id is not None, (
        "browser-overlay reply invisible to day context (NULL day_chat_id)"
    )


# ──────────────────────────────────────────────────────────────────────
# (c) /api/chat conversations are day-stamped + deduped via resolver
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_api_chat_session_day_stamped_and_deduped():
    from app.api.chat import _get_or_create_session
    from app.db import async_session_maker
    from app.db.models import Conversation

    user_id = await _mk_user(tz=LA)

    async with async_session_maker() as db:
        s1, _ = await _get_or_create_session(db=db, user_id=user_id)
        await db.commit()
        s1_id, s1_day = s1.id, s1.day_chat_id

    assert s1_day is not None, "api conversation must carry day_chat_id"

    # Second call the same day reuses the day's api thread (Reading-A
    # invariant — 'api' is in the partial unique index, so a blind
    # stamped insert would IntegrityError here instead).
    async with async_session_maker() as db:
        s2, _ = await _get_or_create_session(db=db, user_id=user_id)
        await db.commit()
        assert s2.id == s1_id

    async with async_session_maker() as db:
        convs = (await db.execute(
            select(Conversation).where(
                Conversation.user_id == user_id,
                Conversation.channel == "api",
            )
        )).scalars().all()
    assert len(convs) == 1


# ──────────────────────────────────────────────────────────────────────
# (d) Autopilot ticks create no Conversation row
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ephemeral_session_persists_no_conversation_row():
    """AUTOPILOT-profile ticks (both save flags off, no session id) get
    an in-memory sentinel — one empty row per ~5min tick otherwise.
    `_get_or_create_session` never touches self, so call it unbound."""
    from app.agent.agent_runner import AgentRunner
    from app.db import async_session_maker
    from app.db.models import Conversation

    user_id = await _mk_user()

    async with async_session_maker() as db:
        session, is_new = await AgentRunner._get_or_create_session(
            None, db, user_id, None, None,
            channel="autopilot", ephemeral=True,
        )
        await db.commit()

    assert session.id and session.user_id == user_id
    assert session.channel == "autopilot"

    async with async_session_maker() as db:
        rows = (await db.execute(
            select(Conversation).where(Conversation.user_id == user_id)
        )).scalars().all()
    assert rows == [], "ephemeral run must not persist a Conversation"


@pytest.mark.asyncio
async def test_non_ephemeral_session_still_persists():
    """Control: default path unchanged — a normal run creates the row."""
    from app.agent.agent_runner import AgentRunner
    from app.db import async_session_maker
    from app.db.models import Conversation

    user_id = await _mk_user(tz=LA)

    async with async_session_maker() as db:
        session, is_new = await AgentRunner._get_or_create_session(
            None, db, user_id, None, None, channel="web",
        )
        await db.commit()
        sid = session.id

    async with async_session_maker() as db:
        row = (await db.execute(
            select(Conversation).where(Conversation.id == sid)
        )).scalar_one_or_none()
    assert row is not None and row.channel == "web"
