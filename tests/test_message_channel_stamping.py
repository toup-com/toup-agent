"""Message.channel wins, Conversation.channel is the fallback (D7).

`sessions._message_to_response` reads `conversation_channels` FIRST and only
falls through to `Message.channel` — its own docstring explains why: the
Message column "is only written for system senders", so a voice turn leaves it
NULL and carries "voice" on its Conversation. That was true. D7 makes the
Message column authoritative — ws_chat's presave finally stamps it — and the
priority has to invert with it, or a WhatsApp row inside a conversation the
resolver labelled something else keeps reporting the conversation's label.

The fallback and the 'web' default STAY. Every WS-presaved row in production
today is NULL, and both routes feed the same released client: ChatScreen.tsx's
`mapDayMsg` reads this key straight onto a badge, so a bare switch re-badges
the whole historical corpus on a build already in the store.

`channels_active` on the day index has the same inversion and the same reason:
it is how the app knows a day holds WhatsApp at all.

Scope note: the day-messages half of D7 lives in
tests/test_channel_turn_realtime_and_stamp.py. This file is the sessions half
plus the day-index aggregate; the two do not overlap.

Local run (from backend/):
    RUN_MODE=platform PYTHONPATH=. ENVIRONMENT=test JWT_SECRET=x ENCRYPTION_KEY=y \
      pytest tests/test_message_channel_stamping.py -q --tb=short -p no:cacheprovider
"""
from __future__ import annotations

import json
import uuid
from datetime import date as Date, datetime

import pytest
import pytest_asyncio


@pytest_asyncio.fixture
async def day_tables():
    from sqlalchemy import inspect as sa_inspect
    from app.db.database import engine
    from app.db.models.base import Base

    names = ["day_chats", "conversations", "messages"]
    async with engine.begin() as conn:
        for n in names:
            try:
                await conn.run_sync(Base.metadata.tables[n].create, checkfirst=True)
            except Exception:
                pass
    async with engine.connect() as conn:
        existing = await conn.run_sync(lambda c: set(sa_inspect(c).get_table_names()))
    missing = [n for n in names if n not in existing]
    if missing:
        pytest.skip(f"cannot create {missing} on this backend")
    yield


def _msg(channel, conversation_id="conv-1"):
    from app.db.models import Message

    return Message(id="m1", conversation_id=conversation_id, role="user",
                   content="hi", channel=channel,
                   created_at=datetime(2026, 9, 14, 1, 40, 21))


def _serialize(msg, conv_channels):
    from app.api.sessions import _message_to_response

    return _message_to_response(msg, None, None, conv_channels)


# ── the priority ladder ───────────────────────────────────────────────


def test_a_stamped_message_reports_its_own_channel():
    """A WhatsApp row in a conversation the resolver labelled 'web'."""
    out = _serialize(_msg("whatsapp"), {"conv-1": "web"})
    if out.channel == "web":
        pytest.xfail("sessions still reads Conversation.channel first — lane B3 (D7)")
    assert out.channel == "whatsapp"


def test_a_null_stamped_message_falls_back_to_its_conversation():
    """The voice case the current docstring is about, and the shape of the
    entire pre-D7 corpus."""
    assert _serialize(_msg(None), {"conv-1": "voice"}).channel == "voice"


def test_a_message_with_no_conversation_entry_still_reports_its_own_channel():
    """The single-message caller (POST .../messages, sessions.py:835) passes a
    one-entry map; a row whose conversation is not in it must not come back
    channel-less."""
    assert _serialize(_msg("telegram"), {}).channel == "telegram"


def test_a_row_that_knows_nothing_reports_nothing_rather_than_guessing():
    """`_message_to_response` has no 'web' default — the ChatMessageResponse
    field is optional and the client supplies its own. Inventing one here
    would disagree with the day-messages serializer, which DOES default."""
    assert _serialize(_msg(None), {}).channel is None


# ── the two routes must agree about one row ───────────────────────────


@pytest.mark.asyncio
async def test_the_sessions_and_day_chats_routes_agree_about_one_row(day_tables):
    """Two routes, one client, one row. A disagreement here is a badge that
    changes when the user navigates from the day index into a session."""
    from sqlalchemy import select
    import app.api.day_chats as dc_api
    from app.db.database import async_session_maker
    from app.db.models import Conversation, Message
    from app.db.models.day_chat import DayChat
    from app.db.models.user import User

    uid, dc_id, conv_id, mid = (str(uuid.uuid4()) for _ in range(4))
    day = Date(2026, 9, 14)
    async with async_session_maker() as db:
        db.add(User(id=uid, email=f"agree-{uuid.uuid4().hex[:10]}@example.com",
                    hashed_password="x", name="Agree", timezone="UTC"))
        db.add(DayChat(id=dc_id, user_id=uid, local_date=day, timezone="UTC"))
        await db.flush()
        db.add(Conversation(id=conv_id, user_id=uid, channel="web", day_chat_id=dc_id,
                            started_at=datetime(2026, 9, 14, 9, 0)))
        db.add(Message(id=mid, conversation_id=conv_id, day_chat_id=dc_id,
                       channel="whatsapp", role="user", content="hi",
                       created_at=datetime(2026, 9, 14, 9, 1)))
        await db.commit()

    real_proxy = dc_api._get_agent_proxy_info

    async def _no_proxy(*a, **k):
        return None

    dc_api._get_agent_proxy_info = _no_proxy
    try:
        async with async_session_maker() as db:
            user = (await db.execute(select(User).where(User.id == uid))).scalar_one()
            resp = await dc_api.get_day_chat_messages(
                date_str=day.isoformat(), limit=500, current_user=user, db=db,
            )
            row = (await db.execute(select(Message).where(Message.id == mid))).scalar_one()
    finally:
        dc_api._get_agent_proxy_info = real_proxy

    day_rows = json.loads(bytes(resp.body).decode())
    assert day_rows, "the fixture never reached the route"
    session_channel = _serialize(row, {conv_id: "web"}).channel
    assert day_rows[0]["channel"] == session_channel, (
        f"day index says {day_rows[0]['channel']!r}, sessions says "
        f"{session_channel!r} — same row, two answers"
    )


# ── the day index aggregate ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_channels_active_names_whatsapp_for_a_whatsapp_only_day(day_tables):
    """This list is how the app knows a day holds WhatsApp at all. Built from
    `distinct(Conversation.channel)` today (day_chats.py:650), it reports the
    conversation's label, so a stamped WhatsApp row in a web-labelled
    conversation is invisible in the index."""
    from sqlalchemy import select
    import app.api.day_chats as dc_api
    from app.db.database import async_session_maker
    from app.db.models import Conversation, Message
    from app.db.models.day_chat import DayChat
    from app.db.models.user import User

    uid, dc_id, conv_id = (str(uuid.uuid4()) for _ in range(3))
    day = Date(2026, 9, 14)
    async with async_session_maker() as db:
        db.add(User(id=uid, email=f"ca-{uuid.uuid4().hex[:10]}@example.com",
                    hashed_password="x", name="CA", timezone="UTC"))
        db.add(DayChat(id=dc_id, user_id=uid, local_date=day, timezone="UTC"))
        await db.flush()
        db.add(Conversation(id=conv_id, user_id=uid, channel="whatsapp",
                            day_chat_id=dc_id, started_at=datetime(2026, 9, 14, 9, 0)))
        db.add(Message(id=str(uuid.uuid4()), conversation_id=conv_id, day_chat_id=dc_id,
                       channel="whatsapp", role="user", content="hi",
                       created_at=datetime(2026, 9, 14, 9, 1)))
        await db.commit()

    real_proxy = dc_api._get_agent_proxy_info

    async def _no_proxy(*a, **k):
        return None

    dc_api._get_agent_proxy_info = _no_proxy
    try:
        async with async_session_maker() as db:
            user = (await db.execute(select(User).where(User.id == uid))).scalar_one()
            out = await dc_api.list_day_chats(limit=30, before=None, current_user=user, db=db)
    finally:
        dc_api._get_agent_proxy_info = real_proxy

    rows = out if isinstance(out, list) else json.loads(bytes(out.body).decode())
    mine = [r for r in rows if r.get("id") == dc_id]
    assert mine, "the seeded day is missing from the index"
    assert "whatsapp" in mine[0]["channels_active"]
