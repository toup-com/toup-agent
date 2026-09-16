"""The REST half of C1: the identity the app pairs on must reach the client.

Round 46, incident 2 (2026-09-15). The app's whole fix for the vanishing
optimistic bubble is `threadOrder.ts` rung 1 — pair by `clientMsgId` — and it
reads that field off the REST rows, because the WS frame is gone by the time a
resync runs. `Message.client_msg_id` / `occurred_at` are therefore only half a
fix until the two message serializers actually emit them.

Nothing covered that. `grep -rn client_msg_id backend/tests` matched no test of
day_chats or sessions; the one file that names the endpoint,
`test_day_chats_api.py`, is excused from the platform sweep in COVERAGE_DEBT
(240 s budget), is not an agent-mode entry either, and is red in both worktrees
for an unrelated fixture reason. So a regression that dropped the key from a
row, or that left the ORDER BY naming a column an unmigrated tenant does not
have (a guaranteed 500 on every day-chat load), failed nothing.

Lane: RUN_MODE=agent — `day_chats`/`conversations`/`messages` are AGENT_ONLY
and both routes SELECT them.

Local run (from backend/):
    RUN_MODE=agent PYTHONPATH=. pytest tests/test_rest_turn_identity_rows.py \
      -q -p no:cacheprovider
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest

pytestmark = pytest.mark.asyncio


async def _seed_day(*, stamped: bool):
    """A user with one day chat holding a user row and an assistant row.

    `stamped` decides whether the pair carries the round-46 identity at all —
    every pre-existing row and every row a container on image 1cd801aacb11
    writes has neither, for a full rollout cycle, in the same day thread.
    """
    from app.db.database import async_session_maker
    from app.db.models import Conversation, Message
    from app.db.models.day_chat import DayChat
    from app.db.models.user import User

    uid = str(uuid.uuid4())
    today = datetime.now(timezone.utc).date()
    dc = f"{uid}-dc"
    conv = f"{uid}-conv"
    cmid = "cm-" + uuid.uuid4().hex[:10]
    now = datetime.utcnow()

    async with async_session_maker() as db:
        db.add(User(id=uid, email=f"r46-{uuid.uuid4().hex[:10]}@example.com",
                    hashed_password="x", name="R46", timezone="UTC"))
        db.add(DayChat(id=dc, user_id=uid, local_date=today, timezone="UTC"))
        await db.flush()
        db.add(Conversation(id=conv, user_id=uid, channel="mobile", day_chat_id=dc,
                            started_at=now))
        await db.flush()
        ident = {"client_msg_id": cmid, "occurred_at": now} if stamped else {}
        db.add(Message(id=f"{uid}-u", conversation_id=conv, day_chat_id=dc,
                       channel="mobile", role="user", content="Hi",
                       created_at=now, **ident))
        db.add(Message(id=f"{uid}-a", conversation_id=conv, day_chat_id=dc,
                       channel="mobile", role="assistant", content="Hello",
                       created_at=now + timedelta(seconds=2),
                       **({"client_msg_id": cmid} if stamped else {})))
        await db.commit()
    return uid, dc, conv, cmid, today


async def _day_rows(uid, day):
    """The REAL day-chat message route."""
    from sqlalchemy import select
    import app.api.day_chats as api
    from app.db.database import async_session_maker
    from app.db.models.user import User

    real = api._get_agent_proxy_info

    async def _no_proxy(*a, **k):
        return None

    api._get_agent_proxy_info = _no_proxy
    try:
        async with async_session_maker() as db:
            user = (await db.execute(select(User).where(User.id == uid))).scalar_one()
            out = await api.get_day_chat_messages(
                date_str=day.isoformat(), limit=500, current_user=user, db=db,
            )
    finally:
        api._get_agent_proxy_info = real
    if isinstance(out, list):
        return out
    return json.loads(bytes(out.body).decode())


async def test_every_day_chat_row_carries_the_identity_keys():
    uid, dc, conv, cmid, day = await _seed_day(stamped=True)
    rows = await _day_rows(uid, day)
    assert len(rows) == 2, rows
    for r in rows:
        assert "client_msg_id" in r, r
        assert "occurred_at" in r, r
    assert [r["role"] for r in rows] == ["user", "assistant"], rows
    assert all(r["client_msg_id"] == cmid for r in rows)
    # The turn's receipt time belongs to the QUESTION. Given to the answer as
    # well, both rows share the sort key and the uuid tie-break decides which
    # one renders first.
    assert rows[0]["occurred_at"] is not None
    assert rows[1]["occurred_at"] is None, rows


async def test_the_keys_are_present_and_NULL_on_an_unstamped_row():
    """Null means UNKNOWN, never "not mine" — and a key that is simply MISSING
    from the JSON is a third state the client has no branch for."""
    uid, dc, conv, cmid, day = await _seed_day(stamped=False)
    rows = await _day_rows(uid, day)
    assert len(rows) == 2, rows
    for r in rows:
        assert "client_msg_id" in r and r["client_msg_id"] is None, r
        assert "occurred_at" in r and r["occurred_at"] is None, r


async def test_a_row_that_HAPPENED_earlier_sorts_earlier():
    """A voice or WhatsApp row can be WRITTEN long after it happened; sorted by
    write time it lands in the wrong place in the day."""
    from app.db.database import async_session_maker
    from app.db.models import Message

    uid, dc, conv, cmid, day = await _seed_day(stamped=False)
    now = datetime.utcnow()
    async with async_session_maker() as db:
        db.add(Message(id=f"{uid}-spoken", conversation_id=conv, day_chat_id=dc,
                       channel="voice", role="user", content="spoken",
                       created_at=now + timedelta(minutes=10),
                       occurred_at=now - timedelta(minutes=5)))
        await db.commit()

    rows = await _day_rows(uid, day)
    assert [r["id"] for r in rows][0] == f"{uid}-spoken", [r["id"] for r in rows]


async def test_the_session_serializer_carries_them_too():
    """`/api/sessions/...` is the FALLBACK the mobile client takes whenever
    /api/day-chats fails, so a field it drops is a field that disappears
    exactly when the app is already degraded."""
    from sqlalchemy import select
    from app.api.sessions import _message_to_response
    from app.db.database import async_session_maker
    from app.db.models import Message

    uid, dc, conv, cmid, day = await _seed_day(stamped=True)
    async with async_session_maker() as db:
        msgs = (await db.execute(
            select(Message).where(Message.conversation_id == conv)
            .order_by(Message.created_at.asc())
        )).scalars().all()
        out = [_message_to_response(m, None, None, {conv: "mobile"}) for m in msgs]

    assert [o.client_msg_id for o in out] == [cmid, cmid]
    assert out[0].occurred_at is not None
    assert out[1].occurred_at is None


def test_the_self_heal_creates_the_columns_AND_their_indexes():
    """Agent DBs have no alembic — `_alter_statements` IS the migration. The
    ORM backstop (`_reconcile_missing_columns`) would add the COLUMNS on its
    own, so a `create_all`-built test database proves nothing about them; the
    INDEXES have no backstop at all, and the one on `client_msg_id` is what
    makes `turn_receipt`'s lookup cheap on a long day thread."""
    import inspect

    from app.db import database as db_mod

    src = inspect.getsource(db_mod.init_db)
    for stmt in (
        "ALTER TABLE messages ADD COLUMN IF NOT EXISTS client_msg_id VARCHAR(100)",
        "CREATE INDEX IF NOT EXISTS ix_messages_client_msg_id ON messages (client_msg_id)",
        "ALTER TABLE messages ADD COLUMN IF NOT EXISTS occurred_at TIMESTAMP",
        "CREATE INDEX IF NOT EXISTS ix_messages_occurred_at ON messages (occurred_at)",
    ):
        assert stmt in src, f"missing self-heal statement: {stmt}"
