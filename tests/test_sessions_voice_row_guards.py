"""Tenant persistence guards for voice rows (P3 tenant, PR-11, 2026-10-02 call).

Two defects in `POST /api/sessions/{id}/messages`, both seen in the incident:

* **A late commit rewound a newer revision.** The relay abandons a POST after a
  ReadTimeout, but a stalled tenant still commits it, sometimes a minute
  later. The upsert on (conversation_id, client_msg_id, role) overwrote content
  and metadata unconditionally, so an abandoned revision N that commits after
  revision N+1 put the older sentence back. The relay already sends the row's
  monotonic `revision`. The tenant now keeps the highest one it applied and
  ignores a write that carries a LOWER one. Equal is applied, because a retry
  of the same revision is idempotent.
* **A late row landed in the wrong day.** The DayChat was resolved from the
  moment the request ARRIVED, not from `occurred_at` (the relay's stamp of
  when the utterance happened). A row persisted after the user's local
  midnight was therefore filed under the next day. A flag-only revision, which
  carries no `occurred_at`, also MOVED an existing row into "today".

Lane: RUN_MODE=agent. conversations / messages / day_chats are AGENT_ONLY and
this drives the REAL writer.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from app.config import settings
from app.db import async_session_maker
from app.db.models import Conversation, DayChat, Message, User
from app.schemas import SessionMessageCreate

AGENT_KEY = "test-agent-key-row-guards"


class _Req:
    def __init__(self, headers: dict | None = None):
        self.headers = dict(headers or {})


@pytest.fixture(autouse=True)
def _agent_key(monkeypatch):
    monkeypatch.setattr(settings, "agent_api_key", AGENT_KEY, raising=False)
    yield


async def _seed(tz: str | None = None, started_at: datetime | None = None):
    user_id, conv_id = str(uuid.uuid4()), str(uuid.uuid4())
    extra = {"started_at": started_at} if started_at is not None else {}
    async with async_session_maker() as db:
        db.add(User(id=user_id, email=f"guard-{user_id[:8]}@example.test",
                    hashed_password="x", name="Guard Test", timezone=tz))
        db.add(Conversation(id=conv_id, user_id=user_id, title="Voice",
                            channel="voice", message_count=0, total_tokens=0,
                            **extra))
        await db.commit()
    return user_id, conv_id


async def _post(conv_id, user_id, body, frames):
    from app.api import sessions as sessions_mod
    import app.api.ws_chat as ws_chat

    async def _capture(uid, frame):
        frames.append(frame)

    orig = ws_chat.broadcast_to_user
    ws_chat.broadcast_to_user = _capture
    try:
        async with async_session_maker() as db:
            user = await db.get(User, user_id)
            out = await sessions_mod.create_session_message(
                session_id=conv_id, body=body,
                request=_Req({"x-agent-key": AGENT_KEY}),
                current_user=user, db=db,
            )
        await asyncio.sleep(0)
        return out
    finally:
        ws_chat.broadcast_to_user = orig


async def _rows(conv_id):
    from sqlalchemy import select
    async with async_session_maker() as db:
        return (await db.execute(select(Message).where(
            Message.conversation_id == conv_id))).scalars().all()


async def _day_rows(user_id, day):
    """What GET /api/day-chats/{date}/messages serves for that local date."""
    import app.api.day_chats as api

    async def _no_proxy(*a, **k):
        return None

    orig = api._get_agent_proxy_info
    api._get_agent_proxy_info = _no_proxy
    try:
        async with async_session_maker() as db:
            user = await db.get(User, user_id)
            out = await api.get_day_chat_messages(
                date_str=day.isoformat(), limit=500, current_user=user, db=db,
            )
    finally:
        api._get_agent_proxy_info = orig
    return out if isinstance(out, list) else json.loads(bytes(out.body).decode())


async def _local_date_of(day_chat_id):
    async with async_session_maker() as db:
        dc = await db.get(DayChat, day_chat_id)
        return dc.local_date if dc else None


# ── (1) a late commit does not rewind a newer revision ──────────────────

@pytest.mark.asyncio
async def test_a_late_commit_of_an_older_revision_does_not_rewind_the_row():
    """The incident's shape: revision 1 is abandoned by the relay after a
    ReadTimeout, revision 2 lands, and then the stalled tenant commits
    revision 1. The row must keep revision 2's sentence and metadata."""
    user_id, conv_id = await _seed()
    frames: list = []
    key = str(uuid.uuid4())

    await _post(conv_id, user_id, SessionMessageCreate(
        role="assistant", content="I couldn't", client_msg_id=key, revision=0,
        voice={"source": "live_spoken", "epoch": 7},
    ), frames)
    await _post(conv_id, user_id, SessionMessageCreate(
        role="assistant", content="I couldn't find specific info.",
        client_msg_id=key, revision=2,
        voice={"source": "live_spoken", "epoch": 7, "played_ms": 4100},
    ), frames)
    # The abandoned POST, committing late.
    late = await _post(conv_id, user_id, SessionMessageCreate(
        role="assistant", content="I couldn't find", client_msg_id=key,
        revision=1, voice={"source": "live_spoken", "epoch": 7},
    ), frames)

    rows = await _rows(conv_id)
    assert len(rows) == 1, [r.content for r in rows]
    assert rows[0].content == "I couldn't find specific info."
    meta = json.loads(rows[0].metadata_json or "{}")
    assert meta["voice"].get("played_ms") == 4100, meta
    # The relay treats the stale write as handled (no retry loop) and the
    # response describes the row that exists, not the stale body.
    assert late.content == "I couldn't find specific info."
    # No frame claims the stale sentence: an open thread replaces on
    # revision, and must not be handed the older text.
    assert [f.get("revision") for f in frames] == [0, 2], frames


@pytest.mark.asyncio
async def test_a_stale_flag_revision_does_not_undo_a_newer_one():
    """`write_spoken_flag` revises metadata only. An older flag revision that
    commits late must not resurrect the older verdict."""
    user_id, conv_id = await _seed()
    frames: list = []
    key = str(uuid.uuid4())
    base = {"source": "delegated", "delegation_id": "dlg-1"}

    await _post(conv_id, user_id, SessionMessageCreate(
        role="assistant", content="The answer.", client_msg_id=key,
        revision=1, voice=dict(base),
    ), frames)
    await _post(conv_id, user_id, SessionMessageCreate(
        role="assistant", content="The answer.", client_msg_id=key,
        revision=3, voice={**base, "spoken": True},
    ), frames)
    await _post(conv_id, user_id, SessionMessageCreate(
        role="assistant", content="The answer.", client_msg_id=key,
        revision=2, voice={**base, "spoken": False},
    ), frames)

    rows = await _rows(conv_id)
    meta = json.loads(rows[0].metadata_json or "{}")
    assert meta["voice"]["spoken"] is True, meta


@pytest.mark.asyncio
async def test_a_retry_of_the_same_revision_is_still_applied():
    """Greater-or-equal, not strictly greater: the relay's retry of a write
    re-sends the same revision, and that upsert stays idempotent."""
    user_id, conv_id = await _seed()
    frames: list = []
    key = str(uuid.uuid4())
    for content in ("I need a hotel", "I need a hotel"):
        await _post(conv_id, user_id, SessionMessageCreate(
            role="user", content=content, client_msg_id=key, revision=4,
        ), frames)
    rows = await _rows(conv_id)
    assert len(rows) == 1 and rows[0].content == "I need a hotel"
    assert [f.get("revision") for f in frames] == [4, 4]


@pytest.mark.asyncio
async def test_a_write_without_a_revision_keeps_today_s_overwrite():
    """Absent is "unknown", never "older": a caller that sends no revision (the
    Realtime path, an older relay) still upserts as it always has."""
    user_id, conv_id = await _seed()
    frames: list = []
    key = str(uuid.uuid4())
    await _post(conv_id, user_id, SessionMessageCreate(
        role="user", content="first", client_msg_id=key, revision=5,
    ), frames)
    await _post(conv_id, user_id, SessionMessageCreate(
        role="user", content="second", client_msg_id=key,
    ), frames)
    rows = await _rows(conv_id)
    assert rows[0].content == "second"
    # …and the stored high-water mark survives it, so a late revision 3 is
    # still recognised as stale afterwards.
    await _post(conv_id, user_id, SessionMessageCreate(
        role="user", content="late", client_msg_id=key, revision=3,
    ), frames)
    rows = await _rows(conv_id)
    assert rows[0].content == "second"


@pytest.mark.asyncio
async def test_the_stored_revision_is_not_part_of_any_client_response():
    """The high-water mark is bookkeeping inside metadata_json. Every reader
    picks named keys out of that column, so none of them returns it."""
    user_id, conv_id = await _seed()
    frames: list = []
    resp = await _post(conv_id, user_id, SessionMessageCreate(
        role="user", content="hello", client_msg_id=str(uuid.uuid4()),
        revision=2,
    ), frames)
    assert "persist_revision" not in resp.model_dump_json()
    assert "persist_revision" not in json.dumps(frames[0], default=str)
    # The day-chat route, the primary history reader.
    rows = await _rows(conv_id)
    day = await _local_date_of(rows[0].day_chat_id)
    served = await _day_rows(user_id, day)
    assert [r["id"] for r in served] == [rows[0].id]
    assert "persist_revision" not in json.dumps(served, default=str)


# ── (2) the day comes from occurred_at ──────────────────────────────────

def _zone_with_local_morning(now_utc: datetime) -> str:
    """A fixed-offset zone where it is currently between 01:00 and 13:00, so
    "half an hour before local midnight" is unambiguously the previous local
    day and well inside the route's 24 h skew window, whatever the real clock
    says when the suite runs."""
    for off in range(-12, 15):
        name = f"Etc/GMT{'+' if off <= 0 else '-'}{abs(off)}" if off else "Etc/GMT"
        local = now_utc.astimezone(ZoneInfo(name))
        if 1 <= local.hour < 13:
            return name
    raise AssertionError("no fixed-offset zone has a local morning right now")


@pytest.mark.asyncio
async def test_a_row_that_occurred_yesterday_is_filed_under_yesterday():
    now_utc = datetime.now(timezone.utc)
    tz = _zone_with_local_morning(now_utc)
    local_now = now_utc.astimezone(ZoneInfo(tz))
    local_midnight = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
    occurred = (local_midnight - timedelta(minutes=30)).astimezone(timezone.utc)
    yesterday = (local_midnight - timedelta(minutes=30)).date()

    # The call started before midnight too (naive UTC, like every column).
    user_id, conv_id = await _seed(
        tz, started_at=(occurred - timedelta(minutes=5)).replace(tzinfo=None))
    frames: list = []
    key = str(uuid.uuid4())
    resp = await _post(conv_id, user_id, SessionMessageCreate(
        role="assistant", content="Spoken before midnight, saved after.",
        client_msg_id=key, occurred_at=occurred, revision=1,
    ), frames)

    rows = await _rows(conv_id)
    assert await _local_date_of(rows[0].day_chat_id) == yesterday, (
        "a row persisted late must land in the day it happened"
    )
    assert frames[0]["day_chat_id"] == rows[0].day_chat_id
    # …and that is where the day route serves it from.
    assert [r["id"] for r in await _day_rows(user_id, yesterday)] == [rows[0].id]
    assert await _day_rows(user_id, local_now.date()) == []

    # A flag-only revision carries no occurred_at. It must not MOVE the row
    # into today.
    await _post(conv_id, user_id, SessionMessageCreate(
        role="assistant", content="Spoken before midnight, saved after.",
        client_msg_id=key, revision=2, voice={"source": "delegated",
                                              "spoken": False},
    ), frames)
    rows = await _rows(conv_id)
    assert await _local_date_of(rows[0].day_chat_id) == yesterday
    assert frames[-1]["day_chat_id"] == rows[0].day_chat_id
    assert resp.id == rows[0].id


@pytest.mark.asyncio
async def test_a_row_without_occurred_at_is_still_filed_under_today():
    """Unchanged: no stamp means the arrival time decides."""
    now_utc = datetime.now(timezone.utc)
    tz = _zone_with_local_morning(now_utc)
    today = now_utc.astimezone(ZoneInfo(tz)).date()
    user_id, conv_id = await _seed(tz)
    frames: list = []
    await _post(conv_id, user_id, SessionMessageCreate(
        role="user", content="hello", client_msg_id=str(uuid.uuid4()),
    ), frames)
    rows = await _rows(conv_id)
    assert await _local_date_of(rows[0].day_chat_id) == today
