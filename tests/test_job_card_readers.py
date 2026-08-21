"""Round 16 — the three history READERS, driven for real.

The pure half of this round is ``test_job_card_leak.py`` (platform sweep).
This file is the half that needs the AGENT_ONLY tables — ``messages``,
``conversations``, ``day_chats``, ``build_jobs`` — because the only honest
proof that a serializer does not leak its marker is to seed a real voice
turn and call the real route.

RUN_MODE=agent. Listed in COVERAGE_DEBT.txt for that reason.

Control: check out ``app/api/day_chats.py`` from the parent commit and
``test_day_chats_never_returns_the_marker_as_text`` fails — the row comes
back with the raw ``{"job_id": …}`` marker as its body. Verified 2026-08-21.
"""
from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timedelta, date as Date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.test_job_card_leak import (  # noqa: E402
    LEAKED_JOB_ID, LEAKED_MARKER, LEAKED_TITLE, _FakeJob, _web_record,
)

# asyncio_mode = auto (pytest.ini) — the async tests below need no mark.


async def _seed_voice_turn(day: Date):
    """A voice turn exactly as it lands: the user's transcript, the card
    row the agent's runner wrote, and the answer the platform relay wrote
    carrying the run's tool records."""
    from app.db.database import async_session_maker
    from app.db.models import BuildJob, Conversation, Message, User
    from app.db.models.day_chat import DayChat

    user_id = str(uuid.uuid4())
    conv_id = str(uuid.uuid4())
    dc_id = str(uuid.uuid4())
    base = datetime(day.year, day.month, day.day, 10, 0, 0)

    async with async_session_maker() as db:
        db.add(User(id=user_id, email=f"{user_id}@t.local",
                    hashed_password="x", name="T"))
        db.add(DayChat(id=dc_id, user_id=user_id, local_date=day,
                       timezone="UTC", started_at=base, last_message_at=base))
        db.add(Conversation(id=conv_id, user_id=user_id, channel="voice",
                            day_chat_id=dc_id, started_at=base, updated_at=base))
        db.add(BuildJob(
            id=LEAKED_JOB_ID, user_id=user_id, title=LEAKED_TITLE,
            prompt=LEAKED_TITLE, status="completed", layer=0,
            steps_json=_FakeJob.steps_json,
            config_json=_FakeJob.config_json,
        ))
        db.add(Message(id="m-user", conversation_id=conv_id, day_chat_id=dc_id,
                       role="user", content="بهترین مدل تصویرسازی چیه؟",
                       created_at=base))
        db.add(Message(id=f"job-{LEAKED_JOB_ID}", conversation_id=conv_id,
                       day_chat_id=dc_id, role="job", content=LEAKED_MARKER,
                       created_at=base + timedelta(seconds=1)))
        db.add(Message(
            id="m-asst", conversation_id=conv_id, day_chat_id=dc_id,
            role="assistant", content="بر اساس جست‌وجو…",
            created_at=base + timedelta(seconds=2),
            metadata_json=json.dumps({"tool_events": [_web_record(LEAKED_JOB_ID)]}),
        ))
        await db.commit()
    return user_id, conv_id


async def _current_user(user_id: str):
    from app.db.database import async_session_maker
    from app.db.models import User
    async with async_session_maker() as db:
        return await db.get(User, user_id)


async def test_day_chats_never_returns_the_marker_as_text():
    """THE regression. This is the route every client asks first."""
    from app.api.day_chats import get_day_chat_messages
    from app.db.database import async_session_maker

    day = Date(2026, 8, 21)
    user_id, _ = await _seed_voice_turn(day)
    user = await _current_user(user_id)

    async with async_session_maker() as db:
        resp = await get_day_chat_messages(
            day.isoformat(), limit=500, current_user=user, db=db,
        )
    rows = json.loads(resp.body)

    card = next(r for r in rows if r["role"] == "job")
    assert card["content"] == ""
    assert LEAKED_JOB_ID not in json.dumps(
        [r.get("content") for r in rows], ensure_ascii=False
    )
    assert "\\u" not in "".join(r.get("content") or "" for r in rows)
    # …and it is a CARD, with everything the live one had.
    assert card["job_id"] == LEAKED_JOB_ID
    assert card["job_name"] == LEAKED_TITLE
    assert card["job_total_steps"] == 3
    assert len(card["job_steps"]) == 3
    # One job, one card: the run's records — sources and all — ride it.
    assert card["tool_events"][0]["sources"][0]["title"] == "A paper"


async def test_sessions_by_date_agrees_with_day_chats():
    """The FALLBACK reader. A field only one serializer emits vanishes the
    moment the client falls back — the bug class this round is built on."""
    from app.api.sessions import get_messages_by_date
    from app.db.database import async_session_maker

    day = Date(2026, 8, 21)
    user_id, _ = await _seed_voice_turn(day)
    user = await _current_user(user_id)

    async with async_session_maker() as db:
        resp = await get_messages_by_date(
            day.isoformat(), limit=200, tz_offset=0, current_user=user, db=db,
        )
    rows = json.loads(resp.body)
    card = next(r for r in rows if r["role"] == "job")
    assert card["content"] == ""
    assert card["job_name"] == LEAKED_TITLE
    assert card["job_total_steps"] == 3
    assert len(card["job_steps"]) == 3
    assert card["tool_events"][0]["sources"][0]["title"] == "A paper"


async def test_messages_since_never_returns_the_marker_as_text():
    """The third reader — the resync path after a dropped socket."""
    from app.api.messages_recover import messages_since
    from app.db.database import async_session_maker

    day = Date(2026, 8, 21)
    user_id, _ = await _seed_voice_turn(day)
    user = await _current_user(user_id)

    async with async_session_maker() as db:
        resp = await messages_since(
            "m-user", limit=100, current_user=user, db=db,
        )
    rows = json.loads(resp.body)
    card = next(r for r in rows if r["role"] == "job")
    assert card["content"] == ""
    assert card["job_name"] == LEAKED_TITLE
    assert len(card["job_steps"]) == 3


