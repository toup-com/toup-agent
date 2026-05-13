"""Ticket 5 regression tests — channel unification invariant.

Locks two properties:

  1. `day_chats` table has `UNIQUE(user_id, local_date)`. Every channel
     (web, telegram, whatsapp, voice, extension) writes to the same
     DayChat for a given calendar day because there can only be ONE.

  2. `resolve_day_chat_id_for_now` is the canonical resolver every
     channel ingress flows through. Pinned via source-grep so a future
     channel addition can't bypass the helper.
"""

from __future__ import annotations

import uuid
from datetime import date

import pytest
import pytest_asyncio
from pathlib import Path
from sqlalchemy.exc import IntegrityError


BACKEND = Path(__file__).resolve().parent.parent
_DAY_CHAT_MODEL = (BACKEND / "app/db/models/day_chat.py").read_text()
_WS_CHAT = (BACKEND / "app/api/ws_chat.py").read_text()
_MESSAGE_HELPERS = (BACKEND / "app/db/message_helpers.py").read_text()


def test_day_chat_carries_unique_user_date_constraint():
    """Floor invariant: `(user_id, local_date)` uniqueness on
    `day_chats`. Without this, two channels could create parallel
    DayChat rows for the same user on the same day and the agent's
    context would fragment."""
    assert 'UniqueConstraint("user_id", "local_date"' in _DAY_CHAT_MODEL, (
        "DayChat model must declare a UniqueConstraint on (user_id, "
        "local_date). Drop this and two channels on the same day "
        "produce two DayChat rows — Day-as-Chat unification breaks."
    )


def test_ws_chat_routes_through_resolve_day_chat_helper():
    """Every channel ingress (web/telegram/whatsapp/voice/extension)
    that lands in `/ws/chat` must resolve its DayChat via
    `resolve_day_chat_id_for_now`. Bypassing the helper would let a
    caller create a Message with a misaligned `day_chat_id`."""
    assert "_resolve_day_chat_id_for_now" in _WS_CHAT, (
        "ws_chat.py must use _resolve_day_chat_id_for_now to assign "
        "day_chat_id to every persisted Message. Without this, the "
        "Day-as-Chat invariant breaks: messages from different "
        "channels could land in different DayChat rows."
    )


def test_resolve_day_chat_helper_is_canonical():
    """The helper itself must exist with the expected signature. Pin
    via source-grep so a refactor renaming/removing it gets caught."""
    assert "async def resolve_day_chat_id_for_now(" in _MESSAGE_HELPERS, (
        "resolve_day_chat_id_for_now must remain the canonical helper "
        "for day_chat_id assignment. Channel ingresses depend on its "
        "exact name (ws_chat.py grep-imports it)."
    )


@pytest_asyncio.fixture
async def unification_user():
    from app.db import User, async_session_maker
    user_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(
            id=user_id,
            email=f"unif-{user_id[:8]}@example.com",
            hashed_password="x",
            name="Unification Test",
        ))
        await db.commit()
    return user_id


@pytest.mark.asyncio
async def test_cannot_insert_two_day_chats_for_same_user_date(unification_user):
    """Floor invariant exercised at the DB layer — two raw inserts of
    DayChat with the same (user_id, local_date) MUST fail."""
    from app.db import async_session_maker
    from app.db.models import DayChat

    async with async_session_maker() as db:
        db.add(DayChat(
            id=str(uuid.uuid4()),
            user_id=unification_user,
            local_date=date.today(),
            timezone="UTC",
        ))
        await db.commit()

    async with async_session_maker() as db:
        db.add(DayChat(
            id=str(uuid.uuid4()),
            user_id=unification_user,
            local_date=date.today(),
            timezone="UTC",
        ))
        with pytest.raises(IntegrityError):
            await db.commit()
