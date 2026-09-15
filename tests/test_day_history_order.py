"""The model and the user must see one order (independent defect, group 7).

`app/agent/day_context_loader.py:191` orders history by
`(Message.created_at.asc(), Message.id.asc())`. Both REST serializers —
`day_chats.py:792` and `:875` — order by `created_at` alone. Two rows sharing a
timestamp can therefore come back in a different order on two fetches, and in
a different order than the model was shown.

Theoretical until this incident; ordinary after it. The whole point of the tz
and rebucket work is that WhatsApp and mobile rows finally share one day chat,
and the presave user row plus a fast-path assistant row land in the same
millisecond routinely. A question and its answer swapping places is a
history-honesty bug the client cannot fix: ChatScreen re-sorts, but it sorts
by the same `created_at` the server did.

The two source pins are RED on this base by design and are xfail(strict=True):
they go red again if lane B3 lands without the tiebreaker.

Local run (from backend/):
    RUN_MODE=platform PYTHONPATH=. ENVIRONMENT=test JWT_SECRET=x ENCRYPTION_KEY=y \
      pytest tests/test_day_history_order.py -q --tb=short -p no:cacheprovider
"""
from __future__ import annotations

import json
import re
import uuid
from datetime import date as Date, datetime
from pathlib import Path

import pytest
import pytest_asyncio

_BACKEND = Path(__file__).resolve().parent.parent
_DAY_CHATS_SRC = (_BACKEND / "app" / "api" / "day_chats.py").read_text()

# `order_by(Message.created_at.asc())` with nothing after it.
_UNTIEBROKEN = re.compile(r"order_by\(\s*Message\.created_at\.asc\(\)\s*\)")


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


# ── source pins ───────────────────────────────────────────────────────


@pytest.mark.xfail(bool(_UNTIEBROKEN.search(_DAY_CHATS_SRC)),
                   reason="RED until lane B3 adds the id tiebreaker (group 7)",
                   strict=True)
def test_both_rest_serializers_break_ties_on_id():
    assert not _UNTIEBROKEN.search(_DAY_CHATS_SRC), (
        "a day-messages serializer still orders by created_at alone — two "
        "rows sharing a timestamp come back in an arbitrary order, and in a "
        "different order than day_context_loader showed the model"
    )


def test_the_loader_already_has_the_tiebreaker_this_pins_against():
    """ANTI-VACUITY: the test above is only meaningful because there IS a
    canonical order to agree with."""
    src = (_BACKEND / "app" / "agent" / "day_context_loader.py").read_text()
    assert "Message.created_at.asc(), Message.id.asc()" in src


# ── behaviour ─────────────────────────────────────────────────────────


async def _seed_collision(n=6):
    """`n` rows sharing one created_at to the microsecond, across two
    channels, inserted in DESCENDING id order so insertion order and id order
    disagree — otherwise sqlite's rowid order masks the missing tiebreaker."""
    from app.db.database import async_session_maker
    from app.db.models import Conversation, Message
    from app.db.models.day_chat import DayChat
    from app.db.models.user import User

    uid, dc_id = str(uuid.uuid4()), str(uuid.uuid4())
    day = Date(2026, 9, 14)
    stamp = datetime(2026, 9, 14, 9, 30, 15, 123456)
    ids = [f"m-{i:02d}" for i in range(n)]

    async with async_session_maker() as db:
        db.add(User(id=uid, email=f"ord-{uuid.uuid4().hex[:10]}@example.com",
                    hashed_password="x", name="Ord", timezone="UTC"))
        db.add(DayChat(id=dc_id, user_id=uid, local_date=day, timezone="UTC"))
        await db.flush()
        convs = {}
        for ch in ("web", "whatsapp"):
            cid = str(uuid.uuid4())
            convs[ch] = cid
            db.add(Conversation(id=cid, user_id=uid, channel=ch, day_chat_id=dc_id,
                                started_at=datetime(2026, 9, 14, 9, 0)))
        await db.flush()
        for k, mid in enumerate(reversed(ids)):
            ch = "web" if k % 2 == 0 else "whatsapp"
            db.add(Message(id=mid, conversation_id=convs[ch], day_chat_id=dc_id,
                           channel=ch, role="user", content=mid, created_at=stamp))
        await db.commit()
    return uid, dc_id, day, ids


async def _rest_ids(uid, day):
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
            resp = await api.get_day_chat_messages(
                date_str=day.isoformat(), limit=500, current_user=user, db=db,
            )
    finally:
        api._get_agent_proxy_info = real
    return [r["id"] for r in json.loads(bytes(resp.body).decode())]


@pytest.mark.asyncio
async def test_two_consecutive_fetches_return_the_same_sequence(day_tables):
    uid, dc_id, day, ids = await _seed_collision()
    assert await _rest_ids(uid, day) == await _rest_ids(uid, day)


@pytest.mark.xfail(bool(_UNTIEBROKEN.search(_DAY_CHATS_SRC)),
                   reason="RED until lane B3 adds the id tiebreaker (group 7)",
                   strict=True)
@pytest.mark.asyncio
async def test_the_rest_order_matches_the_order_the_model_was_shown(day_tables):
    """The invariant that matters: the user reading the thread and the model
    reading the day must not disagree about which message came first."""
    from app.agent.day_context_loader import load_day_context
    from app.db.database import async_session_maker

    uid, dc_id, day, ids = await _seed_collision()

    async with async_session_maker() as db:
        ctx = await load_day_context(db, dc_id, model_context_tokens=200_000, tz_name="UTC")
    loader_order = [m.get("id") for m in ctx["messages"] if m.get("id")]
    rest_order = await _rest_ids(uid, day)

    if not loader_order:
        # The loader's rows carry no id on this build; fall back to the body,
        # which is the id by construction above.
        loader_order = [m["content"].split("] ")[-1] for m in ctx["messages"]]

    assert rest_order == loader_order, (
        f"REST and loader disagree on tied timestamps: REST {rest_order} vs "
        f"loader {loader_order} — the user and the model are reading two "
        "different conversations"
    )


def test_created_at_cannot_be_null_so_the_sink_order_question_does_not_arise():
    """The tests-ci plan asked for a "NULL created_at sinks deterministically"
    case. It cannot be written: `messages.created_at` is NOT NULL (the insert
    raises IntegrityError), so the only rows that could float do not exist.

    Recorded rather than dropped, because both serializers carry a defensive
    `m.created_at.isoformat() if m.created_at else None` whose premise this
    is. If the column ever becomes nullable, this test goes red and the
    ordering question becomes real.
    """
    from app.db.models import Message

    assert Message.__table__.c.created_at.nullable is False
