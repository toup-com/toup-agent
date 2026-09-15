"""One WhatsApp session per (chat, day) — and never someone else's chat (D9).

`_resolve_session_id` (message_handler.py:97) selects the newest ACTIVE
conversation for `(user_id, channel)` — with no predicate on the chat id and
none on the day. Three consequences, all live:

  1. Two different WhatsApp chats share one Conversation row, so the turn from
     chat B is threaded into chat A's session.
  2. A conversation opened yesterday is still "the newest active whatsapp row"
     today, so a turn after local midnight lands in yesterday's thread.
  3. The cache is checked and filled without a lock, so two inbound messages
     arriving together both miss, both reach `run()`, and both create a row.

D9 adds `run(channel_chat_id=...)` stamped into `Conversation.metadata_json`,
filters the resolve by today's day chat + channel + chat id, and serialises
resolve→run→cache per (channel, chat).

Local run (from backend/):
    RUN_MODE=platform PYTHONPATH=. ENVIRONMENT=test JWT_SECRET=x ENCRYPTION_KEY=y \
      pytest tests/test_whatsapp_session_per_day.py -q --tb=short -p no:cacheprovider
"""
from __future__ import annotations

import asyncio
import inspect
import uuid
from datetime import date as Date, datetime, timedelta
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from app.agent.channels.base import BaseChannel, ChannelType, InboundMessage

CHAT_A = "+14155552671"
CHAT_B = "+14155559999"


def _d9_landed() -> bool:
    """One probe for the group: `run(channel_chat_id=)` is what the rest of
    D9 is built on, so marking each case on its own private guess would leave
    a case xfailing after its fix arrived."""
    from app.agent.agent_runner import AgentRunner

    return "channel_chat_id" in inspect.signature(AgentRunner.run).parameters


_XFAIL_D9 = pytest.mark.xfail(
    not _d9_landed(), reason="RED until lane B2 lands D9 (per-chat, per-day sessions)",
    strict=True,
)


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


class FakeChannel(BaseChannel):
    def __init__(self):
        super().__init__(ChannelType.WHATSAPP)
        self.sent: List[tuple] = []

    async def start(self):
        pass

    async def stop(self):
        pass

    async def send_text(self, chat_id, text, parse_mode=None):
        self.sent.append((chat_id, text))

    async def send_typing(self, chat_id):
        pass


def _inbound(text="hi", chat_id=CHAT_A):
    return InboundMessage(channel=ChannelType.WHATSAPP, channel_user_id=chat_id,
                          channel_chat_id=chat_id, text=text, media_paths=[])


def _runner_that_records(created: List[dict], delay: float = 0.0):
    """A stub AgentRunner that mints a Conversation row per call, the way the
    real one does when `session_id` is None."""
    async def run(**kw):
        if delay:
            await asyncio.sleep(delay)
        sid = kw.get("session_id") or str(uuid.uuid4())
        created.append(dict(kw, session_id_out=sid))
        resp = MagicMock()
        resp.text = "ok"
        resp.session_id = sid
        resp.tokens_total = 0
        resp.tool_calls = []
        resp.processing_time_ms = 1
        resp.persisted = {}
        resp.day_chat_id = None  # a MagicMock attribute here is a truthy fake day id
        return resp

    runner = MagicMock()
    runner.run = AsyncMock(side_effect=run)
    return runner


async def _seed_conversation(uid, *, chat_id, day, active=True, updated_at=None):
    from app.db.database import async_session_maker
    from app.db.models import Conversation
    from app.db.models.day_chat import DayChat

    dc_id, conv_id = str(uuid.uuid4()), str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(DayChat(id=dc_id, user_id=uid, local_date=day, timezone="UTC"))
        await db.flush()
        conv = Conversation(
            id=conv_id, user_id=uid, channel="whatsapp", day_chat_id=dc_id,
            is_active=active,
            started_at=datetime(day.year, day.month, day.day, 9, 0),
            updated_at=updated_at or datetime(day.year, day.month, day.day, 9, 0),
        )
        if hasattr(conv, "metadata_json"):
            conv.metadata_json = f'{{"channel_chat_id": "{chat_id}"}}'
        db.add(conv)
        await db.commit()
    return dc_id, conv_id


async def _mk_user():
    from app.db.database import async_session_maker
    from app.db.models import User

    uid = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(id=uid, email=f"wa-{uuid.uuid4().hex[:10]}@example.com",
                    hashed_password="x", name="WA", timezone="UTC"))
        await db.commit()
    return uid


# ── the contract surface ──────────────────────────────────────────────


def test_run_accepts_channel_chat_id():
    from app.agent.agent_runner import AgentRunner

    if "channel_chat_id" not in inspect.signature(AgentRunner.run).parameters:
        pytest.skip("run(channel_chat_id=) not landed yet — lane B2 (D9)")
    assert inspect.signature(AgentRunner.run).parameters["channel_chat_id"].default is None


def test_the_handler_passes_the_chat_id_through():
    """Without it there is nothing to filter on and the rest of D9 cannot be
    implemented at all — a silent no-op rather than a failure."""
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "app" / "agent" / "channels"
           / "shared" / "message_handler.py").read_text()
    if "channel_chat_id=" not in src:
        pytest.skip("handler does not pass channel_chat_id yet — lane B2 (D9)")
    assert "channel_chat_id=chat_id" in src or "channel_chat_id=msg.channel_chat_id" in src


# ── behaviour ─────────────────────────────────────────────────────────


@_XFAIL_D9
@pytest.mark.asyncio
async def test_another_chats_session_is_never_reused(day_tables):
    """The resolve orders by `updated_at` with no chat predicate, so the
    newest conversation wins whoever it belongs to. Chat B must not be
    threaded into chat A's session."""
    from app.agent.channels.shared import make_channel_handler
    from app.agent.channels.shared.message_handler import _resolve_session_id

    uid = await _mk_user()
    today = Date(2026, 9, 14)
    _, conv_a = await _seed_conversation(uid, chat_id=CHAT_A, day=today)

    resolved = await _resolve_session_id(uid, ChannelType.WHATSAPP, CHAT_B, {})
    assert resolved != conv_a, "chat B was threaded into chat A's session"


@_XFAIL_D9
@pytest.mark.asyncio
async def test_yesterdays_session_is_a_miss(day_tables):
    """Day-as-Chat rolls at LOCAL midnight. A conversation from yesterday is
    still the newest active whatsapp row, so without a day predicate the
    first turn of a new day is threaded into the old one."""
    from app.agent.channels.shared.message_handler import _resolve_session_id

    uid = await _mk_user()
    yesterday = Date(2026, 9, 13)
    _, conv_y = await _seed_conversation(
        uid, chat_id=CHAT_A, day=yesterday,
        updated_at=datetime(2026, 9, 13, 23, 59),
    )

    resolved = await _resolve_session_id(uid, ChannelType.WHATSAPP, CHAT_A, {})
    assert resolved != conv_y, "today's first turn landed in yesterday's thread"


@_XFAIL_D9
@pytest.mark.asyncio
async def test_a_cached_session_from_yesterday_is_not_reused(day_tables):
    """The in-process cache is keyed `(channel, chat)` with no date, and the
    container runs for weeks. A cache hit skips the DB entirely, so the day
    predicate above never even gets a chance."""
    from app.agent.channels.shared.message_handler import _resolve_session_id

    uid = await _mk_user()
    stale = {("whatsapp", CHAT_A): "conv-from-yesterday"}
    resolved = await _resolve_session_id(uid, ChannelType.WHATSAPP, CHAT_A, stale)
    assert resolved != "conv-from-yesterday", "the session cache carries no day"


@_XFAIL_D9
@pytest.mark.asyncio
async def test_two_concurrent_messages_for_one_chat_create_one_session(day_tables):
    """Check-then-fill with an await in between. Two inbound frames in the
    same tick both miss the cache, both reach run(), and both mint a row."""
    from app.agent.channels.shared import make_channel_handler

    uid = await _mk_user()
    created: List[dict] = []
    ch = FakeChannel()
    handler = make_channel_handler(
        channel=ch, agent_runner=_runner_that_records(created, delay=0.05), user_id=uid,
    )

    await asyncio.gather(handler(_inbound("one")), handler(_inbound("two")))

    sessions = {c["session_id_out"] for c in created}
    assert len(sessions) == 1, (
        f"{len(sessions)} sessions created for one chat — resolve→run→cache "
        "is not serialized"
    )


@pytest.mark.asyncio
async def test_two_chats_get_two_sessions(day_tables):
    """ANTI-VACUITY for the case above: a lock that serialises ACROSS chats
    would also produce one session, and that is the bug in the other
    direction — two people's conversations merged."""
    from app.agent.channels.shared import make_channel_handler

    uid = await _mk_user()
    created: List[dict] = []
    ch = FakeChannel()
    handler = make_channel_handler(
        channel=ch, agent_runner=_runner_that_records(created, delay=0.05), user_id=uid,
    )

    await asyncio.gather(
        handler(_inbound("a", CHAT_A)), handler(_inbound("b", CHAT_B)),
    )

    sessions = {c["session_id_out"] for c in created}
    assert len(sessions) == 2, "two chats were merged into one session"
