"""The realtime frame's exact key set, and the channel stamp that was never written.

Two halves of the same defect (E7 + D7):

REALTIME. A turn that arrives over WhatsApp produces no frame at all, so an
open app shows nothing until it refetches. D6's echo fixes that, and its KEY
SET is a compatibility surface for a client already in the App Store:

  * mobile accepts a frame when `!channel || channel === MY_CHANNEL ||
    msg.type === 'message'` (api.ts:3189) — so a `message` frame arrives
    whatever channel it names;
  * mobile decides whether a landed row settles the phone's own in-flight
    turn from `THIS_TURNS_ANSWER_CHANNELS = ['mobile','app']` plus null
    (api.ts:2703) — so a channel-LESS assistant frame would be read as the
    answer to whatever the user just typed on the phone. The `channel` key is
    therefore MANDATORY, and that is why this file asserts key-set EQUALITY
    rather than containment: an extra key is a client-visible change, and a
    missing one is a settle bug.

STAMP. `Message.channel` is NULL for every WS-presaved user row
(ws_chat.py:3705-3714 builds `_msg_kwargs` with no `channel=`, four lines
after `_ensure_presave_conversation` has already resolved one), while
`_save_messages` at agent_runner.py:6740 does pass it. So the origin of a user
row is unrecoverable, and both REST serializers answer with the CONVERSATION's
channel instead. D7 stamps the row and inverts the serializers to
Message-first — with the Conversation fallback and the 'web' default kept,
because the entire existing corpus is NULL and a bare switch would re-badge
every historic row on the released build.

Local run (from backend/):
    RUN_MODE=platform PYTHONPATH=. ENVIRONMENT=test JWT_SECRET=x ENCRYPTION_KEY=y \
      pytest tests/test_channel_turn_realtime_and_stamp.py -q --tb=short -p no:cacheprovider
"""
from __future__ import annotations

import json
import uuid
from datetime import date as Date, datetime
from pathlib import Path

import pytest
import pytest_asyncio

_BACKEND = Path(__file__).resolve().parent.parent

# D6. Exactly these, on every frame.
FRAME_KEYS = {"type", "id", "role", "content", "created_at", "channel",
              "day_chat_id", "conversation_id"}
# Only these may additionally appear, and only on the assistant frame.
ASSISTANT_EXTRAS = {"media", "tool_events", "attachments"}


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


# ══════════════════════════════════════════════════════════════════════
# 1. The frame
# ══════════════════════════════════════════════════════════════════════


def _echo_mod():
    return pytest.importorskip(
        "app.agent.channel_echo",
        reason="channel_echo not landed yet — lane B2 (D6)",
    )


@pytest.fixture
def frames(monkeypatch):
    import app.api.ws_chat as ws_chat

    got: list[dict] = []

    async def recorder(user_id, event, exclude=None):
        got.append(event)
        return 1

    monkeypatch.setattr(ws_chat, "broadcast_to_user", recorder)
    m = _echo_mod()
    if hasattr(m, "broadcast_to_user"):
        monkeypatch.setattr(m, "broadcast_to_user", recorder)
    return got


PERSISTED = {
    "user_message_id": "m-user-1",
    "user_created_at": datetime(2026, 9, 14, 1, 40, 21),
    "asst_message_id": "m-asst-1",
    "asst_created_at": datetime(2026, 9, 14, 1, 40, 29),
    "day_chat_id": "dc-1",
    "conversation_id": "conv-1",
    "channel": "whatsapp",
}


@pytest.mark.asyncio
async def test_the_user_frames_key_set_is_exactly_the_contract(frames):
    """EQUALITY, not containment. Every extra key is something a shipped
    client will see and did not ask for."""
    await _echo_mod().broadcast_channel_turn(
        "u-1", origin_channel="whatsapp", session_id="conv-1",
        persisted=PERSISTED, user_text="play stronger", assistant_text="Putting it on.",
    )
    user_frame = frames[0]
    assert set(user_frame) == FRAME_KEYS, (
        f"user frame keys {sorted(set(user_frame) ^ FRAME_KEYS)} differ from "
        "the D6 contract"
    )


@pytest.mark.asyncio
async def test_the_assistant_frame_adds_only_the_three_payload_keys(frames):
    await _echo_mod().broadcast_channel_turn(
        "u-1", origin_channel="whatsapp", session_id="conv-1",
        persisted=dict(PERSISTED, media={"type": "youtube", "video_id": "x"},
                       tool_events=[{"name": "play_media"}],
                       attachments=[{"id": "a1"}]),
        user_text="play stronger", assistant_text="Putting it on.",
    )
    extra = set(frames[1]) - FRAME_KEYS
    assert extra <= ASSISTANT_EXTRAS, f"unexpected assistant-frame keys: {sorted(extra)}"


@pytest.mark.asyncio
async def test_the_channel_key_is_present_and_is_the_origin(frames):
    """Without it the released client treats the assistant row as the answer
    to the phone's own pending turn — see api.ts:2703's allowlist."""
    await _echo_mod().broadcast_channel_turn(
        "u-1", origin_channel="whatsapp", session_id="conv-1",
        persisted=PERSISTED, user_text="hi", assistant_text="hey",
    )
    for f in frames:
        assert "channel" in f and f["channel"] == "whatsapp"
        assert f["channel"] not in ("mobile", "app", None)


@pytest.mark.asyncio
async def test_created_at_is_a_string_the_client_can_parse(frames):
    """`new Date(undefined)` sorts a row to the top of the thread; a datetime
    object is not JSON-serialisable at all and the send would throw inside
    the fan-out."""
    await _echo_mod().broadcast_channel_turn(
        "u-1", origin_channel="whatsapp", session_id="conv-1",
        persisted=PERSISTED, user_text="hi", assistant_text="hey",
    )
    for f in frames:
        assert isinstance(f["created_at"], str) and f["created_at"]
        json.dumps(f)   # the fan-out serialises it; a raw datetime raises here


# ══════════════════════════════════════════════════════════════════════
# 2. The presave stamp
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.xfail(
    "channel=" not in (_BACKEND / "app" / "api" / "ws_chat.py").read_text()[
        (_BACKEND / "app" / "api" / "ws_chat.py").read_text().index("_msg_kwargs: dict = dict("):
    ][:600],
    reason="RED until lane B2 stamps Message.channel on the presave (D7)",
    strict=True,
)
def test_the_ws_presave_stamps_message_channel():
    """SOURCE PROBE. The insert sits inside a 3,000-line websocket handler
    that no test can reach, and the value is already in scope four lines
    above (`_ensure_presave_conversation` resolves it for the Conversation).
    A one-line omission, not a design gap."""
    src = (_BACKEND / "app" / "api" / "ws_chat.py").read_text()
    anchor = src.index("_msg_kwargs: dict = dict(")
    block = src[anchor:anchor + 600]
    assert "channel=" in block, (
        "ws_chat's presave still writes Message.channel NULL — the origin of "
        "every WS user row is unrecoverable, and the day index answers with "
        "the CONVERSATION's channel instead"
    )


def test_the_reply_to_retry_path_writes_the_same_channel():
    """There are TWO inserts: the normal one and the retry after a
    'column reply_to_message_id does not exist' commit failure. One
    `_msg_kwargs` covering both is what makes them agree — a second literal
    dict is how one of them silently loses the stamp."""
    src = (_BACKEND / "app" / "api" / "ws_chat.py").read_text()
    assert src.count("_msg_kwargs: dict = dict(") == 1, (
        "a second presave kwargs literal appeared — both inserts must build "
        "from ONE dict or they will drift"
    )


# ══════════════════════════════════════════════════════════════════════
# 3. The serializers
# ══════════════════════════════════════════════════════════════════════


async def _seed_day(*, msg_channel, conv_channel, day=Date(2026, 9, 14)):
    from app.db.database import async_session_maker
    from app.db.models import Conversation, Message
    from app.db.models.day_chat import DayChat
    from app.db.models.user import User

    uid, dc_id, conv_id, mid = (str(uuid.uuid4()) for _ in range(4))
    async with async_session_maker() as db:
        db.add(User(id=uid, email=f"stamp-{uuid.uuid4().hex[:10]}@example.com",
                    hashed_password="x", name="Stamp", timezone="UTC"))
        db.add(DayChat(id=dc_id, user_id=uid, local_date=day, timezone="UTC"))
        await db.flush()
        db.add(Conversation(id=conv_id, user_id=uid, channel=conv_channel,
                            day_chat_id=dc_id,
                            started_at=datetime(day.year, day.month, day.day, 9, 0)))
        db.add(Message(id=mid, conversation_id=conv_id, day_chat_id=dc_id,
                       channel=msg_channel, role="user", content="hi",
                       created_at=datetime(day.year, day.month, day.day, 9, 1)))
        await db.commit()
    return uid, mid, day


async def _serialize(uid, day):
    """Call the route function directly (the tests/test_day_chats_api.py
    convention — endpoint logic, not an HTTP server)."""
    from sqlalchemy import select
    import app.api.day_chats as api
    from app.db.database import async_session_maker
    from app.db.models.user import User

    real_proxy = api._get_agent_proxy_info

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
        api._get_agent_proxy_info = real_proxy
    return json.loads(bytes(resp.body).decode())


@pytest.mark.asyncio
async def test_a_stamped_row_reports_its_own_channel(day_tables):
    """The point of the inversion: a WhatsApp row inside a conversation the
    day index resolved as something else still says 'whatsapp'."""
    uid, mid, day = await _seed_day(msg_channel="whatsapp", conv_channel="web")
    rows = await _serialize(uid, day)
    assert rows, "the day serialized empty — the fixture never reached the route"
    got = rows[0]["channel"]
    if got == "web":
        pytest.xfail("serializer still reads Conversation.channel — lane B3 (D7)")
    assert got == "whatsapp"


@pytest.mark.asyncio
async def test_a_null_stamped_row_still_falls_back_to_the_conversation(day_tables):
    """The released build's compatibility gate. Message.channel is NULL for
    the ENTIRE existing corpus; without this fallback every historic row
    re-badges on build 123."""
    uid, mid, day = await _seed_day(msg_channel=None, conv_channel="telegram")
    rows = await _serialize(uid, day)
    assert rows and rows[0]["channel"] == "telegram"


def test_the_last_resort_default_is_still_the_literal_web():
    """`Conversation.channel` is NOT NULL with a column default of 'api', so
    the third rung of D7's ladder is not reachable from a seeded row — which
    is precisely why it needs a source pin rather than a behavioural one.

    It matters because the released client renders whatever the key says:
    ChatScreen.tsx:386 maps it straight onto a badge, and `CHANNEL_BADGES`
    deliberately has no entry for 'web'/'mobile'/'app'. Changing the default
    to anything else draws a badge on every row that has nothing to say.
    """
    src = (_BACKEND / "app" / "api" / "day_chats.py").read_text()
    assert src.count('"web"') >= 1 and "_row_channel" in src, (
        "the day-messages serializers no longer default to 'web' — every row "
        "with no resolvable channel re-badges on the released build"
    )


@pytest.mark.asyncio
async def test_the_hidden_day_channel_filter_is_unchanged(day_tables):
    """The inversion must not widen what the day shows. An `automation` row
    belongs to its own thread (CONTRACTS-R31 §4.1) and appearing in the main
    chat is the 26-August regression."""
    from app.api.day_chats import HIDDEN_DAY_CHANNELS

    hidden = next(iter(HIDDEN_DAY_CHANNELS))
    uid, mid, day = await _seed_day(msg_channel="whatsapp", conv_channel=hidden)
    rows = await _serialize(uid, day)
    assert rows == [], (
        f"a conversation on the hidden channel {hidden!r} became visible — "
        "Message.channel must not be allowed to override the day filter"
    )


def test_the_ws_presave_never_stamps_resolve_channels_default():
    """SOURCE PROBE (same reason as above: the handler is unreachable by a
    test). `resolve_channel` answers 'unknown' when nothing is known; stored
    as a row's origin it would be served verbatim where every serializer's
    Conversation → 'web' fallback used to answer. The presave converts it
    to NULL before the insert."""
    src = (_BACKEND / "app" / "api" / "ws_chat.py").read_text()
    resolve = src.index('site="ws_presave_message"')
    kwargs = src.index("_msg_kwargs: dict = dict(")
    between = src[resolve:kwargs]
    assert 'if _presave_channel == "unknown":' in between and "_presave_channel = None" in between, (
        "the presave stores resolve_channel's 'unknown' default as a row's origin again"
    )
    assert "channel=_presave_channel," in src[kwargs:kwargs + 600]
