"""The fast-path media card was erased before it could be persisted (E5).

ws_chat resolves a "play X" message on its own fast path and hands the card to
the runner by setting `_agent_runner.tools._last_media` immediately before
`run()` (ws_chat.py:4121-4126). `_run_inner`'s transient-state reset then does
`self.tools._last_media = None` at agent_runner.py:1728 — at run START, after
ws_chat has already written — and `_save_messages` reads that same attribute at
:6759. So the card rendered live and was gone on every reload, for every typed
play. Present on main AND on the fleet image 7edaed3ab644.

D3 carries the card BY VALUE instead: `run(preset_media=...)` →
`_save_messages(media_meta_override=...)`, with the tool-set value winning and
the mailbox always drained. The reset at :1728 STAYS — it is correct, it is why
a card from two turns ago cannot be stapled onto this one.

The last test in this file is the one that survives a refactor: a behavioural
test alone passes the moment someone moves the reset. This repo's own rule —
read new guards for ORDER, not just for logic.

Local run (from backend/):
    RUN_MODE=platform PYTHONPATH=. ENVIRONMENT=test JWT_SECRET=x ENCRYPTION_KEY=y \
      pytest tests/test_fast_media_persists.py -q --tb=short -p no:cacheprovider
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

_BACKEND = Path(__file__).resolve().parent.parent
_SRC = (_BACKEND / "app" / "agent" / "agent_runner.py").read_text()

PRESET = {"type": "youtube", "video_id": "seed123", "title": "Kanye West - Stronger",
          "artwork": "https://i.ytimg.com/vi/seed123/hqdefault.jpg"}
TOOLSET = {"type": "youtube", "video_id": "tool456", "title": "A Tribe Called Quest - Can I Kick It"}


@pytest_asyncio.fixture
async def runner_db(tmp_path):
    """The tests/test_atomic_turn_counters.py:73-97 harness: a real
    AgentRunner, an AsyncMock ToolExecutor, three real tables."""
    from app.db.models import Conversation, Message, User

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/media.db")
    async with engine.begin() as conn:
        for table in (User.__table__, Conversation.__table__, Message.__table__):
            await conn.run_sync(lambda sc, t=table: t.create(sc, checkfirst=True))
    maker = async_sessionmaker(engine, expire_on_commit=False)

    user_id = "u-media"
    async with maker() as db:
        db.add(User(id=user_id, email="m@t.local", hashed_password="x", name="M"))
        conv = Conversation(user_id=user_id, title="t", message_count=0, total_tokens=0)
        db.add(conv)
        await db.commit()
        conv_id = conv.id

    yield maker, user_id, conv_id
    await engine.dispose()


def _runner(last_media=None):
    from app.agent.agent_runner import AgentRunner

    tools = AsyncMock()
    tools._last_media = last_media
    tools._last_pending_action = None
    tools.pending_attachments = []
    return AgentRunner(llm_service=AsyncMock(), tool_executor=tools), tools


async def _save(maker, user_id, conv_id, runner, **kw):
    async with maker() as db:
        out = await runner._save_messages(
            db=db, session_id=conv_id, user_id=user_id,
            user_message="play stronger", assistant_response="Putting it on.",
            tokens_input=10, tokens_output=5, model="gpt-4o-mini",
            processing_time_ms=5, **kw,
        )
        await db.commit()
    return out


async def _assistant_meta(maker, conv_id):
    from app.db.models import Message

    async with maker() as db:
        rows = (await db.execute(
            select(Message).where(Message.conversation_id == conv_id)
                           .where(Message.role == "assistant")
        )).scalars().all()
    assert len(rows) == 1, f"expected one assistant row, got {len(rows)}"
    raw = rows[0].metadata_json
    if raw is None:
        return None
    return json.loads(raw) if isinstance(raw, str) else raw


def _require_override():
    if "media_meta_override" not in inspect.signature(
        __import__("app.agent.agent_runner", fromlist=["AgentRunner"]).AgentRunner._save_messages
    ).parameters:
        pytest.skip("_save_messages(media_meta_override=) not landed yet — lane B1 (D3)")


# ══════════════════════════════════════════════════════════════════════
# Behaviour
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_the_fast_path_preset_is_persisted(runner_db):
    """THE regression. The card the user saw live must survive a reload."""
    _require_override()
    maker, uid, conv_id = runner_db
    runner, _ = _runner(last_media=None)   # the reset has already run

    await _save(maker, uid, conv_id, runner, media_meta_override=PRESET)

    meta = await _assistant_meta(maker, conv_id)
    assert meta and meta.get("media") == PRESET


@pytest.mark.asyncio
async def test_a_tool_play_during_the_run_still_persists(runner_db):
    """No regression on the path that already worked."""
    maker, uid, conv_id = runner_db
    runner, _ = _runner(last_media=TOOLSET)

    await _save(maker, uid, conv_id, runner)

    meta = await _assistant_meta(maker, conv_id)
    assert meta and meta.get("media") == TOOLSET


@pytest.mark.asyncio
async def test_a_tool_play_beats_the_preset_and_exactly_one_card_persists(runner_db):
    """Precedence asserted rather than inherited: the preset is what ws_chat
    GUESSED the user meant, the tool result is what actually played."""
    _require_override()
    maker, uid, conv_id = runner_db
    runner, _ = _runner(last_media=TOOLSET)

    await _save(maker, uid, conv_id, runner, media_meta_override=PRESET)

    meta = await _assistant_meta(maker, conv_id)
    assert meta.get("media") == TOOLSET
    assert isinstance(meta.get("media"), dict), "two cards were merged into a list"


@pytest.mark.asyncio
async def test_no_media_leaves_metadata_json_null_not_empty(runner_db):
    """`{}` on every assistant row is bloat on the hottest table in the
    product, and it makes 'this turn had no card' indistinguishable from
    'something wrote an empty one'."""
    maker, uid, conv_id = runner_db
    runner, _ = _runner(last_media=None)

    await _save(maker, uid, conv_id, runner)

    assert await _assistant_meta(maker, conv_id) is None


@pytest.mark.asyncio
async def test_the_mailbox_is_always_drained(runner_db):
    """A value left on the executor is stapled onto the NEXT turn's assistant
    row — the documented bug the voice route's explicit clear exists for."""
    maker, uid, conv_id = runner_db
    runner, tools = _runner(last_media=TOOLSET)

    await _save(maker, uid, conv_id, runner)

    assert tools._last_media is None


@pytest.mark.asyncio
async def test_the_mailbox_is_drained_even_when_the_override_wins(runner_db):
    """The case a naive override misses: the override short-circuits the read,
    the tool value is never consumed, and it lands on the next turn."""
    _require_override()
    maker, uid, conv_id = runner_db
    runner, tools = _runner(last_media=TOOLSET)

    await _save(maker, uid, conv_id, runner, media_meta_override=PRESET)

    assert tools._last_media is None


@pytest.mark.asyncio
async def test_save_messages_reports_what_it_persisted(runner_db):
    """D3/D6: `_save_messages` returns `persisted`, and the realtime echo
    (channel_echo.broadcast_channel_turn) builds its frames from it. A media
    key that is present in the row and absent from the return means the
    WhatsApp echo announces a turn with no card."""
    _require_override()
    maker, uid, conv_id = runner_db
    runner, _ = _runner(last_media=None)

    out = await _save(maker, uid, conv_id, runner, media_meta_override=PRESET)

    assert isinstance(out, dict), "_save_messages must return the persisted dict"
    for key in ("user_message_id", "asst_message_id", "day_chat_id", "channel"):
        assert key in out, f"persisted dict missing {key!r}"
    assert out.get("media") == PRESET


# ══════════════════════════════════════════════════════════════════════
# Structure — the pins that survive a refactor
# ══════════════════════════════════════════════════════════════════════


def test_run_inner_accepts_preset_media():
    from app.agent.agent_runner import AgentRunner

    params = inspect.signature(AgentRunner._run_inner).parameters
    if "preset_media" not in params:
        pytest.skip("_run_inner(preset_media=) not landed yet — lane B1 (D3)")
    assert params["preset_media"].default is None


def test_the_carrier_is_not_reachable_from_the_run_start_reset():
    """SOURCE-ORDER PROBE. The whole defect is that the carrier and the reset
    were the same attribute; a behavioural test passes the moment someone
    moves the reset, this one does not."""
    if "preset_media" not in _SRC:
        pytest.skip("preset_media carrier not landed yet — lane B1 (D3)")

    anchor = _SRC.index("self.tools._last_media = None")
    block = _SRC[anchor - 900:anchor + 200]
    assert "preset_media" not in block, (
        "the preset carrier appears inside the run-start transient reset — "
        "that is the E5 defect, reintroduced under a new name"
    )
    assert "self.tools._last_media = None" in _SRC, (
        "the run-start reset was DELETED. It is correct and load-bearing: "
        "without it a card from a previous turn is stapled onto this one."
    )


def test_ws_chat_still_presets_last_media():
    """ws_chat.py:1152's radio seed reads `_last_media` and degrades to
    'no_seed_track' without it, so the preset assignment must stay even once
    the card rides a kwarg. Nothing else in this repo says so."""
    ws = (_BACKEND / "app" / "api" / "ws_chat.py").read_text()
    assert "_agent_runner.tools._last_media = " in ws, (
        "ws_chat no longer presets tools._last_media — the radio-toggle seed "
        "silently loses its seed track"
    )


def test_the_voice_path_still_clears_last_media():
    """`/internal/play-media` runs no agent turn; left set, the voice play's
    card is stapled onto the next unrelated chat turn. Mirrors the existing
    pin at tests/test_voice_media_card.py:145 so a media fix cannot
    accidentally delete it."""
    api = (_BACKEND / "app" / "api" / "api_v1.py").read_text()
    assert "_last_media = None" in api


def test_play_media_is_not_parallel_safe():
    """The tool path's persistence depends on `_last_media` being written by
    the ONE tool call in flight. Nothing else in the repo states this."""
    import app.agent.agent_runner as ar

    safe = getattr(ar, "PARALLEL_SAFE_TOOLS", None)
    if safe is None:
        pytest.skip("PARALLEL_SAFE_TOOLS not defined on this build")
    assert "play_media" not in safe and "play_netflix" not in safe
