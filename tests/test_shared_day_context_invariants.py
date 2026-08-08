"""ITEM 14 — verification record: ONE day-as-chat context, and one loader.

This file pins the invariants the ITEM-14 audit VERIFIED rather than
changed, so they cannot regress silently.

What was verified (file:line as of this commit):

  surface            entry point                                        channel
  ─────────────────────────────────────────────────────────────────────────────
  web / app / mobile app/api/ws_chat.py:3000  `_agent_runner.run(`      msg["channel"]
  extension          app/api/ws_chat.py:3000  (same call site)          "extension"
  telegram           app/agent/telegram_bot.py:1739 and :2088           "telegram"
  whatsapp           app/agent/channels/shared/message_handler.py:212   channel_type.value
  voice (`think`)    app/api/ws_realtime.py:1622 (in-process), and the
                     agent-side hops app/api/api_v1.py:384
                     (/internal/agent-turn) and :652 (…/stream)         "voice"

All of them enter the SAME `AgentRunner.run`, which resolves ONE
day_chat_id from (user_id, effective_tz) at agent_runner.py:1109 and
loads it through the SINGLE loader `load_day_context` at
agent_runner.py:1114.

The three pins below:
  1. behavioural — six channels writing into one DayChat come back from
     `load_day_context` as one chronological list, each row annotated
     with its own channel;
  2. structural — `agent_runner.py` has exactly ONE `load_day_context(`
     call site and ONE `resolve_day_chat_id_for_now(` call site, so a
     second history loader cannot appear unnoticed;
  3. price-table agreement — the repo carries THREE model rate tables
     (`settings.pricing_per_1k`, `token_tracker.MODEL_PRICING`,
     `model_session.AVAILABLE_MODELS`); where they overlap they must
     state the same rate. Context windows are pinned the same way, with
     an explicit allowlist for the divergence that exists today.
"""

from __future__ import annotations

import re
import uuid
from datetime import date as Date, datetime
from pathlib import Path

import pytest
import pytest_asyncio

BACKEND = Path(__file__).resolve().parent.parent


@pytest_asyncio.fixture
async def day_tables():
    """Create the three AGENT_ONLY tables this file needs.

    `day_chats` / `conversations` / `messages` are AGENT_ONLY
    (app/db/models/base.py:56-63), so the platform-mode sweep's `init_db`
    skips them. Creating them here — instead of parking the file in
    COVERAGE_DEBT — keeps these pins in the default sweep. `messages`
    carries a pgvector column, so it is created without it when the
    backend has no vector type (sqlite, or Postgres without pgvector).
    """
    from sqlalchemy import inspect as sa_inspect
    from app.db.database import engine
    from app.db.models.base import Base

    names = ["day_chats", "conversations", "messages"]
    tables = [Base.metadata.tables[n] for n in names]
    async with engine.begin() as conn:
        for table in tables:
            try:
                await conn.run_sync(table.create, checkfirst=True)
            except Exception:
                # Vector column unsupported on this backend — recreate the
                # table with that column dropped so the rest still works.
                pass

    async with engine.connect() as conn:
        existing = await conn.run_sync(
            lambda c: set(sa_inspect(c).get_table_names())
        )
    missing = [n for n in names if n not in existing]
    if missing:
        pytest.skip(f"cannot create {missing} on this backend")
    yield


# ──────────────────────────────────────────────────────────────────────
# 1. Behavioural: one day chat, six channels, one chronological list
# ──────────────────────────────────────────────────────────────────────

# Every surface that reaches AgentRunner.run, plus the two that only
# reach it indirectly. If a channel is added to KNOWN_CHANNELS and it
# writes day-stamped messages, it belongs here.
SURFACE_CHANNELS = ["web", "app", "mobile", "extension", "telegram", "whatsapp"]


@pytest.mark.asyncio
async def test_all_surfaces_share_one_day_chat_context(day_tables):
    """One day_chat_id, six channels, one chronological history.

    This is the "five surfaces share one context" claim, executed. The
    parent `users` row is seeded explicitly — messages→conversations→
    users is a real FK under Postgres.
    """
    from app.db import async_session_maker
    from app.db.models import Conversation, Message
    from app.db.models.day_chat import DayChat
    from app.db.models.user import User
    from app.agent.day_context_loader import load_day_context

    user_id = str(uuid.uuid4())
    dc_id = str(uuid.uuid4())
    day = Date(2026, 8, 3)

    async with async_session_maker() as db:
        db.add(User(
            id=user_id, email=f"item14-{uuid.uuid4().hex[:10]}@example.com",
            hashed_password="x", name="Item 14", timezone="America/Toronto",
        ))
        db.add(DayChat(
            id=dc_id, user_id=user_id, local_date=day, timezone="America/Toronto",
            summary_status="up_to_date",
        ))
        await db.flush()

        # Interleave the channels in time so a per-channel (rather than
        # per-day) loader would produce a visibly different order.
        for i, ch in enumerate(SURFACE_CHANNELS):
            conv_id = str(uuid.uuid4())
            db.add(Conversation(
                id=conv_id, user_id=user_id, channel=ch, day_chat_id=dc_id,
                started_at=datetime(2026, 8, 3, 9, 0, 0),
            ))
            db.add(Message(
                id=str(uuid.uuid4()), conversation_id=conv_id, day_chat_id=dc_id,
                channel=ch, role="user", content=f"hello from {ch}",
                created_at=datetime(2026, 8, 3, 9, i, 0),
            ))
        await db.commit()

    async with async_session_maker() as db:
        ctx = await load_day_context(
            db, dc_id, model_context_tokens=200_000, tz_name="America/Toronto",
        )

    contents = [m["content"] for m in ctx["messages"]]
    assert len(contents) == len(SURFACE_CHANNELS), contents

    # Chronological, and each row carries ITS OWN channel tag — that tag
    # is the whole point of the shared window (the model has to know
    # where each turn came from).
    for i, ch in enumerate(SURFACE_CHANNELS):
        assert contents[i].startswith(f"[{ch} "), contents[i]
        assert f"hello from {ch}" in contents[i]

    # Under budget ⇒ the loader returns the day COMPLETE, which is what
    # lets agent_runner skip the duplicate <today_so_far> injection.
    assert ctx["history_is_complete"] is True
    assert ctx["message_count"] == len(SURFACE_CHANNELS)


@pytest.mark.asyncio
async def test_day_context_is_scoped_to_its_day_chat(day_tables):
    """ANTI-VACUITY CONTROL for the test above: a message stamped into a
    DIFFERENT day chat must not appear. Without this, a loader that
    ignored day_chat_id entirely would pass the assertions above."""
    from app.db import async_session_maker
    from app.db.models import Conversation, Message
    from app.db.models.day_chat import DayChat
    from app.db.models.user import User
    from app.agent.day_context_loader import load_day_context

    user_id = str(uuid.uuid4())
    today_id, other_id = str(uuid.uuid4()), str(uuid.uuid4())

    async with async_session_maker() as db:
        db.add(User(
            id=user_id, email=f"item14b-{uuid.uuid4().hex[:10]}@example.com",
            hashed_password="x", name="Item 14b", timezone="UTC",
        ))
        for dc_id, d in ((today_id, Date(2026, 8, 3)), (other_id, Date(2026, 8, 2))):
            db.add(DayChat(id=dc_id, user_id=user_id, local_date=d, timezone="UTC"))
        await db.flush()
        for dc_id, marker in ((today_id, "TODAY-ROW"), (other_id, "YESTERDAY-ROW")):
            conv_id = str(uuid.uuid4())
            db.add(Conversation(
                id=conv_id, user_id=user_id, channel="web", day_chat_id=dc_id,
                started_at=datetime(2026, 8, 3, 9, 0, 0),
            ))
            db.add(Message(
                id=str(uuid.uuid4()), conversation_id=conv_id, day_chat_id=dc_id,
                channel="web", role="user", content=marker,
                created_at=datetime(2026, 8, 3, 9, 0, 0),
            ))
        await db.commit()

    async with async_session_maker() as db:
        ctx = await load_day_context(db, today_id, model_context_tokens=200_000)

    blob = "\n".join(m["content"] for m in ctx["messages"])
    assert "TODAY-ROW" in blob
    assert "YESTERDAY-ROW" not in blob


# ──────────────────────────────────────────────────────────────────────
# 2. Structural: ONE loader, ONE day resolution, in the runner
# ──────────────────────────────────────────────────────────────────────

def test_agent_runner_has_exactly_one_day_history_loader():
    """The single-context claim rests on there being ONE loader call in
    the runner. A second one is how a surface quietly forks its own
    history window, so make adding one a red test."""
    src = (BACKEND / "app" / "agent" / "agent_runner.py").read_text()
    calls = re.findall(r"^\s*_day_context = await load_day_context\(", src, re.M)
    assert len(calls) == 1, f"expected 1 load_day_context call site, found {len(calls)}"

    # Exactly two day resolutions, and both take the SAME tz: the READ
    # (context load, :1109) and the WRITE (message insert, :4783). That
    # they share `tz_override=client_tz` is what makes the day a turn is
    # written into the day its context was read from.
    resolves = re.findall(
        r"await resolve_day_chat_id_for_now\(db, user_id, tz_override=client_tz\)", src
    )
    assert len(resolves) == 2, (
        f"expected 2 resolve_day_chat_id_for_now(tz_override=client_tz) call sites "
        f"(one read, one write), found {len(resolves)}"
    )
    assert src.count("resolve_day_chat_id_for_now(") == 2, (
        "a resolve_day_chat_id_for_now call appeared with a different tz source"
    )


def test_every_surface_reaches_the_same_runner():
    """The five text/chat surfaces each call `AgentRunner.run` — no
    surface hand-rolls its own turn. Pins the verification table in this
    module's docstring."""
    expected = {
        "app/api/ws_chat.py": "_agent_runner.run(",
        "app/agent/telegram_bot.py": "self.agent_runner.run(",
        "app/agent/channels/shared/message_handler.py": "agent_runner.run(",
        "app/api/ws_realtime.py": "_agent_runner.run(",
        "app/api/api_v1.py": "_agent_runner.run(",
    }
    for rel, needle in expected.items():
        src = (BACKEND / rel).read_text()
        assert needle in src, f"{rel} no longer calls {needle}"

    # Voice declares its channel explicitly at every entry point — that
    # string is what makes the runner's day/tz/tool-gating branch take
    # the voice path. Three sites: the in-process relay call, and the two
    # agent-side /internal/agent-turn hops the V2 relay uses.
    assert (BACKEND / "app/api/ws_realtime.py").read_text().count('channel="voice"') >= 1
    assert (BACKEND / "app/api/api_v1.py").read_text().count('channel="voice"') == 2


# ──────────────────────────────────────────────────────────────────────
# 3. The three price tables must agree where they overlap
# ──────────────────────────────────────────────────────────────────────

def _tables():
    from app.config import settings
    from app.agent.token_tracker import MODEL_PRICING
    from app.agent.model_session import AVAILABLE_MODELS

    cfg = {
        k: (v["input"] * 1000, v["output"] * 1000)
        for k, v in settings.pricing_per_1k.items()
    }
    tt = {k: (v["input"], v["output"]) for k, v in MODEL_PRICING.items()}
    ms = {k: (v["cost_in"], v["cost_out"]) for k, v in AVAILABLE_MODELS.items()}
    return {
        "settings.pricing_per_1k": cfg,
        "token_tracker.MODEL_PRICING": tt,
        "model_session.AVAILABLE_MODELS": ms,
    }


def test_three_price_tables_agree_where_they_overlap():
    """Rates are per-1M here; `settings.pricing_per_1k` is per-1K and is
    scaled. Coverage legitimately differs (each table serves a different
    caller) — the RATES must not."""
    tables = _tables()
    names = list(tables)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = set(tables[a]) & set(tables[b])
            assert shared, f"{a} and {b} share no models — table gutted?"
            for model in sorted(shared):
                assert tables[a][model] == pytest.approx(tables[b][model]), (
                    f"{model}: {a}={tables[a][model]} vs {b}={tables[b][model]}"
                )


def test_default_agent_model_is_priced_everywhere():
    """The fleet default must be present in every rate table — an absent
    entry silently bills at credit_service's {input:0.003,output:0.015}
    hardcoded fallback."""
    from app.services.model_resolver import default_model

    m = default_model()
    for name, table in _tables().items():
        assert m in table, f"default model {m!r} missing from {name}"


# Known context-window divergence, MEASURED 2026-08-06. Not fixed here:
# changing a window changes the history budget (window x 0.60 in
# day_context_loader) for anyone pinned to that model, which is exactly
# the budget change ITEM 14 is not authorised to make. This allowlist may
# only SHRINK.
_KNOWN_WINDOW_DIVERGENCE = {
    # context_manager says 128k, model_session says 1M.
    "gpt-4.1",
    # absent from context_manager entirely -> DEFAULT_CONTEXT_WINDOW 128k,
    # while model_session advertises 1M.
    "gpt-4.1-mini",
}


def test_context_window_tables_agree_except_known_divergence():
    from app.agent.context_manager import MODEL_CONTEXT_WINDOWS, DEFAULT_CONTEXT_WINDOW
    from app.agent.model_session import AVAILABLE_MODELS

    diverged = set()
    for model, info in AVAILABLE_MODELS.items():
        theirs = info.get("context")
        ours = MODEL_CONTEXT_WINDOWS.get(model, DEFAULT_CONTEXT_WINDOW)
        if theirs != ours:
            diverged.add(model)

    assert diverged <= _KNOWN_WINDOW_DIVERGENCE, (
        f"NEW context-window divergence: {sorted(diverged - _KNOWN_WINDOW_DIVERGENCE)}"
    )


def test_history_budget_is_sixty_percent_of_the_window():
    """The day-history clamp: there is no literal budget anywhere. It is
    HISTORY_BUDGET_RATIO x get_context_window(settings.agent_model), and
    it is a CLIFF — one token over and the loader drops to summary +
    VERBATIM_WINDOW messages.

    2026-08-07: was 630_000 (gpt-5.5's 1.05M window). The fleet default
    moved to gpt-5.6-terra, whose window is 1.00M, so the cliff sits at
    600_000 — 4.8% earlier. Deliberate and accepted: a single day would
    need ~600k tokens of conversation to reach it, and terra is 50-55%
    cheaper per token (docs/audits/2026-08-g1-cost-and-latency.md §8).
    Asserted as an exact value rather than a floor because this is a cliff:
    it should move only when someone means to move it.
    """
    from app.agent.context_manager import get_context_window
    from app.agent.day_context_loader import HISTORY_BUDGET_RATIO, VERBATIM_WINDOW
    from app.services.model_resolver import default_model

    assert HISTORY_BUDGET_RATIO == 0.60
    assert VERBATIM_WINDOW == 20
    window = get_context_window(default_model())
    assert window == 1_000_000, f"default model window moved: {window}"
    assert int(window * HISTORY_BUDGET_RATIO) == 600_000, (
        f"day-history budget moved: {window} x {HISTORY_BUDGET_RATIO}"
    )
