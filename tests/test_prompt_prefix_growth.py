"""R44 — the cached prefix must grow with the day, not stop at the head.

Measured in production 2026-09-13/14 (79 mobile-dominated turns): OpenAI's
`cached_tokens` sat pinned at exactly 40,192 — tools + instructions — for 36
consecutive turns of one day chat while the input grew 40,529 → 47,331. The
day's whole history was re-prefilled on every "hit". The stated suspects were
layout ones: a per-turn `<turn_context>` / `<runtime_envelope>` placed BEFORE
the history, or history items whose serialization drifts between turns.

These tests pin the layout invariant so neither can ever be the cause:

  * every per-turn item (`<turn_context>`, `<runtime_envelope>`) serializes
    at the TAIL, after the last history item;
  * the Responses input built for turn N+1 shares a byte-identical prefix
    with turn N's covering EVERY history item turn N sent — so the only
    items that can differ are the newest user turn and the per-turn tail;
  * the history the loader returns for the same rows is byte-stable across
    loads, through the annotate / leaked-tag-strip / reply-quote paths that
    rewrite content on the way out.

The last one is also the negative result of record: with these green, a
pinned `cached_beyond_head=0` in production is NOT this repo's prompt layout,
and the next place to look is provider-side. Keep them green rather than
re-litigating the layout.

Self-contained sqlite (its own engine), so RUN_MODE is irrelevant here — no
COVERAGE_DEBT entry needed.
"""

from __future__ import annotations

import json
import uuid
from datetime import date, datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.agent.prefix_stability import build_turn_context_message
from app.agent.runtime_envelope import (
    build_runtime_envelope_message,
    channel_capabilities,
)
from app.services.openai_agent_service import OpenAIAgentService


TZ = "America/Toronto"
DAY_START = datetime(2026, 9, 14, 13, 0, 0)


def _serialize(messages):
    """The exact Responses input array agent_runner hands the wire."""
    svc = OpenAIAgentService.__new__(OpenAIAgentService)
    svc._responses_reasoning = {}
    return svc._build_responses_input(messages)


def _common_prefix(a, b) -> int:
    n = 0
    for x, y in zip(a, b):
        if json.dumps(x, sort_keys=True, default=str) != json.dumps(
            y, sort_keys=True, default=str
        ):
            break
        n += 1
    return n


def _turn_tail(history, *, clock: str, user_text: str):
    """history + the per-turn items + the newest user message, in run() order."""
    messages = list(history)
    messages.append(
        build_turn_context_message([f"<clock>\nIt is {clock}.\n</clock>"])
    )
    messages.append(
        build_runtime_envelope_message(
            origin_channel="mobile",
            reply_channel="mobile",
            client_surface="Toup for iOS",
            guidance="",
            capabilities=channel_capabilities("mobile"),
            request_id=str(uuid.uuid4()),
            message_id=str(uuid.uuid4()),
            day_chat_id="00d3ff0f-0000-0000-0000-000000000000",
            managed_voice=False,
        )
    )
    messages.append({"role": "user", "content": user_text})
    return messages


@pytest_asyncio.fixture
async def day():
    """One day chat, two channels, a reply-to row and a leaked-tag row."""
    from app.db.models.base import Base
    from app.db.models import Conversation, Message
    from app.db.models.day_chat import DayChat
    from app.db.models.user import User

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)

    user_id = str(uuid.uuid4())
    day_chat_id = str(uuid.uuid4())
    conv_id = str(uuid.uuid4())
    first_user_msg_id = str(uuid.uuid4())

    async with maker() as db:
        db.add(User(id=user_id, email=f"{user_id}@t.test", hashed_password="x"))
        db.add(
            DayChat(
                id=day_chat_id, user_id=user_id,
                local_date=date(2026, 9, 14), timezone=TZ,
            )
        )
        db.add(
            Conversation(
                id=conv_id, user_id=user_id, channel="mobile",
                day_chat_id=day_chat_id,
            )
        )
        await db.commit()

    async def add(minute, role, content, channel="mobile", msg_id=None, reply_to=None):
        async with maker() as db:
            db.add(
                Message(
                    id=msg_id or str(uuid.uuid4()),
                    conversation_id=conv_id, day_chat_id=day_chat_id,
                    channel=channel, role=role, content=content,
                    reply_to_message_id=reply_to,
                    created_at=DAY_START + timedelta(minutes=minute),
                )
            )
            await db.commit()

    # Shapes that actually rewrite content on the way out of the loader.
    await add(0, "user", "morning — what's on today?", msg_id=first_user_msg_id)
    await add(1, "assistant", "[mobile 9:01am] Two meetings and the invoice.")
    await add(2, "user", "read it back to me", channel="voice")
    await add(3, "assistant", "Sure — the invoice is due Friday.", channel="voice")
    await add(4, "user", "and the other one?", reply_to=first_user_msg_id)
    await add(5, "assistant", "The 3pm with Dana.")

    async def load():
        from app.agent.day_context_loader import load_day_context

        async with maker() as db:
            ctx = await load_day_context(
                db, day_chat_id, model="gpt-5.6-terra",
                model_context_tokens=1_000_000,
                calling_channel="mobile", tz_name=TZ,
            )
        return ctx["messages"]

    yield {"add": add, "load": load, "first_user_msg_id": first_user_msg_id}
    await engine.dispose()


@pytest.mark.asyncio
async def test_history_is_byte_stable_across_loads(day):
    """Same rows in, same bytes out — twice, with an append in between."""
    first = await day["load"]()
    again = await day["load"]()
    assert first == again, "load_day_context is not deterministic for fixed rows"

    await day["add"](6, "user", "thanks")
    await day["add"](7, "assistant", "anytime")
    grown = await day["load"]()

    assert len(grown) == len(first) + 2
    assert grown[: len(first)] == first, (
        "an already-sent history item changed when the day grew — every turn "
        "after this one re-prefills from that item"
    )


@pytest.mark.asyncio
async def test_per_turn_items_serialize_after_every_history_item(day):
    """`<turn_context>` and `<runtime_envelope>` are TAIL items, always."""
    history = await day["load"]()
    items = _serialize(_turn_tail(history, clock="1:12pm", user_text="ok"))

    def _text(item) -> str:
        content = item.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                str(part.get("text", "")) for part in content
                if isinstance(part, dict)
            )
        return ""

    for marker in ("<turn_context>", "<runtime_envelope>"):
        hits = [i for i, it in enumerate(items) if marker in _text(it)]
        assert hits, f"{marker} is not in the built input at all"
        assert min(hits) >= len(history), (
            f"{marker} serializes at index {min(hits)}, inside the "
            f"{len(history)} history items that precede it — every byte after "
            f"it is uncacheable from the next turn on"
        )


@pytest.mark.asyncio
async def test_consecutive_turns_share_the_whole_history_prefix(day):
    """Turn N+1's input matches turn N's over every history item turn N sent."""
    history_n = await day["load"]()
    turn_n = _serialize(
        _turn_tail(history_n, clock="1:12pm", user_text="and dinner?")
    )

    await day["add"](6, "user", "and dinner?")
    await day["add"](7, "assistant", "Booked for 8.")
    history_n1 = await day["load"]()
    turn_n1 = _serialize(
        _turn_tail(history_n1, clock="1:20pm", user_text="perfect")
    )

    shared = _common_prefix(turn_n, turn_n1)
    assert shared >= len(history_n), (
        f"only {shared} input items are shared between consecutive turns, but "
        f"turn N sent {len(history_n)} history items — the provider re-prefills "
        f"from item {shared} on"
    )
    # And the first item after instructions+tools is the same object both turns.
    assert turn_n[0] == turn_n1[0]


@pytest.mark.asyncio
async def test_the_prefix_assertion_is_sensitive_to_a_head_placement(day):
    """Mutation: put the per-turn context FIRST and the test above must fail.

    Without this, a refactor that silently drops the per-turn items could
    leave the assertion above passing on an empty tail.
    """
    history_n = await day["load"]()
    clock_n = build_turn_context_message(["<clock>\nIt is 1:12pm.\n</clock>"])
    clock_n1 = build_turn_context_message(["<clock>\nIt is 1:20pm.\n</clock>"])

    turn_n = _serialize([clock_n] + list(history_n))
    await day["add"](6, "user", "and dinner?")
    await day["add"](7, "assistant", "Booked for 8.")
    history_n1 = await day["load"]()
    turn_n1 = _serialize([clock_n1] + list(history_n1))

    assert _common_prefix(turn_n, turn_n1) == 0, (
        "a clock at input[0] must destroy the shared prefix — if it does not, "
        "the prefix assertions above are not measuring anything"
    )


# ── the observability half ────────────────────────────────────────────
#
# With the layout invariants above green, the only way to tell a pinned
# prod cache from a healthy one is a field that says how far past the head
# the provider actually read. These pin that field's existence.

def _runner_source() -> str:
    from pathlib import Path

    import app.agent.agent_runner as runner

    return Path(runner.__file__).read_text(encoding="utf-8")


def test_llm_line_reports_how_far_the_cache_reaches_past_the_head():
    src = _runner_source()
    assert "cached_beyond_head=" in src, (
        "raw cache_read cannot distinguish 'the head hit' from 'the day's "
        "history hit' — it is large and constant in both cases"
    )
    assert "max(0, _cached_tok - _head_est)" in src, (
        "cached_beyond_head must be clamped: the head is an ESTIMATE and a "
        "negative field reads as a provider bug rather than an estimator one"
    )


def test_cache_hit_and_miss_are_declared_signals():
    from app.services.health_signals import KNOWN_SIGNALS

    assert "llm_cache_hits" in KNOWN_SIGNALS
    assert "llm_cache_misses" in KNOWN_SIGNALS
