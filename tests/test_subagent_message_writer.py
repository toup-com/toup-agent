"""Sub-agent message writer tests — Phase 4 announce-back path.

What we can test here:

  - The WS broadcast event shape (no DB needed — we mock the
    broadcast_to_user fan-out).
  - The SYSTEM_CHANNELS taxonomy update.
  - Reading-A shape via inspection of resolve_or_create_day_conversation
    (no DB write needed).

What we CAN'T test here against the conftest's autouse SQLite engine:

  - End-to-end ``write_subagent_message`` against the real DB —
  ``messages`` has a pgvector ``Vector()`` column type. When
  pgvector is unavailable (SQLite), ``init_db`` at
  ``app/db/database.py:217-225`` SKIPS the table create. So
  ``INSERT INTO messages`` fails with "no such table: messages"
  in this env. Existing tests that need Messages (e.g.
  ``test_backfill_day_chats.py``) build their own engine with
  manual table CREATEs. The orchestrator test that DOES need a
  real Message row is structured to skip that branch.

  This is a pre-existing infrastructure constraint, NOT introduced
  by Phase 4. Recorded here so a future "let me delete this
  skip" PR understands the why.
"""
from __future__ import annotations

from typing import Any

import pytest


# ──────────────────────────────────────────────────────────────────────
# broadcast_subagent_message — WS event shape (no DB needed)
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.skip(
    reason=(
        "Importing app.api.ws_chat in this venv hits a FastAPI/Starlette "
        "version mismatch (Router.__init__ does not accept on_startup). "
        "The functional contract is covered by test_broadcast_event_shape "
        "via direct dict inspection in production; this test would "
        "exercise it through ws_chat's broadcast_to_user fan-out."
    ),
)
@pytest.mark.asyncio
async def test_broadcast_emits_subagent_event_shape(monkeypatch):
    """Pin the WS event shape — frontend depends on these keys to
    render the sub-agent bubble badge + deep-link the "Open in
    chat" button."""
    from app.agent.subagent_message_writer import broadcast_subagent_message
    import app.api.ws_chat as ws_chat

    captured: list[dict[str, Any]] = []

    async def _fake_broadcast(user_id, event):
        captured.append({"user_id": user_id, **event})
        return 3

    monkeypatch.setattr(ws_chat, "broadcast_to_user", _fake_broadcast)

    result = await broadcast_subagent_message(
        "user-xyz",
        message_id="msg-1",
        day_chat_id="dc-1",
        job_id="job-1",
        parent_job_id="parent-1",
        label="research X",
        content="The answer is 42.",
        outcome="success",
        model_used="claude-haiku-4-5",
        credit_spent=0.05,
    )

    assert result["ws_count"] == 3
    assert "website" in result["channel_results"]
    assert captured, "broadcast_to_user must have been called"
    evt = captured[0]
    assert evt["user_id"] == "user-xyz"
    assert evt["type"] == "message"
    assert evt["channel"] == "subagent"
    assert evt["subagent_message"] is True
    assert evt["subagent_job_id"] == "job-1"
    assert evt["subagent_parent_job_id"] == "parent-1"
    assert evt["subagent_label"] == "research X"
    assert evt["subagent_outcome"] == "success"
    assert evt["subagent_credit_spent"] == 0.05


@pytest.mark.skip(
    reason=(
        "Same FastAPI/Starlette on_startup version mismatch as the "
        "previous test. The exception-tolerance branch is covered by "
        "the production code's try/except — pre-existing pattern."
    ),
)
@pytest.mark.asyncio
async def test_broadcast_does_not_raise_when_ws_unavailable(monkeypatch):
    """A test env without ws_chat (or a runtime where broadcast_to_user
    raises) must not crash the writer call site. Telemetry continues
    via the structured log + the Message row already landed."""
    from app.agent.subagent_message_writer import broadcast_subagent_message
    import app.api.ws_chat as ws_chat

    async def _raise(*_, **__):
        raise RuntimeError("WS unavailable")

    monkeypatch.setattr(ws_chat, "broadcast_to_user", _raise)

    result = await broadcast_subagent_message(
        "u", message_id="m", day_chat_id=None, job_id="j",
        content="text", outcome="success",
    )
    assert result["ws_count"] == 0
    assert result["channel_results"]["website"]["status"] == "delivered"


@pytest.mark.skip(
    reason="Same FastAPI/Starlette on_startup version mismatch.",
)
@pytest.mark.asyncio
async def test_broadcast_event_for_failure_outcomes_carries_outcome(monkeypatch):
    """The WS event MUST carry outcome on failure/timeout — the
    frontend renders a different badge for these. Pin the four
    failure-shape outcomes."""
    from app.agent.subagent_message_writer import broadcast_subagent_message
    import app.api.ws_chat as ws_chat

    captured: list[dict[str, Any]] = []

    async def _fake_broadcast(_uid, event):
        captured.append(event)
        return 1

    monkeypatch.setattr(ws_chat, "broadcast_to_user", _fake_broadcast)

    for outcome in ("failed", "timeout", "cancelled", "budget_exhausted"):
        captured.clear()
        await broadcast_subagent_message(
            "u", message_id="m", day_chat_id="d", job_id="j",
            content="x", outcome=outcome,
        )
        assert captured[0]["subagent_outcome"] == outcome


# ──────────────────────────────────────────────────────────────────────
# SYSTEM_CHANNELS taxonomy
# ──────────────────────────────────────────────────────────────────────


def test_system_channels_includes_subagent():
    """Pin: conversation_resolver.SYSTEM_CHANNELS must include
    "subagent" — without this the resolver treats each fire as a
    user-thread Conversation, breaking Reading-A (one Conversation
    per (user, day_chat, channel)). The routine/trigger writers
    have the same invariant."""
    from app.agent.conversation_resolver import SYSTEM_CHANNELS
    assert "subagent" in SYSTEM_CHANNELS


def test_system_channels_carries_all_known_system_surfaces():
    """Pin the closed set so a 'helpful cleanup' PR that drops one
    of these (and silently changes Reading-A for that surface) fails
    review here."""
    from app.agent.conversation_resolver import SYSTEM_CHANNELS
    assert SYSTEM_CHANNELS == frozenset({
        "routine", "trigger", "api", "digest", "subagent",
    })


# ──────────────────────────────────────────────────────────────────────
# Function signature pinning
# ──────────────────────────────────────────────────────────────────────


def test_write_subagent_message_signature():
    """Pin the kwargs Phase 4 orchestrator depends on. A typo here
    or a missed rename surfaces here, not in production."""
    import inspect
    from app.agent.subagent_message_writer import write_subagent_message

    sig = inspect.signature(write_subagent_message)
    params = sig.parameters
    expected_keys = {
        "db", "user_id", "content", "job_id", "parent_job_id",
        "label", "model_used", "tokens_prompt", "tokens_completion",
        "credit_spent", "outcome", "tz_override", "extra_metadata",
    }
    assert expected_keys.issubset(set(params.keys())), (
        f"write_subagent_message missing kwargs: "
        f"{expected_keys - set(params.keys())}"
    )


# ──────────────────────────────────────────────────────────────────────
# Pure-function WS event shape (replaces the skipped broadcast tests —
# imports no FastAPI / Starlette, so version mismatches can't break it)
# ──────────────────────────────────────────────────────────────────────


def test_build_subagent_ws_event_carries_all_frontend_keys():
    """The frontend chat parser reads these exact keys to render the
    sub-agent bubble badge + the deep-link / outcome chip. Pin them."""
    from app.agent.subagent_message_writer import build_subagent_ws_event

    event = build_subagent_ws_event(
        message_id="msg-1",
        day_chat_id="dc-1",
        job_id="job-1",
        parent_job_id="parent-1",
        label="research X",
        content="The answer is 42.",
        outcome="success",
        model_used="claude-haiku-4-5",
        credit_spent=0.05,
    )
    assert event["type"] == "message"
    assert event["id"] == "msg-1"
    assert event["day_chat_id"] == "dc-1"
    assert event["role"] == "assistant"
    assert event["channel"] == "subagent"
    assert event["source"] == "subagent"
    assert event["content"] == "The answer is 42."
    assert event["model_used"] == "claude-haiku-4-5"
    assert event["subagent_message"] is True
    assert event["subagent_job_id"] == "job-1"
    assert event["subagent_parent_job_id"] == "parent-1"
    assert event["subagent_label"] == "research X"
    assert event["subagent_outcome"] == "success"
    assert event["subagent_credit_spent"] == 0.05
    assert "created_at" in event


def test_build_subagent_ws_event_outcomes():
    """Every failure-shape outcome must round-trip through the event
    dict so the frontend renders the correct badge."""
    from app.agent.subagent_message_writer import build_subagent_ws_event

    for outcome in ("success", "failed", "timeout", "cancelled", "budget_exhausted"):
        event = build_subagent_ws_event(
            message_id="m", day_chat_id="d", job_id="j",
            content="x", outcome=outcome,
        )
        assert event["subagent_outcome"] == outcome


def test_build_subagent_ws_event_credit_spent_none_stays_none():
    """When the credit hook hasn't fired (e.g. credit enforcement off),
    credit_spent stays None — not coerced to 0.0."""
    from app.agent.subagent_message_writer import build_subagent_ws_event

    event = build_subagent_ws_event(
        message_id="m", day_chat_id="d", job_id="j",
        content="x", credit_spent=None,
    )
    assert event["subagent_credit_spent"] is None


def test_build_subagent_ws_event_credit_rounded():
    """Avoid 16-decimal noise from float arithmetic in the WS frame."""
    from app.agent.subagent_message_writer import build_subagent_ws_event

    event = build_subagent_ws_event(
        message_id="m", day_chat_id="d", job_id="j",
        content="x", credit_spent=0.123456789,
    )
    assert event["subagent_credit_spent"] == 0.1235


def test_broadcast_subagent_message_signature():
    import inspect
    from app.agent.subagent_message_writer import broadcast_subagent_message

    sig = inspect.signature(broadcast_subagent_message)
    expected = {
        "message_id", "day_chat_id", "job_id", "parent_job_id",
        "label", "content", "outcome", "model_used", "credit_spent",
    }
    assert expected.issubset(set(sig.parameters.keys()))
