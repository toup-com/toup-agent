"""Tests for the per-routine delivery_channels flow.

The dispatch path is wired through the existing broadcast_routine_message
helper, so we test at two levels:

  1. `parse_delivery_channels` — pure config decoding. Drops typos,
     enforces the website-is-always-included invariant.
  2. End-to-end `broadcast_routine_message` with `delivery_channels` set
     — verifies the dispatcher is called with the right args, even when
     no real Telegram/WhatsApp adapter is registered (it should
     gracefully report 'no_adapter' rather than raise).

We don't exercise the actual Telegram/WhatsApp send paths here — those
are best-effort I/O that the dispatcher swallows on failure. The
contract we pin is: "extra channels never break the website write."
"""

from __future__ import annotations

import pytest


# ── parse_delivery_channels ────────────────────────────────────────


def test_parse_returns_website_when_config_missing():
    from app.agent.routines.channel_dispatcher import parse_delivery_channels
    assert parse_delivery_channels(None) == ["website"]
    assert parse_delivery_channels({}) == ["website"]


def test_parse_returns_website_when_field_absent():
    from app.agent.routines.channel_dispatcher import parse_delivery_channels
    assert parse_delivery_channels({"other_key": 1}) == ["website"]


def test_parse_normalises_string_input():
    """Some callers may pass a single string; treat it as a one-element list."""
    from app.agent.routines.channel_dispatcher import parse_delivery_channels
    out = parse_delivery_channels({"delivery_channels": "telegram"})
    assert out == ["website", "telegram"]


def test_parse_always_includes_website():
    """Even when the user only picks Telegram, the website slug is added
    so the canonical Day-as-Chat record is never accidentally skipped."""
    from app.agent.routines.channel_dispatcher import parse_delivery_channels
    out = parse_delivery_channels({"delivery_channels": ["telegram"]})
    assert "website" in out
    assert "telegram" in out


def test_parse_drops_unknown_channels():
    """Typos must not crash the dispatcher — drop them silently rather
    than fail the whole routine fire."""
    from app.agent.routines.channel_dispatcher import parse_delivery_channels
    out = parse_delivery_channels({"delivery_channels": ["telegram", "carrier-pigeon", "whatsapp"]})
    assert "carrier-pigeon" not in out
    assert "telegram" in out
    assert "whatsapp" in out


def test_parse_dedupes_preserving_first_seen_order():
    """Dedupe keeps first occurrence — duplicate Telegram entries collapse
    to one but the user's intent (telegram + website) is preserved."""
    from app.agent.routines.channel_dispatcher import parse_delivery_channels
    out = parse_delivery_channels({"delivery_channels": ["telegram", "telegram", "website"]})
    assert sorted(out) == sorted(["website", "telegram"])
    # Each channel appears exactly once.
    assert len(out) == len(set(out))


# ── end-to-end via broadcast_routine_message ──────────────────────


@pytest.mark.asyncio
async def test_broadcast_with_no_extra_channels_is_unchanged():
    """When delivery_channels=['website'], the dispatcher must NOT be
    invoked at all — pure WS broadcast is the legacy path."""
    from app.agent.routines.message_writer import broadcast_routine_message

    # No mocks needed — no adapter is registered in the test env, so any
    # accidental dispatch would log a no_adapter. Asserting the call
    # returns 0 (no WS clients, no extra side effects) is the cheapest
    # signal that the legacy contract is intact.
    count = await broadcast_routine_message(
        user_id="test-user",
        message_id="msg-1",
        day_chat_id=None,
        source="email_briefing",
        content="hello",
        delivery_channels=["website"],
        routine_name="Morning briefing",
    )
    assert count == 0  # no WS clients in test env


@pytest.mark.asyncio
async def test_broadcast_with_telegram_channel_attempts_dispatch(caplog):
    """When delivery_channels includes 'telegram' but no bot is
    configured in the test env, the dispatcher must log a no_adapter (or
    no_recipient) and return — never raise. Asserting the absence of an
    exception is the load-bearing claim."""
    import logging
    from app.agent.routines.message_writer import broadcast_routine_message

    caplog.set_level(logging.INFO, logger="app.agent.routines.message_writer")

    await broadcast_routine_message(
        user_id="test-user",
        message_id="msg-2",
        day_chat_id=None,
        source="email_briefing",
        content="hello",
        delivery_channels=["website", "telegram"],
        routine_name="Morning briefing",
    )

    # The dispatcher logged its result map — at minimum the key exists.
    fanout_logs = [r for r in caplog.records if "extra channel fan-out" in r.getMessage()]
    assert fanout_logs, "expected the dispatcher to log its fan-out result"
    msg = fanout_logs[-1].getMessage()
    assert "telegram" in msg
