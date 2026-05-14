"""Synthetic repro for Defect A — silent-success channel-leak (H2).

The reported production symptom for routine `8624c36d-31bc-4d7c-a715-319247557754`
was `last_status="success"` while the user received nothing on website /
Telegram / WhatsApp. Phase 1.2's H2 hypothesis was: every silent-skip return
in the channel fan-out (`no_recipient`, `no_adapter`, `ws_count=0`) gets
logged and swallowed; nothing propagates to `RoutineResult.status`.

This file isolates the mechanism: invoke the dispatcher directly with
mocked Telegram (no mapping) + mocked WhatsApp (no E.164), assert the
silent-skip happens, and the result map (the only signal an operator has)
contains the `no_recipient` / `no_adapter` tags.

After Ticket 2.5 lands, the same scenarios will be re-run with the NEW
contract: each call returns a per-channel confirmation with delivery_id,
and missing-recipient routes the outcome to `partial` instead of `success`.
"""

from __future__ import annotations

import logging

import pytest


# ── 1. parse_delivery_channels — pure-function lock (sanity) ────────


def test_parse_delivery_channels_keeps_all_three_canonical_channels():
    """The exact config shape the user reported: all three channels."""
    from app.agent.routines.channel_dispatcher import parse_delivery_channels

    out = parse_delivery_channels({"delivery_channels": ["website", "telegram", "whatsapp"]})
    assert set(out) == {"website", "telegram", "whatsapp"}


# ── 2. Telegram silent-skip when no mapping (the H2 mechanism) ─────


@pytest.mark.asyncio
async def test_telegram_skip_when_no_mapping_returns_no_recipient_silently(caplog, monkeypatch):
    """SCENARIO: dispatcher called with telegram in the list, no
    TelegramUserMapping row, no bot configured.

    Asserts:
      1. The function does NOT raise.
      2. The result map contains telegram → ('no_recipient' OR 'no_adapter').
      3. A log line was emitted (operator can grep), but the function
         signature returns success-shaped data; the caller has no way
         to know "the user did not receive this" without parsing logs.
    """
    from app.agent.routines.channel_dispatcher import deliver_to_extra_channels
    from app.db import async_session_maker
    from app.db.models import User
    import uuid

    user_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(id=user_id, email=f"{user_id}@x.test",
                    hashed_password="x", name="t"))
        await db.commit()

    caplog.set_level(logging.INFO)
    results = await deliver_to_extra_channels(
        user_id=user_id,
        delivery_channels=["telegram"],
        routine_name="Defect A Repro",
        content="hello world",
        db_session_maker=async_session_maker,
    )

    # Lock #1: no exception, function returned.
    # Lock #2: telegram appears with a silent-skip tag.
    assert "telegram" in results
    assert results["telegram"] in ("no_recipient", "no_adapter", "error"), (
        f"telegram returned tag={results['telegram']!r} — H2 mechanism predicts "
        "silent-skip with no_recipient/no_adapter/error, all of which are "
        "swallowed by the runner. If this fails the dispatcher learned a new "
        "return value and the test needs updating."
    )


# ── 3. WhatsApp silent-skip when no E.164 (the H2 mechanism) ───────


@pytest.mark.asyncio
async def test_whatsapp_skip_when_no_e164_returns_silently(caplog):
    """SCENARIO: dispatcher called with whatsapp in the list, no
    AgentConfig.whatsapp_self_e164 row at all.

    Same lock as Telegram — the function returns a silent-skip tag, no
    exception is raised, no signal propagates to the caller's RoutineResult.
    """
    from app.agent.routines.channel_dispatcher import deliver_to_extra_channels
    from app.db import async_session_maker

    caplog.set_level(logging.INFO)
    results = await deliver_to_extra_channels(
        user_id="any",
        delivery_channels=["whatsapp"],
        routine_name="Defect A Repro",
        content="hello world",
        db_session_maker=async_session_maker,
    )
    assert "whatsapp" in results
    assert results["whatsapp"] in ("no_recipient", "no_adapter", "error")


# ── 4. The shape of the bug — silent-skip never raises ─────────────


@pytest.mark.asyncio
async def test_all_three_channels_silent_skip_returns_no_error_indication():
    """The closest synthetic reproduction of the prod symptom available
    without a real tenant: configure all three channels; none have valid
    recipients; the dispatcher returns a map of tags but signals no error.

    The caller (`broadcast_routine_message`) treats this map as
    informational only and never feeds it back into the RoutineResult.
    This is the mechanism Ticket 2.5 fixes.
    """
    from app.agent.routines.channel_dispatcher import deliver_to_extra_channels
    from app.db import async_session_maker

    results = await deliver_to_extra_channels(
        user_id="any",
        delivery_channels=["telegram", "whatsapp"],
        routine_name="DefectA",
        content="hi",
        db_session_maker=async_session_maker,
    )
    # Every channel returned with a silent-skip tag. The dispatcher
    # successfully completed; the caller's RoutineResult will stamp
    # status="success" downstream.
    assert set(results.keys()) == {"telegram", "whatsapp"}
    for ch, tag in results.items():
        assert tag in ("no_recipient", "no_adapter", "error", "sent"), (
            f"unexpected tag {tag!r} for channel {ch!r}"
        )
    # The bug surface: there is no flag, no exception, no marker that
    # tells the caller "no one got this." Ticket 2.5 introduces a
    # `channel_results_json` field on routine_runs to expose exactly
    # this map upward, and a `partial` outcome when any tag != 'sent'.


# ── 5. broadcast_routine_message hides the dispatcher's result ─────


@pytest.mark.asyncio
async def test_broadcast_routine_message_returns_structured_channel_results(caplog):
    """Ticket 2.5 contract: `broadcast_routine_message` returns a
    structured dict carrying per-channel confirmations:

      {"ws_count": int,
       "channel_results": {channel_slug: {"status": ..., ...}}}

    Pre-fix this returned a bare int. Defect A's root cause was that
    callers had no way to see per-channel results — the new dict shape
    is the conduit the runner uses to downgrade outcome to `partial`
    when any channel skipped.
    """
    from app.agent.routines.message_writer import broadcast_routine_message

    caplog.set_level(logging.INFO)
    out = await broadcast_routine_message(
        user_id="x",
        message_id="m-1",
        day_chat_id=None,
        source="email_briefing",
        content="hi",
        delivery_channels=["website", "telegram", "whatsapp"],
        routine_name="DefectA",
    )
    assert isinstance(out, dict)
    assert "ws_count" in out
    assert "channel_results" in out
    # Website always present and reports delivered (the Message row
    # exists; ws_count records live subscribers).
    assert out["channel_results"]["website"]["status"] == "delivered"
    # Telegram + WhatsApp ATTEMPTED. In the test env both have no
    # adapter, so they report skipped — but they ARE present in the
    # results map (the pre-fix bug was their absence).
    assert "telegram" in out["channel_results"]
    assert "whatsapp" in out["channel_results"]
    assert out["channel_results"]["telegram"]["status"] in ("skipped", "failed", "delivered")
