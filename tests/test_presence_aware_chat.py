"""Presence-aware chat delivery — payload contract pins (ws_chat).

The force-quit path emits mission_started / mission_completed /
mission_failed rows through the notify outbox. These pins guarantee
the payloads pass the platform ingest validator (unknown kinds or
non-scalar data would poison the outbox with permanent 4xx rejects)
and that the LA lane can route them (mission_id present).
"""

from __future__ import annotations

import pytest


def _ingest_model():
    from app.api.agent_notify import AgentNotifyRequest
    return AgentNotifyRequest


@pytest.mark.parametrize("kind", ["mission_started", "mission_completed", "mission_failed"])
def test_chat_turn_kinds_are_known(kind):
    from app.db.models import KNOWN_NOTIFY_KINDS
    assert kind in KNOWN_NOTIFY_KINDS


def test_chat_turn_payloads_pass_ingest_validation():
    NotifyIn = _ingest_model()
    mission_id = "chatturn:abc123def456"

    started = NotifyIn(
        user_id="u" * 36, idempotency_key="row-1-" + "x" * 8,
        event_kind="mission_started", title="Working on your answer",
        body="Research the top UI/UX design tools"[:180],
        data={
            "mission_id": mission_id,
            "mission_title": "Working on your answer",
            "route": "chat", "kind": "chat_turn", "urgent": True,
            "timer_end_ms": 1_784_140_000_000,
        },
        priority="default", dedup_key=f"{mission_id}:started",
    )
    assert started.data["mission_id"] == mission_id

    completed = NotifyIn(
        user_id="u" * 36, idempotency_key="row-2-" + "x" * 8,
        event_kind="mission_completed", title="Answer ready",
        body="Figma is still the default king…"[:180],
        data={
            "mission_id": mission_id,
            "mission_title": "Working on your answer",
            "route": "chat", "kind": "chat_turn", "urgent": True,
            "progress": 100, "dismiss_after_s": 900,
            "session_id": "s" * 36, "day_chat_id": "d" * 36,
            "message_id": "m" * 36,
        },
        priority="high", dedup_key=f"{mission_id}:completed",
    )
    assert completed.data["progress"] == 100

    failed = NotifyIn(
        user_id="u" * 36, idempotency_key="row-3-" + "x" * 8,
        event_kind="mission_failed", title="Couldn't finish your answer",
        body="Something went wrong — open the app and ask again.",
        data={
            "mission_id": mission_id,
            "mission_title": "Working on your answer",
            "route": "chat", "kind": "chat_turn", "urgent": True,
            "dismiss_after_s": 900,
        },
        priority="high", dedup_key=f"{mission_id}:failed",
    )
    assert failed.event_kind == "mission_failed"


def test_chat_turn_card_deeplinks_to_chat():
    """A tapped 'Working on your answer' card must land in the chat
    (where the answer lives), not Mission Control."""
    from app.services import apns_push

    p = apns_push.build_start_payload(
        mission_id="chatturn:abc", title="Working on your answer",
        deep_link="toup://chat", timestamp=1,
    )
    assert p["aps"]["attributes"]["deepLinkUrl"] == "toup://chat"

    default = apns_push.build_start_payload(
        mission_id="m-1", title="Mission", timestamp=1,
    )
    assert default["aps"]["attributes"]["deepLinkUrl"] == "toup://mission-control"


# ── status frames (2026-07-16 blank-response fix) ──────────────────


def test_agent_runner_accepts_on_status():
    """run() must accept the on_status callback — ws_chat passes it on
    every turn; a signature regression breaks all chat."""
    import inspect
    from app.agent.agent_runner import AgentRunner

    params = inspect.signature(AgentRunner.run).parameters
    assert "on_status" in params
    assert params["on_status"].default is None


def test_ws_chat_emits_status_frames():
    """Protocol pin: the chat WS acks every accepted message with
    {'type':'status','stage':'received'} and relays the runner's
    'thinking' liveness signal. Clients render the pre-token
    indicator off these frames."""
    import inspect
    from app.api import ws_chat

    src = inspect.getsource(ws_chat)
    assert '{"type": "status", "stage": "received"}' in src
    assert 'async def on_status' in src
    assert 'on_status=on_status' in src
