"""GA run: `api` is a first-class channel, not an unknown value.

Found during the GA channel-matrix inventory: `channel="api"` is stamped
on every /v1/chat turn (api_v1.py) and on chat.py's API path, but the
value was in NEITHER `KNOWN_CHANNELS` (so `resolve_channel` logged an
`unknown_value` WARNING on every single API turn) NOR `CHANNEL_GUIDANCE`
(so paying API consumers got "Unknown channel — format conservatively:
short, minimal markdown" — the most restrictive formatting in the fleet,
for the one surface that is by definition programmatic).

`agent_task` (apps.py agent-task runner) and `health_probe`
(connector_health_probe) are internal stamps with the same per-turn
warning spam; they join KNOWN_CHANNELS but deliberately get NO guidance
entry — the conservative fallback is right for internal turns, the
warning is not.

Red-first: all three assertions fail on the pre-fix tree.
"""
from __future__ import annotations

from app.agent.agent_runner import CHANNEL_GUIDANCE
from app.agent.channel_util import KNOWN_CHANNELS


def test_api_is_a_known_channel():
    assert "api" in KNOWN_CHANNELS, (
        "every /v1/chat turn logs '[channel] resolve_channel unknown_value' "
        "— the developer API is a first-class channel"
    )


def test_internal_stamps_are_known():
    for ch in ("agent_task", "health_probe"):
        assert ch in KNOWN_CHANNELS, (
            f"'{ch}' is stamped by internal runners and warns on every turn"
        )


def test_api_has_guidance_and_it_is_programmatic():
    guidance = CHANNEL_GUIDANCE.get("api")
    assert guidance, (
        "API consumers got 'Unknown channel — format conservatively' — "
        "the one surface that is by definition programmatic was the most "
        "restricted"
    )
    assert "markdown" in guidance.lower()


def test_internal_stamps_keep_the_conservative_fallback():
    """No guidance entry for internal stamps — the fallback is correct for
    them; only the warning was wrong."""
    for ch in ("agent_task", "health_probe"):
        assert ch not in CHANNEL_GUIDANCE
