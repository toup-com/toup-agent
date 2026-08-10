"""Channel resolution helper — single source of truth for channel values.

Historically the agent had three silently-different defaults when `channel`
was missing across call sites:

  - agent_runner.py:1445   -> `channel or "telegram"` (prompt label)
  - agent_runner.py:1069   -> `channel or "agent"`    (session creation)
  - day_context_loader.py  -> `channel or "web"`      (history annotation)

A missing-channel bug could therefore persist as conv.channel="agent",
render in history as "[web ...]", and appear in Runtime Context as
"Channel: telegram" — three layers of silent misattribution for one
unknown value.

`resolve_channel` replaces all three sites with one resolution chain that
returns "unknown" and logs loudly when no value is resolvable, rather
than guessing. See SKILL.md Rule 12 (in progress).
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Channels the agent recognizes end-to-end. Keep in sync with
# channel_util tests, CHANNEL_GUIDANCE in agent_runner, and the
# RadioSessionManager.RADIO_ALLOWED_CHANNELS (which is a subset).
# The background channels (trigger/routine/autopilot/subagent) are
# established policy values — connector_dispatcher's deny sets and the
# per-channel prompt code key on them — but each fired the per-turn
# unknown_value warning below until they were admitted here (G-19b).
KNOWN_CHANNELS = frozenset({
    "web",
    "app",
    "mobile",
    "voice",
    "telegram",
    "discord",
    "slack",
    "whatsapp",
    "vibecoding",
    "extension",  # Chrome side-panel chat — first-class Day-as-Chat channel
    "agent",  # internal / legacy
    "trigger",  # unattended email_received turn (G-19b runner path)
    "routine",  # scheduled agent_task turn (routines runner)
    "autopilot",  # autonomous mission tick
    "subagent",  # spawned child run
})


def resolve_channel(
    explicit: Optional[str] = None,
    payload_hint: Optional[str] = None,
    conversation_hint: Optional[str] = None,
    *,
    user_id: Optional[str] = None,
    site: str = "unknown",
    default: str = "unknown",
) -> str:
    """Resolve a channel value with strict priority + loud fallback.

    Priority:
      1. `explicit`          — caller's declared channel (most authoritative)
      2. `payload_hint`      — value from WS msg.get("channel") or similar
      3. `conversation_hint` — Conversation.channel from the DB
      4. `default`           — typically "unknown"; NEVER silently guess

    Logs a warning at steps 3 and 4 so production ops can see where the
    fallback happened and instrument the caller.

    Args:
      explicit / payload_hint / conversation_hint: candidate values, any or
        all may be None / "".
      user_id: for log attribution (truncated to 8 chars).
      site: short label for the call site (e.g. "prompt_label",
        "session_create", "history_annotation") so logs tell you which
        layer is missing channel info.
      default: fallback string when nothing resolves — "unknown" by default.

    Returns:
      str — the resolved channel, guaranteed non-empty.
    """
    for candidate, source in (
        (explicit, "explicit"),
        (payload_hint, "payload"),
        (conversation_hint, "conversation"),
    ):
        if candidate and isinstance(candidate, str):
            c = candidate.strip().lower()
            if c:
                if c not in KNOWN_CHANNELS:
                    logger.warning(
                        "[channel] resolve_channel unknown_value site=%s source=%s value=%r user=%s — passing through",
                        site, source, c, (user_id[:8] if user_id else "-"),
                    )
                if source != "explicit":
                    logger.info(
                        "[channel] resolve_channel fallback site=%s source=%s value=%r user=%s",
                        site, source, c, (user_id[:8] if user_id else "-"),
                    )
                return c

    logger.warning(
        "[channel] resolve_channel default site=%s user=%s — no explicit/payload/conversation value; "
        "using %r. Fix at the ingress handler.",
        site, (user_id[:8] if user_id else "-"), default,
    )
    return default
