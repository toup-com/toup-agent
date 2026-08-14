"""
Structured-log events for the mobile web shell rollout.

Same shape and the same reason as `onboarding_events.py`: single-line key=value
records on a dedicated logger, picked up by Promtail and queried in Loki as
`{logger="web_shell.events"}`. NOT a new table — the funnel this answers is
"did the cohort that got the new shell behave differently", which is a log
query, and `product_events` would be a migration, a model and a retention
policy for data nobody joins against.

The three events are the minimum that make a ramp readable:

  shell_rendered  the DENOMINATOR. Without it a rise in drawer_opened is
                  indistinguishable from a rise in traffic. Carries which shell
                  the session actually got, which is the only honest source for
                  "is the flag doing anything" — the rollout percentage is what
                  we asked for, this is what was delivered.
  drawer_opened   the shell's one new interaction. If the cohort never opens
                  the drawer, navigation got worse, not better.
  shell_disabled  someone used the kill switch. One of these is a bug report
                  that did not get filed.

`viewport` is a BUCKET, not a width. A raw pixel width plus a user id is a
device fingerprint, and none of these questions need one.
"""

from __future__ import annotations

import logging
from typing import Optional

event_log = logging.getLogger("web_shell.events")

# Buckets match the media queries the shell is actually built on: the mobile
# rules are `max-width: 767px`, and 1024 is where the desktop layout is
# guaranteed untouched.
VIEWPORT_BUCKETS = ("phone", "tablet", "desktop")


def bucket_for_width(width: int) -> str:
    if width < 768:
        return "phone"
    if width < 1024:
        return "tablet"
    return "desktop"


def _fmt_kv(payload: dict) -> str:
    parts: list[str] = []
    for k, v in payload.items():
        if v is None:
            continue
        s = str(v)
        if any(c.isspace() for c in s):
            s = f'"{s}"'
        parts.append(f"{k}={s}")
    return " ".join(parts)


def emit_shell_rendered(
    *, user_id: str, shell: str, viewport: str, standalone: bool,
) -> None:
    """`shell` is 'mobile' or 'legacy' — what the session was actually served."""
    event_log.info(
        "web_shell.shell_rendered %s",
        _fmt_kv({
            "user_id": user_id,
            "shell": shell,
            "viewport": viewport,
            "standalone": "true" if standalone else "false",
        }),
    )


def emit_drawer_opened(*, user_id: str, via: str) -> None:
    """`via` is 'button' or 'swipe' — the two ways in."""
    event_log.info(
        "web_shell.drawer_opened %s",
        _fmt_kv({"user_id": user_id, "via": via}),
    )


def emit_shell_disabled(*, user_id: str, reason: Optional[str] = None) -> None:
    event_log.info(
        "web_shell.shell_disabled %s",
        _fmt_kv({"user_id": user_id, "reason": reason or "kill_switch"}),
    )
