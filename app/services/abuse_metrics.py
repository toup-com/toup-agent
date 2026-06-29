"""Structured, PII-safe telemetry for the signup/free-grant abuse controls.

One emitter so every control logs the same shape, greppable as
``[abuse] event=…``. Critically, controls that default OFF emit a
SHADOW ``*_would`` event while disabled — so the real-world hit /
false-positive rate is measurable BEFORE the flag is ever flipped.

NEVER pass a raw email/IP. Use a canonical-hash prefix (``hp=``) or a
truncated user id (``uid=``). Helpers below enforce the truncation.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("abuse_controls")


def hp(canonical_hash: Optional[str]) -> str:
    """First 12 chars of a canonical-email hash — opaque, non-reversible."""
    return (canonical_hash or "?")[:12]


def uidp(user_id: Optional[str]) -> str:
    return (str(user_id) if user_id else "?")[:8]


def emit(event: str, **fields) -> None:
    """Emit one structured abuse-control event. Drop None fields.

    Events (real): tombstone_claimed, grant_suppressed, grant_deferred,
    disposable_rejected, signup_rate_limited, apple_sub_dedupe_hit.
    Events (shadow, flag OFF): grant_suppressed_would, disposable_would_reject,
    apple_sub_dedupe_would.
    """
    parts = " ".join(f"{k}={v}" for k, v in fields.items() if v is not None)
    logger.info("[abuse] event=%s %s", event, parts)
