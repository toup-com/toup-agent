"""Pending credential-confirm state for the Vault CP4 chat-save flow.

When the LLM calls `save_streaming_credential`, a pending entry is stored
here keyed by a server-generated request_id. The client later returns a
`credential_confirm_response` WS frame carrying that request_id plus the
user-supplied email + password; the server validates the pair, pops the
entry, and writes via the CP2-encrypted path. Model B guarantee: the
password NEVER transits the LLM.

# NOTE (CP4.1, 2026-04): This pending-confirm dict is process-local.
# It assumes platform-api runs as a single Railway replica. If this service
# is scaled horizontally (>1 replica), confirms will fail with
# expired_or_unknown whenever the tool-call and the confirm land on different
# replicas, with no clear error signal. Before scaling horizontally, migrate
# this state to Redis (or equivalent shared store) keyed by request_id.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)


DEFAULT_TTL_SECONDS = 300          # 5-minute confirmation window
RATE_LIMIT_WINDOW_SECONDS = 3600   # 1-hour rolling window
RATE_LIMIT_MAX_ATTEMPTS = 5        # 5 tool invocations per user per hour


@dataclass
class PendingConfirm:
    request_id: str
    user_id: str
    channel: str
    email_hint: Optional[str]
    created_at: float
    expires_at: float


_pending: Dict[str, PendingConfirm] = {}
_pending_lock = asyncio.Lock()

# Rate limit state — keyed by user_id. Stores a list of attempt
# timestamps (epoch seconds). Per amendment 3: EVERY invocation of
# _tool_save_streaming_credential increments this; credential_confirm_response
# handling does NOT increment it.
_attempts_by_user: Dict[str, list] = {}
_attempts_lock = asyncio.Lock()


async def register_attempt(user_id: str) -> bool:
    """Record a tool invocation. Returns True if within limit, False if over.

    Prunes timestamps older than the rolling window, appends `now`, and
    checks count. Separate from `create_pending` so the counter ticks
    even when we're about to reject (rate_limited callers skip the pending
    create).
    """
    now = time.time()
    cutoff = now - RATE_LIMIT_WINDOW_SECONDS
    async with _attempts_lock:
        lst = _attempts_by_user.get(user_id, [])
        lst = [t for t in lst if t >= cutoff]
        lst.append(now)
        _attempts_by_user[user_id] = lst
        return len(lst) <= RATE_LIMIT_MAX_ATTEMPTS


async def create_pending(
    user_id: str,
    channel: str,
    email_hint: Optional[str] = None,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
) -> PendingConfirm:
    request_id = str(uuid.uuid4())
    now = time.time()
    entry = PendingConfirm(
        request_id=request_id,
        user_id=user_id,
        channel=channel,
        email_hint=email_hint,
        created_at=now,
        expires_at=now + ttl_seconds,
    )
    async with _pending_lock:
        _pending[request_id] = entry
    return entry


async def pop_pending(
    request_id: str,
    user_id: str,
) -> Optional[PendingConfirm]:
    """Atomically remove and return a pending entry.

    Returns None if:
      - request_id is unknown
      - request_id belongs to a different user (cross-user protection)
      - entry is expired
    """
    now = time.time()
    async with _pending_lock:
        entry = _pending.get(request_id)
        if entry is None:
            return None
        if entry.user_id != user_id:
            logger.warning(
                "credential_confirm cross-user attempt: request owner=%s, caller=%s",
                entry.user_id, user_id,
            )
            return None
        if entry.expires_at < now:
            _pending.pop(request_id, None)
            return None
        _pending.pop(request_id, None)
        return entry


async def cleanup_expired() -> int:
    now = time.time()
    removed = 0
    async with _pending_lock:
        for rid in [r for r, e in _pending.items() if e.expires_at < now]:
            _pending.pop(rid, None)
            removed += 1
    return removed


# Test-only helpers — not for production use.
async def _reset_all_for_tests() -> None:
    async with _pending_lock:
        _pending.clear()
    async with _attempts_lock:
        _attempts_by_user.clear()


async def _peek_pending_count_for_tests() -> int:
    async with _pending_lock:
        return len(_pending)
