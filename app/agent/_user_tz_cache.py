"""TKT-LAT-004 — Process-wide TTL cache for ``User.timezone``.

The agent's per-turn timezone seed query (in ``agent_runner.run``) fires
for every channel that does not supply ``client_tz`` over the wire —
Telegram, voice, WhatsApp, scheduled routines — i.e. every user turn
from those surfaces. The value changes only when the user edits their
profile, so a short TTL cache eliminates ~70–100 ms per turn with
negligible staleness risk.

Profile-update endpoints call :func:`invalidate_cached_user_tz` to drop
a user's entry so the next agent turn refetches.

Isolated in its own module (no ``app.config`` import) so unit tests can
exercise the cache without booting the full Settings stack.
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Tuple


_USER_TZ_TTL_S: float = 300.0
_USER_TZ_CACHE: Dict[str, Tuple[str, float]] = {}


def get_cached_user_tz(user_id: str) -> Optional[str]:
    """Return the cached timezone for ``user_id`` or ``None`` if absent/expired."""
    entry = _USER_TZ_CACHE.get(user_id)
    if not entry:
        return None
    tz_name, expires_at = entry
    if time.monotonic() >= expires_at:
        _USER_TZ_CACHE.pop(user_id, None)
        return None
    return tz_name


def set_cached_user_tz(user_id: str, tz_name: Optional[str]) -> None:
    """Cache the resolved IANA timezone for ``user_id``. A falsy ``tz_name`` is a no-op."""
    if not tz_name:
        return
    _USER_TZ_CACHE[user_id] = (tz_name, time.monotonic() + _USER_TZ_TTL_S)


def invalidate_cached_user_tz(user_id: str) -> None:
    """Drop the cached entry. Idempotent — safe to call for unknown users."""
    _USER_TZ_CACHE.pop(user_id, None)


def invalidate_cached_user_tz_with_day_chat(user_id: str) -> None:
    """TKT-LAT-011: convenience hook for timezone-update sites — drops
    both the tz cache and every cached day_chat_id for the user, since
    a tz change can shift their local_date and therefore the resolved
    day_chat row."""
    invalidate_cached_user_tz(user_id)
    try:
        from app.agent._day_chat_cache import invalidate_user as _inv_dc
        _inv_dc(user_id)
    except Exception:
        pass
