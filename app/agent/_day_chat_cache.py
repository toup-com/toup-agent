"""TKT-LAT-011 — process-wide cache for (user_id, local_date) → day_chat_id.

The day-chat resolver runs on every user message. The first SELECT
(existing-check) is a hot lookup that always returns the same row for
a given (user, local-date) pair for the entire day. Caching it skips
one DB round-trip per turn for the steady-state case.

Cache key includes the local date, so day boundary invalidation is
automatic — a new day produces a new key and the old one expires by
size eviction (or the explicit ``invalidate_*`` helpers below).

Isolated module (no ``app.config`` import) so unit tests can drive
the cache without booting Settings.
"""

from __future__ import annotations

from collections import OrderedDict
from datetime import date as _Date
from typing import Optional, Tuple


_MAX_ENTRIES: int = 4096
_CACHE: "OrderedDict[Tuple[str, _Date], str]" = OrderedDict()


def get_cached_day_chat_id(user_id: str, local_date: _Date) -> Optional[str]:
    """Return the cached DayChat id for ``(user_id, local_date)`` or ``None``."""
    key = (user_id, local_date)
    val = _CACHE.get(key)
    if val is None:
        return None
    _CACHE.move_to_end(key)  # LRU promotion
    return val


def set_cached_day_chat_id(user_id: str, local_date: _Date, day_chat_id: str) -> None:
    """Cache the resolved DayChat id. Bounded LRU; evicts oldest at capacity."""
    if not day_chat_id:
        return
    key = (user_id, local_date)
    _CACHE[key] = day_chat_id
    _CACHE.move_to_end(key)
    while len(_CACHE) > _MAX_ENTRIES:
        _CACHE.popitem(last=False)


def invalidate_cached_day_chat_id(user_id: str, local_date: _Date) -> None:
    """Drop a single entry. Idempotent — safe for unknown keys."""
    _CACHE.pop((user_id, local_date), None)


def invalidate_user(user_id: str) -> None:
    """Drop every cached entry for a user (timezone change, account reset)."""
    keys = [k for k in _CACHE if k[0] == user_id]
    for k in keys:
        _CACHE.pop(k, None)
