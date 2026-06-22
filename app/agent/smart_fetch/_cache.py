"""Tiny in-process TTL + LRU cache for smart_fetch.

Tenant isolation: the agent runtime serves ONE tenant at a time, but a generic
pool container CAN be re-bound to a new identity in-place via POST /admin/bind
without a restart. Isolation is therefore guaranteed by a clear_caches() call in
the bind handler (run right after the identity apply, before the bind is
committed / is_bound() flips true) — see smart_fetch.clear_caches and
app/api/admin_pool.py. DO NOT remove that call. Entries also expire (TTL) and
never persist across a restart.

The cache is synchronous; under asyncio (single-threaded) get/set are atomic
between awaits, so no lock is needed. A concurrent miss may fetch twice (benign
stampede), never a wrong/leaked value.
"""
from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any, Callable, Hashable, Optional


class TTLCache:
    def __init__(self, maxsize: int, ttl_s: float, clock: Callable[[], float] = time.monotonic):
        self.maxsize = max(1, int(maxsize))
        self.ttl_s = float(ttl_s)
        self._clock = clock
        self._data: "OrderedDict[Hashable, tuple[float, Any]]" = OrderedDict()

    def get(self, key: Hashable) -> Optional[Any]:
        item = self._data.get(key)
        if item is None:
            return None
        expires_at, value = item
        if self._clock() >= expires_at:
            self._data.pop(key, None)
            return None
        self._data.move_to_end(key)  # LRU: mark most-recently used
        return value

    def set(self, key: Hashable, value: Any) -> None:
        self._data[key] = (self._clock() + self.ttl_s, value)
        self._data.move_to_end(key)
        while len(self._data) > self.maxsize:
            self._data.popitem(last=False)  # evict least-recently used

    def clear(self) -> None:
        self._data.clear()

    def __len__(self) -> int:
        return len(self._data)
