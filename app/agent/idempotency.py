"""
Idempotency Keys — Prevent duplicate agent runs.

Tracks idempotency keys for agent operations. If a request
is received with an already-seen key, the cached result is
returned instead of re-executing.

Usage:
    from app.agent.idempotency import get_idempotency_store

    store = get_idempotency_store()

    if store.has_key("req-abc-123"):
        cached = store.get_result("req-abc-123")
    else:
        result = await run_agent(...)
        store.store_result("req-abc-123", result)
"""

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class IdempotencyEntry:
    """An idempotency entry with cached result."""
    key: str
    result: Any = None
    status: str = "pending"  # pending, completed, failed
    created_at: float = 0.0
    completed_at: Optional[float] = None
    ttl_seconds: int = 3600  # 1 hour default
    hit_count: int = 0

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()

    @property
    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.ttl_seconds

    @property
    def is_completed(self) -> bool:
        return self.status == "completed"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "status": self.status,
            "hit_count": self.hit_count,
            "is_expired": self.is_expired,
            "age_seconds": round(time.time() - self.created_at, 1),
        }


class IdempotencyStore:
    """
    In-memory idempotency store for agent operations.

    Tracks request keys and caches results. Expired entries
    are periodically cleaned up.
    """

    def __init__(self, default_ttl: int = 3600, max_entries: int = 10000):
        self._entries: Dict[str, IdempotencyEntry] = {}
        self._default_ttl = default_ttl
        self._max_entries = max_entries

    def has_key(self, key: str) -> bool:
        """Check if an idempotency key exists and is not expired."""
        entry = self._entries.get(key)
        if entry is None:
            return False
        if entry.is_expired:
            self._entries.pop(key, None)
            return False
        return True

    def acquire(self, key: str, *, ttl: Optional[int] = None) -> bool:
        """
        Acquire an idempotency key.

        Returns True if this is a new key (should execute).
        Returns False if key exists (should return cached/wait).
        """
        if self.has_key(key):
            entry = self._entries[key]
            entry.hit_count += 1
            return False

        entry = IdempotencyEntry(
            key=key,
            ttl_seconds=ttl or self._default_ttl,
        )
        self._entries[key] = entry

        # Evict old entries if over capacity
        if len(self._entries) > self._max_entries:
            self._evict_expired()

        return True

    def store_result(self, key: str, result: Any, *, status: str = "completed") -> bool:
        """Store the result for an idempotency key."""
        entry = self._entries.get(key)
        if entry is None:
            return False

        entry.result = result
        entry.status = status
        entry.completed_at = time.time()
        return True

    def get_result(self, key: str) -> Optional[Any]:
        """Get the cached result for an idempotency key."""
        entry = self._entries.get(key)
        if entry and not entry.is_expired and entry.is_completed:
            entry.hit_count += 1
            return entry.result
        return None

    def get_entry(self, key: str) -> Optional[IdempotencyEntry]:
        """Get full entry for a key."""
        entry = self._entries.get(key)
        if entry and entry.is_expired:
            self._entries.pop(key, None)
            return None
        return entry

    def mark_failed(self, key: str, error: Optional[str] = None) -> bool:
        """Mark a key as failed (allows retry)."""
        entry = self._entries.get(key)
        if entry:
            entry.status = "failed"
            entry.result = {"error": error} if error else None
            entry.completed_at = time.time()
            return True
        return False

    def remove(self, key: str) -> bool:
        """Remove an idempotency key."""
        return self._entries.pop(key, None) is not None

    def generate_key(self, *parts: str) -> str:
        """Generate a deterministic idempotency key from parts."""
        combined = ":".join(str(p) for p in parts)
        return hashlib.sha256(combined.encode()).hexdigest()[:32]

    def _evict_expired(self) -> int:
        """Remove expired entries."""
        expired = [k for k, v in self._entries.items() if v.is_expired]
        for k in expired:
            self._entries.pop(k, None)
        return len(expired)

    def cleanup(self) -> int:
        """Manually clean up expired entries."""
        return self._evict_expired()

    def list_entries(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all entries."""
        entries = list(self._entries.values())
        if status:
            entries = [e for e in entries if e.status == status]
        return [e.to_dict() for e in entries]

    def stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        by_status: Dict[str, int] = {}
        total_hits = 0
        expired = 0
        for e in self._entries.values():
            by_status[e.status] = by_status.get(e.status, 0) + 1
            total_hits += e.hit_count
            if e.is_expired:
                expired += 1

        return {
            "total_entries": len(self._entries),
            "by_status": by_status,
            "total_hits": total_hits,
            "expired": expired,
            "max_entries": self._max_entries,
        }


# ── Singleton ────────────────────────────────────────────
_store: Optional[IdempotencyStore] = None


def get_idempotency_store() -> IdempotencyStore:
    """Get the global idempotency store."""
    global _store
    if _store is None:
        _store = IdempotencyStore()
    return _store
