"""
Agent Lanes — Separate execution contexts for different agent activities.

Lanes provide isolation and concurrency control:
- **main**: User-initiated chat interactions (default)
- **subagent**: Spawned sub-agent sessions
- **cron**: Scheduled/cron-triggered runs
- **hook**: Event-driven hook executions

Each lane can have its own model override, concurrency limits,
and resource constraints.
"""

from __future__ import annotations

import asyncio
import enum
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class LaneType(str, enum.Enum):
    MAIN = "main"
    SUBAGENT = "subagent"
    CRON = "cron"
    HOOK = "hook"


@dataclass
class LaneRun:
    """A single agent run within a lane."""
    run_id: str
    lane: LaneType
    user_id: str
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    model: Optional[str] = None
    idempotency_key: Optional[str] = None
    status: str = "running"  # running | completed | failed | cancelled
    error: Optional[str] = None
    tokens_in: int = 0
    tokens_out: int = 0


class LaneManager:
    """
    Manages agent execution lanes with concurrency control and idempotency.
    """

    def __init__(self, max_concurrent: int = 5):
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._runs: Dict[str, LaneRun] = {}
        self._idempotency_cache: Dict[str, str] = {}  # key → run_id
        self._lock = asyncio.Lock()

    async def acquire(
        self,
        lane: LaneType,
        user_id: str,
        model: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> Optional[LaneRun]:
        """
        Acquire a slot in the lane. Returns a LaneRun on success,
        or None if the idempotency key is already in-flight.
        """
        async with self._lock:
            # Check idempotency
            if idempotency_key and idempotency_key in self._idempotency_cache:
                existing_id = self._idempotency_cache[idempotency_key]
                existing = self._runs.get(existing_id)
                if existing and existing.status == "running":
                    logger.info(f"[LANE] Idempotency key {idempotency_key} already running: {existing_id}")
                    return None

            run_id = str(uuid.uuid4())[:12]
            run = LaneRun(
                run_id=run_id,
                lane=lane,
                user_id=user_id,
                model=model,
                idempotency_key=idempotency_key,
            )
            self._runs[run_id] = run
            if idempotency_key:
                self._idempotency_cache[idempotency_key] = run_id

        await self._semaphore.acquire()
        return run

    async def release(self, run: LaneRun, status: str = "completed", error: Optional[str] = None):
        """Release the lane slot and mark the run as finished."""
        run.finished_at = time.time()
        run.status = status
        run.error = error
        self._semaphore.release()
        logger.debug(f"[LANE] Released {run.lane.value}/{run.run_id} — {status}")

    def get_active_runs(self) -> List[LaneRun]:
        """Return all currently running lane runs."""
        return [r for r in self._runs.values() if r.status == "running"]

    def get_run(self, run_id: str) -> Optional[LaneRun]:
        return self._runs.get(run_id)

    def get_stats(self) -> Dict[str, Any]:
        """Return lane statistics."""
        active = self.get_active_runs()
        by_lane = {}
        for lane in LaneType:
            lane_runs = [r for r in active if r.lane == lane]
            by_lane[lane.value] = len(lane_runs)

        total = len(self._runs)
        completed = len([r for r in self._runs.values() if r.status == "completed"])
        failed = len([r for r in self._runs.values() if r.status == "failed"])

        return {
            "active": len(active),
            "total_runs": total,
            "completed": completed,
            "failed": failed,
            "by_lane": by_lane,
            "max_concurrent": self._semaphore._value + len(active),
        }

    def clear_history(self, keep_active: bool = True):
        """Clear finished runs from history."""
        if keep_active:
            self._runs = {k: v for k, v in self._runs.items() if v.status == "running"}
        else:
            self._runs.clear()
        # Clean idempotency cache for finished runs
        active_ids = set(self._runs.keys())
        self._idempotency_cache = {
            k: v for k, v in self._idempotency_cache.items() if v in active_ids
        }


# Singleton
_lane_manager: Optional[LaneManager] = None


def get_lane_manager() -> LaneManager:
    global _lane_manager
    if _lane_manager is None:
        from app.config import settings
        _lane_manager = LaneManager(max_concurrent=settings.lane_max_concurrent)
    return _lane_manager
