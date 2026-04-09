"""
Job Logger — Lightweight structured logging for agent tasks and vibe coding.

Reuses the same log entry format and WebSocket broadcast pattern as BuildLogger,
but without the build-step-specific methods. All three job types (auto_builder,
vibe_code, agent_task) produce logs that land in BuildJob.build_logs_json and
stream via the 'job_log' WebSocket event.

Log entry format:
{
    "ts": "2026-04-08T10:30:00.123Z",
    "level": "info",        # info, tool, edit, error
    "step": "running",      # phase label (freeform)
    "message": "...",       # Human-readable
    "detail": "...",        # Optional extra context
    "meta": { ... }         # Optional structured data
}
"""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class JobLogger:
    """Captures structured logs for a job and broadcasts them via WebSocket."""

    def __init__(
        self,
        job_id: str,
        user_id: str,
        ws_broadcast: Optional[Callable] = None,
        max_logs: int = 500,
    ):
        self.job_id = job_id
        self.user_id = user_id
        self._ws_broadcast = ws_broadcast
        self._logs: List[Dict[str, Any]] = []
        self._max_logs = max_logs
        self._total_tokens = 0

    async def info(self, message: str, detail: str = "", meta: Optional[Dict] = None):
        await self._log("info", message, detail, meta)

    async def tool(self, message: str, detail: str = "", meta: Optional[Dict] = None):
        await self._log("tool", message, detail, meta)

    async def edit(self, message: str, detail: str = "", meta: Optional[Dict] = None):
        await self._log("edit", message, detail, meta)

    async def error(self, message: str, detail: str = "", meta: Optional[Dict] = None):
        await self._log("error", message, detail, meta)

    async def _log(self, level: str, message: str, detail: str = "", meta: Optional[Dict] = None):
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "step": "running",
            "message": message,
            "detail": detail or "",
        }
        if meta:
            entry["meta"] = meta

        self._logs.append(entry)
        if len(self._logs) > self._max_logs:
            self._logs = self._logs[-self._max_logs:]

        py_level = {"info": "info", "tool": "info", "edit": "info", "error": "error"}.get(level, "info")
        getattr(logger, py_level)(f"[JOB:{self.job_id[:8]}] {message}" + (f" | {detail}" if detail else ""))

        if self._ws_broadcast:
            try:
                await self._ws_broadcast(self.user_id, {
                    "type": "job_log",
                    "job_id": self.job_id,
                    "entry": entry,
                })
            except Exception:
                pass

    async def persist(self):
        """Save accumulated logs to the database."""
        from app.db.database import async_session_maker
        from app.db.models import BuildJob

        try:
            async with async_session_maker() as db:
                job = await db.get(BuildJob, self.job_id)
                if job:
                    job.build_logs_json = json.dumps(self._logs)
                    if self._total_tokens > 0:
                        job.total_tokens = self._total_tokens
                    await db.commit()
        except Exception as e:
            logger.warning(f"[JOB:{self.job_id[:8]}] Failed to persist logs: {e}")
