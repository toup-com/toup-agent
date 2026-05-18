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

        # ── Dual-write to job_events (PR 3 of unified jobs arc) ──
        # ``build_logs_json`` stays as the per-job verbose log capped
        # at 500 entries (rendered in the job detail drawer). The new
        # ``job_events`` table is the cross-job indexable activity-
        # feed surface — one row per material event, JOINed to
        # ``build_jobs.title`` for attribution. Writes are
        # best-effort: a failed event insert must not fail the job.
        await self._write_job_event(level, message, detail, meta)

    async def _write_job_event(
        self,
        level: str,
        message: str,
        detail: str = "",
        meta: Optional[Dict] = None,
    ) -> None:
        """Insert one ``job_events`` row for this log entry.

        Maps JobLogger levels to the ``job_events.kind`` closed enum:

          info → ``info`` (a generic progress signal; not in the
                  closed enum but used as a catch-all for plain log
                  lines until Auto Builder + handlers emit explicit
                  phase_started / phase_completed / output_posted
                  events from PR 5 onwards)
          tool → ``tool_call``
          edit → ``tool_call`` (file edit is a kind of tool call)
          error → ``error``

        Best-effort write: any exception is swallowed and warning-
        logged. Logging must never cascade into a job failure.
        """
        try:
            from app.db.database import async_session_maker
            from app.db.models import JobEvent

            kind_map = {"info": "info", "tool": "tool_call", "edit": "tool_call", "error": "error"}
            kind = kind_map.get(level, "info")
            metadata_payload: Optional[str] = None
            if detail or meta:
                metadata_payload = json.dumps(
                    {"detail": detail or "", "meta": meta or {}},
                    default=str,
                )

            async with async_session_maker() as db:
                db.add(JobEvent(
                    job_id=self.job_id,
                    user_id=self.user_id,
                    kind=kind,
                    label=message[:200] if message else None,
                    level=level,
                    metadata_json=metadata_payload,
                ))
                await db.commit()
        except Exception as e:
            # Swallow — a failed event write must not propagate into
            # the job's success path. The build_logs_json fallback
            # still captures everything; we just lose this row from
            # the cross-job feed.
            logger.warning(
                f"[JOB:{self.job_id[:8]}] job_events write failed: {e}"
            )

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
