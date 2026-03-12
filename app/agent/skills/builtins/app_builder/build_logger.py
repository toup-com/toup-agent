"""
Build Logger — Structured logging for the app build pipeline.

Each build produces a stream of log entries that are:
  1. Stored in BuildJob.build_logs_json (persistent)
  2. Broadcast via WebSocket as 'job_log' events (real-time)
  3. Available via GET /apps/jobs/{job_id}/logs (REST)

Log entry format:
{
    "ts": "2026-03-10T10:30:00.123Z",  # ISO timestamp
    "level": "info",                     # info, warn, error, debug, success
    "step": "planning",                  # Current build step type
    "message": "Calling Claude Opus...", # Human-readable log line
    "detail": "model=claude-opus-4-6",   # Optional extra context
    "meta": { ... }                      # Optional structured data (tokens, file sizes, etc.)
}
"""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Dict, List, Optional

logger = logging.getLogger(__name__)


class BuildLogger:
    """Captures structured logs for a single build job and broadcasts them."""

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
        self._current_step = "init"
        self._step_start: float = time.time()
        self._total_tokens = 0
        self._total_cost_usd = 0.0

    @property
    def logs(self) -> List[Dict[str, Any]]:
        return self._logs

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    def set_step(self, step: str):
        """Set the current build step context."""
        self._current_step = step
        self._step_start = time.time()

    async def info(self, message: str, detail: str = "", meta: Optional[Dict] = None):
        await self._log("info", message, detail, meta)

    async def success(self, message: str, detail: str = "", meta: Optional[Dict] = None):
        await self._log("success", message, detail, meta)

    async def warn(self, message: str, detail: str = "", meta: Optional[Dict] = None):
        await self._log("warn", message, detail, meta)

    async def error(self, message: str, detail: str = "", meta: Optional[Dict] = None):
        await self._log("error", message, detail, meta)

    async def debug(self, message: str, detail: str = "", meta: Optional[Dict] = None):
        await self._log("debug", message, detail, meta)

    async def llm_call(
        self,
        model: str,
        purpose: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        duration_s: float = 0.0,
        cost_usd: float = 0.0,
    ):
        """Log an LLM API call with token/cost tracking."""
        total = input_tokens + output_tokens
        self._total_tokens += total
        self._total_cost_usd += cost_usd
        await self._log("info", f"LLM: {purpose}", f"{model} · {total:,} tokens · {duration_s:.1f}s", {
            "type": "llm_call",
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "duration_s": round(duration_s, 2),
            "cost_usd": round(cost_usd, 4),
        })

    async def file_written(self, path: str, size_bytes: int, code: str = ""):
        """Log a file write operation with optional code preview."""
        size_str = f"{size_bytes:,} bytes" if size_bytes < 10000 else f"{size_bytes / 1024:.1f} KB"
        lines = code.split('\n') if code else []
        total_lines = len(lines)
        # Send first 40 lines as preview (enough to see structure)
        preview = '\n'.join(lines[:40]) if lines else ""
        if total_lines > 40:
            preview += f"\n// ... {total_lines - 40} more lines"
        await self._log("info", f"Wrote {path}", size_str, {
            "type": "file_write",
            "path": path,
            "size_bytes": size_bytes,
            "total_lines": total_lines,
            "code_preview": preview,
        })

    async def command_run(self, cmd: str, exit_code: int = 0, duration_s: float = 0.0, output: str = ""):
        """Log a shell command execution."""
        status = "success" if exit_code == 0 else "error"
        truncated = output[:500] if output else ""
        await self._log(status, f"$ {cmd}", f"exit={exit_code} · {duration_s:.1f}s", {
            "type": "command",
            "cmd": cmd,
            "exit_code": exit_code,
            "duration_s": round(duration_s, 2),
            "output": truncated,
        })

    async def _log(self, level: str, message: str, detail: str = "", meta: Optional[Dict] = None):
        """Core logging method — append to buffer + persist + broadcast."""
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "step": self._current_step,
            "message": message,
        }
        if detail:
            entry["detail"] = detail
        if meta:
            entry["meta"] = meta

        # Append to in-memory buffer
        self._logs.append(entry)
        if len(self._logs) > self._max_logs:
            self._logs = self._logs[-self._max_logs:]

        # Also log to Python logger
        py_level = {"info": "info", "success": "info", "warn": "warning", "error": "error", "debug": "debug"}.get(level, "info")
        getattr(logger, py_level)(f"[BUILD:{self._current_step}] {message}" + (f" | {detail}" if detail else ""))

        # Broadcast to mobile via WebSocket
        if self._ws_broadcast:
            try:
                await self._ws_broadcast(self.user_id, {
                    "type": "job_log",
                    "job_id": self.job_id,
                    "entry": entry,
                })
            except Exception:
                pass  # Don't fail the build because WS broadcast failed

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
            logger.warning(f"[BUILD] Failed to persist logs: {e}")

    def summary(self) -> Dict[str, Any]:
        """Return a summary of the build."""
        error_count = sum(1 for l in self._logs if l["level"] == "error")
        warn_count = sum(1 for l in self._logs if l["level"] == "warn")
        llm_calls = [l for l in self._logs if l.get("meta", {}).get("type") == "llm_call"]
        file_writes = [l for l in self._logs if l.get("meta", {}).get("type") == "file_write"]
        commands = [l for l in self._logs if l.get("meta", {}).get("type") == "command"]

        return {
            "total_logs": len(self._logs),
            "errors": error_count,
            "warnings": warn_count,
            "llm_calls": len(llm_calls),
            "total_tokens": self._total_tokens,
            "total_cost_usd": round(self._total_cost_usd, 4),
            "files_written": len(file_writes),
            "commands_run": len(commands),
        }
