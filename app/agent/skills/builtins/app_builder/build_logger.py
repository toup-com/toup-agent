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

    # Model context windows for token budget tracking
    _MODEL_CONTEXT = {
        "claude-opus-4-6": 200_000,
        "claude-sonnet-4-6": 200_000,
        "claude-sonnet-4-5-20250514": 200_000,
        "claude-sonnet-4-20250514": 200_000,
        "gpt-5.4": 1_000_000,
        "gpt-5.2": 400_000,
        "gpt-5": 128_000,
        "gpt-4o": 128_000,
        "gpt-4o-mini": 128_000,
    }
    _DEFAULT_CONTEXT = 200_000

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
        # Token budget tracking for auto-compaction
        self._session_input_tokens = 0
        self._session_output_tokens = 0
        self._compaction_count = 0
        self._model_context_window = self._DEFAULT_CONTEXT

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

    def set_model(self, model: str):
        """Set the model being used so we can track context budget accurately."""
        self._model_context_window = self._MODEL_CONTEXT.get(model, self._DEFAULT_CONTEXT)

    @property
    def context_usage_pct(self) -> float:
        """Current session token usage as a percentage of model context window."""
        if self._model_context_window <= 0:
            return 0.0
        return (self._session_input_tokens + self._session_output_tokens) / self._model_context_window * 100

    @property
    def needs_compaction(self) -> bool:
        """True when session token usage exceeds 80% of model context window."""
        return self.context_usage_pct >= 80.0

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
        self._session_input_tokens += input_tokens
        self._session_output_tokens += output_tokens

        usage_pct = self.context_usage_pct
        detail = f"{model} · {total:,} tokens · {duration_s:.1f}s"
        if usage_pct >= 50:
            detail += f" · Session: {usage_pct:.0f}%"

        await self._log("info", f"LLM: {purpose}", detail, {
            "type": "llm_call",
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "duration_s": round(duration_s, 2),
            "cost_usd": round(cost_usd, 4),
            "session_usage_pct": round(usage_pct, 1),
        })

        # Broadcast context usage update when above 50%
        if usage_pct >= 50 and self._ws_broadcast:
            try:
                await self._ws_broadcast(self.user_id, {
                    "type": "job_context",
                    "job_id": self.job_id,
                    "usage_pct": round(usage_pct, 1),
                    "total_tokens": self._session_input_tokens + self._session_output_tokens,
                    "context_window": self._model_context_window,
                })
            except Exception:
                pass

    async def file_written(self, path: str, size_bytes: int, code: str = ""):
        """Log a file write operation with full code content."""
        size_str = f"{size_bytes:,} bytes" if size_bytes < 10000 else f"{size_bytes / 1024:.1f} KB"
        total_lines = len(code.split('\n')) if code else 0
        await self._log("info", f"Wrote {path}", size_str, {
            "type": "file_write",
            "path": path,
            "size_bytes": size_bytes,
            "total_lines": total_lines,
            "code_preview": code,
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

    async def compact_session(self, build_state: Dict[str, Any]):
        """
        Auto-compact the build session when context usage is high.

        Resets the session token counters and logs a compaction event.
        The build_state dict should contain current progress info so the
        build can continue seamlessly after compaction.
        """
        self._compaction_count += 1
        old_usage = self.context_usage_pct
        # Reset session counters (the build continues with fresh context)
        self._session_input_tokens = 0
        self._session_output_tokens = 0

        state_summary = (
            f"Compaction #{self._compaction_count}: "
            f"was at {old_usage:.0f}% ({self._total_tokens:,} total tokens used). "
            f"Build step: {build_state.get('current_step', '?')}, "
            f"completed: {', '.join(build_state.get('completed_steps', []))}"
        )

        await self._log("info", f"Auto-compaction #{self._compaction_count} — context reset", state_summary, {
            "type": "compaction",
            "count": self._compaction_count,
            "old_usage_pct": round(old_usage, 1),
            "total_tokens_so_far": self._total_tokens,
            "build_state": build_state,
        })

        # Broadcast compaction event
        if self._ws_broadcast:
            try:
                await self._ws_broadcast(self.user_id, {
                    "type": "job_context",
                    "job_id": self.job_id,
                    "usage_pct": 0.0,
                    "total_tokens": 0,
                    "context_window": self._model_context_window,
                    "compaction_count": self._compaction_count,
                    "message": f"Context compacted (#{self._compaction_count}). Continuing build...",
                })
            except Exception:
                pass

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
