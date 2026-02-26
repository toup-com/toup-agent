"""
Sub-Agent Manager — Background task system for HexBrain.

The agent can spawn independent background tasks that:
- Run with their own isolated session
- Optionally use a different model
- Have a configurable timeout
- Report results back to the user via Telegram when done
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SubAgentRun:
    """A single sub-agent background task."""

    id: str
    task: str
    label: str
    user_id: str
    telegram_chat_id: int
    model: Optional[str] = None
    started_at: datetime = field(default_factory=datetime.utcnow)
    finished_at: Optional[datetime] = None
    status: str = "running"  # running, done, error, timeout, cancelled
    result: Optional[str] = None
    tokens_used: int = 0
    _task_handle: Optional[asyncio.Task] = field(default=None, repr=False)


class SubAgentManager:
    """Manages background sub-agent tasks."""

    def __init__(self):
        self._agent_runner = None
        self._telegram_bot = None
        # All runs (active + recent completed)
        self._runs: Dict[str, SubAgentRun] = {}
        # Keep at most N completed runs in memory
        self._max_history = 50

    def set_agent_runner(self, agent_runner):
        self._agent_runner = agent_runner

    def set_bot(self, telegram_bot):
        self._telegram_bot = telegram_bot

    async def spawn(
        self,
        task: str,
        user_id: str,
        telegram_chat_id: int,
        label: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: int = 300,
    ) -> Dict[str, Any]:
        """Spawn a background task. Returns run info dict."""
        if not self._agent_runner:
            return {"error": "Agent runner not available"}

        run_id = str(uuid.uuid4())[:8]
        run = SubAgentRun(
            id=run_id,
            task=task,
            label=label or f"task-{run_id}",
            user_id=user_id,
            telegram_chat_id=telegram_chat_id,
            model=model,
        )
        self._runs[run_id] = run

        # Launch in background
        task_handle = asyncio.create_task(self._execute(run, timeout_seconds))
        run._task_handle = task_handle

        logger.info(f"[SUBAGENT] Spawned '{run.label}' (id={run_id}) for user {user_id}")

        return {
            "id": run_id,
            "label": run.label,
            "status": "running",
            "model": model or "default",
            "timeout": timeout_seconds,
        }

    async def cancel(self, run_id: str, user_id: str) -> Dict[str, Any]:
        """Cancel a running sub-agent task."""
        run = self._runs.get(run_id)
        if not run:
            return {"error": f"Run not found: {run_id}"}
        if run.user_id != user_id:
            return {"error": "Not your task"}
        if run.status != "running":
            return {"error": f"Task already {run.status}"}

        if run._task_handle and not run._task_handle.done():
            run._task_handle.cancel()
        run.status = "cancelled"
        run.finished_at = datetime.utcnow()

        return {"id": run_id, "label": run.label, "status": "cancelled"}

    def list_runs(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all runs, optionally filtered by user."""
        runs = list(self._runs.values())
        if user_id:
            runs = [r for r in runs if r.user_id == user_id]

        return [
            {
                "id": r.id,
                "label": r.label,
                "status": r.status,
                "model": r.model or "default",
                "started_at": r.started_at.isoformat(),
                "finished_at": r.finished_at.isoformat() if r.finished_at else None,
                "tokens_used": r.tokens_used,
                "result_preview": (r.result[:200] + "...") if r.result and len(r.result) > 200 else r.result,
            }
            for r in sorted(runs, key=lambda x: x.started_at, reverse=True)
        ]

    def get_run(self, run_id: str) -> Optional[SubAgentRun]:
        return self._runs.get(run_id)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _execute(self, run: SubAgentRun, timeout_seconds: int):
        """Execute a sub-agent task in the background."""
        try:
            response = await asyncio.wait_for(
                self._agent_runner.run(
                    user_message=(
                        f"[Background Task: {run.label}]\n\n"
                        f"TASK: {run.task}\n\n"
                        "Complete this task fully. Provide a clear summary of what you did and the results."
                    ),
                    user_id=run.user_id,
                    telegram_chat_id=run.telegram_chat_id,
                    model_override=run.model,
                ),
                timeout=timeout_seconds,
            )
            run.status = "done"
            run.result = response.text
            run.tokens_used = response.tokens_total

        except asyncio.TimeoutError:
            run.status = "timeout"
            run.result = f"Task timed out after {timeout_seconds}s."

        except asyncio.CancelledError:
            run.status = "cancelled"
            run.result = "Task was cancelled."
            return  # Don't announce cancellations

        except Exception as e:
            run.status = "error"
            run.result = f"Error: {type(e).__name__}: {str(e)[:500]}"
            logger.exception(f"[SUBAGENT] Task '{run.label}' failed")

        run.finished_at = datetime.utcnow()

        # Announce result to user via Telegram
        await self._announce_result(run)

        # Prune old completed runs
        self._prune_history()

    async def _announce_result(self, run: SubAgentRun):
        """Send the sub-agent result back to the user, split across multiple messages if needed."""
        if not self._telegram_bot or not self._telegram_bot.app:
            return

        try:
            from app.agent.streaming import postprocess_for_telegram, split_message

            bot = self._telegram_bot.app.bot

            status_emoji = {
                "done": "✅",
                "error": "❌",
                "timeout": "⏰",
            }.get(run.status, "📋")

            result_text = run.result or "(no output)"
            result_text = postprocess_for_telegram(result_text)

            header = (
                f"{status_emoji} <b>Background task '{run.label}'</b> — {run.status}\n"
                f"Tokens: {run.tokens_used:,}\n\n"
            )

            full_text = header + result_text
            chunks = split_message(full_text, 4096)
            for chunk in chunks:
                await bot.send_message(
                    chat_id=run.telegram_chat_id,
                    text=chunk,
                    parse_mode="HTML",
                )
        except Exception as e:
            logger.warning(f"[SUBAGENT] Failed to announce result: {e}")
            # Try plain text fallback — still split instead of truncating
            try:
                from app.agent.streaming import split_message
                bot = self._telegram_bot.app.bot
                fallback = f"📋 Background task '{run.label}' {run.status}:\n\n{run.result or '(no output)'}"
                for chunk in split_message(fallback, 4096):
                    await bot.send_message(
                        chat_id=run.telegram_chat_id,
                        text=chunk,
                    )
            except Exception:
                pass

    def _prune_history(self):
        """Remove old completed runs to keep memory bounded."""
        completed = [
            r for r in self._runs.values() if r.status != "running"
        ]
        if len(completed) > self._max_history:
            # Sort by finished_at, remove oldest
            completed.sort(key=lambda x: x.finished_at or x.started_at)
            for r in completed[: len(completed) - self._max_history]:
                self._runs.pop(r.id, None)
