"""Generic routine handler — runs an arbitrary user-supplied prompt
against the agent on a schedule and posts the result into Day-as-Chat.

This is the primitive the user actually asked for: "let me give my
agent a recurring task." Examples:
  - "Check my GitHub PRs and tell me which ones have been open >24h"
  - "Look at my calendar for tomorrow and surface conflicts"
  - "Read my Notion 'reading list', pick one item, and summarise it"

`email_briefing` remains as a Gmail-specialised preset for the most
common case. `agent_task` is the catch-all that lets users go beyond
presets.

Execution model:
  1. Run `AgentRunner.run(user_message=prompt_text, channel="routine")`.
     The runner handles tool use (Gmail/Calendar/GitHub via MCP, web
     search, etc.) and returns the assistant response.
  2. The runner ALREADY persists the assistant Message — we stamp
     `channel=routine` + `source=agent_task` via the runner's channel
     param. We don't double-write.
  3. The user-prompt Message ALSO gets `channel=routine`. The frontend
     can choose to hide routine-channel user messages (they're a
     breadcrumb the routine fired, not something the user typed).

Falls back to a plain `internal_llm` call (no tools) if no AgentRunner
is wired into the RoutineRunner — preserves utility for pure-text
prompts even before the runner is fully integrated.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from .base_handler import RoutineResult


logger = logging.getLogger(__name__)


_DEFAULT_SYSTEM_PROMPT = (
    "You are this user's personal agent, running on a scheduled routine. "
    "The user wrote the prompt below ahead of time and won't be there to "
    "clarify — interpret it, use whatever tools you need, and report back "
    "concisely. Keep the response under 300 words unless the task "
    "specifically needs more detail. Markdown supported."
)


class AgentTaskHandler:
    """RoutineHandler for `kind="agent_task"`."""

    kind = "agent_task"

    def __init__(self, agent_runner: Any = None, llm_fn: Any = None, writer: Any = None):
        # Injected at agent_main boot via RoutineRunner.set_agent_runner.
        # None is acceptable (we'll fall back to a no-tools internal_llm
        # call), so handler stays unit-testable without spinning up the
        # full agent stack.
        self._agent_runner = agent_runner
        self._llm_fn = llm_fn
        self._writer = writer

    async def execute(self, routine: Any, run: Any, db: AsyncSession) -> RoutineResult:
        prompt = (routine.prompt_text or "").strip()
        if not prompt:
            return RoutineResult(
                status="failed",
                error_class="no_prompt",
                error_detail="Routine has no prompt_text — nothing to run.",
            )

        runner = self._agent_runner
        if runner is not None and hasattr(runner, "run"):
            return await self._run_via_agent_runner(routine, runner, prompt, db)
        return await self._run_via_internal_llm(routine, prompt, db)

    # ------------------------------------------------------------------
    async def _run_via_agent_runner(
        self, routine: Any, runner: Any, prompt: str, db: AsyncSession,
    ) -> RoutineResult:
        """Full path — uses the agent's normal turn pipeline with tool
        access. The runner persists its own Message (we set channel +
        source via metadata so the frontend can render it as a routine
        post, not a normal assistant reply)."""
        try:
            response = await runner.run(
                user_message=prompt,
                user_id=routine.user_id,
                channel="routine",
                # Don't propagate to other channels — this is a scheduled
                # background turn, not an inbound user message.
                telegram_chat_id=None,
            )
        except Exception as e:
            logger.exception(
                "[agent_task] runner.run failed routine_id=%s err=%s",
                routine.id, e,
            )
            return RoutineResult(
                status="failed",
                error_class=type(e).__name__,
                error_detail=str(e)[:300],
            )

        text = getattr(response, "text", None) or getattr(response, "content", None) or ""
        if not text.strip():
            return RoutineResult(
                status="failed",
                error_class="empty_response",
                error_detail="Agent returned empty response",
            )

        # The agent runner already wrote its own Message rows. Persist a
        # SEPARATE routine-tagged Message so Mission Control can find it
        # by `source=agent_task` without coupling to the runner's
        # internal Conversation. Cheap (one extra row) and the duplication
        # makes the routine output discoverable independent of any
        # changes to the agent runner.
        return await self._persist_and_return(
            routine, db,
            content=text,
            model_used=getattr(response, "model", None),
            emails_fetched=0,
            metrics={"path": "agent_runner", "tokens": getattr(response, "tokens_total", 0)},
        )

    # ------------------------------------------------------------------
    async def _run_via_internal_llm(
        self, routine: Any, prompt: str, db: AsyncSession,
    ) -> RoutineResult:
        """Fallback path — no tools, just a single LLM call. Works for
        pure text tasks ("daily haiku", "translate today's date to 5
        languages") and is always available even if the agent runner
        couldn't be wired.

        Routes through `call_system_llm` so the call honours the user's
        active model (settings.agent_model) + bundle proxy when applicable.
        Pre-refactor we hard-pinned `default_anthropic_model()` here,
        which sent every GPT-5.5 user's fallback to Anthropic. Now the
        provider follows the user's actual preference."""
        llm = self._llm_fn
        if llm is None:
            from app.services.internal_llm import call_system_llm
            llm = call_system_llm

        # Per-routine model override (same contract as email_briefing).
        cfg = routine.config_json or {}
        model_override = (cfg.get("model") or "").strip() or None

        text = await llm(
            user_id=routine.user_id,
            operation_type="system.routine.agent_task",
            model=model_override,
            max_tokens=2000,
            system=_DEFAULT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            timeout=120,
        )
        if not text:
            return RoutineResult(
                status="failed",
                error_class="llm_returned_none",
                error_detail="call_system_llm returned None (timeout / auth / parse)",
            )

        from app.services.model_resolver import default_model
        return await self._persist_and_return(
            routine, db,
            content=text,
            model_used=model_override or default_model(),
            emails_fetched=0,
            metrics={"path": "internal_llm", "summary_chars": len(text)},
        )

    # ------------------------------------------------------------------
    async def _persist_and_return(
        self, routine: Any, db: AsyncSession,
        *,
        content: str,
        model_used: Optional[str],
        emails_fetched: int,
        metrics: dict,
    ) -> RoutineResult:
        writer = self._writer
        if writer is None:
            from .message_writer import write_routine_message, broadcast_routine_message
            writer = write_routine_message
            broadcaster = broadcast_routine_message
        else:
            broadcaster = None

        title = routine.name or f"Routine: {routine.kind}"
        msg_id, day_chat_id = await writer(
            db,
            user_id=routine.user_id,
            content=content,
            source=self.kind,
            routine_id=routine.id,
            title=f"{title} — {datetime.utcnow().date().isoformat()}",
            model_used=model_used,
        )
        if broadcaster is not None:
            from .channel_dispatcher import parse_delivery_channels
            await broadcaster(
                routine.user_id,
                message_id=msg_id,
                day_chat_id=day_chat_id,
                source=self.kind,
                content=content,
                model_used=model_used,
                delivery_channels=parse_delivery_channels(routine.config_json),
                routine_name=routine.name or routine.kind,
            )
        return RoutineResult(
            status="success",
            emails_fetched=emails_fetched,
            summary_message_id=msg_id,
            new_watermark={"last_run_at": datetime.utcnow().isoformat()},
            metrics=metrics,
        )
