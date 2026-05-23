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

import json
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

        # Credit pre-flight: skip cleanly when the user is out. See
        # email_briefing_handler.py for the rationale.
        try:
            from app.services.credit_reporter import raise_if_exhausted
            from app.services.credit_exhausted import OutOfCreditsError
            try:
                raise_if_exhausted()
            except OutOfCreditsError as _oce:
                return RoutineResult(
                    status="skipped",
                    error_class="insufficient_credits",
                    error_detail=_oce.response.message[:300],
                )
        except ImportError:
            pass

        runner = self._agent_runner
        # Phase 8: extract the BuildJob id from the run shim if the
        # caller provided one. Powers parent_job_id linkage on any
        # sub-agent the routine's agent run spawns during its turn.
        job_id = getattr(run, "job_id", None) or getattr(run, "id", None)
        if runner is not None and hasattr(runner, "run"):
            return await self._run_via_agent_runner(
                routine, runner, prompt, db, current_job_id=job_id,
            )
        return await self._run_via_internal_llm(routine, prompt, db)

    # ------------------------------------------------------------------
    async def _run_via_agent_runner(
        self, routine: Any, runner: Any, prompt: str, db: AsyncSession,
        *, current_job_id: Optional[str] = None,
    ) -> RoutineResult:
        """Full path — uses the agent's normal turn pipeline with tool
        access. AgentRunner persists its own assistant Message; we reuse
        that row (stamping `source` + routine metadata onto it) instead
        of writing a duplicate."""
        try:
            response = await runner.run(
                user_message=prompt,
                user_id=routine.user_id,
                channel="routine",
                current_job_id=current_job_id,
                # Don't propagate to other channels — this is a scheduled
                # background turn, not an inbound user message.
                telegram_chat_id=None,
                # CRITICAL — do NOT persist the routine's prompt_text as
                # a user Message in the day-chat. Pre-2026-05-12 the
                # runner saved every `user_message=` with role="user"
                # by default, which made the SYSTEM-generated routine
                # prompt ("Every day at 1:21 PM Toronto time, fetch the
                # user's latest 5 Gmail messages…") appear in chat as
                # if the user had typed it. The user never wrote that
                # text — it's the routine's internal instruction, not
                # conversational content. The assistant's reply still
                # saves (channel=routine), which is what the user
                # actually wants to see.
                save_user_message=False,
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

        # Best-effort tool-invocation capture for `tools_invoked_json`
        # (Ticket 2.5). The AgentRunner response shape evolves; we look
        # for a handful of plausible attributes and fall back to empty.
        tools_used: list[str] = []
        for attr in ("tools_invoked", "tool_calls", "tools"):
            raw = getattr(response, attr, None)
            if raw:
                for entry in raw:
                    if isinstance(entry, str):
                        tools_used.append(entry)
                    elif isinstance(entry, dict):
                        name = entry.get("name") or entry.get("tool")
                        if name:
                            tools_used.append(name)
                    else:
                        n = getattr(entry, "name", None)
                        if n:
                            tools_used.append(n)
                break  # first attribute that exists wins

        # Reuse the Message AgentRunner already persisted, instead of
        # writing a second row. Pre-fix (≤2026-05-17) this handler
        # unconditionally called `_persist_and_return` here on top of the
        # runner's own `_save_messages`, producing TWO identical
        # `role=assistant channel=routine` bubbles in the user's day-chat
        # on every agent_task fire — the bug visible in the user-report
        # screenshots where one Gmail summary appeared twice.
        #
        # The runner exposes its row id on `AgentResponse.asst_message_id`
        # (added by the 2026-05-10 "thread server message id through `done`"
        # fix). When present, decorate that row with routine metadata +
        # broadcast + fan out — one row, no duplicate. When absent (older
        # response shape, or a unit-test mock), fall back to the legacy
        # write path so test stubs that don't model `asst_message_id`
        # keep working.
        metrics = {
            "path": "agent_runner",
            "tokens": getattr(response, "tokens_total", 0),
        }
        asst_id = (getattr(response, "asst_message_id", "") or "").strip()
        if asst_id:
            return await self._decorate_existing_and_fanout(
                routine, db,
                existing_msg_id=asst_id,
                content=text,
                model_used=getattr(response, "model", None),
                tools_invoked=tools_used,
                metrics=metrics,
            )

        return await self._persist_and_return(
            routine, db,
            content=text,
            model_used=getattr(response, "model", None),
            emails_fetched=0,
            tools_invoked=tools_used,
            metrics=metrics,
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

        # Per-routine model with default-cheap policy (2026-05-13 cost
        # audit, same contract as email_briefing). Default to
        # gpt-4o-mini when no override; power users set
        # `config_json.model` to upgrade. Routines are user-scheduled;
        # defaulting to gpt-4o-mini makes a "10 routines firing per day"
        # footprint predictable instead of "10× your chat-model rate".
        cfg = routine.config_json or {}
        model_choice = (cfg.get("model") or "").strip() or "gpt-4o-mini"

        text = await llm(
            user_id=routine.user_id,
            operation_type="user.routine.agent_task",
            model=model_choice,
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

        return await self._persist_and_return(
            routine, db,
            content=text,
            model_used=model_choice,
            emails_fetched=0,
            metrics={"path": "internal_llm", "summary_chars": len(text)},
        )

    # ------------------------------------------------------------------
    async def _decorate_existing_and_fanout(
        self, routine: Any, db: AsyncSession,
        *,
        existing_msg_id: str,
        content: str,
        model_used: Optional[str],
        tools_invoked: Optional[list[str]],
        metrics: dict,
    ) -> RoutineResult:
        """Reuse the assistant Message that AgentRunner already persisted.

        Stamps `source = self.kind` and the routine identity on the row
        so Mission Control filtering on `source=agent_task` finds it
        (the runner doesn't set `source` by default — that field is
        reserved for non-user-originated message kinds). Then broadcasts
        the row over the user's WS connections so the chat UI renders
        the routine card in real time, and fans out to extra delivery
        channels (Telegram / WhatsApp) per the routine's config.

        No new Message row is written here. That's the entire point of
        this path — one routine fire produces exactly one assistant
        bubble.
        """
        from app.db.models import Message

        msg_day_chat_id: Optional[str] = None
        msg_row_present = False
        try:
            msg = await db.get(Message, existing_msg_id)
            if msg is not None:
                msg_row_present = True
                msg_day_chat_id = msg.day_chat_id
                if not msg.source:
                    msg.source = self.kind
                existing_meta: dict = {}
                if msg.metadata_json:
                    try:
                        parsed = json.loads(msg.metadata_json)
                        if isinstance(parsed, dict):
                            existing_meta = parsed
                    except (json.JSONDecodeError, TypeError):
                        # Corrupt JSON — overwrite rather than crash. The
                        # routine card needs the flag more than we need
                        # to preserve unparseable bytes.
                        existing_meta = {}
                existing_meta["routine_message"] = True
                existing_meta["routine_id"] = routine.id
                existing_meta["routine_name"] = routine.name or routine.kind
                msg.metadata_json = json.dumps(existing_meta)
                await db.commit()
            else:
                logger.warning(
                    "[agent_task] decorate skipped — Message %s vanished "
                    "between runner save and routine decorate",
                    existing_msg_id,
                )
        except Exception as e:
            logger.warning(
                "[agent_task] decorate_existing_message failed msg_id=%s err=%s",
                existing_msg_id, e,
            )

        # WS broadcast + extra-channel fan-out. AgentRunner does NOT
        # broadcast on its own when run from a routine (no WS callbacks
        # wired by the handler), so this is the first time the user's
        # active chat clients hear about the new row.
        channel_results: dict[str, dict[str, Any]] = {}
        ws_count = 0
        try:
            from app.api.ws_chat import broadcast_to_user
        except Exception as e:
            logger.debug(
                "[agent_task] WS broadcast skipped — ws_chat unavailable: %s", e,
            )
        else:
            event = {
                "type": "message",
                "id": existing_msg_id,
                "day_chat_id": msg_day_chat_id,
                "role": "assistant",
                "channel": "routine",
                "source": self.kind,
                "content": content,
                "model_used": model_used,
                "routine_message": True,
                "created_at": datetime.utcnow().isoformat(),
            }
            try:
                ws_count = await broadcast_to_user(routine.user_id, event)
            except Exception as e:
                logger.warning("[agent_task] WS broadcast failed: %s", e)

        channel_results["website"] = {
            "status": "delivered" if msg_row_present else "failed",
            "message_id": existing_msg_id,
            "day_chat_id": msg_day_chat_id,
            "ws_count": ws_count,
        }

        from .channel_dispatcher import (
            deliver_to_extra_channels_detailed,
            parse_delivery_channels,
        )
        delivery_channels = parse_delivery_channels(routine.config_json)
        extras = [c for c in delivery_channels if c and c != "website"]
        if extras:
            from app.db.database import async_session_maker
            try:
                extras_results = await deliver_to_extra_channels_detailed(
                    user_id=routine.user_id,
                    delivery_channels=extras,
                    routine_name=routine.name or routine.kind,
                    content=content,
                    db_session_maker=async_session_maker,
                )
                if extras_results:
                    channel_results.update(extras_results)
            except Exception as e:
                logger.warning(
                    "[agent_task] extra channel dispatch failed: %s: %s",
                    type(e).__name__, e,
                )

        return RoutineResult(
            status="success",
            emails_fetched=0,
            summary_message_id=existing_msg_id,
            new_watermark={"last_run_at": datetime.utcnow().isoformat()},
            channel_results=channel_results,
            tools_invoked=list(tools_invoked or []),
            metrics=metrics,
        )

    # ------------------------------------------------------------------
    async def _persist_and_return(
        self, routine: Any, db: AsyncSession,
        *,
        content: str,
        model_used: Optional[str],
        emails_fetched: int,
        metrics: dict,
        tools_invoked: Optional[list[str]] = None,
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
            extra_metadata={"routine_message": True, "routine_id": routine.id,
                            "routine_name": routine.name or routine.kind},
        )

        # Ticket 2.5 — capture per-channel delivery confirmations so the
        # runner can downgrade outcome to `partial` when a configured
        # channel silently skipped (same Defect A contract as
        # email_briefing). Without this, agent_task routines have the
        # same silent-success bug Phase 2 was designed to close.
        channel_results: dict[str, dict[str, Any]] = {}
        if broadcaster is not None:
            from .channel_dispatcher import parse_delivery_channels
            broadcast_out = await broadcaster(
                routine.user_id,
                message_id=msg_id,
                day_chat_id=day_chat_id,
                source=self.kind,
                content=content,
                model_used=model_used,
                delivery_channels=parse_delivery_channels(routine.config_json),
                routine_name=routine.name or routine.kind,
            )
            if isinstance(broadcast_out, dict):
                channel_results = broadcast_out.get("channel_results", {}) or {}
        return RoutineResult(
            status="success",
            emails_fetched=emails_fetched,
            summary_message_id=msg_id,
            new_watermark={"last_run_at": datetime.utcnow().isoformat()},
            channel_results=channel_results,
            tools_invoked=list(tools_invoked or []),
            metrics=metrics,
        )
