"""Reminder handler — routine kind #3.

The third RoutineHandler kind, alongside `email_briefing` (Gmail
summary preset) and `agent_task` (generic LLM-driven scheduled work).

A reminder is the cheapest, simplest routine shape:
  • No LLM call. No MCP tool. No retry loop necessary (text delivery
    is deterministic).
  • Writes the literal `routine.reminder_text` into Day-as-Chat as
    role=assistant, channel=routine, source=reminder.
  • Fans out to every configured delivery_channels entry the same way
    `email_briefing` does (Ticket 2.5 contract).

Phase A of the CronJob → Routine consolidation: this replaces the
"agent posts a quick scheduled message to Telegram" use case that
CronJob previously owned, but now in the Day-as-Chat model with
multi-channel delivery.

Schedule shapes:
  • schedule_kind='cron'  — daily reminder at HH:MM ("take vitamins
    every morning at 9").
  • schedule_kind='at'    — one-shot at a specific datetime ("remind
    me to call mom at 5pm today"). The runner auto-disables the
    routine in _post_terminal because `auto_disable_after_fire=true`
    is set server-side for this shape.
  • schedule_kind='every' — interval reminder ("ping me every 30
    minutes between 9 and 5 to stand up"). Window gating is handled
    in `RoutineRunner._fire` BEFORE the handler runs, so this handler
    only sees in-window fires.

Idempotency:
  • Daily cron reminders: protected by the
    UNIQUE (routine_id, scheduled_for_local_date) gate on routine_runs.
  • One-shot 'at': only one routine_run can ever exist (auto-disable
    prevents a re-fire), so dedupe is structural.
  • Interval 'every': the in-_fire minute gate prevents the same minute
    from firing twice across blue-green container swaps.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from .base_handler import RoutineResult


logger = logging.getLogger(__name__)


class ReminderHandler:
    """RoutineHandler for `kind="reminder"`."""

    kind = "reminder"

    def __init__(self, writer: Any = None):
        # Writer is injectable for tests. In prod we use the canonical
        # `message_writer.write_routine_message` + `broadcast_routine_message`.
        self._writer = writer

    async def execute(self, routine: Any, run: Any, db: AsyncSession) -> RoutineResult:
        text = (routine.reminder_text or "").strip()
        if not text:
            # The CHECK constraint (mig 042) should prevent this from
            # landing in the DB, but defend in depth — a misconfigured
            # row shouldn't crash the runner.
            return RoutineResult(
                status="failed",
                error_class="empty_reminder",
                error_detail="reminder_text is null/blank — cannot deliver",
            )

        writer = self._writer
        if writer is None:
            from .message_writer import write_routine_message, broadcast_routine_message
            writer = write_routine_message
            broadcaster = broadcast_routine_message
        else:
            broadcaster = None  # tests inject writer + skip broadcast

        try:
            msg_id, day_chat_id = await writer(
                db,
                user_id=routine.user_id,
                content=text,
                source=self.kind,
                routine_id=routine.id,
                title=routine.name or "Reminder",
                model_used=None,
                # Token counts irrelevant — no LLM call.
                tokens_prompt=None,
                tokens_completion=None,
                extra_metadata={
                    "routine_message": True,
                    "routine_id": routine.id,
                    "routine_name": routine.name or "Reminder",
                    "reminder": True,
                },
            )
        except Exception as e:
            logger.exception(
                "[reminder] write failed routine_id=%s err=%s",
                routine.id, e,
            )
            return RoutineResult(
                status="failed",
                error_class=type(e).__name__,
                error_detail=str(e)[:300],
            )

        # Fan out to telegram / whatsapp / etc. Same contract as
        # email_briefing handler — broadcaster returns the structured
        # `{ws_count, channel_results}` dict; the runner uses
        # channel_results to derive outcome (success vs partial).
        channel_results: dict[str, dict[str, Any]] = {}
        if broadcaster is not None:
            from .channel_dispatcher import parse_delivery_channels
            broadcast_out = await broadcaster(
                routine.user_id,
                message_id=msg_id,
                day_chat_id=day_chat_id,
                source=self.kind,
                content=text,
                model_used=None,
                delivery_channels=parse_delivery_channels(routine.config_json),
                routine_name=routine.name or "Reminder",
            )
            if isinstance(broadcast_out, dict):
                channel_results = broadcast_out.get("channel_results", {}) or {}

        logger.info(
            "[reminder] delivered routine_id=%s user_id=%s msg_id=%s chars=%d "
            "channels=%s",
            routine.id, routine.user_id, msg_id, len(text),
            list(channel_results.keys()),
        )

        return RoutineResult(
            status="success",
            emails_fetched=0,  # not applicable for reminders
            summary_message_id=msg_id,
            # Reminders don't advance a watermark — there's no upstream
            # cursor to track. Leave None so `_post_terminal`'s
            # else-branch stamps last_status without watermark advance.
            new_watermark=None,
            channel_results=channel_results,
            tools_invoked=[],  # no MCP tools used
            metrics={"reminder_chars": len(text)},
        )
