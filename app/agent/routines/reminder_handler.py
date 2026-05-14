"""Reminder handler — kind='reminder'.

Text-only routine: writes `routine.reminder_text` verbatim into
Day-as-Chat as `role=assistant, channel=routine, source=reminder`,
fans out to every configured delivery channel. No LLM call, no MCP
tools, no retry-eligible failures (delivery is deterministic).

Phase A of the CronJob → Routine consolidation. Replaces the
"agent posts a quick scheduled message to Telegram" use case with a
multi-channel (website + Telegram + WhatsApp) delivery model.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from .base_handler import RoutineResult


logger = logging.getLogger(__name__)


class ReminderHandler:
    """RoutineHandler implementation for `kind="reminder"`."""

    kind = "reminder"

    def __init__(self, writer: Any = None):
        self._writer = writer

    async def execute(self, routine: Any, run: Any, db: AsyncSession) -> RoutineResult:
        text = (routine.reminder_text or "").strip()
        if not text:
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
            broadcaster = None

        try:
            msg_id, day_chat_id = await writer(
                db,
                user_id=routine.user_id,
                content=text,
                source=self.kind,
                routine_id=routine.id,
                title=routine.name or "Reminder",
                model_used=None,
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
            emails_fetched=0,
            summary_message_id=msg_id,
            new_watermark=None,
            channel_results=channel_results,
            tools_invoked=[],
            metrics={"reminder_chars": len(text)},
        )
