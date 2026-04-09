"""
Context Budget Telemetry — writes ContextBudgetLog on every turn that uses the new day-context path.

Written BEFORE the LLM call so we can measure what went into the context window.
Includes summary_was_stale flag for monitoring async summarizer lag.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


async def log_context_budget(
    db: AsyncSession,
    day_chat_id: Optional[str],
    conversation_id: Optional[str],
    user_id: str,
    turn_id: Optional[str],
    system_tokens: int,
    summary_tokens: int,
    history_tokens: int,
    tool_tokens: int,
    memory_tokens: int,
    total_tokens: int,
    model: Optional[str],
    summary_was_stale: bool,
):
    """Write a ContextBudgetLog row. Non-fatal — logs and swallows errors."""
    try:
        from app.db.models.day_chat import ContextBudgetLog

        log_entry = ContextBudgetLog(
            id=str(uuid.uuid4()),
            day_chat_id=day_chat_id,
            conversation_id=conversation_id,
            user_id=user_id,
            turn_id=turn_id,
            system_tokens=system_tokens,
            summary_tokens=summary_tokens,
            history_tokens=history_tokens,
            tool_tokens=tool_tokens,
            memory_tokens=memory_tokens,
            total_tokens=total_tokens,
            model=model,
            summary_was_stale=summary_was_stale,
            created_at=datetime.utcnow(),  # Naive UTC — matches codebase convention
        )
        db.add(log_entry)
        await db.flush()
    except Exception as e:
        logger.warning("[context_budget] Failed to log: %s", e)
