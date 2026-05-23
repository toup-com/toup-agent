"""Shared write path for sub-agent-generated Messages.

Mirrors ``app/agent/routines/message_writer.py`` and
``app/agent/triggers/message_writer.py`` one-for-one — same Conversation
resolution, same DayChat counter bump, same WS broadcast pattern.

Reading-A invariant: a single Conversation per (user, day_chat,
channel="subagent") collects all sub-agent results for the day.
This mirrors what routine fires do — the user sees ONE thread per
day per system channel, not one thread per fire.

The Phase 4 orchestrator calls ``write_subagent_message`` after the
child's ``agent_runner.run()`` returns, then calls
``broadcast_subagent_message`` so any live web/RN/extension client
receives the row in real time. The Telegram fan-out (if the parent
originated on Telegram) is the orchestrator's concern; the helper
here is channel="subagent" only.

What this module does NOT do
----------------------------
- Does not call the LLM (the child run did that and returned the
  final text to the orchestrator).
- Does not write to ``BuildJob.summary_message_id`` — the
  orchestrator does that after this helper returns so the
  Mission Control row deep-links to the assistant message.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, Optional, Tuple

from sqlalchemy.ext.asyncio import AsyncSession


logger = logging.getLogger(__name__)


async def write_subagent_message(
    db: AsyncSession,
    *,
    user_id: str,
    content: str,
    job_id: str,
    parent_job_id: Optional[str] = None,
    label: Optional[str] = None,
    model_used: Optional[str] = None,
    tokens_prompt: Optional[int] = None,
    tokens_completion: Optional[int] = None,
    credit_spent: Optional[float] = None,
    outcome: str = "success",
    tz_override: Optional[str] = None,
    extra_metadata: Optional[dict] = None,
) -> Tuple[str, Optional[str]]:
    """Create the channel="subagent" Conversation + Message pair,
    bump DayChat counters, commit. Returns (message_id, day_chat_id).

    Does NOT broadcast — the caller does that AFTER receiving the id
    so the WS event can carry it. Broadcast is also explicitly
    isolated so a test environment without ws_chat doesn't error
    here. Mirrors the routine/trigger writer pattern exactly.

    ``outcome`` is mirrored into ``Message.metadata_json["outcome"]``
    so the frontend's subagent-card renderer can show a status
    badge (success / failed / timeout / cancelled / budget_exhausted)
    without having to JOIN against build_jobs from the chat fetch.

    ``credit_spent`` is also mirrored into metadata so the user
    sees "this run cost $X" right on the bubble — no extra API
    round-trip needed.
    """
    from app.agent.conversation_resolver import resolve_or_create_day_conversation
    from app.db.message_helpers import resolve_day_chat_id_for_now
    from app.db.models import DayChat, Message

    day_chat_id = await resolve_day_chat_id_for_now(
        db, user_id, tz_override=tz_override,
    )

    convo_meta: dict[str, Any] = {"subagent_job_id": job_id}
    if parent_job_id:
        convo_meta["subagent_parent_job_id"] = parent_job_id
    if label:
        convo_meta["subagent_label"] = label
    if extra_metadata:
        convo_meta.update(extra_metadata)

    # Reading A (Day-as-Chat): one sub-agent Conversation per
    # (user, day_chat, channel="subagent"). All sub-agent results
    # for the day append to the same row, matching the routine
    # pattern. See app/agent/conversation_resolver.py.
    conv = await resolve_or_create_day_conversation(
        db,
        user_id=user_id,
        day_chat_id=day_chat_id,
        channel="subagent",
        title=f"Sub-agent: {label}" if label else "Sub-agent",
        metadata=convo_meta,
    )

    # Per-message metadata — what the frontend reads to render the
    # bubble badge.
    msg_meta: dict[str, Any] = {
        "subagent_job_id": job_id,
        "outcome": outcome,
    }
    if parent_job_id:
        msg_meta["subagent_parent_job_id"] = parent_job_id
    if label:
        msg_meta["subagent_label"] = label
    if credit_spent is not None:
        msg_meta["credit_spent"] = round(float(credit_spent), 4)

    import json as _json
    msg_id = str(uuid.uuid4())
    msg = Message(
        id=msg_id,
        conversation_id=conv.id,
        day_chat_id=day_chat_id,
        role="assistant",
        content=content,
        channel="subagent",
        source="subagent",
        tokens_prompt=tokens_prompt,
        tokens_completion=tokens_completion,
        model_used=model_used,
        metadata_json=_json.dumps(msg_meta),
    )
    db.add(msg)
    await db.flush()

    # DayChat counters — same bookkeeping as agent_runner._save_messages.
    if day_chat_id:
        dc = await db.get(DayChat, day_chat_id)
        if dc:
            dc.message_count = (dc.message_count or 0) + 1
            tt = (tokens_prompt or 0) + (tokens_completion or 0)
            if tt:
                dc.total_tokens = (dc.total_tokens or 0) + tt
            dc.last_message_at = datetime.utcnow()

    await db.commit()
    logger.info(
        "[subagent_writer] wrote job=%s user=%s msg=%s day_chat=%s "
        "outcome=%s tokens=%d",
        job_id, user_id, msg_id, day_chat_id, outcome,
        (tokens_prompt or 0) + (tokens_completion or 0),
    )
    return msg_id, day_chat_id


def build_subagent_ws_event(
    *,
    message_id: str,
    day_chat_id: Optional[str],
    job_id: str,
    parent_job_id: Optional[str] = None,
    label: Optional[str] = None,
    content: str,
    outcome: str = "success",
    model_used: Optional[str] = None,
    credit_spent: Optional[float] = None,
) -> dict[str, Any]:
    """Construct the WS event dict the frontend consumes. Extracted
    as a pure function so the shape can be unit-tested without
    importing ``app.api.ws_chat`` (which transitively pulls in
    FastAPI / Starlette and is brittle to test-env version
    mismatches)."""
    return {
        "type": "message",
        "id": message_id,
        "day_chat_id": day_chat_id,
        "role": "assistant",
        "channel": "subagent",
        "source": "subagent",
        "content": content,
        "model_used": model_used,
        # Subagent-card flag — chat UI parser picks this up to
        # render the sub-agent bubble vs a normal assistant
        # message. Mirrors the "routine_message"/"trigger_message"
        # flags on those writers.
        "subagent_message": True,
        "subagent_job_id": job_id,
        "subagent_parent_job_id": parent_job_id,
        "subagent_label": label,
        "subagent_outcome": outcome,
        "subagent_credit_spent": (
            round(float(credit_spent), 4)
            if credit_spent is not None else None
        ),
        "created_at": datetime.utcnow().isoformat(),
    }


async def broadcast_subagent_message(
    user_id: str,
    *,
    message_id: str,
    day_chat_id: Optional[str],
    job_id: str,
    parent_job_id: Optional[str] = None,
    label: Optional[str] = None,
    content: str,
    outcome: str = "success",
    model_used: Optional[str] = None,
    credit_spent: Optional[float] = None,
) -> dict[str, Any]:
    """Push the sub-agent Message to any live WS clients (web, RN,
    extension). Returns {"ws_count": int, "channel_results": ...}
    matching the routine broadcast shape.

    Telegram fan-out (when the parent was on Telegram) is handled
    by the orchestrator, not here — the writer is channel="subagent"
    only and decoupled from delivery surfaces.
    """
    channel_results: dict[str, dict[str, Any]] = {}
    event = build_subagent_ws_event(
        message_id=message_id, day_chat_id=day_chat_id, job_id=job_id,
        parent_job_id=parent_job_id, label=label, content=content,
        outcome=outcome, model_used=model_used, credit_spent=credit_spent,
    )

    try:
        from app.api.ws_chat import broadcast_to_user
    except Exception as e:  # pragma: no cover — defensive
        logger.debug(
            "[subagent_writer] broadcast skipped — ws_chat unavailable: %s", e,
        )
        ws_count = 0
    else:
        try:
            ws_count = await broadcast_to_user(user_id, event)
        except Exception as e:  # pragma: no cover — defensive
            logger.warning("[subagent_writer] broadcast failed: %s", e)
            ws_count = 0

    # Website always counts as delivered when the Message row landed —
    # ws_count==0 just means the user wasn't viewing live; the row
    # is in the DB and shows up on next /messages/since fetch.
    channel_results["website"] = {
        "status": "delivered",
        "message_id": message_id,
        "day_chat_id": day_chat_id,
        "ws_count": ws_count,
    }

    return {"ws_count": ws_count, "channel_results": channel_results}
