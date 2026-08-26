"""Confirm-mode surfaces for automation runs (Round 29).

A confirm-grant write does not execute at flush time: the platform
stages a `ConnectorPendingAction` and answers `confirmation_required`,
and the run parks (`waiting_on_user`). Round 28 left that park mute —
no card in the session thread, no `error_class`, so the reaper's
card-park sweep (which matches `ERR_AWAITING_CONFIRMATION`) could never
see an automation park and the 25-hour backstop outbox.py claimed did
not actually apply.

This module is the session-side voice of the park, plus its terminals:

  - The pending-action card reuses the SAME `pending_action` metadata
    key + WS frame the elevated chat-tool path mints (tool_executor),
    so both clients render it with the renderer they already have.
    `automation_id`/`run_id` ride as additive keys.
  - Status flips (approved/rejected/expired/failed) re-persist onto the
    SAME message row and re-broadcast — the cards.py upsert contract.
  - Terminals route through `executor._finalize_job` — the exactly-once
    rowcount gate every run terminal rides (CONTRACTS-R29 §2): a card
    the user rejected or let expire closes the run as
    `status="cancelled", outcome="skipped"` with copy that tells the
    "nothing was sent" truth. Only an actually-executed or
    actually-failed send touches the health streak — a user decision is
    not an automation failure.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Optional

logger = logging.getLogger(__name__)

#: Card outcome → (job status, run outcome) for automation runs. The
#: generic chat mapping (api/agent.py) says expired→cancelled with no
#: outcome; a run's vocabulary is richer — `skipped` is the honest word
#: for "the user did not let this happen" (tone: warn, never err).
ACTION_OUTCOME_TO_RUN_TERMINAL: dict[str, tuple[str, str]] = {
    "executed": ("completed", "sent"),
    "failed": ("failed", "write_failed"),
    "rejected": ("cancelled", "skipped"),
    "expired": ("cancelled", "skipped"),
}

#: Card status the message payload shows for each resolution.
_OUTCOME_TO_CARD_STATUS = {
    "executed": "approved",
    "failed": "failed",
    "rejected": "rejected",
    "expired": "expired",
}

SKIPPED_REJECTED_COPY = "You didn't approve this, so nothing was sent."
SKIPPED_EXPIRED_COPY = (
    "The approval expired before you confirmed it, so nothing was sent."
)

#: Backstop park age when the platform never told us the card's TTL —
#: mirrors job_reaper.PARKED_ON_CARD_STALE_AFTER for the chat parks.
PARK_EXPIRY_FALLBACK = timedelta(hours=25)
#: Grace past the card's own expiry before the sweep closes the run —
#: the platform's lazy-expiry hop should normally win this race.
PARK_EXPIRY_GRACE = timedelta(minutes=10)


def pending_card_payload(
    *,
    action_id: str,
    connector_id: str,
    tool_name: str,
    summary: str,
    payload: Optional[dict],
    expires_at: Optional[str],
    automation_id: str,
    job_id: str,
    status: str = "pending",
) -> dict:
    """The chat card shape (tool_executor's), plus additive run keys."""
    return {
        "action_id": action_id,
        "connector_id": connector_id,
        "tool_name": tool_name,
        "summary": summary or "",
        "payload": payload or {},
        "expires_at": expires_at,
        "status": status,
        "automation_id": automation_id,
        "run_id": job_id,
    }


async def write_pending_card(
    db,
    *,
    automation,
    job_id: str,
    card: dict,
) -> Optional[str]:
    """Broadcast the live pending frame. Best-effort — the park never
    depends on its card.

    CONTRACTS-R31 §4.1: the DURABLE half of a park is the `waiting`
    turn in the automation's thread (written by the caller, R30 §11),
    not a day-chat Message. This function used to write one, carrying
    `**{name}** staged an action…` — raw markdown bold, in the main
    chat, for an automation the main chat should not be narrating.

    The frame still goes out so a live client repaints at once; it
    carries `automation_id` and no `message_id`, because there is no
    message to address any more.
    """
    del db
    await _broadcast_pending(automation.user_id, card, message_id=None)
    return None


async def update_pending_card(
    db,
    *,
    user_id: str,
    status: str,
    automation_id: Optional[str] = None,
    action_id: Optional[str] = None,
    message_id: Optional[str] = None,
) -> None:
    """Record the decision where the card lives — the THREAD.

    CONTRACTS-R31 §4.1. This used to flip a day-chat Message's metadata
    in place; there is no such message any more, so the surface that
    told the user "Nothing happens until you approve." would have said
    it forever, whatever they then pressed. A card that cannot report
    the decision it asked for is worse than no card.

    `message_id` is accepted and ignored so a caller carrying an old
    `config_json.pending_card_message_id` (there are runs with one) does
    not have to be found first.
    """
    del message_id
    if not automation_id or not action_id:
        return
    from . import ledger as _ledger
    from app.db.models import AutomationTurn

    try:
        thread = await _ledger.thread_for(db, automation_id)
        if thread is None:
            return
        turns, _more = await _ledger.list_turns(
            db, thread_id=thread.id, limit=200,
        )
        target = None
        for t in turns:
            if t.get("kind") == "waiting" \
                    and t.get("pending_action_id") == action_id:
                target = t
        if target is None:
            return
        # The waiting turn BECOMES the record of what was decided. It is
        # replaced rather than appended to, so the thread does not carry
        # a standing "waiting" card under the answer to it (§4.5's
        # replacement-by-turn_id, the same seam the per-source merge
        # uses).
        await _ledger.replace_turn(
            db, user_id=user_id, thread=thread, turn_id=target["id"],
            kind="agent",
            payload={"text": _DECISION_TEXT.get(
                status, "That is decided.")},
        )
        await _broadcast_pending(
            user_id,
            {"action_id": action_id, "automation_id": automation_id,
             "status": status},
            message_id=None,
        )
    except Exception as e:  # noqa: BLE001 — the card is a companion
        logger.warning(
            "[automations] pending card update failed action=%s: %s",
            str(action_id)[:16], e,
        )
        try:
            await db.rollback()
        except Exception:  # noqa: BLE001
            pass


# What the thread says once a park is decided. Plain sentences, the
# user's vocabulary, and no status word doing the work of a sentence.
_DECISION_TEXT = {
    "approved": "You approved it, so I went ahead.",
    "executed": "You approved it, so I went ahead.",
    "rejected": "You said no, so nothing was sent.",
    "skipped": "You said no, so nothing was sent.",
    "expired": "You did not get to this in time, so nothing was sent.",
    "failed": "You approved it, but the change did not go through.",
}


async def _broadcast_pending(user_id: str, card: dict,
                             *, message_id: Optional[str]) -> None:
    """Same frame type as the chat path (`pending_action`, card keys at
    top level, NO channel key) so both clients reuse their renderer."""
    try:
        from app.api.ws_chat import broadcast_to_user
        await broadcast_to_user(user_id, {
            "type": "pending_action",
            "message_id": message_id,
            **card,
        })
    except Exception as e:  # noqa: BLE001 — no live socket is normal
        logger.debug("[automations] pending card broadcast skipped: %s", e)


async def resolve_parked_run(
    db,
    job,
    *,
    outcome: str,
    detail: Optional[str] = None,
) -> bool:
    """Terminalize one parked automation run for a decided card.

    Routes through `_finalize_job` (the rowcount gate: last-outcome
    stamp + noteworthy push ride it, and a replay is a no-op), flips
    the session card, and touches health only for executed/failed.
    Returns True when this call owned the terminal.
    """
    terminal = ACTION_OUTCOME_TO_RUN_TERMINAL.get(outcome)
    if terminal is None:
        return False
    status, run_outcome = terminal

    from app.agent.job_status import STATUS_WAITING_ON_USER
    from .executor import _finalize_job, _record_health

    if getattr(job, "status", None) != STATUS_WAITING_ON_USER:
        # A replayed delivery: the finalize gate would rowcount-0 the
        # job anyway, but the card and health writes below are not
        # gated — refuse here so a replay touches NOTHING.
        return False

    cfg = dict(job.config_json or {}) if isinstance(job.config_json, dict) else {}
    if status == "completed" and cfg.get("steps_partial"):
        # The v2 skip-tolerant nuance — outbox._finalize_run applies the
        # same downgrade on the direct path.
        run_outcome = "partial"

    if outcome == "failed":
        user_message = (detail or "The write failed.")[:300]
        error_class: Optional[str] = "tool_error"
    elif outcome == "rejected":
        user_message, error_class = SKIPPED_REJECTED_COPY, None
    elif outcome == "expired":
        user_message, error_class = SKIPPED_EXPIRED_COPY, None
    else:
        user_message, error_class = None, None

    await _finalize_job(
        db, job.id, status=status, outcome=run_outcome,
        error_class=error_class, user_message=user_message,
    )

    await update_pending_card(
        db,
        user_id=job.user_id,
        automation_id=job.source_id,
        action_id=cfg.get("pending_action_id"),
        status=_OUTCOME_TO_CARD_STATUS.get(outcome, outcome),
    )

    automation_id = job.source_id
    if automation_id and outcome in ("executed", "failed"):
        # rejected/expired are the USER's decision — an automation that
        # keeps asking and keeps being ignored must not auto-pause.
        await _record_health(
            db, automation_id,
            ok=(outcome == "executed"),
            error=None if outcome == "executed" else (detail or "write failed"),
        )
    return True


def park_expiry_deadline(job, now: Optional[datetime] = None) -> datetime:
    """When the sweep may close this park: the card's own expiry
    (stamped at park time) plus grace, else the 25-hour fallback."""
    cfg = job.config_json if isinstance(job.config_json, dict) else {}
    raw = (cfg or {}).get("pending_action_expires_at")
    if raw:
        try:
            parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            if parsed.tzinfo is not None:
                parsed = parsed.replace(tzinfo=None)
            return parsed + PARK_EXPIRY_GRACE
        except (ValueError, TypeError):
            pass
    return (job.created_at or (now or datetime.utcnow())) + PARK_EXPIRY_FALLBACK


async def sweep_expired_confirm_parks(now: Optional[datetime] = None) -> int:
    """Agent-side backstop for parks the platform's lazy-expiry hop
    never resolved (agent asleep at decision time, hop timed out).
    Same terminal as an `expired` resolution; own session per row."""
    from app.agent.job_status import (
        ERR_AWAITING_CONFIRMATION, STATUS_WAITING_ON_USER,
    )
    from app.db.database import async_session_maker
    from app.db.models import BuildJob
    from sqlalchemy import select

    now = now or datetime.utcnow()
    closed = 0
    async with async_session_maker() as db:
        rows = (await db.execute(
            select(BuildJob)
            .where(BuildJob.job_type == "automation_run")
            .where(BuildJob.status == STATUS_WAITING_ON_USER)
            .where(BuildJob.error_class == ERR_AWAITING_CONFIRMATION)
        )).scalars().all()
        for job in rows:
            if park_expiry_deadline(job, now) > now:
                continue
            if await resolve_parked_run(db, job, outcome="expired"):
                closed += 1
                logger.info(
                    "[automations] closed expired confirm park job=%s",
                    job.id[:8],
                )
    return closed
