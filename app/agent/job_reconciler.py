"""Job completion — the one terminal shape, and the watchdog that enforces it.

Round 8 (2026-08-19). A ``create_job`` job is closed by the runner's turn-end
finalizer. When that finalizer did not run — the founder's job 6dfd5833 sat
"In progress · 2/3 · 67%" for hours after its answer rendered, because Round
4's follow-up lost the finalizer's id list across a Task boundary — NOTHING
else closed it honestly: the 30-minute reaper eventually called it
"stopped before it finished", and the next deploy's boot recovery called it
"interrupted". Both are lies about work the user already has.

Two things live here:

1. :func:`close_job_completed` — the ONE way an agent-authored job becomes
   ``completed``: status + ``completed_at``, every remaining step done with
   its real window (``job_steps.finish_all_steps``), the "Open in chat"
   back-link (``summary_message_id``), a ``job_events`` row, and — via
   :func:`announce_completed` — the in-app ``job_update`` frame and the phone
   card's terminal push. The turn-end finalizer and the reconciler both call
   it, so the surfaces cannot disagree about what "done" looks like.

2. :func:`reconcile_delivered_turn_jobs` — the server-side watchdog. Rule:
   **no job may remain 'running' after its turn's answer has been
   delivered.** The proof of delivery is the persisted assistant message:
   the job's own answer row (``config_json.asst_message_id``, stamped by
   create_job from the runner's pre-minted id) when the job is new enough
   to carry it, else any assistant message in the job's conversation created
   after the job. Delivered ⇒ force-complete, at the message's timestamp.

   Runs every :data:`RECONCILE_INTERVAL_S` from ``agent_main`` (before the
   stall reaper's sweep, which imports it too), and once at boot BEFORE
   ``job_recovery`` so an already-delivered job is completed rather than
   failed as an orphan of the restart. Idempotent with the finalizer: both
   go through the guarded ``status == 'running'`` UPDATE, and the phone
   push dedups on ``<job_id>:completed``.

   Left alone: jobs with no conversation (dashboard-created), jobs the turn
   handed to ``spawn`` / ``start_mission`` (``config_json.handed_off``),
   parked (``waiting_on_user``) and terminal rows, and rows younger than
   :data:`MIN_AGE` (the finalizer's own window).
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import select, update

from app.agent.job_steps import (
    counts as _step_counts,
    dump_steps,
    finish_all_steps,
    parse_steps,
)

logger = logging.getLogger(__name__)

RECONCILE_INTERVAL_S = 60
_BOOT_DELAY_S = 45
#: A row younger than this is still the finalizer's to close.
MIN_AGE = timedelta(seconds=20)
#: Without the job's own answer id we accept "any later assistant message in
#: the conversation" only for a job this old — a concurrent sibling turn's
#: answer must not close a job whose turn is still in flight.
LEGACY_PROOF_MIN_AGE = timedelta(minutes=3)
#: How far back the sweep looks. Older running rows are the reaper's.
LOOKBACK = timedelta(hours=48)


@dataclass
class ClosedJob:
    job_id: str
    title: str
    user_id: str
    chat_id: Optional[str]
    job_type: Optional[str]
    steps_total: int
    last_step: str
    completed_at: datetime


def _job_type_of(job: Any) -> Optional[str]:
    cfg = getattr(job, "config_json", None)
    if isinstance(cfg, dict) and cfg.get("job_type"):
        return str(cfg.get("job_type"))
    try:
        from app.agent.job_type import classify_job_type
        return classify_job_type(job.title, getattr(job, "prompt", None))
    except Exception:  # noqa: BLE001
        return None


async def close_job_completed(
    db,
    job_id: str,
    *,
    user_id: str,
    now: datetime,
    message_id: Optional[str] = None,
    total_tokens: Optional[int] = None,
    model: Optional[str] = None,
    reason: str = "turn_end",
) -> Optional[ClosedJob]:
    """Guarded close: only a ``running`` row is completed. Writes the row and
    a ``job_events`` heartbeat in the caller's session (caller commits).
    Returns the card facts for :func:`announce_completed`, or None when the
    row was not ours to close (already terminal, parked, missing)."""
    from app.db.models import BuildJob, JobEvent

    job = await db.get(BuildJob, job_id)
    if job is None or job.status != "running" or job.user_id != user_id:
        return None

    steps = finish_all_steps(parse_steps(job.steps_json), now,
                             fallback_start=getattr(job, "created_at", None))
    done, total = _step_counts(steps)
    cfg = dict(job.config_json or {})
    if reason != "turn_end":
        cfg["reconciled_at"] = now.isoformat()
        cfg["reconciled_reason"] = reason
    values: Dict[str, Any] = {
        "status": "completed",
        "completed_at": now,
        "steps_json": dump_steps(steps),
        "config_json": cfg,
    }
    if message_id and not job.summary_message_id:
        values["summary_message_id"] = str(message_id)[:50]
    if total_tokens is not None:
        values["total_tokens"] = int(total_tokens)
    if model:
        values["model"] = model

    res = await db.execute(
        update(BuildJob)
        .where(BuildJob.id == job_id, BuildJob.user_id == user_id,
               BuildJob.status == "running")
        .values(**values)
        .returning(BuildJob.id)
    )
    if res.first() is None:
        return None
    # The activity feed / stall reaper key on job_events; a completion that
    # leaves no row looks, in the timeline, like a job that just stopped.
    db.add(JobEvent(
        job_id=job_id, user_id=user_id, kind="info", level="info",
        status="completed", ts=now,
        label=(f"Completed: {done}/{total} steps"
               if total else "Completed")[:200],
        metadata_json=json.dumps({"reason": reason}),
    ))
    last_step = str(steps[-1].get("label") or "") if steps else ""
    return ClosedJob(
        job_id=job_id, title=job.title or "", user_id=user_id,
        chat_id=getattr(job, "conversation_id", None), job_type=_job_type_of(job),
        steps_total=total, last_step=last_step, completed_at=now,
    )


async def announce_completed(
    closed: ClosedJob,
    *,
    message_id: Optional[str],
    preview: Optional[str],
    day_chat_id: Optional[str] = None,
    chat_id_fallback: Optional[str] = None,
) -> None:
    """Tell every surface: the in-app ``job_update`` frame (the web card is
    WS-driven only, and the app refetches on it) and the phone card's
    terminal push (answer preview, n/n steps, deep link). Best-effort."""
    chat_id = closed.chat_id or chat_id_fallback
    try:
        from app.api.ws_chat import broadcast_to_user
        await broadcast_to_user(closed.user_id, {
            "type": "job_update",
            "job_id": closed.job_id,
            "job_type": closed.job_type,
            "name": closed.title,
            "status": "completed",
            "step": closed.last_step or "Done",
            "total_steps": closed.steps_total,
            "completed_steps": closed.steps_total,
            "chat_id": chat_id,
            "message_id": message_id,
        })
    except Exception:  # noqa: BLE001 — a job must never fail on plumbing
        logger.debug("[job_reconciler] job_update broadcast failed", exc_info=True)
    try:
        from app.agent.subagent_orchestrator import (
            JOB_CARD_END_AFTER_S, _notify_job_event,
        )
        from app.services.plain_text import (
            answer_preview as _ap, humanize_label as _hl, plain_preview as _plain,
        )
        _preview = _ap(preview, 100) if preview else ""
        await _notify_job_event(
            job_id=closed.job_id, label=closed.title,
            kind="mission_completed",
            title=f"✅ Done: {_hl(_plain(closed.title or '', 150))}",
            body=_preview or "Finished.", progress=100,
            dismiss_after_s=900, dedup_suffix="completed",
            chat_id=chat_id, message_id=message_id,
            day_chat_id=day_chat_id or None,
            job_type=closed.job_type, step_name="Done",
            steps_done=closed.steps_total, steps_total=closed.steps_total,
            preview=_preview or None,
            end_after_s=JOB_CARD_END_AFTER_S,
        )
    except Exception:  # noqa: BLE001
        logger.debug("[job_reconciler] terminal push failed", exc_info=True)


# ── The watchdog ─────────────────────────────────────────────────────────


async def reconcile_delivered_turn_jobs(now: Optional[datetime] = None) -> int:
    """Complete every running agent-authored job whose turn's answer has been
    delivered. Returns how many were closed."""
    from app.db.database import async_session_maker
    from app.db.models import BuildJob, Message

    now = now or datetime.utcnow()
    candidates: List[Dict[str, Any]] = []
    async with async_session_maker() as db:
        rows = (await db.execute(
            select(BuildJob).where(
                BuildJob.status == "running",
                BuildJob.job_type == "agent_task",
                BuildJob.conversation_id.isnot(None),
                BuildJob.created_at < now - MIN_AGE,
                BuildJob.created_at > now - LOOKBACK,
                BuildJob.paused_at.is_(None),
            )
        )).scalars().all()
        for job in rows:
            cfg = job.config_json if isinstance(job.config_json, dict) else {}
            if cfg.get("handed_off"):
                continue  # spawn / start_mission owns it now
            proof = None
            asst_id = cfg.get("asst_message_id")
            if asst_id:
                proof = (await db.execute(
                    select(Message.id, Message.created_at, Message.content).where(
                        Message.id == str(asst_id), Message.role == "assistant",
                    )
                )).first()
            elif job.created_at <= now - LEGACY_PROOF_MIN_AGE:
                # Rows written before create_job stamped the answer id.
                proof = (await db.execute(
                    select(Message.id, Message.created_at, Message.content).where(
                        Message.conversation_id == job.conversation_id,
                        Message.role == "assistant",
                        Message.created_at > job.created_at,
                    ).order_by(Message.created_at.asc()).limit(1)
                )).first()
            if proof is None:
                continue
            candidates.append({
                "job_id": job.id, "user_id": job.user_id,
                "message_id": proof[0],
                "delivered_at": proof[1] or now,
                "preview": proof[2] or "",
                "chat_id": job.conversation_id,
                "day_chat_id": None,
            })

    closed_n = 0
    for c in candidates:
        try:
            async with async_session_maker() as db:
                closed = await close_job_completed(
                    db, c["job_id"], user_id=c["user_id"],
                    # The honest completion instant is when the answer landed,
                    # not when the sweep noticed. Never earlier than the job's
                    # own steps could have closed, though: clamp to >= now-48h.
                    now=max(c["delivered_at"], now - LOOKBACK),
                    message_id=c["message_id"], reason="answer_delivered",
                )
                if closed is None:
                    continue
                await db.commit()
            closed_n += 1
            logger.warning(
                "[job_reconciler] completed %s (%s) — answer delivered at %s, "
                "row was still running",
                closed.job_id[:8], closed.title[:60], c["delivered_at"],
            )
            await announce_completed(
                closed, message_id=c["message_id"], preview=c["preview"],
                chat_id_fallback=c["chat_id"],
            )
        except Exception:  # noqa: BLE001 — one bad row must not stop the sweep
            logger.exception("[job_reconciler] failed on job %s", c.get("job_id"))
    return closed_n


async def reconcile_loop() -> None:
    """Forever loop for agent_main. Boot delay lets init_db / pool bind settle."""
    await asyncio.sleep(_BOOT_DELAY_S)
    while True:
        try:
            n = await reconcile_delivered_turn_jobs()
            if n:
                logger.info("[job_reconciler] completed %d delivered job(s)", n)
        except Exception as e:  # noqa: BLE001 — pre-bind lobby / DB hiccup
            logger.warning("[job_reconciler] sweep failed: %s", e)
        await asyncio.sleep(RECONCILE_INTERVAL_S)
