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

    # ── An app build is never completed by a watchdog (round 23) ─────
    # `finish_all_steps` marks every remaining step done — the right shape
    # for a create_job task whose answer WAS the work. For an html-artifact
    # build still `running` at turn end, the work is `present_app`, and it
    # demonstrably never ran: force-greening its rows would paint
    # "Published the app ✓" on an app that was never published — the exact
    # overclaim the pipeline's own gates exist to end. Settle it honestly
    # instead: unreported phases become `skipped`, and the job closes
    # `cancelled` (the model moved on) or `failed` (a phase had failed).
    # `finish_job` remains the ONE way a build completes.
    if (getattr(job, "job_type", None) == "auto_builder"
            and getattr(job, "app_id", None)):
        # Own session, own commit: the finalizer only commits its session
        # when THIS function returned a ClosedJob, so a settle that piggy-
        # backed on the caller's transaction would silently be thrown away.
        await _settle_unpublished_build(job_id, user_id=user_id,
                                        now=now, reason=reason)
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


async def _settle_unpublished_build(
    job_id: str, *, user_id: str, now: datetime, reason: str,
) -> None:
    """Close a build job whose turn ended without a publish. Never 'completed'.

    Steps that reported a terminal state keep it; phases that never reported
    back become `skipped` (the same rule the pipeline's own `finish_job`
    applies). Status is `failed` when a phase had failed, else `cancelled` —
    "stopped before the app was published" is a stop, not a success and not
    an invented diagnosis. Opens and commits its OWN session: the finalizer
    only commits when a job was completed, and this path never returns one.
    The guarded WHERE re-checks 'running' at write time, so a publish racing
    this settle wins.
    """
    from app.db.database import async_session_maker
    from app.db.models import BuildJob, JobEvent

    try:
        from app.agent.skills.builtins.app_html.steps import phase_label
    except Exception:  # noqa: BLE001 - words are not worth a failed settle
        def phase_label(_t: str, _s: str) -> str:  # type: ignore[misc]
            return ""

    final = "cancelled"
    total = done = 0
    title = ""
    try:
        async with async_session_maker() as db:
            job = await db.get(BuildJob, job_id)
            if job is None or job.user_id != user_id or job.status != "running":
                return
            steps = parse_steps(job.steps_json)
            any_failed = False
            for s in steps:
                if not isinstance(s, dict):
                    continue
                st = s.get("status")
                if st == "failed":
                    any_failed = True
                    # Round 25, item 2. `recoverable` means "the gate found
                    # something and the build is about to fix it". This settle
                    # only runs because the build STOPPED, so nothing is going
                    # to fix it — leaving the badge on would paint a friendly
                    # in-progress row under a terminal status. Same resolution
                    # `finish_job` performs on its own close path.
                    if s.pop("recoverable", None):
                        s.pop("was_done", None)
                        s["rev"] = int(s.get("rev") or 0) + 1
                elif st == "running" and s.get("was_done"):
                    # A row mid-RETRY when the turn died was already done once
                    # (`was_done`, stamped by emit_step). Restoring it — the
                    # same rule finish_job applies — is what keeps the done
                    # count from regressing at death: the recorded 1:26 AM
                    # card fell 2/5 → 1/5 exactly here.
                    s["status"] = "done"
                    s["rev"] = int(s.get("rev") or 0) + 1
                elif st in ("pending", "running"):
                    s["status"] = "skipped"
                    label = phase_label(str(s.get("type") or ""), "skipped")
                    if label:
                        s["label"] = label
                    s["rev"] = int(s.get("rev") or 0) + 1
            final = "failed" if any_failed else "cancelled"
            new_msg = getattr(job, "user_message", None) or (
                "The app isn't ready to open yet — it needs a fix first."
                if any_failed else
                "The build stopped before the app was published. Ask me to "
                "finish it."
            )
            cfg = dict(job.config_json or {})
            cfg["settled_reason"] = f"unpublished_build:{reason}"
            res = await db.execute(
                update(BuildJob)
                .where(BuildJob.id == job_id, BuildJob.user_id == user_id,
                       BuildJob.status == "running")
                .values(status=final, completed_at=now,
                        steps_json=dump_steps(steps), config_json=cfg,
                        user_message=new_msg)
                .returning(BuildJob.id)
            )
            if res.first() is None:
                return
            db.add(JobEvent(
                job_id=job_id, user_id=user_id, kind="info",
                level="error" if any_failed else "info",
                status=final, ts=now,
                label="Build stopped before publish"[:200],
                metadata_json=json.dumps({"reason": reason}),
            ))
            await db.commit()
            title = (job.title or "").replace("Build: ", "")
            # One arithmetic everywhere: skipped rows are excluded from BOTH
            # numbers (steps.step_counts), or this closing frame disagrees
            # with the card that watched the build live.
            try:
                from app.agent.skills.builtins.app_html.steps import progress
                # Round 25, item 8: the same monotonic percent the live frames
                # carried, read off the same persisted high-water mark, so the
                # closing frame cannot undercut the last one the user saw.
                done, total, percent, _cfg = progress(steps, cfg, job_id=job_id)
            except Exception:  # noqa: BLE001 - arithmetic drift beats a failed settle
                total = len(steps)
                done = sum(1 for s in steps
                           if isinstance(s, dict) and s.get("status") == "done")
                percent = 0
    except Exception:  # noqa: BLE001 - a failed settle leaves the reaper's net
        logger.warning("[job_reconciler] build settle failed for %s",
                       job_id[:8], exc_info=True)
        return
    try:
        from app.api.ws_chat import broadcast_to_user
        await broadcast_to_user(user_id, {
            "type": "job_update",
            "job_id": job_id,
            "name": title,
            "status": final,
            "total_steps": total,
            "completed_steps": done,
            # Round 25: one frame shape for a build card, live or settled.
            "percent": percent,
            "steps": [s for s in steps if isinstance(s, dict)],
        })
    except Exception:  # noqa: BLE001 - the DB row is already honest
        logger.debug("[job_reconciler] build settle broadcast failed",
                     exc_info=True)
    logger.info(
        "[job_reconciler] settled unpublished build %s as %s (%s)",
        job_id[:8], final, reason,
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
