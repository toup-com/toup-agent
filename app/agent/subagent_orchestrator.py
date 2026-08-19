"""Sub-agent orchestrator — Phase 4 of the sub-agent spawning arc.

The orchestrator is the bridge between the LLM-callable ``spawn``
tool (re-pointed in Phase 4) and the dispatcher + announce-back
machinery (Phase 2 + the message_writer in this commit).

What it does:

  1. Runs ``pre_spawn_checks`` from
     ``app/agent/subagent_dispatcher.py`` — kill switch, depth,
     caps, idempotency.
  2. On green light, calls ``JobRunner.create_job`` with the
     sub-agent shape (``job_type='subagent'``, ``parent_job_id``
     in ``config_json``, ``source_kind='subagent'``,
     idempotency key from
     ``subagent_dispatcher.derive_idempotency_key``).
  3. Fires the child via ``_spawn_bg(_run_child(...))``
     and returns the job_id to the parent — non-blocking.
  4. The child task acquires a ``LaneType.SUBAGENT`` slot on the
     process-wide ``LaneManager``, flips the job to ``running``,
     calls ``agent_runner.run()`` with
     ``prompt_profile=PromptProfile.SUBAGENT``,
     ``save_assistant_message=False``,
     ``disable_post_processing=True``, and the merged
     memory-write-disabled tool set.
  5. On completion: writes the child's final assistant text via
     ``write_subagent_message`` + ``broadcast_subagent_message``,
     stamps ``BuildJob.summary_message_id``, and transitions the
     job to ``completed`` (or ``failed`` / ``timeout`` /
     ``cancelled`` / ``budget_exhausted`` depending on outcome).
  6. Uses ``transition_job_status`` (Phase 2's chokepoint) so
     terminal cascades fire automatically if the child has
     descendants (no-op in v1 — depth=1 prevents grandchildren).

Threading model
---------------
The orchestrator function returns immediately after the
``asyncio.create_task`` fire. The child coroutine handles its own
session lifecycle (one ``async with async_session_maker()`` per
phase: status flip, agent_runner.run, message write, terminal
transition).

Telegram fan-out: the spawn tool handler's ``_chat_id`` (if set)
is passed in as ``telegram_chat_id``. After the in-DB write +
WS broadcast, the orchestrator best-effort sends a Telegram
message via ``broadcast_subagent_telegram``. A Telegram outage
must NOT block the DB write or the WS broadcast.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Any, Optional

from app.agent.subagent_dispatcher import (
    REJECTION_CODES,
    SUBAGENT_JOB_TYPE,
    SpawnRejection,
    cascade_cancel,
    derive_idempotency_key,
    pre_spawn_checks,
    subagent_depth_of,
    transition_job_status,
)
from app.services.background_tasks import spawn as _spawn_bg


from app.services.plain_text import strip_markdown as _strip_md

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Public API — what the spawn tool handler calls
# ──────────────────────────────────────────────────────────────────────


async def spawn_subagent(
    *,
    user_id: str,
    task: str,
    label: Optional[str],
    model: Optional[str],
    timeout_seconds: Optional[int],
    parent_job_id: Optional[str],
    channel: Optional[str],
    telegram_chat_id: Optional[int],
    agent_runner: Any,
    session_maker: Any = None,
    credit_budget: Optional[float] = None,
) -> dict[str, Any]:
    """Pre-spawn gate + create job + fire child task. Returns the
    tool-result dict the LLM sees (job_id on success,
    {error, message} on rejection).

    Returns immediately. The child runs in the spawned asyncio.Task
    and announces back via Day-as-Chat when done. The parent's tool
    result tells the LLM the job_id + ``"status": "queued"`` so the
    LLM can end its turn cleanly.

    ``agent_runner`` is injected (the same singleton wired at
    app/main.py:168) so the orchestrator can drive a child run.
    ``session_maker`` is overridable for tests; defaults to the
    real ``async_session_maker``.
    """
    from app.config import settings

    if session_maker is None:
        from app.db.database import async_session_maker as _maker
        session_maker = _maker

    # Clamp timeout to the configured ceiling.
    if timeout_seconds is None:
        timeout_seconds = settings.subagent_default_timeout_seconds
    timeout_seconds = min(
        max(1, int(timeout_seconds)),
        settings.subagent_max_timeout_seconds,
    )

    # Phase 6: spawn attempt metric — counted regardless of accept/
    # reject so rejection rate can be derived as
    # rejected/attempted at scrape time.
    from app.agent import subagent_metrics
    parent_depth_metric = 0
    if parent_job_id:
        async with session_maker() as db:
            parent_depth_metric = await subagent_depth_of(db, parent_job_id)
    subagent_metrics.record_spawn_attempted(
        channel=channel or "unknown",
        parent_depth=parent_depth_metric,
    )
    subagent_metrics.record_spawn_attempt(user_id)  # runaway-detect window

    # 1. Pre-spawn checks — kill switch, empty task, depth, caps.
    async with session_maker() as db:
        rejection: Optional[SpawnRejection] = await pre_spawn_checks(
            db,
            user_id=user_id,
            parent_job_id=parent_job_id,
            task=task,
        )
    if rejection is not None:
        subagent_metrics.record_spawn_rejected(reason=rejection.error_code)
        subagent_metrics.log_lifecycle(
            event="rejected",
            user_id=user_id,
            job_id="-",
            parent_job_id=parent_job_id,
            reason=rejection.error_code,
        )
        return rejection.to_dict()

    # 2. Create the BuildJob row via JobRunner. Idempotency: the
    # composite (source_id=parent_job_id, idempotency_key=hash)
    # UNIQUE catches a retried tool call automatically.
    idempotency_key = derive_idempotency_key(parent_job_id, task)

    # Depth at create time so it lands on the row for telemetry.
    async with session_maker() as db:
        parent_depth = (
            await subagent_depth_of(db, parent_job_id) if parent_job_id else 0
        )
    child_depth = parent_depth + 1

    config = {
        "task": task,
        "label": label,
        "model": model,
        "timeout_seconds": timeout_seconds,
        "parent_job_id": parent_job_id,
        "parent_depth": parent_depth,
        "depth": child_depth,
        "telegram_chat_id": telegram_chat_id,
        "parent_channel": channel,
        "credit_budget": credit_budget,
    }

    from app.agent.job_runner import JobRunner, TaskSpec

    spec = TaskSpec(
        user_id=user_id,
        channel="subagent",
        source_kind="subagent",
        source_id=parent_job_id,
        prompt=task,
        config_json=config,
        conversation_id=None,
    )

    job = await JobRunner(session_maker=session_maker).create_job(
        job_type=SUBAGENT_JOB_TYPE,
        spec=spec,
        title=(label or "Sub-agent task")[:200],
        prompt=task,
        idempotency_key=idempotency_key,
        status="queued",
        model=model or "",
        layer=0,
        steps_json="[]",
    )

    # Detect idempotency-conflict short-circuit. JobRunner.create_job
    # returns the existing row if (source_id, idempotency_key)
    # already exists; we tell the LLM the same job_id back so a
    # retry collapses to one logical spawn.
    was_fresh = (
        job.created_at is not None
        and (datetime.utcnow() - job.created_at).total_seconds() < 10
    )

    # Phase 9: persist credit_budget_allocated onto the row so the
    # Jobs API / Mission Control sees the budget the parent allocated.
    # Also gives the orphan sweep + a follow-up dashboard "budget vs
    # spent" rendering something to read.
    if credit_budget is not None:
        try:
            from sqlalchemy import update
            from app.db.models import BuildJob
            async with session_maker() as db:
                await db.execute(
                    update(BuildJob)
                    .where(BuildJob.id == job.id)
                    .values(credit_budget_allocated=float(credit_budget))
                    .execution_options(synchronize_session=False)
                )
                await db.commit()
        except Exception as _bud_err:
            logger.warning(
                "[subagent_orchestrator] credit_budget_allocated write "
                "skipped (non-fatal): %s", _bud_err,
            )

    if was_fresh:
        # 3. Fire the child. The task owns its own session lifecycle.
        _spawn_bg(_run_child(
            job_id=job.id,
            user_id=user_id,
            task=task,
            label=label,
            model=model,
            timeout_seconds=timeout_seconds,
            telegram_chat_id=telegram_chat_id,
            parent_job_id=parent_job_id,
            agent_runner=agent_runner,
            session_maker=session_maker,
            credit_budget=credit_budget,
            parent_channel=channel,
        ))
        subagent_metrics.log_lifecycle(
            event="accepted",
            user_id=user_id,
            job_id=job.id,
            parent_job_id=parent_job_id,
            depth=child_depth,
            extra={"label": label or "-", "timeout_s": timeout_seconds},
        )
    else:
        subagent_metrics.log_lifecycle(
            event="idempotency_collapsed",
            user_id=user_id,
            job_id=job.id,
            parent_job_id=parent_job_id,
        )

    return {
        "job_id": job.id,
        "status": job.status,
        "label": label,
        "idempotency_collapsed": not was_fresh,
    }


# ──────────────────────────────────────────────────────────────────────
# Cost computation — Phase 9 soft credit-budget enforcement
#
# Uses the existing MODEL_PRICING table from app/agent/token_tracker.py
# rather than maintaining a parallel one. When the LLM-proxy credit
# hook lands on feat/credit-billing-system this helper becomes the
# fallback (the hook writes credit_spent in real time per LLM call);
# until then it's the canonical computation at run end.
# ──────────────────────────────────────────────────────────────────────


def _compute_run_cost(
    *, model: str, tokens_in: int, tokens_out: int,
) -> float:
    """Token-based USD cost computation for a sub-agent run.

    Unknown model → 0.0 (pricing dict returns the {"input":0,
    "output":0} fallback so an unrecognised model never trips
    budget exhaustion). Operators who add a new model name should
    add its rates to MODEL_PRICING in app/agent/token_tracker.py
    so cost attribution stays accurate. The
    ``subagent_credit_multiplier`` setting scales the result so
    operators can dial sub-agent spend up/down without redeploying.
    """
    from app.agent.token_tracker import MODEL_PRICING
    from app.config import settings

    pricing = MODEL_PRICING.get(model, {"input": 0.0, "output": 0.0})
    in_cost = (tokens_in or 0) / 1_000_000.0 * pricing["input"]
    out_cost = (tokens_out or 0) / 1_000_000.0 * pricing["output"]
    base = in_cost + out_cost
    multiplier = getattr(settings, "subagent_credit_multiplier", 1.0) or 1.0
    return round(base * multiplier, 6)


# ──────────────────────────────────────────────────────────────────────
# Child execution
# ──────────────────────────────────────────────────────────────────────


async def _run_child(
    *,
    job_id: str,
    user_id: str,
    task: str,
    label: Optional[str],
    model: Optional[str],
    timeout_seconds: int,
    telegram_chat_id: Optional[int],
    parent_job_id: Optional[str],
    agent_runner: Any,
    session_maker: Any,
    credit_budget: Optional[float] = None,
    parent_channel: Optional[str] = None,
) -> None:
    """Background task: flip status to running, run the child agent
    under a SUBAGENT lane slot, announce-back the result.

    All exceptions terminate the run — they're caught at the outer
    layer and converted to ``status='failed'`` so a buggy child
    never leaves a row stuck in 'running'.
    """
    from app.agent.lanes import LaneType, get_lane_manager
    from app.agent.prompt_profile import PromptProfile

    started_at = datetime.utcnow()
    lane_mgr = get_lane_manager()
    lane_run = None
    final_text: Optional[str] = None
    outcome = "failed"
    error_message: Optional[str] = None
    tokens_prompt: Optional[int] = None
    tokens_completion: Optional[int] = None
    model_used: Optional[str] = None

    try:
        # Flip status → running and emit a job_update so Mission
        # Control / Jobs page lights up immediately.
        async with session_maker() as db:
            from app.db.models import BuildJob
            from sqlalchemy import update
            await db.execute(
                update(BuildJob)
                .where(BuildJob.id == job_id, BuildJob.status == "queued")
                .values(status="running")
                .execution_options(synchronize_session=False)
            )
            await db.commit()
        await _emit_job_update(user_id, job_id, status="running", label=label)

        # Acquire a SUBAGENT lane slot. The semaphore is process-wide
        # (lane_max_concurrent=5 default). If the lane is saturated
        # we block here — that's fine; spawn was already accepted
        # and the user already saw the queued event.
        lane_run = await lane_mgr.acquire(
            LaneType.SUBAGENT, user_id=user_id, model=model,
        )
        from app.agent import subagent_metrics
        subagent_metrics.record_lane_acquired()
        subagent_metrics.log_lifecycle(
            event="started",
            user_id=user_id,
            job_id=job_id,
            parent_job_id=parent_job_id,
        )

        # Phone surface: start the lock-screen/Dynamic-Island card. The
        # bar animates ON-DEVICE toward the job's timeout window (zero
        # update pushes needed) — the honest ceiling for a bounded job.
        await _notify_job_event(
            job_id=job_id, label=label, kind="mission_started",
            title=f"🛠 Working on: {(label or 'background task')[:150]}",
            body=(task or "")[:200],
            timer_end_ms=int((time.time() + min(timeout_seconds, 1800)) * 1000),
            dedup_suffix="started",
            urgent=(parent_channel != "autopilot"),
        )

        # Live status on the card while the child works (Claude
        # parity): tool-boundary subtitles + interpolated progress ride
        # update_only rows onto the card started above. The first
        # update swaps the blind timer bar for a discrete bar with a
        # real status line ("Searching the web…").
        from app.agent.turn_progress import TurnProgressEmitter

        _job_emitter = TurnProgressEmitter(
            mission_id=job_id,
            mission_title=(label or task or "Background task")[:60],
            base_progress=5,
            ceiling=90,
            route="mission-control",
        )

        try:
            response = await asyncio.wait_for(
                agent_runner.run(
                    user_message=f"[Sub-agent task]\n{task}",
                    user_id=user_id,
                    session_id=f"subagent:{job_id}",
                    channel="subagent",
                    telegram_chat_id=None,  # never deliver via tg from child run
                    prompt_profile=PromptProfile.SUBAGENT,
                    subagent_task_label=label or task[:80],
                    save_user_message=False,
                    save_assistant_message=False,
                    disable_post_processing=True,
                    model_override=model,
                    on_tool_start=_job_emitter.on_tool_start,
                ),
                timeout=timeout_seconds,
            )
            final_text = getattr(response, "text", None) or "(no output)"
            # AgentResponse field names are tokens_input/tokens_output/
            # model — the old input_tokens/output_tokens/model_used
            # getattrs silently returned None forever, so sub-agent token
            # accounting and the soft budget check computed from 0 tokens
            # (found by the Autopilot investigation, 2026-07-08).
            tokens_prompt = getattr(response, "tokens_input", None)
            tokens_completion = getattr(response, "tokens_output", None)
            model_used = getattr(response, "model", None)
            outcome = "success"

            # Phase 9: soft credit-budget enforcement.
            #
            # The LLM-proxy hook on feat/credit-billing-system is the
            # canonical enforcement (real-time, per-call). Until it
            # lands, we provide a soft enforcement: after the run
            # completes, compute cost from tokens + model pricing,
            # and flip outcome to 'budget_exhausted' if we overran.
            # This is a safety net that closes the spec's Phase 3
            # acceptance criterion ("a run that exceeds its
            # credit_budget terminates and reports budget_exhausted")
            # without depending on the unmerged hook.
            #
            # When the credit hook lands, BuildJob.credit_spent is
            # already being incremented in real time; this block
            # then just reads the row's value instead of computing
            # from tokens. The flip-to-budget_exhausted logic stays
            # identical.
            if credit_budget is not None and credit_budget > 0:
                computed_spend = _compute_run_cost(
                    model=model_used or model or "",
                    tokens_in=tokens_prompt or 0,
                    tokens_out=tokens_completion or 0,
                )
                if computed_spend > float(credit_budget):
                    outcome = "budget_exhausted"
                    error_message = (
                        f"Sub-agent exceeded credit budget: "
                        f"spent ${computed_spend:.4f} vs ${float(credit_budget):.4f} "
                        f"allocated."
                    )
                    logger.warning(
                        "[subagent_orchestrator] job=%s BUDGET_EXHAUSTED "
                        "spent=$%.4f budget=$%.4f model=%s tokens=%d/%d",
                        job_id, computed_spend, float(credit_budget),
                        model_used or model, tokens_prompt or 0,
                        tokens_completion or 0,
                    )

        except asyncio.TimeoutError:
            outcome = "timeout"
            error_message = f"Sub-agent task timed out after {timeout_seconds}s."
            final_text = error_message
            logger.warning(
                "[subagent_orchestrator] job=%s TIMEOUT after %ds",
                job_id, timeout_seconds,
            )

        except asyncio.CancelledError:
            outcome = "cancelled"
            error_message = "Sub-agent task cancelled."
            final_text = error_message
            logger.info("[subagent_orchestrator] job=%s CANCELLED", job_id)
            # Re-raise after cleanup so the asyncio.create_task knows
            # it was cancelled and the lane is released properly.
            raise

        except Exception as exc:
            outcome = "failed"
            error_message = f"{type(exc).__name__}: {str(exc)[:500]}"
            final_text = f"Sub-agent task failed: {error_message}"
            logger.exception(
                "[subagent_orchestrator] job=%s FAILED", job_id,
            )

    finally:
        if lane_run is not None:
            try:
                await lane_mgr.release(
                    lane_run,
                    status=outcome if outcome != "success" else "completed",
                    error=error_message,
                )
                from app.agent import subagent_metrics
                subagent_metrics.record_lane_released(outcome=outcome)
            except Exception as e:  # pragma: no cover — defensive
                logger.warning(
                    "[subagent_orchestrator] lane release failed: %s", e,
                )

        # Duration + credit-spent metrics + lifecycle log. Fire
        # regardless of lane_run state so a fail-before-acquire path
        # (e.g. lane manager exception) still gets observability.
        try:
            from app.agent import subagent_metrics
            from app.db.database import async_session_maker
            from app.db.models import BuildJob
            from sqlalchemy import select

            elapsed_ms = (datetime.utcnow() - started_at).total_seconds() * 1000.0
            subagent_metrics.record_run_duration(
                duration_ms=elapsed_ms, outcome=outcome,
            )
            # Pull credit_spent for the histogram (post-finalize
            # so the row's already been UPDATEd with the latest
            # value from the credit hook, if any).
            async with session_maker() as db:
                row = (await db.execute(
                    select(BuildJob.credit_spent).where(BuildJob.id == job_id)
                )).scalar_one_or_none()
            if row is not None:
                subagent_metrics.record_credit_spent(
                    credit=row, outcome=outcome,
                )
            subagent_metrics.log_lifecycle(
                event=outcome,
                user_id=user_id,
                job_id=job_id,
                parent_job_id=parent_job_id,
                outcome=outcome,
                duration_ms=elapsed_ms,
                credit_spent=row,
            )
        except Exception as _met_err:  # pragma: no cover — defensive
            logger.warning(
                "[subagent_orchestrator] metrics emit skipped (non-fatal): %s",
                _met_err,
            )

        # Announce + terminal transition. Wrap in its own try so
        # an announce failure doesn't lose the row in 'running'.
        try:
            await _finalize(
                job_id=job_id,
                user_id=user_id,
                final_text=final_text or "(no output)",
                outcome=outcome,
                error_message=error_message,
                label=label,
                parent_job_id=parent_job_id,
                telegram_chat_id=telegram_chat_id,
                tokens_prompt=tokens_prompt,
                tokens_completion=tokens_completion,
                model_used=model_used,
                started_at=started_at,
                session_maker=session_maker,
                parent_channel=parent_channel,
            )
        except Exception as fin_err:  # pragma: no cover — defensive
            logger.exception(
                "[subagent_orchestrator] _finalize failed for job=%s — "
                "row may be stuck in 'running'; orphan sweep will catch it. "
                "err=%s",
                job_id, fin_err,
            )


#: Live Activity mission-id prefix for jobs created inside a chat turn.
#: ONE card per conversation (Round 3, item 4): every job the agent
#: creates in the same chat targets ``chatjob:<session_id>`` so a second
#: job UPDATES the live card instead of push-to-starting a new one (the
#: LA lane's ``refresh_if_started`` handles the swap). Jobs with no
#: conversation (dashboard/API-created, ephemeral routine turns) keep
#: ``mission_id = job_id`` — the historic per-job card.
CHAT_JOB_MISSION_PREFIX = "chatjob:"

#: How long a finished job card shows its completed state (✓ + response
#: preview, in the Dynamic Island) before the ``event=end`` push retires
#: it to the lock screen (Round 3, item 2: "end after a brief delay").
#: The LA lane realises it as a chained queue row on the dispatcher tick,
#: so the effective delay is this + up to one tick (30s) — never a sleep
#: inside a worker holding a DB connection.
JOB_CARD_END_AFTER_S = 8


def job_mission_id(job_id: str, chat_id: Optional[str]) -> str:
    """The Live Activity mission id a job's pushes address."""
    if chat_id:
        return f"{CHAT_JOB_MISSION_PREFIX}{chat_id}"[:64]
    return job_id


async def _notify_job_event(
    *,
    job_id: str,
    label: Optional[str],
    kind: str,
    title: str,
    body: Optional[str] = None,
    progress: Optional[int] = None,
    timer_end_ms: Optional[int] = None,
    dismiss_after_s: Optional[int] = None,
    priority: str = "default",
    dedup_suffix: str,
    urgent: bool = True,
    # Round 3 (2026-08-18): deep link + card content. All optional; all
    # flat scalars (POST /api/agent/notify rejects nested values). See
    # docs/design/round3-notifications.md for the wire contract.
    chat_id: Optional[str] = None,
    message_id: Optional[str] = None,
    day_chat_id: Optional[str] = None,
    job_type: Optional[str] = None,
    step_name: Optional[str] = None,
    steps_done: Optional[int] = None,
    steps_total: Optional[int] = None,
    preview: Optional[str] = None,
    end_after_s: Optional[int] = None,
    refresh_if_started: bool = False,
    mission_id: Optional[str] = None,
    route: Optional[str] = None,
) -> None:
    """Phone-surface lifecycle event for a spawned job: durable outbox →
    platform → APNs Live Activity (lock screen card + Dynamic Island),
    which keeps working when the app is force-quit. ``urgent`` defaults
    True — spawns from an interactive chat turn mean the user is awake
    and quiet hours must not defer the card. Callers pass urgent=False
    for spawns whose parent is a background surface (autopilot ticks):
    a 3am mission-tick spawn must NOT bypass quiet hours (2026-07-16).

    Round 3 fields:
      * ``chat_id`` / ``message_id`` — the deep-link payload every push
        and every Live Activity update carries (item 3). ``chat_id`` is
        the conversation (session) id; ``message_id`` the assistant
        message the answer lands in (pre-minted by the runner, so it is
        known mid-turn). When ``chat_id`` is set the card is keyed
        ``chatjob:<chat_id>`` and routed to the chat (item 4).
      * ``job_type`` / ``step_name`` / ``steps_done`` / ``steps_total`` /
        ``preview`` — card content-state (item 2). ``progress`` stays the
        0-100 percent the LA lane already renders.
      * ``end_after_s`` — on a terminal kind, ask the LA lane to show the
        completed state for this long before the ``event=end`` push (a
        chained queue row; never an in-process sleep).
      * ``refresh_if_started`` — on ``mission_started``, refresh a card
        that is already up for this mission instead of skipping it as an
        at-least-once duplicate (a NEW job on the conversation's card).
      * ``mission_id`` / ``route`` — explicit overrides; derived from
        ``chat_id`` when omitted.

    Best-effort by contract: a job must never fail on notification
    plumbing."""
    try:
        from app.services.agent_notify_client import notify

        _mission = mission_id or job_mission_id(job_id, chat_id)
        _route = route or ("chat" if chat_id else "mission-control")
        data: dict[str, Any] = {
            "route": _route,
            "mission_id": _mission,
            "mission_title": (label or "Background task")[:80],
            "kind": "job",
            "job_id": job_id,
            "urgent": urgent,
        }
        if progress is not None:
            data["progress"] = progress
        if timer_end_ms:
            data["timer_end_ms"] = timer_end_ms
        if dismiss_after_s is not None:
            data["dismiss_after_s"] = dismiss_after_s
        if chat_id:
            data["chat_id"] = str(chat_id)[:64]
        if message_id:
            data["message_id"] = str(message_id)[:64]
        if day_chat_id:
            data["day_chat_id"] = str(day_chat_id)[:64]
        if job_type:
            data["job_type"] = str(job_type)[:16]
        if step_name:
            data["step_name"] = " ".join(str(step_name).split())[:80]
        if steps_done is not None:
            data["steps_done"] = int(steps_done)
        if steps_total is not None:
            data["steps_total"] = int(steps_total)
        if preview:
            # Round 4 (item 4): plain text on the card, stripped before cut.
            from app.services.plain_text import plain_preview as _plain_preview
            data["preview"] = _plain_preview(str(preview), 120)
        if end_after_s is not None:
            data["end_after_s"] = int(end_after_s)
        if refresh_if_started:
            data["refresh_if_started"] = True
        await notify(
            event_kind=kind,
            title=title[:200],
            body=(body or "")[:300] or None,
            data=data,
            priority=priority,
            dedup_key=f"{job_id}:{dedup_suffix}",
        )
    except Exception as e:  # noqa: BLE001
        logger.debug(
            "[subagent_orchestrator] notify %s failed job=%s: %s",
            kind, job_id, e,
        )


#: `required_action.type` values that are a yes/no gate rather than a
#: request for information. Drives which notify kind the card gets.
_APPROVAL_ACTIONS = frozenset({"purchase_approval", "permission"})


async def notify_job_needs_user(
    *,
    job_id: str,
    label: Optional[str],
    summary: str,
    action_type: str,
    cta_label: Optional[str] = None,
    **card: Any,
) -> None:
    """Announce that a job has parked on the user (``waiting_on_user``).

    Deliberately NOT ``mission_failed``: that kind sends an alerting
    update followed by ``event=end``, which would tear down the Live
    Activity. ``needs_input`` / ``needs_approval`` update the card's
    content-state and alert at priority 10 while **keeping the activity
    alive** — which is the honest presentation, because the job resumes
    the moment the user acts (see live_activity_service.py's kind table).

    Best-effort by contract: a job must never fail on notification
    plumbing.
    """
    kind = "needs_approval" if action_type in _APPROVAL_ACTIONS else "needs_input"
    await _notify_job_event(
        job_id=job_id,
        label=label,
        kind=kind,
        title=(label or "Background task")[:150],
        body=(f"{summary} {cta_label}." if cta_label else summary)[:300],
        # No dismiss_after_s — a waiting card must persist until resolved.
        priority="high",
        dedup_suffix=f"waiting:{action_type}",
        # The user is being asked for something; this is the one class of
        # background event that legitimately interrupts.
        urgent=True,
        **card,
    )


async def _finalize(
    *,
    job_id: str,
    user_id: str,
    final_text: str,
    outcome: str,
    error_message: Optional[str],
    label: Optional[str],
    parent_job_id: Optional[str],
    telegram_chat_id: Optional[int],
    tokens_prompt: Optional[int],
    tokens_completion: Optional[int],
    model_used: Optional[str],
    started_at: datetime,
    session_maker: Any,
    parent_channel: Optional[str] = None,
) -> None:
    """Announce-back + terminal status transition."""
    from app.agent.subagent_message_writer import (
        broadcast_subagent_message,
        write_subagent_message,
    )

    # Map outcome → BuildJob.status terminal value.
    status_map = {
        "success": "completed",
        "timeout": "timeout",
        "cancelled": "cancelled",
        "failed": "failed",
        "budget_exhausted": "budget_exhausted",
    }
    terminal_status = status_map.get(outcome, "failed")

    # 1. Write the Day-as-Chat message. Always announce — even on
    # failure/timeout the user wants to know "the sub-agent I asked
    # for didn't complete" rather than silent loss.
    # Skip the announce on cancelled — the user (or supervisor)
    # explicitly cancelled, no need to surface noise.
    msg_id: Optional[str] = None
    day_chat_id: Optional[str] = None
    if outcome != "cancelled":
        async with session_maker() as db:
            msg_id, day_chat_id = await write_subagent_message(
                db,
                user_id=user_id,
                content=final_text,
                job_id=job_id,
                parent_job_id=parent_job_id,
                label=label,
                model_used=model_used,
                tokens_prompt=tokens_prompt,
                tokens_completion=tokens_completion,
                outcome=outcome,
            )

        # 2. Broadcast over WS (web / RN / extension).
        try:
            await broadcast_subagent_message(
                user_id,
                message_id=msg_id,
                day_chat_id=day_chat_id,
                job_id=job_id,
                parent_job_id=parent_job_id,
                label=label,
                content=final_text,
                outcome=outcome,
                model_used=model_used,
            )
        except Exception as e:
            logger.warning(
                "[subagent_orchestrator] WS broadcast failed (non-fatal) "
                "job=%s err=%s",
                job_id, e,
            )

        # 3. Telegram fan-out (best-effort) when parent was on
        # Telegram. Failure must not block the in-DB record.
        if telegram_chat_id:
            try:
                await _broadcast_subagent_telegram(
                    telegram_chat_id=telegram_chat_id,
                    label=label,
                    outcome=outcome,
                    final_text=final_text,
                )
            except Exception as e:
                logger.warning(
                    "[subagent_orchestrator] Telegram fan-out failed "
                    "(non-fatal) job=%s err=%s",
                    job_id, e,
                )

    # 4. Terminal transition + cascade (Phase 2 chokepoint). Stamps
    # summary_message_id and completed_at; cascade-cancels any
    # descendants if the outcome is failure-shape.
    async with session_maker() as db:
        from app.db.models import BuildJob
        from sqlalchemy import update

        # Stamp summary_message_id + tokens on the row alongside
        # the status flip. transition_job_status handles status +
        # error_message + completed_at; the additional columns we
        # set ourselves in the same UPDATE batch.
        if msg_id:
            await db.execute(
                update(BuildJob)
                .where(BuildJob.id == job_id)
                .values(summary_message_id=msg_id, outcome=outcome)
                .execution_options(synchronize_session=False)
            )
        if tokens_prompt or tokens_completion:
            total = (tokens_prompt or 0) + (tokens_completion or 0)
            await db.execute(
                update(BuildJob)
                .where(BuildJob.id == job_id)
                .values(total_tokens=total)
                .execution_options(synchronize_session=False)
            )

            # Phase 9: stamp credit_spent from the token-based
            # computation. When the LLM-proxy hook lands and writes
            # credit_spent in real time per LLM call, this becomes
            # a redundant final-stamp; the hook's value will overwrite
            # ours with the more accurate per-call sum. Until then,
            # this is the canonical credit_spent for sub-agent rows.
            try:
                computed = _compute_run_cost(
                    model=model_used or "",
                    tokens_in=tokens_prompt or 0,
                    tokens_out=tokens_completion or 0,
                )
                if computed > 0:
                    await db.execute(
                        update(BuildJob)
                        .where(BuildJob.id == job_id)
                        .values(credit_spent=computed)
                        .execution_options(synchronize_session=False)
                    )
            except Exception as _cost_err:  # pragma: no cover — defensive
                logger.warning(
                    "[subagent_orchestrator] credit_spent stamp skipped: %s",
                    _cost_err,
                )
        await transition_job_status(
            db,
            job_id,
            new_status=terminal_status,
            error_message=error_message,
            propagate_cancel=(outcome != "success"),
        )
        await db.commit()

    # 5. Final job_update WS event so Mission Control transitions.
    await _emit_job_update(
        user_id, job_id,
        status=terminal_status,
        label=label,
        outcome=outcome,
        summary_message_id=msg_id,
    )

    # 6. Phone surface: end the Live Activity card. Success cards
    # dismiss after 15 min (quick jobs shouldn't clutter the lock
    # screen for hours); failures linger the default 4h so the user
    # can't miss them; cancels vanish fast and quietly.
    _urgent = parent_channel != "autopilot"
    if outcome == "success":
        await _notify_job_event(
            job_id=job_id, label=label, kind="mission_completed",
            title=f"✅ Done: {(label or 'background task')[:150]}",
            body=_strip_md(final_text or "")[:300],
            progress=100,
            dismiss_after_s=900, dedup_suffix="completed",
            urgent=_urgent,
            # Round 3: the summary message the child posted into the
            # day chat is the deep-link target; a preview of it rides
            # the card's final state.
            message_id=msg_id,
            preview=(final_text or "")[:120] or None,
            step_name="Done",
        )
    elif outcome == "cancelled":
        await _notify_job_event(
            job_id=job_id, label=label, kind="mission_failed",
            title=f"⏹ Stopped: {(label or 'background task')[:150]}",
            body="Cancelled.", priority="low",
            dismiss_after_s=60, dedup_suffix="cancelled",
            urgent=_urgent,
        )
    else:  # failed | timeout | budget_exhausted
        await _notify_job_event(
            job_id=job_id, label=label, kind="mission_failed",
            title=f"⚠️ Didn't finish: {(label or 'background task')[:150]}",
            body=(error_message or outcome)[:300],
            dedup_suffix="failed",
            urgent=_urgent,
        )

    logger.info(
        "[subagent_orchestrator] FINALIZED job=%s outcome=%s status=%s "
        "elapsed=%.1fs tokens=%d msg=%s",
        job_id, outcome, terminal_status,
        (datetime.utcnow() - started_at).total_seconds(),
        (tokens_prompt or 0) + (tokens_completion or 0),
        msg_id,
    )


# ──────────────────────────────────────────────────────────────────────
# Emit a job_update WS event (dashboard + Jobs page)
# ──────────────────────────────────────────────────────────────────────


async def _emit_job_update(
    user_id: str,
    job_id: str,
    *,
    status: str,
    label: Optional[str] = None,
    outcome: Optional[str] = None,
    summary_message_id: Optional[str] = None,
) -> None:
    try:
        from app.api.ws_chat import broadcast_to_user

        event: dict[str, Any] = {
            "type": "job_update",
            "job_id": job_id,
            "job_type": SUBAGENT_JOB_TYPE,
            "status": status,
            "name": label or "Sub-agent task",
        }
        if outcome is not None:
            event["outcome"] = outcome
        if summary_message_id is not None:
            event["summary_message_id"] = summary_message_id
        await broadcast_to_user(user_id, event)
    except Exception as e:
        logger.debug(
            "[subagent_orchestrator] job_update broadcast skipped: %s", e,
        )


# ──────────────────────────────────────────────────────────────────────
# Telegram fan-out
# ──────────────────────────────────────────────────────────────────────


async def _broadcast_subagent_telegram(
    *,
    telegram_chat_id: int,
    label: Optional[str],
    outcome: str,
    final_text: str,
) -> None:
    """Best-effort Telegram message. A Telegram outage must not block
    the in-DB Day-as-Chat record — caller's try/except handles that."""
    from app.agent.subagent import SubAgentManager  # singleton holding the bot ref

    # We don't have direct access to the bot here without the
    # singleton — but SubAgentManager exposes its set_bot() target.
    # Reuse the same instance the legacy spawn handler used so we
    # don't double-wire.
    # Defensive: SubAgentManager may not be instantiated in some
    # test setups; skip silently if not.
    mgr = getattr(_broadcast_subagent_telegram, "_singleton", None)
    if mgr is None:
        return
    bot_holder = getattr(mgr, "_telegram_bot", None)
    if bot_holder is None or not getattr(bot_holder, "app", None):
        return

    status_emoji = {
        "success": "✅",
        "failed": "❌",
        "timeout": "⏰",
        "budget_exhausted": "💸",
    }.get(outcome, "📋")

    from app.agent.streaming import postprocess_for_telegram, split_message

    header = (
        f"{status_emoji} <b>Sub-agent task'{label or ''}'</b> — {outcome}\n\n"
    )
    body = postprocess_for_telegram(final_text)
    full = header + body
    for chunk in split_message(full, 4096):
        try:
            await bot_holder.app.bot.send_message(
                chat_id=telegram_chat_id,
                text=chunk,
                parse_mode="HTML",
            )
        except Exception:
            # Plain-text fallback
            await bot_holder.app.bot.send_message(
                chat_id=telegram_chat_id,
                text=chunk,
            )


def set_telegram_bot_holder(holder: Any) -> None:
    """Wire the Telegram singleton at app startup so the orchestrator
    can fan out without hard-importing the bot module. Called by
    ``app/main.py`` after the bot is constructed."""
    _broadcast_subagent_telegram._singleton = holder  # type: ignore[attr-defined]


# ──────────────────────────────────────────────────────────────────────
# Orphan sweep — crash recovery
#
# Mirrors app/agent/triggers/runner.py:177-206 (_restart_sweep). When
# platform-api restarts mid-sub-agent-run, the in-memory asyncio task
# is killed but the BuildJob row stays in status='running'. Without
# this sweep the user sees an "in-flight" sub-agent forever and the
# per-parent / per-user concurrent caps count it against the live
# budget.
#
# Called once at platform-api boot. NOT a polling loop — sub-agent
# runs are inline-fired by the orchestrator (asyncio.create_task),
# not picked up by a background scheduler, so one sweep at startup
# is sufficient. A row that ages into orphan territory mid-uptime
# means the asyncio.create_task crashed silently — caught by
# _finalize's try/except "row may be stuck" log; an operational
# concern rather than a sweep concern.
# ──────────────────────────────────────────────────────────────────────


async def orphan_sweep_on_boot(session_maker: Any = None) -> int:
    """Flip ``status='running'`` sub-agent rows older than the
    configured threshold to ``status='failed'`` with a clear
    ``error_message``. Returns the count of rows swept (for the
    structured boot log).

    Idempotent — safe to call multiple times (the WHERE clause
    only matches rows that haven't been finalized).

    The sweep does NOT cascade-cancel children because v1's depth=1
    cap means a sub-agent can't have sub-agent children. If
    max_depth bumps to 2+, this function should additionally call
    ``cascade_cancel`` for each swept job_id.
    """
    from datetime import timedelta
    from sqlalchemy import select, update

    from app.config import settings
    from app.db.models import BuildJob

    if session_maker is None:
        from app.db.database import async_session_maker as _maker
        session_maker = _maker

    threshold = timedelta(minutes=settings.subagent_orphan_sweep_threshold_minutes)
    cutoff = datetime.utcnow() - threshold
    now = datetime.utcnow()

    async with session_maker() as db:
        result = await db.execute(
            update(BuildJob)
            .where(
                BuildJob.job_type == "subagent",
                BuildJob.status == "running",
                BuildJob.created_at < cutoff,
            )
            .values(
                status="failed",
                outcome="failed",
                error_message="agent_restarted: orphaned by platform restart",
                completed_at=now,
            )
            .execution_options(synchronize_session=False)
        )
        await db.commit()
        swept = getattr(result, "rowcount", 0) or 0

        # WS notification for any rows we just flipped. Best-effort —
        # a frontend that isn't connected just renders the fresh
        # state on next /jobs fetch.
        if swept:
            rows = (await db.execute(
                select(
                    BuildJob.id, BuildJob.user_id, BuildJob.title,
                    BuildJob.config_json,
                ).where(
                    BuildJob.job_type == "subagent",
                    BuildJob.status == "failed",
                    BuildJob.completed_at == now,
                )
            )).all()
            for row in rows:
                label = None
                try:
                    if isinstance(row.config_json, dict):
                        label = row.config_json.get("label")
                except Exception:
                    pass
                try:
                    await _emit_job_update(
                        row.user_id, row.id,
                        status="failed",
                        label=label or row.title,
                        outcome="failed",
                    )
                except Exception:
                    pass  # already best-effort inside
                # End the orphan's Live Activity card too — otherwise a
                # crash strands a "Working on…" card until the 8h cap.
                await _notify_job_event(
                    job_id=row.id, label=label or row.title,
                    kind="mission_failed",
                    title=f"⚠️ Didn't finish: {(label or row.title or 'background task')[:150]}",
                    body="The agent restarted while this task was running.",
                    dedup_suffix="failed",
                )

    if swept:
        logger.warning(
            "[subagent_orchestrator] orphan_sweep_on_boot flipped %d row(s) "
            "to status='failed' (threshold=%dm, cutoff=%s)",
            swept,
            settings.subagent_orphan_sweep_threshold_minutes,
            cutoff.isoformat(),
        )
    else:
        logger.info(
            "[subagent_orchestrator] orphan_sweep_on_boot — no stuck rows "
            "(threshold=%dm)",
            settings.subagent_orphan_sweep_threshold_minutes,
        )

    return swept
