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
import contextvars
import json
import logging
import time
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional

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
    supervised: bool = False,
    worker_meta: Optional[dict] = None,
    caps: Optional[dict] = None,
) -> dict[str, Any]:
    """Pre-spawn gate + create job + fire child task. Returns the
    tool-result dict the LLM sees (job_id on success,
    {error, message} on rejection).

    ``supervised=True`` is a ``delegate`` worker (see
    :func:`run_supervised_plan`): ``worker_meta`` carries its card identity
    (``worker_index, plan_step_index, role, attempt, label, model``; keys
    starting with ``_`` are in-memory only), the row is keyed by
    ``attempt``/``_nonce`` so a retry is a NEW row, and a future the
    supervisor joins on is registered BEFORE the child is fired.

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
            caps=caps,
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
    if supervised:
        idempotency_key = derive_idempotency_key(
            parent_job_id, task,
            attempt=int((worker_meta or {}).get("attempt") or 1),
            nonce=(worker_meta or {}).get("_nonce"),
        )
    else:
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
    if supervised:
        config.update(_worker_public_meta(worker_meta, model))

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
        if supervised:
            _WORKER_FUTURES[job.id] = asyncio.get_running_loop().create_future()
        # 3. Fire the child. The task owns its own session lifecycle.
        _child_task = _spawn_bg(_run_child(
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
            supervised=supervised,
            worker_meta=worker_meta,
        ))
        if supervised:
            _WORKER_TASKS[job.id] = _child_task
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
    supervised: bool = False,
    worker_meta: Optional[dict] = None,
) -> None:
    """Background task: flip status to running, run the child agent
    under a SUBAGENT lane slot, announce-back the result.

    A SUPERVISED child (a ``delegate`` worker) announces nothing of its own:
    no Live Activity start/terminal (its progress rides the supervisor's
    card), no day-chat message, no Telegram. Its result and sources land in
    ``config_json``, its tool boundaries write ``job_events`` heartbeats and
    throttled worker ``job_update`` frames, and its future is resolved last.

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
    # Set only when the failure has a story the user may read (the proxy's
    # monthly model budget); `error_message` stays the raw text.
    job_user_message: Optional[str] = None
    job_error_class: Optional[str] = None
    tokens_prompt: Optional[int] = None
    tokens_completion: Optional[int] = None
    model_used: Optional[str] = None
    observer = (
        _WorkerObserver(job_id=job_id, user_id=user_id, meta=worker_meta or {},
                        model=model, session_maker=session_maker)
        if supervised else None
    )

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
        await _emit_job_update(
            user_id, job_id, status="running", label=label,
            extra=observer.frame_fields() if observer else None,
            sink=observer.sink if observer else None,
        )

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
        from app.services.plain_text import humanize_label as _hl
        # R30 (D-03): plain words, no glyph — this title is an OS banner and
        # a lock-screen card, and emoji there is the machinery's dialect.
        # A supervised worker starts NO card: every Live Activity start
        # pre-empts the cards already on the device, and its progress belongs
        # on the supervisor's card (the emitter below targets that mission).
        if not supervised:
            await _notify_job_event(
                job_id=job_id, label=label, kind="mission_started",
                title=f"Working on: {_hl(label)[:150]}",
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

        _wm = worker_meta or {}
        _job_emitter = TurnProgressEmitter(
            mission_id=(_wm.get("_mission_id") if supervised else None) or job_id,
            mission_title=(
                (_wm.get("_mission_title") if supervised else None)
                or label or task or "Background task"
            )[:60],
            base_progress=5,
            ceiling=90,
            route=("chat" if supervised and _wm.get("_chat_id") else "mission-control"),
            chat_id=_wm.get("_chat_id") if supervised else None,
            message_id=_wm.get("_message_id") if supervised else None,
        )
        _extra_run_kwargs: dict = (
            {"on_tool_event": observer.on_tool_event} if observer else {}
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
                    **_extra_run_kwargs,
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
            # `final_text` is posted into the user's day chat and the
            # failed push. For the proxy's monthly model budget the raw
            # text is an exception repr carrying cents and timestamps, so
            # both carry the dated sentence instead; the raw text stays in
            # `error_message` (operators, lane release).
            try:
                from app.agent.job_status import model_budget_verdict, user_tz_name
                from app.services.budget_refusal import is_budget_refusal
                if is_budget_refusal(exc):
                    _verdict = model_budget_verdict(exc, await user_tz_name(user_id))
                    if _verdict is not None:
                        job_user_message = _verdict.user_message
                        job_error_class = _verdict.error_class
                        final_text = _verdict.user_message
            except Exception:  # noqa: BLE001 — copy must never cost the close
                logger.debug("[subagent_orchestrator] budget verdict skipped",
                             exc_info=True)
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
                user_message=job_user_message,
                error_class=job_error_class,
                observer=observer,
            )
        except Exception as fin_err:  # pragma: no cover — defensive
            logger.exception(
                "[subagent_orchestrator] _finalize failed for job=%s — "
                "row may be stuck in 'running'; orphan sweep will catch it. "
                "err=%s",
                job_id, fin_err,
            )

        # LAST: the supervisor joins on this. Resolved on every path —
        # success, failure, timeout and cancel — or its wait would only end
        # at its own deadline.
        _resolve_worker(job_id, {
            "job_id": job_id,
            "outcome": outcome,
            "result": (final_text or "") if outcome == "success" else "",
            "error": error_message,
            "sources": observer.sources if observer else [],
            "model": model_used or model,
        })


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

#: Round 8: the widget's phase vocabulary (LiveActivityWidget.swift
#: ``JobFace.Phase``) by notify kind. Stamped on every job push as
#: ``data.phase``; the platform's LA lane mirrors it into the content
#: state (``phase``) and derives the same value for older agents' rows.
JOB_PHASE_BY_KIND = {
    "mission_started": "starting",
    "progress": "running",
    "mission_completed": "completed",
    "mission_failed": "failed",
    "needs_input": "needs_you",
    "needs_approval": "needs_you",
}


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
    phase: Optional[str] = None,
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
      * ``phase`` (Round 8) — the card's phase in the widget's own
        vocabulary (starting · running · completed · failed · needs_you);
        derived from ``kind`` when omitted, so every job push carries the
        state the phone should draw instead of leaving the widget to infer
        it from the bar and the subtitle.

    Best-effort by contract: a job must never fail on notification
    plumbing."""
    try:
        from app.services.agent_notify_client import notify

        from app.services.plain_text import humanize_label

        _mission = mission_id or job_mission_id(job_id, chat_id)
        _route = route or ("chat" if chat_id else "mission-control")
        data: dict[str, Any] = {
            "route": _route,
            "mission_id": _mission,
            # The card's HEADLINE. Model-authored labels leak the model's own
            # vocabulary ("Dispatch background conversation" reached a lock
            # screen, 2026-08-19) — gate every one.
            "mission_title": humanize_label(label)[:80],
            "kind": "job",
            "job_id": job_id,
            "urgent": urgent,
        }
        _phase = phase or JOB_PHASE_BY_KIND.get(kind)
        if _phase:
            data["phase"] = _phase
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
    # A fixed HUMAN title — this was the one job lane whose OS banner title
    # was the raw model-authored label with no prefix at all (the
    # "Dispatch background conversation" banner, 2026-08-19). The label
    # (gated) moves into the body where it is context, not a headline.
    from app.services.plain_text import humanize_label as _hl
    _label_h = _hl(label, fallback="")
    _body = (f"{summary} {cta_label}." if cta_label else summary)
    if _label_h:
        _body = f"{_label_h} — {_body}"
    await _notify_job_event(
        job_id=job_id,
        label=label,
        kind=kind,
        title="Needs your OK" if kind == "needs_approval" else "Needs your input",
        body=_body[:300],
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
    user_message: Optional[str] = None,
    error_class: Optional[str] = None,
    observer: Optional["_WorkerObserver"] = None,
) -> None:
    """Announce-back + terminal status transition.

    With an ``observer`` (a supervised ``delegate`` worker) the announce
    is replaced by the result landing in ``config_json`` — the supervisor
    merges it into ITS answer, so a second "Sub-agent: …" message beside
    that answer would be the same content twice — and no Live Activity
    terminal is sent (the worker never started a card).

    ``user_message`` / ``error_class`` are written onto the row, ride the
    terminal ``job_update`` frame, and replace the raw ``error_message`` as
    the failed push body when set."""
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
    if observer is not None:
        await observer.store_result(
            final_text if outcome == "success" else None, outcome,
        )
    elif outcome != "cancelled":
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
        if user_message or error_class:
            # `transition_job_status` writes only status + error_message;
            # the two fields a client may render are stamped here.
            await db.execute(
                update(BuildJob)
                .where(BuildJob.id == job_id)
                .values(user_message=user_message, error_class=error_class)
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
        user_message=user_message,
        error_class=error_class,
        extra=observer.frame_fields(terminal=True) if observer else None,
        sink=observer.sink if observer else None,
    )
    if observer is not None:
        logger.info(
            "[subagent_orchestrator] FINALIZED supervised job=%s outcome=%s "
            "status=%s parent=%s", job_id, outcome, terminal_status,
            parent_job_id,
        )
        return

    # 6. Phone surface: end the Live Activity card. Success cards
    # dismiss after 15 min (quick jobs shouldn't clutter the lock
    # screen for hours); failures linger the default 4h so the user
    # can't miss them; cancels vanish fast and quietly.
    _urgent = parent_channel != "autopilot"
    if outcome == "success":
        await _notify_job_event(
            job_id=job_id, label=label, kind="mission_completed",
            title=f"Done: {(label or 'background task')[:150]}",
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
            title=f"Stopped: {(label or 'background task')[:150]}",
            body="Cancelled.", priority="low",
            dismiss_after_s=60, dedup_suffix="cancelled",
            urgent=_urgent,
        )
    else:  # failed | timeout | budget_exhausted
        await _notify_job_event(
            job_id=job_id, label=label, kind="mission_failed",
            title=f"Didn't finish: {(label or 'background task')[:150]}",
            body=(user_message or error_message or outcome)[:300],
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
    user_message: Optional[str] = None,
    error_class: Optional[str] = None,
    extra: Optional[dict] = None,
    sink: Optional[Callable[[dict], Awaitable[None]]] = None,
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
        if user_message is not None:
            event["user_message"] = user_message
        if error_class is not None:
            event["error_class"] = error_class
        if extra:
            event.update(extra)
        await broadcast_to_user(user_id, event)
        if sink is not None:
            await sink({**event, "type": "job.update"})
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
                    title=f"Didn't finish: {(label or row.title or 'background task')[:150]}",
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


# ──────────────────────────────────────────────────────────────────────
# Supervised mode — the `delegate` tool's executors (voice round 2026-10-05)
#
# The user's own agent is the supervisor: ONE turn plans 2–6 steps, runs
# them on supervised children (this module), JOINS on them inside the turn
# and writes the merged answer itself. No migration: the plan is the
# supervisor job's `steps_json`, every worker is a `build_jobs` row with
# `parent_job_id` and its identity/result in `config_json`, and the join is a
# per-process future (children run in this process — `_spawn_bg`).
# ──────────────────────────────────────────────────────────────────────

#: The voice SSE route binds this for its run (``api_v1``); the supervisor and
#: its workers emit ``supervisor.started``, ``job.update`` and worker-keyed
#: ``tool.start``/``tool.end`` events through it. Unset (None) on chat, where
#: the same frames ride the user's chat socket instead.
SUPERVISOR_EVENT_SINK: "contextvars.ContextVar[Optional[Callable[[dict], Awaitable[None]]]]" = (
    contextvars.ContextVar("supervisor_event_sink", default=None)
)

#: Worker orb colours, picked by ``worker_index``. A client that does not
#: read ``color`` falls back to the index on its own palette.
WORKER_COLORS = ("#6366F1", "#10B981", "#F59E0B", "#EC4899", "#0EA5E9", "#8B5CF6")

#: A worker's result as stored on its row (config_json.result).
WORKER_RESULT_MAX = 16_000
#: Each step's result as handed back to the supervisor model.
SUPERVISOR_RESULT_MAX = 4_000
WORKER_SOURCES_MAX = 6
_WORKER_FRAME_MIN_INTERVAL_S = 1.5
_HEARTBEAT_MIN_INTERVAL_S = 10.0

_WORKER_FUTURES: Dict[str, "asyncio.Future"] = {}
_WORKER_TASKS: Dict[str, "asyncio.Task"] = {}


def worker_color(worker_index: Optional[int]) -> str:
    return WORKER_COLORS[int(worker_index or 0) % len(WORKER_COLORS)]


def _display_model(model: Optional[str]) -> Optional[str]:
    if not model:
        return None
    from app.config import settings
    if getattr(settings, "security_leak_filter", False):
        from app.services.model_alias import public_model_label
        return public_model_label(model)
    return str(model)[:64]


def _worker_public_meta(meta: Optional[dict], model: Optional[str]) -> dict:
    """The persisted half of ``worker_meta`` (config_json keys)."""
    meta = meta or {}
    # `role` is the card's vocabulary ('executor'); the plan's own name for
    # the helper ("Researcher") is kept beside it, never in its place.
    out = {"supervised": True, "role": "executor"}
    if meta.get("role") and meta.get("role") != "executor":
        out["worker_role"] = str(meta["role"])[:40]
    for k in ("worker_index", "plan_step_index", "attempt"):
        if isinstance(meta.get(k), int):
            out[k] = meta[k]
    if meta.get("label"):
        out["worker_label"] = str(meta["label"])[:120]
    if model:
        out["worker_model"] = str(model)[:64]
    return out


def _resolve_worker(job_id: str, value: dict) -> None:
    fut = _WORKER_FUTURES.pop(job_id, None)
    _WORKER_TASKS.pop(job_id, None)
    if fut is not None and not fut.done():
        fut.set_result(value)


def _sources_from_event(ev: dict) -> List[dict]:
    """Sources from the runner's own ``domains``/``urls`` on a tool end."""
    doms = ev.get("domains")
    if not isinstance(doms, list):
        return []
    urls = ev.get("urls") if isinstance(ev.get("urls"), list) else []
    from urllib.parse import urlparse
    by_host: Dict[str, str] = {}
    for u in urls:
        try:
            h = (urlparse(str(u)).hostname or "").lower()
        except Exception:  # noqa: BLE001
            continue
        by_host.setdefault(h[4:] if h.startswith("www.") else h, str(u))
    return [
        {"title": "", "url": by_host.get(str(d), "")[:500], "domain": str(d)[:120]}
        for d in doms if isinstance(d, str) and d
    ]


def _merge_sources(into: List[dict], more: List[dict]) -> List[dict]:
    seen = {s.get("domain") for s in into}
    for s in more:
        if s.get("domain") and s["domain"] not in seen and len(into) < WORKER_SOURCES_MAX:
            into.append(s)
            seen.add(s["domain"])
    return into


class _WorkerObserver:
    """One supervised child's live surface: heartbeats, throttled worker
    ``job_update`` frames, sources, and its tool events forwarded to the
    supervisor's SSE with the worker keys."""

    def __init__(self, *, job_id: str, user_id: str, meta: dict,
                 model: Optional[str], session_maker: Any) -> None:
        self.job_id = job_id
        self.user_id = user_id
        self.meta = meta
        self.model = model
        self.session_maker = session_maker
        self.sink = meta.get("_sink")
        self.sources: List[dict] = []
        self.step: Optional[str] = None
        self.result_preview: Optional[str] = None
        self._last_frame = 0.0
        self._last_beat = 0.0

    def frame_fields(self, *, terminal: bool = False) -> dict:
        m = self.meta
        out: dict = {
            "parent_job_id": m.get("parent_job_id"),
            "role": "executor",
            "worker_index": m.get("worker_index"),
            "label": m.get("label"),
            "color": worker_color(m.get("worker_index")),
            "plan_step_index": m.get("plan_step_index"),
            "attempt": m.get("attempt") or 1,
        }
        dm = _display_model(self.model)
        if dm:
            out["model"] = dm
        if self.step:
            out["step"] = self.step
        if self.sources:
            out["sources"] = list(self.sources)
        if terminal and self.result_preview:
            out["result_preview"] = self.result_preview
        return {k: v for k, v in out.items() if v is not None}

    async def on_tool_event(self, ev: dict) -> None:
        try:
            name = str(ev.get("name") or "")
            if ev.get("phase") == "start":
                from app.agent.tool_display import public_step_label
                self.step = public_step_label(name)
                await self._heartbeat(self.step)
                now = time.monotonic()
                if now - self._last_frame >= _WORKER_FRAME_MIN_INTERVAL_S:
                    self._last_frame = now
                    await _emit_job_update(
                        self.user_id, self.job_id, status="running",
                        label=self.meta.get("label"),
                        extra=self.frame_fields(), sink=self.sink,
                    )
            else:
                _merge_sources(self.sources, _sources_from_event(ev))
            if self.sink is not None:
                m = self.meta
                fwd = {
                    **ev,
                    # The worker's action belongs on the SUPERVISOR's card,
                    # under the plan step it serves.
                    "job_id": m.get("parent_job_id"),
                    "step_index": m.get("_step_index", m.get("plan_step_index")),
                    "step_name": m.get("label"),
                    "steps_total": m.get("_steps_total"),
                    "worker_job_id": self.job_id,
                    "parent_job_id": m.get("parent_job_id"),
                    "worker_index": m.get("worker_index"),
                    "label": m.get("label"),
                    "model": _display_model(self.model),
                }
                await self.sink({k: v for k, v in fwd.items() if v is not None})
        except Exception:  # noqa: BLE001 — a frame never fails a worker
            logger.debug("[supervisor] worker event skipped", exc_info=True)

    async def _heartbeat(self, label: str) -> None:
        """`job_events` row at a tool boundary — the reaper's liveness oracle
        (job_reaper reads max(job_events.ts)); throttled."""
        now = time.monotonic()
        if now - self._last_beat < _HEARTBEAT_MIN_INTERVAL_S:
            return
        self._last_beat = now
        try:
            from app.db.models import JobEvent
            async with self.session_maker() as db:
                db.add(JobEvent(job_id=self.job_id, user_id=self.user_id,
                                kind="tool_call", level="info",
                                label=(label or "Working")[:200]))
                await db.commit()
        except Exception:  # noqa: BLE001
            logger.debug("[supervisor] heartbeat skipped", exc_info=True)

    async def store_result(self, final_text: Optional[str], outcome: str) -> None:
        from app.services.plain_text import plain_preview
        text = (final_text or "")[:WORKER_RESULT_MAX]
        if text:
            self.result_preview = plain_preview(text, 160)
        try:
            from app.db.models import BuildJob
            async with self.session_maker() as db:
                row = await db.get(BuildJob, self.job_id)
                if row is None:
                    return
                cfg = dict(row.config_json or {}) if isinstance(row.config_json, dict) else {}
                cfg["result"] = text
                cfg["sources"] = list(self.sources)
                if self.result_preview:
                    cfg["result_preview"] = self.result_preview
                cfg["worker_outcome"] = outcome
                row.config_json = cfg
                await db.commit()
        except Exception:  # noqa: BLE001
            logger.warning("[supervisor] result store failed job=%s", self.job_id,
                           exc_info=True)


def _generic_job_type() -> str:
    from app.agent.job_type import JOB_TYPE_GENERIC
    return JOB_TYPE_GENERIC


def _rtl(text: str) -> bool:
    return any("֐" <= ch <= "ࣿ" for ch in (text or ""))


def supervisor_plan_steps(plan: List[dict], title: str) -> List[dict]:
    """The supervisor job's ``steps_json``: one row per plan step, then the
    supervisor's own two (it checks/merges and writes the answer itself)."""
    import uuid as _uuid
    fa = _rtl(title) or any(_rtl(p.get("label", "")) for p in plan)
    tail = (["بررسی و ادغام نتیجه‌ها", "نوشتن پاسخ"] if fa
            else ["Check and merge", "Write the answer"])
    labels = [p["label"] for p in plan] + tail
    return [
        {"id": str(_uuid.uuid4()), "type": f"step_{i}", "label": lbl,
         "status": "pending"}
        for i, lbl in enumerate(labels)
    ]


def _worker_task_text(title: str, step: dict, plan_state: List[dict], n: int) -> str:
    parts = [
        f"[Step {step['i'] + 1} of {n} — {step['label']}] of the job: {title}"
        + (f" (your role: {step['role']})" if step.get("role") not in (None, "executor") else ""),
        "You are one helper on a bigger job. Do ONLY this step, then reply "
        "with a concise, factual result and list the sources (URLs) you used.",
        "",
        step["task"],
    ]
    inputs = [plan_state[d] for d in step["deps"] if plan_state[d].get("result")]
    if inputs:
        parts += ["", "Inputs from earlier steps:"]
        for d in inputs:
            parts += [f"### {d['label']}", d["result"][:SUPERVISOR_RESULT_MAX]]
    return "\n".join(parts)


async def run_supervised_plan(
    *,
    user_id: str,
    supervisor_job_id: str,
    title: str,
    plan: List[dict],
    agent_runner: Any,
    channel: Optional[str],
    executor_model: str,
    alternate_model: str,
    max_workers: int,
    deadline_s: float,
    worker_timeout_s: int,
    caps: Optional[dict] = None,
    chat_id: Optional[str] = None,
    message_id: Optional[str] = None,
    mission_id: Optional[str] = None,
    job_type_tag: Optional[str] = None,
    sink: Optional[Callable[[dict], Awaitable[None]]] = None,
    session_maker: Any = None,
) -> List[dict]:
    """Run ``plan`` on supervised workers and JOIN. Returns one entry per plan
    step: ``{step, label, model, outcome, result, sources}``.

    * Runnable steps (every ``depends_on`` done) start up to ``max_workers``
      at a time; a dependent step gets its upstreams' results in its task.
    * A failed/timed-out/rejected worker is re-spawned ONCE (``attempt=2``,
      a new row) on ``alternate_model``; a step whose upstream finally failed
      is ``skipped``.
    * ``deadline_s`` bounds the whole plan, retries included; workers still
      running at the deadline are cancelled and reported ``timeout``.
    * A cancel of the supervisor (Stop, hang-up, the tool's own timeout)
      cancels every child task and cascade-cancels the rows, then re-raises.
    """
    import uuid as _uuid
    from app.agent.subagent_dispatcher import REJECTION_CODES as _RC

    if session_maker is None:
        from app.db.database import async_session_maker as _maker
        session_maker = _maker

    loop = asyncio.get_running_loop()
    t_end = loop.time() + max(1.0, float(deadline_s))
    nonce = _uuid.uuid4().hex[:12]
    n = len(plan)
    state: List[dict] = [
        {"i": i, "label": p["label"], "task": p["task"],
         "role": p.get("role") or "executor",
         "deps": [d for d in (p.get("depends_on") or []) if isinstance(d, int) and 0 <= d < i],
         "status": "pending", "attempts": 0, "model": None, "result": "",
         "sources": [], "outcome": None, "error": None,
         "started_at": None, "completed_at": None}
        for i, p in enumerate(plan)
    ]
    # Card row of plan step 0 (see `_update_supervisor`): the worker's own
    # actions are filed under the card's row for its step, not the raw index.
    from app.agent.voice_jobs import current_voice_job as _cvj
    _tj = _cvj()
    step_offset = _tj.plan_offset if _tj is not None and _tj.job_id == supervisor_job_id else 0
    running: Dict[str, int] = {}          # worker job id -> plan index
    futures: Dict["asyncio.Future", str] = {}
    worker_seq = 0

    async def _publish(final: bool = False) -> None:
        await _update_supervisor(
            user_id=user_id, job_id=supervisor_job_id, title=title, state=state,
            final=final, chat_id=chat_id, message_id=message_id,
            job_type_tag=job_type_tag, sink=sink, session_maker=session_maker,
        )

    async def _start(s: dict) -> bool:
        nonlocal worker_seq
        remaining = t_end - loop.time()
        if remaining < 5:
            return False
        attempt = s["attempts"] + 1
        model = executor_model if attempt == 1 else alternate_model
        meta = {
            "parent_job_id": supervisor_job_id, "worker_index": worker_seq,
            "plan_step_index": s["i"], "role": s["role"], "attempt": attempt,
            "label": s["label"], "_nonce": nonce, "_sink": sink,
            "_step_index": step_offset + s["i"],
            "_steps_total": step_offset + n + 2, "_mission_id": mission_id,
            "_mission_title": title, "_chat_id": chat_id,
            "_message_id": message_id,
        }
        res = await spawn_subagent(
            user_id=user_id, task=_worker_task_text(title, s, state, n),
            label=s["label"], model=model,
            timeout_seconds=int(max(5, min(worker_timeout_s, remaining))),
            parent_job_id=supervisor_job_id, channel=channel,
            telegram_chat_id=None, agent_runner=agent_runner,
            session_maker=session_maker, supervised=True, worker_meta=meta,
            caps=caps,
        )
        if res.get("error"):
            code = res.get("error")
            if code in (_RC.PARENT_CAP, _RC.USER_CONCURRENT_CAP) and running:
                return False  # a slot frees when a running worker ends
            s["attempts"] = attempt
            s["error"] = res.get("message") or code
            return await _after_failure(s, "failed")
        fut = _WORKER_FUTURES.get(res["job_id"])
        if fut is None:  # collapsed onto an existing row: never ours to join
            s["attempts"] = attempt
            s["error"] = "duplicate worker"
            return await _after_failure(s, "failed")
        worker_seq += 1
        s["attempts"] = attempt
        s["model"] = model
        s["status"] = "running"
        s["started_at"] = s["started_at"] or datetime.utcnow()
        running[res["job_id"]] = s["i"]
        futures[fut] = res["job_id"]
        return True

    async def _after_failure(s: dict, outcome: str) -> bool:
        if s["attempts"] < 2 and (t_end - loop.time()) > 15:
            s["status"] = "pending"   # retried by the scheduler, another helper
            return False
        s["status"] = "failed"
        s["outcome"] = outcome
        s["completed_at"] = datetime.utcnow()
        return False

    async def _schedule() -> None:
        changed = True
        while changed:
            changed = False
            for s in state:
                if s["status"] != "pending":
                    continue
                dep_states = [state[d]["status"] for d in s["deps"]]
                if any(st in ("failed", "skipped") for st in dep_states):
                    s["status"], s["outcome"] = "skipped", "skipped"
                    s["completed_at"] = datetime.utcnow()
                    changed = True
                    continue
                if any(st != "done" for st in dep_states):
                    continue
                if len(running) >= max(1, int(max_workers)):
                    return
                before = (s["status"], s["attempts"])
                started = await _start(s)
                # A refused or collapsed spawn consumes an attempt and may
                # leave the step pending for its retry — go round again.
                if started or (s["status"], s["attempts"]) != before:
                    changed = True

    try:
        await _schedule()
        await _publish()
        while futures:
            remaining = t_end - loop.time()
            if remaining <= 0:
                break
            done, _ = await asyncio.wait(
                list(futures), timeout=remaining,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                break
            for fut in done:
                job_id = futures.pop(fut)
                i = running.pop(job_id, None)
                if i is None:
                    continue
                s = state[i]
                r = fut.result()
                if r.get("outcome") == "success":
                    s["status"], s["outcome"] = "done", "success"
                    s["result"] = (r.get("result") or "")[:SUPERVISOR_RESULT_MAX]
                    s["sources"] = list(r.get("sources") or [])
                    s["completed_at"] = datetime.utcnow()
                else:
                    s["error"] = r.get("error")
                    await _after_failure(s, r.get("outcome") or "failed")
            await _schedule()
            await _publish()
        # Deadline: whatever is still running is stopped and reported.
        if futures:
            for job_id in list(futures.values()):
                t = _WORKER_TASKS.get(job_id)
                if t is not None and not t.done():
                    t.cancel()
            await asyncio.wait(list(futures), timeout=3.0)
            for fut, job_id in list(futures.items()):
                s = state[running.pop(job_id)]
                s["status"], s["outcome"] = "failed", "timeout"
                s["completed_at"] = datetime.utcnow()
            futures.clear()
        for s in state:
            if s["status"] == "pending":
                s["status"], s["outcome"] = "failed", "timeout"
                s["completed_at"] = datetime.utcnow()
        await _publish(final=True)
    except asyncio.CancelledError:
        for job_id in list(futures.values()):
            t = _WORKER_TASKS.get(job_id)
            if t is not None and not t.done():
                t.cancel()
        try:
            from app.agent.subagent_dispatcher import cascade_cancel
            async with session_maker() as db:
                await cascade_cancel(db, supervisor_job_id,
                                     reason="Supervisor stopped.")
                await db.commit()
        except Exception:  # noqa: BLE001
            logger.warning("[supervisor] cascade cancel failed", exc_info=True)
        raise

    return [
        {"step": s["i"], "label": s["label"],
         "model": _display_model(s["model"]), "outcome": s["outcome"] or "failed",
         "result": s["result"], "sources": s["sources"],
         **({"error": str(s["error"])[:300]} if s["outcome"] != "success" and s["error"] else {})}
        for s in state
    ]


async def _update_supervisor(
    *, user_id: str, job_id: str, title: str, state: List[dict], final: bool,
    chat_id: Optional[str], message_id: Optional[str],
    job_type_tag: Optional[str], sink: Any, session_maker: Any,
) -> None:
    """Write the plan's step statuses onto the supervisor row and announce
    it (chat ``job_update`` + the SSE ``job.update``). The supervisor's own
    "Check and merge" step opens once every plan step has an outcome."""
    from app.agent.job_steps import counts, current_label, dump_steps, parse_steps
    from app.db.models import BuildJob, JobEvent

    def _iso(dt: Optional[datetime]) -> Optional[str]:
        return dt.replace(microsecond=(dt.microsecond // 1000) * 1000).isoformat() if dt else None

    status_map = {"pending": "pending", "running": "running", "done": "done",
                  "failed": "failed", "skipped": "skipped"}
    steps: List[dict] = []
    try:
        async with session_maker() as db:
            row = await db.get(BuildJob, job_id)
            if row is None:
                return
            # A TurnJob that follows this plan is the authority on its rows
            # (its own writer may have just rewritten the column).
            from app.agent.voice_jobs import current_voice_job
            _tj = current_voice_job()
            _tj = _tj if _tj is not None and _tj.job_id == job_id else None
            steps = (_tj.followed_steps() if _tj is not None else None) or parse_steps(row.steps_json)
            # Plan step i is card row `offset + i`: a card the turn was
            # already drawing keeps the steps that ran before the plan
            # (TurnJob.follow_steps), ahead of the plan's rows.
            offset = (_tj.plan_offset if _tj is not None
                      else max(0, len(steps) - (len(state) + 2)))
            for s in state:
                if offset + s["i"] >= len(steps):
                    continue
                st = steps[offset + s["i"]]
                st["status"] = status_map[s["status"]]
                if s["started_at"]:
                    st["started_at"] = _iso(s["started_at"])
                if s["completed_at"] and s["status"] in ("done", "failed", "skipped"):
                    st["completed_at"] = _iso(s["completed_at"])
                    if s["started_at"]:
                        dur = int(max(0.0, (s["completed_at"] - s["started_at"]).total_seconds() * 1000))
                        st["duration_ms"] = dur
                        st["durationMs"] = dur
            settled = all(s["status"] in ("done", "failed", "skipped") for s in state)
            if (final or settled) and len(steps) > offset + len(state):
                merge = steps[offset + len(state)]
                if merge.get("status") in (None, "pending"):
                    merge["status"] = "running"
                    merge["started_at"] = _iso(datetime.utcnow())
            row.steps_json = dump_steps(steps)
            if _tj is not None:
                _tj.follow_steps(steps)
            if row.status not in ("running", "queued"):
                await db.commit()
                return
            db.add(JobEvent(job_id=job_id, user_id=user_id, kind="info",
                            level="info",
                            label=(f"Helpers: {sum(1 for s in state if s['status'] in ('done','failed','skipped'))}/{len(state)}")[:200]))
            await db.commit()
    except Exception:  # noqa: BLE001
        logger.warning("[supervisor] step write failed job=%s", job_id, exc_info=True)
        return
    done, total = counts(steps)
    frame = {
        "type": "job_update",
        "job_id": job_id,
        "job_type": job_type_tag or _generic_job_type(),
        "name": title,
        "status": "running",
        "step": current_label(steps),
        "total_steps": total,
        "completed_steps": done,
        "chat_id": chat_id,
        "message_id": message_id,
        "steps": steps,
        "role": "supervisor",
        "workers_total": len(state),
        "workers_done": sum(1 for s in state if s["status"] in ("done", "failed", "skipped")),
    }
    try:
        from app.api.ws_chat import broadcast_to_user
        await broadcast_to_user(user_id, frame)
    except Exception:  # noqa: BLE001
        logger.debug("[supervisor] job_update broadcast skipped", exc_info=True)
    if sink is not None:
        try:
            await sink({**frame, "type": "job.update"})
        except Exception:  # noqa: BLE001
            logger.debug("[supervisor] job.update sink skipped", exc_info=True)
