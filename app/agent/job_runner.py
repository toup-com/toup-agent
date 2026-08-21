"""JobRunner — unified envelope for every "thing that runs".

PR 3 of the unified jobs/tasks/logs arc. See
``docs/architecture/jobs-investigation-2026-05-18.md`` D2 for the
design context.

This module introduces:

  - ``TaskSpec`` — the input contract for every job handler. Bundles
    the user, channel, source-kind discriminator, source_id back-link,
    prompt, and config_json that intake code passes in.
  - ``JobRunner.create_job(...)`` — single intake point that writes a
    ``BuildJob`` row with the unified-arc columns populated, honoring
    the composite UNIQUE ``(source_id, idempotency_key)``. On
    conflict the existing row is returned instead of inserting a
    duplicate. This collapses the existing ``UNIQUE (routine_id,
    scheduled_for_local_date)`` and ``UNIQUE (trigger_id,
    event_dedupe_id)`` semantics into one mechanism.
  - ``JobRunner.execute(job, spec)`` — dispatches by ``job.job_type``
    to a handler registered via ``JobRunner.register``. Handlers are
    coroutines ``(job, spec, db_unused) -> None`` responsible for
    setting ``job.status='completed'`` (or raising). If no handler is
    registered for the job_type, or the handler crashes, the runner
    marks the row failed with a sensible ``error_message``.

What this PR does NOT do
------------------------
Source paths (Auto Builder intake, dashboard task POST, chat-intent,
trigger runner, routine runner) still write ``BuildJob(...)``
directly — they are repointed in PR 4. The handler registry starts
empty; PR 4 wires in ``AppBuilderSkill._build_app``, ``AgentRunner.
run``, ``EmailReceivedHandler.execute``, and ``RoutineHandler.
execute`` against their respective ``job_type`` keys.

Threading model
---------------
No new poller / worker pool. ``JobRunner.execute`` is called inline
by the caller (typically inside an ``asyncio.create_task`` from the
intake site) — matches today's behaviour for Auto Builder and Agent
Task jobs. Trigger runs and routine runs already have their own
schedulers (APScheduler for routines, webhook for triggers); those
keep their schedulers and call ``JobRunner.execute`` post-create.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Awaitable, Callable, Optional

from sqlalchemy import select

logger = logging.getLogger(__name__)


# ── TaskSpec ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TaskSpec:
    """Input contract for every job handler.

    Constructed by intake code (chat-intent regex, dashboard POST,
    trigger inbound, routine fire, Auto Builder skill) and passed to
    ``JobRunner.create_job``. The runner copies the discriminator
    + back-link fields onto the ``BuildJob`` row before dispatching;
    the handler reads the rest at execute time.

    Frozen so test setups can't accidentally mutate a spec mid-
    dispatch — handlers must build their own internal state from
    the spec, not write back into it.
    """

    user_id: str
    # ``channel`` is the AgentRunner / Day-as-Chat channel — "web",
    # "chat_intent", "trigger", "routine", "app_builder". Distinct
    # from ``source_kind`` because some sources (e.g. chat_intent)
    # produce jobs that talk back on the same channel they came
    # from, but other sources (routine) write into a dedicated
    # channel.
    channel: str
    # ``source_kind`` ∈ {manual, chat_intent, trigger, routine,
    # app_builder_skill}. Matches the closed enum on
    # ``BuildJob.source_kind``.
    source_kind: str
    # ``source_id`` is a soft pointer to the originating row when
    # one exists: triggers.id, routines.id, conversations.id.
    # NULL for dashboard-input "manual" jobs.
    source_id: Optional[str] = None
    prompt: Optional[str] = None
    config_json: Optional[dict] = None
    # Day-as-Chat conversation this job is "of" — populated for
    # chat-intent and any future source that's chat-attached.
    conversation_id: Optional[str] = None


# Handler signature. The third positional is reserved for an
# optional shared db session in case the runner ever wants to pass
# one in; current handlers open their own sessions, so it's None
# today.
JobHandler = Callable[[Any, TaskSpec, Any], Awaitable[None]]


# ── JobRunner ────────────────────────────────────────────────────────


class JobRunner:
    """Unified envelope for every Job. See module docstring."""

    # Class-level registry so any module can import JobRunner and
    # register a handler at import time. ``register`` is idempotent —
    # re-registering the same job_type silently replaces, which is
    # what tests want.
    HANDLERS: dict[str, JobHandler] = {}

    def __init__(self, session_maker=None):
        if session_maker is None:
            from app.db.database import async_session_maker as _proxy

            session_maker = _proxy
        self._session_maker = session_maker

    @classmethod
    def register(cls, job_type: str, handler: JobHandler) -> None:
        """Register a handler for one job_type."""
        cls.HANDLERS[job_type] = handler

    @classmethod
    def registered_job_types(cls) -> set[str]:
        return set(cls.HANDLERS)

    # ── Create ────────────────────────────────────────────────────────

    async def create_job(
        self,
        *,
        job_type: str,
        spec: TaskSpec,
        title: str,
        prompt: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        app_id: Optional[str] = None,
        model: str = "",
        status: str = "queued",
        steps_json: Optional[str] = None,
        layer: int = 1,
        job_id: Optional[str] = None,
    ) -> Any:  # BuildJob — return type imported lazily to keep this
              # module's import surface narrow.
        """Insert a ``BuildJob`` row with the unified-arc columns
        populated. Honors composite UNIQUE on ``(source_id,
        idempotency_key)`` — on conflict, returns the *existing* row
        without writing.

        ``status`` defaults to ``"queued"`` (the production-shaped
        flow for trigger and routine fires, where the runner picks
        up the row asynchronously). Agent-task intake paths
        (dashboard POST, chat-intent regex, ``create_job`` tool)
        execute inline via ``asyncio.create_task`` and want
        ``status="running"`` from the start so the dashboard
        doesn't briefly show "queued."

        ``layer`` defaults to ``1`` (the existing default on the
        ``BuildJob`` model — auto_builder). Agent-task intake paths
        pass ``layer=0`` to match their pre-PR-4c behavior.

        ``steps_json`` is for sources that pre-populate phases at
        create time (``create_job`` tool with a user-supplied
        ``steps`` array). Defaults to ``"[]"``.

        ``job_id`` lets a caller that must know the id BEFORE the row
        exists supply its own (``voice_jobs.VoiceTurnJob``: the runner
        stamps ``job_id`` onto the tool frames of the very round that
        opens the job, while the INSERT runs off the turn's critical
        path). Everyone else omits it and gets a fresh uuid4.

        Callers can distinguish "fresh insert" vs "conflict-hit" by
        comparing the returned row's ``created_at`` to ``now``."""
        from app.db.models import BuildJob

        # ── 1. Idempotency conflict short-circuit ────────────────
        if idempotency_key is not None and spec.source_id is not None:
            async with self._session_maker() as db:
                existing = (
                    await db.execute(
                        select(BuildJob).where(
                            BuildJob.source_id == spec.source_id,
                            BuildJob.idempotency_key == idempotency_key,
                        )
                    )
                ).scalar_one_or_none()
                if existing is not None:
                    logger.info(
                        "[job_runner] idempotency_conflict job_id=%s "
                        "source=%s key=%s",
                        existing.id, spec.source_id, idempotency_key,
                    )
                    return existing

        # ── 2. Fresh insert ──────────────────────────────────────
        job_id = str(job_id or uuid.uuid4())
        async with self._session_maker() as db:
            # Phase 4 (sub-agent arc): copy spec.config_json onto
            # the BuildJob row so the handler can read its own
            # config without a JOIN. Existing non-subagent callers
            # don't populate spec.config_json so this is a no-op
            # (None default on the model).
            parent_job_id = None
            if spec.config_json:
                parent_job_id = spec.config_json.get("parent_job_id")
            job = BuildJob(
                id=job_id,
                user_id=spec.user_id,
                app_id=app_id,
                title=title,
                prompt=prompt if prompt is not None else (spec.prompt or ""),
                job_type=job_type,
                status=status,
                model=model,
                layer=layer,
                steps_json=steps_json if steps_json is not None else "[]",
                source_kind=spec.source_kind,
                source_id=spec.source_id,
                conversation_id=spec.conversation_id,
                idempotency_key=idempotency_key,
                config_json=spec.config_json,
                parent_job_id=parent_job_id,
            )
            db.add(job)
            await db.commit()
            await db.refresh(job)
        logger.info(
            "[job_runner] created job_id=%s job_type=%s status=%s source=%s/%s",
            job_id, job_type, status, spec.source_kind, spec.source_id,
        )
        return job

    # ── Execute ──────────────────────────────────────────────────────

    async def execute(self, job: Any, spec: TaskSpec) -> None:
        """Dispatch a Job to its registered handler.

        Contract for handlers:

          - On success: set ``job.status='completed'`` and
            ``job.completed_at`` before returning.
          - On failure: raise. The runner catches and marks
            ``status='failed'`` + ``error_message=<repr exc>`` +
            ``completed_at``. If the handler already set status,
            the runner preserves it (don't double-write).

        If no handler is registered for ``job.job_type``, the row is
        marked failed with a clear error_message and the call returns
        without raising — same fail-loud-but-don't-crash pattern as
        the trigger runner's missing-handler branch."""
        from app.db.models import BuildJob

        handler = self.HANDLERS.get(job.job_type)
        if handler is None:
            logger.error(
                "[job_runner] no_handler job_id=%s job_type=%s registered=%s",
                job.id, job.job_type, sorted(self.HANDLERS),
            )
            await self._mark_failed(
                job.id,
                f"no handler registered for job_type={job.job_type!r}",
            )
            return

        try:
            await handler(job, spec, None)
        except Exception as e:
            logger.exception(
                "[job_runner] handler_crash job_id=%s job_type=%s err=%s",
                job.id, job.job_type, e,
            )
            await self._mark_failed(job.id, repr(e)[:1000])
            raise

    async def _mark_failed(self, job_id: str, error_message: str) -> None:
        """Race-safe terminal failure write — preserves an existing
        terminal status set by the handler before raising (e.g.
        skipped_filter, skipped_rate_limit)."""
        from app.db.models import BuildJob

        async with self._session_maker() as db:
            fresh = await db.get(BuildJob, job_id)
            if fresh is None:
                return
            if fresh.status in ("completed", "failed"):
                return  # handler already wrote terminal state
            fresh.status = "failed"
            fresh.error_message = error_message
            fresh.completed_at = datetime.utcnow()
            title = fresh.title
            await db.commit()

        # Runner-internal job deaths reach the phone too — update_job /
        # subagent finalize cover their own terminal paths; this covers
        # handler crashes and missing handlers. Same dedup key as the
        # subagent reaper's failed notify so the dispatcher collapses a
        # double-report. Best-effort: a job must never fail on
        # notification plumbing.
        try:
            from app.services.agent_notify_client import notify

            label = (title or "Background job")[:80]
            await notify(
                event_kind="mission_failed",
                title=f"⚠️ {label} failed"[:200],
                body=(error_message or "")[:300] or None,
                data={
                    "route": "mission-control",
                    "mission_id": job_id,
                    "mission_title": label,
                    "kind": "job",
                    "urgent": True,
                    "dismiss_after_s": 900,
                },
                priority="high",
                dedup_key=f"{job_id}:failed",
            )
        except Exception as e:  # noqa: BLE001
            logger.debug(
                "[job_runner] mission_failed notify skipped job=%s: %s",
                job_id, e,
            )
