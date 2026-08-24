"""App builds: the ONE terminal close, and the watchdog that guarantees it.

Round 27 (2026-08-23). A card reading **"Build: Habit Garden · In progress ·
4/7 steps · 57%"** sat in a founder's chat for hours. The build had died
inside the looks-right loop — ``create`` done, ``verify`` done, ``review``
and ``edit`` done from one repair pass, ``look`` still ``running``, ``logo``
and ``present`` still ``pending``: four of seven, no failed row, nothing
terminal ever written. Three separate mechanisms each declined to resolve
it, and each for its own reason:

1. **The delivered-turn reconciler never looked.**
   ``reconcile_delivered_turn_jobs`` selects ``job_type == 'agent_task'``.
   An app build is ``job_type == 'auto_builder'``, so the minute-loop
   watchdog that exists precisely to stop zombie progress bars had the
   build lane out of scope. ``close_job_completed`` grew an auto_builder
   branch in round 23, but only the *finalizer* ever reached it — the
   sweep's own SELECT filtered the rows out one layer up.

2. **The 30-minute reaper flips a status and nothing else.**
   ``job_reaper.sweep_stalled_jobs`` sets ``status='cancelled'``, leaves
   ``steps_json`` untouched — ``look`` keeps spinning — and broadcasts a
   status-ONLY ``job_update``. The row becomes terminal; the card does not.

3. **Both clients read a build card off its STEPS.** The phone's
   ``JobProgressCard`` derives everything from the step list and never
   consults ``job.status`` at all::

       hasFailed  = steps.some(s => s.status === 'failed')   → "Didn't finish"
       isComplete = steps.every(s => s.status === 'done')    → "App built"
       otherwise  → `In progress · ${done}/${total} steps`

   and ``JobsContext`` drops a card from the live map on ``completed`` or
   ``failed`` — **not** on ``cancelled``. So the one word the abnormal-close
   paths wrote was the one word no client could act on: a build settled
   ``cancelled`` with rows resolved to ``skipped`` renders "In progress ·
   4/7 steps · 57%" forever, and stops being polled while it does it.

The rule this module makes true everywhere:

    **A build that did not publish is a build that failed to publish.**

``present`` is not one phase among seven — it is the job. A build that
stopped without it did not deliver an app, so the ``present`` row resolves
``failed`` ("Couldn't publish the app") and the job resolves ``failed``.
That is a fact about what happened, not an invented diagnosis, and it is
the only shape every surface already renders terminally: the phone says
"Didn't finish", the web card collapses to "Couldn't build X · Try again",
and ``JobsContext`` removes the card from the live map. ``cancelled`` is
gone from the build lane entirely — which is what makes the zombie
impossible by construction rather than merely unlikely.

Round 23's actual invariant is untouched and re-asserted in the tests: **no
phase may be invented done by a watchdog.** Rows that never reported back
are still ``skipped`` with their honest per-phase words; a row that was ever
done (``was_done``) still keeps its result; the done count never decreases.
The one row that changes meaning is the one whose absence defines the
failure.

Two things live here:

* :func:`settle_build` — the ONE way an app build reaches a terminal state
  without going through ``finish_job``. Every abnormal-close path calls it
  (the turn-end finalizer, the interrupted-turn sweep, boot recovery, the
  stall reaper, the watchdog below), so no two of them can disagree about
  what a dead build looks like. Robust to partial completion: a build whose
  ``present`` phase DID report done is closed ``completed``, because the app
  is published and only the bookkeeping died.

* :func:`sweep_stuck_builds` — the watchdog. Any build with no sign of life
  for :data:`BUILD_STALE_AFTER` is settled, and a terminal ``job_update``
  frame carrying the FULL step list (never a bare status) plus a terminal
  Live Activity push go out, so the card ends on every surface. Followed by
  :func:`assert_no_zombie_cards`, which logs at ERROR for anything still
  active past the window — by construction there is nothing to log.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import func, select, update

logger = logging.getLogger(__name__)

#: How long an app build may go without a single ``job_events`` heartbeat
#: before it is presumed dead.
#:
#: Every phase of the pipeline is individually budgeted — syntax 10s, smoke
#: 45s, the visual review 25s, the icon 45s — so the longest phase that can
#: legitimately hold the row silent is under a minute. What is NOT budgeted
#: is the model's own generation between two emits: ``create_app_file``'s
#: argument IS the whole app, and a large rewrite streams for minutes. This
#: is set at ~20x the longest measured phase and comfortably past any single
#: generation, and it is deliberately HALF the generic reaper's 30 minutes
#: so a build gets a build-shaped close before the generic net reaches it.
BUILD_STALE_AFTER = timedelta(minutes=15)

#: How far back the sweep looks. A week covers every card a user can still
#: scroll to in a live thread; older rows are history nobody is watching.
LOOKBACK = timedelta(days=7)

SWEEP_INTERVAL_S = 60
_BOOT_DELAY_S = 50

#: The job types this module owns. A build is identified by BOTH its type
#: and its ``app_id`` — the ``auto_builder`` type predates this pipeline and
#: a row without an app is not an app build.
BUILD_JOB_TYPE = "auto_builder"

#: Statuses a build can be settled FROM. ``paused`` is deliberately absent:
#: the token-limit pause legitimately sleeps for hours with a checkpoint.
SETTLEABLE = ("queued", "running")

#: ``reason`` → ``error_class``. The taxonomy already has words for each of
#: these; inventing a build-only class would give every reader a value it
#: does not route.
_REASON_ERROR_CLASS: Dict[str, str] = {
    "turn_end": "turn_interrupted",
    "turn_interrupted": "turn_interrupted",
    "present_crashed": "tool_failure",
    "restart": "infra_interrupted",
    "watchdog": "turn_interrupted",
    "reaper": "turn_interrupted",
    "manual": "turn_interrupted",
}

_STOPPED_MESSAGE = (
    "The build stopped before the app was published. Ask me to finish it."
)
_BROKEN_MESSAGE = "The app isn't ready to open yet — it needs a fix first."


@dataclass
class SettledBuild:
    """What a settle did, so the caller can log it and the sweep can count."""

    job_id: str
    user_id: str
    title: str
    status: str
    app_id: Optional[str] = None
    slug: Optional[str] = None
    chat_id: Optional[str] = None
    published: bool = False
    done: int = 0
    total: int = 0
    percent: int = 0
    steps: List[Dict[str, Any]] = field(default_factory=list)


def _phase_label(step_type: str, status: str) -> str:
    """The pipeline's own words, or nothing. Never a fabricated label."""
    try:
        from app.agent.skills.builtins.app_html.steps import phase_label
        return phase_label(step_type, status)
    except Exception:  # noqa: BLE001 - words are never worth a failed settle
        return ""


def _resolve_publish_row(steps: List[Dict[str, Any]], *, detail: str) -> bool:
    """Mark the ``present`` row failed. Returns True if the app WAS published.

    Called after :func:`settle_steps` has resolved everything else, so the
    row's status here is already terminal. Three cases:

    * ``done`` (or ever-done) — the app is live. Nothing to do; the caller
      closes the job ``completed``.
    * ``failed`` — the publish already reported its own failure, with its
      own detail. Left exactly as it is; overwriting it would replace a real
      diagnosis with a generic one.
    * anything else (``skipped`` from the settle, or missing entirely on a
      row written before ``present`` existed) — the publish did not happen,
      which is the definition of this build not finishing. The row says so.
    """
    row: Optional[Dict[str, Any]] = None
    for s in steps:
        if isinstance(s, dict) and s.get("type") == "present":
            row = s
    if row is not None and (row.get("status") == "done" or row.get("was_done")):
        return True
    if row is None:
        row = {"type": "present"}
        steps.append(row)
    if row.get("status") == "failed":
        return False
    row["status"] = "failed"
    row.pop("was_done", None)
    row.pop("recoverable", None)
    row["label"] = _phase_label("present", "failed") or "Couldn't publish the app"
    row["detail"] = detail
    row["rev"] = int(row.get("rev") or 0) + 1
    return False


async def settle_build(
    job_id: str,
    *,
    user_id: Optional[str] = None,
    now: Optional[datetime] = None,
    reason: str = "watchdog",
    announce: bool = True,
) -> Optional[SettledBuild]:
    """Terminalise one app build. The ONE abnormal close for the build lane.

    Opens and commits its OWN session on purpose: the turn-end finalizer
    only commits its session when a job came back CLOSED, so a settle that
    piggy-backed on the caller's transaction would be silently discarded.
    The guarded ``WHERE status IN (queued, running)`` re-checks at write
    time, so a publish racing this settle wins.

    Returns None when the row was not ours to settle — already terminal,
    paused, missing, not an app build, or another sweep got there first.
    """
    from app.db.database import async_session_maker
    from app.db.models import BuildJob, JobEvent
    from app.agent.skills.builtins.app_html.steps import (
        _HIGH_WATER_KEY, progress as _progress, public_steps, settle_steps,
    )

    now = now or datetime.utcnow()
    out: Optional[SettledBuild] = None
    try:
        async with async_session_maker() as db:
            job = await db.get(BuildJob, job_id)
            if job is None:
                return None
            if user_id and job.user_id != user_id:
                return None
            if job.status not in SETTLEABLE or job.paused_at is not None:
                return None
            if (getattr(job, "job_type", None) != BUILD_JOB_TYPE
                    or not getattr(job, "app_id", None)):
                return None

            try:
                steps = json.loads(job.steps_json) if job.steps_json else []
            except (TypeError, ValueError):
                steps = []
            if not isinstance(steps, list):
                steps = []
            steps = [s for s in steps if isinstance(s, dict)]

            # Round 23's rule, unchanged and shared: a row that never
            # reported back is `skipped` with its honest words, a row that
            # was ever done keeps its result, a `recoverable` failure
            # resolves terminal because nothing is going to fix it now.
            settle_steps(steps, detail="the build stopped before this ran")
            published = _resolve_publish_row(
                steps, detail="the build stopped before the app went live",
            )

            any_failed = any(s.get("status") == "failed" for s in steps)
            final = "failed" if any_failed else "completed"
            if published and not any_failed:
                final = "completed"

            cfg = dict(job.config_json or {})
            cfg["settled_reason"] = f"build_watchdog:{reason}"
            cfg["settled_at"] = now.isoformat()
            done, total, percent, cfg = _progress(steps, cfg, job_id=job_id)
            if final == "completed":
                # Belt and braces, exactly as `finish_job` does it: a build
                # that published is all the way along, and a high-water mark
                # from an earlier frame cannot exceed 100.
                percent = 100
                cfg[_HIGH_WATER_KEY] = 100

            values: Dict[str, Any] = {
                "status": final,
                "completed_at": now,
                "steps_json": json.dumps(steps),
                "config_json": cfg,
            }
            if final == "completed":
                # The app is live and nothing is failing, so any error copy
                # left over from a phase that was repaired is stale — and
                # `user_message` is the ONE field a client may render.
                values["user_message"] = None
                values["error_class"] = None
            if final == "failed":
                values["user_message"] = (
                    getattr(job, "user_message", None)
                    or (_BROKEN_MESSAGE if published else _STOPPED_MESSAGE)
                )
                values["error_class"] = _REASON_ERROR_CLASS.get(
                    reason, "tool_failure",
                )
            res = await db.execute(
                update(BuildJob)
                .where(BuildJob.id == job_id,
                       BuildJob.user_id == job.user_id,
                       BuildJob.status.in_(SETTLEABLE))
                .values(**values)
                .returning(BuildJob.id)
            )
            if res.first() is None:
                return None

            db.add(JobEvent(
                job_id=job_id, user_id=job.user_id, kind="info",
                level="info" if final == "completed" else "error",
                status=final, ts=now,
                label=("Published, closed by watchdog" if published
                       else "Build stopped before publish")[:200],
                metadata_json=json.dumps({"reason": reason,
                                          "published": published}),
            ))

            await db.commit()
            out = SettledBuild(
                job_id=job_id, user_id=job.user_id,
                title=(job.title or "").replace("Build: ", ""),
                status=final, app_id=job.app_id,
                chat_id=getattr(job, "conversation_id", None),
                published=published, done=done, total=total, percent=percent,
                steps=public_steps(steps),
            )
    except Exception:  # noqa: BLE001 - a failed settle leaves the reaper's net
        logger.warning("[build_watchdog] settle failed for %s", job_id[:8],
                       exc_info=True)
        return None

    # The App row is touched in its OWN session, AFTER the job is committed.
    # Not tidiness: `apps` is one of the AGENT_ONLY tables, so under
    # RUN_MODE=platform (and in any test whose fixture creates only the job
    # tables) the SELECT raises `no such table: apps` — and reading it inside
    # the transaction above put that exception on the settle's own path,
    # where the blanket `except` swallowed it and returned None. The card
    # then stayed exactly as stuck as before, with a warning in the log
    # saying so. A cosmetic row on a second table must not be able to veto
    # the close.
    out.slug = await _sync_app_row(out.app_id, published=out.published)

    logger.warning(
        "[build_watchdog] settled build %s (%s) as %s — %d/%d, reason=%s",
        out.job_id[:8], out.title[:60], out.status, out.done, out.total, reason,
    )
    if announce:
        await announce_settled(out)
    return out


async def _sync_app_row(app_id: Optional[str], *, published: bool) -> Optional[str]:
    """Take the app off ``building``, and hand back its slug. Never raises.

    A build that died mid-flight leaves its App row on ``building``, which is
    its own spinner one surface over — in the library rather than the chat.
    A build that DID publish and then died leaves the same row on
    ``building`` with a live app behind it. Both are resolved here, and the
    slug comes back either way because the published branch needs it to
    address the ``app_ready`` frame.

    Own session, and a total swallow: ``apps`` is AGENT_ONLY, so this is the
    one part of a settle that can simply be absent.
    """
    if not app_id:
        return None
    try:
        from app.db.database import async_session_maker
        from app.db.models import App as AppModel
        async with async_session_maker() as db:
            row = await db.get(AppModel, app_id)
            if row is None:
                return None
            slug = row.slug
            if row.status == "building":
                row.status = "ready" if published else "error"
                await db.commit()
            return slug
    except Exception:  # noqa: BLE001 - never worth a settle
        logger.debug("[build_watchdog] app row sync skipped for %s",
                     app_id, exc_info=True)
        return None


async def announce_settled(settled: SettledBuild) -> None:
    """Tell every surface the build is over. Best-effort, never raises.

    The frame carries the FULL step list, the counts and the percent — never
    a bare status. Both clients derive a build card's state from its steps
    (the phone card does not read ``job.status`` at all), so a status-only
    terminal frame is a frame that changes nothing: it is exactly what the
    stall reaper has been broadcasting at these rows, and exactly why the
    card kept saying "In progress" under a row that was already closed.
    """
    try:
        from app.api.ws_chat import broadcast_to_user
        await broadcast_to_user(settled.user_id, {
            "type": "job_update",
            "job_id": settled.job_id,
            "job_type": BUILD_JOB_TYPE,
            "name": settled.title,
            "status": settled.status,
            "step": _phase_label(
                "present", "done" if settled.published else "failed",
            ),
            "total_steps": settled.total,
            "completed_steps": settled.done,
            "percent": settled.percent,
            "steps": settled.steps,
            "app_id": settled.app_id,
            "chat_id": settled.chat_id,
        })
    except Exception:  # noqa: BLE001 - the DB row is already honest
        logger.debug("[build_watchdog] terminal broadcast failed", exc_info=True)

    # ── A published build gets the frame that says the app is THERE ──
    # This is the other half of "robust to partial completion", and it is
    # not decoration. `app_ready` (with `app_artifact` ahead of it) is what
    # the happy path sends after `finish_job`, and it is what puts a client
    # into its "App built" state directly rather than by inference. The
    # phone's `isComplete` is `appReady || steps.every(done)` — and a real
    # build legitimately ends with rows that are `skipped` (a `logo` that
    # timed out, a `look` the renderer could not run), so WITHOUT this frame
    # a published-but-unfinalised build would settle `completed` in the
    # database and still read "In progress · 4/5 steps" on the phone. The
    # zombie, one branch over.
    if settled.published and settled.app_id:
        try:
            from app.agent.skills.builtins.app_html.steps import announce_ready
            await announce_ready(
                user_id=settled.user_id, job_id=settled.job_id,
                app_id=settled.app_id, title=settled.title,
                slug=settled.slug,
            )
        except Exception:  # noqa: BLE001
            logger.debug("[build_watchdog] app_ready replay failed",
                         exc_info=True)

    # The phone card is closed ONLY by a terminal notification — a DB write
    # does nothing to it. Addressed with the job's OWN conversation_id
    # rather than the turn context, which is empty in a sweep (round 25,
    # item 7: without a chat id the push lands on `job_mission_id`'s raw-job
    # fallback, i.e. a card nobody is looking at).
    try:
        from app.agent.subagent_orchestrator import _notify_job_event
        await _notify_job_event(
            job_id=settled.job_id,
            label=settled.title or "your app",
            kind=("mission_completed" if settled.status == "completed"
                  else "mission_failed"),
            title=settled.title or "Your app",
            body=(f"{settled.title} is ready to open." if settled.status == "completed"
                  else "This build stopped before it finished. Ask me to pick "
                       "it up again."),
            progress=settled.percent,
            step_name=_phase_label(
                "present", "done" if settled.published else "failed",
            ),
            steps_done=settled.done, steps_total=settled.total,
            job_type=BUILD_JOB_TYPE,
            chat_id=settled.chat_id,
            route="chat" if settled.chat_id else "mission-control",
            dismiss_after_s=900,
            dedup_suffix=f"settled:{settled.status}",
            urgent=False,
        )
    except Exception:  # noqa: BLE001
        logger.debug("[build_watchdog] terminal push failed", exc_info=True)


def is_build_row(job: Any) -> bool:
    """True for a row this module owns — an app build with an app."""
    return (getattr(job, "job_type", None) == BUILD_JOB_TYPE
            and bool(getattr(job, "app_id", None)))


async def _stale_build_ids(
    db: Any, now: datetime, *, stale_after: timedelta,
) -> List[str]:
    """Ids of every app build with no sign of life for ``stale_after``.

    Sign of life is the newest ``job_events`` row, else ``created_at`` — the
    same liveness definition the stall reaper uses, so the two nets cannot
    disagree about which rows are moving.
    """
    from app.db.models import BuildJob, JobEvent

    cutoff = now - stale_after
    rows = list((await db.execute(
        select(BuildJob.id, BuildJob.created_at).where(
            BuildJob.status.in_(SETTLEABLE),
            BuildJob.job_type == BUILD_JOB_TYPE,
            BuildJob.app_id.isnot(None),
            BuildJob.paused_at.is_(None),
            BuildJob.created_at < cutoff,
            BuildJob.created_at > now - LOOKBACK,
        )
    )).all())
    if not rows:
        return []
    ids = [r[0] for r in rows]
    last_event = dict((await db.execute(
        select(JobEvent.job_id, func.max(JobEvent.ts))
        .where(JobEvent.job_id.in_(ids))
        .group_by(JobEvent.job_id)
    )).all())
    out: List[str] = []
    for jid, created_at in rows:
        last_alive = last_event.get(jid) or created_at
        if last_alive is None or last_alive < cutoff:
            out.append(jid)
    return out


async def sweep_stuck_builds(
    now: Optional[datetime] = None,
    *,
    stale_after: timedelta = BUILD_STALE_AFTER,
    dry_run: bool = False,
) -> int:
    """Settle every app build that has gone silent. Returns how many closed.

    ``dry_run`` counts without writing — what the fleet operator script
    reports before it is allowed to change anything.
    """
    from app.db.database import async_session_maker

    now = now or datetime.utcnow()
    async with async_session_maker() as db:
        stale = await _stale_build_ids(db, now, stale_after=stale_after)
    if not stale:
        return 0
    if dry_run:
        logger.info("[build_watchdog] %d stale build(s) would be settled",
                    len(stale))
        return len(stale)

    closed = 0
    for jid in stale:
        try:
            if await settle_build(jid, now=now, reason="watchdog"):
                closed += 1
        except Exception:  # noqa: BLE001 - one bad row never stops the sweep
            logger.exception("[build_watchdog] failed on build %s", jid)
    return closed


async def assert_no_zombie_cards(
    now: Optional[datetime] = None,
    *,
    stale_after: timedelta = BUILD_STALE_AFTER,
) -> List[str]:
    """Log every build card still reading "In progress" past the window.

    Round 27, item 4. After :func:`sweep_stuck_builds` has run there is
    nothing here to find — a card older than its watchdog window is
    impossible by construction, so anything this returns is a defect in the
    construction and it logs like one. Returns the offending ids so a test
    can assert the check itself can fire.
    """
    from app.db.database import async_session_maker
    from app.db.models import BuildJob

    now = now or datetime.utcnow()
    async with async_session_maker() as db:
        stale = await _stale_build_ids(db, now, stale_after=stale_after)
        if not stale:
            return []
        rows = list((await db.execute(
            select(BuildJob.id, BuildJob.title, BuildJob.status,
                   BuildJob.created_at)
            .where(BuildJob.id.in_(stale))
        )).all())
    minutes = int(stale_after.total_seconds() // 60)
    for jid, title, status, created_at in rows:
        age = ""
        if created_at:
            age = f", opened {int((now - created_at).total_seconds() // 60)}m ago"
        logger.error(
            "[build_watchdog] ZOMBIE CARD: build %s (%s) is still %r with no "
            "sign of life for over %dm%s — this is impossible by "
            "construction; a close path is not calling settle_build",
            jid[:8], (title or "")[:60], status, minutes, age,
        )
    return [r[0] for r in rows]


async def watchdog_loop() -> None:
    """Forever loop for agent_main. Boot delay lets init_db / pool bind settle.

    This IS the fleet-wide cleanup: every agent container runs it against its
    own tenant, so a zombie anywhere in the fleet resolves itself within a
    minute of this image rolling out. ``scripts/sweep_zombie_builds.py`` is
    the operator's view of the same sweep, not a second implementation.
    """
    await asyncio.sleep(_BOOT_DELAY_S)
    while True:
        try:
            n = await sweep_stuck_builds()
            if n:
                logger.info("[build_watchdog] settled %d stuck build(s)", n)
            await assert_no_zombie_cards()
        except Exception as e:  # noqa: BLE001 - pre-bind lobby / DB hiccup
            logger.warning("[build_watchdog] sweep failed: %s", e)
        await asyncio.sleep(SWEEP_INTERVAL_S)
