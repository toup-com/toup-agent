"""Step / activity reporting for the HTML-artifact tools.

This deliberately adds **no new UI contract**. It reuses the two frames the
chat already understands (``MIGRATION_INVENTORY.md`` §5a/§5b):

* ``job_update`` — creates and then updates one ``role:'job'`` card per app
  in ``ChatPage``, and one row in Mission Control / the activity feed. Backed
  by a real ``BuildJob`` (``steps_json``) plus one ``JobEvent`` per phase
  transition, so the feed can render durations without re-parsing the blob.
* ``app_ready`` — flips that card to *completed* and attaches ``app_id``,
  which is what turns it into an openable artifact card.

One job per app, not one per turn: ``JobRunner.create_job`` honours the
composite UNIQUE on ``(source_id, idempotency_key)``, so a second
``create_app_file`` for the same slug re-attaches to the same card instead of
stacking a new one on every edit.

Everything here is **fail-open**. A reporting outage must never turn a
successful file write into a failed tool call — the lesson
``_meter_flat_tool`` already encodes one layer down.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

#: Namespace for deterministic app ids. Same (user, slug) → same id forever,
#: so the App row, the BuildJob and the chat card all agree across turns,
#: restarts and container recreations without a lookup table.
_APP_NS = uuid.UUID("6f1d5f2e-9b3c-4a71-8d55-0c9a2f7e41b0")

#: The five phases, in the order the pipeline walks them.
#:
#: Round 18. A caller used to pass its own ``label=`` and the card rendered
#: whatever it got, which is how a build came to show a step called
#: ``9299 bytes``, one called ``exit 1``, one called ``revision 2`` and one
#: called ``wc -c nokia-snake.html``. Those are a size, a POSIX status, a
#: counter and a shell command — four kinds of machine output presented to
#: someone who asked for a snake game, in the slot reserved for "what is
#: happening now".
#:
#: So labels are no longer an argument. :func:`phase_label` owns every word
#: that can appear on a step, keyed on the phase and its state; a caller may
#: contribute a ``detail``, which the card shows only when the row is opened.
#: There is no code path left through which a byte count can become a label.
_PHASE_WORDS: Dict[str, Dict[str, str]] = {
    "create":  {"pending": "Write the app",     "running": "Writing the app",
                "done": "Wrote the app file",   "failed": "Couldn't write the app file"},
    "review":  {"pending": "Read the app",      "running": "Reading the app",
                "done": "Read the app",         "failed": "Couldn't read the app"},
    "edit":    {"pending": "Update the app",    "running": "Updating the app",
                "done": "Updated the app",      "failed": "Couldn't update the app"},
    "verify":  {"pending": "Check the app",     "running": "Checking the app",
                "done": "Checked the app",      "failed": "The app needs a fix"},
    "present": {"pending": "Publish the app",   "running": "Publishing the app",
                "done": "Published the app",    "failed": "Couldn't publish the app"},
}

STEP_TYPES: List[str] = ["create", "review", "edit", "verify", "present"]

SOURCE_HTML_ARTIFACT = "html_artifact"


#: What the card says under the header when a phase fails.
#:
#: ``BuildJob.user_message`` is the ONLY error text the clients are allowed to
#: render, and this pipeline never wrote one — so a failed app build fell
#: through to the mobile card's fallback, which reads:
#:
#:     The build stopped partway. Nothing was changed.
#:
#: For a build whose `create` phase had already written the file, that is
#: simply false, and it is false in the direction that matters: it tells
#: someone their work is gone when it is on disk and one edit from working.
#: A phase knows what it was doing, so it can say what actually happened.
_PHASE_USER_MESSAGE: Dict[str, str] = {
    "create":  "The app file couldn't be saved. Try again.",
    "review":  "I couldn't read the app back just now. Try again.",
    "edit":    "That change didn't go in — the app is exactly as it was.",
    "verify":  "The app has a problem in its code, so I haven't shown it to "
               "you yet. Ask me to fix it.",
    "present": "The app isn't ready to open yet — it needs a fix first.",
}

#: Closed enum, from job_status. A phase of the pipeline failing IS a tool
#: failure; inventing a new class would give every reader a value it does not
#: route.
_ERROR_CLASS = "tool_failure"


def phase_label(step_type: str, status: str) -> str:
    """The ONE place a step's user-visible words are chosen."""
    words = _PHASE_WORDS.get(step_type)
    if not words:
        return "Working on your app"
    return words.get(status) or words["running"]


def app_id_for(user_id: str, slug: str) -> str:
    return str(uuid.uuid5(_APP_NS, f"{user_id or 'local'}:{slug}"))


async def job_id_for_slug(user_id: str, slug: str) -> Optional[str]:
    """The build job backing this app, by the key the surfaces join on.

    Exists so the TURN can record which job its build card is, the same way
    it already records which app it presented. Without it the job id lived
    only on the `job_update` frames — a live-only channel — so a thread
    reopened after a relaunch had the artifact card (a persisted field) and
    no build card, and the row fell back to the generic "N actions" rail
    that round 18 exists to remove. Verified on a simulator: correct while
    the turn was on screen, degraded on the next launch.

    The derivation stays HERE rather than in the client, because `_APP_NS`
    and the `user:slug` key are this module's private business and a client
    that recomputed them would break silently the day either changed.
    """
    return await _existing_job_for_app(app_id_for(user_id, slug))


def initial_steps() -> List[Dict[str, Any]]:
    return [
        {"id": str(uuid.uuid4()), "type": t,
         "label": phase_label(t, "pending"), "status": "pending"}
        for t in STEP_TYPES
    ]


async def _broadcast(user_id: str, payload: Dict[str, Any]) -> None:
    if not user_id:
        return
    try:
        from app.api.ws_chat import broadcast_to_user
        await broadcast_to_user(user_id, payload)
    except Exception:
        logger.debug("[app_html] ws broadcast failed (non-fatal)", exc_info=True)


async def _existing_job_for_app(app_id: str) -> Optional[str]:
    """The job already backing this app, if any. Any status: a second card
    for an app that has one is the defect, whether the first is running or
    finished three turns ago."""
    try:
        from sqlalchemy import select
        from app.db.database import async_session_maker
        from app.db.models import BuildJob
        async with async_session_maker() as db:
            row = (await db.execute(
                select(BuildJob.id)
                .where(BuildJob.app_id == app_id)
                .order_by(BuildJob.created_at.desc())
                .limit(1)
            )).first()
            return row[0] if row else None
    except Exception:
        logger.debug("[app_html] existing-job lookup failed", exc_info=True)
        return None


async def _adopt_turn_job(user_id: str, app_id: str, title: str) -> Optional[str]:
    """Take over the job the MODEL opened for this turn, instead of adding
    a second one beside it.

    Round 15. A build showed the user two cards: "Building your app" and
    "Build: Nokia Snake Classic". The first is the model narrating its own
    work through the ordinary `create_job` tool; the second is this pipeline
    opening the card it drives. Both are legitimate on their own and together
    they are one build rendered twice, with two progress bars that disagree.

    One job wins, and it is the one already on screen. The row is retyped to
    an app build — `job_type` is the column every surface discriminates on,
    so this is what moves the card from "task" rendering to "build"
    rendering — and its steps are replaced with this pipeline's phases.

    Four conditions, each preventing a different way of getting this wrong:

    * **Created this turn.** The registry is per-turn, so an older job can
      never be swept up.
    * **Exactly one.** A turn that opened two or more jobs is tracking two or
      more things, and guessing which one is the build is worse than a second
      card. "Build me a snake game and research X" must not retitle the
      research.
    * **No app of its own.** Otherwise the second app of a turn would hijack
      the first one's card.
    * **Nothing reported against it yet** — no step marked `done`. A job the
      model has already been ticking off is work in progress; replacing its
      steps with this pipeline's would erase what the user watched happen.
      (`create_job` opens step 0 as *running*, not done, so a genuinely fresh
      job passes.)
    """
    try:
        from app.agent.tool_executor import created_job_ids
        candidates = created_job_ids()
    except Exception:
        return None
    if len(candidates) != 1:
        return None
    try:
        from app.db.database import async_session_maker
        from app.db.models import BuildJob
        async with async_session_maker() as db:
            for jid in candidates:
                job = await db.get(BuildJob, jid)
                if job is None or job.user_id != user_id:
                    continue
                if job.status not in ("queued", "running"):
                    continue
                if job.app_id:
                    continue  # already drives some app's card
                try:
                    existing_steps = json.loads(job.steps_json) if job.steps_json else []
                except (TypeError, ValueError):
                    existing_steps = []
                if any(s.get("status") == "done" for s in existing_steps
                       if isinstance(s, dict)):
                    continue  # real progress already recorded against it
                job.app_id = app_id
                job.job_type = "auto_builder"
                job.layer = 1
                job.title = f"Build: {title}"
                job.steps_json = json.dumps(initial_steps())
                # NOT source_id/idempotency_key: those carry a composite
                # UNIQUE, and stamping this pipeline's pair onto an adopted
                # row would collide with the row of an earlier build of the
                # same slug. `app_id` is what the surfaces join on.
                cfg = dict(job.config_json or {})
                cfg["pipeline"] = SOURCE_HTML_ARTIFACT
                cfg["adopted_by"] = "app_html"
                job.config_json = cfg
                await db.commit()
                logger.info(
                    "[app_html] adopted this turn's job %s as the build card "
                    "for %s", jid[:8], app_id[:8],
                )
                return job.id
    except Exception:
        logger.debug("[app_html] job adoption failed (non-fatal)", exc_info=True)
    return None


async def ensure_job(user_id: str, slug: str, title: str) -> Optional[str]:
    """Find-or-create the ONE BuildJob backing this app's card.

    Order matters: an app that already has a job keeps it (so an edit three
    turns later updates the same card), otherwise this turn's model-created
    job is adopted, and only if there is neither do we open a new one.
    """
    app_id = app_id_for(user_id, slug)
    existing = await _existing_job_for_app(app_id)
    if existing:
        return existing
    adopted = await _adopt_turn_job(user_id, app_id, title)
    if adopted:
        return adopted
    try:
        from app.agent.job_runner import JobRunner, TaskSpec
        spec = TaskSpec(
            user_id=user_id,
            channel="app_builder",
            source_kind="app_builder_skill",
            source_id=app_id,
        )
        job = await JobRunner().create_job(
            # Deliberately the EXISTING job type. Every surface that renders
            # an app build — the Jobs tab, the chat card, the Live Activity
            # lane — discriminates on this column, so a new value would mean
            # a new artifact rendering nowhere. The pipeline changed; the
            # thing being built did not.
            job_type="auto_builder",
            spec=spec,
            title=f"Build: {title}",
            prompt=f"Single-file HTML app: {title}",
            status="running",
            steps_json=json.dumps(initial_steps()),
            app_id=app_id,
            idempotency_key=f"html:{slug}",
            layer=1,
        )
        return job.id
    except Exception:
        logger.warning("[app_html] could not open a job for %s", slug, exc_info=True)
        return None


async def emit_step(
    *,
    user_id: str,
    job_id: Optional[str],
    step_type: str,
    status: str,
    detail: str = "",
) -> None:
    """Mark one phase and push it to the chat card + activity feed.

    ``status`` ∈ {running, done, failed}. The label is derived from
    ``step_type`` and ``status`` (see :func:`phase_label`) and is NOT a
    parameter — see the note on :data:`_PHASE_WORDS`.
    """
    if not job_id:
        return
    label = phase_label(step_type, status)
    detail = (detail or "").strip()[:200]
    try:
        from app.db.database import async_session_maker
        from app.db.models import BuildJob, JobEvent

        duration_ms: Optional[int] = None
        job_status = "running"
        job_title = label
        total = completed = 0

        async with async_session_maker() as db:
            job = await db.get(BuildJob, job_id)
            if not job:
                return
            steps = json.loads(job.steps_json) if job.steps_json else initial_steps()
            step_dict = None
            for s in steps:
                if s.get("type") != step_type:
                    continue
                s["label"] = label
                s["status"] = status
                if detail:
                    s["detail"] = detail
                if status == "running":
                    s["started_at"] = datetime.utcnow().isoformat()
                elif s.get("started_at"):
                    try:
                        started = datetime.fromisoformat(s["started_at"])
                        duration_ms = int(
                            (datetime.utcnow() - started).total_seconds() * 1000
                        )
                        s["duration_ms"] = duration_ms
                    except ValueError:
                        pass
                step_dict = s
                break

            if step_dict is None:
                return

            job.steps_json = json.dumps(steps)
            # Read off the steps, not latched from this one emit. A `verify`
            # that failed and was then repaired and re-run is not a failed
            # build — the retry overwrites the same row (steps are keyed by
            # type), so the only honest reading of "is this job failing" is
            # "is any step failing RIGHT NOW".
            #
            # A `completed` job is deliberately left alone when nothing is
            # failing: an edit three turns later re-opens the same card, and
            # flipping it back to *running* on a `view_app_file` that is never
            # followed by a `present_app` would leave a finished build
            # spinning forever.
            any_failed = any(s.get("status") == "failed" for s in steps
                             if isinstance(s, dict))
            if any_failed:
                job.status = "failed"
                # Say which phase, in the field the clients are allowed to
                # render. Cleared below when the repair lands, so a card that
                # recovered does not keep an explanation of a fixed problem.
                if hasattr(job, "user_message"):
                    job.user_message = _PHASE_USER_MESSAGE.get(
                        step_type, "Something went wrong building this app. "
                                   "Try again.")
                if hasattr(job, "error_class"):
                    job.error_class = _ERROR_CLASS
            else:
                if job.status in ("queued", "failed"):
                    job.status = "running"
                if getattr(job, "error_class", None) == _ERROR_CLASS:
                    job.error_class = None
                    job.user_message = None

            try:
                meta: Dict[str, Any] = {"phase_type": step_type, "pipeline": "html_artifact"}
                if duration_ms is not None:
                    meta["duration_ms"] = duration_ms
                if detail:
                    meta["detail"] = detail
                db.add(JobEvent(
                    job_id=job_id,
                    user_id=user_id,
                    kind="phase_started" if status == "running" else "phase_completed",
                    label=label,
                    status=status,
                    level="error" if status == "failed" else "info",
                    metadata_json=json.dumps(meta),
                ))
            except Exception:
                logger.debug("[app_html] job_events write failed", exc_info=True)

            await db.commit()
            job_status = job.status
            job_title = (job.title or "").replace("Build: ", "") or label
            total = len(steps)
            completed = sum(1 for s in steps if s.get("status") == "done")

        await _broadcast(user_id, {
            "type": "job_update",
            "job_id": job_id,
            "name": job_title,
            "status": job_status,
            "step": label,
            "total_steps": total,
            "completed_steps": completed,
        })
    except Exception:
        logger.debug("[app_html] emit_step failed (non-fatal)", exc_info=True)


async def finish_job(user_id: str, job_id: Optional[str]) -> Optional[str]:
    """Close the card, and make its arithmetic true.

    Two things were wrong with the finished card and they were the same bug
    seen from both ends. It always opened with all five phases pending, but a
    plain build only ever walks three of them (write, check, publish) — so the
    steps list settled at "3/5" while the header said *Built* and the bar was
    hard-coded to 100%. The user could read 100%, 3 of 5, and a row with no
    tick, all in one card.

    Phases that never happened are dropped here rather than counted as
    outstanding: they are not work left undone, they are work the build did
    not need. What is left is every phase that actually ran, all of them
    ``done``, which is a card whose bar, whose count and whose ticks agree.

    Returns the final status, so the caller can tell a finished build from one
    that closed with a phase still broken.
    """
    if not job_id:
        return None
    final = "completed"
    try:
        from app.db.database import async_session_maker
        from app.db.models import BuildJob
        async with async_session_maker() as db:
            job = await db.get(BuildJob, job_id)
            if not job:
                return None
            try:
                steps = json.loads(job.steps_json) if job.steps_json else []
            except (TypeError, ValueError):
                steps = []
            walked = [s for s in steps
                      if isinstance(s, dict) and s.get("status") != "pending"]
            # A phase left `running` at the end never reported back. Calling
            # it done would be inventing a result; calling it failed would be
            # inventing a diagnosis. It is dropped for the same reason a
            # pending one is — nothing is known about it.
            walked = [s for s in walked if s.get("status") != "running"]
            if any(s.get("status") == "failed" for s in walked):
                final = "failed"
            job.steps_json = json.dumps(walked or steps)
            job.status = final
            job.completed_at = datetime.utcnow()
            title = (job.title or "").replace("Build: ", "")
            await db.commit()

        total = len(walked)
        await _broadcast(user_id, {
            "type": "job_update",
            "job_id": job_id,
            "name": title,
            "status": final,
            "step": phase_label("present", "done") if final == "completed" else "",
            "total_steps": total,
            "completed_steps": sum(
                1 for s in walked if s.get("status") == "done"
            ),
        })
    except Exception:
        logger.debug("[app_html] finish_job failed (non-fatal)", exc_info=True)
    return final


async def upsert_app_row(
    *,
    user_id: str,
    slug: str,
    title: str,
    html_path: str,
    size_bytes: int,
    job_id: Optional[str],
) -> Optional[str]:
    """Create-or-refresh the ``apps`` row so the artifact is listable.

    ``source='html_artifact'`` is the discriminator every read path uses to
    tell a single-file artifact from a legacy Expo project — it rides on the
    existing ``AppResponse.source`` field, so no schema change is needed.
    """
    try:
        from sqlalchemy import select
        from app.db.database import async_session_maker
        from app.db.models import App

        app_id = app_id_for(user_id, slug)
        async with async_session_maker() as db:
            row = await db.get(App, app_id)
            if row is None:
                # A legacy Expo app may already own this slug (slug is UNIQUE).
                existing = (
                    await db.execute(select(App).where(App.slug == slug))
                ).scalar_one_or_none()
                if existing is not None and existing.id != app_id:
                    logger.warning(
                        "[app_html] slug %r already owned by app %s — "
                        "refreshing that row in place", slug, existing.id[:8],
                    )
                    row = existing
            if row is None:
                row = App(
                    id=app_id,
                    user_id=user_id,
                    name=title,
                    slug=slug,
                    app_dir=html_path,
                    source=SOURCE_HTML_ARTIFACT,
                )
                db.add(row)
            row.name = title or row.name
            row.status = "ready"
            row.source = SOURCE_HTML_ARTIFACT
            row.app_dir = html_path
            row.platforms = "web,ios"
            row.db_type = "none"
            row.deps_json = "{}"
            row.files_json = json.dumps({"size_bytes": size_bytes})
            if job_id:
                row.build_job_id = job_id
            row.updated_at = datetime.utcnow()
            await db.commit()
            return row.id
    except Exception:
        logger.warning("[app_html] app row upsert failed for %s", slug, exc_info=True)
        return None


async def announce_ready(
    *, user_id: str, job_id: Optional[str], app_id: Optional[str], title: str,
    slug: Optional[str] = None,
) -> None:
    """The artifact card. Same frame the Expo pipeline emits on completion.

    ``slug`` is the app's IDENTITY — the handle the chat card stores, the
    runner opens and ``/api/artifacts/{slug}`` serves. Without it a client can
    only get there by fetching the ``apps`` row to translate ``app_id`` back
    into a slug, which is a round-trip between the reply landing and the card
    appearing. ``kind`` already tells a client which pipeline this is; the slug
    is what lets it act.
    """
    await _broadcast(user_id, {
        "type": "app_ready",
        "job_id": job_id,
        "app_id": app_id,
        "id": app_id,
        "name": title,
        "slug": slug,
        "kind": SOURCE_HTML_ARTIFACT,
    })
