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

import hashlib
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
                "done": "Checked the app",      "failed": "The app needs a fix",
                "skipped": "Check the app"},
    # Round 20, item 3. Its own phase, because it answers its own question:
    # `verify` is "does it run", `look` is "does it look right". An app can
    # pass the first and fail the second — white text on a white card throws
    # nothing — and folding the two into one row would leave the user
    # watching a step called "Checking the app" for the length of two
    # different checks, one of which involves a model.
    "look":    {"pending": "Look at the app",   "running": "Looking at the app",
                "done": "Checked the app looks right",
                "failed": "The app doesn't look right yet",
                # Round 23: a look that never reported back used to be DROPPED
                # at finish. The brief's rule is the opposite: a planned phase
                # may be shown as skipped, never vanish — so the skip gets its
                # own honest words instead of a deleted row.
                "skipped": "Couldn't look at the app here"},
    # Round 21, item 1. The mark used to be drawn inside the `present` step,
    # so the one phase that involves two model calls had no row of its own and
    # a failed drawing was invisible. It is a phase: it can be slow, it can
    # fail, and the user should see which of the two happened.
    "logo":    {"pending": "Draw the app's icon", "running": "Drawing the app's icon",
                "done": "Drew the app's icon",
                "failed": "Kept a plain icon for now",
                "skipped": "Kept a plain icon for now"},
    "present": {"pending": "Publish the app",   "running": "Publishing the app",
                "done": "Published the app",    "failed": "Couldn't publish the app",
                "skipped": "Publish the app"},
}

STEP_TYPES: List[str] = ["create", "review", "edit", "verify", "look", "logo",
                         "present"]

#: The phases a build actually WALKS, in order — what the up-front plan shows.
#:
#: Round 23. The plan used to be all seven types, and two of them (`review`,
#: `edit`) are phases a clean first build never enters — so every pristine
#: build opened with two grey rows that were GUARANTEED to vanish at the end
#: (the recorded "7 steps became 4"). The plan is now the walk every build
#: makes; `review`/`edit` rows are appended by `emit_step` at the moment they
#: actually happen, which is also what makes a later edit round legible as
#: appended work rather than as rows that were always mysteriously there.
PLANNED_TYPES: List[str] = ["create", "verify", "look", "logo", "present"]

SOURCE_HTML_ARTIFACT = "html_artifact"


def step_counts(steps: List[Any]) -> "tuple[int, int]":
    """``(done, total)`` — the ONE step arithmetic, for every surface.

    Skipped rows stay VISIBLE (dashed) but are excluded from BOTH numbers.
    Counting them anywhere is how one recorded build showed "Done in 5
    steps", "5/5", six rows and "…/6" at the same time: the ws frame and
    both REST hydrators used ``len(steps)`` while the clients excluded
    skipped. Every producer — the live ``job_update`` frame, the closing
    frame, message_cards hydration, sessions hydration — must call this.
    """
    done = total = 0
    for s in steps:
        if not isinstance(s, dict):
            continue
        status = s.get("status")
        if status == "skipped":
            continue
        total += 1
        # A row EVER done counts as done while it re-runs (`was_done`, see
        # emit_step): retry-in-place takes a finished row back through
        # `running`, and counting only the current status is how a frame —
        # or a job stranded mid-retry by an abnormal close, which never
        # reaches finish_job's restore — reported 2/5 and then 1/5. The
        # done-count never decreases for work that completed.
        #
        # Round 25: `failed` joins `running` here when the row is
        # RECOVERABLE — the publish gate refusing a check it already passed
        # is the pipeline's designed loop, not work being un-done. Without
        # this the measured build dipped 2/5 → 1/5 (40% → 20%) the instant
        # the gate found something to fix, which is the single largest
        # backwards jump on the recorded card.
        if status == "done" or (
            status in ("running", "failed") and s.get("was_done")
        ):
            done += 1
    return done, total


#: Where the monotonic high-water mark is parked between emits.
#:
#: ``BuildJob.config_json`` is a free-form JSON column the pipeline already
#: writes (``_adopt_turn_job`` stamps ``pipeline``/``adopted_by`` into it), so
#: this needs no migration and survives a process restart — which matters,
#: because the ONLY monotonicity guarantee before round 25 was a per-mount
#: `useRef` in ONE of the three clients that guarded the bar and not the
#: "N/M steps" text beside it.
_HIGH_WATER_KEY = "progress_high_water"


def percent_for(done: int, total: int) -> int:
    """``steps_done / steps_total`` as a whole percent. The ONE formula.

    Separate from the clamp below so the arithmetic can be asserted on its
    own: this is the value the brief calls "mathematically correct", and
    :func:`progress` is what additionally makes it non-decreasing.
    """
    if total <= 0:
        return 0
    return max(0, min(100, round(100.0 * done / total)))


def progress(steps: List[Any], config: Optional[Dict[str, Any]] = None,
             *, job_id: str = "") -> "tuple[int, int, int, Dict[str, Any]]":
    """``(done, total, percent, config)`` — counts plus a percent that cannot
    go backwards.

    Round 25, item 8. Percent was never computed server-side at all: the wire
    carried the pair and each client divided, so every client re-derived — and
    re-broke — the same number. Three mechanisms made the honest quotient fall:

    * **Plan growth.** ``review``/``edit`` are not in the up-front plan (round
      23 removed them deliberately, because a pristine build otherwise opened
      with two grey rows guaranteed to vanish). They are appended at the moment
      they happen, so the DENOMINATOR rises mid-build: 1/5 → 1/6, 20% → 17%.
    * **A refused re-check**, now fixed at the source above by keeping
      ``was_done`` on a recoverable failure.
    * **The ordering net** force-skipping an abandoned row.

    The denominator rising is by design and is not going to be undone here —
    so the guarantee is provided where it belongs, on the number itself, and
    it is PERSISTED (``config_json``) rather than latched in one client.

    A clamp is a real disagreement between two true numbers, so it is logged
    at WARNING with both. It never raises: this module is fail-open by
    contract (see the module docstring) and a progress bar is never worth a
    build.
    """
    done, total = step_counts(steps)
    raw = percent_for(done, total)
    cfg = dict(config or {})
    try:
        previous = int(cfg.get(_HIGH_WATER_KEY) or 0)
    except (TypeError, ValueError):
        previous = 0
    previous = max(0, min(100, previous))
    if raw < previous:
        logger.warning(
            "[app_html] progress for %s went backwards: %d%% (%d/%d) after "
            "%d%% — holding the high-water mark",
            job_id or "?", raw, done, total, previous,
        )
        percent = previous
    else:
        percent = raw
    cfg[_HIGH_WATER_KEY] = percent
    return done, total, percent, cfg


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
    "look":    "The app runs, but something on screen isn't right yet, so I "
               "haven't shown it to you. Ask me to fix it.",
    # Not a failure the user has to do anything about: the app is published
    # either way and a later run upgrades the mark.
    "logo":    "The app is ready — it just has a plain icon for now.",
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
    if status == "skipped":
        # A skipped phase reads as the thing that DIDN'T happen, never as
        # the running form ("Writing the app" on a row nothing wrote).
        return words.get("skipped") or words.get("pending") or words["running"]
    return words.get(status) or words["running"]


#: What the card's header says while the pipeline is repairing its own work.
#:
#: Round 25, item 2. The brief's words, deliberately: finding-and-fixing is
#: the designed loop, so the state it puts the card in needs a name that reads
#: as progress rather than as an alarm.
FIXING_LABEL = "Fixing it"


def _headline(steps: List[Any], label: str) -> str:
    """The header line for a frame: the phase, or ``Fixing it``.

    A recoverable failure anywhere in the list means the gate found something
    and the model is working through it. Saying so is more useful — and more
    honest — than showing the label of whichever row happened to be touched
    last, which during a repair is "Reading the app" or "Updating the app".
    """
    for s in steps:
        if isinstance(s, dict) and s.get("status") == "failed" and s.get("recoverable"):
            return FIXING_LABEL
    return label


def public_steps(steps: List[Any]) -> List[Dict[str, Any]]:
    """The step rows as a client should receive them.

    Deliberately the SAME shape as the rows persisted in ``steps_json`` and
    shipped by ``message_cards``, not a second projection. Round 25, item 6:
    the live frame and the history hydrator disagreeing about the shape of a
    build card is precisely why a returning user saw a card that could not be
    reconciled with the one they left. One shape, two transports.
    """
    return [s for s in steps if isinstance(s, dict)]


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
        for t in PLANNED_TYPES
    ]


def turn_deep_link() -> "tuple[Optional[str], Optional[str]]":
    """``(chat_id, message_id)`` for this turn — the phone card's address.

    Read from the tool executor's own context variables rather than from the
    job row, and that is the whole difficulty of round 25 item 7. A build job
    is opened by this pipeline with a ``TaskSpec`` of its own, so historically
    its ``conversation_id`` was NULL — and ``job_mission_id`` falls back to the
    RAW JOB ID when there is no chat id. A push addressed that way does not
    land on the conversation's ``chatjob:<sid>`` card; it opens a second,
    orphaned one. ``ensure_job`` now stamps the conversation onto the row as
    well, but this stays the live source: an ADOPTED job already carries one,
    and the context variable is correct for both.
    """
    try:
        from app.agent.tool_executor import (
            _ASST_MESSAGE_ID_CTX, _SESSION_ID_CTX,
        )
        return _SESSION_ID_CTX.get(), _ASST_MESSAGE_ID_CTX.get()
    except Exception:  # noqa: BLE001 - no deep link is better than no card
        return None, None


async def _push_live_activity(
    *, job_id: str, title: str, kind: str, headline: str,
    done: int, total: int, percent: int, body: str = "",
) -> None:
    """Put this build on the lock screen and the Dynamic Island.

    Round 25, item 7. The app builder was the one long-running job with no
    phone surface of its own: ``grep -rn notify`` over this package returned
    nothing, so a build that took two minutes showed the user nothing once
    they left the app. Everything needed already existed — ``_notify_job_event``
    carries a title, a step name, ``steps_done``/``steps_total``, a 0-100
    ``progress`` and the deep-link ids — it had simply never been called from
    here.

    Sent on real phase transitions only, never on the byte-counter ticks: a
    content-state update per delta would be a push per kilobyte.

    Best-effort by contract, like everything else in this module. A phone card
    is never worth a build.
    """
    try:
        from app.agent.subagent_orchestrator import _notify_job_event
        chat_id, message_id = turn_deep_link()
        await _notify_job_event(
            job_id=job_id,
            label=title or "your app",
            kind=kind,
            title=title or "Building your app",
            body=body or None,
            progress=percent,
            # `step_name` is what the island shows under the title. During a
            # repair this is "Fixing it" — the same words the chat card uses,
            # so the two surfaces cannot disagree about what is happening.
            step_name=headline,
            steps_done=done,
            steps_total=total,
            job_type="auto_builder",
            chat_id=chat_id,
            message_id=message_id,
            # With a chat id the card is keyed `chatjob:<chat_id>` and routed
            # to the conversation, so tapping it lands on the build.
            route="chat" if chat_id else "mission-control",
            # A build reuses ONE card per conversation: a second app in the
            # same chat must refresh the card rather than be dropped as a
            # duplicate `mission_started`.
            refresh_if_started=True,
            dedup_suffix=f"app:{kind}:{done}/{total}:{percent}",
        )
    except Exception:  # noqa: BLE001 - never worth a build
        logger.debug("[app_html] live-activity push failed", exc_info=True)


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


def _register_with_turn_finalizers(job_id: str) -> str:
    """Put a job this pipeline drives into the turn's created-jobs registry.

    The system prompt forbids the model calling `create_job` for a build, so
    `ensure_job` mints (or revives) the job itself — and an unregistered
    build job is invisible to both turn-end finalizers (agent_runner's
    happy-path close and `_sweep_unclosed_created_jobs`); only the 30-minute
    reaper would close it, which is how an abnormally-ended build sat
    "running" for half an hour. Both finalizers guard on status, so
    registering a job `finish_job` already closed costs nothing.

    Appends through the same live list the `create_job` tool path appends to
    (`_created_job_registry`) — the registry contract is mutate-in-place,
    never re-`set()`. An adopted job is already in the list, hence the
    membership check.
    """
    try:
        from app.agent.tool_executor import _created_job_registry
        reg = _created_job_registry()
        if job_id not in reg:
            reg.append(job_id)
    except Exception:  # noqa: BLE001 - bookkeeping is never worth the build
        logger.debug("[app_html] created-jobs registry append failed",
                     exc_info=True)
    return job_id


async def ensure_job(user_id: str, slug: str, title: str) -> Optional[str]:
    """Find-or-create the ONE BuildJob backing this app's card.

    Order matters: an app that already has a job keeps it (so an edit three
    turns later updates the same card), otherwise this turn's model-created
    job is adopted, and only if there is neither do we open a new one.
    Every id returned is registered with the turn finalizers — see
    :func:`_register_with_turn_finalizers`.
    """
    app_id = app_id_for(user_id, slug)
    existing = await _existing_job_for_app(app_id)
    if existing:
        return _register_with_turn_finalizers(existing)
    adopted = await _adopt_turn_job(user_id, app_id, title)
    if adopted:
        return _register_with_turn_finalizers(adopted)
    try:
        from app.agent.job_runner import JobRunner, TaskSpec
        spec = TaskSpec(
            user_id=user_id,
            channel="app_builder",
            source_kind="app_builder_skill",
            source_id=app_id,
            # Round 25, item 7. Without this the row's `conversation_id` was
            # NULL, and every consumer that derives a Live Activity address
            # from the JOB — `job_mission_id`, and the interrupted-job sweep's
            # `mission_failed` push — fell back to the raw job id and
            # addressed a card nobody was looking at. An adopted job has
            # carried this all along (the `create_job` tool stamps it); a
            # pipeline-opened one now does too, so the two behave alike.
            conversation_id=turn_deep_link()[0],
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
        return _register_with_turn_finalizers(job.id)
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
    recoverable: bool = False,
) -> Optional[Dict[str, Any]]:
    """Mark one phase and push it to the chat card + activity feed.

    ``status`` ∈ {running, done, failed, skipped}. A skip is emitted AT THE
    MOMENT a phase is known not to run (the skill owns that call) — never
    deferred to build end, which is how a spinner outlived the rows beneath
    it. The label is derived from ``step_type`` and ``status`` (see
    :func:`phase_label`) and is NOT a parameter — see the note on
    :data:`_PHASE_WORDS`.

    ``recoverable`` marks a failure the pipeline EXPECTS and is about to act
    on — the publish gate finding something to fix and handing the model a
    repair list. Round 25, item 2.

    Before this flag there was exactly one way to say "a check did not pass",
    and it flipped ``BuildJob.status`` to ``failed`` (below) — the field the
    web card keys its red "Couldn't build X · Try again" pill on. And it was
    not for one frame: ``any_failed`` re-scans the whole list on EVERY
    subsequent emit, so the refused ``verify`` row held the whole job at
    ``failed`` through the model's read, through its edit, and through the
    write-back — the entire repair window — until the next ``present_app``
    took the row back to ``running``. Finding-and-fixing is the designed loop,
    so on the wire it now reads as work in progress: the row still says what
    happened, the job stays ``running``, and red is reserved for a build that
    truly cannot finish.
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
            # Is this the first thing that has happened on this build? Read
            # BEFORE the row below is mutated. Decides `mission_started` vs
            # `progress` on the phone card (round 25, item 7) — a build that
            # opened with `progress` would update a card that was never
            # started, and one that sent `mission_started` twice would be
            # dropped as a duplicate.
            first_transition = not any(
                isinstance(s, dict) and s.get("status") != "pending"
                for s in steps
            )

            # ── Retry-in-place (round N correction) ───────────────────
            # Round 23 APPENDED a fresh row when a completed phase
            # re-entered running ("occurrence-append") — and EVERY build
            # re-enters verify, because create runs the checks and the
            # publish gate runs them again. So every card grew a duplicate
            # "Checking the app" row mid-run, the plan went 5 → 6, and the
            # header's percentage regressed 40% → 33% on the recording that
            # ended the design. A re-run is now a RETRY of the SAME row:
            # done → running (a visible re-check on a stable step id) and
            # back. The denominator is stable by construction; the
            # momentary done-count dip during a retry is clamped by the
            # clients' high-water display latch — which is where round 23's
            # "2/7 fell back to 1/7" objection is answered now, at the
            # display, not by mutating the list.
            last_idx = -1
            for i, s in enumerate(steps):
                if isinstance(s, dict) and s.get("type") == step_type:
                    last_idx = i
            step_dict: Optional[Dict[str, Any]] = None
            if last_idx >= 0:
                step_dict = steps[last_idx]
            else:
                # `review`/`edit` are not in the up-front plan — they appear
                # at the moment they actually happen, inserted in WORK order:
                # after everything that has started, before the untouched
                # plan, so the card reads finished-work → this → still-to-do.
                step_dict = {"id": str(uuid.uuid4()), "type": step_type}
                insert_at = 0
                for i, existing in enumerate(steps):
                    if isinstance(existing, dict) and existing.get("status") != "pending":
                        insert_at = i + 1
                steps.insert(insert_at, step_dict)

            # ── Top-to-bottom is structural, not situational ──────────
            # Nothing may sit spinning while rows beneath it finish — the
            # recorded card had "Looking at the app" spinning while icon and
            # publish completed under it, because the look's skip was
            # deferred to build end. The skill now resolves its own skips at
            # the moment they happen; this net is for the path nobody
            # wrote: if a LATER row is being touched while an EARLIER row is
            # still `running`, that earlier row was abandoned — resolve it
            # skipped right now and say so loudly, because reaching this
            # line at all is a pipeline bug.
            cur_idx = steps.index(step_dict)
            for earlier in steps[:cur_idx]:
                if isinstance(earlier, dict) and earlier.get("status") == "running":
                    earlier["status"] = "skipped"
                    earlier["label"] = phase_label(earlier.get("type") or "", "skipped")
                    earlier.setdefault("detail", "the build moved on without it")
                    # Round 25: clear the markers too. `step_counts` excludes
                    # a skipped row from BOTH numbers before it ever reaches
                    # the `was_done` test, so a force-skipped row that still
                    # carried the marker was a phase counted in neither — and
                    # `finish_job`'s restore only inspects pending/running, so
                    # nothing downstream could repair it.
                    earlier.pop("was_done", None)
                    earlier.pop("recoverable", None)
                    earlier["rev"] = int(earlier.get("rev") or 0) + 1
                    logger.warning(
                        "[app_html] step %r abandoned running while %r advanced "
                        "— resolved skipped (pipeline ordering bug)",
                        earlier.get("type"), step_type,
                    )

            s = step_dict
            s["label"] = label
            s["status"] = status
            # Ever-done marker. Retry-in-place takes a done row back through
            # `running`, so at an abnormal close (turn death, the reconciler's
            # settle) the row's CURRENT status no longer says the phase ever
            # completed — which is how a card that read 2/5 died reading 1/5.
            # `rev` cannot answer it (a failed→running retry bumps rev too),
            # so the fact is recorded outright. A skill-REPORTED failed/skip
            # is a verdict and supersedes the history — without the clear, a
            # rebuild's genuinely-skipped check would still count as done.
            if status == "done":
                s["was_done"] = True
            elif status == "failed" and recoverable:
                # Deliberately KEEPS `was_done`. A gate refusal is a re-check
                # of a phase that already passed once, so the work it
                # represents was really done; popping the marker here is what
                # dropped the measured card from 2/5 to 1/5 at the exact
                # moment the pipeline started helping. The verdict is carried
                # by `recoverable` instead, which every surface can read.
                pass
            elif status in ("failed", "skipped"):
                s.pop("was_done", None)
            # The flag is a property of THIS result, so it is rewritten on
            # every emit — a row that fails recoverably, is repaired, and then
            # fails terminally must not still be wearing the friendly badge.
            if status == "failed" and recoverable:
                s["recoverable"] = True
            else:
                s.pop("recoverable", None)
            # Per-row write counter: what lets the clients tell a genuine
            # retry (done → running, HIGHER rev) from a stale poll whose rows
            # predate the frame that started the retry (older rev). Without
            # it the clients' rank-merge — correct against races — would
            # suppress the visible re-check retry-in-place exists to show.
            s["rev"] = int(s.get("rev") or 0) + 1
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
            # A RECOVERABLE failure is not a failing job. It is the gate
            # doing its job and the model about to act on it, so it must not
            # reach the field the clients paint red — see the `recoverable`
            # note in this function's docstring. It still resolves terminal at
            # `finish_job`: once the build stops, a check that never passed is
            # a check that failed, and the card says so.
            any_failed = any(
                s.get("status") == "failed" and not s.get("recoverable")
                for s in steps if isinstance(s, dict)
            )
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
                elif job.status == "completed" and step_type == "create" and status == "running":
                    # A REBUILD of this slug (one card per app: the job row
                    # is reused). Only the CREATE step revives a completed
                    # job — a later `view_app_file`/`bash_app` must not (the
                    # eternal-spinner hazard the leave-completed-alone rule
                    # above exists for). Without this, the whole rebuild ran
                    # under status=completed: the client refused to claim
                    # the job, the live turn kept its generic card, and the
                    # OLD chip mutated in place (observed b2, 2026-08-23).
                    job.status = "running"
                    if hasattr(job, "completed_at"):
                        job.completed_at = None
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

            completed, total, percent, job.config_json = progress(
                steps, job.config_json, job_id=job_id,
            )
            await db.commit()
            job_status = job.status
            job_title = (job.title or "").replace("Build: ", "") or label
            headline = _headline(steps, label)

        frame = {
            "type": "job_update",
            "job_id": job_id,
            "name": job_title,
            "status": job_status,
            "step": headline,
            "total_steps": total,
            "completed_steps": completed,
            # Round 25, item 8. The server now says what the percentage IS,
            # rather than leaving three clients to divide and each get it
            # wrong in its own way. Monotonic and persisted — see `progress`.
            "percent": percent,
            # The steps themselves. `job_update` has never carried them, so a
            # client that had not seen every earlier frame — one that
            # reconnected, or hydrated a card from history — had a header, a
            # bar and no rows to put under them.
            "steps": public_steps(steps),
        }
        await _broadcast(user_id, frame)
        # Round 25, item 7: the same state, on the phone. `mission_started`
        # only for the very first transition of a build — after that the card
        # is up and every push is a content-state update.
        await _push_live_activity(
            job_id=job_id, title=job_title,
            kind="mission_started" if first_transition else "progress",
            headline=headline, done=completed, total=total, percent=percent,
        )
        # Returned so a caller watching a LONG tool call can keep the card
        # moving without going back to the database for every tick — see
        # `retouch`. Every existing caller ignores this.
        return frame
    except Exception:
        logger.debug("[app_html] emit_step failed (non-fatal)", exc_info=True)
    return None


def settle_steps(steps: List[Any], *, detail: str) -> List[Any]:
    """Resolve every unfinished row of a build that has STOPPED. In place.

    Round 23's rule, and now the ONE implementation of it. A phase left
    ``running`` never reported back — calling it done would invent a result
    and calling it failed would invent a diagnosis — so it and any untouched
    ``pending`` row become ``skipped``, with the honest per-phase words from
    :data:`_PHASE_WORDS` ("Couldn't look at the app here"). A planned phase may
    be shown as skipped; it may never vanish, which is how the recorded card's
    seven rows became four at the moment it completed.

    Two exceptions, both about not un-counting finished work:

    * a row mid-RETRY (``was_done``) keeps its last reported result — skipping
      it is how a card that read 2/5 died reading 1/5;
    * a ``recoverable`` failure resolves terminal, because "the build is about
      to fix this" stops being true the moment the build stops (round 25).

    Extracted because the THIRD caller was the one that got it wrong: the
    interrupted-turn sweep flipped a job's status in a bulk UPDATE and never
    touched ``steps_json`` at all, so a cancelled build kept its ``running``
    and ``pending`` rows and rendered as "In progress · 4/7" under a status
    that was already terminal.
    """
    for s in steps:
        if not isinstance(s, dict):
            continue
        if s.get("status") == "failed" and s.pop("recoverable", None):
            s.pop("was_done", None)
            s["rev"] = int(s.get("rev") or 0) + 1
            continue
        if s.get("status") in ("pending", "running"):
            if s.get("was_done"):
                s["status"] = "done"
                s["label"] = phase_label(s.get("type") or "", "done")
            else:
                s["status"] = "skipped"
                s["label"] = phase_label(s.get("type") or "", "skipped")
                s.setdefault("detail", detail)
            s["rev"] = int(s.get("rev") or 0) + 1
    return steps


async def retouch(
    user_id: str, frame: Optional[Dict[str, Any]], *, step_type: str,
    detail: str,
) -> Optional[Dict[str, Any]]:
    """Re-send a frame with a fresh ``detail`` on one row. NO database write.

    Round 25, item 1. While ``create_app_file``'s argument streams, the phase
    it belongs to is genuinely running and genuinely making progress, but
    nothing has happened that is worth a row transition — the status is
    unchanged and will be for a minute or more. Writing the job row on every
    tick to say so would be a heavy write loop for a detail line; not saying
    anything is the silence this round exists to remove.

    So the card is kept moving from the frame the last real transition already
    produced. If that frame never arrived (the open failed), there is nothing
    to retouch and this is a no-op — never a second source of truth.
    """
    if not frame:
        return None
    detail = (detail or "").strip()[:200]
    out = dict(frame)
    rows: List[Dict[str, Any]] = []
    for s in frame.get("steps") or []:
        if isinstance(s, dict) and s.get("type") == step_type:
            s = dict(s)
            s["detail"] = detail
        rows.append(s)
    out["steps"] = rows
    await _broadcast(user_id, out)
    return out


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
            # Round 23's rule, now shared with every other close path — see
            # `settle_steps` for what it does and why it is not inline here
            # any more.
            settle_steps(steps, detail="the build finished without it")
            # A build that closes green around a skipped VERIFICATION is the
            # false-success pattern — with the browser shipped in the image
            # this must never fire, so when it does it is an infrastructure
            # defect and it logs like one. The card stays honest either way:
            # the row is dashed with its reason, and the counts exclude it.
            skipped_checks = [
                s.get("type") for s in steps
                if isinstance(s, dict) and s.get("status") == "skipped"
                and s.get("type") in ("verify", "look")
            ]
            if skipped_checks:
                logger.error(
                    "[app_html] build %s completed with verification skipped: "
                    "%s — infrastructure defect (renderer unavailable?)",
                    job_id, ", ".join(skipped_checks),
                )
            # `settle_steps` above has already resolved any `recoverable`
            # failure to a plain one — the loop has stopped, so nothing is
            # going to fix it — which is what makes this scan terminal.
            if any(isinstance(s, dict) and s.get("status") == "failed"
                   for s in steps):
                final = "failed"
            job.steps_json = json.dumps(steps)
            job.status = final
            job.completed_at = datetime.utcnow()
            title = (job.title or "").replace("Build: ", "")
            done, total, percent, job.config_json = progress(
                steps, job.config_json, job_id=job_id,
            )
            if final == "completed":
                # Every row that was going to resolve has resolved above, so
                # a completed build's own arithmetic already reads N/N. This
                # is belt-and-braces for the one case that would otherwise
                # show a finished card at 80%: a clamp holding an older,
                # higher mark cannot exceed 100, and a completed build is by
                # definition all the way along.
                percent = 100
                cfg = dict(job.config_json or {})
                cfg[_HIGH_WATER_KEY] = 100
                job.config_json = cfg
            await db.commit()

        await _broadcast(user_id, {
            "type": "job_update",
            "job_id": job_id,
            "name": title,
            "status": final,
            "step": phase_label("present", "done") if final == "completed" else "",
            "total_steps": total,
            "completed_steps": done,
            "percent": percent,
            "steps": public_steps(steps),
        })
        # Round 25, item 7: success (or failure) on the phone, and the card
        # ends rather than being left up. `end_after_s` lets the completed
        # state be READ before the card goes — a build that vanishes the
        # instant it finishes never showed the user that it worked.
        await _push_live_activity(
            job_id=job_id, title=title,
            kind="mission_completed" if final == "completed" else "mission_failed",
            headline=(phase_label("present", "done") if final == "completed"
                      else "Couldn't finish this one"),
            done=done, total=total, percent=percent,
            body=(f"{title} is ready to open." if final == "completed" else ""),
        )
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


async def register_in_library(*, user_id: str, slug: str) -> bool:
    """Make the published app a Files row NOW. Never raises.

    Round 21, item 2. `upsert_app_row` above puts the app in the *Apps* list;
    this puts it in *Files*, which is a different manifest with a different
    refresh rule — a throttled scan of the disk that only runs when something
    is listed. Measured before this call existed: list Files immediately
    after a publish and the Apps folder was empty; only `?refresh=true`
    found the app.

    Best-effort by contract. The app is published; a manifest row that could
    not be written is a listing that lags by one sync, which is exactly the
    behaviour this replaces — never a failed publish.
    """
    if not user_id:
        return False
    try:
        from app.db.database import async_session_maker
        from app.services import library_service as lib

        async with async_session_maker() as db:
            row = await lib.register_app_file(db, user_id, slug)
        return row is not None
    except Exception:  # noqa: BLE001 - a listing is never worth a publish
        logger.warning("[app_html] could not register %s in Files", slug,
                       exc_info=True)
        return False


#: Beyond this an inlined mark is not worth the frame it rides on. Real
#: designed marks measure 1–3 KB; `logo.MAX_ICON_BYTES` allows 24 KB, and a
#: 24 KB SVG on a WS frame is a worse deal than one conditional GET.
MAX_INLINE_ICON_BYTES = 8 * 1024


def artifact_payload(slug: str, *, include_icon: bool = False) -> Dict[str, Any]:
    """The app's HANDLE — everything a card needs and no body.

    ``revision`` is the field this whole round turns on. It is what a client
    compares to decide that the app in front of it is stale: the mobile
    registry drops its cached body when the number moves
    (``appArtifacts.upsertArtifact``) and the runner keys its WebView on it, so
    a revision that never arrives is a runner that never reloads.

    Deliberately NO ``html``. The body is 10–60 KB and would ride every frame
    AND every persisted message; the clients fetch it once, at open time, which
    is the contract they were written to. What must travel is the number that
    tells them the fetch is now worth making.
    """
    from app.agent.skills.builtins.app_html import logo, store

    rec = store.read_manifest().get(slug)
    if rec is None:
        return {"slug": slug}
    # The mark, in two halves, because this payload has two readers with
    # opposite needs.
    #
    # `icon_etag` always. It is the sha of the SVG's bytes — the same value
    # the icon route sends as its ETag — so a client can cache the drawing
    # under (slug, etag) forever and refetch on the one event that means
    # anything: a redraw. Round 20 shipped only `has_icon`, which says an
    # icon exists but never that it CHANGED, so every surface that wanted to
    # be correct had to re-request the SVG on every render. That is the
    # per-app-card request this round was asked to remove.
    #
    # `icon_svg` only when asked for, and only for a small one. The live
    # `app_artifact` frame wants it — the card is about to be drawn and a
    # second round trip is a visible pop-in — while the copy persisted into
    # a message's metadata must not carry it, or a thread with thirty app
    # cards would carry thirty SVGs in its history for ever.
    svg = ""
    try:
        svg = logo.read_icon(slug) or ""
    except Exception:  # noqa: BLE001 - fail-open, like everything here
        svg = ""
    # The preview follows the icon's contract exactly: an etag on the handle,
    # bytes behind their own route, cached client-side under (slug, etag) and
    # refetched only when a publish moved it. Never inline — a PNG is 40–120
    # KB and this payload rides frames AND persisted messages.
    preview = ""
    try:
        preview = store.preview_etag(slug)
    except Exception:  # noqa: BLE001 - fail-open, like everything here
        preview = ""
    out: Dict[str, Any] = {
        "slug": slug,
        "title": rec.title or slug,
        "revision": rec.revision,
        "updated_at": rec.updated_at or None,
        "size_bytes": rec.size_bytes,
        "has_icon": bool(svg),
        "icon_etag": (
            hashlib.sha256(svg.encode("utf-8")).hexdigest()[:32] if svg else ""
        ),
        "has_preview": bool(preview),
        "preview_etag": preview,
    }
    if include_icon and svg and len(svg.encode("utf-8")) <= MAX_INLINE_ICON_BYTES:
        out["icon_svg"] = svg
    return out


async def announce_ready(
    *, user_id: str, job_id: Optional[str], app_id: Optional[str], title: str,
    slug: Optional[str] = None,
) -> None:
    """The artifact card, and the fact that it moved.

    TWO frames, because they answer two different questions and the second one
    had no answer at all until now.

    ``app_ready`` closes the build card. It carries ``slug`` because the app's
    IDENTITY is the handle the chat card stores, the runner opens and
    ``/api/artifacts/{slug}`` serves — without it a client can only get there by
    translating ``app_id`` back into a slug.

    ``app_artifact`` says WHICH VERSION is now live. Both clients were built
    against it — ``AppArtifactCard`` redraws from the registry on it,
    ``AppRunner`` bumps the WebView key when the revision moves, and
    ``AppArtifactFrame`` refetches its token — and nothing has ever sent it. So
    an edit could report "revision 2, republished", be genuinely on disk, be
    genuinely what ``/api/artifacts/{slug}`` served on the next request, and
    still leave the person looking at revision 1: no surface had been told a
    revision 2 existed, and the runner had no reason to ask again.

    Order matters. ``app_artifact`` goes FIRST so the registry already holds the
    new revision when ``app_ready`` makes a client draw the card — the reverse
    order draws it from the previous revision and corrects it a frame later.
    """
    if slug:
        try:
            # `include_icon`: this is the frame that makes a client DRAW the
            # card, so the mark travels with it. Without it the card paints,
            # then fires a request for `/artifacts/{slug}/icon`, then repaints
            # — a visible pop-in on the one frame where the drawing is already
            # in this process's hands.
            payload = artifact_payload(slug, include_icon=True)
        except Exception:  # noqa: BLE001 - fail-open, like everything here
            # An unreadable manifest costs the revision, never the card: the
            # slug alone is exactly what this used to carry.
            logger.debug("[app_html] artifact payload failed", exc_info=True)
            payload = {"slug": slug}
        await _broadcast(user_id, {
            "type": "app_artifact",
            "job_id": job_id,
            "app_id": app_id,
            "artifact": payload,
        })
    await _broadcast(user_id, {
        "type": "app_ready",
        "job_id": job_id,
        "app_id": app_id,
        "id": app_id,
        "name": title,
        "slug": slug,
        "kind": SOURCE_HTML_ARTIFACT,
    })
