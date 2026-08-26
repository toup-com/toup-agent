"""The automations executor — fire → evaluate → prepare → write → record.

A fire is a headless run in the same context autopilot ticks run in:
no user present, `channel="automation"`, one BuildJob per run
(`job_type="automation_run"`, source_kind="automation") with steps_json
carrying per-step durations. Connector calls go EXCLUSIVELY through the
platform RPC (`registry.dispatch_via_platform`) — the dispatcher's
grant gate is the write authority, this module never talks to a
provider directly and never sees a token.

Rails enforced here:
  - events: insert-or-skip on UNIQUE (automation_id, dedupe_key)
  - runs:   idempotency_key = "evt:<event_id>" (or the fire key for
            schedule mode) → the build_jobs partial UNIQUE collapses
            double-fires
  - writes: staged to automation_outbox with an idempotency key and an
            execute_after undo window; the flush loop (outbox.py) is
            the only thing that sends
  - cap:    AUTOMATION_RUN_CAP_S via asyncio.wait_for at every entry
  - mail:   gmail__send_message is refused outright — drafts only
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Optional

from sqlalchemy import select, update as sa_update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError

from app.db.models import (
    Automation, AutomationEvent, AutomationOutbox, BuildJob,
    AUTOMATION_OUTBOX_UNDO_WINDOW_S, AUTOMATION_RUN_CAP_S,
)
from app.agent.job_runner import JobRunner, TaskSpec
from app.agent import job_steps
from .spec import ValidatedSpec, render_params, resolve_path
from .session import on_run_created
from . import registry as reg

logger = logging.getLogger(__name__)

# The one hard product rail that is not config: automations never send
# mail. The dispatcher's grant machinery would also refuse (no grant is
# ever minted for send — the setup tool refuses first), but the
# executor is the first line and must not depend on the others.
_FORBIDDEN_TOOLS = frozenset({"gmail__send_message", "outlook__send_message"})

_STEPS = ("evaluate", "prepare", "write", "record")


def _new_steps(vspec: Optional[ValidatedSpec] = None) -> str:
    """Step ledger with HUMANIZED labels at mint (R29): steps_json is
    the shared substrate (runs API, job cards, web), so the verb
    dictionary applies here, not at render time. The write step wears
    the spec's action verb + connector brand; engine phases brand as
    the orb (brand None)."""
    from app.services.automation_verbs import step_verb

    now = datetime.utcnow()
    steps = []
    for s in _STEPS:
        if s == "write" and vspec is not None and vspec.action_tool:
            v = step_verb(vspec.action_tool, vspec.action_connector_id)
        else:
            v = step_verb(None, None, phase=s)
        steps.append({
            "id": s, "type": "generic", "label": v["label"],
            "brand": v["brand"],
            "status": "pending", "started_at": None, "completed_at": None,
        })
    return job_steps.dump_steps(job_steps.open_first_step(steps, now))


async def _advance(db: AsyncSession, job_id: str, done_step: str) -> None:
    job = await db.get(BuildJob, job_id)
    if job is None:
        return
    steps = job_steps.parse_steps(job.steps_json)
    job.steps_json = job_steps.dump_steps(
        job_steps.advance_steps(
            steps, _STEPS.index(done_step), datetime.utcnow(),
            fallback_start=job.created_at,
        )
    )
    await db.commit()


async def _finalize_job(
    db: AsyncSession, job_id: str, *, status: str,
    outcome: Optional[str] = None, error_class: Optional[str] = None,
    user_message: Optional[str] = None,
) -> bool:
    """Guarded terminal transition — only a running/queued row moves.

    Returns whether the UPDATE actually applied. R31-31: callers that
    write a user-facing NOTE after finalizing have to know. `handle_stop`
    did not, and appended `YOU STOPPED IT HERE` unconditionally — so
    when the stuck-run reaper had already terminalized the row as
    `failed/lost`, the thread said the user stopped it while the ledger
    said it broke. Two surfaces, one run, opposite stories, and the
    ledger row is the one the home card reads.
    """
    now = datetime.utcnow()
    job = await db.get(BuildJob, job_id)
    values: dict[str, Any] = {
        "status": status,
        "completed_at": now,
        "outcome": outcome,
        "error_class": error_class,
        "user_message": user_message,
    }
    if job is not None and status == "completed":
        steps = job_steps.parse_steps(job.steps_json)
        values["steps_json"] = job_steps.dump_steps(
            job_steps.finish_all_steps(steps, now,
                                       fallback_start=job.created_at)
        )
    result = await db.execute(
        sa_update(BuildJob)
        .where(BuildJob.id == job_id)
        .where(BuildJob.status.in_(("queued", "running", "waiting_on_user")))
        .values(**values)
    )
    await db.commit()

    # R31-42: the run stops holding a deploy open HERE — the one gate
    # every terminal passes, whichever executor or sweep got there.
    # `on_terminal` was the obvious place and is the wrong one: it is
    # reached only when the rowcount won AND the run has a v3 thread, so
    # a pre-v3 or thread-less run would hold a drain open forever.
    # `discard` is idempotent, so a double release is free; a MISSED one
    # is the failure mode worth designing against.
    try:
        from app.services import drain_state as _drain
        _drain.run_finished(job_id)
    except Exception:  # noqa: BLE001 — a drain hook never fails a run
        pass

    # R28: every terminal transition — v1, v2, and the outbox aggregate
    # finalizer — funnels through here, and the guarded UPDATE's
    # rowcount makes "exactly once" free: only the call that actually
    # flipped the row notifies. The composer gates on noteworthiness
    # and never raises.
    #
    # R29 ordering contract (CONTRACTS-R29.md §2): the last-outcome
    # stamp lands BEFORE the notify — a push can never reference a
    # sentence that isn't stamped yet. R29-C's additions go after.
    stamp = None
    if result.rowcount == 1 and job is not None and job.source_id:
        stamp = await _stamp_last_outcome(
            db, job, status=status, outcome=outcome,
        )
    # R30: a run with a v3 thread notifies through the §4.10 pipeline
    # (on_terminal below — card + completed-only push + LA end with one
    # body); the legacy per-run push would double-notify it.
    _has_v3_thread = False
    if job is not None:
        try:
            cfg = job.config_json or {}
            if isinstance(cfg, str):
                cfg = json.loads(cfg or "{}")
            _has_v3_thread = bool(cfg.get("thread_id"))
        except (ValueError, TypeError):
            _has_v3_thread = False
    if (result.rowcount == 1 and status == "completed" and job is not None
            and not _has_v3_thread):
        try:
            if job.source_id:
                a = await db.get(Automation, job.source_id)
                if a is not None:
                    from .notify import notify_run_outcome
                    await notify_run_outcome(
                        user_id=job.user_id,
                        automation_id=a.id,
                        automation_name=a.name,
                        job_id=job_id,
                        outcome=outcome,
                        wrote_count=(stamp or {}).get("wrote_count", 0),
                        chat_id=job.conversation_id,
                        message_id=job.summary_message_id,
                    )
        except Exception as e:  # noqa: BLE001 — a push never fails a run
            logger.warning(
                "[automations] finalize notify skipped job=%s: %s",
                job_id[:8], e,
            )
    if result.rowcount == 1 and job is not None and _has_v3_thread:
        try:
            from .run_v3 import on_terminal
            await on_terminal(db, job_id)
        except Exception as e:  # noqa: BLE001 — v3 close never fails a run
            logger.warning(
                "[automations] v3 terminal close skipped job=%s: %s",
                job_id[:8], e,
            )
    return result.rowcount == 1


def _spec_write_shape(spec_raw: dict) -> tuple[Optional[str], Optional[str],
                                               int, dict]:
    """(write_tool, connector_id, wrote_count, step_id→connector) from a
    persisted canonical spec dict — shape-only, no validation."""
    if spec_raw.get("version") == 2:
        steps = [s for s in (spec_raw.get("steps") or [])
                 if isinstance(s, dict)]
        writes = [s for s in steps if s.get("grant_id")]
        by_id = {s.get("id"): s.get("connector_id") for s in steps}
        first = writes[0] if writes else {}
        return (first.get("tool"), first.get("connector_id"),
                len(writes), by_id)
    action = spec_raw.get("action") or {}
    return (action.get("tool"), action.get("connector_id"),
            1 if action.get("tool") else 0, {})


async def _stamp_last_outcome(
    db: AsyncSession, job: BuildJob, *, status: str, outcome: Optional[str],
) -> Optional[dict]:
    """Stamp the automation's last-outcome columns (R29 §2). The stamp
    itself makes the row UNSEEN (last_outcome_at outruns
    outcome_seen_at); `POST /{id}/seen` is the CAS that clears it.
    Best-effort — never fails the finalize."""
    try:
        a = await db.get(Automation, job.source_id)
        if a is None:
            return None
        try:
            spec_raw = json.loads(a.spec_json or "{}")
        except (ValueError, TypeError):
            spec_raw = {}
        write_tool, connector_id, wrote_count, by_id = \
            _spec_write_shape(spec_raw)
        counts: dict[str, int] = {}
        try:
            for s in json.loads(job.steps_json or "[]"):
                if isinstance(s, dict) and isinstance(s.get("count"), int):
                    cid = by_id.get(s.get("id"))
                    if cid:
                        counts[cid] = counts.get(cid, 0) + s["count"]
        except (ValueError, TypeError):
            pass
        from app.services.automation_verbs import outcome_sentence
        stored = outcome or status
        if (outcome or "") in ("stopped", "superseded"):
            # ND-14: the stopped sentence names how many changes were
            # ALREADY made, so it must come from the write ledger — the
            # spec's write-STEP count would claim a change the stop
            # prevented, and the card would contradict the thread.
            from sqlalchemy import func as _func
            from app.db.models import AutomationWrite
            wrote_count = (await db.execute(
                select(_func.count()).select_from(AutomationWrite)
                .where(AutomationWrite.run_id == job.id)
            )).scalar_one() or 0
        sent = outcome_sentence(
            outcome, write_tool=write_tool, connector_id=connector_id,
            counts=counts, wrote_count=wrote_count,
        )
        a.last_outcome = (stored or "")[:24] or None
        a.last_outcome_text = sent["sentence"][:300]
        a.last_outcome_at = datetime.utcnow()
        await db.commit()
        return {"wrote_count": wrote_count, "counts": counts}
    except Exception as e:  # noqa: BLE001 — a stamp never fails a run
        logger.warning(
            "[automations] last-outcome stamp skipped job=%s: %s",
            job.id[:8], e,
        )
        try:
            await db.rollback()
        except Exception:  # noqa: BLE001
            pass
        return None


async def _record_health(
    db: AsyncSession, automation_id: str, *, ok: bool, error: Optional[str],
    ran: bool = True, clean: Optional[bool] = None,
) -> None:
    """Success resets the failure streak; failure increments it. The
    sweep (sweep.py) owns the auto-pause decision — one place.

    `ran` separates two things this used to conflate: **is the automation
    healthy** (the connector answered, the streak may reset) and **did a
    run happen** (`last_run_at` / `last_status`). A poll that finds no
    fresh events is the first without being the second — nothing ran, so
    there is no run row — and stamping it as a successful run made health
    claim a run the ledger had never heard of.

    Observed live 2026-08-26 (ND-25, R30-D): automation `7681a214` carried
    `last_status: "success"` at `05:21:40`, fifteen seconds after the
    container booted, with **no run row at that time in either the
    per-automation or the global runs feed**. That is this line: a boot-time
    poll found nothing fresh and reported a run.

    Health is the observability surface for the ramp, so a false success
    there is worse than a missing one — it is the reading a human trusts
    when deciding whether an automation is working.

    R31-43 adds `clean`, and it is what re-arms the 3-strike rule.
    `consecutive_failures = 0` was reset on ANY `ok=True` — including an
    empty poll (`ran=False`), which for a polling automation happens at
    least every five minutes against a three-strike threshold, so the
    streak could essentially never reach 3. And measured live on
    2026-08-26 by R31-D: after a run that came back `partial`, health
    still read `last_status: "success"`, so an automation losing one
    source on EVERY run built no streak at all and the auto-pause could
    not arm on the exact shape it exists for.

    So the streak now moves only on a run that actually happened
    (`ran`) and came back clean (`clean`). A `partial` is not a failure
    either — it is not a strike, it just does not clear one. That is
    deliberate: a source that has been broken for a week should keep its
    card standing rather than have the streak wiped by the four
    accounts that still work.
    """
    a = await db.get(Automation, automation_id)
    if a is None:
        return
    if ran:
        a.last_run_at = datetime.utcnow()
    if ok:
        # `clean` defaults to the old meaning when a caller has not been
        # taught the distinction; every caller that CAN tell passes it.
        was_clean = ran if clean is None else bool(clean and ran)
        if was_clean:
            a.consecutive_failures = 0
        if ran:
            a.last_status = "success" if was_clean else "partial"
        if was_clean:
            a.last_error = None
    else:
        a.consecutive_failures = (a.consecutive_failures or 0) + 1
        a.last_status = "failed"
        a.last_error = (error or "")[:2000] or None
    await db.commit()


# ── Event intake ─────────────────────────────────────────────────────


async def ingest_items(
    db: AsyncSession,
    automation: Automation,
    vspec: ValidatedSpec,
    items: list[dict],
) -> list[AutomationEvent]:
    """Insert-or-skip each observed item into the event stream.

    Returns only the FRESH events (a dedupe conflict means an earlier
    poll/push already owns that key). Payloads are trimmed to the
    event fields the spec can reference — the raw provider item never
    lands at rest.
    """
    ev_spec = vspec.event_spec or {}
    fields: dict[str, str] = dict(ev_spec.get("fields") or {})
    dedupe_field = vspec.dedupe_key_field or ev_spec.get("dedupe_field") or "id"

    fresh: list[AutomationEvent] = []
    for item in items:
        key = resolve_path(item, fields.get(dedupe_field, dedupe_field))
        if key is None:
            key = item.get(dedupe_field)
        if key is None:
            continue
        payload = {
            name: resolve_path(item, path) for name, path in fields.items()
        }
        payload.setdefault(dedupe_field, key)
        event = AutomationEvent(
            automation_id=automation.id,
            user_id=automation.user_id,
            dedupe_key=str(key)[:255],
            payload_json=json.dumps(payload, default=str)[:8000],
        )
        try:
            # Savepoint per item: a dedupe conflict rolls back ONLY its
            # own insert — a batch with one replayed item must not lose
            # its genuinely-new siblings (ON CONFLICT DO NOTHING
            # semantics, portable to the sqlite test lane).
            async with db.begin_nested():
                db.add(event)
                await db.flush()
            fresh.append(event)
        except IntegrityError:
            pass
    await db.commit()
    return fresh


def _passes_filter(vspec: ValidatedSpec, payload: dict,
                   facts_ctx: Optional[dict] = None) -> bool:
    """Filter rules: {field: [substrings]} — the event passes when, for
    every constrained field, at least one needle matches
    (case-insensitive). Empty/absent rules match everything.

    R29: a `{{facts.<category>}}` needle matches against the
    automation's fact ledger instead (facts_context — the
    "memory-filtered" leg); `facts_ctx` is the pre-loaded context and
    None keeps the pure-literal fast path byte-identical."""
    from .facts_context import needle_matches

    for fld, needles in (vspec.filter_rules or {}).items():
        if not needles:
            continue
        if not isinstance(needles, list):
            needles = [needles]
        value = str(payload.get(fld) or "").lower()
        if not any(needle_matches(n, value, facts_ctx) for n in needles):
            return False
    return True


# ── Run pipeline ─────────────────────────────────────────────────────


async def run_event(
    db: AsyncSession,
    automation: Automation,
    vspec: ValidatedSpec,
    event: AutomationEvent,
) -> str:
    """Full pipeline for one fresh event. Returns the event's terminal
    status. Bounded by the run cap."""
    try:
        return await asyncio.wait_for(
            _run_event_inner(db, automation, vspec, event),
            timeout=AUTOMATION_RUN_CAP_S,
        )
    except asyncio.TimeoutError:
        logger.warning("[automations] run cap hit automation=%s event=%s",
                       automation.id, event.id)
        if event.job_id:
            await _finalize_job(
                db, event.job_id, status="failed", outcome="run_cap",
                error_class="timeout",
                user_message="The run exceeded the 3-minute cap and was stopped.",
            )
        await _record_health(db, automation.id, ok=False, error="run cap exceeded")
        return "failed"


async def _run_event_inner(
    db: AsyncSession,
    automation: Automation,
    vspec: ValidatedSpec,
    event: AutomationEvent,
) -> str:
    try:
        payload = json.loads(event.payload_json) if event.payload_json else {}
    except (ValueError, TypeError):
        payload = {}

    # evaluate — a filtered event is recorded, never run (the honest
    # answer to "why didn't my rule fire?").
    from .facts_context import load_facts_context
    facts_ctx = await load_facts_context(
        db, automation.id, vspec.filter_rules,
    )
    if not _passes_filter(vspec, payload, facts_ctx):
        event.status = "skipped_filter"
        await db.commit()
        return event.status

    # Atomic run claim: build_jobs (source_id, idempotency_key).
    job = await JobRunner().create_job(
        job_type="automation_run",
        spec=TaskSpec(
            user_id=automation.user_id,
            channel="automation",
            source_kind="automation",
            source_id=automation.id,
            config_json={"automation_event_id": event.id},
        ),
        title=f"{automation.name}"[:100],
        idempotency_key=f"evt:{event.id}",
        status="running",
        steps_json=_new_steps(vspec),
        layer=0,
    )
    event.status = "run"
    event.job_id = job.id
    await db.commit()
    await on_run_created(db, job=job, automation=automation)
    await _advance(db, job.id, "evaluate")

    return await _prepare_and_write(db, automation, vspec, job.id, payload,
                                    idem_prefix=f"evt:{event.id}")


async def run_schedule_fire(
    db: AsyncSession,
    automation: Automation,
    vspec: ValidatedSpec,
    fire_key: str,
) -> str:
    """Schedule-mode fire: no event, straight to the action. The fire
    key (user-local date or instant) is the idempotency claim."""

    async def _inner() -> str:
        job = await JobRunner().create_job(
            job_type="automation_run",
            spec=TaskSpec(
                user_id=automation.user_id,
                channel="automation",
                source_kind="automation",
                source_id=automation.id,
                config_json={"fire_key": fire_key},
            ),
            title=f"{automation.name}"[:100],
            idempotency_key=f"fire:{fire_key}"[:120],
            status="running",
            steps_json=_new_steps(vspec),
            layer=0,
        )
        await on_run_created(db, job=job, automation=automation)
        await _advance(db, job.id, "evaluate")
        return await _prepare_and_write(
            db, automation, vspec, job.id, {},
            idem_prefix=f"fire:{fire_key}",
        )

    try:
        return await asyncio.wait_for(_inner(), timeout=AUTOMATION_RUN_CAP_S)
    except asyncio.TimeoutError:
        await _record_health(db, automation.id, ok=False,
                             error="run cap exceeded")
        return "failed"


async def _prepare_and_write(
    db: AsyncSession,
    automation: Automation,
    vspec: ValidatedSpec,
    job_id: str,
    event_payload: dict,
    *,
    idem_prefix: str,
    stage_only: bool = False,
) -> str:
    """prepare (render params) → write (stage to outbox) → record."""
    raw = json.loads(automation.spec_json)
    grant_target = (raw.get("action") or {}).get("grant_target") or {}

    tool = vspec.action_tool
    if tool in _FORBIDDEN_TOOLS:
        await _finalize_job(
            db, job_id, status="failed", outcome="forbidden_tool",
            error_class="policy",
            user_message="Automations never send mail — use a draft action.",
        )
        await _record_health(db, automation.id, ok=False,
                             error="forbidden tool " + tool)
        return "failed"

    params = render_params(
        vspec.action_params_template,
        event=event_payload, grant_target=grant_target,
    )
    await _advance(db, job_id, "prepare")

    outbox = AutomationOutbox(
        user_id=automation.user_id,
        automation_id=automation.id,
        job_id=job_id,
        connector_id=vspec.action_connector_id,
        tool_name=tool,
        payload_json=json.dumps(params, sort_keys=True, default=str),
        grant_id=vspec.grant_id,
        idempotency_key=f"{idem_prefix}:w0"[:128],
        execute_after=datetime.utcnow()
        + timedelta(seconds=AUTOMATION_OUTBOX_UNDO_WINDOW_S),
    )
    db.add(outbox)
    try:
        await db.flush()
    except IntegrityError:
        # A retried run already staged this exact write — the outbox
        # idempotency gate holds; nothing new to stage.
        await db.rollback()
        logger.info("[automations] outbox idempotency hit %s", idem_prefix)
        return "run"
    await db.commit()
    await _advance(db, job_id, "write")

    if stage_only:
        # test_run stops here: the row exists, the flush loop will send
        # it after the same undo window a real fire gets.
        return "staged"

    # The flush loop finalizes the job when the row executes; the run
    # stays `running` across the undo window (seconds), inside the cap.
    from .outbox import flush_row_when_due
    await flush_row_when_due(db, outbox.id)
    return "run"


async def execute_test_run(
    db: AsyncSession, automation: Automation, vspec: ValidatedSpec,
) -> dict:
    """One synthetic fire. Poll modes poll for real (read-only); push/
    schedule modes build a sample event from the spec's declared
    fields. The write stages into the real outbox with the real undo
    window — a test run is a real run with a synthetic trigger."""
    sample: dict[str, Any] = {}
    if vspec.trigger_mode == "poll":
        items = await _poll_once(automation, vspec)
        if items:
            ev_spec = vspec.event_spec or {}
            fields = dict(ev_spec.get("fields") or {})
            sample = {
                name: resolve_path(items[0], path)
                for name, path in fields.items()
            }
    if not sample:
        fields = dict((vspec.event_spec or {}).get("fields") or {})
        sample = {name: f"<{name}>" for name in fields} or {"sample": "<test>"}

    job = await JobRunner().create_job(
        job_type="automation_run",
        spec=TaskSpec(
            user_id=automation.user_id,
            channel="automation",
            source_kind="automation",
            source_id=automation.id,
            config_json={"test_run": True},
        ),
        title=f"[test] {automation.name}"[:100],
        idempotency_key=f"test:{uuid.uuid4()}",
        status="running",
        steps_json=_new_steps(vspec),
        layer=0,
    )
    await on_run_created(db, job=job, automation=automation)
    await _advance(db, job.id, "evaluate")
    status = await _prepare_and_write(
        db, automation, vspec, job.id, sample,
        idem_prefix=f"test:{job.id}", stage_only=True,
    )
    return {"job_id": job.id, "status": status, "sample_event": sample}


# ── Poll leg ─────────────────────────────────────────────────────────


async def _poll_once(
    automation: Automation, vspec: ValidatedSpec,
) -> list[dict]:
    """One read call via the platform RPC; returns the item list."""
    ev_spec = vspec.event_spec or {}
    source_tool = ev_spec.get("source_tool")
    if not source_tool:
        return []
    args = dict(ev_spec.get("poll_args") or {})
    args.update(vspec.trigger_params or {})
    result = await reg.dispatch_via_platform(
        automation.user_id,
        connector_id=vspec.trigger_connector_id or "",
        tool_name=source_tool,
        tool_input=args,
        automation_id=automation.id,
    )
    if result.get("kind") != "ok":
        raise RuntimeError(
            f"poll failed: {result.get('kind')}: "
            f"{str(result.get('message') or '')[:200]}"
        )
    try:
        content = json.loads(result.get("content") or "{}")
    except (ValueError, TypeError):
        content = {}
    items_path = ev_spec.get("items_path")
    items = resolve_path(content, items_path) if items_path else content
    return items if isinstance(items, list) else []


async def poll_and_run(
    db: AsyncSession, automation: Automation, vspec: ValidatedSpec,
) -> dict:
    """The poll-mode fire: read, diff via the event dedupe gate, run
    each fresh event. Catch-up is capped by the poll page size — a
    paused week does not replay a week."""
    items = await _poll_once(automation, vspec)
    fresh = await ingest_items(db, automation, vspec, items)
    ran = 0
    failed = 0
    for event in fresh:
        status = await run_event(db, automation, vspec, event)
        if status == "failed":
            failed += 1
        elif status == "run":
            ran += 1
    if failed == 0:
        # `ran=ran > 0`: a poll that found nothing fresh proves the
        # connector answered, not that a run happened. See `_record_health`.
        await _record_health(db, automation.id, ok=True, error=None,
                             ran=ran > 0)
    return {"observed": len(items), "fresh": len(fresh),
            "ran": ran, "failed": failed}
