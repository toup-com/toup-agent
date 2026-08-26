"""The v2 run pipeline — fire → evaluate → steps → writes → record.

Round 28. v1 runs keep executor.py untouched; this module owns specs
with `version: 2`. Same rails, same primitives:

  - runs ARE BuildJobs (`job_type='automation_run'`), idempotency keys
    unchanged; the job's steps_json is dynamic: `evaluate`, one entry
    per spec step, `record`.
  - events: insert-or-skip on UNIQUE (automation_id, dedupe_key), with
    the v2 key namespaced per source: "<source_id>:<value>".
  - read steps run inline through the platform dispatch RPC (non-
    mutating, no grant — the dispatcher still fails closed on unknown
    tools); `collect` folds their items into {{steps.<id>.text}} /
    {{steps.<id>.count}}.
  - write steps stage to automation_outbox with keys
    "<prefix>:w<n>" (v1's single write is w0 — the same scheme), then
    flush after the normal undo window. One grant per write step.
  - mail rail unchanged: send tools are refused outright.
  - memory: the run reads its namespace once at fire time
    ({{memory.<key>}}) and writes it back after the run, in its own
    session (memory.py).
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Optional

from sqlalchemy.exc import IntegrityError

from app.db.models import (
    Automation, AutomationEvent, AutomationOutbox, BuildJob,
    AUTOMATION_OUTBOX_UNDO_WINDOW_S, AUTOMATION_RUN_CAP_S,
)
from app.agent.job_runner import JobRunner, TaskSpec
from app.agent import job_steps
from .spec import render_value, render_with_ctx, resolve_path
from .spec_v2 import ValidatedSpecV2, ValidatedSource, ValidatedStep
from .executor import _FORBIDDEN_TOOLS, _finalize_job, _record_health
from .session import on_run_created
from . import memory as engine_memory
from . import registry as reg

logger = logging.getLogger(__name__)


async def merge_job_config(db, job_id: str, **extras) -> None:
    """Merge keys into a run job's config_json (read-modify-write,
    commits). The one seam for run-scoped extras — used here for
    steps_partial; sibling rounds stamp their own keys through it
    rather than growing second implementations."""
    from app.db.models import BuildJob
    job = await db.get(BuildJob, job_id)
    if job is None:
        return
    cfg = dict(job.config_json or {})
    cfg.update(extras)
    job.config_json = cfg
    await db.commit()


# ── Job-steps plumbing (dynamic step list) ───────────────────────────


def _step_order(vspec: ValidatedSpecV2) -> list[str]:
    return ["evaluate", *[st.id for st in vspec.steps], "record"]


def _new_steps_v2(vspec: ValidatedSpecV2) -> str:
    """Humanized labels at mint (R29): steps_json is the shared
    substrate (runs API, job cards, web) — spec steps wear their
    tool's verb + connector brand, engine phases the orb (brand
    None). The verb dictionary is the only composer."""
    from app.services.automation_verbs import step_verb

    by_id = {st.id: st for st in vspec.steps}
    now = datetime.utcnow()
    steps = []
    for s in _step_order(vspec):
        st = by_id.get(s)
        if st is not None:
            v = step_verb(st.tool, st.connector_id)
        else:
            v = step_verb(None, None, phase=s)
        steps.append({
            "id": s, "type": "generic", "label": v["label"],
            "brand": v["brand"],
            "status": "pending", "started_at": None, "completed_at": None,
        })
    return job_steps.dump_steps(job_steps.open_first_step(steps, now))


async def _advance_v2(db, job_id: str, vspec: ValidatedSpecV2,
                      done_step: str,
                      count: Optional[int] = None) -> None:
    from app.db.models import BuildJob
    job = await db.get(BuildJob, job_id)
    if job is None:
        return
    order = _step_order(vspec)
    if done_step not in order:
        return
    steps = job_steps.parse_steps(job.steps_json)
    if count is not None:
        # A collected read's count rides the step dict (R29): the runs
        # API's done-form verbs and the last-outcome sentence both read
        # it back — steps_json is the one substrate.
        for s in steps:
            if s.get("id") == done_step:
                s["count"] = count
                break
    job.steps_json = job_steps.dump_steps(
        job_steps.advance_steps(
            steps, order.index(done_step), datetime.utcnow(),
            fallback_start=job.created_at,
        )
    )
    await db.commit()


# ── Event intake (per-source dedupe namespace) ───────────────────────


async def ingest_items_v2(
    db,
    automation: Automation,
    source: ValidatedSource,
    items: list[dict],
) -> list[AutomationEvent]:
    """Insert-or-skip each observed item, dedupe-keyed as
    "<source_id>:<value>" so two sources can never collide in the
    per-automation UNIQUE gate. Payloads are trimmed to the declared
    event fields plus `_source`."""
    ev_spec = source.event_spec or {}
    fields: dict[str, str] = dict(ev_spec.get("fields") or {})
    dedupe_field = source.dedupe_key_field or ev_spec.get("dedupe_field") or "id"

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
        payload["_source"] = source.id
        event = AutomationEvent(
            automation_id=automation.id,
            user_id=automation.user_id,
            dedupe_key=f"{source.id}:{key}"[:255],
            payload_json=json.dumps(payload, default=str)[:8000],
        )
        try:
            async with db.begin_nested():
                db.add(event)
                await db.flush()
            fresh.append(event)
        except IntegrityError:
            pass
    await db.commit()
    return fresh


def _passes_filter_v2(source: ValidatedSource, payload: dict,
                      variables: dict,
                      facts_ctx: Optional[dict] = None) -> bool:
    """v1 filter semantics per source, with {{var.*}} rendered in the
    needles so templates can parameterize filters.

    R29: a `{{facts.<category>}}` needle matches against the fact
    ledger (facts_context, the "memory-filtered" leg) and is
    intercepted BEFORE render_value — the var renderer would blank the
    unknown template and turn the needle into a match-nothing literal
    instead of a ledger lookup."""
    from .facts_context import facts_needle_category, needle_matches

    ctx = {"var": variables or {}}
    for fld, needles in (source.filter_rules or {}).items():
        if not needles:
            continue
        if not isinstance(needles, list):
            needles = [needles]
        value = str(payload.get(fld) or "").lower()
        ok = False
        for n in needles:
            if facts_needle_category(n) is not None:
                if needle_matches(n, value, facts_ctx):
                    ok = True
                    break
                continue
            rendered = str(render_value(str(n), ctx)).lower()
            if rendered and rendered in value:
                ok = True
                break
        if not ok:
            return False
    return True


# ── Read-step execution ──────────────────────────────────────────────


def _collect_result(step: ValidatedStep, content: dict,
                    variables: dict) -> dict:
    """Fold a read step's JSON result into {text, count, ok} per the
    step's collect declaration."""
    collect = step.collect
    if not collect:
        return {"ok": True, "text": "", "count": 0}
    items = resolve_path(content, collect["items_path"])
    if not isinstance(items, list):
        items = []
    count = len(items)
    limit = int(collect.get("limit") or 10)
    fmt = collect.get("format")
    lines: list[str] = []
    raw_fields: list[dict] = []
    if fmt:
        for item in items[:limit]:
            item_fields = {
                name: resolve_path(item, path)
                for name, path in (collect.get("fields") or {}).items()
            }
            raw_fields.append(item_fields)
            lines.append(str(render_value(
                fmt, {"item": item_fields, "var": variables or {}},
            )))
    text = (collect.get("join") or "\n").join(lines)
    if count == 0:
        text = collect.get("empty_text") or ""
    # `lines`/`raw_fields` are R30 ledger inputs (mechanical item titles
    # + the narrator's raw material); templates keep reading only
    # text/count via resolve_path — extra keys are inert there.
    return {"ok": True, "text": text, "count": count,
            "lines": lines, "raw_fields": raw_fields}


def _skipped_result(step: ValidatedStep) -> dict:
    empty = (step.collect or {}).get("empty_text") or ""
    return {"ok": False, "text": empty, "count": 0}


async def _execute_read_step(
    automation: Automation,
    step: ValidatedStep,
    ctx: dict,
) -> dict:
    """One inline read via the platform RPC. Raises RuntimeError on a
    non-ok result — the caller applies on_error."""
    params = render_with_ctx(step.params_template, ctx)
    result = await reg.dispatch_via_platform(
        automation.user_id,
        connector_id=step.connector_id,
        tool_name=step.tool,
        tool_input=params,
        automation_id=automation.id,
    )
    if result.get("kind") != "ok":
        raise RuntimeError(
            f"step {step.id!r} failed: {result.get('kind')}: "
            f"{str(result.get('message') or '')[:200]}"
        )
    try:
        content = json.loads(result.get("content") or "{}")
    except (ValueError, TypeError):
        content = {}
    if not isinstance(content, dict):
        content = {}
    return _collect_result(step, content, ctx.get("var") or {})


# ── The run pipeline ─────────────────────────────────────────────────


async def _run_steps(
    db,
    automation: Automation,
    vspec: ValidatedSpecV2,
    job_id: str,
    event_payload: dict,
    source: Optional[ValidatedSource],
    *,
    idem_prefix: str,
    stage_only: bool = False,
    resume: bool = False,
) -> str:
    """steps (reads) → stage writes → flush → record + memory.

    `resume` (R30 §4.3): the run is a reopened stopped run — reads
    re-execute (the honest answer to a moved dedupe window), already-
    staged outbox rows are REUSED instead of conflicting, and the
    narration does not repeat its opening line."""
    mem_ctx = await engine_memory.read_context(db, automation)
    ctx: dict[str, Any] = {
        "event": event_payload or {},
        "source": {
            "id": source.id if source else "",
            "connector_id": (source.connector_id or "") if source else "",
            "event": (source.event or "") if source else "",
        },
        "var": vspec.variables or {},
        "steps": {},
        "memory": mem_ctx,
    }

    # Mail rail first — checked before any step runs, exactly like v1.
    for st in vspec.write_steps:
        if st.tool in _FORBIDDEN_TOOLS:
            await _finalize_job(
                db, job_id, status="failed", outcome="forbidden_tool",
                error_class="policy",
                user_message="Automations never send mail — use a draft "
                             "action.",
            )
            await _record_health(db, automation.id, ok=False,
                                 error="forbidden tool " + st.tool)
            return "failed"

    # R30 v3 ledger context — best-effort throughout: a ledger failure
    # never changes the run's outcome (the typed record degrades, the
    # work does not).
    from . import ledger as _ledger
    from . import run_v3 as _rv3
    from app.services import automation_verbs as _verbs
    import time as _time
    thread = None
    job_row = await db.get(BuildJob, job_id)
    try:
        thread = await _ledger.thread_for(db, automation.id)
    except Exception:  # noqa: BLE001
        thread = None
    total_steps = len(vspec.steps)
    step_no = 0
    tool_turn_by_step: dict[str, dict] = {}

    async def _stop_boundary(at_step: int) -> bool:
        """§4.3: the stop takes effect at the next step boundary."""
        if job_row is None:
            return False
        try:
            if await _rv3.stop_requested(db, job_id):
                await _rv3.handle_stop(
                    db, automation=automation, job=job_row,
                    step_index=at_step,
                )
                return True
        except Exception as e:  # noqa: BLE001
            logger.debug("[automations] stop boundary check failed: %s", e)
        return False

    partial = False
    for st in vspec.steps:
        if st.mutates:
            break
        if await _stop_boundary(step_no):
            return "stopped"
        step_no += 1
        sentence = _verbs.live_sentence(st.connector_id, st.tool)
        try:
            await _ledger.emit_progress(
                automation.user_id, run_id=job_id,
                automation_id=automation.id, step=step_no,
                total=total_steps, sentence=sentence,
                fraction=(step_no - 1) / max(total_steps, 1),
                status="running",
            )
            if job_row is not None:
                await _rv3.notify_progress(
                    db, automation=automation, job=job_row, step=step_no,
                    total=total_steps, sentence=sentence,
                    fraction=(step_no - 1) / max(total_steps, 1),
                )
        except Exception:  # noqa: BLE001
            pass
        _t0 = _time.monotonic()
        step_failed_reason = None
        try:
            ctx["steps"][st.id] = await _execute_read_step(automation, st, ctx)
        except Exception as e:  # noqa: BLE001 — transport/shape errors
            step_failed_reason = _failure_reason(e)
            if st.on_error == "skip":
                logger.info("[automations] step %s skipped (%s) on %s",
                            st.id, e, automation.id)
                ctx["steps"][st.id] = _skipped_result(st)
                partial = True
            else:
                await _append_read_turn(
                    db, thread=thread, automation=automation, job_id=job_id,
                    step=st, result=None, ms=int((_time.monotonic() - _t0) * 1000),
                    ok=False, reason=step_failed_reason,
                    turn_index=tool_turn_by_step,
                )
                await _finalize_job(
                    db, job_id, status="failed", outcome="step_failed",
                    error_class="tool_error",
                    user_message=f"Step {st.id!r} failed: {str(e)[:200]}",
                )
                await _record_health(db, automation.id, ok=False,
                                     error=str(e)[:500])
                return "failed"
        step_result = ctx["steps"].get(st.id) or {}
        await _append_read_turn(
            db, thread=thread, automation=automation, job_id=job_id,
            step=st, result=step_result,
            ms=int((_time.monotonic() - _t0) * 1000),
            ok=step_failed_reason is None, reason=step_failed_reason,
            turn_index=tool_turn_by_step,
        )
        await _advance_v2(
            db, job_id, vspec, st.id,
            count=step_result.get("count")
            if isinstance(step_result.get("count"), int) else None,
        )

    if partial:
        # The aggregate finalizer reads this to report `partial`
        # honestly when the writes themselves succeed.
        await merge_job_config(db, job_id, steps_partial=True)

    # §4.3: the last boundary before writes — a stop that arrived during
    # the reads must land HERE; no write step may start after it.
    if vspec.write_steps and await _stop_boundary(step_no):
        return "stopped"

    # Stage every write in one transaction: a replayed run conflicts on
    # w0 and rolls the whole batch back — all-or-nothing, never a
    # half-staged retry.
    rows: list[AutomationOutbox] = []
    execute_after = datetime.utcnow() + timedelta(
        seconds=AUTOMATION_OUTBOX_UNDO_WINDOW_S)
    for n, st in enumerate(vspec.write_steps):
        step_ctx = dict(ctx)
        step_ctx["grant"] = {"target": st.grant_target or {}}
        params = render_with_ctx(st.params_template, step_ctx)
        rows.append(AutomationOutbox(
            user_id=automation.user_id,
            automation_id=automation.id,
            job_id=job_id,
            connector_id=st.connector_id,
            tool_name=st.tool,
            payload_json=json.dumps(params, sort_keys=True, default=str),
            grant_id=st.grant_id,
            idempotency_key=f"{idem_prefix}:w{n}"[:128],
            execute_after=execute_after,
            display_json=json.dumps(
                _write_display(st), sort_keys=True, default=str,
            ),
        ))
    db.add_all(rows)
    try:
        await db.flush()
    except IntegrityError:
        await db.rollback()
        if not resume:
            logger.info("[automations] outbox idempotency hit %s",
                        idem_prefix)
            return "run"
        # Resume: the stop landed after staging — reuse the surviving
        # staged rows (an executed/cancelled one stays terminal; the
        # claim gate keeps a double-send impossible either way).
        from sqlalchemy import select as _select
        rows = list((await db.execute(
            _select(AutomationOutbox)
            .where(AutomationOutbox.job_id == job_id)
            .where(AutomationOutbox.status == "staged")
        )).scalars())
        now2 = datetime.utcnow()
        for row in rows:
            if row.execute_after and row.execute_after < now2:
                row.execute_after = now2 + timedelta(
                    seconds=AUTOMATION_OUTBOX_UNDO_WINDOW_S)
        await db.commit()
    else:
        await db.commit()
    # ND-4: a write step's done-form verb lands when the write actually
    # EXECUTES (outbox ok branch, keyed by display_json.step_id) — never
    # at staging. A refused write must not wear "Posted to Slack".

    if stage_only:
        return "staged"

    # R30 narration phase 1 (pre-write): the opening line, item whys,
    # think turns, the draft's text. Result + close land in phase 2 so
    # the thread never claims a change before the write executed.
    narration = await _narrate_phase1(
        db, automation=automation, vspec=vspec, job_id=job_id,
        thread=thread, tool_turn_by_step=tool_turn_by_step,
        partial=partial,
    )

    from .outbox import flush_row_when_due
    statuses = []
    for row in rows:
        statuses.append(await flush_row_when_due(db, row.id))

    await _narrate_phase2(
        db, automation=automation, job_id=job_id, thread=thread,
        narration=narration,
        writes_ok=not any(s == "failed" for s in statuses),
    )

    outcome = "sent"
    if any(s == "failed" for s in statuses):
        outcome = "failed"
    elif partial:
        outcome = "partial"
    counts = {
        sid: res.get("count")
        for sid, res in ctx["steps"].items()
        if isinstance(res, dict) and res.get("count") is not None
    }
    await engine_memory.write_after_run(
        user_id=automation.user_id,
        automation_id=automation.id,
        automation_name=automation.name,
        outcome=outcome,
        counts=counts,
    )
    return "run"


# ── Entry points (cap-bounded, mirroring executor.py) ────────────────


async def run_event_v2(
    db,
    automation: Automation,
    vspec: ValidatedSpecV2,
    source: ValidatedSource,
    event: AutomationEvent,
) -> str:
    try:
        return await asyncio.wait_for(
            _run_event_inner(db, automation, vspec, source, event),
            timeout=AUTOMATION_RUN_CAP_S,
        )
    except asyncio.TimeoutError:
        logger.warning("[automations] run cap hit automation=%s event=%s",
                       automation.id, event.id)
        if event.job_id:
            await _finalize_job(
                db, event.job_id, status="failed", outcome="run_cap",
                error_class="timeout",
                user_message="The run exceeded the 3-minute cap and was "
                             "stopped.",
            )
        await _record_health(db, automation.id, ok=False,
                             error="run cap exceeded")
        return "failed"


async def _run_event_inner(
    db,
    automation: Automation,
    vspec: ValidatedSpecV2,
    source: ValidatedSource,
    event: AutomationEvent,
) -> str:
    try:
        payload = json.loads(event.payload_json) if event.payload_json else {}
    except (ValueError, TypeError):
        payload = {}

    from .facts_context import load_facts_context
    facts_ctx = await load_facts_context(
        db, automation.id, source.filter_rules,
    )
    if not _passes_filter_v2(source, payload, vspec.variables, facts_ctx):
        event.status = "skipped_filter"
        await db.commit()
        return event.status

    job = await JobRunner().create_job(
        job_type="automation_run",
        spec=TaskSpec(
            user_id=automation.user_id,
            channel="automation",
            source_kind="automation",
            source_id=automation.id,
            config_json={"automation_event_id": event.id,
                         "source_id": source.id},
        ),
        title=f"{automation.name}"[:100],
        idempotency_key=f"evt:{event.id}",
        status="running",
        steps_json=_new_steps_v2(vspec),
        layer=0,
    )
    event.status = "run"
    event.job_id = job.id
    await db.commit()
    await on_run_created(db, job=job, automation=automation)
    await _advance_v2(db, job.id, vspec, "evaluate")
    from . import run_v3 as _rv3_open
    await _rv3_open.open_run(db, automation=automation, job=job,
                             kind="scheduled", total_steps=len(vspec.steps))

    return await _run_steps(db, automation, vspec, job.id, payload, source,
                            idem_prefix=f"evt:{event.id}")


async def _finalize_on_cap(job_id: str) -> None:
    """ND-7b: terminalize a cap-hit run on a FRESH session — the run's
    own session was cancelled mid-flight by wait_for and may be wedged
    in an open transaction; reusing it can hang or raise, which is how
    the row stayed `running` forever (the R27 zombie class reborn)."""
    from app.db.database import async_session_maker
    try:
        async with async_session_maker() as fresh:
            await _finalize_job(
                fresh, job_id, status="failed", outcome="run_cap",
                error_class="timeout",
                user_message="The run exceeded the 3-minute cap and was "
                             "stopped.",
            )
    except Exception as e:  # noqa: BLE001 — the sweep is the backstop
        logger.warning("[automations] cap finalize failed job=%s: %s",
                       job_id[:8], e)


async def run_schedule_fire_v2(
    db,
    automation: Automation,
    vspec: ValidatedSpecV2,
    source: ValidatedSource,
    fire_key: str,
    run_kind: str = "scheduled",
) -> str:
    # ND-7b: the job is minted INSIDE the wait_for — the ref carries its
    # id out so the cap handler can finalize (the old handler could not
    # even name the job and returned with the row still `running`).
    job_ref: dict = {}

    async def _inner() -> str:
        # §4.3: the next scheduled fire supersedes a still-stopped run —
        # its stop note was already written; no new turn.
        try:
            from . import run_v3 as _rv3_sup
            await _rv3_sup.supersede_stopped_run(
                db, automation_id=automation.id,
            )
        except Exception:  # noqa: BLE001
            pass
        job = await JobRunner().create_job(
            job_type="automation_run",
            spec=TaskSpec(
                user_id=automation.user_id,
                channel="automation",
                source_kind="automation",
                source_id=automation.id,
                config_json={"fire_key": fire_key, "source_id": source.id},
            ),
            title=f"{automation.name}"[:100],
            idempotency_key=f"fire:{fire_key}"[:120],
            status="running",
            steps_json=_new_steps_v2(vspec),
            layer=0,
        )
        job_ref["id"] = job.id
        await on_run_created(db, job=job, automation=automation)
        await _advance_v2(db, job.id, vspec, "evaluate")
        from . import run_v3 as _rv3_open
        await _rv3_open.open_run(db, automation=automation, job=job,
                                 kind=run_kind,
                                 total_steps=len(vspec.steps))
        return await _run_steps(db, automation, vspec, job.id, {}, source,
                                idem_prefix=f"fire:{fire_key}")

    try:
        return await asyncio.wait_for(_inner(), timeout=AUTOMATION_RUN_CAP_S)
    except asyncio.TimeoutError:
        logger.warning("[automations] run cap hit automation=%s fire=%s",
                       automation.id, fire_key[:40])
        if job_ref.get("id"):
            await _finalize_on_cap(job_ref["id"])
        try:
            await _record_health(db, automation.id, ok=False,
                                 error="run cap exceeded")
        except Exception:  # noqa: BLE001 — the session may be wedged too
            logger.warning("[automations] cap health record failed "
                           "automation=%s", automation.id)
        return "failed"


# ── Poll leg ─────────────────────────────────────────────────────────


async def _poll_once_v2(
    automation: Automation, source: ValidatedSource,
) -> list[dict]:
    ev_spec = source.event_spec or {}
    source_tool = ev_spec.get("source_tool")
    if not source_tool:
        return []
    args = dict(ev_spec.get("poll_args") or {})
    args.update(source.params or {})
    result = await reg.dispatch_via_platform(
        automation.user_id,
        connector_id=source.connector_id or "",
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


async def poll_and_run_v2(
    db, automation: Automation, vspec: ValidatedSpecV2,
    source: ValidatedSource,
) -> dict:
    items = await _poll_once_v2(automation, source)
    fresh = await ingest_items_v2(db, automation, source, items)
    ran = 0
    failed = 0
    for event in fresh:
        status = await run_event_v2(db, automation, vspec, source, event)
        if status == "failed":
            failed += 1
        elif status == "run":
            ran += 1
    if failed == 0:
        # See `_record_health`: a poll with nothing fresh is connector
        # health, not a run. Stamping it as a run made the health object
        # claim runs the ledger had never heard of (ND-25).
        await _record_health(db, automation.id, ok=True, error=None,
                             ran=ran > 0)
    return {"observed": len(items), "fresh": len(fresh),
            "ran": ran, "failed": failed}


# ── Test run ─────────────────────────────────────────────────────────


async def execute_test_run_v2(
    db, automation: Automation, vspec: ValidatedSpecV2,
) -> dict:
    """One synthetic fire through the real rails: read steps run for
    real, writes stop at the staged outbox row and go out after the
    normal undo window — a test run is a real run with a synthetic
    trigger."""
    sample: dict[str, Any] = {}
    source: Optional[ValidatedSource] = None
    for s in vspec.sources:
        if s.mode in ("push", "poll"):
            source = s
            break
    if source is not None and source.mode == "poll":
        try:
            items = await _poll_once_v2(automation, source)
        except Exception:  # noqa: BLE001 — sample is best-effort
            items = []
        if items:
            ev_spec = source.event_spec or {}
            fields = dict(ev_spec.get("fields") or {})
            sample = {
                name: resolve_path(items[0], path)
                for name, path in fields.items()
            }
    if not sample and source is not None:
        fields = dict((source.event_spec or {}).get("fields") or {})
        sample = {name: f"<{name}>" for name in fields} or {"sample": "<test>"}
    if source is None:
        source = vspec.schedule_source()

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
        steps_json=_new_steps_v2(vspec),
        layer=0,
    )
    await on_run_created(db, job=job, automation=automation)
    await _advance_v2(db, job.id, vspec, "evaluate")
    from . import run_v3 as _rv3_open
    await _rv3_open.open_run(db, automation=automation, job=job,
                             kind="run_now", total_steps=len(vspec.steps))
    status = await _run_steps(
        db, automation, vspec, job.id, sample, source,
        idem_prefix=f"test:{job.id}", stage_only=True,
    )
    return {"job_id": job.id, "status": status, "sample_event": sample}


# ── R30: the v3 ledger + narration seams ─────────────────────────────


def _failure_reason(e: Exception) -> str:
    """Map a read-step exception onto the verb dictionary's failure
    reasons. The dispatch error message carries the RPC envelope kind."""
    msg = str(e)
    for token in ("reauth_required", "scope_missing", "provider_down",
                  "rate_limited", "timeout"):
        if token in msg:
            return token
    return "unreachable"


def _write_display(st: ValidatedStep) -> dict:
    """The write's honest display form, snapshotted at staging
    (CONTRACTS-R30 §4.8) — the flush path cannot read platform grants,
    so the pinned target label rides here."""
    from app.services.automation_verbs import turn_action
    from .draft_card import DRAFT_TOOLS
    target = (st.grant_target or {})
    label = target.get("label") or target.get("id")
    is_draft = st.tool in DRAFT_TOOLS
    audience = "you" if is_draft or str(label or "").lower().startswith("dm") \
        else "others"
    act = turn_action(
        st.connector_id, st.tool, kind="write", ok=True,
        target=label, audience=audience,
    )
    return {
        "what": act["action"],
        "target": label,
        "audience": audience,
        "reversible": is_draft,
        # ND-4: lets the outbox flip THIS step's verb to its done form
        # only when the write actually executed (or to failed when not).
        "step_id": st.id,
    }


async def _append_read_turn(
    db, *, thread, automation: Automation, job_id: str,
    step: ValidatedStep, result: Optional[dict], ms: int, ok: bool,
    reason: Optional[str], turn_index: dict,
) -> None:
    """The mechanical tool turn for one read step — engine facts only
    (action/detail/ms/ok/items); the narrator fills whys in place."""
    if thread is None:
        return
    try:
        from . import ledger as _ledger
        from app.services import automation_verbs as _verbs
        if ok:
            count = (result or {}).get("count")
            act = _verbs.turn_action(
                step.connector_id, step.tool, kind="read", ok=True,
                count=count if isinstance(count, int) else None,
            )
            items = [
                {"title": str(line)[:200], "sub": "", "why": ""}
                for line in ((result or {}).get("lines") or [])
            ]
            steps_lines = []
        else:
            act = _verbs.failure_action(step.connector_id, reason)
            name = _verbs.display_name(step.connector_id) or "the account"
            items = []
            steps_lines = [
                {"text": f"Asked {name} for what changed", "ok": True},
                {"text": act["detail"].capitalize() or "It did not answer",
                 "ok": False},
            ]
        turn = await _ledger.append_turn(
            db, user_id=automation.user_id, thread=thread, run_id=job_id,
            kind="tool",
            payload={
                "account_id": step.connector_id or "",
                "tool_kind": "read",
                "action": act["action"], "detail": act["detail"],
                "ok": ok, "ms": max(int(ms), 0),
                "steps": steps_lines, "items": items,
                "write_ids": [], "rest": "",
            },
        )
        turn_index[step.id] = turn
    except Exception as e:  # noqa: BLE001 — the ledger degrades, the run does not
        logger.debug("[automations] read turn skipped step=%s: %s",
                     step.id, e)


async def _recall_facts(db, automation: Automation) -> list[dict]:
    """What the narrator may use to judge: this automation's scoped
    facts + globals (memory v2), falling back to the R29 ledger."""
    try:
        from sqlalchemy import select as _select
        from app.db.models import MemoryFact
        rows = list((await db.execute(
            _select(MemoryFact)
            .where(MemoryFact.user_id == automation.user_id)
            .where(MemoryFact.scope.in_((automation.id, "global")))
            .order_by(MemoryFact.learned_at.desc())
            .limit(24)
        )).scalars())
        if rows:
            return [{"category": r.category, "text": r.text} for r in rows]
        from app.db.models import AutomationFact
        legacy = list((await db.execute(
            _select(AutomationFact)
            .where(AutomationFact.automation_id == automation.id)
            .order_by(AutomationFact.updated_at.desc())
            .limit(24)
        )).scalars())
        return [{"category": r.category, "text": r.text} for r in legacy]
    except Exception:  # noqa: BLE001
        return []


def _rules_of(automation: Automation) -> list[str]:
    try:
        rules = json.loads(automation.rules_json or "[]")
    except (ValueError, TypeError):
        return []
    return [str(r.get("text") or "").strip()
            for r in rules if isinstance(r, dict) and r.get("text")]


async def _narrate_phase1(
    db, *, automation: Automation, vspec: ValidatedSpecV2, job_id: str,
    thread, tool_turn_by_step: dict, partial: bool,
) -> Optional[dict]:
    """Run the narrator and persist what may precede the writes: the
    opening agent line + the per-item whys (in-place annotates). The
    result/thinks/draft/close land in phase 2, after the writes, so
    the thread never claims a change before it happened."""
    if thread is None:
        return None
    try:
        job = await db.get(BuildJob, job_id)
        from . import ledger as _ledger
        if job is None or _ledger.run_kind_of(job) not in ("scheduled",
                                                           "run_now"):
            return None
        # R31-37. This asked "does it write anything that is not a
        # draft?", so posting the brief to Slack made a reads-only
        # brief a change-making run and the founder's morning read
        # "CHANGED YOUR WEEK · 1 item". The question is whether the run
        # changed something the user OWNS — `narrator.vocabulary_for`
        # holds that judgement.
        from .narrator import vocabulary_for
        vocabulary = vocabulary_for(st.tool for st in vspec.write_steps)
        steps_record = []
        from app.services import automation_verbs as _verbs
        for st in vspec.steps:
            if st.mutates:
                d = _write_display(st)
                steps_record.append({
                    "step_ref": st.id,
                    "connector_name": _verbs.display_name(st.connector_id)
                    or st.connector_id,
                    "account_id": st.connector_id,
                    "tool_kind": "write",
                    "action": d["what"], "detail": d.get("target") or "",
                    "ok": True, "failure_reason": None, "items": [],
                    "write": d,
                })
                continue
            turn = tool_turn_by_step.get(st.id) or {}
            steps_record.append({
                "step_ref": st.id,
                "connector_name": _verbs.display_name(st.connector_id)
                or st.connector_id,
                "account_id": st.connector_id or "",
                "tool_kind": "read",
                "action": turn.get("action") or "",
                "detail": turn.get("detail") or "",
                "ok": bool(turn.get("ok", True)),
                "failure_reason": None if turn.get("ok", True)
                else (turn.get("detail") or "unreachable"),
                "items": [
                    {"id": it["id"], "title": it.get("title") or "",
                     "sub": it.get("sub") or "",
                     "msgs": it.get("msgs") or []}
                    for it in (turn.get("items") or [])
                ],
                "write": None,
            })
        record = {
            "automation": {"title": automation.name,
                           "mode": (vspec.raw or {}).get("mode") or "auto"},
            "run_kind": _ledger.run_kind_of(job),
            "vocabulary": vocabulary,
            "status": "partial" if partial else "completed",
            "rules": _rules_of(automation),
            "memory_facts": await _recall_facts(db, automation),
            "steps": steps_record,
        }
        from .narrator import narrate_run
        outcome = await narrate_run(record)
        drafts = outcome.get("turns") or []
        if outcome.get("problems"):
            logger.info("[automations] narration problems on %s: %s",
                        job_id[:8], outcome["problems"][:5])
        held: list[dict] = []
        # A resumed run already has its opening line — never repeat it.
        existing = await _ledger.run_turns(db, run_id=job_id)
        opened = any(t["kind"] == "agent" for t in existing)
        for d in drafts:
            kind = d.get("kind")
            if kind == "annotate":
                await _apply_annotate(db, automation, tool_turn_by_step, d)
            elif kind == "agent" and not opened:
                opened = True
                await _ledger.append_turn(
                    db, user_id=automation.user_id, thread=thread,
                    run_id=job_id, kind="agent",
                    payload={"text": d.get("text") or ""},
                )
            else:
                held.append(d)
        return {"held": held, "vocabulary": vocabulary}
    except Exception as e:  # noqa: BLE001 — narration must not kill the run
        logger.warning("[automations] narration phase 1 skipped: %s", e)
        return None


async def _apply_annotate(
    db, automation: Automation, tool_turn_by_step: dict, draft: dict,
) -> None:
    """Fill item whys / msg whys / rest into the persisted tool turn
    the annotate addresses (matched by minted item ids)."""
    from app.db.models import AutomationTurn
    from . import ledger as _ledger
    step_ref = draft.get("step_ref")
    turn = None
    for sid, t in tool_turn_by_step.items():
        if sid == step_ref or t.get("id") == step_ref:
            turn = t
            break
    if turn is None:
        return
    row = await db.get(AutomationTurn, turn["id"])
    if row is None:
        return
    body = json.loads(row.payload_json)
    by_id = {it.get("id"): it for it in body.get("items") or []}
    for ann in draft.get("items") or []:
        it = by_id.get(ann.get("id"))
        if it is None:
            continue
        if ann.get("why"):
            it["why"] = str(ann["why"])[:400]
        for m in ann.get("msgs") or []:
            idx = m.get("idx")
            msgs = it.get("msgs") or []
            if isinstance(idx, int) and 0 <= idx < len(msgs) and m.get("why"):
                msgs[idx]["why"] = str(m["why"])[:400]
    if draft.get("rest"):
        body["rest"] = str(draft["rest"])[:400]
    row.payload_json = json.dumps(body, default=str)
    await db.commit()
    turn.update(body)
    await _ledger._broadcast(automation.user_id, {
        "type": "automation.turn",
        "thread_id": row.thread_id,
        "run_id": row.run_id,
        "turn": _ledger._serialize_row(row),
    })


async def _narrate_phase2(
    db, *, automation: Automation, job_id: str, thread,
    narration: Optional[dict], writes_ok: bool,
) -> None:
    """Post-write narration: result (only when every write landed),
    thinks, the draft (unless the outbox already appended one), the
    closing line."""
    if thread is None or not narration:
        return
    try:
        from . import ledger as _ledger
        existing = await _ledger.run_turns(db, run_id=job_id)
        has_draft = any(t["kind"] == "draft" for t in existing)
        for d in narration.get("held") or []:
            kind = d.get("kind")
            if kind == "result":
                if not writes_ok:
                    continue
                await _ledger.append_turn(
                    db, user_id=automation.user_id, thread=thread,
                    run_id=job_id, kind="result",
                    payload={
                        "title": d.get("title") or "",
                        "vocabulary": d.get("vocabulary")
                        or narration.get("vocabulary") or "brief",
                        "groups": d.get("groups") or [],
                    },
                )
            elif kind == "think":
                await _ledger.append_turn(
                    db, user_id=automation.user_id, thread=thread,
                    run_id=job_id, kind="think",
                    payload={"text": d.get("text") or ""},
                )
            elif kind == "draft":
                if has_draft:
                    continue
                await _ledger.append_turn(
                    db, user_id=automation.user_id, thread=thread,
                    run_id=job_id, kind="draft",
                    payload={
                        "text": d.get("text") or "",
                        "target": {
                            "account_id": d.get("target_account_id") or "",
                            "ref": d.get("target_ref"),
                        },
                        "sent_at": None,
                    },
                )
                has_draft = True
            elif kind == "agent":
                if not writes_ok:
                    continue
                await _ledger.append_turn(
                    db, user_id=automation.user_id, thread=thread,
                    run_id=job_id, kind="agent",
                    payload={"text": d.get("text") or ""},
                )
    except Exception as e:  # noqa: BLE001
        logger.warning("[automations] narration phase 2 skipped: %s", e)
