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
    Automation, AutomationEvent, AutomationOutbox,
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
                      variables: dict) -> bool:
    """v1 filter semantics per source, with {{var.*}} rendered in the
    needles so templates can parameterize filters."""
    ctx = {"var": variables or {}}
    for fld, needles in (source.filter_rules or {}).items():
        if not needles:
            continue
        if not isinstance(needles, list):
            needles = [needles]
        value = str(payload.get(fld) or "").lower()
        rendered = [str(render_value(str(n), ctx)).lower() for n in needles]
        if not any(n in value for n in rendered if n):
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
    if fmt:
        for item in items[:limit]:
            item_fields = {
                name: resolve_path(item, path)
                for name, path in (collect.get("fields") or {}).items()
            }
            lines.append(str(render_value(
                fmt, {"item": item_fields, "var": variables or {}},
            )))
    text = (collect.get("join") or "\n").join(lines)
    if count == 0:
        text = collect.get("empty_text") or ""
    return {"ok": True, "text": text, "count": count}


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
) -> str:
    """steps (reads) → stage writes → flush → record + memory."""
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

    partial = False
    for st in vspec.steps:
        if st.mutates:
            break
        try:
            ctx["steps"][st.id] = await _execute_read_step(automation, st, ctx)
        except Exception as e:  # noqa: BLE001 — transport/shape errors
            if st.on_error == "skip":
                logger.info("[automations] step %s skipped (%s) on %s",
                            st.id, e, automation.id)
                ctx["steps"][st.id] = _skipped_result(st)
                partial = True
            else:
                await _finalize_job(
                    db, job_id, status="failed", outcome="step_failed",
                    error_class="tool_error",
                    user_message=f"Step {st.id!r} failed: {str(e)[:200]}",
                )
                await _record_health(db, automation.id, ok=False,
                                     error=str(e)[:500])
                return "failed"
        step_result = ctx["steps"].get(st.id) or {}
        await _advance_v2(
            db, job_id, vspec, st.id,
            count=step_result.get("count")
            if isinstance(step_result.get("count"), int) else None,
        )

    if partial:
        # The aggregate finalizer reads this to report `partial`
        # honestly when the writes themselves succeed.
        await merge_job_config(db, job_id, steps_partial=True)

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
        ))
    db.add_all(rows)
    try:
        await db.flush()
    except IntegrityError:
        await db.rollback()
        logger.info("[automations] outbox idempotency hit %s", idem_prefix)
        return "run"
    await db.commit()
    for st in vspec.write_steps:
        await _advance_v2(db, job_id, vspec, st.id)

    if stage_only:
        return "staged"

    from .outbox import flush_row_when_due
    statuses = []
    for row in rows:
        statuses.append(await flush_row_when_due(db, row.id))

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

    if not _passes_filter_v2(source, payload, vspec.variables):
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

    return await _run_steps(db, automation, vspec, job.id, payload, source,
                            idem_prefix=f"evt:{event.id}")


async def run_schedule_fire_v2(
    db,
    automation: Automation,
    vspec: ValidatedSpecV2,
    source: ValidatedSource,
    fire_key: str,
) -> str:
    async def _inner() -> str:
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
        await on_run_created(db, job=job, automation=automation)
        await _advance_v2(db, job.id, vspec, "evaluate")
        return await _run_steps(db, automation, vspec, job.id, {}, source,
                                idem_prefix=f"fire:{fire_key}")

    try:
        return await asyncio.wait_for(_inner(), timeout=AUTOMATION_RUN_CAP_S)
    except asyncio.TimeoutError:
        await _record_health(db, automation.id, ok=False,
                             error="run cap exceeded")
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
        await _record_health(db, automation.id, ok=True, error=None)
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
    status = await _run_steps(
        db, automation, vspec, job.id, sample, source,
        idem_prefix=f"test:{job.id}", stage_only=True,
    )
    return {"job_id": job.id, "status": status, "sample_event": sample}
