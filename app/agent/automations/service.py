"""Automation lifecycle service — shared by the skill tools and the
HTTP API so chat-built and API-built automations are byte-identical
(the same economy as routines' skill delegating to its API functions).

Every function takes an open AsyncSession and commits itself at the
end of a complete, consistent unit — same convention as the routines
API. Nothing here runs unless `settings.automations_enabled` is true;
the callers gate.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    Automation, AutomationBinding, AutomationEvent, BuildJob,
)
from . import compiler, registry as reg
from .spec import SpecError, ValidatedSpec, validate_spec

logger = logging.getLogger(__name__)


class AutomationNotFound(LookupError):
    pass


async def _load_owned(
    db: AsyncSession, automation_id: str, user_id: str,
) -> Automation:
    row = (await db.execute(
        select(Automation)
        .where(Automation.id == automation_id)
        .where(Automation.user_id == user_id)
    )).scalar_one_or_none()
    if row is None:
        raise AutomationNotFound(automation_id)
    return row


async def create_automation(
    db: AsyncSession,
    *,
    user_id: str,
    spec: dict,
    template_slug: Optional[str] = None,
    domain: Optional[str] = None,
) -> tuple[Automation, ValidatedSpec]:
    """Validate + persist a spec as a DRAFT automation with compiled
    (but disabled) bindings. Raises SpecError with every problem."""
    from app.agent.automations.memory_notes import normalize_domain

    capability = await reg.fetch_registry(user_id)
    vspec = validate_spec(spec, capability)

    automation = Automation(
        user_id=user_id,
        name=vspec.name,
        description=vspec.raw.get("description"),
        status="draft",
        spec_json=json.dumps(vspec.raw, sort_keys=True),
        trigger_mode=vspec.trigger_mode,
        connector_id=vspec.trigger_connector_id or vspec.action_connector_id,
        template_slug=template_slug,
        # An unrecognizable domain becomes NULL (no facts filed), never
        # an error — domain is metadata, not a validity condition.
        domain=normalize_domain(domain),
    )
    db.add(automation)
    await db.flush()
    await compiler.compile_bindings(db, automation, vspec)
    await db.commit()
    logger.info("[automations] created id=%s mode=%s user=%s",
                automation.id, vspec.trigger_mode, user_id[:8])
    return automation, vspec


async def update_automation(
    db: AsyncSession,
    *,
    automation_id: str,
    user_id: str,
    spec: dict,
) -> tuple[Automation, ValidatedSpec]:
    """Replace the spec and re-compile bindings. An armed automation
    stays armed only if the new spec passes the same arm checks —
    otherwise it drops to draft (never silently keep firing an old
    shape)."""
    automation = await _load_owned(db, automation_id, user_id)
    capability = await reg.fetch_registry(user_id)
    vspec = validate_spec(spec, capability)

    was_armed = automation.status == "armed"
    automation.name = vspec.name
    automation.description = vspec.raw.get("description")
    automation.spec_json = json.dumps(vspec.raw, sort_keys=True)
    automation.trigger_mode = vspec.trigger_mode
    automation.connector_id = (
        vspec.trigger_connector_id or vspec.action_connector_id
    )
    automation.status = "draft"
    automation.paused_reason = None
    await compiler.compile_bindings(db, automation, vspec)
    await db.commit()

    if was_armed:
        try:
            await arm_automation(
                db, automation_id=automation.id, user_id=user_id,
            )
        except (compiler.CompileError, SpecError) as e:
            logger.info(
                "[automations] update left %s in draft (re-arm failed: %s)",
                automation.id, e,
            )
    await db.refresh(automation)
    return automation, vspec


async def arm_automation(
    db: AsyncSession, *, automation_id: str, user_id: str,
) -> Automation:
    """draft/paused → armed. Verifies the grant against the platform
    (fail closed), snapshots the pinned target into the spec for
    template rendering, enables the primitives, provisions push."""
    automation = await _load_owned(db, automation_id, user_id)
    vspec = await parse_spec_live(automation)

    from .spec_v2 import ValidatedSpecV2
    if isinstance(vspec, ValidatedSpecV2):
        # Round 28: every write step's grant, fail closed; snapshot
        # each pinned target into ITS step for template rendering.
        grants = await compiler.verify_grants_for_arm_v2(automation, vspec)
        if grants:
            raw = dict(vspec.raw)
            steps = [dict(s) for s in raw.get("steps") or []]
            for s in steps:
                g = grants.get(s.get("id"))
                if g is not None:
                    s["grant_target"] = g.get("target") or {}
            raw["steps"] = steps
            automation.spec_json = json.dumps(raw, sort_keys=True)
    else:
        grant = await compiler.verify_grant_for_arm(automation, vspec)
        if grant is not None:
            raw = dict(vspec.raw)
            action = dict(raw.get("action") or {})
            action["grant_target"] = grant.get("target") or {}
            raw["action"] = action
            automation.spec_json = json.dumps(raw, sort_keys=True)

    await compiler.set_bindings_active(db, automation, True)
    automation.status = "armed"
    automation.paused_reason = None
    automation.consecutive_failures = 0
    automation.error_notice_at = None
    await db.commit()

    has_push = vspec.trigger_mode == "push" or (
        isinstance(vspec, ValidatedSpecV2)
        and any(s.mode == "push" for s in vspec.sources)
    )
    if has_push:
        await _provision_push(db, automation)
    logger.info("[automations] armed id=%s user=%s",
                automation.id, user_id[:8])
    return automation


async def pause_automation(
    db: AsyncSession, *, automation_id: str, user_id: str,
    reason: str = "user",
) -> Automation:
    automation = await _load_owned(db, automation_id, user_id)
    await compiler.set_bindings_active(db, automation, False)
    automation.status = "paused" if reason == "user" else "error"
    automation.paused_reason = reason
    await db.commit()
    return automation


async def resume_automation(
    db: AsyncSession, *, automation_id: str, user_id: str,
) -> Automation:
    """paused/error → armed, through the same checks as arm (a grant
    revoked while paused must block the resume)."""
    return await arm_automation(db, automation_id=automation_id, user_id=user_id)


async def delete_automation(
    db: AsyncSession, *, automation_id: str, user_id: str,
) -> None:
    automation = await _load_owned(db, automation_id, user_id)
    await compiler.teardown_bindings(db, automation)
    from . import memory as engine_memory
    await engine_memory.delete_for_automation(
        db, user_id=user_id, automation_id=automation_id,
    )
    await db.delete(automation)   # events/outbox cascade via FK
    await db.commit()
    logger.info("[automations] deleted id=%s user=%s",
                automation_id, user_id[:8])


async def test_run(
    db: AsyncSession, *, automation_id: str, user_id: str,
) -> dict:
    """One synthetic fire: builds a sample event from the spec (or
    polls once for a real one), runs the full evaluate→prepare path,
    and STOPS at the staged outbox row — the write goes out only after
    the normal undo window, exactly like a real fire, so a test run is
    a real run with a synthetic trigger."""
    from . import executor
    automation = await _load_owned(db, automation_id, user_id)
    vspec = await parse_spec_live(automation)
    from .spec_v2 import ValidatedSpecV2
    if isinstance(vspec, ValidatedSpecV2):
        from . import executor_v2
        return await executor_v2.execute_test_run_v2(db, automation, vspec)
    return await executor.execute_test_run(db, automation, vspec)


# ── Read shapes (API + tools + Activity page) ─────────────────────────


def _parse_spec(automation: Automation) -> ValidatedSpec:
    """Re-validate the persisted spec against an offline registry view.

    Shape-only: good for payload rendering and lifecycle checks. The
    permissive event entry carries no source_tool/items_path, so FIRE
    paths must use `parse_spec_live` — a poll parsed from this snapshot
    would silently observe nothing."""
    raw = json.loads(automation.spec_json)
    if raw.get("version") == 2:
        return validate_spec(raw, _permissive_registry_v2(raw))
    trig = raw.get("trigger") or {}
    act = raw.get("action") or {}
    dk = str(raw.get("dedupe_key") or "")
    dk_field = dk[len("event."):] if dk.startswith("event.") else ""
    permissive: dict[str, dict] = {}
    for cid in {trig.get("connector_id"), act.get("connector_id")} - {None}:
        permissive[cid] = {
            "connector_id": cid,
            "push": True, "poll": True, "floor_s": 300,
            "events": [
                {"key": trig.get("event") or "",
                 "dedupe_field": dk_field,
                 "fields": {dk_field: dk_field} if dk_field else {}},
            ] if trig.get("event") else [],
            "scopes_write_by_action": (
                {act.get("tool"): []}
                if act.get("grant_id") and act.get("connector_id") == cid
                else {}
            ),
        }
    return validate_spec(raw, permissive)


def _permissive_registry_v2(raw: dict) -> dict:
    """Offline registry snapshot for a persisted v2 spec — every
    referenced connector, each source's event reconstructed from its
    own dedupe key, each granted step's tool marked as a write. Shape-
    only, like the v1 permissive path: fire paths use the live
    registry."""
    permissive: dict[str, dict] = {}

    def _entry(cid: str) -> dict:
        return permissive.setdefault(cid, {
            "connector_id": cid,
            "push": True, "poll": True, "floor_s": 300,
            "events": [],
            "scopes_write_by_action": {},
        })

    for src in (raw.get("trigger") or {}).get("sources") or []:
        if not isinstance(src, dict):
            continue
        cid = src.get("connector_id")
        if not cid:
            continue
        entry = _entry(cid)
        dk = str(src.get("dedupe_key") or "")
        dk_field = dk[len("event."):] if dk.startswith("event.") else ""
        if src.get("event"):
            entry["events"].append({
                "key": src["event"],
                "dedupe_field": dk_field,
                "fields": {dk_field: dk_field} if dk_field else {},
            })
    for step in raw.get("steps") or []:
        if not isinstance(step, dict):
            continue
        cid = step.get("connector_id")
        if not cid:
            continue
        entry = _entry(cid)
        if step.get("grant_id") and step.get("tool"):
            entry["scopes_write_by_action"][step["tool"]] = []
    return permissive


async def parse_spec_live(automation: Automation) -> ValidatedSpec:
    """The FIRE-path parse: validate against the live capability
    registry (5-min cached; a network blip serves the cache) so
    `event_spec` carries the real source_tool / items_path / fields.
    Falls back to the offline shape when the registry is empty — the
    poll then observes nothing and the health loop records the miss
    rather than inventing capabilities."""
    capability = await reg.fetch_registry(automation.user_id)
    raw = json.loads(automation.spec_json)
    if capability:
        try:
            return validate_spec(raw, capability)
        except SpecError as e:
            logger.warning(
                "[automations] persisted spec no longer validates against "
                "the live registry for %s: %s", automation.id, e,
            )
    return _parse_spec(automation)


def automation_payload(a: Automation) -> dict:
    raw = json.loads(a.spec_json)
    if raw.get("version") == 2:
        # v2 shape: steps[] (system-written grant_target withheld, as
        # v1 withholds it from action). `action` stays populated with
        # the first write step so list surfaces keyed on it keep
        # rendering.
        steps = [
            {k: v for k, v in s.items() if k != "grant_target"}
            for s in (raw.get("steps") or []) if isinstance(s, dict)
        ]
        first_write = next(
            (s for s in steps if s.get("grant_id")), steps[0] if steps else {},
        )
        return {
            "id": a.id,
            "name": a.name,
            "description": a.description,
            "status": a.status,
            "paused_reason": a.paused_reason,
            "version": 2,
            "trigger": raw.get("trigger") or {},
            "steps": steps,
            "variables": raw.get("variables") or {},
            "action": {
                "connector_id": first_write.get("connector_id"),
                "tool": first_write.get("tool"),
                "params_template": first_write.get("params") or {},
            },
            "mode": raw.get("mode"),
            "template_slug": a.template_slug,
            "health": {
                "consecutive_failures": a.consecutive_failures,
                "last_run_at": (a.last_run_at.isoformat() + "Z")
                if a.last_run_at else None,
                "last_status": a.last_status,
                "last_error": a.last_error,
            },
            "created_at": a.created_at.isoformat() + "Z",
            "updated_at": a.updated_at.isoformat() + "Z",
        }
    return {
        "id": a.id,
        "name": a.name,
        "description": a.description,
        "status": a.status,
        "paused_reason": a.paused_reason,
        "trigger": raw.get("trigger") or {},
        "action": {
            k: v for k, v in (raw.get("action") or {}).items()
            if k != "grant_target"
        },
        "mode": raw.get("mode"),
        "template_slug": a.template_slug,
        "domain": a.domain,
        "health": {
            "consecutive_failures": a.consecutive_failures,
            "last_run_at": (a.last_run_at.isoformat() + "Z") if a.last_run_at else None,
            "last_status": a.last_status,
            "last_error": a.last_error,
        },
        "created_at": a.created_at.isoformat() + "Z",
        "updated_at": a.updated_at.isoformat() + "Z",
    }


async def list_automations(db: AsyncSession, user_id: str) -> list[dict]:
    rows = (await db.execute(
        select(Automation)
        .where(Automation.user_id == user_id)
        .order_by(Automation.created_at.desc())
    )).scalars().all()
    return [automation_payload(a) for a in rows]


async def list_runs(
    db: AsyncSession, user_id: str,
    *, automation_id: Optional[str] = None, limit: int = 50,
) -> list[dict]:
    """Runs ARE BuildJobs (job_type='automation_run'). Names come from
    a single automations read, not a per-row join."""
    q = (
        select(BuildJob)
        .where(BuildJob.user_id == user_id)
        .where(BuildJob.job_type == "automation_run")
        .order_by(BuildJob.created_at.desc())
        .limit(limit)
    )
    if automation_id:
        q = q.where(BuildJob.source_id == automation_id)
    jobs = (await db.execute(q)).scalars().all()

    names: dict[str, str] = {}
    if jobs:
        autos = (await db.execute(
            select(Automation.id, Automation.name)
            .where(Automation.user_id == user_id)
        )).all()
        names = {a_id: a_name for a_id, a_name in autos}

    out = []
    for j in jobs:
        steps = []
        try:
            for s in (json.loads(j.steps_json) if j.steps_json else []):
                steps.append({
                    "id": s.get("id"),
                    "label": s.get("label"),
                    "status": s.get("status"),
                    "duration_ms": s.get("duration_ms"),
                })
        except (ValueError, TypeError):
            pass
        out.append({
            "id": j.id,
            "automation_id": j.source_id,
            "automation_name": names.get(j.source_id or "", None),
            "status": j.status,
            "outcome": j.outcome,
            "fire_instant": (j.fire_instant.isoformat() + "Z") if j.fire_instant else None,
            "started_at": j.created_at.isoformat() + "Z",
            "finished_at": (j.completed_at.isoformat() + "Z") if j.completed_at else None,
            "steps": steps,
            "error_class": j.error_class,
            "user_message": j.user_message,
        })
    return out


async def find_by_binding_target(
    db: AsyncSession, target_id: str,
) -> Optional[Automation]:
    """Routine/Trigger id → its automation (handler entry path)."""
    b = (await db.execute(
        select(AutomationBinding)
        .where(AutomationBinding.target_id == target_id)
    )).scalar_one_or_none()
    if b is None:
        return None
    return await db.get(Automation, b.automation_id)


async def _provision_push(db: AsyncSession, automation: Automation) -> None:
    """Arm the Gmail watch for a push automation, via the same platform
    RPC the triggers API uses. Best-effort here: the 6-hour platform
    refresh cron re-arms lapsed watches, and the trigger row already
    gates events on enabled=true."""
    try:
        from app.api.triggers import _provision_email_watch
        rows = (await db.execute(
            select(AutomationBinding)
            .where(AutomationBinding.automation_id == automation.id)
            .where(AutomationBinding.kind == "trigger")
        )).scalars().all()
        for b in rows:
            await _provision_email_watch(b.target_id)
    except Exception as e:  # noqa: BLE001 — provisioning is retried by cron
        logger.warning("[automations] push provisioning failed for %s: %s",
                       automation.id, e)
