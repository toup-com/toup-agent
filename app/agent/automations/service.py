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
    *, include_deleted: bool = False,
) -> Automation:
    q = (
        select(Automation)
        .where(Automation.id == automation_id)
        .where(Automation.user_id == user_id)
    )
    if not include_deleted:
        # R30 §4.8: a soft-deleted automation is invisible to every
        # list/read/edit path; only the sweep and the archived-thread
        # reader opt in.
        q = q.where(Automation.deleted_at.is_(None))
    row = (await db.execute(q)).scalar_one_or_none()
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
    template_mode: bool = False,
) -> tuple[Automation, ValidatedSpec]:
    """Validate + persist a spec as a DRAFT automation with compiled
    (but disabled) bindings. Raises SpecError with every problem.
    `template_mode` (R30 from-template): write steps may lack grants —
    the draft cannot ARM until the grant conversation completes
    (arm_automation verifies fail-closed), so nothing weakens."""
    from app.agent.automations.memory_notes import normalize_domain

    capability = await reg.fetch_registry(user_id)
    vspec = validate_spec(spec, capability, template_mode=template_mode)

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

    routine_ids = await compiler.set_bindings_active(db, automation, True)
    automation.status = "armed"
    automation.paused_reason = None
    automation.consecutive_failures = 0
    automation.error_notice_at = None
    await db.commit()
    # AFTER the commit — a pre-commit nudge reads the old (disabled)
    # row and unregisters the routine this arm just enabled (R28-D).
    await compiler.nudge_routines(routine_ids)

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
    routine_ids = await compiler.set_bindings_active(db, automation, False)
    automation.status = "paused" if reason == "user" else "error"
    automation.paused_reason = reason
    await db.commit()
    await compiler.nudge_routines(routine_ids)
    return automation


async def resume_automation(
    db: AsyncSession, *, automation_id: str, user_id: str,
) -> Automation:
    """paused/error → armed, through the same checks as arm (a grant
    revoked while paused must block the resume)."""
    return await arm_automation(db, automation_id=automation_id, user_id=user_id)


async def delete_automation(
    db: AsyncSession, *, automation_id: str, user_id: str,
    undo: bool = False,
) -> None:
    """R30 §4.8. Two shapes: `undo=True` (allowed only before the first
    run starts) hard-deletes with no ledger residue; the normal delete
    is SOFT — schedule disarmed, thread archived (openable 30 days),
    drafts untouched, memory kept 30 days — and the sweep purges."""
    automation = await _load_owned(db, automation_id, user_id)
    if undo:
        from app.db.models import BuildJob
        has_run = automation.last_run_at is not None or (
            (await db.execute(
                select(BuildJob.id)
                .where(BuildJob.source_id == automation_id)
                .where(BuildJob.job_type == "automation_run")
                .limit(1)
            )).scalar_one_or_none() is not None
        )
        if has_run:
            raise MembershipError(
                "undo_window_closed",
                "the first run already started — delete keeps the record",
            )
        await _hard_delete(db, automation, user_id)
        return
    await compiler.teardown_bindings(db, automation)
    automation.deleted_at = datetime.utcnow()
    automation.status = "paused"
    automation.paused_reason = "user"
    from app.db.models import AutomationThread
    thread = (await db.execute(
        select(AutomationThread)
        .where(AutomationThread.automation_id == automation_id)
    )).scalar_one_or_none()
    if thread is not None and thread.archived_at is None:
        thread.archived_at = datetime.utcnow()

    # R31-09. Soft is about the RECORD — the thread and its facts keep
    # their 30 days so the main-chat card can still open them. Nothing
    # that can FIRE or NOTIFY may survive, and two things did:
    #
    #  - the migrated routine pair. §4.11a leaves the original routine
    #    disabled and stamped `migrated_to`; disabled is not deleted,
    #    and a routine row that outlives the automation it became is one
    #    re-enable away from a deleted automation running again.
    #  - pending notifications. A queued `automation_run` card or push
    #    for a run that finished after the delete is a message from an
    #    automation the user just removed.
    from sqlalchemy import delete as sa_delete
    from app.db.models import AutomationNotification, Routine
    try:
        routines = (await db.execute(
            select(Routine).where(Routine.user_id == user_id)
        )).scalars().all()
        for r in routines:
            cfg = r.config_json if isinstance(r.config_json, dict) else {}
            if cfg.get("migrated_to") == automation_id \
                    or cfg.get("superseded_by") == automation_id:
                await db.delete(r)
    except Exception as e:  # noqa: BLE001 — a delete never half-fails
        logger.warning("[automations] routine-pair cleanup skipped %s: %s",
                       automation_id, e)
    try:
        await db.execute(
            sa_delete(AutomationNotification)
            .where(AutomationNotification.automation_id == automation_id)
            .where(AutomationNotification.status.in_(
                ("running", "queued", "waiting_on_user")))
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("[automations] notification cleanup skipped %s: %s",
                       automation_id, e)

    await db.commit()
    # Every list, summary, thread and workflow reader already excludes
    # `deleted_at`; the frame is what makes another device's home list
    # drop the card without a reload (§4.6).
    try:
        from . import ledger as _ledger
        await _ledger.emit_updated(
            db, user_id, automation_id=automation_id,
        )
    except Exception as e:  # noqa: BLE001
        logger.debug("[automations] delete frame skipped: %s", e)
    logger.info("[automations] soft-deleted id=%s user=%s",
                automation_id, user_id[:8])


async def _hard_delete(
    db: AsyncSession, automation: Automation, user_id: str,
) -> None:
    automation_id = automation.id
    await compiler.teardown_bindings(db, automation)
    from . import memory as engine_memory
    await engine_memory.delete_for_automation(
        db, user_id=user_id, automation_id=automation_id,
    )
    # Curated facts (R29): explicit, not cascade-only — a sqlite tenant
    # (the live harness, dev) doesn't enforce FK cascades. The brain
    # projection is deliberately NOT unwound: facts about a life
    # outlive the tool that learned them (CONTRACTS-R29.md §4).
    from sqlalchemy import delete as sa_delete
    from app.db.models import AutomationFact, MemoryFact
    await db.execute(
        sa_delete(AutomationFact)
        .where(AutomationFact.automation_id == automation_id)
        .where(AutomationFact.user_id == user_id)
    )
    await db.execute(
        sa_delete(MemoryFact)
        .where(MemoryFact.scope == automation_id)
        .where(MemoryFact.user_id == user_id)
    )
    await db.delete(automation)   # events/outbox/threads cascade via FK
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


class MembershipError(ValueError):
    """A connector/schedule/mode edit the spec cannot absorb — carries
    the stable machine `code` the routes serve as HTTP 409."""

    def __init__(self, code: str, message: str = ""):
        super().__init__(message or code)
        self.code = code


async def mark_outcome_seen(
    db: AsyncSession, *, automation_id: str, user_id: str,
) -> bool:
    """`POST /{id}/seen` — stamp the current outcome as seen. A later
    terminal stamp makes the row unseen again; there is nothing to
    race (the stamp is monotonic now())."""
    automation = await _load_owned(db, automation_id, user_id)
    automation.outcome_seen_at = datetime.utcnow()
    await db.commit()
    return True


def _spec_dict(automation: Automation) -> dict:
    try:
        raw = json.loads(automation.spec_json)
    except (ValueError, TypeError):
        raw = {}
    return raw if isinstance(raw, dict) else {}


async def set_schedule(
    db: AsyncSession, *, automation_id: str, user_id: str, schedule: dict,
) -> tuple[Automation, ValidatedSpec]:
    """Replace the automation's ONE schedule (v2 schedule source / v1
    schedule trigger) and rerun the full update path — revalidation,
    recompile, re-arm-if-was-armed. `schedule` must be exactly one of
    {cron_local} / {at} / {every_s}; the spec's own validation owns the
    value rules."""
    keys = set(schedule or {}) & {"cron_local", "at", "every_s"}
    if len(schedule or {}) != 1 or len(keys) != 1:
        raise MembershipError(
            "bad_schedule",
            "body must be exactly one of cron_local / at / every_s",
        )
    automation = await _load_owned(db, automation_id, user_id)
    raw = _spec_dict(automation)
    placed = False
    if raw.get("version") == 2:
        for src in (raw.get("trigger") or {}).get("sources") or []:
            if isinstance(src, dict) and src.get("mode") == "schedule":
                src["schedule"] = dict(schedule)
                placed = True
                break
    else:
        trig = raw.get("trigger") or {}
        if trig.get("mode") == "schedule" or "schedule" in trig:
            trig["schedule"] = dict(schedule)
            raw["trigger"] = trig
            placed = True
    if not placed:
        raise MembershipError(
            "no_schedule", "this automation has no schedule to edit",
        )
    return await update_automation(
        db, automation_id=automation_id, user_id=user_id, spec=raw,
    )


async def set_mode(
    db: AsyncSession, *, automation_id: str, user_id: str, mode: str,
) -> tuple[Automation, ValidatedSpec]:
    """Flip spec-level auto/confirm through the full update path. The
    GRANT rows' mode is the enforcer and is flipped platform-side by
    the user-JWT proxy route — never from here (an agent RPC must not
    be able to widen consent)."""
    if mode not in ("auto", "confirm"):
        raise MembershipError("bad_mode", "mode must be auto or confirm")
    automation = await _load_owned(db, automation_id, user_id)
    raw = _spec_dict(automation)
    raw["mode"] = mode
    return await update_automation(
        db, automation_id=automation_id, user_id=user_id, spec=raw,
    )


def _connector_membership(raw: dict, connector_id: str) -> dict:
    steps = [s for s in (raw.get("steps") or []) if isinstance(s, dict)]
    sources = [
        s for s in ((raw.get("trigger") or {}).get("sources") or [])
        if isinstance(s, dict)
    ]
    return {
        "read_steps": [s for s in steps
                       if s.get("connector_id") == connector_id
                       and not s.get("grant_id")],
        "write_steps": [s for s in steps
                        if s.get("connector_id") == connector_id
                        and s.get("grant_id")],
        "sources": [s for s in sources
                    if s.get("connector_id") == connector_id],
    }


async def add_connector(
    db: AsyncSession, *, automation_id: str, user_id: str, connector_id: str,
) -> tuple[Automation, ValidatedSpec]:
    """Re-add a connector's READ presence from the automation's
    template skeleton (CONTRACTS-R29.md §3.2): its read step(s) and
    poll/push source(s), inserted ahead of the writes. Writes never
    ride this path — a write needs its own grant conversation."""
    automation = await _load_owned(db, automation_id, user_id)
    raw = _spec_dict(automation)
    if raw.get("version") != 2:
        raise MembershipError(
            "not_supported_v1",
            "connector membership edits need a v2 automation",
        )
    if any(_connector_membership(raw, connector_id).values()):
        raise MembershipError("already_member", "connector already present")
    if not automation.template_slug:
        raise MembershipError(
            "no_template_step", "this automation has no template to add from",
        )
    template = next(
        (t for t in await reg.fetch_templates(user_id)
         if t.get("slug") == automation.template_slug),
        None,
    )
    tspec = (template or {}).get("spec") or {}
    donor = _connector_membership(tspec, connector_id)
    if not donor["read_steps"] and not donor["sources"]:
        raise MembershipError(
            "no_template_step",
            "the template has nothing for this connector",
        )
    steps = [s for s in (raw.get("steps") or []) if isinstance(s, dict)]
    first_write = next(
        (i for i, s in enumerate(steps) if s.get("grant_id")), len(steps),
    )
    existing_ids = {s.get("id") for s in steps}
    new_steps = [dict(s) for s in donor["read_steps"]
                 if s.get("id") not in existing_ids]
    raw["steps"] = steps[:first_write] + new_steps + steps[first_write:]
    if donor["sources"]:
        trig = dict(raw.get("trigger") or {})
        sources = [s for s in (trig.get("sources") or [])
                   if isinstance(s, dict)]
        src_ids = {s.get("id") for s in sources}
        sources.extend(dict(s) for s in donor["sources"]
                       if s.get("id") not in src_ids)
        trig["sources"] = sources
        raw["trigger"] = trig
    return await update_automation(
        db, automation_id=automation_id, user_id=user_id, spec=raw,
    )


async def remove_connector(
    db: AsyncSession, *, automation_id: str, user_id: str, connector_id: str,
) -> tuple[Automation, ValidatedSpec]:
    """Remove a connector's read steps and sources. Refused when the
    connector backs a write step or removal would leave the spec
    unable to fire (no sources / no read material the spec needs)."""
    automation = await _load_owned(db, automation_id, user_id)
    raw = _spec_dict(automation)
    if raw.get("version") != 2:
        raise MembershipError(
            "connector_required",
            "a v1 automation needs both of its connectors",
        )
    member = _connector_membership(raw, connector_id)
    if member["write_steps"]:
        raise MembershipError(
            "connector_required", "this connector performs the write",
        )
    if not member["read_steps"] and not member["sources"]:
        raise MembershipError("not_member", "connector not present")
    steps = [
        s for s in (raw.get("steps") or [])
        if not (isinstance(s, dict)
                and s.get("connector_id") == connector_id
                and not s.get("grant_id"))
    ]
    trig = dict(raw.get("trigger") or {})
    sources = [
        s for s in (trig.get("sources") or [])
        if not (isinstance(s, dict)
                and s.get("connector_id") == connector_id)
    ]
    if not sources:
        raise MembershipError(
            "connector_required",
            "removing this connector would leave no trigger",
        )
    trig["sources"] = sources
    raw["trigger"] = trig
    raw["steps"] = steps
    try:
        return await update_automation(
            db, automation_id=automation_id, user_id=user_id, spec=raw,
        )
    except SpecError as e:
        raise MembershipError(
            "connector_required",
            f"removal would leave the automation invalid: {e}",
        ) from e


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


def _spec_connectors(raw: dict) -> list[str]:
    """Ordered, deduped connector ids: trigger/sources first, then
    reads, then writes — the card's mark row (R29 §2)."""
    out: list[str] = []

    def _add(cid) -> None:
        if isinstance(cid, str) and cid and cid not in out:
            out.append(cid)

    trig = raw.get("trigger") or {}
    if raw.get("version") == 2:
        for src in trig.get("sources") or []:
            if isinstance(src, dict):
                _add(src.get("connector_id"))
        steps = [s for s in (raw.get("steps") or []) if isinstance(s, dict)]
        for s in steps:
            if not s.get("grant_id"):
                _add(s.get("connector_id"))
        for s in steps:
            if s.get("grant_id"):
                _add(s.get("connector_id"))
    else:
        _add(trig.get("connector_id"))
        _add((raw.get("action") or {}).get("connector_id"))
    return out


def _payload_extras(a: Automation, raw: dict) -> dict:
    """The R29 list-payload additions, shared by both spec versions."""
    from app.services import automation_verbs as verbs

    last_outcome = None
    if a.last_outcome_at is not None:
        last_outcome = {
            "outcome": a.last_outcome,
            "sentence": a.last_outcome_text or "",
            "at": a.last_outcome_at.isoformat() + "Z",
            "tone": verbs.tone_for(a.last_outcome),
        }
    unseen = bool(
        a.last_outcome_at is not None
        and (a.outcome_seen_at is None
             or a.outcome_seen_at < a.last_outcome_at)
    )
    if a.paused_reason == "grant_revoked":
        attention = "grant_revoked"
    elif a.status == "error":
        attention = "auto_paused"
    else:
        attention = None
    return {
        "connectors": _spec_connectors(raw),
        "last_outcome": last_outcome,
        "unseen": unseen,
        "schedule_human": verbs.schedule_human(raw),
        "rule_text": verbs.rule_sentence(raw),
        "attention": attention,
    }


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
            "domain": a.domain,
            "health": {
                "consecutive_failures": a.consecutive_failures,
                "last_run_at": (a.last_run_at.isoformat() + "Z")
                if a.last_run_at else None,
                "last_status": a.last_status,
                "last_error": a.last_error,
            },
            "created_at": a.created_at.isoformat() + "Z",
            "updated_at": a.updated_at.isoformat() + "Z",
            **_payload_extras(a, raw),
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
        **_payload_extras(a, raw),
    }


async def list_automations(db: AsyncSession, user_id: str) -> list[dict]:
    rows = (await db.execute(
        select(Automation)
        .where(Automation.user_id == user_id)
        .where(Automation.deleted_at.is_(None))
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
    tools_by_auto: dict[str, dict] = {}
    if jobs:
        autos = (await db.execute(
            select(Automation.id, Automation.name, Automation.spec_json)
            .where(Automation.user_id == user_id)
        )).all()
        for a_id, a_name, a_spec in autos:
            names[a_id] = a_name
            try:
                raw = json.loads(a_spec or "{}")
            except (ValueError, TypeError):
                raw = {}
            by_id: dict[str, tuple] = {}
            if raw.get("version") == 2:
                for s in raw.get("steps") or []:
                    if isinstance(s, dict) and s.get("id"):
                        by_id[s["id"]] = (s.get("tool"), s.get("connector_id"))
            else:
                action = raw.get("action") or {}
                by_id["write"] = (action.get("tool"),
                                  action.get("connector_id"))
            tools_by_auto[a_id] = by_id

    from app.services import automation_verbs as verbs

    out = []
    for j in jobs:
        by_id = tools_by_auto.get(j.source_id or "", {})
        steps = []
        try:
            for s in (json.loads(j.steps_json) if j.steps_json else []):
                sid = s.get("id")
                status = s.get("status")
                count = s.get("count") if isinstance(s.get("count"), int) \
                    else None
                tool, connector = by_id.get(sid, (None, None))
                # One render path for every era: spec steps get their
                # tool's verb (done-form + count when terminal), engine
                # phases the orb's; a step the spec no longer knows
                # falls through the dictionary's total fallback.
                v = verbs.step_verb(
                    tool, connector,
                    phase=sid if tool is None else None,
                    status=status or "pending",
                    count=count,
                )
                steps.append({
                    "id": sid,
                    "label": v["label"],
                    "verb": v["label"],
                    "brand": s.get("brand", v["brand"]),
                    "status": status,
                    "duration_ms": s.get("duration_ms"),
                })
        except (ValueError, TypeError):
            pass
        fix = None
        # ND-15: a "Fix this" chip asserts there is something to
        # diagnose. A failure recorded with NO story — no outcome, no
        # user_message, no error_class — has nothing to offer, and a
        # chip on it invites the agent to investigate a blank.
        if j.status == "failed" and (
            j.outcome or j.user_message or j.error_class
        ):
            fix = verbs.fix_chip(
                names.get(j.source_id or "", "this automation") or
                "this automation",
                j.outcome, j.user_message or j.error_class,
            )
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
            "fix": fix,
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
