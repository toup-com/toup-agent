"""Spec → primitives. Create, arm, pause, resume, delete.

The compiler owns the mapping decided in MAPPING.md:

    push     → Trigger row (kind="email_received",
               action="run_automation") — Gmail-only in v1; the
               existing Pub/Sub pipeline, dedupe and coalescing do the
               work. Watch provisioning reuses the trigger API helper.
    poll     → hidden system Routine (kind="automation_poll",
               schedule_kind="every") — excluded from user lists.
    schedule → user-visible Routine (kind="automation_schedule").

Bindings are the switch: `arm` validates the grant against the
platform, enables the primitive rows and flips binding.active; `pause`
disables them; `delete` removes them. The spec row itself is never
mutated by lifecycle verbs — re-compiling is deterministic.

Sessions: callers pass an open AsyncSession and own the commit — the
compiler never commits mid-flight, so an arm that fails halfway rolls
back atomically (no half-armed automation).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import delete as sa_delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    Automation, AutomationBinding, Routine, Trigger,
)
from .spec import ValidatedSpec

logger = logging.getLogger(__name__)

# Routine kinds owned by this engine. automation_poll is hidden from
# user-facing routine lists; automation_schedule is visible.
ROUTINE_KIND_POLL = "automation_poll"
ROUTINE_KIND_SCHEDULE = "automation_schedule"
AUTOMATION_ROUTINE_KINDS = frozenset({ROUTINE_KIND_POLL, ROUTINE_KIND_SCHEDULE})

# Trigger action owned by this engine (added to TRIGGER_ACTIONS).
TRIGGER_ACTION_RUN_AUTOMATION = "run_automation"


class CompileError(RuntimeError):
    """Lifecycle verb failed for a user-explainable reason."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


async def _reload_runner_routine(routine_id: str) -> None:
    """Ask the live RoutineRunner to pick up a row change. Fail-soft:
    in tests (no runner) the reconcile loop would cover it in prod."""
    try:
        from app.api import routines as routines_api
        runner = routines_api._runner
        if runner is not None:
            await runner.reload_routine(routine_id)
    except Exception as e:  # noqa: BLE001 — reload is a courtesy, reconcile covers
        logger.warning("[automations] runner reload failed for %s: %s",
                       routine_id, e)


async def compile_bindings(
    db: AsyncSession,
    automation: Automation,
    vspec: ValidatedSpec,
) -> list[AutomationBinding]:
    """Create (or replace) the primitive rows for a spec. Rows are
    created DISABLED — `arm` is the only thing that enables. Existing
    bindings for the automation are torn down first so update is
    re-compile, never patch."""
    from .spec_v2 import ValidatedSpecV2
    if isinstance(vspec, ValidatedSpecV2):
        return await _compile_bindings_v2(db, automation, vspec)

    await teardown_bindings(db, automation)

    bindings: list[AutomationBinding] = []
    if vspec.trigger_mode == "push":
        trigger = Trigger(
            user_id=automation.user_id,
            kind="email_received",
            action=TRIGGER_ACTION_RUN_AUTOMATION,
            enabled=False,
            name=f"[automation] {vspec.name}"[:100],
            filter_json=vspec.filter_rules or None,
            config_json={
                "automation_id": automation.id,
                **({"params": vspec.trigger_params} if vspec.trigger_params else {}),
            },
        )
        db.add(trigger)
        await db.flush()
        binding = AutomationBinding(
            automation_id=automation.id,
            user_id=automation.user_id,
            kind="trigger",
            target_id=trigger.id,
            active=False,
        )
    else:
        kind = (
            ROUTINE_KIND_POLL if vspec.trigger_mode == "poll"
            else ROUTINE_KIND_SCHEDULE
        )
        routine = Routine(
            user_id=automation.user_id,
            kind=kind,
            enabled=False,
            name=f"[automation] {vspec.name}"[:100],
            config_json={"automation_id": automation.id},
        )
        if vspec.trigger_mode == "poll":
            routine.schedule_kind = "every"
            routine.schedule_interval_seconds = vspec.poll_interval_s
            # 5-part cron column is NOT NULL; interval routines carry a
            # placeholder the runner never reads for schedule_kind='every'.
            routine.schedule_cron_local = "@every"
        else:
            sched = vspec.schedule or {}
            if sched.get("cron_local"):
                routine.schedule_kind = "cron"
                routine.schedule_cron_local = str(sched["cron_local"])
            elif sched.get("at"):
                routine.schedule_kind = "at"
                routine.schedule_cron_local = "@at"
                routine.schedule_at = datetime.fromisoformat(
                    str(sched["at"]).replace("Z", "+00:00")
                ).replace(tzinfo=None)
                routine.auto_disable_after_fire = True
            else:
                routine.schedule_kind = "every"
                routine.schedule_interval_seconds = int(sched["every_s"])
                routine.schedule_cron_local = "@every"
        db.add(routine)
        await db.flush()
        binding = AutomationBinding(
            automation_id=automation.id,
            user_id=automation.user_id,
            kind="routine",
            target_id=routine.id,
            active=False,
            detail_json=json.dumps({"routine_kind": kind}),
        )

    db.add(binding)
    await db.flush()
    bindings.append(binding)
    return bindings


async def _compile_bindings_v2(
    db: AsyncSession,
    automation: Automation,
    vspec,
) -> list[AutomationBinding]:
    """Round 28: one primitive row PER SOURCE. Each poll source gets
    its own hidden routine (its own interval, its own dedupe stream);
    the schedule source a visible routine; a push source a trigger
    row. `config_json.source_id` is how the fire handler finds its
    lane back in the spec."""
    await teardown_bindings(db, automation)

    bindings: list[AutomationBinding] = []
    for source in vspec.sources:
        if source.mode == "push":
            trigger = Trigger(
                user_id=automation.user_id,
                kind="email_received",
                action=TRIGGER_ACTION_RUN_AUTOMATION,
                enabled=False,
                name=f"[automation] {vspec.name}"[:100],
                filter_json=source.filter_rules or None,
                config_json={
                    "automation_id": automation.id,
                    "source_id": source.id,
                    **({"params": source.params} if source.params else {}),
                },
            )
            db.add(trigger)
            await db.flush()
            binding = AutomationBinding(
                automation_id=automation.id,
                user_id=automation.user_id,
                kind="trigger",
                target_id=trigger.id,
                active=False,
                detail_json=json.dumps({"source_id": source.id}),
            )
        else:
            kind = (
                ROUTINE_KIND_POLL if source.mode == "poll"
                else ROUTINE_KIND_SCHEDULE
            )
            routine = Routine(
                user_id=automation.user_id,
                kind=kind,
                enabled=False,
                name=f"[automation] {vspec.name}"[:100],
                config_json={"automation_id": automation.id,
                             "source_id": source.id},
            )
            if source.mode == "poll":
                routine.schedule_kind = "every"
                routine.schedule_interval_seconds = source.poll_interval_s
                routine.schedule_cron_local = "@every"
            else:
                sched = source.schedule or {}
                if sched.get("cron_local"):
                    routine.schedule_kind = "cron"
                    routine.schedule_cron_local = str(sched["cron_local"])
                elif sched.get("at"):
                    routine.schedule_kind = "at"
                    routine.schedule_cron_local = "@at"
                    routine.schedule_at = datetime.fromisoformat(
                        str(sched["at"]).replace("Z", "+00:00")
                    ).replace(tzinfo=None)
                    routine.auto_disable_after_fire = True
                else:
                    routine.schedule_kind = "every"
                    routine.schedule_interval_seconds = int(sched["every_s"])
                    routine.schedule_cron_local = "@every"
            db.add(routine)
            await db.flush()
            binding = AutomationBinding(
                automation_id=automation.id,
                user_id=automation.user_id,
                kind="routine",
                target_id=routine.id,
                active=False,
                detail_json=json.dumps({"routine_kind": kind,
                                        "source_id": source.id}),
            )
        db.add(binding)
        await db.flush()
        bindings.append(binding)
    return bindings


async def teardown_bindings(db: AsyncSession, automation: Automation) -> None:
    """Remove every primitive row this automation owns, then the
    binding rows. Missing targets are fine (stale binding)."""
    rows = (await db.execute(
        select(AutomationBinding)
        .where(AutomationBinding.automation_id == automation.id)
    )).scalars().all()
    for b in rows:
        if b.kind == "routine":
            routine = await db.get(Routine, b.target_id)
            if routine is not None:
                await db.delete(routine)
            await _reload_runner_routine(b.target_id)
        elif b.kind == "trigger":
            trigger = await db.get(Trigger, b.target_id)
            if trigger is not None:
                await db.delete(trigger)
    await db.execute(
        sa_delete(AutomationBinding)
        .where(AutomationBinding.automation_id == automation.id)
    )
    await db.flush()


async def nudge_routines(routine_ids: list[str]) -> None:
    """Post-COMMIT scheduler nudge. Must be called AFTER the
    transaction that changed the rows commits — `reload_routine`
    re-reads each row in its own session, so a pre-commit nudge sees
    the OLD state and does the opposite of what was intended."""
    for rid in routine_ids:
        await _reload_runner_routine(rid)


async def set_bindings_active(
    db: AsyncSession, automation: Automation, active: bool,
) -> list[str]:
    """Flip the primitive rows + binding.active. Returns the routine
    ids touched so the CALLER can `nudge_routines()` AFTER its commit.

    R28-D: the nudge used to live here, pre-commit — and
    `reload_routine` re-reads the row in its OWN session, which on any
    non-shared-connection DB (prod postgres, NullPool sqlite) sees only
    committed state, i.e. the OLD enabled value. Every arm therefore
    unregistered the routine it had just enabled, and the automation
    only actually scheduled at the next 10-minute reconcile. (The
    StaticPool test rig shared one connection, so the nudge saw the
    uncommitted flush and the bug was structurally invisible there.)"""
    rows = (await db.execute(
        select(AutomationBinding)
        .where(AutomationBinding.automation_id == automation.id)
    )).scalars().all()
    if not rows:
        raise CompileError(
            "no_bindings",
            "This automation has no compiled bindings — update it to "
            "re-compile before arming.",
        )
    for b in rows:
        if b.kind == "routine":
            routine = await db.get(Routine, b.target_id)
            if routine is None:
                raise CompileError(
                    "stale_binding",
                    "The schedule behind this automation is missing — "
                    "update the automation to rebuild it.",
                )
            routine.enabled = active
        elif b.kind == "trigger":
            trigger = await db.get(Trigger, b.target_id)
            if trigger is None:
                raise CompileError(
                    "stale_binding",
                    "The event subscription behind this automation is "
                    "missing — update the automation to rebuild it.",
                )
            trigger.enabled = active
        b.active = active
    await db.flush()
    return [b.target_id for b in rows if b.kind == "routine"]


async def verify_grants_for_arm_v2(
    automation: Automation, vspec,
) -> dict[str, dict]:
    """Round 28: verify EVERY write step's grant, fail closed on the
    first problem. Returns {step_id: grant dict} so arm can snapshot
    each pinned target into its step."""
    from .registry import fetch_grant

    grants: dict[str, dict] = {}
    for st in vspec.write_steps:
        grant = await fetch_grant(automation.user_id, st.grant_id or "")
        if grant is None:
            raise CompileError(
                "grant_unverifiable",
                f"The write permission for step {st.id!r} could not be "
                f"verified — it may not exist, or the platform is "
                f"unreachable. Nothing was armed.",
            )
        if grant.get("status") != "approved":
            raise CompileError(
                "grant_not_approved",
                f"The write permission for step {st.id!r} is "
                f"{grant.get('status')!r}, not approved. Ask for "
                f"permission first.",
            )
        if (
            grant.get("connector_id") != st.connector_id
            or grant.get("tool_name") != st.tool
        ):
            raise CompileError(
                "grant_target_mismatch",
                f"The approved permission for step {st.id!r} is for a "
                f"different action than the step performs.",
            )
        grants[st.id] = grant
    return grants


async def verify_grant_for_arm(
    automation: Automation, vspec: ValidatedSpec,
) -> Optional[dict]:
    """Arm-time grant check against the platform (the dispatcher
    re-verifies independently at every call). Returns the grant dict.

    Fails CLOSED: an unreachable platform blocks arming a write
    automation — an armed rule that cannot verify its permission is the
    exact thing the round brief forbids.
    """
    if not vspec.action_mutates:
        return None
    from .registry import fetch_grant

    grant = await fetch_grant(automation.user_id, vspec.grant_id or "")
    if grant is None:
        raise CompileError(
            "grant_unverifiable",
            "The write permission for this automation could not be "
            "verified — it may not exist, or the platform is "
            "unreachable. Nothing was armed.",
        )
    if grant.get("status") != "approved":
        raise CompileError(
            "grant_not_approved",
            f"The write permission is {grant.get('status')!r}, not "
            f"approved. Ask for permission first.",
        )
    if (
        grant.get("connector_id") != vspec.action_connector_id
        or grant.get("tool_name") != vspec.action_tool
    ):
        raise CompileError(
            "grant_target_mismatch",
            "The approved permission is for a different action than "
            "this automation performs.",
        )
    return grant
