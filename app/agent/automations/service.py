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
from .spec import (
    SpecError, ValidatedSpec, unanswered_variables, validate_spec,
)

logger = logging.getLogger(__name__)


class AutomationNotFound(LookupError):
    pass


class MissingSettings(compiler.CompileError):
    """A lifecycle or fire refusal for a spec whose setup questions are
    still unanswered.

    A `CompileError` subclass on purpose: the automation is well formed
    and simply not ready, which is exactly what that type means here —
    and both arm call sites (the HTTP `_lifecycle` and the skill's)
    already render one as `{code, message}` for the user, so the
    refusal reaches a person rather than a 500. `missing` carries the
    variables by name and label so a caller can ask for them.
    """

    def __init__(self, missing: list[dict]):
        self.missing = list(missing)
        super().__init__("needs_answer", missing_settings_sentence(missing))


def missing_settings_sentence(missing: list[dict]) -> str:
    """The ONE sentence every surface shows for an unanswered setting.

    Deliberately the grammar `from-template` already opens a setup
    thread with: it is the same question, asked later, and the thread
    is where it is answered (the user replies, the thread agent writes
    the value back through `automations__update`). It names the
    setting, because the failure it replaces named nothing — an empty
    `owner` reached GitHub, came back "owner/repo required", and the
    user was shown "I could not read GitHub, and it did not tell me
    why" over a button that probed a healthy account.
    """
    from .account_health import join_names

    names = join_names([
        str(m.get("label") or m.get("name") or "") for m in missing
    ])
    return (f"Before this can run I need {names}. Tell me here and I "
            f"will set it up.")


async def missing_settings(
    automation: Automation, raw: Optional[dict] = None,
) -> list[dict]:
    """`[{"name", "label"}]` for every setup question this spec still
    needs answered — empty when it is ready to fire.

    The labels come from the catalog card the automation was adopted
    from, which is where they were authored; the fetch is skipped
    entirely when nothing is missing, and a blip degrades to the
    variable's own name rather than failing the refusal that needs it.
    """
    if raw is None:
        try:
            raw = json.loads(automation.spec_json or "{}")
        except (ValueError, TypeError):
            raw = {}
    names = unanswered_variables(raw)
    if not names:
        return []
    labels: dict[str, str] = {}
    slug = getattr(automation, "template_slug", None)
    if slug:
        try:
            for tpl in await reg.fetch_templates(automation.user_id):
                if slug not in (tpl.get("slug"), tpl.get("id")):
                    continue
                for v in tpl.get("variables") or []:
                    if v.get("name") and v.get("label"):
                        labels[str(v["name"])] = str(v["label"])
                break
        except Exception as e:  # noqa: BLE001 — the name is a fallback
            logger.warning(
                "[automations] variable labels unreadable for %s: %s",
                automation.id, e,
            )
    return [
        {"name": n, "label": labels.get(n) or n.replace("_", " ")}
        for n in names
    ]


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
    template_vars: Optional[set] = None,
) -> tuple[Automation, ValidatedSpec]:
    """Validate + persist a spec as a DRAFT automation with compiled
    (but disabled) bindings. Raises SpecError with every problem.
    `template_mode` (R30 from-template): write steps may lack grants —
    the draft cannot ARM until the grant conversation completes
    (arm_automation verifies fail-closed), so nothing weakens.
    `template_vars` (R36-1): the template's DECLARED variable names —
    an unanswered required variable is a setup-thread question, not a
    validation error, so its references must count as declared here."""
    from app.agent.automations.memory_notes import normalize_domain

    capability = await reg.fetch_registry(user_id)
    vspec = validate_spec(spec, capability, template_mode=template_mode,
                          template_vars=template_vars)

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


def _carry_grants_forward(old_raw: dict, spec: dict) -> dict:
    """R38 — an edit that does not touch a write step must not lose its
    grant.

    A model editing a spec routinely drops the grant id — the payload
    carries it, but a recompiled or hand-edited step comes back without
    one — and every `automations__update` then stripped the write's grant
    and hit the wall it could never climb. A new step lacking `grant_id`
    inherits the OLD spec's grant (and its pinned target, unless the edit
    pinned its own).

    Matched BY STEP ID first. The first cut keyed a pool on
    (connector_id, tool) and popped in list order, which is stable only
    for order-preserving edits: two writes on the same connector and tool
    (post to #eng, post to #ops) plus a drop or a reorder handed the
    survivor the FIRST old grant — and since the canonical param is
    `{{grant.target.id}}`, it renders to the inherited grant's target and
    the dispatcher's pinned-target check passes. A silent post to the
    wrong channel. The id is what the model echoes back unchanged, so it
    is what pairs; (connector_id, tool) order remains the fallback for a
    step that was genuinely re-created, where there is nothing better.

    A genuinely NEW write matches nothing and stays ungranted — legal at
    parse (template_mode), un-armable until granted
    (`verify_grants_for_arm_v2` fail-closes).
    """
    if not isinstance(spec, dict) or spec.get("version") != 2 \
            or (old_raw or {}).get("version") != 2:
        return spec
    # A grant a new step already claims is not up for inheritance — a
    # SECOND write with the same tool is genuinely new and must stay
    # ungranted rather than quietly wearing its sibling's permission.
    claimed = {
        s.get("grant_id") for s in (spec.get("steps") or [])
        if isinstance(s, dict) and s.get("grant_id")
    }
    granted_old = [
        s for s in (old_raw.get("steps") or [])
        if isinstance(s, dict) and s.get("grant_id")
        and s.get("grant_id") not in claimed
    ]
    by_id: dict[str, dict] = {
        str(s.get("id")): s for s in granted_old if s.get("id")
    }
    pool: dict[tuple, list[dict]] = {}
    for s in granted_old:
        pool.setdefault((s.get("connector_id"), s.get("tool")), []).append(s)
    if not granted_old:
        return spec

    def _take(step: dict) -> Optional[dict]:
        """The old granted step this one IS, or the next one that merely
        looks like it. An id match must also agree on connector and tool:
        a step that kept its id while being pointed at a different tool is
        a different write, and inheriting there is the widening this
        function exists to avoid."""
        sid = str(step.get("id") or "")
        old = by_id.get(sid)
        if old is not None \
                and old.get("connector_id") == step.get("connector_id") \
                and old.get("tool") == step.get("tool"):
            by_id.pop(sid, None)
            bucket = pool.get((old.get("connector_id"), old.get("tool")))
            if bucket and old in bucket:
                bucket.remove(old)
            return old
        candidates = pool.get((step.get("connector_id"), step.get("tool")))
        while candidates:
            cand = candidates.pop(0)
            cid = str(cand.get("id") or "")
            if cid and cid not in by_id:
                continue  # already paired by id above
            by_id.pop(cid, None)
            return cand
        return None

    steps: list = []
    changed = False
    for s in (spec.get("steps") or []):
        if isinstance(s, dict) and not s.get("grant_id"):
            old = _take(s)
            if old is not None:
                s = dict(s)
                s["grant_id"] = old.get("grant_id")
                if old.get("grant_target") and not s.get("grant_target"):
                    s["grant_target"] = old.get("grant_target")
                changed = True
        steps.append(s)
    if not changed:
        return spec
    out = dict(spec)
    out["steps"] = steps
    return out


async def update_automation(
    db: AsyncSession,
    *,
    automation_id: str,
    user_id: str,
    spec: dict,
    edited_note: bool = False,
) -> tuple[Automation, ValidatedSpec]:
    """Replace the spec and re-compile bindings. An armed automation
    stays armed only if the new spec passes the same arm checks —
    otherwise it drops to draft (never silently keep firing an old
    shape).

    R38: validates with template_mode=True — grants are enforced at ARM
    and DISPATCH, never by parse (the R36 doctrine `parse_spec_live`
    and `_replace_spec_template` already follow); refusing an edit over
    an unpinned write reproduced the founder's item 6 through the tool
    built to end it. Grants carry forward for unchanged write steps
    (`_carry_grants_forward`). `edited_note=True` (the agent-facing
    surfaces: `automations__update`, PATCH) appends the EDITED note and
    broadcasts `automation.updated` — one edit showed a divider and the
    next did not, because only the workflow writers ever stamped it.
    Internal callers that stamp their own note keep the default.
    """
    automation = await _load_owned(db, automation_id, user_id)
    capability = await reg.fetch_registry(user_id)
    try:
        old_raw = json.loads(automation.spec_json or "{}")
    except (ValueError, TypeError):
        old_raw = {}
    spec = _carry_grants_forward(old_raw, spec)
    # `template_vars=None` (the default) waives the grant rule AND the
    # undeclared-variable rule, and BOTH waivers are deliberate: a spec
    # mid-setup legitimately carries a variable whose question the user
    # has not answered yet, which is the state run-now used to answer 500
    # about (R37, the founder's dead Run button). An EDIT must be able to
    # persist that state; what may not happen is firing it, and that is
    # now refused where it is decidable — `arm_automation` and
    # `parse_spec_live` both raise `MissingSettings` rather than letting
    # `render_value` resolve the dangling `{{var.x}}` to "".
    vspec = validate_spec(spec, capability, template_mode=True)

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
    if edited_note:
        from . import workflow as _workflow
        await _workflow._edited_note(db, automation)
    await db.refresh(automation)
    return automation, vspec


async def arm_automation(
    db: AsyncSession, *, automation_id: str, user_id: str,
) -> Automation:
    """draft/paused → armed. Refuses while a setup question is
    unanswered, verifies the grant against the platform (fail closed),
    snapshots the pinned target into the spec for template rendering,
    enables the primitives, provisions push.

    The settings check lives HERE rather than in each caller because
    arming is the moment a draft becomes something that fires on its
    own: `POST /{id}/arm`, the skill's `automations__arm`, the
    grant-approval hook and this module's own re-arm after an edit all
    inherit it. `from-template` refuses the same state at creation, but
    it was the only one that did — approving the write permission card
    armed an automation whose questions had never been answered, and
    every weekday after that it read GitHub with an empty owner and
    blamed a healthy account for the refusal.
    """
    automation = await _load_owned(db, automation_id, user_id)
    missing = await missing_settings(automation)
    if missing:
        raise MissingSettings(missing)
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
                    # R39: keep the step's own pinned target when the
                    # platform's grant echo omits `target` — overwriting
                    # with {} un-pinned a destination the user had just
                    # set, one re-arm later.
                    s["grant_target"] = (g.get("target")
                                         or s.get("grant_target") or {})
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


async def _replace_spec_template(
    db: AsyncSession, automation: Automation, new_raw: dict,
) -> Automation:
    """Persist a replacement spec for a possibly-MID-SETUP automation.

    `update_automation` validates with template_mode=False, which is
    right for a finished spec and wrong for the drafts the R37
    write-backs exist to move: a from-template draft may still carry a
    dangling {{var.x}} or another connector's ungranted write, and
    refusing the WHOLE change over those reproduces the founder's
    item 6 through the very tool that was built to end it. Grants are
    enforced at ARM and DISPATCH — the same rule parse_spec_live
    follows — so validation here matches the spec's actual state.
    """
    capability = await reg.fetch_registry(automation.user_id)
    vspec = validate_spec(new_raw, capability, template_mode=True)
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
    await db.refresh(automation)
    return automation


async def set_destination_chat(
    db: AsyncSession, *, automation_id: str, user_id: str,
) -> dict:
    """R37: "send it here in the chat" becomes a real spec change.

    The chat destination is not a new delivery lane — it is the
    reads-only shape the engine has always had: the run's brief lands
    as typed turns in the automation's own thread plus the one
    notification card in the day chat. So "in this chat" = remove the
    outside write steps, which also removes the grant the write was
    blocked on, which is what lets a template draft finally ARM. The
    founder's Inbox summary sat unarmed for exactly this: the agent
    agreed "I'll keep it in this chat" twice and nothing could ever
    move, because agreeing was the only tool it had.
    """
    from app.services.automation_verbs import is_write_tool
    from . import workflow

    automation = await _load_owned(db, automation_id, user_id)
    raw = workflow._spec_raw(automation)
    if raw.get("version") != 2:
        raise SpecError([{"code": "v1_spec",
                          "message": "This automation predates "
                                     "destination changes."}])
    steps = [dict(s) for s in raw.get("steps") or []]
    kept = [s for s in steps
            if not (s.get("grant_id") or is_write_tool(s.get("tool")))]
    if len(kept) == len(steps):
        return {"changed": False, "armed": automation.status == "armed",
                "sentence": "It already lands here — nothing to change."}
    if not kept:
        raise SpecError([{"code": "no_steps",
                          "message": "Removing the write would leave "
                                     "this automation with nothing to "
                                     "do."}])
    was_armed = automation.status == "armed"
    new_raw = dict(raw)
    new_raw["steps"] = kept
    automation = await _replace_spec_template(db, automation, new_raw)

    # Arm when it was armed before (removing a write can only narrow
    # what fires), or when every remaining member is connected — the
    # same predicate from-template uses. An armed automation whose only
    # source is disconnected fires straight into a NEEDS YOU card.
    missing: list[str] = []
    if not was_armed:
        try:
            state = await reg.fetch_connection_state(user_id) or {}
            members = workflow._member_connectors(new_raw)
            missing = [c for c in members
                       if not (state.get(c) or {}).get("connected")]
        except Exception as e:  # noqa: BLE001 — unknown state arms nothing
            logger.warning("[automations] connection state unreadable: %s", e)
            missing = ["(unknown)"]
    armed = False
    if was_armed or not missing:
        try:
            automation = await arm_automation(
                db, automation_id=automation_id, user_id=user_id,
            )
            armed = True
        except (compiler.CompileError, SpecError) as e:
            logger.info("[automations] chat destination left %s in "
                        "draft: %s", automation_id, e)
    await workflow._edited_note(db, automation)
    sentence = (
        "The brief lands here in this thread now — nothing posts "
        "anywhere else" + (", and it is armed." if armed else
                           ", but it is not armed yet.")
    )
    return {"changed": True, "armed": armed, "sentence": sentence,
            "missing": missing}


async def pin_destination(
    db: AsyncSession, *, automation_id: str, user_id: str,
    connector_id: str, tool: str, grant: dict, target: dict,
) -> dict:
    """R37: pin ONE write step's target and remember the grant that was
    just requested for it. The automation stays DRAFT until the user
    approves — `_grant_decided` finishes the arm.

    Exactly one step: the grant was minted for one (connector, tool,
    target) triple, and stamping every write of the connector silently
    redirected an ALREADY-APPROVED sibling destination on multi-write
    specs. The first unpinned step with that tool is the one being set
    up; with none unpinned, the first with that tool is the redirect
    the user asked for."""
    from . import workflow

    automation = await _load_owned(db, automation_id, user_id)
    raw = workflow._spec_raw(automation)
    if raw.get("version") != 2:
        raise SpecError([{"code": "v1_spec",
                          "message": "This automation predates "
                                     "destination changes."}])
    steps = [dict(s) for s in raw.get("steps") or []]
    matching = [s for s in steps
                if s.get("connector_id") == connector_id
                and s.get("tool") == tool]
    if not matching:
        raise SpecError([{"code": "no_write_step",
                          "message": f"No {connector_id} write to pin."}])
    chosen = next(
        (s for s in matching if not (s.get("grant_target") or {}).get("id")),
        matching[0],
    )
    chosen["grant_id"] = str(grant.get("id") or "")
    chosen["grant_target"] = {
        "kind": str((target or {}).get("kind") or ""),
        "id": str((target or {}).get("id") or ""),
        "label": str((target or {}).get("label") or ""),
    }
    new_raw = dict(raw)
    new_raw["steps"] = steps
    automation = await _replace_spec_template(db, automation, new_raw)
    await workflow._edited_note(db, automation)
    return {"changed": True, "armed": False}


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

    # R36-8: the day-chat cards stop pointing at a thread that no
    # longer opens. The card stays (it is the user's record that this
    # ran), but its status says so and its body stops inviting a tap
    # into a dead screen — the founder's 6:54 card still read "open the
    # run" two hours after he deleted the automation.
    try:
        from sqlalchemy import select as _select
        from .cards import update_card_message, broadcast_card
        rows = (await db.execute(
            _select(AutomationNotification)
            .where(AutomationNotification.automation_id == automation_id)
            .where(AutomationNotification.message_id.isnot(None))
        )).scalars().all()
        seen_msgs: set = set()
        for row in rows:
            if row.message_id in seen_msgs:
                continue
            seen_msgs.add(row.message_id)
            row.status = "deleted"
            row.body = "This automation was deleted."
            payload = None
            try:
                from .run_v3 import _notification_payload
                payload = _notification_payload(row)
            except Exception:  # noqa: BLE001
                payload = None
            if payload is not None:
                await update_card_message(
                    db, message_id=row.message_id,
                    metadata_key="automation_notification",
                    payload=payload,
                )
                await broadcast_card(
                    user_id, "automation_notification",
                    {**payload, "message_id": row.message_id},
                )
    except Exception as e:  # noqa: BLE001 — retiring copy never blocks
        logger.warning("[automations] card retire skipped %s: %s",
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


async def rehearse(
    db: AsyncSession, *, automation_id: str, user_id: str,
) -> dict:
    """A rehearsal: the reads run for real against live data, the
    writes are RENDERED and reported, and nothing is staged.

    R38, and the reason this replaced `test_run` outright rather than
    growing a flag: the old path committed the outbox row and returned,
    and `outbox.flush_loop` sweeps every staged row whose undo window
    has closed — so the "rehearsal" posted to the user's real channel
    seconds later, from a background loop, with the caller gone. There
    is no outbox row now, so no loop, restart or retry can turn one
    into a send.
    """
    from . import executor
    automation = await _load_owned(db, automation_id, user_id)
    vspec = await parse_spec_live(automation)
    from .spec_v2 import ValidatedSpecV2
    if isinstance(vspec, ValidatedSpecV2):
        from . import executor_v2
        return await executor_v2.rehearse_v2(db, automation, vspec)
    return await executor.rehearse(db, automation, vspec)


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
    # R38: the pins go with the account. A pin under a connector the
    # spec no longer uses is a place nothing will ever start from, and
    # re-adding the account later would silently restore a target the
    # user last saw months ago.
    focus = raw.get("focus")
    if isinstance(focus, dict) and connector_id in focus:
        focus = {k: v for k, v in focus.items() if k != connector_id}
        if focus:
            raw["focus"] = focus
        else:
            raw.pop("focus", None)
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
        # template_mode: a persisted spec already passed the create
        # gate; this re-parse is shape recovery, not re-litigation — an
        # ungranted write (legal for a from-template draft) must parse
        # as the WRITE it is, not fail or demote to a read (R36-5).
        return validate_spec(raw, _permissive_registry_v2(raw),
                             template_mode=True)
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
        # R36-5: writes are writes by TOOL. Keying this on grant_id let
        # an ungranted template draft re-validate as a READ, and the
        # fire path then executed gmail__create_draft through the read
        # loop — straight into the dispatcher's grant gate, whose
        # refusal the thread reported as "Could not reach Gmail".
        from app.services.automation_verbs import is_write_tool
        if step.get("tool") and (step.get("grant_id")
                                 or is_write_tool(step.get("tool"))):
            entry["scopes_write_by_action"][step["tool"]] = []
    return permissive


async def parse_spec_live(automation: Automation) -> ValidatedSpec:
    """The FIRE-path parse: validate against the live capability
    registry (5-min cached; a network blip serves the cache) so
    `event_spec` carries the real source_tool / items_path / fields.
    Falls back to the offline shape when the registry is empty — the
    poll then observes nothing and the health loop records the miss
    rather than inventing capabilities.

    Refuses outright while a setup question is unanswered. The parse
    below runs in `template_mode`, which waives the
    undeclared-variable rule so a mid-setup draft can still be read —
    and downstream `render_value` turns the dangling `{{var.x}}` into
    an empty string, so the run reached GitHub with `owner=""` and
    Teams with `chat_id=""` and then reported healthy accounts as
    unreadable. There is nothing honest a run can do with a missing
    answer, so it does not start: `MissingSettings` carries the
    sentence the thread shows instead.
    """
    raw = json.loads(automation.spec_json)
    missing = await missing_settings(automation, raw)
    if missing:
        raise MissingSettings(missing)
    capability = await reg.fetch_registry(automation.user_id)
    if capability:
        try:
            # template_mode (R36-5): grants are enforced at DISPATCH
            # (the grant gate) and at ARM (fail-closed verify) — not by
            # this parse. Without it, every from-template automation
            # whose write is still ungranted failed `write_without_
            # grant` HERE on the fire path, fell back to the permissive
            # shape, and ran its draft step as a read.
            return validate_spec(raw, capability, template_mode=True)
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
