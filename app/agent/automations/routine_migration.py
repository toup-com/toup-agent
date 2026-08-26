"""Routine → automation migration (CONTRACTS-R30.md §4.11a).

`email_briefing` routines collapse ONCE into spec-v2 automations so the
single list is the only place a briefing lives (the routine/automation
pair was D-12's "two objects for one intent"). Mechanics, per the
written decision:

  - enabled routine  → armed automation; disabled → paused draft.
  - The routine row is disabled and stamped
    `config_json.migrated_to = <automation_id>` so it can never
    double-fire — the stamp is also the idempotency key: a second call
    migrates nothing.
  - The migrated spec has ONE schedule source carrying the routine's
    schedule VERBATIM — the routine's cron IS the promised time
    (§4.11b); nothing is re-derived from a creation instant.
  - Steps are reads only: one gmail read (the §4.11a documented
    deviation — delivery channels map to the notification pipeline's
    existing channel fan-out, never to write steps, so no grant is
    needed and mode "auto" is safe).
  - `reminder` routines are NEVER touched; they keep the main-chat path
    (§4.11a), and neither are the engine-owned `automation_poll` /
    `automation_schedule` kinds — those ARE an automation's own compiled
    bindings.
  - `agent_task` routines ARE in scope when they are automation-shaped:
    RECURRING (cron/every, never a one-shot) and mail-shaped, because
    the migrated spec reads gmail. ND-9: the founder's "Morning
    new-email briefing" is an `agent_task`, so selecting on
    `email_briefing` alone matched nothing on the real account and
    reported success having done nothing.
  - Two honest refusals instead of a silent mis-migration:
    `needs_review` for a recurring task whose intent the gmail-read spec
    would misstate (a Jira alerter is not a mail brief) and for
    one-shots; `superseded` for a routine whose intent an automation
    ALREADY owns — that is the D-12 collapse: the duplicate is retired
    against the existing automation rather than cloned into a third
    object.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Routine

from . import compiler, registry as reg, service
from .spec import SpecError
from .spec_v2 import validate_spec_v2

logger = logging.getLogger(__name__)

MIGRATABLE_KIND = "email_briefing"
DEFAULT_NAME = "Morning mail brief"

# ND-9 (found live on the founder, 2026-08-25): selecting on
# kind == "email_briefing" alone matched NOTHING on the real account —
# the founder's "Morning new-email briefing" is kind `agent_task` with
# cron `0 8 * * *`, and the route answered a clean 200 with an empty
# result, which would have been recorded as "migration done". The
# selector now matches what the DATA looks like, not what the kind
# vocabulary suggested.
MIGRATABLE_KINDS = ("email_briefing", "agent_task")

# NEVER migrated, for two different reasons:
#   reminder            — §4.11a: pure reminders keep the main-chat path.
#   automation_poll     — engine-owned. These ARE an automation's own
#   automation_schedule   compiled bindings; "migrating" one would
#                         duplicate the automation that owns it.
NEVER_MIGRATE_KINDS = ("reminder", "automation_poll", "automation_schedule")

# An agent_task is only automation-shaped if it RECURS. A one-shot
# (`at`, or auto_disable_after_fire) is a reminder wearing another kind.
_MAIL_WORDS = ("mail", "email", "inbox", "brief")


def _norm_name(name: Optional[str]) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (name or "").lower()).strip()


def _recurring(routine: Routine) -> bool:
    if getattr(routine, "auto_disable_after_fire", False):
        return False
    kind = (routine.schedule_kind or "cron").lower()
    if kind == "cron":
        cron = (routine.schedule_cron_local or "").strip()
        return bool(cron) and not cron.startswith("@")
    if kind == "every":
        return int(routine.schedule_interval_seconds or 0) > 0
    return False  # "at" is a one-shot


def _mail_shaped(routine: Routine) -> bool:
    """Whether the migrated spec's gmail read step honestly represents
    this routine's intent. A Jira alerter is NOT a mail brief, and
    converting one into a gmail read would misstate what it does — those
    are reported as `needs_review` instead of silently mis-migrated."""
    if routine.kind == MIGRATABLE_KIND:
        return True
    haystack = f"{routine.name or ''} {routine.prompt_text or ''}".lower()
    return any(w in haystack for w in _MAIL_WORDS)

# The gmail read step, shaped like the template catalog's morning-brief
# gmail section (automation_template_catalog._MORNING_WORK_BRIEF) so
# migrated briefings render and collect exactly like template-built ones.
_GMAIL_READ_STEP = {
    "id": "mail",
    "connector_id": "gmail",
    "tool": "gmail__list_messages",
    "params": {"query": "is:unread newer_than:1d", "max_results": 10},
    "collect": {
        "items_path": "messages",
        "fields": {"subject": "headers.Subject", "from": "headers.From"},
        "format": "• {{item.from}} — {{item.subject}}",
        "limit": 8,
        "empty_text": "Gmail inbox is clear.",
    },
}


def promised_time_cron(text_hhmm: str) -> str:
    """§4.11(b) regression seam: the time the user STATED ("8:00")
    rendered as the 5-part cron that must be armed ("0 8 * * *").
    Setup paths arm at the stated time; the migration never calls this
    — it copies the routine's cron verbatim, because that cron already
    IS the promised time."""
    hh, _, mm = text_hhmm.strip().partition(":")
    hour = int(hh)
    minute = int(mm or 0)
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"not a wall-clock time: {text_hhmm!r}")
    return f"{minute} {hour} * * *"


def _schedule_for(routine: Routine) -> dict:
    """The routine's REAL schedule, verbatim — exactly one of the three
    spec-v2 schedule shapes (§4.11b: never re-derived)."""
    kind = routine.schedule_kind or "cron"
    if kind == "at" and routine.schedule_at is not None:
        return {"at": routine.schedule_at.isoformat() + "Z"}
    if kind == "every" and routine.schedule_interval_seconds:
        return {"every_s": int(routine.schedule_interval_seconds)}
    return {"cron_local": routine.schedule_cron_local}


def _spec_for(routine: Routine) -> dict:
    return {
        "version": 2,
        "name": (routine.name or "").strip() or DEFAULT_NAME,
        "description": "Migrated from the email briefing routine.",
        "mode": "auto",
        "trigger": {
            "sources": [
                {"id": "sched", "mode": "schedule",
                 "schedule": _schedule_for(routine)},
            ],
        },
        "steps": [dict(_GMAIL_READ_STEP)],
    }


async def migrate_email_briefings(
    db: AsyncSession, *, user_id: str,
) -> dict:
    """Migrate every un-migrated `email_briefing` routine for this user.

    Returns {"migrated": [...], "skipped": [...], "errors": [...]} —
    one entry per routine either way, so the founder pass (§4.11a) can
    read exactly what happened. Never raises for a single routine's
    failure; a routine that could not migrate keeps firing unchanged
    (disabling a briefing we failed to replace would silently lose it).
    """
    rows = (await db.execute(
        select(Routine)
        .where(Routine.user_id == user_id)
        .where(Routine.kind.in_(MIGRATABLE_KINDS))
        .order_by(Routine.created_at)
    )).scalars().all()

    capability = await reg.fetch_registry(user_id)

    # D-12 collapse: an intent that ALREADY exists as an automation must
    # not be migrated into a second copy of itself. The founder's
    # per-minute "Jira → Slack new-issue alerts" routine duplicates
    # automation "Jira → Slack" — one intent, two objects, one of them
    # firing every minute. Those are retired against the automation that
    # already owns the intent instead of being cloned.
    from app.db.models import Automation as _Automation
    existing = (await db.execute(
        select(_Automation)
        .where(_Automation.user_id == user_id)
        .where(_Automation.deleted_at.is_(None))
    )).scalars().all()
    by_name = {_norm_name(a.name): a for a in existing if a.name}

    migrated: list[dict] = []
    superseded: list[dict] = []
    needs_review: list[dict] = []
    skipped: list[dict] = []
    errors: list[dict] = []
    for routine in rows:
        cfg = dict(routine.config_json or {})
        if cfg.get("migrated_to") or cfg.get("superseded_by"):
            skipped.append({
                "routine_id": routine.id,
                "migrated_to": cfg.get("migrated_to"),
                "superseded_by": cfg.get("superseded_by"),
            })
            continue

        if routine.kind in NEVER_MIGRATE_KINDS:  # defence in depth
            continue

        if not _recurring(routine):
            needs_review.append({
                "routine_id": routine.id, "name": routine.name,
                "reason": "one-shot — a reminder, not an automation",
            })
            continue

        twin = by_name.get(_norm_name(routine.name))
        if twin is not None:
            # Retire the duplicate against the automation that owns the
            # intent. Disabled + stamped in one commit, nudged AFTER.
            routine.enabled = False
            routine.config_json = {**cfg, "superseded_by": twin.id}
            await db.commit()
            await compiler.nudge_routines([routine.id])
            superseded.append({
                "routine_id": routine.id, "name": routine.name,
                "superseded_by": twin.id, "automation_name": twin.name,
            })
            logger.info(
                "[automations] routine %s retired — automation %s already "
                "owns that intent (D-12)", routine.id, twin.id,
            )
            continue

        if not _mail_shaped(routine):
            # The migrated spec reads GMAIL. Converting a non-mail task
            # into a mail brief would misstate what the user set up.
            needs_review.append({
                "routine_id": routine.id, "name": routine.name,
                "reason": "not mail-shaped — migrating it would misstate "
                          "its intent; needs a compiled spec",
            })
            continue

        spec = _spec_for(routine)
        try:
            # Validate HERE first (the canonical v2 shape), then persist
            # through the same service path chat- and API-built
            # automations use — byte-identical lifecycle.
            validate_spec_v2(spec, capability)
            automation, _vspec = await service.create_automation(
                db, user_id=user_id, spec=spec, template_slug=None,
            )
        except SpecError as e:
            logger.warning(
                "[automations] routine %s did not migrate (spec invalid): %s",
                routine.id, e,
            )
            errors.append({"routine_id": routine.id, "error": str(e)})
            continue

        was_enabled = bool(routine.enabled)
        arm_error = None
        if was_enabled:
            try:
                await service.arm_automation(
                    db, automation_id=automation.id, user_id=user_id,
                )
            except (compiler.CompileError, SpecError) as e:
                # Log, never raise — the automation stays a draft and
                # the report says why (the routine is still retired
                # below: the automation now owns the intent).
                arm_error = str(e)
                logger.warning(
                    "[automations] migrated automation %s stayed draft "
                    "(arm failed): %s", automation.id, e,
                )

        # Retire the routine: disabled + stamped, in one commit. The
        # stamp MERGES into config_json (connector_identity_id etc.
        # survive) — and is what makes a second call a no-op.
        routine.enabled = False
        routine.config_json = {**cfg, "migrated_to": automation.id}
        await db.commit()
        # AFTER the commit — a pre-commit nudge reads the old row
        # (R28-D) and would re-schedule the routine we just disabled.
        await compiler.nudge_routines([routine.id])

        migrated.append({
            "routine_id": routine.id,
            "automation_id": automation.id,
            "status": automation.status,
            "armed": was_enabled and arm_error is None,
            **({"arm_error": arm_error} if arm_error else {}),
        })
        logger.info(
            "[automations] migrated routine %s -> automation %s (%s)",
            routine.id, automation.id, automation.status,
        )

    return {"migrated": migrated, "superseded": superseded,
            "needs_review": needs_review, "skipped": skipped,
            "errors": errors}


async def migration_report(db: AsyncSession, *, user_id: str) -> dict:
    """Audit view: every email_briefing routine with its stamp."""
    rows = (await db.execute(
        select(Routine)
        .where(Routine.user_id == user_id)
        .where(Routine.kind == MIGRATABLE_KIND)
        .order_by(Routine.created_at)
    )).scalars().all()
    return {
        "routines": [
            {
                "routine_id": r.id,
                "name": r.name,
                "migrated_to": (r.config_json or {}).get("migrated_to"),
                "enabled": bool(r.enabled),
            }
            for r in rows
        ],
    }
