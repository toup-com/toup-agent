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

BRIEFING_KIND = "email_briefing"
DEFAULT_NAME = "Morning mail brief"

# ND-9 (found live on the founder, 2026-08-25): selecting on
# kind == "email_briefing" alone matched NOTHING on the real account —
# the founder's "Morning new-email briefing" is kind `agent_task` with
# cron `0 8 * * *`, and the route answered a clean 200 with an empty
# result, which would have been recorded as "migration done". The
# selector now matches what the DATA looks like, not what the kind
# vocabulary suggested.
MIGRATABLE_KINDS = (BRIEFING_KIND, "agent_task")

# NEVER migrated, for two different reasons:
#   reminder            — §4.11a: pure reminders keep the main-chat path.
#   automation_poll     — engine-owned. These ARE an automation's own
#   automation_schedule   compiled bindings; "migrating" one would
#                         duplicate the automation that owns it.
NEVER_MIGRATE_KINDS = ("reminder", "automation_poll", "automation_schedule")

# An agent_task is only automation-shaped if it RECURS. A one-shot
# (`at`, or auto_disable_after_fire) is a reminder wearing another kind.


def _norm_name(name: Optional[str]) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (name or "").lower()).strip()


def _same_intent(routine_name: Optional[str],
                 automation_name: Optional[str]) -> bool:
    """Whether an automation already owns a routine's intent.

    ND-13 (live): exact normalised equality never fired — the founder's
    routine "Jira → Slack new-issue alerts" and the automation
    "Jira → Slack" are the same intent under different titles, so the
    per-minute duplicate kept running. One title being a token SUBSET
    of the other is the honest test; ">= 2 shared tokens" keeps a
    single common word ("Daily", "Morning") from collapsing two
    unrelated things.
    """
    r = set(_norm_name(routine_name).split())
    a = set(_norm_name(automation_name).split())
    if not r or not a:
        return False
    shared = r & a
    return len(shared) >= 2 and (a <= r or r <= a)


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


_MAIL_RE = re.compile(r"\b(e-?mails?|inbox|mailbox|unread mail)\b", re.I)


def _likely_mail(routine: Routine) -> bool:
    """A HINT for the report — never a gate.

    ND-12 (live, 2026-08-25): this used to decide, on a substring scan
    of free prose, whether a routine could be rewritten into a Gmail
    read. It converted "Daily motivational quote" — "Send Nariman one
    short motivational quote every day" — into an automation whose rule
    read "Every day at 16:39, check Gmail." The intent was not
    misstated, it was REPLACED. No keyword predicate over prose can
    decide that safely: "don't email me about this" contains "email",
    and the founder's own quote routine matched on text the operator
    never saw. So intent is no longer inferred; it is SELECTED (see
    `migrate_email_briefings`), and this only annotates the report.
    """
    if routine.kind == BRIEFING_KIND:
        return True
    return bool(_MAIL_RE.search(
        f"{routine.name or ''} {routine.prompt_text or ''}"))

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
    routine_ids: Optional[list] = None,
) -> dict:
    """Migrate every un-migrated `email_briefing` routine for this user.

    Returns {"migrated": [...], "skipped": [...], "errors": [...]} —
    one entry per routine either way, so the founder pass (§4.11a) can
    read exactly what happened. Never raises for a single routine's
    failure; a routine that could not migrate keeps firing unchanged
    (disabling a briefing we failed to replace would silently lose it).
    """
    rows = await _candidate_routines(db, user_id)

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

        selected = routine_ids is not None and routine.id in routine_ids

        # ND-12: a routine the user SWITCHED OFF must not be resurrected
        # as a new object. The founder's list went 3 -> 6 that way.
        if not routine.enabled and not selected:
            needs_review.append({
                "routine_id": routine.id, "name": routine.name,
                "reason": "disabled — the user turned it off; select it "
                          "explicitly to migrate it anyway",
            })
            continue

        twin = _find_twin(routine, existing)
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

        # ND-12: INTENT IS SELECTED, NEVER INFERRED. The migrated spec
        # reads Gmail, so the only routines migrated without an explicit
        # instruction are the ones that are structurally a briefing
        # (kind == email_briefing). Everything else — however mail-shaped
        # its prose looks — is reported for a human to choose: a keyword
        # scan cannot tell "summarise my inbox" from "don't email me
        # about this", and it rewrote a motivational-quote routine into
        # an automation whose rule read "Every day at 16:39, check Gmail".
        if routine.kind != BRIEFING_KIND and not selected:
            needs_review.append({
                "routine_id": routine.id, "name": routine.name,
                "likely_mail": _likely_mail(routine),
                "reason": "not a briefing by kind — select it explicitly "
                          "to migrate it into a mail brief",
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
        routine.config_json = {
            **cfg, "migrated_to": automation.id,
            # A repair must put the routine back exactly as it was;
            # without this the undo has to guess.
            "migrated_from_enabled": bool(routine.enabled),
        }
        routine.enabled = False
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


def _find_twin(routine: Routine, automations: list):
    for a in automations:
        if _same_intent(routine.name, a.name):
            return a
    return None


async def repair_mismigrations(db: AsyncSession, *, user_id: str) -> dict:
    """Undo migrations this module should never have made.

    ND-12's trap: a mis-migrated routine is STAMPED, so a corrected
    selector skips it — the bad automation persists and the fix cannot
    self-heal. This walks the stamps and reverses any pair today's rules
    would not produce unprompted: the automation is deleted (we created
    it, and a mis-migration is caught while it is still a draft that
    never fired) and the routine is restored to the state recorded at
    migration time.

    Refuses to touch an automation that has RUN — a repair must never
    delete a record of work the user can see.
    """
    from app.db.models import Automation as _Automation, BuildJob
    from . import service as _svc

    rows = await _candidate_routines(db, user_id)
    repaired: list[dict] = []
    kept: list[dict] = []
    for routine in rows:
        cfg = dict(routine.config_json or {})
        aid = cfg.get("migrated_to")
        if not aid:
            continue
        if routine.kind == BRIEFING_KIND:
            kept.append({"routine_id": routine.id, "automation_id": aid,
                         "reason": "a briefing by kind — correct"})
            continue
        automation = await db.get(_Automation, aid)
        if automation is None:
            cfg.pop("migrated_to", None)
            routine.config_json = cfg
            await db.commit()
            repaired.append({"routine_id": routine.id,
                             "automation_id": aid,
                             "action": "stamp_cleared"})
            continue
        ran = (await db.execute(
            select(BuildJob.id)
            .where(BuildJob.source_id == aid)
            .where(BuildJob.job_type == "automation_run")
            .limit(1)
        )).scalar_one_or_none()
        if ran is not None:
            kept.append({"routine_id": routine.id, "automation_id": aid,
                         "reason": "it has already run — kept; delete it "
                                   "by hand if it is wrong"})
            continue

        await _svc.delete_automation(
            db, automation_id=aid, user_id=user_id, undo=True,
        )
        was_enabled = bool(cfg.pop("migrated_from_enabled", False))
        cfg.pop("migrated_to", None)
        routine.config_json = cfg
        routine.enabled = was_enabled
        await db.commit()
        await compiler.nudge_routines([routine.id])
        repaired.append({
            "routine_id": routine.id, "name": routine.name,
            "automation_id": aid, "action": "automation_deleted",
            "routine_enabled_restored": was_enabled,
        })
        logger.info("[automations] repaired mis-migration %s -> %s",
                    routine.id, aid)
    return {"repaired": repaired, "kept": kept}


async def _candidate_routines(db: AsyncSession, user_id: str):
    """THE selector — shared by the migration and its report.

    ND-11: these were two queries and only one of them was widened, so
    the audit view could not see its own subjects: the report answered
    `{"routines": []}` on an account whose routines the migration was
    about to act on, and an empty report reads as "nothing to migrate".
    One query, one place to change.
    """
    return (await db.execute(
        select(Routine)
        .where(Routine.user_id == user_id)
        .where(Routine.kind.in_(MIGRATABLE_KINDS))
        .order_by(Routine.created_at)
    )).scalars().all()


async def migration_report(db: AsyncSession, *, user_id: str) -> dict:
    """A DRY RUN, not a listing: each candidate with the outcome the
    migration would give it, so a before/after capture is legible and
    the report can never disagree with what running it does. Classes
    mirror `migrate_email_briefings` exactly: already_migrated,
    already_superseded, would_supersede, would_need_review, would_migrate.
    """
    from app.db.models import Automation as _Automation

    rows = await _candidate_routines(db, user_id)
    existing = (await db.execute(
        select(_Automation)
        .where(_Automation.user_id == user_id)
        .where(_Automation.deleted_at.is_(None))
    )).scalars().all()
    by_name = {_norm_name(a.name): a for a in existing if a.name}

    out = []
    for r in rows:
        cfg = r.config_json or {}
        entry = {
            "routine_id": r.id,
            "name": r.name,
            "kind": r.kind,
            "enabled": bool(r.enabled),
            "schedule": _schedule_for(r),
            "migrated_to": cfg.get("migrated_to"),
            "superseded_by": cfg.get("superseded_by"),
        }
        twin = by_name.get(_norm_name(r.name))
        if cfg.get("migrated_to"):
            entry["outcome"] = "already_migrated"
        elif cfg.get("superseded_by"):
            entry["outcome"] = "already_superseded"
        elif not _recurring(r):
            entry["outcome"] = "would_need_review"
            entry["reason"] = "one-shot — a reminder, not an automation"
        elif twin is not None:
            entry["outcome"] = "would_supersede"
            entry["superseded_by"] = twin.id
            entry["automation_name"] = twin.name
        elif not r.enabled:
            entry["outcome"] = "would_need_review"
            entry["reason"] = "disabled — the user turned it off"
        elif r.kind != BRIEFING_KIND:
            entry["outcome"] = "would_need_review"
            entry["likely_mail"] = _likely_mail(r)
            entry["reason"] = ("not a briefing by kind — select it "
                               "explicitly to migrate it")
        else:
            entry["outcome"] = "would_migrate"
        out.append(entry)
    return {"routines": out}
