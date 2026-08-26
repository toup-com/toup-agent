"""The automations home summary — §4.1 of CONTRACTS-R30.

One GET feeds the home cards, the sidebar badge and the menu header.
Times inside server-rendered sentences (`meta`) are rendered with the
user's stored `tz`, hours without a leading zero — never the server's
day (`date.today()` is the server's day, not the user's).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Automation, AutomationThread, BuildJob, Routine
from app.services import automation_verbs as verbs
from . import account_health, ledger, workflow
from . import registry as reg

logger = logging.getLogger(__name__)


async def _tz_name(db: AsyncSession, user_id: str) -> Optional[str]:
    """The user's IANA zone — cache first, then the DB.

    R31-26. Every stamp on the founder's home screen read UTC: `Tried
    14:21` on a 10:29 clock, `EDITED · 14:14` at 10:14, `Tried 14:36` at
    11:17. The cause is here rather than in the formatter — the tz came
    ONLY from a 5-minute in-process cache whose single writer is
    `agent_runner`'s per-turn seed. A summary served by a cold worker,
    or to a user who had not taken a chat turn in five minutes, had no
    tz at all and silently rendered every line in UTC, with the
    payload's own `tz` field null.

    So: fall back to the row. `users.timezone` is TENANT-authoritative
    (the CI-enforced shared-column map in db/models/base.py) and this is
    the agent lane, so `db` is the right database — the same column read
    from the platform lane is a different row and a known trap.
    """
    from app.agent._user_tz_cache import get_cached_user_tz
    name = get_cached_user_tz(user_id)
    if name:
        return name
    try:
        from app.db.models import User
        row = await db.get(User, user_id)
        name = getattr(row, "timezone", None) if row is not None else None
        if name:
            # Warm the cache for the rest of this request's renders.
            try:
                from app.agent._user_tz_cache import set_cached_user_tz
                set_cached_user_tz(user_id, name)
            except Exception:  # noqa: BLE001 — a cache is an optimisation
                pass
        return name
    except Exception as e:  # noqa: BLE001 — never fail a summary on tz
        logger.debug("[summary] tz fallback skipped: %s", e)
        return None


def _zone(name: Optional[str]) -> ZoneInfo:
    try:
        return ZoneInfo(name or "UTC")
    except Exception:  # noqa: BLE001
        return ZoneInfo("UTC")


def _clock(dt: Optional[datetime], tz: ZoneInfo) -> str:
    if dt is None:
        return ""
    local = dt.replace(tzinfo=timezone.utc).astimezone(tz)
    return f"{local.hour}:{local.minute:02d}"


def _when_label(dt: Optional[datetime], tz: ZoneInfo) -> str:
    if dt is None:
        return "soon"
    now = datetime.utcnow().replace(tzinfo=timezone.utc).astimezone(tz)
    local = dt.replace(tzinfo=timezone.utc).astimezone(tz)
    days = (local.date() - now.date()).days
    if days <= 0:
        return "tonight" if local.hour >= 18 else f"today {_clock(dt, tz)}"
    if days == 1:
        return f"tomorrow {_clock(dt, tz)}"
    return local.strftime("%A")


def _acct_count(n: int) -> str:
    return f"{n} account" + ("" if n == 1 else "s")


def _and_list(names: list[str]) -> str:
    """`A` / `A and B` / `A, B and C` — every name, at any count."""
    clean = [n for n in (names or []) if n]
    if not clean:
        return ""
    if len(clean) == 1:
        return clean[0]
    return ", ".join(clean[:-1]) + " and " + clean[-1]


def _brief_not_posted(job) -> bool:
    """Did the reads succeed and the WRITE fail? (§4.2's fourth row.)

    The distinction matters to the user's next action: a partial run
    that posted is missing information, a partial run that did not is
    missing the delivery, and only the second one leaves them waiting
    for a message that will never arrive.
    """
    return (job.outcome or "") == "failed" and bool(
        (ledger._cfg_of(job).get("accounts_touched") or [])
    )


async def _latest_run(
    db: AsyncSession, automation_id: str,
) -> Optional[BuildJob]:
    return (
        await db.execute(
            select(BuildJob)
            .where(BuildJob.source_id == automation_id)
            .where(BuildJob.job_type == "automation_run")
            .order_by(BuildJob.created_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()


async def _next_run_at(
    db: AsyncSession, automation_id: str,
) -> Optional[datetime]:
    from app.db.models import AutomationBinding
    rows = list((await db.execute(
        select(Routine.next_run_at)
        .join(AutomationBinding, AutomationBinding.target_id == Routine.id)
        .where(AutomationBinding.automation_id == automation_id)
        .where(AutomationBinding.kind == "routine")
        .where(AutomationBinding.active.is_(True))
    )).scalars())
    times = [t for t in rows if t is not None]
    return min(times) if times else None


def _status_of(a: Automation, missing_account: bool) -> tuple[str, str]:
    """(status, pill) per §4.1."""
    if a.status == "armed":
        return "active", "On"
    if a.status == "error" or a.paused_reason in (
        "connector_reauth", "grant_revoked", "auto_failures",
    ):
        return "needs_attention", "Needs you"
    if a.status == "draft":
        if missing_account:
            return "unarmed", "Just added"
        return "just_added", "Just added"
    return "paused", "Paused"


def _description_of(
    a: Automation, status: str, raw: dict, expired_name: Optional[str],
) -> str:
    base = (a.description or "").strip() \
        or (verbs.rule_sentence(raw) or "Runs on your behalf.")
    if status == "paused":
        return ("Paused. It keeps its setup and will not run until you "
                "resume.")
    if status == "needs_attention":
        who = expired_name or "An account"
        return (f"{who} refused the token. Nothing was missed — reconnect "
                "and it picks up where it stopped.")
    return base


async def summary_payload(db: AsyncSession, *, user_id: str) -> dict:
    tz_name = await _tz_name(db, user_id)
    tz = _zone(tz_name)
    rows = list((await db.execute(
        select(Automation)
        .where(Automation.user_id == user_id)
        .where(Automation.deleted_at.is_(None))
        .order_by(Automation.created_at.desc())
    )).scalars())
    connections = await reg.fetch_connection_state(user_id)

    items = []
    running_count = 0
    for a in rows:
        raw = workflow._spec_raw(a)
        members = workflow._member_connectors(raw)
        accounts = []
        expired_name = None
        missing_account = False
        for cid in members:
            entry = workflow._account_entry(cid, connections.get(cid) or {})
            if entry["state"] == "expired":
                expired_name = expired_name or entry["name"]
            if entry["state"] == "missing":
                missing_account = True
                entry["state"] = "expired"
            accounts.append(entry)

        status, pill = _status_of(a, missing_account)
        mode, mode_label = workflow.mode_of(a, raw)
        last = await _latest_run(db, a.id)
        v3 = ledger.run_v3_status(last) if last is not None else None

        run_in_flight = None
        if last is not None and v3 in ("running", "stopped_by_user") \
                and ledger.run_kind_of(last) != "question":
            total = int(last.progress_total or 0) or len(
                [s for s in (raw.get("steps") or [])]
            )
            step = int(last.progress_step or 0)
            checkpoint = ledger.checkpoint_of(last) or {}
            if v3 == "stopped_by_user":
                step = int(checkpoint.get("step_index") or step)
            if v3 == "stopped_by_user":
                # §3.2's stopped sentence is TWO clauses, and the second
                # is the point of it: a run the user stopped has to say
                # whether anything went out before it stopped. This
                # emitted "Paused at step 6." alone, so the one question
                # a stop raises — did it already do something? — was the
                # one the card would not answer. R30-B caught it on the
                # first live render of the founder's home.
                #
                # §4.3 already requires the stop note to carry the honest
                # writes count, so the number is in the ledger; the card
                # just was not reading it.
                from .run_v3 import writes_count as _writes_count
                n_writes = await _writes_count(db, last.id)
                if n_writes == 0:
                    made = "Nothing was sent."
                elif n_writes == 1:
                    made = "1 change already made."
                else:
                    made = f"{n_writes} changes already made."
                sentence = f"Paused at step {max(step, 1)}. {made}"
            else:
                sentence = "Working now"
            run_in_flight = {
                "run_id": last.id,
                "kind": ledger.run_kind_of(last),
                "step": step, "total": total,
                "sentence": sentence,
                "fraction": round(step / total, 3) if total else 0.0,
                "status": v3,
            }
            if v3 == "running":
                running_count += 1

        # meta (§3.2 forms) — rendered with the stored tz.
        touched = len(ledger._cfg_of(last).get("accounts_touched") or []) \
            if last is not None else 0
        if last is None:
            when = _when_label(await _next_run_at(db, a.id), tz)
            meta = f"First run {when} · {_acct_count(len(members))}"
            if mode_label:
                meta += f" · {mode_label}"
        elif status == "needs_attention" and expired_name:
            meta = f"{_acct_count(len(members))} · needs reconnecting"
        elif v3 == "waiting_on_user":
            meta = f"Waiting for you since {_clock(last.created_at, tz)}"
        elif v3 == "failed":
            # ND-16 (live): this used to read "could not reach an
            # account" for EVERY failure. §4.1's failed form names a
            # connector because it describes a connector problem —
            # applying it to a run that died mid-"Wrapping up" asserts a
            # refusal that never happened, on the home screen. Only
            # claim a connector when the ledger recorded one.
            #
            # R31-07: and when it recorded three, say three. The old
            # form took `failed_names[0]` and dropped the rest, so a run
            # that could not reach GitHub, Outlook and Teams told the
            # user about GitHub — and the user fixed GitHub, re-ran, and
            # met the next name. Every name, at any count.
            failed_ids = ledger._cfg_of(last).get("accounts_failed") or []
            failed_names = [x["name"] for x in accounts
                            if x["account_id"] in failed_ids]
            when = _clock(last.completed_at or last.created_at, tz)
            if not failed_names and expired_name \
                    and status == "needs_attention":
                failed_names = [expired_name]
            if failed_names:
                meta = account_health.form(
                    "tried_names_need_you", t=when,
                    names=_and_list(failed_names),
                )
            else:
                meta = account_health.form("tried_did_not_finish", t=when)
        elif v3 == "partial":
            # §4.2: a run that read some of its accounts and could not
            # read the others. It had NO form at all — it fell through
            # to "Ran 10:15 · 4 accounts touched", which is true and
            # says nothing about the two that are broken.
            failed_ids = ledger._cfg_of(last).get("accounts_failed") or []
            when = _clock(last.completed_at or last.created_at, tz)
            n_ok = max(touched - len(failed_ids), 0)
            total_accounts = max(len(members), touched)
            if _brief_not_posted(last):
                meta = account_health.form("ran_brief_not_posted", t=when)
            else:
                meta = account_health.form(
                    "ran_n_of_m", t=when, n=n_ok, m=total_accounts,
                )
        elif missing_account:
            missing_names = [x["name"] for x in accounts
                             if x["state"] == "expired"]
            # `an account` is banned in a status string (§4.3(7)): it is
            # the sentence this round exists to delete, and a fallback is
            # exactly where it survives a search-and-replace. With no
            # name to give, say the true thing that has one.
            meta = (
                f"Waiting on {_and_list(missing_names)} · "
                f"{_acct_count(len(members))}"
                if missing_names else
                f"Waiting on a connection · {_acct_count(len(members))}"
            )
        elif status == "paused" and last is not None:
            meta = account_health.form(
                "paused_ran_once",
                t=_clock(last.completed_at or last.created_at, tz),
            )
        else:
            meta = f"Ran {_clock(last.completed_at or last.created_at, tz)}" \
                   f" · {_acct_count(max(touched, 1))} touched"

        # The one blue text action.
        if status == "unarmed":
            action = "connect"
        elif status == "needs_attention":
            action = "reconnect"
        elif status == "active":
            action = "pause"
        elif status == "paused":
            action = "resume"
        elif status == "just_added" and last is None:
            action = "undo"
        else:
            action = None

        sched = workflow.schedule_block(a, raw)
        sched["next_run_label"] = _when_label(
            await _next_run_at(db, a.id), tz,
        )
        thread = (await db.execute(
            select(AutomationThread.id)
            .where(AutomationThread.automation_id == a.id)
        )).scalar_one_or_none()

        items.append({
            "id": a.id,
            "title": a.name,
            "status": status,
            "pill": pill,
            "description": _description_of(a, status, raw, expired_name),
            "accounts": accounts,
            "meta": meta,
            "action": action,
            "schedule": {k: sched[k] for k in
                         ("preset_id", "label", "sentence", "sub",
                          "next_run_label")},
            "mode": mode,
            "mode_label": mode_label,
            "run_in_flight": run_in_flight,
            "thread_id": thread,
        })

    n = len(items)
    if n == 0:
        headline = "Nothing set up yet"
    else:
        m = running_count
        working = f"{m} working now" if m else "none running"
        headline = f"{n} set up · {working}"

    unused = 0
    try:
        templates = await reg.fetch_templates(user_id)
        used_slugs = {a.template_slug for a in rows if a.template_slug}
        unused = sum(1 for t in templates
                     if (t.get("slug") or t.get("id")) not in used_slugs)
    except Exception:  # noqa: BLE001
        unused = 0

    from app.agent._user_tz_cache import get_cached_user_tz
    return {
        "automations": items,
        "headline": headline,
        "automation_count": n,
        "unused_count": unused,
        "tz": tz_name,
    }
