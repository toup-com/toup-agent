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
from . import ledger, workflow
from . import registry as reg

logger = logging.getLogger(__name__)


def _tz(user_id: str) -> ZoneInfo:
    from app.agent._user_tz_cache import get_cached_user_tz
    name = get_cached_user_tz(user_id) or "UTC"
    try:
        return ZoneInfo(name)
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
    tz = _tz(user_id)
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
            sentence = (
                f"Paused at step {max(step, 1)}."
                if v3 == "stopped_by_user" else "Working now"
            )
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
        elif v3 in ("failed",):
            who = expired_name or "an account"
            meta = f"Tried {_clock(last.completed_at, tz)} · " \
                   f"could not reach {who}"
        elif missing_account:
            missing_names = [x["name"] for x in accounts
                             if x["state"] == "expired"]
            meta = f"Waiting on {missing_names[0] if missing_names else 'an account'} · " \
                   f"{_acct_count(len(members))}"
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
        "tz": get_cached_user_tz(user_id),
    }
