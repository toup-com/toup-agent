"""The whole workflow — one GET, per-sheet writes, the composer (R30 §4.4).

Everything the canvas and its sheets read comes from ONE call
(`workflow_payload`), resolved from one source of truth per concern:
the spec (steps, sources, mode), `permissions.py` (the per-automation
account permissions), the platform connection state (account states),
`rules_json` (the user's standing rules), and the verb dictionary
(every human string). Writes are per sheet, never one big PUT; every
workflow write appends an `EDITED` note turn to the automation's
thread so the thread stays the full record.

The composer is a real conversation with the agent: C's classifier
(`composer.classify_change`) names the intents; THIS module applies
the safe ones, refuses the rest into `needs`, mints 10-second undo
tokens, and writes the thread record (user turn → EDITED note →
agent confirmation).
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Automation
from app.services import automation_verbs as verbs
from . import ledger, permissions
from . import registry as reg
from .draft_card import DRAFT_TOOLS

logger = logging.getLogger(__name__)


class WorkflowError(Exception):
    def __init__(self, code: str, sentence: str, extra: Optional[dict] = None):
        super().__init__(sentence)
        self.code = code
        self.sentence = sentence
        self.extra = extra or {}


# ------------------------------------------------------------- presets

# The four canvas presets (§4.4) — ids are wire-stable.
SCHEDULE_PRESETS = (
    {"id": "weekdays-8", "cron_local": "0 8 * * 1-5",
     "sentence": "Weekdays at 8:00",
     "sub": "Finishes before your first meeting"},
    {"id": "weekdays-730", "cron_local": "30 7 * * 1-5",
     "sentence": "Weekdays at 7:30",
     "sub": "Ready earlier, before the commute"},
    {"id": "daily-8", "cron_local": "0 8 * * *",
     "sentence": "Every morning at 8:00", "sub": "Weekends included"},
    {"id": "weekdays-9", "cron_local": "0 9 * * 1-5",
     "sentence": "Weekdays at 9:00", "sub": "Right before standup"},
)


def _label_of(sentence: str) -> str:
    """`"Weekdays at 8:00"` → `"Weekdays 8:00"` — label and sentence
    come from one source (§4.1)."""
    return sentence.replace(" at ", " ", 1)


def _schedule_of(raw: dict) -> Optional[dict]:
    """The one schedule dict out of a spec (v2 source or v1 trigger)."""
    if raw.get("version") == 2:
        for s in (raw.get("trigger") or {}).get("sources") or []:
            if s.get("schedule"):
                return s["schedule"]
        return None
    return (raw.get("trigger") or {}).get("schedule")


def _current_sentence(raw: dict) -> str:
    human = verbs.schedule_human(raw) or "On its own schedule"
    s = human[0].upper() + human[1:]
    # "Weekdays 8:00" → "Weekdays at 8:00" for the rows; label strips it.
    parts = s.rsplit(" ", 1)
    if len(parts) == 2 and ":" in parts[1] and " at " not in s:
        s = f"{parts[0]} at {parts[1]}"
    return s


def schedule_block(automation: Automation, raw: dict) -> dict:
    sched = _schedule_of(raw) or {}
    cron = sched.get("cron_local")
    current_id = None
    for p in SCHEDULE_PRESETS:
        if cron and cron == p["cron_local"]:
            current_id = p["id"]
            break
    presets = [
        {"id": p["id"], "label": _label_of(p["sentence"]),
         "sentence": p["sentence"], "sub": p["sub"]}
        for p in SCHEDULE_PRESETS
    ]
    sentence = _current_sentence(raw)
    if current_id is None:
        current_id = "current"
        presets.append({
            "id": "current", "label": _label_of(sentence),
            "sentence": sentence, "sub": "What it does today",
        })
    else:
        sentence = next(p["sentence"] for p in SCHEDULE_PRESETS
                        if p["id"] == current_id)
    return {
        "preset_id": current_id,
        "label": _label_of(sentence),
        "sentence": sentence,
        "sub": next((p["sub"] for p in presets if p["id"] == current_id), ""),
        "presets": presets,
    }


# ------------------------------------------------------------ the GET

def _spec_raw(automation: Automation) -> dict:
    try:
        raw = json.loads(automation.spec_json or "{}")
    except (ValueError, TypeError):
        raw = {}
    return raw if isinstance(raw, dict) else {}


def _member_connectors(raw: dict) -> list[str]:
    ids: list[str] = []
    if raw.get("version") == 2:
        for s in (raw.get("trigger") or {}).get("sources") or []:
            cid = s.get("connector_id")
            if cid and cid not in ids:
                ids.append(cid)
        for s in raw.get("steps") or []:
            cid = s.get("connector_id")
            if cid and cid not in ids:
                ids.append(cid)
    else:
        # Same v1 shape correction as `_steps_human`: flat, not `.source`.
        # Pre-existing — a v1 automation's TRIGGER connector was never
        # counted as a member, so it appeared in no accounts list.
        _trig = raw.get("trigger") or {}
        for part in (_trig.get("source") or _trig,
                     raw.get("action") or {}):
            cid = part.get("connector_id")
            if cid and cid not in ids:
                ids.append(cid)
    return ids


def focus_of(raw: dict) -> dict[str, list[dict]]:
    """The spec's per-account pins, `{connector_id: [{kind,id,label}]}`.

    Total: a spec written before R38 has no `focus` key and answers
    `{}`, which is the same thing as "nothing pinned" — there is no
    third state to distinguish, because a pin is a user action and its
    absence means only that they have not taken it.
    """
    focus = raw.get("focus")
    if not isinstance(focus, dict):
        return {}
    out: dict[str, list[dict]] = {}
    for cid, pins in focus.items():
        if not isinstance(cid, str) or not isinstance(pins, list):
            continue
        rows = [
            {"kind": str(p.get("kind") or ""),
             "id": str(p.get("id") or ""),
             "label": str(p.get("label") or p.get("id") or "")}
            for p in pins
            if isinstance(p, dict) and p.get("id") and p.get("kind")
        ]
        if rows:
            out[cid] = rows
    return out


def _write_tools(raw: dict) -> list[tuple[str, str, dict]]:
    """[(connector_id, tool, grant_target)] for the spec's writes.

    R36-5: a write is a write by its TOOL, never by its grant. The old
    predicate required `grant_id` (or a grant_target, which canonical
    form drops when empty), so an UNGRANTED template write step was
    invisible here — and everything derived from this function told the
    user the opposite of the spec's purpose: `mode_of` said "reads
    only" about an automation whose whole job is a Gmail draft, the
    setup script said "I cannot change anything", and the menu label
    agreed. The founder read all three on a Newsletter roundup whose
    template promises a draft.
    """
    out = []
    if raw.get("version") == 2:
        for s in raw.get("steps") or []:
            if s.get("grant_id") or verbs.is_write_tool(s.get("tool")):
                out.append((s.get("connector_id") or "",
                            s.get("tool") or "",
                            s.get("grant_target") or {}))
    else:
        action = raw.get("action") or {}
        if action.get("tool"):
            out.append((action.get("connector_id") or "",
                        action.get("tool") or "", {}))
    return out


def mode_of(automation: Automation, raw: dict) -> tuple[str, str]:
    """(mode, mode_label) per §4.1 derivation."""
    writes = _write_tools(raw)
    if not writes:
        return "reads_only", "reads only"
    if (raw.get("mode") or "auto") == "confirm":
        return "asks_first", "asks first"
    if all(tool in DRAFT_TOOLS for _, tool, _ in writes):
        return "drafts_only", "drafts only"
    _, tool, target = writes[0]
    label = (target or {}).get("label") or (target or {}).get("id")
    where = f"posts to {label}" if label else "posts"
    return "posts", where


def output_block(automation: Automation, raw: dict) -> dict:
    mode, _ = mode_of(automation, raw)
    writes = _write_tools(raw)
    header_sub = {
        "reads_only": "nothing is sent on your behalf",
        "drafts_only": "nothing is sent on your behalf",
        "posts": "only where you allowed it",
        "asks_first": "after you say yes",
    }[mode]
    lines: list[dict] = [{
        "title": "A brief on your phone",
        "body": "Ranked by what breaks if you ignore it.",
    }]
    node_label, node_sub = "Brief to you", "on your phone"
    first_write_action = None
    for cid, tool, target in writes:
        act = verbs.turn_action(
            cid, tool, kind="write", ok=True,
            target=(target or {}).get("label") or (target or {}).get("id"),
            audience="you" if tool in DRAFT_TOOLS else "others",
        )
        if first_write_action is None:
            first_write_action = act["action"]
        if tool in DRAFT_TOOLS:
            name = verbs.display_name(cid) or "your mail"
            node_sub = f"drafts wait in {name}"
            lines.append({
                "title": f"Drafts waiting in {name}",
                "body": "Written, never sent — you read and send them "
                        "yourself.",
            })
        else:
            lines.append({
                "title": act["action"],
                "body": act["detail"] or "Only where you allowed it.",
            })
    if mode == "posts" and first_write_action:
        node_label = first_write_action
        extra = [t for c, t, g in writes[1:]]
        node_sub = "and nothing else" if not extra else node_sub
    lines.append({
        "title": "It tells you when it fails",
        "body": "If an account refuses it, the run stops and you hear why.",
    })
    return {"node_label": node_label, "node_sub": node_sub,
            "header_sub": header_sub, "lines": lines}


def counts_block(raw: dict, mode: str) -> dict:
    sources = (raw.get("trigger") or {}).get("sources") or []
    noun = {"reads_only": "brief", "drafts_only": "brief",
            "posts": "post", "asks_first": "draft"}.get(mode, "brief")
    return {"items_per_fire": max(1, len(sources)),
            "briefs_per_run": 1, "noun": noun}


def _steps_human(automation: Automation, raw: dict) -> list[dict]:
    try:
        stored = json.loads(automation.steps_human_json or "[]")
    except (ValueError, TypeError):
        stored = []
    if stored:
        return [
            {"n": i + 1, "text": s.get("text") or "", "sub": s.get("sub") or ""}
            for i, s in enumerate(stored)
        ]
    out = []
    if raw.get("version") == 2:
        for i, s in enumerate(raw.get("steps") or []):
            if s.get("kind") == "agent":
                # R38. Derived through the same dictionary as every
                # other step; the sub is the ask itself, because on this
                # sheet "Thought it through" alone says nothing about
                # WHAT it thought through. `turn_action` would answer
                # "Checked the account" here — a step with no account.
                out.append({
                    "n": i + 1,
                    "text": verbs.engine_action("think")["action"],
                    "sub": str(s.get("prompt") or "")[:120],
                })
                continue
            cid, tool = s.get("connector_id") or "", s.get("tool") or ""
            is_write = bool(s.get("grant_id"))
            act = verbs.turn_action(
                cid, tool, kind="write" if is_write else "read", ok=True,
            )
            name = verbs.display_name(cid) or ""
            out.append({
                "n": i + 1, "text": act["action"],
                "sub": act["detail"] or name,
            })
    else:
        # AUDIT-8: only v2 was derived, so every v1 automation opened an
        # EMPTY Steps sheet — the canvas asserting the thing does
        # nothing. A v1 spec is exactly one read (the trigger's source)
        # and one write (its action); derive both through the same verb
        # dictionary, so the sheet reads like a v2 one.
        # A v1 trigger carries its connector FLAT (`spec._TRIGGER_KEYS`
        # is {mode, connector_id, event, params, poll_interval_s,
        # schedule, filter} — there is no `source` key and `validate_spec`
        # REJECTS one). This first read `trigger.source`, so it always
        # resolved to {} and the read step this branch exists to derive
        # was never emitted; only the write survived. The pin passed
        # because its fixture invented a `trigger.source` the validator
        # would have refused. `source` stays as a tolerated alias.
        trigger = raw.get("trigger") or {}
        source = trigger.get("source") or trigger
        action = raw.get("action") or {}
        if source.get("connector_id"):
            cid = source["connector_id"]
            act = verbs.turn_action(cid, None, kind="read", ok=True)
            out.append({
                "n": len(out) + 1, "text": act["action"],
                "sub": act["detail"] or verbs.display_name(cid) or "",
            })
        if action.get("tool"):
            cid = action.get("connector_id") or ""
            act = verbs.turn_action(
                cid, action["tool"], kind="write", ok=True,
                audience="you" if action["tool"] in DRAFT_TOOLS
                else "others",
            )
            out.append({
                "n": len(out) + 1, "text": act["action"],
                "sub": act["detail"] or verbs.display_name(cid) or "",
            })
    return out


def rules_list(automation: Automation) -> list[dict]:
    try:
        rules = json.loads(automation.rules_json or "[]")
    except (ValueError, TypeError):
        rules = []
    return [r for r in rules if isinstance(r, dict) and r.get("text")]


def _account_entry(cid: str, conn: dict) -> dict:
    status = (conn or {}).get("status") or ""
    connected = bool((conn or {}).get("connected"))
    state = "connected" if connected and status == "active" else (
        "expired" if status in ("reauth_required", "provider_down")
        else ("connected" if connected else "missing")
    )
    return {
        "account_id": cid,
        "connector_id": cid,
        "name": verbs.display_name(cid) or cid,
        "account_label": (conn or {}).get("account") or "",
        "state": state,
    }


async def workflow_payload(
    db: AsyncSession, *, automation: Automation, user_id: str,
) -> dict:
    raw = _spec_raw(automation)
    connections = await reg.fetch_connection_state(user_id)
    members = _member_connectors(raw)
    mode, _mode_label = mode_of(automation, raw)

    pinned = focus_of(raw)
    accounts = []
    for cid in members:
        entry = _account_entry(cid, connections.get(cid) or {})
        if entry["state"] == "missing":
            entry["state"] = "expired"
        perms = await permissions.resolve(
            db, automation=automation, account_id=cid,
        )
        entry.update(perms)
        entry["last_use"] = await _last_use(db, automation, cid)
        # R38 — where this account STARTS every run. On the account
        # entry rather than a sibling map, because the canvas draws one
        # node per account and a node must not have to join two lists
        # to know its own pins.
        entry["focus"] = pinned.get(cid) or []
        accounts.append(entry)

    capability = await reg.fetch_registry(user_id)
    available = [
        _account_entry(cid, connections.get(cid) or {})
        for cid in sorted(set(capability) | set(connections))
        if cid not in members and cid != "stub"
    ]

    from app.agent._user_tz_cache import get_cached_user_tz
    from . import build_ledger
    return {
        "automation_id": automation.id,
        "name": automation.name,
        "workflow_rev": int(getattr(automation, "workflow_rev", 0) or 0),
        "schedule": schedule_block(automation, raw),
        "accounts": accounts,
        "available": available,
        "steps": _steps_human(automation, raw),
        "rules": rules_list(automation),
        "output": output_block(automation, raw),
        "counts": counts_block(raw, mode),
        # R38. `null` when the automation predates the build ledger —
        # never `[]`, which would claim it was built in no steps.
        "build_history": build_ledger.read(automation),
        "tz": get_cached_user_tz(user_id),
    }


async def _last_use(
    db: AsyncSession, automation: Automation, cid: str,
) -> dict:
    """`{"sentence", "at"}` from the automation's last tool turn on
    that account — the ledger is the truth, not the provider."""
    from sqlalchemy import select
    from app.db.models import AutomationThread, AutomationTurn
    thread = (
        await db.execute(
            select(AutomationThread).where(
                AutomationThread.automation_id == automation.id,
            )
        )
    ).scalar_one_or_none()
    if thread is None:
        return {"sentence": "No runs yet", "at": None}
    rows = list((await db.execute(
        select(AutomationTurn)
        .where(AutomationTurn.thread_id == thread.id)
        .where(AutomationTurn.kind == "tool")
        .order_by(AutomationTurn.seq.desc())
        .limit(40)
    )).scalars())
    for r in rows:
        try:
            body = json.loads(r.payload_json)
        except (ValueError, TypeError):
            continue
        if body.get("account_id") != cid:
            continue
        # R31-25 at the READ boundary. These are stored verbatim from the
        # turn that wrote them, so a row persisted before the renderer was
        # made total still carries `{need_count}` — and this is the call
        # that puts it on the connector card. Drop the clause here too.
        from app.services.automation_verbs import drop_unfilled
        action = drop_unfilled(body.get("action") or "") or "Used it"
        detail = drop_unfilled(body.get("detail") or "")
        sentence = f"{action} · {detail}" if detail else action
        return {"sentence": sentence[:120],
                "at": r.created_at.isoformat() + "Z"}
    return {"sentence": "No runs yet", "at": None}


# ------------------------------------------------------------- writes

async def bump_rev(db: AsyncSession, automation: Automation) -> int:
    """Advance the workflow revision (§4.6). Total; never raises."""
    try:
        row = await db.get(Automation, automation.id)
        if row is None:
            return 0
        row.workflow_rev = int(row.workflow_rev or 0) + 1
        await db.commit()
        return int(row.workflow_rev)
    except Exception as e:  # noqa: BLE001 — a rev never fails a write
        logger.warning("[workflow] rev bump skipped: %s", e)
        return int(getattr(automation, "workflow_rev", 0) or 0)


async def _edited_note(
    db: AsyncSession, automation: Automation,
) -> Optional[str]:
    """Every workflow write appends the EDITED note (§4.4) and
    broadcasts `automation.updated` (§4.6).

    One seam for both, because they are one fact: the workflow changed.
    R31-11 is what happens when the second half is missing — the
    founder removed Outlook in the Workflow, came back to the thread,
    and the header still showed five chips while the ⋯ menu still said
    `5 accounts`, because those two surfaces read a summary nobody had
    told. Every writer in this module already calls this function, so
    putting the broadcast here means a new writer cannot forget it.
    """
    try:
        thread = await ledger.ensure_thread(
            db, user_id=automation.user_id, automation_id=automation.id,
        )
        turn = await ledger.append_turn(
            db, user_id=automation.user_id, thread=thread, run_id=None,
            kind="note",
            payload={"stamp": "edited",
                     "at": datetime.utcnow().isoformat() + "Z"},
        )
        turn_id = turn["id"]
    except Exception as e:  # noqa: BLE001
        logger.warning("[workflow] EDITED note skipped: %s", e)
        turn_id = None
    try:
        rev = await bump_rev(db, automation)
        await ledger.emit_updated(
            db, automation.user_id, automation_id=automation.id,
            workflow_rev=rev,
        )
    except Exception as e:  # noqa: BLE001 — a frame never fails a write
        logger.warning("[workflow] automation.updated skipped: %s", e)
    return turn_id


async def set_schedule_preset(
    db: AsyncSession, *, automation: Automation, user_id: str,
    preset_id: str,
) -> dict:
    # AUDIT-7: the payload offers a synthetic `current` row so the sheet
    # can show a schedule that is none of the four presets. Selecting it
    # is a no-op, not an error — the writer used to 409 on the very id
    # it had just served.
    if preset_id == "current":
        raw_now = _spec_raw(automation)
        block = schedule_block(automation, raw_now)
        return {"schedule": block,
                "sentence": "Left the time as it was."}
    preset = next((p for p in SCHEDULE_PRESETS if p["id"] == preset_id),
                  None)
    if preset is None:
        raise WorkflowError("unknown_preset", "Pick one of the times shown.")
    from . import service
    await service.set_schedule(
        db, user_id=user_id, automation_id=automation.id,
        schedule={"cron_local": preset["cron_local"]},
    )
    await db.refresh(automation)
    await _edited_note(db, automation)
    sentence = preset["sentence"]
    return {
        "schedule": schedule_block(automation, _spec_raw(automation)),
        "sentence": f"Moved it to {sentence[0].lower()}{sentence[1:]}.",
    }


_ISO_WEEKDAY_CRON = {1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6",
                    7: "0"}   # cron: Sunday is 0, ISO: Sunday is 7


def custom_cron(custom: dict) -> tuple[str, dict]:
    """Turn the picker's `{time, days, date?, tz}` into a schedule dict.

    CONTRACTS-R31 §4.7. The picker sends what the USER chose; the cron
    it compiles to is evaluated in the user's own zone
    (`Routine.schedule_cron_local`), which is the semantics R30-D
    already proved works — `Daily 22:52` fired at 22:52 Toronto exactly.
    What was broken was never the timezone: it was that a chat promise
    of "8:00 Toronto" and the armed cron were two objects nobody
    reconciled. Here they cannot diverge, because the sentence the
    thread shows is rendered FROM the schedule that was armed.
    """
    raw_time = str(custom.get("time") or "").strip()
    try:
        hh, mm = raw_time.split(":")
        hour, minute = int(hh), int(mm)
    except (ValueError, AttributeError):
        raise WorkflowError("bad_time", "Pick a time first.")
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise WorkflowError("bad_time", "Pick a time first.")

    date = str(custom.get("date") or "").strip()
    if date:
        # A one-time automation: an `at` schedule, not a cron. It must
        # be in the FUTURE — an `at` in the past is R31-33's shape, and
        # a picker is exactly where one gets typed.
        try:
            when = datetime.strptime(f"{date} {hour:02d}:{minute:02d}",
                                     "%Y-%m-%d %H:%M")
        except ValueError:
            raise WorkflowError("bad_date", "Pick a date first.")
        if when <= datetime.utcnow() - timedelta(days=1):
            raise WorkflowError(
                "past_date", "Pick a date that has not happened yet.")
        return "at", {"at": when.isoformat()}

    days = [int(d) for d in (custom.get("days") or [])
            if isinstance(d, (int, str)) and str(d).isdigit()]
    days = sorted({d for d in days if 1 <= d <= 7})
    dow = ",".join(_ISO_WEEKDAY_CRON[d] for d in days) if days else "*"
    return "cron_local", {"cron_local": f"{minute} {hour} * * {dow}"}


async def set_schedule_custom(
    db: AsyncSession, *, automation: Automation, user_id: str,
    custom: dict,
) -> dict:
    """§4.7's `Custom…` row — a real time, in the user's own zone."""
    from . import service
    _kind, schedule = custom_cron(custom)
    await service.set_schedule(
        db, user_id=user_id, automation_id=automation.id,
        schedule=schedule,
    )
    await db.refresh(automation)
    await _edited_note(db, automation)
    block = schedule_block(automation, _spec_raw(automation))
    return {
        "schedule": block,
        # The sentence renders the schedule that was ARMED, so a
        # mismatch between what the user asked for and what fired is
        # visible in the thread rather than discovered days later.
        "sentence": f"Moved it to {block['sentence'][0].lower()}"
                    f"{block['sentence'][1:]}.",
    }


async def commit_workflow(
    db: AsyncSession, *, automation: Automation, user_id: str,
    workflow_rev: Optional[int], schedule: Optional[dict] = None,
    permissions: Optional[list] = None, steps: Optional[list] = None,
    rules: Optional[dict] = None, accounts: Optional[dict] = None,
) -> dict:
    """The workflow's ✓ — every draft, in ONE transaction (§4.6).

    Supersedes R30 §4.4's "never one big PUT" for this path only. The
    per-sheet routes stay for `/workflow/ask` and the web; what changes
    is that the CANVAS commits once. The reason is not tidiness: a
    canvas that saved per sheet left the user's edits half-applied
    whenever one of them was refused, and on a tenant that boots dark
    for 40-70 s each of those was its own round trip.

    Refusal semantics, in the order the app depends on:
      - `409 stale` — the drafts were made against an older workflow.
        Nothing is applied; the caller re-bases and re-layers its
        drafts. Checked FIRST, so a stale commit cannot half-apply.
      - `409 refused {item, sentence}` — one item is impossible (a hard
        rail, the last read, a missing consent). Nothing is applied.
      - `200` — everything applied.

    `steps` are deliberately NOT in the transaction: they go through
    C's recompiler, which is an LLM call that can take seconds and can
    legitimately refuse one step. They are applied after, and reported
    as `pending: ["steps"]`; a refused step reverts that step alone.
    """
    current_rev = int(getattr(automation, "workflow_rev", 0) or 0)
    if workflow_rev is not None and int(workflow_rev) != current_rev:
        return {
            "code": "stale",
            "workflow_rev": current_rev,
            "workflow": await workflow_payload(
                db, automation=automation, user_id=user_id,
            ),
        }

    from . import service

    # ── phase 1: everything that can refuse, before anything is written.
    #
    # `save_permissions` and the membership writers raise WorkflowError /
    # MembershipError on refusal. Doing the whole set inside one
    # transaction is what makes "a 409 applies nothing" true; the
    # alternative — apply, hit a refusal, unwind — is the half-applied
    # state this route exists to remove.
    applied: list[str] = []
    async with db.begin_nested():
        if accounts:
            for account_id in (accounts.get("remove") or []):
                await service.remove_connector(
                    db, automation_id=automation.id, user_id=user_id,
                    connector_id=str(account_id),
                )
                applied.append(f"account-{account_id}")
            for account_id in (accounts.get("add") or []):
                await service.add_connector(
                    db, automation_id=automation.id, user_id=user_id,
                    connector_id=str(account_id),
                )
                applied.append(f"account+{account_id}")

        if rules:
            for rule_id in (rules.get("remove") or []):
                await delete_rule(db, automation=automation,
                                  rule_id=str(rule_id), note=False)
                applied.append("rule-")
            for edit in (rules.get("edit") or []):
                await update_rule(db, automation=automation,
                                  rule_id=str(edit.get("id") or ""),
                                  text=str(edit.get("text") or ""),
                                  note=False)
                applied.append("rule~")
            for text in (rules.get("add") or []):
                await add_rule(db, automation=automation, text=str(text),
                               note=False)
                applied.append("rule+")

        if permissions:
            for perm in permissions:
                await save_permissions(
                    db, automation=automation, user_id=user_id,
                    account_id=str(perm.get("account_id") or ""),
                    can_ids=list(perm.get("can") or []),
                    cant_ids=list(perm.get("cant") or []),
                    note=False,
                )
                applied.append("perm")

        if schedule:
            if schedule.get("custom"):
                await set_schedule_custom(
                    db, automation=automation, user_id=user_id,
                    custom=schedule["custom"],
                )
            elif schedule.get("preset_id"):
                await set_schedule_preset(
                    db, automation=automation, user_id=user_id,
                    preset_id=str(schedule["preset_id"]),
                )
            applied.append("schedule")

    await db.commit()
    await db.refresh(automation)

    # ONE note and ONE frame for the whole commit — not one per draft.
    # The thread should read "EDITED" once for something the user did
    # once.
    await _edited_note(db, automation)

    out = {
        "workflow": await workflow_payload(
            db, automation=automation, user_id=user_id,
        ),
        "workflow_rev": int(getattr(automation, "workflow_rev", 0) or 0),
    }
    if steps:
        out["pending"] = ["steps"]
    return out


async def set_steps(
    db: AsyncSession, *, automation: Automation, user_id: str,
    steps: list[dict],
) -> dict:
    """The Steps sheet's one debounced recompile (§4.4). The human
    sentences persist immediately; the spec regeneration goes through
    C's recompiler seam when present — a missing recompiler keeps the
    old spec and says so honestly."""
    cleaned = [
        {"text": str(s.get("text") or "").strip()[:200],
         "sub": str(s.get("sub") or "").strip()[:200]}
        for s in steps if str(s.get("text") or "").strip()
    ]
    if not cleaned:
        raise WorkflowError("empty_steps", "A plan needs at least one step.")
    before_human = automation.steps_human_json
    automation.steps_human_json = json.dumps(cleaned)
    await db.commit()
    sentence = "Rewrote the steps."
    recompiled = False
    try:
        from .recompiler import recompile_steps  # C's seam (§5.5)
        outcome = await recompile_steps(
            db, automation=automation, user_id=user_id, steps=cleaned,
        )
        recompiled = bool(outcome.get("recompiled"))
        sentence = outcome.get("sentence") or sentence
        if outcome.get("code"):
            # R38: a refusal must leave NOTHING applied — this file's
            # own contract for a 409 (`commit_workflow`'s docstring).
            # The wording persists above so a failed recompile can still
            # say "the wording is saved"; a REFUSAL is the other case,
            # and leaving the new sentences on screen under a "that
            # needs your yes" made the sheet show a plan the engine had
            # not accepted. The agent's edit tool would have reported
            # `not_changed` over a change it had in fact made.
            automation.steps_human_json = before_human
            await db.commit()
            raise WorkflowError(outcome["code"],
                                outcome.get("sentence") or "Refused.",
                                outcome.get("extra"))
    except ImportError:
        sentence = ("Rewrote the steps — the wording is saved; the plan "
                    "itself changes on the next update.")
    await _edited_note(db, automation)
    return {"steps": _steps_human(automation, _spec_raw(automation)),
            "sentence": sentence, "recompiled": recompiled}


async def add_rule(
    db: AsyncSession, *, automation: Automation, text: str,
    note: bool = True,
) -> dict:
    text = " ".join(str(text or "").split())[:300]
    if not text:
        raise WorkflowError("empty_rule", "Write the rule first.")
    rules = rules_list(automation)
    rule = {"id": str(uuid.uuid4()), "text": text,
            "added_at": datetime.utcnow().isoformat() + "Z"}
    rules.append(rule)
    automation.rules_json = json.dumps(rules)
    await db.commit()
    if note:
        # `commit_workflow` writes ONE note for the whole
        # commit — the user edited once, so the thread says
        # EDITED once, not once per draft.
        await _edited_note(db, automation)
    return {"rule": rule, "rules": rules,
            "sentence": f"Added a rule — {text[0].lower()}{text[1:]}"
            + ("" if text.endswith(".") else ".")}


async def update_rule(
    db: AsyncSession, *, automation: Automation, rule_id: str, text: str,
    note: bool = True,
) -> dict:
    rules = rules_list(automation)
    for r in rules:
        if r.get("id") == rule_id:
            r["text"] = " ".join(str(text or "").split())[:300]
            break
    else:
        raise WorkflowError("not_found", "That rule is gone.")
    automation.rules_json = json.dumps(rules)
    await db.commit()
    if note:
        # `commit_workflow` writes ONE note for the whole
        # commit — the user edited once, so the thread says
        # EDITED once, not once per draft.
        await _edited_note(db, automation)
    return {"rules": rules, "sentence": "Changed the rule."}


async def delete_rule(
    db: AsyncSession, *, automation: Automation, rule_id: str,
    note: bool = True,
) -> dict:
    before = rules_list(automation)
    rules = [r for r in before if r.get("id") != rule_id]
    # AUDIT-10: deleting a rule that does not exist used to answer 200
    # "Removed the rule." and stamp an EDITED note into the thread — a
    # record of an edit that never happened.
    if len(rules) == len(before):
        raise WorkflowError("not_found", "That rule is gone.")
    automation.rules_json = json.dumps(rules)
    await db.commit()
    if note:
        # `commit_workflow` writes ONE note for the whole
        # commit — the user edited once, so the thread says
        # EDITED once, not once per draft.
        await _edited_note(db, automation)
    return {"rules": rules, "sentence": "Removed the rule."}


async def save_permissions(
    db: AsyncSession, *, automation: Automation, user_id: str,
    account_id: str, can_ids: list[str], cant_ids: list[str],
    note: bool = True,
) -> dict:
    """The green ✓ (§4.4). The consent question is answered by the
    platform grant: allowing a write with no approved grant behind it
    returns needs_consent and the app runs the §3.7 flow first."""
    # AUDIT-2: this used to fall back to the connector's OAuth `scopes`
    # when the spec carried no grant_id. Every connected Slack has a
    # write scope, so the fallback made the green ✓ accept "Post as you"
    # for an automation with no grant at all — the consent flow this 409
    # exists to trigger simply never ran. The platform's grant row is
    # the only thing that can answer the question, and it fails closed.
    #
    # Close the read transaction FIRST. `_load_owned` autobegins one, and
    # `has_approved_write_grant` awaits an httpx GET per write grant id
    # at a 10s timeout — so the green ✓ was holding a database
    # connection open across N sequential network calls. That is the same
    # rule as "never await an LLM call inside an open transaction",
    # which has cost this codebase connection-pool exhaustion before;
    # HTTP is not different because it is faster on a good day. Nothing
    # is pending here, so this ends the read txn rather than writing
    # anything, and `expire_on_commit=False` keeps `automation` usable.
    await db.commit()
    has_grant = await permissions.has_approved_write_grant(
        automation=automation, user_id=user_id, connector_id=account_id,
    )
    try:
        result = await permissions.save(
            db, automation=automation, account_id=account_id,
            can_ids=can_ids, cant_ids=cant_ids,
            has_write_grant=has_grant,
        )
    except permissions.PermissionError409 as e:
        raise WorkflowError(e.code, e.sentence, e.extra)
    if note:
        # `commit_workflow` writes ONE note for the whole
        # commit — the user edited once, so the thread says
        # EDITED once, not once per draft.
        await _edited_note(db, automation)
    return result


# ---------------------------------------------------------- focus pins

async def _write_focus(
    db: AsyncSession, *, automation: Automation, user_id: str,
    account_id: str, pins: list[dict], sentence: str, note: bool = True,
) -> dict:
    """Persist one account's pins through the SAME spec write path
    every other structural edit uses (`service.update_automation`), so
    a pin is revalidated, recompiled and re-armed exactly like a
    schedule change — never poked into `spec_json` behind the
    validator's back."""
    from . import service
    raw = _spec_raw(automation)
    focus = {k: list(v) for k, v in focus_of(raw).items()}
    if pins:
        focus[account_id] = pins
    else:
        focus.pop(account_id, None)
    if focus:
        raw["focus"] = focus
    else:
        raw.pop("focus", None)
    try:
        automation, _vspec = await service.update_automation(
            db, automation_id=automation.id, user_id=user_id, spec=raw,
        )
    except Exception as e:  # noqa: BLE001 — surfaced, never swallowed
        from .spec import SpecError
        if isinstance(e, SpecError):
            raise WorkflowError(
                "bad_focus", "I could not start it there.",
                {"errors": e.errors},
            ) from e
        raise
    if note:
        await _edited_note(db, automation)
    await db.refresh(automation)
    return {
        "focus": focus_of(_spec_raw(automation)).get(account_id) or [],
        "sentence": sentence,
        "workflow": await workflow_payload(
            db, automation=automation, user_id=user_id,
        ),
    }


async def add_focus(
    db: AsyncSession, *, automation: Automation, user_id: str,
    account_id: str, kind: str, target_id: str, label: str = "",
) -> dict:
    """Pin one place under an account — the automation starts there.

    Membership is the gate HERE rather than in the validator: a pin
    under an account this automation does not use is a user error with
    a sentence, not a malformed spec. Re-pinning the same place is a
    no-op with the same sentence, because a user who taps twice meant
    it once.
    """
    from .spec import FOCUS_KINDS, MAX_FOCUS_PER_ACCOUNT
    raw = _spec_raw(automation)
    if account_id not in _member_connectors(raw):
        raise WorkflowError(
            "not_member",
            "This automation does not use that account yet — add it "
            "first, then pick where it starts.",
        )
    if kind not in FOCUS_KINDS:
        raise WorkflowError(
            "bad_focus_kind", "I do not know how to start there.")
    target_id = str(target_id or "").strip()
    if not target_id:
        raise WorkflowError("bad_focus_id", "Pick a place first.")
    name = verbs.display_name(account_id) or account_id
    pins = list(focus_of(raw).get(account_id) or [])
    if any(p["kind"] == kind and p["id"] == target_id for p in pins):
        return {
            "focus": pins, "workflow": await workflow_payload(
                db, automation=automation, user_id=user_id,
            ),
            "sentence": f"It already starts at "
                        f"{label or target_id} in {name}.",
        }
    if len(pins) >= MAX_FOCUS_PER_ACCOUNT:
        raise WorkflowError(
            "too_many_focus",
            f"It can start in {MAX_FOCUS_PER_ACCOUNT} places in {name} "
            f"at most. Take one out first.",
        )
    shown = str(label or "").strip() or target_id
    pins.append({"kind": kind, "id": target_id, "label": shown})
    return await _write_focus(
        db, automation=automation, user_id=user_id, account_id=account_id,
        pins=pins, sentence=f"It starts at {shown} in {name} now.",
    )


async def remove_focus(
    db: AsyncSession, *, automation: Automation, user_id: str,
    account_id: str, kind: str, target_id: str,
) -> dict:
    """Unpin one place. A pin that is already gone is a 409, not a
    silent success — the app would otherwise redraw a row it never
    removed."""
    raw = _spec_raw(automation)
    pins = list(focus_of(raw).get(account_id) or [])
    target_id = str(target_id or "").strip()
    kept = [p for p in pins
            if not (p["kind"] == kind and p["id"] == target_id)]
    if len(kept) == len(pins):
        raise WorkflowError("not_found", "It does not start there.")
    gone = next(p for p in pins
                if p["kind"] == kind and p["id"] == target_id)
    name = verbs.display_name(account_id) or account_id
    return await _write_focus(
        db, automation=automation, user_id=user_id, account_id=account_id,
        pins=kept,
        sentence=f"It no longer starts at {gone['label']} in {name}.",
    )


# ------------------------------------------------------------ composer

_UNDO: dict[str, dict] = {}
_UNDO_TTL_S = 12.0


def _mint_undo(kind: str, revert: dict) -> str:
    token = str(uuid.uuid4())
    _UNDO[token] = {"kind": kind, "revert": revert, "at": time.monotonic()}
    if len(_UNDO) > 64:
        cutoff = time.monotonic() - _UNDO_TTL_S
        for k in [k for k, v in _UNDO.items() if v["at"] < cutoff]:
            _UNDO.pop(k, None)
    return token


async def _apply_all(
    db: AsyncSession, *, automation: Automation, user_id: str,
    intents: list[dict],
) -> dict:
    """Apply a list of policy-approved intents. Returns
    `{"applied": [entry], "refused": [sentence]}`.

    AUDIT-6: every applier commits its own write, so letting a
    `WorkflowError` out of this loop left the earlier intents COMMITTED,
    answered the whole call 409, and skipped the `agent` turn the caller
    writes — a half-applied change with no record of itself in the
    thread. "Move it to 7:15 and add a rule" was enough to do it. A
    refusal is a sentence, not an abort: the safe siblings still land
    and the caller says what did not.
    """
    applied: list[dict] = []
    refused: list[str] = []
    for intent in intents:
        try:
            entry = await _apply_intent(
                db, automation=automation, user_id=user_id, intent=intent,
            )
        except WorkflowError as e:
            refused.append(e.sentence)
            continue
        except Exception as e:  # noqa: BLE001
            logger.warning("[workflow] intent %s failed: %s",
                           intent.get("kind"), e)
            refused.append("I could not make that one.")
            continue
        if entry is not None:
            applied.append(entry)
    return {"applied": applied, "refused": refused}


async def apply_intents(
    db: AsyncSession, *, automation: Automation, user_id: str,
    intents: list[dict],
) -> dict:
    """The agent's edit tools, and the only door they get.

    The five `automations__edit_*` tools hand STRUCTURED intents of
    composer.py's documented shape straight to this function — no
    classifier, because the agent already read the user's sentence.
    Everything after that is the sheet's own code: `apply_policy`
    decides what may be applied silently, `_apply_intent` writes it,
    each write mints the same undo token and stamps the same EDITED
    note through `_edited_note`, which is also what broadcasts
    `automation.updated`.

    So a tool can no more widen access than a sentence can: a grant, an
    account add or a hard rail comes back in `needs`/`answer` exactly as
    it does for the composer, and the tool tells the model to ask.

    Returns `{"applied", "needs", "answer", "refused", "workflow"}` —
    never a silent nothing: an intent the writers could not make lands
    in `refused` carrying the sentence that says why. `workflow` is the
    payload AFTER the writes, so the caller can say what it looks like
    now and the model has the ids for its next edit without a second
    read.
    """
    from .composer import _normalize_intent, apply_policy

    wf = await workflow_payload(db, automation=automation, user_id=user_id)
    outcome = apply_policy(
        [_normalize_intent(i) for i in intents if isinstance(i, dict)], wf,
    )
    landed = await _apply_all(
        db, automation=automation, user_id=user_id,
        intents=outcome.get("applied") or [],
    )
    after = wf
    if landed["applied"]:
        await db.refresh(automation)
        after = await workflow_payload(
            db, automation=automation, user_id=user_id,
        )
    return {
        "applied": landed["applied"],
        "needs": outcome.get("needs") or [],
        "answer": outcome.get("answer"),
        "refused": landed["refused"],
        "workflow": after,
        "workflow_rev": int(getattr(automation, "workflow_rev", 0) or 0),
    }


async def composer_ask(
    db: AsyncSession, *, automation: Automation, user_id: str, text: str,
) -> dict:
    """§4.4: classify → apply the safe intents → thread record.
    Anything granting access or changing an output target comes back in
    `needs`, never applied silently."""
    text = " ".join(str(text or "").split())
    if not text:
        raise WorkflowError("empty", "Say what to change.")

    thread = await ledger.ensure_thread(
        db, user_id=automation.user_id, automation_id=automation.id,
    )
    await ledger.append_turn(
        db, user_id=automation.user_id, thread=thread, run_id=None,
        kind="user", payload={"text": text},
    )

    applied: list[dict] = []
    needs: list[dict] = []
    refused: list[str] = []
    answer: Optional[str] = None
    try:
        from .composer import classify_change
        wf = await workflow_payload(db, automation=automation,
                                    user_id=user_id)
        outcome = await classify_change(text, wf)
        needs = outcome.get("needs") or []
        answer = outcome.get("answer")
        landed = await _apply_all(
            db, automation=automation, user_id=user_id,
            intents=outcome.get("applied") or [],
        )
        applied = landed["applied"]
        refused = landed["refused"]
    except ImportError:
        # C's classifier not merged yet — one honest fallback: treat the
        # sentence as a rule (the only always-safe change).
        result = await add_rule(db, automation=automation, text=text)
        applied.append({
            "kind": "rule", "sentence": result["sentence"],
            "sheet": "rules",
            "undo_token": _mint_undo(
                "rule", {"rule_id": result["rule"]["id"]}),
        })
    except WorkflowError:
        raise
    except Exception as e:  # noqa: BLE001
        logger.warning("[workflow] composer failed: %s", e)
        answer = ("I could not place that change. Tell me in the thread "
                  "and I will do it there.")

    if refused:
        # §4.4: `answer` is the slot for "a sentence the agent cannot
        # place". A refusal is exactly that — it is not a `needs` entry,
        # whose kinds are fixed at consent|confirm.
        #
        # This was `if refused and not answer`, which dropped the refusal
        # whenever the classifier had ALSO said something — e.g. "never
        # let slack read private DMs, and move it to 7:15": the rail
        # remark fills `answer`, the schedule intent is refused by the
        # writer, and the user was told only about the rail while the
        # time silently did not move. Both sentences are true; the user
        # gets both.
        answer = f"{answer} {' '.join(refused)}".strip() if answer \
            else " ".join(refused)
    if applied:
        confirmation = " ".join(a["sentence"] for a in applied)
        if refused:
            confirmation = f"{confirmation} {' '.join(refused)}"
    else:
        confirmation = answer or (
            "That needs your say-so first — nothing was changed."
            if needs else
            "I could not place that change. Tell me in the thread and I "
            "will do it there."
        )
    await ledger.append_turn(
        db, user_id=automation.user_id, thread=thread, run_id=None,
        kind="agent", payload={"text": confirmation},
    )
    return {"applied": applied, "needs": needs, "answer": answer}


async def _apply_intent(
    db: AsyncSession, *, automation: Automation, user_id: str, intent: dict,
) -> Optional[dict]:
    """Apply ONE policy-approved intent, or raise `WorkflowError` with
    the sentence that says why not.

    R38: every branch that used to `return None` now raises instead.
    `None` was a silent no-op — `composer_ask` appended it to nothing
    and said nothing about it, so "change step 4" on a three-step
    automation, or "stop Slack posting" for a permission that was
    already off, came back as a confirmation of the OTHER intents in
    the same sentence and no word about the one that did nothing. The
    agent's edit tools would have inherited exactly that: a tool that
    reports success for a change it did not make. The only surviving
    `None` is the unreachable tail, which raises too.
    """
    kind = intent.get("kind")
    if kind == "rule":
        op = str(intent.get("op") or "add").strip().lower()
        if op == "add":
            result = await add_rule(db, automation=automation,
                                    text=intent.get("text") or "")
            return {"kind": "rule", "sentence": result["sentence"],
                    "sheet": "rules",
                    "undo_token": _mint_undo(
                        "rule", {"rule_id": result["rule"]["id"]})}
        # remove/edit revert through the WHOLE list rather than a
        # single id: putting a deleted rule back at its original index
        # with its original id is what keeps a second undo token (or a
        # later edit) pointing at the same row.
        before = [dict(r) for r in rules_list(automation)]
        rule_id = str(intent.get("rule_id") or "")
        if op == "remove":
            result = await delete_rule(db, automation=automation,
                                       rule_id=rule_id)
        else:
            result = await update_rule(db, automation=automation,
                                       rule_id=rule_id,
                                       text=intent.get("text") or "")
        return {"kind": "rule", "sentence": result["sentence"],
                "sheet": "rules",
                "undo_token": _mint_undo("rule_set", {"rules": before})}
    if kind == "schedule":
        raw_before = _spec_raw(automation)
        before = _schedule_of(raw_before) or {}
        custom = intent.get("custom")
        if isinstance(custom, dict) and custom:
            result = await set_schedule_custom(
                db, automation=automation, user_id=user_id, custom=custom,
            )
        else:
            result = await set_schedule_preset(
                db, automation=automation, user_id=user_id,
                preset_id=intent.get("preset_id") or "",
            )
        return {"kind": "schedule", "sentence": result["sentence"],
                "sheet": "schedule",
                "undo_token": _mint_undo("schedule", {"schedule": before})}
    if kind == "step":
        steps = _steps_human(automation, _spec_raw(automation))
        n = int(intent.get("n") or 0)
        if not (1 <= n <= len(steps)):
            raise WorkflowError(
                "no_such_step",
                f"There is no step {n} — this one has "
                f"{len(steps)} of them." if steps else
                "This one has no steps yet.")
        before = [dict(s) for s in steps]
        steps[n - 1]["text"] = str(intent.get("text") or "")[:200]
        result = await set_steps(db, automation=automation,
                                 user_id=user_id, steps=steps)
        return {"kind": "step", "sentence": result["sentence"],
                "sheet": "steps",
                "undo_token": _mint_undo("step", {"steps": before})}
    if kind == "permission":
        # Only REMOVALS apply silently; allows go through needs.
        # AUDIT-1: the classifier emits `permission_id`; this read
        # `remove_id` and silently dropped every revoke — "slack must
        # never post as me" answered "Added a rule …" while the account
        # kept the permission. `remove_id` stays as an alias so an older
        # caller keeps working.
        account_id = intent.get("account_id") or ""
        remove_id = (intent.get("permission_id")
                     or intent.get("remove_id") or "")
        current = await permissions.resolve(
            db, automation=automation, account_id=account_id,
        )
        before_can = [p["id"] for p in current["can"]]
        if remove_id not in before_can:
            name = verbs.display_name(account_id) or account_id or "It"
            label = next(
                (p.get("label") for p in current["cant"]
                 if p.get("id") == remove_id), None)
            raise WorkflowError(
                "not_granted",
                f"{name} already cannot "
                f"{label[0].lower()}{label[1:]}." if label else
                f"{name} does not have that permission to take away.")
        can = [p for p in before_can if p != remove_id]
        cant = [p["id"] for p in current["cant"]
                if p.get("kind") == "ungranted"] + [remove_id]
        result = await save_permissions(
            db, automation=automation, user_id=user_id,
            account_id=account_id, can_ids=can, cant_ids=cant,
        )
        del result
        return {"kind": "permission",
                "sentence": intent.get("sentence")
                or "Took that permission away.",
                "sheet": f"account:{account_id}",
                "undo_token": _mint_undo(
                    "permission",
                    {"account_id": account_id, "can": before_can})}
    if kind == "account":
        # AUDIT-5: the classifier emits account add/remove; there was no
        # branch, so "take Jira out of this automation" reported nothing
        # wrong while the account stayed wired in. Only REMOVAL applies
        # silently — an add grants new reach and belongs in `needs`.
        if (intent.get("direction") or "remove") != "remove":
            raise WorkflowError(
                "needs_consent",
                "Adding an account is the user's call — it is not "
                "something I apply on my own.")
        account_id = intent.get("account_id") or ""
        raw_before = _spec_raw(automation)
        from . import service as _svc
        name = verbs.display_name(account_id) or account_id
        if account_id not in _member_connectors(raw_before):
            raise WorkflowError(
                "not_member",
                f"{name} is not part of this automation, so there is "
                f"nothing to take out.")
        was_armed = (automation.status == "armed")
        try:
            await _svc.remove_connector(
                db, automation_id=automation.id, user_id=user_id,
                connector_id=account_id,
            )
        except _svc.MembershipError as e:
            # Refuse out loud. The spec says why it cannot absorb the
            # removal; silence here reads as "done" to the user.
            if e.code == "not_member":
                raise WorkflowError(
                    "not_member",
                    f"{name} is not part of this automation, so there "
                    f"is nothing to take out.")
            raise WorkflowError(e.code, {
                "connector_required":
                    f"{name} is doing work this automation depends on, "
                    f"so I left it in.",
                "not_supported_v1":
                    "This automation needs both of its accounts.",
            }.get(e.code, f"I could not take {name} out."))
        await _edited_note(db, automation)
        # `update_automation` drops the automation to `draft`, commits,
        # and only RE-ARMS on a best-effort basis — a CompileError or
        # SpecError there is logged, not raised. So a removal can leave
        # an automation that was running silently stopped, and reporting
        # "Took Jira out of this automation." alone would be a tidy
        # sentence over a change the user did not ask for and cannot
        # see. Say the part that matters.
        await db.refresh(automation)
        stopped = was_armed and automation.status != "armed"
        sentence = (intent.get("sentence")
                    or f"Took {name} out of this automation.")
        if stopped:
            sentence = (f"Took {name} out, and this automation stopped "
                        f"running — it needs setting up again.")
        return {"kind": "account",
                "sentence": sentence,
                "sheet": "accounts",
                # The revert carries the WHOLE pre-removal spec, not just
                # the id: `service.add_connector` rebuilds a connector's
                # read presence from the automation's TEMPLATE, and a
                # chat-built automation has `template_slug` NULL — it
                # raises `no_template_step`. Undo cannot depend on a
                # template the automation never had.
                "undo_token": _mint_undo(
                    "account",
                    {"account_id": account_id, "spec": raw_before})}
    raise WorkflowError(
        "unknown_change",
        "I could not place that change — tell me which part to move "
        "and I will do it.")


async def composer_undo(
    db: AsyncSession, *, automation: Automation, user_id: str, token: str,
) -> dict:
    entry = _UNDO.pop(token, None)
    if entry is None or time.monotonic() - entry["at"] > _UNDO_TTL_S:
        raise WorkflowError("undo_expired", "That change already settled.")
    kind, revert = entry["kind"], entry["revert"]
    if kind == "rule":
        await delete_rule(db, automation=automation,
                          rule_id=revert["rule_id"])
    elif kind == "rule_set":
        # A remove or an edit: the whole list goes back, ids and order
        # intact, so a rule that was deleted returns as the SAME row a
        # later token or edit still points at.
        automation.rules_json = json.dumps(revert.get("rules") or [])
        await db.commit()
    elif kind == "schedule":
        from . import service
        sched = {k: v for k, v in (revert.get("schedule") or {}).items()
                 if k in ("cron_local", "at", "every_s") and v}
        if sched:
            await service.set_schedule(
                db, user_id=user_id, automation_id=automation.id,
                schedule=sched,
            )
    elif kind == "step":
        automation.steps_human_json = json.dumps(revert.get("steps") or [])
        await db.commit()
    elif kind == "permission":
        current = await permissions.resolve(
            db, automation=automation, account_id=revert["account_id"],
        )
        cant = [p["id"] for p in current["cant"]
                if p.get("kind") == "ungranted"
                and p["id"] not in revert["can"]]
        await save_permissions(
            db, automation=automation, user_id=user_id,
            account_id=revert["account_id"],
            can_ids=revert["can"], cant_ids=cant,
        )
    elif kind == "account":
        # AUDIT-5 added the account REMOVAL but not its undo, so an
        # `account` token fell past every branch to the unconditional
        # `{"undone": True}` below: the button reported success, stamped
        # an EDITED note for an edit that never happened, and left the
        # connector removed. Exactly the shape AUDIT-10 fixed in
        # `delete_rule` — reintroduced two hundred lines away.
        from . import service
        spec = revert.get("spec") or {}
        if not spec:
            raise WorkflowError(
                "undo_unavailable",
                "I could not put that account back — open the accounts "
                "sheet and add it there.")
        await service.update_automation(
            db, automation_id=automation.id, user_id=user_id, spec=spec,
        )
    else:
        # Never answer "undone" for a kind this function cannot revert.
        # Silence here is how the account token lied for a whole round.
        raise WorkflowError(
            "undo_unavailable", "That change cannot be undone from here.")
    await _edited_note(db, automation)
    return {"undone": True}
