"""The whole workflow — one GET, per-sheet writes, the composer (R30 §4.4).

Everything the canvas and its sheets read comes from ONE call
(`workflow_payload`), resolved from one source of truth per concern:
the spec (steps, sources, mode), `permissions.py` (the per-automation
account permissions), the platform connection state (account states),
`rules_json` (the user's standing rules), and the verb dictionary
(every human string). Writes are per sheet, never one big PUT; every
workflow write stamps an `EDITED` note turn on the automation's
thread so the thread stays the full record (writes inside one minute
share the one note — see `_edited_note`).

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

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Automation, AutomationTurn
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
#
# R39: the subs describe the CRON, not a use case. "Finishes before your
# first meeting" / "Right before standup" were the Morning brief's own
# copy, served to every automation — a repo digest's schedule sheet was
# selling commute times (founder P13).
SCHEDULE_PRESETS = (
    {"id": "weekdays-8", "cron_local": "0 8 * * 1-5",
     "sentence": "Weekdays at 8:00",
     "sub": "Monday to Friday"},
    {"id": "weekdays-730", "cron_local": "30 7 * * 1-5",
     "sentence": "Weekdays at 7:30",
     "sub": "Monday to Friday, earlier"},
    {"id": "daily-8", "cron_local": "0 8 * * *",
     "sentence": "Every morning at 8:00", "sub": "Weekends included"},
    {"id": "weekdays-9", "cron_local": "0 9 * * 1-5",
     "sentence": "Weekdays at 9:00", "sub": "Monday to Friday"},
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


def _event_source_of(raw: dict) -> Optional[dict]:
    """The first push/poll source carrying an event key, v1 or v2."""
    if raw.get("version") == 2:
        for s in (raw.get("trigger") or {}).get("sources") or []:
            if isinstance(s, dict) and s.get("mode") in ("push", "poll") \
                    and s.get("event"):
                return s
        return None
    trig = raw.get("trigger") or {}
    if trig.get("mode") in ("push", "poll") and trig.get("event"):
        return trig
    return None


def trigger_block(raw: dict) -> dict:
    """R39 — the trigger's OWN vocabulary (founder P12).

    The payload used to carry only `schedule`, so an event automation
    ("New event → Slack", trigger poll/event_created) rendered as "On
    its own schedule", its sheet offered four cron presets, and saving
    one silently REPAINTED the event automation as a daily schedule.
    The canvas must speak each automation's own trigger; the app
    branches on `kind` and never shows cron UI for an event.

    R42: the SCHEDULE wins when a spec has both. §5.3 lets a user add
    an instant lane to a scheduled automation, and reading the event
    first would have repainted a daily brief as an event automation and
    hidden its cron UI the moment they tapped one row — P12's own
    defect, in the other direction. No shipped template carries both;
    an event-only spec has no schedule, so it is unaffected.
    """
    if _schedule_of(raw):
        sentence = _current_sentence(raw)
        return {"kind": "schedule", "label": _label_of(sentence),
                "sub": "", "event": None}
    ev = _event_source_of(raw)
    if ev is not None:
        key = str(ev.get("event") or "")
        conn = ev.get("connector_id")
        clause = verbs._EVENT_CLAUSES.get(key) or "when something new lands"
        tag = verbs.event_tag(key) or "on new activity"
        name = verbs.display_name(conn) or ""
        return {
            "kind": "event",
            "label": tag[0].upper() + tag[1:],
            "sub": f"watches {name}" if name else "watches for it",
            "event": {
                "key": key,
                "connector_id": conn,
                "sentence": clause[0].upper() + clause[1:],
            },
        }
    return {"kind": "manual", "label": "On request",
            "sub": "runs when you ask", "event": None}


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


def run_blockers(raw: dict) -> list[dict]:
    """Why this spec cannot fire yet — the SPEC-side truth, in one place.

    R39: run-now's gate, the thread agent's grounding and the setup
    copy each hand-maintained their own version of "can it run?", and
    they drifted: the thread promised "First run is soon. The post will
    be waiting for your edit" about an automation whose write step
    run-now refused every single time (founder P6 — three surfaces,
    two answers). Every surface that speaks about runnability reads
    THIS. The route adds its async grant-STATUS refinement on top;
    everything here is decidable from the spec alone.
    """
    out: list[dict] = []
    if raw.get("version") != 2:
        return out
    for st in raw.get("steps") or []:
        if st.get("kind") == "agent":
            continue
        tool = str(st.get("tool") or "")
        if not (st.get("grant_id") or verbs.is_write_tool(tool)):
            continue
        params = st.get("params_template") or st.get("params") or {}
        needs_pin = ("{{grant.target." in json.dumps(params)
                     and not (st.get("grant_target") or {}).get("id"))
        if needs_pin or not st.get("grant_id"):
            clause = verbs._WRITE_CLAUSES.get(tool) or "write"
            out.append({
                "code": "needs_destination",
                "connector_id": st.get("connector_id"),
                "tool": tool,
                "sentence": (f"before it can {clause}, tell me where "
                             f"that should go and I will pin it"),
            })
    return out


def focus_of(raw: dict) -> dict[str, list[dict]]:
    """The spec's per-account pins,
    `{connector_id: [{kind,id,label[,note]}]}`.

    Total: a spec written before R38 has no `focus` key and answers
    `{}`, which is the same thing as "nothing pinned" — there is no
    third state to distinguish, because a pin is a user action and its
    absence means only that they have not taken it.

    R39: `note` rides through. This normalizer used to rebuild pins as
    exactly {kind,id,label}, and BOTH writers rebuild the persisted list
    through it — so the user's instruction survived only until the next
    pin or unpin anywhere on the automation, and the payload never
    served it back for the sheet to edit.
    """
    focus = raw.get("focus")
    if not isinstance(focus, dict):
        return {}
    out: dict[str, list[dict]] = {}
    for cid, pins in focus.items():
        if not isinstance(cid, str) or not isinstance(pins, list):
            continue
        rows = []
        for p in pins:
            if not (isinstance(p, dict) and p.get("id") and p.get("kind")):
                continue
            row = {"kind": str(p.get("kind") or ""),
                   "id": str(p.get("id") or ""),
                   "label": str(p.get("label") or p.get("id") or "")}
            if p.get("note"):
                row["note"] = str(p["note"])
            rows.append(row)
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
    # R39: the brief row only for automations that actually DELIVER a
    # brief here. It was unconditional, so "New event → Slack" — whose
    # only output is a Slack post — opened "Where it lands" selling "A
    # brief on your phone — Ranked by what breaks if you ignore it",
    # the Morning brief's own copy (founder P13). Every sheet speaks
    # its automation's own spec.
    lines: list[dict] = []
    if mode in ("reads_only", "drafts_only"):
        lines.append({
            "title": "A brief on your phone",
            "body": "It lands in this thread; nothing is sent on your "
                    "behalf.",
        })
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
    # Stored rows are the USER's own words (`set_steps` is the only
    # writer) — served verbatim, never re-tensed. The plan-tense pass
    # below applies only to the DERIVED branches, whose strings come
    # from the verb dictionary's past tense (founder P18: "Checked your
    # calendar" on an automation that has never run).
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
            # R42 (B6): the same predicate `_write_tools` above spells
            # out — a write is a write by its TOOL. Keyed on grant
            # presence, this sheet described every UNGRANTED template
            # write as a read: an unpinned Morning brief said "Checks
            # Slack" about the step whose whole job is to POST there,
            # and it was true for all fifteen v2 templates.
            is_write = bool(s.get("grant_id")) or verbs.is_write_tool(tool)
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
    # Re-tense the whole plan in one place (see the stored branch above).
    return [{**s, "text": verbs.plan_action(s["text"])} for s in out]


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
    capability = await reg.fetch_registry(user_id)
    members = _member_connectors(raw)
    mode, _mode_label = mode_of(automation, raw)

    pinned = focus_of(raw)
    stored_filters = filters_of(raw)
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
        # R42 §5.2 / §5.3 — what this account is narrowed by, and what
        # it can announce the moment it happens. Beside `focus` for the
        # same reason: the canvas draws one node per account, and a
        # node must not have to join two lists to know its own state.
        # All four are served as `[]` when empty, and NEVER omitted: the app
        # distinguishes an empty list from an absent key, because §5.3 renders
        # a SENTENCE for the empty case ("nothing it can tell you the moment it
        # happens") and that is a claim only this payload can support. Dropping
        # a key here would make the app state it against no evidence.
        # Empty filters draw no §5.2 section at all; empty triggers draw §5.3
        # with the sentence — a picker asserts nothing by being absent, prose
        # does.
        entry["filters"] = stored_filters.get(cid) or []
        entry["filters_available"] = available_filters(raw, cid)
        entry["triggers"] = triggers_of(raw, cid)
        entry["triggers_available"] = available_triggers(raw, capability, cid)
        accounts.append(entry)

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
        # R39: the trigger's own vocabulary. `schedule` above stays for
        # older clients; new ones branch on trigger.kind and only show
        # cron UI for kind == "schedule".
        "trigger": trigger_block(raw),
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


# R42 (P9): one edit gets one divider. A rolling window rather than a
# clock minute, so two writes two seconds apart do not become two
# dividers because one of them fell the other side of :00.
_EDITED_COLLAPSE_S = 60.0


def _edited_stamp_at(row: Optional[AutomationTurn]) -> Optional[datetime]:
    """When an `edited` note says it happened, or None if `row` is not
    one. The payload's own `at` is the answer the client renders —
    `ledger._serialize_row` spreads the body OVER the row's
    `created_at` — so a collapse decision has to read the same field it
    is about to rewrite."""
    if row is None or row.kind != "note":
        return None
    try:
        body = json.loads(row.payload_json or "{}") or {}
    except (ValueError, TypeError):
        body = {}
    if not isinstance(body, dict) or body.get("stamp") != "edited":
        return None
    raw = str(body.get("at") or "")
    if raw:
        try:
            return datetime.fromisoformat(raw.rstrip("Z"))
        except ValueError:
            pass
    return row.created_at


async def _edited_note(
    db: AsyncSession, automation: Automation,
) -> Optional[str]:
    """Every workflow write stamps the EDITED note (§4.4) and
    broadcasts `automation.updated` (§4.6).

    One seam for both, because they are one fact: the workflow changed.
    R31-11 is what happens when the second half is missing — the
    founder removed Outlook in the Workflow, came back to the thread,
    and the header still showed five chips while the ⋯ menu still said
    `5 accounts`, because those two surfaces read a summary nobody had
    told. Every writer in this module already calls this function, so
    putting the broadcast here means a new writer cannot forget it.

    R42 (P9): it APPENDS only when the thread does not already end in a
    fresh EDITED note. One user action is routinely several workflow
    writes — a pin, then the grant approval that follows it — and the
    founder's thread carried three stacked dividers, two of them
    identical and back to back with nothing between them. A repeat
    rewrites the standing note in place; `ledger.replace_turn` keeps the
    turn's id and seq, so the `automation.turn` frame repaints that row
    instead of the client gaining a second one.
    """
    try:
        thread = await ledger.ensure_thread(
            db, user_id=automation.user_id, automation_id=automation.id,
        )
        now = datetime.utcnow()
        payload = {"stamp": "edited", "at": now.isoformat() + "Z"}
        last = (await db.execute(
            select(AutomationTurn)
            .where(AutomationTurn.thread_id == thread.id)
            .order_by(AutomationTurn.seq.desc())
            .limit(1)
        )).scalars().first()
        prev = _edited_stamp_at(last)
        if prev is not None and 0 <= (now - prev).total_seconds() \
                <= _EDITED_COLLAPSE_S:
            turn = await ledger.replace_turn(
                db, user_id=automation.user_id, thread=thread,
                turn_id=last.id, kind="note", payload=payload,
            )
        else:
            turn = await ledger.append_turn(
                db, user_id=automation.user_id, thread=thread, run_id=None,
                kind="note", payload=payload,
            )
        # `replace_turn` answers None when the turn it was given is
        # already gone — a race with a delete, not an error.
        turn_id = turn["id"] if turn else None
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

async def _persist_spec(
    db: AsyncSession, *, automation: Automation, user_id: str, raw: dict,
    code: str, refusal: str, note: bool = True,
) -> Automation:
    """Persist an edited spec through the SAME write path every other
    structural edit uses (`service.update_automation`), so an edit is
    revalidated, recompiled and re-armed exactly like a schedule
    change — never poked into `spec_json` behind the validator's back.

    One note per write, stamped here, so a caller cannot forget it and
    two callers cannot stamp two dividers for one edit.
    """
    from . import service
    try:
        automation, _vspec = await service.update_automation(
            db, automation_id=automation.id, user_id=user_id, spec=raw,
        )
    except Exception as e:  # noqa: BLE001 — surfaced, never swallowed
        from .spec import SpecError
        if isinstance(e, SpecError):
            raise WorkflowError(code, refusal, {"errors": e.errors}) from e
        raise
    if note:
        await _edited_note(db, automation)
    await db.refresh(automation)
    return automation


async def _write_focus(
    db: AsyncSession, *, automation: Automation, user_id: str,
    account_id: str, pins: list[dict], sentence: str, note: bool = True,
) -> dict:
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
    automation = await _persist_spec(
        db, automation=automation, user_id=user_id, raw=raw,
        code="bad_focus", refusal="I could not start it there.", note=note,
    )
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
    note: Optional[str] = None,
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
    from .spec import FOCUS_NOTE_MAX
    # None = no note intent (a bare "+" tap must never clear anything);
    # "" = an explicit clear from the instruction sheet.
    if note is not None:
        note = str(note).strip()[:FOCUS_NOTE_MAX]
    pins = list(focus_of(raw).get(account_id) or [])
    existing = next((p for p in pins
                     if p["kind"] == kind and p["id"] == target_id), None)
    if existing is not None:
        # R39: re-pinning the same place with a NOTE is how the app
        # writes/edits the per-pin instruction — that is an update,
        # not a double tap.
        if note is not None and note != (existing.get("note") or ""):
            updated = [dict(p) for p in pins]
            for p in updated:
                if p["kind"] == kind and p["id"] == target_id:
                    if note:
                        p["note"] = note
                    else:
                        p.pop("note", None)
            shown = existing.get("label") or target_id
            return await _write_focus(
                db, automation=automation, user_id=user_id,
                account_id=account_id, pins=updated,
                sentence=(f"Noted — I will treat {shown} that way."
                          if note else
                          f"Cleared — {shown} is a plain pin again."),
            )
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
    new_pin: dict = {"kind": kind, "id": target_id, "label": shown}
    if note:  # None and "" both mean: a new pin starts plain
        new_pin["note"] = note
    pins.append(new_pin)
    out = await _write_focus(
        db, automation=automation, user_id=user_id, account_id=account_id,
        pins=pins, sentence=f"It starts at {shown} in {name} now.",
    )
    # R39: a channel/chat pinned under the connector that still owes a
    # WRITE DESTINATION sets that too — the founder followed "pick
    # all-toup and I'll set it there" by tapping "+" on all-toup, and a
    # bare focus left the run refusing about the very thing he had just
    # picked. `only_if_unpinned`: "+" never redirects an approved
    # destination. A failed grant does not poison the focus pin; the
    # sentence carries what happened either way.
    if _pin_names_a_destination(account_id, new_pin):
        try:
            dest = await pin_write_destination(
                db, automation=automation, user_id=user_id,
                connector_id=account_id,
                target={"kind": kind, "id": target_id, "label": shown},
                only_if_unpinned=True,
            )
        except Exception as e:  # noqa: BLE001 — the pin stood; the
            # destination half can be asked for again.
            logger.warning("[automations] focus destination bridge "
                           "failed automation=%s: %s", automation.id, e)
            dest = None
        if dest is not None:
            out = dict(out)
            out["sentence"] = (f"It starts at {shown} in {name} now. "
                               f"{dest['sentence']}")
            out["destination"] = dest
            await db.refresh(automation)
            out["workflow"] = await workflow_payload(
                db, automation=automation, user_id=user_id,
            )
    return out


# ── R42 §5.2 — the account's read filters ────────────────────────────

def filters_of(raw: dict) -> dict[str, list[str]]:
    """The spec's per-account filter ids, `{connector_id: [id]}`.

    Total, like `focus_of`: a spec written before R42 has no `filters`
    key and answers `{}` — "nothing narrowed", which is the same thing
    and has no third state.
    """
    filters = raw.get("filters")
    if not isinstance(filters, dict):
        return {}
    out: dict[str, list[str]] = {}
    for cid, ids in filters.items():
        if not isinstance(cid, str) or not isinstance(ids, list):
            continue
        kept = [str(i) for i in ids if isinstance(i, str) and i]
        if kept:
            out[cid] = kept
    return out


def _account_read_tools(raw: dict, connector_id: str) -> set:
    """The read tools this automation actually runs on that account."""
    if raw.get("version") != 2:
        return set()
    return {
        str(st.get("tool") or "")
        for st in raw.get("steps") or []
        if isinstance(st, dict)
        and st.get("connector_id") == connector_id
        and st.get("kind") != "agent"
        and not verbs.is_write_tool(st.get("tool"))
    }


def available_filters(raw: dict, connector_id: str) -> list[dict]:
    """`[{id, label}]` — the chips this account can honestly offer.

    Two gates, and the second is the one that matters: the connector
    has to be able to EXPRESS the filter (`spec.CONNECTOR_FILTERS`),
    and this automation has to run a step the filter composes into. An
    account here only to draft mail narrows nothing, so it offers
    nothing — the app then draws no section at all, which is the
    correct rendering and the whole reason the list is optional.

    A filter already ON is always offered, whatever the steps say now:
    a step edited after the fact must never leave a stored filter the
    user cannot see or take off.
    """
    from .spec import filter_options
    tools = _account_read_tools(raw, connector_id)
    on = set(filters_of(raw).get(connector_id) or [])
    return [
        {"id": f["id"], "label": f["label"]}
        for f in filter_options(connector_id)
        if f["id"] in on or (tools & set(f.get("tools") or ()))
    ]


async def set_filters(
    db: AsyncSession, *, automation: Automation, user_id: str,
    connector_id: str, filters: list,
) -> dict:
    """Replace one account's read filters (design §5.2).

    A whole set rather than a toggle, because that is what the chips
    are: the app sends the state it drew, so two quick taps cannot
    interleave into a set neither of them meant.

    Membership is the gate here for the same reason it is in
    `add_focus` — a filter under an account this automation does not
    use is a user error with a sentence, not a malformed spec.
    """
    from .spec import filter_ids, filter_options
    raw = _spec_raw(automation)
    name = verbs.display_name(connector_id) or connector_id
    # v2 only, like `set_triggers`: the v1 executor reads neither `focus`
    # nor `filters`, so a filter stored on a v1 spec is consumed by nothing
    # and can never be un-lit by anything but a second write. It is
    # unreachable from the app today only because `available_filters`
    # answers [] for v1 — an accident, not a guard.
    if raw.get("version") != 2:
        raise WorkflowError(
            "not_supported",
            f"This automation is too old to narrow {name}. Ask me to "
            f"rebuild it and I will set it up the new way.",
        )
    if connector_id not in _member_connectors(raw):
        raise WorkflowError(
            "not_member",
            f"This automation does not use {name} yet — add it first, "
            f"then narrow it.",
        )
    known = filter_ids(connector_id)
    wanted = set()
    for fid in (filters or []):
        if not isinstance(fid, str) or fid not in known:
            raise WorkflowError(
                "unknown_filter",
                f"{name} cannot narrow a read that way.",
            )
        wanted.add(fid)
    kept = [f["id"] for f in filter_options(connector_id) if f["id"] in wanted]

    stored = {k: list(v) for k, v in filters_of(raw).items()}
    if kept:
        stored[connector_id] = kept
    else:
        stored.pop(connector_id, None)
    if stored:
        raw["filters"] = stored
    else:
        raw.pop("filters", None)

    labels = [f["label"] for f in filter_options(connector_id)
              if f["id"] in wanted]
    sentence = (f"In {name} it now reads {verbs.join_list(labels)}."
                if labels else f"It reads all of {name} again.")
    automation = await _persist_spec(
        db, automation=automation, user_id=user_id, raw=raw,
        code="bad_filters", refusal="I could not narrow it that way.",
    )
    return {
        "filters": filters_of(_spec_raw(automation)).get(connector_id) or [],
        "sentence": sentence,
        "workflow": await workflow_payload(
            db, automation=automation, user_id=user_id,
        ),
    }


# ── R42 §5.3 — the account's instant triggers ────────────────────────
#
# A trigger is a `trigger.sources[]` entry, so turning one on adds a
# firing lane to the automation rather than changing a setting beside
# it. Two consequences the writer below owes the user:
#
#   - the SCHEDULE is a source too, and it survives untouched. An
#     automation can hold a schedule and an event lane at once (the
#     compiler already builds one primitive per source), and R42 §5.3
#     is exactly that shape: "it all waits for the run, except these".
#   - an automation with only events and no schedule is legal, so the
#     last event cannot be taken off one: `trigger.sources` may not go
#     empty, and the refusal says what to do instead.

# Event param → (what to ask the user for, the pin kinds that can fill
# it). Some events are only meaningful about a PLACE the user picked —
# github wants a repository, teams a chat — and the manifest says so in
# `params_required`. The pin is where that place already lives.
_EVENT_PARAM_PINS: dict[str, tuple[str, tuple[str, ...]]] = {
    "owner": ("repository", ("repo",)),
    "repo": ("repository", ("repo",)),
    "chat_id": ("chat", ("thread", "channel")),
    # Forward declarations, deliberately: no manifest asks for either today
    # (Slack declares no events at all, and Jira's `issue_created` requires
    # no params), so `_event_params_from_pins` never looks them up. They are
    # here so the mapping is written down once when those events land, not
    # because they were verified against a manifest.
    "channel": ("channel", ("channel",)),
    "project_key": ("project", ("project",)),
}


def _event_params_from_pins(
    event_spec: dict, pins: list[dict],
) -> tuple[dict, list[str]]:
    """`(params, missing)` for an event that names a place.

    `owner`/`repo` are one pin ("owner/repo") answering two params —
    the same shape `_apply_focus_scope` reads for github.
    """
    params: dict = {}
    missing: list[str] = []
    by_kind: dict[str, list] = {}
    for p in pins or []:
        if isinstance(p, dict) and p.get("id"):
            by_kind.setdefault(str(p.get("kind") or ""), []).append(p)
    for field in (event_spec.get("params_required") or []):
        what, kinds = _EVENT_PARAM_PINS.get(str(field), ("place", ()))
        value = ""
        for kind in kinds:
            for pin in by_kind.get(kind, []):
                pid = str(pin.get("id") or "")
                if field in ("owner", "repo"):
                    if "/" not in pid:
                        continue
                    owner, repo = pid.split("/", 1)
                    value = owner if field == "owner" else repo
                else:
                    value = pid
                if value:
                    break
            if value:
                break
        if value:
            params[str(field)] = value
        elif what not in missing:
            missing.append(what)
    return params, missing


def triggers_of(raw: dict, connector_id: str) -> list[str]:
    """The event keys this account currently fires on."""
    out: list[str] = []
    for s in (raw.get("trigger") or {}).get("sources") or []:
        if not isinstance(s, dict):
            continue
        if s.get("connector_id") == connector_id and s.get("event"):
            key = str(s["event"])
            if key not in out:
                out.append(key)
    return out


def available_triggers(
    raw: dict, capability: dict, connector_id: str,
) -> list[dict]:
    """`[{id, label}]` — every EVENT the connector's manifest declares.

    Nothing is invented: the design lists 31 instant triggers and the
    platform declares eight across seven connectors, Slack none at all.
    An empty list is the honest answer and the app says so in words.

    Version-gated like `available_filters`, because `set_triggers` refuses
    a v1 spec: without this a v1 automation drew tappable rows whose every
    tap 409'd and rolled back, which is the picker-that-writes-nowhere the
    round exists to remove. A trigger already ON is still listed, so a
    stored one can always be turned off.
    """
    from .spec import event_label
    on = set(triggers_of(raw, connector_id))
    if raw.get("version") != 2 and not on:
        return []
    entry = capability.get(connector_id) or {}
    out = []
    for ev in entry.get("events") or []:
        key = str((ev or {}).get("key") or "")
        if not key:
            continue
        out.append({
            "id": key,
            "label": event_label(connector_id, key, ev.get("description") or ""),
        })
    return out


def _new_source_id(taken: set, connector_id: str, event_key: str) -> str:
    """A source id inside `spec_v2._ID_RE` that collides with nothing —
    not another source, not a STEP (the v2 validator shares one id
    space between them)."""
    import re as _re
    base = _re.sub(r"[^a-z0-9_]", "_",
                   f"{connector_id}_{event_key}".lower())[:24].strip("_")
    if not base or not base[0].isalpha():
        base = f"s{base}"[:24]
    cand, n = base, 2
    while cand in taken:
        cand = f"{base[:22]}_{n}"
        n += 1
    return cand


async def set_triggers(
    db: AsyncSession, *, automation: Automation, user_id: str,
    connector_id: str, triggers: list,
) -> dict:
    """Replace the event lanes this account fires on (design §5.3).

    Gated on the live manifest, so this can only ever turn on an event
    the connector actually declares — the registry is the same one
    `validate_spec_v2` re-checks against on the way through
    `_persist_spec`, and an unreachable registry refuses rather than
    guessing (fail closed: `fetch_registry` answers {} when the
    platform is down, and {} must not read as "nothing is allowed"
    written into a spec).
    """
    from .spec_v2 import MAX_SOURCES
    raw = _spec_raw(automation)
    name = verbs.display_name(connector_id) or connector_id
    if connector_id not in _member_connectors(raw):
        raise WorkflowError(
            "not_member",
            f"This automation does not use {name} yet — add it first.",
        )
    if raw.get("version") != 2:
        raise WorkflowError(
            "not_supported",
            "This automation is on the older engine, so it cannot watch "
            "for anything between runs yet.",
        )
    capability = await reg.fetch_registry(user_id)
    if not capability:
        raise WorkflowError(
            "registry_unavailable",
            "I could not check what your accounts can announce. Try that "
            "again in a moment.",
        )
    events = {str(e.get("key")): e
              for e in (capability.get(connector_id) or {}).get("events") or []
              if e.get("key")}
    wanted: list[str] = []
    for tid in (triggers or []):
        if not isinstance(tid, str) or tid not in events:
            raise WorkflowError(
                "unknown_trigger",
                f"{name} cannot tell you the moment that happens.",
            )
        if tid not in wanted:
            wanted.append(tid)

    sources = [s for s in (raw.get("trigger") or {}).get("sources") or []
               if isinstance(s, dict)]
    kept = [s for s in sources
            if not (s.get("connector_id") == connector_id and s.get("event"))]
    mine = {str(s.get("event")): s for s in sources
            if s.get("connector_id") == connector_id and s.get("event")}
    if not wanted and not kept:
        raise WorkflowError(
            "last_trigger",
            "Then nothing would ever start it. Give it a schedule first "
            "and I will take this off.",
        )
    cap = capability.get(connector_id) or {}
    mode = "push" if cap.get("push") else "poll"
    pins = focus_of(raw).get(connector_id) or []
    taken = {str(s.get("id") or "") for s in kept}
    taken |= {str(st.get("id") or "") for st in raw.get("steps") or []
              if isinstance(st, dict)}
    added: list[dict] = []
    for key in wanted:
        if key in mine:
            # Untouched, id and all: re-minting it would tear down and
            # rebuild a live primitive for a no-op.
            added.append(mine[key])
            taken.add(str(mine[key].get("id") or ""))
            continue
        ev = events[key]
        params, missing = _event_params_from_pins(ev, pins)
        if missing:
            raise WorkflowError(
                "needs_pin",
                f"Pick the {verbs.join_list(missing)} in {name} first — "
                f"tap + on the one I should watch.",
            )
        sid = _new_source_id(taken, connector_id, key)
        taken.add(sid)
        added.append({
            "id": sid,
            "mode": mode,
            "connector_id": connector_id,
            "event": key,
            **({"params": params} if params else {}),
            # The manifest's own dedupe field. `poll_interval_s` is left
            # out on purpose: the validator fills the connector's floor,
            # which is the only interval a user who tapped a row asked
            # for.
            "dedupe_key": f"event.{ev.get('dedupe_field') or 'id'}",
        })
    if len(kept) + len(added) > MAX_SOURCES:
        raise WorkflowError(
            "too_many_sources",
            f"It can watch {MAX_SOURCES} things at once at most. Take one "
            f"off first.",
        )
    raw.setdefault("trigger", {})["sources"] = kept + added

    labels = {t["id"]: t["label"]
              for t in available_triggers(raw, capability, connector_id)}
    on = [labels.get(k) or k for k in wanted]
    sentence = (f"I will tell you the moment {verbs.join_list(on).lower()}."
                if on else
                f"Nothing in {name} interrupts you now — it waits for the "
                f"run.")
    automation = await _persist_spec(
        db, automation=automation, user_id=user_id, raw=raw,
        code="bad_triggers", refusal="I could not set that up.",
    )
    return {
        "triggers": triggers_of(_spec_raw(automation), connector_id),
        "sentence": sentence,
        "workflow": await workflow_payload(
            db, automation=automation, user_id=user_id,
        ),
    }


# R42: `thread` names two different things, and only one of them is a
# place an automation can POST to. `contents._read_teams` pins a Teams
# CHAT as kind `thread`, and that chat IS the destination of
# `teams__send_chat_message` (`executor_v2` resolves `chat_id` from the
# `thread`/`channel` pins) — so it has to keep bridging. Everywhere
# else a `thread` is a MESSAGE thread: a preview ROW, pinned to say
# "start from this conversation". Bridging that would quietly redirect
# where the automation posts every time someone pinned something to
# READ. On Teams the message ROWS carry this kind too, which the kind
# alone cannot see — `_pin_names_a_destination` is the other half.
_DESTINATION_KINDS = frozenset({"channel"})
_CHAT_AS_THREAD_CONNECTORS = frozenset({"teams"})


def _names_a_destination(connector_id: str, kind: str) -> bool:
    """Could a pin of this KIND, on this connector, be a place to post?"""
    return (kind in _DESTINATION_KINDS
            or (kind == "thread"
                and connector_id in _CHAT_AS_THREAD_CONNECTORS))


def _pin_names_a_destination(connector_id: str, pin: dict) -> bool:
    """The pin names a place this automation could POST to.

    Two halves, and neither answers it alone. The KIND half above is
    what keeps a Jira project, a GitHub repo and a mail thread out:
    they are containers, but not ones anything writes INTO — the write
    there targets a ticket, an issue, a draft. The CONTAINER half is
    what R42's own reader made necessary: `contents._read_teams` pins
    the CHAT as kind `thread` — that chat really is the destination of
    `teams__send_chat_message` — and now pins every MESSAGE ROW in it
    as kind `thread` too, with a `<chat>#<message>` id the app posts
    verbatim. So the ordinary read-pin gesture on a Teams message asked
    to make that message the automation's write destination, and
    `only_if_unpinned` cannot catch it: an unpinned destination has
    nothing to refuse with. `contents.container_of` already knows the
    row format per connector, so ask it — a pin whose container is
    ITSELF is the whole place; a row inside one is not.
    """
    from . import contents
    if not _names_a_destination(connector_id, str(pin.get("kind") or "")):
        return False
    got = contents.container_of(connector_id, pin)
    return got is not None and got[0] == str(pin.get("id") or "").strip()


async def pin_write_destination(
    db: AsyncSession, *, automation: Automation, user_id: str,
    connector_id: str, target: dict, only_if_unpinned: bool = False,
) -> Optional[dict]:
    """Point the connector's write step at `target` and ask for the
    grant — the WHOLE flow: grant request, step pin, grant card in the
    main chat, needs_you ask in the thread.

    R39: this is one function because it has two callers with one
    contract — the agent's `automations__set_destination` tool and
    `add_focus` (the canvas "+" on a channel). The founder followed the
    agent's own instruction ("pick all-toup and I'll set it there") by
    tapping "+" on all-toup, and got `bad_focus_kind`: the canvas pin
    wrote a FOCUS, and a focus is not a destination. Now a channel
    pinned under the connector that still owes a destination sets both.

    `only_if_unpinned` is the canvas caller's safety: a "+" must never
    silently REDIRECT an already-approved destination (the agent tool
    may — an explicit ask is a redirect).

    Returns None when there is nothing to do (no write step on this
    connector, or pinned already and `only_if_unpinned`). Returns
    `{"ok": False, "sentence": …}` when the grant could not be prepared
    — the caller decides whether that poisons its own operation.
    """
    from app.services.automation_verbs import (
        _WRITE_CLAUSES, display_name, is_write_tool,
    )
    raw = _spec_raw(automation)
    if raw.get("version") != 2:
        return None
    steps = [s for s in raw.get("steps") or []
             if s.get("connector_id") == connector_id
             and (s.get("grant_id") or is_write_tool(s.get("tool")))]
    if not steps:
        return None
    unpinned = [s for s in steps
                if not (s.get("grant_target") or {}).get("id")]
    if only_if_unpinned and not unpinned:
        return None
    tool = str((unpinned or steps)[0].get("tool") or "")
    label = str(target.get("label") or target.get("id") or "")
    clause = _WRITE_CLAUSES.get(tool) or "write"
    grant = await reg.create_grant_request(
        user_id,
        connector_id=connector_id,
        tool_name=tool,
        target=target,
        cadence=None,
        mode=raw.get("mode") or "auto",
        summary=f"{automation.name}: {clause} in {label}",
        preview=None,
        automation_id=automation.id,
    )
    if grant is None:
        return {"ok": False, "sentence": (
            "The permission ask could not be prepared just now — "
            "nothing about the destination changed.")}
    from .service import pin_destination
    from .spec import SpecError
    try:
        await pin_destination(
            db, automation_id=automation.id, user_id=user_id,
            connector_id=connector_id, tool=tool, grant=grant,
            target=target,
        )
    except SpecError as e:
        return {"ok": False, "sentence": (
            "; ".join(str(x.get("message") or "") for x in e.errors)
            or "The destination could not be pinned.")}

    name = display_name(connector_id) or connector_id
    payload = {
        "id": grant["id"],
        "automation_id": grant.get("automation_id") or automation.id,
        "connector_id": grant["connector_id"],
        "action": grant["tool_name"],
        "action_label": grant["tool_name"].split("__", 1)[-1]
        .replace("_", " "),
        "target": grant.get("target") or {},
        "cadence": grant.get("cadence") or {},
        "mode": grant.get("mode"),
        "summary": grant.get("summary"),
        "preview": None,
        "status": grant.get("status"),
        "created_at": grant.get("created_at"),
        "expires_at": grant.get("expires_at"),
        "decided_at": None,
        "decided_via": None,
    }
    try:
        from . import cards
        await cards.write_card_message(
            db,
            user_id=user_id,
            content=f"Permission needed: {grant.get('summary')}",
            metadata_key=cards.GRANT_CARD_KEY,
            payload=payload,
            title="Permission request",
        )
        await cards.broadcast_card(user_id, cards.GRANT_CARD_KEY, payload)
    except Exception as e:  # noqa: BLE001 — the thread turn below still
        # carries the ask; a missing card is a missing duplicate.
        logger.warning("[automations] grant card skipped: %s", e)
    try:
        thread = await ledger.ensure_thread(
            db, user_id=user_id, automation_id=automation.id,
        )
        await ledger.append_turn(
            db, user_id=user_id, thread=thread, run_id=None,
            kind="needs_you",
            payload={
                "account_id": connector_id,
                "connector_id": connector_id,
                "name": name,
                "reason_code": "grant_missing",
                "sentence": (f"It needs your permission to "
                             f"{clause} in {label}."),
                "fix": "grant",
                "fix_label": "Allow it",
                "grant_request_id": grant["id"],
            },
        )
    except Exception as e:  # noqa: BLE001 — the main-chat card stands;
        # the thread just lost its copy of the ask.
        logger.warning("[automations] thread grant ask skipped: %s", e)
    return {
        "ok": True,
        "grant_id": grant["id"],
        "tool": tool,
        "sentence": (f"I asked for your permission to {clause} in "
                     f"{label} — approve it and it runs on its own."),
    }


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
