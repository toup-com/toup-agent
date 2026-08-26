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
from datetime import datetime
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


def _write_tools(raw: dict) -> list[tuple[str, str, dict]]:
    """[(connector_id, tool, grant_target)] for the spec's writes."""
    out = []
    if raw.get("version") == 2:
        for s in raw.get("steps") or []:
            if s.get("grant_id") or (s.get("tool") and s.get("grant_target")):
                if s.get("grant_id"):
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
        accounts.append(entry)

    capability = await reg.fetch_registry(user_id)
    available = [
        _account_entry(cid, connections.get(cid) or {})
        for cid in sorted(set(capability) | set(connections))
        if cid not in members and cid != "stub"
    ]

    from app.agent._user_tz_cache import get_cached_user_tz
    return {
        "schedule": schedule_block(automation, raw),
        "accounts": accounts,
        "available": available,
        "steps": _steps_human(automation, raw),
        "rules": rules_list(automation),
        "output": output_block(automation, raw),
        "counts": counts_block(raw, mode),
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
        action = body.get("action") or "Used it"
        detail = body.get("detail") or ""
        sentence = f"{action} · {detail}" if detail else action
        return {"sentence": sentence[:120],
                "at": r.created_at.isoformat() + "Z"}
    return {"sentence": "No runs yet", "at": None}


# ------------------------------------------------------------- writes

async def _edited_note(
    db: AsyncSession, automation: Automation,
) -> Optional[str]:
    """Every workflow write appends the EDITED note (§4.4)."""
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
        return turn["id"]
    except Exception as e:  # noqa: BLE001
        logger.warning("[workflow] EDITED note skipped: %s", e)
        return None


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
    await _edited_note(db, automation)
    return {"rule": rule, "rules": rules,
            "sentence": f"Added a rule — {text[0].lower()}{text[1:]}"
            + ("" if text.endswith(".") else ".")}


async def update_rule(
    db: AsyncSession, *, automation: Automation, rule_id: str, text: str,
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
    await _edited_note(db, automation)
    return {"rules": rules, "sentence": "Changed the rule."}


async def delete_rule(
    db: AsyncSession, *, automation: Automation, rule_id: str,
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
    await _edited_note(db, automation)
    return {"rules": rules, "sentence": "Removed the rule."}


async def save_permissions(
    db: AsyncSession, *, automation: Automation, user_id: str,
    account_id: str, can_ids: list[str], cant_ids: list[str],
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
    await _edited_note(db, automation)
    return result


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
        intents = outcome.get("applied") or []
        needs = outcome.get("needs") or []
        answer = outcome.get("answer")
        for intent in intents:
            # AUDIT-6: every applier commits its own write, so letting a
            # WorkflowError out of this loop left the earlier intents
            # COMMITTED, answered the whole call 409, and skipped the
            # `agent` turn below — a half-applied change with no record
            # of itself in the thread. "Move it to 7:15 and add a rule"
            # was enough to do it. A refusal is now a sentence, not an
            # abort: the safe siblings still land and the agent turn says
            # what did not.
            try:
                entry = await _apply_intent(
                    db, automation=automation, user_id=user_id,
                    intent=intent,
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
    kind = intent.get("kind")
    if kind == "rule":
        result = await add_rule(db, automation=automation,
                                text=intent.get("text") or "")
        return {"kind": "rule", "sentence": result["sentence"],
                "sheet": "rules",
                "undo_token": _mint_undo(
                    "rule", {"rule_id": result["rule"]["id"]})}
    if kind == "schedule":
        raw_before = _spec_raw(automation)
        before = _schedule_of(raw_before) or {}
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
            return None
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
            return None
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
            return None
        account_id = intent.get("account_id") or ""
        raw_before = _spec_raw(automation)
        if account_id not in _member_connectors(raw_before):
            return None
        from . import service as _svc
        name = verbs.display_name(account_id) or account_id
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
                return None
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
    return None


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
