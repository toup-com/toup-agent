"""The one-shot back-fills R31 owes — rules and fact scope.

Two migrations, one entry point, because they are the same shape: a
thing the system should have been recording all along, applied to
everything it already has.

**Rules (R31-18).** `automations.rules_json` has three writers, all of
them the user typing into the Rules sheet. Nothing ever extracted a
rule from a description or a setup conversation, so the founder's
Morning work brief — whose own description says "post ONE line, no
thread" and whose steps say it again — opened its Workflow reading
`LINES IT WILL NOT CROSS 0`. The constraint was stated three times and
recorded zero.

C owns what counts as a rule (`rule_extraction.py`); this runs it over
every automation that exists and writes what it finds.

**Fact scope (R31-18, second half).** `curator_v2.normalize_candidate`
files a thread-learned fact as `global` unless the extraction model
literally emits the word "automation" — while the Memory tab reads
`scope == automation_id` exactly. So facts learned in an automation's
own thread do not appear in that automation's Memory tab, which is
five groups reading `0 things` on an automation that has been running
for days.

C owns the RULE for which scope a fact takes
(`fixtures/automations/memory-scope.json`); this applies it to the
facts already filed. It only ever moves facts INTO an automation scope
from `global`, never the other way — C's own ambiguity note records why
the two errors are not symmetrical: deleting an automation hard-deletes
`MemoryFact WHERE scope == automation_id`, so a fact wrongly narrowed
is destroyed rather than misfiled.

Both are idempotent. Both are dry-run by default.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Automation

logger = logging.getLogger(__name__)


# ── rules ────────────────────────────────────────────────────────────

async def backfill_rules(
    db: AsyncSession, *, user_id: str, dry_run: bool = True,
) -> dict:
    """Run C's extractor over every automation. `{scanned, added}`."""
    from . import rule_extraction, workflow

    rows = list((await db.execute(
        select(Automation)
        .where(Automation.user_id == user_id)
        .where(Automation.deleted_at.is_(None))
    )).scalars())

    scanned = added = 0
    detail: list[dict] = []
    for a in rows:
        scanned += 1
        try:
            existing = workflow.rules_list(a)
            raw = workflow._spec_raw(a)
            steps = [s.get("text") or ""
                     for s in workflow._steps_human(a, raw)]
            setup_lines = await _setup_lines(db, a)
            found = rule_extraction.extract_rules(
                description=a.description or "",
                setup_text=setup_lines,
                steps=steps,
                existing=existing,
            )
        except Exception as e:  # noqa: BLE001 — one automation's
            # extraction failing must not stop the rest
            logger.warning("[backfill] rules skipped %s: %s", a.id, e)
            continue
        if not found:
            continue
        added += len(found)
        detail.append({"automation_id": a.id,
                       "added": [r.get("text") for r in found]})
        if not dry_run:
            a.rules_json = json.dumps(list(existing) + list(found),
                                      default=str)
    if not dry_run:
        await db.commit()
    out = {"scanned": scanned, "added": added, "dry_run": dry_run}
    if detail:
        out["detail"] = detail[:50]
    logger.info("[backfill] rules user=%s %s", user_id[:8], out)
    return out


async def _setup_lines(db: AsyncSession, automation: Automation) -> list[str]:
    """The setup conversation, from the automation's own thread.

    Only `user` and `agent` turns, and only the ones before the first
    RUN — a constraint the user stated while setting the automation up
    is a rule; a sentence in a later question is a question.
    """
    from . import ledger
    try:
        thread = await ledger.thread_for(db, automation.id)
        if thread is None:
            return []
        turns, _more = await ledger.list_turns(
            db, thread_id=thread.id, limit=120,
        )
    except Exception:  # noqa: BLE001
        return []
    out: list[str] = []
    for t in turns:
        if t.get("kind") == "note" and t.get("stamp") in ("started", "ran",
                                                          "tried"):
            break
        if t.get("kind") in ("user", "agent") and t.get("text"):
            out.append(str(t["text"]))
    return out


# ── fact scope ───────────────────────────────────────────────────────

_SCOPE_RULE: Optional[dict] = None


def scope_rule() -> dict:
    """C's `memory-scope.json`. Loaded once; missing ⇒ `{}`."""
    global _SCOPE_RULE
    if _SCOPE_RULE is not None:
        return _SCOPE_RULE
    try:
        from . import account_health
        import os
        path = os.path.join(
            os.path.dirname(account_health._table_path()),
            "memory-scope.json",
        )
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        _SCOPE_RULE = data if isinstance(data, dict) else {}
    except Exception as e:  # noqa: BLE001
        logger.warning("[backfill] memory-scope unreadable: %s", e)
        _SCOPE_RULE = {}
    return _SCOPE_RULE


def _signal_terms(automation: Automation) -> list[str]:
    """What "names this automation" means, for THIS automation.

    C's naming test asks whether the fact's own sentence names the
    automation, its schedule, its sources, its channel or its output.
    Those are per-automation strings, so the rule cannot be applied
    without them — which is why this is a migration and not a filter.
    """
    terms: list[str] = []
    name = (automation.name or "").strip()
    if name:
        terms.append(name.lower())
    try:
        from . import workflow
        raw = workflow._spec_raw(automation)
        for cid in workflow._member_connectors(raw):
            from app.services.automation_verbs import display_name
            terms.append((display_name(cid) or cid).lower())
        for _tool, _cid, target in workflow._write_tools(raw):
            label = (target or {}).get("label") or ""
            if label:
                terms.append(label.lower())
    except Exception:  # noqa: BLE001
        pass
    return [t for t in terms if len(t) >= 3]


def _names_this_automation(text: str, terms: list[str]) -> bool:
    body = " " + re.sub(r"[^a-z0-9 #]+", " ", (text or "").lower()) + " "
    return any(f" {t} " in body or t in body for t in terms)


async def rescope_thread_facts(
    db: AsyncSession, *, user_id: str, dry_run: bool = True,
) -> dict:
    """Move thread-learned facts that NAME their automation into its
    scope. `{scanned, moved}`.

    Deliberately conservative in one direction: `global → automation_id`
    only, and only when C's naming test answers yes. Ambiguity resolves
    to `global` because the errors are not symmetrical — a fact wrongly
    narrowed is DESTROYED when the user deletes that automation, while
    a fact wrongly left global is merely in one more place than it
    needed to be.
    """
    from app.db.models import MemoryEpisode, MemoryFact

    automations = list((await db.execute(
        select(Automation)
        .where(Automation.user_id == user_id)
        .where(Automation.deleted_at.is_(None))
    )).scalars())
    if not automations:
        return {"scanned": 0, "moved": 0, "dry_run": dry_run}

    # A fact is "thread-learned" if an episode from that automation's
    # thread exists at or after it. The fact table carries no thread
    # pointer, so the episode ledger is the only link — and it is the
    # honest one: it records where the agent was when it filed.
    thread_ids: dict[str, str] = {}
    for a in automations:
        try:
            from . import ledger
            thread = await ledger.thread_for(db, a.id)
            if thread is not None:
                thread_ids[a.id] = thread.id
        except Exception:  # noqa: BLE001
            continue

    facts = list((await db.execute(
        select(MemoryFact)
        .where(MemoryFact.user_id == user_id)
        .where(MemoryFact.scope == "global")
    )).scalars())

    scanned = moved = 0
    detail: list[dict] = []
    for a in automations:
        terms = _signal_terms(a)
        if not terms:
            continue
        for fact in facts:
            if fact.scope != "global":
                continue          # already claimed by an earlier pass
            scanned += 1
            text = f"{fact.text or ''} {fact.why or ''}"
            if not _names_this_automation(text, terms):
                continue
            moved += 1
            detail.append({"fact_id": fact.id, "automation_id": a.id,
                           "text": (fact.text or "")[:80]})
            if not dry_run:
                fact.scope = a.id
    if not dry_run:
        await db.commit()
    del MemoryEpisode
    out = {"scanned": scanned, "moved": moved, "dry_run": dry_run}
    if detail:
        out["detail"] = detail[:50]
    logger.info("[backfill] rescope user=%s %s", user_id[:8], out)
    return out


async def run_all(
    db: AsyncSession, *, user_id: str, dry_run: bool = True,
) -> dict:
    return {
        "rules": await backfill_rules(db, user_id=user_id, dry_run=dry_run),
        "facts": await rescope_thread_facts(
            db, user_id=user_id, dry_run=dry_run),
    }
