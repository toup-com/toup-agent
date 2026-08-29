"""Purge the curator's own failure reports from the fact store (round 33).

Curator v2 filed the AGENT's connector failures and third-party ticket
states as facts about the USER, re-voiced in the second person:

    "You cannot read messages in GitHub because the org blocks Toup's
     GitHub access and an org owner needs to approve it in GitHub's
     OAuth app policy."  ·  "You stated that GitHub access is blocked."

The write gate refuses that class now (`curation_rules.refuse_reason` →
`agent_capability` / `item_status`), but nothing removes what is already
stored — and `_recall_facts` reads those rows back into every run and
every thread answer, so a stale "you cannot read GitHub" keeps telling
the agent GitHub is blocked long after an owner approved it.

ONE classifier: the purge asks `refuse_reason` the same question the
write gate asks, so the two can never disagree about what junk is.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from sqlalchemy import select

from app.db.models import MemoryFact
from .curation_rules import refuse_reason

logger = logging.getLogger(__name__)

#: The classes this purge removes. `definition` and `run_status` are the
#: two the gate already refused at write time — a row carrying one of
#: those predates the gate, and is junk by the same rule.
PURGED_REASONS = frozenset({
    "agent_capability", "item_status", "definition", "run_status", "empty",
})


async def purge(db, *, user_id: str, apply: bool = False) -> dict[str, Any]:
    """Find (and with ``apply``, delete) this user's junk facts.

    Dry run by default: a delete over someone's memory should have to be
    asked twice, the same precedent `backfill_route` sets.
    """
    rows = list((await db.execute(
        select(MemoryFact).where(MemoryFact.user_id == user_id)
    )).scalars().all())

    doomed: list[tuple[MemoryFact, str]] = []
    for row in rows:
        reason = refuse_reason(row.text or "")
        if reason in PURGED_REASONS:
            doomed.append((row, reason))

    by_reason: dict[str, int] = {}
    for _row, reason in doomed:
        by_reason[reason] = by_reason.get(reason, 0) + 1

    out: dict[str, Any] = {
        "scanned": len(rows),
        "matched": len(doomed),
        "by_reason": by_reason,
        "samples": [r.text[:160] for r, _ in doomed[:8]],
        "applied": False,
        "entities_removed": 0,
    }
    if not apply or not doomed:
        return out

    texts = [r.text for r, _ in doomed]
    for row, _reason in doomed:
        await db.delete(row)
    await db.commit()
    out["applied"] = True
    out["entities_removed"] = await _drop_orphan_entities(db, user_id=user_id)

    # The brain carries projections of these sentences (`memory_v2_service.
    # _project_fact` writes them into the areas/ files through the
    # curator), so deleting the rows alone would leave the agent still
    # saying them. Best-effort and AFTER the commit, exactly as
    # `forget_fact` does it.
    for text in texts[:40]:
        try:
            from app.services.memory_v2_service import _project_removal
            await _project_removal(db, user_id, text)
        except Exception as e:  # noqa: BLE001 — a projection never fails a purge
            logger.warning("[junk-facts] removal projection skipped: %s", e)
    return out


async def _drop_orphan_entities(db, *, user_id: str) -> int:
    """The entities those facts minted, now pointed at by nothing.

    Every one of them is typed `person`, because the curator hardcoded
    the kind — which is why a Slack channel, a Jira ticket and a GitHub
    repo all appeared under People & things.
    """
    try:
        from app.db.models import MemoryEntity
        ents = list((await db.execute(
            select(MemoryEntity).where(MemoryEntity.user_id == user_id)
        )).scalars().all())
        if not ents:
            return 0
        used = {
            fid for (fid,) in (await db.execute(
                select(MemoryFact.subject_entity_id)
                .where(MemoryFact.user_id == user_id)
            )).all() if fid
        }
        removed = 0
        for e in ents:
            if e.id in used:
                continue
            name = str(e.name or "")
            if e.kind == "person" and (
                name.startswith("#") or "/" in name
                or (name.upper() == name and "-" in name)
            ):
                await db.delete(e)
                removed += 1
        if removed:
            await db.commit()
        return removed
    except Exception as e:  # noqa: BLE001
        logger.warning("[junk-facts] entity sweep skipped: %s", e)
        return 0
