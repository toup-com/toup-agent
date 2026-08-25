"""Curated memory facts for automations (Round 29) — the UI ledger.

Two stores with a defined direction (CONTRACTS-R29.md §4, the R28
engine-memory precedent):

  - `automation_facts` rows — THE truth for the automation's Memory
    tab: first-class ids, categories, source attribution, timestamps.
    v3 bullet files cannot carry any of that per fact.
  - The brain — a best-effort PROJECTION through the sanctioned
    curator seam (`instruct_file`/`instruct_global`, the same entry
    `memory_notes` uses; never `disable_post_processing`, never raw
    provider payloads). The table never waits on the curator and a
    projection failure never loses the row; the curator's durability
    rules still judge what the brain keeps.

Category routes the projection: `people` → the curator's own person
files (global instruct — it owns person identity), `preferences` →
`topics/preferences`, `deadlines` → `topics/deadlines`, anything else
is a life domain → `areas/<domain>` (the R28 path). Deleting the
automation cascades the rows; the brain keeps what the curator judged
durable — facts about a life outlive the tool that learned them.
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    AUTOMATION_FACT_CANONICAL_CATEGORIES,
    AUTOMATION_FACT_SOURCE_KINDS,
    AUTOMATION_FACT_SOURCES,
    AutomationFact,
)

logger = logging.getLogger(__name__)

_CATEGORY_RE = re.compile(r"^[a-z][a-z0-9-]{1,31}$")

# Fixed topic files for the canonical non-people categories.
_TOPIC_FILES = {
    "preferences": (
        "topics/preferences", "Preferences",
        "The user's standing preferences and tastes — likes, dislikes, "
        "defaults they expect; read when choosing on their behalf.",
    ),
    "deadlines": (
        "topics/deadlines", "Deadlines",
        "The user's dated commitments and deadlines — what is due and "
        "when; read when planning or scheduling their work.",
    ),
}


def normalize_category(value: object) -> Optional[str]:
    """Canonical category slug, or None when the value isn't one."""
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    if not v or not _CATEGORY_RE.fullmatch(v):
        return None
    return v


def _clean_text(text: object, limit: int = 400) -> str:
    return " ".join(str(text).split())[:limit]


def category_sort_key(category: str) -> tuple:
    """Canonical order: people, preferences, deadlines, then the rest
    alphabetically."""
    try:
        return (0, AUTOMATION_FACT_CANONICAL_CATEGORIES.index(category))
    except ValueError:
        return (1, category)


# ── The write seam (R29-C writes through this) ───────────────────────


async def record(
    db: AsyncSession,
    *,
    user_id: str,
    automation_id: str,
    facts: list,
    category: str,
    source: str,
    source_kind: str,
    run_id: Optional[str] = None,
) -> dict:
    """Persist a batch of curated facts and project them to the brain.

    Returns `{"saved": int, "ids": [...]}` — exact-text duplicates per
    (automation, category) are skipped, invalid inputs save nothing.
    Commits itself; the projection runs AFTER the commit (a curator
    failure never loses the row).
    """
    cat = normalize_category(category)
    if (
        cat is None
        or source not in AUTOMATION_FACT_SOURCES
        or source_kind not in AUTOMATION_FACT_SOURCE_KINDS
    ):
        return {"saved": 0, "ids": []}
    clean = [t for t in (_clean_text(f) for f in facts or []) if t]
    if not clean:
        return {"saved": 0, "ids": []}

    existing = {
        row.text
        for row in (await db.execute(
            select(AutomationFact)
            .where(AutomationFact.automation_id == automation_id)
            .where(AutomationFact.user_id == user_id)
            .where(AutomationFact.category == cat)
        )).scalars().all()
    }
    ids: list[str] = []
    saved_texts: list[str] = []
    for text in clean:
        if text in existing:
            continue
        existing.add(text)
        row = AutomationFact(
            id=str(uuid.uuid4()),
            user_id=user_id,
            automation_id=automation_id,
            category=cat,
            text=text,
            source=source,
            source_kind=source_kind,
            run_id=run_id,
        )
        db.add(row)
        ids.append(row.id)
        saved_texts.append(text)
    if ids:
        await db.commit()
        if source == "agent":
            try:
                await _project_to_brain(db, user_id, cat, saved_texts)
            except Exception as e:  # noqa: BLE001 — projection companion
                logger.warning(
                    "[automations] fact projection escaped: %s", e,
                )
    return {"saved": len(ids), "ids": ids}


# ── Reads ────────────────────────────────────────────────────────────


def _fact_payload(row: AutomationFact) -> dict:
    return {
        "id": row.id,
        "text": row.text,
        "category": row.category,
        "source": row.source,
        "updated_at": row.updated_at.isoformat() + "Z",
    }


async def list_facts(
    db: AsyncSession, *, user_id: str, automation_id: str,
) -> dict:
    """The Memory-tab shape: facts in canonical category order (newest
    first within a category) + the most recent agent write batch."""
    rows = (await db.execute(
        select(AutomationFact)
        .where(AutomationFact.automation_id == automation_id)
        .where(AutomationFact.user_id == user_id)
    )).scalars().all()
    rows.sort(
        key=lambda r: (category_sort_key(r.category),
                       -(r.created_at or datetime.min).timestamp()),
    )

    last_update = None
    agent_rows = [r for r in rows if r.source == "agent"]
    if agent_rows:
        latest = max(agent_rows, key=lambda r: r.created_at or datetime.min)
        cluster = [
            r for r in agent_rows
            if r.source_kind == latest.source_kind
            and (
                (r.run_id == latest.run_id) if latest.run_id
                else r.created_at == latest.created_at
            )
        ]
        last_update = {
            "count": len(cluster),
            "at": (latest.created_at or datetime.utcnow()).isoformat() + "Z",
        }
    return {
        "facts": [_fact_payload(r) for r in rows],
        "last_agent_update": last_update,
    }


# ── User CRUD (the tab's add/edit/delete) ────────────────────────────


async def add_fact(
    db: AsyncSession, *, user_id: str, automation_id: str,
    text: str, category: str,
) -> Optional[dict]:
    """One user-authored fact; None when the input doesn't validate."""
    result = await record(
        db, user_id=user_id, automation_id=automation_id,
        facts=[text], category=category, source="user", source_kind="edit",
    )
    if not result["ids"]:
        return None
    row = await db.get(AutomationFact, result["ids"][0])
    return _fact_payload(row) if row else None


async def update_fact(
    db: AsyncSession, *, user_id: str, automation_id: str,
    fact_id: str, text: Optional[str] = None,
    category: Optional[str] = None,
) -> Optional[dict]:
    row = await db.get(AutomationFact, fact_id)
    if row is None or row.user_id != user_id \
            or row.automation_id != automation_id:
        return None
    before = row.text
    if text is not None:
        cleaned = _clean_text(text)
        if not cleaned:
            return None
        row.text = cleaned
    if category is not None:
        cat = normalize_category(category)
        if cat is None:
            return None
        row.category = cat
    changed_text = row.text if row.text != before else None
    await db.commit()
    # commit expires ORM attributes (and the projection's failure path
    # rolls back, expiring them again) — refresh and build the payload
    # BEFORE any projection touches the session, or the sync attribute
    # read needs lazy IO and dies with MissingGreenlet under asyncio.
    await db.refresh(row)
    payload = _fact_payload(row)
    if changed_text is not None:
        await _project_correction(db, user_id, before, changed_text)
    return payload


async def delete_fact(
    db: AsyncSession, *, user_id: str, automation_id: str, fact_id: str,
) -> bool:
    row = await db.get(AutomationFact, fact_id)
    if row is None or row.user_id != user_id \
            or row.automation_id != automation_id:
        return False
    text = row.text
    await db.delete(row)
    await db.commit()
    # The agent must not keep "knowing" a deleted fact — best-effort
    # removal from the brain, after the delete committed.
    await _project_removal(db, user_id, text)
    return True


# ── Brain projection (best-effort, always after the commit) ──────────


async def _ensure_topic_file(db: AsyncSession, user_id: str, key: str) -> bool:
    """Deterministically create the fixed topic file if absent — the
    `memory_notes` validate→apply walk, no LLM."""
    slug, title, description = _TOPIC_FILES[key]
    try:
        from app.services import memory_file_ops as ops
        rows = await ops._all_files(db, user_id)
        if any(r.slug == slug for r in rows):
            return True
        plan = ops.validate_ops(
            [{
                "op": "create_file",
                "section": "topics",
                "slug": slug,
                "title": title,
                "description": description,
            }],
            rows,
        )
        applied = await ops.apply_ops(db, user_id, plan)
        await db.commit()
        return bool(applied.get("applied"))
    except Exception as e:  # noqa: BLE001 — projection is a companion
        logger.warning("[automations] topic file create failed %s: %s",
                       slug, e)
        try:
            await db.rollback()
        except Exception:  # noqa: BLE001
            pass
        return False


async def _project_to_brain(
    db: AsyncSession, user_id: str, category: str, texts: list[str],
) -> None:
    """File the batch through the curator; one instruct per batch."""
    if not texts:
        return
    joined = "; ".join(texts)
    try:
        from app.services import memory_curator

        if category == "people":
            await memory_curator.instruct_global(
                db, user_id,
                "Record what the user shared about people in their life "
                f"(merging with anything already known): {joined}",
            )
            return
        if category in _TOPIC_FILES:
            if not await _ensure_topic_file(db, user_id, category):
                return
            slug = _TOPIC_FILES[category][0]
            await memory_curator.instruct_file(
                db, user_id, slug,
                f"Keep the user's {category} file up to date. Record "
                f"(merging with anything it already says): {joined}",
            )
            return
        # Everything else is a life domain — the R28 path.
        from .memory_notes import record_automation_fact
        for text in texts:
            await record_automation_fact(
                db, user_id=user_id, domain=category, fact=text,
            )
    except Exception as e:  # noqa: BLE001 — see module docstring
        logger.warning(
            "[automations] brain projection failed cat=%s: %s: %s",
            category, type(e).__name__, str(e)[:200],
        )
        try:
            await db.rollback()
        except Exception:  # noqa: BLE001
            pass


async def _project_correction(
    db: AsyncSession, user_id: str, before: str, after: str,
) -> None:
    try:
        from app.services import memory_curator
        await memory_curator.instruct_global(
            db, user_id,
            f"The user corrected a saved fact. It used to say: {before} "
            f"— it should now say: {after}",
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("[automations] correction projection failed: %s", e)
        try:
            await db.rollback()
        except Exception:  # noqa: BLE001
            pass


async def _project_removal(db: AsyncSession, user_id: str, text: str) -> None:
    try:
        from app.services import memory_curator
        await memory_curator.instruct_global(
            db, user_id,
            f"The user deleted a saved fact — remove it if it is "
            f"recorded anywhere: {text}",
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("[automations] removal projection failed: %s", e)
        try:
            await db.rollback()
        except Exception:  # noqa: BLE001
            pass
