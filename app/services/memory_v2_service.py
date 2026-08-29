"""Memory v2 — the one service over the four §4.5 tables (R30).

One platform memory per user; an automation's memory is a scoped VIEW
(`scope = <automation_id>`), never a second store. Rules this module
enforces rather than documents:

  - **Global wins dedupe.** A fact already known globally is confirmed
    (`last_confirmed_at`), never re-created inside an automation scope
    — otherwise every automation re-learns the user's life and the
    Memory screen fills with echoes.
  - **Forget means forget.** A forget signal (user + normalized text
    hash, 30 days) refuses the re-add regardless of who is asking —
    the curator, an automation run, or a migration replay.
  - **The brain is a projection, not a dependency.** Fact writes
    project through the sanctioned curator seam exactly like the R29
    facts ledger (`instruct_global` for people, `memory_notes.
    record_automation_fact` for domains) — best-effort, AFTER the
    commit, so a curator failure never loses or vetoes the row
    (`test_curator_producers._ALLOWED_WRITE_SITES` unchanged).
  - **Total functions.** Old data carries categories and sources this
    schema never promised; unknown values map to safe defaults
    (`work_you_own`, `agent`) instead of raising — a migration must
    not die on the 400th row of 500.

Episodes are engine-written at ledger close (`automations/ledger.py`);
this module only reads them — plus the one historical back-fill for
runs that terminated before the v3 cut-over.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    MEMORY_ENTITY_KINDS,
    MEMORY_FORGET_SUPPRESS_DAYS,
    MEMORY_V2_CATEGORIES,
    MEMORY_V2_CATEGORY_LABELS,
    MEMORY_V2_CATEGORY_TONES,
    MEMORY_V2_SCOPE_GLOBAL,
    MEMORY_V2_SOURCES,
    MemoryEntity,
    MemoryEpisode,
    MemoryFact,
    MemoryForget,
)

logger = logging.getLogger(__name__)

# Fallbacks for data that predates this schema — mapping, never raising.
_FALLBACK_CATEGORY = "work_you_own"
_FALLBACK_SOURCE = "agent"

# Canonical domain render order (matches memory_notes.CANONICAL_DOMAINS,
# spelled here so the platform image never imports app/agent to list a
# memory screen).
_DOMAIN_ORDER = ("work", "university", "personal")

# §4.5 migration (D-20 / ND-2): definition-facts describe the TOOL, not
# the user, and run-status sentences describe a moment that has passed.
# Both are dropped, counted, and banned from re-entry by the curator.
_DEFINITION_PREFIX = "has an automation"
_DEFINITION_QUOTE_SHAPES = (("automation '", "': "), ('automation "', '": '))
_STATUS_MARKERS = ("is currently paused", "there was a problem with",
                   "last run")

# Best-effort person extraction for migrated rows: a capitalized
# First Last pair. Deliberately conservative — two title-case words,
# nothing clever; a miss costs nothing, the entity index is additive.
_PERSON_RE = re.compile(r"\b([A-Z][a-z]+ [A-Z][a-z]+)\b")


# ── Normalization ────────────────────────────────────────────────────


def normalize_text(s: object) -> str:
    """Dedupe key: casefolded, whitespace-collapsed, terminal
    punctuation stripped — "Boss is Sarah." and "boss  is sarah" are
    the same belief."""
    line = " ".join(str(s or "").split()).casefold()
    return line.rstrip(".!? ")


def text_hash(s: object) -> str:
    """sha256 of the normalized text — the forget-signal key."""
    return hashlib.sha256(normalize_text(s).encode("utf-8")).hexdigest()


def _clean(text: object, limit: int = 400) -> str:
    return " ".join(str(text or "").split())[:limit]


def _iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() + "Z" if dt else None


def _safe_category(value: object) -> str:
    """Total: anything outside the five canvas keys is `work_you_own`
    — old data must migrate, not raise."""
    if isinstance(value, str) and value.strip().lower() in MEMORY_V2_CATEGORIES:
        return value.strip().lower()
    return _FALLBACK_CATEGORY


def _safe_source(value: object) -> str:
    if isinstance(value, str) and value.strip().lower() in MEMORY_V2_SOURCES:
        return value.strip().lower()
    return _FALLBACK_SOURCE


def _parse_since(since: object) -> Optional[datetime]:
    """ISO string → naive UTC datetime (DB timestamps are naive UTC);
    unparseable input filters nothing rather than raising."""
    if isinstance(since, datetime):
        dt = since
    elif isinstance(since, str) and since.strip():
        try:
            dt = datetime.fromisoformat(since.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if dt.tzinfo is not None:
        offset = dt.utcoffset()
        dt = (dt - offset).replace(tzinfo=None) if offset else \
            dt.replace(tzinfo=None)
    return dt


# ── Payloads ─────────────────────────────────────────────────────────


def fact_payload(row: MemoryFact) -> dict:
    return {
        "id": row.id,
        "text": row.text,
        "why": row.why,
        "category": row.category,
        "scope": row.scope,
        "source": row.source,
        "learned_at": _iso(row.learned_at),
        # ── The raw ISO row must never reach a UI (round 33, item 6) ──────
        # `automations.py`'s own docstring says exactly that, and this
        # payload shipped `2026-08-26T15:17:33.226653Z` straight through it
        # onto a card that printed "Learned 2026-08-26T15:17:33.226653Z".
        # Both keys ride: the ISO for machines, a human phrase for screens,
        # so a client cannot be the only thing standing between the two.
        "learned_at_label": _when_label(row.learned_at),
        "last_confirmed_at": _iso(row.last_confirmed_at),
        "subject_entity_id": row.subject_entity_id,
    }


def _when_label(dt) -> Optional[str]:
    """"Today", "Yesterday", "3 days ago", "12 Aug", "12 Aug 2025"."""
    if dt is None:
        return None
    try:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        when = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        days = (now.date() - when.date()).days
        if days <= 0:
            return "Today"
        if days == 1:
            return "Yesterday"
        if days < 7:
            return f"{days} days ago"
        if when.year == now.year:
            return when.strftime("%-d %b")
        return when.strftime("%-d %b %Y")
    except Exception:  # noqa: BLE001 — a label never fails a payload
        return None


def _episode_payload(row: MemoryEpisode) -> dict:
    return {
        "id": row.id,
        "at": _iso(row.at),
        "at_label": _when_label(row.at),
        "text": row.text,
        "outcome": row.outcome,
        "automation_id": row.automation_id,
        "run_id": row.run_id,
        "thread_id": row.thread_id,
        "turn_id": row.turn_id,
        "item_ref": row.item_ref,
    }


def _entity_payload(row: MemoryEntity) -> dict:
    try:
        aliases = json.loads(row.aliases_json or "[]")
    except (ValueError, TypeError):
        aliases = []
    return {"id": row.id, "kind": row.kind, "name": row.name,
            "aliases": aliases}


# ── Entities ─────────────────────────────────────────────────────────


async def _resolve_entity(
    db: AsyncSession, user_id: str, kind: object, name: object,
) -> Optional[MemoryEntity]:
    """Find-or-create by the (user, kind, lower(name)) uniqueness key.
    Invalid kind or empty name resolves to no entity — the fact still
    saves, just unindexed."""
    if not isinstance(kind, str) or kind not in MEMORY_ENTITY_KINDS:
        return None
    clean_name = _clean(name, 200)
    if not clean_name:
        return None
    norm = clean_name.lower()
    row = (await db.execute(
        select(MemoryEntity)
        .where(MemoryEntity.user_id == user_id)
        .where(MemoryEntity.kind == kind)
        .where(MemoryEntity.name_norm == norm)
    )).scalars().first()
    if row is not None:
        return row
    row = MemoryEntity(user_id=user_id, kind=kind, name=clean_name,
                       name_norm=norm)
    db.add(row)
    await db.flush()
    return row


async def _matching_entity_ids(
    db: AsyncSession, user_id: str, needle: str,
) -> list[str]:
    """Case-insensitive: substring over names, exact over aliases."""
    want = needle.strip().lower()
    if not want:
        return []
    ids: list[str] = []
    rows = (await db.execute(
        select(MemoryEntity).where(MemoryEntity.user_id == user_id)
    )).scalars().all()
    for row in rows:
        if want in (row.name or "").lower():
            ids.append(row.id)
            continue
        try:
            aliases = json.loads(row.aliases_json or "[]")
        except (ValueError, TypeError):
            aliases = []
        if any(want == str(a).lower() for a in aliases):
            ids.append(row.id)
    return ids


# ── The write path ───────────────────────────────────────────────────


async def add_fact(
    db: AsyncSession,
    *,
    user_id: str,
    text: str,
    category: str,
    scope: str = MEMORY_V2_SCOPE_GLOBAL,
    why: Optional[str] = None,
    source: str = "agent",
    domain: Optional[str] = None,
    subject_entity: Optional[dict] = None,
    confidence: float = 0.7,
    run_id: Optional[str] = None,
) -> dict:
    """Save one fact, or confirm the one already held.

    Dedupe is by normalized text over the target scope AND the global
    scope — a globally-known fact is confirmed in place, never echoed
    into an automation scope. A live forget signal on the text refuses
    the write entirely (`{"suppressed": True}`). Commits itself; the
    brain projection runs AFTER the commit and can only ever fail
    quietly. `run_id` is accepted for call-site symmetry with the R29
    ledger but not stored — run linkage lives on episodes.
    """
    del run_id  # facts carry beliefs, episodes carry runs
    clean = _clean(text)
    if not clean:
        return {}
    cat = _safe_category(category)
    src = _safe_source(source)
    scope = _clean(scope, 36) or MEMORY_V2_SCOPE_GLOBAL
    now = datetime.utcnow()

    # Forget signals outrank every writer (§4.5) — WITHIN THEIR SCOPE.
    #
    # This filtered on `user_id + text_hash` alone while `memory_forgets`
    # carries `scope`, so a forget was global whatever it was written
    # against. Both directions are wrong and the second is the one that
    # loses data: forgetting a fact inside one automation suppressed the
    # same sentence everywhere, including the global memory the main
    # chat answers from — and the user's own reason for forgetting it
    # ("not for this automation") was exactly the distinction being
    # thrown away.
    #
    # A GLOBAL forget still reaches a scoped write: "stop remembering
    # this about me" has to mean everywhere, or it means nothing.
    h = text_hash(clean)
    live_forget = (await db.execute(
        select(MemoryForget)
        .where(MemoryForget.user_id == user_id)
        .where(MemoryForget.text_hash == h)
        .where(MemoryForget.scope.in_(
            sorted({scope, MEMORY_V2_SCOPE_GLOBAL})))
        .where(MemoryForget.until > now)
    )).scalars().first()
    if live_forget is not None:
        return {"suppressed": True}

    # Same belief in this scope or globally → confirm, don't duplicate.
    scopes = {scope, MEMORY_V2_SCOPE_GLOBAL}
    norm = normalize_text(clean)
    candidates = (await db.execute(
        select(MemoryFact)
        .where(MemoryFact.user_id == user_id)
        .where(MemoryFact.scope.in_(sorted(scopes)))
    )).scalars().all()
    match = None
    for row in candidates:
        if normalize_text(row.text) == norm:
            # A global holding outranks a scoped one as the survivor.
            if match is None or (
                row.scope == MEMORY_V2_SCOPE_GLOBAL
                and match.scope != MEMORY_V2_SCOPE_GLOBAL
            ):
                match = row
    if match is not None:
        match.last_confirmed_at = now
        await db.commit()
        await db.refresh(match)
        # `saved` is the seam contract with C's curator (it counts
        # truthy result["saved"]); a confirmation counts as saved.
        return {**fact_payload(match), "saved": True, "confirmed": True}

    entity = None
    if isinstance(subject_entity, dict):
        entity = await _resolve_entity(
            db, user_id, subject_entity.get("kind"),
            subject_entity.get("name"),
        )

    row = MemoryFact(
        user_id=user_id,
        domain=_clean(domain, 32) or None,
        category=cat,
        scope=scope,
        subject_entity_id=entity.id if entity is not None else None,
        text=clean,
        why=_clean(why) or None,
        source=src,
        confidence=float(confidence),
        learned_at=now,
    )
    db.add(row)
    await db.commit()
    # Payload BEFORE the projection: commit expires ORM attributes and
    # the projection's failure path rolls back, so a later sync read
    # would need lazy IO and die with MissingGreenlet under asyncio.
    await db.refresh(row)
    payload = {**fact_payload(row), "saved": True}
    await _project_fact(db, user_id, category=cat,
                        domain=row.domain, text=clean)
    return payload


async def forget_fact(
    db: AsyncSession, *, user_id: str, fact_id: str,
) -> bool:
    """Delete the fact everywhere and refuse its relearning for
    MEMORY_FORGET_SUPPRESS_DAYS. Best-effort brain removal after the
    delete committed (the R29 `_project_removal` precedent)."""
    row = await db.get(MemoryFact, fact_id)
    if row is None or row.user_id != user_id:
        return False
    text, scope, category = row.text, row.scope, row.category
    await db.delete(row)
    db.add(MemoryForget(
        user_id=user_id,
        scope=scope,
        category=category,
        text_hash=text_hash(text),
        until=datetime.utcnow() + timedelta(days=MEMORY_FORGET_SUPPRESS_DAYS),
    ))
    await db.commit()
    await _project_removal(db, user_id, text)
    return True


# ── Reads ────────────────────────────────────────────────────────────


async def list_facts_for_scope(
    db: AsyncSession, *, user_id: str, scope: str,
) -> dict:
    """The §3.10 sheet payload — all five categories always present, so
    the sheet never has to invent an empty section."""
    rows = (await db.execute(
        select(MemoryFact)
        .where(MemoryFact.user_id == user_id)
        .where(MemoryFact.scope == scope)
        .order_by(MemoryFact.learned_at.desc())
    )).scalars().all()
    by_cat: dict[str, list[dict]] = {k: [] for k in MEMORY_V2_CATEGORIES}
    for row in rows:
        by_cat.setdefault(_safe_category(row.category), []).append(
            fact_payload(row),
        )
    return {
        "count": len(rows),
        "categories": [
            {
                "key": key,
                "label": MEMORY_V2_CATEGORY_LABELS[key],
                "tone": MEMORY_V2_CATEGORY_TONES[key],
                "items": by_cat[key],
            }
            for key in MEMORY_V2_CATEGORIES
        ],
    }


def _domain_sort_key(domain: str) -> tuple:
    try:
        return (0, _DOMAIN_ORDER.index(domain))
    except ValueError:
        return (1, domain)


async def global_memory(
    db: AsyncSession, *, user_id: str, automation_titles: dict[str, str],
) -> dict:
    """The §4.5 GET /api/memory body: the whole store, grouped three
    ways — by life domain (global facts), by entity, and by teaching
    automation (scoped facts). `automation_titles` comes from the
    caller because automations are not this module's table to read."""
    facts = (await db.execute(
        select(MemoryFact)
        .where(MemoryFact.user_id == user_id)
        .order_by(MemoryFact.learned_at.desc())
    )).scalars().all()
    episodes = (await db.execute(
        select(MemoryEpisode)
        .where(MemoryEpisode.user_id == user_id)
        .order_by(MemoryEpisode.at.desc())
    )).scalars().all()
    entities = (await db.execute(
        select(MemoryEntity).where(MemoryEntity.user_id == user_id)
    )).scalars().all()

    # Global facts, domain → category. Only non-empty categories: the
    # always-five contract is the per-automation sheet's (§3.10), not
    # this screen's.
    by_domain: dict[str, dict[str, list[dict]]] = {}
    by_scope: dict[str, list[dict]] = {}
    for row in facts:
        if row.scope == MEMORY_V2_SCOPE_GLOBAL:
            dom = row.domain or "work"
            by_domain.setdefault(dom, {}).setdefault(
                _safe_category(row.category), [],
            ).append(fact_payload(row))
        else:
            by_scope.setdefault(row.scope, []).append(fact_payload(row))

    domains = [
        {
            "key": dom,
            "categories": [
                {
                    "key": key,
                    "label": MEMORY_V2_CATEGORY_LABELS[key],
                    "tone": MEMORY_V2_CATEGORY_TONES[key],
                    "facts": cats[key],
                }
                for key in MEMORY_V2_CATEGORIES if key in cats
            ],
        }
        for dom, cats in sorted(
            by_domain.items(), key=lambda kv: _domain_sort_key(kv[0]),
        )
    ]

    entity_blocks = []
    for ent in sorted(entities, key=lambda e: (e.kind, e.name_norm)):
        ent_facts = [
            fact_payload(f) for f in facts if f.subject_entity_id == ent.id
        ]
        ent_episodes = []
        for ep in episodes:
            try:
                ids = json.loads(ep.subject_entity_ids_json or "[]")
            except (ValueError, TypeError):
                ids = []
            if ent.id in ids:
                ent_episodes.append(_episode_payload(ep))
        entity_blocks.append({
            "entity": _entity_payload(ent),
            "facts": ent_facts,
            "episodes": ent_episodes,
        })

    from_your_automations = [
        {
            "automation_id": scope,
            "title": automation_titles.get(scope) or "Automation",
            "facts": scoped,
        }
        for scope, scoped in sorted(by_scope.items())
    ]

    return {
        "domains": domains,
        "entities": entity_blocks,
        "from_your_automations": from_your_automations,
        "counts": {
            "facts": len(facts),
            "episodes": len(episodes),
            "entities": len(entities),
        },
    }


async def recall(
    db: AsyncSession,
    *,
    user_id: str,
    query: Optional[str] = None,
    entity: Optional[str] = None,
    category: Optional[str] = None,
    scope: Optional[str] = None,
    since: Optional[str] = None,
    limit: int = 24,
) -> dict:
    """The `memory.recall` tool body: `{facts, episodes}`.

    `scope=None` spans every scope (main chat); an automation id widens
    to that scope PLUS global (an automation may use everything known
    about the user, but not a sibling's scoped view). `category`
    filters facts only — episodes carry outcomes, not categories.
    """
    limit = max(1, min(int(limit or 24), 100))
    since_dt = _parse_since(since)

    entity_ids: Optional[list[str]] = None
    if entity and str(entity).strip():
        entity_ids = await _matching_entity_ids(db, user_id, str(entity))
        if not entity_ids:
            return {"facts": [], "episodes": []}

    fq = select(MemoryFact).where(MemoryFact.user_id == user_id)
    if scope:
        if scope == MEMORY_V2_SCOPE_GLOBAL:
            fq = fq.where(MemoryFact.scope == MEMORY_V2_SCOPE_GLOBAL)
        else:
            fq = fq.where(MemoryFact.scope.in_(
                sorted({scope, MEMORY_V2_SCOPE_GLOBAL}),
            ))
    if category:
        fq = fq.where(MemoryFact.category == _safe_category(category))
    if query and str(query).strip():
        needle = f"%{str(query).strip()}%"
        fq = fq.where(or_(MemoryFact.text.ilike(needle),
                          MemoryFact.why.ilike(needle)))
    if entity_ids is not None:
        fq = fq.where(MemoryFact.subject_entity_id.in_(entity_ids))
    if since_dt is not None:
        fq = fq.where(MemoryFact.learned_at >= since_dt)
    fact_rows = (await db.execute(
        fq.order_by(MemoryFact.learned_at.desc()).limit(limit)
    )).scalars().all()

    eq = select(MemoryEpisode).where(MemoryEpisode.user_id == user_id)
    if scope:
        if scope == MEMORY_V2_SCOPE_GLOBAL:
            eq = eq.where(MemoryEpisode.automation_id.is_(None))
        else:
            eq = eq.where(MemoryEpisode.automation_id == scope)
    if query and str(query).strip():
        eq = eq.where(MemoryEpisode.text.ilike(f"%{str(query).strip()}%"))
    if entity_ids is not None:
        eq = eq.where(or_(*[
            MemoryEpisode.subject_entity_ids_json.like(f'%{eid}%')
            for eid in entity_ids
        ]))
    if since_dt is not None:
        eq = eq.where(MemoryEpisode.at >= since_dt)
    episode_rows = (await db.execute(
        eq.order_by(MemoryEpisode.at.desc()).limit(limit)
    )).scalars().all()

    return {
        "facts": [fact_payload(r) for r in fact_rows],
        "episodes": [_episode_payload(r) for r in episode_rows],
    }


# ── §4.5 migration ───────────────────────────────────────────────────


def _is_definition_fact(text: str) -> bool:
    """D-20: the fact describes the automation, not the user."""
    low = text.strip().lower()
    if low.startswith(_DEFINITION_PREFIX):
        return True
    return any(
        opener in low and closer in text
        for opener, closer in _DEFINITION_QUOTE_SHAPES
    )


def _is_status_fact(text: str) -> bool:
    """ND-2: run-status leakage — a moment, not a belief."""
    low = text.lower()
    return any(marker in low for marker in _STATUS_MARKERS)


_TIME_WORDS = ("time", "schedule", "meeting", "calendar")


def _migrate_category(old_category: str, text: str) -> str:
    """The §4.5 mapping — total, never raising on old data."""
    cat = (old_category or "").strip().lower()
    if cat == "people":
        return "people"
    if cat == "deadlines":
        return "your_time"
    if cat == "preferences":
        low = text.lower()
        if any(w in low for w in _TIME_WORDS):
            return "your_time"
        return "work_you_own"
    return _FALLBACK_CATEGORY


async def migrate_user(db: AsyncSession, *, user_id: str) -> dict:
    """Move the R29 `automation_facts` ledger into `memory_facts`.

    Idempotent by (text_hash, scope): a replay skips what the first
    pass wrote instead of duplicating it. Definition-facts and
    run-status leakage are DROPPED and counted — never silently.
    No brain projection here: R29 already projected these rows when
    they were written, and a migration that instructs the curator N
    times is a migration that costs N LLM calls.
    """
    from app.db.models import Automation, AutomationFact

    old_rows = (await db.execute(
        select(AutomationFact)
        .where(AutomationFact.user_id == user_id)
        .order_by(AutomationFact.created_at.asc())
    )).scalars().all()

    domains: dict[str, Optional[str]] = {}
    if old_rows:
        autos = (await db.execute(
            select(Automation.id, Automation.domain)
            .where(Automation.user_id == user_id)
        )).all()
        domains = {a_id: dom for a_id, dom in autos}

    existing = {
        (text_hash(row.text), row.scope)
        for row in (await db.execute(
            select(MemoryFact).where(MemoryFact.user_id == user_id)
        )).scalars().all()
    }

    counts = {"migrated": 0, "dropped_definition": 0,
              "dropped_status": 0, "skipped_existing": 0}
    for old in old_rows:
        text = _clean(old.text)
        if not text:
            continue
        if _is_definition_fact(text):
            counts["dropped_definition"] += 1
            continue
        if _is_status_fact(text):
            counts["dropped_status"] += 1
            continue
        scope = old.automation_id or MEMORY_V2_SCOPE_GLOBAL
        key = (text_hash(text), scope)
        if key in existing:
            counts["skipped_existing"] += 1
            continue

        entity = None
        person = _PERSON_RE.search(text)
        if person:
            entity = await _resolve_entity(
                db, user_id, "person", person.group(1),
            )

        db.add(MemoryFact(
            user_id=user_id,
            domain=domains.get(old.automation_id),
            category=_migrate_category(old.category, text),
            scope=scope,
            subject_entity_id=entity.id if entity is not None else None,
            text=text,
            why="Saved from an earlier version",
            learned_at=old.created_at or datetime.utcnow(),
            source="told" if old.source == "user" else "agent",
        ))
        existing.add(key)
        counts["migrated"] += 1

    await db.commit()
    logger.info("[memory-v2] migrated user=%s %s", user_id, counts)
    return counts


async def backfill_episodes(
    db: AsyncSession, *, user_id: str, limit: int = 200,
) -> int:
    """One episode per pre-v3 terminal run — the historical half of
    the engine's ledger-close writes. Idempotent: a run that already
    has any episode is left alone (the live writer mints several per
    run and must never be doubled by a back-fill)."""
    from app.db.models import Automation, BuildJob

    jobs = (await db.execute(
        select(BuildJob)
        .where(BuildJob.user_id == user_id)
        .where(BuildJob.job_type == "automation_run")
        .where(BuildJob.completed_at.is_not(None))
        .order_by(BuildJob.completed_at.desc())
        .limit(max(1, int(limit or 200)))
    )).scalars().all()
    if not jobs:
        return 0

    covered = {
        rid for (rid,) in (await db.execute(
            select(MemoryEpisode.run_id)
            .where(MemoryEpisode.user_id == user_id)
            .where(MemoryEpisode.run_id.is_not(None))
        )).all()
    }
    autos = {
        a_id: (name, dom)
        for a_id, name, dom in (await db.execute(
            select(Automation.id, Automation.name, Automation.domain)
            .where(Automation.user_id == user_id)
        )).all()
    }

    written = 0
    for job in jobs:
        if job.id in covered:
            continue
        name, dom = autos.get(job.source_id or "", (job.title, None))
        outcome = (job.user_message or job.outcome or job.status or "")
        cfg = job.config_json if isinstance(job.config_json, dict) else {}
        db.add(MemoryEpisode(
            user_id=user_id,
            domain=dom,
            automation_id=job.source_id,
            run_id=job.id,
            thread_id=cfg.get("thread_id"),
            at=job.completed_at,
            text=f"{name} — {outcome}"[:400],
            outcome=(job.outcome or job.status or "")[:24] or None,
        ))
        covered.add(job.id)
        written += 1
    if written:
        await db.commit()
    return written


# ── Brain projection (best-effort, always after the commit) ──────────


async def _project_fact(
    db: AsyncSession, user_id: str, *,
    category: str, domain: Optional[str], text: str,
) -> None:
    """The same seam the R29 facts ledger used: people through the
    curator's global instruct (it owns person identity), everything
    else filed under the fact's life domain. Imports stay inside the
    function — the platform image ships without `app/agent`."""
    try:
        if category == "people":
            from app.services import memory_curator
            await memory_curator.instruct_global(
                db, user_id,
                "Record what the user shared about people in their life "
                f"(merging with anything already known): {text}",
            )
            return
        from app.agent.automations.memory_notes import record_automation_fact
        await record_automation_fact(
            db, user_id=user_id, domain=domain or "work", fact=text,
        )
    except Exception as e:  # noqa: BLE001 — projection is a companion
        logger.warning(
            "[memory-v2] brain projection failed cat=%s: %s: %s",
            category, type(e).__name__, str(e)[:200],
        )
        try:
            await db.rollback()
        except Exception:  # noqa: BLE001
            pass


async def _project_removal(db: AsyncSession, user_id: str, text: str) -> None:
    """The agent must not keep "knowing" a deleted fact — mirror of
    R29 `facts._project_removal`, after the delete committed."""
    try:
        from app.services import memory_curator
        await memory_curator.instruct_global(
            db, user_id,
            f"The user deleted a saved fact — remove it if it is "
            f"recorded anywhere: {text}",
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("[memory-v2] removal projection failed: %s", e)
        try:
            await db.rollback()
        except Exception:  # noqa: BLE001
            pass
