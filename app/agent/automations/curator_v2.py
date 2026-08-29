"""Curator v2 — classification in, one platform memory out (R30 §4.5, §5.6).

The write half of the memory contract: candidates (extracted from a
thread exchange, told in any chat, or observed as reactions) pass the
`curation_rules` refusal gate, get a category from the five keys, a
scope, a subject entity and second-person evidence, and land in the one
platform store through A's `memory_v2_service.add_fact` seam — which
dedupes across scopes, honours 30-day forget suppression, resolves
entities and projects to the brain.

Until the integration merge, `memory_v2_service` may be absent on this
branch; the fallback files through the R29 `facts.record` seam with the
legacy category mapping so nothing is lost and the branch stays green —
A's migration re-files legacy rows into v2 shapes.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from .curation_rules import (
    CATEGORY_KEYS,
    LEGACY_CATEGORY_MAP,
    refuse_reason,
)

logger = logging.getLogger(__name__)

#: v2 → legacy bucket for the pre-merge fallback path. team_workspace
#: and noise_filters have no legacy home; they ride `preferences` until
#: A's migration re-classifies (better carried imprecisely than lost).
_V2_TO_LEGACY = {
    "people": "people",
    "your_time": "preferences",
    "work_you_own": "deadlines",
    "team_workspace": "preferences",
    "noise_filters": "preferences",
}

#: Accept model output in either vocabulary.
_NORMALIZE_CATEGORY = {**{k: k for k in CATEGORY_KEYS}, **LEGACY_CATEGORY_MAP}


def normalize_candidate(item: Any) -> Optional[dict]:
    """One extracted candidate → the v2 fact shape, or None when it is
    malformed, mis-categorised, or refused by the gate."""
    if not isinstance(item, dict):
        return None
    text = " ".join(str(item.get("text") or "").split())[:400]
    if not text:
        return None
    category = _NORMALIZE_CATEGORY.get(
        str(item.get("category") or "").strip().lower())
    if category is None:
        return None
    refused = refuse_reason(text)
    if refused:
        logger.info("[automations] curator refused (%s): %.80s", refused, text)
        return None
    scope = str(item.get("scope") or "").strip().lower()
    subject = item.get("subject")
    subject_name = " ".join(str(subject).split())[:80] if subject else None
    # ── One subject, one entity (round 33, item 6) ───────────────────────
    # A comma-joined subject was resolved as ONE entity, which is how
    # "#all-toup, #social" became a single row on the People & things tab.
    # Two channels are two things; if the model gives us a list we cannot
    # honestly index the fact under one of them, so it goes unindexed —
    # the fact is still stored and still readable, it just does not claim
    # to be about an entity that does not exist.
    if subject_name and ("," in subject_name or " and " in subject_name):
        subject_name = None
    return {
        "text": text,
        "category": category,
        "scope": "automation" if scope == "automation" else "global",
        "subject": subject_name,
        # What the subject IS. The model is asked for it; when it does not
        # answer, the NAME's shape decides — and the three shapes that
        # decide are exactly the three that were mis-typed on the founder's
        # Memory page ("#all-toup" a channel, "SCRUM-1" a ticket,
        # "toup-com/toup-platform" a repo, every one of them filed as a
        # person). Anything else is a person, which is what it was and what
        # it usually is: dropping the entity for every unlabelled fact
        # would un-index real people to fix three rows.
        "subject_kind": _entity_kind(item.get("subject_kind"), subject_name),
        "why": " ".join(str(item.get("why") or "").split())[:400] or None,
    }


#: `memory_v2.MEMORY_ENTITY_KINDS`, restated here so this module does not
#: import the model layer for one frozenset.
_ENTITY_KINDS = ("person", "channel", "ticket", "repo", "project", "account")


#: Names whose shape says what they are. Narrow on purpose — a guess that is
#: wrong is the defect this replaces.
_TICKET_RE = re.compile(r"^[A-Z][A-Z0-9]*-\d+$")


def _subject_entity(fact: dict) -> Optional[dict]:
    """`{kind, name}` for a fact's subject, or None when it names none."""
    name = fact.get("subject")
    if not name:
        return None
    kind = _entity_kind(fact.get("subject_kind"), str(name))
    return {"kind": kind, "name": name} if kind else None


def _entity_kind(raw, name: Optional[str] = None) -> Optional[str]:
    kind = str(raw or "").strip().lower()
    if kind in _ENTITY_KINDS:
        return kind
    n = (name or "").strip()
    if not n:
        return None
    if n.startswith("#"):
        return "channel"
    if _TICKET_RE.match(n):
        return "ticket"
    if "/" in n:
        return "repo"
    return "person"


async def file_facts(
    db,
    *,
    user_id: str,
    facts: list[dict],
    automation_id: Optional[str] = None,
    domain: Optional[str] = None,
    source: str = "agent",
    run_id: Optional[str] = None,
) -> int:
    """File normalized candidates into the platform memory. Returns the
    number saved (suppressed and duplicate facts count zero)."""
    if not facts:
        return 0
    try:
        from app.services import memory_v2_service
    except ImportError:
        memory_v2_service = None

    saved = 0
    if memory_v2_service is not None:
        for fact in facts:
            try:
                result = await memory_v2_service.add_fact(
                    db,
                    user_id=user_id,
                    text=fact["text"],
                    category=fact["category"],
                    scope=(automation_id if fact["scope"] == "automation"
                           and automation_id else "global"),
                    why=fact.get("why"),
                    source=source,
                    domain=domain,
                    # Only index a fact under an entity when we know BOTH
                    # what it is called and what it IS. This hardcoded
                    # "person" for every subject, so every channel, ticket
                    # and repo the curator ever saw is typed as a person in
                    # `memory_entities` — and the app prints that kind raw.
                    # `_entity_kind` again, not `fact["subject_kind"]`: this
                    # is a public entry point and is called with raw dicts
                    # that never went through `normalize_candidate`, so it
                    # cannot assume the key is there.
                    subject_entity=_subject_entity(fact),
                    run_id=run_id,
                )
                if result and result.get("saved"):
                    saved += 1
                elif result and result.get("suppressed"):
                    logger.info("[automations] curator: fact suppressed "
                                "by a forget signal")
            except Exception as e:  # noqa: BLE001 — one bad fact never vetoes the rest
                logger.warning("[automations] curator v2 write failed: %s: %s",
                               type(e).__name__, str(e)[:200])
        return saved

    # Pre-merge fallback: the R29 seam, legacy categories.
    if automation_id is None:
        logger.info("[automations] curator: v2 store absent and no "
                    "automation scope — %d fact(s) not filed", len(facts))
        return 0
    from . import facts as facts_seam

    by_category: dict[str, list[str]] = {}
    for fact in facts:
        by_category.setdefault(
            _V2_TO_LEGACY[fact["category"]], []).append(fact["text"])
    for category, texts in by_category.items():
        result = await facts_seam.record(
            db, user_id=user_id, automation_id=automation_id,
            facts=texts, category=category, source=source,
            source_kind="interview", run_id=run_id,
        )
        saved += int((result or {}).get("saved", 0))
    return saved
