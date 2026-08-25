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
    return {
        "text": text,
        "category": category,
        "scope": "automation" if scope == "automation" else "global",
        "subject": " ".join(str(subject).split())[:80] if subject else None,
        "why": " ".join(str(item.get("why") or "").split())[:400] or None,
    }


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
                    subject_entity=({"kind": "person", "name": fact["subject"]}
                                    if fact.get("subject") else None),
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
