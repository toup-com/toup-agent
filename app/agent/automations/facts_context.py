"""Facts-aware event filtering — the "memory-filtered" leg of the
proactive draft flow (Round 29).

A filter needle of the exact form `{{facts.<category>}}` matches an
event field against the automation's curated fact ledger
(`automation_facts`, CONTRACTS-R29 §4) instead of a literal substring:

    "filter": {"from": ["{{facts.people}}"]}

reads "only fire for senders my memory knows". The matching direction
is deliberately REVERSED from literal needles: a fact is a sentence
("Boss: Sarah <sarah@acme.com> — replies urgent"), so the event
VALUE's tokens (email addresses, else the trimmed value) are searched
INSIDE the fact texts, case-insensitively. A category with no facts
matches nothing — an empty allowlist is an allowlist, and "draft for
everyone because memory is empty" would be the unsafe default.

Literal needles in the same list keep their v1 semantics; the needle
kinds OR together per field like any other needle. No spec change:
needle strings were always free-form, and no pre-R29 spec can contain
this shape (it would never have matched anything).

Both executors hook here: v1 `_passes_filter` and v2
`_passes_filter_v2` pass the pre-loaded context; loading happens once
per event via `load_facts_context` (None when the spec references no
facts — the common case costs one regex scan, no query).
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

FACTS_NEEDLE_RE = re.compile(r"^\{\{\s*facts\.([a-z][a-z0-9-]{1,31})\s*\}\}$")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_MIN_TOKEN_LEN = 3


def facts_needle_category(needle: object) -> Optional[str]:
    """The category a `{{facts.<category>}}` needle references, else
    None (a literal needle)."""
    if not isinstance(needle, str):
        return None
    m = FACTS_NEEDLE_RE.match(needle.strip())
    return m.group(1) if m else None


def referenced_categories(filter_rules: Optional[dict]) -> set[str]:
    """Every fact category any needle in these rules references."""
    cats: set[str] = set()
    for needles in (filter_rules or {}).values():
        if not isinstance(needles, list):
            needles = [needles]
        for n in needles:
            cat = facts_needle_category(n)
            if cat:
                cats.add(cat)
    return cats


async def load_facts_context(
    db, automation_id: str, filter_rules: Optional[dict],
) -> Optional[dict[str, list[str]]]:
    """Fact texts per referenced category, or None when the rules
    reference no facts. A referenced category always gets a key — an
    empty list means "knows nobody", which the matcher treats as
    match-nothing."""
    cats = referenced_categories(filter_rules)
    if not cats:
        return None
    ctx: dict[str, list[str]] = {c: [] for c in cats}
    try:
        from sqlalchemy import select
        from app.db.models import AutomationFact

        rows = (await db.execute(
            select(AutomationFact)
            .where(AutomationFact.automation_id == automation_id)
            .where(AutomationFact.category.in_(sorted(cats)))
        )).scalars().all()
        for row in rows:
            text = (row.text or "").strip()
            if text:
                ctx[row.category].append(text.lower())
    except ImportError:
        # The facts table ships with the R29-A half; a spec referencing
        # facts before it exists filters to nothing, which is honest.
        logger.info("[automations] facts filter: AutomationFact model "
                    "unavailable; treating categories %s as empty", cats)
    except Exception as e:  # noqa: BLE001 — filtering must never crash a run
        logger.warning(
            "[automations] facts context load failed automation=%s: %s",
            str(automation_id)[:8], e,
        )
    return ctx


def value_matches_facts(value: str, fact_texts: list[str]) -> bool:
    """Reverse containment: any token of the event value found inside
    any fact text. Tokens are the value's email addresses when it has
    them (the from-header case: `Sarah X <sarah@acme.com>`), else the
    whole trimmed value."""
    v = (value or "").strip().lower()
    if not v or not fact_texts:
        return False
    tokens = [t.lower() for t in _EMAIL_RE.findall(v)]
    if not tokens:
        tokens = [v]
    tokens = [t for t in tokens if len(t) >= _MIN_TOKEN_LEN]
    return any(t in fact for t in tokens for fact in fact_texts)


def needle_matches(
    needle: object,
    value_lower: str,
    facts_ctx: Optional[dict[str, list[str]]],
) -> bool:
    """One needle against one lowercased event value — facts needles
    match via the ledger, literal needles keep substring semantics.
    Callers that render `{{var.*}}` templates do so BEFORE this (a
    facts needle is never a var needle; the shapes are disjoint)."""
    cat = facts_needle_category(needle)
    if cat is not None:
        return value_matches_facts(value_lower, (facts_ctx or {}).get(cat) or [])
    n = str(needle).lower()
    return bool(n) and n in value_lower
