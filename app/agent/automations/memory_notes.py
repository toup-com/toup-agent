"""Memory domains + clean fact-writing for automations (Round 28).

Two memory stores exist and this module touches exactly ONE of them:

  - The engine's working state (`{{memory.<key>}}`, R28-A §6) is an
    isolated `Memory` row per automation — machine state, invisible to
    the brain. NOT this module.
  - The user's brain (memory v3 FILES) — human facts. THIS module, and
    only through the curator (`instruct_file`), the same sanctioned
    scoped entry `agent_reflection` uses. Never by lifting
    `disable_post_processing` (pinned by test_curator_producers), never
    raw provider payloads: the instruction we hand the curator is
    composed from the automation's NAME, its DOMAIN, and a plain-English
    outcome — nothing from a tool result ever reaches it, and the
    curator's own durability rules + bullet lint still apply on top.

Domains map onto the v3 `areas/` namespace: an automation assigned to
"work" files its facts in `areas/work`. Domain is setup metadata on the
Automation row (nullable — R26 rows have none and write no facts).
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

CANONICAL_DOMAINS: tuple[str, ...] = ("work", "university", "personal")

# Custom domains are slugs: lowercase, digit/hyphen tail, 2-32 chars —
# the same shape the v3 slug namespace accepts under areas/.
_DOMAIN_RE = re.compile(r"^[a-z][a-z0-9-]{1,31}$")


def normalize_domain(value: object) -> Optional[str]:
    """Canonical domain string, or None when the value isn't one.
    Accepts the three canonical domains and any custom slug."""
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    if not v:
        return None
    if v in CANONICAL_DOMAINS:
        return v
    if _DOMAIN_RE.fullmatch(v):
        return v
    return None


def domain_file_slug(domain: str) -> str:
    """The brain file a domain's facts live in."""
    return f"areas/{domain}"


def _clean_line(text: str, limit: int = 200) -> str:
    """One printable line — the composer's half of "clean facts": no
    newlines, no runaway length. The curator's lint is the enforcer;
    this keeps us from asking it to reject us."""
    line = " ".join(str(text).split())
    return line[:limit]


async def record_automation_fact(
    db: AsyncSession,
    *,
    user_id: str,
    domain: Optional[str],
    fact: str,
) -> bool:
    """File one clean fact under the domain's areas/ file, through the
    curator. Best-effort: False means "not written", never an exception
    — memory is a companion to the engine, not a dependency of it.

    The target file is created on first use (`create_file` is part of
    the curator's op contract); facts about an automation the user
    deleted are the curator's merge problem, not ours.
    """
    dom = normalize_domain(domain)
    if dom is None or not fact or not str(fact).strip():
        return False
    slug = domain_file_slug(dom)
    instruction = (
        f"Keep the user's {dom} area file up to date. "
        f"Record (merging with anything it already says): "
        f"{_clean_line(fact)}"
    )
    try:
        from app.services import memory_curator
        from app.services import memory_file_ops as ops

        # instruct_file refuses a missing file; make sure the domain
        # file exists first with a deterministic create (no LLM) —
        # the same validate→apply walk every writer uses.
        rows = await ops._all_files(db, user_id)
        if not any(r.slug == slug for r in rows):
            plan = ops.validate_ops(
                [{
                    "op": "create_file",
                    "section": "areas",
                    "slug": slug,
                    "title": dom.capitalize(),
                    # The ops lint requires the house description shape:
                    # '<what this is> — <scope>; read when <trigger>.'
                    "description": (
                        f"The user's {dom} area — ongoing facts and "
                        f"commitments; read when a conversation touches "
                        f"their {dom} matters."
                    ),
                }],
                rows,
            )
            applied_create = await ops.apply_ops(db, user_id, plan)
            await db.commit()
            if not applied_create.get("applied"):
                logger.info(
                    "[automations] domain file create declined slug=%s: %s",
                    slug, plan.complaints[:2] if plan.complaints else "",
                )
                return False
        result = await memory_curator.instruct_file(
            db, user_id, slug, instruction,
        )
        applied = int(result.get("applied", 0))
        if applied:
            logger.info(
                "[automations] filed %d fact op(s) in %s", applied, slug,
            )
        return applied > 0
    except Exception as e:  # noqa: BLE001 — see docstring
        logger.warning(
            "[automations] fact write failed slug=%s: %s: %s",
            slug, type(e).__name__, str(e)[:200],
        )
        try:
            await db.rollback()
        except Exception:  # noqa: BLE001
            pass
        return False


def setup_fact(
    *,
    automation_name: str,
    trigger_summary: str,
    action_summary: str,
) -> str:
    """The fact filed when an automation is armed — composed from setup
    intent, never from provider data."""
    return (
        f"Has an automation \"{_clean_line(automation_name, 60)}\" that "
        f"{_clean_line(trigger_summary, 80)} and "
        f"{_clean_line(action_summary, 80)}."
    )
