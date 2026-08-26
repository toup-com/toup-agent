"""Told facts from the day chat reach the one platform memory (R30 §5.6).

`memory_curator.curate_turn` stays the brain-file writer; this hook
runs AFTER it, only on turns where the curator actually applied
something (the cheapest honest gate: if nothing was durable enough for
the brain, nothing is durable enough for a fact row), and files the
exchange's told facts into `memory_facts` with the v2 classification —
category from the five keys, scope global (a day-chat fact is about the
person; automation scopes are only assigned inside automation threads),
subject entity, second-person evidence.
"""

from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)

_MAX_FACTS = 5


def _prompt(user_text: str, assistant_text: str) -> str:
    return (
        "You file durable facts about the user into one platform "
        "memory. The exchange happened in their main chat.\n\n"
        "From the exchange below, extract facts WORTH KEEPING. For "
        "each:\n"
        '- "category": people (who matters and how) · team_workspace '
        "(channels, ownership, team habits) · your_time (blocks, "
        "holds, when things reach the user) · work_you_own (surfaces, "
        "tickets, priorities) · noise_filters (what never surfaces).\n"
        '- "subject": the person/channel/ticket/repo it is about, or '
        "null.\n"
        '- "why": the evidence in one second-person sentence '
        '("You said Sarah is your boss.").\n'
        "Rules: only what the USER stated or clearly confirmed this "
        "turn; nothing inferred. NEVER file what an automation is or "
        "does, its schedule, its status, or run outcomes. One short "
        f"self-contained sentence each; dates absolute; at most "
        f"{_MAX_FACTS}; an empty list is the right answer for small "
        "talk.\n\n"
        f"USER: {user_text[:2000]}\n\nASSISTANT: {assistant_text[:1500]}\n\n"
        'Reply as JSON: {"facts": [{"text", "category", "subject", '
        '"why"}]}'
    )


async def file_told_facts(
    db,
    *,
    user_id: str,
    user_text: str,
    assistant_text: str,
    complete=None,
) -> int:
    """Extract and file the turn's told facts. Returns the count saved
    (0 on any failure — a background companion, never a veto)."""
    text = (user_text or "").strip()
    if not text:
        return 0
    try:
        if complete is None:
            complete = _default_complete
        payload = await complete(_prompt(text, assistant_text or ""))
        items = payload.get("facts") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            return 0

        from .curator_v2 import file_facts, normalize_candidate

        candidates = []
        for item in items[:_MAX_FACTS]:
            if isinstance(item, dict):
                item = dict(item)
                item["scope"] = "global"
            fact = normalize_candidate(item)
            if fact is not None:
                candidates.append(fact)
        if not candidates:
            return 0
        return await file_facts(
            db, user_id=user_id, facts=candidates,
            automation_id=None, source="told",
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("[automations] told-facts hook failed: %s: %s",
                       type(e).__name__, str(e)[:200])
        return 0


async def _default_complete(prompt: str) -> dict:
    from app.config import settings
    from app.services.llm_service import get_llm_service

    response = await get_llm_service().complete_with_json(
        messages=[{"role": "user", "content": prompt}],
        model=settings.memory_extraction_model,
        temperature=0.0,
    )
    raw = response.content if hasattr(response, "content") else response
    if isinstance(raw, str):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
        return json.loads(raw)
    return raw if isinstance(raw, dict) else {}
