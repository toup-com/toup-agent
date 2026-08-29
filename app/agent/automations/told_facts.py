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
    # ── The two blocks are NOT symmetric (round 33, item 6) ─────────────
    # This used to end `USER: … ASSISTANT: …` with one prose line asking
    # for "only what the USER stated". The v3 curator has always labelled
    # them asymmetrically, on purpose, and it is the reason it does not
    # make this mistake: the reply is CONTEXT, never a source. Without the
    # label the model lifted the agent's own connector failures out of the
    # reply and — obeying the second-person "why" rule below — filed them
    # as the user's testimony: "You have access to the Slack channels
    # #all-toup and #social, but you don't have message-reading access
    # there. · You mentioned that…". The user said no such thing.
    #
    # `TURN_DURABILITY_RULES` is IMPORTED, not forked: it already bans
    # this class by name ("Tool, search or playback OUTCOMES… 'You have
    # Gmail messages' is a tool result, not a fact about a life"), and two
    # copies of that text is how the two writers drift apart again.
    from app.services.memory_curator import TURN_DURABILITY_RULES
    return (
        "You file durable facts about the user into one platform "
        "memory. The exchange happened in their main chat.\n\n"
        f"{TURN_DURABILITY_RULES}\n\n"
        "From the exchange below, extract facts WORTH KEEPING. For "
        "each:\n"
        '- "category": people (who matters and how) · team_workspace '
        "(channels, ownership, team habits) · your_time (blocks, "
        "holds, when things reach the user) · work_you_own (surfaces, "
        "tickets, priorities) · noise_filters (what never surfaces).\n"
        '- "subject": the person/channel/ticket/repo it is about, or '
        "null.\n"
        '- "subject_kind": what that subject IS — one of person, '
        "channel, ticket, repo, project, account. Omit it when there is "
        "no subject. Do not guess: an unindexed fact is better than a "
        "Slack channel filed as a person.\n"
        '- "why": the evidence in one second-person sentence '
        '("You said Sarah is your boss."). It must quote or paraphrase '
        "something in the USER block. If the only support for a fact is "
        "in the reply, the fact is not the user's and does not belong "
        "here.\n"
        "Rules: only what the USER stated or clearly confirmed this "
        "turn; nothing inferred. NEVER file what an automation is or "
        "does, its schedule, its status, or run outcomes; never what "
        "YOU can or cannot read, reach or access, and never the status "
        "of a ticket, issue or message in someone else's system. One "
        f"short self-contained sentence each; dates absolute; at most "
        f"{_MAX_FACTS}; an empty list is the right answer for small "
        "talk.\n\n"
        f"WHAT THE USER SAID (the ONLY source of facts):\n"
        f"{user_text[:2000]}\n\n"
        f"WHAT YOU REPLIED (CONTEXT ONLY — never a source of facts):\n"
        f"{assistant_text[:1500]}\n\n"
        'Reply as JSON: {"facts": [{"text", "category", "subject", '
        '"subject_kind", "why"}]}'
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
    # ── The v3 pre-gate, before an LLM call (round 33, item 6) ──────────
    # This gated on non-empty user text alone, so a pure QUESTION — "can
    # you read messages in all my channels?" — was handed to the extractor
    # with the agent's failure report attached, and the only assertions in
    # the window were the agent's own. The v3 writer skips that turn by
    # name (`question_only`), and running the same gate here costs one
    # function call and saves the LLM round trip as well.
    try:
        from app.services.memory_curator import turn_skip_reason
        _skip = turn_skip_reason(text)
    except Exception:  # noqa: BLE001 — the gate never vetoes on its own failure
        _skip = None
    if _skip:
        logger.info("[automations] curator skipped this turn (%s)", _skip)
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
