"""The `learned` file's producer — what the agent learns about working with
a user.

This is the file the Memory page shows under "Learned" and the block the
prompt injects as "how to work with this user". Until 2026-07-29 it was
permanently empty: `brain_type='agent'` had no producer anywhere in the
codebase (0 rows across all 54 production containers) and no consumer either,
since the runtime discarded agent-brain rows during retrieval.

v3: this module still owns the CHEAP GATE and the extraction call — a regex
sweep on every turn, a model call only on turns that look like a correction —
and hands the result to `memory_curator.instruct_file(LEARNED_SLUG, …)`. It
no longer writes rows, chooses categories, or talks to the dedup service:
the curator is the one writer, and a second one with its own idea of the
rules is exactly what v3 exists to remove.

Scope is deliberately narrow. The agent's IDENTITY (its name, personality,
voice) is owned by SoulConfig and is NOT memory — an earlier `agent_soul`
category was retired precisely because it competed with Soul, and every Soul
save still archives those rows. What lives here instead is the relational
knowledge Soul cannot hold: how THIS user works, what they have corrected, how
they want to be spoken to.

Cost control
------------
Extraction is a THIRD LLM call per turn, so it does not run per turn. A cheap
lexical gate fires first, and only turns that look like a correction or a
stated working preference reach the model. On a normal turn this module costs
one regex sweep and nothing else.
"""

import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.memory_files import LEARNED_SLUG
from app.services.memory_log import describe_memory

logger = logging.getLogger(__name__)


# The user telling the agent it got something wrong, or telling it how to
# behave. These are the only turns worth spending a reflection call on.
_REFLECTION_SIGNALS = [
    # ── Corrections ──
    # A leading "no"/"nope" only counts when what follows looks like a
    # correction. A bare `\b(no|nope)\b[\s,.!]` was tried first and was far too
    # greedy — it fired on "no idea what to cook", "no rush on this" and
    # "no worries, can you check my calendar", which are ordinary conversation.
    re.compile(r"^\s*(?:no|nope)\b[\s,.!]+(?:that|this|it|i|you|we|the|my)\b", re.IGNORECASE),
    re.compile(r"\b(?:that's|thats|this is|you are|you're)\s+(?:not right|not correct|wrong|incorrect)", re.IGNORECASE),
    re.compile(r"\b(?:wrong|incorrect)\b[\s,.!]*$", re.IGNORECASE),
    re.compile(r"\bactually,?\s+(?:it|i|we|the|my|you)\b", re.IGNORECASE),
    re.compile(r"\bi\s+(?:didn't|did not|never)\s+(?:say|ask|mean|want)", re.IGNORECASE),
    re.compile(r"\bi\s+meant\b", re.IGNORECASE),
    re.compile(r"\b(?:you|it)\s+(?:got|had)\s+(?:it|that)\s+wrong\b", re.IGNORECASE),
    re.compile(r"\bnot\s+what\s+i\s+(?:asked|meant|wanted)\b", re.IGNORECASE),
    # ── Stated instructions about how to behave ──
    re.compile(r"\b(?:stop|don't|do not|never)\s+(?:doing|saying|adding|using|asking|sending|showing|including)", re.IGNORECASE),
    re.compile(r"\b(?:from now on|in future|in the future|next time)\b", re.IGNORECASE),
    re.compile(r"\balways\s+(?:use|send|reply|answer|write|give|show|include|ask|keep|put)\b", re.IGNORECASE),
    re.compile(r"\bi\s+(?:prefer|'d rather|would rather|like it when)\b", re.IGNORECASE),
    re.compile(r"\b(?:be|keep it|make it)\s+(?:shorter|briefer|concise|brief|terse|detailed)\b", re.IGNORECASE),
]

# Guardrail: the reflection prompt is fed the assistant's own text, so it is
# the single most likely place to mint sycophantic self-praise. Anything
# matching this is dropped before it can be stored.
_SELF_PRAISE = re.compile(
    r"\b(?:did (?:a )?(?:great|good|excellent)|was (?:helpful|correct|right)|"
    r"successfully|worked well|user (?:was )?(?:happy|satisfied|pleased))\b",
    re.IGNORECASE,
)

MAX_REFLECTIONS_PER_TURN = 3


def should_reflect(user_message: str) -> bool:
    """Cheap gate — does this turn plausibly contain a correction or a rule?"""
    if not user_message or len(user_message) < 8:
        return False
    return any(p.search(user_message) for p in _REFLECTION_SIGNALS)


def _build_prompt(user_message: str, assistant_response: str) -> str:
    return f"""You maintain an AI assistant's notes about how to work with ONE specific user.

The user just said something that looks like a correction or an instruction about
how the assistant should behave. Capture what the assistant should do differently
from now on.

USER MESSAGE:
{user_message}

ASSISTANT RESPONSE (what the user was reacting to):
{assistant_response}

## Rules
- Write each note as an instruction to the assistant, in the second person:
  "Send reminders to the current chat, not Telegram." / "Keep answers short unless asked."
- ONLY record something the USER actually expressed. Never invent a preference.
- NEVER record praise, successes, or how well the assistant did. These notes exist
  to change future behaviour, not to flatter. If the user was simply happy, return [].
- Do NOT record facts about the user's life (their job, family, plans). Those belong
  in a different store and will be captured separately.
- Do NOT record anything about the assistant's name, personality or persona.
- Prefer ONE precise note over several vague ones. Returning [] is the correct
  answer whenever the user was not actually correcting or instructing.

Return ONLY valid JSON:
{{
  "notes": [
    {{"content": "Second-person instruction, one standalone sentence"}}
  ]
}}

At most {MAX_REFLECTIONS_PER_TURN} notes. If nothing qualifies, return {{"notes": []}}."""


async def extract_agent_reflections(
    user_message: str,
    assistant_response: str,
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Run the reflection LLM call. Returns [] on any failure — never raises."""
    from app.services.llm_service import get_llm_service, LLMService

    try:
        llm = LLMService(api_key=api_key) if api_key else get_llm_service()
        response = await llm.complete_with_json(
            messages=[{"role": "user", "content": _build_prompt(user_message, assistant_response)}],
            temperature=0.2,
            max_tokens=600,
        )
        raw = (response.content or "").strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
        parsed = json.loads(raw)
    except Exception as e:
        logger.warning("[agent_reflection] extraction failed: %s: %s", type(e).__name__, str(e)[:200])
        return []

    notes: List[Dict[str, Any]] = []
    for item in (parsed.get("notes") or [])[:MAX_REFLECTIONS_PER_TURN]:
        content = (item.get("content") or "").strip()
        if len(content) < 15 or content.count(" ") < 3:
            continue
        if _SELF_PRAISE.search(content):
            logger.info("[agent_reflection] dropped self-praise: %s", describe_memory(content))
            continue
        notes.append({"content": content})
    return notes


async def store_agent_reflections(
    db: AsyncSession,
    user_id: str,
    notes: List[Dict[str, Any]],
    api_key: Optional[str] = None,
) -> int:
    """Write the notes into the `learned` FILE, through the curator.

    v3: corrections are bullets in one agent-facing file, not agent-brain
    rows. The call is scoped with `instruct_file(LEARNED_SLUG, …)` — the
    same entry the Learned page's own input box uses — so a reflection can
    never land in you/profile or invent a people/ file, and the writer's
    merge rule applies: a correction the user has made before rewrites the
    existing bullet instead of stacking a second one.

    Returns the number of ops applied (0 is the ordinary answer).
    """
    if not notes:
        return 0

    from app.services import memory_curator

    # ONE call for all of them. The notes come from a single turn about a
    # single correction; three separate calls would let the second and third
    # dedupe against a file the first had already changed, which is the same
    # ordering hazard the ops engine's single-walk simulation exists to
    # remove.
    instruction = (
        "The user just corrected you or told you how they want things done. "
        "Record what you should do differently from now on:\n"
        + "\n".join(f"- {n['content']}" for n in notes)
    )
    try:
        result = await memory_curator.instruct_file(
            db, user_id, LEARNED_SLUG, instruction, api_key=api_key,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "[agent_reflection] curator call failed: %s: %s",
            type(e).__name__, str(e)[:200],
        )
        return 0

    applied = int(result.get("applied", 0))
    if applied:
        logger.info(
            "[agent_reflection] learned x%d: %s", applied,
            describe_memory(notes[0]["content"]),
        )
    elif result.get("rejected"):
        logger.info("[agent_reflection] declined: %s", result["rejected"][:2])
    return applied


async def _resolve_tenant_api_key(db: AsyncSession, user_id: str) -> Optional[str]:
    """Fetch the tenant's own OpenAI key so reflection bills them, not us.

    Mirrors AgentRunner._extract_memories. Without this the reflection call
    would fall back to the shared platform key — the same mis-attribution the
    relationship extractor already has.
    """
    try:
        from sqlalchemy import select

        from app.db import AgentConfig

        async with db.begin_nested():
            result = await db.execute(
                select(AgentConfig.openai_api_key).where(AgentConfig.user_id == user_id)
            )
            return result.scalar_one_or_none()
    except Exception:
        return None


async def reflect_on_turn(
    db: AsyncSession,
    user_id: str,
    user_message: str,
    assistant_response: str,
    api_key: Optional[str] = None,
) -> int:
    """Gate → extract → store. Safe to call on every turn; usually a no-op."""
    if not should_reflect(user_message):
        return 0
    if api_key is None:
        api_key = await _resolve_tenant_api_key(db, user_id)

    # Release the pooled connection BEFORE the LLM round-trip — same defect as
    # #407 fixed in day_summarizer, reached by a different path so that scan's
    # `call_system_llm` predicate could not match it (this goes through
    # LLMService.complete_with_json). _resolve_tenant_api_key above is a read,
    # and a read autobegins a transaction that pins the connection until the
    # session commits; store_agent_reflections below simply re-acquires one.
    #
    # It matters here for the same reason it mattered there, and more often:
    # the caller is agent_runner's _background_post_processing, fire-and-forget
    # on every turn, on a path that dies routinely (a voice caller hanging up
    # cancels the parent 1.5s later). A death mid-call left the connection
    # checked out forever -> GC terminated it -> the pool degraded -> later
    # turns died on PendingRollbackError -> 500 from /internal/agent-turn.
    #
    # Best-effort: releasing the connection is an optimisation, and this
    # function's whole contract is "safe to call on every turn". A failed
    # release must not become a raised exception the caller has to absorb.
    try:
        await db.commit()
    except Exception as e:
        logger.warning(
            "[agent_reflection] could not release the connection before the LLM "
            "call (continuing, connection stays pinned): %s", e,
        )

    notes = await extract_agent_reflections(user_message, assistant_response, api_key=api_key)
    if not notes:
        return 0
    return await store_agent_reflections(db, user_id, notes, api_key=api_key)
