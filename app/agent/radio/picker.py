"""Radio Mode next-track picker — silent background LLM call.

Takes the current session (seed intent, played titles) and returns a
YouTube search query for the next track. Does NOT create an assistant
message or touch the conversation history.

Uses the agent's configured AnthropicService rather than internal_llm
because most deployments run on Claude Code OAuth (no raw API key
present for internal_llm to use).
"""
from __future__ import annotations

import logging
from typing import Optional

from app.agent.radio.session import RadioSession
from app.services.anthropic_service import get_anthropic_service

logger = logging.getLogger(__name__)

_HAIKU_MODEL = "claude-haiku-4-5-20251001"

_SYSTEM_PROMPT = (
    "You are a music curator powering a radio-mode feature. "
    "Given a seed request and recently played tracks, pick ONE new track that "
    "matches the vibe (same artist, adjacent artists, similar genre, energy, mood, era). "
    "Rules:\n"
    "- Never repeat a track that's in the recently-played list.\n"
    "- Prefer diversity over same-artist repetition after the first 2 tracks.\n"
    "- Match the seed's energy and context (gym → high-energy, chill → mellow, etc.).\n"
    "- For non-English seeds, stay in the same language/region unless the user asked otherwise.\n"
    "- Output ONLY a single YouTube search query. No prose, no quotes, no commentary.\n"
    "- Format: 'Artist - Track Title' (e.g. 'Eminem - Till I Collapse').\n"
    "- If unsure, pick a well-known track from the seed artist's top catalog that isn't in history."
)


def _build_user_prompt(sess: RadioSession) -> str:
    seed = sess.seed_intent or "music"
    seed_title = sess.seed_track.title if sess.seed_track else ""
    played = "\n".join(f"- {t}" for t in sess.played_titles[-15:] if t) or "(none yet)"
    return (
        f"Seed request: {seed}\n"
        f"Seed track: {seed_title}\n\n"
        f"Recently played (DO NOT repeat):\n{played}\n\n"
        f"Pick the next track. Output one search query only."
    )


async def pick_next_query(sess: RadioSession) -> Optional[str]:
    """Return a YouTube search query for the next track, or None on failure.

    Uses the agent's AnthropicService so OAuth-authenticated deployments
    (Claude Code mode) work without a separate API key.
    """
    svc = get_anthropic_service()
    try:
        resp = await svc.create_message(
            messages=[{"role": "user", "content": _build_user_prompt(sess)}],
            system=_SYSTEM_PROMPT,
            model=_HAIKU_MODEL,
            max_tokens=60,
            temperature=0.7,
        )
        raw = (resp.content or "").strip() if resp else ""
    except Exception as e:
        logger.warning("[radio/picker] LLM call raised: %s", e)
        print(f"[radio/picker] LLM call raised: {type(e).__name__}: {e}", flush=True)
        return None

    if not raw:
        print(f"[radio/picker] empty response seed_intent={sess.seed_intent!r}", flush=True)
        return None

    # Strip quotes/newlines the model sometimes adds anyway.
    q = raw.strip().strip('"').strip("'").splitlines()[0].strip()
    if not q or len(q) > 200:
        logger.warning("[radio/picker] bad query output: %r", raw[:120])
        print(f"[radio/picker] bad query output raw={raw[:120]!r}", flush=True)
        return None
    print(f"[radio/picker] picked query={q!r} seed_intent={sess.seed_intent!r}", flush=True)
    return q
