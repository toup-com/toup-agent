"""Radio Mode next-track picker — silent Haiku call.

Takes the current session (seed intent, played titles) and returns a
YouTube search query for the next track. Does NOT create an assistant
message or touch the conversation history.
"""
from __future__ import annotations

import logging
from typing import Optional

from app.agent.radio.session import RadioSession
from app.services.internal_llm import call_anthropic_system

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
    """Return a YouTube search query for the next track, or None on failure."""
    try:
        raw = await call_anthropic_system(
            user_id=sess.user_id,
            operation_type="system.radio_pick",
            model=_HAIKU_MODEL,
            max_tokens=60,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _build_user_prompt(sess)}],
            timeout=20,
        )
    except Exception as e:
        logger.warning("[radio/picker] Haiku call raised: %s", e)
        return None

    if not raw:
        return None

    # Strip quotes/newlines the model sometimes adds anyway.
    q = raw.strip().strip('"').strip("'").splitlines()[0].strip()
    if not q or len(q) > 200:
        logger.warning("[radio/picker] bad query output: %r", raw[:120])
        return None
    return q
