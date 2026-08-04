"""
User Portrait Service — builds and caches a synthesized user profile.

Instead of dumping 15 scattered memory fragments into the system prompt,
this service creates a coherent 3-5 sentence portrait of who the user is,
what they're working on, and what matters to them.

The portrait is:
- Served from in-memory cache (instant) when available
- Rebuilt in the BACKGROUND when cache expires (never blocks chat)
- Built on first request with a lightweight fallback (no LLM call)
  and then upgraded via background task
"""

import asyncio
import logging
import time
import traceback
from datetime import datetime
from typing import Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func

from app.db.models import Memory
from app.services.background_tasks import spawn as _spawn_bg

logger = logging.getLogger(__name__)

# Categories that form the user portrait. Canonical values only — see
# app/memory_taxonomy.py. This was a stale copy of the taxonomy: it listed
# `projects` (retired, now `work`) and omitted `locations`/`skills` entirely,
# so a memory in either of those could reach the model through no path at all
# on turns where the category filter also excluded it.
PORTRAIT_CATEGORIES = {
    "identity": "Basic identity (name, age, background)",
    "work": "Professional context and active projects",
    "goals": "Current goals and aspirations",
    "preferences": "Key preferences and style",
    "knowledge": "Expertise and domain knowledge",
    "skills": "Abilities, education, what they're learning",
    "people": "Important people in their life",
    "locations": "Where they live and work",
    # `health` was missing entirely. Combined with similarity-gated retrieval
    # (a dinner question scores 0.072 against "severely allergic to peanuts",
    # under the 0.35 floor) that left an allergy reachable through NO path on
    # the turns where it matters most. Standing constraints are now also
    # injected directly by MemoryService.get_core_facts; this closes the second
    # hole rather than relying on that one alone.
    "health": "Health conditions, allergies, dietary needs, medications",
    "habits": "Routines and standing arrangements",
}

# In-memory cache: {user_id: {"summary": str, "ts": float}}
_portrait_cache: Dict[str, dict] = {}
# TKT-LAT-008: portraits are stable for hours (only change on memory
# writes / manual edit). Extended from 1h → 6h so the stale-while-
# revalidate fall-through fires far less often, and the [PERF] log
# line below shows hit/stale/miss explicitly so prod can verify.
_CACHE_TTL_SECONDS = 6 * 3600  # 6 hours

# Track in-flight background rebuilds to avoid duplicate work
_rebuilding: Dict[str, bool] = {}


def get_cached_portrait(user_id: str) -> str:
    """Return cached portrait if available, or empty string. Never blocks."""
    cached = _portrait_cache.get(user_id)
    if cached:
        return cached["summary"]
    return ""


class UserPortraitService:
    """Builds and maintains an evolving user portrait from memory categories."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def build_portrait(self, user_id: str) -> Dict:
        """Build a structured user portrait from high-strength memories."""
        portrait_parts: Dict[str, List[str]] = {}

        for category in PORTRAIT_CATEGORIES:
            result = await self.db.execute(
                select(Memory.content, Memory.strength)
                .where(
                    and_(
                        Memory.user_id == user_id,
                        Memory.brain_type == "user",
                        Memory.category == category,
                        Memory.is_deleted == False,
                        Memory.is_active == True,
                        Memory.strength >= 0.3,
                    )
                )
                .order_by(Memory.strength.desc(), Memory.importance.desc())
                .limit(5)
            )
            rows = result.all()
            if rows:
                portrait_parts[category] = [row.content for row in rows]

        if not portrait_parts:
            return {"summary": "", "raw": {}, "updated_at": None}

        # Synthesize a coherent portrait via LLM
        summary = await self._synthesize(portrait_parts)

        return {
            "summary": summary,
            "raw": portrait_parts,
            "updated_at": datetime.utcnow().isoformat(),
        }

    async def get_or_build_portrait(self, user_id: str) -> str:
        """Get cached portrait. If expired or missing, return stale/empty and rebuild in background.

        NEVER blocks the calling coroutine on an LLM call. The worst case is returning
        an empty string on the very first request, with the portrait appearing on the next message.
        """
        cached = _portrait_cache.get(user_id)

        if cached and (time.time() - cached["ts"]) < _CACHE_TTL_SECONDS:
            # Fresh cache — return immediately
            logger.info("[PERF] portrait_cache=hit user=%s", user_id[:8])
            return cached["summary"]

        if cached:
            # Stale cache — return stale value, rebuild in background
            logger.info("[PERF] portrait_cache=stale user=%s age=%.0fs",
                        user_id[:8], time.time() - cached["ts"])
            self._schedule_background_rebuild(user_id)
            return cached["summary"]

        # No cache at all — schedule build, return empty for this request
        logger.info("[PERF] portrait_cache=miss user=%s", user_id[:8])
        self._schedule_background_rebuild(user_id)
        return ""

    def _schedule_background_rebuild(self, user_id: str):
        """Schedule a background portrait rebuild if one isn't already running."""
        if _rebuilding.get(user_id):
            return  # Already in progress

        _rebuilding[user_id] = True
        _spawn_bg(self._background_rebuild(user_id))

    async def _background_rebuild(self, user_id: str):
        """Rebuild portrait in background with its own DB session. Updates cache when done."""
        try:
            from app.db.database import async_session_maker

            async with async_session_maker() as db:
                svc = UserPortraitService(db)
                portrait = await svc.build_portrait(user_id)
                summary = portrait["summary"]
                await db.commit()

            if summary:
                _portrait_cache[user_id] = {
                    "summary": summary,
                    "ts": time.time(),
                }
                logger.info(f"[PERF] portrait_rebuild: completed for user={user_id[:8]}… ({len(summary)} chars)")
            else:
                logger.info(f"[PERF] portrait_rebuild: no data for user={user_id[:8]}…")

        except Exception as e:
            logger.warning(
                f"Background portrait rebuild failed for user={user_id[:8]}…: "
                f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            )
        finally:
            _rebuilding.pop(user_id, None)

    def invalidate_cache(self, user_id: str):
        """Invalidate the cached portrait for a user."""
        _portrait_cache.pop(user_id, None)

    # ------------------------------------------------------------------

    async def _synthesize(self, parts: Dict[str, List[str]]) -> str:
        """Ask Claude to synthesize a coherent user portrait."""
        formatted = ""
        for category, contents in parts.items():
            label = PORTRAIT_CATEGORIES.get(category, category)
            formatted += f"\n{label}:\n"
            for c in contents:
                formatted += f"  - {c}\n"

        prompt = (
            "Based on these facts about a user, write a concise user portrait "
            "in 3-5 sentences. Focus on: who they are, what they're working on, "
            "and what matters to them.\n"
            "Do NOT list facts — write a flowing description as if describing a friend.\n"
            "Do NOT start with 'The user' — use their name if known, otherwise 'They'.\n\n"
            f"Facts by category:{formatted}\n"
            "Write ONLY the portrait, nothing else."
        )

        try:
            from app.services.llm_service import get_llm_service
            llm = get_llm_service()
            resp = await llm.complete(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=300,
            )
            text = resp.content.strip()
            if len(text) > 20:
                return text
        except Exception as e:
            logger.warning(f"Portrait synthesis failed: {e}")

        # Fallback: concatenate top facts (no LLM needed)
        lines = []
        for contents in parts.values():
            lines.extend(contents[:2])
        return ". ".join(lines[:5])


def get_user_portrait_service(db: AsyncSession) -> UserPortraitService:
    """Factory function."""
    return UserPortraitService(db)
