"""
Feature flags — onboarding v2 rollout gate.

Used by:
  - `frontend/src/api.ts :: system.featureFlags()` via `GET /api/system/feature-flags`
  - `frontend/src/onboarding/OnboardingShell.tsx` to decide which LLM
    step UI to render (legacy binary card vs new 5-tier credit picker —
    actual branch lives in the PR 2 follow-up; this file ships the
    plumbing only)
  - Admin panel toggle at `PUT /api/admin/feature-flags/onboarding-v2`

Storage model:
  * Default lives in `settings.onboarding_v2_rollout_pct` (env var; 0).
  * Runtime override lives in `platform_settings` table, key
    `onboarding.v2_rollout_pct`, value = stringified integer 0-100.
  * DB row wins when present so admins can flip 10 → 50 → 100 without
    redeploying. Settings env-var default is only the fresh-deploy floor.

Per-user opt-in is deterministic — `user_id` → first 8 hex of SHA-256 →
int mod 100. Same user always gets the same answer at a given rollout %,
so users don't flip in/out of the cohort across requests. Anonymous /
pre-signup callers (the `/signup` page before account creation) hash the
caller IP if present, else fall back to "off" so we never expose the new
flow to an unattributed visitor.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models.platform import PlatformSetting

logger = logging.getLogger(__name__)


ONBOARDING_V2_KEY = "onboarding.v2_rollout_pct"


def _clamp_pct(value: int) -> int:
    if value < 0:
        return 0
    if value > 100:
        return 100
    return value


async def get_onboarding_v2_rollout_pct(db: AsyncSession) -> int:
    # DB row wins. settings env-var default is the boot-time floor for
    # fresh deploys where no admin has touched the toggle yet.
    row = (await db.execute(
        select(PlatformSetting).where(PlatformSetting.key == ONBOARDING_V2_KEY)
    )).scalar_one_or_none()
    if row is not None:
        try:
            return _clamp_pct(int((row.value or "").strip()))
        except (ValueError, TypeError):
            logger.warning(
                "feature_flags: %s has non-integer value %r, falling back to env",
                ONBOARDING_V2_KEY, row.value,
            )
    return _clamp_pct(settings.onboarding_v2_rollout_pct)


async def set_onboarding_v2_rollout_pct(db: AsyncSession, pct: int) -> int:
    pct = _clamp_pct(pct)
    row = (await db.execute(
        select(PlatformSetting).where(PlatformSetting.key == ONBOARDING_V2_KEY)
    )).scalar_one_or_none()
    if row is None:
        db.add(PlatformSetting(key=ONBOARDING_V2_KEY, value=str(pct)))
    else:
        row.value = str(pct)
    await db.commit()
    return pct


def _hash_to_bucket(seed: str) -> int:
    # First 8 hex chars of SHA-256 → 32-bit int → mod 100. Deterministic
    # per seed, uniformly distributed enough for rollout cohorts of any
    # size we'll see in production.
    h = hashlib.sha256((seed or "").encode("utf-8")).hexdigest()
    return int(h[:8], 16) % 100


def is_onboarding_v2_enabled_for(seed: Optional[str], rollout_pct: int) -> bool:
    if rollout_pct <= 0:
        return False
    if rollout_pct >= 100:
        # Full launch: everyone gets v2, including anonymous visitors
        # mid-signup. The old flow stays in the codebase per the
        # 7-day soak per the rollout plan, but no one is routed to it.
        return True
    # Partial rollout: anonymous visitors (no user_id, no IP) stay on
    # the stable flow until we have a signal we can pin them to.
    # Bucketing on a missing seed would flip them between cohorts on
    # every refresh, which is worse than just keeping them on a
    # consistent path through their session.
    if not seed:
        return False
    return _hash_to_bucket(seed) < rollout_pct


async def is_onboarding_v2_enabled(
    db: AsyncSession, seed: Optional[str] = None,
) -> bool:
    pct = await get_onboarding_v2_rollout_pct(db)
    return is_onboarding_v2_enabled_for(seed, pct)
