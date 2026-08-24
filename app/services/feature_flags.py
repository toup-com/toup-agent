"""
Feature flags — deterministic percentage rollouts.

Used by:
  - `frontend/src/api.ts :: system.featureFlags()` via `GET /api/system/feature-flags`
  - `frontend/src/onboarding/OnboardingShell.tsx` for the LLM step UI
  - `frontend/src/shared/mobileShell.ts` for the mobile web shell
  - Admin toggles at `PUT /api/admin/feature-flags/{flag}`

Storage model, per flag:
  * Default lives in a `settings.*_rollout_pct` env var (0).
  * Runtime override lives in `platform_settings`, value = stringified 0-100.
  * DB row wins when present so admins can flip 10 -> 50 -> 100 without
    redeploying. The env default is only the fresh-deploy floor.

Per-user opt-in is deterministic — seed -> first 8 hex of SHA-256 -> int mod
100. Same user always gets the same answer at a given rollout %, so users don't
flip in/out of the cohort across requests. Anonymous / pre-signup callers (the
`/signup` page before account creation) hash the caller IP if present, else fall
back to "off" so we never expose a new flow to an unattributed visitor.

WRITTEN for one flag and hard-coded to it throughout. Generalised when the
second one arrived: a registry, one set of generic functions, and the
`onboarding_v2` names kept as wrappers because `onboarding_events.py` and
`tests/test_onboarding_v2_feature_flag.py` both import them by name.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models.platform import PlatformSetting

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FlagSpec:
    """One rollout-gated flag."""

    # The wire name. This is the key the frontend reads out of
    # `GET /api/system/feature-flags`, so it is part of a published contract —
    # renaming one silently turns that flag off for every deployed client.
    name: str
    # `platform_settings.key` holding the runtime override.
    setting_key: str
    # Attribute on `settings` holding the fresh-deploy default percentage.
    env_attr: str
    # Bucketing salt. Two flags sharing a seed put the SAME users in both
    # cohorts, so a 10% ramp of one is measured entirely against the 10% who
    # also got the other. Salting per flag decorrelates them.
    #
    # `onboarding_v2` is pinned to "" — its rollout is already in flight, and
    # changing the hash input reshuffles who is in the cohort, which would move
    # live users between two onboarding flows mid-session. New flags get the
    # decorrelated behaviour; this one keeps the buckets it was launched with.
    salt: str = ""


FLAGS: dict[str, FlagSpec] = {
    "onboarding_v2": FlagSpec(
        name="onboarding_v2",
        setting_key="onboarding.v2_rollout_pct",
        env_attr="onboarding_v2_rollout_pct",
        salt="",
    ),
    "web_mobile_shell": FlagSpec(
        name="web_mobile_shell",
        setting_key="web.mobile_shell_rollout_pct",
        env_attr="web_mobile_shell_rollout_pct",
        salt="web_mobile_shell",
    ),
    "automations": FlagSpec(
        name="automations",
        setting_key="automations.rollout_pct",
        env_attr="automations_rollout_pct",
        salt="automations",
    ),
}

# Back-compat: the key literal was exported and may be referenced elsewhere.
ONBOARDING_V2_KEY = FLAGS["onboarding_v2"].setting_key


def _clamp_pct(value: int) -> int:
    if value < 0:
        return 0
    if value > 100:
        return 100
    return value


def _spec(flag: str) -> FlagSpec:
    try:
        return FLAGS[flag]
    except KeyError:
        raise KeyError(f"unknown feature flag {flag!r}") from None


async def get_rollout_pct(db: AsyncSession, flag: str) -> int:
    """Current rollout percentage. DB row wins; env var is the boot-time floor."""
    spec = _spec(flag)
    row = (await db.execute(
        select(PlatformSetting).where(PlatformSetting.key == spec.setting_key)
    )).scalar_one_or_none()
    if row is not None:
        try:
            return _clamp_pct(int((row.value or "").strip()))
        except (ValueError, TypeError):
            logger.warning(
                "feature_flags: %s has non-integer value %r, falling back to env",
                spec.setting_key, row.value,
            )
    return _clamp_pct(int(getattr(settings, spec.env_attr, 0) or 0))


async def set_rollout_pct(db: AsyncSession, flag: str, pct: int) -> int:
    spec = _spec(flag)
    pct = _clamp_pct(pct)
    row = (await db.execute(
        select(PlatformSetting).where(PlatformSetting.key == spec.setting_key)
    )).scalar_one_or_none()
    if row is None:
        db.add(PlatformSetting(key=spec.setting_key, value=str(pct)))
    else:
        row.value = str(pct)
    await db.commit()
    return pct


# ── Per-user allowlist override (R28-D) ──────────────────────────────
# The pct rollout can only move whole hash buckets, which makes "turn
# this on for the dev/test tenant, nobody else" impossible — the R28-D
# simulator round hit exactly that: the app points at production, the
# flag is dark, and the only lever would have enrolled a percentage of
# the real fleet. The allowlist is the missing lever: a short
# comma-separated list of user ids in `platform_settings` under
# `<setting_key>.allow`, checked BEFORE bucketing. An allowlisted user
# reads the flag as ON at any pct, including 0; everyone else is
# untouched. Admin surface: PUT /api/admin/feature-flags/flag/{flag}/allow.

_ALLOWLIST_MAX = 50


def _allow_key(spec: FlagSpec) -> str:
    return spec.setting_key + ".allow"


async def get_allowlist(db: AsyncSession, flag: str) -> list[str]:
    spec = _spec(flag)
    row = (await db.execute(
        select(PlatformSetting).where(PlatformSetting.key == _allow_key(spec))
    )).scalar_one_or_none()
    if row is None or not (row.value or "").strip():
        return []
    return [t.strip() for t in row.value.split(",") if t.strip()]


async def set_allowlist(
    db: AsyncSession, flag: str, user_ids: list[str],
) -> list[str]:
    """Replace the allowlist (empty list clears it). Normalizes,
    dedupes preserving order, caps at _ALLOWLIST_MAX — this is a dev/
    canary lever, not a rollout mechanism; ramps use the percentage."""
    spec = _spec(flag)
    ids: list[str] = []
    for u in user_ids:
        u = (u or "").strip()
        if u and len(u) <= 64 and u not in ids:
            ids.append(u)
    ids = ids[:_ALLOWLIST_MAX]
    value = ",".join(ids)
    row = (await db.execute(
        select(PlatformSetting).where(PlatformSetting.key == _allow_key(spec))
    )).scalar_one_or_none()
    if row is None:
        db.add(PlatformSetting(key=_allow_key(spec), value=value))
    else:
        row.value = value
    await db.commit()
    return ids


def _hash_to_bucket(seed: str) -> int:
    # First 8 hex chars of SHA-256 → 32-bit int → mod 100. Deterministic
    # per seed, uniformly distributed enough for rollout cohorts of any
    # size we'll see in production.
    h = hashlib.sha256((seed or "").encode("utf-8")).hexdigest()
    return int(h[:8], 16) % 100


def is_enabled_for(seed: Optional[str], rollout_pct: int, salt: str = "") -> bool:
    if rollout_pct <= 0:
        return False
    if rollout_pct >= 100:
        # Full launch: everyone, including anonymous visitors mid-signup.
        return True
    # Partial rollout: anonymous visitors (no user_id, no IP) stay on the
    # stable path. Bucketing a missing seed would flip them between cohorts
    # on every refresh, which is worse than a consistent path through the
    # session.
    if not seed:
        return False
    return _hash_to_bucket(f"{salt}:{seed}" if salt else seed) < rollout_pct


async def is_enabled(
    db: AsyncSession, flag: str, seed: Optional[str] = None,
) -> bool:
    spec = _spec(flag)
    # Allowlist first: an allowlisted user is ON at any pct, incl. 0.
    # Only user-id seeds can match — anonymous/IP seeds never do,
    # because the list holds user ids.
    if seed and seed in await get_allowlist(db, flag):
        return True
    pct = await get_rollout_pct(db, flag)
    return is_enabled_for(seed, pct, spec.salt)


async def all_flags_for(
    db: AsyncSession, seed: Optional[str] = None,
) -> dict[str, bool]:
    """Every registered flag, resolved for one caller.

    One query per flag, which is two. Kept simple deliberately: the readout is
    public and uncached, and a client that receives a PARTIAL map cannot tell a
    missing flag from a disabled one — `mobileShell.ts` reads an absent key as
    off. Failing a whole request is recoverable; silently shipping "off" for a
    flag that is at 100% is not.
    """
    return {name: await is_enabled(db, name, seed) for name in FLAGS}


# ── onboarding_v2 wrappers ───────────────────────────────────────────
# Kept because `onboarding_events.py` and the test suite import them by name.

async def get_onboarding_v2_rollout_pct(db: AsyncSession) -> int:
    return await get_rollout_pct(db, "onboarding_v2")


async def set_onboarding_v2_rollout_pct(db: AsyncSession, pct: int) -> int:
    return await set_rollout_pct(db, "onboarding_v2", pct)


def is_onboarding_v2_enabled_for(seed: Optional[str], rollout_pct: int) -> bool:
    return is_enabled_for(seed, rollout_pct, FLAGS["onboarding_v2"].salt)


async def is_onboarding_v2_enabled(
    db: AsyncSession, seed: Optional[str] = None,
) -> bool:
    return await is_enabled(db, "onboarding_v2", seed)
