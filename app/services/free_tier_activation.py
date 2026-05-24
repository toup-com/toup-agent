"""
Free-tier activation — the credit-system bridge for users who don't pay
through Stripe.

The legacy paid-bundle flow set `AgentConfig.bundle_status='active'`,
minted `connect_token` and `llm_token_hash`, and pushed the new env to
the running tenant container via the Stripe `invoice.payment_succeeded`
webhook. Free users go through no webhook, so the same state changes
need to happen server-side when they click "Continue on Free" in
onboarding (or click "Wake your agent up" on the Install step).

`activate_free_tier(db, user_id, *, force_env_push=True)` is the
single source of truth. It is idempotent — paid users (already in
`active` / `cancelling`) are left untouched on the bundle_status front,
but the container env push still runs so a stale TOUP_TOKEN in the
agent gets refreshed.

Callers:
  * `POST /api/onboarding/events/activate-free-tier` (frontend
    LlmRoute.onSelectFree + InstallRoute mount-time backstop)
  * `POST /api/managed-agent/provision` (the "Wake your agent up"
    button) — this is the critical-path call. Without it the
    provision endpoint returns idempotently on the existing pool
    container and the user's stale TOUP_TOKEN never gets refreshed.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import AgentConfig

logger = logging.getLogger(__name__)


@dataclass
class ActivationResult:
    activated: bool          # True if we flipped bundle_status this call
    already_active: bool     # True if user was already active going in
    bundle_status: str       # post-call bundle_status value
    env_pushed: bool         # True if update_container_env succeeded
    env_error: Optional[str] = None


async def activate_free_tier(
    db: AsyncSession,
    user_id: str,
    *,
    force_env_push: bool = True,
) -> ActivationResult:
    """Make sure the user can talk to the LLM proxy.

    Steps (all idempotent):
      1. Materialise AgentConfig if missing.
      2. Mint `connect_token` if NULL.
      3. Set `llm_token_hash = sha256(connect_token)` if drifted.
      4. Flip `bundle_status='active'` + `bundle_started_at=now()` +
         `llm_mode='bundle'` if not already in active/cancelling.
      5. If `force_env_push`: push the new env to the running tenant
         container via update_container_env (force recreate). Failures
         are reported in `env_error` but do NOT raise — the caller
         decides whether a stale-env outcome is fatal.

    Step 5 is the load-bearing one for users with a freshly-provisioned
    container that booted with LLM_MODE=manual + empty TOUP_TOKEN. The
    DB flip in step 4 doesn't reach the agent process unless the env
    actually gets pushed.
    """
    cfg = (await db.execute(
        select(AgentConfig).where(AgentConfig.user_id == user_id)
    )).scalar_one_or_none()
    if cfg is None:
        # Defer to the canonical materializer so we don't drift from
        # default field values.
        from app.api.agent_setup import _get_or_create_config
        cfg = await _get_or_create_config(str(user_id), db)

    was_already_active = cfg.bundle_status in ("active", "cancelling")

    if not cfg.connect_token:
        cfg.connect_token = f"toup_ct_{secrets.token_urlsafe(32)}"

    expected_hash = hashlib.sha256(cfg.connect_token.encode()).hexdigest()
    if cfg.llm_token_hash != expected_hash:
        cfg.llm_token_hash = expected_hash
        logger.info(
            "free_tier_activation: refreshed llm_token_hash user=%s",
            str(user_id)[:8],
        )

    if not was_already_active:
        cfg.bundle_status = "active"
        cfg.bundle_started_at = cfg.bundle_started_at or datetime.utcnow()
        cfg.llm_mode = "bundle"
        logger.info(
            "free_tier_activation: bundle_status -> active user=%s",
            str(user_id)[:8],
        )
    else:
        # Paid user — still make sure llm_mode is 'bundle' so the agent
        # routes through the proxy. The bundle activation webhook does
        # this; we mirror it for safety.
        if cfg.llm_mode != "bundle":
            cfg.llm_mode = "bundle"

    # Per-user OpenAI project auto-provisioning. The paid-bundle path
    # has called this since the β architecture shipped (see
    # billing.py::_provision_openai_project_if_needed), but Free signups
    # were skipping it — every fresh Free user's OpenAI calls fell back
    # to the platform master key, with no per-project usage attribution
    # in the OpenAI dashboard (operator screenshot 2026-05-24 showed
    # `toup-tenant-*` projects stopped being created post-credit-system).
    #
    # Pass the AGENT name through so the OpenAI dashboard shows
    # `toup-<AGENT>-<prefix>` (e.g. `toup-Aria-9a7f2fe9`) — that's the
    # persona behind the project's API calls and is the more meaningful
    # identifier when scanning the project list. Falls back to the user's
    # display name if no agent_name is set yet (e.g. activation runs
    # before Soul step), then to the legacy `toup-tenant-<prefix>`.
    #
    # Idempotent — early-returns if cfg.bundle_openai_project_id is
    # already set. Failures swallow & log; the proxy's master-key
    # fallback keeps the agent working even if OpenAI Admin API is
    # unavailable.
    project_label: Optional[str] = None
    try:
        if cfg.agent_name:
            project_label = cfg.agent_name
        else:
            from app.db.models import User
            u = await db.get(User, user_id)
            project_label = (u.name if u else None) or None
    except Exception:
        project_label = None
    try:
        from app.api.billing import _provision_openai_project_if_needed
        _provision_openai_project_if_needed(cfg, user_name=project_label)
    except Exception as e:
        logger.warning(
            "free_tier_activation: OpenAI project auto-provisioning failed "
            "user=%s err=%s — falling back to platform master key",
            str(user_id)[:8], e,
        )

    await db.commit()
    await db.refresh(cfg)

    env_pushed = False
    env_error: Optional[str] = None
    if force_env_push:
        # The container env push is the load-bearing piece. If this
        # fails the agent process keeps stale settings.toup_token /
        # llm_mode and _bundle_active() returns False forever — every
        # LLM call falls back to BYOK and surfaces "Error: Something
        # went wrong" to the user.
        try:
            from app.services.docker_host_service import update_container_env
            container = await update_container_env(db, str(user_id), cfg)
            env_pushed = container is not None
            if not env_pushed:
                env_error = "no_managed_container"
        except Exception as e:
            env_error = f"{type(e).__name__}: {e}"
            logger.warning(
                "free_tier_activation: container env push failed user=%s err=%s",
                str(user_id)[:8], env_error,
            )

    return ActivationResult(
        activated=not was_already_active,
        already_active=was_already_active,
        bundle_status=cfg.bundle_status or "none",
        env_pushed=env_pushed,
        env_error=env_error,
    )
