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

import asyncio
import hashlib
import logging
import secrets
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import AgentConfig
from app.services.background_tasks import spawn as _spawn_bg

logger = logging.getLogger(__name__)


@dataclass
class ActivationResult:
    activated: bool          # True if we flipped bundle_status this call
    already_active: bool     # True if user was already active going in
    bundle_status: str       # post-call bundle_status value
    env_pushed: bool         # True if update_container_env succeeded
    env_error: Optional[str] = None


def _schedule_openai_project_provision(user_id: str) -> None:
    """Fire-and-forget: create the user's per-tenant OpenAI project off the
    request critical path. Keeps signup/activation from blocking on the OpenAI
    Admin API (the per-tenant key is non-load-bearing — the proxy falls back to
    the platform master key until it lands)."""
    import asyncio
    try:
        _spawn_bg(_run_openai_project_provision(user_id))
    except RuntimeError:
        # No running event loop (e.g. a sync management context) — skip; the
        # backfill reconciler will provision the project on its next pass.
        logger.warning(
            "free_tier_activation: no event loop to schedule OpenAI provisioning "
            "user=%s — backfill reconciler will cover it", str(user_id)[:8],
        )


async def _run_openai_project_provision(user_id: str) -> None:
    """Background entry point for `_schedule_openai_project_provision`. Opens its
    OWN session (the request's is closed once the endpoint returns) and runs the
    blocking Admin-API call in a thread so it never stalls the event loop.
    Best-effort — the master-key fallback + backfill reconciler cover failures."""
    import asyncio
    from app.db.database import async_session_maker
    try:
        async with async_session_maker() as db:
            cfg = (await db.execute(
                select(AgentConfig).where(AgentConfig.user_id == user_id)
            )).scalar_one_or_none()
            if cfg is None or (cfg.bundle_openai_project_id and cfg.bundle_openai_api_key):
                return
            # Prefer the agent persona name for the OpenAI dashboard label
            # (toup-<AGENT>-<prefix>), falling back to the user's display name.
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
            from app.api.billing import _provision_openai_project_if_needed
            await asyncio.to_thread(
                _provision_openai_project_if_needed, cfg, user_name=project_label
            )
            await db.commit()
    except Exception as e:
        logger.warning(
            "free_tier_activation: background OpenAI project provisioning failed "
            "user=%s err=%s — proxy uses master key; reconciler will retry",
            str(user_id)[:8], e,
        )


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

    await db.commit()
    await db.refresh(cfg)

    # Per-user OpenAI project auto-provisioning, scheduled OFF the request path.
    # _provision_openai_project_if_needed makes SYNCHRONOUS, blocking calls to
    # OpenAI's Admin API (create project + service-account key) — several
    # seconds that used to land squarely in the signup critical path, so a new
    # user watched a spinner for it. The per-tenant project is NOT load-bearing:
    # until it lands the proxy uses the platform master key (documented fallback
    # in billing._provision_openai_project_if_needed) and the backfill
    # reconciler cures any config that never got one. The load-bearing
    # connect_token / bundle_status above is already committed synchronously, so
    # the user's first chat still authenticates immediately. Idempotent +
    # guarded so we never spawn the work when the project already exists.
    if not (cfg.bundle_openai_project_id and cfg.bundle_openai_api_key):
        _schedule_openai_project_provision(str(user_id))

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
