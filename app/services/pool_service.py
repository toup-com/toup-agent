"""
Platform-side pool service (Phase A.2 — never-sleep plan).

Calls the bridge's `POST /v1/pool/claim` to acquire a pre-booted
generic agent container for a freshly-registered user. Falls back to
the slow `provision_container` path if the pool is exhausted or the
feature flag is off.

Wires into `app.api.auth.register` — the user signs up, this service
fires fire-and-forget, and by the time the user reaches Welcome the
container is bound + ready.

The key difference from `schedule_prewarm`:

- `schedule_prewarm` calls `provision_container(recreate=False)`,
  which spawns a fresh container from `toup-agent:<image>`.
  Cold-boot takes ~15s end-to-end.
- `claim_for_user` asks the bridge to bind an already-running pool
  container. Wall-clock <1s if a generic is available; ~15s fallback
  if not.

After claim succeeds, the resulting `ManagedContainer` row + AgentConfig
are populated identically to the provision path so existing code
(rollouts, reaper, telemetry) sees no shape difference.
"""
from __future__ import annotations

import asyncio
import logging
import secrets
from datetime import datetime
from typing import Optional

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import AgentConfig, ManagedContainer

logger = logging.getLogger(__name__)


async def _build_bind_payload(
    db: AsyncSession,
    user_id: str,
    agent_config: AgentConfig,
) -> dict:
    """Compose the body for `POST /v1/pool/claim`. Mirrors the
    bridge's /admin/bind contract — see backend/app/api/admin_pool.py
    `_BIND_FIELDS`."""
    prefix = user_id[:8]
    # Generate the per-tenant agent_api_key here. The bridge would
    # do this in the slow path; for the pool path the platform owns
    # generation so the key lands in our DB BEFORE the bind succeeds.
    # If claim fails after this, the key is harmless (bridge never
    # gets it).
    agent_api_key = agent_config.agent_api_key or secrets.token_urlsafe(48)

    payload = {
        "user_id": str(user_id),
        "agent_api_key": agent_api_key,
        "prefix": prefix,
        # Image tag — needed by the bridge's refill spawn after this
        # claim. Bridge stores it in BRIDGE_AGENT_IMAGE env normally;
        # passing it explicitly here keeps platform → bridge in sync
        # if the bridge env drifts.
        "_image_tag": settings.docker_agent_image,
    }

    # Channel + identity fields — same set as
    # `_agent_config_to_bridge_body` in docker_host_service. Pulled
    # one-by-one (not a kwarg loop) so adding a new field is an
    # explicit edit here AND in admin_pool._BIND_FIELDS.
    for field in (
        "agent_color",
        "agent_name",
        "llm_mode",
        "openai_api_key",
        "anthropic_api_key",
        "google_api_key",
        "mistral_api_key",
        "groq_api_key",
        "xai_api_key",
        "deepseek_api_key",
        "agent_model",
        "telegram_bot_token",
        "discord_bot_token",
        "slack_bot_token",
        "slack_app_token",
        "whatsapp_phone_number_id",
        "whatsapp_access_token",
        "whatsapp_verify_token",
        "whatsapp_app_secret",
        "whatsapp_mode",
        "connect_token",
        "supabase_url",
        "supabase_anon_key",
    ):
        v = getattr(agent_config, field, None)
        if v:
            payload[field] = v

    return payload


async def claim_for_user(db: AsyncSession, user_id: str) -> Optional[ManagedContainer]:
    """Claim a pool container for `user_id`. Returns the populated
    ManagedContainer row on success, None on pool-exhausted-or-disabled.

    Failure modes:
    - Feature flag off (`settings.use_container_pool=False`): None.
    - Pool exhausted (bridge returns 503): None — caller falls back to
      `provision_container`.
    - Bridge unreachable: None + log; caller falls back.
    - Bridge OK but DB write fails: rolls back the bind via
      `bridge.POST /v1/pool/release` (TODO; for now logged as
      ASSIGNED-without-DB-row drift, reconciler picks up).

    Caller is responsible for the fallback decision. We never fall
    back ourselves — the platform's signup path may not always want
    the slow path on pool failure (e.g., during a planned maintenance
    window when the pool is intentionally drained).
    """
    if not getattr(settings, "use_container_pool", False):
        logger.info("[pool_service] use_container_pool=False — skipping claim")
        return None

    # Existing container check — same idempotency the slow path has.
    existing = (await db.execute(
        select(ManagedContainer).where(ManagedContainer.user_id == user_id)
    )).scalar_one_or_none()
    if existing and existing.status in ("running", "provisioning"):
        return existing

    # AgentConfig must already exist (created by /api/agent-setup or
    # in auth.register's prewarm block). If it doesn't, bail — pool
    # claim assumes the row is there for the bind payload.
    agent_config = (await db.execute(
        select(AgentConfig).where(AgentConfig.user_id == user_id)
    )).scalar_one_or_none()
    if agent_config is None:
        logger.warning("[pool_service] No AgentConfig for user %s — skipping claim", user_id[:8])
        return None

    payload = await _build_bind_payload(db, user_id, agent_config)

    # Call bridge FIRST — only persist the ManagedContainer row once
    # we have a real host_port (the column is NOT NULL on the schema).
    # Pre-Phase-A code did a placeholder insert+commit before bridge,
    # which 500'd the whole register endpoint via NotNullViolation
    # AND poisoned the session so the User row's response build also
    # blew up. Lesson: never insert a row with required fields you
    # don't yet have.
    from app.services.docker_host_service import _bridge_client
    try:
        async with _bridge_client() as client:
            resp = await client.post("/v1/pool/claim", json=payload)
            if resp.status_code == 503:
                logger.info(
                    "[pool_service] Pool exhausted — fallback to provision for user %s",
                    user_id[:8],
                )
                return None
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as e:
        logger.warning(
            "[pool_service] bridge claim returned %s: %s",
            e.response.status_code, e.response.text[:200],
        )
        return None
    except httpx.HTTPError as e:
        logger.warning("[pool_service] bridge unreachable: %s", e)
        return None

    # Bridge response shape: {ok, container_name, host_port, db_pool_slot}
    container_name = data.get("container_name")
    host_port = data.get("host_port")
    if not container_name or not host_port:
        logger.error("[pool_service] bridge claim returned bad shape: %s", data)
        return None

    # Now we have everything required by the schema — upsert the row.
    if existing:
        container = existing
    else:
        import uuid as _uuid
        container = ManagedContainer(
            id=str(_uuid.uuid4()),
            user_id=user_id,
            container_name=container_name,
            host_port=int(host_port),
        )
        db.add(container)
    container.container_name = container_name
    container.host_port = int(host_port)
    container.image_tag = settings.docker_agent_image
    container.status = "running"
    container.started_at = datetime.utcnow()
    container.error_message = None
    container.db_name = data.get("db_pool_slot") or container.db_name

    # AgentConfig: agent_url is the same shape as the slow path.
    prefix = user_id[:8]
    agent_url = f"https://agent-{prefix}.agents.toup.ai"
    agent_config.agent_url = agent_url
    agent_config.agent_api_key = payload["agent_api_key"]
    agent_config.hosting_mode = "managed"
    agent_config.deploy_status = "active"

    try:
        await db.commit()
    except Exception:
        await db.rollback()
        logger.exception("[pool_service] DB commit after bind failed for %s", user_id[:8])
        return None

    logger.info(
        "[pool_service] Claimed %s for user %s (agent_url=%s)",
        container_name, user_id[:8], agent_url,
    )
    return container


async def claim_or_prewarm(db: AsyncSession, user_id: str) -> bool:
    """Try the pool first; fall back to schedule_prewarm.

    Used by `auth.register` so the existing prewarm-on-register
    behavior is preserved when the pool isn't ready or is exhausted.
    Returns True on any successful claim/prewarm-schedule, False if
    everything failed (caller logs and continues — registration
    itself doesn't depend on this).
    """
    try:
        c = await claim_for_user(db, user_id)
        if c is not None:
            return True
    except Exception:
        logger.exception("[pool_service] claim_for_user raised; falling through to prewarm")
        # Critical: roll back the session so the rest of auth.register
        # (the User-row response build) doesn't fail on a poisoned
        # session. SQLAlchemy raises InvalidRequestError or "current
        # transaction is aborted" on any further use until rollback.
        try:
            await db.rollback()
        except Exception:
            pass

    try:
        from app.services.prewarm_service import schedule_prewarm
        await schedule_prewarm(user_id)
        return True
    except Exception as e:
        logger.warning(
            "[pool_service] schedule_prewarm fallback failed for %s: %s",
            user_id[:8], e,
        )
        return False
