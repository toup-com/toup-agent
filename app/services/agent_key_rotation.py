"""
T0b — Agent API key rotation.

Atomic rotation primitive for `agent_configs.agent_api_key`:

  1. Generate a new key (32-byte url-safe, matching provision-time format).
  2. Snapshot the old key for rollback.
  3. UPDATE agent_configs SET agent_api_key=<new>, flush (no commit yet).
  4. Push new env to the bridge via `provision_container(recreate=True)`.
     Bridge force-removes the old container, rewrites .env with the new
     key, runs the new container. Active WS sessions die abruptly when
     the old container is killed — that is the desired behavior for a
     security action.
  5. Verify: poll the new container's `/agent/health` with the new key
     until it responds 200, bounded by `agent_key_rotation_verify_*`
     settings.
  6. On any failure: roll back the DB to the old key, attempt a second
     bridge recreate to restore the old container with the old env
     (so the user is not left with a dead tenant), write a
     `agent_key_rotation_failed` audit event, raise.
  7. On success: commit, write `agent_key_rotated` audit event, return.

Why we use `provision_container(recreate=True)` rather than a targeted
env-update endpoint:

  * The bridge has no targeted env-update endpoint for non-pool tenants.
    The pool refresh-config path WOULD update env without recreate, but
    that's wrong here — we *want* old WS sessions to drop, since they
    were authenticated against the old key.
  * Acceptable for v1: rotation is rare (operator-triggered or
    user-initiated security action), and dropping active sessions is
    expected behavior on key rotation. Documented in the rollout doc.

Self-hosted (`hosting_mode != "managed"`) is NOT supported in v1 — the
platform has no way to push env to a self-hosted agent. The endpoint
returns 400 for those users with a clear message.
"""

from __future__ import annotations

import json
import logging
import secrets
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import asyncio
import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import (
    AgentConfig,
    AgentSecurityEvent,
    EVENT_AGENT_KEY_ROTATED,
    EVENT_AGENT_KEY_ROTATION_FAILED,
    ManagedContainer,
)

logger = logging.getLogger(__name__)


# ─── Public types ─────────────────────────────────────────────────────


class AgentKeyRotationError(Exception):
    """Wraps any failure during rotation. The caller maps this to a 502
    HTTP response with the message exposed (no secrets in the message)."""


@dataclass(frozen=True)
class RotationResult:
    new_agent_api_key: str
    rotated_at: datetime
    key_prefix: str  # First 8 chars for display


# ─── Public entry point ───────────────────────────────────────────────


async def rotate_agent_api_key(
    db: AsyncSession,
    user_id: str,
) -> RotationResult:
    """Rotate the user's agent_api_key end-to-end.

    Caller is responsible for:
      - Authenticating the user (FastAPI dependency).
      - Verifying a `ROTATE_AGENT_KEY` sensitive-action token BEFORE
        invoking this function. This service does not re-check.
      - Mapping `AgentKeyRotationError` to an HTTP response.
    """
    # 1. Look up agent_config + container.
    cfg = (
        await db.execute(
            select(AgentConfig).where(AgentConfig.user_id == user_id)
        )
    ).scalar_one_or_none()
    if cfg is None:
        raise AgentKeyRotationError(
            "No agent_config exists for this user. Complete agent setup first."
        )
    if cfg.hosting_mode != "managed":
        raise AgentKeyRotationError(
            f"Rotation is only supported for managed-mode agents (this user is "
            f"hosting_mode={cfg.hosting_mode!r}). Self-hosted agents must "
            f"rotate manually by editing their .env and restarting."
        )

    container = (
        await db.execute(
            select(ManagedContainer).where(ManagedContainer.user_id == user_id)
        )
    ).scalar_one_or_none()
    if container is None or container.status not in ("running", "provisioning"):
        raise AgentKeyRotationError(
            f"No running managed container found (status="
            f"{getattr(container, 'status', None)!r}). Provision the agent first."
        )

    # 2. Snapshot old key for rollback.
    old_key = cfg.agent_api_key
    if not old_key:
        raise AgentKeyRotationError(
            "Existing agent_api_key is empty — cannot rotate (this should "
            "not happen for a provisioned tenant)."
        )

    # 3. Generate new key. Match provision-time format
    #    (`secrets.token_urlsafe(32)` → 43 chars, ≥32 contract minimum).
    new_key = secrets.token_urlsafe(32)
    if new_key == old_key:  # extraordinarily unlikely
        raise AgentKeyRotationError("Generated key collided with old key — retry")

    cfg.agent_api_key = new_key
    await db.flush()
    # IMPORTANT: do NOT commit yet. We want bridge success before
    # making the DB write durable. If the bridge fails, the rollback
    # below resets cfg.agent_api_key back to old_key without ever
    # having committed the new value.

    # 4. Push to bridge. Reuse provision_container(recreate=True) — the
    #    only path that updates env on non-pool tenants today.
    #
    #    WS-DISCONNECT CONTRACT (intentional):
    #    The bridge `docker rm -f`s the old container as part of recreate.
    #    Any active WebSocket connections die with WS code 1006 (abnormal
    #    closure) — we cannot send a graceful close frame because the
    #    process is killed mid-flight. Reconnect attempts then fail at
    #    auth: WS chat uses settings.agent_api_key as JWT secret on the
    #    agent side (ws_chat.py:1009), so old tokens fail to verify
    #    against the new key. Clients see a clean reject, not a hang.
    #
    #    This behavior is the desired security semantic for a rotation,
    #    not a bug. The reconnect-rejection contract is exercised in T5d's
    #    hot-flip smoke suite alongside the connector-dispatch flag
    #    flip — search docs/integrations/02-rollout.md for "Hot-flip
    #    smoke (mandatory, blocks T5e)".
    try:
        from app.services.docker_host_service import provision_container
        await provision_container(db, user_id, agent_config=cfg, recreate=True)
    except Exception as bridge_exc:
        logger.exception("[rotate-key] bridge recreate failed for %s", user_id[:8])
        await _rollback_to_old_key(db, cfg, old_key, bridge_exc)
        raise AgentKeyRotationError(
            f"Bridge recreate failed: {type(bridge_exc).__name__}: {bridge_exc}"
        ) from bridge_exc

    # 5. Verify the new container actually responds to the new key.
    #    The bridge call returned, so the container exists, but the
    #    FastAPI app inside it may still be booting.
    #    cfg.agent_url is set/refreshed by provision_container.
    if not cfg.agent_url:
        await _rollback_to_old_key(db, cfg, old_key, RuntimeError("no agent_url"))
        raise AgentKeyRotationError(
            "Bridge succeeded but no agent_url was returned — cannot verify"
        )

    verified = await _verify_new_key(cfg.agent_url, new_key)
    if not verified:
        logger.error(
            "[rotate-key] verification failed for %s — rolling back",
            user_id[:8],
        )
        await _rollback_to_old_key(
            db, cfg, old_key, RuntimeError("verify_health_timed_out")
        )
        raise AgentKeyRotationError(
            "New container did not respond to the new X-Agent-Key within "
            f"{settings.agent_key_rotation_verify_timeout_s}s — rolled back"
        )

    # 6. Commit + audit.
    rotated_at = datetime.utcnow()
    db.add(AgentSecurityEvent(
        user_id=user_id,
        event_type=EVENT_AGENT_KEY_ROTATED,
        occurred_at=rotated_at,
        metadata_json=json.dumps({
            "old_key_prefix": old_key[:8],
            "new_key_prefix": new_key[:8],
            "agent_url": cfg.agent_url,
        }),
    ))
    await db.commit()

    return RotationResult(
        new_agent_api_key=new_key,
        rotated_at=rotated_at,
        key_prefix=new_key[:8],
    )


# ─── Verify ───────────────────────────────────────────────────────────


async def _verify_new_key(agent_url: str, new_key: str) -> bool:
    """Poll `<agent_url>/agent/health` with the new X-Agent-Key.

    Returns True on the first 200, False on timeout. Network errors
    during boot are expected (container may still be coming up) — they
    are retried, not propagated.

    Every attempt logs at INFO level with the elapsed time, so when
    this rolls out to dogfood we can see the real Contabo
    recreate-and-boot distribution and tune
    `agent_key_rotation_verify_timeout_s` from data, not guesswork.
    """
    deadline_s = settings.agent_key_rotation_verify_timeout_s
    interval_s = settings.agent_key_rotation_verify_retry_interval_s
    started = asyncio.get_event_loop().time()
    deadline = started + deadline_s
    health_url = f"{agent_url.rstrip('/')}/agent/health"

    attempts = 0
    async with httpx.AsyncClient(timeout=httpx.Timeout(3.0)) as client:
        while asyncio.get_event_loop().time() < deadline:
            attempts += 1
            elapsed = asyncio.get_event_loop().time() - started
            try:
                r = await client.get(
                    health_url,
                    headers={"X-Agent-Key": new_key},
                )
                if r.status_code == 200:
                    logger.info(
                        "[rotate-key] verify ok: attempt=%d elapsed=%.2fs",
                        attempts, elapsed,
                    )
                    return True
                logger.info(
                    "[rotate-key] verify retry: attempt=%d elapsed=%.2fs status=%d",
                    attempts, elapsed, r.status_code,
                )
            except (httpx.HTTPError, OSError) as e:
                logger.info(
                    "[rotate-key] verify retry: attempt=%d elapsed=%.2fs error=%s",
                    attempts, elapsed, type(e).__name__,
                )
            await asyncio.sleep(interval_s)
    elapsed_total = asyncio.get_event_loop().time() - started
    logger.warning(
        "[rotate-key] verify FAILED: attempts=%d elapsed=%.2fs deadline=%ds",
        attempts, elapsed_total, deadline_s,
    )
    return False


# ─── Rollback ─────────────────────────────────────────────────────────


async def _rollback_to_old_key(
    db: AsyncSession,
    cfg: AgentConfig,
    old_key: str,
    cause: Exception,
) -> None:
    """Restore cfg.agent_api_key to old_key in the DB AND attempt to
    push the old env back to the bridge so the user is not stranded.

    Best-effort on the bridge side: if even the rollback recreate
    fails, we still write the failure audit and let the caller raise.
    The DB is the source of truth — the audit row plus operator
    observability will catch it.
    """
    cfg.agent_api_key = old_key
    db.add(AgentSecurityEvent(
        user_id=cfg.user_id,
        event_type=EVENT_AGENT_KEY_ROTATION_FAILED,
        occurred_at=datetime.utcnow(),
        metadata_json=json.dumps({
            "old_key_prefix": old_key[:8],
            "cause_class": type(cause).__name__,
            "cause_str": str(cause)[:300],
        }),
    ))
    try:
        await db.commit()
    except Exception:
        logger.exception(
            "[rotate-key] rollback DB commit failed for %s — manual "
            "operator intervention required", cfg.user_id[:8],
        )
        raise

    # Try to restore the bridge to the old env. If this fails, the
    # tenant container may be running with whatever env the failed
    # rotation half-set; operator alert via the audit row.
    try:
        from app.services.docker_host_service import provision_container
        await provision_container(
            db, cfg.user_id, agent_config=cfg, recreate=True,
        )
    except Exception as e:
        logger.exception(
            "[rotate-key] rollback bridge recreate failed for %s: %s "
            "— tenant may need operator intervention",
            cfg.user_id[:8], e,
        )
        # Don't re-raise here. The original AgentKeyRotationError is
        # what the caller maps to HTTP 502. The audit row records the
        # rollback-recreate failure for ops.
