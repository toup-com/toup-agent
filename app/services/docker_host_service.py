"""
Managed Docker Host Service — Phase 3 rewrite.

Tenant lifecycle operations route through the typed mTLS provisioning bridge
at settings.bridge_url (e.g. https://bridge.agents.toup.ai). Platform never
SSHes into the Docker host.

Public API is stable across the Phase 2 → Phase 3 transition. Callers
(`agent_setup.py`, `admin/users.py`, `admin/infrastructure.py`, etc.) use
the same function signatures; only the internals changed.

Gone in Phase 3:
  - `_run_ssh` — replaced by `_bridge_client()` (httpx with mTLS)
  - `_get_ssh_key_file` — SSH key material no longer needed
  - `upgrade_container`, `upgrade_all_containers` — rollouts are orchestrated
    by `rollout_service.py` now; this module no longer owns them
  - `_build_env` — bridge's `create_tenant` builds the env internally from
    the agent_config payload we send it (see AgentEnvContract for the shape)

See docs/new-vps/14-AUTOMATED-DEPLOYMENT-DESIGN.md §4 for the platform ↔
bridge protocol rationale.
"""

from __future__ import annotations

import asyncio
import logging
import os
import ssl
import tempfile
from datetime import datetime
from typing import Optional

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import ManagedContainer, AgentConfig

logger = logging.getLogger(__name__)


# ─── Bridge httpx client (mTLS) ───────────────────────────────────

_cert_tmpfiles: dict[str, str] = {}


def _write_cert_tmpfile(name: str, pem: str) -> str:
    """Write a PEM-encoded cert/key to a private tempfile once per process.
    Returns the path. httpx requires file paths for ssl_context loading.

    The files live in /tmp with mode 600 and are never deleted — they die
    with the platform process. Rewriting per-request would create thousands
    of tempfiles under sustained load.
    """
    if name in _cert_tmpfiles:
        return _cert_tmpfiles[name]
    fd, path = tempfile.mkstemp(prefix=f"bridge_{name}_", suffix=".pem")
    with os.fdopen(fd, "w") as f:
        f.write(pem)
    os.chmod(path, 0o600)
    _cert_tmpfiles[name] = path
    return path


def _bridge_client(timeout_s: Optional[int] = None) -> httpx.AsyncClient:
    """Build an httpx.AsyncClient configured for mTLS to the bridge.

    Uses cert material from settings.bridge_{ca,client}_{cert,key}. Those
    are PEM strings stored as Railway env vars (multi-line via Raw Editor).

    Caller is responsible for closing the client (use `async with`).
    """
    if not settings.bridge_url:
        raise RuntimeError(
            "settings.bridge_url is empty — Phase 3 platform requires the bridge to be configured"
        )
    if not (settings.bridge_ca_cert and settings.bridge_client_cert and settings.bridge_client_key):
        raise RuntimeError(
            "bridge mTLS certs missing from settings — set bridge_ca_cert, bridge_client_cert, bridge_client_key"
        )

    cert_path = _write_cert_tmpfile("client_cert", settings.bridge_client_cert)
    key_path = _write_cert_tmpfile("client_key", settings.bridge_client_key)

    # TLS architecture has TWO cert surfaces:
    #   1. server cert — Caddy presents a Let's Encrypt wildcard for
    #      *.agents.toup.ai. We verify it via the SYSTEM CA bundle.
    #   2. client cert auth — Caddy requires a client cert signed by our
    #      bridge CA (bridge_ca_cert). We PRESENT bridge_client_cert/key;
    #      the server side verifies against bridge_ca_cert on the VPS.
    # So we do NOT need bridge_ca_cert in the platform client's SSL context
    # at all — using it as cafile would reject the Let's Encrypt server cert
    # with "unable to get local issuer certificate". The CA only lives on
    # the server side, burned into /etc/caddy/bridge-ca.crt.
    ctx = ssl.create_default_context()
    ctx.load_cert_chain(certfile=cert_path, keyfile=key_path)

    return httpx.AsyncClient(
        base_url=settings.bridge_url.rstrip("/"),
        verify=ctx,
        timeout=httpx.Timeout(timeout_s or settings.bridge_request_timeout_s),
    )


# ─── Lifecycle operations (stable signatures) ────────────────────


def _agent_config_to_bridge_body(agent_config: Optional[AgentConfig]) -> dict:
    """Flatten an AgentConfig row into the shape bridge's CreateTenantReq expects."""
    if not agent_config:
        return {}
    out = {}
    for field in (
        "openai_api_key", "anthropic_api_key", "google_api_key",
        "mistral_api_key", "xai_api_key", "deepseek_api_key",
        "telegram_bot_token", "discord_bot_token",
        "agent_color", "agent_model",
    ):
        val = getattr(agent_config, field, None)
        if val:
            out[field] = val
    # connect_token lives as agent_config.connect_token but the bridge names
    # the env var TOUP_TOKEN; the bridge's CreateTenantReq exposes it as
    # `connect_token`.
    if getattr(agent_config, "connect_token", None):
        out["connect_token"] = agent_config.connect_token
    return out


async def provision_container(
    db: AsyncSession,
    user_id: str,
    agent_config: Optional[AgentConfig] = None,
) -> ManagedContainer:
    """Provision a new Docker container for a user's agent, via the bridge.

    Idempotent: if a running container already exists for this user, returns
    the existing row.
    """
    prefix = user_id[:8]
    result = await db.execute(
        select(ManagedContainer).where(ManagedContainer.user_id == user_id)
    )
    existing = result.scalar_one_or_none()
    if existing and existing.status in ("running", "provisioning"):
        return existing

    # Fetch agent_config if caller didn't pass one
    if not agent_config:
        result = await db.execute(
            select(AgentConfig).where(AgentConfig.user_id == user_id)
        )
        agent_config = result.scalar_one_or_none()

    body = {
        "prefix": prefix,
        "user_id": user_id,
        "image_tag": settings.docker_agent_image,
        "agent_config": _agent_config_to_bridge_body(agent_config),
    }

    if existing:
        existing.status = "provisioning"
        existing.error_message = None
        await db.commit()

    try:
        async with _bridge_client() as client:
            r = await client.post("/v1/tenants", json=body)
            r.raise_for_status()
            data = r.json()
    except httpx.HTTPStatusError as e:
        err = f"bridge create_tenant failed: {e.response.status_code} {e.response.text[:200]}"
        logger.error(err)
        if existing:
            existing.status = "error"
            existing.error_message = err[:500]
            await db.commit()
        raise RuntimeError(err)
    except httpx.HTTPError as e:
        err = f"bridge unreachable: {e}"
        logger.error(err)
        if existing:
            existing.status = "error"
            existing.error_message = err[:500]
            await db.commit()
        raise RuntimeError(err)

    # Upsert ManagedContainer row
    if existing:
        container = existing
    else:
        import uuid
        container = ManagedContainer(
            id=str(uuid.uuid4()),
            user_id=user_id,
            container_name=data["container_name"],
        )
        db.add(container)

    container.container_id = data["container_id"]
    container.container_name = data["container_name"]
    container.host_port = int(data["host_port"])
    container.image_tag = data["image_tag"]
    container.status = "running"
    container.started_at = datetime.utcnow()
    container.error_message = None
    # db_name is deterministic from prefix; bridge's create_tenant_db produces it
    container.db_name = f"toup_agent_{prefix}"

    # Update AgentConfig with the HTTPS agent URL + bridge-issued api_key
    if agent_config:
        agent_config.agent_url = data["agent_url"]
        agent_config.agent_api_key = data["agent_api_key"]
        agent_config.hosting_mode = "managed"
        agent_config.deploy_status = "active"

    await db.commit()
    await db.refresh(container)
    logger.info(
        "provisioned_container user=%s prefix=%s port=%s agent_url=%s",
        user_id[:8], prefix, container.host_port, data["agent_url"],
    )

    # Post-provision soul sync (background, non-blocking)
    asyncio.create_task(
        _sync_soul_after_start(user_id, data["agent_url"], data["agent_api_key"])
    )
    return container


async def stop_container(db: AsyncSession, user_id: str) -> Optional[ManagedContainer]:
    """Stop a tenant container via the bridge.

    Note: bridge doesn't have a /stop endpoint in Phase 3 v1 — stopping is
    equivalent to leaving the container in place but marking our DB row
    as stopped. If future ops need `docker stop`, we add a bridge endpoint
    then. For now, this is a DB-status-only op.
    """
    result = await db.execute(
        select(ManagedContainer).where(ManagedContainer.user_id == user_id)
    )
    container = result.scalar_one_or_none()
    if not container:
        return None
    container.status = "stopped"
    container.stopped_at = datetime.utcnow()
    await db.commit()
    logger.info("marked_stopped user=%s", user_id[:8])
    return container


async def start_container(db: AsyncSession, user_id: str) -> Optional[ManagedContainer]:
    """Start/re-provision a stopped container. Delegates to provision_container."""
    result = await db.execute(
        select(ManagedContainer).where(ManagedContainer.user_id == user_id)
    )
    container = result.scalar_one_or_none()
    if not container:
        return None
    return await provision_container(db, user_id)


async def restart_container(db: AsyncSession, user_id: str) -> Optional[ManagedContainer]:
    """Restart tenant container via bridge."""
    result = await db.execute(
        select(ManagedContainer).where(ManagedContainer.user_id == user_id)
    )
    container = result.scalar_one_or_none()
    if not container:
        return None
    prefix = user_id[:8]
    try:
        async with _bridge_client() as client:
            r = await client.post(f"/v1/tenants/{prefix}/restart")
            r.raise_for_status()
    except httpx.HTTPError as e:
        err = f"bridge restart failed: {e}"
        logger.error(err)
        container.status = "error"
        container.error_message = err[:500]
        await db.commit()
        return container
    container.status = "running"
    container.started_at = datetime.utcnow()
    container.error_message = None
    await db.commit()
    return container


async def destroy_container(db: AsyncSession, user_id: str) -> bool:
    """Remove a tenant container (DB + network + Caddy route) via bridge."""
    result = await db.execute(
        select(ManagedContainer).where(ManagedContainer.user_id == user_id)
    )
    container = result.scalar_one_or_none()
    if not container:
        return False
    prefix = user_id[:8]
    try:
        async with _bridge_client() as client:
            r = await client.delete(f"/v1/tenants/{prefix}")
            # 204 is the expected response; 404 (already gone) is also OK
            if r.status_code not in (204, 404):
                r.raise_for_status()
    except httpx.HTTPError as e:
        logger.error("bridge destroy failed for %s: %s", user_id[:8], e)
        # Continue — mark DB row regardless. Operator can reconcile if bridge
        # state drifts.
    container.status = "deleted"
    container.stopped_at = datetime.utcnow()
    await db.commit()
    return True


async def get_container_status(db: AsyncSession, user_id: str) -> Optional[dict]:
    """Get runtime status via bridge + DB row."""
    result = await db.execute(
        select(ManagedContainer).where(ManagedContainer.user_id == user_id)
    )
    container = result.scalar_one_or_none()
    if not container:
        return None
    prefix = user_id[:8]

    bridge_status: dict = {}
    try:
        async with _bridge_client() as client:
            r = await client.get(f"/v1/tenants/{prefix}/status")
            if r.status_code == 200:
                bridge_status = r.json()
    except httpx.HTTPError as e:
        logger.warning("bridge status unreachable for %s: %s", user_id[:8], e)

    return {
        "id": container.id,
        "status": container.status,
        "docker_status": bridge_status.get("status"),
        "docker_health": bridge_status.get("health"),
        "port": container.host_port,
        "image_tag": container.image_tag,
        "pin_image_tag": container.pin_image_tag,
        "container_name": container.container_name,
        "created_at": container.created_at.isoformat() if container.created_at else None,
        "started_at": container.started_at.isoformat() if container.started_at else None,
        "error": container.error_message,
    }


async def update_container_env(
    db: AsyncSession,
    user_id: str,
    agent_config: AgentConfig,
) -> Optional[ManagedContainer]:
    """Push a new agent_config to the bridge and recreate the container.

    In Phase 2 this SSH'd to rewrite the .env file then `docker rm/run`.
    In Phase 3 we re-provision via bridge (idempotent — bridge detects
    existing container and updates in place).
    """
    result = await db.execute(
        select(ManagedContainer).where(ManagedContainer.user_id == user_id)
    )
    container = result.scalar_one_or_none()
    if not container:
        return None
    # Delegate to provision_container (bridge handles the in-place update)
    # with the fresh agent_config payload.
    return await provision_container(db, user_id, agent_config=agent_config)


async def upgrade_tenant_image(
    db: AsyncSession,
    user_id: str,
    image_tag: str,
    rollout_id: Optional[str] = None,
    timeout_s: Optional[int] = None,
) -> dict:
    """Upgrade one tenant to a specific image tag.

    Called by rollout_service.py. NOT called directly from the legacy
    upgrade_container() path (deleted in Phase 3).

    Returns the bridge's UpgradeResp dict on success. Raises on failure;
    caller (rollout_service) decides whether to retry / rollback.
    """
    prefix = user_id[:8]
    async with _bridge_client(timeout_s or settings.bridge_upgrade_timeout_s) as client:
        r = await client.post(
            f"/v1/tenants/{prefix}/upgrade",
            json={"image_tag": image_tag, "rollout_id": rollout_id},
        )
        if r.status_code == 503:
            # Health-check timeout — bridge returns structured detail
            try:
                detail = r.json().get("detail", {})
            except ValueError:
                detail = {"error": "unhealthy", "raw": r.text[:200]}
            raise BridgeUpgradeUnhealthy(detail)
        r.raise_for_status()
        data = r.json()

    # Platform-side record-keeping: update ManagedContainer.image_tag
    result = await db.execute(
        select(ManagedContainer).where(ManagedContainer.user_id == user_id)
    )
    container = result.scalar_one_or_none()
    if container:
        container.image_tag = image_tag
        await db.commit()
    return data


class BridgeUpgradeUnhealthy(RuntimeError):
    """Raised when bridge /upgrade returns 503 — tenant ended up unhealthy.
    Wraps the structured detail body so the rollout service can decide on
    rollback with prior_tag.
    """

    def __init__(self, detail: dict):
        self.detail = detail
        super().__init__(f"bridge upgrade unhealthy: {detail}")


# ─── Post-provision soul sync (unchanged from Phase 2) ────────────


async def _sync_soul_after_start(
    user_id: str, agent_url: str, agent_api_key: str, db_session_maker=None,
):
    """Wait for container health, then push soul config.

    Runs as a background task after provision or recreate. Unchanged from
    Phase 2 except the agent_url is now the HTTPS subdomain (via Caddy)
    rather than http://ip:port.
    """
    import httpx as _httpx
    from app.db.database import async_session_maker
    from app.db.models import SoulConfig

    for _ in range(10):
        await asyncio.sleep(3)
        try:
            async with _httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{agent_url}/agent/health")
                if resp.status_code == 200:
                    break
        except Exception:
            continue
    else:
        logger.warning("[SOUL] container not healthy after 30s for %s", user_id[:8])
        return

    try:
        async with async_session_maker() as db:
            result = await db.execute(
                select(SoulConfig).where(SoulConfig.user_id == user_id)
            )
            soul = result.scalar_one_or_none()
            if not soul:
                return

        async with _httpx.AsyncClient(timeout=15) as client:
            resp = await client.put(
                f"{agent_url}/api/soul/sync",
                json={
                    "user_id": user_id,
                    "name": soul.name,
                    "compiled_text": soul.compiled_text,
                    "deactivate_agent_soul_memories": False,
                    "agent_config_updates": {
                        "agent_name": soul.name,
                        "agent_color": soul.color,
                    },
                },
                headers={"X-Agent-Key": agent_api_key},
            )
            if resp.status_code == 200:
                logger.info("[SOUL] synced for %s", user_id[:8])
            else:
                logger.warning("[SOUL] sync failed for %s: %s", user_id[:8], resp.status_code)
    except Exception as e:
        logger.warning("[SOUL] sync error for %s: %s", user_id[:8], e)
