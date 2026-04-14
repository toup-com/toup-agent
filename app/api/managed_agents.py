"""
Managed Agent API — provision, start, stop, and monitor Docker containers
for Quick Setup users on the shared Docker host.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

logger = logging.getLogger(__name__)

from app.db.database import get_db
from app.db.models import AgentConfig
from app.api.auth import get_current_user
from app.services import docker_host_service
from app.config import settings

router = APIRouter(prefix="/managed-agent", tags=["managed-agent"])


async def _get_agent_config(user_id: str, db: AsyncSession) -> AgentConfig | None:
    result = await db.execute(
        select(AgentConfig).where(AgentConfig.user_id == user_id)
    )
    return result.scalar_one_or_none()


@router.post("/provision")
async def provision(
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Provision a managed Docker container for the current user's agent."""
    if not settings.managed_hosting_enabled:
        raise HTTPException(503, "Managed hosting is not enabled on this platform")

    agent_config = await _get_agent_config(current_user.id, db)
    if not agent_config:
        raise HTTPException(400, "Agent config not found. Complete setup first.")

    try:
        container = await docker_host_service.provision_container(
            db, current_user.id, agent_config
        )

        # ── Sync Soul identity to newly provisioned agent ──
        # The user may have configured their Soul (name, pronouns, style)
        # during onboarding BEFORE the agent was provisioned. The Soul save
        # skips VPS sync when agent_url is not set. Now that the agent is
        # running, sync the Soul identity so the agent uses the correct name.
        agent_url = f"http://{settings.docker_host_ip}:{container.host_port}"
        try:
            from app.api.soul import _sync_soul_to_vps
            from app.db.models.soul_config import SoulConfig

            soul_cfg = (await db.execute(
                select(SoulConfig).where(SoulConfig.user_id == current_user.id)
            )).scalar_one_or_none()

            if soul_cfg and soul_cfg.compiled_text:
                import asyncio
                async def _deferred_soul_sync():
                    """Wait for agent to finish starting, then sync Soul."""
                    await asyncio.sleep(10)  # Agent needs time to start
                    try:
                        await _sync_soul_to_vps(
                            agent_url=agent_url,
                            agent_api_key=agent_config.agent_api_key,
                            user_id=current_user.id,
                            name=soul_cfg.name,
                            compiled_text=soul_cfg.compiled_text,
                            deactivate_agent_soul_memories=True,
                            agent_config_updates={
                                "agent_name": soul_cfg.name,
                                "agent_color": soul_cfg.color,
                            },
                        )
                        logger.info(f"[PROVISION] Soul synced to agent for user {current_user.id}")
                    except Exception as e:
                        logger.error(f"[PROVISION] Soul sync failed for user {current_user.id}: {e}")
                asyncio.create_task(_deferred_soul_sync())
        except Exception as e:
            logger.warning(f"[PROVISION] Could not queue soul sync: {e}")

        return {
            "status": container.status,
            "port": container.host_port,
            "container_name": container.container_name,
            "agent_url": agent_url,
        }
    except RuntimeError as e:
        raise HTTPException(500, str(e))


@router.get("/status")
async def status(
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get the status of the current user's managed container."""
    result = await docker_host_service.get_container_status(db, current_user.id)
    if not result:
        return {"status": "not_provisioned"}
    return result


@router.post("/start")
async def start(
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Start a stopped managed container."""
    container = await docker_host_service.start_container(db, current_user.id)
    if not container:
        raise HTTPException(404, "No managed container found")
    return {"status": container.status}


@router.post("/stop")
async def stop(
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Stop a running managed container."""
    container = await docker_host_service.stop_container(db, current_user.id)
    if not container:
        raise HTTPException(404, "No managed container found")
    return {"status": container.status}


@router.post("/restart")
async def restart(
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Restart a managed container (e.g. after config change)."""
    container = await docker_host_service.restart_container(db, current_user.id)
    if not container:
        raise HTTPException(404, "No managed container found")
    return {"status": container.status}


@router.post("/destroy")
async def destroy(
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Destroy a managed container."""
    success = await docker_host_service.destroy_container(db, current_user.id)
    if not success:
        raise HTTPException(404, "No managed container found")
    return {"status": "deleted"}
