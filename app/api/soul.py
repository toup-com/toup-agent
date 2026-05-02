"""
Soul Config API — structured agent personality configuration.

GET  /api/soul  → current soul config (or defaults)
PUT  /api/soul  → save/update soul config → compiles → updates Identity record
PUT  /api/soul/sync  → agent-side endpoint to receive soul sync from platform
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from app.db import get_db
from app.db.models import Identity, User, AgentConfig
from app.db.models.soul_config import SoulConfig
from app.schemas import (
    SoulConfigUpdate, SoulConfigResponse,
    VALID_SOUL_STYLES, VALID_SOUL_TRAITS,
)
from app.services.soul_compiler import compile_soul
from app.api.auth import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/soul", tags=["soul"])

# Default soul config for new users
_DEFAULTS = {
    "name": "Agent",
    "color": "#9B59B6",
    "pronouns": "they",
    "style": "casual",
    "traits": ["uses_humor", "concise", "proactive", "references_past"],
    "custom_instructions": "",
}


@router.get("", response_model=SoulConfigResponse)
async def get_soul(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get current soul config. Returns defaults if none saved."""
    result = await db.execute(
        select(SoulConfig).where(SoulConfig.user_id == current_user.id)
    )
    config = result.scalar_one_or_none()

    if config:
        return SoulConfigResponse(
            name=config.name,
            color=config.color,
            pronouns=config.pronouns,
            style=config.style,
            traits=config.traits or [],
            custom_instructions=config.custom_instructions or "",
            compiled_text=config.compiled_text or "",
            updated_at=config.updated_at,
        )

    # Return defaults
    compiled = compile_soul(_DEFAULTS)
    return SoulConfigResponse(
        **_DEFAULTS,
        compiled_text=compiled,
        updated_at=datetime.utcnow(),
    )


@router.put("", response_model=SoulConfigResponse)
async def save_soul(
    req: SoulConfigUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Save soul config → compile → update Identity record."""
    try:
        return await _save_soul_impl(req, current_user, db)
    except HTTPException:
        raise
    except Exception as e:
        # FastAPI's default 500 returns a bare {"detail":"Internal Server Error"}
        # body, which becomes "HTTP 500" in the frontend's apiFetch fallback.
        # Re-raise as an explicit 500 with the underlying exception type +
        # truncated message so the user (and Railway logs) see what broke.
        await db.rollback()
        logger.exception("[SOUL] save failed for user %s", current_user.id)
        raise HTTPException(500, f"{type(e).__name__}: {str(e)[:300]}")


async def _save_soul_impl(
    req: SoulConfigUpdate,
    current_user: User,
    db: AsyncSession,
) -> "SoulConfigResponse":
    # Validate style
    if req.style not in VALID_SOUL_STYLES:
        raise HTTPException(400, f"Invalid style. Must be one of: {VALID_SOUL_STYLES}")

    # Validate traits
    invalid = set(req.traits) - VALID_SOUL_TRAITS
    if invalid:
        raise HTTPException(400, f"Invalid traits: {invalid}")

    # Validate pronouns
    if req.pronouns not in ("it", "they", "she", "he"):
        raise HTTPException(400, "Pronouns must be one of: it, they, she, he")

    # Compile soul text
    config_dict = req.model_dump()
    compiled_text = compile_soul(config_dict)

    # Upsert SoulConfig
    result = await db.execute(
        select(SoulConfig).where(SoulConfig.user_id == current_user.id)
    )
    config = result.scalar_one_or_none()

    if config:
        config.name = req.name
        config.color = req.color
        config.pronouns = req.pronouns
        config.style = req.style
        config.traits = req.traits
        config.custom_instructions = req.custom_instructions
        config.compiled_text = compiled_text
        config.updated_at = datetime.utcnow()
    else:
        config = SoulConfig(
            user_id=current_user.id,
            name=req.name,
            color=req.color,
            pronouns=req.pronouns,
            style=req.style,
            traits=req.traits,
            custom_instructions=req.custom_instructions,
            compiled_text=compiled_text,
        )
        db.add(config)

    # Upsert Identity(type='soul') — same record prompt_builder reads
    id_result = await db.execute(
        select(Identity).where(
            and_(
                Identity.user_id == current_user.id,
                Identity.identity_type == "soul",
            )
        )
    )
    identity = id_result.scalar_one_or_none()

    if identity:
        identity.content = compiled_text
        identity.name = f"{req.name} Soul"
        identity.updated_at = datetime.utcnow()
    else:
        identity = Identity(
            user_id=current_user.id,
            identity_type="soul",
            name=f"{req.name} Soul",
            content=compiled_text,
            priority=100,
            is_active=True,
        )
        db.add(identity)

    # Sync color + name to AgentConfig (used by floating orb, app proxy, etc.)
    ac_result = await db.execute(
        select(AgentConfig).where(AgentConfig.user_id == current_user.id)
    )
    agent_cfg = ac_result.scalar_one_or_none()
    onboarding_was_done = False
    if agent_cfg:
        onboarding_was_done = agent_cfg.onboarding_completed
        agent_cfg.agent_color = req.color
        agent_cfg.agent_name = req.name
        # Complete onboarding on first soul save (Soul page IS the onboarding)
        if not agent_cfg.onboarding_completed:
            agent_cfg.onboarding_completed = True
            logger.info(f"[SOUL] Onboarding completed via Soul page for user {current_user.id}")

    # Deactivate old onboarding-created agent_soul memories to prevent conflicts
    try:
        from sqlalchemy import update
        from app.db.models.memory import Memory
        deactivated = await db.execute(
            update(Memory).where(
                and_(
                    Memory.user_id == current_user.id,
                    Memory.brain_type == "agent",
                    Memory.category == "agent_soul",
                    Memory.is_active == True,
                )
            ).values(is_active=False, updated_at=datetime.utcnow())
        )
        if deactivated.rowcount:
            logger.info(f"[SOUL] Deactivated {deactivated.rowcount} old agent_soul memories for user {current_user.id}")
    except Exception as e:
        logger.warning(f"[SOUL] Failed to deactivate old agent_soul memories: {e}")

    # Track whether onboarding was just completed in this save
    onboarding_just_completed = (
        agent_cfg
        and agent_cfg.onboarding_completed
        and not onboarding_was_done
    )

    # Snapshot the now-known field values BEFORE commit so we can build
    # the response without re-touching `config` (post-commit refresh has
    # been the suspect for the bare 500s we've been seeing — pgbouncer
    # can drop the named-prepared-statement session between commit and
    # refresh, surfacing as an unhandled DBAPIError outside any except).
    saved_at = datetime.utcnow()
    response_payload = SoulConfigResponse(
        name=req.name,
        color=req.color,
        pronouns=req.pronouns,
        style=req.style,
        traits=list(req.traits or []),
        custom_instructions=req.custom_instructions or "",
        compiled_text=compiled_text,
        updated_at=saved_at,
    )

    await db.commit()

    # ── VPS sync (invisible to user) ──
    if agent_cfg and agent_cfg.agent_url and agent_cfg.agent_api_key:
        try:
            vps_synced = await _sync_soul_to_vps(
                agent_url=agent_cfg.agent_url,
                agent_api_key=agent_cfg.agent_api_key,
                user_id=current_user.id,
                name=req.name,
                compiled_text=compiled_text,
                deactivate_agent_soul_memories=True,
                agent_config_updates={
                    "onboarding_completed": True,
                    "agent_name": req.name,
                    "agent_color": req.color,
                } if onboarding_just_completed else {
                    "agent_name": req.name,
                    "agent_color": req.color,
                },
            )
            if vps_synced:
                # Best-effort timestamp update — failure here doesn't
                # invalidate the save, so swallow any DB error rather
                # than letting it 500 a successful save.
                try:
                    config.vps_soul_synced_at = datetime.now(timezone.utc)
                    await db.commit()
                except Exception as inner:
                    logger.warning(
                        "[SOUL] vps_soul_synced_at update skipped: %s",
                        type(inner).__name__,
                    )
        except Exception as e:
            logger.error(f"[SOUL] VPS sync failed for user {current_user.id}: {e}")

    return response_payload


async def _sync_soul_to_vps(
    agent_url: str, agent_api_key: str,
    user_id: str, name: str, compiled_text: str,
    deactivate_agent_soul_memories: bool = False,
    agent_config_updates: Optional[dict] = None,
) -> bool:
    """Sync compiled soul + related data to the user's VPS agent.

    Calls PUT /api/soul/sync on the VPS agent which handles:
    - Upsert Identity(type='soul') record
    - Optionally deactivate old agent_soul memories
    - Optionally update AgentConfig fields (name, color, onboarding)

    Returns True on success, False on failure. Never raises.
    """
    import httpx
    url = f"{agent_url}/api/soul/sync"
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.put(url, json={
                "user_id": user_id,
                "name": name,
                "compiled_text": compiled_text,
                "deactivate_agent_soul_memories": deactivate_agent_soul_memories,
                "agent_config_updates": agent_config_updates,
            }, headers={"X-Agent-Key": agent_api_key})
            if resp.status_code == 200:
                logger.info(f"[SOUL] Synced soul to VPS at {agent_url}")
                return True
            else:
                logger.error(f"[SOUL] VPS soul sync failed: {resp.status_code} {resp.text}")
                return False
    except Exception as e:
        logger.error(f"[SOUL] VPS soul sync error: {e}")
        return False


class SoulSyncRequest(BaseModel):
    user_id: str
    name: str
    compiled_text: str
    deactivate_agent_soul_memories: bool = False
    agent_config_updates: Optional[dict] = None


@router.put("/sync")
async def sync_soul(
    req: SoulSyncRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Agent-side endpoint: receive soul sync from platform.

    Handles:
    1. Upsert Identity(type='soul') in local DB
    2. Optionally deactivate old agent_soul memories
    3. Optionally update AgentConfig fields (name, color, onboarding)
    """
    from app.config import settings

    # Auth: only accept from platform via agent key
    agent_key = request.headers.get("X-Agent-Key", "")
    if not settings.agent_api_key or agent_key != settings.agent_api_key:
        raise HTTPException(401, "Invalid agent key")

    # 1. Upsert Identity(type='soul') in the agent's local DB
    id_result = await db.execute(
        select(Identity).where(
            and_(
                Identity.user_id == req.user_id,
                Identity.identity_type == "soul",
            )
        )
    )
    identity = id_result.scalar_one_or_none()

    if identity:
        identity.content = req.compiled_text
        identity.name = f"{req.name} Soul"
        identity.updated_at = datetime.utcnow()
    else:
        identity = Identity(
            user_id=req.user_id,
            identity_type="soul",
            name=f"{req.name} Soul",
            content=req.compiled_text,
            priority=100,
            is_active=True,
        )
        db.add(identity)

    # 2. Deactivate old agent_soul memories (prevents conflict with Soul config)
    memories_deactivated = 0
    if req.deactivate_agent_soul_memories:
        try:
            from sqlalchemy import update as sql_update
            from app.db.models.memory import Memory
            async with db.begin_nested():
                result = await db.execute(
                    sql_update(Memory).where(
                        and_(
                            Memory.user_id == req.user_id,
                            Memory.brain_type == "agent",
                            Memory.category == "agent_soul",
                            Memory.is_active == True,
                        )
                    ).values(is_active=False, updated_at=datetime.utcnow())
                )
                memories_deactivated = result.rowcount
            if memories_deactivated:
                logger.info(f"[SOUL] Deactivated {memories_deactivated} agent_soul memories on VPS for user {req.user_id}")
        except Exception as e:
            logger.warning(f"[SOUL] Failed to deactivate agent_soul memories on VPS: {e}")

    # 3. Update AgentConfig fields if provided
    # Wrapped in savepoint because agent DBs may not have agent_configs table
    # (it's a platform-only table). Without the savepoint, a failed query
    # poisons the transaction and the identity+memory changes are lost.
    if req.agent_config_updates:
        try:
            async with db.begin_nested():
                ac_result = await db.execute(
                    select(AgentConfig).where(AgentConfig.user_id == req.user_id)
                )
                ac = ac_result.scalar_one_or_none()
                if ac:
                    for field, value in req.agent_config_updates.items():
                        if hasattr(ac, field):
                            setattr(ac, field, value)
                    logger.info(f"[SOUL] Updated AgentConfig on VPS: {list(req.agent_config_updates.keys())}")
                else:
                    # Older tenant containers were provisioned before the
                    # platform started writing AgentConfig rows into the
                    # tenant DB, so the row simply doesn't exist — and the
                    # sync silently no-ops, leaving WhatsApp self-chat
                    # headers stuck on the "Agent" fallback. Create a row
                    # with the synced fields so future reads return the
                    # user's chosen agent name.
                    valid = {
                        f: v for f, v in req.agent_config_updates.items()
                        if hasattr(AgentConfig, f)
                    }
                    if valid:
                        db.add(AgentConfig(user_id=req.user_id, **valid))
                        logger.info(
                            f"[SOUL] Created AgentConfig on VPS for {req.user_id[:8]} "
                            f"with {list(valid.keys())}"
                        )
        except Exception as e:
            logger.warning(f"[SOUL] Failed to update AgentConfig on VPS (table may not exist): {e}")

    await db.commit()
    logger.info(f"[SOUL] Synced soul identity for user {req.user_id}: {req.name}")
    return {"status": "ok", "memories_deactivated": memories_deactivated}
