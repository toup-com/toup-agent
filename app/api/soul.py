"""
Soul Config API — structured agent personality configuration.

GET  /api/soul  → current soul config (or defaults)
PUT  /api/soul  → save/update soul config → compiles → updates Identity record
"""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from app.db import get_db
from app.db.models import Identity, User
from app.db.models.soul_config import SoulConfig
from app.schemas import (
    SoulConfigUpdate, SoulConfigResponse,
    VALID_SOUL_STYLES, VALID_SOUL_TRAITS,
)
from app.services.soul_compiler import compile_soul
from app.api.auth import get_current_user

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

    await db.commit()
    await db.refresh(config)

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
