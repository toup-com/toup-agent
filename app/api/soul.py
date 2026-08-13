"""
Soul Config API — structured agent personality configuration.

GET  /api/soul  → current soul config (or defaults)
PUT  /api/soul  → save/update soul config → compiles → updates Identity record
PUT  /api/soul/sync  → agent-side endpoint to receive soul sync from platform
"""

import asyncio
import logging
import time as _time
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
from app.config import settings

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
    # Read the id BEFORE the try. `current_user` is a live ORM instance, and
    # after `db.rollback()` its attributes are expired — touching one issues
    # lazy IO from a sync context and raises MissingGreenlet, so the handler
    # written to REPORT the failure raised its own exception on top of it.
    # That is what turned this into a bare "Internal Server Error" with the
    # real cause (an aware datetime into a naive column) buried.
    user_id = current_user.id
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
        logger.exception("[SOUL] save failed for user %s", user_id)
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
        # Complete onboarding on first soul save (Soul page IS the
        # onboarding) — this matches the LEGACY flow where Soul was
        # the very last step the user did, after Install. The Phase-2
        # unified flow puts Soul FIRST, so it opts out of this flip
        # via `defer_onboarding_complete=true`; in that flow, the
        # terminal flip is owned by `/agent-setup/register` (fires
        # when the agent container self-registers post-Install).
        # Without the opt-out, saving Soul on /onboarding/soul would
        # mark the whole onboarding done after step 2 of 5, letting
        # the user skip Channel/LLM/Install entirely.
        if not agent_cfg.onboarding_completed and not req.defer_onboarding_complete:
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

    # ── Fail-closed write-through for BOUND tenants (Last Mile L-1) ──
    # The tenant copy of agent_configs.agent_name is what text chat and
    # the voice assembler actually read; a rename that reaches only the
    # platform copy is silent drift — the defect that blanked agent names
    # across the tenant fleet. For a bound tenant the tenant write
    # happens BEFORE the platform commit and its failure fails the save:
    # nothing is saved anywhere, and the user retries. (Bounded
    # transaction hold: ≤2 HTTP attempts on a rare, user-initiated path —
    # the rename contract outweighs the low-frequency connection hold.)
    if agent_cfg and agent_cfg.agent_url and agent_cfg.agent_api_key:
        _sync_kwargs = dict(
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
            # Correct the owner identity in the same sync that sets the
            # agent name — fixes "Hey Agent Owner" on the pool-claim path
            # even when the bridge doesn't forward user_name to /admin/bind.
            owner_name=getattr(current_user, "name", None),
            owner_email=getattr(current_user, "email", None),
        )
        # Retry to a WALL-CLOCK deadline rather than a fixed attempt
        # count. `_sync_soul_to_vps` carries a 15s timeout but returns
        # False in milliseconds when the connection is refused — which is
        # exactly what a blue-green swap looks like from here. A fixed
        # "one retry, sleep 2s" therefore gave up ~2s into an outage it
        # was written to ride out, while still being able to run for 30s
        # against a merely slow agent. A deadline inverts both: many
        # cheap attempts across a refusal window, at most one retry
        # against a slow one.
        _deadline_s, _delay = 12.0, 1.0
        _t0 = _time.monotonic()
        vps_synced = await _sync_soul_to_vps(**_sync_kwargs)
        while not vps_synced and (_time.monotonic() - _t0) < _deadline_s:
            await asyncio.sleep(_delay)
            _delay = min(_delay * 2, 4.0)
            vps_synced = await _sync_soul_to_vps(**_sync_kwargs)
        if not vps_synced:
            await db.rollback()
            raise HTTPException(
                502,
                "Your agent couldn't be updated, so nothing was saved. "
                "Please try again in a moment.",
            )
        # NAIVE UTC. `soul_configs.vps_soul_synced_at` is `DateTime` without
        # `timezone=True`, i.e. TIMESTAMP WITHOUT TIME ZONE, and asyncpg
        # refuses an aware value for one ("can't subtract offset-naive and
        # offset-aware datetimes"). This was the ONLY aware write in the file
        # — every other timestamp here is `datetime.utcnow()` — and because
        # the commit below is the single commit on this path, it did not just
        # lose a timestamp: it failed the INSERT of the whole soul_configs
        # row, so "Save & Apply" 500'd and saved nothing.
        config.vps_soul_synced_at = datetime.utcnow()

    await db.commit()

    # ── Pre-warm managed container (Phase 1, gated by feature flag) ──
    # Fire AFTER the AgentConfig commit so prewarm_service.schedule_prewarm
    # — which opens its own session — sees the just-saved
    # hosting_mode/llm_mode/agent_color values. Errors are swallowed; the
    # Install-step provision call remains the contract-bearing path.
    # Phase 0 doc: docs/onboarding/prewarm-phase0.md.
    if settings.prewarm_on_soul_save and agent_cfg and agent_cfg.hosting_mode == "managed":
        try:
            from app.services.prewarm_service import schedule_prewarm
            await schedule_prewarm(current_user.id)
        except Exception as e:
            logger.warning(
                "[SOUL] prewarm schedule failed for user %s: %s",
                str(current_user.id)[:8], e,
            )

    return response_payload


async def _sync_soul_to_vps(
    agent_url: str, agent_api_key: str,
    user_id: str, name: str, compiled_text: str,
    deactivate_agent_soul_memories: bool = False,
    agent_config_updates: Optional[dict] = None,
    owner_name: Optional[str] = None,
    owner_email: Optional[str] = None,
) -> bool:
    """Sync compiled soul + related data to the user's VPS agent.

    Calls PUT /api/soul/sync on the VPS agent which handles:
    - Upsert Identity(type='soul') record
    - Optionally deactivate old agent_soul memories
    - Optionally update AgentConfig fields (name, color, onboarding)
    - Upsert the owner User row's real name/email (so the agent stops
      greeting "Hey Agent Owner" — see the receiver's section 4)

    owner_name/owner_email are threaded so the user-initiated soul save
    (which the receiver proved reaches the agent — the agent name updates)
    ALSO corrects the owner identity. Without this the pool-claimed
    container kept its stub User(name="Agent Owner") even after the user
    set their soul, because the recreate/bind env body omits user_name.
    Robust even if the bridge fails to forward user_name into /admin/bind.

    Returns True on success, False on failure. Never raises.
    """
    # TKT-LAT-007 (wave 3): shared agent_http client.
    from app.services.agent_http import get_agent_http_client
    url = f"{agent_url}/api/soul/sync"
    try:
        client = get_agent_http_client()
        resp = await client.put(
            url,
            json={
                "user_id": user_id,
                "name": name,
                "compiled_text": compiled_text,
                "deactivate_agent_soul_memories": deactivate_agent_soul_memories,
                "agent_config_updates": agent_config_updates,
                "owner_name": owner_name,
                "owner_email": owner_email,
            },
            headers={"X-Agent-Key": agent_api_key},
            timeout=15.0,
        )
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
    # name/compiled_text are OPTIONAL so the platform can push an owner-only
    # identity sync (owner_name/owner_email) for a freshly provisioned
    # container that has no SoulConfig yet, WITHOUT carrying — and thereby
    # wiping — the agent's existing/default soul. When both are None the soul
    # upsert in the handler is skipped.
    name: Optional[str] = None
    compiled_text: Optional[str] = None
    deactivate_agent_soul_memories: bool = False
    agent_config_updates: Optional[dict] = None
    # Owner identity — the human's real (Gmail) name/email. Threaded so a
    # (re)provisioned tenant container, which boots with a stub
    # User(name="Agent Owner"), gets the real owner name and stops greeting
    # "Hey Agent Owner". The recreate env body omits user_name, so this
    # soul-sync channel is the durable correction after any recreate.
    owner_name: Optional[str] = None
    owner_email: Optional[str] = None


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
    # Auth: only accept from platform via agent key
    agent_key = request.headers.get("X-Agent-Key", "")
    if not settings.agent_api_key or agent_key != settings.agent_api_key:
        raise HTTPException(401, "Invalid agent key")

    # 1. Upsert Identity(type='soul') in the agent's local DB — ONLY when soul
    #    content is actually provided. An owner-only sync (a freshly
    #    provisioned container that has no SoulConfig yet) sends
    #    name/compiled_text=None and MUST NOT wipe the agent's existing/default
    #    soul; it only carries owner_name/owner_email for the User upsert below.
    if req.compiled_text is not None and req.name is not None:
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
            # Report, don't swallow: the sender's fail-closed rename
            # contract (Last Mile L-1) has to SEE this failure — a
            # warn-and-200 receiver turns a failed tenant write into
            # silent drift. The savepoint has already rolled back;
            # raising skips the endpoint's commit, so the identity and
            # memory sections roll back too and the sync stays atomic
            # on the receiver.
            logger.error(f"[SOUL] agent_config update failed on VPS: {e}")
            raise HTTPException(502, "agent_config update failed on tenant")

    # 4. Upsert the owner User row's real name/email. A (re)provisioned
    #    container boots with a stub User(name="Agent Owner") and the
    #    recreate env body never carries user_name, so without this the
    #    agent keeps greeting "Hey Agent Owner" after any recreate (e.g. the
    #    verify-and-heal path). Savepoint-guarded so a failure can't poison
    #    the identity/memory writes above.
    if req.owner_name or req.owner_email:
        try:
            async with db.begin_nested():
                u = (await db.execute(
                    select(User).where(User.id == req.user_id)
                )).scalar_one_or_none()
                _name = (req.owner_name or "").strip()
                _email = (req.owner_email or "").strip()
                if u:
                    changed = False
                    # Only overwrite with a real value; never clobber a real
                    # name back to the stub.
                    if _name and _name != "Agent Owner" and u.name != _name:
                        u.name = _name
                        changed = True
                    if _email and not _email.endswith("@agent.local") and u.email != _email:
                        u.email = _email
                        changed = True
                    if changed:
                        logger.info(f"[SOUL] Owner User row updated on VPS for {req.user_id[:8]}")
                elif _name:
                    db.add(User(
                        id=req.user_id,
                        email=(_email or f"{req.user_id[:8]}@agent.local"),
                        hashed_password="",
                        name=_name,
                    ))
                    logger.info(f"[SOUL] Owner User row created on VPS for {req.user_id[:8]}")
        except Exception as e:
            logger.warning(f"[SOUL] Failed to upsert owner User row on VPS: {e}")

    await db.commit()
    logger.info(
        f"[SOUL] Synced for user {req.user_id}: "
        f"{req.name if req.name is not None else '(owner-only)'}"
    )
    return {"status": "ok", "memories_deactivated": memories_deactivated}
