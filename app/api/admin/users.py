"""
Admin — User & Invite Management (closed beta access control)

Moved from app/api/admin_users.py → app/api/admin/users.py
"""

import secrets
from datetime import datetime, timedelta
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select, func, text
from sqlalchemy.ext.asyncio import AsyncSession

from sqlalchemy import delete as sa_delete

from app.db import get_db, User, Invite
from app.api.auth import get_current_user
from app.api.admin.deps import require_admin
from app.services.auth_service import create_user, get_user_by_email

# ─── Schemas ───────────────────────────────────────────────────

class InviteCreate(BaseModel):
    email: Optional[str] = None
    role: str = Field(default="beta_user", pattern="^(admin|beta_user)$")
    note: Optional[str] = Field(None, max_length=500)
    expires_in_days: int = Field(default=7, ge=1, le=90)


class InviteResponse(BaseModel):
    id: str
    token: str
    email: Optional[str]
    role: str
    note: Optional[str]
    status: str
    created_by: str
    used_by: Optional[str] = None
    used_at: Optional[datetime] = None
    expires_at: datetime
    created_at: datetime
    invite_url: str

    class Config:
        from_attributes = True


class InviteListResponse(BaseModel):
    invites: List[InviteResponse]
    total: int


class UserAdminResponse(BaseModel):
    id: str
    email: str
    name: Optional[str]
    role: str
    is_active: bool
    created_at: datetime
    memory_count: int = 0
    session_count: int = 0
    password_plain: Optional[str] = None  # Beta only
    hosting_mode: Optional[str] = None
    deploy_status: Optional[str] = None
    agent_url: Optional[str] = None
    setup_completed: bool = False

    class Config:
        from_attributes = True


class UserListResponse(BaseModel):
    users: List[UserAdminResponse]
    total: int


class UserUpdateRequest(BaseModel):
    role: Optional[str] = Field(None, pattern="^(admin|beta_user)$")
    is_active: Optional[bool] = None
    name: Optional[str] = None


class InviteSignupRequest(BaseModel):
    token: str
    email: str
    password: str = Field(min_length=6)
    name: Optional[str] = None


class InviteValidateResponse(BaseModel):
    valid: bool
    email: Optional[str] = None
    role: Optional[str] = None
    expires_at: Optional[datetime] = None
    message: Optional[str] = None


# ─── Admin Router (protected) ─────────────────────────────────

router = APIRouter(prefix="/admin", tags=["Admin — Users & Invites"])

INVITE_BASE_URL = "https://brain.toup.ai/admin/invite"


@router.post("/invites", response_model=InviteResponse, status_code=201)
async def create_invite(
    body: InviteCreate,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Create a new invite token. Only admins."""
    token = secrets.token_urlsafe(32)
    expires_at = datetime.utcnow() + timedelta(days=body.expires_in_days)

    invite = Invite(
        token=token,
        created_by=admin.id,
        email=body.email,
        role=body.role,
        note=body.note,
        status="pending",
        expires_at=expires_at,
    )
    db.add(invite)
    await db.commit()
    await db.refresh(invite)
    return _invite_to_response(invite)


@router.get("/invites", response_model=InviteListResponse)
async def list_invites(
    status_filter: Optional[str] = None,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """List all invites. Optional filter by status."""
    query = select(Invite).order_by(Invite.created_at.desc())
    if status_filter:
        query = query.where(Invite.status == status_filter)

    result = await db.execute(query)
    invites = result.scalars().all()

    now = datetime.utcnow()
    for inv in invites:
        if inv.status == "pending" and inv.expires_at < now:
            inv.status = "expired"
    await db.commit()

    return InviteListResponse(
        invites=[_invite_to_response(i) for i in invites],
        total=len(invites),
    )


@router.delete("/invites/{invite_id}")
async def revoke_invite(
    invite_id: str,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Revoke a pending invite."""
    result = await db.execute(select(Invite).where(Invite.id == invite_id))
    invite = result.scalar_one_or_none()
    if not invite:
        raise HTTPException(404, "Invite not found")
    if invite.status != "pending":
        raise HTTPException(400, f"Cannot revoke invite with status '{invite.status}'")
    invite.status = "revoked"
    await db.commit()
    return {"success": True, "message": "Invite revoked"}


@router.get("/users", response_model=UserListResponse)
async def list_users(
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """List all users with stats. Only admins."""
    from app.db.models import Memory, Conversation, AgentConfig

    mem_sub = (
        select(Memory.user_id, func.count(Memory.id).label("mem_count"))
        .where(Memory.is_deleted == False)
        .group_by(Memory.user_id)
        .subquery()
    )
    sess_sub = (
        select(Conversation.user_id, func.count(Conversation.id).label("sess_count"))
        .group_by(Conversation.user_id)
        .subquery()
    )

    query = (
        select(
            User,
            func.coalesce(mem_sub.c.mem_count, 0).label("memory_count"),
            func.coalesce(sess_sub.c.sess_count, 0).label("session_count"),
            AgentConfig.hosting_mode,
            AgentConfig.deploy_status,
            AgentConfig.agent_url,
            AgentConfig.setup_completed,
        )
        .outerjoin(mem_sub, User.id == mem_sub.c.user_id)
        .outerjoin(sess_sub, User.id == sess_sub.c.user_id)
        .outerjoin(AgentConfig, User.id == AgentConfig.user_id)
        .order_by(User.created_at.desc())
    )

    result = await db.execute(query)
    rows = result.all()

    users = []
    for user, mem_count, sess_count, hosting_mode, deploy_status, agent_url, setup_completed in rows:
        users.append(UserAdminResponse(
            id=user.id,
            email=user.email,
            name=user.name,
            role=getattr(user, "role", "beta_user"),
            is_active=user.is_active,
            created_at=user.created_at,
            memory_count=mem_count,
            session_count=sess_count,
            password_plain=getattr(user, "password_plain", None),
            hosting_mode=hosting_mode,
            deploy_status=deploy_status,
            agent_url=agent_url,
            setup_completed=setup_completed or False,
        ))
    return UserListResponse(users=users, total=len(users))


async def _build_usage_for_conv_ids(
    conv_ids: list[str], db: AsyncSession,
) -> list[dict]:
    """Shared helper: build usage summary for a set of conversation IDs."""
    from datetime import timedelta
    from app.db import Message
    from app.api.llm_setup import (
        UsageSummaryOut, ProviderUsageOut, ModelUsageOut, model_to_provider,
    )
    from app.config import settings

    now = datetime.utcnow()
    periods = {
        "today": now.replace(hour=0, minute=0, second=0, microsecond=0),
        "7d": now - timedelta(days=7),
        "30d": now - timedelta(days=30),
        "all": datetime(2000, 1, 1),
    }

    if not conv_ids:
        return [
            UsageSummaryOut(
                period=p, total_input_tokens=0, total_output_tokens=0,
                total_tokens=0, total_cost_usd=0.0, total_requests=0, providers=[],
            ).model_dump()
            for p in periods
        ]

    results = []
    pricing = settings.pricing_per_1k

    for period_name, since in periods.items():
        stmt = (
            select(
                Message.model_used,
                func.coalesce(func.sum(func.coalesce(Message.tokens_prompt, 0)), 0).label("inp"),
                func.coalesce(func.sum(func.coalesce(Message.tokens_completion, 0)), 0).label("out"),
                func.count().label("cnt"),
            )
            .where(
                Message.conversation_id.in_(conv_ids),
                Message.role == "assistant",
                Message.model_used.isnot(None),
                Message.created_at >= since,
            )
            .group_by(Message.model_used)
        )
        rows = (await db.execute(stmt)).all()

        provider_map: dict[str, dict] = {}
        for model_used, inp, out, cnt in rows:
            provider = model_to_provider(model_used or "")
            p_cost = pricing.get(model_used, {"input": 0.003, "output": 0.012})
            cost = (inp * p_cost["input"] / 1000) + (out * p_cost["output"] / 1000)

            model_entry = {
                "model": model_used or "unknown", "provider": provider,
                "input_tokens": int(inp), "output_tokens": int(out),
                "total_tokens": int(inp + out), "cost_usd": round(cost, 4),
                "request_count": int(cnt),
            }

            if provider not in provider_map:
                provider_map[provider] = {
                    "provider": provider, "input_tokens": 0, "output_tokens": 0,
                    "total_tokens": 0, "cost_usd": 0.0, "request_count": 0, "models": [],
                }
            prov = provider_map[provider]
            prov["input_tokens"] += int(inp)
            prov["output_tokens"] += int(out)
            prov["total_tokens"] += int(inp + out)
            prov["cost_usd"] = round(prov["cost_usd"] + cost, 4)
            prov["request_count"] += int(cnt)
            prov["models"].append(model_entry)

        providers = sorted(provider_map.values(), key=lambda p: p["total_tokens"], reverse=True)
        total_inp = sum(p["input_tokens"] for p in providers)
        total_out = sum(p["output_tokens"] for p in providers)

        results.append({
            "period": period_name,
            "total_input_tokens": total_inp,
            "total_output_tokens": total_out,
            "total_tokens": total_inp + total_out,
            "total_cost_usd": round(sum(p["cost_usd"] for p in providers), 4),
            "total_requests": sum(p["request_count"] for p in providers),
            "providers": providers,
        })

    return results


@router.get("/users/{user_id}/usage")
async def get_user_usage(
    user_id: str,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Get token usage summary for a specific user. Admin only."""
    import httpx, logging
    from app.db import Conversation, AgentConfig

    logger = logging.getLogger(__name__)

    # Verify user exists
    target = (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
    if not target:
        raise HTTPException(404, "User not found")

    # Try proxying to user's remote agent first
    try:
        result = await db.execute(
            select(AgentConfig.agent_url, AgentConfig.agent_api_key)
            .where(AgentConfig.user_id == user_id, AgentConfig.deploy_status == "active")
        )
        row = result.first()
        if row and row.agent_url and row.agent_api_key:
            url = f"{row.agent_url}/api/llm-setup/usage/summary"
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.get(url, headers={"X-Agent-Key": row.agent_api_key})
                if resp.status_code == 200:
                    return resp.json()
            logger.warning("Agent usage proxy failed for user %s, falling back to local DB", user_id)
    except Exception as e:
        logger.warning("Agent usage proxy error for user %s: %s", user_id, e)

    # Fallback: query local DB
    conv_result = await db.execute(
        select(Conversation.id).where(Conversation.user_id == user_id)
    )
    conv_ids = [r[0] for r in conv_result.all()]
    return await _build_usage_for_conv_ids(conv_ids, db)


@router.get("/usage/overview")
async def get_usage_overview(
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Get platform-wide usage overview: per-user totals + aggregate. Admin only."""
    import httpx, logging
    from app.db import Message, Conversation, AgentConfig
    from app.api.llm_setup import model_to_provider
    from app.config import settings
    from datetime import timedelta

    logger = logging.getLogger(__name__)
    now = datetime.utcnow()
    periods = {
        "today": now.replace(hour=0, minute=0, second=0, microsecond=0),
        "7d": now - timedelta(days=7),
        "30d": now - timedelta(days=30),
        "all": datetime(2000, 1, 1),
    }

    # Get all users
    users_result = await db.execute(select(User).order_by(User.created_at.desc()))
    all_users = users_result.scalars().all()

    # For each user, try agent proxy first, then local DB
    user_usage_list = []
    pricing = settings.pricing_per_1k

    for u in all_users:
        user_entry = {
            "user_id": u.id,
            "email": u.email,
            "name": u.name or "",
            "role": getattr(u, "role", "beta_user"),
        }

        # Try agent proxy
        agent_data = None
        try:
            result = await db.execute(
                select(AgentConfig.agent_url, AgentConfig.agent_api_key)
                .where(AgentConfig.user_id == u.id, AgentConfig.deploy_status == "active")
            )
            row = result.first()
            if row and row.agent_url and row.agent_api_key:
                url = f"{row.agent_url}/api/llm-setup/usage/summary"
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.get(url, headers={"X-Agent-Key": row.agent_api_key})
                    if resp.status_code == 200:
                        agent_data = resp.json()
        except Exception:
            pass

        if agent_data:
            # Extract 30d summary for the overview row
            summary_30d = next((s for s in agent_data if s.get("period") == "30d"), None)
            if summary_30d:
                user_entry["total_tokens_30d"] = summary_30d.get("total_tokens", 0)
                user_entry["total_cost_30d"] = summary_30d.get("total_cost_usd", 0.0)
                user_entry["total_requests_30d"] = summary_30d.get("total_requests", 0)
            else:
                user_entry["total_tokens_30d"] = 0
                user_entry["total_cost_30d"] = 0.0
                user_entry["total_requests_30d"] = 0
            summary_all = next((s for s in agent_data if s.get("period") == "all"), None)
            if summary_all:
                user_entry["total_tokens_all"] = summary_all.get("total_tokens", 0)
                user_entry["total_cost_all"] = summary_all.get("total_cost_usd", 0.0)
                user_entry["total_requests_all"] = summary_all.get("total_requests", 0)
            else:
                user_entry["total_tokens_all"] = 0
                user_entry["total_cost_all"] = 0.0
                user_entry["total_requests_all"] = 0
        else:
            # Local DB fallback
            conv_result = await db.execute(
                select(Conversation.id).where(Conversation.user_id == u.id)
            )
            conv_ids = [r[0] for r in conv_result.all()]

            for suffix, since in [("30d", periods["30d"]), ("all", periods["all"])]:
                if not conv_ids:
                    user_entry[f"total_tokens_{suffix}"] = 0
                    user_entry[f"total_cost_{suffix}"] = 0.0
                    user_entry[f"total_requests_{suffix}"] = 0
                    continue

                stmt = (
                    select(
                        Message.model_used,
                        func.coalesce(func.sum(func.coalesce(Message.tokens_prompt, 0)), 0).label("inp"),
                        func.coalesce(func.sum(func.coalesce(Message.tokens_completion, 0)), 0).label("out"),
                        func.count().label("cnt"),
                    )
                    .where(
                        Message.conversation_id.in_(conv_ids),
                        Message.role == "assistant",
                        Message.model_used.isnot(None),
                        Message.created_at >= since,
                    )
                    .group_by(Message.model_used)
                )
                rows = (await db.execute(stmt)).all()
                total_tokens = 0
                total_cost = 0.0
                total_reqs = 0
                for model_used, inp, out, cnt in rows:
                    total_tokens += int(inp + out)
                    total_reqs += int(cnt)
                    p_cost = pricing.get(model_used, {"input": 0.003, "output": 0.012})
                    total_cost += (inp * p_cost["input"] / 1000) + (out * p_cost["output"] / 1000)

                user_entry[f"total_tokens_{suffix}"] = total_tokens
                user_entry[f"total_cost_{suffix}"] = round(total_cost, 4)
                user_entry[f"total_requests_{suffix}"] = total_reqs

        user_usage_list.append(user_entry)

    # Aggregate totals
    agg_tokens_30d = sum(u.get("total_tokens_30d", 0) for u in user_usage_list)
    agg_cost_30d = round(sum(u.get("total_cost_30d", 0) for u in user_usage_list), 4)
    agg_reqs_30d = sum(u.get("total_requests_30d", 0) for u in user_usage_list)
    agg_tokens_all = sum(u.get("total_tokens_all", 0) for u in user_usage_list)
    agg_cost_all = round(sum(u.get("total_cost_all", 0) for u in user_usage_list), 4)
    agg_reqs_all = sum(u.get("total_requests_all", 0) for u in user_usage_list)

    return {
        "aggregate": {
            "total_tokens_30d": agg_tokens_30d,
            "total_cost_30d": agg_cost_30d,
            "total_requests_30d": agg_reqs_30d,
            "total_tokens_all": agg_tokens_all,
            "total_cost_all": agg_cost_all,
            "total_requests_all": agg_reqs_all,
        },
        "users": sorted(user_usage_list, key=lambda u: u.get("total_tokens_30d", 0), reverse=True),
    }


@router.patch("/users/{user_id}", response_model=UserAdminResponse)
async def update_user(
    user_id: str,
    body: UserUpdateRequest,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Update a user's role, active status, or name. Only admins."""
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(404, "User not found")

    if user.id == admin.id and body.is_active is False:
        raise HTTPException(400, "Cannot deactivate your own account")

    if user.id == admin.id and body.role and body.role != "admin":
        admin_count_result = await db.execute(
            select(func.count(User.id)).where(User.role == "admin", User.is_active == True)
        )
        if admin_count_result.scalar() <= 1:
            raise HTTPException(400, "Cannot remove the last admin")

    if body.role is not None:
        user.role = body.role
    if body.is_active is not None:
        user.is_active = body.is_active
    if body.name is not None:
        user.name = body.name

    await db.commit()
    await db.refresh(user)

    return UserAdminResponse(
        id=user.id,
        email=user.email,
        name=user.name,
        role=getattr(user, "role", "beta_user"),
        is_active=user.is_active,
        created_at=user.created_at,
        password_plain=getattr(user, "password_plain", None),
    )


@router.post("/users/{user_id}/reset-password")
async def admin_reset_password(
    user_id: str,
    body: dict,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Admin resets a user's password. Invalidates their existing tokens."""
    new_password = body.get("new_password", "")
    if len(new_password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    from app.services.auth_service import change_user_password
    await change_user_password(db, user, new_password)
    return {"success": True, "message": f"Password reset for {user.email}"}


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: str,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Delete a user and all their data. Only admins.

    The platform DB is a legacy monolith — it has both platform-only tables
    AND agent-only tables (conversations, memories, etc.) with FK constraints
    to users. We must clean up BOTH to delete the user row. Newer agent-only
    tables (day_chats, context_budget_logs) may not exist, so each cleanup
    step uses a savepoint to avoid poisoning the transaction.
    """
    from app.db.models import (
        Identity, Conversation, Message, Memory, Entity, EntityLink,
        EntityRelationship, BrainStats, MemoryEvent, RetrievalEvent,
        Document, DocumentChunk, Media, CronJob, TelegramUserMapping,
        AgentError, ApiKey, VPSInstance, ManagedContainer, AgentConfig,
        LLMBundleAllocation, LLMUsageRecord,
        App, BuildJob, SoulConfig, StreamingCredential,
    )

    # Safety: cannot delete yourself
    if user_id == admin.id:
        raise HTTPException(400, "Cannot delete your own account")

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(404, "User not found")

    # Safety: cannot delete the last admin
    if getattr(user, "role", "") == "admin":
        admin_count = await db.execute(
            select(func.count(User.id)).where(User.role == "admin", User.is_active == True)
        )
        if admin_count.scalar() <= 1:
            raise HTTPException(400, "Cannot delete the last admin")

    # ── Helper: run a delete inside a savepoint so missing tables/columns
    #    don't abort the outer transaction ──
    async def _safe_exec(stmt):
        try:
            async with db.begin_nested():
                await db.execute(stmt)
        except Exception:
            pass  # table/column missing or other DB error — continue

    # ── 1. Destroy Docker container + drop per-user agent DB on VPS ──
    prefix = user_id[:8]
    try:
        from app.services.docker_host_service import _run_ssh
        mc_result = await db.execute(
            select(ManagedContainer).where(ManagedContainer.user_id == user_id)
        )
        mc = mc_result.scalar_one_or_none()
        if mc:
            await _run_ssh(f"docker rm -f {mc.container_name} 2>/dev/null || true")
            await _run_ssh(
                f"PGPASSWORD=postgres psql -U postgres -h localhost -c "
                f"\"DROP DATABASE IF EXISTS {mc.db_name}\" 2>/dev/null || true"
            )
    except Exception:
        pass

    # ── 2. Clean up agent-only tables (legacy monolith data) ──
    # Each wrapped in savepoint — tables/columns may not exist.
    # Order: children before parents (FK deps).

    # Nullify Memory → Message FK before deleting messages
    await _safe_exec(text(
        "UPDATE memories SET source_message_id = NULL WHERE user_id = :uid"
    ).bindparams(uid=user_id))

    # Messages (FK to conversations)
    await _safe_exec(text(
        "DELETE FROM messages WHERE conversation_id IN "
        "(SELECT id FROM conversations WHERE user_id = :uid)"
    ).bindparams(uid=user_id))

    # Document chunks (FK to documents and memories)
    await _safe_exec(text(
        "DELETE FROM document_chunks WHERE document_id IN "
        "(SELECT id FROM documents WHERE user_id = :uid)"
    ).bindparams(uid=user_id))
    await _safe_exec(text(
        "DELETE FROM document_chunks WHERE memory_id IN "
        "(SELECT id FROM memories WHERE user_id = :uid)"
    ).bindparams(uid=user_id))

    # Entity links and relationships
    await _safe_exec(text(
        "DELETE FROM entity_links WHERE entity_id IN "
        "(SELECT id FROM entities WHERE user_id = :uid)"
    ).bindparams(uid=user_id))
    await _safe_exec(text(
        "DELETE FROM entity_links WHERE memory_id IN "
        "(SELECT id FROM memories WHERE user_id = :uid)"
    ).bindparams(uid=user_id))
    await _safe_exec(text(
        "DELETE FROM entity_relationships WHERE user_id = :uid"
    ).bindparams(uid=user_id))

    # Memory children
    await _safe_exec(text(
        "DELETE FROM memory_events WHERE memory_id IN "
        "(SELECT id FROM memories WHERE user_id = :uid)"
    ).bindparams(uid=user_id))
    await _safe_exec(text(
        "DELETE FROM memory_events WHERE user_id = :uid"
    ).bindparams(uid=user_id))
    await _safe_exec(text(
        "DELETE FROM media WHERE memory_id IN "
        "(SELECT id FROM memories WHERE user_id = :uid)"
    ).bindparams(uid=user_id))
    await _safe_exec(text(
        "DELETE FROM media WHERE user_id = :uid"
    ).bindparams(uid=user_id))
    await _safe_exec(text(
        "DELETE FROM memory_relationships WHERE source_id IN "
        "(SELECT id FROM memories WHERE user_id = :uid) "
        "OR target_id IN (SELECT id FROM memories WHERE user_id = :uid)"
    ).bindparams(uid=user_id))

    # Retrieval events
    await _safe_exec(text(
        "DELETE FROM retrieval_events WHERE user_id = :uid"
    ).bindparams(uid=user_id))

    # Build jobs, reconciliation logs, apps
    await _safe_exec(text("DELETE FROM build_jobs WHERE user_id = :uid").bindparams(uid=user_id))
    await _safe_exec(text("DELETE FROM reconciliation_logs WHERE user_id = :uid").bindparams(uid=user_id))
    await _safe_exec(text("DELETE FROM apps WHERE user_id = :uid").bindparams(uid=user_id))

    # Context budget logs + day chats (may not exist on platform DB)
    await _safe_exec(text("DELETE FROM context_budget_logs WHERE user_id = :uid").bindparams(uid=user_id))

    # Direct user_id tables
    for table_name in [
        "identities", "documents", "memories", "conversations",
        "entities", "brain_stats", "cron_jobs", "telegram_user_mappings",
        "api_keys", "agent_errors", "soul_configs",
        "workflows",
    ]:
        await _safe_exec(text(
            f"DELETE FROM {table_name} WHERE user_id = :uid"
        ).bindparams(uid=user_id))

    # Day chats (may not exist)
    await _safe_exec(text("DELETE FROM day_chats WHERE user_id = :uid").bindparams(uid=user_id))

    # ── 3. Delete platform-only tables ──
    try:
        from app.db.models import LLMProxyEvent
        await _safe_exec(sa_delete(LLMProxyEvent).where(LLMProxyEvent.user_id == user_id))
    except Exception:
        pass

    await db.execute(sa_delete(StreamingCredential).where(StreamingCredential.user_id == user_id))
    await db.execute(sa_delete(LLMUsageRecord).where(LLMUsageRecord.user_id == user_id))
    await db.execute(sa_delete(LLMBundleAllocation).where(LLMBundleAllocation.user_id == user_id))
    await db.execute(sa_delete(ManagedContainer).where(ManagedContainer.user_id == user_id))
    await db.execute(sa_delete(VPSInstance).where(VPSInstance.user_id == user_id))
    await db.execute(sa_delete(AgentConfig).where(AgentConfig.user_id == user_id))

    # ── 4. Update/delete invites ──
    inv_result = await db.execute(select(Invite).where(Invite.used_by == user_id))
    for inv in inv_result.scalars().all():
        inv.used_by = None
        inv.used_at = None
    await db.execute(sa_delete(Invite).where(Invite.created_by == user_id))

    # ── 5. Delete the user (raw SQL to avoid ORM loading relationships
    #    that reference columns missing from legacy platform DB) ──
    await db.execute(text("DELETE FROM users WHERE id = :uid").bindparams(uid=user_id))
    await db.commit()

    return {"success": True, "message": f"User {user.email} and all data deleted"}


# ─── Public Invite Endpoints (no auth) ────────────────────────

invite_router = APIRouter(prefix="/auth", tags=["Authentication"])


@invite_router.get("/invite/{token}", response_model=InviteValidateResponse)
async def validate_invite(
    token: str,
    db: AsyncSession = Depends(get_db),
):
    """Public endpoint: check if an invite token is valid."""
    result = await db.execute(select(Invite).where(Invite.token == token))
    invite = result.scalar_one_or_none()
    if not invite:
        return InviteValidateResponse(valid=False, message="Invalid invite link")

    now = datetime.utcnow()
    if invite.status != "pending":
        return InviteValidateResponse(valid=False, message=f"Invite has been {invite.status}")
    if invite.expires_at < now:
        invite.status = "expired"
        await db.commit()
        return InviteValidateResponse(valid=False, message="Invite has expired")

    return InviteValidateResponse(
        valid=True,
        email=invite.email,
        role=invite.role,
        expires_at=invite.expires_at,
    )


@invite_router.post("/register/invite")
async def register_with_invite(
    body: InviteSignupRequest,
    db: AsyncSession = Depends(get_db),
):
    """Redeem an invite token and create a new account."""
    from app.services.auth_service import create_access_token

    result = await db.execute(select(Invite).where(Invite.token == body.token))
    invite = result.scalar_one_or_none()
    if not invite:
        raise HTTPException(400, "Invalid invite token")

    now = datetime.utcnow()
    if invite.status != "pending":
        raise HTTPException(400, f"Invite has already been {invite.status}")
    if invite.expires_at < now:
        invite.status = "expired"
        await db.commit()
        raise HTTPException(400, "Invite has expired")

    if invite.email and invite.email.lower() != body.email.lower():
        raise HTTPException(400, f"This invite is reserved for {invite.email}")

    existing = await get_user_by_email(db, body.email)
    if existing:
        raise HTTPException(400, "Email already registered")

    user = await create_user(db, body.email, body.password, body.name)
    user.role = invite.role
    await db.flush()

    invite.status = "used"
    invite.used_by = user.id
    invite.used_at = now

    await db.commit()
    await db.refresh(user)

    token = create_access_token(user.id)
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "email": user.email,
            "name": user.name,
            "role": user.role,
        },
    }


# ─── Helpers ───────────────────────────────────────────────────

def _invite_to_response(invite: Invite) -> InviteResponse:
    return InviteResponse(
        id=invite.id,
        token=invite.token,
        email=invite.email,
        role=invite.role,
        note=invite.note,
        status=invite.status,
        created_by=invite.created_by,
        used_by=invite.used_by,
        used_at=invite.used_at,
        expires_at=invite.expires_at,
        created_at=invite.created_at,
        invite_url=f"{INVITE_BASE_URL}/{invite.token}",
    )
