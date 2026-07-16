"""
Admin — User & Invite Management (closed beta access control)

Moved from app/api/admin_users.py → app/api/admin/users.py
"""

import logging
import secrets
from datetime import datetime, timedelta
from typing import Optional, List

logger = logging.getLogger(__name__)

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
    timezone: Optional[str] = None


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
            hosting_mode=hosting_mode,
            deploy_status=deploy_status,
            agent_url=agent_url,
            setup_completed=setup_completed or False,
        ))
    return UserListResponse(users=users, total=len(users))


class RoleUpdateRequest(BaseModel):
    role: str = Field(pattern="^(admin|beta_user)$")


@router.put("/users/{user_id}/role")
async def set_user_role(
    user_id: str,
    body: RoleUpdateRequest,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Promote/demote a user's role. Admin role implies unlimited
    credits (credit enforcement skips role='admin'). Self-demotion is
    refused so the last admin can't lock everyone out by accident."""
    if user_id == admin.id and body.role != "admin":
        raise HTTPException(status_code=400, detail="Refusing self-demotion")
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    old_role = getattr(user, "role", None)
    user.role = body.role
    # Build the response BEFORE commit — pgbouncer txn-mode rule.
    resp = {"id": user.id, "email": user.email, "role": body.role, "was": old_role}
    await db.commit()
    logger.info("[admin] role %s → %s for %s by %s",
                old_role, body.role, user_id[:8], admin.id[:8])
    return resp


async def _build_usage_for_conv_ids(
    conv_ids: list[str], db: AsyncSession, user_id: Optional[str] = None,
) -> list[dict]:
    """Shared helper: build usage summary for a set of conversation IDs.

    When `user_id` is provided, per-user auto-builder usage (BuildUsage table) is
    merged into the per-provider totals, tagged with `source=\"auto_builder\"` on
    the model rows. Chat rows get `source=\"chat\"`."""
    from datetime import timedelta
    from app.db import Message
    from app.api.llm_setup import (
        UsageSummaryOut, ProviderUsageOut, ModelUsageOut, model_to_provider,
        _merge_build_usage,
    )
    from app.config import settings

    now = datetime.utcnow()
    periods = {
        "today": now.replace(hour=0, minute=0, second=0, microsecond=0),
        "7d": now - timedelta(days=7),
        "30d": now - timedelta(days=30),
        "all": datetime(2000, 1, 1),
    }

    results = []
    pricing = settings.pricing_per_1k

    for period_name, since in periods.items():
        provider_map: dict[str, dict] = {}

        if conv_ids:
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

            for model_used, inp, out, cnt in rows:
                provider = model_to_provider(model_used or "")
                p_cost = pricing.get(model_used, {"input": 0.003, "output": 0.012})
                cost = (inp * p_cost["input"] / 1000) + (out * p_cost["output"] / 1000)

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
                prov["models"].append({
                    "model": model_used or "unknown", "provider": provider,
                    "input_tokens": int(inp), "output_tokens": int(out),
                    "total_tokens": int(inp + out), "cost_usd": round(cost, 4),
                    "request_count": int(cnt), "source": "chat",
                })

        if user_id:
            await _merge_build_usage(user_id, since, provider_map, db)

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
            # TKT-LAT-007 (wave 3): shared agent_http client.
            from app.services.agent_http import get_agent_http_client
            client = get_agent_http_client()
            resp = await client.get(
                url,
                headers={"X-Agent-Key": row.agent_api_key},
                timeout=8.0,
            )
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
    return await _build_usage_for_conv_ids(conv_ids, db, user_id=user_id)


# ─────────────────────────────────────────────────────────────────────
#  Platform Costs — admin-editable monthly fixed costs
#  Stored in platform_settings table (key/value). Reads cascade:
#    DB row → settings.platform_cost_*_monthly_usd env default → 0.0
#  This way newly-deployed environments start with sensible env defaults
#  and only diverge once an admin saves a custom value.
# ─────────────────────────────────────────────────────────────────────


_COST_KEYS = ["anthropic_max", "openai_fixed", "vps", "railway", "other"]
_SUPPORTED_CURRENCIES = {"USD", "CAD"}


async def _get_fx_rate(db: AsyncSession) -> float:
    """CAD→USD rate. Stored as a platform setting; defaults to ~0.73 (1 USD ≈ 1.36 CAD).
    Admin can update via PUT /admin/platform-costs body.fx_cad_to_usd."""
    from app.db.models import PlatformSetting
    row = await db.execute(select(PlatformSetting).where(PlatformSetting.key == "fx.cad_to_usd"))
    setting = row.scalar_one_or_none()
    if setting:
        try:
            return float(setting.value)
        except ValueError:
            pass
    return 0.73  # current ~ as of 2026-04


def _to_usd(amount: float, currency: str, fx_cad_to_usd: float) -> float:
    """Convert a native amount to USD. Unknown currencies pass through (assumed USD)."""
    if currency == "CAD":
        return round(amount * fx_cad_to_usd, 4)
    return round(amount, 4)


async def _get_platform_cost(db: AsyncSession, key: str, fx_cad_to_usd: float) -> dict:
    """Read a single cost from DB; fall back to settings.platform_cost_*.

    Returns: {value: float, currency: str, value_usd: float, source: str}
    where `source` is 'db' (admin saved) or 'env_default' (still on the
    deploy-time default).

    Storage format is JSON: {"value": 71.39, "currency": "USD"}. Legacy rows
    stored as a bare number string are read as USD for backwards compat.
    """
    from app.db.models import PlatformSetting
    from app.config import settings as _s
    import json as _json

    db_key = f"cost.{key}"
    row = await db.execute(select(PlatformSetting).where(PlatformSetting.key == db_key))
    setting = row.scalar_one_or_none()
    if setting:
        value = 0.0
        currency = "USD"
        raw = setting.value or ""
        # Try JSON first; fall back to plain-number (legacy pre-currency rows).
        try:
            parsed = _json.loads(raw)
            if isinstance(parsed, dict):
                value = float(parsed.get("value", 0.0))
                currency = str(parsed.get("currency", "USD")).upper()
            elif isinstance(parsed, (int, float)):
                # Legacy: just `71.39`. Treat as USD.
                value = float(parsed)
                currency = "USD"
            else:
                value = 0.0
                currency = "USD"
        except (ValueError, _json.JSONDecodeError):
            try:
                value = float(raw)
                currency = "USD"
            except ValueError:
                pass
        if currency not in _SUPPORTED_CURRENCIES:
            currency = "USD"
        return {
            "value": value,
            "currency": currency,
            "value_usd": _to_usd(value, currency, fx_cad_to_usd),
            "source": "db",
        }
    # Fallback to env default (always treated as USD)
    env_attr = f"platform_cost_{key if key != 'openai_fixed' else 'openai'}_monthly_usd"
    value = float(getattr(_s, env_attr, 0.0))
    return {"value": value, "currency": "USD", "value_usd": value, "source": "env_default"}


async def _get_all_platform_costs(db: AsyncSession) -> dict:
    fx = await _get_fx_rate(db)
    rows = {k: await _get_platform_cost(db, k, fx) for k in _COST_KEYS}
    total_usd = round(sum(r["value_usd"] for r in rows.values()), 4)
    return {"costs": rows, "fx_cad_to_usd": fx, "total_usd": total_usd}


@router.get("/platform-costs")
async def get_platform_costs(
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Get current monthly platform costs with currency.

    Each cost line returns native value + currency + USD-converted value, so
    the admin UI can show both "what you actually pay" (e.g. $215 CAD for
    Anthropic Max) and "USD-equivalent for P&L math" (~$158 USD).
    """
    return await _get_all_platform_costs(db)


class CostLine(BaseModel):
    """One cost edit. Either send `value` alone (currency unchanged) or both."""
    value: float = Field(..., ge=0)
    currency: Optional[str] = Field(None, pattern="^(USD|CAD)$")


class PlatformCostsUpdate(BaseModel):
    """Partial-update payload — only present keys are written."""
    anthropic_max: Optional[CostLine] = None
    openai_fixed: Optional[CostLine] = None
    vps: Optional[CostLine] = None
    railway: Optional[CostLine] = None
    other: Optional[CostLine] = None
    fx_cad_to_usd: Optional[float] = Field(None, gt=0, lt=10)


@router.put("/platform-costs")
async def update_platform_costs(
    body: PlatformCostsUpdate,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Upsert one or more platform costs (value + currency) and/or FX rate.

    Returns the full snapshot so the UI can re-render without a second roundtrip.
    """
    from app.db.models import PlatformSetting
    import json as _json

    # Costs
    payload = body.dict(exclude_unset=True)
    for key in _COST_KEYS:
        line = payload.get(key)
        if line is None:
            continue
        if line.get("value") is None or line.get("value") < 0:
            raise HTTPException(400, f"{key}: value must be non-negative")
        currency = (line.get("currency") or "USD").upper()
        if currency not in _SUPPORTED_CURRENCIES:
            raise HTTPException(400, f"{key}: currency must be USD or CAD")
        db_key = f"cost.{key}"
        new_val = _json.dumps({"value": float(line["value"]), "currency": currency})
        existing = (await db.execute(
            select(PlatformSetting).where(PlatformSetting.key == db_key)
        )).scalar_one_or_none()
        if existing:
            existing.value = new_val
            existing.updated_at = datetime.utcnow()
            existing.updated_by_user_id = admin.id
        else:
            db.add(PlatformSetting(
                key=db_key, value=new_val, updated_by_user_id=admin.id,
            ))

    # FX rate (separate setting key)
    if body.fx_cad_to_usd is not None:
        existing_fx = (await db.execute(
            select(PlatformSetting).where(PlatformSetting.key == "fx.cad_to_usd")
        )).scalar_one_or_none()
        if existing_fx:
            existing_fx.value = str(body.fx_cad_to_usd)
            existing_fx.updated_at = datetime.utcnow()
            existing_fx.updated_by_user_id = admin.id
        else:
            db.add(PlatformSetting(
                key="fx.cad_to_usd", value=str(body.fx_cad_to_usd),
                updated_by_user_id=admin.id,
            ))

    await db.commit()
    return await _get_all_platform_costs(db)


# Legacy USD-only view of costs, used by the overview endpoint until it's
# fully ported to the currency-aware shape. Returns {key: usd_amount}.
async def _legacy_costs_usd_only(db: AsyncSession) -> dict[str, float]:
    snapshot = await _get_all_platform_costs(db)
    return {k: v["value_usd"] for k, v in snapshot["costs"].items()}


@router.get("/usage/overview")
async def get_usage_overview(
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Get platform-wide usage overview: per-user totals + aggregate. Admin only.

    Source-of-truth strategy:
      1. `llm_proxy_events` is authoritative for bundle-mode traffic — every
         hit through `make_anthropic_client`/`make_openai_client` lands there
         tagged with the user_id, INCLUDING auto-builder, voice, day archival,
         and any other internal_llm callers. Source of cost truth.
      2. For users with no proxy events (manual mode, or fresh accounts), fall
         back to the agent's local DB via /api/llm-setup/usage/summary which
         queries Message + BuildUsage on the tenant container.
      3. If the agent proxy is unreachable, fall back to platform-local DB
         (covers self-hosted edge cases).

    This avoids double-counting (bundle users' chat is logged in BOTH
    llm_proxy_events on platform AND messages on agent — we trust the
    platform's record and skip the agent for those users).
    """
    import httpx, logging
    from app.db import Message, Conversation, AgentConfig
    from app.db.models import LLMProxyEvent
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

    # Bulk-aggregate llm_proxy_events per (user_id, period) in 2 round-trips
    # rather than N×M. Both system.* and user-attributable events count for
    # the admin dashboard — admins want total cost visibility, not just
    # cap-relevant cost.
    proxy_aggs: dict[str, dict[str, dict]] = {}
    for suffix, since in [("30d", periods["30d"]), ("all", periods["all"])]:
        stmt = (
            select(
                LLMProxyEvent.user_id,
                func.coalesce(
                    func.sum(LLMProxyEvent.input_tokens + LLMProxyEvent.output_tokens), 0
                ).label("toks"),
                func.coalesce(func.sum(LLMProxyEvent.cost_cents), 0).label("cost_c"),
                func.count().label("reqs"),
            )
            .where(LLMProxyEvent.created_at >= since)
            .group_by(LLMProxyEvent.user_id)
        )
        for uid, toks, cost_c, reqs in (await db.execute(stmt)).all():
            proxy_aggs.setdefault(uid, {})[suffix] = {
                "tokens": int(toks),
                "cost": round(float(cost_c) / 100.0, 4),
                "requests": int(reqs),
            }

    # Provider-level cost breakdown for the platform P&L. Two queries (30d, all)
    # grouped by provider. Lets the admin see the anthropic-vs-openai split at a
    # glance and verify that the Claude Max subscription is being utilized
    # (low Anthropic API cost = users mostly hitting the included sub quota).
    llm_cost_by_provider: dict[str, dict[str, float]] = {"30d": {}, "all": {}}
    for suffix, since in [("30d", periods["30d"]), ("all", periods["all"])]:
        stmt = (
            select(
                LLMProxyEvent.provider,
                func.coalesce(func.sum(LLMProxyEvent.cost_cents), 0).label("cost_c"),
                func.coalesce(func.sum(LLMProxyEvent.input_tokens + LLMProxyEvent.output_tokens), 0).label("toks"),
                func.count().label("reqs"),
            )
            .where(LLMProxyEvent.created_at >= since)
            .group_by(LLMProxyEvent.provider)
        )
        for provider, cost_c, toks, reqs in (await db.execute(stmt)).all():
            llm_cost_by_provider[suffix][provider or "unknown"] = {
                "cost": round(float(cost_c) / 100.0, 4),
                "tokens": int(toks),
                "requests": int(reqs),
            }

    # For each user, choose the best source: proxy events > agent > local DB.
    user_usage_list = []
    pricing = settings.pricing_per_1k

    for u in all_users:
        user_entry = {
            "user_id": u.id,
            "email": u.email,
            "name": u.name or "",
            "role": getattr(u, "role", "beta_user"),
        }

        # Source 1: platform's llm_proxy_events (authoritative for bundle).
        if u.id in proxy_aggs:
            for suffix in ("30d", "all"):
                d = proxy_aggs[u.id].get(suffix, {"tokens": 0, "cost": 0.0, "requests": 0})
                user_entry[f"total_tokens_{suffix}"] = d["tokens"]
                user_entry[f"total_cost_{suffix}"] = d["cost"]
                user_entry[f"total_requests_{suffix}"] = d["requests"]
            user_entry["source"] = "proxy_events"
            user_usage_list.append(user_entry)
            continue

        # Source 2: agent's /usage/summary (manual mode, or bundle user yet to use proxy).
        agent_data = None
        try:
            result = await db.execute(
                select(AgentConfig.agent_url, AgentConfig.agent_api_key)
                .where(AgentConfig.user_id == u.id, AgentConfig.deploy_status == "active")
            )
            row = result.first()
            if row and row.agent_url and row.agent_api_key:
                url = f"{row.agent_url}/api/llm-setup/usage/summary"
                # TKT-LAT-007 (wave 3): shared agent_http client.
                from app.services.agent_http import get_agent_http_client
                client = get_agent_http_client()
                resp = await client.get(
                    url,
                    headers={"X-Agent-Key": row.agent_api_key},
                    timeout=5.0,
                )
                if resp.status_code == 200:
                    agent_data = resp.json()
        except Exception:
            pass

        if agent_data:
            user_entry["source"] = "agent_local"
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
            # Source 3: Local DB (chat messages + BuildUsage). Only reached
            # when (a) no proxy events for this user AND (b) the agent is
            # unreachable. Mostly hits self-hosted edge cases.
            user_entry["source"] = "platform_local"
            from app.db.models import BuildUsage  # noqa: WPS433
            from app.db.database import async_session_maker  # noqa: WPS433

            conv_result = await db.execute(
                select(Conversation.id).where(Conversation.user_id == u.id)
            )
            conv_ids = [r[0] for r in conv_result.all()]

            for suffix, since in [("30d", periods["30d"]), ("all", periods["all"])]:
                total_tokens = 0
                total_cost = 0.0
                total_reqs = 0

                if conv_ids:
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
                    for model_used, inp, out, cnt in rows:
                        total_tokens += int(inp + out)
                        total_reqs += int(cnt)
                        p_cost = pricing.get(model_used, {"input": 0.003, "output": 0.012})
                        total_cost += (inp * p_cost["input"] / 1000) + (out * p_cost["output"] / 1000)

                try:
                    async with async_session_maker() as iso:
                        bu_stmt = (
                            select(
                                func.coalesce(func.sum(BuildUsage.input_tokens), 0).label("inp"),
                                func.coalesce(func.sum(BuildUsage.output_tokens), 0).label("out"),
                                func.coalesce(func.sum(BuildUsage.cost_usd), 0.0).label("cost"),
                                func.count().label("cnt"),
                            )
                            .where(BuildUsage.user_id == u.id, BuildUsage.created_at >= since)
                        )
                        bu_row = (await iso.execute(bu_stmt)).first()
                        if bu_row:
                            total_tokens += int(bu_row.inp) + int(bu_row.out)
                            total_cost += float(bu_row.cost or 0.0)
                            total_reqs += int(bu_row.cnt)
                except Exception as _e:
                    logger.warning("BuildUsage aggregation skipped for user %s: %s", u.id, _e)

                user_entry[f"total_tokens_{suffix}"] = total_tokens
                user_entry[f"total_cost_{suffix}"] = round(total_cost, 4)
                user_entry[f"total_requests_{suffix}"] = total_reqs

        user_usage_list.append(user_entry)

    # ── Revenue + ROI ─────────────────────────────────────────────
    # Per-user revenue from Stripe paid invoices on the user's customer.
    # `bundle_status`, `bundle_currency`, `subscription_id` shipped on each
    # row so the frontend can show the right labels / health indicators
    # without a second roundtrip. Manual users contribute $0 — that's the
    # truthful number; they're not paying us.
    from app.services.stripe_service import sum_invoice_revenue_for_customer

    # Bulk-load each user's bundle config + customer id once. agent_configs
    # rows are 1:1 with users; if a user has none they're flagged as no-bundle.
    cfg_rows = (
        await db.execute(
            select(
                AgentConfig.user_id,
                AgentConfig.bundle_status,
                AgentConfig.bundle_stripe_subscription_id,
            )
        )
    ).all()
    cfg_by_user: dict[str, dict] = {
        row.user_id: {
            "bundle_status": row.bundle_status or "none",
            "subscription_id": row.bundle_stripe_subscription_id,
        }
        for row in cfg_rows
    }

    since_30d_unix = int(periods["30d"].timestamp())
    since_all_unix = int(periods["all"].timestamp())

    # Revenue lookups are network calls; they share the run loop so we'd
    # rather not block the whole endpoint on Stripe latency. Run them in
    # parallel — Stripe's rate limit is ~100 req/s/account so 9 users is
    # comfortably fine.
    import asyncio
    user_id_to_record = {u.id: u for u in all_users}

    async def _revenue_for_user(uid: str) -> dict:
        u = user_id_to_record.get(uid)
        cfg = cfg_by_user.get(uid, {})
        if not u or not u.stripe_customer_id:
            return {
                "user_id": uid,
                "revenue_30d": 0.0,
                "revenue_all": 0.0,
                "currency_30d": {},
                "currency_all": {},
                "bundle_status": cfg.get("bundle_status", "none"),
                "subscription_id": cfg.get("subscription_id"),
            }
        # Off-load the sync Stripe SDK calls to a thread so we don't block
        # the event loop. Two calls: one for 30d, one for all-time.
        rev_30d = await asyncio.to_thread(
            sum_invoice_revenue_for_customer, u.stripe_customer_id, since_30d_unix
        )
        rev_all = await asyncio.to_thread(
            sum_invoice_revenue_for_customer, u.stripe_customer_id, since_all_unix
        )
        return {
            "user_id": uid,
            "revenue_30d": rev_30d["total_usd"],
            "revenue_all": rev_all["total_usd"],
            "currency_30d": rev_30d["currency_breakdown"],
            "currency_all": rev_all["currency_breakdown"],
            "bundle_status": cfg.get("bundle_status", "none"),
            "subscription_id": cfg.get("subscription_id"),
        }

    revenue_results = await asyncio.gather(
        *(_revenue_for_user(u.id) for u in all_users),
        return_exceptions=False,
    )
    rev_by_user = {r["user_id"]: r for r in revenue_results}

    # Decorate each user row with revenue + margin.
    for entry in user_usage_list:
        rev = rev_by_user.get(entry["user_id"], {})
        cost_30d = entry.get("total_cost_30d", 0.0)
        cost_all = entry.get("total_cost_all", 0.0)
        rev_30d = rev.get("revenue_30d", 0.0)
        rev_all = rev.get("revenue_all", 0.0)
        entry["revenue_30d"] = rev_30d
        entry["revenue_all"] = rev_all
        entry["margin_30d"] = round(rev_30d - cost_30d, 4)
        entry["margin_all"] = round(rev_all - cost_all, 4)
        # Margin pct as fraction of revenue (None when revenue=0 to avoid div/0).
        entry["margin_pct_30d"] = (
            round((rev_30d - cost_30d) / rev_30d * 100, 1) if rev_30d > 0 else None
        )
        entry["margin_pct_all"] = (
            round((rev_all - cost_all) / rev_all * 100, 1) if rev_all > 0 else None
        )
        entry["bundle_status"] = rev.get("bundle_status", "none")
        entry["subscription_id"] = rev.get("subscription_id")
        entry["currency_breakdown_30d"] = rev.get("currency_30d", {})
        entry["currency_breakdown_all"] = rev.get("currency_all", {})

    # ── Fixed (recurring) platform costs ─────────────────────────
    # These cover one calendar month each. The 30d window is one full month
    # by convention. For "all time" we conservatively report one month of
    # fixed costs since we don't track when each subscription started — so
    # `total_margin_all` is only meaningful as "P&L of the most recent
    # billing month". A future enhancement could backfill from a billing
    # ledger; for now this matches what the admin actually wants to see.
    # Read cost values from DB (admin-edited via /admin/platform-costs);
    # the helper falls back to settings.platform_cost_* env defaults if
    # the row hasn't been written yet. Each cost has native value +
    # currency + USD-equivalent for the unified P&L math.
    cost_snapshot = await _get_all_platform_costs(db)
    fixed_costs_rich = cost_snapshot["costs"]   # {key: {value, currency, value_usd, source}}
    fixed_costs_breakdown = {k: v["value_usd"] for k, v in fixed_costs_rich.items()}
    fx_cad_to_usd = cost_snapshot["fx_cad_to_usd"]
    fixed_costs_total = round(sum(fixed_costs_breakdown.values()), 4)

    # LLM (variable) costs — sum of per-user proxy event costs.
    llm_cost_30d = round(sum(u.get("total_cost_30d", 0) for u in user_usage_list), 4)
    llm_cost_all = round(sum(u.get("total_cost_all", 0) for u in user_usage_list), 4)

    # True total cost = LLM (variable) + fixed (subscriptions/infra).
    true_total_cost_30d = round(llm_cost_30d + fixed_costs_total, 4)
    true_total_cost_all = round(llm_cost_all + fixed_costs_total, 4)

    # Aggregate totals
    agg_tokens_30d = sum(u.get("total_tokens_30d", 0) for u in user_usage_list)
    agg_reqs_30d = sum(u.get("total_requests_30d", 0) for u in user_usage_list)
    agg_revenue_30d = round(sum(u.get("revenue_30d", 0.0) for u in user_usage_list), 4)
    agg_margin_30d = round(agg_revenue_30d - true_total_cost_30d, 4)
    agg_margin_pct_30d = (
        round(agg_margin_30d / agg_revenue_30d * 100, 1) if agg_revenue_30d > 0 else None
    )

    agg_tokens_all = sum(u.get("total_tokens_all", 0) for u in user_usage_list)
    agg_reqs_all = sum(u.get("total_requests_all", 0) for u in user_usage_list)
    agg_revenue_all = round(sum(u.get("revenue_all", 0.0) for u in user_usage_list), 4)
    agg_margin_all = round(agg_revenue_all - true_total_cost_all, 4)
    agg_margin_pct_all = (
        round(agg_margin_all / agg_revenue_all * 100, 1) if agg_revenue_all > 0 else None
    )

    # Counts of users in each bundle bucket — for quick at-a-glance health.
    paying_users = sum(1 for u in user_usage_list if u.get("bundle_status") == "active")
    free_users = sum(1 for u in user_usage_list if u.get("bundle_status") in (None, "none"))
    cancelling_users = sum(1 for u in user_usage_list if u.get("bundle_status") == "cancelling")
    # Per-user margin uses ONLY their own LLM cost vs revenue (no fixed-cost
    # allocation — those are platform-level, not per-user). The "losing money"
    # bucket means "this user spent more on LLM than they paid us".
    losing_money_users = sum(
        1 for u in user_usage_list if u.get("margin_30d", 0.0) < 0
    )

    # Break-even analysis: how many paying customers would we need to clear
    # all fixed costs at the current ARPU? Useful goal-setting metric.
    avg_revenue_per_paying_user = (
        agg_revenue_30d / paying_users if paying_users > 0 else 0.0
    )
    breakeven_users = (
        int(fixed_costs_total // avg_revenue_per_paying_user) + 1
        if avg_revenue_per_paying_user > 0 else None
    )

    # Revenue currency mix: aggregate paid invoices by currency across users.
    # Stripe gives us native paid amounts (USD vs CAD); we keep both visible
    # so the admin sees "you collected $X USD + $Y CAD" rather than a
    # silently-FX'd blob.
    revenue_by_currency_30d: dict[str, float] = {}
    revenue_by_currency_all: dict[str, float] = {}
    for u in user_usage_list:
        for cur, amt in (u.get("currency_breakdown_30d") or {}).items():
            revenue_by_currency_30d[cur] = round(revenue_by_currency_30d.get(cur, 0.0) + float(amt), 4)
        for cur, amt in (u.get("currency_breakdown_all") or {}).items():
            revenue_by_currency_all[cur] = round(revenue_by_currency_all.get(cur, 0.0) + float(amt), 4)

    # Connection / data-source disclosure. Honest about what's pulled live
    # vs estimated. Drives the "Connection Status" panel on the frontend so
    # the admin knows which numbers come from actual provider invoices vs
    # token-cost calculations.
    data_sources = {
        "stripe_revenue": {
            "connected": bool(settings.stripe_secret_key),
            "mode": (
                "live" if settings.stripe_secret_key and settings.stripe_secret_key.startswith("sk_live_")
                else "test" if settings.stripe_secret_key
                else "not_configured"
            ),
            "label": "Stripe (revenue)",
            "kind": "actual",
        },
        "openai_billing": {
            # Stub for future official OpenAI Costs API integration.
            "connected": bool(settings.openai_admin_key),
            "label": "OpenAI Costs API",
            "kind": "estimated" if not settings.openai_admin_key else "actual",
            "note": (
                "Set OPENAI_ADMIN_KEY to pull official invoice numbers."
                if not settings.openai_admin_key else
                "Pulling official invoice numbers."
            ),
        },
        "anthropic_billing": {
            "connected": bool(settings.anthropic_admin_key),
            "label": "Anthropic Costs API",
            "kind": "estimated" if not settings.anthropic_admin_key else "actual",
            "note": (
                "Set ANTHROPIC_ADMIN_KEY to pull official invoice numbers."
                if not settings.anthropic_admin_key else
                "Pulling official invoice numbers."
            ),
        },
        "llm_proxy_events": {
            "connected": True,
            "label": "Per-call token tracking (estimated cost)",
            "kind": "estimated",
            "note": "Cost computed from input/output tokens × current pricing tables. Within ~5% of official invoices.",
        },
    }

    return {
        "aggregate": {
            # Variable + total
            "total_tokens_30d": agg_tokens_30d,
            "total_cost_30d": llm_cost_30d,            # variable LLM cost only (legacy field)
            "llm_cost_30d": llm_cost_30d,              # explicit variable cost
            "fixed_cost_30d": fixed_costs_total,
            "total_cost_with_fixed_30d": true_total_cost_30d,
            "total_requests_30d": agg_reqs_30d,
            "total_revenue_30d": agg_revenue_30d,
            "total_margin_30d": agg_margin_30d,        # includes fixed costs
            "margin_pct_30d": agg_margin_pct_30d,
            "revenue_by_currency_30d": revenue_by_currency_30d,
            "total_tokens_all": agg_tokens_all,
            "total_cost_all": llm_cost_all,
            "llm_cost_all": llm_cost_all,
            "fixed_cost_all": fixed_costs_total,       # one billing month of fixed (see comment)
            "total_cost_with_fixed_all": true_total_cost_all,
            "total_requests_all": agg_reqs_all,
            "total_revenue_all": agg_revenue_all,
            "total_margin_all": agg_margin_all,
            "margin_pct_all": agg_margin_pct_all,
            "revenue_by_currency_all": revenue_by_currency_all,
            # User buckets
            "paying_users": paying_users,
            "free_users": free_users,
            "cancelling_users": cancelling_users,
            "losing_money_users": losing_money_users,
            "total_users": len(user_usage_list),
            # Break-even
            "avg_revenue_per_paying_user": round(avg_revenue_per_paying_user, 2),
            "breakeven_users": breakeven_users,
            # Cost line items (USD-equivalents)
            "fixed_costs_breakdown": fixed_costs_breakdown,
            # Currency-aware fixed-cost breakdown:
            # {key: {value, currency, value_usd, source}}
            "fixed_costs_rich": fixed_costs_rich,
            "fx_cad_to_usd": fx_cad_to_usd,
            "llm_cost_by_provider_30d": llm_cost_by_provider["30d"],
            "llm_cost_by_provider_all": llm_cost_by_provider["all"],
            # Data sources / connection status
            "data_sources": data_sources,
        },
        "users": sorted(user_usage_list, key=lambda u: u.get("total_tokens_30d", 0), reverse=True),
    }


@router.get("/usage/diagnose/{user_id}")
async def diagnose_user_usage(
    user_id: str,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Explain why a user's usage row is zero. Reports agent-config state, agent-proxy
    reachability, and local-DB message counts so admins can see which branch fired."""
    import httpx
    from app.db import Message, Conversation, AgentConfig
    from app.db.models import BuildUsage, BuildJob

    target = (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
    if not target:
        raise HTTPException(404, "User not found")

    out: dict = {
        "user": {"id": target.id, "email": target.email, "name": target.name or ""},
        "agent_config": None,
        "agent_proxy": None,
        "local_db": None,
        "auto_builder": None,
        "verdict": None,
    }

    cfg_row = (await db.execute(
        select(
            AgentConfig.agent_url,
            AgentConfig.agent_api_key,
            AgentConfig.deploy_status,
            AgentConfig.hosting_mode,
            AgentConfig.setup_completed,
        ).where(AgentConfig.user_id == user_id)
    )).first()

    if cfg_row is None:
        out["agent_config"] = {"exists": False}
    else:
        out["agent_config"] = {
            "exists": True,
            "hosting_mode": cfg_row.hosting_mode,
            "deploy_status": cfg_row.deploy_status,
            "setup_completed": bool(cfg_row.setup_completed),
            "has_agent_url": bool(cfg_row.agent_url),
            "has_agent_key": bool(cfg_row.agent_api_key),
            "agent_url": cfg_row.agent_url,
        }

    if (
        cfg_row
        and cfg_row.deploy_status == "active"
        and cfg_row.agent_url
        and cfg_row.agent_api_key
    ):
        proxy: dict = {"attempted": True}
        try:
            url = f"{cfg_row.agent_url}/api/llm-setup/usage/summary"
            # TKT-LAT-007 (wave 3): shared agent_http client.
            from app.services.agent_http import get_agent_http_client
            client = get_agent_http_client()
            resp = await client.get(
                url,
                headers={"X-Agent-Key": cfg_row.agent_api_key},
                timeout=5.0,
            )
            proxy["status_code"] = resp.status_code
            if resp.status_code == 200:
                data = resp.json()
                summary_30d = next((s for s in data if s.get("period") == "30d"), None) or {}
                summary_all = next((s for s in data if s.get("period") == "all"), None) or {}
                proxy["reported_tokens_30d"] = summary_30d.get("total_tokens", 0)
                proxy["reported_requests_30d"] = summary_30d.get("total_requests", 0)
                proxy["reported_tokens_all"] = summary_all.get("total_tokens", 0)
                proxy["reported_requests_all"] = summary_all.get("total_requests", 0)
            else:
                proxy["body_preview"] = resp.text[:500]
        except Exception as e:
            proxy["error"] = f"{type(e).__name__}: {e}"
        out["agent_proxy"] = proxy
    else:
        out["agent_proxy"] = {"attempted": False, "reason": "agent not active or missing url/key"}

    conv_ids = [r[0] for r in (await db.execute(
        select(Conversation.id).where(Conversation.user_id == user_id)
    )).all()]

    if not conv_ids:
        out["local_db"] = {
            "conversations": 0,
            "assistant_messages": 0,
            "messages_with_model": 0,
            "messages_with_tokens": 0,
        }
    else:
        assistant_count = (await db.execute(
            select(func.count()).select_from(Message).where(
                Message.conversation_id.in_(conv_ids),
                Message.role == "assistant",
            )
        )).scalar_one()
        with_model = (await db.execute(
            select(func.count()).select_from(Message).where(
                Message.conversation_id.in_(conv_ids),
                Message.role == "assistant",
                Message.model_used.isnot(None),
            )
        )).scalar_one()
        with_tokens = (await db.execute(
            select(func.count()).select_from(Message).where(
                Message.conversation_id.in_(conv_ids),
                Message.role == "assistant",
                Message.tokens_prompt.isnot(None),
            )
        )).scalar_one()
        out["local_db"] = {
            "conversations": len(conv_ids),
            "assistant_messages": int(assistant_count),
            "messages_with_model": int(with_model),
            "messages_with_tokens": int(with_tokens),
        }

    # Auto-builder usage (lives on agent DB; absent on platform-only DBs).
    # Run in an isolated session so a missing table can't poison the outer tx.
    from app.db.database import async_session_maker as _sm  # noqa: WPS433
    try:
        async with _sm() as iso:
            try:
                job_count = (await iso.execute(
                    select(func.count()).select_from(BuildJob).where(BuildJob.user_id == user_id)
                )).scalar_one()
            except Exception:
                job_count = 0
            bu_stmt = (
                select(
                    BuildUsage.provider,
                    func.coalesce(func.sum(BuildUsage.input_tokens), 0).label("inp"),
                    func.coalesce(func.sum(BuildUsage.output_tokens), 0).label("out"),
                    func.coalesce(func.sum(BuildUsage.cost_usd), 0.0).label("cost"),
                    func.count().label("cnt"),
                )
                .where(BuildUsage.user_id == user_id)
                .group_by(BuildUsage.provider)
            )
            by_provider: list[dict] = []
            total_tokens = 0
            total_cost = 0.0
            total_calls = 0
            try:
                bu_rows = (await iso.execute(bu_stmt)).all()
            except Exception:
                bu_rows = []
            for provider, inp, out, cost, cnt in bu_rows:
                inp_i = int(inp); out_i = int(out); cnt_i = int(cnt); cost_f = float(cost or 0.0)
                total_tokens += inp_i + out_i
                total_cost += cost_f
                total_calls += cnt_i
                by_provider.append({
                    "provider": provider,
                    "input_tokens": inp_i,
                    "output_tokens": out_i,
                    "total_tokens": inp_i + out_i,
                    "cost_usd": round(cost_f, 4),
                    "llm_calls": cnt_i,
                })
            out["auto_builder"] = {
                "build_jobs": int(job_count),
                "total_tokens": total_tokens,
                "total_cost_usd": round(total_cost, 4),
                "total_llm_calls": total_calls,
                "by_provider": sorted(by_provider, key=lambda p: p["total_tokens"], reverse=True),
            }
    except Exception as e:
        out["auto_builder"] = {"error": f"{type(e).__name__}: {e}"}

    proxy_ok = (
        out["agent_proxy"].get("attempted")
        and out["agent_proxy"].get("status_code") == 200
        and (out["agent_proxy"].get("reported_tokens_all", 0) > 0)
    )
    local_has_data = out["local_db"]["messages_with_model"] > 0

    if proxy_ok:
        out["verdict"] = "agent_proxy_has_data"
    elif local_has_data:
        out["verdict"] = "local_db_has_data"
    elif not out["agent_config"]["exists"]:
        out["verdict"] = "no_agent_config_and_no_local_messages"
    elif cfg_row and cfg_row.deploy_status != "active":
        out["verdict"] = f"agent_not_active (status={cfg_row.deploy_status}) and no local messages"
    elif out["agent_proxy"].get("attempted") and out["agent_proxy"].get("status_code") != 200:
        out["verdict"] = "agent_unreachable_and_no_local_messages"
    elif out["agent_proxy"].get("attempted") and out["agent_proxy"].get("reported_tokens_all", 0) == 0:
        out["verdict"] = "agent_reports_zero_and_no_local_messages"
    else:
        out["verdict"] = "no_usage_recorded"

    return out


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
    if body.timezone is not None:
        user.timezone = body.timezone

    await db.commit()
    await db.refresh(user)

    return UserAdminResponse(
        id=user.id,
        email=user.email,
        name=user.name,
        role=getattr(user, "role", "beta_user"),
        is_active=user.is_active,
        created_at=user.created_at,
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

    Cascade implementation lives in `app.services.user_deletion` so the
    user-facing /auth/delete-account endpoint runs the same logic. This
    handler only adds the admin-specific safety checks on top.
    """
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

    from app.services.user_deletion import (
        DeletionActor, DeletionAbortedError, delete_user_completely,
    )
    user_email = user.email
    try:
        receipt = await delete_user_completely(
            db,
            user,
            actor=DeletionActor.ADMIN,
            actor_user_id=str(admin.id),
        )
    except DeletionAbortedError as e:
        # The cascade aborted at Stripe or container teardown. Audit row
        # records the failure step. Surface a 502 with the step so the
        # admin UI can render a retry-or-escalate message.
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={"step": e.step.value, "message": e.detail},
        )
    return {
        "success": True,
        "message": f"User {user_email} and all data deleted",
        "audit_id": receipt.audit_id,
    }



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


# ─── Sub-agent spawning kill-switch ──────────────────────────────


class SubagentSpawningRequest(BaseModel):
    enabled: bool


class SubagentSpawningResponse(BaseModel):
    user_id: str
    enabled: bool
    container_restart: str  # "ok" | "skipped_no_container" | "failed:<reason>"


@router.post(
    "/users/{user_id}/subagent-spawning",
    response_model=SubagentSpawningResponse,
)
async def set_subagent_spawning(
    user_id: str,
    body: SubagentSpawningRequest,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Flip a tenant's sub-agent spawning kill-switch.

    Updates ``agent_configs.subagent_spawning_enabled`` AND triggers
    a bridge ``/v1/tenants/<prefix>/restart`` so the tenant agent's
    running process picks up the change immediately. Without the
    restart, the flag flip is invisible to the agent until its next
    rollout or manual restart — that gap was diagnosed live on
    2026-05-23 across 4 production tenants.

    Restart failures do NOT roll back the DB change — the column is
    the source of truth; the restart is an "eager apply" optimization.
    Operators see ``container_restart`` in the response so they can
    spot when manual intervention is needed.
    """
    # Late imports to keep the module's top-level light.
    from app.db.models.agent import AgentConfig
    from app.services.docker_host_service import restart_container as svc_restart

    # 1. Verify the user exists (defense-in-depth — UPDATE silently
    #    succeeds with 0 rows for unknown user_ids).
    user_row = (await db.execute(
        select(User).where(User.id == user_id)
    )).scalar_one_or_none()
    if user_row is None:
        raise HTTPException(status_code=404, detail="User not found")

    # 2. Update agent_configs.subagent_spawning_enabled.
    ac = (await db.execute(
        select(AgentConfig).where(AgentConfig.user_id == user_id)
    )).scalar_one_or_none()
    if ac is None:
        raise HTTPException(
            status_code=404,
            detail="User has no AgentConfig row (not provisioned)",
        )
    ac.subagent_spawning_enabled = bool(body.enabled)
    ac.updated_at = datetime.utcnow()
    await db.commit()

    # 3. Restart the tenant container so the change takes effect now
    #    (not at the next rollout). svc_restart is best-effort: if the
    #    user has no managed container OR the bridge is unreachable,
    #    the DB state is still correct — the next rollout will
    #    eventually pick it up.
    restart_status = "skipped_no_container"
    try:
        c = await svc_restart(db, user_id)
        if c is None:
            restart_status = "skipped_no_container"
        elif getattr(c, "status", None) == "error":
            restart_status = f"failed:{(getattr(c, 'error_message', '') or 'unknown')[:80]}"
        else:
            restart_status = "ok"
    except Exception as e:
        restart_status = f"failed:{type(e).__name__}:{str(e)[:80]}"

    return SubagentSpawningResponse(
        user_id=user_id,
        enabled=bool(body.enabled),
        container_restart=restart_status,
    )


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


# ─── Duplicate-account detection (bulletproof plan J, detection-only) ──
#
# One person with two accounts invalidates container-level diagnosis (law
# 5: the 2026-07-04 incident burned hours debugging a healthy agent while
# the user's app was logged into her OTHER account). This endpoint gives
# support/ops the probable-duplicate sets up front. Detection only — no
# auto-merge; merging accounts is an operator decision.

def _email_key(email: str) -> str:
    """Normalize an email to its dedupe key: local part, lowercased,
    dots/plus-suffix stripped, domain ignored (gmail/icloud variants of
    the same person differ exactly this way)."""
    local = (email or "").split("@", 1)[0].lower()
    local = local.split("+", 1)[0].replace(".", "")
    return local


@router.get("/users/duplicates")
async def list_probable_duplicate_accounts(
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Groups of accounts that likely belong to one person. Grouped by
    (a) identical apple_sub, (b) normalized email local-part, (c)
    normalized full name. Returns newest-first inside each group so the
    canonical (usually oldest) account is last."""
    rows = (await db.execute(
        select(User.id, User.email, User.name, User.apple_sub,
               User.created_at, User.is_active)
    )).all()

    def _add(groups: dict, key: str, row) -> None:
        if key:
            groups.setdefault(key, []).append(row)

    by_apple: dict = {}
    by_email: dict = {}
    by_name: dict = {}
    for r in rows:
        _add(by_apple, r.apple_sub or "", r)
        _add(by_email, _email_key(r.email), r)
        _add(by_name, " ".join((r.name or "").lower().split()), r)

    def _emit(groups: dict, kind: str) -> list:
        out = []
        for key, members in groups.items():
            if len(members) < 2:
                continue
            out.append({
                "kind": kind,
                "key": key,
                "accounts": [
                    {
                        "id": str(m.id),
                        "email": m.email,
                        "name": m.name,
                        "created_at": str(m.created_at),
                        "is_active": m.is_active,
                    }
                    for m in sorted(members, key=lambda x: x.created_at or datetime.min,
                                    reverse=True)
                ],
            })
        return out

    groups = _emit(by_apple, "apple_sub") + _emit(by_email, "email_local_part")
    # Name collisions are the noisiest signal — only report a name group
    # when it isn't already fully covered by an email/apple group.
    covered = {frozenset(a["id"] for a in g["accounts"]) for g in groups}
    for g in _emit(by_name, "full_name"):
        if frozenset(a["id"] for a in g["accounts"]) not in covered:
            groups.append(g)

    return {"groups": groups, "count": len(groups)}
