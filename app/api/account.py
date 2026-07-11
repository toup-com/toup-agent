"""Account-page endpoints.

Surfaces user-level concerns that don't fit on the existing /auth
endpoints: notification preferences, active device sessions, and
session-authed API key management. The /api/v1/keys endpoints require
an existing API key to call, which is a chicken-and-egg problem for
the account UI — these mirrors use the standard get_current_user
session auth instead so a user logged in via password can mint their
first key from the browser.

Mounted under /api/account/* by platform_main.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import secrets
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import and_, select
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import get_current_user
from app.db import get_db
from app.db.models import ApiKey, Routine, UserSession
from app.db.models.user import DEFAULT_NOTIFICATION_PREFERENCES

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/account", tags=["Account"])


# =====================================================================
# Notification preferences
# =====================================================================

_HHMM_RE = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


class QuietHours(BaseModel):
    """Local-time DND window (user's platform-side timezone). The
    window may cross midnight (start 22:00, end 08:00)."""

    enabled: bool = True
    start: str = "22:00"
    end: str = "08:00"

    @field_validator("start", "end")
    @classmethod
    def _hhmm(cls, v: str) -> str:
        if not _HHMM_RE.match(v):
            raise ValueError("must be HH:MM (24h)")
        return v


class NotificationPreferences(BaseModel):
    """The shape returned by GET and accepted by PATCH. All keys are
    always present in the response (NULL on the user row is backfilled
    from DEFAULT_NOTIFICATION_PREFERENCES on read), but PATCH treats
    every field as optional so the frontend can flip one toggle at a
    time. Every DEFAULT_NOTIFICATION_PREFERENCES key MUST have a field
    here — PATCH whitelists through this model."""

    morning_briefing: bool
    security_alerts: bool
    product_updates: bool
    autopilot_push: bool
    autopilot_channel_fallback: bool
    quiet_hours: QuietHours
    daily_push_cap: int


class NotificationPreferencesPatch(BaseModel):
    morning_briefing: Optional[bool] = None
    security_alerts: Optional[bool] = None
    product_updates: Optional[bool] = None
    autopilot_push: Optional[bool] = None
    autopilot_channel_fallback: Optional[bool] = None
    quiet_hours: Optional[QuietHours] = None
    daily_push_cap: Optional[int] = Field(default=None, ge=0, le=100)


def _prefs_dict(raw: Any) -> Optional[Dict[str, Any]]:
    """The JSONB value can be a dict (ORM writes) OR a JSON string
    (extension.py writes json.dumps via raw SQL) — readers must
    handle both (see browsing_memory.py for the same defense)."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return None
    return raw if isinstance(raw, dict) else None


def _merged_prefs(row_value: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge stored prefs over the defaults so newly-added toggles
    get their default value for users whose row predates them.
    Type-checked per key against the default's type — a corrupted or
    hand-edited row value never poisons the merged view."""
    merged: Dict[str, Any] = {
        k: (dict(v) if isinstance(v, dict) else v)
        for k, v in DEFAULT_NOTIFICATION_PREFERENCES.items()
    }
    row_value = _prefs_dict(row_value)
    if isinstance(row_value, dict):
        for k, v in row_value.items():
            if k not in merged:
                continue
            default = DEFAULT_NOTIFICATION_PREFERENCES[k]
            if isinstance(default, bool):
                if isinstance(v, bool):
                    merged[k] = v
            elif isinstance(default, int):
                # bool is an int subclass — exclude it explicitly.
                if isinstance(v, int) and not isinstance(v, bool):
                    merged[k] = max(0, min(100, v))
            elif isinstance(default, dict):
                if isinstance(v, dict):
                    sub = dict(default)
                    for sk, sv in v.items():
                        if sk in sub and isinstance(sv, type(sub[sk])):
                            sub[sk] = sv
                    merged[k] = sub
    return merged


async def _try_sync_briefing_routine(
    db: AsyncSession, user_id: str, enabled: bool
) -> None:
    """Best-effort sync of the user's email_briefing Routine row to
    match the morning_briefing toggle. The routines table is
    AGENT-ONLY (see base.py: routines, routine_runs ∈ AGENT_ONLY_TABLES)
    — it doesn't exist on the platform DB where this endpoint runs.
    On platform: the query raises UndefinedTableError and we swallow
    it; the toggle's source of truth is User.notification_preferences,
    and the agent-side Routines UI is the place where the routine row
    actually gets created. On agent: the query lands a real update.

    Logged as debug — this is expected on platform, not an error."""
    try:
        result = await db.execute(
            select(Routine).where(
                and_(
                    Routine.user_id == user_id,
                    Routine.kind == "email_briefing",
                )
            )
        )
        rows = list(result.scalars().all())
        for r in rows:
            r.enabled = enabled
            r.updated_at = datetime.utcnow()
        if rows:
            await db.commit()
    except ProgrammingError as e:
        # Most common: relation "routines" does not exist (platform DB).
        await db.rollback()
        logger.debug("briefing routine sync skipped (routines table unreachable): %s", e)
    except Exception as e:  # noqa: BLE001
        await db.rollback()
        logger.warning("briefing routine sync failed: %s", e)


@router.get("/preferences", response_model=NotificationPreferences)
async def get_preferences(
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> NotificationPreferences:
    # All three toggles persist on the User.notification_preferences
    # JSONB column — single source of truth on the platform side.
    # The morning_briefing toggle ALSO mirrors to the user's
    # email_briefing Routine when that table is reachable (agent DB
    # only), but the toggle's value is read from the JSON column so
    # the account page works identically regardless of which DB it's
    # being served from.
    prefs = _merged_prefs(getattr(current_user, "notification_preferences", None))
    return NotificationPreferences(**prefs)


@router.patch("/preferences", response_model=NotificationPreferences)
async def update_preferences(
    body: NotificationPreferencesPatch,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> NotificationPreferences:
    raw = _prefs_dict(getattr(current_user, "notification_preferences", None))
    current = _merged_prefs(raw)
    if body.morning_briefing is not None:
        current["morning_briefing"] = body.morning_briefing
    if body.security_alerts is not None:
        current["security_alerts"] = body.security_alerts
    if body.product_updates is not None:
        current["product_updates"] = body.product_updates
    if body.autopilot_push is not None:
        current["autopilot_push"] = body.autopilot_push
    if body.autopilot_channel_fallback is not None:
        current["autopilot_channel_fallback"] = body.autopilot_channel_fallback
    if body.quiet_hours is not None:
        current["quiet_hours"] = body.quiet_hours.model_dump()
    if body.daily_push_cap is not None:
        current["daily_push_cap"] = body.daily_push_cap
    # Preserve keys OUTSIDE the whitelist instead of wiping them: the
    # extension writes extension_memory_optout into this JSONB via raw
    # SQL (extension.py), and the old write-whole-dict behavior erased
    # it on every prefs PATCH. Whitelisted keys still only change
    # through their typed fields above.
    if isinstance(raw, dict):
        for k, v in raw.items():
            if k not in DEFAULT_NOTIFICATION_PREFERENCES:
                current[k] = v
    current_user.notification_preferences = current
    current_user.updated_at = datetime.utcnow()
    await db.commit()

    # Best-effort sync to the routine row (no-op on platform DB).
    if body.morning_briefing is not None:
        await _try_sync_briefing_routine(db, current_user.id, bool(body.morning_briefing))

    return NotificationPreferences(**current)


# =====================================================================
# Active sessions (device list)
# =====================================================================

class SessionOut(BaseModel):
    id: str
    device_label: str
    ip_address: Optional[str]
    created_at: str
    last_seen_at: str
    is_current: bool = Field(
        ...,
        description="True for the session whose JWT made this request, so the UI can label it as 'This device' and hide the Sign-out CTA on the current row.",
    )


def _current_jti(request: Request) -> Optional[str]:
    """Extract the jti of the request's session. Populated by
    get_current_user via request.state."""
    return getattr(request.state, "user_session_jti", None)


@router.get("/sessions", response_model=List[SessionOut])
async def list_sessions(
    request: Request,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> List[SessionOut]:
    """List the current user's active (non-revoked) sessions ordered
    newest-first. The current device is marked so the UI can label it
    and hide the per-row "Sign out" button on the row the user is
    currently using."""
    current_jti = _current_jti(request)
    result = await db.execute(
        select(UserSession)
        .where(
            and_(
                UserSession.user_id == current_user.id,
                UserSession.is_revoked == False,  # noqa: E712
            )
        )
        .order_by(UserSession.last_seen_at.desc())
    )
    rows = result.scalars().all()
    return [
        SessionOut(
            id=s.id,
            device_label=s.device_label,
            ip_address=s.ip_address,
            created_at=s.created_at.isoformat(),
            last_seen_at=s.last_seen_at.isoformat(),
            is_current=(current_jti is not None and s.jti == current_jti),
        )
        for s in rows
    ]


@router.post("/sessions/{session_id}/revoke", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_session(
    session_id: str,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Revoke a single session by id. The user can revoke any of
    their own sessions — including the current one, which signs them
    out everywhere. 404 if the row doesn't belong to this user, so we
    don't leak existence of someone else's session ids."""
    result = await db.execute(
        select(UserSession).where(
            and_(
                UserSession.id == session_id,
                UserSession.user_id == current_user.id,
            )
        )
    )
    row = result.scalar_one_or_none()
    if not row:
        raise HTTPException(status_code=404, detail="Session not found")
    if not row.is_revoked:
        row.is_revoked = True
        await db.commit()


@router.post("/sessions/revoke-all-others", status_code=status.HTTP_200_OK)
async def revoke_all_other_sessions(
    request: Request,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, int]:
    """Revoke every session for the user EXCEPT the current one. The
    user keeps their current login; every other device gets booted
    on its next request. Returns {revoked: N}."""
    from app.services.session_tracker import revoke_all_user_sessions_except

    current_jti = _current_jti(request)
    revoked = await revoke_all_user_sessions_except(
        db, current_user.id, except_jti=current_jti
    )
    return {"revoked": revoked}


# =====================================================================
# API key management (session-authed mirror of /api/v1/keys)
# =====================================================================
#
# The /api/v1/keys endpoints require an existing API key to call —
# that's correct for programmatic use but creates a chicken-and-egg
# for the account UI (you can't list keys to make your first one).
# These mirrors use get_current_user instead so the browser-logged-in
# user can manage their keys.

# Centralised in api_v1 — re-implement here only the bits we need so
# this file isn't coupled to api_v1's internals beyond the model.

def _hash_key(raw_key: str) -> str:
    """SHA-256 of the raw key — matches api_v1's hashing so a key
    minted here is usable via /api/v1/* once the user pastes it into
    their script's Authorization header."""
    return hashlib.sha256(raw_key.encode()).hexdigest()


class CreateKeyRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    expires_in_days: Optional[int] = Field(default=None, ge=1, le=3650)
    rate_limit: int = Field(default=60, ge=1, le=10000)


class CreateKeyResponse(BaseModel):
    id: str
    name: str
    key_prefix: str
    key: str = Field(
        ...,
        description="The raw API key. Shown once at creation, never again — frontend must capture and reveal to the user with a copy affordance.",
    )


class KeyOut(BaseModel):
    id: str
    name: str
    key_prefix: str
    rate_limit: int
    is_active: bool
    last_used_at: Optional[str]
    expires_at: Optional[str]
    created_at: str


@router.get("/keys", response_model=List[KeyOut])
async def list_keys(
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> List[KeyOut]:
    result = await db.execute(
        select(ApiKey)
        .where(ApiKey.user_id == current_user.id)
        .order_by(ApiKey.created_at.desc())
    )
    keys = result.scalars().all()
    return [
        KeyOut(
            id=k.id,
            name=k.name,
            key_prefix=k.key_prefix,
            rate_limit=k.rate_limit,
            is_active=k.is_active,
            last_used_at=k.last_used_at.isoformat() if k.last_used_at else None,
            expires_at=k.expires_at.isoformat() if k.expires_at else None,
            created_at=k.created_at.isoformat(),
        )
        for k in keys
    ]


@router.post("/keys", response_model=CreateKeyResponse)
async def create_key(
    body: CreateKeyRequest,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> CreateKeyResponse:
    raw_key = f"hx_{secrets.token_urlsafe(32)}"
    expires_at = (
        datetime.utcnow() + timedelta(days=body.expires_in_days)
        if body.expires_in_days
        else None
    )
    row = ApiKey(
        user_id=current_user.id,
        name=body.name,
        key_hash=_hash_key(raw_key),
        key_prefix=raw_key[:10],
        rate_limit=body.rate_limit,
        expires_at=expires_at,
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return CreateKeyResponse(
        id=row.id, name=row.name, key_prefix=row.key_prefix, key=raw_key
    )


@router.delete("/keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_key(
    key_id: str,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Soft-revoke (is_active=False). Existing requests using the key
    will start failing 401 on their next call. We don't hard-delete
    so audit trails (last_used_at on the row) survive — the user can
    still see "you had a key called X that was last used Y ago" if we
    later add an archive view."""
    result = await db.execute(
        select(ApiKey).where(
            and_(ApiKey.id == key_id, ApiKey.user_id == current_user.id)
        )
    )
    row = result.scalar_one_or_none()
    if not row:
        raise HTTPException(status_code=404, detail="API key not found")
    if row.is_active:
        row.is_active = False
        await db.commit()
