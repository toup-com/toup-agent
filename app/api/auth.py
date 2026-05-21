"""Authentication endpoints"""

import logging
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from typing import Optional
from jose import jwt, JWTError

from app.db import get_db
from app.db.models import UserSession
from app.schemas import (
    UserCreate, UserLogin, UserResponse, Token,
    ChangePasswordRequest, UpdateProfileRequest,
)
from app.services import (
    authenticate_user, create_user, get_user_by_email,
    get_user_by_id, create_access_token, decode_access_token,
    verify_password, change_user_password,
)
from app.services.rate_limiter import login_rate_limiter, signup_rate_limiter
from app.services.turnstile import verify_turnstile_token
from app.services.session_tracker import (
    parse_device_label, record_login_session, JTI_REVOCATION_GRACE_SECONDS,
)
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])
security = HTTPBearer(auto_error=False)

# SSO cookie config
SSO_COOKIE_NAME = "hex_sso_token"
SSO_COOKIE_DOMAIN = ".toup.ai"
SSO_COOKIE_MAX_AGE = 60 * 60 * 24 * 7  # 1 week


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: AsyncSession = Depends(get_db)
):
    """Dependency to get the current authenticated user.
    Checks Bearer token first, then falls back to SSO cookie.
    Validates token was issued after last password change."""
    token = None
    user_id = None

    # 1. Try Bearer token
    if credentials and credentials.credentials:
        token = credentials.credentials
        user_id = decode_access_token(token)

    # 2. Fall back to SSO cookie
    if not user_id:
        token = request.cookies.get(SSO_COOKIE_NAME)
        if token:
            user_id = decode_access_token(token)

    # 3. Fall back to agent mode
    if not user_id:
        agent_key = request.headers.get("x-agent-key", "")
        if settings.agent_api_key and agent_key == settings.agent_api_key and settings.user_id:
            user_id = settings.user_id
            token = None  # Skip token revocation check for agent mode

    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = await get_user_by_id(db, user_id)
    if not user:
        # In agent mode, auto-create a stub user
        if settings.agent_api_key and settings.user_id and user_id == settings.user_id:
            from app.db.models import User
            user = User(id=user_id, email=f"{user_id[:8]}@agent.local", hashed_password="", name="Agent Owner")
            db.add(user)
            await db.commit()
            await db.refresh(user)
        else:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User account is disabled")

    # Token revocation: reject tokens issued before last password change.
    # Also: per-session revocation via the UserSession table (account
    # page "sign out this device"). The two checks are complementary —
    # password change scorches everything via password_changed_at; the
    # per-session table lets users sign out one device at a time.
    if token and getattr(user, 'password_changed_at', None):
        try:
            payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
            token_iat = datetime.utcfromtimestamp(payload.get("iat", 0))
            if token_iat < user.password_changed_at:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token invalidated by password change. Please log in again.",
                )
        except JWTError:
            pass  # Token already validated by decode_access_token

    # Per-session revocation. Skip for agent-mode (no token, no jti).
    # The grace window swallows the race where login returns the JWT
    # before the session-row commit lands in the read connection.
    if token:
        try:
            payload = jwt.decode(
                token, settings.jwt_secret, algorithms=[settings.jwt_algorithm]
            )
            jti = payload.get("jti")
            token_iat = datetime.utcfromtimestamp(payload.get("iat", 0))
            if jti:
                from app.services.session_tracker import (
                    get_session_by_jti, maybe_bump_last_seen,
                )
                session_row = await get_session_by_jti(db, jti)
                if session_row is None:
                    # No matching session row. Could be: legacy token
                    # issued before the table existed (let through —
                    # grace), or a freshly-issued token whose commit
                    # hasn't been observed yet (grace window), or a
                    # forged/tampered jti (reject after grace).
                    age = (datetime.utcnow() - token_iat).total_seconds()
                    if age > JTI_REVOCATION_GRACE_SECONDS:
                        # Treat as legacy/missing. Don't reject —
                        # we'd nuke every pre-rollout session. The
                        # password_changed_at gate above is still the
                        # blunt-but-reliable revocation lever.
                        pass
                elif session_row.is_revoked:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="This session was signed out. Please log in again.",
                    )
                else:
                    # Live session — keep last_seen_at fresh.
                    request.state.user_session_jti = jti  # for endpoints that need it
                    await maybe_bump_last_seen(db, session_row)
        except HTTPException:
            raise
        except JWTError:
            pass

    return user


# ── Register ─────────────────────────────────────────────────────

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    user_data: UserCreate,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Register a new user.

    Phase-3 hardening (docs/onboarding/prewarm-phase0.md):
      * IP-keyed rate limit (5 signups per IP per hour by default).
        Caps the "spam signups burn pre-warmed containers" attack now
        that Soul.save provisions a Docker container.
      * Cloudflare Turnstile verification when TURNSTILE_SECRET_KEY is
        set. Skipped in dev / CI to keep tests deterministic.
    """
    client_ip = request.client.host if request.client else "unknown"

    # Rate-limit FIRST so we don't waste DB / Turnstile work on
    # already-blocked IPs.
    retry_after = signup_rate_limiter.check(client_ip)
    if retry_after:
        raise HTTPException(
            status_code=429,
            detail=f"Too many signup attempts from this network. Try again in {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)},
        )

    # Turnstile gate — skipped when no secret configured (dev / CI).
    # Verifies BEFORE we touch the DB so a failing token never hits
    # the User uniqueness constraint.
    if not await verify_turnstile_token(
        user_data.cf_turnstile_token, remote_ip=client_ip,
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="CAPTCHA verification failed. Please refresh and try again.",
        )

    existing = await get_user_by_email(db, user_data.email)
    if existing:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")

    # Record the attempt only after both gates pass — we don't count a
    # bot's failed Turnstile attempts toward the rate limit (otherwise
    # one bot can lock out a shared IP). Successful create is the
    # signal that consumes a slot.
    user = await create_user(db, email=user_data.email, password=user_data.password, name=user_data.name)
    signup_rate_limiter.record(client_ip)

    # Kick off the prewarm IMMEDIATELY at signup — before the user
    # even navigates to /onboarding/welcome. Previously the prewarm
    # fired from OnboardingShell.mount, which was 1-2s later (auto-
    # login round-trip + React Router navigate + chunk hydrate +
    # mount). Moving it here:
    #   - Cuts ~1-2s off the boot budget the user-visible flow can
    #     overlap with prewarm. By Install time, the container is
    #     ~1-2s further along its boot.
    #   - Provides a more deterministic start point for telemetry —
    #     [PREWARM-START] fires synchronously with the User row insert.
    # Fire-and-forget; failures are logged + non-blocking. The
    # OnboardingShell-mount fallback at OnboardingShell.tsx remains in
    # place as a defense (schedule_prewarm short-circuits on already-
    # running/provisioning containers, so the second call is a cheap
    # no-op rather than a duplicate provision).
    if settings.prewarm_on_soul_save:
        try:
            from app.api.agent_setup import _get_or_create_config
            config = await _get_or_create_config(str(user.id), db)
            # Default to managed for unified onboarding. The signup
            # form has no hosting choice, and the frontend's
            # Welcome.advance writes hosting_mode='managed' anyway —
            # priming here lets the prewarm container boot once with
            # the correct env so we don't recreate at Welcome→Soul.
            dirty = False
            if config.hosting_mode in (None, "self-hosted"):
                config.hosting_mode = "managed"
                dirty = True
            if not config.whatsapp_mode:
                config.whatsapp_mode = "qr_link"
                dirty = True
            if dirty:
                await db.commit()
            # Phase A.2 (never-sleep plan): prefer pool claim — sub-
            # second bind to a pre-booted container vs ~15s cold-boot
            # via schedule_prewarm. claim_or_prewarm internally falls
            # back to the legacy prewarm path on pool-exhausted /
            # feature-flag-off, so the slow-path safety net remains.
            from app.services.pool_service import claim_or_prewarm
            await claim_or_prewarm(db, str(user.id))
        except Exception as e:
            logger.warning(
                "[REGISTER] Prewarm/claim schedule failed for user %s: %s",
                str(user.id)[:8], e,
            )
    return user


# ── Login (with rate limiting) ───────────────────────────────────

@router.post("/login", response_model=Token)
async def login(credentials: UserLogin, request: Request, response: Response, db: AsyncSession = Depends(get_db)):
    """Login and get access token. Rate-limited: 5 attempts per 5 minutes."""
    login_id = credentials.email.strip()
    client_ip = request.client.host if request.client else "unknown"

    # Rate limit check
    retry_after = login_rate_limiter.check(client_ip, login_id)
    if retry_after:
        raise HTTPException(
            status_code=429,
            detail=f"Too many login attempts. Try again in {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)},
        )

    user = await authenticate_user(db, login_id, credentials.password)
    if not user and '@' not in login_id:
        user = await authenticate_user(db, f"{login_id}@toup.ai", credentials.password)

    if not user:
        login_rate_limiter.record(client_ip, login_id)
        # Check if user exists at all to give a more helpful message
        from app.services import get_user_by_email
        exists = await get_user_by_email(db, login_id)
        if not exists and '@' not in login_id:
            exists = await get_user_by_email(db, f"{login_id}@toup.ai")
        if not exists:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="No account found with this email. Please sign up first.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    login_rate_limiter.clear(client_ip, login_id)
    token = create_access_token(user.id)

    # Record the session BEFORE returning the token. This guarantees
    # the row is committed before the client uses the JWT to call
    # protected endpoints — avoids the otherwise-tiny race window where
    # get_current_user sees a jti with no matching row. The grace
    # window in session_tracker is a belt-and-suspenders safeguard.
    await record_login_session(db, user.id, token, request)

    response.set_cookie(
        key=SSO_COOKIE_NAME, value=token, domain=SSO_COOKIE_DOMAIN,
        max_age=SSO_COOKIE_MAX_AGE, httponly=True, secure=True,
        samesite="none", path="/",
    )
    return Token(access_token=token)


# ── Profile ──────────────────────────────────────────────────────

@router.get("/me", response_model=UserResponse)
async def get_me(current_user=Depends(get_current_user)):
    """Get current user info"""
    return current_user


@router.patch("/profile")
async def update_profile(
    body: UpdateProfileRequest,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update user profile (name, timezone).

    Timezone updates are silent: frontend captures the OS-resolved IANA
    name via `Intl` on app boot and PATCHes here. We validate against
    zoneinfo and no-op the write if the value already matches — so the
    boot-time call is cheap and idempotent.
    """
    changed = False
    if body.name is not None and body.name != current_user.name:
        current_user.name = body.name
        changed = True
    if body.timezone is not None and body.timezone != current_user.timezone:
        # Validate IANA name. zoneinfo is stdlib (3.9+); raises
        # ZoneInfoNotFoundError for typos / spoofed values.
        from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
        try:
            ZoneInfo(body.timezone)
        except ZoneInfoNotFoundError:
            raise HTTPException(status_code=400, detail="Invalid timezone")
        current_user.timezone = body.timezone
        changed = True
        # TKT-LAT-004: drop the in-process tz cache so the next agent
        # turn picks up the new value instead of serving stale.
        try:
            from app.agent._user_tz_cache import invalidate_cached_user_tz
            invalidate_cached_user_tz(current_user.id)
        except Exception:
            pass
    if changed:
        current_user.updated_at = datetime.utcnow()
        await db.commit()
        await db.refresh(current_user)
    return {
        "id": current_user.id,
        "email": current_user.email,
        "name": current_user.name,
        "timezone": current_user.timezone,
    }


# ── Timezone-from-coordinates ────────────────────────────────────
#
# Account-page "Share precise location" flow. The browser asks the
# user for navigator.geolocation permission, then POSTs the resolved
# (lat, lng) here. We derive the IANA timezone via the bundled
# timezonefinder shapefile (no external API, no per-request network).
# This is the explicit-consent counterpart to the silent Intl capture
# wired into /auth/profile.

# Module-level instance — TimezoneFinder() loads the shapefile on
# construction, so reusing one instance across requests avoids paying
# that cost (~50-200ms) per call.
_tz_finder = None


def _get_tz_finder():
    global _tz_finder
    if _tz_finder is None:
        from timezonefinder import TimezoneFinder
        _tz_finder = TimezoneFinder()
    return _tz_finder


class TimezoneFromCoordsRequest(BaseModel):
    # Latitude bounds match the WGS84 valid range. Anything outside
    # this is either a typo or hostile — reject early so timezonefinder
    # never sees garbage.
    lat: float
    lng: float


@router.post("/timezone-from-coords")
async def timezone_from_coords(
    body: TimezoneFromCoordsRequest,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Resolve precise IANA timezone from (lat, lng) and persist on
    User.timezone. Called by the account page after the user grants
    browser geolocation permission. Returns the resolved zone so the
    frontend can show "Now using America/Toronto" in the share card.
    """
    if not (-90.0 <= body.lat <= 90.0) or not (-180.0 <= body.lng <= 180.0):
        raise HTTPException(status_code=400, detail="Coordinates out of range")

    tf = _get_tz_finder()
    tz_name = tf.timezone_at(lat=body.lat, lng=body.lng)
    if not tz_name:
        # timezonefinder returns None only for a handful of unmapped
        # polar / disputed-water polygons. Unlikely for real user
        # devices but the frontend has a fallback to keep the silently
        # captured Intl value, so we surface the failure honestly.
        raise HTTPException(
            status_code=404, detail="Could not determine timezone for these coordinates"
        )

    # Defense in depth: zoneinfo confirms the string timezonefinder
    # returned is one Python (and the rest of our scheduling stack)
    # accepts. If a bundled shapefile ever drifts ahead of stdlib
    # tzdata this catches it before we corrupt User.timezone.
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
    try:
        ZoneInfo(tz_name)
    except ZoneInfoNotFoundError:
        raise HTTPException(
            status_code=500, detail="Resolved zone is not a known IANA name"
        )

    if current_user.timezone != tz_name:
        current_user.timezone = tz_name
        current_user.updated_at = datetime.utcnow()
        await db.commit()
        # TKT-LAT-004: drop the in-process tz cache so the next agent
        # turn picks up the new value instead of serving stale.
        try:
            from app.agent._user_tz_cache import invalidate_cached_user_tz
            invalidate_cached_user_tz(current_user.id)
        except Exception:
            pass

    return {"timezone": tz_name}


# ── Change Password ──────────────────────────────────────────────

@router.post("/change-password")
async def change_password(
    body: ChangePasswordRequest,
    response: Response,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Change own password. Requires current password. Returns new token."""
    if not verify_password(body.current_password, current_user.hashed_password):
        raise HTTPException(status_code=400, detail="Current password is incorrect")

    await change_user_password(db, current_user, body.new_password)
    new_token = create_access_token(current_user.id)

    # Revoke every existing session — including this one — then record
    # the freshly-minted token. password_changed_at already invalidates
    # old JWTs via iat comparison; this also clears the
    # account-page sessions list so the user sees a clean "just one
    # active device" state after changing their password.
    from app.services.session_tracker import revoke_all_user_sessions_except
    await revoke_all_user_sessions_except(db, current_user.id, except_jti=None)
    await record_login_session(db, current_user.id, new_token, request=None)

    response.set_cookie(
        key=SSO_COOKIE_NAME, value=new_token, domain=SSO_COOKIE_DOMAIN,
        max_age=SSO_COOKIE_MAX_AGE, httponly=True, secure=True,
        samesite="none", path="/",
    )
    return {"access_token": new_token, "message": "Password changed successfully"}


# ── Token Validation ─────────────────────────────────────────────

class ValidateRequest(BaseModel):
    token: Optional[str] = None

class ValidateResponse(BaseModel):
    valid: bool
    user_id: Optional[str] = None
    email: Optional[str] = None
    name: Optional[str] = None


@router.post("/validate", response_model=ValidateResponse)
async def validate_token(request: Request, body: Optional[ValidateRequest] = None, db: AsyncSession = Depends(get_db)):
    """Validate a JWT token and return user info."""
    token = None
    if body and body.token:
        token = body.token
    if not token:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
    if not token:
        token = request.cookies.get(SSO_COOKIE_NAME)
    if not token:
        return ValidateResponse(valid=False)

    user_id = decode_access_token(token)
    if not user_id:
        return ValidateResponse(valid=False)

    user = await get_user_by_id(db, user_id)
    if not user or not user.is_active:
        return ValidateResponse(valid=False)

    # Token revocation check
    if getattr(user, 'password_changed_at', None):
        try:
            payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
            token_iat = datetime.utcfromtimestamp(payload.get("iat", 0))
            if token_iat < user.password_changed_at:
                return ValidateResponse(valid=False)
        except JWTError:
            return ValidateResponse(valid=False)

    return ValidateResponse(valid=True, user_id=str(user.id), email=user.email, name=user.name)


# ── SSO Exchange ─────────────────────────────────────────────────

class SSOExchangeRequest(BaseModel):
    token: str

@router.post("/sso", response_model=Token)
async def sso_exchange(body: SSOExchangeRequest, request: Request, response: Response, db: AsyncSession = Depends(get_db)):
    """Exchange an SSO token for a fresh JWT."""
    user_id = decode_access_token(body.token)
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired SSO token")
    user = await get_user_by_id(db, user_id)
    if not user or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found or inactive")
    new_token = create_access_token(user.id)
    # New token = new session row, same way as /login does it.
    await record_login_session(db, user.id, new_token, request)
    response.set_cookie(
        key=SSO_COOKIE_NAME, value=new_token, domain=SSO_COOKIE_DOMAIN,
        max_age=SSO_COOKIE_MAX_AGE, httponly=True, secure=True, samesite="none", path="/",
    )
    return Token(access_token=new_token)


# ── Logout ───────────────────────────────────────────────────────

@router.post("/logout")
async def logout_user(request: Request, response: Response, db: AsyncSession = Depends(get_db)):
    """Logout — clear SSO cookie + revoke current session."""
    # Revoke the current session row so the JWT can't be reused (e.g.
    # if the user is on a shared device and the cookie is still
    # readable elsewhere). Best-effort — if we can't decode the token
    # we still clear the cookie.
    token = (
        request.cookies.get(SSO_COOKIE_NAME)
        or request.headers.get("authorization", "").removeprefix("Bearer ").strip()
    )
    if token:
        try:
            payload = jwt.decode(
                token, settings.jwt_secret, algorithms=[settings.jwt_algorithm]
            )
            jti = payload.get("jti")
            if jti:
                result = await db.execute(
                    select(UserSession).where(UserSession.jti == jti)
                )
                row = result.scalar_one_or_none()
                if row and not row.is_revoked:
                    row.is_revoked = True
                    await db.commit()
        except (JWTError, Exception) as e:  # noqa: BLE001
            logger.debug("logout session revoke swallowed: %s", e)
    response.delete_cookie(key=SSO_COOKIE_NAME, domain=SSO_COOKIE_DOMAIN, path="/")
    return {"success": True}


# ── Delete Account ───────────────────────────────────────────────
# Apple App Store Guideline 5.1.1(v): users must be able to initiate
# account deletion from within the app. Runs the SAME cascade as the
# admin DELETE /users/{id} endpoint — destroys the managed container +
# tenant DB + Caddy route via the bridge, deletes the Stripe customer
# (cancels every subscription tied to it), archives the per-user
# OpenAI bundle project, wipes 17+ tables (messages, memories, agent
# configs with API keys, etc.), then DELETEs the user row.
#
# Authentication is two-step (per §1.3 sign-off, 2026-05-01):
#   1. Client calls POST /auth/reauth with its existing access token
#      to obtain a 5-minute single-use sensitive-action token.
#   2. Client calls POST /auth/delete-account passing the token in
#      the X-Sensitive-Action-Token header.
#
# Why fresh-JWT instead of password re-entry: avoids storing or
# transmitting the plaintext password on mobile (SecureStore doesn't
# expose it). The token has a stable replay defense via the
# sensitive_action_redemptions table.
#
# Single source of truth for "delete me everywhere" lives in
# `app.services.user_deletion.delete_user_completely`.

from app.services.sensitive_action_token import (
    SensitiveActionPurpose,
    issue_sensitive_action_token,
    verify_sensitive_action_token,
    DEFAULT_TTL_SECONDS,
)
from app.services.user_deletion import (
    DeletionActor,
    DeletionAbortedError,
    delete_user_completely,
)
from dataclasses import asdict


class ReauthRequest(BaseModel):
    # Optional — defaults to delete_account so the existing mobile flow,
    # which calls /reauth with no body, keeps working unchanged. New
    # destructive endpoints pass the matching purpose explicitly.
    purpose: Optional[str] = "delete_account"


class ReauthResponse(BaseModel):
    sensitive_action_token: str
    expires_in: int
    purpose: str


@router.post("/reauth", response_model=ReauthResponse)
async def reauth(
    body: Optional[ReauthRequest] = None,
    current_user=Depends(get_current_user),
):
    """Issue a 5-minute single-use sensitive-action token, bound to the
    calling user and to one specific destructive purpose.

    Defaults to `delete_account` for backward compatibility with the
    mobile delete-account flow that POSTs an empty body. New endpoints
    pass `{"purpose": "<value>"}` explicitly.
    """
    purpose_str = (body.purpose if body else None) or SensitiveActionPurpose.DELETE_ACCOUNT.value
    try:
        purpose = SensitiveActionPurpose(purpose_str)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown sensitive-action purpose: {purpose_str!r}",
        )
    token, _exp = issue_sensitive_action_token(
        user_id=str(current_user.id),
        purpose=purpose,
    )
    return ReauthResponse(
        sensitive_action_token=token,
        expires_in=DEFAULT_TTL_SECONDS,
        purpose=purpose.value,
    )


@router.post("/delete-account")
async def delete_account(
    request: Request,
    response: Response,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete the current user's account and ALL associated data.

    Requires a fresh sensitive-action token issued via POST /auth/
    reauth within the last 5 minutes. Synchronous — by the time this
    returns, Stripe customer is canceled, the managed container is
    destroyed, OpenAI project is archived, and the user row is gone.

    Last admin / cannot-delete-self guards do not apply here: this
    is the user's own account. Apple's guideline doesn't let us
    refuse deletion just because the user is an admin.

    Failure modes:
      - 401: missing / invalid / replayed sensitive-action token
      - 502: cascade aborted (Stripe or container teardown failed).
        User's data is intact; client should surface a step-keyed
        retry message and let the user try again.
      - 200 with cookie cleared: deletion succeeded.
    """
    sat = request.headers.get("X-Sensitive-Action-Token", "")
    await verify_sensitive_action_token(
        db,
        sat,
        expected_user_id=str(current_user.id),
        expected_purpose=SensitiveActionPurpose.DELETE_ACCOUNT,
    )
    # Persist the redemption row before we touch destructive state, so
    # a token that authorized this run is irreversibly burned even if
    # the cascade aborts. Otherwise a Stripe-fail abort would let the
    # user re-issue + re-redeem the same token.
    await db.commit()

    try:
        receipt = await delete_user_completely(
            db,
            current_user,
            actor=DeletionActor.SELF,
            request_ip=(request.client.host if request.client else None),
            request_user_agent=request.headers.get("user-agent"),
        )
    except DeletionAbortedError as e:
        # Don't clear the cookie — the user is still logged in, their
        # data is intact, and they can retry once the underlying
        # service recovers.
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={"step": e.step.value, "message": e.detail},
        )

    # Success — clear the SSO cookie so the now-deleted user's session
    # is invalidated immediately on the client.
    response.delete_cookie(
        key=SSO_COOKIE_NAME, domain=SSO_COOKIE_DOMAIN, path="/",
    )
    return {
        "success": True,
        "message": "Account and all associated data deleted.",
        "receipt": asdict(receipt),
    }


# ── Demo ─────────────────────────────────────────────────────────

@router.post("/demo", response_model=Token)
async def demo_login(db: AsyncSession = Depends(get_db)):
    """Create or login as demo user (for testing)"""
    demo_email = "demo@toup.local"
    demo_password = "demo123456"
    user = await get_user_by_email(db, demo_email)
    if not user:
        user = await create_user(db, email=demo_email, password=demo_password, name="Demo User")
    token = create_access_token(user.id)
    return Token(access_token=token)


# ── Direct-to-agent session token ────────────────────────────────

@router.post("/agent-session-token")
async def agent_session_token(
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Issue a short-lived (5min) token the browser uses to open a
    WebSocket directly to the agent at `agent-<prefix>.agents.toup.ai`,
    bypassing the Railway platform proxy entirely.

    Why: a platform redeploy (Railway swap) would otherwise drop every
    chat WebSocket for ~30s. Direct-to-agent connections live between
    the browser and Contabo VPS, completely independent of Railway,
    so chat keeps streaming through any number of platform deploys.

    Token format: HS256 JWT, secret = the user's `agent_api_key`
    (already known to both platform and agent — the same key the
    platform uses for X-Agent-Key proxy auth, so no new bootstrap).
    Claims:
        sub: user_id
        iss: "toup-platform"
        aud: "toup-agent-session"
        exp: now + 300s

    Refresh: client calls again before expiry. The 5-min window keeps
    the token short enough that leaks don't matter much, while long
    enough that a refresh during a long deploy still succeeds.

    Returns 200 with `{token, agent_url, expires_in}`. Returns 404 if
    the user has no provisioned agent — caller falls back to the
    legacy platform-proxy chat path."""
    from sqlalchemy import select
    from app.db.models import AgentConfig

    result = await db.execute(
        select(AgentConfig).where(AgentConfig.user_id == current_user.id)
    )
    cfg = result.scalar_one_or_none()
    if not cfg or not cfg.agent_url or not cfg.agent_api_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No provisioned agent for this user",
        )
    if cfg.deploy_status != "active":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Agent is not active (status={cfg.deploy_status})",
        )

    expires_in = 300  # 5 min
    now = int(datetime.utcnow().timestamp())
    payload = {
        "sub": str(current_user.id),
        "iss": "toup-platform",
        "aud": "toup-agent-session",
        "iat": now,
        "exp": now + expires_in,
    }
    # python-jose, already in requirements (`from jose import jwt` at top).
    token = jwt.encode(payload, cfg.agent_api_key, algorithm="HS256")
    return {
        "token": token,
        "agent_url": cfg.agent_url,
        "expires_in": expires_in,
    }
