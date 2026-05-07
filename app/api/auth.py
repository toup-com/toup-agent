"""Authentication endpoints"""

from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from typing import Optional
from jose import jwt, JWTError

from app.db import get_db
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
from app.config import settings

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

    # Token revocation: reject tokens issued before last password change
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
    """Update user profile (name)."""
    if body.name is not None:
        current_user.name = body.name
        current_user.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(current_user)
    return {"id": current_user.id, "email": current_user.email, "name": current_user.name}


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
async def sso_exchange(body: SSOExchangeRequest, response: Response, db: AsyncSession = Depends(get_db)):
    """Exchange an SSO token for a fresh JWT."""
    user_id = decode_access_token(body.token)
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired SSO token")
    user = await get_user_by_id(db, user_id)
    if not user or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found or inactive")
    new_token = create_access_token(user.id)
    response.set_cookie(
        key=SSO_COOKIE_NAME, value=new_token, domain=SSO_COOKIE_DOMAIN,
        max_age=SSO_COOKIE_MAX_AGE, httponly=True, secure=True, samesite="none", path="/",
    )
    return Token(access_token=new_token)


# ── Logout ───────────────────────────────────────────────────────

@router.post("/logout")
async def logout_user(response: Response):
    """Logout — clear SSO cookie"""
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


class ReauthResponse(BaseModel):
    sensitive_action_token: str
    expires_in: int


@router.post("/reauth", response_model=ReauthResponse)
async def reauth(
    current_user=Depends(get_current_user),
):
    """Issue a 5-minute single-use sensitive-action token. Required
    before destructive endpoints like /auth/delete-account. The
    returned token is bound to the calling user's id and is invalid
    for any other user or any other purpose.

    Future destructive endpoints (rotate API keys, change billing
    email, etc.) will share this issuance path with a different
    `expected_purpose` parameter on verification.
    """
    token, _exp = issue_sensitive_action_token(
        user_id=str(current_user.id),
        purpose=SensitiveActionPurpose.DELETE_ACCOUNT,
    )
    return ReauthResponse(
        sensitive_action_token=token,
        expires_in=DEFAULT_TTL_SECONDS,
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
