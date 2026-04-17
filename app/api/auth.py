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
from app.services.rate_limiter import login_rate_limiter
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
async def register(user_data: UserCreate, db: AsyncSession = Depends(get_db)):
    """Register a new user"""
    existing = await get_user_by_email(db, user_data.email)
    if existing:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")
    user = await create_user(db, email=user_data.email, password=user_data.password, name=user_data.name)
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
# account deletion from within the app. We anonymize PII and disable
# the account immediately; a background job later purges the row.

class DeleteAccountRequest(BaseModel):
    password: str


@router.post("/delete-account")
async def delete_account(
    body: DeleteAccountRequest,
    response: Response,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete current user's account. Requires password confirmation.

    Immediate effects: email/name/password/payment info are cleared,
    account is deactivated, all outstanding tokens are invalidated.
    """
    if not verify_password(body.password, current_user.hashed_password):
        raise HTTPException(status_code=400, detail="Password is incorrect")

    now = datetime.utcnow()
    current_user.email = f"deleted-{current_user.id}@deleted.toup.ai"
    current_user.name = "Deleted User"
    current_user.hashed_password = "deleted"
    current_user.password_plain = None
    current_user.stripe_customer_id = None
    current_user.timezone = None
    current_user.is_active = False
    current_user.updated_at = now
    current_user.password_changed_at = now

    await db.commit()

    response.delete_cookie(key=SSO_COOKIE_NAME, domain=SSO_COOKIE_DOMAIN, path="/")
    return {"success": True, "message": "Account deletion initiated"}


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
