"""Authentication endpoints"""

from fastapi import APIRouter, Depends, HTTPException, status, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from typing import Optional

from app.db import get_db
from app.schemas import UserCreate, UserLogin, UserResponse, Token
from app.services import (
    authenticate_user, create_user, get_user_by_email,
    get_user_by_id, create_access_token, decode_access_token
)
from app.config import settings

router = APIRouter(prefix="/auth", tags=["Authentication"])
security = HTTPBearer(auto_error=False)  # Don't auto-reject — we also check SSO cookie

# SSO cookie config
SSO_COOKIE_NAME = "hex_sso_token"
SSO_COOKIE_DOMAIN = ".toup.ai"  # Shared across all *.toup.ai subdomains
SSO_COOKIE_MAX_AGE = 60 * 60 * 24 * 7  # 1 week


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: AsyncSession = Depends(get_db)
):
    """Dependency to get the current authenticated user.
    Checks Bearer token first, then falls back to SSO cookie."""
    token = None
    user_id = None
    
    # 1. Try Bearer token
    if credentials and credentials.credentials:
        token = credentials.credentials
        user_id = decode_access_token(token)
    
    # 2. Fall back to SSO cookie
    if not user_id:
        cookie_token = request.cookies.get(SSO_COOKIE_NAME)
        if cookie_token:
            user_id = decode_access_token(cookie_token)
    
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user = await get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled",
        )
    
    return user


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db)
):
    """Register a new user"""
    # Check if email already exists
    existing = await get_user_by_email(db, user_data.email)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    user = await create_user(
        db,
        email=user_data.email,
        password=user_data.password,
        name=user_data.name
    )
    return user


@router.post("/login", response_model=Token)
async def login(
    credentials: UserLogin,
    response: Response,
    db: AsyncSession = Depends(get_db)
):
    """Login and get access token. Also sets SSO cookie for cross-domain auth.
    Accepts email (nariman@toup.ai) or bare username (nariman)."""
    login_id = credentials.email.strip()
    
    # Try exact match first
    user = await authenticate_user(db, login_id, credentials.password)
    
    # If no match and input doesn't contain @, try appending @toup.ai
    if not user and '@' not in login_id:
        user = await authenticate_user(db, f"{login_id}@toup.ai", credentials.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = create_access_token(user.id)
    
    # Set SSO cookie for cross-domain auth (shared across *.toup.ai)
    response.set_cookie(
        key=SSO_COOKIE_NAME,
        value=token,
        domain=SSO_COOKIE_DOMAIN,
        max_age=SSO_COOKIE_MAX_AGE,
        httponly=True,
        secure=True,
        samesite="none",  # Required for cross-domain cookies
        path="/",
    )
    
    return Token(access_token=token)


@router.get("/me", response_model=UserResponse)
async def get_me(
    current_user = Depends(get_current_user)
):
    """Get current user info"""
    return current_user


class ValidateRequest(BaseModel):
    token: Optional[str] = None

class ValidateResponse(BaseModel):
    valid: bool
    user_id: Optional[str] = None
    email: Optional[str] = None
    name: Optional[str] = None


@router.post("/validate", response_model=ValidateResponse)
async def validate_token(
    request: Request,
    body: Optional[ValidateRequest] = None,
    db: AsyncSession = Depends(get_db)
):
    """Validate a JWT token and return user info. Used by Dashboard for SSO.
    Accepts token in body, Authorization header, or SSO cookie."""
    token = None
    
    # 1. Check request body
    if body and body.token:
        token = body.token
    
    # 2. Check Authorization header
    if not token:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
    
    # 3. Check SSO cookie
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
    
    return ValidateResponse(
        valid=True,
        user_id=str(user.id),
        email=user.email,
        name=user.name,
    )


class SSOExchangeRequest(BaseModel):
    token: str


@router.post("/sso", response_model=Token)
async def sso_exchange(
    body: SSOExchangeRequest,
    response: Response,
    db: AsyncSession = Depends(get_db),
):
    """Exchange an SSO token (Brain JWT from the Dashboard) for a fresh JWT.
    The Dashboard passes the hex_sso_token cookie value as a query-param;
    this endpoint validates it and returns a Bearer token the SPA can store
    in localStorage."""
    user_id = decode_access_token(body.token)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired SSO token",
        )

    user = await get_user_by_id(db, user_id)
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
        )

    # Issue a fresh JWT
    new_token = create_access_token(user.id)

    # Also set the SSO cookie so subsequent navigations work
    response.set_cookie(
        key=SSO_COOKIE_NAME,
        value=new_token,
        domain=SSO_COOKIE_DOMAIN,
        max_age=SSO_COOKIE_MAX_AGE,
        httponly=True,
        secure=True,
        samesite="none",
        path="/",
    )

    return Token(access_token=new_token)


@router.post("/logout")
async def logout_user(response: Response):
    """Logout — clear SSO cookie"""
    response.delete_cookie(
        key=SSO_COOKIE_NAME,
        domain=SSO_COOKIE_DOMAIN,
        path="/",
    )
    return {"success": True}


# For demo mode: auto-create a demo user if none exists
@router.post("/demo", response_model=Token)
async def demo_login(db: AsyncSession = Depends(get_db)):
    """Create or login as demo user (for testing)"""
    demo_email = "demo@hexbrain.local"
    demo_password = "demo123456"
    
    user = await get_user_by_email(db, demo_email)
    if not user:
        user = await create_user(
            db,
            email=demo_email,
            password=demo_password,
            name="Demo User"
        )
    
    token = create_access_token(user.id)
    return Token(access_token=token)
