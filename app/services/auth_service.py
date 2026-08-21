"""Authentication service - JWT token handling and password hashing"""

import asyncio
import secrets
from datetime import datetime, timedelta
from typing import Optional
import uuid

from jose import JWTError, jwt
import bcrypt
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.config import settings
from app.db.models import User, Identity

# Marks an account that has no usable password (OAuth / Sign-in-with-Apple
# users). The "!" prefix is not a valid bcrypt hash, so verify_password
# always returns False for it — Django uses the same convention. Lets us skip
# the ~50-100ms of event-loop-blocking bcrypt that hashing a throwaway random
# secret would cost on every OAuth signup.
_UNUSABLE_PW_PREFIX = "!"


def make_unusable_password() -> str:
    """Return an unusable password-hash sentinel for accounts that never set a
    password. Unguessable + unverifiable; costs ~0ms (no bcrypt)."""
    return _UNUSABLE_PW_PREFIX + secrets.token_urlsafe(32)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash using bcrypt directly.

    CPU-bound (~50-100ms) — offload via asyncio.to_thread when calling from an
    async path so it doesn't block the single-worker event loop. Returns False
    (never raises) for the unusable-password sentinel and for any malformed
    hash, so an OAuth account can't be logged into via /login.
    """
    if not hashed_password or hashed_password.startswith(_UNUSABLE_PW_PREFIX):
        return False
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8"),
        )
    except (ValueError, TypeError):
        return False


def get_password_hash(password: str) -> str:
    """Hash a password using bcrypt directly.

    CPU-bound (~50-100ms) — offload via asyncio.to_thread from async paths.
    """
    return bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt(),
    ).decode("utf-8")


def create_access_token(user_id: str) -> str:
    """Create a JWT access token"""
    expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)
    to_encode = {
        "sub": user_id,
        "exp": expire,
        "iat": datetime.utcnow(),
        "jti": str(uuid.uuid4()),  # Unique token ID
    }
    encoded_jwt = jwt.encode(
        to_encode,
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm
    )
    return encoded_jwt


def decode_platform_jwt(token: str) -> dict:
    """jwt.decode against {jwt_secret, jwt_secret_previous} (L-3).

    THE one decode path for platform-minted JWTs. Signing is always
    new-only (create_access_token above); during a rotation the platform
    sets JWT_SECRET=<new> + JWT_SECRET_PREVIOUS=<old> so tokens minted
    before the flip stay valid until they expire or the previous is
    cleared. Every revocation check that runs AFTER decode
    (password_changed_at vs iat, jti) sees the same payload shape
    regardless of which secret verified — old-secret tokens are NOT
    exempt from revocation, which is the regression trap a per-site
    single-secret decode would open (a site that catches JWTError and
    `pass`es would silently skip its check for old-secret tokens).

    Raises JWTError (the last one) when no configured secret verifies.
    An empty jwt_secret_previous is never tried.
    """
    secrets_to_try = [settings.jwt_secret]
    if settings.jwt_secret_previous:
        secrets_to_try.append(settings.jwt_secret_previous)
    last_err: Optional[JWTError] = None
    for _secret in secrets_to_try:
        try:
            return jwt.decode(
                token, _secret, algorithms=[settings.jwt_algorithm]
            )
        except JWTError as e:
            last_err = e
    assert last_err is not None
    raise last_err


def decode_access_token(token: str) -> Optional[str]:
    """Decode a JWT token and return the user ID.

    A full account credential has no ``scope`` claim (or ``scope == "full"``).
    Narrowly-scoped tokens — e.g. the app-preview token minted for the
    in-app preview iframe — carry a ``scope`` this general path must REJECT,
    so a scoped token that leaks out of an agent-authored preview page can
    never be replayed as the account bearer. This is the safe-by-construction
    half of the preview-token fix (docs/security/audit-2026.md, round 12);
    the preview/chat proxies validate scoped tokens via decode_preview_token.
    """
    try:
        payload = decode_platform_jwt(token)
        scope = payload.get("scope")
        if scope is not None and scope != "full":
            return None
        user_id: str = payload.get("sub")
        if user_id is None:
            return None
        return user_id
    except JWTError:
        return None


ARTIFACT_SCOPE = "artifact"


def create_artifact_token(user_id: str, slug: str) -> str:
    """Mint a short-lived token that fetches exactly ONE HTML artifact.

    Deliberately a DIFFERENT scope from ``app_preview`` rather than a reuse
    of it. An ``app_preview`` token is accepted by
    ``/api/apps/{id}/chat`` — i.e. it can talk to the user's agent. An
    artifact token must only ever pull static bytes, so a copy of it lifted
    out of an iframe URL (referrer, history, a screenshot) buys the holder
    one HTML file and nothing else. General auth rejects both.
    """
    expire = datetime.utcnow() + timedelta(
        minutes=settings.artifact_token_expire_minutes)
    to_encode = {
        "sub": user_id,
        "exp": expire,
        "iat": datetime.utcnow(),
        "jti": str(uuid.uuid4()),
        "scope": ARTIFACT_SCOPE,
        "artifact": str(slug),
    }
    return jwt.encode(
        to_encode, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_artifact_token(token: str, slug: str) -> Optional[str]:
    """Return the user_id iff ``token`` is an artifact token for ``slug``.

    Exact scope match and exact slug match. A full account token is NOT
    accepted — an artifact origin must never be reachable with the account
    credential, or the whole point of the separate origin is lost."""
    try:
        payload = jwt.decode(
            token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except JWTError:
        return None
    if payload.get("scope") != ARTIFACT_SCOPE:
        return None
    if str(payload.get("artifact")) != str(slug):
        return None
    return payload.get("sub") or None


def create_preview_token(user_id: str, app_id: str) -> str:
    """Mint a short-lived, single-app token for the in-app preview iframe.

    Deliberately NOT a full account credential. It authorizes ONLY the
    preview reverse-proxy + the agent-bridge chat for THIS one ``app_id``,
    and general auth (decode_access_token / get_current_user) rejects it via
    the ``scope`` claim. The preview page is agent-authored HTML that runs
    same-origin with the SPA, so a full JWT embedded there is exfiltratable
    by prompt-injected app code; a scoped token keeps the blast radius to one
    app for minutes instead of the whole account (round 12)."""
    expire = datetime.utcnow() + timedelta(
        minutes=settings.preview_token_expire_minutes)
    to_encode = {
        "sub": user_id,
        "exp": expire,
        "iat": datetime.utcnow(),
        "jti": str(uuid.uuid4()),
        "scope": "app_preview",
        "app_id": str(app_id),
    }
    return jwt.encode(
        to_encode, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_preview_token(token: str, app_id: str) -> Optional[str]:
    """Validate an app-preview-scoped token bound to a specific ``app_id``.

    Returns the user_id iff the token is ``scope == "app_preview"`` AND its
    ``app_id`` claim matches exactly. Used only by the preview/chat proxies —
    never by general auth. A full account token (no scope) is intentionally
    NOT accepted here; callers fall back to decode_access_token for those."""
    try:
        payload = decode_platform_jwt(token)
    except JWTError:
        return None
    if payload.get("scope") != "app_preview":
        return None
    if str(payload.get("app_id")) != str(app_id):
        return None
    user_id = payload.get("sub")
    return user_id or None


async def authenticate_user(
    db: AsyncSession,
    email: str,
    password: str
) -> Optional[User]:
    """Authenticate a user by email and password.

    Email lookup is case-insensitive — without this, signing up as
    `Arshia@toup.ai` and `arshia@toup.ai` produced two separate User
    rows (the 2026-05-01 incident: a fresh signup hit a 500 on /config,
    user retried with a different case, ended up with a duplicate
    account whose AgentConfig the original was never going to get).
    """
    email_norm = (email or "").strip().lower()
    result = await db.execute(
        select(User).where(func.lower(User.email) == email_norm)
    )
    user = result.scalar_one_or_none()

    if not user:
        return None
    # Offload bcrypt off the event loop — on the single worker an inline
    # checkpw would stall every other concurrent request for ~50-100ms.
    if not await asyncio.to_thread(verify_password, password, user.hashed_password):
        return None
    return user


async def create_user(
    db: AsyncSession,
    email: str,
    password: Optional[str] = None,
    name: Optional[str] = None,
    *,
    email_verified: bool = False,
    apple_sub: Optional[str] = None,
) -> User:
    """Create a new user with default identities.

    Email is canonicalised to lowercase before storage so the unique
    constraint on `users.email` actually enforces "one account per
    address" rather than "one per case spelling".

    `password=None` (OAuth / Sign-in-with-Apple) stores an unusable sentinel
    hash with no bcrypt cost; a real password is hashed off the event loop.
    `email_verified=True` stamps email_verified_at, and `apple_sub` stores the
    stable Apple identity, both folded into this single transaction so OAuth
    callers don't pay separate commit+refresh round-trips.
    """
    hashed = (
        await asyncio.to_thread(get_password_hash, password)
        if password
        else make_unusable_password()
    )
    user = User(
        email=(email or "").strip().lower(),
        hashed_password=hashed,
        name=name,
    )
    if email_verified:
        user.email_verified_at = datetime.utcnow()
    if apple_sub is not None:
        user.apple_sub = apple_sub
    db.add(user)
    await db.flush()  # Get user.id before creating identities

    # Create default identities for the new user
    await _seed_default_identities(db, user.id)

    # Pre-seed credit_balances with plan_id='free' so the provision
    # gate (managed_agents.py) and any other downstream check sees a
    # consistent invariant: every user has a credit_balances row from
    # signup onward. Without this, the row is lazy-created on first
    # credit charge — meaning fresh users provision without it, which
    # used to mis-route them through the paid-awaiting-Stripe path.
    #
    # Guarded: in CI / dev the subscription_plans.free row may not be
    # seeded (alembic 053 doesn't run in init_db()), so we swallow the
    # error rather than break signup. Production has the row from mig
    # 053.
    try:
        from app.services.credit_service import CreditService
        await CreditService().get_or_create_balance(db, user.id)
    except Exception:
        import logging
        logging.getLogger(__name__).warning(
            "[create_user] credit_balances seed skipped for user %s",
            str(user.id)[:8], exc_info=True,
        )

    await db.commit()
    await db.refresh(user)
    return user


async def _seed_default_identities(db: AsyncSession, user_id: str) -> None:
    """Create default identity documents for a new user"""
    
    # Default Soul - Agent's core personality (name is set during onboarding)
    default_soul = Identity(
        user_id=user_id,
        identity_type="soul",
        name="Agent Core Personality",
        content="""# Your Core Identity

## Who You Are
You are a helpful, intelligent AI assistant with persistent memory. You remember everything the user tells you across conversations.
Your name will be chosen by the user during onboarding — if you don't know your name yet, ask the user what they'd like to call you.

## Personality Traits
- Friendly and warm, but professional
- Curious and eager to learn about the user
- Proactive in recalling relevant memories
- Clear and concise in communication
- Honest about limitations

## Communication Style
- Use natural, conversational language — talk like a person, not a document
- Address the user by name when known
- Reference past conversations naturally ("As you mentioned before...")
- Ask clarifying questions when needed
- Be concise — say what's needed, skip the filler
- Default to short prose answers; only use structured formatting when it genuinely helps

## Key Behaviors
- Always check memories for relevant context before responding
- Proactively surface useful information from memory
- Remember user preferences and adapt accordingly
- Build on previous conversations to provide continuity""",
        priority=100,
        is_active=True
    )
    
    # Default Agent Instructions
    default_instructions = Identity(
        user_id=user_id,
        identity_type="agent_instructions",
        name="Default Behavioral Guidelines",
        content="""# Behavioral Guidelines

## Memory Usage
- When the user shares information about themselves, acknowledge it
- When memories are retrieved, use them naturally in responses
- Don't explicitly say "According to my memories..." - integrate naturally
- If unsure about recalled information, ask for confirmation

## Response Format
- Be conversational and concise by default — answer like a knowledgeable friend, not a textbook
- For simple questions, give short direct answers in natural prose (1-3 sentences)
- Only use headings, bullet lists, and structured formatting when the content genuinely requires it (step-by-step instructions, comparisons, technical breakdowns)
- Never over-structure: no headings for single-topic answers, no bullet lists for 2 items, no extra "note" or "warning" sections unless truly needed
- Match response length to question complexity — short questions deserve short answers
- Use markdown formatting sparingly: bold for emphasis, code blocks for code, lists only when listing 3+ items

## Safety Guidelines
- Never share user data with third parties
- Respect user privacy and confidentiality
- Decline requests for harmful or unethical actions
- Acknowledge when you don't know something""",
        priority=90,
        is_active=True
    )
    
    db.add(default_soul)
    db.add(default_instructions)


async def change_user_password(db: AsyncSession, user: User, new_password: str) -> User:
    """Change a user's password and invalidate all existing tokens."""
    user.hashed_password = await asyncio.to_thread(get_password_hash, new_password)
    user.password_changed_at = datetime.utcnow()
    await db.commit()
    await db.refresh(user)
    return user


async def get_user_by_id(db: AsyncSession, user_id: str) -> Optional[User]:
    """Get a user by ID"""
    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    return result.scalar_one_or_none()


async def get_user_by_email(db: AsyncSession, email: str) -> Optional[User]:
    """Get a user by email (case-insensitive)."""
    email_norm = (email or "").strip().lower()
    result = await db.execute(
        select(User).where(func.lower(User.email) == email_norm)
    )
    return result.scalar_one_or_none()
