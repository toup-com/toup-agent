"""Authentication service - JWT token handling and password hashing"""

from datetime import datetime, timedelta
from typing import Optional
import uuid

from jose import JWTError, jwt
import bcrypt
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.config import settings
from app.db.models import User, Identity


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash using bcrypt directly."""
    return bcrypt.checkpw(
        plain_password.encode("utf-8"),
        hashed_password.encode("utf-8"),
    )


def get_password_hash(password: str) -> str:
    """Hash a password using bcrypt directly."""
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


def decode_access_token(token: str) -> Optional[str]:
    """Decode a JWT token and return the user ID"""
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm]
        )
        user_id: str = payload.get("sub")
        if user_id is None:
            return None
        return user_id
    except JWTError:
        return None


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
    if not verify_password(password, user.hashed_password):
        return None
    return user


async def create_user(
    db: AsyncSession,
    email: str,
    password: str,
    name: Optional[str] = None
) -> User:
    """Create a new user with default identities.

    Email is canonicalised to lowercase before storage so the unique
    constraint on `users.email` actually enforces "one account per
    address" rather than "one per case spelling".
    """
    user = User(
        email=(email or "").strip().lower(),
        hashed_password=get_password_hash(password),
        name=name,
    )
    db.add(user)
    await db.flush()  # Get user.id before creating identities
    
    # Create default identities for the new user
    await _seed_default_identities(db, user.id)
    
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
    user.hashed_password = get_password_hash(new_password)
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
