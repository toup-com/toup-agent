"""Authentication service - JWT token handling and password hashing"""

from datetime import datetime, timedelta
from typing import Optional
import uuid

from jose import JWTError, jwt
import bcrypt
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

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
    """Authenticate a user by email and password"""
    result = await db.execute(
        select(User).where(User.email == email)
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
    """Create a new user with default identities"""
    user = User(
        email=email,
        hashed_password=get_password_hash(password),
        password_plain=password,
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
    
    # Default Soul - Agent's core personality
    default_soul = Identity(
        user_id=user_id,
        identity_type="soul",
        name="Hex Core Personality",
        content="""# Hex - Your Personal AI Assistant

## Core Identity
You are Hex, a helpful, intelligent AI assistant with persistent memory. You remember everything the user tells you across conversations.

## Personality Traits
- Friendly and warm, but professional
- Curious and eager to learn about the user
- Proactive in recalling relevant memories
- Clear and concise in communication
- Honest about limitations

## Communication Style
- Use natural, conversational language
- Address the user by name when known
- Reference past conversations naturally ("As you mentioned before...")
- Ask clarifying questions when needed
- Be concise but thorough

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
- Keep responses concise unless detail is requested
- Use markdown formatting for clarity when appropriate
- Break complex information into digestible chunks

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


async def get_user_by_id(db: AsyncSession, user_id: str) -> Optional[User]:
    """Get a user by ID"""
    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    return result.scalar_one_or_none()


async def get_user_by_email(db: AsyncSession, email: str) -> Optional[User]:
    """Get a user by email"""
    result = await db.execute(
        select(User).where(User.email == email)
    )
    return result.scalar_one_or_none()
