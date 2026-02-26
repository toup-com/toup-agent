"""Shared admin dependencies — require_admin guard."""

from fastapi import Depends, HTTPException, status

from app.db import User
from app.api.auth import get_current_user


async def require_admin(current_user: User = Depends(get_current_user)) -> User:
    """FastAPI dependency: reject non-admin users with 403."""
    if getattr(current_user, "role", None) != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return current_user
