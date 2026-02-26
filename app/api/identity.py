"""
Identity API - CRUD operations for agent identity documents

Identity documents define how the agent behaves:
- SOUL: Core personality, values, communication style
- USER_PROFILE: Information about the user being served
- AGENT_INSTRUCTIONS: Specific behavioral instructions
- TOOLS: Available tools/capabilities description
- CONTEXT: Dynamic context (e.g., current project)
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from typing import List, Optional

from app.db import get_db, Identity, User
from app.db.models import IdentityType
from app.schemas import (
    IdentityCreate, IdentityUpdate, IdentityResponse, IdentityListResponse
)
from app.api.auth import get_current_user

router = APIRouter(prefix="/identity", tags=["identity"])


@router.post("", response_model=IdentityResponse, status_code=status.HTTP_201_CREATED)
async def create_identity(
    request: IdentityCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new identity document.
    
    Identity documents are core to how the agent behaves. Each type serves a purpose:
    - SOUL: The agent's core personality and values
    - USER_PROFILE: What the agent knows about the user
    - AGENT_INSTRUCTIONS: Specific behavioral rules
    - TOOLS: Available capabilities
    - CONTEXT: Dynamic situational context
    """
    # Create identity record
    identity = Identity(
        user_id=current_user.id,
        identity_type=request.identity_type.value,
        name=request.name,
        content=request.content,
        priority=request.priority,
        is_active=request.is_active
    )
    
    db.add(identity)
    await db.commit()
    await db.refresh(identity)
    
    return identity


@router.get("", response_model=IdentityListResponse)
async def list_identities(
    identity_type: Optional[str] = None,
    active_only: bool = True,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    List all identity documents for the current user.
    
    Optionally filter by type or active status.
    Returns identities ordered by priority (highest first).
    """
    # Build query
    conditions = [Identity.user_id == current_user.id]
    
    if identity_type:
        conditions.append(Identity.identity_type == identity_type)
    
    if active_only:
        conditions.append(Identity.is_active == True)
    
    query = (
        select(Identity)
        .where(and_(*conditions))
        .order_by(Identity.priority.desc(), Identity.created_at.asc())
    )
    
    result = await db.execute(query)
    identities = result.scalars().all()
    
    return IdentityListResponse(
        identities=identities,
        total_count=len(identities)
    )


@router.get("/{identity_id}", response_model=IdentityResponse)
async def get_identity(
    identity_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get a specific identity document by ID."""
    query = select(Identity).where(
        and_(
            Identity.id == identity_id,
            Identity.user_id == current_user.id
        )
    )
    
    result = await db.execute(query)
    identity = result.scalar_one_or_none()
    
    if not identity:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Identity not found"
        )
    
    return identity


@router.put("/{identity_id}", response_model=IdentityResponse)
async def update_identity(
    identity_id: str,
    request: IdentityUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Update an existing identity document.
    
    Only provided fields will be updated.
    """
    query = select(Identity).where(
        and_(
            Identity.id == identity_id,
            Identity.user_id == current_user.id
        )
    )
    
    result = await db.execute(query)
    identity = result.scalar_one_or_none()
    
    if not identity:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Identity not found"
        )
    
    # Update only provided fields
    update_data = request.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(identity, field, value)
    
    await db.commit()
    await db.refresh(identity)
    
    return identity


@router.delete("/{identity_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_identity(
    identity_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete an identity document."""
    query = select(Identity).where(
        and_(
            Identity.id == identity_id,
            Identity.user_id == current_user.id
        )
    )
    
    result = await db.execute(query)
    identity = result.scalar_one_or_none()
    
    if not identity:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Identity not found"
        )
    
    await db.delete(identity)
    await db.commit()


@router.post("/{identity_id}/activate", response_model=IdentityResponse)
async def activate_identity(
    identity_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Activate an identity document (set is_active=True)."""
    query = select(Identity).where(
        and_(
            Identity.id == identity_id,
            Identity.user_id == current_user.id
        )
    )
    
    result = await db.execute(query)
    identity = result.scalar_one_or_none()
    
    if not identity:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Identity not found"
        )
    
    identity.is_active = True
    await db.commit()
    await db.refresh(identity)
    
    return identity


@router.post("/{identity_id}/deactivate", response_model=IdentityResponse)
async def deactivate_identity(
    identity_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Deactivate an identity document (set is_active=False)."""
    query = select(Identity).where(
        and_(
            Identity.id == identity_id,
            Identity.user_id == current_user.id
        )
    )
    
    result = await db.execute(query)
    identity = result.scalar_one_or_none()
    
    if not identity:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Identity not found"
        )
    
    identity.is_active = False
    await db.commit()
    await db.refresh(identity)
    
    return identity


@router.get("/types/list", response_model=List[str])
async def list_identity_types():
    """List all available identity types."""
    return [t.value for t in IdentityType]
