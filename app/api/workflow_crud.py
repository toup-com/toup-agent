"""
Workflow CRUD API — runs on the VPS agent (data owner).

Provides full CRUD for workflows stored in the local toup_brain database.
The platform proxies all workflow requests here via X-Agent-Key auth.
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.db.models import Workflow
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/workflows", tags=["Workflows"])


# ── Request / Response schemas ───────────────────────────────────────

class WorkflowCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None


class WorkflowUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    nodes_json: Optional[str] = None
    edges_json: Optional[str] = None


class WorkflowOut(BaseModel):
    id: str
    name: str
    description: Optional[str]
    status: str
    nodes_json: str
    edges_json: str
    run_count: int
    last_run_at: Optional[str]
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True


def _to_out(w: Workflow) -> dict:
    return {
        "id": w.id,
        "name": w.name,
        "description": w.description,
        "status": w.status,
        "nodes_json": w.nodes_json,
        "edges_json": w.edges_json,
        "run_count": w.run_count,
        "last_run_at": w.last_run_at.isoformat() if w.last_run_at else None,
        "created_at": w.created_at.isoformat() if w.created_at else None,
        "updated_at": w.updated_at.isoformat() if w.updated_at else None,
    }


# ── Routes ───────────────────────────────────────────────────────────

@router.get("")
@router.get("/")
async def list_workflows(db: AsyncSession = Depends(get_db)):
    """List all workflows for the agent owner."""
    user_id = settings.user_id
    query = select(Workflow).order_by(Workflow.updated_at.desc())
    if user_id:
        query = query.where(Workflow.user_id == user_id)
    result = await db.execute(query)
    workflows = result.scalars().all()
    return [_to_out(w) for w in workflows]


@router.get("/{workflow_id}")
async def get_workflow(workflow_id: str, db: AsyncSession = Depends(get_db)):
    """Get a single workflow by ID."""
    result = await db.execute(select(Workflow).where(Workflow.id == workflow_id))
    w = result.scalar_one_or_none()
    if not w:
        raise HTTPException(404, "Workflow not found")
    return _to_out(w)


@router.post("")
@router.post("/")
async def create_workflow(req: WorkflowCreate, db: AsyncSession = Depends(get_db)):
    """Create a new workflow."""
    w = Workflow(
        id=str(uuid.uuid4()),
        user_id=settings.user_id,
        name=req.name,
        description=req.description,
        status="draft",
        nodes_json="[]",
        edges_json="[]",
        run_count=0,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    db.add(w)
    await db.commit()
    await db.refresh(w)
    return _to_out(w)


@router.put("/{workflow_id}")
async def update_workflow(
    workflow_id: str, req: WorkflowUpdate, db: AsyncSession = Depends(get_db)
):
    """Update a workflow (name, description, status, nodes, edges)."""
    result = await db.execute(select(Workflow).where(Workflow.id == workflow_id))
    w = result.scalar_one_or_none()
    if not w:
        raise HTTPException(404, "Workflow not found")

    if req.name is not None:
        w.name = req.name
    if req.description is not None:
        w.description = req.description
    if req.status is not None:
        w.status = req.status
    if req.nodes_json is not None:
        w.nodes_json = req.nodes_json
    if req.edges_json is not None:
        w.edges_json = req.edges_json
    w.updated_at = datetime.utcnow()

    await db.commit()
    await db.refresh(w)
    return _to_out(w)


@router.delete("/{workflow_id}")
async def delete_workflow(workflow_id: str, db: AsyncSession = Depends(get_db)):
    """Delete a workflow."""
    result = await db.execute(select(Workflow).where(Workflow.id == workflow_id))
    w = result.scalar_one_or_none()
    if not w:
        raise HTTPException(404, "Workflow not found")
    await db.delete(w)
    await db.commit()
    return {"status": "deleted", "id": workflow_id}


@router.post("/{workflow_id}/duplicate")
async def duplicate_workflow(workflow_id: str, db: AsyncSession = Depends(get_db)):
    """Duplicate a workflow."""
    result = await db.execute(select(Workflow).where(Workflow.id == workflow_id))
    original = result.scalar_one_or_none()
    if not original:
        raise HTTPException(404, "Workflow not found")

    copy = Workflow(
        id=str(uuid.uuid4()),
        user_id=original.user_id,
        name=f"{original.name} (Copy)",
        description=original.description,
        status="draft",
        nodes_json=original.nodes_json,
        edges_json=original.edges_json,
        run_count=0,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    db.add(copy)
    await db.commit()
    await db.refresh(copy)
    return _to_out(copy)


# ── Execution ────────────────────────────────────────────────────

# Module-level reference set by agent_main.py lifespan
_workflow_engine = None


def set_workflow_engine(engine):
    """Set the workflow engine reference (called from agent_main.py)."""
    global _workflow_engine
    _workflow_engine = engine


@router.post("/{workflow_id}/run")
async def run_workflow(
    workflow_id: str,
    req: Optional[dict] = None,
    db: AsyncSession = Depends(get_db),
):
    """Execute a workflow and return the results."""
    result = await db.execute(select(Workflow).where(Workflow.id == workflow_id))
    w = result.scalar_one_or_none()
    if not w:
        raise HTTPException(404, "Workflow not found")

    if not _workflow_engine:
        raise HTTPException(503, "Workflow engine not initialized")

    trigger_data = req or {}
    ctx = await _workflow_engine.execute(w, trigger_data=trigger_data, user_id=settings.user_id)
    return ctx.to_dict()


# ── Workspace Generation (post-onboarding) ──────────────────────

@router.post("/generate-from-onboarding")
async def generate_workspace_from_onboarding(
    background_tasks: BackgroundTasks,
):
    """Auto-generate personalized workflows from onboarding goals. Runs in background."""
    from app.agent.workspace.workspace_generator import generate_workspace
    background_tasks.add_task(generate_workspace, settings.user_id)
    return {"status": "generating", "message": "Workspace generation started"}
