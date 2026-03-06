"""
Workflow routes — CRUD runs directly on platform DB (Supabase).
Execution (run) is proxied to the user's VPS agent.
"""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import select, delete as sa_delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import get_current_user
from app.db import get_db, User, AgentConfig, Workflow, Memory

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/workflows", tags=["Workflows"])


# ── Helpers ───────────────────────────────────────────────────────

def _wf_to_dict(wf: Workflow) -> dict:
    """Serialize a Workflow ORM object to dict."""
    return {
        "id": wf.id,
        "user_id": wf.user_id,
        "name": wf.name,
        "description": wf.description or "",
        "status": wf.status,
        "nodes_json": wf.nodes_json,
        "edges_json": wf.edges_json,
        "run_count": wf.run_count,
        "last_run_at": wf.last_run_at.isoformat() if wf.last_run_at else None,
        "created_at": wf.created_at.isoformat() if wf.created_at else None,
        "updated_at": wf.updated_at.isoformat() if wf.updated_at else None,
    }


async def _get_agent_proxy_info(
    user_id: str, db: AsyncSession
) -> Optional[Tuple[str, str]]:
    """Return (agent_url, agent_api_key) if the user has a remote agent."""
    result = await db.execute(
        select(AgentConfig.agent_url, AgentConfig.agent_api_key)
        .where(
            AgentConfig.user_id == user_id,
            AgentConfig.deploy_status == "active",
        )
    )
    row = result.first()
    if row and row.agent_url and row.agent_api_key:
        return (row.agent_url, row.agent_api_key)
    return None


# ── CRUD (direct DB) ─────────────────────────────────────────────

@router.get("")
@router.get("/")
async def list_workflows(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all workflows for the current user."""
    result = await db.execute(
        select(Workflow)
        .where(Workflow.user_id == current_user.id)
        .order_by(Workflow.updated_at.desc())
    )
    workflows = result.scalars().all()
    return [_wf_to_dict(wf) for wf in workflows]


@router.get("/{workflow_id}")
async def get_workflow(
    workflow_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get a single workflow by ID."""
    wf = (await db.execute(
        select(Workflow).where(
            Workflow.id == workflow_id,
            Workflow.user_id == current_user.id,
        )
    )).scalar_one_or_none()
    if not wf:
        raise HTTPException(404, "Workflow not found")
    return _wf_to_dict(wf)


@router.post("")
@router.post("/")
async def create_workflow(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a new workflow."""
    body = await request.json()
    wf = Workflow(
        id=str(uuid.uuid4()),
        user_id=current_user.id,
        name=body.get("name", "Untitled Workflow"),
        description=body.get("description", ""),
        status="draft",
        nodes_json=body.get("nodes_json", "[]"),
        edges_json=body.get("edges_json", "[]"),
        run_count=0,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    db.add(wf)
    await db.commit()
    await db.refresh(wf)
    return _wf_to_dict(wf)


@router.put("/{workflow_id}")
async def update_workflow(
    workflow_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update a workflow."""
    wf = (await db.execute(
        select(Workflow).where(
            Workflow.id == workflow_id,
            Workflow.user_id == current_user.id,
        )
    )).scalar_one_or_none()
    if not wf:
        raise HTTPException(404, "Workflow not found")

    body = await request.json()
    for field in ("name", "description", "status", "nodes_json", "edges_json"):
        if field in body:
            setattr(wf, field, body[field])
    wf.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(wf)
    return _wf_to_dict(wf)


@router.delete("/{workflow_id}")
async def delete_workflow(
    workflow_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a workflow."""
    result = await db.execute(
        sa_delete(Workflow).where(
            Workflow.id == workflow_id,
            Workflow.user_id == current_user.id,
        )
    )
    if result.rowcount == 0:
        raise HTTPException(404, "Workflow not found")
    await db.commit()
    return {"status": "deleted"}


@router.post("/{workflow_id}/duplicate")
async def duplicate_workflow(
    workflow_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Duplicate a workflow."""
    original = (await db.execute(
        select(Workflow).where(
            Workflow.id == workflow_id,
            Workflow.user_id == current_user.id,
        )
    )).scalar_one_or_none()
    if not original:
        raise HTTPException(404, "Workflow not found")

    copy = Workflow(
        id=str(uuid.uuid4()),
        user_id=current_user.id,
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
    return _wf_to_dict(copy)


@router.post("/{workflow_id}/run")
async def run_workflow(
    workflow_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Execute a workflow — proxied to VPS agent (needs agent runtime)."""
    proxy = await _get_agent_proxy_info(current_user.id, db)
    if not proxy:
        raise HTTPException(503, "Agent not deployed. Deploy your agent to run workflows.")
    agent_url, agent_api_key = proxy
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{agent_url}/api/workflows/{workflow_id}/run",
                headers={"X-Agent-Key": agent_api_key},
                json=body,
            )
            if resp.status_code in (200, 201):
                return resp.json()
            raise HTTPException(resp.status_code, resp.text)
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("Workflow run proxy failed: %s", e)
        raise HTTPException(502, "Agent unreachable")


# ── Template pack loading & matching ─────────────────────────────

_packs_cache: Optional[dict] = None


def _load_template_packs() -> dict:
    """Load all JSON template packs from the templates/ directory."""
    global _packs_cache
    if _packs_cache is not None:
        return _packs_cache

    packs = {}
    templates_dir = Path(__file__).resolve().parent.parent / "agent" / "workspace" / "templates"
    if not templates_dir.is_dir():
        logger.warning("Templates directory not found: %s", templates_dir)
        return packs

    for f in sorted(templates_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            packs[data.get("name", f.stem)] = data
        except Exception as e:
            logger.warning("Failed to load template pack %s: %s", f.name, e)

    _packs_cache = packs
    logger.info("Loaded %d template packs", len(packs))
    return packs


def _keyword_match(description: str, packs: dict) -> list[str]:
    """Match description to template packs using keyword scoring."""
    desc_lower = description.lower()
    scores: dict[str, int] = {}
    for name, pack in packs.items():
        score = sum(1 for kw in pack.get("keywords", []) if kw in desc_lower)
        if score > 0:
            scores[name] = score

    if not scores:
        return ["personal_productivity"]

    return sorted(scores, key=scores.get, reverse=True)[:3]


class GenerateRequest(BaseModel):
    description: str


@router.post("/generate")
async def generate_workspace(
    request: GenerateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Generate personalized workflows based on user description."""
    user_id = current_user.id

    # Guard: skip if workflows already exist
    existing = (await db.execute(
        select(Workflow.id).where(Workflow.user_id == user_id).limit(1)
    )).scalar_one_or_none()
    if existing:
        return {"status": "exists", "message": "Workflows already exist"}

    description = request.description.strip()
    if not description:
        raise HTTPException(400, "Description is required")

    # Also pull onboarding goals from memories (if available)
    try:
        goals_rows = (await db.execute(
            select(Memory.content).where(
                Memory.user_id == user_id,
                Memory.brain_type == "user",
                Memory.category == "goals",
            )
        )).scalars().all()
        if goals_rows:
            description = description + "\n" + "\n".join(goals_rows)
    except Exception:
        pass

    # Match template packs
    packs = _load_template_packs()
    if not packs:
        raise HTTPException(500, "No template packs available")

    matched = _keyword_match(description, packs)
    logger.info("Matched packs for %s: %s", user_id[:8], matched)

    # Create workflows in DB
    created = []
    for pack_name in matched:
        pack = packs.get(pack_name)
        if not pack:
            continue
        for wf_template in pack.get("workflows", []):
            wf = Workflow(
                id=str(uuid.uuid4()),
                user_id=user_id,
                name=wf_template["name"],
                description=wf_template.get("description", ""),
                status="draft",
                nodes_json=json.dumps(wf_template["nodes_json"]),
                edges_json=json.dumps(wf_template["edges_json"]),
                run_count=0,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
            db.add(wf)
            created.append({"id": wf.id, "name": wf.name})

    await db.commit()
    logger.info("Generated %d workflows for %s from packs: %s", len(created), user_id[:8], matched)

    return {
        "status": "created",
        "count": len(created),
        "packs": matched,
        "workflows": created,
    }
