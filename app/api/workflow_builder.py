"""
App Builder — Platform proxy that forwards builder chat to the user's VPS agent.

The user's agent has their API keys and uses their strongest model.
Platform just authenticates the user and proxies the SSE stream.
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Optional, Tuple

import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import get_current_user
from app.config import settings
from app.db import get_db, User, Workflow, AgentConfig

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/workflows/builder", tags=["App Builder"])


async def _get_agent_info(user_id: str, db: AsyncSession) -> Optional[Tuple[str, str]]:
    """Return (agent_url, agent_api_key) if the user has a deployed agent."""
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


class BuilderMessage(BaseModel):
    messages: list[dict]


@router.post("/chat")
async def builder_chat(
    request: BuilderMessage,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Proxy builder chat to the user's VPS agent (SSE stream)."""
    agent = await _get_agent_info(current_user.id, db)
    if not agent:
        raise HTTPException(503, "Agent not deployed. Deploy your agent first to use the App Builder.")

    agent_url, agent_api_key = agent

    async def proxy_stream():
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
                async with client.stream(
                    "POST",
                    f"{agent_url}/api/workflows/builder/chat",
                    headers={"X-Agent-Key": agent_api_key, "Content-Type": "application/json"},
                    json={"messages": request.messages},
                ) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        error_msg = f"Agent returned {resp.status_code}: {body.decode()[:200]}"
                        yield f"data: {json.dumps({'type': 'error', 'content': error_msg})}\n\n"
                        return
                    async for line in resp.aiter_lines():
                        if line.startswith("data: "):
                            yield f"{line}\n\n"
        except httpx.ConnectError:
            yield f"data: {json.dumps({'type': 'error', 'content': 'Could not connect to your agent. Is it running?'})}\n\n"
        except Exception as e:
            logger.error("Builder proxy error: %s", e)
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(
        proxy_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Personalized suggestions ──────────────────────────────────────

@router.get("/suggestions")
async def builder_suggestions(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get personalized app suggestions from the user's VPS agent."""
    agent = await _get_agent_info(current_user.id, db)
    if not agent:
        return {"suggestions": [
            "Project management dashboard with kanban board",
            "Personal CRM to track clients and deals",
            "Habit tracker with streaks and analytics",
            "E-commerce admin panel with order management",
        ], "personalized": False}

    agent_url, agent_api_key = agent
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(45.0, connect=5.0)) as client:
            resp = await client.get(
                f"{agent_url}/api/workflows/builder/suggestions",
                headers={"X-Agent-Key": agent_api_key},
            )
            if resp.status_code == 200:
                data = resp.json()
                logger.info("Builder suggestions from agent: personalized=%s", data.get("personalized"))
                return data
            logger.warning("Builder suggestions agent returned %s: %s", resp.status_code, resp.text[:200])
    except Exception as e:
        logger.warning("Builder suggestions proxy failed: %s", e)

    return {"suggestions": [
        "Project management dashboard with kanban board",
        "Personal CRM to track clients and deals",
        "Habit tracker with streaks and analytics",
        "E-commerce admin panel with order management",
    ], "personalized": False}


# ── Save endpoints ────────────────────────────────────────────────

class SaveWorkflowsRequest(BaseModel):
    workflows: list[dict]

@router.post("/save")
async def save_built_workflows(
    request: SaveWorkflowsRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Save workflows generated by the builder (legacy)."""
    created = []
    for wf_data in request.workflows:
        wf = Workflow(
            id=str(uuid.uuid4()),
            user_id=current_user.id,
            name=wf_data.get("name", "Untitled Workflow"),
            description=wf_data.get("description", ""),
            status="draft",
            nodes_json=json.dumps(wf_data.get("nodes", [])),
            edges_json=json.dumps(wf_data.get("edges", [])),
            run_count=0,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        db.add(wf)
        created.append({"id": wf.id, "name": wf.name})

    await db.commit()
    return {"status": "saved", "count": len(created), "workflows": created}


class SaveAppRequest(BaseModel):
    name: str
    files: Optional[dict] = None         # Multi-file: {"/App.jsx": "code", ...}
    dependencies: Optional[dict] = None  # {"lucide-react": "latest", ...}
    code: Optional[str] = None           # Legacy single-file fallback

@router.post("/save-app")
async def save_built_app(
    request: SaveAppRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Save an app generated by the builder — proxied to user's VPS agent."""
    agent = await _get_agent_info(current_user.id, db)
    if not agent:
        raise HTTPException(503, "Agent not deployed. Deploy your agent first.")

    agent_url, agent_api_key = agent

    # Build node data — supports both multi-file and legacy single-file
    node_data: dict = {
        "label": request.name,
        "templateType": "app_component",
        "config": {},
    }
    if request.files:
        node_data["files"] = request.files
        if request.dependencies:
            node_data["dependencies"] = request.dependencies
    elif request.code:
        node_data["code"] = request.code

    wf_payload = {
        "name": request.name,
        "description": "Generated app",
        "nodes_json": json.dumps([{
            "id": "app",
            "type": "app_component",
            "position": {"x": 0, "y": 0},
            "data": node_data,
        }]),
        "edges_json": "[]",
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{agent_url}/api/workflows/",
                headers={"X-Agent-Key": agent_api_key, "Content-Type": "application/json"},
                json=wf_payload,
            )
            if resp.status_code in (200, 201):
                data = resp.json()
                return {"status": "saved", "id": data.get("id"), "name": data.get("name")}
            raise HTTPException(resp.status_code, resp.text)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Save app proxy failed: %s", e)
        raise HTTPException(502, f"Agent unreachable: {e}")
