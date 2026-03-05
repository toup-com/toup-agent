"""
Workflow proxy routes — platform forwards all CRUD to the user's VPS agent.

Workflows are personal data: they live exclusively on the user's VPS.
The platform is a passthrough proxy only.
"""

import logging
from typing import Optional, Tuple

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import get_current_user
from app.db import get_db, User, AgentConfig

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/workflows", tags=["Workflows"])


# ── Agent proxy helpers (same pattern as memories.py / sessions.py) ──

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


async def _proxy_workflow(
    agent_url: str, agent_api_key: str, path: str,
    method: str = "GET", params: Optional[dict] = None,
    body: Optional[dict] = None,
):
    """Proxy a workflow request to the VPS agent."""
    url = f"{agent_url}/api/workflows/{path}" if path else f"{agent_url}/api/workflows"
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            headers = {"X-Agent-Key": agent_api_key}
            if method == "GET":
                resp = await client.get(url, headers=headers, params=params or {})
            elif method == "POST":
                resp = await client.post(url, headers=headers, json=body or {})
            elif method == "PUT":
                resp = await client.put(url, headers=headers, json=body or {})
            elif method == "DELETE":
                resp = await client.delete(url, headers=headers)
            else:
                return None
            if resp.status_code in (200, 201):
                return resp.json()
            logger.warning("Agent workflows proxy %s %s returned %s", method, url, resp.status_code)
            return {"error": True, "status": resp.status_code, "detail": resp.text}
    except Exception as e:
        logger.warning("Agent workflows proxy %s %s failed: %s", method, url, e)
    return None


def _require_agent(proxy_info):
    """Raise 503 if agent is not connected."""
    if not proxy_info:
        raise HTTPException(
            status_code=503,
            detail="Agent not deployed or not reachable. Deploy your agent first.",
        )


# ── Routes ───────────────────────────────────────────────────────────

@router.get("")
@router.get("/")
async def list_workflows(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all workflows — proxied to VPS agent."""
    proxy = await _get_agent_proxy_info(current_user.id, db)
    _require_agent(proxy)
    result = await _proxy_workflow(*proxy, path="", method="GET")
    if result is None:
        raise HTTPException(502, "Agent unreachable")
    if isinstance(result, dict) and result.get("error"):
        raise HTTPException(result["status"], result.get("detail", "Agent error"))
    return result


@router.get("/{workflow_id}")
async def get_workflow(
    workflow_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get a single workflow by ID — proxied to VPS agent."""
    proxy = await _get_agent_proxy_info(current_user.id, db)
    _require_agent(proxy)
    result = await _proxy_workflow(*proxy, path=workflow_id, method="GET")
    if result is None:
        raise HTTPException(502, "Agent unreachable")
    if isinstance(result, dict) and result.get("error"):
        raise HTTPException(result["status"], result.get("detail", "Agent error"))
    return result


@router.post("")
@router.post("/")
async def create_workflow(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a new workflow — proxied to VPS agent."""
    proxy = await _get_agent_proxy_info(current_user.id, db)
    _require_agent(proxy)
    body = await request.json()
    result = await _proxy_workflow(*proxy, path="", method="POST", body=body)
    if result is None:
        raise HTTPException(502, "Agent unreachable")
    if isinstance(result, dict) and result.get("error"):
        raise HTTPException(result["status"], result.get("detail", "Agent error"))
    return result


@router.put("/{workflow_id}")
async def update_workflow(
    workflow_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update a workflow — proxied to VPS agent."""
    proxy = await _get_agent_proxy_info(current_user.id, db)
    _require_agent(proxy)
    body = await request.json()
    result = await _proxy_workflow(*proxy, path=workflow_id, method="PUT", body=body)
    if result is None:
        raise HTTPException(502, "Agent unreachable")
    if isinstance(result, dict) and result.get("error"):
        raise HTTPException(result["status"], result.get("detail", "Agent error"))
    return result


@router.delete("/{workflow_id}")
async def delete_workflow(
    workflow_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a workflow — proxied to VPS agent."""
    proxy = await _get_agent_proxy_info(current_user.id, db)
    _require_agent(proxy)
    result = await _proxy_workflow(*proxy, path=workflow_id, method="DELETE")
    if result is None:
        raise HTTPException(502, "Agent unreachable")
    if isinstance(result, dict) and result.get("error"):
        raise HTTPException(result["status"], result.get("detail", "Agent error"))
    return result


@router.post("/{workflow_id}/duplicate")
async def duplicate_workflow(
    workflow_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Duplicate a workflow — proxied to VPS agent."""
    proxy = await _get_agent_proxy_info(current_user.id, db)
    _require_agent(proxy)
    result = await _proxy_workflow(*proxy, path=f"{workflow_id}/duplicate", method="POST")
    if result is None:
        raise HTTPException(502, "Agent unreachable")
    if isinstance(result, dict) and result.get("error"):
        raise HTTPException(result["status"], result.get("detail", "Agent error"))
    return result
