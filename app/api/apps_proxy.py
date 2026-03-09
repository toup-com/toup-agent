"""
Apps proxy routes — platform forwards apps/jobs requests to the user's VPS agent.

App data (built apps, build jobs) lives on the user's VPS.
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
from app.db import get_db, AgentConfig

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/apps", tags=["Apps Proxy"])


# ── Agent proxy helpers ─────────────────────────────────────

async def _get_agent(user_id: str, db: AsyncSession) -> Optional[Tuple[str, str]]:
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


async def _proxy(
    agent_url: str, agent_api_key: str, path: str,
    method: str = "GET", body: Optional[dict] = None,
    timeout: float = 30.0,
):
    url = f"{agent_url}/api/apps/{path}"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            headers = {"X-Agent-Key": agent_api_key}
            if method == "GET":
                resp = await client.get(url, headers=headers)
            elif method == "POST":
                resp = await client.post(url, headers=headers, json=body or {})
            elif method == "DELETE":
                resp = await client.delete(url, headers=headers)
            else:
                return None
            return JSONResponse(content=resp.json(), status_code=resp.status_code)
    except Exception as e:
        logger.warning("Apps proxy %s %s failed: %s", method, url, e)
        raise HTTPException(502, "Agent unreachable")


def _require(info):
    if not info:
        raise HTTPException(503, "Agent not deployed or not reachable.")
    return info


# ── App endpoints ───────────────────────────────────────────

@router.get("/")
async def list_apps(current_user=Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    agent_url, key = _require(await _get_agent(current_user.id, db))
    return await _proxy(agent_url, key, "")


@router.get("/jobs/")
async def list_jobs(current_user=Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    agent_url, key = _require(await _get_agent(current_user.id, db))
    return await _proxy(agent_url, key, "jobs/")


@router.get("/jobs/{job_id}")
async def get_job(job_id: str, current_user=Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    agent_url, key = _require(await _get_agent(current_user.id, db))
    return await _proxy(agent_url, key, f"jobs/{job_id}")


@router.get("/{app_id}")
async def get_app(app_id: str, current_user=Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    agent_url, key = _require(await _get_agent(current_user.id, db))
    return await _proxy(agent_url, key, app_id)


@router.post("/{app_id}/start")
async def start_app(app_id: str, current_user=Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    agent_url, key = _require(await _get_agent(current_user.id, db))
    return await _proxy(agent_url, key, f"{app_id}/start", method="POST", timeout=60.0)


@router.post("/{app_id}/stop")
async def stop_app(app_id: str, current_user=Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    agent_url, key = _require(await _get_agent(current_user.id, db))
    return await _proxy(agent_url, key, f"{app_id}/stop", method="POST")


@router.post("/{app_id}/publish-web")
async def publish_web(app_id: str, request: Request, current_user=Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    agent_url, key = _require(await _get_agent(current_user.id, db))
    body = None
    try:
        body = await request.json()
    except Exception:
        pass
    return await _proxy(agent_url, key, f"{app_id}/publish-web", method="POST", body=body, timeout=120.0)


@router.post("/{app_id}/push-github")
async def push_github(app_id: str, current_user=Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    agent_url, key = _require(await _get_agent(current_user.id, db))
    return await _proxy(agent_url, key, f"{app_id}/push-github", method="POST", timeout=60.0)


@router.delete("/{app_id}")
async def delete_app(app_id: str, current_user=Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    agent_url, key = _require(await _get_agent(current_user.id, db))
    return await _proxy(agent_url, key, app_id, method="DELETE")
