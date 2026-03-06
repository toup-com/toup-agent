"""
Dashboard proxy routes — platform forwards all dashboard requests to the user's VPS agent.

Dashboard data is personal: tasks, inbox, docs, goals, alerts all live on the user's VPS.
The platform is a passthrough proxy only.
"""

import logging
from typing import Optional, Tuple

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import get_current_user
from app.db import get_db, User, AgentConfig

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Dashboard"])


# ── Agent proxy helpers (same pattern as workflows.py) ───────────────

async def _get_agent_proxy_info(
    user_id: str, db: AsyncSession
) -> Optional[Tuple[str, str]]:
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
    method: str = "GET", params: Optional[dict] = None,
    body: Optional[dict] = None,
):
    url = f"{agent_url}/api/{path}"
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            headers = {"X-Agent-Key": agent_api_key}
            if method == "GET":
                resp = await client.get(url, headers=headers, params=params or {})
            elif method == "POST":
                resp = await client.post(url, headers=headers, json=body or {})
            elif method == "DELETE":
                resp = await client.delete(url, headers=headers, params=params or {})
            else:
                return None
            if resp.status_code in (200, 201):
                return resp.json()
            logger.warning("Dashboard proxy %s %s returned %s", method, url, resp.status_code)
            return {"error": True, "status": resp.status_code, "detail": resp.text}
    except Exception as e:
        logger.warning("Dashboard proxy %s %s failed: %s", method, url, e)
    return None


def _require_agent(proxy_info):
    if not proxy_info:
        raise HTTPException(503, "Agent not deployed or not reachable.")


def _check_result(result):
    if result is None:
        raise HTTPException(502, "Agent unreachable")
    if isinstance(result, dict) and result.get("error"):
        raise HTTPException(result.get("status", 500), result.get("detail", "Agent error"))
    return result


# ── Routes ───────────────────────────────────────────────────────────

@router.get("/dashboard")
async def get_dashboard(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    proxy = await _get_agent_proxy_info(current_user.id, db)
    _require_agent(proxy)
    return _check_result(await _proxy(*proxy, path="dashboard"))


@router.get("/tasks")
async def list_tasks(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    proxy = await _get_agent_proxy_info(current_user.id, db)
    _require_agent(proxy)
    return _check_result(await _proxy(*proxy, path="tasks"))


@router.post("/tasks")
async def save_tasks(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    proxy = await _get_agent_proxy_info(current_user.id, db)
    _require_agent(proxy)
    body = await request.json()
    return _check_result(await _proxy(*proxy, path="tasks", method="POST", body=body))


@router.get("/status")
async def get_status(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    proxy = await _get_agent_proxy_info(current_user.id, db)
    _require_agent(proxy)
    return _check_result(await _proxy(*proxy, path="status"))


@router.get("/agents")
async def list_agents(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    proxy = await _get_agent_proxy_info(current_user.id, db)
    _require_agent(proxy)
    return _check_result(await _proxy(*proxy, path="agents"))


@router.get("/activity")
async def get_activity(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    proxy = await _get_agent_proxy_info(current_user.id, db)
    _require_agent(proxy)
    return _check_result(await _proxy(*proxy, path="activity"))


@router.get("/stats")
async def get_stats(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    proxy = await _get_agent_proxy_info(current_user.id, db)
    _require_agent(proxy)
    return _check_result(await _proxy(*proxy, path="stats"))


@router.get("/goals")
async def list_goals(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    proxy = await _get_agent_proxy_info(current_user.id, db)
    _require_agent(proxy)
    return _check_result(await _proxy(*proxy, path="goals"))


@router.get("/alerts")
async def list_alerts(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    proxy = await _get_agent_proxy_info(current_user.id, db)
    _require_agent(proxy)
    return _check_result(await _proxy(*proxy, path="alerts"))


@router.post("/alerts/generate")
async def generate_alerts(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    proxy = await _get_agent_proxy_info(current_user.id, db)
    _require_agent(proxy)
    return _check_result(await _proxy(*proxy, path="alerts/generate", method="POST"))


@router.post("/alerts/{alert_id}/dismiss")
async def dismiss_alert(
    alert_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    proxy = await _get_agent_proxy_info(current_user.id, db)
    _require_agent(proxy)
    return _check_result(await _proxy(*proxy, path=f"alerts/{alert_id}/dismiss", method="POST"))


@router.get("/inbox")
async def list_inbox(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    proxy = await _get_agent_proxy_info(current_user.id, db)
    _require_agent(proxy)
    return _check_result(await _proxy(*proxy, path="inbox"))


@router.post("/inbox")
async def create_inbox_item(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    proxy = await _get_agent_proxy_info(current_user.id, db)
    _require_agent(proxy)
    body = await request.json()
    return _check_result(await _proxy(*proxy, path="inbox", method="POST", body=body))


@router.delete("/inbox/{item_id}")
async def delete_inbox_item(
    item_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    proxy = await _get_agent_proxy_info(current_user.id, db)
    _require_agent(proxy)
    return _check_result(await _proxy(*proxy, path=f"inbox/{item_id}", method="DELETE"))


@router.get("/jobs")
async def list_jobs(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    proxy = await _get_agent_proxy_info(current_user.id, db)
    _require_agent(proxy)
    return _check_result(await _proxy(*proxy, path="jobs"))


@router.post("/delegate")
async def delegate_task(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    proxy = await _get_agent_proxy_info(current_user.id, db)
    _require_agent(proxy)
    body = await request.json()
    return _check_result(await _proxy(*proxy, path="delegate", method="POST", body=body))


@router.get("/docs")
async def list_docs(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    proxy = await _get_agent_proxy_info(current_user.id, db)
    _require_agent(proxy)
    return _check_result(await _proxy(*proxy, path="docs"))


@router.post("/docs")
async def save_doc(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    proxy = await _get_agent_proxy_info(current_user.id, db)
    _require_agent(proxy)
    body = await request.json()
    return _check_result(await _proxy(*proxy, path="docs", method="POST", body=body))


@router.delete("/docs")
async def delete_doc(
    name: str = Query(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    proxy = await _get_agent_proxy_info(current_user.id, db)
    _require_agent(proxy)
    return _check_result(await _proxy(*proxy, path="docs", method="DELETE", params={"name": name}))


# ── Memory file routes ──────────────────────────────────────────────

@router.get("/memory/search")
async def search_memory(
    q: str = Query(""),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    proxy = await _get_agent_proxy_info(current_user.id, db)
    _require_agent(proxy)
    return _check_result(await _proxy(*proxy, path="memory/search", params={"q": q}))


@router.get("/memory/file/{filepath:path}")
async def get_memory_file(
    filepath: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    proxy = await _get_agent_proxy_info(current_user.id, db)
    _require_agent(proxy)
    return _check_result(await _proxy(*proxy, path=f"memory/file/{filepath}"))


@router.get("/memory/{directory:path}")
async def list_memory_dir(
    directory: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    proxy = await _get_agent_proxy_info(current_user.id, db)
    _require_agent(proxy)
    return _check_result(await _proxy(*proxy, path=f"memory/{directory}"))


@router.post("/memory/entry")
async def add_memory_entry(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    proxy = await _get_agent_proxy_info(current_user.id, db)
    _require_agent(proxy)
    body = await request.json()
    return _check_result(await _proxy(*proxy, path="memory/entry", method="POST", body=body))


# ── Agent update (git pull + restart) ──────────────────────────────

async def _ssh_update_agent(user_id: str, db: AsyncSession) -> dict:
    """Fallback: update agent via SSH through Lambda proxy."""
    from app.services.ssh_deploy_service import _invoke_lambda

    result = await db.execute(
        select(AgentConfig).where(AgentConfig.user_id == user_id)
    )
    cfg = result.scalar_one_or_none()
    if not cfg or not cfg.ssh_host:
        return {"success": False, "error": "No SSH credentials stored for this agent"}

    try:
        resp = await _invoke_lambda({
            "action": "execute",
            "ssh_host": cfg.ssh_host,
            "ssh_port": cfg.ssh_port or 22,
            "ssh_user": cfg.ssh_user or "root",
            "ssh_password": cfg.ssh_password,
            "ssh_key": cfg.ssh_key,
            "commands": [
                "git config --global --add safe.directory /opt/toup-agent",
                "cd /opt/toup-agent && git remote get-url origin 2>/dev/null || (git init -b main && git remote add origin https://github.com/toup-com/toup-agent.git)",
                "cd /opt/toup-agent && git fetch --depth 1 origin main && git checkout -f origin/main",
                "cd /opt/toup-agent && source venv/bin/activate && pip install -q -r requirements.txt 2>/dev/null || true",
                "systemctl restart toup-agent",
            ],
        })
        if "error" in resp:
            return {"success": False, "error": resp["error"]}
        return {"success": True, "method": "ssh", "output": "Agent updated via SSH"}
    except Exception as e:
        logger.error("SSH update fallback failed: %s", e)
        return {"success": False, "error": f"SSH fallback failed: {e}"}


@router.post("/agent-update")
async def update_agent(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Trigger git pull + restart on the user's agent VPS.

    Strategy:
    1. Try the agent's own /agent/update endpoint (fast path)
    2. If that fails, fall back to SSH via Lambda (bulletproof path)
    """
    proxy = await _get_agent_proxy_info(current_user.id, db)
    _require_agent(proxy)
    agent_url, agent_api_key = proxy

    # 1. Try agent's self-update endpoint
    url = f"{agent_url}/agent/update"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, headers={"X-Agent-Key": agent_api_key})
            if resp.status_code == 200:
                data = resp.json()
                if data.get("success") is not False:
                    return data
                # Agent returned failure — fall through to SSH
                logger.warning("Agent self-update failed: %s", data)
    except Exception as e:
        logger.warning("Agent self-update unreachable: %s", e)

    # 2. Fallback: SSH via Lambda
    logger.info("Falling back to SSH update for user %s", current_user.id[:8])
    return await _ssh_update_agent(current_user.id, db)
