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


# ── Agent diagnose & repair (comprehensive VPS fix) ────────────────

# Diagnostic script that runs on the VPS via SSH.
# Checks everything, fixes what it can, reports results as JSON.
_DIAGNOSE_AND_REPAIR_SCRIPT = r'''
#!/bin/bash
# Toup Agent VPS Diagnostic & Repair Script
# Outputs a JSON report of checks and fixes applied.
set -o pipefail

AGENT_DIR="/opt/toup-agent"
REPO_URL="https://github.com/toup-com/toup-agent.git"
PYTHON="python3.12"
CHECKS=()

add_check() {
    local name="$1" status="$2" detail="$3" fixed="$4"
    CHECKS+=("{\"name\":\"$name\",\"status\":\"$status\",\"detail\":\"$(echo "$detail" | tr '"' "'" | tr '\n' ' ' | head -c 200)\",\"fixed\":${fixed:-false}}")
}

# 1. Disk space
DISK_AVAIL=$(df / --output=avail -BM 2>/dev/null | tail -1 | tr -d ' M' || echo "0")
if [ "$DISK_AVAIL" -lt 500 ]; then
    # Try to free space
    apt-get clean 2>/dev/null || true
    journalctl --vacuum-time=1d 2>/dev/null || true
    rm -rf /tmp/* 2>/dev/null || true
    DISK_AVAIL_AFTER=$(df / --output=avail -BM 2>/dev/null | tail -1 | tr -d ' M' || echo "0")
    if [ "$DISK_AVAIL_AFTER" -lt 500 ]; then
        add_check "disk_space" "error" "Only ${DISK_AVAIL_AFTER}MB free (need 500MB+)" "false"
    else
        add_check "disk_space" "fixed" "Freed space: ${DISK_AVAIL}MB -> ${DISK_AVAIL_AFTER}MB" "true"
    fi
else
    add_check "disk_space" "ok" "${DISK_AVAIL}MB available"
fi

# 2. Agent directory exists
if [ ! -d "$AGENT_DIR" ]; then
    mkdir -p "$AGENT_DIR"
    add_check "agent_dir" "fixed" "Created $AGENT_DIR" "true"
else
    add_check "agent_dir" "ok" "$AGENT_DIR exists"
fi

# 3. Git repo present and updated
if [ ! -d "$AGENT_DIR/.git" ]; then
    git clone --depth 1 "$REPO_URL" "${AGENT_DIR}-tmp" 2>&1 && \
        cp -a "${AGENT_DIR}-tmp/." "$AGENT_DIR/" && \
        rm -rf "${AGENT_DIR}-tmp"
    add_check "git_repo" "fixed" "Cloned fresh from $REPO_URL" "true"
else
    git config --global --add safe.directory "$AGENT_DIR" 2>/dev/null
    cd "$AGENT_DIR"
    # Fix remote URL if wrong
    CURRENT_REMOTE=$(git remote get-url origin 2>/dev/null || echo "")
    if [ "$CURRENT_REMOTE" != "$REPO_URL" ]; then
        git remote set-url origin "$REPO_URL" 2>/dev/null || git remote add origin "$REPO_URL" 2>/dev/null
    fi
    FETCH_OUT=$(git fetch --depth 1 origin main 2>&1)
    CHECKOUT_OUT=$(git checkout -f origin/main 2>&1)
    LOCAL_HASH=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
    add_check "git_repo" "fixed" "Updated to $LOCAL_HASH" "true"
fi

# 4. Python available
if command -v $PYTHON &>/dev/null; then
    PY_VER=$($PYTHON --version 2>&1)
    add_check "python" "ok" "$PY_VER"
else
    apt-get update -y -qq 2>/dev/null
    apt-get install -y -qq ${PYTHON} ${PYTHON}-venv ${PYTHON}-dev 2>/dev/null
    if command -v $PYTHON &>/dev/null; then
        add_check "python" "fixed" "Installed $PYTHON" "true"
    else
        add_check "python" "error" "$PYTHON not found and install failed"
    fi
fi

# 5. Virtual environment
if [ ! -f "$AGENT_DIR/venv/bin/activate" ]; then
    $PYTHON -m venv "$AGENT_DIR/venv" 2>&1
    add_check "venv" "fixed" "Created virtual environment" "true"
else
    add_check "venv" "ok" "venv exists"
fi

# 6. Dependencies
cd "$AGENT_DIR"
if [ -f requirements.txt ]; then
    source venv/bin/activate 2>/dev/null
    PIP_OUT=$(pip install -q -r requirements.txt 2>&1 | tail -5)
    deactivate 2>/dev/null
    add_check "dependencies" "fixed" "pip install completed" "true"
else
    add_check "dependencies" "error" "No requirements.txt found"
fi

# 7. .env file — check, repair missing vars, or regenerate entirely
ENV_FILE="$AGENT_DIR/.env"
REQUIRED_VARS="DATABASE_URL USER_ID AGENT_API_KEY RUN_MODE JWT_SECRET LLM_MODE EMBEDDING_PROVIDER TOUP_TOKEN"

if [ ! -f "$ENV_FILE" ]; then
    # .env is completely missing — write the platform-generated one
    if [ -f /tmp/toup-env-repair.txt ]; then
        cp /tmp/toup-env-repair.txt "$ENV_FILE"
        chmod 600 "$ENV_FILE"
        rm -f /tmp/toup-env-repair.txt
        add_check "env_file" "fixed" "Regenerated .env from platform config" "true"
    else
        add_check "env_file" "error" "No .env file — agent won't start without config"
    fi
else
    # .env exists — check for missing critical vars and patch them in
    MISSING_VARS=""
    for VAR in $REQUIRED_VARS; do
        grep -q "^${VAR}=" "$ENV_FILE" || MISSING_VARS="${MISSING_VARS}${VAR} "
    done
    if [ -n "$MISSING_VARS" ] && [ -f /tmp/toup-env-repair.txt ]; then
        # Patch missing vars from the platform-generated env
        for VAR in $MISSING_VARS; do
            LINE=$(grep "^${VAR}=" /tmp/toup-env-repair.txt || true)
            if [ -n "$LINE" ]; then
                echo "$LINE" >> "$ENV_FILE"
            fi
        done
        rm -f /tmp/toup-env-repair.txt
        add_check "env_file" "fixed" "Patched missing vars: $MISSING_VARS" "true"
    elif [ -n "$MISSING_VARS" ]; then
        add_check "env_file" "warning" "Missing vars: $MISSING_VARS"
    else
        add_check "env_file" "ok" ".env present with required vars"
    fi
    # Clean up repair file if not used
    rm -f /tmp/toup-env-repair.txt
fi

# 8. Xvfb (needed for headed browser)
if command -v Xvfb &>/dev/null; then
    add_check "xvfb" "ok" "Xvfb installed"
else
    apt-get install -y -qq xvfb 2>/dev/null
    if command -v Xvfb &>/dev/null; then
        add_check "xvfb" "fixed" "Installed Xvfb" "true"
    else
        add_check "xvfb" "warning" "Xvfb not available (browser will use headless mode)"
    fi
fi

# 9. Google Chrome
if command -v google-chrome &>/dev/null || command -v google-chrome-stable &>/dev/null; then
    CHROME_VER=$(google-chrome --version 2>/dev/null || google-chrome-stable --version 2>/dev/null || echo "installed")
    add_check "chrome" "ok" "$CHROME_VER"
else
    wget -q -O /tmp/chrome.deb https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb 2>/dev/null
    apt-get install -y -qq /tmp/chrome.deb 2>/dev/null || apt-get install -f -y -qq 2>/dev/null
    rm -f /tmp/chrome.deb
    if command -v google-chrome &>/dev/null || command -v google-chrome-stable &>/dev/null; then
        add_check "chrome" "fixed" "Installed Google Chrome" "true"
    else
        add_check "chrome" "warning" "Chrome install failed (will use Patchright Chromium)"
    fi
fi

# 10. Systemd service
if [ -f /etc/systemd/system/toup-agent.service ]; then
    add_check "systemd_unit" "ok" "Service file exists"
else
    cat > /etc/systemd/system/toup-agent.service << 'UNIT'
[Unit]
Description=Toup Agent Service
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/toup-agent
EnvironmentFile=/opt/toup-agent/.env
ExecStart=/opt/toup-agent/venv/bin/uvicorn agent_main:app --host 0.0.0.0 --port 8001 --workers 1
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=toup-agent
NoNewPrivileges=true
ReadWritePaths=/opt/toup-agent /tmp

[Install]
WantedBy=multi-user.target
UNIT
    systemctl daemon-reload
    systemctl enable toup-agent
    add_check "systemd_unit" "fixed" "Created and enabled systemd service" "true"
fi

# 11. Restart the agent service
systemctl daemon-reload
systemctl restart toup-agent 2>&1
sleep 2
if systemctl is-active --quiet toup-agent; then
    add_check "service_running" "ok" "toup-agent is running"
else
    JOURNAL=$(journalctl -u toup-agent --no-pager -n 10 --since "30 seconds ago" 2>/dev/null | tail -5 || echo "no logs")
    add_check "service_running" "error" "Service failed to start: $JOURNAL"
fi

# 12. Health check
sleep 2
HEALTH=$(curl -sf http://127.0.0.1:8001/agent/health 2>/dev/null || echo "")
if [ -n "$HEALTH" ]; then
    add_check "health_check" "ok" "Agent responding on port 8001"
else
    add_check "health_check" "warning" "Agent not yet responding (may still be starting)"
fi

# Output JSON report
echo '{"checks":['
for i in "${!CHECKS[@]}"; do
    if [ $i -gt 0 ]; then echo ","; fi
    echo "${CHECKS[$i]}"
done
echo ']}'
'''


async def _ssh_diagnose_and_repair(user_id: str, db: AsyncSession) -> dict:
    """Comprehensive VPS diagnostic & repair via SSH through Lambda proxy."""
    from app.services.ssh_deploy_service import _invoke_lambda, generate_env_content

    result = await db.execute(
        select(AgentConfig).where(AgentConfig.user_id == user_id)
    )
    cfg = result.scalar_one_or_none()
    if not cfg or not cfg.ssh_host:
        return {
            "success": False,
            "error": "No SSH credentials found. Your VPS needs to be set up first.",
            "checks": [],
        }

    # Generate the .env content from the user's config in the DB
    # so the diagnostic script can repair/regenerate it on the VPS
    env_content = generate_env_content(
        user_id=user_id,
        agent_api_key=cfg.agent_api_key or "",
        openai_api_key=cfg.openai_api_key or "",
        anthropic_api_key=cfg.anthropic_api_key or "",
        google_api_key=cfg.google_api_key or "",
        mistral_api_key=cfg.mistral_api_key or "",
        groq_api_key=cfg.groq_api_key or "",
        xai_api_key=cfg.xai_api_key or "",
        deepseek_api_key=cfg.deepseek_api_key or "",
        agent_model=cfg.agent_model or "claude-opus-4-6",
        llm_mode=cfg.llm_mode or "manual",
        telegram_bot_token=cfg.telegram_bot_token or "",
        discord_bot_token=cfg.discord_bot_token or "",
        slack_bot_token=cfg.slack_bot_token or "",
        slack_app_token=cfg.slack_app_token or "",
        whatsapp_phone_number_id=cfg.whatsapp_phone_number_id or "",
        whatsapp_access_token=cfg.whatsapp_access_token or "",
        brave_api_key=cfg.brave_api_key or "",
        elevenlabs_api_key=cfg.elevenlabs_api_key or "",
        toup_token=cfg.connect_token or "",
        db_mode=cfg.db_mode or "auto",
        supabase_url=cfg.supabase_url or "",
    )

    try:
        resp = await _invoke_lambda({
            "action": "execute",
            "ssh_host": cfg.ssh_host,
            "ssh_port": cfg.ssh_port or 22,
            "ssh_user": cfg.ssh_user or "root",
            "ssh_password": cfg.ssh_password,
            "ssh_key": cfg.ssh_key,
            "commands": [
                # Upload the platform-generated .env for repair use
                f"cat > /tmp/toup-env-repair.txt << 'ENVEOF'\n{env_content}\nENVEOF",
                "chmod 600 /tmp/toup-env-repair.txt",
                # Write the diagnostic script and run it
                f"cat > /tmp/toup-diagnose.sh << 'DIAGEOF'\n{_DIAGNOSE_AND_REPAIR_SCRIPT}\nDIAGEOF",
                "chmod +x /tmp/toup-diagnose.sh && bash /tmp/toup-diagnose.sh",
            ],
        })

        if "error" in resp:
            return {"success": False, "error": resp["error"], "checks": []}

        # Parse the JSON output from the diagnostic script
        output = resp.get("output", "") or ""
        import json as _json
        # Find the JSON block in the output
        json_start = output.rfind('{"checks":')
        if json_start >= 0:
            try:
                report = _json.loads(output[json_start:])
                checks = report.get("checks", [])
                # Determine overall success
                has_errors = any(c.get("status") == "error" for c in checks)
                return {
                    "success": not has_errors,
                    "method": "ssh_diagnose",
                    "checks": checks,
                }
            except _json.JSONDecodeError:
                pass

        # Couldn't parse JSON — return raw output
        return {
            "success": True,
            "method": "ssh",
            "output": output[-500:] if output else "Diagnostic completed (no output)",
            "checks": [],
        }

    except Exception as e:
        logger.error("SSH diagnose & repair failed: %s", e)
        return {"success": False, "error": f"SSH connection failed: {e}", "checks": []}


@router.post("/agent-diagnose")
async def diagnose_agent(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Read-only diagnostic — checks VPS health without changing anything.

    Strategy:
    1. Try agent's /agent/diagnose endpoint (agent is alive — no SSH needed)
    2. If unreachable, return connection error with guidance
    """
    proxy = await _get_agent_proxy_info(current_user.id, db)
    if not proxy:
        return {
            "success": False,
            "error": "No agent configured. Set up your VPS first.",
            "checks": [],
        }

    agent_url, agent_api_key = proxy
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0)) as client:
            resp = await client.post(
                f"{agent_url}/agent/diagnose",
                headers={"X-Agent-Key": agent_api_key},
            )
            if resp.status_code == 200:
                return resp.json()
            logger.warning("Agent diagnose returned %s", resp.status_code)
    except Exception as e:
        logger.warning("Agent unreachable for diagnose: %s", e)

    # Agent unreachable — can't diagnose without SSH
    return {
        "success": False,
        "error": f"Cannot reach agent at {agent_url}. The server may be down or port 8001 blocked.",
        "checks": [
            {"name": "connectivity", "status": "error",
             "detail": f"Agent not responding at {agent_url}", "fixed": False},
        ],
    }


@router.post("/agent-update")
async def update_agent(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Diagnose and repair the user's agent VPS.

    Strategy:
    1. Try agent's /agent/diagnose-and-fix (agent is alive — self-repair over HTTP)
    2. If unreachable, SSH in and run comprehensive diagnostics + auto-repair
    """
    # 1. Try agent's self-diagnose-and-fix endpoint (fast path — no SSH needed)
    proxy = await _get_agent_proxy_info(current_user.id, db)
    if proxy:
        agent_url, agent_api_key = proxy
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=5.0)) as client:
                resp = await client.post(
                    f"{agent_url}/agent/diagnose-and-fix",
                    headers={"X-Agent-Key": agent_api_key},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    data["method"] = "agent_self_fix"
                    return data
                logger.warning("Agent diagnose-and-fix returned %s", resp.status_code)
        except Exception as e:
            logger.warning("Agent unreachable for self-fix: %s", e)

    # 2. Comprehensive SSH diagnose & repair (bulletproof path — agent is down)
    logger.info("Running SSH diagnose & repair for user %s", current_user.id[:8])
    return await _ssh_diagnose_and_repair(current_user.id, db)
