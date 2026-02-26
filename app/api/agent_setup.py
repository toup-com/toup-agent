"""
Agent setup API — configuration wizard + deployment endpoints.

GET  /api/agent-setup/config           — get current config
PUT  /api/agent-setup/config           — update config (partial)
POST /api/agent-setup/generate-key     — generate agent API key
POST /api/agent-setup/test-ssh         — test SSH connectivity
POST /api/agent-setup/deploy           — trigger agent deployment
GET  /api/agent-setup/deploy-status    — check deployment status
GET  /api/agent-setup/env              — download .env content
POST /api/agent-setup/test-connection  — test deployed agent health
POST /api/agent-setup/register         — agent self-registration on startup
"""

import logging
import secrets
import uuid
from datetime import datetime
from typing import Optional

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import get_current_user
from app.config import settings
from app.db import get_db, AgentConfig, VPSInstance

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent-setup", tags=["Agent Setup"])

# ── Shared state for deploy log streaming ─────────────────────────────────
# Maps user_id → list of (line, level) tuples + done flag
_deploy_logs: dict[str, dict] = {}


# ── Schemas ───────────────────────────────────────────────────────────────

class AgentConfigOut(BaseModel):
    hosting_mode: str = "self-hosted"
    ssh_host: str | None = None
    ssh_port: int = 22
    ssh_user: str | None = "ubuntu"
    # SSH creds never returned — only whether they're set
    has_ssh_password: bool = False
    has_ssh_key: bool = False
    # LLM
    llm_mode: str = "manual"
    openai_api_key_set: bool = False
    anthropic_api_key_set: bool = False
    google_api_key_set: bool = False
    mistral_api_key_set: bool = False
    groq_api_key_set: bool = False
    xai_api_key_set: bool = False
    deepseek_api_key_set: bool = False
    agent_model: str = "gpt-5.2"
    # Bundle
    bundle_status: str = "none"
    bundle_current_period_end: str | None = None
    # Channels
    telegram_bot_token_set: bool = False
    discord_bot_token_set: bool = False
    slack_bot_token_set: bool = False
    slack_app_token_set: bool = False
    whatsapp_phone_number_id: str | None = None
    whatsapp_access_token_set: bool = False
    brave_api_key_set: bool = False
    elevenlabs_api_key_set: bool = False
    agent_api_key: str | None = None
    agent_url: str | None = None
    deploy_status: str = "none"
    setup_completed: bool = False
    setup_step: int = 1
    # VPS info (if hosting_mode == "vps")
    vps_ip: str | None = None
    vps_status: str | None = None
    vps_plan: str | None = None

    class Config:
        from_attributes = True


class AgentConfigUpdate(BaseModel):
    hosting_mode: str | None = None
    ssh_host: str | None = None
    ssh_port: int | None = None
    ssh_user: str | None = None
    ssh_password: str | None = None
    ssh_key: str | None = None
    llm_mode: str | None = None
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    google_api_key: str | None = None
    mistral_api_key: str | None = None
    groq_api_key: str | None = None
    xai_api_key: str | None = None
    deepseek_api_key: str | None = None
    agent_model: str | None = None
    telegram_bot_token: str | None = None
    discord_bot_token: str | None = None
    slack_bot_token: str | None = None
    slack_app_token: str | None = None
    whatsapp_phone_number_id: str | None = None
    whatsapp_access_token: str | None = None
    brave_api_key: str | None = None
    elevenlabs_api_key: str | None = None
    setup_step: int | None = None
    setup_completed: bool | None = None


class SSHTestRequest(BaseModel):
    ssh_host: str | None = None
    ssh_port: int | None = None
    ssh_user: str | None = None
    ssh_password: str | None = None
    ssh_key: str | None = None


class AgentRegisterRequest(BaseModel):
    agent_url: str
    agent_api_key: str


# ── Helpers ───────────────────────────────────────────────────────────────

async def _get_or_create_config(
    user_id: str, db: AsyncSession,
) -> AgentConfig:
    """Get the user's AgentConfig, or create one."""
    result = await db.execute(
        select(AgentConfig).where(AgentConfig.user_id == user_id)
    )
    config = result.scalar_one_or_none()
    if not config:
        config = AgentConfig(
            id=str(uuid.uuid4()),
            user_id=user_id,
        )
        db.add(config)
        await db.commit()
        await db.refresh(config)
    return config


def _config_to_out(config: AgentConfig, vps: VPSInstance | None = None) -> AgentConfigOut:
    """Convert AgentConfig to safe output (no secrets)."""
    out = AgentConfigOut(
        hosting_mode=config.hosting_mode,
        ssh_host=config.ssh_host,
        ssh_port=config.ssh_port,
        ssh_user=config.ssh_user,
        has_ssh_password=bool(config.ssh_password),
        has_ssh_key=bool(config.ssh_key),
        llm_mode=config.llm_mode or "manual",
        openai_api_key_set=bool(config.openai_api_key),
        anthropic_api_key_set=bool(config.anthropic_api_key),
        google_api_key_set=bool(config.google_api_key),
        mistral_api_key_set=bool(config.mistral_api_key),
        groq_api_key_set=bool(config.groq_api_key),
        xai_api_key_set=bool(config.xai_api_key),
        deepseek_api_key_set=bool(config.deepseek_api_key),
        agent_model=config.agent_model,
        bundle_status=config.bundle_status or "none",
        bundle_current_period_end=(
            config.bundle_current_period_end.isoformat()
            if config.bundle_current_period_end else None
        ),
        telegram_bot_token_set=bool(config.telegram_bot_token),
        discord_bot_token_set=bool(config.discord_bot_token),
        slack_bot_token_set=bool(config.slack_bot_token),
        slack_app_token_set=bool(config.slack_app_token),
        whatsapp_phone_number_id=config.whatsapp_phone_number_id,
        whatsapp_access_token_set=bool(config.whatsapp_access_token),
        brave_api_key_set=bool(config.brave_api_key),
        elevenlabs_api_key_set=bool(config.elevenlabs_api_key),
        agent_api_key=config.agent_api_key,
        agent_url=config.agent_url,
        deploy_status=config.deploy_status,
        setup_completed=config.setup_completed,
        setup_step=config.setup_step,
    )
    if vps:
        out.vps_ip = vps.public_ip
        out.vps_status = vps.status
        out.vps_plan = vps.plan_id
    return out


def _build_env(config: AgentConfig, user_id: str) -> str:
    """Build .env content from an AgentConfig."""
    from app.services.ssh_deploy_service import generate_env_content
    return generate_env_content(
        user_id=user_id,
        agent_api_key=config.agent_api_key or "",
        openai_api_key=config.openai_api_key or "",
        anthropic_api_key=config.anthropic_api_key or "",
        google_api_key=config.google_api_key or "",
        mistral_api_key=config.mistral_api_key or "",
        groq_api_key=config.groq_api_key or "",
        xai_api_key=config.xai_api_key or "",
        deepseek_api_key=config.deepseek_api_key or "",
        agent_model=config.agent_model or "gpt-5.2",
        llm_mode=config.llm_mode or "manual",
        telegram_bot_token=config.telegram_bot_token or "",
        discord_bot_token=config.discord_bot_token or "",
        slack_bot_token=config.slack_bot_token or "",
        slack_app_token=config.slack_app_token or "",
        whatsapp_phone_number_id=config.whatsapp_phone_number_id or "",
        whatsapp_access_token=config.whatsapp_access_token or "",
        brave_api_key=config.brave_api_key or "",
        elevenlabs_api_key=config.elevenlabs_api_key or "",
    )


# ── Endpoints ─────────────────────────────────────────────────────────────

@router.get("/config", response_model=AgentConfigOut)
async def get_config(
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get the user's agent configuration."""
    config = await _get_or_create_config(current_user.id, db)

    # Also fetch VPS info if available
    vps_result = await db.execute(
        select(VPSInstance)
        .where(VPSInstance.user_id == current_user.id)
        .where(VPSInstance.status.in_(["active", "provisioning", "pending"]))
        .order_by(VPSInstance.created_at.desc())
    )
    vps = vps_result.scalars().first()

    # If user has an active VPS, auto-set hosting_mode
    if vps and vps.status == "active" and config.hosting_mode != "vps":
        config.hosting_mode = "vps"
        await db.commit()

    return _config_to_out(config, vps)


@router.put("/config", response_model=AgentConfigOut)
async def update_config(
    body: AgentConfigUpdate,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update agent configuration (partial update)."""
    config = await _get_or_create_config(current_user.id, db)

    update_data = body.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        if hasattr(config, field):
            setattr(config, field, value)
    config.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(config)

    # Fetch VPS info
    vps_result = await db.execute(
        select(VPSInstance)
        .where(VPSInstance.user_id == current_user.id)
        .where(VPSInstance.status.in_(["active", "provisioning"]))
        .order_by(VPSInstance.created_at.desc())
    )
    vps = vps_result.scalars().first()

    return _config_to_out(config, vps)


@router.post("/generate-key")
async def generate_key(
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Generate or regenerate an agent API key."""
    config = await _get_or_create_config(current_user.id, db)
    key = f"toup_ak_{secrets.token_urlsafe(32)}"
    config.agent_api_key = key
    config.updated_at = datetime.utcnow()
    await db.commit()
    return {"agent_api_key": key}


@router.post("/test-ssh")
async def test_ssh(
    body: SSHTestRequest,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Test SSH connectivity to the target machine."""
    config = await _get_or_create_config(current_user.id, db)

    # Determine SSH credentials
    ssh_host = body.ssh_host or config.ssh_host
    ssh_port = body.ssh_port or config.ssh_port or 22
    ssh_user = body.ssh_user or config.ssh_user or "ubuntu"
    ssh_password = body.ssh_password or config.ssh_password
    ssh_key = body.ssh_key or config.ssh_key

    # For VPS users, auto-use VPS credentials
    if config.hosting_mode == "vps" and not ssh_host:
        vps_result = await db.execute(
            select(VPSInstance)
            .where(VPSInstance.user_id == current_user.id)
            .where(VPSInstance.status == "active")
            .order_by(VPSInstance.created_at.desc())
        )
        vps = vps_result.scalars().first()
        if vps:
            ssh_host = vps.public_ip
            ssh_password = ssh_password or vps.ssh_password
            ssh_user = "ubuntu"

    if not ssh_host:
        raise HTTPException(status_code=400, detail="No SSH host specified")

    from app.services.ssh_deploy_service import test_ssh_connection
    result = await test_ssh_connection(ssh_host, ssh_port, ssh_user, ssh_password, ssh_key)
    return result


@router.post("/deploy")
async def trigger_deploy(
    background_tasks: BackgroundTasks,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Trigger agent deployment via SSH."""
    config = await _get_or_create_config(current_user.id, db)

    if config.deploy_status == "deploying":
        raise HTTPException(status_code=409, detail="Deployment already in progress")

    if not config.openai_api_key:
        raise HTTPException(status_code=400, detail="OpenAI API key is required")

    # Generate API key if not set
    if not config.agent_api_key:
        config.agent_api_key = f"toup_ak_{secrets.token_urlsafe(32)}"

    # Determine SSH credentials
    ssh_host = config.ssh_host
    ssh_port = config.ssh_port or 22
    ssh_user = config.ssh_user or "ubuntu"
    ssh_password = config.ssh_password
    ssh_key = config.ssh_key

    # For VPS users, use VPS credentials
    if config.hosting_mode == "vps":
        vps_result = await db.execute(
            select(VPSInstance)
            .where(VPSInstance.user_id == current_user.id)
            .where(VPSInstance.status == "active")
            .order_by(VPSInstance.created_at.desc())
        )
        vps = vps_result.scalars().first()
        if not vps:
            raise HTTPException(status_code=400, detail="No active VPS found")
        ssh_host = vps.public_ip
        ssh_password = vps.ssh_password
        ssh_user = "ubuntu"

    if not ssh_host:
        raise HTTPException(status_code=400, detail="No target machine configured")

    # Generate .env content
    from app.services.ssh_deploy_service import generate_env_content
    env_content = _build_env(config, current_user.id)

    config.deploy_status = "deploying"
    config.deploy_log = ""
    await db.commit()

    # Initialize log stream
    _deploy_logs[current_user.id] = {"lines": [], "done": False, "success": False}

    # Run deployment in background
    from app.db.database import async_session_maker
    background_tasks.add_task(
        _run_deploy,
        user_id=current_user.id,
        ssh_host=ssh_host,
        ssh_port=ssh_port,
        ssh_user=ssh_user,
        ssh_password=ssh_password,
        ssh_key=ssh_key,
        env_content=env_content,
        agent_url=f"http://{ssh_host}:8001",
        db_factory=async_session_maker,
    )

    return {"status": "deploying", "message": "Deployment started"}


async def _run_deploy(
    user_id: str,
    ssh_host: str,
    ssh_port: int,
    ssh_user: str,
    ssh_password: str | None,
    ssh_key: str | None,
    env_content: str,
    agent_url: str,
    db_factory,
):
    """Background task: run SSH deployment and update status."""
    log_state = _deploy_logs.get(user_id, {"lines": [], "done": False, "success": False})

    async def on_log(line: str, level: str):
        log_state["lines"].append({"line": line, "level": level})

    try:
        from app.services.ssh_deploy_service import deploy_agent
        success = await deploy_agent(
            ssh_host, ssh_port, ssh_user, ssh_password, ssh_key,
            env_content, on_log,
        )
    except Exception as e:
        logger.exception("Deploy background task failed: %s", e)
        success = False
        await on_log(f"Fatal error: {e}", "error")

    log_state["done"] = True
    log_state["success"] = success

    # Update database
    async with db_factory() as db:
        result = await db.execute(
            select(AgentConfig).where(AgentConfig.user_id == user_id)
        )
        config = result.scalar_one_or_none()
        if config:
            config.deploy_status = "active" if success else "error"
            config.agent_url = agent_url if success else None
            config.deploy_log = "\n".join(
                entry["line"] for entry in log_state["lines"][-100:]
            )
            config.updated_at = datetime.utcnow()
            if success:
                config.setup_completed = True
            await db.commit()


@router.get("/deploy-status")
async def get_deploy_status(
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Check current deployment status and recent logs."""
    config = await _get_or_create_config(current_user.id, db)

    log_state = _deploy_logs.get(current_user.id)
    recent_lines = []
    if log_state:
        recent_lines = log_state["lines"][-50:]

    return {
        "deploy_status": config.deploy_status,
        "done": log_state["done"] if log_state else True,
        "success": log_state["success"] if log_state else config.deploy_status == "active",
        "lines": recent_lines,
        "agent_url": config.agent_url,
    }


@router.get("/env")
async def get_env_content(
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Generate .env file content for manual download."""
    config = await _get_or_create_config(current_user.id, db)

    if not config.agent_api_key:
        config.agent_api_key = f"toup_ak_{secrets.token_urlsafe(32)}"
        await db.commit()

    content = _build_env(config, current_user.id)
    return {"content": content}


@router.get("/setup-script")
async def get_setup_script(
    platform: str = "bash",
    format: str = "json",
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Generate a setup script for local machines. platform: 'bash' or 'windows'."""
    config = await _get_or_create_config(current_user.id, db)

    if not config.agent_api_key:
        config.agent_api_key = f"toup_ak_{secrets.token_urlsafe(32)}"
        await db.commit()

    env_content = _build_env(config, current_user.id)

    if platform == "windows":
        from app.services.ssh_deploy_service import generate_setup_script_windows
        script = generate_setup_script_windows(env_content)
    else:
        from app.services.ssh_deploy_service import generate_setup_script
        script = generate_setup_script(env_content)

    if format == "raw":
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(script, media_type="text/plain")

    return {"script": script}


@router.post("/test-connection")
async def test_agent_connection(
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Test if the deployed agent is reachable."""
    config = await _get_or_create_config(current_user.id, db)

    agent_url = config.agent_url
    if not agent_url:
        # Check VPS
        vps_result = await db.execute(
            select(VPSInstance)
            .where(VPSInstance.user_id == current_user.id)
            .where(VPSInstance.status == "active")
        )
        vps = vps_result.scalars().first()
        if vps and vps.public_ip:
            agent_url = f"http://{vps.public_ip}:8001"

    if not agent_url:
        return {"reachable": False, "error": "No agent URL configured"}

    try:
        import time
        start = time.time()
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{agent_url}/agent/health")
            latency = int((time.time() - start) * 1000)
            if resp.status_code == 200:
                return {"reachable": True, "latency_ms": latency, "error": None}
            return {"reachable": False, "latency_ms": latency, "error": f"HTTP {resp.status_code}"}
    except httpx.ConnectError:
        return {"reachable": False, "error": "Connection refused"}
    except httpx.ConnectTimeout:
        return {"reachable": False, "error": "Connection timed out"}
    except Exception as e:
        return {"reachable": False, "error": str(e)}


@router.post("/register")
async def agent_register(
    body: AgentRegisterRequest,
    db: AsyncSession = Depends(get_db),
):
    """Called by the agent on startup to register its URL.

    Authenticates via agent_api_key (not JWT).
    """
    # Find the AgentConfig with this API key
    result = await db.execute(
        select(AgentConfig).where(AgentConfig.agent_api_key == body.agent_api_key)
    )
    config = result.scalar_one_or_none()

    if not config:
        raise HTTPException(status_code=401, detail="Invalid agent API key")

    config.agent_url = body.agent_url
    config.deploy_status = "active"
    config.updated_at = datetime.utcnow()
    await db.commit()

    logger.info("Agent registered: user=%s url=%s", config.user_id, body.agent_url)
    return {"registered": True}


def get_deploy_logs(user_id: str) -> dict | None:
    """Get deploy log state for WebSocket streaming."""
    return _deploy_logs.get(user_id)
