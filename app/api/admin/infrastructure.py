"""
Admin Infrastructure API — VPS monitoring, container management,
resource usage per user.
"""

import asyncio
import logging
from datetime import datetime

from fastapi import APIRouter, Depends
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.db.models import ManagedContainer, User, AgentConfig
from app.api.admin.deps import require_admin
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin/infrastructure", tags=["admin-infrastructure"])


async def _ssh_cmd(cmd: str) -> str:
    """DEPRECATED — SSH-as-root to the Docker host is removed in Phase 3.

    Left as a stub so existing admin endpoints don't crash on import, but
    every caller will get an empty string. Admin metrics / log views that
    rely on this are degraded until someone builds equivalent endpoints
    on the provisioning bridge (Phase 4 work).

    Call sites that still invoke this are:
      - overview()      host-level metrics (disk, mem, CPU, container list)
      - container_details()   per-container logs, ps, env
      - various tenant data_volume queries

    All of those will show empty/N-A in the admin UI until ported.
    """
    logger.debug("[INFRA-SSH] _ssh_cmd called but SSH is removed (cmd=%s...)", cmd[:60])
    return ""


@router.get("/debug-bridge")
async def debug_bridge(_=Depends(require_admin)):
    """Debug provisioning-bridge connectivity (replaces debug-ssh).

    Returns whether bridge mTLS is configured + a round-trip health check.
    """
    info = {
        "bridge_url": settings.bridge_url or "(not set)",
        "bridge_ca_cert_configured": bool(settings.bridge_ca_cert),
        "bridge_client_cert_configured": bool(settings.bridge_client_cert),
        "bridge_client_key_configured": bool(settings.bridge_client_key),
        "managed_hosting_enabled": settings.managed_hosting_enabled,
    }
    if not settings.bridge_url or not settings.bridge_ca_cert:
        info["bridge_health"] = "NOT CONFIGURED"
        return info
    try:
        from app.services.docker_host_service import _bridge_client
        async with _bridge_client(timeout_s=5) as client:
            r = await client.get("/v1/health")
            info["bridge_health"] = f"{r.status_code} {r.text[:100]}"
    except Exception as e:
        info["bridge_health"] = f"FAILED: {type(e).__name__}: {str(e)[:200]}"
    return info


@router.get("/overview")
async def overview(
    _=Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Get full VPS infrastructure overview.

    Host + docker stats come from the bridge's /v1/host/overview endpoint
    (Phase 4 rebuild — was previously SSH-as-root which is now removed).
    If the bridge is unreachable, the host sections come back as a
    degraded stub and the UI shows a banner — better than hard-500'ing.
    """
    from app.services.docker_host_service import _bridge_client

    # Bridge host overview (parallel with DB reads below)
    async def _fetch_bridge():
        try:
            async with _bridge_client(timeout_s=10) as client:
                r = await client.get("/v1/host/overview")
                if r.status_code == 200:
                    return r.json()
                logger.warning("bridge /v1/host/overview returned HTTP %s", r.status_code)
        except Exception as e:
            logger.warning("bridge host overview fetch failed: %s", e)
        return None

    # Managed containers from platform DB
    async def _fetch_managed():
        result = await db.execute(
            select(ManagedContainer, User.name, User.email)
            .outerjoin(User, ManagedContainer.user_id == User.id)
            .order_by(ManagedContainer.created_at.desc())
        )
        return result.all()

    total_users_task = db.execute(select(func.count(User.id)))

    bridge_data, managed_rows, total_users_result = await asyncio.gather(
        _fetch_bridge(), _fetch_managed(), total_users_task
    )

    # Host stats — use bridge data if we got it, else degraded stubs
    if bridge_data:
        host = {
            "ip": (settings.bridge_url or "").replace("https://", "").replace("http://", "").split("/")[0],
            "hostname": bridge_data["host"]["hostname"],
            "uptime_sec": bridge_data["host"]["uptime_sec"],
            "disk": bridge_data["host"]["disk"],
            "memory": bridge_data["host"]["memory"],
            "cpu": bridge_data["host"]["cpu"],
        }
        containers_running = bridge_data.get("containers_running", [])
        container_stats = bridge_data.get("container_stats", {})
    else:
        host = {
            "ip": "bridge unreachable",
            "hostname": "?",
            "uptime_sec": 0,
            "disk": {"total_bytes": 0, "used_bytes": 0, "free_bytes": 0, "percent": 0},
            "memory": {"total_mb": 0, "available_mb": 0, "used_mb": 0, "percent": 0},
            "cpu": {"cores": 0, "load_1m": 0, "load_5m": 0, "load_15m": 0},
        }
        containers_running = []
        container_stats = {}

    # Managed containers from DB
    managed = []
    for container, user_name, user_email in managed_rows:
        managed.append({
            "id": container.id,
            "user_id": container.user_id,
            "user_name": user_name or "Unknown",
            "user_email": user_email or "",
            "container_name": container.container_name,
            "container_id": container.container_id,
            "port": container.host_port,
            "db_name": container.db_name,
            "status": container.status,
            "image_tag": container.image_tag,
            "error": container.error_message,
            "created_at": container.created_at.isoformat() if container.created_at else None,
            "started_at": container.started_at.isoformat() if container.started_at else None,
            "stopped_at": container.stopped_at.isoformat() if container.stopped_at else None,
        })

    total_users = total_users_result.scalar() or 0
    total_containers = len(managed)
    running_containers = sum(1 for c in managed if c["status"] == "running")

    return {
        "host": host,
        "docker": {
            "containers_running": containers_running,
            "container_stats": container_stats,
        },
        "managed": managed,
        "totals": {
            "users": total_users,
            "containers": total_containers,
            "running": running_containers,
            "port_range": f"{settings.docker_port_range_start}-{settings.docker_port_range_end}",
            "max_capacity": settings.docker_port_range_end - settings.docker_port_range_start + 1,
        },
        "bridge_reachable": bridge_data is not None,
    }


@router.post("/containers/{container_id}/start")
async def start_container(
    container_id: str,
    _=Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Start a managed container."""
    from app.services.docker_host_service import start_container as svc_start
    result = await db.execute(
        select(ManagedContainer).where(ManagedContainer.id == container_id)
    )
    container = result.scalar_one_or_none()
    if not container:
        return {"error": "Not found"}
    c = await svc_start(db, container.user_id)
    return {"status": c.status if c else "error"}


@router.post("/containers/{container_id}/stop")
async def stop_container(
    container_id: str,
    _=Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Stop a managed container."""
    from app.services.docker_host_service import stop_container as svc_stop
    result = await db.execute(
        select(ManagedContainer).where(ManagedContainer.id == container_id)
    )
    container = result.scalar_one_or_none()
    if not container:
        return {"error": "Not found"}
    c = await svc_stop(db, container.user_id)
    return {"status": c.status if c else "error"}


@router.post("/containers/{container_id}/restart")
async def restart_container(
    container_id: str,
    _=Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Restart a managed container."""
    from app.services.docker_host_service import restart_container as svc_restart
    result = await db.execute(
        select(ManagedContainer).where(ManagedContainer.id == container_id)
    )
    container = result.scalar_one_or_none()
    if not container:
        return {"error": "Not found"}
    c = await svc_restart(db, container.user_id)
    return {"status": c.status if c else "error"}


@router.post("/containers/{container_id}/destroy")
async def destroy_container(
    container_id: str,
    _=Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Destroy a managed container."""
    from app.services.docker_host_service import destroy_container as svc_destroy
    result = await db.execute(
        select(ManagedContainer).where(ManagedContainer.id == container_id)
    )
    container = result.scalar_one_or_none()
    if not container:
        return {"error": "Not found"}
    await svc_destroy(db, container.user_id)
    return {"status": "deleted"}


@router.post("/deploy-update", deprecated=True)
async def deploy_update_deprecated(
    _=Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """DEPRECATED — use POST /api/admin/rollout/start (W4) instead.

    The old implementation SSH'd to /opt/toup-agent, ran `git pull` +
    `docker build` on the VPS, and rolling-updated containers inline.
    Phase 3 replaced all of that with:

      - CI builds + pushes ghcr.io/toup-com/toup-agent:<sha>  (build-agent.yml)
      - CI POSTs to /api/admin/rollout/start                   (webhook)
      - rollout_service.py orchestrates canary → batches → health gates

    See docs/new-vps/14-AUTOMATED-DEPLOYMENT-DESIGN.md for the pipeline.
    """
    return {
        "error": "deploy_update is removed in Phase 3",
        "use_instead": "POST /api/admin/rollout/start or POST /api/admin/rollout/manual",
        "docs": "docs/new-vps/14-AUTOMATED-DEPLOYMENT-DESIGN.md",
    }


@router.post("/provision/{user_id}")
async def admin_provision(
    user_id: str,
    _=Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Admin: provision a managed container for any user."""
    from app.services.docker_host_service import provision_container

    # Get the user's agent config
    result = await db.execute(
        select(AgentConfig).where(AgentConfig.user_id == user_id)
    )
    agent_config = result.scalar_one_or_none()

    try:
        container = await provision_container(db, user_id, agent_config)
        return {
            "status": container.status,
            "port": container.host_port,
            "container_name": container.container_name,
        }
    except Exception as e:
        return {"error": str(e)}


@router.get("/containers/{container_id}/files")
async def list_files(
    container_id: str,
    path: str = "",
    _=Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Browse files INSIDE a managed container (not just host volumes)."""
    result = await db.execute(
        select(ManagedContainer).where(ManagedContainer.id == container_id)
    )
    container = result.scalar_one_or_none()
    if not container:
        return {"error": "Not found"}

    # Sanitize path
    clean_path = path.strip("/").replace("..", "")
    target = f"/app/{clean_path}" if clean_path else "/app"

    # List files INSIDE the container via docker exec
    raw = await _ssh_cmd(
        f"docker exec {container.container_name} "
        f"find {target} -maxdepth 1 -printf '%y|%s|%T@|%f\\n' 2>/dev/null | head -200"
    )
    if not raw:
        return {"path": clean_path, "files": [], "base": "/app"}

    files = []
    for line in raw.strip().split("\n"):
        parts = line.split("|", 3)
        if len(parts) < 4 or parts[3] == ".":
            continue
        ftype = "dir" if parts[0] == "d" else "file"
        size = int(parts[1]) if parts[1].isdigit() else 0
        mtime = float(parts[2]) if parts[2].replace(".", "").isdigit() else 0
        name = parts[3]
        files.append({
            "name": name,
            "type": ftype,
            "size": size,
            "modified": datetime.fromtimestamp(mtime).isoformat() if mtime else None,
        })

    # Sort: dirs first, then by name
    files.sort(key=lambda f: (0 if f["type"] == "dir" else 1, f["name"]))

    return {"path": clean_path, "files": files, "base": "/app"}


@router.get("/containers/{container_id}/file-content")
async def read_file(
    container_id: str,
    path: str,
    _=Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Read a file's content from inside the container."""
    result = await db.execute(
        select(ManagedContainer).where(ManagedContainer.id == container_id)
    )
    container = result.scalar_one_or_none()
    if not container:
        return {"error": "Not found"}

    clean_path = path.strip("/").replace("..", "")
    target = f"/app/{clean_path}"

    # Check file size inside container
    size_check = await _ssh_cmd(
        f"docker exec {container.container_name} stat -c%s {target} 2>/dev/null"
    )
    if not size_check or not size_check.isdigit() or int(size_check) > 1_000_000:
        return {"error": "File too large or not found", "path": clean_path}

    content = await _ssh_cmd(
        f"docker exec {container.container_name} cat {target} 2>/dev/null"
    )
    return {"path": clean_path, "content": content, "size": int(size_check)}


@router.post("/containers/{container_id}/terminal")
async def terminal_exec(
    container_id: str,
    body: dict,
    _=Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Execute a command inside a managed container's terminal."""
    cmd = body.get("command", "")
    if not cmd:
        return {"error": "No command provided"}

    result = await db.execute(
        select(ManagedContainer).where(ManagedContainer.id == container_id)
    )
    container = result.scalar_one_or_none()
    if not container:
        return {"error": "Container not found"}

    # Run command inside the Docker container
    # We pipe to the SSH host which then pipes to docker exec
    output = await _ssh_cmd(
        f"echo {repr(cmd)} | docker exec -i {container.container_name} bash 2>&1"
    )
    return {"output": output, "command": cmd}


@router.get("/containers/{container_id}/details")
async def container_details(
    container_id: str,
    _=Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Get detailed info about a container — logs, env, processes, disk usage."""
    result = await db.execute(
        select(ManagedContainer, User.name, User.email)
        .outerjoin(User, ManagedContainer.user_id == User.id)
        .where(ManagedContainer.id == container_id)
    )
    row = result.one_or_none()
    if not row:
        return {"error": "Not found"}

    container, user_name, user_email = row
    name = container.container_name

    # Gather info in parallel
    logs_task = _ssh_cmd(f"docker logs --tail 50 {name} 2>&1")
    processes_task = _ssh_cmd(f"docker exec {name} ps aux 2>&1")
    disk_task = _ssh_cmd(f"du -sh /data/agents/{container.user_id[:8]}/workspace /data/agents/{container.user_id[:8]}/skills 2>/dev/null")
    env_task = _ssh_cmd(f"cat /data/agents/{container.user_id[:8]}/.env 2>/dev/null")
    health_task = _ssh_cmd(f"curl -sf http://localhost:{container.host_port}/agent/health 2>/dev/null")

    logs, processes, disk, env_raw, health = await asyncio.gather(
        logs_task, processes_task, disk_task, env_task, health_task
    )

    # Mask API keys in env
    env_lines = []
    for line in (env_raw or "").split("\n"):
        if "=" in line:
            key, val = line.split("=", 1)
            if any(s in key.upper() for s in ["KEY", "TOKEN", "PASSWORD", "SECRET"]):
                env_lines.append(f"{key}={val[:8]}...{val[-4:]}" if len(val) > 16 else f"{key}=****")
            else:
                env_lines.append(line)
        else:
            env_lines.append(line)

    # Get user's apps
    from app.db.models import App
    apps_result = await db.execute(
        select(App).where(App.user_id == container.user_id).order_by(App.created_at.desc())
    )
    apps = [{
        "id": a.id,
        "name": a.name,
        "slug": a.slug,
        "status": a.status,
        "port": a.port,
        "web_port": a.web_port,
        "platforms": a.platforms,
        "created_at": a.created_at.isoformat() if a.created_at else None,
    } for a in apps_result.scalars().all()]

    return {
        "id": container.id,
        "user_id": container.user_id,
        "user_name": user_name,
        "user_email": user_email,
        "container_name": name,
        "port": container.host_port,
        "db_name": container.db_name,
        "status": container.status,
        "logs": logs or "",
        "processes": processes or "",
        "disk_usage": disk or "",
        "env": "\n".join(env_lines),
        "health": health or "",
        "apps": apps,
        "created_at": container.created_at.isoformat() if container.created_at else None,
        "started_at": container.started_at.isoformat() if container.started_at else None,
    }
