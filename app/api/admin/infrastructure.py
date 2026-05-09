"""
Admin Infrastructure API — VPS monitoring, container management,
resource usage per user.
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional

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


@router.get("/bridge-status")
async def bridge_status(_=Depends(require_admin)):
    """Live state of the platform-side bridge supervisor + the bridge's
    own tenant healthchecker. One stop for "is the never-sleep stack
    working right now?"
    """
    from app.services.bridge_supervisor import last_status as supervisor_status

    info = {
        "supervisor": supervisor_status(),
        "tenant_healthcheck": None,
    }

    # Pull the bridge's tenant_health/state too. If the supervisor says
    # the bridge is unreachable we skip this — saves a 5s timeout per
    # admin pageload during an outage.
    if info["supervisor"].get("last_status") == "healthy":
        try:
            from app.services.docker_host_service import _bridge_client
            async with _bridge_client(timeout_s=5) as client:
                r = await client.get("/v1/tenants/health/state")
                if r.status_code == 200:
                    info["tenant_healthcheck"] = r.json()
                else:
                    info["tenant_healthcheck"] = {"error": f"HTTP {r.status_code}"}
        except Exception as e:
            info["tenant_healthcheck"] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
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

    def _fmt_bytes(n: int) -> str:
        """Human-readable bytes like `df -h` — 5.8G / 24M / ... so the UI
        backwards-compat format keeps rendering without a frontend redeploy."""
        if n <= 0:
            return "0B"
        for unit, div in (("T", 1024**4), ("G", 1024**3), ("M", 1024**2), ("K", 1024)):
            if n >= div:
                return f"{n / div:.1f}{unit}".rstrip("0").rstrip(".") + ("" if unit == "B" else "")
        return f"{n}B"

    # Host stats — use bridge data if we got it, else degraded stubs.
    # Bridge returns numeric fields (*_bytes, percent as float, cpu_pct as float);
    # the existing frontend reads `disk.total/used` as strings (like `df -h`) and
    # `container_stats[name].cpu` as a percentage string. Keep both shapes here
    # so both old frontend builds and any future bytes-aware callers work.
    if bridge_data:
        d = bridge_data["host"]["disk"]
        host = {
            "ip": (settings.bridge_url or "").replace("https://", "").replace("http://", "").split("/")[0],
            "hostname": bridge_data["host"]["hostname"],
            "uptime_sec": bridge_data["host"]["uptime_sec"],
            "disk": {
                # string forms (frontend-facing)
                "total": _fmt_bytes(d.get("total_bytes", 0)),
                "used": _fmt_bytes(d.get("used_bytes", 0)),
                "free": _fmt_bytes(d.get("free_bytes", 0)),
                "percent": f"{d.get('percent', 0):.0f}%",
                # numeric forms (future callers)
                "total_bytes": d.get("total_bytes", 0),
                "used_bytes": d.get("used_bytes", 0),
                "free_bytes": d.get("free_bytes", 0),
                "percent_num": d.get("percent", 0),
            },
            "memory": bridge_data["host"]["memory"],
            "cpu": bridge_data["host"]["cpu"],
        }
        containers_running = bridge_data.get("containers_running", [])
        # Translate container_stats: bridge uses cpu_pct/mem_mb/mem_pct,
        # frontend reads cpu/mem_usage/mem_percent as strings.
        container_stats = {}
        for name, s in (bridge_data.get("container_stats") or {}).items():
            if "error" in s:
                container_stats[name] = {"cpu": "?", "mem_usage": "?", "mem_percent": "?", "error": s["error"]}
            else:
                container_stats[name] = {
                    "cpu": f"{s.get('cpu_pct', 0):.1f}%",
                    "mem_usage": f"{s.get('mem_mb', 0):.0f}MB",
                    "mem_percent": f"{s.get('mem_pct', 0):.1f}%",
                    # numeric forms for future use
                    "cpu_pct": s.get("cpu_pct", 0),
                    "mem_mb": s.get("mem_mb", 0),
                    "mem_pct": s.get("mem_pct", 0),
                }
    else:
        host = {
            "ip": "bridge unreachable",
            "hostname": "?",
            "uptime_sec": 0,
            "disk": {"total": "?", "used": "?", "free": "?", "percent": "?",
                     "total_bytes": 0, "used_bytes": 0, "free_bytes": 0, "percent_num": 0},
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


async def _bridge_get(path: str, timeout_s: int = 15) -> Optional[dict]:
    """Helper: GET a bridge endpoint. Returns the parsed JSON or None on failure."""
    from app.services.docker_host_service import _bridge_client
    try:
        async with _bridge_client(timeout_s=timeout_s) as client:
            r = await client.get(path)
            if r.status_code == 200:
                return r.json()
            logger.warning("bridge %s → HTTP %s: %s", path, r.status_code, r.text[:200])
    except Exception as e:
        logger.warning("bridge %s failed: %s", path, e)
    return None


async def _bridge_post(path: str, json_body: dict, timeout_s: int = 35) -> Optional[dict]:
    from app.services.docker_host_service import _bridge_client
    try:
        async with _bridge_client(timeout_s=timeout_s) as client:
            r = await client.post(path, json=json_body)
            if r.status_code == 200:
                return r.json()
            logger.warning("bridge POST %s → HTTP %s: %s", path, r.status_code, r.text[:200])
    except Exception as e:
        logger.warning("bridge POST %s failed: %s", path, e)
    return None


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

    # Delegate to bridge's ls endpoint (runs `find` inside the container).
    prefix = container.user_id[:8]
    from urllib.parse import quote
    data = await _bridge_get(f"/v1/tenants/{prefix}/ls?path={quote(path)}")
    if not data:
        return {"path": path, "files": [], "base": "/app", "error": "bridge unreachable"}

    files = []
    for f in data.get("files", []):
        files.append({
            "name": f["name"],
            "type": f["type"],
            "size": f["size"],
            "modified": datetime.fromtimestamp(f["mtime"]).isoformat() if f.get("mtime") else None,
        })
    return {"path": data.get("path", path), "files": files, "base": data.get("base", "/app")}


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

    prefix = container.user_id[:8]
    from urllib.parse import quote
    data = await _bridge_get(f"/v1/tenants/{prefix}/cat?path={quote(path)}")
    if not data:
        return {"error": "bridge unreachable or file not found", "path": path}
    return {"path": data["path"], "content": data["content"], "size": data["size"]}


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

    prefix = container.user_id[:8]
    data = await _bridge_post(f"/v1/tenants/{prefix}/exec", {"command": cmd})
    if not data:
        return {"output": "(bridge unreachable)", "command": cmd, "exit_code": -1}
    return {"output": data.get("output", ""), "command": cmd, "exit_code": data.get("exit_code", -1)}


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

    # Gather info in parallel via bridge
    prefix = container.user_id[:8]
    logs_task = _bridge_get(f"/v1/tenants/{prefix}/logs?tail=50")
    ps_task = _bridge_get(f"/v1/tenants/{prefix}/ps")
    disk_task = _bridge_get(f"/v1/tenants/{prefix}/disk")
    env_task = _bridge_get(f"/v1/tenants/{prefix}/env")

    logs_d, ps_d, disk_d, env_d = await asyncio.gather(
        logs_task, ps_task, disk_task, env_task
    )

    logs = (logs_d or {}).get("logs", "(bridge unreachable)")
    # docker top format → flatten to `ps aux`-like string
    if ps_d:
        titles = ps_d.get("titles") or []
        rows = ps_d.get("processes") or []
        processes = "\t".join(titles) + "\n" + "\n".join("\t".join(r) for r in rows)
    else:
        processes = "(bridge unreachable)"
    disk = f"workspace={(disk_d or {}).get('workspace_bytes', 0)} skills={(disk_d or {}).get('skills_bytes', 0)}"
    # Bridge already masks secrets
    env_lines = ((env_d or {}).get("env_masked") or "").split("\n")
    # Self-probe health via public URL
    from app.db.models import AgentConfig
    ac_row = (await db.execute(
        select(AgentConfig.agent_url).where(AgentConfig.user_id == container.user_id)
    )).first()
    health = "unknown"
    if ac_row and ac_row[0]:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5) as hc:
                r = await hc.get(f"{ac_row[0].rstrip('/')}/agent/health")
                health = f"HTTP {r.status_code}: {r.text[:100]}" if r.status_code != 200 else r.text[:200]
        except Exception as e:
            health = f"unreachable: {type(e).__name__}"

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
