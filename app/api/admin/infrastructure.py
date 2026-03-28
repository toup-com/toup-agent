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
    """Run a command on the Docker host and return stdout."""
    if not settings.docker_host_ip:
        return ""
    ssh_prefix = (
        f"sshpass -p '{settings.docker_host_ssh_password}' "
        if settings.docker_host_ssh_password else ""
    )
    full_cmd = (
        f"{ssh_prefix}ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 "
        f"-o PreferredAuthentications=password "
        f"root@{settings.docker_host_ip} '{cmd}'"
    )
    try:
        proc = await asyncio.create_subprocess_shell(
            full_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
        return stdout.decode().strip()
    except Exception as e:
        logger.warning(f"SSH command failed: {e}")
        return ""


@router.get("/overview")
async def overview(
    _=Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Get full VPS infrastructure overview."""

    # Get host stats via SSH (parallel)
    disk_task = _ssh_cmd("df -h / | tail -1 | awk '{print $2, $3, $4, $5}'")
    mem_task = _ssh_cmd("free -m | grep Mem | awk '{print $2, $3, $4}'")
    cpu_task = _ssh_cmd("nproc && cat /proc/loadavg | awk '{print $1, $2, $3}'")
    docker_task = _ssh_cmd("docker ps --format '{{.Names}}|{{.Status}}|{{.Ports}}|{{.Image}}' 2>/dev/null")
    uptime_task = _ssh_cmd("uptime -p")

    disk, mem, cpu, docker_ps, uptime = await asyncio.gather(
        disk_task, mem_task, cpu_task, docker_task, uptime_task
    )

    # Parse host stats
    disk_parts = disk.split() if disk else []
    mem_parts = mem.split() if mem else []
    cpu_parts = cpu.split() if cpu else []

    host = {
        "ip": settings.docker_host_ip,
        "uptime": uptime or "unknown",
        "disk": {
            "total": disk_parts[0] if len(disk_parts) > 0 else "?",
            "used": disk_parts[1] if len(disk_parts) > 1 else "?",
            "free": disk_parts[2] if len(disk_parts) > 2 else "?",
            "percent": disk_parts[3] if len(disk_parts) > 3 else "?",
        },
        "memory": {
            "total_mb": int(mem_parts[0]) if mem_parts and mem_parts[0].isdigit() else 0,
            "used_mb": int(mem_parts[1]) if len(mem_parts) > 1 and mem_parts[1].isdigit() else 0,
            "free_mb": int(mem_parts[2]) if len(mem_parts) > 2 and mem_parts[2].isdigit() else 0,
        },
        "cpu": {
            "cores": int(cpu_parts[0]) if cpu_parts and cpu_parts[0].isdigit() else 0,
            "load_1m": cpu_parts[1] if len(cpu_parts) > 1 else "?",
            "load_5m": cpu_parts[2] if len(cpu_parts) > 2 else "?",
            "load_15m": cpu_parts[3] if len(cpu_parts) > 3 else "?",
        },
    }

    # Parse running containers
    containers_running = []
    if docker_ps:
        for line in docker_ps.strip().split("\n"):
            parts = line.split("|")
            if len(parts) >= 3:
                containers_running.append({
                    "name": parts[0],
                    "status": parts[1],
                    "ports": parts[2],
                    "image": parts[3] if len(parts) > 3 else "",
                })

    # Get managed containers from DB
    result = await db.execute(
        select(
            ManagedContainer,
            User.name,
            User.email,
        )
        .outerjoin(User, ManagedContainer.user_id == User.id)
        .order_by(ManagedContainer.created_at.desc())
    )
    rows = result.all()

    managed = []
    for container, user_name, user_email in rows:
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

    # Count totals
    total_users = (await db.execute(select(func.count(User.id)))).scalar() or 0
    total_containers = len(managed)
    running_containers = sum(1 for c in managed if c["status"] == "running")

    # Memory per container via docker stats
    container_stats = {}
    if containers_running:
        stats_raw = await _ssh_cmd(
            "docker stats --no-stream --format '{{.Name}}|{{.CPUPerc}}|{{.MemUsage}}|{{.MemPerc}}' 2>/dev/null"
        )
        if stats_raw:
            for line in stats_raw.strip().split("\n"):
                parts = line.split("|")
                if len(parts) >= 4:
                    container_stats[parts[0]] = {
                        "cpu": parts[1],
                        "mem_usage": parts[2],
                        "mem_percent": parts[3],
                    }

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


@router.get("/containers/{container_id}/files")
async def list_files(
    container_id: str,
    path: str = "",
    _=Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Browse files in a managed container's workspace and skills."""
    result = await db.execute(
        select(ManagedContainer).where(ManagedContainer.id == container_id)
    )
    container = result.scalar_one_or_none()
    if not container:
        return {"error": "Not found"}

    user_prefix = container.user_id[:8]
    base = f"/data/agents/{user_prefix}"

    # Sanitize path to prevent directory traversal
    clean_path = path.strip("/").replace("..", "")
    target = f"{base}/{clean_path}" if clean_path else base

    # List files via SSH
    raw = await _ssh_cmd(
        f"find {target} -maxdepth 1 -printf '%y|%s|%T@|%f\\n' 2>/dev/null | head -200"
    )
    if not raw:
        return {"path": clean_path, "files": [], "base": base}

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

    return {"path": clean_path, "files": files, "base": base}


@router.get("/containers/{container_id}/file-content")
async def read_file(
    container_id: str,
    path: str,
    _=Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Read a file's content from a managed container's data directory."""
    result = await db.execute(
        select(ManagedContainer).where(ManagedContainer.id == container_id)
    )
    container = result.scalar_one_or_none()
    if not container:
        return {"error": "Not found"}

    user_prefix = container.user_id[:8]
    clean_path = path.strip("/").replace("..", "")
    target = f"/data/agents/{user_prefix}/{clean_path}"

    # Only allow text files under 1MB
    size_check = await _ssh_cmd(f"stat -c%s {target} 2>/dev/null")
    if not size_check or not size_check.isdigit() or int(size_check) > 1_000_000:
        return {"error": "File too large or not found", "path": clean_path}

    content = await _ssh_cmd(f"cat {target} 2>/dev/null")
    return {"path": clean_path, "content": content, "size": int(size_check)}
