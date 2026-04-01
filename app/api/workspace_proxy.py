"""
Workspace file browser proxy — lets authenticated users browse files
in their own VPS container's workspace directory.

Proxies via SSH + docker exec (same as admin infrastructure but scoped
to the authenticated user's container only).
"""

import logging
from datetime import datetime
from typing import Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import get_current_user
from app.db import get_db
from app.db.models import ManagedContainer
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/workspace", tags=["Workspace"])


async def _ssh_cmd(cmd: str) -> str:
    """Run a command on the Docker host via SSH."""
    if not settings.docker_host_ip:
        return ""
    from app.services.docker_host_service import _get_ssh_key_file
    key_file = _get_ssh_key_file()
    ssh_args = ["ssh"]
    if key_file:
        ssh_args += ["-i", key_file, "-o", "PasswordAuthentication=no"]
    ssh_args += [
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=5",
        f"root@{settings.docker_host_ip}",
        "bash", "-s",
    ]
    import asyncio
    proc = await asyncio.create_subprocess_exec(
        *ssh_args,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate(input=cmd.encode())
    return stdout.decode().strip()


async def _get_user_container(
    user_id: str, db: AsyncSession
) -> Optional[ManagedContainer]:
    """Get the user's running managed container."""
    result = await db.execute(
        select(ManagedContainer).where(
            ManagedContainer.user_id == user_id,
            ManagedContainer.status == "running",
        )
    )
    return result.scalar_one_or_none()


@router.get("/files")
async def list_workspace_files(
    path: str = "",
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List files in the user's workspace directory."""
    container = await _get_user_container(current_user.id, db)
    if not container:
        raise HTTPException(404, "No active container found")

    # Sanitize path — only allow browsing within /app/workspace
    clean_path = path.strip("/").replace("..", "")
    target = f"/app/workspace/{clean_path}" if clean_path else "/app/workspace"

    raw = await _ssh_cmd(
        f"docker exec {container.container_name} "
        f"find {target} -maxdepth 1 -printf '%y|%s|%T@|%f\\n' 2>/dev/null | head -500"
    )
    if not raw:
        return {"path": clean_path, "files": [], "base": "/workspace"}

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

    files.sort(key=lambda f: (0 if f["type"] == "dir" else 1, f["name"]))
    return {"path": clean_path, "files": files, "base": "/workspace"}


@router.get("/file-content")
async def read_workspace_file(
    path: str = Query(..., description="File path relative to workspace"),
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Read a file from the user's workspace."""
    container = await _get_user_container(current_user.id, db)
    if not container:
        raise HTTPException(404, "No active container found")

    clean_path = path.strip("/").replace("..", "")
    if not clean_path:
        raise HTTPException(400, "Path is required")
    target = f"/app/workspace/{clean_path}"

    # Check file size
    size_check = await _ssh_cmd(
        f"docker exec {container.container_name} stat -c%s {target} 2>/dev/null"
    )
    if not size_check or not size_check.strip().isdigit():
        raise HTTPException(404, "File not found")
    if int(size_check.strip()) > 1_000_000:
        raise HTTPException(413, "File too large (>1MB)")

    content = await _ssh_cmd(
        f"docker exec {container.container_name} cat {target} 2>/dev/null"
    )
    return {"path": clean_path, "content": content, "size": int(size_check.strip())}


@router.get("/tree")
async def workspace_tree(
    path: str = "",
    depth: int = Query(3, ge=1, le=5),
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get a recursive file tree of the workspace (up to `depth` levels)."""
    container = await _get_user_container(current_user.id, db)
    if not container:
        raise HTTPException(404, "No active container found")

    clean_path = path.strip("/").replace("..", "")
    target = f"/app/workspace/{clean_path}" if clean_path else "/app/workspace"

    # Get recursive listing with depth limit
    raw = await _ssh_cmd(
        f"docker exec {container.container_name} "
        f"find {target} -maxdepth {depth} -printf '%y|%s|%P\\n' 2>/dev/null | head -1000"
    )
    if not raw:
        return {"path": clean_path, "tree": [], "base": "/workspace"}

    files = []
    for line in raw.strip().split("\n"):
        parts = line.split("|", 2)
        if len(parts) < 3 or not parts[2]:
            continue
        ftype = "dir" if parts[0] == "d" else "file"
        size = int(parts[1]) if parts[1].isdigit() else 0
        rel_path = parts[2]
        name = rel_path.split("/")[-1]
        files.append({
            "name": name,
            "path": rel_path,
            "type": ftype,
            "size": size,
        })

    files.sort(key=lambda f: (0 if f["type"] == "dir" else 1, f["path"]))
    return {"path": clean_path, "tree": files, "base": "/workspace"}
