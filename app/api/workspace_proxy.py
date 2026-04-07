"""
Workspace file browser proxy — lets authenticated users browse files
in their own VPS container's workspace directory.

Uses host-level SSH commands (not docker exec) because container volumes
may not be mounted. Files live at /data/agents/{user_prefix}/workspace/
on the Docker host.
"""

import io
import logging
import base64
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
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
    import asyncio
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
    proc = await asyncio.create_subprocess_exec(
        *ssh_args,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate(input=cmd.encode())
    return stdout.decode().strip()


def _host_workspace(user_id: str) -> str:
    """Return the host-level workspace path for a user."""
    return f"/data/agents/{user_id[:8]}/workspace"


def _container_workspace() -> str:
    """Return the workspace path inside the container."""
    return "/app/workspace"


async def _require_container(user_id: str, db: AsyncSession) -> ManagedContainer:
    """Get the user's managed container or raise 404."""
    result = await db.execute(
        select(ManagedContainer).where(
            ManagedContainer.user_id == user_id,
            ManagedContainer.status == "running",
        )
    )
    container = result.scalar_one_or_none()
    if not container:
        raise HTTPException(404, "No active container found")
    return container


# ── List files ───────────────────────────────────────────────────

@router.get("/files")
async def list_workspace_files(
    path: str = "",
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List files in the user's workspace directory."""
    await _require_container(current_user.id, db)
    base = _host_workspace(current_user.id)

    clean_path = path.strip("/").replace("..", "")
    target = f"{base}/{clean_path}" if clean_path else base

    raw = await _ssh_cmd(
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


# ── Read file content ────────────────────────────────────────────

@router.get("/file-content")
async def read_workspace_file(
    path: str = Query(..., description="File path relative to workspace"),
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Read a file from the user's workspace (tries container then host)."""
    container = await _require_container(current_user.id, db)
    host_base = _host_workspace(current_user.id)

    clean_path = path.strip("/").replace("..", "")
    if not clean_path:
        raise HTTPException(400, "Path is required")

    # Try container first (live files), fall back to host
    container_target = f"{_container_workspace()}/{clean_path}"
    size_check = await _ssh_cmd(
        f"docker exec {container.container_name} stat -c%s {container_target} 2>/dev/null"
    )
    if size_check and size_check.strip().isdigit():
        if int(size_check.strip()) > 1_000_000:
            raise HTTPException(413, "File too large (>1MB)")
        content = await _ssh_cmd(
            f"docker exec {container.container_name} cat {container_target} 2>/dev/null"
        )
        return {"path": clean_path, "content": content, "size": int(size_check.strip())}

    # Fall back to host path
    host_target = f"{host_base}/{clean_path}"
    size_check = await _ssh_cmd(f"stat -c%s {host_target} 2>/dev/null")
    if not size_check or not size_check.strip().isdigit():
        raise HTTPException(404, "File not found")
    if int(size_check.strip()) > 1_000_000:
        raise HTTPException(413, "File too large (>1MB)")

    content = await _ssh_cmd(f"cat {host_target} 2>/dev/null")
    return {"path": clean_path, "content": content, "size": int(size_check.strip())}


# ── Recursive tree ───────────────────────────────────────────────

@router.get("/tree")
async def workspace_tree(
    path: str = "",
    depth: int = Query(3, ge=1, le=5),
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get a recursive file tree of the workspace (checks both container and host)."""
    container = await _require_container(current_user.id, db)
    host_base = _host_workspace(current_user.id)
    container_base = _container_workspace()

    clean_path = path.strip("/").replace("..", "")

    # Try container first (live files from write_file), then host (legacy/mounted files)
    container_target = f"{container_base}/{clean_path}" if clean_path else container_base
    raw = await _ssh_cmd(
        f"docker exec {container.container_name} "
        f"find {container_target} -maxdepth {depth} -printf '%y|%s|%P\\n' 2>/dev/null | head -1000"
    )

    # Also check host path for files that may not be in container
    host_target = f"{host_base}/{clean_path}" if clean_path else host_base
    host_raw = await _ssh_cmd(
        f"find {host_target} -maxdepth {depth} -printf '%y|%s|%P\\n' 2>/dev/null | head -1000"
    )

    # Merge: container files take priority, host files fill gaps
    combined = (raw or "") + "\n" + (host_raw or "")
    if not combined.strip():
        return {"path": clean_path, "tree": [], "base": "/workspace"}

    seen = set()
    files = []
    for line in combined.strip().split("\n"):
        parts = line.split("|", 2)
        if len(parts) < 3 or not parts[2]:
            continue
        ftype = "dir" if parts[0] == "d" else "file"
        size = int(parts[1]) if parts[1].isdigit() else 0
        rel_path = parts[2]
        if rel_path in seen:
            continue
        seen.add(rel_path)
        name = rel_path.split("/")[-1]
        files.append({
            "name": name,
            "path": rel_path,
            "type": ftype,
            "size": size,
        })

    # Merge DB-only apps into the tree (apps that exist in DB but have no directory on disk).
    # This ensures failed builds that never created a folder still appear in the Explorer.
    # The apps table lives on the AGENT's database — we proxy to the agent's /api/apps/ endpoint.
    if not clean_path or clean_path == "apps":
        try:
            import httpx
            agent_url = f"http://{settings.docker_host_ip}:{container.host_port}"
            # Get the agent API key from AgentConfig
            from app.db.models import AgentConfig
            ac_result = await db.execute(
                select(AgentConfig.agent_api_key).where(
                    AgentConfig.user_id == current_user.id,
                    AgentConfig.deploy_status == "active",
                )
            )
            ac_row = ac_result.first()
            agent_key = ac_row.agent_api_key if ac_row else ""

            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(
                    f"{agent_url}/api/apps/",
                    headers={"X-Agent-Key": agent_key},
                )
                if resp.status_code == 200:
                    agent_apps = resp.json()
                    for app in agent_apps:
                        slug = app.get("slug", "")
                        source = app.get("source", "app_builder")
                        if not slug:
                            continue
                        # Derive workspace-relative path from source + slug
                        if source == "vibecoding":
                            rel = f"vibecoding/{slug}"
                        else:
                            rel = f"apps/{slug}"

                        if clean_path:
                            if not rel.startswith(clean_path + "/") and rel != clean_path:
                                continue
                            rel = rel[len(clean_path):].strip("/")

                        if rel and rel not in seen:
                            seen.add(rel)
                            name = rel.split("/")[-1]
                            files.append({
                                "name": name,
                                "path": rel if not clean_path else f"{clean_path}/{rel}",
                                "type": "dir",
                                "size": 0,
                                "status": app.get("status", ""),
                                "source": app.get("source", ""),
                            })
        except Exception as e:
            logger.warning("[WORKSPACE] Failed to merge agent apps into tree: %s", e)

    files.sort(key=lambda f: (0 if f["type"] == "dir" else 1, f["path"]))
    return {"path": clean_path, "tree": files, "base": "/workspace"}


# ── Write / Edit ─────────────────────────────────────────────────

class FileWriteRequest(BaseModel):
    path: str
    content: str


@router.post("/file-write")
async def write_workspace_file(
    body: FileWriteRequest,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Write (create or overwrite) a file in the user's workspace."""
    await _require_container(current_user.id, db)
    base = _host_workspace(current_user.id)

    clean_path = body.path.strip("/").replace("..", "")
    if not clean_path:
        raise HTTPException(400, "Path is required")
    target = f"{base}/{clean_path}"

    b64 = base64.b64encode(body.content.encode()).decode()
    await _ssh_cmd(
        f"mkdir -p \"$(dirname {target})\" && echo \"{b64}\" | base64 -d > {target}"
    )

    check = await _ssh_cmd(f"stat -c%s {target} 2>/dev/null")
    if not check or not check.strip().isdigit():
        raise HTTPException(500, "Failed to write file")

    return {"success": True, "path": clean_path, "size": int(check.strip())}


# ── Delete ───────────────────────────────────────────────────────

@router.delete("/file")
async def delete_workspace_file(
    path: str = Query(..., description="File or directory path relative to workspace"),
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a file or directory from the user's workspace."""
    await _require_container(current_user.id, db)
    base = _host_workspace(current_user.id)

    clean_path = path.strip("/").replace("..", "")
    if not clean_path:
        raise HTTPException(400, "Cannot delete workspace root")
    target = f"{base}/{clean_path}"

    exists = await _ssh_cmd(f"test -e {target} && echo yes || echo no")
    if exists != "yes":
        raise HTTPException(404, "File or directory not found")

    await _ssh_cmd(f"rm -rf {target}")
    return {"success": True, "path": clean_path}


# ── Rename / Move ────────────────────────────────────────────────

class FileRenameRequest(BaseModel):
    old_path: str
    new_path: str


@router.post("/file-rename")
async def rename_workspace_file(
    body: FileRenameRequest,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Rename or move a file/directory in the user's workspace."""
    await _require_container(current_user.id, db)
    base = _host_workspace(current_user.id)

    old_clean = body.old_path.strip("/").replace("..", "")
    new_clean = body.new_path.strip("/").replace("..", "")
    if not old_clean or not new_clean:
        raise HTTPException(400, "Both old_path and new_path are required")

    old_target = f"{base}/{old_clean}"
    new_target = f"{base}/{new_clean}"

    await _ssh_cmd(
        f"mkdir -p \"$(dirname {new_target})\" && mv {old_target} {new_target}"
    )

    check = await _ssh_cmd(f"test -e {new_target} && echo yes || echo no")
    if check != "yes":
        raise HTTPException(500, "Rename failed")

    return {"success": True, "old_path": old_clean, "new_path": new_clean}


# ── Create directory ─────────────────────────────────────────────

class CreateDirRequest(BaseModel):
    path: str


@router.post("/create-dir")
async def create_workspace_dir(
    body: CreateDirRequest,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a directory in the user's workspace."""
    await _require_container(current_user.id, db)
    base = _host_workspace(current_user.id)

    clean_path = body.path.strip("/").replace("..", "")
    if not clean_path:
        raise HTTPException(400, "Path is required")
    target = f"{base}/{clean_path}"

    await _ssh_cmd(f"mkdir -p {target}")

    check = await _ssh_cmd(f"test -d {target} && echo yes || echo no")
    if check != "yes":
        raise HTTPException(500, "Failed to create directory")

    return {"success": True, "path": clean_path}


# ── Download ─────────────────────────────────────────────────────

@router.get("/file-download")
async def download_workspace_file(
    path: str = Query(..., description="File path relative to workspace"),
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Download a file from the user's workspace."""
    await _require_container(current_user.id, db)
    base = _host_workspace(current_user.id)

    clean_path = path.strip("/").replace("..", "")
    if not clean_path:
        raise HTTPException(400, "Path is required")
    target = f"{base}/{clean_path}"

    size_check = await _ssh_cmd(f"stat -c%s {target} 2>/dev/null")
    if not size_check or not size_check.strip().isdigit():
        raise HTTPException(404, "File not found")
    if int(size_check.strip()) > 50_000_000:
        raise HTTPException(413, "File too large for download (>50MB)")

    b64_content = await _ssh_cmd(f"base64 {target} 2>/dev/null")
    if not b64_content:
        raise HTTPException(500, "Failed to read file")

    content_bytes = base64.b64decode(b64_content)
    filename = clean_path.split("/")[-1]

    return StreamingResponse(
        io.BytesIO(content_bytes),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
