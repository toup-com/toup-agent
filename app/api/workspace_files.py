"""
Workspace files API — runs on the VPS agent (data owner).

Lists and reads files the agent wrote to its own workspace (reports,
notes, generated documents) so the platform proxy and the mobile
Workspace screen can surface them. Pool containers have no host bind
for /app/workspace, so the agent itself is the only process that can
see live files — the platform's SSH host-path listing cannot.

Auth: AgentAPIKeyMiddleware in agent_main.py validates X-Agent-Key on
every /api route, so no per-route dependency is needed (same posture
as dashboard.py).
"""

import logging
import mimetypes
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/workspace", tags=["Workspace Files"])

# Listing cap — a runaway workspace (node_modules, build output) must
# not serialize thousands of entries into a single response.
_MAX_ENTRIES = 500

# Content cap — this route serves text reports into a JSON field.
# Larger blobs go through the platform's download/preview routes.
_MAX_READ_BYTES = 1_000_000

# Non-text/* MIME types that are still plain text and safe to return
# in a JSON content field.
_TEXT_MIMES = {
    "application/json",
    "application/javascript",
    "application/xml",
    "application/x-sh",
    "application/x-yaml",
    "application/toml",
    "image/svg+xml",
}


def _workspace_root() -> Path:
    base = getattr(settings, "agent_workspace_dir", None) or "./workspace"
    return Path(base).resolve()


def resolve_workspace_path(path: str) -> Path:
    """Resolve a client-supplied path against the workspace root.

    resolve() collapses ``..`` segments and follows symlinks BEFORE the
    containment check, so both traversal and symlink escapes fail
    is_relative_to(). Absolute paths also fail it (``base / "/etc"``
    is just ``/etc``). Escapes return 404 — never 403 — so callers
    can't confirm that paths outside the workspace exist.
    """
    base = _workspace_root()
    target = (base / path).resolve()
    if not target.is_relative_to(base):
        raise HTTPException(404, "Not found")
    return target


def _iso_mtime(st) -> str:
    return datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()


# ── List directory ───────────────────────────────────────────────────

@router.get("/files")
async def list_workspace_files(
    path: str = Query("", description="Directory path relative to the workspace root"),
):
    """List one directory level of the workspace (all file types)."""
    target = resolve_workspace_path(path)
    if not target.is_dir():
        raise HTTPException(404, "Directory not found")

    files = []
    for child in target.iterdir():
        # Dotfiles (.dashboard, .git, …) are agent internals, not user documents
        if child.name.startswith("."):
            continue
        try:
            st = child.stat()
        except OSError:
            continue  # broken symlink / raced deletion
        files.append({
            "name": child.name,
            "type": "dir" if child.is_dir() else "file",
            "size": st.st_size,
            "modified": _iso_mtime(st),
        })
        if len(files) >= _MAX_ENTRIES:
            break

    files.sort(key=lambda f: (0 if f["type"] == "dir" else 1, f["name"]))
    base = _workspace_root()
    rel = "" if target == base else str(target.relative_to(base))
    # Shape matches the platform's /api/workspace/files response so the
    # proxy can return this body verbatim.
    return {"path": rel, "files": files, "base": "/workspace"}


# ── Read file content ────────────────────────────────────────────────

@router.get("/file")
async def read_workspace_file(
    path: str = Query(..., description="File path relative to the workspace root"),
):
    """Read a text file (report, note, markdown) from the workspace."""
    target = resolve_workspace_path(path)
    if not target.is_file():
        raise HTTPException(404, "File not found")

    st = target.stat()
    if st.st_size > _MAX_READ_BYTES:
        raise HTTPException(413, "File too large (>1MB)")

    # Guessed-binary types can't round-trip through a JSON content
    # field — callers use the preview/download routes for those.
    # Unknown extensions fall through as text (read with errors=replace).
    mime, _ = mimetypes.guess_type(target.name)
    if mime and not mime.startswith("text/") and mime not in _TEXT_MIMES:
        raise HTTPException(415, f"Binary file ({mime}) — use the preview/download route")

    content = target.read_text(encoding="utf-8", errors="replace")
    return {
        "path": str(target.relative_to(_workspace_root())),
        "content": content,
        "size": st.st_size,
        "mime": mime or "text/plain",
        "modified": _iso_mtime(st),
    }
