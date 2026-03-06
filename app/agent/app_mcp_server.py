"""
App MCP Server — exposes builder app tools via FastMCP.

Mounted at /api/app-mcp on the agent. Provides MCP tools for
external clients (Claude Desktop, other agents) to interact
with user-built apps.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from fastmcp import FastMCP

logger = logging.getLogger(__name__)

app_mcp = FastMCP(
    "Toup App Tools",
    instructions="MCP server for inspecting and modifying user-built apps.",
)

_skill_loader = None


def set_mcp_skill_loader(loader):
    """Wire the skill loader so MCP tools can dispatch to AppSkill instances."""
    global _skill_loader
    _skill_loader = loader


def _get_app_skill(app_slug: str):
    """Find an AppSkill by slug."""
    if not _skill_loader:
        return None
    skill_name = f"app_{app_slug}"
    return _skill_loader.get_skill(skill_name)


@app_mcp.tool()
async def list_apps() -> dict:
    """List all builder apps and their tool namespaces."""
    if not _skill_loader:
        return {"apps": []}
    from app.agent.skills.builtins.app_skill import AppSkill
    apps = []
    for name, skill in _skill_loader.skills.items():
        if isinstance(skill, AppSkill):
            apps.append({
                "slug": skill.app_slug,
                "name": skill.app_name,
                "skill_name": skill.meta.name,
                "workflow_id": skill.workflow_id,
            })
    return {"apps": apps}


@app_mcp.tool()
async def app_list_files(app_slug: str) -> dict:
    """List all files in a builder app."""
    skill = _get_app_skill(app_slug)
    if not skill:
        return {"error": f"App '{app_slug}' not found"}
    files, _ = await skill._get_app_data()
    return {"files": {p: len(c) for p, c in files.items()}}


@app_mcp.tool()
async def app_read_file(app_slug: str, file_path: str) -> dict:
    """Read a file from a builder app."""
    skill = _get_app_skill(app_slug)
    if not skill:
        return {"error": f"App '{app_slug}' not found"}
    files, _ = await skill._get_app_data()
    if file_path not in files:
        return {"error": f"File '{file_path}' not found", "available": list(files.keys())}
    return {"path": file_path, "content": files[file_path]}


@app_mcp.tool()
async def app_write_file(app_slug: str, file_path: str, content: str) -> dict:
    """Write/update a file in a builder app. Persists to database."""
    skill = _get_app_skill(app_slug)
    if not skill:
        return {"error": f"App '{app_slug}' not found"}
    ok = await skill._write_app_files({file_path: content})
    if ok:
        return {"success": True, "path": file_path, "chars": len(content)}
    return {"error": f"Failed to write '{file_path}'"}


@app_mcp.tool()
async def app_get_structure(app_slug: str) -> dict:
    """Get the file structure and metadata of a builder app."""
    skill = _get_app_skill(app_slug)
    if not skill:
        return {"error": f"App '{app_slug}' not found"}
    files, deps = await skill._get_app_data()
    return {
        "name": skill.app_name,
        "workflow_id": skill.workflow_id,
        "files": list(files.keys()),
        "file_count": len(files),
        "dependencies": deps,
        "dependency_count": len(deps),
    }
