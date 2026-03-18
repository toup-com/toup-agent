"""
App MCP Server — exposes builder app tools via FastMCP.

Mounted at /api/app-mcp on the agent. Provides MCP tools for
external clients (Claude Desktop, other agents) to interact
with user-built filesystem-backed apps (AppFsSkill).
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
    """Wire the skill loader so MCP tools can dispatch to AppFsSkill instances."""
    global _skill_loader
    _skill_loader = loader


def _get_app_skill(app_slug: str):
    """Find an AppFsSkill by slug."""
    if not _skill_loader:
        return None
    skill_name = f"app_{app_slug}"
    return _skill_loader.get_skill(skill_name)


@app_mcp.tool()
async def list_apps() -> dict:
    """List all builder apps and their tool namespaces."""
    if not _skill_loader:
        return {"apps": []}
    try:
        from app.agent.skills.builtins.app_builder.app_fs_skill import AppFsSkill
    except ImportError:
        AppFsSkill = None
    apps = []
    for name, skill in _skill_loader.skills.items():
        if AppFsSkill and isinstance(skill, AppFsSkill):
            apps.append({
                "slug": skill.app_slug,
                "name": skill.app_name,
                "skill_name": skill.meta.name,
                "type": "filesystem",
                "app_dir": skill.app_dir,
            })
    return {"apps": apps}


@app_mcp.tool()
async def app_list_files(app_slug: str) -> dict:
    """List all files in a builder app."""
    skill = _get_app_skill(app_slug)
    if not skill:
        return {"error": f"App '{app_slug}' not found"}
    return await skill.execute_tool(f"app_{app_slug}__list_files", {})


@app_mcp.tool()
async def app_read_file(app_slug: str, file_path: str) -> dict:
    """Read a file from a builder app."""
    skill = _get_app_skill(app_slug)
    if not skill:
        return {"error": f"App '{app_slug}' not found"}
    return await skill.execute_tool(f"app_{app_slug}__read_file", {"path": file_path})


@app_mcp.tool()
async def app_write_file(app_slug: str, file_path: str, content: str) -> dict:
    """Write/update a file in a builder app."""
    skill = _get_app_skill(app_slug)
    if not skill:
        return {"error": f"App '{app_slug}' not found"}
    return await skill.execute_tool(f"app_{app_slug}__write_file", {"path": file_path, "content": content})


@app_mcp.tool()
async def app_edit_file(app_slug: str, file_path: str, old_text: str, new_text: str) -> dict:
    """Search-and-replace edit on a file in a builder app."""
    skill = _get_app_skill(app_slug)
    if not skill:
        return {"error": f"App '{app_slug}' not found"}
    return await skill.execute_tool(f"app_{app_slug}__edit_file", {
        "path": file_path, "old_text": old_text, "new_text": new_text,
    })


@app_mcp.tool()
async def app_get_structure(app_slug: str) -> dict:
    """Get the file structure and metadata of a builder app."""
    skill = _get_app_skill(app_slug)
    if not skill:
        return {"error": f"App '{app_slug}' not found"}
    return await skill.execute_tool(f"app_{app_slug}__status", {})


@app_mcp.tool()
async def app_query_db(app_slug: str, sql: str) -> dict:
    """Execute a SQL query against an app's SQLite database."""
    skill = _get_app_skill(app_slug)
    if not skill:
        return {"error": f"App '{app_slug}' not found"}
    return await skill.execute_tool(f"app_{app_slug}__query_db", {"sql": sql})


@app_mcp.tool()
async def app_logs(app_slug: str, lines: int = 50) -> dict:
    """Tail Metro/web server logs for an app."""
    skill = _get_app_skill(app_slug)
    if not skill:
        return {"error": f"App '{app_slug}' not found"}
    return await skill.execute_tool(f"app_{app_slug}__logs", {"lines": lines})


@app_mcp.tool()
async def app_restart(app_slug: str) -> dict:
    """Restart Metro/web servers for an app."""
    skill = _get_app_skill(app_slug)
    if not skill:
        return {"error": f"App '{app_slug}' not found"}
    return await skill.execute_tool(f"app_{app_slug}__restart", {})


@app_mcp.tool()
async def app_status(app_slug: str) -> dict:
    """Get status of an app (processes, ports, URLs, disk usage)."""
    skill = _get_app_skill(app_slug)
    if not skill:
        return {"error": f"App '{app_slug}' not found"}
    return await skill.execute_tool(f"app_{app_slug}__status", {})


@app_mcp.tool()
async def app_navigate(app_slug: str, screen: str = "", params: dict = {}) -> dict:
    """Navigate to a screen within an app, or list available screens."""
    skill = _get_app_skill(app_slug)
    if not skill:
        return {"error": f"App '{app_slug}' not found"}
    tool_args = {}
    if screen:
        tool_args["screen"] = screen
    if params:
        tool_args["params"] = params
    return await skill.execute_tool(f"app_{app_slug}__navigate", tool_args)


@app_mcp.tool()
async def app_git_push(app_slug: str, message: str = "Update from MCP") -> dict:
    """Commit and push changes for an app to GitHub."""
    skill = _get_app_skill(app_slug)
    if not skill:
        return {"error": f"App '{app_slug}' not found"}
    return await skill.execute_tool(f"app_{app_slug}__git_push", {"message": message})
