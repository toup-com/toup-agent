"""
AppSkill — Auto-generated virtual skill for each Builder app.

Each saved app (Workflow with app_component node) gets an AppSkill instance
that lets the agent inspect, read, and modify the app's files via DB-backed tools.
No filesystem files needed — the skill reads/writes directly from Workflow.nodes_json.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from app.agent.skills.base import Skill, SkillContext, SkillMeta

logger = logging.getLogger(__name__)


def slugify(name: str, workflow_id: str = "") -> str:
    """Generate a unique, deterministic slug for an app skill."""
    slug = name[:40].lower().replace(" ", "_").replace("-", "_")
    slug = re.sub(r"[^a-z0-9_]", "", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    if not slug:
        slug = "app"
    if workflow_id:
        slug = f"{slug}_{workflow_id[:6]}"
    return slug


def _find_app_node(nodes: list) -> Optional[dict]:
    """Find the app_component node in a list of workflow nodes."""
    for node in nodes:
        if node.get("type") == "app_component" or \
           node.get("data", {}).get("templateType") == "app_component":
            return node
    return None


class AppSkill(Skill):
    """Virtual skill for a Builder app, backed by a Workflow row in the DB."""

    def __init__(self, workflow_id: str, app_name: str, app_slug: str):
        self.workflow_id = workflow_id
        self.app_name = app_name
        self.app_slug = app_slug
        self.meta = SkillMeta(
            name=f"app_{app_slug}",
            version="1.0.0",
            description=f"Tools for the '{app_name}' app",
            author="auto-generated",
        )

    # ── DB access helpers ────────────────────────────────────────

    async def _get_app_data(self) -> tuple[dict, dict]:
        """Return (files_dict, dependencies_dict) from the workflow."""
        from app.db.database import async_session_maker
        from app.db.models import Workflow

        async with async_session_maker() as db:
            wf = await db.get(Workflow, self.workflow_id)
            if not wf or not wf.nodes_json:
                return {}, {}
            nodes = json.loads(wf.nodes_json)
            node = _find_app_node(nodes)
            if not node:
                return {}, {}
            data = node.get("data", {})
            files = data.get("files", {})
            # Legacy single-file fallback
            if not files and data.get("code"):
                files = {"/App.js": data["code"]}
            deps = data.get("dependencies", {})
            return files, deps

    async def _write_app_files(self, updated_files: dict) -> bool:
        """Merge updated files into the app's nodes_json and commit."""
        from app.db.database import async_session_maker
        from app.db.models import Workflow

        async with async_session_maker() as db:
            wf = await db.get(Workflow, self.workflow_id)
            if not wf or not wf.nodes_json:
                return False
            nodes = json.loads(wf.nodes_json)
            node = _find_app_node(nodes)
            if not node:
                return False

            data = node.get("data", {})
            files = data.get("files", {})
            files.update(updated_files)
            data["files"] = files
            node["data"] = data
            wf.nodes_json = json.dumps(nodes)
            await db.commit()
            return True

    # ── Skill interface ──────────────────────────────────────────

    def get_tools(self) -> List[Dict[str, Any]]:
        p = self._prefix
        return [
            {
                "name": p("list_files"),
                "description": f"List all files in the '{self.app_name}' app with their sizes.",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": p("read_file"),
                "description": f"Read a specific file from the '{self.app_name}' app.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "File path (e.g. '/App.js', '/pages/Dashboard.jsx')",
                        },
                    },
                    "required": ["file_path"],
                },
            },
            {
                "name": p("write_file"),
                "description": f"Update or create a file in the '{self.app_name}' app. Persists to database.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "File path (e.g. '/App.js')",
                        },
                        "content": {
                            "type": "string",
                            "description": "Full file content to write",
                        },
                    },
                    "required": ["file_path", "content"],
                },
            },
            {
                "name": p("get_structure"),
                "description": f"Get the '{self.app_name}' app structure: file tree, dependencies, and component summary.",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": p("get_dependencies"),
                "description": f"List npm dependencies used by the '{self.app_name}' app.",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
        ]

    def _prefix(self, name: str) -> str:
        return f"{self.meta.name}__{name}"

    async def execute_tool(
        self, tool_name: str, args: Dict[str, Any], ctx: SkillContext
    ) -> str:
        dispatch = {
            self._prefix("list_files"): self._list_files,
            self._prefix("read_file"): self._read_file,
            self._prefix("write_file"): self._write_file,
            self._prefix("get_structure"): self._get_structure,
            self._prefix("get_dependencies"): self._get_dependencies,
        }
        handler = dispatch.get(tool_name)
        if not handler:
            return f"ERROR: Unknown tool: {tool_name}"
        return await handler(args, ctx)

    def get_system_prompt_section(self) -> Optional[str]:
        return (
            f"# App: {self.app_name}\n"
            f"You can inspect and modify this app using `{self.meta.name}__*` tools:\n"
            f"- `{self._prefix('list_files')}` — see all files\n"
            f"- `{self._prefix('read_file')}` — read a file\n"
            f"- `{self._prefix('write_file')}` — update a file\n"
            f"- `{self._prefix('get_structure')}` — understand the app\n"
            f"- `{self._prefix('get_dependencies')}` — list packages\n"
        )

    # ── Tool handlers ────────────────────────────────────────────

    async def _list_files(self, args: Dict, ctx: SkillContext) -> str:
        files, _ = await self._get_app_data()
        if not files:
            return "No files found in this app."
        lines = [f"Files in '{self.app_name}':"]
        for path, code in sorted(files.items()):
            lines.append(f"  {path}  ({len(code)} chars)")
        return "\n".join(lines)

    async def _read_file(self, args: Dict, ctx: SkillContext) -> str:
        file_path = args.get("file_path", "")
        if not file_path:
            return "ERROR: file_path is required"
        files, _ = await self._get_app_data()
        if file_path not in files:
            available = ", ".join(sorted(files.keys()))
            return f"File '{file_path}' not found. Available: {available}"
        return files[file_path]

    async def _write_file(self, args: Dict, ctx: SkillContext) -> str:
        file_path = args.get("file_path", "")
        content = args.get("content", "")
        if not file_path:
            return "ERROR: file_path is required"
        if not content:
            return "ERROR: content is required"
        ok = await self._write_app_files({file_path: content})
        if ok:
            return f"Updated '{file_path}' in '{self.app_name}' ({len(content)} chars)"
        return f"ERROR: Failed to write to '{file_path}'"

    async def _get_structure(self, args: Dict, ctx: SkillContext) -> str:
        files, deps = await self._get_app_data()
        if not files:
            return "No app data found."
        lines = [
            f"App: {self.app_name}",
            f"Workflow ID: {self.workflow_id}",
            f"Files ({len(files)}):",
        ]
        for path in sorted(files.keys()):
            lines.append(f"  {path}")
        if deps:
            lines.append(f"\nDependencies ({len(deps)}):")
            for pkg, ver in sorted(deps.items()):
                lines.append(f"  {pkg}: {ver}")
        # Detect components (exports)
        components = []
        for path, code in files.items():
            if "export default" in code:
                match = re.search(r"export default (?:function |class )?(\w+)", code)
                if match:
                    components.append(f"  {path} → {match.group(1)}")
        if components:
            lines.append(f"\nComponents:")
            lines.extend(components)
        return "\n".join(lines)

    async def _get_dependencies(self, args: Dict, ctx: SkillContext) -> str:
        _, deps = await self._get_app_data()
        if not deps:
            return "No dependencies."
        lines = [f"Dependencies for '{self.app_name}':"]
        for pkg, ver in sorted(deps.items()):
            lines.append(f"  {pkg}: {ver}")
        return "\n".join(lines)


# ── Meta skill: list all apps (registered once) ─────────────────

class AppMetaSkill(Skill):
    """Single skill that lists all builder apps. Registered once, not per-app."""

    meta = SkillMeta(
        name="app",
        version="1.0.0",
        description="List all builder apps",
        author="auto-generated",
    )

    def __init__(self, skill_loader=None):
        self._skill_loader = skill_loader

    def get_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "app__list_all_apps",
                "description": "List all builder apps and their available tool namespaces.",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
        ]

    async def execute_tool(
        self, tool_name: str, args: Dict[str, Any], ctx: SkillContext
    ) -> str:
        if tool_name != "app__list_all_apps":
            return f"ERROR: Unknown tool: {tool_name}"
        if not self._skill_loader:
            return "No skill loader available."
        apps = []
        for name, skill in self._skill_loader.skills.items():
            if name.startswith("app_") and name != "app" and isinstance(skill, AppSkill):
                apps.append(f"- {skill.app_name} (tools: {skill.meta.name}__*)")
        if not apps:
            return "No builder apps found."
        return "Builder apps:\n" + "\n".join(apps)

    def get_system_prompt_section(self) -> Optional[str]:
        return (
            "# Builder Apps\n"
            "Use `app__list_all_apps` to see all apps the user has built.\n"
            "Each app has tools to inspect and modify its files.\n"
        )


# ── Startup loader ──────────────────────────────────────────────

async def load_app_skills_from_db(skill_loader) -> int:
    """Scan workflows table for app_component workflows and register AppSkills."""
    from app.db.database import async_session_maker
    from app.db.models import Workflow
    from sqlalchemy import select

    count = 0
    try:
        async with async_session_maker() as db:
            result = await db.execute(select(Workflow))
            workflows = result.scalars().all()
            for wf in workflows:
                try:
                    nodes = json.loads(wf.nodes_json or "[]")
                    node = _find_app_node(nodes)
                    if node:
                        slug = slugify(wf.name, wf.id)
                        skill = AppSkill(wf.id, wf.name, slug)
                        await skill_loader.register_dynamic(skill)
                        count += 1
                except Exception as e:
                    logger.debug(f"[APP-SKILL] Skipping workflow {wf.id}: {e}")
    except Exception as e:
        logger.warning(f"[APP-SKILL] DB scan failed: {e}")

    # Register the meta skill (list all apps)
    try:
        meta_skill = AppMetaSkill(skill_loader=skill_loader)
        await skill_loader.register_dynamic(meta_skill)
    except Exception as e:
        logger.debug(f"[APP-SKILL] Meta skill already registered: {e}")

    return count
