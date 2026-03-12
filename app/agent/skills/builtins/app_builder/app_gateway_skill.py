"""
AppGatewaySkill — Single skill that provides generic app tools + domain actions.

Instead of registering 11 tools PER APP (which explodes with many apps),
this registers 13 tools TOTAL that take app_slug as a parameter.
AppFsSkill instances are stored internally and dispatched to by slug.

Generic tools (12):
  app__list, app__status, app__list_files, app__read_file, app__write_file,
  app__edit_file, app__query_db, app__logs, app__restart, app__navigate,
  app__git_push, app__publish_web

Domain-specific tool (1):
  app__action — Execute domain actions defined in each app's /lib/agentSkill.json

Each app's agentSkill.json is generated at build time by the LLM and defines
domain-specific operations (SQL queries, mutations, navigation) that the agent
can invoke by name.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from app.agent.skills.base import Skill, SkillContext, SkillMeta

logger = logging.getLogger(__name__)


class AppGatewaySkill(Skill):
    """Single skill gateway for all builder apps."""

    meta = SkillMeta(
        name="app",
        version="3.0.0",
        description="Manage and interact with all builder apps",
        author="toup",
    )

    def __init__(self):
        self._apps: Dict[str, Any] = {}  # slug -> AppFsSkill instance
        self._manifests: Dict[str, Dict] = {}  # slug -> parsed agentSkill.json

    # ── App Registration ───────────────────────────────────────────

    def register_app(self, slug: str, fs_skill):
        """Register an AppFsSkill instance for a slug."""
        self._apps[slug] = fs_skill
        logger.info(f"[AppGateway] Registered app: {slug} ({fs_skill.app_name})")
        # Auto-load domain manifest
        if hasattr(fs_skill, 'app_dir'):
            self.load_app_manifest(slug, fs_skill.app_dir)

    def unregister_app(self, slug: str):
        """Remove an app from the gateway."""
        self._apps.pop(slug, None)
        self._manifests.pop(slug, None)

    def _resolve(self, slug: str):
        """Find app by slug (case-insensitive fallback)."""
        if slug in self._apps:
            return self._apps[slug]
        for k, v in self._apps.items():
            if k.lower() == slug.lower():
                return v
        return None

    # ── Manifest Loading ───────────────────────────────────────────

    def load_app_manifest(self, slug: str, app_dir: str):
        """Load /lib/agentSkill.json from an app's directory."""
        manifest_path = os.path.join(app_dir, "lib", "agentSkill.json")
        if not os.path.exists(manifest_path):
            return
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            if "actions" in manifest and isinstance(manifest["actions"], list):
                self._manifests[slug] = manifest
                logger.info(
                    f"[AppGateway] Loaded {len(manifest['actions'])} domain actions "
                    f"for '{slug}' ({manifest.get('domain', 'unknown')})"
                )
            else:
                logger.warning(f"[AppGateway] Invalid manifest for '{slug}': missing 'actions' list")
        except Exception as e:
            logger.warning(f"[AppGateway] Failed to load manifest for '{slug}': {e}")

    def get_app_actions_summary(self) -> List[Dict]:
        """Return summary of all app domain actions (for /agent/capabilities)."""
        result = []
        for slug, manifest in sorted(self._manifests.items()):
            fs = self._apps.get(slug)
            result.append({
                "app_slug": slug,
                "app_name": fs.app_name if fs else slug,
                "domain": manifest.get("domain", ""),
                "description": manifest.get("description", ""),
                "actions": [
                    {
                        "name": a["name"],
                        "description": a.get("description", ""),
                        "type": a.get("type", "query"),
                        "params": list(a.get("params", {}).keys()),
                    }
                    for a in manifest.get("actions", [])
                ],
            })
        return result

    # ── Tool Definitions (13 tools, always) ─────────────────────

    def _action_description(self) -> str:
        """Build dynamic description listing available domain actions."""
        base = "Execute a domain-specific action within an app."
        if not self._manifests:
            return base + " No domain actions loaded yet — build an app first."
        parts = [base, "\nAvailable actions:"]
        for slug, manifest in sorted(self._manifests.items()):
            domain = manifest.get("domain", slug)
            actions = manifest.get("actions", [])
            names = [a["name"] for a in actions]
            parts.append(f"  {slug} ({domain}): {', '.join(names)}")
        return "\n".join(parts)

    def get_tools(self) -> List[Dict[str, Any]]:
        app_enum_desc = "App slug"
        if self._apps:
            slugs = ", ".join(sorted(self._apps.keys()))
            app_enum_desc = f"App slug. Available: {slugs}"

        return [
            {
                "name": "app__list",
                "description": "List all builder apps with their slugs, status, and available operations.",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "app__status",
                "description": "Get status of an app (ports, URLs, processes, disk usage).",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "app": {"type": "string", "description": app_enum_desc},
                    },
                    "required": ["app"],
                },
            },
            {
                "name": "app__list_files",
                "description": "List all files in an app with sizes.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "app": {"type": "string", "description": app_enum_desc},
                    },
                    "required": ["app"],
                },
            },
            {
                "name": "app__read_file",
                "description": "Read a file from an app (with line numbers).",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "app": {"type": "string", "description": app_enum_desc},
                        "path": {"type": "string", "description": "File path (e.g. 'App.tsx', 'screens/Home.tsx')"},
                    },
                    "required": ["app", "path"],
                },
            },
            {
                "name": "app__write_file",
                "description": "Write/update a file in an app. Metro auto-hot-reloads.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "app": {"type": "string", "description": app_enum_desc},
                        "path": {"type": "string", "description": "File path"},
                        "content": {"type": "string", "description": "Full file content"},
                    },
                    "required": ["app", "path", "content"],
                },
            },
            {
                "name": "app__edit_file",
                "description": "Search/replace edit on a file in an app.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "app": {"type": "string", "description": app_enum_desc},
                        "path": {"type": "string", "description": "File path"},
                        "old_text": {"type": "string", "description": "Text to find (must match exactly)"},
                        "new_text": {"type": "string", "description": "Replacement text"},
                    },
                    "required": ["app", "path", "old_text", "new_text"],
                },
            },
            {
                "name": "app__query_db",
                "description": "Execute SQL against an app's SQLite database.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "app": {"type": "string", "description": app_enum_desc},
                        "sql": {"type": "string", "description": "SQL query"},
                    },
                    "required": ["app", "sql"],
                },
            },
            {
                "name": "app__logs",
                "description": "Get recent Metro/web server logs for an app.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "app": {"type": "string", "description": app_enum_desc},
                        "lines": {"type": "integer", "description": "Number of lines (default 50)"},
                    },
                    "required": ["app"],
                },
            },
            {
                "name": "app__restart",
                "description": "Restart Metro + web servers for an app.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "app": {"type": "string", "description": app_enum_desc},
                    },
                    "required": ["app"],
                },
            },
            {
                "name": "app__navigate",
                "description": "Navigate to a screen in an app, or list available screens.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "app": {"type": "string", "description": app_enum_desc},
                        "screen": {"type": "string", "description": "Screen name (omit to list screens)"},
                        "params": {"type": "object", "description": "Navigation params"},
                    },
                    "required": ["app"],
                },
            },
            {
                "name": "app__git_push",
                "description": "Commit and push changes for an app to GitHub.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "app": {"type": "string", "description": app_enum_desc},
                        "message": {"type": "string", "description": "Commit message"},
                    },
                    "required": ["app", "message"],
                },
            },
            {
                "name": "app__publish_web",
                "description": "Export and deploy the web version of an app.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "app": {"type": "string", "description": app_enum_desc},
                        "domain": {"type": "string", "description": "Custom domain (optional)"},
                    },
                    "required": ["app"],
                },
            },
            # ── Domain-specific action tool ──
            {
                "name": "app__action",
                "description": self._action_description(),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "app": {"type": "string", "description": app_enum_desc},
                        "action": {"type": "string", "description": "Domain action name from the app's skill manifest"},
                        "params": {"type": "object", "description": "Parameters for the action (varies per action)"},
                    },
                    "required": ["app", "action"],
                },
            },
        ]

    # ── Tool Dispatch ──────────────────────────────────────────────

    async def execute_tool(
        self, tool_name: str, args: Dict[str, Any], ctx: SkillContext
    ) -> str:
        if tool_name == "app__list":
            return await self._list(args, ctx)

        if tool_name == "app__action":
            return await self._exec_action(args, ctx)

        # All other tools need an app slug
        slug = args.get("app", "")
        if not slug:
            return "ERROR: 'app' parameter is required. Use app__list to see available apps."

        fs = self._resolve(slug)
        if not fs:
            available = ", ".join(sorted(self._apps.keys())) or "(none)"
            return f"ERROR: App '{slug}' not found. Available apps: {available}"

        # Map gateway tool names to AppFsSkill tool names
        action_map = {
            "app__status": "status",
            "app__list_files": "list_files",
            "app__read_file": "read_file",
            "app__write_file": "write_file",
            "app__edit_file": "edit_file",
            "app__query_db": "query_db",
            "app__logs": "logs",
            "app__restart": "restart",
            "app__navigate": "navigate",
            "app__git_push": "git_push",
            "app__publish_web": "publish_web",
        }

        action = action_map.get(tool_name)
        if not action:
            return f"ERROR: Unknown tool: {tool_name}"

        # Remap args: gateway uses 'path', AppFsSkill uses 'file_path'
        fs_args = dict(args)
        fs_args.pop("app", None)
        if "path" in fs_args:
            fs_args["file_path"] = fs_args.pop("path")

        fs_tool_name = f"{fs.meta.name}__{action}"
        try:
            return await fs.execute_tool(fs_tool_name, fs_args, ctx)
        except Exception as e:
            logger.error(f"[AppGateway] {tool_name} error for '{slug}': {e}")
            return f"ERROR: {e}"

    # ── List handler ───────────────────────────────────────────────

    async def _list(self, args: Dict, ctx: SkillContext) -> str:
        if not self._apps:
            return "No builder apps found. Use app_builder__build_app to create one."

        lines = [f"Builder apps ({len(self._apps)}):"]
        for slug, fs in sorted(self._apps.items()):
            manifest = self._manifests.get(slug)
            domain = f" [{manifest['domain']}]" if manifest else ""
            action_count = len(manifest.get("actions", [])) if manifest else 0
            lines.append(f"  - {fs.app_name}{domain}  (app: \"{slug}\")")
            if action_count:
                names = [a["name"] for a in manifest["actions"]]
                lines.append(f"    Domain actions: {', '.join(names)}")
        lines.append("")
        lines.append("Use app__* tools with app=\"<slug>\". Use app__action for domain-specific operations.")
        return "\n".join(lines)

    # ── Domain Action Execution ────────────────────────────────────

    async def _exec_action(self, args: Dict, ctx: SkillContext) -> str:
        slug = args.get("app", "")
        action_name = args.get("action", "")
        params = args.get("params") or {}

        if not slug or not action_name:
            return "ERROR: 'app' and 'action' are required."

        fs = self._resolve(slug)
        if not fs:
            available = ", ".join(sorted(self._apps.keys())) or "(none)"
            return f"ERROR: App '{slug}' not found. Available: {available}"

        manifest = self._manifests.get(slug)
        if not manifest:
            # Try loading on demand
            if hasattr(fs, 'app_dir'):
                self.load_app_manifest(slug, fs.app_dir)
                manifest = self._manifests.get(slug)
            if not manifest:
                return f"ERROR: No domain actions defined for app '{slug}'. Use app__query_db for raw SQL."

        # Find the action definition
        action_def = None
        for a in manifest.get("actions", []):
            if a["name"] == action_name:
                action_def = a
                break
        if not action_def:
            names = [a["name"] for a in manifest.get("actions", [])]
            return f"ERROR: Action '{action_name}' not found. Available: {', '.join(names)}"

        action_type = action_def.get("type", "query")

        # ── Navigate actions ──
        if action_type == "navigate":
            screen = action_def.get("screen", "")
            fs_tool = f"{fs.meta.name}__navigate"
            return await fs.execute_tool(fs_tool, {"screen": screen, "params": params}, ctx)

        # ── Query / Mutation actions ──
        if action_type in ("query", "mutation"):
            sql_template = action_def.get("sql", "")
            if not sql_template:
                return f"ERROR: Action '{action_name}' has no SQL defined."

            # Resolve params with defaults
            param_defs = action_def.get("params", {})
            resolved = {}
            for pname, pdef in param_defs.items():
                if pname in params:
                    resolved[pname] = params[pname]
                elif isinstance(pdef, dict) and "default" in pdef:
                    resolved[pname] = pdef["default"]
                elif isinstance(pdef, dict):
                    return f"ERROR: Required parameter '{pname}' not provided for action '{action_name}'."
                # Simple param defs (just type string) — require the param
                else:
                    return f"ERROR: Required parameter '{pname}' not provided for action '{action_name}'."

            # Convert :param_name to ? for SQLite parameterized queries
            param_order = []

            def _replace(match):
                name = match.group(1)
                param_order.append(name)
                return "?"

            sql = re.sub(r':(\w+)', _replace, sql_template)
            sql_params = tuple(resolved.get(p) for p in param_order)

            # Find database
            db_path = os.path.join(fs.app_dir, "data", "app.db")
            if not os.path.exists(db_path):
                db_path = os.path.join(fs.app_dir, "data.db")
            if not os.path.exists(db_path):
                return f"ERROR: No SQLite database found for app '{slug}'."

            try:
                import aiosqlite
                async with aiosqlite.connect(db_path) as db:
                    db.row_factory = aiosqlite.Row
                    cursor = await db.execute(sql, sql_params)

                    if action_type == "mutation":
                        await db.commit()
                        return (
                            f"Action '{action_name}' executed successfully.\n"
                            f"Rows affected: {cursor.rowcount}"
                        )

                    # Query — return results
                    rows = await cursor.fetchall()
                    if not rows:
                        return f"Action '{action_name}' returned 0 results."

                    cols = [d[0] for d in cursor.description]
                    lines = [
                        f"{action_def.get('description', action_name)}",
                        "",
                        " | ".join(cols),
                        "-" * max(len(" | ".join(cols)), 20),
                    ]
                    for row in rows[:100]:
                        lines.append(" | ".join(str(row[c]) for c in cols))
                    if len(rows) > 100:
                        lines.append(f"... ({len(rows)} total, showing 100)")
                    return "\n".join(lines)

            except ImportError:
                return "ERROR: aiosqlite not installed."
            except Exception as e:
                return f"ERROR executing action '{action_name}': {e}"

        return f"ERROR: Unknown action type '{action_type}' for '{action_name}'."

    # ── System Prompt ──────────────────────────────────────────────

    def get_system_prompt_section(self) -> Optional[str]:
        if not self._apps:
            return None

        lines = ["# Builder Apps"]
        lines.append(f"You have {len(self._apps)} app(s). Use `app__*` tools with the app slug:\n")

        for slug, fs in sorted(self._apps.items()):
            lines.append(f"- **{fs.app_name}** → app=\"{slug}\"")

            # Include domain actions from manifest
            manifest = self._manifests.get(slug)
            if manifest:
                domain = manifest.get("domain", "")
                if domain:
                    lines.append(f"  Domain: {domain}")
                actions = manifest.get("actions", [])
                if actions:
                    lines.append(f"  Domain actions (use `app__action` tool):")
                    for a in actions:
                        pnames = list(a.get("params", {}).keys())
                        pstr = f"({', '.join(pnames)})" if pnames else ""
                        lines.append(f"    - **{a['name']}**{pstr}: {a.get('description', '')}")

        lines.append("")
        lines.append(
            "Generic tools: app__list, app__status, app__list_files, app__read_file, "
            "app__write_file, app__edit_file, app__query_db, app__logs, "
            "app__restart, app__navigate, app__git_push, app__publish_web\n"
            "Domain tool: app__action (app, action, params) — run app-specific operations"
        )
        return "\n".join(lines)
