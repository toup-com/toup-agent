"""
AppFsSkill — Filesystem-backed per-app skill for builder apps.

Each built app gets an AppFsSkill instance that reads/writes directly
from the app directory on disk (/opt/toup-agent/apps/{app_id}/).

Tools (11):
  - list_files    — directory tree with sizes
  - read_file     — read file with line numbers
  - write_file    — write file (Metro auto-hot-reloads)
  - edit_file     — search/replace edit
  - query_db      — SQL against app's SQLite DB
  - logs          — tail Metro/web stdout
  - restart       — kill + restart servers
  - status        — ports, URLs, process info
  - navigate      — change screens/pages within the app
  - git_push      — commit + push to GitHub
  - publish_web   — export + deploy web version
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.agent.skills.base import Skill, SkillContext, SkillMeta

logger = logging.getLogger(__name__)


async def _record_layer2_change(app_id: str, file_path: str, action: str, summary: str):
    """Append a Layer 2 change to the most recent build job for this app."""
    try:
        from app.db.database import async_session_maker
        from app.db.models import BuildJob
        from sqlalchemy import select

        async with async_session_maker() as db:
            # Find the most recent build job for this app
            result = await db.execute(
                select(BuildJob)
                .where(BuildJob.app_id == app_id)
                .order_by(BuildJob.created_at.desc())
                .limit(1)
            )
            job = result.scalar_one_or_none()
            if not job:
                return

            # Parse existing changes
            changes = []
            if job.layer2_changes_json:
                try:
                    changes = json.loads(job.layer2_changes_json)
                except (json.JSONDecodeError, TypeError):
                    changes = []

            changes.append({
                "file": file_path,
                "action": action,
                "summary": summary,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            job.layer2_changes_json = json.dumps(changes)
            await db.commit()
    except Exception as e:
        logger.warning(f"Failed to record Layer 2 change: {e}")


class AppFsSkill(Skill):
    """Filesystem-backed skill for a builder app."""

    def __init__(
        self,
        app_id: str,
        app_name: str,
        app_slug: str,
        app_dir: str,
        app_manager=None,
    ):
        self.app_id = app_id
        self.app_name = app_name
        self.app_slug = app_slug
        self.app_dir = app_dir
        self._app_manager = app_manager
        self.meta = SkillMeta(
            name=f"app_{app_slug}",
            version="2.0.0",
            description=f"Full toolkit for '{app_name}' app",
            author="auto-generated",
        )

    def _p(self, name: str) -> str:
        return f"{self.meta.name}__{name}"

    # ── Tool Definitions ───────────────────────────────────────────

    def get_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": self._p("list_files"),
                "description": f"List all files in the '{self.app_name}' app with sizes.",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": self._p("read_file"),
                "description": f"Read a file from the '{self.app_name}' app (from filesystem).",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Relative path (e.g. '/App.tsx', '/screens/Home.tsx')"},
                    },
                    "required": ["file_path"],
                },
            },
            {
                "name": self._p("write_file"),
                "description": f"Write/update a file in the '{self.app_name}' app. Metro auto-hot-reloads.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Relative path"},
                        "content": {"type": "string", "description": "Full file content"},
                    },
                    "required": ["file_path", "content"],
                },
            },
            {
                "name": self._p("edit_file"),
                "description": f"Apply a search/replace edit to a file in the '{self.app_name}' app.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Relative path"},
                        "old_text": {"type": "string", "description": "Text to find"},
                        "new_text": {"type": "string", "description": "Replacement text"},
                    },
                    "required": ["file_path", "old_text", "new_text"],
                },
            },
            {
                "name": self._p("query_db"),
                "description": f"Execute SQL against the '{self.app_name}' app's SQLite database.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "sql": {"type": "string", "description": "SQL query to execute"},
                    },
                    "required": ["sql"],
                },
            },
            {
                "name": self._p("logs"),
                "description": f"Get recent Metro/web server logs for the '{self.app_name}' app.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "lines": {"type": "integer", "description": "Number of lines (default 50)"},
                    },
                    "required": [],
                },
            },
            {
                "name": self._p("restart"),
                "description": f"Restart Metro + web servers for the '{self.app_name}' app.",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": self._p("status"),
                "description": f"Get status of the '{self.app_name}' app (ports, URLs, processes).",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": self._p("navigate"),
                "description": (
                    f"Navigate to a screen/page within the '{self.app_name}' app. "
                    f"The agent uses this to change which screen the user sees. "
                    f"Also lists available screens when called without a target."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "screen": {
                            "type": "string",
                            "description": "Screen name to navigate to (e.g. 'Home', 'Settings', 'Detail'). Omit to list available screens.",
                        },
                        "params": {
                            "type": "object",
                            "description": "Optional navigation params (e.g. {\"id\": \"123\"} for detail screens)",
                        },
                    },
                    "required": [],
                },
            },
            {
                "name": self._p("git_push"),
                "description": f"Commit and push changes in the '{self.app_name}' app to GitHub.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string", "description": "Commit message"},
                    },
                    "required": ["message"],
                },
            },
            {
                "name": self._p("publish_web"),
                "description": f"Export and deploy the web version of the '{self.app_name}' app.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "domain": {"type": "string", "description": "Custom domain (optional)"},
                    },
                    "required": [],
                },
            },
        ]

    # ── Tool Dispatch ──────────────────────────────────────────────

    async def execute_tool(
        self, tool_name: str, args: Dict[str, Any], ctx: SkillContext
    ) -> str:
        dispatch = {
            self._p("list_files"): self._list_files,
            self._p("read_file"): self._read_file,
            self._p("write_file"): self._write_file,
            self._p("edit_file"): self._edit_file,
            self._p("query_db"): self._query_db,
            self._p("logs"): self._logs,
            self._p("restart"): self._restart,
            self._p("status"): self._status,
            self._p("navigate"): self._navigate,
            self._p("git_push"): self._git_push,
            self._p("publish_web"): self._publish_web,
        }
        handler = dispatch.get(tool_name)
        if not handler:
            return f"ERROR: Unknown tool: {tool_name}"
        try:
            return await handler(args, ctx)
        except Exception as e:
            logger.error(f"[AppFsSkill] {tool_name} error: {e}")
            return f"ERROR: {e}"

    # ── System Prompt ──────────────────────────────────────────────

    def get_system_prompt_section(self) -> Optional[str]:
        p = self._p
        return (
            f"# App: {self.app_name}\n"
            f"You have full access to this agentic app via `{self.meta.name}__*` tools.\n"
            f"This app has an Agent Placeholder on every screen — you can dock into it and help the user.\n\n"
            f"**Navigation & Docking:**\n"
            f"- `{p('navigate')}` — change screens/pages (or list available screens). Use this when the user says 'go to settings' or 'open my app'\n\n"
            f"**Files & Code:**\n"
            f"- `{p('list_files')}` — list all files with sizes\n"
            f"- `{p('read_file')}` — read a file from disk\n"
            f"- `{p('write_file')}` — write a file (auto-hot-reloads)\n"
            f"- `{p('edit_file')}` — search/replace edit on a file\n\n"
            f"**Data & Logs:**\n"
            f"- `{p('query_db')}` — run SQL against the app's SQLite database\n"
            f"- `{p('logs')}` — view Metro/web server output\n\n"
            f"**Operations:**\n"
            f"- `{p('restart')}` — restart preview servers\n"
            f"- `{p('status')}` — check ports, URLs, process status\n"
            f"- `{p('git_push')}` — commit and push to GitHub\n"
            f"- `{p('publish_web')}` — deploy web version with optional custom domain\n"
        )

    # ── Tool Handlers ──────────────────────────────────────────────

    async def _list_files(self, args: Dict, ctx: SkillContext) -> str:
        if not os.path.exists(self.app_dir):
            return f"App directory not found: {self.app_dir}"

        lines = [f"Files in '{self.app_name}' ({self.app_dir}):"]
        total_size = 0
        file_count = 0
        for root, dirs, files in os.walk(self.app_dir):
            dirs[:] = [d for d in dirs if d not in ('node_modules', '.expo', '.git')]
            for f in sorted(files):
                abs_path = os.path.join(root, f)
                rel_path = "/" + os.path.relpath(abs_path, self.app_dir)
                try:
                    size = os.path.getsize(abs_path)
                    total_size += size
                    file_count += 1
                    size_str = f"{size:,} bytes" if size < 10000 else f"{size / 1024:.1f} KB"
                    lines.append(f"  {rel_path}  ({size_str})")
                except OSError:
                    lines.append(f"  {rel_path}  (error reading)")

        lines.append(f"\n{file_count} files, {total_size / 1024:.1f} KB total")
        return "\n".join(lines)

    async def _read_file(self, args: Dict, ctx: SkillContext) -> str:
        file_path = args.get("file_path", "")
        if not file_path:
            return "ERROR: file_path is required"

        abs_path = os.path.join(self.app_dir, file_path.lstrip("/"))
        if not os.path.exists(abs_path):
            return f"File '{file_path}' not found in {self.app_dir}"

        try:
            with open(abs_path, 'r', encoding='utf-8') as f:
                content = f.read()
            # Add line numbers
            lines = content.split('\n')
            numbered = [f"{i+1:4}| {line}" for i, line in enumerate(lines)]
            return f"File: {file_path} ({len(content)} chars, {len(lines)} lines)\n\n" + "\n".join(numbered)
        except UnicodeDecodeError:
            return f"File '{file_path}' is binary (cannot read as text)"
        except Exception as e:
            return f"ERROR reading '{file_path}': {e}"

    async def _write_file(self, args: Dict, ctx: SkillContext) -> str:
        file_path = args.get("file_path", "")
        content = args.get("content", "")
        if not file_path:
            return "ERROR: file_path is required"
        if not content:
            return "ERROR: content is required"

        abs_path = os.path.join(self.app_dir, file_path.lstrip("/"))
        is_new = not os.path.exists(abs_path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)

        try:
            with open(abs_path, 'w', encoding='utf-8') as f:
                f.write(content)
            # Track Layer 2 change
            action = "create" if is_new else "edit"
            lines = content.count('\n') + 1
            await _record_layer2_change(
                self.app_id, file_path, action,
                f"{'Created' if is_new else 'Updated'} {file_path} ({lines} lines, {len(content)} chars)"
            )
            return f"Wrote '{file_path}' ({len(content)} chars). Call restart to apply changes if the app is not running."
        except Exception as e:
            return f"ERROR writing '{file_path}': {e}"

    async def _edit_file(self, args: Dict, ctx: SkillContext) -> str:
        file_path = args.get("file_path", "")
        old_text = args.get("old_text", "")
        new_text = args.get("new_text", "")
        if not file_path or not old_text:
            return "ERROR: file_path and old_text are required"

        abs_path = os.path.join(self.app_dir, file_path.lstrip("/"))
        if not os.path.exists(abs_path):
            return f"File '{file_path}' not found"

        try:
            with open(abs_path, 'r', encoding='utf-8') as f:
                content = f.read()

            if old_text not in content:
                return f"ERROR: old_text not found in '{file_path}'. Make sure the text matches exactly."

            count = content.count(old_text)
            if count > 1:
                return f"ERROR: old_text appears {count} times in '{file_path}'. Provide more context to make it unique."

            new_content = content.replace(old_text, new_text, 1)
            with open(abs_path, 'w', encoding='utf-8') as f:
                f.write(new_content)

            # Track Layer 2 change
            await _record_layer2_change(
                self.app_id, file_path, "edit",
                f"Edited {file_path} (replaced {len(old_text)} chars with {len(new_text)} chars)"
            )
            return f"Edited '{file_path}'. Call restart to apply changes if the app is not running."
        except Exception as e:
            return f"ERROR editing '{file_path}': {e}"

    async def _query_db(self, args: Dict, ctx: SkillContext) -> str:
        sql = args.get("sql", "")
        if not sql:
            return "ERROR: sql is required"

        # Find SQLite DB
        db_path = os.path.join(self.app_dir, "data", "app.db")
        if not os.path.exists(db_path):
            db_path = os.path.join(self.app_dir, "data.db")
        if not os.path.exists(db_path):
            return "No SQLite database found for this app."

        try:
            import aiosqlite
            async with aiosqlite.connect(db_path) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute(sql)
                if sql.strip().upper().startswith(("SELECT", "PRAGMA")):
                    rows = await cursor.fetchall()
                    if not rows:
                        return "Query returned 0 rows."
                    cols = [d[0] for d in cursor.description]
                    lines = [" | ".join(cols)]
                    lines.append("-" * len(lines[0]))
                    for row in rows[:100]:
                        lines.append(" | ".join(str(row[c]) for c in cols))
                    result = "\n".join(lines)
                    if len(rows) > 100:
                        result += f"\n... ({len(rows)} total rows, showing first 100)"
                    return result
                else:
                    await db.commit()
                    return f"Query executed. Rows affected: {cursor.rowcount}"
        except ImportError:
            return "ERROR: aiosqlite not installed. Cannot query database."
        except Exception as e:
            return f"ERROR: {e}"

    async def _logs(self, args: Dict, ctx: SkillContext) -> str:
        lines_count = args.get("lines", 50)

        if self._app_manager:
            try:
                log_lines = await self._app_manager.get_logs(self.app_id, lines=lines_count)
                if log_lines:
                    return f"Recent logs for '{self.app_name}':\n\n" + "\n".join(log_lines)
            except Exception:
                pass

        # Fallback: try reading log files
        log_dir = os.path.join(self.app_dir, ".logs")
        if not os.path.exists(log_dir):
            return "No logs available."

        output = []
        for log_file in ("metro.log", "web.log"):
            path = os.path.join(log_dir, log_file)
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        all_lines = f.readlines()
                        tail = all_lines[-lines_count:]
                        output.append(f"── {log_file} ──")
                        output.extend(line.rstrip() for line in tail)
                except Exception:
                    pass

        return "\n".join(output) if output else "No logs available."

    async def _restart(self, args: Dict, ctx: SkillContext) -> str:
        """Hot-restart: kill and respawn processes WITHOUT wiping node_modules.

        Uses restart_processes() for fast 2-3s restarts after file edits.
        Falls back to full install only if node_modules is somehow missing.
        """
        if not self._app_manager:
            return "ERROR: app_manager not available"

        try:
            result = await self._app_manager.restart_processes(self.app_id)
            metro_port = result["metro_port"]
            web_port = result["web_port"]
            duration_ms = result.get("duration_ms", 0)

            from app.db.database import async_session_maker
            from app.db.models import App
            async with async_session_maker() as db:
                app = await db.get(App, self.app_id)
                if app:
                    app.status = "running"
                    app.port = metro_port
                    app.web_port = web_port
                    await db.commit()

            web_url = await self._app_manager.get_web_url(self.app_id)
            return (
                f"Restarted '{self.app_name}' successfully ({duration_ms}ms). "
                f"Preview: {web_url or f'localhost:{web_port}'}"
            )
        except Exception as e:
            return f"ERROR restarting: {e}"

    async def _status(self, args: Dict, ctx: SkillContext) -> str:
        from app.db.database import async_session_maker
        from app.db.models import App

        async with async_session_maker() as db:
            app = await db.get(App, self.app_id)
            if not app:
                return "App not found in database."

        lines = [
            f"App: {self.app_name}",
            f"Status: {app.status}",
            f"Directory: {self.app_dir}",
        ]

        if app.port:
            lines.append(f"Metro port: {app.port}")
        if app.web_port:
            lines.append(f"Web port: {app.web_port}")
        if app.db_type and app.db_type != "none":
            lines.append(f"Database: {app.db_type}")
        if app.github_url:
            lines.append(f"GitHub: {app.github_url}")
        if app.publish_url:
            lines.append(f"Published: {app.publish_url}")

        if self._app_manager and app.status == "running":
            try:
                qr_url = await self._app_manager.get_qr_url(self.app_id)
                web_url = await self._app_manager.get_web_url(self.app_id)
                if qr_url:
                    lines.append(f"QR URL: {qr_url}")
                if web_url:
                    lines.append(f"Web URL: {web_url}")
            except Exception:
                pass

        # Disk usage
        if os.path.exists(self.app_dir):
            total = 0
            for root, dirs, files in os.walk(self.app_dir):
                dirs[:] = [d for d in dirs if d != 'node_modules']
                for f in files:
                    try:
                        total += os.path.getsize(os.path.join(root, f))
                    except OSError:
                        pass
            lines.append(f"Disk (excl. node_modules): {total / 1024:.1f} KB")

        return "\n".join(lines)

    async def _navigate(self, args: Dict, ctx: SkillContext) -> str:
        """Navigate to a screen or list available screens in the app."""
        screen = args.get("screen")
        params = args.get("params", {})

        # Discover available screens from the filesystem
        screens_dir = os.path.join(self.app_dir, "screens")
        available_screens = []
        if os.path.isdir(screens_dir):
            for f in sorted(os.listdir(screens_dir)):
                if f.endswith((".tsx", ".ts")) and not f.startswith("_"):
                    name = f.replace("Screen.tsx", "").replace("Screen.ts", "").replace(".tsx", "").replace(".ts", "")
                    available_screens.append(name)

        # Also check agentBridge for screen registry
        bridge_path = os.path.join(self.app_dir, "lib", "agentBridge.ts")
        bridge_info = ""
        if os.path.exists(bridge_path):
            try:
                with open(bridge_path, 'r') as fh:
                    bridge_info = fh.read()
            except Exception:
                pass

        # Check agentActions for per-screen actions
        actions_path = os.path.join(self.app_dir, "lib", "agentActions.ts")
        actions_info = ""
        if os.path.exists(actions_path):
            try:
                with open(actions_path, 'r') as fh:
                    actions_info = fh.read()
            except Exception:
                pass

        if not screen:
            # List mode — show available screens and actions
            lines = [f"Available screens in '{self.app_name}':"]
            for s in available_screens:
                lines.append(f"  - {s}")
            if not available_screens:
                lines.append("  (no screen files found in /screens/)")

            if actions_info:
                lines.append("\nAgent actions registry found at /lib/agentActions.ts")
                # Extract action registrations
                import re
                action_matches = re.findall(r"['\"](\w+)['\"]:\s*\[", actions_info)
                if action_matches:
                    lines.append("Screens with registered actions:")
                    for m in action_matches:
                        lines.append(f"  - {m}")

            lines.append(f"\nTo navigate: call {self._p('navigate')} with screen='ScreenName'")
            return "\n".join(lines)

        # Navigate mode — send navigation command via WebSocket
        if self._app_manager:
            try:
                # Broadcast navigation command to the app via WS
                from app.db.database import async_session_maker
                from app.db.models import App
                async with async_session_maker() as db:
                    app = await db.get(App, self.app_id)
                    if app and app.user_id:
                        # Get broadcast function from app_manager or ws_chat
                        try:
                            from app.api.ws_chat import broadcast_to_user
                            await broadcast_to_user(app.user_id, {
                                "type": "app_navigate",
                                "app_id": self.app_id,
                                "screen": screen,
                                "params": params,
                            })
                        except Exception:
                            pass
            except Exception as e:
                logger.debug(f"[navigate] WS broadcast failed: {e}")

        # Build response with screen info
        screen_file = None
        for s in available_screens:
            if s.lower() == screen.lower():
                screen_file = f"/screens/{s}Screen.tsx"
                break
        if not screen_file:
            # Try fuzzy match
            for s in available_screens:
                if screen.lower() in s.lower():
                    screen_file = f"/screens/{s}Screen.tsx"
                    break

        result = f"Navigating to '{screen}' in '{self.app_name}'."
        if screen_file:
            abs_path = os.path.join(self.app_dir, screen_file.lstrip("/"))
            if os.path.exists(abs_path):
                result += f"\nScreen file: {screen_file}"
        else:
            result += f"\nNote: No exact screen file found for '{screen}'. Available: {', '.join(available_screens)}"

        if params:
            result += f"\nParams: {json.dumps(params)}"

        return result

    async def _git_push(self, args: Dict, ctx: SkillContext) -> str:
        message = args.get("message", "Update from Toup agent")

        if self._app_manager:
            try:
                result = await self._app_manager.push_to_github(self.app_id)
                return f"Pushed to GitHub: {result}"
            except Exception as e:
                return f"ERROR pushing to GitHub: {e}"

        # Fallback: direct git commands
        try:
            proc = await asyncio.create_subprocess_exec(
                "git", "add", ".", "-A",
                cwd=self.app_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()

            proc = await asyncio.create_subprocess_exec(
                "git", "commit", "-m", message,
                cwd=self.app_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            proc = await asyncio.create_subprocess_exec(
                "git", "push",
                cwd=self.app_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode == 0:
                return f"Pushed to GitHub: {message}"
            return f"Push failed: {stderr.decode()[:500]}"
        except Exception as e:
            return f"ERROR: {e}"

    async def _publish_web(self, args: Dict, ctx: SkillContext) -> str:
        domain = args.get("domain")

        if self._app_manager:
            try:
                url = await self._app_manager.publish_web(self.app_id, domain=domain)
                from app.db.database import async_session_maker
                from app.db.models import App
                async with async_session_maker() as db:
                    app = await db.get(App, self.app_id)
                    if app:
                        app.publish_url = url
                        await db.commit()
                return f"Published to: {url}"
            except Exception as e:
                return f"ERROR publishing: {e}"

        return "ERROR: app_manager not available for publishing"


# ── Startup loader ──────────────────────────────────────────────

async def load_app_fs_skills_from_db(skill_loader, app_manager=None) -> int:
    """Scan apps table and register AppFsSkill for each existing app."""
    from app.db.database import async_session_maker
    from app.db.models import App
    from sqlalchemy import select

    count = 0
    try:
        async with async_session_maker() as db:
            result = await db.execute(
                select(App).where(App.status.in_(["running", "ready", "stopped"]))
            )
            apps = result.scalars().all()
            for app in apps:
                try:
                    skill = AppFsSkill(
                        app_id=app.id,
                        app_name=app.name,
                        app_slug=app.slug,
                        app_dir=app.app_dir,
                        app_manager=app_manager,
                    )
                    await skill_loader.register_dynamic(skill)
                    count += 1
                except Exception as e:
                    logger.debug(f"[AppFsSkill] Skipping app {app.id}: {e}")
    except Exception as e:
        logger.warning(f"[AppFsSkill] DB scan failed: {e}")

    return count
