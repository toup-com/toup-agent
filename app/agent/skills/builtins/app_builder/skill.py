"""
App Builder Skill — Builds React Native/Expo apps from natural language.

When the LLM detects the user wants to build an app, it calls `app_builder__create_app`.
The tool creates a BuildJob + App in DB, spawns a background build task, and returns
immediately so the agent can tell the user to check the Jobs tab.

The background task:
  1. Plans the app (LLM → file list, deps, DB schema)
  2. Scaffolds Expo project
  3. Generates code for each file
  4. Sets up database (SQLite)
  5. Installs npm dependencies
  6. Creates GitHub repo + initial commit
  7. Starts Metro (mobile) + Expo Web (browser)
  8. Registers per-app tools via SkillLoader

Progress is broadcast via WebSocket job_update events.
"""

import asyncio
import json
import logging
import os
import re
import time
import uuid
from datetime import datetime
from typing import Any, Callable, Coroutine, Dict, List, Optional

from app.agent.skills.base import Skill, SkillContext, SkillMeta

logger = logging.getLogger(__name__)


# ── LLM prompts ─────────────────────────────────────────────────────

PLANNING_PROMPT = """You are planning a cross-platform app built with React Native/Expo.
The app must work on iPhone, iPad, and Web.

User wants: "{description}"

Output ONLY valid JSON (no markdown fences, no explanation):
{{
  "files": ["/App.tsx", "/screens/HomeScreen.tsx", ...],
  "dependencies": ["@react-navigation/native", "expo-sqlite", ...],
  "app_name": "TodoApp",
  "summary": "A todo app with ...",
  "needs_database": true,
  "db_type": "sqlite",
  "platforms": ["web", "ios"]
}}

Rules:
- Use TypeScript (.tsx/.ts files)
- Entry point MUST be /App.tsx
- Split into screens/, components/, lib/ directories
- For data persistence, use expo-sqlite (works offline on all platforms)
- Include @react-navigation/native + @react-navigation/native-stack for navigation
- Include react-native-safe-area-context and react-native-screens
- For responsive layout, plan to use useWindowDimensions
- If the app needs charts, use react-native-chart-kit (NOT recharts — that's web-only)
- Keep the file list focused — don't over-engineer
"""

CODE_GEN_PROMPT = """Generate the complete TypeScript code for {file_path} in a React Native/Expo app.

App description: {description}
App name: {app_name}
All files in the app: {all_files}
Dependencies: {all_deps}
Database: {db_type}

Rules:
- Use TypeScript with proper types
- Use React Native components (View, Text, ScrollView, Pressable, TextInput, FlatList, etc.)
- Use StyleSheet.create() for styles — dark theme (background: #161B22, text: #F0F2F5)
- Accent color: #58A6FF (blue)
- For responsive design, use useWindowDimensions() and Platform.select()
- If this file uses database, import from '../lib/db' (expo-sqlite helper)
- For navigation, use @react-navigation/native-stack
- Make it fully functional — not a skeleton
- Include proper error handling and loading states
- Output ONLY the code — no markdown fences, no explanation
"""


def _slugify(name: str, suffix: str = "") -> str:
    """Convert app name to a slug for tool naming."""
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:30]
    if suffix:
        slug = f"{slug}_{suffix}"
    return slug


class AppBuilderSkill(Skill):
    """Registers app_builder__create_app tool for building apps from chat."""

    meta = SkillMeta(
        name="app_builder",
        version="1.0.0",
        description="Build React Native/Expo apps from natural language",
        author="toup",
    )

    def __init__(
        self,
        app_manager=None,
        ws_broadcast: Optional[Callable] = None,
        skill_loader=None,
    ):
        self._app_manager = app_manager
        self._ws_broadcast = ws_broadcast
        self._skill_loader = skill_loader

    def set_refs(self, app_manager=None, ws_broadcast=None, skill_loader=None):
        """Set references after construction (for late binding)."""
        if app_manager:
            self._app_manager = app_manager
        if ws_broadcast:
            self._ws_broadcast = ws_broadcast
        if skill_loader:
            self._skill_loader = skill_loader

    def get_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "app_builder__create_app",
                "description": (
                    "Build a mobile + web app based on the user's description. "
                    "Use when the user asks you to create, build, or make an app. "
                    "The app will be built as a React Native/Expo project that runs on "
                    "iPhone, iPad, and Web. Builds in the background — returns immediately. "
                    "The user can track progress in the Jobs tab."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Short app name (2-4 words, e.g. 'Todo App')",
                        },
                        "description": {
                            "type": "string",
                            "description": (
                                "Detailed description of what the app should do, "
                                "its features, screens, and behavior."
                            ),
                        },
                    },
                    "required": ["name", "description"],
                },
            },
        ]

    async def execute_tool(
        self, tool_name: str, args: Dict[str, Any], ctx: SkillContext
    ) -> str:
        if tool_name != "app_builder__create_app":
            return f"Unknown tool: {tool_name}"

        name = args.get("name", "Untitled App")
        description = args.get("description", "")
        user_id = ctx.user_id

        if not self._app_manager:
            return "App builder is not available — app_manager not configured."

        # Create DB records
        from app.db.database import async_session_maker
        from app.db.models import App, BuildJob

        app_id = str(uuid.uuid4())
        job_id = str(uuid.uuid4())
        slug = _slugify(name, app_id[:6])
        app_dir = os.path.join(self._app_manager.APPS_DIR if hasattr(self._app_manager, 'APPS_DIR') else "/opt/toup-agent/apps", app_id)

        async with async_session_maker() as db:
            app = App(
                id=app_id,
                user_id=user_id,
                name=name,
                description=description,
                slug=slug,
                status="building",
                app_dir=app_dir,
                platforms="web,ios",
            )
            db.add(app)

            job = BuildJob(
                id=job_id,
                user_id=user_id,
                app_id=app_id,
                title=f"Build: {name}",
                prompt=description,
                status="queued",
                steps_json=json.dumps(self._initial_steps()),
            )
            db.add(job)
            await db.commit()

        # Spawn background build task
        asyncio.create_task(
            self._build_app(job_id, app_id, name, description, user_id, slug)
        )

        return (
            f"Started building '{name}'! Job ID: {job_id}\n\n"
            f"Track progress in the **Jobs** tab. "
            f"When it's done, you'll find the app in the **Apps** tab with:\n"
            f"- QR code to open in Expo Go (iPhone/iPad)\n"
            f"- Web URL to open in browser\n"
            f"- GitHub repo with all the code"
        )

    def get_system_prompt_section(self) -> Optional[str]:
        return (
            "# App Builder\n"
            "You can build cross-platform apps (iPhone, iPad, Web) using React Native/Expo.\n"
            "When the user asks to build/create/make an app, use `app_builder__create_app` with:\n"
            "- name: Short app name (2-4 words)\n"
            "- description: Detailed description of features, screens, and behavior\n\n"
            "The app builds in the background with:\n"
            "- Live preview (QR code for Expo Go + web URL for browser)\n"
            "- SQLite database for data persistence\n"
            "- GitHub repo (auto-created)\n"
            "- Publishing to web or App Store\n\n"
            "Tell the user to check the Jobs tab for progress. After building, each app "
            "gets its own tools for file editing, DB queries, GitHub push, and more."
        )

    # ── Build Pipeline ──────────────────────────────────────────────

    def _initial_steps(self) -> List[Dict]:
        """Create the initial steps list for a build job."""
        step_types = [
            ("planning", "Planning app architecture..."),
            ("scaffolding", "Creating Expo project..."),
            ("writing", "Generating app code..."),
            ("database", "Setting up database..."),
            ("installing", "Installing dependencies..."),
            ("github", "Creating GitHub repository..."),
            ("starting", "Starting preview servers..."),
            ("ready", "App is ready!"),
        ]
        return [
            {
                "id": str(uuid.uuid4()),
                "type": step_type,
                "label": label,
                "status": "pending",
            }
            for step_type, label in step_types
        ]

    async def _build_app(
        self,
        job_id: str,
        app_id: str,
        name: str,
        description: str,
        user_id: str,
        slug: str,
    ):
        """Background task that orchestrates the full build pipeline."""
        from app.db.database import async_session_maker
        from app.db.models import App, BuildJob

        logger.info(f"[BUILD] Starting build for '{name}' (job={job_id}, app={app_id})")

        try:
            # ── Step 1: Planning ────────────────────────────────────
            plan = await self._step_plan(job_id, user_id, description)
            if not plan:
                await self._fail_job(job_id, app_id, "Planning failed — could not generate app plan")
                return

            files_to_generate = plan.get("files", ["/App.tsx"])
            deps = plan.get("dependencies", [])
            db_type = plan.get("db_type", "none")
            app_name = plan.get("app_name", name)
            needs_db = plan.get("needs_database", False)

            # ── Step 2: Scaffolding ─────────────────────────────────
            await self._update_step(job_id, user_id, "scaffolding", "running")
            try:
                await self._app_manager.scaffold_app(app_id, app_name)
                await self._update_step(job_id, user_id, "scaffolding", "done")
            except Exception as e:
                await self._fail_job(job_id, app_id, f"Scaffold failed: {e}")
                return

            # ── Step 3: Writing code ────────────────────────────────
            await self._update_step(job_id, user_id, "writing", "running",
                                    detail=f"Generating {len(files_to_generate)} files...")
            try:
                generated_files = await self._generate_code(
                    description, app_name, files_to_generate, deps, db_type
                )
                await self._app_manager.write_app_files(app_id, generated_files)

                # Save files backup to DB
                async with async_session_maker() as db:
                    app = await db.get(App, app_id)
                    if app:
                        app.files_json = json.dumps(generated_files)
                        app.deps_json = json.dumps(deps)
                        await db.commit()

                await self._update_step(job_id, user_id, "writing", "done",
                                        detail=f"Generated {len(generated_files)} files")
            except Exception as e:
                await self._fail_job(job_id, app_id, f"Code generation failed: {e}")
                return

            # ── Step 4: Database ────────────────────────────────────
            if needs_db and db_type != "none":
                await self._update_step(job_id, user_id, "database", "running")
                try:
                    db_url = await self._app_manager.setup_database(app_id, db_type)
                    async with async_session_maker() as db:
                        app = await db.get(App, app_id)
                        if app:
                            app.db_type = db_type
                            app.db_url = db_url
                            app.storage_dir = await self._app_manager.setup_storage(app_id)
                            await db.commit()
                    await self._update_step(job_id, user_id, "database", "done")
                except Exception as e:
                    logger.warning(f"[BUILD] Database setup failed (non-fatal): {e}")
                    await self._update_step(job_id, user_id, "database", "done",
                                            detail=f"Skipped: {e}")
            else:
                await self._update_step(job_id, user_id, "database", "done", detail="Not needed")

            # ── Step 5: Installing deps ─────────────────────────────
            await self._update_step(job_id, user_id, "installing", "running")
            try:
                install_output = await self._app_manager.install_deps(app_id, deps)
                await self._update_step(job_id, user_id, "installing", "done")
            except Exception as e:
                await self._fail_job(job_id, app_id, f"npm install failed: {e}")
                return

            # ── Step 6: GitHub ──────────────────────────────────────
            await self._update_step(job_id, user_id, "github", "running")
            try:
                repo_info = await self._app_manager.create_github_repo(app_id, app_name)
                github_url = repo_info.get("repo_url", "")
                github_repo = repo_info.get("repo_name", "")
                async with async_session_maker() as db:
                    app = await db.get(App, app_id)
                    if app:
                        app.github_url = github_url
                        app.github_repo = github_repo
                        await db.commit()
                await self._update_step(job_id, user_id, "github", "done",
                                        detail=github_url or "Skipped (no gh CLI)")
            except Exception as e:
                logger.warning(f"[BUILD] GitHub repo creation failed (non-fatal): {e}")
                await self._update_step(job_id, user_id, "github", "done",
                                        detail=f"Skipped: {e}")

            # ── Step 7: Starting servers ────────────────────────────
            await self._update_step(job_id, user_id, "starting", "running")
            try:
                metro_port = await self._app_manager.start_metro(app_id)
                web_port = await self._app_manager.start_web(app_id)

                qr_url = await self._app_manager.get_qr_url(app_id)
                web_url = await self._app_manager.get_web_url(app_id)

                async with async_session_maker() as db:
                    app = await db.get(App, app_id)
                    if app:
                        app.status = "running"
                        app.port = metro_port
                        app.web_port = web_port
                        managed = self._app_manager._running.get(app_id)
                        if managed:
                            app.metro_pid = managed.metro_process.pid if managed.metro_process else None
                            app.web_pid = managed.web_process.pid if managed.web_process else None
                        await db.commit()

                await self._update_step(job_id, user_id, "starting", "done",
                                        detail=f"Metro:{metro_port} Web:{web_port}")
            except Exception as e:
                await self._fail_job(job_id, app_id, f"Server start failed: {e}")
                return

            # ── Step 8: Ready! ──────────────────────────────────────
            await self._update_step(job_id, user_id, "ready", "done")

            # Mark job completed
            async with async_session_maker() as db:
                job = await db.get(BuildJob, job_id)
                if job:
                    job.status = "completed"
                    job.completed_at = datetime.utcnow()
                    await db.commit()

            # Register per-app skill
            if self._skill_loader:
                try:
                    from app.agent.skills.builtins.app_skill import AppSkill
                    app_skill = AppSkill(app_id, name, slug)
                    await self._skill_loader.register_dynamic(app_skill)
                    logger.info(f"[BUILD] Registered skill for app '{name}' ({slug})")
                except Exception as e:
                    logger.warning(f"[BUILD] Failed to register app skill: {e}")

            # Broadcast app_ready
            if self._ws_broadcast:
                await self._ws_broadcast(user_id, {
                    "type": "app_ready",
                    "app_id": app_id,
                    "name": name,
                    "qr_url": qr_url,
                    "web_url": web_url,
                })

            logger.info(f"[BUILD] Build complete for '{name}' (job={job_id})")

        except Exception as e:
            logger.exception(f"[BUILD] Unexpected error building '{name}'")
            await self._fail_job(job_id, app_id, f"Unexpected error: {e}")

    # ── Step helpers ────────────────────────────────────────────────

    async def _step_plan(
        self, job_id: str, user_id: str, description: str
    ) -> Optional[Dict]:
        """Run the planning step — call LLM to get app plan."""
        await self._update_step(job_id, user_id, "planning", "running")

        try:
            llm_response = await self._call_llm(
                PLANNING_PROMPT.format(description=description),
                "Plan this app and output JSON.",
            )

            # Parse JSON from response
            plan = self._extract_json(llm_response)
            if not plan:
                logger.error(f"[BUILD] Failed to parse plan JSON from: {llm_response[:500]}")
                await self._update_step(job_id, user_id, "planning", "failed",
                                        detail="Could not parse LLM planning output")
                return None

            await self._update_step(job_id, user_id, "planning", "done",
                                    detail=f"{len(plan.get('files', []))} files planned")
            return plan
        except Exception as e:
            await self._update_step(job_id, user_id, "planning", "failed", detail=str(e))
            return None

    async def _generate_code(
        self,
        description: str,
        app_name: str,
        files: List[str],
        deps: List[str],
        db_type: str,
    ) -> Dict[str, str]:
        """Generate code for each file using LLM."""
        generated = {}
        all_files = json.dumps(files)
        all_deps = json.dumps(deps)

        for file_path in files:
            prompt = CODE_GEN_PROMPT.format(
                file_path=file_path,
                description=description,
                app_name=app_name,
                all_files=all_files,
                all_deps=all_deps,
                db_type=db_type,
            )
            code = await self._call_llm(prompt, f"Generate code for {file_path}")

            # Strip markdown fences if the LLM added them
            code = self._strip_fences(code)
            generated[file_path] = code

        return generated

    async def _call_llm(self, system_prompt: str, user_message: str) -> str:
        """Call the LLM (OpenAI or Anthropic) for code generation."""
        from app.config import settings

        # Try Anthropic first (stronger for code gen)
        anthropic_key = settings.anthropic_api_key or ""
        if anthropic_key:
            try:
                import anthropic
                import os
                is_oauth = "sk-ant-oat" in anthropic_key
                if is_oauth:
                    # OAuth tokens need auth_token + beta headers
                    # SDK auto-reads ANTHROPIC_API_KEY env → sends wrong X-Api-Key header
                    os.environ.pop("ANTHROPIC_API_KEY", None)
                    client = anthropic.AsyncAnthropic(
                        auth_token=anthropic_key,
                        default_headers={
                            "anthropic-beta": "claude-code-20250219,oauth-2025-04-20",
                            "user-agent": "claude-cli/2.1.2 (external, cli)",
                            "x-app": "cli",
                        },
                    )
                else:
                    client = anthropic.AsyncAnthropic(api_key=anthropic_key)

                response = await client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=8192,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_message}],
                )
                return response.content[0].text
            except Exception as e:
                logger.warning(f"[BUILD] Anthropic call failed, falling back to OpenAI: {e}")

        # Fallback to OpenAI
        if settings.openai_api_key:
            try:
                from openai import AsyncOpenAI
                client = AsyncOpenAI(api_key=settings.openai_api_key)
                model = settings.agent_model or "gpt-4o-mini"
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    max_completion_tokens=8192,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                logger.error(f"[BUILD] OpenAI call failed: {e}")
                raise

        raise RuntimeError("No LLM provider configured (need ANTHROPIC_API_KEY or OPENAI_API_KEY)")

    async def _update_step(
        self,
        job_id: str,
        user_id: str,
        step_type: str,
        status: str,
        detail: Optional[str] = None,
    ):
        """Update a step in the build job and broadcast via WebSocket."""
        from app.db.database import async_session_maker
        from app.db.models import BuildJob

        async with async_session_maker() as db:
            job = await db.get(BuildJob, job_id)
            if not job:
                return

            steps = json.loads(job.steps_json) if job.steps_json else []
            step_dict = None
            for s in steps:
                if s["type"] == step_type:
                    s["status"] = status
                    if detail:
                        s["detail"] = detail
                    if status == "running":
                        s["started_at"] = datetime.utcnow().isoformat()
                    elif status in ("done", "failed"):
                        started = s.get("started_at")
                        if started:
                            try:
                                start_dt = datetime.fromisoformat(started)
                                s["duration_ms"] = int((datetime.utcnow() - start_dt).total_seconds() * 1000)
                            except Exception:
                                pass
                    step_dict = s
                    break

            if job.status == "queued" and status == "running":
                job.status = "running"

            job.steps_json = json.dumps(steps)
            await db.commit()

        # Broadcast to user
        if self._ws_broadcast and step_dict:
            await self._ws_broadcast(user_id, {
                "type": "job_update",
                "job_id": job_id,
                "status": job.status if job else "running",
                "step": step_dict,
            })

    async def _fail_job(self, job_id: str, app_id: str, error_msg: str):
        """Mark a job as failed."""
        from app.db.database import async_session_maker
        from app.db.models import App, BuildJob

        logger.error(f"[BUILD] Job {job_id} failed: {error_msg}")

        async with async_session_maker() as db:
            job = await db.get(BuildJob, job_id)
            if job:
                job.status = "failed"
                job.error_message = error_msg
                job.completed_at = datetime.utcnow()
                await db.commit()

            app = await db.get(App, app_id)
            if app:
                app.status = "error"
                await db.commit()

    # ── Parsing helpers ─────────────────────────────────────────────

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict]:
        """Extract JSON object from LLM output (handles markdown fences)."""
        # Try direct parse
        text = text.strip()
        if text.startswith("{"):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass

        # Try extracting from code fences
        patterns = [
            r"```json\s*\n(.*?)\n```",
            r"```\s*\n(.*?)\n```",
            r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}",
        ]
        import re
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1) if match.lastindex else match.group(0))
                except (json.JSONDecodeError, IndexError):
                    continue
        return None

    @staticmethod
    def _strip_fences(code: str) -> str:
        """Strip markdown code fences from generated code."""
        code = code.strip()
        if code.startswith("```"):
            # Remove opening fence (with optional language tag)
            first_newline = code.index("\n") if "\n" in code else len(code)
            code = code[first_newline + 1:]
        if code.endswith("```"):
            code = code[:-3].rstrip()
        return code
