"""
App Builder Skill — Conversational app builder for React Native/Expo.

The agent converses with the user (using Opus 4.6) to understand requirements,
presents a plan for approval, then builds in the background (using Sonnet 4.6).
After build, the user can preview and iterate.

Tools:
  - app_builder__build_app    — Start build after plan approval
  - app_builder__get_status   — Check build progress
  - app_builder__modify_app   — Apply changes to an existing app

The conversational flow is driven by the system prompt section — no state machine.
The agent asks questions, presents a plan, and only calls build_app after approval.
"""

import asyncio
import json
import logging
import os
import re
import time
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from app.agent.skills.base import Skill, SkillContext, SkillMeta

logger = logging.getLogger(__name__)


# ── LLM prompts ─────────────────────────────────────────────────────

PLANNING_PROMPT = """You are planning a cross-platform app built with React Native/Expo.
The app must work on iPhone, iPad, and Web.

User wants: "{description}"

{extra_context}

Output ONLY valid JSON (no markdown fences, no explanation):
{{
  "files": ["/App.tsx", "/screens/HomeScreen.tsx", "/components/AgentPlaceholder.tsx", "/lib/agentBridge.ts", ...],
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

CRITICAL — Agent Placeholder System:
Every app is an "agentic app" — the user's AI agent must be able to work inside it.
You MUST include these files in EVERY plan:
  - /components/AgentPlaceholder.tsx — Floating agent widget (bottom-right corner) that:
    - Shows the agent avatar as a small circle (idle state)
    - Expands into an inline chat interface when tapped (active state)
    - Is overlaid on EVERY screen via absolute positioning
    - Can be docked (minimized) or expanded (full chat panel)
  - /lib/agentBridge.ts — Bridge module that:
    - Exposes the current screen/route name to the agent
    - Provides a navigate(screenName, params?) function for agent-driven navigation
    - Lists available screens and their purposes
    - Exposes app-specific actions the agent can trigger (e.g. createTodo, deleteItem)
    - Uses postMessage/event-based communication pattern for agent ↔ app messaging
  - /lib/agentActions.ts — Screen-specific action registry:
    - Maps each screen to actions the agent can perform on it
    - E.g. HomeScreen → [addItem, searchItems, filterBy], SettingsScreen → [toggleTheme, exportData]
The AgentPlaceholder MUST be rendered in EVERY screen's root layout (in App.tsx navigator).
"""

CODE_GEN_PROMPT = """Generate the complete TypeScript code for {file_path} in a React Native/Expo app.

App description: {description}
App name: {app_name}
All files in the app: {all_files}
Dependencies: {all_deps}
Database: {db_type}
{design_notes}

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

Agent Placeholder System (CRITICAL — every app is "agentic"):
- If this is /components/AgentPlaceholder.tsx:
  Build a floating agent widget component. It renders a small circular avatar (40x40, position: absolute,
  bottom: 24, right: 24, zIndex: 9999) with a pulsing border animation. When tapped, it expands into
  an inline chat panel (width: 340, height: 480, borderRadius: 16, dark background #1C2128).
  The chat panel has: a header with "Agent" title + minimize button + connection dot (green=connected, gray=disconnected),
  a ScrollView for messages, and a TextInput + send button at the bottom.
  Messages are sent/received via the agentBridge (sendMessage / onAgentMessage).
  Show a typing indicator when agentBridge.onToolActivity fires (agent is using tools).
  If agentBridge.isConnected is false, show "Connecting..." text in the header.
  In minimized state, show just the circle with a subtle glow effect.
  The component accepts: onMessage callback, agentColor prop (default #58A6FF).
  Use Animated API for smooth expand/collapse transitions.

- If this is /lib/agentBridge.ts:
  Export an AgentBridge class (singleton) that connects to the user's REAL AI agent via WebSocket.
  On construction, read window.__TOUP_AUTH_TOKEN, window.__TOUP_APP_ID, window.__TOUP_WS_URL.
  If all three globals exist, connect to WS_URL with ?token=AUTH_TOKEN query param.

  Core methods:
  - currentScreen: string (updated by navigation listener)
  - navigate(screenName: string, params?: object): void — calls navigation ref
  - getScreens(): Array<{{name: string, description: string}}> — lists all screens
  - getActions(screenName?: string): Array<{{id: string, label: string, handler: string}}> — per-screen actions
  - sendMessage(text: string): void — sends {{"type":"message","text":text,"app_id":APP_ID,"channel":"app"}} over WebSocket
  - onAgentMessage(callback: (msg: string) => void): void — register callback for agent responses
  - onToolActivity(callback: (tool: string, done: boolean) => void): void — register callback for tool_start/tool_end
  - setNavigationRef(ref: any): void — store navigation container ref
  - isConnected: boolean — tracks WebSocket connection state

  WebSocket message handling:
  - On {{"type":"text_chunk","text":"..."}} — accumulate chunks into a response buffer
  - On {{"type":"done","text":"..."}} — fire onAgentMessage callbacks with the full text, clear buffer
  - On {{"type":"app_navigate","screen":"...","params":{{}}}} — call this.navigate(screen, params)
  - On {{"type":"tool_start","tool":"..."}} — fire onToolActivity callback (for typing indicator)
  - On {{"type":"tool_end","tool":"..."}} — fire onToolActivity callback
  - On {{"type":"error"}} — fire onAgentMessage with error text

  Reconnect: if WS closes unexpectedly, reconnect with exponential backoff (1s, 2s, 4s, max 3 retries).
  Expose window.__TOUP_AGENT_BRIDGE for external access (web).
  Do NOT process messages locally or simulate responses. ALL messages go through the WebSocket to the real agent.
  If WebSocket is not connected (globals missing), queue messages and show a fallback "Connecting..." message.

- If this is /lib/agentActions.ts:
  Export a registry mapping screen names to available actions:
  type AgentAction = {{ id: string; label: string; description: string; handler: (...args: any[]) => Promise<any> }}
  const screenActions: Record<string, AgentAction[]> = {{ ... }}
  Populate with meaningful actions for EACH screen in the app.
  Export registerAction(screen, action) and getActions(screen) functions.

- If this is /App.tsx:
  Import AgentPlaceholder and render it as the LAST child inside NavigationContainer,
  positioned absolutely so it overlays all screens. Pass the navigation ref to agentBridge.
  IMPORTANT: Only show AgentPlaceholder when NOT loaded through the platform (the platform shows
  the user's real agent). Wrap it in a condition:
  {{typeof window !== "undefined" && !(window as any).__TOUP_AUTH_TOKEN && <AgentPlaceholder />}}
  Example: after <Stack.Navigator>...</Stack.Navigator>, add the conditional AgentPlaceholder.

- For ANY screen file:
  Register that screen's agent actions in a useEffect via agentActions.registerAction().
"""

MODIFY_ANALYSIS_PROMPT = """You are analyzing a modification request for an existing React Native/Expo app.

App name: {app_name}
Current files: {file_list}
User wants: "{changes}"

Based on the change request, determine which files need to be modified or created.
Output ONLY valid JSON (no markdown fences):
{{
  "affected_files": ["/screens/HomeScreen.tsx", "/components/ThemeProvider.tsx"],
  "new_files": ["/lib/theme.ts"],
  "summary": "What changes will be made"
}}
"""


def _slugify(name: str, suffix: str = "") -> str:
    """Convert app name to a URL-friendly slug (e.g. 'GRE Success Tracker' → 'GRE-Success-Tracker')."""
    # Preserve original casing, replace non-alphanumeric with hyphens
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", name).strip("-")[:40]
    if suffix:
        slug = f"{slug}-{suffix}"
    return slug


class AppBuilderSkill(Skill):
    """Conversational app builder — asks questions, plans, builds, iterates."""

    meta = SkillMeta(
        name="app_builder",
        version="2.0.0",
        description="Build React Native/Expo apps through conversation",
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

    # ── Tool Definitions ───────────────────────────────────────────

    def get_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "app_builder__build_app",
                "description": (
                    "Build a mobile + web app AFTER discussing requirements with the user "
                    "and getting their approval on the plan. Do NOT call this without first "
                    "asking clarifying questions and presenting a plan. The app builds in "
                    "the background as a React Native/Expo project (iPhone, iPad, Web)."
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
                            "description": "Detailed description including ALL features and screens discussed with the user",
                        },
                        "screens": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of screens (e.g. ['Home', 'Settings', 'Detail'])",
                        },
                        "features": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of features discussed with the user",
                        },
                        "db_type": {
                            "type": "string",
                            "enum": ["sqlite", "none"],
                            "description": "Database type — sqlite for persistent data, none for UI-only",
                        },
                        "platforms": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Target platforms (default: ['web', 'ios'])",
                        },
                        "design_notes": {
                            "type": "string",
                            "description": "Design preferences from user (theme, colors, style)",
                        },
                    },
                    "required": ["name", "description"],
                },
            },
            {
                "name": "app_builder__get_status",
                "description": "Check the build progress of an app. Returns current step, progress, and URLs when ready.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "job_id": {
                            "type": "string",
                            "description": "The build job ID returned by build_app",
                        },
                    },
                    "required": ["job_id"],
                },
            },
            {
                "name": "app_builder__modify_app",
                "description": (
                    "Modify an existing app based on user feedback after preview. "
                    "Only regenerates affected files — much faster than a full rebuild. "
                    "Use when the user wants changes like 'make it dark theme' or 'add a settings page'."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "app_id": {
                            "type": "string",
                            "description": "The app ID to modify",
                        },
                        "changes": {
                            "type": "string",
                            "description": "Detailed description of what to change (from user feedback)",
                        },
                    },
                    "required": ["app_id", "changes"],
                },
            },
        ]

    # ── Tool Execution ─────────────────────────────────────────────

    async def execute_tool(
        self, tool_name: str, args: Dict[str, Any], ctx: SkillContext
    ) -> str:
        dispatch = {
            "app_builder__build_app": self._exec_build_app,
            "app_builder__get_status": self._exec_get_status,
            "app_builder__modify_app": self._exec_modify_app,
        }
        handler = dispatch.get(tool_name)
        if not handler:
            return f"Unknown tool: {tool_name}"
        return await handler(args, ctx)

    async def _exec_build_app(self, args: Dict[str, Any], ctx: SkillContext) -> str:
        """Start a new app build."""
        name = args.get("name", "Untitled App")
        description = args.get("description", "")
        screens = args.get("screens", [])
        features = args.get("features", [])
        db_type = args.get("db_type", "sqlite")
        platforms = args.get("platforms", ["web", "ios"])
        design_notes = args.get("design_notes", "")
        user_id = ctx.user_id

        if not self._app_manager:
            return "App builder is not available — app_manager not configured."

        from app.db.database import async_session_maker
        from app.db.models import App, BuildJob

        app_id = str(uuid.uuid4())
        job_id = str(uuid.uuid4())
        slug = _slugify(name)
        apps_dir = self._app_manager.APPS_DIR if hasattr(self._app_manager, 'APPS_DIR') else "/opt/toup-agent/apps"
        app_dir = os.path.join(apps_dir, app_id)

        # Store the conversational context as plan JSON
        plan_context = {
            "screens": screens,
            "features": features,
            "db_type": db_type,
            "platforms": platforms,
            "design_notes": design_notes,
        }

        async with async_session_maker() as db:
            app = App(
                id=app_id,
                user_id=user_id,
                name=name,
                description=description,
                slug=slug,
                status="building",
                app_dir=app_dir,
                platforms=",".join(platforms),
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

        # Spawn background build
        asyncio.create_task(
            self._build_app(job_id, app_id, name, description, user_id, slug, app_dir, plan_context)
        )

        return (
            f"Building '{name}'! Job ID: {job_id}\n\n"
            f"Track progress in the **Jobs** tab. "
            f"When done, you'll find it in the **Apps** tab with:\n"
            f"- QR code for Expo Go (iPhone/iPad)\n"
            f"- Web URL for browser preview\n"
            f"- GitHub repo with all code\n\n"
            f"Use `app_builder__get_status` with job_id='{job_id}' to check progress."
        )

    async def _exec_get_status(self, args: Dict[str, Any], ctx: SkillContext) -> str:
        """Check build job status."""
        from app.db.database import async_session_maker
        from app.db.models import App, BuildJob

        job_id = args.get("job_id", "")
        if not job_id:
            return "ERROR: job_id is required"

        async with async_session_maker() as db:
            job = await db.get(BuildJob, job_id)
            if not job:
                return f"Job '{job_id}' not found."

            steps = json.loads(job.steps_json) if job.steps_json else []
            current_step = next((s for s in steps if s["status"] == "running"), None)
            done_count = sum(1 for s in steps if s["status"] == "done")

            result = f"Job: {job.title}\nStatus: {job.status}\nProgress: {done_count}/{len(steps)} steps"

            if current_step:
                result += f"\nCurrent: {current_step['label']}"
                if current_step.get("detail"):
                    result += f" — {current_step['detail']}"

            if job.status == "completed" and job.app_id:
                app = await db.get(App, job.app_id)
                if app:
                    result += f"\n\nApp is {app.status}!"
                    if app.port:
                        result += f"\nMetro port: {app.port}"
                    if app.web_port:
                        result += f"\nWeb port: {app.web_port}"
                    if self._app_manager:
                        try:
                            qr_url = await self._app_manager.get_qr_url(app.id)
                            web_url = await self._app_manager.get_web_url(app.id)
                            if qr_url:
                                result += f"\nQR URL: {qr_url}"
                            if web_url:
                                result += f"\nWeb URL: {web_url}"
                        except Exception:
                            pass
                    if app.github_url:
                        result += f"\nGitHub: {app.github_url}"

            if job.status == "failed":
                result += f"\nError: {job.error_message or 'Unknown error'}"

            return result

    async def _exec_modify_app(self, args: Dict[str, Any], ctx: SkillContext) -> str:
        """Modify an existing app based on user feedback."""
        from app.db.database import async_session_maker
        from app.db.models import App, BuildJob

        app_id = args.get("app_id", "")
        changes = args.get("changes", "")
        user_id = ctx.user_id

        if not app_id or not changes:
            return "ERROR: app_id and changes are required"

        if not self._app_manager:
            return "App builder is not available — app_manager not configured."

        async with async_session_maker() as db:
            app = await db.get(App, app_id)
            if not app:
                return f"App '{app_id}' not found."
            if app.status == "building":
                return "App is currently being built. Wait for the build to complete before modifying."

        # Create a modification job
        job_id = str(uuid.uuid4())
        async with async_session_maker() as db:
            job = BuildJob(
                id=job_id,
                user_id=user_id,
                app_id=app_id,
                title=f"Modify: {app.name}",
                prompt=changes,
                status="queued",
                steps_json=json.dumps([
                    {"id": str(uuid.uuid4()), "type": "planning", "label": "Analyzing changes...", "status": "pending"},
                    {"id": str(uuid.uuid4()), "type": "writing", "label": "Updating files...", "status": "pending"},
                    {"id": str(uuid.uuid4()), "type": "installing", "label": "Updating dependencies...", "status": "pending"},
                    {"id": str(uuid.uuid4()), "type": "starting", "label": "Restarting servers...", "status": "pending"},
                    {"id": str(uuid.uuid4()), "type": "ready", "label": "Ready!", "status": "pending"},
                ]),
            )
            db.add(job)
            app.status = "building"
            await db.commit()

        # Spawn background modification
        asyncio.create_task(
            self._modify_app(job_id, app_id, app.name, app.description or "", changes, user_id, app.slug, app.app_dir)
        )

        return (
            f"Modifying '{app.name}'! Job ID: {job_id}\n\n"
            f"Changes: {changes}\n"
            f"Only affected files will be regenerated — this is faster than a full rebuild.\n"
            f"Track progress in the **Jobs** tab."
        )

    # ── System Prompt ──────────────────────────────────────────────

    def get_system_prompt_section(self) -> Optional[str]:
        return (
            "# App Builder\n"
            "You can build cross-platform **agentic apps** (iPhone, iPad, Web) using React Native/Expo.\n"
            "Every app you build includes an **Agent Placeholder** — a floating widget where YOU (the agent) "
            "can dock into the app and help the user. You'll have full access to navigate, read data, and "
            "perform actions within every app you build.\n\n"
            "## When the user wants to build an app, follow this process:\n\n"
            "### Step 1: Understand Requirements\n"
            "Ask clarifying questions ONE at a time. Key questions:\n"
            "- What is the main purpose of this app?\n"
            "- Who will use it? (personal, team, public)\n"
            "- What are the key features? Suggest 3-4 based on the app type and let them pick.\n"
            "- Does it need to save data? (SQLite for local persistence, or none for UI-only)\n"
            "- Any design preferences? (dark/light theme, specific colors, style)\n\n"
            "Keep it natural and conversational — 2-4 questions total, not an interrogation.\n"
            "If the user gives a detailed enough description upfront, skip to Step 2.\n\n"
            "**CRITICAL — Quick Reply Buttons:**\n"
            "Whenever you present options or ask the user to choose, ALWAYS include clickable "
            "quick-reply buttons using double-bracket syntax at the END of your message:\n"
            "[[Option A]] [[Option B]] [[Option C]]\n\n"
            "Examples:\n"
            "- Feature suggestions: [[Practice tests]] [[Study planner]] [[Progress tracker]] [[All of them]]\n"
            "- Yes/no: [[Looks good, build it!]] [[I want to change something]]\n"
            "- Theme: [[Dark theme]] [[Light theme]] [[Match system]]\n"
            "- Database: [[Yes, save my data]] [[No, UI only]]\n\n"
            "Always provide 2-5 buttons. The user can tap one OR type a custom response.\n"
            "Place the [[buttons]] on their own line at the very end of your message.\n\n"
            "### Step 2: Present Plan\n"
            "After gathering requirements, present a structured plan:\n"
            "- **App name** and 1-2 sentence summary\n"
            "- **Screens**: List each screen with its purpose\n"
            "- **Features**: List each feature\n"
            "- **Agent Integration**: Mention that your agent placeholder will be on every page, "
            "with per-screen actions (e.g. 'On the Home screen, I can help you add/search/filter items')\n"
            "- **Database**: SQLite or none\n"
            "- **Platforms**: web + mobile\n\n"
            "Ask: \"Does this plan look good?\"\n"
            "[[Build it!]] [[Change something]]\n\n"
            "### Step 3: Build (ONLY after explicit approval)\n"
            "Call `app_builder__build_app` with ALL the context from the conversation.\n"
            "Include screens, features, design_notes, and db_type from the discussion.\n"
            "Tell the user to check the **Jobs** tab for live progress.\n\n"
            "### Step 4: Preview & Iterate\n"
            "After build completes, share the preview URL and QR code.\n"
            "Ask: \"How does it look?\"\n"
            "[[Looks great!]] [[Change colors]] [[Add a feature]] [[Rebuild it]]\n"
            "If they want changes, use `app_builder__modify_app` with their feedback.\n"
            "This only regenerates affected files — much faster than a full rebuild.\n\n"
            "**IMPORTANT**: NEVER call app_builder__build_app without first discussing "
            "requirements and getting the user's approval on the plan.\n\n"
            "### Status Check\n"
            "Use `app_builder__get_status` with the job_id to check build progress.\n"
            "After building, each app gets its own tools (file editing, DB queries, "
            "navigation, GitHub push, restart, etc.) under `app_{slug}__*`.\n\n"
            "### Agent Docking (after build)\n"
            "Once an app is built, you can dock into it using the app's tools:\n"
            "- `app_{slug}__navigate` — change pages/screens within the app\n"
            "- `app_{slug}__read_file` / `app_{slug}__write_file` — modify any app code\n"
            "- `app_{slug}__query_db` — read/write the app's database\n"
            "- The user sees your agent placeholder on every screen of their app\n"
            "- When the user says 'go to my app' or 'open the todo app', use navigate to go to the right screen\n"
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
        app_dir: str,
        plan_context: Optional[Dict] = None,
    ):
        """Background task that orchestrates the full build pipeline."""
        from app.db.database import async_session_maker
        from app.db.models import App, BuildJob
        from .build_logger import BuildLogger

        blog = BuildLogger(job_id, user_id, ws_broadcast=self._ws_broadcast)
        blog.set_step("init")
        await blog.info(f"Starting build for '{name}'", f"app={app_id}")
        await blog.info(f"User prompt: {description[:200]}{'...' if len(description) > 200 else ''}")

        # Build extra context from the conversational plan
        extra_context = ""
        design_notes = ""
        if plan_context:
            screens = plan_context.get("screens", [])
            features = plan_context.get("features", [])
            design_notes = plan_context.get("design_notes", "")
            if screens:
                extra_context += f"\nScreens: {', '.join(screens)}"
                await blog.info(f"Screens from conversation: {', '.join(screens)}")
            if features:
                extra_context += f"\nFeatures: {', '.join(features)}"
                await blog.info(f"Features from conversation: {', '.join(features)}")
            if design_notes:
                extra_context += f"\nDesign preferences: {design_notes}"
                await blog.info(f"Design notes: {design_notes}")

        try:
            # ── Step 1: Planning ────────────────────────────────────
            plan = await self._step_plan(job_id, user_id, description, blog, extra_context)
            if not plan:
                await blog.error("Planning failed — could not generate app plan")
                await blog.persist()
                await self._fail_job(job_id, app_id, "Planning failed — could not generate app plan")
                return

            files_to_generate = plan.get("files", ["/App.tsx"])
            deps = plan.get("dependencies", [])
            db_type = plan.get("db_type", plan_context.get("db_type", "none") if plan_context else "none")
            app_name = plan.get("app_name", name)
            needs_db = plan.get("needs_database", False)

            await blog.info(f"Plan: {len(files_to_generate)} files, {len(deps)} deps, db={db_type}")

            # Save plan to DB
            async with async_session_maker() as db:
                app = await db.get(App, app_id)
                if app:
                    try:
                        app.plan_json = json.dumps(plan)
                    except Exception:
                        pass
                    await db.commit()

            # ── Step 2: Scaffolding ─────────────────────────────────
            blog.set_step("scaffolding")
            await self._update_step(job_id, user_id, "scaffolding", "running")
            try:
                await blog.info(f"Creating Expo project '{app_name}'...")
                t0 = time.time()
                await self._app_manager.scaffold_app(app_id, app_name)
                elapsed = time.time() - t0
                await blog.success(f"Expo project created", f"{elapsed:.1f}s")
                await self._update_step(job_id, user_id, "scaffolding", "done")
            except Exception as e:
                await blog.error(f"Scaffold failed: {e}")
                await blog.persist()
                await self._fail_job(job_id, app_id, f"Scaffold failed: {e}")
                return

            # ── Step 3: Writing code ────────────────────────────────
            blog.set_step("writing")
            await self._update_step(job_id, user_id, "writing", "running",
                                    detail=f"Generating {len(files_to_generate)} files...")
            try:
                await blog.info(f"Generating {len(files_to_generate)} files (5 concurrent)...")
                generated_files = await self._generate_code(
                    description, app_name, files_to_generate, deps, db_type,
                    job_id=job_id, user_id=user_id, blog=blog,
                    design_notes=design_notes,
                )
                await self._app_manager.write_app_files(app_id, generated_files)

                total_bytes = sum(len(c.encode()) for c in generated_files.values())
                await blog.success(f"Generated {len(generated_files)} files", f"{total_bytes:,} bytes total")

                async with async_session_maker() as db:
                    app = await db.get(App, app_id)
                    if app:
                        app.files_json = json.dumps(generated_files)
                        app.deps_json = json.dumps(deps)
                        await db.commit()

                await self._update_step(job_id, user_id, "writing", "done",
                                        detail=f"Generated {len(generated_files)} files")
            except Exception as e:
                await blog.error(f"Code generation failed: {e}")
                await blog.persist()
                await self._fail_job(job_id, app_id, f"Code generation failed: {e}")
                return

            # ── Step 4: Database ────────────────────────────────────
            blog.set_step("database")
            if needs_db and db_type != "none":
                await self._update_step(job_id, user_id, "database", "running")
                try:
                    await blog.info(f"Setting up {db_type} database...")
                    db_url = await self._app_manager.setup_database(app_id, db_type)
                    storage_dir = await self._app_manager.setup_storage(app_id)
                    async with async_session_maker() as db:
                        app = await db.get(App, app_id)
                        if app:
                            app.db_type = db_type
                            app.db_url = db_url
                            app.storage_dir = storage_dir
                            await db.commit()
                    await blog.success(f"Database ready: {db_type}", db_url or "")
                    await self._update_step(job_id, user_id, "database", "done")
                except Exception as e:
                    await blog.warn(f"Database setup failed (non-fatal): {e}")
                    await self._update_step(job_id, user_id, "database", "done",
                                            detail=f"Skipped: {e}")
            else:
                await blog.info("No database needed, skipping")
                await self._update_step(job_id, user_id, "database", "done", detail="Not needed")

            # ── Step 5: Installing deps ─────────────────────────────
            blog.set_step("installing")
            await self._update_step(job_id, user_id, "installing", "running")
            try:
                web_deps = ["react-dom", "react-native-web"]
                all_deps = list(set((deps or []) + web_deps))
                await blog.info(f"Installing {len(all_deps)} packages...")
                for dep in sorted(all_deps):
                    await blog.debug(f"  {dep}")
                t0 = time.time()
                install_output = await self._app_manager.install_deps(app_id, all_deps)
                elapsed = time.time() - t0
                await blog.command_run(f"npm install {' '.join(all_deps[:5])}{'...' if len(all_deps) > 5 else ''}",
                                       exit_code=0, duration_s=elapsed, output=install_output[:300] if install_output else "")
                await blog.success(f"Dependencies installed", f"{elapsed:.1f}s")
                await self._update_step(job_id, user_id, "installing", "done")
            except Exception as e:
                await blog.error(f"npm install failed: {e}")
                await blog.persist()
                await self._fail_job(job_id, app_id, f"npm install failed: {e}")
                return

            # ── Step 6: GitHub ──────────────────────────────────────
            blog.set_step("github")
            await self._update_step(job_id, user_id, "github", "running")
            try:
                await blog.info("Creating GitHub repository...")
                repo_info = await self._app_manager.create_github_repo(app_id, app_name)
                github_url = repo_info.get("repo_url", "")
                github_repo = repo_info.get("repo_name", "")
                async with async_session_maker() as db:
                    app = await db.get(App, app_id)
                    if app:
                        app.github_url = github_url
                        app.github_repo = github_repo
                        await db.commit()
                await blog.success(f"GitHub repo created", github_url)
                await self._update_step(job_id, user_id, "github", "done",
                                        detail=github_url or "Skipped (no gh CLI)")
            except Exception as e:
                await blog.warn(f"GitHub repo creation failed (non-fatal): {e}")
                await self._update_step(job_id, user_id, "github", "done",
                                        detail=f"Skipped: {e}")

            # ── Step 7: Starting servers ────────────────────────────
            blog.set_step("starting")
            await self._update_step(job_id, user_id, "starting", "running")
            qr_url = ""
            web_url = ""
            try:
                await blog.info("Starting Metro bundler (mobile)...")
                metro_port = await self._app_manager.start_metro(app_id)
                await blog.success(f"Metro running on port {metro_port}")

                await blog.info("Starting Expo Web server...")
                web_port = await self._app_manager.start_web(app_id)
                await blog.success(f"Web server running on port {web_port}")

                qr_url = await self._app_manager.get_qr_url(app_id)
                web_url = await self._app_manager.get_web_url(app_id)
                await blog.info(f"QR URL: {qr_url}")
                await blog.info(f"Web URL: {web_url}")

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
                await blog.error(f"Server start failed: {e}")
                await blog.persist()
                await self._fail_job(job_id, app_id, f"Server start failed: {e}")
                return

            # ── Step 8: Ready! ──────────────────────────────────────
            blog.set_step("ready")
            await self._update_step(job_id, user_id, "ready", "done")

            summary = blog.summary()
            await blog.success(
                f"Build complete! {summary['files_written']} files, {summary['llm_calls']} LLM calls, {summary['total_tokens']:,} tokens",
                f"Errors: {summary['errors']}, Warnings: {summary['warnings']}"
            )

            await blog.persist()
            async with async_session_maker() as db:
                job = await db.get(BuildJob, job_id)
                if job:
                    job.status = "completed"
                    job.completed_at = datetime.utcnow()
                    job.model = "claude-sonnet-4-6"
                    await db.commit()

            # Register per-app filesystem skill
            if self._skill_loader:
                try:
                    from .app_fs_skill import AppFsSkill
                    app_fs_skill = AppFsSkill(app_id, name, slug, app_dir, self._app_manager)
                    await self._skill_loader.register_dynamic(app_fs_skill)
                    logger.info(f"[BUILD] Registered AppFsSkill for '{name}' ({slug})")
                except Exception as e:
                    logger.warning(f"[BUILD] Failed to register AppFsSkill: {e}")

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
            await blog.error(f"Unexpected error: {e}")
            await blog.persist()
            await self._fail_job(job_id, app_id, f"Unexpected error: {e}")

    # ── Modify Pipeline ────────────────────────────────────────────

    async def _modify_app(
        self,
        job_id: str,
        app_id: str,
        app_name: str,
        description: str,
        changes: str,
        user_id: str,
        slug: str,
        app_dir: str,
    ):
        """Background task that modifies an existing app."""
        from app.db.database import async_session_maker
        from app.db.models import App, BuildJob
        from .build_logger import BuildLogger

        blog = BuildLogger(job_id, user_id, ws_broadcast=self._ws_broadcast)
        blog.set_step("init")
        await blog.info(f"Modifying '{app_name}'", f"app={app_id}")
        await blog.info(f"Changes: {changes[:200]}{'...' if len(changes) > 200 else ''}")

        try:
            # ── Step 1: Analyze changes (Opus) ─────────────────────
            blog.set_step("planning")
            await self._update_step(job_id, user_id, "planning", "running")

            # Read current file list from filesystem
            file_list = []
            if os.path.exists(app_dir):
                for root, dirs, files in os.walk(app_dir):
                    # Skip node_modules, .expo, .git
                    dirs[:] = [d for d in dirs if d not in ('node_modules', '.expo', '.git', '.logs')]
                    for f in files:
                        rel = os.path.relpath(os.path.join(root, f), app_dir)
                        file_list.append(f"/{rel}")

            await blog.info(f"Current app has {len(file_list)} files")

            # Ask Opus which files need changes
            analysis_response = await self._call_llm(
                MODIFY_ANALYSIS_PROMPT.format(
                    app_name=app_name,
                    file_list=json.dumps(file_list[:100]),
                    changes=changes,
                ),
                "Analyze which files need modification.",
                model="claude-opus-4-6",
                blog=blog,
                purpose="Modification analysis",
            )

            analysis = self._extract_json(analysis_response)
            if not analysis:
                await blog.warn("Could not parse analysis, regenerating all screen files")
                analysis = {
                    "affected_files": [f for f in file_list if '/screens/' in f or f == '/App.tsx'],
                    "new_files": [],
                    "summary": changes,
                }

            affected = analysis.get("affected_files", []) + analysis.get("new_files", [])
            if not affected:
                affected = ["/App.tsx"]

            await blog.success(f"Analysis complete: {len(affected)} files to update")
            await self._update_step(job_id, user_id, "planning", "done",
                                    detail=f"{len(affected)} files: {', '.join(affected[:5])}")

            # ── Step 2: Regenerate affected files (Sonnet) ─────────
            blog.set_step("writing")
            await self._update_step(job_id, user_id, "writing", "running",
                                    detail=f"Regenerating {len(affected)} files...")

            # Read existing file contents for context
            existing_contents = {}
            for fp in file_list[:50]:
                abs_path = os.path.join(app_dir, fp.lstrip("/"))
                if os.path.exists(abs_path):
                    try:
                        with open(abs_path, 'r') as fh:
                            existing_contents[fp] = fh.read()
                    except Exception:
                        pass

            # Load deps from DB
            deps = []
            async with async_session_maker() as db:
                app = await db.get(App, app_id)
                if app and app.deps_json:
                    try:
                        deps = json.loads(app.deps_json)
                    except Exception:
                        pass

            modify_description = f"{description}\n\nMODIFICATION REQUEST: {changes}"
            generated = await self._generate_code(
                modify_description, app_name, affected, deps,
                app.db_type if app else "none",
                job_id=job_id, user_id=user_id, blog=blog,
                existing_files=existing_contents,
            )

            # Write files to disk
            await self._app_manager.write_app_files(app_id, generated)
            await blog.success(f"Updated {len(generated)} files")
            await self._update_step(job_id, user_id, "writing", "done",
                                    detail=f"Updated {len(generated)} files")

            # ── Step 3: Install any new deps ───────────────────────
            blog.set_step("installing")
            await self._update_step(job_id, user_id, "installing", "done", detail="No new deps needed")

            # ── Step 4: Restart servers ────────────────────────────
            blog.set_step("starting")
            await self._update_step(job_id, user_id, "starting", "running")
            try:
                await self._app_manager.stop_app(app_id)
                metro_port = await self._app_manager.start_metro(app_id)
                web_port = await self._app_manager.start_web(app_id)
                await blog.success(f"Servers restarted — Metro:{metro_port} Web:{web_port}")

                async with async_session_maker() as db:
                    app = await db.get(App, app_id)
                    if app:
                        app.status = "running"
                        app.port = metro_port
                        app.web_port = web_port
                        await db.commit()

                await self._update_step(job_id, user_id, "starting", "done",
                                        detail=f"Metro:{metro_port} Web:{web_port}")
            except Exception as e:
                await blog.warn(f"Restart failed (non-fatal): {e}")
                await self._update_step(job_id, user_id, "starting", "done",
                                        detail=f"Restart skipped: {e}")

            # ── Step 5: Ready! ─────────────────────────────────────
            blog.set_step("ready")
            await self._update_step(job_id, user_id, "ready", "done")

            await blog.persist()
            async with async_session_maker() as db:
                job = await db.get(BuildJob, job_id)
                if job:
                    job.status = "completed"
                    job.completed_at = datetime.utcnow()
                    job.model = "claude-sonnet-4-6"
                    await db.commit()

            if self._ws_broadcast:
                qr_url = await self._app_manager.get_qr_url(app_id) if self._app_manager else ""
                web_url = await self._app_manager.get_web_url(app_id) if self._app_manager else ""
                await self._ws_broadcast(user_id, {
                    "type": "app_ready",
                    "app_id": app_id,
                    "name": app_name,
                    "qr_url": qr_url,
                    "web_url": web_url,
                })

            logger.info(f"[BUILD] Modification complete for '{app_name}' (job={job_id})")

        except Exception as e:
            logger.exception(f"[BUILD] Modification error for '{app_name}'")
            await blog.error(f"Modification error: {e}")
            await blog.persist()
            await self._fail_job(job_id, app_id, f"Modification error: {e}")

    # ── Step helpers ────────────────────────────────────────────────

    async def _step_plan(
        self, job_id: str, user_id: str, description: str, blog=None,
        extra_context: str = "",
    ) -> Optional[Dict]:
        """Run the planning step — call LLM to get app plan."""
        if blog:
            blog.set_step("planning")
        await self._update_step(job_id, user_id, "planning", "running")

        try:
            if blog:
                await blog.info("Calling Claude Opus for app architecture planning...")
            llm_response = await self._call_llm(
                PLANNING_PROMPT.format(description=description, extra_context=extra_context),
                "Plan this app and output JSON.",
                model="claude-opus-4-6",
                blog=blog,
                purpose="App architecture planning",
            )

            plan = self._extract_json(llm_response)
            if not plan:
                if blog:
                    await blog.error("Failed to parse plan JSON from LLM response",
                                     llm_response[:300])
                await self._update_step(job_id, user_id, "planning", "failed",
                                        detail="Could not parse LLM planning output")
                return None

            files_list = plan.get("files", [])
            deps_list = plan.get("dependencies", [])
            if blog:
                await blog.success(f"Plan generated: {plan.get('app_name', 'App')}")
                await blog.info(f"Summary: {plan.get('summary', 'N/A')}")
                await blog.info(f"Files planned: {len(files_list)}")
                for f in files_list:
                    await blog.debug(f"  {f}")
                await blog.info(f"Dependencies: {len(deps_list)}")
                for d in deps_list:
                    await blog.debug(f"  {d}")
                await blog.info(f"Database: {plan.get('db_type', 'none')}")

            plan_detail = (
                f"{plan.get('summary', '')}\n\n"
                f"Files ({len(files_list)}):\n" +
                "\n".join(f"  {f}" for f in files_list) +
                f"\n\nDependencies ({len(deps_list)}):\n" +
                "\n".join(f"  {d}" for d in deps_list) +
                f"\n\nDatabase: {plan.get('db_type', 'none')}"
            )
            await self._update_step(job_id, user_id, "planning", "done",
                                    detail=plan_detail.strip())
            return plan
        except Exception as e:
            if blog:
                await blog.error(f"Planning failed: {e}")
            await self._update_step(job_id, user_id, "planning", "failed", detail=str(e))
            return None

    async def _generate_code(
        self,
        description: str,
        app_name: str,
        files: List[str],
        deps: List[str],
        db_type: str,
        job_id: str = "",
        user_id: str = "",
        blog=None,
        design_notes: str = "",
        existing_files: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """Generate code for all files using LLM — parallel with semaphore."""
        generated = {}
        all_files = json.dumps(files)
        all_deps = json.dumps(deps)
        total = len(files)
        completed = 0
        lock = asyncio.Lock()
        semaphore = asyncio.Semaphore(5)

        design_section = f"Design notes: {design_notes}" if design_notes else ""

        async def _gen_one(i: int, file_path: str):
            nonlocal completed
            async with semaphore:
                if blog:
                    await blog.info(f"Generating {file_path}...", f"[{i+1}/{total}]")

                # Include existing file content for modifications
                existing_hint = ""
                if existing_files and file_path in existing_files:
                    existing_hint = (
                        f"\n\nCurrent content of {file_path} (modify this, don't start from scratch):\n"
                        f"```\n{existing_files[file_path][:4000]}\n```"
                    )

                prompt = CODE_GEN_PROMPT.format(
                    file_path=file_path,
                    description=description,
                    app_name=app_name,
                    all_files=all_files,
                    all_deps=all_deps,
                    db_type=db_type,
                    design_notes=design_section,
                )
                if existing_hint:
                    prompt += existing_hint

                try:
                    code = await self._call_llm(
                        prompt, f"Generate code for {file_path}",
                        blog=blog, purpose=f"Generate {file_path}",
                    )
                    code = self._strip_fences(code)
                    async with lock:
                        generated[file_path] = code
                        completed += 1
                    if blog:
                        await blog.file_written(file_path, len(code.encode()))
                except Exception as e:
                    if blog:
                        await blog.error(f"Failed to generate {file_path}: {e}")
                    async with lock:
                        generated[file_path] = f"// Error generating {file_path}: {e}\nexport default function() {{ return null; }}"
                        completed += 1

                if job_id:
                    async with lock:
                        current = completed
                    await self._update_step(
                        job_id, user_id, "writing", "running",
                        detail=f"File {current}/{total}: {file_path}"
                    )

        await asyncio.gather(*[_gen_one(i, fp) for i, fp in enumerate(files)])
        return generated

    async def _call_llm(
        self, system_prompt: str, user_message: str, model: str = "claude-sonnet-4-6",
        blog=None, purpose: str = "",
    ) -> str:
        """Call the LLM (Anthropic or OpenAI). Default: Sonnet for speed, Opus for planning."""
        import time as _time
        from app.config import settings

        anthropic_key = settings.anthropic_api_key or ""
        if anthropic_key:
            try:
                import anthropic
                import os
                is_oauth = "sk-ant-oat" in anthropic_key
                if is_oauth:
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

                t0 = _time.time()
                response = await client.messages.create(
                    model=model,
                    max_tokens=8192,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_message}],
                )
                elapsed = _time.time() - t0
                text = response.content[0].text
                input_tok = getattr(response.usage, 'input_tokens', 0)
                output_tok = getattr(response.usage, 'output_tokens', 0)

                if blog:
                    await blog.llm_call(
                        model=model,
                        purpose=purpose or user_message[:50],
                        input_tokens=input_tok,
                        output_tokens=output_tok,
                        duration_s=elapsed,
                    )

                return text
            except Exception as e:
                if blog:
                    await blog.warn(f"Anthropic call failed, falling back to OpenAI: {e}")
                else:
                    logger.warning(f"[BUILD] Anthropic call failed, falling back to OpenAI: {e}")

        if settings.openai_api_key:
            try:
                from openai import AsyncOpenAI
                client = AsyncOpenAI(api_key=settings.openai_api_key)
                oai_model = settings.agent_model or "gpt-4o-mini"
                t0 = _time.time()
                response = await client.chat.completions.create(
                    model=oai_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    max_completion_tokens=8192,
                )
                elapsed = _time.time() - t0
                text = response.choices[0].message.content or ""
                usage = response.usage
                if blog and usage:
                    await blog.llm_call(
                        model=oai_model,
                        purpose=purpose or user_message[:50],
                        input_tokens=usage.prompt_tokens or 0,
                        output_tokens=usage.completion_tokens or 0,
                        duration_s=elapsed,
                    )
                return text
            except Exception as e:
                if blog:
                    await blog.error(f"OpenAI call failed: {e}")
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
        text = text.strip()
        if text.startswith("{"):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass

        patterns = [
            r"```json\s*\n(.*?)\n```",
            r"```\s*\n(.*?)\n```",
            r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}",
        ]
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
            first_newline = code.index("\n") if "\n" in code else len(code)
            code = code[first_newline + 1:]
        if code.endswith("```"):
            code = code[:-3].rstrip()
        return code
