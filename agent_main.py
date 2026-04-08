"""
Toup Agent Service — Agent runtime that lives on user's VPS.

This entry point starts ONLY the Agent: AgentRunner, ToolExecutor,
Telegram bot, Discord/Slack/WhatsApp channels, cron, skills, and hooks.
It connects to the shared Supabase PostgreSQL for memory access
and exposes HTTP/WebSocket endpoints for the platform to proxy chat.

Usage:
    uvicorn agent_main:app --host 0.0.0.0 --port 8001

This is what runs on each user's provisioned EC2 instance.
"""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI, Request, Response

# Configure logging so agent_runner [PERF] and [AGENT] logs show in journalctl
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
from fastapi.middleware.cors import CORSMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send

from app.config import settings
from app.db import init_db

# ── Agent routers (invoke AgentRunner) ───────────────────────────────
from app.api import agent_router
from app.api.stats import router as stats_router
from app.api.memories import router as memories_router
from app.api.sessions import router as sessions_router
from app.api.chat import router as chat_router
from app.api.ws_chat import router as ws_chat_router, set_ws_refs
from app.api.api_v1 import router as api_v1_router
from app.api.webhooks import router as webhooks_router, set_webhook_refs
from app.api.voice import router as voice_router, set_voice_refs
from app.api.ws_realtime import router as ws_realtime_router, set_realtime_refs
from app.api.ws_browser import router as ws_browser_router, set_ws_browser_refs
from app.api.dashboard import router as dashboard_router
from app.api.soul import router as soul_router
from app.api.llm_setup import router as llm_setup_router
from app.api.apps import router as apps_router, set_app_manager, set_app_gateway, set_app_builder_skill

_app_start_time = None
_skill_loader = None

# ── Paths that skip API key auth (health checks, root) ─────────────
_PUBLIC_PATHS = frozenset({"/", "/agent/health", "/agent/system", "/docs", "/openapi.json", "/redoc"})


# ── Module-level refs for hot-restart of channel bots ──────────────
_telegram_bot = None
_agent_runner = None
_tool_executor = None
_subagent_manager = None
_skill_loader = None
_cron_service = None


async def restart_telegram_bot():
    """Hot-restart the Telegram bot with a new token from settings.

    Called by tunnel_client after config sync when the token changes.
    Stops the old bot (if running) and starts a new one.
    """
    global _telegram_bot
    from app.config import settings
    from app.agent.telegram_bot import ToupTelegramBot

    new_token = (settings.telegram_bot_token or "").strip()

    # Stop existing bot
    if _telegram_bot:
        try:
            await _telegram_bot.stop()
            print("🛑 Telegram bot stopped (token changed)")
        except Exception as e:
            logging.warning(f"[RESTART] Bot stop error: {e}")
        _telegram_bot = None

    if not new_token:
        print("ℹ️  No Telegram token — bot not started")
        return

    if not _agent_runner:
        logging.warning("[RESTART] Cannot start Telegram bot — agent_runner not initialized")
        return

    try:
        _telegram_bot = ToupTelegramBot(
            token=new_token,
            agent_runner=_agent_runner,
        )
        if _tool_executor:
            _tool_executor.telegram_bot = _telegram_bot
        if _subagent_manager:
            _subagent_manager.set_bot(_telegram_bot)
            _telegram_bot.subagent_manager = _subagent_manager
        if _skill_loader:
            _telegram_bot.skill_loader = _skill_loader
        if _cron_service:
            _cron_service.set_bot(_telegram_bot)
            _telegram_bot.cron_service = _cron_service

        await _telegram_bot.start()
        print("🤖 Telegram bot restarted with new token")

        from app.api.admin import set_bot_refs
        set_bot_refs(_telegram_bot, _cron_service, _telegram_bot._start_time)
    except Exception as e:
        logging.exception(f"[RESTART] Failed to start Telegram bot: {e}")
        _telegram_bot = None


class AgentAPIKeyMiddleware:
    """Raw ASGI middleware that validates the X-Agent-Key header.

    Uses raw ASGI instead of BaseHTTPMiddleware so WebSocket connections
    pass through correctly (BaseHTTPMiddleware blocks WebSockets).
    WebSocket endpoints handle their own auth via query params.
    """

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        # Let WebSocket connections through — they handle their own auth
        if scope["type"] == "websocket":
            await self.app(scope, receive, send)
            return

        # Only check HTTP requests
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Skip auth if no key is configured (local dev / monolith mode)
        if not settings.agent_api_key:
            await self.app(scope, receive, send)
            return

        # Allow public endpoints without auth
        path = scope.get("path", "")
        if path in _PUBLIC_PATHS:
            await self.app(scope, receive, send)
            return

        # Check the API key from headers
        headers = dict(scope.get("headers", []))
        provided_key = headers.get(b"x-agent-key", b"").decode()

        # Also check query params (for backwards compat)
        if not provided_key:
            qs = scope.get("query_string", b"").decode()
            for part in qs.split("&"):
                if part.startswith("agent_key="):
                    provided_key = part[len("agent_key="):]
                    break

        if provided_key != settings.agent_api_key:
            response = Response(
                content='{"detail":"Invalid or missing agent API key"}',
                status_code=401,
                media_type="application/json",
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _app_start_time
    import time as _time
    _app_start_time = _time.time()

    # ── Startup ───────────────────────────────────────────────
    print("🤖 Toup Agent starting up...")
    await init_db()
    print("✅ Database initialized")

    # ── Auto-update check (non-blocking) ────────────────────
    try:
        import subprocess
        agent_dir = os.environ.get("AGENT_DIR") or os.path.abspath(os.path.dirname(__file__))
        result = subprocess.run(
            ["git", "pull", "--ff-only"],
            cwd=agent_dir, capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0 and "Already up to date" not in result.stdout:
            print(f"📦 Auto-updated: {result.stdout.strip()[:100]}")
            # Install any new deps
            venv_pip = os.path.join(agent_dir, "venv", "bin", "pip")
            if os.path.exists(venv_pip):
                subprocess.run(
                    [venv_pip, "install", "-q", "-r", os.path.join(agent_dir, "requirements.txt")],
                    cwd=agent_dir, capture_output=True, timeout=60,
                )
    except Exception as e:
        print(f"⚠️ Auto-update check skipped: {e}")

    # ── Ensure owner user exists in DB ─────────────────────────
    # Must run BEFORE session migration (which references user_id as FK)
    if settings.user_id:
        try:
            from app.db.database import async_session_maker as _sm
            from app.services.auth_service import get_user_by_id
            from app.db.models import User

            async with _sm() as _udb:
                owner = await get_user_by_id(_udb, settings.user_id)
                if not owner:
                    owner = User(
                        id=settings.user_id,
                        email=f"{settings.user_id[:8]}@agent.local",
                        hashed_password="",
                        name="Agent Owner",
                    )
                    _udb.add(owner)
                    await _udb.commit()
                    print(f"✅ Created owner user: {settings.user_id[:8]}...")
                else:
                    print(f"✅ Owner user exists: {settings.user_id[:8]}...")
        except Exception as e:
            print(f"⚠️ Could not ensure owner user: {e}")

    # ── Migrate orphaned Telegram sessions to platform owner ──
    # Runs before any services start to avoid lock conflicts
    if settings.user_id:
        try:
            from app.db.database import async_session_maker as _sm
            from sqlalchemy import select as _sel, update as _upd, and_ as _and
            from app.db.models import Conversation, TelegramUserMapping

            async with _sm() as _mdb:
                # Set a short timeout so this never blocks startup
                from sqlalchemy import text as _text
                await _mdb.execute(_text("SET LOCAL statement_timeout = '5s'"))

                _res = await _mdb.execute(
                    _sel(TelegramUserMapping.user_id)
                    .where(TelegramUserMapping.user_id != settings.user_id)
                    .distinct()
                )
                stale_ids = [r[0] for r in _res.all()]

                if stale_ids:
                    result = await _mdb.execute(
                        _upd(Conversation)
                        .where(_and(
                            Conversation.user_id.in_(stale_ids),
                            Conversation.channel == "telegram",
                        ))
                        .values(user_id=settings.user_id)
                    )
                    if result.rowcount:
                        await _mdb.commit()
                        print(f"📋 Migrated {result.rowcount} Telegram session(s) to platform owner")
        except Exception as e:
            print(f"⚠️ Session migration skipped: {e}")

    # Pre-load embedding service (needed for memory retrieval in system prompt)
    try:
        from app.services import get_embedding_service
        svc = get_embedding_service()
        if svc.is_openai:
            _ = svc.openai_client
            print(f"✅ Embedding service ready ({settings.embedding_model})")
        else:
            _ = svc.local_model
            print("✅ Local embedding model loaded")
    except Exception as e:
        print(f"⚠️ Could not pre-load embedding service: {e}")

    # ── Agent stack initialization ────────────────────────────
    telegram_bot = None
    cron_service = None
    subagent_manager = None
    skill_loader = None
    agent_runner = None
    app_manager = None

    try:
        from app.agent.telegram_bot import ToupTelegramBot
        from app.agent.agent_runner import AgentRunner
        from app.agent.tool_executor import ToolExecutor
        from app.services.openai_agent_service import OpenAIAgentService
        from app.agent.cron_service import CronService
        from app.agent.subagent import SubAgentManager
        from app.agent.skills.loader import SkillLoader

        # Load skills
        skill_loader = SkillLoader(extra_dirs=[settings.skills_dir])
        try:
            count = await skill_loader.load_all()
            print(f"🧩 Loaded {count} skill(s): {list(skill_loader.skills.keys())}")
        except Exception as e:
            print(f"⚠️ Skill loading error: {e}")
            skill_loader = SkillLoader()

        # NOTE: Per-app AppSkill registration removed — use AppGatewaySkill instead
        # (see app_builder block below)


        # Store skill_loader at module level for /agent/capabilities endpoint
        global _skill_loader
        _skill_loader = skill_loader

        # Wire skill_loader to App MCP server
        try:
            from app.agent.app_mcp_server import set_mcp_skill_loader
            set_mcp_skill_loader(skill_loader)
        except Exception:
            pass

        # Build the agent pipeline
        openai_agent_svc = OpenAIAgentService()
        subagent_manager = SubAgentManager()
        tool_executor = ToolExecutor(subagent_manager=subagent_manager)
        tool_executor.skill_loader = skill_loader
        agent_runner = AgentRunner(
            llm_service=openai_agent_svc,
            tool_executor=tool_executor,
            skill_loader=skill_loader,
        )

        # Wire sub-agent manager
        subagent_manager.set_agent_runner(agent_runner)

        # Cron service
        cron_service = CronService()
        cron_service.set_agent_runner(agent_runner)
        tool_executor.cron_service = cron_service

        # Wire WebSocket and API v1 refs (so they can invoke the agent)
        from app.api.api_v1 import set_api_v1_refs
        set_ws_refs(agent_runner, skill_loader)
        set_api_v1_refs(agent_runner, skill_loader)
        set_realtime_refs(tool_executor, agent_runner)
        set_ws_browser_refs(agent_runner, skill_loader)

        # ── Clean up orphaned build jobs from previous crash/restart ──
        try:
            from app.db.database import async_session_maker
            from app.db.models import BuildJob, App as AppModel
            from sqlalchemy import select as _sel

            async with async_session_maker() as _jdb:
                # Find all jobs stuck in running/queued state (orphaned by restart)
                result = await _jdb.execute(
                    _sel(BuildJob).where(BuildJob.status.in_(["running", "queued"]))
                )
                orphaned_jobs = result.scalars().all()

                for job in orphaned_jobs:
                    job.status = "failed"
                    job.error_message = "Interrupted by restart"
                    job.completed_at = datetime.utcnow()

                    # Mark any in-progress steps as failed with correct duration
                    try:
                        steps = json.loads(job.steps_json) if job.steps_json else []
                        for s in steps:
                            if s.get("status") == "running":
                                s["status"] = "failed"
                                started = s.get("started_at")
                                if started:
                                    try:
                                        start_dt = datetime.fromisoformat(started)
                                        s["duration_ms"] = int(
                                            (datetime.utcnow() - start_dt).total_seconds() * 1000
                                        )
                                    except Exception:
                                        pass
                        job.steps_json = json.dumps(steps)
                    except (json.JSONDecodeError, TypeError):
                        pass

                    # Also fix the parent app status
                    if job.app_id:
                        app = await _jdb.get(AppModel, job.app_id)
                        if app and app.status == "building":
                            app.status = "error"

                if orphaned_jobs:
                    await _jdb.commit()
                    print(f"🧹 Cleaned up {len(orphaned_jobs)} orphaned build job(s)")
        except Exception as e:
            print(f"⚠️ Orphan job cleanup skipped: {e}")

        # ── App Manager + App Builder Skill ────────────────────
        try:
            from app.agent.app_manager import AppManager
            app_manager = AppManager()
            restored = await app_manager.restore_on_startup(async_session_maker)
            set_app_manager(app_manager)
            if restored:
                print(f"📱 App Manager: restored {restored} running app(s)")
            else:
                print("📱 App Manager ready")

            # Register AppBuilderSkill
            from app.agent.skills.builtins.app_builder.skill import AppBuilderSkill
            from app.api.ws_chat import broadcast_to_user
            builder_skill = AppBuilderSkill(
                app_manager=app_manager,
                ws_broadcast=broadcast_to_user,
            )
            await skill_loader.register_dynamic(builder_skill)
            set_app_builder_skill(builder_skill)  # Wire for resume API endpoint
            print("🏗️ App Builder skill registered")

            # Register AppGatewaySkill (single skill, 12 tools for ALL apps)
            from app.agent.skills.builtins.app_builder.app_gateway_skill import AppGatewaySkill
            app_gateway = AppGatewaySkill()

            # Load existing apps into the gateway (not as separate agent tools)
            try:
                from app.agent.skills.builtins.app_builder.app_fs_skill import AppFsSkill
                from app.db.models import App
                from sqlalchemy import select as sa_select
                async with async_session_maker() as _db:
                    result = await _db.execute(
                        sa_select(App).where(App.status.in_(["running", "ready", "stopped"]))
                    )
                    db_apps = result.scalars().all()
                    for db_app in db_apps:
                        try:
                            fs_skill = AppFsSkill(
                                app_id=db_app.id,
                                app_name=db_app.name,
                                app_slug=db_app.slug,
                                app_dir=db_app.app_dir,
                                app_manager=app_manager,
                            )
                            app_gateway.register_app(db_app.slug, fs_skill)
                        except Exception as e:
                            logger.debug(f"[AppGateway] Skipping app {db_app.id}: {e}")
                if db_apps:
                    print(f"📱 App Gateway: {len(db_apps)} app(s) loaded")
            except Exception as e:
                print(f"⚠️ App Gateway loading error: {e}")

            await skill_loader.register_dynamic(app_gateway)
            builder_skill._app_gateway = app_gateway  # so builder can register new apps
            set_app_gateway(app_gateway)  # so delete endpoint can unregister apps
            print(f"📱 App Gateway skill registered ({len(app_gateway.get_tools())} tools for all apps)")
        except Exception as e:
            print(f"⚠️ App Manager/Builder error: {e}")

        # ── Store module refs for hot-restart ────────────────
        import agent_main as _self
        _self._agent_runner = agent_runner
        _self._tool_executor = tool_executor
        _self._subagent_manager = subagent_manager
        _self._skill_loader = skill_loader
        _self._cron_service = cron_service

        # ── Start Telegram bot (if configured) ────────────────
        if settings.telegram_bot_token:
            telegram_bot = ToupTelegramBot(
                token=settings.telegram_bot_token,
                agent_runner=agent_runner,
            )
            tool_executor.telegram_bot = telegram_bot
            subagent_manager.set_bot(telegram_bot)
            telegram_bot.subagent_manager = subagent_manager
            telegram_bot.skill_loader = skill_loader
            cron_service.set_bot(telegram_bot)
            telegram_bot.cron_service = cron_service

            await telegram_bot.start()
            _self._telegram_bot = telegram_bot
            print("🤖 Telegram bot started")

            # Admin dashboard refs
            from app.api.admin import set_bot_refs
            set_bot_refs(telegram_bot, cron_service, telegram_bot._start_time)

            # Webhook refs
            set_webhook_refs(agent_runner, telegram_bot)

        # Start cron scheduler
        try:
            await cron_service.start()
            print("⏰ Cron service started")
        except Exception as e:
            print(f"⚠️ Could not start cron service: {e}")
            cron_service = None

        # Heartbeat service
        if settings.heartbeat_enabled and cron_service:
            try:
                from app.agent.heartbeat_service import HeartbeatService
                from apscheduler.triggers.interval import IntervalTrigger

                heartbeat_svc = HeartbeatService()
                heartbeat_svc.set_agent_runner(agent_runner)
                if telegram_bot:
                    heartbeat_svc.set_bot(telegram_bot)
                if cron_service.scheduler:
                    cron_service.scheduler.add_job(
                        heartbeat_svc.tick,
                        trigger=IntervalTrigger(hours=settings.heartbeat_interval_hours),
                        id="heartbeat",
                        name="Proactive Agent Heartbeat",
                        replace_existing=True,
                    )
                    print(f"💓 Heartbeat (every {settings.heartbeat_interval_hours}h)")
            except Exception as e:
                print(f"⚠️ Could not start heartbeat: {e}")

        # ── Periodic self-update check (every 6 hours) ────────
        if cron_service and cron_service.scheduler:
            try:
                from apscheduler.triggers.interval import IntervalTrigger as _IT

                async def _periodic_update():
                    """Check for updates and restart if new code is available."""
                    import subprocess as _sp, os as _os, sys as _sys
                    agent_dir = _os.environ.get("AGENT_DIR") or _os.path.abspath(_os.path.dirname(__file__))
                    git_dir = _os.path.join(agent_dir, ".git")
                    if not _os.path.isdir(git_dir):
                        return  # No git repo, skip

                    # Fetch and check if behind
                    _sp.run(["git", "fetch", "--depth", "1", "origin", "main"],
                            cwd=agent_dir, capture_output=True, timeout=15)
                    result = _sp.run(["git", "rev-parse", "HEAD"], cwd=agent_dir, capture_output=True, text=True, timeout=5)
                    local = result.stdout.strip()
                    result = _sp.run(["git", "rev-parse", "origin/main"], cwd=agent_dir, capture_output=True, text=True, timeout=5)
                    remote = result.stdout.strip()

                    if local == remote:
                        return  # Already up to date

                    print(f"📦 Update available: {local[:8]} → {remote[:8]}, updating...")

                    # Pull + install deps
                    _sp.run(["git", "checkout", "-f", "origin/main"], cwd=agent_dir, capture_output=True, timeout=15)
                    _sp.run(["git", "branch", "-f", "main", "origin/main"], cwd=agent_dir, capture_output=True, timeout=5)
                    _sp.run(["git", "checkout", "main"], cwd=agent_dir, capture_output=True, timeout=5)

                    venv_pip = _os.path.join(agent_dir, "venv", "bin", "pip")
                    if _os.path.exists(venv_pip):
                        _sp.run([venv_pip, "install", "-q", "-r", _os.path.join(agent_dir, "requirements.txt")],
                                cwd=agent_dir, capture_output=True, timeout=120)

                    # Restart — re-exec the process
                    print("🔄 Restarting agent with new code...")
                    _os.execv(_sys.executable, [_sys.executable] + _sys.argv)

                cron_service.scheduler.add_job(
                    _periodic_update,
                    trigger=_IT(hours=6),
                    id="auto_update",
                    name="Periodic Self-Update",
                    replace_existing=True,
                )
                print("🔄 Auto-update check (every 6h)")
            except Exception as e:
                print(f"⚠️ Could not start auto-update: {e}")

    except Exception as e:
        print(f"⚠️ Agent initialization error: {e}")
        import traceback
        traceback.print_exc()

    # ── Hook Bus ──────────────────────────────────────────────
    from app.agent.hooks import get_hook_bus, HookEvent
    _hook_bus = get_hook_bus()
    await _hook_bus.emit(HookEvent.STARTUP, {"app": "toup-agent"})
    print("🔌 Hook bus started")

    # ── MCP Client (connect to Platform MCP server) ──────────
    mcp_client = None
    if settings.platform_api_url and settings.agent_api_key:
        try:
            from fastmcp import Client as MCPClient

            mcp_url = f"{settings.platform_api_url}/mcp"
            mcp_client = MCPClient(mcp_url)

            # List available MCP tools (non-blocking discovery)
            try:
                async with mcp_client:
                    tools = await mcp_client.list_tools()
                    tool_names = [t.name for t in tools]
                    print(f"🔗 MCP connected ({len(tool_names)} tools): {tool_names}")

                    # Register MCP tools with the agent's tool executor
                    if tool_executor:
                        tool_executor.mcp_client = mcp_client
                        tool_executor.mcp_tools = tool_names
            except Exception as e:
                print(f"⚠️ MCP tool discovery failed (will retry on use): {e}")
                # Store client anyway — tools can be discovered lazily
                if tool_executor:
                    tool_executor.mcp_client = mcp_client
                    tool_executor.mcp_tools = []
        except ImportError:
            print("⚠️ fastmcp not installed — MCP client disabled")
        except Exception as e:
            print(f"⚠️ MCP client error: {e}")

    # ── Discord Channel ───────────────────────────────────────
    discord_channel = None
    if settings.discord_bot_token:
        try:
            from app.agent.channels.discord_channel import DiscordChannel
            from app.agent.channels.registry import ChannelRegistry
            discord_channel = DiscordChannel(
                bot_token=settings.discord_bot_token,
                allowed_guilds=settings.discord_allowed_guilds or None,
                allowed_users=settings.discord_allowed_users or None,
            )
            await discord_channel.start()
            ChannelRegistry.register(discord_channel)
            print("💬 Discord channel started")
        except Exception as e:
            print(f"⚠️ Discord error: {e}")

    # ── Slack Channel ─────────────────────────────────────────
    slack_channel = None
    if settings.slack_bot_token and settings.slack_app_token:
        try:
            from app.agent.channels.slack_channel import SlackChannel
            from app.agent.channels.registry import ChannelRegistry
            slack_channel = SlackChannel(
                bot_token=settings.slack_bot_token,
                app_token=settings.slack_app_token,
                allowed_channels=settings.slack_allowed_channels or None,
            )
            await slack_channel.start()
            ChannelRegistry.register(slack_channel)
            print("💼 Slack channel started")
        except Exception as e:
            print(f"⚠️ Slack error: {e}")

    # ── WhatsApp Channel ──────────────────────────────────────
    whatsapp_channel = None
    if settings.whatsapp_phone_number_id and settings.whatsapp_access_token:
        try:
            from app.agent.channels.whatsapp_channel import WhatsAppChannel
            from app.agent.channels.registry import ChannelRegistry
            whatsapp_channel = WhatsAppChannel(
                phone_number_id=settings.whatsapp_phone_number_id,
                access_token=settings.whatsapp_access_token,
                verify_token=settings.whatsapp_verify_token,
                app_secret=settings.whatsapp_app_secret,
                allowed_numbers=settings.whatsapp_allowed_numbers or None,
            )
            whatsapp_channel.register_routes(app)
            await whatsapp_channel.start()
            ChannelRegistry.register(whatsapp_channel)
            print("📱 WhatsApp channel started")
        except Exception as e:
            print(f"⚠️ WhatsApp error: {e}")

    # ── Platform Tunnel (connect terminal agent to toup.ai) ──
    tunnel_client = None
    if settings.platform_api_url and settings.user_id and settings.toup_token and tool_executor:
        try:
            from app.agent.tunnel_client import AgentTunnelClient

            tunnel_client = AgentTunnelClient(
                platform_url=settings.platform_api_url,
                auth_token=settings.toup_token,
                tool_executor=tool_executor,
            )
            await tunnel_client.start()
            print("🔗 Platform tunnel connecting...")
        except Exception as e:
            print(f"⚠️ Platform tunnel not available: {e}")
    elif settings.user_id and not settings.toup_token:
        print("💡 To connect to toup.ai, generate a Connect Token in Agent Settings and pass it as TOUP_TOKEN")

    # ── Self-register with platform (belt-and-suspenders) ────
    if settings.platform_api_url and settings.agent_api_key:
        async def _self_register():
            """Detect public IP and register with platform. Non-blocking."""
            import httpx as _hx
            try:
                async with _hx.AsyncClient(timeout=8.0) as _c:
                    # Detect public IP
                    try:
                        _ip_resp = await _c.get("https://api.ipify.org")
                        public_ip = _ip_resp.text.strip()
                    except Exception:
                        try:
                            import subprocess as _sp
                            _r = _sp.run(["hostname", "-I"], capture_output=True, text=True, timeout=3)
                            public_ip = _r.stdout.strip().split()[0] if _r.returncode == 0 else None
                        except Exception:
                            public_ip = None
                    if not public_ip:
                        print("⚠️ Could not detect public IP for registration", flush=True)
                        return
                    agent_url = f"http://{public_ip}:{settings.agent_port}"
                    # Ensure /api prefix is included
                    base = settings.platform_api_url.rstrip("/")
                    if not base.endswith("/api"):
                        base = base + "/api"
                    reg_url = f"{base}/agent-setup/register"
                    print(f"📡 Registering with platform: {reg_url} as {agent_url}", flush=True)
                    resp = await _c.post(
                        reg_url,
                        json={"agent_api_key": settings.agent_api_key, "agent_url": agent_url},
                        timeout=10.0,
                    )
                    if resp.status_code == 200:
                        print(f"✅ Registered with platform ({agent_url})", flush=True)
                    else:
                        print(f"⚠️ Platform register returned {resp.status_code}: {resp.text[:200]}", flush=True)
            except Exception as e:
                print(f"⚠️ Platform registration failed: {e}", flush=True)
        asyncio.create_task(_self_register())

    # ── Background reconciliation sweep (every 5 min) ──────────
    _reconcile_task = None

    async def _reconcile_sweep():
        """Periodically reconcile filesystem ↔ DB for active user.

        Only runs when the user has an active WebSocket connection (app is open).
        Skips idle/logged-out users to avoid wasting cycles.
        """
        import os
        from app.services.reconciliation_service import reconcile_apps
        from app.agent.app_manager import APPS_DIR
        from app.api.ws_chat import _user_ws_queues

        workspace = getattr(settings, 'agent_workspace_dir', None) or './workspace'
        user_id = settings.user_id

        async def _local_cmd(cmd: str) -> str:
            proc = await asyncio.create_subprocess_shell(
                cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            return stdout.decode().strip()

        while True:
            await asyncio.sleep(300)  # 5 minutes
            # Skip if no active WebSocket connections (user is idle/logged out)
            if not _user_ws_queues.get(user_id):
                continue
            try:
                async with async_session_maker() as db:
                    await reconcile_apps(
                        user_id=user_id, db=db, ssh_cmd_fn=_local_cmd,
                        trigger="sweep", apps_dir=APPS_DIR,
                        workspace_dir=os.path.abspath(workspace), force=True,
                    )
            except Exception as e:
                logging.getLogger(__name__).warning("[SWEEP] Reconciliation failed: %s", e)

    _reconcile_task = asyncio.create_task(_reconcile_sweep())

    # ── DayChat backfill (non-blocking background task) ─────
    # Fires AFTER all services are initialized. The agent is fully
    # responsive immediately — backfill runs in the background.
    # Feature flag USE_DAY_CHAT_CONTEXT stays off until backfill is
    # verified complete. Backfill failure is never fatal to the agent.
    _backfill_task = None
    try:
        from app.db.database import async_session_maker as _bsm

        async def _run_day_chat_backfill():
            try:
                # Import without triggering app.services.__init__
                import importlib.util as _ilu
                _spec = _ilu.spec_from_file_location(
                    "backfill_day_chats",
                    os.path.join(os.path.dirname(__file__), "app", "services", "backfill_day_chats.py"),
                )
                _bmod = _ilu.module_from_spec(_spec)
                _spec.loader.exec_module(_bmod)

                result = await _bmod.run_backfill(_bsm)
                if result == "already_completed":
                    logger.info("day_chat_backfill.skipped reason=already_completed")
                elif result == "completed":
                    logger.info("day_chat_backfill.done")
                elif result == "failed":
                    logger.error("day_chat_backfill.skipped reason=previously_failed")
            except Exception as e:
                # Backfill failure is never fatal — agent continues on old path
                logger.error("day_chat_backfill.crash error=%s", e)

        _backfill_task = asyncio.create_task(_run_day_chat_backfill())
        logger.info("day_chat_backfill.scheduled")
    except Exception as e:
        logger.warning("day_chat_backfill.skipped reason=import_error error=%s", e)

    print("🤖 Toup Agent ready.")
    print(f"   Server:  http://0.0.0.0:8001")
    print(f"   Health:  http://localhost:8001/agent/health")
    print(f"   Web UI:  https://toup.ai")
    print(f"   Press Ctrl+C to stop.\n")
    yield

    # ── Shutdown (reverse order) ──────────────────────────────
    if _backfill_task and not _backfill_task.done():
        _backfill_task.cancel()
    if _reconcile_task:
        _reconcile_task.cancel()

    await _hook_bus.emit(HookEvent.SHUTDOWN, {"app": "toup-agent"})
    print("🤖 Toup Agent shutting down...")

    if tunnel_client:
        try:
            await tunnel_client.stop()
            print("🔗 Platform tunnel disconnected")
        except Exception:
            pass

    if cron_service:
        try:
            await cron_service.stop()
            print("⏰ Cron stopped")
        except Exception:
            pass

    if subagent_manager:
        try:
            active = [r for r in subagent_manager._runs.values() if r.status == "running"]
            for run in active:
                if run._task_handle and not run._task_handle.done():
                    run._task_handle.cancel()
            if active:
                print(f"🧵 Cancelled {len(active)} sub-agent tasks")
        except Exception:
            pass

    if app_manager:
        try:
            await app_manager.cleanup()
            print("📱 App Manager cleaned up")
        except Exception:
            pass

    if skill_loader:
        try:
            await skill_loader.unload_all()
            print("🧩 Skills unloaded")
        except Exception:
            pass

    for ch_name, ch_obj in [
        ("Discord", discord_channel),
        ("Slack", slack_channel),
        ("WhatsApp", whatsapp_channel),
    ]:
        if ch_obj:
            try:
                await ch_obj.stop()
                print(f"📴 {ch_name} stopped")
            except Exception:
                pass

    if telegram_bot:
        try:
            await telegram_bot.stop()
            print("🤖 Telegram bot stopped")
        except Exception:
            pass

    print("🤖 Toup Agent shutdown complete.")


app = FastAPI(
    title="Toup Agent",
    description="Personal AI Agent with tools, channels, and memory access",
    version="6.0.0",
    lifespan=lifespan,
)

# CORS — allow the platform frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Agent VPS accepts requests from toup.ai frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API key auth — only when AGENT_API_KEY is configured (on user VPS)
app.add_middleware(AgentAPIKeyMiddleware)

# ── Register agent routers ───────────────────────────────────────────
app.include_router(agent_router, prefix=settings.api_prefix)
app.include_router(stats_router, prefix=settings.api_prefix)
app.include_router(memories_router, prefix=settings.api_prefix)
app.include_router(sessions_router, prefix=settings.api_prefix)
app.include_router(chat_router, prefix=settings.api_prefix)
app.include_router(ws_chat_router, prefix=settings.api_prefix)
app.include_router(api_v1_router, prefix=settings.api_prefix)
app.include_router(webhooks_router, prefix=settings.api_prefix)
app.include_router(voice_router, prefix=settings.api_prefix)
app.include_router(ws_realtime_router, prefix=settings.api_prefix)
app.include_router(dashboard_router, prefix=settings.api_prefix)
app.include_router(ws_browser_router, prefix=settings.api_prefix)
app.include_router(apps_router, prefix=settings.api_prefix)
# Netflix streaming (HLS)
try:
    from app.api.ws_netflix import router as netflix_stream_router
    app.include_router(netflix_stream_router, prefix=settings.api_prefix)
except ImportError as e:
    print(f"⚠️ Netflix stream not mounted: {e}")
app.include_router(soul_router, prefix=settings.api_prefix)
app.include_router(llm_setup_router, prefix=settings.api_prefix)

# Mount App MCP server for external MCP clients
try:
    from app.agent.app_mcp_server import app_mcp, set_mcp_skill_loader
    mcp_app = app_mcp.http_app(path="/mcp")
    app.mount("/api/app-mcp", mcp_app)
except Exception as _mcp_err:
    print(f"⚠️ App MCP server not mounted: {_mcp_err}")


@app.get("/")
def _get_version() -> str:
    """Get version from git commit hash, fallback to 6.0.0."""
    try:
        import subprocess
        agent_dir = os.environ.get("AGENT_DIR") or os.path.abspath(os.path.dirname(__file__))
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=agent_dir,
                                capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return f"6.0.0-{result.stdout.strip()}"
    except Exception:
        pass
    return "6.0.0"

_agent_version = _get_version()


@app.get("/")
async def root():
    return {
        "name": "Toup Agent",
        "status": "healthy",
        "version": _agent_version,
        "mode": "agent",
    }


@app.get("/agent/health")
async def agent_health():
    import time as _time
    uptime = _time.time() - _app_start_time if _app_start_time else 0
    return {
        "status": "healthy",
        "version": _agent_version,
        "mode": "agent",
        "uptime_seconds": round(uptime, 1),
        "agent_model": settings.agent_model,
        "channels": {
            "telegram": "enabled" if settings.telegram_bot_token else "disabled",
            "discord": "enabled" if settings.discord_bot_token else "disabled",
            "slack": "enabled" if settings.slack_bot_token else "disabled",
            "whatsapp": "enabled" if settings.whatsapp_phone_number_id else "disabled",
        },
    }


@app.get("/agent/capabilities")
async def agent_capabilities():
    """Return all loaded tools and skills (including dynamic app skills)."""
    from app.agent.tool_definitions import get_agent_tools, get_extended_tools

    # Core tools
    core_tools = []
    for t in get_agent_tools() + get_extended_tools():
        core_tools.append({
            "name": t["name"],
            "description": t.get("description", ""),
        })

    # Skills (including app builder + per-app filesystem skills)
    skills = []
    if _skill_loader:
        for s in _skill_loader.get_summary():
            skills.append({
                "name": s["name"],
                "version": s.get("version", ""),
                "description": s.get("description", ""),
                "tools": s.get("tools", []),
            })

    # App domain actions (from agentSkill.json manifests)
    app_actions = []
    if _skill_loader:
        app_skill = _skill_loader.skills.get("app")
        if app_skill and hasattr(app_skill, 'get_app_actions_summary'):
            app_actions = app_skill.get_app_actions_summary()

    return {
        "core_tools": core_tools,
        "skills": skills,
        "app_actions": app_actions,
        "total_tools": (
            len(core_tools)
            + sum(len(s["tools"]) for s in skills)
            + sum(len(a["actions"]) for a in app_actions)
        ),
    }


@app.get("/agent/system")
async def agent_system_info():
    """Return VPS system resource info (CPU, RAM, disk, OS)."""
    import time as _time, os, shutil
    uptime = _time.time() - _app_start_time if _app_start_time else 0

    # CPU
    cpu_count = os.cpu_count() or 1
    load_1m = 0.0
    try:
        load_1m = os.getloadavg()[0]
    except (OSError, AttributeError):
        pass

    # Memory
    mem_total_mb, mem_used_mb = 0, 0
    try:
        with open("/proc/meminfo") as f:
            info = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    info[parts[0].rstrip(":")] = int(parts[1])
            mem_total_mb = info.get("MemTotal", 0) // 1024
            mem_avail_mb = info.get("MemAvailable", info.get("MemFree", 0)) // 1024
            mem_used_mb = mem_total_mb - mem_avail_mb
    except Exception:
        pass

    # Disk
    disk_total_gb, disk_used_gb = 0, 0
    try:
        usage = shutil.disk_usage("/")
        disk_total_gb = round(usage.total / (1024 ** 3), 1)
        disk_used_gb = round(usage.used / (1024 ** 3), 1)
    except Exception:
        pass

    # OS
    os_name = "Linux"
    try:
        with open("/etc/os-release") as f:
            for line in f:
                if line.startswith("PRETTY_NAME="):
                    os_name = line.split("=", 1)[1].strip().strip('"')
                    break
    except Exception:
        pass

    # Running apps count
    apps_running = 0
    try:
        from app.agent.app_manager import app_manager as _am
        if _am:
            apps_running = len([a for a in _am._running.values() if a.metro_process or a.web_process])
    except Exception:
        pass

    return {
        "cpu_count": cpu_count,
        "load_1m": round(load_1m, 2),
        "mem_total_mb": mem_total_mb,
        "mem_used_mb": mem_used_mb,
        "disk_total_gb": disk_total_gb,
        "disk_used_gb": disk_used_gb,
        "os": os_name,
        "hostname": os.uname().nodename if hasattr(os, "uname") else "unknown",
        "uptime_seconds": round(uptime, 1),
        "apps_running": apps_running,
        "agent_model": settings.agent_model,
        "version": "6.0.0",
    }


@app.post("/agent/diagnose")
async def agent_diagnose():
    """Comprehensive self-diagnostic — checks everything on the VPS.

    Runs from inside the agent process, so no SSH needed.
    Returns structured checks with auto-fix results.
    """
    import asyncio
    import os
    import shutil
    import subprocess

    agent_dir = os.environ.get("AGENT_DIR") or os.path.abspath(os.path.dirname(__file__))
    checks = []

    def add_check(name, status, detail, fixed=False):
        checks.append({"name": name, "status": status, "detail": detail, "fixed": fixed})

    # 1. Disk space
    try:
        usage = shutil.disk_usage("/")
        free_mb = usage.free // (1024 * 1024)
        if free_mb < 500:
            # Try to free space
            for cmd in [
                "apt-get clean 2>/dev/null || true",
                "journalctl --vacuum-time=1d 2>/dev/null || true",
            ]:
                subprocess.run(cmd, shell=True, capture_output=True, timeout=10)
            usage_after = shutil.disk_usage("/")
            free_after = usage_after.free // (1024 * 1024)
            if free_after < 500:
                add_check("disk_space", "error", f"Only {free_after}MB free (need 500MB+)")
            else:
                add_check("disk_space", "fixed", f"Freed space: {free_mb}MB → {free_after}MB", True)
        else:
            add_check("disk_space", "ok", f"{free_mb}MB available")
    except Exception as e:
        add_check("disk_space", "warning", str(e))

    # 2. Memory usage
    try:
        import psutil
        mem = psutil.virtual_memory()
        mem_used_pct = mem.percent
        mem_avail_mb = mem.available // (1024 * 1024)
        if mem_used_pct > 90:
            add_check("memory", "warning", f"{mem_used_pct}% used, {mem_avail_mb}MB available")
        else:
            add_check("memory", "ok", f"{mem_used_pct}% used, {mem_avail_mb}MB available")
    except ImportError:
        # psutil not installed — use /proc/meminfo
        try:
            with open("/proc/meminfo") as f:
                lines = f.readlines()
            mem_info = {}
            for line in lines[:5]:
                parts = line.split(":")
                mem_info[parts[0].strip()] = int(parts[1].strip().split()[0])
            total = mem_info.get("MemTotal", 0)
            avail = mem_info.get("MemAvailable", mem_info.get("MemFree", 0))
            pct = round((1 - avail / max(total, 1)) * 100, 1)
            add_check("memory", "ok" if pct < 90 else "warning", f"{pct}% used, {avail // 1024}MB available")
        except Exception:
            add_check("memory", "warning", "Could not check memory")

    # 3. Git status — is code up to date?
    try:
        git_dir = os.path.join(agent_dir, ".git")
        if os.path.isdir(git_dir):
            subprocess.run(
                ["git", "config", "--global", "--add", "safe.directory", agent_dir],
                capture_output=True, timeout=5,
            )
            # Get local hash
            r = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=agent_dir, capture_output=True, text=True, timeout=5,
            )
            local_hash = r.stdout.strip() if r.returncode == 0 else "unknown"

            # Fetch remote
            r = subprocess.run(
                ["git", "fetch", "--depth", "1", "origin", "main"],
                cwd=agent_dir, capture_output=True, text=True, timeout=15,
            )
            if r.returncode == 0:
                r2 = subprocess.run(
                    ["git", "rev-parse", "--short", "origin/main"],
                    cwd=agent_dir, capture_output=True, text=True, timeout=5,
                )
                remote_hash = r2.stdout.strip() if r2.returncode == 0 else "unknown"
                if local_hash == remote_hash:
                    add_check("git_status", "ok", f"Up to date ({local_hash})")
                else:
                    add_check("git_status", "warning", f"Behind: local={local_hash} remote={remote_hash}")
            else:
                add_check("git_status", "warning", f"Fetch failed: {r.stderr.strip()[:100]}")
        else:
            add_check("git_status", "error", "No .git directory — code may not be managed by git")
    except Exception as e:
        add_check("git_status", "warning", str(e)[:100])

    # 4. Database connectivity
    try:
        from app.db.database import async_session_maker
        async with async_session_maker() as db:
            from sqlalchemy import text
            result = await db.execute(text("SELECT 1"))
            result.scalar()
        add_check("database", "ok", "Database is accessible")
    except Exception as e:
        add_check("database", "error", f"DB error: {str(e)[:150]}")

    # 5. .env file — check required vars
    try:
        env_path = os.path.join(agent_dir, ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                env_content = f.read()
            required = ["DATABASE_URL", "USER_ID", "AGENT_API_KEY", "RUN_MODE"]
            missing = [v for v in required if f"{v}=" not in env_content]
            if missing:
                add_check("env_file", "warning", f"Missing vars: {', '.join(missing)}")
            else:
                add_check("env_file", "ok", "All required vars present")
        else:
            add_check("env_file", "error", "No .env file found")
    except Exception as e:
        add_check("env_file", "warning", str(e)[:100])

    # 6. Python & venv
    try:
        import sys
        py_ver = f"Python {sys.version.split()[0]}"
        venv = os.path.join(agent_dir, "venv")
        if os.path.exists(venv):
            add_check("python", "ok", f"{py_ver}, venv exists")
        else:
            add_check("python", "warning", f"{py_ver}, but no venv directory")
    except Exception as e:
        add_check("python", "warning", str(e)[:100])

    # 7. Systemd service
    try:
        r = subprocess.run(
            ["systemctl", "is-active", "toup-agent"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            add_check("systemd_service", "ok", "toup-agent is active")
        else:
            # Try to check why it's not active
            r2 = subprocess.run(
                ["systemctl", "status", "toup-agent", "--no-pager"],
                capture_output=True, text=True, timeout=5,
            )
            status_out = r2.stdout.strip()[:200] if r2.stdout else r.stdout.strip()
            add_check("systemd_service", "warning", f"Status: {status_out}")
    except FileNotFoundError:
        add_check("systemd_service", "warning", "systemctl not found (not on systemd)")
    except Exception as e:
        add_check("systemd_service", "warning", str(e)[:100])

    # 8. Port 8001 — are we actually listening
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(("127.0.0.1", 8001))
        sock.close()
        if result == 0:
            add_check("port_8001", "ok", "Listening on port 8001")
        else:
            add_check("port_8001", "error", "Port 8001 not listening")
    except Exception as e:
        add_check("port_8001", "warning", str(e)[:100])

    # 9. Uptime & process health
    import time as _time
    uptime = _time.time() - _app_start_time if _app_start_time else 0
    add_check("uptime", "ok", f"{int(uptime)}s ({int(uptime // 3600)}h {int((uptime % 3600) // 60)}m)")

    # 10. Recent errors from journal (last 5 min)
    try:
        r = subprocess.run(
            ["journalctl", "-u", "toup-agent", "--no-pager", "-p", "err",
             "--since", "5 minutes ago", "-n", "5", "--output", "cat"],
            capture_output=True, text=True, timeout=5,
        )
        errors = r.stdout.strip()
        if errors:
            add_check("recent_errors", "warning", errors[:200])
        else:
            add_check("recent_errors", "ok", "No errors in last 5 minutes")
    except Exception:
        add_check("recent_errors", "ok", "Could not check journal (not critical)")

    # 11. Network — can we reach GitHub and platform?
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get("https://api.github.com/zen")
            if r.status_code == 200:
                add_check("network", "ok", "Can reach GitHub")
            else:
                add_check("network", "warning", f"GitHub returned {r.status_code}")
    except Exception as e:
        add_check("network", "warning", f"Cannot reach GitHub: {str(e)[:80]}")

    # 12. Chrome / browser (for web tools)
    try:
        r = subprocess.run(
            ["google-chrome", "--version"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            add_check("chrome", "ok", r.stdout.strip())
        else:
            r2 = subprocess.run(
                ["google-chrome-stable", "--version"],
                capture_output=True, text=True, timeout=5,
            )
            if r2.returncode == 0:
                add_check("chrome", "ok", r2.stdout.strip())
            else:
                add_check("chrome", "warning", "Chrome not found (will use Patchright Chromium)")
    except FileNotFoundError:
        add_check("chrome", "warning", "Chrome not installed")
    except Exception:
        add_check("chrome", "warning", "Could not check Chrome")

    has_errors = any(c["status"] == "error" for c in checks)
    return {
        "success": not has_errors,
        "method": "agent_self_diagnose",
        "checks": checks,
    }


@app.post("/agent/diagnose-and-fix")
async def agent_diagnose_and_fix():
    """Diagnose AND auto-fix issues. Runs git pull, pip install, restarts if needed."""
    import os
    import subprocess

    agent_dir = os.environ.get("AGENT_DIR") or os.path.abspath(os.path.dirname(__file__))
    checks = []

    def add_check(name, status, detail, fixed=False):
        checks.append({"name": name, "status": status, "detail": detail, "fixed": fixed})

    # 1. Git update
    try:
        subprocess.run(
            ["git", "config", "--global", "--add", "safe.directory", agent_dir],
            capture_output=True, timeout=5,
        )
        r = subprocess.run(
            ["git", "fetch", "--depth", "1", "origin", "main"],
            cwd=agent_dir, capture_output=True, text=True, timeout=30,
        )
        if r.returncode == 0:
            r2 = subprocess.run(
                ["git", "reset", "--hard", "origin/main"],
                cwd=agent_dir, capture_output=True, text=True, timeout=10,
            )
            local_hash_r = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=agent_dir, capture_output=True, text=True, timeout=5,
            )
            local_hash = local_hash_r.stdout.strip() if local_hash_r.returncode == 0 else "?"
            add_check("git_update", "fixed", f"Updated to {local_hash}", True)
        else:
            add_check("git_update", "error", f"Fetch failed: {r.stderr.strip()[:100]}")
    except Exception as e:
        add_check("git_update", "error", str(e)[:100])

    # 2. Install dependencies
    venv_pip = os.path.join(agent_dir, "venv", "bin", "pip")
    req_file = os.path.join(agent_dir, "requirements.txt")
    if os.path.exists(venv_pip) and os.path.exists(req_file):
        try:
            r = subprocess.run(
                [venv_pip, "install", "-q", "-r", req_file],
                cwd=agent_dir, capture_output=True, text=True, timeout=120,
            )
            if r.returncode == 0:
                add_check("dependencies", "fixed", "pip install completed", True)
            else:
                add_check("dependencies", "error", f"pip install failed: {r.stderr.strip()[:150]}")
        except Exception as e:
            add_check("dependencies", "error", str(e)[:100])
    else:
        add_check("dependencies", "warning", "pip or requirements.txt not found")

    # 3. Disk space cleanup
    try:
        import shutil
        usage = shutil.disk_usage("/")
        free_mb = usage.free // (1024 * 1024)
        if free_mb < 500:
            for cmd in [
                "apt-get clean 2>/dev/null || true",
                "journalctl --vacuum-time=1d 2>/dev/null || true",
            ]:
                subprocess.run(cmd, shell=True, capture_output=True, timeout=10)
            usage_after = shutil.disk_usage("/")
            free_after = usage_after.free // (1024 * 1024)
            add_check("disk_space", "fixed", f"Freed space: {free_mb}MB → {free_after}MB", True)
        else:
            add_check("disk_space", "ok", f"{free_mb}MB available")
    except Exception as e:
        add_check("disk_space", "warning", str(e)[:100])

    # 4. Database — run migrations (add missing columns)
    try:
        from app.db.database import init_db
        await init_db()
        add_check("database", "fixed", "Migrations applied", True)
    except Exception as e:
        add_check("database", "error", f"Migration failed: {str(e)[:150]}")

    # 5. Schedule restart
    import asyncio, sys
    needs_restart = any(c["name"] == "git_update" and c["status"] == "fixed" for c in checks)
    if needs_restart:
        async def _delayed_restart():
            await asyncio.sleep(1.5)
            service_name = os.environ.get("SYSTEMD_SERVICE") or "toup-agent"
            r = subprocess.run(
                ["systemctl", "is-active", service_name],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0:
                subprocess.Popen(["systemctl", "restart", service_name])
            else:
                os.execv(sys.executable, [sys.executable] + sys.argv)

        asyncio.get_event_loop().create_task(_delayed_restart())
        add_check("restart", "fixed", "Scheduled restart in 1.5s", True)
    else:
        add_check("restart", "ok", "No restart needed")

    has_errors = any(c["status"] == "error" for c in checks)
    return {
        "success": not has_errors,
        "method": "agent_self_fix",
        "checks": checks,
    }


@app.post("/agent/update")
async def agent_self_update():
    """Pull latest code, install deps, and restart the agent process.

    The restart works by:
    1. git pull --ff-only (or clone if no .git)
    2. pip install -r requirements.txt (if changed)
    3. Re-exec the process (works with or without systemd)
    """
    import subprocess, os, sys

    agent_dir = os.environ.get("AGENT_DIR") or os.path.abspath(
        os.path.dirname(__file__)
    )
    venv_pip = os.path.join(agent_dir, "venv", "bin", "pip")
    agent_repo = "https://github.com/toup-com/toup-agent.git"

    steps = []

    # Ensure git safe.directory is set (prevents "dubious ownership" errors)
    subprocess.run(
        ["git", "config", "--global", "--add", "safe.directory", agent_dir],
        capture_output=True, timeout=5,
    )

    # 1. Git pull (or bootstrap .git if missing)
    git_dir = os.path.join(agent_dir, ".git")
    try:
        if not os.path.isdir(git_dir):
            # No .git — bootstrap by initializing and pulling
            subprocess.run(["git", "init"], cwd=agent_dir, capture_output=True, timeout=10)
            subprocess.run(["git", "remote", "add", "origin", agent_repo],
                           cwd=agent_dir, capture_output=True, timeout=10)
            result = subprocess.run(
                ["git", "fetch", "--depth", "1", "origin", "main"],
                cwd=agent_dir, capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                subprocess.run(["git", "checkout", "-f", "origin/main"],
                               cwd=agent_dir, capture_output=True, timeout=10)
                subprocess.run(["git", "branch", "-f", "main", "origin/main"],
                               cwd=agent_dir, capture_output=True, timeout=10)
                subprocess.run(["git", "checkout", "main"],
                               cwd=agent_dir, capture_output=True, timeout=10)
            pull_output = "bootstrapped .git and fetched latest"
            steps.append({"step": "git_bootstrap", "ok": True, "output": pull_output})
        else:
            # Fetch + checkout is more reliable than pull --ff-only
            # (handles cases where local branch doesn't track remote)
            result = subprocess.run(
                ["git", "fetch", "--depth", "1", "origin", "main"],
                cwd=agent_dir, capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                pull_output = result.stderr.strip() or result.stdout.strip()
                steps.append({"step": "git_fetch", "ok": False, "output": pull_output})
                return {"success": False, "steps": steps, "error": "git fetch failed"}
            # Reset to exact remote state (handles detached HEAD, local changes, etc.)
            result = subprocess.run(
                ["git", "reset", "--hard", "origin/main"],
                cwd=agent_dir, capture_output=True, text=True, timeout=10,
            )
            pull_output = result.stdout.strip() or result.stderr.strip()
            steps.append({"step": "git_update", "ok": result.returncode == 0, "output": pull_output})
            if result.returncode != 0:
                return {"success": False, "steps": steps, "error": "git reset failed"}
    except Exception as e:
        return {"success": False, "steps": [{"step": "git_pull", "ok": False, "output": str(e)}]}

    # 2. Install deps (only if requirements changed)
    needs_install = "requirements.txt" in pull_output or "Already up to date" not in pull_output
    if needs_install and os.path.exists(venv_pip):
        try:
            result = subprocess.run(
                [venv_pip, "install", "-q", "-r", os.path.join(agent_dir, "requirements.txt")],
                cwd=agent_dir, capture_output=True, text=True, timeout=120,
            )
            steps.append({"step": "pip_install", "ok": result.returncode == 0, "output": result.stdout.strip()[:200]})
        except Exception as e:
            steps.append({"step": "pip_install", "ok": False, "output": str(e)})
    else:
        steps.append({"step": "pip_install", "ok": True, "output": "skipped (no changes)"})

    # 3. Schedule restart — use systemctl if available, else re-exec
    import asyncio

    async def _delayed_restart():
        await asyncio.sleep(1.0)  # Give time for HTTP response to be sent
        print("\n🔄 Restarting after update...")

        # Prefer systemctl restart if running as a systemd service
        service_name = os.environ.get("SYSTEMD_SERVICE") or "toup-agent"
        systemctl = subprocess.run(
            ["systemctl", "is-active", service_name],
            capture_output=True, text=True, timeout=5,
        )
        if systemctl.returncode == 0:
            # We're running under systemd — let it handle the restart
            subprocess.Popen(["systemctl", "restart", service_name])
        else:
            # Not systemd — re-exec ourselves
            os.execv(sys.executable, [sys.executable] + sys.argv)

    asyncio.get_event_loop().create_task(_delayed_restart())

    steps.append({"step": "restart", "ok": True, "output": "scheduled"})
    return {"success": True, "steps": steps}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("agent_main:app", host="0.0.0.0", port=8001, reload=settings.debug)
