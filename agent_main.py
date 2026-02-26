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

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import settings
from app.db import init_db

# ── Agent routers (invoke AgentRunner) ───────────────────────────────
from app.api import agent_router
from app.api.chat import router as chat_router
from app.api.ws_chat import router as ws_chat_router, set_ws_refs
from app.api.api_v1 import router as api_v1_router
from app.api.webhooks import router as webhooks_router, set_webhook_refs
from app.api.voice import router as voice_router, set_voice_refs

_app_start_time = None

# ── Paths that skip API key auth (health checks, root) ─────────────
_PUBLIC_PATHS = frozenset({"/", "/agent/health", "/docs", "/openapi.json", "/redoc"})


class AgentAPIKeyMiddleware(BaseHTTPMiddleware):
    """Validates the X-Agent-Key header on non-public endpoints.

    When AGENT_API_KEY is set (i.e. running on a user's VPS), every request
    must include a matching header. In monolith/dev mode (no key set), all
    requests pass through.
    """

    async def dispatch(self, request: Request, call_next):
        # Skip auth if no key is configured (local dev / monolith mode)
        if not settings.agent_api_key:
            return await call_next(request)

        # Allow public endpoints without auth
        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)

        # Check the API key header
        provided_key = request.headers.get("x-agent-key", "")
        if provided_key != settings.agent_api_key:
            return Response(
                content='{"detail":"Invalid or missing agent API key"}',
                status_code=401,
                media_type="application/json",
            )

        return await call_next(request)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _app_start_time
    import time as _time
    _app_start_time = _time.time()

    # ── Startup ───────────────────────────────────────────────
    print("🤖 Toup Agent starting up...")
    await init_db()
    print("✅ Database initialized")

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

    try:
        from app.agent.telegram_bot import HexBrainTelegramBot
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

        # ── Start Telegram bot (if configured) ────────────────
        if settings.telegram_bot_token:
            telegram_bot = HexBrainTelegramBot(
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

    print("🤖 Toup Agent ready.")
    print(f"   Server:  http://0.0.0.0:8001")
    print(f"   Health:  http://localhost:8001/agent/health")
    print(f"   Web UI:  https://toup.ai")
    print(f"   Press Ctrl+C to stop.\n")
    yield

    # ── Shutdown (reverse order) ──────────────────────────────
    await _hook_bus.emit(HookEvent.SHUTDOWN, {"app": "toup-agent"})
    print("🤖 Toup Agent shutting down...")

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
app.include_router(chat_router, prefix=settings.api_prefix)
app.include_router(ws_chat_router, prefix=settings.api_prefix)
app.include_router(api_v1_router, prefix=settings.api_prefix)
app.include_router(webhooks_router, prefix=settings.api_prefix)
app.include_router(voice_router, prefix=settings.api_prefix)


@app.get("/")
async def root():
    return {
        "name": "Toup Agent",
        "status": "healthy",
        "version": "6.0.0",
        "mode": "agent",
    }


@app.get("/agent/health")
async def agent_health():
    import time as _time
    uptime = _time.time() - _app_start_time if _app_start_time else 0
    return {
        "status": "healthy",
        "version": "6.0.0",
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


@app.post("/agent/update")
async def agent_self_update():
    """Pull latest code, install deps, and restart the agent process.

    The restart works by:
    1. git pull --ff-only
    2. pip install -r requirements.txt (if changed)
    3. Exit with code 0 — the service manager (systemd/launchd) restarts us
    """
    import subprocess, os, sys

    agent_dir = os.environ.get("AGENT_DIR") or os.path.abspath(
        os.path.dirname(__file__)
    )
    venv_pip = os.path.join(agent_dir, "venv", "bin", "pip")

    steps = []

    # 1. Git pull
    try:
        result = subprocess.run(
            ["git", "pull", "--ff-only"],
            cwd=agent_dir, capture_output=True, text=True, timeout=30,
        )
        pull_output = result.stdout.strip() or result.stderr.strip()
        steps.append({"step": "git_pull", "ok": result.returncode == 0, "output": pull_output})
        if result.returncode != 0:
            return {"success": False, "steps": steps, "error": "git pull failed"}
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

    # 3. Schedule restart — exit after response is sent
    # The service manager (launchd/systemd) will restart us automatically
    import asyncio

    async def _delayed_exit():
        await asyncio.sleep(1.0)  # Give time for HTTP response to be sent
        print("\n🔄 Restarting after update...")
        os._exit(0)  # Service manager restarts us

    asyncio.get_event_loop().create_task(_delayed_exit())

    steps.append({"step": "restart", "ok": True, "output": "scheduled"})
    return {"success": True, "steps": steps}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("agent_main:app", host="0.0.0.0", port=8001, reload=settings.debug)
