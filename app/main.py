"""
Toup Agent Platform - Main Application Entry Point
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.db import init_db
from app.api import (
    auth_router,
    memories_router,
    ingest_router,
    agent_router,
    stats_router,
)
from app.api.admin import system_router as admin_router, users_router as admin_users_router, invite_router
from app.api.documents import router as documents_router
# NEW: Agent Platform routers
from app.api.identity import router as identity_router
from app.api.soul import router as soul_router
from app.api.api_v1 import router as api_v1_router
from app.api.graph import router as graph_router
from app.api.feedback import router as feedback_router
from app.api.webhooks import router as webhooks_router, set_webhook_refs
# Chat routers
from app.api.chat import router as chat_router
from app.api.sessions import router as sessions_router
from app.api.ws_chat import router as ws_chat_router, set_ws_refs
# Layer 6 routers
from app.api.canvas import router as canvas_router
from app.api.doctor import router as doctor_router
from app.api.voice import router as voice_router, set_voice_refs
# Dashboard proxy
from app.api.dashboard_proxy import router as dashboard_proxy_router
# Combined checkout
from app.api.checkout import router as checkout_router
from app.api.llm_setup import router as llm_setup_router
# Generated-file attachments (Phase 2 of doc-delivery)
from app.api.files import router as files_router

# Global start time for uptime tracking
_app_start_time = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler - runs on startup and shutdown"""
    global _app_start_time
    import time as _time
    _app_start_time = _time.time()

    # Startup
    print("🧠 Toup Agent starting up...")
    await init_db()
    print("✅ Database initialized")
    
    # Pre-load embedding service (optional, for faster first request)
    try:
        from app.services import get_embedding_service
        svc = get_embedding_service()
        if svc.is_openai:
            _ = svc.openai_client  # Initialize OpenAI client
            print(f"✅ OpenAI embedding service ready (model: {settings.embedding_model})")
        else:
            _ = svc.local_model  # Load local model
            print("✅ Local embedding model loaded")
    except Exception as e:
        print(f"⚠️ Could not pre-load embedding service: {e}")
    
    # Start scheduled tasks for memory maintenance (decay/consolidation)
    if settings.enable_scheduler:
        try:
            from app.scripts.scheduled_tasks import start_scheduler
            start_scheduler()
            print("✅ Memory maintenance scheduler started")
        except Exception as e:
            print(f"⚠️ Could not start scheduler: {e}")

    # Rollout reconciler — runs forever in the background, ticks every 30 s.
    # This replaces the APScheduler-dependent recovery path. It re-derives
    # rollout intent from the DB on every tick, so any process death is
    # recovered within 30 s of the next process start. Also auto-orphans
    # rollouts stuck past 30 min, preventing them from blocking future CI
    # rollouts with HTTP 409 indefinitely.
    #
    # The startup `resume_orphaned_rollouts()` call is kept as a warm-start
    # so a fresh process picks up orphans immediately rather than waiting
    # the full 30 s tick.
    import asyncio as _asyncio
    rollout_reconciler_task = None
    try:
        from app.services.rollout_service import (
            resume_orphaned_rollouts,
            rollout_reconciler_loop,
        )
        await resume_orphaned_rollouts()
        rollout_reconciler_task = _asyncio.create_task(rollout_reconciler_loop())
    except Exception as e:
        # Must not block app boot. If startup recovery fails the loop will
        # retry on its next tick anyway.
        print(f"⚠️ Rollout reconciler bootstrap failed: {e}")
    
    # Start Telegram bot if configured
    telegram_bot = None
    cron_service = None
    subagent_manager = None
    skill_loader = None
    if settings.telegram_bot_token:
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
                skill_loader = SkillLoader()  # empty fallback
            
            openai_agent_svc = OpenAIAgentService()
            subagent_manager = SubAgentManager()
            tool_executor = ToolExecutor(subagent_manager=subagent_manager)
            tool_executor.skill_loader = skill_loader
            agent_runner = AgentRunner(
                llm_service=openai_agent_svc,
                tool_executor=tool_executor,
                skill_loader=skill_loader,
            )
            telegram_bot = ToupTelegramBot(
                token=settings.telegram_bot_token,
                agent_runner=agent_runner,
            )
            # Give tool executor access to the bot for send_file/send_photo
            tool_executor.telegram_bot = telegram_bot

            # Wire sub-agent manager
            subagent_manager.set_agent_runner(agent_runner)
            subagent_manager.set_bot(telegram_bot)
            telegram_bot.subagent_manager = subagent_manager

            # Set up cron service
            cron_service = CronService()
            cron_service.set_bot(telegram_bot)
            cron_service.set_agent_runner(agent_runner)
            tool_executor.cron_service = cron_service
            telegram_bot.cron_service = cron_service

            # Wire skill loader into bot for /skills command
            telegram_bot.skill_loader = skill_loader

            await telegram_bot.start()
            print(f"🤖 Telegram bot started (polling mode)")

            # Set admin dashboard refs
            from app.api.admin import set_bot_refs
            set_bot_refs(telegram_bot, cron_service, telegram_bot._start_time)

            # Set WebSocket and API v1 refs
            from app.api.api_v1 import set_api_v1_refs
            set_ws_refs(agent_runner, skill_loader)
            set_api_v1_refs(agent_runner, skill_loader)

            # Start cron scheduler (after bot is running)
            try:
                await cron_service.start()
                print(f"⏰ Cron service started")
            except Exception as e:
                print(f"⚠️ Could not start cron service: {e}")
                cron_service = None

            # Start heartbeat service if enabled
            if settings.heartbeat_enabled:
                try:
                    from app.agent.heartbeat_service import HeartbeatService
                    from apscheduler.triggers.interval import IntervalTrigger

                    heartbeat_svc = HeartbeatService()
                    heartbeat_svc.set_agent_runner(agent_runner)
                    heartbeat_svc.set_bot(telegram_bot)

                    # Register with the cron service's scheduler
                    if cron_service and cron_service.scheduler:
                        cron_service.scheduler.add_job(
                            heartbeat_svc.tick,
                            trigger=IntervalTrigger(hours=settings.heartbeat_interval_hours),
                            id="heartbeat",
                            name="Proactive Agent Heartbeat",
                            replace_existing=True,
                        )
                        print(f"💓 Heartbeat service started (every {settings.heartbeat_interval_hours}h)")
                except Exception as e:
                    print(f"⚠️ Could not start heartbeat service: {e}")

        except Exception as e:
            print(f"⚠️ Could not start Telegram bot: {e}")
            import traceback
            traceback.print_exc()
            telegram_bot = None
            cron_service = None
    
    # ── Hook Bus ─────────────────────────────────────────────
    from app.agent.hooks import get_hook_bus, HookEvent
    _hook_bus = get_hook_bus()
    await _hook_bus.emit(HookEvent.STARTUP, {"app": "toup"})
    print("🔌 Hook bus started")

    # ── Set webhook refs ─────────────────────────────────────
    if telegram_bot:
        try:
            set_webhook_refs(agent_runner, telegram_bot)
            print("🪝 Webhook triggers wired")
        except Exception as e:
            print(f"⚠️ Could not wire webhook refs: {e}")

    # ── Discord Channel ──────────────────────────────────────
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
            print(f"⚠️ Could not start Discord channel: {e}")
            discord_channel = None

    # ── Slack Channel ────────────────────────────────────────
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
            print(f"⚠️ Could not start Slack channel: {e}")
            slack_channel = None

    # ── WhatsApp Channel ─────────────────────────────────────
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
            print(f"⚠️ Could not start WhatsApp channel: {e}")
            whatsapp_channel = None

    yield
    
    # ── Shutdown hooks ───────────────────────────────────────
    await _hook_bus.emit(HookEvent.SHUTDOWN, {"app": "toup"})

    # Shutdown — reverse order for clean teardown
    print("🧠 Toup Agent shutting down gracefully...")

    # 0. Cancel rollout reconciler so its 30 s sleep doesn't drag shutdown.
    if rollout_reconciler_task and not rollout_reconciler_task.done():
        rollout_reconciler_task.cancel()
        try:
            await rollout_reconciler_task
        except (Exception, BaseException):
            pass

    # 1. Stop cron scheduler first (no new jobs)
    if cron_service:
        try:
            await cron_service.stop()
            print("⏰ Cron service stopped")
        except Exception as e:
            print(f"⚠️ Cron shutdown error: {e}")
    
    # 2. Cancel any running sub-agent tasks
    if subagent_manager:
        try:
            active = [r for r in subagent_manager._runs.values() if r.status == "running"]
            for run in active:
                if run._task_handle and not run._task_handle.done():
                    run._task_handle.cancel()
            if active:
                print(f"🧵 Cancelled {len(active)} active sub-agent tasks")
        except Exception as e:
            print(f"⚠️ Sub-agent cleanup error: {e}")
    
    # 3. Unload skills
    if skill_loader:
        try:
            await skill_loader.unload_all()
            print("🧩 Skills unloaded")
        except Exception as e:
            print(f"⚠️ Skills unload error: {e}")

    # 4a. Stop Discord channel
    if discord_channel:
        try:
            await discord_channel.stop()
            print("💬 Discord channel stopped")
        except Exception as e:
            print(f"⚠️ Discord shutdown error: {e}")

    # 4b. Stop Slack channel
    if slack_channel:
        try:
            await slack_channel.stop()
            print("💼 Slack channel stopped")
        except Exception as e:
            print(f"⚠️ Slack shutdown error: {e}")

    # 4c. Stop WhatsApp channel
    if whatsapp_channel:
        try:
            await whatsapp_channel.stop()
            print("�� WhatsApp channel stopped")
        except Exception as e:
            print(f"⚠️ WhatsApp shutdown error: {e}")

    # 4. Stop Telegram bot
    if telegram_bot:
        try:
            await telegram_bot.stop()
            print("🤖 Telegram bot stopped")
        except Exception as e:
            print(f"⚠️ Bot shutdown error: {e}")
    
    # 5. Stop scheduler
    if settings.enable_scheduler:
        try:
            from app.scripts.scheduled_tasks import stop_scheduler
            stop_scheduler()
            print("📅 Scheduler stopped")
        except Exception:
            pass
    
    print("🧠 Toup Agent shutdown complete.")


app = FastAPI(
    title=settings.app_name,
    description="AI Agent Platform with persistent memory, identity system, chat orchestration, and Telegram bot",
    version="5.0.0",  # v5: Toup Platform — Skills, WebSocket, Public API
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router, prefix=settings.api_prefix)
app.include_router(memories_router, prefix=settings.api_prefix)
app.include_router(ingest_router, prefix=settings.api_prefix)
app.include_router(documents_router, prefix=settings.api_prefix)
app.include_router(agent_router, prefix=settings.api_prefix)
app.include_router(stats_router, prefix=settings.api_prefix)
app.include_router(admin_router, prefix=settings.api_prefix)
# Beta access & invite management
app.include_router(admin_users_router, prefix=settings.api_prefix)
app.include_router(invite_router, prefix=settings.api_prefix)
# Brain core
app.include_router(identity_router, prefix=settings.api_prefix)
app.include_router(soul_router, prefix=settings.api_prefix)
app.include_router(api_v1_router, prefix=settings.api_prefix)
app.include_router(graph_router, prefix=settings.api_prefix)  # Entity graph API
app.include_router(feedback_router, prefix=settings.api_prefix)  # Phase 5: Retrieval feedback
app.include_router(webhooks_router, prefix=settings.api_prefix)  # Webhook triggers
# Chat
app.include_router(sessions_router, prefix=settings.api_prefix)
app.include_router(chat_router, prefix=settings.api_prefix)
app.include_router(ws_chat_router, prefix=settings.api_prefix)
# Layer 6: Canvas, Doctor, Voice
app.include_router(canvas_router, prefix=settings.api_prefix)
app.include_router(doctor_router, prefix=settings.api_prefix)
app.include_router(voice_router, prefix=settings.api_prefix)
# Dashboard (proxied to VPS agent)
app.include_router(dashboard_proxy_router, prefix=settings.api_prefix)
# Combined checkout
app.include_router(checkout_router, prefix=settings.api_prefix)
# LLM setup & usage
app.include_router(llm_setup_router, prefix=settings.api_prefix)
# Generated-file attachments
app.include_router(files_router, prefix=settings.api_prefix)



@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "name": settings.app_name,
        "status": "healthy",
        "version": "5.0.0",
        "features": ["identity", "sessions", "chat", "memories", "documents", "telegram_bot", "discord", "slack", "whatsapp", "agent_tools", "skills", "websocket", "api_v1", "webhooks", "hooks"]
    }


@app.get("/health")
async def health():
    """Detailed health check with per-channel probes and uptime."""
    import time as _time

    # Database probe
    db_status = "connected"
    try:
        from app.db.database import async_session_maker
        async with async_session_maker() as db:
            from sqlalchemy import text
            await db.execute(text("SELECT 1"))
    except Exception as e:
        db_status = f"error: {e}"

    # Lane stats
    try:
        from app.agent.lanes import get_lane_manager
        lane_stats = get_lane_manager().get_stats()
    except Exception:
        lane_stats = {}

    uptime = _time.time() - _app_start_time if _app_start_time else 0

    return {
        "status": "healthy" if db_status == "connected" else "degraded",
        "version": "6.0.0",
        "uptime_seconds": round(uptime, 1),
        "database": db_status,
        "embedding_model": settings.embedding_model,
        "chat_model": settings.default_model,
        "agent_model": settings.agent_model,
        "thinking_budget": settings.thinking_budget_default,
        "tts_mode": settings.tts_auto_mode,
        "platform": "Toup Agent Platform v6",
        "channels": {
            "telegram": "enabled" if settings.telegram_bot_token else "disabled",
            "discord": "enabled" if settings.discord_bot_token else "disabled",
            "slack": "enabled" if settings.slack_bot_token else "disabled",
            "whatsapp": "enabled" if settings.whatsapp_phone_number_id else "disabled",
        },
        "lanes": lane_stats,
        "features": {
            "reranker": settings.enable_reranker,
            "sandbox": settings.sandbox_enabled,
            "moderation": settings.moderation_enabled,
            "config_hot_reload": settings.config_reload_enabled,
            "forum_support": settings.telegram_forum_support,
            "reactions": settings.telegram_reactions_enabled,
            "polls": settings.telegram_polls_enabled,
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
    )

# ------------------------------------------------------------------
# Static files + WebChat
# ------------------------------------------------------------------
import os as _os
_static_dir = _os.path.join(_os.path.dirname(__file__), "static")
if _os.path.isdir(_static_dir):
    from starlette.responses import FileResponse

    @app.get("/webchat", include_in_schema=False)
    async def webchat_page():
        """Serve the standalone WebChat HTML page."""
        return FileResponse(_os.path.join(_static_dir, "webchat.html"))
