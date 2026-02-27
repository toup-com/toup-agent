"""
Toup Platform Service — Data & API layer.

This is the Platform-only entry point. It exposes all data CRUD routes
(auth, memories, sessions, identity, graph, ingest, admin, stats, VPS)
but does NOT start the Agent, Telegram bot, channels, cron, or skills.

Usage:
    uvicorn platform_main:app --host 0.0.0.0 --port 8000

Deploy to Vercel via /api/index.py which imports this app.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.db import init_db

# ── Platform routers (data CRUD, no agent invocation) ────────────────
from app.api import auth_router, memories_router, ingest_router, stats_router
from app.api.admin import (
    system_router as admin_router,
    users_router as admin_users_router,
    invite_router,
)
from app.api.documents import router as documents_router
from app.api.identity import router as identity_router
from app.api.graph import router as graph_router
from app.api.feedback import router as feedback_router
from app.api.sessions import router as sessions_router
from app.api.canvas import router as canvas_router
from app.api.doctor import router as doctor_router
from app.api.voice import router as voice_router
try:
    from app.api.vps import router as vps_router
except ImportError:
    vps_router = None

_app_start_time = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _app_start_time
    import time as _time
    _app_start_time = _time.time()

    # ── Startup ───────────────────────────────────────────────
    print("🌐 Toup Platform starting up...")
    await init_db()
    print("✅ Database initialized")

    # Pre-load embedding service
    try:
        from app.services import get_embedding_service
        svc = get_embedding_service()
        if svc.is_openai:
            _ = svc.openai_client
            print(f"✅ OpenAI embedding service ready ({settings.embedding_model})")
        else:
            _ = svc.local_model
            print("✅ Local embedding model loaded")
    except Exception as e:
        print(f"⚠️ Could not pre-load embedding service: {e}")

    # Memory maintenance scheduler (decay/consolidation are platform concerns)
    if settings.enable_scheduler:
        try:
            from app.scripts.scheduled_tasks import start_scheduler
            start_scheduler()
            print("✅ Memory maintenance scheduler started")
        except Exception as e:
            print(f"⚠️ Could not start scheduler: {e}")

    # Voice tool calls are routed through the agent tunnel to the user's
    # terminal agent. No local ToolExecutor — the platform is data-only.
    print("🎤 Voice ready (tools execute via terminal agent tunnel)")

    print("🌐 Toup Platform ready.")
    yield

    # ── Shutdown ──────────────────────────────────────────────
    print("🌐 Toup Platform shutting down...")
    if settings.enable_scheduler:
        try:
            from app.scripts.scheduled_tasks import stop_scheduler
            stop_scheduler()
        except Exception:
            pass
    print("🌐 Toup Platform shutdown complete.")


app = FastAPI(
    title="Toup Platform",
    description="Central platform API: auth, memories, sessions, identity, graph, VPS provisioning",
    version="6.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register platform routers ────────────────────────────────────────
app.include_router(auth_router, prefix=settings.api_prefix)
app.include_router(memories_router, prefix=settings.api_prefix)
app.include_router(ingest_router, prefix=settings.api_prefix)
app.include_router(documents_router, prefix=settings.api_prefix)
app.include_router(stats_router, prefix=settings.api_prefix)
# Admin
app.include_router(admin_router, prefix=settings.api_prefix)
app.include_router(admin_users_router, prefix=settings.api_prefix)
app.include_router(invite_router, prefix=settings.api_prefix)
# Brain core
app.include_router(identity_router, prefix=settings.api_prefix)
app.include_router(graph_router, prefix=settings.api_prefix)
app.include_router(feedback_router, prefix=settings.api_prefix)
# Sessions (data only, no agent invocation)
app.include_router(sessions_router, prefix=settings.api_prefix)
# VPS provisioning
if vps_router:
    app.include_router(vps_router, prefix=settings.api_prefix)
# Canvas & Doctor (UI push, health checks)
app.include_router(canvas_router, prefix=settings.api_prefix)
app.include_router(doctor_router, prefix=settings.api_prefix)
# Voice (transcription & TTS)
app.include_router(voice_router, prefix=settings.api_prefix)
# Realtime voice (OpenAI Realtime API proxy)
try:
    from app.api.ws_realtime import router as ws_realtime_router
    app.include_router(ws_realtime_router, prefix=settings.api_prefix)
except ImportError as e:
    print(f"⚠️ Realtime voice not mounted: {e}")
# Agent tunnel (terminal agent connects here for tool dispatch)
try:
    from app.api.ws_agent_tunnel import router as ws_tunnel_router
    app.include_router(ws_tunnel_router, prefix=settings.api_prefix)
except ImportError as e:
    print(f"⚠️ Agent tunnel not mounted: {e}")
# Agent setup & deployment
try:
    from app.api.agent_setup import router as agent_setup_router
    from app.api.ws_deploy import router as ws_deploy_router
    from app.api.llm_setup import router as llm_setup_router
    app.include_router(agent_setup_router, prefix=settings.api_prefix)
    app.include_router(ws_deploy_router, prefix=settings.api_prefix)
    app.include_router(llm_setup_router, prefix=settings.api_prefix)
except ImportError as e:
    print(f"⚠️ Agent setup routes not mounted: {e}")


# ── MCP Server ──────────────────────────────────────────────────────
try:
    from app.mcp_server import mcp
    mcp.mount(app, path="/api/mcp")
except ImportError as e:
    print(f"⚠️ MCP server not mounted (fastmcp not installed): {e}")
except Exception as e:
    print(f"⚠️ MCP server mount error: {e}")


@app.get("/api")
async def api_root():
    return {
        "name": "Toup Platform",
        "status": "healthy",
        "version": "6.0.0",
        "mode": "platform",
    }


@app.get("/health")
@app.get("/api/health")
async def health():
    import time as _time

    db_status = "connected"
    try:
        from app.db.database import async_session_maker
        async with async_session_maker() as db:
            from sqlalchemy import text
            await db.execute(text("SELECT 1"))
    except Exception as e:
        db_status = f"error: {e}"

    uptime = _time.time() - _app_start_time if _app_start_time else 0

    return {
        "status": "healthy" if db_status == "connected" else "degraded",
        "version": "6.0.0",
        "mode": "platform",
        "uptime_seconds": round(uptime, 1),
        "database": db_status,
        "embedding_model": settings.embedding_model,
    }


# ── Serve React frontend (static files) ─────────────────────────────
# IMPORTANT: This must be LAST — the catch-all route intercepts all paths
import os
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

_static_dir = Path(__file__).resolve().parent.parent.parent / "static"
if _static_dir.is_dir():
    app.mount("/assets", StaticFiles(directory=str(_static_dir / "assets")), name="assets")

    @app.get("/{path:path}")
    async def serve_spa(path: str):
        """Serve the React SPA for all non-API routes."""
        file_path = _static_dir / path
        if file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(_static_dir / "index.html"))
else:
    @app.get("/")
    async def root():
        return {"name": "Toup Platform", "status": "healthy", "version": "6.0.0"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("platform_main:app", host="0.0.0.0", port=port, reload=settings.debug)
