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

logger = logging.getLogger(__name__)
from datetime import datetime, timedelta
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
from app.api.day_chats import router as day_chats_router
from app.api.chat import router as chat_router
from app.api.messages_recover import router as messages_recover_router
from app.api.ws_chat import router as ws_chat_router, set_ws_refs, broadcast_to_user
from app.api.api_v1 import router as api_v1_router
from app.api.models import router as models_router
from app.api.webhooks import router as webhooks_router, set_webhook_refs
from app.api.voice import router as voice_router
from app.api.ws_realtime import router as ws_realtime_router, set_realtime_refs
from app.api.ws_browser import router as ws_browser_router, set_ws_browser_refs
from app.api.dashboard import router as dashboard_router
from app.api.library import router as library_router
from app.api.soul import router as soul_router
from app.api.identity import router as identity_router
from app.api.llm_setup import router as llm_setup_router
from app.api.apps import router as apps_router, set_app_manager, set_app_gateway, set_app_builder_skill, set_agent_runner
# Generated-file attachments (doc-delivery feature). Serves the actual files
# from local disk on the agent; the platform's files_router proxies here.
from app.api.files import router as files_router
# WhatsApp QR-pairing endpoints (agent-side; platform proxies via
# /api/agent-setup/whatsapp/qr-* using X-Agent-Key over Caddy TLS).
from app.api.whatsapp_qr import router as whatsapp_qr_router
# Phase A: pool admin (bind / drain / status). Auth by POOL_ADMIN_TOKEN
# (NOT X-Agent-Key) — bridge holds the token, agent containers receive
# it as an env var at docker run.
from app.api.admin_pool import router as admin_pool_router
from app.api.routines import router as routines_router, set_runner_ref as set_routines_runner_ref
# Triggers inbound (Gate T1) — platform-side webhook dispatches here.
# No runner yet (Gate T2). Same X-Agent-Key contract as other authed
# agent endpoints; defence-in-depth user_id check inside the handler.
from app.api.triggers_inbound import router as triggers_inbound_router
# Triggers user-facing CRUD (Gate T2) — list/create/update/delete/events/test.
from app.api.triggers import router as triggers_router, set_runner_ref as set_triggers_crud_runner_ref
from app.services import runtime_identity, drain_state

_app_start_time = None
_skill_loader = None

# ── Boot progress tracking (exposed via /agent/health) ────────────
_boot_progress = {"percent": 0, "phase": "starting", "ready": False}

# ── Paths that skip API key auth (health checks, root) ─────────────
# Pool admin endpoints (`/api/admin/bind` etc.) are public to the
# X-Agent-Key middleware but enforce their own POOL_ADMIN_TOKEN check
# inside the handler. Same posture as `/agent/health`: the middleware
# delegates auth to the route, not the other way around.
_PUBLIC_PATHS = frozenset({
    "/", "/agent/health", "/agent/system", "/docs", "/openapi.json", "/redoc",
    "/api/admin/bind", "/api/admin/drain", "/api/admin/status",
})

# Routes that remain reachable in pool-lobby mode (TOUP_POOL_GENERIC=1
# AND not yet bound). Everything else 503s with `X-Lobby-Mode: 1`.
# Bridge polls /agent/health to know when generic boot finishes; it
# calls /api/admin/bind to claim. Once `runtime_identity.is_bound()`
# returns True, the lobby gate is off and the rest of the API serves
# normally.
_LOBBY_ALLOWED = frozenset({
    "/", "/agent/health", "/agent/system",
    "/api/admin/bind", "/api/admin/drain", "/api/admin/status",
})


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


def is_passive_boot() -> bool:
    """True while this container is in blue-green passive boot — started with
    TOUP_BG_PASSIVE=1 and not yet promoted (marker file absent).

    Used by wake_lazy_channels to DEFER starting WhatsApp Baileys at
    /admin/bind time: during a pool assigned-upgrade the new (green) container
    is bound while the OLD container is still connected with the same WhatsApp
    device credentials, and two concurrent same-device sessions can force the
    user to re-scan the QR (gap A3, 2026-07-04). The lifespan's delayed-promote
    starts WhatsApp ~90s later, after the old container is gone. The promote
    calls restart_whatsapp_channel() directly (not via wake_lazy_channels), so
    it is unaffected by this gate."""
    try:
        from pathlib import Path as _P
        return (os.environ.get("TOUP_BG_PASSIVE") == "1"
                and not _P("/app/workspace/.toup_bg_promoted").exists())
    except Exception:
        return False


async def restart_whatsapp_channel():
    """Hot-restart the WhatsApp channel after config sync.

    Called by tunnel_client when whatsapp_mode / allowlist / cloud-API
    creds change. Stops the existing channel (if any), then re-runs the
    same selection logic main() uses at boot to pick the right adapter.

    Without this, a tunnel-pushed config change writes the new mode to
    .env but the running process still has whichever channel main()
    spawned at boot — typically NONE for fresh users where mode was
    NULL. They click Connect via QR, the platform proxies /qr/start to
    the agent, and `_require_active_channel()` returns 503 because
    `BaileysWhatsAppChannel` was never instantiated. This function
    closes the gap.
    """
    global _agent_runner
    from app.config import settings as _s
    from app.agent.channels.registry import ChannelRegistry
    from app.agent.channels.shared import make_channel_handler

    if not _agent_runner:
        logging.warning("[RESTART] Cannot start WhatsApp channel — agent_runner not initialized")
        return

    # Tear down whichever WhatsApp adapter is currently registered. Both
    # cloud-API and QR-link extend BaseChannel(WHATSAPP), so the
    # registry slot is the same.
    from app.agent.channels.base import ChannelType as _CT
    existing = ChannelRegistry.get(_CT.WHATSAPP)
    if existing is not None:
        try:
            await existing.stop()
            print("🛑 WhatsApp channel stopped (config changed)")
        except Exception as e:
            logging.warning(f"[RESTART] WhatsApp stop error: {e}")
        # Unregister so the next register() doesn't log a "replacing" warning.
        try:
            ChannelRegistry._channels.pop(_CT.WHATSAPP, None)
        except Exception:
            pass

    _wa_mode = (_s.whatsapp_mode or "").strip().lower()
    if not _wa_mode:
        if _s.whatsapp_phone_number_id and _s.whatsapp_access_token:
            _wa_mode = "cloud_api"
        else:
            _wa_mode = "qr_link"

    try:
        if _wa_mode == "qr_link":
            from app.agent.channels.whatsapp_baileys import BaileysWhatsAppChannel
            _allowlist = [
                s.strip() for s in (_s.whatsapp_baileys_allowlist or "").split(",")
                if s.strip()
            ]
            ch = BaileysWhatsAppChannel(allowed_numbers=_allowlist)
            ch.set_message_callback(
                make_channel_handler(channel=ch, agent_runner=_agent_runner, user_id=_s.user_id)
            )
            await ch.start()
            ChannelRegistry.register(ch)
            print("📱 WhatsApp channel restarted (QR-link / Baileys sidecar)")
        elif _wa_mode == "cloud_api" and _s.whatsapp_phone_number_id and _s.whatsapp_access_token:
            from app.agent.channels.whatsapp_channel import WhatsAppChannel
            ch = WhatsAppChannel(
                phone_number_id=_s.whatsapp_phone_number_id,
                access_token=_s.whatsapp_access_token,
                verify_token=_s.whatsapp_verify_token,
                app_secret=_s.whatsapp_app_secret,
                allowed_numbers=_s.whatsapp_allowed_numbers or None,
            )
            ch.set_message_callback(
                make_channel_handler(channel=ch, agent_runner=_agent_runner, user_id=_s.user_id)
            )
            await ch.start()
            ChannelRegistry.register(ch)
            print("📱 WhatsApp channel restarted (Cloud API)")
    except Exception as e:
        logging.exception(f"[RESTART] Failed to start WhatsApp channel: {e}")


class LobbyAndDrainMiddleware:
    """Phase A/B gate: block traffic when the agent is in pool-lobby
    mode (not yet bound) or actively draining (Phase B blue-green
    cutover).

    Lobby mode (`TOUP_POOL_GENERIC=1` + `runtime_identity.is_bound()=False`):
        Everything except _LOBBY_ALLOWED returns 503 with
        `X-Lobby-Mode: 1`. WebSocket upgrades get a 503 close
        (Starlette translates http.response 503 into ws-close 1011 for
        ASGI clients before accept). Existing tenants — `TOUP_POOL_GENERIC`
        unset — are unaffected; they pass straight through.

    Drain mode (`drain_state.is_draining()=True`):
        New WebSocket upgrades close with code 1012 (Service Restart);
        in-flight handlers continue. HTTP requests pass through —
        a drain doesn't break short-lived requests, only long-lived
        ones, and we want any final cleanup HTTP to succeed."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")

        # Lobby gate
        if runtime_identity.is_pool_generic() and not runtime_identity.is_bound():
            if path not in _LOBBY_ALLOWED:
                if scope["type"] == "websocket":
                    # Reject before accept — code 1011 (Internal Error) /
                    # we use 1013 (Try Again Later). Starlette accepts
                    # only `websocket.close` after `websocket.accept`,
                    # but for unaccepted WS we send `websocket.close`
                    # straight off the receive queue.
                    await send({"type": "websocket.close", "code": 1013, "reason": "pool-lobby"})
                    return
                response = Response(
                    content='{"detail":"Agent in pool-lobby mode; awaiting bind"}',
                    status_code=503,
                    media_type="application/json",
                    headers={"X-Lobby-Mode": "1", "Retry-After": "5"},
                )
                await response(scope, receive, send)
                return

        # Drain gate (only blocks NEW WS upgrades; HTTP unaffected)
        if drain_state.is_draining() and scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1012, "reason": "draining"})
            return

        await self.app(scope, receive, send)


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

    # ── Bind-state boot restore (keyless-restart fix, 2026-07-04) ──
    # /admin/bind persists its payload to runtime.json AND mutates the
    # live settings, but the mutation is process-local: any restart
    # (host reboot, docker restart, OOM) reverted settings.agent_api_key
    # / session-JWT secret / LLM keys / TOUP_TOKEN to spawn-time env
    # values while the agent still looked healthy and bound — every
    # chat message then failed "Authentication required" until the
    # platform's reclaim sweep forced a re-bind (≤6 min, and only for
    # pool containers it was watching). Re-applying the persisted bind
    # BEFORE anything reads settings makes bind state restart-proof by
    # design; the sweeps stay as backstops.
    try:
        _pre_restore_db_url = settings.database_url
        _restored_fields = runtime_identity.restore_at_boot()
        if _restored_fields:
            print(f"🔐 Bind restore: re-applied {_restored_fields} fields from runtime.json")
            if settings.database_url and settings.database_url != _pre_restore_db_url:
                from app.db.database import rebind_database as _rebind
                await _rebind(settings.database_url)
                print("🔐 Bind restore: engine rebound to persisted tenant DB URL")
    except Exception:
        logging.getLogger(__name__).exception(
            "[boot-restore] failed — continuing with env identity"
        )

    # ── Workspace permissions (replaces the manual post-rollout chmod) ──
    # The agent runs as root; every exec/PTY child drops to uid 1000. Files
    # root wrote in a previous container life are unwritable to that uid, so
    # a human had to run `chmod -R a+rwX workspace/generated` on each
    # hardened tenant after EVERY recreate. The container can do this for
    # itself (chmod needs no capability when you own the file), so it does.
    try:
        from app.services.workspace_perms import sweep_workspace_perms
        _perm_summary = sweep_workspace_perms()
        if _perm_summary.get("changed"):
            print(f"🔧 Workspace perms: {_perm_summary}")
    except Exception:
        logging.getLogger(__name__).exception(
            "[workspace-perms] boot sweep failed — continuing"
        )

    # ── Blue-green passive-boot gate ──────────────────────────
    # When the bridge does a blue-green upgrade, it spawns this
    # container as `toup-agent-<prefix>-bg` alongside the still-running
    # canonical container. Both share `/app/workspace`. If we restore
    # apps + start Baileys WhatsApp at boot, our spawn fights the old
    # container's still-locked workspace files (Expo `.expo/state.json`,
    # `node_modules/.bin/expo` lockfiles, Baileys auth state) — green's
    # /agent/health never goes ready and the bridge's _wait_for_health
    # times out at 120 s. Caught for tenant 871bac24 running Nokia-Snake.
    #
    # When TOUP_BG_PASSIVE=1 and the marker file is absent (first boot
    # of a fresh green), we skip workspace-touching initialisation and
    # schedule it for ~90 s later — by which point bridge has cut Caddy
    # over to us, drained the old container, and removed it. Workspace
    # files are released; our delayed init succeeds cleanly.
    #
    # The marker file in workspace ensures that an in-place container
    # restart (OOM, crash) after promotion boots normally instead of
    # re-entering passive mode. Bridge clears the marker before creating
    # each new green so the next upgrade gets a fresh passive cycle.
    from pathlib import Path as _Path
    _BG_PASSIVE_ENV = os.environ.get("TOUP_BG_PASSIVE") == "1"
    _BG_MARKER = _Path("/app/workspace/.toup_bg_promoted")
    _bg_passive_active = _BG_PASSIVE_ENV and not _BG_MARKER.exists()
    if _bg_passive_active:
        print("🟢 [BG_PASSIVE] Boot in passive mode — apps + Baileys WhatsApp deferred")
    elif _BG_PASSIVE_ENV:
        print("🟢 [BG_PASSIVE] Marker present, booting normally (post-promote restart)")

    # ── Startup ───────────────────────────────────────────────
    print("🤖 Toup Agent starting up...")
    _boot_progress.update(percent=5, phase="database")
    await init_db()
    print("✅ Database initialized")

    # ── Auto-update: REMOVED in Phase 3 ────────────────────────
    # This block used to run `git pull --ff-only` at startup from the agent's
    # working directory. In practice it was ALWAYS a silent no-op — `.git` is
    # excluded by .dockerignore so the container image never contained a
    # working tree for `git` to update. The log warning "Auto-update check
    # skipped" fired on every boot.
    #
    # Phase 3 ships SHA-tagged images via CI (docs/new-vps/14-*.md). The
    # image IS the release artifact. Remove any future impulse to add
    # startup self-update logic — it defeats the immutable-image invariant
    # that the rollout service's audit log depends on.
    _boot_progress.update(percent=10, phase="ready")
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

    _boot_progress.update(percent=15, phase="user_setup")
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

    _boot_progress.update(percent=25, phase="embeddings")
    # ── Agent stack initialization ────────────────────────────
    telegram_bot = None
    cron_service = None
    routine_runner = None  # Sibling scheduler for system-managed routines (email briefing, …)
    trigger_runner = None  # Event-driven dispatcher for trigger_events (email_received, …)
    subagent_manager = None
    skill_loader = None
    agent_runner = None
    tool_executor = None
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

        _boot_progress.update(percent=35, phase="skills")
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
        set_agent_runner(agent_runner, ws_broadcast=broadcast_to_user)
        set_ws_browser_refs(agent_runner, skill_loader)

        # ── Recover jobs orphaned by a previous crash/restart ─────────
        # Logic + rationale live in app/agent/job_recovery.py (extracted so
        # it is unit-testable; see tests/test_job_recovery.py). Summary:
        # `queued` untouched, `running` re-queued only where a drain loop
        # exists, everything else terminalised honestly.
        try:
            from app.db.database import async_session_maker
            from app.agent.job_recovery import recover_orphaned_jobs

            # Round 8: a job whose answer was ALREADY delivered is completed,
            # not "interrupted by the restart". Runs first, so recovery only
            # ever sees rows that genuinely have no answer behind them.
            # (job_reconciler.py — the same rule runs every minute after boot.)
            try:
                from app.agent.job_reconciler import reconcile_delivered_turn_jobs
                _reconciled = await reconcile_delivered_turn_jobs()
                if _reconciled:
                    print(f"🧾 Boot reconcile: completed {_reconciled} delivered job(s)")
            except Exception as _re:  # noqa: BLE001 — never block boot
                print(f"⚠️ Boot reconcile skipped: {_re}")

            _rec = await recover_orphaned_jobs(async_session_maker)
            _interrupted = _rec.interrupted
            _gave_up = _rec.gave_up
            if _rec.touched:
                print(
                    f"🧹 Restart recovery: re-queued {_rec.requeued}, "
                    f"interrupted {len(_interrupted)}, gave up {len(_gave_up)}"
                )

            # ── Live Activity / notification reconciliation ───────────
            #
            # A Live Activity card is ONLY closed by a terminal
            # notification (`mission_completed` / `mission_failed` →
            # alerting update + event=end; see
            # app/services/live_activity_service.py). A job row going
            # terminal in the DB does NOT touch the card. So every row we
            # terminalise here MUST emit a terminal notification, or its
            # lock screen / Dynamic Island card lingers for hours showing
            # stale progress.
            #
            # RE-QUEUED jobs get NO notification at all. Two reasons, both
            # load-bearing:
            #   1. They are still going to run, so a terminal event would
            #      end their card on a lie.
            #   2. Re-queueing is restricted to source_kind='trigger', and
            #      the trigger runner never emits `mission_started` — so a
            #      trigger job HAS no Live Activity card. Sending a
            #      `progress` row would not "heal" anything; the lane's
            #      restart-if-missing behaviour would CREATE a lock screen
            #      card for a background Gmail fire the user never asked to
            #      see. Silence is the correct output here.
            try:
                from app.agent.subagent_orchestrator import _notify_job_event

                for _jid, _title, _uid in _interrupted[:20]:
                    await _notify_job_event(
                        job_id=_jid, label=_title, kind="mission_failed",
                        title=f"Stopped: {(_title or 'background task')[:150]}",
                        body="Your agent restarted. Tap to run it again.",
                        dismiss_after_s=900, dedup_suffix="restart_interrupted",
                        # Background housekeeping at boot — a 3am container
                        # roll must not bypass quiet hours.
                        urgent=False,
                    )
                for _jid, _title, _uid in _gave_up[:20]:
                    await _notify_job_event(
                        job_id=_jid, label=_title, kind="mission_failed",
                        title=f"⚠️ Didn't finish: {(_title or 'background task')[:150]}",
                        body="We've been notified and are looking into it.",
                        dismiss_after_s=900, dedup_suffix="restart_orphan",
                        urgent=False,
                    )
            except Exception as _ne:
                logger.debug("[SWEEP] orphan card notify skipped: %s", _ne)
        except Exception as e:
            print(f"⚠️ Orphan job recovery skipped: {e}")

        _boot_progress.update(percent=50, phase="agent_pipeline")
        # ── App Manager + App Builder Skill ────────────────────
        # Each step has its own try/except so partial init succeeds.
        # If restore_on_startup fails, App Builder still registers.
        app_manager = None
        builder_skill = None
        app_gateway = None

        # Round 12: the legacy Expo pipeline is feature-gated. When it is
        # off we skip the ENTIRE block — AppManager, the restore sweep, the
        # 30s watchdog loop and both skills — not just the tool
        # registration. Leaving AppManager alive with no skills would keep a
        # background task polling ports for apps nothing can create, and
        # would keep `restore_on_startup` re-launching Metro processes on
        # every boot. The single-file HTML pipeline (`app_html`) is loaded
        # by `skill_loader.load_all()` above and needs none of this.
        from app.agent.tool_entitlements import pipeline_enabled as _pipeline_enabled
        _expo_on = _pipeline_enabled("expo")
        if not _expo_on:
            print(
                "🏗️ Expo app pipeline disabled (APP_BUILDER_EXPO_ENABLED=0) — "
                "apps build as single-file HTML artifacts",
                flush=True,
            )

        # Step 1: Create AppManager
        if _expo_on:
            try:
                from app.agent.app_manager import AppManager
                app_manager = AppManager()
                set_app_manager(app_manager)
                print("📱 App Manager created")
            except Exception as e:
                logger.error(f"[INIT] AppManager construction failed: {e}", exc_info=True)

        # Step 2: Restore previously-running apps (non-fatal).
        # Skipped during blue-green passive boot — see _bg_passive_active
        # comment at top of lifespan. Delayed-promote task below will
        # restore apps after the old container has been removed and
        # workspace lockfiles are released.
        if app_manager and not _bg_passive_active:
            try:
                restored = await app_manager.restore_on_startup(async_session_maker)
                if restored:
                    print(f"📱 App Manager: restored {restored} running app(s)")
            except Exception as e:
                logger.warning(f"[INIT] restore_on_startup failed (non-fatal, App Builder still available): {e}", exc_info=True)
        elif app_manager and _bg_passive_active:
            print("📱 [BG_PASSIVE] App restore deferred to post-promote phase")

        # Step 2b: Start app watchdog — TCP-probes each running app every
        # 30s and auto-revives any whose web server died. Without this, a
        # crashed Expo dev server stays dead until the user explicitly
        # restarts the app, which surfaces as a 503 preview loop and a
        # "Reconnecting…" banner that never resolves on its own.
        if app_manager:
            try:
                app.state.app_watchdog_task = asyncio.create_task(
                    app_manager.watchdog_loop()
                )
                print("🐕 App watchdog started")
            except Exception as e:
                logger.warning(f"[INIT] watchdog start failed (non-fatal): {e}", exc_info=True)

        # Step 3: Register AppBuilderSkill
        if app_manager:
            try:
                from app.agent.skills.builtins.app_builder.skill import AppBuilderSkill
                builder_skill = AppBuilderSkill(
                    app_manager=app_manager,
                    ws_broadcast=broadcast_to_user,
                )
                # register_dynamic returns False when the `app_builder` tool
                # family is withheld from this tenant
                # (app/agent/tool_entitlements.py). Don't hand the skill to
                # the /api/apps surface in that case, and don't print a
                # success line for a skill that isn't there — a wrong boot
                # log is how an unentitled tenant gets debugged as a bug.
                if await skill_loader.register_dynamic(builder_skill):
                    set_app_builder_skill(builder_skill)
                    print("🏗️ App Builder skill registered")
                else:
                    builder_skill = None
                    print("🏗️ App Builder skill withheld (not entitled)")
            except Exception as e:
                logger.error(f"[INIT] AppBuilderSkill registration failed: {e}", exc_info=True)
        else:
            logger.error("[INIT] AppBuilderSkill NOT registered — AppManager is None")

        # Step 4: Register AppGatewaySkill + load existing apps
        # Gated with the rest of the Expo pipeline. Without this guard the
        # gateway would still be constructed and handed to /api/apps while
        # `register_dynamic` silently refused it — 13 tools listed as the
        # app surface with no execution path behind them, which is exactly
        # the half-gated state SkillLoader._register warns about.
        if _expo_on:
            try:
                from app.agent.skills.builtins.app_builder.app_gateway_skill import AppGatewaySkill
                app_gateway = AppGatewaySkill()

                # Load existing apps into the gateway (non-fatal per app)
                if app_manager:
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
                        logger.warning(f"[INIT] App Gateway loading error (non-fatal): {e}", exc_info=True)

                await skill_loader.register_dynamic(app_gateway)
                if builder_skill:
                    builder_skill._app_gateway = app_gateway
                set_app_gateway(app_gateway)
                print(f"📱 App Gateway skill registered ({len(app_gateway.get_tools())} tools)")
            except Exception as e:
                logger.error(f"[INIT] AppGatewaySkill registration failed: {e}", exc_info=True)

        # ── Store module refs for hot-restart ────────────────
        import agent_main as _self
        _self._agent_runner = agent_runner
        _self._tool_executor = tool_executor
        _self._subagent_manager = subagent_manager
        _self._skill_loader = skill_loader
        _self._cron_service = cron_service

        _boot_progress.update(percent=65, phase="apps")
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
        # TKT-LAT-017 (wave 3): when agent_defer_scheduler_init is ON,
        # schedule cron_service.start() in the background so uvicorn can
        # accept the first request without waiting for APScheduler +
        # _load_jobs_from_db. Cron fires are time-based (daily/hourly),
        # so a 200-800ms startup delay never misses a tick.
        async def _boot_start_cron() -> None:
            import time as _t
            _t0 = _t.monotonic()
            try:
                await cron_service.start()
                _ms = int((_t.monotonic() - _t0) * 1000)
                print(f"⏰ Cron service started [PERF] boot_cron_start_ms={_ms}")
            except Exception as e:
                print(f"⚠️ Could not start cron service: {e}")

        if settings.agent_defer_scheduler_init:
            asyncio.create_task(_boot_start_cron(), name="lat017-cron-start")
            print("[PERF] boot_deferred=cron_start")
        else:
            try:
                await cron_service.start()
                print("⏰ Cron service started")
            except Exception as e:
                print(f"⚠️ Could not start cron service: {e}")
                cron_service = None

        # Routine scheduler — sibling of CronService. Starts AFTER CronService
        # so its DB session pool / async_session_maker proxy is already warm,
        # and after the WS server scaffolding (still pre-listen at this point;
        # the runner just registers triggers, doesn't fire until tick time).
        # The MCP client doesn't exist yet at this point (constructed later
        # in the lifespan around the MCP block), so the runner is constructed
        # without one; `set_mcp_client()` is called once the client is built.
        # Failure here is non-fatal: routines silently don't run, the rest of
        # the agent boots normally.
        # TKT-LAT-017 (wave 3): defer routine_runner.start() behind
        # the same flag. The runner just registers triggers; first
        # scheduled fire is the user's tz-local wake time — far future.
        try:
            from app.agent.routines import RoutineRunner
            routine_runner = RoutineRunner()

            async def _boot_start_routine() -> None:
                import time as _t
                _t0 = _t.monotonic()
                try:
                    await routine_runner.start()
                    _ms = int((_t.monotonic() - _t0) * 1000)
                    print(f"📅 Routine runner started [PERF] boot_routine_start_ms={_ms}")
                except Exception as e:
                    print(f"⚠️ Routine runner start failed (post-boot): {e}")

            if settings.agent_defer_scheduler_init:
                asyncio.create_task(_boot_start_routine(), name="lat017-routine-start")
                set_routines_runner_ref(routine_runner)
                print("[PERF] boot_deferred=routine_start")
            else:
                await routine_runner.start()
                set_routines_runner_ref(routine_runner)
                print("📅 Routine runner started")
        except Exception as e:
            print(f"⚠️ Could not start routine runner: {e}")
            routine_runner = None

        # Notify-outbox flush loop (Autopilot PR4) — drains the durable
        # agent_notify_outbox to the platform's /api/agent/notify. Plain
        # asyncio loop (no APScheduler dependency): tolerates pre-bind
        # lobby mode by sleeping through DB errors, and the platform
        # dedupes on the row id so restarts can only re-send acked-lost
        # rows, never double-notify.
        try:
            from app.services.agent_notify_client import notify_outbox_loop
            app.state.notify_outbox_task = asyncio.create_task(
                notify_outbox_loop(), name="notify-outbox-flush",
            )
            print("📮 Notify outbox loop started")
        except Exception as e:
            print(f"⚠️ Could not start notify outbox loop: {e}")

        # Stalled-job reaper — fails running jobs that stopped showing
        # signs of life so no progress surface (job card, capsule,
        # phone Live Activity) sits on a dead percentage forever.
        try:
            from app.agent.job_reaper import stalled_jobs_sweep_loop
            app.state.job_reaper_task = asyncio.create_task(
                stalled_jobs_sweep_loop(), name="stalled-jobs-sweep",
            )
            print("🧹 Stalled-job reaper started")
        except Exception as e:
            print(f"⚠️ Could not start stalled-job reaper: {e}")

        # Round 8: delivered-turn reconciler — no job may stay 'running'
        # after its turn's answer has been persisted. The turn-end finalizer
        # is the primary closer; this is the server-side guarantee behind it
        # (app/agent/job_reconciler.py), every 60 s.
        try:
            from app.agent.job_reconciler import reconcile_loop
            app.state.job_reconciler_task = asyncio.create_task(
                reconcile_loop(), name="delivered-jobs-reconcile",
            )
            print("🧾 Delivered-job reconciler started")
        except Exception as e:
            print(f"⚠️ Could not start job reconciler: {e}")

        # TriggerRunner — event-driven sibling. Started after RoutineRunner
        # so its restart sweep + rate-bucket warmup run before any inbound
        # webhook can dispatch. Auto-imports the email_received handler
        # which self-registers via the registry. MCP client wires later
        # (same lifespan ordering as routines).
        try:
            from app.agent.triggers import TriggerRunner
            # Importing email_received_handler triggers the auto-register
            # call at module load.
            from app.agent.triggers import email_received_handler  # noqa: F401
            trigger_runner = TriggerRunner()
            # G-19b: hand the AgentRunner to the trigger handlers so the
            # flag-gated email_received turn (trigger_turns_via_runner)
            # can run the full pipeline. Unlike the MCP client (which
            # late-binds via mcp_bootstrap), the AgentRunner already
            # exists at this lifespan point — wire it at construction.
            trigger_runner.set_agent_runner(agent_runner)
            await trigger_runner.start()
            from app.api.triggers_inbound import set_runner_ref as set_triggers_runner_ref
            set_triggers_runner_ref(trigger_runner)
            # Also hand the runner to the CRUD module so the test-fire
            # endpoint can dispatch synthesised events directly.
            set_triggers_crud_runner_ref(trigger_runner)
            print("⚡ Trigger runner started")
        except Exception as e:
            print(f"⚠️ Could not start trigger runner: {e}")
            trigger_runner = None

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

        # ── Memory maintenance (audit A6-1) ───────────────────
        # Decay / consolidation / end-of-day archival / retrieval-feedback
        # analysis were registered only in platform_main's scheduler, whose
        # DB excludes the AGENT_ONLY memories/day_chats tables — so they
        # never ran against tenant data. Mirror the exact same entry points
        # here, on the tenant scheduler where those tables actually live.
        # Flag-gated (default OFF); each registration is individually
        # guarded so one failing job never kills the lifespan.
        # NOTE (review pr4-#1): these jobs ride CronService's APScheduler,
        # so they inherit its cron_service_enabled kill switch — if that
        # Phase-C deprecation gate is ever flipped off, the registrations
        # below sit on a never-started scheduler (same as the existing
        # heartbeat/auto-update jobs). Migrate them together when
        # CronService is retired.
        if (
            settings.agent_memory_maintenance_enabled
            and cron_service
            and cron_service.scheduler
        ):
            from apscheduler.triggers.cron import CronTrigger as _MMCron

            _mm_jobs = []
            try:
                from app.scripts.scheduled_tasks import run_end_of_day_archival
                from app.services.current_context import (
                    run_context_rollover as _context_rollover,
                )
                from app.services.memory_file_ops import run_memory_maintenance

                _mm_jobs = [
                    (
                        # v3 (docs/memory/rebuild-2026-08-v3.md §1.1): the two
                        # jobs that used to sit here are DELETED with the row
                        # engine. `memory_decay` walked `memories.strength`
                        # through an Ebbinghaus curve; `memory_consolidation`
                        # ran `run_memory_file_maintenance`, an LLM curation
                        # pass over ROWS grouped into files. Neither has an
                        # output any client can see.
                        #
                        # One cheap slot replaces both, and it keeps the id
                        # `memory_consolidation` so a redeploy REPLACES the
                        # old registration on a scheduler that survived
                        # (replace_existing=True keys on the id) rather than
                        # leaving an orphan pointing at a deleted function.
                        # WS-5's migration hooks in here.
                        "memory_consolidation",
                        run_memory_maintenance,
                        # CRON, not interval (rebuild-2026-08 RC3.1): an
                        # interval's first fire is measured from scheduler
                        # start, and the fleet is recreated on every merge to
                        # main (median gap 0.3h at the 2026-08 audit) — so an
                        # interval job never fired once.
                        _MMCron(hour=settings.consolidation_cron_hour, minute=0),
                    ),
                    # Current context's day rollover (memory v3 §6). HOURLY
                    # and CRON — hourly because a fleet of users in every
                    # timezone crosses midnight at 24 different UTC hours, so
                    # a daily job would age half of them a day late; cron
                    # rather than interval for the same RC3.1 reason as the
                    # slot above. Cheap: it reads one row and returns unless
                    # the user's LOCAL date has actually advanced.
                    (
                        "current_context_rollover",
                        _context_rollover,
                        _MMCron(minute=5),
                    ),
                    # retrieval_feedback_analysis is RETIRED with sentence
                    # retrieval (memory v3 §3.1). It read `retrieval_events`,
                    # whose only feeder was the runner's per-turn
                    # `log_retrieval_feedback` call — and that call's input
                    # was hybrid_search's results, which the file model no
                    # longer produces. A weekly job over a table nothing
                    # writes is not observability, it is a cron that always
                    # reports zero.
                ]
                # Hourly archival summaries — same enable_day_recall gate as
                # platform_main's setup_scheduler (forced ON in the agent
                # image, Dockerfile.agent). The job already summarizes any
                # rolled-over day with >=1 user/assistant message (A6-9):
                # light days that never hit the rolling-summary debounce
                # still get an archival summary, so <recent_days> and
                # recall_day stop silently skipping them.
                if settings.enable_day_recall:
                    # Cron for the same reason as memory_decay above — an
                    # hourly interval also never fires on a fleet that
                    # restarts more often than hourly.
                    _mm_jobs.append(
                        ("day_archival", run_end_of_day_archival, _MMCron(minute=0))
                    )
            except Exception as e:
                print(f"⚠️ Memory maintenance imports failed: {e}")

            _mm_registered = []
            for _mm_id, _mm_fn, _mm_trigger in _mm_jobs:
                try:
                    cron_service.scheduler.add_job(
                        _mm_fn,
                        trigger=_mm_trigger,
                        id=_mm_id,
                        name=f"Memory Maintenance: {_mm_id}",
                        replace_existing=True,
                    )
                    _mm_registered.append(_mm_id)
                except Exception as e:
                    print(f"⚠️ Could not register memory job {_mm_id}: {e}")
            if _mm_registered:
                print(f"🧠 Memory maintenance jobs: {', '.join(_mm_registered)}")
            try:
                from datetime import datetime as _dt, timedelta as _td
                from apscheduler.triggers.date import DateTrigger as _MMDate
                from app.services.memory_file_ops import (
                    run_memory_maintenance as _mm_boot,
                )
                cron_service.scheduler.add_job(
                    _mm_boot,
                    trigger=_MMDate(run_date=_dt.now() + _td(seconds=180)),
                    id="memory_file_migration_boot",
                    name="Memory Maintenance: boot system files",
                    replace_existing=True,
                )
                print("🧠 Memory system-file check scheduled for T+180s")
            except Exception as e:
                print(f"⚠️ Could not schedule memory file migration: {e}")

    except Exception as e:
        print(f"⚠️ Agent initialization error: {e}")
        import traceback
        traceback.print_exc()

    _boot_progress.update(percent=80, phase="channels")
    # ── Hook Bus ──────────────────────────────────────────────
    from app.agent.hooks import get_hook_bus, HookEvent
    _hook_bus = get_hook_bus()
    await _hook_bus.emit(HookEvent.STARTUP, {"app": "toup-agent"})
    print("🔌 Hook bus started")

    # ── MCP Client (connect to Platform MCP server) ──────────
    # Extracted to app/agent/mcp_bootstrap.ensure_mcp_initialized so
    # POST /admin/bind and PUT /api/agent/refresh-tools can run the
    # same bootstrap on pool containers. Those boot in lobby mode
    # WITHOUT agent_api_key, so this lifespan call is a no-op for them
    # ("not_configured") — historically that meant they NEVER got
    # connector tools (gmail__* …) while the platform showed the OAuth
    # identity as Connected. The T1g auth + T1h cache notes moved into
    # that module. Runner refs go on app.state so the bootstrap can
    # wire them no matter which caller runs first.
    app.state.tool_executor = tool_executor
    app.state.agent_runner = agent_runner
    app.state.routine_runner = routine_runner
    app.state.trigger_runner = trigger_runner
    from app.agent.mcp_bootstrap import ensure_mcp_initialized
    await ensure_mcp_initialized(
        app, defer_initial_refresh=bool(settings.agent_defer_boot_init)
    )

    _boot_progress.update(percent=85, phase="hooks")
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
    # Mode-gated: pick exactly one transport per container based on
    # settings.whatsapp_mode. Cloud API (Path A) and QR-link (Path C)
    # are mutually exclusive — both extend BaseChannel(WHATSAPP) so
    # only one slot in the registry is ever populated.
    #
    # Defaulting:
    #   1) Legacy Cloud API tenants (phone_id + access_token in DB,
    #      mode column never set) → cloud_api. Keeps them working
    #      without forcing a one-time settings save.
    #   2) Everyone else (new users, mode column empty) → qr_link.
    #      This makes Connect-via-QR work on first click. Without
    #      this, brand-new users got "Agent didn't come back in QR
    #      mode within 90 s" because the agent booted with NO
    #      WhatsApp channel and the mode-switch save never had a
    #      live channel to flip into.
    whatsapp_channel = None
    _wa_mode = (settings.whatsapp_mode or "").strip().lower()
    if not _wa_mode:
        if settings.whatsapp_phone_number_id and settings.whatsapp_access_token:
            _wa_mode = "cloud_api"
        else:
            _wa_mode = "qr_link"

    if _wa_mode == "qr_link" and not _bg_passive_active:
        try:
            from app.agent.channels.whatsapp_baileys import BaileysWhatsAppChannel
            from app.agent.channels.registry import ChannelRegistry
            from app.agent.channels.shared import make_channel_handler
            _allowlist = [
                s.strip() for s in (settings.whatsapp_baileys_allowlist or "").split(",")
                if s.strip()
            ]
            whatsapp_channel = BaileysWhatsAppChannel(allowed_numbers=_allowlist)
            whatsapp_channel.set_message_callback(
                make_channel_handler(
                    channel=whatsapp_channel,
                    agent_runner=agent_runner,
                    user_id=settings.user_id,
                )
            )
            await whatsapp_channel.start()
            ChannelRegistry.register(whatsapp_channel)
            print("📱 WhatsApp channel started (QR-link / Baileys sidecar)")
        except Exception as e:
            print(f"⚠️ WhatsApp QR-link error: {e}")
    elif _wa_mode == "qr_link" and _bg_passive_active:
        # Baileys' /app/workspace/.whatsapp_auth/ would contend with the
        # still-running old container's auth state. Defer to post-promote.
        print("📱 [BG_PASSIVE] WhatsApp Baileys deferred to post-promote phase")
    elif _wa_mode == "cloud_api" and settings.whatsapp_phone_number_id and settings.whatsapp_access_token:
        try:
            from app.agent.channels.whatsapp_channel import WhatsAppChannel
            from app.agent.channels.registry import ChannelRegistry
            from app.agent.channels.shared import make_channel_handler
            whatsapp_channel = WhatsAppChannel(
                phone_number_id=settings.whatsapp_phone_number_id,
                access_token=settings.whatsapp_access_token,
                verify_token=settings.whatsapp_verify_token,
                app_secret=settings.whatsapp_app_secret,
                allowed_numbers=settings.whatsapp_allowed_numbers or None,
            )
            whatsapp_channel.set_message_callback(
                make_channel_handler(
                    channel=whatsapp_channel,
                    agent_runner=agent_runner,
                    user_id=settings.user_id,
                )
            )
            whatsapp_channel.register_routes(app)
            await whatsapp_channel.start()
            ChannelRegistry.register(whatsapp_channel)
            print("📱 WhatsApp channel started (Cloud API)")
        except Exception as e:
            print(f"⚠️ WhatsApp Cloud API error: {e}")

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
            # TKT-LAT-017: tunnel start sets up a WS to the platform
            # for config sync + chat relay. The first chat request does
            # NOT require it (chat lands via the agent's own /ws/chat),
            # so we can defer the start to a background task and let
            # uvicorn begin serving immediately. The tunnel reconnects
            # itself on connection drops, so any race during boot is
            # handled by its own reconnect loop.
            async def _boot_start_tunnel() -> None:
                import time as _t
                _t0 = _t.monotonic()
                try:
                    await tunnel_client.start()
                    _ms = int((_t.monotonic() - _t0) * 1000)
                    print(
                        f"🔗 Platform tunnel connecting... "
                        f"[PERF] boot_tunnel_start_ms={_ms}"
                    )
                except Exception as e:
                    print(f"⚠️ Platform tunnel not available: {e}")

            if settings.agent_defer_boot_init:
                asyncio.create_task(
                    _boot_start_tunnel(), name="lat017-tunnel-start"
                )
                print("[PERF] boot_deferred=tunnel_start")
            else:
                await _boot_start_tunnel()
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
    # use_day_chat_context defaults to True (config.py). The day-chat
    # context path activates once backfill completes. Backfill failure
    # is never fatal — agent falls back to session-based history.
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
                    print("day_chat_backfill.skipped reason=already_completed")
                elif result == "completed":
                    print("day_chat_backfill.done")
                elif result == "failed":
                    print("⚠️ day_chat_backfill.skipped reason=previously_failed")
            except Exception as e:
                # Backfill failure is never fatal — agent continues on old path
                print(f"⚠️ day_chat_backfill.crash error={e}")

        _backfill_task = asyncio.create_task(_run_day_chat_backfill())
        print("day_chat_backfill.scheduled")
    except Exception as e:
        print(f"⚠️ day_chat_backfill.skipped reason=import_error error={e}")

    # ── DB watchdog: detect + cure a poisoned engine in-process ──
    # SELECT 1 every 60s on a fresh session; 2 consecutive failures →
    # recover_engine() (rebuild + holder swap — the DB-layer effect of a
    # docker restart, without one). Closes the 2026-07-04 class where an
    # interrupted transaction left the agent in PendingRollbackError
    # churn for days ("Connection lost" on every message) while
    # /agent/health stayed green. State surfaces as db_ok/db_recoveries
    # in /agent/health — diagnostic only, nothing gates on it.
    try:
        from app.services.db_watchdog import db_watchdog_loop
        app.state.db_watchdog_task = asyncio.create_task(db_watchdog_loop())
        print("🩺 DB watchdog started")
    except Exception as e:
        print(f"⚠️ DB watchdog failed to start: {e}")

    # One-time cleanup: remove legacy dashboard/ folder (replaced by .dashboard/)
    try:
        import os as _os
        _ws = getattr(settings, 'agent_workspace_dir', None) or './workspace'
        _legacy_dash = _os.path.join(_os.path.abspath(_ws), 'dashboard')
        if _os.path.isdir(_legacy_dash):
            if not _os.listdir(_legacy_dash):
                _os.rmdir(_legacy_dash)
                print("🧹 Removed empty legacy dashboard/ folder")
            else:
                print(f"⚠️ Legacy dashboard/ folder is not empty ({len(_os.listdir(_legacy_dash))} items), skipping removal")
    except Exception as _e:
        print(f"⚠️ dashboard/ cleanup skipped: {_e}")

    _boot_progress.update(percent=100, phase="ready", ready=True)
    print("🤖 Toup Agent ready.")
    print(f"   Server:  http://0.0.0.0:8001")
    print(f"   Health:  http://localhost:8001/agent/health")
    print(f"   Web UI:  https://toup.ai")
    print(f"   Press Ctrl+C to stop.\n")

    # ── Blue-green delayed promote ────────────────────────────
    # When this container booted in passive mode, schedule the
    # workspace-touching initialisation we deferred. By the time this
    # runs (~90 s), the bridge has cut Caddy over to us, drained, and
    # removed the old container — workspace lockfiles are released so
    # AppManager.restore_on_startup and Baileys' .whatsapp_auth/ open
    # both succeed. Marker file ensures in-place restarts of THIS
    # container after promotion boot normally.
    if _bg_passive_active:
        async def _bg_delayed_promote():
            # 90 s = bridge's drain (10 s) + old kill + buffer.
            # Tune via TOUP_BG_PROMOTE_DELAY_SEC if rollout timing changes.
            delay = int(os.environ.get("TOUP_BG_PROMOTE_DELAY_SEC", "90"))
            try:
                await asyncio.sleep(delay)
                print(f"🟢 [BG_PROMOTE] starting deferred init after {delay}s")

                # Restore previously-running apps (the actual reason we
                # gated boot — old container's Expo lockfiles are gone now).
                if app_manager:
                    try:
                        restored = await app_manager.restore_on_startup(async_session_maker)
                        print(f"🟢 [BG_PROMOTE] restored {restored} app(s)")
                    except Exception as e:
                        logger.warning(
                            "[BG_PROMOTE] restore_on_startup failed (non-fatal): %s", e,
                            exc_info=True,
                        )

                # Start WhatsApp Baileys if configured. Reuses the same
                # restart helper used by tunnel_client config-sync, so
                # the channel boots with the current settings snapshot.
                try:
                    if (settings.whatsapp_mode or "").strip().lower() == "qr_link":
                        await restart_whatsapp_channel()
                        print("🟢 [BG_PROMOTE] WhatsApp Baileys started")
                except Exception as e:
                    logger.warning(
                        "[BG_PROMOTE] WhatsApp restart failed (non-fatal): %s", e,
                        exc_info=True,
                    )

                # Write marker: future restarts of this container boot
                # normally, NOT passive. Bridge clears the marker before
                # creating the next green so it gets a fresh passive cycle.
                try:
                    _BG_MARKER.parent.mkdir(parents=True, exist_ok=True)
                    _BG_MARKER.write_text(str(int(_time.time())))
                    print(f"🟢 [BG_PROMOTE] marker written at {_BG_MARKER}")
                except Exception as e:
                    logger.warning("[BG_PROMOTE] marker write failed: %s", e)
            except asyncio.CancelledError:
                # Container shutting down before we finished — fine.
                raise
            except Exception as e:
                logger.error("[BG_PROMOTE] unexpected error: %s", e, exc_info=True)

        try:
            app.state.bg_promote_task = asyncio.create_task(_bg_delayed_promote())
            print("🟢 [BG_PASSIVE] Promote scheduled")
        except Exception as e:
            logger.warning("[BG_PASSIVE] could not schedule promote task: %s", e)

    yield

    # ── Shutdown (reverse order) ──────────────────────────────
    logger.info("[SHUTDOWN] Agent shutting down — marking in-flight jobs as failed")

    # Checkpoint: mark any running jobs as failed before exit
    try:
        from app.db.models import BuildJob
        from sqlalchemy import select as _shutdown_sel
        async with async_session_maker() as _sdb:
            _running = await _sdb.execute(
                _shutdown_sel(BuildJob).where(BuildJob.status == "running")
            )
            _running_jobs = _running.scalars().all()
            for _rj in _running_jobs:
                _rj.status = "failed"
                _rj.error_message = "Agent shutdown during execution"
                _rj.completed_at = datetime.utcnow()
                logger.info("[SHUTDOWN] Marked job %s as failed", _rj.id[:8])
            if _running_jobs:
                await _sdb.commit()
                logger.info("[SHUTDOWN] Checkpointed %d in-flight jobs", len(_running_jobs))
    except Exception as _se:
        logger.warning("[SHUTDOWN] Failed to checkpoint jobs: %s", _se)

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

    # RoutineRunner must stop BEFORE CronService: handlers may use the DB
    # session pool that CronService also holds — clean shutdown order avoids
    # "session already closed" noise during the drain window.
    if routine_runner:
        try:
            await routine_runner.stop()
            print("📅 Routine runner stopped")
        except Exception:
            pass

    # TriggerRunner shutdown order mirrors RoutineRunner — must drain
    # in-flight handlers before the DB pool tears down.
    if trigger_runner:
        try:
            await trigger_runner.stop()
            print("⚡ Trigger runner stopped")
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

    # Stop watchdog before cleanup so it doesn't try to revive apps we're
    # tearing down.
    wd_task = getattr(app.state, "app_watchdog_task", None)
    if wd_task and not wd_task.done():
        wd_task.cancel()
        try:
            await wd_task
        except (asyncio.CancelledError, Exception):
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


# The App-MCP ASGI app must exist BEFORE the FastAPI constructor: its
# lifespan starts FastMCP's StreamableHTTPSessionManager task group, and
# mounting alone never runs a sub-app's lifespan. Without this
# composition every authenticated /api/app-mcp/mcp request 500s
# (GA audit, closeout B-1). Fail-open on build failure — app-MCP
# degrades, the tenant container still boots.
try:
    from app.agent.app_mcp_server import app_mcp as _app_mcp_server
    mcp_app = _app_mcp_server.http_app(path="/mcp")
except Exception as _mcp_build_err:
    mcp_app = None
    print(f"⚠️ App MCP app not built: {_mcp_build_err}", flush=True)

if mcp_app is not None:
    @asynccontextmanager
    async def _combined_lifespan(_app: FastAPI):
        # Agent services up first, MCP session manager second,
        # torn down in reverse.
        async with lifespan(_app):
            async with mcp_app.lifespan(_app):
                yield

    _boot_lifespan = _combined_lifespan
else:
    _boot_lifespan = lifespan

app = FastAPI(
    title="Toup Agent",
    description="Personal AI Agent with tools, channels, and memory access",
    version="6.0.0",
    lifespan=_boot_lifespan,
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

# Lobby + drain gate. ORDER MATTERS: this needs to run BEFORE the
# X-Agent-Key check so a generic pool container (which has no
# X-Agent-Key configured) can serve /admin/bind without first
# having the API-key middleware 401 it. FastAPI applies middlewares
# in REVERSE add order, so adding LobbyAndDrain after AgentAPIKey
# means LobbyAndDrain runs FIRST in the request chain.
app.add_middleware(LobbyAndDrainMiddleware)

# ── Register agent routers ───────────────────────────────────────────
app.include_router(agent_router, prefix=settings.api_prefix)
app.include_router(stats_router, prefix=settings.api_prefix)
app.include_router(memories_router, prefix=settings.api_prefix)
# Toup Media — saved playlists. AGENT_ONLY store lives HERE; the platform
# proxies user requests in with X-Agent-Key (memories pattern), and the play
# endpoint broadcasts over this process's chat WS queues.
from app.api.media_playlists import router as media_playlists_router
app.include_router(media_playlists_router, prefix=settings.api_prefix)
# Document / media / conversation ingestion. `documents`, `document_chunks`,
# `media`, `conversations`, `messages`, `entities` and `memories` are ALL
# AGENT_ONLY, so this is where the store is — yet both routers used to be
# mounted ONLY by platform_main.py, where those relations do not exist and
# every upload was a 500. Same shape as memories/playlists: the platform
# proxies in with X-Agent-Key, these handlers execute locally.
# ingest_router first, matching platform_main.py's order — both use the
# /ingest prefix and the paths are disjoint.
from app.api.ingest import router as ingest_router
from app.api.documents import router as documents_router
app.include_router(ingest_router, prefix=settings.api_prefix)
app.include_router(documents_router, prefix=settings.api_prefix)
app.include_router(sessions_router, prefix=settings.api_prefix)
app.include_router(day_chats_router, prefix=settings.api_prefix)
app.include_router(chat_router, prefix=settings.api_prefix)
app.include_router(messages_recover_router, prefix=settings.api_prefix)  # GET /api/messages/since/<id>
app.include_router(ws_chat_router, prefix=settings.api_prefix)
app.include_router(api_v1_router, prefix=settings.api_prefix)
app.include_router(models_router, prefix=settings.api_prefix)    # GET /api/models — model registry
app.include_router(webhooks_router, prefix=settings.api_prefix)
app.include_router(voice_router, prefix=settings.api_prefix)
app.include_router(ws_realtime_router, prefix=settings.api_prefix)
app.include_router(dashboard_router, prefix=settings.api_prefix)
# The file library — /api/library/* (id-based) and the /api/workspace/*
# path shapes the shipped clients call — a VIRTUAL tree over this tenant's
# files (app/services/library_service.py). The platform is a pass-through
# proxy (app/api/workspace_proxy.py). AgentAPIKeyMiddleware (added above)
# gates every /api route behind X-Agent-Key; get_current_user resolves it
# to the tenant owner.
app.include_router(library_router, prefix=settings.api_prefix)
app.include_router(ws_browser_router, prefix=settings.api_prefix)
# Unified jobs activity feed (PR 6 of the jobs/tasks/logs arc) —
# server-side ``GET /apps/jobs/events`` query over job_events JOIN
# build_jobs. Replaces the dashboard's client-side flatten of
# BuildJob.steps_json that mis-attributed every entry to the most
# recent Auto Builder app.
#
# MUST be registered BEFORE apps_router: apps.py defines
# ``GET /jobs/{job_id}`` and Starlette matches routes in
# registration order, so registering the literal ``/jobs/events``
# after the param route shadows it — the feed 404s as
# ``db.get(BuildJob, "events")`` and the dashboard silently falls
# back to the legacy client-side flatten (2026-07-08 prod bug).
try:
    from app.api.jobs_events import router as jobs_events_router
    app.include_router(
        jobs_events_router, prefix=f"{settings.api_prefix}/apps/jobs",
    )
except ImportError as _e:
    print(f"⚠️ jobs_events router not loaded: {_e}", flush=True)
app.include_router(apps_router, prefix=settings.api_prefix)
# Single-file HTML artifacts (round 12) — raw bytes for the platform's
# artifact_proxy, which adds the browser-facing CSP. Behind the global
# X-Agent-Key middleware like every other /api route here.
try:
    from app.api.artifacts import router as artifacts_router
    app.include_router(artifacts_router, prefix=settings.api_prefix)
except ImportError as _e:
    print(f"⚠️ artifacts router not loaded: {_e}", flush=True)
# Netflix streaming (HLS)
try:
    from app.api.ws_netflix import router as netflix_stream_router
    app.include_router(netflix_stream_router, prefix=settings.api_prefix)
except ImportError as e:
    print(f"⚠️ Netflix stream not mounted: {e}")
app.include_router(soul_router, prefix=settings.api_prefix)
# The `identities` table is AGENT_ONLY (base.py) and `PUT /api/soul` compiles
# into it right here on the agent — but until now nothing on the agent SERVED
# it, so `GET /api/identity` answered 404 on every tenant while the platform's
# realtime bootstrap called exactly that URL. Mounted beside soul because they
# are two views of the same agent-side data.
app.include_router(identity_router, prefix=settings.api_prefix)
app.include_router(llm_setup_router, prefix=settings.api_prefix)
# Generated-file attachments — data + files live here on the agent.
app.include_router(files_router, prefix=settings.api_prefix)
app.include_router(whatsapp_qr_router, prefix=settings.api_prefix)
# Pool admin endpoints (Phase A/B): /api/admin/bind, /api/admin/drain,
# /api/admin/status. Auth enforced inside the handlers via
# X-Pool-Admin-Token; bypassed by the X-Agent-Key middleware via
# _PUBLIC_PATHS membership.
app.include_router(admin_pool_router, prefix=settings.api_prefix)

# Routines (system-managed scheduled actions — email briefing, etc.).
# Gate 1 ships /api/routines/_runner_status only; full CRUD lands in Gate 3.
app.include_router(routines_router, prefix=settings.api_prefix)

# Triggers — event-driven automations (Gmail Pub/Sub, etc.).
# Gate T1: /api/triggers/inbound (platform dispatches here).
# Gate T2: /api/triggers/* (CRUD + event history + test-fire).
app.include_router(triggers_inbound_router, prefix=settings.api_prefix)
app.include_router(triggers_router, prefix=settings.api_prefix)

# Out-of-band notification delivery (Autopilot PR4) — the platform
# dispatcher POSTs /api/notify/deliver when push can't reach the user;
# Telegram/WhatsApp connections live in THIS container only.
try:
    from app.api.notify_deliver import router as notify_deliver_router
    app.include_router(notify_deliver_router, prefix=settings.api_prefix)
except ImportError as _e:
    print(f"⚠️ notify_deliver router not loaded: {_e}", flush=True)

# Admin Dispatch (operator → user) — the platform's fan-out worker POSTs
# /api/internal/admin-notice per recipient. The chat row must be written
# here: `messages` is AGENT_ONLY and broadcast_to_user is in-process.
try:
    from app.api.admin_notice import router as admin_notice_router
    app.include_router(admin_notice_router, prefix=settings.api_prefix)
except ImportError as _e:
    print(f"⚠️ admin_notice router not loaded: {_e}", flush=True)

# Autopilot approvals (Autopilot PR7) — the durable ask-the-user store;
# Mission Control + push deep links decide through the platform proxy.
try:
    from app.api.autopilot import router as autopilot_router
    app.include_router(autopilot_router, prefix=settings.api_prefix)
except ImportError as _e:
    print(f"⚠️ autopilot router not loaded: {_e}", flush=True)

# Chrome extension — /ws/extension lives here (per-tenant agent VPS).
# Pairing/suggest/ground routes are platform-only; mounting the whole
# router on the agent is a no-op for those (the extension never calls
# the agent for pairing — it calls the platform).
from app.api.extension import router as extension_router
app.include_router(extension_router, prefix=settings.api_prefix)

# Mount App MCP server for external MCP clients. The ASGI app was built
# BEFORE the FastAPI constructor so its lifespan is composed into
# _combined_lifespan (see above) — mounting here keeps router
# registration order unchanged.
if mcp_app is not None:
    app.mount("/api/app-mcp", mcp_app)
else:
    print("⚠️ App MCP server not mounted: build failed at boot", flush=True)


def _resolved_default_model() -> str:
    """Lazy resolver call so health endpoints report the actual default
    a tenant call would resolve to, not whatever stale value was on
    `settings.agent_model` at import time."""
    from app.services.model_resolver import default_model
    return default_model()


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
    from app.services.model_resolver import default_model

    # WhatsApp surfaces a rich status object so an operator (and the
    # Settings UI) can diagnose "is it working?" without SSHing into
    # the container. Two transport modes are mutually exclusive —
    # check QR-link first (newer default), fall through to Cloud API,
    # then "configured" / "disabled" fallbacks.
    #
    # Always attaches `qr_supported` (bool): whether the Baileys
    # sidecar bundle is present on this image so the Settings UI can
    # surface "QR mode not available on this image" instead of letting
    # the user wait forever for a QR that'll never come.
    try:
        from app.agent.channels.whatsapp_channel import get_active_channel as _wa_cloud
        from app.agent.channels.whatsapp_baileys import get_active_baileys_channel as _wa_baileys

        _qr_supported = False
        try:
            from pathlib import Path as _Path
            _sidecar_dir = _Path("/app/whatsapp_sidecar")
            _qr_supported = (
                (_sidecar_dir / "sidecar.mjs").is_file()
                and (_sidecar_dir / "node_modules").is_dir()
            )
        except Exception:
            pass

        _baileys = _wa_baileys()
        if _baileys is not None:
            whatsapp_status = _baileys.health()
            whatsapp_status["qr_supported"] = _qr_supported
        else:
            _cloud = _wa_cloud()
            if _cloud is not None:
                whatsapp_status = _cloud.health()
                whatsapp_status["qr_supported"] = _qr_supported
            else:
                # Wrap the fallback into a dict so the frontend always
                # gets a consistent shape it can read qr_supported from.
                whatsapp_status = {
                    "configured": bool(
                        settings.whatsapp_mode or settings.whatsapp_phone_number_id
                    ),
                    "started": False,
                    "mode": settings.whatsapp_mode or None,
                    "qr_supported": _qr_supported,
                    "session_status": "not_linked",
                }
    except Exception:
        whatsapp_status = {"configured": False, "started": False, "qr_supported": False}

    # Pool-bind state — lets the platform distinguish a GENERIC, not-yet-bound
    # warm container (which still reports boot_progress.ready=true) from one
    # actually bound to a user with credentials applied. Without this the
    # platform declared a claim "ready" on mere reachability, so the user's
    # first message raced /admin/bind and 401'd ("Your API key is invalid").
    # Readiness must require is_bound AND bound_user_id == the claiming user.
    try:
        from app.services import runtime_identity as _ri
        _is_bound = _ri.is_bound()
        _bound_uid = _ri.get_user_id()
        _pool_generic = _ri.is_pool_generic()
    except Exception:
        _is_bound, _bound_uid, _pool_generic = None, None, None

    # Honest-health fields (bulletproof plan H). The 2026-07-04 arc
    # proved "healthy" here can coexist with every chat failing:
    # keyless agents (restart lost the bind secrets) and a poisoned DB
    # engine (PendingRollbackError churn) were both invisible. These
    # fields make those states visible in one glance. Diagnostic ONLY —
    # law 1 says never gate service on a label, so nothing may branch
    # on them except dashboards/alerts.
    try:
        _has_session_secret = bool(
            settings.agent_api_key or _ri.get_agent_api_key()
        )
    except Exception:
        _has_session_secret = None
    try:
        from app.services.db_watchdog import watchdog_state as _wd_state
        _wd = _wd_state()
        _db_ok = _wd.get("db_ok")
        _db_recoveries = _wd.get("recoveries")
    except Exception:
        _db_ok, _db_recoveries = None, None

    # Embedding degrade observability. Memory writes never fail on a broken
    # embedding backend — they degrade to unembedded rows (dedup off, vector
    # recall blind for those rows) with only a container-log WARNING. Surface
    # the counter here so fleet-wide unembedded writes are visible in one
    # health poll; `provider` reports the CACHED resolution only (never
    # forces a resolve from a health probe). Diagnostic ONLY — law 1, nothing
    # gates on it. Recovery lever: app/scripts/backfill_embeddings.py.
    try:
        from app.services.embedding_service import (
            EmbeddingService as _EmbSvc,
            embed_degrade_stats as _embed_degrade_stats,
        )
        _emb_inst = _EmbSvc._instance
        _emb_provider = (
            _emb_inst.__dict__.get("_resolved_provider") if _emb_inst is not None else None
        ) or _EmbSvc._resolved_provider
        _embeddings_status = {"provider": _emb_provider, **_embed_degrade_stats()}
    except Exception:
        _embeddings_status = None

    return {
        "status": "healthy",
        "version": _agent_version,
        "mode": "agent",
        "uptime_seconds": round(uptime, 1),
        "has_session_secret": _has_session_secret,
        "db_ok": _db_ok,
        "db_recoveries": _db_recoveries,
        # Reports the actual resolved default — what the runtime will use —
        # not just the raw settings field. Closes the gap where /agent/health
        # advertised a stale model after a settings.agent_model bump.
        "agent_model": default_model(),
        "embeddings": _embeddings_status,
        "boot_progress": _boot_progress,
        "is_bound": _is_bound,
        "bound_user_id": _bound_uid,
        "pool_generic": _pool_generic,
        "channels": {
            "telegram": "enabled" if settings.telegram_bot_token else "disabled",
            "discord": "enabled" if settings.discord_bot_token else "disabled",
            "slack": "enabled" if settings.slack_bot_token else "disabled",
            "whatsapp": whatsapp_status,
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
        # Resolved default — see /agent/health for the same rationale.
        "agent_model": _resolved_default_model(),
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
