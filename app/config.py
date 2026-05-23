from pydantic_settings import BaseSettings
from pydantic import model_validator
from functools import lru_cache
from typing import Optional


class Settings(BaseSettings):
    # App
    app_name: str = "Toup Agent Platform"
    debug: bool = True

    # Deployment environment. Drives the sk_live_ / sk_test_ guard below.
    # Anything other than "production" treats the deployment as non-prod and
    # forbids live Stripe keys. Set via ENVIRONMENT env var.
    environment: str = "production"
    
    # Database
    database_url: str = "sqlite+aiosqlite:///./toup.db"
    
    # For production PostgreSQL with pgvector (per-tenant DBs on the VPS):
    # database_url: str = "postgresql+asyncpg://toup_agent_<prefix>:pw@host.docker.internal:6432/toup_agent_<prefix>"
    
    # Property alias for Alembic compatibility
    @property
    def DATABASE_URL(self) -> str:
        return self.database_url
    
    # Embedding — "openai" requires OPENAI_API_KEY, "local" uses sentence-transformers
    embedding_provider: str = "local"  # "openai" or "local"
    embedding_model: str = "all-MiniLM-L6-v2"  # Local model (384 dims) or OpenAI model name
    embedding_dimension: int = 384  # 384 for all-MiniLM-L6-v2, 1536 for text-embedding-3-small
    openai_api_key: Optional[str] = None  # Set via OPENAI_API_KEY env var
    
    # LLM Settings (for chat)
    default_model: str = "gpt-4o-mini"  # Used for memory extraction only (via OpenAI)
    fallback_model: str = "gpt-4o-mini"  # Fallback if main model fails
    max_tokens: int = 4096  # Max response tokens
    temperature: float = 0.7  # Response creativity
    
    # Auth
    jwt_secret: str = "toup-dev-secret-change-in-production"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24 * 7  # 1 week
    
    # API
    api_prefix: str = "/api"
    cors_origins: list[str] = [
        "http://localhost:5173", "http://localhost:3000", 
        "http://localhost:80", "http://localhost",
        "https://toup.ai",
    ]
    
    # Memory extraction
    max_memories_per_message: int = 10
    similarity_threshold: float = 0.7
    
    # Chat & Session Settings
    memory_recall_limit: int = 15  # How many memories to recall per message
    auto_extract_memories: bool = True  # Auto-extract memories from conversations
    max_history_messages: int = 20  # Max conversation history to include
    
    # Scheduler (for memory decay/consolidation)
    enable_scheduler: bool = True  # Set to False in multi-worker deployments
    decay_interval_hours: int = 6  # How often to run decay
    consolidation_cron_hour: int = 3  # Hour to run consolidation (3 AM)
    
    # Telegram Bot
    telegram_bot_token: Optional[str] = None  # Set via TELEGRAM_BOT_TOKEN env var
    telegram_allowed_user_ids: list[int] = []  # Restrict to specific Telegram user IDs
    telegram_polling_mode: bool = True  # True=polling, False=webhook
    telegram_user_map: dict[str, str] = {}  # Map Telegram user ID → Toup user ID
    
    # Anthropic Claude (kept for future use)
    anthropic_api_key: Optional[str] = None  # Set via ANTHROPIC_API_KEY env var
    # DEPRECATED: prefer `agent_model`. Kept here so existing .env files in
    # the wild don't fail validation (pydantic-settings rejects unknown env
    # vars). `model_resolver.default_anthropic_model()` reads it only as a
    # last-resort fallback when the primary model is OpenAI; otherwise
    # ignored at runtime. Slated for removal in a future config sweep.
    anthropic_model: str = "claude-opus-4-7"
    anthropic_max_tokens: int = 16000
    
    # LLM Provider Keys (set by platform setup wizard)
    llm_mode: str = "manual"  # "manual" or "bundle"
    google_api_key: Optional[str] = None
    mistral_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    xai_api_key: Optional[str] = None
    deepseek_api_key: Optional[str] = None

    # Agent Runtime
    agent_model: str = "claude-opus-4-7"  # Primary agent model
    agent_fallback_model: str = "gpt-5.5"  # Fallback if primary model fails (cross-provider)
    # Auto-builder Planner/Builder model overrides. None means "share the
    # agent's default model" — both phases use `agent_model` unless an
    # operator deliberately splits them. See model_resolver.app_builder_*_model().
    app_builder_planner_model: Optional[str] = None
    app_builder_builder_model: Optional[str] = None
    agent_max_tokens: int = 16000  # Max output tokens for agent
    agent_max_tool_iterations: int = 40  # Max tool call loops before forcing stop
    agent_workspace_dir: str = "/app/workspace"  # Working directory for file operations
    agent_port: int = 8001  # Host-facing port (set by managed container provisioning)
    brave_api_key: Optional[str] = None  # For web search
    skills_dir: str = "/app/skills"  # External skills directory

    # Day-as-Chat context architecture (feature flag)
    use_day_chat_context: bool = True  # Day-as-Chat: agent loads all sessions/channels for the day as context

    # Document Generation + Web Attachments (Phase 1+2 of doc-delivery feature)
    # When false: generate_* tools are not registered, attachment WS events are not emitted.
    # Frontend has its own localStorage gate (TOUP_DOC_ATTACHMENTS) for the two-pane UI.
    # Default ON as of the doc-delivery rollout — disable via FEATURE_DOC_GENERATION=false.
    feature_doc_generation: bool = True

    # PR 8 of the unified-jobs arc: when True, every Auto Builder
    # job completion (success or failure) writes one Message into
    # the user's current-day Conversation with channel='app_builder'.
    # The Mission Control card's "Built X" terminal step is replaced
    # by a single timeline entry the user sees in Day-as-Chat
    # alongside their other agent surfaces. Flipped ON post-smoke
    # so every user sees the new surface; per-tenant override via
    # the FEATURE_AUTO_BUILDER_CHAT_OUTPUT env var if rollback is
    # needed before a code revert.
    feature_auto_builder_chat_output: bool = True

    # ── Latency flags (TKT-LAT wave 3) ──
    # TKT-LAT-013: trim Chrome-extension page-context readable_content
    # from 8000 chars to 2000 chars per turn. The agent can still pull
    # more on demand via extension_read / browser_action. Default ON
    # because the larger payload was burning ~6–8% of the context window
    # on every sidepanel message regardless of relevance.
    extension_page_context_compact: bool = True
    # TKT-LAT-012: route voice TTS through the streaming variant by
    # default so playback can start as soon as first audio bytes arrive
    # (~200 ms first-byte vs. 500–2 500 ms full-body wait). Defaults to
    # OFF until per-provider streaming is verified across mobile.
    tts_streaming_enabled: bool = False
    # TKT-LAT-015: pin Haiku for the Toup-Code supervisor loop. The
    # supervisor only makes routing decisions (click/type/scroll/done) in
    # ≤800 tokens — Opus/GPT-5.5 is overkill and burns ~$0.20–$1 per
    # active session. When ON, force-pin claude-haiku-4-5-20251001 and
    # ignore the user's `model=None` fallthrough. Defaults to OFF so the
    # original "user OWNS this orchestration" semantics are preserved
    # until product signoff on the model-quality trade.
    toup_code_supervisor_use_haiku: bool = False
    # TKT-LAT-017: defer non-essential agent-boot init (MCP tool cache
    # refresh + platform tunnel start) to background tasks so uvicorn
    # can begin serving the first request ~1–3 s sooner. Both targets
    # already have "errors silently swallowed, retry handles it"
    # semantics (mcp_tools_cache has a 60-s periodic refresh loop;
    # tunnel_client.start() is wrapped in try/except and the first
    # chat request doesn't need it). Default ON because the win is
    # real and the surface is safe; operators can disable by flipping
    # to False if a regression surfaces.
    agent_defer_boot_init: bool = True
    # TKT-LAT-017 (wave 3): defer cron + routine scheduler starts to
    # background tasks. cron_service.start() and routine_runner.start()
    # each load jobs from DB + warm APScheduler — ~200–800 ms each,
    # totaling ~400 ms–1.6 s of boot time that the first chat request
    # doesn't need. Both schedulers fire on a time basis (cron daily
    # 3 am, routines at user tz-local wake), so a 1–2 s startup delay
    # never causes a missed fire. trigger_runner is NOT deferred — it
    # registers webhook handlers + a restart sweep, and an inbound
    # Gmail push arriving during start() would land in undefined
    # state. Default ON for the same reason as agent_defer_boot_init.
    agent_defer_scheduler_init: bool = True
    # TKT-LAT-019: skip portrait + hybrid_search + entity_search +
    # active_tasks for trivial queries (greetings, acknowledgments,
    # simple time/date questions). The 24 548-token "what time is it?"
    # observation is the motivator — these context blocks are useful
    # for substantive turns but waste ~5-15 k tokens + 500-2000 ms of
    # portrait generation on one-word answers. Conservative classifier
    # (services/query_classifier.py:is_trivial_query) favors false
    # negatives. Default ON; operator can disable if a regression
    # surfaces by setting CONTEXT_TRIM_FOR_TRIVIAL_QUERIES=false.
    context_trim_for_trivial_queries: bool = True
    # TKT-LAT-003: when the WS chat proxy can't find an active agent
    # for the user, the current behavior retries 6 × 5 s = 30 s
    # before giving up — visible to the user as a 30-second
    # "Connecting…" spinner. When this flag is ON, the proxy
    # fast-fails on the *first* lookup with WS close code 4503
    # "agent_starting" so the frontend can immediately render a
    # "Waking your agent…" UI and either retry the WS connection or
    # poll a status endpoint. Default OFF until the frontend ships
    # the corresponding handler — flipping ON without that work
    # would just turn a 30-s spinner into a hard error. Observability
    # log line `[PERF] ws_proxy_agent_wait_ms=…` fires regardless of
    # the flag so we can see real-world wait distributions.
    agent_ws_proxy_fast_fail: bool = False
    # Storage backend for generated files. "local" writes to {agent_workspace_dir}/generated/.
    # "s3" is stubbed for a follow-up PR.
    files_storage_backend: str = "local"

    # ── Credit-based billing (docs/credits/design.md) ────
    credit_enforcement_enabled: bool = False
    stripe_price_id_starter: str = ""
    stripe_price_id_builder: str = ""
    stripe_price_id_pro: str = ""
    stripe_price_id_elite: str = ""

    # ── Email verification (F13) + Gmail Workspace SMTP ──
    email_provider: str = "resend"
    resend_api_key: str = ""
    email_from_address: str = "Toup <noreply@toup.ai>"
    email_from_reply_to: str = "support@toup.ai"
    app_public_base_url: str = "https://toup.ai"
    aws_ses_region: str = "us-east-1"
    aws_ses_access_key_id: str = ""
    aws_ses_secret_access_key: str = ""
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_use_tls: bool = True
    require_email_verification_for_credits: bool = False
    email_verification_required_after_iso: str = ""

    # Day recall (recall_day tool + end-of-day archival summaries)
    enable_day_recall: bool = False  # When true, exposes recall_day tool + runs hourly archival job

    # Agent Routines — system-managed scheduled actions (email briefing, etc.).
    # Gates RoutineRunner scheduler registration, API visibility, and Mission
    # Control UI surface. Per-tenant; default off until the canary cohort runs
    # for 48h with <2% failure rate.
    routines_email_briefing_enabled: bool = False
    # Mig 042 — `reminder` kind gate. Default False so a code deploy
    # doesn't immediately activate reminders for every tenant; operators
    # flip via `ROUTINES_REMINDERS_ENABLED=true` env var per tenant once
    # the canary is happy. Reminders are text-only (no LLM, no MCP) and
    # safe to enable broadly, but staged rollout is still the right call.
    routines_reminders_enabled: bool = False

    # Phase C — CronService deprecation switch (2026-05-14, `ecb348a`'s
    # follow-up). The legacy CronJob/CronService system is being replaced
    # by Routines (kind=reminder). Default True keeps current behaviour
    # exactly — CronService starts, loads jobs, fires them. When False
    # CronService.start() no-ops gracefully and the legacy `/cron`
    # Telegram command surfaces a one-line deprecation banner pointing
    # users at `/reminders`. Flip to False ONLY after every active
    # CronJob row for the tenant has `migrated_to_routine_id` set
    # (run mig 043 first) — otherwise those reminders stop firing.
    cron_service_enabled: bool = True

    # Workspace Bootstrap
    workspace_per_user: bool = True  # Create per-user workspace subdirectories
    workspace_create_readme: bool = True  # Create README.md in new workspaces

    # Message Queuing & Debounce (Telegram)
    telegram_debounce_ms: int = 1500  # Debounce delay in ms for rapid messages
    telegram_max_queue: int = 5  # Max messages to queue before forcing flush

    # Heartbeat / Proactive Agent
    heartbeat_enabled: bool = False  # Enable proactive agent heartbeat
    heartbeat_interval_hours: int = 6  # How often to run heartbeat
    heartbeat_prompt: str = (
        "You are running as a proactive heartbeat. Check if there are any "
        "pending reminders, scheduled tasks coming up, or anything you should "
        "proactively tell the user about. If nothing is notable, respond with "
        "exactly: __HEARTBEAT_SKIP__"
    )

    # DM Pairing & Access Control
    telegram_require_pairing: bool = False  # Require /pair before using bot
    telegram_pairing_code: str = ""  # Pairing code users must provide

    # Docker Sandbox
    sandbox_enabled: bool = False  # Route exec through Docker container
    sandbox_image: str = "python:3.12-slim"  # Sandbox container image
    
    # Cross-Encoder Re-ranker (Phase 6)
    enable_reranker: bool = True  # Enable cross-encoder re-ranking after RRF
    cohere_api_key: Optional[str] = None  # Set via COHERE_API_KEY env var
    reranker_model: str = "rerank-v3.5"  # Cohere rerank model
    
    # ── Discord ──────────────────────────────────────────────
    discord_bot_token: Optional[str] = None  # Set via DISCORD_BOT_TOKEN env var
    discord_allowed_guilds: list[str] = []  # Restrict to specific guild IDs
    discord_allowed_users: list[str] = []  # Restrict to specific Discord user IDs

    # ── Slack ────────────────────────────────────────────────
    slack_bot_token: Optional[str] = None  # xoxb-... via SLACK_BOT_TOKEN env var
    slack_app_token: Optional[str] = None  # xapp-... via SLACK_APP_TOKEN env var
    slack_allowed_channels: list[str] = []  # Restrict to specific channel IDs

    # ── WhatsApp ─────────────────────────────────────────────
    # Transport mode. "cloud_api" (Path A / BYOA Meta App) or
    # "qr_link" (Path C / Baileys-style QR pairing via neonize).
    # Empty / unset = no WhatsApp configured for this tenant.
    whatsapp_mode: Optional[str] = None
    # Cloud API (Path A) credentials
    whatsapp_phone_number_id: Optional[str] = None  # via WHATSAPP_PHONE_NUMBER_ID
    whatsapp_access_token: Optional[str] = None  # via WHATSAPP_ACCESS_TOKEN
    whatsapp_verify_token: str = ""  # Webhook verification token
    whatsapp_app_secret: Optional[str] = None  # For payload signature verification
    whatsapp_allowed_numbers: list[str] = []  # Restrict to specific phone numbers
    # QR-link mode (Path C) state
    # The dedicated phone number "B" that holds the linked-device session.
    # Display only; the JID derived from the actual scanned device is what
    # neonize uses internally.
    whatsapp_self_e164: Optional[str] = None
    # Comma-separated E.164 list of senders the agent will respond to.
    # Anything else is silently dropped at the ACL gate. Empty = block all
    # (the secure default — pairs scan-then-allowlist into one explicit step).
    whatsapp_baileys_allowlist: str = ""
    # Pairing snapshot mirrored from the Baileys sidecar. Tracks
    # not_linked / linking / linked / logged_out so /admin/bind can apply
    # the value via runtime_identity.apply_to_settings without warning.
    whatsapp_session_status: Optional[str] = None

    # ── Thinking / Extended Thinking ─────────────────────────
    thinking_budget_default: int = 0  # 0 = disabled, >0 = max thinking tokens
    thinking_model_override: Optional[str] = None  # Model to use when thinking enabled

    # ── Tool Policies ────────────────────────────────────────
    tool_deny_list: list[str] = []  # Tools completely blocked (e.g. ["exec", "write_file"])
    tool_elevated_list: list[str] = ["exec", "apply_patch", "process"]  # Tools requiring user confirmation
    tool_max_output_chars: int = 60000  # Global tool output truncation limit
    tool_timeout_default: int = 30  # Default per-tool timeout in seconds
    tool_timeout_overrides: dict[str, int] = {  # Per-tool timeout overrides
        "exec": 120, "web_fetch": 60, "web_search": 30,
        "browser": 120, "spawn": 300, "process": 300,
    }

    # ── DM / Group Policy ────────────────────────────────────
    # dm_policy: pairing | allowlist | open | disabled
    dm_policy: str = "allowlist"  # Default: only allowed user IDs
    # group_policy: open | allowlist | disabled
    group_policy: str = "open"  # Default: respond in any group where added
    group_require_mention: bool = True  # Require @mention in groups


    # ── TTS Auto-mode ────────────────────────────────────────
    tts_auto_mode: str = "off"  # off | always | inbound | tagged
    tts_default_voice: str = "alloy"  # OpenAI TTS voice
    tts_model: str = "gpt-4o-mini-tts"  # TTS model
    tts_speed: float = 1.0  # TTS speed multiplier

    # ── Config Hot-Reload ────────────────────────────────────
    config_reload_enabled: bool = True  # Allow hot-reload from /config

    # ── Agent Lanes ──────────────────────────────────────────
    # Lanes: main, subagent, cron, hook — separate execution contexts
    lane_max_concurrent: int = 5  # Max concurrent agent runs across all lanes
    lane_cron_model: Optional[str] = None  # Override model for cron lane
    lane_hook_model: Optional[str] = None  # Override model for hook lane

    # ── Sub-agent spawning ───────────────────────────────────
    # Phase 1 of the sub-agent spawning arc. Off by default; operator
    # flips per-environment after the smoke matrix in
    # docs/runbooks/subagent-rollout.md passes. While off, the spawn
    # tool returns SUBAGENT_DISABLED to the LLM without creating any
    # rows.
    subagent_spawning_enabled: bool = False
    # Depth = 1 in v1: no grandchildren. The spawn dispatcher walks
    # parent_job_id and rejects with SUBAGENT_DEPTH_EXCEEDED past this
    # value. Bump to 2-3 after telemetry shows v1 is healthy.
    subagent_max_depth: int = 1
    # Per-parent live-children cap (count_running_children > this
    # rejects with SUBAGENT_PARENT_CAP). Stops a chatty supervisor
    # from fanning out indefinitely on one turn.
    subagent_max_children_per_parent: int = 5
    # Per-user concurrent sub-agent cap. With the global
    # `lane_max_concurrent=5` semaphore, this also implicitly bounds
    # the SUBAGENT lane share so the user's foreground MAIN run never
    # starves.
    subagent_max_per_user_concurrent: int = 3
    # Rolling 24h ceiling on sub-agent spawns per user. Backstop
    # against runaway loops the per-parent + concurrent caps miss
    # (e.g. a user prompt that legitimately spawns 5 in series, 10x).
    subagent_max_per_user_24h: int = 50
    # Cost knob — multiplier the credit hook applies to LLM spend
    # inside a sub-agent run. 1.0 = same rate as the parent.
    # Crank up to dampen sub-agent abuse without disabling outright.
    subagent_credit_multiplier: float = 1.0
    # Default wall-clock budget for a sub-agent run. The spawn tool
    # accepts an override on the LLM call clamped to
    # `subagent_max_timeout_seconds`.
    subagent_default_timeout_seconds: int = 300
    subagent_max_timeout_seconds: int = 900
    # Threshold beyond which a `status=running` sub-agent BuildJob
    # row is considered orphaned by a platform restart and gets swept
    # to `failed`. See app/agent/subagent_dispatcher.py orphan sweep
    # (mirrors triggers/runner.py:177 pattern).
    subagent_orphan_sweep_threshold_minutes: int = 10

    # ── Enhanced Telegram ────────────────────────────────────
    telegram_forum_support: bool = True  # Support Telegram forum topics
    telegram_topic_routing: bool = True  # Route by topic thread_id
    telegram_reactions_enabled: bool = True  # Send reactions (e.g., 👍)
    telegram_inline_buttons: bool = True  # Allow inline keyboard buttons
    telegram_polls_enabled: bool = True  # Allow sending polls

    # ── Moderation ───────────────────────────────────────────
    moderation_enabled: bool = False  # Enable moderation tools
    moderation_log_channel: Optional[str] = None  # Channel to log mod actions

    # ── TTS Providers ────────────────────────────────────────
    tts_provider: str = "openai"  # openai | elevenlabs | edge
    elevenlabs_api_key: Optional[str] = None  # Set via ELEVENLABS_API_KEY
    elevenlabs_model: str = "eleven_multilingual_v2"
    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"  # Rachel
    tts_per_user_prefs: bool = True  # Save per-user TTS preferences

    # ── Platform / Agent Architecture ──────────────────────────
    platform_api_url: str = "https://toup.ai/api"   # Agent calls this to reach the platform
    agent_api_url: str = ""                          # Platform calls this to proxy chat to user's Agent VPS
    supabase_url: str = ""                           # Supabase project URL (for edge-case direct access)
    supabase_anon_key: str = ""                      # Supabase anon key
    run_mode: str = "monolith"                       # "monolith" | "platform" | "agent"
    agent_api_key: str = ""                          # API key for authenticating requests to this Agent VPS
    user_id: str = ""                                # Owner user ID (set on Agent VPS via cloud-init)
    toup_token: str = ""                             # Connect token from toup.ai dashboard (for tunnel auth)
    # Phase A — Pool of pre-booted agent containers. When True,
    # `auth.register` calls `pool_service.claim_for_user` instead of
    # the slow `schedule_prewarm` path (~1-2s pool bind vs ~30-90s
    # cold container build). Pool requires the bridge's pool endpoints
    # to be installed (verified live on Contabo since 2026-04-30).
    #
    # Default flipped to True 2026-05-08 after the WhatsApp QR onboarding
    # incident: USE_CONTAINER_POOL was unset in Railway env, so every
    # signup fell through to schedule_prewarm. Users reached the Channel
    # step before the slow path finished and saw "Agent took too long to
    # start." Setting the default to True makes the fast path the
    # production-correct behavior; self-hosters without pool endpoints
    # can still set USE_CONTAINER_POOL=false explicitly to opt out.
    use_container_pool: bool = True
    # Phase B — Blue-green tenant rollouts. When True,
    # `docker_host_service.upgrade_tenant_image` calls the bridge's
    # `/v1/tenants/<prefix>/blue-green-upgrade` (zero-WS-drop) instead
    # of the legacy `/v1/tenants/<prefix>/upgrade` (recreate). Requires
    # the bridge's `blue_green.py` module installed (see
    # `bridge/blue_green.py` + `INSTALL.md`).
    #
    # Default flipped to True 2026-05-13: the legacy recreate flow was
    # closing every active sidepanel/web WS with code 1001 on every CI
    # push, and was the actual source of the "Reconnecting…" cycles the
    # user kept seeing during deploy bursts. Blue-green has been live
    # on the bridge since 2026-05-10 and validated at 81/81 health
    # probes during a 62-s image swap on tenant `871bac24` — production-
    # ready, just gated behind a flag that was never flipped.
    use_blue_green_rollouts: bool = True
    # Drain timeout for blue-green cutovers. After Caddy flips to the
    # new slot, the old slot rejects new WS but lets in-flight stream
    # for up to this many seconds before the bridge force-stops it.
    # 60s is the plan default — enough for any reasonable LLM stream
    # while bounding the rollout pipeline's per-tenant time.
    blue_green_drain_timeout_s: int = 60

    # ── MCP transport auth (T0c) ─────────────────────────────
    # When False (default), the platform MCP server logs unauthenticated
    # requests as warnings but lets them through. When True, missing or
    # invalid X-Agent-Key returns 401 from the transport before any tool
    # dispatch. Flip to True only after a clean staging soak shows zero
    # warn-only rejections (per docs/integrations/02-rollout.md T0c).
    mcp_require_x_agent_key: bool = False

    # ── Agent-key rotation primitive (T0b) ──────────────────
    # When False (default), POST /api/agent/rotate-key returns 503. Flip
    # to True after the smoke matrix in docs/integrations/02-rollout.md
    # T0b passes (rotate while WS active → clean disconnect; bridge fail
    # → DB rollback; replay token rejected; etc.).
    enable_agent_key_rotation: bool = False
    # Total time we'll spend polling the new container's /agent/health
    # with the new X-Agent-Key after rotation, before declaring failure
    # and rolling back. 12s is empirically enough for a bridge recreate
    # (~2-5s) plus FastAPI lifespan boot (~3-5s) on Contabo.
    agent_key_rotation_verify_timeout_s: int = 12
    agent_key_rotation_verify_retry_interval_s: float = 2.0

    # ── Connector OAuth (T1d) ────────────────────────────────
    # Default ON since the T1d staging soak passed and the connector
    # arc shipped. Set ENABLE_CONNECTOR_OAUTH=false to disable as a
    # kill switch (returns 503 on /api/oauth/connect/<id> and
    # /api/oauth/disconnect/<id>; /api/oauth/callback always responds).
    enable_connector_oauth: bool = True

    # ── Connector dispatch (T1g — closes the agent→platform loop) ──
    # Default ON: connector tool names merge into the agent's tool
    # list, the LLM can emit tool_use for them, and the platform
    # dispatcher routes the call. Set USE_CONNECTOR_DISPATCH=false
    # to mute connectors from the agent without breaking the OAuth
    # connect flow.
    use_connector_dispatch: bool = True

    # T3a — Google OAuth client. One client backs Gmail + Calendar +
    # Drive (architecture §6.1). Without both set the provider app
    # silently doesn't register and Google connectors are skipped at
    # registry-load time. Required for the Google verification path
    # in `docs/integrations/google-verification.md`.
    google_oauth_client_id: str = ""
    google_oauth_client_secret: str = ""

    # ── Gmail Pub/Sub triggers (Gate T1) ─────────────────────
    # GCP project that owns the Pub/Sub topic. The fully-qualified
    # topic name is `projects/{gcp_project}/topics/{pubsub_topic}` —
    # we assemble it from these two so the operator can rotate either
    # piece without surgery on the config.
    # Empty when the trigger system isn't provisioned yet; the
    # platform webhook returns 503 in that state (fail-closed —
    # never silently accepts an unsigned push).
    gcp_project: str = ""
    pubsub_topic: str = ""
    # The `aud` claim Google embeds in the push-subscription JWT.
    # In the standard setup this is the full webhook URL
    # (https://toup.ai/api/v1/webhooks/gmail). The webhook rejects
    # any push whose `aud` doesn't match — without this, anyone who
    # can forge a JWT signed by Google's keys for any audience could
    # reach our endpoint.
    pubsub_push_audience: str = ""
    # Service-account email Google signs the push JWT as. Match
    # against the `email` claim. Leave empty during dev to skip the
    # signer check (relies on JWT signature alone) — flip on for
    # prod.
    pubsub_push_signer: str = ""
    # Per-tenant kill switch for the email_received trigger kind.
    # Set false to silently drop incoming Pub/Sub pushes (the
    # webhook still 200s — Pub/Sub stops retrying — but no agent
    # dispatch happens). Mirrors the routine flag pattern.
    triggers_email_enabled: bool = True

    # T4a — GitHub OAuth client. PKCE off (provider doesn't support
    # it on the standard OAuth app type — see provider_apps.py).
    github_oauth_client_id: str = ""
    github_oauth_client_secret: str = ""

    # Microsoft Identity Platform v2 — backs the Outlook connector
    # (and future Microsoft 365 surfaces: Teams, OneDrive, Calendar).
    # Register an "App registration" in https://portal.azure.com →
    # Azure Active Directory → App registrations. PKCE supported.
    # Without both set the Microsoft provider app silently doesn't
    # register and Microsoft connectors are skipped at registry-load
    # time — same gate pattern as Google + GitHub above.
    microsoft_oauth_client_id: str = ""
    microsoft_oauth_client_secret: str = ""
    # `common` works for both personal Microsoft accounts and Azure AD
    # (work/school) tenants. Set to a specific tenant id to lock
    # consent to a single org. See
    # https://learn.microsoft.com/en-us/azure/active-directory/develop/v2-protocols-oidc.
    microsoft_oauth_tenant: str = "common"

    # LinkedIn OAuth 2.0 — backs the LinkedIn connector. Register an
    # app at https://www.linkedin.com/developers/apps and enable the
    # "Sign In with LinkedIn using OpenID Connect" + "Share on
    # LinkedIn" products. PKCE NOT supported by LinkedIn (yet).
    linkedin_oauth_client_id: str = ""
    linkedin_oauth_client_secret: str = ""

    # HMAC-SHA256 signing key for the OAuth state token (architecture D2).
    # Independent from jwt_secret so rotating it doesn't log out users.
    # Generate via `secrets.token_urlsafe(32)`. Required when
    # enable_connector_oauth=True; assert at platform lifespan.
    oauth_state_secret: str = ""
    # State token TTL — 10 minutes per architecture §3.1. Long enough to
    # walk through Google's consent screen unhurried, short enough that
    # a captured state is rapidly unredeemable.
    oauth_state_ttl_seconds: int = 600
    # Where Google/GitHub/etc. redirect the browser after consent.
    # MUST exact-match the redirect_uri registered in each provider's
    # OAuth-app config (per docs/integrations/google-verification.md §3.2).
    # Override in tests / non-default deploys via OAUTH_CALLBACK_URL env.
    # Default uses the canonical platform host `toup.ai` (no `app.`
    # prefix) — matches how the frontend is currently served.
    oauth_callback_url: str = "https://toup.ai/api/oauth/callback"
    # Sign-in (NOT connector) Google OAuth callback. Distinct path from
    # the connector flow so the sign-in handler can have its own
    # state-validation rules. MUST be added to the same Google Cloud
    # Console project's Authorized Redirect URIs. Override per deploy
    # via GOOGLE_AUTH_CALLBACK_URL when the API is fronted on a non-
    # default host.
    google_auth_callback_url: str = ""

    # ── VPS Provisioning (AWS + Stripe) ──────────────────────
    aws_access_key_id: Optional[str] = None        # Set via AWS_ACCESS_KEY_ID
    aws_secret_access_key: Optional[str] = None    # Set via AWS_SECRET_ACCESS_KEY
    aws_region: str = "us-east-1"
    aws_ami_id: str = ""                           # Custom AMI with platform pre-installed
    aws_key_pair_name: str = ""                    # EC2 key pair name
    aws_security_group_id: str = ""                # Security group allowing SSH + HTTP(S)
    stripe_secret_key: Optional[str] = None        # Set via STRIPE_SECRET_KEY
    stripe_publishable_key: str = ""               # Set via STRIPE_PUBLISHABLE_KEY (public, safe for frontend)
    stripe_webhook_secret: Optional[str] = None    # Set via STRIPE_WEBHOOK_SECRET
    stripe_starter_price_id: str = ""              # Stripe Price ID for Starter plan
    stripe_standard_price_id: str = ""             # Stripe Price ID for Standard plan
    stripe_pro_price_id: str = ""                  # Stripe Price ID for Pro plan
    stripe_llm_bundle_price_id: str = ""           # Stripe Price ID for $30/mo LLM bundle (USD, default)
    # Optional regional Price IDs. When set and the request's country
    # matches, the corresponding Price is used for /billing/prices and
    # subscription creation. Falls back to the USD price above when no
    # regional override is configured.
    stripe_llm_bundle_price_id_cad: str = ""       # Stripe Price ID for $40 CAD/mo LLM bundle (Canadian users)
    stripe_supabase_price_id: str = ""             # Stripe Price ID for $5/mo managed Supabase

    # ── OpenAI Admin API (per-user project + key auto-provisioning) ──
    # Optional. When set, bundle activation auto-creates an OpenAI project
    # + service-account key per user and stores it in
    # AgentConfig.bundle_openai_{project_id,api_key}. The LLM proxy uses
    # that per-user key on outbound (β architecture: proxy-mediated, per-
    # user billing). When unset, bundle OpenAI calls fall back to the
    # platform_openai_api_key master key.
    openai_admin_api_key: Optional[str] = None     # Set via OPENAI_ADMIN_API_KEY (sk-admin-...)

    vps_provisioning_enabled: bool = False         # Gate: set True once AWS creds are configured
    ssh_lambda_function_name: str = "toup-ssh-proxy"  # Lambda function for SSH proxy

    # ── Hostinger VPS ─────────────────────────────────────────────
    hostinger_api_token: Optional[str] = None     # Set via HOSTINGER_API_TOKEN
    hostinger_data_center_id: int = 1             # Hostinger data center (1=US, etc.)
    hostinger_template_id: int = 1               # OS template ID (Ubuntu 24.04)
    hostinger_payment_method_id: str = ""         # Billing payment method ID
    hostinger_enabled: bool = False               # Gate: set True once Hostinger is configured

    # ── Hetzner Cloud VPS ─────────────────────────────────────────
    hetzner_api_token: Optional[str] = None       # Set via HETZNER_API_TOKEN
    hetzner_location: str = "ash"                 # Hetzner location (ash=Ashburn US, fsn1=Falkenstein DE)
    hetzner_ssh_key_id: str = ""                  # Hetzner SSH key ID (optional, for backup access)
    hetzner_enabled: bool = False                 # Gate: set True once Hetzner is configured

    # ── Managed Docker Host (containerized multi-tenant) ──────────
    #
    # Phase 3 cutover: the platform no longer SSHes to the Docker host. All
    # tenant lifecycle goes through a typed FastAPI provisioning bridge
    # (see new-vps/08-provisioning-bridge.sh + docs/new-vps/14-AUTOMATED-DEPLOYMENT-DESIGN.md).
    #
    # Legacy SSH-related vars (docker_host_url, docker_host_ssh_key,
    # docker_host_ssh_password, docker_host_pg_url) are gone. Anything still
    # reading them is a bug.
    docker_host_ip: str = ""                # Public IP of Docker host — used only by legacy callers being phased out
    # Sentinel only — fresh-install hard fallback. Production resolves the
    # real SHA via _latest_known_good_image_tag() in docker_host_service.py.
    # Do NOT publish this tag to GHCR; if it ever reaches the bridge, that's
    # a fresh-install-with-no-rollouts edge case that should be visible.
    docker_agent_image: str = "toup-agent:latest"
    docker_port_range_start: int = 9000     # Kept for the bridge's port allocation
    docker_port_range_end: int = 9999
    managed_hosting_enabled: bool = False   # Gate: set True once bridge is reachable
    # Container reaper grace period — how many days a managed tenant
    # container keeps running after its bundle is cancelled AND the
    # user account is deactivated, before the nightly reaper destroys
    # it. Phase C of the never-sleep plan bumped this from 7 to 30 and
    # added the `User.is_active=False` AND-gate (see
    # scheduled_tasks.run_managed_container_reaper). Active-but-cancelled
    # users keep their container indefinitely so they don't hit a cold
    # start on return. Set to 0 in dev to disable the reaper entirely.
    bundle_cancel_grace_days: int = 30
    # Phase-1 prewarm-on-Soul.save feature flag. When True, the Soul.save
    # handler fires `provision_container` in the background after commit so
    # the user's managed container is fully booted by the time they reach
    # Install. When False (default during ST-3 cutover bake, dial true
    # 2026-05-09+), the existing OnboardingShell Welcome-mount prewarm
    # remains the only trigger. See docs/onboarding/prewarm-phase0.md.
    prewarm_on_soul_save: bool = False
    # Abandoned-onboarding reaper grace window. Containers tied to users
    # who haven't completed onboarding AND have no day_chats AND aren't
    # paying get destroyed after this many days. The agent_configs row
    # is preserved so a returning user can re-onboard cleanly. Set to 0
    # to disable.
    abandoned_onboarding_grace_days: int = 14
    # Cloudflare Turnstile secret for signup verification (Phase 3).
    # When set, /auth/register requires a valid Turnstile token. When
    # blank, verification is skipped (dev / test path).
    turnstile_secret_key: str = ""
    # Signup rate limit — IP-keyed. 5 signups per hour per IP, copied
    # from the login rate limiter pattern. Disabled if either value is 0.
    signup_rate_limit_per_hour: int = 5
    admin_alert_telegram_token: str = ""    # Bot token for admin alerts (shared / user-facing bot)
    admin_alert_telegram_chat_id: str = ""

    # ── Provisioning bridge (Phase 3, replaces SSH-as-root) ───────
    # mTLS-secured FastAPI service on the VPS. Platform talks to it with a
    # client cert; bridge does all docker / PG lifecycle work.
    bridge_url: str = ""                    # e.g. https://bridge.agents.toup.ai
    bridge_ca_cert: str = ""                # CA cert (PEM) that issued bridge's server cert
    bridge_client_cert: str = ""            # Client cert (PEM) platform presents for mTLS
    bridge_client_key: str = ""             # Client private key (PEM)
    bridge_request_timeout_s: int = 30      # default httpx timeout for non-upgrade calls
    bridge_upgrade_timeout_s: int = 180     # upgrade endpoint can take longer (pull + recreate + health)

    # ── Audio stream proxy (Phase 1, /api/media/{id}/audio_stream) ─
    # Per-tenant concurrent stream cap, enforced PER-REPLICA via an
    # in-process semaphore. Effective global cap is N × replicas until
    # Redis lands. Tune via AUDIO_STREAM_MAX_CONCURRENT_PER_TENANT.
    # Phase 2 will allow per-tenant override via agent_configs.
    audio_stream_max_concurrent_per_tenant: int = 5
    # Optional base64-encoded Netscape cookies.txt for yt-dlp. When set,
    # the cookie-boot helper decodes it to /tmp/yt-cookies.txt and points
    # YT_DLP_COOKIES_PATH at it. Leave unset until extraction hit-rate
    # degrades; provisioning a dedicated YouTube account is documented in
    # docs/skills/radio-mode/proxy-handoff-design.md §3.3.
    yt_dlp_cookies_b64: Optional[str] = None

    # ── Platform fixed costs (admin ROI dashboard) ───────────────
    # Recurring monthly costs that don't scale with user count: subscriptions
    # we pay regardless of usage, plus infrastructure. These flow into the
    # admin Revenue & Usage tab as a "Fixed costs" row in the platform P&L
    # so net margin reflects reality, not just LLM cost.
    # All values are USD/month. Override per-environment via env vars.
    platform_cost_anthropic_max_monthly_usd: float = 158.20  # Claude Max subscription
    platform_cost_openai_monthly_usd: float = 0.0            # Any fixed OpenAI plan (Team etc.)
    platform_cost_vps_monthly_usd: float = 0.0               # Contabo / DigitalOcean / etc.
    platform_cost_railway_monthly_usd: float = 5.0           # Railway hobby plan baseline
    platform_cost_other_monthly_usd: float = 0.0             # Domain, Cloudflare paid tier, misc.
    # Optional admin-API keys for pulling actual provider bills (vs proxy
    # estimates). Leave blank to fall back to llm_proxy_events sums.
    openai_admin_key: str = ""                               # OpenAI org admin key (sk-org-admin-...)
    anthropic_admin_key: str = ""                            # Anthropic admin key for org cost API

    # ── Rollout pipeline (Phase 3) ────────────────────────────────
    rollout_secret: str = ""                # Shared secret CI → platform webhook (X-Rollout-Secret)
    # Hard cap (NOT target) on canary observation duration. The observe
    # loop is signal-based: it exits as soon as boot gate + stability hold
    # pass (~90s on healthy code). This setting is the upper bound — set
    # higher for high-risk migrations to extend the safety budget. 5 min
    # comfortably covers the ~90s happy path with margin for slow boots.
    rollout_canary_wait_minutes_default: int = 5
    rollout_batch_size: int = 5             # Post-canary parallel upgrades per batch
    infra_alert_telegram_token: str = ""    # Dedicated infra bot (split from admin_alert_*)
    infra_alert_telegram_chat_id: str = ""

    # ── Browser Proxy & Captcha ──
    browser_proxy: str = ""  # e.g. "http://user:pass@proxy.example.com:8080"
    captcha_solver_enabled: bool = True  # Auto-solve captchas on navigation

    # ── Agent Spawn Policies ─────────────────────────────────
    allow_agents: list[str] = []  # If non-empty, only these agent IDs can be spawned

    # ── Custom Model Providers ───────────────────────────────
    # Map model prefix to base URL: {"ollama": "http://localhost:11434/v1", "groq": "https://..."}
    custom_model_providers: dict[str, str] = {}
    # Map model name to provider: {"llama3": "ollama", "mixtral": "groq"}
    custom_model_map: dict[str, str] = {}

    # ── Multi-Agent Routing ──────────────────────────────────
    multi_agent_enabled: bool = False  # Enable persona-based routing
    multi_agent_default: str = "default"  # Default persona name

    # ── Platform LLM Proxy (bundle mode) ───────────────────
    platform_anthropic_api_key: Optional[str] = None  # Platform-owned Anthropic key
    platform_openai_api_key: Optional[str] = None     # Platform-owned OpenAI key
    platform_encryption_key: str = ""                  # Fernet key (primary, used for encrypt + decrypt)
    platform_encryption_key_previous: str = ""          # Optional comma-separated list of prior Fernet keys — decrypt-only, used during rotation

    # Bundle budget configuration (all values in cents)
    bundle_total_budget_cents: int = 4000              # $40/month total
    bundle_anthropic_budget_cents: int = 3000           # $30/month Anthropic allocation
    bundle_openai_budget_cents: int = 1000              # $10/month OpenAI allocation
    bundle_anthropic_daily_cap_cents: int = 100          # $1/day Anthropic soft cap (triggers fallback)

    # Pricing per 1K tokens (USD)
    pricing_per_1k: dict[str, dict[str, float]] = {
        "gpt-5.5": {"input": 0.005, "output": 0.030},
        "gpt-5.4": {"input": 0.003, "output": 0.012},
        "gpt-5": {"input": 0.003, "output": 0.012},
        "gpt-4.1": {"input": 0.002, "output": 0.008},
        "gpt-4o": {"input": 0.0025, "output": 0.01},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "claude-opus-4-6": {"input": 0.015, "output": 0.075},
        "claude-sonnet-4-6": {"input": 0.003, "output": 0.015},
        "claude-sonnet-4-5-20250514": {"input": 0.003, "output": 0.015},
        "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
    }

    # Strip whitespace from API key fields on load (users often paste with spaces)
    _KEY_FIELDS = {
        "openai_api_key", "anthropic_api_key", "google_api_key",
        "mistral_api_key", "groq_api_key", "xai_api_key", "deepseek_api_key",
        "telegram_bot_token", "discord_bot_token", "slack_bot_token",
        "slack_app_token", "brave_api_key", "elevenlabs_api_key",
        "cohere_api_key", "agent_api_key", "toup_token",
        "stripe_secret_key", "stripe_webhook_secret",
    }

    @model_validator(mode="after")
    def _strip_api_keys(self) -> "Settings":
        for field in self._KEY_FIELDS:
            val = getattr(self, field, None)
            if isinstance(val, str):
                stripped = val.strip()
                if stripped != val:
                    object.__setattr__(self, field, stripped)
        return self

    @model_validator(mode="after")
    def _validate_stripe_environment_match(self) -> "Settings":
        # Refuse to boot when the configured Stripe key disagrees with the
        # declared deployment environment. Guards against the two catastrophic
        # mistakes: (a) sk_live_ in non-prod (charges real cards from a dev
        # build), (b) sk_test_ in prod (real users get test charges that
        # don't actually settle). Empty / unset key allows boot — Stripe is
        # optional infrastructure, only the prefix mismatch is a fatal misconfig.
        key = (self.stripe_secret_key or "").strip()
        if not key:
            return self
        is_live = key.startswith("sk_live_")
        if self.environment == "production" and not is_live:
            raise ValueError(
                "Refusing to boot: ENVIRONMENT=production requires "
                "STRIPE_SECRET_KEY to start with 'sk_live_'. "
                f"Got prefix '{key[:8]}...'. "
                "Either set ENVIRONMENT to a non-prod value or use a live key."
            )
        if self.environment != "production" and is_live:
            raise ValueError(
                f"Refusing to boot: ENVIRONMENT={self.environment!r} forbids "
                "live STRIPE_SECRET_KEY (prefix 'sk_live_'). "
                "Use 'sk_test_...' for non-prod environments."
            )
        return self

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
