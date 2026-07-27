from pydantic_settings import BaseSettings
from pydantic import model_validator
from functools import lru_cache
from typing import List, Optional


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
    # App-preview iframe token: short-lived + app-scoped (round 12). It rides in
    # an agent-authored, same-origin preview page, so it must never be a full
    # account credential — keep the TTL tight to bound a leak.
    preview_token_expire_minutes: int = 60 * 6  # 6 hours

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
    # Audit A6-1 (2026-07-23): the memory maintenance jobs (decay,
    # consolidation, end-of-day archival, retrieval feedback) are started
    # only by platform_main, whose DB excludes the AGENT_ONLY memories/
    # day_chats tables — they have never run against tenant data. When
    # True, agent_main's lifespan registers the same job entry points on
    # the tenant container's scheduler where those tables actually live.
    agent_memory_maintenance_enabled: bool = False

    # Proactive-notification dispatcher (Autopilot arc PR3). Runs on
    # EVERY replica — safety comes from per-row status CAS, not from
    # this flag. The flag is the ops kill switch.
    notification_dispatch_enabled: bool = True
    notification_dispatch_interval_seconds: int = 30
    notification_receipt_delay_minutes: int = 15  # Expo says ~15 min
    notification_max_attempts: int = 5
    # Progress fast lane (2026-07-16): dispatch event_kind='progress'
    # rows inline at ingest (CAS-claimed, replica-safe) instead of
    # waiting up to notification_dispatch_interval_seconds — interim
    # Live Activity bar motion is worthless 30s late. Kill switch only;
    # on failure the row stays queued for the normal loop.
    notification_progress_fastlane_enabled: bool = True

    # APNs (direct, token-based auth) — drives iOS Live Activities for
    # Autopilot missions (push-to-start + progress updates + end). All
    # four must be set or Live Activity sends silently no-op (the rest
    # of the notification pipeline is unaffected). The .p8 signing key
    # is passed base64-encoded so it fits in a single env var.
    apns_key_b64: Optional[str] = None      # base64(AuthKey_XXXX.p8 contents)
    apns_key_id: Optional[str] = None       # 10-char key id from the portal
    apns_team_id: Optional[str] = None      # Apple team id (5W2R26Z4H7)
    apns_bundle_id: str = "ai.toup.app"     # LA topic = <bundle>.push-type.liveactivity
    live_activity_enabled: bool = True      # ops kill switch for the LA lane
    # Reminder countdown card (2026-07-17): arm an on-device Live
    # Activity timer at reminder creation for near-term one-shots.
    # Producer-side kill switch, independent of the LA lane switch above.
    reminder_countdown_live_activity_enabled: bool = True
    
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
    # Default to an OpenAI model: bundle users authenticate Anthropic via the
    # platform's shared Claude account, and when that account runs out of
    # credit EVERY bundle user calling Claude gets a hard 400 ("credit balance
    # too low") — a single-point-of-failure outage (2026-05-29: new users
    # couldn't even say "hi"). OpenAI calls instead use each user's OWN
    # per-tenant project key (provisioned at signup, isolated billing), so an
    # OpenAI default has no shared-account dependency. Anthropic models remain
    # fully available as an explicit per-user `agent_config.agent_model`
    # choice once the platform Claude account is funded.
    agent_model: str = "gpt-5.5"  # Primary agent model (OpenAI — per-tenant key)
    agent_fallback_model: str = "gpt-4o"  # Distinct OpenAI fallback if primary fails
    # analyze_image tool (vision Q&A on a URL/workspace image). Routed through
    # the bundle LLM proxy in bundle mode so it is metered + governed like every
    # other LLM call; gpt-4o matches the previously hardcoded model. Override
    # via ANALYZE_IMAGE_MODEL for a cheaper default (e.g. gpt-4o-mini).
    analyze_image_model: str = "gpt-4o"
    # Anthropic provider master switch. DEACTIVATED platform-wide on
    # 2026-05-29: bundle Anthropic calls share ONE platform Claude account
    # and it ran out of credit, hard-400ing every bundle user's Claude call
    # ("credit balance too low"). Until that account is funded, we run
    # OpenAI-only (each user has their own isolated per-tenant OpenAI key).
    # When False: the model router never picks Claude, /api/models hides
    # Anthropic models, and the LLM proxy rejects Claude requests as a hard
    # backstop. Re-enable by setting env ANTHROPIC_ENABLED=true once the
    # platform Claude account has credit again. See model_router +
    # llm_proxy._route_chat for the enforcement points.
    anthropic_enabled: bool = False
    # Auto-builder Planner/Builder model overrides. None means "share the
    # agent's default model" — both phases use `agent_model` unless an
    # operator deliberately splits them. See model_resolver.app_builder_*_model().
    app_builder_planner_model: Optional[str] = None
    app_builder_builder_model: Optional[str] = None
    agent_max_tokens: int = 16000  # Max output tokens for agent
    agent_max_tool_iterations: int = 40  # Max tool call loops before forcing stop
    # Max idempotent read-only tools (web_search/web_fetch/extension_*) run
    # concurrently within a single assistant turn. Bounds outbound load so a
    # multi-fetch turn finishes in ~max(individual) latency, not sum.
    agent_parallel_tool_cap: int = 6
    agent_workspace_dir: str = "/app/workspace"  # Working directory for file operations
    agent_port: int = 8001  # Host-facing port (set by managed container provisioning)
    brave_api_key: Optional[str] = None  # For web search
    # Kill-switch (default on): web_search races the primary engines
    # (Whoogle/DuckDuckGo/Bing) concurrently over a shared client and takes the
    # first non-empty result, so a dead or slow backend can't block the chain
    # (Mojeek stays a last resort). Flip off to fall back to the legacy
    # sequential priority chain.
    search_engine_race: bool = True
    # Kill-switch (default on): the server-side research fallback reads the
    # top-N result pages concurrently (bounded by research_read_concurrency)
    # instead of one-at-a-time, so a depth-N research call costs ~max(page)
    # latency, not the sum. Flip off for the legacy sequential reads — the
    # output bytes are identical either way.
    research_parallel_reads: bool = True
    research_read_concurrency: int = 6
    # Per-tenant (one process per container) TTL+LRU caches for web search and
    # page fetch — a repeat query/URL within the TTL returns with zero network.
    # Kill-switches (default on); empties/no-results are never cached.
    search_cache_enabled: bool = True
    search_cache_ttl_s: int = 420          # 7 min
    search_cache_max: int = 256
    fetch_cache_enabled: bool = True
    fetch_cache_ttl_s: int = 720           # 12 min
    fetch_cache_max: int = 256
    # Kill-switch (default on): dedup near-duplicate URLs, drop empty results,
    # and BM25-rerank web_search results by relevance before the model sees them.
    # Pure-Python (no LLM/network), so no latency cost. Off → raw engine order.
    search_rerank_enabled: bool = True
    # Kill-switch (default on): truncate web_search/web_fetch output ONCE to a
    # token budget (cheap ~4 chars/token estimate) instead of the legacy double
    # char+byte caps, and cap the AGGREGATE tokens across a parallel web batch so
    # a multi-fetch turn can't flood the context. Off → legacy byte/char caps.
    web_token_budget_enabled: bool = True
    fetch_token_budget: int = 4000          # tokens per single web_fetch
    search_token_budget: int = 2000         # tokens per web_search result block
    web_turn_token_budget: int = 12000      # aggregate cap across one parallel web batch
    # Kill-switches (default on): the no-API-key headless-browser fallbacks.
    # browser_search → when the fast httpx engines return "No results", race
    # Brave Search (search.brave.com, browser-rendered, no key) for real results.
    # browser_fetch → when httpx+trafilatura returns empty (JS-rendered page,
    # 403, or timeout), render the page in the headless browser. Both rely on the
    # Chromium binary mounted read-only from the host at /root/.cache/ms-playwright
    # (host dir /opt/toup/playwright, populated by `patchright install chromium`).
    # Off → degrade to the httpx-only paths.
    browser_search_enabled: bool = True
    browser_fetch_enabled: bool = True
    skills_dir: str = "/app/skills"  # External skills directory

    # Day-as-Chat context architecture (feature flag)
    use_day_chat_context: bool = True  # Day-as-Chat: agent loads all sessions/channels for the day as context

    # Prefix-stable prompt layout (token-efficiency PR-1; audit findings
    # F-1/F-2/F-3 in docs/audits/2026-07-token-efficiency.md — measured 0%
    # OpenAI prompt-cache hit rate in prod). When true:
    #   - the wire tools array is fixed per run (channel strips only);
    #     intent gating becomes a tool_choice allowed_tools restriction,
    #     and the mid-run full-toolset escalation stops mutating the array
    #   - the minute clock leaves the system prompt (date stays); the exact
    #     clock, retrieved memories, active tasks, and day-summary blocks
    #     render into ONE per-turn <turn_context> message appended after
    #     history, behind the cacheable prefix
    #   - OpenAI calls also send safety_identifier + prompt_cache_retention
    # Default OFF: model-visible prompt layout change — flip after canary
    # validation (STABLE_PREFIX_LAYOUT=true).
    # Known residual (accepted): the coarse time-of-day word in about_you
    # still flips ~4x/day (morning/afternoon/evening/late-night), so expect
    # scheduled cache-hit dips near 05/12/17/22 local. OpenAI-only benefit:
    # Claude models keep the legacy intent-filtered tools array (Anthropic
    # tool_choice cannot express an allowlist).
    stable_prefix_layout: bool = False

    # Per-tenant CANARY for the stable prefix layout. Agent feature flags are
    # otherwise baked fleet-wide by the bridge's docker-run env — there is no
    # per-container override — so a proper single-tenant canary of
    # stable_prefix_layout is impossible without this. Comma-separated
    # user_ids: the stable layout activates for a turn when the global flag is
    # on OR the turn's user_id is in this list. Set it fleet-wide on the
    # bridge (STABLE_PREFIX_CANARY_USER_IDS=<one-user-id>) to enable the layout
    # for exactly one tenant while every other container stays on the legacy
    # path — measure that tenant's [PERF] cache_read= / cache-daily, then flip
    # the global flag once proven. Empty = nobody (default).
    stable_prefix_canary_user_ids: str = ""

    # Cache-aware overflow rollover (token-efficiency PR-3; audit finding
    # F-6 / A8-5..A8-6 in docs/audits/2026-07-token-efficiency.md). When
    # true, compact_messages:
    #   - keeps the FIRST messages (the cached prompt-prefix head)
    #     byte-untouched and summarizes the MIDDLE of the conversation
    #     instead of rewriting messages[0] with a fresh summary
    #   - generates the summary deterministically (temperature 0.0) and
    #     persists it on the Conversation row
    #     (metadata_json["compaction_summary"], keyed by a hash of the
    #     summarized span) so re-compacting the same span reuses it
    #   - promotes the dropped span to durable memory via the existing
    #     background extractor before it leaves the context window
    # Default OFF: changes the model-visible message layout under
    # compaction — flip via CACHE_AWARE_OVERFLOW=true after canary
    # validation. The unflagged A8-2/A8-3/A8-4 fixes (overflow
    # compact-and-retry, active-model window math, tool-pair-safe cut)
    # are pure bug fixes and do NOT ride this flag.
    cache_aware_overflow: bool = False

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
    # ── Realtime voice V2 (env VOICE_REALTIME_V2) ──────────────
    # Feature-flagged upgrade of the /api/ws/realtime relay: current model
    # generation (gpt-realtime-2.1), semantic turn detection, marin voice,
    # streaming transcription, barge-in truncation, budgeted instructions,
    # and per-turn credit metering. OFF ⇒ byte-identical v1 behavior.
    # Decision record: toup-platform-app/docs/decisions/voice-architecture.md
    voice_realtime_v2: bool = False
    # Per-account rollout: comma-separated user_ids that get V2 even while the
    # global flag is OFF. Lets us enable one account (the founder's) and verify
    # on-device before flipping V2 on for everyone. A connection uses V2 when the
    # global flag is on OR its user_id is listed here.
    voice_realtime_v2_user_ids: str = ""
    voice_realtime_model: str = "gpt-realtime-2.1"
    voice_realtime_transcription_model: str = "gpt-realtime-whisper"
    # Character budget for the personalized parts of the instructions blob
    # (memories + day history; identity docs are never trimmed). ~4 chars per
    # token ⇒ 24k chars ≈ 6k tokens, leaving headroom under OpenAI's
    # 16,384-token instructions+tools cap, which the tool schemas share.
    voice_realtime_instructions_budget_chars: int = 24000
    # Full-parity `think` (V2): the realtime relay runs on platform-api, where
    # the in-process agent_runner is absent, so `think` runs the user's OWN
    # agent over its HTTP /api/chat (the SAME AgentRunner text chat uses — full
    # tools, skills, and every connected MCP connector). A tool-using agent turn
    # (e.g. a web browse or a calendar action) can outlast the 15 s default VPS
    # timeout, so give it a generous ceiling. The realtime session has no turn
    # watchdog once past connect, and the model verbally holds the floor.
    voice_realtime_think_timeout_s: float = 60.0
    # Relay the hosted agent's INNER tool activity (which tool, the exact
    # query, the sources) to the phone during a voice `think`. Ships DARK:
    # it requires an agent image exposing /api/v1/internal/agent-turn/stream.
    # Off — or agent 404 — falls back to the blocking POST, i.e. today's exact
    # behavior. Nested inside _v2_active(), so v1 accounts are unaffected
    # regardless. Platform-api only; the agent side needs no flag, because
    # route existence IS the flag.
    voice_realtime_tool_events: bool = False
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
    # Sybil resistance: when True, the one-time free-credit grant is deduped
    # per CANONICAL email identity (Gmail dot/+alias variants AND
    # delete→re-signup collapse to a single grant) via the grant_eligibility
    # tombstone. Default False preserves the legacy "grant on every balance
    # creation" behavior while STILL recording tombstones, so enabling this
    # later is safe — the history already exists. See email_canonical.py.
    free_grant_dedupe_enabled: bool = False
    stripe_price_id_starter: str = ""
    stripe_price_id_builder: str = ""
    stripe_price_id_pro: str = ""
    stripe_price_id_elite: str = ""

    # ── Email verification (F13) + Gmail Workspace SMTP ──
    email_provider: str = "resend"
    resend_api_key: str = ""
    email_from_address: str = "Toup <mrhx@toup.ai>"
    email_from_reply_to: str = "mrhx@toup.ai"
    app_public_base_url: str = "https://toup.ai"
    aws_ses_region: str = "us-east-1"
    aws_ses_access_key_id: str = ""
    aws_ses_secret_access_key: str = ""
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_use_tls: bool = True
    # Gmail API (OAuth2) — send as a Google Workspace user over HTTPS.
    # Railway blocks outbound SMTP (25/465/587), so smtp times out from the
    # container; the Gmail REST API uses HTTPS:443 and still sends From: the
    # authenticated Workspace address (email_from_address). Set EMAIL_PROVIDER=gmail.
    gmail_oauth_client_id: str = ""
    gmail_oauth_client_secret: str = ""
    gmail_oauth_refresh_token: str = ""
    require_email_verification_for_credits: bool = False
    email_verification_required_after_iso: str = ""
    # Sybil resistance (grant-time gate, distinct from the spend-time gate
    # above): when True, the one-time free GRANT is withheld until the email
    # is verified. Account creation and product entry are NEVER blocked —
    # only the grant is deferred. OAuth/Apple users are provider-verified so
    # they grant instantly; password users unlock the grant by clicking the
    # verification link (sent in the background at signup, off the hot path).
    # Default False = grant at creation (legacy behavior).
    require_verified_email_for_grant: bool = False
    # Reject signups on known disposable/temp-mail domains (mailinator,
    # guerrillamail, …). Fail-open: a broken list never blocks signup. Off
    # by default so false-positives can be monitored before enforcing.
    disposable_email_blocklist_enabled: bool = False
    # Optional path to a fuller maintained disposable-domain list; extends
    # the bundled core list. Empty = bundled list only.
    disposable_email_domains_path: str = ""
    # When True, Apple sign-in dedupes on the stable `sub` claim before the
    # (mutable) relay email, so a Hide-My-Email relay change can't fork one
    # Apple-ID into two accounts. The `sub` is STORED regardless (additive);
    # this flag only controls whether the sub LOOKUP precedes email lookup.
    apple_sub_dedupe_enabled: bool = False

    # ── Founder / company-owner identity ──
    # Emails of Toup's owner(s). Drives the per-account "you're talking to
    # the founder" recognition block in the agent system prompt
    # (agent_runner._build_system_prompt → owner_recognition section). The
    # static company fact ("Toup was founded by …") lives in
    # app/agent/toup_facts.py; this list only gates the richer block to the
    # owner's OWN agent. Compared case-insensitively against the bound
    # user's email. NOT a security boundary — purely a prompt enhancement.
    founder_emails: list[str] = ["mrhx@toup.ai"]

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

    # ── Maintenance / support agent (app.support) ────────────────────
    # Master switch. Off ⇒ every /api/support route 503s. Dark-launched.
    support_agent_enabled: bool = False
    # Model for support classify/route/diagnose/implement LLM calls. Pinned
    # to gpt-4o-mini (2026-07 SOTA audit): None resolved to the platform
    # default (settings.agent_model, gpt-5.5) for 4+ calls/ticket the day
    # support_agent_enabled flips on. Env-overridable via
    # SUPPORT_AGENT_MODEL; empty string ⇒ platform default. Never pin
    # Claude while Anthropic is deactivated.
    support_agent_model: Optional[str] = "gpt-4o-mini"
    # Intake size cap (chars) — 413 above this.
    support_intake_max_chars: int = 8000
    # Concurrency cap for diagnosis context-gathering fan-out (file IO /
    # any per-subsystem analysis) — keeps us under provider/IO limits.
    support_diagnose_concurrency: int = 3
    support_diagnose_max_files: int = 6
    support_diagnose_max_file_lines: int = 160
    # IMPLEMENTATION gate (separate from approval). Off by default so the
    # diagnosis+approval loop can ship before the autonomous code-writer is
    # trusted, and so it stays inert where there's no git checkout / gh
    # (e.g. the Railway platform-api). Flip on in an ops runner that has a
    # repo checkout + gh credentials.
    support_implement_enabled: bool = False
    support_auto_implement_on_approve: bool = True
    support_open_pr: bool = True
    support_pr_base_branch: str = "main"
    # Repo checkout the implementer operates on. None ⇒ inferred from the
    # running module (the deployed image — usually NOT a git repo).
    support_repo_dir: Optional[str] = None
    # Verification commands run on the fix branch before opening a PR.
    # None ⇒ a safe default import check (see implementer._verify).
    support_verify_commands: Optional[List[str]] = None
    # ── Mobile support-intake (screenshots + admin alert) ────────────
    # Recipient of the "a support card arrived" alert. This is the
    # NOTIFICATION TARGET — distinct from the reporter and from the admin
    # account (nariman@toup.ai) that actually handles the queue. Configurable.
    support_notify_email: str = "mrhx@toup.ai"
    support_notify_enabled: bool = True
    # Screenshot attachment limits (bytes in the platform DB). Capped small;
    # served only via the auth'd reporter/admin endpoint.
    support_attachment_max_bytes: int = 3_000_000  # ~3 MB
    support_attachment_allowed_mime: List[str] = ["image/png", "image/jpeg", "image/webp"]

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

    # Autopilot — autonomous mission engine (Autopilot arc PR6).
    # DISTINCT from the legacy heartbeat_* block above (Telegram-only
    # nudger) — do not merge the flags; reusing heartbeat_enabled would
    # arm the old path. Default OFF until the PR7 action policy lands.
    autopilot_enabled: bool = False
    # Staged rollout: comma-separated full user ids that get Autopilot
    # even while the master flag is off (admin/canary tenants). Baked
    # image-wide via Dockerfile.agent; see app/agent/autopilot_gate.py.
    autopilot_user_allowlist: str = ""
    autopilot_default_budget_credits: int = 100   # ≈ $1 per mission
    autopilot_base_interval_seconds: int = 300    # APScheduler tick base
    autopilot_tick_timeout_seconds: int = 300     # asyncio.wait_for per turn
    autopilot_max_ticks: int = 100                # per-mission safety cap
    autopilot_backoff_cap_seconds: int = 7200     # adaptive-cadence ceiling
    autopilot_no_progress_blocker_threshold: int = 3  # strikes → ask user
    autopilot_platform_failclosed_threshold: int = 3  # unreachable ticks → pause

    # DM Pairing & Access Control
    telegram_require_pairing: bool = False  # Require /pair before using bot
    telegram_pairing_code: str = ""  # Pairing code users must provide

    # Docker Sandbox
    sandbox_enabled: bool = False  # Route exec through Docker container
    sandbox_image: str = "python:3.12-slim"  # Sandbox container image

    # ── Security hardening (docs/security/audit-2026.md) ───────────────
    # Production defaults. The platform is fully hosted (no user-supplied
    # provider keys — the LLM is provided per-user through the proxy), so the
    # hardened posture is the correct default. Each flag can still be turned
    # OFF via env for staging/debug.
    #
    # voice_identity_anchor: mirror the text-channel "you are the user's
    #   agent, never name the underlying LLM provider" guard into the voice /
    #   realtime + think prompts. Additive guard.
    voice_identity_anchor: bool = True
    # security_leak_filter: server-side alias of real provider/model ids to
    #   neutral tier labels on user-facing surfaces (usage, jobs, messages,
    #   telegram, proxy). Enforces the white-label story.
    security_leak_filter: bool = True
    # injection_fencing_v2: wrap ingested web/doc/trigger content in an
    #   untrusted-data envelope + a standing data≠instructions rule, and deny
    #   MUTATING CONNECTOR calls on no-user-present channels (routine/trigger/
    #   background). NOTE: internal tools (reminders, notifications, memory) are
    #   unaffected — only external connector WRITES (gmail send, drive/calendar
    #   write) are denied in an unattended run, because there is no human to
    #   confirm an action injected content might have requested (INJ-1).
    injection_fencing_v2: bool = True
    # exec_env_scrub: strip platform/tenant secrets from the env handed to the
    #   exec/PTY/code-engine CHILD shells. The agent PROCESS keeps them (used by
    #   embeddings, the proxy, DB), only the tool subprocess loses them, so one
    #   `printenv` in a tool call can't dump a credential. Safe under the fully-
    #   hosted model: no tenant shell command legitimately needs a provider key;
    #   the code engines re-inject their own auth explicitly after scrubbing.
    #   DATABASE_URL is intentionally KEPT so the documented `psql $DATABASE_URL`
    #   via-exec capability still works (tenant's own, per-DB-scoped credential).
    exec_env_scrub: bool = True
    exec_env_scrub_keys: list[str] = [
        "POOL_ADMIN_TOKEN",          # platform pool-admin bearer (cross-tenant)
        "AGENT_API_KEY",             # the agent's own inbound auth key
        "BRAVE_API_KEY",             # platform-shared search key
        "TOUP_TOKEN",                # LLM-proxy bearer (agent process only)
        "OPENAI_API_KEY",            # provider keys — used by the process, not shells
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "MISTRAL_API_KEY",
        "XAI_API_KEY",
        "DEEPSEEK_API_KEY",
        "CLAUDE_CODE_OAUTH_TOKEN",   # code engines re-inject their own after scrub
        "OPENAI_CODEX_TOKEN",
        "TELEGRAM_BOT_TOKEN",
        "DISCORD_BOT_TOKEN",
        # DATABASE_URL intentionally NOT listed — preserves psql-via-exec.
    ]
    # embeddings_via_proxy: route the agent's embedding calls through the
    #   platform LLM proxy (TOUP_TOKEN) instead of a raw OpenAI key in the
    #   container — the last step to remove ALL provider keys from tenant
    #   containers (audit EXF-1 residual / hardening-runbook.md Step 1).
    #   Default OFF: this is a MEMORY-CRITICAL path — enable + validate on
    #   staging (store/search a memory, confirm a proxy `embeddings` event)
    #   BEFORE flipping it on and dropping OPENAI_API_KEY from the bridge env.
    embeddings_via_proxy: bool = False
    # exec_sandbox_user: run the `exec`/PTY tool AND its children as a separate,
    #   lower-privileged OS user (e.g. "sandbox") instead of the agent's uid.
    #   This is the REAL fix for "exec can read /app platform source and
    #   /proc/<agent>/environ" — those become unreadable to a different uid,
    #   while the agent process itself keeps full access. The container image
    #   must define this user and give it write access ONLY to the workspace
    #   (see Dockerfile.agent). Empty = disabled (current behaviour). The agent
    #   process must be root (or hold CAP_SETUID) to drop privileges.
    #   hardening-runbook.md Step 2. Enable + validate on staging first.
    exec_sandbox_user: str = ""

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
        # Image tools must outlast the SLOWEST render, not the average one.
        # Kie/Nano Banana is highly variable — 25s/36s/39s/74s and a measured
        # 399s success — and at 200s the wrapper killed the tool while the job
        # was still rendering, so the user waited the full 200s, got nothing,
        # and was still billed for the abandoned task. Keep this ABOVE
        # kie_job_timeout_s (420) so the polling loop reports a real reason
        # first instead of being cut off mid-flight.
        "generate_image": 480,
        "edit_image": 480,
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
    # Periodic reconciler that re-runs `backfill_sentinel_image_containers`
    # so a signup that lands in the broken state (container_id NULL or the
    # ":latest" sentinel image — both mean "no real bridge container, user
    # can't chat") self-heals within minutes instead of waiting for the next
    # platform redeploy. The pool fast-path's bridge claim response carries no
    # container_id, so EVERY pool-claimed signup is born with container_id
    # NULL; before this loop the boot-only backfill left such users stuck for
    # hours (2026-05-31: mrvviinn@gmail.com signed up post-boot and couldn't
    # talk to their agent at all). 0 disables the loop.
    container_reconciler_interval_s: int = 180
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

    # Sign in with Apple (Guideline 4.8): the audience the identity
    # token must carry. For native iOS Sign in with Apple this is the
    # app's bundle id; override via APPLE_CLIENT_ID for a web Service ID.
    apple_client_id: str = "ai.toup.app"

    # Sign in with Apple — token revocation (Guideline 5.1.1(v)). When a
    # user deletes their account we must revoke their Apple tokens via
    # Apple's REST API (auth/token + auth/revoke), which requires a
    # client_secret JWT signed with a "Sign in with Apple" key (.p8)
    # created in the Apple Developer portal (Keys → enable Sign in with
    # Apple, scoped to the primary App ID). These are SEPARATE from the
    # App Store Connect API key. If any is unset, the exchange/revoke
    # calls no-op and we log a warning — so deploying this code before
    # the key exists is safe (sign-in is unaffected).
    apple_team_id: str = ""        # Set via APPLE_TEAM_ID (10-char, e.g. 5W2R26Z4H7)
    apple_key_id: str = ""         # Set via APPLE_KEY_ID — the Sign in with Apple key's Key ID
    apple_private_key: str = ""    # Set via APPLE_PRIVATE_KEY — the .p8 PEM contents (literal \n tolerated)

    # StoreKit In-App Purchase — consumable credit-pack receipt validation
    # (App Store Server API + App Store Server Notifications V2). These come
    # from an "In-App Purchase" key created in App Store Connect → Users and
    # Access → Integrations → In-App Purchase (a .p8 + Key ID + Issuer ID),
    # which is SEPARATE from both the Sign in with Apple key above AND the
    # generic App Store Connect API key. If any of key_id / issuer_id /
    # private_key is unset, apple_iap_service.iap_configured() returns False
    # and the /iap/apple/verify endpoint returns 503 — so deploying this code
    # before the key exists is safe and provably inert.
    apple_iap_key_id: str = ""        # Set via APPLE_IAP_KEY_ID — the IAP key's Key ID
    apple_iap_issuer_id: str = ""     # Set via APPLE_IAP_ISSUER_ID — issuer id from the key page
    apple_iap_private_key: str = ""   # Set via APPLE_IAP_PRIVATE_KEY — the .p8 PEM (literal \n tolerated)
    apple_iap_bundle_id: str = "ai.toup.app"   # Set via APPLE_IAP_BUNDLE_ID
    apple_iap_app_apple_id: str = "6774296172"  # Set via APPLE_IAP_APP_APPLE_ID — numeric ASC app id (notification verification)

    # ── Cloudflare R2 (audio serving cache, S3-compatible) ───
    # When r2_audio_enabled + the four R2_* vars are set, the audio remux cache
    # stores AND serves audio from R2 (free egress, uncapped) instead of leaning
    # on the small Postgres BYTEA L2. Decouples repeat playback from the user's
    # Mac/residential proxy entirely. Fail-open: any R2 miss/error falls back to
    # the Postgres L2, then the itag-18 proxy — never a playback regression.
    r2_audio_enabled: bool = False                  # AUDIO_R2: set R2_AUDIO_ENABLED=true to activate
    r2_account_id: Optional[str] = None             # R2_ACCOUNT_ID (S3 endpoint host)
    r2_bucket: Optional[str] = None                 # R2_BUCKET
    r2_access_key_id: Optional[str] = None          # R2_ACCESS_KEY_ID
    r2_secret_access_key: Optional[str] = None      # R2_SECRET_ACCESS_KEY

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
    # Signup provisioning (config prime + pool claim in the register /
    # OAuth finalize paths). Deliberately its OWN flag: it used to ride
    # on prewarm_on_soul_save, so flipping that prewarm experiment off
    # would have silently stopped provisioning agents for new signups —
    # the worst possible failure (bulletproof plan L). Default True;
    # never couple this to an experiment flag again.
    signup_provisioning_enabled: bool = True
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
    # Cloudflare Turnstile is a web-only widget that can't render in the
    # native iOS/Android app. When True (default), native clients (identified
    # by the X-Toup-Client header) are admitted on the signup IP rate limiter
    # instead of a Turnstile token — so enabling TURNSTILE_SECRET_KEY to gate
    # WEB signups doesn't silently 400 every mobile signup. App Attest /
    # Play Integrity is the tracked follow-up to attest native clients.
    turnstile_exempt_native_clients: bool = True
    # Signup rate limit — IP-keyed. 5 signups per hour per IP, copied
    # from the login rate limiter pattern. Disabled if either value is 0.
    signup_rate_limit_per_hour: int = 5
    # When True, the signup IP limit is enforced via the signup_attempts
    # table (cross-replica, survives deploys) instead of the per-process
    # in-memory limiter. Default False = legacy in-memory behavior.
    signup_rate_limit_persistent: bool = False
    admin_alert_telegram_token: str = ""    # Bot token for admin alerts (shared / user-facing bot)
    admin_alert_telegram_chat_id: str = ""

    # ── Onboarding v2 rollout (PR 4 of 4) ─────────────────────────
    # `onboarding.v2_rollout_pct` (0-100) gates the new Free+credit-tier
    # LLM step that replaces the binary Toup-bundle / BYO card. Stored in
    # platform_settings.value as an integer string so admins can flip
    # 10 → 50 → 100 without a backend redeploy. This env-var default is
    # the boot-time fallback when no DB row exists yet (e.g. fresh
    # deploys); per-user opt-in is computed via a deterministic hash of
    # user_id mod 100 in feature_flags.py.
    onboarding_v2_rollout_pct: int = 0

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

    # Optional residential/rotating proxy for yt-dlp. YouTube's bot challenge
    # ("Sign in to confirm you're not a bot") fires on datacenter egress IPs
    # (Railway) and is the root cause of mobile background-audio extraction
    # failing. Routing yt-dlp through a residential IP removes that signal.
    # Format: http(s)://user:pass@host:port or socks5://...  Set env YT_DLP_PROXY.
    # IMPORTANT: googlevideo signs the stream URL to the extracting IP, so the
    # /audio_stream byte-pump must use the SAME proxy (see media_proxy.py) or
    # the fetch 403s. Leave unset to extract direct (works only when YouTube
    # isn't blocking the server IP). Cookies (yt_dlp_cookies_b64) are the
    # cheaper alternative — no per-GB proxy bandwidth — but need refresh.
    yt_dlp_proxy: Optional[str] = None

    # Base URL of the bgutil PO-token provider (the `bgutil-pot` Railway
    # service, reachable on the private network). When set, yt-dlp's web
    # clients fetch a GVS "proof of origin" token from it — the free,
    # no-bandwidth-cost way past YouTube's datacenter-IP bot challenge (vs.
    # yt_dlp_proxy, which fixes the same block but meters every audio byte).
    # Format: http://bgutil-pot.railway.internal:4416  Set env BGUTIL_POT_BASE_URL.
    bgutil_pot_base_url: str = ""

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

    # Realtime voice pricing (USD per 1M tokens). Voice needs its own table
    # because audio and text tokens price differently and the cached-input
    # rate is the dominant cost lever (the whole conversation re-bills on
    # every response; cached context is $0.40/M vs $32/M uncached).
    # Verified against developers.openai.com/api/docs/pricing 2026-07-17.
    voice_realtime_pricing_per_1m: dict[str, dict[str, float]] = {
        "gpt-realtime-2.1": {
            "audio_in": 32.0, "audio_in_cached": 0.40, "audio_out": 64.0,
            "text_in": 4.0, "text_in_cached": 0.40, "text_out": 24.0,
        },
        "gpt-realtime-2.1-mini": {
            "audio_in": 10.0, "audio_in_cached": 0.30, "audio_out": 20.0,
            "text_in": 0.60, "text_in_cached": 0.06, "text_out": 2.40,
        },
        "gpt-realtime": {
            "audio_in": 32.0, "audio_in_cached": 0.40, "audio_out": 64.0,
            "text_in": 4.0, "text_in_cached": 0.40, "text_out": 16.0,
        },
    }

    # ── Image generation (ChatGPT / gpt-image-1) ───────────────
    # The agent's generate_image tool and the bundle LLM proxy's images route
    # both read these. gpt-image-1 is priced PER IMAGE (not per token) and the
    # price depends on (size, quality). We peg 1 credit = 1 cent of true OpenAI
    # cost, so image_gen_pricing_cents holds OpenAI's published per-image USD
    # cost expressed in cents. Admins can tune these live via platform_settings
    # (credit.image_price.* handled the same way as flat-fee overrides is a
    # future extension; for now edit here or via env override).
    image_gen_enabled: bool = True
    # gpt-image-2 (OpenAI's GPT Image 2, GA 2026-04-21) — same /images/generations
    # and /images/edits endpoints and same low|medium|high quality tokens as
    # gpt-image-1, so it is a drop-in. gpt-image-1 is scheduled for shutdown
    # ~2026-10-23, so we default to v2 and keep v1 only as a same-API fallback
    # (works through the bundle proxy AND the direct path) for the unlikely case
    # the platform org isn't yet verified for v2. Override live via IMAGE_GEN_MODEL.
    image_gen_model: str = "gpt-image-2"
    image_gen_fallback_model: str = "gpt-image-1"  # same Images API; used if the primary model errors (not on moderation)
    image_gen_default_size: str = "1024x1024"     # 1024x1024 | 1024x1536 (portrait) | 1536x1024 (landscape)
    image_gen_default_quality: str = "high"       # low | medium | high  (default HIGH per product decision)
    image_gen_timeout_s: float = 180.0            # gpt-image edits/gens can be slow
    # edit_image (the /images/edits endpoint) — modify a user's uploaded image.
    # Reuses image_gen_* pricing/size/quality/timeout. dall-e has no edits
    # endpoint, so only a gpt-image-* model can be the edit fallback.
    image_edit_enabled: bool = True

    # Per-image cost in CENTS, keyed "<size>:<quality>" — the PRIMARY model's
    # (gpt-image-2) published per-image USD cost expressed in cents. Unknown
    # combos fall back to image_gen_fallback_cents. 1 cent -> 1 credit at settle.
    # Source: OpenAI image-generation guide per-image table (2026).
    image_gen_pricing_cents: dict[str, float] = {
        "1024x1024:low": 0.6,   "1024x1024:medium": 5.3,  "1024x1024:high": 21.1,
        "1024x1536:low": 0.5,   "1024x1536:medium": 4.1,  "1024x1536:high": 16.5,
        "1536x1024:low": 0.5,   "1536x1024:medium": 4.1,  "1536x1024:high": 16.5,
    }
    image_gen_fallback_cents: float = 21.1        # unknown size/quality -> charge ~high 1024²
    # Legacy gpt-image-1 per-image cents — used ONLY to bill the rare fallback
    # path correctly (a v1 image must not be billed at v2's higher rate).
    image_gen_pricing_cents_legacy: dict[str, float] = {
        "1024x1024:low": 1.1,   "1024x1024:medium": 4.2,  "1024x1024:high": 16.7,
        "1024x1536:low": 1.6,   "1024x1536:medium": 6.3,  "1024x1536:high": 25.0,
        "1536x1024:low": 1.6,   "1536x1024:medium": 6.3,  "1536x1024:high": 25.0,
    }
    image_gen_fallback_cents_legacy: float = 17.0

    # ── Image engine: Kie.ai / Nano Banana (PRIMARY) ───────────────
    # Highest-quality natural image gen/edit. When image_provider == "kie" the
    # agent routes generate_image/edit_image through the platform's Kie proxy
    # (one shared platform key, like bundle OpenAI — users never hold it) and
    # falls back to OpenAI (gpt-image-2) on any Kie failure. Set image_provider
    # to "openai" to disable Kie instantly (env IMAGE_PROVIDER=openai).
    image_provider: str = "kie"                   # "kie" (Nano Banana primary) | "openai"
    kie_api_key: str = ""                          # platform secret (env KIE_API_KEY); one shared key
    kie_api_base: str = "https://api.kie.ai"       # jobs/createTask + jobs/recordInfo
    kie_upload_base: str = "https://kieai.redpandaai.co"  # file-base64-upload host (NOT api.kie.ai)
    kie_image_model: str = "nano-banana-pro"       # Gemini 3 Pro Image — unified generate + edit
    kie_image_size: str = "4K"                     # 1K | 2K | 4K (Pro). 4K measured same 18 Kie credits as 2K, ~35s.
    kie_output_format: str = "png"
    # Legacy SYNCHRONOUS path (kie_client.generate/edit) only. The agent now uses
    # start+poll, which is bounded by kie_job_timeout_s below instead.
    kie_timeout_s: float = 120.0
    kie_poll_interval_s: float = 2.5
    # How long the AGENT keeps polling a started Kie job. Measured Kie renders:
    # 25s / 36s / 39s / 74s … and a 399s success. The old single synchronous call
    # gave up at kie_timeout_s and ABANDONED a task the user had ALREADY been
    # charged 18 Kie credits for — the founder's 20:08 edit sat `running` on Kie,
    # billed, never delivered. Polling is cheap, so this is generous enough to
    # actually collect that slow tail instead of paying for nothing.
    kie_job_timeout_s: float = 420.0
    # Billing: Kie returns creditsConsumed (Kie credits) per task; we charge the
    # user (kie credits × kie_credit_cents) our-credits. ~0.5¢/Kie-credit
    # (nano-banana-pro 2K ≈ 18 credits ≈ 9¢). Tune once reconciled with Kie's
    # live pricing; env KIE_CREDIT_CENTS.
    kie_credit_cents: float = 0.5
    kie_fallback_cents: float = 9.0                # charge if creditsConsumed is missing

    # Free-tier monthly image quota — a free user may generate/edit this many
    # images per calendar month; paid users and admins are unlimited. 0 = off.
    free_tier_monthly_image_limit: int = 10

    # Strip whitespace from API key fields on load (users often paste with spaces)
    _KEY_FIELDS = {
        "openai_api_key", "anthropic_api_key", "google_api_key",
        "mistral_api_key", "groq_api_key", "xai_api_key", "deepseek_api_key",
        "telegram_bot_token", "discord_bot_token", "slack_bot_token",
        "slack_app_token", "brave_api_key", "elevenlabs_api_key",
        "cohere_api_key", "agent_api_key", "toup_token", "kie_api_key",
        "stripe_secret_key", "stripe_webhook_secret",
        "gmail_oauth_client_id", "gmail_oauth_client_secret",
        "gmail_oauth_refresh_token",
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
