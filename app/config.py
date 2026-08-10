from pydantic_settings import BaseSettings
from pydantic import model_validator
from functools import lru_cache
from typing import List, Optional


class Settings(BaseSettings):
    # App
    app_name: str = "Toup Agent Platform"
    debug: bool = True

    # SQLAlchemy statement echo. Deliberately NOT tied to `debug`.
    #
    # `debug` defaults True and the agent containers never set DEBUG, so
    # `echo=settings.debug` meant every one of the 55 tenant agents ran with
    # full statement echo in production. That is not just noise:
    #   - echo logs BOUND PARAMETERS, so `INSERT INTO messages (... content ...)`
    #     wrote the user's message body verbatim into the container log
    #     (confirmed on a live tenant, 2026-08-01);
    #   - every statement was emitted twice (echo's own handler plus root
    #     propagation), ~6k log lines per 20 min on one container, which is
    #     what makes a real incident unreadable;
    #   - it formats every statement and its parameters on the hot path.
    #
    # Echo is a debugging tool that exposes user data, so it must be opted
    # into explicitly (SQL_ECHO=true) and never ride on a general dev flag.
    sql_echo: bool = False

    # POST /auth/demo mints a real session token for a shared demo account,
    # unauthenticated, with a password hardcoded in source. That is a login
    # backdoor when reachable in production (the demo user was found live in
    # the prod DB, created 2026-07-08). Like sql_echo, a convenience for
    # local development must be opted into explicitly (DEMO_LOGIN_ENABLED=true)
    # — when off, the route 404s so it does not even advertise itself.
    demo_login_enabled: bool = False

    # Plan init_db()'s boot-time ALTER list against the live catalog and issue
    # only the statements whose effect is actually missing, instead of
    # replaying all ~358 every start.
    #
    # This is not a micro-optimisation. Postgres takes ACCESS EXCLUSIVE BEFORE
    # evaluating `IF NOT EXISTS`, so each "idempotent" no-op still queues for
    # an exclusive lock on a table the blue container is actively serving
    # during a blue-green swap (proven on postgres:16, 2026-08-03). It is why
    # rollouts abort with `health_checks_passed: 0` and why the health budget
    # was raised 30 -> 120 -> 240 without ever fixing it.
    #
    # Default OFF: this governs schema delivery to tenants, and a wrong skip
    # is a silent missing column. Enable per-tenant (INIT_DB_PLAN_DDL=true) to
    # soak on the canary before the fleet. See app/db/ddl_plan.py.
    init_db_plan_ddl: bool = False

    # Record the stack at every pool checkout and replay it if that connection
    # is garbage-collected without ever being checked in.
    #
    # `SAWarning: ... non-checked-in connection` names no origin, and it is
    # emitted by the GC — so the surrounding log lines are where the COLLECTION
    # happened, not where the session leaked. Reading them has now aimed three
    # fixes (#407, #408, #418) at real defects that each failed to drive the
    # rate to zero. This makes the origin an observation instead of a guess.
    #
    # Formats a stack on every checkout, so it is a diagnostic to switch on for
    # one tenant while reproducing — not something to leave running.
    pool_leak_debug: bool = False

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
    
    # Runtime memory retrieval (agent_runner's per-turn hybrid_search).
    # Both were hardcoded at the call site until 2026-07-29: limit=15 and
    # min_similarity=0.1. Retrieval telemetry rated only 7.3% of retrievals
    # "good" (27 of 372) and nearly every query returned a full 15 rows —
    # "say OK" retrieved 15 memories.
    #
    # The floor gates RAW COSINE in the vector path only (see
    # MemoryService.hybrid_search: it is deliberately NOT applied to the
    # RRF-fused set, so keyword/graph matches still surface at any cosine).
    # 0.35 is calibrated against the real corpus, not guessed — all-pairs
    # cosine over the 119 live memories on the founder's tenant:
    #     p50 0.154 · p75 0.222 · p90 0.342 · p95 0.481 · p99 0.713
    # The old 0.10 floor admitted 76% of ALL pairs, i.e. no floor at all.
    # 0.35 sits at the p90 boundary and admits 9.6% — enough to fill k=10
    # when genuinely relevant memories exist, and nothing when they don't.
    # Re-measure before changing; tune against retrieval_events.
    memory_retrieval_limit: int = 10
    memory_retrieval_min_similarity: float = 0.35

    # Retrieval-time auto-reinforcement (2026-08-06). OFF by default, and it
    # is a kill switch in the "put it back" direction, not a feature gate.
    #
    # MemoryService.hybrid_search used to reinforce its own top 5 results on
    # EVERY call, unconditionally. Two problems, both live:
    #   1. Retrieval is not evidence of relevance — it is the ranker's opinion.
    #      Reinforcing on retrieval makes a mis-retrieval permanently stronger
    #      and therefore more likely to be retrieved again. With decay never
    #      running against tenant memories (agent_memory_maintenance_enabled is
    #      False, and platform_main's decay job runs against the platform DB
    #      where `memories` does not live), strength is a ONE-WAY RATCHET.
    #   2. DecayService.reinforce_memory COMMITS the session it is handed, so
    #      the loop fired up to 5 commits on the caller's turn-scoped session
    #      in the middle of system-prompt assembly — ×2, because a turn calls
    #      hybrid_search on both the user leg and the agent-brain leg.
    #
    # Reinforcement now happens in RetrievalFeedback.log_retrieval_feedback,
    # off the request path, for the memories the response actually cited.
    # Flip this True only to restore the old behaviour for a controlled
    # comparison; it stacks on top of the cite path, it does not replace it.
    memory_reinforce_on_retrieval: bool = False

    # Cite-time reinforcement — the replacement path. ON by default because it
    # is the corrected form of a behaviour that already shipped (retrieval-time
    # reinforcement); turning both this and the flag above off means memories
    # are never reinforced by conversation at all. Ops kill switch only.
    memory_reinforce_on_cite: bool = True

    # Chat & Session Settings
    memory_recall_limit: int = 15  # How many memories to recall per message
    auto_extract_memories: bool = True  # Auto-extract memories from conversations
    max_history_messages: int = 20  # Max conversation history to include
    # W1.4a: pin the background extraction fan-out (fact extraction,
    # relationship extraction, dedup adjudication, dedup merge) to a cheap
    # model. LLMService.default_model resolves to settings.agent_model
    # (gpt-5.5) whenever the Anthropic key is present (llm_service.py:69) —
    # any unpinned call in this pipeline silently rides the premium chat
    # model. Every call site passes this explicitly.
    memory_extraction_model: str = "gpt-4o-mini"
    # W1.4c: skip memory + relationship extraction on turns the
    # trivial-query classifier flagged (greetings, acks, "thanks"). The
    # context_trim_for_trivial_queries gate covered retrieval only —
    # extraction still burned 2+ LLM calls on every trivial turn.
    skip_extraction_for_trivial_queries: bool = True
    # D-mem-A (2026-07-29, canary 533354ce): the dedup auto-duplicate shortcut
    # (similarity >= 0.90) reinforced the EXISTING row and discarded the new
    # content unconditionally — "…passphrase is kestrel-dbf7" restated as
    # "…kestrel-13b4" embeds >= 0.90, so users could never change a stored
    # fact. With this ON, a high-similarity pair whose texts disagree on a
    # value token is sent to the LLM adjudicator, whose contradiction_update
    # verdict supersedes the old row (memories.superseded_by, is_active=False)
    # instead of throwing the new value away. True paraphrase duplicates keep
    # the cheap no-LLM shortcut. Kill switch because this changes WRITE
    # semantics for restated facts: OFF restores unconditional reinforcement.
    memory_supersede_on_conflict: bool = True
    # D-mem-C (2026-07-29): explicit remember-phrasing ("please remember:",
    # "for my records", "note this down", "save this") now boosts the memory
    # intent in query classification, so the save ask gets the memory prompt
    # sections/tool set even when its payload collides with code keywords
    # ("…my parking spot CODE is …" used to tie into code intent). Kill
    # switch: OFF restores the old intent scoring. (memory_store/memory_search
    # exposure itself is unaffected — those are always-included tools.)
    memory_tools_on_remember: bool = True
    
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
    #
    # DEFAULT ON since 2026-08-10 (G-17), after the curve was finally run
    # against a real tenant — a pg_dump of the founder's 143-memory store
    # restored into a throwaway pgvector instance and driven through the
    # real `run_decay_for_all_users` entry point (128 of 133 rows updated).
    # What it would do to live rows, measured on that copy:
    #
    #   rows <30d old   n=50   strength 0.999 -> 0.971
    #   rows 30-90d     n=15            1.000 -> 0.690
    #   rows >90d       n=1             1.000 -> 0.539
    #   high-importance n=24            1.000 -> 0.937   (min 0.539)
    #   rows landing at the 0.1 floor: 0     mean drop: 0.098
    #
    # The load-bearing structural fact: decay CLAMPS at MIN_STRENGTH=0.1
    # and hybrid_search's floor is `strength >= 0.1`, so no amount of decay
    # can push a memory out of recall — the worst case is re-weighting
    # (strength is 20% of final_score), never forgetting. Meanwhile
    # reinforcement-on-cite is on, which without decay makes strength a
    # ONE-WAY RATCHET: every row drifts to 1.0 and the 20% term stops
    # discriminating anything. This is the counterweight that was designed
    # for it. TTL/expiry remains a separate, always-on per-turn sweep.
    # This flag starts FOUR jobs, not one, so decay alone was not enough
    # evidence. Consolidation was dry-run the same way, on the same copy:
    # 27 candidates -> 5 groups -> 11 memories consolidated, and the diff
    # against the pre-run snapshot is the reassuring part —
    #
    #   rows whose content changed .......... 0
    #   rows retired/superseded/deactivated . 0
    #   memory levels changed ............... 0
    #   source rows marked consolidated ..... 11
    #   NEW semantic summary rows ........... 5
    #
    # i.e. consolidation is ADDITIVE: it writes summary rows and stamps its
    # sources, it does not rewrite or retire anything. Cost is ~10
    # gpt-4o-mini calls per tenant per day at 03:00. The remaining two jobs
    # are retrieval_feedback_analysis (weekly, reads retrieval_events) and
    # day_archival (hourly, already gated separately on enable_day_recall).
    #
    # Rollback: AGENT_MEMORY_MAINTENANCE_ENABLED=false in the bridge env
    # (already forwarded by pool_addon) + a recreate wave.
    agent_memory_maintenance_enabled: bool = True

    # Agent-brain reflection (2026-07-29). The ONLY producer of
    # brain_type='agent' rows, which back the app's "Learned" tab — that tab
    # was permanently empty because nothing wrote them. Costs one extra LLM
    # call, but only on turns where a cheap regex gate detects a correction or
    # a stated working preference, so typical turns are unaffected.
    agent_reflection_enabled: bool = True

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
    # 2026-08-07: gpt-5.5 → gpt-5.6-terra. Gate G1 passed on OpenAI's own
    # organization billing, normalised for prompt-cache hit rate — terra is
    # 50-55% cheaper per token on EVERY token class (uncached input -53.8%,
    # cached input -62.5%, output -62.1%), p50 -17.5%, p95 -54.1%, and its
    # error rate is marginally lower (2.1% vs 2.6% over 755 and 697 calls).
    # Full working, including the retracted earlier "-14.2%, G1 = NO" cut
    # that failed to control for cache hit rate:
    # docs/audits/2026-08-g1-cost-and-latency.md §8.
    # The Responses wire follows automatically — model_resolver.wire_api_for()
    # derives it, so this cannot be flipped into the broken half-state.
    agent_model: str = "gpt-5.6-terra"  # Primary agent model (OpenAI — per-tenant key)
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
    # Voice's own ceiling. The realtime relay abandons a `think` call at
    # voice_realtime_think_timeout_s (60s), and a measured voice research turn
    # ran 17 tool calls in 113s — a correct answer delivered after the caller
    # had already been handed the tool-less fallback. 8 is enough for a search,
    # a few reads and a synthesis; the model is told the number so it plans to
    # it rather than being cut off. Raise only alongside the relay timeout.
    #
    # Re-measured after the ceiling AND the search gateway shipped: the same
    # Farsi research question went 113s/17 calls -> 56.3s/9 calls. Inside the
    # 60s relay budget, but by 3.7s — a coin flip, not a fix, and 56s is still
    # a bad spoken experience. Dropped to 5 and paired with a snippet-first
    # instruction in the voice channel guidance, because Brave returns
    # extra_snippets precisely so an answer does not need a web_fetch per
    # source, and web_fetch is where the seconds actually go.
    voice_max_tool_iterations: int = 5
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

    # ── web_search / web_fetch metering ──────────────────────────
    # Until 2026-07 these two tools were the only priced-in-FLAT_FEES events
    # with no charge call site anywhere: `_flat_fee_for_tool`'s web_search /
    # web_fetch branches are reachable only from the connector dispatcher,
    # which never sees those names. Result: zero `tool_call` rows in
    # credit_ledger and no per-user view of internet usage at all.
    #
    # metering_enabled writes one ledger row per OUTBOUND call (cache hits
    # are free and are never billed, only logged). metering_charge decides
    # whether that row moves the balance:
    #   False (default) → meter_only, amount=0, nobody is billed. The row
    #                     still carries credits_quoted + meter_would_deny, so
    #                     you get the full usage series AND a dry-run of the
    #                     denial rate before switching pricing on.
    #   True            → the FLAT_FEES amount is deducted for real.
    # Landing charge=False is deliberate: CREDIT_ENFORCEMENT_ENABLED is true
    # in production, so flipping both at once would start denying searches on
    # the same deploy that first measures them.
    web_tool_metering_enabled: bool = True
    web_tool_metering_charge: bool = False
    # Same split for the OTHER flat-fee tools. FLAT_FEES has priced
    # browser_action, browser_screenshot and doc_gen_* since the credits arc
    # shipped, but `_flat_fee_for_tool`'s only caller is connector_dispatcher,
    # which never sees those names — so seven priced entries were unreachable
    # and browser/document work produced no usage record at all. Metered from
    # ToolExecutor.execute (the single dispatch point) via `_FLAT_FEE_TOOLS`.
    # Charge defaults OFF for the same reason web tools do: these have never
    # been billed, and measuring must come before taking credits.
    flat_tool_metering_enabled: bool = True
    flat_tool_metering_charge: bool = False
    # Brave's 50 req/s is an ACCOUNT ceiling shared by every key and therefore
    # by every tenant on the platform's shared key, so one runaway agent loop
    # can starve the whole fleet. These bound ONE tenant's share, enforced in
    # its own container (agents share no memory, so a true fleet-wide limiter
    # would need a coordinator that does not exist yet).
    # 2/s sustained with a burst of 5 comfortably covers a real turn — even
    # one that fans out several parallel web_search calls — while capping a
    # runaway loop far below the shared ceiling.
    brave_rate_per_sec: float = 2.0
    brave_burst: int = 5
    # After a 429 the account ceiling is already hit; retrying into it just
    # adds a wasted round-trip to every subsequent search.
    brave_cooldown_s: float = 30.0
    # Fleet headroom floor. Brave reports remaining requests against the
    # shared per-second ACCOUNT ceiling on every response; the gateway sheds
    # to lower tiers while that number sits at or below this. It is the only
    # fleet-wide signal available — there is no Redis, and platform-api runs
    # more than one replica, so no counter of ours can see the whole fleet.
    brave_fleet_floor: int = 5
    # platform-api replica count (railway.json deploy.numReplicas). Rate-limit
    # buckets in search_proxy are per PROCESS, so this is what makes
    # brave_rate_per_sec/brave_burst mean their stated value fleet-wide rather
    # than N times it. Measured 2026-07-31: with this unset, a configured burst
    # of 5 let 10 requests through in 762ms across 2 replicas.
    # KEEP IN SYNC with railway.json.
    platform_replicas: int = 2

    # Per-USER daily search ceiling — the only VOLUME limit in the system.
    # Everything else here bounds RATE and, on breach, degrades to a free
    # lower tier (scrape / headless Chromium) rather than refusing, so a
    # runaway loop was converted into unbounded scraping, never into less
    # work. `search_web` says so outright: "There is deliberately NO deny
    # path." Metering is meter-only (web_tool_metering_charge=False), so the
    # credit ledger cannot stop it either: measured 2026-08-03, 99 tool_call
    # rows quoting 99.00 credits and charging 0.00.
    # One deep-research turn spent 22 Brave calls, so this is a per-user
    # ABUSE ceiling, not a product limit — it should sit well above a normal
    # day and well below what would burn brave_monthly_budget (2000/mo across
    # the whole account) on one user. Counted from `search_events` rows that
    # reached Brave, over the user's rolling 24h.
    # 0 disables the ceiling entirely.
    search_daily_cap_enabled: bool = False
    search_daily_cap_per_user: int = 200

    # Realtime voice billing. The metering gate was dead
    # (`using_platform_key = not openai_key`, always False on a surviving
    # session), so live voice has NEVER been charged: zero realtime rows in
    # llm_proxy_events, ever. The gate is fixed, but it lands in meter-only —
    # rows with amount=0 carrying the real cost — because switching a
    # never-billed feature straight to billing, with no idea what a session
    # actually costs, is how you surprise every voice user at once. Read the
    # series off `credit_ledger` where endpoint='realtime_voice', then flip.
    # NOTE: realtime INPUT transcription is billed separately by OpenAI and
    # does not appear in `response.usage`, so the recorded cost UNDERSTATES
    # reality. Price with that in mind.
    voice_metering_charge: bool = False

    # Credit-health monitor (app/services/credit_health_monitor.py). Asserts
    # hourly the invariants that make the meter mean anything. Every symptom of
    # the 2026-08-03 incident — 274 denied-but-served calls, two accounts with
    # 48 and 117 duplicate grants, $17.17 given away — was queryable the whole
    # time; nothing looked. Thresholds are deliberately tight: steady state for
    # the first two alarms is exactly zero.
    credit_health_monitor_enabled: bool = True
    credit_health_check_interval_s: int = 3600
    credit_health_window_h: int = 24
    # Provider spend attached to denied-and-served rows before it pages as
    # critical rather than a warning.
    credit_health_unbilled_usd_critical: float = 5.0
    # Credits charged ÷ provider cents. 1.0 is break-even by design (1 credit
    # = 1¢ of underlying cost); below this the meter is undercounting.
    credit_health_ratio_critical: float = 0.5
    # Sample floors — below these a ratio or a zero is noise, not signal.
    credit_health_min_cost_cents: float = 200.0
    credit_health_min_events: int = 50

    # Search-quota monitor (app/services/search_quota_monitor.py). Brave's
    # monthly rate-limit bucket reports limit 0 / remaining 0 on this plan, so
    # remaining credits CANNOT be read off a response header — the monthly
    # alarm is our own count of search_events rows that reached Brave, checked
    # against a budget stated here. If the plan changes, this number is the
    # only thing that knows.
    search_quota_monitor_enabled: bool = True
    search_quota_check_interval_s: int = 3600
    # Look-back for the rate-based alarms, and the sample floor below which a
    # rate is noise rather than signal.
    search_alert_window_h: int = 3
    search_alert_min_samples: int = 20
    # Operator-stated monthly allowance. Nothing can discover it for us.
    brave_monthly_budget: int = 2000
    brave_monthly_warn_pct: float = 80.0
    brave_monthly_critical_pct: float = 95.0
    # Share of calls shed to slower tiers over the window.
    search_throttle_warn_pct: float = 25.0
    search_throttle_critical_pct: float = 50.0
    # Share of readings at or below brave_fleet_floor.
    search_headroom_warn_pct: float = 10.0
    # Kill-switch (default on): keep the tenant workspace writable by BOTH the
    # root agent and the uid-1000 exec sandbox it drops children to. Replaces
    # the post-rollout manual `chmod -R a+rwX workspace/generated` that had to
    # be re-run by hand on every hardened tenant after every recreate.
    # Off → files revert to root-owned 0644 and the agent's own shell cannot
    # edit what the agent wrote. See services/workspace_perms.
    workspace_shared_perms_enabled: bool = True
    # Kill-switch (default on): when a streaming LLM call fails AFTER it has
    # already handed text/tool events to the consumer, raise instead of
    # restarting the request. The retry loop wraps the whole stream, so a
    # restart appends a second, independently-generated answer to the one the
    # user is already reading — and that doubled text is what gets persisted
    # and re-billed (F-12). Off -> pre-2026-07-31 replay behavior.
    llm_stream_duplicate_guard: bool = True
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
    # Default ON since 2026-08-05. It shipped dark, soaked on 59 of 61 fleet
    # containers, and the default was flipped because OFF-by-default turned a
    # missing env var into a silent, permanent regression:
    #
    #   Assigned tenants take env from /data/agents/<prefix>/.env, written at
    #   provision time. Pool containers pick up new fleet flags from the bridge
    #   env (pool_addon._feature_flag_env_args); assigned tenants never do. Two
    #   tenants — the canary and the founder's own account — therefore ran
    #   WITHOUT this flag from the day it was introduced. Nothing failed; they
    #   just quietly paid full price for every prefix.
    #
    # Measured on the canary, same container/model/wire, flag off then on:
    # 9.9% -> 92.2-93.9% of input tokens served from cache. Disable with
    # STABLE_PREFIX_LAYOUT=false.
    # Known residual (accepted): the coarse time-of-day word in about_you
    # still flips ~4x/day (morning/afternoon/evening/late-night), so expect
    # scheduled cache-hit dips near 05/12/17/22 local. OpenAI-only benefit:
    # Claude models keep the legacy intent-filtered tools array (Anthropic
    # tool_choice cannot express an allowlist).
    stable_prefix_layout: bool = True

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

    # Prefix diet (token-efficiency W2.1a; docs/audits/2026-07-remediation.md,
    # gap #6 in docs/audits/2026-07-sota-assessment.md — ~5,750 trimmable
    # tokens of the measured 27.2k wire prefix). When true:
    #   - app_builder's system-prompt essay (2,862 tok) is replaced by a
    #     ~400-tok contract (the skill's tools carry the detail at build time)
    #   - the 3 fattest tool schemas (routines__remind 998, routines__create
    #     766, triggers__create 657) serve ~250-tok descriptions; arg shapes
    #     (properties/enums/required) are byte-identical to the full schemas
    #   - platform_knowledge drops the per-tool capability blurbs that
    #     restate tool schemas and compresses Decision rules to one-liners
    #     (Pages map + NEVER list + owner fact + fencing all kept)
    #   - doc_generation shrinks 654 → ~200 tok (tool-choice rules + the
    #     convert-vs-regenerate rule kept)
    # Default ON since 2026-08-05 — same reasoning as stable_prefix_layout
    # above: it was already set on 60 of 61 fleet containers, and the one
    # container missing it (the founder's own tenant) was silently paying
    # ~5,750 extra prefix tokens on every single turn. Flag-off output stays
    # byte-identical to the pre-diet prompt (regression-pinned in
    # tests/test_prompt_diet.py); disable with PROMPT_DIET=false.
    prompt_diet: bool = True

    # Channel convergence (token-efficiency W2.3a). Today every channel gets
    # its own wire tools array (vault strip, vibecoding/app strips) — and
    # tools serialize AHEAD of system+history, so web→Telegram→voice hops
    # inside one Day-as-Chat day each start a separate provider cache
    # lineage. When true (and the stable prefix layout is active):
    #   - the wire tools array is the FULL channel-invariant set; the
    #     channel policy moves to the allowed_tools restriction (not part
    #     of the cached prefix) + executor-side refusal (defense in depth)
    #   - prompt_cache_key scope collapses from {user}:{day_chat_id} to
    #     {user}:all — the key is a routing hint, and a per-day key threw
    #     away cross-midnight head reuse for zero benefit
    # Default OFF: model-visible only via allowed_tools; flip after canary
    # A/B (same recipe as PROMPT_DIET). Flag-off is byte-identical.
    channel_converge: bool = False

    # OpenAI wire API selection (gate G1 blocker; docs/audits/2026-07-g1-model-gate.md).
    # gpt-5.6-* 400s on /v1/chat/completions when function tools are present
    # ("use /v1/responses or set reasoning_effort to 'none'" — canary abort
    # 2026-07-28). When 'responses', openai_agent_service routes through
    # client.responses.create() in stateless mode (store=false, full input
    # each turn, include=reasoning.encrypted_content) with the identical
    # StreamEvent contract, [PERF] line, metering, and retry behavior; the
    # platform proxy serves /openai/v1/responses with chat-parity accounting.
    # Values: 'chat' | 'responses'.
    #
    # 2026-08-07: this is NO LONGER the sole selector. The wire for a
    # gpt-5.6-* model is derived from the model itself
    # (model_resolver.wire_api_for) because it is a hard provider
    # constraint, not a preference — and because two settings that must be
    # flipped together are a fleet outage waiting for someone to flip one.
    # This setting now governs every OTHER family, so it stays 'chat': the
    # gpt-4o cross-provider fallback keeps the wire it has always used, and
    # that path is not exercised often enough to move it on inference.
    openai_wire_api: str = "chat"

    # Document Generation + Web Attachments (Phase 1+2 of doc-delivery feature)
    # When false: generate_* tools are not registered, attachment WS events are not emitted.
    # Frontend has its own localStorage gate (TOUP_DOC_ATTACHMENTS) for the two-pane UI.
    # Default ON as of the doc-delivery rollout — disable via FEATURE_DOC_GENERATION=false.
    feature_doc_generation: bool = True

    # Per-tenant entitlement for OPTIONAL tool families. Every container
    # ships doc_generation (1,160 tok), the app_builder skill (1,036 tok +
    # a 547-tok prompt section) and the toup dev skill (684 tok) to every
    # tenant on every turn — 3,427 tok of wire prefix most tenants never
    # use. Measured on the OpenAI chat wire, o200k_base, 2026-08-06.
    #
    # The gate lives in app/agent/tool_entitlements.py and is resolved ONCE
    # per process, because the tools array heads the provider cache prefix:
    # anything turn-conditional here forks the cache lineage and costs more
    # than it saves (see [PERF] tools_array_changed).
    #
    # "*" (and the value used when unset or blank) = every family = the
    # pre-gate behaviour, byte-identical. "none"/"-" = no optional family.
    # Otherwise a comma-separated subset.
    #
    # Delivered fleet-uniform via pool_addon's _FEATURE_FLAG_ENVS (G-15
    # close-out, 2026-08-10): a bridge-env value reaches every spawn and
    # blue-green recreate, and overrides the default below. Per-TENANT
    # values remain unwired — that needs the AgentEnvContract + the
    # hand-deployed /opt/toup-bridge/main.py — and were evaluated and
    # declined: the only per-tenant candidate, app_builder (~2,800 tok),
    # must stay reachable wherever a tenant can ask to build an app.
    #
    # ENABLED 2026-08-10 (G-15). The loadout is decided from 14 days of
    # production tool-call telemetry (Loki's full retention; 553 of 553
    # `[AGENT] Tool called:` lines parsed, attributed by channel):
    #
    #   toup            0 invocations by ANY tenant, canary included, and
    #                   no durable artifacts  -> WITHHELD (685 tok/request)
    #   app_builder     0 direct calls, BUT the founder tenant has 2 `apps`
    #                   rows and three pool tenants ran channel="app" turns;
    #                   a built app needs `app__*` at runtime -> KEPT
    #   doc_generation  real users invoked generate_pdf on voice and mobile
    #                   inside the window -> KEPT
    #
    # Withholding is not "not-default", it is UNREACHABLE: the intent
    # filter iterates loaded definitions, so a withheld family cannot be
    # merged back in for a turn that needs it (and making it
    # turn-conditional is the exact defect class tool_entitlements.py
    # exists to prevent). That asymmetry is why only the family with zero
    # observed use anywhere is withheld.
    #
    # Side effect worth the change on its own: the founder tenant's wire
    # array measured 117-129 definitions, i.e. at or over OpenAI's hard
    # 128-tool cap, where the tail is silently truncated. Dropping `toup`
    # lands it at <=124.
    #
    # Rollback: AGENT_TOOL_FAMILIES=* in the bridge env (forwarded by
    # pool_addon's _FEATURE_FLAG_ENVS) + a recreate wave; blank and "*"
    # both parse to ALL. Expect exactly one cache-lineage bust per tenant
    # on the flip, and one on the rollback.
    agent_tool_families: str = "doc_generation,app_builder"

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
    # Voice day-context DATE guard (ITEM 14). `build_realtime_instructions`
    # builds the Realtime model's own context on platform-api — a different
    # assembly path from the text runner's `load_day_context`. It asks the
    # agent for `/api/day-chats?limit=1`, which is ordered `local_date DESC`
    # (app/api/day_chats.py:205-207), and prints whatever comes back under
    # the header "# Today's Full Conversation History". When the user has
    # not spoken yet today — the ordinary voice-first morning — the newest
    # day chat is a PREVIOUS day, so the voice model is told a stale day
    # happened today. The text runner never has this problem: it resolves
    # the day from the user's local *now* (agent_runner.py:1109 →
    # resolve_day_chat_id_for_now) and creates today's row if missing.
    #
    # ON: (a) the header names the day's real date and says it is not
    # today when it isn't, and (b) the direct-date fallback asks for the
    # user's LOCAL date instead of the server's UTC date. Nothing is
    # dropped from context either way — only the label changes, and only
    # when the loaded day is not today.
    #
    # Shipped dark 2026-07-31 pending "a canary listen" that never
    # happened; flipped ON 2026-08-09 (audit G-9): nine days with the
    # wrong-day header live by default is strictly worse than the
    # label-only change this guards, and the OFF escape hatch
    # (VOICE_DAY_CONTEXT_DATE_GUARD=false) remains.
    voice_day_context_date_guard: bool = True
    # G-19a PR-B: ask the AGENT to assemble voice's instructions instead of
    # rebuilding them here from the platform's own copy of the data.
    #
    # The relay's builder reads `identities` from the PLATFORM database and
    # the day chat over HTTP, then renders them with a hand-copy of the
    # runner's persona logic. That split is where #448 (voice with no
    # persona) and #488 (voice narrating yesterday) both came from. The
    # agent-side assembler runs inside the tenant, on the same rows text
    # chat reads, and resolves the day the way every other surface does.
    #
    # Ships OFF. `voice_context_shadow` is the safer half: call BOTH, serve
    # the legacy string, and log a section-level comparison — so the two
    # can be shown to agree on real traffic before anything depends on it.
    # Any failure of the agent call falls back to the legacy builder, and
    # then to the base stub; a voice session must never open with nothing.
    voice_context_from_agent: bool = False
    voice_context_shadow: bool = True
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
    # The daily cap must gate ADMISSION of new work, never zero the cost of
    # work already paid for. Today it does the opposite: `_split_message_charge`
    # denies any charge larger than the remaining cap headroom, `try_charge`
    # writes an amount=0 "denied" ledger row, and `llm_proxy._log_event` never
    # branches on `result.success` — so the answer streams for free. A single
    # heavy gpt-5.5 call quotes 26-28 credits against a free cap of 15, so the
    # cap is UNSATISFIABLE for exactly the calls that cost the most. Production
    # 2026-08-03: 274 calls, $17.17 of provider spend, 59% of all real-user LLM
    # cost, served free with reason="daily_cap_exceeded".
    #
    # ON flips both halves of one policy:
    #   * `check_balance` consults the daily cap, so the PRE-FLIGHT can refuse
    #     a turn before any provider spend (it is cap-blind today, and is asked
    #     for 0.1 credits, so it always answers yes).
    #   * `try_charge(already_incurred=True)` lets an incurred cost land against
    #     the plan wallet even past the cap. `used_today` then climbs past the
    #     cap and the next pre-flight refuses — the plan wallet stays the hard
    #     ceiling. Nothing is ever served unbilled.
    # OFF (default) is byte-for-byte today's behavior.
    credit_cap_admission_control: bool = False
    # SHADOW MODE — measurement only, and the ONLY way to answer "what would
    # enforcement actually deny?" without denying anything.
    #
    # When True, every MESSAGE-bucket charge asks the real admission gate
    # (`_admission_verdict`, shared with `check_balance`) what it WOULD have
    # answered had both flags above been on, writes the verdict to the log as a
    # `[CREDIT-SHADOW]` line, and throws it away. Nothing is denied, no balance
    # moves, no ledger row changes — the verdict has no caller. Roll the lines
    # up with `python -m scripts.credit_shadow_rollup`.
    #
    # Why the existing signal is not enough: `try_charge` already stashes
    # `shadow_would_deny` in ledger metadata, but it is computed WITHOUT
    # `ignore_daily_cap` (which needs `credit_cap_admission_control`, off), so
    # it models the PRE-#471 policy — the one that zeroed costs already paid.
    # The 2026-08-03 "59% of real-user LLM spend served free" figure came from
    # those rows and therefore describes the broken gate, not the fixed one.
    #
    # DEFAULT OFF. Log-only either way, but flipping it starts writing one line
    # per charge carrying a user id, so it is an explicit ops decision. Turning
    # it on cannot change a single user's balance or refusal; see
    # tests/test_credit_shadow_mode.py.
    credit_shadow_admission_logging: bool = False
    # Sybil resistance: when True, the one-time free-credit grant is deduped
    # per CANONICAL email identity (Gmail dot/+alias variants AND
    # delete→re-signup collapse to a single grant) via the grant_eligibility
    # tombstone. Default False preserves the legacy "grant on every balance
    # creation" behavior while STILL recording tombstones, so enabling this
    # later is safe — the history already exists. See email_canonical.py.
    free_grant_dedupe_enabled: bool = False
    # W4.2 / gate G2 preparation (assessment A9-3 / F-11). The direct
    # (manual-mode/BYOK) OpenAI metering path historically deduped credit
    # deductions on a per-SESSION constant idempotency key (fallback: the
    # now day-scoped prompt_cache_key) — every call after the first
    # idempotent-hit the ledger and was never metered. When True, that
    # path keys each deduction on the per-REQUEST OpenAI completion id
    # (fresh UUID fallback) so every completed stream meters exactly once.
    # FORWARD-only correctness: no retroactive charges, no pricing/tier
    # changes; flag OFF keeps the legacy deduction path byte-identical
    # (regression-tested in tests/test_metering_correctness_v2.py).
    #
    # DEFAULT ON since 2026-08-10 (G-16), under the written approval in
    # docs/audits/2026-07-g2-billing-gate.md. What the flip actually moves
    # TODAY: nothing. Measured on production the same day — 44 of 44
    # agent_configs are llm_mode='bundle', and `agent_deduct`'s bundle
    # guard returns an idempotent hit BEFORE the key is consulted, so the
    # only path this flag touches has zero tenants on it. The last
    # legacy-key ledger row is from 2026-06-05. It ships ON so that the
    # first manual/BYOK tenant to appear meters per REQUEST instead of
    # once per session — the bug is closed before it can bite, not after.
    # Rollback: METERING_CORRECTNESS_V2=false in the bridge env (already
    # forwarded by pool_addon's _FEATURE_FLAG_ENVS) + a recreate wave.
    metering_correctness_v2: bool = True
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
        "BRAVE_API_KEY",             # tenant BYOK only — the platform key no
                                     # longer ships to containers (search
                                     # gateway holds it); kept for containers
                                     # that still carry a stale or BYOK value
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
    # Hard wall-clock budget for the whole rerank attempt (Cohere AND the
    # LLM fallback share it). On expiry the search degrades to hybrid
    # (RRF+weighted) order — never fails, never blocks the turn.
    #
    # 900ms is sized for COHERE, which is purpose-built and answers in
    # ~100-300ms. It is deliberately NOT enough for the gpt-4o-mini
    # fallback — see reranker_llm_fallback_enabled for why that is a
    # setting and not a race.
    reranker_timeout_ms: int = 900
    # The gpt-4o-mini fallback re-ranker, OFF on the turn path.
    #
    # Measured in production (Loki, 7d, 2026-08-10): the fallback really
    # does run — three successful re-ranks on the founder's tenant at
    # 2754ms, 2776ms and 4054ms. That is the G-18 latency, and it was not
    # a test artefact: an earlier read of `docker logs --since 168h` found
    # "zero" only because the fleet had been recreated minutes before, so
    # the window could not reach back past the container's own birth.
    #
    # Given it is real, the choice is explicit rather than accidental.
    # 2.7-4s of turn latency for a precision improvement over hybrid order
    # is a bad trade on a path a user is waiting on, and the alternative
    # the budget produced on its own was the WORST of the three: start the
    # hop, burn the full 900ms, then throw the answer away and return
    # hybrid order anyway (4 of 4 attempts after the budget shipped). Dead
    # latency for nothing.
    #
    # So the fallback is opt-in. Default OFF means hybrid order is reached
    # immediately and honestly. Turning it on is a considered decision to
    # buy precision with seconds; the real fix for precision is a COHERE
    # key, which fits inside the budget.
    reranker_llm_fallback_enabled: bool = False
    
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
    # When False, the platform MCP server logs unauthenticated requests as
    # warnings but lets them through. When True, missing or invalid
    # X-Agent-Key returns 401 from the transport before any tool dispatch.
    #
    # Flipped to True 2026-08-09 (audit G-10) on the soak evidence the
    # warn-only mode existed to produce: 0 "would-reject in enforce mode"
    # warnings across 20 production containers (all 5 assigned + 15 pool)
    # over 7 days of logs. Nothing observed breaks; an unauthenticated
    # future caller SHOULD 401 — that is the point. Escape hatch:
    # MCP_REQUIRE_X_AGENT_KEY=false.
    mcp_require_x_agent_key: bool = True

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
    # G-19b: routes email_received trigger turns through AgentRunner.run
    # (day context + memory + persona + attributable metering) instead of
    # the bare call_system_llm summarize. Ships default-OFF; ops flips it
    # later via env TRIGGER_TURNS_VIA_RUNNER. Fail-open: any runner
    # failure falls back to the summarize path so a trigger never goes
    # silent because the new path broke.
    trigger_turns_via_runner: bool = False
    # Per-fire runner credit ceiling (credits; 1 credit = 1 cent).
    # Enforced by AgentRunner's in-loop budget hard-stop
    # (credit_budget kwarg) — a runaway tool loop on an unattended
    # turn stops here, not at the user's monthly balance.
    trigger_turn_credit_budget: float = 25.0

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
    # Upgrade endpoint can take longer (pull + recreate + health). Must exceed
    # the bridge's green-health ceiling (240 s as of 2026-07-28) + drain (60 s
    # max) + pull/swap overhead: at 180 s the platform ReadTimeout'd while the
    # bridge finished the swap anyway, recording false rolled_back attempts
    # and burning a convergence-sweep slot (canary 533354ce, 2026-07-29).
    bridge_upgrade_timeout_s: int = 330

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
    # Optional SECOND egress for the audio pipeline. When set, `yt_dlp_proxy`
    # is the PRIMARY (a fast static ISP IP) and this is the fallback the
    # pipeline retries through when the primary refuses a CONNECT with 403.
    # IPRoyal static (ISP) proxies are IP-BOUND — one source IP at a time,
    # held for a ~10s sticky window from connection start (measured
    # 2026-08-10) — and platform-api runs 2 replicas with distinct egress
    # IPs, so concurrent cold plays on different replicas lock each other
    # out of a static primary. Same format as yt_dlp_proxy; point it at the
    # rotating residential gateway (sticky-session tokens supported). Unset =
    # single-proxy behaviour through whatever yt_dlp_proxy points at — NOTE
    # that with prod's yt_dlp_proxy repointed at the static ISP IP, the full
    # ROLLBACK to the pre-tier world is "yt_dlp_proxy = the residential
    # gateway URL, this unset", never just unsetting this field. The two URLs
    # must DIFFER: an equal pair is treated as no fallback
    # (media_proxy._fallback_available), because retrying through the same
    # egress doubles every honest failure and makes the logs lie.
    yt_dlp_proxy_fallback: Optional[str] = None

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
    # How long a rollout waits for the bridge's pool to stop recycling before
    # it touches the canary. A COMPLETED rollout notifies the bridge to refresh
    # the pool image, and the bridge then recycles every pool member — measured
    # 2026-08-01, 49 of 50 members inside 30 minutes. A rollout that starts in
    # that window contends with it and its canary upgrade dies in whichever way
    # the contention bites: ConnectError, a stale heartbeat -> aborted_orphan,
    # or 0 health checks in 259s. 15 min covers the observed churn; on timeout
    # the rollout PROCEEDS and says so, because a late rollout beats a wedged
    # one. Set 0 to disable the wait entirely.
    # 300s, not 900: the wait now exits early on a wedged pool (see
    # wait_for_pool_quiescence), so this is only the backstop, and a rollout
    # that has not started after five minutes is worse than one that races.
    rollout_pool_quiesce_timeout_s: int = 300
    # Convergence sweep (2026-07-28 incident: rollout 01a945e2's driver was
    # killed by its own merge's Railway redeploy; the re-drive d79584ea hit
    # the bridge mid-restart — 502 — on tenant 2739b5c6, reported 'complete',
    # and left a real beta user silently on the OLD image for ~40 min). When
    # true, every reconciler tick with no active rollout compares eligible
    # running tenants' image_tag against the most recent complete /
    # complete_with_failures rollout's tag and re-drives only the divergent
    # ones through a trigger='sweep' rollout — bounded at 3 sweeps per tag
    # per 24h, then a critical alert. Default ON — this is a KILL SWITCH,
    # not a rollout gate: flip via ROLLOUT_CONVERGENCE_SWEEP=false if a
    # sweep ever misbehaves. Pinned in tests/test_rollout_convergence_sweep.py.
    rollout_convergence_sweep: bool = True
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
    # G-20: per-tenant request cap on the LLM proxy (sliding 60s window,
    # ALL endpoints share one budget — chat, responses and embeddings).
    #
    # Sized from the real distribution, re-measured 2026-08-10 after an
    # earlier figure of "33/min peak" turned out to be a narrower slice.
    # Over 30 days and 1,072 active user-minutes the true per-user maxima
    # are: 80, 59, 33, 33, 32, ... with p99.9 = 57 and exactly ONE minute
    # above 60.
    #
    # So the observed ceiling is 80, and the first value shipped here (90)
    # sat only 12% above it — which is the wrong shape for this control.
    # What it defends against is a leaked or runaway token, and that does
    # not look like 1.2x normal traffic; it looks like thousands of calls
    # a minute. A margin that thin buys almost no safety and risks real
    # 429s, and an embedding 429 is not free: create_memory stores the row
    # UNEMBEDDED rather than failing, so the memory survives but silently
    # drops to keyword-only retrieval.
    #
    # A true sliding-60s measurement (not calendar-minute buckets) puts the
    # real ceiling higher still: 93, set by this run's own load harness at
    # 5 virtual users — i.e. the first value shipped here (90) was already
    # below traffic WE generated on purpose.
    #
    # 300 is ~3.2x that — still an instant stop for a runaway, with enough
    # headroom that a legitimate burst (a memory backfill fanning out
    # embeddings) is not quietly degraded. Note the window is per PROCESS
    # and platform-api runs 2 replicas, so the fleet-effective ceiling is
    # up to 600/min; that is acceptable for a runaway backstop and would
    # NOT be acceptable for a quota. 0 disables.
    llm_proxy_rate_limit_per_min: int = 300

    # Pricing per 1K tokens (USD)
    #
    # `cached_input` / `cache_write` are OPTIONAL per-model columns (G1 prep,
    # docs/audits/2026-07-g1-model-gate.md): the gpt-5.6 explicit-caching
    # regime bills cache READS at the cached_input rate and cache WRITES at
    # 1.25x the input rate (terra $3.125/M). Cost math (_calc_cost_cents,
    # tokens_to_credits) only applies these columns when present, so every
    # model WITHOUT them (all currently-live models) bills exactly as before.
    # gpt-5.5-pro is deliberately ABSENT: it has NO cached-input rate at all
    # ($30/$180) and is barred from chat/agent selection by the resolver
    # guard (model_resolver.has_cached_input_rate).
    pricing_per_1k: dict[str, dict[str, float]] = {
        # cached_input MEASURED 2026-08-07 against OpenAI organization billing
        # (billed `cached input` line / cached tokens, 14d): $0.5493/M, which
        # is 0.098x the same window's uncached rate — i.e. the published 10%.
        # Its absence billed every cached 5.5 token at the FULL input rate, and
        # 56.8% of 5.5's input comes back cached.
        "gpt-5.5": {"input": 0.005, "cached_input": 0.0005, "output": 0.030},
        # cache_write was 0.003125 (a modelled 1.25x input). MEASURED, that
        # line prices at $2.548/M — terra's LIST INPUT rate — while the
        # separate `terra, input` line is $0.0045, i.e. nil. OpenAI is not
        # surcharging cache writes; it files terra's ordinary uncached input
        # under that label. See docs/audits/2026-08-g1-cost-and-latency.md 8.2.
        "gpt-5.6-terra": {"input": 0.0025, "cached_input": 0.00025, "cache_write": 0.0025, "output": 0.015},
        # sol/luna keep their MODELLED 1.25x cache_write: neither has carried
        # production traffic, so there is nothing to measure. Do not "correct"
        # them by analogy with terra — that would be inventing a number.
        "gpt-5.6-sol": {"input": 0.005, "cached_input": 0.0005, "cache_write": 0.00625, "output": 0.030},
        "gpt-5.6-luna": {"input": 0.001, "cached_input": 0.0001, "cache_write": 0.00125, "output": 0.006},
        "gpt-5.4": {"input": 0.003, "output": 0.012},
        "gpt-5": {"input": 0.003, "output": 0.012},
        "gpt-4.1": {"input": 0.002, "output": 0.008},
        "gpt-4o": {"input": 0.0025, "output": 0.01},
        # cached_input MEASURED 2026-08-07: $0.0750/M = exactly 0.500x the
        # measured uncached rate ($0.1502/M), matching the published $0.075/M.
        # 4o-mini is the memory-extraction model, so it runs a very high cache
        # hit rate (43.3% of 46.2M input tokens in 14 days) — this was the
        # second-largest instance of the same missing-column defect.
        "gpt-4o-mini": {"input": 0.00015, "cached_input": 0.000075, "output": 0.0006},
        "claude-opus-4-6": {"input": 0.015, "output": 0.075},
        # The Anthropic CANONICAL (model_resolver._CANONICAL_ANTHROPIC_MODEL)
        # had a context-window row (A8-3) but no pricing row, so if Anthropic
        # is re-enabled it would fall through _calc_cost_cents' conservative
        # default and bill Opus at Sonnet rates (5x under). MODELLED at the
        # opus-4-6 rate, not measured — correct against real org billing
        # before trusting any 4-7 cost figure (the #509 lesson).
        "claude-opus-4-7": {"input": 0.015, "output": 0.075},
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
    def _normalize_platform_api_url(self) -> "Settings":
        """Guarantee `platform_api_url` ends with the API prefix.

        Every agent→platform callback is built as f"{platform_api_url}/...",
        and every one of those routes is mounted under `api_prefix`. When the
        configured base omits it the request lands on the SPA catch-all, which
        serves index.html for GET and **405** for POST — a failure that reads
        like a broken feature rather than a broken URL.

        This has bitten three separate subsystems. `tool_executor` and
        `subagent_dispatcher` each grew their own local retry-with-/api after
        being caught by it (see the 2026-05-24 note in tool_executor), and on
        2026-07-31 it silently broke radio station building for an entire
        evening — 23 consecutive 405s, every one falling back to the YouTube
        Music call that is anti-bot-blocked from agent IPs, which is the whole
        reason the platform RPC exists. The code default is correct
        ("https://toup.ai/api"); the deployed value is not, and the
        provisioning bridge that sets it is not reachable from here. Fixing it
        at load time repairs every call site at once, including ones not yet
        written.

        Appends `api_prefix` rather than a hard-coded "/api" so a deployment
        that serves the API under a different prefix stays correct, and no-ops
        when the base already ends with it.
        """
        base = (self.platform_api_url or "").strip().rstrip("/")
        prefix = (self.api_prefix or "").strip().rstrip("/")
        if base and prefix and not base.endswith(prefix):
            object.__setattr__(self, "platform_api_url", f"{base}{prefix}")
        elif base != (self.platform_api_url or ""):
            object.__setattr__(self, "platform_api_url", base)
        return self

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
