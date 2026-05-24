"""Database connection and session management.

Phase A (never-sleep plan) added late-bind support: the agent now
starts with `TOUP_POOL_GENERIC=1` pointed at a generic pool DB, and
on `POST /admin/bind` the engine is rebound to the renamed tenant DB.
The module-level `engine` and `async_session_maker` are wrapped in a
proxy/holder so existing `from app.db.database import async_session_maker`
imports continue to resolve correctly through a rebind without
touching 141 call sites.
"""

import logging
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, AsyncEngine, async_sessionmaker
from sqlalchemy.pool import StaticPool, NullPool
from app.config import settings
from app.db.models import Base

logger = logging.getLogger(__name__)


def _build_engine(database_url: str) -> AsyncEngine:
    """Construct an AsyncEngine for the given URL using the same
    knobs as the original module-level setup. Centralized so
    `rebind_database()` builds an identical engine to the one created
    at import."""
    if database_url.startswith("sqlite"):
        return create_async_engine(
            database_url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            echo=settings.debug,
        )
    if settings.run_mode in ("platform", "agent"):
        _db_url = database_url
        _sep = "&" if "?" in _db_url else "?"
        _db_url += f"{_sep}prepared_statement_cache_size=0"
        return create_async_engine(
            _db_url,
            echo=settings.debug,
            poolclass=NullPool,
            pool_pre_ping=True,
            connect_args={
                "statement_cache_size": 0,
                "prepared_statement_name_func": lambda: "",
                "command_timeout": 30,
                "server_settings": {"statement_timeout": "30000"},
            },
        )
    return create_async_engine(
        database_url,
        echo=settings.debug,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
    )


def _build_sessionmaker(eng: AsyncEngine):
    return async_sessionmaker(
        eng,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )


# ── Holder + proxy ─────────────────────────────────────────────────
# Single-element registry, mutated on rebind. Modules that imported
# `engine` directly continue to see the OLD instance — that's why
# `rebind_database` reassigns the module attribute below as well, and
# why direct imports of `engine` are limited to a handful of sites
# (refactored to use `get_engine()` in Phase A).
_holder: dict = {"engine": None, "session_maker": None}


class _SessionMakerProxy:
    """Thin proxy so `from app.db.database import async_session_maker`
    continues to work after a rebind. The proxy resolves the current
    sessionmaker on every call.

    Calls that look like `async_session_maker()` build a session via
    the proxy's `__call__`; the resulting session uses whichever
    engine is currently in the holder."""

    def __call__(self, *args, **kwargs):
        sm = _holder["session_maker"]
        if sm is None:
            raise RuntimeError(
                "async_session_maker called before engine init. "
                "If you're in pool-lobby mode, this likely means a route "
                "that needs the DB ran before /admin/bind."
            )
        return sm(*args, **kwargs)


# Initialize the holder at import time using the configured URL.
engine: AsyncEngine = _build_engine(settings.database_url)
_holder["engine"] = engine
_holder["session_maker"] = _build_sessionmaker(engine)
async_session_maker = _SessionMakerProxy()


def get_engine() -> AsyncEngine:
    """Return the current engine. Use this instead of importing
    `engine` directly when you need a guaranteed-live reference (the
    direct import captures the engine at module-import time, which
    becomes stale after a pool→bound rebind)."""
    return _holder["engine"]


async def rebind_database(new_database_url: str) -> None:
    """Atomically swap the engine + sessionmaker to a new database URL.

    Called by `POST /admin/bind` when a pool container is claimed for a
    user. The flow is:
    1. Build a new engine at the new URL.
    2. Smoke-test: open a connection (catches credential / DB-name typos
       BEFORE we tear down the live engine). Failure here leaves the
       old engine intact — caller treats this as bind failure.
    3. Dispose the old engine.
    4. Replace holder + module attribute.

    This function is single-shot in practice (generic→bound happens
    once per container lifetime). Concurrent callers are not supported;
    `/admin/bind` is the only caller and is itself serialized by
    `pool_member.state='ASSIGNING'` on the bridge side.
    """
    global engine
    logger.info("[database] Rebinding engine to new URL")
    new_engine = _build_engine(new_database_url)
    # Smoke-test the new connection. asyncpg raises OperationalError on
    # bad credentials / missing DB / unreachable host. We propagate the
    # error so /admin/bind returns 500 and the bridge moves the pool
    # member to TEARDOWN — no half-bound state.
    async with new_engine.connect() as conn:
        from sqlalchemy import text
        await conn.execute(text("SELECT 1"))

    old_engine = _holder["engine"]
    _holder["engine"] = new_engine
    _holder["session_maker"] = _build_sessionmaker(new_engine)
    engine = new_engine
    logger.info("[database] Engine rebound; disposing old engine")
    if old_engine is not None:
        try:
            await old_engine.dispose()
        except Exception as e:
            # Old engine disposal is best-effort. A leaked connection
            # at the OS level here is preferable to crashing the bind
            # — pool DBs have unused connections only.
            logger.warning("[database] Old engine dispose failed: %s", e)


async def get_db() -> AsyncSession:
    """Dependency for getting database sessions"""
    async with async_session_maker() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db():
    """Initialize database tables and add any missing columns.

    Tables are partitioned by run_mode (see base.py):
      - platform: creates PLATFORM_ONLY + SHARED tables
      - agent: creates AGENT_ONLY + SHARED tables
      - monolith: creates all tables
    """
    from sqlalchemy import text, inspect as sa_inspect
    import logging
    _logger = logging.getLogger(__name__)

    from app.db.models.base import AGENT_ONLY_TABLES, PLATFORM_ONLY_TABLES, SHARED_TABLES

    # Determine which tables belong in this database
    _run_mode = settings.run_mode
    if _run_mode == "platform":
        _excluded = AGENT_ONLY_TABLES
    elif _run_mode == "agent":
        _excluded = PLATFORM_ONLY_TABLES
    else:  # monolith
        _excluded = set()

    _allowed_tables = [t for t in Base.metadata.sorted_tables if t.name not in _excluded]

    # Ensure pgvector extension exists before create_all tries to use VECTOR columns
    _has_pgvector = False
    async with engine.begin() as conn:
        try:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            _has_pgvector = True
        except Exception:
            _logger.warning("pgvector extension not available — vector columns will be skipped")

    async with engine.begin() as conn:
        try:
            await conn.run_sync(Base.metadata.create_all, tables=_allowed_tables)
        except Exception as _e:
            if "vector" in str(_e).lower() and not _has_pgvector:
                _logger.warning("create_all failed due to missing vector type — creating tables individually")
            else:
                raise

    # If create_all failed due to missing vector, create tables one by one
    # in a FRESH connection (the old transaction is in an aborted state).
    if not _has_pgvector:
        _vector_tables = set()
        for table in _allowed_tables:
            for col in table.columns:
                if "vector" in str(col.type).lower():
                    _vector_tables.add(table.name)
                    break

        async with engine.begin() as conn:
            for table in _allowed_tables:
                if table.name in _vector_tables:
                    _logger.info("Skipping table %s (needs pgvector)", table.name)
                    continue
                try:
                    await conn.run_sync(table.create, checkfirst=True)
                except Exception:
                    _logger.warning("Failed to create table %s", table.name)

    # ── Runtime safety assertion ──────────────────────────────────
    # Verify that forbidden tables were NOT created in this DB.
    # Catches the case where someone adds a model and forgets to
    # update the partition sets in base.py.
    try:
        async with engine.connect() as conn:
            _existing = await conn.run_sync(lambda sync_conn: sa_inspect(sync_conn).get_table_names())
        _existing_set = set(_existing)
        if _run_mode == "platform":
            _leaked = _existing_set & AGENT_ONLY_TABLES
            if _leaked:
                # Warn but don't crash — legacy monolith DBs have all tables.
                # New tables won't be created (filtered above), but old ones remain.
                _logger.warning(
                    "Agent-only tables found in platform DB (legacy monolith): %s. "
                    "These are harmless leftovers — new agent tables won't be created.",
                    _leaked,
                )
        elif _run_mode == "agent":
            _leaked = _existing_set & PLATFORM_ONLY_TABLES
            if _leaked:
                _logger.warning(
                    "Platform-only tables found in agent DB (legacy monolith): %s. "
                    "These are harmless leftovers — new platform tables won't be created.",
                    _leaked,
                )
    except RuntimeError:
        raise
    except Exception as _e:
        _logger.warning("Could not verify table partitioning: %s", _e)

    # Add missing columns to existing tables (create_all only creates new tables).
    #
    # IMPORTANT: All ALTER statements run in ALL modes. The table partitioning
    # (AGENT_ONLY / PLATFORM_ONLY) only gates table CREATION, not column additions.
    # Agent DBs are monolith-style — they contain all tables including platform ones
    # (agent_configs, vps_plans, etc.) from before the partitioning was introduced.
    # ALTER TABLE ... IF NOT EXISTS is safe on any table whether or not it "belongs"
    # to this run_mode. Trying to ALTER a non-existent table just silently fails
    # in the try/except below.

    _alter_statements = [
        # ── Users ──
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS password_changed_at TIMESTAMP",
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS timezone VARCHAR(50)",
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS stripe_customer_id VARCHAR(255) UNIQUE",
        # Email/notification toggles. JSONB so we can add new keys
        # without an ALTER per toggle. NULL on existing rows is
        # interpreted as DEFAULT_NOTIFICATION_PREFERENCES by the
        # account preferences endpoint.
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS notification_preferences JSONB",
        # Email verification (alembic 055). Mirrored here because the
        # agent's boot path runs init_db() but NOT alembic upgrade —
        # without these ALTERs, every endpoint that loads a User row
        # 500s with "column users.email_verified_at does not exist"
        # the moment an agent is rolled to a build that references the
        # new column on the ORM model. Self-healing on next boot.
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS email_verified_at TIMESTAMP",
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS email_verification_token VARCHAR(64)",
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS email_verification_sent_at TIMESTAMP",
        # ── Agent configs ──
        "ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS llm_mode VARCHAR(20) DEFAULT 'manual'",
        # Mig 057 (onboarding-v2 PR 2) — backfill column preserves the
        # pre-v2 llm_mode value so the migration is reversible. Per the
        # READ FIRST memory: any alembic column added on a shared
        # platform table must mirror here so agent boots that ran
        # `create_all` against an older snapshot self-heal on restart.
        "ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS llm_mode_pre_v2 VARCHAR(20)",
        "ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS google_api_key TEXT",
        "ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS mistral_api_key TEXT",
        "ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS groq_api_key TEXT",
        "ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS xai_api_key TEXT",
        "ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS deepseek_api_key TEXT",
        "ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS bundle_stripe_subscription_id VARCHAR(255)",
        "ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS bundle_status VARCHAR(20) DEFAULT 'none'",
        "ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS bundle_started_at TIMESTAMP",
        "ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS bundle_current_period_end TIMESTAMP",
        # Managed-container reaper input — see scheduled_tasks.run_managed_container_reaper.
        "ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS bundle_cancelled_at TIMESTAMP",
        "ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS onboarding_completed BOOLEAN DEFAULT FALSE",
        "ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS agent_color VARCHAR(7)",
        "ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS disabled_tools TEXT DEFAULT ''",
        "ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS connect_token VARCHAR(100)",
        "ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS agent_name VARCHAR(100)",
        "ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS db_mode VARCHAR(20) DEFAULT 'auto'",
        "ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS supabase_url TEXT",
        "ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS target_os VARCHAR(20)",
        "ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS setup_type VARCHAR(20)",
        # LLM proxy auth + budget (migration 020)
        "ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS llm_token_hash VARCHAR(128)",
        "ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS bundle_anthropic_budget_cents INTEGER DEFAULT 3000",
        "ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS bundle_openai_budget_cents INTEGER DEFAULT 1000",
        "ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS bundle_anthropic_daily_cap_cents INTEGER DEFAULT 100",
        "ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS bundle_period_start TIMESTAMP",
        "ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS bundle_period_end TIMESTAMP",
        # Auto-builder Planner/Builder split (migration 029).
        # NULL means "fall through to agent_model" — see model_resolver.app_builder_*_model().
        "ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS app_builder_planner_model VARCHAR(50)",
        "ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS app_builder_builder_model VARCHAR(50)",
        # Drop the stale agent_model server_default (was 'gpt-5.2' in migration 015).
        # Existing rows are untouched here; the matching alembic migration 029 does
        # the gpt-5.2 → NULL backfill on platform DBs.
        "ALTER TABLE agent_configs ALTER COLUMN agent_model DROP DEFAULT",
        # WhatsApp BYOA (migration 030). Per-tenant Meta Cloud API
        # credentials and connection status. Mirrors the alembic
        # migration so platform boots that skip alembic still get the
        # columns. All three nullable so existing rows remain valid.
        "ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS whatsapp_verify_token VARCHAR(100)",
        "ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS whatsapp_app_secret VARCHAR(200)",
        "ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS whatsapp_connected_at TIMESTAMP",
        # WhatsApp QR-link mode (migration 032). Per-tenant transport
        # mode, dedicated phone identity, allowlist of senders, and the
        # current session lifecycle state. NULL whatsapp_mode = the
        # tenant hasn't picked a WhatsApp transport yet.
        "ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS whatsapp_mode VARCHAR(20)",
        "ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS whatsapp_self_e164 VARCHAR(20)",
        "ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS whatsapp_baileys_allowlist TEXT",
        "ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS whatsapp_session_status VARCHAR(20)",
        # Toup Code (experimental Claude Code IDE, 2026-05-12). Long-lived
        # OAuth token from `claude setup-token`; nullable so existing
        # rows stay valid. 2000 chars to fit refresh-token-embedded
        # payloads without truncation.
        "ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS claude_code_oauth_token VARCHAR(2000)",
        # Toup Code dual-provider — OpenAI Codex CLI sibling (2026-05-13,
        # commit af73c47). Migration 039 is the canonical add path; this
        # init_db ALTER is the safety net because the Dockerfile CMD was
        # recently changed (commit 06f550a) to NOT block uvicorn on
        # alembic failures — so if alembic crashes for any reason, the
        # column never appears via migrations and every SELECT from
        # agent_configs 500s with "column does not exist". This safety
        # net mirrors the claude_code_oauth_token entry above and
        # guarantees schema-completeness regardless of alembic state.
        "ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS openai_codex_token VARCHAR(2000)",
        # Sub-agent spawning kill-switch (alembic 057). Mirrored here
        # per the READ-FIRST rule in MEMORY.md — agents boot via
        # init_db, NOT alembic upgrade. Without this, every SELECT
        # from agent_configs 500s with "column agent_configs.
        # subagent_spawning_enabled does not exist" the moment an
        # agent rolls to a build that references the new column.
        "ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS subagent_spawning_enabled BOOLEAN NOT NULL DEFAULT FALSE",
        # ── Connector identities ──
        # Per-identity read-only switch (2026-05-11). True means the
        # MCP tool filter drops every manifest tool with mutates=true
        # and the dispatcher refuses any mutating tool call. Toggled
        # from the connected-card "Switch to read-only" action.
        "ALTER TABLE connector_identities ADD COLUMN IF NOT EXISTS read_only BOOLEAN NOT NULL DEFAULT FALSE",
        # ── VPS ──
        "ALTER TABLE vps_plans ADD COLUMN IF NOT EXISTS provider VARCHAR(20) DEFAULT 'aws'",
        "ALTER TABLE vps_plans ADD COLUMN IF NOT EXISTS hostinger_plan_id VARCHAR(50)",
        "ALTER TABLE vps_plans ADD COLUMN IF NOT EXISTS hetzner_server_type VARCHAR(20)",
        "ALTER TABLE vps_instances ADD COLUMN IF NOT EXISTS provider VARCHAR(20) DEFAULT 'aws'",
        "ALTER TABLE vps_instances ADD COLUMN IF NOT EXISTS hostinger_vm_id VARCHAR(50)",
        "ALTER TABLE vps_instances ADD COLUMN IF NOT EXISTS hetzner_vm_id VARCHAR(50)",
        # ── Build jobs ──
        # Build logs for app builder jobs
        "ALTER TABLE build_jobs ADD COLUMN IF NOT EXISTS build_logs_json TEXT DEFAULT '[]'",
        # Approved plan from conversational app builder
        "ALTER TABLE apps ADD COLUMN IF NOT EXISTS plan_json TEXT",
        # Token limit pause/resume for build jobs
        "ALTER TABLE build_jobs ADD COLUMN IF NOT EXISTS paused_at TIMESTAMP",
        "ALTER TABLE build_jobs ADD COLUMN IF NOT EXISTS resume_after TIMESTAMP",
        "ALTER TABLE build_jobs ADD COLUMN IF NOT EXISTS checkpoint_json TEXT",
        # Layer 2 support
        "ALTER TABLE build_jobs ADD COLUMN IF NOT EXISTS layer INTEGER DEFAULT 1",
        "ALTER TABLE build_jobs ADD COLUMN IF NOT EXISTS layer2_changes_json TEXT",
        # VPS soul sync tracking
        "ALTER TABLE soul_configs ADD COLUMN IF NOT EXISTS vps_soul_synced_at TIMESTAMP",
        # Rich content metadata on messages (media cards, etc.)
        "ALTER TABLE messages ADD COLUMN IF NOT EXISTS metadata_json TEXT",
        # Per-message channel tag (time-channel-fix PR). Denormalized at
        # write time from the ingress handler, not from conversations.channel,
        # so the day-as-chat history can render channel transitions within
        # a single day. See channel_util.resolve_channel.
        "ALTER TABLE messages ADD COLUMN IF NOT EXISTS channel VARCHAR(50)",
        "CREATE INDEX IF NOT EXISTS ix_messages_channel ON messages (channel)",
        # Cross-channel reply-to pointer (migration 049). Soft pointer —
        # no FK — so cleaning up stale day chats doesn't cascade through
        # replies. Index lets the frontend resolve a referenced row in
        # one shot when rendering the quoted-message card.
        "ALTER TABLE messages ADD COLUMN IF NOT EXISTS reply_to_message_id VARCHAR(50)",
        "CREATE INDEX IF NOT EXISTS ix_messages_reply_to ON messages (reply_to_message_id)",
        # Idempotent backfill: messages.channel = conversations.channel when
        # messages.channel is NULL. Safe to re-run; WHERE clause prevents
        # re-writing rows that already have a value. Rows where BOTH are
        # NULL stay NULL (don't guess).
        """
        UPDATE messages m SET channel = c.channel
        FROM conversations c
        WHERE m.conversation_id = c.id
          AND m.channel IS NULL
          AND c.channel IS NOT NULL
        """,
        # Reconciliation: app source tracking
        "ALTER TABLE apps ADD COLUMN IF NOT EXISTS source VARCHAR(30) DEFAULT 'app_builder'",
        "UPDATE apps SET source = 'vibecoding' WHERE source = 'app_builder' AND app_dir LIKE '%/vibecoding/%'",
        # Builder mode per-chat
        "ALTER TABLE conversations ADD COLUMN IF NOT EXISTS builder_mode VARCHAR(10)",
        # Reconciliation log cleanup
        "DELETE FROM reconciliation_logs WHERE created_at < NOW() - INTERVAL '30 days'",
        # Job type classification (auto_builder, vibe_code, agent_task)
        "ALTER TABLE build_jobs ADD COLUMN IF NOT EXISTS job_type VARCHAR(20) DEFAULT 'auto_builder'",
        # Backfill: existing rows without job_type get auto_builder
        "UPDATE build_jobs SET job_type = 'auto_builder' WHERE job_type IS NULL",
        # Unified-jobs cutover (alembic 046 + 051 + 052). Mirrored here
        # because agents boot via init_db, not alembic upgrade — without
        # these ALTERs, /api/apps/jobs/ and any other build_jobs reader
        # 500s with "column build_jobs.<X> does not exist" the moment an
        # agent rolls to a build that references the new column. The
        # cascade then surfaces as the "Reconnecting…" banner because
        # the frontend's idempotent /api/apps/jobs/ poll keeps retrying
        # 502s. Self-healing on next boot.
        # 046 — unified-jobs discriminator + back-links
        "ALTER TABLE build_jobs ADD COLUMN IF NOT EXISTS source_kind VARCHAR(20)",
        "ALTER TABLE build_jobs ADD COLUMN IF NOT EXISTS source_id VARCHAR(36)",
        "ALTER TABLE build_jobs ADD COLUMN IF NOT EXISTS conversation_id VARCHAR(36)",
        "ALTER TABLE build_jobs ADD COLUMN IF NOT EXISTS summary_message_id VARCHAR(50)",
        "ALTER TABLE build_jobs ADD COLUMN IF NOT EXISTS outcome VARCHAR(30)",
        "ALTER TABLE build_jobs ADD COLUMN IF NOT EXISTS idempotency_key VARCHAR(120)",
        # 051 — runner-state columns
        "ALTER TABLE build_jobs ADD COLUMN IF NOT EXISTS fire_instant TIMESTAMP",
        "ALTER TABLE build_jobs ADD COLUMN IF NOT EXISTS attempt INTEGER",
        "ALTER TABLE build_jobs ADD COLUMN IF NOT EXISTS coalesced_into_job_id VARCHAR(36)",
        # 052 — routine-terminal columns
        "ALTER TABLE build_jobs ADD COLUMN IF NOT EXISTS emails_fetched INTEGER",
        "ALTER TABLE build_jobs ADD COLUMN IF NOT EXISTS finished_local_at VARCHAR(40)",
        "ALTER TABLE build_jobs ADD COLUMN IF NOT EXISTS error_json JSONB",
        "ALTER TABLE build_jobs ADD COLUMN IF NOT EXISTS channel_results_json JSONB",
        "ALTER TABLE build_jobs ADD COLUMN IF NOT EXISTS tools_invoked_json JSONB",
        # 056 — sub-agent spawning columns. Mirrored here per the
        # READ-FIRST rule in MEMORY.md: agents boot via init_db, NOT
        # alembic upgrade. Without these, the moment an agent rolls
        # to a build that references BuildJob.parent_job_id /
        # credit_spent on the ORM model, every SELECT from build_jobs
        # 500s. config_json is JSONB on PG; SQLite tolerates JSONB as
        # an alias for TEXT so the same DDL works in both dialects.
        # credit_spent has NOT NULL DEFAULT 0.0 to match the migration.
        "ALTER TABLE build_jobs ADD COLUMN IF NOT EXISTS parent_job_id VARCHAR(36)",
        "ALTER TABLE build_jobs ADD COLUMN IF NOT EXISTS config_json JSONB",
        "ALTER TABLE build_jobs ADD COLUMN IF NOT EXISTS credit_budget_allocated DOUBLE PRECISION",
        "ALTER TABLE build_jobs ADD COLUMN IF NOT EXISTS credit_spent DOUBLE PRECISION NOT NULL DEFAULT 0.0",
        "CREATE INDEX IF NOT EXISTS ix_build_jobs_parent_status ON build_jobs (parent_job_id, status)",
        # Day-as-Chat: FK from conversations/messages to day_chats.
        # On agent (day_chats exists): add with FK constraint.
        # On platform (day_chats missing): add without FK so ORM queries don't crash.
        "DO $$ BEGIN "
        "IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name='day_chats') THEN "
        "EXECUTE 'ALTER TABLE conversations ADD COLUMN IF NOT EXISTS day_chat_id VARCHAR(36) REFERENCES day_chats(id)'; "
        "EXECUTE 'ALTER TABLE messages ADD COLUMN IF NOT EXISTS day_chat_id VARCHAR(36) REFERENCES day_chats(id)'; "
        "ELSE "
        "EXECUTE 'ALTER TABLE conversations ADD COLUMN IF NOT EXISTS day_chat_id VARCHAR(36)'; "
        "EXECUTE 'ALTER TABLE messages ADD COLUMN IF NOT EXISTS day_chat_id VARCHAR(36)'; "
        "END IF; END $$",
        # Day-as-Chat: composite index for hot-path query
        "CREATE INDEX IF NOT EXISTS ix_messages_day_chat_created ON messages(day_chat_id, created_at)",
        # Day Recall + LLM proxy operation tagging (migration 021).
        # These run on every startup so existing agent DBs self-heal on container
        # restart without needing manual psql. New DBs get the columns from
        # Base.metadata.create_all() via the ORM model; these ALTERs are no-ops
        # there. See docs/day-recall-system.md.
        "ALTER TABLE llm_proxy_events ADD COLUMN IF NOT EXISTS operation_type VARCHAR(50)",
        "CREATE INDEX IF NOT EXISTS ix_llm_proxy_operation_type ON llm_proxy_events (operation_type)",
        "ALTER TABLE day_chats ADD COLUMN IF NOT EXISTS archival_summary TEXT",
        "ALTER TABLE day_chats ADD COLUMN IF NOT EXISTS archival_summary_generated_at TIMESTAMP",
        "ALTER TABLE day_chats ADD COLUMN IF NOT EXISTS archival_summary_status VARCHAR(20) NOT NULL DEFAULT 'not_needed'",
        # FF-B.2 — bounded-retry policy for the rolling summarizer. Replaces
        # the previous fail-once-fail-forever behaviour. summary_failure_count
        # starts at 0 (default for existing rows). summary_last_failure_at is
        # nullable so retry-eligibility logic can identify never-failed days.
        # See docs/memory/changelog.md FF-B.2.
        "ALTER TABLE day_chats ADD COLUMN IF NOT EXISTS summary_failure_count INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE day_chats ADD COLUMN IF NOT EXISTS summary_last_failure_at TIMESTAMP",
        "ALTER TABLE day_chats ADD COLUMN IF NOT EXISTS summary_last_failure_reason VARCHAR(50)",
        # Partial index supports the retry-eligibility scan: tiny because most
        # day_chats are 'up_to_date', not 'failed'.
        "CREATE INDEX IF NOT EXISTS ix_day_chats_failed_retry_eligible ON day_chats (summary_status, summary_last_failure_at) WHERE summary_status = 'failed'",
        # Doc-delivery feature (generated-file attachments on assistant messages).
        # JSONB on Postgres, TEXT on SQLite — ORM uses Text for portability, ALTER promotes to JSONB on PG.
        "ALTER TABLE messages ADD COLUMN IF NOT EXISTS attachments JSONB",
        # Orphan-proof rollouts: persisted phase + canary observation deadline so
        # a Railway redeploy mid-rollout can be resumed by the startup hook
        # (see rollout_service.resume_orphaned_rollouts).
        "ALTER TABLE rollouts ADD COLUMN IF NOT EXISTS phase VARCHAR(32) NOT NULL DEFAULT ''",
        "ALTER TABLE rollouts ADD COLUMN IF NOT EXISTS resume_after TIMESTAMP",
        # Heartbeat for stuck-rollout detection: reconciler orphans
        # rollouts with no progress in N min, regardless of total age.
        # Catches the "Railway redeploys keep killing the orchestrator"
        # case that the 30-min total-age fallback was too slow for.
        "ALTER TABLE rollouts ADD COLUMN IF NOT EXISTS last_progress_at TIMESTAMP",
        "CREATE INDEX IF NOT EXISTS ix_rollouts_last_progress_at ON rollouts (last_progress_at)",
        # Backfill: set last_progress_at = started_at on existing rows
        # so the threshold check treats them consistently until they
        # pick up real heartbeats from new orchestrator activity.
        "UPDATE rollouts SET last_progress_at = started_at WHERE last_progress_at IS NULL",
        # User-selectable LLM provider for bundle mode (anthropic | openai).
        # See model_router.classify_request — overrides the bundle default.
        "ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS preferred_provider VARCHAR(20) NOT NULL DEFAULT 'anthropic'",
        # Phase-1 prewarm reconciler input (migration 037, see
        # docs/onboarding/prewarm-phase0.md). updated_at auto-bumps via
        # the ManagedContainer model's onupdate hook; the (status,
        # updated_at) composite index keeps the reconciler query plan
        # locked. Backfill row on add — created_at is the closest stand-in
        # for "first known activity time" so freshly-added rows don't get
        # immediately treated as stuck.
        "ALTER TABLE managed_containers ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
        "UPDATE managed_containers SET updated_at = COALESCE(started_at, created_at) WHERE updated_at IS NULL",
        "CREATE INDEX IF NOT EXISTS ix_managed_containers_status_updated_at ON managed_containers (status, updated_at)",
        # Routines (system-managed scheduled actions). Tables themselves are
        # created via Base.metadata.create_all on fresh containers; this ALTER
        # adds the per-message discriminator that lets the frontend tell a
        # routine-generated assistant message apart from a normal reply.
        # `messages.channel` already exists (added by the time-channel-fix PR
        # above) and routine writes will set it to "routine".
        "ALTER TABLE messages ADD COLUMN IF NOT EXISTS source VARCHAR(50)",
        # Routines generalised from email-only → arbitrary scheduled agent
        # tasks (2026-05-12). `name` is user-visible; `prompt_text` is the
        # NL prompt for `kind='agent_task'`. Nullable for existing rows.
        "ALTER TABLE routines ADD COLUMN IF NOT EXISTS name VARCHAR(100)",
        "ALTER TABLE routines ADD COLUMN IF NOT EXISTS prompt_text TEXT",
        # Reminder shapes (alembic 042). Without these, `routines__remind`
        # fails INSERT with "column schedule_kind of relation routines does
        # not exist" the moment the user asks the agent to set ANY
        # reminder — diagnosed live on 2026-05-21 across three tenants
        # (mrhx, sam, parmida) whose schemas were created by an earlier
        # `create_all` snapshot. Mirrors mig 042 verbatim so fresh agent
        # boots self-heal without an alembic upgrade.
        "ALTER TABLE routines ADD COLUMN IF NOT EXISTS schedule_kind VARCHAR(20)",
        "ALTER TABLE routines ADD COLUMN IF NOT EXISTS schedule_at TIMESTAMP",
        "ALTER TABLE routines ADD COLUMN IF NOT EXISTS schedule_interval_seconds INTEGER",
        "ALTER TABLE routines ADD COLUMN IF NOT EXISTS schedule_window_start_local VARCHAR(10)",
        "ALTER TABLE routines ADD COLUMN IF NOT EXISTS schedule_window_end_local VARCHAR(10)",
        "ALTER TABLE routines ADD COLUMN IF NOT EXISTS auto_disable_after_fire BOOLEAN",
        "ALTER TABLE routines ADD COLUMN IF NOT EXISTS reminder_text TEXT",
        # One-active-per-kind partial UNIQUE (alembic 041 + 053a). The
        # current predicate exempts agent_task AND reminder — without
        # `reminder` in the exempt list, a user's second reminder
        # 409s with "An enabled 'reminder' routine already exists"
        # which the model paraphrases as a misleading "wired to
        # Telegram" reply (root-cause for PR #59). Drop+recreate so
        # the predicate is correct regardless of which alembic state
        # we land in.
        "DROP INDEX IF EXISTS uq_routines_one_per_kind",
        "CREATE UNIQUE INDEX uq_routines_one_per_kind "
        "ON routines (user_id, kind) "
        "WHERE enabled = true AND kind NOT IN ('agent_task', 'reminder')",
        # Routine-run terminal mapping (alembic 040). Same self-heal
        # rationale — without these the routine runner's UPDATE of
        # routine_runs after fire 500s.
        "ALTER TABLE routine_runs ADD COLUMN IF NOT EXISTS outcome VARCHAR(30)",
        "ALTER TABLE routine_runs ADD COLUMN IF NOT EXISTS retry_attempt INTEGER",
        "ALTER TABLE routine_runs ADD COLUMN IF NOT EXISTS retry_of_run_id VARCHAR(36)",
        "ALTER TABLE routine_runs ADD COLUMN IF NOT EXISTS latency_ms INTEGER",
        "ALTER TABLE routine_runs ADD COLUMN IF NOT EXISTS delivery_json TEXT",
        # Unified-jobs mirrors (alembic 047 + 048). Adds the `job_id`
        # backreference column on the legacy event tables so the runner's
        # dual-write phase (now removed; mig 050 drops these tables in a
        # later opt-in step) doesn't 500 if the migration didn't run.
        "ALTER TABLE trigger_events ADD COLUMN IF NOT EXISTS job_id VARCHAR(36)",
        "ALTER TABLE routine_runs ADD COLUMN IF NOT EXISTS job_id VARCHAR(36)",
        # CronJob backfill pointer (alembic 042 — paired with the
        # routines schedule shapes above). Lets the cron→routine
        # migrator stamp each legacy cron_jobs row with its new routine
        # id. Column is harmless on platform DBs where cron_jobs has 0
        # rows; load-bearing for tenant DBs mid-cutover.
        "ALTER TABLE cron_jobs ADD COLUMN IF NOT EXISTS migrated_to_routine_id VARCHAR(36)",
        # Triggers (event-driven automations — Gate T1). `triggers` and
        # `trigger_events` tables are created by Base.metadata.create_all
        # on fresh containers; these ALTERs ensure idempotent re-runs on
        # containers that were spun up before any later schema additions
        # land. Empty for v1 — listed here so the migration path is
        # obvious when we add columns later.
        # (Reserved space; no v1 ALTERs needed beyond create_all.)
        # Bug sweep 2026-05-13 / Ticket 2: explicit memory↔entity linkage
        # (ref_kind, ref_id) so memories ABOUT a specific routine/trigger
        # upsert instead of duplicating. Columns are nullable — legacy
        # memories don't need them. Partial unique index enforces "one
        # active memory per (user, ref_kind, ref_id)" when set.
        "ALTER TABLE memories ADD COLUMN IF NOT EXISTS ref_kind VARCHAR(50)",
        "ALTER TABLE memories ADD COLUMN IF NOT EXISTS ref_id VARCHAR(100)",
        "CREATE INDEX IF NOT EXISTS ix_memories_ref ON memories (user_id, ref_kind, ref_id)",
        "CREATE UNIQUE INDEX IF NOT EXISTS ix_memories_ref_unique "
        "ON memories (user_id, ref_kind, ref_id) "
        "WHERE ref_id IS NOT NULL AND is_deleted = FALSE",
        # Bug sweep 2026-05-13 / Ticket 1: enforce Reading-A invariant —
        # one Conversation per (user, day, channel) for system channels.
        # Predicate matches SYSTEM_CHANNELS in
        # app.agent.conversation_resolver. Partial-on-is_active so soft-
        # deleted (post-merge) rows from the cleanup script don't block
        # future inserts.
        "CREATE UNIQUE INDEX IF NOT EXISTS ix_conversations_system_channel_per_day "
        "ON conversations (user_id, day_chat_id, channel) "
        "WHERE channel IN ('routine', 'trigger', 'api', 'digest') AND is_active = TRUE",
    ]

    # Vector dimension migration (agent-only: memories, entities, messages, document_chunks)
    if _run_mode in ("agent", "monolith"):
        from app.config import settings as _cfg
        _dim = _cfg.embedding_dimension
        _vec_migration = [
            f"DO $$ BEGIN IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='memories' AND column_name='embedding') THEN "
            f"IF (SELECT atttypmod FROM pg_attribute WHERE attrelid='memories'::regclass AND attname='embedding') != {_dim} THEN "
            f"EXECUTE 'ALTER TABLE memories DROP COLUMN embedding'; "
            f"EXECUTE 'ALTER TABLE memories ADD COLUMN embedding vector({_dim})'; "
            f"END IF; END IF; END $$",
            f"DO $$ BEGIN IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='entities' AND column_name='embedding') THEN "
            f"IF (SELECT atttypmod FROM pg_attribute WHERE attrelid='entities'::regclass AND attname='embedding') != {_dim} THEN "
            f"EXECUTE 'ALTER TABLE entities DROP COLUMN embedding'; "
            f"EXECUTE 'ALTER TABLE entities ADD COLUMN embedding vector({_dim})'; "
            f"END IF; END IF; END $$",
            f"DO $$ BEGIN IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='messages' AND column_name='embedding') THEN "
            f"IF (SELECT atttypmod FROM pg_attribute WHERE attrelid='messages'::regclass AND attname='embedding') != {_dim} THEN "
            f"EXECUTE 'ALTER TABLE messages DROP COLUMN embedding'; "
            f"EXECUTE 'ALTER TABLE messages ADD COLUMN embedding vector({_dim})'; "
            f"END IF; END IF; END $$",
            f"DO $$ BEGIN IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='document_chunks' AND column_name='embedding') THEN "
            f"IF (SELECT atttypmod FROM pg_attribute WHERE attrelid='document_chunks'::regclass AND attname='embedding') != {_dim} THEN "
            f"EXECUTE 'ALTER TABLE document_chunks DROP COLUMN embedding'; "
            f"EXECUTE 'ALTER TABLE document_chunks ADD COLUMN embedding vector({_dim})'; "
            f"END IF; END IF; END $$",
        ]
        _alter_statements.extend(_vec_migration)
        _alter_statements.append(
            f"CREATE INDEX IF NOT EXISTS ix_memories_embedding_hnsw "
            f"ON memories USING hnsw (embedding vector_cosine_ops) "
            f"WITH (m = 16, ef_construction = 64)"
        )

    # Seed VPS plans (platform-only)
    _seed_statements = []
    if _run_mode in ("platform", "monolith"):
        _seed_statements = [
        # Starter: CPX21 — 3 vCPU, 4GB RAM, 80GB SSD ($10.59 cost, $15 price)
        # For: 1-5 users, light usage, 1 embedding model (lazy-loaded)
        """INSERT INTO vps_plans (id, name, instance_type, vcpu, ram_gb, storage_gb, price_cents, stripe_price_id, provider, hetzner_server_type)
           VALUES ('starter', 'Starter', 'cpx21', 3, 4, 80, 1500, '', 'hetzner', 'cpx21')
           ON CONFLICT (id) DO UPDATE SET provider='hetzner', instance_type='cpx21',
             hetzner_server_type='cpx21', vcpu=3, ram_gb=4, storage_gb=80, price_cents=1500, name='Starter'""",
        # Standard: CPX31 — 4 vCPU, 8GB RAM, 160GB SSD ($18.59 cost, $25 price)
        # For: 5-20 users, recommended tier, embedding model always loaded
        """INSERT INTO vps_plans (id, name, instance_type, vcpu, ram_gb, storage_gb, price_cents, stripe_price_id, provider, hetzner_server_type)
           VALUES ('standard', 'Standard', 'cpx31', 4, 8, 160, 2500, '', 'hetzner', 'cpx31')
           ON CONFLICT (id) DO UPDATE SET provider='hetzner', instance_type='cpx31',
             hetzner_server_type='cpx31', vcpu=4, ram_gb=8, storage_gb=160, price_cents=2500, name='Standard'""",
        # Pro: CPX41 — 8 vCPU, 16GB RAM, 240GB SSD ($34.09 cost, $45 price)
        # For: 20-50+ users, power use, multiple models, large vector DBs
        """INSERT INTO vps_plans (id, name, instance_type, vcpu, ram_gb, storage_gb, price_cents, stripe_price_id, provider, hetzner_server_type)
           VALUES ('pro', 'Pro', 'cpx41', 8, 16, 240, 4500, '', 'hetzner', 'cpx41')
           ON CONFLICT (id) DO UPDATE SET provider='hetzner', instance_type='cpx41',
             hetzner_server_type='cpx41', vcpu=8, ram_gb=16, storage_gb=240, price_cents=4500, name='Pro'""",
        # Remove ALL plans except the 3 user-facing tiers
        "DELETE FROM vps_plans WHERE id NOT IN ('starter', 'standard', 'pro')",
        # ── Credit-system subscription_plans (mirror of alembic 053) ──
        # READ FIRST in MEMORY.md — agents/tests boot via init_db, NOT
        # alembic. Without this seed, credit_service.get_or_create_balance
        # raises "subscription_plans 'free' row missing — alembic 053
        # not applied?" the first time any deduction fires. Tests that
        # exercise the credit system silently 500 because the test
        # fixture never seeds plans. Same numbers as
        # alembic/versions/20260521_0053_053_credit_system_tables.py
        # `_PLAN_SEED` — keep the two in sync.
        # rollover_message_credits / rollover_integration_credits / active
        # are BOOLEAN columns (alembic 053 schema, see _PLAN_SEED there).
        # Previously these literals were `0`/`1` integers, which Postgres
        # refuses to coerce to bool ("column ... is of type boolean but
        # expression is of type integer"). Every boot logged
        # `[init_db] seed skipped` for all 5 rows — harmless when the
        # alembic seed had already run, but loud and misleading. Use
        # `false`/`true` so the seed actually inserts on fresh DBs
        # without an applied alembic 053 (i.e. the CI test fixture
        # path).
        """INSERT INTO subscription_plans
            (id, display_name, price_cents, message_credits_monthly,
             integration_credits_monthly, message_credits_daily_cap,
             rollover_message_credits, rollover_integration_credits,
             rollover_max_pct, sort_order, active, created_at)
           VALUES ('free', 'Free', 0, 30, 120, 5, false, false, 0, 0, true,
                   CURRENT_TIMESTAMP)
           ON CONFLICT (id) DO NOTHING""",
        """INSERT INTO subscription_plans
            (id, display_name, price_cents, message_credits_monthly,
             integration_credits_monthly, message_credits_daily_cap,
             rollover_message_credits, rollover_integration_credits,
             rollover_max_pct, sort_order, active, created_at)
           VALUES ('starter', 'Starter', 1600, 130, 2500, NULL, false, false, 0,
                   10, true, CURRENT_TIMESTAMP)
           ON CONFLICT (id) DO NOTHING""",
        """INSERT INTO subscription_plans
            (id, display_name, price_cents, message_credits_monthly,
             integration_credits_monthly, message_credits_daily_cap,
             rollover_message_credits, rollover_integration_credits,
             rollover_max_pct, sort_order, active, created_at)
           VALUES ('builder', 'Builder', 4000, 320, 12500, NULL, false, false, 0,
                   20, true, CURRENT_TIMESTAMP)
           ON CONFLICT (id) DO NOTHING""",
        """INSERT INTO subscription_plans
            (id, display_name, price_cents, message_credits_monthly,
             integration_credits_monthly, message_credits_daily_cap,
             rollover_message_credits, rollover_integration_credits,
             rollover_max_pct, sort_order, active, created_at)
           VALUES ('pro', 'Pro', 8000, 650, 25000, NULL, false, false, 0,
                   30, true, CURRENT_TIMESTAMP)
           ON CONFLICT (id) DO NOTHING""",
        """INSERT INTO subscription_plans
            (id, display_name, price_cents, message_credits_monthly,
             integration_credits_monthly, message_credits_daily_cap,
             rollover_message_credits, rollover_integration_credits,
             rollover_max_pct, sort_order, active, created_at)
           VALUES ('elite', 'Elite', 16000, 1500, 60000, NULL, true, true, 50,
                   40, true, CURRENT_TIMESTAMP)
           ON CONFLICT (id) DO NOTHING""",
    ]
    # Each migration statement runs in its OWN transaction so a failure
    # doesn't poison the rest of the loop. With a single transaction
    # (engine.begin), one failed ALTER aborts the whole thing — every
    # subsequent statement then fails with "current transaction is
    # aborted". This was the root cause of the 2026-04-28 incident
    # where preferred_provider + rollouts.phase columns silently failed
    # to land on production despite being in the loop.
    from sqlalchemy import text
    for stmt in _alter_statements:
        try:
            async with engine.begin() as conn:
                await conn.execute(text(stmt))
        except Exception as _e:
            _logger.warning("[init_db] alter skipped: %s — %s", stmt[:80], str(_e)[:200])
    for stmt in _seed_statements:
        try:
            async with engine.begin() as conn:
                await conn.execute(text(stmt))
        except Exception as _e:
            _logger.warning("[init_db] seed skipped: %s — %s", stmt[:80], str(_e)[:200])


async def drop_db():
    """Drop all database tables (for testing)"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
