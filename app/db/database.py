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


class _SkipPlanning(Exception):
    """Internal signal: boot-DDL planning is not armed for this tenant.

    Raised and caught inside init_db() only, so the "not armed" path and the
    "planning blew up" path stay visibly distinct — the first is normal and
    silent, the second must warn.
    """



def _build_engine(database_url: str) -> AsyncEngine:
    """Construct an AsyncEngine, optionally with connection-leak tracing.

    Wraps `_build_engine_inner` so the tracing also covers the engines built
    by `rebind_database()` — the agent swaps generic→tenant on /admin/bind,
    and a leak that only appeared on the post-bind engine would otherwise be
    invisible to the instrument.
    """
    eng = _build_engine_inner(database_url)
    try:
        from app.db import pool_debug
        if pool_debug.should_enable(settings.pool_leak_debug):
            pool_debug.install(eng)
    except Exception as _pd_err:  # never let a diagnostic break boot
        logger.warning("[pool-leak] install failed: %s", _pd_err)
    return eng


def _build_engine_inner(database_url: str) -> AsyncEngine:
    """Construct an AsyncEngine for the given URL using the same
    knobs as the original module-level setup. Centralized so
    `rebind_database()` builds an identical engine to the one created
    at import."""
    if database_url.startswith("sqlite"):
        if ":memory:" in database_url:
            # In-memory databases NEED the one shared connection —
            # every new connection would otherwise open a fresh empty
            # DB (plain :memory:) or race the last-close teardown
            # (file::memory:?cache=shared). This is the CI/test shape.
            return create_async_engine(
                database_url,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
                echo=settings.sql_echo,
            )
        # R28-D: file-backed sqlite (dev agents, the e2e harnesses)
        # must NOT share one connection across the whole process. On
        # StaticPool a background loop's rollback lands on the same
        # connection as a request handler mid-transaction and EATS its
        # flushed-but-uncommitted writes — a live agent lost its own
        # `arm` this way (routine.enabled=1 flushed, a concurrent
        # replay sweep rolled back, the later commit committed
        # nothing). NullPool gives each session its own connection;
        # the busy timeout rides out writer contention.
        return create_async_engine(
            database_url,
            connect_args={"check_same_thread": False, "timeout": 30},
            poolclass=NullPool,
            echo=settings.sql_echo,
        )
    if settings.run_mode in ("platform", "agent"):
        _db_url = database_url
        _sep = "&" if "?" in _db_url else "?"
        _db_url += f"{_sep}prepared_statement_cache_size=0"
        # Connection pooling. The platform API is a long-lived, single-bind
        # process, so it reuses a WARM pool: NullPool re-handshaked the Supabase
        # transaction pooler (full TCP+TLS+SCRAM round-trip) on EVERY request,
        # flooring every endpoint — including the auth-less /health doing only
        # SELECT 1 — at ~1.3s. Reusing connections removes that per-request
        # handshake (the dominant latency floor). Statement caching stays
        # disabled (connect_args + the URL param above) so it remains safe
        # against the :6543 transaction pooler; pool_pre_ping is a single cheap
        # round-trip on an already-WARM connection (not a fresh handshake) that
        # guards against a stale one; pool_recycle drops connections the pooler
        # may have closed; the bounded pool keeps the 2 replicas comfortably
        # under the pooler's client-connection cap (and is FEWER connections
        # than NullPool's per-request churn, not more).
        #
        # The agent keeps NullPool: it rebinds its DB generic→tenant on
        # /admin/bind, and a connectionless pool keeps that swap trivially clean.
        if settings.run_mode == "platform":
            _pool_kwargs = {"pool_size": 10, "max_overflow": 10, "pool_recycle": 300}
        else:
            _pool_kwargs = {"poolclass": NullPool}
        return create_async_engine(
            _db_url,
            echo=settings.sql_echo,
            pool_pre_ping=True,
            connect_args={
                "statement_cache_size": 0,
                "prepared_statement_name_func": lambda: "",
                "command_timeout": 30,
                "server_settings": {"statement_timeout": "30000"},
            },
            **_pool_kwargs,
        )
    return create_async_engine(
        database_url,
        echo=settings.sql_echo,
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
_holder["url"] = settings.database_url
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
    _holder["url"] = new_database_url
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


def current_database_url() -> str:
    """URL the live engine was built against (tracks rebinds)."""
    return _holder.get("url") or settings.database_url


async def recover_engine() -> None:
    """Replace the live engine with a freshly-built one on the SAME URL.

    This is the poisoned-pool cure (2026-07-04, tenant 3134fece): an
    interrupted transaction (blue-green cutover mid-txn, network blip)
    can leave engine/session state in endless PendingRollbackError /
    BEGIN-ROLLBACK churn where every DB touch fails while the process
    itself looks healthy — before this, the only cure was a manual
    `docker restart`. Building a new engine and swapping the holder is
    exactly the DB-layer effect of a restart, without dropping
    WebSockets or losing in-process bind state.

    The new engine is smoke-tested (SELECT 1) BEFORE the swap: if the
    database itself is down, we raise and keep the old engine — the
    caller (db_watchdog) retries on its next tick.
    """
    global engine
    url = current_database_url()
    logger.warning("[database] recover_engine: rebuilding engine on current URL")
    new_engine = _build_engine(url)
    async with new_engine.connect() as conn:
        from sqlalchemy import text
        await conn.execute(text("SELECT 1"))

    old_engine = _holder["engine"]
    _holder["engine"] = new_engine
    _holder["session_maker"] = _build_sessionmaker(new_engine)
    engine = new_engine
    if old_engine is not None:
        try:
            await old_engine.dispose()
        except Exception as e:
            logger.warning("[database] recover_engine: old dispose failed: %s", e)


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

    # Tables whose DDL needs the pgvector extension (embedding columns).
    # Hoisted out of the fallback branch so the missing-table backstop
    # below can reuse the same skip set.
    _vector_tables = set()
    for table in _allowed_tables:
        for col in table.columns:
            if "vector" in str(col.type).lower():
                _vector_tables.add(table.name)
                break

    # If create_all failed due to missing vector, create tables one by one
    # in a FRESH connection (the old transaction is in an aborted state).
    # Each CREATE gets its OWN transaction: with one shared engine.begin(),
    # the first failure (e.g. an FK onto a skipped vector table — entity_links
    # → entities) aborts the transaction and every later CREATE then fails
    # with "current transaction is aborted". That poisoning is how tenant
    # 871bac24's DB ended up permanently missing memory_events +
    # memory_relationships (W0.1b) — same lesson as the _alter_statements
    # loop's 2026-04-28 incident below.
    if not _has_pgvector:
        for table in _allowed_tables:
            if table.name in _vector_tables:
                _logger.info("Skipping table %s (needs pgvector)", table.name)
                continue
            try:
                async with engine.begin() as conn:
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

    # ── Missing-table backstop (W0.1b) ────────────────────────────────
    # PROVEN in prod (tenant 871bac24): a DB can drift to a state where
    # some model tables were never created — an old boot went down the
    # no-pgvector fallback above when it still ran in ONE shared
    # transaction, so the first failed CREATE silently killed every
    # later one (memory_events, memory_relationships), and nothing ever
    # retried them. `create_all` heals missing tables on a healthy boot,
    # but any bulk-pass failure leaves the DB short forever. Re-check
    # here and create stragglers individually, each in its own
    # transaction, so one failure can't poison the rest. No-op (one
    # inspection round-trip) when the schema is complete. Runs BEFORE
    # the ALTER loop so healed tables also receive their index/backfill
    # statements below.
    try:
        async with engine.connect() as conn:
            _present_tables = set(await conn.run_sync(
                lambda sync_conn: sa_inspect(sync_conn).get_table_names()
            ))
        for table in _allowed_tables:
            if table.name in _present_tables:
                continue
            if not _has_pgvector and table.name in _vector_tables:
                continue  # same skip as the fallback path above
            try:
                async with engine.begin() as conn:
                    await conn.run_sync(table.create, checkfirst=True)
                _logger.warning(
                    "[init_db] table backstop: created missing table %s", table.name
                )
            except Exception as _e:
                _logger.warning(
                    "[init_db] table backstop: FAILED %s — %s",
                    table.name, str(_e)[:200],
                )
    except Exception as _e:
        _logger.warning("[init_db] table backstop skipped: %s", str(_e)[:200])

    # Add missing columns to existing tables (create_all only creates new tables).
    #
    # IMPORTANT: All ALTER statements run in ALL modes. The table partitioning
    # (AGENT_ONLY / PLATFORM_ONLY) only gates table CREATION, not column additions.
    # Agent DBs are monolith-style — they contain all tables including platform ones
    # (agent_configs, vps_plans, etc.) from before the partitioning was introduced.
    # ALTER TABLE ... IF NOT EXISTS is safe on any table whether or not it "belongs"
    # to this run_mode. Trying to ALTER a non-existent table just silently fails
    # in the try/except below.

    # The system-channel partial unique index further down names its channels;
    # it reads them from the resolver that depends on it rather than keeping a
    # second copy here. Imported inside init_db — this module is on everyone's
    # import path and app.agent.conversation_resolver must not become part of it.
    from app.agent.conversation_resolver import INDEXED_SYSTEM_CHANNELS
    _system_channel_sql = ", ".join(f"'{_c}'" for _c in INDEXED_SYSTEM_CHANNELS)

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
        # Mig 086. First time this account heard media play. Platform owns the
        # value (SHARED_COLUMN_AUTHORITY), but the column still has to EXIST
        # tenant-side: the ORM model declares it, so without this every agent
        # query that loads a User row 500s against a pre-086 tenant DB — the
        # same failure mode the email-verification block below documents.
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS first_media_played_at TIMESTAMP",
        # Email verification (alembic 055). Mirrored here because the
        # agent's boot path runs init_db() but NOT alembic upgrade —
        # without these ALTERs, every endpoint that loads a User row
        # 500s with "column users.email_verified_at does not exist"
        # the moment an agent is rolled to a build that references the
        # new column on the ORM model. Self-healing on next boot.
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS email_verified_at TIMESTAMP",
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS email_verification_token VARCHAR(64)",
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS email_verification_sent_at TIMESTAMP",
        # Mig 061 (Sign in with Apple). Tenant DBs are migrated by THIS list,
        # not alembic — without it, an agent on the new image whose User model
        # has `apple_refresh_token` 500s on every User query (sessions, ws/chat
        # → "Connection lost") against its pre-mig tenant DB. Mirror it here so
        # every agent boot/recycle self-heals.
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS apple_refresh_token TEXT",
        # Mig 068 (Sign-in-with-Apple STABLE-sub dedupe). Same self-heal reason
        # as apple_refresh_token above: an agent rolled to a build whose User
        # model has `apple_sub` 500s on EVERY User query (get_user_by_id →
        # ws/chat 1006 → "Connection lost") against its pre-mig tenant DB until
        # this ALTER runs. The 2026-06-29 incident: brand-new users couldn't chat
        # because this line was missing when apple_sub was added — the agent
        # crashed on the first message. (The unique index is a platform dedup
        # concern; the column alone satisfies the agent's ORM SELECTs.)
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS apple_sub VARCHAR(255)",
        # ── Agent configs ──
        "ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS llm_mode VARCHAR(20) DEFAULT 'manual'",
        # L-1 (Last Mile): tenant agent_configs tables created by old
        # snapshots carry NOT NULL on agent_model; the ORM says
        # nullable=True. The soul-sync receiver's row-CREATE (which
        # writes only identity fields — it must never invent a model,
        # R-6) hit `NotNullViolationError: agent_model` on every tenant
        # without a pre-existing row, was swallowed by the old
        # warn-and-200, and the agent_name backfill silently no-opped
        # for exactly those tenants. Live evidence: toup-agent-3134fece
        # container log, 2026-08-12. DROP NOT NULL converges the tenant
        # schema to the ORM's contract at next boot.
        "ALTER TABLE agent_configs ALTER COLUMN agent_model DROP NOT NULL",
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
        # 077 — error taxonomy + archival + progress (Mission Control
        # overhaul). Same READ-FIRST rule as 056 above: build_jobs is
        # AGENT_ONLY and agents boot via init_db, never alembic, so a
        # column referenced by the ORM but missing here 500s every
        # SELECT from build_jobs (and poisons the surrounding
        # transaction — see the agent_configs incident in base.py).
        "ALTER TABLE build_jobs ADD COLUMN IF NOT EXISTS error_class VARCHAR(40)",
        "ALTER TABLE build_jobs ADD COLUMN IF NOT EXISTS user_message TEXT",
        "ALTER TABLE build_jobs ADD COLUMN IF NOT EXISTS technical_detail TEXT",
        "ALTER TABLE build_jobs ADD COLUMN IF NOT EXISTS archived_at TIMESTAMP",
        "ALTER TABLE build_jobs ADD COLUMN IF NOT EXISTS progress_step INTEGER",
        "ALTER TABLE build_jobs ADD COLUMN IF NOT EXISTS progress_total INTEGER",
        "CREATE INDEX IF NOT EXISTS ix_build_jobs_archived_at ON build_jobs (archived_at)",
        # The Activity list's hot query is "this user's un-archived jobs,
        # newest first". Without this it is a seq-scan + sort over the
        # whole table on every 5s poll.
        "CREATE INDEX IF NOT EXISTS ix_build_jobs_user_active "
        "ON build_jobs (user_id, created_at DESC) WHERE archived_at IS NULL",
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
        # Who the message is FROM as a PARTY — 'admin' for an operator notice
        # (Admin Dispatch), NULL for the ordinary user↔agent case. This is the
        # predicate `load_day_context` uses to keep an operator's words out of
        # the assembled LLM context, so it is a security boundary and not a
        # rendering hint. Indexed because that filter runs on every turn.
        #
        # Belongs HERE and not in alembic: `messages` is AGENT_ONLY, agent DBs
        # self-heal their columns through this list, and a tenant that has
        # never seen the column must acquire it on the next boot rather than
        # failing every INSERT that names it.
        "ALTER TABLE messages ADD COLUMN IF NOT EXISTS origin VARCHAR(16)",
        "CREATE INDEX IF NOT EXISTS ix_messages_origin ON messages (origin)",
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
        # ── Tenant memory-schema drift heal (W0.1b, 2026-07-27) ──
        # PROVEN in prod: tenant 871bac24's memories table predates the
        # User Brain upgrades and was missing source_* (UndefinedColumnError
        # on every memory query) — create_all only creates NEW tables, and
        # the no-pgvector fallback skips memories entirely, so an ancient
        # table never gains columns. Mirror the FULL current model column
        # set with model defaults: the DEFAULTs matter because Postgres
        # backfills existing rows, so filter-critical columns (is_deleted,
        # is_active, strength …) read their model default instead of NULL —
        # `WHERE is_deleted = FALSE` must keep matching legacy rows. The
        # generic reconcile backstop below adds bare NULLable columns only,
        # which would silently hide every legacy memory from retrieval.
        # (embedding is deliberately absent: the vector-dim DO block below
        # owns it, and it can't be added where pgvector is unavailable.)
        "ALTER TABLE memories ADD COLUMN IF NOT EXISTS brain_type VARCHAR(20) DEFAULT 'user'",
        "ALTER TABLE memories ADD COLUMN IF NOT EXISTS summary VARCHAR(500)",
        "ALTER TABLE memories ADD COLUMN IF NOT EXISTS embedding_json TEXT",
        "ALTER TABLE memories ADD COLUMN IF NOT EXISTS search_vector TSVECTOR",
        "ALTER TABLE memories ADD COLUMN IF NOT EXISTS importance FLOAT DEFAULT 0.5",
        "ALTER TABLE memories ADD COLUMN IF NOT EXISTS confidence FLOAT DEFAULT 1.0",
        "ALTER TABLE memories ADD COLUMN IF NOT EXISTS strength FLOAT DEFAULT 1.0",
        "ALTER TABLE memories ADD COLUMN IF NOT EXISTS memory_level VARCHAR(20) DEFAULT 'episodic'",
        "ALTER TABLE memories ADD COLUMN IF NOT EXISTS emotional_salience FLOAT DEFAULT 0.5",
        "ALTER TABLE memories ADD COLUMN IF NOT EXISTS last_reinforced_at TIMESTAMP",
        "ALTER TABLE memories ADD COLUMN IF NOT EXISTS consolidation_count INTEGER DEFAULT 0",
        "ALTER TABLE memories ADD COLUMN IF NOT EXISTS decay_rate FLOAT DEFAULT 0.1",
        "ALTER TABLE memories ADD COLUMN IF NOT EXISTS created_at TIMESTAMP",
        "ALTER TABLE memories ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP",
        "ALTER TABLE memories ADD COLUMN IF NOT EXISTS last_accessed_at TIMESTAMP",
        "ALTER TABLE memories ADD COLUMN IF NOT EXISTS access_count INTEGER DEFAULT 0",
        "ALTER TABLE memories ADD COLUMN IF NOT EXISTS source_message_id VARCHAR(36)",
        "ALTER TABLE memories ADD COLUMN IF NOT EXISTS source_type VARCHAR(50) DEFAULT 'conversation'",
        "ALTER TABLE memories ADD COLUMN IF NOT EXISTS metadata_json TEXT",
        "ALTER TABLE memories ADD COLUMN IF NOT EXISTS tags_json TEXT",
        "ALTER TABLE memories ADD COLUMN IF NOT EXISTS canonical_content TEXT",
        "ALTER TABLE memories ADD COLUMN IF NOT EXISTS history_json TEXT",
        "ALTER TABLE memories ADD COLUMN IF NOT EXISTS merged_from_json TEXT",
        "ALTER TABLE memories ADD COLUMN IF NOT EXISTS superseded_by VARCHAR(36)",
        "ALTER TABLE memories ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT TRUE",
        "ALTER TABLE memories ADD COLUMN IF NOT EXISTS is_deleted BOOLEAN DEFAULT FALSE",
        "ALTER TABLE memories ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMP",
        # Same heal for entities — also skipped by the no-pgvector fallback,
        # so an ancient entities table drifts identically.
        "ALTER TABLE entities ADD COLUMN IF NOT EXISTS description TEXT",
        "ALTER TABLE entities ADD COLUMN IF NOT EXISTS embedding_json TEXT",
        "ALTER TABLE entities ADD COLUMN IF NOT EXISTS schema_type VARCHAR(50)",
        "ALTER TABLE entities ADD COLUMN IF NOT EXISTS attributes_json TEXT",
        "ALTER TABLE entities ADD COLUMN IF NOT EXISTS name_search TSVECTOR",
        "ALTER TABLE entities ADD COLUMN IF NOT EXISTS mention_count INTEGER DEFAULT 1",
        "ALTER TABLE entities ADD COLUMN IF NOT EXISTS first_seen_at TIMESTAMP",
        "ALTER TABLE entities ADD COLUMN IF NOT EXISTS last_seen_at TIMESTAMP",
        "ALTER TABLE entities ADD COLUMN IF NOT EXISTS created_at TIMESTAMP",
        "ALTER TABLE entities ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP",
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
        # Memory taxonomy/TTL work 2026-07-29. `expires_at` gives the memory
        # system its first notion of temporal validity: before this, a
        # "remind me in 2 minutes" row was indistinguishable from "the user's
        # daughter is called Mira" and survived indefinitely. NULL = never
        # expires, which is the correct default for every pre-existing row —
        # this is purely additive and safe to run against live tenants.
        # Partial index: the expiry sweep only ever scans rows that have one.
        "ALTER TABLE memories ADD COLUMN IF NOT EXISTS expires_at TIMESTAMP",
        "CREATE INDEX IF NOT EXISTS ix_memories_expires_at "
        "ON memories (user_id, expires_at) "
        "WHERE expires_at IS NOT NULL AND is_active = TRUE",
        # Decay clock (alembic 080, 2026-08-06). The decay pass writes
        # `strength` but used to leave no mark, so every run re-applied the
        # FULL span since the last reinforcement to an already-decayed value
        # — the accumulated exponent grew with the number of runs instead of
        # with elapsed time. `last_decayed_at` is the point the stored
        # strength is accurate as of; DecayService decays FROM it. NULL =
        # never decayed, correct for every pre-existing row (they fall back
        # to the reinforcement/creation reference and get one catch-up
        # curve), so this is purely additive and safe on live tenants. No
        # index: the decay query filters on user_id/is_deleted/strength and
        # reads this column per row it already selected.
        "ALTER TABLE memories ADD COLUMN IF NOT EXISTS last_decayed_at TIMESTAMP",
        # Memory files 2026-08-19 (docs/memory/rebuild-2026-08.md). Which
        # curated file a row belongs to, and its order within it. NULL =
        # the tenant's organize pass hasn't touched the row yet — readers
        # fall back to the category→section map, so this is purely additive
        # and safe on live tenants. The memory_files table itself is new and
        # created by create_all on agent boot; only the memories columns
        # need the self-heal.
        "ALTER TABLE memories ADD COLUMN IF NOT EXISTS file_slug VARCHAR(160)",
        "ALTER TABLE memories ADD COLUMN IF NOT EXISTS file_position INTEGER",
        "CREATE INDEX IF NOT EXISTS ix_memories_user_file "
        "ON memories (user_id, file_slug) "
        "WHERE file_slug IS NOT NULL AND is_deleted = FALSE",
        # Memory v3 2026-08-20 (docs/memory/rebuild-2026-08-v3.md §1.1).
        # `memory_files` ALREADY EXISTS on every tenant that booted a round-8
        # image, so create_all will NOT add the new columns — a model column
        # missing from this list 500s every pre-existing tenant on the first
        # SELECT. All three are additive and nullable:
        #   body_md          the file's markdown bullet list — the memory
        #                    itself. NULL on a round-8 row until WS-5's
        #                    migration writes one.
        #   links_json       validated `[[slug]]` cross-links (round 8's
        #                    related_json stays, unread by v3, as the
        #                    migration's input).
        #   pinned_meta_json internal-only cursors (Current context
        #                    rollover). Never serialized to a client.
        # `description` is NOT here on purpose: it is the round-8 `purpose`
        # column under a new attribute name, so there is nothing to add.
        "ALTER TABLE memory_files ADD COLUMN IF NOT EXISTS body_md TEXT",
        "ALTER TABLE memory_files ADD COLUMN IF NOT EXISTS links_json TEXT",
        "ALTER TABLE memory_files ADD COLUMN IF NOT EXISTS pinned_meta_json TEXT",
        # The change log. New table — create_all makes it on a fresh agent
        # boot, but a tenant whose create_all already ran will not get it,
        # so state it here too. AGENT_ONLY: this whole block is swallowed on
        # the platform, where the table does not exist.
        """
        CREATE TABLE IF NOT EXISTS memory_file_changes (
            id VARCHAR(36) PRIMARY KEY,
            user_id VARCHAR(36) NOT NULL,
            file_slug VARCHAR(160) NOT NULL,
            file_title VARCHAR(200) NOT NULL,
            kind VARCHAR(20) NOT NULL,
            summary TEXT NOT NULL,
            day_key VARCHAR(10) NOT NULL,
            created_at TIMESTAMP
        )
        """,
        "CREATE INDEX IF NOT EXISTS ix_memory_file_changes_user_day "
        "ON memory_file_changes (user_id, day_key)",
        "CREATE INDEX IF NOT EXISTS ix_memory_file_changes_user_created "
        "ON memory_file_changes (user_id, created_at)",
        # The tsvector column has existed since the decay migration but its
        # maintenance trigger shipped ONLY in alembic — and agent containers
        # boot via create_all, so 100% of tenant rows had search_vector NULL
        # and the "keyword" leg of hybrid_search silently matched nothing.
        # Installing it here is what actually turns hybrid search on.
        """
        CREATE OR REPLACE FUNCTION memories_search_vector_update() RETURNS trigger AS $$
        BEGIN
            NEW.search_vector :=
                setweight(to_tsvector('english', coalesce(NEW.content, '')), 'A') ||
                setweight(to_tsvector('english', coalesce(NEW.summary, '')), 'B');
            RETURN NEW;
        END
        $$ LANGUAGE plpgsql
        """,
        "DROP TRIGGER IF EXISTS trg_memories_search_vector ON memories",
        "CREATE TRIGGER trg_memories_search_vector BEFORE INSERT OR UPDATE OF content, summary "
        "ON memories FOR EACH ROW EXECUTE FUNCTION memories_search_vector_update()",
        "CREATE INDEX IF NOT EXISTS ix_memories_search_vector "
        "ON memories USING gin (search_vector)",
        # Bug sweep 2026-05-13 / Ticket 1: enforce Reading-A invariant —
        # one Conversation per (user, day, channel) for system channels.
        # The predicate is BUILT from conversation_resolver's
        # INDEXED_SYSTEM_CHANNELS (see _system_channel_sql above), so the
        # resolver's IntegrityError→re-SELECT recovery and the constraint
        # that has to fire it can no longer disagree. Partial-on-is_active
        # so soft-deleted (post-merge) rows from the cleanup script don't
        # block future inserts.
        #
        # Drop first, and CREATE without IF NOT EXISTS, because both of
        # those skip on the index NAME: editing the predicate in place
        # would never reach a tenant that already has the index — Postgres
        # skips the CREATE, and ddl_plan skips the statement before
        # Postgres even sees it. Same drop+recreate shape as
        # uq_routines_one_per_kind above; keep the two adjacent, the gap
        # between them is the only moment the invariant is unenforced.
        "DROP INDEX IF EXISTS ix_conversations_system_channel_per_day",
        "CREATE UNIQUE INDEX ix_conversations_system_channel_per_day "
        "ON conversations (user_id, day_chat_id, channel) "
        f"WHERE channel IN ({_system_channel_sql}) AND is_active = TRUE",
        # ── Support agent: Phase-0 diagnosis-quality grade (alembic 066) ──
        # Mirrored here per the READ-FIRST rule: the platform-api boots via
        # init_db, and the Dockerfile CMD runs `alembic upgrade head` only
        # best-effort (it does NOT block uvicorn on failure). Without these,
        # every SELECT from support_issues 500s once the ORM model references
        # grade_verdict. support_issues is PLATFORM_ONLY, so this ALTER is a
        # no-op on agent DBs (table absent → swallowed by the try/except below).
        "ALTER TABLE support_issues ADD COLUMN IF NOT EXISTS grade_verdict VARCHAR(40)",
        "ALTER TABLE support_issues ADD COLUMN IF NOT EXISTS grade_note TEXT",
        "ALTER TABLE support_issues ADD COLUMN IF NOT EXISTS graded_by_user_id VARCHAR(36)",
        "ALTER TABLE support_issues ADD COLUMN IF NOT EXISTS graded_at TIMESTAMP",
        "CREATE INDEX IF NOT EXISTS ix_support_issues_grade_verdict ON support_issues (grade_verdict)",
        # ── Live Activity install_id (alembic 072) ──
        # Mirrored per the support_issues precedent above: platform-api
        # boots via init_db and `alembic upgrade head` is best-effort —
        # without this, device registration 500s the moment the ORM
        # model references install_id on a pre-072 platform DB.
        # live_activity_devices is PLATFORM_ONLY → no-op on agent DBs
        # (table absent → swallowed). The 072 partial unique index is
        # NOT mirrored: it requires the migration's dedupe pass first.
        "ALTER TABLE live_activity_devices ADD COLUMN IF NOT EXISTS install_id VARCHAR(64)",
        "CREATE INDEX IF NOT EXISTS ix_live_activity_devices_user_install ON live_activity_devices (user_id, install_id)",
        # ── AlarmKit ownership + observability (alembic 073) ──
        # Same precedent as 072 above: platform-api boots via init_db,
        # so the ORM must never reference a column a pre-073 DB lacks.
        # Both tables PLATFORM_ONLY → no-op on agent DBs.
        "ALTER TABLE live_activity_devices ADD COLUMN IF NOT EXISTS alarm_auth VARCHAR(16)",
        "ALTER TABLE live_activity_devices ADD COLUMN IF NOT EXISTS alarms_armed INTEGER",
        "ALTER TABLE live_activities ADD COLUMN IF NOT EXISTS alarm_owned_at TIMESTAMP",
        # ── Global consumable-IAP replay guard (alembic 074, round 12) ──
        # A consumable StoreKit / Play transaction may be redeemed by EXACTLY
        # ONE account, ever. The credit_ledger idempotency index is per-user
        # (user_id, idempotency_key), so a farmed second account could replay a
        # real purchase and mint credits (docs/security/audit-2026.md). This
        # table's transaction_id PRIMARY KEY enforces GLOBAL uniqueness. No FK
        # to users, so `CREATE TABLE IF NOT EXISTS` is safe on any DB partition
        # (an unused empty table on agent DBs). This init_db line is the
        # authoritative heal on the platform (the Dockerfile `alembic upgrade
        # head` is best-effort); migration 074 mirrors it for schema tracking.
        "CREATE TABLE IF NOT EXISTS redeemed_iap_transactions ("
        "transaction_id VARCHAR(120) PRIMARY KEY, "
        "user_id VARCHAR(36) NOT NULL, "
        "platform VARCHAR(16), "
        "created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP)",
        # ── Cache telemetry (alembic 075, F-7 / A9-1) ──
        # Same precedent as 072/073 above: platform-api boots via init_db,
        # so the ORM must never reference a column a pre-075 DB lacks —
        # without this, every _log_event INSERT 500s once LLMProxyEvent
        # references cached_tokens. llm_proxy_events is PLATFORM_ONLY →
        # no-op on agent DBs (table absent → swallowed).
        "ALTER TABLE llm_proxy_events ADD COLUMN IF NOT EXISTS cached_tokens INTEGER",
        # ── Toup Media library (2026-08-03) ──
        # media_playlists is created by create_all on a fresh agent, but an
        # agent that booted on an earlier build of this table already has it —
        # and create_all never adds columns to an existing table. Same lesson
        # as every ALTER above: the agent's boot path runs init_db(), not
        # alembic, so a new column has to self-heal here or every query
        # referencing it 500s on exactly the tenants that upgraded.
        "ALTER TABLE media_playlists ADD COLUMN IF NOT EXISTS seed_video_id VARCHAR(32)",
        # automations.domain (R28): life-domain assignment for memory
        # fact filing. The automations table is create_all-managed like
        # every AGENT_ONLY table, so a tenant that booted on the R26
        # build self-heals the column here.
        "ALTER TABLE automations ADD COLUMN IF NOT EXISTS domain VARCHAR(32)",
        # automations last-outcome + unseen (R29): stamped by
        # _finalize_job's exactly-once gate; same self-heal reasoning
        # as `domain` above.
        "ALTER TABLE automations ADD COLUMN IF NOT EXISTS last_outcome VARCHAR(24)",
        "ALTER TABLE automations ADD COLUMN IF NOT EXISTS last_outcome_text VARCHAR(300)",
        "ALTER TABLE automations ADD COLUMN IF NOT EXISTS last_outcome_at TIMESTAMP",
        "ALTER TABLE automations ADD COLUMN IF NOT EXISTS outcome_seen_at TIMESTAMP",
        # R30 (CONTRACTS-R30 §3): user rules + human step sentences on
        # the automation, the §4.8 soft delete, and the run-stop stamp
        # on build_jobs. Same self-heal reasoning as every ALTER above.
        "ALTER TABLE automations ADD COLUMN IF NOT EXISTS rules_json TEXT",
        "ALTER TABLE automations ADD COLUMN IF NOT EXISTS steps_human_json TEXT",
        "ALTER TABLE automations ADD COLUMN IF NOT EXISTS workflow_rev "
        "INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE automations ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMP",
        "ALTER TABLE build_jobs ADD COLUMN IF NOT EXISTS stop_requested_at TIMESTAMP",
        # The write's display form snapshotted at staging (R30 §4.8) —
        # an upgraded tenant's outbox INSERT names this column on the
        # very first staged write, so it must self-heal here.
        "ALTER TABLE automation_outbox ADD COLUMN IF NOT EXISTS display_json TEXT",
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
        # Backfill: autopilot tick jobs minted before 56006150 (founder
        # bug 2026-07-16) were job_type='routine_run' titled 'Routine
        # fire: autopilot <date>' — they pollute the Jobs page and
        # Mission Control's Done column, which now hide only
        # job_type='autopilot_tick'. Both predicates self-extinguish
        # (a flipped row no longer matches), so re-running at every
        # boot is a free no-op — the established data-fix pattern
        # (messages.channel, apps.source backfills above).
        _alter_statements.extend([
            # Authoritative: join to the parent routine, don't trust titles.
            "UPDATE build_jobs SET job_type = 'autopilot_tick' "
            "FROM routines r WHERE build_jobs.source_id = r.id "
            "AND r.kind = 'autopilot' AND build_jobs.job_type = 'routine_run'",
            # Fallback for rows whose parent routine was deleted. The
            # trailing space keeps a future kind like 'autopilot_v2'
            # from matching.
            "UPDATE build_jobs SET job_type = 'autopilot_tick' "
            "WHERE job_type = 'routine_run' "
            "AND title LIKE 'Routine fire: autopilot %'",
        ])

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
        # ── Credit-system subscription_plans (free-tier reflects mig 059) ──
        # READ FIRST in MEMORY.md — agents/tests boot via init_db, NOT
        # alembic. Without this seed, credit_service.get_or_create_balance
        # raises "subscription_plans 'free' row missing — alembic 053
        # not applied?" the first time any deduction fires. Tests that
        # exercise the credit system silently 500 because the test
        # fixture never seeds plans.
        #
        # Free-tier numbers reflect alembic 059 (post-bump):
        # 100 msg / 500 int / 15 day-cap (was 30/120/5 from mig 053).
        # On Postgres production, mig 059's UPDATE is authoritative for
        # the existing row; this seed only fires on fresh DBs (CI test
        # fixture, brand-new tenant Postgres) where DO NOTHING is a
        # no-op once any row exists.
        #
        # rollover_message_credits / rollover_integration_credits / active
        # are BOOLEAN columns. Use `false`/`true` literals — Postgres
        # refuses to coerce integer `0`/`1` to bool with a typed CHECK
        # constraint ("column ... is of type boolean but expression is
        # of type integer"). Previous literal regression: PR #119.
        """INSERT INTO subscription_plans
            (id, display_name, price_cents, message_credits_monthly,
             integration_credits_monthly, message_credits_daily_cap,
             rollover_message_credits, rollover_integration_credits,
             rollover_max_pct, sort_order, active, created_at)
           VALUES ('free', 'Free', 0, 100, 500, 15, false, false, 0, 0, true,
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
    from sqlalchemy import text, inspect as _sa_inspect
    _is_sqlite = engine.dialect.name == "sqlite"

    # ── Plan the pass against the live catalog (2026-08-03) ───────────
    # The list is idempotent, but idempotent is NOT free: Postgres takes
    # ACCESS EXCLUSIVE *before* it evaluates `IF NOT EXISTS`, so a no-op
    # ALTER still queues behind any open transaction on that table — and
    # the blue container is serving this same database during a
    # blue-green swap. Proven on postgres:16 (isolated container,
    # 2026-08-03): with one plain `BEGIN; SELECT count(*) FROM t;` held
    # open, a no-op `ADD COLUMN IF NOT EXISTS` hit `lock timeout` with
    # pg_locks showing AccessExclusiveLock granted=false. That is how a
    # green got through 41 of 358 statements in 251s on 2026-08-01 while
    # three siblings on the same host each did all 358 in under a minute.
    #
    # So: ask the catalog ONCE (read-only, no table locks) and issue only
    # the statements whose effect is actually missing. Deliberately NOT a
    # stored schema-version marker — tenant DBs have no alembic_version,
    # a marker can be written when a statement actually failed, and it
    # cannot see out-of-band drift. Catalog-derived planning can only
    # skip a statement whose effect is already present, and it re-plans a
    # dropped column back in on the next boot.
    #
    # Fail-open everywhere: a failed snapshot leaves _ddl_skipped empty
    # and every statement runs, which is exactly the previous behaviour.
    _statements_to_run = _alter_statements
    if not _is_sqlite:
        try:
            import time as _time
            from app.db.ddl_plan import (
                SNAPSHOT_SQL, plan, should_plan, snapshot_from_rows,
            )
            if not should_plan(settings.init_db_plan_ddl):
                raise _SkipPlanning()
            _t_plan = _time.perf_counter()
            async with engine.connect() as conn:
                _cols = (await conn.execute(text(SNAPSHOT_SQL["columns"]))).all()
                _idxs = (await conn.execute(text(SNAPSHOT_SQL["indexes"]))).all()
                _tbls = (await conn.execute(text(SNAPSHOT_SQL["tables"]))).all()
            _snap = snapshot_from_rows(
                [(r[0], r[1]) for r in _cols], [r[0] for r in _idxs], [r[0] for r in _tbls]
            )
            _to_run, _skip = plan(_alter_statements, _snap)
            _statements_to_run = _to_run
            _logger.info(
                "[init_db] ddl_plan: %d of %d statements need to run "
                "(%d already satisfied) — planned in %.0fms",
                len(_to_run), len(_alter_statements), len(_skip),
                (_time.perf_counter() - _t_plan) * 1000,
            )
        except _SkipPlanning:
            pass  # not armed for this tenant — full list, unchanged behaviour
        except Exception as _plan_err:
            _statements_to_run = _alter_statements
            _logger.warning(
                "[init_db] ddl_plan unavailable, running the full list: %s",
                str(_plan_err)[:200],
            )

    for stmt in _statements_to_run:
        try:
            if _is_sqlite:
                # sqlite doesn't support `ADD COLUMN IF NOT EXISTS` — rewrite
                # by inspecting existing columns and skipping if present.
                # Format we emit everywhere is:
                #   ALTER TABLE <tbl> ADD COLUMN IF NOT EXISTS <col> <type...>
                _rewritten = _rewrite_alter_for_sqlite(stmt)
                if _rewritten is None:
                    # Non-ALTER statement (or shape we can't rewrite); on sqlite
                    # we skip it rather than emit invalid SQL.
                    continue
                _tbl, _col, _tail = _rewritten
                async with engine.begin() as conn:
                    _cols = await conn.run_sync(
                        lambda sc, t=_tbl: {c["name"] for c in _sa_inspect(sc).get_columns(t)}
                        if t in _sa_inspect(sc).get_table_names() else None
                    )
                    if _cols is None:
                        # Table doesn't exist in this run_mode partition; skip.
                        continue
                    if _col in _cols:
                        continue
                    # Drop sqlite-incompatible type modifiers (JSONB → JSON,
                    # `UNIQUE` inline constraints, etc.) so the ADD COLUMN parses.
                    _tail_sqlite = _sqlite_safe_column_type(_tail)
                    await conn.execute(text(f'ALTER TABLE {_tbl} ADD COLUMN {_col} {_tail_sqlite}'))
            else:
                async with engine.begin() as conn:
                    await conn.execute(text(stmt))
        except Exception as _e:
            _logger.warning("[init_db] alter skipped: %s — %s", stmt[:80], str(_e)[:200])

    # ── Structural backstop: auto-reconcile forgotten model columns ──
    # The explicit _alter_statements list above is the PRIMARY schema-heal
    # mechanism and stays authoritative for column DEFAULTS, constraints,
    # indexes and backfills. This pass is the SAFETY NET for the one failure
    # mode that list can't prevent: a column added to a shared model whose
    # mirror ALTER line was forgotten. That is exactly the 2026-06-29
    # users.apple_sub incident — every agent rolled to the new image 500'd on
    # get_user_by_id (every chat turn) with "column users.apple_sub does not
    # exist" → ws/chat 1006 → mobile "Connection lost". Additive-only; runs
    # against the same `engine` (so the same DB) the ALTERs above just healed.
    try:
        await _reconcile_missing_columns(engine, _has_pgvector, _logger)
    except Exception as _e:
        _logger.warning("[init_db] column reconcile skipped: %s", str(_e)[:200])

    for stmt in _seed_statements:
        try:
            if _is_sqlite and "ON CONFLICT" in stmt:
                # sqlite supports ON CONFLICT clause as of 3.24, but the (id)
                # specifier requires a UNIQUE constraint on the column — which
                # ORM-created tables have via PRIMARY KEY. Run as-is; skip on
                # parse failure.
                pass
            async with engine.begin() as conn:
                await conn.execute(text(stmt))
        except Exception as _e:
            _logger.warning("[init_db] seed skipped: %s — %s", stmt[:80], str(_e)[:200])


def _rewrite_alter_for_sqlite(stmt: str):
    """Parse `ALTER TABLE <tbl> ADD COLUMN IF NOT EXISTS <col> <tail>`
    and return (tbl, col, tail). Returns None on shape mismatch."""
    import re
    m = re.match(
        r"\s*ALTER\s+TABLE\s+(\w+)\s+ADD\s+COLUMN\s+IF\s+NOT\s+EXISTS\s+(\w+)\s+(.+?)\s*$",
        stmt,
        re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3)


def _sqlite_safe_column_type(tail: str) -> str:
    """Best-effort rewrite of a Postgres column-type spec for sqlite.

    - JSONB → JSON (sqlite has no JSONB; JSON is a text alias)
    - inline UNIQUE constraint → dropped (sqlite ALTER TABLE ADD COLUMN
      doesn't accept inline UNIQUE)
    """
    import re
    tail = re.sub(r"\bJSONB\b", "JSON", tail, flags=re.IGNORECASE)
    tail = re.sub(r"\s+UNIQUE\b", "", tail, flags=re.IGNORECASE)
    return tail


async def _reconcile_missing_columns(eng, has_pgvector: bool, logger) -> None:
    """Auto-add any ORM-mapped column missing from its live table.

    STRUCTURAL BACKSTOP for the hand-maintained ``_alter_statements`` list in
    ``init_db()``. That list stays the primary mechanism and remains
    authoritative for column DEFAULTS, constraints, indexes and data
    backfills. This pass closes the ONE gap the list can't: a column added to
    a shared model (User, AgentConfig, …) whose mirror ALTER line was
    forgotten.

    Agent DBs migrate via ``init_db()``, NOT alembic, so a forgotten line
    means every agent rolled to that image 500s on every query that SELECTs
    the column. The 2026-06-29 ``users.apple_sub`` incident is the canonical
    case: ``get_user_by_id`` ran on every chat turn → ``UndefinedColumnError``
    → ws/chat closed 1006 → mobile "Connection lost". This diff-and-ADD pass
    makes that whole class of bug self-healing on the next boot/recycle, for
    any future column — without anyone having to remember the mirror line.

    Contract — ADDITIVE AND NON-DESTRUCTIVE:
      * Only ever emits ``ALTER TABLE … ADD COLUMN``. Never drops a column,
        never retypes one, never touches a row.
      * Missing columns are added NULLABLE, type only — no DEFAULT, no
        NOT NULL, no UNIQUE/FK, no index. A nullable add is all that's needed
        to stop the SELECT-time 500, and (unlike a NOT NULL add) it can't fail
        on a table that already has rows. Anything richer stays the job of an
        explicit ``_alter_statements`` entry; this is a safety net, not a
        migration engine.
      * Identifiers are dialect-quoted, types dialect-compiled. Vector columns
        are skipped when pgvector is unavailable (mirrors the create_all skip).
        Each ADD runs in its own transaction so one failure can't poison the
        rest. Runs against the same ``engine`` the ALTERs above just healed,
        so it targets the same DB (generic-pool or bound tenant).

    A non-empty result is logged loudly: it means a model column was added
    without its explicit mirror, and that gap should still be closed in
    ``_alter_statements`` (so the column gets its proper default/constraint),
    even though the bare column is self-healed here.
    """
    from sqlalchemy import text, inspect as sa_inspect

    dialect = eng.dialect
    is_sqlite = dialect.name == "sqlite"
    preparer = dialect.identifier_preparer
    model_tables = {t.name: t for t in Base.metadata.sorted_tables}

    # Reflect the columns each model table actually has right now.
    cols_by_table: dict = {}
    try:
        async with eng.connect() as conn:
            present = set(await conn.run_sync(
                lambda sc: sa_inspect(sc).get_table_names()
            ))
            if is_sqlite:
                for tname in model_tables:
                    if tname not in present:
                        continue
                    cols_by_table[tname] = await conn.run_sync(
                        lambda sc, t=tname: {c["name"] for c in sa_inspect(sc).get_columns(t)}
                    )
            else:
                # One round-trip for the whole DB, then intersect with the
                # model tables that exist in the default schema — avoids
                # per-table reflection chatter on the cross-region platform DB.
                rows = (await conn.execute(text(
                    "SELECT table_name, column_name FROM information_schema.columns "
                    "WHERE table_schema NOT IN ('pg_catalog', 'information_schema')"
                ))).all()
                for tname, cname in rows:
                    if tname in model_tables and tname in present:
                        cols_by_table.setdefault(tname, set()).add(cname)
    except Exception as _e:
        logger.warning("[init_db] reconcile reflection failed — skipping: %s", str(_e)[:200])
        return

    pending = []  # (table_name, column_name, ddl)
    for tname, have in cols_by_table.items():
        if not have:
            # A present table that reflected ZERO columns is anomalous (a
            # reflection hiccup, not a real schema) — never act on it, or we'd
            # try to "add everything". Skip and let create_all / the explicit
            # list own this table.
            logger.warning(
                "[init_db] reconcile: table %s present but reflected 0 columns — skipping",
                tname,
            )
            continue
        have_lower = {c.lower() for c in have}
        tbl = model_tables[tname]
        for col in tbl.columns:
            # Compare case-insensitively: Postgres folds unquoted identifiers to
            # lowercase, so information_schema can report a different case than
            # the ORM attribute even though it's the same physical column.
            if col.name.lower() in have_lower:
                continue
            try:
                type_sql = col.type.compile(dialect=dialect)
            except Exception as _e:
                logger.warning(
                    "[init_db] reconcile: can't compile type for %s.%s (%s) — skipping",
                    tname, col.name, str(_e)[:80],
                )
                continue
            # Skip pgvector VECTOR / VECTOR(n) columns when the extension isn't
            # available (create_all skipped their tables too). PREFIX test, not
            # a substring — Postgres's native TSVECTOR (full-text search) also
            # contains "vector" but must NEVER be skipped, and
            # "TSVECTOR".startswith("VECTOR") is False, so it won't be.
            if not has_pgvector and type_sql.upper().startswith("VECTOR"):
                continue
            qtbl = preparer.format_table(tbl)
            qcol = preparer.quote(col.name)
            if is_sqlite:
                ddl = f"ALTER TABLE {qtbl} ADD COLUMN {qcol} {_sqlite_safe_column_type(type_sql)}"
            else:
                ddl = f"ALTER TABLE {qtbl} ADD COLUMN IF NOT EXISTS {qcol} {type_sql}"
            pending.append((tname, col.name, ddl))

    if not pending:
        return

    logger.warning(
        "[init_db] reconcile: %d model column(s) missing from DB, auto-adding "
        "nullable — add an explicit _alter_statements mirror for these: %s",
        len(pending), ", ".join(f"{t}.{c}" for t, c, _ in pending),
    )
    for tname, cname, ddl in pending:
        try:
            async with eng.begin() as conn:
                await conn.execute(text(ddl))
            logger.warning("[init_db] reconcile: added %s.%s", tname, cname)
        except Exception as _e:
            logger.warning("[init_db] reconcile: FAILED %s.%s — %s", tname, cname, str(_e)[:200])


async def drop_db():
    """Drop all database tables (for testing)"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
