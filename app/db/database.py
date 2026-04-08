"""Database connection and session management"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import StaticPool, NullPool
from app.config import settings
from app.db.models import Base

# Create async engine
if settings.database_url.startswith("sqlite"):
    # SQLite configuration for development
    engine = create_async_engine(
        settings.database_url,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=settings.debug,
    )
elif settings.run_mode in ("platform", "agent"):
    # Supabase / PgBouncer (platform + remote agents)
    # - NullPool: no local pooling — PgBouncer handles it
    # - statement_cache_size=0: disables asyncpg's LRU statement cache
    # - prepared_statement_name_func returns '': forces unnamed prepared
    #   statements which PgBouncer handles correctly in transaction mode
    #   (named ones like __asyncpg_stmt_1__ collide across connections)
    # - pool_pre_ping: detect stale connections after cold starts
    _db_url = settings.database_url
    _sep = "&" if "?" in _db_url else "?"
    _db_url += f"{_sep}prepared_statement_cache_size=0"
    engine = create_async_engine(
        _db_url,
        echo=settings.debug,
        poolclass=NullPool,
        pool_pre_ping=True,
        connect_args={
            "statement_cache_size": 0,
            "prepared_statement_name_func": lambda: "",
            "command_timeout": 30,
            # Override Supabase's default statement_timeout (often 8s)
            # Agent sessions stay open during LLM calls, so we need more
            "server_settings": {"statement_timeout": "30000"},
        },
    )
else:
    # PostgreSQL — monolith mode (long-running process, direct connection)
    engine = create_async_engine(
        settings.database_url,
        echo=settings.debug,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
    )

# Session factory
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


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
                _logger.error(
                    "FATAL: Agent-only tables found in platform DB: %s. "
                    "This means table partitioning in base.py is broken. "
                    "Fix AGENT_ONLY_TABLES or remove these tables manually.",
                    _leaked,
                )
                raise RuntimeError(f"Agent-only tables leaked into platform DB: {_leaked}")
        elif _run_mode == "agent":
            _leaked = _existing_set & PLATFORM_ONLY_TABLES
            if _leaked:
                _logger.error(
                    "FATAL: Platform-only tables found in agent DB: %s. "
                    "This means table partitioning in base.py is broken. "
                    "Fix PLATFORM_ONLY_TABLES or remove these tables manually.",
                    _leaked,
                )
                raise RuntimeError(f"Platform-only tables leaked into agent DB: {_leaked}")
    except RuntimeError:
        raise
    except Exception as _e:
        _logger.warning("Could not verify table partitioning: %s", _e)

    # Add missing columns to existing tables (create_all only creates new tables).
    # Split by run_mode: shared statements run everywhere, platform/agent-specific
    # statements only run in their respective modes.

    # ── Shared (both platform and agent) ──
    _alter_shared = [
        # Password change tracking (token revocation)
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS password_changed_at TIMESTAMP",
        # Day-as-Chat: user timezone
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS timezone VARCHAR(50)",
    ]

    # ── Platform-only ALTER statements ──
    _alter_platform = [
        # LLM bundle columns on agent_configs (migration 016)
        "ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS llm_mode VARCHAR(20) DEFAULT 'manual'",
        "ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS google_api_key TEXT",
        "ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS mistral_api_key TEXT",
        "ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS groq_api_key TEXT",
        "ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS xai_api_key TEXT",
        "ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS deepseek_api_key TEXT",
        "ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS bundle_stripe_subscription_id VARCHAR(255)",
        "ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS bundle_status VARCHAR(20) DEFAULT 'none'",
        "ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS bundle_started_at TIMESTAMP",
        "ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS bundle_current_period_end TIMESTAMP",
        # Onboarding flag
        "ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS onboarding_completed BOOLEAN DEFAULT FALSE",
        "ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS agent_color VARCHAR(7)",
        # Per-user tool access control
        "ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS disabled_tools TEXT DEFAULT ''",
        # Connect token for tunnel authentication
        "ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS connect_token VARCHAR(100)",
        # Agent display name (migration 017)
        "ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS agent_name VARCHAR(100)",
        # Database mode for agent deployment
        "ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS db_mode VARCHAR(20) DEFAULT 'auto'",
        "ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS supabase_url TEXT",
        # Multi-provider VPS support (Hostinger + Hetzner)
        "ALTER TABLE vps_plans ADD COLUMN IF NOT EXISTS provider VARCHAR(20) DEFAULT 'aws'",
        "ALTER TABLE vps_plans ADD COLUMN IF NOT EXISTS hostinger_plan_id VARCHAR(50)",
        "ALTER TABLE vps_plans ADD COLUMN IF NOT EXISTS hetzner_server_type VARCHAR(20)",
        "ALTER TABLE vps_instances ADD COLUMN IF NOT EXISTS provider VARCHAR(20) DEFAULT 'aws'",
        "ALTER TABLE vps_instances ADD COLUMN IF NOT EXISTS hostinger_vm_id VARCHAR(50)",
        "ALTER TABLE vps_instances ADD COLUMN IF NOT EXISTS hetzner_vm_id VARCHAR(50)",
        # Target OS for remote deploy (linux/macos/windows)
        "ALTER TABLE agent_configs ADD COLUMN IF NOT EXISTS target_os VARCHAR(20)",
        # Stripe customer linkage on users (migration 018)
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS stripe_customer_id VARCHAR(255) UNIQUE",
    ]

    # ── Agent-only ALTER statements ──
    _alter_agent = [
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
        # Reconciliation: app source tracking
        "ALTER TABLE apps ADD COLUMN IF NOT EXISTS source VARCHAR(30) DEFAULT 'app_builder'",
        "UPDATE apps SET source = 'vibecoding' WHERE source = 'app_builder' AND app_dir LIKE '%/vibecoding/%'",
        # Builder mode per-chat
        "ALTER TABLE conversations ADD COLUMN IF NOT EXISTS builder_mode VARCHAR(10)",
        # Reconciliation log cleanup
        "DELETE FROM reconciliation_logs WHERE created_at < NOW() - INTERVAL '30 days'",
        # Day-as-Chat: FK from conversations to day_chats (nullable during migration)
        "ALTER TABLE conversations ADD COLUMN IF NOT EXISTS day_chat_id VARCHAR(36) REFERENCES day_chats(id)",
        # Day-as-Chat: denormalized FK from messages to day_chats for fast loading
        "ALTER TABLE messages ADD COLUMN IF NOT EXISTS day_chat_id VARCHAR(36) REFERENCES day_chats(id)",
        # Day-as-Chat: composite index for hot-path query
        "CREATE INDEX IF NOT EXISTS ix_messages_day_chat_created ON messages(day_chat_id, created_at)",
    ]

    # Assemble final list based on run_mode
    _alter_statements = list(_alter_shared)
    if _run_mode in ("platform", "monolith"):
        _alter_statements.extend(_alter_platform)
    if _run_mode in ("agent", "monolith"):
        _alter_statements.extend(_alter_agent)

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
    ]
    async with engine.begin() as conn:
        from sqlalchemy import text
        for stmt in _alter_statements:
            try:
                await conn.execute(text(stmt))
            except Exception:
                pass  # column already exists or DB doesn't support IF NOT EXISTS
        for stmt in _seed_statements:
            try:
                await conn.execute(text(stmt))
            except Exception:
                pass  # already seeded or table doesn't exist yet


async def drop_db():
    """Drop all database tables (for testing)"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
