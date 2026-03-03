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
    """Initialize database tables and add any missing columns."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Add missing columns to existing tables (create_all only creates new tables)
    _alter_statements = [
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
    ]
    # Seed user-facing VPS plans
    # TEMPORARY: Using AWS until Hetzner account is verified, then switch to Hetzner
    # Hetzner targets: CX23=$3.80, CX33=$6, CX43=$10.30/mo (much cheaper)
    _seed_statements = [
        # Starter: AWS t3.small — 2 vCPU, 2GB RAM, 20GB — keeping same specs/price for now
        """INSERT INTO vps_plans (id, name, instance_type, vcpu, ram_gb, storage_gb, price_cents, stripe_price_id, provider)
           VALUES ('starter', 'Starter', 't3.small', 2, 2, 20, 900, '', 'aws')
           ON CONFLICT (id) DO UPDATE SET provider='aws', instance_type='t3.small',
             vcpu=2, ram_gb=2, storage_gb=20, price_cents=900""",
        # Standard: AWS t3.medium — 2 vCPU, 4GB RAM, 40GB
        """INSERT INTO vps_plans (id, name, instance_type, vcpu, ram_gb, storage_gb, price_cents, stripe_price_id, provider)
           VALUES ('standard', 'Standard', 't3.medium', 2, 4, 40, 1900, '', 'aws')
           ON CONFLICT (id) DO UPDATE SET provider='aws', instance_type='t3.medium',
             vcpu=2, ram_gb=4, storage_gb=40, price_cents=1900""",
        # Pro: AWS t3.large — 2 vCPU, 8GB RAM, 80GB
        """INSERT INTO vps_plans (id, name, instance_type, vcpu, ram_gb, storage_gb, price_cents, stripe_price_id, provider)
           VALUES ('pro', 'Pro', 't3.large', 2, 8, 80, 3900, '', 'aws')
           ON CONFLICT (id) DO UPDATE SET provider='aws', instance_type='t3.large',
             vcpu=2, ram_gb=8, storage_gb=80, price_cents=3900""",
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
