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
    ]
    # Seed user-facing VPS plans — Hetzner Cloud CPX (AMD, Shared Regular Performance)
    # Location: ASH (Ashburn, Virginia, USA)
    # NOTE: CPX11 (2GB RAM) too small for local embeddings + PostgreSQL + Python
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
