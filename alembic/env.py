"""
Alembic environment configuration for Toup.
Supports async SQLAlchemy with PostgreSQL.
"""

import asyncio
import logging
import time
from logging.config import fileConfig
import os
import sys

from sqlalchemy import pool, create_engine
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context
from alembic.script import ScriptDirectory
from alembic.runtime.migration import MigrationContext

# Add the app directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your models' Base metadata
from app.db.models import Base
from app.config import settings

# this is the Alembic Config object
config = context.config

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Get DATABASE_URL and prepare for sync driver
db_url = settings.DATABASE_URL
# Convert async driver to sync for Alembic migrations
sync_url = db_url.replace("+asyncpg", "").replace("+aiosqlite", "")
# asyncpg accepts ?ssl=require; psycopg2 (sync driver used by alembic)
# rejects that param and wants ?sslmode=require instead. Normalize.
sync_url = sync_url.replace("?ssl=require", "?sslmode=require") \
                   .replace("&ssl=require", "&sslmode=require")
# configparser (used under set_main_option) treats '%' as an interpolation
# marker. URL-encoded passwords (e.g. %21 for '!') trigger ValueError
# "invalid interpolation syntax". Escape every bare '%' to '%%' before set.
sync_url_cfg_safe = sync_url.replace("%", "%%")
config.set_main_option("sqlalchemy.url", sync_url_cfg_safe)

# Add your model's MetaData object here for 'autogenerate' support
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL and not an Engine,
    though an Engine is acceptable here as well.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Run migrations with a connection."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


# TKT-LAT-016: short-circuit the migration run when the DB is already
# at the script-directory head(s). Saves ~1–5 s of script discovery +
# version-table chatter per agent boot. Gated by env var so operators
# can disable if some out-of-band schema drift sneaks in.
_LAT016_SKIP_DEFAULT = "true"


def _is_at_head(connection: Connection) -> bool:
    """Return True iff the DB schema is already at the script head(s).

    Defensive: any failure (no alembic_version table, dialect quirk,
    multiple-heads ambiguity) → False so the regular migration path
    runs. The cost of a false negative is one full alembic evaluation,
    which is exactly what we'd have done without this optimization.
    """
    try:
        ctx = MigrationContext.configure(connection)
        current_heads = set(ctx.get_current_heads())
        script = ScriptDirectory.from_config(config)
        target_heads = set(script.get_heads())
        # Require the DB to have at least one revision row to skip;
        # a brand-new database (empty set) must go through migrations.
        return bool(current_heads) and current_heads == target_heads
    except Exception:
        return False


def _is_head_invocation() -> bool:
    """True only when alembic was invoked with target == 'head'.

    Migration tests (and any programmatic caller that wants to advance
    to a specific revision) use `command.upgrade(cfg, "050")` — those
    MUST NOT short-circuit, even if the DB happens to be at the
    script-directory head. We're stricter than `current_heads ==
    script_heads` for exactly that reason: skip only when the user
    actually asked for "head". CLI `alembic upgrade head` sets
    `config.cmd_opts.revision = "head"`. Programmatic
    `command.upgrade(cfg, "050")` leaves `cmd_opts` as None or the
    revision string the caller passed.
    """
    try:
        cmd_opts = getattr(config, "cmd_opts", None)
        if cmd_opts is None:
            return False
        target = getattr(cmd_opts, "revision", None)
        return target in ("head", "heads")
    except Exception:
        return False


def run_migrations_online() -> None:
    """Run migrations in 'online' mode using synchronous driver."""
    url = config.get_main_option("sqlalchemy.url")
    connectable = create_engine(url, poolclass=pool.NullPool)

    log = logging.getLogger("alembic.env")
    skip_enabled = os.environ.get(
        "LAT_SKIP_NOOP_MIGRATIONS", _LAT016_SKIP_DEFAULT
    ).strip().lower() in ("1", "true", "yes", "on")

    # Only consider skipping when (a) the operator hasn't disabled the
    # optimization AND (b) alembic was invoked with target == "head".
    # The second guard prevents migration tests (`command.upgrade(cfg,
    # "050")`) from being silently no-op'd.
    can_skip = skip_enabled and _is_head_invocation()

    # IMPORTANT: the head check MUST run on its own short-lived connection,
    # fully closed before the migration connection is opened.
    #
    # `_is_at_head` issues a SELECT (get_current_heads), which under
    # SQLAlchemy 2.0 *autobegins* a transaction on that connection. If we did
    # the check on the same connection we then hand to `do_run_migrations`,
    # alembic's `begin_transaction()` would see a transaction it did not open,
    # decline to own it, and NOT commit on exit — so when the connection
    # closes, SQLAlchemy rolls back the entire migration span, INCLUDING the
    # alembic_version bumps. The DB then re-runs the same (idempotent) span on
    # every boot and the version never advances past where it was when the DB
    # first fell behind head. (This is exactly how prod got pinned at 063 while
    # 064–066 "ran" as no-ops every boot.) A separate, closed connection leaves
    # the migration connection pristine so alembic owns and commits its txn.
    if can_skip:
        t0 = time.monotonic()
        with connectable.connect() as check_conn:
            at_head = _is_at_head(check_conn)
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        if at_head:
            log.info(
                "[PERF] alembic_noop=true elapsed_ms=%d "
                "skipped_full_evaluation=true",
                elapsed_ms,
            )
            connectable.dispose()
            return
        log.info(
            "[PERF] alembic_noop=false elapsed_ms=%d "
            "head_check_ran_then_proceeding=true",
            elapsed_ms,
        )

    with connectable.connect() as connection:
        do_run_migrations(connection)

    connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
