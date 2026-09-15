"""
Alembic environment configuration for Toup.
Supports async SQLAlchemy with PostgreSQL.


DESIGN NOTE — this chain does not run on a tenant database, and fixing 007
did not change that. Read before "fixing" the agent's boot.
─────────────────────────────────────────────────────────────────────────
`Dockerfile.agent`'s CMD runs `alembic upgrade head || echo '[boot] alembic
upgrade failed — continuing'` before uvicorn, against the TENANT database.
It has never once succeeded. Verified on the live fleet 2026-09-13:
`to_regclass('alembic_version')` is NULL on `toup_agent_feed0081`,
`toup_agent_feed0086` and `toup_agent_f261b564` alike, each with 62 public
tables. The whole fleet, every boot, since 2026-02-13.

Reproduced locally against a disposable `pgvector/pgvector:pg16` (the exact
sequence is in the commit that added this note):

  1. empty DB + `alembic upgrade head`
     → dies at 007 on `document_chunks`, which NO migration creates. Alembic
       runs the span in one transaction ("Will assume transactional DDL"), so
       001-006 roll back and `alembic_version` is never written. 0 tables.
  2. `init_db()` on that empty DB
     → 62 tables, matching the live containers exactly. `create_all` plus
       ~223 hand-maintained `_alter_statements`, with the platform-only ones
       logging `[init_db] alter skipped: … relation "live_activities" does
       not exist` and similar.
  3. `alembic upgrade head` again, which is the fleet's SECOND boot
     → `DuplicateTable: relation "users" already exists`, because step 1
       stamped nothing and 001 re-runs against a populated schema.

Guarding 007 (done) moves the wall; it does not remove it. Walking the chain
one revision at a time on a fresh DB — each its own transaction, so a failure
does not roll back its predecessors — reaches head 101 but with **11
migrations failing**:

  tenant tables that only `create_all` creates
    021 day_chats · 040 routine_runs · 041 routines
  platform-only tables a tenant DB will never have
    022, 037, 060 managed_containers · 023 rollout_attempts ·
    025 streaming_credentials · 027 ix_managed_containers_host_port ·
    033 rollouts
  a data backfill
    086 `msg.metadata_json` (users.first_media_played_at)

That distribution is the finding. Seven of the eleven touch tables that exist
only in the PLATFORM database, so this is not a chain with a bug in it — it is
a chain written for a different database, pointed at a tenant. Migration 086's
own docstring already says so in plain words: *"Tenant DBs have no
alembic_version row — their only migrator is app/db/database.py::init_db's
`_alter_statements` list, which carries the mirror of this ALTER."*

THE OPEN QUESTION: should boot `alembic stamp head` when `alembic_version` is
absent but the schema is already populated? Not done here, deliberately.

  What it buys: the `DuplicateTable` on every boot stops, and `ec99e8bb`'s
  `/agent/health.schema` starts reporting something true.

  What it risks:
  • It declares 11 migrations applied that have never run. 086's backfill
    genuinely has not — its ALTER is mirrored at `database.py:447` but its
    UPDATE is mirrored NOWHERE, so `users.first_media_played_at` is NULL for
    every pre-existing tenant row and the Media nav entry stays hidden for
    users who have played something. Any future migration that assumes 086's
    data landed would be wrong in a way nothing detects.
  • DOUBLE-APPLY. Stamping head turns the future on: migration 102+ would then
    run on tenant DBs on top of `_alter_statements` doing the same work. The
    overlap already exists — `_alter_statements` carries 7 UPDATEs, of which
    `UPDATE build_jobs SET job_type='auto_builder' WHERE job_type IS NULL` and
    the `messages.channel` backfill are tenant tables also written by
    migrations. Those two are idempotent by predicate (`WHERE … IS NULL`), so
    today the collision is harmless — by luck, not by design. A column
    removal, a type change, or a backfill without an idempotent predicate
    would not be.
  • A stamp is irreversible per database and there are ~96 of them.

  How to verify it before shipping, on a disposable Postgres (never the fleet):
    docker run --rm -d -p 55444:5432 -e POSTGRES_PASSWORD=x \
      --name probe pgvector/pgvector:pg16
    # A: the claim that stamping is safe TODAY
    createdb probe_a; init_db(); alembic stamp head; alembic upgrade head
      → must be a clean no-op.
    # B: the claim that matters — the NEXT migration
    write a throwaway 102 that both ALTERs and backfills, mirror it in
    `_alter_statements` as the house style requires, then boot twice.
      → if the backfill runs twice, option (b) is unsafe as stated and the
        mirror rule has to change with it.
    # C: the honest alternative
    drop `alembic upgrade head` from Dockerfile.agent's CMD and boot.
      → nothing should change, because `init_db` is already the only thing
        delivering tenant schema. If that holds, (c) is the real fix and (b)
        is a way of keeping a mechanism that does no work.

  Recommendation: (c) — the chain is platform-authored and the agent should
  not run it — with (b) acceptable only if test B above comes back clean AND
  the `_alter_statements` mirror rule is rewritten to say which side owns a
  backfill. Do not ship either as part of an incident fix; both change what
  every container does to its own database at boot.
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
