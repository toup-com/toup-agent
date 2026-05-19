"""Alembic migration 050 — opt-in drop of legacy job tables.

What we pin:

  1. The migration claims revision='050', chains off '049'.
  2. Gate OFF (env var unset / 0 / false): ``upgrade()`` is a strict
     NO-OP. ``trigger_events`` + ``routine_runs`` + ``build_jobs.
     steps_json`` survive untouched.
  3. Gate ON (``ALLOW_LEGACY_JOB_TABLES_DROP=true``): every legacy
     table is dropped and ``steps_json`` is gone from ``build_jobs``.
  4. ``downgrade()`` re-creates the dropped tables + the column
     (empty — structural only; data restoration is out of scope).
  5. Re-running ``upgrade 050`` after the downgrade is idempotent.

This test pins the migration on SQLite (in-memory file). The
Postgres path of the same DDL is exercised in the prod-shaped CI
service-container job; here we just verify the dialect-portable
surface of the migration (drop_table / drop_column / batch_alter).
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
import pytest_asyncio
import sqlalchemy as sa
from alembic import command
from alembic.config import Config


BACKEND_DIR = Path(__file__).resolve().parent.parent


# Override conftest's autouse `_reset_database` for this module so it
# doesn't run init_db on the global app engine — every test here
# builds its own per-file SQLite DB and drives Alembic against it.
@pytest_asyncio.fixture(autouse=True)
async def _reset_database():
    yield


# ──────────────────────────────────────────────────────────────────────
# Source-grep guards — bug-recurrence pins.
# ──────────────────────────────────────────────────────────────────────


_MIG_PATH = BACKEND_DIR / "alembic/versions/20260519_0050_050_optin_drop_legacy_job_tables.py"
_MIG_SRC = _MIG_PATH.read_text() if _MIG_PATH.exists() else ""


def test_migration_file_exists_with_right_revision_chain():
    """The migration must claim revision='050' and chain off '049'.
    A stray copy-paste of an older down_revision would break alembic
    upgrade ordering silently — pin it."""
    assert _MIG_PATH.exists(), f"migration 050 missing at {_MIG_PATH}"
    assert 'revision = "050"' in _MIG_SRC, "revision must be '050'"
    assert 'down_revision = "049"' in _MIG_SRC, (
        "down_revision must be '049' (chains off the cross-channel "
        "reply_to migration)"
    )


def test_migration_reads_gate_env_var_in_upgrade():
    """The destructive drop is gated by ALLOW_LEGACY_JOB_TABLES_DROP.
    The migration must read the env var INSIDE upgrade() (not at
    module import) so tests can monkeypatch os.environ before
    invoking alembic. Pin the variable name and the guard pattern."""
    assert "ALLOW_LEGACY_JOB_TABLES_DROP" in _MIG_SRC, (
        "gate variable name must be ALLOW_LEGACY_JOB_TABLES_DROP — "
        "any rename here breaks the runbook + parity-check script."
    )
    # Find the upgrade() body and assert the gate read lives there.
    start = _MIG_SRC.find("def upgrade()")
    assert start != -1
    end = _MIG_SRC.find("\ndef ", start + 1)
    body = _MIG_SRC[start:end if end > -1 else None]
    assert "os.environ" in body, (
        "upgrade() must read os.environ for the gate at runtime."
    )


# ──────────────────────────────────────────────────────────────────────
# End-to-end Alembic gate-off / gate-on coverage.
# ──────────────────────────────────────────────────────────────────────


def _make_alembic_config(db_url: str) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(BACKEND_DIR / "alembic"))
    cfg.set_main_option("sqlalchemy.url", db_url)
    cfg.set_main_option("prepend_sys_path", str(BACKEND_DIR))
    return cfg


# Minimal table shape the migration cares about — we seed at 049 then
# run 050 on top.
_BUILD_JOBS_CREATE_SQL = (
    "CREATE TABLE build_jobs ("
    "  id VARCHAR(36) PRIMARY KEY,"
    "  user_id VARCHAR(36) NOT NULL,"
    "  title VARCHAR(200) NOT NULL,"
    "  prompt TEXT NOT NULL,"
    "  job_type VARCHAR(20) DEFAULT 'auto_builder',"
    "  status VARCHAR(20) DEFAULT 'queued',"
    "  steps_json TEXT DEFAULT '[]',"
    "  build_logs_json TEXT DEFAULT '[]',"
    "  source_kind VARCHAR(20),"
    "  source_id VARCHAR(36),"
    "  idempotency_key VARCHAR(120),"
    "  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
    ")"
)

_TRIGGER_EVENTS_CREATE_SQL = (
    "CREATE TABLE trigger_events ("
    "  id VARCHAR(36) PRIMARY KEY,"
    "  trigger_id VARCHAR(36) NOT NULL,"
    "  user_id VARCHAR(36) NOT NULL,"
    "  event_dedupe_id VARCHAR(255) NOT NULL,"
    "  received_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,"
    "  status VARCHAR(20) DEFAULT 'queued',"
    "  job_id VARCHAR(36)"
    ")"
)

_ROUTINE_RUNS_CREATE_SQL = (
    "CREATE TABLE routine_runs ("
    "  id VARCHAR(36) PRIMARY KEY,"
    "  routine_id VARCHAR(36) NOT NULL,"
    "  user_id VARCHAR(36) NOT NULL,"
    "  scheduled_for_local_date DATE NOT NULL,"
    "  started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,"
    "  status VARCHAR(20) DEFAULT 'running',"
    "  job_id VARCHAR(36)"
    ")"
)


def _set_required_env(monkeypatch, async_url: str) -> None:
    """app.config validates a handful of env vars on import. Same
    shape as test_migration_046_unified_job_model._set_required_env."""
    import importlib
    import sys

    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("DATABASE_URL", async_url)
    monkeypatch.setenv("JWT_SECRET", "test-jwt-secret-mig050")
    monkeypatch.setenv("ENCRYPTION_KEY", "test-32-byte-encryption-key--x12")
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_dummy_for_mig050")

    if "app.config" in sys.modules:
        importlib.reload(sys.modules["app.config"])


def _seed_pre_050_db(sync_url: str) -> None:
    """Pre-create build_jobs + trigger_events + routine_runs and
    stamp alembic_version=049, simulating the dual-write state of an
    agent that has run migrations 046..049 but not yet 050."""
    engine = sa.create_engine(sync_url)
    with engine.begin() as conn:
        conn.execute(sa.text(_BUILD_JOBS_CREATE_SQL))
        conn.execute(sa.text(_TRIGGER_EVENTS_CREATE_SQL))
        conn.execute(sa.text(_ROUTINE_RUNS_CREATE_SQL))
        conn.execute(sa.text(
            "INSERT INTO build_jobs (id, user_id, title, prompt, "
            "steps_json) VALUES ('bj-1', 'user-1', 'Pre-existing', "
            "'prompt', '[\"step\"]')"
        ))
        conn.execute(sa.text(
            "INSERT INTO trigger_events (id, trigger_id, user_id, "
            "event_dedupe_id, job_id) VALUES ('te-1', 'trig-1', "
            "'user-1', 'msg-1', 'bj-1')"
        ))
        conn.execute(sa.text(
            "INSERT INTO routine_runs (id, routine_id, user_id, "
            "scheduled_for_local_date, job_id) VALUES ('rr-1', "
            "'rout-1', 'user-1', '2026-05-19', 'bj-1')"
        ))
        # alembic_version table — stamp at 049 so the next upgrade
        # picks up just 050.
        conn.execute(sa.text(
            "CREATE TABLE alembic_version (version_num VARCHAR(32) "
            "PRIMARY KEY)"
        ))
        conn.execute(sa.text(
            "INSERT INTO alembic_version (version_num) VALUES ('049')"
        ))
    engine.dispose()


@pytest.fixture
def fresh_db_url():
    """Per-test SQLite file — see test_migration_046 for why we use
    a temp file rather than :memory:."""
    fd, path = tempfile.mkstemp(suffix=".sqlite", prefix="mig050_")
    os.close(fd)
    yield f"sqlite+aiosqlite:///{path}", f"sqlite:///{path}"
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


# ──────────────────────────────────────────────────────────────────────
# Gate OFF: upgrade is a no-op.
# ──────────────────────────────────────────────────────────────────────


def test_upgrade_with_gate_off_is_noop(fresh_db_url, monkeypatch):
    """Default deploy path: operator runs ``alembic upgrade head``
    without setting the gate. Every legacy table + column must
    survive untouched."""
    async_url, sync_url = fresh_db_url
    _set_required_env(monkeypatch, async_url)
    monkeypatch.delenv("ALLOW_LEGACY_JOB_TABLES_DROP", raising=False)
    _seed_pre_050_db(sync_url)

    cfg = _make_alembic_config(sync_url)
    command.upgrade(cfg, "050")

    engine = sa.create_engine(sync_url)
    with engine.connect() as conn:
        insp = sa.inspect(conn)
        names = set(insp.get_table_names())
        assert "trigger_events" in names, (
            "gate OFF must leave trigger_events alone"
        )
        assert "routine_runs" in names, (
            "gate OFF must leave routine_runs alone"
        )
        bj_cols = {c["name"] for c in insp.get_columns("build_jobs")}
        assert "steps_json" in bj_cols, (
            "gate OFF must leave build_jobs.steps_json alone"
        )

        # Stamped version progressed to 050 even though upgrade was a
        # no-op — that's the whole point of the opt-in pattern, the
        # version moves forward so the next migration chains cleanly.
        version = conn.execute(
            sa.text("SELECT version_num FROM alembic_version")
        ).scalar_one()
        assert version == "050"

        # Pre-seeded data is intact.
        n_te = conn.execute(
            sa.text("SELECT COUNT(*) FROM trigger_events")
        ).scalar_one()
        assert n_te == 1
        n_rr = conn.execute(
            sa.text("SELECT COUNT(*) FROM routine_runs")
        ).scalar_one()
        assert n_rr == 1
    engine.dispose()


@pytest.mark.parametrize("gate_value", ["", "0", "false", "no", "off"])
def test_upgrade_gate_falsy_values_treated_as_off(
    fresh_db_url, monkeypatch, gate_value,
):
    """Defensive: a typo like ``ALLOW_LEGACY_JOB_TABLES_DROP=False``
    or ``=0`` must NOT be treated as ON. Pin the truthy set
    explicitly so a permissive ``bool(env)`` regression doesn't
    accidentally drop production data."""
    async_url, sync_url = fresh_db_url
    _set_required_env(monkeypatch, async_url)
    monkeypatch.setenv("ALLOW_LEGACY_JOB_TABLES_DROP", gate_value)
    _seed_pre_050_db(sync_url)

    cfg = _make_alembic_config(sync_url)
    command.upgrade(cfg, "050")

    engine = sa.create_engine(sync_url)
    with engine.connect() as conn:
        insp = sa.inspect(conn)
        assert "trigger_events" in insp.get_table_names()
        assert "routine_runs" in insp.get_table_names()
        bj_cols = {c["name"] for c in insp.get_columns("build_jobs")}
        assert "steps_json" in bj_cols
    engine.dispose()


# ──────────────────────────────────────────────────────────────────────
# Gate ON: upgrade drops legacy tables + steps_json.
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("gate_value", ["true", "1", "yes", "on", "TRUE", "True"])
def test_upgrade_with_gate_on_drops_legacy(
    fresh_db_url, monkeypatch, gate_value,
):
    """With the gate explicitly enabled, every legacy surface goes
    away. Pin the truthy values the migration accepts."""
    async_url, sync_url = fresh_db_url
    _set_required_env(monkeypatch, async_url)
    monkeypatch.setenv("ALLOW_LEGACY_JOB_TABLES_DROP", gate_value)
    _seed_pre_050_db(sync_url)

    cfg = _make_alembic_config(sync_url)
    command.upgrade(cfg, "050")

    engine = sa.create_engine(sync_url)
    with engine.connect() as conn:
        insp = sa.inspect(conn)
        names = set(insp.get_table_names())
        assert "trigger_events" not in names, (
            f"gate ON ({gate_value!r}) must drop trigger_events"
        )
        assert "routine_runs" not in names, (
            f"gate ON ({gate_value!r}) must drop routine_runs"
        )
        # build_jobs survives — only steps_json is removed.
        assert "build_jobs" in names
        bj_cols = {c["name"] for c in insp.get_columns("build_jobs")}
        assert "steps_json" not in bj_cols, (
            f"gate ON ({gate_value!r}) must drop build_jobs.steps_json"
        )
        # A historical build_jobs row survives the column drop.
        n_bj = conn.execute(
            sa.text("SELECT COUNT(*) FROM build_jobs WHERE id='bj-1'")
        ).scalar_one()
        assert n_bj == 1, "dropping steps_json must not destroy build_jobs rows"
    engine.dispose()


# ──────────────────────────────────────────────────────────────────────
# Downgrade re-creates the structures (empty).
# ──────────────────────────────────────────────────────────────────────


def test_downgrade_recreates_legacy_tables_empty(fresh_db_url, monkeypatch):
    """After a gate-ON upgrade the legacy tables are gone. A
    downgrade must put empty shells back so the older code paths
    can boot without immediate schema errors. Data is NOT restored
    (documented in the migration module)."""
    async_url, sync_url = fresh_db_url
    _set_required_env(monkeypatch, async_url)
    monkeypatch.setenv("ALLOW_LEGACY_JOB_TABLES_DROP", "true")
    _seed_pre_050_db(sync_url)

    cfg = _make_alembic_config(sync_url)
    command.upgrade(cfg, "050")
    command.downgrade(cfg, "049")

    engine = sa.create_engine(sync_url)
    with engine.connect() as conn:
        insp = sa.inspect(conn)
        names = set(insp.get_table_names())
        assert "trigger_events" in names, (
            "downgrade should have re-created trigger_events"
        )
        assert "routine_runs" in names, (
            "downgrade should have re-created routine_runs"
        )
        bj_cols = {c["name"] for c in insp.get_columns("build_jobs")}
        assert "steps_json" in bj_cols, (
            "downgrade should have re-added build_jobs.steps_json"
        )

        # The shells are empty — downgrade is structural-only.
        n_te = conn.execute(
            sa.text("SELECT COUNT(*) FROM trigger_events")
        ).scalar_one()
        assert n_te == 0, "downgrade must NOT restore trigger_events rows"
        n_rr = conn.execute(
            sa.text("SELECT COUNT(*) FROM routine_runs")
        ).scalar_one()
        assert n_rr == 0, "downgrade must NOT restore routine_runs rows"
    engine.dispose()


# ──────────────────────────────────────────────────────────────────────
# Idempotency — re-applying upgrade after downgrade.
# ──────────────────────────────────────────────────────────────────────


def test_upgrade_then_downgrade_then_upgrade_is_idempotent(
    fresh_db_url, monkeypatch,
):
    """An operator who needs to roll forward, back, and forward again
    must not hit a constraint or DDL error on the second upgrade."""
    async_url, sync_url = fresh_db_url
    _set_required_env(monkeypatch, async_url)
    monkeypatch.setenv("ALLOW_LEGACY_JOB_TABLES_DROP", "true")
    _seed_pre_050_db(sync_url)

    cfg = _make_alembic_config(sync_url)
    command.upgrade(cfg, "050")
    command.downgrade(cfg, "049")
    # Re-upgrade — must succeed without "table already exists" or
    # "column already missing" errors.
    command.upgrade(cfg, "050")

    engine = sa.create_engine(sync_url)
    with engine.connect() as conn:
        insp = sa.inspect(conn)
        assert "trigger_events" not in set(insp.get_table_names())
        assert "routine_runs" not in set(insp.get_table_names())
        bj_cols = {c["name"] for c in insp.get_columns("build_jobs")}
        assert "steps_json" not in bj_cols
    engine.dispose()
