"""Alembic migration 051 — additive runner-state columns on
``build_jobs``.

PR #45 of the cutover arc (PRs #45→#49). This migration is purely
additive — three nullable columns that subsequent PRs in the arc
will populate from the runner code paths. By itself, applying mig
051 changes nothing observable: the new columns sit NULL on every
existing row and are not yet read.

What we pin:

  1. revision='051', chains off '050'.
  2. Upgrade adds all three columns to existing build_jobs;
     pre-existing rows survive with the new columns set to NULL.
  3. Downgrade removes them cleanly.
  4. Idempotency on re-upgrade after downgrade.
  5. The ORM (app/db/models/app.py::BuildJob) declares the same
     columns so the migration and the model stay in sync.
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


@pytest_asyncio.fixture(autouse=True)
async def _reset_database():
    """Override conftest's autouse — this test drives Alembic
    against per-test SQLite temp files, not the app engine."""
    yield


# ──────────────────────────────────────────────────────────────────────
# Source-grep guards.
# ──────────────────────────────────────────────────────────────────────


_MIG_PATH = BACKEND_DIR / "alembic/versions/20260520_0051_051_build_jobs_runner_state_columns.py"
_MIG_SRC = _MIG_PATH.read_text() if _MIG_PATH.exists() else ""


def test_migration_file_exists_with_right_revision_chain():
    assert _MIG_PATH.exists(), f"migration 051 missing at {_MIG_PATH}"
    assert 'revision = "051"' in _MIG_SRC
    assert 'down_revision = "050"' in _MIG_SRC, (
        "051 must chain off 050 (the opt-in legacy-drop migration)"
    )


def test_orm_declares_all_three_new_columns():
    """The ORM (BuildJob) must declare the new columns so reads
    don't fall over after the migration runs. Pin the names so a
    rename refactor lights up here."""
    from app.db.models import BuildJob
    cols = {c.name for c in BuildJob.__table__.columns}
    for expected in ("fire_instant", "attempt", "coalesced_into_job_id"):
        assert expected in cols, (
            f"BuildJob ORM missing column {expected!r} after mig 051"
        )


# ──────────────────────────────────────────────────────────────────────
# End-to-end Alembic apply/downgrade/reapply against SQLite.
# ──────────────────────────────────────────────────────────────────────


def _make_alembic_config(db_url: str) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(BACKEND_DIR / "alembic"))
    cfg.set_main_option("sqlalchemy.url", db_url)
    cfg.set_main_option("prepend_sys_path", str(BACKEND_DIR))
    return cfg


_BUILD_JOBS_PRE_051_CREATE_SQL = (
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


def _set_required_env(monkeypatch, async_url: str) -> None:
    import importlib
    import sys

    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("DATABASE_URL", async_url)
    monkeypatch.setenv("JWT_SECRET", "test-jwt-secret-mig051")
    monkeypatch.setenv("ENCRYPTION_KEY", "test-32-byte-encryption-key--x12")
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_dummy_for_mig051")

    if "app.config" in sys.modules:
        importlib.reload(sys.modules["app.config"])


def _seed_pre_051_db(sync_url: str) -> None:
    """Pre-create build_jobs without the new columns and stamp
    alembic_version=050. Insert one row so we can prove the upgrade
    preserves existing data (NULL on the new columns)."""
    engine = sa.create_engine(sync_url)
    with engine.begin() as conn:
        conn.execute(sa.text(_BUILD_JOBS_PRE_051_CREATE_SQL))
        conn.execute(sa.text(
            "INSERT INTO build_jobs (id, user_id, title, prompt) "
            "VALUES ('preexisting', 'user-1', 'Pre-existing', 'p')"
        ))
        conn.execute(sa.text(
            "CREATE TABLE alembic_version (version_num VARCHAR(32) PRIMARY KEY)"
        ))
        conn.execute(sa.text(
            "INSERT INTO alembic_version (version_num) VALUES ('050')"
        ))
    engine.dispose()


@pytest.fixture
def fresh_db_url():
    fd, path = tempfile.mkstemp(suffix=".sqlite", prefix="mig051_")
    os.close(fd)
    yield f"sqlite+aiosqlite:///{path}", f"sqlite:///{path}"
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


# ──────────────────────────────────────────────────────────────────────
# Upgrade adds all three columns; pre-existing row survives.
# ──────────────────────────────────────────────────────────────────────


def test_upgrade_adds_all_three_columns_and_preserves_data(
    fresh_db_url, monkeypatch,
):
    async_url, sync_url = fresh_db_url
    _set_required_env(monkeypatch, async_url)
    _seed_pre_051_db(sync_url)

    cfg = _make_alembic_config(sync_url)
    command.upgrade(cfg, "051")

    engine = sa.create_engine(sync_url)
    with engine.connect() as conn:
        insp = sa.inspect(conn)
        cols = {c["name"] for c in insp.get_columns("build_jobs")}
        for expected in ("fire_instant", "attempt", "coalesced_into_job_id"):
            assert expected in cols, (
                f"mig 051 should add build_jobs.{expected}"
            )
        # Existing row survives with NULL on the new columns.
        row = conn.execute(sa.text(
            "SELECT fire_instant, attempt, coalesced_into_job_id "
            "FROM build_jobs WHERE id='preexisting'"
        )).fetchone()
        assert row is not None, (
            "mig 051 destroyed the pre-existing row — upgrade must "
            "be data-preserving"
        )
        assert row.fire_instant is None
        assert row.attempt is None
        assert row.coalesced_into_job_id is None

        version = conn.execute(
            sa.text("SELECT version_num FROM alembic_version")
        ).scalar_one()
        assert version == "051"
    engine.dispose()


# ──────────────────────────────────────────────────────────────────────
# Downgrade drops the columns cleanly.
# ──────────────────────────────────────────────────────────────────────


def test_downgrade_drops_all_three_columns(fresh_db_url, monkeypatch):
    async_url, sync_url = fresh_db_url
    _set_required_env(monkeypatch, async_url)
    _seed_pre_051_db(sync_url)

    cfg = _make_alembic_config(sync_url)
    command.upgrade(cfg, "051")
    command.downgrade(cfg, "050")

    engine = sa.create_engine(sync_url)
    with engine.connect() as conn:
        insp = sa.inspect(conn)
        cols = {c["name"] for c in insp.get_columns("build_jobs")}
        for not_expected in ("fire_instant", "attempt", "coalesced_into_job_id"):
            assert not_expected not in cols, (
                f"downgrade should have dropped build_jobs.{not_expected}"
            )
        # Pre-existing row still alive.
        n = conn.execute(sa.text(
            "SELECT COUNT(*) FROM build_jobs WHERE id='preexisting'"
        )).scalar_one()
        assert n == 1
    engine.dispose()


# ──────────────────────────────────────────────────────────────────────
# Idempotency on re-upgrade.
# ──────────────────────────────────────────────────────────────────────


def test_upgrade_then_downgrade_then_upgrade_is_idempotent(
    fresh_db_url, monkeypatch,
):
    async_url, sync_url = fresh_db_url
    _set_required_env(monkeypatch, async_url)
    _seed_pre_051_db(sync_url)

    cfg = _make_alembic_config(sync_url)
    command.upgrade(cfg, "051")
    command.downgrade(cfg, "050")
    command.upgrade(cfg, "051")  # must NOT raise

    engine = sa.create_engine(sync_url)
    with engine.connect() as conn:
        insp = sa.inspect(conn)
        cols = {c["name"] for c in insp.get_columns("build_jobs")}
        for expected in ("fire_instant", "attempt", "coalesced_into_job_id"):
            assert expected in cols
    engine.dispose()
