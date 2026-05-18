"""Alembic migration 046 — additive schema for the unified-jobs arc.

What we pin:

  1. Apply ``alembic upgrade 046`` from a clean DB state and assert
     every new column / table / index lands.
  2. ``alembic downgrade 045`` removes the new columns + the
     ``job_events`` table cleanly. ``build_jobs`` itself is NOT
     dropped (documented in the migration module — production rows
     would be destroyed).
  3. Re-applying ``upgrade 046`` after the downgrade succeeds without
     error (idempotency).

The test programmatically drives Alembic against an in-memory SQLite
DB so it stays self-contained and runs in CI without a Postgres
service. The migration's dialect-aware branches (the partial UNIQUE
index uses Postgres-style ``CREATE UNIQUE INDEX … WHERE`` syntax,
which SQLite also accepts) are exercised on SQLite here; the
Postgres path is exercised in the CI integration stage by the
service-container pytest job.
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


# Override the autouse `_reset_database` fixture from conftest.py for
# this module only — that fixture calls ``init_db()`` against the
# global engine, which runs a Postgres-only ``UPDATE memories SET
# history_json = ...`` data migration that errors on SQLite. Our
# tests create their own per-test SQLite files and bypass the global
# engine entirely, so init_db is both unnecessary and harmful here.
@pytest_asyncio.fixture(autouse=True)
async def _reset_database():
    yield


# ──────────────────────────────────────────────────────────────────────
# Source-grep guards — bug-recurrence pins.
# ──────────────────────────────────────────────────────────────────────


_MIG_PATH = BACKEND_DIR / "alembic/versions/20260518_0046_046_unified_job_model.py"
_MIG_SRC = _MIG_PATH.read_text() if _MIG_PATH.exists() else ""


def test_migration_file_exists_with_right_revision_chain():
    """The migration must claim revision='046' and chain off '045'.
    A stray copy-paste of an older down_revision would break alembic
    upgrade ordering silently — pin it."""
    assert _MIG_PATH.exists(), f"migration 046 missing at {_MIG_PATH}"
    assert "revision = \"046\"" in _MIG_SRC, "revision must be '046'"
    assert "down_revision = \"045\"" in _MIG_SRC, (
        "down_revision must be '045' (chains off the trigger-runner "
        "backfill migration)"
    )


def test_migration_does_not_drop_build_jobs_on_downgrade():
    """The module-level docstring promises the table is never dropped
    on downgrade — production data would be destroyed. Source-grep
    the downgrade body for the negative invariant."""
    # Find the downgrade() function body.
    start = _MIG_SRC.find("def downgrade()")
    assert start != -1
    end = _MIG_SRC.find("\ndef ", start + 1)
    body = _MIG_SRC[start:end if end > -1 else None]
    assert "drop_table(\"build_jobs\")" not in body, (
        "downgrade() must NOT drop build_jobs — production data lives "
        "there. The asymmetry is documented in the module docstring."
    )
    assert "drop_table('build_jobs')" not in body, (
        "downgrade() must NOT drop build_jobs (single-quoted form)."
    )


def test_migration_is_idempotent_on_existing_table():
    """If build_jobs already exists, upgrade() must use add_column with
    a column_exists guard, not create_table. A bare create_table
    against an existing table errors. Pin the guard."""
    assert "_column_exists(conn, \"build_jobs\"" in _MIG_SRC, (
        "upgrade() must check _column_exists before add_column so "
        "re-runs are idempotent."
    )
    assert "_table_exists(conn, \"build_jobs\")" in _MIG_SRC, (
        "upgrade() must check _table_exists before deciding to "
        "create_table or add_column."
    )


# ──────────────────────────────────────────────────────────────────────
# End-to-end Alembic apply/downgrade/reapply against SQLite.
# ──────────────────────────────────────────────────────────────────────


def _make_alembic_config(db_url: str) -> Config:
    """Build an Alembic Config that points at the in-test DB. We
    create it programmatically rather than reading alembic.ini so the
    test doesn't mutate the project's config file."""
    cfg = Config()
    cfg.set_main_option("script_location", str(BACKEND_DIR / "alembic"))
    cfg.set_main_option("sqlalchemy.url", db_url)
    cfg.set_main_option("prepend_sys_path", str(BACKEND_DIR))
    return cfg


# Columns added to build_jobs by migration 046, mirrored here so the
# test fails loudly if migration and ORM drift apart.
_NEW_BUILD_JOB_COLUMNS = {
    "source_kind",
    "source_id",
    "conversation_id",
    "summary_message_id",
    "outcome",
    "idempotency_key",
}


def _assert_post_upgrade_state(conn) -> None:
    """Every invariant the migration is supposed to land."""
    insp = sa.inspect(conn)

    # build_jobs exists with all the new columns.
    assert "build_jobs" in insp.get_table_names(), (
        "upgrade() should have created build_jobs (fresh DB path)"
    )
    cols = {c["name"] for c in insp.get_columns("build_jobs")}
    missing = _NEW_BUILD_JOB_COLUMNS - cols
    assert not missing, f"build_jobs missing new columns: {missing}"

    # The historical columns must still be present (no accidental
    # drop / rename).
    for legacy_col in ("id", "user_id", "title", "prompt", "job_type",
                        "status", "steps_json", "build_logs_json"):
        assert legacy_col in cols, (
            f"build_jobs lost legacy column {legacy_col!r} — migration "
            f"must be purely additive."
        )

    # job_events table + indexes.
    assert "job_events" in insp.get_table_names(), "job_events not created"
    job_event_cols = {c["name"] for c in insp.get_columns("job_events")}
    for expected in ("id", "job_id", "user_id", "ts", "kind",
                     "label", "status", "level", "metadata_json"):
        assert expected in job_event_cols, (
            f"job_events missing column {expected!r}"
        )

    idx_names = {i["name"] for i in insp.get_indexes("job_events")}
    assert "ix_job_events_user_ts" in idx_names, (
        "ix_job_events_user_ts index missing — activity-feed query "
        "scans without it."
    )
    assert "ix_job_events_job_ts" in idx_names, (
        "ix_job_events_job_ts index missing — job detail drawer "
        "scans without it."
    )


def _assert_post_downgrade_state(conn) -> None:
    """Every invariant the migration is supposed to revert.
    build_jobs stays (the asymmetry documented in the module). The
    new columns and job_events are gone."""
    insp = sa.inspect(conn)

    # job_events is dropped.
    assert "job_events" not in insp.get_table_names(), (
        "downgrade() should have dropped job_events"
    )

    # build_jobs survives but without the new columns.
    if "build_jobs" in insp.get_table_names():
        cols = {c["name"] for c in insp.get_columns("build_jobs")}
        for new_col in _NEW_BUILD_JOB_COLUMNS:
            assert new_col not in cols, (
                f"downgrade() failed to drop build_jobs.{new_col}"
            )


@pytest.fixture
def fresh_db_url():
    """Per-test SQLite file. In-memory ``:memory:`` doesn't survive
    the Alembic Engine teardown between commands, so we use a temp
    file the test cleans up explicitly.

    Returns a tuple ``(async_url, sync_url)``. The async URL goes
    into ``DATABASE_URL`` (which ``app.db.database`` consumes via
    ``create_async_engine``); the sync URL is what
    ``sa.create_engine`` and alembic actually use to talk to the DB.
    They point at the same SQLite file."""
    fd, path = tempfile.mkstemp(suffix=".sqlite", prefix="mig046_")
    os.close(fd)
    yield f"sqlite+aiosqlite:///{path}", f"sqlite:///{path}"
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


# ── Test helpers ──────────────────────────────────────────────────────


_BUILD_JOBS_CREATE_SQL = (
    "CREATE TABLE build_jobs ("
    "  id VARCHAR(36) PRIMARY KEY,"
    "  user_id VARCHAR(36) NOT NULL,"
    "  app_id VARCHAR(36),"
    "  title VARCHAR(200) NOT NULL,"
    "  prompt TEXT NOT NULL,"
    "  job_type VARCHAR(20) DEFAULT 'auto_builder',"
    "  status VARCHAR(20) DEFAULT 'queued',"
    "  steps_json TEXT DEFAULT '[]',"
    "  model VARCHAR(50) DEFAULT '',"
    "  total_tokens INTEGER DEFAULT 0,"
    "  error_message TEXT,"
    "  build_logs_json TEXT DEFAULT '[]',"
    "  paused_at TIMESTAMP,"
    "  resume_after TIMESTAMP,"
    "  checkpoint_json TEXT,"
    "  layer INTEGER DEFAULT 1,"
    "  layer2_changes_json TEXT,"
    "  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,"
    "  completed_at TIMESTAMP"
    ")"
)


def _set_required_env(monkeypatch, async_url: str) -> None:
    """alembic env.py imports app.config.settings which validates a
    handful of env vars on import. Set them once per test so the
    test author doesn't have to.

    Also FORCE-RELOAD ``app.config`` so the module-level ``settings``
    singleton picks up the new ``DATABASE_URL`` instead of returning
    the value cached from the first test that ran. Without this,
    every subsequent test's alembic call would talk to test #1's
    (now-deleted) SQLite file."""
    import importlib

    monkeypatch.setenv("ENVIRONMENT", "test")
    # app.db.database creates an AsyncEngine from DATABASE_URL at
    # module-import time, so it needs the +aiosqlite driver. Alembic's
    # env.py strips that suffix before handing off to its sync engine.
    monkeypatch.setenv("DATABASE_URL", async_url)
    monkeypatch.setenv("JWT_SECRET", "test-jwt-secret-mig046")
    monkeypatch.setenv("ENCRYPTION_KEY", "test-32-byte-encryption-key--x12")
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_dummy_for_mig046")

    # If app.config has already been imported by a prior test, reload
    # it so Settings() is re-instantiated against the fresh env vars
    # above. alembic env.py does ``from app.config import settings``
    # at every command call; reloading ensures that import picks up
    # the new DATABASE_URL.
    import sys
    if "app.config" in sys.modules:
        importlib.reload(sys.modules["app.config"])


def _seed_pre_046_db(sync_url: str, with_row: bool = False) -> None:
    """Pre-create build_jobs WITHOUT the new columns and stamp
    alembic_version=045, mimicking the production reality: every
    agent has build_jobs from ``Base.metadata.create_all()`` long
    before 046 runs. Optionally insert a historical row to prove
    the migration doesn't destroy data."""
    engine = sa.create_engine(sync_url)
    with engine.begin() as conn:
        conn.execute(sa.text(_BUILD_JOBS_CREATE_SQL))
        if with_row:
            conn.execute(sa.text(
                "INSERT INTO build_jobs (id, user_id, title, prompt) "
                "VALUES ('pre-existing-row', 'user-1', 'Pre-existing', 'p')"
            ))
    engine.dispose()


# ── End-to-end Alembic apply/downgrade/reapply ───────────────────────


def test_upgrade_045_to_046_with_preexisting_build_jobs(
    fresh_db_url, monkeypatch,
):
    """Production-shaped path: build_jobs already exists from
    ``Base.metadata.create_all()``; alembic_version is stamped at
    045 (post the trigger-runner backfill PR). The migration must
    take the ``add_column`` branch and leave existing rows alone."""
    async_url, sync_url = fresh_db_url
    _set_required_env(monkeypatch, async_url)
    _seed_pre_046_db(sync_url, with_row=True)

    cfg = _make_alembic_config(sync_url)
    command.stamp(cfg, "045")
    command.upgrade(cfg, "046")

    engine = sa.create_engine(sync_url)
    try:
        with engine.connect() as conn:
            _assert_post_upgrade_state(conn)
            # Pre-existing row survives the migration verbatim.
            result = conn.execute(sa.text(
                "SELECT title FROM build_jobs WHERE id='pre-existing-row'"
            )).fetchone()
            assert result is not None, "migration destroyed pre-existing row"
            assert result[0] == "Pre-existing"
    finally:
        engine.dispose()


def test_downgrade_046_to_045_reverts_columns_and_drops_job_events(
    fresh_db_url, monkeypatch,
):
    """After upgrade 046 + downgrade 045: ``job_events`` is gone,
    the new columns on ``build_jobs`` are gone, but ``build_jobs``
    itself stays (the documented asymmetry — production data lives
    there)."""
    async_url, sync_url = fresh_db_url
    _set_required_env(monkeypatch, async_url)
    _seed_pre_046_db(sync_url, with_row=True)

    cfg = _make_alembic_config(sync_url)
    command.stamp(cfg, "045")
    command.upgrade(cfg, "046")
    command.downgrade(cfg, "045")

    engine = sa.create_engine(sync_url)
    try:
        with engine.connect() as conn:
            _assert_post_downgrade_state(conn)
            # build_jobs itself must still exist (asymmetry).
            insp = sa.inspect(conn)
            assert "build_jobs" in insp.get_table_names(), (
                "downgrade() must NOT drop build_jobs — production "
                "data lives there."
            )
            # Pre-existing row still intact.
            result = conn.execute(sa.text(
                "SELECT title FROM build_jobs WHERE id='pre-existing-row'"
            )).fetchone()
            assert result is not None and result[0] == "Pre-existing"
    finally:
        engine.dispose()


def test_reapply_046_after_downgrade_is_idempotent(
    fresh_db_url, monkeypatch,
):
    """Full reversibility cycle: 045 → 046 → 045 → 046. The second
    upgrade must succeed without errors — i.e. the migration's
    column-exists and table-exists guards work on both fresh and
    re-applied paths."""
    async_url, sync_url = fresh_db_url
    _set_required_env(monkeypatch, async_url)
    _seed_pre_046_db(sync_url)

    cfg = _make_alembic_config(sync_url)
    command.stamp(cfg, "045")
    command.upgrade(cfg, "046")
    command.downgrade(cfg, "045")
    command.upgrade(cfg, "046")  # <— the load-bearing call

    engine = sa.create_engine(sync_url)
    try:
        with engine.connect() as conn:
            _assert_post_upgrade_state(conn)
    finally:
        engine.dispose()
