"""Alembic migration 056 — additive sub-agent columns on ``build_jobs``.

Phase 1 of the sub-agent spawning arc. Purely additive: four columns
plus one composite index. By itself, applying mig 056 changes
nothing observable — the new columns sit NULL on every existing row
(except ``credit_spent`` which gets the server_default 0.0) and are
not yet read.

What we pin:

  1. revision='056', chains off '055'.
  2. Upgrade adds all four columns and the composite index;
     pre-existing rows survive (NULL on the three nullable cols,
     0.0 on credit_spent via server_default).
  3. Downgrade removes them cleanly.
  4. Idempotency on re-upgrade after downgrade.
  5. The ORM (app/db/models/app.py::BuildJob) declares the same
     columns so the migration and the model stay in sync.
  6. The composite index ``ix_build_jobs_parent_status`` exists with
     the expected column order — sub-agent cap-check queries depend
     on it.

Mirrors ``test_migration_052_routine_terminal_columns.py`` shape
one-for-one — only the new column list and the pre-state SQL differ
(055 must already have run; the pre-state CREATE includes the
052 columns since 053/053a/054/055 do not touch build_jobs).
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


_MIG_PATH = (
    BACKEND_DIR
    / "alembic/versions/20260522_0056_056_build_jobs_subagent_columns.py"
)
_MIG_SRC = _MIG_PATH.read_text() if _MIG_PATH.exists() else ""


_EXPECTED_NEW_COLUMNS = (
    "parent_job_id",
    "config_json",
    "credit_budget_allocated",
    "credit_spent",
)
_EXPECTED_INDEX = "ix_build_jobs_parent_status"
_EXPECTED_INDEX_COLS = ("parent_job_id", "status")


def test_migration_file_exists_with_right_revision_chain():
    assert _MIG_PATH.exists(), f"migration 056 missing at {_MIG_PATH}"
    assert 'revision = "056"' in _MIG_SRC
    assert 'down_revision = "055"' in _MIG_SRC, (
        "056 must chain off 055 (additive routine-terminal columns)"
    )


def test_orm_declares_all_four_new_columns():
    """The ORM (BuildJob) must declare the new columns so reads
    don't fall over after the migration runs. Pin the names so a
    rename refactor lights up here."""
    from app.db.models import BuildJob
    cols = {c.name for c in BuildJob.__table__.columns}
    for expected in _EXPECTED_NEW_COLUMNS:
        assert expected in cols, (
            f"BuildJob ORM missing column {expected!r} after mig 056"
        )


def test_orm_credit_spent_is_not_null_with_default():
    """credit_spent is NOT NULL with default 0.0 — aggregations
    over it must not need COALESCE. Pin both the nullability and the
    default so a careless ORM refactor doesn't silently relax it."""
    from app.db.models import BuildJob
    col = BuildJob.__table__.columns["credit_spent"]
    assert col.nullable is False, (
        "credit_spent must be NOT NULL so SUM/AVG don't need COALESCE"
    )
    # server_default is a DefaultClause; render its arg as string.
    sd = col.server_default
    assert sd is not None, (
        "credit_spent must have a server_default for existing-row backfill"
    )
    rendered = str(getattr(sd, "arg", sd)).strip("'\"")
    assert rendered == "0.0", (
        f"credit_spent server_default must be 0.0, got {rendered!r}"
    )


def test_orm_parent_job_id_is_indexed():
    """parent_job_id must be indexed; the depth-walk on every spawn
    follows the FK and the cap-check counts children — both pay for
    the index."""
    from app.db.models import BuildJob
    col = BuildJob.__table__.columns["parent_job_id"]
    assert col.index is True or any(
        "parent_job_id" in ix.columns
        for ix in BuildJob.__table__.indexes
    ), "parent_job_id must be indexed (depth walk + child cap queries)"


# ──────────────────────────────────────────────────────────────────────
# End-to-end Alembic apply/downgrade/reapply against SQLite.
# ──────────────────────────────────────────────────────────────────────


def _make_alembic_config(db_url: str) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(BACKEND_DIR / "alembic"))
    cfg.set_main_option("sqlalchemy.url", db_url)
    cfg.set_main_option("prepend_sys_path", str(BACKEND_DIR))
    return cfg


# pre-056 schema: build_jobs with the mig 052 additions (the
# routine-terminal JSON columns) plus enough of the prior schema for
# an INSERT to succeed. One row inserted so we can prove the upgrade
# is data-preserving.
_BUILD_JOBS_PRE_056_CREATE_SQL = (
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
    "  fire_instant DATETIME,"
    "  attempt INTEGER,"
    "  coalesced_into_job_id VARCHAR(36),"
    "  emails_fetched INTEGER,"
    "  finished_local_at VARCHAR(40),"
    "  error_json TEXT,"
    "  channel_results_json TEXT,"
    "  tools_invoked_json TEXT,"
    "  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
    ")"
)


def _set_required_env(monkeypatch, async_url: str) -> None:
    import importlib
    import sys

    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("DATABASE_URL", async_url)
    monkeypatch.setenv("JWT_SECRET", "test-jwt-secret-mig056")
    monkeypatch.setenv("ENCRYPTION_KEY", "test-32-byte-encryption-key--x12")
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_dummy_for_mig056")

    if "app.config" in sys.modules:
        importlib.reload(sys.modules["app.config"])


def _seed_pre_056_db(sync_url: str) -> None:
    """Pre-create build_jobs with the mig 052 columns present and
    stamp alembic_version=055 (056's down_revision). Insert one row
    so we can prove the upgrade preserves existing data (NULL on the
    new nullable cols, 0.0 on credit_spent via server_default)."""
    engine = sa.create_engine(sync_url)
    with engine.begin() as conn:
        conn.execute(sa.text(_BUILD_JOBS_PRE_056_CREATE_SQL))
        conn.execute(sa.text(
            "INSERT INTO build_jobs (id, user_id, title, prompt) "
            "VALUES ('preexisting-053', 'user-1', 'Pre-existing', 'p')"
        ))
        conn.execute(sa.text(
            "CREATE TABLE alembic_version (version_num VARCHAR(32) PRIMARY KEY)"
        ))
        conn.execute(sa.text(
            "INSERT INTO alembic_version (version_num) VALUES ('055')"
        ))
    engine.dispose()


@pytest.fixture
def fresh_db_url():
    fd, path = tempfile.mkstemp(suffix=".sqlite", prefix="mig056_")
    os.close(fd)
    yield f"sqlite+aiosqlite:///{path}", f"sqlite:///{path}"
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


# ──────────────────────────────────────────────────────────────────────
# Upgrade adds all four columns + the index; pre-existing row survives.
# ──────────────────────────────────────────────────────────────────────


def test_upgrade_adds_all_four_columns_and_preserves_data(
    fresh_db_url, monkeypatch,
):
    async_url, sync_url = fresh_db_url
    _set_required_env(monkeypatch, async_url)
    _seed_pre_056_db(sync_url)

    cfg = _make_alembic_config(sync_url)
    command.upgrade(cfg, "056")

    engine = sa.create_engine(sync_url)
    with engine.connect() as conn:
        insp = sa.inspect(conn)
        cols = {c["name"] for c in insp.get_columns("build_jobs")}
        for expected in _EXPECTED_NEW_COLUMNS:
            assert expected in cols, (
                f"mig 056 should add build_jobs.{expected}"
            )
        # Existing row survives. Nullable cols are NULL.
        # credit_spent backfills to 0.0 via server_default.
        row = conn.execute(sa.text(
            "SELECT parent_job_id, config_json, credit_budget_allocated, "
            "credit_spent FROM build_jobs WHERE id='preexisting-053'"
        )).fetchone()
        assert row is not None, (
            "mig 056 destroyed the pre-existing row — upgrade must "
            "be data-preserving"
        )
        assert row.parent_job_id is None
        assert row.config_json is None
        assert row.credit_budget_allocated is None
        # server_default backfills existing rows. SQLite returns Python
        # float; some dialects/builds return numeric. Compare loosely.
        assert row.credit_spent is not None
        assert float(row.credit_spent) == 0.0

        version = conn.execute(
            sa.text("SELECT version_num FROM alembic_version")
        ).scalar_one()
        assert version == "056"
    engine.dispose()


def test_upgrade_creates_composite_index_on_parent_and_status(
    fresh_db_url, monkeypatch,
):
    """The cap-check query (count children with status='running' for
    parent_job_id=?) runs on every spawn. Pin the index name + column
    set so a later "clean up unused indexes" pass can't silently kill
    the hot path."""
    async_url, sync_url = fresh_db_url
    _set_required_env(monkeypatch, async_url)
    _seed_pre_056_db(sync_url)

    cfg = _make_alembic_config(sync_url)
    command.upgrade(cfg, "056")

    engine = sa.create_engine(sync_url)
    with engine.connect() as conn:
        insp = sa.inspect(conn)
        indexes = {ix["name"]: ix for ix in insp.get_indexes("build_jobs")}
        assert _EXPECTED_INDEX in indexes, (
            f"mig 056 must create index {_EXPECTED_INDEX} for child-cap queries"
        )
        # Column order matters for composite-index prefix scans; the
        # parent_job_id-only walk and the (parent, status) cap-check
        # both benefit from the parent_job_id-first ordering.
        idx_cols = tuple(indexes[_EXPECTED_INDEX]["column_names"])
        assert idx_cols == _EXPECTED_INDEX_COLS, (
            f"index {_EXPECTED_INDEX} columns must be {_EXPECTED_INDEX_COLS}, "
            f"got {idx_cols}"
        )
    engine.dispose()


# ──────────────────────────────────────────────────────────────────────
# Downgrade drops everything cleanly.
# ──────────────────────────────────────────────────────────────────────


def test_downgrade_drops_all_four_columns_and_the_index(
    fresh_db_url, monkeypatch,
):
    async_url, sync_url = fresh_db_url
    _set_required_env(monkeypatch, async_url)
    _seed_pre_056_db(sync_url)

    cfg = _make_alembic_config(sync_url)
    command.upgrade(cfg, "056")
    command.downgrade(cfg, "055")

    engine = sa.create_engine(sync_url)
    with engine.connect() as conn:
        insp = sa.inspect(conn)
        cols = {c["name"] for c in insp.get_columns("build_jobs")}
        for not_expected in _EXPECTED_NEW_COLUMNS:
            assert not_expected not in cols, (
                f"downgrade should have dropped build_jobs.{not_expected}"
            )
        indexes = {ix["name"] for ix in insp.get_indexes("build_jobs")}
        assert _EXPECTED_INDEX not in indexes, (
            f"downgrade should have dropped index {_EXPECTED_INDEX}"
        )
        # Pre-existing row still alive.
        n = conn.execute(sa.text(
            "SELECT COUNT(*) FROM build_jobs WHERE id='preexisting-053'"
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
    _seed_pre_056_db(sync_url)

    cfg = _make_alembic_config(sync_url)
    command.upgrade(cfg, "056")
    command.downgrade(cfg, "055")
    command.upgrade(cfg, "056")  # must NOT raise

    engine = sa.create_engine(sync_url)
    with engine.connect() as conn:
        insp = sa.inspect(conn)
        cols = {c["name"] for c in insp.get_columns("build_jobs")}
        for expected in _EXPECTED_NEW_COLUMNS:
            assert expected in cols
        indexes = {ix["name"] for ix in insp.get_indexes("build_jobs")}
        assert _EXPECTED_INDEX in indexes
    engine.dispose()


# ──────────────────────────────────────────────────────────────────────
# Parent-child INSERT after upgrade — sanity that the new column
# accepts a sub-agent's parent_job_id pointer and the index covers
# the cap-check query.
# ──────────────────────────────────────────────────────────────────────


def test_parent_child_insert_works_after_upgrade(fresh_db_url, monkeypatch):
    async_url, sync_url = fresh_db_url
    _set_required_env(monkeypatch, async_url)
    _seed_pre_056_db(sync_url)

    cfg = _make_alembic_config(sync_url)
    command.upgrade(cfg, "056")

    engine = sa.create_engine(sync_url)
    with engine.begin() as conn:
        # Parent
        conn.execute(sa.text(
            "INSERT INTO build_jobs (id, user_id, title, prompt, job_type, status) "
            "VALUES ('parent-1', 'user-1', 'Parent', 'p', 'agent_task', 'running')"
        ))
        # Child with parent_job_id linkage and config_json
        conn.execute(sa.text(
            "INSERT INTO build_jobs "
            "(id, user_id, title, prompt, job_type, status, "
            " parent_job_id, config_json, credit_budget_allocated) "
            "VALUES ('child-1', 'user-1', 'Child', 'p', 'subagent', 'running', "
            " 'parent-1', '{\"task\":\"do thing\",\"label\":\"thing\"}', 0.50)"
        ))

    with engine.connect() as conn:
        # The cap-check query the dispatcher will run on every spawn.
        running_children = conn.execute(sa.text(
            "SELECT COUNT(*) FROM build_jobs "
            "WHERE parent_job_id = :pid AND status = 'running'"
        ), {"pid": "parent-1"}).scalar_one()
        assert running_children == 1

        # credit_spent backfilled to 0.0 for both rows.
        spends = list(conn.execute(sa.text(
            "SELECT id, credit_spent FROM build_jobs WHERE id IN ('parent-1','child-1') ORDER BY id"
        )))
        assert {row.id: float(row.credit_spent) for row in spends} == {
            "child-1": 0.0,
            "parent-1": 0.0,
        }
    engine.dispose()
