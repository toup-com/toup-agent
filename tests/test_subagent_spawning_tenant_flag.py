"""Per-tenant sub-agent spawning kill-switch (mig 057).

Pins three surfaces:

  1. The ``agent_configs.subagent_spawning_enabled`` column exists on
     the ORM with the right type, nullability, and default — operators
     who SELECT or UPDATE this column from psql rely on the schema.
  2. ``_agent_config_to_bridge_body`` forwards the field to the bridge
     dict ONLY when True (omitted when False so existing tenants don't
     get a redundant ``SUBAGENT_SPAWNING_ENABLED=false`` line in their
     .env). The bridge silently drops unknown fields until the VPS
     contract is updated, but this forward-compat wiring keeps the
     contract single-source-of-truth.
  3. ``_read_subagent_flag_for_user`` (the runtime path that actually
     unlocks per-tenant spawning today) returns the column value and
     fails closed on any error.
"""
from __future__ import annotations

import uuid

import pytest
import pytest_asyncio
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine


# ──────────────────────────────────────────────────────────────────────
# 1. ORM column declaration
# ──────────────────────────────────────────────────────────────────────


def test_orm_declares_subagent_spawning_enabled_column():
    """The column must be on AgentConfig with the right shape so
    operators can SELECT/UPDATE it from psql without surprise."""
    from app.db.models.agent import AgentConfig
    col = AgentConfig.__table__.columns.get("subagent_spawning_enabled")
    assert col is not None, (
        "AgentConfig missing subagent_spawning_enabled column — "
        "migration 057 expects the ORM to declare it"
    )
    # NOT NULL with default false — matches the migration's server_default.
    assert col.nullable is False, (
        "subagent_spawning_enabled must be NOT NULL so a NULL row can't "
        "silently disable the kill-switch for tenants who skipped the "
        "backfill"
    )
    # Pin the boolean type — a stray Integer would land as a different
    # column shape on Postgres and surprise the runtime read.
    assert isinstance(col.type, sa.Boolean)


# ──────────────────────────────────────────────────────────────────────
# 2. Bridge whitelist forwarding behaviour
# ──────────────────────────────────────────────────────────────────────


def test_bridge_body_omits_flag_when_false():
    """When the column is False (the default), the bridge body must NOT
    carry the field. We don't want every tenant's .env to grow a
    redundant ``SUBAGENT_SPAWNING_ENABLED=false`` line just because
    the column was added."""
    from app.db.models.agent import AgentConfig
    from app.services.docker_host_service import _agent_config_to_bridge_body

    cfg = AgentConfig(
        user_id="11111111-1111-1111-1111-111111111111",
        subagent_spawning_enabled=False,
    )
    body = _agent_config_to_bridge_body(cfg)
    assert "subagent_spawning_enabled" not in body, (
        "subagent_spawning_enabled=False must be omitted from bridge body"
    )


def test_bridge_body_includes_flag_when_true():
    """When ops flips the column to True (the rollout step), the bridge
    body must carry ``subagent_spawning_enabled: True`` so the .env
    write path sees it."""
    from app.db.models.agent import AgentConfig
    from app.services.docker_host_service import _agent_config_to_bridge_body

    cfg = AgentConfig(
        user_id="22222222-2222-2222-2222-222222222222",
        subagent_spawning_enabled=True,
    )
    body = _agent_config_to_bridge_body(cfg)
    assert body.get("subagent_spawning_enabled") is True, (
        f"subagent_spawning_enabled=True must be forwarded; got {body!r}"
    )


def test_bridge_body_handles_missing_attribute_gracefully():
    """An older AgentConfig instance (or a mock that doesn't have the
    field) MUST NOT raise — getattr defaults to False and the field
    is omitted. Protects against rolling updates where the platform
    code is one revision behind the DB."""
    from app.services.docker_host_service import _agent_config_to_bridge_body

    class _Stub:
        # Only the absolute minimum the function needs; deliberately
        # missing subagent_spawning_enabled.
        user_id = "33333333-3333-3333-3333-333333333333"

    body = _agent_config_to_bridge_body(_Stub())
    assert "subagent_spawning_enabled" not in body


# ──────────────────────────────────────────────────────────────────────
# 3. AgentEnvContract field declaration
# ──────────────────────────────────────────────────────────────────────


def test_agent_env_contract_declares_subagent_spawning_enabled():
    """Forward-compat: the field is in the contract so when the bridge
    VPS code is updated to know it, the .env path lights up. Until
    then the contract field default (False) makes the absent .env
    line a no-op."""
    from app.types.agent_env import AgentEnvContract
    fields = AgentEnvContract.model_fields
    assert "subagent_spawning_enabled" in fields
    assert fields["subagent_spawning_enabled"].default is False


def test_agent_env_contract_emits_uppercase_env_key_when_true():
    """``to_env_lines`` must emit ``SUBAGENT_SPAWNING_ENABLED=true``
    (lowercase boolean rendering matches pydantic-settings parsing)
    when the field is True. Pins the serialization shape so a future
    refactor that drops the field silently is caught."""
    from app.types.agent_env import AgentEnvContract

    contract = AgentEnvContract(
        user_id="44444444-4444-4444-4444-444444444444",
        agent_api_key="x" * 40,
        database_url=(
            "postgresql+asyncpg://toup_agent_44444444:pw@"
            "host.docker.internal:6432/toup_agent_44444444"
        ),
        platform_api_url="https://toup.ai/api",
        subagent_spawning_enabled=True,
    )
    lines = contract.to_env_lines()
    assert "SUBAGENT_SPAWNING_ENABLED=true" in lines


# ──────────────────────────────────────────────────────────────────────
# 4. Runtime DB-read path (the actual unlock)
# ──────────────────────────────────────────────────────────────────────


@pytest_asyncio.fixture(autouse=True)
async def _reset_database():
    """Override conftest autouse — this test owns its own engine."""
    yield


@pytest_asyncio.fixture
async def db_with_agent_configs(monkeypatch):
    """In-memory SQLite with a minimal agent_configs table — enough
    for _read_subagent_flag_for_user to query it."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///file:flagtest?mode=memory&cache=shared&uri=true",
        connect_args={"check_same_thread": False, "uri": True},
    )
    async with engine.begin() as conn:
        await conn.execute(sa.text(
            "CREATE TABLE agent_configs ("
            "  id VARCHAR(36) PRIMARY KEY,"
            "  user_id VARCHAR(36) UNIQUE NOT NULL,"
            "  subagent_spawning_enabled BOOLEAN NOT NULL DEFAULT 0"
            ")"
        ))
    session_maker = async_sessionmaker(engine, expire_on_commit=False)

    import app.db.database as db_database
    import app.db as app_db
    monkeypatch.setattr(db_database, "engine", engine)
    monkeypatch.setattr(db_database, "async_session_maker", session_maker)
    monkeypatch.setattr(app_db, "async_session_maker", session_maker)

    yield session_maker
    await engine.dispose()


@pytest.mark.asyncio
async def test_read_subagent_flag_returns_true_when_column_true(
    db_with_agent_configs,
):
    """The unlock path: SELECT subagent_spawning_enabled WHERE user_id=?
    returns True when ops has flipped the column."""
    from app.agent.tool_executor import _read_subagent_flag_for_user

    uid = str(uuid.uuid4())
    async with db_with_agent_configs() as s:
        await s.execute(sa.text(
            "INSERT INTO agent_configs (id, user_id, subagent_spawning_enabled) "
            "VALUES (:id, :uid, 1)"
        ), {"id": str(uuid.uuid4()), "uid": uid})
        await s.commit()

    assert await _read_subagent_flag_for_user(uid) is True


@pytest.mark.asyncio
async def test_read_subagent_flag_returns_false_when_column_false(
    db_with_agent_configs,
):
    """Default-False rows stay on the legacy path."""
    from app.agent.tool_executor import _read_subagent_flag_for_user

    uid = str(uuid.uuid4())
    async with db_with_agent_configs() as s:
        await s.execute(sa.text(
            "INSERT INTO agent_configs (id, user_id, subagent_spawning_enabled) "
            "VALUES (:id, :uid, 0)"
        ), {"id": str(uuid.uuid4()), "uid": uid})
        await s.commit()

    assert await _read_subagent_flag_for_user(uid) is False


@pytest.mark.asyncio
async def test_read_subagent_flag_returns_false_for_unknown_user(
    db_with_agent_configs,
):
    """No row for the user → fail closed."""
    from app.agent.tool_executor import _read_subagent_flag_for_user
    assert await _read_subagent_flag_for_user(str(uuid.uuid4())) is False


@pytest.mark.asyncio
async def test_read_subagent_flag_returns_false_for_empty_user_id():
    """Empty/None user_id should short-circuit without a DB hit. Belt
    and suspenders — _tool_spawn rejects empty user_id earlier, but
    the helper should still be safe to call from anywhere."""
    from app.agent.tool_executor import _read_subagent_flag_for_user
    assert await _read_subagent_flag_for_user("") is False
    assert await _read_subagent_flag_for_user(None) is False


@pytest.mark.asyncio
async def test_read_subagent_flag_swallows_db_errors():
    """The helper MUST NOT raise on DB errors — a stale agent that
    predates the migration would see "no such column" and that must
    not blow up every spawn turn. Fail closed (return False)."""
    from app.agent.tool_executor import _read_subagent_flag_for_user

    # Force a session that has no agent_configs table at all.
    engine = create_async_engine(
        "sqlite+aiosqlite:///file:flagtest-err?mode=memory&cache=shared&uri=true",
        connect_args={"check_same_thread": False, "uri": True},
    )
    session_maker = async_sessionmaker(engine, expire_on_commit=False)

    import app.db.database as db_database
    import app.db as app_db
    original_db = db_database.async_session_maker
    original_app = app_db.async_session_maker
    db_database.async_session_maker = session_maker
    app_db.async_session_maker = session_maker
    try:
        result = await _read_subagent_flag_for_user(str(uuid.uuid4()))
        assert result is False
    finally:
        db_database.async_session_maker = original_db
        app_db.async_session_maker = original_app
        await engine.dispose()
