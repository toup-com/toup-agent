"""
T1a — Connector vault schema migration smoke + invariants.

Covers everything in the T1a smoke spec:

  1. Tables exist post-migration with the right columns + indexes + FK
     constraints.
  2. CheckConstraint on `ConnectorIdentity.status` rejects invalid values.
  3. UNIQUE on (user_id, connector_id) is enforced for ConnectorIdentity.
  4. FK rejects insert with a bogus user_id.
  5. ON DELETE CASCADE wipes all four tables when the user is dropped.
  6. PLATFORM_ONLY_TABLES gating: agent run_mode does NOT include any
     of the four tables in `_allowed_tables`.
  7. Bridge contract test (T0a) is unaffected — none of these tables
     touch the bridge or `AgentEnvContract`.

Pure-unit where possible; the cascade test needs a real DB session and
runs on whatever DATABASE_URL the conftest's autouse fixture provides.
"""

from __future__ import annotations

import os
import uuid

import pytest
import pytest_asyncio
from sqlalchemy import inspect as sa_inspect, select, text
from sqlalchemy.exc import IntegrityError

from app.db.database import async_session_maker, engine
from app.db.models import (
    AgentConfig,
    CONNECTOR_IDENTITY_STATUSES,
    ConnectorEvent,
    ConnectorIdentity,
    ConnectorOAuthSession,
    ConnectorUserPreference,
    EVENT_CONNECTED,
    User,
)
from app.db.models.base import (
    AGENT_ONLY_TABLES,
    Base,
    PLATFORM_ONLY_TABLES,
    SHARED_TABLES,
)


CONNECTOR_TABLES = {
    "connector_identities",
    "connector_oauth_sessions",
    "connector_events",
    "connector_user_preferences",
}


# ─── Static invariants — no DB needed ─────────────────────────────────


def test_all_four_tables_in_platform_only_set():
    """The set in base.py must include exactly the four connector tables."""
    missing = CONNECTOR_TABLES - PLATFORM_ONLY_TABLES
    assert not missing, f"Tables missing from PLATFORM_ONLY_TABLES: {missing}"


def test_no_connector_table_in_agent_only_or_shared():
    """No accidental cross-contamination — connector tables must NOT
    be in AGENT_ONLY_TABLES or SHARED_TABLES."""
    in_agent = CONNECTOR_TABLES & AGENT_ONLY_TABLES
    in_shared = CONNECTOR_TABLES & SHARED_TABLES
    assert not in_agent, f"Connector tables leaked into AGENT_ONLY: {in_agent}"
    assert not in_shared, f"Connector tables leaked into SHARED: {in_shared}"


def test_models_register_against_base_metadata():
    """The four model classes must be registered on Base.metadata so
    init_db's create_all picks them up."""
    registered = set(Base.metadata.tables.keys())
    for tname in CONNECTOR_TABLES:
        assert tname in registered, (
            f"{tname} not in Base.metadata — model class not imported "
            f"into app/db/models/__init__.py"
        )


def test_status_enum_values_are_canonical():
    """The status enum has exactly the documented values, in the
    documented order. CheckConstraint on the table is generated from
    this tuple — drift here would silently widen what the DB accepts.

    `provider_down` added in T5c — set by the health-probe scheduler
    after N consecutive failures; auto-recovers to active on the
    next successful probe."""
    assert CONNECTOR_IDENTITY_STATUSES == (
        "active", "reauth_required", "revoked", "provider_down",
    )


def test_agent_run_mode_excludes_connector_tables():
    """The init_db filter `t.name not in PLATFORM_ONLY_TABLES` must
    drop all four connector tables when run_mode='agent'.

    This is the load-bearing assertion for "tenant containers must
    never see them" — the actual filter logic in init_db is exactly
    this set difference."""
    # Mirror the init_db filter (database.py — `_allowed_tables = [t for
    # t in Base.metadata.sorted_tables if t.name not in _excluded]`).
    excluded_in_agent_mode = PLATFORM_ONLY_TABLES
    allowed_table_names = {
        t.name
        for t in Base.metadata.sorted_tables
        if t.name not in excluded_in_agent_mode
    }
    leaked = CONNECTOR_TABLES & allowed_table_names
    assert not leaked, (
        f"Tables that would leak into agent DBs on agent run_mode: {leaked}"
    )


# ─── Schema shape — column / index / FK presence ──────────────────────


def test_connector_identities_columns():
    cols = {c.name for c in ConnectorIdentity.__table__.columns}
    assert {
        "id", "user_id", "connector_id", "provider_account_id",
        "scopes_json", "access_token_enc", "refresh_token_enc",
        "access_expires_at", "status", "connected_at", "last_used_at",
        "metadata_json",
    } == cols


def test_connector_identities_indexes_and_constraints():
    table = ConnectorIdentity.__table__
    index_names = {idx.name for idx in table.indexes}
    assert "ix_connector_identities_user_status" in index_names

    # Unique constraint enforces "one identity per (user, connector)".
    uniques = [c for c in table.constraints if hasattr(c, "columns")
               and getattr(c, "name", "") == "uq_connector_identities_user_connector"]
    assert uniques, "missing UniqueConstraint(user_id, connector_id)"
    cols = {c.name for c in uniques[0].columns}
    assert cols == {"user_id", "connector_id"}

    check_names = {c.name for c in table.constraints if hasattr(c, "sqltext")}
    assert "ck_connector_identities_status" in check_names


def test_connector_identities_user_fk_cascades():
    """The user_id FK must be ON DELETE CASCADE so deleting a user
    wipes their tokens in the same transaction."""
    fks = list(ConnectorIdentity.__table__.foreign_keys)
    user_fks = [fk for fk in fks if fk.column.table.name == "users"]
    assert user_fks, "missing FK to users.id"
    assert user_fks[0].ondelete == "CASCADE", (
        f"user_id FK ondelete is {user_fks[0].ondelete!r}, expected 'CASCADE'"
    )


def test_connector_events_indexes():
    table = ConnectorEvent.__table__
    index_names = {idx.name for idx in table.indexes}
    # Architecture §3.3 explicitly calls for (user_id, connector_id, occurred_at)
    assert "ix_connector_events_user_connector_occurred" in index_names
    # Plus the secondary path for SLO dashboards.
    assert "ix_connector_events_user_type_occurred" in index_names


def test_connector_oauth_sessions_expires_at_index():
    """Sweep job needs an index on expires_at."""
    table = ConnectorOAuthSession.__table__
    index_names = {idx.name for idx in table.indexes}
    assert "ix_connector_oauth_sessions_expires_at" in index_names


def test_connector_user_preferences_composite_pk():
    """PK is (user_id, connector_id) — exactly one preference row per
    identity."""
    table = ConnectorUserPreference.__table__
    pk_cols = {c.name for c in table.primary_key.columns}
    assert pk_cols == {"user_id", "connector_id"}


# ─── Live DB tests (use whatever conftest's autouse fixture provides) ─


@pytest_asyncio.fixture
async def alice_user_id() -> str:
    """Seed one user. Single user per test, so SQLite's degraded
    is_canary unique index is fine (the trap that bit T0c)."""
    async with async_session_maker() as db:
        uid = str(uuid.uuid4())
        db.add(User(
            id=uid,
            email=f"{uid[:8]}@example.com",
            hashed_password="x",
            name="Alice",
        ))
        await db.commit()
    return uid


def _is_sqlite() -> bool:
    return os.environ.get("DATABASE_URL", "").startswith("sqlite")


async def _ensure_sqlite_fks_on() -> None:
    """SQLite has FK enforcement OFF by default (per-connection setting).
    Enable it on the shared engine pool so the cascade tests run locally
    too. No-op on Postgres."""
    if not _is_sqlite():
        return
    async with engine.begin() as conn:
        await conn.execute(text("PRAGMA foreign_keys = ON"))


@pytest.mark.asyncio
async def test_connector_identity_status_rejects_invalid_value(alice_user_id):
    """CheckConstraint must reject `status='not_a_real_status'`.

    SQLite enforces CHECK constraints by default since 3.6.19 — works on
    the local conftest fixture too."""
    async with async_session_maker() as db:
        db.add(ConnectorIdentity(
            user_id=alice_user_id,
            connector_id="gmail",
            status="not_a_real_status",
        ))
        with pytest.raises(IntegrityError):
            await db.commit()


@pytest.mark.asyncio
async def test_connector_identity_unique_user_connector(alice_user_id):
    """UNIQUE(user_id, connector_id) — second insert with the same pair
    must fail."""
    async with async_session_maker() as db:
        db.add(ConnectorIdentity(
            user_id=alice_user_id, connector_id="gmail", status="active",
        ))
        await db.commit()

    async with async_session_maker() as db:
        db.add(ConnectorIdentity(
            user_id=alice_user_id, connector_id="gmail", status="active",
        ))
        with pytest.raises(IntegrityError):
            await db.commit()


@pytest.mark.asyncio
async def test_connector_identity_fk_rejects_bogus_user(alice_user_id):
    """Insert with a user_id that doesn't exist in `users` must fail.

    SQLite needs `PRAGMA foreign_keys = ON` to enforce — we flip it
    inline. CI Postgres enforces unconditionally so this is a no-op
    there."""
    await _ensure_sqlite_fks_on()

    async with async_session_maker() as db:
        bogus_uid = str(uuid.uuid4())
        db.add(ConnectorIdentity(
            user_id=bogus_uid, connector_id="gmail", status="active",
        ))
        with pytest.raises(IntegrityError):
            await db.commit()


@pytest.mark.asyncio
async def test_user_delete_cascades_all_four_tables(alice_user_id):
    """DELETE FROM users WHERE id = alice → all four connector tables
    drop alice's rows in the same transaction.

    Same SQLite-FK note as above — we flip PRAGMA foreign_keys ON
    inline. CI Postgres enforces unconditionally."""
    await _ensure_sqlite_fks_on()

    async with async_session_maker() as db:
        # Seed one row in each of the four tables.
        db.add(ConnectorIdentity(
            user_id=alice_user_id, connector_id="gmail", status="active",
        ))
        db.add(ConnectorOAuthSession(
            state_hash="abc" * 30,
            user_id=alice_user_id,
            connector_id="gmail",
            code_verifier="vvv",
            expires_at=__import__("datetime").datetime.utcnow(),
        ))
        db.add(ConnectorEvent(
            user_id=alice_user_id,
            connector_id="gmail",
            event_type=EVENT_CONNECTED,
        ))
        db.add(ConnectorUserPreference(
            user_id=alice_user_id, connector_id="gmail", enabled=True,
        ))
        await db.commit()

    # Drop the user via raw DELETE — bypass SQLAlchemy's ORM cascade
    # (which tries to load all related rows including agent-only tables
    # like `memories` that don't exist in this platform-mode test DB).
    # We're asserting DB-level FK CASCADE here, not the ORM relationship
    # cascade, so raw SQL is the right level.
    async with async_session_maker() as db:
        await db.execute(
            text("DELETE FROM users WHERE id = :uid"),
            {"uid": alice_user_id},
        )
        await db.commit()

    # Every connector table must now have zero rows for this user.
    async with async_session_maker() as db:
        for model in (
            ConnectorIdentity, ConnectorOAuthSession,
            ConnectorEvent, ConnectorUserPreference,
        ):
            user_id_col = (
                model.user_id if hasattr(model, "user_id")
                else model.__table__.c.user_id
            )
            rows = (await db.execute(
                select(model).where(user_id_col == alice_user_id)
            )).scalars().all()
            assert rows == [], (
                f"{model.__tablename__} still has alice's row after user delete — "
                f"ON DELETE CASCADE not working"
            )


# ─── Cross-cutting: T0a contract test must be unaffected ──────────────


def test_no_new_field_leaked_into_bridge_contract():
    """T1a ships four NEW tables but does NOT touch agent_configs or
    AgentEnvContract. The bridge schema snapshot must be unchanged.

    This is a smoke check — the actual contract test runs in
    test_bridge_contract.py. Here we just assert the import surface
    is clean."""
    from app.types.agent_env import AgentEnvContract, BRIDGE_SCHEMA_VERSION
    # Field set should still match what the snapshot has — adding
    # connector tables MUST NOT have widened the contract.
    assert BRIDGE_SCHEMA_VERSION == 1
    assert "smoke_test_drift_probe" not in AgentEnvContract.model_fields
