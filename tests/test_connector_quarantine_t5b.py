"""
T5b — Force-quarantine tests.

Three layers:

  Q1 (in-process state):
    - add/remove/is_quarantined round-trip
    - list_active ordering
    - reset_for_tests clears state

  Q2 (dispatcher integration):
    - quarantined connector → ConnectorToolError with operator reason
    - quarantine takes priority over user pref + manifest
    - non-quarantined connector unaffected
    - quarantine + release → next call proceeds normally

  Q3 (canonical event types):
    - EVENT_FORCE_QUARANTINED + EVENT_FORCE_RELEASED importable from
      models package (catches drift if someone moves them)
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta
from typing import ClassVar

import pytest
import pytest_asyncio
from cryptography.fernet import Fernet
from sqlalchemy import select

from app.config import settings
from app.connectors.base import (
    BaseConnectorProvider,
    ConnectorOk,
    ConnectorReauthRequired,
    ConnectorResult,
    ConnectorToolError,
    HealthResult,
    RefreshResult,
)
from app.db.database import async_session_maker
from app.db.models import (
    EVENT_FORCE_QUARANTINED,
    EVENT_FORCE_RELEASED,
    ConnectorIdentity,
    User,
)
from app.services import connector_dispatcher as dispatcher
from app.services import connector_quarantine as q
from app.services import connector_vault as vault
from app.services.connector_registry import (
    ConnectorEntry,
    ConnectorManifest,
    ConnectorTool,
    ChannelPolicy,
    HealthSpec,
    OAuthSpec,
    get_registry,
    reset_registry_for_tests,
)
from app.services.credential_crypto import _multi_fernet


# ─── Q3: canonical event imports ────────────────────────────────────────


def test_q3_event_constants_importable_from_models_package():
    """If someone moves these constants, this test catches the drift
    before the admin endpoint silently writes a stale event_type."""
    from app.db.models import EVENT_FORCE_QUARANTINED, EVENT_FORCE_RELEASED
    assert EVENT_FORCE_QUARANTINED == "force_quarantined"
    assert EVENT_FORCE_RELEASED == "force_released"


# ─── Q1: in-process state ───────────────────────────────────────────────


def test_q1_add_remove_round_trip():
    q.reset_for_tests()
    assert q.is_quarantined("gmail") is None
    q.add(q.QuarantineEntry(
        connector_id="gmail",
        reason="API outage",
        actor_user_id="alice",
        quarantined_at=datetime.utcnow(),
    ))
    entry = q.is_quarantined("gmail")
    assert entry is not None
    assert entry.reason == "API outage"
    assert entry.actor_user_id == "alice"

    q.remove("gmail")
    assert q.is_quarantined("gmail") is None


def test_q1_list_active_orders_by_quarantined_at():
    q.reset_for_tests()
    now = datetime.utcnow()
    q.add(q.QuarantineEntry("b", "second", "u", now + timedelta(seconds=1)))
    q.add(q.QuarantineEntry("a", "first", "u", now))
    entries = q.list_active()
    assert [e.connector_id for e in entries] == ["a", "b"]


def test_q1_remove_unknown_is_noop():
    q.reset_for_tests()
    q.remove("never_added")  # must not raise


# ─── Q2: dispatcher integration ─────────────────────────────────────────


class _StubProvider(BaseConnectorProvider):
    manifest_id: ClassVar[str] = "qstub"

    def __init__(self):
        self.execute_calls = 0

    async def execute(self, tool_name, tool_input, ctx):
        self.execute_calls += 1
        return ConnectorOk(content="ok")

    async def revoke(self, user_id, access_token, refresh_token=None):
        return None

    async def refresh(self, refresh_token):
        return RefreshResult(
            access_token="a", refresh_token=refresh_token,
            expires_at=datetime.utcnow() + timedelta(hours=1),
        )

    async def health_probe(self, ctx):
        return HealthResult(ok=True)


def _install_qstub() -> _StubProvider:
    reset_registry_for_tests()
    manifest = ConnectorManifest(
        manifest_version=1,
        id="qstub",
        name="Q Stub",
        short_description="t5b harness",
        status="experimental",
        category="test",
        oauth=OAuthSpec(
            provider_app="stub_provider_app", scopes=[], pkce=True, refresh=True,
        ),
        health=HealthSpec(probe="qstub__do"),
        tools=[
            ConnectorTool(
                name="qstub__do",
                description="x",
                input_schema={"type": "object", "properties": {}},
                mutates=False,
                elevation=False,
                output_redaction=[],
                channel_policy=ChannelPolicy(default="allow", deny=[]),
            )
        ],
    )
    provider = _StubProvider()
    reg = get_registry()
    reg._entries["qstub"] = ConnectorEntry(manifest=manifest, provider=provider)
    for tool in manifest.tools:
        reg._tool_index[tool.name] = manifest.id
    return provider


@pytest.fixture(autouse=True)
def _provision_crypto():
    prev = settings.platform_encryption_key
    prev_prev = settings.platform_encryption_key_previous
    settings.platform_encryption_key = Fernet.generate_key().decode()
    settings.platform_encryption_key_previous = ""
    _multi_fernet.cache_clear()
    try:
        yield
    finally:
        settings.platform_encryption_key = prev
        settings.platform_encryption_key_previous = prev_prev
        _multi_fernet.cache_clear()


@pytest.fixture(autouse=True)
def _reset_quarantine():
    q.reset_for_tests()
    yield
    q.reset_for_tests()


@pytest_asyncio.fixture
async def alice_id() -> str:
    async with async_session_maker() as db:
        uid = str(uuid.uuid4())
        db.add(User(
            id=uid, email=f"{uid[:8]}@example.com",
            hashed_password="x", name="Alice",
        ))
        await db.commit()
    return uid


async def _seed_active_identity(user_id: str) -> None:
    async with async_session_maker() as db:
        await vault.put(
            db, user_id, "qstub",
            access_token="seed",
            refresh_token="rt",
            access_expires_at=datetime.utcnow() + timedelta(hours=1),
        )


@pytest.mark.asyncio
async def test_q2_quarantined_returns_tool_error_with_operator_reason(alice_id):
    provider = _install_qstub()
    await _seed_active_identity(alice_id)

    q.add(q.QuarantineEntry(
        connector_id="qstub",
        reason="upstream API outage",
        actor_user_id="ops",
        quarantined_at=datetime.utcnow(),
    ))

    async with async_session_maker() as db:
        result = await dispatcher.execute(
            db, alice_id, "qstub", "qstub__do",
            tool_input={}, channel="web",
        )

    # T5b spec: surface operator reason, never a misleading "reconnect"
    # chip (since reconnecting won't help while quarantined).
    assert isinstance(result, ConnectorToolError)
    assert "Service paused by operator" in result.message
    assert "upstream API outage" in result.message
    assert result.retryable is True  # eligible to retry once lifted

    # Provider was NEVER called — quarantine pre-empts execute.
    assert provider.execute_calls == 0


@pytest.mark.asyncio
async def test_q2_quarantine_takes_priority_over_user_pref(alice_id):
    """User has explicitly enabled qstub → still blocked when quarantined."""
    provider = _install_qstub()
    await _seed_active_identity(alice_id)

    # Pretend the user has the connector explicitly allowed (per T2c).
    from app.db.models import ConnectorUserPreference
    async with async_session_maker() as db:
        db.add(ConnectorUserPreference(
            user_id=alice_id,
            connector_id="qstub",
            enabled=True,
            per_tool_channel_overrides_json=json.dumps({
                "qstub__do": {"web": True},
            }),
        ))
        await db.commit()

    q.add(q.QuarantineEntry(
        connector_id="qstub",
        reason="ops sweep",
        actor_user_id="ops",
        quarantined_at=datetime.utcnow(),
    ))

    async with async_session_maker() as db:
        result = await dispatcher.execute(
            db, alice_id, "qstub", "qstub__do",
            tool_input={}, channel="web",
        )

    assert isinstance(result, ConnectorToolError)
    assert "Service paused by operator" in result.message
    assert provider.execute_calls == 0


@pytest.mark.asyncio
async def test_q2_unrelated_connector_unaffected(alice_id):
    """Quarantining 'gmail' must not affect 'qstub' dispatch."""
    provider = _install_qstub()
    await _seed_active_identity(alice_id)

    q.add(q.QuarantineEntry(
        connector_id="gmail", reason="x", actor_user_id="ops",
        quarantined_at=datetime.utcnow(),
    ))

    async with async_session_maker() as db:
        result = await dispatcher.execute(
            db, alice_id, "qstub", "qstub__do",
            tool_input={}, channel="web",
        )
    assert isinstance(result, ConnectorOk)
    assert provider.execute_calls == 1


@pytest.mark.asyncio
async def test_q2_release_lets_next_call_proceed(alice_id):
    provider = _install_qstub()
    await _seed_active_identity(alice_id)

    # Quarantine → blocked.
    q.add(q.QuarantineEntry(
        "qstub", "x", "ops", datetime.utcnow(),
    ))
    async with async_session_maker() as db:
        r = await dispatcher.execute(
            db, alice_id, "qstub", "qstub__do",
            tool_input={}, channel="web",
        )
    assert isinstance(r, ConnectorToolError)

    # Release → next call succeeds.
    q.remove("qstub")
    async with async_session_maker() as db:
        r = await dispatcher.execute(
            db, alice_id, "qstub", "qstub__do",
            tool_input={}, channel="web",
        )
    assert isinstance(r, ConnectorOk)
