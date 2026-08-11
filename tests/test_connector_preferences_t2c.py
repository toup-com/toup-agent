"""
T2c — Connector preferences + dispatcher pref-allow hook tests.

Two layers:

  P1 (dispatcher unit — `_resolve_user_preference`):
    - no row → returns None
    - enabled=false → ConnectorToolError (kill switch)
    - explicit channel=False → ConnectorToolError
    - explicit channel=True → _PREF_EXPLICIT_ALLOW sentinel
    - no override for this tool/channel → returns None

  P2 (dispatcher integration — pref overrides manifest):
    - explicit allow on voice for a mutating tool → bypasses
      manifest's mutating-default-deny
    - explicit deny on web for a non-mutating tool → blocks
    - kill switch on connector → blocks regardless
    - no pref → manifest defaults apply
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta
from typing import ClassVar

import pytest
import pytest_asyncio
from cryptography.fernet import Fernet

from app.config import settings
from app.connectors.base import (
    BaseConnectorProvider,
    ConnectorOk,
    ConnectorReauthRequired,
    ConnectorToolError,
    HealthResult,
    RefreshResult,
)
from app.db.database import async_session_maker
from app.db.models import ConnectorUserPreference, User
from app.services import connector_dispatcher as dispatcher
from app.services import connector_quarantine as q
from app.services import connector_vault as vault
from app.services.connector_dispatcher import (
    _PREF_EXPLICIT_ALLOW,
    _resolve_user_preference,
)
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


class _StubProvider(BaseConnectorProvider):
    manifest_id: ClassVar[str] = "pstub"

    def __init__(self):
        self.execute_calls = 0

    async def execute(self, tool_name, tool_input, ctx):
        self.execute_calls += 1
        return ConnectorOk(content="ok")

    async def revoke(self, user_id, access_token, refresh_token=None):
        return None

    async def refresh(self, refresh_token, *, scopes=None):
        return RefreshResult(
            access_token="a", refresh_token=refresh_token,
            expires_at=datetime.utcnow() + timedelta(hours=1),
        )

    async def health_probe(self, ctx):
        return HealthResult(ok=True)


def _install(*, mutates: bool = False, channel_deny: list[str] | None = None) -> _StubProvider:
    reset_registry_for_tests()
    manifest = ConnectorManifest(
        manifest_version=1,
        id="pstub",
        name="P Stub",
        short_description="t2c harness",
        status="experimental",
        category="test",
        oauth=OAuthSpec(
            provider_app="stub_provider_app", scopes=[], pkce=True, refresh=True,
        ),
        health=HealthSpec(probe="pstub__do"),
        tools=[
            ConnectorTool(
                name="pstub__do",
                description="x",
                input_schema={"type": "object", "properties": {}},
                mutates=mutates,
                elevation=False,
                output_redaction=[],
                channel_policy=ChannelPolicy(
                    default="allow",
                    deny=channel_deny or [],
                ),
            )
        ],
    )
    provider = _StubProvider()
    reg = get_registry()
    reg._entries["pstub"] = ConnectorEntry(manifest=manifest, provider=provider)
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


async def _seed(user_id: str) -> None:
    async with async_session_maker() as db:
        await vault.put(
            db, user_id, "pstub",
            access_token="seed",
            refresh_token="rt",
            access_expires_at=datetime.utcnow() + timedelta(hours=1),
        )


# ─── P1: _resolve_user_preference returns ───────────────────────────────


@pytest.mark.asyncio
async def test_p1_no_row_returns_none(alice_id):
    async with async_session_maker() as db:
        out = await _resolve_user_preference(
            db, alice_id, "pstub", "pstub__do", "web",
        )
    assert out is None


@pytest.mark.asyncio
async def test_p1_kill_switch_returns_tool_error(alice_id):
    async with async_session_maker() as db:
        db.add(ConnectorUserPreference(
            user_id=alice_id, connector_id="pstub", enabled=False,
        ))
        await db.commit()
        out = await _resolve_user_preference(
            db, alice_id, "pstub", "pstub__do", "web",
        )
    assert isinstance(out, ConnectorToolError)
    assert "disabled" in out.message.lower()


@pytest.mark.asyncio
async def test_p1_explicit_deny_returns_tool_error(alice_id):
    async with async_session_maker() as db:
        db.add(ConnectorUserPreference(
            user_id=alice_id, connector_id="pstub", enabled=True,
            per_tool_channel_overrides_json=json.dumps({
                "pstub__do": {"web": False},
            }),
        ))
        await db.commit()
        out = await _resolve_user_preference(
            db, alice_id, "pstub", "pstub__do", "web",
        )
    assert isinstance(out, ConnectorToolError)
    assert "disabled" in out.message.lower()


@pytest.mark.asyncio
async def test_p1_explicit_allow_returns_sentinel(alice_id):
    async with async_session_maker() as db:
        db.add(ConnectorUserPreference(
            user_id=alice_id, connector_id="pstub", enabled=True,
            per_tool_channel_overrides_json=json.dumps({
                "pstub__do": {"voice": True},
            }),
        ))
        await db.commit()
        out = await _resolve_user_preference(
            db, alice_id, "pstub", "pstub__do", "voice",
        )
    # Sentinel comparison via `is` is the contract.
    assert out is _PREF_EXPLICIT_ALLOW


@pytest.mark.asyncio
async def test_p1_other_channel_no_override_returns_none(alice_id):
    """Override for voice exists, but call is on web → no opinion."""
    async with async_session_maker() as db:
        db.add(ConnectorUserPreference(
            user_id=alice_id, connector_id="pstub", enabled=True,
            per_tool_channel_overrides_json=json.dumps({
                "pstub__do": {"voice": True},
            }),
        ))
        await db.commit()
        out = await _resolve_user_preference(
            db, alice_id, "pstub", "pstub__do", "web",
        )
    assert out is None


# ─── P2: dispatcher end-to-end with pref overrides ──────────────────────


@pytest.mark.asyncio
async def test_p2_explicit_allow_bypasses_mutating_default_deny(alice_id):
    """Manifest declares mutates=true. By default, voice/telegram are
    denied. Explicit user grant for voice → call proceeds."""
    provider = _install(mutates=True)
    await _seed(alice_id)

    async with async_session_maker() as db:
        db.add(ConnectorUserPreference(
            user_id=alice_id, connector_id="pstub", enabled=True,
            per_tool_channel_overrides_json=json.dumps({
                "pstub__do": {"voice": True},
            }),
        ))
        await db.commit()

    async with async_session_maker() as db:
        result = await dispatcher.execute(
            db, alice_id, "pstub", "pstub__do",
            tool_input={}, channel="voice",
        )
    assert isinstance(result, ConnectorOk)
    assert provider.execute_calls == 1


@pytest.mark.asyncio
async def test_p2_explicit_deny_blocks_otherwise_allowed_call(alice_id):
    """Manifest allows on web; user explicitly denies; dispatch blocks."""
    provider = _install(mutates=False)
    await _seed(alice_id)

    async with async_session_maker() as db:
        db.add(ConnectorUserPreference(
            user_id=alice_id, connector_id="pstub", enabled=True,
            per_tool_channel_overrides_json=json.dumps({
                "pstub__do": {"web": False},
            }),
        ))
        await db.commit()

    async with async_session_maker() as db:
        result = await dispatcher.execute(
            db, alice_id, "pstub", "pstub__do",
            tool_input={}, channel="web",
        )
    assert isinstance(result, ConnectorToolError)
    assert provider.execute_calls == 0


@pytest.mark.asyncio
async def test_p2_kill_switch_blocks_regardless(alice_id):
    provider = _install(mutates=False)
    await _seed(alice_id)

    async with async_session_maker() as db:
        db.add(ConnectorUserPreference(
            user_id=alice_id, connector_id="pstub", enabled=False,
        ))
        await db.commit()

    async with async_session_maker() as db:
        result = await dispatcher.execute(
            db, alice_id, "pstub", "pstub__do",
            tool_input={}, channel="web",
        )
    assert isinstance(result, ConnectorToolError)
    assert provider.execute_calls == 0


@pytest.mark.asyncio
async def test_p2_no_pref_falls_through_to_manifest(alice_id):
    """Mutating tool on telegram, no user pref → manifest default-deny
    fires (existing T1e behaviour preserved)."""
    provider = _install(mutates=True)
    await _seed(alice_id)
    # No ConnectorUserPreference row.

    async with async_session_maker() as db:
        result = await dispatcher.execute(
            db, alice_id, "pstub", "pstub__do",
            tool_input={}, channel="telegram",
        )
    assert isinstance(result, ConnectorToolError)
    assert "mutating" in result.message.lower()
    assert provider.execute_calls == 0


@pytest.mark.asyncio
async def test_p2_explicit_allow_on_web_doesnt_break_normal_flow(alice_id):
    """Allow=True on web for a non-mutating tool → still proceeds.
    Sanity check that the sentinel branch doesn't accidentally short-
    circuit success cases."""
    provider = _install(mutates=False)
    await _seed(alice_id)

    async with async_session_maker() as db:
        db.add(ConnectorUserPreference(
            user_id=alice_id, connector_id="pstub", enabled=True,
            per_tool_channel_overrides_json=json.dumps({
                "pstub__do": {"web": True},
            }),
        ))
        await db.commit()

    async with async_session_maker() as db:
        result = await dispatcher.execute(
            db, alice_id, "pstub", "pstub__do",
            tool_input={}, channel="web",
        )
    assert isinstance(result, ConnectorOk)
