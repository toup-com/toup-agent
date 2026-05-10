"""
T1e — Connector dispatcher tests.

Smoke matrix (per the T1e spec):

  1. Happy path — stub returns ConnectorOk → audit rows tool_called +
     tool_succeeded → result is ConnectorOk.
  2. Channel deny — manifest declares channel_policy.deny=[voice], call
     with channel='voice' → ConnectorToolError, NO vault read, NO
     tool_called audit row.
  3. Mutates default-deny — manifest declares mutates=true, call with
     channel='telegram' → rejected, same no-vault-touch invariant.
  4. Reauth required — vault returns identity with status=reauth_required
     → ConnectorReauthRequired, NO provider call.
  5. Refresh on expiring token — seed identity with access_expires_at =
     now+5s, stub.refresh returns new tokens, dispatcher refreshes
     under lock, second concurrent call coalesces (one refresh).
  6. Refresh fail — stub.refresh raises RefreshFailed → identity
     flipped to reauth_required, ConnectorReauthRequired returned,
     refresh_failed audit row.
  7. Provider 5xx — stub.execute returns ConnectorProviderDown →
     audit row tool_failed with reason, identity status NOT flipped.
  8. Audit-then-act fail-closed — monkey-patch tool_called audit insert
     to raise → dispatcher does NOT call provider.execute.
  9. Output redaction — stub returns {body:'secret', subject:'hello'},
     manifest declares output_redaction=[body] → audit metadata lacks
     body, contains subject. Result returned to caller is unredacted.
  10. Concurrent refresh coalescing — 50 concurrent execute() calls on
      expiring-token identity. provider.refresh invocation count == 1.

Plus a few invariants worth nailing:
  - Unknown connector → ConnectorToolError.
  - Unknown tool name on a known connector → ConnectorToolError.
  - No-refresh-token + expiring → reauth_required (not provider call).

Each test seeds at most ONE user — the SQLite is_canary trap stays clear.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import ClassVar
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from cryptography.fernet import Fernet
from sqlalchemy import select

from app.config import settings
from app.connectors.base import (
    BaseConnectorProvider,
    ConnectorContext,
    ConnectorOk,
    ConnectorProviderDown,
    ConnectorRateLimited,
    ConnectorReauthRequired,
    ConnectorResult,
    ConnectorToolError,
    HealthResult,
    RefreshFailed,
    RefreshResult,
)
from app.db.database import async_session_maker
from app.db.models import (
    ConnectorEvent,
    ConnectorIdentity,
    EVENT_REFRESH_FAILED,
    EVENT_REFRESH_SUCCEEDED,
    EVENT_TOOL_CALLED,
    EVENT_TOOL_FAILED,
    EVENT_TOOL_SUCCEEDED,
    User,
)
from app.services import connector_dispatcher as dispatcher
from app.services import connector_vault as vault
from app.services.connector_registry import (
    ConnectorRegistry,
    get_registry,
    reset_registry_for_tests,
)
from app.services.credential_crypto import _multi_fernet


# ─── Test provider that lets each test inject behaviour ──────────────


class _ScriptedProvider(BaseConnectorProvider):
    """Provider whose execute / refresh / revoke / health_probe can be
    swapped per-test by setting `provider.<method>_returns` or
    `provider.<method>_raises`.

    The manifest_id is matched to whatever stub_test manifest is loaded
    in the registry fixture below."""

    manifest_id: ClassVar[str] = "scripted"

    def __init__(self) -> None:
        self.execute_call_count = 0
        self.refresh_call_count = 0
        self._execute_result: ConnectorResult = ConnectorOk(content="default_ok")
        self._execute_raises: Exception | None = None
        self._refresh_result = RefreshResult(
            access_token="refreshed_at",
            refresh_token="refreshed_rt",
            expires_at=datetime.utcnow() + timedelta(hours=1),
        )
        self._refresh_raises: Exception | None = None
        self._refresh_delay_s: float = 0.0

    async def execute(self, tool_name, tool_input, ctx):
        self.execute_call_count += 1
        if self._execute_raises:
            raise self._execute_raises
        return self._execute_result

    async def revoke(self, user_id, access_token, refresh_token=None):
        return None

    async def refresh(self, refresh_token):
        self.refresh_call_count += 1
        if self._refresh_delay_s:
            await asyncio.sleep(self._refresh_delay_s)
        if self._refresh_raises:
            raise self._refresh_raises
        return self._refresh_result

    async def health_probe(self, ctx):
        return HealthResult(ok=True)


# ─── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _provision_crypto():
    """Per-test Fernet key for encrypting tokens via the vault."""
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


@pytest.fixture
def make_manifest():
    """Factory for an in-memory ConnectorManifest with whatever tool
    settings the test needs (mutates, channel deny list, output_redaction).

    Returns a tuple (manifest, provider) ready to register. The
    `_register_in_registry` fixture wires it into the singleton."""
    from app.services.connector_registry import (
        ChannelPolicy, ConnectorManifest, ConnectorTool, HealthSpec, OAuthSpec,
    )

    def _make(
        *,
        tool_name: str = "scripted__do",
        mutates: bool = False,
        channel_deny: list[str] | None = None,
        channel_default: str = "allow",
        output_redaction: list[str] | None = None,
    ) -> tuple:
        manifest = ConnectorManifest(
            manifest_version=1,
            id="scripted",
            name="Scripted",
            short_description="test-only",
            status="experimental",
            category="test",
            oauth=OAuthSpec(
                provider_app="stub_provider_app",
                scopes=[],
                pkce=True,
                refresh=True,
            ),
            health=HealthSpec(probe=tool_name),
            tools=[
                ConnectorTool(
                    name=tool_name,
                    description="x",
                    input_schema={"type": "object", "properties": {}},
                    mutates=mutates,
                    elevation=False,
                    output_redaction=output_redaction or [],
                    channel_policy=ChannelPolicy(
                        default=channel_default,
                        deny=channel_deny or [],
                    ),
                )
            ],
        )
        return manifest, _ScriptedProvider()
    return _make


@pytest.fixture
def register_scripted(make_manifest):
    """Build a manifest+provider, swap them into the registry singleton
    for the test's duration."""
    from app.services.connector_registry import ConnectorEntry

    def _register(**kwargs):
        manifest, provider = make_manifest(**kwargs)
        reset_registry_for_tests()
        reg = get_registry()
        reg._entries["scripted"] = ConnectorEntry(manifest=manifest, provider=provider)
        for tool in manifest.tools:
            reg._tool_index[tool.name] = manifest.id
        return manifest, provider
    yield _register
    reset_registry_for_tests()
    dispatcher.reset_locks_for_tests()


@pytest_asyncio.fixture
async def alice_user_id() -> str:
    """One user per test. SQLite is_canary trap stays clear."""
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


async def _seed_active_identity(
    user_id: str,
    *,
    expires_at: datetime | None = None,
    refresh_token: str | None = "rt_seed",
) -> str:
    """Seed an active identity for (user, scripted) via vault.put.
    Returns the identity id."""
    async with async_session_maker() as db:
        await vault.put(
            db, user_id, "scripted",
            access_token="seed_at",
            refresh_token=refresh_token,
            access_expires_at=expires_at,
        )
        ident = (await db.execute(
            select(ConnectorIdentity).where(
                ConnectorIdentity.user_id == user_id
            )
        )).scalar_one()
        return ident.id


# ─── 1. Happy path ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_happy_path(register_scripted, alice_user_id):
    manifest, provider = register_scripted()
    provider._execute_result = ConnectorOk(content='{"hello":"world"}')
    await _seed_active_identity(alice_user_id)

    async with async_session_maker() as db:
        result = await dispatcher.execute(
            db, alice_user_id, "scripted", "scripted__do",
            tool_input={"q": "anything"},
            channel="web",
        )

    assert isinstance(result, ConnectorOk)
    assert result.content == '{"hello":"world"}'
    assert provider.execute_call_count == 1

    async with async_session_maker() as db:
        events = (await db.execute(
            select(ConnectorEvent.event_type)
            .where(ConnectorEvent.user_id == alice_user_id)
        )).scalars().all()
        # connected (from _seed) + tool_called + tool_succeeded
        assert EVENT_TOOL_CALLED in events
        assert EVENT_TOOL_SUCCEEDED in events
        assert EVENT_TOOL_FAILED not in events


# ─── 2. Channel deny (manifest deny list) ────────────────────────────


@pytest.mark.asyncio
async def test_channel_deny_pre_flight_no_vault_touch(
    register_scripted, alice_user_id, monkeypatch,
):
    """Manifest declares channel_policy.deny=[voice]. Call with
    channel='voice' → ConnectorToolError, vault.get NOT called, no
    tool_called audit row."""
    manifest, provider = register_scripted(channel_deny=["voice"])
    # Don't even seed an identity — the rejection must happen BEFORE
    # the vault is touched.
    vault_get_calls = []

    real_get = vault.get

    async def watching_get(*args, **kwargs):
        vault_get_calls.append(args)
        return await real_get(*args, **kwargs)

    monkeypatch.setattr(vault, "get", watching_get)

    async with async_session_maker() as db:
        result = await dispatcher.execute(
            db, alice_user_id, "scripted", "scripted__do",
            tool_input={}, channel="voice",
        )

    assert isinstance(result, ConnectorToolError)
    assert "voice" in result.message
    assert vault_get_calls == [], "channel rejection must happen pre-vault"
    assert provider.execute_call_count == 0

    async with async_session_maker() as db:
        events = (await db.execute(
            select(ConnectorEvent.event_type)
            .where(ConnectorEvent.user_id == alice_user_id)
        )).scalars().all()
        assert EVENT_TOOL_CALLED not in events


# ─── 3. Mutates default-deny on voice/telegram ───────────────────────


@pytest.mark.asyncio
async def test_mutates_default_deny_on_telegram(
    register_scripted, alice_user_id, monkeypatch,
):
    """mutates=true → telegram default-denied even when channel_policy
    doesn't list it explicitly."""
    manifest, provider = register_scripted(mutates=True)
    vault_get_calls = []
    real_get = vault.get

    async def watching_get(*args, **kwargs):
        vault_get_calls.append(args)
        return await real_get(*args, **kwargs)

    monkeypatch.setattr(vault, "get", watching_get)

    async with async_session_maker() as db:
        result = await dispatcher.execute(
            db, alice_user_id, "scripted", "scripted__do",
            tool_input={}, channel="telegram",
        )

    assert isinstance(result, ConnectorToolError)
    assert "mutating" in result.message.lower()
    assert vault_get_calls == []
    assert provider.execute_call_count == 0


@pytest.mark.asyncio
async def test_mutates_allowed_on_web(register_scripted, alice_user_id):
    """mutates=true on web is fine — only voice/telegram default-deny."""
    manifest, provider = register_scripted(mutates=True)
    await _seed_active_identity(alice_user_id)

    async with async_session_maker() as db:
        result = await dispatcher.execute(
            db, alice_user_id, "scripted", "scripted__do",
            tool_input={}, channel="web",
        )
    assert isinstance(result, ConnectorOk)


# ─── 4. Reauth-required identity ─────────────────────────────────────


@pytest.mark.asyncio
async def test_reauth_required_identity_short_circuits(
    register_scripted, alice_user_id,
):
    manifest, provider = register_scripted()
    identity_id = await _seed_active_identity(alice_user_id)
    async with async_session_maker() as db:
        await vault.mark_reauth_required(
            db, identity_id, reason="test setup",
        )

    async with async_session_maker() as db:
        result = await dispatcher.execute(
            db, alice_user_id, "scripted", "scripted__do",
            tool_input={}, channel="web",
        )

    assert isinstance(result, ConnectorReauthRequired)
    assert "/agent/integrations/scripted" in result.reauth_url
    assert provider.execute_call_count == 0


@pytest.mark.asyncio
async def test_no_identity_short_circuits(register_scripted, alice_user_id):
    """User never connected → ConnectorReauthRequired, no provider call."""
    manifest, provider = register_scripted()
    # No _seed_active_identity call.

    async with async_session_maker() as db:
        result = await dispatcher.execute(
            db, alice_user_id, "scripted", "scripted__do",
            tool_input={}, channel="web",
        )

    assert isinstance(result, ConnectorReauthRequired)
    assert provider.execute_call_count == 0


# ─── 5. Refresh on expiring token ────────────────────────────────────


@pytest.mark.asyncio
async def test_refresh_on_expiring_token(register_scripted, alice_user_id):
    manifest, provider = register_scripted()
    # Seed an identity that expires inside the 30s skew window.
    await _seed_active_identity(
        alice_user_id,
        expires_at=datetime.utcnow() + timedelta(seconds=5),
    )

    async with async_session_maker() as db:
        result = await dispatcher.execute(
            db, alice_user_id, "scripted", "scripted__do",
            tool_input={}, channel="web",
        )

    assert isinstance(result, ConnectorOk)
    assert provider.refresh_call_count == 1
    assert provider.execute_call_count == 1

    async with async_session_maker() as db:
        events = (await db.execute(
            select(ConnectorEvent.event_type)
            .where(ConnectorEvent.user_id == alice_user_id)
        )).scalars().all()
        assert EVENT_REFRESH_SUCCEEDED in events
        assert EVENT_TOOL_SUCCEEDED in events


@pytest.mark.asyncio
async def test_no_refresh_when_token_fresh(register_scripted, alice_user_id):
    """Token expires in 1 hour → no refresh call."""
    manifest, provider = register_scripted()
    await _seed_active_identity(
        alice_user_id,
        expires_at=datetime.utcnow() + timedelta(hours=1),
    )

    async with async_session_maker() as db:
        result = await dispatcher.execute(
            db, alice_user_id, "scripted", "scripted__do",
            tool_input={}, channel="web",
        )

    assert isinstance(result, ConnectorOk)
    assert provider.refresh_call_count == 0


@pytest.mark.asyncio
async def test_no_refresh_when_no_expiry(register_scripted, alice_user_id):
    """access_expires_at=None (e.g. GitHub PAT) → never refresh."""
    manifest, provider = register_scripted()
    await _seed_active_identity(alice_user_id, expires_at=None)

    async with async_session_maker() as db:
        result = await dispatcher.execute(
            db, alice_user_id, "scripted", "scripted__do",
            tool_input={}, channel="web",
        )

    assert isinstance(result, ConnectorOk)
    assert provider.refresh_call_count == 0


# ─── 6. Refresh fail ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_refresh_failed_marks_reauth_required(
    register_scripted, alice_user_id,
):
    manifest, provider = register_scripted()
    provider._refresh_raises = RefreshFailed("provider returned 401")
    await _seed_active_identity(
        alice_user_id,
        expires_at=datetime.utcnow() + timedelta(seconds=5),
    )

    async with async_session_maker() as db:
        result = await dispatcher.execute(
            db, alice_user_id, "scripted", "scripted__do",
            tool_input={}, channel="web",
        )

    assert isinstance(result, ConnectorReauthRequired)
    assert provider.execute_call_count == 0  # Never reached the call

    # Identity status flipped + audit row written.
    async with async_session_maker() as db:
        ident = (await db.execute(
            select(ConnectorIdentity).where(
                ConnectorIdentity.user_id == alice_user_id
            )
        )).scalar_one()
        assert ident.status == "reauth_required"
        events = (await db.execute(
            select(ConnectorEvent.event_type)
            .where(ConnectorEvent.user_id == alice_user_id)
        )).scalars().all()
        assert EVENT_REFRESH_FAILED in events


@pytest.mark.asyncio
async def test_no_refresh_token_when_expiring_marks_reauth(
    register_scripted, alice_user_id,
):
    """Token expiring AND no refresh_token on file → can't refresh →
    reauth required immediately, no provider.refresh call."""
    manifest, provider = register_scripted()
    await _seed_active_identity(
        alice_user_id,
        expires_at=datetime.utcnow() + timedelta(seconds=5),
        refresh_token=None,
    )

    async with async_session_maker() as db:
        result = await dispatcher.execute(
            db, alice_user_id, "scripted", "scripted__do",
            tool_input={}, channel="web",
        )

    assert isinstance(result, ConnectorReauthRequired)
    assert provider.refresh_call_count == 0


# ─── 7. Provider returns ConnectorProviderDown ───────────────────────


@pytest.mark.asyncio
async def test_provider_down_returns_provider_down_audits_failed(
    register_scripted, alice_user_id,
):
    manifest, provider = register_scripted()
    provider._execute_result = ConnectorProviderDown(provider_status_url="https://status.example.com")
    await _seed_active_identity(alice_user_id)

    async with async_session_maker() as db:
        result = await dispatcher.execute(
            db, alice_user_id, "scripted", "scripted__do",
            tool_input={}, channel="web",
        )

    assert isinstance(result, ConnectorProviderDown)
    assert provider.execute_call_count == 1

    # Audit row is tool_failed; identity status is unchanged (transient).
    async with async_session_maker() as db:
        events = (await db.execute(
            select(ConnectorEvent.event_type)
            .where(ConnectorEvent.user_id == alice_user_id)
        )).scalars().all()
        assert EVENT_TOOL_FAILED in events
        assert EVENT_TOOL_SUCCEEDED not in events

        ident = (await db.execute(
            select(ConnectorIdentity).where(
                ConnectorIdentity.user_id == alice_user_id
            )
        )).scalar_one()
        assert ident.status == "active", (
            "provider-down is transient — identity must NOT be flipped"
        )


@pytest.mark.asyncio
async def test_provider_unexpected_exception_maps_to_provider_down(
    register_scripted, alice_user_id,
):
    """Unhandled exceptions in provider.execute become
    ConnectorProviderDown (don't bubble 500s to the LLM)."""
    manifest, provider = register_scripted()
    provider._execute_raises = RuntimeError("kaboom")
    await _seed_active_identity(alice_user_id)

    async with async_session_maker() as db:
        result = await dispatcher.execute(
            db, alice_user_id, "scripted", "scripted__do",
            tool_input={}, channel="web",
        )

    assert isinstance(result, ConnectorProviderDown)


# ─── 8. Audit-then-act fail-closed ───────────────────────────────────


@pytest.mark.asyncio
async def test_audit_failure_blocks_provider_call(
    register_scripted, alice_user_id, monkeypatch,
):
    """Monkey-patch the tool_called audit insert to raise. Dispatcher
    must NOT call provider.execute → no provider state change."""
    manifest, provider = register_scripted()
    await _seed_active_identity(alice_user_id)

    real_audit = dispatcher._audit_then_commit
    call_count = {"n": 0}

    async def maybe_fail(db, **kwargs):
        call_count["n"] += 1
        if kwargs.get("event_type") == EVENT_TOOL_CALLED:
            raise vault.VaultAuditError("simulated audit failure")
        return await real_audit(db, **kwargs)

    monkeypatch.setattr(dispatcher, "_audit_then_commit", maybe_fail)

    async with async_session_maker() as db:
        result = await dispatcher.execute(
            db, alice_user_id, "scripted", "scripted__do",
            tool_input={}, channel="web",
        )

    assert isinstance(result, ConnectorToolError)
    assert "audit" in result.message.lower()
    assert provider.execute_call_count == 0


# ─── 9. Output redaction ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_output_redaction_strips_listed_fields_from_audit_only(
    register_scripted, alice_user_id,
):
    """Manifest declares output_redaction=['body']. Stub returns
    {body: 'secret', subject: 'hello'}. Audit metadata lacks 'body'
    but contains 'subject'. Result returned to caller is unredacted."""
    manifest, provider = register_scripted(output_redaction=["body"])
    payload = {"body": "secret email content", "subject": "hello world"}
    provider._execute_result = ConnectorOk(content=json.dumps(payload))
    await _seed_active_identity(alice_user_id)

    async with async_session_maker() as db:
        result = await dispatcher.execute(
            db, alice_user_id, "scripted", "scripted__do",
            tool_input={"body": "draft body", "to": "x@y.com"},
            channel="web",
        )

    # Caller sees the unredacted result.
    assert isinstance(result, ConnectorOk)
    parsed = json.loads(result.content)
    assert parsed["body"] == "secret email content"
    assert parsed["subject"] == "hello world"

    # Audit rows have the redacted version.
    async with async_session_maker() as db:
        called_event = (await db.execute(
            select(ConnectorEvent)
            .where(ConnectorEvent.user_id == alice_user_id)
            .where(ConnectorEvent.event_type == EVENT_TOOL_CALLED)
        )).scalar_one()
        called_meta = json.loads(called_event.metadata_json)
        # Input had 'body' — should be stripped from audit.
        assert "body" not in called_meta["input"], (
            f"redacted input must drop 'body', got: {called_meta['input']}"
        )
        assert called_meta["input"]["to"] == "x@y.com"

        succeeded_event = (await db.execute(
            select(ConnectorEvent)
            .where(ConnectorEvent.user_id == alice_user_id)
            .where(ConnectorEvent.event_type == EVENT_TOOL_SUCCEEDED)
        )).scalar_one()
        succeeded_meta = json.loads(succeeded_event.metadata_json)
        # Output redaction applied to the parsed JSON.
        assert "body" not in succeeded_meta["output"]
        assert succeeded_meta["output"]["subject"] == "hello world"


# ─── 10. Concurrent refresh coalescing ───────────────────────────────


@pytest.mark.asyncio
async def test_concurrent_refresh_coalesces_to_one(
    register_scripted, alice_user_id,
):
    """50 concurrent execute() calls on an expiring-token identity →
    provider.refresh invocation count == 1.

    The slight refresh delay simulates a real network round-trip and
    lets the lock contention actually occur."""
    manifest, provider = register_scripted()
    provider._refresh_delay_s = 0.05  # 50ms — guarantees overlap
    await _seed_active_identity(
        alice_user_id,
        expires_at=datetime.utcnow() + timedelta(seconds=5),
    )

    async def one_call():
        async with async_session_maker() as db:
            return await dispatcher.execute(
                db, alice_user_id, "scripted", "scripted__do",
                tool_input={}, channel="web",
            )

    results = await asyncio.gather(*[one_call() for _ in range(50)])

    # All callers got an Ok result.
    assert all(isinstance(r, ConnectorOk) for r in results)
    # But the provider.refresh was called exactly once.
    assert provider.refresh_call_count == 1, (
        f"expected exactly 1 refresh under coalescing, got "
        f"{provider.refresh_call_count}"
    )


# ─── Bonus invariants ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_unknown_connector_returns_tool_error(
    alice_user_id,
):
    """No registry entry for connector_id → ConnectorToolError, not
    ReauthRequired (the user never tried to connect → there's nothing
    to reauth)."""
    reset_registry_for_tests()  # ensure empty registry

    async with async_session_maker() as db:
        result = await dispatcher.execute(
            db, alice_user_id, "totally_unknown", "totally_unknown__do",
            tool_input={}, channel="web",
        )
    assert isinstance(result, ConnectorToolError)
    assert "unknown connector" in result.message


@pytest.mark.asyncio
async def test_unknown_tool_on_known_connector_returns_tool_error(
    register_scripted, alice_user_id,
):
    register_scripted()

    async with async_session_maker() as db:
        result = await dispatcher.execute(
            db, alice_user_id, "scripted", "scripted__not_declared",
            tool_input={}, channel="web",
        )
    assert isinstance(result, ConnectorToolError)
    assert "does not declare tool" in result.message


@pytest.mark.asyncio
async def test_refresh_coalesce_no_double_audit(
    register_scripted, alice_user_id,
):
    """50 coalescing callers → exactly ONE refresh_succeeded audit row,
    not 50."""
    manifest, provider = register_scripted()
    provider._refresh_delay_s = 0.05
    await _seed_active_identity(
        alice_user_id,
        expires_at=datetime.utcnow() + timedelta(seconds=5),
    )

    async def one_call():
        async with async_session_maker() as db:
            await dispatcher.execute(
                db, alice_user_id, "scripted", "scripted__do",
                tool_input={}, channel="web",
            )

    await asyncio.gather(*[one_call() for _ in range(50)])

    async with async_session_maker() as db:
        refresh_events = (await db.execute(
            select(ConnectorEvent)
            .where(ConnectorEvent.event_type == EVENT_REFRESH_SUCCEEDED)
            .where(ConnectorEvent.user_id == alice_user_id)
        )).scalars().all()
        assert len(refresh_events) == 1, (
            f"expected 1 refresh_succeeded audit row under coalescing, "
            f"got {len(refresh_events)}"
        )
