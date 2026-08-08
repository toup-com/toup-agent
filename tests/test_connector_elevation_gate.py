"""Confirmation gate for `elevation: true` connector tools.

What these tests are actually protecting:

  1. An elevated tool does NOT reach the provider. Before this arc,
     `elevation: true` was declared on ten tools and read by nobody —
     `gmail__send_message` went straight out. The first test is the
     regression that would catch that returning.
  2. Approving twice sends once. A double-tap on a phone with a slow
     network is the DEFAULT user behaviour when a button does not
     visibly respond, so "two taps, two emails" is not a hypothetical.
     `test_double_approve_executes_once` runs both approvals
     concurrently and asserts the provider saw exactly one call.
  3. The edited payload cannot smuggle arguments the card never showed,
     and cannot redirect the call to a different tool.
  4. An expired card is dead, not merely stale-looking.

The endpoint functions are called directly with a stub user rather than
over HTTP — same as the sibling connector test modules, and it keeps
the atomic-claim assertions on the real code path.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import ClassVar

import pytest
import pytest_asyncio
from cryptography.fernet import Fernet
from fastapi import HTTPException
from sqlalchemy import select

from app.api import connector_pending_actions as api
from app.config import settings
from app.connectors.base import (
    BaseConnectorProvider,
    ConnectorConfirmationRequired,
    ConnectorOk,
    ConnectorResult,
    ConnectorToolError,
    HealthResult,
    RefreshResult,
)
from app.db.database import async_session_maker
from app.db.models import (
    ConnectorEvent,
    ConnectorIdentity,
    ConnectorPendingAction,
    EVENT_TOOL_ELEVATION_REQUIRED,
    User,
)
from app.services import connector_dispatcher as dispatcher
from app.services import connector_vault as vault
from app.services.connector_registry import (
    get_registry,
    reset_registry_for_tests,
)
from app.services.credential_crypto import _multi_fernet


CONNECTOR = "elevated"
TOOL = "elevated__send"


class _CountingProvider(BaseConnectorProvider):
    """Counts execute() calls. The count IS the assertion in most of
    these tests — "did the side effect happen, and how many times"."""

    manifest_id: ClassVar[str] = CONNECTOR

    def __init__(self) -> None:
        self.execute_call_count = 0
        self.seen_inputs: list[dict] = []
        self._result: ConnectorResult = ConnectorOk(content='{"id":"msg_1"}')

    async def execute(self, tool_name, tool_input, ctx):
        self.execute_call_count += 1
        self.seen_inputs.append(dict(tool_input))
        return self._result

    async def revoke(self, user_id, access_token, refresh_token=None):
        return None

    async def refresh(self, refresh_token):
        return RefreshResult(
            access_token="at2",
            refresh_token="rt2",
            expires_at=datetime.utcnow() + timedelta(hours=1),
        )

    async def health_probe(self, ctx):
        return HealthResult(ok=True)


class _StubUser:
    """Stands in for the `get_current_user` dependency."""

    def __init__(self, user_id: str):
        self.id = user_id


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


@pytest.fixture
def register_elevated():
    """Register an `elevation: true` tool whose input_schema mirrors the
    real gmail send — required to/subject/body plus an optional cc, so
    the whitelist and required-field tests have something honest to bite
    on."""
    from app.services.connector_registry import (
        ChannelPolicy, ConnectorEntry, ConnectorManifest, ConnectorTool,
        HealthSpec, OAuthSpec,
    )

    def _register(*, elevation: bool = True):
        manifest = ConnectorManifest(
            manifest_version=1,
            id=CONNECTOR,
            name="Elevated",
            short_description="test-only",
            status="experimental",
            category="test",
            oauth=OAuthSpec(
                provider_app="stub_provider_app", scopes=[], pkce=True, refresh=True,
            ),
            health=HealthSpec(probe=TOOL),
            tools=[ConnectorTool(
                name=TOOL,
                description="x",
                input_schema={
                    "type": "object",
                    "properties": {
                        "to": {"type": "string"},
                        "subject": {"type": "string"},
                        "body": {"type": "string"},
                        "cc": {"type": "string"},
                    },
                    "required": ["to", "subject", "body"],
                },
                mutates=True,
                elevation=elevation,
                channel_policy=ChannelPolicy(default="allow", deny=[]),
            )],
        )
        provider = _CountingProvider()
        reset_registry_for_tests()
        reg = get_registry()
        reg._entries[CONNECTOR] = ConnectorEntry(manifest=manifest, provider=provider)
        reg._tool_index[TOOL] = CONNECTOR
        return manifest, provider

    yield _register
    reset_registry_for_tests()
    dispatcher.reset_locks_for_tests()


@pytest_asyncio.fixture
async def alice_user_id() -> str:
    async with async_session_maker() as db:
        uid = str(uuid.uuid4())
        db.add(User(
            id=uid, email=f"{uid[:8]}@example.com",
            hashed_password="x", name="Alice",
        ))
        await db.commit()
    return uid


async def _seed_identity(user_id: str) -> None:
    async with async_session_maker() as db:
        await vault.put(
            db, user_id, CONNECTOR,
            access_token="seed_at",
            refresh_token="seed_rt",
            access_expires_at=datetime.utcnow() + timedelta(hours=1),
        )


DRAFT = {"to": "sam@example.com", "subject": "Hi", "body": "I love you."}


async def _stage(user_id: str, payload: dict | None = None) -> ConnectorResult:
    async with async_session_maker() as db:
        return await dispatcher.execute(
            db, user_id, CONNECTOR, TOOL,
            tool_input=dict(payload or DRAFT),
            channel="web",
        )


# ─── The gate ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_elevated_tool_is_staged_not_executed(register_elevated, alice_user_id):
    """The regression that matters: elevation:true must not reach the
    provider. This is exactly what was broken — declared and unread."""
    _, provider = register_elevated()
    await _seed_identity(alice_user_id)

    result = await _stage(alice_user_id)

    assert isinstance(result, ConnectorConfirmationRequired)
    assert provider.execute_call_count == 0, "elevated tool reached the provider"
    assert "sam@example.com" in result.summary

    async with async_session_maker() as db:
        row = (await db.execute(
            select(ConnectorPendingAction).where(
                ConnectorPendingAction.id == result.action_id
            )
        )).scalar_one()
        assert row.status == "pending"
        assert row.tool_name == TOOL
        assert json.loads(row.payload_json)["to"] == "sam@example.com"

        events = (await db.execute(
            select(ConnectorEvent).where(
                ConnectorEvent.event_type == EVENT_TOOL_ELEVATION_REQUIRED
            )
        )).scalars().all()
        assert len(events) == 1


@pytest.mark.asyncio
async def test_non_elevated_tool_still_runs_straight_through(
    register_elevated, alice_user_id,
):
    """The gate must not leak onto tools that never asked for it."""
    _, provider = register_elevated(elevation=False)
    await _seed_identity(alice_user_id)

    result = await _stage(alice_user_id)

    assert isinstance(result, ConnectorOk)
    assert provider.execute_call_count == 1


@pytest.mark.asyncio
async def test_identical_draft_dedupes_to_one_card(register_elevated, alice_user_id):
    """Models re-issue the send after reading "awaiting confirmation".
    That must reuse the card, not stack a second one on the thread."""
    register_elevated()
    await _seed_identity(alice_user_id)

    first = await _stage(alice_user_id)
    second = await _stage(alice_user_id)

    assert first.action_id == second.action_id
    async with async_session_maker() as db:
        rows = (await db.execute(select(ConnectorPendingAction))).scalars().all()
        assert len(rows) == 1


@pytest.mark.asyncio
async def test_unconfirmable_channel_refuses_rather_than_executes(
    register_elevated, alice_user_id,
):
    """Fail SAFE. With nowhere to draw a card, the one outcome that must
    never happen is running the tool anyway."""
    _, provider = register_elevated()
    await _seed_identity(alice_user_id)

    async with async_session_maker() as db:
        result = await dispatcher.execute(
            db, alice_user_id, CONNECTOR, TOOL,
            tool_input=dict(DRAFT), channel="whatsapp",
        )

    assert isinstance(result, ConnectorToolError)
    assert provider.execute_call_count == 0


@pytest.mark.asyncio
async def test_approved_action_id_lifts_the_gate(register_elevated, alice_user_id):
    _, provider = register_elevated()
    await _seed_identity(alice_user_id)
    staged = await _stage(alice_user_id)

    async with async_session_maker() as db:
        result = await dispatcher.execute(
            db, alice_user_id, CONNECTOR, TOOL,
            tool_input=dict(DRAFT), channel="web",
            approved_action_id=staged.action_id,
        )

    assert isinstance(result, ConnectorOk)
    assert provider.execute_call_count == 1


# ─── Approve / reject ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_approve_runs_it_once_and_records_the_outcome(
    register_elevated, alice_user_id,
):
    _, provider = register_elevated()
    await _seed_identity(alice_user_id)
    staged = await _stage(alice_user_id)

    async with async_session_maker() as db:
        out = await api.approve_pending_action(
            staged.action_id, api.ApproveRequest(decided_via="web"),
            current_user=_StubUser(alice_user_id), db=db,
        )

    assert provider.execute_call_count == 1
    assert out.status == "executed"
    assert out.result and out.result.get("kind") == "ok"


@pytest.mark.asyncio
async def test_double_approve_executes_once(register_elevated, alice_user_id):
    """Two taps, one email. The guarded UPDATE is the only thing standing
    between a slow network and a duplicate send."""
    _, provider = register_elevated()
    await _seed_identity(alice_user_id)
    staged = await _stage(alice_user_id)

    async def _approve():
        async with async_session_maker() as db:
            return await api.approve_pending_action(
                staged.action_id, api.ApproveRequest(),
                current_user=_StubUser(alice_user_id), db=db,
            )

    results = await asyncio.gather(_approve(), _approve(), return_exceptions=True)

    assert provider.execute_call_count == 1, "the same draft was sent twice"
    conflicts = [r for r in results if isinstance(r, HTTPException) and r.status_code == 409]
    assert len(conflicts) == 1, f"expected exactly one 409, got {results}"


@pytest.mark.asyncio
async def test_edits_are_applied_and_undeclared_keys_dropped(
    register_elevated, alice_user_id,
):
    """The card exists so the user can fix the draft — the edit has to
    actually reach the provider. And an argument the manifest never
    declared must not ride along on the approved call."""
    _, provider = register_elevated()
    await _seed_identity(alice_user_id)
    staged = await _stage(alice_user_id)

    async with async_session_maker() as db:
        await api.approve_pending_action(
            staged.action_id,
            api.ApproveRequest(payload={
                "to": "someone-else@example.com",
                "body": "Edited body.",
                "evil_extra": "should not survive",
            }),
            current_user=_StubUser(alice_user_id), db=db,
        )

    sent = provider.seen_inputs[0]
    assert sent["to"] == "someone-else@example.com"
    assert sent["body"] == "Edited body."
    assert sent["subject"] == "Hi", "unedited fields must survive"
    assert "evil_extra" not in sent


@pytest.mark.asyncio
async def test_clearing_a_required_field_is_rejected_and_card_stays_open(
    register_elevated, alice_user_id,
):
    """A 422 must leave the draft actionable — burning the row on a typo
    would force the user to ask the agent all over again."""
    _, provider = register_elevated()
    await _seed_identity(alice_user_id)
    staged = await _stage(alice_user_id)

    with pytest.raises(HTTPException) as exc:
        async with async_session_maker() as db:
            await api.approve_pending_action(
                staged.action_id,
                api.ApproveRequest(payload={"to": "   "}),
                current_user=_StubUser(alice_user_id), db=db,
            )

    assert exc.value.status_code == 422
    assert provider.execute_call_count == 0
    async with async_session_maker() as db:
        row = (await db.execute(
            select(ConnectorPendingAction).where(
                ConnectorPendingAction.id == staged.action_id
            )
        )).scalar_one()
        assert row.status == "pending"


@pytest.mark.asyncio
async def test_expired_card_cannot_fire(register_elevated, alice_user_id):
    _, provider = register_elevated()
    await _seed_identity(alice_user_id)
    staged = await _stage(alice_user_id)

    async with async_session_maker() as db:
        row = (await db.execute(
            select(ConnectorPendingAction).where(
                ConnectorPendingAction.id == staged.action_id
            )
        )).scalar_one()
        row.expires_at = datetime.utcnow() - timedelta(seconds=1)
        await db.commit()

    with pytest.raises(HTTPException) as exc:
        async with async_session_maker() as db:
            await api.approve_pending_action(
                staged.action_id, api.ApproveRequest(),
                current_user=_StubUser(alice_user_id), db=db,
            )

    assert exc.value.status_code == 410
    assert provider.execute_call_count == 0


@pytest.mark.asyncio
async def test_another_users_card_is_not_found(register_elevated, alice_user_id):
    """404 not 403 — a 403 would confirm the id exists, which is an
    enumeration oracle over other people's pending sends."""
    register_elevated()
    await _seed_identity(alice_user_id)
    staged = await _stage(alice_user_id)

    async with async_session_maker() as db:
        bob = str(uuid.uuid4())
        db.add(User(
            id=bob, email=f"{bob[:8]}@example.com",
            hashed_password="x", name="Bob",
        ))
        await db.commit()

    with pytest.raises(HTTPException) as exc:
        async with async_session_maker() as db:
            await api.approve_pending_action(
                staged.action_id, api.ApproveRequest(),
                current_user=_StubUser(bob), db=db,
            )
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_reject_blocks_later_approval(register_elevated, alice_user_id):
    _, provider = register_elevated()
    await _seed_identity(alice_user_id)
    staged = await _stage(alice_user_id)

    async with async_session_maker() as db:
        out = await api.reject_pending_action(
            staged.action_id, api.RejectRequest(decided_via="app"),
            current_user=_StubUser(alice_user_id), db=db,
        )
    assert out.status == "rejected"

    with pytest.raises(HTTPException) as exc:
        async with async_session_maker() as db:
            await api.approve_pending_action(
                staged.action_id, api.ApproveRequest(),
                current_user=_StubUser(alice_user_id), db=db,
            )
    assert exc.value.status_code == 409
    assert provider.execute_call_count == 0


@pytest.mark.asyncio
async def test_double_reject_is_idempotent(register_elevated, alice_user_id):
    """A second tap on Cancel should be a no-op, not an error the user
    has to read and dismiss."""
    register_elevated()
    await _seed_identity(alice_user_id)
    staged = await _stage(alice_user_id)

    for _ in range(2):
        async with async_session_maker() as db:
            out = await api.reject_pending_action(
                staged.action_id, api.RejectRequest(),
                current_user=_StubUser(alice_user_id), db=db,
            )
            assert out.status == "rejected"


@pytest.mark.asyncio
async def test_identity_errors_win_over_staging(register_elevated, alice_user_id):
    """No identity → say so. Staging a draft against a connector that
    was never connected would produce a card that can never succeed."""
    _, provider = register_elevated()
    # deliberately no _seed_identity

    result = await _stage(alice_user_id)

    from app.connectors.base import ConnectorReauthRequired
    assert isinstance(result, ConnectorReauthRequired)
    assert provider.execute_call_count == 0
    async with async_session_maker() as db:
        rows = (await db.execute(select(ConnectorPendingAction))).scalars().all()
        assert rows == []
