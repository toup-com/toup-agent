"""The actual Live WebSocket must bound tenant-dependent startup."""

import asyncio
import logging
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from app.api import ws_realtime as rt
from tests.test_voice_ws_lifecycle import ClientSocket, ProviderSocket


def _wire_live(monkeypatch, provider, *, budget=0.06):
    monkeypatch.setattr(rt, "_authenticate_ws", AsyncMock(return_value="owner-1"))
    monkeypatch.setattr(rt, "_get_user_openai_key_ex", AsyncMock(return_value=("key", True)))
    monkeypatch.setattr(rt, "_resolve_v2_for_user", lambda _user: False)
    monkeypatch.setattr(rt, "_live_protocol_selected", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(rt, "_agent_ctx_enabled_for", lambda _user: True)
    monkeypatch.setattr(rt, "_ensure_vps_user", AsyncMock())
    monkeypatch.setattr(rt, "resolve_voice_language", AsyncMock(return_value=None))
    monkeypatch.setattr(rt, "_get_user_tz_name", AsyncMock(return_value="America/Toronto"))
    monkeypatch.setattr(rt, "_defer_voice_la_end", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(rt, "_LIVE_STARTUP_CONTEXT_BUDGET_S", budget)
    monkeypatch.setattr(rt.websockets, "connect", AsyncMock(return_value=provider))


@pytest.mark.asyncio
@pytest.mark.parametrize("slow_leg", ["session", "context"])
async def test_slow_tenant_startup_fails_visibly_and_cancels_old_attempt(
    monkeypatch, slow_leg, caplog,
):
    client, provider = ClientSocket(), ProviderSocket()
    _wire_live(monkeypatch, provider)
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def blocked(*_args, **_kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    async def session(_user, _session, *, require_persisted):
        assert require_persisted is True
        return "durable-session"

    async def vps(_user):
        return "https://tenant.example", "agent-key"

    async def context_api(*_args, **_kwargs):
        return {"instructions": "# Today's Full Conversation History\nOwner's latest work"}

    monkeypatch.setattr(rt, "_get_or_create_voice_session",
                        blocked if slow_leg == "session" else session)
    monkeypatch.setattr(rt, "_get_vps_info", blocked if slow_leg == "context" else vps)
    monkeypatch.setattr(rt, "_vps_api", context_api)
    relay = AsyncMock()
    monkeypatch.setattr("app.services.live_voice_protocol.run_live_voice_session", relay)

    started = time.monotonic()
    with caplog.at_level(logging.INFO, logger=rt.logger.name):
        await asyncio.wait_for(rt.realtime_voice_ws(
            client, token=None, session_id="durable-session", onboarding=False,
        ), timeout=1.0)
    assert time.monotonic() - started < 0.5
    assert entered.is_set() and cancelled.is_set()
    assert any(frame.get("type") == "status" and frame.get("stage") == "restoring_context"
               for frame in client.sent)
    assert any(frame.get("type") == "error"
               and frame.get("code") == "live_context_unavailable"
               and frame.get("recoverable") is False for frame in client.sent)
    assert provider.closed and client.closed
    relay.assert_not_awaited()
    startup_lines = [record.getMessage() for record in caplog.records
                     if "[LIVE] startup leg" in record.getMessage()
                     or "[LIVE] context deadline" in record.getMessage()]
    assert any(f"leg={slow_leg} outcome=cancelled" in line for line in startup_lines)
    other_leg = "context" if slow_leg == "session" else "session"
    assert any(f"leg={other_leg} outcome=ready" in line for line in startup_lines)
    assert any(f"{slow_leg}=cancelled" in line and f"{other_leg}=ready" in line
               for line in startup_lines)
    assert "durable-session" not in "\n".join(startup_lines)
    assert "Owner's latest work" not in "\n".join(startup_lines)


@pytest.mark.asyncio
async def test_both_slow_owner_reads_are_identified_without_starting_empty_live(
    monkeypatch, caplog,
):
    client, provider = ClientSocket(), ProviderSocket()
    _wire_live(monkeypatch, provider)
    cancelled = []

    async def blocked_session(*_args, **_kwargs):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.append("session")
            raise

    async def blocked_vps(_user):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.append("context")
            raise

    monkeypatch.setattr(rt, "_get_or_create_voice_session", blocked_session)
    monkeypatch.setattr(rt, "_get_vps_info", blocked_vps)
    relay = AsyncMock()
    monkeypatch.setattr("app.services.live_voice_protocol.run_live_voice_session", relay)
    with caplog.at_level(logging.INFO, logger=rt.logger.name):
        await asyncio.wait_for(rt.realtime_voice_ws(
            client, token=None, session_id="durable-session", onboarding=False,
        ), timeout=1.0)
    assert set(cancelled) == {"session", "context"}
    assert not any(frame.get("type") == "ready" for frame in client.sent)
    assert any(frame.get("code") == "live_context_unavailable" for frame in client.sent)
    relay.assert_not_awaited()
    deadline = [record.getMessage() for record in caplog.records
                if "[LIVE] context deadline" in record.getMessage()]
    assert len(deadline) == 1
    assert "session=cancelled" in deadline[0] and "context=cancelled" in deadline[0]
    assert "durable-session" not in deadline[0]


@pytest.mark.asyncio
async def test_verified_day_context_reaches_first_live_start(monkeypatch):
    client, provider = ClientSocket(), ProviderSocket()
    _wire_live(monkeypatch, provider, budget=0.5)
    session_calls = []

    async def session(user, sid, *, require_persisted):
        session_calls.append((user, sid, require_persisted))
        return "durable-session"

    async def vps(_user):
        return "https://tenant.example", "agent-key"

    async def context_api(_url, _key, _method, path, **_kwargs):
        assert path == "/api/v1/internal/voice-context"
        return {"instructions": "# Today's Full Conversation History\nOwner's latest work",
                "degraded": []}

    monkeypatch.setattr(rt, "_get_or_create_voice_session", session)
    monkeypatch.setattr(rt, "_get_vps_info", vps)
    monkeypatch.setattr(rt, "_vps_api", context_api)
    accepted = []

    async def live_session(**kwargs):
        accepted.append((kwargs["db_session_id"], kwargs["instructions"]))
        await kwargs["websocket"].send_json({"type": "ready"})

    monkeypatch.setattr("app.services.live_voice_protocol.run_live_voice_session", live_session)
    await asyncio.wait_for(rt.realtime_voice_ws(
        client, token=None, session_id="durable-session", onboarding=False,
    ), timeout=1.0)
    assert session_calls == [("owner-1", "durable-session", True)]
    assert accepted == [("durable-session", "# Today's Full Conversation History\nOwner's latest work")]
    assert any(frame.get("type") == "ready" for frame in client.sent)


@pytest.mark.asyncio
async def test_failed_day_context_does_not_open_generic_live_greeting(monkeypatch):
    client, provider = ClientSocket(), ProviderSocket()
    _wire_live(monkeypatch, provider, budget=0.5)
    monkeypatch.setattr(rt, "_get_or_create_voice_session",
                        AsyncMock(return_value="durable-session"))
    monkeypatch.setattr(rt, "_get_vps_info",
                        AsyncMock(return_value=("https://tenant.example", "agent-key")))
    monkeypatch.setattr(rt, "_vps_api", AsyncMock(return_value={
        "instructions": "Generic identity with no day context",
        "degraded": ["day"],
    }))
    relay = AsyncMock()
    monkeypatch.setattr("app.services.live_voice_protocol.run_live_voice_session", relay)

    await asyncio.wait_for(rt.realtime_voice_ws(
        client, token=None, session_id="durable-session", onboarding=False,
    ), timeout=1.0)
    assert any(frame.get("code") == "live_context_unavailable" for frame in client.sent)
    assert not any(frame.get("type") == "ready" for frame in client.sent)
    relay.assert_not_awaited()


@pytest.mark.asyncio
async def test_live_session_resolver_does_not_mint_local_or_duplicate_on_outage(monkeypatch):
    async def vps(_user):
        return "https://tenant.example", "agent-key"

    calls = []

    async def lost_lookup(_url, _key, _method, path, **kwargs):
        calls.append(path)
        kwargs["outcome"].update(error="ReadTimeout", failure="transient")
        return None

    monkeypatch.setattr(rt, "_get_vps_info", vps)
    monkeypatch.setattr(rt, "_vps_api", lost_lookup)
    assert await rt._get_or_create_voice_session(
        "owner-1", "old-session", require_persisted=True,
    ) is None
    assert calls == ["/api/sessions/old-session"]

    monkeypatch.setattr(rt, "_get_vps_info", AsyncMock(return_value=None))
    assert await rt._get_or_create_voice_session(
        "owner-1", None, require_persisted=True,
    ) is None


@pytest.mark.asyncio
async def test_missing_client_id_reuses_verified_current_day_voice_session(monkeypatch):
    async def vps(_user):
        return "https://tenant.example", "agent-key"

    async def api(_url, _key, method, path, **kwargs):
        calls.append((method, path))
        if path == "/api/sessions/stale-id":
            kwargs["outcome"]["status"] = 404
            return None
        if path == "/api/sessions":
            return {"sessions": [{
                "id": "verified-today", "started_at": datetime.now(timezone.utc).isoformat(),
            }]}
        raise AssertionError("a current-day session must not be created twice")

    calls = []
    monkeypatch.setattr(rt, "_get_vps_info", vps)
    monkeypatch.setattr(rt, "_get_user_tz_name",
                        AsyncMock(return_value="America/Toronto"))
    monkeypatch.setattr(rt, "_vps_api", api)
    assert await rt._get_or_create_voice_session(
        "owner-1", "stale-id", require_persisted=True,
    ) == "verified-today"
    assert calls == [("GET", "/api/sessions/stale-id"), ("GET", "/api/sessions")]


@pytest.mark.asyncio
async def test_slow_old_provider_connect_cannot_steal_retry_live_activity(monkeypatch):
    old_client, old_provider = ClientSocket(), ProviderSocket()
    new_client, new_provider = ClientSocket(), ProviderSocket()
    _wire_live(monkeypatch, new_provider, budget=0.5)
    monkeypatch.setattr(rt, "_LIVE_STARTUP_READY_BUDGET_S", 1.0)
    monkeypatch.setattr(rt, "_get_or_create_voice_session",
                        AsyncMock(return_value="durable-session"))
    monkeypatch.setattr(rt, "_get_vps_info",
                        AsyncMock(return_value=("https://tenant.example", "agent-key")))
    monkeypatch.setattr(rt, "_vps_api", AsyncMock(return_value={
        "instructions": "# Today's Full Conversation History\nOwner's latest work",
        "degraded": [],
    }))
    old_connect_entered = asyncio.Event()
    release_old_connect = asyncio.Event()
    release_new_relay = asyncio.Event()
    relay_clients = []
    end_calls = []

    async def connect(*_args, **_kwargs):
        if not old_connect_entered.is_set():
            old_connect_entered.set()
            await release_old_connect.wait()
            return old_provider
        return new_provider

    async def relay(**kwargs):
        relay_clients.append(kwargs["websocket"])
        await release_new_relay.wait()

    async def old_send(frame):
        if old_client.closed:
            raise RuntimeError("client has closed")
        await old_client._send(frame)

    monkeypatch.setattr(rt.websockets, "connect", connect)
    monkeypatch.setattr("app.services.live_voice_protocol.run_live_voice_session", relay)
    monkeypatch.setattr(rt, "_defer_voice_la_end",
                        lambda *args, **_kwargs: end_calls.append(args))
    monkeypatch.setattr(old_client, "send_json", old_send)
    rt._live_connect_latest.pop("owner-1", None)
    rt._voice_session_owner.pop("owner-1", None)
    old_task = asyncio.create_task(rt.realtime_voice_ws(
        old_client, token=None, session_id="durable-session", onboarding=False,
    ))
    new_task = None
    try:
        await asyncio.wait_for(old_connect_entered.wait(), timeout=0.5)
        # Manual Retry closes the old phone socket while its provider connect
        # is still in flight, then the successor reaches the active relay.
        await old_client.close()
        new_task = asyncio.create_task(rt.realtime_voice_ws(
            new_client, token=None, session_id="durable-session", onboarding=False,
        ))
        async with asyncio.timeout(0.5):
            while not relay_clients:
                await asyncio.sleep(0)
        assert relay_clients == [new_client]
        new_owner = rt._voice_session_owner["owner-1"]
        release_old_connect.set()
        await asyncio.wait_for(old_task, timeout=0.5)
        assert rt._voice_session_owner["owner-1"] == new_owner
        assert relay_clients == [new_client]
        assert old_provider.closed
        assert not end_calls, "a stale attempt must not end the new Live Activity"
    finally:
        release_old_connect.set()
        release_new_relay.set()
        await asyncio.gather(old_task, *([new_task] if new_task else []),
                             return_exceptions=True)
        rt._live_connect_latest.pop("owner-1", None)
        rt._voice_session_owner.pop("owner-1", None)


@pytest.mark.asyncio
async def test_provider_connect_is_bounded_by_accept_to_ready_deadline(monkeypatch):
    client, provider = ClientSocket(), ProviderSocket()
    _wire_live(monkeypatch, provider, budget=0.5)
    monkeypatch.setattr(rt, "_LIVE_STARTUP_READY_BUDGET_S", 0.08)
    monkeypatch.setattr(rt, "_get_or_create_voice_session",
                        AsyncMock(return_value="durable-session"))
    monkeypatch.setattr(rt, "_get_vps_info",
                        AsyncMock(return_value=("https://tenant.example", "agent-key")))
    monkeypatch.setattr(rt, "_vps_api", AsyncMock(return_value={
        "instructions": "# Today's Full Conversation History\nOwner's latest work",
        "degraded": [],
    }))
    connect_cancelled = asyncio.Event()

    async def stalled_connect(*_args, **_kwargs):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            connect_cancelled.set()
            raise

    monkeypatch.setattr(rt.websockets, "connect", stalled_connect)
    relay = AsyncMock()
    monkeypatch.setattr("app.services.live_voice_protocol.run_live_voice_session", relay)
    started = time.monotonic()
    await asyncio.wait_for(rt.realtime_voice_ws(
        client, token=None, session_id="durable-session", onboarding=False,
    ), timeout=0.5)
    assert time.monotonic() - started < 0.5
    assert connect_cancelled.is_set()
    assert any(frame.get("type") == "error" for frame in client.sent)
    assert client.closed
    relay.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_retry_does_not_strand_prior_live_activity(monkeypatch):
    from app.services import live_activity_service

    ended = asyncio.Event()

    async def end(_user, _mission):
        ended.set()

    monkeypatch.setattr(live_activity_service, "end_voice_activities", end)
    monkeypatch.setattr(rt, "_LIVE_STARTUP_READY_BUDGET_S", 0.5)
    rt._voice_session_owner["owner-1"] = "old"
    rt._live_connect_latest["owner-1"] = "retry"
    try:
        rt._defer_voice_la_end("owner-1", "mission-1", "old", immediate=True)
        await asyncio.sleep(0.05)
        assert not ended.is_set()
        rt._live_connect_latest.pop("owner-1", None)  # retry failed
        await asyncio.wait_for(ended.wait(), timeout=0.5)
        assert "owner-1" not in rt._voice_session_owner
    finally:
        rt._live_connect_latest.pop("owner-1", None)
        rt._voice_session_owner.pop("owner-1", None)


@pytest.mark.asyncio
async def test_auth_db_stall_is_bounded_before_first_status(monkeypatch):
    client = ClientSocket()
    monkeypatch.setattr(rt, "_VOICE_AUTH_BUDGET_S", 0.08)
    cancelled = asyncio.Event()

    async def blocked_auth(_token):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    monkeypatch.setattr(rt, "_authenticate_ws", blocked_auth)
    started = time.monotonic()
    await asyncio.wait_for(rt.realtime_voice_ws(
        client, token=None, session_id="durable-session", onboarding=False,
    ), timeout=0.5)
    assert time.monotonic() - started < 0.5
    assert cancelled.is_set()
    assert client.closed
    assert not any(frame.get("type") == "status" for frame in client.sent)
