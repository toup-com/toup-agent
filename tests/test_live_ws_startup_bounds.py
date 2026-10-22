"""The actual Live WebSocket must bound tenant-dependent startup."""

import asyncio
import logging
import re
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
               and frame.get("recoverable") is False
               # D-R2(2): the startup dependency that failed, named.
               and frame.get("cause") == "deadline" for frame in client.sent)
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
    assert any(frame.get("code") == "live_context_unavailable"
               and frame.get("cause") == "personal_context" for frame in client.sent)
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


@pytest.mark.asyncio
async def test_a_reconnect_on_the_other_replica_keeps_the_calls_live_activity(monkeypatch):
    """D-P12: `_voice_session_owner` is per process and `numReplicas: 2`, so a
    call repaired onto the OTHER replica had its island card ended 6 s later
    by the dying socket here.  Every connect now claims the shared
    `infra_leases` row `voice-owner:<user>` (newest wins); a socket that is
    still the LOCAL owner reads it before ending the card."""

    from app.services import live_activity_service
    from app.services.infra_lease import claim_lease, lease_holder

    ended = []

    async def end(_user, mission):
        ended.append(mission)

    monkeypatch.setattr(live_activity_service, "end_voice_activities", end)
    monkeypatch.setattr(rt, "_VOICE_LA_END_GRACE_S", 0.01)
    name = rt._voice_owner_lease("owner-x")
    rt._voice_session_owner["owner-x"] = "old"
    try:
        assert await claim_lease(name, "old", 60)
        # The repaired call connected on replica B: its claim is the newest.
        assert await claim_lease(name, "new-on-replica-b", 60)
        rt._defer_voice_la_end("owner-x", "m1", "old", immediate=False)
        await asyncio.sleep(0.3)
        assert ended == [], "the card of a call live on the other replica was ended"
        assert await lease_holder(name) == "new-on-replica-b"

        # No successor anywhere: the card ends, and the lease is released.
        rt._voice_session_owner["owner-x"] = "solo"
        assert await claim_lease(name, "solo", 60)
        rt._defer_voice_la_end("owner-x", "m2", "solo", immediate=False)
        for _ in range(100):
            if ended:
                break
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)
        assert ended == ["m2"]
        assert await lease_holder(name) is None
    finally:
        rt._voice_session_owner.pop("owner-x", None)



# ── Voice round 2026-10-07, lane RB (D-R2(1)/(2)/(4)) ─────────────────────

def _wire_ready_context(monkeypatch, provider):
    _wire_live(monkeypatch, provider, budget=0.5)
    monkeypatch.setattr(rt, "_get_or_create_voice_session",
                        AsyncMock(return_value="durable-session"))
    monkeypatch.setattr(rt, "_get_vps_info",
                        AsyncMock(return_value=("https://tenant.example", "agent-key")))
    monkeypatch.setattr(rt, "_vps_api", AsyncMock(return_value={
        "instructions": "# Today's Full Conversation History\nOwner's latest work",
        "degraded": [],
    }))


@pytest.mark.asyncio
async def test_a_named_relay_fault_reaches_the_phone_with_its_cause(monkeypatch, caplog):
    """The 12:04 call's mid-call provider 1006 reached the phone as a bare
    `live_relay_error`; the trail said only `after=relay_error`."""

    from app.services.live_voice_protocol import LiveRelayFault

    client, provider = ClientSocket(), ProviderSocket()
    _wire_ready_context(monkeypatch, provider)

    async def dying_session(**_kwargs):
        raise LiveRelayFault("GPT-Live closed", cause="provider_closed", phase="call",
                             retried=1, provider_code=1006)

    monkeypatch.setattr("app.services.live_voice_protocol.run_live_voice_session", dying_session)
    with caplog.at_level(logging.INFO, logger=rt.logger.name):
        await asyncio.wait_for(rt.realtime_voice_ws(
            client, token=None, session_id="durable-session", onboarding=False,
        ), timeout=1.0)
    errors = [f for f in client.sent if f.get("type") == "error"]
    assert len(errors) == 1
    frame = errors[0]
    assert frame["code"] == "live_relay_error" and frame["recoverable"] is False
    assert (frame["cause"], frame["phase"], frame["retried"], frame["provider_code"]) == (
        "provider_closed", "call", 1, 1006,
    )
    assert isinstance(frame["elapsed_ms"], int) and frame["elapsed_ms"] >= 0
    failed = [r for r in caplog.records if "[LIVE] relay failed" in r.getMessage()]
    assert len(failed) == 1
    assert failed[0].levelno == logging.WARNING and failed[0].exc_info is None
    assert re.search(
        r"cause=provider_closed phase=call provider_code=1006 retried=1 elapsed_ms=\d+",
        failed[0].getMessage(),
    )


@pytest.mark.asyncio
async def test_an_unnamed_relay_fault_is_relay_internal_with_its_phase(monkeypatch, caplog):
    client, provider = ClientSocket(), ProviderSocket()
    _wire_ready_context(monkeypatch, provider)

    async def broken_session(**_kwargs):
        exc = KeyError("a bug")
        exc.relay_phase = "startup"
        raise exc

    monkeypatch.setattr("app.services.live_voice_protocol.run_live_voice_session", broken_session)
    with caplog.at_level(logging.INFO, logger=rt.logger.name):
        await asyncio.wait_for(rt.realtime_voice_ws(
            client, token=None, session_id="durable-session", onboarding=False,
        ), timeout=1.0)
    frame = next(f for f in client.sent if f.get("type") == "error")
    assert (frame["code"], frame["cause"], frame["phase"], frame["retried"]) == (
        "live_relay_error", "relay_internal", "startup", 0,
    )
    assert "provider_code" not in frame
    failed = [r for r in caplog.records if "[LIVE] relay failed" in r.getMessage()]
    assert len(failed) == 1 and failed[0].exc_info is not None, "a bug keeps its traceback"


@pytest.mark.parametrize(
    ("cause", "phase", "code"),
    [("provider_session_start_timeout", "startup", None),
     ("provider_start_rejected", "startup", None),
     ("provider_connect_failed", "startup", None),
     ("provider_closed", "startup", 1006)],
)
def test_every_named_cause_builds_its_frame(cause, phase, code):
    from app.services.live_voice_protocol import LiveRelayFault, LiveStartupTimeout

    cls = LiveStartupTimeout if cause == "provider_session_start_timeout" else LiveRelayFault
    fault = cls("x", cause=cause, phase=phase, retried=0, provider_code=code)
    frame = rt._live_relay_error_frame(fault, time.monotonic())
    assert (frame["cause"], frame["phase"], frame.get("provider_code")) == (cause, phase, code)
    # An unknown cause can never reach the wire: it reads as a relay bug.
    rogue = LiveRelayFault("x", cause="mystery", phase="call")
    assert rt._live_relay_error_frame(rogue, time.monotonic())["cause"] == "relay_internal"


@pytest.mark.asyncio
async def test_the_live_relay_is_handed_the_provider_connect_closure(monkeypatch):
    client, provider = ClientSocket(), ProviderSocket()
    _wire_ready_context(monkeypatch, provider)
    handed = {}

    async def live_session(**kwargs):
        handed.update(kwargs)
        await kwargs["websocket"].send_json({"type": "ready"})

    monkeypatch.setattr("app.services.live_voice_protocol.run_live_voice_session", live_session)
    await asyncio.wait_for(rt.realtime_voice_ws(
        client, token=None, session_id="durable-session", onboarding=False,
    ), timeout=1.0)
    factory = handed.get("provider_factory")
    assert callable(factory)
    connects_before = rt.websockets.connect.await_count
    assert await factory() is provider
    assert rt.websockets.connect.await_count == connects_before + 1


# ── D-R2(4): the warm-up ──────────────────────────────────────────────────

def _warm_wiring(monkeypatch, *, cohort=True, vps=True, answer=None, delay=0.0):
    monkeypatch.setattr(rt, "_voice_warm_last", {})
    monkeypatch.setattr(rt, "_voice_warm_inflight", {})
    monkeypatch.setattr(rt, "_live_protocol_selected", lambda *_a, **_k: cohort)
    monkeypatch.setattr(rt, "_resolve_v2_for_user", lambda _user: True)
    monkeypatch.setattr(rt, "_get_user_tz_name", AsyncMock(return_value="America/Toronto"))
    monkeypatch.setattr(rt, "_get_vps_info", AsyncMock(
        return_value=("https://tenant.example", "agent-key") if vps else None,
    ))
    bodies: list = []

    async def api(_url, _key, _method, path, *, json_body=None, timeout=None, **_kw):
        bodies.append((path, dict(json_body or {}), timeout))
        if delay:
            await asyncio.sleep(delay)
        return answer if answer is not None else {"instructions": "x", "cached": False, "age_ms": 0}

    monkeypatch.setattr(rt, "_vps_api", api)
    return bodies


@pytest.mark.asyncio
async def test_the_warm_sends_the_calls_own_body_plus_warm_and_is_debounced(monkeypatch, caplog):
    bodies = _warm_wiring(monkeypatch)
    with caplog.at_level(logging.INFO, logger=rt.logger.name):
        assert await rt.warm_live_voice("owner-12345678", "presence") == "ok"
        assert await rt.warm_live_voice("owner-12345678", "explicit") == "skipped_recent"
    assert len(bodies) == 1
    path, body, timeout = bodies[0]
    assert path == "/api/v1/internal/voice-context" and timeout == rt._VOICE_WARM_TIMEOUT_S
    assert body.pop("warm") is True
    # Byte-identical to the call's own context body (the tenant cache key).
    rt._v2_ctx.set(True)
    call_body = await rt._voice_context_body("owner-12345678", onboarding=False, live=True)
    body.pop("now"), call_body.pop("now")
    assert body == call_body
    lines = [r.getMessage() for r in caplog.records if "[LIVE] voice warm" in r.getMessage()]
    assert re.search(r"user=owner-12 reason=presence outcome=ok ms=\d+ cached=0 age_ms=0", lines[0])
    assert "reason=explicit outcome=skipped_recent" in lines[1]


@pytest.mark.asyncio
async def test_the_warm_is_live_cohort_only_and_survives_a_missing_tenant(monkeypatch):
    bodies = _warm_wiring(monkeypatch, cohort=False)
    assert await rt.warm_live_voice("u1", "presence") == "skipped_cohort"
    assert bodies == []
    bodies = _warm_wiring(monkeypatch, vps=False)
    assert await rt.warm_live_voice("u1", "presence") == "no_vps"
    assert bodies == []

    async def broken(*_a, **_k):
        raise OSError("tenant down")

    _warm_wiring(monkeypatch)
    monkeypatch.setattr(rt, "_vps_api", broken)
    assert await rt.warm_live_voice("u1", "presence") == "failed"
    assert "u1" not in rt._voice_warm_inflight


@pytest.mark.asyncio
async def test_the_warm_route_says_what_the_next_call_will_find(monkeypatch):
    from types import SimpleNamespace
    from app.api import voice

    user = SimpleNamespace(id="owner-1")
    _warm_wiring(monkeypatch, cohort=False)
    assert await voice.warm_voice(current_user=user) == {
        "ok": True, "warming": False, "cache": "skipped", "ttl_ms": 0,
    }
    _warm_wiring(monkeypatch)
    built = await voice.warm_voice(current_user=user)
    assert built == {"ok": True, "warming": True, "cache": "built", "ttl_ms": rt._VOICE_WARM_TTL_MS}
    again = await voice.warm_voice(current_user=user)
    assert again["cache"] == "fresh" and again["warming"] is False
    assert 0 < again["ttl_ms"] <= rt._VOICE_WARM_TTL_MS
    # The tenant answered from ITS cache: fresh, with the tenant's own age.
    _warm_wiring(monkeypatch, answer={"instructions": "x", "cached": True, "age_ms": 30000})
    cached = await voice.warm_voice(current_user=user)
    assert cached["cache"] == "fresh" and cached["ttl_ms"] == rt._VOICE_WARM_TTL_MS - 30000
    # A slow tenant: the route answers within its bound, the warm goes on.
    _warm_wiring(monkeypatch, delay=0.3)
    monkeypatch.setattr(rt, "_VOICE_WARM_ROUTE_WAIT_S", 0.05)
    pending = await voice.warm_voice(current_user=user)
    assert pending == {"ok": True, "warming": True, "cache": "pending",
                       "ttl_ms": rt._VOICE_WARM_TTL_MS}
    await asyncio.sleep(0.4)
    routes = {(r.path, tuple(sorted(r.methods)), r.status_code) for r in voice.router.routes
              if getattr(r, "path", "") == "/voice/warm"}
    assert routes == {("/voice/warm", ("POST",), 202)}


@pytest.mark.asyncio
async def test_a_foreground_presence_report_warms_the_live_voice_context(monkeypatch):
    from types import SimpleNamespace
    from app.api import live_activity_devices as lad

    warmed: list = []
    monkeypatch.setattr(rt, "schedule_live_voice_warm",
                        lambda user, reason: warmed.append((user, reason)))
    # No call open (Phase D2 ruling 48 skips the presence warm while one is).
    # Stated, not inherited: `_wire_live` stubs `_defer_voice_la_end`, the
    # only code that pops a call's owner entry, so every Live socket test
    # above leaves "owner-1" held when the whole file runs.
    monkeypatch.setattr(rt, "_voice_session_owner", {})

    class NoDevice:
        async def execute(self, _q):
            return SimpleNamespace(scalars=lambda: SimpleNamespace(first=lambda: None))

    user = SimpleNamespace(id="owner-1")
    out = await lad.report_presence(
        lad.LiveActivityPresence(foreground=True), current_user=user, db=NoDevice(),
    )
    assert out == {"ok": True, "updated": 0}
    assert warmed == [("owner-1", "presence")]
    await lad.report_presence(
        lad.LiveActivityPresence(foreground=False), current_user=user, db=NoDevice(),
    )
    assert warmed == [("owner-1", "presence")], "leaving the app warms nothing"


@pytest.mark.asyncio
async def test_no_presence_warm_while_the_users_live_call_is_open(monkeypatch, caplog):
    """Phase D2 ruling 48 (turns-startup-6): the app re-reports presence
    `foreground:true` every 4 minutes while it is on screen, voice mode
    included, and each report warmed the tenant — a voice-context rebuild the
    open call never reads.  While `_voice_session_owner` holds the user the
    presence warm is skipped (and logged); another user's open call skips
    nothing, and the user's own warm resumes once the call has ended."""

    from types import SimpleNamespace
    from app.api import live_activity_devices as lad

    warmed: list = []
    monkeypatch.setattr(rt, "schedule_live_voice_warm",
                        lambda user, reason: warmed.append((user, reason)))
    monkeypatch.setattr(rt, "_voice_session_owner", {"owner-12345678": "nonce-1"})

    class NoDevice:
        async def execute(self, _q):
            return SimpleNamespace(scalars=lambda: SimpleNamespace(first=lambda: None))

    async def foreground(user_id):
        return await lad.report_presence(
            lad.LiveActivityPresence(foreground=True),
            current_user=SimpleNamespace(id=user_id), db=NoDevice(),
        )

    with caplog.at_level(logging.INFO, logger=lad.logger.name):
        assert await foreground("owner-12345678") == {"ok": True, "updated": 0}
    assert warmed == [], "a presence warm during the user's own open call"
    assert any(
        r.getMessage() == "[LIVE] voice warm skipped user=owner-12 reason=call_open source=presence"
        for r in caplog.records
    ), [r.getMessage() for r in caplog.records]
    await foreground("other-user")
    assert warmed == [("other-user", "presence")]
    rt._voice_session_owner.pop("owner-12345678")
    await foreground("owner-12345678")
    assert warmed[-1] == ("owner-12345678", "presence")


@pytest.mark.asyncio
async def test_the_warm_body_and_answer_are_the_tenants_own_models(monkeypatch):
    """Phase C seam (DESIGN-R3 D-R2(4)): the relay's warm body parses as the
    tenant's `VoiceContextRequest` with `warm` set (an unknown key would be
    silently IGNORED by pydantic, i.e. the tenant would build no cache), and
    the tenant's `VoiceContextResponse` cache fields are what the relay's
    `cached=`/`age_ms=` reader reads."""

    from app.api.api_v1 import VoiceContextRequest, VoiceContextResponse

    bodies = _warm_wiring(monkeypatch, answer=VoiceContextResponse(
        instructions="x", cached=True, age_ms=41_250,
    ).model_dump(mode="json"))
    assert await rt.warm_live_voice("owner-12345678", "explicit") == "ok"
    req = VoiceContextRequest.model_validate(bodies[0][1])
    assert req.warm is True and req.live is True and req.onboarding is False
    assert req.tz_name == "America/Toronto" and req.budget_chars > 0
    assert rt._voice_warm_last["owner-12345678"][1:] == (True, 41_250)


@pytest.mark.asyncio
@pytest.mark.parametrize("why", ["skipped_recent", "skipped_busy"])
async def test_the_warm_route_never_says_pending_when_nothing_was_sent(monkeypatch, why):
    """Phase D turns-startup-7: a warm the scheduler skipped (debounced after
    a cache hit that has since aged out, or over the concurrency cap) sent
    nothing to the tenant, and the route still answered `{warming: true,
    cache: pending}` — the phone's trail said a build was in flight."""

    from types import SimpleNamespace

    from app.api import voice

    bodies = _warm_wiring(monkeypatch)
    if why == "skipped_recent":
        # answered 41 s ago from the tenant's cache aged 80 s (now 121 s old)
        rt._voice_warm_last["owner-1"] = (time.monotonic() - 41.0, True, 80_000)
    else:
        for i in range(rt._VOICE_WARM_MAX_CONCURRENT):
            rt._voice_warm_inflight[f"other-{i}"] = object()
    out = await voice.warm_voice(current_user=SimpleNamespace(id="owner-1"))
    assert bodies == [], bodies
    assert out.get("warming") is False and out.get("cache") == "skipped", out


@pytest.mark.asyncio
async def test_a_typed_fault_keeps_the_providers_own_code(monkeypatch, caplog):
    """Phase D turns-startup-8: the typed `[LIVE] relay failed` line logs no
    exception (a warning), so the provider's rejection code — once printed by
    `logger.exception` — reached neither the log nor the frame
    (`cause=provider_start_rejected pcode=-` for a `rate_limit_exceeded`).
    It rides both now: `detail=` on the line, `provider_error` on the frame."""

    from app.services.live_voice_protocol import LiveRelayFault

    client, provider = ClientSocket(), ProviderSocket()
    _wire_ready_context(monkeypatch, provider)

    async def rejected(**_kwargs):
        raise LiveRelayFault(
            "GPT-Live startup rejected: rate_limit_exceeded",
            cause="provider_start_rejected", phase="startup", detail="rate_limit_exceeded",
        )

    monkeypatch.setattr("app.services.live_voice_protocol.run_live_voice_session", rejected)
    with caplog.at_level(logging.INFO, logger=rt.logger.name):
        await asyncio.wait_for(rt.realtime_voice_ws(
            client, token=None, session_id="durable-session", onboarding=False,
        ), timeout=1.0)
    frame = next(f for f in client.sent if f.get("type") == "error")
    assert frame.get("provider_error") == "rate_limit_exceeded", frame
    failed = [r.getMessage() for r in caplog.records if "[LIVE] relay failed" in r.getMessage()]
    assert failed and failed[0].endswith("detail=rate_limit_exceeded"), failed
