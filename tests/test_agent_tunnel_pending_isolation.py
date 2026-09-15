import asyncio

import pytest
from fastapi import WebSocketDisconnect

from app.api import ws_agent_tunnel as tunnel


class FakeWebSocket:
    def __init__(self, *, fail_send: bool = False) -> None:
        self.sent: list[dict] = []
        self.close_codes: list[int] = []
        self.fail_send = fail_send

    async def send_json(self, value: dict) -> None:
        if self.fail_send:
            raise ConnectionError("closed")
        self.sent.append(value)

    async def close(self, code: int) -> None:
        self.close_codes.append(code)


def ready_connection(user_id: str, ws: FakeWebSocket) -> tunnel.TunnelConnection:
    connection = tunnel.TunnelConnection(user_id, ws)
    connection.accepting_calls = True
    return connection


@pytest.fixture(autouse=True)
def clean_module_state():
    tunnel._tunnels.clear()
    tunnel._pending_calls.clear()
    yield
    for pending in tunnel._pending_calls.values():
        if not pending.future.done():
            pending.future.cancel()
    tunnel._pending_calls.clear()
    tunnel._tunnels.clear()


@pytest.mark.asyncio
async def test_two_users_overlap_and_a_replacement_never_cancels_b():
    a_old = ready_connection("user-a", FakeWebSocket())
    b = ready_connection("user-b", FakeWebSocket())
    tunnel._tunnels.update({"user-a": a_old, "user-b": b})
    _, a_pending = tunnel._new_pending("user-a", a_old)
    _, b_pending = tunnel._new_pending("user-b", b)

    a_new = tunnel.TunnelConnection("user-a", FakeWebSocket())
    assert await tunnel._install_tunnel(a_new) is a_old
    assert tunnel._retire_tunnel(a_old) == 1

    with pytest.raises(tunnel.TunnelDisconnected):
        await a_pending.future
    assert not b_pending.future.done()
    assert tunnel._local_pending_count("user-a") == 0
    assert tunnel._local_pending_count("user-b") == 1
    assert tunnel._tunnels == {"user-a": a_new, "user-b": b}
    assert a_old.ws.close_codes == [4000]


@pytest.mark.asyncio
async def test_old_disconnect_after_replacement_keeps_new_current():
    old = ready_connection("same-user", FakeWebSocket())
    new = tunnel.TunnelConnection("same-user", FakeWebSocket())
    tunnel._tunnels["same-user"] = old

    await tunnel._install_tunnel(new)
    assert tunnel._retire_tunnel(old) == 0
    assert tunnel.get_tunnel("same-user") is new
    assert tunnel.is_agent_connected("same-user") is False
    assert tunnel._mark_tunnel_ready(new) is True
    assert tunnel.is_agent_connected("same-user") is True


@pytest.mark.asyncio
async def test_failed_connected_or_config_control_retires_only_exact_current():
    other = ready_connection("other-user", FakeWebSocket())
    failing = tunnel.TunnelConnection(
        "same-user", FakeWebSocket(fail_send=True),
    )
    tunnel._tunnels.update({"same-user": failing, "other-user": other})
    _, own_pending = tunnel._new_pending("same-user", failing)
    _, other_pending = tunnel._new_pending("other-user", other)

    assert await tunnel._send_current_control(
        failing, {"type": "connected", "user_id": "same-user"},
    ) is False
    with pytest.raises(tunnel.TunnelDisconnected):
        await own_pending.future
    assert tunnel.get_tunnel("same-user") is None
    assert tunnel.get_tunnel("other-user") is other
    assert not other_pending.future.done()


@pytest.mark.asyncio
async def test_connected_control_send_timeout_retires_exact_tunnel(monkeypatch):
    never = asyncio.Event()

    class HangingSend(FakeWebSocket):
        async def send_json(self, value: dict) -> None:
            await never.wait()

    current = tunnel.TunnelConnection("same-user", HangingSend())
    tunnel._tunnels["same-user"] = current
    monkeypatch.setattr(tunnel, "TUNNEL_CONTROL_SEND_TIMEOUT", 0.001)

    assert await tunnel._send_current_control(
        current, {"type": "connected", "user_id": "same-user"},
    ) is False
    assert tunnel.get_tunnel("same-user") is None


@pytest.mark.asyncio
async def test_control_for_superseded_tunnel_is_never_sent():
    old_ws = FakeWebSocket()
    old = tunnel.TunnelConnection("same-user", old_ws)
    newest = tunnel.TunnelConnection("same-user", FakeWebSocket())
    tunnel._tunnels["same-user"] = newest

    assert await tunnel._send_current_control(
        old, {"type": "connected", "user_id": "same-user"},
    ) is False
    assert old_ws.sent == []
    assert tunnel.get_tunnel("same-user") is newest


@pytest.mark.asyncio
async def test_control_replaced_during_send_cannot_retire_newest():
    newest = tunnel.TunnelConnection("same-user", FakeWebSocket())

    class ReplaceDuringSend(FakeWebSocket):
        async def send_json(self, value: dict) -> None:
            await super().send_json(value)
            tunnel._tunnels["same-user"] = newest

    old_ws = ReplaceDuringSend()
    old = tunnel.TunnelConnection("same-user", old_ws)
    tunnel._tunnels["same-user"] = old

    assert await tunnel._send_current_control(
        old, {"type": "connected", "user_id": "same-user"},
    ) is False
    assert len(old_ws.sent) == 1
    assert tunnel.get_tunnel("same-user") is newest


@pytest.mark.asyncio
async def test_replacement_close_has_finite_timeout_and_new_stays_current(monkeypatch):
    never = asyncio.Event()

    class HangingClose(FakeWebSocket):
        async def close(self, code: int) -> None:
            await never.wait()

    old = ready_connection("same-user", HangingClose())
    new = tunnel.TunnelConnection("same-user", FakeWebSocket())
    tunnel._tunnels["same-user"] = old
    monkeypatch.setattr(tunnel, "TUNNEL_REPLACE_CLOSE_TIMEOUT", 0.001)

    assert await tunnel._install_tunnel(new) is old
    assert tunnel.get_tunnel("same-user") is new


@pytest.mark.asyncio
async def test_result_is_accepted_only_from_owning_user_and_connection():
    a = tunnel.TunnelConnection("user-a", FakeWebSocket())
    b = tunnel.TunnelConnection("user-b", FakeWebSocket())
    call_id, pending = tunnel._new_pending("user-a", a)

    assert tunnel._resolve_pending(b, call_id, "foreign") is False
    assert not pending.future.done()
    assert tunnel._resolve_pending(a, call_id, "owned") is True
    assert await pending.future == "owned"


@pytest.mark.asyncio
async def test_same_user_replacement_does_not_retry_dispatched_tool_call():
    old_ws = FakeWebSocket()
    new_ws = FakeWebSocket()
    old = ready_connection("same-user", old_ws)
    new = tunnel.TunnelConnection("same-user", new_ws)
    tunnel._tunnels["same-user"] = old

    call = asyncio.create_task(
        tunnel.send_tool_call("same-user", "write_file", {"path": "opaque"})
    )
    await asyncio.sleep(0)
    assert len(old_ws.sent) == 1

    await tunnel._install_tunnel(new)
    tunnel._retire_tunnel(old)
    result = await call

    assert "outcome is unknown" in result
    assert "do not retry automatically" in result
    assert len(old_ws.sent) == 1
    assert new_ws.sent == []
    assert tunnel.get_tunnel("same-user") is new
    assert tunnel._pending_calls == {}


@pytest.mark.asyncio
async def test_side_effecting_http_forward_raises_typed_unknown_after_dispatch():
    old_ws = FakeWebSocket()
    old = ready_connection("same-user", old_ws)
    tunnel._tunnels["same-user"] = old
    call = asyncio.create_task(
        tunnel.send_http_forward(
            "same-user", "DELETE", "/api/apps/opaque", timeout=1.0,
        )
    )
    await asyncio.sleep(0)
    tunnel._retire_tunnel(old)

    with pytest.raises(tunnel.HTTPForwardOutcomeUnknown, match="not replayed"):
        await call
    assert len(old_ws.sent) == 1
    assert tunnel._pending_calls == {}


@pytest.mark.asyncio
async def test_tool_call_timeout_is_finite_and_never_retries(monkeypatch):
    ws = FakeWebSocket()
    tunnel._tunnels["user-a"] = ready_connection("user-a", ws)
    monkeypatch.setattr(tunnel, "TOOL_CALL_TIMEOUT", 0.001)

    result = await tunnel.send_tool_call("user-a", "write_file", {})

    assert "finite 0.001s tunnel timeout" in result
    assert "do not retry automatically" in result
    assert len(ws.sent) == 1
    assert tunnel._pending_calls == {}


@pytest.mark.asyncio
async def test_tool_call_hanging_send_is_inside_total_timeout(monkeypatch):
    entered = asyncio.Event()
    never = asyncio.Event()

    class HangingSend(FakeWebSocket):
        async def send_json(self, value: dict) -> None:
            entered.set()
            await never.wait()

    ws = HangingSend()
    tunnel._tunnels["user-a"] = ready_connection("user-a", ws)
    monkeypatch.setattr(tunnel, "TOOL_CALL_TIMEOUT", 0.001)

    result = await tunnel.send_tool_call("user-a", "write_file", {})

    assert entered.is_set()
    assert "outcome is unknown" in result
    assert "do not retry automatically" in result
    assert tunnel._pending_calls == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [0, -1, float("inf"), float("nan"), True, 30.1])
async def test_http_forward_rejects_nonfinite_or_unbounded_timeout(bad):
    tunnel._tunnels["user-a"] = ready_connection(
        "user-a", FakeWebSocket()
    )
    with pytest.raises(ValueError, match="timeout must be finite"):
        await tunnel.send_http_forward("user-a", "GET", "/api/apps/", timeout=bad)


@pytest.mark.asyncio
async def test_http_forward_owned_result_completes_without_retry():
    ws = FakeWebSocket()
    current = ready_connection("user-a", ws)
    tunnel._tunnels["user-a"] = current
    call = asyncio.create_task(
        tunnel.send_http_forward("user-a", "GET", "/api/apps/", timeout=1.0)
    )
    await asyncio.sleep(0)
    call_id = ws.sent[0]["id"]
    assert tunnel._resolve_pending(current, call_id, '{"ok":true}') is True
    assert await call == {"ok": True}
    assert len(ws.sent) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_result",
    [None, "null", "NaN", "Infinity", {"nested": [float("nan")]}],
)
async def test_mutating_http_invalid_result_is_unknown_not_fallback(
    invalid_result,
):
    ws = FakeWebSocket()
    current = ready_connection("user-a", ws)
    tunnel._tunnels["user-a"] = current
    call = asyncio.create_task(
        tunnel.send_http_forward(
            "user-a", "DELETE", "/api/apps/opaque", timeout=1.0,
        )
    )
    await asyncio.sleep(0)
    call_id = ws.sent[0]["id"]
    assert tunnel._resolve_pending(current, call_id, invalid_result) is True

    with pytest.raises(tunnel.HTTPForwardOutcomeUnknown, match="not replayed"):
        await call


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_result", [None, "null", "NaN", {"nested": float("inf")}],
)
async def test_safe_get_invalid_result_returns_none_for_direct_fallback(
    invalid_result,
):
    ws = FakeWebSocket()
    current = ready_connection("user-a", ws)
    tunnel._tunnels["user-a"] = current
    call = asyncio.create_task(
        tunnel.send_http_forward("user-a", "GET", "/api/apps/", timeout=1.0)
    )
    await asyncio.sleep(0)
    assert tunnel._resolve_pending(current, ws.sent[0]["id"], invalid_result)

    assert await call is None


@pytest.mark.asyncio
async def test_safe_get_timeout_returns_none_for_one_direct_fallback():
    ws = FakeWebSocket()
    tunnel._tunnels["user-a"] = ready_connection("user-a", ws)

    assert await tunnel.send_http_forward(
        "user-a", "GET", "/api/apps/", timeout=0.001,
    ) is None
    assert len(ws.sent) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "expected_unknown"), [("GET", False), ("DELETE", True)],
)
async def test_http_hanging_send_is_bounded_and_method_aware(
    method, expected_unknown,
):
    entered = asyncio.Event()
    never = asyncio.Event()

    class HangingSend(FakeWebSocket):
        async def send_json(self, value: dict) -> None:
            entered.set()
            await never.wait()

    tunnel._tunnels["user-a"] = ready_connection("user-a", HangingSend())
    call = tunnel.send_http_forward(
        "user-a", method, "/api/apps/opaque", timeout=0.001,
    )

    if expected_unknown:
        with pytest.raises(tunnel.HTTPForwardOutcomeUnknown, match="not replayed"):
            await call
    else:
        assert await call is None
    assert entered.is_set()
    assert tunnel._pending_calls == {}


@pytest.mark.asyncio
async def test_discard_consumes_unawaited_disconnect_exception():
    connection = tunnel.TunnelConnection("user-a", FakeWebSocket())
    call_id, pending = tunnel._new_pending("user-a", connection)
    assert tunnel._fail_pending_for_tunnel(connection) == 1

    tunnel._discard_pending(call_id, pending)

    assert call_id not in tunnel._pending_calls
    assert isinstance(pending.future.exception(), tunnel.TunnelDisconnected)


@pytest.mark.asyncio
async def test_install_pause_does_not_admit_calls_and_cancel_cleans_publication():
    close_started = asyncio.Event()
    never = asyncio.Event()

    class HangingClose(FakeWebSocket):
        async def close(self, code: int) -> None:
            close_started.set()
            await never.wait()

    old = ready_connection("same-user", HangingClose())
    new_ws = FakeWebSocket()
    new = tunnel.TunnelConnection("same-user", new_ws)
    tunnel._tunnels["same-user"] = old

    install = asyncio.create_task(tunnel._install_tunnel(new))
    await close_started.wait()
    assert tunnel.get_tunnel("same-user") is new
    assert tunnel.is_agent_connected("same-user") is False
    assert "not connected" in await tunnel.send_tool_call(
        "same-user", "write_file", {},
    )
    assert await tunnel.send_http_forward(
        "same-user", "GET", "/api/apps/",
    ) is None
    assert new_ws.sent == []

    install.cancel()
    with pytest.raises(asyncio.CancelledError):
        await install
    assert tunnel.get_tunnel("same-user") is None


@pytest.mark.asyncio
async def test_rapid_replacements_leave_only_newest_ready():
    close_started = asyncio.Event()
    never = asyncio.Event()

    class HangingClose(FakeWebSocket):
        async def close(self, code: int) -> None:
            close_started.set()
            await never.wait()

    old = ready_connection("same-user", HangingClose())
    middle = tunnel.TunnelConnection("same-user", FakeWebSocket())
    newest = tunnel.TunnelConnection("same-user", FakeWebSocket())
    tunnel._tunnels["same-user"] = old

    middle_install = asyncio.create_task(tunnel._install_tunnel(middle))
    await close_started.wait()
    assert await tunnel._install_tunnel(newest) is middle
    assert tunnel._mark_tunnel_ready(newest) is True

    middle_install.cancel()
    with pytest.raises(asyncio.CancelledError):
        await middle_install
    assert tunnel.get_tunnel("same-user") is newest
    assert tunnel.is_agent_connected("same-user") is True
    assert middle.accepting_calls is False


class EndpointWebSocket(FakeWebSocket):
    def __init__(self, *, blocking_send: bool = False) -> None:
        super().__init__()
        self.scope = {"subprotocols": []}
        self.query_params = {"token": "opaque"}
        self.accepted = False
        self.send_started = asyncio.Event()
        self.blocking_send = blocking_send
        self.never = asyncio.Event()

    async def accept(self, **kwargs) -> None:
        self.accepted = True

    async def send_json(self, value: dict) -> None:
        self.send_started.set()
        if self.blocking_send:
            await self.never.wait()
        await super().send_json(value)

    async def receive_text(self) -> str:
        await self.never.wait()
        raise WebSocketDisconnect


@pytest.mark.asyncio
async def test_endpoint_cancel_during_install_retires_published_tunnel(
    monkeypatch,
):
    close_started = asyncio.Event()
    never = asyncio.Event()

    class HangingClose(FakeWebSocket):
        async def close(self, code: int) -> None:
            close_started.set()
            await never.wait()

    old = ready_connection("same-user", HangingClose())
    tunnel._tunnels["same-user"] = old
    ws = EndpointWebSocket()

    async def authenticate(_token):
        return "same-user"

    monkeypatch.setattr(tunnel, "_authenticate_tunnel", authenticate)
    task = asyncio.create_task(tunnel.agent_tunnel_ws(ws, token="opaque"))
    await close_started.wait()
    published = tunnel.get_tunnel("same-user")
    assert published is not None and published is not old
    assert tunnel.is_agent_connected("same-user") is False

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert tunnel.get_tunnel("same-user") is None


@pytest.mark.asyncio
async def test_endpoint_cancel_during_connected_send_retires_published_tunnel(
    monkeypatch,
):
    ws = EndpointWebSocket(blocking_send=True)

    async def authenticate(_token):
        return "same-user"

    monkeypatch.setattr(tunnel, "_authenticate_tunnel", authenticate)
    task = asyncio.create_task(tunnel.agent_tunnel_ws(ws, token="opaque"))
    await ws.send_started.wait()
    assert tunnel.get_tunnel("same-user") is not None

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert tunnel.get_tunnel("same-user") is None


@pytest.mark.asyncio
async def test_endpoint_cancel_during_config_await_retires_published_tunnel(
    monkeypatch,
):
    ws = EndpointWebSocket()
    config_started = asyncio.Event()
    never = asyncio.Event()

    async def authenticate(_token):
        return "same-user"

    async def push_config(_tunnel):
        config_started.set()
        await never.wait()

    monkeypatch.setattr(tunnel, "_authenticate_tunnel", authenticate)
    monkeypatch.setattr(tunnel, "_push_latest_config", push_config)
    task = asyncio.create_task(tunnel.agent_tunnel_ws(ws, token="opaque"))
    await config_started.wait()
    assert tunnel.get_tunnel("same-user") is not None
    assert tunnel.is_agent_connected("same-user") is False

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert tunnel.get_tunnel("same-user") is None


@pytest.mark.asyncio
async def test_endpoint_config_timeout_retires_published_tunnel(monkeypatch):
    ws = EndpointWebSocket()
    never = asyncio.Event()

    async def authenticate(_token):
        return "same-user"

    async def push_config(_tunnel):
        await never.wait()

    monkeypatch.setattr(tunnel, "_authenticate_tunnel", authenticate)
    monkeypatch.setattr(tunnel, "_push_latest_config", push_config)
    monkeypatch.setattr(tunnel, "TUNNEL_CONFIG_SYNC_TIMEOUT", 0.001)

    await tunnel.agent_tunnel_ws(ws, token="opaque")
    assert tunnel.get_tunnel("same-user") is None
    assert tunnel.is_agent_connected("same-user") is False


@pytest.mark.asyncio
async def test_endpoint_config_send_failure_retires_published_tunnel(monkeypatch):
    ws = EndpointWebSocket()

    async def authenticate(_token):
        return "same-user"

    async def config_send_failed(_tunnel):
        return False

    monkeypatch.setattr(tunnel, "_authenticate_tunnel", authenticate)
    monkeypatch.setattr(tunnel, "_push_latest_config", config_send_failed)

    await tunnel.agent_tunnel_ws(ws, token="opaque")
    assert tunnel.get_tunnel("same-user") is None
    assert tunnel.is_agent_connected("same-user") is False
