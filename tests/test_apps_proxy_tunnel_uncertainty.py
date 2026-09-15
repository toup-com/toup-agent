import json
from types import SimpleNamespace

import pytest

from app.api import apps_proxy
from app.api import ws_agent_tunnel as tunnel


def _body(response):
    return json.loads(response.body)


@pytest.fixture(autouse=True)
def clean_tunnels():
    tunnel._tunnels.clear()
    tunnel._pending_calls.clear()
    yield
    tunnel._tunnels.clear()
    tunnel._pending_calls.clear()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("endpoint", "resource_id"),
    [
        (apps_proxy.delete_job, "job-id"),
        (apps_proxy.delete_app, "app-id"),
    ],
)
async def test_delete_unknown_is_non_2xx_and_never_directly_replayed(
    monkeypatch, endpoint, resource_id,
):
    direct_calls = 0

    async def uncertain(*args, **kwargs):
        raise tunnel.HTTPForwardOutcomeUnknown("already dispatched")

    async def forbidden_direct(*args, **kwargs):
        nonlocal direct_calls
        direct_calls += 1
        raise AssertionError("mutating fallback would duplicate an uncertain call")

    monkeypatch.setattr(tunnel, "is_agent_connected", lambda _user_id: True)
    monkeypatch.setattr(tunnel, "send_http_forward", uncertain)
    monkeypatch.setattr(apps_proxy, "_get_agent", forbidden_direct)

    response = await endpoint(
        resource_id,
        current_user=SimpleNamespace(id="same-user"),
        db=object(),
    )

    assert response.status_code == 502
    assert _body(response) == {
        "detail": "agent request outcome unknown; not retried",
        "retry_safe": False,
    }
    assert direct_calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_result",
    [None, "null", "NaN", "Infinity", {"nested": float("nan")}],
)
async def test_delete_actual_invalid_tunnel_result_never_directly_replays(
    monkeypatch, invalid_result,
):
    direct_calls = 0

    class NullResultWebSocket:
        async def send_json(self, payload):
            current = tunnel.get_tunnel("same-user")
            assert current is not None
            assert tunnel._resolve_pending(
                current, payload["id"], invalid_result,
            ) is True

    current = tunnel.TunnelConnection("same-user", NullResultWebSocket())
    current.accepting_calls = True
    tunnel._tunnels["same-user"] = current

    async def forbidden_direct(*args, **kwargs):
        nonlocal direct_calls
        direct_calls += 1
        raise AssertionError("invalid post-dispatch result must not replay DELETE")

    monkeypatch.setattr(apps_proxy, "_get_agent", forbidden_direct)
    response = await apps_proxy.delete_job(
        "job-id",
        current_user=SimpleNamespace(id="same-user"),
        db=object(),
    )

    assert response.status_code == 502
    assert _body(response)["retry_safe"] is False
    assert direct_calls == 0


@pytest.mark.asyncio
async def test_delete_predispatch_disconnect_falls_back_exactly_once(monkeypatch):
    tunnel_calls = 0
    direct_lookups = 0
    direct_result = object()

    async def disconnected_before_dispatch(*args, **kwargs):
        nonlocal tunnel_calls
        tunnel_calls += 1
        return None

    async def get_agent(*args, **kwargs):
        nonlocal direct_lookups
        direct_lookups += 1
        return "https://agent.invalid", "key", None

    async def direct(*args, **kwargs):
        return direct_result

    monkeypatch.setattr(tunnel, "is_agent_connected", lambda _user_id: True)
    monkeypatch.setattr(tunnel, "send_http_forward", disconnected_before_dispatch)
    monkeypatch.setattr(apps_proxy, "_get_agent", get_agent)
    monkeypatch.setattr(apps_proxy, "_proxy", direct)

    response = await apps_proxy.delete_job(
        "job-id",
        current_user=SimpleNamespace(id="same-user"),
        db=object(),
    )

    assert response is direct_result
    assert tunnel_calls == 1
    assert direct_lookups == 1


@pytest.mark.asyncio
async def test_safe_get_tunnel_failure_keeps_direct_fallback(monkeypatch):
    direct_calls = 0
    direct_result = object()

    async def tunnel_failed(*args, **kwargs):
        return None

    async def get_agent(*args, **kwargs):
        return "https://agent.invalid", "key", None

    async def direct(*args, **kwargs):
        nonlocal direct_calls
        direct_calls += 1
        return direct_result

    monkeypatch.setattr(tunnel, "is_agent_connected", lambda _user_id: True)
    monkeypatch.setattr(tunnel, "send_http_forward", tunnel_failed)
    monkeypatch.setattr(apps_proxy, "_get_agent", get_agent)
    monkeypatch.setattr(apps_proxy, "_proxy", direct)

    response = await apps_proxy.list_jobs(
        SimpleNamespace(url=SimpleNamespace(query="")),
        current_user=SimpleNamespace(id="same-user"),
        db=object(),
    )

    assert response is direct_result
    assert direct_calls == 1
