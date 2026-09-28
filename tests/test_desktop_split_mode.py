"""Exercise the actual tenant receiver without a platform DB replica.

Run with RUN_MODE=agent: the desktop tables deliberately do not exist here.
"""
import asyncio
import json
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from app.api import desktop as desktop_api, apps
from app.agent import desktop_bridge
from app.config import settings
from app.db.database import async_session_maker
from app.db.models import User, Conversation

pytestmark = pytest.mark.skipif(
    settings.run_mode != "agent",
    reason="tenant receiver requires agent-only conversation tables",
)


@pytest.fixture(autouse=True)
def bound_runtime(monkeypatch):
    monkeypatch.setattr(settings, "desktop_relay_enabled", True)
    monkeypatch.setattr(desktop_api, "_platform_db_local", lambda: False)
    monkeypatch.setattr("app.services.runtime_identity.get_user_id", lambda: "owner")
    monkeypatch.setattr("app.services.runtime_identity.get_agent_api_key", lambda: "owned-key")
    desktop_api._RUN_TASKS.clear()
    desktop_api._DISPATCHED.clear()
    desktop_bridge.reset_for_tests()
    yield
    desktop_bridge.reset_for_tests()


@pytest.fixture
async def api():
    app = FastAPI()
    app.include_router(desktop_api.router, prefix="/api")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client


class FakeSocket:
    def __init__(self): self.sent = []
    async def send_text(self, text): self.sent.append(json.loads(text))


async def test_agent_callbacks_use_bound_runtime_key(monkeypatch):
    import httpx

    seen = []
    original_client = httpx.AsyncClient

    def respond(request):
        seen.append((request.url.path, request.headers.get("X-Agent-Key")))
        return httpx.Response(200, json={"ok": True})

    monkeypatch.setattr(settings, "agent_api_key", "stale-key")
    monkeypatch.setattr(settings, "platform_api_url", "https://platform.example/api")
    monkeypatch.setattr(
        httpx, "AsyncClient",
        lambda **kwargs: original_client(
            transport=httpx.MockTransport(respond), **kwargs,
        ),
    )
    assert (await desktop_api._verify_device_token("device-token"))["ok"]
    assert (await desktop_api._post_heartbeat("owner", "mac", "jti", online=True))["ok"]
    assert (await desktop_api._post_to_platform("desktop/internal/task-status", {"user_id": "owner"}))["ok"]
    assert seen == [
        ("/api/desktop/internal/verify-token", "owned-key"),
        ("/api/desktop/heartbeat", "owned-key"),
        ("/api/desktop/internal/task-status", "owned-key"),
    ]


@pytest.mark.parametrize(
    ("stopped_reason", "expected"),
    [("", "ended"), ("credit_budget", "errored")],
)
async def test_task_turn_reports_runner_stop_reason(
    monkeypatch, stopped_reason, expected,
):
    reported = []

    class Runner:
        async def run(self, **kwargs):
            return SimpleNamespace(stopped_reason=stopped_reason)

    async def record(user_id, task_id, state):
        reported.append(state)

    monkeypatch.setattr(desktop_api, "_post_task_status", record)
    task = desktop_api.RunTaskReq(
        user_id="owner", task_id=str(uuid.uuid4()), device_id="mac",
        text="List a folder",
    )
    target = desktop_bridge.DesktopTarget(task.task_id, "mac", "My Mac")
    await desktop_api._run_task_turn(Runner(), "owner", task, "conversation", target)
    assert reported == ["running", expected]


async def test_run_uses_bound_key_and_own_conversation_and_is_idempotent(api, monkeypatch):
    async with async_session_maker() as db:
        db.add(User(id="owner", email="owner@example.com", hashed_password=""))
        await db.commit()
    sock = FakeSocket()
    await desktop_bridge.register("owner", sock, device_id="mac", token_jti="jti")
    entered = asyncio.Event()
    finish = asyncio.Event()
    seen = []
    async def run(**kw):
        seen.append((kw, desktop_bridge.DESKTOP_TARGET.get()))
        entered.set()
        await finish.wait()
    monkeypatch.setattr(apps, "_agent_runner", type("Runner", (), {"run": staticmethod(run)})())
    post = AsyncMock(return_value={"ok": True})
    monkeypatch.setattr(desktop_api, "_post_to_platform", post)
    body = {"user_id":"owner", "task_id":str(uuid.uuid4()), "device_id":"mac", "text":"List a folder"}
    path = "/api/desktop/internal/run-task"
    assert (await api.post(path, json=body)).status_code == 401
    assert (await api.post(path, json=body, headers={"X-Agent-Key":"other-key"})).status_code == 403
    assert (await api.post(path, json={**body,"user_id":"other"}, headers={"X-Agent-Key":"owned-key"})).status_code == 403
    first, second = await asyncio.gather(*[api.post(path, json=body, headers={"X-Agent-Key":"owned-key"}) for _ in range(2)])
    assert first.status_code == second.status_code == 202
    assert first.json() == second.json()
    await asyncio.wait_for(entered.wait(), 2)
    assert len(seen) == 1
    kw, target = seen[0]
    assert kw["session_id"] == first.json()["conversation_id"]
    assert kw["connector_scope"] == [] and kw["channel"] == "app"
    assert target.device_id == "mac" and target.task_id == body["task_id"]
    async with async_session_maker() as db:
        conv = await db.get(Conversation, kw["session_id"])
        assert conv.user_id == "owner" and conv.day_chat_id is None
    finish.set()
    await desktop_api.wait_for_handoffs()
    assert [call.args[1]["turn_state"] for call in post.call_args_list] == ["running", "ended"]


async def test_receiver_refuses_disabled_and_disconnected_mac(api, monkeypatch):
    body = {"user_id":"owner", "task_id":str(uuid.uuid4()), "device_id":"mac", "text":"List a folder"}
    headers = {"X-Agent-Key":"owned-key"}
    monkeypatch.setattr(settings, "desktop_relay_enabled", False)
    off = await api.post("/api/desktop/internal/run-task", json=body, headers=headers)
    assert off.status_code == 409 and off.json()["error"] == "disabled"
    monkeypatch.setattr(settings, "desktop_relay_enabled", True)
    offline = await api.post("/api/desktop/internal/run-task", json=body, headers=headers)
    assert offline.status_code == 409 and offline.json()["error"] == "mac_offline"


async def test_action_result_posts_to_platform_and_cancel_sends_targeted_frame(api, monkeypatch):
    post = AsyncMock(return_value={"ok": True})
    monkeypatch.setattr(desktop_api, "_post_to_platform", post)
    await desktop_api._record_action_result("action", "executed", "Read 1 entry", user_id="owner")
    post.assert_awaited_once_with("desktop/internal/action-result", {
        "user_id":"owner", "action_id":"action", "status":"executed", "summary":"Read 1 entry"})
    sock = FakeSocket()
    await desktop_bridge.register("owner", sock, device_id="mac", token_jti="jti")
    desktop_bridge.track_relay_id("owner", "relay-id", "action", "task")
    dispatch = asyncio.create_task(desktop_bridge.dispatch("owner", "desktop__fs_list", {}, device_id="mac", task_id="relay-id"))
    for _ in range(20):
        if sock.sent: break
        await asyncio.sleep(0)
    response = await api.post("/api/desktop/internal/cancel-task", json={"user_id":"owner","task_id":"task"}, headers={"X-Agent-Key":"owned-key"})
    assert response.status_code == 200 and response.json()["relay_cancelled"] == 1
    assert {"type":"cancel", "id":"relay-id"} in sock.sent
    assert desktop_bridge.is_task_cancelled("task")
    dispatch.cancel()
    await asyncio.gather(dispatch, return_exceptions=True)


async def test_approved_handoff_returns_before_mac_result(api, monkeypatch):
    finish = asyncio.Event()
    calls = []
    async def execute(**kw):
        calls.append(kw)
        await finish.wait()
    monkeypatch.setattr(desktop_api, "_run_approved_on_device", execute)
    body = {"user_id":"owner", "action_id":str(uuid.uuid4()), "device_id":"mac", "tool_name":"desktop__exec_run", "task_id":"relay", "payload":{"command":"pwd"}, "remote_task_id":"task"}
    response = await api.post('/api/desktop/internal/dispatch-approved', json=body, headers={"X-Agent-Key":"owned-key"})
    assert response.status_code == 202
    assert not finish.is_set()
    await asyncio.sleep(0)
    assert calls[0]["device_id"] == "mac" and calls[0]["remote_task_id"] == "task"
    repeat = await api.post('/api/desktop/internal/dispatch-approved', json=body, headers={"X-Agent-Key":"owned-key"})
    assert repeat.json()["duplicate"] is True
    finish.set()
    await desktop_api.wait_for_handoffs()
    assert len(calls) == 1
