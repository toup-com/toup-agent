"""Phone task -> approval -> Mac result, using real platform rows/account auth."""
import uuid
from unittest.mock import AsyncMock
import pytest
from sqlalchemy import select, update
from app.agent import desktop_bridge
from app.agent.skills.base import SkillContext
from app.agent.skills.builtins.desktop.skill import DesktopSkill, _post_stage
from app.api import desktop as api_module
from app.config import settings
from app.db.database import async_session_maker
from app.db.models import AgentConfig, DesktopPendingAction, DesktopTask
from test_desktop_connections_security import api, owner, _paired, _mark_online


@pytest.fixture(autouse=True)
def _runtime(monkeypatch):
    monkeypatch.setattr(settings, "desktop_relay_enabled", True)
    api_module._RATE_BUCKETS.clear()
    api_module._RUN_TASKS.clear()
    desktop_bridge.reset_for_tests()
    yield
    desktop_bridge.reset_for_tests()


async def _submit(api, owner, device, monkeypatch):
    conv_id = str(uuid.uuid4())
    handoff = AsyncMock(return_value=(202, {"conversation_id": conv_id}))
    monkeypatch.setattr(api_module, "_hand_off_task", handoff)
    body = {"text": "List the approved folder", "client_request_id": str(uuid.uuid4())}
    response = await api.post(f"/api/desktop/devices/{device}/tasks", json=body, headers=owner.headers("phone"))
    return response, handoff, body, conv_id


async def test_offline_writes_nothing_and_replay_does_not_run_twice(api, owner, monkeypatch):
    device, _ = await _paired(api, owner)
    response, handoff, body, _ = await _submit(api, owner, device, monkeypatch)
    assert response.status_code == 409
    handoff.assert_not_called()
    async with async_session_maker() as db:
        assert not list((await db.execute(select(DesktopTask))).scalars())
    await _mark_online(device)
    first = await api.post(f"/api/desktop/devices/{device}/tasks", json=body, headers=owner.headers("phone"))
    second = await api.post(f"/api/desktop/devices/{device}/tasks", json=body, headers=owner.headers("phone"))
    assert first.status_code == 202
    assert second.json()["id"] == first.json()["id"]
    assert handoff.await_count == 1


async def test_card_stays_with_task_until_phone_approves_and_mac_answers(api, owner, monkeypatch):
    device, _ = await _paired(api, owner)
    await _mark_online(device)
    response, _, _, conv_id = await _submit(api, owner, device, monkeypatch)
    assert response.status_code == 202, response.text
    task_id = response.json()["id"]
    await api_module._apply_task_status(owner.user_id, task_id, "running")
    async with async_session_maker() as db:
        await db.execute(update(AgentConfig).where(AgentConfig.user_id == owner.user_id).values(agent_api_key="owned-key"))
        await db.commit()
    monkeypatch.setattr(settings, "agent_api_key", "owned-key")
    token = desktop_bridge.DESKTOP_TARGET.set(desktop_bridge.DesktopTarget(task_id, device, "My Mac"))
    try:
        card = await _post_stage(user_id=owner.user_id, device_id=device,
            tool_name="desktop__fs_mkdir", payload={"path": "/approved/demo"}, reason="Create demo folder",
            ctx=SkillContext(user_id=owner.user_id, session_id=conv_id))
    finally:
        desktop_bridge.DESKTOP_TARGET.reset(token)
    await api_module._apply_task_status(owner.user_id, task_id, "ended")
    before = await api.get(f"/api/desktop/tasks/{task_id}", headers=owner.headers("phone"))
    assert before.json()["status"] == "waiting_on_you"
    assert before.json()["actions"][0]["action_id"] == card["action_id"]
    async with async_session_maker() as db:
        row = await db.get(DesktopPendingAction, card["action_id"])
        assert row.remote_task_id == task_id
        assert row.conversation_id == conv_id
    # Staging never dispatches. The phone POST below owns the approval.
    dispatch = AsyncMock(return_value={"summary": "Created the demo folder", "data": {}})
    monkeypatch.setattr(desktop_bridge, "is_device_connected", lambda uid, did: uid == owner.user_id and did == device)
    monkeypatch.setattr(desktop_bridge, "dispatch", dispatch)
    dispatch.assert_not_called()
    # Defer the same-process test handoff until this request has returned.
    # SQLite StaticPool has one connection, unlike the production two-process
    # deployment; overlapping sessions can roll back each other's writes.
    handoffs = []
    monkeypatch.setattr(api_module, "_spawn_tracked", lambda coro, **_: handoffs.append(coro))
    approved = await api.post(f'/api/desktop/pending-actions/{card["action_id"]}/approve', headers=owner.headers("phone"))
    assert approved.status_code == 200, approved.text
    for handoff in handoffs:
        await handoff
    done = (await api.get(f"/api/desktop/tasks/{task_id}", headers=owner.headers("phone"))).json()
    assert done["status"] == "done"
    assert done["outcome"] == "Created the demo folder"
    assert dispatch.call_args.kwargs["device_id"] == device


async def test_fast_running_status_keeps_conversation_link(api, owner, monkeypatch):
    device, _ = await _paired(api, owner)
    await _mark_online(device)
    conv_id = str(uuid.uuid4())
    async def fast(uid, task_id, *_):
        await api_module._apply_task_status(uid, task_id, "running")
        return 202, {"conversation_id": conv_id}
    monkeypatch.setattr(api_module, "_hand_off_task", fast)
    response = await api.post(f"/api/desktop/devices/{device}/tasks", json={"text":"Read folder", "client_request_id":str(uuid.uuid4())}, headers=owner.headers("phone"))
    assert response.json()["conversation_id"] == conv_id
    assert response.json()["status"] == "running"


async def test_cancelled_or_disabled_task_cannot_stage_or_dispatch(monkeypatch):
    token = desktop_bridge.DESKTOP_TARGET.set(desktop_bridge.DesktopTarget("task", "mac", "My Mac"))
    dispatch = AsyncMock()
    monkeypatch.setattr(desktop_bridge, "dispatch", dispatch)
    try:
        desktop_bridge.mark_task_cancelled("task")
        result = await DesktopSkill().execute_tool("desktop__fs_read", {"path":"/approved"}, SkillContext(user_id="owner"))
        assert "cancelled" in result
        dispatch.assert_not_called()
        monkeypatch.setattr(settings, "desktop_relay_enabled", False)
        result = await DesktopSkill().execute_tool("desktop__fs_read", {}, SkillContext(user_id="owner"))
        assert "disabled" in result
    finally:
        desktop_bridge.DESKTOP_TARGET.reset(token)


async def test_unlisted_account_cannot_enroll_and_removing_allowlist_blocks_execution(api, owner):
    from test_desktop_connections_security import _make_account, _init
    from app.services.feature_flags import set_allowlist
    other = await _make_account()
    denied = await api.post('/api/desktop/pair/init', json={"device_name":"Other Mac"}, headers=other.headers("mac"))
    assert denied.status_code == 404
    pending = await _init(api, owner)
    device, bundle = await _paired(api, owner)
    async with async_session_maker() as db:
        await set_allowlist(db, "desktop_connections", [])
    approval = await api.post('/api/desktop/pair/approve', json={"user_code":pending["user_code"]}, headers=owner.headers("phone"))
    assert approval.status_code == 404
    verdict = await api_module._verify_device_token(bundle["access_token"])
    assert verdict == {"ok":False, "reason":"disabled"}
    status = await api.get('/api/desktop/status', headers=owner.headers("phone"))
    assert status.status_code == 200 and status.json()["connections_enabled"] is False
    revoked = await api.post('/api/desktop/devices/revoke', json={"device_id":device}, headers=owner.headers("phone"))
    assert revoked.status_code == 200
