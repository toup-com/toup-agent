"""Toup for Mac — the consent precondition for a local write, run or click.

What this file pins
-------------------
`RELAY_PROTOCOL.md` §6 and `agent-tool-relay.md` §4.1/§4.2: the connector
pending-action invariants, applied to an action whose executor is the user's
own Mac.

    atomic claim guarded on status='pending'  — the double-tap defence
    a hard expires_at                         — a card found by scrolling
                                                through last week cannot fire
    tool_name as a COLUMN                     — the approve endpoint cannot be
                                                talked into running a different
                                                tool than the card showed
    rows retained after they go terminal      — they ARE the audit trail

Plus the one failure mode the connector flow does not have and has no
precedent to copy (§4.2): **approved-but-undeliverable.** The Mac is asleep,
the row is `approved` (unclaimable again, correctly) and nothing ran — or
worse, it runs hours later when the laptop wakes. The safe default the audit
recommends is taken: the tap is refused with a sentence, and the row stays
`pending` so the user can tap again.

Lane: RUN_MODE=platform (`desktop_pending_actions` is PLATFORM_ONLY).

Run:
    cd backend && DATABASE_URL="sqlite+aiosqlite:///:memory:" RUN_MODE=platform \
        python -m pytest tests/test_desktop_pending_actions.py -q
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from app.agent import desktop_bridge
from app.api import desktop as desktop_api
from app.config import settings
from app.db.database import async_session_maker
from app.db.models import AgentConfig, DesktopDevice, DesktopPendingAction

DEVICE_ID = "dev-1"


@pytest.fixture(autouse=True)
def _enabled_account_cohort(monkeypatch):
    # These tests exercise the enabled protocol; rollout-off has its own tests.
    monkeypatch.setattr(settings, "desktop_connections_rollout_pct", 100)
    monkeypatch.setattr(settings, "desktop_relay_enabled", True)


@pytest.fixture(autouse=True)
def _enabled_agent(monkeypatch):
    monkeypatch.setattr(settings, "desktop_relay_enabled", True)


def _app() -> FastAPI:
    from app.api.auth import router as auth_router

    app = FastAPI()
    app.include_router(auth_router, prefix=settings.api_prefix)
    app.include_router(desktop_api.router, prefix=settings.api_prefix)
    return app


class FakeSocket:
    def __init__(self) -> None:
        self.sent: List[Dict[str, Any]] = []

    async def send_text(self, text: str) -> None:
        self.sent.append(json.loads(text))

    async def close(self, code: int = 1000, reason: str = "") -> None:
        pass


@pytest.fixture(autouse=True)
def _clean():
    desktop_bridge.reset_for_tests()
    desktop_api._RATE_BUCKETS.clear()
    yield
    desktop_bridge.reset_for_tests()


@pytest.fixture
async def api() -> AsyncClient:
    transport = ASGITransport(app=_app())
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def tenant(test_user_id):
    key = f"agent-key-{uuid.uuid4().hex}"
    async with async_session_maker() as db:
        db.add(AgentConfig(user_id=test_user_id, agent_api_key=key))
        db.add(DesktopDevice(
            id=DEVICE_ID, user_id=test_user_id, device_name="Nariman's MacBook",
            flavor="direct", token_jti="jti-1",
            last_seen_at=datetime.utcnow(),
        ))
        await db.commit()
    prev = settings.agent_api_key
    settings.agent_api_key = key
    yield {"user_id": test_user_id, "key": key}
    settings.agent_api_key = prev


async def _stage(
    api: AsyncClient, tenant: dict, *, tool: str = "desktop__exec_run",
    payload: Optional[Dict[str, Any]] = None, reason: str = "Run the tests",
) -> Dict[str, Any]:
    r = await api.post(
        "/api/desktop/internal/stage-action",
        headers={"X-Agent-Key": tenant["key"]},
        json={
            "user_id": tenant["user_id"], "device_id": DEVICE_ID,
            "tool_name": tool,
            "payload": payload or {"command": "npm", "args": ["test"],
                                   "cwd": "Projects/toup"},
            "reason": reason, "channel": "desktop",
        },
    )
    assert r.status_code == 200, r.text
    return r.json()


async def _row(action_id: str) -> DesktopPendingAction:
    async with async_session_maker() as db:
        return (await db.execute(
            select(DesktopPendingAction).where(
                DesktopPendingAction.id == action_id
            )
        )).scalar_one()


async def _approve_with_device(
    api: AsyncClient, action_id: str, headers: dict, sock: FakeSocket,
    user_id: str, *, status: str = "ok", summary: str = "Tests passed.",
    data: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, List[Dict[str, Any]]]:
    """Tap approve and answer as the Mac, concurrently."""
    before = len(sock.sent)

    async def answer():
        for _ in range(300):
            if len(sock.sent) > before:
                frame = sock.sent[-1]
                desktop_bridge.deliver_inbound(user_id, {
                    "type": "result", "id": frame["id"],
                    "ok": status == "ok", "status": status,
                    "summary": summary, "data": data or {},
                })
                return
            await asyncio.sleep(0.01)

    t = asyncio.create_task(answer())
    resp = await api.post(
        f"/api/desktop/pending-actions/{action_id}/approve", headers=headers,
    )
    await t
    await desktop_api.wait_for_handoffs()
    return resp, sock.sent


# ══════════════════════════════════════════════════════════════════════
# 1. Staging — the card, and what it carries
# ══════════════════════════════════════════════════════════════════════
async def test_staging_writes_a_pending_row_and_a_renderable_card(api, tenant):
    card = await _stage(api, tenant)
    # The keys `tool_executor.py:1438-1470` broadcasts, so every client that
    # already renders a connector confirm card renders this one.
    assert set(card) >= {
        "action_id", "connector_id", "tool_name", "summary", "payload",
        "expires_at", "status",
    }
    assert card["connector_id"] == "desktop"
    assert card["tool_name"] == "desktop__exec_run"
    assert card["status"] == "pending"
    assert card["payload"] == {"command": "npm", "args": ["test"],
                               "cwd": "Projects/toup"}

    row = await _row(card["action_id"])
    assert row.status == "pending"
    assert row.task_id
    # SHORTER than the connector flow's 24 h, and deliberately so.
    ttl = (row.expires_at - row.created_at).total_seconds()
    assert ttl == desktop_api.PENDING_ACTION_TTL_S <= 24 * 3600


async def test_staging_requires_an_agent_key_that_owns_the_user(api, tenant):
    body = {"user_id": tenant["user_id"], "device_id": DEVICE_ID,
            "tool_name": "desktop__exec_run", "payload": {}, "reason": "x"}
    assert (await api.post(
        "/api/desktop/internal/stage-action", json=body,
    )).status_code == 401
    assert (await api.post(
        "/api/desktop/internal/stage-action", json=body,
        headers={"X-Agent-Key": "not-a-key"},
    )).status_code == 403
    # A valid key, someone else's user: the key identifies the tenant, and a
    # user id in the body may only confirm it.
    other = {**body, "user_id": str(uuid.uuid4())}
    assert (await api.post(
        "/api/desktop/internal/stage-action", json=other,
        headers={"X-Agent-Key": tenant["key"]},
    )).status_code == 403


async def test_the_list_belongs_to_its_owner(api, tenant, auth_headers):
    card = await _stage(api, tenant)
    mine = (await api.get(
        "/api/desktop/pending-actions", headers=auth_headers,
    )).json()["actions"]
    assert [a["action_id"] for a in mine] == [card["action_id"]]
    assert (await api.get("/api/desktop/pending-actions")).status_code in (401, 403)


# ══════════════════════════════════════════════════════════════════════
# 2. Approve — the claim, and what it may run
# ══════════════════════════════════════════════════════════════════════
async def test_approving_runs_exactly_the_staged_tool_and_payload(
    api, tenant, auth_headers,
):
    card = await _stage(api, tenant)
    sock = FakeSocket()
    await desktop_bridge.register(
        tenant["user_id"], sock, device_id=DEVICE_ID, token_jti="jti-1",
    )
    resp, sent = await _approve_with_device(
        api, card["action_id"], auth_headers, sock, tenant["user_id"],
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] in {"approved", "dispatched", "executed"}

    assert len(sent) == 1
    frame = sent[0]
    assert frame["action"] == "desktop__exec_run"
    assert frame["params"] == {"command": "npm", "args": ["test"],
                               "cwd": "Projects/toup"}
    row = await _row(card["action_id"])
    # §5.3: the id it was STAGED with, so a redelivery is answered from the
    # device's result cache rather than running `rm` twice.
    assert frame["id"] == row.task_id
    assert row.status == "executed"
    assert row.decided_at is not None and row.dispatched_at is not None


async def test_the_approve_endpoint_accepts_no_arguments_at_all(
    api, tenant, auth_headers,
):
    """`connector_pending_actions` takes edited arguments through an
    allowlist. For a tool whose argument is a shell command, the difference
    between a confirmation and a formality is that this one takes none —
    `tool_name` and `payload_json` come off the ROW."""
    card = await _stage(api, tenant)
    sock = FakeSocket()
    await desktop_bridge.register(
        tenant["user_id"], sock, device_id=DEVICE_ID, token_jti="jti-1",
    )
    before = len(sock.sent)

    async def answer():
        for _ in range(300):
            if len(sock.sent) > before:
                desktop_bridge.deliver_inbound(tenant["user_id"], {
                    "type": "result", "id": sock.sent[-1]["id"], "ok": True,
                    "status": "ok", "summary": "done", "data": {},
                })
                return
            await asyncio.sleep(0.01)

    t = asyncio.create_task(answer())
    resp = await api.post(
        f"/api/desktop/pending-actions/{card['action_id']}/approve",
        headers=auth_headers,
        json={"tool_name": "desktop__fs_trash",
              "payload": {"path": "Projects"},
              "command": "rm", "args": ["-rf", "/"]},
    )
    await t
    assert resp.status_code == 200, resp.text
    frame = sock.sent[-1]
    assert frame["action"] == "desktop__exec_run", (
        "the approve endpoint was talked into running a different tool"
    )
    assert frame["params"] == {"command": "npm", "args": ["test"],
                               "cwd": "Projects/toup"}


async def test_a_second_approval_of_one_action_runs_nothing(
    api, tenant, auth_headers,
):
    """The double-tap defence: every transition out of `pending` is one
    atomic UPDATE guarded on `status = 'pending'`, so two approvals race on
    one row and exactly one wins."""
    card = await _stage(api, tenant)
    sock = FakeSocket()
    await desktop_bridge.register(
        tenant["user_id"], sock, device_id=DEVICE_ID, token_jti="jti-1",
    )
    first, sent = await _approve_with_device(
        api, card["action_id"], auth_headers, sock, tenant["user_id"],
    )
    assert first.status_code == 200
    assert len(sent) == 1

    second = await api.post(
        f"/api/desktop/pending-actions/{card['action_id']}/approve",
        headers=auth_headers,
    )
    assert second.status_code == 409
    assert len(sock.sent) == 1, "the command ran twice"


async def test_approving_after_the_deadline_does_nothing(
    api, tenant, auth_headers,
):
    card = await _stage(api, tenant)
    sock = FakeSocket()
    await desktop_bridge.register(
        tenant["user_id"], sock, device_id=DEVICE_ID, token_jti="jti-1",
    )
    async with async_session_maker() as db:
        row = await db.get(DesktopPendingAction, card["action_id"])
        row.expires_at = datetime.utcnow() - timedelta(seconds=1)
        await db.commit()

    resp = await api.post(
        f"/api/desktop/pending-actions/{card['action_id']}/approve",
        headers=auth_headers,
    )
    assert resp.status_code == 410
    assert "expired" in resp.json()["detail"]
    assert sock.sent == []
    assert (await _row(card["action_id"])).status == "expired"


async def test_an_offline_mac_refuses_the_tap_and_keeps_the_row_tappable(
    api, tenant, auth_headers,
):
    """§4.2's safe default. Claiming first would BURN the row — `approved` is
    terminal-for-claiming by design, so an offline tap would produce an
    action that can never run and can never be re-approved."""
    card = await _stage(api, tenant)
    async with async_session_maker() as db:
        dev = await db.get(DesktopDevice, DEVICE_ID)
        dev.last_seen_at = datetime.utcnow() - timedelta(
            seconds=desktop_api.DEVICE_ONLINE_WINDOW_S + 5,
        )
        await db.commit()

    resp = await api.post(
        f"/api/desktop/pending-actions/{card['action_id']}/approve",
        headers=auth_headers,
    )
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert "Nariman's MacBook" in detail and "offline" in detail
    assert (await _row(card["action_id"])).status == "pending", (
        "the row was burned; the user can never tap it again"
    )


async def test_a_revoked_mac_cannot_be_the_target_of_an_approval(
    api, tenant, auth_headers,
):
    card = await _stage(api, tenant)
    async with async_session_maker() as db:
        dev = await db.get(DesktopDevice, DEVICE_ID)
        dev.revoked_at = datetime.utcnow()
        await db.commit()
    resp = await api.post(
        f"/api/desktop/pending-actions/{card['action_id']}/approve",
        headers=auth_headers,
    )
    assert resp.status_code == 409
    assert "no longer connected" in resp.json()["detail"]


async def test_another_account_cannot_approve_or_reject_my_action(
    api, tenant, auth_headers,
):
    from app.db.models import User
    from app.services.auth_service import create_access_token, get_password_hash

    card = await _stage(api, tenant)
    intruder_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(id=intruder_id, email=f"{intruder_id}@x.test",
                    hashed_password=get_password_hash("x" * 12), name="Other"))
        await db.commit()
    theirs = {"Authorization": f"Bearer {create_access_token(intruder_id)}"}

    assert (await api.post(
        f"/api/desktop/pending-actions/{card['action_id']}/approve",
        headers=theirs,
    )).status_code == 404
    assert (await api.post(
        f"/api/desktop/pending-actions/{card['action_id']}/reject",
        headers=theirs,
    )).status_code == 409
    assert (await _row(card["action_id"])).status == "pending"


# ══════════════════════════════════════════════════════════════════════
# 3. The device is still the authority (§6, recommendation 10)
# ══════════════════════════════════════════════════════════════════════
async def test_the_mac_may_refuse_what_the_server_approved(
    api, tenant, auth_headers,
):
    """A relay that trusted "the server says the user approved this" would
    make the server's compromise equal to the Mac's compromise."""
    card = await _stage(api, tenant)
    sock = FakeSocket()
    await desktop_bridge.register(
        tenant["user_id"], sock, device_id=DEVICE_ID, token_jti="jti-1",
    )
    resp, _ = await _approve_with_device(
        api, card["action_id"], auth_headers, sock, tenant["user_id"],
        status="denied", summary="That command is not on your allowed list.",
    )
    assert resp.status_code == 200
    assert resp.json()["status"] in {"approved", "dispatched", "failed"}
    row = await _row(card["action_id"])
    assert row.status == "failed"
    assert "allowed list" in (row.result_json or "")


async def test_the_audit_row_records_the_summary_and_never_the_payload(
    api, tenant, auth_headers,
):
    """§8 / ARCHITECTURE.md §5.8: the audit records THAT a tool ran — tool,
    target, decision, exit status, byte counts — and never contents."""
    card = await _stage(api, tenant)
    sock = FakeSocket()
    await desktop_bridge.register(
        tenant["user_id"], sock, device_id=DEVICE_ID, token_jti="jti-1",
    )
    await _approve_with_device(
        api, card["action_id"], auth_headers, sock, tenant["user_id"],
        summary="Ran npm test: 42 passed.",
        data={"stdout": "BEGIN RSA PRIVATE KEY …", "exit_code": 0},
    )
    row = await _row(card["action_id"])
    assert "42 passed" in (row.result_json or "")
    assert "RSA PRIVATE KEY" not in (row.result_json or "")


async def test_a_terminal_row_is_retained_as_the_audit_trail(
    api, tenant, auth_headers,
):
    card = await _stage(api, tenant)
    assert (await api.post(
        f"/api/desktop/pending-actions/{card['action_id']}/reject",
        headers=auth_headers,
    )).status_code == 200
    row = await _row(card["action_id"])
    assert row.status == "rejected"
    assert row.decided_via == "web"
    listed = (await api.get(
        "/api/desktop/pending-actions", params={"action_status": "all"},
        headers=auth_headers,
    )).json()["actions"]
    assert [a["status"] for a in listed] == ["rejected"]


async def test_rejecting_relays_nothing_and_cannot_be_repeated(
    api, tenant, auth_headers,
):
    card = await _stage(api, tenant)
    sock = FakeSocket()
    await desktop_bridge.register(
        tenant["user_id"], sock, device_id=DEVICE_ID, token_jti="jti-1",
    )
    assert (await api.post(
        f"/api/desktop/pending-actions/{card['action_id']}/reject",
        headers=auth_headers,
    )).status_code == 200
    assert sock.sent == []
    assert (await api.post(
        f"/api/desktop/pending-actions/{card['action_id']}/reject",
        headers=auth_headers,
    )).status_code == 409


async def test_a_rejected_action_can_never_be_approved_afterwards(
    api, tenant, auth_headers,
):
    card = await _stage(api, tenant)
    sock = FakeSocket()
    await desktop_bridge.register(
        tenant["user_id"], sock, device_id=DEVICE_ID, token_jti="jti-1",
    )
    await api.post(
        f"/api/desktop/pending-actions/{card['action_id']}/reject",
        headers=auth_headers,
    )
    resp = await api.post(
        f"/api/desktop/pending-actions/{card['action_id']}/approve",
        headers=auth_headers,
    )
    assert resp.status_code == 409
    assert sock.sent == []


async def test_listing_expires_a_card_the_user_is_looking_at(
    api, tenant, auth_headers,
):
    """Lazy expiry on read (`connector_pending_actions.py:401-419`): a card
    the user is looking at RIGHT NOW must never render as ACTIONABLE past
    its deadline, and this is the moment we know.

    Note the shape, which is the connector list's shape exactly: the row is
    still in this response (it was selected before it was expired) and it
    carries `status: "expired"`. That is the contract clients already
    implement, so the invariant to hold is "never `pending` past the
    deadline", not "absent". Tapping it anyway is refused — see
    `test_approving_after_the_deadline_does_nothing`."""
    card = await _stage(api, tenant)
    async with async_session_maker() as db:
        row = await db.get(DesktopPendingAction, card["action_id"])
        row.expires_at = datetime.utcnow() - timedelta(seconds=1)
        await db.commit()

    listed = (await api.get(
        "/api/desktop/pending-actions", headers=auth_headers,
    )).json()["actions"]
    assert [a["status"] for a in listed] == ["expired"]
    assert (await _row(card["action_id"])).status == "expired"

    # And it is gone from the next read, because the row is no longer
    # `pending`.
    again = (await api.get(
        "/api/desktop/pending-actions", headers=auth_headers,
    )).json()["actions"]
    assert again == []
