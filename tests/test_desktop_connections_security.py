"""Toup for Mac — the five rules a phone connection stands on.

What this file pins
-------------------
`desktop/docs/CONNECTIONS.md` §3.3 (the challenge lifecycle: one use, one
window), §5.1-§5.3 (isolation between accounts, Macs and phones), §6
(revocation is a teardown, not a label) and §9 (least privilege: a task may
use only what the Mac's owner switched on).

Each test was watched to FAIL against a one-line mutation of the product
code it guards; the mutation is named above the test. A test nobody has
seen fail is a decoration, and this file exists precisely because these
five rules have no other alarm: every one of them fails SILENTLY and in the
safe-looking direction — a pairing that works twice, a challenge that
outlives its window, another account's answer accepted, a disconnected Mac
that still holds a credential, a tool nobody granted.

Lane: RUN_MODE=platform. `desktop_pairings`, `desktop_devices`,
`desktop_tasks` and `desktop_pending_actions` are all PLATFORM_ONLY
(`models/base.py`), so this file belongs to the platform sweep and needs no
`tests/COVERAGE_DEBT.txt` entry.

Run:
    cd backend && DATABASE_URL="sqlite+aiosqlite:///:memory:" RUN_MODE=platform \
        python -m pytest tests/test_desktop_connections_security.py -q
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select, update as sa_update

from app.agent import desktop_bridge
from app.agent.skills.base import SkillContext
from app.agent.skills.builtins.desktop.skill import DesktopSkill
from app.api import desktop as desktop_api
from app.config import settings
from app.db.database import async_session_maker
from app.db.models import (
    AgentConfig,
    DesktopDevice,
    DesktopPairing,
    DesktopPendingAction,
    DesktopTask,
    User,
    UserSession,
)
from app.services.auth_service import create_access_token, get_password_hash


@pytest.fixture(autouse=True)
def _enabled_agent(monkeypatch):
    monkeypatch.setattr(settings, "desktop_relay_enabled", True)

# ── A test app carrying only what these rules touch ──────────────────
def _app() -> FastAPI:
    from app.api.auth import router as auth_router

    app = FastAPI()
    app.include_router(auth_router, prefix=settings.api_prefix)
    app.include_router(desktop_api.router, prefix=settings.api_prefix)
    return app


@pytest.fixture(autouse=True)
def _fresh_rate_buckets():
    desktop_api._RATE_BUCKETS.clear()
    yield
    desktop_api._RATE_BUCKETS.clear()


@pytest.fixture(autouse=True)
def _clean_registry():
    desktop_bridge.reset_for_tests()
    yield
    desktop_bridge.reset_for_tests()


@pytest.fixture
async def api() -> AsyncClient:
    transport = ASGITransport(app=_app())
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class Account:
    """One signed-in account, with one session per client it signs in on.

    A session row is not a detail here: `_session_jti` reads what
    `get_current_user` exposes, and it exposes a `jti` only when a live
    `user_sessions` row carries it (`auth.py:262-295`). Without one every
    route that identifies a phone answers 403 `session_unknown`.
    """

    def __init__(self, user_id: str) -> None:
        self.user_id = user_id
        self.sessions: Dict[str, Tuple[str, str]] = {}

    def headers(self, label: str = "mac") -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.sessions[label][0]}",
                "X-Toup-Client": "ios"}

    def jti(self, label: str = "mac") -> str:
        return self.sessions[label][1]


async def _sign_in(account: Account, label: str) -> None:
    token = create_access_token(account.user_id)
    from app.services.auth_service import decode_platform_jwt

    jti = decode_platform_jwt(token).get("jti")
    async with async_session_maker() as db:
        db.add(UserSession(
            id=str(uuid.uuid4()), user_id=account.user_id, jti=jti,
            device_label=label, created_at=datetime.utcnow(),
            last_seen_at=datetime.utcnow(), is_revoked=False,
        ))
        await db.commit()
    account.sessions[label] = (token, jti)


async def _make_account(*, provisioned: bool = True) -> Account:
    user_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(
            id=user_id, email=f"conn-{uuid.uuid4().hex[:10]}@example.com",
            hashed_password=get_password_hash("test-password-1234"),
            name="Connections Test",
        ))
        if provisioned:
            # `pair/approve` refuses without a relay URL to hand over, and
            # `pool_service` mints this shape.
            db.add(AgentConfig(
                user_id=user_id,
                agent_url="https://agent-testtenant.agents.toup.ai",
                deploy_status="active",
            ))
        await db.commit()
    acc = Account(user_id)
    await _sign_in(acc, "mac")
    await _sign_in(acc, "phone")
    return acc


async def _enable_connections(*user_ids: str) -> None:
    """The one flag behind every QR and task route (§12)."""
    from app.services.feature_flags import set_allowlist

    async with async_session_maker() as db:
        await set_allowlist(db, "desktop_connections", list(user_ids))


@pytest.fixture
async def owner() -> Account:
    acc = await _make_account()
    await _enable_connections(acc.user_id)
    return acc


@pytest.fixture
async def stranger(owner: Account) -> Account:
    """A second, fully enabled account. Enabled on purpose: a rule that
    holds only because the other party's flag is off is not isolation."""
    acc = await _make_account()
    await _enable_connections(owner.user_id, acc.user_id)
    return acc


# ── The pairing flow, as the Mac and the phone drive it ──────────────
async def _init(api: AsyncClient, acc: Account, name: str = "A Mac") -> dict:
    r = await api.post(
        "/api/desktop/pair/init",
        json={"device_name": name, "app_version": "0.1.0",
              "os_version": "15.7", "flavor": "direct"},
        headers=acc.headers("mac"),
    )
    assert r.status_code == 200, r.text
    return r.json()


async def _qr(api: AsyncClient, device_code: str) -> dict:
    r = await api.post("/api/desktop/pair/qr", json={"device_code": device_code})
    assert r.status_code == 200, r.text
    return r.json()


async def _claim(api: AsyncClient, acc: Account, challenge: str, label: str = "phone"):
    return await api.post(
        "/api/desktop/pair/claim",
        json={"challenge": challenge, "phone_label": "iPhone"},
        headers=acc.headers(label),
    )


async def _approve(api: AsyncClient, acc: Account, label: str = "phone", **body):
    return await api.post(
        "/api/desktop/pair/approve", json=body, headers=acc.headers(label),
    )


async def _paired(api: AsyncClient, acc: Account) -> Tuple[str, dict]:
    """A Mac paired through the QR path. Returns (device_id, bundle)."""
    started = await _init(api, acc)
    qr = await _qr(api, started["device_code"])
    claimed = await _claim(api, acc, qr["challenge"])
    assert claimed.status_code == 200, claimed.text
    ok = await _approve(api, acc, challenge=qr["challenge"])
    assert ok.status_code == 200, ok.text
    poll = await api.post(
        "/api/desktop/pair/poll", json={"device_code": started["device_code"]},
    )
    assert poll.status_code == 200, poll.text
    return ok.json()["device_id"], poll.json()


async def _devices_of(user_id: str) -> List[DesktopDevice]:
    async with async_session_maker() as db:
        return list((await db.execute(
            select(DesktopDevice).where(DesktopDevice.user_id == user_id)
        )).scalars().all())


async def _pairing_of(user_code: str) -> DesktopPairing:
    async with async_session_maker() as db:
        return (await db.execute(
            select(DesktopPairing).where(DesktopPairing.user_code == user_code)
        )).scalar_one()


async def _mark_online(device_id: str) -> None:
    async with async_session_maker() as db:
        await db.execute(
            sa_update(DesktopDevice).where(DesktopDevice.id == device_id)
            .values(last_seen_at=datetime.utcnow())
        )
        await db.commit()


# ══════════════════════════════════════════════════════════════════════
# 1. Single use — one challenge is one pairing (§3.3, T3)
# ══════════════════════════════════════════════════════════════════════
#
# MUTATION WATCHED: in `pair_approve`, drop
# `.where(DesktopPairing.status == "pending")` from the guarded UPDATE.
# The second approve then lands, mints a second token and writes a second
# device row — one photographed QR, two paired Macs.
async def test_a_second_approve_of_one_challenge_mints_nothing_more(api, owner):
    started = await _init(api, owner)
    qr = await _qr(api, started["device_code"])
    assert (await _claim(api, owner, qr["challenge"])).status_code == 200

    first = await _approve(api, owner, challenge=qr["challenge"])
    assert first.status_code == 200, first.text

    second = await _approve(api, owner, challenge=qr["challenge"])
    assert second.status_code != 200, second.text
    assert second.json()["error"] in ("decided", "expired")

    devices = await _devices_of(owner.user_id)
    assert len(devices) == 1, [d.id for d in devices]


# MUTATION WATCHED: in `pair_claim`, drop
# `.where(DesktopPairing.qr_claimed_at.is_(None))`. A second phone then
# claims a challenge that is already in someone else's hands and can
# approve with it.
async def test_a_second_phone_cannot_claim_a_challenge_already_in_use(api, owner):
    started = await _init(api, owner)
    qr = await _qr(api, started["device_code"])
    assert (await _claim(api, owner, qr["challenge"])).status_code == 200

    await _sign_in(owner, "other-phone")
    again = await _claim(api, owner, qr["challenge"], label="other-phone")
    assert again.status_code == 404
    assert again.json()["error"] == "expired"

    # And that second phone cannot decide with it either (S1).
    stolen = await _approve(api, owner, label="other-phone",
                            challenge=qr["challenge"])
    assert stolen.status_code == 404
    assert not await _devices_of(owner.user_id)


# MUTATION WATCHED: in `pair_poll`, keep `approved_token` on the row
# instead of clearing it. The bundle is then collectable twice.
async def test_the_bundle_is_collectable_once(api, owner):
    started = await _init(api, owner)
    qr = await _qr(api, started["device_code"])
    await _claim(api, owner, qr["challenge"])
    await _approve(api, owner, challenge=qr["challenge"])

    first = await api.post("/api/desktop/pair/poll",
                           json={"device_code": started["device_code"]})
    assert first.status_code == 200 and first.json()["access_token"]
    second = await api.post("/api/desktop/pair/poll",
                            json={"device_code": started["device_code"]})
    assert second.status_code != 200


# ══════════════════════════════════════════════════════════════════════
# 2. Expiry — a window that has passed is not a window (§3.3)
# ══════════════════════════════════════════════════════════════════════
#
# MUTATION WATCHED: in `pair_claim`, drop
# `.where(DesktopPairing.qr_expires_at > now)`. A photographed QR then
# works for the pairing's whole 600 s instead of its own 120 s.
async def test_a_challenge_past_its_own_window_is_refused(api, owner):
    started = await _init(api, owner)
    qr = await _qr(api, started["device_code"])
    row = await _pairing_of(started["user_code"])
    async with async_session_maker() as db:
        await db.execute(
            sa_update(DesktopPairing).where(DesktopPairing.id == row.id)
            .values(qr_expires_at=datetime.utcnow() - timedelta(seconds=1))
        )
        await db.commit()

    late = await _claim(api, owner, qr["challenge"])
    assert late.status_code == 404
    assert late.json()["error"] == "expired"
    # The pairing itself is still alive: only the QR ran out.
    assert (await _pairing_of(started["user_code"])).status == "pending"


# MUTATION WATCHED: in `pair_approve`'s challenge guard, drop
# `DesktopPairing.claim_expires_at > now`. An approval then lands long
# after the person stopped looking at their phone.
async def test_an_approval_after_the_claim_window_is_refused(api, owner):
    started = await _init(api, owner)
    qr = await _qr(api, started["device_code"])
    assert (await _claim(api, owner, qr["challenge"])).status_code == 200
    row = await _pairing_of(started["user_code"])
    async with async_session_maker() as db:
        await db.execute(
            sa_update(DesktopPairing).where(DesktopPairing.id == row.id)
            .values(claim_expires_at=datetime.utcnow() - timedelta(seconds=1))
        )
        await db.commit()

    late = await _approve(api, owner, challenge=qr["challenge"])
    assert late.status_code == 404
    assert not await _devices_of(owner.user_id)


# MUTATION WATCHED: in `_gc_pairings`, expire only `pending` rows (drop the
# `_withdraw_approved` branch). An approved bundle nobody collected then
# keeps a live device token at rest for as long as the JWT lives (F-L).
async def test_an_uncollected_bundle_is_withdrawn_and_its_device_revoked(api, owner):
    started = await _init(api, owner)
    qr = await _qr(api, started["device_code"])
    await _claim(api, owner, qr["challenge"])
    ok = await _approve(api, owner, challenge=qr["challenge"])
    device_id = ok.json()["device_id"]

    async with async_session_maker() as db:
        await db.execute(
            sa_update(DesktopPairing)
            .where(DesktopPairing.user_code == started["user_code"])
            .values(decided_at=datetime.utcnow() - timedelta(
                seconds=desktop_api.APPROVED_UNCOLLECTED_S + 5))
        )
        await db.commit()
        await desktop_api._gc_pairings(db)
        await db.commit()

    row = await _pairing_of(started["user_code"])
    assert row.status == "expired" and row.approved_token is None
    dev = [d for d in await _devices_of(owner.user_id) if d.id == device_id][0]
    assert dev.revoked_at is not None and dev.token_jti is None


# ══════════════════════════════════════════════════════════════════════
# 3. Account isolation (§5.1, T4, T5)
# ══════════════════════════════════════════════════════════════════════
#
# MUTATION WATCHED: in `pair_lookup` / `pair_approve` / `pair_deny`, drop
# `DesktopPairing.user_id == user_id`. The stranger's approval then mints a
# device token on the OWNER's account — F-K, device-code phishing, live.
async def test_another_account_cannot_look_up_approve_or_deny_my_pairing(
    api, owner, stranger,
):
    started = await _init(api, owner)

    look = await api.get("/api/desktop/pair/lookup",
                         params={"user_code": started["user_code"]},
                         headers=stranger.headers("phone"))
    assert look.status_code == 404

    theirs = await _approve(api, stranger, user_code=started["user_code"])
    assert theirs.status_code == 404
    assert not await _devices_of(owner.user_id)
    assert not await _devices_of(stranger.user_id)

    denied = await api.post("/api/desktop/pair/deny",
                            json={"user_code": started["user_code"]},
                            headers=stranger.headers("phone"))
    assert denied.json().get("denied") is False
    assert (await _pairing_of(started["user_code"])).status == "pending"

    # The owner's own approval still works, so the guard is not simply
    # refusing everyone.
    mine = await _approve(api, owner, user_code=started["user_code"])
    assert mine.status_code == 200, mine.text


# MUTATION WATCHED: in `pair_claim`, return `gone` instead of
# `other_account` for a mismatched `user_id` — i.e. let the claim burn the
# challenge. Anyone signed in to any account could then spoil every QR
# they can photograph.
async def test_another_accounts_scan_is_refused_and_does_not_burn_the_challenge(
    api, owner, stranger,
):
    started = await _init(api, owner)
    qr = await _qr(api, started["device_code"])

    theirs = await _claim(api, stranger, qr["challenge"])
    assert theirs.status_code == 403
    assert theirs.json()["error"] == "other_account"

    mine = await _claim(api, owner, qr["challenge"])
    assert mine.status_code == 200, mine.text
    assert mine.json()["user_code"] == started["user_code"]


# MUTATION WATCHED: in `submit_task`, drop
# `DesktopDevice.user_id == user_id` from the device lookup; and in
# `_revoke_devices`, drop the same filter. A stranger can then send work
# to — or disconnect — a Mac that is not theirs.
async def test_another_account_cannot_submit_to_or_revoke_my_mac(
    api, owner, stranger,
):
    device_id, _ = await _paired(api, owner)
    await _mark_online(device_id)

    sent = await api.post(
        f"/api/desktop/devices/{device_id}/tasks",
        json={"text": "run the tests", "client_request_id": str(uuid.uuid4())},
        headers=stranger.headers("phone"),
    )
    assert sent.status_code == 404
    assert sent.json()["error"] == "not_found"
    async with async_session_maker() as db:
        assert not list((await db.execute(select(DesktopTask))).scalars().all())

    killed = await api.post("/api/desktop/devices/revoke",
                            json={"device_id": device_id},
                            headers=stranger.headers("phone"))
    assert killed.status_code in (200, 404)
    dev = (await _devices_of(owner.user_id))[0]
    assert dev.revoked_at is None, "a stranger disconnected someone else's Mac"


# MUTATION WATCHED: in `_owned_task` (and `list_tasks`), drop
# `DesktopTask.user_id == user_id`. Another account then reads, and
# cancels, work sent from a phone that is not theirs.
async def test_another_account_cannot_read_or_cancel_my_tasks(api, owner, stranger):
    device_id, _ = await _paired(api, owner)
    task_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(DesktopTask(
            id=task_id, user_id=owner.user_id, device_id=device_id,
            title="run the tests", status="queued", turn_state="queued",
            client_request_id=str(uuid.uuid4()),
            created_at=datetime.utcnow(), updated_at=datetime.utcnow(),
        ))
        await db.commit()

    theirs = await api.get(f"/api/desktop/tasks/{task_id}",
                           headers=stranger.headers("phone"))
    assert theirs.status_code == 404

    listed = await api.get("/api/desktop/tasks", headers=stranger.headers("phone"))
    assert listed.status_code == 200
    assert listed.json()["tasks"] == []

    cancelled = await api.post(f"/api/desktop/tasks/{task_id}/cancel",
                               headers=stranger.headers("phone"))
    assert cancelled.status_code == 404
    async with async_session_maker() as db:
        assert (await db.get(DesktopTask, task_id)).status == "queued"

    # The owner sees their own task, so the filter is not simply hiding it
    # from everyone.
    mine = await api.get(f"/api/desktop/tasks/{task_id}", headers=owner.headers("phone"))
    assert mine.status_code == 200 and mine.json()["id"] == task_id


# ══════════════════════════════════════════════════════════════════════
# 4. Revocation — a disconnected Mac is dead, not labelled (§6, T7)
# ══════════════════════════════════════════════════════════════════════
#
# MUTATION WATCHED: in `_revoke_devices`, stop clearing `token_jti` (set
# only `revoked_at`). `_jti_accepted` then still matches and the revoked
# Mac's own credential keeps working on the socket.
async def test_revoking_a_mac_kills_its_credential_and_its_queued_work(api, owner):
    device_id, bundle = await _paired(api, owner)
    await _mark_online(device_id)
    token = bundle["access_token"]

    alive = await desktop_api._verify_device_token(token)
    assert alive["ok"] is True, alive

    task_id, action_id = str(uuid.uuid4()), str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(DesktopTask(
            id=task_id, user_id=owner.user_id, device_id=device_id,
            title="run the tests", status="queued", turn_state="queued",
            client_request_id=str(uuid.uuid4()),
            created_at=datetime.utcnow(), updated_at=datetime.utcnow(),
        ))
        db.add(DesktopPendingAction(
            id=action_id, user_id=owner.user_id, device_id=device_id,
            remote_task_id=task_id, task_id=str(uuid.uuid4()),
            tool_name="desktop__exec_run",
            payload_json=json.dumps({"command": "make test"}),
            reason="run the tests", status="pending", channel="app",
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(minutes=15),
        ))
        await db.commit()

    killed = await api.post("/api/desktop/devices/revoke",
                            json={"device_id": device_id},
                            headers=owner.headers("phone"))
    assert killed.status_code == 200, killed.text

    dead = await desktop_api._verify_device_token(token)
    assert dead["ok"] is False and dead["reason"] == "revoked"

    async with async_session_maker() as db:
        dev = await db.get(DesktopDevice, device_id)
        assert dev.revoked_at is not None and dev.token_jti is None
        assert (await db.get(DesktopTask, task_id)).status == "cancelled"
        card = await db.get(DesktopPendingAction, action_id)
        assert card.status == "rejected"


# MUTATION WATCHED: in `_revoke_devices`, ignore the `device_ids` filter
# (revoke every row for the user). Disconnecting one Mac then silently
# disconnects the other — M1.
async def test_revoking_one_mac_leaves_the_other_alone(api, owner):
    first, _ = await _paired(api, owner)
    second, _ = await _paired(api, owner)
    assert first != second

    killed = await api.post("/api/desktop/devices/revoke",
                            json={"device_id": first},
                            headers=owner.headers("phone"))
    assert killed.status_code == 200, killed.text

    by_id = {d.id: d for d in await _devices_of(owner.user_id)}
    assert by_id[first].revoked_at is not None
    assert by_id[second].revoked_at is None
    assert by_id[second].token_jti is not None


# MUTATION WATCHED: in `revoke_device`, drop the `device_required` branch
# so an account bearer with no body revokes everything again.
async def test_an_account_bearer_with_no_device_revokes_nothing(api, owner):
    device_id, _ = await _paired(api, owner)

    vague = await api.post("/api/desktop/devices/revoke", json={},
                           headers=owner.headers("phone"))
    assert vague.status_code == 400
    assert vague.json()["error"] == "device_required"
    assert (await _devices_of(owner.user_id))[0].revoked_at is None


# ══════════════════════════════════════════════════════════════════════
# 5. Least privilege — only what the Mac's owner switched on (§9)
# ══════════════════════════════════════════════════════════════════════
def _ctx(user_id: str) -> SkillContext:
    return SkillContext(user_id=user_id, session_id="sess-1")


class _Socket:
    def __init__(self) -> None:
        self.sent: List[Dict[str, Any]] = []

    async def send_text(self, text: str) -> None:
        self.sent.append(json.loads(text))

    async def close(self, code: int = 1000, reason: str = "") -> None:
        pass


async def _mac_advertising(user_id: str, device_id: str, tools: List[str]):
    """A connected Mac whose advertised list is what its owner allowed."""
    ep = await desktop_bridge.register(
        user_id, _Socket(), device_id=device_id, token_jti="jti-1",
    )
    desktop_bridge.handle_tools(ep, {"tools": [{"name": t} for t in tools]})
    return ep


class _Pin:
    """The DESKTOP_TARGET a phone task's turn runs under."""

    def __init__(self, device_id: str, name: str = "Studio Mac") -> None:
        self.target = desktop_bridge.DesktopTarget(
            task_id=str(uuid.uuid4()), device_id=device_id, device_name=name,
        )

    def __enter__(self):
        self._token = desktop_bridge.DESKTOP_TARGET.set(self.target)
        return self.target

    def __exit__(self, *exc):
        desktop_bridge.DESKTOP_TARGET.reset(self._token)


# MUTATION WATCHED: in `skill._refuse_ungranted`, return None before
# reading the advertised list. A phone task can then ask for a family the
# owner never switched on, and the request reaches the staging table as a
# card the person is invited to approve.
async def test_a_pinned_task_cannot_use_a_family_the_mac_never_advertised(
    monkeypatch,
):
    user_id = str(uuid.uuid4())
    device_id = str(uuid.uuid4())
    await _mac_advertising(user_id, device_id, ["desktop__fs_read"])

    staged: List[str] = []

    async def _never(*a, **kw):
        staged.append("staged")
        return {"action_id": "x"}

    monkeypatch.setattr(
        "app.agent.skills.builtins.desktop.skill._post_stage", _never,
    )

    with _Pin(device_id):
        out = await DesktopSkill().execute_tool(
            "desktop__exec_run",
            {"command": "make test", "cwd": "/tmp", "reason": "run the tests"},
            _ctx(user_id),
        )
    assert str(out).startswith("REFUSED:"), out
    assert not staged, "a card was staged for a capability the Mac does not offer"


# The other direction of the same rule: what the Mac DOES advertise is not
# refused. Without this, "refuse everything" would pass the test above.
async def test_a_pinned_task_may_use_what_the_mac_does_advertise(monkeypatch):
    user_id = str(uuid.uuid4())
    device_id = str(uuid.uuid4())
    await _mac_advertising(user_id, device_id, ["desktop__fs_read"])

    seen: Dict[str, Any] = {}

    async def _dispatch(uid, action, params, timeout_s=30.0, **kw):
        seen.update({"uid": uid, "action": action, "device_id": kw.get("device_id")})
        return {"ok": True, "data": {"text": "hello"}, "summary": "read it"}

    monkeypatch.setattr(desktop_bridge, "dispatch", _dispatch)

    with _Pin(device_id):
        out = await DesktopSkill().execute_tool(
            "desktop__fs_read", {"path": "/tmp/x"}, _ctx(user_id),
        )
    assert not str(out).startswith(("REFUSED:", "ERROR:")), out
    assert seen["action"] == "desktop__fs_read"
    # M3: and it went to the pinned Mac, not to "the newest socket".
    assert seen["device_id"] == device_id


# MUTATION WATCHED: in `skill._require_device`, answer the pinned branch
# with `is_connected(user_id)` instead of `is_device_connected(...)`. A
# task sent to a Mac that has gone offline then runs on whichever other
# Mac happens to be online — M3.
async def test_a_pinned_task_refuses_when_its_own_mac_is_gone(monkeypatch):
    user_id = str(uuid.uuid4())
    gone = str(uuid.uuid4())
    other = str(uuid.uuid4())
    await _mac_advertising(user_id, other, ["desktop__fs_read"])

    async def _never(*a, **kw):
        raise AssertionError("dispatched to another Mac")

    monkeypatch.setattr(desktop_bridge, "dispatch", _never)

    pin = _Pin(gone, name="Studio Mac")
    with pin as target:
        out = await DesktopSkill().execute_tool(
            "desktop__fs_read", {"path": "/tmp/x"}, _ctx(user_id),
        )
    assert str(out).startswith("ERROR:") and "Studio Mac" in str(out)
    # The turn runner reads this after the turn to set the task's status.
    assert target.flags.get("mac_unavailable") is True
