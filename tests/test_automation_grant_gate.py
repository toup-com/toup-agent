"""Dispatcher grant gate (Round 26) — the write-side fail-closed rails.

PLATFORM lane: automation_grants is PLATFORM_ONLY, and the gate lives
in the connector dispatcher next to the tokens.

The core claims proven here, each against the real `execute()` entry
point with the real registry (no kinder cage: the same gate ladder
production runs — a call that fails here fails in prod for the same
reason):

  1. mutating + channel=automation + no grant       → refused, provider untouched
  2. grant for a different tool                      → refused
  3. grant pinned to a different target              → refused
  4. revoked grant                                   → refused
  5. cadence budget exhausted                        → refused (retryable)
  6. approved grant + matching target                → passes the gate
     (fails LATER at the vault with reauth_required — proving the
     provider path was reached, not short-circuited)
  7. reads on the automation channel need no grant   → same vault outcome
  8. grant_id on a non-automation channel            → refused
"""

import json
import uuid
from datetime import datetime, timedelta

import pytest

from app.db.database import async_session_maker
from app.db.models import AutomationGrant, User
from app.connectors.base import ConnectorReauthRequired, ConnectorToolError
from app.services import connector_dispatcher as dispatcher
from app.services.connector_registry import get_registry


@pytest.fixture(autouse=True)
def _loaded_registry():
    reg = get_registry()
    if not reg.automation_registry():
        reg.load_all(include_experimental=True)
    yield


async def _mk_user() -> str:
    uid = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(
            id=uid, email=f"{uid[:8]}@example.com",
            hashed_password="x", name="Grant Gate",
        ))
        await db.commit()
    return uid


async def _mk_grant(user_id: str, **over) -> str:
    row = AutomationGrant(
        user_id=user_id,
        automation_id=over.get("automation_id"),
        connector_id=over.get("connector_id", "slack"),
        tool_name=over.get("tool_name", "slack__send_message"),
        target_json=over.get("target_json", json.dumps(
            {"kind": "channel", "id": "C123", "label": "#eng"},
        )),
        cadence_json=over.get("cadence_json"),
        mode=over.get("mode", "auto"),
        summary="may post to #eng",
        status=over.get("status", "approved"),
        expires_at=datetime.utcnow() + timedelta(hours=1),
        uses_day_key=over.get("uses_day_key"),
        uses_today=over.get("uses_today", 0),
        uses_hour_key=over.get("uses_hour_key"),
        uses_this_hour=over.get("uses_this_hour", 0),
    )
    async with async_session_maker() as db:
        db.add(row)
        await db.commit()
        return row.id


async def _call(user_id, *, tool="slack__send_message", connector="slack",
                tool_input=None, channel="automation", grant_id=None):
    async with async_session_maker() as db:
        return await dispatcher.execute(
            db=db, user_id=user_id, connector_id=connector,
            tool_name=tool,
            tool_input=tool_input or {"channel": "C123", "text": "hi"},
            channel=channel, grant_id=grant_id,
        )


@pytest.mark.asyncio
async def test_mutating_without_grant_fails_closed():
    uid = await _mk_user()
    res = await _call(uid)
    assert isinstance(res, ConnectorToolError)
    assert "fail closed" in res.message or "permission" in res.message


@pytest.mark.asyncio
async def test_grant_for_different_tool_is_refused():
    uid = await _mk_user()
    gid = await _mk_grant(uid, tool_name="slack__send_message")
    res = await _call(uid, tool="jira__add_comment", connector="jira",
                      tool_input={"issue_key": "ENG-1", "body": "x"},
                      grant_id=gid)
    assert isinstance(res, ConnectorToolError)
    assert "different action" in res.message


@pytest.mark.asyncio
async def test_grant_target_mismatch_is_refused():
    uid = await _mk_user()
    gid = await _mk_grant(uid)
    res = await _call(uid, tool_input={"channel": "C999", "text": "hi"},
                      grant_id=gid)
    assert isinstance(res, ConnectorToolError)
    assert "may only write to" in res.message


@pytest.mark.asyncio
async def test_revoked_grant_is_refused():
    uid = await _mk_user()
    gid = await _mk_grant(uid, status="revoked")
    res = await _call(uid, grant_id=gid)
    assert isinstance(res, ConnectorToolError)
    assert "revoked" in res.message


@pytest.mark.asyncio
async def test_cadence_exhausted_is_refused_and_retryable():
    uid = await _mk_user()
    day_key = datetime.utcnow().strftime("%Y-%m-%d")
    gid = await _mk_grant(
        uid, cadence_json=json.dumps({"per_day": 2}),
        uses_day_key=day_key, uses_today=2,
    )
    res = await _call(uid, grant_id=gid)
    assert isinstance(res, ConnectorToolError)
    assert res.retryable is True
    assert "daily budget" in res.message


@pytest.mark.asyncio
async def test_valid_grant_passes_gate_and_reaches_vault():
    uid = await _mk_user()
    gid = await _mk_grant(uid)
    res = await _call(uid, grant_id=gid)
    # No slack identity exists for this fresh user, so the call that
    # PASSED the grant gate dies at the vault — the reauth answer is
    # the proof the gate let it through.
    assert isinstance(res, ConnectorReauthRequired)
    # And the cadence counter was charged (conservative accounting).
    async with async_session_maker() as db:
        row = await db.get(AutomationGrant, gid)
        assert row.uses_today == 1
        assert row.uses_this_hour == 1


@pytest.mark.asyncio
async def test_reads_need_no_grant_on_automation_channel():
    uid = await _mk_user()
    res = await _call(
        uid, tool="slack__list_channels", tool_input={}, grant_id=None,
    )
    assert isinstance(res, ConnectorReauthRequired)


@pytest.mark.asyncio
async def test_grant_id_on_web_channel_is_refused():
    uid = await _mk_user()
    gid = await _mk_grant(uid)
    res = await _call(uid, channel="web", grant_id=gid)
    assert isinstance(res, ConnectorToolError)
    assert "only valid on the automation channel" in res.message
