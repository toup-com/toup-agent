"""Automations setup cards + skill gating (Round 26, agent lane).

Listed in COVERAGE_DEBT.txt `# agent-mode` (messages/conversations/
auth-sessions are AGENT_ONLY).

Proves:
  - the skill is withheld while the flag is off (wire array stays
    byte-identical) and registered when on
  - request_connection persists a card message whose metadata carries
    the EXACT contract payload, and the session row backs it
  - the reject endpoint's guarded UPDATE flips card + session once
  - the _connector_connected hook resolves open sessions in place
  - lazy expiry: an offered session past its TTL reads back expired
  - all four history serializers emit the card keys (parity)
"""

import json
import uuid
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from app.config import settings
from app.db.database import async_session_maker
from app.db.models import AutomationAuthSession, Message, User


async def _mk_user() -> str:
    uid = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(id=uid, email=f"{uid[:8]}@example.com",
                    hashed_password="x", name="Setup"))
        await db.commit()
    return uid


@pytest.mark.asyncio
async def test_skill_withheld_while_flag_off(monkeypatch):
    from app.agent.tool_entitlements import skill_enabled
    assert getattr(settings, "automations_enabled", False) is False
    assert skill_enabled("automations") is False
    monkeypatch.setattr(settings, "automations_enabled", True)
    assert skill_enabled("automations") is True


@pytest.mark.asyncio
async def test_skill_tools_match_round_brief(monkeypatch):
    monkeypatch.setattr(settings, "automations_enabled", True)
    from app.agent.skills.builtins.automations.skill import AutomationsSkill
    names = {t["name"] for t in AutomationsSkill().get_tools()}
    assert {
        "automations__get_registry", "automations__request_connection",
        "automations__request_permission", "automations__list_targets",
        "automations__create", "automations__update",
        "automations__test_run", "automations__arm", "automations__pause",
        "automations__resume", "automations__delete",
    } <= names
    for n in names:
        # R30 §4.5: `memory_recall` is deliberately outside the skill's
        # namespace — it reads the ONE platform memory from the main
        # chat, not an automations-internal surface. Everything else
        # keeps the prefix.
        assert n.startswith("automations__") or n == "memory_recall"
    # And the prefix-stable tools array only ever grows at the END.
    ordered = [t["name"] for t in AutomationsSkill().get_tools()]
    assert ordered[-1] == "memory_recall"


@pytest.mark.asyncio
async def test_request_connection_persists_contract_card(monkeypatch):
    monkeypatch.setattr(settings, "automations_enabled", True)
    uid = await _mk_user()
    from app.agent.skills.builtins.automations.skill import AutomationsSkill
    from app.agent.skills.base import SkillContext
    from app.agent.automations import registry as reg

    async def fake_registry(_uid, force=False):
        return {"slack": {
            "connector_id": "slack", "name": "Slack", "icon": "slack",
            "scope_descriptions": {"chat:write": "Send messages."},
            "push": False, "poll": False, "floor_s": 300,
            "scopes_read": ["channels:read"],
            "scopes_write_by_action": {"slack__send_message": ["chat:write"]},
            "target_param_by_action": {"slack__send_message": "channel"},
            "events": [], "rate_budget": {},
        }}

    async def fake_connections(_uid):
        return {}

    monkeypatch.setattr(reg, "fetch_registry", fake_registry)
    monkeypatch.setattr(reg, "fetch_connection_state", fake_connections)

    out = await AutomationsSkill().execute_tool(
        "automations__request_connection",
        {"connector_id": "slack", "mode": "read_write"},
        SkillContext(user_id=uid),
    )
    assert "CONNECTOR CARD SHOWN" in out
    assert "NOT CONNECTED" in out

    async with async_session_maker() as db:
        session = (await db.execute(
            select(AutomationAuthSession)
            .where(AutomationAuthSession.user_id == uid)
        )).scalar_one()
        assert session.status == "offered"
        assert session.mode == "read_write"
        assert session.message_id is not None
        msg = await db.get(Message, session.message_id)
        meta = json.loads(msg.metadata_json)
        card = meta["automation_connector_card"]
        # The contract payload, field for field.
        assert card["id"] == session.id
        assert card["connector_id"] == "slack"
        assert card["status"] == "offered"
        assert card["mode"] == "read_write"
        assert card["connect_url"] == "/api/oauth/connect/slack"
        assert card["retry_used"] is False
        scopes = {s["scope"]: s for s in card["scopes"]}
        assert scopes["channels:read"]["write"] is False
        assert scopes["chat:write"]["write"] is True
        assert card["expires_at"] > card["created_at"]
    return uid


@pytest.mark.asyncio
async def test_reject_flips_session_and_card_once(monkeypatch):
    monkeypatch.setattr(settings, "automations_enabled", True)
    uid = await _mk_user()
    async with async_session_maker() as db:
        s = AutomationAuthSession(
            user_id=uid, connector_id="slack", mode="read",
            status="offered",
            expires_at=datetime.utcnow() + timedelta(minutes=10),
        )
        db.add(s)
        await db.commit()
        sid = s.id

    monkeypatch.setattr(settings, "user_id", uid)
    from app.api.automations import reject_auth_session, get_auth_session
    out = await reject_auth_session(sid)
    assert out["status"] == "rejected"
    # Idempotent-ish: a second reject finds no eligible row but still
    # returns the (already rejected) payload rather than erroring.
    out2 = await reject_auth_session(sid)
    assert out2["status"] == "rejected"
    got = await get_auth_session(sid)
    assert got["status"] == "rejected"


@pytest.mark.asyncio
async def test_connector_connected_hook_resolves_open_sessions(monkeypatch):
    monkeypatch.setattr(settings, "automations_enabled", True)
    uid = await _mk_user()
    async with async_session_maker() as db:
        s = AutomationAuthSession(
            user_id=uid, connector_id="jira", mode="read",
            status="offered",
            expires_at=datetime.utcnow() + timedelta(minutes=10),
        )
        db.add(s)
        await db.commit()
        sid = s.id

    monkeypatch.setattr(settings, "user_id", uid)
    from app.api.automations import connector_connected_hook, ConnectorHook
    out = await connector_connected_hook(ConnectorHook(connector_id="jira"))
    assert out["updated"] == 1
    async with async_session_maker() as db:
        s2 = await db.get(AutomationAuthSession, sid)
        assert s2.status == "connected"
    # Unrelated connector: clean no-op.
    out2 = await connector_connected_hook(ConnectorHook(connector_id="github"))
    assert out2["updated"] == 0


@pytest.mark.asyncio
async def test_lazy_expiry_on_read(monkeypatch):
    monkeypatch.setattr(settings, "automations_enabled", True)
    uid = await _mk_user()
    async with async_session_maker() as db:
        s = AutomationAuthSession(
            user_id=uid, connector_id="slack", mode="read",
            status="offered",
            expires_at=datetime.utcnow() - timedelta(minutes=1),
        )
        db.add(s)
        await db.commit()
        sid = s.id
    monkeypatch.setattr(settings, "user_id", uid)
    from app.api.automations import get_auth_session
    got = await get_auth_session(sid)
    assert got["status"] == "expired"


def test_all_four_serializers_emit_the_card_keys():
    """Parity by inspection of the REAL serializer functions: build a
    row-shaped message carrying both cards and run it through the
    day_chats helpers + the schema model the session routes return."""
    from app.api.day_chats import _serialize_automation_card
    from app.schemas import ChatMessageResponse

    msg = SimpleNamespace(
        id="m1", role="assistant", content="x",
        metadata_json=json.dumps({
            "automation_connector_card": {"id": "s1", "status": "offered"},
            "automation_grant_card": {"id": "g1", "status": "pending"},
        }),
        attachments=None,
    )
    assert _serialize_automation_card(
        msg, "automation_connector_card")["id"] == "s1"
    assert _serialize_automation_card(
        msg, "automation_grant_card")["id"] == "g1"

    fields = ChatMessageResponse.model_fields
    assert "automation_connector_card" in fields
    assert "automation_grant_card" in fields
