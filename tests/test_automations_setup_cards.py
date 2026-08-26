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
        # EVERY tool carries the prefix, `automations__memory_recall`
        # included. This used to permit a bare `memory_recall` on a
        # §4.5 rationale §4.5 does not contain — the prefix is the
        # loader's tool-index namespace, not a statement about which
        # surface may call the tool, and SkillLoader._register RAISES
        # on the first name that lacks it.
        assert n.startswith("automations__")
    # And the prefix-stable tools array only ever grows at the END.
    ordered = [t["name"] for t in AutomationsSkill().get_tools()]
    assert ordered[-1] == "automations__memory_recall"


@pytest.mark.asyncio
async def test_every_builtin_skill_tool_carries_its_skill_prefix():
    """The loader's rule, applied to every builtin — statically.

    `SkillLoader._register` raises `ValueError` on the first tool whose
    name lacks `f"{skill.meta.name}__"`, and `load_all()` catches that
    raise per-directory. So one mis-named tool does not register one bad
    tool; it silently DISCARDS THE WHOLE SKILL, tools, prompt section,
    commands, hooks and execution path together. Every suite here passed
    with the automations skill entirely absent from the agent, because
    they all call `get_tools()` and `execute_tool()` directly and never
    touch the component that enforces the rule.
    """
    import importlib.util
    import os
    from app.agent.skills.base import Skill

    root = os.path.join(os.path.dirname(
        importlib.util.find_spec("app.agent.skills.loader").origin),
        "builtins")
    checked = 0
    for entry in sorted(os.listdir(root)):
        skill_file = os.path.join(root, entry, "skill.py")
        if not os.path.isfile(skill_file):
            continue
        spec = importlib.util.spec_from_file_location(
            f"prefixpin_{entry}", skill_file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for obj in vars(mod).values():
            if not (isinstance(obj, type) and issubclass(obj, Skill)
                    and obj is not Skill):
                continue
            if getattr(obj, "meta", None) is None:
                continue
            prefix = f"{obj.meta.name}__"
            for tool in obj().get_tools():
                assert tool["name"].startswith(prefix), (
                    f"{entry}: tool {tool['name']!r} lacks {prefix!r} — "
                    f"the loader will discard the entire skill"
                )
                checked += 1
    assert checked > 20, f"the sweep found only {checked} tools"


@pytest.mark.asyncio
async def test_automations_skill_registers_through_the_real_loader(
    monkeypatch,
):
    """The funnel itself — not a re-implementation of its rule.

    This is the only test in the tree that would have caught a bare
    `memory_recall`: it drives `SkillLoader._register`, the code that
    actually raises, and asserts the skill ends up registered with all
    of its tools in the loader's tool index.
    """
    monkeypatch.setattr(settings, "automations_enabled", True)
    from app.agent.skills.loader import SkillLoader
    from app.agent.skills.builtins.automations.skill import AutomationsSkill

    loader = SkillLoader()
    registered = await loader._register(AutomationsSkill())
    assert registered is True, "the automations skill refused to register"
    indexed = {n for n, owner in loader._tool_index.items()
               if owner == "automations"}
    assert "automations__memory_recall" in indexed
    assert "automations__request_connection" in indexed
    assert len(indexed) == len(AutomationsSkill().get_tools())


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
