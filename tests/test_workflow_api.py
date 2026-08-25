# agent-mode: automations/threads/permissions are AGENT_ONLY tables.
"""The workflow API — CONTRACTS-R30 §4.4 proof (R30-A).

GET shape from one source, the permission registry (rails that can
never be allowed, read-only defaults, the last-read guard, the
needs_consent refusal), schedule presets with the post-commit arm
nudge preserved, rules CRUD, the composer's applied/needs contract
with undo, and the EDITED thread record on every write.
"""

import json
import uuid

import pytest
from sqlalchemy import select

from app.db.database import async_session_maker
from app.db.models import (
    Automation, AutomationAccountPermission, AutomationTurn, User,
)
from app.agent.automations import compiler
from app.agent.automations.spec import validate_spec

from tests.test_run_ledger_v3 import (  # noqa: F401 — shared fixtures
    REGISTRY_V2, _mk_user, _v2_spec, _mk_automation_v2,
)


@pytest.fixture(autouse=True)
def _offline_platform(monkeypatch):
    """The workflow reads connection state + capability over the
    platform RPC — stub both so tests run without a platform."""
    async def _conn_state(user_id):
        return {
            "jira": {"connector_id": "jira", "connected": True,
                     "status": "active", "scopes": ["r"],
                     "account": "TP project"},
            "slack": {"connector_id": "slack", "connected": True,
                      "status": "active", "scopes": ["w"],
                      "account": "toup.ai workspace"},
            "github": {"connector_id": "github", "connected": True,
                       "status": "reauth_required", "scopes": [],
                       "account": "toup-ai/platform"},
        }

    async def _registry(user_id, force=False):
        return REGISTRY_V2

    async def _templates(user_id):
        return []

    monkeypatch.setattr(
        "app.agent.automations.registry.fetch_connection_state",
        _conn_state,
    )
    monkeypatch.setattr(
        "app.agent.automations.registry.fetch_registry", _registry,
    )
    monkeypatch.setattr(
        "app.agent.automations.registry.fetch_templates", _templates,
    )

    async def _grant(user_id, grant_id):
        return {"id": grant_id, "status": "approved",
                "connector_id": "slack",
                "tool_name": "slack__send_message",
                "target": {"kind": "channel", "id": "C-PIN",
                           "label": "#platform"},
                "mode": "auto"}

    async def _bind(user_id, *, grant_id, automation_id):
        return {"id": grant_id, "automation_id": automation_id,
                "status": "approved"}

    monkeypatch.setattr(
        "app.agent.automations.registry.fetch_grant", _grant,
    )
    try:
        monkeypatch.setattr(
            "app.agent.automations.registry.bind_grant", _bind,
        )
    except AttributeError:
        pass


async def _edited_notes(automation_id: str) -> int:
    from app.agent.automations import ledger
    async with async_session_maker() as db:
        thread = await ledger.thread_for(db, automation_id)
        if thread is None:
            return 0
        rows = (await db.execute(
            select(AutomationTurn)
            .where(AutomationTurn.thread_id == thread.id)
            .where(AutomationTurn.kind == "note")
        )).scalars().all()
        return sum(
            1 for r in rows
            if json.loads(r.payload_json).get("stamp") == "edited"
        )


@pytest.mark.asyncio
async def test_workflow_get_shape_from_one_source():
    from app.agent.automations.workflow import workflow_payload
    uid = await _mk_user()
    vspec = _v2_spec()
    a = await _mk_automation_v2(uid, vspec)
    async with async_session_maker() as db:
        a2 = await db.get(Automation, a.id)
        wf = await workflow_payload(db, automation=a2, user_id=uid)
    assert {p["id"] for p in wf["schedule"]["presets"]} >= {
        "weekdays-8", "weekdays-730", "daily-8", "weekdays-9",
    }
    assert wf["schedule"]["preset_id"] == "weekdays-8"
    assert wf["schedule"]["label"] == "Weekdays 8:00"
    assert wf["schedule"]["sentence"] == "Weekdays at 8:00"
    accounts = {x["account_id"]: x for x in wf["accounts"]}
    assert set(accounts) == {"jira", "slack"}
    # Reads default to CAN; the granted write (grant_id in the spec)
    # defaults to CAN; rails are cant/kind=rail.
    slack = accounts["slack"]
    assert any(p["id"].startswith("slack.read") for p in slack["can"])
    rails = [p for p in slack["cant"] if p["kind"] == "rail"]
    assert rails, "slack must carry at least one rail row"
    assert wf["counts"]["briefs_per_run"] == 1
    assert wf["counts"]["items_per_fire"] == 1
    assert wf["output"]["lines"][-1]["title"] == "It tells you when it fails"
    assert wf["steps"], "human steps derived from the spec"


@pytest.mark.asyncio
async def test_permission_saves_and_three_refusals():
    from app.agent.automations import permissions
    from app.agent.automations.workflow import (
        WorkflowError, save_permissions,
    )
    uid = await _mk_user()
    vspec = _v2_spec()
    a = await _mk_automation_v2(uid, vspec)
    async with async_session_maker() as db:
        a2 = await db.get(Automation, a.id)
        cat = permissions.catalog_for("slack")
        read_ids = [p["id"] for p in cat["reads"]]
        write_ids = [p["id"] for p in cat["writes"]]
        rail_ids = [p["id"] for p in cat["rails"]]

        # hard_rail: a rail id in `can` can never commit.
        with pytest.raises(WorkflowError) as e1:
            await save_permissions(
                db, automation=a2, user_id=uid, account_id="slack",
                can_ids=read_ids + rail_ids[:1], cant_ids=[],
            )
        assert e1.value.code == "hard_rail"

        # last_read: removing the last read keeps the row put.
        with pytest.raises(WorkflowError) as e2:
            await save_permissions(
                db, automation=a2, user_id=uid, account_id="slack",
                can_ids=write_ids, cant_ids=read_ids,
            )
        assert e2.value.code == "last_read"

        # A write REMOVAL commits and repaints from the one source.
        result = await save_permissions(
            db, automation=a2, user_id=uid, account_id="slack",
            can_ids=read_ids, cant_ids=write_ids,
        )
        assert all(p["id"] not in write_ids for p in result["can"])
        resolved = await permissions.resolve(
            db, automation=a2, account_id="slack",
        )
        assert resolved == result
        row = (await db.execute(
            select(AutomationAccountPermission).where(
                AutomationAccountPermission.automation_id == a2.id,
                AutomationAccountPermission.account_id == "slack",
            )
        )).scalar_one()
        assert json.loads(row.cant_json) == sorted(write_ids)

    # needs_consent: allowing a write on an account with NO grant
    # behind it refuses (jira has no grant in this spec).
    async with async_session_maker() as db:
        a3 = await db.get(Automation, a.id)
        jira_cat = permissions.catalog_for("jira")
        with pytest.raises(permissions.PermissionError409) as e3:
            await permissions.save(
                db, automation=a3, account_id="jira",
                can_ids=[p["id"] for p in jira_cat["reads"]]
                + [p["id"] for p in jira_cat["writes"]],
                cant_ids=[], has_write_grant=False,
            )
        assert e3.value.code == "needs_consent"

    assert await _edited_notes(a.id) >= 1


@pytest.mark.asyncio
async def test_schedule_preset_rearms_with_post_commit_nudge(monkeypatch):
    """The R28-D ordering invariant survives the preset path: the
    runner nudge happens AFTER the commit (a pre-commit nudge reads
    the OLD row and unschedules what it just armed)."""
    from app.agent.automations.workflow import set_schedule_preset
    uid = await _mk_user()
    vspec = _v2_spec()
    a = await _mk_automation_v2(uid, vspec)

    order: list[str] = []
    real_nudge = compiler.nudge_routines

    async def _spy_nudge(routine_ids):
        order.append("nudge")
        return await real_nudge(routine_ids)

    monkeypatch.setattr(compiler, "nudge_routines", _spy_nudge)

    async with async_session_maker() as db:
        # Arm first so the preset write re-arms.
        from app.agent.automations.service import arm_automation
        await arm_automation(db, automation_id=a.id, user_id=uid)
        order.append("armed")
        a2 = await db.get(Automation, a.id)
        result = await set_schedule_preset(
            db, automation=a2, user_id=uid, preset_id="weekdays-730",
        )
    assert result["schedule"]["preset_id"] == "weekdays-730"
    assert result["sentence"].startswith("Moved it to")
    async with async_session_maker() as db:
        a3 = await db.get(Automation, a.id)
        raw = json.loads(a3.spec_json)
        scheds = [s.get("schedule") for s in raw["trigger"]["sources"]
                  if s.get("schedule")]
        assert scheds == [{"cron_local": "30 7 * * 1-5"}]
    # The arm nudged once; the preset write re-armed and nudged AGAIN
    # after its own commit (the post-commit ordering itself is pinned by
    # test_arm_nudges_the_runner_after_the_commit — mutation-tested).
    assert order[-1] == "nudge" and order.count("nudge") == 2
    assert await _edited_notes(a.id) >= 1


@pytest.mark.asyncio
async def test_rules_crud_and_injection_boundary():
    from app.agent.automations.workflow import (
        add_rule, delete_rule, rules_list, update_rule,
    )
    uid = await _mk_user()
    vspec = _v2_spec()
    a = await _mk_automation_v2(uid, vspec)
    async with async_session_maker() as db:
        a2 = await db.get(Automation, a.id)
        r = await add_rule(db, automation=a2,
                           text="Never post in a channel — DM me instead.")
        assert r["sentence"].startswith("Added a rule")
        rid = r["rule"]["id"]
        await update_rule(db, automation=a2, rule_id=rid,
                          text="Leave anything finance owns alone.")
        assert rules_list(a2)[0]["text"].startswith("Leave anything")
        # Rules are NOT memory items — nothing lands in memory_facts.
        from app.db.models import MemoryFact
        facts = (await db.execute(
            select(MemoryFact).where(MemoryFact.user_id == uid)
        )).scalars().all()
        assert facts == []
        await delete_rule(db, automation=a2, rule_id=rid)
        assert rules_list(a2) == []
    assert await _edited_notes(a.id) == 3


@pytest.mark.asyncio
async def test_composer_applies_rule_with_undo_and_record(monkeypatch):
    """The composer applies a classified rule intent with a working
    undo and the full thread record (user turn, EDITED note, agent
    line). C's classifier is stubbed — its own judgement is pinned by
    C's eval suite; THIS pins A's application machinery."""
    from app.agent.automations import ledger
    from app.agent.automations.workflow import composer_ask, composer_undo

    async def _classify(text, workflow, complete=None):
        return {"applied": [{"kind": "rule", "text": text}],
                "needs": [], "answer": None}

    import app.agent.automations.composer as _composer
    monkeypatch.setattr(_composer, "classify_change", _classify)

    uid = await _mk_user()
    vspec = _v2_spec()
    a = await _mk_automation_v2(uid, vspec)
    async with async_session_maker() as db:
        a2 = await db.get(Automation, a.id)
        out = await composer_ask(
            db, automation=a2, user_id=uid,
            text="skip anything from recruiters",
        )
        assert len(out["applied"]) == 1
        applied = out["applied"][0]
        assert applied["kind"] == "rule"
        assert applied["sheet"] == "rules"
        token = applied["undo_token"]

        thread = await ledger.thread_for(db, a.id)
        turns, _ = await ledger.list_turns(db, thread_id=thread.id)
        kinds = [t["kind"] for t in turns]
        assert "user" in kinds and "agent" in kinds
        assert any(t["kind"] == "note" and t.get("stamp") == "edited"
                   for t in turns)

        undone = await composer_undo(
            db, automation=a2, user_id=uid, token=token,
        )
        assert undone == {"undone": True}
        from app.agent.automations.workflow import rules_list
        assert rules_list(a2) == []


@pytest.mark.asyncio
async def test_mode_and_output_derivation():
    from app.agent.automations.workflow import mode_of, output_block
    uid = await _mk_user()
    vspec = _v2_spec()
    a = await _mk_automation_v2(uid, vspec)
    raw = json.loads(a.spec_json)
    mode, label = mode_of(a, raw)
    assert mode == "posts" and label.startswith("posts")
    out = output_block(a, raw)
    assert out["header_sub"] == "only where you allowed it"

    # Reads-only spec.
    reads_spec = dict(raw)
    reads_spec["steps"] = [s for s in raw["steps"] if not s.get("grant_id")]
    mode2, label2 = mode_of(a, reads_spec)
    assert (mode2, label2) == ("reads_only", "reads only")
    out2 = output_block(a, reads_spec)
    assert out2["header_sub"] == "nothing is sent on your behalf"

    # Confirm mode ⇒ asks_first.
    confirm_spec = dict(raw)
    confirm_spec["mode"] = "confirm"
    assert mode_of(a, confirm_spec)[0] == "asks_first"
