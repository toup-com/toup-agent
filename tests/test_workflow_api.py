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

    # needs_consent through the PRODUCTION CALLER. This used to hand
    # `has_write_grant=False` straight to `permissions.save`, which
    # proved the inner branch fires when told there is no grant while
    # never exercising the thing that decides it. The caller derived the
    # answer from the connector's OAuth `scopes` — and jira's stub below
    # carries `scopes: ["r"]`, so the real path ALLOWED the write this
    # test claimed to refuse. Drive `save_permissions`.
    async with async_session_maker() as db:
        a3 = await db.get(Automation, a.id)
        jira_cat = permissions.catalog_for("jira")
        jira_writes = [p["id"] for p in jira_cat["writes"]]
        assert jira_writes, "fixture needs a jira write to refuse"
        with pytest.raises(WorkflowError) as e3:
            await save_permissions(
                db, automation=a3, user_id=uid, account_id="jira",
                can_ids=[p["id"] for p in jira_cat["reads"]] + jira_writes,
                cant_ids=[],
            )
        assert e3.value.code == "needs_consent"
        # §4.4 serves the NESTED consent object; the app has nothing to
        # run without it.
        consent = e3.value.extra.get("consent") or {}
        assert consent.get("connector_id") == "jira"
        assert consent.get("mode") in ("auto", "confirm")
        assert [x["id"] for x in consent.get("scopes") or []] == jira_writes
        assert all(x.get("label") for x in consent["scopes"])

    # ...and nothing was written: a refused save must not leave the
    # permission half-committed.
    async with async_session_maker() as db:
        a4 = await db.get(Automation, a.id)
        after = await permissions.resolve(
            db, automation=a4, account_id="jira",
        )
        assert all(p["id"] not in jira_writes for p in after["can"])

    assert await _edited_notes(a.id) >= 1


@pytest.mark.asyncio
async def test_revoked_grant_leaves_no_write_in_it_can(monkeypatch):
    """AUDIT-4: revoking the grant must move the write out of IT CAN.

    The spec keeps its `grant_id` after a revoke, so the unsaved default
    still resolves the write to CAN — the account sheet, whose whole job
    is to answer "what may this thing do?", kept naming a permission the
    platform had already taken away.
    """
    from app.agent.automations import permissions
    uid = await _mk_user()
    a = await _mk_automation_v2(uid, _v2_spec())

    async with async_session_maker() as db:
        a1 = await db.get(Automation, a.id)
        before = await permissions.resolve(
            db, automation=a1, account_id="slack",
        )
        write_ids = {p["id"] for p in
                     permissions.catalog_for("slack")["writes"]}
        allowed = {p["id"] for p in before["can"]} & write_ids
        assert allowed, "fixture must start with an allowed slack write"

        moved = await permissions.revoke_writes(
            db, automation=a1, connector_id="slack",
        )
        assert set(moved) == allowed

    async with async_session_maker() as db:
        a2 = await db.get(Automation, a.id)
        after = await permissions.resolve(
            db, automation=a2, account_id="slack",
        )
        assert not ({p["id"] for p in after["can"]} & write_ids)
        demoted = [p for p in after["cant"] if p["id"] in allowed]
        assert demoted and all(p["kind"] == "ungranted" for p in demoted)
        # A read survives — revoking a write is not taking the account.
        assert after["can"], "revoking a write must not strip the reads"


@pytest.mark.asyncio
async def test_oauth_scopes_are_not_an_approved_grant(monkeypatch):
    """AUDIT-2, stated as its own pin: a connector whose token carries a
    write scope still needs a grant. Scopes say the platform may hold a
    token with that reach; a grant says the user approved THIS
    automation to make THAT call. Conflating them let the green ✓
    allow "Post as you" for an automation nobody had granted anything.
    """
    from app.agent.automations import permissions
    from app.agent.automations import registry as reg
    from app.agent.automations.workflow import WorkflowError, save_permissions
    uid = await _mk_user()
    a = await _mk_automation_v2(uid, _v2_spec())

    # The platform says: that grant is gone.
    async def _revoked(user_id, grant_id):
        return {"id": grant_id, "status": "revoked", "connector_id": "slack"}
    monkeypatch.setattr(
        "app.agent.automations.registry.fetch_grant", _revoked,
    )

    cat = permissions.catalog_for("slack")
    async with async_session_maker() as db:
        a1 = await db.get(Automation, a.id)
        # The connection stub still reports slack `scopes: ["w"]`.
        conn = await reg.fetch_connection_state(uid)
        assert conn["slack"]["scopes"], "fixture must keep a write scope"
        assert await permissions.has_approved_write_grant(
            automation=a1, user_id=uid, connector_id="slack",
        ) is False
        with pytest.raises(WorkflowError) as e:
            await save_permissions(
                db, automation=a1, user_id=uid, account_id="slack",
                can_ids=[p["id"] for p in cat["reads"]]
                + [p["id"] for p in cat["writes"]],
                cant_ids=[],
            )
        assert e.value.code == "needs_consent"


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


@pytest.mark.asyncio
async def test_from_template_seeds_the_setup_thread(monkeypatch):
    """The C-caught silent-skip class: the seeding loop's best-effort
    except swallowed a TypeError and the setup script never ran. Pin:
    from-template leaves the YOU ADDED THIS note, the mode-aware agent
    line and the capability-check tool turn in the thread."""
    from app.api.automations import FromTemplateBody, from_template
    from app.agent.automations import ledger
    from app.config import settings

    uid = await _mk_user()
    monkeypatch.setattr(settings, "automations_enabled", True)
    monkeypatch.setattr(settings, "user_id", uid)

    template_spec = {
        "version": 2, "name": "Ledger brief", "mode": "auto",
        "trigger": {"sources": [
            {"id": "sched", "mode": "schedule",
             "schedule": {"cron_local": "0 8 * * 1-5"}},
        ]},
        "steps": [
            {"id": "issues", "connector_id": "jira",
             "tool": "jira__search_issues", "params": {"jql": "x"},
             "collect": {"items_path": "issues",
                         "fields": {"key": "key"},
                         "format": "{{item.key}}",
                         "empty_text": "none"},
             "on_error": "skip"},
        ],
    }

    async def _templates(user_id):
        return [{"id": "t-brief", "slug": "t-brief", "name": "Brief",
                 "category": "work", "variables": [],
                 "spec": template_spec}]

    monkeypatch.setattr(
        "app.agent.automations.registry.fetch_templates", _templates,
    )
    out = await from_template(FromTemplateBody(template_id="t-brief"))
    thread_id = out["thread_id"]
    assert out["automation"]["id"]

    async with async_session_maker() as db:
        turns, _more = await ledger.list_turns(
            db, thread_id=thread_id, limit=50,
        )
    kinds = [t["kind"] for t in turns]
    assert any(t["kind"] == "note" and t.get("stamp") == "added"
               for t in turns)
    assert "agent" in kinds, f"setup script silently skipped: {kinds}"
    cap = [t for t in turns if t["kind"] == "tool"]
    assert cap and cap[0]["action"] == "Checked what I can do", kinds


@pytest.mark.asyncio
async def test_every_composer_intent_kind_reaches_its_applier():
    """The contract between C's classifier and A's applier.

    AUDIT-1/AUDIT-5: `_apply_intent` read `intent["remove_id"]` for a
    permission revoke while `apply_policy` emits `permission_id`, and it
    had NO `account` branch at all. Both failures were silent — the
    applier returned None, the intent vanished, and the composer
    answered with whatever the sibling intents did. "Slack must never
    post as me" came back "Added a rule —" with the permission intact.

    So this feeds `apply_policy`'s REAL output in, for every kind in
    CHANGE_KINDS, and requires the state to actually move. A fixture
    hand-written to match the applier would have passed throughout.
    """
    from app.agent.automations import composer, permissions
    from app.agent.automations.workflow import (
        _apply_intent, workflow_payload,
    )

    uid = await _mk_user()
    a = await _mk_automation_v2(uid, _v2_spec())

    async with async_session_maker() as db:
        a1 = await db.get(Automation, a.id)
        wf = await workflow_payload(db, automation=a1, user_id=uid)

        slack_can = next(
            acc for acc in wf["accounts"] if acc["account_id"] == "slack"
        )
        # A write the automation currently HAS — the thing to revoke.
        writes = {p["id"] for p in permissions.catalog_for("slack")["writes"]}
        target = next(
            p for p in slack_can["can"] if p["id"] in writes
        )

        raw_intents = [
            {"kind": "rule", "text": "never on weekends"},
            {"kind": "schedule", "preset_id": "weekdays-9"},
            {"kind": "step", "n": 1, "text": "check the board first"},
            {"kind": "permission", "account_id": "slack",
             "direction": "revoke", "permission": target["label"]},
            {"kind": "account", "account_id": "jira",
             "direction": "remove"},
        ]
        assert {i["kind"] for i in raw_intents} == set(composer.CHANGE_KINDS)

        policy = composer.apply_policy(raw_intents, wf)
        by_kind = {i["kind"]: i for i in policy["applied"]}
        assert set(by_kind) == set(composer.CHANGE_KINDS), (
            f"policy dropped kinds: {set(composer.CHANGE_KINDS) - set(by_kind)}"
        )

        for kind in composer.CHANGE_KINDS:
            entry = await _apply_intent(
                db, automation=a1, user_id=uid, intent=by_kind[kind],
            )
            assert entry is not None, (
                f"{kind}: the applier silently dropped C's own intent "
                f"({by_kind[kind]!r})"
            )
            assert entry["kind"] == kind
            assert entry["sentence"], f"{kind}: no confirmation sentence"
            assert entry["undo_token"], f"{kind}: nothing to undo"

    # ...and the state moved, which is the claim that matters.
    async with async_session_maker() as db:
        a2 = await db.get(Automation, a.id)
        after = await permissions.resolve(
            db, automation=a2, account_id="slack",
        )
        assert target["id"] not in {p["id"] for p in after["can"]}, (
            "the revoke was reported but the permission is still allowed")
        from app.agent.automations.workflow import (
            _member_connectors, _spec_raw, rules_list,
        )
        assert "jira" not in _member_connectors(_spec_raw(a2))
        assert any("weekend" in r["text"] for r in rules_list(a2))


@pytest.mark.asyncio
async def test_a_refused_intent_does_not_abort_its_siblings(monkeypatch):
    """AUDIT-6: every applier commits its own write, so a WorkflowError
    escaping the loop left the earlier intents COMMITTED, answered 409,
    and skipped the agent turn — a half-applied change with no record of
    itself in the thread.
    """
    from app.agent.automations import composer, ledger
    from app.agent.automations.workflow import composer_ask

    uid = await _mk_user()
    a = await _mk_automation_v2(uid, _v2_spec())

    async def _fake_classify(text, wf, complete=None):
        return {
            "applied": [
                {"kind": "rule", "text": "skip anything from bots",
                 "sheet": "rules", "sentence": "Added a rule."},
                # ...and one the writer refuses.
                {"kind": "schedule", "preset_id": "no-such-preset",
                 "sheet": "schedule", "sentence": "Moved it."},
            ],
            "needs": [], "answer": None,
        }
    monkeypatch.setattr(composer, "classify_change", _fake_classify)

    async with async_session_maker() as db:
        a1 = await db.get(Automation, a.id)
        out = await composer_ask(
            db, automation=a1, user_id=uid,
            text="skip bot mail and move it to some time",
        )

    # The safe sibling landed...
    assert [e["kind"] for e in out["applied"]] == ["rule"]
    # ...and the refusal is SAID, not swallowed.
    assert out["answer"] and "time" in out["answer"].lower()

    # The thread carries the agent turn, and it mentions both halves.
    async with async_session_maker() as db:
        thread = await ledger.thread_for(db, a.id)
        turns = (await db.execute(
            select(AutomationTurn)
            .where(AutomationTurn.thread_id == thread.id)
            .order_by(AutomationTurn.seq)
        )).scalars().all()
        agent_turns = [t for t in turns if t.kind == "agent"]
        assert agent_turns, "no agent turn — the composer aborted"
        said = json.loads(agent_turns[-1].payload_json)["text"]
        assert "rule" in said.lower()
        assert "time" in said.lower(), f"the refusal is missing: {said!r}"


@pytest.mark.asyncio
async def test_schedule_sheet_never_offers_a_preset_the_writer_rejects():
    """AUDIT-7: the sheet appends a synthetic `current` row whenever the
    schedule is none of the four presets — and the writer 409'd on the
    very id it had just served. Selecting it is a no-op, not an error.
    """
    from app.agent.automations.workflow import (
        schedule_block, set_schedule_preset, _spec_raw,
    )
    uid = await _mk_user()
    # A schedule that is none of the four presets.
    spec = _v2_spec(trigger={"sources": [
        {"id": "sched", "mode": "schedule",
         "schedule": {"cron_local": "17 6 * * 2"}},
    ]})
    a = await _mk_automation_v2(uid, spec)

    async with async_session_maker() as db:
        a1 = await db.get(Automation, a.id)
        block = schedule_block(a1, _spec_raw(a1))
        assert block["preset_id"] == "current"
        for row in block["presets"]:
            result = await set_schedule_preset(
                db, automation=a1, user_id=uid, preset_id=row["id"],
            )
            assert result["sentence"]


@pytest.mark.asyncio
async def test_v1_automation_has_a_steps_sheet():
    """AUDIT-8: the derivation handled v2 only, so every v1 automation
    opened an EMPTY Steps sheet — the canvas asserting it does nothing.
    """
    from app.agent.automations.workflow import _steps_human, _spec_raw

    uid = await _mk_user()
    v1 = {
        "trigger": {"mode": "poll", "source": {
            "connector_id": "jira", "event": "issue_updated"},
            "poll_interval_s": 900},
        "action": {"connector_id": "slack",
                   "tool": "slack__send_message",
                   "params_template": {"text": "{{event.summary}}"}},
        "dedupe_key": "event.id",
    }
    async with async_session_maker() as db:
        a = Automation(
            id=str(uuid.uuid4()), user_id=uid, name="v1 relay",
            spec_json=json.dumps(v1), status="draft",
            trigger_mode="poll", connector_id="jira",
        )
        db.add(a)
        await db.commit()
        steps = _steps_human(a, _spec_raw(a))

    assert steps, "a v1 automation showed an empty Steps sheet"
    assert [s["n"] for s in steps] == list(range(1, len(steps) + 1))
    assert all(s["text"] for s in steps)


@pytest.mark.asyncio
async def test_deleting_a_rule_that_is_gone_is_not_an_edit():
    """AUDIT-10: DELETE of an unknown rule answered 200 "Removed the
    rule." and stamped an EDITED note — a record of an edit that never
    happened.
    """
    from app.agent.automations.workflow import (
        WorkflowError, add_rule, delete_rule,
    )
    uid = await _mk_user()
    a = await _mk_automation_v2(uid, _v2_spec())

    async with async_session_maker() as db:
        a1 = await db.get(Automation, a.id)
        await add_rule(db, automation=a1, text="a real rule")
    before = await _edited_notes(a.id)

    async with async_session_maker() as db:
        a2 = await db.get(Automation, a.id)
        with pytest.raises(WorkflowError) as e:
            await delete_rule(db, automation=a2, rule_id="no-such-rule")
        assert e.value.code == "not_found"

    assert await _edited_notes(a.id) == before, (
        "a failed delete stamped an EDITED note")


@pytest.mark.asyncio
async def test_last_use_is_scoped_to_its_user_and_automation():
    """AUDIT-9: the scan had no user or automation scope, so the sheet
    opened inside an automation reported a NEIGHBOUR's activity as this
    one's, and a busy neighbour could push a real use out of the window.
    """
    from app.api.automations import account_last_use
    from app.agent.automations import ledger

    uid = await _mk_user()
    mine = await _mk_automation_v2(uid, _v2_spec())
    other = await _mk_automation_v2(uid, _v2_spec())

    async with async_session_maker() as db:
        for automation, action in ((other, "Posted in #other"),
                                   (mine, "Posted in #mine")):
            thread = await ledger.ensure_thread(
                db, user_id=uid, automation_id=automation.id,
            )
            await ledger.append_turn(
                db, user_id=uid, thread=thread, run_id=None, kind="tool",
                payload={"account_id": "slack", "action": action,
                         "detail": "", "ok": True,
                         "tool_kind": "write"},
            )

    async with async_session_maker() as db:
        scoped = await account_last_use(
            db, user_id=uid, account_id="slack", automation_id=mine.id,
        )
        assert scoped["sentence"] == "Posted in #mine"
        neighbour = await account_last_use(
            db, user_id=uid, account_id="slack", automation_id=other.id,
        )
        assert neighbour["sentence"] == "Posted in #other"

    # Another user's activity is never this user's last use.
    stranger = await _mk_user()
    async with async_session_maker() as db:
        assert (await account_last_use(
            db, user_id=stranger, account_id="slack",
        ))["at"] is None
