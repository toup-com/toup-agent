# agent-mode: automations/automation_threads/_turns are AGENT_ONLY.
"""R38 — the agent gets real hands.

Four things the agent could not do, each of which it had to lie about
instead:

  A  a live automation could only be changed by replacing its WHOLE
     spec (`automations__update`), so "make it 9am" or "never post
     anywhere else" was agreed to in words and never applied. Five
     tools now lift the five composer change kinds through the SAME
     policy, the same writers, the same undo and the same EDITED note.
  B  `automations__test_run` was DEV-only for two reasons that were
     both true: its "staged" write was swept and sent by the outbox
     like any other, and it short-circuited the ledger so the work it
     did had no surface. A rehearsal now stages NOTHING — it renders
     what would be written and reports it.
  C  a connector WRITE from the automation thread was denied outright,
     because there was no surface to elevate on. There is one now: the
     call is staged and rendered as a `needs_you` turn with
     `fix="approve"`, wired to the pending-action endpoints that
     already existed.
  D  the 16 tools withheld from the thread were a bare frozenset — a
     list with no reasons, which is a list nobody can audit.
"""

from __future__ import annotations

import json
import uuid

import pytest

from app.config import settings
from app.db.database import async_session_maker
from app.db.models import Automation, AutomationTurn, User

from tests.test_run_ledger_v3 import (  # noqa: F401 — shared fixtures
    REGISTRY_V2, _mk_user, _v2_spec, _mk_automation_v2,
)


def _ctx(user_id: str):
    from types import SimpleNamespace
    return SimpleNamespace(user_id=user_id, conversation_id=None,
                           message_id=None, job_id=None)


@pytest.fixture(autouse=True)
def _offline_platform(monkeypatch):
    """Every surface here reads connection state + capability over the
    platform RPC. Stub both so the tests run without a platform — the
    same stubs `test_workflow_api` uses, so the two suites agree about
    what the world looks like."""
    async def _conn_state(user_id):
        return {
            "jira": {"connector_id": "jira", "connected": True,
                     "status": "active", "scopes": ["r"],
                     "account": "TP project"},
            "slack": {"connector_id": "slack", "connected": True,
                      "status": "active", "scopes": ["w"],
                      "account": "toup.ai workspace"},
        }

    async def _registry(user_id, force=False):
        return REGISTRY_V2

    async def _grant(user_id, grant_id):
        return {"id": grant_id, "status": "approved",
                "connector_id": "slack",
                "tool_name": "slack__send_message",
                "target": {"kind": "channel", "id": "C-PIN",
                           "label": "#platform"},
                "mode": "auto"}

    for path, fn in (
        ("app.agent.automations.registry.fetch_connection_state", _conn_state),
        ("app.agent.automations.registry.fetch_registry", _registry),
        ("app.agent.automations.registry.fetch_grant", _grant),
    ):
        monkeypatch.setattr(path, fn)
    monkeypatch.setattr(settings, "automations_enabled", True)


async def _edited_notes(automation_id: str) -> int:
    from app.agent.automations import ledger
    async with async_session_maker() as db:
        thread = await ledger.thread_for(db, automation_id)
        if thread is None:
            return 0
        rows = (await db.execute(
            __import__("sqlalchemy").select(AutomationTurn)
            .where(AutomationTurn.thread_id == thread.id)
            .where(AutomationTurn.kind == "note")
        )).scalars().all()
        return sum(1 for r in rows
                   if json.loads(r.payload_json).get("stamp") == "edited")


async def _call(tool: str, args: dict, uid: str):
    from app.agent.skills.builtins.automations.skill import AutomationsSkill
    return await AutomationsSkill().execute_tool(tool, args, _ctx(uid))


def _body(result) -> dict:
    assert not str(result).startswith("ERROR"), str(result)
    return json.loads(str(result))


# ─────────────────────────────────────────────────────────────────────
# D · the withheld set carries its reasons
# ─────────────────────────────────────────────────────────────────────

#: The tripwire's own copy. Changing the withheld set means changing
#: THIS list too, in the same diff, which is the point: a tool added or
#: removed silently is exactly what nobody could audit before.
_WITHHELD = {
    "create_job",
    "update_job",
    "spawn",
    "start_mission",
    "save_streaming_credential",
    "memory_store",
    "memory_write_file",
    "memory_edit_file",
    "memory_delete",
    "routines__create",
    "routines__update",
    "routines__delete",
    "routines__remind",
    "triggers__create",
    "triggers__update",
    "triggers__delete",
}


def test_the_withheld_set_is_exactly_this():
    """The tripwire. It fails on an ADD and on a REMOVAL, so neither
    can happen without someone writing the reason down here as well."""
    from app.agent.prompt_profile import AUTOMATION_THREAD_WITHHELD

    got = set(AUTOMATION_THREAD_WITHHELD)
    assert got == _WITHHELD, {
        "added": sorted(got - _WITHHELD),
        "removed": sorted(_WITHHELD - got),
    }


def test_every_withheld_tool_carries_a_real_reason():
    """A reason that repeats the tool's name, or says "TODO", is the
    absent reason in a costume."""
    from app.agent.prompt_profile import AUTOMATION_THREAD_WITHHELD

    for tool, reason in AUTOMATION_THREAD_WITHHELD.items():
        assert isinstance(reason, str), tool
        assert len(reason.strip()) >= 30, (tool, reason)
        assert "TODO" not in reason, tool
        assert reason.strip().lower() != tool.lower(), tool
        # A reason is prose about consequence, not a restatement of the
        # identifier — no tool ids inside it.
        assert "__" not in reason, (tool, reason)


def test_the_memory_writers_stay_withheld_and_say_why():
    """The one exemption R38 does NOT make.

    C gives the thread an elevation surface for CONNECTOR writes — the
    user sees the target and taps. A memory write has no target to
    show, and Round 33 item 6 is what it does when it is available: a
    run's connector failure ("Gmail could not be read") filed as a
    durable fact about the person.
    """
    from app.agent.prompt_profile import AUTOMATION_THREAD_WITHHELD

    for writer in ("memory_store", "memory_write_file",
                   "memory_edit_file", "memory_delete"):
        assert writer in AUTOMATION_THREAD_WITHHELD, writer
    assert "Round 33" in AUTOMATION_THREAD_WITHHELD["memory_store"]


def test_the_frozenset_every_consumer_reads_is_the_dict_keys():
    """`disabled_tools_for_channel` and `agent_runner` read the
    frozenset. Deriving it from the dict is what makes the reasons
    impossible to skip — two literals would drift on the first edit."""
    from app.agent.prompt_profile import (
        AUTOMATION_THREAD_DISABLED_TOOLS, AUTOMATION_THREAD_WITHHELD,
        disabled_tools_for_channel,
    )

    assert AUTOMATION_THREAD_DISABLED_TOOLS == frozenset(
        AUTOMATION_THREAD_WITHHELD)
    assert disabled_tools_for_channel("automation_thread") == \
        AUTOMATION_THREAD_DISABLED_TOOLS
    assert disabled_tools_for_channel("app") == frozenset()


# ─────────────────────────────────────────────────────────────────────
# A · five edit tools, one applier
# ─────────────────────────────────────────────────────────────────────

#: The tools array is PREFIX-STABLE per channel: new tools only ever
#: join at the end, because the proxy's 128-tool cap trims from a
#: namespace's tail and a reordered prefix invalidates every cached
#: provider prefix. This is the array as it stood before R38.
_PREFIX_BEFORE_R38 = [
    "automations__get_registry",
    "automations__request_connection",
    "automations__request_permission",
    "automations__list_targets",
    "automations__list",
    "automations__list_templates",
    "automations__create",
    "automations__update",
    "automations__test_run",
    "automations__arm",
    "automations__pause",
    "automations__resume",
    "automations__delete",
    "automations__memory_recall",
    "automations__run_now",
    "automations__set_destination",
]

_R38_TOOLS = [
    "automations__edit_schedule",
    "automations__edit_rules",
    "automations__edit_steps",
    "automations__edit_permissions",
    "automations__edit_accounts",
]


def test_the_five_join_at_the_tail_and_the_prefix_is_byte_stable():
    from app.agent.skills.builtins.automations.skill import AutomationsSkill

    names = [t["name"] for t in AutomationsSkill()._all_tools()]
    assert names[:len(_PREFIX_BEFORE_R38)] == _PREFIX_BEFORE_R38
    assert names[len(_PREFIX_BEFORE_R38):] == _R38_TOOLS


def test_every_edit_tool_is_dispatchable_and_prefixed():
    """`SkillLoader._register` RAISES on the first unprefixed name and
    `load_all` swallows the raise — so one bare name here discards the
    ENTIRE automations skill (wire 90 → 77 on main, once). And a tool
    in the array with no dispatch entry answers "Unknown automations
    tool" to a model that can see it."""
    from app.agent.skills.builtins.automations.skill import AutomationsSkill

    skill = AutomationsSkill()
    declared = {t["name"] for t in skill._all_tools()}
    for name in _R38_TOOLS:
        assert name.startswith("automations__"), name
        assert name in declared, name
    # Every declared name resolves to a handler.
    import inspect
    src = inspect.getsource(type(skill).execute_tool)
    for name in _R38_TOOLS:
        assert f'"{name}"' in src, name


def test_the_edit_tools_never_reach_a_writer_of_their_own():
    """The one door. Each tool hands intents to
    `workflow.apply_intents`, which runs `composer.apply_policy` and
    then `workflow._apply_intent` — the canvas sheet's own writers. A
    handler that imported `add_rule` or `save_permissions` directly
    would be a second implementation of the policy, free to drift from
    the one the sheet enforces, which is exactly how a tool call could
    come to widen access that a sentence cannot.
    """
    import pathlib

    src = (pathlib.Path(__file__).resolve().parents[1] / "app" / "agent" /
           "skills" / "builtins" / "automations" / "skill.py").read_text()
    # The comment block above the tools NAMES these on purpose — read
    # the code, not the prose about it.
    code = "\n".join(
        line for line in src.splitlines()
        if not line.lstrip().startswith("#")
    )
    for writer in ("add_rule", "delete_rule", "update_rule", "set_steps",
                   "save_permissions", "set_schedule_preset",
                   "set_schedule_custom", "remove_connector",
                   "_apply_intent"):
        assert writer not in code, (
            f"{writer} is reachable from the skill — the edit tools must "
            f"go through workflow.apply_intents and nothing else"
        )
    assert "apply_intents" in code


@pytest.mark.asyncio
async def test_edit_schedule_moves_the_time_for_real():
    uid = await _mk_user()
    a = await _mk_automation_v2(uid, _v2_spec())

    out = _body(await _call("automations__edit_schedule",
                            {"automation_id": a.id,
                             "preset_id": "weekdays-730"}, uid))
    assert out["changed"] == ["Moved it to weekdays at 7:30."]
    assert out["it_now_looks_like"]["runs"] == "Weekdays at 7:30"

    # A time the four presets do not carry still lands, and the
    # sentence is rendered from the schedule that was ARMED.
    out = _body(await _call("automations__edit_schedule",
                            {"automation_id": a.id, "time": "06:15",
                             "days": [1, 2, 3, 4, 5]}, uid))
    assert out["changed"], out
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        spec = json.loads(row.spec_json)
    cron = [s for s in spec["trigger"]["sources"] if s.get("schedule")]
    assert cron[0]["schedule"]["cron_local"] == "15 6 * * 1,2,3,4,5"


@pytest.mark.asyncio
async def test_edit_schedule_without_a_time_refuses_instead_of_guessing():
    uid = await _mk_user()
    a = await _mk_automation_v2(uid, _v2_spec())
    out = await _call("automations__edit_schedule",
                      {"automation_id": a.id}, uid)
    assert str(out).startswith("ERROR")


@pytest.mark.asyncio
async def test_edit_rules_adds_removes_and_finds_one_by_its_own_words():
    """The model has the user's words far more often than it has a
    rule id ("drop the no-thread rule"), so `remove` takes either."""
    uid = await _mk_user()
    a = await _mk_automation_v2(uid, _v2_spec())

    out = _body(await _call(
        "automations__edit_rules",
        {"automation_id": a.id,
         "add": ["Never post anywhere else.", "One line only."]}, uid))
    assert len(out["changed"]) == 2
    rules = out["it_now_looks_like"]["rules"]
    assert [r["text"] for r in rules] == [
        "Never post anywhere else.", "One line only."]
    assert all(r["id"] for r in rules)

    # Remove by TEXT; reword by id.
    out = _body(await _call(
        "automations__edit_rules",
        {"automation_id": a.id, "remove": ["One line only."],
         "edit": [{"rule_id": rules[0]["id"], "text": "Never post at all."}]},
        uid))
    assert len(out["changed"]) == 2
    assert [r["text"] for r in out["it_now_looks_like"]["rules"]] == [
        "Never post at all."]


@pytest.mark.asyncio
async def test_removing_a_rule_that_is_gone_says_so_and_changes_nothing():
    uid = await _mk_user()
    a = await _mk_automation_v2(uid, _v2_spec())
    out = _body(await _call("automations__edit_rules",
                            {"automation_id": a.id,
                             "remove": ["a rule nobody wrote"]}, uid))
    assert out["changed"] == []
    assert out["not_changed"] == ["That rule is gone."]
    assert "NOTHING changed" in out["next"]
    assert await _edited_notes(a.id) == 0


@pytest.mark.asyncio
async def test_edit_steps_rewords_step_n(monkeypatch):
    from app.agent.automations import recompiler

    async def _complete(prompt):
        spec = json.loads(_v2_spec().raw and json.dumps(_v2_spec().raw))
        spec["name"] = "Ledger brief"
        return {"spec": spec}

    monkeypatch.setattr(recompiler, "_default_complete", _complete)

    uid = await _mk_user()
    a = await _mk_automation_v2(uid, _v2_spec())
    out = _body(await _call(
        "automations__edit_steps",
        {"automation_id": a.id, "n": 1,
         "text": "Read only the issues assigned to me"}, uid))
    assert out["changed"], out
    steps = out["it_now_looks_like"]["steps"]
    assert steps[0]["text"] == "Read only the issues assigned to me"
    assert steps[0]["n"] == 1


@pytest.mark.asyncio
async def test_a_step_that_is_not_there_is_named_not_swallowed():
    """`_apply_intent` used to `return None` here and `composer_ask`
    appended None to nothing — so "change step 4" on a two-step
    automation reported the OTHER changes in the same sentence and said
    not one word about the one that did nothing. A tool inheriting that
    would report success for a change it did not make."""
    uid = await _mk_user()
    a = await _mk_automation_v2(uid, _v2_spec())
    out = _body(await _call("automations__edit_steps",
                            {"automation_id": a.id, "n": 9,
                             "text": "do something else"}, uid))
    assert out["changed"] == []
    assert out["not_changed"] == ["There is no step 9 — this one has 2 of them."]
    assert await _edited_notes(a.id) == 0


@pytest.mark.asyncio
async def test_edit_permissions_revokes_for_real_and_will_not_grant():
    uid = await _mk_user()
    a = await _mk_automation_v2(uid, _v2_spec())

    out = _body(await _call(
        "automations__edit_permissions",
        {"automation_id": a.id, "account_id": "slack",
         "permission": "Post as you", "direction": "revoke"}, uid))
    assert out["changed"], out
    slack = [x for x in out["it_now_looks_like"]["accounts"]
             if x["account_id"] == "slack"][0]
    assert "Post as you" in slack["cannot"]
    assert "Post as you" not in slack["can"]

    # And now it is already off — a second revoke says so rather than
    # reporting a change it did not make.
    out = _body(await _call(
        "automations__edit_permissions",
        {"automation_id": a.id, "account_id": "slack",
         "permission": "Post as you", "direction": "revoke"}, uid))
    assert out["changed"] == []
    assert out["not_changed"] == ["Slack already cannot post as you."]


@pytest.mark.asyncio
async def test_granting_a_permission_is_the_users_call_not_the_tools():
    uid = await _mk_user()
    a = await _mk_automation_v2(uid, _v2_spec())
    async with async_session_maker() as db:
        from app.agent.automations.workflow import save_permissions
        row = await db.get(Automation, a.id)
        await save_permissions(
            db, automation=row, user_id=uid, account_id="slack",
            can_ids=["slack.read_channels"], cant_ids=["slack.post_as_you"],
        )

    out = _body(await _call(
        "automations__edit_permissions",
        {"automation_id": a.id, "account_id": "slack",
         "permission": "Post as you", "direction": "grant"}, uid))
    assert out["changed"] == []
    assert out["needs_your_approval"], out
    assert "your yes" in out["needs_your_approval"][0]
    assert "automations__request_permission" in out["next"]


@pytest.mark.asyncio
async def test_a_hard_rail_is_refused_in_plain_words():
    """The policy a tool must not be able to route around."""
    uid = await _mk_user()
    a = await _mk_automation_v2(uid, _v2_spec())
    out = _body(await _call(
        "automations__edit_permissions",
        {"automation_id": a.id, "account_id": "slack",
         "permission": "Read private DMs", "direction": "grant"}, uid))
    assert out["changed"] == []
    assert out["needs_your_approval"] == []
    assert out["not_changed"] == ["It can never do this."]


@pytest.mark.asyncio
async def test_edit_accounts_removes_for_real_and_will_not_add():
    uid = await _mk_user()
    a = await _mk_automation_v2(uid, _v2_spec())

    out = _body(await _call(
        "automations__edit_accounts",
        {"automation_id": a.id, "account_id": "jira",
         "direction": "add"}, uid))
    assert out["changed"] == []
    assert out["needs_your_approval"], out
    assert "needs your yes" in out["needs_your_approval"][0]
    assert "automations__request_connection" in out["next"]

    out = _body(await _call(
        "automations__edit_accounts",
        {"automation_id": a.id, "account_id": "jira",
         "direction": "remove"}, uid))
    assert out["changed"], out
    assert "Jira" in out["changed"][0]
    assert [x["account_id"] for x in out["it_now_looks_like"]["accounts"]] \
        == ["slack"]

    # Gone means gone — a second removal names the account rather than
    # answering nothing at all.
    out = _body(await _call(
        "automations__edit_accounts",
        {"automation_id": a.id, "account_id": "jira",
         "direction": "remove"}, uid))
    assert out["changed"] == []
    assert out["not_changed"] == [
        "Jira is not part of this automation, so there is nothing to "
        "take out."]


@pytest.mark.asyncio
async def test_removing_the_account_that_does_the_writing_is_refused_aloud():
    """`remove_connector` refuses a connector that performs the write.
    The tool must say that sentence, not report a removal that did not
    happen and not answer with silence."""
    uid = await _mk_user()
    a = await _mk_automation_v2(uid, _v2_spec())
    out = _body(await _call(
        "automations__edit_accounts",
        {"automation_id": a.id, "account_id": "slack",
         "direction": "remove"}, uid))
    assert out["changed"] == []
    assert out["not_changed"] == [
        "Slack is doing work this automation depends on, so I left it in."]
    assert await _edited_notes(a.id) == 0

    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        assert "slack" in json.dumps(json.loads(row.spec_json))


async def test_every_applied_edit_stamps_the_edited_note_once():
    """`_edited_note` is ONE seam for two facts — the EDITED turn and
    the `automation.updated` broadcast. An agent edit that skipped it
    is the R38 defect one layer down: the founder's first edit drew a
    divider in the thread and the second did not."""
    uid = await _mk_user()
    a = await _mk_automation_v2(uid, _v2_spec())
    assert await _edited_notes(a.id) == 0

    await _call("automations__edit_rules",
                {"automation_id": a.id, "add": ["Only unread."]}, uid)
    assert await _edited_notes(a.id) == 1

    await _call("automations__edit_schedule",
                {"automation_id": a.id, "preset_id": "daily-8"}, uid)
    assert await _edited_notes(a.id) == 2


@pytest.mark.asyncio
async def test_an_applied_edit_is_undoable_on_the_same_token(monkeypatch):
    """The tools mint the composer's own undo token, from the composer's
    own writers — so the 12-second take-back the sheet offers is the
    take-back an agent edit offers too."""
    from app.agent.automations import workflow

    uid = await _mk_user()
    a = await _mk_automation_v2(uid, _v2_spec())
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        out = await workflow.apply_intents(
            db, automation=row, user_id=uid,
            intents=[{"kind": "rule", "text": "Never post anywhere."}],
        )
        token = out["applied"][0]["undo_token"]
        assert token
        await workflow.composer_undo(
            db, automation=row, user_id=uid, token=token,
        )
        assert workflow.rules_list(row) == []


@pytest.mark.asyncio
async def test_a_rule_removed_by_the_agent_comes_back_whole():
    """The undo for a remove/edit reverts the WHOLE list: a rule that
    returns with a new id is a rule every other token and every later
    edit has lost track of."""
    from app.agent.automations import workflow

    uid = await _mk_user()
    a = await _mk_automation_v2(uid, _v2_spec())
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        added = await workflow.apply_intents(
            db, automation=row, user_id=uid,
            intents=[{"kind": "rule", "text": "Only unread."}],
        )
        del added
        before = [dict(r) for r in workflow.rules_list(row)]
        removed = await workflow.apply_intents(
            db, automation=row, user_id=uid,
            intents=[{"kind": "rule", "op": "remove",
                      "rule_id": before[0]["id"]}],
        )
        assert workflow.rules_list(row) == []
        await workflow.composer_undo(
            db, automation=row, user_id=uid,
            token=removed["applied"][0]["undo_token"],
        )
        assert workflow.rules_list(row) == before


@pytest.mark.asyncio
async def test_the_sheet_and_the_tools_go_through_the_same_applier(monkeypatch):
    """Not a source probe — both routes are driven and both are
    observed landing in `_apply_intent`."""
    from app.agent.automations import workflow

    seen = []
    real = workflow._apply_intent

    async def _spy(db, *, automation, user_id, intent):
        seen.append(intent.get("kind"))
        return await real(db, automation=automation, user_id=user_id,
                          intent=intent)

    monkeypatch.setattr(workflow, "_apply_intent", _spy)

    async def _classify(text, wf, complete=None):
        return {"applied": [{"kind": "rule", "text": "From the sheet.",
                             "sentence": "Added a rule — from the sheet.",
                             "sheet": "rules"}],
                "needs": [], "answer": None}

    import app.agent.automations.composer as _composer
    monkeypatch.setattr(_composer, "classify_change", _classify)

    uid = await _mk_user()
    a = await _mk_automation_v2(uid, _v2_spec())
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        await workflow.composer_ask(
            db, automation=row, user_id=uid, text="never post anywhere")
        await workflow.apply_intents(
            db, automation=row, user_id=uid,
            intents=[{"kind": "rule", "text": "From the tool."}])
    assert seen == ["rule", "rule"]


# ─────────────────────────────────────────────────────────────────────
# B · a rehearsal stages nothing, so nothing can send it
# ─────────────────────────────────────────────────────────────────────

def test_no_executor_can_stage_a_write_without_flushing_it():
    """The structural half, and the one that matters.

    `stage_only=True` committed the outbox row and returned. A staged
    row is not a held row: `outbox.flush_loop` sweeps EVERY row whose
    `execute_after` has passed, every 5 s — so the "rehearsal" posted
    to the user's real channel seconds later, from a background loop,
    with the caller already gone. The mode is gone from both executors;
    a rehearsal has no row for any loop to find.
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "app" / "agent" \
        / "automations"
    for name in ("executor.py", "executor_v2.py"):
        code = "\n".join(
            line for line in (root / name).read_text().splitlines()
            if not line.lstrip().startswith("#")
        )
        assert "stage_only" not in code, name


@pytest.mark.asyncio
async def test_a_rehearsal_reads_for_real_and_writes_nothing(monkeypatch):
    from sqlalchemy import select

    from app.db.models import AutomationOutbox
    from tests.test_run_ledger_v3 import _ISSUES, _OK, _fake_dispatch

    dispatch = _fake_dispatch({"jira__search_issues": _ISSUES,
                               "slack__send_message": _OK})
    monkeypatch.setattr(
        "app.agent.automations.registry.dispatch_via_platform", dispatch)

    uid = await _mk_user()
    a = await _mk_automation_v2(uid, _v2_spec())
    out = _body(await _call("automations__test_run",
                            {"automation_id": a.id}, uid))

    assert out["nothing_was_sent"] is True
    # The read really happened — that is what makes a rehearsal worth
    # anything — and the write really did not.
    assert [c["tool"] for c in dispatch.calls] == ["jira__search_issues"]
    assert out["it_read"][0] == {
        "step_id": "issues", "account_id": "jira", "ok": True, "count": 2,
        "text": "TP-482 Rate-limit the export endpoint\n"
                "TP-476 Flaky memverify test",
    }
    # And it says exactly what WOULD have gone out, rendered by the
    # same renderer the run uses.
    write = out["it_would_write"][0]
    assert write["account_id"] == "slack"
    assert write["blocked"] is None
    assert write["params"]["channel"] == "C-PIN"
    assert write["params"]["text"].startswith("TP-482")

    async with async_session_maker() as db:
        rows = (await db.execute(select(AutomationOutbox))).scalars().all()
    assert rows == [], "a rehearsal staged an outbox row"


@pytest.mark.asyncio
async def test_a_rehearsal_is_not_a_run(monkeypatch):
    """It opened a real `BuildJob`, fired the start notification and
    then short-circuited the ledger's close, so the row sat `running`
    until the stuck-run reaper marked it `failed/lost`. A rehearsal
    changes nothing, so there is nothing for a run row to record."""
    from sqlalchemy import select

    from app.db.models import BuildJob
    from tests.test_run_ledger_v3 import _ISSUES, _OK, _fake_dispatch

    monkeypatch.setattr(
        "app.agent.automations.registry.dispatch_via_platform",
        _fake_dispatch({"jira__search_issues": _ISSUES,
                        "slack__send_message": _OK}))

    uid = await _mk_user()
    a = await _mk_automation_v2(uid, _v2_spec())
    await _call("automations__test_run", {"automation_id": a.id}, uid)

    async with async_session_maker() as db:
        jobs = (await db.execute(
            select(BuildJob).where(BuildJob.source_id == a.id)
        )).scalars().all()
    assert jobs == [], "a rehearsal opened a run"


@pytest.mark.asyncio
async def test_a_rehearsal_names_what_would_stop_the_real_write(monkeypatch):
    """"It would post your board to #platform" is the wrong answer for
    a write no approved permission backs — the real run refuses it, and
    a rehearsal that hides the refusal rehearses something that cannot
    happen."""
    from tests.test_run_ledger_v3 import _ISSUES, _fake_dispatch

    monkeypatch.setattr(
        "app.agent.automations.registry.dispatch_via_platform",
        _fake_dispatch({"jira__search_issues": _ISSUES}))

    async def _pending(user_id, grant_id):
        return {"id": grant_id, "status": "pending"}

    monkeypatch.setattr(
        "app.agent.automations.registry.fetch_grant", _pending)

    uid = await _mk_user()
    a = await _mk_automation_v2(uid, _v2_spec())
    result = await _call("automations__test_run",
                         {"automation_id": a.id}, uid)
    out = _body(result)
    assert out["it_would_write"][0]["blocked"] == (
        "The permission it needs is pending.")
    assert result.display == "Rehearsed it — something is in the way"


@pytest.mark.asyncio
async def test_a_broken_read_is_reported_not_swallowed(monkeypatch):
    from tests.test_run_ledger_v3 import _fake_dispatch

    monkeypatch.setattr(
        "app.agent.automations.registry.dispatch_via_platform",
        _fake_dispatch({"jira__search_issues": {"kind": "tool_error",
                                                "message": "boom"}}))

    uid = await _mk_user()
    a = await _mk_automation_v2(uid, _v2_spec())
    out = _body(await _call("automations__test_run",
                            {"automation_id": a.id}, uid))
    read = out["it_read"][0]
    assert read["ok"] is False
    assert read["account_id"] == "jira"
    assert "boom" in read["error"]
    # The write is still reported — seeing the hole the failed read
    # leaves in it is the reason to rehearse at all.
    assert out["it_would_write"], out


def test_the_tool_never_lets_the_model_call_a_rehearsal_a_run():
    """The one thing the model must not do with a safe tool: report it
    as work. Both the description and the answer say so."""
    from app.agent.skills.builtins.automations.skill import AutomationsSkill

    tool = next(t for t in AutomationsSkill()._all_tools()
                if t["name"] == "automations__test_run")
    assert "Nothing is sent" in tool["description"]
    assert "automations__run_now" in tool["description"]

    prompt = AutomationsSkill().get_system_prompt_section() or ""
    assert "NEVER the answer to 'run it'" in prompt


# ─────────────────────────────────────────────────────────────────────
# C · the thread's elevation surface (its own half — the dispatcher's
#     half is test_r38_thread_elevation.py, platform lane)
# ─────────────────────────────────────────────────────────────────────

def _card(**over) -> dict:
    card = {"action_id": "pa-1", "connector_id": "slack",
            "tool_name": "slack__send_message",
            "summary": "Post to #platform: shipped", "payload": {},
            "expires_at": None, "status": "pending"}
    card.update(over)
    return card


@pytest.mark.asyncio
async def test_a_staged_write_becomes_an_approvable_turn():
    """The surface itself. Without it, `automation_thread` could not be
    a confirmable channel at all — the dispatcher would stage a call
    into a thread with nothing to tap."""
    from app.agent.automations import ledger, thread_agent

    uid = await _mk_user()
    a = await _mk_automation_v2(uid, _v2_spec())
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        thread = await ledger.ensure_thread(db, user_id=uid,
                                            automation_id=a.id)
        await thread_agent._append_approval_turns(
            db, automation=row, thread=thread, run_id=None,
            staged=[_card()],
        )
        turns, _ = await ledger.list_turns(db, thread_id=thread.id)

    card = [t for t in turns if t["kind"] == "needs_you"]
    assert len(card) == 1
    body = card[0]
    assert body["fix"] == "approve"
    assert body["pending_action_id"] == "pa-1"
    assert body["account_id"] == "slack"
    assert body["name"] == "Slack"
    assert "Post to #platform: shipped" in body["sentence"]
    # It must not claim the write happened — that is the one sentence
    # this whole surface exists to make impossible.
    assert "has not done this yet" in body["sentence"]


@pytest.mark.asyncio
async def test_a_card_with_no_bespoke_line_keeps_its_buttons():
    """`summarize_pending_action` falls back to `Run <tool>` for a
    connector with no bespoke line, and `ledger._RAW_TOOL_RE` rejects a
    raw tool id in any sentence we author — so the whole turn would be
    sanitized into a plain `agent` bubble with NO BUTTONS in prod, and
    raise in dev. Drop the summary, never the card."""
    from app.agent.automations import ledger, thread_agent

    uid = await _mk_user()
    a = await _mk_automation_v2(uid, _v2_spec())
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        thread = await ledger.ensure_thread(db, user_id=uid,
                                            automation_id=a.id)
        await thread_agent._append_approval_turns(
            db, automation=row, thread=thread, run_id=None,
            staged=[_card(summary="Run slack__send_message",
                          connector_id="")],
        )
        turns, _ = await ledger.list_turns(db, thread_id=thread.id)

    body = [t for t in turns if t["kind"] == "needs_you"][0]
    assert body["fix"] == "approve"
    assert body["pending_action_id"] == "pa-1"
    # The connector is recovered from the tool name when the card omits
    # it, so the turn still names the account it is about.
    assert body["account_id"] == "slack"
    assert "__" not in body["sentence"]


@pytest.mark.asyncio
async def test_the_turn_never_offers_a_tap_it_cannot_honour(monkeypatch):
    """The refusal to ship a button that lies.

    The turn asks the DISPATCHER whether an approval on this channel
    would execute. If that ever answers no — a channel removed from the
    staging set, a policy change — the card says so instead of drawing
    an Approve button whose every tap comes back a refusal.
    """
    from app.agent.automations import ledger, thread_agent

    monkeypatch.setattr(
        "app.services.connector_dispatcher.stages_writes_for_approval",
        lambda channel: False,
    )

    uid = await _mk_user()
    a = await _mk_automation_v2(uid, _v2_spec())
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        thread = await ledger.ensure_thread(db, user_id=uid,
                                            automation_id=a.id)
        await thread_agent._append_approval_turns(
            db, automation=row, thread=thread, run_id=None,
            staged=[_card()],
        )
        turns, _ = await ledger.list_turns(db, thread_id=thread.id)

    body = [t for t in turns if t["kind"] == "needs_you"][0]
    assert body["fix"] != "approve"
    assert body["pending_action_id"] is None
    assert "cannot be approved from here" in body["sentence"]


@pytest.mark.asyncio
async def test_a_thread_turn_collects_only_its_own_staged_cards(monkeypatch):
    """The collector is a ContextVar list, not an attribute on the
    shared `ToolExecutor`. One instance serves every concurrent turn in
    the process, so an instance list would let one user's staged card
    be drawn into another user's thread. A child Task inherits the
    reference, so an append inside a gathered tool call still lands."""
    import asyncio

    from app.agent import tool_executor as te

    async def _turn(tag: str, out: dict):
        with te.collect_staged_actions() as cards:
            # The append happens inside a CHILD task, which is where a
            # parallel tool call runs.
            async def _tool():
                sink = te._STAGED_ACTIONS_CTX.get()
                await asyncio.sleep(0)
                sink.append({"action_id": tag})
            await asyncio.gather(_tool())
            await asyncio.sleep(0)
            out[tag] = [c["action_id"] for c in cards]

    out: dict = {}
    await asyncio.gather(_turn("a", out), _turn("b", out))
    assert out == {"a": ["a"], "b": ["b"]}

    # And outside any collector, the sink is absent rather than global.
    assert te._STAGED_ACTIONS_CTX.get() is None


@pytest.mark.asyncio
async def test_the_thread_turn_wires_the_collector_around_the_run(monkeypatch):
    """The seam, driven. A collector that is not wrapped around
    `runner.run` collects nothing, and the failure looks exactly like
    "the model did not write anything" — silence, on the surface whose
    whole job is to stop silence."""
    from app.agent import tool_executor as te
    from app.agent.automations import ledger, thread_agent

    uid = await _mk_user()
    a = await _mk_automation_v2(uid, _v2_spec())

    class _FakeRunner:
        async def run(self, **kw):
            sink = te._STAGED_ACTIONS_CTX.get()
            assert isinstance(sink, list), "no collector around the run"
            sink.append(_card(action_id="pa-77"))

            class _R:
                text = "Slack is ready to post that — approve it below."
            return _R()

    monkeypatch.setattr(thread_agent, "_runner", lambda: _FakeRunner())
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        thread = await ledger.ensure_thread(db, user_id=uid,
                                            automation_id=a.id)
        await thread_agent.answer_in_thread(
            db, automation=row, thread=thread,
            user_text="post the summary to slack",
        )
        turns, _ = await ledger.list_turns(db, thread_id=thread.id)

    kinds = [t["kind"] for t in turns]
    # The answer first, then the card it is talking about.
    assert kinds == ["agent", "needs_you"], kinds
    assert turns[-1]["pending_action_id"] == "pa-77"


def test_the_thread_persona_forbids_calling_a_staged_write_done():
    """`[confirmation_required]` already tells the model this; the
    persona repeats it because "Posted it to Slack" over a card still
    waiting is the single worst thing this surface can produce."""
    from app.agent.automations.thread_agent import _ANSWER_RULES

    assert "it has NOT happened" in _ANSWER_RULES
    assert "Never say it is done" in _ANSWER_RULES
    assert "the same write twice" in _ANSWER_RULES


# ─── R38 follow-up: the review's findings, each with a test ──────────

def test_a_reordered_write_keeps_its_own_grant():
    """`_carry_grants_forward` paired by (connector_id, tool) in LIST
    ORDER. Two writes on the same connector and tool — post to #eng and
    post to #ops — plus an edit that drops the first handed the survivor
    the FIRST old grant. The canonical param is `{{grant.target.id}}`, so
    the message renders to the inherited grant's target and the
    dispatcher's pinned-target check passes: a silent post to the wrong
    channel. The step id is what pairs now."""
    from app.agent.automations.service import _carry_grants_forward
    old = {"version": 2, "steps": [
        {"id": "eng", "connector_id": "slack", "tool": "slack__send_message",
         "grant_id": "g-eng", "grant_target": {"id": "C_ENG"}},
        {"id": "ops", "connector_id": "slack", "tool": "slack__send_message",
         "grant_id": "g-ops", "grant_target": {"id": "C_OPS"}},
    ]}
    # The edit drops #eng and keeps #ops, grant stripped as a model's
    # round trip strips it.
    new = {"version": 2, "steps": [
        {"id": "ops", "connector_id": "slack", "tool": "slack__send_message"},
    ]}
    out = _carry_grants_forward(old, new)
    assert out["steps"][0]["grant_id"] == "g-ops"
    assert out["steps"][0]["grant_target"]["id"] == "C_OPS"


def test_a_step_that_kept_its_id_but_changed_tool_inherits_nothing():
    """An id match must agree on connector AND tool, or the pairing is
    the widening it exists to avoid."""
    from app.agent.automations.service import _carry_grants_forward
    old = {"version": 2, "steps": [
        {"id": "w", "connector_id": "slack", "tool": "slack__send_message",
         "grant_id": "g-1", "grant_target": {"id": "C1"}},
    ]}
    new = {"version": 2, "steps": [
        {"id": "w", "connector_id": "slack", "tool": "slack__set_topic"},
    ]}
    out = _carry_grants_forward(old, new)
    assert not out["steps"][0].get("grant_id")


def test_a_recreated_step_still_inherits_by_connector_and_tool():
    """Anti-vacuity: the id fallback is what keeps an ordinary rewording
    of one write from losing its grant — the whole reason this function
    exists."""
    from app.agent.automations.service import _carry_grants_forward
    old = {"version": 2, "steps": [
        {"id": "old-id", "connector_id": "slack", "tool": "slack__send_message",
         "grant_id": "g-1", "grant_target": {"id": "C1"}},
    ]}
    new = {"version": 2, "steps": [
        {"id": "new-id", "connector_id": "slack", "tool": "slack__send_message"},
    ]}
    out = _carry_grants_forward(old, new)
    assert out["steps"][0]["grant_id"] == "g-1"


def test_the_head_note_and_the_terminal_frame_tell_one_story():
    """Two tables disagreed on two of six reachable statuses. `skipped`
    is the common one — `confirm.py` finalizes a declined or expired
    confirm card as `outcome="skipped"` — so a user who said no watched
    the divider flip to TRIED from the live frame and revert to RAN on
    the next refetch, which is the exact symptom flipping the head note
    was added to end."""
    from app.agent.automations.run_v3 import head_note_stamp
    assert head_note_stamp("skipped") == "ran"
    assert head_note_stamp("completed") == "ran"
    assert head_note_stamp("failed") == "tried"
    assert head_note_stamp("partial") == "ran"
    assert head_note_stamp("partial", {"failed_sources": [{"account_id": "gmail"}]}) \
        == "needs_you"
    # None means "the divider keeps its own note" — a stop note, or the
    # STARTED a superseded run leaves for the run that replaced it. The
    # frame sends it as None; a client that reads absence as "ran" would
    # rewrite a stopped run's divider as if it had finished.
    assert head_note_stamp("stopped_by_user") is None
    assert head_note_stamp("superseded") is None
    assert head_note_stamp("waiting_on_user") is None


def test_an_update_waives_the_variable_rule_on_purpose_and_it_is_not_caught_later():
    """A RECORDED GAP, pinned so it cannot be closed by accident in
    either direction.

    `template_mode=True, template_vars=None` waives the grant rule AND
    the undeclared-variable rule, and both waivers are deliberate: a spec
    mid-setup carries variables whose questions the user has not answered
    yet — the state run-now used to answer 500 about (R37, the founder's
    dead Run button). `parse_spec_live`, and therefore ARM, waives it the
    same way.

    The gap is that nothing downstream enforces it either: `render_value`
    resolves a missing path to "", so an automation armed with a dangling
    `{{var.summary}}` fires and posts " today". Tightening the update call
    would reject the mid-setup state the doctrine exists to allow, so the
    fix belongs at dispatch. This test asserts BOTH halves — the waiver
    and the strictness it is a waiver FROM — so whoever closes the gap
    sees exactly what they are changing.
    """
    from app.agent.automations.spec import SpecError
    from app.agent.automations.spec_v2 import validate_spec_v2
    registry = {
        "jira": {
            "connector_id": "jira",
            "push": False, "poll": True, "floor_s": 300,
            "scopes_write_by_action": {},
            "target_param_by_action": {},
            "events": [{
                "key": "issue_created", "description": "",
                "source_tool": "jira__search_issues",
                "poll_args": {"jql": "created >= -1d"},
                "params_required": [],
                "items_path": "issues", "dedupe_field": "key",
                "fields": {"key": "key", "summary": "summary"},
            }],
        },
    }
    spec = {
        "version": 2, "name": "x", "mode": "auto",
        "trigger": {"sources": [
            {"id": "sched", "mode": "schedule",
             "schedule": {"cron_local": "0 8 * * 1-5"}},
        ]},
        "steps": [{"id": "s1", "connector_id": "jira",
                   "tool": "jira__search_issues",
                   "params": {"jql": "{{var.summary}} today"},
                   "on_error": "skip"}],
    }
    # The update path's arguments, verbatim: accepted.
    validate_spec_v2(spec, registry, template_mode=True, template_vars=None)
    # …and the rule it is a waiver from is real — this is what would fire
    # if the waiver were ever narrowed.
    with pytest.raises(SpecError) as ei:
        validate_spec_v2(spec, registry, template_mode=True,
                         template_vars=set())
    assert "unknown_variable" in str(ei.value)
