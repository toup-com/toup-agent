# agent-mode: automations/automation_threads/_turns are AGENT_ONLY.
"""R42 §5.2 / §5.3 — the connector popup's two missing halves.

  §5.2 NARROW IT        — per-account read filters. Offered only where
                          the connector can express them AND this
                          automation runs a step they compose into;
                          composed into the provider call itself, which
                          is the exact opposite of what a PIN does
                          (`_apply_focus_scope`: pins rank, they never
                          filter). Jira's ORDER BY stays trailing.
  §5.3 TELL ME THE      — instant triggers are real `trigger.sources`
       MOMENT             lanes, gated on the live manifest. The
                          schedule survives beside them, the last lane
                          of an event-only automation cannot be taken
                          off, and an event that names a place needs
                          the pin that names it.

The registry here is the REAL one, loaded from the shipped manifests:
this half of the design is bounded by what the platform actually
declares (eight events, Slack none), and a fixture would let the code
claim capabilities that do not exist.
"""

import json
import uuid
from datetime import datetime, timezone

import pytest

from app.agent.automations import compiler, workflow as wf
from app.agent.automations.executor_v2 import _apply_read_filters
from app.agent.automations.spec import validate_spec
from app.db.database import async_session_maker
from app.db.models import Automation, User
from app.services.connector_registry import ConnectorRegistry

from tests.test_workflow_api import _edited_notes  # noqa: F401 — the
# EDITED-note counter; one workflow write means one divider.


def _real_registry() -> dict:
    reg = ConnectorRegistry()
    reg.load_all()
    return {e["connector_id"]: e for e in reg.automation_registry()}


REGISTRY = _real_registry()
CLOCK = {"now": datetime(2026, 8, 31, 9, 30, tzinfo=timezone.utc)}


@pytest.fixture(autouse=True)
def _offline_platform(monkeypatch):
    async def _conn_state(user_id):
        return {
            cid: {"connector_id": cid, "connected": True,
                  "status": "active", "scopes": ["r"], "account": f"{cid} acct"}
            for cid in ("gmail", "slack", "github", "jira", "outlook")
        }

    async def _registry(user_id, force=False):
        return REGISTRY

    async def _templates(user_id):
        return []

    async def _grant(user_id, grant_id):
        return {"id": grant_id, "status": "approved", "connector_id": "slack",
                "tool_name": "slack__send_message",
                "target": {"kind": "channel", "id": "C-PIN",
                           "label": "#platform"},
                "mode": "auto"}

    for name, fn in (("fetch_connection_state", _conn_state),
                     ("fetch_registry", _registry),
                     ("fetch_templates", _templates),
                     ("fetch_grant", _grant)):
        monkeypatch.setattr(f"app.agent.automations.registry.{name}", fn)


def _read_step(sid, connector_id, tool, params):
    return {
        "id": sid, "connector_id": connector_id, "tool": tool,
        "params": params,
        "collect": {"items_path": "items", "fields": {"t": "title"},
                    "format": "{{item.t}}", "empty_text": "none"},
        "on_error": "skip",
    }


_POST = {
    "id": "post", "connector_id": "slack", "tool": "slack__send_message",
    "params": {"channel": "{{grant.target.id}}", "text": "{{steps.mail.text}}"},
    "grant_id": "g-1",
    "grant_target": {"kind": "channel", "id": "C-PIN", "label": "#platform"},
}


def _spec(*, sources=None, steps=None, **over):
    spec = {
        "version": 2, "name": "Popup brief", "mode": "auto",
        "trigger": {"sources": sources or [
            {"id": "sched", "mode": "schedule",
             "schedule": {"cron_local": "0 8 * * 1-5"}},
        ]},
        "steps": steps or [
            _read_step("mail", "gmail", "gmail__list_messages",
                       {"query": "in:inbox", "max_results": 10}),
            _POST,
        ],
    }
    spec.update(over)
    return validate_spec(spec, REGISTRY)


async def _mk_user() -> str:
    uid = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(id=uid, email=f"{uid[:8]}@example.com",
                    hashed_password="x", name="Popup"))
        await db.commit()
    return uid


async def _mk_automation(uid, vspec):
    async with async_session_maker() as db:
        a = Automation(user_id=uid, name=vspec.name, status="draft",
                       spec_json=json.dumps(vspec.raw, sort_keys=True),
                       trigger_mode=vspec.trigger_mode,
                       connector_id=vspec.trigger_connector_id)
        db.add(a)
        await db.flush()
        await compiler.compile_bindings(db, a, vspec)
        await db.commit()
        return a


# ── §5.2 the composition ────────────────────────────────────────────

def test_a_filter_narrows_the_provider_call_where_a_pin_never_would():
    # The same account, the same step: a PIN leaves a broad query alone
    # (R42's own reversal, `_apply_focus_scope`), a FILTER composes.
    from app.agent.automations.executor_v2 import _apply_focus_scope
    params = {"query": "in:inbox", "max_results": 10}
    pinned = _apply_focus_scope(
        "gmail", "gmail__list_messages", params,
        [{"kind": "person", "id": "dana@x.com", "label": "Dana"}])
    assert pinned["query"] == "in:inbox"
    narrowed = _apply_read_filters(
        "gmail", "gmail__list_messages", params, ["unread", "me"], CLOCK)
    assert narrowed["query"] == "in:inbox to:me is:unread"
    assert params["query"] == "in:inbox"  # the step's own params stand


def test_jira_keeps_its_order_by_trailing_the_whole_query():
    jql = ("assignee = currentUser() AND statusCategory != Done "
           "AND (duedate <= 7d OR priority in (Highest, High)) "
           "ORDER BY duedate ASC, updated DESC")
    out = _apply_read_filters("jira", "jira__search_issues", {"jql": jql},
                              ["priority", "day"], CLOCK)["jql"]
    where, _, order = out.partition("ORDER BY")
    assert order.strip() == "duedate ASC, updated DESC"
    assert "ORDER BY" not in where
    # The clause inside the OR group narrows nothing, so it is ANDed
    # for real — that is the whole chip.
    assert "AND priority in (Highest, High) AND updated >= -1d" in where
    # One pass, one wrap: four filters must not nest four parens.
    assert where.count("(assignee") == 1


def test_a_term_already_anded_is_not_repeated():
    out = _apply_read_filters(
        "gmail", "gmail__list_messages", {"query": "is:unread newer_than:1d"},
        ["unread", "day"], CLOCK)
    assert out["query"] == "is:unread newer_than:1d"


def test_outlook_read_state_and_window_are_params_not_query_text():
    out = _apply_read_filters("outlook", "outlook__list_messages",
                              {"max_results": 5}, ["unread", "day"], CLOCK)
    assert out["is_read"] is False
    assert out["since"] == "2026-08-30T09:30:00+00:00"
    assert "query" not in out
    # …and Graph gets ONE legal query out of it, ordered newest first.
    from app.connectors.outlook.provider import _list_messages_params
    params, scan, _ = _list_messages_params(out)
    assert params["$filter"] == (
        "receivedDateTime ge 2026-08-30T09:30:00Z and isRead eq false")
    assert params["$orderby"] == "receivedDateTime desc" and scan is None


def test_a_filter_never_widens_a_bound_the_spec_already_set():
    out = _apply_read_filters(
        "slack", "slack__read_messages",
        {"channel": "C1", "oldest": "9999999999"}, ["day"], CLOCK)
    assert out["oldest"] == "9999999999"


def test_a_filter_is_not_composed_into_a_tool_it_cannot_express():
    # The table's own `tools` list is the one answer to this question.
    out = _apply_read_filters("gmail", "gmail__create_draft",
                              {"to": "x@y.z"}, ["unread", "day"], CLOCK)
    assert out == {"to": "x@y.z"}
    # …and a run with no clock drops the time filters rather than
    # inventing a bound.
    out = _apply_read_filters("outlook", "outlook__list_messages",
                              {"max_results": 5}, ["unread", "day"], {})
    assert out == {"max_results": 5, "is_read": False}


# ── §5.2 the payload and the write ──────────────────────────────────

@pytest.mark.asyncio
async def test_filters_are_offered_only_where_the_step_can_express_them():
    uid = await _mk_user()
    a = await _mk_automation(uid, _spec())
    async with async_session_maker() as db:
        payload = await wf.workflow_payload(db, automation=a, user_id=uid)
    by_id = {e["account_id"]: e for e in payload["accounts"]}
    assert [f["id"] for f in by_id["gmail"]["filters_available"]] == [
        "me", "unread", "no_promos", "day"]
    assert by_id["gmail"]["filters"] == []
    # Slack is here to POST. It narrows nothing, so it offers nothing
    # and the app draws no section.
    assert by_id["slack"]["filters_available"] == []


@pytest.mark.asyncio
async def test_every_account_carries_all_four_keys_even_when_empty():
    """Absent is not empty, and the app can tell them apart.

    §5.3 renders a SENTENCE for an empty trigger list — "X has nothing it can
    tell you the moment it happens" — which is a claim about the connector.
    The app only makes it when the payload actually said so, and falls back to
    "has not told me yet" when the key is missing. That fallback exists for
    OLD backends; this payload must never trigger it, so every account entry
    carries all four keys whether or not it has anything to put in them.
    """
    uid = await _mk_user()
    a = await _mk_automation(uid, _spec())
    async with async_session_maker() as db:
        payload = await wf.workflow_payload(db, automation=a, user_id=uid)
    assert payload["accounts"], "fixture must produce at least one account"
    for e in payload["accounts"]:
        for k in ("filters", "filters_available", "triggers", "triggers_available"):
            assert k in e, f"{e['account_id']} is missing {k}"
            assert isinstance(e[k], list), f"{e['account_id']}.{k} is not a list"


@pytest.mark.asyncio
async def test_set_filters_persists_through_the_validator_and_answers_all_three():
    uid = await _mk_user()
    a = await _mk_automation(uid, _spec())
    async with async_session_maker() as db:
        out = await wf.set_filters(db, automation=a, user_id=uid,
                                   connector_id="gmail",
                                   filters=["day", "unread"])
    # Table order, not the caller's.
    assert out["filters"] == ["unread", "day"]
    assert out["sentence"] == "In Gmail it now reads Unread only and Last 24 hours."
    entry = next(e for e in out["workflow"]["accounts"]
                 if e["account_id"] == "gmail")
    assert entry["filters"] == ["unread", "day"]
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        assert json.loads(row.spec_json)["filters"] == {"gmail": ["unread", "day"]}
    # And they reach the run through the spec, not a side table.
    from app.agent.automations.spec_v2 import validate_spec_v2
    vspec = validate_spec_v2(json.loads(row.spec_json), REGISTRY,
                             template_mode=True)
    assert vspec.filters == {"gmail": ["unread", "day"]}
    # One write, one divider on the thread — a filter changes what the
    # automation does, and `_persist_spec` stamps it exactly once.
    assert await _edited_notes(a.id) == 1


@pytest.mark.asyncio
async def test_clearing_the_filters_drops_the_key_rather_than_storing_empty():
    uid = await _mk_user()
    a = await _mk_automation(uid, _spec())
    async with async_session_maker() as db:
        await wf.set_filters(db, automation=a, user_id=uid,
                             connector_id="gmail", filters=["unread"])
        out = await wf.set_filters(db, automation=a, user_id=uid,
                                   connector_id="gmail", filters=[])
    assert out["filters"] == [] and out["sentence"] == "It reads all of Gmail again."
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        assert "filters" not in json.loads(row.spec_json)


@pytest.mark.asyncio
async def test_an_unknown_filter_is_refused_with_a_sentence():
    uid = await _mk_user()
    a = await _mk_automation(uid, _spec())
    async with async_session_maker() as db:
        with pytest.raises(wf.WorkflowError) as e:
            await wf.set_filters(db, automation=a, user_id=uid,
                                 connector_id="gmail", filters=["p1_and_p2"])
    assert e.value.code == "unknown_filter"
    assert e.value.sentence == "Gmail cannot narrow a read that way."
    async with async_session_maker() as db:
        with pytest.raises(wf.WorkflowError) as e:
            await wf.set_filters(db, automation=a, user_id=uid,
                                 connector_id="notion", filters=["day"])
    assert e.value.code == "not_member"


@pytest.mark.asyncio
async def test_a_stored_filter_stays_offerable_even_if_the_step_moves():
    """A step edited afterwards must never strand a filter the user can
    see the effect of but cannot take off."""
    uid = await _mk_user()
    a = await _mk_automation(uid, _spec())
    async with async_session_maker() as db:
        await wf.set_filters(db, automation=a, user_id=uid,
                             connector_id="gmail", filters=["unread"])
        row = await db.get(Automation, a.id)
        raw = json.loads(row.spec_json)
        raw["steps"][0] = _read_step("mail", "gmail", "gmail__search_threads",
                                     {"query": "x"})
        # search_threads expresses `unread` too, so swap to the one tool
        # that expresses nothing: a draft.
        raw["steps"] = [dict(_POST, id="post")]
        raw["steps"][0]["connector_id"] = "gmail"
        raw["steps"][0]["tool"] = "gmail__create_draft"
        raw["steps"][0]["params"] = {"to": "a@b.c", "subject": "s", "body": "b"}
        assert wf.available_filters(raw, "gmail") == [
            {"id": "unread", "label": "Unread only"}]


# ── §5.3 the manifest is the whole inventory ────────────────────────

def test_triggers_available_is_the_manifest_and_nothing_else():
    # Gmail is still ONE row. R43 §7 names four, and the other three
    # stay undeclared until the manifest carries the `default_filter`
    # that narrows them — all four ride the same `users.watch` push, so
    # declaring them unnarrowed would mean four runs per incoming mail.
    assert [t["id"] for t in wf.available_triggers({"version": 2}, REGISTRY, "gmail")] == [
        "email_received"]
    # Slack was the empty one this test was written to pin. R43 gave it
    # five: `channel_message` still names a pinned place, and four more
    # that resolve the owner's identity at call time.
    assert [t["id"] for t in wf.available_triggers({"version": 2}, REGISTRY, "slack")] == [
        "channel_message", "mentioned", "dm_arrived", "thread_moved",
        "oncall_message"]
    labels = {t["id"]: t["label"]
              for t in wf.available_triggers({"version": 2}, REGISTRY, "jira")}
    assert labels == {
        "issue_created": "A ticket is created in that project",
        "issue_assigned": "A ticket is assigned to me",
        "issue_reopened": "My ticket is reopened",
        "p1_raised": "A P1 is raised",
    }
    # A connector with no `automation.events` block is still the honest
    # empty answer, and the app says so in words.
    assert wf.available_triggers({"version": 2}, REGISTRY, "docs") == []
    # 27 events across the shipped manifests — the design lists 31, and
    # the payload never invents the difference. The four it does not
    # name are recorded, each with its reason, in `test_r43_catalog`'s
    # `_R43_EVENTS`; this number is the count of what a user can
    # actually tap.
    total = sum(len(wf.available_triggers({"version": 2}, REGISTRY, c)) for c in REGISTRY)
    assert total == 27


@pytest.mark.asyncio
async def test_an_instant_lane_joins_the_schedule_it_does_not_replace_it():
    uid = await _mk_user()
    a = await _mk_automation(uid, _spec())
    async with async_session_maker() as db:
        out = await wf.set_triggers(db, automation=a, user_id=uid,
                                    connector_id="gmail",
                                    triggers=["email_received"])
    assert out["triggers"] == ["email_received"]
    # R43 §7 fixed this row's wording. The sentence is composed from
    # `EVENT_LABELS`, not from the manifest description, so the label
    # the picker draws and the sentence the write returns cannot drift.
    assert out["sentence"] == (
        "I will tell you the moment mail addressed to me arrives.")
    async with async_session_maker() as db:
        raw = json.loads((await db.get(Automation, a.id)).spec_json)
    modes = {s["mode"] for s in raw["trigger"]["sources"]}
    assert modes == {"schedule", "push"}
    # The canvas still shows the cron it still runs on — reading the
    # event first would hide the schedule sheet behind one tap.
    assert wf.trigger_block(raw)["kind"] == "schedule"
    entry = next(e for e in out["workflow"]["accounts"]
                 if e["account_id"] == "gmail")
    assert entry["triggers"] == ["email_received"]


@pytest.mark.asyncio
async def test_turning_a_lane_off_leaves_the_schedule_and_the_other_lanes():
    uid = await _mk_user()
    a = await _mk_automation(uid, _spec())
    async with async_session_maker() as db:
        await wf.set_triggers(db, automation=a, user_id=uid,
                              connector_id="gmail",
                              triggers=["email_received"])
        out = await wf.set_triggers(db, automation=a, user_id=uid,
                                    connector_id="gmail", triggers=[])
    assert out["triggers"] == []
    async with async_session_maker() as db:
        raw = json.loads((await db.get(Automation, a.id)).spec_json)
    assert [s["mode"] for s in raw["trigger"]["sources"]] == ["schedule"]


@pytest.mark.asyncio
async def test_the_last_lane_of_an_event_only_automation_cannot_be_taken_off():
    uid = await _mk_user()
    vspec = _spec(sources=[{
        "id": "inbox", "mode": "push", "connector_id": "gmail",
        "event": "email_received", "dedupe_key": "event.message_id",
    }])
    a = await _mk_automation(uid, vspec)
    async with async_session_maker() as db:
        with pytest.raises(wf.WorkflowError) as e:
            await wf.set_triggers(db, automation=a, user_id=uid,
                                  connector_id="gmail", triggers=[])
    assert e.value.code == "last_trigger"
    async with async_session_maker() as db:
        raw = json.loads((await db.get(Automation, a.id)).spec_json)
    assert len(raw["trigger"]["sources"]) == 1


@pytest.mark.asyncio
async def test_an_event_that_names_a_place_needs_the_pin_that_names_it():
    uid = await _mk_user()
    steps = [_read_step("issues", "github", "github__list_issues",
                        {"owner": "toup", "repo": "platform", "state": "open"}),
             _POST]
    a = await _mk_automation(uid, _spec(steps=steps))
    async with async_session_maker() as db:
        with pytest.raises(wf.WorkflowError) as e:
            await wf.set_triggers(db, automation=a, user_id=uid,
                                  connector_id="github",
                                  triggers=["issue_opened"])
        assert e.value.code == "needs_pin"
        assert "repository" in e.value.sentence
        # The pin the canvas already writes is where that place lives.
        await wf.add_focus(db, automation=a, user_id=uid, account_id="github",
                           kind="repo", target_id="toup/platform",
                           label="toup/platform")
    async with async_session_maker() as db:
        # Fresh, exactly as the route loads it between two requests.
        fresh = await db.get(Automation, a.id)
        out = await wf.set_triggers(db, automation=fresh, user_id=uid,
                                    connector_id="github",
                                    triggers=["issue_opened"])
    assert out["triggers"] == ["issue_opened"]
    async with async_session_maker() as db:
        raw = json.loads((await db.get(Automation, a.id)).spec_json)
    src = next(s for s in raw["trigger"]["sources"] if s.get("event"))
    assert src["params"] == {"owner": "toup", "repo": "platform"}
    assert src["mode"] == "poll" and src["dedupe_key"] == "event.number"


@pytest.mark.asyncio
async def test_an_event_the_manifest_does_not_declare_is_refused():
    uid = await _mk_user()
    a = await _mk_automation(uid, _spec())
    async with async_session_maker() as db:
        with pytest.raises(wf.WorkflowError) as e:
            await wf.set_triggers(db, automation=a, user_id=uid,
                                  connector_id="slack",
                                  triggers=["message_posted"])
    assert e.value.code == "unknown_trigger"
    assert e.value.sentence == (
        "Slack cannot tell you the moment that happens.")
