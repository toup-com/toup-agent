# agent-mode: the delivery-availability half persists real automations
# through `service.update_automation` (automations/_threads are
# AGENT_ONLY). The scope-pass half is pure and rides along.
"""R43 repair — the platform's claims, made true.

Three defects, all of the round's own forbidden shape (a control that
does not do what it says), found by the adversarial pass after ship:

  13 + 2  SOURCES WERE STORED AND NEVER READ. `set_sources` wrote
          `spec["sources"]`, the validator canonicalised it, the payload
          served it and the canvas drew a scope line from it — and no
          run consulted it. The sheet said "Nothing picked — I will skip
          Gmail on the next run", about EVERY account (nothing has ever
          been picked), and the run read Gmail exactly as before. The
          run scopes now (`executor_v2._apply_source_scope`), an empty
          pick means "read the account as before" because that is the
          state every automation is in, and a pick the run cannot
          express is refused at the writer rather than stored.

  6 + 14  A DELIVERY CHANNEL WAS OFFERED ON OAUTH STATE ALONE.
          `deliver._deliver_one` additionally requires an APPROVED
          grant on this automation whose tool is the channel's writer
          and whose pinned target is the USER — so all five connector
          channels were pickable and refused on every run, and
          `slack_dm` was guaranteed to fail for anyone whose only Slack
          grant is pinned at a channel.

  15      A NARRATION RETRY THAT DIED THREW THE FIRST ATTEMPT AWAY.
          `narrate_run` reports every result unservable when its retry
          raises; the drafts it hands back are attempt one's, already
          validated, often servable.
"""

import json
import uuid
from types import SimpleNamespace

import pytest

from app.agent.automations import compiler, workflow as wf
from app.agent.automations.spec import validate_spec
from app.agent.automations.executor_v2 import (
    _apply_source_scope, _execute_read_step, _merge_read_contents,
    _recheck_unservable, source_scope_max, source_scope_supports,
)
from app.db.database import async_session_maker
from app.db.models import Automation, User
from app.services.connector_registry import ConnectorRegistry


# ── §2.2 the scope pass ─────────────────────────────────────────────

def test_a_picked_gmail_label_narrows_the_query():
    one = _apply_source_scope(
        "gmail", "gmail__list_messages",
        {"query": "in:inbox newer_than:1d", "max_results": 10}, ["STARRED"])
    assert len(one) == 1
    assert one[0]["query"] == "in:inbox newer_than:1d is:starred"
    # Untouched params ride along.
    assert one[0]["max_results"] == 10

    # Two picks are a CHOICE, so they are grouped before they are ANDed
    # — a bare append would attach to the last branch only.
    two = _apply_source_scope(
        "gmail", "gmail__list_messages", {"query": "in:inbox"},
        ["inbox", "IMPORTANT"])
    assert two[0]["query"] == "in:inbox (in:inbox OR is:important)"

    # The contract's user-label form is already a Gmail term, quoted
    # because a label name may carry spaces.
    named = _apply_source_scope(
        "gmail", "gmail__search_threads", {"query": "to:me"},
        ["label:Big Clients"])
    assert named[0]["query"] == 'to:me label:"Big Clients"'


def test_a_gmail_id_no_query_can_select_is_not_scopable():
    # `label:INBOX` matches NOTHING in Gmail's grammar (`label:` takes a
    # name), so a label id would empty the brief in silence. Refused at
    # the writer, ignored by the run — never guessed at.
    assert source_scope_supports("gmail", "INBOX")
    assert not source_scope_supports("gmail", "Label_12")
    assert not source_scope_supports("gmail", 'label:say "hi"')
    assert _apply_source_scope(
        "gmail", "gmail__list_messages", {"query": "in:inbox"},
        ["Label_12"]) == [{"query": "in:inbox"}]


def test_picked_jira_projects_and_the_order_by_stays_trailing():
    out = _apply_source_scope(
        "jira", "jira__search_issues",
        {"jql": "assignee = currentUser() ORDER BY updated DESC"},
        ["ENG", "OPS"])
    assert out[0]["jql"] == (
        '(assignee = currentUser()) AND project in ("ENG", "OPS") '
        "ORDER BY updated DESC")
    assert not source_scope_supports("jira", 'ENG" OR project = "X')


def test_a_target_shaped_read_becomes_one_call_per_place():
    sets = _apply_source_scope(
        "slack", "slack__read_messages", {"limit": 20},
        ["C_PLATFORM", "C_ONCALL"])
    assert [s["channel"] for s in sets] == ["C_PLATFORM", "C_ONCALL"]
    assert all(s["limit"] == 20 for s in sets)

    # An explicit pick OVERRIDES a channel the spec (or a pin) named:
    # a pin ranks and may never filter, while a picked source is the
    # user answering "which places" out loud.
    named = _apply_source_scope(
        "slack", "slack__read_messages", {"channel": "C_OLD"}, ["C_NEW"])
    assert [s["channel"] for s in named] == ["C_NEW"]


def test_github_splits_owner_and_repo_and_refuses_anything_else():
    sets = _apply_source_scope(
        "github", "github__list_issues", {"state": "open"}, ["acme/api"])
    assert sets == [{"state": "open", "owner": "acme", "repo": "api"}]
    assert not source_scope_supports("github", "api")
    assert _apply_source_scope(
        "github", "github__list_issues", {"state": "open"},
        ["api"]) == [{"state": "open"}]


def test_the_pass_is_total():
    p = {"query": "in:inbox"}
    # no picks, a connector with no scopable read, a tool the table
    # does not reach, malformed params — all untouched, one call.
    assert _apply_source_scope("gmail", "gmail__list_messages", p, []) == [p]
    assert _apply_source_scope("notion", "notion__search", p, ["x"]) == [p]
    assert _apply_source_scope("gmail", "gmail__send", p, ["inbox"]) == [p]
    assert _apply_source_scope("gmail", "gmail__list_messages", None,
                               ["inbox"]) == [None]


def test_the_cap_the_writer_enforces_is_the_cap_the_run_reads():
    # gmail/jira compose a set into one query; every other read takes
    # ONE place per call, so its cap is the fan-out bound.
    assert source_scope_max("gmail") == 10
    assert source_scope_max("jira") == 10
    assert source_scope_max("slack") == 4
    assert source_scope_max("notion") == 0
    sets = _apply_source_scope(
        "slack", "slack__read_messages", {},
        ["a", "b", "c", "d", "e", "f"])
    assert len(sets) == source_scope_max("slack")


def test_merging_touches_only_the_step_s_own_items_path():
    merged = _merge_read_contents(
        [{"messages": [1, 2], "next_cursor": "one"},
         {"messages": [3], "next_cursor": "two"}], "messages")
    assert merged["messages"] == [1, 2, 3]
    # One of the calls' cursors, never a sum of them — there is no
    # honest way to add two providers' pagination together.
    assert merged["next_cursor"] == "one"
    nested = _merge_read_contents(
        [{"d": {"items": [1]}}, {"d": {"items": [2]}}], "d.items")
    assert nested["d"]["items"] == [1, 2]
    assert _merge_read_contents([], "items") == {}
    # No path to merge at: the first answer stands rather than a guess.
    assert _merge_read_contents([{"a": 1}, {"a": 2}], "") == {"a": 1}


@pytest.mark.asyncio
async def test_the_run_really_reads_every_picked_place(monkeypatch):
    """The end-to-end seam that did not exist. Before the repair a
    picked set changed NOTHING about the calls a step made."""
    from app.agent.automations import registry as reg

    calls = []

    async def _dispatch(user_id, *, connector_id, tool_name, tool_input,
                        automation_id=None, **kw):
        calls.append(dict(tool_input))
        chan = tool_input.get("channel")
        return {"kind": "ok", "content": json.dumps(
            {"messages": [{"title": f"{chan}-1"}, {"title": f"{chan}-2"}]})}

    monkeypatch.setattr(reg, "dispatch_via_platform", _dispatch)

    step = SimpleNamespace(
        id="chat", connector_id="slack", tool="slack__read_messages",
        params_template={"limit": 20},
        collect={"items_path": "messages", "fields": {"t": "title"},
                 "format": "{{item.t}}", "limit": 10, "empty_text": "none"},
        on_error="continue",
    )
    automation = SimpleNamespace(id="a1", user_id="u1")

    out = await _execute_read_step(
        automation, step,
        {"_account_sources": {"slack": ["C_PLATFORM", "C_ONCALL"]}},
    )
    assert [c.get("channel") for c in calls] == ["C_PLATFORM", "C_ONCALL"]
    assert out["count"] == 4
    assert out["lines"][0] == "C_PLATFORM-1"
    assert out["lines"][-1] == "C_ONCALL-2"


@pytest.mark.asyncio
async def test_one_dead_place_fails_the_step_rather_than_half_reading_it(
        monkeypatch):
    from app.agent.automations import registry as reg

    async def _dispatch(user_id, *, connector_id, tool_name, tool_input,
                        automation_id=None, **kw):
        if tool_input.get("channel") == "C_GONE":
            return {"kind": "error", "message": "channel_not_found"}
        return {"kind": "ok", "content": json.dumps({"messages": [{}]})}

    monkeypatch.setattr(reg, "dispatch_via_platform", _dispatch)
    step = SimpleNamespace(
        id="chat", connector_id="slack", tool="slack__read_messages",
        params_template={}, collect={"items_path": "messages", "fields": {},
                                     "format": "", "empty_text": ""},
        on_error="continue")
    with pytest.raises(RuntimeError):
        await _execute_read_step(
            SimpleNamespace(id="a1", user_id="u1"), step,
            {"_account_sources": {"slack": ["C_OK", "C_GONE"]}})


# ── §15 the narration retry that died ───────────────────────────────

def _brief_record():
    return {"vocabulary": "brief", "status": "completed",
            "automation": {"title": "Morning brief"}, "steps": []}


def test_a_retry_that_died_keeps_the_first_attempt_s_ranking():
    """`narrate_run` answers "every result" when the retry raises. The
    drafts in hand are attempt one's — validated, and kept by
    `unservable_results` when only a row or two was flagged."""
    from app.agent.automations.narrator import (
        unservable_results, validate_drafts,
    )
    record = _brief_record()
    good = [{"kind": "result", "title": "Your morning",
             "vocabulary": "brief",
             "groups": [{"rank": 1, "label": "DO FIRST · BLOCKS OTHERS",
                         "tone": "danger",
                         "rows": [{"text": "Dana is waiting", "sub": "on the "
                                   "retry flag", "item_refs": ["it_1"]}]}]}]
    # Whatever this record makes of that draft, the two answers must
    # agree — and the throw path's "every result" must not win.
    honest = unservable_results(good, validate_drafts(good, record))
    assert _recheck_unservable(good, record, {0}) == honest

    # A result the validator really cannot serve stays dropped.
    empty = [{"kind": "result", "title": "Your morning",
              "vocabulary": "brief", "groups": []}]
    assert _recheck_unservable(empty, record, {0}) == {0}

    # Nothing came back at all: there is nothing to re-judge.
    assert _recheck_unservable([], record, {0, 1}) == {0, 1}
    # Nothing was reported: no second validation pass is paid for.
    assert _recheck_unservable(good, record, set()) == set()


# ── §2.1 delivery availability ──────────────────────────────────────

def _registry() -> dict:
    reg = ConnectorRegistry()
    reg.load_all()
    return {e["connector_id"]: e for e in reg.automation_registry()}


REGISTRY = _registry()

_CONNECTED = {
    cid: {"connector_id": cid, "connected": True, "status": "active",
          "scopes": ["r"], "account": f"{cid}@acme.com"}
    for cid in ("gmail", "slack", "jira", "calendar", "notion")
}


@pytest.fixture
def platform(monkeypatch):
    """No platform. The grants this user has are the fixture."""
    state = {"connections": dict(_CONNECTED), "grants": {}}

    async def _conn_state(user_id):
        return state["connections"]

    async def _registry_fn(user_id, force=False):
        return REGISTRY

    async def _templates(user_id):
        return []

    async def _grant(user_id, grant_id):
        return state["grants"].get(grant_id)

    for name, fn in (("fetch_connection_state", _conn_state),
                     ("fetch_registry", _registry_fn),
                     ("fetch_templates", _templates),
                     ("fetch_grant", _grant)):
        monkeypatch.setattr(f"app.agent.automations.registry.{name}", fn)
    from app.config import settings
    monkeypatch.setattr(settings, "whatsapp_session_status", "not_linked",
                        raising=False)
    monkeypatch.setattr(settings, "telegram_bot_token", "", raising=False)
    wf.invalidate_sources_cache()
    yield state
    wf.invalidate_sources_cache()


def _read_step(sid, connector_id, tool, params):
    return {"id": sid, "connector_id": connector_id, "tool": tool,
            "params": params,
            "collect": {"items_path": "items", "fields": {"t": "title"},
                        "format": "{{item.t}}", "empty_text": "none"},
            "on_error": "skip"}


def _spec(steps=None):
    return validate_spec({
        "version": 2, "name": "Morning brief", "mode": "auto",
        "trigger": {"sources": [
            {"id": "sched", "mode": "schedule",
             "schedule": {"cron_local": "0 8 * * 1-5"}}]},
        "steps": steps or [
            _read_step("mail", "gmail", "gmail__list_messages",
                       {"query": "in:inbox", "max_results": 10})],
    }, REGISTRY)


async def _mk_user() -> str:
    uid = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(id=uid, email=f"{uid}@t.test", name="T",
                    hashed_password="x"))
        await db.commit()
    return uid


async def _mk_automation(uid: str, vspec) -> Automation:
    async with async_session_maker() as db:
        row = Automation(user_id=uid, name=vspec.name, status="draft",
                         spec_json=json.dumps(vspec.raw, sort_keys=True),
                         trigger_mode=vspec.trigger_mode,
                         connector_id=vspec.trigger_connector_id)
        db.add(row)
        await db.flush()
        await compiler.compile_bindings(db, row, vspec)
        await db.commit()
        return row


def _by_id(rows):
    return {r["id"]: r for r in rows}


@pytest.mark.asyncio
async def test_a_channel_with_no_write_grant_is_not_offered(platform):
    uid = await _mk_user()
    a = await _mk_automation(uid, _spec())
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        rows = _by_id((await wf.workflow_payload(
            db, automation=row, user_id=uid))["delivery"]
            ["channels_available"])

    # Gmail is CONNECTED and the brief has no Gmail write step, so
    # `deliver._grant_for` would raise `grant_missing` on every run.
    assert rows["gmail_draft"]["available"] is False
    assert "allowed to write" in rows["gmail_draft"]["reason"]
    # And the reason says a permission can be ASKED for, rather than
    # leaving the user with a dead row and no way forward.
    assert "request permission" in rows["gmail_draft"]["reason"]
    # This app needs neither an account nor a grant.
    assert rows["app"]["available"] and rows["app"]["reason"] is None
    # A disconnected account keeps the TRUER sentence — sign in first.
    assert rows["outlook_mail"]["reason"] == "Outlook is not connected."


@pytest.mark.asyncio
async def test_a_grant_pinned_at_a_channel_cannot_address_the_user(platform):
    """The founder's case exactly: his one Slack grant is pinned to
    `#all-toup`, a `C…` id, which `_check_addressed_to_the_user`
    forbids — so `slack_dm` was offered and refused on every run."""
    uid = await _mk_user()
    platform["grants"]["g_slack"] = {
        "id": "g_slack", "status": "approved", "connector_id": "slack",
        "tool_name": "slack__send_message",
        "target": {"kind": "channel", "id": "C_ALL",
                   "label": "#all-toup"}, "mode": "auto"}
    spec = _spec(steps=[
        _read_step("mail", "gmail", "gmail__list_messages",
                   {"query": "in:inbox", "max_results": 10}),
        {"id": "post", "connector_id": "slack", "tool": "slack__send_message",
         "params": {"channel": "C_ALL", "text": "{{steps.mail.text}}"},
         "grant_id": "g_slack"},
    ])
    a = await _mk_automation(uid, spec)
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        rows = _by_id((await wf.workflow_payload(
            db, automation=row, user_id=uid))["delivery"]
            ["channels_available"])
    assert rows["slack_dm"]["available"] is False
    assert rows["slack_dm"]["reason"] == ("Slack can write to #all-toup, "
                                          "not to you.")


@pytest.mark.asyncio
async def test_a_grant_pinned_at_the_user_is_offered_and_writable(platform):
    uid = await _mk_user()
    platform["grants"]["g_dm"] = {
        "id": "g_dm", "status": "approved", "connector_id": "slack",
        "tool_name": "slack__send_message",
        "target": {"kind": "channel", "id": "D_ME", "label": "you"},
        "mode": "auto"}
    spec = _spec(steps=[
        _read_step("mail", "gmail", "gmail__list_messages",
                   {"query": "in:inbox", "max_results": 10}),
        {"id": "post", "connector_id": "slack", "tool": "slack__send_message",
         "params": {"channel": "D_ME", "text": "{{steps.mail.text}}"},
         "grant_id": "g_dm"},
    ])
    a = await _mk_automation(uid, spec)
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        rows = _by_id((await wf.workflow_payload(
            db, automation=row, user_id=uid))["delivery"]
            ["channels_available"])
    assert rows["slack_dm"]["available"] is True
    assert rows["slack_dm"]["reason"] is None

    # And the writer accepts what the picker offered — the two ask the
    # same question of the same table.
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        out = await wf.set_delivery(db, automation=row, user_id=uid,
                                    channels=["app", "slack_dm"])
    assert out["delivery"]["channels"] == ["app", "slack_dm"]


@pytest.mark.asyncio
async def test_the_writer_refuses_a_channel_the_picker_would_not_offer(
        platform):
    uid = await _mk_user()
    a = await _mk_automation(uid, _spec())
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        with pytest.raises(wf.WorkflowError) as e:
            await wf.set_delivery(db, automation=row, user_id=uid,
                                  channels=["gmail_draft"])
    assert e.value.code == "channel_unavailable"
    # The refusal carries the same sentence the row does — the sheet
    # prints it, so it may never be a code.
    assert "allowed to write" in str(e.value)


# ── §2.2 the writer, and the cap it now shares with the run ─────────

@pytest.mark.asyncio
async def test_the_writer_refuses_a_pick_no_run_can_honour(platform,
                                                           monkeypatch):
    from app.agent.automations import contents

    async def _sources(user_id, connector_id, focus=None):
        return []

    monkeypatch.setattr(contents, "account_sources", _sources, raising=False)
    uid = await _mk_user()
    spec = _spec(steps=[
        _read_step("mail", "gmail", "gmail__list_messages",
                   {"query": "in:inbox", "max_results": 10}),
        _read_step("chat", "slack", "slack__read_messages", {"limit": 20}),
        _read_step("pages", "notion", "notion__search", {"page_size": 10}),
    ])
    a = await _mk_automation(uid, spec)

    # Notion's read is a free-text search that cannot be aimed at a
    # page id, so a pick would be stored and read by nothing.
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        with pytest.raises(wf.WorkflowError) as e:
            await wf.set_sources(db, automation=row, user_id=uid,
                                 connector_id="notion", sources=["page_1"])
    assert e.value.code == "not_scopable"

    # Slack reads ONE channel per call, and the fan-out bound is the
    # writer's cap: the picker may not promise a set the run drops.
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        with pytest.raises(wf.WorkflowError) as e:
            await wf.set_sources(db, automation=row, user_id=uid,
                                 connector_id="slack",
                                 sources=["a", "b", "c", "d", "e"])
    assert e.value.code == "too_many_sources_picked"

    # A Gmail id no query can select is refused rather than stored.
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        with pytest.raises(wf.WorkflowError) as e:
            await wf.set_sources(db, automation=row, user_id=uid,
                                 connector_id="gmail", sources=["Label_12"])
    assert e.value.code == "unknown_source"


@pytest.mark.asyncio
async def test_the_payload_says_how_many_places_each_account_can_open(
        platform, monkeypatch):
    from app.agent.automations import contents

    async def _sources(user_id, connector_id, focus=None):
        return []

    monkeypatch.setattr(contents, "account_sources", _sources, raising=False)
    uid = await _mk_user()
    a = await _mk_automation(uid, _spec(steps=[
        _read_step("mail", "gmail", "gmail__list_messages",
                   {"query": "in:inbox", "max_results": 10}),
        _read_step("chat", "slack", "slack__read_messages", {"limit": 20}),
        _read_step("pages", "notion", "notion__search", {"page_size": 10}),
    ]))
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        accounts = {e["account_id"]: e for e in (await wf.workflow_payload(
            db, automation=row, user_id=uid))["accounts"]}
    # Always served, never omitted — the app has to tell "one place at a
    # time" from "this read cannot be aimed at all".
    assert accounts["gmail"]["sources_max"] == 10
    assert accounts["slack"]["sources_max"] == 4
    assert accounts["notion"]["sources_max"] == 0


# ── integrator: the keys the app gates on, and the narrator ROOT ─────
# The three fixes above each left a claim the app had no way to check.
# These are the wire keys that close them, plus the one-line narrator
# fix that is the ROOT of finding 15 (`_recheck_unservable` above is
# the consumer-side belt; this is the braces).

def test_the_narrator_itself_carries_the_verdict_through_a_dead_retry():
    """narrator.py's own `except` — the ROOT of finding 15.

    Attempt 1 comes back with a complete five-tier ranking and one nit
    (an empty `tag` on one of two rows), which `unservable_results`
    deliberately KEEPS — its whole rule is that a bad tag on one line is
    not a reason to replace a real ranking with "I could not rank them".
    The retry's LLM call then raises, which a ReadTimeout against the
    model makes the ordinary case. The verdict that travels out must be
    the one already computed for the drafts in hand, not "every result".
    """
    import asyncio
    from app.agent.automations import narrator

    record = {"vocabulary": "brief", "status": "completed",
              "automation": {"title": "Morning brief"},
              "steps": [{"step_ref": "mail",
                         "items": [{"id": "it_1", "msgs": []},
                                   {"id": "it_2", "msgs": []}]}]}
    groups = [{"rank": r, "label": lb, "tone": t, "rows": []}
              for r, lb, t in narrator.BRIEF_GROUPS]
    groups[0]["rows"] = [
        {"text": "Dana is waiting", "sub": "on the retry flag",
         "item_refs": ["it_1"]},
        # The nit, and only the nit: one row of two.
        {"text": "Sam needs the invoice", "sub": "before payroll closes",
         "tag": "", "item_refs": ["it_2"]},
    ]
    good = [
        {"kind": "agent", "text": "Here is your morning."},
        {"kind": "annotate", "step_ref": "mail", "items": [
            {"id": "it_1", "why": "it blocks the client fix", "msgs": []},
            {"id": "it_2", "why": "payroll closes at noon", "msgs": []}]},
        {"kind": "result",
         "title": narrator.expected_result_title("brief", record),
         "vocabulary": "brief", "groups": groups},
    ]
    problems = narrator.validate_drafts(good, record)
    assert problems, "the fixture must be rejected, or there is no retry"
    assert narrator.unservable_results(good, problems) == set(), \
        "the fixture must be a draft the rule KEEPS"

    calls = {"n": 0}

    async def _complete(prompt, tool):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"turns": good}
        raise TimeoutError("the model took too long")

    out = asyncio.run(narrator.narrate_run(record, complete=_complete))
    assert calls["n"] == 2, "the fixture must reach the retry"
    assert out["problems"] and out["problems"][0].startswith("llm:")
    assert out["turns"] == good
    # Before the fix this was [2] — every result index, unconditionally
    # — and `_narrate_phase1` served the mechanical fallback over a
    # complete brief.
    assert out["unservable"] == [], out["unservable"]

    # A result the validator really cannot serve still travels as
    # unservable through the same path: the fix carries the verdict, it
    # does not suppress it.
    bad = [{"kind": "agent", "text": "Here is your morning."},
           {"kind": "result", "title": "wrong", "vocabulary": "brief",
            "groups": []}]

    async def _complete_bad(prompt, tool):
        calls["n"] += 1
        if calls["n"] == 3:
            return {"turns": bad}
        raise TimeoutError("the model took too long")

    out2 = asyncio.run(narrator.narrate_run(record, complete=_complete_bad))
    assert out2["unservable"] == [1], out2["unservable"]


@pytest.mark.asyncio
async def test_an_empty_source_list_says_whether_it_is_a_fact_or_a_failure(
        platform, monkeypatch):
    """§0.1's absent-≠-empty rule has a third state, and the app cannot
    see it: `sources_available: []` is served for a connector that holds
    nothing separate AND for every failure to enumerate one. Rendering
    the second as the first put "Gmail has nothing separate inside it"
    under a Reconnect button (finding 1)."""
    from app.agent.automations import contents

    uid = await _mk_user()
    a = await _mk_automation(uid, _spec())

    async def _payload():
        wf.invalidate_sources_cache()
        async with async_session_maker() as db:
            row = await db.get(Automation, a.id)
            return {e["account_id"]: e for e in (await wf.workflow_payload(
                db, automation=row, user_id=uid))["accounts"]}

    # A genuine empty is a FACT and says nothing.
    async def _none(user_id, connector_id, focus=None):
        return []
    monkeypatch.setattr(contents, "account_sources", _none, raising=False)
    accounts = await _payload()
    assert accounts["gmail"]["sources_available"] == []
    assert accounts["gmail"]["sources_reason"] is None

    # A provider that raised is a FAILURE and says so, in words.
    async def _boom(user_id, connector_id, focus=None):
        raise RuntimeError("provider is having a minute")
    monkeypatch.setattr(contents, "account_sources", _boom, raising=False)
    accounts = await _payload()
    assert accounts["gmail"]["sources_available"] == []
    reason = accounts["gmail"]["sources_reason"]
    assert reason and "Gmail" in reason and "gmail" not in reason.lower()[:1]
    # A sentence, never a code — the app prints it verbatim.
    assert reason[0].isupper() and reason.endswith(".")


@pytest.mark.asyncio
async def test_the_deadline_names_itself_on_every_account_it_emptied(
        platform, monkeypatch):
    """The failure the app is least able to see: the ONE budget over all
    the accounts empties healthy connectors too."""
    from app.agent.automations import contents

    async def _slow(user_id, connector_id, focus=None):
        import asyncio as _a
        await _a.sleep(5)
        return []

    monkeypatch.setattr(contents, "account_sources", _slow, raising=False)
    monkeypatch.setattr(wf, "_SOURCES_BUDGET_S", 0.05)
    uid = await _mk_user()
    a = await _mk_automation(uid, _spec(steps=[
        _read_step("mail", "gmail", "gmail__list_messages",
                   {"query": "in:inbox", "max_results": 10}),
        _read_step("chat", "slack", "slack__read_messages", {"limit": 20}),
    ]))
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        accounts = {e["account_id"]: e for e in (await wf.workflow_payload(
            db, automation=row, user_id=uid))["accounts"]}
    for cid in ("gmail", "slack"):
        assert accounts[cid]["sources_available"] == []
        assert accounts[cid]["sources_reason"], cid


@pytest.mark.asyncio
async def test_a_v1_automation_says_its_editors_are_shut(platform):
    """Finding 22. `set_sources` / `set_delivery` / `set_ping` are all
    v2-only; every read beside them is served unconditionally, so
    without this key the app gates on PRESENCE and lights three controls
    the platform answers 409 `not_supported` to."""
    uid = await _mk_user()
    a = await _mk_automation(uid, _spec())

    async def _payload():
        async with async_session_maker() as db:
            row = await db.get(Automation, a.id)
            return await wf.workflow_payload(
                db, automation=row, user_id=uid)

    assert (await _payload())["edits_locked"] is None

    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        raw = json.loads(row.spec_json)
        raw["version"] = 1
        row.spec_json = json.dumps(raw, sort_keys=True)
        await db.commit()

    locked = (await _payload())["edits_locked"]
    # A sentence the app prints, not a version number it would have to
    # find words for (§0.2).
    assert isinstance(locked, str) and locked[0].isupper()
    assert locked.endswith(".") and "v1" not in locked and "2" not in locked

    # And it is the truth: the writer really does refuse.
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        with pytest.raises(wf.WorkflowError) as e:
            await wf.set_sources(db, automation=row, user_id=uid,
                                 connector_id="gmail", sources=["inbox"])
    assert e.value.code == "not_supported"
