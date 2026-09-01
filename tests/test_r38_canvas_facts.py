# agent-mode: automations/threads/permissions are AGENT_ONLY tables.
"""R38 — the canvas's four facts, server side.

  1. BUILD LEDGER      — how the automation was built, recorded once,
                         survives a re-read days later.
  2. CONNECTOR CONTENTS— what is inside THAT account, in one envelope,
                         where absent is never presented as empty.
  3. SUB-NODE PINNING  — `focus` in the spec, on the payload, in the run
                         context and in the thread grounding.
  4. PER-AUTOMATION    — every fact `GET /workflow` serves belongs to
                         the automation in the path and to no other.
"""

import json
import uuid

import pytest

from app.db.database import async_session_maker
from app.db.models import Automation, User

from tests.test_workflow_api import _offline_platform  # noqa: F401
from tests.test_run_ledger_v3 import (  # noqa: F401 — shared fixtures
    REGISTRY_V2, _mk_user, _mk_automation_v2,
)


def _raw():
    """The same v2 spec `_v2_spec` builds, as the RAW dict — these tests
    edit the spec before it is validated, and the shared helper hands
    back a `ValidatedSpecV2`."""
    return {
        "version": 2,
        "name": "Ledger brief",
        "mode": "auto",
        "trigger": {"sources": [
            {"id": "sched", "mode": "schedule",
             "schedule": {"cron_local": "0 8 * * 1-5"}},
        ]},
        "steps": [
            {"id": "issues", "connector_id": "jira",
             "tool": "jira__search_issues",
             "params": {"jql": "x"},
             "collect": {"items_path": "issues",
                         "fields": {"key": "key", "summary": "summary"},
                         "format": "{{item.key}} {{item.summary}}",
                         "empty_text": "none"},
             "on_error": "skip"},
            {"id": "post", "connector_id": "slack",
             "tool": "slack__send_message",
             "params": {"channel": "{{grant.target.id}}",
                        "text": "{{steps.issues.text}}"},
             "grant_id": "g-1",
             "grant_target": {"kind": "channel", "id": "C-PIN",
                              "label": "#platform"}},
        ],
    }


async def _mk(uid: str, raw: dict = None):
    from app.agent.automations.spec import validate_spec
    return await _mk_automation_v2(
        uid, validate_spec(raw or _raw(), REGISTRY_V2, template_mode=True),
    )


# ─────────────────────────────────────────────── 1. the build ledger

@pytest.mark.asyncio
async def test_the_build_ledger_records_measured_phases_and_survives():
    from app.agent.automations import build_ledger
    from app.agent.automations.workflow import workflow_payload

    uid = await _mk_user()
    a = await _mk(uid)

    rec = build_ledger.BuildRecorder("template")
    with rec.phase("trigger"):
        pass
    with rec.phase("agent"):
        pass
    with rec.phase("output"):
        pass
    with rec.phase("account:jira"):
        pass
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        payload = await build_ledger.record(db, automation=row, recorder=rec)

    assert payload is not None
    assert payload["source"] == "template"
    ids = [s["id"] for s in payload["steps"]]
    # R43: the account phases are TIMED per account and TOLD per BAND
    # (design §8), so one Jira account becomes the WORK band's one step.
    assert ids == ["trigger", "agent", "output", "band:work"], ids
    for step in payload["steps"]:
        assert step["title"] and isinstance(step["title"], str)
        assert isinstance(step["sub"], str)
        assert isinstance(step["ms"], int) and step["ms"] >= 0
        assert step["did"], f"a phase with no lines says nothing: {step}"

    # It is per-automation, and it survives a fresh read.
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        wf = await workflow_payload(db, automation=row, user_id=uid)
    assert wf["build_history"]["steps"] == payload["steps"]
    assert wf["automation_id"] == a.id


@pytest.mark.asyncio
async def test_the_phases_are_reordered_into_reading_order():
    """The template path resolves the destination before it arms and the
    described path arms first. A history that served measurement order
    would tell the same story in two sequences."""
    from app.agent.automations import build_ledger

    rec = build_ledger.BuildRecorder("described")
    for pid in ("output", "account:slack", "trigger", "agent"):
        with rec.phase(pid):
            pass
    assert [p for p, _ in rec.timings()] == [
        "trigger", "agent", "output", "account:slack",
    ]


@pytest.mark.asyncio
async def test_a_re_entered_phase_adds_its_time_rather_than_replacing_it():
    from app.agent.automations import build_ledger
    import time

    rec = build_ledger.BuildRecorder("chat")
    with rec.phase("trigger"):
        time.sleep(0.02)
    with rec.phase("trigger"):
        time.sleep(0.02)
    (_pid, ms), = rec.timings()
    assert ms >= 35, ms


@pytest.mark.asyncio
async def test_no_history_is_null_never_an_empty_list():
    """`[]` would claim the automation was built in no steps."""
    from app.agent.automations import build_ledger
    from app.agent.automations.workflow import workflow_payload

    uid = await _mk_user()
    a = await _mk(uid)
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        assert build_ledger.read(row) is None
        wf = await workflow_payload(db, automation=row, user_id=uid)
    assert wf["build_history"] is None


@pytest.mark.asyncio
async def test_every_authored_build_line_passes_the_copy_guard():
    """The design sketch said "Set the trigger"; `trigger` is a banned
    word. Every string THIS module authors is scanned — the output
    phase quotes `workflow.output_block` verbatim and is that surface's
    copy, not ours to launder."""
    from app.agent.automations import build_ledger, copy_guard

    uid = await _mk_user()
    a = await _mk(uid)
    rec = build_ledger.BuildRecorder("template")
    for pid in ("trigger", "agent", "output", "account:jira",
                "account:slack"):
        with rec.phase(pid):
            pass
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        payload = await build_ledger.record(db, automation=row, recorder=rec)

    for step in payload["steps"]:
        assert copy_guard.clean(step["title"]), copy_guard.scan(step["title"])
        assert copy_guard.clean(step["sub"]), copy_guard.scan(step["sub"])
        if step["id"] == "output":
            continue
        for line in step["did"]:
            assert copy_guard.clean(line), (line, copy_guard.scan(line))


@pytest.mark.asyncio
async def test_from_template_writes_the_history_it_returns(monkeypatch):
    from app.api.automations import FromTemplateBody, from_template
    from app.agent.automations import build_ledger
    from app.config import settings

    uid = await _mk_user()
    monkeypatch.setattr(settings, "automations_enabled", True)
    monkeypatch.setattr(settings, "user_id", uid)

    async def _templates(user_id):
        return [{"id": "t-brief", "slug": "t-brief", "name": "Brief",
                 "category": "work", "variables": [],
                 "spec": _raw()}]
    monkeypatch.setattr(
        "app.agent.automations.registry.fetch_templates", _templates,
    )
    out = await from_template(FromTemplateBody(template_id="t-brief"))
    history = out["build_history"]
    assert history and history["source"] == "template"
    ids = [s["id"] for s in history["steps"]]
    assert ids[:3] == ["trigger", "agent", "output"], ids
    assert "band:work" in ids and "band:chat" in ids, ids

    async with async_session_maker() as db:
        row = await db.get(Automation, out["automation"]["id"])
        assert build_ledger.read(row)["steps"] == history["steps"]


# ────────────────────────────────────────────── 2. account contents

def _ok(payload: dict) -> dict:
    return {"kind": "ok", "content": json.dumps(payload)}


@pytest.mark.asyncio
async def test_gmail_contents_carry_sender_subject_and_snippet(monkeypatch):
    from app.agent.automations import contents

    async def _dispatch(user_id, *, connector_id, tool_name, tool_input,
                        **kw):
        assert tool_name == "gmail__list_messages"
        return _ok({"messages": [{
            "id": "m1",
            "headers": {"From": "Sara Chen <sara@x.com>",
                        "Subject": "Re: launch",
                        "Date": "Fri, 29 Aug 2026 09:14:00 +0000"},
            "snippet": "can we move it to Monday",
        }]})
    monkeypatch.setattr(
        "app.agent.automations.registry.dispatch_via_platform", _dispatch,
    )
    env = await contents.account_contents(
        "u", connector_id="gmail",
        connection={"connected": True, "status": "active"},
    )
    assert env["ok"] is True and env["reason"] is None
    assert env["connector_id"] == "gmail" and env["count"] == 1
    item = env["groups"][0]["items"][0]
    assert item["title"] == "Re: launch"
    assert "Sara Chen" in item["sub"] and "Monday" in item["sub"]
    assert item["at"] == "2026-08-29T09:14:00Z"


@pytest.mark.asyncio
async def test_slack_contents_read_only_joined_channels(monkeypatch):
    from app.agent.automations import contents
    seen = []

    async def _dispatch(user_id, *, connector_id, tool_name, tool_input,
                        **kw):
        seen.append((tool_name, tool_input))
        if tool_name == "slack__list_channels":
            return _ok({"channels": [
                {"id": "C1", "name": "eng", "is_member": True},
                {"id": "C2", "name": "general", "is_member": False},
            ]})
        return _ok({"messages": [
            {"ts": "1787000000.001200", "from": "Sara",
             "text": "shipping today"},
        ]})
    monkeypatch.setattr(
        "app.agent.automations.registry.dispatch_via_platform", _dispatch,
    )
    env = await contents.account_contents(
        "u", connector_id="slack",
        connection={"connected": True, "status": "active"},
    )
    read = [t for t, _ in seen if t == "slack__read_messages"]
    assert len(read) == 1, "a channel the workspace never joined is not read"
    assert [g["key"] for g in env["groups"]] == ["C1"]
    assert env["groups"][0]["label"] == "#eng"
    assert env["groups"][0]["items"][0]["at"].endswith("Z")
    assert env["groups"][0]["items"][0]["at"].startswith("2026-")


@pytest.mark.asyncio
async def test_jira_contents_name_the_due_date(monkeypatch):
    from app.agent.automations import contents

    async def _dispatch(user_id, *, connector_id, tool_name, tool_input,
                        **kw):
        assert "duedate" in tool_input["fields"]
        assert "currentUser()" in tool_input["jql"]
        return _ok({"issues": [
            {"key": "ENG-1", "summary": "Ship it", "status": "To Do",
             "duedate": "2026-09-01"},
            {"key": "ENG-2", "summary": "Later", "status": "To Do"},
        ]})
    monkeypatch.setattr(
        "app.agent.automations.registry.dispatch_via_platform", _dispatch,
    )
    env = await contents.account_contents(
        "u", connector_id="jira",
        connection={"connected": True, "status": "active"},
    )
    # R42: the groups are the PROJECTS now — the due date is named on
    # the row that has one, which is where it was always read from.
    items = {g["key"]: g for g in env["groups"]}["ENG"]["items"]
    assert "due 2026-09-01" in items[0]["sub"]
    assert items[1]["id"] == "ENG-2" and "due" not in items[1]["sub"]


@pytest.mark.asyncio
async def test_github_contents_are_pull_requests_not_issues(monkeypatch):
    from app.agent.automations import contents

    async def _dispatch(user_id, *, connector_id, tool_name, tool_input,
                        **kw):
        if tool_name == "github__list_repos":
            return _ok({"repos": [{"full_name": "toup/platform"}]})
        return _ok({"issues": [
            {"number": 7, "title": "A PR", "user": "sara",
             "is_pull_request": True},
            {"number": 8, "title": "An issue", "user": "sara",
             "is_pull_request": False},
        ]})
    monkeypatch.setattr(
        "app.agent.automations.registry.dispatch_via_platform", _dispatch,
    )
    env = await contents.account_contents(
        "u", connector_id="github",
        connection={"connected": True, "status": "active"},
    )
    items = env["groups"][0]["items"]
    assert [i["id"] for i in items] == ["7"]
    assert items[0]["kind"] == "pull_request"


@pytest.mark.asyncio
async def test_an_unreachable_account_is_a_reason_not_an_empty_list(
    monkeypatch,
):
    """The repo's own scar: an unreachable source served as `[]` reads
    as data loss. `ok` is the branch, and the sentence names the
    account."""
    from app.agent.automations import contents

    async def _dispatch(user_id, *, connector_id, tool_name, tool_input,
                        **kw):
        return {"kind": "tool_error", "retryable": True,
                "message": "dispatch unreachable: connect timeout"}
    monkeypatch.setattr(
        "app.agent.automations.registry.dispatch_via_platform", _dispatch,
    )
    env = await contents.account_contents(
        "u", connector_id="gmail",
        connection={"connected": True, "status": "active"},
    )
    assert env["ok"] is False
    assert env["groups"] == [] and env["count"] == 0
    assert env["reason"]["code"] == "unreachable"
    assert "Gmail" in env["reason"]["sentence"]
    assert env["reason"]["retryable"] is True


@pytest.mark.asyncio
async def test_a_dead_credential_and_an_empty_account_are_different(
    monkeypatch,
):
    from app.agent.automations import contents

    async def _reauth(user_id, *, connector_id, tool_name, tool_input, **kw):
        return {"kind": "reauth_required", "retryable": False,
                "reauth_url": "https://toup.ai/reconnect"}
    monkeypatch.setattr(
        "app.agent.automations.registry.dispatch_via_platform", _reauth,
    )
    dead = await contents.account_contents(
        "u", connector_id="gmail",
        connection={"connected": True, "status": "active"},
    )
    assert dead["ok"] is False and dead["reason"]["code"] == "reconnect"
    assert dead["reason"]["reauth_url"] == "https://toup.ai/reconnect"

    async def _empty(user_id, *, connector_id, tool_name, tool_input, **kw):
        return _ok({"messages": []})
    monkeypatch.setattr(
        "app.agent.automations.registry.dispatch_via_platform", _empty,
    )
    empty = await contents.account_contents(
        "u", connector_id="gmail",
        connection={"connected": True, "status": "active"},
    )
    assert empty["ok"] is True and empty["reason"] is None
    assert empty["count"] == 0 and empty["groups"][0]["reason"] is None


@pytest.mark.asyncio
async def test_an_expired_connection_never_makes_the_call(monkeypatch):
    from app.agent.automations import contents
    called = []

    async def _dispatch(user_id, **kw):
        called.append(kw)
        return _ok({})
    monkeypatch.setattr(
        "app.agent.automations.registry.dispatch_via_platform", _dispatch,
    )
    env = await contents.account_contents(
        "u", connector_id="gmail",
        connection={"connected": True, "status": "reauth_required"},
    )
    assert not called
    assert env["ok"] is False and env["reason"]["code"] == "reconnect"


@pytest.mark.asyncio
async def test_a_connector_with_no_reader_says_so_by_name():
    from app.agent.automations import contents

    # R43 gave Notion a reader (it is in the design's PLANS band), so the
    # example moved to one that still has none. The behaviour under test is
    # the SENTENCE naming the connector — an unnamed "not supported" reads
    # as the whole popup being broken.
    env = await contents.account_contents("u", connector_id="drive")
    assert env["ok"] is False
    assert env["reason"]["code"] == "not_supported"
    assert "Drive" in env["reason"]["sentence"]


@pytest.mark.asyncio
async def test_one_failing_channel_does_not_erase_the_others(monkeypatch):
    """A group that failed says so IN the group. Dropping it would make
    a broken channel indistinguishable from a quiet one."""
    from app.agent.automations import contents

    async def _dispatch(user_id, *, connector_id, tool_name, tool_input,
                        **kw):
        if tool_name == "slack__list_channels":
            return _ok({"channels": [
                {"id": "C1", "name": "eng", "is_member": True},
                {"id": "C2", "name": "design", "is_member": True},
            ]})
        if tool_input.get("channel") == "C2":
            return {"kind": "tool_error", "retryable": False,
                    "message": "channel_not_found"}
        return _ok({"messages": [{"ts": "1756400000.0", "from": "S",
                                  "text": "hi"}]})
    monkeypatch.setattr(
        "app.agent.automations.registry.dispatch_via_platform", _dispatch,
    )
    env = await contents.account_contents(
        "u", connector_id="slack",
        connection={"connected": True, "status": "active"},
    )
    assert env["ok"] is True
    by_key = {g["key"]: g for g in env["groups"]}
    assert by_key["C1"]["reason"] is None and by_key["C1"]["items"]
    assert by_key["C2"]["reason"]["code"] == "refused"
    assert by_key["C2"]["items"] == []


@pytest.mark.asyncio
async def test_the_contents_route_uses_this_automations_pins(monkeypatch):
    from app.api.automations import get_account_contents
    from app.config import settings

    uid = await _mk_user()
    monkeypatch.setattr(settings, "automations_enabled", True)
    monkeypatch.setattr(settings, "user_id", uid)
    spec = _raw()
    spec["focus"] = {"slack": [{"kind": "channel", "id": "C-PIN",
                                "label": "#platform"}]}
    a = await _mk(uid, spec)

    seen = []

    async def _dispatch(user_id, *, connector_id, tool_name, tool_input,
                        **kw):
        seen.append(tool_name)
        return _ok({"messages": [{"ts": "1756400000.0", "from": "S",
                                  "text": "hi"}]})
    monkeypatch.setattr(
        "app.agent.automations.registry.dispatch_via_platform", _dispatch,
    )
    env = await get_account_contents(a.id, "slack")
    # R42: the workspace is listed even so — a pin LEADS this list, it
    # never shortens it, or the first pin is the last channel the user
    # can ever pick in here (founder P6).
    assert "slack__list_channels" in seen
    assert env["groups"][0]["key"] == "C-PIN"
    assert env["groups"][0]["pinned"] is True
    assert env["focus"][0]["label"] == "#platform"


# ────────────────────────────────────────────────── 3. focus pinning

@pytest.mark.asyncio
async def test_a_pin_round_trips_through_the_spec_and_the_payload():
    from app.agent.automations.workflow import (
        add_focus, remove_focus, workflow_payload,
    )

    uid = await _mk_user()
    a = await _mk(uid)
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        out = await add_focus(
            db, automation=row, user_id=uid, account_id="slack",
            kind="channel", target_id="C-PIN", label="#platform",
        )
    assert out["focus"] == [{"kind": "channel", "id": "C-PIN",
                             "label": "#platform"}]
    assert "#platform" in out["sentence"]

    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        raw = json.loads(row.spec_json)
        assert raw["focus"]["slack"][0]["id"] == "C-PIN"
        wf = await workflow_payload(db, automation=row, user_id=uid)
    slack = next(x for x in wf["accounts"] if x["account_id"] == "slack")
    assert slack["focus"][0]["label"] == "#platform"
    jira = next(x for x in wf["accounts"] if x["account_id"] == "jira")
    assert jira["focus"] == []

    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        gone = await remove_focus(
            db, automation=row, user_id=uid, account_id="slack",
            kind="channel", target_id="C-PIN",
        )
    assert gone["focus"] == []
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        assert "focus" not in json.loads(row.spec_json)


@pytest.mark.asyncio
async def test_pinning_writes_the_edited_note():
    from app.agent.automations.workflow import add_focus
    from tests.test_workflow_api import _edited_notes

    uid = await _mk_user()
    a = await _mk(uid)
    before = await _edited_notes(a.id)
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        await add_focus(db, automation=row, user_id=uid, account_id="slack",
                        kind="channel", target_id="C-PIN", label="#platform")
    assert await _edited_notes(a.id) == before + 1


@pytest.mark.asyncio
async def test_a_pin_under_an_account_the_automation_does_not_use_is_refused():
    from app.agent.automations.workflow import WorkflowError, add_focus

    uid = await _mk_user()
    a = await _mk(uid)
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        with pytest.raises(WorkflowError) as e:
            await add_focus(db, automation=row, user_id=uid,
                            account_id="github", kind="repo",
                            target_id="toup/platform")
    assert e.value.code == "not_member"
    assert e.value.sentence


@pytest.mark.asyncio
async def test_unpinning_something_that_is_not_pinned_is_a_refusal():
    """Answering 200 would let the app redraw a row it never removed."""
    from app.agent.automations.workflow import WorkflowError, remove_focus

    uid = await _mk_user()
    a = await _mk(uid)
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        with pytest.raises(WorkflowError) as e:
            await remove_focus(db, automation=row, user_id=uid,
                               account_id="slack", kind="channel",
                               target_id="C-NOPE")
    assert e.value.code == "not_found"


@pytest.mark.asyncio
async def test_the_same_place_pinned_twice_is_one_pin():
    from app.agent.automations.workflow import add_focus

    uid = await _mk_user()
    a = await _mk(uid)
    for _ in range(2):
        async with async_session_maker() as db:
            row = await db.get(Automation, a.id)
            out = await add_focus(
                db, automation=row, user_id=uid, account_id="slack",
                kind="channel", target_id="C-PIN", label="#platform",
            )
    assert len(out["focus"]) == 1


@pytest.mark.asyncio
async def test_removing_the_account_removes_its_pins():
    from app.agent.automations import service
    from app.agent.automations.workflow import add_focus, focus_of

    uid = await _mk_user()
    spec = _raw()
    # A second read source so removing jira leaves a live spec.
    spec["trigger"]["sources"].append(
        {"id": "jissue", "mode": "poll", "connector_id": "jira",
         "event": "issue_created", "poll_interval_s": 300,
         "dedupe_key": "event.key"},
    )
    a = await _mk(uid, spec)
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        await add_focus(db, automation=row, user_id=uid, account_id="jira",
                        kind="project", target_id="TP", label="TP")
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        assert focus_of(json.loads(row.spec_json))["jira"]

    async with async_session_maker() as db:
        await service.remove_connector(
            db, automation_id=a.id, user_id=uid, connector_id="jira",
        )
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        raw = json.loads(row.spec_json)
    assert "jira" not in focus_of(raw), (
        "a pin under a removed account is a place nothing can start from")
    assert "focus" not in raw, "the last pin gone takes the key with it"


@pytest.mark.asyncio
async def test_the_spec_validator_refuses_a_malformed_pin():
    from app.agent.automations.spec import SpecError, validate_spec

    spec = _raw()
    spec["focus"] = {"slack": [{"kind": "wormhole", "id": "C1"}]}
    with pytest.raises(SpecError) as e:
        validate_spec(spec, REGISTRY_V2, template_mode=True)
    assert any(x["code"] == "bad_focus_kind" for x in e.value.errors)

    spec["focus"] = {"slack": [{"kind": "channel"}]}
    with pytest.raises(SpecError) as e:
        validate_spec(spec, REGISTRY_V2, template_mode=True)
    assert any(x["code"] == "bad_focus_id" for x in e.value.errors)

    spec["focus"] = {"slack": [{"kind": "channel", "id": "C1",
                                "sneaky": "x"}]}
    with pytest.raises(SpecError) as e:
        validate_spec(spec, REGISTRY_V2, template_mode=True)
    assert any(x["code"] == "unknown_field" for x in e.value.errors)


@pytest.mark.asyncio
async def test_a_step_may_not_be_called_focus():
    """`focus` is a render root now; a step with that id would shadow
    the namespace every later step reads from."""
    from app.agent.automations.spec import SpecError, validate_spec

    spec = _raw()
    spec["steps"][0]["id"] = "focus"
    with pytest.raises(SpecError) as e:
        validate_spec(spec, REGISTRY_V2, template_mode=True)
    assert any("focus" in json.dumps(x) for x in e.value.errors)


@pytest.mark.asyncio
async def test_the_pin_reaches_the_run_context_as_flat_leaves():
    from app.agent.automations.spec import focus_render_ctx, render_with_ctx

    ctx = {"focus": focus_render_ctx({
        "slack": [{"kind": "channel", "id": "C1", "label": "#eng"},
                  {"kind": "channel", "id": "C2", "label": "#design"}],
    })}
    assert ctx["focus"]["slack"]["first"]["id"] == "C1"
    assert ctx["focus"]["slack"]["ids"] == "C1,C2"
    assert ctx["focus"]["slack"]["count"] == 2
    out = render_with_ctx(
        {"channel": "{{focus.slack.first.id}}",
         "note": "reading {{focus.slack.labels}}"}, ctx,
    )
    assert out == {"channel": "C1", "note": "reading #eng, #design"}


@pytest.mark.asyncio
async def test_the_executor_builds_the_focus_root_from_the_spec():
    from app.agent.automations.spec import validate_spec

    spec = _raw()
    spec["focus"] = {"slack": [{"kind": "channel", "id": "C-PIN",
                                "label": "#platform"}]}
    vspec = validate_spec(spec, REGISTRY_V2, template_mode=True)
    assert vspec.focus == {"slack": [{"kind": "channel", "id": "C-PIN",
                                      "label": "#platform"}]}


@pytest.mark.asyncio
async def test_the_thread_grounding_names_where_it_starts():
    from app.agent.automations.thread_agent import _setup_lines

    uid = await _mk_user()
    spec = _raw()
    spec["focus"] = {"slack": [{"kind": "channel", "id": "C-PIN",
                                "label": "#platform"}]}
    a = await _mk(uid, spec)
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        lines = _setup_lines(row)
    assert any("#platform" in line and "Starts at" in line
               for line in lines), lines


@pytest.mark.asyncio
async def test_an_agent_step_is_told_where_the_run_starts():
    from app.agent.automations.agent_step import build_prompt
    from app.agent.automations.spec import focus_render_ctx

    class _Step:
        prompt = "Summarise it."
        output_var = "x"

    ctx = {"event": {}, "var": {}, "steps": {},
           "focus": focus_render_ctx(
               {"slack": [{"kind": "channel", "id": "C1", "label": "#eng"}]})}
    prompt = build_prompt("Brief", _Step(), ctx)
    assert "starts_at" in prompt and "#eng" in prompt

    bare = build_prompt("Brief", _Step(), {"event": {}, "var": {},
                                           "steps": {}, "focus": {}})
    assert "starts_at" not in bare


# ─────────────────────────────────────────── 4. per-automation facts

@pytest.mark.asyncio
async def test_two_automations_serve_two_different_canvases():
    """The founder's complaint: every canvas read the same. Every fact
    the payload carries is derived from the automation in the path."""
    from app.agent.automations.workflow import add_focus, workflow_payload

    uid = await _mk_user()
    spec_a = _raw()
    spec_a["name"] = "Morning brief"
    a = await _mk(uid, spec_a)

    spec_b = _raw()
    spec_b["name"] = "Evening wrap"
    spec_b["trigger"]["sources"][0]["schedule"] = {"cron_local": "0 9 * * 1-5"}
    spec_b["steps"] = [spec_b["steps"][0]]        # reads only, no write
    b = await _mk(uid, spec_b)

    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        await add_focus(db, automation=row, user_id=uid, account_id="slack",
                        kind="channel", target_id="C-PIN", label="#platform")

    async with async_session_maker() as db:
        wf_a = await workflow_payload(
            db, automation=await db.get(Automation, a.id), user_id=uid)
        wf_b = await workflow_payload(
            db, automation=await db.get(Automation, b.id), user_id=uid)

    assert wf_a["automation_id"] == a.id
    assert wf_b["automation_id"] == b.id
    assert wf_a["name"] == "Morning brief"
    assert wf_b["name"] == "Evening wrap"
    # schedule
    assert wf_a["schedule"]["preset_id"] == "weekdays-8"
    assert wf_b["schedule"]["preset_id"] == "weekdays-9"
    # accounts + pins
    assert {x["account_id"] for x in wf_a["accounts"]} == {"jira", "slack"}
    assert {x["account_id"] for x in wf_b["accounts"]} == {"jira"}
    a_slack = next(x for x in wf_a["accounts"] if x["account_id"] == "slack")
    assert a_slack["focus"] and a_slack["focus"][0]["id"] == "C-PIN"
    # output destination
    assert wf_a["output"]["node_label"] != wf_b["output"]["node_label"]
    assert wf_b["output"]["node_label"] == "Brief to you"
    # rail captions — the per-edge item counts
    assert wf_a["counts"]["noun"] != wf_b["counts"]["noun"]
    assert wf_b["counts"]["noun"] == "brief"


@pytest.mark.asyncio
async def test_the_rail_captions_count_this_automations_own_edges():
    from app.agent.automations.workflow import counts_block, workflow_payload

    uid = await _mk_user()
    spec = _raw()
    spec["trigger"]["sources"].append(
        {"id": "jissue", "mode": "poll", "connector_id": "jira",
         "event": "issue_created", "poll_interval_s": 300,
         "dedupe_key": "event.key"},
    )
    a = await _mk(uid, spec)
    async with async_session_maker() as db:
        wf = await workflow_payload(
            db, automation=await db.get(Automation, a.id), user_id=uid)
    assert wf["counts"]["items_per_fire"] == 2
    assert counts_block({"trigger": {"sources": []}}, "reads_only")[
        "items_per_fire"] == 1


@pytest.mark.asyncio
async def test_the_workflow_rev_the_commit_checks_is_on_the_payload():
    """The app drafts against a rev; serving it only on the commit
    response meant a fresh GET could not tell what it was holding."""
    from app.agent.automations.workflow import add_rule, workflow_payload

    uid = await _mk_user()
    a = await _mk(uid)
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        first = (await workflow_payload(
            db, automation=row, user_id=uid))["workflow_rev"]
        await add_rule(db, automation=row, text="Never on weekends")
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        second = (await workflow_payload(
            db, automation=row, user_id=uid))["workflow_rev"]
    assert second > first


@pytest.mark.asyncio
async def test_a_workflow_belongs_to_its_owner_only(monkeypatch):
    from app.api.automations import get_account_contents, get_workflow
    from app.agent.automations.service import AutomationNotFound
    from app.config import settings
    from fastapi import HTTPException

    owner = await _mk_user()
    stranger = await _mk_user()
    a = await _mk(owner)

    monkeypatch.setattr(settings, "automations_enabled", True)
    monkeypatch.setattr(settings, "user_id", stranger)
    with pytest.raises(HTTPException) as e:
        await get_workflow(a.id)
    assert e.value.status_code == 404
    with pytest.raises(HTTPException) as e:
        await get_account_contents(a.id, "slack")
    assert e.value.status_code == 404


# ──────────────────────────────────────── the seams the tests cannot run

def test_both_creation_doors_record_all_four_phase_kinds():
    """A source probe, in the repo's own idiom.

    The described path's phases wrap an LLM call and a real permission
    resolution, so a unit test of it is a test of the stubs. What CAN
    be checked without lying is that both doors still open every phase
    — a deleted `with rec.phase(...)` is a history that silently loses
    a chapter, and `build_ledger.record` skips ids it was never handed.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "app"
    template = (root / "api" / "automations.py").read_text()
    described = (root / "agent" / "automations"
                 / "describe_compile.py").read_text()
    for name, src in (("from_template", template),
                      ("compile_describe", described)):
        for phase in ('rec.phase("trigger")', 'rec.phase("agent")',
                      'rec.phase("output")', 'rec.phase(f"account:'):
            assert phase in src, f"{name} no longer records {phase}"
        assert "_build.record(" in src, f"{name} never persists the history"


def test_the_focus_and_contents_routes_are_registered():
    """A route the app cannot reach is a route that does not exist."""
    from app.api.automations import router

    paths = {(m, r.path) for r in router.routes
             for m in getattr(r, "methods", ())}
    base = "/automations/{automation_id}/workflow/accounts/{account_id}"
    assert ("GET", f"{base}/contents") in paths
    assert ("POST", f"{base}/focus") in paths
    assert ("DELETE", f"{base}/focus") in paths


def test_every_reader_the_envelope_advertises_actually_exists():
    from app.agent.automations import contents

    assert set(contents.SUPPORTED) == set(contents._READERS)
    for cid in contents.SUPPORTED:
        assert callable(contents._READERS[cid])
