# agent-mode: automations/automation_threads/_turns are AGENT_ONLY.
"""R39 — the canvas's pins get teeth.

  1. PINS ON CONTENTS   — every row/group carries the exact focus entry
                          that tapping "+" on it MEANS, in the pin
                          endpoint's own vocabulary. The app sends it
                          verbatim; a null pin draws no "+" at all.
  2. ONE RUNNABILITY    — `run_blockers` is the single predicate behind
     PREDICATE            run-now's 409, the home card's meta and the
                          thread agent's grounding (founder P6: three
                          surfaces, two answers).
  3. THE TRIGGER'S OWN  — an event automation stops wearing a schedule
     VOCABULARY           it does not have (founder P12), and the
                          presets stop selling the Morning brief's
                          commute (founder P13).
  4. NOTES ON PINS      — a pin can carry the user's own instruction,
                          and re-pinning with a new note is an edit.
  5. THE PIN STEERS THE — `_apply_focus_scope` fills a read's EMPTY
     RUN                  target from a pin. R42 REVERSED the rest of
                          it: pins rank, they never filter (founder
                          P6), so a broad query is left alone and the
                          pins reach the ranking step instead.
  6. PLAN TENSE         — the Steps sheet describes what an automation
                          WILL do; "Checked your calendar" on a
                          never-run automation reads as a record that
                          does not exist (founder P18).
"""

import datetime as _dt
import json
import uuid
from types import SimpleNamespace

import pytest

from app.config import settings
from app.db.database import async_session_maker
from app.db.models import Automation, User

from tests.test_workflow_api import _offline_platform  # noqa: F401
from tests.test_run_ledger_v3 import (  # noqa: F401 — shared fixtures
    REGISTRY_V2, _mk_user, _mk_automation_v2,
)


def _raw(*, granted: bool = True):
    """The R38 canvas spec, with the write step's grant OPTIONAL — an
    ungranted write is the whole subject of `run_blockers`."""
    post = {
        "id": "post", "connector_id": "slack",
        "tool": "slack__send_message",
        "params": {"channel": "{{grant.target.id}}",
                   "text": "{{steps.issues.text}}"},
    }
    if granted:
        post["grant_id"] = "g-1"
        post["grant_target"] = {"kind": "channel", "id": "C-PIN",
                                "label": "#platform"}
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
            post,
        ],
    }


async def _mk(uid: str, raw: dict = None):
    from app.agent.automations.spec import validate_spec
    return await _mk_automation_v2(
        uid, validate_spec(raw or _raw(), REGISTRY_V2, template_mode=True),
    )


def _ok(payload: dict) -> dict:
    return {"kind": "ok", "content": json.dumps(payload)}


_LIVE = {"connected": True, "status": "active"}


# ──────────────────────────────────────────────── 1. pins on contents

@pytest.mark.asyncio
async def test_a_gmail_read_asks_for_bodies_because_bare_ids_have_no_subject(
    monkeypatch,
):
    """gmail's include_body=False strips the list to bare {id, threadId}
    — which is how every Gmail row rendered as "(no subject)" with no
    sender and no time on the founder's canvas."""
    from app.agent.automations import contents
    seen = []

    async def _dispatch(user_id, *, connector_id, tool_name, tool_input,
                        **kw):
        seen.append(tool_input)
        return _ok({"messages": [{
            "id": "m1",
            "headers": {"From": "Sara Chen <sara@x.com>",
                        "Subject": "Re: launch",
                        "Date": "Fri, 29 Aug 2026 09:14:00 +0000"},
            "snippet": "can we move it",
        }]})
    monkeypatch.setattr(
        "app.agent.automations.registry.dispatch_via_platform", _dispatch,
    )
    env = await contents.account_contents(
        "u", connector_id="gmail", connection=_LIVE,
    )
    assert seen[0]["include_body"] is True
    item = env["groups"][0]["items"][0]
    assert item["title"] == "Re: launch" and "Sara Chen" in item["sub"]


@pytest.mark.asyncio
async def test_a_gmail_row_pins_its_conversation_and_the_group_its_sender(
    monkeypatch,
):
    """R42: a row pins the mail it IS. Carrying the sender on every row
    made one tap tick every other mail from Sara — the group a person
    pin makes is where a whole correspondent is pinned."""
    from app.agent.automations import contents
    from app.agent.automations.spec import FOCUS_KINDS

    async def _dispatch(user_id, *, connector_id, tool_name, tool_input,
                        **kw):
        return _ok({"messages": [
            {"id": "m1", "threadId": "t1",
             "headers": {"From": "Sara Chen <SARA@X.com>",
                         "Subject": "Re: launch",
                         "Date": "Fri, 29 Aug 2026 09:14:00 +0000"},
             "snippet": "ok"},
            {"id": "m2", "threadId": "t2",
             "headers": {"From": "Sara Chen <SARA@X.com>",
                         "Subject": "Budget"}},
        ]})
    monkeypatch.setattr(
        "app.agent.automations.registry.dispatch_via_platform", _dispatch,
    )
    env = await contents.account_contents(
        "u", connector_id="gmail", connection=_LIVE,
        focus=[{"kind": "person", "id": "sara@x.com", "label": "Sara Chen"}],
    )
    sender = env["groups"][0]
    assert sender["pinned"] and sender["pin"] == {
        "kind": "person", "id": "sara@x.com", "label": "Sara Chen"}
    pins = [i["pin"] for i in sender["items"]]
    assert pins == [{"kind": "thread", "id": "t1", "label": "Re: launch"},
                    {"kind": "thread", "id": "t2", "label": "Budget"}]
    assert all(p["kind"] in FOCUS_KINDS for p in pins)


@pytest.mark.asyncio
async def test_a_mail_with_no_id_carries_no_pin(monkeypatch):
    """A row that cannot say WHICH mail it is must claim no check — it
    would lose it on the next read."""
    from app.agent.automations import contents

    async def _dispatch(user_id, *, connector_id, tool_name, tool_input,
                        **kw):
        return _ok({"messages": [{
            "headers": {"Subject": "System notice"},
        }]})
    monkeypatch.setattr(
        "app.agent.automations.registry.dispatch_via_platform", _dispatch,
    )
    env = await contents.account_contents(
        "u", connector_id="gmail", connection=_LIVE,
    )
    assert env["groups"][0]["items"][0]["pin"] is None


@pytest.mark.asyncio
async def test_a_person_pin_orders_the_mailbox_it_never_scopes_it(
    monkeypatch,
):
    """R42, founder P6: the pinned query used to REPLACE the recent one,
    so the first pin a user made was the last mail they could ever pick
    in here."""
    from app.agent.automations import contents
    seen = []

    async def _dispatch(user_id, *, connector_id, tool_name, tool_input,
                        **kw):
        seen.append(tool_input.get("query") or "")
        return _ok({"messages": [{"id": "m1", "threadId": "t1",
                                  "headers": {"Subject": "S"}}]})
    monkeypatch.setattr(
        "app.agent.automations.registry.dispatch_via_platform", _dispatch,
    )
    env = await contents.account_contents(
        "u", connector_id="gmail", connection=_LIVE,
        focus=[{"kind": "person", "id": "sara@x.com", "label": "Sara"}],
    )
    assert sorted(seen) == ["", "from:sara@x.com"]
    assert [(g["key"], g["pinned"]) for g in env["groups"]] == [
        ("from:sara@x.com", True), ("recent", False)]


@pytest.mark.asyncio
async def test_an_outlook_row_pins_the_message_and_keeps_the_cheap_read(
    monkeypatch,
):
    """outlook's include_body=False still $selects subject/from/preview
    — only gmail needs the expensive form. That $select carries no
    conversation id, so an Outlook row pins the message it is."""
    from app.agent.automations import contents
    seen = []

    async def _dispatch(user_id, *, connector_id, tool_name, tool_input,
                        **kw):
        seen.append(tool_input)
        return _ok({"messages": [{
            "id": "m1", "from": "Omid <omid@x.com>", "subject": "Budget",
            "received_at": "2026-08-29T09:14:00Z", "preview": "numbers",
        }]})
    monkeypatch.setattr(
        "app.agent.automations.registry.dispatch_via_platform", _dispatch,
    )
    env = await contents.account_contents(
        "u", connector_id="outlook", connection=_LIVE,
    )
    assert seen[0]["include_body"] is False
    pin = env["groups"][0]["items"][0]["pin"]
    assert pin == {"kind": "thread", "id": "m1", "label": "Budget"}


@pytest.mark.asyncio
async def test_a_slack_row_pins_its_own_thread_and_the_group_the_channel(
    monkeypatch,
):
    """R42, founder P4: every row carried its CHANNEL, and the app reads
    "is this row pinned?" off that id — so one tap on one #all-toup
    message drew a checkmark on all ten while the badge said 1. A reply
    and its parent are one conversation and one pin; two separate
    messages are two."""
    from app.agent.automations import contents

    async def _dispatch(user_id, *, connector_id, tool_name, tool_input,
                        **kw):
        if tool_name == "slack__list_channels":
            return _ok({"channels": [
                {"id": "C1", "name": "eng", "is_member": True},
            ]})
        return _ok({"messages": [
            {"ts": "1756400000.0", "from": "Sara", "text": "shipping"},
            {"ts": "1756400100.0", "from": "Omid", "text": "blockers?",
             "reply_count": 2, "thread_ts": "1756400100.0"},
            {"ts": "1756400200.0", "from": "Ali", "text": "none",
             "in_thread_of": "1756400100.0"},
        ]})
    monkeypatch.setattr(
        "app.agent.automations.registry.dispatch_via_platform", _dispatch,
    )
    env = await contents.account_contents(
        "u", connector_id="slack", connection=_LIVE,
    )
    assert env["groups"][0]["pin"] == {"kind": "channel", "id": "C1",
                                       "label": "#eng"}
    ids = [i["pin"]["id"] for i in env["groups"][0]["items"]]
    assert ids == ["C1#1756400000.0", "C1#1756400100.0", "C1#1756400100.0"]
    assert env["groups"][0]["items"][0]["pin"]["label"] == "Sara: shipping"


@pytest.mark.asyncio
async def test_a_slack_pin_orders_the_channels_it_never_hides_them(
    monkeypatch,
):
    """R42, founder P6: `slack__list_channels` was skipped outright when
    any pin existed, so the sheet showed that one channel and nothing
    else could ever be picked. And a pinned channel the listing does not
    confirm is still read — the user pinned it."""
    from app.agent.automations import contents
    seen = []

    async def _dispatch(user_id, *, connector_id, tool_name, tool_input,
                        **kw):
        seen.append(tool_name)
        if tool_name == "slack__list_channels":
            return _ok({"channels": [
                {"id": "C1", "name": "eng", "is_member": True},
                {"id": "C2", "name": "all-toup", "is_member": True},
            ]})
        return _ok({"messages": [
            {"ts": "1756400000.0", "from": "Sara", "text": "shipping"},
        ]})
    monkeypatch.setattr(
        "app.agent.automations.registry.dispatch_via_platform", _dispatch,
    )
    env = await contents.account_contents(
        "u", connector_id="slack", connection=_LIVE,
        focus=[{"kind": "thread", "id": "C2#1756400000.0",
                "label": "Sara: shipping"}],
    )
    assert "slack__list_channels" in seen
    assert [(g["key"], g["label"], g["pinned"]) for g in env["groups"]] == [
        ("C2", "#all-toup", True), ("C1", "#eng", False)]


@pytest.mark.asyncio
async def test_a_jira_row_pins_its_ticket_and_its_group_the_project(
    monkeypatch,
):
    """"SCRUM-1" → project "SCRUM": the container a read can be aimed
    at, and now the GROUP the ticket sits in. A key with no dash names
    no project, so that group is a plain bucket with no pin of its own.
    """
    from app.agent.automations import contents

    async def _dispatch(user_id, *, connector_id, tool_name, tool_input,
                        **kw):
        return _ok({"issues": [
            {"key": "ENG-12", "summary": "Ship", "status": "To Do"},
            {"key": "ODD", "summary": "No project", "status": "To Do"},
        ]})
    monkeypatch.setattr(
        "app.agent.automations.registry.dispatch_via_platform", _dispatch,
    )
    env = await contents.account_contents(
        "u", connector_id="jira", connection=_LIVE,
    )
    groups = {g["key"]: g for g in env["groups"]}
    assert groups["ENG"]["pin"] == {"kind": "project", "id": "ENG",
                                    "label": "ENG"}
    assert groups["ENG"]["items"][0]["pin"] == {
        "kind": "ticket", "id": "ENG-12", "label": "ENG-12"}
    assert groups["other"]["pin"] is None
    assert groups["other"]["items"][0]["pin"] == {
        "kind": "ticket", "id": "ODD", "label": "ODD"}


@pytest.mark.asyncio
async def test_a_jira_pin_leads_the_groups_it_never_enters_the_jql(
    monkeypatch,
):
    """R42, founder P6: the pin used to prefix the JQL with
    `project in (…)`, so every other project vanished from the sheet the
    moment one was pinned — and the ORDER BY has to stay at the very
    end, which a scope built by string prefixing is one edit away from
    breaking."""
    from app.agent.automations import contents
    seen = []

    async def _dispatch(user_id, *, connector_id, tool_name, tool_input,
                        **kw):
        seen.append(tool_input["jql"])
        return _ok({"issues": [
            {"key": "ENG-12", "summary": "Ship", "project": "ENG"},
            {"key": "OPS-4", "summary": "Rotate", "project": "OPS"},
        ]})
    monkeypatch.setattr(
        "app.agent.automations.registry.dispatch_via_platform", _dispatch,
    )
    env = await contents.account_contents(
        "u", connector_id="jira", connection=_LIVE,
        focus=[{"kind": "project", "id": "OPS", "label": "OPS"}],
    )
    assert "project in" not in seen[0]
    assert seen[0].endswith("ORDER BY duedate ASC, updated DESC")
    assert [(g["key"], g["pinned"]) for g in env["groups"]] == [
        ("OPS", True), ("ENG", False)]


@pytest.mark.asyncio
async def test_a_github_row_pins_its_pr_and_its_group_the_repo(monkeypatch):
    from app.agent.automations import contents

    async def _dispatch(user_id, *, connector_id, tool_name, tool_input,
                        **kw):
        if tool_name == "github__list_repos":
            return _ok({"repos": [{"full_name": "toup/platform"}]})
        return _ok({"issues": [
            {"number": 7, "title": "A PR", "user": "sara",
             "is_pull_request": True},
        ]})
    monkeypatch.setattr(
        "app.agent.automations.registry.dispatch_via_platform", _dispatch,
    )
    env = await contents.account_contents(
        "u", connector_id="github", connection=_LIVE,
    )
    assert env["groups"][0]["pin"] == {
        "kind": "repo", "id": "toup/platform", "label": "toup/platform"}
    # R42: the row is the pull request, and "owner/repo#7" is both its
    # own name and the repo it hangs off.
    assert env["groups"][0]["items"][0]["pin"] == {
        "kind": "ticket", "id": "toup/platform#7", "label": "#7 A PR"}


@pytest.mark.asyncio
async def test_teams_is_readable_now_and_its_rows_pin_themselves(monkeypatch):
    """Teams was not in SUPPORTED at all, so the one connector the
    Morning work brief reads a chat from answered "no way to look
    inside" — with an expired credential underneath that the sheet
    therefore never surfaced. And Graph bodies arrive as HTML; ink on
    the phone must never carry markup."""
    from app.agent.automations import contents

    assert "teams" in contents.SUPPORTED

    async def _dispatch(user_id, *, connector_id, tool_name, tool_input,
                        **kw):
        if tool_name == "teams__list_chats":
            return _ok({"chats": [{"id": "chat1", "topic": "Quarterly"}]})
        assert tool_name == "teams__read_chat_messages"
        assert tool_input["chat_id"] == "chat1"
        return _ok({"messages": [{
            "id": "m1", "sender": "Sara",
            "body": "<p>Hello <b>world</b></p>",
            "body_content_type": "html",
            "created_at": "2026-08-29T09:14:00Z",
        }]})
    monkeypatch.setattr(
        "app.agent.automations.registry.dispatch_via_platform", _dispatch,
    )
    env = await contents.account_contents(
        "u", connector_id="teams", connection=_LIVE,
    )
    item = env["groups"][0]["items"][0]
    assert item["sub"] == "Hello world"
    assert env["groups"][0]["pin"] == {"kind": "thread", "id": "chat1",
                                       "label": "Quarterly"}
    # A Graph chat message has no thread of its own, so the row IS the
    # message — and pinning one must not tick the whole chat (R42).
    assert item["pin"] == {"kind": "thread", "id": "chat1#m1",
                           "label": "Sara: Hello world"}


@pytest.mark.asyncio
async def test_calendar_events_are_moments_not_places(monkeypatch):
    from app.agent.automations import contents

    assert "calendar" in contents.SUPPORTED

    async def _dispatch(user_id, *, connector_id, tool_name, tool_input,
                        **kw):
        assert tool_name == "calendar__list_events"
        return _ok({"events": [{
            "id": "e1", "summary": "Board sync",
            "start": {"dateTime": "2026-08-31T10:00:00Z"},
            "location": "Room 4", "attendee_count": 5,
        }]})
    monkeypatch.setattr(
        "app.agent.automations.registry.dispatch_via_platform", _dispatch,
    )
    env = await contents.account_contents(
        "u", connector_id="calendar", connection=_LIVE,
    )
    item = env["groups"][0]["items"][0]
    assert item["title"] == "Board sync" and item["pin"] is None
    assert "Room 4" in item["sub"] and "5 people" in item["sub"]


@pytest.mark.asyncio
async def test_a_dead_credential_outranks_a_missing_reader(monkeypatch):
    """R39 reordered the gates: an expired credential is the truer
    answer than "no way to look inside" for a connector with no reader
    — it is the one with a Reconnect action on it."""
    from app.agent.automations import contents

    async def _dispatch(user_id, **kw):
        raise AssertionError("a known-dead connection is never called")
    monkeypatch.setattr(
        "app.agent.automations.registry.dispatch_via_platform", _dispatch,
    )
    expired = await contents.account_contents(
        "u", connector_id="notion",
        connection={"connected": True, "status": "reauth_required"},
    )
    assert expired["reason"]["code"] == "reconnect"

    missing = await contents.account_contents(
        "u", connector_id="notion", connection={"connected": False},
    )
    assert missing["reason"]["code"] == "not_connected"
    assert missing["reason"]["consent_url"].endswith("/notion")

    # With a live connection the old answer survives, by name.
    unsupported = await contents.account_contents(
        "u", connector_id="notion", connection=_LIVE,
    )
    assert unsupported["reason"]["code"] == "not_supported"
    assert "Notion" in unsupported["reason"]["sentence"]


# ─────────────────────────────────────── 2. one runnability predicate

def test_an_unpinned_write_is_one_blocker_that_names_the_write():
    from app.agent.automations.workflow import run_blockers

    out = run_blockers(_raw(granted=False))
    assert len(out) == 1
    b = out[0]
    assert b["code"] == "needs_destination"
    assert b["connector_id"] == "slack"
    assert b["tool"] == "slack__send_message"
    assert "post to Slack" in b["sentence"]
    assert "tell me where" in b["sentence"]


def test_a_granted_write_still_awaiting_its_target_blocks():
    """grant_id alone is not enough while the params still reference
    {{grant.target.…}} and no target is pinned — that write renders an
    empty channel."""
    from app.agent.automations.workflow import run_blockers

    raw = _raw(granted=True)
    raw["steps"][1].pop("grant_target")
    assert run_blockers(raw)[0]["code"] == "needs_destination"


def test_a_pinned_and_granted_write_blocks_nothing():
    from app.agent.automations.workflow import run_blockers

    assert run_blockers(_raw(granted=True)) == []


def test_agent_steps_and_v1_specs_never_block():
    from app.agent.automations.workflow import run_blockers

    raw = _raw(granted=False)
    raw["steps"].insert(1, {"id": "think", "kind": "agent",
                            "prompt": "Rank it.", "output_var": "x"})
    out = run_blockers(raw)
    assert len(out) == 1 and out[0]["tool"] == "slack__send_message"

    v1 = {"version": 1, "name": "old",
          "trigger": {"mode": "schedule",
                      "schedule": {"cron_local": "0 8 * * *"}},
          "action": {"connector_id": "slack",
                     "tool": "slack__send_message", "params": {}}}
    assert run_blockers(v1) == []


@pytest.mark.asyncio
async def test_run_now_refuses_an_unpinned_write_with_the_blocker_sentence(
    monkeypatch,
):
    """Founder P6 end to end: run-now's gate and the blocker speak the
    SAME sentence, because they are the same predicate."""
    from fastapi import HTTPException
    from app.api import automations as api
    from app.agent.automations.workflow import run_blockers

    uid = await _mk_user()
    aid = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(Automation(
            id=aid, user_id=uid, name="Ledger brief", status="draft",
            spec_json=json.dumps(_raw(granted=False)),
            trigger_mode="schedule",
        ))
        await db.commit()
    monkeypatch.setattr(settings, "automations_enabled", True)
    monkeypatch.setattr(settings, "user_id", uid)
    with pytest.raises(HTTPException) as exc:
        await api.run_now(aid)
    assert exc.value.status_code == 409
    detail = exc.value.detail
    assert detail["code"] == "needs_setup"
    blocker = run_blockers(_raw(granted=False))[0]["sentence"]
    assert blocker in detail["sentence"]
    assert detail["sentence"].startswith("It is not finished being set up")


@pytest.mark.asyncio
async def test_the_home_card_never_promises_a_first_run_the_gate_refuses():
    """"First run soon" stood in the same thread as run-now's
    needs_setup 409 — three surfaces, two answers."""
    from app.agent.automations.summary import summary_payload

    uid = await _mk_user()
    blocked = await _mk(uid, _raw(granted=False))
    runnable = await _mk(uid, _raw(granted=True))

    async with async_session_maker() as db:
        items = (await summary_payload(db, user_id=uid))["automations"]
    by_id = {i["id"]: i for i in items}
    assert by_id[blocked.id]["meta"].startswith(
        "Needs a destination first · ")
    assert "2 accounts" in by_id[blocked.id]["meta"]
    assert by_id[runnable.id]["meta"].startswith("First run ")


# ─────────────────────────────────── 3. the trigger's own vocabulary

def test_an_event_automation_speaks_event_not_schedule():
    """Founder P12: "New event → Slack" rendered as "On its own
    schedule", its sheet offered cron presets, and saving one silently
    repainted the event automation as a daily schedule."""
    from app.agent.automations.workflow import trigger_block

    raw = _raw(granted=True)
    raw["trigger"]["sources"] = [
        {"id": "jissue", "mode": "poll", "connector_id": "jira",
         "event": "issue_created", "poll_interval_s": 300,
         "dedupe_key": "event.key"},
    ]
    block = trigger_block(raw)
    assert block["kind"] == "event"
    assert block["label"] == "On new Jira issues"
    assert block["sub"] == "watches Jira"
    assert block["event"] == {
        "key": "issue_created", "connector_id": "jira",
        "sentence": "When a new Jira issue appears",
    }


def test_a_scheduled_automation_speaks_its_cron_sentence():
    from app.agent.automations.workflow import trigger_block

    block = trigger_block(_raw(granted=True))
    assert block["kind"] == "schedule"
    assert block["label"] == "Weekdays 8:00"
    assert block["event"] is None


def test_no_trigger_at_all_is_on_request():
    from app.agent.automations.workflow import trigger_block

    block = trigger_block({})
    assert block["kind"] == "manual"
    assert block["label"] == "On request"
    assert "when you ask" in block["sub"]


@pytest.mark.asyncio
async def test_the_workflow_payload_carries_the_trigger_block():
    from app.agent.automations.workflow import workflow_payload

    uid = await _mk_user()
    a = await _mk(uid)
    async with async_session_maker() as db:
        wf = await workflow_payload(
            db, automation=await db.get(Automation, a.id), user_id=uid)
    assert wf["trigger"]["kind"] == "schedule"
    # `schedule` stays beside it for older clients.
    assert wf["schedule"]["preset_id"] == "weekdays-8"


def test_schedule_presets_describe_the_cron_not_a_use_case():
    """Founder P13: "Finishes before your first meeting" / "Right before
    standup" were the Morning brief's own copy, served to every
    automation — a repo digest's schedule sheet was selling commute
    times."""
    from app.agent.automations.workflow import SCHEDULE_PRESETS

    for p in SCHEDULE_PRESETS:
        text = f"{p['sentence']} {p['sub']}".lower()
        for word in ("commute", "standup", "meeting"):
            assert word not in text, (p["id"], word)


# ────────────────────────────────────────────────── 4. notes on pins

@pytest.mark.asyncio
async def test_a_note_rides_the_pin_into_the_spec():
    from app.agent.automations.workflow import add_focus

    uid = await _mk_user()
    a = await _mk(uid)
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        out = await add_focus(
            db, automation=row, user_id=uid, account_id="jira",
            kind="project", target_id="ENG", label="ENG",
            note="anything due this week outranks the rest",
        )
    assert "It starts at ENG" in out["sentence"]
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        pin = json.loads(row.spec_json)["focus"]["jira"][0]
    assert pin["note"] == "anything due this week outranks the rest"


@pytest.mark.asyncio
async def test_repinning_with_a_new_note_updates_in_place_and_says_noted():
    """The same place tapped again with a NOTE is how the app writes or
    edits the per-pin instruction — an update, not a double tap."""
    from app.agent.automations.workflow import add_focus

    uid = await _mk_user()
    a = await _mk(uid)
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        await add_focus(db, automation=row, user_id=uid, account_id="jira",
                        kind="project", target_id="ENG", label="ENG",
                        note="first thought")
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        out = await add_focus(
            db, automation=row, user_id=uid, account_id="jira",
            kind="project", target_id="ENG", label="ENG",
            note="second thought",
        )
    assert out["sentence"].startswith("Noted — ")
    assert len(out["focus"]) == 1
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        pins = json.loads(row.spec_json)["focus"]["jira"]
    assert len(pins) == 1 and pins[0]["note"] == "second thought"


@pytest.mark.asyncio
async def test_repinning_with_no_note_stays_the_polite_noop():
    from app.agent.automations.workflow import add_focus

    uid = await _mk_user()
    a = await _mk(uid)
    for _ in range(2):
        async with async_session_maker() as db:
            row = await db.get(Automation, a.id)
            out = await add_focus(
                db, automation=row, user_id=uid, account_id="jira",
                kind="project", target_id="ENG", label="ENG",
            )
    assert "already starts" in out["sentence"]
    assert len(out["focus"]) == 1


def test_a_non_string_note_errors_but_the_pin_survives_without_it():
    """The exact semantics of `validate_focus`: the error is reported so
    the caller can refuse, and the RETURNED pin never carries the
    malformed note — a caller that ignores `errors` still persists
    nothing malformed."""
    from app.agent.automations.spec import validate_focus

    errors = []
    out = validate_focus(
        {"focus": {"slack": [{"kind": "channel", "id": "C1",
                              "label": "#eng", "note": 42}]}},
        errors,
    )
    assert any(e["code"] == "bad_focus_note" for e in errors)
    assert out["slack"] == [{"kind": "channel", "id": "C1",
                             "label": "#eng"}]


def test_an_overlong_note_is_clamped_and_an_empty_one_dropped():
    from app.agent.automations.spec import FOCUS_NOTE_MAX, validate_focus

    errors = []
    out = validate_focus(
        {"focus": {"slack": [
            {"kind": "channel", "id": "C1", "note": "x" * 400},
            {"kind": "channel", "id": "C2", "note": "   "},
        ]}},
        errors,
    )
    assert errors == []
    assert out["slack"][0]["note"] == "x" * FOCUS_NOTE_MAX
    assert "note" not in out["slack"][1]


def test_the_notes_leaf_reaches_the_render_ctx_joined():
    from app.agent.automations.spec import focus_render_ctx

    ctx = focus_render_ctx({
        "gmail": [
            {"kind": "person", "id": "boss@x.com", "label": "Sara",
             "note": "anything from her outranks the rest"},
            {"kind": "person", "id": "peer@x.com", "label": "Omid"},
        ],
    })
    assert ctx["gmail"]["notes"] == (
        "Sara: anything from her outranks the rest")
    assert ctx["gmail"]["count"] == 2


# ─────────────────────────────── 4b. the pin that is also a destination

def _grant_stub(calls: list):
    async def _create(user_id, *, connector_id, tool_name, target,
                      cadence=None, mode="confirm", summary,
                      preview=None, automation_id=None):
        calls.append({"connector_id": connector_id, "tool": tool_name,
                      "target": target, "summary": summary})
        return {"id": "g-new", "automation_id": automation_id,
                "connector_id": connector_id, "tool_name": tool_name,
                "target": target, "cadence": None, "mode": mode,
                "summary": summary, "status": "pending",
                "created_at": "2026-08-30T00:00:00Z", "expires_at": None}
    return _create


@pytest.mark.asyncio
async def test_pinning_the_channel_sets_the_owed_write_destination_too(
    monkeypatch,
):
    """The founder followed "pick all-toup and I'll set it there" by
    tapping "+" on all-toup, and a bare focus left the run refusing
    about the very thing he had just picked. One tap now sets both —
    and closes the run blocker."""
    from app.agent.automations import ledger
    from app.agent.automations.workflow import add_focus, run_blockers

    calls = []
    monkeypatch.setattr(
        "app.agent.automations.registry.create_grant_request",
        _grant_stub(calls),
    )
    uid = await _mk_user()
    a = await _mk(uid, _raw(granted=False))
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        out = await add_focus(
            db, automation=row, user_id=uid, account_id="slack",
            kind="channel", target_id="C-TOUP", label="#all-toup",
        )
    assert out["destination"]["ok"] is True
    assert out["destination"]["grant_id"] == "g-new"
    assert "It starts at #all-toup" in out["sentence"]
    assert "permission" in out["sentence"]
    assert calls[0]["target"]["id"] == "C-TOUP"

    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        raw = json.loads(row.spec_json)
    post = next(s for s in raw["steps"] if s["id"] == "post")
    assert post["grant_id"] == "g-new"
    assert post["grant_target"]["id"] == "C-TOUP"
    assert run_blockers(raw) == [], "the tap must close the blocker"

    # The ask landed in the thread as a needs_you turn.
    async with async_session_maker() as db:
        thread = await ledger.thread_for(db, a.id)
        turns, _ = await ledger.list_turns(db, thread_id=thread.id)
    needs = [t for t in turns if t["kind"] == "needs_you"]
    assert needs and needs[-1]["grant_request_id"] == "g-new"


@pytest.mark.asyncio
async def test_a_pin_never_redirects_an_approved_destination(monkeypatch):
    """`only_if_unpinned` is the canvas caller's safety: a "+" must
    never silently point an already-approved destination somewhere
    else."""
    from app.agent.automations.workflow import add_focus

    calls = []
    monkeypatch.setattr(
        "app.agent.automations.registry.create_grant_request",
        _grant_stub(calls),
    )

    # The focus write re-arms the armed automation, and arm re-snapshots
    # every write step's grant_target from the platform's grant. The
    # shared fixture's bind_grant answers a MINIMAL grant (no target),
    # which would wipe the pin from the stub side; a grant already bound
    # to an automation skips the bind and keeps its target, like the
    # real platform's payload does.
    async def _bound_grant(user_id, grant_id):
        return {"id": grant_id, "status": "approved",
                "connector_id": "slack",
                "tool_name": "slack__send_message",
                "target": {"kind": "channel", "id": "C-PIN",
                           "label": "#platform"},
                "mode": "auto", "automation_id": "already-bound"}
    monkeypatch.setattr(
        "app.agent.automations.registry.fetch_grant", _bound_grant,
    )

    uid = await _mk_user()
    a = await _mk(uid, _raw(granted=True))   # pinned at C-PIN already
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        out = await add_focus(
            db, automation=row, user_id=uid, account_id="slack",
            kind="channel", target_id="C-ELSE", label="#elsewhere",
        )
    assert "destination" not in out
    assert calls == []
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        post = next(s for s in json.loads(row.spec_json)["steps"]
                    if s["id"] == "post")
    assert post["grant_target"]["id"] == "C-PIN"


@pytest.mark.asyncio
async def test_a_project_pin_asks_for_no_destination(monkeypatch):
    """Only a pin that NAMES A PLACE can be a write destination; a
    project scopes a read and nothing else."""
    from app.agent.automations.workflow import add_focus

    calls = []
    monkeypatch.setattr(
        "app.agent.automations.registry.create_grant_request",
        _grant_stub(calls),
    )
    uid = await _mk_user()
    a = await _mk(uid, _raw(granted=False))  # slack write still owed
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        out = await add_focus(
            db, automation=row, user_id=uid, account_id="jira",
            kind="project", target_id="ENG", label="ENG",
        )
    assert "destination" not in out
    assert calls == []


@pytest.mark.asyncio
async def test_a_message_thread_pin_does_not_redirect_the_write(monkeypatch):
    """R42: `thread` names two things and only one of them is a place.

    `contents._read_teams` pins a Teams CHAT as kind `thread`, and that
    chat IS where `teams__send_chat_message` posts — it must keep
    bridging. Everywhere else a `thread` is a MESSAGE thread, a preview
    ROW: bridging it would point the automation's POST at whatever the
    user last pinned to READ, and `only_if_unpinned` cannot catch that
    — an unpinned destination has nothing to refuse with.
    """
    from app.agent.automations.workflow import _names_a_destination, add_focus

    calls = []
    monkeypatch.setattr(
        "app.agent.automations.registry.create_grant_request",
        _grant_stub(calls),
    )
    uid = await _mk_user()
    a = await _mk(uid, _raw(granted=False))   # the slack write is still owed
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        out = await add_focus(
            db, automation=row, user_id=uid, account_id="slack",
            kind="thread", target_id="1724968000.001", label="Re: launch",
        )
    assert "destination" not in out
    assert calls == []
    assert not _names_a_destination("slack", "thread")
    assert _names_a_destination("teams", "thread")
    assert _names_a_destination("slack", "channel")


@pytest.mark.asyncio
async def test_a_failed_grant_mint_does_not_poison_the_focus_pin(
    monkeypatch,
):
    """The pin stood; the destination half can be asked for again — and
    the sentence says what happened rather than pretending."""
    from app.agent.automations.workflow import add_focus, focus_of

    async def _none(user_id, **kw):
        return None
    monkeypatch.setattr(
        "app.agent.automations.registry.create_grant_request", _none,
    )
    uid = await _mk_user()
    a = await _mk(uid, _raw(granted=False))
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        out = await add_focus(
            db, automation=row, user_id=uid, account_id="slack",
            kind="channel", target_id="C-TOUP", label="#all-toup",
        )
    assert out["destination"]["ok"] is False
    assert "could not be prepared" in out["destination"]["sentence"]
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        raw = json.loads(row.spec_json)
    assert focus_of(raw)["slack"][0]["id"] == "C-TOUP"
    post = next(s for s in raw["steps"] if s["id"] == "post")
    assert not post.get("grant_id"), "no grant was minted"


@pytest.mark.asyncio
async def test_no_write_step_means_no_destination_to_pin():
    from app.agent.automations.workflow import pin_write_destination

    raw = _raw(granted=False)
    raw["steps"] = [raw["steps"][0]]         # reads only
    fake = SimpleNamespace(id="a1", name="Reads",
                           spec_json=json.dumps(raw))
    out = await pin_write_destination(
        None, automation=fake, user_id="u", connector_id="slack",
        target={"kind": "channel", "id": "C1", "label": "#eng"},
    )
    assert out is None


# ──────────────────────────────────────── 5. the pin steers the run

def test_a_broad_query_is_left_exactly_as_the_spec_wrote_it():
    """R42, founder P6 — pins RANK, they never FILTER.

    R39 composed them into the provider call: a person pin became
    `from:boss@x.com` on the Gmail query, a project pin wrapped the
    step's own JQL in `project in (…) AND (…)`. Material the user did
    not pin was then never FETCHED, so no ranking step could see it and
    nothing downstream could put the pinned item first — which is the
    whole thing a pin is for. The Jira composition was also invalid
    JQL: every shipped template's JQL ends in ORDER BY, which JQL
    forbids inside parentheses, so the read 400'd, `on_error: continue`
    swallowed it, and the brief blamed a healthy board.
    """
    from app.agent.automations.executor_v2 import _apply_focus_scope

    gmail = {"query": "is:unread"}
    assert _apply_focus_scope(
        "gmail", "gmail__list_messages", dict(gmail),
        [{"kind": "person", "id": "boss@x.com"},
         {"kind": "label", "id": "urgent"}],
    ) == gmail

    outlook = {"query": "isRead:false"}
    assert _apply_focus_scope(
        "outlook", "outlook__list_messages", dict(outlook),
        [{"kind": "person", "id": "boss@x.com"}],
    ) == outlook

    jql = {"jql": "assignee = currentUser() AND statusCategory != Done "
                  "ORDER BY updated DESC"}
    assert _apply_focus_scope(
        "jira", "jira__search_issues", dict(jql),
        [{"kind": "project", "id": "ENG"}, {"kind": "project", "id": "OPS"}],
    ) == jql


def test_a_pin_fills_an_empty_target_and_never_overrides_a_set_one():
    """The one honest case: a tool that REQUIRES a target it does not
    have. Filling a hole is not filtering; replacing what the spec said
    is (R39 did, and a second pin was silently dropped)."""
    from app.agent.automations.executor_v2 import _apply_focus_scope

    filled = _apply_focus_scope(
        "slack", "slack__read_messages", {"channel": "", "limit": 10},
        [{"kind": "channel", "id": "C-PIN"},
         {"kind": "channel", "id": "C-TWO"}],
    )
    assert filled["channel"] == "C-PIN" and filled["limit"] == 10

    kept = {"channel": "C-OLD", "limit": 10}
    assert _apply_focus_scope(
        "slack", "slack__read_messages", dict(kept),
        [{"kind": "channel", "id": "C-PIN"}],
    ) == kept

    teams = _apply_focus_scope(
        "teams", "teams__read_chat_messages", {},
        [{"kind": "thread", "id": "chat9"}],
    )
    assert teams["chat_id"] == "chat9"
    assert _apply_focus_scope(
        "teams", "teams__read_chat_messages", {"chat_id": "old"},
        [{"kind": "thread", "id": "chat9"}],
    ) == {"chat_id": "old"}


def test_a_github_repo_pin_fills_only_a_repository_the_spec_left_empty():
    from app.agent.automations.executor_v2 import _apply_focus_scope

    out = _apply_focus_scope(
        "github", "github__list_issues", {"state": "open"},
        [{"kind": "repo", "id": "toup/platform"}],
    )
    assert (out["owner"], out["repo"], out["state"]) == (
        "toup", "platform", "open")

    named = {"owner": "someone", "repo": "else", "state": "open"}
    assert _apply_focus_scope(
        "github", "github__list_issues", dict(named),
        [{"kind": "repo", "id": "toup/platform"}],
    ) == named

    # A repo pin that names no owner half steers nothing.
    assert _apply_focus_scope(
        "github", "github__list_issues", {},
        [{"kind": "repo", "id": "just-a-name"}],
    ) == {}


def test_no_pins_unknown_tools_and_malformed_pins_change_nothing():
    """Pure and total — a pin must never break a read."""
    from app.agent.automations.executor_v2 import _apply_focus_scope

    params = {"channel": ""}
    assert _apply_focus_scope("slack", "slack__read_messages",
                              params, []) is params
    assert _apply_focus_scope(
        "slack", "slack__read_channels", dict(params),
        [{"kind": "channel", "id": "C1"}],
    ) == params
    assert _apply_focus_scope(
        "slack", "slack__read_messages", dict(params),
        ["not-a-dict", {"kind": "channel"}, {"no": "id"}],
    ) == params
    assert _apply_focus_scope(
        "slack", "slack__read_messages", "not-a-dict",
        [{"kind": "channel", "id": "C1"}],
    ) == "not-a-dict"


def test_a_calendar_read_gets_the_clock_the_spec_cannot_have():
    """R42 B1 — `calendar__list_events` windows only when asked and
    orders ASCENDING, and no shipped template passed a window, so every
    "your day's calendar" posted the oldest events in the account.
    `window_days` is spec vocabulary three templates already carry; it
    is popped here so the provider never sees a key its schema does not
    declare."""
    from datetime import datetime, timezone
    from app.agent.automations.executor_v2 import _apply_time_window

    clock = {"now": datetime(2026, 8, 31, 7, 30, 15, 123456,
                             tzinfo=timezone.utc)}
    out = _apply_time_window(
        "calendar__list_events",
        {"window_days": 7, "max_results": 25}, clock,
    )
    assert out == {"max_results": 25,
                   "time_min": "2026-08-31T07:30:15+00:00",
                   "time_max": "2026-09-07T07:30:15+00:00"}

    # Default horizon is one day, and a bound the spec set is kept.
    assert _apply_time_window(
        "calendar__list_events", {"max_results": 10}, clock,
    )["time_max"] == "2026-09-01T07:30:15+00:00"
    pinned = _apply_time_window(
        "calendar__list_events",
        {"time_min": "2020-01-01T00:00:00+00:00",
         "time_max": "2020-01-02T00:00:00+00:00"}, clock,
    )
    assert pinned["time_min"].startswith("2020-01-01")
    assert pinned["time_max"].startswith("2020-01-02")

    # No clock (a ctx without `_clock`) windows nothing, and no tool
    # ever receives `window_days`.
    assert _apply_time_window(
        "calendar__list_events", {"window_days": 3}, {},
    ) == {}
    assert _apply_time_window(
        "slack__read_messages", {"channel": "C1", "window_days": 5}, clock,
    ) == {"channel": "C1"}


@pytest.mark.asyncio
async def test_the_executor_applies_the_scope_at_the_read_seam(monkeypatch):
    """Before R39 an accepted pin reached a run only as a `{{focus.*}}`
    render root that no compiled step referenced — the canvas said "it
    starts at #all-toup" and the run read the whole account anyway."""
    from app.agent.automations.executor_v2 import _execute_read_step

    seen = []

    async def _dispatch(user_id, *, connector_id, tool_name, tool_input,
                        **kw):
        seen.append(tool_input)
        return {"kind": "ok", "content": "{}"}
    monkeypatch.setattr(
        "app.agent.automations.registry.dispatch_via_platform", _dispatch,
    )
    step = SimpleNamespace(
        id="read", connector_id="slack", tool="slack__read_messages",
        params_template={"channel": "", "limit": 10}, collect=None,
    )
    ctx = {"event": {}, "var": {}, "steps": {}, "focus": {},
           "_focus_pins": {"slack": [{"kind": "channel", "id": "C-PIN",
                                      "label": "#platform"}]},
           "_clock": {"now": _dt.datetime.now(_dt.timezone.utc)}}
    out = await _execute_read_step(
        SimpleNamespace(user_id="u", id="a1"), step, ctx,
    )
    assert out["ok"] is True
    assert seen[0]["channel"] == "C-PIN"


# ─────────────────────────────────────────────────── 6. plan tense

def test_plan_action_re_tenses_the_dictionary_heads():
    from app.services.automation_verbs import plan_action

    assert plan_action("Checked your calendar") == "Checks your calendar"
    assert plan_action("Told you in Slack") == "Tells you in Slack"
    assert plan_action("Drafted a reply") == "Drafts a reply"
    assert plan_action("Thought it through") == "Thinks it through"


def test_an_unknown_leading_verb_passes_through_unchanged():
    """An unknown verb in the past is legible; a bad transform is
    garbage."""
    from app.services.automation_verbs import plan_action

    assert plan_action("Summoned the daemon") == "Summoned the daemon"
    assert plan_action("") == ""
    assert plan_action("Checked") == "Checks"


def test_derived_steps_speak_plan_tense_and_stored_rows_stay_verbatim():
    """Founder P18: "Checked your calendar" above "WHAT IT DOES, IN
    ORDER" on a never-run automation reads as a record that does not
    exist. The derived branch is re-tensed; a stored row is the USER's
    own words and is served untouched."""
    from app.agent.automations.workflow import _steps_human
    from app.services.automation_verbs import plan_action

    derived = _steps_human(
        SimpleNamespace(steps_human_json=None), _raw(granted=True),
    )
    assert len(derived) == 2
    for s in derived:
        assert s["text"] == plan_action(s["text"]), (
            f"still in the past tense: {s['text']!r}")

    stored = _steps_human(
        SimpleNamespace(steps_human_json=json.dumps(
            [{"text": "Checked your calendar", "sub": "my words"}])),
        _raw(granted=True),
    )
    assert stored == [{"n": 1, "text": "Checked your calendar",
                       "sub": "my words"}]
