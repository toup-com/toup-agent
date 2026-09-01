# agent-mode: automations/automation_threads/_turns are AGENT_ONLY.
"""R42 review — the "+" on a row must not choose where the run POSTS,
and one tap must not draw two dividers.

  1. THE BRIDGE  — `add_focus` bridges a pin into the owed write
                   destination. R42's own reader made the kind test
                   insufficient: `contents._read_teams` pins the CHAT
                   as kind `thread` (it really is the destination of
                   `teams__send_chat_message`) and every MESSAGE ROW in
                   it as kind `thread` too, with a `<chat>#<message>`
                   id. So the ordinary read-pin gesture asked to make a
                   message the automation's destination, and
                   `only_if_unpinned` cannot refuse for an unpinned
                   one. The bridge now also asks whether the pin names
                   a WHOLE CONTAINER (`contents.container_of`).
  2. ONE ACTION, — that same tap is two writers (the focus pin, then
     ONE DIVIDER    the destination), and each stamped its own EDITED
                    note: the identical back-to-back dividers in the
                    founder's thread.
"""

import json

import pytest

from app.db.database import async_session_maker
from app.db.models import Automation

from tests.test_workflow_api import _offline_platform  # noqa: F401
from tests.test_run_ledger_v3 import (  # noqa: F401 — shared fixtures
    REGISTRY_V2, _mk_user, _mk_automation_v2,
)
from tests.test_r39_canvas_pins import _grant_stub

#: Teams is the connector where one KIND names both a place and a row,
#: so it is the connector this file needs. `REGISTRY_V2` is jira+slack.
TEAMS_REGISTRY = dict(REGISTRY_V2)
TEAMS_REGISTRY["teams"] = {
    "connector_id": "teams", "push": False, "poll": False, "floor_s": 300,
    "rate_budget": {}, "scopes_read": [],
    "scopes_write_by_action": {"teams__send_chat_message": ["w"]},
    "target_param_by_action": {"teams__send_chat_message": "chat_id"},
    "events": [],
}


@pytest.fixture(autouse=True)
def _teams_platform(monkeypatch):
    async def _registry(user_id, force=False):
        return TEAMS_REGISTRY

    async def _conn_state(user_id):
        return {
            "jira": {"connector_id": "jira", "connected": True,
                     "status": "active", "scopes": ["r"]},
            "teams": {"connector_id": "teams", "connected": True,
                      "status": "active", "scopes": ["w"]},
        }
    monkeypatch.setattr(
        "app.agent.automations.registry.fetch_registry", _registry)
    monkeypatch.setattr(
        "app.agent.automations.registry.fetch_connection_state", _conn_state)


def _teams_raw() -> dict:
    """A brief that reads Jira and posts to a Teams chat it has not
    been pointed at yet — the state in which a pin can redirect it."""
    return {
        "version": 2, "name": "Teams brief", "mode": "auto",
        "trigger": {"sources": [
            {"id": "sched", "mode": "schedule",
             "schedule": {"cron_local": "0 8 * * 1-5"}}]},
        "steps": [
            {"id": "issues", "connector_id": "jira",
             "tool": "jira__search_issues", "params": {"jql": "x"},
             "collect": {"items_path": "issues", "fields": {"key": "key"},
                         "format": "{{item.key}}", "empty_text": "none"},
             "on_error": "skip"},
            {"id": "post", "connector_id": "teams",
             "tool": "teams__send_chat_message",
             "params": {"chat_id": "{{grant.target.id}}",
                        "text": "{{steps.issues.text}}"}},
        ],
    }


async def _mk_teams(uid: str):
    from app.agent.automations.spec import validate_spec
    return await _mk_automation_v2(
        uid, validate_spec(_teams_raw(), TEAMS_REGISTRY,
                           template_mode=True))


@pytest.mark.asyncio
async def test_a_teams_message_row_pin_does_not_choose_where_it_posts(
    monkeypatch,
):
    """The `#` half is a MESSAGE. Pinning it says "start here", and it
    must not end up in `chat_id`."""
    from app.agent.automations.workflow import add_focus

    calls = []
    monkeypatch.setattr(
        "app.agent.automations.registry.create_grant_request",
        _grant_stub(calls))
    uid = await _mk_user()
    a = await _mk_teams(uid)
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        out = await add_focus(
            db, automation=row, user_id=uid, account_id="teams",
            kind="thread", target_id="19:chat-abc#msg-77",
            label="Sara: can we move it")
    assert "destination" not in out, out
    assert calls == [], "no permission was asked for a row"
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        post = next(s for s in json.loads(row.spec_json)["steps"]
                    if s["id"] == "post")
    assert not post.get("grant_target"), post


@pytest.mark.asyncio
async def test_the_teams_chat_itself_still_sets_the_owed_destination(
    monkeypatch,
):
    """The half that must keep working: the chat IS the destination, and
    pinning it is how the founder answered "pick one and I'll set it
    there"."""
    from app.agent.automations.workflow import add_focus

    calls = []
    monkeypatch.setattr(
        "app.agent.automations.registry.create_grant_request",
        _grant_stub(calls))
    uid = await _mk_user()
    a = await _mk_teams(uid)
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        out = await add_focus(
            db, automation=row, user_id=uid, account_id="teams",
            kind="thread", target_id="19:chat-abc", label="Platform chat")
    assert out["destination"]["ok"] is True, out
    assert calls[0]["target"]["id"] == "19:chat-abc"


@pytest.mark.asyncio
async def test_one_tap_draws_one_divider(monkeypatch):
    """Two writers, one action: the focus pin and the destination it
    bridges share the EDITED note rather than stacking identical ones."""
    from tests.test_workflow_api import _edited_notes
    from app.agent.automations.workflow import add_focus

    calls = []
    monkeypatch.setattr(
        "app.agent.automations.registry.create_grant_request",
        _grant_stub(calls))
    uid = await _mk_user()
    a = await _mk_teams(uid)
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        await add_focus(
            db, automation=row, user_id=uid, account_id="teams",
            kind="thread", target_id="19:chat-abc", label="Platform chat")
    assert calls, "the destination half ran — otherwise this proves nothing"
    assert await _edited_notes(a.id) == 1
