# agent-mode: automations/automation_threads/_turns/build_jobs are AGENT_ONLY.
"""R37 — the thread agent gets hands (founder's Inbox-summary session).

Fifteen screenshots, one shape: the operator of a workflow could not
operate it. "Run it now" got a question back; "whatever is there" got
the same question again; "keep it in this chat" was agreed to twice and
changed nothing; "Run it now then" was refused with the setup the agent
had just agreed to change. Four mechanisms, each pinned here:

  1. the intent gate classified thread turns `question` and stripped
     every automations tool but `list`, every connector read, and the
     skill prompt that names them (agent_runner.intent_for_channel);
  2. the full connector surface put the turn over the proxy's 128-tool
     tail-trim, eating exactly the reads the thread needed
     (agent_runner.scope_connector_tools);
  3. the grounding carried no current setup, so the agent answered
     from whatever the last 40 turns claimed (thread_agent._setup_lines);
  4. there was no one-call write-back for the one decision every setup
     ends on (service.set_destination_chat / pin_destination), and a
     lost answer had no honest retry (the replay branch answered
     `replayed` and scheduled nothing).
"""

import json
import uuid

import pytest

from app.config import settings
from app.db.database import async_session_maker
from app.db.models import Automation, User

REGISTRY = {
    "gmail": {
        "connector_id": "gmail", "push": True, "poll": True, "floor_s": 300,
        "rate_budget": {}, "scopes_read": ["r"],
        "scopes_write_by_action": {"gmail__create_draft": ["w"]},
        "target_param_by_action": {},
        "events": [{
            "key": "message_received", "description": "",
            "poll_args": {}, "params_required": [],
            "items_path": "messages", "dedupe_field": "id",
            "fields": {"id": "id"},
        }],
    },
    "slack": {
        "connector_id": "slack", "push": False, "poll": False, "floor_s": 300,
        "rate_budget": {}, "scopes_read": ["channels:read"],
        "scopes_write_by_action": {"slack__send_message": ["chat:write"]},
        "target_param_by_action": {"slack__send_message": "channel"},
        "events": [],
    },
}


def _inbox_spec(pinned: bool = False, grant_id: str = "") -> dict:
    """The founder's Inbox-summary shape: one read, one Slack post."""
    step_post = {
        "id": "post", "connector_id": "slack",
        "tool": "slack__send_message",
        "params": {"channel": "{{grant.target.id}}",
                   "text": "{{steps.mail.text}}"},
    }
    if grant_id:
        step_post["grant_id"] = grant_id
    if pinned:
        step_post["grant_target"] = {"kind": "channel", "id": "C1",
                                     "label": "#general"}
    return {
        "version": 2, "name": "Inbox summary", "mode": "auto",
        "trigger": {"sources": [{"id": "sched", "mode": "schedule",
                                 "schedule": {"cron_local": "0 8 * * 1-5"}}]},
        "steps": [
            {"id": "mail", "connector_id": "gmail",
             "tool": "gmail__list_messages",
             "params": {"query": "is:unread"}, "on_error": "continue",
             "collect": {"items_path": "messages",
                         "fields": {"subject": "subject"},
                         "format": "• {{item.subject}}", "limit": 8,
                         "empty_text": "Clear."}},
            step_post,
        ],
    }


async def _mk(spec: dict) -> tuple[str, str]:
    uid = str(uuid.uuid4())
    aid = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(id=uid, email=f"{uid[:8]}@example.com",
                    hashed_password="x", name="R37"))
        db.add(Automation(
            id=aid, user_id=uid, name=spec["name"], status="draft",
            spec_json=json.dumps(spec), trigger_mode="schedule",
        ))
        await db.commit()
    return uid, aid


# ── 1. the intent gate ───────────────────────────────────────────────

def test_thread_channel_forces_full_intent():
    from app.agent.agent_runner import intent_for_channel
    from app.agent.query_intent import classify_query_intent

    base = classify_query_intent("run it now")
    forced = intent_for_channel(base, "automation_thread")
    assert forced.category == "full"
    assert forced.include_skill_prompts is True
    # Every other channel keeps whatever the classifier said.
    assert intent_for_channel(base, "app") is base
    assert intent_for_channel(base, None) is base


# ── 2. the connector scope ───────────────────────────────────────────

def test_scope_keeps_members_and_never_touches_skill_tools():
    from app.agent.agent_runner import scope_connector_tools

    tools = [
        {"name": "web_search"},
        {"name": "automations__run_now"},   # skill tool with a dunder
        {"name": "slack__list_channels"},
        {"name": "jira__search_issues"},
    ]
    mcp = frozenset({"slack__list_channels", "jira__search_issues"})
    out = scope_connector_tools(tools, ["slack", "gmail"], mcp)
    names = {t["name"] for t in out}
    assert "slack__list_channels" in names
    assert "jira__search_issues" not in names
    # A skill tool is identified by the MCP name set, never by its
    # dunder — `automations__` would otherwise be scoped out for every
    # thread whose members don't include a connector called
    # "automations", which is all of them.
    assert "automations__run_now" in names
    assert "web_search" in names
    # None = unscoped (every caller but the thread).
    assert scope_connector_tools(tools, None, mcp) == tools


# ── 3. the grounding carries the setup ───────────────────────────────

def test_setup_lines_say_unpinned_draft_truthfully():
    from app.agent.automations.thread_agent import _setup_lines

    a = Automation(
        id="a1", user_id="u1", name="Inbox summary", status="draft",
        spec_json=json.dumps(_inbox_spec()), trigger_mode="schedule",
    )
    text = "\n".join(_setup_lines(a))
    assert "RIGHT NOW" in text
    assert "draft" in text
    assert "NOT pinned" in text
    # Once the write is gone, the same derivation says where it lands.
    a.spec_json = json.dumps({
        **_inbox_spec(), "steps": _inbox_spec()["steps"][:1],
    })
    text = "\n".join(_setup_lines(a))
    assert "in this thread" in text
    assert "NOT pinned" not in text


@pytest.mark.asyncio
async def test_thread_turn_passes_the_member_scope(monkeypatch):
    from app.agent.automations import ledger, thread_agent

    uid, aid = await _mk(_inbox_spec())
    seen = {}

    class _FakeRunner:
        async def run(self, **kw):
            seen.update(kw)

            class _R:
                text = "ok"
            return _R()

    monkeypatch.setattr(thread_agent, "_runner", lambda: _FakeRunner())
    async with async_session_maker() as db:
        automation = await db.get(Automation, aid)
        thread = await ledger.ensure_thread(db, user_id=uid,
                                            automation_id=aid)
        await thread_agent.answer_in_thread(
            db, automation=automation, thread=thread, user_text="run it",
        )
    assert sorted(seen["connector_scope"]) == ["gmail", "slack"]
    assert seen["channel"] == "automation_thread"
    # The setup rides the composed message, so the model answers from
    # the automation as it is, not as the thread remembers it.
    assert "RIGHT NOW" in seen["user_message"]


# ── 4a. "keep it in this chat" is a real change ──────────────────────

@pytest.mark.asyncio
async def test_set_destination_chat_strips_the_write_and_arms(monkeypatch):
    from app.agent.automations import service

    async def _registry(user_id, *, force=False):
        return REGISTRY

    async def _connections(user_id):
        return {"gmail": {"connected": True, "status": "ok"},
                "slack": {"connected": True, "status": "ok"}}

    monkeypatch.setattr(
        "app.agent.automations.registry.fetch_registry", _registry)
    monkeypatch.setattr(
        "app.agent.automations.registry.fetch_connection_state",
        _connections)

    uid, aid = await _mk(_inbox_spec())
    async with async_session_maker() as db:
        out = await service.set_destination_chat(
            db, automation_id=aid, user_id=uid,
        )
    assert out["changed"] is True
    assert out["armed"] is True

    async with async_session_maker() as db:
        a = await db.get(Automation, aid)
        raw = json.loads(a.spec_json)
        assert a.status == "armed"
        assert [s["id"] for s in raw["steps"]] == ["mail"]

    # And the thread records the change — the canvas/summary broadcast
    # rides _edited_note, so the EDITED stamp is the proof it ran.
    from app.agent.automations import ledger
    async with async_session_maker() as db:
        thread = await ledger.ensure_thread(db, user_id=uid,
                                            automation_id=aid)
        turns, _more = await ledger.list_turns(db, thread_id=thread.id,
                                              limit=20)
    stamps = [t.get("stamp") for t in turns if t.get("kind") == "note"]
    assert "edited" in stamps


@pytest.mark.asyncio
async def test_set_destination_chat_is_honest_when_nothing_changes(monkeypatch):
    from app.agent.automations import service

    async def _registry(user_id, *, force=False):
        return REGISTRY

    monkeypatch.setattr(
        "app.agent.automations.registry.fetch_registry", _registry)

    spec = _inbox_spec()
    spec["steps"] = spec["steps"][:1]  # already reads-only
    uid, aid = await _mk(spec)
    async with async_session_maker() as db:
        out = await service.set_destination_chat(
            db, automation_id=aid, user_id=uid,
        )
    assert out["changed"] is False


# ── 4b. pinning stamps the grant and stays draft ─────────────────────

@pytest.mark.asyncio
async def test_pin_destination_stamps_grant_and_target(monkeypatch):
    from app.agent.automations import service

    async def _registry(user_id, *, force=False):
        return REGISTRY

    monkeypatch.setattr(
        "app.agent.automations.registry.fetch_registry", _registry)

    uid, aid = await _mk(_inbox_spec())
    async with async_session_maker() as db:
        out = await service.pin_destination(
            db, automation_id=aid, user_id=uid, connector_id="slack",
            tool="slack__send_message", grant={"id": "g-77"},
            target={"kind": "channel", "id": "C9", "label": "#social"},
        )
    assert out == {"changed": True, "armed": False}
    async with async_session_maker() as db:
        a = await db.get(Automation, aid)
        raw = json.loads(a.spec_json)
        post = next(s for s in raw["steps"] if s["id"] == "post")
        assert post["grant_id"] == "g-77"
        assert post["grant_target"]["label"] == "#social"
        assert a.status == "draft"


@pytest.mark.asyncio
async def test_pin_destination_touches_one_step_only(monkeypatch):
    """Two slack posts, one already pinned+approved: pinning the second
    must not silently redirect the first (the round's review catch)."""
    from app.agent.automations import service

    async def _registry(user_id, *, force=False):
        return REGISTRY

    monkeypatch.setattr(
        "app.agent.automations.registry.fetch_registry", _registry)

    spec = _inbox_spec()
    pinned = {
        "id": "post2", "connector_id": "slack",
        "tool": "slack__send_message",
        "params": {"channel": "{{grant.target.id}}", "text": "alerts"},
        "grant_id": "g-old",
        "grant_target": {"kind": "channel", "id": "C-old",
                         "label": "#alerts"},
    }
    spec["steps"] = [spec["steps"][0], pinned, spec["steps"][1]]
    uid, aid = await _mk(spec)
    async with async_session_maker() as db:
        await service.pin_destination(
            db, automation_id=aid, user_id=uid, connector_id="slack",
            tool="slack__send_message", grant={"id": "g-new"},
            target={"kind": "channel", "id": "C-new", "label": "#general"},
        )
    async with async_session_maker() as db:
        a = await db.get(Automation, aid)
        raw = json.loads(a.spec_json)
        by_id = {st["id"]: st for st in raw["steps"]}
        assert by_id["post2"]["grant_id"] == "g-old"
        assert by_id["post2"]["grant_target"]["label"] == "#alerts"
        assert by_id["post"]["grant_id"] == "g-new"
        assert by_id["post"]["grant_target"]["label"] == "#general"


# ── 4c. Try again is idempotent server-side ──────────────────────────

@pytest.mark.asyncio
async def test_replay_reschedules_only_an_unanswered_turn(monkeypatch):
    from app.api import automations as api
    from app.agent.automations import ledger

    uid, aid = await _mk(_inbox_spec())
    monkeypatch.setattr(settings, "automations_enabled", True)
    monkeypatch.setattr(settings, "user_id", uid)

    scheduled = []
    monkeypatch.setattr(
        api, "_schedule_thread_answer",
        lambda *a, **k: scheduled.append(a),
    )

    body = api.ThreadMessageBody(text="Run it now",
                                 client_msg_id="cmid-r37")
    first = await api.post_thread_message(aid, body)
    assert not first.get("replayed")
    assert len(scheduled) == 1

    # Same client_msg_id, no answer yet → one more attempt, no new turn.
    second = await api.post_thread_message(aid, body)
    assert second.get("replayed") is True
    assert len(scheduled) == 2

    # First answer still RUNNING → the replay schedules nothing. Two
    # concurrent agent turns for one question is two reply bubbles and
    # doubled tool side effects; only a LOST answer earns a retry, and
    # lost is what an empty in-process set after a restart means.
    api._ANSWERING_THREADS.add(first["thread_id"])
    try:
        blocked = await api.post_thread_message(aid, body)
        assert blocked.get("replayed") is True
        assert len(scheduled) == 2
    finally:
        api._ANSWERING_THREADS.discard(first["thread_id"])

    # Answered → the replay schedules nothing (a re-tap after the reply
    # landed must not run the agent again).
    async with async_session_maker() as db:
        thread = await ledger.ensure_thread(db, user_id=uid,
                                            automation_id=aid)
        await ledger.append_turn(
            db, user_id=uid, thread=thread, run_id=None, kind="agent",
            payload={"text": "Started."},
        )
    third = await api.post_thread_message(aid, body)
    assert third.get("replayed") is True
    assert len(scheduled) == 2


# ── 1b. the founder's dead Run button, reproduced ────────────────────

@pytest.mark.asyncio
async def test_run_now_on_an_unanswered_variable_draft_is_a_409_not_a_500(monkeypatch):
    """Live repro, 2026-08-29 02:52 on the R33 sim: every run-now POST
    for a fresh from-template draft answered 500. `from_template`
    persists only the ANSWERED variable values, so the draft carries a
    dangling {{var.boss_email}}; `parse_spec_live` re-validated it with
    template_mode=True and NO template_vars, raised SpecError inside the
    route, and FastAPI called that a server error. The app's non-409
    path then asked the summary what happened, the summary said nothing,
    and the alert read "Nothing ran" — the founder's item 1, end to end.
    Two fixes, both pinned here: the re-parse waives unknown variables
    exactly as it waives grants (a setup question, not an authoring
    error), and run_now converts any residual SpecError into the honest
    409. This draft must land on needs_setup naming the pin."""
    from fastapi import HTTPException
    from app.agent.automations.service import create_automation
    from app.api import automations as api

    async def _registry(user_id, *, force=False):
        return REGISTRY

    monkeypatch.setattr(
        "app.agent.automations.registry.fetch_registry", _registry)

    spec = {
        "version": 2,
        "name": "Boss email -> draft reply",
        "mode": "confirm",
        "trigger": {"sources": [
            {"id": "mail", "mode": "push", "connector_id": "gmail",
             "event": "message_received",
             "filter": {"id": ["{{var.boss_email}}"]},
             "dedupe_key": "event.id"},
        ]},
        "steps": [
            {"id": "draft", "connector_id": "gmail",
             "tool": "gmail__create_draft",
             "params": {"to": "{{grant.target.id}}",
                        "subject": "Re: x", "body": "y"}},
        ],
        # The endpoint's exact stamp: only ANSWERED values. boss_email
        # was never answered, so it is a dangling reference on purpose.
        "variables": {},
    }
    uid = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(id=uid, email=f"{uid[:8]}@example.com",
                    hashed_password="x", name="R37b"))
        await db.commit()
    async with async_session_maker() as db:
        automation, _ = await create_automation(
            db, user_id=uid, spec=spec, template_slug="boss-email-draft",
            template_mode=True, template_vars={"boss_email", "draft_style"},
        )
    monkeypatch.setattr(settings, "automations_enabled", True)
    monkeypatch.setattr(settings, "user_id", uid)
    with pytest.raises(HTTPException) as exc:
        await api.run_now(automation.id)
    assert exc.value.status_code == 409, exc.value.detail
    assert exc.value.detail["code"] == "needs_setup"
    assert "500" not in str(exc.value.detail)


# ── 9. the menu's noun phrase ────────────────────────────────────────

def test_when_label_none_is_soon_and_summary_never_serves_it():
    """`_when_label(None)` stays "soon" (the meta line wants it); the
    schedule block's `next_run_label` must never be — the app composes
    "Without waiting for {label}" over it."""
    from zoneinfo import ZoneInfo
    from app.agent.automations.summary import _when_label

    assert _when_label(None, ZoneInfo("UTC")) == "soon"


@pytest.mark.asyncio
async def test_summary_next_run_label_is_a_noun_phrase(monkeypatch):
    from app.agent.automations.summary import summary_payload

    uid, aid = await _mk(_inbox_spec())
    async with async_session_maker() as db:
        items = (await summary_payload(db, user_id=uid))["automations"]
    mine = next(i for i in items if i["id"] == aid)
    assert mine["schedule"]["next_run_label"] == "its next run"
