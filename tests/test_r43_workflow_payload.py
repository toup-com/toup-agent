# agent-mode: automations/automation_threads/_turns are AGENT_ONLY — the
# write tests persist real specs through `service.update_automation` and
# count the EDITED note the writers stamp. The catalogue and scope-line
# tests are pure and ride along rather than earning the file a second entry.
"""R43 §2 / §3 — the workflow payload's delivery half, and its writers.

Three things the round adds to one GET and four PUT/POSTs:

  §2.1 DELIVERY   — where the brief reaches the user. All nine channels
                    always, each carrying whether it can be picked RIGHT
                    NOW and, when it cannot, why in words. The canvas
                    node title, the rail and the sheet's footer are one
                    pair of values rendered three ways, composed here so
                    they cannot disagree.
  §2.2 SOURCES    — what the agent may open inside each account, the
                    account's real objects behind it, and the chip's
                    scope line. The enumeration is a LIVE provider read
                    on the path of every canvas open, so its failure has
                    to cost that one list and nothing else.
  §3   THE WRITES — a picker that writes nowhere is this round's
                    forbidden shape, so `set_delivery` refuses an
                    unavailable channel outright rather than storing a
                    delivery that silently never happens.
"""

import json
import uuid

import pytest
from sqlalchemy import select

from app.agent.automations import catalog, compiler, workflow as wf
from app.agent.automations.spec import validate_spec
from app.db.database import async_session_maker
from app.db.models import Automation, User
from app.services.connector_registry import ConnectorRegistry


def _real_registry() -> dict:
    reg = ConnectorRegistry()
    reg.load_all()
    return {e["connector_id"]: e for e in reg.automation_registry()}


REGISTRY = _real_registry()

_ALL_CONNECTED = {
    cid: {"connector_id": cid, "connected": True, "status": "active",
          "scopes": ["r"], "account": f"{cid} acct"}
    for cid in ("gmail", "outlook", "slack", "teams", "jira", "github",
                "notion", "calendar")
}


@pytest.fixture(autouse=True)
def _offline_platform(monkeypatch):
    """No platform, no provider. Everything the payload reads live is a
    fixture here, because the point of these tests is what the payload
    does with the answers — including the answers that never come."""
    state = {"connections": dict(_ALL_CONNECTED)}

    async def _conn_state(user_id):
        return state["connections"]

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
    wf.invalidate_sources_cache()
    yield state
    wf.invalidate_sources_cache()


@pytest.fixture
def unlinked(monkeypatch):
    """Neither link-only channel linked — the shipped default."""
    from app.config import settings
    monkeypatch.setattr(settings, "whatsapp_session_status", "not_linked",
                        raising=False)
    monkeypatch.setattr(settings, "telegram_bot_token", "", raising=False)


def _read_step(sid, connector_id, tool, params):
    return {
        "id": sid, "connector_id": connector_id, "tool": tool,
        "params": params,
        "collect": {"items_path": "items", "fields": {"t": "title"},
                    "format": "{{item.t}}", "empty_text": "none"},
        "on_error": "skip",
    }


def _spec(**over):
    spec = {
        "version": 2, "name": "Morning brief", "mode": "auto",
        "trigger": {"sources": [
            {"id": "sched", "mode": "schedule",
             "schedule": {"cron_local": "0 8 * * 1-5"}},
        ]},
        "steps": [
            _read_step("mail", "gmail", "gmail__list_messages",
                       {"query": "in:inbox", "max_results": 10}),
            _read_step("tickets", "jira", "jira__search_issues",
                       {"jql": "assignee = currentUser()"}),
        ],
    }
    spec.update(over)
    return validate_spec(spec, REGISTRY)


async def _mk_user() -> str:
    uid = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(id=uid, email=f"{uid[:8]}@example.com",
                    hashed_password="x", name="R43"))
        await db.commit()
    return uid


async def _mk_automation(uid, vspec) -> Automation:
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


async def _payload(automation, uid) -> dict:
    async with async_session_maker() as db:
        a = await db.get(Automation, automation.id)
        return await wf.workflow_payload(db, automation=a, user_id=uid)


def _by_id(rows):
    return {r["id"]: r for r in rows}


# ── §2.1 the delivery block ─────────────────────────────────────────

def test_the_block_is_the_default_when_the_spec_has_never_said(unlinked):
    block = wf.delivery_block(None, {}, _ALL_CONNECTED)
    assert block["channels"] == ["app"]
    assert block["format"] == "ranked" and block["cadence"] == "run"
    # An automation written before R43 reads as the DEFAULT, never as
    # "nowhere" — the node has to name a real place on first open.
    assert block["node"]["label"] == "This app"
    assert block["node"]["sub"] == "as a ranked list"
    assert block["rail"] == "1 list"
    assert block["done_label"] == "Done — This app · a ranked list"


def test_one_two_and_five_channels_each_render_their_own_node(unlinked):
    def block(channels, fmt="ranked"):
        return wf.delivery_block(
            None, {"delivery": {"channels": channels, "format": fmt}},
            _ALL_CONNECTED)

    one = block(["slack_dm"])
    assert one["node"]["label"] == "Slack DM"
    assert one["node"]["chips"] == ["slack_dm"]
    assert one["done_label"] == "Done — Slack DM · a ranked list"

    two = block(["slack_dm", "app"], "pdf")
    # Catalogue order, never the caller's: two deliveries that reach the
    # same person the same way must serialize identically.
    assert two["channels"] == ["app", "slack_dm"]
    assert two["node"]["label"] == "This app +1"
    assert two["node"]["sub"] == "as a one-page PDF"
    assert two["rail"] == "1 PDF"
    assert two["done_label"] == "Done — This app + Slack DM · a one-page PDF"

    five = block(["app", "slack_dm", "teams_chat", "gmail_draft",
                  "notion_page"])
    assert five["node"]["label"] == "This app +4"
    # The canvas chip row is four wide, so the fifth is dropped from the
    # CHIPS and from nothing else — the footer still names all five.
    assert five["node"]["chips"] == ["app", "slack_dm", "teams_chat",
                                     "gmail_draft"]
    assert five["done_label"].count(" + ") == 4


def test_every_channel_is_always_served_even_the_ones_you_cannot_pick():
    block = wf.delivery_block(None, {}, {})
    rows = block["channels_available"]
    assert [r["id"] for r in rows] == list(catalog.CHANNEL_IDS)
    assert len(block["formats"]) == 5
    # A row the account cannot use is drawn with its reason, not dropped:
    # nine rows minus the unusable ones reads as a smaller product.
    assert all(r["reason"] for r in rows if not r["available"])


def test_availability_names_the_reason_for_every_case(unlinked):
    rows = _by_id(wf.delivery_block(None, {}, {
        "slack": {"connector_id": "slack", "connected": True,
                  "status": "active"},
        "gmail": {"connector_id": "gmail", "connected": True,
                  "status": "reauth_required"},
    })["channels_available"])

    # needs no account and no link
    assert rows["app"]["available"] and rows["app"]["reason"] is None
    assert rows["app"]["linked"] is True and rows["app"]["connector_id"] is None
    # connected account
    assert rows["slack_dm"]["available"] and rows["slack_dm"]["reason"] is None
    # expired account: named, and told what to do
    assert rows["gmail_draft"]["available"] is False
    assert rows["gmail_draft"]["reason"] == "Gmail needs signing in again."
    # missing account
    assert rows["teams_chat"]["available"] is False
    assert rows["teams_chat"]["reason"] == "Teams is not connected."
    # R43 repair (finding 16): a channel this platform cannot PROVE
    # reaches the user alone is refused for good, so it says the
    # permanent reason and not "Notion is not connected" — connecting
    # Notion would not light it, and sending the user to do that is the
    # picker that writes nowhere with an extra step in front of it.
    assert rows["notion_page"]["available"] is False
    assert rows["notion_page"]["reason"].startswith(
        "Notion cannot say who else can read a page")
    # unlinked channel: the reason quotes the catalogue's own ask
    assert rows["whatsapp"]["available"] is False
    assert rows["whatsapp"]["linked"] is False
    assert rows["whatsapp"]["needs_link"] == "your number"
    assert "your number" in rows["whatsapp"]["reason"]


def test_a_linked_channel_becomes_pickable(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "whatsapp_session_status", "linked",
                        raising=False)
    monkeypatch.setattr(settings, "telegram_bot_token", "  ", raising=False)
    rows = _by_id(
        wf.delivery_block(None, {}, {})["channels_available"])
    assert rows["whatsapp"]["available"] and rows["whatsapp"]["linked"]
    assert rows["whatsapp"]["reason"] is None
    # A whitespace-only token is not a bot. Failing to FALSE is the
    # whole contract of this read: an unlinked channel offered as
    # available is the picker that writes nowhere.
    assert rows["telegram"]["linked"] is False


# ── §2.2 the scope line ─────────────────────────────────────────────

def test_the_scope_line_has_four_branches_and_expiry_wins():
    avail = [{"id": "inbox", "short": "Inbox", "name": "Inbox"},
             {"id": "label:Clients", "short": "Clients", "name": "Clients"}]
    assert wf.scope_line("connected", [], avail) == "nothing picked yet"
    assert wf.scope_line("connected", ["inbox"], avail) == "Inbox"
    assert wf.scope_line(
        "connected", ["inbox", "label:Clients"], avail) == "Inbox +1"
    # Expired wins over BOTH — a list of places under an account that
    # cannot be read at all describes an intention, not a scope.
    assert wf.scope_line("expired", [], avail) == "access expired"
    assert wf.scope_line("expired", ["inbox"], avail) == "access expired"


def test_a_pick_the_enumeration_no_longer_lists_still_names_itself():
    # A renamed Slack channel must not blank the chip.
    assert wf.scope_line("connected", ["#gone"], []) == "#gone"


# ── §7 the instant triggers ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_triggers_carry_their_mode_and_the_two_defaults():
    uid = await _mk_user()
    a = await _mk_automation(uid, _spec())
    payload = await _payload(a, uid)
    accounts = {e["account_id"]: e for e in payload["accounts"]}

    gmail = _by_id(accounts["gmail"]["triggers_available"])
    assert gmail, "gmail declares events in the shipped manifests"
    # Gmail is the one connector with a subscription pipeline, so its
    # events are the only ones that may say "the moment it happens".
    assert {r["mode"] for r in gmail.values()} == {"push"}
    assert all(r["default"] is False for r in gmail.values())

    jira = _by_id(accounts["jira"]["triggers_available"])
    assert {r["mode"] for r in jira.values()} == {"poll"}
    assert "issue_assigned" in jira, "§7 names it as jira's default"
    # §7's two defaults are reported, never written: seeding them into an
    # automation that already exists would switch on an interruption its
    # owner did not ask for.
    assert [r["id"] for r in jira.values() if r["default"]] == [
        "issue_assigned"]
    assert wf.DEFAULT_TRIGGERS["github"] == "build_red"


def test_event_mode_is_gated_by_the_connector_not_only_the_event():
    # `spec_v2._validate_source` refuses `mode: "push"` on a connector
    # with no push path, so an event claiming one there would compile to
    # nothing at all — and the row would promise an interruption that
    # never arrives.
    assert wf.event_mode({"push": False}, {"push": True}) == "poll"
    assert wf.event_mode({"push": True}, {}) == "push"
    assert wf.event_mode({"push": True}, {"push": False}) == "poll"


# ── §2.2 the live enumeration, and its failures ─────────────────────

@pytest.mark.asyncio
async def test_sources_available_is_served_empty_when_the_provider_dies(
        monkeypatch):
    from app.agent.automations import contents

    async def _boom(user_id, connector_id, focus=None):
        raise RuntimeError("provider is having a minute")

    monkeypatch.setattr(contents, "account_sources", _boom, raising=False)
    uid = await _mk_user()
    a = await _mk_automation(uid, _spec())
    payload = await _payload(a, uid)

    for entry in payload["accounts"]:
        # Absent ≠ empty: the KEY is always there, so the app can tell a
        # backend that never shipped this from an account with nothing.
        assert entry["sources_available"] == []
        assert entry["sources"] == []
        assert entry["scope"] == "nothing picked yet"
    # …and the rest of the payload is intact. One slow or sick provider
    # costing the whole screen is R40's 22.7 s answer again.
    assert payload["delivery"]["channels"] == ["app"]
    assert payload["counts"]["accounts"] == 2
    assert [e["account_id"] for e in payload["accounts"]] == ["gmail", "jira"]


@pytest.mark.asyncio
async def test_a_real_enumeration_reaches_the_scope_line(monkeypatch):
    from app.agent.automations import contents
    calls = {"n": 0}

    async def _sources(user_id, connector_id, focus=None):
        calls["n"] += 1
        if connector_id != "gmail":
            return []
        return [
            {"id": "inbox", "name": "Inbox", "meta": "18 unread",
             "short": "Inbox", "kind": "label", "count": 18},
            {"id": "label:Clients", "name": "Clients", "meta": "3 threads",
             "short": "Clients", "kind": "label", "count": None},
        ]

    monkeypatch.setattr(contents, "account_sources", _sources, raising=False)
    uid = await _mk_user()
    a = await _mk_automation(uid, _spec(sources={"gmail": ["inbox",
                                                          "label:Clients"]}))
    payload = await _payload(a, uid)
    gmail = next(e for e in payload["accounts"] if e["account_id"] == "gmail")
    assert [s["id"] for s in gmail["sources_available"]] == [
        "inbox", "label:Clients"]
    assert gmail["sources_available"][1]["count"] is None
    assert gmail["sources"] == ["inbox", "label:Clients"]
    assert gmail["scope"] == "Inbox +1"

    before = calls["n"]
    await _payload(a, uid)
    # The enumeration is cached: `workflow_payload` is re-read after
    # every workflow write, and a provider call per write is a rate
    # budget spent on a list nobody asked to refresh.
    assert calls["n"] == before


@pytest.mark.asyncio
async def test_an_expired_account_is_never_enumerated(monkeypatch,
                                                      _offline_platform):
    from app.agent.automations import contents
    seen = []

    async def _sources(user_id, connector_id, focus=None):
        seen.append(connector_id)
        return []

    monkeypatch.setattr(contents, "account_sources", _sources, raising=False)
    _offline_platform["connections"] = {
        "gmail": {"connector_id": "gmail", "connected": True,
                  "status": "reauth_required"},
        "jira": {"connector_id": "jira", "connected": True,
                 "status": "active"},
    }
    uid = await _mk_user()
    a = await _mk_automation(uid, _spec())
    payload = await _payload(a, uid)
    assert seen == ["jira"]
    gmail = next(e for e in payload["accounts"] if e["account_id"] == "gmail")
    assert gmail["scope"] == "access expired"


# ── §2.3 the caption's numbers ──────────────────────────────────────

@pytest.mark.asyncio
async def test_counts_feed_the_panel_caption():
    uid = await _mk_user()
    a = await _mk_automation(uid, _spec())
    counts = (await _payload(a, uid))["counts"]
    assert counts["accounts"] == 2
    assert counts["instant"] == 0 and counts["pins"] == 0
    # `access` is a PHRASE because the caption prints it verbatim.
    assert counts["access"] == "read only"


@pytest.mark.asyncio
async def test_a_write_permission_turns_the_caption_to_read_and_write():
    uid = await _mk_user()
    post = {
        "id": "post", "connector_id": "slack", "tool": "slack__send_message",
        "params": {"channel": "{{grant.target.id}}", "text": "x"},
        "grant_id": "g-1",
        "grant_target": {"kind": "channel", "id": "C-PIN",
                         "label": "#platform"},
    }
    spec = _spec()
    raw = dict(spec.raw)
    raw["steps"] = list(raw["steps"]) + [post]
    a = await _mk_automation(uid, validate_spec(raw, REGISTRY))
    # A DRAFT is still "read only": `permissions._default_can_write`
    # refuses a grant nobody has decided yet, and the caption reads the
    # RESOLVED permission rather than the step table — the spec can
    # carry a write whose permission the user has not given.
    assert (await _payload(a, uid))["counts"]["access"] == "read only"
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        row.status = "armed"
        await db.commit()
    counts = (await _payload(a, uid))["counts"]
    assert counts["accounts"] == 3
    assert counts["access"] == "read and write"


# ── §3 the writers ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_set_delivery_is_partial_and_round_trips(unlinked):
    uid = await _mk_user()
    a = await _mk_automation(uid, _spec())
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        # `app` and not `slack_dm`: R43 repair (finding 14). A
        # connector-backed channel needs an APPROVED write grant on
        # THIS automation, pinned at the user — this spec has no write
        # step, so Slack DM is now correctly refused, and
        # `test_repair_*` in test_r43_repair_platform.py covers that.
        out = await wf.set_delivery(
            db, automation=row, user_id=uid,
            channels=["app"], format_id="pdf")
    assert out["delivery"]["channels"] == ["app"]
    assert out["delivery"]["format"] == "pdf"
    assert out["delivery"]["cadence"] == "run"
    assert "This app" in out["sentence"] and "one-page PDF" in out["sentence"]
    assert out["workflow"]["workflow_rev"] >= 1

    # A second write that names only the cadence must not rewrite the
    # channels the first one set — three independent controls, one sheet.
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        out2 = await wf.set_delivery(
            db, automation=row, user_id=uid, cadence="instant")
    assert out2["delivery"]["channels"] == ["app"]
    assert out2["delivery"]["format"] == "pdf"
    assert out2["delivery"]["cadence"] == "instant"


@pytest.mark.asyncio
async def test_set_delivery_refuses_an_unavailable_channel(
        unlinked, _offline_platform):
    _offline_platform["connections"] = {
        "gmail": {"connector_id": "gmail", "connected": True,
                  "status": "active"},
        "jira": {"connector_id": "jira", "connected": True,
                 "status": "active"},
    }
    uid = await _mk_user()
    a = await _mk_automation(uid, _spec())
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        with pytest.raises(wf.WorkflowError) as e:
            await wf.set_delivery(db, automation=row, user_id=uid,
                                  channels=["app", "slack_dm"])
    assert e.value.code == "channel_unavailable"
    assert e.value.sentence == "Slack is not connected."
    # …and nothing was stored: the canvas redraws from the payload, so a
    # refusal has to leave the payload showing what is really there.
    assert (await _payload(a, uid))["delivery"]["channels"] == ["app"]

    # An unlinked channel is refused the same way, by its own reason.
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        with pytest.raises(wf.WorkflowError) as e2:
            await wf.set_delivery(db, automation=row, user_id=uid,
                                  channels=["whatsapp"])
    assert e2.value.code == "channel_unavailable"
    assert "your number" in e2.value.sentence


@pytest.mark.asyncio
async def test_set_delivery_refuses_names_the_catalogue_does_not_hold():
    uid = await _mk_user()
    a = await _mk_automation(uid, _spec())
    for kwargs, code in ((dict(channels=["sms"]), "unknown_channel"),
                         (dict(format_id="haiku"), "unknown_format"),
                         (dict(cadence="whenever"), "unknown_cadence")):
        async with async_session_maker() as db:
            row = await db.get(Automation, a.id)
            with pytest.raises(wf.WorkflowError) as e:
                await wf.set_delivery(db, automation=row, user_id=uid,
                                      **kwargs)
        assert e.value.code == code


def test_an_unknown_name_is_a_400_and_an_unusable_one_a_409():
    # The app hard-codes the nine ids, so it can only send an unknown one
    # by being out of date; a name that exists but cannot be used is a
    # fact about the account, and the app draws that row with its reason
    # instead of retrying.
    from app.api.automations import _workflow_error
    assert _workflow_error(
        wf.WorkflowError("unknown_channel", "x")).status_code == 400
    assert _workflow_error(
        wf.WorkflowError("channel_unavailable", "x")).status_code == 409


@pytest.mark.asyncio
async def test_link_reports_where_it_got_to_and_never_selects(unlinked):
    uid = await _mk_user()
    a = await _mk_automation(uid, _spec())
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        out = await wf.link_channel(db, automation=row, user_id=uid,
                                    channel="telegram")
    assert out["linked"] is False
    assert out["url"] == wf.CHANNELS_DEEP_LINK
    assert "BotFather" in out["instructions"]
    # It never selects: a link that did not complete must not leave a
    # delivery pointing at a channel that cannot receive it.
    assert out["workflow"]["delivery"]["channels"] == ["app"]

    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        with pytest.raises(wf.WorkflowError) as e:
            await wf.link_channel(db, automation=row, user_id=uid,
                                  channel="app")
    assert e.value.code == "not_linkable"


@pytest.mark.asyncio
async def test_sources_round_trip_and_a_stale_pick_is_refused(monkeypatch):
    from app.agent.automations import contents

    async def _sources(user_id, connector_id, focus=None):
        if connector_id != "gmail":
            return []
        return [{"id": "inbox", "name": "Inbox", "meta": "18 unread",
                 "short": "Inbox", "kind": "label", "count": 18}]

    monkeypatch.setattr(contents, "account_sources", _sources, raising=False)
    uid = await _mk_user()
    a = await _mk_automation(uid, _spec())
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        out = await wf.set_sources(db, automation=row, user_id=uid,
                                   connector_id="gmail", sources=["inbox"])
    assert out["sources"] == ["inbox"]
    assert out["sentence"] == "In Gmail it now opens Inbox."
    gmail = next(e for e in out["workflow"]["accounts"]
                 if e["account_id"] == "gmail")
    assert gmail["scope"] == "Inbox"

    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        with pytest.raises(wf.WorkflowError) as e:
            await wf.set_sources(db, automation=row, user_id=uid,
                                 connector_id="gmail", sources=["label:gone"])
    assert e.value.code == "unknown_source"

    # Clearing it says what happens next, in words — and what happens
    # is the state every automation has always been in. R43 repair
    # (finding 2): "I will skip Gmail" promised a destructive behaviour
    # the engine has never had, about every account, on day one.
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        out2 = await wf.set_sources(db, automation=row, user_id=uid,
                                    connector_id="gmail", sources=[])
    assert out2["sources"] == []
    assert out2["sentence"] == ("Nothing picked — it reads all of Gmail, "
                                "as before.")


@pytest.mark.asyncio
async def test_a_pick_is_accepted_when_there_is_no_enumeration_to_check(
        monkeypatch):
    # The provider answered nothing (or `contents.account_sources` has
    # not shipped): validating against that empty list would refuse
    # every pick the sheet is currently showing — the user's screen and
    # the writer disagreeing about what exists.
    from app.agent.automations import contents

    async def _none(user_id, connector_id, focus=None):
        return []

    monkeypatch.setattr(contents, "account_sources", _none, raising=False)
    uid = await _mk_user()
    a = await _mk_automation(uid, _spec())
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        out = await wf.set_sources(db, automation=row, user_id=uid,
                                   connector_id="gmail",
                                   sources=["inbox", "label:Clients"])
    assert out["sources"] == ["inbox", "label:Clients"]


@pytest.mark.asyncio
async def test_the_ping_round_trips_and_null_clears_it(unlinked):
    uid = await _mk_user()
    a = await _mk_automation(uid, _spec())

    # No lane, no ping: a setting nothing can ever fire is refused
    # rather than stored.
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        with pytest.raises(wf.WorkflowError) as e:
            await wf.set_ping(db, automation=row, user_id=uid,
                              connector_id="jira", channel="app")
    assert e.value.code == "no_instant_lane"

    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        await wf.set_triggers(db, automation=row, user_id=uid,
                              connector_id="jira",
                              triggers=["issue_assigned"])
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        out = await wf.set_ping(db, automation=row, user_id=uid,
                                connector_id="jira", channel="app",
                                format_id="lines")
    assert out["ping"] == {"channel": "app", "format": "lines"}
    jira = next(e for e in out["workflow"]["accounts"]
                if e["account_id"] == "jira")
    assert jira["ping"] == {"channel": "app", "format": "lines"}
    assert out["sentence"] == ("Jira's alerts reach you in this chat, as "
                               "five short lines.")

    # An explicit null clears one half and leaves the other alone.
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        out2 = await wf.set_ping(db, automation=row, user_id=uid,
                                 connector_id="jira", format_id=None)
    assert out2["ping"] == {"channel": "app", "format": None}

    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        out3 = await wf.set_ping(db, automation=row, user_id=uid,
                                 connector_id="jira", channel=None)
    assert out3["ping"] == {"channel": None, "format": None}
    assert out3["sentence"] == "Jira's alerts follow the brief again."


@pytest.mark.asyncio
async def test_a_second_lane_on_the_same_account_inherits_the_ping(unlinked):
    uid = await _mk_user()
    a = await _mk_automation(uid, _spec())
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        keys = [t["id"] for t in wf.available_triggers(
            wf._spec_raw(row), REGISTRY, "jira")]
    assert len(keys) >= 2, "§7 gives jira four events"
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        await wf.set_triggers(db, automation=row, user_id=uid,
                              connector_id="jira", triggers=keys[:1])
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        await wf.set_ping(db, automation=row, user_id=uid,
                          connector_id="jira", channel="app")
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        await wf.set_triggers(db, automation=row, user_id=uid,
                              connector_id="jira", triggers=keys[:2])
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        raw = wf._spec_raw(row)
    # Every lane of one account answers the same way, or the account
    # sends two different kinds of ping with nothing on screen to say so.
    lanes = [s for s in raw["trigger"]["sources"]
             if s.get("connector_id") == "jira"]
    assert len(lanes) == 2
    assert all(s.get("ping_channel") == "app" for s in lanes)


@pytest.mark.asyncio
async def test_a_ping_channel_the_account_cannot_reach_is_refused(unlinked):
    uid = await _mk_user()
    a = await _mk_automation(uid, _spec())
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        await wf.set_triggers(db, automation=row, user_id=uid,
                              connector_id="jira",
                              triggers=["issue_assigned"])
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        with pytest.raises(wf.WorkflowError) as e:
            await wf.set_ping(db, automation=row, user_id=uid,
                              connector_id="jira", channel="whatsapp")
    assert e.value.code == "channel_unavailable"


# ── §7 the lanes actually compile ───────────────────────────────────

@pytest.mark.asyncio
async def test_every_declared_event_compiles_to_a_primitive_that_can_fire():
    """R43 raises the instant-event count from eight to twenty-seven.

    An event the user can turn ON that compiles to nothing — or to a
    Gmail Pub/Sub trigger row on a connector with no subscription
    pipeline — is the picker that writes nowhere in its worst form: it
    looks armed and simply never happens. `TRIGGER_KINDS` holds
    `email_received` and nothing else, so push is Gmail's alone and
    every other lane has to land on the poll routine.
    """
    from app.agent.automations.compiler import (
        ROUTINE_KIND_POLL, _push_pipeline,
    )
    from app.db.models import AutomationBinding, Routine, Trigger

    uid = await _mk_user()
    checked = 0
    for cid, entry in sorted(REGISTRY.items()):
        events = entry.get("events") or []
        if not events or cid == "stub":
            continue
        for ev in events:
            key = ev["key"]
            params = {p: "placeholder" for p in ev.get("params_required") or []}
            source = {
                "id": "lane", "mode": "push" if entry.get("push") else "poll",
                "connector_id": cid, "event": key,
                "dedupe_key": f"event.{ev.get('dedupe_field') or 'id'}",
                **({"params": params} if params else {}),
                "ping_channel": "app", "ping_format": "lines",
            }
            vspec = validate_spec(
                {"version": 2, "name": f"{cid} {key}", "mode": "auto",
                 "trigger": {"sources": [source]},
                 "steps": [_read_step("read", cid,
                                      entry["events"][0].get("source_tool")
                                      or f"{cid}__noop", {})]},
                REGISTRY, template_mode=True)
            a = await _mk_automation(uid, vspec)
            async with async_session_maker() as db:
                bindings = list((await db.execute(
                    select(AutomationBinding)
                    .where(AutomationBinding.automation_id == a.id)
                )).scalars())
                assert len(bindings) == 1, f"{cid}.{key} compiled nothing"
                b = bindings[0]
                if b.kind == "trigger":
                    row = await db.get(Trigger, b.target_id)
                    # Only the connector with a real subscription
                    # pipeline may claim one.
                    assert _push_pipeline(vspec.sources[0])
                    assert row is not None and row.kind == "email_received"
                else:
                    row = await db.get(Routine, b.target_id)
                    assert row is not None
                    assert row.kind == ROUTINE_KIND_POLL
                    # The floor, never NULL: a poll routine with no
                    # interval is a lane the scheduler cannot place.
                    assert (row.schedule_interval_seconds or 0) > 0
                # §8 rides on the primitive too, so a routine row answers
                # "where does this one land" without parsing a spec.
                assert row.config_json.get("ping_channel") == "app"
                assert row.config_json.get("ping_format") == "lines"
            checked += 1
    # §7 names 27 events across eight connectors and the manifests are
    # another package's this round, so the floor is what is declared
    # TODAY: the number may only grow, and every one of them compiles.
    assert checked >= 13, f"only {checked} events declared"
