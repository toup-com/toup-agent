# agent-mode: automations/automation_threads/_turns/_outbox/build_jobs are
# AGENT_ONLY — the brief and delivery tests drive the REAL executor and the
# REAL outbox against them (they borrow test_run_ledger_v3's fixtures, which
# are agent-mode for the same reason). The pure ones — the five formats, the
# filter applier, the addressing rule — are mode-agnostic and ride along.
"""R43 — the brief that opens, the filters that bite, delivery that arrives.

Three defects, one file, because they are three faces of the same thing:
what a run says it did versus what it did.

  1. The brief's five tiers each read "0 items" and opened nothing. The
     count was `rows.length` — ranked SENTENCES — and a group carried no
     items at all. Worse, every net that should have caught an empty
     brief was inert on exactly the run that produced one:
     `close_ledger`'s invariant asked whether the run produced ITEMS, and
     `_narrate_phase1` logged the narrator's rejections at INFO and
     persisted the drafts regardless.

  2. A filter chip was drawn from a table and composed by a hand-written
     per-connector ladder, so the two could disagree — and a chip that
     narrowed nothing looked identical to one that did.

  3. Delivery had no implementation at all: the automation's picked
     channels reached nothing, and the Slack post was a template
     interpolating one read while the card was the narrator's ranking.
"""

from __future__ import annotations

import base64
import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from app.agent.automations import brief_render as br
from app.agent.automations import catalog, deliver, executor_v2, ledger
from app.agent.automations import narrator, spec as spec_mod
from app.db.database import async_session_maker
from app.db.models import Automation, AutomationOutbox, AutomationTurn
from app.db.models.automation_ledger import RESULT_VOCABULARIES

from tests.test_run_ledger_v3 import (  # noqa: F401 — shared fixtures
    REGISTRY_V2, _fire, _mk_automation_v2, _mk_user, _one_run,
)
from app.agent.automations.spec import validate_spec


# ── fixtures ─────────────────────────────────────────────────────────


def _brief_spec(**over):
    """One read that collects items, no write — the shape the founder's
    morning brief actually has, and the one the empty-tier defect lived
    in."""
    spec = {
        "version": 2, "name": "Morning brief", "mode": "auto",
        "trigger": {"sources": [
            {"id": "sched", "mode": "schedule",
             "schedule": {"cron_local": "0 8 * * 1-5"}}]},
        "steps": [
            {"id": "issues", "connector_id": "jira",
             "tool": "jira__search_issues", "params": {"jql": "x"},
             "collect": {"items_path": "issues",
                         "fields": {"key": "key", "summary": "summary",
                                    "assignee": "assignee",
                                    "priority": "priority",
                                    "due": "duedate"},
                         "format": "{{item.key}} {{item.summary}}",
                         "empty_text": "none"},
             "on_error": "continue"},
        ],
    }
    spec.update(over)
    return validate_spec(spec, REGISTRY_V2)


_ISSUES = {"kind": "ok", "content": json.dumps({"issues": [
    {"key": "TP-482", "summary": "Rate-limit the export endpoint",
     "assignee": "Dana Cole", "priority": "Highest", "duedate": "2026-09-01"},
    {"key": "TP-476", "summary": "Flaky memverify test",
     "assignee": "Sam Ito", "priority": "Low", "duedate": ""},
]})}
_EMPTY = {"kind": "ok", "content": json.dumps({"issues": []})}


def _narrator_ranking_everything(monkeypatch):
    async def _narrate(record, *, complete=None):
        ids = [it["id"] for st in record["steps"] for it in st["items"]]
        groups = [{"rank": i + 1, "label": label, "tone": tone, "rows": []}
                  for i, (label, tone)
                  in enumerate(RESULT_VOCABULARIES["brief"])]
        groups[0]["rows"].append({"text": "Both of them need you",
                                  "sub": "One is late.",
                                  "tag": str(len(ids)), "item_refs": ids})
        return {"turns": [{"kind": "result", "title": "Your morning",
                           "vocabulary": "brief", "groups": groups}],
                "problems": [], "attempts": 1, "unservable": []}
    monkeypatch.setattr(
        "app.agent.automations.narrator.narrate_run", _narrate)


async def _result_of(a_id: str) -> dict:
    job = await _one_run(a_id)
    async with async_session_maker() as db:
        turns = await ledger.run_turns(db, run_id=job.id)
    results = [t for t in turns if t["kind"] == "result"]
    assert results, [t["kind"] for t in turns]
    return results[-1]


# ── 1. the brief that opens ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_ranked_tier_carries_the_items_its_rows_point_at(
        monkeypatch):
    """§9: `count` is `items.length`, and the row opens into what the
    run read. Before this, a group carried `rows` and nothing else — the
    item ids were on a DIFFERENT turn kind and no surface joined them."""
    _narrator_ranking_everything(monkeypatch)
    uid = await _mk_user()
    vspec = _brief_spec()
    a = await _mk_automation_v2(uid, vspec)
    assert await _fire(monkeypatch, uid, a, vspec,
                       responses={"jira__search_issues": _ISSUES}) == "run"

    result = await _result_of(a.id)
    tier1 = result["groups"][0]
    assert len(tier1["items"]) == 2, tier1
    assert tier1["empty_reason"] is None
    # The §9 slots, from the step's OWN collect fields — not re-parsed
    # out of the rendered line.
    by_title = {i["title"]: i for i in tier1["items"]}
    hot = by_title["Rate-limit the export endpoint"]
    assert hot["who"] == "Dana Cole"
    assert hot["source"] == "jira"
    assert hot["at"] == "2026-09-01"
    assert hot["hot"] is True, "Highest priority is the blue dot"
    assert by_title["Flaky memverify test"]["hot"] is False
    # And every id the row named is openable.
    assert {i["id"] for i in tier1["items"]} == set(
        tier1["rows"][0]["item_refs"])


@pytest.mark.asyncio
async def test_an_empty_tier_says_nothing_matched(monkeypatch):
    _narrator_ranking_everything(monkeypatch)
    uid = await _mk_user()
    vspec = _brief_spec()
    a = await _mk_automation_v2(uid, vspec)
    await _fire(monkeypatch, uid, a, vspec,
                responses={"jira__search_issues": _ISSUES})

    result = await _result_of(a.id)
    empties = [g for g in result["groups"] if not g["rows"]]
    assert empties, "the ranking put everything in tier 1"
    assert all(g["empty_reason"] == ledger.EMPTY_NOTHING_MATCHED
               for g in empties), empties
    assert all(g["items"] == [] for g in empties)


@pytest.mark.asyncio
async def test_an_empty_tier_names_the_account_that_could_not_be_read(
        monkeypatch):
    """§9: `empty_reason` names every account whose read failed, in the
    same sentence the reconnect card uses — "Nothing matched" over a
    Jira that never answered is a brief the user cannot trust.

    TWO reads, one of which answers: a run where EVERY source failed is
    `failed` and carries no brief at all (the executor refuses to
    assemble one from nothing), so it cannot exercise this."""
    _narrator_ranking_everything(monkeypatch)
    uid = await _mk_user()
    vspec = _brief_spec(steps=[
        {"id": "issues", "connector_id": "jira",
         "tool": "jira__search_issues", "params": {"jql": "x"},
         "collect": {"items_path": "issues", "fields": {"key": "key"},
                     "format": "{{item.key}}", "empty_text": "none"},
         "on_error": "continue"},
        {"id": "rooms", "connector_id": "slack",
         "tool": "slack__search_messages", "params": {"query": "to:me"},
         "collect": {"items_path": "matches", "fields": {"text": "text"},
                     "format": "{{item.text}}", "empty_text": "none"},
         "on_error": "continue"},
    ])
    a = await _mk_automation_v2(uid, vspec)
    await _fire(monkeypatch, uid, a, vspec, responses={
        "jira__search_issues": {"kind": "tool_error", "retryable": False,
                                "message": "it did not answer"},
        "slack__search_messages": {"kind": "ok", "content": json.dumps(
            {"matches": [{"text": "Anyone own the retry flag?"}]})},
    })

    result = await _result_of(a.id)
    reasons = {g["empty_reason"] for g in result["groups"]
               if g["empty_reason"]}
    assert reasons, result["groups"]
    assert all("Jira" in r for r in reasons), reasons
    assert ledger.EMPTY_NOTHING_MATCHED not in reasons


@pytest.mark.asyncio
async def test_a_run_that_read_nothing_does_not_publish_five_silent_tiers(
        monkeypatch):
    """The founder's card: five headings, "0 items" under each, chevrons
    that open nothing. `close_ledger`'s net asked whether the run
    produced ITEMS, so on the one run that produced none it was inert."""
    async def _narrate(record, *, complete=None):
        raise RuntimeError("model down")
    monkeypatch.setattr(
        "app.agent.automations.narrator.narrate_run", _narrate)

    uid = await _mk_user()
    vspec = _brief_spec()
    a = await _mk_automation_v2(uid, vspec)
    assert await _fire(monkeypatch, uid, a, vspec,
                       responses={"jira__search_issues": _EMPTY}) == "run"

    result = await _result_of(a.id)
    rows = [r for g in result["groups"] for r in g["rows"]]
    assert len(rows) == 1, rows
    assert "Nothing came back" in rows[0]["text"]
    # And it does not claim to have read things it did not.
    assert rows[0]["tag"] == "0"
    assert "could not rank" not in rows[0]["text"]


@pytest.mark.asyncio
async def test_a_result_whose_every_row_was_rejected_is_not_served(
        monkeypatch):
    """`_narrate_phase1` logged `problems` at INFO and persisted the
    drafts anyway. A result the validator rejected outright reached the
    thread as the user's brief."""
    async def _narrate(record, *, complete=None):
        ids = [it["id"] for st in record["steps"] for it in st["items"]]
        # Right tiers, but every row's text is empty — `_guard` rejects
        # each one, so nothing in this result survived validation.
        groups = [{"rank": i + 1, "label": label, "tone": tone, "rows": []}
                  for i, (label, tone)
                  in enumerate(RESULT_VOCABULARIES["brief"])]
        groups[0]["rows"].append({"text": "", "sub": "", "tag": "",
                                  "item_refs": ids})
        drafts = [{"kind": "agent", "text": "Morning."},
                  {"kind": "result", "title": "Your morning, in order",
                   "vocabulary": "brief", "groups": groups}]
        problems = narrator.validate_drafts(drafts, record)
        return {"turns": drafts, "problems": problems, "attempts": 2,
                "unservable": sorted(
                    narrator.unservable_results(drafts, problems))}
    monkeypatch.setattr(
        "app.agent.automations.narrator.narrate_run", _narrate)

    uid = await _mk_user()
    vspec = _brief_spec()
    a = await _mk_automation_v2(uid, vspec)
    await _fire(monkeypatch, uid, a, vspec,
                responses={"jira__search_issues": _ISSUES})

    result = await _result_of(a.id)
    rows = [r for g in result["groups"] for r in g["rows"]]
    assert all(r["text"] for r in rows), "an empty row reached the thread"
    assert any("could not rank" in r["text"] for r in rows), rows
    # The net accounts for every item even though the ranking is gone.
    assert sum(len(g["items"]) for g in result["groups"]) == 2


def test_a_partly_good_result_is_kept():
    """The gate must not throw a real ranking away over one bad tag: a
    result with SOME accepted rows is served, and the ledger's
    reconciliation picks up whatever it missed."""
    drafts = [
        {"kind": "agent", "text": "Morning."},
        {"kind": "result", "title": "t", "vocabulary": "brief",
         "groups": [{"rank": 1, "label": "L", "tone": "danger", "rows": [
             {"text": "good", "sub": "s", "item_refs": []},
             {"text": "bad", "sub": "s", "item_refs": []},
         ]}]},
    ]
    problems = ["turn[1].groups[0].rows[1].tag: not a sentence"]
    assert narrator.unservable_results(drafts, problems) == set()
    assert narrator.unservable_results(
        drafts, problems + ["turn[1].groups[0].rows[0].text: empty"]) == {1}
    # A structural violation is never servable, however good the rows.
    assert narrator.unservable_results(
        drafts, ["turn[1]: vocabulary must be 'brief'"]) == {1}


# ── 2. one composition, five formats ─────────────────────────────────


_GROUPS = [
    {"rank": 1, "label": "DO FIRST · BLOCKS OTHERS", "tone": "danger",
     "rows": [{"text": "Dana needs an owner for the retry flag",
               "sub": "It blocks the client fix going out tonight.",
               "tag": "P1 · due Thu", "item_refs": ["it_1"]}],
     "items": [{"id": "it_1", "who": "Dana Cole",
                "title": "Anyone own the retry flag?",
                "sub": "Asking before I ship the client fix tonight.",
                "why": "blocks the client fix", "at": "2026-08-31T22:40:00Z",
                "source": "slack", "where": "#platform", "hot": True}],
     "empty_reason": None},
    {"rank": 2, "label": "ANSWER TODAY", "tone": "warning",
     "rows": [], "items": [], "empty_reason": "Nothing matched"},
]


@pytest.mark.parametrize("fid", [f["id"] for f in catalog.FORMATS])
def test_every_catalogue_format_renders(fid):
    b = br.brief_render(_GROUPS, fid, title="Your morning, in order",
                        slug="Morning brief")
    assert b.format == fid
    assert b.text.strip(), fid
    # The empty tier is never a heading with nothing under it — that is
    # the "0 items" defect wearing a different surface.
    assert "ANSWER TODAY" not in b.text
    if fid in ("markdown", "csv", "pdf"):
        name, mime, blob = b.document
        assert name.startswith("morning-brief."), name
        assert blob, fid
    else:
        assert b.document is None


def test_lines_is_five_short_lines_and_ranked_counts_items():
    many = [{"rank": 1, "label": "L", "tone": "danger",
             "rows": [{"text": f"row {i}", "sub": "", "tag": "",
                       "item_refs": []} for i in range(9)],
             "items": [], "empty_reason": None}]
    assert len(br.brief_render(many, "lines").text.splitlines()) == br.MAX_LINES
    # `ranked`'s count is items, never rows — the whole §9 correction.
    ranked = br.brief_render(_GROUPS, "ranked", title="T").text
    assert "DO FIRST · BLOCKS OTHERS · 1" in ranked
    assert "Dana needs an owner for the retry flag · P1 · due Thu" in ranked


def test_the_pdf_is_a_real_pdf_and_never_markdown_under_that_name():
    b = br.brief_render(_GROUPS, "pdf", title="Your morning, in order")
    name, mime, blob = b.document
    assert mime == "application/pdf" and name.endswith(".pdf")
    assert blob.startswith(b"%PDF-"), blob[:16]


def test_the_csv_is_one_row_per_item():
    import csv as _csv
    import io as _io
    text = br.brief_render(_GROUPS, "csv").text
    rows = list(_csv.reader(_io.StringIO(text)))
    assert rows[0] == list(br.CSV_COLUMNS)
    assert len(rows) == 2, rows
    assert rows[1][2] == "Dana Cole" and rows[1][8] == "#platform"


def test_an_unknown_format_falls_back_rather_than_dropping_the_brief():
    b = br.brief_render(_GROUPS, "telepathy", title="T")
    assert b.format == catalog.DEFAULT_DELIVERY["format"]
    assert b.text.strip()


# ── 3. filters that bite ─────────────────────────────────────────────


_CLOCK = {"now": datetime(2026, 8, 31, 9, 0, tzinfo=timezone.utc)}

#: One READ call per connector the filter table covers, with the params
#: a shipped template would send.
_BASE_PARAMS = {
    "gmail__list_messages": {"query": "in:inbox", "max_results": 10},
    "gmail__search_threads": {"query": "in:inbox", "max_results": 10},
    "outlook__list_messages": {"max_results": 10},
    "slack__read_messages": {"channel": "C1", "limit": 20},
    "slack__search_messages": {"query": "to:me", "count": 15},
    "jira__search_issues": {"jql": "assignee = currentUser()",
                            "max_results": 15},
    "calendar__list_events": {"max_results": 10},
    # R43 wave 3 — the connectors whose new read tools made §6's
    # remaining chips expressible. Every one of these is a tool a
    # shipped step really calls, so a chip naming a tool with no row
    # here is a chip aimed at a read that does not exist.
    "teams__read_chat_messages": {"chat_id": "19:x", "max_results": 25},
    "github__search_issues": {"q": "is:open is:pr author:@me",
                              "per_page": 30},
    "notion__search": {"page_size": 25},
    "notion__query_database": {"database_id": "d1", "page_size": 25},
}


def test_every_offered_filter_changes_the_request():
    """§6: "Every filter must demonstrably change the request; a chip
    that compiles to nothing must not be offered." Proven against the
    SHIPPED table, so adding a chip with no compiler fails here rather
    than in a user's popup."""
    seen = 0
    for cid, filters in spec_mod.CONNECTOR_FILTERS.items():
        for f in filters:
            assert f.get("compile"), f"{cid}.{f['id']} compiles to nothing"
            for tool in f["tools"]:
                assert tool in _BASE_PARAMS, f"{tool} has no base params here"
                base = dict(_BASE_PARAMS[tool])
                out = executor_v2._apply_read_filters(
                    cid, tool, base, [f["id"]], _CLOCK)
                dropped = executor_v2._apply_read_drops(
                    cid, tool,
                    # One row carrying every field a `drop` chip reads:
                    # the assertion is that the chip removes it, so a
                    # row missing the field would read as a chip that
                    # narrows nothing.
                    {"messages": [{
                        "from": "noreply@x.com", "bot_id": "B1",
                        "author_type": "application", "state": "archived",
                        "my_response": "declined",
                        "last_edited_time": "2020-01-01T00:00:00Z",
                    }]},
                    [f["id"]], {"items_path": "messages"}, _CLOCK)
                changed = (out != base) or (dropped["messages"] == [])
                assert changed, f"{cid}.{f['id']} on {tool} narrowed nothing"
                seen += 1
    assert seen >= 11, seen


def test_gmail_terms_and_the_window_a_step_already_owns():
    out = executor_v2._apply_read_filters(
        "gmail", "gmail__list_messages", {"query": "in:inbox"},
        ["me", "unread", "no_promos", "day"], _CLOCK)
    for term in ("to:me", "is:unread", "-category:promotions",
                 "-category:updates", "newer_than:1d"):
        assert term in out["query"], out["query"]
    # The Morning brief's `waiting` step IS its own age window; ANDing
    # `newer_than:1d` onto `older_than:1d` makes a query that can never
    # match, which `empty_text` then states as a fact about the user.
    waiting = executor_v2._apply_read_filters(
        "gmail", "gmail__list_messages",
        {"query": "to:me is:unread older_than:1d newer_than:7d"},
        ["day"], _CLOCK)
    assert "newer_than:1d" not in waiting["query"]


def test_jql_clauses_and_the_sort_that_stays_trailing():
    out = executor_v2._apply_read_filters(
        "jira", "jira__search_issues",
        {"jql": "assignee = currentUser() ORDER BY duedate ASC"},
        ["priority", "open", "due_week", "day"], _CLOCK)
    jql = out["jql"]
    assert jql.endswith("ORDER BY duedate ASC"), jql
    for clause in ("priority in (Highest, High)", "statusCategory != Done",
                   "duedate <= endOfWeek()", "updated >= -1d"):
        assert clause in jql, jql
    # ONE pass: four filters must not nest four layers of parentheses.
    assert jql.count("((") == 0, jql


def test_outlook_sets_a_param_and_a_bound_that_only_narrows():
    out = executor_v2._apply_read_filters(
        "outlook", "outlook__list_messages",
        {"is_read": True, "since": "2026-08-25T00:00:00"},
        ["unread", "day"], _CLOCK)
    assert out["is_read"] is False, "the lit chip must win over the spec"
    assert out["since"] == "2026-08-30T09:00:00+00:00", out["since"]
    # A spec bound already TIGHTER than the filter is kept.
    kept = executor_v2._apply_read_filters(
        "outlook", "outlook__list_messages",
        {"since": "2026-08-31T08:00:00"}, ["day"], _CLOCK)
    assert kept["since"] == "2026-08-31T08:00:00"


def test_outlook_skip_automated_mail_drops_the_rows_no_query_can():
    content = {"messages": [
        {"from": "dana@acme.com", "subject": "retry flag"},
        {"from": "no-reply@notifications.example", "subject": "digest"},
        {"from": "MAILER-DAEMON@example.com", "subject": "undeliverable"},
    ]}
    out = executor_v2._apply_read_drops(
        "outlook", "outlook__list_messages", content, ["no_auto"],
        {"items_path": "messages"}, _CLOCK)
    assert [m["from"] for m in out["messages"]] == ["dana@acme.com"]
    # Total: a filter that is not on changes nothing at all.
    assert executor_v2._apply_read_drops(
        "outlook", "outlook__list_messages", content, [],
        {"items_path": "messages"}, _CLOCK) is content


def test_slack_one_chip_two_units_and_calendar_windows_both_ends():
    hist = executor_v2._apply_read_filters(
        "slack", "slack__read_messages", {"channel": "C1"}, ["day"], _CLOCK)
    assert hist["oldest"] == f"{(_CLOCK['now'] - timedelta(days=1)).timestamp():.0f}"
    search = executor_v2._apply_read_filters(
        "slack", "slack__search_messages", {"query": "to:me"}, ["day"],
        _CLOCK)
    assert "after:2026-08-29" in search["query"], search["query"]
    cal = executor_v2._apply_read_filters(
        "calendar", "calendar__list_events", {}, ["next24"], _CLOCK)
    assert cal["time_min"] == "2026-08-31T09:00:00+00:00"
    assert cal["time_max"] == "2026-09-01T09:00:00+00:00"


def test_the_applier_is_total():
    for args in (
        ("nosuch", "nosuch__tool", {"a": 1}, ["day"], _CLOCK),
        ("gmail", "gmail__list_messages", {"query": "x"}, ["nope"], _CLOCK),
        ("outlook", "outlook__list_messages", {}, ["day"], {}),
    ):
        out = executor_v2._apply_read_filters(*args)
        assert out == args[2], args


# ── 4. delivery that reaches you ─────────────────────────────────────


def test_the_ping_override_wins_only_for_its_own_connector_run():
    from app.agent.automations.spec_v2 import ValidatedSource

    default = {"channels": ["app"], "format": "ranked", "cadence": "run"}
    sched = ValidatedSource(
        id="sched", mode="schedule", connector_id=None, event=None,
        params={}, poll_interval_s=None, schedule={"cron_local": "0 8 * * *"},
        filter_rules={}, dedupe_key_field=None, event_spec=None,
    )
    ping = ValidatedSource(
        id="jira-p1", mode="poll", connector_id="jira", event="p1_raised",
        params={}, poll_interval_s=300, schedule=None, filter_rules={},
        dedupe_key_field=None, event_spec=None,
        ping_channel="slack_dm", ping_format="lines",
    )
    assert deliver.effective_delivery(default, sched) == default
    assert deliver.effective_delivery(default, None) == default
    hot = deliver.effective_delivery(default, ping)
    assert hot["channels"] == ["slack_dm"], "one channel, never a fan-out"
    assert hot["format"] == "lines"
    assert hot["cadence"] == "run"
    # An unknown id is ignored rather than silently emptying the plan.
    junk = ValidatedSource(
        id="x", mode="poll", connector_id="jira", event="p1_raised",
        params={}, poll_interval_s=300, schedule=None, filter_rules={},
        dedupe_key_field=None, event_spec=None,
        ping_channel="carrier_pigeon", ping_format="stone_tablet",
    )
    assert deliver.effective_delivery(default, junk) == default


@pytest.mark.parametrize("channel_id,target,account,ok", [
    ("gmail_draft", "me@acme.com", "me@acme.com", True),
    ("gmail_draft", "ME@ACME.COM", "me@acme.com", True),
    ("gmail_draft", "dana@acme.com", "me@acme.com", False),
    ("outlook_mail", "boss@acme.com", "me@acme.com", False),
    ("gmail_draft", "me@acme.com", "", False),
    ("slack_dm", "D0FOUNDER", "U0ME", True),
    ("slack_dm", "U0ME", "U0ME", True),
    ("slack_dm", "U0DANA", "U0ME", False),
    ("slack_dm", "C0ALL", "U0ME", False),
    ("slack_dm", "", "U0ME", False),
    # An owner nobody could resolve is a refusal, not a pass: a `U…`
    # that cannot be checked against an identity might be a colleague,
    # and posting to it opens a DM with THEM.
    ("slack_dm", "U0ME", "", False),
    # A `D…` needs no proof — it is a conversation out of this token's
    # own DM list, so it is the user's by construction.
    ("slack_dm", "D0FOUNDER", "", True),
])
def test_nothing_is_ever_addressed_to_anyone_but_the_user(
        channel_id, target, account, ok):
    """§1.3, and the one rule that cannot be left to the picker. A grant
    is the user approving a target, and `#platform` is a perfectly legal
    grant for a write STEP — the delivery node's promise is that this
    list is not that."""
    call = lambda: deliver._check_addressed_to_the_user(  # noqa: E731
        channel_id, deliver._CONNECTOR_CHANNELS[channel_id]["connector_id"],
        target, account)
    if ok:
        call()
    else:
        with pytest.raises(deliver.DeliveryRefused):
            call()


@pytest.mark.asyncio
@pytest.mark.parametrize("connector_id,column,live,expected", [
    # Gmail's column is real — `oauth._gmail_post_connect` backfills it —
    # so no provider call is made.
    ("gmail", "me@acme.com", None, "me@acme.com"),
    # Slack's and Outlook's are structurally empty: `oauth.py` writes
    # `provider_account_id` from `tokens["account"] or tokens["login"]`,
    # which is Microsoft's field and GitHub's. Slack puts the id under
    # `id`, and Microsoft's token endpoint returns neither — so the
    # column has been NULL for every one of those identities ever made.
    ("slack", "", "U0ME", "U0ME"),
    ("outlook", "", "me@acme.com", "me@acme.com"),
])
async def test_the_owner_check_reads_a_source_that_actually_has_the_answer(
        monkeypatch, connector_id, column, live, expected):
    """§1.3 needs to know WHO the connection is, and the stored column
    could not say for two of the three connectors that need it. The
    consequence was not cosmetic: `outlook_mail` was offered to every
    Outlook user and refused every one of them with `unknown_account`,
    and `slack_dm` could not tell the owner's `U…` from a colleague's.

    Resolved live instead, which is the better answer anyway — it reads
    the token that is about to be used rather than a column written once
    at connect time.
    """
    async def _conn(user_id):
        return {connector_id: {"connected": True, "account": column}}
    monkeypatch.setattr(
        "app.agent.automations.registry.fetch_connection_state", _conn)
    if connector_id == "slack":
        async def _ident(user_id):
            return {"user_id": live, "handle": "me"}
        monkeypatch.setattr(
            "app.connectors.slack.provider.self_identity_for_user", _ident)
    if connector_id == "outlook":
        async def _tok(user_id):
            return "tok"

        async def _mbox(user_id, token):
            return live
        monkeypatch.setattr(
            "app.connectors.outlook.provider._resolve_token", _tok)
        monkeypatch.setattr(
            "app.connectors.outlook.provider._mailbox_address", _mbox)
    assert await deliver._account_for("u", connector_id) == expected


@pytest.mark.asyncio
async def test_an_unprovable_owner_declines_it_never_guesses(monkeypatch):
    """The resolver may not raise into a delivery, and it may not pass
    either: an identity nobody can prove is a REFUSAL, in words."""
    async def _conn(user_id):
        return {"outlook": {"connected": True, "account": ""}}
    monkeypatch.setattr(
        "app.agent.automations.registry.fetch_connection_state", _conn)

    async def _boom(user_id):
        raise RuntimeError("token vault unreachable")
    monkeypatch.setattr(
        "app.connectors.outlook.provider._resolve_token", _boom)

    assert await deliver._account_for("u", "outlook") == ""
    with pytest.raises(deliver.DeliveryRefused) as e:
        deliver._check_addressed_to_the_user(
            "outlook_mail", "outlook", "me@acme.com", "")
    assert e.value.reason_code == "unknown_account"
    assert e.value.sentence == "It could not tell which mailbox is yours"


def _delivery_env(monkeypatch, *, grants: dict, accounts: dict, calls: list):
    async def _fetch_grant(user_id, grant_id):
        return grants.get(grant_id)

    async def _connections(user_id):
        return {cid: {"connected": True, "account": acc}
                for cid, acc in accounts.items()}

    async def _dispatch(user_id, *, connector_id, tool_name, tool_input,
                        grant_id=None, automation_id=None, request_id=None,
                        timeout_s=60.0):
        calls.append({"connector_id": connector_id, "tool": tool_name,
                      "input": tool_input, "grant_id": grant_id})
        return {"kind": "ok", "content": "{}"}

    monkeypatch.setattr(
        "app.agent.automations.registry.fetch_grant", _fetch_grant)
    monkeypatch.setattr(
        "app.agent.automations.registry.fetch_connection_state", _connections)
    monkeypatch.setattr(
        "app.agent.automations.registry.dispatch_via_platform", _dispatch)


async def _automation_with_write(uid, connector_id, tool, grant_id):
    vspec = validate_spec({
        "version": 2, "name": "Morning brief", "mode": "auto",
        "trigger": {"sources": [
            {"id": "sched", "mode": "schedule",
             "schedule": {"cron_local": "0 8 * * 1-5"}}]},
        "steps": [
            {"id": "issues", "connector_id": "jira",
             "tool": "jira__search_issues", "params": {"jql": "x"},
             "collect": {"items_path": "issues",
                         "fields": {"key": "key"},
                         "format": "{{item.key}}", "empty_text": "none"},
             "on_error": "continue"},
        ],
    }, REGISTRY_V2)
    a = await _mk_automation_v2(uid, vspec)
    # The delivery grant is not a spec STEP — it is the permission the
    # delivery picker stages. `permissions.write_grant_ids` reads the
    # spec's steps, so the grant rides there in this fixture the same
    # way a real write step's does.
    async with async_session_maker() as db:
        row = await db.get(Automation, a.id)
        raw = json.loads(row.spec_json)
        raw["steps"].append({
            "id": "deliver", "connector_id": connector_id, "tool": tool,
            "params": {}, "grant_id": grant_id,
            "grant_target": {"kind": "person", "id": "x"},
        })
        row.spec_json = json.dumps(raw, sort_keys=True)
        await db.commit()
    return a, vspec


_DELIVERY_CASES = [
    ("slack_dm", "slack", "slack__send_message",
     {"kind": "thread", "id": "D0FOUNDER", "label": "you"}, "U0ME",
     "channel", "text"),
    ("teams_chat", "teams", "teams__send_chat_message",
     {"kind": "thread", "id": "19:chat", "label": "your chat"}, "me@acme.com",
     "chat_id", "message"),
    ("gmail_draft", "gmail", "gmail__create_draft",
     {"kind": "person", "id": "me@acme.com", "label": "you"}, "me@acme.com",
     "to", "body"),
    ("outlook_mail", "outlook", "outlook__create_draft",
     {"kind": "person", "id": "me@acme.com", "label": "you"}, "me@acme.com",
     "to", "body"),
    # R43 — `notion__append_blocks`, not `notion__create_page`. §1.2
    # promises "appended under today's date"; a create makes a CHILD
    # page, so the pinned page never gains a line.
    ("notion_page", "notion", "notion__append_blocks",
     {"kind": "doc", "id": "page-1", "label": "Journal"}, "me@acme.com",
     "page_id", "content"),
    # R43 — the calendar manifest declares `calendar_id` as this write's
    # pinned target now, so a hold is grantable and this channel joins
    # the others instead of refusing every time it is picked.
    ("calendar_hold", "calendar", "calendar__create_event",
     {"kind": "folder", "id": "primary", "label": "Your calendar"},
     "me@acme.com", "calendar_id", "description"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "channel_id,connector_id,tool,target,account,param,body_param",
    _DELIVERY_CASES)
async def test_each_channel_stages_the_right_outbox_row(
        monkeypatch, channel_id, connector_id, tool, target, account,
        param, body_param):
    """Rule 2: a delivery is a write like any other — same outbox row,
    same grant, same undo/audit machinery."""
    uid = await _mk_user()
    a, vspec = await _automation_with_write(uid, connector_id, tool, "g-d")
    calls: list = []
    _delivery_env(
        monkeypatch,
        grants={"g-d": {"status": "approved", "tool_name": tool,
                        "target": target}},
        accounts={connector_id: account}, calls=calls)
    await _fire(monkeypatch, uid, a, vspec,
                responses={"jira__search_issues": _EMPTY})
    _delivery_env(
        monkeypatch,
        grants={"g-d": {"status": "approved", "tool_name": tool,
                        "target": target}},
        accounts={connector_id: account}, calls=calls)

    job = await _one_run(a.id)
    async with async_session_maker() as db:
        auto = await db.get(Automation, a.id)
        thread = await ledger.thread_for(db, a.id)
        result = [t for t in await ledger.run_turns(db, run_id=job.id)
                  if t["kind"] == "result"][-1]
        out = await deliver.deliver_brief(
            db, automation=auto, job_id=job.id, thread=thread,
            groups=result["groups"], title=result["title"],
            delivery={"channels": ["app", channel_id], "format": "ranked",
                      "cadence": "run"},
            idem_prefix=f"t:{uuid.uuid4()}",
        )
    assert out["app"]["status"] == "skipped", "the app brief is not re-posted"
    assert out[channel_id]["status"] == "delivered", out

    sent = [c for c in calls if c["tool"] == tool]
    assert len(sent) == 1, calls
    assert sent[0]["grant_id"] == "g-d"
    assert sent[0]["input"][param] == target["id"]
    assert sent[0]["input"][body_param].strip()

    async with async_session_maker() as db:
        rows = list((await db.execute(
            select(AutomationOutbox)
            .where(AutomationOutbox.job_id == job.id)
            .where(AutomationOutbox.tool_name == tool)
        )).scalars())
    assert len(rows) == 1 and rows[0].status == "executed"


@pytest.mark.asyncio
async def test_a_calendar_hold_has_no_attendees_and_names_its_calendar(
        monkeypatch):
    """A hold, not an invite (§1.2). `attendees` is empty and stays
    empty: the moment this list has a name in it the block becomes a
    meeting somebody else is now expected at.

    R43 — the manifest declares `calendar_id` as this write's pinned
    target, so the hold is grantable. The id is passed EXPLICITLY even
    though the tool schema defaults it: `connector_dispatcher` compares
    `tool_input[target_param]` against the grant's pinned id, and a
    schema default is not an argument.
    """
    now = datetime(2026, 8, 31, 9, 0, tzinfo=timezone.utc)
    b = br.brief_render(_GROUPS, "ranked", title="T")

    class _A:
        name = "Morning brief"
    payload = deliver._payload_for(
        "calendar_hold", target="primary", brief=b, automation=_A(), now=now)
    assert payload["attendees"] == []
    assert payload["end"] == "2026-08-31T09:15:00+00:00"
    assert payload["calendar_id"] == "primary"

    assert deliver._CONNECTOR_CHANNELS["calendar_hold"]["param"] == (
        "calendar_id")
    # And no channel may be left without one: a tool with no pinned
    # target cannot be covered by a grant, so the write dies at fire
    # time — after the run has told the user it delivered.
    for cid, spec in deliver._CONNECTOR_CHANNELS.items():
        assert spec["param"], cid


def test_a_file_format_reaches_the_mail_channels_as_a_file():
    """§1.3/§6 — "One-page PDF" naming a text blob is the same lie as a
    chip that narrows nothing. Both draft tools take the same
    `attachments` shape, so the payload builder does not branch on which
    mail channel it is writing to; the two chat formats carry no
    document and must not grow an empty attachment list."""
    now = datetime(2026, 8, 31, 9, 0, tzinfo=timezone.utc)

    class _A:
        name = "Morning brief"

    for channel_id in ("gmail_draft", "outlook_mail"):
        for fmt, suffix in (("pdf", ".pdf"), ("markdown", ".md"),
                            ("csv", ".csv")):
            b = br.brief_render(_GROUPS, fmt, title="T", slug="brief")
            assert b.document, fmt
            body = deliver._payload_for(
                channel_id, target="me@acme.com", brief=b, automation=_A(),
                now=now)
            att = body["attachments"]
            assert len(att) == 1
            assert att[0]["filename"].endswith(suffix)
            assert att[0]["content_type"] == b.mime
            # The bytes, not a re-render: decoding must give back
            # exactly what `brief_render` produced.
            assert base64.b64decode(att[0]["content_base64"]) == b.document[2]
        for fmt in ("lines", "ranked"):
            b = br.brief_render(_GROUPS, fmt, title="T")
            assert b.document is None, fmt
            body = deliver._payload_for(
                channel_id, target="me@acme.com", brief=b, automation=_A(),
                now=now)
            assert "attachments" not in body

    # §1.2's own words: "mail to yourself, marked read" — Outlook only;
    # Gmail's draft tool has no such parameter.
    b = br.brief_render(_GROUPS, "ranked", title="T")
    assert deliver._payload_for("outlook_mail", target="me@acme.com",
                                brief=b, automation=_A(),
                                now=now)["is_read"] is True
    assert "is_read" not in deliver._payload_for(
        "gmail_draft", target="me@acme.com", brief=b, automation=_A(),
        now=now)


@pytest.mark.asyncio
async def test_a_channel_with_no_approved_grant_fails_visibly(monkeypatch):
    """Rule 3. A brief that quietly did not arrive is worse than one
    that arrived late — the user has no way to find out."""
    uid = await _mk_user()
    a, vspec = await _automation_with_write(
        uid, "slack", "slack__send_message", "g-d")
    calls: list = []
    _delivery_env(monkeypatch,
                  grants={"g-d": {"status": "pending",
                                  "tool_name": "slack__send_message",
                                  "target": {"id": "D0FOUNDER"}}},
                  accounts={"slack": "U0ME"}, calls=calls)
    await _fire(monkeypatch, uid, a, vspec,
                responses={"jira__search_issues": _EMPTY})
    _delivery_env(monkeypatch,
                  grants={"g-d": {"status": "pending",
                                  "tool_name": "slack__send_message",
                                  "target": {"id": "D0FOUNDER"}}},
                  accounts={"slack": "U0ME"}, calls=calls)

    job = await _one_run(a.id)
    async with async_session_maker() as db:
        auto = await db.get(Automation, a.id)
        thread = await ledger.thread_for(db, a.id)
        result = [t for t in await ledger.run_turns(db, run_id=job.id)
                  if t["kind"] == "result"][-1]
        out = await deliver.deliver_brief(
            db, automation=auto, job_id=job.id, thread=thread,
            groups=result["groups"], title=result["title"],
            delivery={"channels": ["slack_dm"], "format": "ranked",
                      "cadence": "run"},
            idem_prefix=f"t:{uuid.uuid4()}")
    assert out["slack_dm"] == {"status": "failed", "reason": "grant_missing"}
    assert not calls, "nothing may be sent without an approved permission"

    async with async_session_maker() as db:
        turns = await ledger.run_turns(db, run_id=job.id)
        job2 = await _one_run(a.id)
    failed = [t for t in turns
              if t["kind"] == "tool" and t.get("tool_kind") == "write"
              and not t.get("ok", True)]
    assert failed, [t["kind"] for t in turns]
    assert failed[0]["reason_code"] == "grant_missing"
    assert failed[0].get("line")
    # And the run's own record answers "did it reach me".
    assert (job2.config_json or {})["delivery"]["results"]["slack_dm"][
        "status"] == "failed"


@pytest.mark.asyncio
async def test_the_agents_own_channels_go_through_the_existing_dispatcher(
        monkeypatch):
    """WhatsApp and Telegram are not connected third-party ACCOUNTS —
    the agent is messaging its owner from its own number. There is no
    grant to check and nothing to undo, so they must not acquire an
    outbox row, and `routines.channel_dispatcher` (which owns recipient
    resolution and per-adapter formatting) must not be rebuilt here."""
    seen: list = []

    async def _detailed(*, user_id, delivery_channels, routine_name,
                        content, db_session_maker):
        seen.append({"channels": list(delivery_channels),
                     "content": content, "name": routine_name})
        return {delivery_channels[0]: {"status": "delivered"}}

    monkeypatch.setattr(
        "app.agent.routines.channel_dispatcher"
        ".deliver_to_extra_channels_detailed", _detailed)

    uid = await _mk_user()
    vspec = _brief_spec()
    a = await _mk_automation_v2(uid, vspec)
    await _fire(monkeypatch, uid, a, vspec,
                responses={"jira__search_issues": _EMPTY})
    job = await _one_run(a.id)

    async with async_session_maker() as db:
        auto = await db.get(Automation, a.id)
        thread = await ledger.thread_for(db, a.id)
        out = await deliver.deliver_brief(
            db, automation=auto, job_id=job.id, thread=thread,
            groups=_GROUPS, title="Your morning, in order",
            delivery={"channels": ["telegram", "whatsapp"],
                      "format": "lines", "cadence": "run"},
            idem_prefix=f"t:{uuid.uuid4()}")

    assert out["telegram"]["status"] == "delivered"
    assert out["whatsapp"]["status"] == "delivered"
    # Catalogue order, never the caller's (§1.2): WhatsApp precedes
    # Telegram in the table, so it precedes it here whatever the picked
    # list happened to be.
    assert [s["channels"] for s in seen] == [["whatsapp"], ["telegram"]]
    # The SAME brief both times — one composition, §9's last rule.
    assert seen[0]["content"] == seen[1]["content"]
    assert "Dana needs an owner for the retry flag" in seen[0]["content"]

    async with async_session_maker() as db:
        rows = list((await db.execute(
            select(AutomationOutbox)
            .where(AutomationOutbox.job_id == job.id)
        )).scalars())
    assert rows == [], "an own-channel send is not an outbox write"


@pytest.mark.asyncio
async def test_a_second_delivery_of_the_same_run_is_a_no_op(monkeypatch):
    """The idempotency key is the CHANNEL, not an index: a resumed run
    re-delivers to the same places, and the outbox's unique index makes
    the second attempt a no-op rather than a second copy of the brief."""
    uid = await _mk_user()
    a, vspec = await _automation_with_write(
        uid, "gmail", "gmail__create_draft", "g-d")
    calls: list = []
    env = dict(
        grants={"g-d": {"status": "approved",
                        "tool_name": "gmail__create_draft",
                        "target": {"id": "me@acme.com", "label": "you"}}},
        accounts={"gmail": "me@acme.com"}, calls=calls)
    _delivery_env(monkeypatch, **env)
    await _fire(monkeypatch, uid, a, vspec,
                responses={"jira__search_issues": _EMPTY})
    _delivery_env(monkeypatch, **env)

    job = await _one_run(a.id)
    idem = f"t:{uuid.uuid4()}"
    for expected in ("delivered", "already_delivered"):
        async with async_session_maker() as db:
            auto = await db.get(Automation, a.id)
            thread = await ledger.thread_for(db, a.id)
            out = await deliver.deliver_brief(
                db, automation=auto, job_id=job.id, thread=thread,
                groups=_GROUPS, title="T",
                delivery={"channels": ["gmail_draft"], "format": "ranked",
                          "cadence": "run"},
                idem_prefix=idem)
        got = out["gmail_draft"]
        assert (got["status"] if expected == "delivered"
                else got["reason"]) == expected, out
    assert len([c for c in calls
                if c["tool"] == "gmail__create_draft"]) == 1, calls
