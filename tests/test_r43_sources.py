"""R43 — the account's real objects, and the preview in the service's shape.

Three things this file holds, and each one shipped as a defect before it
was a test.

  1. SOURCES ARE THE ACCOUNT  — `account_sources` enumerates what the
                                connected account actually holds, with a
                                `kind` out of `spec.FOCUS_KINDS`, because
                                picking one becomes a pin descriptor and
                                R39 proved an invented kind 409s every
                                write while the canvas has already drawn
                                the tick.
  2. WHO IS ITS OWN FIELD     — the sender used to be glued onto the
                                snippet, so the design's
                                who/line/snippet/when row could not be
                                drawn at all. `hot` is the blue dot, and
                                it means a DIFFERENT provable thing per
                                service — unread mail, a message that
                                names you, a P1 or a deadline, a meeting
                                inside a day.
  3. THE HEADER IS THE SERVICE — "Inbox · 18 unread", "#platform · 52
                                messages", "Pull requests · 2 open". One
                                generic caption over eight services read
                                as a debug dump.

And the rule the whole module exists for, re-proved on the new keys: an
unreachable provider is a REASON, never an empty list — but
`account_sources` returns a plain list and therefore collapses the two,
which is why it holds its own tight deadline and why the failure case is
asserted here rather than assumed.
"""

import json
from datetime import datetime, timedelta, timezone

import pytest

from app.agent.automations import contents
from app.agent.automations.spec import FOCUS_KINDS


def _ok(payload: dict) -> dict:
    return {"kind": "ok", "content": json.dumps(payload)}


_LIVE = {"connected": True, "status": "active"}
_DOWN = {"kind": "provider_down", "retryable": True,
         "message": "provider not responding"}


def _iso(delta_hours: float) -> str:
    return (datetime.now(timezone.utc) + timedelta(hours=delta_hours)).replace(
        microsecond=0).isoformat().replace("+00:00", "Z")


def _fake(monkeypatch, handler):
    """The one way this repo fakes a provider: monkeypatch the platform
    dispatch RPC, which is the single seam every reader goes through."""
    async def _dispatch(user_id, *, connector_id, tool_name, tool_input,
                        **kw):
        return handler(tool_name, tool_input)
    monkeypatch.setattr(
        "app.agent.automations.registry.dispatch_via_platform", _dispatch,
    )


# ─────────────────────────────────────────── 1. the account's objects

@pytest.mark.asyncio
async def test_every_source_kind_is_pinnable(monkeypatch):
    """The rule that makes a source pickable at all. A kind outside
    `FOCUS_KINDS` is refused `bad_focus_kind` by the pin endpoint —
    after the canvas has drawn the tick, so the user sees a pin that is
    not there."""
    def _h(tool, ti):
        if tool == "slack__list_channels":
            return _ok({"channels": [
                {"id": "C1", "name": "platform", "type": "public_channel",
                 "is_member": True, "num_members": 24},
                {"id": "D1", "type": "im", "is_member": True,
                 "user_name": "Sara Chen"},
            ]})
        if tool == "teams__list_chats":
            return _ok({"chats": [{"id": "19:x", "topic": "Billing squad",
                                   "last_updated_at": _iso(-3)}]})
        if tool == "jira__search_issues":
            return _ok({"issues": [
                {"key": "ENG-1", "summary": "Ship", "project": "ENG",
                 "status": "To Do", "duedate": _iso(30)[:10]},
            ]})
        if tool == "github__list_repos":
            return _ok({"repos": [{"full_name": "toup/platform",
                                   "open_issues_count": 12,
                                   "pushed_at": _iso(-2)}]})
        if tool == "notion__search":
            return _ok({"results": [
                {"id": "p1", "object": "page", "title": "Billing rewrite",
                 "last_edited_time": _iso(-2)},
                {"id": "d1", "object": "data_source", "title": "Q3",
                 "last_edited_time": _iso(-50)},
            ]})
        if tool == "calendar__list_events":
            return _ok({"events": [{"id": "e1", "summary": "Standup"}]})
        if tool == "gmail__list_messages":
            return _ok({"messages": [], "result_size": 18})
        return _ok({})
    _fake(monkeypatch, _h)
    for cid in ("gmail", "outlook", "slack", "teams", "jira", "github",
                "notion", "calendar"):
        found = await contents.account_sources("u", cid)
        assert found, cid
        for src in found:
            assert src["kind"] in FOCUS_KINDS, (cid, src)
            assert set(src) == {"id", "name", "meta", "short", "kind",
                                "count"}
            assert src["id"] and src["name"]


@pytest.mark.asyncio
async def test_gmail_inbox_is_counted_and_the_other_labels_are_not(
    monkeypatch,
):
    """One probe, two answers. Gmail has no label-listing tool, so the
    system labels are named from a constant — but the Inbox's number is
    the account's real unread, and a label nothing counted says None
    rather than 0."""
    _fake(monkeypatch, lambda t, ti: _ok(
        {"messages": [], "result_size": 18}))
    found = await contents.account_sources("u", "gmail")
    by_id = {s["id"]: s for s in found}
    assert by_id["INBOX"]["count"] == 18
    assert by_id["INBOX"]["meta"] == "18 unread"
    assert by_id["STARRED"]["count"] is None
    assert by_id["STARRED"]["meta"] == "kept for later"


@pytest.mark.asyncio
async def test_slack_sources_are_joined_conversations_only(monkeypatch):
    """`is_member` is the filter, for the same reason the reader uses
    it: the manifest has no `chat:write.public`, so a channel the
    workspace never joined is neither readable nor postable."""
    _fake(monkeypatch, lambda t, ti: _ok({"channels": [
        {"id": "C1", "name": "platform", "type": "public_channel",
         "is_member": True, "num_members": 24},
        {"id": "C9", "name": "general", "type": "public_channel",
         "is_member": False, "num_members": 900},
        {"id": "D1", "type": "im", "is_member": True,
         "user_name": "Sara Chen"},
    ]}))
    found = await contents.account_sources("u", "slack")
    assert [s["id"] for s in found] == ["C1", "D1"]
    assert found[0]["short"] == "#platform"
    assert found[0]["meta"] == "24 in the channel"
    assert found[1]["name"] == "Sara Chen"


@pytest.mark.asyncio
async def test_jira_sources_are_the_boards_the_user_has_tickets_on(
    monkeypatch,
):
    """`jira__list_projects` would list the hundred nobody here works
    on. One `assignee = currentUser()` search names the boards that are
    theirs and counts them in the same call — which is where "4 open ·
    1 due <day>" comes from."""
    due = (datetime.now(timezone.utc) + timedelta(days=2)).date().isoformat()
    _fake(monkeypatch, lambda t, ti: _ok({"issues": [
        {"key": "ENG-1", "summary": "a", "project": "ENG",
         "status": "To Do", "duedate": due},
        {"key": "ENG-2", "summary": "b", "project": "ENG",
         "status": "To Do"},
        {"key": "OPS-9", "summary": "c", "project": "OPS",
         "status": "To Do"},
    ]}))
    found = {s["id"]: s for s in await contents.account_sources("u", "jira")}
    assert found["ENG"]["kind"] == "project" and found["ENG"]["count"] == 2
    assert found["ENG"]["meta"].startswith("2 open · 1 due ")
    assert found["OPS"]["meta"] == "1 open"  # "open" is an adjective


@pytest.mark.asyncio
async def test_github_source_meta_says_open_never_prs(monkeypatch):
    """GitHub's `open_issues_count` counts issues AND pull requests
    together — its own documented meaning. "2 PRs" over that number is
    a lie the user can check in one tap."""
    _fake(monkeypatch, lambda t, ti: _ok({"repos": [
        {"full_name": "toup/platform", "open_issues_count": 12,
         "pushed_at": _iso(-2)},
    ]}))
    src = (await contents.account_sources("u", "github"))[0]
    assert src["kind"] == "repo" and src["short"] == "platform"
    assert src["meta"] == "12 open · pushed today"
    assert "PR" not in src["meta"]


@pytest.mark.asyncio
async def test_notion_pages_and_databases_take_different_kinds(monkeypatch):
    _fake(monkeypatch, lambda t, ti: _ok({"results": [
        {"id": "p1", "object": "page", "title": "Billing rewrite",
         "last_edited_time": _iso(-2)},
        {"id": "d1", "object": "data_source", "title": "Q3 planning",
         "last_edited_time": _iso(-50)},
    ]}))
    found = {s["id"]: s for s in await contents.account_sources("u", "notion")}
    assert found["p1"]["kind"] == "doc" and found["p1"]["meta"] == "edited today"
    assert found["d1"]["kind"] == "board"


@pytest.mark.asyncio
async def test_calendar_offers_the_one_calendar_it_can_actually_read(
    monkeypatch,
):
    """Not a shortcut: the connector holds `calendar.events` and
    `calendarList` needs `calendar.readonly`, which no user has ever
    granted. Listing calendars it cannot read would be a picker that
    writes nowhere."""
    _fake(monkeypatch, lambda t, ti: _ok(
        {"events": [{"id": f"e{n}"} for n in range(6)]}))
    found = await contents.account_sources("u", "calendar")
    assert len(found) == 1
    assert found[0]["short"] == "Your week"
    assert found[0]["meta"] == "6 meetings · this week"
    assert found[0]["count"] == 6


@pytest.mark.asyncio
async def test_a_pinned_source_leads_the_list(monkeypatch):
    """Same reason `_ordered` leads the contents sheet with a pinned
    channel: the pin is where this automation already starts, and a
    picker that buries it reads as though the pin was lost."""
    _fake(monkeypatch, lambda t, ti: _ok({"channels": [
        {"id": "C1", "name": "a", "type": "public_channel",
         "is_member": True},
        {"id": "C2", "name": "b", "type": "public_channel",
         "is_member": True},
        {"id": "C3", "name": "c", "type": "public_channel",
         "is_member": True},
    ]}))
    found = await contents.account_sources(
        "u", "slack", focus=[{"kind": "channel", "id": "C3", "label": "#c"}],
    )
    assert [s["id"] for s in found] == ["C3", "C1", "C2"]


@pytest.mark.asyncio
async def test_the_list_is_bounded(monkeypatch):
    _fake(monkeypatch, lambda t, ti: _ok({"channels": [
        {"id": f"C{n}", "name": f"c{n}", "type": "public_channel",
         "is_member": True} for n in range(40)
    ]}))
    assert len(await contents.account_sources("u", "slack")) == 8


@pytest.mark.asyncio
async def test_a_refusing_provider_yields_a_list_not_an_exception(
    monkeypatch,
):
    """`account_sources` is called while a whole canvas payload is being
    built; one dead account may not take the other seven with it."""
    _fake(monkeypatch, lambda t, ti: _DOWN)
    for cid in ("slack", "teams", "jira", "github", "notion", "calendar"):
        assert await contents.account_sources("u", cid) == []
    # Gmail's list is a constant; the probe that failed only costs the
    # Inbox its number, and a picker with three real labels in it beats
    # an empty one.
    gmail = await contents.account_sources("u", "gmail")
    assert [s["id"] for s in gmail] == ["INBOX", "IMPORTANT", "STARRED"]
    assert gmail[0]["count"] is None


@pytest.mark.asyncio
async def test_an_unknown_connector_offers_nothing(monkeypatch):
    _fake(monkeypatch, lambda t, ti: _ok({}))
    assert await contents.account_sources("u", "linkedin") == []


# ───────────────────────────────────────────────── 2. who, and hot

@pytest.mark.asyncio
async def test_gmail_who_is_its_own_field_and_hot_is_unread(monkeypatch):
    """The sender used to be glued onto the snippet ("Sara Chen — can we
    move it"), so the app had one string where the design has two and no
    way to take it apart. And Gmail's list returns no `labelIds` even
    with `include_body`, so the unread set comes from a second id-only
    read — which is also where the header's real total comes from."""
    seen = []

    def _h(tool, ti):
        seen.append(ti)
        if ti.get("query") == "is:unread in:inbox":
            return _ok({"messages": [{"id": "m2"}], "result_size": 18})
        return _ok({"messages": [
            {"id": "m1", "headers": {"From": "Sara Chen <sara@x.com>",
                                     "Subject": "Re: launch",
                                     "Date": "Fri, 29 Aug 2026 09:14:00 +0000"},
             "snippet": "can we move it"},
            {"id": "m2", "headers": {"From": "Omid <omid@x.com>",
                                     "Subject": "Invoice",
                                     "Date": "Fri, 29 Aug 2026 10:00:00 +0000"},
             "snippet": "attached"},
        ]})
    _fake(monkeypatch, _h)
    env = await contents.account_contents(
        "u", connector_id="gmail", connection=_LIVE,
    )
    rows = env["groups"][0]["rows"]
    assert rows[0]["who"] == "Sara Chen"
    assert rows[0]["primary"] == "Re: launch"
    assert rows[0]["secondary"] == "can we move it"
    assert rows[0]["hot"] is False and rows[1]["hot"] is True
    # The shipped app still reads title/sub; both keys keep meaning what
    # they meant before `who` existed.
    assert rows[0]["title"] == "Re: launch"
    assert rows[0]["sub"] == "Sara Chen — can we move it"
    assert env["total"] == 18
    assert env["preview"] == {"title": "Inbox", "meta": "18 unread"}
    assert env["noun"] == "messages"


@pytest.mark.asyncio
async def test_an_uncounted_mailbox_does_not_borrow_the_word_unread(
    monkeypatch,
):
    """The probe that counts unread is a separate call, and it can fail
    on its own. Falling back to the rows in hand under the word
    "unread" would put a number in the header that the body cannot
    support — so the unit falls back with the count."""
    def _h(tool, ti):
        if ti.get("query") == "is:unread in:inbox":
            return _DOWN
        return _ok({"messages": [
            {"id": "m1", "headers": {"Subject": "S"}},
            {"id": "m2", "headers": {"Subject": "T"}},
        ]})
    _fake(monkeypatch, _h)
    env = await contents.account_contents(
        "u", connector_id="gmail", connection=_LIVE,
    )
    assert env["ok"] is True and env["total"] is None
    assert env["preview"] == {"title": "Inbox", "meta": "2 messages"}
    assert all(r["hot"] is False for r in env["groups"][0]["rows"])


@pytest.mark.asyncio
async def test_gmail_hot_prefers_the_row_over_the_probe(monkeypatch):
    """R43 — the list call carries `labelIds` now (the per-message GET
    it already makes has always returned them), so unread is a property
    of the ROW. The `is:unread` probe shrank to one id and survives only
    for `total`; the ids it returns stay as the fallback for a row that
    carries no `labelIds` KEY at all.

    Presence, not truthiness: `labelIds: []` is "Gmail says this is
    read" and absence is "nobody said", and those are different answers.
    """
    seen = []

    def _h(tool, ti):
        seen.append(ti)
        if ti.get("query") == "is:unread in:inbox":
            return _ok({"messages": [{"id": "zz"}], "result_size": 41})
        return _ok({"messages": [
            {"id": "m1", "headers": {"Subject": "read"}, "labelIds": ["INBOX"]},
            {"id": "m2", "headers": {"Subject": "unread"},
             "labelIds": ["INBOX", "UNREAD"]},
            {"id": "m3", "headers": {"Subject": "said nothing"}},
            {"id": "zz", "headers": {"Subject": "said nothing, but probed"}},
        ]})
    _fake(monkeypatch, _h)
    env = await contents.account_contents(
        "u", connector_id="gmail", connection=_LIVE,
    )
    assert [r["hot"] for r in env["groups"][0]["rows"]] == [
        False, True, False, True]
    # One id is all `result_size` needs, and the count is the MAILBOX's,
    # never the length of what came back.
    probe = [t for t in seen if t.get("query") == "is:unread in:inbox"]
    assert probe and probe[0]["max_results"] == 1
    assert env["total"] == 41


@pytest.mark.asyncio
async def test_an_outlook_folder_pin_scopes_the_read_it_is_not_searched_for(
    monkeypatch,
):
    """R43 — a Graph folder is a COLLECTION, not a search term. The id
    used to go out as `query`, i.e. a KQL `$search` for a base64 string:
    the whole mailbox was read and the pinned group matched nothing."""
    seen = []
    _fake(monkeypatch, lambda t, ti: (seen.append((t, ti)),
                                      _ok({"messages": []}))[1])
    await contents.account_contents(
        "u", connector_id="outlook", connection=_LIVE,
        focus=[{"kind": "folder", "id": "AAMkAD==", "label": "Vendors"}],
    )
    reads = [ti for t, ti in seen if t == "outlook__list_messages"]
    assert {"folder": "AAMkAD=="}.items() <= reads[0].items()
    assert "query" not in reads[0]
    # The unpinned "Recent" group still reads the whole mailbox.
    assert "folder" not in reads[-1]


@pytest.mark.asyncio
async def test_outlook_sources_are_the_real_folders_and_fall_back_named(
    monkeypatch,
):
    """§5's "one per mail folder (≤6)". `well_known` is matched BY ID,
    so Inbox / Archive / Sent Items keep the design's names in a
    localised mailbox; a listing that cannot be read still offers those
    three, because they are addressable by well-known segment in every
    locale and an empty picker would read as an empty mailbox."""
    _fake(monkeypatch, lambda t, ti: _ok({"folders": [
        {"id": "f-in", "name": "Posteingang", "well_known": "inbox",
         "unread_count": 18},
        {"id": "f-sent", "name": "Gesendete", "well_known": "sentitems"},
        {"id": "f-v", "name": "Vendors", "unread_count": 2},
    ]}))
    found = {s["name"]: s for s in await contents.account_sources("u", "outlook")}
    assert found["Inbox"]["id"] == "f-in" and found["Inbox"]["count"] == 18
    assert found["Inbox"]["meta"] == "18 unread"
    assert found["Sent, for context"]["meta"] == "what you sent, for context"
    assert found["Vendors"]["meta"] == "2 unread"
    assert all(s["kind"] == "folder" for s in found.values())

    _fake(monkeypatch, lambda t, ti: _DOWN)
    named = await contents.account_sources("u", "outlook")
    assert [s["id"] for s in named] == ["inbox", "archive", "sentitems"]


@pytest.mark.asyncio
async def test_outlook_hot_is_the_graph_read_flag(monkeypatch):
    _fake(monkeypatch, lambda t, ti: _ok({"messages": [
        {"id": "m1", "from": "sara@x.com", "subject": "Re: launch",
         "preview": "can we move it", "received_at": "2026-08-29T09:14:00Z",
         "is_read": False},
        {"id": "m2", "from": "omid@x.com", "subject": "FYI",
         "preview": "no action", "received_at": "2026-08-29T08:00:00Z",
         "is_read": True},
    ]}))
    env = await contents.account_contents(
        "u", connector_id="outlook", connection=_LIVE,
    )
    rows = env["groups"][0]["rows"]
    assert [r["hot"] for r in rows] == [True, False]
    assert rows[0]["who"] == "sara@x.com"
    assert env["preview"]["meta"] == "1 unread"


@pytest.mark.asyncio
async def test_slack_hot_is_a_message_that_names_you(monkeypatch):
    """A user token gets no per-message read state from the Web API, so
    "unread" is not answerable here. A message that NAMES you is — the
    provider marks it off the raw `<@U…>` run before rendering — and
    "someone is asking you" is what the dot is for."""
    def _h(tool, ti):
        if tool == "slack__list_channels":
            return _ok({"channels": [{"id": "C1", "name": "platform",
                                      "is_member": True}]})
        return _ok({"messages": [
            {"ts": "1756400000.0", "from": "Dana Cole",
             "text": "Anyone own the retry flag?", "mentions_me": True},
            {"ts": "1756400100.0", "from": "Omid", "text": "shipping"},
        ]})
    _fake(monkeypatch, _h)
    env = await contents.account_contents(
        "u", connector_id="slack", connection=_LIVE,
    )
    group = env["groups"][0]
    assert [r["hot"] for r in group["rows"]] == [True, False]
    assert group["rows"][0]["who"] == "Dana Cole"
    assert group["rows"][0]["primary"] == "Anyone own the retry flag?"
    # …and the pre-R43 pair keeps its own meaning: Slack titled a row
    # with the speaker, not with the message.
    assert group["rows"][0]["title"] == "Dana Cole"
    assert group["rows"][0]["sub"] == "Anyone own the retry flag?"
    assert group["meta"] == "2 messages · 1 name you"
    # The header is the channel, not the picker: a Slack card names what
    # it is showing.
    assert env["preview"] == {"title": "#platform", "meta": "2 messages"}


@pytest.mark.asyncio
async def test_teams_hot_is_the_chats_own_read_mark(monkeypatch):
    """Graph puts the caller's `viewpoint.lastMessageReadDateTime` on
    every /chats row. Chat messages carry no read state of their own, so
    this is the one honest definition of unread the connector has."""
    def _h(tool, ti):
        if tool == "teams__list_chats":
            return _ok({"chats": [{"id": "19:x", "topic": "Billing squad",
                                   "last_read_at": _iso(-2)}]})
        return _ok({"messages": [
            {"id": "1", "sender": "Dana", "body": "new one",
             "created_at": _iso(-1)},
            {"id": "2", "sender": "Omid", "body": "old one",
             "created_at": _iso(-5)},
        ]})
    _fake(monkeypatch, _h)
    env = await contents.account_contents(
        "u", connector_id="teams", connection=_LIVE,
    )
    rows = env["groups"][0]["rows"]
    assert [r["hot"] for r in rows] == [True, False]
    assert rows[0]["who"] == "Dana"
    assert env["preview"] == {"title": "Team channels", "meta": "1 channel"}
    assert env["noun"] == "posts"


@pytest.mark.asyncio
async def test_teams_marks_nothing_when_graph_did_not_say(monkeypatch):
    """No read mark is "Graph did not answer", not "everything is
    unread". A dot the module invented is the app claiming urgency on no
    evidence."""
    def _h(tool, ti):
        if tool == "teams__list_chats":
            return _ok({"chats": [{"id": "19:x", "topic": "Billing"}]})
        return _ok({"messages": [{"id": "1", "sender": "Dana", "body": "hi",
                                  "created_at": _iso(-1)}]})
    _fake(monkeypatch, _h)
    env = await contents.account_contents(
        "u", connector_id="teams", connection=_LIVE,
    )
    assert env["groups"][0]["rows"][0]["hot"] is False


@pytest.mark.asyncio
async def test_jira_hot_is_a_p1_or_a_deadline_inside_two_days(monkeypatch):
    soon = (datetime.now(timezone.utc) + timedelta(days=1)).date().isoformat()
    far = (datetime.now(timezone.utc) + timedelta(days=20)).date().isoformat()
    over = (datetime.now(timezone.utc) - timedelta(days=9)).date().isoformat()
    _fake(monkeypatch, lambda t, ti: _ok({"issues": [
        {"key": "ENG-0", "summary": "Overdue", "project": "ENG",
         "status": "To Do", "duedate": over, "priority": "Low"},
        {"key": "ENG-1", "summary": "Due soon", "project": "ENG",
         "status": "To Do", "duedate": soon, "priority": "Medium"},
        {"key": "ENG-2", "summary": "P1", "project": "ENG",
         "status": "To Do", "duedate": far, "priority": "Highest"},
        {"key": "ENG-3", "summary": "Quiet", "project": "ENG",
         "status": "To Do", "duedate": far, "priority": "Low",
         "assignee": {"display_name": "Nariman"}},
    ]}))
    env = await contents.account_contents(
        "u", connector_id="jira", connection=_LIVE,
    )
    rows = {r["id"]: r for r in env["groups"][0]["rows"]}
    assert rows["ENG-1"]["hot"] is True and rows["ENG-2"]["hot"] is True
    assert rows["ENG-3"]["hot"] is False
    # A deadline that passed nine days ago is the most urgent row on the
    # board. A symmetric "near now" window drops it off the dot the
    # moment it goes past the bound — which is exactly backwards.
    assert rows["ENG-0"]["hot"] is True
    assert rows["ENG-3"]["who"] == "Nariman"
    # The due date is still on the pre-R43 `sub`, where the shipped app
    # reads it from.
    assert f"due {soon}" in rows["ENG-1"]["sub"]
    assert env["preview"] == {"title": "Assigned to you", "meta": "4 open"}
    assert env["noun"] == "issues"


@pytest.mark.asyncio
async def test_github_hot_is_a_pull_request_that_is_not_a_draft(monkeypatch):
    """`/repos/{o}/{r}/issues` carries no check status, so "failing
    checks" cannot be answered from this read. `draft` it does carry,
    and a PR that is not a draft is the one waiting on a human."""
    def _h(tool, ti):
        if tool == "github__list_repos":
            return _ok({"repos": [{"full_name": "toup/platform"}]})
        return _ok({"issues": [
            {"number": 7, "title": "Retry flag", "user": "sara",
             "is_pull_request": True, "draft": False},
            {"number": 8, "title": "WIP", "user": "omid",
             "is_pull_request": True, "draft": True},
            {"number": 9, "title": "An issue", "user": "sara",
             "is_pull_request": False},
        ]})
    _fake(monkeypatch, _h)
    env = await contents.account_contents(
        "u", connector_id="github", connection=_LIVE,
    )
    rows = env["groups"][0]["rows"]
    assert [r["id"] for r in rows] == ["7", "8"]
    assert [r["hot"] for r in rows] == [True, False]
    assert rows[0]["who"] == "sara"
    assert env["preview"] == {"title": "Pull requests", "meta": "2 open"}
    assert env["noun"] == "pull requests"


@pytest.mark.asyncio
async def test_calendar_has_no_who_and_is_hot_inside_a_day(monkeypatch):
    """§4: a service with no who serves `""`, and the app draws the row
    without that slot — never the string "None"."""
    _fake(monkeypatch, lambda t, ti: _ok({"events": [
        {"id": "e1", "summary": "Standup",
         "start": {"dateTime": _iso(3)}},
        {"id": "e2", "summary": "Next week",
         "start": {"dateTime": _iso(120)}},
    ]}))
    env = await contents.account_contents(
        "u", connector_id="calendar", connection=_LIVE,
    )
    rows = env["groups"][0]["rows"]
    assert [r["who"] for r in rows] == ["", ""]
    assert [r["hot"] for r in rows] == [True, False]
    assert env["preview"] == {"title": "Your week", "meta": "2 meetings"}
    assert env["noun"] == "meetings"


# ────────────────────────────────────────────────── 3. notion reads

@pytest.mark.asyncio
async def test_notion_can_be_looked_inside_at_all(monkeypatch):
    """R43: Notion was not in `_READERS`, so a connector the design puts
    in the PLANS band answered "There is no way to look inside Notion
    yet" — the same refusal Teams wore in R39, with the same
    consequence, that an expired credential underneath it was never
    surfaced either."""
    _fake(monkeypatch, lambda t, ti: _ok({"results": [
        {"id": "p1", "object": "page", "title": "Billing rewrite",
         "last_edited_time": _iso(-2)},
        {"id": "d1", "object": "data_source", "title": "Q3 planning",
         "last_edited_time": _iso(-80)},
    ]}))
    env = await contents.account_contents(
        "u", connector_id="notion", connection=_LIVE,
    )
    assert env["ok"] is True and env["reason"] is None
    groups = {g["key"]: g for g in env["groups"]}
    assert groups["pages"]["rows"][0]["pin"] == {
        "kind": "doc", "id": "p1", "label": "Billing rewrite"}
    assert groups["databases"]["rows"][0]["pin"]["kind"] == "board"
    assert groups["pages"]["rows"][0]["hot"] is True
    assert env["preview"] == {"title": "Tagged to you", "meta": "2 pages"}
    assert env["noun"] == "pages"


@pytest.mark.asyncio
async def test_a_pinned_notion_page_leads_its_group(monkeypatch):
    _fake(monkeypatch, lambda t, ti: _ok({"results": [
        {"id": "p1", "object": "page", "title": "A"},
        {"id": "p2", "object": "page", "title": "B"},
    ]}))
    env = await contents.account_contents(
        "u", connector_id="notion", connection=_LIVE,
        focus=[{"kind": "doc", "id": "p2", "label": "B"}],
    )
    assert [r["id"] for r in env["groups"][0]["rows"]] == ["p2", "p1"]


@pytest.mark.asyncio
async def test_an_unreadable_notion_is_a_reason_not_an_empty_page_list(
    monkeypatch,
):
    _fake(monkeypatch, lambda t, ti: _DOWN)
    env = await contents.account_contents(
        "u", connector_id="notion", connection=_LIVE,
    )
    assert env["ok"] is False
    assert env["reason"]["code"] == "unreachable"
    assert "Notion" in env["reason"]["sentence"]


# ──────────────────────────────────── 4. the envelope's own promises

@pytest.mark.asyncio
async def test_every_new_key_is_served_even_when_nothing_could_be_read(
    monkeypatch,
):
    """§4 says each of these is ALWAYS served. An app that must branch on
    "is this key here" before it can branch on `ok` has two failure
    vocabularies where this module promises one."""
    _fake(monkeypatch, lambda t, ti: _DOWN)
    for connection, cid in ((_LIVE, "slack"),
                            ({"connected": False}, "slack"),
                            ({"connected": True, "status": "expired"},
                             "slack"),
                            (_LIVE, "drive")):
        env = await contents.account_contents(
            "u", connector_id=cid, connection=connection,
        )
        assert set(env) >= {"noun", "total", "preview", "groups", "ok",
                            "reason", "count", "focus", "truncated"}
        assert set(env["preview"]) == {"title", "meta"}
        assert env["total"] is None
        assert env["ok"] is False


@pytest.mark.asyncio
async def test_rows_and_items_are_the_same_rows(monkeypatch):
    """Two names, ONE list — the same object, not a copy. An older phone
    reads `items`, a current one reads `rows`, and a copy would let the
    two drift the first time anything downstream trimmed one of them.
    The projection is taken after `_MAX_ITEMS`, so a cap that fires
    cannot leave `rows` holding the items it just dropped."""
    def _h(tool, ti):
        if tool == "slack__list_channels":
            return _ok({"channels": [
                {"id": f"C{n}", "name": f"c{n}", "is_member": True}
                for n in range(6)]})
        return _ok({"messages": [
            {"ts": f"175640000{n}.0", "from": "Sara", "text": "x"}
            for n in range(10)]})
    _fake(monkeypatch, _h)
    env = await contents.account_contents(
        "u", connector_id="slack", connection=_LIVE,
    )
    assert env["count"] == 60
    for g in env["groups"]:
        assert g["rows"] is g["items"]
    assert sum(len(g["rows"]) for g in env["groups"]) == env["count"]


@pytest.mark.asyncio
async def test_a_group_carries_its_own_meta_count_short_and_selection(
    monkeypatch,
):
    """`selected` is "did the automation PICK this source", which is NOT
    "did the user pin something in it" — two writes, two meanings, and
    merging them ticks a source nobody chose."""
    def _h(tool, ti):
        if tool == "slack__list_channels":
            return _ok({"channels": [
                {"id": "C1", "name": "platform", "is_member": True},
                {"id": "C2", "name": "oncall", "is_member": True},
            ]})
        return _ok({"messages": [
            {"ts": "1756400000.0", "from": "Sara", "text": "x"}]})
    _fake(monkeypatch, _h)
    env = await contents.account_contents(
        "u", connector_id="slack", connection=_LIVE, sources=["C2"],
        focus=[{"kind": "channel", "id": "C1", "label": "#platform"}],
    )
    by_key = {g["key"]: g for g in env["groups"]}
    assert by_key["C1"]["pinned"] is True and by_key["C1"]["selected"] is False
    assert by_key["C2"]["pinned"] is False and by_key["C2"]["selected"] is True
    assert by_key["C1"]["short"] == "#platform"
    assert by_key["C1"]["count"] == 1 and by_key["C1"]["meta"] == "1 message"


@pytest.mark.asyncio
async def test_an_unsaid_selection_is_false_not_a_guess(monkeypatch):
    def _h(tool, ti):
        if tool == "slack__list_channels":
            return _ok({"channels": [{"id": "C1", "name": "a",
                                      "is_member": True}]})
        return _ok({"messages": []})
    _fake(monkeypatch, _h)
    env = await contents.account_contents(
        "u", connector_id="slack", connection=_LIVE,
    )
    assert env["groups"][0]["selected"] is False


@pytest.mark.asyncio
async def test_a_row_is_pinned_by_its_own_id_never_its_containers(
    monkeypatch,
):
    """R42's founder P4, restated on the new key: one tap on one message
    drew a checkmark on all ten of them because every row carried its
    GROUP's descriptor."""
    def _h(tool, ti):
        if tool == "slack__list_channels":
            return _ok({"channels": [{"id": "C1", "name": "a",
                                      "is_member": True}]})
        return _ok({"messages": [
            {"ts": "1.0", "from": "Sara", "text": "x"},
            {"ts": "2.0", "from": "Omid", "text": "y"},
        ]})
    _fake(monkeypatch, _h)
    env = await contents.account_contents(
        "u", connector_id="slack", connection=_LIVE,
        focus=[{"kind": "thread", "id": "C1#1.0", "label": "Sara: x"}],
    )
    assert [r["pinned"] for r in env["groups"][0]["rows"]] == [True, False]


@pytest.mark.asyncio
async def test_a_failed_group_counts_nothing(monkeypatch):
    """0 there would be this module's word for "there is nothing here",
    said about an account it could not read."""
    def _h(tool, ti):
        if tool == "slack__list_channels":
            return _ok({"channels": [
                {"id": "C1", "name": "a", "is_member": True},
                {"id": "C2", "name": "b", "is_member": True},
            ]})
        if ti.get("channel") == "C2":
            return _DOWN
        return _ok({"messages": [{"ts": "1.0", "from": "S", "text": "x"}]})
    _fake(monkeypatch, _h)
    env = await contents.account_contents(
        "u", connector_id="slack", connection=_LIVE,
    )
    by_key = {g["key"]: g for g in env["groups"]}
    assert by_key["C1"]["count"] == 1
    assert by_key["C2"]["count"] is None and by_key["C2"]["reason"]
