"""R43 §5/§6/§7 — the tool surface Teams, Calendar and Notion needed
before their pickers could offer anything.

Every option R43 asks these three connectors for was blocked on the
same thing: the tool could not SAY it. A "Meetings I own" chip has
nowhere to compile to while the event row carries no organiser; "I am
@mentioned" is not an event while a chat message carries no mentions;
`calendar_hold` cannot be staged while `calendar__create_event`
declares no pinned target and so validates as a read. These tests pin
the surface that fixes each one, and — the half that matters — pin that
each new field and parameter actually CHANGES the request or the rows,
rather than merely existing.

They talk to no network: every provider's one request helper is
replaced with a recorder, so what is asserted is the request this code
would have sent and the payload it would have returned.
"""

import json

import pytest

from app.connectors import base as cbase
from app.connectors.calendar import provider as cal
from app.connectors.notion import provider as notion
from app.connectors.teams import provider as teams
from app.services.connector_registry import ConnectorRegistry


CTX = cbase.ConnectorContext(user_id="u1", access_token="tok")


def _ok(result):
    assert isinstance(result, cbase.ConnectorOk), result
    return json.loads(result.content)


class _Recorder:
    """Stands in for a provider's single request helper."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, *a, **kw):
        self.calls.append((a, kw))

        async def _run():
            if not self.responses:
                return {}
            return (self.responses.pop(0) if len(self.responses) > 1
                    else self.responses[0])

        return _run()


# ── Registry: declared AND dispatchable ──────────────────────────────


@pytest.fixture(scope="module")
def registry():
    r = ConnectorRegistry()
    r.load_all(include_experimental=True)
    assert not r.alarms(), [a.reason for a in r.alarms()]
    return r


@pytest.fixture(scope="module")
def entries(registry):
    return {e["connector_id"]: e for e in registry.automation_registry()}


@pytest.mark.asyncio
@pytest.mark.parametrize("mod,helper,tools", [
    (teams, "microsoft_graph_request",
     ("teams__list_chats", "teams__read_chat_messages")),
    (cal, "google_request",
     ("calendar__list_events", "calendar__create_event",
      "calendar__delete_event")),
    (notion, "_notion_request",
     ("notion__append_blocks", "notion__query_database",
      "notion__search")),
])
async def test_every_declared_tool_is_reachable_in_execute(
    mod, helper, tools, monkeypatch,
):
    """A tool declared in a manifest but not dispatched by its provider
    is worse than no tool — the model can call it and gets "unknown
    tool" back from the branch every `execute` ends with."""
    prov = {teams: teams.TeamsProvider, cal: cal.CalendarProvider,
            notion: notion.NotionProvider}[mod]()
    for name in tools:
        monkeypatch.setattr(mod, helper, _Recorder({}))
        result = await prov.execute(name, {}, CTX)
        message = getattr(result, "message", "")
        assert "unknown" not in message.lower(), (
            f"{name} is declared but the provider does not dispatch it"
        )


def test_new_tools_are_declared_where_they_are_dispatched(registry):
    for name in ("notion__append_blocks",):
        spec = registry.get_tool_spec(name)
        assert spec is not None, f"{name} dispatched but not declared"
        assert spec.mutates and spec.elevation


def test_teams_channel_messages_stays_undeclared(registry):
    """The one R43 ask refused. `ChannelMessage.Read.All` is
    admin-consent-only, and Entra fails the WHOLE consent request on an
    admin-only scope — so declaring the tool would cost every user their
    Teams connection. It is recorded in `scopes_optional`, never
    requested."""
    assert registry.get_tool_spec("teams__read_channel_messages") is None
    entry = registry.get("teams")
    assert ("https://graph.microsoft.com/ChannelMessage.Read.All"
            in entry.manifest.oauth.scopes_optional)
    assert ("https://graph.microsoft.com/ChannelMessage.Read.All"
            not in entry.manifest.oauth.scopes)


def test_no_connector_gained_an_oauth_scope(registry):
    """The R43 budget: these three connectors buy every new capability
    out of what was already consented. A new scope invalidates every
    existing grant, so one appearing here is a contract change."""
    assert registry.get("calendar").manifest.oauth.scopes == [
        "https://www.googleapis.com/auth/calendar.events"]
    assert registry.get("notion").manifest.oauth.scopes == []
    assert registry.get("teams").manifest.oauth.scopes == [
        "https://graph.microsoft.com/Team.ReadBasic.All",
        "https://graph.microsoft.com/Channel.ReadBasic.All",
        "https://graph.microsoft.com/Chat.Read",
        "https://graph.microsoft.com/ChatMessage.Send",
        "offline_access",
    ]


def test_calendar_hold_is_a_granted_pinned_write(entries, registry):
    """R28 declared calendar observe-only because create_event had no
    pinnable target, which made §1.2's `calendar_hold` validate as a
    READ and take no grant. Both halves now exist, and the scope behind
    it is the one already in `oauth.scopes`."""
    e = entries["calendar"]
    assert e["target_param_by_action"]["calendar__create_event"] == "calendar_id"
    assert e["scopes_write_by_action"]["calendar__create_event"] == [
        "https://www.googleapis.com/auth/calendar.events"]
    props = (registry.get_tool_spec(
        "calendar__create_event").input_schema or {}).get("properties") or {}
    assert "calendar_id" in props


def test_the_events_r43_names_are_declared(entries):
    assert [e["key"] for e in entries["calendar"]["events"]] == [
        "event_created", "meeting_soon", "no_agenda", "invitation_arrived"]
    assert [e["key"] for e in entries["teams"]["events"]] == [
        "chat_message_received", "mentioned"]
    assert [e["key"] for e in entries["notion"]["events"]] == [
        "page_added", "page_changed", "deadline_moved"]


def test_every_new_event_dedupes_on_a_key_its_rows_carry(entries):
    """A dedupe field the provider never emits makes every poll skip
    every row — silently, because `_extract_events` drops a row whose
    key is None."""
    keys = {
        ("notion", "page_changed"): "change_key",
        ("notion", "deadline_moved"): "deadline_key",
        ("teams", "mentioned"): "id",
        ("calendar", "meeting_soon"): "id",
    }
    for (cid, key), field in keys.items():
        ev = next(e for e in entries[cid]["events"] if e["key"] == key)
        assert ev["dedupe_field"] == field
    row = notion._page_summary({
        "id": "p1", "last_edited_time": "2026-08-31T10:00:00.000Z",
        "properties": {"Due": {"type": "date",
                               "date": {"start": "2026-09-04"}}},
    })
    assert row["change_key"] and row["deadline_key"]
    assert teams._message_row({"id": "m"}, None)["id"] == "m"
    assert cal._event_row({"id": "e"})["id"] == "e"


# ── Teams: who is "me" ───────────────────────────────────────────────


def test_self_is_deduced_from_a_one_member_one_on_one_chat():
    self_id, why = teams._identify_self([
        {"chatType": "oneOnOne", "members": [{"userId": "ME"}]},
        {"chatType": "group",
         "members": [{"userId": "ME"}, {"userId": "A"}]},
    ])
    assert self_id == "me" and "yourself" in why


def test_self_is_deduced_from_the_intersection_of_two_chats():
    self_id, _ = teams._identify_self([
        {"chatType": "oneOnOne",
         "members": [{"userId": "ME"}, {"userId": "A"}]},
        {"chatType": "oneOnOne",
         "members": [{"userId": "B"}, {"userId": "ME"}]},
    ])
    assert self_id == "me"


def test_self_is_refused_when_two_identities_share_every_chat():
    """The deduction must CLOSE. A colleague in every one of the user's
    chats leaves two candidates, and two is not an answer — guessing
    here misroutes a delivery to another person."""
    self_id, why = teams._identify_self([
        {"chatType": "group",
         "members": [{"userId": "ME"}, {"userId": "A"}, {"userId": "X"}]},
        {"chatType": "group",
         "members": [{"userId": "ME"}, {"userId": "A"}, {"userId": "Y"}]},
    ])
    assert self_id is None and "2 people" in why


def test_self_is_refused_from_a_single_chat():
    self_id, why = teams._identify_self([
        {"chatType": "group",
         "members": [{"userId": "ME"}, {"userId": "A"}]},
    ])
    assert self_id is None and "two or more" in why


@pytest.mark.asyncio
async def test_list_chats_marks_only_the_users_own_chat(monkeypatch):
    """§1.2's `teams_chat` needs a chat it can prove nobody else reads.
    `is_self_chat` is that proof and it is true for exactly one row."""
    rec = _Recorder({"value": [
        {"id": "c-self", "chatType": "oneOnOne",
         "members": [{"userId": "ME", "displayName": "Me"}]},
        {"id": "c-dana", "chatType": "oneOnOne",
         "members": [{"userId": "ME"}, {"userId": "DANA"}]},
    ]})
    monkeypatch.setattr(teams, "microsoft_graph_request", rec)
    out = _ok(await teams.TeamsProvider().execute(
        "teams__list_chats", {}, CTX))
    assert out["self_identified"] is True
    assert [c["id"] for c in out["chats"] if c["is_self_chat"]] == ["c-self"]
    assert out["chats"][1]["members"][0]["is_self"] is True
    assert out["chats"][1]["members"][1]["is_self"] is False


@pytest.mark.asyncio
async def test_list_chats_marks_nothing_when_self_is_unknown(monkeypatch):
    rec = _Recorder({"value": [
        {"id": "c1", "chatType": "group",
         "members": [{"userId": "ME"}, {"userId": "A"}]},
    ]})
    monkeypatch.setattr(teams, "microsoft_graph_request", rec)
    out = _ok(await teams.TeamsProvider().execute(
        "teams__list_chats", {}, CTX))
    assert out["self_identified"] is False
    assert out["self_note"]
    assert all(not c["is_self_chat"] for c in out["chats"])
    assert all(not m["is_self"] for c in out["chats"] for m in c["members"])


# ── Teams: the two §6 chips and the §7 event ─────────────────────────


def test_a_message_row_carries_its_author_kind_and_mentions():
    bot = teams._message_row(
        {"id": "m1", "from": {"application": {"id": "B", "displayName": "Jira"}},
         "body": {"content": "build red"}}, None)
    assert bot["author_type"] == "application"
    system = teams._message_row({"id": "m2", "from": None}, None)
    assert system["author_type"] == ""
    named = teams._message_row(
        {"id": "m3", "from": {"user": {"id": "DANA"}},
         "mentions": [{"mentionText": "Me",
                       "mentioned": {"user": {"id": "ME"}}}]}, "me")
    assert named["author_type"] == "user"
    assert named["mentions"][0]["id"] == "ME"
    assert named["mentions_me"] is True


def test_skip_bots_can_tell_a_bot_from_a_system_message():
    """"Skip bots" must not eat "Dana joined the chat": a system message
    has no author at all, and the row says so with "" rather than
    guessing at "application"."""
    rows = [
        teams._message_row({"id": "1", "from": {"user": {"id": "D"}}}, None),
        teams._message_row(
            {"id": "2", "from": {"application": {"id": "B"}}}, None),
        teams._message_row({"id": "3", "from": None}, None),
    ]
    kept = [r for r in rows if "application" not in (r["author_type"] or "")]
    assert [r["id"] for r in kept] == ["1", "3"]


@pytest.mark.asyncio
async def test_mentions_only_narrows_and_is_refused_rather_than_widened(
    monkeypatch,
):
    chats = {"value": [
        {"id": "a", "chatType": "oneOnOne",
         "members": [{"userId": "ME"}, {"userId": "A"}]},
        {"id": "b", "chatType": "oneOnOne",
         "members": [{"userId": "ME"}, {"userId": "B"}]},
    ]}
    msgs = {"value": [
        {"id": "m1", "from": {"user": {"id": "A"}},
         "mentions": [{"mentioned": {"user": {"id": "ME"}}}]},
        {"id": "m2", "from": {"user": {"id": "A"}}, "mentions": []},
    ]}
    rec = _Recorder(chats, msgs)
    monkeypatch.setattr(teams, "microsoft_graph_request", rec)
    out = _ok(await teams.TeamsProvider().execute(
        "teams__read_chat_messages",
        {"chat_id": "a", "mentions_only": True}, CTX))
    assert [m["id"] for m in out["messages"]] == ["m1"]
    assert out["count"] == 1 and out["mentions_only_applied"] is True

    # Self unresolvable → refuse. Returning everything under a request
    # for "messages that mention me" is the narrowing that lies.
    rec2 = _Recorder({"value": [
        {"id": "a", "chatType": "group",
         "members": [{"userId": "ME"}, {"userId": "A"}]}]})
    monkeypatch.setattr(teams, "microsoft_graph_request", rec2)
    refused = await teams.TeamsProvider().execute(
        "teams__read_chat_messages",
        {"chat_id": "a", "mentions_only": True}, CTX)
    assert isinstance(refused, cbase.ConnectorToolError)
    assert "mention you" in refused.message


@pytest.mark.asyncio
async def test_mentions_only_costs_no_extra_call_when_not_asked(monkeypatch):
    rec = _Recorder({"value": []})
    monkeypatch.setattr(teams, "microsoft_graph_request", rec)
    await teams.TeamsProvider().execute(
        "teams__read_chat_messages", {"chat_id": "a"}, CTX)
    assert len(rec.calls) == 1


@pytest.mark.asyncio
async def test_since_last_read_moves_orderby_with_the_filter(monkeypatch):
    """Graph IGNORES `$filter` on this endpoint unless `$orderby` names
    the same property, so a filter shipped under the default ordering
    would silently return everything."""
    rec = _Recorder(
        {"id": "a", "viewpoint": {"lastMessageReadDateTime": "2026-08-30T09:00:00Z"}},
        {"value": []},
    )
    monkeypatch.setattr(teams, "microsoft_graph_request", rec)
    out = _ok(await teams.TeamsProvider().execute(
        "teams__read_chat_messages",
        {"chat_id": "a", "since_last_read": True}, CTX))
    params = rec.calls[-1][1]["params"]
    assert params["$orderby"] == "lastModifiedDateTime desc"
    assert params["$filter"] == (
        "lastModifiedDateTime gt 2026-08-30T09:00:00Z")
    assert out["since_last_read_applied"] is True


@pytest.mark.asyncio
async def test_a_chat_never_opened_says_so_rather_than_reading_as_empty(
    monkeypatch,
):
    rec = _Recorder({"id": "a", "viewpoint": {}}, {"value": []})
    monkeypatch.setattr(teams, "microsoft_graph_request", rec)
    out = _ok(await teams.TeamsProvider().execute(
        "teams__read_chat_messages",
        {"chat_id": "a", "since_last_read": True}, CTX))
    assert out["since_last_read_applied"] is False
    assert "$filter" not in rec.calls[-1][1]["params"]


# ── Calendar: the three fields §6 and §7 turn on ─────────────────────


def test_the_event_row_answers_owner_agenda_and_response():
    row = cal._event_row({
        "id": "e1", "summary": "Standup",
        "organizer": {"email": "me@x.com", "self": True},
        "description": "  ",
        "attendees": [{"self": True, "responseStatus": "accepted"},
                      {"email": "a@x.com"}],
    })
    assert row["role"] == "organizer"
    assert row["my_response"] == "accepted"
    assert row["agenda"] == ""            # whitespace is not an agenda
    assert row["organizer_email"] == "me@x.com"
    assert row["attendee_count"] == 2

    invited = cal._event_row({
        "id": "e2", "organizer": {"email": "them@x.com"},
        "description": "the plan", "attachments": [{"title": "deck"}],
        "attendees": [{"self": True, "responseStatus": "needsAction"}],
    })
    assert invited["role"] == "attendee"
    assert invited["my_response"] == "needsAction"
    assert invited["agenda"] == "description+attachment"


def test_the_row_carries_the_shape_of_the_agenda_never_its_text():
    """A meeting description is the most sensitive field on the row, and
    "No agenda yet" only ever needs to know whether there is one."""
    row = cal._event_row({"id": "e", "description": "salary review notes"})
    assert row["agenda"] == "description"
    assert "salary" not in json.dumps(row)


def test_role_and_response_are_empty_rather_than_guessed():
    """Every §6 narrowing keeps a row it cannot judge, so the unknown
    value has to be distinguishable from a real one."""
    row = cal._event_row({"id": "e"})
    assert row["role"] == "" and row["my_response"] == ""


@pytest.mark.asyncio
async def test_within_hours_is_a_window_the_provider_resolves(monkeypatch):
    """A poll event's `poll_args` are static, so "the next 24 hours"
    cannot be a timestamp written in a manifest. This is what makes
    `meeting_soon` declarable at all."""
    rec = _Recorder({"items": []})
    monkeypatch.setattr(cal, "google_request", rec)
    await cal.CalendarProvider().execute(
        "calendar__list_events", {"within_hours": 24}, CTX)
    params = rec.calls[-1][1]["params"]
    assert params["timeMin"] < params["timeMax"]
    lo = cal.datetime.fromisoformat(params["timeMin"])
    hi = cal.datetime.fromisoformat(params["timeMax"])
    assert abs((hi - lo).total_seconds() - 24 * 3600) < 5


@pytest.mark.asyncio
async def test_an_explicit_bound_outranks_the_window(monkeypatch):
    rec = _Recorder({"items": []})
    monkeypatch.setattr(cal, "google_request", rec)
    await cal.CalendarProvider().execute(
        "calendar__list_events",
        {"within_hours": 24, "time_min": "2026-01-01T00:00:00+00:00"}, CTX)
    assert rec.calls[-1][1]["params"]["timeMin"] == "2026-01-01T00:00:00+00:00"


@pytest.mark.asyncio
async def test_each_narrowing_changes_which_meetings_come_back(monkeypatch):
    items = [
        {"id": "mine-bare", "organizer": {"self": True}},
        {"id": "mine-agenda", "organizer": {"self": True},
         "description": "plan"},
        {"id": "invite", "organizer": {"email": "t@x"},
         "attendees": [{"self": True, "responseStatus": "needsAction"}]},
        {"id": "declined", "organizer": {"email": "t@x"},
         "attendees": [{"self": True, "responseStatus": "declined"}]},
    ]
    prov = cal.CalendarProvider()

    async def ids(**extra):
        rec = _Recorder({"items": items})
        monkeypatch.setattr(cal, "google_request", rec)
        out = _ok(await prov.execute(
            "calendar__list_events", {"max_results": 50, **extra}, CTX))
        return [e["id"] for e in out["events"]], out, rec

    all_ids, out, _ = await ids()
    assert len(all_ids) == 4 and "narrowed_by" not in out

    mine, out, _ = await ids(organized_by_me=True)
    assert mine == ["mine-bare", "mine-agenda"]
    assert out["narrowed_by"] == ["organized_by_me"]

    bare, _, _ = await ids(organized_by_me=True, without_agenda=True)
    assert bare == ["mine-bare"]

    waiting, _, _ = await ids(awaiting_my_response=True)
    assert waiting == ["invite"]         # declined and owned are not invitations


@pytest.mark.asyncio
async def test_a_narrowed_read_fetches_more_than_it_returns(monkeypatch):
    """The narrowings run over the page that came back, so asking for
    exactly `max_results` would answer "4 of my meetings" with whichever
    of the first 4 on the calendar happened to be mine."""
    rec = _Recorder({"items": []})
    monkeypatch.setattr(cal, "google_request", rec)
    await cal.CalendarProvider().execute(
        "calendar__list_events", {"max_results": 10}, CTX)
    assert rec.calls[-1][1]["params"]["maxResults"] == 10
    await cal.CalendarProvider().execute(
        "calendar__list_events",
        {"max_results": 10, "organized_by_me": True}, CTX)
    assert rec.calls[-1][1]["params"]["maxResults"] == 50


# ── Calendar: the hold itself ────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_hold_notifies_nobody_and_an_invitation_still_does(
    monkeypatch,
):
    """§1.2's hold is a 15-minute block with no attendees. `sendUpdates:
    all` on it asks Google to run the notification path for a guest list
    that does not exist."""
    rec = _Recorder({"id": "e1"})
    monkeypatch.setattr(cal, "google_request", rec)
    args = {"calendar_id": "primary", "summary": "Read your brief",
            "start": "2026-09-01T08:00:00Z", "end": "2026-09-01T08:15:00Z"}
    out = _ok(await cal.CalendarProvider().execute(
        "calendar__create_event", args, CTX))
    assert rec.calls[-1][1]["params"]["sendUpdates"] == "none"
    assert "attendees" not in rec.calls[-1][1]["json_body"]
    assert out["attendee_count"] == 0

    await cal.CalendarProvider().execute(
        "calendar__create_event", {**args, "attendees": ["a@x.com"]}, CTX)
    assert rec.calls[-1][1]["params"]["sendUpdates"] == "all"


@pytest.mark.asyncio
async def test_calendar_id_reaches_the_url_and_cannot_leave_its_segment(
    monkeypatch,
):
    """`calendar_id` is the parameter a grant is pinned to, so it is
    also an LLM-supplied path segment: `quote()`'s default `safe='/'`
    would let one address a different Calendar resource."""
    rec = _Recorder({"items": []})
    monkeypatch.setattr(cal, "google_request", rec)
    await cal.CalendarProvider().execute(
        "calendar__list_events", {"calendar_id": "team@group.calendar.google.com"},
        CTX)
    assert "/calendars/team@group.calendar.google.com/events" in rec.calls[-1][0][1]

    await cal.CalendarProvider().execute(
        "calendar__list_events", {"calendar_id": "../../users/me/settings"}, CTX)
    url = rec.calls[-1][0][1]
    assert "/calendars/..%2F..%2Fusers%2Fme%2Fsettings/events" in url


@pytest.mark.asyncio
async def test_a_hold_is_deleted_from_the_calendar_it_was_made_on(
    monkeypatch,
):
    rec = _Recorder({})
    monkeypatch.setattr(cal, "google_request", rec)
    out = _ok(await cal.CalendarProvider().execute(
        "calendar__delete_event",
        {"calendar_id": "team@group.calendar.google.com", "event_id": "e/1"},
        CTX))
    url = rec.calls[-1][0][1]
    assert "/calendars/team@group.calendar.google.com/events/e%2F1" in url
    assert out["calendar_id"] == "team@group.calendar.google.com"


# ── Notion ───────────────────────────────────────────────────────────


def test_state_is_a_word_because_false_and_unset_are_the_same_value():
    assert notion._state({}) == "active"
    assert notion._state({"archived": True}) == "archived"
    assert notion._state({"in_trash": True, "archived": True}) == "in_trash"


def test_change_key_separates_two_pages_edited_in_the_same_minute():
    """R28's recorded reason for not declaring a page-changed event.
    Notion rounds `last_edited_time` to the minute; the page id is what
    keeps the two apart."""
    at = "2026-08-31T10:00:00.000Z"
    a = notion._page_summary({"id": "p1", "last_edited_time": at})
    b = notion._page_summary({"id": "p2", "last_edited_time": at})
    assert a["change_key"] != b["change_key"]
    later = notion._page_summary(
        {"id": "p1", "last_edited_time": "2026-08-31T10:01:00.000Z"})
    assert later["change_key"] != a["change_key"]
    assert notion._page_summary({"id": "p1"})["change_key"] is None


def test_deadline_key_moves_with_the_date_and_not_with_the_body():
    """"A deadline moves" and "a page changed" have to be different
    signals or the event fires on every edit."""
    def row(due, edited, end=None):
        return notion._page_summary({
            "id": "p1", "last_edited_time": edited,
            "properties": {
                # A `created_time` flattens to an ISO STRING, so a scan
                # of the flattened row would key the "deadline" off a
                # stamp that never moves. The key reads raw types.
                "Created": {"type": "created_time",
                            "created_time": "2026-01-01T00:00:00.000Z"},
                "Due": {"type": "date",
                        "date": {"start": due, "end": end}},
            },
        })

    a = row("2026-09-04", "2026-08-31T10:00:00.000Z")
    assert "2026-09-04" in a["deadline_key"]
    assert "2026-01-01" not in a["deadline_key"]
    edited_only = row("2026-09-04", "2026-08-31T11:00:00.000Z")
    moved = row("2026-09-08", "2026-08-31T11:00:00.000Z")
    end_moved = row("2026-09-04", "2026-08-31T10:00:00.000Z", end="2026-09-05")
    assert edited_only["deadline_key"] == a["deadline_key"]
    assert moved["deadline_key"] != a["deadline_key"]
    assert end_moved["deadline_key"] != a["deadline_key"]
    assert edited_only["change_key"] != a["change_key"]


def test_a_row_with_no_date_never_fires_a_deadline_event():
    """`_extract_events` skips a row whose dedupe key is None — which is
    what a None here buys, rather than every dateless row colliding on
    one empty key."""
    assert notion._page_summary(
        {"id": "p1", "properties": {"Name": {"type": "title", "title": []}}}
    )["deadline_key"] is None


def test_people_are_read_off_the_raw_property_not_the_flattened_one():
    """`_flatten_property` renders people as display NAMES, and a name
    cannot be compared against a user id."""
    row = notion._page_summary({
        "id": "p1",
        "properties": {"Owner": {"type": "people", "people": [
            {"id": "AB-CD", "name": "Dana"}]}},
    })
    assert row["people"] == ["abcd"]


def test_a_composed_filter_stays_one_level_deep():
    assert notion._and_filter(None, []) is None
    one = notion._and_filter(None, [{"a": 1}])
    assert one == {"a": 1}
    assert notion._and_filter({"and": [{"a": 1}]}, [{"b": 2}]) == {
        "and": [{"a": 1}, {"b": 2}]}
    assert notion._and_filter({"a": 1}, [{"b": 2}]) == {
        "and": [{"a": 1}, {"b": 2}]}


@pytest.mark.asyncio
async def test_edited_since_composes_onto_the_callers_own_filter(monkeypatch):
    rec = _Recorder({"results": []})
    monkeypatch.setattr(notion, "_notion_request", rec)
    await notion.NotionProvider()._query_database(
        {"data_source_id": "ds1", "edited_since": "2026-08-30T00:00:00Z",
         "filter": {"property": "Status", "status": {"equals": "Todo"}}},
        "tok")
    body = rec.calls[-1][1]["json_body"]
    assert body["filter"]["and"][0]["property"] == "Status"
    assert body["filter"]["and"][1]["last_edited_time"] == {
        "on_or_after": "2026-08-30T00:00:00Z"}
    # A time-windowed read in table order shows the OLDEST edits first.
    assert body["sorts"] == [{"timestamp": "last_edited_time",
                              "direction": "descending"}]


@pytest.mark.asyncio
async def test_tagged_to_me_ors_over_every_person_column(monkeypatch):
    """Notion's people filter is per COLUMN — there is no "any people
    property" predicate."""
    rec = _Recorder(
        {"bot": {"owner": {"type": "user", "user": {"id": "AB-CD"}}}},
        {"properties": {"Owner": {"type": "people"},
                        "Reviewer": {"type": "people"},
                        "Status": {"type": "status"}}},
        {"results": []},
    )
    monkeypatch.setattr(notion, "_notion_request", rec)
    out = _ok(await notion.NotionProvider()._query_database(
        {"data_source_id": "ds1", "assigned_to_me": True}, "tok"))
    body = rec.calls[-1][1]["json_body"]
    clauses = body["filter"]["or"]
    assert [c["property"] for c in clauses] == ["Owner", "Reviewer"]
    assert all(c["people"] == {"contains": "abcd"} for c in clauses)
    assert out["narrowed_by"] == ["assigned_to_me"]


@pytest.mark.asyncio
async def test_tagged_to_me_refuses_rather_than_returning_the_whole_table(
    monkeypatch,
):
    """The one failure mode a "Tagged to me" chip must not have: a
    request for one person's rows answered with everyone's."""
    rec = _Recorder({"bot": {"owner": {"type": "workspace"}}})
    monkeypatch.setattr(notion, "_notion_request", rec)
    refused = await notion.NotionProvider()._query_database(
        {"data_source_id": "ds1", "assigned_to_me": True}, "tok")
    assert isinstance(refused, cbase.ConnectorToolError)
    assert "tagged to you" in refused.message

    rec2 = _Recorder(
        {"bot": {"owner": {"type": "user", "user": {"id": "AB"}}}},
        {"properties": {"Status": {"type": "status"}}},
    )
    monkeypatch.setattr(notion, "_notion_request", rec2)
    no_column = await notion.NotionProvider()._query_database(
        {"data_source_id": "ds1", "assigned_to_me": True}, "tok")
    assert isinstance(no_column, cbase.ConnectorToolError)
    assert "person column" in no_column.message


@pytest.mark.asyncio
async def test_append_writes_to_the_end_of_an_existing_page(monkeypatch):
    """§1.2 promises "appended under today's date". `create_page` under
    a page parent makes a CHILD page, so a daily brief left the user
    with thirty pages named after thirty days."""
    rec = _Recorder({"results": [{}, {}, {}]})
    monkeypatch.setattr(notion, "_notion_request", rec)
    out = _ok(await notion.NotionProvider().execute(
        "notion__append_blocks",
        {"page_id": "ab-cd", "heading": "31 August",
         "content": "one\ntwo"}, CTX))
    method, path = rec.calls[-1][0][0], rec.calls[-1][0][1]
    assert method == "PATCH" and path.endswith("/children")
    children = rec.calls[-1][1]["json_body"]["children"]
    assert children[0]["type"] == "heading_2"
    assert children[0]["heading_2"]["rich_text"][0]["text"]["content"] == (
        "31 August")
    assert [c["type"] for c in children[1:]] == ["paragraph", "paragraph"]
    assert out["appended"] is True


@pytest.mark.asyncio
async def test_the_heading_is_one_of_notions_hundred_children(monkeypatch):
    """A `children` array over 100 elements is a silent 400, and the
    heading occupies one of them."""
    rec = _Recorder({"results": []})
    monkeypatch.setattr(notion, "_notion_request", rec)
    out = _ok(await notion.NotionProvider().execute(
        "notion__append_blocks",
        {"page_id": "ab", "heading": "h",
         "content": "\n".join(str(i) for i in range(200))}, CTX))
    children = rec.calls[-1][1]["json_body"]["children"]
    assert len(children) == notion._MAX_BLOCKS_PER_REQUEST
    assert out["content_truncated"] is True


@pytest.mark.asyncio
async def test_append_refuses_an_empty_write(monkeypatch):
    rec = _Recorder({"results": []})
    monkeypatch.setattr(notion, "_notion_request", rec)
    for args in ({"page_id": ""}, {"page_id": "ab", "content": "   "}):
        result = await notion.NotionProvider().execute(
            "notion__append_blocks", args, CTX)
        assert isinstance(result, cbase.ConnectorToolError)
    assert not rec.calls


# ── The §6 chips these tools were built for ──────────────────────────
#
# `spec.CONNECTOR_FILTERS` is the integrator's file, so these are the
# entries this package is handing over — pinned here against the REAL
# compiler in `executor_v2` and the REAL rows the providers above
# return. A chip is only honest if the same table that draws it also
# changes the read, and this is where that is proven before the entry
# is pasted anywhere.

TEAMS_FILTERS = (
    {"id": "mentions", "label": "Mentions me",
     "tools": ("teams__read_chat_messages",),
     "compile": ({"kind": "param", "name": "mentions_only", "value": True},)},
    {"id": "no_bots", "label": "Skip bots",
     "tools": ("teams__read_chat_messages",),
     "compile": ({"kind": "drop", "field": "author_type", "when": "contains",
                  "values": ("application", "device")},)},
    {"id": "since_read", "label": "Since my last read",
     "tools": ("teams__read_chat_messages",),
     "compile": ({"kind": "param", "name": "since_last_read",
                  "value": True},)},
)

CALENDAR_FILTERS = (
    {"id": "next24", "label": "Next 24 hours",
     "tools": ("calendar__list_events",),
     "compile": ({"kind": "time_window", "direction": "ahead", "hours": 24,
                  "param": "time_min", "max_param": "time_max",
                  "unit": "iso"},)},
    {"id": "mine", "label": "Meetings I own",
     "tools": ("calendar__list_events",),
     "compile": ({"kind": "param", "name": "organized_by_me",
                  "value": True},)},
    {"id": "no_agenda", "label": "No agenda yet",
     "tools": ("calendar__list_events",),
     "compile": ({"kind": "param", "name": "without_agenda", "value": True},)},
    {"id": "no_declined", "label": "Skip declined",
     "tools": ("calendar__list_events",),
     "compile": ({"kind": "drop", "field": "my_response", "when": "contains",
                  "values": ("declined",)},)},
)

NOTION_FILTERS = (
    {"id": "day", "label": "Changed since yesterday",
     "tools": ("notion__search", "notion__query_database"),
     "compile": ({"kind": "drop", "field": "last_edited_time",
                  "when": "older_than", "hours": 24,
                  "tools": ("notion__search",)},
                 {"kind": "time_window", "direction": "back", "hours": 24,
                  "param": "edited_since", "unit": "iso",
                  "tools": ("notion__query_database",)})},
    {"id": "mine", "label": "Tagged to me",
     "tools": ("notion__query_database",),
     "compile": ({"kind": "param", "name": "assigned_to_me", "value": True},)},
    {"id": "no_archived", "label": "Skip archived",
     "tools": ("notion__search", "notion__query_database"),
     "compile": ({"kind": "drop", "field": "state", "when": "contains",
                  "values": ("archived", "in_trash")},)},
)

_PROPOSED = {"teams": TEAMS_FILTERS, "calendar": CALENDAR_FILTERS,
             "notion": NOTION_FILTERS}


@pytest.fixture
def filters(monkeypatch):
    """The proposed table, merged over whatever `spec` already carries.

    Merged rather than replaced so that the day the integrator pastes
    these in, this test starts asserting the SHIPPED entry — and fails
    if it drifts from the one this package proved.
    """
    from app.agent.automations import spec

    merged = dict(spec.CONNECTOR_FILTERS)
    for cid, proposed in _PROPOSED.items():
        live = {f["id"]: f for f in merged.get(cid, ())}
        for f in proposed:
            if f["id"] in live:
                assert live[f["id"]] == f, (
                    f"{cid}.{f['id']} in spec.CONNECTOR_FILTERS differs from "
                    f"the entry this package proved"
                )
        merged[cid] = tuple({**live, **{f["id"]: f for f in proposed}}.values())
    monkeypatch.setattr(spec, "CONNECTOR_FILTERS", merged)
    return merged


def _params(cid, tool, params, on, now=None):
    from app.agent.automations import executor_v2 as ex
    return ex._apply_read_filters(cid, tool, params, on, {"now": now})


def _rows(cid, tool, items, on, path="messages"):
    from app.agent.automations import executor_v2 as ex
    out = ex._apply_read_drops(
        cid, tool, {path: items}, on, {"items_path": path}, {})
    return out[path]


def test_every_proposed_filter_uses_a_declared_compile_kind(filters):
    from app.agent.automations import spec
    for cid, entries_ in _PROPOSED.items():
        for f in entries_:
            for m in f["compile"]:
                assert m["kind"] in spec.FILTER_COMPILE_KINDS, (cid, f["id"])


def test_teams_chips_compile_onto_the_real_parameters(filters):
    p = _params("teams", "teams__read_chat_messages", {"chat_id": "a"},
                ["mentions", "since_read"])
    assert p["mentions_only"] is True and p["since_last_read"] is True
    # Both are real parameters of the real tool, not invented keys.
    assert _params("teams", "teams__read_chat_messages",
                   {"chat_id": "a"}, []) == {"chat_id": "a"}


def test_skip_bots_drops_the_bot_and_keeps_the_system_message(filters):
    rows = [teams._message_row(m, None) for m in (
        {"id": "1", "from": {"user": {"id": "D"}}},
        {"id": "2", "from": {"application": {"id": "B"}}},
        {"id": "3", "from": None},
    )]
    kept = _rows("teams", "teams__read_chat_messages", rows, ["no_bots"])
    assert [r["id"] for r in kept] == ["1", "3"]


def test_calendar_chips_compile_onto_the_real_parameters(filters):
    p = _params("calendar", "calendar__list_events", {},
                ["mine", "no_agenda"])
    assert p["organized_by_me"] is True and p["without_agenda"] is True


def test_skip_declined_reads_the_field_the_row_now_carries(filters):
    rows = [
        cal._event_row({"id": "yes", "attendees": [
            {"self": True, "responseStatus": "accepted"}]}),
        cal._event_row({"id": "no", "attendees": [
            {"self": True, "responseStatus": "declined"}]}),
        cal._event_row({"id": "unknown"}),
    ]
    kept = _rows("calendar", "calendar__list_events", rows, ["no_declined"],
                 path="events")
    # The unjudgeable row STAYS — every drop in this vocabulary fails
    # safe, and an event the user is not an attendee of is not declined.
    assert [r["id"] for r in kept] == ["yes", "unknown"]


def test_notion_day_compiles_differently_per_tool(filters):
    """One chip, one meaning, two substrates: search cannot narrow on
    time at all, so it drops afterwards; a data source query can, so it
    narrows the request."""
    from datetime import datetime, timedelta, timezone
    now = datetime(2026, 8, 31, 12, 0, tzinfo=timezone.utc)
    q = _params("notion", "notion__query_database", {}, ["day"], now=now)
    assert q["edited_since"] == "2026-08-30T12:00:00+00:00"
    assert _params("notion", "notion__search", {}, ["day"], now=now) == {}

    fresh = notion._page_summary({
        "id": "a", "last_edited_time":
        (now - timedelta(hours=2)).isoformat()})
    stale = notion._page_summary({
        "id": "b", "last_edited_time":
        (now - timedelta(hours=48)).isoformat()})
    from app.agent.automations import executor_v2 as ex
    kept = ex._apply_read_drops(
        "notion", "notion__search", {"results": [fresh, stale]},
        ["day"], {"items_path": "results"}, {"now": now})["results"]
    assert [r["id"] for r in kept] == ["a"]


def test_skip_archived_reads_the_state_word(filters):
    rows = [notion._page_summary(o) for o in (
        {"id": "live"},
        {"id": "old", "archived": True},
        {"id": "gone", "in_trash": True},
    )]
    kept = _rows("notion", "notion__search", rows, ["no_archived"],
                 path="results")
    assert [r["id"] for r in kept] == ["live"]


def test_tagged_to_me_compiles_only_where_it_can_be_honoured(filters):
    """§0.2: a picker that writes nowhere is forbidden. Notion's search
    has no person predicate, so the chip declares only the tool that
    does — and `available_filters`' existing gate then hides it unless
    the automation runs such a step."""
    from app.agent.automations import spec
    assert spec.filter_tools("notion", "mine") == ("notion__query_database",)
    assert _params("notion", "notion__search", {}, ["mine"]) == {}
    assert _params("notion", "notion__query_database", {}, ["mine"]) == {
        "assigned_to_me": True}
