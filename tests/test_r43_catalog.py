"""R43 §1 / §6 / §7 — the catalogue, the filter grammar, the events.

Three tables and one rule between them: a choice the app can offer must
be a choice the platform can honour. So this file is as much about what
is NOT declared as about what is. Both directions are pinned:

  §1  every channel, format and cadence id round-trips, in the frozen
      order, and the four canvas labels compose from them.
  §6  every filter offered carries a `compile` the applier can act on,
      in the closed vocabulary, scoped to tools the filter declares —
      and the chips the design asks for that no provider can express
      are recorded as an omission rather than shipped as a lit chip
      that narrows nothing.
  §7  every event any manifest declares can actually BIND: a poll event
      names one of its own connector's tools, says where the items are,
      and asks only for params the canvas pin vocabulary can fill. An
      event that fails that last test is the picker-that-writes-nowhere
      with an extra step.

The registry is the REAL one, loaded from the shipped manifests, for
the reason R42's file gives: a fixture would let this claim
capabilities the platform does not have.
"""

import pytest

from app.agent.automations import catalog
from app.agent.automations.spec import (
    CONNECTOR_FILTERS, EVENT_LABELS, FILTER_COMPILE_KINDS, event_label,
    filter_compile, filter_options,
)
from app.agent.automations.spec import SpecError, validate_spec
from app.agent.automations.spec_v2 import (
    MAX_ACCOUNT_SOURCES, MAX_SOURCES, MAX_STEPS, validate_account_sources,
    validate_delivery,
)
from app.agent.automations.workflow import _EVENT_PARAM_PINS
from app.services.connector_registry import ConnectorRegistry


def _real_registry() -> dict:
    reg = ConnectorRegistry()
    reg.load_all()
    return {e["connector_id"]: e for e in reg.automation_registry()}


REGISTRY = _real_registry()


# ── §1 the catalogue ─────────────────────────────────────────────────


def test_channel_ids_are_the_contract_list_in_order():
    assert catalog.CHANNEL_IDS == (
        "app", "whatsapp", "telegram", "slack_dm", "teams_chat",
        "gmail_draft", "outlook_mail", "notion_page", "calendar_hold",
    )
    assert catalog.FORMAT_IDS == ("lines", "ranked", "pdf", "markdown", "csv")
    # §1.4 named three. `hourly` "Batched hourly" is NOT offered: there
    # is no batcher, so the picker delivered WITH the run and logged a
    # warning — a control that does not do what its label says, which
    # §0.2 forbids as plainly as a chip that narrows nothing. Removing
    # it HERE is what makes it unreachable: this tuple is the closed
    # table `validate_delivery` and `PUT /delivery` both check against.
    assert catalog.CADENCE_IDS == ("run", "instant")
    assert not catalog.is_cadence("hourly")


def test_every_id_round_trips_through_its_lookup():
    for entry in catalog.CHANNELS:
        assert catalog.is_channel(entry["id"])
        assert catalog.channel(entry["id"]) is entry
        for key in ("name", "meta", "land"):
            assert entry[key], f"{entry['id']}.{key}"
    for entry in catalog.FORMATS:
        assert catalog.is_format(entry["id"])
        assert catalog.format_(entry["id"]) is entry
        for key in ("name", "meta", "rail", "noun"):
            assert entry[key], f"{entry['id']}.{key}"
    for entry in catalog.CADENCES:
        assert catalog.is_cadence(entry["id"])
        assert catalog.cadence(entry["id"]) is entry
    assert not catalog.is_channel("sms")
    assert catalog.channel("sms") is None


def test_only_whatsapp_and_telegram_need_a_link():
    linked = {c["id"] for c in catalog.CHANNELS if c["needs_link"]}
    assert linked == {"whatsapp", "telegram"}
    # A channel needs an account or a link, never both: the app draws
    # one affordance for the unavailable case.
    for c in catalog.CHANNELS:
        assert not (c["needs_link"] and c["connector_id"])


def test_default_delivery_needs_no_account_and_no_link():
    assert catalog.DEFAULT_DELIVERY == {
        "channels": ["app"], "format": "ranked", "cadence": "run",
    }
    app = catalog.channel("app")
    assert app["connector_id"] is None and app["needs_link"] is None


def test_labels_compose_for_one_two_and_five_channels():
    one = ["app"]
    two = ["app", "slack_dm"]
    five = ["calendar_hold", "app", "gmail_draft", "slack_dm", "telegram"]

    assert catalog.node_label(one) == "This app"
    assert catalog.node_label(two) == "This app +1"
    assert catalog.node_label(five) == "This app +4"

    assert catalog.node_sub("ranked") == "as a ranked list"
    assert catalog.node_sub("lines") == "as five short lines"
    assert catalog.rail("ranked") == "1 list"
    assert catalog.rail("pdf") == "1 PDF"

    assert catalog.done_label(one, "ranked") == "Done — This app · a ranked list"
    assert catalog.done_label(two, "ranked") == (
        "Done — This app + Slack DM · a ranked list"
    )
    # Every channel is named in the footer, not just the first: it is
    # the last thing read before the sheet closes.
    assert catalog.done_label(five, "pdf") == (
        "Done — This app + Telegram + Slack DM + Gmail draft + "
        "Calendar hold · a one-page PDF"
    )


def test_picked_channels_are_ordered_by_the_catalogue_not_the_caller():
    assert catalog.order_channels(["slack_dm", "app"]) == ["app", "slack_dm"]
    assert catalog.order_channels(["app", "app"]) == ["app"]
    assert catalog.order_channels(["nope"]) == []
    # The canvas chip row is four wide; the label still counts all five.
    five = ["app", "whatsapp", "telegram", "slack_dm", "teams_chat"]
    assert catalog.node_chips(five) == five[:4]
    assert catalog.node_label(five) == "This app +4"


def test_no_channel_picked_still_reads_as_a_sentence():
    assert catalog.node_label([]) == "Nowhere yet"
    assert catalog.done_label([], "ranked") == "Done — nowhere yet · a ranked list"


def test_an_unknown_format_falls_back_rather_than_raising():
    assert catalog.rail("gif") == catalog.rail("ranked")
    assert catalog.node_sub("gif") == catalog.node_sub("ranked")


# ── §2 the spec keys the catalogue gates ─────────────────────────────


def test_delivery_validates_against_the_catalogue():
    errors: list = []
    out = validate_delivery(
        {"delivery": {"channels": ["slack_dm", "app"], "format": "pdf",
                      "cadence": "instant"}}, errors,
    )
    assert not errors
    assert out == {"channels": ["app", "slack_dm"], "format": "pdf",
                   "cadence": "instant"}

    errors = []
    validate_delivery({"delivery": {"channels": ["sms"], "format": "gif",
                                    "cadence": "daily"}}, errors)
    assert {e["code"] for e in errors} == {
        "unknown_channel", "unknown_format", "unknown_cadence",
    }

    # The removed cadence is refused at the WRITE, not honoured at the
    # read: an automation can no longer be left holding a promise the
    # engine has no batcher for.
    errors = []
    validate_delivery({"delivery": {"cadence": "hourly"}}, errors)
    assert [e["code"] for e in errors] == ["unknown_cadence"]


def test_delivery_is_partial_so_a_single_key_write_re_validates():
    errors: list = []
    assert validate_delivery({"delivery": {"format": "csv"}}, errors) == {
        "format": "csv",
    }
    assert not errors
    assert validate_delivery({}, errors) == {}


def test_account_sources_are_shape_only_and_capped():
    errors: list = []
    out = validate_account_sources(
        {"sources": {"gmail": ["inbox", "label:Clients", "inbox"]}}, errors,
    )
    assert not errors
    # Deduped, and in the ACCOUNT's order — there is no table to sort
    # a live enumeration against.
    assert out == {"gmail": ["inbox", "label:Clients"]}

    errors = []
    validate_account_sources(
        {"sources": {"slack": [f"c{i}" for i in range(MAX_ACCOUNT_SOURCES + 1)]}},
        errors,
    )
    assert [e["code"] for e in errors] == ["too_many_sources_picked"]


def test_the_caps_are_the_ones_the_round_raised():
    assert (MAX_SOURCES, MAX_STEPS) == (12, 12)


# ── §6 filters ───────────────────────────────────────────────────────

#: The chips R43 §6 asks for, id → the spec's exact label, per
#: connector. Everything offered must match this text; everything not
#: offered must be in `_FILTERS_NOT_OFFERED` with a reason.
_SPEC_FILTER_LABELS = {
    "gmail": {"me": "Addressed to me", "unread": "Unread only",
              "no_promos": "Skip newsletters", "day": "Last 24 hours"},
    "outlook": {"me": "Addressed to me", "unread": "Unread only",
                "no_auto": "Skip automated mail", "day": "Last 24 hours"},
    "slack": {"mentions": "Mentions me", "mine": "Threads I am in",
              "no_bots": "Skip bots", "since_read": "Since my last read",
              "day": "Since yesterday"},
    "teams": {"mentions": "Mentions me", "no_bots": "Skip bots",
              "since_read": "Since my last read"},
    "jira": {"due_week": "Due this week", "priority": "P1 and P2 only",
             "day": "Changed since yesterday", "open": "Skip closed"},
    "github": {"failing": "Failing checks first", "no_drafts": "Skip drafts",
               "day": "Changed since yesterday"},
    "notion": {"day": "Changed since yesterday", "mine": "Tagged to me",
               "no_archived": "Skip archived"},
    "calendar": {"next24": "Next 24 hours", "mine": "Meetings I own",
                 "no_agenda": "No agenda yet", "no_declined": "Skip declined"},
}

#: Not offered, and why — each reason would be a fact about the
#: provider's tool surface. R43's wave three gave every one of §6's 30
#: chips a tool that can express it, so this set is EMPTY and the design
#: and the platform agree chip for chip.
#:
#: It is kept, empty, rather than deleted: it is the mechanism by which
#: an omission has to be written down. A chip that stops being
#: expressible has one honest home — here, with its reason — and
#: deleting the set would make quietly dropping one the path of least
#: resistance. `test_every_offered_filter_uses_the_specs_exact_label`
#: fails until an absent chip is listed here.
_FILTERS_NOT_OFFERED: set = set()


def test_every_offered_filter_uses_the_specs_exact_label():
    for connector_id, wanted in _SPEC_FILTER_LABELS.items():
        offered = {f["id"]: f["label"] for f in filter_options(connector_id)}
        for fid, label in wanted.items():
            if (connector_id, fid) in _FILTERS_NOT_OFFERED:
                assert fid not in offered, f"{connector_id}.{fid} is offered"
                continue
            assert offered.get(fid) == label, f"{connector_id}.{fid}"


def test_the_table_offers_nothing_the_design_did_not_ask_for():
    for connector_id, entries in CONNECTOR_FILTERS.items():
        wanted = _SPEC_FILTER_LABELS.get(connector_id, {})
        for f in entries:
            assert f["id"] in wanted, f"{connector_id}.{f['id']} is not in §6"


def test_every_offered_filter_compiles_into_something():
    for connector_id, entries in CONNECTOR_FILTERS.items():
        for f in entries:
            tools = set(f["tools"])
            assert tools, f"{connector_id}.{f['id']} composes into no tool"
            mutations = f.get("compile") or ()
            assert mutations, (
                f"{connector_id}.{f['id']} is a chip that narrows nothing"
            )
            for m in mutations:
                assert m["kind"] in FILTER_COMPILE_KINDS, m
                # A mutation may pin itself to a subset of the filter's
                # tools; it may never name one the filter does not
                # compose into, or it would be applied to a call that
                # never sees it.
                assert set(m.get("tools") or ()) <= tools, m


def test_a_filter_compiles_differently_per_tool_where_the_units_differ():
    read = filter_compile("slack", "day", "slack__read_messages")
    search = filter_compile("slack", "day", "slack__search_messages")
    assert [m["unit"] for m in read] == ["unix"]
    assert [m["unit"] for m in search] == ["slack_after"]
    # Unscoped, the caller gets both and must scope it itself.
    assert len(filter_compile("slack", "day")) == 2
    # Gmail's newsletters chip is two terms, both on every tool.
    assert len(filter_compile("gmail", "no_promos",
                              "gmail__list_messages")) == 2
    assert filter_compile("gmail", "nope") == ()


# ── §7 instant triggers ──────────────────────────────────────────────

#: Every event id R43 §7 names, and whether the shipped tool surface can
#: bind it. False is not a TODO marker — it is the recorded finding that
#: no tool on that connector can produce the event, so the picker must
#: not offer it. Flipping one to True means a manifest declares it and
#: every assertion below applies.
_R43_EVENTS = {
    ("gmail", "email_received"): True,
    # ── the five §7 names no manifest can honestly declare ───────────
    # Gmail's four all ride the ONE `users.watch` push feed, so each
    # compiles to its own Trigger row on the same push and the
    # narrowing has to be the row's `filter_json`. R43 landed that path
    # end to end — `AutomationEventSpec.default_filter`,
    # `automation_registry()` serialising it, `set_triggers` writing it
    # as the source's `filter`, `spec_v2` carrying it as `filter_rules`,
    # `compiler._compile_bindings_v2` copying it into
    # `Trigger.filter_json` — but the gmail manifest declares no
    # `default_filter` on any event yet, so nothing narrows and these
    # three would each fire on every message. Flip one the moment the
    # manifest carries its filter, never before.
    ("gmail", "invoice_landed"): False,     # needs a static
    #   `subject_contains` in the manifest; the key exists, the value
    #   does not.
    ("gmail", "vip_wrote"): False,          # needs one hook further:
    #   its `from_contains` comes from the user's own `person` pins, not
    #   from the manifest, so `set_triggers` needs a `filter_from_pins`
    #   mapping to fill from `focus_of(raw)[connector_id]`.
    ("gmail", "pinned_thread_reply"): False,  # not declarable at all:
    #   the filter vocabulary has no thread predicate and the Pub/Sub
    #   item carries no thread id, so the row could only ever mean
    #   `email_received` under a second name.
    ("outlook", "email_received"): True,
    ("outlook", "vendor_moved"): False,     # needs a KQL query naming
    #   the vendors, and no pin kind in `_EVENT_PARAM_PINS` can fill a
    #   free-text query — the row would refuse forever.
    ("outlook", "meeting_request"): True,
    ("slack", "mentioned"): True,
    ("slack", "dm_arrived"): True,
    ("slack", "thread_moved"): True,
    ("slack", "oncall_message"): True,
    ("teams", "mentioned"): True,
    ("teams", "chat_message_received"): True,
    ("jira", "issue_assigned"): True,
    ("jira", "issue_reopened"): True,
    ("jira", "p1_raised"): True,
    ("jira", "due_moved"): False,           # JQL's CHANGED operator
    #   covers status, assignee, priority, reporter, resolution and
    #   fixVersion only — no query watches a due date — and a poll on
    #   `duedate` would need a dedupe key that is a DATE, shared by
    #   every ticket due that day.
    ("github", "review_requested"): True,
    ("github", "pr_commented"): True,
    ("github", "build_red"): True,
    ("github", "pr_approved"): True,
    ("notion", "page_changed"): True,
    ("notion", "deadline_moved"): True,
    ("calendar", "meeting_soon"): True,
    ("calendar", "no_agenda"): True,
    ("calendar", "invitation_arrived"): True,
}


def _declared(connector_id: str) -> dict:
    entry = REGISTRY.get(connector_id) or {}
    return {e["key"]: e for e in entry.get("events") or []}


@pytest.mark.parametrize("pair,expected", sorted(_R43_EVENTS.items()))
def test_r43_event_is_declared_exactly_where_it_can_bind(pair, expected):
    connector_id, key = pair
    assert (key in _declared(connector_id)) is expected


def test_every_r43_event_has_the_wording_the_round_agreed():
    for connector_id, key in _R43_EVENTS:
        assert EVENT_LABELS.get(f"{connector_id}.{key}"), f"{connector_id}.{key}"
        assert event_label(connector_id, key) == (
            EVENT_LABELS[f"{connector_id}.{key}"]
        )


def test_slack_declares_the_rounds_four_and_keeps_the_pinned_one():
    # `channel_message` is the row that names the place the user pinned,
    # and it is still the only one that needs a pin. The other four are
    # account-wide: each resolves the owner's identity at CALL time
    # rather than asking an automation author to hard-code a handle.
    events = _declared("slack")
    assert set(events) == {
        "channel_message", "mentioned", "dm_arrived", "thread_moved",
        "oncall_message",
    }
    assert REGISTRY["slack"]["poll"] is True
    assert events["channel_message"]["params_required"] == ["channel"]
    for key in ("mentioned", "dm_arrived", "thread_moved", "oncall_message"):
        assert events[key]["params_required"] == [], key
    # `oncall_message` is literal: the row says "#oncall" and it means it.
    assert events["oncall_message"]["poll_args"]["channel"] == "#oncall"


def test_every_declared_event_can_actually_be_bound():
    for connector_id, entry in REGISTRY.items():
        for ev in entry.get("events") or []:
            where = f"{connector_id}.{ev['key']}"
            assert ev["description"].strip(), where
            assert ev["dedupe_field"], where
            if entry["push"]:
                # A push event rides the Trigger pipeline; it needs no
                # read tool and the compiler asks it for none.
                continue
            assert entry["poll"], where
            assert ev["source_tool"], f"{where} polls with no tool"
            assert ev["items_path"], f"{where} says nothing about where"
            fields = ev["fields"] or {}
            assert ev["dedupe_field"] in fields, (
                f"{where} dedupes on a field it does not collect"
            )


def test_every_required_event_param_can_be_filled_by_a_pin():
    """`set_triggers` fills `params_required` from the canvas pins, and
    refuses with "pick the place first" when it cannot. A param with no
    entry in `_EVENT_PARAM_PINS` therefore refuses FOREVER — the row is
    offered, tapping it never works, and nothing says why."""
    for connector_id, entry in REGISTRY.items():
        for ev in entry.get("events") or []:
            for param in ev.get("params_required") or []:
                assert param in _EVENT_PARAM_PINS, (
                    f"{connector_id}.{ev['key']} requires {param!r}, which "
                    f"no pin kind can fill"
                )


# ── §8 the per-connector ping override ───────────────────────────────


def _spec(**extra) -> dict:
    return {
        "version": 2,
        "name": "Morning work brief",
        "mode": "auto",
        "trigger": {"sources": [
            {"id": "sched", "mode": "schedule",
             "schedule": {"cron_local": "0 8 * * 1-5"}},
            {"id": "jira_lane", "mode": "poll", "connector_id": "jira",
             "event": "issue_assigned", "dedupe_key": "event.key",
             **extra},
        ]},
        "steps": [{"id": "read", "connector_id": "jira",
                   "tool": "jira__search_issues",
                   "params": {"jql": "assignee = currentUser()"}}],
    }


def test_a_ping_override_survives_canonicalisation():
    """The canonical form re-emits each source key by key, so a new one
    that is not listed there is accepted on the way in and gone on the
    way out — the shape of R39's dropped pin fields."""
    v = validate_spec(_spec(ping_channel="slack_dm", ping_format="lines"),
                      REGISTRY, template_mode=True)
    lane = v.source_by_id("jira_lane")
    assert (lane.ping_channel, lane.ping_format) == ("slack_dm", "lines")
    stored = v.raw["trigger"]["sources"][1]
    assert stored["ping_channel"] == "slack_dm"
    assert stored["ping_format"] == "lines"


def test_clearing_a_ping_is_indistinguishable_from_never_setting_one():
    v = validate_spec(_spec(ping_channel=None, ping_format=None),
                      REGISTRY, template_mode=True)
    lane = v.source_by_id("jira_lane")
    assert (lane.ping_channel, lane.ping_format) == (None, None)
    assert "ping_channel" not in v.raw["trigger"]["sources"][1]


def test_a_ping_is_checked_against_the_same_catalogue_delivery_is():
    with pytest.raises(SpecError) as e:
        validate_spec(_spec(ping_channel="sms", ping_format="gif"),
                      REGISTRY, template_mode=True)
    assert {err["code"] for err in e.value.errors} == {
        "unknown_channel", "unknown_format",
    }


def test_delivery_and_sources_ride_the_spec_and_come_back_out():
    raw = _spec()
    raw["delivery"] = {"channels": ["slack_dm", "app"], "format": "pdf",
                       "cadence": "run"}
    raw["sources"] = {"jira": ["assigned", "sprint"]}
    v = validate_spec(raw, REGISTRY, template_mode=True)
    assert v.delivery == {"channels": ["app", "slack_dm"], "format": "pdf",
                          "cadence": "run"}
    # NOT `v.sources` — that is already the firing lanes, and the two
    # are unrelated things with one name on the wire.
    assert v.account_sources == {"jira": ["assigned", "sprint"]}
    assert len(v.sources) == 2
