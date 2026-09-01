"""AutomationSpec — the canonical spec shape and its validator.

Hand-rolled validation, deliberately: the repo's precedent is that a
security boundary must not rest on the transitively-present `jsonschema`
package (`connector_pending_actions.py` documents the call). Every
rejection the round brief names is a distinct, tested error code:

    write_without_grant       mutating action with no grant ref
    grant_target_mismatch     grant pins a different target (arm-time)
    unknown_tool              action tool not in the connector registry
    unknown_event             trigger event not declared by the connector
    missing_dedupe_key        push/poll spec without a dedupe key
    interval_below_floor      poll interval under the connector/global floor

`validate_spec` is pure — it takes the spec dict plus the capability
registry snapshot and returns a `ValidatedSpec` or raises `SpecError`
carrying every problem at once (a setup agent that gets one error per
round-trip burns a turn per field).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from app.db.models.automation import (
    AUTOMATION_POLL_FLOOR_S, AUTOMATION_TRIGGER_MODES,
)


class SpecError(ValueError):
    """Validation failed. `errors` is a list of {code, field, message}."""

    def __init__(self, errors: list[dict]):
        self.errors = errors
        super().__init__("; ".join(
            f"{e['code']}: {e['message']}" for e in errors
        ))


# ── Dev fast-lane (Round 28) ─────────────────────────────────────────
#
# An env-gated override for dev/e2e tenants that lowers the poll and
# every_s floors to seconds so a full trigger→fire loop is watchable in
# one sitting. Two-sided refusal, same as the e2e metering marker: the
# flag alone is not enough — a production ENVIRONMENT ignores it, so a
# stray env var on a prod tenant changes nothing. The manifest load
# lint and the platform registry keep the honest production floors
# either way; this bends only spec validation/compile on this tenant.

AUTOMATION_DEV_POLL_FLOOR_S = 5


def dev_fast_lane_active() -> bool:
    from app.config import settings
    return (
        bool(getattr(settings, "automations_dev_fast_lane", False))
        and (settings.environment or "").strip().lower() != "production"
    )


def effective_poll_floor(cap_floor: Any) -> int:
    """The poll floor for spec validation: the connector's declared
    floor, never below the global rail — unless the dev fast-lane is
    active, in which case seconds are allowed."""
    if dev_fast_lane_active():
        return AUTOMATION_DEV_POLL_FLOOR_S
    try:
        declared = int(cap_floor or AUTOMATION_POLL_FLOOR_S)
    except (TypeError, ValueError):
        declared = AUTOMATION_POLL_FLOOR_S
    return max(declared, AUTOMATION_POLL_FLOOR_S)


def effective_every_floor() -> int:
    """Floor for schedule {every_s}."""
    return AUTOMATION_DEV_POLL_FLOOR_S if dev_fast_lane_active() else 60


@dataclass(frozen=True)
class ValidatedSpec:
    """The parsed, validated spec. `raw` is the canonical dict to
    persist (unknown keys already rejected, defaults filled)."""

    raw: dict
    name: str
    mode: str                       # "auto" | "confirm"
    trigger_mode: str               # push | poll | schedule
    trigger_connector_id: Optional[str]
    trigger_event: Optional[str]
    trigger_params: dict
    poll_interval_s: Optional[int]
    schedule: Optional[dict]
    filter_rules: dict
    action_connector_id: str
    action_tool: str
    action_params_template: dict
    grant_id: Optional[str]
    action_mutates: bool
    dedupe_key_field: Optional[str]
    event_spec: Optional[dict] = field(default=None)


def _err(errors: list, code: str, fld: str, message: str) -> None:
    errors.append({"code": code, "field": fld, "message": message})


_TOP_KEYS = {"name", "description", "trigger", "action", "dedupe_key", "mode",
             "version", "focus", "filters"}
_TRIGGER_KEYS = {
    "mode", "connector_id", "event", "params", "poll_interval_s",
    "schedule", "filter",
}
# `grant_target` is SYSTEM-written (the arm step snapshots the approved
# grant's pinned target for template rendering) — accepted on
# re-validation, never authored by the user or the model.
_ACTION_KEYS = {"connector_id", "tool", "params_template", "grant_id",
                "grant_target"}
_SCHEDULE_KEYS = {"cron_local", "at", "every_s"}


# ── Sub-node focus (R38) ─────────────────────────────────────────────
#
# `focus` is where the automation STARTS, per account: the channel, the
# person, the ticket or the thread the user pinned under that connector
# on the canvas. It is not a filter and it is not a permission — a pin
# narrows nothing the grant already allows and widens nothing it does
# not. It is the answer to "when this runs, look HERE first", and it
# has to live in the spec rather than in a side table because three
# readers need the same answer: the workflow payload the canvas draws,
# the run context the steps render against, and the grounding the
# thread agent answers from. A side table would be a fourth truth.
#
#   focus: {"slack": [{"kind": "channel", "id": "C123",
#                      "label": "#eng"}]}
#
# Shape only is enforced here. MEMBERSHIP (is this connector actually
# in the spec?) is the workflow writer's gate, deliberately: a pin
# whose account is removed must not make the whole spec unparseable at
# run time — `remove_connector` drops the pins with the account, and a
# stale pin that survives some other path is inert rather than fatal.

FOCUS_KINDS = frozenset({
    "channel", "thread", "person", "ticket", "project",
    "repo", "label", "folder", "board", "doc",
})
MAX_FOCUS_PER_ACCOUNT = 10
FOCUS_ID_MAX = 200
FOCUS_LABEL_MAX = 120
#: R39 — a pin can carry the user's own instruction for that place
#: ("boss — anything from her outranks the rest"). It rides the pin
#: into the agent step's context and the run's grounding.
FOCUS_NOTE_MAX = 280

_FOCUS_KEYS = {"kind", "id", "label", "note"}


def validate_focus(spec: dict, errors: list) -> dict:
    """`spec["focus"]` → the canonical `{connector_id: [pin]}` map.

    Shared by the v1 and v2 validators so the two grammars cannot
    drift. Every problem is reported; the return value carries only the
    pins that survived, so a caller that ignores `errors` still never
    persists a malformed pin.
    """
    focus = spec.get("focus")
    if focus is None:
        return {}
    if not isinstance(focus, dict):
        _err(errors, "bad_focus", "focus",
             "focus must map a connector id to a list of pins")
        return {}
    out: dict[str, list] = {}
    for cid, pins in focus.items():
        fld = f"focus.{cid}"
        if not isinstance(cid, str) or not cid.strip():
            _err(errors, "bad_focus", "focus",
                 "focus keys must be connector ids")
            continue
        if not isinstance(pins, list):
            _err(errors, "bad_focus", fld, "each account's focus is a list")
            continue
        if len(pins) > MAX_FOCUS_PER_ACCOUNT:
            _err(errors, "too_many_focus", fld,
                 f"at most {MAX_FOCUS_PER_ACCOUNT} pins per account")
            pins = pins[:MAX_FOCUS_PER_ACCOUNT]
        kept: list = []
        seen: set = set()
        for i, pin in enumerate(pins):
            pfld = f"{fld}[{i}]"
            if not isinstance(pin, dict):
                _err(errors, "bad_focus", pfld, "each pin must be an object")
                continue
            unknown = set(pin) - _FOCUS_KEYS
            if unknown:
                _err(errors, "unknown_field", pfld,
                     f"unknown focus fields {sorted(unknown)}")
                continue
            kind = pin.get("kind")
            if kind not in FOCUS_KINDS:
                _err(errors, "bad_focus_kind", f"{pfld}.kind",
                     f"kind must be one of {sorted(FOCUS_KINDS)}")
                continue
            pid = pin.get("id")
            if not isinstance(pid, str) or not (1 <= len(pid.strip())
                                                <= FOCUS_ID_MAX):
                _err(errors, "bad_focus_id", f"{pfld}.id",
                     f"id must be 1-{FOCUS_ID_MAX} characters")
                continue
            pid = pid.strip()
            label = pin.get("label")
            if label is not None and not isinstance(label, str):
                _err(errors, "bad_focus_label", f"{pfld}.label",
                     "label must be a string")
                continue
            label = (label or "").strip()[:FOCUS_LABEL_MAX] or pid
            if (kind, pid) in seen:
                # Not an error: the same place pinned twice is one pin.
                continue
            seen.add((kind, pid))
            note = pin.get("note")
            if note is not None and not isinstance(note, str):
                _err(errors, "bad_focus_note", f"{pfld}.note",
                     "note must be a string")
                note = ""
            note = (note or "").strip()[:FOCUS_NOTE_MAX]
            entry = {"kind": kind, "id": pid, "label": label}
            if note:
                entry["note"] = note
            kept.append(entry)
        if kept:
            out[cid] = kept
    return out


def focus_render_ctx(focus: dict) -> dict:
    """The `{{focus.…}}` root the run context renders against.

    Flat leaves only, because `render_value` interpolates a leaf and
    `str()`s anything else — `{{focus.slack}}` on a list would put a
    Python repr into a connector's params. `ids`/`labels` are the
    joined forms a params template actually wants; `first` is the
    single-target case (a pinned channel to read, a pinned ticket to
    comment on), which is the overwhelming majority of pins.
    """
    out: dict = {}
    for cid, pins in (focus or {}).items():
        pins = [p for p in (pins or []) if isinstance(p, dict)]
        if not pins:
            continue
        first = pins[0]
        out[cid] = {
            "ids": ",".join(str(p.get("id") or "") for p in pins),
            "labels": ", ".join(str(p.get("label") or p.get("id") or "")
                                for p in pins),
            # R39: the user's own per-pin instructions, one joined leaf
            # ("boss@x: anything from her outranks the rest; …").
            "notes": "; ".join(
                f"{p.get('label') or p.get('id')}: {p['note']}"
                for p in pins if p.get("note")),
            "count": len(pins),
            "first": {
                "kind": str(first.get("kind") or ""),
                "id": str(first.get("id") or ""),
                "label": str(first.get("label") or first.get("id") or ""),
            },
        }
    return out


# ── Per-account read filters (R42, design §5.2) ──────────────────────
#
# `filters` is the user asking for LESS, per account:
#
#   filters: {"gmail": ["unread", "me"]}
#
# It is the deliberate opposite of `focus` above. A pin RANKS — it says
# "look here first" and must never stop other material from being
# fetched (R42, founder P6, and `executor_v2._apply_focus_scope` says
# so at length). A filter NARROWS: the user tapped "Unread only", so an
# unread-only read is the answer they asked for. That is why filters
# compose into the provider query and pins do not.
#
# THE ONE TABLE. Three readers need the same answer and a second copy
# is how this drifts: `validate_filters` below (which ids are legal),
# `workflow.available_filters` (the chips the app draws) and
# `executor_v2._apply_read_filters` (how each one composes, and which
# tools it composes INTO — read from `tools` here, never restated).
#
# A filter this connector cannot really express is NOT in the table. A
# chip that changes nothing is exactly the defect this round removes
# elsewhere, so under-offering is the correct error. R43 §6 asks for a
# fuller seed list than the tool surface can honour, and these are the
# chips it names that are NOT here, each for a reason that is a fact
# about the provider rather than a gap in this file:
#
#   github  every chip §6 names — failing checks, drafts, changed-since
#           — needs a pull request or a check run, and this connector
#           has neither tool. `github__list_issues` forwards only
#           `state` and `per_page`, excludes PRs outright, and `state`
#           already defaults to "open". No filters.
#   slack   "Mentions me" and "Threads I am in" need the user's own
#           handle, which no Slack tool here answers (`auth.test` is not
#           exposed). "Since my last read" needs `conversations.info`'s
#           `last_read`, also unexposed. "Skip bots" needs `bot_id`,
#           which `_message_row` does not carry. The time window is
#           real, and it is the one chip.
#   outlook "Addressed to me" would need the mailbox's own address to
#           write Graph's `to:`; nothing here knows it. Newsletters have
#           no Graph counterpart to Gmail's category tabs, so §6's
#           "Skip automated mail" is a `drop` on the sender instead.
#   teams   `teams__read_chat_messages` takes a chat id and a count and
#           nothing else; mentions, bots and read state are all
#           unreachable from it.
#   notion  `notion__search` sorts by last-edited but cannot filter on
#           it, and returns no person or archived flag to drop on. A
#           person filter is `notion__query_database` territory, which
#           needs a database the user has not been asked for.
#   calendar §6's "Meetings I own", "No agenda yet" and "Skip declined"
#           all need fields the provider's event row does not return
#           (organiser, description, responseStatus). The horizon is
#           real and is the one chip.
#   drive   nothing this vocabulary can narrow that the step's own
#           params do not already own.

#: The closed vocabulary of `compile` mutations (R43 §6). A filter
#: declares WHAT it does to the read; `executor_v2` owns HOW. Five
#: kinds, each with one shape:
#:
#:   query_term   {"param", "value"}   append one term to a provider
#:                                     search string, ANDed with the
#:                                     step's own query.
#:   jql_and      {"value"}            AND one clause into the WHERE
#:                                     half of `jql`, sort left trailing.
#:   param        {"name", "value"}    set one request param outright.
#:   time_window  {"direction", "hours", "param", "unit", "max_param"?}
#:                                     a bound computed from the run
#:                                     clock. direction "back" writes
#:                                     now-hours into `param`; "ahead"
#:                                     writes now into `param` and
#:                                     now+hours into `max_param`. unit
#:                                     is the encoding the provider
#:                                     takes: "iso", "unix", or
#:                                     "slack_after" (Slack search's own
#:                                     `after:<date>` term, appended to
#:                                     the query named by `param`).
#:   drop         {"field", "when", "values"?, "hours"?}
#:                                     remove RETURNED items — the one
#:                                     kind that runs after the call,
#:                                     for narrowings no provider query
#:                                     can express. `when` is "present",
#:                                     "absent", "contains" (any of
#:                                     `values`, case-insensitive) or
#:                                     "older_than" (`hours`).
#:
#: A filter's `compile` is a LIST, because one chip can be two
#: mutations (Gmail's newsletters are two categories) and because the
#: same chip can compile differently per tool — an entry may carry its
#: own `tools` to say which of the filter's tools it applies to, and
#: omitting it means all of them.
#:
#: Two kinds the contract names are deliberately absent. `graph_filter`
#: has no user: `outlook__list_messages` speaks `$search` and typed
#: params, and forwards no OData `$filter` a caller could compose into.
#: `sort` still has no user, and R43 chose not to add one: GitHub
#: search cannot order by check status at all (its `sort` accepts
#: created/updated/comments/reactions), so "Failing checks first"
#: compiles as a `query_term` that SELECTS the red ones. A `sort`
#: kind would have had exactly one caller and no provider able to
#: honour it.
FILTER_COMPILE_KINDS = frozenset({
    "query_term", "jql_and", "param", "time_window", "drop",
})

CONNECTOR_FILTERS: dict[str, tuple[dict, ...]] = {
    "gmail": (
        {"id": "me", "label": "Addressed to me",
         "tools": ("gmail__list_messages", "gmail__search_threads"),
         "compile": ({"kind": "query_term", "param": "query",
                      "value": "to:me"},)},
        {"id": "unread", "label": "Unread only",
         "tools": ("gmail__list_messages", "gmail__search_threads"),
         "compile": ({"kind": "query_term", "param": "query",
                      "value": "is:unread"},)},
        # Gmail's own tabs, which is where every bulk mailing this chip
        # means already lands. There is no "newsletter" predicate.
        {"id": "no_promos", "label": "Skip newsletters",
         "tools": ("gmail__list_messages", "gmail__search_threads"),
         "compile": ({"kind": "query_term", "param": "query",
                      "value": "-category:promotions"},
                     {"kind": "query_term", "param": "query",
                      "value": "-category:updates"})},
        {"id": "day", "label": "Last 24 hours",
         "tools": ("gmail__list_messages", "gmail__search_threads"),
         "compile": ({"kind": "query_term", "param": "query",
                      "value": "newer_than:1d"},)},
    ),
    "outlook": (
        # R43 §6. A typed param rather than a `graph_filter`: the tool
        # resolves the mailbox's own address (Outlook stores none) and
        # puts it in the KQL search, which is the one place Graph will
        # take a recipient restriction beside everything else this read
        # already does.
        {"id": "me", "label": "Addressed to me",
         "tools": ("outlook__list_messages",),
         "compile": ({"kind": "param", "name": "to_me", "value": True},)},
        {"id": "unread", "label": "Unread only",
         "tools": ("outlook__list_messages",),
         "compile": ({"kind": "param", "name": "is_read", "value": False},)},
        # A `drop` rather than a query term: Graph's KQL has no
        # automated-mail predicate, and `NOT from:noreply` inside a
        # `$search` narrows only the one literal address. The senders
        # below are what automated mail actually uses.
        {"id": "no_auto", "label": "Skip automated mail",
         "tools": ("outlook__list_messages",),
         "compile": ({"kind": "drop", "field": "from", "when": "contains",
                      "values": ("noreply", "no-reply", "donotreply",
                                 "do-not-reply", "mailer-daemon")},)},
        {"id": "day", "label": "Last 24 hours",
         "tools": ("outlook__list_messages",),
         "compile": ({"kind": "time_window", "direction": "back",
                      "hours": 24, "param": "since", "unit": "iso"},)},
    ),
    # Date-granular in search, exact in a channel read — one label that
    # is true of both, and two compile entries because the two tools
    # take the bound in different units.
    "slack": (
        {"id": "day", "label": "Since yesterday",
         "tools": ("slack__read_messages", "slack__search_messages"),
         "compile": ({"kind": "time_window", "direction": "back",
                      "hours": 24, "param": "oldest", "unit": "unix",
                      "tools": ("slack__read_messages",)},
                     {"kind": "time_window", "direction": "back",
                      "hours": 24, "param": "query", "unit": "slack_after",
                      "tools": ("slack__search_messages",)})},
        # R43 §6. Three of the four are provider params rather than
        # query terms because each needs to know WHO is asking, and the
        # compile vocabulary has no clock and no identity — only the
        # provider can resolve `auth.test`. `slack__list_mentions` and
        # `slack__list_threads` are deliberately NOT in `tools`: a step
        # that already runs those IS the narrowing, and a chip over it
        # would change nothing.
        {"id": "mentions", "label": "Mentions me",
         "tools": ("slack__read_messages",),
         "compile": ({"kind": "param", "name": "mentions_only",
                      "value": True},)},
        {"id": "mine", "label": "Threads I am in",
         "tools": ("slack__read_messages",),
         "compile": ({"kind": "param", "name": "threads_only",
                      "value": True},)},
        # A `drop`, and the right kind for it: Slack's API has no bot
        # predicate, and `bot_id` is the only reliable not-a-person
        # signal on a message — `subtype` answers a different question
        # and is absent on a plain bot post.
        {"id": "no_bots", "label": "Skip bots",
         "tools": ("slack__read_messages",),
         "compile": ({"kind": "drop", "field": "bot_id",
                      "when": "present"},)},
        # Read state lives on `conversations.info` and nowhere else, so
        # the provider pays one extra call. An explicit `oldest` — the
        # `day` chip above, or the step's own — always wins: a bound
        # only ever moves inward.
        {"id": "since_read", "label": "Since my last read",
         "tools": ("slack__read_messages",),
         "compile": ({"kind": "param", "name": "since_last_read",
                      "value": True},)},
    ),
    "teams": (
        {"id": "mentions", "label": "Mentions me",
         "tools": ("teams__read_chat_messages",),
         "compile": ({"kind": "param", "name": "mentions_only",
                      "value": True},)},
        # A `drop`, not a query term: Graph exposes no author-kind
        # predicate on /chats/{id}/messages. A system message ("Dana
        # joined the chat") has author_type "" and is not a bot.
        {"id": "no_bots", "label": "Skip bots",
         "tools": ("teams__read_chat_messages",),
         "compile": ({"kind": "drop", "field": "author_type",
                      "when": "contains",
                      "values": ("application", "device")},)},
        {"id": "since_read", "label": "Since my last read",
         "tools": ("teams__read_chat_messages",),
         "compile": ({"kind": "param", "name": "since_last_read",
                      "value": True},)},
    ),
    # R43 §6. All three compose into `github__search_issues` — the only
    # GitHub tool that speaks a query — so a step that lists one repo's
    # issues still offers nothing, which is the R42 gate working.
    "github": (
        # NOT an ordering. GitHub search's `sort` accepts created /
        # updated / comments / reactions and nothing about checks, so
        # this SELECTS the pull requests whose checks are red rather
        # than putting them first. The spec's label is kept verbatim;
        # "Failing checks only" is the honest wording if copy may move.
        {"id": "failing", "label": "Failing checks first",
         "tools": ("github__search_issues",),
         "compile": ({"kind": "query_term", "param": "q",
                      "value": "status:failure"},)},
        {"id": "no_drafts", "label": "Skip drafts",
         "tools": ("github__search_issues",),
         "compile": ({"kind": "query_term", "param": "q",
                      "value": "-is:draft"},)},
        # A PARAM, not a query term: a chip carries no clock. The
        # provider turns the ISO written here into GitHub's own
        # `updated:>=`, which is what keeps the compile vocabulary at
        # five kinds instead of six.
        {"id": "day", "label": "Changed since yesterday",
         "tools": ("github__search_issues",),
         "compile": ({"kind": "time_window", "direction": "back",
                      "hours": 24, "param": "updated_since",
                      "unit": "iso"},)},
    ),
    # `day` compiles differently per tool — Notion's search cannot
    # narrow on time at all, a data-source query can — the same
    # two-entry shape Slack's `day` uses. `mine` names ONLY
    # `notion__query_database`, because Notion's search has no person
    # predicate and returns no `properties` to post-filter on; the R42
    # `available_filters` gate then hides the chip unless the automation
    # runs such a step.
    "notion": (
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
         "compile": ({"kind": "param", "name": "assigned_to_me",
                      "value": True},)},
        {"id": "no_archived", "label": "Skip archived",
         "tools": ("notion__search", "notion__query_database"),
         "compile": ({"kind": "drop", "field": "state", "when": "contains",
                      "values": ("archived", "in_trash")},)},
    ),
    "jira": (
        {"id": "priority", "label": "P1 and P2 only",
         "tools": ("jira__search_issues",),
         "compile": ({"kind": "jql_and",
                      "value": "priority in (Highest, High)"},)},
        {"id": "open", "label": "Skip closed",
         "tools": ("jira__search_issues",),
         "compile": ({"kind": "jql_and",
                      "value": "statusCategory != Done"},)},
        {"id": "due_week", "label": "Due this week",
         "tools": ("jira__search_issues",),
         "compile": ({"kind": "jql_and",
                      "value": "duedate <= endOfWeek()"},)},
        {"id": "day", "label": "Changed since yesterday",
         "tools": ("jira__search_issues",),
         "compile": ({"kind": "jql_and", "value": "updated >= -1d"},)},
    ),
    "calendar": (
        # The only calendar narrowing this vocabulary can express, and
        # it is a real one: `calendar__list_events` orders ASCENDING and
        # windows only when asked, so an unwindowed read answers with
        # the OLDEST events in the account (R42, B1).
        {"id": "next24", "label": "Next 24 hours",
         "tools": ("calendar__list_events",),
         "compile": ({"kind": "time_window", "direction": "ahead",
                      "hours": 24, "param": "time_min",
                      "max_param": "time_max", "unit": "iso"},)},
        {"id": "mine", "label": "Meetings I own",
         "tools": ("calendar__list_events",),
         "compile": ({"kind": "param", "name": "organized_by_me",
                      "value": True},)},
        {"id": "no_agenda", "label": "No agenda yet",
         "tools": ("calendar__list_events",),
         "compile": ({"kind": "param", "name": "without_agenda",
                      "value": True},)},
        # A `drop`: the Calendar API has no responseStatus predicate.
        # An event the user is not an attendee of answers "" and STAYS —
        # not being invited is not being declined.
        {"id": "no_declined", "label": "Skip declined",
         "tools": ("calendar__list_events",),
         "compile": ({"kind": "drop", "field": "my_response",
                      "when": "contains", "values": ("declined",)},)},
    ),
}

MAX_FILTERS_PER_ACCOUNT = 8


def filter_options(connector_id: str) -> tuple[dict, ...]:
    """Every filter this connector can express, in chip order."""
    return CONNECTOR_FILTERS.get(str(connector_id or ""), ())


def filter_ids(connector_id: str) -> frozenset:
    return frozenset(f["id"] for f in filter_options(connector_id))


def filter_tools(connector_id: str, filter_id: str) -> tuple[str, ...]:
    """The tools this filter composes into — the table's own answer to
    "can this step express it", so the executor never restates it."""
    for f in filter_options(connector_id):
        if f["id"] == filter_id:
            return tuple(f.get("tools") or ())
    return ()


def filter_compile(
    connector_id: str, filter_id: str, tool: str = "",
) -> tuple[dict, ...]:
    """The mutations this filter makes to a read, for `tool`.

    The applier asks the table rather than carrying its own per-
    connector branches, so a chip that is offered and a chip that
    narrows are the same fact. An entry may pin itself to a subset of
    the filter's tools (Slack's one time window is a unix `oldest` on a
    channel read and an `after:` term in a search); an entry with no
    `tools` applies to every tool the filter declares.
    """
    for f in filter_options(connector_id):
        if f["id"] != filter_id:
            continue
        return tuple(
            m for m in (f.get("compile") or ())
            if not tool or not m.get("tools") or tool in m["tools"]
        )
    return ()


def validate_filters(spec: dict, errors: list) -> dict:
    """`spec["filters"]` → the canonical `{connector_id: [id]}` map.

    Shape AND membership, unlike `validate_focus`: a filter id is drawn
    from a closed table, so an unknown one is a malformed spec rather
    than a user error about their own account. Order is the TABLE's,
    not the caller's — the chips render in one order everywhere, and
    two specs that narrow identically serialize identically.
    """
    filters = spec.get("filters")
    if filters is None:
        return {}
    if not isinstance(filters, dict):
        _err(errors, "bad_filters", "filters",
             "filters must map a connector id to a list of filter ids")
        return {}
    out: dict[str, list[str]] = {}
    for cid, ids in filters.items():
        fld = f"filters.{cid}"
        if not isinstance(cid, str) or not cid.strip():
            _err(errors, "bad_filters", "filters",
                 "filters keys must be connector ids")
            continue
        if not isinstance(ids, list):
            _err(errors, "bad_filters", fld,
                 "each account's filters is a list of ids")
            continue
        if len(ids) > MAX_FILTERS_PER_ACCOUNT:
            _err(errors, "too_many_filters", fld,
                 f"at most {MAX_FILTERS_PER_ACCOUNT} filters per account")
            ids = ids[:MAX_FILTERS_PER_ACCOUNT]
        known = filter_ids(cid)
        wanted = set()
        for i, fid in enumerate(ids):
            if not isinstance(fid, str) or fid not in known:
                _err(errors, "unknown_filter", f"{fld}[{i}]",
                     f"{fid!r} is not a filter {cid} can express "
                     f"(known: {sorted(known)})")
                continue
            wanted.add(fid)
        kept = [f["id"] for f in filter_options(cid) if f["id"] in wanted]
        if kept:
            out[cid] = kept
    return out


# ── Instant triggers (R42, design §5.3) ──────────────────────────────
#
# The label table for `trigger.sources[]` events, beside the filter
# table because they are the same kind of thing: the connector declares
# the capability, this file names it in the product's voice. The
# manifest's own `description` is written for the MODEL ("A new issue
# appears in a project you pick"); these are written for the person
# tapping the row.
#
# Nothing here invents an event. A key with no entry falls back to its
# manifest description, so a connector that declares a new event shows
# up honestly — worded for the model — rather than not at all.
#
# R43 §7 fixes the wording for every event the round names. Entries for
# events no manifest declares are kept here on purpose: the picker is
# built from the REGISTRY, so a label with no event is inert, and the
# alternative — deleting it — loses the agreed wording the day the
# provider gains the tool that makes the event real.
EVENT_LABELS: dict[str, str] = {
    "calendar.event_created": "A new event lands on my calendar",
    "calendar.invitation_arrived": "An invitation arrives",
    "calendar.meeting_soon": "A meeting lands in the next 24 hours",
    "calendar.no_agenda": "A meeting I own has no agenda",
    "drive.file_added": "A file shows up in my Drive",
    "github.build_red": "A build goes red",
    "github.issue_opened": "An issue is opened in that repo",
    "github.pr_approved": "My pull request is approved",
    "github.pr_commented": "My pull request gets a comment",
    "github.review_requested": "A review is requested from me",
    "gmail.email_received": "Mail addressed to me arrives",
    "gmail.invoice_landed": "An invoice lands",
    "gmail.pinned_thread_reply": "A thread I pinned gets a reply",
    "gmail.vip_wrote": "Someone in my VIPs writes",
    "jira.due_moved": "A due date moves",
    "jira.issue_assigned": "A ticket is assigned to me",
    "jira.issue_created": "A ticket is created in that project",
    "jira.issue_reopened": "My ticket is reopened",
    "jira.p1_raised": "A P1 is raised",
    "notion.deadline_moved": "A deadline moves",
    "notion.page_added": "A page shows up in my workspace",
    "notion.page_changed": "A page tagged to me changes",
    "outlook.email_received": "Mail addressed to me arrives",
    "outlook.meeting_request": "A meeting request arrives",
    "outlook.vendor_moved": "A vendor thread moves",
    # The one Slack event a user token can honestly poll, and it is
    # deliberately not one of §7's four ids: `conversations.history`
    # reads ONE conversation, so the row has to name the place the pin
    # names rather than promise "#oncall" or "any DM".
    "slack.channel_message": "A message lands in a channel I picked",
    "slack.dm_arrived": "A DM arrives",
    "slack.mentioned": "I am @mentioned",
    "slack.oncall_message": "#oncall gets a message",
    "slack.thread_moved": "A thread I am in moves",
    "teams.chat_message_received": "A chat arrives",
    "teams.mentioned": "I am @mentioned",
}


def event_label(connector_id: str, event_key: str, fallback: str = "") -> str:
    return (EVENT_LABELS.get(f"{connector_id}.{event_key}")
            or (fallback or "").strip()
            or str(event_key or ""))


def unanswered_variables(spec: Any) -> list[str]:
    """The spec's `{{var.<name>}}` references that have no answer yet.

    Version dispatch, like `validate_spec`: variables are a v2 grammar,
    so a v1 spec has none and answers `[]`. Derived from what the spec
    REFERENCES rather than from the template's declared list, so an
    automation the agent edited after adoption is covered too.
    """
    if not isinstance(spec, dict) or spec.get("version") != 2:
        return []
    from .spec_v2 import unanswered_variables as _v2_unanswered
    return _v2_unanswered(spec)


def validate_spec(
    spec: Any,
    registry: dict[str, dict],
    *,
    template_mode: bool = False,
    template_vars: Optional[set] = None,
):
    """Validate one AutomationSpec against the capability registry.

    `registry` maps connector_id → the automation_registry() entry
    (push/poll/floor_s/events/scopes_write_by_action/...). Raises
    SpecError with EVERY problem found.

    Dispatch (Round 28): a spec with `version: 2` returns a
    `ValidatedSpecV2` from spec_v2.py; anything else takes the v1 path
    below, unchanged. `template_mode` waives grant references — the
    create, edit and fire paths all set it, because a grant is
    enforced at ARM and at DISPATCH, never by a parse — and treats
    `template_vars` as declared. `template_mode` with NO
    `template_vars` additionally waives the undeclared-variable
    rule, which is why an unanswered setting is caught by
    `unanswered_variables` at arm and fire instead of here.
    """
    errors: list[dict] = []

    if not isinstance(spec, dict):
        raise SpecError([{
            "code": "not_an_object", "field": "",
            "message": f"spec must be an object, got {type(spec).__name__}",
        }])

    version = spec.get("version", 1)
    if version == 2:
        from .spec_v2 import validate_spec_v2
        return validate_spec_v2(
            spec, registry,
            template_mode=template_mode, template_vars=template_vars,
        )
    if version != 1:
        raise SpecError([{
            "code": "bad_version", "field": "version",
            "message": f"spec version must be 1 or 2, got {version!r}",
        }])

    for k in spec:
        if k not in _TOP_KEYS:
            _err(errors, "unknown_field", k, f"unknown top-level field {k!r}")

    name = spec.get("name")
    if not isinstance(name, str) or not (1 <= len(name.strip()) <= 120):
        _err(errors, "bad_name", "name", "name must be 1-120 characters")
        name = ""
    else:
        name = name.strip()

    mode = spec.get("mode", "confirm")
    if mode not in ("auto", "confirm"):
        _err(errors, "bad_mode", "mode", "mode must be 'auto' or 'confirm'")
        mode = "confirm"

    # ── trigger ──────────────────────────────────────────────────────
    trig = spec.get("trigger")
    if not isinstance(trig, dict):
        _err(errors, "missing_trigger", "trigger", "trigger object is required")
        trig = {}
    for k in trig:
        if k not in _TRIGGER_KEYS:
            _err(errors, "unknown_field", f"trigger.{k}",
                 f"unknown trigger field {k!r}")

    t_mode = trig.get("mode")
    if t_mode not in AUTOMATION_TRIGGER_MODES:
        _err(errors, "bad_trigger_mode", "trigger.mode",
             f"trigger.mode must be one of {sorted(AUTOMATION_TRIGGER_MODES)}")
        t_mode = "schedule"

    t_connector = trig.get("connector_id")
    t_event = trig.get("event")
    t_params = trig.get("params") or {}
    poll_interval: Optional[int] = None
    schedule: Optional[dict] = None
    event_spec: Optional[dict] = None
    filter_rules = trig.get("filter") or {}
    if not isinstance(filter_rules, dict):
        _err(errors, "bad_filter", "trigger.filter", "filter must be an object")
        filter_rules = {}
    if not isinstance(t_params, dict):
        _err(errors, "bad_params", "trigger.params", "params must be an object")
        t_params = {}

    if t_mode in ("push", "poll"):
        cap = registry.get(t_connector) if isinstance(t_connector, str) else None
        if cap is None:
            _err(errors, "unknown_connector", "trigger.connector_id",
                 f"connector {t_connector!r} is not automatable "
                 f"(known: {sorted(registry)})")
        else:
            if t_mode == "push" and not cap.get("push"):
                _err(errors, "push_unavailable", "trigger.mode",
                     f"{t_connector} has no push path — use poll")
            if t_mode == "poll" and not cap.get("poll"):
                _err(errors, "poll_unavailable", "trigger.mode",
                     f"{t_connector} does not support polling")
            events = {e["key"]: e for e in cap.get("events", [])}
            if not isinstance(t_event, str) or t_event not in events:
                _err(errors, "unknown_event", "trigger.event",
                     f"event {t_event!r} not declared by {t_connector} "
                     f"(known: {sorted(events)})")
            else:
                event_spec = events[t_event]
        if t_mode == "poll":
            # Identical to the pre-R28 max() when the fast lane is off
            # (the default); seconds only for env-gated dev tenants.
            floor = effective_poll_floor((cap or {}).get("floor_s"))
            raw_iv = trig.get("poll_interval_s", floor)
            if not isinstance(raw_iv, int) or isinstance(raw_iv, bool):
                _err(errors, "bad_interval", "trigger.poll_interval_s",
                     "poll_interval_s must be an integer number of seconds")
            elif raw_iv < floor:
                _err(errors, "interval_below_floor", "trigger.poll_interval_s",
                     f"poll_interval_s={raw_iv} is below the floor of "
                     f"{floor}s for {t_connector}")
            else:
                poll_interval = raw_iv
    elif t_mode == "schedule":
        schedule = trig.get("schedule")
        if not isinstance(schedule, dict) or not schedule:
            _err(errors, "missing_schedule", "trigger.schedule",
                 "schedule mode requires trigger.schedule")
            schedule = None
        else:
            unknown = set(schedule) - _SCHEDULE_KEYS
            if unknown:
                _err(errors, "unknown_field", "trigger.schedule",
                     f"unknown schedule fields {sorted(unknown)}")
            shape_keys = [k for k in _SCHEDULE_KEYS if schedule.get(k)]
            if len(shape_keys) != 1:
                _err(errors, "bad_schedule", "trigger.schedule",
                     "schedule must set exactly one of cron_local / at / every_s")
            ev = schedule.get("every_s")
            ev_floor = effective_every_floor()
            if ev is not None and (
                not isinstance(ev, int) or isinstance(ev, bool)
                or ev < ev_floor
            ):
                _err(errors, "bad_schedule", "trigger.schedule.every_s",
                     f"every_s must be an integer >= {ev_floor}")

    # ── dedupe key ───────────────────────────────────────────────────
    dedupe_field: Optional[str] = None
    dk = spec.get("dedupe_key")
    if t_mode in ("push", "poll"):
        if not isinstance(dk, str) or not dk.strip():
            _err(errors, "missing_dedupe_key", "dedupe_key",
                 "push/poll automations require a dedupe_key "
                 "(\"event.<field>\")")
        else:
            dk = dk.strip()
            if not dk.startswith("event."):
                _err(errors, "bad_dedupe_key", "dedupe_key",
                     "dedupe_key must be an \"event.<field>\" reference")
            else:
                fname = dk[len("event."):]
                if event_spec is not None:
                    known = set((event_spec.get("fields") or {}).keys())
                    known.add(event_spec.get("dedupe_field") or "")
                    if fname not in known:
                        _err(errors, "bad_dedupe_key", "dedupe_key",
                             f"{fname!r} is not a field of event "
                             f"{t_event!r} (known: {sorted(known - {''})})")
                dedupe_field = fname

    # ── action ───────────────────────────────────────────────────────
    act = spec.get("action")
    if not isinstance(act, dict):
        _err(errors, "missing_action", "action", "action object is required")
        act = {}
    for k in act:
        if k not in _ACTION_KEYS:
            _err(errors, "unknown_field", f"action.{k}",
                 f"unknown action field {k!r}")

    a_connector = act.get("connector_id")
    a_tool = act.get("tool")
    a_params = act.get("params_template") or {}
    grant_id = act.get("grant_id")
    mutates = False
    if not isinstance(a_params, dict):
        _err(errors, "bad_params", "action.params_template",
             "params_template must be an object")
        a_params = {}

    a_cap = registry.get(a_connector) if isinstance(a_connector, str) else None
    if a_cap is None:
        _err(errors, "unknown_connector", "action.connector_id",
             f"connector {a_connector!r} is not automatable")
    elif not isinstance(a_tool, str) or not a_tool:
        _err(errors, "unknown_tool", "action.tool", "action.tool is required")
    else:
        writes = a_cap.get("scopes_write_by_action") or {}
        if a_tool in writes:
            mutates = True
            if (not template_mode
                    and (not isinstance(grant_id, str) or not grant_id.strip())):
                _err(errors, "write_without_grant", "action.grant_id",
                     f"{a_tool} is a write action — a grant reference is "
                     f"required before it can be part of a spec")
        elif not a_tool.startswith(f"{a_connector}__"):
            _err(errors, "unknown_tool", "action.tool",
                 f"{a_tool!r} does not belong to connector {a_connector!r}")
        # Non-write tools that pass the prefix check are validated
        # against the live manifest at arm/execute time by the
        # dispatcher (unknown tool ⇒ tool_error, fail closed).

    focus = validate_focus(spec, errors)
    filters = validate_filters(spec, errors)

    if errors:
        raise SpecError(errors)

    canonical = {
        "name": name,
        "description": spec.get("description") or None,
        **({"focus": focus} if focus else {}),
        **({"filters": filters} if filters else {}),
        "trigger": {
            "mode": t_mode,
            **({"connector_id": t_connector} if t_connector else {}),
            **({"event": t_event} if t_event else {}),
            **({"params": t_params} if t_params else {}),
            **({"poll_interval_s": poll_interval} if poll_interval else {}),
            **({"schedule": schedule} if schedule else {}),
            **({"filter": filter_rules} if filter_rules else {}),
        },
        "action": {
            "connector_id": a_connector,
            "tool": a_tool,
            "params_template": a_params,
            **({"grant_id": grant_id} if grant_id else {}),
            **({"grant_target": act.get("grant_target")}
               if isinstance(act.get("grant_target"), dict) else {}),
        },
        **({"dedupe_key": dk} if t_mode in ("push", "poll") else {}),
        "mode": mode,
    }
    return ValidatedSpec(
        raw=canonical,
        name=name,
        mode=mode,
        trigger_mode=t_mode,
        trigger_connector_id=t_connector if isinstance(t_connector, str) else None,
        trigger_event=t_event if isinstance(t_event, str) else None,
        trigger_params=t_params,
        poll_interval_s=poll_interval,
        schedule=schedule,
        filter_rules=filter_rules,
        action_connector_id=a_connector or "",
        action_tool=a_tool or "",
        action_params_template=a_params,
        grant_id=grant_id if isinstance(grant_id, str) else None,
        action_mutates=mutates,
        dedupe_key_field=dedupe_field,
        event_spec=event_spec,
    )


# ── Template rendering (executor's prepare step) ─────────────────────


def resolve_path(obj: Any, path: str) -> Any:
    """Dot-path lookup into nested dicts. Returns None on any miss."""
    cur = obj
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def render_value(val: Any, ctx: dict) -> Any:
    """Render one value against a context dict. Only strings are
    templated; a placeholder that resolves to None renders as an empty
    string. Shared by the v1 and v2 render paths."""
    if not isinstance(val, str):
        return val
    out = val
    # Simple, deliberate: find {{...}} spans, resolve, substitute.
    while "{{" in out and "}}" in out:
        start = out.index("{{")
        end = out.index("}}", start)
        expr = out[start + 2:end].strip()
        resolved = resolve_path(ctx, expr)
        out = out[:start] + ("" if resolved is None else str(resolved)) + out[end + 2:]
    return out


def render_with_ctx(template: dict, ctx: dict) -> dict:
    return {k: render_value(v, ctx) for k, v in template.items()}


def render_params(
    template: dict,
    *,
    event: Optional[dict] = None,
    grant_target: Optional[dict] = None,
) -> dict:
    """Fill a params_template's {{event.x}} / {{grant.target.x}}
    placeholders. Only string values are templated; a placeholder that
    resolves to None renders as an empty string (the validator upstream
    keeps required fields from being empty at execute time)."""
    ctx = {
        "event": event or {},
        "grant": {"target": grant_target or {}},
    }
    return render_with_ctx(template, ctx)
