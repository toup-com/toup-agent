"""Verb dictionary v2 — the ENTRIES (R30 CONTRACTS §2; C-owned copy).

Lives in `app/services` because the platform image ships no `app/agent/`
(templates read the dictionary platform-side). `automation_verbs.py`
owns the structure (`turn_action`,
`live_sentence`, `failure_action`, `job_card_label`, `sheet_subtitle`,
`trigger_sub`); this module owns every phrase those functions can say,
per connector. The tables are total over the automatable surface: an
unknown tool falls back to the connector's generic entry, never to a
raw identifier.

Slots are engine-filled: `{count}` (int, singularised by the caller's
grammar), `{when}` (an app-formatted local time), `{channel}` /
`{target}` (real names). A template with an unfilled slot renders
without its clause, never with braces showing.

R31-25 pin: **a slot no renderer can fill is a brace on the user's
screen.** `automation_verbs._n` substitutes `{n}` and `{count}` and
nothing else, so `"{count} issues moved · {need_count} needs you"`
rendered as `"0 issues moved · {need_count} needs you"` on a founder's
job sheet. The old pin could not catch it: it rendered every phrase
with `str.format(count=3, need_count=1, …)`, a kwargs bag more generous
than production. Only `{count}`/`{n}`, and the failure table's `{name}`,
may appear here — the test now renders through the production path and
scans in rendered mode, so a new unfillable slot fails.

ND-4 pin: a failed or refused step NEVER wears a done-form verb —
`failures` phrasings are their own table, and the guard test asserts
none of them collides with a write's done form.
"""

from __future__ import annotations

#: The setup thread's capability check (§5.3) — a served action with no
#: live dispatch behind it; the serializer allows exactly this pair.
CAPABILITY_CHECK_ACTION = "Checked what I can do"
RECONNECTED_ACTION = "Connected again"

#: Failure phrasings shared by every connector; `failure_action` picks
#: by reason, with the connector's display name available as `{name}`.
COMMON_FAILURES: dict[str, dict[str, str]] = {
    "access_expired": {"action": "Could not connect", "detail": "access expired"},
    "provider_down": {"action": "Could not reach {name}", "detail": "it did not answer"},
    "rate_limited": {"action": "Could not finish", "detail": "{name} asked me to slow down"},
    "refused": {"action": "{name} refused", "detail": "it said no to the request"},
    "write_failed": {"action": "Could not make the change", "detail": "nothing was changed"},
    "stopped": {"action": "Stopped before this step", "detail": "you stopped the run"},
    # R31-13 / §4.4 reason codes, entered under their own names so
    # `failure_action`'s DIRECT lookup wins before `_C_FAILURE_ALIASES`
    # can reach them. That alias table folds `scope_missing` onto
    # `access_expired`, which tells a user to reconnect a connection
    # that is working — the missing thing is a permission, and the fix
    # is Grant access, not Reconnect. Wording is kept in step with
    # `fixtures/automations/reason-strings.json`; that file is the
    # authority and these are its dictionary renderings.
    "token_expired": {"action": "Could not connect", "detail": "access expired"},
    "token_revoked": {"action": "Could not connect", "detail": "the connection was removed"},
    "scope_missing": {"action": "Could not connect",
                      "detail": "it does not have that access yet"},
    "org_approval_needed": {"action": "Could not reach {name}",
                            "detail": "the organisation has not approved Toup"},
    "vendor_down": {"action": "Could not reach {name}", "detail": "it did not answer"},
    "timeout": {"action": "Could not reach {name}",
                "detail": "it did not answer in time"},
    # `executor_v2._failure_reason` recognises five tokens in the RPC
    # error and returns "unreachable" for everything else — including
    # `org_approval_needed`, the exact case R31-07 was written for.
    # Without an entry of its own that fell through to
    # `_V2_FAILURE_DEFAULT` ("Could not connect", "it did not answer"),
    # so "we do not know why" was rendered as a specific diagnosis, and
    # the user was sent to wait for a service that was answering fine.
    # An honest unknown is worth more than a confident wrong cause.
    "unreachable": {"action": "Could not reach {name}",
                    "detail": "I could not tell why"},
}

ENTRIES: dict[str, dict] = {
    "gmail": {
        "display": "Gmail",
        "reads": {
            "gmail__list_messages": {
                "action": "Read your unread mail",
                "detail": "{count} new thread(s)",
                "progressive": "reading {count} unread messages",
            },
            "gmail__get_message": {
                "action": "Read one thread",
                "detail": "the one that needed it",
                "progressive": "reading a thread",
            },
            "gmail__search_threads": {
                "action": "Searched your mail",
                "detail": "{count} thread(s) matched",
                "progressive": "searching your mail",
            },
            "*": {
                "action": "Checked Gmail",
                "detail": "{count} item(s)",
                "progressive": "checking Gmail",
            },
        },
        "writes": {
            "gmail__create_draft": {
                "action": "Wrote a draft",
                "detail": "waiting in Gmail — nothing sent",
                "progressive": "writing a draft",
                "job_label": "wrote a draft",
                "clause": "draft an email for you",
            },
        },
        "trigger_sub": "when mail arrives",
        "permission_labels": {
            "read_new_mail": "Read new mail",
            "write_drafts": "Write drafts",
        },
        "rails": ("Send anything", "Delete mail"),
    },
    "outlook": {
        "display": "Outlook",
        "reads": {
            "outlook__list_messages": {
                "action": "Read your unread mail",
                "detail": "{count} new thread(s)",
                "progressive": "reading {count} unread messages",
            },
            "outlook__get_message": {
                "action": "Read one thread",
                "detail": "the one that needed it",
                "progressive": "reading a thread",
            },
            "*": {
                "action": "Checked Outlook",
                "detail": "{count} item(s)",
                "progressive": "checking Outlook",
            },
        },
        "writes": {
            "outlook__create_draft": {
                "action": "Wrote a draft",
                "detail": "waiting in Outlook — nothing sent",
                "progressive": "writing a draft",
                "job_label": "wrote a draft",
                "clause": "draft an email for you",
            },
        },
        "trigger_sub": "when mail arrives",
        "permission_labels": {
            "read_new_mail": "Read new mail",
            "write_drafts": "Write drafts",
        },
        "rails": ("Send anything", "Delete mail"),
    },
    "slack": {
        "display": "Slack",
        "reads": {
            "slack__read_messages": {
                "action": "Read your channels",
                "detail": "{count} new message(s)",
                "progressive": "reading your channels",
            },
            "slack__search_messages": {
                "action": "Searched Slack",
                "detail": "{count} message(s) matched",
                "progressive": "searching Slack",
            },
            "slack__list_channels": {
                "action": "Looked at your channels",
                "detail": "{count} channel(s)",
                "progressive": "looking at your channels",
            },
            "*": {
                "action": "Checked Slack",
                "detail": "{count} item(s)",
                "progressive": "checking Slack",
            },
        },
        "writes": {
            "slack__send_message": {
                "action": "Told you in Slack",
                "action_others": "Posted in {channel}",
                "detail": "one line, no thread",
                "progressive": "writing the Slack line",
                "job_label": "told you in Slack",
                "job_label_others": "posted in {channel}",
                "clause": "post one line in {channel}",
            },
        },
        "trigger_sub": "when a message lands",
        "permission_labels": {
            "read_channels": "Read your channels",
            "post_as_you": "Post as you",
        },
        "rails": ("Read private DMs",),
    },
    "jira": {
        "display": "Jira",
        "reads": {
            "jira__search_issues": {
                # R31-07: a read verb reads like a read. "issues moved"
                # is a WRITE phrasing, and it sat on a turn that moved
                # nothing — "0 issues moved" on a run that only looked.
                "action": "Checked your board",
                "detail": "{count} open issue(s)",
                "progressive": "checking your board",
            },
            "*": {
                "action": "Checked Jira",
                "detail": "{count} issue(s)",
                "progressive": "checking Jira",
            },
        },
        "writes": {
            # R31-25: `{target}` belongs in `action_others`, the ONE
            # field `turn_action` substitutes into. A slot in `action`
            # is returned verbatim on the default `you` audience — that
            # is a brace on the user's screen, the same defect as
            # `{need_count}` and found by the same pin.
            "jira__add_comment": {
                "action": "Commented on your ticket",
                "action_others": "Commented on {target}",
                "detail": "one comment, nothing else touched",
                "progressive": "writing the comment",
                "job_label": "commented on your ticket",
                "job_label_others": "commented on {target}",
                "clause": "comment on your tickets",
            },
            "jira__create_issue": {
                "action": "Filed an issue on your board",
                "action_others": "Filed {target}",
                "detail": "assigned to nobody",
                "progressive": "filing the issue",
                "job_label": "filed an issue",
                "clause": "create a Jira issue",
            },
        },
        "trigger_sub": "when an issue appears",
        "permission_labels": {
            "read_your_tickets": "Read your tickets",
            "comment": "Comment",
        },
        "rails": ("Close or reassign",),
    },
    "github": {
        "display": "GitHub",
        "reads": {
            "github__list_issues": {
                "action": "Read your repositories",
                "detail": "{count} open pull request(s)",
                "progressive": "reading your repositories",
            },
            "*": {
                "action": "Checked GitHub",
                "detail": "{count} item(s)",
                "progressive": "checking GitHub",
            },
        },
        "writes": {
            "github__create_comment": {
                "action": "Commented on your pull request",
                "action_others": "Commented on {target}",
                "detail": "one comment, nothing merged",
                "progressive": "writing the comment",
                "job_label": "commented on your pull request",
                "job_label_others": "commented on {target}",
                "clause": "comment on GitHub",
            },
        },
        "trigger_sub": "when a pull request changes",
        "permission_labels": {
            "read_pull_requests": "Read pull requests",
            "comment": "Comment",
        },
        "rails": ("Push or merge",),
    },
    "teams": {
        "display": "Teams",
        "reads": {
            "teams__read_chat_messages": {
                "action": "Read your chats",
                "detail": "{count} new message(s)",
                "progressive": "reading your chats",
            },
            "*": {
                "action": "Checked Teams",
                "detail": "{count} item(s)",
                "progressive": "checking Teams",
            },
        },
        "writes": {
            "teams__send_chat_message": {
                "action": "Told you in Teams",
                "action_others": "Posted in {channel}",
                "detail": "one line, no thread",
                "progressive": "writing the Teams line",
                "job_label": "told you in Teams",
                "job_label_others": "posted in {channel}",
                "clause": "post one line in {channel}",
            },
        },
        "trigger_sub": "when a chat message lands",
        "permission_labels": {
            "read_your_chats": "Read your chats",
            "post_as_you": "Post as you",
        },
        "rails": ("Delete messages",),
    },
    "notion": {
        "display": "Notion",
        "reads": {
            "notion__search": {
                "action": "Read your pages",
                "detail": "{count} page(s) changed",
                "progressive": "reading your pages",
            },
            "*": {
                "action": "Checked Notion",
                "detail": "{count} page(s)",
                "progressive": "checking Notion",
            },
        },
        "writes": {
            "notion__create_page": {
                "action": "Added a page",
                "action_others": "Added a page in {target}",
                # `detail` is returned raw by `turn_action`; the page's
                # location rides `action_others`, which is filled.
                "detail": "nothing overwritten",
                "progressive": "writing the page",
                "job_label": "added a page",
                "clause": "add a Notion page",
            },
        },
        "trigger_sub": "when a page changes",
        "permission_labels": {
            "read_pages": "Read pages",
            "write_notes": "Write notes",
        },
        "rails": ("Delete pages",),
    },
    "drive": {
        "display": "Drive",
        "reads": {
            "drive__list_files": {
                "action": "Looked at your files",
                "detail": "{count} new file(s)",
                "progressive": "looking at your files",
            },
            "drive__get_file_text": {
                "action": "Read one file",
                "detail": "the one that changed",
                "progressive": "reading a file",
            },
            "*": {
                "action": "Checked Drive",
                "detail": "{count} file(s)",
                "progressive": "checking Drive",
            },
        },
        "writes": {
            "drive__create_doc": {
                "action": "Created a doc",
                "detail": "in your Drive, shared with nobody",
                "progressive": "creating the doc",
                "job_label": "created a doc",
                "clause": "create a doc",
            },
        },
        "trigger_sub": "when a file changes",
        "permission_labels": {
            "read_your_files": "Read your files",
            "create_docs": "Create docs",
        },
        "rails": ("Delete files", "Share anything"),
    },
    "docs": {
        "display": "Docs",
        "reads": {
            "docs__get": {
                "action": "Read the doc",
                "detail": "the one you pointed it at",
                "progressive": "reading the doc",
            },
            "*": {
                "action": "Checked Docs",
                "detail": "{count} doc(s)",
                "progressive": "checking Docs",
            },
        },
        "writes": {
            "docs__append_text": {
                "action": "Added to your doc",
                "detail": "at the end, nothing rewritten",
                "progressive": "updating your doc",
                "job_label": "added to your doc",
                "clause": "update your doc",
            },
            "docs__create": {
                "action": "Created a doc",
                "detail": "shared with nobody",
                "progressive": "creating the doc",
                "job_label": "created a doc",
                "clause": "create a doc",
            },
        },
        "trigger_sub": "when a doc changes",
        "permission_labels": {
            "read_docs": "Read docs",
            "update_docs": "Update docs",
        },
        "rails": ("Delete docs", "Share anything"),
    },
    "calendar": {
        "display": "Calendar",
        "reads": {
            "calendar__list_events": {
                "action": "Read your week",
                "detail": "{count} event(s)",
                "progressive": "reading your week",
            },
            "calendar__check_availability": {
                "action": "Looked for free time",
                "detail": "around what you already have",
                "progressive": "looking for free time",
            },
            "*": {
                "action": "Checked your calendar",
                "detail": "{count} event(s)",
                "progressive": "checking your calendar",
            },
        },
        "writes": {
            "calendar__create_event": {
                # `{when}` had no filler anywhere: `turn_action`
                # substitutes `{target}`/`{channel}` and nothing else,
                # so every held slot read "Held {when}". The time is on
                # the item the turn carries; the verb says what it did.
                "action": "Held time on your calendar",
                "detail": "only you can see it",
                "progressive": "holding the time",
                "job_label": "held time on your calendar",
                "clause": "hold time on your calendar",
            },
        },
        "trigger_sub": "when your week changes",
        "permission_labels": {
            "read_your_week": "Read your week",
            "hold_time": "Hold time",
        },
        "rails": ("Invite other people", "Delete events"),
    },
    "stub": {
        "display": "the test feed",
        "reads": {
            "stub__list_items": {
                "action": "Read the test feed",
                "detail": "{count} test item(s)",
                "progressive": "reading the test feed",
            },
            "*": {
                "action": "Checked the test feed",
                "detail": "{count} item(s)",
                "progressive": "checking the test feed",
            },
        },
        "writes": {
            "stub__post": {
                "action": "Posted to the test channel",
                "detail": "one test line",
                "progressive": "posting the test line",
                "job_label": "posted to the test channel",
                "clause": "post to the test channel",
            },
        },
        "trigger_sub": "when a test item appears",
        "permission_labels": {
            "read_test_items": "Read test items",
            "post_test_lines": "Post test lines",
        },
        "rails": (),
    },
}


def every_phrase() -> list[str]:
    """Every string the tables can serve, slots left visible — for the
    copy-guard sweep and the totality tests."""
    phrases: list[str] = [CAPABILITY_CHECK_ACTION, RECONNECTED_ACTION]
    for failure in COMMON_FAILURES.values():
        phrases.extend(failure.values())
    for entry in ENTRIES.values():
        phrases.append(entry["display"])
        phrases.append(entry["trigger_sub"])
        phrases.extend(entry["permission_labels"].values())
        phrases.extend(entry["rails"])
        for table in ("reads", "writes"):
            for verb in entry.get(table, {}).values():
                phrases.extend(
                    v for k, v in verb.items() if isinstance(v, str)
                )
    return phrases


def write_done_forms() -> list[str]:
    """Every done-form write action — the ND-4 pin's collision set."""
    forms: list[str] = []
    for entry in ENTRIES.values():
        for verb in entry.get("writes", {}).values():
            for key in ("action", "action_others"):
                if verb.get(key):
                    forms.append(verb[key])
    return forms


# ---------------------------------------------------------------------------
# Flat overlay views — `automation_verbs.py` imports these and overlays
# them onto its built-in defaults (dict.update semantics; the defaults
# stay the fallback, so totality never depends on this module).
# ---------------------------------------------------------------------------

def _flat(table: str, key: str) -> dict:
    out = {}
    for entry in ENTRIES.values():
        for tool, verb in entry.get(table, {}).items():
            if tool == "*" or key not in verb:
                continue
            out[tool] = verb[key]
    return out


#: tool → {"action", "detail"} (read turns; detail templates carry slots)
V2_READ: dict[str, dict] = {
    tool: {"action": verb["action"], "detail": verb["detail"]}
    for entry in ENTRIES.values()
    for tool, verb in entry.get("reads", {}).items()
    if tool != "*"
}

#: tool → progressive live sentence ("reading {count} unread messages")
V2_READ_LIVE: dict[str, str] = _flat("reads", "progressive")
V2_READ_LIVE.update(_flat("writes", "progressive"))

#: tool → the write phrasing set (done forms, job labels, clause)
V2_WRITE: dict[str, dict] = {
    tool: {k: v for k, v in verb.items() if k != "progressive"}
    for entry in ENTRIES.values()
    for tool, verb in entry.get("writes", {}).items()
}

#: reason → {"action", "detail"} — never a done form (ND-4)
V2_FAILURE: dict[str, dict] = dict(COMMON_FAILURES)

#: connector_id → the trigger node's sub ("when mail arrives")
V2_TRIGGER_SUB: dict[str, str] = {
    cid: entry["trigger_sub"] for cid, entry in ENTRIES.items()
}

#: Engine-authored actions with no dispatch behind them (the setup
#: capability check and the reconnect catch-up) — `is_served_action`
#: allows exactly these.
V2_ENGINE_ACTIONS: tuple[str, ...] = (
    CAPABILITY_CHECK_ACTION,
    RECONNECTED_ACTION,
)

#: connector_id → {permission id suffix → label} and the rail labels in
#: the connector's own words — the §4.4 permission registry's vocabulary.
V2_PERMISSION_LABELS: dict[str, dict] = {
    cid: dict(entry["permission_labels"]) for cid, entry in ENTRIES.items()
}
V2_RAILS: dict[str, tuple] = {
    cid: tuple(entry["rails"]) for cid, entry in ENTRIES.items()
}
