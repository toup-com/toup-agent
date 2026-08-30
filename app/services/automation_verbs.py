"""Human phrases for automation machinery — the verb dictionary (R29).

A lookup, never a derivation (the `tool_display.public_label`
precedent). It lives in `app/services/` because BOTH images import it:
the platform needs `schedule_human` for template cadence tags and the
agent needs everything else — and the platform image has no
`app/agent/`.

Totality is contractual (CONTRACTS-R29.md §1): no input — unknown
tool, unknown connector, garbage cron — may yield a raw tool name,
step id, or cron string. R29-C pins that with a test over the
fallback paths; nothing outside this module composes step copy.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

# The agent's own (non-connector) work — B renders the orb for these.
BRAND_ORB = None

_CONNECTOR_NAMES = {
    "gmail": "Gmail",
    "outlook": "Outlook",
    "jira": "Jira",
    "github": "GitHub",
    "slack": "Slack",
    "teams": "Teams",
    "notion": "Notion",
    "drive": "Drive",
    "docs": "Docs",
    "calendar": "Calendar",
    "stub": "Stub",
}


def display_name(connector_id: Optional[str]) -> Optional[str]:
    """Human name for a connector id; a well-formed unknown id gets a
    capitalized fallback, anything else None."""
    if not connector_id or not isinstance(connector_id, str):
        return None
    known = _CONNECTOR_NAMES.get(connector_id)
    if known:
        return known
    if re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", connector_id):
        return connector_id.replace("_", " ").replace("-", " ").title()
    return None


# tool → (doing, done, done-with-count or None). The count form is used
# only when a collect count exists at completion.
_TOOL_VERBS: dict[str, tuple[str, str, Optional[str]]] = {
    # reads
    "gmail__list_messages": ("Checking Gmail", "Checked Gmail",
                             "Read {count} new emails"),
    "gmail__get_message": ("Reading an email", "Read the email", None),
    "gmail__search_threads": ("Searching Gmail", "Searched Gmail",
                              "Found {count} threads"),
    "outlook__list_messages": ("Checking Outlook", "Checked Outlook",
                               "Read {count} Outlook emails"),
    "outlook__get_message": ("Reading an email", "Read the email", None),
    "jira__search_issues": ("Checking Jira", "Checked Jira",
                            "Read {count} Jira issues"),
    "github__list_issues": ("Checking GitHub", "Checked GitHub",
                            "Read {count} GitHub issues"),
    "teams__read_chat_messages": ("Checking Teams", "Checked Teams",
                                  "Read {count} Teams messages"),
    "notion__search": ("Checking Notion", "Checked Notion",
                       "Found {count} Notion pages"),
    "drive__list_files": ("Checking Drive", "Checked Drive",
                          "Found {count} new files"),
    "calendar__list_events": ("Checking your calendar",
                              "Checked your calendar",
                              "Found {count} events"),
    "stub__list_items": ("Checking the test feed", "Checked the test feed",
                         "Read {count} test items"),
    # writes
    "slack__send_message": ("Posting to Slack", "Posted to Slack", None),
    "teams__send_chat_message": ("Posting to Teams", "Posted to Teams", None),
    "gmail__create_draft": ("Drafting an email",
                            "Draft saved to Gmail", None),
    "outlook__create_draft": ("Drafting an email",
                              "Draft saved to Outlook", None),
    "jira__add_comment": ("Commenting on Jira", "Commented on Jira", None),
    "jira__create_issue": ("Creating a Jira issue",
                           "Created a Jira issue", None),
    "github__create_comment": ("Commenting on GitHub",
                               "Commented on GitHub", None),
    "notion__create_page": ("Creating a Notion page",
                            "Created a Notion page", None),
    "docs__append_text": ("Updating the doc", "Updated the doc", None),
    "stub__post": ("Posting to the test channel",
                   "Posted to the test channel", None),
}

# Engine phases — the agent's own work, branded as the orb.
_PHASE_VERBS: dict[str, tuple[str, str]] = {
    "evaluate": ("Checking triggers", "Checked triggers"),
    "prepare": ("Composing", "Composed"),
    "record": ("Wrapping up", "Done"),
    "compose": ("Composing", "Composed"),
    "deliver": ("Delivering", "Delivered"),
    # R38 — the `agent` spec step (spec_v2). Not a connector call and
    # not an engine housekeeping phase: it is the run stopping to work
    # something out. Branded as the orb, like every other piece of the
    # agent's own work.
    "think": ("Thinking it through", "Thought it through"),
}

_DONE_STATUSES = frozenset({"done", "completed", "success"})


def step_verb(
    tool: Optional[str],
    connector_id: Optional[str] = None,
    *,
    phase: Optional[str] = None,
    status: str = "pending",
    count: Optional[int] = None,
) -> dict:
    """`{"label": str, "brand": str | None}` for one run step.

    `phase` names engine work (evaluate/prepare/record/…) → orb brand.
    Connector steps brand as their connector. Total: any unknown input
    yields a safe generic phrase, never the raw name.
    """
    done = status in _DONE_STATUSES
    if phase:
        doing, finished = _PHASE_VERBS.get(
            phase, ("Working", "Finished"),
        )
        return {"label": finished if done else doing, "brand": BRAND_ORB}

    brand: Optional[str] = None
    if connector_id and connector_id in _CONNECTOR_NAMES:
        brand = connector_id
    elif isinstance(tool, str) and "__" in tool:
        prefix = tool.split("__", 1)[0]
        if prefix in _CONNECTOR_NAMES:
            brand = prefix
        elif connector_id and display_name(connector_id):
            brand = connector_id
    elif connector_id and display_name(connector_id):
        brand = connector_id

    entry = _TOOL_VERBS.get(tool or "")
    if entry:
        doing, finished, with_count = entry
        if done and count is not None and with_count:
            return {"label": with_count.format(count=count), "brand": brand}
        return {"label": finished if done else doing, "brand": brand}

    name = display_name(brand or connector_id)
    if name:
        label = f"Finished with {name}" if done else f"Working with {name}"
    else:
        label = "Finished" if done else "Working"
    return {"label": label, "brand": brand}


# ── Schedules ────────────────────────────────────────────────────────

_CRON_RE = re.compile(
    r"^\s*(\d{1,2})\s+(\d{1,2})\s+\*\s+\*\s+([\d,\-*]+)\s*$"
)
_DOW_NAMES = {
    "0": "Sundays", "1": "Mondays", "2": "Tuesdays", "3": "Wednesdays",
    "4": "Thursdays", "5": "Fridays", "6": "Saturdays", "7": "Sundays",
}


def _hhmm(hour: int, minute: int) -> str:
    return f"{hour}:{minute:02d}"


def _classify_schedule(sched: dict) -> Optional[tuple[str, str]]:
    """(short, clause) for one schedule dict, or None. The clause is
    the sentence-leading form rule_sentence uses."""
    if not isinstance(sched, dict):
        return None
    every = sched.get("every_s")
    if isinstance(every, (int, float)) and every > 0:
        s = int(every)
        if s % 3600 == 0 and s >= 3600:
            h = s // 3600
            short = "hourly" if h == 1 else f"every {h} hours"
        elif s % 60 == 0 and s >= 60:
            m = s // 60
            short = "every minute" if m == 1 else f"every {m} minutes"
        else:
            short = f"every {s} seconds"
        return short, short.capitalize()
    at = sched.get("at")
    if isinstance(at, str) and re.fullmatch(r"\d{1,2}:\d{2}", at.strip()):
        t = at.strip()
        return f"daily {t}", f"Every day at {t}"
    cron = sched.get("cron_local")
    if isinstance(cron, str):
        m = _CRON_RE.match(cron)
        if m:
            minute, hour, dow = int(m.group(1)), int(m.group(2)), m.group(3)
            t = _hhmm(hour, minute)
            if dow == "*":
                return f"daily {t}", f"Every day at {t}"
            if dow in ("1-5", "1,2,3,4,5"):
                return f"weekdays {t}", f"Every weekday at {t}"
            if dow in ("0,6", "6,0", "6,7", "0-6"):
                if dow == "0-6":
                    return f"daily {t}", f"Every day at {t}"
                return f"weekends {t}", f"On weekends at {t}"
            if dow in _DOW_NAMES:
                day = _DOW_NAMES[dow]
                return f"{day} {t}", f"{day} at {t}"
        return "on a custom schedule", "On its schedule"
    return None


def schedule_human(trigger_or_source: Any) -> Optional[str]:
    """Short human cadence ("weekdays 8:00") for a v1 trigger, a v2
    source, a bare schedule dict, or a whole spec. Never a cron
    string; None when there is no schedule."""
    sched = _find_schedule(trigger_or_source)
    if sched is None:
        return None
    got = _classify_schedule(sched)
    return got[0] if got else None


def _find_schedule(obj: Any) -> Optional[dict]:
    if not isinstance(obj, dict):
        return None
    if any(k in obj for k in ("cron_local", "at", "every_s")):
        return obj
    if isinstance(obj.get("schedule"), dict):
        return obj["schedule"]
    trig = obj.get("trigger")
    if isinstance(trig, dict):
        if isinstance(trig.get("schedule"), dict):
            return trig["schedule"]
        for src in trig.get("sources") or []:
            if isinstance(src, dict) and isinstance(src.get("schedule"), dict):
                return src["schedule"]
    return None


# ── Event clauses (rule_sentence) ────────────────────────────────────

_EVENT_CLAUSES = {
    "email_received": "when a new email arrives",
    "issue_created": "when a new Jira issue appears",
    "issue_opened": "when a GitHub issue opens",
    "chat_message_received": "when a new Teams message arrives",
    "event_created": "when a calendar event is added",
    "file_added": "when a new file appears",
    "page_added": "when a new page appears",
    "item_created": "when a new item appears",
}

# event key → the template card's cadence tag ("on new PRs" energy).
_EVENT_TAGS = {
    "email_received": "on new email",
    "issue_created": "on new Jira issues",
    "issue_opened": "on new GitHub issues",
    "chat_message_received": "on new messages",
    "event_created": "on new calendar events",
    "file_added": "on new files",
    "page_added": "on new pages",
    "item_created": "on new items",
}


def event_tag(event_key: Optional[str]) -> Optional[str]:
    """Cadence tag for an event-triggered template; a safe generic for
    unknown keys, None only when there is no event at all."""
    if not event_key:
        return None
    return _EVENT_TAGS.get(event_key, "on new activity")


# write tool → what the automation does for the user.
_WRITE_CLAUSES = {
    "slack__send_message": "post to Slack",
    "teams__send_chat_message": "post to Teams",
    "gmail__create_draft": "draft an email for you",
    "outlook__create_draft": "draft an email for you",
    "jira__add_comment": "comment on Jira",
    "jira__create_issue": "create a Jira issue",
    "github__create_comment": "comment on GitHub",
    "notion__create_page": "add a Notion page",
    "docs__append_text": "update your doc",
    "stub__post": "post to the test channel",
}


def is_write_tool(tool: Optional[str]) -> bool:
    """True when `tool` is a declared write action (R36-5).

    The manifests' `scopes_write_by_action` keys and this table are the
    same ten tools, and this one is importable without the registry —
    which is what the raw-spec derivations (`workflow._write_tools`,
    `service._permissive_registry_v2`) need: an UNGRANTED template
    write step carries no grant_id and no grant_target key, so grant
    presence cannot be the test for "is this a write". It was, and a
    from-template draft automation derived "reads only" on every
    surface while its draft step ran as a read straight into the
    dispatcher's grant gate.
    """
    return str(tool or "") in _WRITE_CLAUSES


def _trigger_clause(raw: dict) -> str:
    trig = raw.get("trigger") or {}
    sources = trig.get("sources")
    if isinstance(sources, list) and sources:
        sched = next(
            (s for s in sources
             if isinstance(s, dict) and s.get("mode") == "schedule"),
            None,
        )
        if sched is not None:
            got = _classify_schedule(sched.get("schedule") or {})
            if got:
                return got[1]
        first = next((s for s in sources if isinstance(s, dict)), None)
        if first is not None:
            clause = _EVENT_CLAUSES.get(first.get("event") or "")
            if clause:
                name = display_name(first.get("connector_id"))
                return (f"{clause[0].upper()}{clause[1:]} in {name}"
                        if name else clause.capitalize())
        return "When it fires"
    if isinstance(trig.get("schedule"), dict):
        got = _classify_schedule(trig["schedule"])
        if got:
            return got[1]
    clause = _EVENT_CLAUSES.get(trig.get("event") or "")
    if clause:
        name = display_name(trig.get("connector_id"))
        return (f"{clause[0].upper()}{clause[1:]} in {name}"
                if name else clause.capitalize())
    return "When it fires"


def rule_sentence(raw: dict) -> Optional[str]:
    """The standing rule as one plain sentence, from a canonical spec
    dict — "Every weekday at 8:00, check Jira and GitHub and post to
    Slack." Never raw tool names; None only on a shapeless spec."""
    if not isinstance(raw, dict):
        return None
    reads: list[str] = []
    writes: list[str] = []
    if raw.get("version") == 2:
        for s in raw.get("steps") or []:
            if not isinstance(s, dict):
                continue
            if s.get("grant_id") or s.get("grant_target") is not None:
                writes.append(
                    _WRITE_CLAUSES.get(s.get("tool") or "")
                    or "take the action you approved"
                )
            else:
                name = display_name(s.get("connector_id"))
                if name and name not in reads:
                    reads.append(name)
    else:
        action = raw.get("action") or {}
        if isinstance(action, dict) and action.get("tool"):
            writes.append(
                _WRITE_CLAUSES.get(action.get("tool") or "")
                or "take the action you approved"
            )
    trigger = _trigger_clause(raw)
    parts: list[str] = []
    if reads:
        if len(reads) == 1:
            parts.append(f"check {reads[0]}")
        else:
            parts.append("check " + ", ".join(reads[:-1]) + f" and {reads[-1]}")
    seen_writes: list[str] = []
    for w in writes:
        if w not in seen_writes:
            seen_writes.append(w)
    parts.extend(seen_writes)
    if not parts:
        return None
    if len(parts) == 1:
        body = parts[0]
    else:
        body = ", ".join(parts[:-1]) + f" and {parts[-1]}"
    return f"{trigger}, {body}."


# ── Outcomes ─────────────────────────────────────────────────────────

_TONES = {
    "sent": "ok",
    "partial": "warn",
    "undone": "warn",
    "skipped": "warn",
    # R30's two new terminals. ND-14 (live): they were added to the
    # engine's vocabulary but never to this table, so `tone_for` hit the
    # "err" default and the card painted a user's own Stop in the danger
    # tint while the thread beside it said "You stopped it. Nothing was
    # sent." A stop is not a failure — it is the user getting what they
    # asked for, which is exactly why `undone` (also user-initiated)
    # already sits at "warn".
    "stopped": "warn",
    "superseded": "warn",
}


def tone_for(outcome: Optional[str]) -> str:
    """ok | warn | err for a terminal outcome — the pill/tint contract."""
    return _TONES.get(outcome or "", "err")


def outcome_sentence(
    outcome: Optional[str],
    *,
    write_tool: Optional[str] = None,
    connector_id: Optional[str] = None,
    counts: Optional[dict] = None,
    wrote_count: int = 0,
    error: Optional[str] = None,
) -> dict:
    """`{"sentence": str, "tone": "ok"|"warn"|"err"}` for one terminal
    outcome. `counts` maps connector_id → collected count (rendered as
    "Jira 4, Gmail 2"); `wrote_count` > 1 appends the extra actions."""
    tone = _TONES.get(outcome or "", "err")

    def _with_counts(base: str) -> str:
        pieces = []
        for cid, n in (counts or {}).items():
            name = display_name(cid)
            if name is not None and isinstance(n, int):
                pieces.append(f"{name} {n}")
        if pieces:
            base = f"{base} — {', '.join(pieces)}"
        return base

    if outcome == "sent":
        done = step_verb(write_tool, connector_id, status="done")["label"] \
            if write_tool else "Completed"
        if wrote_count > 1:
            done = f"{done} (+{wrote_count - 1} more)"
        return {"sentence": _with_counts(done) + ".", "tone": tone}
    if outcome == "partial":
        done = step_verb(write_tool, connector_id, status="done")["label"] \
            if write_tool else "Ran"
        return {
            "sentence": f"{done} — some sources were unavailable.",
            "tone": tone,
        }
    if outcome == "undone":
        return {"sentence": "You undid the last run.", "tone": tone}
    if outcome == "skipped":
        return {"sentence": "Skipped — the confirmation expired.",
                "tone": tone}
    if outcome == "forbidden_tool":
        return {"sentence": "Blocked: automations never send mail — "
                            "use a draft action.", "tone": tone}
    if outcome == "write_failed":
        return {"sentence": "The write failed — nothing went out.",
                "tone": tone}
    if outcome == "step_failed":
        return {"sentence": "A step failed — nothing was sent.",
                "tone": tone}
    if outcome == "run_cap":
        return {"sentence": "The last run took too long and was stopped.",
                "tone": tone}
    if outcome == "lost":
        return {"sentence": "The last run was interrupted.", "tone": tone}
    if outcome == "stopped":
        # The card must not contradict the thread. `wrote_count` here is
        # the HONEST write-ledger count the caller passes for a stop —
        # never the spec's write-step count, which would claim a change
        # the stop prevented.
        n = int(wrote_count or 0)
        if n <= 0:
            return {"sentence": "You stopped it. Nothing was sent.",
                    "tone": tone}
        return {
            "sentence": f"You stopped it. {n} change"
                        + ("" if n == 1 else "s") + " already made.",
            "tone": tone,
        }
    if outcome == "superseded":
        return {"sentence": "You stopped it, and the next run has taken "
                            "over.", "tone": tone}
    return {"sentence": "The last run didn't complete.", "tone": tone}


def fix_chip(
    automation_name: str,
    outcome: Optional[str],
    error: Optional[str] = None,
) -> dict:
    """The failed-run chip: tapping sends `prompt` as the user's turn
    into the automation's session."""
    why = outcome_sentence(outcome, error=error)["sentence"]
    name = " ".join(str(automation_name or "this automation").split())[:80]
    return {
        "label": "Fix this",
        "prompt": (
            f'My automation "{name}" failed on its last run — {why} '
            f"Look into what went wrong and fix it, or tell me what you "
            f"need from me."
        ),
    }


# =====================================================================
# v2 (R30) — per-connector turn phrasings. CONTRACTS-R30 §2.
#
# Ownership split: A owns the STRUCTURE, fallbacks, and the rejection
# predicate; C owns the per-connector ENTRY STRINGS below (refine in
# place — every function stays total regardless of table contents).
# =====================================================================

# connector → (read action, count-detail template). `{n}` is the item
# count; the "(s)" marker is singularised by _n() below.
_V2_READ: dict[str, tuple[str, str]] = {
    "gmail": ("Read your unread mail", "{n} new thread(s)"),
    "outlook": ("Read your Outlook mail", "{n} new message(s)"),
    "slack": ("Read your channels", "{n} place(s)"),
    # R31-07: "moved" is something the automation DID. It read.
    "jira": ("Checked your board", "{n} open issue(s)"),
    "github": ("Checked your repositories", "{n} pull request(s)"),
    "teams": ("Read your Teams chats", "{n} new message(s)"),
    "notion": ("Checked your pages", "{n} page(s) changed"),
    "drive": ("Checked your files", "{n} new file(s)"),
    "docs": ("Read the document", ""),
    "calendar": ("Read your week", "{n} event(s)"),
    "stub": ("Checked the test feed", "{n} item(s)"),
}

# connector → progressive live-pill / step-sentence form.
_V2_READ_LIVE: dict[str, str] = {
    "gmail": "reading {n} unread messages",
    "outlook": "reading {n} Outlook messages",
    "slack": "reading your channels",
    "jira": "checking your board",
    "github": "checking your repositories",
    "teams": "reading your Teams chats",
    "notion": "checking your pages",
    "drive": "checking your files",
    "docs": "reading the document",
    "calendar": "reading your week",
    "stub": "checking the test feed",
}

# write tool → phrasing. `action` is the you-audience form;
# `action_others` names the real target (`{target}` interpolated).
_V2_WRITE: dict[str, dict[str, str]] = {
    "slack__send_message": {
        "action": "Told you in Slack", "action_others": "Posted in {target}",
        "detail": "one line, no thread",
    },
    "teams__send_chat_message": {
        "action": "Told you in Teams", "action_others": "Posted in {target}",
        "detail": "one line, no thread",
    },
    "gmail__create_draft": {
        "action": "Drafted a reply", "action_others": "Drafted a reply",
        "detail": "waiting in Gmail — nothing sent",
    },
    "outlook__create_draft": {
        "action": "Drafted a reply", "action_others": "Drafted a reply",
        "detail": "waiting in Outlook — nothing sent",
    },
    "jira__add_comment": {
        "action": "Commented on your ticket",
        "action_others": "Commented on {target}",
        "detail": "one comment, nothing else touched",
    },
    "jira__create_issue": {
        "action": "Filed a ticket", "action_others": "Filed a ticket in {target}",
        "detail": "created, not assigned",
    },
    "github__create_comment": {
        "action": "Commented on the pull request",
        "action_others": "Commented on {target}",
        "detail": "a comment, nothing merged",
    },
    "notion__create_page": {
        "action": "Added a page", "action_others": "Added a page in {target}",
        "detail": "a new page, nothing overwritten",
    },
    "docs__append_text": {
        "action": "Added to the doc", "action_others": "Added to {target}",
        "detail": "appended, nothing removed",
    },
    "stub__post": {
        "action": "Posted to the test channel",
        "action_others": "Posted in {target}",
        "detail": "test write",
    },
}

# failure reason → (action, detail).
_V2_FAILURE: dict[str, tuple[str, str]] = {
    "reauth_required": ("Could not connect", "access expired"),
    "scope_missing": ("Could not connect", "it needs more access than you gave it"),
    "provider_down": ("Could not connect", "the service was unreachable"),
    "rate_limited": ("Could not connect", "asked to slow down"),
    "timeout": ("Could not connect", "it did not answer in time"),
}
_V2_FAILURE_DEFAULT = ("Could not connect", "it did not answer")

# event-trigger node sub (§3.9): connector or "connector:event" key.
_V2_TRIGGER_SUB: dict[str, str] = {
    "gmail": "when mail arrives",
    "outlook": "when mail arrives",
    "slack": "when a message lands",
    "jira": "when an issue appears",
    "github": "when a pull request moves",
    "teams": "when a chat message lands",
    "notion": "when a page changes",
    "calendar": "when your week changes",
    "drive": "when a file lands",
    "stub": "when a test item appears",
}

# Engine phases that produce a THREAD TURN of their own: (done, failed).
# R38 — an `agent` spec step is recorded like any other step, so its
# sentence has to come from here rather than from the executor. The
# served-action set below is DERIVED from this table on purpose: an
# action `engine_action` can emit that `is_served_action` would refuse
# is a turn the ledger silently rewrites into a bare agent bubble, and
# a hand-maintained second list is exactly how that drift happens.
_V2_ENGINE_PHASE: dict[str, tuple[str, str]] = {
    "think": ("Thought it through", "Could not think it through"),
}
_V2_ENGINE_PHASE_DEFAULT = ("Finished a step", "Could not finish a step")

# Engine-authored (non-connector) actions the thread can carry.
_V2_ENGINE_ACTIONS = frozenset(
    {
        "Checked what I can do",   # setup capability check (C §5.3)
        "Connected again",         # the reconnect catch-up turn (§4.7)
    }
    | {a for pair in _V2_ENGINE_PHASE.values() for a in pair}
    | set(_V2_ENGINE_PHASE_DEFAULT)
)


def engine_action(phase: str, *, ok: bool = True) -> dict:
    """`{"action", "detail"}` for one turn of the engine's OWN work.

    No connector, no tool, no target — the counterpart of `turn_action`
    for a step the agent performs itself. Total: an unknown phase gets
    the generic pair, never a raw phase name, and every string it can
    return is in `_V2_ENGINE_ACTIONS` above — including the fallback,
    because an unserved action is a turn the ledger rewrites away.
    """
    done, failed = _V2_ENGINE_PHASE.get(phase, _V2_ENGINE_PHASE_DEFAULT)
    return {"action": done if ok else failed, "detail": ""}


_SLOT_RE = re.compile(r"\{[a-z_][a-z0-9_]*\}")


def _n(template: str, count: Optional[int], **extra: Any) -> str:
    """Interpolate `{n}`/`{count}` (+ any `extra`) and singularise `(s)`.

    R31-25. This used to substitute exactly `{n}` and `{count}` and pass
    everything else through verbatim, so `{count} issues moved ·
    {need_count} needs you` reached a user's job sheet with the second
    brace SHOWING — and `is_served_action` could not catch it either,
    because C's templates are compiled to regexes with `{slot}` → `.+?`
    and `{need_count}` matches `.+?`.

    The entries module's own docstring already promised the right
    behaviour — "a template with an unfilled slot renders without its
    clause, never with braces showing" — and nothing implemented it.
    This does: fill what we can, then DROP the clause around anything
    left. Dropping is right rather than guessing, because the alternative
    to a missing clause is an invented number.
    """
    if not template:
        return ""
    n = 0 if count is None else int(count)
    out = template.replace("{n}", str(n)).replace("{count}", str(n))
    for key, value in (extra or {}).items():
        if value is None or value == "":
            continue
        out = out.replace("{" + key + "}", str(value))
    out = out.replace("(s)", "" if n == 1 else "s")
    return drop_unfilled(out, _origin=template)


def drop_unfilled(text: str, *, _origin: str = "") -> str:
    """Return `text` with any clause that still carries a slot removed.

    Extracted from `_n` so the SAME rule can run at read time. `_n` fixes
    what the engine mints from now on; it cannot fix a string a previous
    build already persisted — and `workflow._last_use` serves the stored
    `detail` of a tool turn verbatim, so the founder's Jira card was still
    reading `Checked your board · 0 issues moved · {need_count} needs you`
    on a build whose renderer could no longer produce it.

    A surface that can be reached by data written before the fix needs the
    rule at the boundary it reads, not only at the one it writes.
    """
    out = text or ""
    if not _SLOT_RE.search(out):
        return out
    # ` · ` is the dictionary's own clause separator; a comma is the
    # fallback; a string that is ONE clause with an unfillable slot says
    # nothing rather than showing a brace.
    for sep in (" · ", ", "):
        if sep in out:
            kept = [c for c in out.split(sep) if not _SLOT_RE.search(c)]
            out = sep.join(kept)
            if not _SLOT_RE.search(out):
                return out.strip()
    logger.warning("automation.copy.unfilled template=%r", (_origin or text)[:120])
    return ""


# Past-tense verbs that assert the automation CHANGED something. A read
# step's detail may never wear one: the whole difference between "I
# looked and nothing had changed" and "I changed nothing" is which of
# them the user has to act on.
_WRITE_VOICE_RE = re.compile(
    r"\b(moved|sent|posted|drafted|created|deleted|updated|archived"
    r"|replied|commented|filed|scheduled|assigned|closed|merged)\b",
    re.IGNORECASE,
)


def _is_write_voiced(detail: str) -> bool:
    return bool(detail) and bool(_WRITE_VOICE_RE.search(detail))


_C_ENTRIES_CACHE: Any = None


def _c_entries():
    """C's per-connector entry module (`automation_verb_entries.py`,
    same package so both images can import it) — overlaid when present,
    never required: every v2 function keeps its built-in fallback so
    totality never depends on that module importing."""
    global _C_ENTRIES_CACHE
    if _C_ENTRIES_CACHE is None:
        try:
            from app.services import automation_verb_entries as entries
            _C_ENTRIES_CACHE = entries
        except Exception:  # noqa: BLE001
            _C_ENTRIES_CACHE = False
    return _C_ENTRIES_CACHE or None


def _c_entry(connector_id: str) -> Optional[dict]:
    e = _c_entries()
    if e is None:
        return None
    return (getattr(e, "ENTRIES", {}) or {}).get(connector_id)


def turn_action(
    connector_id: Optional[str],
    tool: Optional[str] = None,
    *,
    kind: str = "read",
    ok: bool = True,
    count: Optional[int] = None,
    target: Optional[str] = None,
    audience: str = "you",
) -> dict:
    """`{"action", "detail"}` for one v3 tool turn. Total."""
    cid = connector_id or ""
    if not ok:
        return failure_action(cid, "reauth_required" if not tool else None)
    ce = _c_entry(cid)
    if kind == "write":
        entry = None
        if ce:
            entry = (ce.get("writes") or {}).get(tool or "")
        entry = entry or _V2_WRITE.get(tool or "")
        if entry:
            # Every slot on BOTH halves, not just `action_others`.
            # `Commented on {target}`, `Filed {target}`, `Held {when}`
            # and `in {target}, nothing overwritten` all reached users
            # with braces showing, because only `action_others` was ever
            # interpolated and only with `{target}`/`{channel}`.
            slots = {"target": target, "channel": target, "when": target}
            if audience == "others" and target and entry.get("action_others"):
                action = _n(entry["action_others"], count, **slots)
            else:
                action = _n(entry["action"], count, **slots)
            detail = _n(entry.get("detail") or "", count, **slots)
            if not action:
                name = display_name(cid) or "the account"
                action = f"Made a change in {name}"
            return {"action": action, "detail": detail}
        name = display_name(cid) or "the account"
        return {"action": f"Made a change in {name}", "detail": ""}
    if ce:
        reads = ce.get("reads") or {}
        entry_c = reads.get(tool or "") or reads.get("*")
        if entry_c:
            detail = (
                _n(entry_c.get("detail") or "", count)
                if count is not None else ""
            )
            if _is_write_voiced(detail):
                # R31-07: a READ step reporting `0 issues moved`. The
                # count is real (issues that changed since the last
                # look), the verb is not — "moved" is something the
                # automation did, and it did nothing. C owns the string;
                # A owns not shipping it. Fall back to this module's own
                # read phrasing and name the entry so C can fix it once.
                logger.warning(
                    "automation.copy.write_voiced_read connector=%s "
                    "tool=%s detail=%r", cid, tool, detail[:80],
                )
                builtin = _V2_READ.get(cid)
                if builtin:
                    return {"action": entry_c["action"],
                            "detail": _n(builtin[1], count)}
                detail = _n("{n} item(s)", count)
            return {"action": entry_c["action"], "detail": detail}
    entry = _V2_READ.get(cid)
    if entry:
        return {"action": entry[0], "detail": _n(entry[1], count)}
    name = display_name(cid) or "the account"
    return {"action": f"Checked {name}", "detail": _n("{n} item(s)", count) if count else ""}


def live_sentence(
    connector_id: Optional[str], tool: Optional[str] = None,
    count: Optional[int] = None, *, phase: Optional[str] = None,
) -> str:
    """Progressive form for the live pill / step sentence. Total.

    `phase` names the engine's OWN work (R38's `agent` spec step is
    "think") — there is no connector to name, and `live_sentence(None,
    None)` would answer the bare "working" that says nothing.
    """
    if phase:
        doing, _finished = _PHASE_VERBS.get(phase, ("Working", "Finished"))
        return _lower_first(doing)
    cid = connector_id or ""
    ce = _c_entry(cid)
    if ce:
        for table in (ce.get("reads") or {}, ce.get("writes") or {}):
            entry = table.get(tool or "") or (
                table.get("*") if table is ce.get("reads") else None
            )
            if entry and entry.get("progressive"):
                form = entry["progressive"]
                if ("{count}" in form or "{n}" in form) and count is None:
                    form = form.replace(" {count}", "").replace(
                        "{count} ", "").replace(" {n}", "").replace("{n} ", "")
                    return form
                return _n(form, count)
    form = _V2_READ_LIVE.get(cid)
    if form:
        if "{n}" in form and count is None:
            # No count yet — drop the number, keep the verb.
            form = form.replace(" {n}", "").replace("{n} ", "")
            return form
        return _n(form, count)
    name = display_name(cid)
    return f"checking {name}" if name else "working"


_C_FAILURE_ALIASES = {
    # This module's reason vocabulary → C's entry keys where they differ.
    #
    # `scope_missing` USED TO ALIAS ONTO `access_expired`, and that one
    # line is why the founder was told his Outlook "access expired" when
    # the truth was that the connection had never been granted mail-read
    # scope. Two different causes, two different repairs: reconnecting an
    # expired token fixes nothing when the scope was never asked for, so
    # the user does the OAuth dance and lands exactly where they started.
    # The built-in table had the right sentence all along ("it needs more
    # access than you gave it"); this alias made it unreachable.
    #
    # `timeout → provider_down` stays: "the service was unreachable" and
    # "it did not answer in time" are the same event from the user's
    # side, and C's table has one entry for it.
    "reauth_required": "access_expired",
    "timeout": "provider_down",
}


def failure_action(
    connector_id: Optional[str], reason: Optional[str] = None,
) -> dict:
    e = _c_entries()
    if e is not None:
        table = getattr(e, "V2_FAILURE", {}) or {}
        entry = table.get(reason or "") or \
            table.get(_C_FAILURE_ALIASES.get(reason or "", ""))
        if isinstance(entry, dict) and entry.get("action"):
            name = display_name(connector_id) or "the account"
            return {
                "action": entry["action"].replace("{name}", name),
                "detail": (entry.get("detail") or "").replace("{name}", name),
            }
    action, detail = _V2_FAILURE.get(reason or "", _V2_FAILURE_DEFAULT)
    return {"action": action, "detail": detail}


def trigger_sub(
    connector_id: Optional[str], event_key: Optional[str] = None,
) -> str:
    cid = connector_id or ""
    ce = _c_entry(cid)
    if ce and ce.get("trigger_sub"):
        return ce["trigger_sub"]
    if event_key and f"{cid}:{event_key}" in _V2_TRIGGER_SUB:
        return _V2_TRIGGER_SUB[f"{cid}:{event_key}"]
    if cid in _V2_TRIGGER_SUB:
        return _V2_TRIGGER_SUB[cid]
    return "when something changes"


def _lower_first(s: str) -> str:
    return s[:1].lower() + s[1:] if s else s


def job_card_label(group: list[dict], failed_accounts: Optional[list] = None) -> str:
    """The job card's label from its grouped tool turns (§3.4/§4.4).

    R31-07. This returned the literal string `Could not reach an
    account` — no name, no count, no reason — which is what a founder
    saw at the top of a run that had failed on GitHub AND Outlook while
    Jira and Gmail answered. It named nobody, so the card could not be
    acted on and the sheet had to be opened to learn anything at all.

    Now: every name, at any count, from C's `could_not_reach_*` forms.

    And a run whose `accounts_failed` is EMPTY can never wear a failure
    label (ND-16's rule, applied to the second reader): pass
    `failed_accounts` and the two are asserted to agree, so a run that
    died mid-"Wrapping up" with no connector recorded says "it did not
    finish" rather than accusing an account that never refused.
    """
    turns = [t for t in group or [] if t]
    failed_here = [t for t in turns if not t.get("ok", True)]
    if failed_here:
        ids = []
        for t in failed_here:
            acc = t.get("account_id") or ""
            if acc and acc not in ids:
                ids.append(acc)
        if failed_accounts is not None:
            allowed = set(failed_accounts)
            ids = [i for i in ids if i in allowed]
        names = [display_name(i) or i for i in ids]
        if names:
            from app.agent.automations import account_health as _ah
            label = _ah.names_sentence(names, prefix="could_not_reach")
            if label:
                return label
            joined = _ah.join_names(names)
            return f"Could not reach {joined}" if joined else ""
        return "It did not finish"
    writes = [t for t in turns if t.get("tool_kind") == "write"]
    if writes:
        parts = [writes[0].get("action") or "Made a change"]
        parts += [_lower_first(t.get("action") or "made a change")
                  for t in writes[1:]]
        return " · ".join(parts)
    accounts = {t.get("account_id") for t in turns if t.get("account_id")}
    n = max(1, len(accounts))
    return f"Checked {n} account" + ("" if n == 1 else "s")


def sheet_subtitle(group: list[dict], run: Optional[dict] = None) -> str:
    """The job sheet's per-group subtitle grammar (§3.5). Total.

    `run` may carry {"status", "stopped_at_step", "failed_whole_run",
    "unreachable": [connector display names]}.
    """
    run = run or {}
    turns = [t for t in group or [] if t]
    parts: list[str] = []
    failed_here = any(not t.get("ok", True) for t in turns)
    writes = [t for t in turns if t.get("tool_kind") == "write"
              and t.get("ok", True)]
    if failed_here:
        parts.append("Stopped before it finished")
    elif writes:
        changes = len(writes)
        msgs_you = sum(1 for t in writes
                       if (t.get("audience") or "you") == "you")
        clauses = [f"{changes} change" + ("" if changes == 1 else "s")]
        if msgs_you:
            clauses.append(
                f"{msgs_you} message" + ("" if msgs_you == 1 else "s")
                + " to you"
            )
        named_others = [
            t.get("target") for t in writes
            if (t.get("audience") or "you") == "others" and t.get("target")
        ]
        if named_others:
            clauses.append(f"posted in {named_others[0]}")
        else:
            clauses.append("nothing sent to anyone else")
        parts.append(" · ".join(clauses))
    else:
        parts.append("Nothing was sent or changed")
    for name in run.get("unreachable") or []:
        parts.append(f"Could not reach {name}")
    if run.get("stopped_at_step"):
        parts.append(f"stopped by you at step {int(run['stopped_at_step'])}")
    return " · ".join(parts)


def _served_action_patterns() -> list[re.Pattern]:
    pats: list[re.Pattern] = []
    for entry in _V2_WRITE.values():
        for key in ("action", "action_others"):
            tmpl = entry.get(key) or ""
            pats.append(re.compile(
                re.escape(tmpl).replace(r"\{target\}", ".+?") + r"\Z"
            ))
    return pats


_SERVED_EXACT: frozenset = frozenset(
    {a for a, _ in _V2_READ.values()}
    | {a for a, _ in _V2_FAILURE.values()}
    | {_V2_FAILURE_DEFAULT[0]}
    | _V2_ENGINE_ACTIONS
)
_SERVED_PATTERNS = _served_action_patterns()
_SERVED_GENERIC_RE = re.compile(
    r"(Checked|Made a change in|Could not reach) [A-Z][\w ]{0,40}\Z"
)


_C_SERVED_CACHE: Optional[tuple] = None


def _c_served() -> tuple:
    """(exact_set, patterns) built from C's entries — cached once."""
    global _C_SERVED_CACHE
    if _C_SERVED_CACHE is not None:
        return _C_SERVED_CACHE
    exact: set = set()
    patterns: list = []

    def _add(tmpl: str) -> None:
        if not tmpl:
            return
        if "{" in tmpl:
            patterns.append(re.compile(
                re.sub(r"\\\{[a-z_]+\\\}", ".+?", re.escape(tmpl)) + r"\Z"
            ))
        else:
            exact.add(tmpl)

    e = _c_entries()
    if e is not None:
        for entry in (getattr(e, "ENTRIES", {}) or {}).values():
            for table_name in ("reads", "writes"):
                for t in (entry.get(table_name) or {}).values():
                    _add(t.get("action") or "")
                    _add(t.get("action_others") or "")
        for f in (getattr(e, "V2_FAILURE", {}) or {}).values():
            if isinstance(f, dict):
                _add(f.get("action") or "")
        exact.update(getattr(e, "V2_ENGINE_ACTIONS", ()) or ())
    _C_SERVED_CACHE = (frozenset(exact), tuple(patterns))
    return _C_SERVED_CACHE


def is_served_action(action: Any) -> bool:
    """True iff this module could have emitted `action` — the v3
    serializer's rejection predicate (CONTRACTS-R30 §1). Total.

    R31-25: an UNFILLED SLOT is never served. This predicate used to
    accept `Held {when}` and `Commented on {target}` because C's
    templates are compiled to regexes with `{slot}` → `.+?`, and a
    literal `{target}` matches `.+?` perfectly. So the one check that
    was supposed to stop raw strings reaching the thread was, for
    exactly this class, an identity function.
    """
    if not isinstance(action, str) or not action.strip():
        return False
    if "__" in action:
        return False
    if _SLOT_RE.search(action):
        return False
    if action in _SERVED_EXACT:
        return True
    if _SERVED_GENERIC_RE.match(action):
        return True
    if any(p.match(action) for p in _SERVED_PATTERNS):
        return True
    c_exact, c_patterns = _c_served()
    if action in c_exact:
        return True
    return any(p.match(action) for p in c_patterns)
