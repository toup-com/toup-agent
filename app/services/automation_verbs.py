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

import re
from typing import Any, Optional

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
