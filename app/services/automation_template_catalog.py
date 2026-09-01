"""The automation template catalog — code is the source of truth.

Round 28. ~28 product-curated templates across six categories
(work / email / code / calendar / school / personal), served from
`automation_templates` (the table is what the app and the setup agent
read; this module is what keeps the table current).

`sync_template_catalog(db)` upserts by slug on platform boot:
  - inserts rows for new slugs,
  - updates content columns (name/description/icon/connectors/spec/
    category/variables/sort_order) when they drift from the catalog,
  - NEVER touches `enabled` on an existing row — an admin kill-switch
    must survive deploys,
  - leaves rows whose slug is not in the catalog alone (admins may
    seed their own).

Template specs are SKELETONS: write actions carry no grant_id (grants
are per-user), and `{{var.<name>}}` placeholders reference the
declared `variables`. `test_automation_template_catalog.py` validates
every spec in template_mode against the REAL manifest-built registry —
a template naming a connector/event/tool that doesn't exist cannot
merge.

Hard rails the catalog respects by construction:
  - mail is only ever READ or DRAFTED (`gmail__create_draft`), never
    sent;
  - every write's pinned-target parameter is `{{grant.target.id}}` —
    a grant pins ONE target, so a template must never point a write at
    an event-derived target;
  - poll intervals are at/above the production floors.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _v(name: str, label: str, description: str, *, required: bool = True,
       example: str = "", default: str = "") -> dict:
    out: dict[str, Any] = {
        "name": name, "label": label, "description": description,
        "required": required,
    }
    if example:
        out["example"] = example
    if default:
        out["default"] = default
    return out


# ── The flagship: Morning Work Brief ─────────────────────────────────
#
# R42. Until this round the brief was five connector reads stitched by a
# STRING TEMPLATE — "*Jira ({{steps.issues.count}})*\n{{steps.issues.text}}"
# — and it published its own failures as facts. A read that fell over
# left `count` at 0 and `text` at "Could not read GitHub.", so the post
# said "GitHub (0)" directly above the sentence saying it never read
# GitHub; a step that never ran at all had no `steps.<id>` node, and
# `render_value` resolves a missing path to "", so an earlier post read
# "Teams ()" and "Outlook unread ()". The run also spoke twice in two
# vocabularies: the thread's result card ranked five groups under "Your
# morning, in order" while Slack got the raw connector dump.
#
# So the post interpolates ONE value now — an agent step's answer — and
# the count/text placeholders are gone with the arithmetic that produced
# them. The agent step's five headings are `narrator.BRIEF_GROUPS`
# word-for-word — the separators are ASCII hyphens here because this text
# is Slack copy, `-` beside `*bold*` — which is what makes the Slack post
# and the thread's result card one ranking of one run rather than two.
#
# What was dropped and why:
#   github  — its owner/repo variables are unanswerable at setup, and an
#             unanswered variable made the read fail and blame a healthy
#             account every weekday.
#   teams   — needs a chat id nobody can name at setup, and has no unread
#             filter.
#   outlook — Gmail already carries the mail; a second inbox bought a
#             second failure mode.
# Calendar JOINS: a morning brief without your meetings is not one, and
# it is the only step in the run that yields a real date.
#
# The template declares NO variables. Every param is a literal, so there
# is nothing for the setup thread to ask and nothing to leave unanswered;
# `needs_destination` (which Slack channel) is the only blocker, and the
# product already asks that well.

#: The ranking step's ask. Bounded by `spec_v2.AGENT_PROMPT_MAX_CHARS`;
#: rendered by `agent_step.build_prompt` against the same bounded view of
#: the steps that is printed beneath it, so `starts_at` below is the pins.
#:
#: It deliberately does NOT tell the model which address is the reader's.
#: The obvious rule — a row where you appear only in cc is not yours —
#: needs their own email, and nothing in a run knows it: `ctx["var"]` is
#: the spec's declared variables, and the only way to fill one is to ask
#: at setup, which is the unanswered-variable defect this template exists
#: to be rid of. The connector account label (`fetch_connection_state`'s
#: `account`) is the right source and reaching it is an executor change,
#: not a catalog one. A `{{var.my_email}}` here would render as "" and
#: leave the sentence saying "with  absent from to".
_MORNING_BRIEF_RANK = """\
Rank this morning's work for the reader. Order by what breaks if it is ignored, never by which app it came from.
The steps are the facts: cal is calendar entries; mail arrived in the last day; waiting is mail to you, unread, 1-7 days old; rooms is Slack naming you, newest first, no times; board is your live Jira. You have no clock: never write "today" or "yesterday", never work out a date, use only a date a line gives you.
Sections, headings exact, in this order, each omitted when it has nothing:
*DO FIRST - BLOCKS OTHERS* - at most 2: someone is stopped until this moves.
*ANSWER TODAY* - at most 3: a question to you, or work due on or before the earliest date cal gives.
*THIS WEEK* - at most 3: dated later, or owed and undated.
*NO ACTION - FOR AWARENESS* - at most 2: worth knowing, needs nothing.
*IGNORED - NOTHING NEEDED YOU* - one line: named categories with counts (cc only, automated, newsletters, chatter).
One item may appear in one section only. If no section has anything, write one line: "Nothing needed you this morning."
STANDING ORDERS: the material's starts_at names the people, channels and projects the reader pinned, with their own instructions. These RANK, never filter: a pinned name outranks others inside its section.
For a step whose ok is false, add a final line: "Missing: name it, I could not read it." Never report a step you could not read as a zero.
Never write a number you did not copy; the ignored counts you get by counting lines. Never write a double brace. Under 900 characters."""

_MORNING_WORK_BRIEF = {
    "version": 2,
    "name": "Morning work brief",
    "description": (
        "Every weekday morning, read what is next on your calendar, what "
        "landed overnight, what has been waiting on you, who named you in "
        "Slack and what is live on your Jira board — then post one ranked "
        "brief to a Slack channel."
    ),
    "mode": "auto",
    "variables": {},
    "trigger": {
        "sources": [
            {"id": "sched", "mode": "schedule",
             "schedule": {"cron_local": "0 8 * * 1-5"}},
        ],
    },
    "steps": [
        # The anchor: the only real date in the run, and what "due today"
        # is measured against — the ranking step has no clock.
        {"id": "cal", "connector_id": "calendar",
         "tool": "calendar__list_events",
         "params": {"window_days": 1, "max_results": 10},
         "collect": {"items_path": "events",
                     "fields": {"title": "summary",
                                "at": "start.dateTime",
                                "day": "start.date"},
                     # Google answers `dateTime` for a timed event and
                     # `date` for an all-day one, never both — the
                     # concatenation is what stops an all-day deadline
                     # rendering as a line with no date on it.
                     "format": "- {{item.at}}{{item.day}} {{item.title}}",
                     "limit": 10,
                     "empty_text": "Nothing on the calendar."},
         "on_error": "continue"},
        {"id": "mail", "connector_id": "gmail",
         "tool": "gmail__list_messages",
         "params": {"query": "in:inbox newer_than:1d", "max_results": 10},
         "collect": {"items_path": "messages",
                     "fields": {"from": "headers.From",
                                "to": "headers.To",
                                "cc": "headers.Cc",
                                "date": "headers.Date",
                                "subject": "headers.Subject",
                                "snippet": "snippet"},
                     "format": "- {{item.date}} · from {{item.from}} · "
                               "to {{item.to}} · cc {{item.cc}} · "
                               "{{item.subject}} — {{item.snippet}}",
                     "limit": 10,
                     "empty_text": "No new mail."},
         "on_error": "continue"},
        # A SEPARATE window rather than an age computed from `mail`:
        # membership of this step is the age, and the ranking step is
        # forbidden to do date arithmetic.
        {"id": "waiting", "connector_id": "gmail",
         "tool": "gmail__list_messages",
         "params": {"query": "to:me is:unread older_than:1d newer_than:7d",
                    "max_results": 10},
         "collect": {"items_path": "messages",
                     "fields": {"from": "headers.From",
                                "subject": "headers.Subject"},
                     "format": "- from {{item.from}} — {{item.subject}}",
                     "limit": 10,
                     "empty_text": "Nothing waiting on you."},
         "on_error": "continue"},
        # Search on the USER token sees DMs and every channel at once, so
        # there is no channel id to interview the user for.
        {"id": "rooms", "connector_id": "slack",
         "tool": "slack__search_messages",
         "params": {"query": "to:me", "sort": "timestamp", "count": 15},
         "collect": {"items_path": "matches",
                     "fields": {"from": "from",
                                "where": "channel_name",
                                "text": "text"},
                     "format": "- {{item.from}} in {{item.where}}: "
                               "{{item.text}}",
                     "limit": 15,
                     "empty_text": "Nobody named you."},
         "on_error": "continue"},
        # Dated, hot, or moved overnight — the rubric is evaluated
        # server-side, so the ranking step never has to date anything.
        {"id": "board", "connector_id": "jira",
         "tool": "jira__search_issues",
         "params": {"jql": "assignee = currentUser() AND statusCategory "
                           "!= Done AND (duedate <= 7d OR priority in "
                           "(Highest, High) OR updated >= -1d) ORDER BY "
                           "duedate ASC, priority DESC, updated DESC",
                    "max_results": 15},
         "collect": {"items_path": "issues",
                     "fields": {"key": "key", "summary": "summary",
                                "status": "status",
                                "priority": "priority",
                                "due": "duedate"},
                     "format": "- {{item.key}} [{{item.status}}] "
                               "[{{item.priority}}] {{item.due}} "
                               "{{item.summary}}",
                     "limit": 15,
                     "empty_text": "Nothing live on the board."},
         "on_error": "continue"},
        # Keeps the agent step's default on_error (`fail`): its answer is
        # the whole post, and a swallowed failure binds "" and publishes a
        # bare title.
        {"id": "rank", "kind": "agent",
         "prompt": _MORNING_BRIEF_RANK,
         "output_var": "brief"},
        {"id": "post", "connector_id": "slack",
         "tool": "slack__send_message",
         "params": {"channel": "{{grant.target.id}}",
                    "text": "*Morning brief*\n\n{{var.brief}}"}},
    ],
}

_MORNING_WORK_BRIEF_VARS: list[dict] = []


# ── Catalog ──────────────────────────────────────────────────────────
#
# sort_order: within a category, lower renders first; the flagship
# leads the whole list.

CATALOG: list[dict] = [
    # ═══ work ═══
    {
        "slug": "morning-work-brief",
        "name": "Morning work brief",
        "description": "One Slack post each weekday morning: your "
                       "calendar, your mail, who named you in Slack and "
                       "your Jira board, ranked by what breaks if you "
                       "ignore it.",
        "icon": "slack",
        "category": "work",
        "connectors": ["calendar", "gmail", "slack", "jira"],
        "variables": _MORNING_WORK_BRIEF_VARS,
        "spec": _MORNING_WORK_BRIEF,
        "sort_order": 0,
    },
    {
        # Adopted from the 095 seed so it stays current with the code.
        "slug": "jira-to-slack",
        "name": "Jira → Slack",
        "description": "Post a Slack message to a channel you pick "
                       "whenever a new Jira issue appears.",
        "icon": "jira",
        "category": "work",
        "connectors": ["jira", "slack"],
        "variables": [],
        "spec": {
            "name": "Jira → Slack",
            "trigger": {"mode": "poll", "connector_id": "jira",
                        "event": "issue_created",
                        "poll_interval_s": 300, "filter": {}},
            "action": {"connector_id": "slack",
                       "tool": "slack__send_message",
                       "params_template": {
                           "channel": "{{grant.target.id}}",
                           "text": "New Jira issue {{event.key}}: "
                                   "{{event.summary}} ({{event.url}})",
                       }},
            "dedupe_key": "event.key",
            "mode": "auto",
        },
        "sort_order": 1,
    },
    {
        "slug": "github-issue-to-jira",
        "name": "GitHub issue → Jira",
        "description": "File a Jira issue in a project you pick for "
                       "every new GitHub issue in a repository.",
        "icon": "github",
        "category": "work",
        "connectors": ["github", "jira"],
        "variables": [
            _v("github_owner", "GitHub owner", "Repo owner/org.",
               example="toup-com"),
            _v("github_repo", "GitHub repo", "Repository name."),
        ],
        "spec": {
            "version": 2,
            "name": "GitHub issue → Jira",
            "mode": "confirm",
            "trigger": {"sources": [
                {"id": "gh", "mode": "poll", "connector_id": "github",
                 "event": "issue_opened",
                 "params": {"owner": "{{var.github_owner}}",
                            "repo": "{{var.github_repo}}"},
                 "poll_interval_s": 600,
                 "dedupe_key": "event.number"},
            ]},
            "steps": [
                {"id": "file", "connector_id": "jira",
                 "tool": "jira__create_issue",
                 "params": {"project_key": "{{grant.target.id}}",
                            "summary": "GH #{{event.number}}: "
                                       "{{event.title}}",
                            "description": "{{event.url}}"}},
            ],
        },
        "sort_order": 2,
    },
    {
        "slug": "teams-chat-to-slack",
        "name": "Teams chat → Slack",
        "description": "Cross-post new messages from a Teams chat into "
                       "a Slack channel.",
        "icon": "teams",
        "category": "work",
        "connectors": ["teams", "slack"],
        "variables": [
            _v("teams_chat_id", "Teams chat", "The chat to watch."),
        ],
        "spec": {
            "version": 2,
            "name": "Teams chat → Slack",
            "mode": "auto",
            "trigger": {"sources": [
                {"id": "chat", "mode": "poll", "connector_id": "teams",
                 "event": "chat_message_received",
                 "params": {"chat_id": "{{var.teams_chat_id}}"},
                 "poll_interval_s": 300,
                 "dedupe_key": "event.id"},
            ]},
            "steps": [
                {"id": "post", "connector_id": "slack",
                 "tool": "slack__send_message",
                 "params": {"channel": "{{grant.target.id}}",
                            "text": "Teams — {{event.sender}}: "
                                    "{{event.body}}"}},
            ],
        },
        "sort_order": 3,
    },
    {
        "slug": "notion-page-alert",
        "name": "New Notion page → Slack",
        "description": "A Slack heads-up when a page first appears in "
                       "your Notion workspace.",
        "icon": "notion",
        "category": "work",
        "connectors": ["notion", "slack"],
        "variables": [],
        "spec": {
            "name": "New Notion page → Slack",
            "trigger": {"mode": "poll", "connector_id": "notion",
                        "event": "page_added", "poll_interval_s": 600,
                        "filter": {}},
            "action": {"connector_id": "slack",
                       "tool": "slack__send_message",
                       "params_template": {
                           "channel": "{{grant.target.id}}",
                           "text": "New Notion page: {{event.title}} "
                                   "({{event.url}})",
                       }},
            "dedupe_key": "event.id",
            "mode": "auto",
        },
        "sort_order": 4,
    },
    {
        "slug": "weekly-work-log",
        "name": "Weekly work log",
        "description": "Every Friday, append your week's Jira issues "
                       "to a Google Doc you pick.",
        "icon": "docs",
        "category": "work",
        "connectors": ["jira", "docs"],
        "variables": [
            _v("jira_jql", "Jira filter", "JQL for the week's issues.",
               default="assignee = currentUser() AND updated >= -7d "
                       "ORDER BY updated DESC"),
        ],
        "spec": {
            "version": 2,
            "name": "Weekly work log",
            "mode": "auto",
            "narration": {
                "style": "digest",
                "title": 'Your week, written up',
                "goal": ("Write the week's work into a readable log entry - what moved, what shipped, what stalled - grouped by project, never a to-do triage."),
            },
            "trigger": {"sources": [
                {"id": "sched", "mode": "schedule",
                 "schedule": {"cron_local": "0 16 * * 5"}},
            ]},
            "steps": [
                {"id": "issues", "connector_id": "jira",
                 "tool": "jira__search_issues",
                 "params": {"jql": "{{var.jira_jql}}", "max_results": 25},
                 "collect": {"items_path": "issues",
                             "fields": {"key": "key",
                                        "summary": "summary",
                                        "status": "status"},
                             "format": "- {{item.key}} [{{item.status}}] "
                                       "{{item.summary}}",
                             "limit": 25,
                             "empty_text": "(no issues touched)"},
                 "on_error": "fail"},
                {"id": "log", "connector_id": "docs",
                 "tool": "docs__append_text",
                 "params": {"document_id": "{{grant.target.id}}",
                            "text": "\nWeek log — {{steps.issues.count}} "
                                    "issues:\n{{steps.issues.text}}\n"}},
            ],
        },
        "sort_order": 5,
    },
    {
        "slug": "daily-standup-notes",
        "name": "Daily standup notes",
        "description": "Each weekday morning, post your in-progress "
                       "Jira issues to a Teams chat.",
        "icon": "teams",
        "category": "work",
        "connectors": ["jira", "teams"],
        "variables": [
            _v("jira_jql", "Jira filter", "JQL for standup.",
               default="assignee = currentUser() AND status = "
                       "\"In Progress\" ORDER BY updated DESC"),
        ],
        "spec": {
            "version": 2,
            "name": "Daily standup notes",
            "mode": "auto",
            "narration": {
                "style": "digest",
                "title": 'Your standup notes',
                "goal": ("Turn yesterday's activity into standup notes - done, in flight, blocked - in the user's own voice, ready to post."),
            },
            "trigger": {"sources": [
                {"id": "sched", "mode": "schedule",
                 "schedule": {"cron_local": "0 9 * * 1-5"}},
            ]},
            "steps": [
                {"id": "wip", "connector_id": "jira",
                 "tool": "jira__search_issues",
                 "params": {"jql": "{{var.jira_jql}}", "max_results": 10},
                 "collect": {"items_path": "issues",
                             "fields": {"key": "key", "summary": "summary"},
                             "format": "• {{item.key}} {{item.summary}}",
                             "limit": 10,
                             "empty_text": "Nothing in progress."},
                 "on_error": "fail"},
                {"id": "post", "connector_id": "teams",
                 "tool": "teams__send_chat_message",
                 "params": {"chat_id": "{{grant.target.id}}",
                            "message": "Standup — in progress "
                                       "({{steps.wip.count}}):\n"
                                       "{{steps.wip.text}}"}},
            ],
        },
        "sort_order": 6,
    },

    # ═══ email ═══
    {
        # R28-C's proactive-assist flagship — lives here so the catalog
        # stays the single source (their round extends it in place).
        "slug": "boss-email-draft",
        "name": "Boss email → draft reply",
        "description": "When a specific sender emails you, stage a "
                       "draft reply for your review — nothing is ever "
                       "sent for you.",
        "icon": "gmail",
        "category": "email",
        "connectors": ["gmail"],
        "variables": [
            _v("boss_email", "Sender to watch",
               "Only mail from this address fires the automation.",
               example="boss@company.com"),
            _v("draft_style", "Draft style",
               "One line of guidance baked into the draft skeleton.",
               required=False,
               default="Short, direct, friendly."),
        ],
        "spec": {
            "version": 2,
            "name": "Boss email → draft reply",
            "mode": "confirm",
            "trigger": {"sources": [
                {"id": "mail", "mode": "push", "connector_id": "gmail",
                 "event": "email_received",
                 "filter": {"from": ["{{var.boss_email}}"]},
                 "dedupe_key": "event.message_id"},
            ]},
            "steps": [
                {"id": "draft", "connector_id": "gmail",
                 "tool": "gmail__create_draft",
                 "params": {"to": "{{grant.target.id}}",
                            "subject": "Re: {{event.subject}}",
                            "body": "({{var.draft_style}})\n\n"
                                    "Re: {{event.snippet}}\n\n"}},
            ],
        },
        "sort_order": 0,
    },
    {
        "slug": "email-to-slack",
        "name": "Important email → Slack",
        "description": "Ping a Slack channel when matching mail "
                       "arrives in Gmail.",
        "icon": "gmail",
        "category": "email",
        "connectors": ["gmail", "slack"],
        "variables": [],
        "spec": {
            "name": "Important email → Slack",
            "trigger": {"mode": "push", "connector_id": "gmail",
                        "event": "email_received", "filter": {}},
            "action": {"connector_id": "slack",
                       "tool": "slack__send_message",
                       "params_template": {
                           "channel": "{{grant.target.id}}",
                           "text": "Mail from {{event.from}}: "
                                   "{{event.subject}}",
                       }},
            "dedupe_key": "event.message_id",
            "mode": "auto",
        },
        "sort_order": 1,
    },
    {
        "slug": "outlook-to-teams",
        "name": "Outlook mail → Teams",
        "description": "Post new Outlook mail into a Teams chat you "
                       "pick.",
        "icon": "outlook",
        "category": "email",
        "connectors": ["outlook", "teams"],
        "variables": [],
        "spec": {
            "name": "Outlook mail → Teams",
            "trigger": {"mode": "poll", "connector_id": "outlook",
                        "event": "email_received",
                        "poll_interval_s": 300, "filter": {}},
            "action": {"connector_id": "teams",
                       "tool": "teams__send_chat_message",
                       "params_template": {
                           "chat_id": "{{grant.target.id}}",
                           "message": "Outlook — {{event.from}}: "
                                      "{{event.subject}} "
                                      "({{event.preview}})",
                       }},
            "dedupe_key": "event.id",
            "mode": "auto",
        },
        "sort_order": 2,
    },
    {
        "slug": "inbox-zero-morning",
        "name": "Inbox summary",
        "description": "A weekday-morning Slack summary of unread "
                       "Gmail and Outlook.",
        "icon": "slack",
        "category": "email",
        "connectors": ["gmail", "outlook", "slack"],
        "variables": [],
        "spec": {
            "version": 2,
            "name": "Inbox summary",
            "mode": "auto",
            "trigger": {"sources": [
                {"id": "sched", "mode": "schedule",
                 "schedule": {"cron_local": "0 8 * * 1-5"}},
            ]},
            "steps": [
                {"id": "mail", "connector_id": "gmail",
                 "tool": "gmail__list_messages",
                 "params": {"query": "is:unread newer_than:1d",
                            "max_results": 10},
                 "collect": {"items_path": "messages",
                             "fields": {"subject": "headers.Subject",
                                        "from": "headers.From"},
                             "format": "• {{item.from}} — "
                                       "{{item.subject}}",
                             "limit": 8,
                             "empty_text": "Gmail is clear."},
                 "on_error": "continue"},
                {"id": "outlook", "connector_id": "outlook",
                 "tool": "outlook__list_messages",
                 "params": {"is_read": False, "max_results": 10},
                 "collect": {"items_path": "messages",
                             "fields": {"subject": "subject",
                                        "from": "from"},
                             "format": "• {{item.from}} — "
                                       "{{item.subject}}",
                             "limit": 8,
                             "empty_text": "Outlook is clear."},
                 "on_error": "continue"},
                {"id": "post", "connector_id": "slack",
                 "tool": "slack__send_message",
                 "params": {"channel": "{{grant.target.id}}",
                            "text": "*Unread this morning*\n\n"
                                    "*Gmail ({{steps.mail.count}})*\n"
                                    "{{steps.mail.text}}\n\n"
                                    "*Outlook ({{steps.outlook.count}})*\n"
                                    "{{steps.outlook.text}}"}},
            ],
        },
        "sort_order": 3,
    },
    {
        "slug": "invoice-email-log",
        "name": "Invoice emails → Doc",
        "description": "Append a line to a Google Doc for each "
                       "matching email (narrow the filter to invoices "
                       "during setup).",
        "icon": "docs",
        "category": "email",
        "connectors": ["gmail", "docs"],
        "variables": [],
        "spec": {
            "name": "Invoice emails → Doc",
            "trigger": {"mode": "push", "connector_id": "gmail",
                        "event": "email_received",
                        "filter": {"subject": ["invoice", "receipt"]}},
            "action": {"connector_id": "docs",
                       "tool": "docs__append_text",
                       "params_template": {
                           "document_id": "{{grant.target.id}}",
                           "text": "\n{{event.from}} — {{event.subject}}",
                       }},
            "dedupe_key": "event.message_id",
            "mode": "auto",
        },
        "sort_order": 4,
    },

    # ═══ code ═══
    {
        "slug": "github-issue-to-slack",
        "name": "GitHub issue → Slack",
        "description": "Post new issues from a repository to a Slack "
                       "channel.",
        "icon": "github",
        "category": "code",
        "connectors": ["github", "slack"],
        "variables": [
            _v("github_owner", "GitHub owner", "Repo owner/org."),
            _v("github_repo", "GitHub repo", "Repository name."),
        ],
        "spec": {
            "version": 2,
            "name": "GitHub issue → Slack",
            "mode": "auto",
            "trigger": {"sources": [
                {"id": "gh", "mode": "poll", "connector_id": "github",
                 "event": "issue_opened",
                 "params": {"owner": "{{var.github_owner}}",
                            "repo": "{{var.github_repo}}"},
                 "poll_interval_s": 600,
                 "dedupe_key": "event.number"},
            ]},
            "steps": [
                {"id": "post", "connector_id": "slack",
                 "tool": "slack__send_message",
                 "params": {"channel": "{{grant.target.id}}",
                            "text": "New issue #{{event.number}}: "
                                    "{{event.title}} ({{event.url}})"}},
            ],
        },
        "sort_order": 0,
    },
    {
        "slug": "daily-repo-digest",
        "name": "Daily repo digest",
        "description": "An end-of-day Slack digest of a repository's "
                       "open issues.",
        "icon": "github",
        "category": "code",
        "connectors": ["github", "slack"],
        "variables": [
            _v("github_owner", "GitHub owner", "Repo owner/org."),
            _v("github_repo", "GitHub repo", "Repository name."),
        ],
        "spec": {
            "version": 2,
            "name": "Daily repo digest",
            "mode": "auto",
            "narration": {
                "style": "digest",
                "title": 'The repo today',
                "goal": ("Digest the repository's open issues - group them by area or theme and say what actually changed, never a personal task triage."),
            },
            "trigger": {"sources": [
                {"id": "sched", "mode": "schedule",
                 "schedule": {"cron_local": "30 17 * * 1-5"}},
            ]},
            "steps": [
                {"id": "open", "connector_id": "github",
                 "tool": "github__list_issues",
                 "params": {"owner": "{{var.github_owner}}",
                            "repo": "{{var.github_repo}}",
                            "state": "open", "per_page": 15},
                 "collect": {"items_path": "issues",
                             "fields": {"number": "number",
                                        "title": "title"},
                             "format": "• #{{item.number}} {{item.title}}",
                             "limit": 15,
                             "empty_text": "No open issues."},
                 "on_error": "fail"},
                {"id": "post", "connector_id": "slack",
                 "tool": "slack__send_message",
                 "params": {"channel": "{{grant.target.id}}",
                            "text": "*{{var.github_repo}} — open issues "
                                    "({{steps.open.count}})*\n"
                                    "{{steps.open.text}}"}},
            ],
        },
        "sort_order": 1,
    },
    {
        "slug": "github-issue-to-teams",
        "name": "GitHub issue → Teams",
        "description": "Post new repository issues into a Teams chat.",
        "icon": "github",
        "category": "code",
        "connectors": ["github", "teams"],
        "variables": [
            _v("github_owner", "GitHub owner", "Repo owner/org."),
            _v("github_repo", "GitHub repo", "Repository name."),
        ],
        "spec": {
            "version": 2,
            "name": "GitHub issue → Teams",
            "mode": "auto",
            "trigger": {"sources": [
                {"id": "gh", "mode": "poll", "connector_id": "github",
                 "event": "issue_opened",
                 "params": {"owner": "{{var.github_owner}}",
                            "repo": "{{var.github_repo}}"},
                 "poll_interval_s": 600,
                 "dedupe_key": "event.number"},
            ]},
            "steps": [
                {"id": "post", "connector_id": "teams",
                 "tool": "teams__send_chat_message",
                 "params": {"chat_id": "{{grant.target.id}}",
                            "message": "New issue #{{event.number}}: "
                                       "{{event.title}} — {{event.url}}"}},
            ],
        },
        "sort_order": 2,
    },
    {
        "slug": "github-to-notion-log",
        "name": "GitHub issues → Notion",
        "description": "Create a Notion page under a parent you pick "
                       "for every new repository issue.",
        "icon": "notion",
        "category": "code",
        "connectors": ["github", "notion"],
        "variables": [
            _v("github_owner", "GitHub owner", "Repo owner/org."),
            _v("github_repo", "GitHub repo", "Repository name."),
        ],
        "spec": {
            "version": 2,
            "name": "GitHub issues → Notion",
            "mode": "confirm",
            "trigger": {"sources": [
                {"id": "gh", "mode": "poll", "connector_id": "github",
                 "event": "issue_opened",
                 "params": {"owner": "{{var.github_owner}}",
                            "repo": "{{var.github_repo}}"},
                 "poll_interval_s": 600,
                 "dedupe_key": "event.number"},
            ]},
            "steps": [
                {"id": "page", "connector_id": "notion",
                 "tool": "notion__create_page",
                 "params": {"parent_page_id": "{{grant.target.id}}",
                            "title": "GH #{{event.number}}: "
                                     "{{event.title}}",
                            "content": "{{event.url}}"}},
            ],
        },
        "sort_order": 3,
    },

    # ═══ calendar ═══
    {
        "slug": "calendar-to-slack",
        "name": "New event → Slack",
        "description": "A Slack note whenever a new event lands on "
                       "your calendar.",
        "icon": "calendar",
        "category": "calendar",
        "connectors": ["calendar", "slack"],
        "variables": [],
        "spec": {
            "name": "New event → Slack",
            "trigger": {"mode": "poll", "connector_id": "calendar",
                        "event": "event_created",
                        "poll_interval_s": 600, "filter": {}},
            "action": {"connector_id": "slack",
                       "tool": "slack__send_message",
                       "params_template": {
                           "channel": "{{grant.target.id}}",
                           "text": "New calendar event: {{event.title}}",
                       }},
            "dedupe_key": "event.id",
            "mode": "auto",
        },
        "sort_order": 0,
    },
    {
        "slug": "daily-agenda",
        "name": "Daily agenda",
        "description": "Your day's calendar posted to Slack each "
                       "weekday morning.",
        "icon": "calendar",
        "category": "calendar",
        "connectors": ["calendar", "slack"],
        "variables": [],
        "spec": {
            "version": 2,
            "name": "Daily agenda",
            "mode": "auto",
            "narration": {
                "style": "digest",
                "title": "Your day's agenda",
                "goal": ("Lay the day's calendar out in order, with what each meeting needs - a schedule, not a task triage."),
            },
            "trigger": {"sources": [
                {"id": "sched", "mode": "schedule",
                 "schedule": {"cron_local": "30 7 * * 1-5"}},
            ]},
            "steps": [
                {"id": "events", "connector_id": "calendar",
                 "tool": "calendar__list_events",
                 "params": {"window_days": 1, "max_results": 15},
                 "collect": {"items_path": "events",
                             "fields": {"title": "summary",
                                        "when": "start.dateTime"},
                             "format": "• {{item.when}} — {{item.title}}",
                             "limit": 15,
                             "empty_text": "Nothing scheduled."},
                 "on_error": "fail"},
                {"id": "post", "connector_id": "slack",
                 "tool": "slack__send_message",
                 "params": {"channel": "{{grant.target.id}}",
                            "text": "*Today ({{steps.events.count}} "
                                    "events)*\n{{steps.events.text}}"}},
            ],
        },
        "sort_order": 1,
    },
    {
        "slug": "meeting-prep-pages",
        "name": "Meeting prep pages",
        "description": "A Notion prep page under a parent you pick for "
                       "each new calendar event.",
        "icon": "notion",
        "category": "calendar",
        "connectors": ["calendar", "notion"],
        "variables": [],
        "spec": {
            "name": "Meeting prep pages",
            "trigger": {"mode": "poll", "connector_id": "calendar",
                        "event": "event_created",
                        "poll_interval_s": 600, "filter": {}},
            "action": {"connector_id": "notion",
                       "tool": "notion__create_page",
                       "params_template": {
                           "parent_page_id": "{{grant.target.id}}",
                           "title": "Prep: {{event.title}}",
                           "content": "Agenda:\n\nNotes:\n",
                       }},
            "dedupe_key": "event.id",
            "mode": "confirm",
        },
        "sort_order": 2,
    },
    {
        "slug": "week-ahead-digest",
        "name": "Week-ahead digest",
        "description": "Monday morning: your coming week's events, "
                       "staged as a Gmail draft to an address you pin.",
        "icon": "gmail",
        "category": "calendar",
        "connectors": ["calendar", "gmail"],
        "variables": [],
        "spec": {
            "version": 2,
            "name": "Week-ahead digest",
            "mode": "confirm",
            "narration": {
                "style": "digest",
                "title": 'Your week ahead',
                "goal": ("Fold next week's calendar into one readable preview - group by day, flag the heavy days - a preview, not a task triage."),
            },
            "trigger": {"sources": [
                {"id": "sched", "mode": "schedule",
                 "schedule": {"cron_local": "0 8 * * 1"}},
            ]},
            "steps": [
                {"id": "events", "connector_id": "calendar",
                 "tool": "calendar__list_events",
                 "params": {"window_days": 7, "max_results": 25},
                 "collect": {"items_path": "events",
                             "fields": {"title": "summary",
                                        "when": "start.dateTime"},
                             "format": "- {{item.when}} — {{item.title}}",
                             "limit": 25,
                             "empty_text": "(empty week)"},
                 "on_error": "fail"},
                {"id": "draft", "connector_id": "gmail",
                 "tool": "gmail__create_draft",
                 "params": {"to": "{{grant.target.id}}",
                            "subject": "Your week ahead",
                            "body": "{{steps.events.count}} events:\n"
                                    "{{steps.events.text}}"}},
            ],
        },
        "sort_order": 3,
    },

    # ═══ school ═══
    {
        "slug": "assignment-email-log",
        "name": "Assignment emails → Notion",
        "description": "Log matching school emails as Notion pages "
                       "under a parent you pick.",
        "icon": "notion",
        "category": "school",
        "connectors": ["gmail", "notion"],
        "variables": [],
        "spec": {
            "name": "Assignment emails → Notion",
            "trigger": {"mode": "push", "connector_id": "gmail",
                        "event": "email_received",
                        "filter": {"subject": ["assignment", "due",
                                               "homework"]}},
            "action": {"connector_id": "notion",
                       "tool": "notion__create_page",
                       "params_template": {
                           "parent_page_id": "{{grant.target.id}}",
                           "title": "{{event.subject}}",
                           "content": "From {{event.from}}\n\n"
                                      "{{event.snippet}}",
                       }},
            "dedupe_key": "event.message_id",
            "mode": "auto",
        },
        "sort_order": 0,
    },
    {
        "slug": "study-block-reminder",
        "name": "Study block reminder",
        "description": "A Slack nudge when your evening study block "
                       "starts.",
        "icon": "slack",
        "category": "school",
        "connectors": ["slack"],
        "variables": [],
        "spec": {
            "name": "Study block reminder",
            "trigger": {"mode": "schedule",
                        "schedule": {"cron_local": "0 18 * * 1-5"}},
            "action": {"connector_id": "slack",
                       "tool": "slack__send_message",
                       "params_template": {
                           "channel": "{{grant.target.id}}",
                           "text": "Study block starts now — phone "
                                   "away, one topic, 50 minutes.",
                       }},
            "mode": "auto",
        },
        "sort_order": 1,
    },
    {
        "slug": "class-email-digest",
        "name": "Class email digest",
        "description": "Each afternoon, append the day's class emails "
                       "to a Google Doc.",
        "icon": "docs",
        "category": "school",
        "connectors": ["gmail", "docs"],
        "variables": [
            _v("gmail_query", "Gmail search",
               "Which mail counts as class mail.",
               example="from:(@university.edu) newer_than:1d",
               default="newer_than:1d"),
        ],
        "spec": {
            "version": 2,
            "name": "Class email digest",
            "mode": "auto",
            "narration": {
                "style": "digest",
                "title": "Today's class mail",
                "goal": ("Collect the day's class emails into one digest - group by course or sender - a record, not a task triage."),
            },
            "trigger": {"sources": [
                {"id": "sched", "mode": "schedule",
                 "schedule": {"cron_local": "0 17 * * 1-5"}},
            ]},
            "steps": [
                {"id": "mail", "connector_id": "gmail",
                 "tool": "gmail__list_messages",
                 "params": {"query": "{{var.gmail_query}}",
                            "max_results": 15},
                 "collect": {"items_path": "messages",
                             "fields": {"subject": "headers.Subject",
                                        "from": "headers.From"},
                             "format": "- {{item.from}} — "
                                       "{{item.subject}}",
                             "limit": 15,
                             "empty_text": "(no class mail today)"},
                 "on_error": "fail"},
                {"id": "log", "connector_id": "docs",
                 "tool": "docs__append_text",
                 "params": {"document_id": "{{grant.target.id}}",
                            "text": "\nClass mail "
                                    "({{steps.mail.count}}):\n"
                                    "{{steps.mail.text}}\n"}},
            ],
        },
        "sort_order": 2,
    },
    {
        "slug": "deadline-watch",
        "name": "Deadline watch",
        "description": "A Slack alert when an exam or due date lands "
                       "on your calendar.",
        "icon": "calendar",
        "category": "school",
        "connectors": ["calendar", "slack"],
        "variables": [],
        "spec": {
            "name": "Deadline watch",
            "trigger": {"mode": "poll", "connector_id": "calendar",
                        "event": "event_created",
                        "poll_interval_s": 600,
                        "filter": {"title": ["exam", "due", "deadline",
                                             "quiz"]}},
            "action": {"connector_id": "slack",
                       "tool": "slack__send_message",
                       "params_template": {
                           "channel": "{{grant.target.id}}",
                           "text": "Deadline on your calendar: "
                                   "{{event.title}}",
                       }},
            "dedupe_key": "event.id",
            "mode": "auto",
        },
        "sort_order": 3,
    },

    # ═══ personal ═══
    {
        "slug": "daily-plan-note",
        "name": "Daily plan note",
        "description": "A fresh planning header appended to a Google "
                       "Doc every morning.",
        "icon": "docs",
        "category": "personal",
        "connectors": ["docs"],
        "variables": [],
        "spec": {
            "name": "Daily plan note",
            "trigger": {"mode": "schedule",
                        "schedule": {"cron_local": "0 7 * * *"}},
            "action": {"connector_id": "docs",
                       "tool": "docs__append_text",
                       "params_template": {
                           "document_id": "{{grant.target.id}}",
                           "text": "\n\n── Today ──\nTop 3:\n1.\n2.\n3.\n",
                       }},
            "mode": "auto",
        },
        "sort_order": 0,
    },
    {
        "slug": "newsletter-roundup",
        "name": "Newsletter roundup",
        "description": "Sunday morning: the week's newsletters folded "
                       "into one Gmail draft to an address you pin.",
        "icon": "gmail",
        "category": "personal",
        "connectors": ["gmail"],
        "variables": [
            _v("gmail_query", "Gmail search",
               "Which mail counts as newsletters.",
               default="category:promotions OR category:updates "
                       "newer_than:7d"),
        ],
        "spec": {
            "version": 2,
            "name": "Newsletter roundup",
            "mode": "confirm",
            # R36-7: this automation's result is a DIGEST of its own
            # material, never the morning triage.
            "narration": {
                "style": "digest",
                "title": "This week's newsletters",
                "goal": "Fold the week's newsletters into one readable "
                        "roundup — group them by theme or sender, pull "
                        "out anything genuinely worth a click, and stage "
                        "the whole thing as one Gmail draft.",
            },
            "trigger": {"sources": [
                {"id": "sched", "mode": "schedule",
                 "schedule": {"cron_local": "0 9 * * 0"}},
            ]},
            "steps": [
                {"id": "mail", "connector_id": "gmail",
                 "tool": "gmail__list_messages",
                 "params": {"query": "{{var.gmail_query}}",
                            "max_results": 25},
                 "collect": {"items_path": "messages",
                             "fields": {"subject": "headers.Subject",
                                        "from": "headers.From"},
                             "format": "- {{item.from}} — "
                                       "{{item.subject}}",
                             "limit": 25,
                             "empty_text": "(a quiet week)"},
                 "on_error": "fail"},
                {"id": "draft", "connector_id": "gmail",
                 "tool": "gmail__create_draft",
                 "params": {"to": "{{grant.target.id}}",
                            "subject": "This week's newsletters",
                            "body": "{{steps.mail.count}} newsletters:\n"
                                    "{{steps.mail.text}}"}},
            ],
        },
        "sort_order": 1,
    },
    {
        "slug": "file-drop-alert",
        "name": "Drive file → Slack",
        "description": "A Slack note when a file the agent can see "
                       "appears in Drive.",
        "icon": "drive",
        "category": "personal",
        "connectors": ["drive", "slack"],
        "variables": [],
        "spec": {
            "name": "Drive file → Slack",
            "trigger": {"mode": "poll", "connector_id": "drive",
                        "event": "file_added",
                        "poll_interval_s": 600, "filter": {}},
            "action": {"connector_id": "slack",
                       "tool": "slack__send_message",
                       "params_template": {
                           "channel": "{{grant.target.id}}",
                           "text": "New file in Drive: {{event.name}} "
                                   "({{event.url}})",
                       }},
            "dedupe_key": "event.id",
            "mode": "auto",
        },
        "sort_order": 2,
    },
    {
        "slug": "weekly-review-page",
        "name": "Weekly review page",
        "description": "A fresh Notion review page under a parent you "
                       "pick, every Friday evening.",
        "icon": "notion",
        "category": "personal",
        "connectors": ["notion"],
        "variables": [],
        "spec": {
            "name": "Weekly review page",
            "trigger": {"mode": "schedule",
                        "schedule": {"cron_local": "0 17 * * 5"}},
            "action": {"connector_id": "notion",
                       "tool": "notion__create_page",
                       "params_template": {
                           "parent_page_id": "{{grant.target.id}}",
                           "title": "Weekly review",
                           "content": "Went well:\n\nDidn't:\n\n"
                                      "Next week:\n",
                       }},
            "mode": "confirm",
        },
        "sort_order": 3,
    },
]


def template_payload(t) -> dict:
    """One AutomationTemplate row → the wire shape both the user route
    and the agent RPC serve (a diverging serializer is how a consumer
    silently loses fields — one function, two mounts)."""
    def _loads(raw, default):
        try:
            out = json.loads(raw) if raw else default
            return out if isinstance(out, type(default)) else default
        except (ValueError, TypeError):
            return default

    spec = _loads(t.spec_json, {})
    from app.services import automation_verbs

    cadence = automation_verbs.schedule_human(spec)
    if cadence is None:
        # Event-triggered template: the tag comes from the event
        # vocabulary ("on new GitHub issues") — never a raw key.
        trig = (spec.get("trigger") or {})
        sources = trig.get("sources") or ([trig] if trig else [])
        first = next(
            (s for s in sources if isinstance(s, dict) and s.get("event")),
            {},
        )
        cadence = automation_verbs.event_tag(first.get("event"))
    return {
        "id": t.id,
        "slug": t.slug,
        "name": t.name,
        "description": t.description,
        "icon": t.icon,
        "category": getattr(t, "category", None) or "work",
        "connectors": _loads(t.connectors_json, []),
        "variables": _loads(getattr(t, "variables_json", None), []),
        "spec": spec,
        "cadence_human": cadence,
    }


async def sync_template_catalog(db) -> dict:
    """Upsert the catalog into automation_templates by slug. Returns
    {inserted, updated, unchanged}. Never touches `enabled` on an
    existing row; never deletes rows outside the catalog. Callers
    treat failures as non-fatal (boot must not block on this)."""
    from sqlalchemy import select
    from app.db.models.platform_automation import AutomationTemplate

    inserted = updated = unchanged = 0
    existing = {
        t.slug: t
        for t in (await db.execute(select(AutomationTemplate))).scalars().all()
    }
    for entry in CATALOG:
        desired = {
            "name": entry["name"],
            "description": entry["description"],
            "icon": entry.get("icon"),
            "category": entry["category"],
            "connectors_json": json.dumps(entry["connectors"]),
            "variables_json": json.dumps(entry.get("variables") or []),
            "spec_json": json.dumps(entry["spec"], sort_keys=True),
            "sort_order": entry.get("sort_order", 0),
        }
        row = existing.get(entry["slug"])
        if row is None:
            db.add(AutomationTemplate(slug=entry["slug"], **desired))
            inserted += 1
            continue
        drift = {k: v for k, v in desired.items() if getattr(row, k) != v}
        if drift:
            for k, v in drift.items():
                setattr(row, k, v)
            updated += 1
        else:
            unchanged += 1
    await db.commit()
    if inserted or updated:
        logger.info("[automations] template catalog synced: +%d ~%d =%d",
                    inserted, updated, unchanged)
    return {"inserted": inserted, "updated": updated, "unchanged": unchanged}
