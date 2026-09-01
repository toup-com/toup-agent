"""099 — re-spec the installed "Morning work brief" (R42).

A catalog fix does not fix an automation somebody already has. The
flagship went out as five connector reads stitched by a string
template, and every account that adopted it is still posting that shape
every weekday: a failed read left `count` at 0 and `text` at "Could not
read GitHub.", so the post printed `GitHub (0)` directly above the
sentence saying it never read GitHub, and a step that never ran had no
`steps.<id>` node at all — which `render_value` resolves to "", giving
`Teams ()`. This rewrites those installations to the seven-step spec
(five reads, one agent step that ranks, one Slack post interpolating
that ONE value).

TENANT DBs — the inverse of the 095-098 guard. `automations` is
AGENT_ONLY, so the platform DB does not have it and this is a no-op
there; `alembic upgrade head` runs on boot in both images, which is
what carries this to every account. The presence of the table is the
whole guard: a brand-new tenant has not run `init_db` yet, has no
`automations` table and no installations to fix, and the next boot's
`create_all` gives it the table with nothing in it.

WHAT IT WILL NOT TOUCH. The spec is replaced only where the row is
still the shipped shape, and everything the USER put there is carried
across, not regenerated:

  - the post step's `grant_id` and `grant_target` ride over verbatim —
    the destination they pinned, and the reason an armed automation
    stays armed and correctly bound;
  - `focus` (their pins, and their per-pin notes) rides over verbatim;
  - the schedule source rides over VERBATIM, so an edited cron is kept
    and the compiled schedule binding still matches the spec — which is
    why no recompile is needed and `status` is not touched;
  - `name` and `mode` ride over from the row's own spec;
  - the description is replaced only when it is still the shipped
    sentence (which names GitHub, Teams and Outlook and would now be
    false); a description the user changed is theirs.

And it SKIPS, loudly, rather than guessing:

  - `steps_human_json` non-empty — `workflow.set_steps` is its only
    writer, so those are the user's own words for the five old steps and
    replacing the steps under them would leave the sheet describing an
    automation that no longer exists;
  - a spec whose step ids/tools are not the shipped six, or whose post
    text is not the shipped template — the plan was edited;
  - a `jira_jql` answer that is not the shipped default — the new board
    step carries one fixed rubric, and silently swapping somebody's own
    filter for it changes which issues they are shown.

A skipped row keeps working exactly as it does today and is named in
the log. There is no automatic second attempt: re-adopting the template
from the catalog is the honest route for an automation somebody has
made their own.

`workflow_rev` is bumped because the app holds local drafts against it;
without the bump a device holding a draft of the OLD workflow would
commit it back over this.

Revision ID: 099
Revises: 098
"""

from __future__ import annotations

import json
import logging

import sqlalchemy as sa
from alembic import op

revision = "099"
down_revision = "098"
branch_labels = None
depends_on = None

logger = logging.getLogger("alembic.runtime.migration")

_AUTOMATIONS = "automations"
_SLUG = "morning-work-brief"

#: The R28 shape, as shipped. Identity, not a heuristic: a row that does
#: not match it is one somebody changed, and it is left alone.
_OLD_STEPS = (
    ("issues", "jira__search_issues"),
    ("repo", "github__list_issues"),
    ("chat", "teams__read_chat_messages"),
    ("mail", "gmail__list_messages"),
    ("outlook", "outlook__list_messages"),
    ("post", "slack__send_message"),
)
_OLD_POST_TEXT = (
    "*Morning work brief*\n\n"
    "*Jira ({{steps.issues.count}})*\n{{steps.issues.text}}\n\n"
    "*GitHub ({{steps.repo.count}})*\n{{steps.repo.text}}\n\n"
    "*Teams ({{steps.chat.count}})*\n{{steps.chat.text}}\n\n"
    "*Gmail unread ({{steps.mail.count}})*\n{{steps.mail.text}}\n\n"
    "*Outlook unread ({{steps.outlook.count}})*\n{{steps.outlook.text}}"
)
_OLD_DESCRIPTION = (
    "Every weekday morning, pull your open Jira issues, the repo's open "
    "GitHub issues, your main Teams chat, and unread Gmail + Outlook \u2014 "
    "and post one sectioned brief to a Slack channel."
)
_OLD_JQL_DEFAULT = (
    "assignee = currentUser() AND statusCategory != Done ORDER BY updated DESC"
)

#: A POINT-IN-TIME COPY of the R42 catalog spec, deliberately frozen:
#: an alembic file must mean the same thing in a year, and the catalog
#: is edited every round. `test_automation_template_catalog.py` is what
#: keeps the catalog itself honest; this is what the founder's already-
#: installed automation becomes.
_NEW_SPEC = json.loads(r"""
{
    "description": "Every weekday morning, read what is next on your calendar, what landed overnight, what has been waiting on you, who named you in Slack and what is live on your Jira board — then post one ranked brief to a Slack channel.",
    "mode": "auto",
    "name": "Morning work brief",
    "steps": [
        {
            "collect": {
                "empty_text": "Nothing on the calendar.",
                "fields": {
                    "at": "start.dateTime",
                    "day": "start.date",
                    "title": "summary"
                },
                "format": "- {{item.at}}{{item.day}} {{item.title}}",
                "items_path": "events",
                "limit": 10
            },
            "connector_id": "calendar",
            "id": "cal",
            "on_error": "continue",
            "params": {
                "max_results": 10,
                "window_days": 1
            },
            "tool": "calendar__list_events"
        },
        {
            "collect": {
                "empty_text": "No new mail.",
                "fields": {
                    "cc": "headers.Cc",
                    "date": "headers.Date",
                    "from": "headers.From",
                    "snippet": "snippet",
                    "subject": "headers.Subject",
                    "to": "headers.To"
                },
                "format": "- {{item.date}} · from {{item.from}} · to {{item.to}} · cc {{item.cc}} · {{item.subject}} — {{item.snippet}}",
                "items_path": "messages",
                "limit": 10
            },
            "connector_id": "gmail",
            "id": "mail",
            "on_error": "continue",
            "params": {
                "max_results": 10,
                "query": "in:inbox newer_than:1d"
            },
            "tool": "gmail__list_messages"
        },
        {
            "collect": {
                "empty_text": "Nothing waiting on you.",
                "fields": {
                    "from": "headers.From",
                    "subject": "headers.Subject"
                },
                "format": "- from {{item.from}} — {{item.subject}}",
                "items_path": "messages",
                "limit": 10
            },
            "connector_id": "gmail",
            "id": "waiting",
            "on_error": "continue",
            "params": {
                "max_results": 10,
                "query": "to:me is:unread older_than:1d newer_than:7d"
            },
            "tool": "gmail__list_messages"
        },
        {
            "collect": {
                "empty_text": "Nobody named you.",
                "fields": {
                    "from": "from",
                    "text": "text",
                    "where": "channel_name"
                },
                "format": "- {{item.from}} in {{item.where}}: {{item.text}}",
                "items_path": "matches",
                "limit": 15
            },
            "connector_id": "slack",
            "id": "rooms",
            "on_error": "continue",
            "params": {
                "count": 15,
                "query": "to:me",
                "sort": "timestamp"
            },
            "tool": "slack__search_messages"
        },
        {
            "collect": {
                "empty_text": "Nothing live on the board.",
                "fields": {
                    "due": "duedate",
                    "key": "key",
                    "priority": "priority",
                    "status": "status",
                    "summary": "summary"
                },
                "format": "- {{item.key}} [{{item.status}}] [{{item.priority}}] {{item.due}} {{item.summary}}",
                "items_path": "issues",
                "limit": 15
            },
            "connector_id": "jira",
            "id": "board",
            "on_error": "continue",
            "params": {
                "jql": "assignee = currentUser() AND statusCategory != Done AND (duedate <= 7d OR priority in (Highest, High) OR updated >= -1d) ORDER BY duedate ASC, priority DESC, updated DESC",
                "max_results": 15
            },
            "tool": "jira__search_issues"
        },
        {
            "id": "rank",
            "kind": "agent",
            "output_var": "brief",
            "prompt": "Rank this morning's work for the reader. Order by what breaks if it is ignored, never by which app it came from.\nThe steps are the facts: cal is calendar entries; mail arrived in the last day; waiting is mail to you, unread, 1-7 days old; rooms is Slack naming you, newest first, no times; board is your live Jira. You have no clock: never write \"today\" or \"yesterday\", never work out a date, use only a date a line gives you.\nSections, headings exact, in this order, each omitted when it has nothing:\n*DO FIRST - BLOCKS OTHERS* - at most 2: someone is stopped until this moves.\n*ANSWER TODAY* - at most 3: a question to you, or work due on or before the earliest date cal gives.\n*THIS WEEK* - at most 3: dated later, or owed and undated.\n*NO ACTION - FOR AWARENESS* - at most 2: worth knowing, needs nothing.\n*IGNORED - NOTHING NEEDED YOU* - one line: named categories with counts (cc only, automated, newsletters, chatter).\nOne item may appear in one section only. If no section has anything, write one line: \"Nothing needed you this morning.\"\nSTANDING ORDERS: the material's starts_at names the people, channels and projects the reader pinned, with their own instructions. These RANK, never filter: a pinned name outranks others inside its section.\nFor a step whose ok is false, add a final line: \"Missing: name it, I could not read it.\" Never report a step you could not read as a zero.\nNever write a number you did not copy; the ignored counts you get by counting lines. Never write a double brace. Under 900 characters."
        },
        {
            "connector_id": "slack",
            "id": "post",
            "params": {
                "channel": "{{grant.target.id}}",
                "text": "*Morning brief*\n\n{{var.brief}}"
            },
            "tool": "slack__send_message"
        }
    ],
    "trigger": {
        "sources": [
            {
                "id": "sched",
                "mode": "schedule",
                "schedule": {
                    "cron_local": "0 8 * * 1-5"
                }
            }
        ]
    },
    "version": 2
}""")


def _target(spec: dict) -> dict | None:
    """The new spec for one installed automation, or None to skip it.

    Pure, so the whole decision is readable in one place and testable
    without a database.
    """
    if spec.get("version") != 2:
        return None
    steps = [s for s in spec.get("steps") or [] if isinstance(s, dict)]
    if [(s.get("id"), s.get("tool")) for s in steps] != list(_OLD_STEPS):
        return None
    old_post = steps[-1]
    if (old_post.get("params") or {}).get("text") != _OLD_POST_TEXT:
        return None
    jql = (spec.get("variables") or {}).get("jira_jql")
    if jql is not None and jql != _OLD_JQL_DEFAULT:
        return None

    new = json.loads(json.dumps(_NEW_SPEC))
    new["name"] = spec.get("name") or new["name"]
    new["mode"] = spec.get("mode") or new["mode"]
    if spec.get("focus"):
        new["focus"] = spec["focus"]
    desc = spec.get("description")
    if desc and desc != _OLD_DESCRIPTION:
        new["description"] = desc
    # The schedule they are on, not the one the catalog ships — an
    # edited time is an edit, and the compiled binding is built from it.
    new["trigger"] = spec.get("trigger") or new["trigger"]
    # The destination. `grant_target` is what `run_blockers` reads to
    # decide the automation has somewhere to post, and `grant_id` is
    # what the dispatcher re-verifies at call time.
    post = new["steps"][-1]
    for key in ("grant_id", "grant_target"):
        if old_post.get(key):
            post[key] = old_post[key]
    return new


def upgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)
    if _AUTOMATIONS not in set(insp.get_table_names()):
        logger.info("[alembic.099] no %s table here (platform DB, or a "
                    "tenant before its first init_db) — nothing to do",
                    _AUTOMATIONS)
        return

    rows = conn.execute(sa.text(
        "SELECT id, spec_json, steps_human_json, description "
        f"FROM {_AUTOMATIONS} "
        "WHERE template_slug = :slug AND deleted_at IS NULL"
    ), {"slug": _SLUG}).fetchall()

    migrated = skipped = 0
    for row in rows:
        if (row.steps_human_json or "").strip() not in ("", "[]"):
            logger.info("[alembic.099] %s: the user wrote its steps — "
                        "left alone", row.id)
            skipped += 1
            continue
        try:
            spec = json.loads(row.spec_json or "{}")
        except (ValueError, TypeError):
            spec = None
        new = _target(spec) if isinstance(spec, dict) else None
        if new is None:
            logger.info("[alembic.099] %s: not the shipped shape any more "
                        "— left alone", row.id)
            skipped += 1
            continue
        params: dict = {
            "spec": json.dumps(new, sort_keys=True),
            "id": row.id,
        }
        desc_sql = ""
        if row.description == _OLD_DESCRIPTION:
            # Kept in step with the spec: `create_automation` copies one
            # into the other, and the app reads the column.
            desc_sql = ", description = :description"
            params["description"] = new["description"]
        conn.execute(sa.text(
            f"UPDATE {_AUTOMATIONS} SET spec_json = :spec{desc_sql}, "
            "workflow_rev = workflow_rev + 1 WHERE id = :id"
        ), params)
        migrated += 1

    logger.info("[alembic.099] morning brief: %d re-spec'd, %d left alone",
                migrated, skipped)


def downgrade() -> None:
    """One-way.

    The old spec is not reconstructible from this side — it carried a
    per-user Jira answer, a GitHub owner/repo and a Teams chat id that
    this migration deliberately does not keep, because the new spec has
    nowhere to put them. Re-adopting the template is the route back.
    """
    logger.info("[alembic.099] no downgrade: the pre-R42 spec cannot be "
                "reconstructed from the new one")
