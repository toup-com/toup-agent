"""Routines Skill — agent self-authors scheduled automations.

Lets the user create/manage routines from chat: "from now on read all my
emails before I wake up and summarize them" → agent calls
`routines__create` with kind=email_briefing + a sensible cron.

Six tools, all single-user-per-container (no user_id arg — read from
`settings.user_id`):

  routines__create   — create one routine (email_briefing | agent_task)
  routines__remind   — friendly shortcut for kind=reminder. Accepts
                       human-shape inputs (once / daily / every) and
                       translates them to the underlying schedule_kind.
                       Use this instead of `routines__create` when the
                       user asks for a reminder/alert/nudge.
  routines__list     — list this user's routines (id, kind, schedule, status)
  routines__update   — toggle / reschedule / rename / rewrite prompt
  routines__delete   — drop a routine
  routines__run_now  — fire today's slot immediately (idempotent — 409 if today
                       already ran)

The skill is a thin wrapper around the same HTTP handlers Mission Control
uses (`app.api.routines`). Calling the handlers in-process keeps validation
+ feature-flag gates + scheduler registration as the single source of truth.
HTTPException is caught and surfaced as `ERROR: …` so the agent can recover
or explain to the user.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from app.agent.skills.base import Skill, SkillContext, SkillMeta

logger = logging.getLogger(__name__)


def _as_json(obj: Any) -> str:
    """Stable JSON for tool return values — small payloads only, so we
    don't bother with truncation. Datetimes serialise via `default=str`."""
    return json.dumps(obj, indent=2, default=str, ensure_ascii=False)


def _is_valid_tz(name: Optional[str]) -> bool:
    """True if `name` is a resolvable IANA zone (e.g. 'America/Toronto')."""
    if not name:
        return False
    try:
        from zoneinfo import ZoneInfo  # py3.9+
    except ImportError:  # pragma: no cover
        from backports.zoneinfo import ZoneInfo  # type: ignore
    try:
        ZoneInfo(name)
        return True
    except Exception:
        return False


def _infer_tz_from_phone(e164: Optional[str]) -> Optional[str]:
    """Best-effort IANA timezone from an E.164 phone number.

    WhatsApp / Telegram users never send a browser timezone, so the
    linked phone number is the only location signal we have for them —
    without this, a reminder is impossible (the reason a real WhatsApp
    user hit "the reminder tool isn't working"). `phonenumbers` is
    offline and area-code aware (+1 437… → America/Toronto). Returns
    None if the dep is missing or the number is unparseable, so the
    caller falls back to asking the user.
    """
    if not e164:
        return None
    try:
        import phonenumbers
        from phonenumbers.timezone import time_zones_for_number
        num = phonenumbers.parse(e164, None)
        zones = [
            z for z in time_zones_for_number(num)
            if z and z != "Etc/Unknown"
        ]
        return zones[0] if zones else None
    except Exception:
        logger.warning(
            "[routines_skill] phone→tz inference unavailable for %s…",
            (e164 or "")[:5], exc_info=True,
        )
        return None


def _routine_summary(r) -> Dict[str, Any]:
    """Compact dict the agent can reason over. Mirrors RoutineResponse
    minus recent_runs (the agent rarely needs them; if it does, it calls
    `routines__list` again or asks the user)."""
    return {
        "id": r.id,
        "kind": r.kind,
        "name": getattr(r, "name", None),
        "prompt_text": getattr(r, "prompt_text", None),
        "schedule_cron_local": r.schedule_cron_local,
        "enabled": bool(r.enabled),
        "last_status": r.last_status,
        "last_run_at": r.last_run_at,
        "next_run_at": r.next_run_at,
        "last_error": r.last_error,
    }


# W2.1a prefix diet (settings.prompt_diet): compact descriptions for the two
# fattest routine schemas (routines__remind 998 tok, routines__create 766 tok
# measured on the wire — docs/audits/2026-07-sota-assessment.md). Only
# description strings shrink; properties/enums/required are byte-identical
# to the full schemas (shape equality pinned in tests/test_prompt_diet.py).
_DIET_TOOL_DESCRIPTIONS = {
    "routines__create": (
        "Create a scheduled routine (recurring agent work — \"every "
        "morning\", \"weekdays at 7am\"). kind=`email_briefing` = Gmail "
        "summary preset (config.mode `latest_n` + max_emails, or "
        "`since_last_run`; no prompt_text). kind=`agent_task` = anything "
        "else — REQUIRES a self-contained `prompt_text` (fires in a fresh "
        "context). `schedule_cron_local` is 5-part cron in the user's local "
        "tz (`30 6 * * *` = 06:30 daily; `0 7 * * 1-5` = 07:00 weekdays) — "
        "confirm the time with the user first. Delivery is automatic to "
        "chat + every connected channel: never ask where to send it; OMIT "
        "`delivery_channels` unless the user explicitly restricts it."
    ),
    "routines__remind": (
        "Create a reminder — literal text delivered at a time, no LLM. Use "
        "instead of `routines__create` for any remind / alert / nudge; call "
        "ONCE per request. `when`: `once` ('in N min' → `in_seconds`, never a "
        "computed clock time; 'at 8:15' → `at_local`), `daily` "
        "(+`daily_at_local`), `every` (+`every_seconds`, optional "
        "`window_start_local`/`window_end_local`; none = 24/7). Times are the "
        "user's local tz. Delivery is automatic to chat + every connected "
        "channel: never ask where; OMIT `delivery_channels` unless the user "
        "restricts it."
    ),
}

_DIET_PROPERTY_DESCRIPTIONS = {
    "routines__create": {
        "kind": "`email_briefing` = Gmail preset; `agent_task` = any other recurring prompt.",
        "schedule_cron_local": "5-part cron, user's local tz, e.g. `30 6 * * *`.",
        "name": "Short name (≤100 chars) shown to the user in their list.",
        "prompt_text": (
            "REQUIRED for kind=`agent_task`: self-contained instruction "
            "executed at fire time (fresh context, no memory of this chat)."
        ),
        "delivery_channels": (
            "OMIT for the default (chat + every connected channel) and do "
            "NOT ask the user where to send it. Set only on explicit user "
            "restriction; `website` always included."
        ),
    },
    "routines__remind": {
        "reminder_text": "Literal text delivered at fire time, written for the user to read.",
        "when": "`once` = one-shot + auto-disable; `daily` = same time every day; `every` = interval.",
        "in_seconds": (
            "when=`once`, RELATIVE ('in 1 min' → 60, 'in 90 seconds' → 90, "
            "'in 2 hours' → 7200): seconds from now. Use instead of `at_local`."
        ),
        "at_local": (
            "when=`once`, ABSOLUTE: \"YYYY-MM-DD HH:MM[:SS]\" or \"HH:MM[:SS]\" "
            "(today if still future, else tomorrow). Relative → `in_seconds`."
        ),
        "daily_at_local": "Required when when=`daily`: \"HH:MM\" local.",
        "every_seconds": "Required when when=`every`: interval seconds, min 60.",
        "window_start_local": "Optional with `every`: \"HH:MM\" — fire only after this time each day.",
        "window_end_local": (
            "Optional with `every`: \"HH:MM\" window end; wraps midnight "
            "if end < start."
        ),
        "name": "Optional short name shown to the user (defaults from reminder_text).",
        "delivery_channels": (
            "OMIT for the default (chat + every connected channel) and do "
            "NOT ask the user where to send it. Set only on explicit user "
            "restriction; `website` always included."
        ),
        "timezone": (
            "IANA tz (e.g. \"America/Toronto\") — pass only after a "
            "`NEEDS_TIMEZONE` error; remembered afterwards."
        ),
    },
}


class RoutinesSkill(Skill):
    """Expose routine CRUD to the agent itself.

    Production-quality contract:
      • Validates kind + cron + prompt requirements via the same path the
        HTTP API uses, so an agent-created routine and a Mission Control
        routine are indistinguishable on disk.
      • Triggers scheduler reload immediately — the routine fires at its
        very next cron tick without an agent restart.
      • Surfaces feature-flag-off as a clear error so the agent can tell
        the user "routines are disabled for this tenant" instead of
        silently noop'ing.
    """

    meta = SkillMeta(
        name="routines",
        version="1.0.0",
        description=(
            "Reminders and scheduled tasks: morning email briefings, "
            "daily check-ins, recurring agent work."
        ),
        author="Toup",
    )

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------
    def get_tools(self) -> List[Dict[str, Any]]:
        tools = [
            {
                "name": "routines__create",
                "description": (
                    "Create a new scheduled routine for the user. Use this when "
                    "the user asks the agent to do something on a recurring "
                    "schedule (\"every morning\", \"weekdays at 7am\", \"every "
                    "Monday\"). Returns the created routine's id + schedule.\n\n"
                    "Picking `kind`:\n"
                    "  • `email_briefing` — Gmail summary preset. Two sub-modes "
                    "via `config.mode`:\n"
                    "      - `latest_n` — \"give me my latest 5 emails every "
                    "morning\". Most recent N regardless of when they arrived. "
                    "Set `config={\"mode\":\"latest_n\",\"max_emails\":5}` (or "
                    "whatever N the user asked for).\n"
                    "      - `since_last_run` (default) — \"summarize my new "
                    "emails since the last briefing\". Posts \"No new emails\" "
                    "on a quiet day. Use only if the user explicitly wants the "
                    "watermark-driven briefing.\n"
                    "    NO prompt_text needed for either sub-mode.\n"
                    "  • `agent_task` — everything else. REQUIRES `prompt_text` "
                    "describing what the agent should do at fire time.\n\n"
                    "Cron format: 5-part `m h dom mon dow` in the user's local "
                    "timezone. Examples: `30 6 * * *` = 06:30 every day; "
                    "`0 7 * * 1-5` = 07:00 weekdays; `0 9 * * 0` = 09:00 Sundays. "
                    "Confirm the time with the user BEFORE calling this tool.\n\n"
                    "**Delivery:** the routine is delivered to the chat AND "
                    "every channel the user has connected (Telegram, WhatsApp) "
                    "automatically. Do NOT ask where to send it — omit "
                    "`delivery_channels` entirely and the default covers "
                    "everything. Only pass it when the user EXPLICITLY "
                    "restricts delivery (\"only telegram\" → [\"telegram\"]; "
                    "\"just here in chat\" → [\"website\"]). The website chat "
                    "is always included as the permanent record."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "kind": {
                            "type": "string",
                            "enum": ["email_briefing", "agent_task"],
                            "description": (
                                "`email_briefing` for Gmail summaries; "
                                "`agent_task` for any other recurring prompt."
                            ),
                        },
                        "schedule_cron_local": {
                            "type": "string",
                            "description": (
                                "5-part cron expression in the user's local tz, "
                                "e.g. `30 6 * * *` for 06:30 daily."
                            ),
                        },
                        "name": {
                            "type": "string",
                            "description": (
                                "Short human-readable name (≤100 chars), e.g. "
                                "\"Morning email briefing\" or \"Check deploys\". "
                                "Shown to the user in their list."
                            ),
                        },
                        "prompt_text": {
                            "type": "string",
                            "description": (
                                "REQUIRED when kind=`agent_task`. The full "
                                "natural-language instruction the agent will "
                                "execute at fire time. Write it as a "
                                "self-contained prompt — the fire is a fresh "
                                "context with no memory of this conversation."
                            ),
                        },
                        "enabled": {
                            "type": "boolean",
                            "description": "Default true. Set false to create dormant.",
                        },
                        "delivery_channels": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["website", "telegram", "whatsapp"],
                            },
                            "description": (
                                "OMIT this field for the default: chat + every "
                                "connected channel, automatically — do NOT ask. "
                                "Only set it when the user explicitly names "
                                "channels (\"only telegram\" → [\"telegram\"]; "
                                "\"just here in chat\" → [\"website\"]). "
                                "`website` is always included server-side."
                            ),
                        },
                    },
                    "required": ["kind", "schedule_cron_local"],
                },
            },
            {
                "name": "routines__remind",
                "description": (
                    "Friendly shortcut for creating a reminder (kind=`reminder`). "
                    "USE THIS instead of `routines__create` whenever the user "
                    "asks for a reminder / alert / nudge / heads-up — anything "
                    "where the value is *literal text delivered at a time*, NOT "
                    "an LLM-generated summary.\n\n"
                    "Three modes via `when`:\n"
                    "  • `once` — fire one time, then auto-disable. RELATIVE "
                    "requests ('in 1 min', 'in 90 seconds', 'in 2 hours') → pass "
                    "`in_seconds` (60 / 90 / 7200) and NOTHING else for the time; "
                    "never convert a relative delay into a wall-clock `at_local` — "
                    "the clock you see is minute-resolution and the reminder would "
                    "fire early. ABSOLUTE requests ('at 8:15', 'tomorrow 6pm') → "
                    "pass `at_local` as `\"YYYY-MM-DD HH:MM[:SS]\"` or "
                    "`\"HH:MM[:SS]\"` (today if still future, else tomorrow). "
                    "Example: \"remind me to call mom at 6pm\".\n"
                    "  • `daily` — fire every day at the same wall-clock time. "
                    "Pass `daily_at_local=\"HH:MM\"`. Example: \"remind me to "
                    "drink water every morning at 9\".\n"
                    "  • `every` — fire on an interval. Pass `every_seconds` "
                    "(min 60). Optional `window_start_local` + `window_end_local` "
                    "(HH:MM) gate fires to a daily wall-clock window — e.g. "
                    "\"every 30 minutes between 9am and 5pm\" → "
                    "`every_seconds=1800, window_start_local=\"09:00\", "
                    "window_end_local=\"17:00\"`. Without a window the interval "
                    "runs 24/7.\n\n"
                    "The skill resolves the user's timezone server-side; all "
                    "the time inputs are interpreted in their local tz. "
                    "**Delivery**: do NOT ask where to deliver. Omitting "
                    "`delivery_channels` delivers the reminder to the chat AND "
                    "every channel the user has connected (Telegram, WhatsApp) "
                    "automatically. Pass it only when the user explicitly "
                    "restricts delivery (\"only telegram\" → [\"telegram\"]). "
                    "Call this tool exactly ONCE per reminder the user asked for; "
                    "an identical reminder within the same minute is refused as a "
                    "duplicate."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "reminder_text": {
                            "type": "string",
                            "description": (
                                "The literal text delivered at fire time. The "
                                "reminder is text-only — no LLM, no tools. Write "
                                "it the way you want the user to read it, "
                                "e.g. \"Time to take your vitamins\"."
                            ),
                        },
                        "when": {
                            "type": "string",
                            "enum": ["once", "daily", "every"],
                            "description": (
                                "Schedule shape. `once` = one-shot + auto-disable. "
                                "`daily` = same time every day. `every` = on "
                                "an interval (optionally bounded by a daily window)."
                            ),
                        },
                        "in_seconds": {
                            "type": "integer",
                            "description": (
                                "when=`once`, RELATIVE request: seconds from now "
                                "('in 1 min' → 60, 'in 90 seconds' → 90, "
                                "'in 2 hours' → 7200). Preferred over `at_local` "
                                "for anything phrased as a delay. Min 5."
                            ),
                        },
                        "at_local": {
                            "type": "string",
                            "description": (
                                "when=`once`, ABSOLUTE time: either "
                                "\"YYYY-MM-DD HH:MM[:SS]\" (explicit date+time) or "
                                "\"HH:MM[:SS]\" (today if still in the future, else "
                                "tomorrow). Interpreted in the user's local tz. "
                                "Do NOT use for 'in N minutes' — pass `in_seconds`."
                            ),
                        },
                        "daily_at_local": {
                            "type": "string",
                            "description": (
                                "Required when when=`daily`. \"HH:MM\" in user's "
                                "local tz, e.g. \"07:30\"."
                            ),
                        },
                        "every_seconds": {
                            "type": "integer",
                            "description": (
                                "Required when when=`every`. Interval in seconds "
                                "(minimum 60). 1800 = every 30 min; 3600 = hourly."
                            ),
                        },
                        "window_start_local": {
                            "type": "string",
                            "description": (
                                "Optional with when=`every`. \"HH:MM\" — only "
                                "fire after this local time each day."
                            ),
                        },
                        "window_end_local": {
                            "type": "string",
                            "description": (
                                "Optional with when=`every`. \"HH:MM\" — stop "
                                "firing after this local time each day. Pair "
                                "with `window_start_local`. Wraps midnight if "
                                "end < start (e.g. 22:00 → 06:00 = overnight)."
                            ),
                        },
                        "name": {
                            "type": "string",
                            "description": (
                                "Optional short name shown to the user. "
                                "Defaults to first 60 chars of `reminder_text`."
                            ),
                        },
                        "delivery_channels": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["website", "telegram", "whatsapp"],
                            },
                            "description": (
                                "OMIT for the default: chat + every connected "
                                "channel, automatically — do NOT ask. Only set "
                                "it when the user explicitly names channels; "
                                "`website` is always included server-side."
                            ),
                        },
                        "enabled": {
                            "type": "boolean",
                            "description": "Default true. False = create dormant.",
                        },
                        "timezone": {
                            "type": "string",
                            "description": (
                                "Optional IANA timezone, e.g. \"America/Toronto\". "
                                "Only needed if a previous call returned "
                                "`NEEDS_TIMEZONE`: ask the user which city/zone "
                                "they're in, map it to the IANA name, and pass it "
                                "here. It's saved after the first time, so you "
                                "won't be asked again."
                            ),
                        },
                    },
                    "required": ["reminder_text", "when"],
                },
            },
            {
                "name": "routines__list",
                "description": (
                    "List the user's existing reminders and scheduled tasks. "
                    "Call this BEFORE creating a new one to check for "
                    "duplicates. These are NOT the user's automations — "
                    "automations are a separate surface with their own tools "
                    "and their own list, so never answer a question about "
                    "automations from this one. "
                    "Returns id, kind, name, schedule, enabled flag, last_status, "
                    "next_run_at."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "name": "routines__update",
                "description": (
                    "Update an existing routine's schedule, name, prompt, "
                    "delivery channels, or enabled flag. Use the `id` from "
                    "`routines__list`. Any field left unset is preserved. "
                    "Triggers an immediate scheduler reload — the next fire "
                    "honours the new schedule."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "routine_id": {
                            "type": "string",
                            "description": "UUID from `routines__list`.",
                        },
                        "schedule_cron_local": {
                            "type": "string",
                            "description": "New cron, same 5-part format as create.",
                        },
                        "enabled": {
                            "type": "boolean",
                            "description": "true to enable, false to pause.",
                        },
                        "name": {"type": "string"},
                        "prompt_text": {
                            "type": "string",
                            "description": "Only meaningful for kind=`agent_task`.",
                        },
                        "delivery_channels": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["website", "telegram", "whatsapp"],
                            },
                            "description": (
                                "New delivery-channel list. Replaces the "
                                "current setting entirely."
                            ),
                        },
                    },
                    "required": ["routine_id"],
                },
            },
            {
                "name": "routines__delete",
                "description": (
                    "Permanently delete a routine and all its run history. "
                    "Use sparingly — prefer `routines__update` with "
                    "`enabled=false` if the user might want it back."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "routine_id": {
                            "type": "string",
                            "description": "UUID from `routines__list`.",
                        },
                    },
                    "required": ["routine_id"],
                },
            },
            {
                "name": "routines__run_now",
                "description": (
                    "Fire today's slot of a routine immediately, without waiting "
                    "for its next scheduled time. Idempotent — if today's run "
                    "already exists (scheduled or force-run), returns an error "
                    "with the existing run's status. Useful for \"run my "
                    "morning briefing now\" mid-day."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "routine_id": {
                            "type": "string",
                            "description": "UUID from `routines__list`.",
                        },
                    },
                    "required": ["routine_id"],
                },
            },
        ]
        # W2.1a prefix diet — compact descriptions only; shapes untouched.
        # Flag-off returns the list above byte-identical.
        from app.agent.prompt_diet import (
            prompt_diet_enabled, apply_tool_description_diet,
        )
        if prompt_diet_enabled():
            apply_tool_description_diet(
                tools, _DIET_TOOL_DESCRIPTIONS, _DIET_PROPERTY_DESCRIPTIONS,
            )
        return tools

    # ------------------------------------------------------------------
    # System prompt — guides when + how to call these tools
    # ------------------------------------------------------------------
    def get_system_prompt_section(self) -> Optional[str]:
        return (
            "# Reminders and scheduled tasks\n"
            "You can put things on a schedule for the user via the "
            "`routines__*` tools — reminders (literal text delivered at a "
            "time) and scheduled tasks (you do the work and report back). "
            "Use this flow whenever the user asks for something on a "
            "recurring schedule:\n"
            "  • \"From now on, read my emails before I wake up and summarize them\"\n"
            "  • \"Every weekday at 7am, give me my calendar for the day\"\n"
            "  • \"Check my GitHub notifications every hour during work hours\"\n"
            "  • \"Remind me to call mom at 6pm\" — *use `routines__remind`*\n"
            "  • \"Buzz me every 30 minutes between 9 and 5 to stretch\" — *use `routines__remind`*\n"
            "\n"
            "**These are not the user's automations.** Automations are a "
            "separate surface with their own tools (`automations__*`) and "
            "their own list. When the user asks what automations they have, "
            "or how many, answer from `automations__list` alone — never fold "
            "reminders or scheduled tasks into that list or that count. When "
            "they ask about these, call them reminders and scheduled tasks.\n"
            "\n"
            "**Tool to pick:**\n"
            "  • If the user wants literal text delivered at a time "
            "(reminder / alert / nudge — \"remind me to X\", \"ping me at Y\"), "
            "use **`routines__remind`** with the friendly `when` modes "
            "(once / daily / every). No LLM call, no MCP — just text on a "
            "schedule. A RELATIVE ask (\"in 1 min\", \"in 90 seconds\", "
            "\"in 2 hours\") is `when=once` + `in_seconds` (60 / 90 / 7200) — "
            "never convert it to a clock time yourself; the clock you see has "
            "no seconds and the reminder would fire early. An ABSOLUTE ask "
            "(\"at 8:15\", \"tomorrow at 6pm\") is `at_local`. Call the tool "
            "exactly ONCE per reminder — one request, one call, one "
            "confirmation sentence.\n"
            "  • If the user wants the agent to DO something "
            "(\"summarize my emails\", \"check my GitHub\"), use "
            "`routines__create` with kind=`email_briefing` or `agent_task`.\n"
            "\n"
            "**Flow:**\n"
            "  1. Confirm the schedule with the user in plain English (\"so "
            "you want this at 6:30am every day, right?\"). The user thinks in "
            "their local time — never echo UTC.\n"
            "  2. **NEVER ask where to deliver it.** Reminders and routines "
            "automatically go to the chat AND every channel the user has "
            "connected (Telegram, WhatsApp) — omit `delivery_channels` and "
            "the default covers everything. Never phrase a question like "
            "\"where do you want to see this?\". Pass the field ONLY when "
            "the user explicitly restricts delivery:\n"
            "     - \"just here\" / \"only the chat\" → `[\"website\"]`\n"
            "     - \"only Telegram\" → `[\"telegram\"]`\n"
            "     - \"only WhatsApp\" → `[\"whatsapp\"]`\n"
            "     (The website is always kept server-side so the user has a "
            "permanent record on their dashboard even when the buzz goes "
            "to phone.)\n"
            "  3. Pick the kind:\n"
            "     - `email_briefing` for \"summarize my unread emails\" "
            "(preset Gmail flow — no prompt needed).\n"
            "     - `agent_task` for everything else. Write a self-contained "
            "`prompt_text` — the fire runs in a fresh context with no memory "
            "of this conversation, so spell out exactly what to do.\n"
            "  4. Convert the spoken schedule to a 5-part cron in the user's "
            "tz: `30 6 * * *` = 06:30 daily; `0 7 * * 1-5` = 07:00 weekdays.\n"
            "  5. Call `routines__create` with kind + schedule + name "
            "(+ prompt_text for agent_task). Confirm success to the user, "
            "mention the channels it will arrive on (read them from the tool "
            "result's `delivery_channels`), and tell them they can change or "
            "cancel it later from the dashboard.\n"
            "\n"
            "**Before creating a ROUTINE (`routines__create`), call "
            "`routines__list` first** to check if a similar routine already "
            "exists — if so, offer to update it instead of duplicating. (Note: "
            "`email_briefing` is one-per-user; the API will 409 a duplicate. "
            "`agent_task` allows many.) A plain reminder does NOT need the "
            "list call — `routines__remind` refuses an exact duplicate itself; "
            "just call it once.\n"
            "\n"
            "Routines are flag-gated. If `routines__create` returns "
            "`ERROR: Feature not available`, tell the user the feature isn't "
            "enabled for their tenant — don't retry.\n"
            "\n"
            "If `routines__remind` returns `NEEDS_TIMEZONE`, do NOT tell the "
            "user the reminder tool is broken — it works, it just needs their "
            "timezone once. Ask which city or timezone they're in, convert "
            "that to an IANA zone (e.g. Toronto → \"America/Toronto\"), and "
            "call `routines__remind` again with the same details plus "
            "`timezone`. It's remembered afterwards."
        )

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------
    async def execute_tool(
        self,
        tool_name: str,
        args: Dict[str, Any],
        ctx: SkillContext,
    ) -> str:
        dispatch = {
            "routines__create": self._create,
            "routines__remind": self._remind,
            "routines__list": self._list,
            "routines__update": self._update,
            "routines__delete": self._delete,
            "routines__run_now": self._run_now,
        }
        handler = dispatch.get(tool_name)
        if not handler:
            return f"ERROR: Unknown routines tool: {tool_name}"
        try:
            return await handler(args, ctx)
        except Exception as e:
            # Last-resort guard — handlers already convert HTTPException
            # to ERROR strings. Anything reaching here is a bug in the
            # skill itself, not user-input fault.
            logger.exception("[routines_skill] unexpected error in %s", tool_name)
            return f"ERROR: {type(e).__name__}: {str(e)[:300]}"

    # ------------------------------------------------------------------
    # Handlers — delegate to app.api.routines for single source of truth
    # ------------------------------------------------------------------
    async def _remind(self, args: Dict[str, Any], ctx: SkillContext) -> str:
        """Friendly wrapper that translates ``when`` + local-time inputs
        into the underlying RoutineCreate schedule shape.

        The user thinks in their local wall clock; the API stores
        ``schedule_at`` in UTC, so we resolve the user's tz here and do
        the conversion before delegating to ``create_routine``.
        """
        from datetime import datetime, time, timedelta
        from fastapi import HTTPException
        from app.config import settings
        from app.api.routines import RoutineCreate, create_routine
        from app.agent.routines.runner import _resolve_tz

        text = (args.get("reminder_text") or "").strip()
        when = (args.get("when") or "").strip().lower()
        if not text:
            return "ERROR: `reminder_text` is required."
        if when not in {"once", "daily", "every"}:
            return "ERROR: `when` must be one of: once, daily, every."

        # ── Up-front input validation (no I/O) ─────────────────────────
        # Surface obvious shape errors BEFORE the DB tz lookup so a bad
        # interval / malformed HH:MM doesn't masquerade as a tz problem
        # and confuse the agent. Each branch records its inputs and
        # defers the per-kind payload build until after we have `tz`.
        once_raw_at: Optional[str] = None
        once_in_seconds: Optional[int] = None
        daily_hhmm: Optional[tuple[int, int]] = None
        every_secs: Optional[int] = None
        window_start: Optional[str] = None
        window_end: Optional[str] = None

        if when == "once":
            # Round 4 (item 5b): a RELATIVE reminder is a delay from NOW,
            # computed HERE at execution time to the second. The model used
            # to translate "in 1 min" into a wall-clock "HH:MM" from a
            # minute-resolution clock, and the parser zeroed the seconds —
            # so "in 1 min" always fired at the next minute BOUNDARY (the
            # founder's 27 s). `in_seconds` wins when both are present.
            _in_s = args.get("in_seconds")
            if _in_s is not None and not isinstance(_in_s, bool):
                try:
                    _in_s = int(_in_s)
                except (TypeError, ValueError):
                    return f"ERROR: `in_seconds` must be an integer number of seconds (got {_in_s!r})."
                if _in_s < 5:
                    return "ERROR: `in_seconds` must be at least 5."
                if _in_s > 366 * 86400:
                    return "ERROR: `in_seconds` is more than a year — pass an `at_local` date instead."
                once_in_seconds = _in_s
            raw = (args.get("at_local") or "").strip()
            if once_in_seconds is None and not raw:
                return (
                    "ERROR: when=`once` needs a time: `in_seconds` for a relative "
                    "reminder ('in 1 min' → 60) or `at_local` (\"YYYY-MM-DD HH:MM\" "
                    "/ \"HH:MM\") for an absolute one."
                )
            once_raw_at = raw or None
        elif when == "daily":
            raw = (args.get("daily_at_local") or "").strip()
            hhmm = self._parse_hhmm(raw)
            if hhmm is None:
                return (
                    "ERROR: `daily_at_local` must be \"HH:MM\" "
                    f"(got {raw!r})."
                )
            daily_hhmm = hhmm
        else:  # when == "every"
            secs = args.get("every_seconds")
            if not isinstance(secs, int) or secs < 60:
                return (
                    "ERROR: `every_seconds` must be an integer ≥ 60 "
                    f"(got {secs!r})."
                )
            every_secs = secs
            for k, target in (
                ("window_start_local", "window_start"),
                ("window_end_local", "window_end"),
            ):
                v = args.get(k)
                if v:
                    if self._parse_hhmm(v) is None:
                        return f"ERROR: `{k}` must be \"HH:MM\" (got {v!r})."
                    if target == "window_start":
                        window_start = v
                    else:
                        window_end = v

        # User-visible name defaults to a trimmed slice of the reminder
        # text so Mission Control shows something readable even when the
        # agent didn't pass `name`.
        name = (args.get("name") or "").strip() or text[:60]

        delivery = args.get("delivery_channels")
        enabled = bool(args.get("enabled", True))

        # Resolve user's tz — required for `once` (local→UTC conversion)
        # AND for daily (formatting the cron string in their wall clock).
        # `_resolve_tz` returns (ZoneInfo, fellback_to_utc_bool); we
        # surface the fallback to the agent so it can warn the user.
        user_id = str(getattr(settings, "user_id", "") or "")

        # Optional tz the agent learned from the user this turn ("I'm in
        # Toronto" → "America/Toronto"), passed back after a prior
        # NEEDS_TIMEZONE. Reject an invalid name up front so it can't slip
        # through and silently fall back to UTC.
        arg_tz = (args.get("timezone") or "").strip() or None
        if arg_tz and not _is_valid_tz(arg_tz):
            return (
                f"ERROR: `timezone`={arg_tz!r} isn't a valid IANA zone. Use a "
                "name like \"America/Toronto\" or \"Europe/London\"."
            )

        try:
            from app.db.database import async_session_maker
            from app.db.models import User
            from sqlalchemy import select
            user_tz_str = None
            resolved = None
            source = "stored"
            async with async_session_maker() as db:
                row = None
                if user_id:
                    row = (await db.execute(
                        select(User).where(User.id == user_id)
                    )).scalar_one_or_none()
                    user_tz_str = getattr(row, "timezone", None) if row else None

                # Resolution chain for users with no stored tz. WhatsApp /
                # Telegram users never send a browser tz, so without this a
                # reminder is impossible for them. Order: stored → tz the
                # agent passed (learned from the user) → inferred from the
                # linked phone number. Anything resolved here is persisted
                # once so every later time-based feature works without asking.
                resolved = user_tz_str if _is_valid_tz(user_tz_str) else None
                if not resolved and arg_tz:
                    resolved, source = arg_tz, "user_provided"
                if not resolved:
                    inferred = _infer_tz_from_phone(
                        getattr(settings, "whatsapp_self_e164", None)
                    )
                    if inferred:
                        resolved, source = inferred, "phone_number"
                if resolved and resolved != user_tz_str and row is not None:
                    try:
                        row.timezone = resolved
                        await db.commit()
                        logger.info(
                            "[routines_skill.remind] self-healed tz user=%s "
                            "source=%s tz=%s",
                            user_id[:8], source, resolved,
                        )
                    except Exception:
                        logger.warning(
                            "[routines_skill.remind] tz persist failed",
                            exc_info=True,
                        )
            tz, fell_back = _resolve_tz(resolved, user_id)
        except Exception as e:
            logger.exception("[routines_skill.remind] tz lookup failed")
            return f"ERROR: could not resolve your timezone — {e}"
        if fell_back:
            # We have no timezone and couldn't infer one. Do NOT fail with a
            # bare ERROR — the agent paraphrases that as "the reminder tool is
            # broken". Tell it to ask the user and retry with `timezone`.
            return (
                "NEEDS_TIMEZONE: I don't know this user's timezone yet, so I "
                "can't schedule at the right local time. Ask them which city "
                "or timezone they're in, then call routines__remind again with "
                "the same details plus `timezone` set to the matching IANA "
                "zone (e.g. \"America/Toronto\"). The tool works — it just "
                "needs the timezone once."
            )

        # Model omitted the param → default to every channel the user is
        # actually connected to (chat + Telegram + WhatsApp), never ask
        # (founder decision 2026-07-17). An explicit list ("only
        # telegram") passes through untouched; create_routine still
        # force-includes website as the canonical record.
        if not delivery:
            try:
                from app.agent.routines.channel_dispatcher import (
                    get_connected_channels,
                )
                delivery = await get_connected_channels(
                    user_id, async_session_maker,
                )
            except Exception:
                logger.warning(
                    "[routines_skill.remind] connected-channel probe failed "
                    "— falling back to website-only", exc_info=True,
                )
                delivery = None

        # ── Build the create payload now that tz is resolved ───────────
        # When the reminder was SET — the request-anchored base for relative
        # one-shots (overwritten below), creation time otherwise. Rides the
        # result as `created_at_utc` so every client surface draws the
        # set→fire span from one absolute base.
        from datetime import timezone as _tz_set
        set_at_naive = datetime.now(_tz_set.utc).replace(tzinfo=None, microsecond=0)
        payload: Dict[str, Any] = {
            "kind": "reminder",
            "name": name,
            "reminder_text": text,
            "enabled": enabled,
            "delivery_channels": delivery,
        }

        if when == "once":
            from datetime import timezone as _tzmod
            if once_in_seconds is not None:
                # Relative: exact seconds from when the user SAID it — the
                # turn's receipt time when the runner set it (the planning
                # round before this tool runs is 5–7 s, measured; counting
                # from execution made "in 60 seconds" ring at +67). Falls
                # back to now outside a turn. Never in the past: a slow turn
                # on a short delay fires in a moment. APScheduler's
                # DateTrigger fires at this instant; no rounding anywhere.
                _base = None
                try:
                    from app.agent.tool_executor import turn_started_at as _tsa
                    _base = _tsa()
                except Exception:  # noqa: BLE001
                    _base = None
                _now_utc = datetime.now(_tzmod.utc)
                if _base:
                    _base_dt = datetime.fromtimestamp(float(_base), tz=_tzmod.utc)
                    # sanity: a stale/foreign value more than 10 min off is ignored
                    if abs((_now_utc - _base_dt).total_seconds()) > 600:
                        _base_dt = _now_utc
                else:
                    _base_dt = _now_utc
                _target = _base_dt + timedelta(seconds=once_in_seconds)
                if _target < _now_utc + timedelta(seconds=2):
                    _target = _now_utc + timedelta(seconds=2)
                # Round to the NEAREST second — replace(microsecond=0) alone
                # truncated DOWN, shaving up to 0.999s off the requested
                # offset on top of the anchor's own lag.
                if _target.microsecond >= 500_000:
                    _target += timedelta(seconds=1)
                dt_utc = _target.replace(tzinfo=None, microsecond=0)
                # The instant the offset was counted from — the result carries
                # it as created_at_utc so every client surface can draw the
                # set→fire span from the same absolute base.
                set_at_naive = _base_dt.replace(tzinfo=None, microsecond=0)
            else:
                dt_local = self._parse_local_datetime(once_raw_at or "", tz)
                if dt_local is None:
                    return (
                        f"ERROR: couldn't parse at_local={once_raw_at!r}. Use "
                        "\"YYYY-MM-DD HH:MM[:SS]\" or \"HH:MM[:SS]\" — or `in_seconds` "
                        "for a relative reminder."
                    )
                # Translate local wall-clock to a UTC-naive datetime, which
                # is the shape `schedule_at` expects.
                dt_utc = dt_local.astimezone(_tzmod.utc).replace(tzinfo=None)
            payload["schedule_kind"] = "at"
            payload["schedule_at"] = dt_utc
            # auto_disable defaults true server-side for 'at'; setting it
            # explicitly is harmless and makes intent obvious in logs.
            payload["auto_disable_after_fire"] = True
        elif when == "daily":
            assert daily_hhmm is not None
            hh, mm = daily_hhmm
            payload["schedule_kind"] = "cron"
            payload["schedule_cron_local"] = f"{mm} {hh} * * *"
        else:  # when == "every"
            assert every_secs is not None
            payload["schedule_kind"] = "every"
            payload["schedule_interval_seconds"] = every_secs
            if window_start:
                payload["schedule_window_start_local"] = window_start
            if window_end:
                payload["schedule_window_end_local"] = window_end

        try:
            req = RoutineCreate(**payload)
        except Exception as e:
            return f"ERROR: invalid arguments: {e}"

        # Round 4 (item 5c): one reminder request = one reminder. A model
        # retry, a duplicated tool call, or a second turn saying the same
        # thing must not create a twin. Same user, same text, same shape,
        # firing within 90 s of an ENABLED existing one → return that one.
        try:
            _dup = await self._find_duplicate_reminder(user_id, payload)
        except Exception:  # noqa: BLE001 — the guard must never block a create
            logger.debug("[routines_skill.remind] duplicate probe failed", exc_info=True)
            _dup = None
        if _dup is not None:
            logger.info(
                "[routines_skill.remind] duplicate suppressed user=%s existing=%s",
                user_id[:8], _dup["id"],
            )
            # Compact like the created result — the duplicate dict leads with
            # id + reminder_text + schedule_at_utc for the same cut-survival
            # reason (its card parse is status-gated today, but the shape
            # must not drift from the created one).
            return json.dumps({
                "status": "already_scheduled",
                "reminder": _dup,
                "hint": (
                    "This exact reminder is already scheduled — nothing new was "
                    "created. Tell the user it is set (once)."
                ),
            }, default=str, ensure_ascii=False)

        try:
            resp = await create_routine(req)
        except HTTPException as e:
            return f"ERROR: {e.detail}"
        except Exception as e:
            # A schema/DB fault must not reach the model as a raw traceback
            # string (it gets paraphrased as "the tool is broken"). Return a
            # clean, model-guided message instead.
            logger.exception("[routines_skill.remind] create_routine failed")
            return (
                "ERROR: hit a temporary problem saving the reminder "
                f"({type(e).__name__}). Tell the user it didn't save and you'll "
                "try again shortly — this is a system issue, not their input."
            )

        delivery_out = (
            (resp.config or {}).get("delivery_channels") if resp.config else None
        ) or ["website"]
        # COMPACT and ORDERED, deliberately: the live tool_end frame cuts this
        # result at 200 chars, and the chat's reminder card draws its FIRST
        # paint from whatever survives the cut. The old shape pretty-printed
        # (indent=2), led with `name`, echoed the mode enum as `when`, and
        # carried no reminder_text at all — so the card's first frame could
        # only say "Wake up reminder / Set for once" and flipped to the real
        # text and countdown a fetch later. id + reminder_text and (for texts
        # up to ~58 chars) schedule_at_utc sit inside the first 200 chars;
        # created_at_utc survives only for very short texts — the card's
        # origin has two deeper fallbacks (the live surface seeds it at the
        # ask, the routines cache carries created_at), so that is fine.
        # Everything the model needs but the card doesn't comes after.
        return json.dumps({
            "status": "created",
            "reminder": {
                "id": resp.id,
                "reminder_text": text,
                "schedule_at_utc": (
                    str(getattr(resp, "schedule_at", None))
                    if getattr(resp, "schedule_at", None) else None
                ),
                "created_at_utc": str(set_at_naive),
                "next_run_at": str(resp.next_run_at) if resp.next_run_at else None,
                "when": when,
                "name": resp.name,
                "schedule_kind": getattr(resp, "schedule_kind", None),
                "schedule_cron_local": resp.schedule_cron_local,
                "schedule_interval_seconds": getattr(
                    resp, "schedule_interval_seconds", None
                ),
                "window_start_local": getattr(
                    resp, "schedule_window_start_local", None
                ),
                "window_end_local": getattr(
                    resp, "schedule_window_end_local", None
                ),
                "enabled": resp.enabled,
                "delivery_channels": delivery_out,
            },
            "hint": (
                "Reminder is live. The user can change or cancel it any time "
                "from the dashboard."
            ),
        }, default=str, ensure_ascii=False)

    # ── Duplicate guard (Round 4, item 5c) ─────────────────────────
    @staticmethod
    async def _find_duplicate_reminder(
        user_id: str, payload: Dict[str, Any], *, window_s: int = 90,
    ) -> Optional[Dict[str, Any]]:
        """An ENABLED reminder of this user with the same text and the same
        schedule shape (one-shot within ``window_s`` seconds; daily/every
        with the identical cron/interval) — or None."""
        if not user_id:
            return None
        from datetime import timedelta
        from sqlalchemy import select
        from app.db.database import async_session_maker
        from app.db.models import Routine

        text_norm = " ".join((payload.get("reminder_text") or "").split()).lower()
        if not text_norm:
            return None
        kind_shape = payload.get("schedule_kind") or "cron"
        async with async_session_maker() as db:
            rows = (await db.execute(
                select(Routine).where(
                    Routine.user_id == user_id,
                    Routine.kind == "reminder",
                    Routine.enabled.is_(True),
                )
            )).scalars().all()
        for r in rows:
            r_text = " ".join((getattr(r, "reminder_text", "") or "").split()).lower()
            if r_text != text_norm:
                continue
            r_shape = getattr(r, "schedule_kind", None) or "cron"
            if r_shape != kind_shape:
                continue
            if kind_shape == "at":
                a = getattr(r, "schedule_at", None)
                b = payload.get("schedule_at")
                if not a or not b:
                    continue
                if abs((a - b).total_seconds()) > window_s:
                    continue
            elif kind_shape == "every":
                if int(getattr(r, "schedule_interval_seconds", 0) or 0) != int(payload.get("schedule_interval_seconds") or 0):
                    continue
            else:  # cron
                if (getattr(r, "schedule_cron_local", "") or "") != (payload.get("schedule_cron_local") or ""):
                    continue
            return {
                # Same lead order as the create result: the card's parser
                # reads whatever survives a 200-char cut.
                "id": r.id,
                "reminder_text": getattr(r, "reminder_text", None),
                "schedule_at_utc": str(getattr(r, "schedule_at", None)) if getattr(r, "schedule_at", None) else None,
                "created_at_utc": str(getattr(r, "created_at", None)) if getattr(r, "created_at", None) else None,
                "next_run_at": str(getattr(r, "next_run_at", None)) if getattr(r, "next_run_at", None) else None,
                "name": r.name,
                "schedule_kind": r_shape,
                "schedule_cron_local": getattr(r, "schedule_cron_local", None),
                "schedule_interval_seconds": getattr(r, "schedule_interval_seconds", None),
                "enabled": True,
            }
        return None

    # ── Parsing helpers (skill-private) ────────────────────────────
    @staticmethod
    def _parse_hhmm(raw: str) -> Optional[tuple[int, int]]:
        """\"HH:MM\" or \"HH:MM:SS\" → (hour, minute). None on malformed."""
        if not raw:
            return None
        parts = raw.strip().split(":")
        if len(parts) not in (2, 3):
            return None
        try:
            hh = int(parts[0])
            mm = int(parts[1])
        except ValueError:
            return None
        if not (0 <= hh <= 23 and 0 <= mm <= 59):
            return None
        return hh, mm

    @staticmethod
    def _parse_local_datetime(raw: str, tz):
        """Parse \"YYYY-MM-DD HH:MM\" (explicit date) or \"HH:MM\" (today/
        tomorrow) into a tz-aware datetime in the user's timezone.

        Returns ``None`` on malformed input.
        """
        from datetime import datetime, time, timedelta
        raw = (raw or "").strip()
        if not raw:
            return None
        # Try ISO-shape first: YYYY-MM-DD HH:MM[:SS] (also accepts "T" sep).
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S",
                    "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M"):
            try:
                dt = datetime.strptime(raw, fmt)
                return dt.replace(tzinfo=tz)
            except ValueError:
                continue
        # Else try HH:MM[:SS] — pick today if still future, else tomorrow.
        # The model_validator on RoutineCreate rejects past schedule_at
        # so we need to land in the future here.
        hhmm = RoutinesSkill._parse_hhmm(raw)
        if hhmm is None:
            return None
        hh, mm = hhmm
        # Round 4 (item 5b): keep the seconds when the caller gave them.
        ss = 0
        _parts = raw.split(":")
        if len(_parts) == 3:
            try:
                ss = max(0, min(59, int(_parts[2])))
            except ValueError:
                ss = 0
        now_local = datetime.now(tz)
        target = now_local.replace(
            hour=hh, minute=mm, second=ss, microsecond=0
        )
        if target <= now_local:
            # A wall-clock time that passed within the last two minutes is
            # almost always the CURRENT minute computed by a model reading a
            # minute-resolution clock during a slow turn ("in 1 min" →
            # "19:19", executed at 19:19:05). Firing tomorrow for that is a
            # silent day-long miss; fire in a moment instead. Anything older
            # keeps the documented roll-to-tomorrow.
            if (now_local - target) <= timedelta(seconds=120):
                return now_local + timedelta(seconds=3)
            target = target + timedelta(days=1)
        return target

    async def _create(self, args: Dict[str, Any], ctx: SkillContext) -> str:
        from fastapi import HTTPException
        from app.api.routines import RoutineCreate, create_routine

        kind = (args.get("kind") or "").strip()
        schedule = (args.get("schedule_cron_local") or "").strip()
        if not kind:
            return "ERROR: `kind` is required (`email_briefing` or `agent_task`)."
        if not schedule:
            return "ERROR: `schedule_cron_local` is required (5-part cron)."

        # Same never-ask default as `_remind`: an omitted param means
        # chat + every connected channel; explicit lists pass through.
        delivery = args.get("delivery_channels")
        if not delivery:
            try:
                from app.config import settings
                from app.db.database import async_session_maker
                from app.agent.routines.channel_dispatcher import (
                    get_connected_channels,
                )
                delivery = await get_connected_channels(
                    str(getattr(settings, "user_id", "") or ""),
                    async_session_maker,
                )
            except Exception:
                logger.warning(
                    "[routines_skill.create] connected-channel probe failed "
                    "— falling back to website-only", exc_info=True,
                )
                delivery = None

        try:
            req = RoutineCreate(
                kind=kind,
                schedule_cron_local=schedule,
                name=args.get("name"),
                prompt_text=args.get("prompt_text"),
                enabled=bool(args.get("enabled", True)),
                delivery_channels=delivery,
                config=args.get("config"),
            )
        except Exception as e:
            return f"ERROR: invalid arguments: {e}"

        try:
            resp = await create_routine(req)
        except HTTPException as e:
            return f"ERROR: {e.detail}"

        delivery = (resp.config or {}).get("delivery_channels") if resp.config else None
        return _as_json({
            "status": "created",
            "routine": {
                "id": resp.id,
                "kind": resp.kind,
                "name": resp.name,
                "schedule_cron_local": resp.schedule_cron_local,
                "enabled": resp.enabled,
                "delivery_channels": delivery or ["website"],
                "next_run_at": str(resp.next_run_at) if resp.next_run_at else None,
            },
            "hint": (
                "It is scheduled and will fire at its next slot. The user "
                "can view or change it later from the dashboard."
            ),
        })

    async def _list(self, args: Dict[str, Any], ctx: SkillContext) -> str:
        from app.api.routines import list_routines

        rows = await list_routines()
        out = [
            {
                "id": r.id,
                "kind": r.kind,
                "name": r.name,
                "prompt_text": r.prompt_text,
                "schedule_cron_local": r.schedule_cron_local,
                "enabled": r.enabled,
                "delivery_channels": (
                    (r.config or {}).get("delivery_channels")
                    if r.config else None
                ) or ["website"],
                "last_status": r.last_status,
                "last_run_at": str(r.last_run_at) if r.last_run_at else None,
                "next_run_at": str(r.next_run_at) if r.next_run_at else None,
                "last_error": r.last_error,
            }
            for r in rows
        ]
        return _as_json({"routines": out, "count": len(out)})

    async def _update(self, args: Dict[str, Any], ctx: SkillContext) -> str:
        from fastapi import HTTPException
        from app.api.routines import RoutineUpdate, update_routine

        routine_id = (args.get("routine_id") or "").strip()
        if not routine_id:
            return "ERROR: `routine_id` is required."

        # Only pass fields the agent actually set — RoutineUpdate's None
        # default means "leave unchanged".
        update_fields: Dict[str, Any] = {}
        for key in ("schedule_cron_local", "enabled", "name", "prompt_text", "config", "delivery_channels"):
            if key in args and args[key] is not None:
                update_fields[key] = args[key]

        try:
            req = RoutineUpdate(**update_fields)
        except Exception as e:
            return f"ERROR: invalid arguments: {e}"

        try:
            resp = await update_routine(routine_id, req)
        except HTTPException as e:
            return f"ERROR: {e.detail}"

        return _as_json({
            "status": "updated",
            "routine": {
                "id": resp.id,
                # `kind` is echoed so the frontend tool-call row can
                # render the connector subject badge (Gmail icon for an
                # email_briefing routine etc.). Drop this field and an
                # "update my morning briefing schedule" turn renders
                # the update row with the bare action glyph, losing the
                # "this is about Gmail" signal at the row level.
                "kind": resp.kind,
                "schedule_cron_local": resp.schedule_cron_local,
                "enabled": resp.enabled,
                "name": resp.name,
                "next_run_at": str(resp.next_run_at) if resp.next_run_at else None,
            },
        })

    async def _delete(self, args: Dict[str, Any], ctx: SkillContext) -> str:
        from fastapi import HTTPException
        from app.api.routines import delete_routine

        routine_id = (args.get("routine_id") or "").strip()
        if not routine_id:
            return "ERROR: `routine_id` is required."

        try:
            await delete_routine(routine_id)
        except HTTPException as e:
            return f"ERROR: {e.detail}"

        return _as_json({"status": "deleted", "routine_id": routine_id})

    async def _run_now(self, args: Dict[str, Any], ctx: SkillContext) -> str:
        from fastapi import HTTPException
        from app.api.routines import force_run

        routine_id = (args.get("routine_id") or "").strip()
        if not routine_id:
            return "ERROR: `routine_id` is required."

        try:
            resp = await force_run(routine_id)
        except HTTPException as e:
            # 409 carries today's existing run details — surface them as
            # structured data so the agent can tell the user "your morning
            # briefing already ran at 06:30".
            detail = e.detail
            if isinstance(detail, dict):
                return _as_json({"status": "already_ran_today", **detail})
            return f"ERROR: {detail}"

        return _as_json({
            "status": "fired",
            "run": {
                "id": resp.id,
                "scheduled_for_local_date": str(resp.scheduled_for_local_date),
                "status": resp.status,
                "emails_fetched": resp.emails_fetched,
                "summary_message_id": resp.summary_message_id,
                "error_class": resp.error_class,
                "error_detail": resp.error_detail,
            },
            "hint": (
                "The result was posted into the day-chat as the assistant. "
                "Tell the user to scroll up to today's date to see it."
            ),
        })
