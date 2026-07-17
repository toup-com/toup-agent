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
            "Self-author scheduled agent automations: morning email briefings, "
            "daily check-ins, recurring agent tasks."
        ),
        author="Toup",
    )

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------
    def get_tools(self) -> List[Dict[str, Any]]:
        return [
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
                                "Shown to the user in Mission Control."
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
                    "  • `once` — fire one time, then auto-disable. Pass "
                    "`at_local` as either `\"YYYY-MM-DD HH:MM\"` (full local "
                    "datetime) or `\"HH:MM\"` (today if still future, else "
                    "tomorrow). Example: \"remind me to call mom at 6pm\".\n"
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
                    "restricts delivery (\"only telegram\" → [\"telegram\"])."
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
                        "at_local": {
                            "type": "string",
                            "description": (
                                "Required when when=`once`. Either "
                                "\"YYYY-MM-DD HH:MM\" (explicit date+time) or "
                                "\"HH:MM\" (today if still in the future, else "
                                "tomorrow). Interpreted in the user's local tz."
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
                                "Optional short name shown in Mission Control. "
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
                    "List all of the user's existing routines. Call this BEFORE "
                    "creating a new routine to check for duplicates, or when "
                    "the user asks \"what automations do I have set up?\". "
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

    # ------------------------------------------------------------------
    # System prompt — guides when + how to call these tools
    # ------------------------------------------------------------------
    def get_system_prompt_section(self) -> Optional[str]:
        return (
            "# Routines — recurring agent automations + reminders\n"
            "You can create scheduled automations for the user via the "
            "`routines__*` tools. Trigger this flow whenever the user asks "
            "for something on a recurring schedule:\n"
            "  • \"From now on, read my emails before I wake up and summarize them\"\n"
            "  • \"Every weekday at 7am, give me my calendar for the day\"\n"
            "  • \"Check my GitHub notifications every hour during work hours\"\n"
            "  • \"Remind me to call mom at 6pm\" — *use `routines__remind`*\n"
            "  • \"Buzz me every 30 minutes between 9 and 5 to stretch\" — *use `routines__remind`*\n"
            "\n"
            "**Tool to pick:**\n"
            "  • If the user wants literal text delivered at a time "
            "(reminder / alert / nudge — \"remind me to X\", \"ping me at Y\"), "
            "use **`routines__remind`** with the friendly `when` modes "
            "(once / daily / every). No LLM call, no MCP — just text on a "
            "schedule.\n"
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
            "result's `delivery_channels`), and tell them they can adjust "
            "later from Mission Control on the dashboard.\n"
            "\n"
            "**Before creating, always call `routines__list` first** to check "
            "if a similar routine already exists — if so, offer to update it "
            "instead of duplicating. (Note: `email_briefing` is one-per-user; "
            "the API will 409 a duplicate. `agent_task` allows many.)\n"
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
        daily_hhmm: Optional[tuple[int, int]] = None
        every_secs: Optional[int] = None
        window_start: Optional[str] = None
        window_end: Optional[str] = None

        if when == "once":
            raw = (args.get("at_local") or "").strip()
            if not raw:
                return (
                    "ERROR: `at_local` is required when when=`once`. "
                    "Pass \"YYYY-MM-DD HH:MM\" or \"HH:MM\"."
                )
            once_raw_at = raw
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
        payload: Dict[str, Any] = {
            "kind": "reminder",
            "name": name,
            "reminder_text": text,
            "enabled": enabled,
            "delivery_channels": delivery,
        }

        if when == "once":
            dt_local = self._parse_local_datetime(once_raw_at or "", tz)
            if dt_local is None:
                return (
                    f"ERROR: couldn't parse at_local={once_raw_at!r}. Use "
                    "\"YYYY-MM-DD HH:MM\" or \"HH:MM\"."
                )
            # Translate local wall-clock to a UTC-naive datetime, which
            # is the shape `schedule_at` expects.
            from datetime import timezone as _tzmod
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
        return _as_json({
            "status": "created",
            "reminder": {
                "id": resp.id,
                "name": resp.name,
                "when": when,
                "schedule_kind": getattr(resp, "schedule_kind", None),
                "schedule_cron_local": resp.schedule_cron_local,
                "schedule_at_utc": (
                    str(getattr(resp, "schedule_at", None))
                    if getattr(resp, "schedule_at", None) else None
                ),
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
                "next_run_at": str(resp.next_run_at) if resp.next_run_at else None,
            },
            "hint": (
                "Reminder is live. The user can change/cancel it any time "
                "from Mission Control on the dashboard."
            ),
        })

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
        # Try ISO-shape first: YYYY-MM-DD HH:MM (also accepts "T" sep).
        for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M"):
            try:
                dt = datetime.strptime(raw, fmt)
                return dt.replace(tzinfo=tz)
            except ValueError:
                continue
        # Else try HH:MM — pick today if still future, else tomorrow.
        # The model_validator on RoutineCreate rejects past schedule_at
        # so we need to land in the future here.
        hhmm = RoutinesSkill._parse_hhmm(raw)
        if hhmm is None:
            return None
        hh, mm = hhmm
        now_local = datetime.now(tz)
        target = now_local.replace(
            hour=hh, minute=mm, second=0, microsecond=0
        )
        if target <= now_local:
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
                "The routine is registered with the scheduler and will fire at "
                "its next cron tick. The user can view/edit it in Mission "
                "Control on the dashboard."
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
