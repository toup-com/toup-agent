"""Routines Skill — agent self-authors scheduled automations.

Lets the user create/manage routines from chat: "from now on read all my
emails before I wake up and summarize them" → agent calls
`routines__create` with kind=email_briefing + a sensible cron.

Five tools, all single-user-per-container (no user_id arg — read from
`settings.user_id`):

  routines__create   — create one routine
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
                    "  • `email_briefing` — preset for \"summarize my unread "
                    "emails\". Uses Gmail MCP + Haiku. NO prompt_text needed.\n"
                    "  • `agent_task` — everything else. REQUIRES `prompt_text` "
                    "describing what the agent should do at fire time.\n\n"
                    "Cron format: 5-part `m h dom mon dow` in the user's local "
                    "timezone. Examples: `30 6 * * *` = 06:30 every day; "
                    "`0 7 * * 1-5` = 07:00 weekdays; `0 9 * * 0` = 09:00 Sundays. "
                    "Confirm the time with the user BEFORE calling this tool.\n\n"
                    "**Delivery channels:** ALWAYS ask the user where they want "
                    "to see the result before creating the routine. Options:\n"
                    "  • `website` — the chat thread on toup.ai (default)\n"
                    "  • `telegram` — the user's Telegram bot DM\n"
                    "  • `whatsapp` — the user's WhatsApp self-chat\n"
                    "Pass the chosen subset as `delivery_channels`. The website "
                    "is always included (it's the canonical record); listing "
                    "telegram/whatsapp fans the summary OUT to those channels too."
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
                                "Where to deliver the routine's output. Ask the "
                                "user before picking. `website` is always "
                                "included server-side. Examples: "
                                "[\"website\",\"telegram\"], [\"telegram\",\"whatsapp\"] "
                                "(both channels), [\"website\"] (chat only — the default)."
                            ),
                        },
                    },
                    "required": ["kind", "schedule_cron_local"],
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
            "# Routines — recurring agent automations\n"
            "You can create scheduled automations for the user via the "
            "`routines__*` tools. Trigger this flow whenever the user asks "
            "for something on a recurring schedule:\n"
            "  • \"From now on, read my emails before I wake up and summarize them\"\n"
            "  • \"Every weekday at 7am, give me my calendar for the day\"\n"
            "  • \"Check my GitHub notifications every hour during work hours\"\n"
            "\n"
            "**Flow:**\n"
            "  1. Confirm the schedule with the user in plain English (\"so "
            "you want this at 6:30am every day, right?\"). The user thinks in "
            "their local time — never echo UTC.\n"
            "  2. **ASK WHERE to deliver it.** Before creating the routine, "
            "ask the user which channel(s) they want to receive it on. "
            "Phrase the question naturally: \"Where do you want to see this "
            "— here in the chat, on Telegram, on WhatsApp, or all of them?\" "
            "Default is `website` (the chat thread). Map the answer to a "
            "`delivery_channels` list:\n"
            "     - \"here\" / \"the chat\" / \"website\" → `[\"website\"]`\n"
            "     - \"Telegram\" → `[\"website\", \"telegram\"]`\n"
            "     - \"WhatsApp\" → `[\"website\", \"whatsapp\"]`\n"
            "     - \"all channels\" / \"everywhere\" → `[\"website\", \"telegram\", \"whatsapp\"]`\n"
            "     (The website is always kept so the user has a permanent "
            "record on their dashboard even when the buzz goes to phone.)\n"
            "  3. Pick the kind:\n"
            "     - `email_briefing` for \"summarize my unread emails\" "
            "(preset Gmail flow — no prompt needed).\n"
            "     - `agent_task` for everything else. Write a self-contained "
            "`prompt_text` — the fire runs in a fresh context with no memory "
            "of this conversation, so spell out exactly what to do.\n"
            "  4. Convert the spoken schedule to a 5-part cron in the user's "
            "tz: `30 6 * * *` = 06:30 daily; `0 7 * * 1-5` = 07:00 weekdays.\n"
            "  5. Call `routines__create` with kind + schedule + name + "
            "delivery_channels (+ prompt_text for agent_task). Confirm "
            "success to the user, mention which channel(s) they'll see it on, "
            "and tell them they can adjust later from Mission Control on the dashboard.\n"
            "\n"
            "**Before creating, always call `routines__list` first** to check "
            "if a similar routine already exists — if so, offer to update it "
            "instead of duplicating. (Note: `email_briefing` is one-per-user; "
            "the API will 409 a duplicate. `agent_task` allows many.)\n"
            "\n"
            "Routines are flag-gated. If `routines__create` returns "
            "`ERROR: Feature not available`, tell the user the feature isn't "
            "enabled for their tenant — don't retry."
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
    async def _create(self, args: Dict[str, Any], ctx: SkillContext) -> str:
        from fastapi import HTTPException
        from app.api.routines import RoutineCreate, create_routine

        kind = (args.get("kind") or "").strip()
        schedule = (args.get("schedule_cron_local") or "").strip()
        if not kind:
            return "ERROR: `kind` is required (`email_briefing` or `agent_task`)."
        if not schedule:
            return "ERROR: `schedule_cron_local` is required (5-part cron)."

        try:
            req = RoutineCreate(
                kind=kind,
                schedule_cron_local=schedule,
                name=args.get("name"),
                prompt_text=args.get("prompt_text"),
                enabled=bool(args.get("enabled", True)),
                delivery_channels=args.get("delivery_channels"),
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
