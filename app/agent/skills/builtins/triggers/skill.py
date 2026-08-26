"""Triggers Skill — agent self-authors event-driven automations.

Lets the user create/manage triggers from chat: "from now on summarize
every new email" → agent calls `triggers__create` with
kind=email_received + action=summarize_and_post.

Five tools mirror the routines skill so the agent's mental model is
consistent (the user doesn't care about the routines-vs-triggers
distinction; the agent picks based on whether the user said "every
morning" vs "when X happens"):

  triggers__create   — create one trigger
  triggers__list     — list this user's triggers + recent events
  triggers__update   — toggle / change filters / change action
  triggers__delete   — drop a trigger
  triggers__test     — fire a synthetic event to verify wiring

The skill is a thin wrapper around `app.api.triggers` (the HTTP
surface). HTTPException is caught and surfaced as `ERROR: …`.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from app.agent.skills.base import Skill, SkillContext, SkillMeta

logger = logging.getLogger(__name__)


def _as_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, default=str, ensure_ascii=False)


def _provision_status_hint(t) -> tuple[str, str]:
    """Translate the trigger row into a `(status, hint)` pair the agent
    surfaces to the user. The agent's prompt-level honesty contract
    (see SkillMeta.description) forbids paraphrasing the failure cases
    as "may take a moment." This function is the source of truth for
    that translation.

    Only meaningful for kind=email_received in v1 — other kinds return
    a generic "live" status since they don't have a provisioning step.
    """
    # Non-email kinds don't have a watch arming step.
    if getattr(t, "kind", "") != "email_received":
        return "live", "The trigger is registered."

    if getattr(t, "watch_provisioned", False):
        return (
            "live",
            "Trigger is armed and live. The agent will react on the next "
            "matching event. Sending yourself a test email is a good way "
            "to confirm the wiring end-to-end.",
        )

    last_status = (getattr(t, "last_status", "") or "").strip()
    ps = getattr(t, "provider_state_json", None) or {}
    err = (ps.get("provision_error") or "").strip() if isinstance(ps, dict) else ""

    if last_status == "skipped_reauth" or err == "needs_reauth":
        return (
            "needs_user_action",
            "Trigger is saved but NOT live. The user's Gmail connection "
            "is missing or expired. Tell the user to reconnect Gmail "
            "(Settings → Integrations → Gmail → Connect). Do NOT say "
            "the trigger is working — it is not.",
        )

    if last_status == "provisioning_failed" or err in (
        "ops_blocked", "scope_error", "transient", "feature_disabled",
        "platform_unreachable", "agent_misconfigured", "unknown",
    ):
        detail = ps.get("provision_detail", "") if isinstance(ps, dict) else ""
        return (
            "ops_blocked",
            "Trigger is saved but the platform couldn't arm the Gmail "
            "watch yet (reason: "
            f"{err or 'unknown'}{' — ' + detail if detail else ''}). "
            "Tell the user the trigger won't fire yet, the issue is on "
            "Toup's side and we're notified. Do NOT promise it works."
        )

    # Newly-created or pending — auto-arm hasn't reported back yet.
    return (
        "pending",
        "Trigger is saved; provisioning is in progress. Tell the user "
        "it's setting up and they should test with a new email in a "
        "minute. If it still doesn't fire after 2 minutes, ask them to "
        "check Settings → Integrations."
    )


def _trigger_summary(t) -> Dict[str, Any]:
    """Compact dict for agent reasoning. Drops recent_events to keep
    the payload small — the agent calls `triggers__list` with an
    explicit `include_events` if it wants them."""
    return {
        "id": t.id,
        "kind": t.kind,
        "action": t.action,
        "name": t.name,
        "enabled": bool(t.enabled),
        "filter_json": t.filter_json or {},
        "delivery_channels": t.delivery_channels,
        "watch_provisioned": t.watch_provisioned,
        "last_status": t.last_status,
        "last_fired_at": t.last_fired_at,
        "fire_count": t.fire_count,
        "last_error": t.last_error,
    }


# W2.1a prefix diet (settings.prompt_diet): compact description for the fat
# triggers__create schema (657 tok measured on the wire —
# docs/audits/2026-07-sota-assessment.md). Only description strings shrink;
# properties/enums/required are byte-identical to the full schema (shape
# equality pinned in tests/test_prompt_diet.py).
_DIET_TOOL_DESCRIPTIONS = {
    "triggers__create": (
        "Create an event-driven trigger — the agent reacts WHEN something "
        "happens (\"when I get an email\"), NOT on a schedule (that's "
        "`routines__create`). v1 kind: `email_received` (user must have "
        "Gmail connected). Actions: `summarize_and_post` = LLM summary to "
        "Day-as-Chat (best default); `notify_only` = raw sender/subject/"
        "snippet, no LLM; `forward_to_telegram` = summary + Telegram "
        "fan-out. Optional `filter_json` rules, AND'd: `from_contains`, "
        "`subject_contains`, `labels` (Gmail label ids), "
        "`exclude_categories` (of promotions/social/updates/forums) — each "
        "a list. Ask the user where to deliver (`delivery_channels`)."
    ),
}

_DIET_PROPERTY_DESCRIPTIONS = {
    "triggers__create": {
        "kind": "v1: `email_received`.",
        "action": "What each fire does — see tool description.",
        "name": "Short name (≤100 chars) shown to the user in their list.",
        "filter_json": "Optional filter rules (see tool description). Omit for match-all.",
        "delivery_channels": "Where each fire's output goes; `website` always included server-side.",
    },
}


class TriggersSkill(Skill):
    """Expose trigger CRUD + test-fire to the agent.

    Production guarantees:
      • Validates kind + action via the API path (same source of
        truth as Mission Control), so agent-created and UI-created
        triggers are indistinguishable on disk.
      • forward_to_telegram auto-implies telegram in delivery
        channels — handled inside the API, the agent doesn't need
        to remember.
      • Feature-flag-off returns `ERROR: Feature not available` so
        the agent can tell the user clearly instead of silently
        no-op'ing.
    """

    meta = SkillMeta(
        name="triggers",
        version="1.0.0",
        description=(
            "Self-author event-driven agent automations: react when a new "
            "email arrives, calendar event is created, etc.\n\n"
            "**HONESTY CONTRACT (non-negotiable):** every successful "
            "`triggers__create`/`triggers__list` response includes a "
            "top-level `status` field. After calling these tools, READ "
            "`status` and adjust what you tell the user:\n"
            "  • `live` — the trigger is armed and WILL fire. Free to "
            "say \"from now on, when X happens, I'll do Y.\"\n"
            "  • `needs_user_action` — the user must reconnect Gmail "
            "before any event fires. Tell them explicitly, do NOT "
            "promise the trigger is working.\n"
            "  • `ops_blocked` — platform-side misconfig (Pub/Sub topic "
            "/ GCP IAM). Tell the user the trigger is saved but won't "
            "fire yet, and that we're notified. Do NOT promise it works.\n"
            "Never paraphrase `needs_user_action` or `ops_blocked` as "
            "\"may take a moment.\" The user needs to know it's blocked."
        ),
        author="Toup",
    )

    # ── Tool schemas ─────────────────────────────────────────────

    def get_tools(self) -> List[Dict[str, Any]]:
        tools = [
            {
                "name": "triggers__create",
                "description": (
                    "Create a new event-driven trigger for the user. Use this "
                    "when the user asks the agent to react to something "
                    "happening (\"when I get an email\", \"when a calendar "
                    "event is created\", \"every time someone messages me on "
                    "Slack\") — i.e., NOT on a schedule. For scheduled actions "
                    "use the `routines__create` tool instead.\n\n"
                    "Picking `kind` (v1 supports one):\n"
                    "  • `email_received` — fires on every new Gmail message. "
                    "Requires the user to have connected Gmail.\n\n"
                    "Picking `action`:\n"
                    "  • `summarize_and_post` — LLM summary, posted to "
                    "Day-as-Chat (and optional extra channels). Best default "
                    "for \"keep me informed\" flows.\n"
                    "  • `notify_only` — raw sender + subject + snippet, no "
                    "LLM call. Best when the user wants the email itself, "
                    "not a summary.\n"
                    "  • `forward_to_telegram` — summary + force-fan-out to "
                    "Telegram. Auto-adds telegram to delivery_channels.\n\n"
                    "Filter rules (all optional, AND'd together):\n"
                    "  • `from_contains`: list of substrings to match against "
                    "the From header. Example: [\"@github.com\", \"@stripe.com\"]\n"
                    "  • `subject_contains`: substrings to match against "
                    "Subject. Example: [\"invoice\", \"receipt\"]\n"
                    "  • `labels`: Gmail label ids — message must have at "
                    "least one. Example: [\"IMPORTANT\", \"Label_42\"]\n"
                    "  • `exclude_categories`: drop messages in any of "
                    "[\"promotions\", \"social\", \"updates\", \"forums\"].\n\n"
                    "**Delivery channels:** Always ask the user where they "
                    "want to see the result. Same options as routines: "
                    "website (default), telegram, whatsapp. Pass the chosen "
                    "subset as `delivery_channels`."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "kind": {
                            "type": "string",
                            "enum": ["email_received"],
                            "description": (
                                "v1 supports `email_received`. More kinds "
                                "ship in follow-ups."
                            ),
                        },
                        "action": {
                            "type": "string",
                            "enum": [
                                "summarize_and_post",
                                "notify_only",
                                "forward_to_telegram",
                            ],
                            "description": (
                                "What the agent does each time the event "
                                "fires. See description for guidance."
                            ),
                        },
                        "name": {
                            "type": "string",
                            "description": (
                                "Short human-readable name (≤100 chars) shown "
                                "to the user. e.g. \"GitHub PR pings\", "
                                "\"Boss emails\"."
                            ),
                        },
                        "filter_json": {
                            "type": "object",
                            "description": (
                                "Optional filter rules — see action "
                                "description. Omit for match-all."
                            ),
                        },
                        "delivery_channels": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["website", "telegram", "whatsapp"],
                            },
                            "description": (
                                "Where to deliver each fire's output. "
                                "`website` is always included server-side."
                            ),
                        },
                        "enabled": {
                            "type": "boolean",
                            "description": "Default true. Set false to create dormant.",
                        },
                    },
                    "required": ["kind", "action"],
                },
            },
            {
                "name": "triggers__list",
                "description": (
                    "List the existing triggers + their recent fires. Call "
                    "this BEFORE creating a new trigger to check for "
                    "duplicates. Triggers are internal wiring, not something "
                    "the user has a word for: they are NOT the user's "
                    "automations and never belong in an answer about what "
                    "the user has set up. Returns id, kind, action, filters, "
                    "delivery channels, last fire, watch provisioning state."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "name": "triggers__update",
                "description": (
                    "Update a trigger's filters, action, delivery channels, "
                    "or enabled flag. Use the `id` from `triggers__list`. "
                    "Any field left unset is preserved."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "trigger_id": {
                            "type": "string",
                            "description": "id from `triggers__list`",
                        },
                        "enabled": {"type": "boolean"},
                        "name": {"type": "string"},
                        "action": {
                            "type": "string",
                            "enum": [
                                "summarize_and_post",
                                "notify_only",
                                "forward_to_telegram",
                            ],
                        },
                        "filter_json": {"type": "object"},
                        "delivery_channels": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["website", "telegram", "whatsapp"],
                            },
                        },
                    },
                    "required": ["trigger_id"],
                },
            },
            {
                "name": "triggers__delete",
                "description": (
                    "Delete a trigger and all its event history. Use the `id` "
                    "from `triggers__list`. Irreversible. Confirm with the "
                    "user first."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "trigger_id": {"type": "string"},
                    },
                    "required": ["trigger_id"],
                },
            },
            {
                "name": "triggers__test",
                "description": (
                    "Fire a synthetic test event against an existing trigger. "
                    "Useful right after `triggers__create` to verify the "
                    "wiring (the dashboard surface, delivery channels, etc.) "
                    "without waiting for a real email to arrive. The handler "
                    "will fail at the message-fetch step (the synthetic id "
                    "doesn't match a real Gmail message), which is expected — "
                    "it proves the dispatch path works."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "trigger_id": {"type": "string"},
                    },
                    "required": ["trigger_id"],
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

    # ── System-prompt augmentation ───────────────────────────────

    def system_prompt(self) -> str:
        return (
            "## Triggers — react to real-world events\n"
            "\n"
            "When the user wants the agent to react to an EVENT (not a "
            "schedule), use the triggers tools:\n"
            "\n"
            "  1. Listen for phrases like \"when I get an email\", \"every "
            "time someone DMs me\", \"as soon as my calendar gets a new "
            "invite\".\n"
            "  2. Pick `kind` — v1 only supports `email_received`. If the "
            "user asked for something else, explain that you can do email "
            "today and the other kind is on the roadmap.\n"
            "  3. Pick `action`:\n"
            "     - `summarize_and_post` for \"tell me about it\".\n"
            "     - `notify_only` for \"just show me the email\".\n"
            "     - `forward_to_telegram` for \"send it to my Telegram\".\n"
            "  4. Pick filters — ASK if anything specific. The user often "
            "says \"only from my boss\" or \"skip newsletters\".\n"
            "  5. ALWAYS ask where to deliver (`delivery_channels`) before "
            "creating: website (default), telegram, whatsapp.\n"
            "  6. Call `triggers__create`. Confirm success + tell the user "
            "they can manage it later from the dashboard.\n"
            "\n"
            "**Before creating, ALWAYS call `triggers__list` first** to "
            "check for duplicates. Two triggers for the same kind+action+ "
            "filters would produce duplicate Day-as-Chat posts.\n"
            "\n"
            "If `triggers__create` returns `ERROR: Feature not available`, "
            "tell the user that event-driven triggers aren't enabled for "
            "their tenant — don't retry."
        )

    # ── Dispatch ─────────────────────────────────────────────────

    async def execute_tool(
        self, tool_name: str, args: Dict[str, Any], ctx: SkillContext,
    ) -> str:
        dispatch = {
            "triggers__create": self._create,
            "triggers__list":   self._list,
            "triggers__update": self._update,
            "triggers__delete": self._delete,
            "triggers__test":   self._test,
        }
        handler = dispatch.get(tool_name)
        if not handler:
            return f"ERROR: Unknown triggers tool: {tool_name}"
        try:
            return await handler(args, ctx)
        except Exception as e:
            logger.exception("[triggers_skill] unexpected error in %s", tool_name)
            return f"ERROR: {type(e).__name__}: {str(e)[:300]}"

    # ── Handlers — delegate to app.api.triggers ──────────────────

    async def _create(self, args: Dict[str, Any], ctx: SkillContext) -> str:
        from fastapi import HTTPException
        from app.api.triggers import TriggerCreate, create_trigger

        kind = (args.get("kind") or "").strip()
        action = (args.get("action") or "").strip()
        if not kind:
            return "ERROR: `kind` is required (v1: `email_received`)."
        if not action:
            return (
                "ERROR: `action` is required "
                "(`summarize_and_post` | `notify_only` | `forward_to_telegram`)."
            )

        try:
            req = TriggerCreate(
                kind=kind,
                action=action,
                name=args.get("name"),
                enabled=bool(args.get("enabled", True)),
                filter_json=args.get("filter_json"),
                config_json=args.get("config_json"),
                delivery_channels=args.get("delivery_channels"),
            )
        except Exception as e:
            return f"ERROR: invalid arguments: {e}"

        try:
            resp = await create_trigger(req)
        except HTTPException as e:
            return f"ERROR: {e.detail}"

        out = _trigger_summary(resp)
        status, hint = _provision_status_hint(resp)
        return _as_json({
            "status": status,                     # live | needs_user_action | ops_blocked
            "trigger": out,
            "hint": hint,
        })

    async def _list(self, args: Dict[str, Any], ctx: SkillContext) -> str:
        from app.api.triggers import list_triggers

        rows = await list_triggers()
        triggers_out = []
        for r in rows:
            t = _trigger_summary(r)
            s, _hint = _provision_status_hint(r)
            t["status"] = s
            triggers_out.append(t)
        return _as_json({
            "count": len(rows),
            "triggers": triggers_out,
        })

    async def _update(self, args: Dict[str, Any], ctx: SkillContext) -> str:
        from fastapi import HTTPException
        from app.api.triggers import TriggerUpdate, update_trigger

        trigger_id = (args.get("trigger_id") or "").strip()
        if not trigger_id:
            return "ERROR: `trigger_id` is required."
        try:
            req = TriggerUpdate(
                enabled=args.get("enabled"),
                name=args.get("name"),
                action=args.get("action"),
                filter_json=args.get("filter_json"),
                config_json=args.get("config_json"),
                delivery_channels=args.get("delivery_channels"),
            )
        except Exception as e:
            return f"ERROR: invalid arguments: {e}"
        try:
            resp = await update_trigger(trigger_id, req)
        except HTTPException as e:
            return f"ERROR: {e.detail}"
        return _as_json({"status": "updated", "trigger": _trigger_summary(resp)})

    async def _delete(self, args: Dict[str, Any], ctx: SkillContext) -> str:
        from fastapi import HTTPException
        from app.api.triggers import delete_trigger

        trigger_id = (args.get("trigger_id") or "").strip()
        if not trigger_id:
            return "ERROR: `trigger_id` is required."
        try:
            await delete_trigger(trigger_id)
        except HTTPException as e:
            return f"ERROR: {e.detail}"
        return _as_json({"status": "deleted", "trigger_id": trigger_id})

    async def _test(self, args: Dict[str, Any], ctx: SkillContext) -> str:
        from fastapi import HTTPException
        from app.api.triggers import test_trigger

        trigger_id = (args.get("trigger_id") or "").strip()
        if not trigger_id:
            return "ERROR: `trigger_id` is required."
        try:
            resp = await test_trigger(trigger_id)
        except HTTPException as e:
            return f"ERROR: {e.detail}"
        return _as_json({
            "status": "test_event_queued",
            "event_id": resp.id,
            "event_dedupe_id": resp.event_dedupe_id,
            "hint": (
                "The runner will dispatch this event to the handler. "
                "The handler will FAIL on the message-fetch step "
                "(synthetic id doesn't match a real message) — this is "
                "EXPECTED. The test proves the dispatch path works. "
                "Watch the Day-as-Chat surface for a Message; check "
                "the dashboard for the event status."
            ),
        })
