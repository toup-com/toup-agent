"""Automations setup skill (Round 26) — the chat-built engine's tools.

Eleven tools mirror the round brief exactly:

  automations__get_registry       — what can fire / what can be written
  automations__request_connection — connector card (10-min TTL)
  automations__request_permission — grant card (1-h TTL)
  automations__list_targets       — pinnable targets for a connector
  automations__create             — validate + save a DRAFT
  automations__update             — replace spec, re-compile
  automations__test_run           — one synthetic fire (real rails)
  automations__arm / pause / resume / delete

Card-emitting tools return "card shown — STOP" prose immediately; the
conversation resumes when the state transition arrives (OAuth callback
or grant decision → agent hook → card update in place). There is no
turn-suspension primitive in this codebase — parking is the pattern
(MAPPING.md §3.2).

The skill registers ONLY when `settings.automations_enabled` is true
(gated in tool_entitlements.skill_enabled) so the dark tools array is
byte-identical to today's.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from app.agent.skills.base import Skill, SkillContext, SkillMeta
from app.config import settings

logger = logging.getLogger(__name__)


def _as_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, default=str, ensure_ascii=False)


def _uid(ctx: SkillContext) -> str:
    return (ctx.user_id or getattr(settings, "user_id", "") or "").strip()


# Per-connector "list the pinnable targets" read tools. The kind names
# what the target IS, so the grant card can say "channel #eng" and not
# just an opaque id.
_TARGET_SOURCES: dict[str, dict] = {
    "slack": {"tool": "slack__list_channels", "items": "channels",
              "id": "id", "label": "name", "kind": "channel"},
    "jira": {"tool": "jira__list_projects", "items": "projects",
             "id": "key", "label": "name", "kind": "project"},
    "github": {"tool": "github__list_repos", "items": "repos",
               "id": "full_name", "label": "full_name", "kind": "repo"},
}


class AutomationsSkill(Skill):
    meta = SkillMeta(
        name="automations",
        version="1.0.0",
        description=(
            "Build automations from conversation: watch an external "
            "service (poll/push) or a schedule, then act through a "
            "grant-gated connector write."
        ),
        author="toup",
    )

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------
    def get_tools(self) -> List[Dict[str, Any]]:
        spec_schema = {
            "type": "object",
            "description": (
                "AutomationSpec. v1: trigger.mode 'poll' (connector_id "
                "+ event + poll_interval_s>=300), 'push' (gmail only), "
                "or 'schedule' ({cron_local | at | every_s}); action: "
                "one connector tool + params_template using "
                "{{event.<field>}} / {{grant.target.id}}. Write actions "
                "REQUIRE grant_id from automations__request_permission. "
                "dedupe_key ('event.<field>') REQUIRED for push/poll. "
                "v2 (set \"version\": 2): trigger.sources[] (up to 4 "
                "lanes, each push/poll lane with its OWN dedupe_key) + "
                "steps[] (up to 8 tool calls: reads first — each may "
                "'collect' items into {{steps.<id>.text}}/"
                "{{steps.<id>.count}} — then 1-3 grant-gated writes, "
                "each with its own grant_id). String params may also "
                "use {{var.<name>}} (declared in top-level variables), "
                "{{source.id}} and {{memory.<key>}} "
                "(last_run_at/last_outcome/last_counts from earlier "
                "runs). Start from a template via "
                "automations__list_templates when one fits."
            ),
        }
        return [
            {
                "name": "automations__get_registry",
                "description": (
                    "What can be automated: per-connector events "
                    "(push/poll + floor), write actions with their "
                    "pinned-target parameter, rate budgets, and this "
                    "user's current connection state. ALWAYS call this "
                    "before proposing an automation."
                ),
                "input_schema": {"type": "object", "properties": {}},
            },
            {
                "name": "automations__request_connection",
                "description": (
                    "Show the user a connector card asking to connect "
                    "(or re-connect with write scopes). Use mode="
                    "'read_write' only when the automation will write. "
                    "The card expires in 10 minutes. After calling: "
                    "STOP and wait — do not call again, do not assume "
                    "the outcome."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "connector_id": {"type": "string"},
                        "mode": {"type": "string",
                                 "enum": ["read", "read_write"],
                                 "default": "read"},
                    },
                    "required": ["connector_id"],
                },
            },
            {
                "name": "automations__request_permission",
                "description": (
                    "Show the user a permission card for ONE write "
                    "action pinned to ONE target (e.g. post to #eng), "
                    "with an optional cadence budget. mode='auto' fires "
                    "without per-run confirmation once approved; "
                    "'confirm' previews every run. The card expires in "
                    "1 hour. After calling: STOP and wait for the "
                    "decision."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "connector_id": {"type": "string"},
                        "tool": {"type": "string",
                                 "description": "The write tool, e.g. "
                                                "slack__send_message."},
                        "target": {
                            "type": "object",
                            "description": "{kind, id, label} from "
                                           "automations__list_targets.",
                        },
                        "cadence": {
                            "type": "object",
                            "description": "{per_day?, per_hour?} caps.",
                        },
                        "mode": {"type": "string",
                                 "enum": ["auto", "confirm"],
                                 "default": "confirm"},
                        "summary": {"type": "string",
                                    "description": "One human sentence: "
                                                   "what this permits."},
                        "preview": {
                            "type": "object",
                            "description": "Example tool arguments shown "
                                           "on the card.",
                        },
                        "automation_id": {"type": "string"},
                    },
                    "required": ["connector_id", "tool", "target", "summary"],
                },
            },
            {
                "name": "automations__list_targets",
                "description": (
                    "List pinnable write targets for a connector "
                    "(slack channels, jira projects, github repos). "
                    "Requires the connector to be connected."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {"connector_id": {"type": "string"}},
                    "required": ["connector_id"],
                },
            },
            {
                "name": "automations__list",
                "description": "This user's automations with status and "
                               "health.",
                "input_schema": {"type": "object", "properties": {}},
            },
            {
                "name": "automations__list_templates",
                "description": (
                    "The server-curated template catalog: ready-made "
                    "automation specs by category (work/email/code/"
                    "calendar/school/personal) with declared variables "
                    "to fill. Prefer starting from a matching template "
                    "over authoring a spec from scratch — pass its "
                    "slug as template_slug to automations__create."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "category": {"type": "string",
                                     "description": "Optional filter."},
                    },
                },
            },
            {
                "name": "automations__create",
                "description": (
                    "Validate and save an automation as a DRAFT (not "
                    "firing yet). Returns every validation problem at "
                    "once. Arm it separately after a test run."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "spec": spec_schema,
                        "template_slug": {
                            "type": "string",
                            "description": "Catalog template this spec "
                                           "started from, for provenance.",
                        },
                        "domain": {
                            "type": "string",
                            "description": (
                                "Life domain this automation belongs to: "
                                "'work', 'university', 'personal', or a "
                                "short custom slug the user named. Facts "
                                "it learns are filed under this domain."
                            ),
                        },
                    },
                    "required": ["spec"],
                },
            },
            {
                "name": "automations__update",
                "description": "Replace an automation's spec and "
                               "re-compile its bindings.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "automation_id": {"type": "string"},
                        "spec": spec_schema,
                    },
                    "required": ["automation_id", "spec"],
                },
            },
            {
                "name": "automations__test_run",
                "description": (
                    "One synthetic fire through the REAL rails "
                    "(evaluate → prepare → staged write with the normal "
                    "undo window). Report the sample event and staged "
                    "action to the user."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {"automation_id": {"type": "string"}},
                    "required": ["automation_id"],
                },
            },
            {
                "name": "automations__arm",
                "description": (
                    "Activate a draft/paused automation. Verifies the "
                    "grant (fail closed) and enables the bindings."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {"automation_id": {"type": "string"}},
                    "required": ["automation_id"],
                },
            },
            {
                "name": "automations__pause",
                "description": "Pause an automation (bindings off, spec "
                               "kept).",
                "input_schema": {
                    "type": "object",
                    "properties": {"automation_id": {"type": "string"}},
                    "required": ["automation_id"],
                },
            },
            {
                "name": "automations__resume",
                "description": "Re-arm a paused automation (same checks "
                               "as arm).",
                "input_schema": {
                    "type": "object",
                    "properties": {"automation_id": {"type": "string"}},
                    "required": ["automation_id"],
                },
            },
            {
                "name": "automations__delete",
                "description": "Delete an automation and its bindings. "
                               "Irreversible.",
                "input_schema": {
                    "type": "object",
                    "properties": {"automation_id": {"type": "string"}},
                    "required": ["automation_id"],
                },
            },
        ]

    def get_system_prompt_section(self) -> Optional[str]:
        return (
            "## Automations\n"
            "You can build automations from conversation: watch a "
            "connected service or a schedule, then act through ONE "
            "approved write action pinned to ONE target.\n"
            "The build order is fixed:\n"
            "  1. `automations__get_registry` — see what can fire and "
            "what can write, plus what's connected. Check "
            "`automations__list_templates` for a matching template "
            "before authoring a spec from scratch.\n"
            "  2. PLAN TURN: before any card, tell the user in one "
            "short paragraph what you will build, WHICH connector and "
            "account it uses — name the account when the registry "
            "shows one, e.g. 'your Gmail (person@gmail.com)'; when "
            "account is null, name just the connector — and the "
            "constraints that apply (draft-only mail, poll floor, "
            "undo window). Ask which life domain this belongs to "
            "(work / university / personal, or their own word) unless "
            "it's obvious — then say the one you picked.\n"
            "  3. If the request is ambiguous between connected "
            "services (e.g. 'my email' with both Gmail and Outlook "
            "connected), ask with quick-reply chips on their own "
            "line — `[[Gmail]] [[Outlook]] [[Both]]` — and wait.\n"
            "     When a template variable is a fact the user may "
            "already have told you (their boss's email address, a "
            "channel they always use), check `memory_search` first "
            "and CONFIRM the value with the user instead of asking "
            "cold — never guess an address from nothing.\n"
            "  4. If a needed connector isn't connected (or lacks write "
            "scopes): `automations__request_connection`, then STOP and "
            "wait for the card. Never poll, never assume.\n"
            "  5. For a write action: `automations__list_targets`, ask "
            "the user which target, then "
            "`automations__request_permission` — then STOP and wait.\n"
            "  6. `automations__create` with the full spec (grant_id "
            "from the approved permission) and the `domain`. Fix every "
            "validation error it returns in ONE more call.\n"
            "  7. `automations__test_run`, report what it staged, then "
            "`automations__arm` when the user is happy. Once armed, "
            "the automation gets its own session thread: run cards and "
            "notices land there, not in this conversation.\n"
            "Hard rules you must repeat to the user when relevant: "
            "checks run at most every 5 minutes; runs are capped at 3 "
            "minutes; 3 failures in a row pauses the automation; "
            "email automations can only create DRAFTS, never send; "
            "every write is undoable for ~6 seconds after it fires.\n"
            "A card tool's answer is NOT the outcome — after emitting a "
            "card, end your turn and tell the user you're waiting on "
            "them.\n"
            "This chat and the automation's thread are two places "
            "(same agent, no leakage):\n"
            "- Once an automation exists, its work lives in ITS thread. "
            "This chat only ever receives its notification card. Never "
            "paste a run's findings, steps or commentary here — point "
            "at the run instead.\n"
            "- A setup request typed here gets its setup card, and the "
            "conversation continues in the setup thread — say so and "
            "finish there, not here.\n"
            "- Asked about an automation here ('what did my morning "
            "brief find?', 'what did Marcus want?'): answer from memory "
            "(recall first — the platform memory holds everything the "
            "automations learned and did) and point at the exact run; "
            "never re-run to answer, never restate the briefing.\n"
            "- Its status comes from the engine, stated ONCE per reply "
            "— never 'active' in one sentence and 'paused' in another.\n"
            "- Speak about automations in the user's words: it reads, "
            "it drafts, it tells you. Never engine jargon (no 'Mission "
            "Control', no 'polling', no 'JQL', no 'workflow is live'), "
            "never invented time promises, never a delivery promise "
            "the armed schedule and targets do not literally back."
        )

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------
    async def execute_tool(
        self, tool_name: str, args: Dict[str, Any], ctx: SkillContext,
    ) -> str:
        dispatch = {
            "automations__get_registry": self._get_registry,
            "automations__request_connection": self._request_connection,
            "automations__request_permission": self._request_permission,
            "automations__list_targets": self._list_targets,
            "automations__list": self._list,
            "automations__list_templates": self._list_templates,
            "automations__create": self._create,
            "automations__update": self._update,
            "automations__test_run": self._test_run,
            "automations__arm": self._arm,
            "automations__pause": self._pause,
            "automations__resume": self._resume,
            "automations__delete": self._delete,
        }
        handler = dispatch.get(tool_name)
        if not handler:
            return f"ERROR: Unknown automations tool: {tool_name}"
        if not getattr(settings, "automations_enabled", False):
            return "ERROR: Feature not available"
        try:
            return await handler(args, ctx)
        except Exception as e:  # noqa: BLE001 — last-resort guard
            logger.exception("[automations_skill] %s failed", tool_name)
            return f"ERROR: {type(e).__name__}: {str(e)[:300]}"

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------
    async def _get_registry(self, args, ctx) -> str:
        from app.agent.automations import registry as reg
        uid = _uid(ctx)
        capability = await reg.fetch_registry(uid)
        if not capability:
            return ("ERROR: The automations registry is unreachable right "
                    "now — nothing can be configured. Tell the user and "
                    "try later.")
        connections = await reg.fetch_connection_state(uid)
        out = []
        for cid, cap in sorted(capability.items()):
            conn = connections.get(cid) or {}
            out.append({
                "connector_id": cid,
                "name": cap.get("name"),
                "connected": bool(conn.get("connected")),
                # R28 disclosure: WHICH account this connector is bound
                # to (Gmail address etc.); None when the provider never
                # told us — say the connector name alone then.
                "account": conn.get("account"),
                "granted_scopes": conn.get("scopes") or [],
                "push": cap.get("push"),
                "poll": cap.get("poll"),
                "floor_s": cap.get("floor_s"),
                "events": cap.get("events") or [],
                "write_actions": {
                    tool: {
                        "scopes": scopes,
                        "target_param": (cap.get("target_param_by_action")
                                         or {}).get(tool),
                    }
                    for tool, scopes in
                    (cap.get("scopes_write_by_action") or {}).items()
                },
                "rate_budget": cap.get("rate_budget") or {},
            })
        return _as_json({"connectors": out})

    async def _request_connection(self, args, ctx) -> str:
        from app.agent.automations import registry as reg
        from app.agent.automations import cards
        from app.db.database import async_session_maker
        from app.db.models import (
            AutomationAuthSession, AUTOMATION_AUTH_SESSION_TTL_S,
        )

        uid = _uid(ctx)
        connector_id = (args.get("connector_id") or "").strip()
        mode = args.get("mode") or "read"
        capability = await reg.fetch_registry(uid)
        cap = capability.get(connector_id)
        if cap is None:
            return (f"ERROR: {connector_id!r} is not an automatable "
                    f"connector. Known: {sorted(capability)}")

        conn = (await reg.fetch_connection_state(uid)).get(connector_id) or {}
        needed: list[dict] = []
        descriptions = cap.get("scope_descriptions") or {}
        for s in cap.get("scopes_read") or []:
            needed.append({"scope": s,
                           "description": descriptions.get(s, ""),
                           "write": False})
        if mode == "read_write":
            for scopes in (cap.get("scopes_write_by_action") or {}).values():
                for s in scopes:
                    if all(n["scope"] != s for n in needed):
                        needed.append({"scope": s,
                                       "description": descriptions.get(s, ""),
                                       "write": True})
        granted = set(conn.get("scopes") or [])
        missing = [n for n in needed if n["scope"] not in granted]
        if conn.get("connected") and not missing:
            return (f"{connector_id} is already connected with every scope "
                    f"this automation needs — no card required. Continue.")

        async with async_session_maker() as db:
            session = AutomationAuthSession(
                user_id=uid,
                connector_id=connector_id,
                mode=mode,
                scopes_json=json.dumps(needed, default=str),
                status="offered",
                conversation_id=(ctx.extra or {}).get("conversation_id"),
                expires_at=datetime.utcnow()
                + timedelta(seconds=AUTOMATION_AUTH_SESSION_TTL_S),
            )
            db.add(session)
            await db.commit()

            payload = cards.connector_card_payload(
                session, name=cap.get("name") or connector_id,
                icon=cap.get("icon"), scopes=needed,
            )
            message_id, _ = await cards.write_card_message(
                db,
                user_id=uid,
                content=(f"To set this up I need access to "
                         f"{cap.get('name') or connector_id}."),
                metadata_key=cards.CONNECTOR_CARD_KEY,
                payload=payload,
                title="Connect a service",
            )
            session2 = await db.get(AutomationAuthSession, session.id)
            session2.message_id = message_id
            await db.commit()
        await cards.broadcast_card(uid, cards.CONNECTOR_CARD_KEY, payload)
        return (
            "CONNECTOR CARD SHOWN — the user must tap Connect or Reject "
            "(expires in 10 minutes). NOT CONNECTED YET. Do NOT call this "
            "tool again and do NOT proceed as if connected. End your turn "
            "telling the user you're waiting on the card, and finish your "
            "message with the quick-reply chip "
            "[[Proceed with connection setup]] so they can nudge you once "
            "they've connected."
        )

    async def _request_permission(self, args, ctx) -> str:
        from app.agent.automations import registry as reg
        from app.agent.automations import cards
        from app.db.database import async_session_maker

        uid = _uid(ctx)
        connector_id = (args.get("connector_id") or "").strip()
        tool = (args.get("tool") or "").strip()
        target = args.get("target") or {}
        if tool in ("gmail__send_message", "outlook__send_message"):
            return ("ERROR: Automations never send mail — request "
                    "gmail__create_draft instead (drafts only).")
        grant = await reg.create_grant_request(
            uid,
            connector_id=connector_id,
            tool_name=tool,
            target=target,
            cadence=args.get("cadence"),
            mode=args.get("mode") or "confirm",
            summary=(args.get("summary") or "")[:300],
            preview=args.get("preview"),
            automation_id=args.get("automation_id"),
        )
        if grant is None:
            return ("ERROR: The permission request could not be prepared "
                    "(invalid action/target, or the platform is "
                    "unreachable). Nothing was shown to the user.")

        payload = {
            "id": grant["id"],
            "automation_id": grant.get("automation_id"),
            "connector_id": grant["connector_id"],
            "action": grant["tool_name"],
            "action_label": grant["tool_name"].split("__", 1)[-1]
            .replace("_", " "),
            "target": grant.get("target") or {},
            "cadence": grant.get("cadence") or {},
            "mode": grant.get("mode"),
            "summary": grant.get("summary"),
            "preview": args.get("preview"),
            "status": grant.get("status"),
            "created_at": grant.get("created_at"),
            "expires_at": grant.get("expires_at"),
            "decided_at": None,
            "decided_via": None,
        }
        async with async_session_maker() as db:
            await cards.write_card_message(
                db,
                user_id=uid,
                content=f"Permission needed: {grant.get('summary')}",
                metadata_key=cards.GRANT_CARD_KEY,
                payload=payload,
                title="Permission request",
            )
        await cards.broadcast_card(uid, cards.GRANT_CARD_KEY, payload)
        return (
            f"PERMISSION CARD SHOWN (grant_id={grant['id']}, expires in "
            f"1 hour) — NOT APPROVED YET. Do NOT use this grant_id in an "
            f"armed automation until the user approves. End your turn "
            f"telling the user you're waiting on their decision, and "
            f"finish your message with the quick-reply chip "
            f"[[Continue setting this up]]."
        )

    async def _list_targets(self, args, ctx) -> str:
        from app.agent.automations import registry as reg
        uid = _uid(ctx)
        connector_id = (args.get("connector_id") or "").strip()
        src = _TARGET_SOURCES.get(connector_id)
        if src is None:
            return (f"ERROR: No target listing exists for "
                    f"{connector_id!r}. Ask the user for the exact "
                    f"target id instead.")
        result = await reg.dispatch_via_platform(
            uid, connector_id=connector_id, tool_name=src["tool"],
            tool_input={},
        )
        if result.get("kind") != "ok":
            return (f"ERROR: Could not list targets — "
                    f"{result.get('kind')}: "
                    f"{str(result.get('message') or '')[:200]}")
        try:
            content = json.loads(result.get("content") or "{}")
        except (ValueError, TypeError):
            content = {}
        items = content.get(src["items"]) or []
        targets = [
            {"kind": src["kind"],
             "id": str(i.get(src["id"]) or ""),
             "label": str(i.get(src["label"]) or i.get(src["id"]) or "")}
            for i in items if isinstance(i, dict) and i.get(src["id"])
        ]
        return _as_json({"targets": targets[:100]})

    async def _list(self, args, ctx) -> str:
        from app.agent.automations.service import list_automations
        from app.db.database import async_session_maker
        async with async_session_maker() as db:
            rows = await list_automations(db, _uid(ctx))
        return _as_json({"automations": rows})

    async def _list_templates(self, args, ctx) -> str:
        from app.agent.automations import registry as reg
        templates = await reg.fetch_templates(_uid(ctx))
        category = (args.get("category") or "").strip().lower()
        if category:
            templates = [t for t in templates
                         if (t.get("category") or "") == category]
        if not templates:
            return ("No templates available" +
                    (f" in category {category!r}" if category else "") +
                    " — build the spec from the registry instead.")
        return _as_json({"templates": templates})

    async def _create(self, args, ctx) -> str:
        from app.agent.automations.service import create_automation
        from app.agent.automations.spec import SpecError
        from app.db.database import async_session_maker
        try:
            async with async_session_maker() as db:
                automation, _ = await create_automation(
                    db, user_id=_uid(ctx), spec=args.get("spec"),
                    template_slug=args.get("template_slug"),
                    domain=args.get("domain"),
                )
        except SpecError as e:
            return "SPEC INVALID:\n" + _as_json(e.errors)
        return (f"Created automation {automation.id!r} "
                f"({automation.name}) as a DRAFT. Run "
                f"automations__test_run, then automations__arm.")

    async def _update(self, args, ctx) -> str:
        from app.agent.automations.service import (
            AutomationNotFound, update_automation,
        )
        from app.agent.automations.spec import SpecError
        from app.db.database import async_session_maker
        try:
            async with async_session_maker() as db:
                automation, _ = await update_automation(
                    db, automation_id=args.get("automation_id") or "",
                    user_id=_uid(ctx), spec=args.get("spec"),
                )
        except AutomationNotFound:
            return "ERROR: No such automation."
        except SpecError as e:
            return "SPEC INVALID:\n" + _as_json(e.errors)
        return (f"Updated {automation.id!r}; status is now "
                f"{automation.status!r}.")

    async def _test_run(self, args, ctx) -> str:
        from app.agent.automations.service import (
            AutomationNotFound, test_run,
        )
        from app.db.database import async_session_maker
        try:
            async with async_session_maker() as db:
                result = await test_run(
                    db, automation_id=args.get("automation_id") or "",
                    user_id=_uid(ctx),
                )
        except AutomationNotFound:
            return "ERROR: No such automation."
        return ("TEST RUN STAGED (the write goes out after the normal "
                "undo window):\n" + _as_json(result))

    async def _lifecycle(self, args, ctx, verb) -> str:
        from app.agent.automations import service
        from app.agent.automations.compiler import CompileError
        from app.agent.automations.service import AutomationNotFound
        from app.db.database import async_session_maker
        fn = {
            "arm": service.arm_automation,
            "pause": service.pause_automation,
            "resume": service.resume_automation,
            "delete": service.delete_automation,
        }[verb]
        try:
            async with async_session_maker() as db:
                out = await fn(
                    db, automation_id=args.get("automation_id") or "",
                    user_id=_uid(ctx),
                )
        except AutomationNotFound:
            return "ERROR: No such automation."
        except CompileError as e:
            return f"ERROR ({e.code}): {e}"
        if verb == "delete":
            return "Deleted."
        return f"OK — status is now {out.status!r}."

    async def _arm(self, args, ctx) -> str:
        out = await self._lifecycle(args, ctx, "arm")
        if out.startswith("OK"):
            try:
                await self._file_setup_fact(args, ctx)
            except Exception as e:  # noqa: BLE001 — memory is a companion
                logger.warning("[automations] setup fact skipped: %s", e)
        return out

    async def _file_setup_fact(self, args, ctx) -> None:
        """On a successful arm, file ONE clean fact under the
        automation's domain — composed from setup intent (name, what it
        watches, what it does), never from provider data. No domain, no
        fact."""
        from sqlalchemy import select
        from app.agent.automations import memory_notes
        from app.db.database import async_session_maker
        from app.db.models import Automation

        async with async_session_maker() as db:
            a = (await db.execute(
                select(Automation).where(
                    Automation.id == (args.get("automation_id") or ""),
                    Automation.user_id == _uid(ctx),
                )
            )).scalar_one_or_none()
            if a is None or not a.domain:
                return
            fact: Optional[str] = None
            try:
                # R29: the verbs module owns the human sentence — a
                # derived "<verb>s via <connector>" was a raw tool name
                # wearing spaces.
                from app.services.automation_verbs import rule_sentence
                raw = json.loads(a.spec_json)
                sentence = rule_sentence(raw) if isinstance(raw, dict) else None
                if sentence:
                    fact = (f'Has an automation "{a.name[:60]}": '
                            f"{str(sentence)[:160]}")
            except Exception:  # noqa: BLE001 — composition falls back
                pass
            if fact is None:
                if a.trigger_mode == "schedule":
                    trigger = "runs on a schedule"
                elif a.connector_id:
                    trigger = f"watches {a.connector_id}"
                else:
                    trigger = "watches a connected service"
                fact = memory_notes.setup_fact(
                    automation_name=a.name,
                    trigger_summary=trigger,
                    action_summary="acts on it",
                )
            # Ledger-first (CONTRACTS-R29 §4): `record` stamps
            # attribution AND projects to the brain itself — the R28
            # memory_notes path stays only as the pre-seam fallback.
            try:
                from app.agent.automations import facts as facts_seam
                result = await facts_seam.record(
                    db,
                    user_id=_uid(ctx),
                    automation_id=a.id,
                    facts=[fact],
                    category=a.domain,
                    source="agent",
                    source_kind="chat",
                )
                if int((result or {}).get("saved", 0)) > 0:
                    return
            except ImportError:
                pass
            await memory_notes.record_automation_fact(
                db, user_id=_uid(ctx), domain=a.domain, fact=fact,
            )

    async def _pause(self, args, ctx) -> str:
        return await self._lifecycle(args, ctx, "pause")

    async def _resume(self, args, ctx) -> str:
        return await self._lifecycle(args, ctx, "resume")

    async def _delete(self, args, ctx) -> str:
        return await self._lifecycle(args, ctx, "delete")
