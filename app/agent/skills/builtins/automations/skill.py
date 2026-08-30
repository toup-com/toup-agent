"""Automations setup skill (Round 26) — the chat-built engine's tools.

Eleven tools mirror the round brief exactly:

  automations__get_registry       — what can fire / what can be written
  automations__request_connection — connector card (10-min TTL)
  automations__request_permission — grant card (1-h TTL)
  automations__list_targets       — pinnable targets for a connector
  automations__create             — validate + save a DRAFT
  automations__update             — replace spec, re-compile
  automations__test_run           — a rehearsal: reads for real, reports
                                    what it WOULD write, sends nothing
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
from app.agent.tool_display import ToolResult
from app.config import settings

logger = logging.getLogger(__name__)


def _as_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, default=str, ensure_ascii=False)


def _n_things(count: int, singular: str, plural: str) -> str:
    return f"1 {singular}" if count == 1 else f"{count} {plural}"


def _recall_summary(result: Dict[str, Any]) -> str:
    """The user-facing half of a memory read (R31-28).

    A tool's summary says what happened in the user's words. The JSON
    the model reads is not that, and neither is a line of coaching
    addressed to the model — both have been on a job sheet.
    """
    facts = len(result.get("facts") or [])
    episodes = len(result.get("episodes") or [])
    parts = []
    if facts:
        parts.append(_n_things(facts, "fact", "facts"))
    if episodes:
        parts.append(_n_things(episodes, "run", "runs"))
    if not parts:
        return "Looked in memory · nothing matched"
    return "Looked in memory · " + " and ".join(parts)


def _uid(ctx: SkillContext) -> str:
    return (ctx.user_id or getattr(settings, "user_id", "") or "").strip()


def _workflow_digest(wf: Dict[str, Any]) -> Dict[str, Any]:
    """The automation as it stands, in the fields an edit needs next.

    The five edit tools answer with this so the model never has to
    guess a rule id, a step number or a permission label — and so a
    refusal ("there is no step 4") arrives beside the three steps that
    do exist. The full §4.4 payload carries presets, connection states
    and last-use sentences that no edit reads; sending them would be
    the largest tool result in the skill for the least use.
    """
    return {
        "runs": (wf.get("schedule") or {}).get("sentence") or "",
        "steps": [{"n": s.get("n"), "text": s.get("text")}
                  for s in (wf.get("steps") or [])],
        "rules": [{"id": r.get("id"), "text": r.get("text")}
                  for r in (wf.get("rules") or [])],
        "accounts": [
            {"account_id": a.get("account_id"), "name": a.get("name"),
             "can": [p.get("label") for p in (a.get("can") or [])],
             "cannot": [p.get("label") for p in (a.get("cant") or [])]}
            for a in (wf.get("accounts") or [])
        ],
    }


# Per-connector "list the pinnable targets" read tools. The kind names
# what the target IS, so the grant card can say "channel #eng" and not
# just an opaque id.
#: Tools the model may not see outside dev/e2e. EMPTY since R38 — kept,
#: with its history, because it is the door gate as well as the array
#: filter and the next dev-only tool needs both.
#:
#: `automations__test_run` was the entry here, for two reasons that
#: were both true of the OLD implementation. It was not a rehearsal:
#: `stage_only=True` returned before narration, but the outbox row it
#: had committed was swept and sent by `outbox.flush_loop` like any
#: other, so a "test" posted to the user's real channel seconds later.
#: And it short-circuited the ledger's phase-2 close, so the run it
#: opened produced no result turn and was later reaped as failed.
#:
#: R38 fixed the implementation instead of keeping the gate:
#: `service.rehearse` stages NOTHING and opens no run, so no loop,
#: restart or retry can turn a rehearsal into a send. The gate came off
#: only once that was structurally true — a flag over the old path
#: would have been the same tool with a smaller audience.
_DEV_ONLY_TOOLS: frozenset[str] = frozenset()


def _dev_tools_active() -> bool:
    from app.agent.automations.spec import dev_fast_lane_active
    return dev_fast_lane_active()


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
        return [t for t in self._all_tools()
                if t["name"] not in _DEV_ONLY_TOOLS or _dev_tools_active()]

    def _all_tools(self) -> List[Dict[str, Any]]:
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
                    "user's current connection state."
                    # Round 33, item 2: this said "ALWAYS call this before
                    # proposing an automation." A tool description is an
                    # instruction the model competes over (§14 rule 1), and
                    # an unconditional imperative in one rides EVERY turn.
                    # The ordering already lives in the section that owns
                    # the flow, which is where a flow rule belongs.
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
                # The counterweight. ND-18: this tool answers the
                # inventory question, but said so in nine bland words
                # while `routines__list` quoted the question verbatim —
                # and lost it. The claim belongs to whoever owns the
                # answer.
                "description": "This user's automations, with status and "
                               "health. Call this whenever the user asks "
                               "what automations they have, how many, or "
                               "what they are called — it is the ONLY "
                               "list of their automations. Reminders and "
                               "scheduled tasks are a different surface "
                               "and never belong in that answer.",
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
                    "A REHEARSAL: read this automation's sources for "
                    "real and answer with exactly what it WOULD write, "
                    "without writing it. Nothing is sent, nothing is "
                    "saved, and it does not count as a run. Use it to "
                    "show someone what a draft automation will produce "
                    "before they arm it, or to find out why one is "
                    "producing the wrong thing. It is NOT how you run "
                    "something a user asked you to run — that is "
                    "automations__run_now, which does the work for "
                    "real."
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
            # Appended LAST (R30) — the tools array is prefix-stable
            # per channel; new tools only ever join at the end.
            #
            # The `automations__` prefix is MANDATORY, not cosmetic:
            # SkillLoader._register RAISES on the first tool name that
            # lacks it and load_all swallows the raise, so a bare name
            # here does not "register one unprefixed tool" — it discards
            # the ENTIRE automations skill. Registered bare, this cost
            # the chat agent all thirteen automations tools on main
            # (wire 90 → 77, zero tool calls in a live run). The prefix
            # is a namespace for the loader's tool index; it has nothing
            # to do with which surface may call the tool.
            {
                "name": "automations__memory_recall",
                # Descriptive, never a flow posture: a "do this first"
                # instruction here competes for the early iterations a
                # setup conversation needs (CONTRACTS-R30 §14 rule 1).
                # The recall-first rule lives in the sections that own
                # those answers — the automations section below and the
                # automation-thread posture (§5.4).
                "description": "Search the one platform memory — facts "
                               "and episodes about people, channels, "
                               "tickets, repos and past automation runs, "
                               "with links to the exact thread turn. Use "
                               "when the user asks about a person, a "
                               "channel, a ticket, or what an automation "
                               "did.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string",
                                  "description": "free-text match"},
                        "entity": {"type": "string",
                                   "description": "a person/channel/"
                                                  "ticket/repo name"},
                        "category": {
                            "type": "string",
                            "enum": ["people", "team_workspace",
                                     "your_time", "work_you_own",
                                     "noise_filters"],
                        },
                        "scope": {"type": "string",
                                  "description": "an automation id to "
                                                 "read its scoped view "
                                                 "(plus global); omit "
                                                 "to span everything"},
                        "since": {"type": "string",
                                  "description": "ISO date floor"},
                    },
                },
            },
            # R31-04. Appended last, per the prefix-stable rule above.
            #
            # Until this existed, the model had NO way to run an
            # automation: `run-now` was a UI-only route, so a user
            # saying "run all of them again" left the model reaching
            # for `automations__test_run` — the synthetic path — which
            # answered "TEST RUN STAGED" and reported a status of
            # `paused` instead of running anything. The test run was
            # not even a dry run: its staged write is swept and sent by
            # the outbox like any other.
            #
            # DEPENDENCY (R31-41 / ND-24): the proxy trims the tools
            # array to 128 FROM THE TAIL, so the last-appended tool is
            # the first dropped. On a user with many connectors this
            # tool is unreachable until that cap is fixed, and the
            # symptom is indistinguishable from the defect it repairs —
            # the model reaching for whatever run-shaped tool it can
            # still see. D verifies reachability on the founder's
            # account after A lands the cap fix.
            {
                "name": "automations__run_now",
                "description": (
                    "Run one automation immediately, for real. This is "
                    "the ONLY way to run an automation — use it "
                    "whenever the user asks to run, re-run, or try one "
                    "again, in the automation's own thread or in chat. "
                    "It starts a real run through the engine: the run "
                    "reports itself, so do not describe what it will "
                    "do or report a status instead of running it. A "
                    "paused automation runs once and stays paused."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {"automation_id": {"type": "string"}},
                    "required": ["automation_id"],
                },
            },
            # R37. Appended last, per the prefix-stable rule above.
            #
            # The one-call write-back for the decision every setup
            # thread ends on: WHERE the result goes. Before this, a
            # destination change needed `automations__update` with a
            # complete replacement spec — a round trip through
            # `automations__list` plus a hand-rebuilt steps array, with
            # a grant id the model does not have — so the agent agreed
            # in words ("I'll keep it in this chat") and the workflow
            # never moved (the founder watched the canvas still say
            # "Told you in Slack" three messages later).
            {
                "name": "automations__set_destination",
                "description": (
                    "Point one automation's delivery somewhere, for "
                    "real. destination 'chat' removes the outside write "
                    "entirely — the result lands in the automation's "
                    "own thread (plus its chat card), no permission "
                    "needed, and the automation is armed if its "
                    "accounts are connected. A connector target (e.g. "
                    "a Slack channel from automations__list_targets) "
                    "pins the write there and asks the user for the "
                    "permission it needs — the run stays blocked until "
                    "they approve. Never claim a destination changed "
                    "without calling this."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "automation_id": {"type": "string"},
                        "destination": {
                            "type": "string",
                            "enum": ["chat", "connector"],
                            "description": "'chat' = this thread; "
                                           "'connector' pins `target` "
                                           "on the write step",
                        },
                        "connector_id": {
                            "type": "string",
                            "description": "required for 'connector'",
                        },
                        "target": {
                            "type": "object",
                            "description": "for 'connector': the pinned "
                                           "place, from "
                                           "automations__list_targets",
                            "properties": {
                                "kind": {"type": "string"},
                                "id": {"type": "string"},
                                "label": {"type": "string"},
                            },
                        },
                    },
                    "required": ["automation_id", "destination"],
                },
            },
            # R38. Appended last, per the prefix-stable rule above; the
            # five join in the order the canvas reads top to bottom.
            #
            # The five change kinds of `composer.CHANGE_KINDS`, as five
            # tools. Before these, the ONLY way for the agent to change
            # a live automation was `automations__update` with a
            # complete replacement spec: a round trip through
            # `automations__list`, a hand-rebuilt steps array, and a
            # grant id the model does not have. So the agent agreed in
            # words and the workflow did not move — R37's item, one
            # layer up.
            #
            # Every one of them goes through `workflow.apply_intents`,
            # which is `composer.apply_policy` (the deterministic gate)
            # followed by `workflow._apply_intent` (the same writers the
            # canvas sheet uses). That means: the same validation, the
            # same undo token, the same EDITED note, the same
            # `automation.updated` broadcast — and the same refusals. A
            # tool cannot widen access that a sentence cannot: a grant,
            # an account add and a hard rail all come back unapplied,
            # with the sentence that says so.
            {
                "name": "automations__edit_schedule",
                "description": (
                    "Change WHEN one automation runs, for real. Give "
                    "either a preset_id (weekdays-8, weekdays-730, "
                    "daily-8, weekdays-9) or a time like '07:30' with "
                    "the days it should run. The change applies "
                    "immediately and the automation says so in its own "
                    "thread. Never tell the user a time changed "
                    "without calling this."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "automation_id": {"type": "string"},
                        "preset_id": {
                            "type": "string",
                            "enum": ["weekdays-8", "weekdays-730",
                                     "daily-8", "weekdays-9"],
                        },
                        "time": {
                            "type": "string",
                            "description": "24-hour local time, 'HH:MM'. "
                                           "Use instead of preset_id "
                                           "for a time the presets do "
                                           "not carry.",
                        },
                        "days": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "ISO weekdays for `time` — "
                                           "1=Monday … 7=Sunday. Omit "
                                           "for every day.",
                        },
                        "date": {
                            "type": "string",
                            "description": "'YYYY-MM-DD' for a one-time "
                                           "run at `time` instead of a "
                                           "repeating one.",
                        },
                    },
                    "required": ["automation_id"],
                },
            },
            {
                "name": "automations__edit_rules",
                "description": (
                    "Add, reword or drop the standing lines one "
                    "automation will not cross ('never post anywhere "
                    "else', 'only unread mail'). A rule is stored in "
                    "the user's own words. To remove or reword one, "
                    "give its id or its exact current text — the tool "
                    "answers with every rule and its id, so one call is "
                    "always enough to find it."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "automation_id": {"type": "string"},
                        "add": {
                            "type": "array", "items": {"type": "string"},
                            "description": "new rules, the user's own "
                                           "words, one sentence each",
                        },
                        "remove": {
                            "type": "array", "items": {"type": "string"},
                            "description": "rule ids, or the exact text "
                                           "of the rule to drop",
                        },
                        "edit": {
                            "type": "array",
                            "description": "reword an existing rule",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "rule_id": {"type": "string"},
                                    "text": {"type": "string"},
                                },
                            },
                        },
                    },
                    "required": ["automation_id"],
                },
            },
            {
                "name": "automations__edit_steps",
                "description": (
                    "Reword what a numbered step of one automation "
                    "does. `n` is the step's position as the thread and "
                    "the canvas show it, counting from 1. This changes "
                    "the WORDING the user reads and asks the engine to "
                    "recompile the plan behind it; it cannot add or "
                    "remove a step, and it cannot point a step at a "
                    "different account — use automations__update for a "
                    "different plan, and automations__set_destination "
                    "to move where the result goes."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "automation_id": {"type": "string"},
                        "n": {"type": "integer",
                              "description": "1-based step position"},
                        "text": {"type": "string",
                                 "description": "what that step should "
                                                "do, in one line"},
                    },
                    "required": ["automation_id", "n", "text"],
                },
            },
            {
                "name": "automations__edit_permissions",
                "description": (
                    "Take one permission away from one account on one "
                    "automation ('Slack must never post as me'). "
                    "`permission` is the label or id the account's card "
                    "shows. Taking one away applies immediately. GIVING "
                    "one is the user's call and this tool will not do "
                    "it — it answers with what to ask for, and "
                    "automations__request_permission is what asks."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "automation_id": {"type": "string"},
                        "account_id": {
                            "type": "string",
                            "description": "the connector, e.g. 'slack'",
                        },
                        "permission": {
                            "type": "string",
                            "description": "the label or id of the one "
                                           "permission to change",
                        },
                        "direction": {
                            "type": "string",
                            "enum": ["revoke", "grant"],
                            "description": "'revoke' applies; 'grant' "
                                           "comes back needing the "
                                           "user's yes",
                        },
                    },
                    "required": ["automation_id", "account_id",
                                 "permission", "direction"],
                },
            },
            {
                "name": "automations__edit_accounts",
                "description": (
                    "Take one account out of an automation ('stop using "
                    "Jira for this'). Removing applies immediately, and "
                    "the answer says whether the automation stopped "
                    "running as a result. CONNECTING a new account is "
                    "the user's call and this tool will not do it — it "
                    "answers with what to ask for, and "
                    "automations__request_connection is what asks."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "automation_id": {"type": "string"},
                        "account_id": {
                            "type": "string",
                            "description": "the connector, e.g. 'jira'",
                        },
                        "direction": {
                            "type": "string",
                            "enum": ["remove", "add"],
                        },
                    },
                    "required": ["automation_id", "account_id",
                                 "direction"],
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
            "constraints that apply (draft-only mail, how often it "
            "checks, and that anything it writes can be taken back "
            "for a few seconds afterwards). Ask which life domain "
            "this belongs to "
            "(work / university / personal, or their own word) unless "
            "it's obvious — then say the one you picked.\n"
            "  3. If the request is ambiguous between connected "
            "services (e.g. 'my email' with both Gmail and Outlook "
            "connected), ask with quick-reply chips on their own "
            "line — `[[Gmail]] [[Outlook]] [[Both]]` — and wait. "
            "That syntax works HERE, in the main chat, and only here: "
            "inside an automation's own thread it is not a button, it "
            "is four literal brackets in a sentence. Ask in words "
            "there.\n"
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
            "  7. `automations__arm` when the user is happy, then "
            "offer to run it once now with `automations__run_now` if "
            "they want to see it work. Once armed, the automation gets "
            "its own thread: run cards and notices land there, not in "
            "this conversation.\n"
            "SHOWING SOMEONE WHAT IT WILL DO, BEFORE IT DOES IT. "
            "`automations__test_run` is a REHEARSAL: it reads the "
            "automation's sources for real and answers with exactly "
            "what it would write, and it sends nothing, saves nothing "
            "and does not count as a run. Reach for it when the user "
            "wants to see what a draft will produce before arming it, "
            "or when one is producing the wrong thing and you need to "
            "see why. It is NEVER the answer to 'run it' or 'run it "
            "again': a rehearsal writes nothing, so reporting one as a "
            "run is a lie about work that did not happen. And never "
            "describe its answer as something the automation did — it "
            "did not post, send, draft or tell anyone anything.\n"
            "RUNNING ONE, NOW OR AGAIN. 'run it', 'run it again', 'run "
            "all of them again', 'try again', 'do it now' — here or in "
            "an automation's own thread — mean `automations__run_now`, "
            "always, with the id from `automations__list`. Never "
            "answer a request to RUN with a description of what it "
            "would do, and never with a status: 'its status is paused' "
            "is not a reply to 'run it again', it is a way of not "
            "doing it. If the tool refuses, say the refusal's own "
            "sentence and then DO the fix it names when you have a "
            "tool for it (an unpinned destination is yours to fix: "
            "offer the places, then `automations__set_destination`). "
            "For several at once, call the tool once per "
            "automation. Then say one short line — the run narrates "
            "itself in its own thread, and anything you add here is a "
            "second account of the same run that will disagree with "
            "the first.\n"
            "INSIDE AN AUTOMATION'S OWN THREAD you are the OPERATOR of "
            "this one workflow, and the turn's context carries its "
            "setup RIGHT NOW — answer from that block, never from "
            "older turns that describe how it used to be. The rules:\n"
            "  - Use what you can already see before asking anything. "
            "'Which Slack channel?' is a failure when "
            "`slack__list_channels` (or `automations__list_targets`) "
            "is one call away — offer the real ones by name, at most "
            "once.\n"
            "  - Only offer channels the workspace has actually joined "
            "— `automations__list_targets` already answers with exactly "
            "those. Never name a channel that is not in the tool's "
            "answer: a channel named from habit ('#general') may not "
            "exist here, and one the workspace has not joined can never "
            "be posted to.\n"
            "  - 'whatever', 'any of them', 'you pick' is an ANSWER: "
            "pick the most sensible existing target yourself, apply "
            "it, and say which you picked in the same breath. Do not "
            "re-ask.\n"
            "  - A decision the user states ('keep it in this chat', "
            "'post it to #general', 'make it 9am') is applied WITH A "
            "TOOL in the same turn. Saying 'got it' without the tool "
            "call is the lie the canvas exposes three messages later. "
            "One tool per kind of change, and each one applies for "
            "real:\n"
            "      · where it delivers → `automations__set_destination`\n"
            "      · when it runs → `automations__edit_schedule`\n"
            "      · a line it must not cross ('never post anywhere "
            "else', 'only unread') → `automations__edit_rules`\n"
            "      · what a numbered step does → "
            "`automations__edit_steps`\n"
            "      · taking a permission away ('never post as me') → "
            "`automations__edit_permissions`\n"
            "      · taking one account out ('stop using Jira for "
            "this') → `automations__edit_accounts`\n"
            "      · anything else about the plan itself → "
            "`automations__update`\n"
            "    Each edit tool answers with what CHANGED, what did "
            "NOT and why, and what the automation looks like now. Read "
            "all three. If it says nothing changed, say that reason in "
            "your own words — never report a change you did not make. "
            "GIVING a permission and CONNECTING a new account are the "
            "user's call: those come back unapplied, and the way to ask is "
            "`automations__request_permission` / "
            "`automations__request_connection`.\n"
            "  - After a confirmed decision, never re-confirm it. "
            "'Yep' means move: apply it, then name the next real step "
            "or offer the first run.\n"
            "  - When the ask is for NOW — 'show me', 'do it', 'run "
            "it', 'read at least 10 and show me here', any present-"
            "tense imperative — apply the setting AND call "
            "`automations__run_now` in the SAME turn. Restating the "
            "schedule ('It will summarize here every weekday at 8:00') "
            "without running is the failure mode: they asked for now, "
            "and a description of later is a way of not doing it.\n"
            "  - Never say 'below' or 'above' about a card or button "
            "you are not attaching in that same message. The account "
            "cards carry their own buttons; your prose states the fact "
            "('I could not read Outlook — the access ran out') and "
            "stops. And if you remove one of its accounts, name it in "
            "one plain sentence ('I took Outlook out of this "
            "automation.') — never remove one silently while talking "
            "about something else.\n"
            # These are promises made to a user about what the engine
            # will do, so every one of them has to be true of the
            # engine as it is now — not as the round intends it. The
            # three-strike rule is restored because it IS true today
            # (`sweep._sweep_auto_pause`, AUTOMATION_AUTO_PAUSE_FAILURES
            # = 3), and a draft of this section had replaced it with
            # "one broken account never pauses an automation and never
            # stops a run", which is R31 §4.2a's intent and not yet the
            # code: `on_error` still defaults to "fail" (spec_v2.py) and
            # a failed read step finalizes the run as failed. Telling a
            # user their automation cannot be paused by a broken account
            # and then having it paused that afternoon is worse than
            # saying nothing. When §4.2a lands, this sentence changes
            # with it — not before.
            "Hard rules you must repeat to the user when relevant: "
            "checks run at most every 5 minutes; runs are capped at 3 "
            "minutes; 3 failed runs in a row pause the automation; "
            "email automations can only create DRAFTS, never send; "
            "every write is undoable for ~6 seconds after it fires. "
            "When one of its accounts is broken, name that account, "
            "say the real reason, and say what fixes it — never leave "
            "the user to guess which one it was.\n"
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
            "- Asked what automations they have, or how many: the answer "
            "is `automations__list`, and only that. Reminders and "
            "scheduled tasks (the `routines__*` surface) are NOT "
            "automations — never add them to that list or that count, "
            "however many of them exist. If the user means their "
            "reminders, answer about reminders and call them reminders. "
            "The same boundary holds for running one: an automation is "
            "run by `automations__run_now` and a reminder by its own "
            "surface. `routines__run_now` describes itself in words "
            "that sound like a briefing ('run my morning briefing "
            "now'), so check which object the user named — if it came "
            "back from `automations__list`, it is an automation.\n"
            "- **Never name an automation you have not just read from "
            "`automations__list`.** Not from memory, not from earlier in "
            "this conversation, not from what you recall of their setup. "
            "If the tool fails or you cannot call it, say plainly that "
            "you cannot see their automations right now and name none — "
            "asking them to try again is a fine answer. Inventing one, "
            "or attaching a guessed status ('paused', 'needs "
            "reconnecting') to a name you did not read, is the worst "
            "answer available: it is confident and wrong, and the user "
            "has no way to tell.\n"
            "- Asked what an automation FOUND or DID ('what did my "
            "morning brief find?', 'what did Marcus want?'): answer from "
            "memory (recall first — the platform memory holds everything "
            "the automations learned and did) and point at the exact "
            "run; never re-run to answer, never restate the briefing. "
            "This is about the contents of PAST runs — an inventory ask "
            "(what they have, how many) is the rule above, and memory is "
            "not the place to count them. It is also not about a request "
            "for something NEW: 'what is in my inbox now' is a fresh "
            "read, not a question about a run that already happened, and "
            "inside the automation's own thread that read is a run of "
            "its own. The rule here is that a finished run is not "
            "repeated to describe itself.\n"
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
            "automations__run_now": self._run_now,
            "automations__arm": self._arm,
            "automations__pause": self._pause,
            "automations__resume": self._resume,
            "automations__delete": self._delete,
            "automations__memory_recall": self._recall,
            "automations__set_destination": self._set_destination,
            "automations__edit_schedule": self._edit_schedule,
            "automations__edit_rules": self._edit_rules,
            "automations__edit_steps": self._edit_steps,
            "automations__edit_permissions": self._edit_permissions,
            "automations__edit_accounts": self._edit_accounts,
        }
        handler = dispatch.get(tool_name)
        if not handler:
            return f"ERROR: Unknown automations tool: {tool_name}"
        if not getattr(settings, "automations_enabled", False):
            return "ERROR: Feature not available"
        if tool_name in _DEV_ONLY_TOOLS and not _dev_tools_active():
            # Unregistering a tool does not un-teach it: a model can
            # still emit a name it saw earlier in the conversation, or
            # in its own history. The gate has to hold at the door too.
            return ("ERROR: That is not available. To run this "
                    "automation for real, use automations__run_now.")
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
        return ToolResult(
            _as_json({"connectors": out}),
            display=("Looked at what you have connected · "
                     + _n_things(len(out), "account", "accounts")),
        )

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
        targets = []
        for i in items:
            if not isinstance(i, dict) or not i.get(src["id"]):
                continue
            row = {
                "kind": src["kind"],
                "id": str(i.get(src["id"]) or ""),
                "label": str(i.get(src["label"]) or i.get(src["id"]) or ""),
            }
            if connector_id == "slack":
                # R38 — rec1 f056: the provider row carries `is_member`
                # and this projection DROPPED it, so every public
                # channel in the workspace was offered as a place to
                # post and the agent named #general, a channel this
                # workspace never joined. The Slack manifest has no
                # chat:write.public, so posting to an un-joined channel
                # can never work — an un-joined channel is not a
                # target, and it is filtered out, not merely marked.
                if not bool(i.get("is_member")):
                    continue
                row["joined"] = True
            targets.append(row)
        return ToolResult(
            _as_json({"targets": targets[:100]}),
            display=("Looked at where it could write · "
                     + _n_things(len(targets[:100]), "place", "places")),
        )

    async def _list(self, args, ctx) -> str:
        from app.agent.automations.service import list_automations
        from app.db.database import async_session_maker
        async with async_session_maker() as db:
            rows = await list_automations(db, _uid(ctx))
        return ToolResult(
            _as_json({"automations": rows}),
            display=("Looked at your automations · "
                     + _n_things(len(rows), "automation", "automations")),
        )

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
        return ToolResult(
            _as_json({"templates": templates}),
            display=("Looked at what it could set up · "
                     + _n_things(len(templates), "option", "options")),
        )

    async def _recall(self, args, ctx) -> str:
        from app.db.database import async_session_maker
        from app.services import memory_v2_service

        async with async_session_maker() as db:
            result = await memory_v2_service.recall(
                db, user_id=_uid(ctx),
                query=args.get("query"),
                entity=args.get("entity"),
                category=args.get("category"),
                scope=args.get("scope"),
                since=args.get("since"),
            )
        if not result.get("facts") and not result.get("episodes"):
            # R31-28. This whole sentence used to BE the return value,
            # so a coaching line addressed to the model was rendered on
            # a founder's job sheet as what the tool "did". The
            # instruction is load-bearing — without it the model fills
            # the silence — so it stays, in the half only the model
            # reads. `display` is what a person sees.
            return ToolResult(
                "No memory matches that. Say so plainly; do not fill "
                "the gap with a guess.",
                display="Looked in memory · nothing matched",
            )
        return ToolResult(
            _as_json(result),
            display=_recall_summary(result),
        )

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
            return ToolResult(
                "SPEC INVALID:\n" + _as_json(e.errors),
                display="Could not set that up yet",
            )
        # ND-1 root cause: the flow stages the grant (step 5) before the
        # automation exists (step 6), so the grant row's automation_id
        # is NULL until the arm-time repair. Bind immediately so even a
        # never-armed draft's grants page is honest. Best-effort — arm
        # retries the repair anyway.
        try:
            from app.agent.automations import registry as reg
            spec = args.get("spec") or {}
            gids = [s.get("grant_id") for s in (spec.get("steps") or [])
                    if isinstance(s, dict) and s.get("grant_id")]
            if (spec.get("action") or {}).get("grant_id"):
                gids.append(spec["action"]["grant_id"])
            for gid in gids:
                await reg.bind_grant(_uid(ctx), grant_id=gid,
                                     automation_id=automation.id)
        except Exception:  # noqa: BLE001 — a repair, not a gate
            pass

        # §4.10 / R31-22: a setup typed in the MAIN CHAT posts ONE
        # `automation_setup` card there — and only that. The card is the
        # user's way back to the thread where the rest of the setup
        # happens; without it a conversation that created something has
        # nothing on screen pointing at what it created.
        #
        # `run_v3.notify_setup` has existed since R30 with ZERO callers,
        # which is why the founder's 11:51 "Set up the Replies drafted
        # before you wake automation for me." produced setup questions
        # in the main chat and no card and no automation. The other half
        # of that defect is B's fallback and is fixed there; this is the
        # half that was missing on the wire.
        #
        # Its own try/except: a missing card is a missing signpost, and
        # an automation that exists without one is better than a create
        # that raises after the row is committed.
        try:
            from app.agent.automations import ledger as _ledger
            from app.agent.automations import run_v3 as _run_v3
            from app.db.database import async_session_maker as _sm
            from app.db.models import Automation as _Automation
            async with _sm() as _db:
                _thread = await _ledger.ensure_thread(
                    _db, user_id=_uid(ctx), automation_id=automation.id,
                )
                _row = await _db.get(_Automation, automation.id)
                if _row is not None and _thread is not None:
                    await _run_v3.notify_setup(
                        _db, automation=_row, thread_id=_thread.id,
                    )
        except Exception as e:  # noqa: BLE001 — see above
            logger.warning(
                "[automations] setup card skipped for %s: %s",
                automation.id, e,
            )

        return ToolResult(
            f"Created automation {automation.id!r} ({automation.name}) "
            f"as a DRAFT. Call automations__arm when the user is "
            f"happy; offer automations__run_now afterwards if they "
            f"want to watch it work.",
            display=f"Set up {automation.name}",
        )

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
                    # R38: an agent edit stamps EDITED and broadcasts
                    # like every other write-back — one of the founder's
                    # two edits got a divider and the other did not.
                    edited_note=True,
                )
        except AutomationNotFound:
            return "ERROR: No such automation."
        except SpecError as e:
            return ToolResult(
                "SPEC INVALID:\n" + _as_json(e.errors),
                display="Could not set that up yet",
            )
        return ToolResult(
            f"Updated {automation.id!r}; status is now "
            f"{automation.status!r}.",
            display=f"Changed {automation.name}",
        )

    async def _set_destination(self, args, ctx) -> str:
        """R37 — the one-call write-back for WHERE the result goes.

        'chat' is a spec change with no permission attached, so it
        applies and (accounts permitting) ARMS in this one call.
        'connector' is two systems moving together: the write step is
        pinned to the target AND the permission that write needs is
        requested — as a card in the main chat, and as a needs_you
        turn in the automation's own thread, because a promise that
        "I will ask for your permission" which never surfaces anywhere
        the user is looking is the founder's item 7.
        """
        from app.agent.automations.service import (
            AutomationNotFound, set_destination_chat,
        )
        from app.agent.automations.spec import SpecError
        from app.db.database import async_session_maker

        uid = _uid(ctx)
        automation_id = (args.get("automation_id") or "").strip()
        destination = (args.get("destination") or "").strip()

        if destination == "chat":
            try:
                async with async_session_maker() as db:
                    out = await set_destination_chat(
                        db, automation_id=automation_id, user_id=uid,
                    )
            except AutomationNotFound:
                return "ERROR: No such automation."
            except SpecError as e:
                return ToolResult(
                    "COULD NOT CHANGE IT:\n" + _as_json(e.errors),
                    display="Could not move the delivery",
                )
            missing = [m for m in out.get("missing") or []]
            tail = (f" Its accounts are not all connected yet "
                    f"({', '.join(missing)}), so it stays a draft."
                    if missing else "")
            return ToolResult(
                f"DONE — {out['sentence']}{tail} Tell the user in one "
                f"short line; the workflow canvas already shows it.",
                display="Delivery moved to this chat",
            )

        if destination != "connector":
            return "ERROR: destination must be 'chat' or 'connector'."

        connector_id = (args.get("connector_id") or "").strip()
        target = args.get("target") or {}
        if not connector_id or not target.get("id"):
            return ("ERROR: a connector destination needs connector_id "
                    "and a target from automations__list_targets.")

        # R39: the whole flow (grant first — nothing half-pinned on a
        # refusal — then the step pin, the grant card and the thread
        # ask) lives in workflow.pin_write_destination, shared with the
        # canvas "+" so the agent's instruction and the user's tap do
        # the same thing.
        from app.agent.automations.service import _load_owned
        from app.agent.automations.workflow import pin_write_destination
        async with async_session_maker() as db:
            try:
                automation = await _load_owned(db, automation_id, uid)
            except AutomationNotFound:
                return "ERROR: No such automation."
            dest = await pin_write_destination(
                db, automation=automation, user_id=uid,
                connector_id=connector_id, target=target,
            )
        if dest is None:
            return (f"ERROR: this automation has no {connector_id} "
                    f"write to point anywhere.")
        if not dest.get("ok"):
            return ToolResult(
                "COULD NOT PIN IT:\n" + str(dest.get("sentence") or ""),
                display="Could not pin the destination",
            )
        label = target.get("label") or target.get("id")
        return ToolResult(
            f"PINNED to {label}, waiting on the user's permission "
            f"(grant_id={dest['grant_id']}, expires in 1 hour). The ask "
            f"is in this thread and in their chat. It arms itself when "
            f"they approve — end your turn telling them that in one "
            f"line.",
            display="Pinned — waiting on you",
        )

    # ------------------------------------------------------------------
    # R38 — the five edit tools, one door
    # ------------------------------------------------------------------

    async def _edit(
        self, args, ctx, *, intents: List[Dict[str, Any]],
        done: str, nothing: str,
    ) -> str:
        """Every `automations__edit_*` tool, through
        `workflow.apply_intents` — one policy, one set of writers, one
        undo, one EDITED note. See that function's docstring for why
        the tools do not get their own copy of any of it.
        """
        from app.agent.automations.service import (
            AutomationNotFound, _load_owned,
        )
        from app.agent.automations.workflow import apply_intents
        from app.db.database import async_session_maker

        automation_id = str(args.get("automation_id") or "").strip()
        if not automation_id:
            return "ERROR: automation_id is required."
        try:
            async with async_session_maker() as db:
                automation = await _load_owned(db, automation_id, _uid(ctx))
                out = await apply_intents(
                    db, automation=automation, user_id=_uid(ctx),
                    intents=intents,
                )
        except AutomationNotFound:
            return "ERROR: No such automation."

        applied = out.get("applied") or []
        payload = {
            "changed": [a.get("sentence") for a in applied],
            # Not applied, and WHY — the model has to be able to say
            # this out loud. A tool that answers an unmade change with
            # silence is the shape R38 exists to remove.
            "not_changed": list(out.get("refused") or [])
                           + ([out["answer"]] if out.get("answer") else []),
            "needs_your_approval": [
                n.get("sentence") for n in (out.get("needs") or [])
            ],
            "it_now_looks_like": _workflow_digest(out.get("workflow") or {}),
        }
        if payload["needs_your_approval"]:
            payload["next"] = (
                "Nothing about that part changed. Say the sentence in "
                "needs_your_approval and, if they want it, ask for it "
                "with automations__request_permission (a permission) or "
                "automations__request_connection (a new account)."
            )
        elif not applied:
            payload["next"] = (
                "NOTHING changed. Say the sentence in not_changed as "
                "your own words — do not report success."
            )
        return ToolResult(
            _as_json(payload),
            display=done if applied else nothing,
        )

    async def _edit_schedule(self, args, ctx) -> str:
        intent: Dict[str, Any] = {"kind": "schedule"}
        if args.get("preset_id"):
            intent["preset_id"] = str(args["preset_id"])
        elif args.get("time"):
            custom: Dict[str, Any] = {"time": str(args["time"])}
            if args.get("days"):
                custom["days"] = list(args["days"])
            if args.get("date"):
                custom["date"] = str(args["date"])
            intent["custom"] = custom
        else:
            return ("ERROR: Give either preset_id or time — a schedule "
                    "change needs a time to move to.")
        return await self._edit(
            args, ctx, intents=[intent],
            done="Changed when it runs", nothing="Left the time as it was",
        )

    async def _edit_rules(self, args, ctx) -> str:
        # A rule id OR its exact text: the model has the user's words
        # ("drop the no-thread rule") far more often than it has an id,
        # and a tool that can only be driven by an id it was never
        # given is a tool the model will not reach for.
        from app.agent.automations.service import (
            AutomationNotFound, _load_owned,
        )
        from app.agent.automations.workflow import rules_list
        from app.db.database import async_session_maker

        automation_id = str(args.get("automation_id") or "").strip()
        by_text: Dict[str, str] = {}
        if automation_id:
            try:
                async with async_session_maker() as db:
                    automation = await _load_owned(
                        db, automation_id, _uid(ctx),
                    )
                    by_text = {
                        " ".join(str(r.get("text") or "").split()).lower():
                            str(r.get("id") or "")
                        for r in rules_list(automation)
                    }
            except AutomationNotFound:
                return "ERROR: No such automation."

        def _rule_id(needle: str) -> str:
            key = " ".join(str(needle or "").split())
            return by_text.get(key.lower()) or key

        intents: List[Dict[str, Any]] = []
        for text in (args.get("add") or []):
            intents.append({"kind": "rule", "op": "add", "text": str(text)})
        for edit in (args.get("edit") or []):
            if not isinstance(edit, dict):
                continue
            intents.append({
                "kind": "rule", "op": "edit",
                "rule_id": _rule_id(edit.get("rule_id") or ""),
                "text": str(edit.get("text") or ""),
            })
        for needle in (args.get("remove") or []):
            intents.append({"kind": "rule", "op": "remove",
                            "rule_id": _rule_id(needle)})
        if not intents:
            return ("ERROR: Give at least one of add, remove or edit — "
                    "there is nothing here to change.")
        return await self._edit(
            args, ctx, intents=intents,
            done="Changed its rules", nothing="Left the rules alone",
        )

    async def _edit_steps(self, args, ctx) -> str:
        try:
            n = int(args.get("n"))
        except (TypeError, ValueError):
            return "ERROR: n must be the step's position, counting from 1."
        return await self._edit(
            args, ctx,
            intents=[{"kind": "step", "n": n,
                      "text": str(args.get("text") or "")}],
            done="Changed a step", nothing="Left the steps alone",
        )

    async def _edit_permissions(self, args, ctx) -> str:
        return await self._edit(
            args, ctx,
            intents=[{
                "kind": "permission",
                "account_id": str(args.get("account_id") or ""),
                "permission": str(args.get("permission") or ""),
                "direction": str(args.get("direction") or "revoke"),
            }],
            done="Changed what it may do",
            nothing="Left the permissions alone",
        )

    async def _edit_accounts(self, args, ctx) -> str:
        return await self._edit(
            args, ctx,
            intents=[{
                "kind": "account",
                "account_id": str(args.get("account_id") or ""),
                "direction": str(args.get("direction") or "remove"),
            }],
            done="Changed which accounts it uses",
            nothing="Left the accounts alone",
        )

    async def _test_run(self, args, ctx) -> str:
        """A rehearsal — reads for real, writes rendered and reported.

        The tool keeps its wire name because the array is prefix-stable
        (see `_all_tools`); what changed is that `service.rehearse`
        stages no outbox row, so there is nothing for the flush loop to
        send.
        """
        from app.agent.automations.service import (
            AutomationNotFound, rehearse,
        )
        from app.db.database import async_session_maker
        try:
            async with async_session_maker() as db:
                result = await rehearse(
                    db, automation_id=args.get("automation_id") or "",
                    user_id=_uid(ctx),
                )
        except AutomationNotFound:
            return "ERROR: No such automation."
        writes = result.get("writes") or []
        blocked = [w for w in writes if w.get("blocked")]
        payload = {
            "nothing_was_sent": True,
            "it_read": result.get("reads") or [],
            "it_would_write": writes,
            "sample_event": result.get("sample_event") or {},
            "next": (
                "Say what it WOULD write, in the user's words. Never "
                "say it ran, posted, sent or drafted anything — none "
                "of that happened. If a write carries `blocked`, say "
                "that reason instead: it is what stops the real run "
                "too."
            ),
        }
        if blocked:
            display = "Rehearsed it — something is in the way"
        elif writes:
            display = "Rehearsed it — nothing was sent"
        else:
            display = "Rehearsed it — it writes nothing"
        return ToolResult(_as_json(payload), display=display)

    async def _run_now(self, args, ctx) -> str:
        """R31-04 — the real re-run, through the engine.

        Calls the same entry point the app's Run it now button calls
        (`app.api.automations.run_now`), so the cadence counters, the
        dedupe namespace, the grant gate and the in-flight refusal are
        the ones surface already proves, rather than a second copy that
        can drift from them.

        The tool deliberately returns no run detail. The run narrates
        itself into the automation's thread; a model that summarises it
        here produces the second, contradictory account of the same run
        that the main chat filled up with.
        """
        from fastapi import HTTPException
        automation_id = str(args.get("automation_id") or "").strip()
        if not automation_id:
            return "ERROR: automation_id is required."
        from app.api.automations import run_now as run_now_route
        try:
            fired = await run_now_route(automation_id)
        except HTTPException as exc:
            detail = exc.detail
            code = detail.get("code") if isinstance(detail, dict) else ""
            sentence = (detail.get("sentence")
                        if isinstance(detail, dict) else str(detail))
            if exc.status_code == 404:
                return "ERROR: No such automation."
            if code == "already_running":
                return ToolResult(
                    f"Not started — it is already running. {sentence} "
                    "Tell the user it is running now; do not start a "
                    "second one.",
                    display="It is already running",
                )
            if isinstance(detail, dict) and detail.get("refusal_turn"):
                # R38: the route posted the refusal sentence into the
                # automation's OWN thread as an agent turn. Whether that
                # makes a reply a duplicate depends entirely on where this
                # turn is being answered.
                #
                # In the thread, the two land side by side and the repeat is
                # the double bubble rec1 recorded. ANYWHERE ELSE — the day
                # chat, WhatsApp, voice — the user cannot see that thread, so
                # "do not repeat it, otherwise add nothing" is an instruction
                # to answer "run my morning brief" with silence. The first
                # cut of this branch did exactly that.
                from app.agent.tool_executor import current_channel
                in_thread = (current_channel() or "") == "automation_thread"
                if in_thread:
                    return ToolResult(
                        f"Could not start it — {sentence} That sentence is "
                        "already posted in this thread; do not repeat it. "
                        "If you have a tool for the fix it names, do the "
                        "fix now (an unpinned destination is "
                        "automations__set_destination); otherwise add "
                        "nothing.",
                        display="Could not start the run",
                    )
                return ToolResult(
                    f"Could not start it — {sentence} Say that plainly "
                    "here, in your own words — the user is not looking at "
                    "the automation's thread. If you have a tool for the "
                    "fix it names, do the fix now (an unpinned destination "
                    "is automations__set_destination).",
                    display="Could not start the run",
                )
            return ToolResult(
                f"Could not start it — {sentence} Say that plainly; do "
                "not report a status instead.",
                display="Could not start the run",
            )
        # R36-2: the route DETACHES the run and answers "started"
        # immediately (holding the HTTP response open for minutes
        # taught the app's client to declare "Nothing ran" about a run
        # the server was mid-way through). The terminal branch below
        # stays for the day a caller gets a finished status back —
        # announcing "it is running" about a run that ended would be
        # the same lie in the other direction.
        status = str((fired or {}).get("status") or "").strip()
        if status in ("completed", "partial", "failed", "stopped_by_user"):
            return ToolResult(
                f"The run finished ({status}). It has already written "
                "itself into the automation's own thread — say in one "
                "short line that it ran and point them there. Do not "
                "summarise what it found; the thread has the real "
                "account of it and yours would be a second one.",
                display="Ran it",
            )
        return ToolResult(
            "Started a real run. It reports itself in the automation's "
            "own thread — reply with one short line saying it is "
            "running, and nothing about what it will do, what it found, "
            "or what its status was before.",
            display="Started the run",
        )

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
        # R31-28: a user reads these on a step row. `OK — status is now
        # 'armed'` is a wire value in quotes; the display half says what
        # happened to their automation.
        _DONE = {
            "arm": "Turned it on",
            "pause": "Paused it",
            "resume": "Turned it back on",
        }
        if verb == "delete":
            return ToolResult("Deleted.", display="Deleted it")
        return ToolResult(
            f"OK — status is now {out.status!r}.",
            display=_DONE.get(verb, "Updated it"),
        )

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
