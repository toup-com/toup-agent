"""R38 — the automation engine, exposed on the platform's MCP server.

Everything the workflow engine can do already exists as an HTTP route on
the tenant agent (`backend/app/api/automations.py`) or as a platform RPC
(`automations_platform.py`). Until now the only clients were the web
canvas, the phone, and the tenant's own in-process skill. This module
adds the third door: **any MCP client holding the user's `X-Agent-Key`**
— Claude Code, another agent, an operator's script.

Modelled on `connector_mcp.py`, which is the pattern of record for a
platform tool group:

  - one `FunctionTool` per operation, registered at boot on the module-
    level `mcp` server, tagged so it is identifiable in a listing;
  - a `Middleware.on_list_tools` that trims the group per requesting
    user, so `tools/list` never advertises a surface the caller cannot
    reach;
  - `MCPAuthMiddleware`'s ContextVars (`get_mcp_user_id`) as the ONLY
    identity source. A handler never takes a `user_id` argument — an
    argument is something the model can invent.

Three decisions worth reading before you edit this file.

**1. The namespace is `workflow__`, and it may never be `automations__`.**
The tenant agent's own skill owns `automations__*` (21 tools today,
`skills/builtins/automations/skill.py`), and `agent_runner.tool_defs`
builds the wire array as `core + skill_defs + mcp_defs` with **no
dedupe**. Registering `automations__create` here would hand the provider
two tools with one name on every turn of every tenant that has connector
dispatch on — an immediate 400, for every user, from a change that looks
purely additive. `register_automation_tools` refuses to register at all
when the namespace collides with a builtin skill directory, because a
warning that boots anyway is not a guard against that.

**2. The group is capped at `MAX_GROUP_TOOLS` and asserted, not hoped.**
`llm_proxy._cap_tools` gives every namespace ≥1 tool and then drops from
the LARGEST namespace first — so an unbounded group here does not lose
its own tail, it starts evicting somebody else's tools once it becomes
the biggest. Twelve is the budget. Adding a thirteenth is a decision
someone has to make on purpose, which is what the assertion is for.

**3. Every handler answers with an envelope, never an exception.**
`{"ok": True, "result": …}` or `{"ok": False, "code": …, "sentence": …}`.
A raised exception reaches the model as a stringified traceback it has to
guess at; a code plus a sentence is something it can act on and repeat to
the user. In particular the run-now refusals (`waiting_on_you`,
`needs_setup`, `already_running`, `grant_pending`) are the whole point of
that route and arrive as `409`s — passed through verbatim rather than
flattened into "it failed".
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import httpx

from app.db.database import async_session_maker
from app.mcp_auth import get_mcp_user_id, try_get_mcp_user_id

logger = logging.getLogger(__name__)


#: Tag applied to every tool in this group. `on_list_tools` filters by
#: NAME PREFIX rather than by this tag — a listed `Tool` is not
#: guaranteed to carry the tags of the `FunctionTool` it was built from,
#: and the prefix is structural. The tag is for humans reading a listing.
AUTOMATION_TOOL_TAG = "automation"

#: The wire namespace. See decision 1 in the module docstring.
AUTOMATION_NAMESPACE = "workflow"

#: See decision 2. Raised deliberately, never drifted into.
MAX_GROUP_TOOLS = 12

#: The authoring guide `workflow__guide` serves, under `docs/skills/`.
#: Loaded through `app.support.skills_index`, i.e. the same loader the
#: maintenance agent uses — so there is one definition of "the skill is
#: installed" and one place it can be absent.
GUIDE_SKILL_NAME = "automation-workflows"

_AGENT_CALL_TIMEOUT_S = 30.0

#: The tenant agent's own `_flag_or_404` answers this when ITS
#: `automations_enabled` is off. `automations_proxy` already documents
#: why that must not be relayed as a 404: once the platform gate has
#: passed, an agent 404 means "this container predates the launch image"
#: — temporary and self-healing — not "the feature is off for you".
_AGENT_DARK_DETAIL = "Feature not available"


# ── The one door to the tenant ───────────────────────────────────────


async def _flag_enabled(user_id: str) -> bool:
    from app.services import feature_flags
    async with async_session_maker() as db:
        return await feature_flags.is_enabled(db, "automations", user_id)


async def _agent_target(user_id: str) -> Optional[tuple[str, str]]:
    from sqlalchemy import select
    from app.db.models import AgentConfig
    async with async_session_maker() as db:
        row = (await db.execute(
            select(AgentConfig.agent_url, AgentConfig.agent_api_key).where(
                AgentConfig.user_id == user_id,
                AgentConfig.deploy_status == "active",
            )
        )).first()
    if row and row.agent_url and row.agent_api_key:
        return (row.agent_url, row.agent_api_key)
    return None


def _fail(code: str, sentence: str, **extra: Any) -> dict:
    return {"ok": False, "code": code, "sentence": sentence, **extra}


async def _agent_call(
    user_id: str, method: str, path: str, *,
    params: Optional[dict] = None, body: Optional[dict] = None,
) -> dict:
    """One call to this user's own `/api/automations{path}`.

    Returns an envelope for every outcome including the failures. The
    status codes that carry meaning are kept as meaning:

      409  the engine's refusals — `{code, sentence}` straight through.
           `run-now` alone has six of them and they are the reason a
           caller can tell "it is already running" from "it needs a
           grant" without reading a log.
      404  ambiguous by construction, so it is split here. A body of
           `{"detail": "Feature not available"}` is the tenant engine
           being behind (`agent_starting`, retryable); anything else is
           a genuine not-found.
    """
    if not await _flag_enabled(user_id):
        return _fail(
            "not_enabled",
            "Automations are not switched on for this account.",
        )
    target = await _agent_target(user_id)
    if target is None:
        return _fail(
            "no_agent",
            "This account has no running agent, so its automations "
            "cannot be reached.",
        )
    agent_url, agent_api_key = target
    url = f"{agent_url.rstrip('/')}/api/automations{path}"
    from app.services.agent_http import get_agent_http_client
    try:
        resp = await get_agent_http_client().request(
            method.upper(), url,
            params=params or None,
            json=body if body is not None else None,
            headers={
                "X-Agent-Key": agent_api_key,
                "accept": "application/json",
            },
            timeout=_AGENT_CALL_TIMEOUT_S,
        )
    except httpx.RequestError as e:
        logger.warning("[automations_mcp] %s %s failed: %s", method, path, e)
        return _fail(
            "unreachable",
            "Your agent did not answer, so nothing was read or changed.",
            retryable=True,
        )

    try:
        payload: Any = resp.json()
    except ValueError:
        payload = None

    if 200 <= resp.status_code < 300:
        return {"ok": True, "result": payload}

    detail = payload.get("detail") if isinstance(payload, dict) else None

    if resp.status_code == 404 and detail == _AGENT_DARK_DETAIL:
        return _fail(
            "agent_starting",
            "Your agent is still coming up to the version that runs "
            "automations. Nothing was read or changed — try again in a "
            "couple of minutes.",
            retryable=True,
        )
    if resp.status_code == 404:
        return _fail("not_found", _sentence_of(detail, "No such automation."))
    if resp.status_code in (403, 409) and isinstance(detail, dict) \
            and detail.get("code"):
        # The engine's own refusal, verbatim: `{code, sentence, …extra}`.
        # A `sentence` floor goes UNDER it rather than over — several
        # refusals are `{"code": …}` alone (MembershipError, the mode
        # routes), and a code with no words is something the caller has
        # to invent wording for, which is how a user gets told the wrong
        # reason. The engine's own sentence still wins when it has one.
        return {
            "ok": False,
            "sentence": f"The engine refused that: {detail['code']}.",
            **detail,
        }
    if resp.status_code == 409:
        return _fail("conflict", _sentence_of(detail, "That could not be applied."))
    if resp.status_code == 422:
        return _fail(
            "invalid",
            "The engine rejected that shape before saving anything.",
            detail=detail,
        )
    return _fail(
        f"http_{resp.status_code}",
        _sentence_of(detail, "The engine refused that."),
    )


def _sentence_of(detail: Any, fallback: str) -> str:
    if isinstance(detail, str) and detail.strip():
        return detail.strip()
    if isinstance(detail, dict):
        s = detail.get("sentence")
        if isinstance(s, str) and s.strip():
            return s.strip()
    return fallback


# ── Handlers ─────────────────────────────────────────────────────────
#
# Each is `async def (**kwargs) -> dict` — the shape FastMCP invokes with
# the client's arguments — and resolves its own user from the auth
# ContextVar. `get_mcp_user_id()` raises `ValueError` when nothing is
# bound, which FastMCP serialises as a structured tool error; that is the
# correct answer for an unauthenticated call and matches every other
# platform tool in `mcp_server.py`.


async def _h_guide(**kw) -> dict:
    """`workflow__guide` — read the authoring guide off disk, now.

    The one handler that reads no user: the guide is the same document
    for everybody, so asking for an identity would be ceremony. The
    transport still gates it — `mcp_require_x_agent_key` is on, so an
    unauthenticated caller never reaches a handler at all.

    Deliberately NOT a baked-in constant. A markdown-only change to
    `docs/skills/automation-workflows/SKILL.md` has to reach the running
    server, or the guide drifts from the engine silently — which is the
    exact way the Round 22 app-builder skill edit deployed nowhere with
    every CI signal green. Reading at call time makes the file the
    artifact; `test_r38_automations_mcp.py` proves the file is in the
    image's build context, and `skills_index` is the loader.
    """
    from app.support import skills_index

    md = skills_index.load_skill(GUIDE_SKILL_NAME)
    if md is None:
        # NEVER an empty guide. A caller that receives "" writes an
        # automation from memory and is confidently wrong.
        return _fail(
            "guide_missing",
            f"The authoring guide is not installed on this server "
            f"(expected {skills_index.skills_dir()}/{GUIDE_SKILL_NAME}"
            f"/SKILL.md). Do not author a spec without it.",
        )
    sections = _section_titles(md)
    wanted = str(kw.get("section") or "").strip()
    if wanted:
        body = _extract_section(md, wanted)
        if body is None:
            return _fail(
                "no_such_section",
                f"The guide has no section {wanted!r}.",
                sections=sections,
            )
        return {"ok": True, "result": {"section": wanted, "sections": sections,
                                       "markdown": body}}
    return {"ok": True, "result": {"sections": sections, "markdown": md}}


def _section_titles(md: str) -> list[str]:
    return [ln.lstrip("# ").strip()
            for ln in md.splitlines() if ln.startswith("## ")]


def _extract_section(md: str, wanted: str) -> Optional[str]:
    """The `## …` block whose title contains `wanted`, case-insensitive.

    Matching on a substring rather than an exact title is deliberate: a
    caller asking for "grammar" should not have to reproduce
    "3. The spec, node by node" byte for byte.
    """
    needle = wanted.lower()
    lines = md.splitlines()
    start = None
    for i, ln in enumerate(lines):
        if ln.startswith("## ") and needle in ln.lower():
            start = i
            break
    if start is None:
        return None
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("## "):
            end = j
            break
    return "\n".join(lines[start:end]).strip()


async def _h_registry(**kw) -> dict:
    """`workflow__registry` — platform-native; no tenant round trip.

    The capability metadata lives in the connector registry on THIS
    process, and the connection state in the platform's vault. The
    tenant's `automations__get_registry` fetches both over HTTP from
    here; an MCP client is already here.
    """
    import json
    user_id = get_mcp_user_id()
    if not await _flag_enabled(user_id):
        return _fail("not_enabled",
                     "Automations are not switched on for this account.")

    from app.services.connector_registry import get_registry
    from app.services import connector_vault as vault

    capability = {
        e.get("connector_id"): e
        for e in (get_registry().automation_registry() or [])
        if e.get("connector_id")
    }
    async with async_session_maker() as db:
        identities = await vault.list_active(db, user_id)
    state: dict[str, dict] = {}
    for ident in identities:
        try:
            scopes = json.loads(ident.scopes_json) if ident.scopes_json else []
        except (ValueError, TypeError):
            scopes = []
        state[ident.connector_id] = {
            "connected": ident.status == "active",
            "status": ident.status,
            "scopes": scopes,
            "account": ident.provider_account_id or None,
        }

    out = []
    for cid, cap in sorted(capability.items()):
        conn = state.get(cid) or {}
        out.append({
            "connector_id": cid,
            "name": cap.get("name"),
            "connected": bool(conn.get("connected")),
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
    return {"ok": True, "result": {"connectors": out}}


async def _h_templates(**kw) -> dict:
    """`workflow__templates` — platform-native (the catalog is a platform
    table; the tenant reads it over the same RPC)."""
    from sqlalchemy import select
    from app.db.models.platform_automation import AutomationTemplate
    from app.services.automation_template_catalog import template_payload

    user_id = get_mcp_user_id()
    if not await _flag_enabled(user_id):
        return _fail("not_enabled",
                     "Automations are not switched on for this account.")
    async with async_session_maker() as db:
        rows = (await db.execute(
            select(AutomationTemplate)
            .where(AutomationTemplate.enabled.is_(True))
            .order_by(AutomationTemplate.sort_order, AutomationTemplate.name)
        )).scalars().all()
    return {"ok": True,
            "result": {"templates": [template_payload(t) for t in rows]}}


async def _h_list(**kw) -> dict:
    return await _agent_call(get_mcp_user_id(), "GET", "")


async def _h_get(**kw) -> dict:
    """`workflow__get` — one automation, and everything a caller needs to
    decide what to change: the saved spec, the canvas the user sees, and
    the recent runs.

    Three reads rather than one because they are three routes; a caller
    that has to make three calls to answer "is this working" makes two
    of them and guesses at the third.
    """
    user_id = get_mcp_user_id()
    aid = str(kw.get("automation_id") or "").strip()
    if not aid:
        return _fail("bad_request", "automation_id is required.")

    listing = await _agent_call(user_id, "GET", "")
    if not listing.get("ok"):
        return listing
    rows = (listing.get("result") or {}).get("automations") or []
    mine = next((r for r in rows if r.get("id") == aid), None)
    if mine is None:
        return _fail("not_found", "No such automation.")

    canvas = await _agent_call(user_id, "GET", f"/{aid}/workflow")
    runs = await _agent_call(user_id, "GET", f"/{aid}/runs",
                             params={"limit": 10})
    return {"ok": True, "result": {
        "automation": mine,
        # A failed sub-read is reported, never silently omitted — an
        # absent canvas and an empty canvas mean opposite things.
        "workflow": canvas.get("result") if canvas.get("ok") else None,
        "workflow_error": None if canvas.get("ok") else canvas,
        "runs": (runs.get("result") or {}).get("runs") if runs.get("ok") else None,
        "runs_error": None if runs.get("ok") else runs,
    }}


async def _h_create(**kw) -> dict:
    body: dict = {"spec": kw.get("spec")}
    if kw.get("template_slug"):
        body["template_slug"] = kw["template_slug"]
    return await _agent_call(get_mcp_user_id(), "POST", "", body=body)


async def _h_update(**kw) -> dict:
    aid = str(kw.get("automation_id") or "").strip()
    if not aid:
        return _fail("bad_request", "automation_id is required.")
    return await _agent_call(
        get_mcp_user_id(), "PATCH", f"/{aid}", body={"spec": kw.get("spec")},
    )


async def _h_change(**kw) -> dict:
    """`workflow__change` — all five change kinds, in one transaction.

    `POST /{id}/workflow/commit` is the only door that applies rules,
    steps, schedule, permissions and accounts together and refuses as a
    unit; the per-sheet routes each apply immediately. Sending
    `workflow_rev` (from `workflow__get`) makes the write optimistic-
    concurrency safe: a stale rev comes back `409 stale` having applied
    NOTHING, with the current workflow attached to re-base against.
    """
    aid = str(kw.get("automation_id") or "").strip()
    if not aid:
        return _fail("bad_request", "automation_id is required.")
    kinds = ("schedule", "permissions", "steps", "rules", "accounts")
    body = {k: kw[k] for k in ("workflow_rev", *kinds)
            if kw.get(k) is not None}
    if not any(k in body for k in kinds):
        return _fail(
            "nothing_to_change",
            "Name at least one of schedule, steps, rules, permissions "
            "or accounts.",
        )
    return await _agent_call(
        get_mcp_user_id(), "POST", f"/{aid}/workflow/commit", body=body,
    )


_LIFECYCLE_ACTIONS = ("arm", "pause", "resume")


async def _h_lifecycle(**kw) -> dict:
    aid = str(kw.get("automation_id") or "").strip()
    action = str(kw.get("action") or "").strip().lower()
    if not aid:
        return _fail("bad_request", "automation_id is required.")
    if action not in _LIFECYCLE_ACTIONS:
        return _fail(
            "bad_request",
            f"action must be one of {', '.join(_LIFECYCLE_ACTIONS)}.",
        )
    return await _agent_call(get_mcp_user_id(), "POST", f"/{aid}/{action}")


async def _h_delete(**kw) -> dict:
    """The delete is SOFT — the schedule is disarmed and the thread is
    archived for 30 days — but there is no un-delete route, so from the
    user's side it is gone. Hence the explicit `confirm`: `delete` living
    in the same tool group as `pause` is one wrong token away from
    ending work the user asked to keep."""
    aid = str(kw.get("automation_id") or "").strip()
    if not aid:
        return _fail("bad_request", "automation_id is required.")
    if kw.get("confirm") is not True:
        return _fail(
            "confirm_required",
            "Deleting an automation cannot be undone. Call again with "
            "confirm=true once the user has said to delete it.",
        )
    return await _agent_call(get_mcp_user_id(), "DELETE", f"/{aid}")


async def _h_run(**kw) -> dict:
    """`workflow__run` — a REAL run. Reads live data and sends the writes
    the automation is armed to send, after its own 6-second undo window.

    The refusals come back as codes because they are each a different
    fix: `needs_setup` (something is unpinned or ungranted),
    `waiting_on_you` (a run is parked on a question), `already_running`,
    `grant_pending` / `grant_expired`, `no_source`, `v1_not_supported`.
    """
    aid = str(kw.get("automation_id") or "").strip()
    if not aid:
        return _fail("bad_request", "automation_id is required.")
    return await _agent_call(get_mcp_user_id(), "POST", f"/{aid}/run-now")


async def _h_test(**kw) -> dict:
    """`workflow__test` — a rehearsal: the reads run against live data,
    the writes are rendered and reported, nothing is staged and nothing
    is sent.

    Off by default. The route is gated on the tenant's
    `automations_dev_tools`, and answers `rehearsal_disabled` when it is
    off — a named refusal, not a 404, so a caller can tell it apart from
    a missing automation and say the true thing to the user.
    """
    aid = str(kw.get("automation_id") or "").strip()
    if not aid:
        return _fail("bad_request", "automation_id is required.")
    return await _agent_call(get_mcp_user_id(), "POST", f"/{aid}/test-run")


# ── Tool definitions ─────────────────────────────────────────────────
#
# Order is the registration order, and registration order is the wire
# order. New tools JOIN AT THE END — the same prefix-stability rule the
# skill loader enforces on `automations__*`, for the same reason: a
# provider's prompt cache keys on the tools array prefix, so inserting in
# the middle invalidates every cached turn for every tenant.

_TOOLS: list[tuple[str, str, dict, Any]] = [
    (
        "guide",
        "READ THIS FIRST, before writing or editing any automation spec. "
        "The authoring guide: the spec grammar node by node, worked "
        "examples, the validate-before-save loop, and the rules that are "
        "not negotiable. Optionally pass `section` to read one part.",
        {"type": "object", "properties": {
            "section": {"type": "string",
                        "description": "Substring of a section heading, "
                                       "e.g. 'grammar' or 'examples'."},
        }},
        _h_guide,
    ),
    (
        "registry",
        "What can be automated for THIS user: per-connector events "
        "(push/poll and the poll floor), write actions with the parameter "
        "that pins their target, rate budgets, and which accounts are "
        "actually connected right now. A connector absent here cannot be "
        "used, and an event not listed for a connector does not exist.",
        {"type": "object", "properties": {}},
        _h_registry,
    ),
    (
        "templates",
        "The catalog of ready-made automations. Each carries a spec you "
        "can adapt; starting from one is cheaper and safer than authoring "
        "a spec from scratch.",
        {"type": "object", "properties": {}},
        _h_templates,
    ),
    (
        "list",
        "Every automation this user has, with its status (draft, armed, "
        "paused), its schedule and its last outcome.",
        {"type": "object", "properties": {}},
        _h_list,
    ),
    (
        "get",
        "One automation in full: the saved spec, the canvas the user sees "
        "(schedule, accounts and their permissions, steps, rules, output) "
        "and its ten most recent runs. `workflow.workflow_rev` from here "
        "is what workflow__change needs.",
        {"type": "object", "properties": {
            "automation_id": {"type": "string"},
        }, "required": ["automation_id"]},
        _h_get,
    ),
    (
        "create",
        "Validate a spec and save it as a DRAFT. Validation happens before "
        "anything is written, so a rejection costs nothing and names the "
        "node it rejected. A draft never fires — arm it when the user says "
        "so. Write steps need a grant first (see the guide).",
        {"type": "object", "properties": {
            "spec": {"type": "object",
                     "description": "AutomationSpec v2. Read "
                                    "workflow__guide section 3 for the "
                                    "grammar; do not guess it."},
            "template_slug": {"type": "string",
                              "description": "The template this spec was "
                                             "adapted from, if any."},
        }, "required": ["spec"]},
        _h_create,
    ),
    (
        "update",
        "Replace an automation's whole spec and re-compile it. Grants "
        "already approved for a step carry forward when the connector and "
        "tool are unchanged. An armed automation drops to draft and "
        "re-arms itself; the thread records that it was edited.",
        {"type": "object", "properties": {
            "automation_id": {"type": "string"},
            "spec": {"type": "object",
                     "description": "The COMPLETE replacement spec, not a "
                                    "patch."},
        }, "required": ["automation_id", "spec"]},
        _h_update,
    ),
    (
        "change",
        "Change one live automation without rewriting its spec — all five "
        "change kinds, applied as ONE transaction: schedule, steps, rules, "
        "an account's permissions, and adding or removing an account. Pass "
        "`workflow_rev` from workflow__get; a stale rev is refused with "
        "'stale' having applied nothing.",
        {"type": "object", "properties": {
            "automation_id": {"type": "string"},
            "workflow_rev": {"type": "integer",
                             "description": "From workflow__get. Omitting "
                                            "it skips the staleness check."},
            "schedule": {"type": "object",
                         "description": "{preset_id} or {custom:{time:'HH:MM',"
                                        " days:[1-7 ISO], date?, tz?}}."},
            "steps": {"type": "array", "items": {"type": "object"},
                      "description": "Up to 8 human step rows "
                                     "[{n, text}] — recompiled by the "
                                     "engine, so it may come back pending."},
            "rules": {"type": "object",
                      "description": "{add:[text], remove:[rule_id], "
                                     "edit:[{id,text}]}."},
            "permissions": {"type": "array", "items": {"type": "object"},
                            "description": "[{account_id, can:[id], "
                                           "cant:[id]}] — ids from "
                                           "workflow__get's accounts."},
            "accounts": {"type": "object",
                         "description": "{add:[connector_id], "
                                        "remove:[connector_id]}."},
        }, "required": ["automation_id"]},
        _h_change,
    ),
    (
        "lifecycle",
        "Arm, pause or resume an automation. Arming is what makes it fire; "
        "it verifies every write grant first and refuses if one is missing "
        "or expired.",
        {"type": "object", "properties": {
            "automation_id": {"type": "string"},
            "action": {"type": "string", "enum": list(_LIFECYCLE_ACTIONS)},
        }, "required": ["automation_id", "action"]},
        _h_lifecycle,
    ),
    (
        "delete",
        "Delete an automation: it stops firing and its thread is archived "
        "for 30 days. There is no un-delete, so this needs confirm=true "
        "and the user has to have asked for it.",
        {"type": "object", "properties": {
            "automation_id": {"type": "string"},
            "confirm": {"type": "boolean"},
        }, "required": ["automation_id", "confirm"]},
        _h_delete,
    ),
    (
        "run",
        "Run it for real, now: live reads, and the writes it is armed to "
        "send. Returns immediately — the run continues in the background, "
        "so read workflow__get afterwards for the outcome. Refusals come "
        "back as a code and a sentence rather than an error.",
        {"type": "object", "properties": {
            "automation_id": {"type": "string"},
        }, "required": ["automation_id"]},
        _h_run,
    ),
    (
        "test",
        "Rehearse it: the reads run against live data, every write is "
        "rendered and reported, and NOTHING is sent. Off unless the tenant "
        "has enabled it — you get 'rehearsal_disabled' back, in which case "
        "validate with workflow__create and inspect with workflow__get "
        "instead.",
        {"type": "object", "properties": {
            "automation_id": {"type": "string"},
        }, "required": ["automation_id"]},
        _h_test,
    ),
]


def tool_names() -> list[str]:
    """The group's wire names, in registration order."""
    return [f"{AUTOMATION_NAMESPACE}__{short}" for short, _d, _s, _h in _TOOLS]


def is_automation_tool(name: str) -> bool:
    return name.startswith(f"{AUTOMATION_NAMESPACE}__")


# ── Registration ─────────────────────────────────────────────────────


def register_automation_tools(
    mcp_server, *, skill_prefixes: Optional[set[str]] = None,
) -> int:
    """Register the group on `mcp_server`. Returns the count registered.

    Refuses outright — registering NOTHING and logging an error — when
    `AUTOMATION_NAMESPACE` collides with a builtin skill directory. See
    decision 1 in the module docstring: a collision is not a cosmetic
    shadow here, it is two identically-named tools in one wire array for
    every tenant with connector dispatch on, i.e. a provider 400 on every
    turn. Half a group is worse than none, so the refusal is total.
    """
    from fastmcp.tools import FunctionTool

    prefixes = skill_prefixes or set()
    if AUTOMATION_NAMESPACE in prefixes:
        logger.error(
            "[automations_mcp] namespace %r collides with a builtin skill "
            "— registering NOTHING. The agent concatenates skill tools and "
            "MCP tools with no dedupe, so a collision is a duplicate tool "
            "name on every turn. Rename the namespace.",
            AUTOMATION_NAMESPACE,
        )
        return 0

    if len(_TOOLS) > MAX_GROUP_TOOLS:
        # Not a warning. `_cap_tools` drops from the largest namespace
        # first, so an oversized group evicts other people's tools.
        raise RuntimeError(
            f"[automations_mcp] {len(_TOOLS)} tools exceeds the "
            f"{MAX_GROUP_TOOLS}-tool budget for one namespace"
        )

    registered = 0
    for short, description, schema, handler in _TOOLS:
        full = f"{AUTOMATION_NAMESPACE}__{short}"
        handler.__name__ = full
        handler.__qualname__ = full
        tool = FunctionTool(
            name=full,
            description=description,
            parameters=schema,
            tags={AUTOMATION_TOOL_TAG},
            fn=handler,
            enabled=True,
        )
        try:
            mcp_server.add_tool(tool)
        except Exception as e:  # noqa: BLE001 — reload path, same as T1f
            logger.warning(
                "[automations_mcp] add_tool(%r) raised %s — already "
                "registered? continuing", full, e,
            )
            continue
        registered += 1
    logger.info("[automations_mcp] %d automation tool(s) registered",
                registered)
    return registered


try:
    # Must inherit FastMCP's Middleware base or `_build_chain` does
    # `partial(middleware, call_next=…)` on a non-callable instance and
    # raises `TypeError: the first argument must be callable` — which
    # reaches the client as an McpError and takes EVERY tools/list down
    # with it, connector tools included. T1f learned this in production;
    # this is the same guard, not a new one.
    from fastmcp.server.middleware import Middleware as _FastMCPMiddleware
except ImportError:  # pragma: no cover — production pins fastmcp 2.11+
    _FastMCPMiddleware = object  # type: ignore[assignment,misc]


class AutomationToolFilterMiddleware(_FastMCPMiddleware):
    """Trim the automation group out of `tools/list` for a user who does
    not have automations.

    Same shape and the same reasoning as `ConnectorToolFilterMiddleware`:
    the surface a caller is shown must be the surface a caller can reach.
    A tool that lists and then answers `not_enabled` on every call is a
    tool that lies about existing.

    Warn-only (nothing bound by `MCPAuthMiddleware`) passes the list
    through unfiltered, matching every other filter here — the handler
    raises on invocation instead, which is a louder and more accurate
    failure than a silently short list.

    Filtering is by NAME PREFIX, not by tag: `on_list_tools` receives
    whatever the server built, and a listed `Tool` carrying the tags of
    the `FunctionTool` it came from is an implementation detail. The
    prefix is the contract.
    """

    async def on_list_tools(self, context, call_next):
        tools = await call_next(context)

        user_id = try_get_mcp_user_id()
        if user_id is None:
            return tools

        try:
            enabled = await _flag_enabled(user_id)
        except Exception as e:  # noqa: BLE001
            # A flag read that fails must not delete the group — that
            # would look exactly like "the feature was turned off".
            logger.warning(
                "[automations_mcp] flag read failed for tools/list (%s) — "
                "leaving the group listed", e,
            )
            return tools
        if enabled:
            return tools

        return [t for t in tools if not is_automation_tool(t.name)]


def deregister_automation_tools_for_tests(mcp_server) -> None:
    """Tests only — production never deregisters at runtime."""
    for name in tool_names():
        try:
            mcp_server.remove_tool(name)
        except Exception:
            pass
