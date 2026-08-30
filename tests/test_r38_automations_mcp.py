"""R38 — the automation engine on MCP, and the authoring skill it teaches.

Two halves, and the second one is the reason this file exists at all.

**The tool group** (`app/services/automations_mcp.py`): registration,
the namespace invariant that keeps it from colliding with the tenant's
own `automations__*` skill, the per-user flag filter on `tools/list`,
and the envelope every handler answers with — including the four
failure translations that are the difference between a caller who knows
what to do next and one who sees "it failed".

**The guide actually being installed.** `docs/skills/` is read at
runtime by two things (`app/support/skills_index.py` for the maintenance
router, and `workflow__guide` for MCP clients) and was copied into NO
image: the platform Dockerfile does `COPY backend/ ./backend/` and
nothing else, so `skills_dir()` resolved to a directory that did not
exist. That is the same shape as the Round 22 app-builder skill edit
that "shipped" with every CI signal green. Three tests cover it: the
index finds the skill, the tool serves the bytes ON DISK (proved by
pointing it at a modified copy — a baked-in constant fails that), and
the image's build context carries the directory to the exact path the
loader looks in.
"""

from __future__ import annotations

import os
import pathlib
import re

import pytest

os.environ.setdefault("ENVIRONMENT", "development")

from app.services import automations_mcp as amcp  # noqa: E402
from app.support import skills_index  # noqa: E402

BACKEND = pathlib.Path(__file__).resolve().parents[1]
REPO = BACKEND.parent
GUIDE = REPO / "docs" / "skills" / amcp.GUIDE_SKILL_NAME / "SKILL.md"


# ══════════════════════════════════════════════════════════════════════
# 1. The skill is installed — index, router, and the file on disk
# ══════════════════════════════════════════════════════════════════════


def test_the_authoring_skill_is_in_the_skills_index():
    """`skills_index.list_skills()` is the platform's definition of "a
    skill exists". A SKILL.md the index cannot see is a file, not a
    skill."""
    skills_index.refresh()
    by_name = {s.name: s for s in skills_index.list_skills()}
    assert amcp.GUIDE_SKILL_NAME in by_name, (
        f"{amcp.GUIDE_SKILL_NAME} is not in the skills index — check the "
        f"frontmatter `name:` matches the directory name"
    )
    entry = by_name[amcp.GUIDE_SKILL_NAME]
    assert entry.description.strip(), "a skill with no description cannot be routed to"
    assert entry.path.endswith(f"{amcp.GUIDE_SKILL_NAME}/SKILL.md")


def test_the_master_router_routes_to_the_authoring_skill():
    """The router table is the curated symptom→skill map and the
    strongest signal `rank_subsystems` has. A skill absent from it is
    reachable only by description overlap."""
    skills_index.refresh()
    rows = [r for r in skills_index.parse_router_table()
            if r.skill == amcp.GUIDE_SKILL_NAME]
    assert rows, "no master-router row points at the authoring skill"

    ranked = skills_index.rank_subsystems(
        "an automation never fires and re-files the same item on every poll",
        top_n=3,
    )
    assert ranked and ranked[0].name == amcp.GUIDE_SKILL_NAME


def test_every_router_row_names_a_skill_that_exists():
    """Guards the rows added for this skill the same way
    test_support_agent does for the rest: a typo'd link is a dead route
    that ranks nothing and reports nothing."""
    skills_index.refresh()
    known = skills_index.skill_names()
    assert all(r.skill in known for r in skills_index.parse_router_table())


def test_the_guide_has_a_section_six_so_the_support_router_can_read_it():
    body = skills_index.failure_modes(amcp.GUIDE_SKILL_NAME)
    assert body and len(body) > 500, (
        "section 6 is the per-symptom router every skill here exposes"
    )


@pytest.mark.asyncio
async def test_the_guide_tool_serves_the_file_on_disk():
    out = await amcp._h_guide()
    assert out["ok"] is True
    assert out["result"]["markdown"] == GUIDE.read_text(encoding="utf-8")
    assert out["result"]["sections"], "the guide's ## headings drive `section`"


@pytest.mark.asyncio
async def test_the_guide_tool_reads_disk_rather_than_a_baked_constant(tmp_path):
    """THE falsifier for "a markdown-only change deployed nowhere".

    Point the loader at a modified copy. A tool that had inlined the
    guide (or cached it at import) returns the original and passes every
    other test in this file while being exactly the bug.
    """
    from app.config import settings

    fake_skills = tmp_path / "docs" / "skills" / amcp.GUIDE_SKILL_NAME
    fake_skills.mkdir(parents=True)
    sentinel = "SENTINEL-2b41f9-only-in-the-modified-copy"
    (fake_skills / "SKILL.md").write_text(
        f"---\nname: {amcp.GUIDE_SKILL_NAME}\ndescription: x\n---\n\n"
        f"## 1. Purpose\n\n{sentinel}\n",
        encoding="utf-8",
    )

    original = getattr(settings, "support_repo_dir", None)
    try:
        settings.support_repo_dir = str(tmp_path)
        skills_index.refresh()
        out = await amcp._h_guide()
        assert out["ok"] is True
        assert sentinel in out["result"]["markdown"]
    finally:
        settings.support_repo_dir = original
        skills_index.refresh()


@pytest.mark.asyncio
async def test_a_missing_guide_is_a_named_refusal_never_an_empty_string(tmp_path):
    """An empty guide is worse than no guide: the caller authors a spec
    from memory and is confidently wrong."""
    from app.config import settings

    (tmp_path / "docs" / "skills").mkdir(parents=True)
    original = getattr(settings, "support_repo_dir", None)
    try:
        settings.support_repo_dir = str(tmp_path)
        skills_index.refresh()
        out = await amcp._h_guide()
        assert out["ok"] is False
        assert out["code"] == "guide_missing"
        assert "SKILL.md" in out["sentence"]
    finally:
        settings.support_repo_dir = original
        skills_index.refresh()


@pytest.mark.asyncio
async def test_the_guide_can_be_read_one_section_at_a_time():
    out = await amcp._h_guide(section="node by node")
    assert out["ok"] is True
    body = out["result"]["markdown"]
    assert body.startswith("## 4."), body[:80]
    assert "## 5." not in body, "a section must stop at the next heading"

    miss = await amcp._h_guide(section="how to brew coffee")
    assert miss["ok"] is False and miss["code"] == "no_such_section"


def test_the_guide_teaches_what_it_claims_to_teach():
    """A stub that passes every structural check is the failure mode a
    'write a real guide' task actually has. These are the four things
    the round asked for, asserted by content."""
    md = GUIDE.read_text(encoding="utf-8")

    # the grammar, including the R38 step kind
    for node in ('"version": 2', "trigger.sources", "dedupe_key",
                 "poll_interval_s", "collect", "items_path", "grant_id",
                 '"kind": "agent"', "output_var", "on_error", "focus",
                 "{{grant.target.id}}", "{{steps.", "{{var."):
        assert node in md, f"the grammar section never mentions {node!r}"

    # the worked examples library
    for example in ("scheduled digest", "triggered reaction",
                    "approval-gated write", "multi-account read"):
        assert example.lower() in md.lower(), f"no worked example: {example}"

    # the loop, and its polarity
    assert "validate" in md.lower() and "Never save and then find out" in md
    assert "workflow__test" in md and "rehearsal_disabled" in md

    # the four rules
    assert "joined" in md and "is_member" in md          # only joined channels
    assert "Never invent a target" in md                  # never invent a target
    assert "A write needs a grant" in md                  # a write needs a grant
    assert "An edit says it was edited" in md             # and says so


# ══════════════════════════════════════════════════════════════════════
# 2. …and the image carries it
# ══════════════════════════════════════════════════════════════════════


def _dockerfile() -> str:
    return (REPO / "Dockerfile").read_text(encoding="utf-8")


def test_the_platform_image_copies_the_skills_to_where_the_loader_looks():
    """The one check nothing else in this repo can make.

    `tsc`-equivalents, the unit sweep and the whole test suite all run
    against the working tree, where `docs/skills` obviously exists. Only
    the build context decides whether it exists in production, and the
    build context is this file.

    The expected path is DERIVED from the loader rather than written
    down, so moving the skills directory moves this assertion with it.
    """
    rel = skills_index.skills_dir().resolve().relative_to(
        skills_index.repo_root().resolve()
    )
    assert str(rel) == "docs/skills"  # sanity: the loader's own layout

    text = _dockerfile()

    workdirs = re.findall(r"^WORKDIR\s+(\S+)", text, re.M)
    assert "/app" in workdirs, "the loader's repo_root() assumes WORKDIR /app"

    # `repo_root()` is `<backend's parent>`: backend/app/support/x.py →
    # parents[3]. So the backend must land at WORKDIR/backend for the
    # skills at WORKDIR/docs/skills to be found.
    assert re.search(r"^COPY\s+backend/\s+\./backend/\s*$", text, re.M), (
        "the backend COPY moved — re-derive where docs/skills must land"
    )

    copies = re.findall(r"^COPY\s+(\S+)\s+(\S+)\s*$", text, re.M)
    dests = {
        pathlib.PurePosixPath("/app") / dst.lstrip("./").rstrip("/")
        for src, dst in copies
        if src.rstrip("/").endswith(str(rel))
    }
    assert pathlib.PurePosixPath("/app") / rel in dests, (
        f"no `COPY {rel}/ ./{rel}/` in the platform Dockerfile — "
        f"skills_index and workflow__guide would both read an empty "
        f"directory in production, on a green build"
    )


def test_dockerignore_does_not_exclude_the_skills():
    """A COPY line is not enough: `.dockerignore` is applied to the
    build context first, so an ignore rule would delete the directory
    before the COPY ever saw it — silently, since COPY of an empty
    directory is not an error."""
    ignore = (REPO / ".dockerignore").read_text(encoding="utf-8")
    patterns = [
        ln.strip() for ln in ignore.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    for pat in patterns:
        norm = pat.rstrip("/").lstrip("/")
        assert norm not in ("docs", "docs/skills", "*.md"), (
            f".dockerignore excludes {pat!r}, which removes docs/skills "
            f"from the build context"
        )


# ══════════════════════════════════════════════════════════════════════
# 3. The tool group
# ══════════════════════════════════════════════════════════════════════


EXPECTED_TOOLS = [
    "workflow__guide",
    "workflow__registry",
    "workflow__templates",
    "workflow__list",
    "workflow__get",
    "workflow__create",
    "workflow__update",
    "workflow__change",
    "workflow__lifecycle",
    "workflow__delete",
    "workflow__run",
    "workflow__test",
]


def test_the_wire_order_is_pinned_and_append_only():
    """The tools array is prefix-stable per namespace — a provider's
    prompt cache keys on the prefix, so inserting in the middle
    invalidates every cached turn. New tools JOIN AT THE END; this pin
    is what makes an insertion show up as a diff."""
    assert amcp.tool_names() == EXPECTED_TOOLS


def test_the_group_stays_inside_its_tool_budget():
    assert len(EXPECTED_TOOLS) <= amcp.MAX_GROUP_TOOLS


def test_the_namespace_is_disjoint_from_every_builtin_skill():
    """`agent_runner.tool_defs` is `core + skill_defs + mcp_defs` with no
    dedupe. A namespace shared with a builtin skill therefore ships the
    provider two tools with one name, on every turn, for every tenant
    with connector dispatch on."""
    from app.services.connector_registry import _collect_skill_prefixes

    assert amcp.AUTOMATION_NAMESPACE not in _collect_skill_prefixes()
    assert amcp.AUTOMATION_NAMESPACE != "automations"


@pytest.mark.asyncio
async def test_registration_exposes_the_whole_group():
    from fastmcp import FastMCP

    mcp = FastMCP("test-r38")
    n = amcp.register_automation_tools(mcp, skill_prefixes=set())
    assert n == len(EXPECTED_TOOLS)

    tools = await mcp.get_tools()
    for name in EXPECTED_TOOLS:
        assert name in tools, f"{name} was not registered"
        assert amcp.AUTOMATION_TOOL_TAG in tools[name].tags
        assert (tools[name].description or "").strip(), (
            f"{name} has no description — the description IS the teaching"
        )


@pytest.mark.asyncio
async def test_a_colliding_namespace_registers_nothing_at_all():
    """Half a group is worse than none: the tenant would get duplicate
    names for the half that registered."""
    from fastmcp import FastMCP

    mcp = FastMCP("test-r38-collide")
    n = amcp.register_automation_tools(
        mcp, skill_prefixes={amcp.AUTOMATION_NAMESPACE},
    )
    assert n == 0
    assert await mcp.get_tools() == {} or not any(
        k.startswith("workflow__") for k in await mcp.get_tools()
    )


@pytest.mark.asyncio
async def test_the_group_shadows_no_tool_the_platform_already_serves():
    """The real server carries the decorator-registered platform tools
    (`memory_*`, `session_*`, `entity_*`, `identity_*`) and, at boot,
    every connector tool. FastMCP's `add_tool` on a duplicate is a warn
    on this path, not a raise — so a collision would register nothing
    and be invisible until a client noticed a tool missing."""
    from app.mcp_server import mcp as real

    existing = set(await real.get_tools())
    assert not (existing & set(EXPECTED_TOOLS))
    assert not any(n.startswith("workflow__") for n in existing)


@pytest.mark.asyncio
async def test_the_registry_tool_answers_from_the_platform_itself(
    monkeypatch, bound_user,
):
    """`workflow__registry` is one of the two tools that never leaves the
    platform: the capability metadata is in THIS process's connector
    registry and the connection state in the vault. The tenant's own
    skill fetches both over HTTP from here; an MCP caller is already
    here, so a round trip would only add a way to fail."""
    async def _on(_uid):
        return True
    monkeypatch.setattr(amcp, "_flag_enabled", _on)

    class _Reg:
        def automation_registry(self):
            return [{
                "connector_id": "slack", "name": "Slack",
                "push": False, "poll": True, "floor_s": 300,
                "events": [{"key": "message_posted"}],
                "scopes_write_by_action": {"slack__send_message": ["chat:write"]},
                "target_param_by_action": {"slack__send_message": "channel"},
                "rate_budget": {"per_day": 20},
            }]

    class _Ident:
        connector_id = "slack"
        status = "active"
        scopes_json = '["chat:write"]'
        provider_account_id = "toup.ai"

    import app.services.connector_registry as cr
    import app.services.connector_vault as cv
    monkeypatch.setattr(cr, "get_registry", lambda: _Reg())

    async def _active(_db, _uid):
        return [_Ident()]
    monkeypatch.setattr(cv, "list_active", _active)

    out = await amcp._h_registry()
    assert out["ok"] is True
    entry = out["result"]["connectors"][0]
    assert entry["connector_id"] == "slack"
    assert entry["connected"] is True
    assert entry["account"] == "toup.ai"
    assert entry["write_actions"]["slack__send_message"]["target_param"] == "channel"


def test_identity_is_never_a_tool_argument():
    """Every handler reads the user from `MCPAuthMiddleware`'s
    ContextVar. A `user_id` parameter would be a cross-tenant read the
    model could ask for in words."""
    for _short, _desc, schema, _handler in amcp._TOOLS:
        props = set((schema.get("properties") or {}).keys())
        assert not (props & {"user_id", "userId", "tenant", "account"}), (
            f"{_short} takes an identity argument"
        )


# ── the tools/list filter ────────────────────────────────────────────


class _T:
    def __init__(self, name):
        self.name = name


async def _list_through(middleware, names):
    async def call_next(_ctx):
        return [_T(n) for n in names]
    return [t.name for t in await middleware.on_list_tools(None, call_next)]


@pytest.mark.asyncio
async def test_the_group_is_hidden_from_a_user_without_automations(monkeypatch):
    monkeypatch.setattr(amcp, "try_get_mcp_user_id", lambda: "u-1")

    async def _off(_uid):
        return False
    monkeypatch.setattr(amcp, "_flag_enabled", _off)

    out = await _list_through(
        amcp.AutomationToolFilterMiddleware(),
        ["memory_search", "gmail__list_messages", *EXPECTED_TOOLS],
    )
    assert out == ["memory_search", "gmail__list_messages"]


@pytest.mark.asyncio
async def test_the_group_is_shown_to_a_user_with_automations(monkeypatch):
    monkeypatch.setattr(amcp, "try_get_mcp_user_id", lambda: "u-1")

    async def _on(_uid):
        return True
    monkeypatch.setattr(amcp, "_flag_enabled", _on)

    out = await _list_through(amcp.AutomationToolFilterMiddleware(),
                              ["memory_search", "workflow__list"])
    assert out == ["memory_search", "workflow__list"]


@pytest.mark.asyncio
async def test_an_unauthenticated_list_is_not_filtered(monkeypatch):
    """Warn-only mode, matching every other filter on this server: the
    handler raises on invocation, which is louder and more accurate than
    a silently short list."""
    monkeypatch.setattr(amcp, "try_get_mcp_user_id", lambda: None)
    out = await _list_through(amcp.AutomationToolFilterMiddleware(),
                              ["memory_search", "workflow__list"])
    assert out == ["memory_search", "workflow__list"]


@pytest.mark.asyncio
async def test_a_failed_flag_read_leaves_the_group_listed(monkeypatch):
    """A DB blip must not look exactly like "the feature was turned
    off" — hiding on error is an outage the user reads as a product
    decision."""
    monkeypatch.setattr(amcp, "try_get_mcp_user_id", lambda: "u-1")

    async def _boom(_uid):
        raise RuntimeError("pool exhausted")
    monkeypatch.setattr(amcp, "_flag_enabled", _boom)

    out = await _list_through(amcp.AutomationToolFilterMiddleware(),
                              ["memory_search", "workflow__list"])
    assert out == ["memory_search", "workflow__list"]


# ── the envelope ─────────────────────────────────────────────────────


class _Resp:
    def __init__(self, status, payload=None, *, not_json=False):
        self.status_code = status
        self._payload = payload
        self._not_json = not_json

    def json(self):
        if self._not_json:
            raise ValueError("not json")
        return self._payload


class _Client:
    def __init__(self, resp=None, raises=None):
        self._resp = resp
        self._raises = raises
        self.calls = []

    async def request(self, method, url, **kw):
        self.calls.append((method, url, kw))
        if self._raises is not None:
            raise self._raises
        return self._resp


@pytest.fixture
def bound_user():
    """`MCPAuthMiddleware` binds the caller's id to a ContextVar; every
    handler reads it from there and from nowhere else. Unbound,
    `get_mcp_user_id()` raises — which is the right answer for an
    unauthenticated MCP call and the wrong one for a unit test."""
    from app.mcp_auth import _current_user_id

    token = _current_user_id.set("u-1")
    try:
        yield "u-1"
    finally:
        _current_user_id.reset(token)


@pytest.fixture
def wired(monkeypatch, bound_user):
    """Flag on, an agent bound, and a settable fake HTTP client."""
    async def _on(_uid):
        return True

    async def _target(_uid):
        return ("https://agent.example", "key-123")

    monkeypatch.setattr(amcp, "_flag_enabled", _on)
    monkeypatch.setattr(amcp, "_agent_target", _target)

    holder: dict = {}

    def _set(client):
        holder["c"] = client
        import app.services.agent_http as ah
        monkeypatch.setattr(ah, "get_agent_http_client", lambda: client)
        return client

    return _set


@pytest.mark.asyncio
async def test_a_successful_call_returns_the_payload_verbatim(wired):
    c = wired(_Client(_Resp(200, {"automations": [{"id": "a1"}]})))
    out = await amcp._agent_call("u-1", "GET", "")
    assert out == {"ok": True, "result": {"automations": [{"id": "a1"}]}}
    method, url, _kw = c.calls[0]
    assert method == "GET"
    assert url == "https://agent.example/api/automations"


@pytest.mark.asyncio
async def test_the_dark_agent_404_is_not_relayed_as_not_found(wired):
    """The tenant's own gate answers `404 Feature not available`, which
    is byte-identical to the platform's "not for you". Once the platform
    gate has passed they mean opposite things — one is permanent, one
    self-heals in about a minute."""
    wired(_Client(_Resp(404, {"detail": "Feature not available"})))
    out = await amcp._agent_call("u-1", "GET", "")
    assert out["ok"] is False
    assert out["code"] == "agent_starting"
    assert out["retryable"] is True


@pytest.mark.asyncio
async def test_an_ordinary_404_is_not_found(wired):
    wired(_Client(_Resp(404, {"detail": "No such automation"})))
    out = await amcp._agent_call("u-1", "GET", "/nope/workflow")
    assert out["code"] == "not_found"
    assert out["sentence"] == "No such automation"


@pytest.mark.asyncio
async def test_a_run_now_refusal_reaches_the_caller_intact(wired):
    """The refusal codes ARE the product of that route — each one has a
    different fix. Flattening them to "it failed" is what makes a caller
    retry a thing that can never succeed."""
    wired(_Client(_Resp(409, {"detail": {
        "code": "needs_setup",
        "sentence": "It is not finished being set up.",
        "refusal_turn": True,
    }})))
    out = await amcp._h_run(automation_id="a1")
    assert out["ok"] is False
    assert out["code"] == "needs_setup"
    assert out["sentence"] == "It is not finished being set up."
    assert out["refusal_turn"] is True


@pytest.mark.asyncio
async def test_a_rehearsal_that_is_switched_off_says_so(wired):
    wired(_Client(_Resp(403, {"detail": {
        "code": "rehearsal_disabled",
        "sentence": "Rehearsals are switched off on this tenant.",
    }})))
    out = await amcp._h_test(automation_id="a1")
    assert out["code"] == "rehearsal_disabled"
    assert "switched off" in out["sentence"]


@pytest.mark.asyncio
async def test_a_refusal_with_only_a_code_still_carries_words(wired):
    """Several engine refusals are `{"code": …}` alone. A code with no
    sentence is wording the caller has to invent, which is how a user
    gets told the wrong reason for a real refusal."""
    wired(_Client(_Resp(409, {"detail": {"code": "last_read"}})))
    out = await amcp._agent_call("u-1", "POST", "/a1/workflow/commit")
    assert out["ok"] is False
    assert out["code"] == "last_read"
    assert out["sentence"].strip(), "a refusal must always carry words"


@pytest.mark.asyncio
async def test_an_unreachable_agent_says_nothing_was_changed(wired):
    import httpx
    wired(_Client(raises=httpx.ConnectError("boom")))
    out = await amcp._agent_call("u-1", "POST", "/a1/arm")
    assert out["code"] == "unreachable"
    assert "nothing was read or changed" in out["sentence"]


@pytest.mark.asyncio
async def test_a_spec_rejection_carries_the_validator_errors(wired):
    wired(_Client(_Resp(422, {"detail": {"errors": [
        {"code": "missing_dedupe_key", "field": "trigger.sources[0].dedupe_key"},
    ]}})))
    out = await amcp._h_create(spec={"version": 2})
    assert out["code"] == "invalid"
    assert out["detail"]["errors"][0]["code"] == "missing_dedupe_key"


@pytest.mark.asyncio
async def test_automations_off_never_reaches_the_agent(monkeypatch):
    async def _off(_uid):
        return False
    monkeypatch.setattr(amcp, "_flag_enabled", _off)

    called = []

    async def _target(_uid):
        called.append(True)
        return ("https://agent.example", "k")
    monkeypatch.setattr(amcp, "_agent_target", _target)

    out = await amcp._agent_call("u-1", "GET", "")
    assert out["code"] == "not_enabled"
    assert not called, "the flag must be checked before the agent is resolved"


# ── the two guarded operations ───────────────────────────────────────


@pytest.mark.asyncio
async def test_delete_refuses_without_an_explicit_confirmation(wired):
    c = wired(_Client(_Resp(200, {"deleted": True})))
    out = await amcp._h_delete(automation_id="a1")
    assert out["code"] == "confirm_required"
    assert not c.calls, "nothing may reach the agent without confirm=true"

    ok = await amcp._h_delete(automation_id="a1", confirm=True)
    assert ok["ok"] is True
    assert c.calls[0][0] == "DELETE"


def test_delete_is_not_hiding_inside_the_lifecycle_enum():
    """`delete` one token away from `pause` in a single enum is how an
    irreversible operation gets picked by accident."""
    assert "delete" not in amcp._LIFECYCLE_ACTIONS
    schema = next(s for short, _d, s, _h in amcp._TOOLS if short == "lifecycle")
    assert schema["properties"]["action"]["enum"] == list(amcp._LIFECYCLE_ACTIONS)


@pytest.mark.asyncio
async def test_lifecycle_rejects_an_unknown_action(wired):
    c = wired(_Client(_Resp(200, {})))
    out = await amcp._h_lifecycle(automation_id="a1", action="delete")
    assert out["code"] == "bad_request"
    assert not c.calls


@pytest.mark.asyncio
async def test_change_refuses_a_call_that_changes_nothing(wired):
    c = wired(_Client(_Resp(200, {})))
    out = await amcp._h_change(automation_id="a1", workflow_rev=3)
    assert out["code"] == "nothing_to_change"
    assert not c.calls


@pytest.mark.asyncio
async def test_change_sends_every_kind_it_was_given(wired):
    c = wired(_Client(_Resp(200, {"workflow_rev": 4})))
    await amcp._h_change(
        automation_id="a1", workflow_rev=3,
        schedule={"preset_id": "weekdays-8"},
        rules={"add": ["only P1s"]},
        steps=[{"n": 1, "text": "read Jira"}],
        permissions=[{"account_id": "slack", "can": ["slack.post"]}],
        accounts={"add": ["github"]},
    )
    _m, url, kw = c.calls[0]
    assert url.endswith("/api/automations/a1/workflow/commit")
    assert set(kw["json"]) == {"workflow_rev", "schedule", "rules", "steps",
                              "permissions", "accounts"}


@pytest.mark.asyncio
async def test_get_reports_a_failed_sub_read_instead_of_omitting_it(
    monkeypatch, bound_user,
):
    """An absent canvas and an empty canvas mean opposite things. A
    handler that dropped the failure would hand back `workflow: null`
    and let the caller read it as "this automation has no schedule and
    no accounts"."""
    async def _on(_uid):
        return True
    monkeypatch.setattr(amcp, "_flag_enabled", _on)

    calls = []

    async def _fake(uid, method, path, **kw):
        calls.append(path)
        if path == "":
            return {"ok": True, "result": {"automations": [{"id": "a1"}]}}
        if path.endswith("/workflow"):
            return {"ok": False, "code": "agent_starting", "sentence": "…"}
        return {"ok": True, "result": {"runs": []}}

    monkeypatch.setattr(amcp, "_agent_call", _fake)
    out = await amcp._h_get(automation_id="a1")
    assert out["ok"] is True
    assert out["result"]["workflow"] is None
    assert out["result"]["workflow_error"]["code"] == "agent_starting"
    assert out["result"]["runs"] == []


# ── the boot wiring ──────────────────────────────────────────────────


def test_platform_main_registers_the_group_and_the_filter():
    """A module nobody calls is a module that ships nothing. Source
    probe rather than a boot test: the lifespan needs a database, and
    what matters here is only that both calls are present."""
    src = (BACKEND / "platform_main.py").read_text(encoding="utf-8")
    assert "register_automation_tools" in src
    assert "AutomationToolFilterMiddleware()" in src
    assert "_collect_skill_prefixes()" in src


def test_the_rehearsal_gate_no_longer_answers_the_ambiguous_404():
    """`automations_proxy._translate_agent_dark` turns any 404 whose
    detail is `Feature not available` into `503 agent_starting`. While
    the dev gate raised exactly that, a rehearsal that was merely
    switched off reached every remote caller as "your agent is still
    booting"."""
    src = (BACKEND / "app" / "api" / "automations.py").read_text(encoding="utf-8")
    branch = src.split("async def test_run(", 1)[1].split("\nasync def ", 1)[0]
    assert "automations_dev_tools" in branch
    assert "rehearsal_disabled" in branch
    assert "status_code=403" in branch
    assert 'detail="Feature not available"' not in branch
