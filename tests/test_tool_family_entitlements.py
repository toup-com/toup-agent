"""Per-tenant tool-family entitlement — and the cache lineage it must not fork.

What this pins
--------------
Every agent container ships four optional blocks to every tenant on every
turn. Measured 2026-08-06 on the real OpenAI chat wire (o200k_base,
`_anthropic_tools_to_openai` + `json.dumps`, delta against the full 95-def /
18,105-tok array):

    doc_generation tools (7 defs)      1,160 tok
    app_builder skill tools (6 defs)   1,036 tok
    toup skill tools (5 defs)            684 tok
    app_builder system-prompt section    547 tok
                                       ---------
                                       3,427 tok

The often-quoted 3,830 figure counted `navigate_to` (400 tok) as document
generation. It is not — see `test_navigate_to_*` below.

THE INVARIANT
-------------
The tools array serializes AHEAD of system+history in the provider prompt
prefix, so **any byte difference starts a separate cache lineage** and every
turn in it re-bills the whole system+history tail. A gate that varies per
turn would cost far more than the tokens it saves. The gate is therefore a
pure function of the TENANT, resolved once per process, and the tests below
assert that with the runtime's own instrument
(`prefix_stability.tools_array_change`, the source of the production
`[PERF] tools_array_changed` line).

Shape follows tests/test_all_channels_one_lineage.py, which pins the
sibling invariant (one lineage per channel).

Run:
    cd backend && RUN_MODE=agent PYTHONPATH=. \
        pytest tests/test_tool_family_entitlements.py
"""
from __future__ import annotations

import hashlib
import json
import tempfile

import pytest

from app.agent import tool_entitlements as te_mod
from app.agent.prefix_stability import tools_array_change
from app.agent.tool_definitions import (
    get_doc_generation_tools,
    get_navigation_tools,
)
from app.config import settings

# sha256 of json.dumps(AgentRunner._core_tool_defs, separators=(",", ":"))
# CAPTURED ON origin/main @ 1b80d28e by constructing a real AgentRunner —
# not recomputed from this branch's code, so it is a genuine before/after
# comparison and not a tautology. 60 definitions.
#
# If this test fails, the default wire tools array moved. That is a new
# provider cache lineage for EVERY tenant on merge; do not "update the
# golden" without deciding that is what you want.
# MOVED DELIBERATELY 2026-08-13 (was 34e3979c…366739f, captured @ 1b80d28e).
# `generate_docx` / `generate_xlsx` descriptions now disclaim Google Docs and
# Sheets. The tools array is otherwise unchanged — still 60 definitions, no
# tool added, removed or reordered; only two description strings differ.
#
# Accepting the new lineage was the point of the change, not a side effect.
# `generate_docx` described itself as "use when the user wants an editable
# document", which is precisely what someone asking for a Google Doc wants, so
# the model created a real Doc AND stapled a stray .docx beside it. The routing
# guidance added to `_build_system_prompt` in #603 was present and un-gated on
# the running image and lost anyway: a tool description is read at the moment of
# choosing, a system-prompt bullet is thousands of tokens away.
#
# Cost accepted: one prompt-cache re-warm per tenant on merge. See
# tests/test_generate_tools_disclaim_google.py for the behaviour this buys.
#
# MOVED AGAIN, DELIBERATELY, 2026-08-13 (was caad2af5…3f656b). `update_job`'s
# status enum gained `waiting_on_user` and `create_job`'s CONTRACT sentence no
# longer offers "mark it 'failed'" as the exit for work that is blocked. Still
# 60 definitions; no tool added, removed or reordered.
#
# This one HAD to be a wire change. The old enum was
# ["running","completed","failed"], so a model whose job was blocked behind a
# confirmation card had no legal value meaning "waiting" — it picked `failed`,
# and the user got a red job card sitting directly beside the card asking them
# to approve something. `waiting_on_user`, `PARKED_STATUSES`, the amber "Waiting
# on you" chip on web and the non-failure branch in the mobile JobDetailScreen
# all already existed; the enum was the only thing standing between them and
# the model. An input_schema is not advisory — it is the set of values the
# provider will let the model emit — so no prompt could have reached this.
#
# Cost accepted a second time: one more prompt-cache re-warm per tenant. See
# tests/test_a_blocked_job_is_not_a_failed_job.py.
#
# MOVED A THIRD TIME, DELIBERATELY, 2026-08-18 (was 0039fbdc…5a473cb).
# `web_search`'s description gained the FRESHNESS paragraph (results are
# date-filtered and carry a published date; for "newest X" run a NEUTRAL
# discovery query, not a site:-restricted one, then confirm on the official
# domain; two agreeing sources or say you could not verify). Still 60
# definitions; no tool added, removed or reordered; input_schema unchanged.
#
# This one had to be on the tool, not only in the system prompt: the incident
# (docs/web-search/freshness-incident.md) showed the model choosing
# site:-anchored queries at the moment it decides HOW to call web_search, and
# the description is the text it reads at that moment.
#
# Cost accepted a third time: one more prompt-cache re-warm per tenant.
MAIN_CORE_TOOLS_SHA256 = (
    "12280b25276446a79737b51b670476d7e9e8b06d71c57bb3562457c1646f5532"
)
MAIN_CORE_TOOLS_COUNT = 60

DOC_TOOL_NAMES = {
    "generate_pdf", "generate_docx", "generate_xlsx", "generate_pptx",
    "generate_markdown", "convert_document", "generate_html_to_pdf",
}


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_entitlements():
    """The gate memoizes on purpose (that memo is what makes it
    container-stable). Reset around every test so each one resolves its
    own value, and restore the process default afterwards."""
    original = getattr(settings, "agent_tool_families", "*")
    original_doc_flag = getattr(settings, "feature_doc_generation", True)
    te_mod.reset_cache_for_tests()
    yield
    settings.agent_tool_families = original
    settings.feature_doc_generation = original_doc_flag
    te_mod.reset_cache_for_tests()


def _entitle(value: str) -> None:
    """Set this tenant's entitlement the way the container env would."""
    settings.agent_tool_families = value
    te_mod.reset_cache_for_tests()


def _runner():
    """A REAL AgentRunner — `__init__` is the code under test, so this
    cannot use `AgentRunner.__new__` the way the channel-lineage test does.
    The LLM services are unused during tool assembly."""
    from app.agent.agent_runner import AgentRunner

    return AgentRunner(llm_service=None, tool_executor=None)


class _FakeSkillLoader:
    """Minimal stand-in for the per-turn skill-tool read path in
    `AgentRunner.tool_defs` — the only place skill tools enter the array."""

    def __init__(self, tools):
        self._tools = list(tools)

    def get_all_tool_definitions(self):
        return list(self._tools)


def _wire(defs) -> str:
    return json.dumps(defs, separators=(",", ":"))


# ----------------------------------------------------------------------
# 1. The "nothing changes on merge" proof — the most important test here
# ----------------------------------------------------------------------

def test_shipped_default_withholds_exactly_the_toup_family():
    """The loadout is a decision, and this is where it is pinned.

    G-15 (2026-08-10) enabled the gate with `doc_generation,app_builder`,
    withholding `toup` — the only family with zero invocations by any
    tenant across 14 days of production telemetry. Withholding a family
    makes it UNREACHABLE (the intent filter iterates loaded definitions),
    so widening or narrowing this set is a product decision, never a
    drive-by edit. Both directions fail here:

      * adding `toup` back silently re-inflates every tenant's prefix by
        685 tokens and can push the founder's array back over OpenAI's
        128-tool cap;
      * dropping `app_builder` or `doc_generation` breaks tenants that
        were measured using them, with a scripted refusal and no way for
        the turn to recover.
    """
    from app.config import Settings
    from app.agent.tool_entitlements import FAMILIES

    default = Settings.model_fields["agent_tool_families"].default
    shipped = {f.strip() for f in default.split(",") if f.strip()}

    assert shipped == {"doc_generation", "app_builder"}, (
        f"the shipped tool-family loadout changed to {sorted(shipped)} — "
        "see the measurement recorded beside the field in config.py"
    )
    assert set(FAMILIES) - shipped == {"toup"}, (
        "the withheld set is no longer exactly {'toup'} — either a family "
        "was added without a loadout decision, or one was renamed"
    )


def test_default_wire_array_is_byte_identical_to_today():
    """Entitlement "*" must reproduce origin/main's array EXACTLY.

    This pins that the GATE ITSELF is transparent: with every family
    entitled, the assembled wire is byte-for-byte what the pre-gate code
    produced, so the only thing that ever moves a tenant's cache lineage
    is a deliberate loadout change (pinned in the test above) — never the
    mechanism.

    ANTI-VACUITY / control note: this test stays GREEN if `navigate_to` is
    moved back inside `get_doc_generation_tools()`, because the assembled
    bytes are the same either way. It is not a blanket "nothing changed"
    assertion — it pins the wire, and the navigation tests below pin the
    grouping.
    """
    _entitle("*")
    defs = _runner()._core_tool_defs
    assert len(defs) == MAIN_CORE_TOOLS_COUNT
    digest = hashlib.sha256(_wire(defs).encode("utf-8")).hexdigest()
    assert digest == MAIN_CORE_TOOLS_SHA256, (
        "the default wire tools array changed — every tenant starts a new "
        "provider cache lineage on merge and re-bills the whole "
        "system+history prefix behind it"
    )


def test_blank_and_unset_entitlement_are_all_families():
    """A bridge that writes `AGENT_TOOL_FAMILIES=` (or never writes it)
    must NOT silently strip a tenant's tools."""
    for value in ("", "   ", "*"):
        _entitle(value)
        assert te_mod.entitled_families() == te_mod.ALL_FAMILIES, value
        assert hashlib.sha256(
            _wire(_runner()._core_tool_defs).encode("utf-8")
        ).hexdigest() == MAIN_CORE_TOOLS_SHA256, value


# ----------------------------------------------------------------------
# 2. The known trap — navigate_to is navigation, not document generation
# ----------------------------------------------------------------------

def test_doc_generation_group_contains_no_navigation():
    """`navigate_to` shipped as the last element of
    `get_doc_generation_tools()`, so gating that group took page transfers
    ("take me to my brain") down with the exporters."""
    names = {t["name"] for t in get_doc_generation_tools()}
    assert names == DOC_TOOL_NAMES
    assert "navigate_to" not in names
    assert [t["name"] for t in get_navigation_tools()] == ["navigate_to"]


def test_navigate_to_survives_feature_doc_generation_false():
    """The pre-existing fleet kill switch must not remove navigation."""
    _entitle("*")
    settings.feature_doc_generation = False
    names = {t["name"] for t in _runner()._core_tool_defs}
    assert "navigate_to" in names, (
        "feature_doc_generation=false removed page navigation — that is the "
        "trap this PR exists to close"
    )
    assert not (names & DOC_TOOL_NAMES), "the exporters should be gone"


def test_navigate_to_survives_a_withheld_doc_generation_entitlement():
    """Same trap, reached through the new per-tenant gate."""
    _entitle("app_builder,toup")
    settings.feature_doc_generation = True
    names = {t["name"] for t in _runner()._core_tool_defs}
    assert "navigate_to" in names
    assert not (names & DOC_TOOL_NAMES)


# ----------------------------------------------------------------------
# 3. Withholding a family removes what it should — and only that
# ----------------------------------------------------------------------

def test_withholding_doc_generation_drops_exactly_the_seven_exporters():
    _entitle("*")
    full = {t["name"] for t in _runner()._core_tool_defs}
    _entitle("app_builder,toup")
    gated = {t["name"] for t in _runner()._core_tool_defs}
    assert full - gated == DOC_TOOL_NAMES
    assert not (gated - full)


def test_withheld_doc_generation_costs_the_measured_token_delta():
    """Measured, not modelled: the real wire conversion + o200k_base.

    1,160 tok on 2026-08-06. Asserted as a band because a description edit
    elsewhere should not fail this file, but a silent collapse to ~0 (or a
    swing that means the family boundary moved) must.
    """
    tiktoken = pytest.importorskip("tiktoken")
    try:
        enc = tiktoken.get_encoding("o200k_base")
    except Exception:  # noqa: BLE001 - no network for the BPE file
        pytest.skip("o200k_base encoding unavailable offline")
    from app.services.openai_agent_service import _anthropic_tools_to_openai

    def wire_tok(defs):
        return len(enc.encode(json.dumps(_anthropic_tools_to_openai(defs))))

    _entitle("*")
    full = wire_tok(_runner()._core_tool_defs)
    _entitle("app_builder,toup")
    gated = wire_tok(_runner()._core_tool_defs)
    delta = full - gated
    assert 1_000 <= delta <= 1_350, (
        f"doc_generation wire cost moved: {delta} tok (measured 1,160)"
    )
    # And navigation is NOT part of that delta.
    assert wire_tok(get_navigation_tools()) > 300


# ----------------------------------------------------------------------
# 4. THE CACHE-LINEAGE INVARIANT — byte-equality across turns
# ----------------------------------------------------------------------

@pytest.mark.parametrize(
    "entitlement", ["*", "app_builder,toup", "toup", "none"]
)
def test_wire_array_is_byte_stable_across_turns(entitlement):
    """One container, one tenant, N turns => ONE tools array, forever.

    Read through the runtime's own instrument: `tools_array_change` is the
    function whose non-None return produces the production
    `[PERF] tools_array_changed old_n=.. new_n=..` line. If it ever fires
    here, production would log a genuine mid-life prefix mutation and the
    tenant would re-bill its whole cached prefix.

    MUTATION: make the gate per-turn — move the family filter out of
    `AgentRunner.__init__` into the `tool_defs` property and key
    `entitled_families()` on a turn counter — and this goes RED on the
    withheld parametrisations.
    """
    _entitle(entitlement)
    # Boot once, exactly as agent_main does.
    runner = _runner()
    runner.skill_loader = _FakeSkillLoader(
        [
            {"name": f"app_builder__{n}", "description": f"ab {n}",
             "input_schema": {"type": "object"}}
            for n in ("create", "deploy")
        ]
    )

    seen: dict[str, tuple] = {}
    arrays = []
    for _turn in range(12):
        current = runner.tool_defs
        arrays.append(_wire(current))
        changed = tools_array_change(seen, "user-fixed-tenant", current)
        assert changed is None or _turn == 0, (
            f"[PERF] tools_array_changed fired on turn {_turn} "
            f"(old_n={changed[0]} new_n={changed[1]}) — the array is not "
            f"tenant-stable, so this tenant forks a new cache lineage "
            f"mid-conversation"
        )
    assert len(set(arrays)) == 1


def test_across_turns_fixture_is_not_vacuous():
    """ANTI-VACUITY control for the test above.

    An empty (or tiny) array would satisfy byte-equality trivially, and a
    `tools_array_change` that never fires for ANY input would make the
    assertion unfalsifiable. Pin both: the array is real, and the
    instrument does fire when the array genuinely moves.
    """
    _entitle("*")
    runner = _runner()
    assert len(runner.tool_defs) > 40, len(runner.tool_defs)

    seen: dict[str, tuple] = {}
    a = runner.tool_defs
    assert tools_array_change(seen, "u", a) is None  # first turn: baseline
    assert tools_array_change(seen, "u", a) is None  # unchanged: silent
    fired = tools_array_change(seen, "u", a[:-1])    # one def removed
    assert fired is not None and fired == (len(a), len(a) - 1), fired


def test_entitled_tenant_is_unaffected():
    """ANTI-VACUITY control for the whole feature: the tenant everyone is
    today (full entitlement) sees the origin/main array, unchanged, on
    every turn, with skills attached."""
    _entitle("*")
    runner = _runner()
    runner.skill_loader = _FakeSkillLoader(
        [{"name": "toup__create_spec", "description": "d",
          "input_schema": {"type": "object"}}]
    )
    core_digest = hashlib.sha256(
        _wire(runner._core_tool_defs).encode("utf-8")
    ).hexdigest()
    assert core_digest == MAIN_CORE_TOOLS_SHA256
    names = {t["name"] for t in runner.tool_defs}
    assert DOC_TOOL_NAMES <= names
    assert "navigate_to" in names
    assert "toup__create_spec" in names
    assert te_mod.refusal_for_tool("generate_pdf") is None
    assert te_mod.refusal_for_tool("app_builder__create_app") is None


# ----------------------------------------------------------------------
# 5. Skill families — withheld at the loader, so tools + prompt section +
#    execution disappear together
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_skill_loader_withholds_an_unentitled_family():
    from app.agent.skills.builtins.routines.skill import RoutinesSkill
    from app.agent.skills.builtins.toup.skill import ToupSkill
    from app.agent.skills.loader import SkillLoader

    _entitle("doc_generation")  # toup + app_builder withheld
    loader = SkillLoader()
    assert await loader.register_dynamic(ToupSkill()) is False
    # CONTROL: `routines` belongs to no gated family and must survive.
    assert await loader.register_dynamic(RoutinesSkill()) is True

    names = {t["name"] for t in loader.get_all_tool_definitions()}
    assert not any(n.startswith("toup__") for n in names), sorted(names)
    assert any(n.startswith("routines__") for n in names)
    assert not loader.is_skill_tool("toup__create_spec")
    # The 547-tok app_builder prompt section rides on registration too.
    assert not any("Auto Builder" in s or "app_builder__" in s
                   for s in loader.get_all_system_prompt_sections())


@pytest.mark.asyncio
async def test_skill_loader_keeps_every_family_by_default():
    """ANTI-VACUITY control: same code path, full entitlement, nothing
    withheld — otherwise the test above would pass on a loader that simply
    rejects everything."""
    from app.agent.skills.builtins.routines.skill import RoutinesSkill
    from app.agent.skills.builtins.toup.skill import ToupSkill
    from app.agent.skills.loader import SkillLoader

    _entitle("*")
    loader = SkillLoader()
    assert await loader.register_dynamic(ToupSkill()) is True
    assert await loader.register_dynamic(RoutinesSkill()) is True
    names = {t["name"] for t in loader.get_all_tool_definitions()}
    assert any(n.startswith("toup__") for n in names)
    assert loader.is_skill_tool("toup__create_spec")


# ----------------------------------------------------------------------
# 6. The graceful path — a withheld capability is refused, not invented
# ----------------------------------------------------------------------

def test_withheld_tool_gets_a_refusal_not_a_hallucination():
    _entitle("app_builder,toup")
    msg = te_mod.refusal_for_tool("generate_xlsx")
    assert msg and msg.startswith("ERROR: ")
    assert "not enabled on this account" in msg
    # It must name the plan boundary and steer the model AWAY from both
    # failure-modes: a transient-error framing it would retry around, and an
    # improvised substitute (exec/write_file) that fakes the capability.
    assert "not part of the current plan" in msg
    assert "do not describe it as temporarily broken" in msg
    assert "do not improvise a substitute" in msg
    # Unconditional tools are never refused.
    for name in ("navigate_to", "web_search", "read_file", ""):
        assert te_mod.refusal_for_tool(name) is None, name


@pytest.mark.asyncio
async def test_executor_refuses_a_withheld_tool_before_the_handler_runs():
    """The backstop: withholding the DEFINITION stops the model picking the
    tool; this stops a call that arrives another way (replayed tool_use
    block, MCP alias, hallucinated name) from reaching the generator."""
    from app.agent.tool_executor import ToolExecutor

    with tempfile.TemporaryDirectory() as tmp:
        # `ToolExecutor(workspace=...)` is NOT sufficient to contain this test.
        # The doc generators write through `workspace_perms.shared_makedirs`,
        # which builds its path from `settings.agent_workspace_dir` — default
        # "/app/workspace" — and ignores the executor's workspace entirely. So
        # the CONTROL half (an entitled tenant, where the handler really runs)
        # tried to create /app and died with
        #   ERROR: OSError: [Errno 30] Read-only file system: '/app'
        # on any host that is not the agent container. Point the setting at the
        # same temp dir so the control exercises the handler for real instead
        # of failing on the filesystem.
        _prev_ws = settings.agent_workspace_dir
        settings.agent_workspace_dir = tmp
        try:
            _entitle("app_builder,toup")
            ex = ToolExecutor(workspace=tmp)
            ex._user_id = "u1"
            out = await ex.execute(
                "generate_markdown", {"content": "# hi", "filename": "n.md"}
            )
            assert out.startswith("ERROR: ")
            assert "not enabled on this account" in out
            assert not ex.pending_attachments, "the handler ran anyway"

            # CONTROL: same executor, entitled tenant — the handler DOES run.
            _entitle("*")
            ok = await ex.execute(
                "generate_markdown", {"content": "# hi", "filename": "n.md"}
            )
            assert not ok.startswith("ERROR: "), ok
            assert len(ex.pending_attachments) == 1
        finally:
            settings.agent_workspace_dir = _prev_ws


def test_unavailable_prompt_section_replaces_the_how_to_guide():
    """A user without the entitlement must be told plainly, not left to a
    model that invents the feature. ~45 tok instead of ~500."""
    from app.agent.agent_runner import _DOC_GENERATION_UNAVAILABLE

    assert "NOT enabled on this account" in _DOC_GENERATION_UNAVAILABLE
    assert "generate_pdf" not in _DOC_GENERATION_UNAVAILABLE
    assert len(_DOC_GENERATION_UNAVAILABLE) < 600


# ----------------------------------------------------------------------
# 7. Value parsing — a typo must not strip a tenant or wedge a boot
# ----------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("*", te_mod.ALL_FAMILIES),
        ("", te_mod.ALL_FAMILIES),
        ("none", frozenset()),
        ("-", frozenset()),
        ("toup", frozenset({"toup"})),
        (" toup , doc_generation ", frozenset({"toup", "doc_generation"})),
        ("toup,not_a_family", frozenset({"toup"})),
        ("not_a_family", frozenset()),
    ],
)
def test_entitlement_value_grammar(raw, expected):
    _entitle(raw)
    assert te_mod.entitled_families() == expected


def test_gate_takes_no_per_turn_input():
    """Structural guard for the invariant. `entitled_families()` must stay
    zero-argument: the moment it accepts a turn/message/channel/user, the
    tools array can fork mid-life and this whole remediation program
    regresses. Enforced here so a future refactor has to argue with a test.
    """
    import inspect

    for fn in (te_mod.entitled_families,):
        assert not inspect.signature(fn).parameters, (
            f"{fn.__name__} gained a parameter — a per-turn gate re-creates "
            "the cache-lineage bug this module exists to prevent"
        )
    # The memo is the mechanism; pin that it exists and survives a settings
    # change (a container restart is required, by design).
    _entitle("toup")
    first = te_mod.entitled_families()
    settings.agent_tool_families = "*"          # no reset — mid-life change
    assert te_mod.entitled_families() == first, (
        "the entitlement re-resolved mid-process — that is a mid-life tools "
        "array mutation and a new cache lineage"
    )


# ----------------------------------------------------------------------
# 8. The app_builder family must gate the WHOLE app builder
#
# `AppGatewaySkill` lives in `skills/builtins/app_builder/` and registers
# under the name "app", so its 13 tools are `app__*` — not `app_builder__*`.
# The family listed only "app_builder", which meant a tenant who withheld
# "the app builder" lost its 6 entry points and kept all 13 tools of the
# machine behind them: two thirds of the tokens, and a set of tools the
# agent can still call with no way to have built anything to call them on.
#
# It stopped being a token-accounting nicety when the wire array went over
# OpenAI's hard 128-tool cap, where the overflow is TRUNCATED — every tool
# this family fails to withhold is paid for by dropping whichever tools
# happen to sort last.
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_withholding_the_app_builder_withholds_its_gateway_too():
    from app.agent.skills.builtins.app_builder.app_gateway_skill import (
        AppGatewaySkill,
    )
    from app.agent.skills.builtins.app_builder.skill import AppBuilderSkill
    from app.agent.skills.builtins.routines.skill import RoutinesSkill
    from app.agent.skills.loader import SkillLoader

    _entitle("doc_generation,toup")  # app_builder withheld
    loader = SkillLoader()
    assert await loader.register_dynamic(AppBuilderSkill()) is False
    assert await loader.register_dynamic(AppGatewaySkill()) is False
    # CONTROL: a skill in no gated family still loads, so this is not a
    # loader that has simply stopped accepting anything.
    assert await loader.register_dynamic(RoutinesSkill()) is True

    names = {t["name"] for t in loader.get_all_tool_definitions()}
    assert not any(n.startswith("app__") for n in names), sorted(names)
    assert not any(n.startswith("app_builder__") for n in names)
    assert any(n.startswith("routines__") for n in names)


@pytest.mark.asyncio
async def test_an_entitled_tenant_still_gets_the_gateway():
    """ANTI-VACUITY control for the test above."""
    from app.agent.skills.builtins.app_builder.app_gateway_skill import (
        AppGatewaySkill,
    )
    from app.agent.skills.loader import SkillLoader

    _entitle("*")
    loader = SkillLoader()
    assert await loader.register_dynamic(AppGatewaySkill()) is True
    names = {t["name"] for t in loader.get_all_tool_definitions()}
    assert any(n.startswith("app__") for n in names)


def test_every_skill_shipped_under_app_builder_is_named_by_the_family():
    """Structural, so the NEXT skill dropped into that directory is covered
    the day it is added rather than the day someone notices the token bill.

    The defect was not that "app" was hard to find — it is one directory
    listing away. It was that nothing connected the family's membership to
    the directory it is named after, so the two could drift silently and
    the only symptom was a number in a token report.
    """
    import ast
    import pathlib

    pkg = pathlib.Path(te_mod.__file__).parent / "skills/builtins/app_builder"
    assert pkg.is_dir(), pkg

    declared = set()
    for path in sorted(pkg.glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            # `meta = SkillMeta(name="app", ...)` — read the literal rather
            # than importing, so this cannot be defeated by an import error.
            if not isinstance(node, ast.Call):
                continue
            if getattr(node.func, "id", None) != "SkillMeta":
                continue
            for kw in node.keywords:
                if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                    declared.add(kw.value.value)

    assert declared, f"no SkillMeta(name=…) found under {pkg} — did skills move?"
    gated = te_mod.FAMILIES["app_builder"].skills
    missing = sorted(declared - set(gated))
    assert not missing, (
        f"skills {missing} ship from the app_builder package but are not in the "
        f"`app_builder` family, so withholding 'the app builder' leaves them "
        f"loaded — tokens on every turn, and tool slots against OpenAI's 128 cap."
    )


def test_skill_enabled_answers_for_the_gateway_by_name():
    """The loader's actual question, asked directly."""
    _entitle("doc_generation,toup")
    assert te_mod.skill_enabled("app") is False
    assert te_mod.skill_enabled("app_builder") is False
    assert te_mod.skill_enabled("routines") is True
    _entitle("*")
    assert te_mod.skill_enabled("app") is True
