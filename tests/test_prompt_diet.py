"""W2.1a prefix diet — regression pins for ``settings.prompt_diet``.

The contract this file locks (docs/audits/2026-07-remediation.md, gap #6):

* the flag defaults OFF, and flag-off output is byte-identical to the
  pre-diet output for every touched section and tool schema;
* flag-on never changes tool ARG SHAPES — properties/enums/required are
  byte-identical between the diet and full schemas, only description
  strings shrink;
* the diet actually diets: token-count ceilings on every compact section
  (tiktoken when available — it's in CI's deps — else chars/4 with slack).

Sections/schemas covered: app_builder system-prompt essay, the three fat
tool schemas (routines__remind / routines__create / triggers__create),
platform_knowledge, doc_generation.
"""
from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path

import pytest

from app.config import Settings, settings
from app.agent import prompt_diet as prompt_diet_mod
from app.agent.prompt_diet import (
    DOC_GENERATION_DIET,
    PLATFORM_KNOWLEDGE_DIET,
    apply_tool_description_diet,
    prompt_diet_enabled,
    skill_prose_diet_enabled,
    skill_section_diet,
    skill_sections_diet,
)


def _tokens(text: str) -> int:
    try:
        import tiktoken

        return len(tiktoken.get_encoding("o200k_base").encode(text))
    except Exception:
        return len(text) // 4


def _strip_descriptions(node):
    """Recursively drop every ``description`` key — what remains IS the
    arg shape (names, types, enums, required, nesting)."""
    if isinstance(node, dict):
        return {
            k: _strip_descriptions(v)
            for k, v in node.items()
            if k != "description"
        }
    if isinstance(node, list):
        return [_strip_descriptions(v) for v in node]
    return node


@pytest.fixture
def diet_on(monkeypatch):
    monkeypatch.setattr(settings, "prompt_diet", True, raising=False)


@pytest.fixture
def diet_off(monkeypatch):
    monkeypatch.setattr(settings, "prompt_diet", False, raising=False)


# ── flag plumbing ──────────────────────────────────────────────────────


def test_flag_defaults_on():
    """Flipped 2026-08-05, after it was already set on 60 of 61 containers.

    The one container without it was the founder's own tenant, silently
    paying ~5,750 extra prefix tokens per turn because the per-tenant .env
    that carries agent flags is written once at provision time and never
    picks up flags introduced later.
    """
    assert Settings.model_fields["prompt_diet"].default is True


def test_flag_can_still_be_turned_off():
    """The kill switch must survive the default flip."""
    assert Settings(_env_file=None, prompt_diet=False).prompt_diet is False


def test_helper_reads_settings(monkeypatch):
    monkeypatch.setattr(settings, "prompt_diet", False, raising=False)
    assert prompt_diet_enabled() is False
    monkeypatch.setattr(settings, "prompt_diet", True, raising=False)
    assert prompt_diet_enabled() is True


def test_bridge_ships_the_flag():
    bridge = (
        Path(__file__).resolve().parents[2] / "bridge" / "pool_addon.py"
    ).read_text()
    assert '"PROMPT_DIET"' in bridge, "flag missing from _FEATURE_FLAG_ENVS"


# ── apply_tool_description_diet mechanics ──────────────────────────────


def test_apply_diet_touches_only_descriptions():
    tools = [
        {
            "name": "t1",
            "description": "fat essay",
            "input_schema": {
                "type": "object",
                "properties": {
                    "a": {"type": "string", "description": "long", "enum": ["x", "y"]},
                    "b": {"type": "integer", "description": "long b"},
                },
                "required": ["a"],
            },
        }
    ]
    before_shape = _strip_descriptions(copy.deepcopy(tools))
    apply_tool_description_diet(
        tools, {"t1": "slim"}, {"t1": {"a": "short", "missing_prop": "ignored"}}
    )
    assert tools[0]["description"] == "slim"
    assert tools[0]["input_schema"]["properties"]["a"]["description"] == "short"
    # b untouched; unknown property ignored without error
    assert tools[0]["input_schema"]["properties"]["b"]["description"] == "long b"
    assert _strip_descriptions(copy.deepcopy(tools)) == before_shape


def test_apply_diet_unknown_tool_is_noop():
    tools = [{"name": "other", "description": "keep", "input_schema": {}}]
    snapshot = copy.deepcopy(tools)
    apply_tool_description_diet(tools, {"t1": "slim"}, {"t1": {"a": "x"}})
    assert tools == snapshot


# ── routines / triggers schemas ────────────────────────────────────────


def _routines_tools():
    from app.agent.skills.builtins.routines.skill import RoutinesSkill

    return RoutinesSkill().get_tools()


def _triggers_tools():
    from app.agent.skills.builtins.triggers.skill import TriggersSkill

    return TriggersSkill().get_tools()


def test_routines_flag_off_serves_full_descriptions(diet_off):
    remind = next(t for t in _routines_tools() if t["name"] == "routines__remind")
    # the legacy essay is the fat one — far above the diet ceiling
    assert _tokens(json.dumps(remind)) > 500


def test_flag_off_then_on_then_off_is_stable(monkeypatch):
    """Toggling the flag must never leak diet strings into the flag-off
    output (the get_tools list is rebuilt per call, not cached)."""
    monkeypatch.setattr(settings, "prompt_diet", False, raising=False)
    before = json.dumps(_routines_tools(), sort_keys=True)
    monkeypatch.setattr(settings, "prompt_diet", True, raising=False)
    during = json.dumps(_routines_tools(), sort_keys=True)
    monkeypatch.setattr(settings, "prompt_diet", False, raising=False)
    after = json.dumps(_routines_tools(), sort_keys=True)
    assert before == after, "flag-off output changed after a flag-on call"
    assert before != during


@pytest.mark.parametrize(
    "get_tools, dieted",
    [
        (_routines_tools, ("routines__remind", "routines__create")),
        (_triggers_tools, ("triggers__create",)),
    ],
)
def test_arg_shapes_identical_between_diet_and_full(monkeypatch, get_tools, dieted):
    """THE core contract: the diet may only shrink description strings.
    Tool names, order, count, and every schema shape byte-match."""
    monkeypatch.setattr(settings, "prompt_diet", False, raising=False)
    full = get_tools()
    monkeypatch.setattr(settings, "prompt_diet", True, raising=False)
    diet = get_tools()

    assert [t["name"] for t in full] == [t["name"] for t in diet]
    assert _strip_descriptions(full) == _strip_descriptions(diet)
    # and the dieted tools actually changed their description
    for name in dieted:
        f = next(t for t in full if t["name"] == name)
        d = next(t for t in diet if t["name"] == name)
        assert f["description"] != d["description"]


@pytest.mark.parametrize(
    "get_tools, name, ceiling",
    [
        # Round 4 (item 5b): +`in_seconds` (a new schema property, ~60 tok
        # with its description) — the relative-reminder fix. Re-measured
        # diet ≈ 700; ceiling moved to sit just above it, same discipline.
        (_routines_tools, "routines__remind", 720),
        (_routines_tools, "routines__create", 500),
        (_triggers_tools, "triggers__create", 450),
    ],
)
def test_diet_schema_token_ceilings(diet_on, get_tools, name, ceiling):
    """Measured (o200k, json.dumps): full remind 1,033 / create 797 /
    triggers ~657 → diet 600 / 455 / 403, of which the untouched shape
    skeletons are 182 / 132 / 141. Ceilings sit just above the measured
    diet sizes so description creep fails the build."""
    tool = next(t for t in get_tools() if t["name"] == name)
    assert _tokens(json.dumps(tool)) <= ceiling


# ── app_builder section ────────────────────────────────────────────────


def _app_builder_section() -> str:
    from app.agent.skills.builtins.app_builder.skill import AppBuilderSkill

    return AppBuilderSkill().get_system_prompt_section()


def test_app_builder_flag_off_serves_legacy_essay(diet_off):
    section = _app_builder_section()
    # legacy-only phrasing — vanishes under the diet
    assert "READ THIS FIRST" in section
    assert _tokens(section) > 1500


def test_app_builder_diet_keeps_the_invariant(diet_on):
    section = _app_builder_section()
    assert "READ THIS FIRST" not in section
    # the one critical invariant + the flow survive the diet
    assert "app_builder__build_app" in section
    assert "Layer 0" in section and "Layer 1B" in section
    assert "app_builder__research_category" in section
    assert "app_builder__gather_requirements" in section
    assert _tokens(section) <= 600  # target ~400


# ── platform_knowledge / doc_generation constants + runner wiring ──────


def test_platform_knowledge_diet_keeps_the_load_bearing_parts():
    for kept in (
        "## Pages — where things live",
        "## What you should NEVER make the user do",
        "Never \nfake a tool call as text.".replace("\n", ""),  # anti-fake rule
        "`/agent/settings`",
        "recall_day",
    ):
        assert kept in PLATFORM_KNOWLEDGE_DIET
    assert _tokens(PLATFORM_KNOWLEDGE_DIET) <= 1400  # measured 1,329; legacy ≈ 2,000


def test_doc_generation_diet_keeps_tool_choice_and_convert_rule():
    for kept in (
        "generate_pdf",
        "generate_docx",
        "generate_xlsx",
        "generate_pptx",
        "generate_markdown",
        "convert_document",
        # Round 46 (C9) gave producers to CSV / JSON / txt / code and to
        # audio. The diet is the LIVE prompt for a diet-enabled tenant, so a
        # producer missing from this list is a tool the model cannot be told
        # to use — which is the same defect the round fixed the other way
        # round (`csv` in the turn-1 gate with no generator behind it).
        "generate_data_file",
        "generate_audio",
        "do NOT call generate_pdf",
    ):
        assert kept in DOC_GENERATION_DIET
    # 256 → 281 in round 46: two new tool lines, at the shortest wording that
    # still names what each one makes. Target is still ~200; the video refusal
    # deliberately does NOT live here — it rides the turn-conditional
    # `# Requested format` section, which is only built on a turn that asks
    # for one (format_intent.requested_format_section).
    assert _tokens(DOC_GENERATION_DIET) <= 285


def test_runner_wiring_order_and_both_swap_sites():
    """Source pins: (1) the platform_knowledge swap happens BEFORE the
    owner-fact append, so OWNER_GLOBAL_FACT + fencing ride both paths;
    (2) the doc_generation swap exists; (3) both sites gate on the
    helper, not a raw settings read."""
    src = (
        Path(__file__).resolve().parents[1] / "app" / "agent" / "agent_runner.py"
    ).read_text()
    # The swap became channel-aware — _platform_knowledge_diet(_voice_now)
    # rather than the bare constant — because the diet's own decision rules
    # named `create_job` and routed search to `browser`, which would have
    # reverted the voice fix the moment PROMPT_DIET was turned on. The
    # ORDERING invariant this test exists for is unchanged; only the symbol is.
    # See tests/test_voice_answers_inline.py::TestPromptDietIsVoiceAware.
    pk_swap = src.find('section_parts["platform_knowledge"] = _platform_knowledge_diet(')
    owner = src.find('section_parts["platform_knowledge"] += "\\n\\n" + OWNER_GLOBAL_FACT')
    doc_swap = src.find('section_parts["doc_generation"] = _DOC_GENERATION_DIET')
    assert pk_swap != -1 and doc_swap != -1
    assert owner != -1 and pk_swap < owner
    assert src.count("_prompt_diet_enabled()") >= 2


# ══════════════════════════════════════════════════════════════════════
# R48 patch I — the skill prose diet SEAM
#
# What is being pinned, and what is deliberately NOT:
#
#   * `skill_section_diet(name, body)` returns `body` for any unknown name.
#     That is the whole safety property: a skill added, renamed or retired in
#     the loader can never make this path raise and can never silently DROP a
#     section.
#   * `_SKILL_SECTION_DIETS` holds ONE entry (round 2): `app_html`, which
#     removes redundancy only. Which prose may be CUT is memo D4b (app_html
#     §8 and §11 are EXECUTED by the publish gate and pass through
#     byte-identical). Widening it is RED by construction:
#     `test_the_shipped_mapping_is_app_html_only` pins the entry set,
#     `test_the_shipped_cuts_are_the_reviewed_ones` pins the cut tuples, and
#     `test_the_diet_changes_exactly_the_reviewed_words` pins every word the
#     diet removes or inserts (any cut, reflow or rule-drop change). Each is an
#     INDEPENDENT copy of what review round 2 read; the atom gate is only a
#     necessary condition and misses plain-prose rules (review mutant X3).
#   * flag OFF ⇒ byte-identical to today's `get_all_system_prompt_sections()`.
#     Executable, against the REAL loaded skills, not source-pinned.
#   * flag ON ⇒ no tool definition, name, argument shape or enum changes
#     anywhere. The seam touches system-prompt strings only, and that is the
#     property that makes the eventual flip safe.
#
# LOCAL, this tree, o200k: the sections this seam sits on are 15,092 tokens
# (app_html 11,603 / automations 2,340 / routines 1,149 / triggers 0) and the
# stable layout sends all of them on every turn of every intent. Round 2's
# entry cuts 447 of them (app_html 11,603 → 11,156), and only once the flag
# is flipped.
# ══════════════════════════════════════════════════════════════════════


@pytest.fixture
def prose_diet_on(monkeypatch):
    monkeypatch.setattr(settings, "skill_prose_diet", True, raising=False)


@pytest.fixture
def prose_diet_off(monkeypatch):
    monkeypatch.setattr(settings, "skill_prose_diet", False, raising=False)


class _FakeSkill:
    def __init__(self, section: str):
        self._section = section

    def get_system_prompt_section(self) -> str:
        return self._section


class _FakeLoader:
    """The two members `skill_sections_diet` reads: `.skills` and each
    skill's `get_system_prompt_section()`."""

    def __init__(self, mapping):
        self.skills = {name: _FakeSkill(body) for name, body in mapping.items()}

    def get_all_system_prompt_sections(self):
        return [s.get_system_prompt_section() for s in self.skills.values()]


# ── flag plumbing ──────────────────────────────────────────────────────


def test_skill_prose_diet_defaults_off():
    """Unlike `prompt_diet`, which ships ON.

    `prompt_diet`'s compact text was reviewed alongside its flag; this flag
    would carry whatever a future mapping holds, so the flip has to be its
    own revertible decision rather than a side effect of shipping the seam.
    """
    assert Settings.model_fields["skill_prose_diet"].default is False
    assert Settings.model_fields["skill_prose_diet_canary_user_ids"].default == ""


def test_skill_prose_diet_can_be_turned_on():
    assert Settings(_env_file=None, skill_prose_diet=True).skill_prose_diet is True


def test_skill_prose_helper_reads_settings(monkeypatch):
    monkeypatch.setattr(settings, "skill_prose_diet", False, raising=False)
    monkeypatch.setattr(settings, "skill_prose_diet_canary_user_ids", "", raising=False)
    assert skill_prose_diet_enabled() is False
    monkeypatch.setattr(settings, "skill_prose_diet", True, raising=False)
    assert skill_prose_diet_enabled() is True


def test_the_two_diet_flags_are_independent(monkeypatch):
    """A reader could reasonably assume PROMPT_DIET covers this too. It does
    not, and an accidental `prompt_diet_enabled()` at the skills call site
    would flip the skill prose on 60+ containers that never approved it."""
    monkeypatch.setattr(settings, "prompt_diet", True, raising=False)
    monkeypatch.setattr(settings, "skill_prose_diet", False, raising=False)
    monkeypatch.setattr(settings, "skill_prose_diet_canary_user_ids", "", raising=False)
    assert prompt_diet_enabled() is True
    assert skill_prose_diet_enabled() is False


def test_skill_prose_canary_requires_exact_canonical_user_id(monkeypatch):
    owner = "aaaaaaaa-2222-4333-8444-555555555555"
    monkeypatch.setattr(settings, "skill_prose_diet", False, raising=False)
    monkeypatch.setattr(settings, "skill_prose_diet_canary_user_ids", owner, raising=False)
    assert skill_prose_diet_enabled(owner, "mobile") is True
    assert skill_prose_diet_enabled(owner, "voice") is False
    assert skill_prose_diet_enabled(owner, "web") is False
    assert skill_prose_diet_enabled(owner.upper(), "mobile") is False
    assert skill_prose_diet_enabled(owner[:8], "mobile") is False
    assert skill_prose_diet_enabled("66666666-2222-4333-8444-555555555555", "mobile") is False
    assert skill_prose_diet_enabled(None, "mobile") is False


def test_skill_prose_canary_changes_only_the_selected_users_sections(monkeypatch):
    owner = "aaaaaaaa-2222-4333-8444-555555555555"
    other = "bbbbbbbb-2222-4333-8444-555555555555"
    monkeypatch.setattr(settings, "skill_prose_diet", False, raising=False)
    monkeypatch.setattr(settings, "skill_prose_diet_canary_user_ids", owner, raising=False)
    monkeypatch.setitem(
        prompt_diet_mod._SKILL_SECTION_DIETS, "a", lambda body: "COMPACT"
    )
    loader = _FakeLoader({"a": "body A", "b": "body B"})
    sections = loader.get_all_system_prompt_sections()
    assert skill_sections_diet(loader, sections, owner, "mobile") == ["COMPACT", "body B"]
    assert skill_sections_diet(loader, sections, owner, "voice") == sections
    assert skill_sections_diet(loader, sections, other, "mobile") == sections


# ── skill_section_diet: the pure primitive ─────────────────────────────


def test_the_shipped_mapping_is_app_html_only():
    """Round 2: ONE entry, and it removes redundancy only.

    If you are here because you added a second entry: anything that changes
    what the model is TOLD is memo D4b — an owner decision (NOTES, R2 list).
    A widening of THIS entry lands in
    `test_the_shipped_cuts_are_the_reviewed_ones` /
    `test_the_diet_changes_exactly_the_reviewed_words` instead, with the same
    message. `automations` and `routines` are absent on purpose: no cut in
    them survives the gate.
    """
    assert set(prompt_diet_mod._SKILL_SECTION_DIETS) == {"app_html"}


def test_unknown_skill_name_passes_the_body_through_unchanged():
    """THE safety property. A skill the mapping has never heard of — a new
    builtin, a renamed one, an external skill from `settings.skills_dir` —
    keeps its full section. Never shortened, never dropped, never empty."""
    body = "## some skill\nrules the model needs\n"
    assert skill_section_diet("never_heard_of_it", body) == body
    assert skill_section_diet("", body) == body


def test_known_skill_name_is_replaced(monkeypatch):
    """The mechanism works — proven against a stub, independent of the
    shipped entry, because a test that only ever exercises the passthrough
    proves the seam is inert, not that it is correct."""
    monkeypatch.setitem(
        prompt_diet_mod._SKILL_SECTION_DIETS, "stub", lambda body: "compact"
    )
    assert skill_section_diet("stub", "the full body") == "compact"
    # and its neighbour is still untouched
    assert skill_section_diet("other", "the full body") == "the full body"


def test_a_replacement_that_raises_falls_back_to_the_full_body(monkeypatch):
    """A broken compaction must cost tokens, never capability."""
    def _boom(body):
        raise RuntimeError("bad rule")

    monkeypatch.setitem(prompt_diet_mod._SKILL_SECTION_DIETS, "stub", _boom)
    assert skill_section_diet("stub", "the full body") == "the full body"


@pytest.mark.parametrize("bad", ["", None, 0, b"bytes"])
def test_a_replacement_that_returns_nothing_usable_falls_back(monkeypatch, bad):
    """An empty string is the dangerous one: it would delete the section
    outright and the model would lose the skill's rules with nothing red."""
    monkeypatch.setitem(
        prompt_diet_mod._SKILL_SECTION_DIETS, "stub", lambda body: bad
    )
    assert skill_section_diet("stub", "the full body") == "the full body"


# ── skill_sections_diet: the call-site adapter ─────────────────────────


def test_flag_off_is_byte_identical_even_with_a_mapping_loaded(
    prose_diet_off, monkeypatch
):
    """The byte-identity guard, with the flag-off path given something to
    break: a live mapping entry that WOULD rewrite `a`. Flag off, it does
    not. Without this, byte-identity is only ever asserted against an empty
    mapping, which proves the mapping is empty rather than the flag works."""
    monkeypatch.setitem(
        prompt_diet_mod._SKILL_SECTION_DIETS, "a", lambda body: "REWRITTEN"
    )
    loader = _FakeLoader({"a": "body A", "b": "body B"})
    sections = loader.get_all_system_prompt_sections()
    assert skill_sections_diet(loader, sections) == ["body A", "body B"]


def test_flag_on_applies_the_mapping_by_skill_name(prose_diet_on, monkeypatch):
    monkeypatch.setitem(
        prompt_diet_mod._SKILL_SECTION_DIETS, "a", lambda body: "REWRITTEN"
    )
    loader = _FakeLoader({"a": "body A", "b": "body B"})
    sections = loader.get_all_system_prompt_sections()
    assert skill_sections_diet(loader, sections) == ["REWRITTEN", "body B"]


def test_a_body_the_loader_cannot_attribute_is_passed_through(
    prose_diet_on, monkeypatch
):
    """The names are recovered by re-rendering, so a body that matches no
    skill (a non-deterministic renderer, a caller passing its own list) must
    survive. Dropping it would delete a whole skill's rules silently."""
    monkeypatch.setitem(
        prompt_diet_mod._SKILL_SECTION_DIETS, "a", lambda body: "REWRITTEN"
    )
    loader = _FakeLoader({"a": "body A"})
    out = skill_sections_diet(loader, ["body A", "an orphan body"])
    assert out == ["REWRITTEN", "an orphan body"]


def test_order_and_length_are_preserved(prose_diet_on):
    loader = _FakeLoader({"a": "A", "b": "B", "c": "C"})
    sections = loader.get_all_system_prompt_sections()
    assert skill_sections_diet(loader, sections) == sections


def test_a_broken_loader_leaves_the_sections_alone(prose_diet_on):
    """A diet must never break the prompt. `None` has no `.skills`; the
    raising skill is the loader-side half of the same failure."""
    assert skill_sections_diet(None, ["x", "y"]) == ["x", "y"]

    class _Raises:
        def get_system_prompt_section(self):
            raise RuntimeError("renderer exploded")

    class _BadLoader:
        skills = {"a": _Raises()}

    assert skill_sections_diet(_BadLoader(), ["x"]) == ["x"]


# ── against the REAL loaded skills ─────────────────────────────────────


def _real_loader():
    """Every builtin skill, loaded the way the runner loads them.

    `automations_enabled` is forced on for the same reason
    test_agent_surface_vocabulary does it: otherwise the fattest section
    after app_html never registers and the measurement below is of a
    different prompt than the fleet's.
    """
    from app.agent.skills.loader import SkillLoader

    previous = getattr(settings, "automations_enabled", False)
    settings.automations_enabled = True
    try:
        loader = SkillLoader()
        asyncio.run(loader.load_all())
        return loader
    finally:
        settings.automations_enabled = previous


def test_real_sections_are_byte_identical_when_the_flag_is_off(prose_diet_off):
    """The regression this patch owes the fleet: today's prompt, unchanged.

    Not a re-implementation of the join — the same call `agent_runner.py`
    makes at the `get_all_system_prompt_sections()` site, run through the
    seam, compared byte for byte against the loader's own output.
    """
    loader = _real_loader()
    before = loader.get_all_system_prompt_sections()
    after = skill_sections_diet(loader, loader.get_all_system_prompt_sections())
    assert after == before
    assert "\n\n".join(after) == "\n\n".join(before)


def test_real_sections_flag_on_touch_app_html_and_nothing_else(prose_diet_on):
    """Round 2: flag ON shortens `app_html` and leaves every other section
    byte-identical, in the same order and number. (Round 1 asserted all of
    them identical, because the mapping was empty.)"""
    loader = _real_loader()
    names = list(loader.skills)
    before = loader.get_all_system_prompt_sections()
    after = skill_sections_diet(loader, loader.get_all_system_prompt_sections())
    assert len(after) == len(before)
    rendered = {
        n: loader.skills[n].get_system_prompt_section() for n in names
    }
    for b, a in zip(before, after):
        owner = next(n for n in names if rendered[n] == b)
        if owner == "app_html":
            assert a != b, "the app_html diet did not apply"
            assert len(a) < len(b)
        else:
            assert a == b, f"{owner} changed under the skill prose diet"


def test_no_skill_section_is_ever_emptied(prose_diet_on):
    """Belt to the braces above: every section that went in comes out with
    content. A silently emptied section is the failure that would look like
    'the model forgot how to build apps'."""
    loader = _real_loader()
    out = skill_sections_diet(loader, loader.get_all_system_prompt_sections())
    assert out, "no sections loaded — the assertions below would be vacuous"
    assert all(isinstance(s, str) and s.strip() for s in out)


def test_flag_on_changes_no_tool_definition_name_shape_or_enum(monkeypatch):
    """The property that makes this seam safe to flip at all.

    The skill prose diet touches system-prompt strings. It is reachable from
    no tool-definition path, so names, descriptions, property names, types,
    enums and `required` lists are identical in both flag states. Compared
    with descriptions INCLUDED — unlike the `prompt_diet` contract above,
    this seam may not change even a description.
    """
    monkeypatch.setattr(settings, "skill_prose_diet", False, raising=False)
    off = json.dumps(_real_loader().get_all_tool_definitions(), sort_keys=True)
    monkeypatch.setattr(settings, "skill_prose_diet", True, raising=False)
    on_loader = _real_loader()
    on = json.dumps(on_loader.get_all_tool_definitions(), sort_keys=True)
    assert off == on

    # Control: this comparison can fail. `prompt_diet` moves the same blob,
    # so an `off == on` that is true no matter what would be caught here.
    monkeypatch.setattr(settings, "prompt_diet", False, raising=False)
    full = json.dumps(_real_loader().get_all_tool_definitions(), sort_keys=True)
    assert full != on, "tool defs are identical with PROMPT_DIET off too — " \
                       "this comparison cannot detect a change"


# ── the SEMANTICS GATE: a NECESSARY condition over rule-bearing atoms ──
#
# Round 2 put real text in `_SKILL_SECTION_DIETS`. Its contract is that it
# removes REDUNDANCY only, and a contract nobody can check is a hope. So: for
# every dieted section, extract the rule-bearing atoms of the ORIGINAL and
# require every one of them in the DIETED text. What this proves is scoped:
# no atom of the classes below is lost. It does NOT prove no rule was lost —
# a plain-prose rule with no modal word, span, bold or number ("Read it
# before you decide what to change.") is invisible to it (review round 2,
# gapprobe: 10 of 10 such rules missed; mutant X3). Bold spans were added
# after RB ("**Fix the class, not the instance.**").
# The atom classes, each required in the dieted text:
#
#   * every fenced code block and every table row, VERBATIM (a block can be
#     the only source of an exact value the model reproduces — memo §4);
#   * app_html §8 and §11, VERBATIM (the publish gate executes them);
#   * every heading; every sentence containing must / never / always /
#     do not / don't / only / required (case-insensitive, whitespace-
#     normalised, as a substring of the whole normalised dieted text);
#   * every backticked span and every **bold** span (normalised substring);
#   * every tool / function / flag / chip name, URL and path (set
#     containment);
#   * every number with its unit — as a MULTISET, because "44" in two rules is
#     two constraints and removing one of them is a rule removed.
#
# The other half is the RETAINED-TWIN check in `prompt_diet._check_twins`,
# pinned further down: each cut names the sentences that still carry it, and
# the diet serves the full body unless all of them are in the output, each in
# its declared region. What neither half sees (a plain-prose rule, review X3)
# is covered only by pinning the reviewed diff itself — see
# `test_the_diet_changes_exactly_the_reviewed_words`.
#
# Whitespace is normalised on both sides, so a soft-wrap reflow passes and a
# dropped word does not. A cut this gate refuses is not redundancy — it is an
# owner decision (NOTES, R2).

import re as _re
from collections import Counter as _Counter

_FENCE_RE = _re.compile(r"^[ \t]*(```|~~~).*?^[ \t]*\1[ \t]*$", _re.S | _re.M)
_MODAL_RE = _re.compile(
    r"\b(must|never|always|do not|don't|only|required)\b", _re.I
)
_NUMBER_RE = _re.compile(
    r"(?<![\w#.-])\d+(?:[.,:]\d+)*(?:\s?(?:px|ms|pt|dp|kg|rem|dvh|vh)\b|%)?"
)
_IDENT_RES = (
    _re.compile(r"\b[a-z][a-z0-9]*(?:__?[a-z0-9]+)+\b"),        # snake / tool
    _re.compile(r"\b[a-z]+[A-Z][A-Za-z0-9]*\b"),                # camelCase
    _re.compile(r"\b[A-Z][a-z0-9]+[A-Z][A-Za-z0-9]*\b"),        # PascalCase
    _re.compile(r"\b[A-Z]{2,}(?:_[A-Z0-9]+)+\b"),               # FLAG_NAME
    _re.compile(r"\[\[[^\]]+\]\]"),                              # chips
)
_URL_RE = _re.compile(r"https?://[^\s`)\"']+")
_PATH_RES = (
    _re.compile(r"(?<![\w:/.])/[A-Za-z][\w.-]*(?:/[\w.*-]+)+"),
    _re.compile(r"\b[\w-]+\.(?:md|html|py|js|json)\b"),
)


def _norm(s: str) -> str:
    return " ".join(s.split())


def _sentences(prose: str):
    out = []
    for para in _re.split(r"\n[ \t]*\n", prose):
        items, cur = [], []
        for line in para.split("\n"):
            s = line.strip()
            starts = s.startswith(("#", "|", "- ", "* ", "+ ", "> ")) or bool(
                _re.match(r"\d+\.\s", s)
            )
            if starts and cur:
                items.append(" ".join(cur))
                cur = []
            cur.append(s)
            if s.startswith(("#", "|")):
                items.append(" ".join(cur))
                cur = []
        if cur:
            items.append(" ".join(cur))
        for it in items:
            out.extend(x for x in _re.split(r"(?<=[.!?])\s+", _norm(it)) if x)
    return out


def _atoms(text: str) -> dict:
    fences = [m.group(0) for m in _FENCE_RE.finditer(text)]
    prose = _FENCE_RE.sub("\n", text)
    lines = prose.split("\n")
    return {
        "fence": set(fences),
        "table_row": {l for l in lines if l.lstrip().startswith("|")},
        "heading": {_norm(l) for l in lines if l.lstrip().startswith("#")},
        "modal": {s for s in _sentences(prose) if _MODAL_RE.search(s)},
        "backtick": set(_re.findall(r"`[^`\n]+`", prose)),
        # a bold span is how this document marks a rule that has no modal
        # word ("**Fix the class, not the instance.**" — review mutant RB)
        "bold": set(_re.findall(r"\*\*[^*\n][^*]*?\*\*", prose)),
        "ident": {m for r in _IDENT_RES for m in r.findall(prose)},
        "url": set(_URL_RE.findall(prose)),
        "path": {m for r in _PATH_RES for m in r.findall(prose)},
        "number": _Counter(
            _re.sub(r"\s", "", n) for n in _NUMBER_RE.findall(prose)
        ),
    }


def _missing_atoms(original: str, dieted: str) -> dict:
    """{kind: [atoms of `original` absent from `dieted`]}; empty ⇒ nothing lost."""
    a, b = _atoms(original), _atoms(dieted)
    flat = _norm(dieted)
    missing = {}
    # verbatim kinds. A fence is a byte-for-byte substring; a table row must
    # still be a WHOLE LINE — two rows joined onto one line would each still
    # be a substring, and the table would be gone.
    lost = sorted(x for x in a["fence"] if x not in dieted)
    if lost:
        missing["fence"] = lost
    dieted_lines = set(dieted.split("\n"))
    lost = sorted(x for x in a["table_row"] if x not in dieted_lines)
    if lost:
        missing["table_row"] = lost
    # normalised-substring kinds
    for kind in ("heading", "modal", "backtick", "bold"):
        lost = sorted(x for x in a[kind] if _norm(x) not in flat)
        if lost:
            missing[kind] = lost
    for kind in ("ident", "url", "path"):
        lost = sorted(a[kind] - b[kind])
        if lost:
            missing[kind] = lost
    lost_n = a["number"] - b["number"]
    if lost_n:
        missing["number"] = sorted(lost_n.elements())
    return missing


def _app_html_protected_spans(body: str):
    """§8 (its heading up to §9's) and §11 (its heading to the end)."""
    doc = body[body.index("\n# Toup frontend design\n"):]
    i8, i9, i11 = (doc.index(a) for a in ("\n## 8. ", "\n## 9. ", "\n## 11. "))
    return doc[i8:i9], doc[i11:]


def _real_pairs():
    loader = _real_loader()
    pairs = []
    for name, skill in loader.skills.items():
        body = skill.get_system_prompt_section() or ""
        if body:
            pairs.append((name, body, skill_section_diet(name, body)))
    return pairs


def test_every_rule_bearing_atom_survives_the_diet():
    """THE gate. For every section the diet touches, original ⊆ dieted."""
    pairs = _real_pairs()
    assert {n for n, _, _ in pairs} >= {"app_html", "automations", "routines"}
    changed = [(n, b, d) for n, b, d in pairs if b != d]
    assert [n for n, _, _ in changed] == ["app_html"], (
        "the gate would be vacuous: nothing was dieted"
    )
    for name, body, dieted in pairs:
        missing = _missing_atoms(body, dieted)
        assert not missing, f"{name}: the diet removed rule-bearing atoms: {missing}"


def test_the_gate_extracts_a_non_trivial_atom_set():
    """Anti-vacuity for the extractor itself: a gate that finds no atoms
    passes every diet. Floors well under today's counts (LOCAL, round 2)."""
    body = next(b for n, b, _ in _real_pairs() if n == "app_html")
    a = _atoms(body)
    assert len(a["fence"]) >= 15
    assert len(a["table_row"]) >= 20
    assert len(a["heading"]) >= 30
    assert len(a["modal"]) >= 50
    assert len(a["backtick"]) >= 80
    assert len(a["bold"]) >= 40
    assert "app_html__create_app_file" in a["ident"]
    assert "https://cdnjs.cloudflare.com" in a["url"]
    assert a["number"]["44"] >= 5


def test_the_gate_goes_red_when_an_atom_is_dropped():
    """In-test control: each kind of atom, removed once from the REAL dieted
    text, must be reported. The external mutation run (NOTES) does the same
    to the diet rules themselves."""
    body = next(b for n, b, _ in _real_pairs() if n == "app_html")
    dieted = skill_section_diet("app_html", body)
    drops = {
        "modal": "Do not ask them which one they meant.",
        "backtick": "`view_app_file` first, every time",
        "heading": "## 10. Changing an app someone is already holding",
        "number": "Mobile-first: 360px, then 768, then 1280.",
        "ident": "Never hand-scaffold an app with exec/write_file.",
        "bold": "**Fix the class, not the instance.**",
    }
    for kind, needle in drops.items():
        assert dieted.count(needle) == 1, needle
        missing = _missing_atoms(body, dieted.replace(needle, ""))
        assert kind in missing, f"dropping {needle!r} was not caught as {kind}"


def test_app_html_sections_8_and_11_are_byte_identical():
    """R3: the publish gate EXECUTES §8b (the state diagram) and §11 (before
    present_app). Not one byte of either may change — whitespace included."""
    body = next(b for n, b, _ in _real_pairs() if n == "app_html")
    dieted = skill_section_diet("app_html", body)
    for span in _app_html_protected_spans(body):
        assert span in dieted
        assert len(span) > 2000


def test_app_html_diet_is_deterministic_and_idempotent_in_identity():
    """Pure and turn-independent: one input, one output, every call — the
    same-lineage property the stable layout exists for."""
    body = next(b for n, b, _ in _real_pairs() if n == "app_html")
    first = skill_section_diet("app_html", body)
    assert all(skill_section_diet("app_html", body) == first for _ in range(3))


@pytest.mark.parametrize(
    "drift",
    [
        lambda b: b.replace("\n# Toup frontend design\n", "\n# Frontend design\n"),
        lambda b: b.replace("\n## 8. ", "\n## Eight. "),
        lambda b: b.replace("\n## 11. ", "\n## Eleven. "),
        lambda b: b.replace("So, before you edit:", "Before editing:"),
        lambda b: b.replace(" It runs in a sandboxed frame", " It runs in a frame"),
        lambda b: b.replace("\n## 2. ", "\n```\nunclosed fence\n\n## 2. ", 1),
    ],
    ids=["doc-marker", "sec8", "sec11", "sec10-anchor", "head-anchor", "fence"],
)
def test_a_drifted_document_gets_the_full_body_back(drift):
    """Every anchor is exact-once. When the design document or the head moves
    under the diet, the section is served WHOLE. This covers drift of the text being CUT;
    drift of the text a cut RELIES ON is the twin tests below."""
    body = next(b for n, b, _ in _real_pairs() if n == "app_html")
    drifted = drift(body)
    assert drifted != body
    assert skill_section_diet("app_html", drifted) == drifted


# ── retained twins: the copy each cut relies on must still be there ─────


def _remove_normalised(body: str, twin: str) -> str:
    """Delete `twin` from `body` wherever its words sit, across soft wraps."""
    pat = r"\s+".join(_re.escape(w) for w in twin.split())
    out, n = _re.subn(pat, "", body, count=1)
    assert n == 1, f"twin not found in the real body: {twin!r}"
    return out


def _app_html_body():
    return next(b for n, b, _ in _real_pairs() if n == "app_html")


def test_every_twin_is_in_the_real_body_and_the_diet_applies():
    """Anti-vacuity: each twin exists today, so the check below is live, and
    today's document still diets (a twin check that always raised would pass
    every drift test and save nothing)."""
    body = _app_html_body()
    flat = _norm(body)
    assert len(prompt_diet_mod._APP_HTML_TWINS) >= 10
    for _, twin in prompt_diet_mod._APP_HTML_TWINS:
        assert _norm(twin) in flat, twin
    assert skill_section_diet("app_html", body) != body


# An INDEPENDENT copy of the justification NOTES R2.3 cites for each cut.
# Deleting a twin from the module weakens the fallback silently (its drift
# test simply stops being generated), so the declared set is pinned here.
_CITED_TWINS = {
    # R1-a, carried by §7
    ("sec7", "The app runs in a sandboxed frame with an **opaque origin**"),
    ("sec7", "The runner replaces all three before your code runs, with objects that cannot throw"),
    ("sec7", "a read taken during first paint returns `null` even when a value exists."),
    ("sec7", "Seed the UI from defaults immediately and reconcile when the data lands"),
    ("sec7", "Still genuinely unavailable in the sandbox: network requests"),
    ("sec7", "top-level navigation, popups, and the parent page."),
    # R1-b, carried by the head and §10 item 2
    ("head", "\"Make the button bigger\" is about the control the person was USING when they said it."),
    ("head", "not the PLAY button on the start screen, which they pressed once."),
    ("head", "A person who says a control is too small and gets a bigger menu button has been answered with the wrong object, and has to ask again."),
    ("head", "change the thing the words are actually about, not the first match for them."),
    ("head", "do not narrate the problem and do not claim the change."),
    ("sec10", "In a game the controls are the D-pad / paddle / fire button; `PLAY`, `RESTART` and menu items are chrome."),
    # R1-c, carried by the head
    ("head", "CHANGE THEM ALL: every control of that kind, in the same way, in one round of edits."),
    ("head", "Widening the change is nearly free; guessing wrong costs a whole turn."),
}


def test_the_declared_twins_cover_every_cited_justification():
    assert _CITED_TWINS <= set(prompt_diet_mod._APP_HTML_TWINS)


def test_a_twin_moved_out_of_its_region_gets_the_full_body_back():
    """Review round 2, N3 / mutant X6: the region half of the twin check. The
    §10 twin moved into the head is still somewhere in the section, but not
    where R1-b relies on it (beside the list it explains), so the diet must
    fall back rather than accept any occurrence anywhere."""
    twin = (
        "In a game the controls are the D-pad / paddle / fire button; `PLAY`, "
        "`RESTART` and menu items are chrome."
    )
    body = _app_html_body()
    mark = prompt_diet_mod._APP_HTML_DOC_MARK
    moved = _remove_normalised(body, twin).replace(mark, "\n" + twin + "\n" + mark, 1)
    assert _norm(twin) in _norm(moved)
    assert skill_section_diet("app_html", moved) == moved


# ── the reviewed diff, pinned (review round 2, B2 / mutant X3) ─────────
#
# The atom gate and the twin check are NECESSARY conditions: X3 added a head
# cut that deleted a plain-prose rule stated nowhere else, and all 73 tests
# stayed green. What makes widening the `app_html` entry red is an INDEPENDENT
# copy of what the reviewers read. If you are here because one of these went
# red: the change you made alters what the model is told. Each new cut needs
# its own retained-twin entry, its own review, and a NOTES row; anything that
# is not pure redundancy is memo D4b, an owner decision.

_REVIEWED_HEAD_CUTS = (
    (
        " It runs in a sandboxed frame on an opaque origin: there is no network, "
        "no navigation and no parent page. Storage cannot throw (the runner "
        "replaces it), but it is not durable within a first paint either, so "
        "seed the UI from in-memory defaults and reconcile after.",
        "",
    ),
)

_REVIEWED_SEC10_CUTS = (
    (
        "A change request is about the thing the person was *using*, and they will\n"
        "describe it with the shortest word that fits. \"Make the button bigger\", said\n"
        "about a game, means the buttons they were pressing to play it.\n"
        "\n"
        "This went wrong exactly that way: asked to make the button bigger on a Snake\n"
        "with a D-pad, the edit landed on the start screen's `PLAY` button — pressed\n"
        "once, already large enough, and the element in the file that most literally\n"
        "answers to the word \"button\". The D-pad, pressed hundreds of times and\n"
        "genuinely too small, was untouched. The app came back with the same defect and\n"
        "a message saying it had been fixed.\n"
        "\n"
        "So, before you edit:",
        "Before you edit:",
    ),
    (
        "3. **If more than one answer is reasonable, change them all** — every control\n"
        "   of that kind, the same way, in one round of edits. Widening the change is\n"
        "   nearly free; guessing wrong costs the person another turn. Do not ask them\n"
        "   which one they meant.",
        "3. **If more than one answer is reasonable, change them all**. Do not ask them\n"
        "   which one they meant.",
    ),
)

#: (removed words, inserted words), in document order, for OFF → ON over the
#: real app_html section, whitespace-split. The reflow is whitespace-only, so
#: it contributes nothing here; anything it ever does to a WORD shows up.
_REVIEWED_WORD_CHANGES = (
    [(
        "It runs in a sandboxed frame on an opaque origin: there is no network, "
        "no navigation and no parent page. Storage cannot throw (the runner "
        "replaces it), but it is not durable within a first paint either, so "
        "seed the UI from in-memory defaults and reconcile after.",
        "",
    )]
    + [("---", "")] * 9
    + [
        (
            "A change request is about the thing the person was *using*, and they "
            "will describe it with the shortest word that fits. \"Make the button "
            "bigger\", said about a game, means the buttons they were pressing to "
            "play it. This went wrong exactly that way: asked to make the button "
            "bigger on a Snake with a D-pad, the edit landed on the start screen's "
            "`PLAY` button — pressed once, already large enough, and the element in "
            "the file that most literally answers to the word \"button\". The D-pad, "
            "pressed hundreds of times and genuinely too small, was untouched. The "
            "app came back with the same defect and a message saying it had been "
            "fixed. So, before",
            "Before",
        ),
        (
            "all** — every control of that kind, the same way, in one round of "
            "edits. Widening the change is nearly free; guessing wrong costs the "
            "person another turn.",
            "all**.",
        ),
        ("---", ""),
    ]
)


def _word_changes(before: str, after: str):
    import difflib

    a, b = before.split(), after.split()
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return [
        (" ".join(a[i1:i2]), " ".join(b[j1:j2]))
        for tag, i1, i2, j1, j2 in sm.get_opcodes()
        if tag != "equal"
    ]


def test_the_shipped_cuts_are_the_reviewed_ones():
    assert prompt_diet_mod._APP_HTML_HEAD_CUTS == _REVIEWED_HEAD_CUTS
    assert prompt_diet_mod._APP_HTML_SEC10_CUTS == _REVIEWED_SEC10_CUTS


def test_the_diet_changes_exactly_the_reviewed_words():
    """Every word the shipped diet removes or inserts, against the REAL
    section. Catches a widening through any mechanism — a new cut, a longer
    cut, a reflow or rule-drop that touches a word — including a plain-prose
    rule that no atom class sees (X3)."""
    body = _app_html_body()
    got = _word_changes(body, skill_section_diet("app_html", body))
    assert got == _REVIEWED_WORD_CHANGES


def test_the_word_pin_goes_red_on_a_plain_prose_cut():
    """In-test control, the X3 shape: one more head cut deleting a rule with no
    modal word, span, bold or number. The atom gate cannot see it; the word
    pin must."""
    body = _app_html_body()
    dieted = skill_section_diet("app_html", body)
    rule = "Read it before you decide what to change."
    assert dieted.count(rule) == 1
    widened = dieted.replace(rule, "", 1)
    assert not _missing_atoms(body, widened), "X3 shape: invisible to the atoms"
    assert _word_changes(body, widened) != _REVIEWED_WORD_CHANGES


@pytest.mark.parametrize(
    "twin", [t for _, t in prompt_diet_mod._APP_HTML_TWINS],
    ids=[f"{r}-{i}" for i, (r, _) in enumerate(prompt_diet_mod._APP_HTML_TWINS)],
)
def test_a_drifted_twin_gets_the_full_body_back(twin):
    """Someone trims the retained copy "because the other one says it". With
    the diet on, that would delete the rule from the prompt entirely. It must
    instead cost tokens: the full (drifted) body is served."""
    drifted = _remove_normalised(_app_html_body(), twin)
    assert skill_section_diet("app_html", drifted) == drifted


def test_review_case_1_section_7_shortened_serves_the_full_body():
    """Round-2 review, twin_drift.py case 1: §7's opaque-origin/throw paragraph
    and its seed-and-reconcile sentence edited out, every anchor intact."""
    body = _app_html_body()
    drifted = _remove_normalised(
        body,
        "The app runs in a sandboxed frame with an **opaque origin**, where "
        "`localStorage`, `sessionStorage` and `document.cookie` would normally "
        "**throw** on the first access and take your whole script down with them.",
    )
    drifted = _remove_normalised(
        drifted,
        "Seed the UI from defaults immediately and reconcile when the data lands:",
    )
    out = skill_section_diet("app_html", drifted)
    assert out == drifted
    assert "reconcile" in out


def test_review_case_2_head_shortened_serves_the_full_body():
    """Round-2 review, twin_drift.py case 2: the head's CHANGE-THEM-ALL clause
    and "Widening the change is nearly free…" edited out of skill.py."""
    body = _app_html_body()
    drifted = _remove_normalised(
        body,
        "CHANGE THEM ALL: every control of that kind, in the same way, in one "
        "round of edits.",
    )
    drifted = _remove_normalised(
        drifted,
        "Widening the change is nearly free; guessing wrong costs a whole turn.",
    )
    out = skill_section_diet("app_html", drifted)
    assert out == drifted
    assert "in one round of edits" in _norm(out)


def test_a_cut_that_removes_a_twin_disables_the_diet(monkeypatch):
    """Mutant RC, as a test: a new head cut that deletes the very sentence
    R1-c relies on. It must not ship a prompt missing the rule — the diet
    falls back, and the atom-gate test's "nothing was dieted" goes red."""
    twin = (
        "CHANGE THEM ALL: every control of that kind, in the same way, in one "
        "round of edits."
    )
    monkeypatch.setattr(
        prompt_diet_mod, "_APP_HTML_HEAD_CUTS",
        prompt_diet_mod._APP_HTML_HEAD_CUTS + ((twin, ""),),
    )
    body = _app_html_body()
    assert skill_section_diet("app_html", body) == body


def test_unrecognised_code_blocks_serve_the_full_body():
    """`~~~` fences and four-space indented code are not what the reflow was
    written against; joining either would rewrite code (review N7)."""
    body = _app_html_body()
    for block in ("~~~\nlet a = 1;\nlet b = 2;\n~~~", "    let a = 1;\n    let b = 2;"):
        drifted = body.replace("\n## 2. ", "\n" + block + "\n\n## 2. ", 1)
        assert drifted != body
        assert skill_section_diet("app_html", drifted) == drifted


def test_a_rule_inside_a_fence_is_code_not_a_separator():
    """`_drop_rules` only removes a bare `---` between blank lines OUTSIDE a
    fence (review N7); inside one it is content and stays."""
    fenced = "text\n\n```\na\n\n---\n\nb\n```\n\nmore"
    assert prompt_diet_mod._drop_rules(fenced) == fenced
    assert prompt_diet_mod._drop_rules("a\n\n---\n\nb") == "a\n\nb"


# ── the runner seam, source-pinned ─────────────────────────────────────


def test_runner_gates_the_skill_prose_diet_and_nothing_else():
    """Order matters here the way it does at the two `prompt_diet` sites.

    (1) the loader's own join is used when both owner prose flags are off;
    the static app_html transform has an independent exact-owner gate and
    takes precedence when both are armed;
    (2) that call is gated on the helper, not a raw settings read;
    (3) the gate sits ABOVE the `if skill_parts:` join, so a diet that
    emptied the list could not slip past it.
    """
    src = (
        Path(__file__).resolve().parents[1] / "app" / "agent" / "agent_runner.py"
    ).read_text()
    load = src.find("skill_parts = self.skill_loader.get_all_system_prompt_sections()")
    gate = src.find("if _skill_prose_diet_enabled(user_id, channel):")
    swap = src.find("skill_parts = _skill_sections_diet(")
    join = src.find('section_parts["skills"] = "\\n\\n".join(skill_parts)')
    assert load != -1 and gate != -1 and swap != -1 and join != -1
    assert load < gate < swap < join, (
        "the skill-prose seam moved: the loader call, its gate, the swap and "
        "the join must stay in that order"
    )
    # The static transform changes one list entry in place; no other path may
    # replace the loader's list before the join.
    assert src.count("skill_parts = ") == 2
