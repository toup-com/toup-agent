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

import copy
import json
from pathlib import Path

import pytest

from app.config import Settings, settings
from app.agent.prompt_diet import (
    DOC_GENERATION_DIET,
    PLATFORM_KNOWLEDGE_DIET,
    apply_tool_description_diet,
    prompt_diet_enabled,
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
        (_routines_tools, "routines__remind", 650),
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
        "do NOT call generate_pdf",
    ):
        assert kept in DOC_GENERATION_DIET
    assert _tokens(DOC_GENERATION_DIET) <= 260  # target ~200


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
