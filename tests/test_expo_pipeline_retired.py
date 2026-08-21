"""P0 2026-08-21 — the Expo app pipeline is retired IN CODE, not by a flag.

Background
----------
The Expo builder was made "off by default" at 01:49 on 2026-08-21. That
default could not, and did not, stop a container from running the old
pipeline, because a default is not a live value — three independent things
outranked it:

  1. a bridge-env pin (`APP_BUILDER_EXPO_ENABLED=1`, set explicitly during
     the canary) forwarded into every container `pool_addon` spawned;
  2. any container built from an image predating the flip, which carries
     the old `= True` compiled in until it is recreated;
  3. `AGENT_TOOL_FAMILIES`, which gates the FAMILY and not the pipeline, so
     it could not close either gap.

Distinguishing those needs a shell on the host, and that host's SSH key was
rotated on 2026-08-20. So the gate stopped being a value to look up and
became `tool_entitlements.EXPO_PIPELINE_RETIRED`.

What this file pins
-------------------
That there is NO configuration — env var, settings field, entitlement
string, or combination — under which the Expo pipeline loads, its tools
reach the wire array, or its prompt text reaches the model. Every test
below sets the MOST permissive posture it can and asserts Expo is still
gone; a test that merely used the default posture would pass with the
whole retirement reverted.

The Expo code itself is deliberately still in the tree (it is the
rollback, and `/api/apps` still serves apps built with it) — so "is it
reachable" is the only question these tests can ask.
"""

from __future__ import annotations

import pytest

from app.agent import tool_entitlements as te
from app.agent.skills.loader import SkillLoader
from app.config import settings

_EXPO_SKILLS = ("app_builder", "app")
_EXPO_TOOL_PREFIXES = ("app_builder__", "app__")


@pytest.fixture(autouse=True)
def _restore():
    yield
    settings.agent_tool_families = "*"
    settings.app_builder_expo_enabled = type(settings)().app_builder_expo_enabled
    settings.app_html_enabled = type(settings)().app_html_enabled
    te.reset_cache_for_tests()


def _most_permissive(monkeypatch) -> None:
    """Everything an operator could turn on, turned on."""
    monkeypatch.setenv("APP_BUILDER_EXPO_ENABLED", "1")
    settings.app_builder_expo_enabled = True
    settings.app_html_enabled = True
    settings.agent_tool_families = "*"
    te.reset_cache_for_tests()


# ── 1. The constant, and the gate it drives ──────────────────────────

def test_the_retirement_constant_is_on():
    """The one line the rest of this file depends on. If someone flips it,
    every assertion below fails loudly rather than silently going vacuous."""
    assert te.EXPO_PIPELINE_RETIRED is True
    assert te.RETIRED_SKILLS == frozenset({"app_builder", "app"})


def test_no_env_or_setting_can_enable_the_expo_pipeline(monkeypatch):
    _most_permissive(monkeypatch)
    assert te.pipeline_enabled("expo") is False
    # Control: the HTML pipeline is still switchable, so this is a real gate
    # and not a function that returns False for everything.
    assert te.pipeline_enabled("html") is True


@pytest.mark.parametrize("families", ["*", "app_builder", "app_builder,toup", "none"])
def test_no_entitlement_string_can_enable_the_expo_skills(families, monkeypatch):
    _most_permissive(monkeypatch)
    settings.agent_tool_families = families
    te.reset_cache_for_tests()
    for skill in _EXPO_SKILLS:
        assert not te.skill_enabled(skill), f"{skill} enabled under {families!r}"


def test_skill_enabled_is_still_a_real_function(monkeypatch):
    """Control for the parametrized test above — `skill_enabled` must return
    True for something, or it proves nothing about `app_builder`."""
    _most_permissive(monkeypatch)
    assert te.skill_enabled("app_html") is True
    settings.agent_tool_families = "doc_generation"
    te.reset_cache_for_tests()
    assert te.skill_enabled("app_html") is False


# ── 2. The loader — both entry points ────────────────────────────────

@pytest.mark.asyncio
async def test_filesystem_discovery_never_loads_the_expo_skill(tmp_path, monkeypatch):
    monkeypatch.setenv("TOUP_HTML_APPS_DIR", str(tmp_path / "apps"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    _most_permissive(monkeypatch)

    loader = SkillLoader(extra_dirs=[str(tmp_path)])
    await loader.load_all()

    for skill in _EXPO_SKILLS:
        assert skill not in loader.skills
    names = {t["name"] for t in loader.get_all_tool_definitions()}
    assert not any(n.startswith(_EXPO_TOOL_PREFIXES) for n in names), sorted(names)
    # …and the replacement really is there, so the absence above is a gate
    # and not a loader that failed to load anything at all.
    assert "app_html" in loader.skills
    assert "app_html__create_app_file" in names


@pytest.mark.asyncio
async def test_the_expo_skill_module_is_not_even_imported(tmp_path, monkeypatch):
    """`load_all` skips the directory before `_load_skill_from_file` execs it.

    `app_builder/skill.py` is 278 KB and its import pulls in the AppManager /
    Metro machinery. Rejecting it at `_register` would still pay for that.
    """
    import sys

    monkeypatch.setenv("TOUP_HTML_APPS_DIR", str(tmp_path / "apps"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    _most_permissive(monkeypatch)
    sys.modules.pop("toup_skill_app_builder", None)

    loader = SkillLoader(extra_dirs=[str(tmp_path)])
    await loader.load_all()

    assert "toup_skill_app_builder" not in sys.modules
    # Control: the skill the loader DID load left its module behind, so the
    # assertion above is about the skip and not about the naming scheme.
    assert "toup_skill_app_html" in sys.modules


@pytest.mark.asyncio
async def test_register_dynamic_refuses_the_expo_skills(tmp_path, monkeypatch):
    """agent_main's late-bound path. `register_dynamic` is how AppBuilderSkill
    and AppGatewaySkill actually reach the loader in production — they take
    constructor args, so filesystem discovery never instantiates them. A
    retirement that only covered `load_all` would miss the real route.
    """
    from app.agent.skills.base import Skill, SkillMeta

    monkeypatch.setenv("TOUP_HTML_APPS_DIR", str(tmp_path / "apps"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    _most_permissive(monkeypatch)

    def _stub(skill_name: str):
        class _Stub(Skill):
            meta = SkillMeta(
                name=skill_name, version="1.0.0",
                description="stub", author="test",
            )

            def get_tools(self):
                return [{"name": f"{skill_name}__build_app", "description": "x"}]

            async def execute_tool(self, tool_name, args, ctx):
                return "ok"

        return _Stub()

    loader = SkillLoader(extra_dirs=[str(tmp_path)])
    for skill_name in _EXPO_SKILLS:
        assert await loader.register_dynamic(_stub(skill_name)) is False
        assert skill_name not in loader.skills

    # Control: the same path accepts a skill that is not retired.
    assert await loader.register_dynamic(_stub("app_html_probe")) is True


# ── 3. Nothing in the prompt or the guards points at Expo ────────────

@pytest.mark.asyncio
async def test_no_prompt_section_mentions_the_expo_flow(tmp_path, monkeypatch):
    """The user-visible half of the bug. The Expo prompt is what produced the
    interrogation ("3 direction cards", then 5-7 researched questions, then
    5+ technical ones) and the scaffold/npm/GitHub/dev-server narration.
    """
    monkeypatch.setenv("TOUP_HTML_APPS_DIR", str(tmp_path / "apps"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    _most_permissive(monkeypatch)

    loader = SkillLoader(extra_dirs=[str(tmp_path)])
    await loader.load_all()
    blob = "\n".join(loader.get_all_system_prompt_sections()).lower()

    assert blob, "no prompt sections at all — the assertions below are vacuous"
    for banned in (
        "expo", "react native", "create-expo-app", "npm install",
        "app_builder__", "research_category", "direction cards",
        "github repo", "dev server", "metro", "clarifying questions",
        "agent placeholder", "package.json",
    ):
        assert banned not in blob, f"prompt still mentions {banned!r}"

    # NOT banned, and deliberately so — checking both keeps this test honest
    # about what it is asserting:
    #   * "scaffold" appears as `toup__scaffold`, a separate family's tool for
    #     a multi-file project the USER owns and runs; it never builds a Toup
    #     app. It also appears in the HTML section's own prohibition ("no
    #     scaffold, no bundler"), where banning the word would forbid the
    #     instruction that prevents the behaviour.
    #   * the app route must still name the pipeline that exists.
    assert "app_html__create_app_file" in blob


@pytest.mark.asyncio
async def test_the_html_prompt_forbids_asking_questions_first(tmp_path, monkeypatch):
    """"Build me a snake game for nokia" must produce a game, not a survey."""
    monkeypatch.setenv("TOUP_HTML_APPS_DIR", str(tmp_path / "apps"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    _most_permissive(monkeypatch)

    loader = SkillLoader(extra_dirs=[str(tmp_path)])
    await loader.load_all()
    section = loader.get_skill("app_html").get_system_prompt_section()

    assert "BUILD IT IMMEDIATELY" in section
    assert "Do NOT ask the user questions first" in section
    assert "cdnjs.cloudflare.com" in section
    assert "app_html__create_app_file" in section
    # The design pass is mandatory and names a real path.
    assert "read_file(path=" in section
    assert "toup-frontend-design.md" in section


def test_the_exec_guard_can_never_emit_the_expo_redirect(monkeypatch):
    """`_PIPELINE_REDIRECT_MSG` tells the model to "Ask the user 10+
    clarifying questions". A redirect that resurrects the interrogation is
    the same defect as a tool that does."""
    from app.agent.tool_executor import _pipeline_redirect_msg

    for html in (True, False):
        _most_permissive(monkeypatch)
        settings.app_html_enabled = html
        te.reset_cache_for_tests()
        msg = _pipeline_redirect_msg()
        assert "app_builder__build_app" not in msg
        assert "10+ clarifying questions" not in msg
        assert "app_html__create_app_file" in msg
