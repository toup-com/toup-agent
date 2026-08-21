"""Round 12 — pipeline selection, and that it selects the whole pipeline.

`APP_BUILDER_EXPO_ENABLED` / `APP_HTML_ENABLED` decide which app-building
skills a container loads. The failure mode this guards is the *half-gated*
state `SkillLoader._register` warns about: tools listed but not executable,
or executable but absent from the capabilities listing. A skill must arrive
or leave with its tool definitions, its system-prompt section AND its
execution path together.

It also pins the two things that go stale silently when a pipeline is
switched off:

  * the exec/write guard's redirect must name a tool that EXISTS on this
    tenant (the defect `_pipeline_guard_active` was fixed for, one axis over);
  * the per-channel strips must cover both prefixes, or the `vibecoding` and
    `app` channels regain app-building tools the moment Expo retires.
"""

from __future__ import annotations

import os

import pytest

from app.agent import tool_entitlements as te
from app.agent.prefix_stability import strip_tools_for_channel
from app.agent.skills.loader import SkillLoader
from app.config import settings


# Restore the SHIPPED defaults, not True/True. `settings` is a process-wide
# singleton, so whatever this fixture leaves behind is what every later test
# module in the same worker sees. Putting `expo=True` back would hand the rest
# of the suite a posture no container runs any more — and any test that
# depends on the Expo pipeline being gone would then pass or fail on file
# ordering.
_DEFAULT_EXPO = type(settings)().app_builder_expo_enabled
_DEFAULT_HTML = type(settings)().app_html_enabled


@pytest.fixture(autouse=True)
def _restore():
    yield
    settings.agent_tool_families = "*"
    settings.app_builder_expo_enabled = _DEFAULT_EXPO
    settings.app_html_enabled = _DEFAULT_HTML
    te.reset_cache_for_tests()


def _configure(*, expo: bool = False, html: bool = True, families: str = "*") -> None:
    """Set a posture. NOTE `expo=True` no longer produces an Expo container —
    it produces a container carrying a DEAD PIN, which `EXPO_PIPELINE_RETIRED`
    ignores. Tests that pass it are asserting the pin is inert, never that the
    pipeline came back. See `test_expo_pipeline_retired.py`."""
    settings.app_builder_expo_enabled = expo
    settings.app_html_enabled = html
    settings.agent_tool_families = families
    te.reset_cache_for_tests()


async def _load(tmp_path) -> SkillLoader:
    loader = SkillLoader(extra_dirs=[str(tmp_path)])
    await loader.load_all()
    return loader


# ── 0. The shipped default ───────────────────────────────────────────
# 2026-08-21: HTML is the only pipeline a default container runs. This is
# the one test in the file that reads config instead of setting it — every
# other one configures a posture explicitly, so none of them would notice
# the default drifting back.

def test_the_shipped_default_is_html_only(monkeypatch):
    """A fresh Settings, with no pipeline env set, must build HTML apps and
    no Expo ones."""
    monkeypatch.delenv("APP_BUILDER_EXPO_ENABLED", raising=False)
    monkeypatch.delenv("APP_HTML_ENABLED", raising=False)
    fresh = type(settings)()
    assert fresh.app_builder_expo_enabled is False
    assert fresh.app_html_enabled is True


def test_the_env_var_still_parses_but_no_longer_selects(monkeypatch):
    """The field still reads the env — that is what lets `_resolved_pipelines`
    LOG a dead pin, the only way left to find containers still carrying
    `APP_BUILDER_EXPO_ENABLED=1` now that the host's SSH key is rotated.

    What it does NOT do any more is select the pipeline. 2026-08-21 made this
    a one-way door on purpose: see `test_expo_pipeline_retired.py`.
    """
    monkeypatch.setenv("APP_BUILDER_EXPO_ENABLED", "1")
    assert type(settings)().app_builder_expo_enabled is True   # parsed…
    settings.app_builder_expo_enabled = True
    te.reset_cache_for_tests()
    assert te.pipeline_enabled("expo") is False                # …and ignored.


@pytest.mark.asyncio
async def test_default_settings_load_html_and_not_expo(tmp_path, monkeypatch):
    """End of the same wire: the default config, run through the real skill
    loader, yields the five HTML tools and zero Expo ones."""
    monkeypatch.setenv("TOUP_HTML_APPS_DIR", str(tmp_path / "apps"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("APP_BUILDER_EXPO_ENABLED", raising=False)
    monkeypatch.delenv("APP_HTML_ENABLED", raising=False)
    (tmp_path / "home").mkdir()
    fresh = type(settings)()
    _configure(expo=fresh.app_builder_expo_enabled, html=fresh.app_html_enabled)
    loader = await _load(tmp_path)
    assert "app_html" in loader.skills
    assert "app_builder" not in loader.skills
    assert "app" not in loader.skills
    names = {t["name"] for t in loader.get_all_tool_definitions()}
    assert not any(n.startswith(("app_builder__", "app__")) for n in names)
    assert "app_html__create_app_file" in names
    assert "app_html__present_app" in names


# ── 1. The pin is inert ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pinning_expo_back_on_does_not_load_it(tmp_path, monkeypatch):
    """This test used to assert the opposite — that `=1` brought the Expo
    skills back. That rollback is now a code change, not an env var."""
    monkeypatch.setenv("TOUP_HTML_APPS_DIR", str(tmp_path / "apps"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    _configure(expo=True)
    loader = await _load(tmp_path)
    assert "app_html" in loader.skills
    assert "app_builder" not in loader.skills


@pytest.mark.asyncio
async def test_app_html_declares_exactly_the_five_tools(tmp_path, monkeypatch):
    monkeypatch.setenv("TOUP_HTML_APPS_DIR", str(tmp_path / "apps"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    _configure()
    loader = await _load(tmp_path)
    names = sorted(t["name"] for t in loader.get_skill("app_html").get_tools())
    assert names == [
        "app_html__bash_app",
        "app_html__create_app_file",
        "app_html__edit_app_file",
        "app_html__present_app",
        "app_html__view_app_file",
    ]
    # Every one is routable — a definition with no dispatch entry is a tool
    # the model can call and always get "Unknown tool" from.
    for name in names:
        assert loader.is_skill_tool(name), name


@pytest.mark.asyncio
async def test_app_html_prompt_section_is_constant(tmp_path, monkeypatch):
    """The section heads the cached prompt prefix. If it varied per call,
    every turn would start a new provider cache lineage."""
    monkeypatch.setenv("TOUP_HTML_APPS_DIR", str(tmp_path / "apps"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    _configure()
    loader = await _load(tmp_path)
    skill = loader.get_skill("app_html")
    first = skill.get_system_prompt_section()
    assert first and "ONE self-contained .html file" in first
    assert first == skill.get_system_prompt_section()
    # And the tool definitions likewise.
    assert skill.get_tools() == skill.get_tools()


@pytest.mark.asyncio
async def test_the_design_skill_path_is_settled_before_get_tools(tmp_path, monkeypatch):
    """SkillLoader calls get_tools() BEFORE on_load(). The design-skill path
    is embedded in a tool description and in the prompt section, so if
    on_load could still move it, one container would put two different byte
    sequences into the wire array over its life — a forked cache lineage."""
    from app.agent.skills.builtins.app_html.skill import AppHtmlSkill

    monkeypatch.setenv("TOUP_HTML_APPS_DIR", str(tmp_path / "apps"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    monkeypatch.setattr(settings, "skills_dir", str(tmp_path / "skills"))

    skill = AppHtmlSkill()
    before_tools = skill.get_tools()
    before_prompt = skill.get_system_prompt_section()
    path_before = skill._design_path

    await skill.on_load()

    assert skill._design_path == path_before
    assert skill.get_tools() == before_tools
    assert skill.get_system_prompt_section() == before_prompt
    # …and the file the prompt points at is really there.
    assert os.path.isfile(path_before)
    assert "Toup frontend design" in open(path_before, encoding="utf-8").read()


# ── 2. Gating removes the WHOLE pipeline ─────────────────────────────

@pytest.mark.asyncio
async def test_expo_off_withholds_both_expo_skills(tmp_path, monkeypatch):
    monkeypatch.setenv("TOUP_HTML_APPS_DIR", str(tmp_path / "apps"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    _configure(expo=False)
    loader = await _load(tmp_path)
    assert "app_builder" not in loader.skills
    assert "app_html" in loader.skills
    assert not any(
        t["name"].startswith(("app_builder__", "app__"))
        for t in loader.get_all_tool_definitions()
    )


@pytest.mark.asyncio
async def test_html_off_withholds_app_html(tmp_path, monkeypatch):
    monkeypatch.setenv("TOUP_HTML_APPS_DIR", str(tmp_path / "apps"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    _configure(html=False)
    loader = await _load(tmp_path)
    assert "app_html" not in loader.skills
    assert not any(
        t["name"].startswith("app_html__")
        for t in loader.get_all_tool_definitions()
    )


def test_skill_enabled_reflects_both_axes():
    _configure()
    assert te.skill_enabled("app_html")
    # Retired: never enabled, on any posture below.
    assert not te.skill_enabled("app_builder")

    _configure(expo=False)
    assert te.skill_enabled("app_html")
    assert not te.skill_enabled("app_builder")
    assert not te.skill_enabled("app")

    _configure(html=False)
    assert not te.skill_enabled("app_html")
    assert not te.skill_enabled("app_builder")

    # The family axis still wins independently of the pipeline axis.
    _configure(families="doc_generation")
    assert not te.skill_enabled("app_html")
    assert not te.skill_enabled("app_builder")


def test_family_membership_covers_the_html_tools():
    _configure()
    assert te.family_for_tool("app_html__create_app_file") == "app_builder"
    assert te.family_for_tool("app_builder__build_app") == "app_builder"
    # Control: an unconditional tool has no family.
    assert te.family_for_tool("web_search") is None


def test_pipeline_resolution_is_memoized():
    """Container-stable, like the family gate: a mid-life flip must not be
    able to fork the cache lineage.

    Keyed on `html`, because `expo` is now a constant and a constant cannot
    demonstrate memoisation — it would pass with the memo ripped out.
    """
    _configure(html=True)
    assert te.pipeline_enabled("html") is True
    settings.app_html_enabled = False           # no reset_cache_for_tests()
    assert te.pipeline_enabled("html") is True, "the memo was bypassed"
    te.reset_cache_for_tests()
    assert te.pipeline_enabled("html") is False


# ── 3. Things that go stale when a pipeline retires ──────────────────

def _toup_section() -> str:
    from app.agent.skills.builtins.toup.skill import ToupSkill

    return ToupSkill().get_system_prompt_section()


def test_the_toup_skill_routes_apps_to_the_pipeline_that_exists():
    """`toup` is a SEPARATE family and survives Expo's retirement, so its
    prompt outlives the tool it used to name. A prompt that mandates a tool
    outside the wire array is how the model emits a call nothing dispatches.
    """
    _configure(expo=False)                       # the shipped posture
    section = _toup_section()
    assert "app_html__create_app_file" in section
    assert "app_builder__build_app" not in section

    # Expo pinned on AND html off: the `_APP_ROUTE_EXPO` sentence is now
    # unreachable, so the section must name NEITHER tool rather than
    # mandating a retired one. A prompt-mandated tool outside the wire array
    # is how the model emits a stub call to a name nothing dispatches.
    _configure(expo=True, html=False)
    section = _toup_section()
    assert "app_builder__build_app" not in section
    assert "app_html__" not in section

    # Family withheld entirely: name neither, and don't imply an app is
    # something this tenant can ask for.
    _configure(families="toup")
    section = _toup_section()
    assert "app_builder__build_app" not in section
    assert "app_html__" not in section
    # Control — the rest of the section is still there, so the assertions
    # above are not passing on an empty string.
    assert "toup__scaffold" in section


def test_the_toup_section_is_constant_within_a_container():
    """Same cache invariant as the app_html section: two reads, one process,
    identical bytes."""
    _configure(expo=False)
    assert _toup_section() == _toup_section()


def test_the_redirect_names_a_tool_that_exists():
    from app.agent.tool_executor import _pipeline_guard_active, _pipeline_redirect_msg

    # Even with the pin set, the redirect names the HTML tools — the Expo
    # message is unreachable, and its "Ask the user 10+ clarifying questions"
    # instruction with it.
    _configure(expo=True)
    assert _pipeline_guard_active()
    msg = _pipeline_redirect_msg()
    assert "app_html__create_app_file" in msg
    assert "app_builder__build_app" not in msg

    _configure(expo=False)
    assert _pipeline_guard_active()
    msg = _pipeline_redirect_msg()
    assert "app_html__create_app_file" in msg
    assert "app_builder__build_app" not in msg

    # Nothing to redirect TO → the guard stands down rather than blocking
    # a request with no permitted route.
    _configure(expo=False, html=False)
    assert not _pipeline_guard_active()
    _configure(expo=True, html=False)
    assert not _pipeline_guard_active(), "a retired pipeline kept the guard up"

    _configure(families="none")
    assert not _pipeline_guard_active()


@pytest.mark.parametrize("channel", ["vibecoding", "app"])
def test_both_app_pipelines_are_stripped_on_build_and_app_channels(channel):
    tools = [
        {"name": "app_builder__build_app"},
        {"name": "app_html__create_app_file"},
        {"name": "app_html__present_app"},
        {"name": "web_search"},
        {"name": "read_file"},
    ]
    kept = {
        t["name"] for t in strip_tools_for_channel(
            tools, channel, strip_vault_tool_for_channel=lambda t, c: t,
        )
    }
    assert "web_search" in kept, "the strip removed an unrelated tool"
    assert not any(n.startswith(("app_builder__", "app_html__")) for n in kept), kept


def test_the_strip_is_a_no_op_on_a_normal_channel():
    """Control — otherwise the assertions above would pass if the function
    simply returned nothing."""
    tools = [{"name": "app_html__create_app_file"}, {"name": "web_search"}]
    kept = {
        t["name"] for t in strip_tools_for_channel(
            tools, "web", strip_vault_tool_for_channel=lambda t, c: t,
        )
    }
    assert kept == {"app_html__create_app_file", "web_search"}


# ── 4. Metering wiring ───────────────────────────────────────────────

def test_create_and_edit_are_priced_at_the_shared_chokepoint():
    from app.agent.tool_executor import ToolExecutor
    from app.services.credit_service import FLAT_FEES

    mapping = ToolExecutor._FLAT_FEE_TOOLS
    assert mapping["app_html__create_app_file"] == "app_html_create"
    assert mapping["app_html__edit_app_file"] == "app_html_edit"
    for key in ("app_html_create", "app_html_edit"):
        assert key in FLAT_FEES, key
        assert FLAT_FEES[key]["credits"] > 0

    # Deep work costs more than a surgical edit — the whole point of pricing
    # them separately.
    assert FLAT_FEES["app_html_create"]["credits"] > FLAT_FEES["app_html_edit"]["credits"]

    # Read-only tools are deliberately unpriced.
    for name in ("app_html__view_app_file", "app_html__bash_app",
                 "app_html__present_app"):
        assert name not in mapping, name
