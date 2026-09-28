"""Owner-only static app_html prose pilot; no tool or publish-gate changes."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from app.config import Settings, app_html_static_diet_enabled, settings
from app.agent.skills.builtins.app_html import skill as app_html_skill


def _section(body: str, number: int, next_number: int | None = None) -> str:
    marker = f"## {number}. "
    assert body.count(marker) == 1
    start = body.index(marker)
    if next_number is None:
        return body[start:]
    next_marker = f"## {next_number}. "
    assert body.count(next_marker) == 1
    return body[start:body.index(next_marker)]


def test_exact_owner_web_and_mobile_gate_is_off_by_default(monkeypatch):
    owner = str(uuid.uuid4())
    other = str(uuid.uuid4())
    assert Settings.model_fields["app_html_static_diet_canary_user_ids"].default == ""
    monkeypatch.setattr(settings, "app_html_static_diet_canary_user_ids", owner)
    assert app_html_static_diet_enabled(owner, "web")
    assert app_html_static_diet_enabled(owner, "mobile")
    for channel in ("voice", "app", None):
        assert not app_html_static_diet_enabled(owner, channel)
    assert not app_html_static_diet_enabled(other, "web")
    assert not app_html_static_diet_enabled(owner[:8], "web")
    monkeypatch.setattr(settings, "app_html_static_diet_canary_user_ids", owner[:8])
    assert not app_html_static_diet_enabled(owner, "web")


def test_compact_prompt_keeps_operational_head_publish_gate_and_tool_shapes():
    instance = app_html_skill.AppHtmlSkill()
    full = instance.get_system_prompt_section()
    compact = instance.get_compact_system_prompt_section()
    assert full and compact and full != compact
    marker = "# Toup frontend design"
    assert full.split(marker, 1)[0] == compact.split(marker, 1)[0]
    # The gate executes the motion/state machine and pre-publish checklist.
    assert _section(full, 8, 9) == _section(compact, 8, 9)
    assert _section(full, 11) == _section(compact, 11)
    # Compact prompt selection does not alter the packaged document written
    # on boot, or any callable app tool/schema.
    assert instance._design_guidance == app_html_skill._design_guidance()
    assert app_html_skill._packaged_design_skill().startswith("---\nname:")
    assert len(instance.get_tools()) == 5
    assert instance.get_tools() == app_html_skill.AppHtmlSkill().get_tools()


@pytest.mark.parametrize(
    "section,required",
    [
        (1, ("create_app_file", "one", "4.5:1", "red/green", "logo", "--accent")),
        (2, ("plausible", "Buttons", "Empty states", "Errors")),
        (3, (":hover", ":focus-visible", ":active", ":disabled", "prefers-reduced-motion")),
        (4, ("44 × 44", "8px", "64 × 64", "D-pad", "swipe", "100dvh", "min-height:0", "360px", "768px", "1280px", "viewport-fit=cover")),
        (5, ("4.5:1", "3:1", "16px", "12px", "tabular-nums", "100ms", "300ms")),
        (6, ("<style>", "<script>", "cdnjs.cloudflare.com", "Google Fonts")),
        (7, ("localStorage", "sessionStorage", "document.cookie", "toup-storage-ready", "first-paint", "fetch")),
        (9, ("ArrowUp", "data-dir", "keydown", "phone", "one vocabulary")),
        (10, ("view_app_file", "every", "D-pad", "shared class", "publish")),
    ],
)
def test_compact_sections_keep_substantive_requirements(section, required):
    body = app_html_skill.AppHtmlSkill()._compact_design_guidance
    next_number = section + 1
    segment = _section(body, section, next_number)
    for phrase in required:
        assert phrase in segment, f"§{section} lost {phrase!r}"


def test_real_o200k_cut_is_bounded_and_static():
    tiktoken = pytest.importorskip("tiktoken")
    encode = tiktoken.get_encoding("o200k_base").encode
    instance = app_html_skill.AppHtmlSkill()
    full = instance.get_system_prompt_section()
    compact = instance.get_compact_system_prompt_section()
    assert full and compact
    saved = len(encode(full)) - len(encode(compact))
    assert 3500 <= saved <= 4200
    assert compact == instance.get_compact_system_prompt_section()


def test_source_drift_fails_closed_to_full(monkeypatch):
    full = app_html_skill._design_guidance()
    raw = app_html_skill._packaged_design_skill()
    monkeypatch.setattr(app_html_skill, "_packaged_design_skill", lambda: raw + "\nchanged")
    assert app_html_skill._compact_design_guidance(full) == full


def test_bridge_forwards_pilot_env():
    bridge = (Path(__file__).resolve().parents[2] / "bridge" / "pool_addon.py").read_text()
    assert '"APP_HTML_STATIC_DIET_CANARY_USER_IDS"' in bridge
