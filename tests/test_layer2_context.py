"""Tests for the consolidated Layer 2 context builder (Checkpoint 5 Part 2)."""

import pytest
from app.services.layer2_context import (
    Layer2Context,
    _build_available_tools,
    _build_behavior_rules,
    _build_layer2_instructions,
)


def test_layer2_context_base_render():
    """Base context renders tool names and behavior rules."""
    ctx = Layer2Context(
        app_name="Test App",
        slug_safe="Test_App",
        app_slug="Test-App",
        preview_url="https://toup.ai/workspace/apps/Test-App",
        available_tools=_build_available_tools("Test_App"),
        behavior_rules=_build_behavior_rules("Test App", "Test_App", "Test-App",
                                              "https://toup.ai/workspace/apps/Test-App"),
    )
    rendered = ctx.render(is_layer2=False)
    assert "Test App" in rendered
    assert "app_Test_App__navigate" in rendered
    assert "app_Test_App__restart" in rendered
    assert "open_app:Test-App" in rendered
    assert "LAYER 2" not in rendered


def test_layer2_context_l2_render():
    """Layer 2 context includes STEP 1-4 instructions."""
    ctx = Layer2Context(
        app_name="Test App",
        slug_safe="Test_App",
        app_slug="Test-App",
        preview_url="https://toup.ai/workspace/apps/Test-App",
        available_tools=_build_available_tools("Test_App"),
        behavior_rules=_build_behavior_rules("Test App", "Test_App", "Test-App",
                                              "https://toup.ai/workspace/apps/Test-App"),
        layer2_instructions=_build_layer2_instructions("Test_App"),
        layer1_history={
            "prompt": "Build a confidence booster app",
            "choices": "screens: Home, History; database: sqlite",
            "description": "A feel-good app",
        },
    )
    rendered = ctx.render(is_layer2=True)
    assert "LAYER 2 CUSTOMIZATION MODE" in rendered
    assert "STEP 1" in rendered
    assert "STEP 2" in rendered
    assert "STEP 3" in rendered
    assert "STEP 4" in rendered
    assert "Build a confidence booster app" in rendered
    assert "tool calls" in rendered.lower()  # efficiency warning (D2: richer version)
    assert "app_Test_App__read_file" in rendered


def test_layer2_context_no_l2_when_flag_false():
    """render(is_layer2=False) excludes Layer 2 block even if instructions are set."""
    ctx = Layer2Context(
        app_name="Test App",
        slug_safe="Test_App",
        app_slug="Test-App",
        preview_url="https://toup.ai/workspace/apps/Test-App",
        available_tools=_build_available_tools("Test_App"),
        behavior_rules=_build_behavior_rules("Test App", "Test_App", "Test-App",
                                              "https://toup.ai/workspace/apps/Test-App"),
        layer2_instructions=_build_layer2_instructions("Test_App"),
    )
    rendered = ctx.render(is_layer2=False)
    assert "LAYER 2" not in rendered
    assert "STEP 1" not in rendered


def test_structured_fields_present():
    """Structured fields are accessible for testing without string parsing."""
    ctx = Layer2Context(
        app_name="Confidence Booster",
        slug_safe="Confidence_Booster",
        app_slug="Confidence-Booster",
        preview_url="https://toup.ai/workspace/apps/Confidence-Booster",
        available_tools=_build_available_tools("Confidence_Booster"),
        behavior_rules=_build_behavior_rules("Confidence Booster", "Confidence_Booster",
                                              "Confidence-Booster",
                                              "https://toup.ai/workspace/apps/Confidence-Booster"),
        layer1_history={"prompt": "Build a hype app", "description": "Compliment generator"},
        recent_changes=[
            {"file": "App.tsx", "action": "edit", "summary": "Changed theme"},
        ],
    )
    assert "app_Confidence_Booster__restart" in " ".join(ctx.available_tools)
    assert any("open_app:Confidence-Booster" in r for r in ctx.behavior_rules)
    assert ctx.layer1_history["prompt"] == "Build a hype app"
    assert len(ctx.recent_changes) == 1
    assert ctx.recent_changes[0]["file"] == "App.tsx"


def test_empty_l1_history_omits_section():
    """When layer1_history is None, the L1 section is omitted from render."""
    ctx = Layer2Context(
        app_name="Test App",
        slug_safe="Test_App",
        app_slug="Test-App",
        preview_url="https://toup.ai/workspace/apps/Test-App",
        available_tools=_build_available_tools("Test_App"),
        behavior_rules=_build_behavior_rules("Test App", "Test_App", "Test-App",
                                              "https://toup.ai/workspace/apps/Test-App"),
        layer2_instructions=_build_layer2_instructions("Test_App"),
        layer1_history=None,
    )
    rendered = ctx.render(is_layer2=True)
    assert "LAYER 2 CUSTOMIZATION MODE" in rendered
    assert "LAYER 1 BUILD REQUEST" not in rendered
