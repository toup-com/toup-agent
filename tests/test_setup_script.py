"""The setup thread reads like the canvas, in every mode (R30 §5.3).

The drafts_only/tonight pair must reproduce the canvas fixture's
strings byte-for-byte (fixtures/automations/setup.json is the diff
target for atlas state parity); the other three modes are extensions
with pinned exact strings. Pure functions, no DB — platform sweep.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.agent.automations.setup_script import MODES, mode_label, setup_turns

REPO_ROOT = Path(__file__).resolve().parents[2]
SETUP_FIXTURE = REPO_ROOT / "fixtures" / "automations" / "setup.json"


def test_drafts_tonight_reproduces_the_canvas_fixture_bytes():
    fixture = json.loads(SETUP_FIXTURE.read_text())
    fixture_turns = fixture["runs"][0]["turns"]
    agent_texts = [t["text"] for t in fixture_turns if t["kind"] == "agent"]
    think_texts = [t["text"] for t in fixture_turns if t["kind"] == "think"]
    tool_turns = [t for t in fixture_turns if t["kind"] == "tool"]

    drafts = setup_turns("drafts_only", scope_lines=[])
    assert [t["text"] for t in drafts if t["kind"] == "agent"] == agent_texts
    assert [t["text"] for t in drafts if t["kind"] == "think"] == think_texts
    tool_request = [t for t in drafts if t["kind"] == "tool"][0]
    assert tool_request["action"] == tool_turns[0]["action"] == "Checked what I can do"
    assert tool_request["detail"] == tool_turns[0]["detail"] == "drafts only"


def test_turn_order_is_agent_tool_think_agent():
    for mode in MODES:
        kinds = [t["kind"] for t in setup_turns(mode, channel_label="#platform")]
        assert kinds == ["agent", "tool", "think", "agent"], mode


def test_each_mode_states_its_own_limit():
    reads = setup_turns("reads_only")
    assert reads[0]["text"] == (
        "Here is what I will be able to do — read, and tell you. "
        "I cannot change anything."
    )
    posts = setup_turns("posts", channel_label="#platform")
    assert posts[0]["text"] == (
        "Here is what I will be able to do — post one line in #platform, "
        "nothing else."
    )
    asks = setup_turns("asks_first")
    assert asks[0]["text"] == (
        "Here is what I will be able to do — prepare the change and wait "
        "for your yes before anything happens."
    )
    assert asks[2]["text"] == (
        "I stage the change and stop. Until you approve, nothing has happened."
    )


def test_non_tonight_labels_use_the_honest_generic_close():
    close = setup_turns("drafts_only", first_run_label="in a few minutes")[-1]["text"]
    assert close == (
        "First run is in a few minutes. The drafts will be waiting, "
        "and every step will be here."
    )
    # "In the morning" is only promised when the run is actually tonight.
    assert "In the morning" not in close


def test_scope_lines_ride_the_capability_check():
    lines = [
        {"text": "Read new mail", "ok": True},
        {"text": "Write drafts", "ok": True},
        {"text": "Send anything", "ok": False},
    ]
    tool = setup_turns("drafts_only", scope_lines=lines)[1]
    assert tool["steps"] == lines


def test_mode_label_names_the_real_target():
    assert mode_label("drafts_only") == "drafts only"
    assert mode_label("reads_only") == "reads only"
    assert mode_label("posts", channel_label="#platform") == "posts to #platform"
    assert mode_label("asks_first") == "asks first"


def test_unknown_mode_refuses():
    with pytest.raises(ValueError):
        setup_turns("autonomous")
