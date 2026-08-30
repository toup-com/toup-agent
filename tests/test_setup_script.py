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


def test_setup_turns_cover_every_account():
    """R35: one capability turn PER account. The flattened single turn
    was stamped `members[0]` at both call sites, so a six-account brief
    opened with "Checked 1 account" and a lone Jira chip."""
    from app.agent.automations.setup_script import setup_turns
    drafts = setup_turns(
        "reads_only", "", "tonight",
        accounts=[
            {"account_id": "jira",
             "steps": [{"text": "Read your board", "ok": True}]},
            {"account_id": "gmail",
             "steps": [{"text": "Read new mail", "ok": True}]},
            {"account_id": "slack", "steps": []},
        ],
    )
    tools = [d for d in drafts if d.get("kind") == "tool"]
    assert [t["account_id"] for t in tools] == ["jira", "gmail", "slack"]
    assert tools[0]["steps"] == [{"text": "Read your board", "ok": True}]
    # The agent open and close still frame the checks, in order.
    assert drafts[0]["kind"] == "agent" and drafts[-1]["kind"] == "agent"


def test_setup_turns_without_accounts_keeps_the_legacy_shape():
    from app.agent.automations.setup_script import setup_turns
    drafts = setup_turns("reads_only", "", "tonight",
                         [{"text": "Read new mail", "ok": True}])
    tools = [d for d in drafts if d.get("kind") == "tool"]
    assert len(tools) == 1 and "account_id" not in tools[0]


def test_per_account_verb_only_the_writer_says_posts():
    """rec1 f007–f011: a posts-to-Slack brief showed Gmail and Outlook
    sub-labelled "posts" while their own drill-in said read-only, and
    the ⋯ menu said "reads only" one screen away. Only the account
    whose step writes wears the write-mode label."""
    from app.agent.automations.setup_script import setup_turns
    drafts = setup_turns(
        "posts", "posts to #all-toup", "tonight",
        accounts=[
            {"account_id": "gmail", "writes": False, "steps": []},
            {"account_id": "outlook", "writes": False, "steps": []},
            {"account_id": "slack", "writes": True, "steps": []},
        ],
    )
    tools = {d["account_id"]: d for d in drafts if d.get("kind") == "tool"}
    assert tools["gmail"]["detail"] == "reads only"
    assert tools["outlook"]["detail"] == "reads only"
    assert tools["slack"]["detail"] == "posts to #all-toup"


def test_per_account_verb_without_the_flag_keeps_the_legacy_stamp():
    """A caller that cannot say who writes still gets the old shape —
    honest degradation, not a crash and not a silent reads-only."""
    from app.agent.automations.setup_script import setup_turns
    drafts = setup_turns(
        "drafts_only", "", "tonight",
        accounts=[{"account_id": "gmail", "steps": []}],
    )
    tools = [d for d in drafts if d.get("kind") == "tool"]
    assert tools[0]["detail"] == "drafts only"


def test_writer_connectors_reads_both_spec_shapes():
    from app.agent.automations.setup_script import writer_connectors
    v2 = {"steps": [
        {"id": "mail", "connector_id": "gmail",
         "tool": "gmail__list_messages"},
        {"id": "post", "connector_id": "slack",
         "tool": "slack__send_message", "grant_id": "g-1"},
    ]}
    assert writer_connectors(v2) == {"slack"}
    # An UNGRANTED write step is still a writer — a write is a write by
    # its TOOL (R36 doctrine), and the verb must not lie mid-setup.
    v2_ungranted = {"steps": [
        {"id": "mail", "connector_id": "gmail",
         "tool": "gmail__list_messages"},
        {"id": "post", "connector_id": "slack",
         "tool": "slack__send_message"},
    ]}
    assert writer_connectors(v2_ungranted) == {"slack"}
    v1 = {"action": {"connector_id": "slack",
                     "tool": "slack__send_message", "grant_id": "g"}}
    assert writer_connectors(v1) == {"slack"}
    assert writer_connectors({}) == set()
