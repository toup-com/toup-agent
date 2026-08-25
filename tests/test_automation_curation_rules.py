"""The curator's refusal gate, caged against reality (R30 §5.6).

Positive controls: the two verbatim strings GROUND-TRUTH-R30 found in
the founder's prod store (ND-2/ND-3) plus the D-20 dispatch example —
each must be refused with the right reason. Negative controls: all
fifteen memory facts from the approved canvas — a gate that refuses a
real memory is wrong, not strict. No DB — platform sweep.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.agent.automations import curation_rules as rules

REPO_ROOT = Path(__file__).resolve().parents[2]
MEMORY_FIXTURE = json.loads(
    (REPO_ROOT / "fixtures" / "automations" / "memory.json").read_text()
)
CANVAS_FACTS = [
    item["text"]
    for cat in MEMORY_FIXTURE["categories"]
    for item in cat["items"]
]

# What the founder's prod store actually contained on 25 August.
PROD_BAD = {
    "The Morning work brief is currently paused.": "run_status",
    (
        "Has an automation 'Morning work brief': Every day at 22:52, "
        "check Jira, GitHub, Teams, Gmail and Outlook and post to Slack."
    ): "definition",
    (
        "Automation 'Morning work brief': last run partial at "
        "2026-08-25T02:52:14.716539Z"
    ): "run_status",
}


@pytest.mark.parametrize("text,reason", list(PROD_BAD.items()))
def test_the_prod_offenders_are_refused_with_the_right_reason(text, reason):
    assert rules.refuse_reason(text) == reason


def test_more_of_each_class_is_refused():
    assert rules.refuse_reason(
        "It watches Notion dates and posts to Slack every evening."
    ) == "definition"
    assert rules.refuse_reason(
        "Every weekday at 8:00 it checks Gmail and Slack for the user."
    ) == "definition"
    assert rules.refuse_reason("The repository watch is currently active.") \
        == "run_status"
    assert rules.refuse_reason("Paused after 3 failures in a row.") \
        == "run_status"
    assert rules.refuse_reason(
        "Carried forward · Issues: 0 · Mail: 3"
    ) == "run_status"
    assert rules.refuse_reason("   ") == "empty"


@pytest.mark.parametrize("text", CANVAS_FACTS)
def test_every_canvas_memory_fact_passes_the_gate(text):
    # Fifteen real memories, including ones with times ("Standup is
    # 9:30, so the brief lands at 8:00") and schedule-adjacent wording.
    assert rules.refuse_reason(text) is None, text


def test_the_canvas_evidence_sentences_pass_too():
    for cat in MEMORY_FIXTURE["categories"]:
        for item in cat["items"]:
            assert rules.refuse_reason(item["why"]) is None, item["why"]


def test_dedupe_key_survives_cosmetic_differences():
    a = rules.dedupe_key("Marcus Webb gets same-day answers.")
    assert a == rules.dedupe_key("  marcus webb   gets same-day answers")
    assert a == rules.dedupe_key("Marcus Webb gets same-day answers!")
    assert a != rules.dedupe_key("Marcus Webb gets next-day answers.")


def test_the_five_category_keys_match_the_fixture():
    assert [c["key"] for c in MEMORY_FIXTURE["categories"]] \
        == list(rules.CATEGORY_KEYS)
    for cat in MEMORY_FIXTURE["categories"]:
        assert rules.CATEGORY_LABELS[cat["key"]] == cat["label"]


def test_legacy_categories_all_map_somewhere():
    assert set(rules.LEGACY_CATEGORY_MAP) == {"people", "preferences", "deadlines"}
    assert set(rules.LEGACY_CATEGORY_MAP.values()) <= set(rules.CATEGORY_KEYS)


def test_the_classification_prompt_carries_the_bans_and_the_scopes():
    prompt = rules.classification_prompt(
        automation_name="Morning work brief",
        candidate_facts=["Marcus Webb gets same-day answers"],
        existing_by_category={"people": []},
    )
    assert "NEVER file" in prompt
    assert "schedule" in prompt and "status" in prompt
    assert '"scope"' in prompt and '"subject"' in prompt
    assert "never" in prompt and "duplicated" in prompt
    assert 'inside the automation "Morning work brief"' in prompt
