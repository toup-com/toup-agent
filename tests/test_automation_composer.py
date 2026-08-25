"""The composer changes the workflow, never widens it (R30 §4.4, §5.5).

Policy is deterministic code — a model classifying creatively can apply
a rule or take a permission away, but can never grant, add, or cross a
rail without `needs`. Tested against the canvas workflow fixture; the
LLM seam is stubbed. No DB — platform sweep.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from app.agent.automations import composer

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = json.loads(
    (REPO_ROOT / "fixtures" / "automations" / "workflow.json").read_text()
)


def _policy(*intents):
    return composer.apply_policy(list(intents), WORKFLOW)


# ------------------------------------------------------------------ applied

def test_a_rule_is_applied_with_the_exact_sentence():
    out = _policy({"kind": "rule", "text": "Skip anything from recruiters."})
    assert out["needs"] == [] and out["answer"] is None
    [intent] = out["applied"]
    assert intent["sheet"] == "rules"
    assert intent["sentence"] == "Added a rule — skip anything from recruiters."


def test_a_schedule_preset_is_applied():
    out = _policy({"kind": "schedule", "preset_id": "weekdays-0730"})
    [intent] = out["applied"]
    assert intent["sheet"] == "sched"
    assert intent["sentence"] == "Moved it to weekdays at 7:30."


def test_a_step_rewording_is_applied():
    out = _policy({"kind": "step", "n": 3, "text": "Stop reading #eng-general."})
    [intent] = out["applied"]
    assert intent["sheet"] == "agent"
    assert intent["sentence"] == "Changed the step — stop reading #eng-general."


def test_a_permission_revoke_is_applied_by_label():
    out = _policy({"kind": "permission", "account_id": "gmail",
                   "permission": "Write drafts", "direction": "revoke"})
    [intent] = out["applied"]
    assert intent["permission_id"] == "write-drafts"
    assert intent["sentence"] == "Gmail can no longer write drafts."


def test_an_account_removal_is_applied():
    out = _policy({"kind": "account", "account_id": "jira",
                   "direction": "remove"})
    [intent] = out["applied"]
    assert intent["sentence"] == "Took Jira out of this automation."


def test_never_post_anywhere_is_a_rule_plus_a_revoke():
    out = _policy(
        {"kind": "rule", "text": "Never post anywhere."},
        {"kind": "permission", "account_id": "slack",
         "permission": "Post as you", "direction": "revoke"},
    )
    kinds = [i["kind"] for i in out["applied"]]
    assert kinds == ["rule", "permission"]
    assert out["applied"][1]["sentence"] == "Slack can no longer post as you."


# -------------------------------------------------------------------- needs

def test_a_grant_always_needs_consent():
    out = _policy({"kind": "permission", "account_id": "jira",
                   "permission": "Close or reassign", "direction": "grant"})
    assert out["applied"] == []
    [need] = out["needs"]
    assert need["kind"] == "consent"
    assert "your yes" in need["sentence"]
    assert "nothing changes until you approve" in need["sentence"]


def test_an_account_add_always_needs_consent():
    out = _policy({"kind": "account", "account_id": "calendar",
                   "direction": "add"})
    assert out["applied"] == []
    [need] = out["needs"]
    assert "Calendar" in need["sentence"]
    assert "read-only" in need["sentence"]


# ------------------------------------------------------------------ refused

def test_a_rail_is_refused_in_the_contract_words():
    out = _policy({"kind": "permission", "account_id": "gmail",
                   "permission": "Send anything", "direction": "grant"})
    assert out["applied"] == [] and out["needs"] == []
    assert out["answer"] == "It can never do this."


def test_revoking_what_is_already_impossible_says_so():
    out = _policy({"kind": "permission", "account_id": "slack",
                   "permission": "Read private DMs", "direction": "revoke"})
    assert out["applied"] == []
    assert "already cannot read private DMs" in out["answer"]


def test_an_unknown_permission_gets_a_helpful_line():
    out = _policy({"kind": "permission", "account_id": "gmail",
                   "permission": "Order groceries", "direction": "revoke"})
    assert out["applied"] == []
    assert "could not find that permission" in out["answer"]


# ----------------------------------------------------------- the full pass

def test_classify_change_applies_the_stub_intents():
    async def complete(prompt):
        assert "THE USER SAYS: run it at 7:30 instead" in prompt
        return {"intents": [{"kind": "schedule", "preset_id": "weekdays-0730"}],
                "question": None, "answer": None}

    out = asyncio.run(composer.classify_change(
        "run it at 7:30 instead", WORKFLOW, complete=complete))
    assert out["applied"][0]["sentence"] == "Moved it to weekdays at 7:30."
    assert out["answer"] is None


def test_an_ambiguous_sentence_asks_one_question():
    async def complete(prompt):
        return {"intents": [], "question": "Which channel should it stop reading?",
                "answer": None}

    out = asyncio.run(composer.classify_change(
        "stop reading it", WORKFLOW, complete=complete))
    assert out["applied"] == [] and out["needs"] == []
    assert out["answer"] == "Which channel should it stop reading?"


def test_a_question_never_overrides_real_changes():
    async def complete(prompt):
        return {"intents": [{"kind": "rule", "text": "Skip recruiters."}],
                "question": "Anything else?", "answer": None}

    out = asyncio.run(composer.classify_change(
        "skip recruiters", WORKFLOW, complete=complete))
    assert out["applied"] and out["answer"] is None


def test_a_dead_model_degrades_to_an_honest_line():
    async def complete(prompt):
        raise RuntimeError("boom")

    out = asyncio.run(composer.classify_change("x" * 5, WORKFLOW,
                                                complete=complete))
    assert out["applied"] == [] and out["needs"] == []
    assert "could not work out what to change" in out["answer"]


def test_the_prompt_shows_the_workflow_but_not_raw_json_noise():
    prompt = composer._extraction_prompt("skip recruiters", WORKFLOW)
    assert "weekdays-0730" in prompt
    assert "Never post in a channel — DM me instead." in prompt
    assert "Read private DMs" in prompt
    assert "$fixture" not in prompt
