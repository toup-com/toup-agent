"""The narration protocol has teeth (R30 §4.2, §5.1–§5.2).

`brief-complete.json` is the ground truth: a dispatch record derived
from its tool turns plus TurnDrafts derived from its authored turns
must validate clean — and every §5.1 rule must FAIL when mutated
(a validator that cannot fail proves nothing). The retry loop is
tested with stub completions; no model, no DB — platform sweep.
"""

from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path

import pytest

from app.agent.automations import narrator

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = json.loads(
    (REPO_ROOT / "fixtures" / "automations" / "brief-complete.json").read_text()
)
B1 = json.loads(
    (REPO_ROOT / "fixtures" / "automations" / "b1.json").read_text()
)
B2 = json.loads(
    (REPO_ROOT / "fixtures" / "automations" / "b2.json").read_text()
)


def record_from(fixture: dict, vocabulary: str, status: str) -> dict:
    run = fixture["runs"][0]
    steps = []
    for t in run["turns"]:
        if t["kind"] != "tool":
            continue
        steps.append({
            "step_ref": t["id"],
            "connector_name": t["account_id"].title(),
            "account_id": t["account_id"],
            "tool_kind": t["tool_kind"],
            "action": t["action"],
            "detail": t["detail"],
            "ok": t["ok"],
            "failure_reason": None if t["ok"] else t["detail"],
            "items": [
                {"id": i["id"], "title": i["title"], "sub": i["sub"],
                 "msgs": [{"who": m["who"], "at": m["at"], "text": m["text"]}
                          for m in i.get("msgs") or []]}
                for i in t.get("items") or []
            ],
            "write": None,
        })
    return {
        "automation": {"title": fixture["automation"]["title"],
                       "mode": "drafts_only"},
        "run_kind": run["kind"],
        "vocabulary": vocabulary,
        "status": status,
        "rules": ["Never post in a channel — DM me instead."],
        "memory_facts": [{"category": "people",
                          "text": "Marcus Webb gets same-day answers"}],
        "steps": steps,
    }


def drafts_from(fixture: dict) -> list[dict]:
    """The fixture's authored turns, re-expressed as TurnDrafts."""
    drafts: list[dict] = []
    for t in fixture["runs"][0]["turns"]:
        kind = t["kind"]
        if kind in ("note", "user"):
            continue  # engine- and user-written, never drafted
        if kind in ("agent", "think"):
            drafts.append({"kind": kind, "text": t["text"]})
        elif kind == "tool":
            drafts.append({
                "kind": "annotate",
                "step_ref": t["id"],
                "items": [
                    {"id": i["id"], "why": i["why"],
                     "msgs": [{"idx": n, "why": m["why"]}
                              for n, m in enumerate(i.get("msgs") or [])]}
                    for i in t.get("items") or []
                ],
                "rest": t.get("rest") or "",
            })
        elif kind == "result":
            drafts.append({
                "kind": "result", "title": t["title"],
                "vocabulary": t["vocabulary"],
                "groups": copy.deepcopy(t["groups"]),
            })
        elif kind == "draft":
            drafts.append({
                "kind": "draft", "text": t["text"],
                "target_account_id": t["target"]["account_id"],
                "target_ref": t["target"]["ref"],
            })
    return drafts


RECORD = record_from(FIXTURE, "brief", "completed")
DRAFTS = drafts_from(FIXTURE)


def _mutated(fn):
    drafts = copy.deepcopy(DRAFTS)
    fn(drafts)
    return narrator.validate_drafts(drafts, RECORD)


# ------------------------------------------------------------------ accepts

def test_the_complete_fixture_validates_clean():
    assert narrator.validate_drafts(DRAFTS, RECORD) == []


def test_b1_changes_fixture_validates_clean():
    record = record_from(B1, "changes", "completed")
    drafts = drafts_from(B1)
    assert narrator.validate_drafts(drafts, record) == []


def test_b2_failed_fixture_validates_clean():
    record = record_from(B2, "brief", "failed")
    drafts = drafts_from(B2)
    assert narrator.validate_drafts(drafts, record) == []


# ------------------------------------------------------------------ rejects

def test_a_wrong_tier_label_is_rejected():
    def mutate(d):
        result = next(x for x in d if x["kind"] == "result")
        result["groups"][0]["label"] = "URGENT"
    assert any("groups must be exactly" in p for p in _mutated(mutate))


def test_an_item_without_a_why_is_rejected():
    def mutate(d):
        ann = next(x for x in d if x["kind"] == "annotate")
        ann["items"].pop()
    assert any("left without a why" in p for p in _mutated(mutate))


def test_a_message_without_a_why_is_rejected():
    def mutate(d):
        for ann in (x for x in d if x["kind"] == "annotate"):
            for item in ann["items"]:
                if item["msgs"]:
                    item["msgs"].pop()
                    return
    assert any("has no why" in p for p in _mutated(mutate))


def test_a_banned_word_in_agent_text_is_rejected():
    def mutate(d):
        d[0]["text"] = "Morning — the workflow executed overnight."
    problems = _mutated(mutate)
    assert any("banned_word 'workflow'" in p for p in problems)
    assert any("banned_word 'executed'" in p for p in problems)


def test_a_double_referenced_item_is_rejected():
    def mutate(d):
        result = next(x for x in d if x["kind"] == "result")
        ref = result["groups"][0]["rows"][0]["item_refs"][0]
        result["groups"][3]["rows"][0]["item_refs"].append(ref)
    assert any("referenced twice" in p for p in _mutated(mutate))


def test_an_unaccounted_item_is_rejected():
    def mutate(d):
        result = next(x for x in d if x["kind"] == "result")
        result["groups"][4]["rows"][-1]["item_refs"] = []
    assert any("unaccounted items" in p for p in _mutated(mutate))


def test_an_unknown_step_ref_is_rejected():
    def mutate(d):
        next(x for x in d if x["kind"] == "annotate")["step_ref"] = "ghost"
    problems = _mutated(mutate)
    assert any("unknown step_ref" in p for p in problems)
    assert any("never annotated" in p for p in problems)


def test_a_result_on_a_failed_run_is_rejected():
    record = record_from(B2, "brief", "failed")
    drafts = drafts_from(B2) + [{
        "kind": "result", "title": "Your morning, in order",
        "vocabulary": "brief",
        "groups": [
            {"rank": r, "label": lb, "tone": tn, "rows": []}
            for r, lb, tn in narrator.BRIEF_GROUPS
        ],
    }]
    assert any("failed run carries no result" in p
               for p in narrator.validate_drafts(drafts, record))


def test_two_results_are_rejected():
    def mutate(d):
        d.append(copy.deepcopy(next(x for x in d if x["kind"] == "result")))
    assert any("exactly one result" in p for p in _mutated(mutate))


def test_a_run_that_read_nothing_still_opens_with_the_agent():
    assert narrator.validate_drafts([], RECORD) == ["no turns emitted"]
    problems = narrator.validate_drafts(
        [{"kind": "think", "text": "hm"}], RECORD)
    assert any("first turn must be the opening agent line" in p
               for p in problems)


# ---------------------------------------------------------------- the pass

def test_narrate_run_accepts_on_first_clean_emission():
    async def complete(prompt, tool):
        return {"turns": copy.deepcopy(DRAFTS)}

    out = asyncio.run(narrator.narrate_run(RECORD, complete=complete))
    assert out["problems"] == []
    assert out["attempts"] == 1
    assert len(out["turns"]) == len(DRAFTS)


def test_narrate_run_retries_with_the_problems_quoted():
    calls = []

    async def complete(prompt, tool):
        calls.append(prompt)
        if len(calls) == 1:
            bad = copy.deepcopy(DRAFTS)
            bad[0]["text"] = "The workflow executed."
            return {"turns": bad}
        return {"turns": copy.deepcopy(DRAFTS)}

    out = asyncio.run(narrator.narrate_run(RECORD, complete=complete))
    assert out["problems"] == []
    assert out["attempts"] == 2
    assert "banned_word 'workflow'" in calls[1]


def test_narrate_run_surfaces_problems_after_the_retry():
    async def complete(prompt, tool):
        return {"turns": [{"kind": "think", "text": "hm"}]}

    out = asyncio.run(narrator.narrate_run(RECORD, complete=complete))
    assert out["attempts"] == 2
    assert out["problems"]  # the engine's completeness net takes over


def test_narrate_run_survives_a_dead_model():
    async def complete(prompt, tool):
        raise RuntimeError("boom")

    out = asyncio.run(narrator.narrate_run(RECORD, complete=complete))
    assert out["turns"] == []
    assert out["problems"] == ["llm: RuntimeError"]


# ------------------------------------------------------------------ prompt

def test_the_prompt_carries_rules_memory_and_the_record():
    prompt = narrator.build_prompt(RECORD)
    assert "Never post in a channel — DM me instead." in prompt
    assert "Marcus Webb gets same-day answers" in prompt
    assert "DISPATCH RECORD:" in prompt
    assert "gmail-01" in prompt
    assert "DO FIRST · BLOCKS OTHERS" in prompt


def test_the_failed_prompt_swaps_the_shape_rules():
    record = record_from(B2, "brief", "failed")
    prompt = narrator.build_prompt(record)
    assert "This run FAILED" in prompt
    assert "DO FIRST · BLOCKS OTHERS" not in prompt
