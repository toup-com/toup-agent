"""Tests for the maintenance / support agent (app.support).

Two layers:
  * Pure functions (no DB, no LLM): the skills index / router, the JSON
    extractor, classifier coercion, enum invariants.
  * DB-backed pipeline guardrails (LLM mocked via monkeypatch): proves the
    state machine PARKS non-bugs and coverage gaps, and only reaches
    awaiting_approval for a BUG with a covering skill + a diagnosis.

The autouse init_db fixture in conftest.py creates the support tables, so
``async_session_maker()`` works here.
"""

from __future__ import annotations

import pytest

from app.support import skills_index, repository as repo, pipeline
from app.support import classifier, router as triage, diagnoser
from app.support.classifier import ClassificationResult, _coerce
from app.support.router import RoutingResult
from app.support.diagnoser import DiagnosisResult
from app.support.llm import _extract_json
from app.support.enums import (
    SupportClassification, SupportIssueStatus as S, TERMINAL_STATUSES,
)


# ── Pure: skills index is the source of truth ────────────────────────

def test_skills_index_loads_real_skills():
    skills = skills_index.list_skills()
    names = {s.name for s in skills}
    assert "radio-mode" in names          # MVP subsystem present
    assert "toup" not in names            # master router excluded from subsystem list
    assert len(skills) >= 15              # full platform coverage


def test_master_router_table_parses():
    rows = skills_index.parse_router_table()
    assert len(rows) > 20
    assert all(r.skill in skills_index.skill_names() for r in rows)


def test_routing_sends_radio_symptom_to_radio_mode():
    ranked = skills_index.rank_subsystems(
        "radio auto-advance stops after a deploy and songs do not queue", top_n=3,
    )
    assert ranked, "expected at least one ranked subsystem"
    assert ranked[0].name == "radio-mode"


def test_routing_sends_lockscreen_symptom_to_mobile():
    ranked = skills_index.rank_subsystems(
        "audio stops about a second after the phone is locked", top_n=3,
    )
    assert ranked[0].name == "mobile-app"


def test_failure_modes_section_extracted():
    fm = skills_index.failure_modes("radio-mode")
    assert fm and fm.lstrip().startswith("## 6")


# ── Pure: JSON extraction tolerance ──────────────────────────────────

@pytest.mark.parametrize("text", [
    '{"classification": "bug", "confidence": 0.9}',
    '```json\n{"classification": "bug"}\n```',
    'Sure! Here is the result:\n{"classification": "bug"}\nHope that helps.',
])
def test_extract_json_variants(text):
    assert _extract_json(text)["classification"] == "bug"


def test_extract_json_garbage_returns_none():
    assert _extract_json("no json here at all") is None


# ── Pure: classifier coercion + enums ────────────────────────────────

def test_classifier_coerce():
    assert _coerce("bug") is SupportClassification.BUG
    assert _coerce("missing_feature") is SupportClassification.MISSING_FEATURE
    assert _coerce("a missing feature request") is SupportClassification.MISSING_FEATURE
    assert _coerce("???") is SupportClassification.UNCLEAR


def test_terminal_statuses():
    assert S.DONE.value in TERMINAL_STATUSES
    assert S.PARKED.value in TERMINAL_STATUSES
    assert S.AWAITING_APPROVAL.value not in TERMINAL_STATUSES


# ── DB pipeline guardrails (LLM mocked) ──────────────────────────────

async def _make_issue(report: str):
    from app.db import async_session_maker
    async with async_session_maker() as db:
        issue = await repo.create_issue(db, raw_report=report, channel="api")
        return issue.id


async def _status(issue_id: str) -> str:
    from app.db import async_session_maker
    async with async_session_maker() as db:
        issue = await repo.get_issue(db, issue_id)
        return issue.status


@pytest.mark.asyncio
async def test_pipeline_parks_missing_feature(monkeypatch):
    async def fake_classify(**kw):
        return ClassificationResult(SupportClassification.MISSING_FEATURE, 0.9, "never built")
    # If routing were reached it would raise — proving we stop at classify.
    async def boom_route(**kw):
        raise AssertionError("must NOT route a missing-feature")
    monkeypatch.setattr(classifier, "classify", fake_classify)
    monkeypatch.setattr(triage, "route", boom_route)

    issue_id = await _make_issue("please add a dark mode toggle")
    await pipeline.run_diagnosis_pipeline(issue_id)
    assert await _status(issue_id) == S.PARKED.value


@pytest.mark.asyncio
async def test_pipeline_parks_coverage_gap(monkeypatch):
    async def fake_classify(**kw):
        return ClassificationResult(SupportClassification.BUG, 0.9, "looks broken")
    async def gap_route(**kw):
        return RoutingResult(subsystems=[], coverage_gap=True,
                             rationale="nothing matched", gap_note="needs new skill")
    async def boom_diag(**kw):
        raise AssertionError("must NOT diagnose without a covering skill")
    monkeypatch.setattr(classifier, "classify", fake_classify)
    monkeypatch.setattr(triage, "route", gap_route)
    monkeypatch.setattr(diagnoser, "diagnose", boom_diag)

    issue_id = await _make_issue("some totally novel uncovered subsystem is broken")
    await pipeline.run_diagnosis_pipeline(issue_id)
    assert await _status(issue_id) == S.PARKED.value


@pytest.mark.asyncio
async def test_pipeline_bug_reaches_awaiting_approval(monkeypatch):
    async def fake_classify(**kw):
        return ClassificationResult(SupportClassification.BUG, 0.95, "regression")
    async def fake_route(**kw):
        return RoutingResult(subsystems=["radio-mode"], coverage_gap=False, rationale="radio")
    async def fake_diag(**kw):
        return DiagnosisResult(
            report={"root_cause": "rc", "affected_files": ["backend/app/api/ws_chat.py"],
                    "rollback_path": "revert", "skills_referenced": ["radio-mode"]},
            summary="rc", skills_referenced=["radio-mode"], ok=True,
        )
    monkeypatch.setattr(classifier, "classify", fake_classify)
    monkeypatch.setattr(triage, "route", fake_route)
    monkeypatch.setattr(diagnoser, "diagnose", fake_diag)

    issue_id = await _make_issue("radio stopped auto-advancing after the last deploy")
    await pipeline.run_diagnosis_pipeline(issue_id)
    assert await _status(issue_id) == S.AWAITING_APPROVAL.value


@pytest.mark.asyncio
async def test_implementation_refuses_without_approval(monkeypatch):
    # run_implementation must no-op unless status == approved.
    called = {"impl": False}
    async def fake_impl(**kw):
        called["impl"] = True
    monkeypatch.setattr("app.support.implementer.implement", fake_impl)

    issue_id = await _make_issue("radio bug")  # status == intake
    await pipeline.run_implementation(issue_id)
    assert called["impl"] is False
    assert await _status(issue_id) == S.INTAKE.value
