"""Tests for the Phase-0 diagnosis-quality grading harness.

The grading verdict is an admin ANNOTATION that records how good a diagnosis
was (the ≥80%-actionable validation gate). Its integrity rests on two
invariants these tests pin down:

  1. Grading is STRICTLY SEPARATE from approval — it never changes the issue's
     status and never sets the decision/approval columns or spawns a fix.
  2. The corpus tally is a real number computed from real DB records.

The conftest test app doesn't mount the support router, so we build a minimal
app and override get_current_user / get_db (mirrors test_support_attachments).
"""

from __future__ import annotations

import types

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api.auth import get_current_user
from app.api.support import router as support_router
from app.config import settings
from app.db import get_db, async_session_maker
from app.support import repository as repo
from app.support.enums import SupportEventType, SupportIssueStatus as S, SupportClassification


_CURRENT = {"user": None}


def _user(uid: str, role: str = "user", email: str = "u@x.com"):
    return types.SimpleNamespace(id=uid, role=role, email=email)


@pytest_asyncio.fixture
async def app_client(monkeypatch):
    monkeypatch.setattr(settings, "support_agent_enabled", True, raising=False)
    app = FastAPI()
    app.include_router(support_router, prefix=settings.api_prefix)

    async def _override_db():
        async with async_session_maker() as db:
            yield db

    app.dependency_overrides[get_db] = _override_db
    app.dependency_overrides[get_current_user] = lambda: _CURRENT["user"]
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://t") as c:
        yield c


async def _make_issue(*, status: str = "intake", classification: str | None = None) -> str:
    async with async_session_maker() as db:
        issue = await repo.create_issue(db, raw_report="something broke", channel="api",
                                        reporter_user_id="reporter-1")
        if classification:
            issue.classification = classification
            db.add(issue)
            await db.commit()
        if status != "intake":
            await repo.set_status(db, issue, status)
        return issue.id


# ── set_grade is annotation only: persists + audits, never transitions ──

@pytest.mark.asyncio
async def test_grade_persists_and_writes_event(app_client):
    issue_id = await _make_issue(status="awaiting_approval", classification="bug")
    _CURRENT["user"] = _user("admin-1", role="admin")
    res = await app_client.post(f"/api/support/issues/{issue_id}/grade",
                                json={"verdict": "actionable", "note": "root cause is right"})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["grade_verdict"] == "actionable"
    assert body["grade_note"] == "root cause is right"
    assert body["graded_by_user_id"] == "admin-1"
    assert body["graded_at"] is not None

    async with async_session_maker() as db:
        events = await repo.list_events(db, issue_id)
    graded = [e for e in events if e.event_type == SupportEventType.GRADED.value]
    assert len(graded) == 1
    assert graded[0].actor == "admin"
    assert graded[0].detail["verdict"] == "actionable"


@pytest.mark.asyncio
async def test_grade_does_not_change_status_or_decision(app_client):
    # The whole point: grading must NOT approve. Status stays put; decision stays None.
    issue_id = await _make_issue(status="awaiting_approval", classification="bug")
    _CURRENT["user"] = _user("admin-1", role="admin")
    res = await app_client.post(f"/api/support/issues/{issue_id}/grade",
                                json={"verdict": "wrong_root_cause"})
    assert res.status_code == 200
    async with async_session_maker() as db:
        issue = await repo.get_issue(db, issue_id)
    assert issue.status == S.AWAITING_APPROVAL.value   # unchanged
    assert issue.decision is None                       # never approved
    assert issue.decision_by_user_id is None
    assert issue.branch_name is None and issue.pr_url is None  # no fix spawned


@pytest.mark.asyncio
async def test_grade_parked_misclassification(app_client):
    # A real bug wrongly parked: gradeable in a terminal/parked state via other+note.
    issue_id = await _make_issue(status="parked", classification="unclear")
    _CURRENT["user"] = _user("admin-1", role="admin")
    res = await app_client.post(
        f"/api/support/issues/{issue_id}/grade",
        json={"verdict": "other", "note": "this is a real bug — misclassified as unclear"},
    )
    assert res.status_code == 200
    async with async_session_maker() as db:
        issue = await repo.get_issue(db, issue_id)
    assert issue.status == S.PARKED.value      # still parked — annotation only
    assert issue.grade_verdict == "other"


@pytest.mark.asyncio
async def test_grade_overwrite_appends_event(app_client):
    issue_id = await _make_issue(status="awaiting_approval", classification="bug")
    _CURRENT["user"] = _user("admin-1", role="admin")
    await app_client.post(f"/api/support/issues/{issue_id}/grade", json={"verdict": "actionable"})
    res = await app_client.post(f"/api/support/issues/{issue_id}/grade",
                                json={"verdict": "wrong_files"})
    assert res.status_code == 200
    assert res.json()["grade_verdict"] == "wrong_files"   # overwritten
    async with async_session_maker() as db:
        events = await repo.list_events(db, issue_id)
    graded = [e for e in events if e.event_type == SupportEventType.GRADED.value]
    assert len(graded) == 2   # history preserved


# ── Auth + validation ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_grade_requires_admin(app_client):
    issue_id = await _make_issue(status="awaiting_approval", classification="bug")
    _CURRENT["user"] = _user("not-admin", role="user")
    res = await app_client.post(f"/api/support/issues/{issue_id}/grade",
                                json={"verdict": "actionable"})
    assert res.status_code == 403


@pytest.mark.asyncio
async def test_invalid_verdict_rejected(app_client):
    issue_id = await _make_issue(status="awaiting_approval", classification="bug")
    _CURRENT["user"] = _user("admin-1", role="admin")
    res = await app_client.post(f"/api/support/issues/{issue_id}/grade",
                                json={"verdict": "looks_fine_to_me"})
    assert res.status_code == 422


@pytest.mark.asyncio
async def test_grade_missing_issue_404(app_client):
    _CURRENT["user"] = _user("admin-1", role="admin")
    res = await app_client.post("/api/support/issues/does-not-exist/grade",
                                json={"verdict": "actionable"})
    assert res.status_code == 404


@pytest.mark.asyncio
async def test_disabled_returns_503(app_client, monkeypatch):
    monkeypatch.setattr(settings, "support_agent_enabled", False, raising=False)
    _CURRENT["user"] = _user("admin-1", role="admin")
    res = await app_client.post("/api/support/issues/whatever/grade",
                                json={"verdict": "actionable"})
    assert res.status_code == 503


# ── Corpus tally: a real number from real records ───────────────────────

@pytest.mark.asyncio
async def test_corpus_tally_and_actionable_rate(app_client):
    _CURRENT["user"] = _user("admin-1", role="admin")
    # 4 actionable + 1 wrong_root_cause = 5 graded; 1 left ungraded (excluded).
    graded_ids = []
    for _ in range(4):
        iid = await _make_issue(status="awaiting_approval", classification="bug")
        await app_client.post(f"/api/support/issues/{iid}/grade", json={"verdict": "actionable"})
        graded_ids.append(iid)
    iid = await _make_issue(status="awaiting_approval", classification="bug")
    await app_client.post(f"/api/support/issues/{iid}/grade", json={"verdict": "wrong_root_cause"})
    graded_ids.append(iid)
    await _make_issue(status="awaiting_approval", classification="bug")  # ungraded

    res = await app_client.get("/api/support/corpus")
    assert res.status_code == 200
    body = res.json()
    tally = body["tally"]
    assert tally["total_graded"] == 5
    assert tally["actionable"] == 4
    assert tally["actionable_rate"] == pytest.approx(0.8)
    assert tally["by_verdict"]["actionable"] == 4
    assert tally["by_verdict"]["wrong_root_cause"] == 1
    # corpus lists only graded issues
    assert len(body["issues"]) == 5
    assert all(i["grade_verdict"] for i in body["issues"])


@pytest.mark.asyncio
async def test_corpus_empty_is_zero_not_division_error(app_client):
    _CURRENT["user"] = _user("admin-1", role="admin")
    res = await app_client.get("/api/support/corpus")
    assert res.status_code == 200
    tally = res.json()["tally"]
    assert tally["total_graded"] == 0
    assert tally["actionable_rate"] == 0.0


@pytest.mark.asyncio
async def test_corpus_requires_admin(app_client):
    _CURRENT["user"] = _user("not-admin", role="user")
    res = await app_client.get("/api/support/corpus")
    assert res.status_code == 403
