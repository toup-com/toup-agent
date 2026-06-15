"""Tests for the mobile support-intake additions on top of #198.

Covers the security-critical attachment paths (auth + ownership + limits),
the DB round-trip, and the admin-alert email renderer. The conftest test app
doesn't mount the support router, so we build a minimal app and override
get_current_user / get_db to exercise the real endpoints.
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
from app.services.email_service import render_support_card_email


# ── Pure: admin-alert email renderer ─────────────────────────────────

def test_render_support_card_email():
    subject, html, text = render_support_card_email(
        issue_id="abc123", symptom="radio stops on lock", severity="high",
        channel="mobile", classification="bug", reporter="u@x.com",
        queue_url="https://toup.ai/admin?tab=support&issue=abc123", has_screenshot=True,
    )
    assert "HIGH" in subject and "radio stops on lock" in subject
    assert "https://toup.ai/admin?tab=support&issue=abc123" in html
    assert "radio stops on lock" in html
    assert "abc123" in text and "mobile" in text
    # The recipient is NOT baked into the template — it's the To: arg to send_email.
    assert "mrhx@toup.ai" not in html


def test_render_support_card_email_escapes_html():
    # User-controlled symptom/reporter must be HTML-escaped in the HTML body.
    subject, html, text = render_support_card_email(
        issue_id="x", symptom="<script>alert(1)</script>", severity="low",
        channel="mobile", classification="bug", reporter="<b>evil</b>@x.com",
        queue_url="https://toup.ai/admin", has_screenshot=False,
    )
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
    assert "<b>evil</b>@x.com" not in html


# ── DB: attachment round-trip ────────────────────────────────────────

@pytest.mark.asyncio
async def test_attachment_db_roundtrip():
    async with async_session_maker() as db:
        issue = await repo.create_issue(db, raw_report="screenshot please", channel="mobile",
                                        reporter_user_id="user-1")
        att = await repo.create_attachment(
            db, issue_id=issue.id, data=b"\x89PNGfakebytes", mime_type="image/png",
            uploaded_by_user_id="user-1",
        )
        assert att.size_bytes == len(b"\x89PNGfakebytes")
        got = await repo.get_attachment(db, att.id)
        assert got is not None and got.data == b"\x89PNGfakebytes"
        lst = await repo.list_attachments(db, issue.id)
        assert len(lst) == 1
        counts = await repo.attachment_counts(db, [issue.id, "nope"])
        assert counts.get(issue.id) == 1 and "nope" not in counts


# ── HTTP: auth + ownership + limits on the attachment endpoints ──────

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


async def _make_issue(reporter_id: str) -> str:
    async with async_session_maker() as db:
        issue = await repo.create_issue(db, raw_report="bug here", channel="mobile",
                                        reporter_user_id=reporter_id)
        return issue.id


PNG = ("shot.png", b"\x89PNG\r\n\x1a\n" + b"x" * 64, "image/png")


@pytest.mark.asyncio
async def test_reporter_can_upload_and_fetch(app_client):
    issue_id = await _make_issue("reporter-1")
    _CURRENT["user"] = _user("reporter-1")
    up = await app_client.post(f"/api/support/issues/{issue_id}/attachment",
                               files={"file": PNG})
    assert up.status_code == 200, up.text
    att_id = up.json()["id"]
    # fetch bytes back
    got = await app_client.get(f"/api/support/issues/{issue_id}/attachments/{att_id}")
    assert got.status_code == 200
    assert got.headers["content-type"].startswith("image/png")
    assert got.content == PNG[1]


@pytest.mark.asyncio
async def test_admin_can_fetch_other_users_attachment(app_client):
    issue_id = await _make_issue("reporter-2")
    _CURRENT["user"] = _user("reporter-2")
    up = await app_client.post(f"/api/support/issues/{issue_id}/attachment", files={"file": PNG})
    att_id = up.json()["id"]
    # admin (different id) can read
    _CURRENT["user"] = _user("admin-9", role="admin")
    got = await app_client.get(f"/api/support/issues/{issue_id}/attachments/{att_id}")
    assert got.status_code == 200


@pytest.mark.asyncio
async def test_other_user_cannot_upload_or_read(app_client):
    issue_id = await _make_issue("reporter-3")
    # a non-owner, non-admin uploads → 404 (no existence leak)
    _CURRENT["user"] = _user("intruder")
    up = await app_client.post(f"/api/support/issues/{issue_id}/attachment", files={"file": PNG})
    assert up.status_code == 404
    # owner uploads, intruder reads → 404
    _CURRENT["user"] = _user("reporter-3")
    att_id = (await app_client.post(f"/api/support/issues/{issue_id}/attachment",
                                    files={"file": PNG})).json()["id"]
    _CURRENT["user"] = _user("intruder")
    got = await app_client.get(f"/api/support/issues/{issue_id}/attachments/{att_id}")
    assert got.status_code == 404


@pytest.mark.asyncio
async def test_bad_mime_rejected(app_client):
    issue_id = await _make_issue("reporter-4")
    _CURRENT["user"] = _user("reporter-4")
    up = await app_client.post(f"/api/support/issues/{issue_id}/attachment",
                               files={"file": ("evil.exe", b"MZ......", "application/octet-stream")})
    assert up.status_code == 415


@pytest.mark.asyncio
async def test_oversize_rejected(app_client, monkeypatch):
    monkeypatch.setattr(settings, "support_attachment_max_bytes", 32, raising=False)
    issue_id = await _make_issue("reporter-5")
    _CURRENT["user"] = _user("reporter-5")
    big = ("big.png", b"\x89PNG" + b"y" * 1000, "image/png")
    up = await app_client.post(f"/api/support/issues/{issue_id}/attachment", files={"file": big})
    assert up.status_code == 413


@pytest.mark.asyncio
async def test_reporter_email_not_spoofable_by_non_admin(app_client, monkeypatch):
    # No background tasks during this test.
    monkeypatch.setattr("app.support.pipeline.spawn", lambda *a, **k: None)
    _CURRENT["user"] = _user("reporter-x", email="real@me.com")
    res = await app_client.post("/api/support/issues", json={
        "raw_report": "something broke", "channel": "mobile",
        "reporter_email": "victim@spoof.com",
    })
    assert res.status_code == 200
    iid = res.json()["id"]
    async with async_session_maker() as db:
        issue = await repo.get_issue(db, iid)
        # Non-admin override ignored — session identity wins.
        assert issue.reporter_email == "real@me.com"


@pytest.mark.asyncio
async def test_admin_can_set_reporter_email(app_client, monkeypatch):
    monkeypatch.setattr("app.support.pipeline.spawn", lambda *a, **k: None)
    _CURRENT["user"] = _user("admin-x", role="admin", email="admin@toup.ai")
    res = await app_client.post("/api/support/issues", json={
        "raw_report": "on behalf of a user", "channel": "api",
        "reporter_email": "onbehalf@user.com",
    })
    assert res.status_code == 200
    async with async_session_maker() as db:
        issue = await repo.get_issue(db, res.json()["id"])
        assert issue.reporter_email == "onbehalf@user.com"


@pytest.mark.asyncio
async def test_disabled_returns_503(app_client, monkeypatch):
    monkeypatch.setattr(settings, "support_agent_enabled", False, raising=False)
    issue_id = await _make_issue("reporter-6")
    _CURRENT["user"] = _user("reporter-6")
    up = await app_client.post(f"/api/support/issues/{issue_id}/attachment", files={"file": PNG})
    assert up.status_code == 503
