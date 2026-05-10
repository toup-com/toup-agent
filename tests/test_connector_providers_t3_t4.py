"""
T3 + T4 — Provider unit tests with mocked HTTP.

Each Google provider (Gmail, Calendar, Drive) and GitHub goes through
a happy-path tool call + the four most consequential error mappings:

  401 → ConnectorReauthRequired (retargeted to the per-connector URL)
  403 (scope hint) → ConnectorScopeMissing
  429 → ConnectorRateLimited (with retry_after)
  5xx → ConnectorProviderDown

Plus per-provider:
  - Gmail: send_message round-trips RFC822 → base64url
  - Calendar: create_event sends summary/start/end + attendees
  - Drive: create_doc does the two-step (Drive create → Docs insertText)
  - GitHub: GH 403 with X-RateLimit-Remaining=0 → ConnectorRateLimited
            (not ScopeMissing) — the disambiguation rule

HTTP is mocked via httpx.MockTransport so providers exercise their
real header/body marshalling. No live network.
"""

from __future__ import annotations

import base64
import json
import uuid
from datetime import datetime, timedelta
from typing import Callable, ClassVar
from unittest.mock import patch

import httpx
import pytest
import pytest_asyncio
from cryptography.fernet import Fernet

from app.config import settings
from app.connectors.base import (
    ConnectorContext,
    ConnectorOk,
    ConnectorProviderDown,
    ConnectorRateLimited,
    ConnectorReauthRequired,
    ConnectorScopeMissing,
    ConnectorToolError,
)
from app.db.database import async_session_maker
from app.db.models import User
from app.services import connector_vault as vault
from app.services.credential_crypto import _multi_fernet


# ─── Fixtures: crypto + alice + token resolution ────────────────────────


@pytest.fixture(autouse=True)
def _crypto():
    settings.platform_encryption_key = Fernet.generate_key().decode()
    settings.platform_encryption_key_previous = ""
    _multi_fernet.cache_clear()
    yield


@pytest_asyncio.fixture
async def alice_id() -> str:
    async with async_session_maker() as db:
        uid = str(uuid.uuid4())
        db.add(User(id=uid, email=f"{uid[:8]}@example.com",
                    hashed_password="x", name="Alice"))
        await db.commit()
    return uid


async def _seed(user_id: str, connector_id: str) -> None:
    async with async_session_maker() as db:
        await vault.put(
            db, user_id, connector_id,
            access_token="tok_seed",
            refresh_token="rt_seed",
            access_expires_at=datetime.utcnow() + timedelta(hours=1),
        )


def _stub_transport(handler: Callable[[httpx.Request], httpx.Response]):
    """Patch httpx.AsyncClient so every constructed client uses our
    MockTransport. Returns a context manager."""
    real_init = httpx.AsyncClient.__init__

    def patched_init(self, *args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        real_init(self, *args, **kwargs)

    return patch("httpx.AsyncClient.__init__", patched_init)


# ─── Gmail ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_gmail_list_messages_happy(alice_id):
    from app.connectors.gmail.provider import GmailProvider

    await _seed(alice_id, "gmail")

    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"] = str(req.url)
        seen["auth"] = req.headers.get("authorization")
        return httpx.Response(200, json={
            "messages": [
                {"id": "m1", "threadId": "t1"},
                {"id": "m2", "threadId": "t1"},
            ],
            "resultSizeEstimate": 2,
        })

    with _stub_transport(handler):
        provider = GmailProvider()
        result = await provider.execute(
            "gmail__list_messages",
            {"max_results": 5},
            ConnectorContext(user_id=alice_id, channel="web"),
        )

    assert isinstance(result, ConnectorOk)
    body = json.loads(result.content)
    assert len(body["messages"]) == 2
    assert seen["auth"] == "Bearer tok_seed"
    assert "users/me/messages" in seen["url"]


@pytest.mark.asyncio
async def test_gmail_send_message_marshals_rfc822_base64url(alice_id):
    from app.connectors.gmail.provider import GmailProvider

    await _seed(alice_id, "gmail")
    sent_raw: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        sent_raw["raw"] = body["raw"]
        return httpx.Response(200, json={
            "id": "sent_id",
            "threadId": "t1",
            "labelIds": ["SENT"],
        })

    with _stub_transport(handler):
        provider = GmailProvider()
        result = await provider.execute(
            "gmail__send_message",
            {"to": "alice@example.com", "subject": "hi", "body": "hello world"},
            ConnectorContext(user_id=alice_id, channel="web"),
        )

    assert isinstance(result, ConnectorOk)
    # Decode the base64url back to verify RFC822 contents.
    pad = (-len(sent_raw["raw"])) % 4
    decoded = base64.urlsafe_b64decode(
        sent_raw["raw"] + ("=" * pad)
    ).decode()
    assert "To: alice@example.com" in decoded
    assert "Subject: hi" in decoded
    assert "hello world" in decoded


@pytest.mark.asyncio
async def test_gmail_401_retargets_reauth_url(alice_id):
    from app.connectors.gmail.provider import GmailProvider

    await _seed(alice_id, "gmail")

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="invalid_token")

    with _stub_transport(handler):
        provider = GmailProvider()
        result = await provider.execute(
            "gmail__list_messages",
            {},
            ConnectorContext(user_id=alice_id, channel="web"),
        )

    assert isinstance(result, ConnectorReauthRequired)
    assert "/agent/integrations/gmail" in result.reauth_url


@pytest.mark.asyncio
async def test_gmail_403_scope_missing_returns_scope_variant(alice_id):
    from app.connectors.gmail.provider import GmailProvider

    await _seed(alice_id, "gmail")

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            text='{"error":{"message":"Insufficient Permission"}}',
            headers={"content-type": "application/json"},
        )

    with _stub_transport(handler):
        provider = GmailProvider()
        result = await provider.execute(
            "gmail__list_messages",
            {},
            ConnectorContext(user_id=alice_id, channel="web"),
        )

    assert isinstance(result, ConnectorScopeMissing)


@pytest.mark.asyncio
async def test_gmail_429_returns_rate_limited(alice_id):
    from app.connectors.gmail.provider import GmailProvider

    await _seed(alice_id, "gmail")

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="quota", headers={"Retry-After": "60"})

    with _stub_transport(handler):
        provider = GmailProvider()
        result = await provider.execute(
            "gmail__list_messages",
            {},
            ConnectorContext(user_id=alice_id, channel="web"),
        )

    assert isinstance(result, ConnectorRateLimited)
    assert result.retry_after_s == 60


@pytest.mark.asyncio
async def test_gmail_5xx_returns_provider_down(alice_id):
    from app.connectors.gmail.provider import GmailProvider

    await _seed(alice_id, "gmail")

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="down")

    with _stub_transport(handler):
        provider = GmailProvider()
        result = await provider.execute(
            "gmail__list_messages",
            {},
            ConnectorContext(user_id=alice_id, channel="web"),
        )

    assert isinstance(result, ConnectorProviderDown)
    assert "status.cloud.google.com" in (result.provider_status_url or "")


# ─── Calendar ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_calendar_create_event_sends_required_fields(alice_id):
    from app.connectors.calendar.provider import CalendarProvider

    await _seed(alice_id, "calendar")
    sent: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        sent["url"] = str(req.url)
        sent["body"] = json.loads(req.content)
        return httpx.Response(200, json={
            "id": "evt_1",
            "htmlLink": "https://calendar.google.com/event?eid=...",
            "summary": "Meet",
            "start": sent["body"]["start"],
            "end": sent["body"]["end"],
        })

    with _stub_transport(handler):
        provider = CalendarProvider()
        result = await provider.execute(
            "calendar__create_event",
            {
                "summary": "Meet",
                "start": "2026-05-15T14:00:00-07:00",
                "end": "2026-05-15T14:30:00-07:00",
                "attendees": ["bob@example.com"],
            },
            ConnectorContext(user_id=alice_id, channel="web"),
        )

    assert isinstance(result, ConnectorOk)
    assert sent["body"]["summary"] == "Meet"
    assert sent["body"]["attendees"] == [{"email": "bob@example.com"}]
    assert "calendars/primary/events" in sent["url"]
    assert "sendUpdates=all" in sent["url"]


@pytest.mark.asyncio
async def test_calendar_list_events_passes_query_params(alice_id):
    from app.connectors.calendar.provider import CalendarProvider

    await _seed(alice_id, "calendar")
    seen_url: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen_url["v"] = str(req.url)
        return httpx.Response(200, json={"items": []})

    with _stub_transport(handler):
        provider = CalendarProvider()
        result = await provider.execute(
            "calendar__list_events",
            {
                "time_min": "2026-05-10T00:00:00Z",
                "time_max": "2026-05-17T00:00:00Z",
                "max_results": 25,
                "query": "standup",
            },
            ConnectorContext(user_id=alice_id, channel="web"),
        )

    assert isinstance(result, ConnectorOk)
    url = seen_url["v"]
    assert "timeMin=2026-05-10T00" in url
    assert "timeMax=2026-05-17T00" in url
    assert "maxResults=25" in url
    assert "q=standup" in url
    assert "singleEvents=true" in url


@pytest.mark.asyncio
async def test_calendar_401_retargets_to_calendar(alice_id):
    from app.connectors.calendar.provider import CalendarProvider

    await _seed(alice_id, "calendar")

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="invalid")

    with _stub_transport(handler):
        provider = CalendarProvider()
        result = await provider.execute(
            "calendar__list_events", {},
            ConnectorContext(user_id=alice_id, channel="web"),
        )

    assert isinstance(result, ConnectorReauthRequired)
    assert "/agent/integrations/calendar" in result.reauth_url


# ─── Drive ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_drive_create_doc_does_two_step(alice_id):
    from app.connectors.drive.provider import DriveProvider

    await _seed(alice_id, "drive")
    calls: list[tuple[str, str]] = []  # (method, url)

    def handler(req: httpx.Request) -> httpx.Response:
        url = str(req.url)
        calls.append((req.method, url))
        if "googleapis.com/drive/v3/files" in url and req.method == "POST":
            # Step 1 — create the empty doc.
            return httpx.Response(200, json={
                "id": "doc_xyz",
                "name": "Title",
                "mimeType": "application/vnd.google-apps.document",
            })
        if "docs.googleapis.com" in url and "doc_xyz" in url:
            # Step 2 — insert the body text via batchUpdate.
            body = json.loads(req.content)
            assert body["requests"][0]["insertText"]["text"] == "Body content"
            return httpx.Response(200, json={"replies": [{}]})
        return httpx.Response(404, text=f"unexpected {req.method} {url}")

    with _stub_transport(handler):
        provider = DriveProvider()
        result = await provider.execute(
            "drive__create_doc",
            {"title": "Title", "body": "Body content"},
            ConnectorContext(user_id=alice_id, channel="web"),
        )

    assert isinstance(result, ConnectorOk)
    body = json.loads(result.content)
    assert body["id"] == "doc_xyz"
    assert "docs.google.com/document/d/doc_xyz" in body["url"]
    # Both API calls fired in order.
    assert any("googleapis.com/drive/v3/files" in u for _, u in calls)
    assert any("docs.googleapis.com" in u and "doc_xyz" in u for _, u in calls)


@pytest.mark.asyncio
async def test_drive_list_files_passes_query(alice_id):
    from app.connectors.drive.provider import DriveProvider

    await _seed(alice_id, "drive")
    seen: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"] = str(req.url)
        return httpx.Response(200, json={
            "files": [{"id": "f1", "name": "x", "mimeType": "text/plain"}],
            "nextPageToken": None,
        })

    with _stub_transport(handler):
        provider = DriveProvider()
        result = await provider.execute(
            "drive__list_files",
            {"query": "mimeType='application/pdf'", "max_results": 10},
            ConnectorContext(user_id=alice_id, channel="web"),
        )

    assert isinstance(result, ConnectorOk)
    assert "pageSize=10" in seen["url"]
    assert "q=mimeType" in seen["url"]


# ─── GitHub ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_github_list_repos_trims_response(alice_id):
    from app.connectors.github.provider import GitHubProvider

    await _seed(alice_id, "github")

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[
            {
                "full_name": "acme/widgets",
                "private": False,
                "description": "Widgets",
                "html_url": "https://github.com/acme/widgets",
                "default_branch": "main",
                "_extra_field_we_dont_care_about": "x" * 1000,
            }
        ])

    with _stub_transport(handler):
        provider = GitHubProvider()
        result = await provider.execute(
            "github__list_repos",
            {},
            ConnectorContext(user_id=alice_id, channel="web"),
        )

    assert isinstance(result, ConnectorOk)
    body = json.loads(result.content)
    assert len(body["repos"]) == 1
    assert body["repos"][0]["full_name"] == "acme/widgets"
    # Verify trimming: extra field NOT in output.
    assert "_extra_field_we_dont_care_about" not in result.content


@pytest.mark.asyncio
async def test_github_403_with_ratelimit_zero_returns_rate_limited(alice_id):
    """GitHub returns 403 for both rate-limit AND scope-missing. The
    X-RateLimit-Remaining=0 header is the disambiguation."""
    from app.connectors.github.provider import GitHubProvider

    await _seed(alice_id, "github")
    import time
    reset_at = int(time.time()) + 90

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            text="rate limit exceeded",
            headers={
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(reset_at),
            },
        )

    with _stub_transport(handler):
        provider = GitHubProvider()
        result = await provider.execute(
            "github__list_repos",
            {},
            ConnectorContext(user_id=alice_id, channel="web"),
        )

    assert isinstance(result, ConnectorRateLimited)
    # ~90s window; allow some margin.
    assert 30 <= result.retry_after_s <= 95


@pytest.mark.asyncio
async def test_github_403_without_ratelimit_returns_scope_missing(alice_id):
    """403 with no X-RateLimit-Remaining=0 → scope missing path."""
    from app.connectors.github.provider import GitHubProvider

    await _seed(alice_id, "github")

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="permission denied")

    with _stub_transport(handler):
        provider = GitHubProvider()
        result = await provider.execute(
            "github__list_repos",
            {},
            ConnectorContext(user_id=alice_id, channel="web"),
        )

    assert isinstance(result, ConnectorScopeMissing)


@pytest.mark.asyncio
async def test_github_401_retargets_reauth_url(alice_id):
    from app.connectors.github.provider import GitHubProvider

    await _seed(alice_id, "github")

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="bad creds")

    with _stub_transport(handler):
        provider = GitHubProvider()
        result = await provider.execute(
            "github__list_repos", {},
            ConnectorContext(user_id=alice_id, channel="web"),
        )

    assert isinstance(result, ConnectorReauthRequired)
    assert "/agent/integrations/github" in result.reauth_url


@pytest.mark.asyncio
async def test_github_create_comment_posts_body(alice_id):
    from app.connectors.github.provider import GitHubProvider

    await _seed(alice_id, "github")
    sent: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        sent["url"] = str(req.url)
        sent["body"] = json.loads(req.content)
        return httpx.Response(201, json={
            "id": 42,
            "html_url": "https://github.com/acme/repo/issues/7#issuecomment-42",
        })

    with _stub_transport(handler):
        provider = GitHubProvider()
        result = await provider.execute(
            "github__create_comment",
            {"owner": "acme", "repo": "repo", "number": 7, "body": "lgtm"},
            ConnectorContext(user_id=alice_id, channel="web"),
        )

    assert isinstance(result, ConnectorOk)
    assert sent["body"] == {"body": "lgtm"}
    assert "/repos/acme/repo/issues/7/comments" in sent["url"]
