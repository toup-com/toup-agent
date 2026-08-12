"""Regression suite for the 2026-08-07 Google connector audit.

Four defects, all of which were invisible to `tsc`, the linter and the
whole existing test suite, because none of them crossed the boundary
between our code and Google:

  1. **Docs' health probe never called Google.** It checked that a
     token could be decrypted and returned ok. So the connector
     reported `active` for two days while every real call returned 403
     SERVICE_DISABLED — the Docs API had never been enabled on the
     Cloud project. A probe that cannot fail is not a probe.

  2. **Calendar's health probe used a scope the connector never
     requests.** It called `users/me/calendarList`, which needs
     `calendar.readonly`. That scope is `scopes_optional`, and
     `oauth.py` only ever sends `oauth.scopes`, so no user has ever
     held it. Every probe 403'd, three sweeps flipped the identity, and
     Calendar read "Provider down" for every user on the platform while
     all four of its tools worked.

  3. **Sheets had a manifest and no provider.** `connector_registry`
     skipped it at every boot and it never appeared in the catalogue.

  4. **The catalogue advertised Connect on providers with no OAuth
     credentials.** LinkedIn and Outlook rendered live buttons whose
     only possible outcome was a 503.

The two probe bugs are mirror images — one reported healthy while
broken, the other broken while healthy — and both come from the same
root cause: a probe is only meaningful if it exercises the same
surface, with the same grant, that the tools do.
"""

from __future__ import annotations

from typing import Any, ClassVar, Optional

import pytest

from app.connectors import _google_base
from app.connectors.base import (
    ConnectorContext,
    ConnectorScopeMissing,
    ConnectorToolError,
)
from app.connectors.calendar.provider import CalendarProvider
from app.connectors.docs.provider import DocsProvider
from app.connectors.sheets import provider as sheets_mod
from app.connectors.sheets.provider import SheetsProvider


# ── test doubles ───────────────────────────────────────────────────


class _Resp:
    def __init__(self, status_code: int, text: str = "", json_body: Any = None):
        self.status_code = status_code
        self.text = text
        self._json = json_body if json_body is not None else {}
        self.headers = {"content-type": "application/json"}

    @property
    def is_success(self) -> bool:
        return 200 <= self.status_code < 300

    def json(self):
        return self._json


class _FakeClient:
    """Records every request so a test can assert the probe actually
    went to the network, and what it asked for."""

    def __init__(self, resp: Any = None, exc: Optional[Exception] = None):
        self.resp = resp or _Resp(404, "Requested entity was not found.")
        self.exc = exc
        self.calls: list[dict] = []

    async def get(self, url, headers=None, **kw):
        return await self.request("GET", url, headers=headers, **kw)

    async def request(self, method, url, headers=None, json=None, params=None, **kw):
        self.calls.append({
            "method": method, "url": url, "json": json, "params": params,
        })
        if self.exc:
            raise self.exc
        return self.resp


@pytest.fixture
def fake_google(monkeypatch):
    """Install a fake pooled client. Returns a setter so each test picks
    the response it wants."""
    holder: dict[str, _FakeClient] = {}

    def install(resp: Any = None, exc: Optional[Exception] = None) -> _FakeClient:
        client = _FakeClient(resp=resp, exc=exc)
        holder["client"] = client

        async def _get_client():
            return client

        monkeypatch.setattr(_google_base, "_get_google_client", _get_client)
        return client

    return install


def _identity(scopes: list[str], token: str = "tok"):
    class _I:
        access_token = token

        def __init__(self):
            self.scopes = scopes

    return _I()


# ── 1. the liveness contract ───────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [200, 400, 404, 429])
async def test_liveness_treats_any_answered_request_as_healthy(fake_google, code):
    """Google authenticates BEFORE it resolves the entity, so any status
    that isn't an auth/availability rejection proves the API is enabled
    and the token was accepted.

    429 is in this list deliberately: being throttled proves the API is
    up. Flipping an identity to `provider_down` because Google
    rate-limited our own probe would be a self-inflicted outage.
    """
    fake_google(_Resp(code))
    ok, _ = await _google_base.google_liveness("https://x/y", access_token="t")
    assert ok is True


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [401, 403, 500, 503])
async def test_liveness_rejects_auth_and_availability_failures(fake_google, code):
    fake_google(_Resp(code, "boom"))
    ok, detail = await _google_base.google_liveness("https://x/y", access_token="t")
    assert ok is False
    assert str(code) in detail


@pytest.mark.asyncio
async def test_liveness_carries_googles_own_403_text(fake_google):
    """Google's 403 body names the project and links the enable page.
    That sentence IS the diagnosis — losing it turns a 30-second fix
    into an investigation."""
    fake_google(_Resp(403, "Google Docs API has not been used in project 1049 ..."))
    ok, detail = await _google_base.google_liveness("https://x/y", access_token="t")
    assert ok is False
    assert "has not been used in project" in detail


@pytest.mark.asyncio
async def test_liveness_survives_transport_failure(fake_google):
    """A DNS blip must be unhealthy, not a 500 out of the probe sweep."""
    fake_google(exc=RuntimeError("connection reset"))
    ok, detail = await _google_base.google_liveness("https://x/y", access_token="t")
    assert ok is False
    assert "transport" in detail


# ── 2. the probes actually probe ───────────────────────────────────


@pytest.mark.asyncio
async def test_docs_probe_calls_google(fake_google, monkeypatch):
    """THE regression. Before the fix this made zero HTTP requests and
    returned ok unconditionally."""
    client = fake_google(_Resp(404, "Requested entity was not found."))
    monkeypatch.setattr(
        "app.connectors.docs.provider._resolve_token",
        _async_return("tok"),
    )
    res = await DocsProvider().health_probe(ConnectorContext(
        user_id="u1", channel="health_probe", request_id="r",
    ))
    assert res.ok is True
    assert len(client.calls) == 1, "the probe must reach Google, not just decrypt a token"
    assert "docs.googleapis.com" in client.calls[0]["url"]


@pytest.mark.asyncio
async def test_docs_probe_fails_when_api_is_disabled(fake_google, monkeypatch):
    client = fake_google(_Resp(403, "Google Docs API has not been used in project"))
    monkeypatch.setattr(
        "app.connectors.docs.provider._resolve_token",
        _async_return("tok"),
    )
    res = await DocsProvider().health_probe(ConnectorContext(
        user_id="u1", channel="health_probe", request_id="r",
    ))
    assert res.ok is False
    assert client.calls


@pytest.mark.asyncio
async def test_calendar_probe_does_not_use_calendarlist(fake_google, monkeypatch):
    """`calendarList` needs `calendar.readonly`, which this connector
    never requests. Probing it made every healthy identity read as
    down. The probe must exercise the same surface the tools do."""
    client = fake_google(_Resp(200, json_body={"items": []}))
    monkeypatch.setattr(
        "app.connectors.calendar.provider._resolve_token",
        _async_return("tok"),
    )
    res = await CalendarProvider().health_probe(ConnectorContext(
        user_id="u1", channel="health_probe", request_id="r",
    ))
    assert res.ok is True
    url = client.calls[0]["url"]
    assert "calendarList" not in url, (
        "calendarList requires calendar.readonly — a scope this "
        "connector's manifest lists as OPTIONAL and oauth.py therefore "
        "never requests. Probing it flips every working identity to "
        "provider_down."
    )
    assert "/events" in url


def test_calendar_manifest_does_not_require_readonly_scope():
    """Pins the premise of the test above. If someone later promotes
    `calendar.readonly` into the required scopes, the calendarList
    probe becomes legitimate and this test should be revisited
    deliberately rather than silently."""
    from app.services.connector_registry import get_registry, reset_registry_for_tests

    reset_registry_for_tests()
    registry = get_registry()
    registry.load_all()
    entry = registry.get("calendar")
    assert entry is not None
    assert "https://www.googleapis.com/auth/calendar.events" in entry.manifest.oauth.scopes
    assert (
        "https://www.googleapis.com/auth/calendar.readonly"
        not in entry.manifest.oauth.scopes
    )


# ── 3. Sheets ──────────────────────────────────────────────────────


def test_sheets_connector_loads_at_boot():
    """It shipped a manifest with no provider.py, so the registry hit
    `_SkipConnector` every boot and Sheets was invisible to everyone."""
    from app.services.connector_registry import get_registry, reset_registry_for_tests

    reset_registry_for_tests()
    registry = get_registry()
    registry.load_all()
    entry = registry.get("sheets")
    assert entry is not None, "sheets manifest present but provider missing"
    assert entry.provider.__class__.__name__ == "SheetsProvider"


@pytest.mark.parametrize("raw,expect", [
    ("1AbC_dEf-123", "1AbC_dEf-123"),
    ("https://docs.google.com/spreadsheets/d/1AbC_dEf-123/edit#gid=0", "1AbC_dEf-123"),
    ("https://docs.google.com/spreadsheets/d/1AbC_dEf-123", "1AbC_dEf-123"),
    ("", None),
    ("https://example.com/not-a-sheet", None),
])
def test_sheet_id_extraction(raw, expect):
    """Users paste links into a field the manifest calls
    `spreadsheet_id`. Accepting both is the difference between working
    and a 404."""
    assert sheets_mod._extract_sheet_id(raw) == expect


def test_range_is_percent_encoded_as_one_path_segment():
    """Sheet titles carry spaces and slashes. A raw `/` would split the
    path and address a different resource entirely."""
    out = sheets_mod._quote_range("Q3 Pipeline!A1:D20")
    assert " " not in out
    assert "%20" in out and "%21" in out
    assert "/" not in sheets_mod._quote_range("a/b!A1")


@pytest.mark.parametrize("raw,expect", [
    ([["a", "b"]], [["a", "b"]]),
    (["a", "b"], [["a", "b"]]),          # flat row lifted, not written as a column
    ([["a"], "b"], None),                # mixed → ambiguous, refuse
    ([], None),
    ("nope", None),
])
def test_row_coercion(raw, expect):
    assert sheets_mod._coerce_rows(raw) == expect


# `test_sheets_list_without_drive_scope_is_scope_missing` and
# `test_drive_query_literal_is_escaped` were removed on 2026-08-11 with
# `sheets__list_spreadsheets` itself. Answering ConnectorScopeMissing was
# a correct response to an impossible situation: `drive.readonly` sits in
# `scopes_optional`, which `_build_authorize_url` never sends, so the
# tool could not once have succeeded and told users to grant a permission
# the consent screen will never show. The reachability rule it was
# standing in for is now enforced for every connector by
# tests/test_no_tool_needs_an_unrequested_scope.py.

@pytest.mark.asyncio
async def test_sheets_append_inserts_rows_rather_than_overwriting(fake_google, monkeypatch):
    """Without `insertDataOption=INSERT_ROWS`, Sheets OVERWRITES
    whatever sits below the detected table instead of shifting it down
    — silent, unrecoverable data loss in the user's own spreadsheet."""
    client = fake_google(_Resp(200, json_body={
        "updates": {"updatedRange": "Sheet1!A5:B5", "updatedCells": 2},
    }))
    monkeypatch.setattr(
        sheets_mod, "_resolve_identity",
        _async_return(_identity(["https://www.googleapis.com/auth/spreadsheets"])),
    )
    res = await SheetsProvider().execute(
        "sheets__append_rows",
        {"spreadsheet_id": "sid", "range": "Sheet1!A:B", "values": [["x", "y"]]},
        ConnectorContext(user_id="u1", channel="web", request_id="r"),
    )
    assert res.__class__.__name__ == "ConnectorOk"
    params = client.calls[0]["params"]
    assert params["insertDataOption"] == "INSERT_ROWS"
    assert params["valueInputOption"] == "USER_ENTERED"


@pytest.mark.asyncio
async def test_sheets_update_uses_put_not_append(fake_google, monkeypatch):
    client = fake_google(_Resp(200, json_body={
        "updatedRange": "Sheet1!A1:B1", "updatedCells": 2,
    }))
    monkeypatch.setattr(
        sheets_mod, "_resolve_identity",
        _async_return(_identity(["https://www.googleapis.com/auth/spreadsheets"])),
    )
    await SheetsProvider().execute(
        "sheets__update_range",
        {"spreadsheet_id": "sid", "range": "Sheet1!A1:B1", "values": [["x", "y"]]},
        ConnectorContext(user_id="u1", channel="web", request_id="r"),
    )
    assert client.calls[0]["method"] == "PUT"
    assert ":append" not in client.calls[0]["url"]


@pytest.mark.asyncio
async def test_sheets_rejects_missing_range(fake_google, monkeypatch):
    fake_google(_Resp(200))
    monkeypatch.setattr(
        sheets_mod, "_resolve_identity",
        _async_return(_identity(["https://www.googleapis.com/auth/spreadsheets"])),
    )
    res = await SheetsProvider().execute(
        "sheets__read_range", {"spreadsheet_id": "sid"},
        ConnectorContext(user_id="u1", channel="web", request_id="r"),
    )
    assert isinstance(res, ConnectorToolError)


@pytest.mark.asyncio
async def test_sheets_probe_does_not_use_the_manifest_probe_tool(fake_google, monkeypatch):
    """The manifest names `sheets__list_spreadsheets` in `health.probe`
    to satisfy the registry lint, but that tool needs the optional
    drive scope. Probing through it would repeat the Calendar bug and
    mark every healthy Sheets identity down."""
    client = fake_google(_Resp(404))
    monkeypatch.setattr(
        sheets_mod, "_resolve_identity",
        _async_return(_identity(["https://www.googleapis.com/auth/spreadsheets"])),
    )
    res = await SheetsProvider().health_probe(ConnectorContext(
        user_id="u1", channel="health_probe", request_id="r",
    ))
    assert res.ok is True
    assert "sheets.googleapis.com" in client.calls[0]["url"]
    assert "drive" not in client.calls[0]["url"]


# ── 4. the catalogue gate fails SAFE ───────────────────────────────


@pytest.mark.asyncio
async def test_configured_list_returns_none_when_the_db_read_fails(monkeypatch):
    """None means "unknown", and the catalogue must gate nothing on it.

    Returning the env-var set here instead looks like a sane fallback and
    is a production trap: platform-api runs with NO `GOOGLE_OAUTH_*` /
    `GITHUB_OAUTH_*` env vars — credentials live in `provider_app_credentials`
    — so `_apps` is empty there. A single DB hiccup would confidently
    answer "nothing is configured" and grey out every connector on the
    page, Google and GitHub included.
    """
    from app.services import provider_apps

    class _Boom:
        def __call__(self, *a, **k): raise RuntimeError("db down")

    monkeypatch.setattr(
        "app.db.database.async_session_maker", _Boom(), raising=False,
    )
    assert await provider_apps.list_configured_async() is None


def test_catalogue_gate_is_skipped_when_configured_is_unknown():
    """Mirrors the endpoint's expression. Pinned as its own test because
    the bug it prevents is invisible in review: `x not in None` raises,
    and `x not in set()` is silently True for every connector."""
    for provider, ident, configured, expected in [
        ("linkedin", None, {"google"}, True),    # genuinely unconfigured
        ("google", None, {"google"}, False),     # configured
        ("linkedin", object(), {"google"}, False),  # connected already → keep tile
        ("linkedin", None, None, False),         # unknown → gate nothing
    ]:
        unconfigured = (
            configured is not None
            and provider not in configured
            and ident is None
        )
        assert unconfigured is expected, (provider, configured, ident)


# ── helpers ────────────────────────────────────────────────────────


def _async_return(value):
    async def _f(*a, **k):
        return value
    return _f
