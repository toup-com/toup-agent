"""OAuth 2.0 is a family of dialects, and the differences are load-bearing.

Every one of these was a real defect or a documented provider requirement:

  GitHub  answers form-urlencoded unless asked for JSON. We called
          r.json() on it, so every GitHub connect since launch 500'd
          AFTER the user consented (fixed 2026-08-08, #520).
  Notion  authenticates the CLIENT with HTTP Basic, wants a JSON body,
          and requires `Notion-Version` even on its token endpoint.
          Body credentials 401; a form body is 400 invalid_json.
  Jira    requires `audience=api.atlassian.com` on the authorize leg or
          the grant is not valid against the api.atlassian.com gateway
          where every Jira Cloud call goes, plus `prompt=consent`.
  PKCE    `code_verifier` used to be sent unconditionally, which made
          `use_pkce` decorative on the token leg. Google/GitHub/LinkedIn
          ignore a stray field so nothing broke and nothing revealed it.

The shape of these bugs is always the same: the user consents, the
provider issues a code, and WE fail. So assert the wire format, not the
happy path — a mock that accepts anything proves nothing.
"""

from __future__ import annotations

import httpx
import pytest

from app.api.oauth import _build_authorize_url, _exchange_code_for_tokens
from app.services.provider_apps import _TEMPLATES


class _Spy:
    """Captures exactly what went on the wire."""

    def __init__(self):
        self.seen = {}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.seen["headers"] = dict(request.headers)
        self.seen["content"] = request.content.decode() if request.content else ""
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json={"access_token": "tok", "token_type": "bearer"},
        )


# Captured ONCE at import. Reading `httpx.AsyncClient.__init__` inside the
# helper meant a second call in the same test wrapped the FIRST patch, so
# the new spy was never installed and its `seen` stayed empty — the
# harness lying rather than the code.
_REAL_ASYNC_CLIENT_INIT = httpx.AsyncClient.__init__


async def _exchange(monkeypatch, **kwargs) -> dict:
    spy = _Spy()

    def patched(self, *a, **kw):
        kw["transport"] = httpx.MockTransport(spy)
        _REAL_ASYNC_CLIENT_INIT(self, *a, **kw)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", patched)
    await _exchange_code_for_tokens(
        token_url="https://provider.example/token",
        code="abc", code_verifier="VERIFIER",
        client_id="CID", client_secret="SEC",
        redirect_uri="https://toup.ai/api/oauth/callback",
        **kwargs,
    )
    return spy.seen


# ── Notion: the three-axis deviation ─────────────────────────────────


@pytest.mark.asyncio
async def test_notion_authenticates_with_basic_and_sends_json(monkeypatch):
    """Driven from `_TEMPLATES["notion"]`, not from literals.

    Passing the dialect in by hand proved only that the exchanger CAN
    speak Basic — deleting `token_auth_style` from the template left this
    green while every real Notion connect would 401. Mutation-caught.
    """
    import base64

    t = _TEMPLATES["notion"]
    seen = await _exchange(
        monkeypatch,
        use_pkce=t.get("use_pkce", True),
        token_auth_style=t.get("token_auth_style", "body"),
        token_body_format=t.get("token_body_format", "form"),
        extra_headers=t.get("extra_token_headers"),
    )
    expected = "Basic " + base64.b64encode(b"CID:SEC").decode()
    assert seen["headers"].get("authorization") == expected
    assert seen["headers"].get("notion-version") == "2026-03-11"
    assert seen["headers"]["content-type"].startswith("application/json")
    # Credentials must NOT also appear in the body.
    assert "SEC" not in seen["content"]
    import json as _json
    assert _json.loads(seen["content"])["grant_type"] == "authorization_code"


# ── Everyone else: unchanged form-encoded body credentials ───────────


@pytest.mark.asyncio
async def test_default_dialect_is_still_form_with_body_credentials(monkeypatch):
    seen = await _exchange(monkeypatch)
    assert "authorization" not in seen["headers"]
    assert seen["headers"]["content-type"] == "application/x-www-form-urlencoded"
    assert "client_secret=SEC" in seen["content"]
    assert seen["headers"].get("accept") == "application/json"  # the GitHub fix


@pytest.mark.asyncio
async def test_code_verifier_is_sent_only_when_pkce_is_on(monkeypatch):
    on = await _exchange(monkeypatch, use_pkce=True)
    assert "code_verifier=VERIFIER" in on["content"]

    off = await _exchange(monkeypatch, use_pkce=False)
    assert "code_verifier" not in off["content"]


# ── Jira: required authorize params ──────────────────────────────────


def _authorize(provider: str, **kw) -> str:
    t = _TEMPLATES[provider]
    return _build_authorize_url(
        base_url=t["authorize_url"],
        client_id="CID",
        redirect_uri="https://toup.ai/api/oauth/callback",
        scopes=["read:jira-work"],
        state="STATE",
        code_challenge="CHAL",
        use_pkce=t.get("use_pkce", True),
        provider_name=provider,
        extra_params=t.get("extra_authorize_params"),
        **kw,
    )


def test_jira_authorize_carries_the_required_audience_and_prompt():
    url = _authorize("jira")
    assert "audience=api.atlassian.com" in url
    assert "prompt=consent" in url


def test_switch_account_cannot_clobber_a_required_prompt():
    """`force_account_selection` sets prompt=select_account. For Atlassian
    `prompt=consent` is REQUIRED, so provider params are applied last —
    otherwise the requirement would hold for first connects and silently
    break only for users who hit Switch account."""
    url = _authorize("jira", force_account_selection=True)
    assert "prompt=consent" in url
    assert "select_account" not in url


def test_switch_account_still_works_where_nothing_is_required():
    url = _authorize("google", force_account_selection=True)
    assert "prompt=select_account" in url


def test_notion_authorize_keeps_owner_user_and_appends_with_ampersand():
    """`owner=user` is required and lives in the template's base URL, so
    the builder must notice the existing query string."""
    url = _authorize("notion")
    assert "owner=user" in url
    assert url.count("?") == 1
    assert "&client_id=CID" in url


# ── The templates themselves ─────────────────────────────────────────


def test_every_template_declares_a_dialect_the_exchanger_understands():
    for name, t in _TEMPLATES.items():
        assert t.get("token_auth_style", "body") in ("body", "basic"), name
        assert t.get("token_body_format", "form") in ("form", "json"), name


def test_google_refresh_token_params_are_still_pinned():
    """The hourly-disconnect regression. Unrelated to this change, and
    exactly the kind of thing a refactor of this function would drop."""
    url = _authorize("google")
    assert "access_type=offline" in url
    assert "prompt=consent" in url


def test_the_db_credential_path_carries_every_declared_quirk():
    """The ONLY path Jira and Notion will ever take.

    `_from_db_credentials` assembles a ProviderAppConfig from the template
    plus an admin-entered credential row. It used to copy three fields, so
    a template could declare `token_auth_style: basic` and the running
    config would still send body credentials — a 401 after the user had
    already consented, with the template looking correct in review.

    Asserted structurally: every quirk key any template declares must be
    read by that function.
    """
    import inspect

    from app.services import provider_apps as pa

    # `get_provider_app_async`, NOT `_from_db_credentials` — the latter
    # unconditionally `return None` ("sync path falls through; async
    # sibling handles DB"), so it constructs nothing. Asserting against
    # it would have passed while the real path dropped every quirk.
    src = inspect.getsource(pa.get_provider_app_async)
    assert "return None" in inspect.getsource(pa._from_db_credentials), (
        "the sync loader now builds a config too — assert quirks on it as well"
    )
    declared = {k for t in _TEMPLATES.values() for k in t}
    declared -= {"authorize_url", "token_url", "use_pkce"}  # already asserted below
    assert declared, "no quirk keys in any template — did the fields get dropped?"
    for key in sorted(declared):
        assert f'"{key}"' in src or f"'{key}'" in src, (
            f"template key {key!r} is never read by _from_db_credentials, so it is "
            f"silently dropped for admin-entered credentials"
        )
    for key in ("authorize_url", "token_url", "use_pkce"):
        assert key in src
