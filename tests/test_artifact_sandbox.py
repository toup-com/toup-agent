"""Round 12 — the artifact sandbox boundary.

An artifact is HTML a language model wrote, so the security question is not
"is this page well-behaved" but "what can it reach if it is not". These tests
pin the four containments, each of which is load-bearing on its own:

  1. **No auth cookie is reachable.** The serving route never reads a cookie
     for auth and never sets one — unlike ``apps_proxy.preview_proxy``, which
     deliberately sets ``preview_token`` (correct for an Expo dev server,
     wrong for model-authored markup).
  2. **No non-allowlisted network.** The CSP permits exactly one external
     origin, ``cdnjs.cloudflare.com``, and nothing else — no fetch, no
     WebSocket, no image host, no font host, no form post.
  3. **Nobody else can frame it.** ``frame-ancestors`` is a Toup allowlist,
     never ``*``.
  4. **The frame is opaque.** Every embedder uses ``sandbox="allow-scripts"``
     WITHOUT ``allow-same-origin`` — the combination that would hand the page
     its own real origin back and undo the sandbox entirely.

Plus the credential-separation property: an artifact token fetches one static
file and is rejected everywhere else, including by the app-preview path that
can reach the user's agent chat.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from app.api.artifact_proxy import artifact_csp, artifact_headers, artifact_url
from app.config import settings
from app.services.auth_service import (
    create_access_token,
    create_artifact_token,
    create_preview_token,
    decode_access_token,
    decode_artifact_token,
    decode_preview_token,
)

REPO = pathlib.Path(__file__).resolve().parents[2]
FRONTEND = REPO / "frontend" / "src"


def _directive(csp: str, name: str) -> str:
    for part in csp.split(";"):
        part = part.strip()
        if part == name or part.startswith(name + " "):
            return part[len(name):].strip()
    raise AssertionError(f"CSP has no {name!r} directive: {csp}")


# ── 1. The CSP ───────────────────────────────────────────────────────

def test_csp_has_every_required_directive():
    csp = artifact_csp()
    for name in (
        "default-src", "script-src", "style-src", "img-src", "font-src",
        "connect-src", "object-src", "base-uri", "form-action",
        "frame-ancestors",
    ):
        _directive(csp, name)


def test_only_cdnjs_is_reachable_and_only_for_code_and_style():
    csp = artifact_csp()
    cdn = "https://cdnjs.cloudflare.com"
    assert _directive(csp, "script-src").split() == [
        "'self'", "'unsafe-inline'", "'unsafe-eval'", "blob:", cdn,
    ]
    assert _directive(csp, "style-src").split() == ["'self'", "'unsafe-inline'", cdn]
    assert _directive(csp, "font-src").split() == ["'self'", "data:", cdn]
    # Data exfiltration surfaces: no external host on ANY of these.
    assert _directive(csp, "connect-src") == "'self'"
    assert _directive(csp, "img-src").split() == ["'self'", "data:", "blob:"]
    assert _directive(csp, "form-action") == "'self'"
    assert _directive(csp, "object-src") == "'none'"
    assert _directive(csp, "base-uri") == "'none'"


_FETCH_DIRECTIVES = (
    "default-src", "script-src", "style-src", "img-src", "font-src",
    "connect-src", "object-src", "form-action",
)


def test_no_wildcard_or_unlisted_host_in_any_fetch_directive():
    """Checked per fetch directive, NOT over the whole policy string.

    `frame-ancestors` legitimately names Toup origins, so a substring scan of
    the whole header would either fail on those or have to whitelist them and
    stop catching anything.
    """
    csp = artifact_csp()
    allowed_hosts = {"https://cdnjs.cloudflare.com"}
    allowed_keywords = {
        "'self'", "'none'", "'unsafe-inline'", "'unsafe-eval'", "blob:", "data:",
    }
    for name in _FETCH_DIRECTIVES:
        for source in _directive(csp, name).split():
            assert source != "*", f"{name} is a wildcard"
            assert source in allowed_keywords or source in allowed_hosts, (
                f"{name} allows an unlisted source: {source}"
            )
            assert not source.startswith("http:"), f"{name} allows cleartext: {source}"
    assert "'unsafe-hashes'" not in csp


def test_frame_ancestors_is_a_toup_allowlist_not_a_wildcard():
    ancestors = _directive(artifact_csp(), "frame-ancestors")
    assert ancestors, "empty frame-ancestors would allow nothing OR everything"
    assert "*" not in ancestors
    assert "https://toup.ai" in ancestors
    for origin in ancestors.split():
        assert re.match(r"^https?://[a-z0-9.\-]+(:\d+)?$", origin), origin


def test_headers_carry_the_csp_and_never_a_cookie():
    headers = artifact_headers()
    assert headers["Content-Security-Policy"] == artifact_csp()
    assert not any(h.lower() == "set-cookie" for h in headers)
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["Referrer-Policy"] == "no-referrer"
    assert "no-store" in headers["Cache-Control"]


def test_the_serving_route_never_sets_a_cookie(monkeypatch):
    """Source-level: `set_cookie` must not appear on the artifact path.

    The neighbouring Expo proxy DOES set one — this test is what keeps a
    future copy-paste from bringing it across.
    """
    src = (REPO / "backend" / "app" / "api" / "artifact_proxy.py").read_text()
    # `.set_cookie(` — the CALL, not the words. Prose about not setting
    # cookies must not fail the test that enforces it.
    assert ".set_cookie(" not in src
    assert "request.cookies" not in src
    assert "cookies.get(" not in src

    # And the control: the thing it was modelled on DOES set one, so the
    # assertion above is testing a real difference, not an absent feature.
    expo = (REPO / "backend" / "app" / "api" / "apps_proxy.py").read_text()
    assert ".set_cookie(" in expo
    assert "request.cookies" in expo


# ── 2. Token separation ──────────────────────────────────────────────

def test_artifact_token_only_opens_its_own_slug():
    tok = create_artifact_token("user-1", "budget-tracker")
    assert decode_artifact_token(tok, "budget-tracker") == "user-1"
    assert decode_artifact_token(tok, "other-app") is None
    assert decode_artifact_token("garbage", "budget-tracker") is None


def test_artifact_token_is_rejected_by_general_auth():
    """An artifact URL leaks easily (referrer, history, a screenshot). It must
    be worth exactly one static file."""
    tok = create_artifact_token("user-1", "budget-tracker")
    assert decode_access_token(tok) is None


def test_artifact_token_cannot_reach_the_agent_chat_path():
    """`app_preview` tokens are accepted by /api/apps/{id}/chat. An artifact
    token must NOT be — that is why it has its own scope rather than reusing
    the preview one."""
    tok = create_artifact_token("user-1", "budget-tracker")
    assert decode_preview_token(tok, "budget-tracker") is None


def test_an_account_token_cannot_be_used_as_an_artifact_token():
    """The reverse direction: the artifact origin must never accept the
    account credential, or the separate origin buys nothing."""
    account = create_access_token("user-1")
    assert decode_artifact_token(account, "budget-tracker") is None
    # Control — the account token IS valid for general auth, so the
    # assertion above is a scope check, not a broken-token check.
    assert decode_access_token(account) == "user-1"


def test_artifact_token_ttl_is_bounded():
    assert 0 < settings.artifact_token_expire_minutes <= 60 * 24


# ── 3. The URL ───────────────────────────────────────────────────────

def test_artifact_url_uses_the_isolated_origin_when_configured(monkeypatch):
    monkeypatch.setattr(settings, "artifact_origin", "https://artifacts.toup.ai")
    url = artifact_url("budget-tracker", "TOK")
    assert url.startswith("https://artifacts.toup.ai/api/artifacts/budget-tracker?token=")
    # The SPA's own origin must not appear — that is the whole isolation.
    assert "toup.ai/api/artifacts" in url and "https://toup.ai/" not in url


def test_artifact_url_falls_back_to_a_relative_path_for_dev(monkeypatch):
    monkeypatch.setattr(settings, "artifact_origin", "")
    assert artifact_url("x-app", "TOK") == "/api/artifacts/x-app?token=TOK"


# ── 4. The embedder ──────────────────────────────────────────────────

_FRAME_FILES = [
    FRONTEND / "modules" / "workspace" / "AppArtifactFrame.tsx",
]


@pytest.mark.parametrize("path", _FRAME_FILES, ids=lambda p: p.name)
def test_artifact_frames_never_combine_allow_scripts_with_allow_same_origin(path):
    """allow-scripts + allow-same-origin lets the framed page remove its own
    sandbox attribute and reload itself unsandboxed. Together they are worse
    than no sandbox at all, because they read as protection."""
    assert path.is_file(), f"missing embedder: {path}"
    src = path.read_text()
    sandboxes = re.findall(r'sandbox=(?:"([^"]*)"|\{`([^`]*)`\})', src)
    assert sandboxes, f"{path.name} renders no sandboxed frame"
    for a, b in sandboxes:
        value = a or b
        assert "allow-scripts" in value, value
        assert "allow-same-origin" not in value, (
            f"{path.name} combines allow-scripts with allow-same-origin: {value}"
        )
        assert "allow-top-navigation" not in value, value


def test_no_artifact_frame_grants_popups_or_downloads():
    src = _FRAME_FILES[0].read_text()
    for value in re.findall(r'sandbox="([^"]*)"', src):
        assert "allow-popups" not in value, value
        assert "allow-modals" not in value, value
