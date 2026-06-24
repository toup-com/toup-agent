"""Browser-native search/fetch (no API key) — 2026-06-24.

The headless-browser fallbacks were dead in prod: the Chromium binary mounted
read-only from the host (/root/.cache/ms-playwright ← host /opt/toup/playwright)
was never populated by `patchright install`, so every Tier-3 browser search/fetch
failed to launch and silently returned empty. With Chromium installed, the browser
is a real fallback — but `browser_api.search` led with Google, which CAPTCHAs /
times out headless (8 s wasted) before returning nothing.

Validated on a live agent container that, of the headless engines, only Brave
Search (search.brave.com — Brave's own index, no API key, the same index Claude's
web search is built on) returns clean results; Bing wraps URLs in redirect junk,
Google/Startpage CAPTCHA, and DuckDuckGo's JS site renders empty. So `search()`
now uses Brave (browser-rendered) first and falls back to httpx DuckDuckGo-HTML;
Google/Bing/Startpage are dropped from the path. Both fallbacks are gated by
`browser_search_enabled` / `browser_fetch_enabled` kill-switches (default on).

Behavioral tests monkeypatch the engine module-globals (no network). Structural
tests follow the source-grep convention (see test_search_engine_race).
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import app.agent.skills.builtins.app_builder.browser_api as BA
from app.config import settings

_BA_SRC = Path(BA.__file__).read_text()
_TE_SRC = (Path(__file__).resolve().parent.parent
           / "app" / "agent" / "tool_executor.py").read_text()
_CFG_SRC = (Path(__file__).resolve().parent.parent
            / "app" / "config.py").read_text()


# ── Behavioral (no network) ───────────────────────────────────────────────

def test_search_uses_brave_first(monkeypatch):
    """search() returns Brave results and never touches the DDG fallback when
    Brave produces results."""
    calls = []

    async def fake_brave(q, c):
        calls.append("brave")
        return [{"title": "T", "url": "http://x/1", "snippet": "s"}]

    async def fake_ddg(q, c):
        calls.append("ddg")
        return [{"title": "DDG", "url": "http://d/1", "snippet": "s"}]

    monkeypatch.setattr(BA, "_brave_browser_search", fake_brave)
    monkeypatch.setattr(BA, "_ddg_html_search", fake_ddg)
    out = asyncio.run(BA.search("q", 5))
    assert calls == ["brave"]            # DDG never reached
    assert out[0]["url"] == "http://x/1"


def test_search_falls_back_to_ddg_when_brave_empty(monkeypatch):
    """When Brave comes back empty, the httpx DuckDuckGo-HTML last resort runs."""
    calls = []

    async def fake_brave(q, c):
        calls.append("brave")
        return []

    async def fake_ddg(q, c):
        calls.append("ddg")
        return [{"title": "DDG", "url": "http://d/1", "snippet": "s"}]

    monkeypatch.setattr(BA, "_brave_browser_search", fake_brave)
    monkeypatch.setattr(BA, "_ddg_html_search", fake_ddg)
    out = asyncio.run(BA.search("q", 5))
    assert calls == ["brave", "ddg"]
    assert out[0]["url"] == "http://d/1"


def test_search_never_calls_google(monkeypatch):
    """Regression: Google led the old chain and timed out headless (8 s tax).
    search() must never invoke _google_search."""
    called = {"google": False}

    async def fake_google(q, c):
        called["google"] = True
        return [{"title": "G", "url": "http://g/1", "snippet": ""}]

    async def empty(q, c):
        return []

    monkeypatch.setattr(BA, "_google_search", fake_google, raising=False)
    monkeypatch.setattr(BA, "_brave_browser_search", empty)
    monkeypatch.setattr(BA, "_ddg_html_search", empty)
    asyncio.run(BA.search("q", 5))
    assert called["google"] is False


def test_search_formatted_renders_numbered_block(monkeypatch):
    async def fake_search(q, c):
        return [{"title": "Title1", "url": "http://x/1", "snippet": "Snippet1"}]

    monkeypatch.setattr(BA, "search", fake_search)
    out = asyncio.run(BA.search_formatted("q", 5))
    assert "1. Title1" in out
    assert "http://x/1" in out
    assert "Snippet1" in out


# ── Structural (source-grep convention) ───────────────────────────────────

def test_brave_engine_exists_and_targets_brave():
    assert "async def _brave_browser_search" in _BA_SRC
    assert "search.brave.com" in _BA_SRC


def test_search_is_brave_primary_not_google():
    body = _BA_SRC.split("async def search(")[1].split("\nasync def ")[0]
    assert "_brave_browser_search" in body
    assert "_ddg_html_search" in body
    assert "_google_search" not in body  # no Google in the default path


def test_config_has_browser_killswitches_default_on():
    assert "browser_search_enabled: bool = True" in _CFG_SRC
    assert "browser_fetch_enabled: bool = True" in _CFG_SRC
    assert settings.browser_search_enabled is True
    assert settings.browser_fetch_enabled is True


def test_tool_executor_gates_browser_fallbacks_on_flags():
    assert "settings.browser_search_enabled" in _TE_SRC
    assert "settings.browser_fetch_enabled" in _TE_SRC
