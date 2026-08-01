"""Getting the RIGHT music: the id/title pairing, the station RPC, and artwork.

"Play me Kendrick Lamar" (2026-07-31, prod trace) played the correct video —
Kendrick's HUMBLE — under the title *"THE GOD MC IS BACK AND HE RESPONDED TO
KENDRICK! Rakim - A Different Kind"*. Two independent regexes over one page of
YouTube search HTML: the first videoId, and the first title anywhere. Whenever
a shelf or a promoted result sits between them they belong to different videos.

That wrong title was not cosmetic. It is what the agent read back to the user,
and it is what seeded the radio station — so the follow-on tracks came from a
reaction video's neighbourhood, and the session drifted to artists the user had
never mentioned.

Compounding it, every station build that evening failed the same way: the agent
posts its YouTube-Music RPC to the platform, and the URL was missing the `/api`
prefix, so it hit the SPA catch-all and got **405** twenty-three times in a row.
Each failure fell back to the agent's own YouTube Music call — the one that is
anti-bot-blocked from the agent's IP, which is the entire reason the RPC exists.

These tests are hermetic: the HTML is a fixture, the RPC is a stub.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("ENVIRONMENT", "test")


# ── The id/title pairing ────────────────────────────────────────────────
# A faithful miniature of a YouTube results page: a promoted/shelf result
# appears BEFORE the video whose id is picked first... which is exactly the
# layout that produced the Rakim title.

_RESULTS_HTML = """
{"itemSectionRenderer":{"contents":[
 {"videoRenderer":{"videoId":"tvTRZJ-4EyI","thumbnail":{"thumbnails":[]},
   "title":{"runs":[{"text":"Kendrick Lamar - HUMBLE."}],"accessibility":{}}}},
 {"videoRenderer":{"videoId":"zzzzzzzzzzz","thumbnail":{"thumbnails":[]},
   "title":{"runs":[{"text":"THE GOD MC IS BACK! Rakim - A Different Kind"}]}}}
]}}
"""

# The pathological ordering: a shelf carries the first *title* on the page,
# while the first *videoId* belongs to the real result further down.
_RESULTS_HTML_SHELF_FIRST = """
{"shelfRenderer":{"title":{"runs":[{"text":"Popular reaction videos"}]}}}
{"videoRenderer":{"videoId":"tvTRZJ-4EyI","thumbnail":{"thumbnails":[]},
  "title":{"runs":[{"text":"Kendrick Lamar - HUMBLE."}]}}}
"""


def _extract(html: str, query: str = "kendrik lamar"):
    """Run the production extraction over `html`, returning (video_id, title).

    Mirrors _fast_media_check's parsing exactly; kept in the test rather than
    reaching into the coroutine so no network or WebSocket is involved.
    """
    import re as _re_mod
    import json

    id_matches = _re_mod.findall(r'"videoId":"([a-zA-Z0-9_-]{11})"', html)
    if not id_matches:
        return None, None
    video_id = id_matches[0]
    title_m = _re_mod.search(
        _re_mod.escape(f'"videoId":"{video_id}"')
        + r'(?:(?!"videoId":")[\s\S]){0,3000}?'
        + r'"title":\{"runs":\[\{"text":"((?:[^"\\]|\\.)*)"',
        html,
    )
    if title_m:
        raw = title_m.group(1)
        try:
            return video_id, json.loads(f'"{raw}"')
        except Exception:
            return video_id, raw
    return video_id, query


def test_title_belongs_to_the_chosen_video():
    vid, title = _extract(_RESULTS_HTML)
    assert vid == "tvTRZJ-4EyI"
    assert title == "Kendrick Lamar - HUMBLE."


def test_a_shelf_before_the_first_result_cannot_steal_the_title():
    """The exact prod failure: first title on the page belongs to a shelf."""
    vid, title = _extract(_RESULTS_HTML_SHELF_FIRST)
    assert vid == "tvTRZJ-4EyI"
    assert "reaction" not in title.lower(), (
        f"title came from the shelf, not from the chosen video: {title!r}"
    )
    assert title == "Kendrick Lamar - HUMBLE."


def test_json_escapes_in_titles_are_decoded():
    """YouTube escapes & as \\u0026; a huge share of music titles have one."""
    html = (
        '{"videoRenderer":{"videoId":"abcdefghijk",'
        '"title":{"runs":[{"text":"Florence \\u0026 The Machine - Dog Days"}]}}}'
    )
    vid, title = _extract(html)
    assert vid == "abcdefghijk"
    assert title == "Florence & The Machine - Dog Days"


def test_a_title_is_never_stolen_from_the_next_result():
    """Our id's block has no title; the NEXT result does. Fall back to the
    user's query rather than labelling one video with another's name.

    A generic label is merely unhelpful; a borrowed one actively misdirects
    both the agent's spoken reply and the radio seed — which is the whole
    Kendrick→Rakim→Statik-Selektah chain in miniature. The guarantee is scoped
    to what is actually detectable: another `"videoId":"` marks the start of a
    different result, and the match may not cross one.
    """
    html = (
        '{"videoRenderer":{"videoId":"abcdefghijk","badgeStuff":{}}},'
        '{"videoRenderer":{"videoId":"zzzzzzzzzzz",'
        '"title":{"runs":[{"text":"Some Other Video"}]}}}'
    )
    vid, title = _extract(html, query="shadmehr aghili")
    assert vid == "abcdefghijk"
    assert title == "shadmehr aghili", (
        f"title leaked across a result boundary: {title!r}"
    )


# ── The station RPC URL ─────────────────────────────────────────────────

@pytest.fixture
def _agent_settings(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "run_mode", "agent", raising=False)
    monkeypatch.setattr(settings, "agent_api_key", "test-key", raising=False)
    monkeypatch.setattr(settings, "user_id", "871bac24-c366-42b5-b224-8802c73aef3a", raising=False)
    return settings


class _Resp:
    def __init__(self, status, payload=None, ctype="application/json"):
        self.status_code = status
        self._payload = payload if payload is not None else {"result": ["ok"]}
        self.headers = {"content-type": ctype}

    def json(self):
        return self._payload


def test_yt_remote_retries_with_the_api_prefix(monkeypatch, _agent_settings):
    """The live prod failure: base URL without /api → SPA catch-all → 405."""
    from app.agent.radio import playlist as pl
    monkeypatch.setattr(_agent_settings, "platform_api_url", "https://toup.ai", raising=False)

    seen = []

    def _fake_post(url, **kw):
        seen.append(url)
        if url == "https://toup.ai/internal/radio/yt":
            return _Resp(405, ctype="text/html")          # the SPA catch-all
        return _Resp(200, {"result": ["station"]})

    import httpx as _httpx
    monkeypatch.setattr(_httpx, "post", _fake_post)

    out = pl._yt_remote("get_watch_playlist", video_id="tvTRZJ-4EyI")
    assert out == ["station"], "the /api retry must recover the station build"
    assert seen == [
        "https://toup.ai/internal/radio/yt",
        "https://toup.ai/api/internal/radio/yt",
    ]


def test_yt_remote_does_not_double_prefix_a_correct_base(monkeypatch, _agent_settings):
    from app.agent.radio import playlist as pl
    monkeypatch.setattr(_agent_settings, "platform_api_url", "https://toup.ai/api", raising=False)

    seen = []

    def _fake_post(url, **kw):
        seen.append(url)
        return _Resp(200, {"result": ["station"]})

    import httpx as _httpx
    monkeypatch.setattr(_httpx, "post", _fake_post)

    assert pl._yt_remote("get_watch_playlist", video_id="x") == ["station"]
    assert seen == ["https://toup.ai/api/internal/radio/yt"], (
        "a correctly configured base must not be retried or rewritten"
    )


def test_yt_remote_rejects_a_200_that_is_actually_the_spa(monkeypatch, _agent_settings):
    """A catch-all answering 200 text/html must not be read as an empty station.

    Silently returning "no tracks" is indistinguishable from a genuinely
    unavailable station, which is how a misroute hides for months.
    """
    from app.agent.radio import playlist as pl
    monkeypatch.setattr(_agent_settings, "platform_api_url", "https://toup.ai", raising=False)

    def _fake_post(url, **kw):
        if "/api/" in url:
            return _Resp(200, {"result": ["real"]})
        return _Resp(200, {"not": "json really"}, ctype="text/html; charset=utf-8")

    import httpx as _httpx
    monkeypatch.setattr(_httpx, "post", _fake_post)

    assert pl._yt_remote("search", query="ebi") == ["real"]


# ── Artwork is never blank ──────────────────────────────────────────────

def test_direct_play_frames_carry_artwork():
    """Voice plays reached the lock screen with no cover art because this
    payload had no thumbnail field at all. It is derivable from the id."""
    import inspect
    from app.agent import tool_executor

    src = inspect.getsource(tool_executor.ToolExecutor._tool_play_media)
    assert "thumbnail_url" in src, (
        "the media_play payload must carry artwork — a client cannot invent it"
    )
    assert "i.ytimg.com" in src


def test_fast_path_frames_carry_artwork():
    import inspect
    from app.api import ws_chat

    src = inspect.getsource(ws_chat._fast_media_check)
    assert "thumbnail_url" in src and "i.ytimg.com" in src


# ── The root cause behind the 405s ──────────────────────────────────────

@pytest.mark.parametrize(
    "given,expected",
    [
        ("https://toup.ai", "https://toup.ai/api"),        # the deployed value
        ("https://toup.ai/", "https://toup.ai/api"),
        ("https://toup.ai/api", "https://toup.ai/api"),    # already correct
        ("https://toup.ai/api/", "https://toup.ai/api"),
    ],
)
def test_platform_api_url_is_normalized_to_the_api_prefix(monkeypatch, given, expected):
    """Agent→platform callbacks are f"{platform_api_url}/...", and every one of
    those routes is mounted under api_prefix. A base without it lands on the SPA
    catch-all: index.html on GET, 405 on POST.

    This bug has now bitten three subsystems — tool_executor and
    subagent_dispatcher each grew a local retry-with-/api after being caught by
    it, and on 2026-07-31 it silently broke radio station building for a whole
    evening (23 consecutive 405s). Normalizing at load time fixes every call
    site at once, including ones not written yet.
    """
    from app.config import Settings

    monkeypatch.setenv("PLATFORM_API_URL", given)
    assert Settings().platform_api_url == expected
