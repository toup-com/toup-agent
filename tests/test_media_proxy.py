"""Tests for the media-proxy audio URL extractor.

Covers `GET /api/media/{video_id}/audio_url` used by the mobile hybrid
player (foreground WebView video → background TrackPlayer audio handoff).

yt-dlp is mocked — we never hit YouTube in CI. Every test patches
`app.api.media_proxy._extract_audio` so the contract-level behaviour is
what's under test, not the extractor plumbing.
"""
from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from typing import AsyncIterator
from unittest.mock import patch

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

pytestmark = pytest.mark.asyncio


# The conftest autouse fixture reaches for Postgres to init schema. These
# tests don't touch the DB — the endpoint is pure yt-dlp extraction — so
# override the autouse with a no-op at the module scope to keep the suite
# runnable on a workstation without Postgres.
@pytest_asyncio.fixture(autouse=True)
async def _reset_database():
    yield


def _build_app(*, authed: bool = True) -> FastAPI:
    """Build a minimal FastAPI app that mounts only media_proxy.

    `authed=True` overrides `get_current_user` to always return a fake
    user. `authed=False` leaves the real dep in place so 401s can be
    exercised.

    The shared `client` fixture in conftest builds an app that imports
    vps → stripe, which isn't in every local dev venv — hence we keep
    this suite self-contained.
    """
    from app.api.auth import get_current_user
    from app.api.media_proxy import router as media_proxy_router
    from app.config import settings

    app = FastAPI()
    app.include_router(media_proxy_router, prefix=settings.api_prefix)
    if authed:
        app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(
            id="test-user-id",
            email="test@example.com",
        )
    return app


@pytest_asyncio.fixture
async def client() -> AsyncIterator[AsyncClient]:
    app = _build_app(authed=True)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest_asyncio.fixture
async def unauth_client() -> AsyncIterator[AsyncClient]:
    """Client that does NOT override auth — real `get_current_user`
    runs and returns 401 without a valid Bearer token."""
    app = _build_app(authed=False)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


_FAKE_URL = (
    "https://rr5---sn-abc123.googlevideo.com/videoplayback"
    "?expire=9999999999&sig=xyz&itag=140"
)

_FAKE_ART_SMALL = "https://i.ytimg.com/vi/abc/default.jpg"
_FAKE_ART_BIG = "https://i.ytimg.com/vi/abc/maxresdefault.jpg"


def _fake_info_ok() -> dict:
    return {
        "url": _FAKE_URL,
        "duration": 234,
        "ext": "m4a",
        "abr": 128,
        "title": "Drake - Nonstop",
        "thumbnails": [
            {"url": _FAKE_ART_SMALL, "width": 120, "height": 90},
            {"url": _FAKE_ART_BIG, "width": 1280, "height": 720},
        ],
        "thumbnail": _FAKE_ART_SMALL,
    }


def _wrap_with_helpers(info: dict) -> dict:
    """Push a fake yt-dlp info dict through the real helper functions so
    tests exercise the actual response-shape code paths."""
    from app.api.media_proxy import _mime_from_ext, _parse_expires, _pick_artwork
    return {
        "url": info["url"],
        "expires_at": _parse_expires(info["url"]),
        "duration": info.get("duration") or 0,
        "mime_type": _mime_from_ext(info.get("ext") or "m4a"),
        "ext": info.get("ext") or "m4a",
        "bitrate": int(info.get("abr") or 0),
        "title": info.get("title") or "",
        "artwork_url": _pick_artwork(info),
    }


async def test_audio_url_happy_path(client):
    def fake(vid: str) -> dict:
        assert vid == "abc123def45"
        return _wrap_with_helpers(_fake_info_ok())

    with patch("app.api.media_proxy._extract_audio", side_effect=fake):
        r = await client.get("/api/media/abc123def45/audio_url")

    assert r.status_code == 200
    body = r.json()
    assert body["url"] == _FAKE_URL
    assert body["expires_at"] == 9999999999
    assert body["duration"] == 234
    assert body["mime_type"] == "audio/mp4"
    assert body["ext"] == "m4a"
    assert body["bitrate"] == 128
    assert body["title"] == "Drake - Nonstop"
    # Largest thumbnail wins (1280x720 > 120x90).
    assert body["artwork_url"] == _FAKE_ART_BIG


async def test_audio_url_expires_defaults_when_param_missing(client):
    """A stream URL with no `expire` param should get a 6h fallback TTL."""
    from app.api.media_proxy import _mime_from_ext, _parse_expires

    url_without_expire = "https://cdn.example.com/stream.m4a?sig=xyz"
    now = int(time.time())

    def fake(vid: str) -> dict:
        return {
            "url": url_without_expire,
            "expires_at": _parse_expires(url_without_expire),
            "duration": 200,
            "mime_type": _mime_from_ext("m4a"),
            "ext": "m4a",
            "bitrate": 128,
            "title": "x",
        }

    with patch("app.api.media_proxy._extract_audio", side_effect=fake):
        r = await client.get("/api/media/abc/audio_url")

    assert r.status_code == 200
    body = r.json()
    # 6h default ± 60s for test clock skew
    assert abs(body["expires_at"] - (now + 6 * 3600)) < 60


async def test_audio_url_no_stream_returns_502(client):
    """yt-dlp returned metadata but no URL (private / deleted / blocked)."""
    def fake(vid: str) -> dict:
        return {"error": "no_stream_url"}

    with patch("app.api.media_proxy._extract_audio", side_effect=fake):
        r = await client.get("/api/media/deadvid/audio_url")

    assert r.status_code == 502
    assert r.json()["error"] == "no_stream_url"


async def test_audio_url_extractor_exception_returns_502(client):
    """yt-dlp itself raised — network error, extractor broken, etc."""
    def fake(vid: str) -> dict:
        raise RuntimeError("yt-dlp: video unavailable")

    with patch("app.api.media_proxy._extract_audio", side_effect=fake):
        r = await client.get("/api/media/err123/audio_url")

    assert r.status_code == 502
    body = r.json()
    assert body["error"] == "extraction_failed"
    assert "yt-dlp" in body["detail"]


async def test_audio_url_timeout_returns_504(client):
    """Extractor exceeds the 25s budget — endpoint returns 504 cleanly."""
    async def immediate_timeout(*_a, **_kw):
        raise asyncio.TimeoutError

    def noop_extract(_vid: str) -> dict:
        return {"error": "unused"}

    # Patch wait_for itself so we don't actually have to wait 25s; patch
    # _extract_audio so the executor future scheduled before wait_for's
    # exception doesn't drag yt_dlp in (not installed in every dev venv).
    with patch("app.api.media_proxy.asyncio.wait_for", side_effect=immediate_timeout), \
         patch("app.api.media_proxy._extract_audio", side_effect=noop_extract):
        r = await client.get("/api/media/slowvid/audio_url")

    assert r.status_code == 504
    assert r.json()["error"] == "extraction_timeout"


async def test_parse_expires_handles_malformed_param():
    """Integer parse should fall back cleanly, not crash."""
    from app.api.media_proxy import _parse_expires

    now = int(time.time())

    for bad_url in [
        "https://x.com/a?expire=notanumber",
        "https://x.com/a?expire=",
        "https://x.com/a",
        "not a url at all",
    ]:
        v = _parse_expires(bad_url)
        # Should be ~6h from now
        assert abs(v - (now + 6 * 3600)) < 60, f"bad url: {bad_url}"


async def test_mime_from_ext_mapping():
    from app.api.media_proxy import _mime_from_ext

    assert _mime_from_ext("m4a") == "audio/mp4"
    assert _mime_from_ext("M4A") == "audio/mp4"
    assert _mime_from_ext("webm") == "audio/webm"
    assert _mime_from_ext("mp3") == "audio/mpeg"
    # Unknown → safe iOS default
    assert _mime_from_ext("opus") == "audio/mp4"
    assert _mime_from_ext("") == "audio/mp4"
    assert _mime_from_ext(None) == "audio/mp4"  # type: ignore[arg-type]


async def test_pick_artwork_prefers_largest():
    from app.api.media_proxy import _pick_artwork

    info = {
        "thumbnails": [
            {"url": "small.jpg", "width": 120, "height": 90},
            {"url": "big.jpg", "width": 1280, "height": 720},
            {"url": "medium.jpg", "width": 480, "height": 360},
        ],
        "thumbnail": "fallback.jpg",
    }
    assert _pick_artwork(info) == "big.jpg"


async def test_pick_artwork_falls_back_to_flat_thumbnail():
    """Empty thumbnails list → use the flat `thumbnail` field."""
    from app.api.media_proxy import _pick_artwork

    assert _pick_artwork({"thumbnails": [], "thumbnail": "flat.jpg"}) == "flat.jpg"
    assert _pick_artwork({"thumbnail": "flat2.jpg"}) == "flat2.jpg"
    # Malformed entries don't crash — just fall back.
    assert _pick_artwork({"thumbnails": [{}, {"url": ""}], "thumbnail": "t.jpg"}) == "t.jpg"


async def test_pick_artwork_empty_when_nothing_provided():
    from app.api.media_proxy import _pick_artwork

    assert _pick_artwork({}) == ""
    assert _pick_artwork({"thumbnails": [], "thumbnail": ""}) == ""


async def test_audio_url_requires_bearer_token(unauth_client):
    """Without a valid Bearer token the endpoint rejects at the auth layer,
    never invoking the extractor. No yt-dlp-as-a-service on toup.ai."""
    # Also assert the extractor was NOT called.
    with patch("app.api.media_proxy._extract_audio") as ext:
        r = await unauth_client.get("/api/media/abc/audio_url")

    assert r.status_code == 401
    ext.assert_not_called()
