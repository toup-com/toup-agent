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


# ── Extraction coalescing ────────────────────────────────────────────────
# `_EXTRACT_CACHE` only helps callers that arrive after an extraction has
# FINISHED. The mobile client's two calls for a cold track — `/audio_url`
# (pre-warm) and `/audio_stream` (the play) — arrive in the same tick, so both
# missed the cache and both ran yt-dlp: two subprocesses and two trips through
# the single residential proxy whose uplink is what the user is waiting on.
async def test_concurrent_extractions_run_yt_dlp_once():
    from app.api import media_proxy

    calls: list[str] = []

    def slow(vid: str) -> dict:
        calls.append(vid)
        time.sleep(0.15)  # blocking, like the real extractor
        return _wrap_with_helpers(_fake_info_ok())

    media_proxy._EXTRACT_CACHE.clear()
    media_proxy._extract_inflight.clear()
    with patch("app.api.media_proxy._extract_audio", side_effect=slow):
        results = await asyncio.gather(*[
            media_proxy._extract_coalesced("abc123def45") for _ in range(4)
        ])

    assert len(calls) == 1, f"expected one extraction, ran {len(calls)}"
    assert all(r["url"] == _FAKE_URL for r in results)


async def test_a_caller_giving_up_does_not_cancel_the_others():
    """A `wait_for` timeout must cancel only that caller's wait.

    The two callers have different budgets (`/audio_url` and `/audio_stream`
    both use `_AUDIO_EXTRACT_TIMEOUT_SECS`, but a client disconnect cancels a
    request at any point). Without the shield, the first one to give up would
    cancel the shared task and take every joined caller down with it —
    converting one slow extraction into a failed play.
    """
    from app.api import media_proxy

    calls: list[str] = []

    def slow(vid: str) -> dict:
        calls.append(vid)
        time.sleep(0.3)
        return _wrap_with_helpers(_fake_info_ok())

    media_proxy._EXTRACT_CACHE.clear()
    media_proxy._extract_inflight.clear()
    with patch("app.api.media_proxy._extract_audio", side_effect=slow):
        patient = asyncio.create_task(media_proxy._extract_coalesced("abc123def45"))
        await asyncio.sleep(0.05)  # let the first caller own the slot
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                media_proxy._extract_coalesced("abc123def45"), timeout=0.1,
            )
        result = await patient

    assert len(calls) == 1
    assert result["url"] == _FAKE_URL


async def test_the_inflight_slot_is_released_after_completion():
    """Otherwise a stale finished task would be re-awaited forever, pinning the
    first extraction's signed URL past its expiry."""
    from app.api import media_proxy

    media_proxy._EXTRACT_CACHE.clear()
    media_proxy._extract_inflight.clear()
    with patch("app.api.media_proxy._extract_audio",
               side_effect=lambda vid: _wrap_with_helpers(_fake_info_ok())):
        await media_proxy._extract_coalesced("abc123def45")

    await asyncio.sleep(0)  # let the done-callback run
    assert "abc123def45" not in media_proxy._extract_inflight


# ── /audio_url must not build the remux (proxy-bandwidth priority) ─────────
# The phone calls /audio_url on the `media_play` frame and starts
# /audio_stream a moment later for the SAME video. When /audio_url kicked off
# a remux build, that build pulled the whole itag-18 through the residential
# proxy while the live stream pulled the same bytes through the same proxy,
# and the two halved each other's share of the hard bottleneck. Measured
# 2026-08-03: 0.45-0.76 MB/s against a 1.2-2.7 MB/s healthy baseline, and iOS
# needs ~2.5MB buffered before it makes a sound — the duplicate turned a ~5s
# cold start into 10s+. Warming is the post-stream hook's job.


async def test_audio_url_does_not_start_a_remux_build(client):
    """/audio_url resolves the URL only. A build here races the play it precedes."""
    built: list[str] = []

    async def spy(vid: str) -> None:
        built.append(vid)

    with patch("app.api.media_proxy._extract_audio",
               side_effect=lambda v: _wrap_with_helpers(_fake_info_ok())), \
         patch("app.api.media_proxy._ensure_remux_bg", side_effect=spy):
        r = await client.get("/api/media/abc123def45/audio_url")
        # Let any task the handler scheduled actually run before asserting —
        # without this the test passes even if create_task() was called.
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    assert r.status_code == 200
    assert r.json()["url"] == _FAKE_URL
    assert built == [], f"/audio_url scheduled a remux build for {built}"


async def test_stream_still_warms_the_cache_after_the_response(client):
    """The warm did not disappear — it moved to where it no longer competes.

    Guards the other half of the fix: if someone deletes the post-stream hook
    too, nothing would ever populate the shared remux cache for played tracks
    and every play would be a cold proxy pull forever.
    """
    from pathlib import Path

    from app.api import media_proxy

    src = Path(media_proxy.__file__).read_text()
    stream_fn = src.split("async def stream_audio", 1)[1]
    assert "_ensure_remux_bg" in stream_fn, (
        "stream_audio no longer warms the remux cache; with /audio_url's build "
        "removed, nothing would cache a played track"
    )


# ── A dead egress proxy must be diagnosable ───────────────────────────────


def test_proxy_outage_is_distinguished_from_an_unavailable_video():
    from app.api.media_proxy import _is_proxy_outage, _should_try_next_client

    # The exact yt-dlp error production emitted on 2026-08-03, when the
    # IPRoyal account ran out of prepaid traffic.
    real = (
        "ERROR: [youtube] fv_5VOUkHYA: Unable to download API page: "
        "('Unable to connect to proxy', OSError('Tunnel connection failed: "
        "402 Payment Required'))"
    )
    assert _is_proxy_outage(Exception(real))
    # …and it must not be retried across clients: they share the one proxy.
    assert not _should_try_next_client(Exception(real))

    for other in (
        "407 Proxy Authentication Required",
        "ProxyError('Cannot connect to proxy')",
    ):
        assert _is_proxy_outage(Exception(other)), other

    # Genuinely per-video / per-client failures stay out of this bucket, or a
    # single private song would page whoever is on call.
    for benign in (
        "Video unavailable. This video is private",
        "Sign in to confirm you're not a bot",
        "Requested format is not available",
        "This video is age-restricted",
    ):
        assert not _is_proxy_outage(Exception(benign)), benign


async def test_proxy_outage_surfaces_its_own_error_code(client):
    """A platform-wide outage must not be reported as 'extraction_failed'.

    yt-dlp is not installed in the test env, so the extractor itself is
    stubbed; the classifier that produces this dict is covered by
    test_proxy_outage_is_distinguished_from_an_unavailable_video above. What
    this pins is that the distinct code survives the handler and reaches the
    client instead of being flattened into the generic failure.
    """
    outage = {"error": "proxy_unavailable", "detail": "Tunnel connection failed: 402"}

    with patch("app.api.media_proxy._extract_audio", return_value=outage):
        r = await client.get("/api/media/fv_5VOUkHYA/audio_url")

    assert r.status_code == 502
    assert r.json().get("error") == "proxy_unavailable", r.json()


def test_the_outage_branch_returns_the_distinct_code():
    """The classifier and the code are wired to each other, not just present."""
    from pathlib import Path

    from app.api import media_proxy

    src = Path(media_proxy.__file__).read_text()
    branch = src.split("if _is_proxy_outage(e):", 1)[1].split("if _should_try_next_client", 1)[0]
    assert "proxy_unavailable" in branch, "outage branch no longer returns its own code"
    assert "logger.error" in branch, "a platform-wide outage must log at ERROR, not WARNING"


def test_extract_warm_does_not_download_media_bytes():
    """`_ensure_extract_bg` must be a metadata handshake, never a build.

    This is the entire safety argument for aiming it at the track that is
    playing RIGHT NOW: a build pulls ~11.8MB of itag-18 through the one
    residential proxy the live stream needs, an extraction pulls none. If this
    ever routes through the remux path it silently becomes the bandwidth
    self-competition bug it was written to avoid.
    """
    import ast
    import inspect
    from app.api import media_proxy

    # Assert on the CODE, not the prose: the docstring names the build path in
    # order to explain why it must not be used, and a plain substring scan of
    # the source counts that as a violation.
    tree = ast.parse(inspect.getsource(media_proxy._ensure_extract_bg).strip())
    fn = tree.body[0]
    if ast.get_docstring(fn):
        fn.body = fn.body[1:]
    called = {
        n.func.id for n in ast.walk(ast.Module(body=fn.body, type_ignores=[]))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "_extract_coalesced" in called, "extract warm must use the extract path"
    assert not called & {"_remux_now", "_ensure_remux_bg", "_bounded_build"}, (
        "extract warm must NOT trigger a build — that is the competing "
        f"download this function exists to avoid (calls: {sorted(called)})"
    )


@pytest.mark.asyncio
async def test_extract_warm_swallows_failure_and_never_raises():
    """A failed pre-warm must be invisible; the phone's own request surfaces
    the real error on the normal path."""
    from app.api import media_proxy

    async def _boom(_vid):
        raise RuntimeError("extraction exploded")

    orig = media_proxy._extract_coalesced
    media_proxy._extract_coalesced = _boom
    try:
        await media_proxy._ensure_extract_bg("abc123def45")  # must not raise
    finally:
        media_proxy._extract_coalesced = orig
