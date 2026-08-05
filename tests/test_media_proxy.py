"""Tests for the media-proxy audio URL extractor.

Covers `GET /api/media/{video_id}/audio_url` used by the mobile hybrid
player (foreground WebView video → background TrackPlayer audio handoff).

yt-dlp is mocked — we never hit YouTube in CI. Every test patches
`app.api.media_proxy._extract_audio` so the contract-level behaviour is
what's under test, not the extractor plumbing.
"""
from __future__ import annotations

import asyncio
import logging
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


def test_no_ad_hoc_ddl_on_the_audio_hot_path():
    """media_proxy must never CREATE a table at request time.

    `audio_remux_cache` was created ad-hoc with CREATE TABLE IF NOT EXISTS on
    the audio path, which is why it was invisible to every guard in the repo:
    no SQLAlchemy model, no alembic migration, absent from the AGENT_ONLY /
    PLATFORM_ONLY / SHARED table lists, so no migration review ever sized it.
    It grew to 1187 MB of a 1233 MB database — 96% — and restricted the
    project. Self-creating DDL also means a DROP is not a fix: the next
    request recreates the table and it regrows.
    """
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[1] / "app" / "api" / "media_proxy.py"
    text = src.read_text().upper()
    assert "CREATE TABLE" not in text, (
        "media_proxy must not create tables at request time — add a model and "
        "a migration instead, so the table is visible to the schema guards"
    )


def test_postgres_blob_cache_is_retired():
    """The shared audio cache is R2 only.

    Postgres held ~3MB audio blobs as BYTEA to give replicas a shared cache
    before R2 existed. R2 does that job without putting media in a relational
    database. Verified before removal (2026-08-04): all 300 PG rows were also
    in R2, so retiring the tier lost no cache entries.
    """
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[1] / "app" / "api" / "media_proxy.py"
    text = src.read_text()
    assert "audio_remux_cache" not in text, "the Postgres blob cache must stay retired"
    assert "_r2_pull_to_local" in text, "R2 is the shared read tier"
    assert "_r2_store_from_local" in text, "R2 is the shared write tier"


# ── Shared (cross-replica) extraction cache ──────────────────────────────
# `_EXTRACT_CACHE` is per-PROCESS and platform-api runs at numReplicas=2, so
# the broadcast-time pre-extract warmed one replica while the phone's play
# request landed on either — half of all first plays threw the warm away and
# paid the full ~5-7s extraction with the user waiting.


def _client_error(code: str) -> Exception:
    """An exception shaped like botocore's ClientError (it carries `.response`)."""
    err = Exception(f"simulated {code}")
    err.response = {"Error": {"Code": code}}
    return err


class _FakeBody:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data


class _FakeR2:
    """Records every call so tests assert on behaviour, not on source text."""

    def __init__(self, store: dict | None = None, raises: Exception | None = None) -> None:
        self.store = store or {}
        self.raises = raises
        self.gets: list[str] = []
        self.puts: list[tuple[str, bytes]] = []
        self.deletes: list[str] = []

    def get_object(self, Bucket, Key):  # noqa: N803 — boto3's kwarg names
        self.gets.append(Key)
        if self.raises is not None:
            raise self.raises
        if Key not in self.store:
            raise _client_error("NoSuchKey")
        return {"Body": _FakeBody(self.store[Key])}

    def put_object(self, Bucket, Key, Body, **kw):  # noqa: N803
        self.puts.append((Key, Body))
        self.store[Key] = Body

    def delete_object(self, Bucket, Key):  # noqa: N803
        self.deletes.append(Key)
        self.store.pop(Key, None)


@pytest.fixture
def shared_r2(monkeypatch):
    """A fake shared tier wired into media_proxy, with both caches reset."""
    from app.api import media_proxy

    fake = _FakeR2()
    media_proxy._EXTRACT_CACHE.clear()
    media_proxy._extract_inflight.clear()
    monkeypatch.setattr(media_proxy, "_get_r2_meta_client", lambda: fake)
    monkeypatch.setattr(media_proxy.settings, "r2_bucket", "test-bucket", raising=False)
    yield fake
    media_proxy._EXTRACT_CACHE.clear()


def _shared_payload(video_id: str, stored_at: float, *, expires_at: float | None = None) -> bytes:
    import json as _json
    return _json.dumps({
        "v": 1,
        "stored_at": stored_at,
        "result": {
            "url": f"https://rr2.googlevideo.com/{video_id}",
            "expires_at": expires_at if expires_at is not None else stored_at + 21600,
            "mime_type": "audio/mp4",
        },
    }).encode()


def test_a_failed_extraction_is_never_cached_in_either_tier():
    """The poison rule: only a result with a real URL may be published.

    A cached error would be served to every replica for an hour, turning one
    transient yt-dlp failure into a fleet-wide outage for that track.
    """
    from app.api import media_proxy

    now = time.time()
    assert media_proxy._extract_cache_deadline({"error": "no_stream"}, now) == 0.0
    assert media_proxy._extract_cache_deadline({"url": ""}, now) == 0.0
    assert media_proxy._extract_cache_deadline({}, now) == 0.0
    assert media_proxy._extract_cache_deadline({"url": "https://x"}, now) > now


def test_the_staleness_cap_is_anchored_to_when_the_extraction_was_taken():
    """The shared tier must age from the extraction, not from the read.

    Recomputing the 1h cap against `now` on every hit renews it on each read,
    so a single extraction would live forever and keep serving a URL whose
    signing IP rotated 30 minutes ago.
    """
    from app.api import media_proxy

    taken = time.time() - 3500          # 58 min old — inside the 1h cap
    result = {"url": "https://x", "expires_at": time.time() + 21600}
    assert media_proxy._extract_cache_deadline(result, taken) > time.time()

    taken = time.time() - 3700          # 61 min old — past it
    assert media_proxy._extract_cache_deadline(result, taken) <= time.time()


def test_the_deadline_stops_short_of_the_signed_urls_own_expiry():
    from app.api import media_proxy

    now = time.time()
    result = {"url": "https://x", "expires_at": now + 600}
    # 5-minute safety margin: serving a URL that expires mid-download is a
    # dead player, not a slow one.
    assert media_proxy._extract_cache_deadline(result, now) == pytest.approx(now + 300, abs=1)


def test_another_replicas_extraction_skips_yt_dlp_entirely(shared_r2):
    """The headline behaviour: replica B reuses replica A's work."""
    from app.api import media_proxy

    vid = "sharedvid01"
    shared_r2.store[f"extract/{vid}.json"] = _shared_payload(vid, time.time() - 30)

    with patch.object(media_proxy, "_extract_audio") as ext:
        result = media_proxy._cached_extract(vid)

    ext.assert_not_called()
    assert result["url"].endswith(vid)
    assert shared_r2.gets == [f"extract/{vid}.json"]


def test_a_shared_hit_is_promoted_so_the_next_read_is_local(shared_r2):
    from app.api import media_proxy

    vid = "sharedvid02"
    shared_r2.store[f"extract/{vid}.json"] = _shared_payload(vid, time.time() - 30)

    with patch.object(media_proxy, "_extract_audio") as ext:
        media_proxy._cached_extract(vid)
        media_proxy._cached_extract(vid)

    ext.assert_not_called()
    # One R2 round trip, not two — the promotion happened.
    assert len(shared_r2.gets) == 1


def test_a_cold_extraction_is_published_for_the_other_replica(shared_r2):
    from app.api import media_proxy
    import json as _json

    vid = "coldvid0001"
    fresh = {"url": f"https://rr2.googlevideo.com/{vid}", "expires_at": time.time() + 21600}

    with patch.object(media_proxy, "_extract_audio", return_value=fresh):
        media_proxy._cached_extract(vid)

    assert [k for k, _ in shared_r2.puts] == [f"extract/{vid}.json"]
    payload = _json.loads(shared_r2.puts[0][1])
    assert payload["result"]["url"] == fresh["url"]
    assert payload["stored_at"] > 0, "without stored_at the reader cannot age the entry"


def test_a_failed_extraction_is_not_published(shared_r2):
    from app.api import media_proxy

    with patch.object(media_proxy, "_extract_audio", return_value={"error": "no_stream"}):
        media_proxy._cached_extract("badvid00001")

    assert shared_r2.puts == [], "an error must never reach the shared tier"


def test_a_stale_shared_entry_is_ignored_and_re_extracted(shared_r2):
    from app.api import media_proxy

    vid = "stalevid001"
    shared_r2.store[f"extract/{vid}.json"] = _shared_payload(vid, time.time() - 7200)
    fresh = {"url": "https://fresh", "expires_at": time.time() + 21600}

    with patch.object(media_proxy, "_extract_audio", return_value=fresh) as ext:
        result = media_proxy._cached_extract(vid)

    ext.assert_called_once()
    assert result["url"] == "https://fresh"


def test_a_miss_is_silent_but_a_credentials_failure_is_logged(monkeypatch, caplog):
    """A cache that never hits must not look the same as one that is cold.

    NoSuchKey is the normal case for any track nobody played this hour and
    would drown the logs; AccessDenied means the tier is dead and nobody would
    ever find out. Both arrive as botocore `ClientError`, so the code has to
    read the error CODE, not the exception class.
    """
    from app.api import media_proxy

    monkeypatch.setattr(media_proxy.settings, "r2_bucket", "test-bucket", raising=False)

    miss = _FakeR2(raises=_client_error("NoSuchKey"))
    monkeypatch.setattr(media_proxy, "_get_r2_meta_client", lambda: miss)
    with caplog.at_level(logging.WARNING, logger="app.api.media_proxy"):
        assert media_proxy._shared_extract_load("v1", time.time()) is None
    assert caplog.records == []

    denied = _FakeR2(raises=_client_error("AccessDenied"))
    monkeypatch.setattr(media_proxy, "_get_r2_meta_client", lambda: denied)
    with caplog.at_level(logging.WARNING, logger="app.api.media_proxy"):
        assert media_proxy._shared_extract_load("v1", time.time()) is None
    assert any("shared extract read failed" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_purging_a_poisoned_url_clears_both_tiers(shared_r2):
    """Clearing only RAM leaves the bad URL in R2 for the fleet to read back."""
    from app.api import media_proxy

    vid = "poisonvid01"
    media_proxy._EXTRACT_CACHE[vid] = ({"url": "https://stale"}, time.time() + 600)
    shared_r2.store[f"extract/{vid}.json"] = _shared_payload(vid, time.time())

    await media_proxy._purge_extraction(vid)

    assert vid not in media_proxy._EXTRACT_CACHE
    assert shared_r2.deletes == [f"extract/{vid}.json"]


def test_the_shared_tier_client_bounds_its_timeouts(monkeypatch):
    """botocore defaults to 60s connect AND 60s read.

    `_r2_pull_to_local` bounds that with `asyncio.wait_for`, which is not
    available here — `_cached_extract` is synchronous, inside an executor — so
    an unreachable R2 would add up to a minute to a play the user is waiting on.
    """
    from app.api import media_proxy

    monkeypatch.setattr(media_proxy, "_r2_ready", lambda: True)
    for attr, val in (
        ("r2_account_id", "acct"), ("r2_bucket", "b"),
        ("r2_access_key_id", "k"), ("r2_secret_access_key", "s"),
    ):
        monkeypatch.setattr(media_proxy.settings, attr, val, raising=False)
    monkeypatch.setattr(media_proxy, "_r2_meta_client", None)
    monkeypatch.setattr(media_proxy, "_r2_meta_disabled", False)

    client = media_proxy._get_r2_meta_client()
    assert client is not None
    cfg = client.meta.config
    assert 0 < cfg.connect_timeout <= 10, f"connect_timeout={cfg.connect_timeout}"
    assert 0 < cfg.read_timeout <= 10, f"read_timeout={cfg.read_timeout}"


def test_the_stream_403_path_purges_before_it_retries():
    """Order matters: re-extracting before the purge just re-reads the poison.

    `_extract_coalesced` reads `_EXTRACT_CACHE` first, so a retry issued while
    the bad entry is still cached returns the same dead URL and the retry is
    theatre. Asserted on the AST because the two calls sit in one branch.
    """
    import ast
    import inspect
    from app.api import media_proxy

    tree = ast.parse(inspect.getsource(media_proxy.stream_audio).strip())
    order: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name in ("_purge_extraction", "_extract_coalesced"):
                order.append((node.lineno, name))
    names = [n for _, n in sorted(order)]
    assert "_purge_extraction" in names, "the 403 branch must purge the poisoned entry"
    purge_at = names.index("_purge_extraction")
    assert "_extract_coalesced" in names[purge_at + 1:], (
        "the re-extraction must come AFTER the purge, or it re-reads the cache"
    )


# ── Live-first build gate ─────────────────────────────────────────────────
#
# The founder reported a music cold start of ~10s on 2026-08-05 where it had
# been ~2s. `warm_audio_cache`'s docstring already names this exact failure —
# a `build` "saturates the single residential proxy, so it is only ever correct
# for UPCOMING tracks. Aimed at the track playing right now it competes with
# that track's own stream, which is the regression this argument exists to keep
# from being re-introduced." `ws_chat.py` then called it with no `mode=`, whose
# default is `build`, on the `_auto` toggle — seconds into the play whose first
# 2.5MB was still arriving.
#
# Measured against the production proxy that day: 0.63 MB/s, consistent over
# five samples. iOS needs ~2.5MB before it makes a sound: ~4.0s alone, ~12s
# split three ways with two builds.
#
# The concurrency cap was never the fix. It bounds how MANY builds run, not
# whether they run while somebody is sitting in silence.


@pytest.fixture(autouse=True)
def _reset_live_gate():
    """The gate is module state; a leaked count would silently disable it for
    every later test in the file."""
    from app.api import media_proxy as mp
    mp._live_starting = 0
    mp._live_idle = None
    yield
    mp._live_starting = 0
    mp._live_idle = None


async def test_a_build_waits_while_a_cold_start_is_still_silent():
    from app.api import media_proxy as mp

    mp._live_start_begin()
    waiter = asyncio.create_task(mp._await_live_idle("vid"))
    await asyncio.sleep(0)
    assert not waiter.done(), "the build did not yield to the live cold start"

    mp._live_start_done()
    await asyncio.wait_for(waiter, timeout=1.0)


async def test_a_build_does_not_wait_when_nothing_is_starting():
    from app.api import media_proxy as mp

    await asyncio.wait_for(mp._await_live_idle("vid"), timeout=0.5)


async def test_two_concurrent_cold_starts_both_have_to_finish():
    """A refcount, not a flag. With a boolean the second play's `done` would
    open the gate while the first was still silent."""
    from app.api import media_proxy as mp

    mp._live_start_begin()
    mp._live_start_begin()
    waiter = asyncio.create_task(mp._await_live_idle("vid"))

    mp._live_start_done()
    await asyncio.sleep(0)
    assert not waiter.done(), "the gate opened while one cold start was still silent"

    mp._live_start_done()
    await asyncio.wait_for(waiter, timeout=1.0)


async def test_the_gate_gives_up_rather_than_starving_the_cache(monkeypatch):
    """A client that stalls without disconnecting must not block builds forever.
    Building anyway is merely today's behaviour."""
    from app.api import media_proxy as mp

    monkeypatch.setattr(mp, "_BUILD_YIELD_TIMEOUT", 0.05)
    mp._live_start_begin()  # never released
    await asyncio.wait_for(mp._await_live_idle("vid"), timeout=1.0)


async def test_an_unbalanced_release_cannot_drive_the_count_negative():
    """A negative count would never return to zero, so the event would never be
    set again and the gate would block every build until restart."""
    from app.api import media_proxy as mp

    mp._live_start_done()
    mp._live_start_done()
    assert mp._live_starting == 0

    mp._live_start_begin()
    waiter = asyncio.create_task(mp._await_live_idle("vid"))
    await asyncio.sleep(0)
    assert not waiter.done(), "the gate stopped working after an unbalanced release"
    mp._live_start_done()
    await asyncio.wait_for(waiter, timeout=1.0)


def test_the_gate_is_taken_before_AND_inside_the_concurrency_cap():
    """Checking only before the semaphore lets a build acquire a slot during a
    quiet moment and then pull 11.8MB straight through somebody's first play.
    Order matters, so assert the order."""
    import ast
    import inspect
    from app.api import media_proxy as mp

    tree = ast.parse(inspect.getsource(mp._bounded_build))
    fn = tree.body[0]

    # calls, in source order, tagged with whether they are inside the `async with`
    seq = []

    def walk(nodes, inside):
        for n in nodes:
            if isinstance(n, ast.AsyncWith):
                walk(n.body, True)
                continue
            for sub in ast.walk(n):
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
                    seq.append((sub.func.id, inside))

    walk(fn.body, False)
    waits = [inside for name, inside in seq if name == "_await_live_idle"]
    assert waits.count(False) >= 1, "no gate check before the concurrency cap"
    assert waits.count(True) >= 1, "no gate re-check after acquiring a build slot"


def test_a_prefetch_stream_does_not_hold_the_gate():
    """`?prefetch=1` is the phone filling its own disk for a future skip. Nobody
    is waiting on it, and holding the gate would deadlock the two halves of
    warming against each other."""
    import ast
    import inspect
    from app.api import media_proxy as mp

    src = inspect.getsource(mp.stream_audio)
    assert "gating = not prefetch" in src, (
        "the build gate must be held by LIVE plays only, never by a prefetch"
    )


def test_the_gate_releases_on_sound_not_on_completion():
    """A progressive itag-18 response keeps pumping for ~30s after playback
    starts. Holding until completion would block builds for the whole track and
    starve the prefetch that makes the next skip instant."""
    import inspect
    from app.api import media_proxy as mp

    src = inspect.getsource(mp.stream_audio)
    assert "_LIVE_GATE_BYTES" in src, "the gate must release on bytes delivered"
    assert mp._LIVE_GATE_BYTES > 0


async def test_the_gate_is_released_when_a_client_disconnects_early():
    """The release lives in `finally`, so an abandoned stream cannot wedge every
    build until the process restarts."""
    import inspect
    from app.api import media_proxy as mp

    src = inspect.getsource(mp.stream_audio)
    body = src[src.index("async def _body"):]
    fin = body.index("finally:")
    assert "_ungate()" in body[fin:fin + 400], (
        "the gate must be released in the body's finally, not only on the happy path"
    )


# ── Egress session pool ───────────────────────────────────────────────────
#
# `YT_DLP_PROXY` names ONE IPRoyal sticky session, so every extraction, every
# byte-pump and every remux build for every user on both replicas left through
# ONE residential peer's home uplink — a single point of bandwidth for the whole
# product, redrawn by lottery every 30 minutes.
#
# Measured 2026-08-05 (2.5MB from Cloudflare through each session, so the number
# is the proxy's uplink rather than googlevideo's):
#   production's pinned session   0.63 MB/s over 5 samples
#   20 fresh draws                median 2.04, p25 0.75, p75 3.90, max 8.41
# 16 of 20 beat the peer production was stuck on. For iOS's ~2.5MB pre-roll that
# is 3.9s against 1.2s.


_POOL_PROXY = "http://user:pw_country-us_session-toupaudio_lifetime-30m@geo.iproyal.com:12321"


@pytest.fixture
def pooled(monkeypatch):
    from app.api import media_proxy as mp
    monkeypatch.setattr(mp.settings, "yt_dlp_proxy", _POOL_PROXY, raising=False)
    monkeypatch.setattr(mp, "_PROXY_SESSION_POOL", 8)
    return mp


def test_different_videos_draw_different_peers(pooled):
    slots = {pooled._proxy_slot_for(v) for v in
             ("kJQP7kiw5Fk", "dQw4w9WgXcQ", "9bZkp7q19f0", "JGwWNGJdvx8",
              "OPf0YbXqDm0", "3JZ_D3ELwOQ", "fJ9rUzIMcZQ", "CevxZvSJLk8")}
    assert len(slots) > 1, "every video mapped to one peer — the pool does nothing"


def test_the_same_video_always_draws_the_same_peer(pooled):
    """googlevideo binds the signed URL to the extracting IP, so extraction, the
    byte-pump and the remux download must agree."""
    a = [pooled._proxy_slot_for("kJQP7kiw5Fk") for _ in range(20)]
    assert len(set(a)) == 1


def test_both_replicas_agree_without_talking_to_each_other(pooled):
    """The slot is a pure function of the video id, so the R2-shared extraction
    stays usable on whichever replica reads it."""
    import hashlib
    vid = "kJQP7kiw5Fk"
    expected = int(hashlib.sha256(vid.encode()).hexdigest()[:8], 16) % 8
    assert pooled._proxy_slot_for(vid) == expected


def test_the_session_token_is_what_varies_and_nothing_else(pooled):
    url = pooled._proxy_with_slot(3)
    assert "session-toupaudio3" in url
    assert url.startswith("http://user:")
    assert url.endswith("@geo.iproyal.com:12321")
    assert "country-us" in url and "lifetime-30m" in url


def test_slot_zero_is_byte_identical_to_today(pooled):
    """AUDIO_PROXY_SESSION_POOL=1 is the rollback, so slot 0 must not touch the
    configured URL at all."""
    assert pooled._proxy_with_slot(0) == _POOL_PROXY


def test_a_pool_of_one_disables_the_whole_thing(monkeypatch, pooled):
    monkeypatch.setattr(pooled, "_PROXY_SESSION_POOL", 1)
    for v in ("kJQP7kiw5Fk", "dQw4w9WgXcQ", "9bZkp7q19f0"):
        assert pooled._proxy_slot_for(v) == 0
        assert pooled._proxy_with_slot(pooled._proxy_slot_for(v)) == _POOL_PROXY


def test_a_password_containing_an_at_sign_survives(monkeypatch, pooled):
    """Splitting on the FIRST '@' would truncate the credentials and send every
    request to a host that does not exist."""
    weird = "http://user:p@ss_session-toupaudio@geo.iproyal.com:12321"
    monkeypatch.setattr(pooled.settings, "yt_dlp_proxy", weird, raising=False)
    out = pooled._proxy_with_slot(4)
    assert out.endswith("@geo.iproyal.com:12321")
    assert "p@ss" in out and "session-toupaudio4" in out


def test_a_proxy_with_no_session_token_is_left_exactly_alone(monkeypatch, pooled):
    """A provider that does not do sticky sessions must still be USED, not
    dropped — returning None here would send music straight out of Railway and
    into YouTube's datacenter-IP bot challenge."""
    plain = "http://user:pw@some.proxy:8080"
    monkeypatch.setattr(pooled.settings, "yt_dlp_proxy", plain, raising=False)
    assert pooled._proxy_with_slot(5) == plain


def test_no_proxy_configured_stays_no_proxy(monkeypatch, pooled):
    monkeypatch.setattr(pooled.settings, "yt_dlp_proxy", "", raising=False)
    assert pooled._proxy_with_slot(0) is None
    assert pooled._proxy_with_slot(3) is None


def test_the_pump_reads_the_slot_the_extraction_RECORDED(pooled):
    """Not recomputed. Recomputing looks equivalent and is not: it would make
    every cached URL depend on the pool size being identical when it is read and
    when it was written, so changing the pool size would remap every warm entry
    onto the wrong egress and 403 the lot at once."""
    assert pooled._proxy_for_result({"proxy_slot": 6}) == pooled._proxy_with_slot(6)
    assert pooled._proxy_for_result({"proxy_slot": 6}) != pooled._proxy_with_slot(2)


def test_an_extraction_cached_before_the_pool_shipped_still_works(pooled):
    """No `proxy_slot` means it was taken through the unmodified URL, which is
    exactly slot 0."""
    assert pooled._proxy_for_result({"url": "https://x"}) == _POOL_PROXY


def test_a_corrupt_slot_falls_back_instead_of_raising(pooled):
    """A bad cache entry must not 500 the play."""
    assert pooled._proxy_for_result({"proxy_slot": "garbage"}) == _POOL_PROXY
    assert pooled._proxy_for_result({"proxy_slot": None}) == _POOL_PROXY


def test_extraction_records_the_slot_it_used():
    """The whole scheme rests on the slot travelling with the result."""
    import ast
    import inspect
    from app.api import media_proxy as mp

    src = inspect.getsource(mp._extract_audio)
    assert '"proxy_slot": slot' in src, "the extraction result must carry its slot"
    assert "opts[\"proxy\"] = proxy" in src, "extraction must use the pooled proxy"


def test_the_403_retry_rebuilds_the_client_for_the_new_egress():
    """The first client is pinned to the OLD entry's slot. Reusing it after a
    re-extraction is a guaranteed second 403 during rollout, when every cached
    entry reads as slot 0 while its retry lands on the video's real slot."""
    import inspect
    from app.api import media_proxy as mp

    src = inspect.getsource(mp.stream_audio)
    branch = src[src.index("if upstream.status_code in (403, 410):"):]
    branch = branch[:branch.index("if upstream.status_code >= 400:")]
    assert "httpx.AsyncClient(" in branch, "the 403 retry must rebuild the client"
    assert "_proxy_for_result(result)" in branch


def test_every_per_video_egress_uses_the_pool_and_none_uses_the_raw_setting():
    """Extraction signs the URL; the pump and the remux download must leave from
    the same address. One of the three reading the raw setting is a 403."""
    import inspect
    from app.api import media_proxy as mp

    for fn in (mp._extract_audio, mp._do_remux, mp.stream_audio):
        src = inspect.getsource(fn)
        assert "settings.yt_dlp_proxy" not in src, (
            f"{fn.__name__} still reaches for the un-pooled proxy directly"
        )


async def test_the_pump_egresses_where_the_EXTRACTION_said_not_where_the_hash_says(
    monkeypatch, client, pooled
):
    """The behavioural half of the recorded-slot contract.

    Recomputing the slot in the pump looks equivalent to reading it and is not:
    it ties every cached URL to the pool size being identical when it is read
    and when it was written. Bumping AUDIO_PROXY_SESSION_POOL would then remap
    every warm entry onto the wrong egress and 403 the lot at once.

    So hand the pump a cached extraction whose recorded slot DISAGREES with the
    hash, and require it to follow the recording.
    """
    vid = "kJQP7kiw5Fk"
    hashed = pooled._proxy_slot_for(vid)
    recorded = (hashed + 3) % 8
    assert recorded != hashed, "precondition: the two must differ"

    async def _fake_extract(v):
        return {
            "url": _FAKE_URL,
            "proxy_slot": recorded,
            "expires_at": 9999999999,
            "duration": 234,
            "mime_type": "audio/mp4",
            "ext": "m4a",
        }

    seen: dict = {}
    real_client_cls = pooled.httpx.AsyncClient

    class _SpyClient(real_client_cls):
        def __init__(self, *a, **kw):
            seen["proxy"] = kw.get("proxy")
            raise RuntimeError("stop here — the proxy choice is all this asserts")

    monkeypatch.setattr(pooled, "_extract_coalesced", _fake_extract)
    monkeypatch.setattr(pooled, "_remuxed_ready", lambda v: None)

    async def _no_r2(v):
        return None
    monkeypatch.setattr(pooled, "_r2_pull_to_local", _no_r2)
    monkeypatch.setattr(pooled.httpx, "AsyncClient", _SpyClient)

    try:
        await client.get(f"/api/media/{vid}/audio_stream")
    except Exception:
        pass  # the spy aborts the request on purpose

    assert seen.get("proxy") == pooled._proxy_with_slot(recorded), (
        "the pump recomputed the slot instead of using the one the extraction recorded"
    )
    assert seen["proxy"] != pooled._proxy_with_slot(hashed)
