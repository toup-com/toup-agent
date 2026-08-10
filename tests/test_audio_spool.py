"""The single-flight upstream spool: one download per video id serves the
2-byte probe, every parallel range connection, AND the remux build.

BACKGROUND (2026-08-09, founder recording + prod TIMING lines from the same
session): a cold MISS cost the residential proxy the same bytes three times —
AVFoundation's bytes=0-1 probe opened a full upstream connection (4.2s of TTFB
for 2 bytes), its real range request opened another (upstream_ms=4239, same
track, same minute), and the post-response remux build re-downloaded the whole
itag-18. ~23.6MB of metered traffic for an ~11.8MB file, through a peer
measured at 0.04-1.14 MB/s, while the user sat in silence. Four of twelve
remuxes in that session were built twice.

These tests are hermetic: no network, no ffmpeg (subprocess is stubbed), no
FastAPI client. They drive the spool primitives directly.
"""
import asyncio
import os

import pytest

import app.api.media_proxy as mp


def _mk_spool(tmp_path, video_id="vid00000001", total=None, data=b"", done=False):
    p = tmp_path / f"{video_id}.spool"
    p.write_bytes(data)
    sp = mp._Spool(video_id, str(p))
    sp.total = total
    sp.received = len(data)
    sp.done = done
    return sp


async def _drain(resp):
    """Collect a StreamingResponse's body bytes."""
    chunks = []
    async for c in resp.body_iterator:
        chunks.append(c if isinstance(c, bytes) else c.encode())
    return b"".join(chunks)


# ── serving from the spool ────────────────────────────────────────────────

def test_probe_served_from_two_bytes(tmp_path):
    """The bytes=0-1 probe must be answerable the moment 2 bytes exist —
    that probe used to open its own upstream connection and pay the full
    cold path for a 2-byte body."""
    sp = _mk_spool(tmp_path, total=1_000_000, data=b"OK")
    resp = mp._spool_response(
        sp, "bytes=0-1", prefetch=True, mime="audio/mp4", video_id=sp.video_id,
        on_release=lambda: None, t0=0.0, cache_ms=0, extract_ms=0,
    )
    assert resp is not None
    assert resp.status_code == 206
    assert resp.headers["content-range"] == "bytes 0-1/1000000"
    assert resp.headers["content-length"] == "2"
    assert asyncio.run(_drain(resp)) == b"OK"


def test_range_streams_while_spool_fills(tmp_path):
    """A reader whose range extends past what has arrived waits and drains
    the rest as the downloader appends — never truncates, never re-fetches."""
    payload = bytes(range(256)) * 64  # 16KB
    sp = _mk_spool(tmp_path, total=len(payload), data=payload[:1000])

    async def go():
        async def writer():
            await asyncio.sleep(0.1)
            with open(sp.path, "ab") as f:
                f.write(payload[1000:])
            sp.received = len(payload)
            sp.done = True

        w = asyncio.ensure_future(writer())
        resp = mp._spool_response(
            sp, "bytes=500-", prefetch=True, mime="audio/mp4", video_id=sp.video_id,
            on_release=lambda: None, t0=0.0, cache_ms=0, extract_ms=0,
        )
        assert resp is not None
        body = await _drain(resp)
        await w
        return body

    body = asyncio.run(go())
    assert body == payload[500:]


def test_full_request_headers_and_body(tmp_path):
    payload = b"x" * 5000
    sp = _mk_spool(tmp_path, total=len(payload), data=payload, done=True)
    resp = mp._spool_response(
        sp, None, prefetch=True, mime="audio/mp4", video_id=sp.video_id,
        on_release=lambda: None, t0=0.0, cache_ms=0, extract_ms=0,
    )
    assert resp is not None
    assert resp.status_code == 200
    assert resp.headers["content-length"] == str(len(payload))
    assert asyncio.run(_drain(resp)) == payload


def test_deep_seek_falls_back_to_legacy(tmp_path):
    """A range starting far past what has arrived must NOT wait for the
    sequential fill — it returns None so the caller serves it with its own
    upstream Range request (the legacy path)."""
    sp = _mk_spool(tmp_path, total=50_000_000, data=b"x" * 1000)
    resp = mp._spool_response(
        sp, f"bytes={1000 + mp._SPOOL_READ_AHEAD_LIMIT + 1}-", prefetch=True,
        mime="audio/mp4", video_id=sp.video_id,
        on_release=lambda: None, t0=0.0, cache_ms=0, extract_ms=0,
    )
    assert resp is None


def test_deep_seek_ok_when_done(tmp_path):
    """Once the download is complete every offset is on local disk — a deep
    seek is served from the spool, not bounced to a second upstream pull."""
    payload = b"y" * (mp._SPOOL_READ_AHEAD_LIMIT + 4096)
    sp = _mk_spool(tmp_path, total=len(payload), data=payload, done=True)
    start = len(payload) - 100
    resp = mp._spool_response(
        sp, f"bytes={start}-", prefetch=True, mime="audio/mp4", video_id=sp.video_id,
        on_release=lambda: None, t0=0.0, cache_ms=0, extract_ms=0,
    )
    assert resp is not None
    assert asyncio.run(_drain(resp)) == payload[start:]


def test_unknown_total_returns_none(tmp_path):
    """No Content-Length yet (headers still in flight after the bounded wait)
    → None → legacy path. The response must never guess a length."""
    sp = _mk_spool(tmp_path, total=None, data=b"xx")
    resp = mp._spool_response(
        sp, "bytes=0-1", prefetch=True, mime="audio/mp4", video_id=sp.video_id,
        on_release=lambda: None, t0=0.0, cache_ms=0, extract_ms=0,
    )
    assert resp is None


def test_spool_error_mid_serve_truncates(tmp_path, monkeypatch):
    """A spool that dies mid-download must END the in-flight body (AVPlayer
    re-ranges; the retry lands on the legacy path because the dead spool is
    unregistered) — and it must kick the legacy build warm for the cache."""
    warmed = []
    async def fake_warm(vid):
        warmed.append(vid)
    monkeypatch.setattr(mp, "_ensure_remux_bg", fake_warm)

    sp = _mk_spool(tmp_path, total=10_000, data=b"z" * 1000)

    async def go():
        async def killer():
            await asyncio.sleep(0.1)
            sp.error = "peer closed"

        k = asyncio.ensure_future(killer())
        resp = mp._spool_response(
            sp, "bytes=0-", prefetch=True, mime="audio/mp4", video_id=sp.video_id,
            on_release=lambda: None, t0=0.0, cache_ms=0, extract_ms=0,
        )
        body = await _drain(resp)
        await k
        await asyncio.sleep(0.01)  # let the create_task run
        return body

    body = asyncio.run(go())
    assert body == b"z" * 1000  # what had arrived, nothing invented
    assert warmed == [sp.video_id]


def test_release_called_exactly_once_in_finally(tmp_path):
    """The per-tenant stream semaphore must be released when the body ends —
    including the truncated-by-error path."""
    calls = []
    sp = _mk_spool(tmp_path, total=4, data=b"abcd", done=True)
    resp = mp._spool_response(
        sp, None, prefetch=True, mime="audio/mp4", video_id=sp.video_id,
        on_release=lambda: calls.append(1), t0=0.0, cache_ms=0, extract_ms=0,
    )
    asyncio.run(_drain(resp))
    assert calls == [1]


def test_live_gate_held_then_released(tmp_path, monkeypatch):
    """A non-prefetch spool serve must hold the live-first gate from first
    byte until the client has enough to make sound (or the body ends) —
    identical discipline to the legacy body it replaces."""
    events = []
    monkeypatch.setattr(mp, "_live_start_begin", lambda: events.append("begin"))
    monkeypatch.setattr(mp, "_live_start_done", lambda: events.append("done"))
    payload = b"g" * (mp._LIVE_GATE_BYTES + 4096)
    sp = _mk_spool(tmp_path, total=len(payload), data=payload, done=True)
    resp = mp._spool_response(
        sp, None, prefetch=False, mime="audio/mp4", video_id=sp.video_id,
        on_release=lambda: None, t0=0.0, cache_ms=0, extract_ms=0,
    )
    asyncio.run(_drain(resp))
    assert events == ["begin", "done"]


def test_prefetch_never_touches_live_gate(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(mp, "_live_start_begin", lambda: events.append("begin"))
    monkeypatch.setattr(mp, "_live_start_done", lambda: events.append("done"))
    sp = _mk_spool(tmp_path, total=4, data=b"abcd", done=True)
    resp = mp._spool_response(
        sp, None, prefetch=True, mime="audio/mp4", video_id=sp.video_id,
        on_release=lambda: None, t0=0.0, cache_ms=0, extract_ms=0,
    )
    asyncio.run(_drain(resp))
    assert events == []


# ── the downloader ────────────────────────────────────────────────────────

class _FakeStream:
    def __init__(self, status, headers, chunks):
        self.status_code = status
        self.headers = headers
        self._chunks = chunks

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def aiter_bytes(self, n):
        for c in self._chunks:
            yield c


class _FakeClient:
    def __init__(self, status=200, headers=None, chunks=()):
        self._resp = _FakeStream(status, headers or {}, list(chunks))

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def stream(self, *a, **k):
        return self._resp


def test_downloader_success_kicks_remux(tmp_path, monkeypatch):
    warmed = []

    async def fake_warm(vid):
        warmed.append(vid)

    monkeypatch.setattr(mp, "_ensure_remux_bg", fake_warm)
    monkeypatch.setattr(
        mp.httpx, "AsyncClient",
        lambda **k: _FakeClient(headers={"content-length": "8"}, chunks=[b"abcd", b"efgh"]),
    )
    sp = _mk_spool(tmp_path)

    async def go():
        await mp._spool_download(sp, {"url": "http://x", "proxy_slot": 0})
        await asyncio.sleep(0.01)

    asyncio.run(go())
    assert sp.done and sp.error is None
    assert sp.total == 8 and sp.received == 8
    assert open(sp.path, "rb").read() == b"abcdefgh"
    assert warmed == [sp.video_id]


def test_downloader_short_body_is_an_error(tmp_path, monkeypatch):
    """content-length says 100, the peer sent 8 (the 'peer closed connection'
    class from prod logs). The spool must FAIL — a silently short file would
    remux into a truncated song and be cached forever."""
    monkeypatch.setattr(
        mp.httpx, "AsyncClient",
        lambda **k: _FakeClient(headers={"content-length": "100"}, chunks=[b"abcdefgh"]),
    )
    sp = _mk_spool(tmp_path)
    mp._spools[sp.video_id] = sp
    try:
        asyncio.run(mp._spool_download(sp, {"url": "http://x", "proxy_slot": 0}))
        assert sp.error is not None
        assert not sp.done
        assert sp.video_id not in mp._spools
        assert not os.path.exists(sp.path)
    finally:
        mp._spools.pop(sp.video_id, None)


def test_downloader_http_error_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(mp.httpx, "AsyncClient", lambda **k: _FakeClient(status=403))
    sp = _mk_spool(tmp_path)
    asyncio.run(mp._spool_download(sp, {"url": "http://x", "proxy_slot": 0}))
    assert sp.error is not None and not sp.done


# ── the build side ────────────────────────────────────────────────────────

def test_spool_file_for_build_joins_done_spool(tmp_path):
    sp = _mk_spool(tmp_path, total=4, data=b"abcd", done=True)
    mp._spools[sp.video_id] = sp
    try:
        got = asyncio.run(mp._spool_file_for_build(sp.video_id))
        assert got == sp.path
    finally:
        mp._spools.pop(sp.video_id, None)


def test_spool_file_for_build_none_on_error(tmp_path, monkeypatch):
    """A failed spool must NOT feed the build a truncated file — None sends
    the build to its own (legacy, gated) download."""
    sp = _mk_spool(tmp_path, total=100, data=b"abcd")
    sp.error = "died"
    mp._spools[sp.video_id] = sp
    # If the join path is broken and the build tries to START a fresh spool,
    # extraction would run — make that loud.
    async def boom(vid):
        raise AssertionError("must not re-extract for a failed spool join")
    try:
        got = asyncio.run(mp._spool_file_for_build(sp.video_id))
        assert got is None
    finally:
        mp._spools.pop(sp.video_id, None)


def test_do_remux_with_src_path_never_downloads(tmp_path, monkeypatch):
    """With a finished spool as src, the build must not open ANY network
    client — the whole point is zero additional proxy bytes."""
    src = tmp_path / "src.mp4"
    src.write_bytes(b"fake-itag18-bytes")

    def no_client(*a, **k):
        raise AssertionError("network client constructed during spool-fed remux")

    monkeypatch.setattr(mp.httpx, "Client", no_client)
    monkeypatch.setattr(mp, "_remuxed_ready", lambda vid: None)
    monkeypatch.setattr(mp, "_ffmpeg_available", lambda: True)
    monkeypatch.setattr(mp, "_ensure_cache_dir", lambda: True)
    out_path = tmp_path / "out.m4a"
    monkeypatch.setattr(mp, "_remuxed_path", lambda vid: str(out_path))
    monkeypatch.setattr(mp, "_prune_audio_cache", lambda: None)

    class _Proc:
        returncode = 0
        stderr = b""

    def fake_run(cmd, timeout, capture_output):
        # ffmpeg reads src, writes tmp — simulate.
        assert str(src) in cmd
        tmp = cmd[-1]
        with open(tmp, "wb") as f:
            f.write(b"remuxed")
        return _Proc()

    monkeypatch.setattr(mp.subprocess, "run", fake_run)
    got = mp._do_remux("vidsrc00001", str(src))
    assert got == str(out_path)
    assert out_path.read_bytes() == b"remuxed"
    # The spool file is owned by the spool — the remux must not delete it.
    assert src.exists()


# NOTE: an earlier version of this file asserted the spool was DISCARDED when
# the remux published. That was the wrong contract — a playing item that
# started on the spool has the itag-18 total baked into its AVPlayer state,
# and retiring the spool mid-track flips its byte space to the ~4x-smaller
# m4a (adversarial review, 2026-08-09). The spool now SURVIVES publish; see
# test_review_fixes_media.py::test_bounded_build_keeps_spool_after_publish
# for the current invariant, and stream_audio's tier-pinning block for the
# fresh-start adoption that eventually retires it.


def test_get_or_start_reuses_live_spool(tmp_path, monkeypatch):
    """Two concurrent consumers must share ONE spool — the single-flight
    property the whole design hangs on."""
    started = []

    async def fake_download(sp, result):
        started.append(sp.video_id)
        sp.total = 4
        sp.received = 4
        sp.done = True

    monkeypatch.setattr(mp, "_spool_download", fake_download)
    monkeypatch.setattr(mp, "_AUDIO_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(mp, "_ensure_cache_dir", lambda: True)

    async def go():
        a = await mp._spool_get_or_start("vidshare0001", {"url": "u", "proxy_slot": 0})
        b = await mp._spool_get_or_start("vidshare0001", {"url": "u", "proxy_slot": 0})
        return a, b

    try:
        a, b = asyncio.run(go())
        assert a is b
        assert started == ["vidshare0001"]
    finally:
        mp._spools.pop("vidshare0001", None)
