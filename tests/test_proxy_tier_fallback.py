"""Egress tiers: static ISP primary + rotating residential fallback.

BACKGROUND (2026-08-10): IPRoyal static (ISP) proxies are IP-BOUND — one
source IP may use them at a time, held for a ~10s sticky window from
connection start (measured: a second source IP gets `Tunnel connection
failed: 403` from the proxy itself while the first is mid-burst, and existing
tunnels survive a takeover). platform-api runs 2 replicas with distinct
egress IPs, so wiring a static primary alone made ~half of concurrent cold
plays 502 ("spool FAILED: 403 Forbidden" → "upstream-connect failed").
The same morning the provider rotated the primary's credentials and every
uncached play failed platform-wide with 407s — both incidents are the classes
`_extract_audio`'s tier orchestration turns into one WARNING line.

These tests are hermetic: no network, no yt-dlp (module is stubbed).
"""
import sys
import types

import pytest

import app.api.media_proxy as mp
from app.config import settings


PRIMARY = "http://user:pass@203.0.113.10:12323"
FALLBACK = "http://ruser:rpass_session-toupaudio_lifetime-30m@geo.example.com:12321"

BUSY_ERR = RuntimeError(
    "Unable to download API page: ('Unable to connect to proxy', "
    "OSError('Tunnel connection failed: 403 Forbidden'))"
)
AUTH_ERR = RuntimeError(
    "Unable to download API page: ('Unable to connect to proxy', "
    "OSError('Tunnel connection failed: 407 Proxy Authentication Required'))"
)


@pytest.fixture(autouse=True)
def _reset_breaker():
    mp._primary_down_until = 0.0
    yield
    mp._primary_down_until = 0.0


@pytest.fixture
def tiers(monkeypatch):
    monkeypatch.setattr(settings, "yt_dlp_proxy", PRIMARY)
    monkeypatch.setattr(settings, "yt_dlp_proxy_fallback", FALLBACK)


@pytest.fixture
def primary_only(monkeypatch):
    monkeypatch.setattr(settings, "yt_dlp_proxy", PRIMARY)
    monkeypatch.setattr(settings, "yt_dlp_proxy_fallback", None)


# ── classification ────────────────────────────────────────────────────────

def test_busy_is_the_403_connect_and_nothing_else():
    assert mp._is_proxy_busy(BUSY_ERR)
    assert not mp._is_proxy_busy(AUTH_ERR)
    assert not mp._is_proxy_busy(RuntimeError("HTTP Error 403: Forbidden"))


def test_busy_matches_httpx_proxyerror_phrasing():
    """httpx says `ProxyError("403 Forbidden")` — no "tunnel" substring. The
    2026-08-10 prod log (`spool FAILED …: 403 Forbidden`) is this phrasing;
    matching only urllib's made the stream path's busy recovery dead code."""
    import httpx
    assert mp._is_proxy_busy(httpx.ProxyError("403 Forbidden"))
    assert not mp._is_proxy_busy(httpx.ProxyError("407 Proxy Authentication Required"))
    # A plain string "403 Forbidden" in a non-ProxyError must NOT read busy.
    assert not mp._is_proxy_busy(RuntimeError("403 Forbidden"))


def test_busy_also_matches_outage_markers_so_order_matters():
    """A busy error IS inside the outage marker set — the call sites check
    busy first, and this test pins the overlap so a marker edit that breaks
    the ordering assumption fails loudly here."""
    assert mp._is_proxy_outage(BUSY_ERR)
    assert mp._is_proxy_outage(AUTH_ERR)


# ── tier → URL resolution ─────────────────────────────────────────────────

def test_primary_static_url_ignores_slots(tiers):
    for slot in (0, 1, 5):
        assert mp._proxy_with_slot(slot, "primary") == PRIMARY


def test_fallback_url_rewrites_session_token(tiers):
    assert "session-toupaudio3" in mp._proxy_with_slot(3, "fallback")
    assert mp._proxy_with_slot(0, "fallback") == FALLBACK


def test_result_tier_routes_the_byte_fetch(tiers):
    assert mp._proxy_for_result({"proxy_slot": 2, "proxy_tier": "fallback"}) == \
        mp._proxy_with_slot(2, "fallback")
    assert mp._proxy_for_result({"proxy_slot": 2}) == PRIMARY  # pre-tier cache
    assert mp._proxy_for_result({}) == PRIMARY


def test_unset_fallback_resolves_none_not_primary(tiers, monkeypatch):
    """A fallback-tier result with the fallback env since removed must NOT
    silently ride the primary — googlevideo would 403 the IP mismatch anyway,
    and None (direct) hands recovery to the existing 403/410 purge path."""
    monkeypatch.setattr(settings, "yt_dlp_proxy_fallback", None)
    assert mp._proxy_for_result({"proxy_tier": "fallback"}) is None


# ── extraction orchestration ──────────────────────────────────────────────

def _stub_ydl(monkeypatch, behavior):
    """Install a fake yt_dlp whose extract_info runs `behavior(opts)`."""
    calls = []

    class _YDL:
        def __init__(self, opts):
            self._opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download=False):
            calls.append(self._opts)
            return behavior(self._opts)

    monkeypatch.setitem(sys.modules, "yt_dlp", types.SimpleNamespace(YoutubeDL=_YDL))
    return calls


def test_busy_primary_falls_back_and_stamps_tier(tiers, monkeypatch):
    def behavior(opts):
        if opts.get("proxy") == PRIMARY:
            raise BUSY_ERR
        return {"url": "https://rr1.example/videoplayback?expire=99", "ext": "mp4"}

    calls = _stub_ydl(monkeypatch, behavior)
    result = mp._extract_audio("vid00000001")
    assert result.get("proxy_tier") == "fallback"
    assert result.get("url")
    # The fallback attempt used the fallback egress (slot-rewritten or base).
    assert any("geo.example.com" in (c.get("proxy") or "") for c in calls)


def test_dead_primary_credentials_fall_back(tiers, monkeypatch):
    """The 2026-08-10 morning incident: provider rotated the primary's
    credentials; with a fallback configured this must degrade, not die."""
    def behavior(opts):
        if opts.get("proxy") == PRIMARY:
            raise AUTH_ERR
        return {"url": "https://rr1.example/videoplayback?expire=99", "ext": "mp4"}

    _stub_ydl(monkeypatch, behavior)
    result = mp._extract_audio("vid00000001")
    assert result.get("proxy_tier") == "fallback"
    assert result.get("url")


def test_busy_without_fallback_aborts_fast(primary_only, monkeypatch):
    """No fallback configured → busy aborts the chain (one attempt per tier,
    never four clients queued behind a locked proxy) and surfaces as an
    error result, exactly like the pre-tier outage behaviour."""
    calls = _stub_ydl(monkeypatch, lambda opts: (_ for _ in ()).throw(BUSY_ERR))
    result = mp._extract_audio("vid00000001")
    assert result.get("error") == "proxy_busy"
    assert len(calls) == 1


def test_permanent_video_error_never_reaches_fallback(tiers, monkeypatch):
    """'Video unavailable' is a property of the video: a second tier cannot
    fix it and doubling the chain would double every honest failure."""
    def behavior(opts):
        raise RuntimeError("ERROR: [youtube] x: Video unavailable")

    calls = _stub_ydl(monkeypatch, behavior)
    result = mp._extract_audio("vid00000001")
    assert result.get("error") in ("extraction_failed", "all_clients_blocked")
    assert not any("geo.example.com" in (c.get("proxy") or "") for c in calls)


def test_bot_blocked_primary_ip_falls_back(tiers, monkeypatch):
    """A static IP can get bot-flagged by YouTube at any time (both of the
    first two US ISP draws were). all_clients_blocked on the primary must
    engage the fallback so a burned primary degrades speed, not audio."""
    def behavior(opts):
        if "geo.example.com" in (opts.get("proxy") or ""):
            return {"url": "https://rr1.example/videoplayback?expire=99", "ext": "mp4"}
        raise RuntimeError("Sign in to confirm you're not a bot")

    _stub_ydl(monkeypatch, behavior)
    result = mp._extract_audio("vid00000001")
    assert result.get("proxy_tier") == "fallback"
    assert result.get("url")


# ── review-round guards (adversarial review, 2026-08-10) ─────────────────

def test_equal_tier_urls_disable_the_fallback(monkeypatch):
    """Both env vars pointing at one URL must read as NO fallback — retrying
    through the same egress doubles every honest failure and makes every
    'retrying via fallback tier' log line a lie."""
    monkeypatch.setattr(settings, "yt_dlp_proxy", PRIMARY)
    monkeypatch.setattr(settings, "yt_dlp_proxy_fallback", PRIMARY)
    assert not mp._fallback_available()
    calls = _stub_ydl(monkeypatch, lambda opts: (_ for _ in ()).throw(BUSY_ERR))
    result = mp._extract_audio("vid00000001")
    assert result.get("error") == "proxy_busy"
    assert len(calls) == 1


def test_fallback_tier_connect_403_is_not_busy(tiers, monkeypatch):
    """Only the static primary HAS a device-lock. A CONNECT 403 from the
    rotating fallback gateway is durable and must classify as an OUTAGE, not
    masquerade as a transient collision that never alarms."""
    def behavior(opts):
        raise BUSY_ERR

    _stub_ydl(monkeypatch, behavior)
    result = mp._extract_audio("vid00000001")
    # primary: busy → fallback engaged; fallback: same 403 → outage, not busy.
    assert result.get("error") == "proxy_unavailable"


def test_durable_primary_failure_trips_the_breaker(tiers, monkeypatch):
    """After dead-creds/bot-flag on the primary, the next extraction must go
    straight to the fallback — not pay the full primary chain per play."""
    def behavior(opts):
        if opts.get("proxy") == PRIMARY:
            raise AUTH_ERR
        return {"url": "https://rr1.example/videoplayback?expire=99", "ext": "mp4"}

    calls = _stub_ydl(monkeypatch, behavior)
    mp._extract_audio("vid00000001")
    assert mp._primary_broken()
    n_before = len(calls)
    result = mp._extract_audio("vid00000002")
    assert result.get("proxy_tier") == "fallback"
    assert all("geo.example.com" in (c.get("proxy") or "") for c in calls[n_before:])


def test_busy_never_trips_the_breaker(tiers, monkeypatch):
    """A ~10s device-lock collision must not demote the primary for minutes."""
    def behavior(opts):
        if opts.get("proxy") == PRIMARY:
            raise BUSY_ERR
        return {"url": "https://rr1.example/videoplayback?expire=99", "ext": "mp4"}

    _stub_ydl(monkeypatch, behavior)
    mp._extract_audio("vid00000001")
    assert not mp._primary_broken()


def test_fallback_tier_cache_deadline_is_short(tiers):
    """A fallback-tier entry is a fleet-wide demotion that nothing re-probes;
    its cache life must be bounded well under the 1h cap so the video
    migrates back to the primary."""
    base = 1_000_000.0
    primary_dl = mp._extract_cache_deadline(
        {"url": "u", "proxy_tier": "primary"}, base)
    fallback_dl = mp._extract_cache_deadline(
        {"url": "u", "proxy_tier": "fallback"}, base)
    assert primary_dl == base + mp._EXTRACT_CACHE_TTL_CAP
    assert fallback_dl == base + mp._FALLBACK_TIER_TTL_SECS
    assert fallback_dl < primary_dl


def test_purge_unlinks_inflight_extraction():
    """A purge-then-re-extract caller must start a FRESH extraction, never
    join a pre-purge future whose result is the thing being purged."""
    import asyncio

    async def scenario():
        fut = asyncio.get_event_loop().create_future()
        mp._extract_inflight["vidpurge0001"] = fut
        await mp._purge_extraction("vidpurge0001")
        assert "vidpurge0001" not in mp._extract_inflight
        assert not fut.cancelled(), "other callers may still await the future"
        fut.set_result({})

    asyncio.run(scenario())


# ── source tripwires (no eslint, no type-level enforcement here) ──────────

def _source() -> str:
    import inspect
    return inspect.getsource(mp)


def test_source_busy_checked_before_outage_in_chain():
    src = _source()
    i_busy = src.index("_is_proxy_busy(e)")
    i_outage = src.index("_is_proxy_outage(e)")
    assert i_busy < i_outage, (
        "the extraction chain must classify busy BEFORE outage — busy errors "
        "match the outage markers, and outage-first turns every replica "
        "collision into a platform-wide OUTAGE alarm"
    )


def test_source_stream_recovery_guards():
    """The stream path's recovery must (a) only fire when a DISTINCT fallback
    is configured, (b) never fire for a prefetch, (c) never loop on a result
    already on the fallback tier, (d) also fire on durable outages (dead
    primary creds strand warm cache entries otherwise), and (e) repoint
    upstream_url before rebuilding the request."""
    src = _source()
    i = src.index("re-extracting toward the")
    window = src[i - 1600 : i + 6000]
    assert "_fallback_available()" in window
    assert "not prefetch" in window
    assert 'result.get("proxy_tier") != "fallback"' in window
    assert "_is_proxy_outage(e)" in window
    assert 'upstream_url = result["url"]' in window


def test_source_stream_recovery_serves_via_spool_and_forces_fallback():
    """(a) The recovery serves through the spool (one download per track,
    even on the recovery path); (b) a second refused CONNECT forces a
    fallback-tier extraction PAST both caches (the purge races concurrent
    cache readers that resurrect the primary entry)."""
    src = _source()
    i = src.index("re-extracting toward the")
    window = src[i : i + 8000]
    assert "_spool_get_or_start(video_id, result)" in window
    assert "_extract_audio_via" in window
    assert '"fallback",' in window
    i_forced = window.index("_extract_audio_via")
    assert "except asyncio.TimeoutError" in window[i_forced:], (
        "the recovery must keep the 504 timeout taxonomy"
    )
