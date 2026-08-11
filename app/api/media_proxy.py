"""Media proxy — fetches direct video / audio stream URLs.

Uses yt-dlp (Python library) to extract stream URLs from YouTube.
Piped APIs are unreliable (frequent downtime), so we use yt-dlp directly.

Two routes:

- `GET /media/stream/{video_id}` — combined video+audio mp4, used by
  cast-to-TV. Uses `format: best[ext=mp4]/best`.
- `GET /media/{video_id}/audio_url` — stream used by the mobile hybrid
  player for background/lock-screen playback. Uses
  `format: 18/22/bestaudio[ext=m4a]/bestaudio/best` — itag 18 (progressive
  360p+AAC) first because iOS AVPlayer can't play YouTube's now-fragmented
  audio-only DASH; see the _PLAYER_CLIENTS note. ToS posture: extracting
  direct audio URLs and playing them outside YouTube's player violates
  YouTube ToS §5.B; the product team accepted the tail risk in exchange
  for the video-when-visible / audio-when-not UX. See SKILL.md in the
  radio-mode skill.
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import time
from urllib.parse import parse_qs, urlparse

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.api.auth import get_current_user
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/media", tags=["Media"])


@router.get("/stream/{video_id}")
async def get_stream(video_id: str):
    """Extract direct video/audio stream URL from YouTube via yt-dlp."""
    import asyncio

    def _extract(vid: str) -> dict:
        # Method 1: yt-dlp (most reliable)
        try:
            import yt_dlp
            opts = {
                "quiet": True,
                "no_warnings": True,
                "skip_download": True,
                "format": "best[ext=mp4]/best",
                "socket_timeout": 10,
            }
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(f"https://www.youtube.com/watch?v={vid}", download=False)
                if info and info.get("url"):
                    return {
                        "url": info["url"],
                        "title": info.get("title", ""),
                        "thumbnail": info.get("thumbnail", ""),
                        "duration": info.get("duration", 0),
                        "ext": info.get("ext", "mp4"),
                    }
        except Exception as e:
            logger.warning("[media_proxy] yt-dlp failed: %s", e)

        # Method 2: Piped API fallback (multiple instances)
        import urllib.request
        import json as _json
        piped_apis = [
            "https://pipedapi.kavin.rocks",
            "https://pipedapi.adminforge.de",
            "https://watchapi.whatever.social",
            "https://api.piped.yt",
        ]
        for api in piped_apis:
            try:
                req = urllib.request.Request(
                    f"{api}/streams/{vid}",
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                with urllib.request.urlopen(req, timeout=8) as resp:
                    data = _json.loads(resp.read())
                    hls = data.get("hls")
                    if hls:
                        return {
                            "url": hls,
                            "title": data.get("title", ""),
                            "thumbnail": data.get("thumbnailUrl", ""),
                            "type": "hls",
                        }
            except Exception as e:
                logger.warning("[media_proxy] Piped %s failed: %s", api, e)

        return {"error": "All extraction methods failed"}

    try:
        result = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(None, _extract, video_id),
            timeout=25,
        )
    except asyncio.TimeoutError:
        return JSONResponse(status_code=504, content={"error": "Stream extraction timed out"})

    if "error" in result:
        return JSONResponse(status_code=502, content=result)
    return result


# ── Audio-only URL extraction for the hybrid foreground/background player ──

# YouTube signed URLs encode expiry as `&expire=<unix_ts>` in the query. If
# we can't parse it, fall back to a conservative 6h default so the caller
# doesn't cache a URL forever.
_DEFAULT_AUDIO_URL_TTL_SECS = 6 * 3600
_AUDIO_EXTRACT_TIMEOUT_SECS = 25


def _parse_expires(url: str) -> int:
    try:
        q = parse_qs(urlparse(url).query)
        raw = (q.get("expire") or [""])[0]
        exp = int(raw) if raw else 0
        if exp > 0:
            return exp
    except Exception:
        pass
    return int(time.time()) + _DEFAULT_AUDIO_URL_TTL_SECS


def _mime_from_ext(ext: str) -> str:
    e = (ext or "").lower()
    if e == "m4a":
        return "audio/mp4"
    if e == "webm":
        return "audio/webm"
    if e == "mp3":
        return "audio/mpeg"
    # Safe default for iOS AVFoundation.
    return "audio/mp4"


def _pick_artwork(info: dict) -> str:
    """Pick the largest thumbnail yt-dlp returned for MPNowPlayingInfoCenter.

    yt-dlp's `thumbnails` is a list of `{url, width, height, ...}`. Score by
    pixel area; tie-break is first-seen. Fall back to the flat `thumbnail`
    string when the list is empty or malformed.
    """
    thumbs = info.get("thumbnails") or []
    if isinstance(thumbs, list) and thumbs:
        def _area(t: dict) -> int:
            try:
                return int(t.get("width") or 0) * int(t.get("height") or 0)
            except Exception:
                return 0
        best = max(thumbs, key=_area, default=None)
        if best and (best.get("url") or "").strip():
            return best["url"].strip()
    return (info.get("thumbnail") or "").strip()


# YouTube anti-bot detection on cloud-provider IPs (Railway, AWS, GCP) is
# aggressive and tightens over time. The default `web` extractor is the
# easiest signature for YouTube to flag — it gets the "Sign in to confirm
# you're not a bot" challenge first. We fall through a chain of clients
# that hit different YouTube API endpoints with different headers; the
# lighter-weight non-web clients are far less frequently flagged.
#
# Order chosen by empirical hit rate on Railway egress IPs (2026-04):
#   - tv_embedded   YouTube TV embedded player; most permissive overall.
#   - android_music YouTube Music app on Android; ideal for music tracks
#                   because that's literally what these are designed to
#                   stream. Often returns higher-bitrate audio than `web`.
#   - ios           iOS YouTube app extractor; reliable for non-music too.
#   - android       Android YouTube app; falls back here when the music
#                   client refuses (live streams, podcasts).
#
# Web/mweb deliberately omitted: they're the first to be bot-flagged and
# rarely succeed when the others fail. Adding them just burns latency.
#
# Order matters; `android` leads because it's the only client that still
# yields a stream iOS AVFoundation can play. Hard-won 2026-06-03:
#  - android_music / tv_embedded are now "unsupported client" in current
#    yt-dlp (silently skipped) — they used to give progressive itag-140 m4a.
#  - Every remaining client now serves AUDIO-ONLY (itag 140) as a *fragmented*
#    DASH MP4 (tiny moov + sidx + moof/mdat). AVPlayer CANNOT play fragmented
#    MP4 over a plain progressive HTTP URL → SwiftAudioEx PlaybackError 4,
#    silent lock screen. (The foreground iframe is unaffected — it's WebKit.)
#  - The ONE progressive container YouTube still serves is itag 18 (360p
#    H.264 + AAC, single faststart moov, no fragments). It carries video we
#    don't show, but the audio decodes natively on the lock screen. The
#    `format` selector below grabs 18 first; the m4a fall-backs remain only so
#    extraction still *succeeds* for non-iOS callers / videos without itag 18.
#  - web clients need a GVS PO token (bgutil plugin) we don't have installed,
#    and even with it would only expose the unplayable fragmented audio. Kept
#    as last-ditch fallbacks; they normally just error through.
_PLAYER_CLIENTS = ("android", "ios", "web_safari", "web")
# Fast first attempt: android + player_skip (one /player POST, no ~1MB
# watch-page through the throttled proxy). Fail-open to the full chain on any
# error — see the attempt loop in _extract_audio. AUDIO_EXTRACT_FAST=0 kills it.
_EXTRACT_FAST_FIRST = os.environ.get("AUDIO_EXTRACT_FAST", "1").strip().lower() not in (
    "0", "false", "no", "off",
)


def _should_try_next_client(err: BaseException) -> bool:
    """Should this error fall through to the next client in the chain?

    Two distinct fall-through cases:

    1. **Bot-block.** YouTube's "Sign in to confirm you're not a bot"
       challenge — a different extractor client uses different headers
       and API endpoints, so it may slip past.

    2. **Per-client format gap.** Some clients (notably `ios`) expose
       only a subset of formats. yt-dlp emits "Requested format is not
       available" rather than degrading. Trying a different client
       (e.g. `android`) often surfaces the full format catalog.

    Real "video unavailable" errors (private / removed / region-blocked
    / age-gated) are unchanged across clients and abort the chain so
    callers get a fast, accurate 502.
    """
    msg = str(err).lower()
    if (
        "sign in to confirm you're not a bot" in msg
        or "sign in to confirm you" in msg
        or "confirm you're not a bot" in msg
    ):
        return True
    if "requested format is not available" in msg:
        return True
    return False


# The residential proxy failing is NOT a property of the video, and every
# player client goes through the same proxy — so retrying clients is wasted
# latency and the per-video log line is actively misleading. On 2026-08-03 the
# IPRoyal account ran out of prepaid traffic and answered every CONNECT with
# `402 Payment Required`. Audio died platform-wide for every track that was not
# already cached, and the only trace was
# `[media_proxy] client=android permanent error video_id=… — abort`, one line
# per video, indistinguishable from a handful of unavailable songs. Nothing
# alerted; it surfaced as a user complaint hours later.
_PROXY_OUTAGE_MARKERS = (
    "tunnel connection failed",
    "unable to connect to proxy",
    "402 payment required",
    "407 proxy authentication required",
    "proxyerror",
)


def _is_proxy_outage(err: BaseException) -> bool:
    """Did extraction fail because the egress proxy itself is unusable?

    True means: out of prepaid traffic, bad credentials, or the proxy is down.
    Whole-platform, operator-actionable, and identical for every video — as
    opposed to a bot-block or an unavailable video, which are per-request.

    NOTE: a device-lock 403 (`_is_proxy_busy`) also matches these markers, so
    callers that can fall back must check busy FIRST — busy is transient by
    construction and must not raise the platform-wide outage alarm.
    """
    msg = str(err).lower()
    return any(m in msg for m in _PROXY_OUTAGE_MARKERS)


# The proxy's own CONNECT rejection when another source IP holds its bind.
# 403 specifically: auth failures are 407, exhausted credit is 402 — both
# durable outages. 403 at CONNECT is "come back in a few seconds".
_PROXY_BUSY_MARKER = "tunnel connection failed: 403"


def _is_proxy_busy(err: BaseException) -> bool:
    """Did the egress proxy refuse the CONNECT because its device-lock is held?

    IPRoyal static (ISP) proxies are IP-BOUND: one source IP may use them at a
    time, and the bind is a ~10s sticky window from connection start (measured
    2026-08-10 — a second source IP gets a CONNECT 403 from the proxy itself,
    then succeeds once the window lapses; tunnels already established survive
    a takeover). platform-api runs 2 replicas with distinct egress IPs, so
    concurrent cold plays on different replicas collide exactly this way.
    Transient, NOT an outage.

    TWO phrasings for the SAME event, and both call sites are live: yt-dlp
    (extraction) wraps urllib's `Tunnel connection failed: 403 Forbidden`,
    while httpx (spool download, stream byte-pump) raises `ProxyError("403
    Forbidden")` — no "tunnel" substring at all. Matching only the urllib
    phrasing made the stream path's busy recovery dead code on 2026-08-10
    (prod log: `spool FAILED …: 403 Forbidden`); an upstream googlevideo 403
    can never reach here as an exception (it arrives as a response object),
    so the ProxyError isinstance check cannot over-match."""
    msg = str(err).lower()
    if _PROXY_BUSY_MARKER in msg:
        return True
    return isinstance(err, httpx.ProxyError) and "403" in msg


_RESOLVED_COOKIEFILE: str | None = None
_COOKIES_RESOLVED = False


def _resolve_cookiefile() -> str | None:
    """Path to a Netscape cookies.txt for yt-dlp, if configured.

    Cookies from a logged-in YouTube session let yt-dlp pass YouTube's
    "Sign in to confirm you're not a bot" challenge that otherwise blocks
    extraction on flagged cloud (Railway) egress IPs. Two ways to supply:
      - YT_DLP_COOKIES_PATH: absolute path to a cookies.txt on disk.
      - YT_DLP_COOKIES_B64: base64 of a cookies.txt — easiest on Railway
        (set it as an env var, no file mounting). Decoded once to a temp file.
    Memoized; a Railway var change restarts the process, re-resolving.
    """
    global _RESOLVED_COOKIEFILE, _COOKIES_RESOLVED
    if _COOKIES_RESOLVED:
        return _RESOLVED_COOKIEFILE
    import os
    import base64
    import tempfile
    path = os.environ.get("YT_DLP_COOKIES_PATH")
    if path and os.path.exists(path):
        _RESOLVED_COOKIEFILE = path
    else:
        b64 = os.environ.get("YT_DLP_COOKIES_B64")
        if b64:
            try:
                data = base64.b64decode(b64)
                fd, tmp = tempfile.mkstemp(prefix="yt_cookies_", suffix=".txt")
                with os.fdopen(fd, "wb") as f:
                    f.write(data)
                _RESOLVED_COOKIEFILE = tmp
                logger.info("[media_proxy] cookies loaded from YT_DLP_COOKIES_B64 (%d bytes)", len(data))
            except Exception as e:
                logger.warning("[media_proxy] failed to decode YT_DLP_COOKIES_B64: %s", e)
                _RESOLVED_COOKIEFILE = None
        else:
            _RESOLVED_COOKIEFILE = None
    _COOKIES_RESOLVED = True
    return _RESOLVED_COOKIEFILE


# ── Egress session pool ───────────────────────────────────────────────────
# `YT_DLP_PROXY` names ONE IPRoyal sticky session (`session-toupaudio`,
# `lifetime-30m`), so every extraction, every byte-pump and every remux build
# for every user on both replicas leaves through ONE residential peer's home
# uplink. That is a single point of bandwidth for the entire product, and the
# draw is a lottery re-run every 30 minutes.
#
# Measured 2026-08-05, downloading 2.5MB through each session (Cloudflare, so
# the number is the proxy's uplink and not googlevideo's):
#
#   production's pinned session   0.63 MB/s   (5 samples: 0.52 0.62 0.63 0.67 0.65)
#   20 fresh draws                median 2.04, p25 0.75, p75 3.90, max 8.41 MB/s
#
# 16 of 20 draws beat the peer production was stuck on. For iOS's ~2.5MB
# pre-roll that is 3.9s against 1.2s — most of the founder's cold-start report,
# with no code path at fault.
#
# So spread the draw. A slot is chosen per VIDEO, not per request, because
# googlevideo binds the signed URL to the extracting IP: extraction, the
# byte-pump and the remux download must all leave from the same address.
#
# The chosen slot is CARRIED IN THE EXTRACTION RESULT rather than recomputed
# from the video id. Recomputing looks equivalent and is not — it would make
# every cached URL (L1 for an hour, R2 across replicas) depend on the pool size
# being the same when it is read as when it was written, so changing
# AUDIO_PROXY_SESSION_POOL would silently remap every warm entry onto the wrong
# egress and 403 the lot at once.
#
# Slot 0 is the unmodified URL, so AUDIO_PROXY_SESSION_POOL=1 is byte-identical
# to today's behaviour and is the rollback.
_PROXY_SESSION_POOL = max(1, int(os.environ.get("AUDIO_PROXY_SESSION_POOL", "8")))
_PROXY_SESSION_RE = re.compile(r"(session-)([A-Za-z0-9]+)")


def _proxy_slot_for(video_id: str) -> int:
    if _PROXY_SESSION_POOL <= 1 or not video_id:
        return 0
    return int(hashlib.sha256(video_id.encode()).hexdigest()[:8], 16) % _PROXY_SESSION_POOL


def _proxy_base(tier: str) -> str:
    """The configured proxy URL for an egress tier.

    "primary" is `yt_dlp_proxy` (in production: a static ISP IP — fast, but
    IP-bound to one source at a time); "fallback" is `yt_dlp_proxy_fallback`
    (the rotating residential gateway — slower, no device-lock). An unset
    fallback returns "" and every fallback engagement point checks
    `_fallback_available` first, so a single-proxy config behaves
    byte-identically to before tiers existed.
    """
    if tier == "fallback":
        return (settings.yt_dlp_proxy_fallback or "").strip()
    return (settings.yt_dlp_proxy or "").strip()


def _fallback_available() -> bool:
    """Is there a DISTINCT fallback egress worth engaging?

    Equal URLs make every "retrying via fallback tier" log line a lie and
    double the latency of every honest failure for zero benefit — a config
    slip the order form invites ("set both"), so it is neutralized here
    rather than documented against (adversarial review, 2026-08-10).
    """
    fb = _proxy_base("fallback")
    return bool(fb) and fb != _proxy_base("primary")


# Circuit breaker for a DURABLY failing primary. A bot-flagged or dead-creds
# primary otherwise costs every uncached play the full 5-attempt primary
# chain (10s+ of tail) before the fallback saves it. Busy NEVER trips this —
# the device-lock is a ~10s window and skipping the primary for minutes over
# it would hand routine collisions a permanent residential demotion.
_PRIMARY_BREAK_SECS = float(os.environ.get("AUDIO_PRIMARY_BREAK_SECS", "120"))
_primary_down_until: float = 0.0


def _primary_broken() -> bool:
    return time.monotonic() < _primary_down_until


def _trip_primary_breaker(reason: str, video_id: str) -> None:
    global _primary_down_until
    _primary_down_until = time.monotonic() + _PRIMARY_BREAK_SECS
    logger.error(
        "[media_proxy] PRIMARY BREAKER tripped (%s) — extraction goes "
        "straight to the fallback tier for %.0fs. video_id=%s",
        reason, _PRIMARY_BREAK_SECS, video_id,
    )


def _proxy_with_slot(slot: int, tier: str = "primary") -> str | None:
    """The tier's proxy URL with its sticky-session token varied by `slot`.

    Returns the URL untouched for slot 0, for a provider whose URL carries no
    session token (the static primary), and for anything it cannot parse — a
    proxy we fail to understand must still be used exactly as configured,
    never dropped.
    """
    base = _proxy_base(tier)
    if not base:
        return None
    if not slot:
        return base
    # Split on the LAST '@' so a password containing '@' stays intact, and edit
    # the credentials as text: `urlsplit` does not percent-decode userinfo, so a
    # decode/re-encode round trip would corrupt an already-encoded password.
    head, sep, host = base.rpartition("@")
    if not sep or not _PROXY_SESSION_RE.search(head):
        return base
    return _PROXY_SESSION_RE.sub(
        lambda m: f"{m.group(1)}{m.group(2)}{slot}", head, count=1
    ) + sep + host


def _proxy_for_result(result: dict) -> str | None:
    """The egress the given extraction was taken through. See `_proxy_slot_for`.

    A cached extraction from before this shipped has no slot; treating that as
    slot 0 is exactly right, because it WAS taken through the unmodified URL.
    Same for `proxy_tier`: absent means primary, which is what every result
    predating tiers was extracted through. googlevideo signs the URL to the
    extracting IP, so a fallback-tier result MUST be fetched via the fallback
    egress — if the operator unsets the fallback env while such a result is
    cached, this returns None (direct fetch), googlevideo 403s the mismatch,
    and the existing 403/410 purge-and-re-extract path self-heals it.
    """
    tier = "fallback" if result.get("proxy_tier") == "fallback" else "primary"
    try:
        return _proxy_with_slot(int(result.get("proxy_slot") or 0), tier)
    except (TypeError, ValueError):
        return _proxy_with_slot(0, tier)


def _extract_audio(video_id: str) -> dict:
    """Blocking extract with EGRESS TIERS; callers run it in an executor.

    Tries the primary egress first. Three failure classes there engage the
    fallback tier when one is configured (`yt_dlp_proxy_fallback`):

      • proxy_busy — the static primary's device-lock is held by the other
        replica (see `_is_proxy_busy`); transient, the fallback serves NOW.
      • proxy_unavailable — dead credentials / no credit / proxy down. This
        is the 2026-08-10 morning incident: the provider rotated the primary's
        credentials and every uncached play 502'd platform-wide; with a
        fallback configured the same event costs one WARNING line instead.
      • all_clients_blocked — YouTube bot-flagged the primary IP itself (how
        the first two US ISP draws died). Static IPs can burn at any time;
        falling back turns that into degraded speed instead of dead audio.

    Permanent per-video errors (private/removed/region-locked) abort on the
    primary WITHOUT a fallback attempt — a second chain can't fix the video
    and would double the latency of every honest failure.

    The winning tier is stamped into the result as `proxy_tier`, and every
    byte-fetch reads it back via `_proxy_for_result` — googlevideo signs the
    URL to the extracting IP, so extraction and bytes must share an egress.
    """
    slot = _proxy_slot_for(video_id)
    if _primary_broken() and _fallback_available():
        return _extract_audio_via(video_id, slot, "fallback")
    result = _extract_audio_via(video_id, slot, "primary")
    err = result.get("error")
    if err in ("proxy_busy", "proxy_unavailable", "all_clients_blocked") and _fallback_available():
        if err != "proxy_busy":
            # Durable primary failure (dead creds / bot-flagged IP): stop
            # paying the full primary chain on every uncached play until the
            # breaker window lapses. Busy never trips it — see the breaker note.
            _trip_primary_breaker(err, video_id)
        else:
            logger.warning(
                "[media_proxy] primary egress busy — retrying via fallback tier video_id=%s",
                video_id,
            )
        result = _extract_audio_via(video_id, slot, "fallback")
    return result


def _extract_audio_via(video_id: str, slot: int, tier: str) -> dict:
    """One egress tier's multi-client extraction chain. Blocking.

    Multi-client fallback is the production posture for working around
    YouTube's anti-bot challenges on cloud egress. We try clients in
    order of empirical reliability and short-circuit on first success.
    Bot-block errors fall through to the next client; non-bot errors
    (real "this video is unavailable") abort the chain so the caller
    sees a fast, accurate 502 rather than waiting on every client.

    Optional cookies file: set YT_DLP_COOKIES_PATH in the Railway env to
    a Netscape-format cookies.txt for the operator's own YouTube session.
    Last-resort defense if every client gets blocked simultaneously.
    Cookies aren't shipped by default — keeping the operator-credential
    surface area small until we actually need it.
    """
    import yt_dlp
    import os

    cookiefile = _resolve_cookiefile()
    pot_base = (settings.bgutil_pot_base_url or "").strip()
    last_err: BaseException | None = None
    # One slot for the whole extraction, and it travels with the result: the
    # byte-pump and the remux download must leave from the address that signed
    # the URL. See the pool note above. The slot is chosen by the ORCHESTRATOR
    # so both tiers share it — it only rewrites URLs that carry a session
    # token, so it is inert on the static primary and active on the fallback.
    proxy = _proxy_with_slot(slot, tier)

    # FAST FIRST ATTEMPT (flag: AUDIO_EXTRACT_FAST=0 to disable): android with
    # player_skip — one innertube /player POST instead of the ~1MB watch-page
    # fetch + /player. Every yt-dlp byte rides the residential proxy (median
    # 0.73 MB/s, TTFB 2.4-4.2s measured 2026-08-09), so the skipped webpage is
    # most of the 3.4s median. Fail-open: ANY failure of this attempt falls
    # through to the unmodified full chain below (proxy outage still aborts —
    # retrying three more clients through a dead egress helps nobody).
    attempts: list[tuple[str, bool]] = [(c, False) for c in _PLAYER_CLIENTS]
    if _EXTRACT_FAST_FIRST:
        attempts.insert(0, ("android", True))

    for client, fast in attempts:
        extractor_args: dict = {"youtube": {"player_client": [client]}}
        if fast:
            extractor_args["youtube"]["player_skip"] = ["webpage", "configs", "initial_data"]
        # Arm the bgutil PO-token provider. No-op for android_music/tv_embedded
        # (they don't request a token), but if YouTube's bot challenge fires on
        # the server IP the android/ios gvs clients fetch a proof-of-origin
        # token from it and slip past — the free anti-bot defense.
        if pot_base:
            extractor_args["youtubepot-bgutilhttp"] = {"base_url": [pot_base]}
        opts: dict = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            # itag 18 first: the only PROGRESSIVE (non-fragmented, faststart
            # moov) container YouTube still serves, and the only thing iOS
            # AVPlayer can play off a plain HTTP URL. It carries 360p video we
            # never show — but YouTube now delivers audio-only itag 140 as
            # fragmented DASH, which AVPlayer rejects (PlaybackError 4, silent
            # lock screen). The m4a/bestaudio tail keeps extraction succeeding
            # for non-iOS callers and the rare video with no itag 18. See the
            # _PLAYER_CLIENTS note. Changed 2026-06-03.
            "format": "18/22/bestaudio[ext=m4a]/bestaudio/best",
            "socket_timeout": 10,
            "extractor_args": extractor_args,
        }
        if cookiefile:
            opts["cookiefile"] = cookiefile
        # Route extraction through a residential proxy when configured — this
        # is the reliable defense against YouTube's datacenter-IP bot challenge.
        if proxy:
            opts["proxy"] = proxy

        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(
                    f"https://www.youtube.com/watch?v={video_id}",
                    download=False,
                )
        except Exception as e:
            last_err = e
            # Check the proxy BEFORE the per-client fall-through: a dead proxy
            # also matches the bot-block heuristic sometimes, and retrying the
            # other three clients through the same dead egress just multiplies
            # the latency of a request that cannot succeed. Busy FIRST — a
            # device-locked proxy also matches the outage markers, and the
            # distinction is what keeps a routine replica collision from
            # raising the platform-wide outage alarm.
            if _is_proxy_busy(e) and tier == "primary":
                # Only the static primary HAS a device-lock. A CONNECT 403
                # from the rotating fallback gateway is durable (whitelist /
                # account trouble) and must fall through to the OUTAGE
                # classification below, not masquerade as a transient
                # collision (adversarial review, 2026-08-10).
                logger.warning(
                    "[media_proxy] egress BUSY (device-lock) tier=%s "
                    "video_id=%s client=%s",
                    tier, video_id, client,
                )
                return {"error": "proxy_busy", "detail": str(e)}
            if _is_proxy_outage(e):
                logger.error(
                    "[media_proxy] PROXY OUTAGE — egress tier=%s is unusable%s. "
                    "Check YT_DLP_PROXY%s credit/credentials. "
                    "video_id=%s client=%s err=%s",
                    tier,
                    (", ALL uncached audio is failing platform-wide (not "
                     "this video)") if tier == "fallback" or not _proxy_base("fallback")
                    else " — retrying via fallback tier",
                    "_FALLBACK" if tier == "fallback" else "",
                    video_id, client, str(e)[:200],
                )
                return {"error": "proxy_unavailable", "detail": str(e)}
            if fast:
                # The fast attempt is an optimization, never an oracle: its
                # error strings differ from the full chain's (no webpage, no
                # configs), so judging permanence from them would wrongly
                # abort videos the full chain can serve. Fall through to the
                # unmodified chain for EVERY non-outage failure.
                logger.info(
                    "[media_proxy] fast-extract fall-through video_id=%s err=%s",
                    video_id, str(e)[:120],
                )
                continue
            if _should_try_next_client(e):
                logger.warning(
                    "[media_proxy] client=%s fall-through video_id=%s err=%s",
                    client, video_id, str(e)[:120],
                )
                continue
            # Real "video unavailable" (private/removed/region-blocked/
            # age-gated) — won't change with a different client. Abort
            # so the caller gets a fast, accurate error.
            logger.warning(
                "[media_proxy] client=%s permanent error video_id=%s: %s — abort",
                client, video_id, e,
            )
            return {"error": "extraction_failed", "detail": str(e)}

        if not info or not info.get("url"):
            # Some clients return None or empty for unsupported video types
            # (e.g. android_music on a non-music video). Try the next one.
            last_err = RuntimeError(f"no_stream_url client={client}")
            continue

        url = info["url"]
        ext = info.get("ext") or "m4a"
        logger.info(
            "[media_proxy] audio_url ok video_id=%s client=%s fast=%d ext=%s bitrate=%s slot=%d tier=%s",
            video_id, client, int(fast), ext, info.get("abr"), slot, tier,
        )
        return {
            "url": url,
            "proxy_slot": slot,
            "proxy_tier": tier,
            "expires_at": _parse_expires(url),
            "duration": int(info.get("duration") or 0),
            "mime_type": _mime_from_ext(ext),
            "ext": ext,
            "bitrate": int(info.get("abr") or 0),
            "title": info.get("title") or "",
            "artwork_url": _pick_artwork(info),
        }

    # Every client got blocked or returned no URL. This is rare in
    # practice; if it starts happening at scale, set YT_DLP_COOKIES_PATH
    # before reaching for residential proxies.
    return {
        "error": "all_clients_blocked",
        "detail": str(last_err) if last_err else "all clients returned no stream",
    }


# video_id -> (result_dict, cache_until_epoch). Caches SUCCESSFUL extractions
# so the lock-screen handoff is instant: AVFoundation fires a range probe, the
# real request, and seeks — each previously triggered a fresh ~2-3s yt-dlp
# call. With the cache they reuse one extraction, and the mobile pre-warm
# (media_play → /audio_url) fills it BEFORE the user locks, so the byte-pump
# skips extraction entirely. Safe because extraction and the byte-pump share
# one stable egress IP (the sticky `YT_DLP_PROXY` session), so the signed
# googlevideo URL stays valid on reuse — the rotating-Railway-IP 403 that
# forced the earlier cache revert (commit 4039597a) can't happen through it.
_EXTRACT_CACHE: dict[str, tuple[dict, float]] = {}
_EXTRACT_CACHE_TTL_CAP = 3600.0  # never serve an extraction older than 1h
_FALLBACK_TIER_TTL_SECS = float(os.environ.get("AUDIO_FALLBACK_TIER_TTL_SECS", "600"))


def _extract_cache_deadline(result: dict, base: float) -> float:
    """When an extraction taken at `base` stops being servable. 0.0 = never cache.

    Both tiers call this, and the shared tier passes the time the extraction was
    TAKEN, not the time it was read — recomputing from `now` on every read would
    renew the 1h staleness cap on each hit and let one extraction live forever.
    """
    if "error" in result or not result.get("url"):
        return 0.0
    deadline = base + _EXTRACT_CACHE_TTL_CAP
    if result.get("proxy_tier") == "fallback":
        # A fallback-tier entry is a DEMOTION: both caches serve it tier-blind
        # and nothing ever re-probes the primary while it lives, so a 10s
        # device-lock collision otherwise parks the video on the slow metered
        # gateway fleet-wide for the full hour (adversarial review,
        # 2026-08-10). A short deadline is the migration path back.
        deadline = min(deadline, base + _FALLBACK_TIER_TTL_SECS)
    exp = result.get("expires_at") or 0
    if exp:
        # Stop serving 5 min before the signed URL itself expires.
        deadline = min(deadline, float(exp) - 300.0)
    return deadline


# ── Shared (cross-replica) extraction cache ───────────────────────────────
# `_EXTRACT_CACHE` above is per-PROCESS, and `railway.json` runs platform-api
# at numReplicas=2. That made the broadcast-time pre-extract a coin flip: the
# agent warms `/api/internal/media/warm` on whichever replica the router picks,
# then the phone's `/audio_stream` lands on either one, so half of all first
# plays threw the warm away and paid the full ~5-7s yt-dlp extraction with the
# user waiting. The remux tier never had this problem because R2 is shared;
# extraction did, because its result lived only in RAM.
#
# The stored value is the signed googlevideo URL, which IS bound to the IP that
# extracted it (`ip=` in the URL; a range probe from an unrelated address 403s —
# measured 2026-08-04). Sharing it between replicas is nonetheless safe, and for
# one specific reason: `YT_DLP_PROXY` pins a STICKY IPRoyal session, so every
# replica reads the same credential and egresses from the same exit address.
# Verified the same day: three consecutive requests through the proxy all left
# from 209.173.194.71, the exact address the signed URLs name.
#
# That session has a 30-minute lifetime while `_EXTRACT_CACHE_TTL_CAP` is an
# hour, so a cached URL CAN outlive the IP that signed it and start 403-ing —
# a pre-existing hazard of the in-process cache that this tier inherits rather
# than introduces. `stream_audio` now treats that 403 as poison: it purges both
# tiers and re-extracts, instead of 502-ing and letting the client's auto-skip
# move the user off the song they asked for.
#
# A dedicated boto3 client because the audio one takes botocore's DEFAULT 60s
# connect/read timeouts. `_r2_pull_to_local` bounds those with
# `asyncio.wait_for`, which is unavailable here — `_cached_extract` is sync,
# inside an executor — so an R2 hiccup would add up to a minute to a hot-path
# play. Tight timeouts and no retries: this tier is an optimisation, and
# giving up on it costs one extraction, not a failure.
_R2_META_TIMEOUT = float(os.environ.get("AUDIO_R2_META_TIMEOUT", "2.5"))
_r2_meta_client = None
_r2_meta_disabled = False


def _get_r2_meta_client():
    global _r2_meta_client, _r2_meta_disabled
    if _r2_meta_disabled or not _r2_ready():
        return None
    if _r2_meta_client is not None:
        return _r2_meta_client
    try:
        import boto3
        from botocore.config import Config
        _r2_meta_client = boto3.client(
            "s3",
            endpoint_url=f"https://{settings.r2_account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=settings.r2_access_key_id,
            aws_secret_access_key=settings.r2_secret_access_key,
            region_name="auto",
            config=Config(
                signature_version="s3v4",
                connect_timeout=_R2_META_TIMEOUT,
                read_timeout=_R2_META_TIMEOUT,
                retries={"max_attempts": 1, "mode": "standard"},
            ),
        )
        return _r2_meta_client
    except Exception as e:
        logger.warning("[media_proxy] R2 meta client init failed — shared extract cache off: %s", e)
        _r2_meta_disabled = True
        return None


def _r2_extract_key(video_id: str) -> str:
    # Own prefix: `_r2_key` puts the audio at the bucket root as `<id>.m4a`.
    return f"extract/{video_id}.json"


def _shared_extract_load(video_id: str, now: float) -> tuple[dict, float] | None:
    """Read another replica's extraction. None on miss, staleness or any error.

    Blocking — same executor contract as `_cached_extract`, its only caller.
    """
    client = _get_r2_meta_client()
    if client is None:
        return None
    t0 = time.monotonic()
    try:
        obj = client.get_object(Bucket=settings.r2_bucket, Key=_r2_extract_key(video_id))
        payload = json.loads(obj["Body"].read())
    except Exception as e:
        # A miss is the normal case for any track nobody has played this hour,
        # so it must not be a warning. Everything else — bad credentials, a
        # missing bucket, a timeout, a corrupt object — is logged, because a
        # cache that silently never hits is indistinguishable from one that is
        # merely cold. Match on the S3 error CODE, not the exception class:
        # AccessDenied and NoSuchKey are both `ClientError`.
        code = ""
        if isinstance(getattr(e, "response", None), dict):
            code = str(e.response.get("Error", {}).get("Code") or "")
        if code in ("NoSuchKey", "404"):
            return None
        logger.warning("[media_proxy] shared extract read failed video_id=%s: %s", video_id, e)
        return None

    result = payload.get("result")
    stored_at = payload.get("stored_at")
    if not isinstance(result, dict) or not isinstance(stored_at, (int, float)):
        return None
    deadline = _extract_cache_deadline(result, float(stored_at))
    if deadline <= now:
        return None
    logger.info(
        "[media_proxy] extract cache HIT tier=shared video_id=%s age_s=%.0f r2_ms=%.0f",
        video_id, now - float(stored_at), (time.monotonic() - t0) * 1000,
    )
    return result, deadline


def _shared_extract_store(video_id: str, result: dict, stored_at: float) -> None:
    """Publish a successful extraction for the other replicas. Best-effort.

    Called OFF the awaited path — see `_publish_shared_extract`. It used to run
    inline at the end of `_cached_extract`, on the reasoning that a sub-second
    PUT is noise beside a ~5-7s extraction. That is true of the median and not
    of the shape: `_cached_extract` runs inside the executor whose future the
    phone is blocked on, so every millisecond here is a millisecond of silence,
    and the R2 client's 2.5s connect + 2.5s read ceiling is 5s of it when R2 is
    slow. Nothing waits on the publish, so nothing should block for it.
    """
    client = _get_r2_meta_client()
    if client is None:
        return
    try:
        client.put_object(
            Bucket=settings.r2_bucket,
            Key=_r2_extract_key(video_id),
            Body=json.dumps({"v": 1, "stored_at": stored_at, "result": result}).encode(),
            ContentType="application/json",
            CacheControl="no-store",
        )
    except Exception as e:
        logger.warning("[media_proxy] shared extract store failed video_id=%s: %s", video_id, e)


async def _purge_extraction(video_id: str, poisoned_url: str | None = None) -> None:
    """Drop a poisoned extraction from BOTH tiers.

    Clearing only the local tier would leave the bad URL in R2 for every other
    replica to read — and for this one to read back on its next miss.

    `poisoned_url` makes the shared delete CONDITIONAL, and that matters with
    two replicas. Replica A can be pumping an hour-old L1 entry that has just
    started 403-ing at the same moment replica B publishes a fresh extraction
    for the same video. An unconditional delete has A destroy B's good entry,
    and the next play on either replica pays a full ~10s yt-dlp run for nothing.
    Deleting only when the stored URL is the one that actually failed makes the
    purge idempotent and safe to race. Omitted (or unreadable) → unconditional,
    which is the old behaviour and the safe direction for a caller that cannot
    say what it was holding.
    """
    _EXTRACT_CACHE.pop(video_id, None)
    # Unlink (never cancel — other callers may be awaiting it) any in-flight
    # extraction: a purge-then-re-extract caller must start a FRESH run, not
    # join a pre-purge future whose result is the very thing being purged
    # (adversarial review, 2026-08-10). The orphaned future's own tail may
    # still re-cache its result; the conditional shared purge plus the busy
    # recovery's post-re-extract connect check bound that window.
    _extract_inflight.pop(video_id, None)

    def _drop() -> None:
        client = _get_r2_meta_client()
        if client is None:
            return
        if poisoned_url:
            try:
                obj = client.get_object(
                    Bucket=settings.r2_bucket, Key=_r2_extract_key(video_id)
                )
                stored = (json.loads(obj["Body"].read()) or {}).get("result") or {}
            except Exception:
                stored = None
            if isinstance(stored, dict) and stored.get("url") != poisoned_url:
                logger.info(
                    "[media_proxy] shared extract already replaced — not purging video_id=%s",
                    video_id,
                )
                return
        client.delete_object(Bucket=settings.r2_bucket, Key=_r2_extract_key(video_id))

    try:
        await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(None, _drop),
            timeout=_R2_META_TIMEOUT * 2,
        )
    except Exception as e:
        # Not fatal — the object carries its own deadline and the local tier is
        # already clear — but a purge that keeps failing means the shared tier
        # is serving a bad URL to the fleet until it expires. Say so.
        logger.warning("[media_proxy] shared extract purge failed video_id=%s: %s", video_id, e)


def _publish_shared_extract(video_id: str, result: dict, stored_at: float) -> None:
    """Hand the shared-tier PUT to a plain daemon thread and return immediately.

    A thread rather than `asyncio.create_task`, because the only caller runs in
    an executor worker with no running event loop of its own. Daemon, because a
    best-effort cache write must never hold up interpreter shutdown; the object
    it would have written is one an extraction re-creates.
    """
    try:
        import threading
        threading.Thread(
            target=_shared_extract_store,
            args=(video_id, result, stored_at),
            name=f"extract-publish-{video_id}",
            daemon=True,
        ).start()
    except Exception as e:
        # A thread we cannot start is not a reason to fail a play that already
        # has its URL. Fall back to publishing inline: slower, still correct.
        logger.warning("[media_proxy] shared extract publish thread failed: %s", e)
        _shared_extract_store(video_id, result, stored_at)


def _cached_extract(video_id: str) -> dict:
    """`_extract_audio` with a two-tier per-video cache. Blocking; run in an
    executor like the underlying call. Only successful results are cached."""
    now = time.time()
    hit = _EXTRACT_CACHE.get(video_id)
    if hit and hit[1] > now:
        logger.info("[media_proxy] extract cache HIT tier=L1 video_id=%s", video_id)
        return hit[0]

    shared = _shared_extract_load(video_id, now)
    if shared is not None:
        result, deadline = shared
        _EXTRACT_CACHE[video_id] = (result, deadline)
        return result

    result = _extract_audio(video_id)
    deadline = _extract_cache_deadline(result, now)
    if deadline > now:
        _EXTRACT_CACHE[video_id] = (result, deadline)
        # Publish to the shared tier WITHOUT holding the caller. See
        # `_publish_shared_extract` — this function's future is what the phone
        # is waiting on, so a PUT here is silence the user hears.
        _publish_shared_extract(video_id, result, now)
    return result


# In-flight extractions, keyed by video_id. `_EXTRACT_CACHE` only coalesces
# calls that arrive AFTER one finished; concurrent callers all missed the cache
# and each ran their own yt-dlp. That is the normal case, not a rare race: the
# mobile client fires `/audio_url` (the pre-warm) and `/audio_stream` (the play)
# for the same track in the same tick, so every cold first play paid for two
# full extractions — two subprocesses and two round trips through the single
# residential proxy, whose uplink is the bottleneck the user is waiting on.
# The remux builder already coalesces this way (`_remux_tasks`); extraction did
# not.
_extract_inflight: dict[str, "asyncio.Future[dict]"] = {}


async def _extract_coalesced(video_id: str) -> dict:
    """`_cached_extract` off the event loop, with one extraction per video.

    Shielded so a caller that gives up (its own `wait_for` timeout, or a client
    disconnect) cancels only its own wait — the extraction keeps running for
    everyone else joined to it, and the result still lands in `_EXTRACT_CACHE`.
    """
    task = _extract_inflight.get(video_id)
    if task is None or task.done():
        # `run_in_executor` already returns an awaitable Future — do NOT wrap it
        # in `create_task`, which demands a coroutine and raises TypeError on a
        # Future. That mistake makes every extraction fail with a 502 rather
        # than merely failing to coalesce.
        task = asyncio.get_event_loop().run_in_executor(
            None, _cached_extract, video_id,
        )
        _extract_inflight[video_id] = task
        # Pop by identity: a later extraction for the same video may already
        # own the slot by the time this one finishes.
        task.add_done_callback(
            lambda t, v=video_id: (
                _extract_inflight.pop(v, None) if _extract_inflight.get(v) is t else None
            )
        )
    else:
        logger.info("[media_proxy] extract JOINED in-flight video_id=%s", video_id)
    return await asyncio.shield(task)


# ── Audio-only remux cache (fast first-play + skip) ───────────────────────
# itag 18 — the only progressive container iOS plays off a plain URL — carries
# 360p VIDEO we never show. In song mode AVPlayer still buffers the interleaved
# video to reach the audio, so ~2-3MB must arrive over the ~1MB/s residential
# proxy before a track starts → the ~5s cold-load measured on device for EVERY
# track (first play AND manual skip; the RNTP queue prebuffer only pre-rolls
# near a track's natural end, not on a mid-track skip).
#
# Fix: once per video, download the UN-THROTTLED itag-18 (the same extraction
# the proxy already uses — android/ios client + POT pulls at ~2-5 MB/s) and
# ffmpeg-strip the video into a PROGRESSIVE faststart audio-only m4a, cached on
# local disk; /audio_stream then serves THAT. Result: audio-only (~25x less data)
# AND served from local disk (no googlevideo round-trip / proxy on playback) →
# sub-second starts. The cold remux (download itag-18 + ffmpeg -c:a copy) is a
# few seconds, hidden by the mobile pre-warm (media_play → /audio_url) during the
# agent turn / current track. (We do NOT source the audio-only itag-140: from
# the default client its `n` challenge is unsolved so YouTube throttles it to
# ~32 KB/s — a full pull is ~111s.) Fail-open: any ffmpeg/extraction/remux miss
# → /audio_stream falls back to the itag-18 proxy (today's behaviour), never a
# regression. Mobile-only: web plays via the YouTube iframe, never /audio_stream.
_AUDIO_CACHE_DIR = os.environ.get("AUDIO_REMUX_CACHE_DIR", "/tmp/toup_audio_remux")
_AUDIO_CACHE_MAX_FILES = 80
# In-flight remux builds keyed by video_id, so the prewarm (/audio_url), a
# concurrent /audio_stream miss, AND AVFoundation's parallel connections all
# await ONE build instead of each downloading+ffmpeg'ing the same track.
_remux_tasks: dict[str, "asyncio.Task[str | None]"] = {}
_remux_inflight_lock = asyncio.Lock()
# How long an /audio_stream MISS waits for the build before falling open to the
# itag-18 proxy. Builds are ~2-3s (itag-18 download over the fast server link +
# ffmpeg -c:a copy); the prewarm usually finishes it earlier. 15s covers long
# tracks without hanging the request.
_REMUX_SYNC_BUDGET_SECS = float(os.environ.get("AUDIO_REMUX_SYNC_BUDGET_SECS", "15"))
# Cap concurrent remux BUILDS. Each build pulls ~11.8MB of itag-18 through the
# residential proxy (the user's Mac), whose uplink is the bottleneck. The mobile
# prefetch window fires several builds at once; left unbounded they saturate the
# uplink → the song the user is actually listening to stalls (PlaybackError 4 /
# silence) and every build crawls. Bounding to 2 keeps the uplink available for
# live playback while still warming the window. Created lazily so it binds to the
# running loop.
_REMUX_BUILD_CONCURRENCY = int(os.environ.get("AUDIO_REMUX_BUILD_CONCURRENCY", "2"))
_build_sem: "asyncio.Semaphore | None" = None


def _get_build_sem() -> "asyncio.Semaphore":
    global _build_sem
    if _build_sem is None:
        _build_sem = asyncio.Semaphore(_REMUX_BUILD_CONCURRENCY)
    return _build_sem


# ── Live-first gate: a build must never race the sound someone is waiting for ─
#
# `_REMUX_BUILD_CONCURRENCY` caps how MANY builds run. It says nothing about
# WHEN, and two builds are two thirds of the proxy's uplink whether or not a
# user is sitting in silence waiting for a first play. On 2026-08-05 that is
# exactly what was happening: `ws_chat` warms `upcoming[:2]` in
# `warm_audio_cache`'s DEFAULT `build` mode on the `_auto` toggle, which fires
# seconds into the very play whose first 2.5MB is still arriving.
#
# Measured the same day against the production proxy session: 0.63 MB/s of
# uplink, consistent across five samples. Alone, iOS's ~2.5MB pre-roll takes
# ~4.0s. Split three ways with two builds it takes ~12s — which is the founder
# report this gate exists to answer.
#
# This is the THIRD appearance of one bug. `get_audio_url` carries
# "DELIBERATELY NO REMUX BUILD HERE" and `stream_audio` defers its own warm to
# the response body's `finally` "now that this response is no longer competing";
# both fixed it at a call site, and a new call site reintroduced it. Fixing it
# at the BUILDER means a fourth trigger cannot.
#
# The gate is held only while someone is WAITING TO HEAR SOMETHING, not for a
# whole track. A progressive itag-18 response keeps pumping long after playback
# starts, so gating on "a stream is open" would block builds for ~30s per track
# and starve the prefetch that makes skip instant. It releases as soon as the
# client has enough bytes to be making sound.
_LIVE_GATE_BYTES = int(os.environ.get("AUDIO_LIVE_GATE_BYTES", "3000000"))
# A client that stalls without disconnecting must not starve the cache forever.
# Timing out and building anyway is just today's behaviour, which is merely slow.
_BUILD_YIELD_TIMEOUT = float(os.environ.get("AUDIO_BUILD_YIELD_TIMEOUT", "45"))
_live_starting = 0
_live_idle: "asyncio.Event | None" = None


def _get_live_idle() -> "asyncio.Event":
    global _live_idle
    if _live_idle is None:
        _live_idle = asyncio.Event()
        _live_idle.set()
    return _live_idle


def _live_start_begin() -> None:
    global _live_starting
    _live_starting += 1
    _get_live_idle().clear()


def _live_start_done() -> None:
    """Idempotent per stream — the caller guards with its own flag.

    Clamped at zero because an unbalanced decrement would set the event while a
    real cold start was still in progress, silently disabling the gate.
    """
    global _live_starting
    _live_starting = max(0, _live_starting - 1)
    if _live_starting == 0:
        _get_live_idle().set()


async def _await_live_idle(video_id: str) -> None:
    ev = _get_live_idle()
    if ev.is_set():
        return
    t0 = time.monotonic()
    try:
        await asyncio.wait_for(ev.wait(), timeout=_BUILD_YIELD_TIMEOUT)
        logger.info(
            "[media_proxy] build yielded %.1fs to a live cold start video_id=%s",
            time.monotonic() - t0, video_id,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "[media_proxy] build gate timed out after %.0fs — building anyway video_id=%s",
            _BUILD_YIELD_TIMEOUT, video_id,
        )


async def _bounded_build(video_id: str) -> str | None:
    """Run _do_remux under the global build-concurrency cap so concurrent prefetch
    builds can't saturate the residential proxy uplink and starve live playback.
    On success, persist the result into the SHARED R2 cache so other replicas
    (and future restarts) serve it without re-fetching through the proxy."""
    await _await_live_idle(video_id)
    async with _get_build_sem():
        # Re-check: this build may have queued behind the concurrency cap for
        # long enough that a new cold start began while it waited. Checking only
        # before the semaphore would let a build acquire a slot during a quiet
        # moment and then pull 11.8MB straight through somebody's first play.
        await _await_live_idle(video_id)
        # Spool-fed when possible: the build reads the (shared, possibly still
        # filling) upstream download instead of opening its own. Slot occupancy
        # is unchanged — the legacy path held this slot for the same download.
        src_path = await _spool_file_for_build(video_id)
        path = await asyncio.get_event_loop().run_in_executor(
            None, _do_remux, video_id, src_path
        )
        # The spool is deliberately NOT discarded on publish. A playing item
        # that started on the spool has the itag-18 total baked into its
        # AVPlayer state; the tier-pinning block in stream_audio keeps serving
        # it from the spool, and only a FRESH item start adopts the m4a (and
        # retires the spool then). MAX_AGE eviction + the prune sweep bound
        # the leftover.
    if path:
        # R2 is the durable, uncapped, Mac-independent store — and now the ONLY
        # one. The Postgres L2 that used to be written here has been retired:
        # it existed to give replicas a shared cache before R2, and R2 does that
        # job without putting multi-megabyte blobs in a relational database.
        # Verified before removal (2026-08-04): all 300 PG rows were present in
        # R2, so nothing was lost. See the note above _r2_pull_to_local.
        await _r2_store_from_local(video_id, path)
    return path


# ── Shared remux cache (Cloudflare R2 L2) ─────────────────────────────────
# Same role as the Postgres L2 above, but stores the finished audio-only m4a in
# Cloudflare R2 (S3-compatible, $0 egress, uncapped). THIS is what lets repeat
# playback serve WITHOUT the user's Mac/residential proxy and removes the
# 300-row Postgres cap — a track built once plays forever, Mac off. boto3 is
# sync, so wrap calls in an executor (mirrors services/aws_service.py). Fully
# fail-open + flag-gated: a no-op until r2_audio_enabled + the four R2_* creds
# are set, and every error returns None / no-ops so playback degrades to the PG
# L2 then the itag-18 proxy. We serve by pulling the object to local /tmp and
# reusing the PROVEN _local_audio_response Range/206 path — NOT a redirect —
# so iOS lock-screen/background AVPlayer behaviour is byte-identical to today.
_R2_OP_TIMEOUT = float(os.environ.get("AUDIO_R2_OP_TIMEOUT", "8"))
_r2_client = None
_r2_disabled = False


def _r2_ready() -> bool:
    return bool(
        settings.r2_audio_enabled
        and settings.r2_account_id
        and settings.r2_bucket
        and settings.r2_access_key_id
        and settings.r2_secret_access_key
    )


def _get_r2_client():
    """Lazily build a boto3 S3 client pointed at the R2 endpoint. None if R2 is
    unconfigured or client init fails (caller then falls back to the PG L2)."""
    global _r2_client, _r2_disabled
    if _r2_disabled or not _r2_ready():
        return None
    if _r2_client is not None:
        return _r2_client
    try:
        import boto3
        from botocore.config import Config
        _r2_client = boto3.client(
            "s3",
            endpoint_url=f"https://{settings.r2_account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=settings.r2_access_key_id,
            aws_secret_access_key=settings.r2_secret_access_key,
            region_name="auto",
            config=Config(signature_version="s3v4", retries={"max_attempts": 2, "mode": "standard"}),
        )
        logger.info("[media_proxy] R2 audio cache ENABLED bucket=%s", settings.r2_bucket)
        return _r2_client
    except Exception as e:
        logger.warning("[media_proxy] R2 client init failed — disabling R2: %s", e)
        _r2_disabled = True
        return None


def _r2_key(video_id: str) -> str:
    return f"{video_id}.m4a"


async def _r2_pull_to_local(video_id: str) -> str | None:
    """If the remux is in R2, download it to local disk and return the path.
    None on miss/error/timeout → caller tries the PG L2, then builds."""
    client = _get_r2_client()
    if client is None or not _ensure_cache_dir():
        return None
    out = _remuxed_path(video_id)
    tmp = f"{out}.{os.getpid()}.r2"

    def _do() -> str | None:
        try:
            client.download_file(settings.r2_bucket, _r2_key(video_id), tmp)
        except Exception:
            return None  # miss (404) or transfer error → fall back to PG L2
        try:
            if os.path.getsize(tmp) <= 0:
                os.remove(tmp)
                return None
        except OSError:
            return None
        os.replace(tmp, out)
        return out

    try:
        path = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(None, _do), timeout=_R2_OP_TIMEOUT
        )
        if path:
            logger.info("[media_proxy] R2 cache HIT video_id=%s", video_id)
        return path
    except Exception as e:
        logger.warning("[media_proxy] R2 pull failed video_id=%s: %s", video_id, e)
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        return None


async def _r2_store_from_local(video_id: str, path: str) -> None:
    """Upload a freshly-built remux to R2 (best-effort, fail-open). Long immutable
    Cache-Control so any future CDN fronting can cache it forever."""
    client = _get_r2_client()
    if client is None:
        return

    def _do() -> None:
        client.upload_file(
            path, settings.r2_bucket, _r2_key(video_id),
            ExtraArgs={
                "ContentType": "audio/mp4",
                "CacheControl": "public, max-age=31536000, immutable",
            },
        )

    try:
        await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(None, _do), timeout=_R2_OP_TIMEOUT * 2
        )
        logger.info("[media_proxy] R2 cache STORE video_id=%s", video_id)
    except Exception as e:
        logger.warning("[media_proxy] R2 store failed video_id=%s: %s", video_id, e)


_FFMPEG_OK: bool | None = None


def _ffmpeg_available() -> bool:
    global _FFMPEG_OK
    if _FFMPEG_OK is None:
        _FFMPEG_OK = shutil.which("ffmpeg") is not None
        if not _FFMPEG_OK:
            logger.warning("[media_proxy] ffmpeg not found — audio remux disabled (itag-18 proxy only)")
    return _FFMPEG_OK


def _remuxed_path(video_id: str) -> str:
    return os.path.join(_AUDIO_CACHE_DIR, f"{video_id}.m4a")


def _remuxed_ready(video_id: str) -> str | None:
    p = _remuxed_path(video_id)
    try:
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return p
    except OSError:
        pass
    return None


def _safe_unlink(p: str) -> None:
    try:
        os.remove(p)
    except OSError:
        pass


def _ensure_cache_dir() -> bool:
    """Create the remux cache dir world-writable and confirm THIS process can
    write it. Returns False (→ caller fails open to the itag-18 proxy) when it
    can't — the usual cause is the dir pre-created by a different uid: a root
    `railway ssh` repro run leaves it root-owned while the app runs as `toup`,
    so the app's open("wb") then trips Errno 13. Sticky world-writable (1777)
    lets either uid populate it without locking the other out."""
    d = _AUDIO_CACHE_DIR
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        return False
    try:
        os.chmod(d, 0o1777)  # best-effort; no-op if we don't own it
    except OSError:
        pass
    return os.access(d, os.W_OK | os.X_OK)


def _do_remux(video_id: str, src_path: str | None = None) -> str | None:
    """Blocking: extract the audio-only URL → ffmpeg remux to a progressive
    faststart m4a on local disk. Returns the cached path, or None so the caller
    falls back to the itag-18 proxy. Idempotent. Run in an executor.

    `src_path`: an already-downloaded itag-18 file (a finished spool). When
    given, the extract + download stages are skipped entirely — ffmpeg reads
    the local file and the build costs zero proxy bytes. The file is owned by
    the SPOOL, not this function: never unlink it here."""
    ready = _remuxed_ready(video_id)
    if ready:
        return ready
    if not _ffmpeg_available():
        return None
    if src_path is not None:
        try:
            if os.path.getsize(src_path) <= 0:
                src_path = None
        except OSError:
            src_path = None
    if src_path is not None:
        if not _ensure_cache_dir():
            return None
        out = _remuxed_path(video_id)
        tmp = f"{out}.{os.getpid()}.tmp"
        _safe_unlink(tmp)
        t0 = time.time()
        cmd = [
            "ffmpeg", "-y", "-nostdin", "-loglevel", "error",
            "-i", src_path,
            "-vn", "-c:a", "copy", "-movflags", "+faststart",
            "-f", "mp4", tmp,
        ]
        try:
            proc = subprocess.run(cmd, timeout=60, capture_output=True)
        except Exception as e:
            logger.warning("[media_proxy] remux(spool) ffmpeg error video_id=%s: %s", video_id, e)
            _safe_unlink(tmp)
            return None
        if proc.returncode != 0 or not (os.path.exists(tmp) and os.path.getsize(tmp) > 0):
            logger.warning(
                "[media_proxy] remux(spool) failed video_id=%s rc=%s err=%s",
                video_id, proc.returncode, (proc.stderr or b"")[:200],
            )
            _safe_unlink(tmp)
            return None
        try:
            os.replace(tmp, out)
        except OSError:
            _safe_unlink(tmp)
            return None
        logger.info(
            "[media_proxy] remux ok (spool) video_id=%s %.2fs size=%d",
            video_id, time.time() - t0, os.path.getsize(out),
        )
        _prune_audio_cache()
        return out
    # Source the UN-THROTTLED itag-18 (android/ios client + POT, ~2-5 MB/s) via
    # the same extraction the proxy already uses — then ffmpeg strips the video.
    # We do NOT use the audio-only itag-140: from the default client it has no
    # solved `n` challenge so YouTube throttles it to ~32 KB/s (a full pull is
    # ~111s). itag-18 is bigger but downloads in seconds, so the remux is fast.
    result = _cached_extract(video_id)
    if "error" in result or not result.get("url"):
        logger.warning("[media_proxy] remux: extract miss for video_id=%s", video_id)
        return None
    url = result["url"]
    if not _ensure_cache_dir():
        logger.warning(
            "[media_proxy] remux: cache dir not writable (%s) — itag-18 proxy only",
            _AUDIO_CACHE_DIR,
        )
        return None
    out = _remuxed_path(video_id)
    pid = os.getpid()
    src = f"{out}.{pid}.src"   # downloaded itag-18 (progressive video+audio mp4)
    tmp = f"{out}.{pid}.tmp"   # remuxed progressive audio-only output
    # Clear any stale temp from a crashed prior run so open("wb") never trips on
    # a leftover we can't overwrite (best-effort).
    _safe_unlink(src)
    _safe_unlink(tmp)
    t0 = time.time()
    # Download the audio through the SAME proxy that signed the URL (googlevideo
    # binds the URL to that egress IP). httpx-through-proxy is the proven path
    # (same client /audio_stream uses); decoupling the fetch from ffmpeg avoids
    # ffmpeg's finicky https-proxy handling.
    try:
        with httpx.Client(
            proxy=_proxy_for_result(result), timeout=60.0, follow_redirects=True
        ) as dl:
            with dl.stream("GET", url, headers={"User-Agent": "Mozilla/5.0"}) as r:
                if r.status_code >= 400:
                    logger.warning("[media_proxy] remux download status=%s video_id=%s", r.status_code, video_id)
                    return None
                with open(src, "wb") as f:
                    for chunk in r.iter_bytes(65536):
                        f.write(chunk)
    except Exception as e:
        logger.warning("[media_proxy] remux download error video_id=%s: %s", video_id, e)
        _safe_unlink(src)
        return None
    # Remux the LOCAL file to a progressive faststart m4a (moov at front, no
    # video). -c:a copy = no re-encode, fast. No proxy needed (local input).
    cmd = [
        "ffmpeg", "-y", "-nostdin", "-loglevel", "error",
        "-i", src,
        "-vn", "-c:a", "copy", "-movflags", "+faststart",
        "-f", "mp4", tmp,
    ]
    try:
        proc = subprocess.run(cmd, timeout=60, capture_output=True)
    except Exception as e:
        logger.warning("[media_proxy] remux ffmpeg error video_id=%s: %s", video_id, e)
        _safe_unlink(src)
        _safe_unlink(tmp)
        return None
    _safe_unlink(src)
    if proc.returncode != 0 or not (os.path.exists(tmp) and os.path.getsize(tmp) > 0):
        logger.warning(
            "[media_proxy] remux failed video_id=%s rc=%s err=%s",
            video_id, proc.returncode, (proc.stderr or b"")[:200],
        )
        _safe_unlink(tmp)
        return None
    try:
        os.replace(tmp, out)  # atomic publish
    except OSError:
        _safe_unlink(tmp)
        return None
    logger.info(
        "[media_proxy] remux ok video_id=%s %.2fs size=%d",
        video_id, time.time() - t0, os.path.getsize(out),
    )
    _prune_audio_cache()
    return out


def _prune_audio_cache() -> None:
    """Keep the newest _AUDIO_CACHE_MAX_FILES remuxed files; drop older ones.
    Also sweep dead .spool files — a spool whose mtime is an hour old stopped
    being written long ago (crash, restart, abandoned download)."""
    try:
        listing = os.listdir(_AUDIO_CACHE_DIR)
    except OSError:
        return
    now = time.time()
    live_spool_paths = {sp.path for sp in _spools.values()}
    for f in listing:
        if f.endswith(".spool"):
            p = os.path.join(_AUDIO_CACHE_DIR, f)
            if p in live_spool_paths:
                continue  # a registered spool is never swept, whatever its mtime
            try:
                if now - os.path.getmtime(p) > 3600:
                    _safe_unlink(p)
            except OSError:
                pass
    entries = [os.path.join(_AUDIO_CACHE_DIR, f) for f in listing if f.endswith(".m4a")]
    if len(entries) <= _AUDIO_CACHE_MAX_FILES:
        return
    try:
        entries.sort(key=lambda p: os.path.getmtime(p))
    except OSError:
        return
    for p in entries[: len(entries) - _AUDIO_CACHE_MAX_FILES]:
        _safe_unlink(p)


async def _remux_now(video_id: str, budget: float) -> str | None:
    """Build (or join an in-flight build of) the audio-only remux and return its
    local path — or None if ffmpeg is unavailable, the build failed, or it didn't
    finish within `budget`. Concurrent callers for the same video share ONE build
    task (deduped under the lock), so the prewarm, the /audio_stream miss, and
    AVFoundation's parallel connections never trigger duplicate downloads. A
    timeout does NOT cancel the build (asyncio.shield) — it keeps running so the
    very next request HITs the finished file."""
    ready = _remuxed_ready(video_id)
    if ready:
        return ready
    if not video_id or not _ffmpeg_available():
        return None
    # Shared store: another replica may have already built this — pull it from
    # R2 (no proxy fetch, no rebuild). This used to read the Postgres L2 and
    # never consult R2 at all, so a track present in R2 but evicted from PG's
    # 300-row cap was rebuilt through the residential proxy for nothing.
    pulled = await _r2_pull_to_local(video_id)
    if pulled:
        return pulled
    async with _remux_inflight_lock:
        # Drop finished tasks so the map stays small over a long session.
        for k in [k for k, t in _remux_tasks.items() if t.done()]:
            _remux_tasks.pop(k, None)
        task = _remux_tasks.get(video_id)
        if task is None:
            task = asyncio.ensure_future(_bounded_build(video_id))
            _remux_tasks[video_id] = task
    try:
        return await asyncio.wait_for(asyncio.shield(task), timeout=budget)
    except asyncio.TimeoutError:
        return None
    except Exception:
        return None


async def _ensure_remux_bg(video_id: str) -> None:
    """Fire-and-forget background pre-warm (from /audio_url at media_play). Builds
    with a generous budget so the remux is ready before the play/skip asks."""
    try:
        await _remux_now(video_id, budget=120.0)
    except Exception:
        pass


async def _ensure_extract_bg(video_id: str) -> None:
    """Fire-and-forget yt-dlp EXTRACTION only — no media bytes.

    This is the cheap half of warming, and separating it from `_ensure_remux_bg`
    is the point. A build downloads the whole ~11.8MB itag-18 through the one
    residential proxy, so it can only ever be aimed at UPCOMING tracks: aimed at
    the track playing right now it competes with the very stream the user is
    waiting to hear. An extraction is a metadata handshake — a webpage fetch and
    a player-API JSON, a few hundred KB — so it can safely be aimed at the
    current track, and that is exactly where the time is.

    Measured against the production proxy on 2026-08-04 (18 samples, 3 videos):
    extraction alone is a median 3.4s, a mean 7.1s, and a worst case of 20.8s.
    That whole cost sits in front of the first byte of audio. Run at broadcast
    time it happens while the agent is still composing its reply and the phone
    is still rendering the card, so by the time `/audio_stream` asks,
    `_extract_coalesced` either hits the cache or joins a call already in
    flight. The user's wait collapses to the buffer fill.

    Hiding the VARIANCE matters as much as the median here — a 20s extraction is
    indistinguishable from a broken player, and it is the same fixed cost
    whether it is paid in front of the user or behind the agent's reply.
    """
    try:
        await _extract_coalesced(video_id)
    except Exception:
        # A failed pre-extract must be invisible: the phone's own request will
        # retry it on the normal path and surface a real error there.
        pass


@router.get("/{video_id}/audio_url")
async def get_audio_url(
    video_id: str,
    current_user=Depends(get_current_user),
):
    """Return an audio-only stream URL for mobile background playback.

    The mobile hybrid player hands off from the YouTube iframe (foreground
    video) to react-native-track-player (background audio) on app
    backgrounding. This endpoint resolves the audio URL yt-dlp exposes for
    a given YouTube videoId plus enough metadata to set up
    `MPNowPlayingInfoCenter` and know when to re-fetch the URL.

    Auth: bearer-gated. An unauthenticated public YouTube audio extractor
    on toup.ai would be scraped, burn CPU on yt-dlp calls, and frame the
    feature as a ripping tool rather than an in-app player. The sibling
    legacy `/media/stream/{video_id}` route is currently unauth (cast-to-TV
    flow predates this hardening); locking it down is a separate follow-up,
    not a precedent to copy.

    Error codes:
      - 401: missing / invalid / expired bearer token
      - 502: yt-dlp extraction failed (private / removed / region-blocked
             video, upstream yt-dlp breakage, etc.)
      - 504: extraction didn't complete within the timeout budget
    """
    try:
        result = await asyncio.wait_for(
            _extract_coalesced(video_id),
            timeout=_AUDIO_EXTRACT_TIMEOUT_SECS,
        )
    except asyncio.TimeoutError:
        logger.warning("[media_proxy] audio_url timeout video_id=%s", video_id)
        return JSONResponse(
            status_code=504,
            content={"error": "extraction_timeout"},
        )
    except Exception as e:
        logger.warning(
            "[media_proxy] audio_url failed video_id=%s err=%s: %s",
            video_id, type(e).__name__, e,
        )
        return JSONResponse(
            status_code=502,
            content={"error": "extraction_failed", "detail": str(e)},
        )

    if "error" in result:
        return JSONResponse(status_code=502, content=result)
    # DELIBERATELY NO REMUX BUILD HERE. This used to fire
    # `create_task(_ensure_remux_bg(video_id))`, and on the mobile play path
    # that is a build of the very track the phone is about to stream: the app
    # calls /audio_url on the `media_play` frame and starts /audio_stream a
    # moment later (ChatScreen.prewarmAudioUrl). The build pulls the whole
    # itag-18 through the residential proxy while the live stream pulls the
    # SAME bytes through the SAME proxy — the two halve each other's share of
    # a link that is the hard bottleneck (measured 0.45-0.76 MB/s on
    # 2026-08-03, against a 1.2-2.7 MB/s healthy baseline). iOS needs ~2.5MB
    # buffered before it makes a sound, so the duplicate turned a ~5s cold
    # start into 10s+.
    #
    # The stream path already fixed this at its own trigger by warming in the
    # response's `finally` ("no longer competing…", below); this was the same
    # bug at the other trigger, firing BEFORE playback instead of during it.
    #
    # Nothing stops being cached. Every track that actually plays is warmed by
    # that post-stream hook, and upcoming tracks are built by the phone's
    # `/audio_stream?prefetch=1` requests, which build-and-wait by design
    # (trackPlayer.prefetchAudioToDisk). A track that is prewarmed but never
    # played simply is not built — which is the correct trade for a bottleneck
    # this tight.
    return result


# ── Audio stream PROXY for native background playback ─────────────────────
#
# Why this exists: `/audio_url` returns a raw googlevideo.com URL that
# yt-dlp signs to the SERVER's egress IP. The phone fetching it directly
# gets HTTP 403 (signature is for a different IP), which is why the mobile
# hybrid player's foreground→background handoff to react-native-track-player
# was disabled. This route fetches those bytes server-side (our IP matches
# the signature) and pipes them to the device, so the native player can
# play audio that survives app-background / screen-lock.
#
# Range requests are forwarded so AVFoundation can seek + progressively
# download. Auth is bearer (TrackPlayer sends Authorization via per-track
# headers — same mobile JWT as every other call).

_STREAM_CHUNK_BYTES = 64 * 1024
_STREAM_TIMEOUT = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)
# Ceiling on getting RESPONSE HEADERS back, separate from `read` above, which
# has to stay long enough to cover a slow progressive body. See the call site.
_UPSTREAM_HEADERS_TIMEOUT = float(os.environ.get("AUDIO_UPSTREAM_HEADERS_TIMEOUT", "20"))
# Mid-body stall bound. When a response has ALREADY sent bytes and the
# upstream then goes quiet, holding the client open is the worst option:
# AVPlayer drains its ~2s buffer, the position freezes, the UI still says
# "playing", and NOTHING errors for the full 60s read timeout — the recorded
# "frozen at 0:02 under an animating equalizer". Cutting the response after a
# few silent seconds turns an invisible hang into an immediate client-side
# error the phone's recovery ladder can act on. This bounds the wait for the
# NEXT chunk mid-body, never time-to-first-byte (extraction/connect have their
# own budgets) and never client backpressure (a paused player blocks on
# `yield`, not on this wait).
_MID_STREAM_STALL_SECS = float(os.environ.get("AUDIO_MID_STREAM_STALL_SECS", "8"))


def _parse_byte_range(range_header: str | None, file_size: int):
    """Parse a single HTTP byte-range request against a known file size.

    Returns an inclusive (start, end) tuple, or None when the header is
    absent / malformed / multi-range / unsatisfiable — in which case the caller
    serves a plain 200 with the full body. AVFoundation only ever sends a single
    `bytes=START-` or `bytes=START-END` (and an initial `bytes=0-1` probe)."""
    if not range_header or file_size <= 0:
        return None
    m = re.match(r"\s*bytes\s*=\s*(\d*)\s*-\s*(\d*)\s*$", range_header)
    if not m:
        return None  # multi-range or garbage → full file
    s, e = m.group(1), m.group(2)
    if s == "" and e == "":
        return None
    if s == "":
        # suffix form: last N bytes
        n = int(e)
        if n <= 0:
            return None
        start, end = max(0, file_size - n), file_size - 1
    else:
        start = int(s)
        end = int(e) if e != "" else file_size - 1
    if start >= file_size or start > end:
        return None
    return start, min(end, file_size - 1)


def _local_audio_response(rpath: str, range_header: str | None,
                          if_range: str | None = None):
    """Range-aware (206) / full (200) StreamingResponse for a cached local audio
    file, or None if the file vanished / is empty (caller falls through). A
    correct Content-Length is what stops AVFoundation re-requesting the whole
    file in a loop. Hand-rolled because Starlette 0.35.1's FileResponse ignores
    Range (returns 200 + whole file) while we advertise Accept-Ranges → seeks
    land on the wrong bytes → PlaybackError 1."""
    try:
        file_size = os.path.getsize(rpath)
    except OSError:
        file_size = 0
    if file_size <= 0:
        return None
    # Entity identity. The same URL serves DIFFERENT entities over time (the
    # ~11.8MB progressive itag-18 while cold, the ~3MB remuxed m4a once
    # built), and a client resuming with byte offsets from the OLD entity gets
    # silently wrong bytes whenever its offset happens to fit inside the new
    # file (the size-only 416 below can't see that case — review finding). A
    # strong ETag + If-Range is the HTTP answer: a client that validates gets
    # a full 200 instead of a mismatched slice.
    etag = f'"{file_size}-{int(os.path.getmtime(rpath)) if os.path.exists(rpath) else 0}"'
    base = {"Accept-Ranges": "bytes", "Cache-Control": "no-store", "ETag": etag}
    if if_range and if_range.strip() != etag:
        # RFC 7233 §3.2: an If-Range that doesn't match ⇒ ignore Range, serve
        # the full current entity.
        range_header = None
    # A range starting AT or BEYOND this file's size means the client is
    # resuming against a DIFFERENT entity — the ~11.8MB progressive itag-18 it
    # buffered before an idle, now answered by the ~3MB remuxed m4a after the
    # spool retired. Serving a silent 200 of the whole other container (the
    # old behaviour, via _parse_byte_range's None) restarts the track at 0:00
    # with CoreMedia none the wiser. 416 is the honest answer: the client
    # knowingly re-fetches from byte 0 instead of being lied to.
    _m = re.match(r"\s*bytes\s*=\s*(\d+)\s*-", range_header or "")
    if _m and int(_m.group(1)) >= file_size:
        from fastapi import Response
        return Response(
            status_code=416,
            headers={**base, "Content-Range": f"bytes */{file_size}"},
        )
    rng = _parse_byte_range(range_header, file_size)
    if rng:
        start, end = rng
        length = end - start + 1

        def _slice():
            with open(rpath, "rb") as fh:
                fh.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = fh.read(min(_STREAM_CHUNK_BYTES, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk

        return StreamingResponse(
            _slice(),
            status_code=206,
            media_type="audio/mp4",
            headers={
                **base,
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Content-Length": str(length),
            },
        )

    def _full():
        with open(rpath, "rb") as fh:
            while True:
                chunk = fh.read(_STREAM_CHUNK_BYTES)
                if not chunk:
                    break
                yield chunk

    return StreamingResponse(
        _full(),
        status_code=200,
        media_type="audio/mp4",
        headers={**base, "Content-Length": str(file_size)},
    )

# Per-tenant in-process concurrency gate. Effective global cap is
# N × replicas until a shared (Redis) limiter lands — see config.py
# `audio_stream_max_concurrent_per_tenant`. Bounds how many full-song
# proxy streams a single account can hold open server-side at once.
_audio_stream_sems: dict[str, asyncio.Semaphore] = {}


def _tenant_stream_sem(user_id: str) -> asyncio.Semaphore:
    sem = _audio_stream_sems.get(user_id)
    if sem is None:
        sem = asyncio.Semaphore(settings.audio_stream_max_concurrent_per_tenant)
        _audio_stream_sems[user_id] = sem
    return sem


# ── Single-flight upstream SPOOL ───────────────────────────────────────────
#
# On a cold MISS, one video used to cost the residential proxy the same bytes
# THREE times: AVFoundation's 2-byte probe opened a full upstream connection
# (4.2s of TTFB for 2 bytes, measured in prod 2026-08-09), its real range
# request opened another (upstream_ms=4239 on the same track, same minute),
# and the post-response remux build downloaded the whole itag-18 AGAIN —
# ~23.6MB of metered traffic for an ~11.8MB file, all through a peer measured
# at 0.04-1.14 MB/s while the user sat in silence.
#
# The spool is ONE download per video id: an async task pulls the itag-18 to a
# local file, and every consumer reads from that file as it fills — the probe
# (answered the moment 2 bytes exist), every parallel range connection, and
# the remux build (which no longer downloads at all; ffmpeg reads the spool).
# A warm(mode="build") issued at broadcast time starts the spool BEFORE the
# phone asks, so the phone's first request joins a download already in flight.
#
# Fail-open by construction: any spool problem (download error, header-wait
# timeout, a deep seek beyond what has arrived) returns None and the caller
# falls through to the legacy direct-upstream path unchanged. Kill switch:
# AUDIO_STREAM_SPOOL=0 restores the legacy path outright.
_SPOOL_ENABLED = os.environ.get("AUDIO_STREAM_SPOOL", "1").strip().lower() not in (
    "0", "false", "no", "off",
)
# Bounded wait for the spool download's response HEADERS (Content-Length) —
# mirrors _UPSTREAM_HEADERS_TIMEOUT plus scheduling slack.
_SPOOL_HEADERS_WAIT = float(os.environ.get("AUDIO_SPOOL_HEADERS_WAIT", "25"))
# How long a build waits for a spool to finish before falling back to its own
# download. A 13-minute video is ~38MB; at the measured worst residential
# throughput that is minutes, so this is generous — the build budget
# (_ensure_remux_bg's 120s) is the real ceiling.
_SPOOL_BUILD_WAIT = float(os.environ.get("AUDIO_SPOOL_BUILD_WAIT", "110"))
# Serve a range from the spool only when its start is within this many bytes
# of what has already arrived (or the download is done). A deep seek would
# otherwise WAIT for the sequential download to reach it — the legacy path
# serves that one request with its own upstream Range instead.
_SPOOL_READ_AHEAD_LIMIT = 3 * 1024 * 1024
# A spool this old is a leak, not a download.
_SPOOL_MAX_AGE_SECS = float(os.environ.get("AUDIO_SPOOL_MAX_AGE_SECS", "900"))


class _Spool:
    """State of one in-flight (or completed) upstream download.

    All mutation happens on the event loop (the downloader task), so plain
    attributes are safe; readers poll rather than wait on a Condition —
    50ms granularity is noise against multi-second network stages, and it
    sidesteps asyncio.Condition's cancellation sharp edges entirely.
    """

    __slots__ = ("video_id", "path", "total", "received", "done", "error", "task", "created", "mime")

    def __init__(self, video_id: str, path: str) -> None:
        self.video_id = video_id
        self.path = path
        self.total: int | None = None
        self.received = 0
        self.done = False
        self.error: str | None = None
        self.task: "asyncio.Task | None" = None
        self.created = time.monotonic()
        self.mime = "audio/mp4"


_spools: dict[str, _Spool] = {}
_spools_lock = asyncio.Lock()


async def _spool_wait(spool: _Spool, predicate, timeout: float) -> bool:
    """Poll until predicate() holds. False on error/timeout — never raises."""
    deadline = time.monotonic() + timeout
    while True:
        if predicate():
            return True
        if spool.error is not None:
            return False
        if time.monotonic() >= deadline:
            return False
        await asyncio.sleep(0.05)


async def _spool_download(spool: _Spool, result: dict) -> None:
    """The one upstream pull. Writes the growing file, then kicks the remux —
    which reads the file instead of re-downloading it."""
    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(
            timeout=_STREAM_TIMEOUT, follow_redirects=True, proxy=_proxy_for_result(result)
        ) as client:
            async with client.stream(
                "GET", result["url"], headers={"User-Agent": "Mozilla/5.0"}
            ) as r:
                if r.status_code >= 400:
                    raise RuntimeError(f"upstream status {r.status_code}")
                try:
                    spool.total = int(r.headers.get("content-length") or 0) or None
                except ValueError:
                    spool.total = None
                with open(spool.path, "wb") as f:
                    async for chunk in r.aiter_bytes(_STREAM_CHUNK_BYTES):
                        f.write(chunk)
                        spool.received += len(chunk)
        if spool.total is None:
            spool.total = spool.received
        if spool.received != spool.total:
            raise RuntimeError(f"short body {spool.received}/{spool.total}")
        spool.done = True
        logger.info(
            "[media_proxy] spool COMPLETE video_id=%s bytes=%d %.1fs",
            spool.video_id, spool.received, time.monotonic() - t0,
        )
        # Build the remux NOW from the finished file — zero additional proxy
        # bytes. Deduped downstream (_remux_tasks / _remuxed_ready).
        asyncio.create_task(_ensure_remux_bg(spool.video_id))
    except Exception as e:
        spool.error = str(e)
        logger.warning("[media_proxy] spool FAILED video_id=%s: %s", spool.video_id, e)
        async with _spools_lock:
            if _spools.get(spool.video_id) is spool:
                _spools.pop(spool.video_id, None)
                # Unlink ONLY when this task still owned the registry entry.
                # A replaced spool (MAX_AGE eviction swapped in a successor at
                # the SAME deterministic path) must never touch the file — it
                # is the successor's live download now, and a stale unlink
                # here left every later reader open()ing a vanished path after
                # 206 headers had already been sent (adversarial review).
                _safe_unlink(spool.path)


async def _spool_get_or_start(video_id: str, result: dict) -> "_Spool | None":
    """Join the live spool for this video, or start one. None = spooling is
    unavailable (dir not writable) and the caller must use the legacy path."""
    if not _SPOOL_ENABLED:
        return None
    async with _spools_lock:
        sp = _spools.get(video_id)
        if sp is not None:
            if sp.error is None and (time.monotonic() - sp.created) < _SPOOL_MAX_AGE_SECS:
                return sp
            _spools.pop(video_id, None)
            # Mark BEFORE cancel/unlink: any reader mid-body on the evicted
            # object polls (received / error / done) — without an error it
            # would sleep-loop forever on a frozen download.
            if sp.error is None and not sp.done:
                sp.error = "evicted"
            # Cancel the evicted download BEFORE unlinking: left running, its
            # failure handler would fire later — its identity check now fails
            # (we popped it), so it won't unlink, but it must also stop
            # trickling metered bytes into an unlinked inode. (CancelledError
            # is a BaseException — the downloader's except Exception does not
            # catch it, so a cancelled task never runs the failure cleanup.)
            if sp.task is not None and not sp.task.done():
                sp.task.cancel()
            _safe_unlink(sp.path)
        if not _ensure_cache_dir():
            return None
        path = os.path.join(_AUDIO_CACHE_DIR, f"{video_id}.spool")
        try:
            # Created empty up front so readers can open it before the first
            # byte lands (the downloader truncates/rewrites the same handle).
            open(path, "wb").close()
        except OSError:
            return None
        sp = _Spool(video_id, path)
        sp.mime = result.get("mime_type") or "audio/mp4"
        _spools[video_id] = sp
        sp.task = asyncio.create_task(_spool_download(sp, result))
        return sp


def _spool_discard(video_id: str) -> None:
    """Drop a spool whose bytes are no longer needed (remux published)."""
    sp = _spools.pop(video_id, None)
    if sp is not None:
        _safe_unlink(sp.path)


async def _spool_file_for_build(video_id: str) -> str | None:
    """Hand the remux build a finished spool file so it never downloads.

    Joins a live spool, or STARTS one when none exists — that is what makes a
    warm(mode="build") at broadcast time pre-start the very download the
    phone's first /audio_stream request will read from. Starting a spool opens
    a new upstream pull, so the live-first gate is honored first, exactly like
    the legacy download this replaces. None = no usable spool (caller falls
    back to its own gated download)."""
    if not _SPOOL_ENABLED:
        return None
    sp = _spools.get(video_id)
    if sp is None or sp.error is not None:
        await _await_live_idle(video_id)
        try:
            result = await asyncio.wait_for(
                _extract_coalesced(video_id), timeout=_AUDIO_EXTRACT_TIMEOUT_SECS
            )
        except Exception:
            return None
        if "error" in result or not result.get("url"):
            return None
        sp = await _spool_get_or_start(video_id, result)
        if sp is None:
            return None
    ok = await _spool_wait(sp, lambda: sp.done, _SPOOL_BUILD_WAIT)
    if not ok or sp.error is not None:
        return None
    return sp.path


def _spool_response(
    spool: _Spool,
    range_header: "str | None",
    *,
    prefetch: bool,
    mime: str,
    video_id: str,
    on_release,
    t0: float,
    cache_ms: int,
    extract_ms: int,
) -> "StreamingResponse | None":
    """Range-aware response served from the spool file as it fills.

    None when this REQUEST can't be served from the spool (headers not yet
    known is handled by the caller's wait; a deep seek past the read-ahead
    limit falls back to legacy). The live-first gate semantics mirror the
    legacy body exactly: held from first byte until _LIVE_GATE_BYTES for a
    non-prefetch stream, released idempotently in finally."""
    total = spool.total
    if total is None or total <= 0:
        return None
    rng = _parse_byte_range(range_header, total)
    if rng:
        start, end = rng
        if not spool.done and start > spool.received + _SPOOL_READ_AHEAD_LIMIT:
            return None  # deep seek — don't make AVPlayer wait for sequential fill
        status = 206
        headers = {
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-store",
            "Content-Range": f"bytes {start}-{end}/{total}",
            "Content-Length": str(end - start + 1),
        }
    else:
        start, end = 0, total - 1
        status = 200
        headers = {
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-store",
            "Content-Length": str(total),
        }

    async def _spool_body():
        sent = 0
        pos = start
        t_first: "float | None" = None
        gating = not prefetch
        if gating:
            _live_start_begin()

        def _ungate() -> None:
            nonlocal gating
            if gating:
                gating = False
                _live_start_done()

        stalled_for = 0.0
        server_cut = False
        try:
            with open(spool.path, "rb") as fh:
                while pos <= end:
                    avail = spool.received
                    if pos < avail:
                        stalled_for = 0.0
                        fh.seek(pos)
                        chunk = fh.read(min(_STREAM_CHUNK_BYTES, end + 1 - pos, avail - pos))
                        if not chunk:
                            break
                        if t_first is None:
                            t_first = time.monotonic()
                            logger.info(
                                "[media_proxy] audio_stream TIMING video_id=%s tier=MISS "
                                "spool=1 cache_lookup_ms=%d extract_ms=%d first_byte_ms=%d",
                                video_id, cache_ms, extract_ms,
                                int((time.monotonic() - t0) * 1000),
                            )
                        pos += len(chunk)
                        sent += len(chunk)
                        if sent >= _LIVE_GATE_BYTES:
                            _ungate()
                        yield chunk
                    elif spool.error is not None:
                        break  # truncated — AVPlayer re-ranges; retry hits legacy
                    elif spool.done:
                        break
                    else:
                        # Mid-body stall watchdog. A quiet upstream must
                        # become a VISIBLE failure fast — held open, the phone
                        # shows a playing UI over dead audio for the
                        # downloader's full 60s read timeout. Applies from
                        # byte 0 too: after a stall cut, AVPlayer's recovery
                        # re-ranges at its buffered edge, which lands right
                        # back in this wait with sent==0 — an exemption there
                        # degraded the 8s bound straight back to 60s (review
                        # finding). Client backpressure never waits here (a
                        # paused player blocks on `yield`), so this bound only
                        # ever measures the upstream.
                        if stalled_for >= _MID_STREAM_STALL_SECS:
                            logger.warning(
                                "[media_proxy] audio_stream STALLED video_id=%s spool=1 "
                                "pos=%d/%d sent=%d silent_for=%.1fs — cutting the response",
                                video_id, pos, end + 1, sent, stalled_for,
                            )
                            server_cut = True
                            break
                        stalled_for += 0.05
                        await asyncio.sleep(0.05)
        finally:
            _ungate()
            if t_first is not None and sent > 0:
                _dur = max(time.monotonic() - t_first, 1e-3)
                logger.info(
                    "[media_proxy] audio_stream DELIVERED video_id=%s spool=1 bytes=%d "
                    "stream_ms=%d throughput_mbps=%.2f total_ms=%d",
                    video_id, sent, int(_dur * 1000), (sent / _dur) / 1e6,
                    int((time.monotonic() - t0) * 1000),
                )
            if (server_cut or spool.error is not None) and sent < (end - start + 1):
                # Distinct signature for a body WE ended short of its declared
                # Content-Length. Gated on a server-side cut: an ordinary
                # client disconnect also lands in this finally with
                # sent < promised, and logging that as TRUNCATED made the new
                # signature noise on day one (review finding).
                logger.warning(
                    "[media_proxy] audio_stream TRUNCATED video_id=%s spool=1 "
                    "sent=%d promised=%d spool_error=%r server_cut=%s",
                    video_id, sent, end - start + 1, spool.error, server_cut,
                )
            on_release()
            # If the spool died mid-serve, give the legacy build path its shot
            # at warming the cache (deduped; no-op when the remux exists).
            if spool.error is not None:
                asyncio.create_task(_ensure_remux_bg(video_id))

    return StreamingResponse(_spool_body(), status_code=status, headers=headers, media_type=mime)


# NOTE: do NOT cache+reuse one googlevideo URL across AVFoundation's multiple
# fetch connections — googlevideo rejects the repeat hits (observed as
# SwiftAudioEx PlaybackError 1 on the device, playback stuck at 0:00). Each
# request re-extracts a fresh signed URL, which plays reliably. The earlier
# "stutter" that motivated caching was actually the foreground AppState
# thrashing (fixed client-side: handoff only on real 'background'), not
# per-request extraction.
@router.get("/{video_id}/audio_stream")
async def stream_audio(
    video_id: str,
    request: Request,
    current_user=Depends(get_current_user),
):
    """Proxy a YouTube video's audio bytes through the server.

    See the module-level comment above for why a proxy (not a redirect) is
    required. Returns 200 (full) or 206 (range) with the audio bytes;
    429 if the tenant is over its concurrent-stream cap; 502/504 on
    extraction or upstream failure.
    """
    user_id = str(getattr(current_user, "id", None) or current_user)
    sem = _tenant_stream_sem(user_id)
    # Non-blocking acquire: an over-cap client fails fast instead of
    # queueing behind a worker while holding an upstream connection.
    try:
        await asyncio.wait_for(sem.acquire(), timeout=0.05)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=429, detail="too_many_concurrent_streams")

    # The semaphore is now held. It MUST be released on every exit path:
    # extraction failure, upstream failure, or — on the happy path — when
    # the streaming generator finishes / the client disconnects (finally).
    released = False

    def _release() -> None:
        nonlocal released
        if not released:
            released = True
            sem.release()

    # Per-stage timing. Time-to-first-audio is the number the user actually
    # feels, and until now nothing recorded where it went: a 12.5s cold start
    # (founder recording, 2026-08-03) and a 0.6s cached one logged the same
    # single line, so a delivery regression was indistinguishable from a cold
    # track and only a screen recording could tell them apart. Every exit path
    # below emits `audio_stream TIMING` with the tier that served it and the
    # milliseconds each stage cost. Cheap: one monotonic read per stage.
    _t0 = time.monotonic()

    def _ms(since: float) -> int:
        return int((time.monotonic() - since) * 1000)

    prefetch = (request.query_params.get("prefetch") or "").lower() in ("1", "true", "yes")
    range_header = request.headers.get("range")

    # ── TIER PINNING ──────────────────────────────────────────────────────
    # While a spool exists for this id, every non-prefetch request serves from
    # the itag-18 byte space — NEVER the m4a. The playing AVPlayer item learned
    # total = the itag-18 size from its probe; if the remux published mid-track
    # and a later range request (stall recovery, seek, buffer drain) were
    # answered from the ~4x-smaller m4a, the offsets would land in a different
    # container — CoreMedia errors out and auto-skip walks the user off the
    # track (adversarial review, 2026-08-09; the hazard predated the spool but
    # the at-completion remux kick made it the common case). A FRESH item start
    # (range absent or from byte 0 — AVFoundation always probes first) is the
    # one safe moment to adopt the m4a: do so and retire the spool.
    pin_itag = False
    if _SPOOL_ENABLED and not prefetch:
        sp_live = _spools.get(video_id)
        if sp_live is not None and sp_live.error is not None:
            sp_live = None
        if sp_live is not None:
            _rng = _parse_byte_range(range_header, sp_live.total or (1 << 62))
            fresh_start = _rng is None or _rng[0] == 0
            if fresh_start and sp_live.done and _remuxed_ready(video_id):
                _spool_discard(video_id)  # new item + m4a ready → adopt it below
                sp_live = None
        if sp_live is not None:
            await _spool_wait(sp_live, lambda: sp_live.total is not None, _SPOOL_HEADERS_WAIT)
            resp = _spool_response(
                sp_live, range_header, prefetch=False, mime=sp_live.mime,
                video_id=video_id, on_release=_release, t0=_t0,
                cache_ms=0, extract_ms=0,
            )
            if resp is not None:
                return resp
            # Unusable for THIS request (deep seek past the fill, headers still
            # unknown): stay in the SAME byte space — skip the remux tiers and
            # let the legacy upstream proxy below serve the itag-18 range.
            pin_itag = True

    # FAST PATH: serve the cached audio-only remux straight off local disk —
    # ~4x smaller than itag-18 and no googlevideo/proxy round-trip on playback.
    # A local-file serve holds no upstream connection, so release the stream
    # semaphore. (See _local_audio_response for the Range/206 detail that fixed
    # the PlaybackError-1-no-audio bug, root-caused on device 2026-06-08.)
    rpath = None if pin_itag else _remuxed_ready(video_id)
    tier = "L1" if rpath else None
    if not rpath and not pin_itag:
        # Shared store — pull a once-built remux to local disk and serve it (no
        # proxy fetch, no throttle). R2 is now the only shared tier; the
        # Postgres blob fallback that used to sit behind it has been retired.
        rpath = await _r2_pull_to_local(video_id)
        if rpath:
            tier = "R2"
    _cache_ms = _ms(_t0)
    if rpath:
        resp = _local_audio_response(
            rpath, request.headers.get("range"), request.headers.get("if-range"),
        )
        if resp is not None:
            _release()
            logger.info(
                "[media_proxy] audio_stream TIMING video_id=%s tier=%s "
                "cache_lookup_ms=%d total_ms=%d",
                video_id, tier, _cache_ms, _ms(_t0),
            )
            return resp
        # cached file vanished / 0 bytes (pruned mid-flight) → fall through.
        tier = None

    # MISS. Two callers, two needs:
    #  • The mobile PREFETCH (?prefetch=1) downloads the NEXT track to the phone's
    #    disk DURING the current song — it has time to spare, so build-and-wait
    #    and hand it the small audio-only file (cheap to store, instant to play
    #    from local disk on the next ⏭). This is what makes skip truly instant.
    #  • An IMMEDIATE play (first song, or a skip the prefetch didn't cover) wants
    #    the FIRST BYTE fast — blocking on the full build is a regression (the
    #    old itag-18 proxy streamed bytes right away). So DON'T wait: serve the
    #    itag-18 proxy progressively now and build the remux in the background so
    #    the prefetch / next request HITs the small file.
    # The build is deduped (one per video) and we release the per-tenant
    # semaphore before waiting — a build holds no stream slot, and the cap is
    # only 5 which AVFoundation's parallel connections would otherwise trip.
    if _ffmpeg_available() and prefetch:
        _release()
        built = await _remux_now(video_id, budget=_REMUX_SYNC_BUDGET_SECS)
        if built:
            resp = _local_audio_response(
                built, request.headers.get("range"), request.headers.get("if-range"),
            )
            if resp is not None:
                logger.info("[media_proxy] audio_stream remux BUILT (prefetch) video_id=%s", video_id)
                return resp
    # Immediate play (or prefetch build failed) — serve the itag-18 proxy
    # progressively now (fast first byte, fail-open).
    #
    # The background remux build that used to start HERE is deferred to the end
    # of the response body instead. It pulls ~11.8MB of itag-18 through the same
    # single residential proxy this response is streaming through, and that
    # proxy's uplink is the bottleneck (~1MB/s, less under load). Racing the
    # build against the bytes the user is waiting to hear made the cold first
    # play measurably slower to sound — the build wins nothing, because nothing
    # can read the remux until it is finished anyway. Deferred, the cache still
    # warms for the next play and the ⏭, which is all it was ever for.
    _t_extract = time.monotonic()
    try:
        result = await asyncio.wait_for(
            _extract_coalesced(video_id),
            timeout=_AUDIO_EXTRACT_TIMEOUT_SECS,
        )
    except asyncio.TimeoutError:
        _release()
        logger.warning("[media_proxy] audio_stream extract timeout video_id=%s", video_id)
        return JSONResponse(status_code=504, content={"error": "extraction_timeout"})
    except Exception as e:
        _release()
        logger.warning("[media_proxy] audio_stream extract failed video_id=%s: %s", video_id, e)
        return JSONResponse(status_code=502, content={"error": "extraction_failed", "detail": str(e)})

    _extract_ms = _ms(_t_extract)

    if "error" in result:
        _release()
        return JSONResponse(status_code=502, content=result)

    mime = result.get("mime_type") or "audio/mp4"

    # SPOOL FIRST: one upstream download serves this request, AVFoundation's
    # other parallel connections, and the remux build (see the spool section).
    # Any None below falls through to the legacy direct-upstream path, which
    # is byte-for-byte the pre-spool behaviour.
    if _SPOOL_ENABLED:
        spool = await _spool_get_or_start(video_id, result)
        if spool is not None:
            # The response needs Content-Length up front — bounded wait for the
            # downloader's headers. A spool started at broadcast-warm time has
            # them already; one started by this very request pays one TTFB.
            await _spool_wait(spool, lambda: spool.total is not None, _SPOOL_HEADERS_WAIT)
            resp = _spool_response(
                spool,
                request.headers.get("range"),
                prefetch=prefetch,
                mime=mime,
                video_id=video_id,
                on_release=_release,
                t0=_t0,
                cache_ms=_cache_ms,
                extract_ms=_extract_ms,
            )
            if resp is not None:
                return resp

    upstream_url = result["url"]
    _t_upstream = time.monotonic()

    # Forward the client's Range header so AVFoundation can seek; googlevideo
    # honours it and replies 206 + Content-Range.
    upstream_headers = {"User-Agent": "Mozilla/5.0"}
    if range_header:
        upstream_headers["Range"] = range_header

    # googlevideo signs the stream URL to the IP that EXTRACTED it, so the
    # byte-pump must leave from that same address or googlevideo 403s the
    # mismatch. `_proxy_for_result` reads the pool slot the extraction recorded
    # rather than recomputing it, so a cached URL keeps working across replicas
    # and across a change to the pool size. When no proxy is set, extraction and
    # fetch are both Railway-direct, which also matches.
    client = httpx.AsyncClient(
        timeout=_STREAM_TIMEOUT,
        follow_redirects=True,
        proxy=_proxy_for_result(result),
    )
    try:
        upstream_req = client.build_request("GET", upstream_url, headers=upstream_headers)
        # Bounded explicitly. `_STREAM_TIMEOUT`'s connect=10 governs the TCP
        # connection to the PROXY; the CONNECT tunnel handshake and the response
        # headers behind it fall under read=60. Production logged three
        # "504 Gateway Timeout" upstream-connect failures on 2026-08-04/05 —
        # IPRoyal's own gateway answering CONNECT — and the phone can sit for a
        # minute before that becomes a 502, by which time its auto-skip has
        # moved the user off the song they asked for. Headers arrive in ~1s on a
        # healthy path (measured), so 20s is generous and still bounded.
        upstream = await asyncio.wait_for(
            client.send(upstream_req, stream=True), timeout=_UPSTREAM_HEADERS_TIMEOUT
        )
    except asyncio.TimeoutError:
        await client.aclose()
        _release()
        logger.warning(
            "[media_proxy] audio_stream upstream-connect timed out after %.0fs video_id=%s",
            _UPSTREAM_HEADERS_TIMEOUT, video_id,
        )
        return JSONResponse(status_code=504, content={"error": "upstream_connect_timeout"})
    except Exception as e:
        await client.aclose()
        # Recover when the PRIMARY egress refused the CONNECT — busy (the
        # other replica holds the device-lock) or a durable outage (dead
        # creds strand every warm primary-tier cache entry at 502 for up to
        # an hour if only busy recovers — adversarial review). Never for a
        # prefetch: it is opportunistic, and recovering it would purge a
        # healthy shared entry and demote the video to the metered fallback
        # over bytes nobody is waiting on.
        if not (
            (_is_proxy_busy(e) or _is_proxy_outage(e))
            and _fallback_available()
            and not prefetch
            and result.get("proxy_tier") != "fallback"
        ):
            _release()
            logger.warning("[media_proxy] audio_stream upstream-connect failed video_id=%s: %s", video_id, e)
            return JSONResponse(status_code=502, content={"error": "upstream_fetch_failed", "detail": str(e)})
        # Purge the now-unfetchable primary-tier entry and re-extract:
        # `_extract_audio` routes a busy/dead primary to the fallback tier on
        # its own, so the fresh result carries an egress that will accept us.
        # Same shape as the signed-URL 403/410 recovery below. Collision
        # cost: ~1-2s and a residential-speed serve — never a failed play.
        logger.warning(
            "[media_proxy] audio_stream egress %s — re-extracting toward the "
            "fallback tier video_id=%s",
            "BUSY" if _is_proxy_busy(e) else "DOWN", video_id,
        )
        await _purge_extraction(video_id, poisoned_url=upstream_url)
        client = None
        try:
            result = await asyncio.wait_for(
                _extract_coalesced(video_id), timeout=_AUDIO_EXTRACT_TIMEOUT_SECS
            )
            if "error" in result or not result.get("url"):
                raise RuntimeError(result.get("error") or "no stream url")
            mime = result.get("mime_type") or mime
            # Serve the recovery through the SPOOL when possible — a
            # legacy-direct recovery broke the one-download-per-track promise
            # and pulled the same bytes through the metered gateway twice
            # (once for this response, once for the deferred remux build).
            if _SPOOL_ENABLED:
                spool = await _spool_get_or_start(video_id, result)
                if spool is not None:
                    await _spool_wait(spool, lambda: spool.total is not None, _SPOOL_HEADERS_WAIT)
                    resp = _spool_response(
                        spool, request.headers.get("range"), prefetch=False,
                        mime=mime, video_id=video_id, on_release=_release,
                        t0=_t0, cache_ms=_cache_ms, extract_ms=_extract_ms,
                    )
                    if resp is not None:
                        return resp
            # The 403/410 handler below purges conditionally on
            # `upstream_url` — point it at the URL actually in flight or a
            # bad retry entry survives its own purge.
            upstream_url = result["url"]
            client = httpx.AsyncClient(
                timeout=_STREAM_TIMEOUT,
                follow_redirects=True,
                proxy=_proxy_for_result(result),
            )
            upstream_req = client.build_request(
                "GET", upstream_url, headers=upstream_headers
            )
            try:
                upstream = await asyncio.wait_for(
                    client.send(upstream_req, stream=True),
                    timeout=_UPSTREAM_HEADERS_TIMEOUT,
                )
            except Exception as e2:
                # The re-extract can hand back a RESURRECTED primary-tier
                # entry: `_purge_extraction` races concurrent cache readers
                # and the in-flight tail, both of which can re-install the
                # purged result (adversarial review — losing that race made
                # the retry a hard 502 despite a healthy fallback). If the
                # egress refused us AGAIN, force a fallback-tier extraction
                # PAST both caches and connect once more.
                if (
                    not (_is_proxy_busy(e2) or _is_proxy_outage(e2))
                    or result.get("proxy_tier") == "fallback"
                ):
                    raise
                await client.aclose()
                client = None
                logger.warning(
                    "[media_proxy] audio_stream recovery hit the primary again "
                    "— forcing a fallback-tier extraction video_id=%s",
                    video_id,
                )
                forced = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, _extract_audio_via,
                        video_id, _proxy_slot_for(video_id), "fallback",
                    ),
                    timeout=_AUDIO_EXTRACT_TIMEOUT_SECS,
                )
                if "error" in forced or not forced.get("url"):
                    raise RuntimeError(forced.get("error") or "no stream url")
                _fnow = time.time()
                _fdl = _extract_cache_deadline(forced, _fnow)
                if _fdl > _fnow:
                    _EXTRACT_CACHE[video_id] = (forced, _fdl)
                result = forced
                mime = result.get("mime_type") or mime
                upstream_url = result["url"]
                client = httpx.AsyncClient(
                    timeout=_STREAM_TIMEOUT,
                    follow_redirects=True,
                    proxy=_proxy_for_result(result),
                )
                upstream_req = client.build_request(
                    "GET", upstream_url, headers=upstream_headers
                )
                upstream = await asyncio.wait_for(
                    client.send(upstream_req, stream=True),
                    timeout=_UPSTREAM_HEADERS_TIMEOUT,
                )
        except asyncio.TimeoutError:
            if client is not None:
                await client.aclose()
            _release()
            logger.warning(
                "[media_proxy] audio_stream recovery timed out video_id=%s", video_id
            )
            return JSONResponse(status_code=504, content={"error": "upstream_connect_timeout"})
        except Exception as e2:
            if client is not None:
                await client.aclose()
            _release()
            logger.warning(
                "[media_proxy] audio_stream busy-fallback retry failed video_id=%s: %s",
                video_id, e2,
            )
            return JSONResponse(
                status_code=502,
                content={"error": "upstream_fetch_failed", "detail": str(e2)},
            )

    if upstream.status_code in (403, 410):
        # The signed URL was refused for our own egress. The cause that
        # actually fires: the URL is bound to the proxy's exit IP, that IP is a
        # sticky session with a 30-minute lifetime, and the extraction caches
        # hold results for an hour — so a URL can outlive the address that
        # signed it. 502-ing here would let the client's auto-skip walk the
        # user off the song they asked for, over a stale cache entry. Purge
        # both tiers and extract once more instead.
        status = upstream.status_code
        await upstream.aclose()
        logger.warning(
            "[media_proxy] audio_stream upstream status=%s — purging cached extraction video_id=%s",
            status, video_id,
        )
        await _purge_extraction(video_id, poisoned_url=upstream_url)
        try:
            result = await asyncio.wait_for(
                _extract_coalesced(video_id), timeout=_AUDIO_EXTRACT_TIMEOUT_SECS
            )
            if "error" in result or not result.get("url"):
                raise RuntimeError(result.get("error") or "no stream url")
            mime = result.get("mime_type") or mime
            # Rebuild the client against the RE-EXTRACTED result's egress. The
            # first one is pinned to the old entry's slot, and reusing it is a
            # guaranteed second 403 during the pool rollout: every extraction
            # cached before this shipped reads as slot 0, while the retry that
            # replaces it lands on the video's real slot.
            await client.aclose()
            client = httpx.AsyncClient(
                timeout=_STREAM_TIMEOUT,
                follow_redirects=True,
                proxy=_proxy_for_result(result),
            )
            upstream_req = client.build_request(
                "GET", result["url"], headers=upstream_headers
            )
            upstream = await client.send(upstream_req, stream=True)
        except Exception as e:
            await client.aclose()
            _release()
            logger.warning(
                "[media_proxy] audio_stream re-extract after %s failed video_id=%s: %s",
                status, video_id, e,
            )
            return JSONResponse(status_code=502, content={"error": "upstream_status", "status": status})

    if upstream.status_code >= 400:
        status = upstream.status_code
        await upstream.aclose()
        await client.aclose()
        _release()
        logger.warning("[media_proxy] audio_stream upstream status=%s video_id=%s", status, video_id)
        return JSONResponse(status_code=502, content={"error": "upstream_status", "status": status})

    # Curate response headers: pass through what the client needs to seek,
    # force the audio mime, and never let the device cache the proxied
    # bytes (the underlying signed URL rotates ~6h).
    resp_headers = {"Accept-Ranges": "bytes", "Cache-Control": "no-store"}
    for h in ("content-length", "content-range"):
        if h in upstream.headers:
            resp_headers[h.title()] = upstream.headers[h]

    _upstream_ms = _ms(_t_upstream)

    async def _body():
        sent = 0
        server_cut = False
        t_first: float | None = None
        # Only a LIVE play holds the build gate. A `?prefetch=1` pull is the
        # phone filling its own disk for a future ⏭ — nobody is waiting on it,
        # and letting it block builds would deadlock the two halves of warming
        # against each other.
        gating = not prefetch
        if gating:
            _live_start_begin()

        def _ungate() -> None:
            nonlocal gating
            if gating:
                gating = False
                _live_start_done()

        try:
            # Per-chunk bound once bytes are flowing (see _MID_STREAM_STALL_SECS):
            # the httpx read timeout is 60s, and a mid-song upstream stall held
            # the response open — silently — for all of it.
            _chunks = upstream.aiter_bytes(_STREAM_CHUNK_BYTES)
            while True:
                try:
                    if sent > 0:
                        chunk = await asyncio.wait_for(
                            _chunks.__anext__(), timeout=_MID_STREAM_STALL_SECS + 2.0,
                        )
                    else:
                        chunk = await _chunks.__anext__()
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError:
                    logger.warning(
                        "[media_proxy] audio_stream STALLED video_id=%s sent=%d "
                        "— cutting the response",
                        video_id, sent,
                    )
                    server_cut = True
                    break
                if t_first is None:
                    t_first = time.monotonic()
                    # The one number that maps to what the user hears. iOS needs
                    # roughly 2.5MB of itag-18 buffered before it makes a sound,
                    # so first-byte plus (2.5MB / throughput) IS the wait — which
                    # is why throughput is logged at the end of the body too.
                    logger.info(
                        "[media_proxy] audio_stream TIMING video_id=%s tier=MISS "
                        "cache_lookup_ms=%d extract_ms=%d upstream_ms=%d "
                        "first_byte_ms=%d",
                        video_id, _cache_ms, _extract_ms, _upstream_ms, _ms(_t0),
                    )
                sent += len(chunk)
                # Release the gate the moment the client has enough to be making
                # sound. Holding it until the response completes would block
                # builds for the whole ~30s progressive pull and starve the
                # prefetch that makes ⏭ instant — the gate is for the silence
                # before playback, not for playback.
                if sent >= _LIVE_GATE_BYTES:
                    _ungate()
                yield chunk
        finally:
            # Covers the short-file case (never reached the byte threshold), an
            # upstream error, and a client disconnect. Idempotent via `gating`.
            _ungate()
            # Throughput over the bytes actually delivered. Below ~1MB/s the
            # residential proxy is the bottleneck and no code change helps —
            # this is the line that tells the two apart without a stopwatch.
            if t_first is not None and sent > 0:
                _dur = max(time.monotonic() - t_first, 1e-3)
                logger.info(
                    "[media_proxy] audio_stream DELIVERED video_id=%s bytes=%d "
                    "stream_ms=%d throughput_mbps=%.2f total_ms=%d",
                    video_id, sent, int(_dur * 1000), (sent / _dur) / 1e6, _ms(_t0),
                )
            try:
                _promised = int(resp_headers.get("Content-Length") or 0)
            except (TypeError, ValueError):
                _promised = 0
            # Server-cut only — a client disconnect also ends short (review).
            if server_cut and 0 < sent < _promised:
                logger.warning(
                    "[media_proxy] audio_stream TRUNCATED video_id=%s sent=%d promised=%d",
                    video_id, sent, _promised,
                )
            await upstream.aclose()
            await client.aclose()
            _release()
            # Warm the remux cache now that this response is no longer competing
            # with it for the residential proxy's uplink. Runs on client
            # disconnect too, which is correct: a track skipped after 5 seconds
            # is still one the station may come back to. `_ensure_remux_bg`
            # dedupes against an in-flight build and no-ops when one is cached.
            asyncio.create_task(_ensure_remux_bg(video_id))

    return StreamingResponse(
        _body(),
        status_code=upstream.status_code,  # 200 (full) or 206 (range)
        headers=resp_headers,
        media_type=mime,
    )
