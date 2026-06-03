"""Media proxy — fetches direct video / audio stream URLs.

Uses yt-dlp (Python library) to extract stream URLs from YouTube.
Piped APIs are unreliable (frequent downtime), so we use yt-dlp directly.

Two routes:

- `GET /media/stream/{video_id}` — combined video+audio mp4, used by
  cast-to-TV. Uses `format: best[ext=mp4]/best`.
- `GET /media/{video_id}/audio_url` — audio-only stream used by the
  mobile hybrid player for background playback. Uses
  `format: bestaudio[ext=m4a]/bestaudio/best`. ToS posture: extracting
  direct audio URLs and playing them outside YouTube's player violates
  YouTube ToS §5.B; the product team accepted the tail risk in exchange
  for the video-when-visible / audio-when-not UX. See SKILL.md in the
  radio-mode skill.
"""

import asyncio
import logging
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
_PLAYER_CLIENTS = ("tv_embedded", "android_music", "ios", "android")

# Clients that consume a GVS PO token. When a bgutil provider is configured
# (settings.bgutil_pot_base_url) these are tried FIRST: on a datacenter IP
# the tokenless clients above get bot-blocked, but a web client armed with a
# fresh proof-of-origin token slips past. Order = empirical reliability.
_POT_PLAYER_CLIENTS = ("web_safari", "web", "mweb")


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


def _extract_audio(video_id: str) -> dict:
    """Blocking extract; callers must run in a thread / executor.

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
    # With a PO-token provider up, lead with the web clients it can arm; they
    # beat the datacenter-IP bot challenge that blocks the tokenless clients.
    clients = (_POT_PLAYER_CLIENTS + _PLAYER_CLIENTS) if pot_base else _PLAYER_CLIENTS
    last_err: BaseException | None = None

    for client in clients:
        extractor_args: dict = {"youtube": {"player_client": [client]}}
        # Point the bgutil plugin at the provider so yt-dlp fetches a fresh
        # GVS proof-of-origin token for the web clients. Harmless for the
        # tokenless clients (they just ignore it).
        if pot_base:
            extractor_args["youtubepot-bgutilhttp"] = {"base_url": [pot_base]}
        opts: dict = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            # Prefer m4a/aac — iOS AVFoundation decodes it natively. Fall
            # back to any bestaudio (usually webm/opus) only if m4a is
            # unavailable.
            "format": "bestaudio[ext=m4a]/bestaudio/best",
            "socket_timeout": 10,
            "extractor_args": extractor_args,
        }
        if cookiefile:
            opts["cookiefile"] = cookiefile
        # Route extraction through a residential proxy when configured — this
        # is the reliable defense against YouTube's datacenter-IP bot challenge.
        if settings.yt_dlp_proxy:
            opts["proxy"] = settings.yt_dlp_proxy

        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(
                    f"https://www.youtube.com/watch?v={video_id}",
                    download=False,
                )
        except Exception as e:
            last_err = e
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
            "[media_proxy] audio_url ok video_id=%s client=%s ext=%s bitrate=%s",
            video_id, client, ext, info.get("abr"),
        )
        return {
            "url": url,
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
            asyncio.get_event_loop().run_in_executor(None, _extract_audio, video_id),
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

    try:
        result = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(None, _extract_audio, video_id),
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

    if "error" in result:
        _release()
        return JSONResponse(status_code=502, content=result)

    upstream_url = result["url"]
    mime = result.get("mime_type") or "audio/mp4"

    # Forward the client's Range header so AVFoundation can seek; googlevideo
    # honours it and replies 206 + Content-Range.
    upstream_headers = {"User-Agent": "Mozilla/5.0"}
    range_header = request.headers.get("range")
    if range_header:
        upstream_headers["Range"] = range_header

    # googlevideo signs the stream URL to the IP that EXTRACTED it. If
    # extraction went through the residential proxy, the byte-pump must use the
    # same proxy or googlevideo 403s the mismatched IP. When no proxy is set,
    # extraction + fetch are both Railway-direct, which also matches.
    client = httpx.AsyncClient(
        timeout=_STREAM_TIMEOUT,
        follow_redirects=True,
        proxy=settings.yt_dlp_proxy or None,
    )
    try:
        upstream_req = client.build_request("GET", upstream_url, headers=upstream_headers)
        upstream = await client.send(upstream_req, stream=True)
    except Exception as e:
        await client.aclose()
        _release()
        logger.warning("[media_proxy] audio_stream upstream-connect failed video_id=%s: %s", video_id, e)
        return JSONResponse(status_code=502, content={"error": "upstream_fetch_failed", "detail": str(e)})

    if upstream.status_code >= 400:
        status = upstream.status_code
        await upstream.aclose()
        await client.aclose()
        _release()
        # 403 here would mean the signed URL rejected OUR IP too (rare —
        # usually a stale extraction); surface as a 502 to the client.
        logger.warning("[media_proxy] audio_stream upstream status=%s video_id=%s", status, video_id)
        return JSONResponse(status_code=502, content={"error": "upstream_status", "status": status})

    # Curate response headers: pass through what the client needs to seek,
    # force the audio mime, and never let the device cache the proxied
    # bytes (the underlying signed URL rotates ~6h).
    resp_headers = {"Accept-Ranges": "bytes", "Cache-Control": "no-store"}
    for h in ("content-length", "content-range"):
        if h in upstream.headers:
            resp_headers[h.title()] = upstream.headers[h]

    async def _body():
        try:
            async for chunk in upstream.aiter_bytes(_STREAM_CHUNK_BYTES):
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()
            _release()

    return StreamingResponse(
        _body(),
        status_code=upstream.status_code,  # 200 (full) or 206 (range)
        headers=resp_headers,
        media_type=mime,
    )
