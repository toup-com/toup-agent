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
    last_err: BaseException | None = None

    for client in _PLAYER_CLIENTS:
        extractor_args: dict = {"youtube": {"player_client": [client]}}
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


# video_id -> (result_dict, cache_until_epoch). Caches SUCCESSFUL extractions
# so the lock-screen handoff is instant: AVFoundation fires a range probe, the
# real request, and seeks — each previously triggered a fresh ~2-3s yt-dlp
# call. With the cache they reuse one extraction, and the mobile pre-warm
# (media_play → /audio_url) fills it BEFORE the user locks, so the byte-pump
# skips extraction entirely. Safe now that extraction + byte-pump share one
# stable egress IP (the Tailscale exit node): the signed googlevideo URL stays
# valid for the same IP on reuse — the rotating-Railway-IP 403 that forced the
# earlier cache revert (commit 4039597a) can't happen through the proxy.
_EXTRACT_CACHE: dict[str, tuple[dict, float]] = {}
_EXTRACT_CACHE_TTL_CAP = 3600.0  # never serve an extraction older than 1h


def _cached_extract(video_id: str) -> dict:
    """`_extract_audio` with a short-lived per-video cache. Blocking; run in an
    executor like the underlying call. Only successful results are cached."""
    now = time.time()
    hit = _EXTRACT_CACHE.get(video_id)
    if hit and hit[1] > now:
        logger.info("[media_proxy] extract cache HIT video_id=%s", video_id)
        return hit[0]
    result = _extract_audio(video_id)
    if "error" not in result and result.get("url"):
        cache_until = now + _EXTRACT_CACHE_TTL_CAP
        exp = result.get("expires_at") or 0
        if exp:
            # Stop serving 5 min before the signed URL itself expires.
            cache_until = min(cache_until, float(exp) - 300.0)
        if cache_until > now:
            _EXTRACT_CACHE[video_id] = (result, cache_until)
    return result


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


def _do_remux(video_id: str) -> str | None:
    """Blocking: extract the audio-only URL → ffmpeg remux to a progressive
    faststart m4a on local disk. Returns the cached path, or None so the caller
    falls back to the itag-18 proxy. Idempotent. Run in an executor."""
    ready = _remuxed_ready(video_id)
    if ready:
        return ready
    if not _ffmpeg_available():
        return None
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
            proxy=settings.yt_dlp_proxy or None, timeout=60.0, follow_redirects=True
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
    """Keep the newest _AUDIO_CACHE_MAX_FILES remuxed files; drop older ones."""
    try:
        entries = [
            os.path.join(_AUDIO_CACHE_DIR, f)
            for f in os.listdir(_AUDIO_CACHE_DIR)
            if f.endswith(".m4a")
        ]
    except OSError:
        return
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
    async with _remux_inflight_lock:
        # Drop finished tasks so the map stays small over a long session.
        for k in [k for k, t in _remux_tasks.items() if t.done()]:
            _remux_tasks.pop(k, None)
        task = _remux_tasks.get(video_id)
        if task is None:
            task = asyncio.ensure_future(
                asyncio.get_event_loop().run_in_executor(None, _do_remux, video_id)
            )
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
            asyncio.get_event_loop().run_in_executor(None, _cached_extract, video_id),
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
    # Pre-warm the audio-only remux in the background so the native
    # /audio_stream play (and the next ⏭) serves it from local disk → fast.
    # Mobile pre-warms the current track + upcoming via this endpoint at
    # media_play, so the remux is usually cached before playback/skip asks.
    asyncio.create_task(_ensure_remux_bg(video_id))
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


def _local_audio_response(rpath: str, range_header: str | None):
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
    base = {"Accept-Ranges": "bytes", "Cache-Control": "no-store"}
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

    # FAST PATH: serve the cached audio-only remux straight off local disk —
    # ~4x smaller than itag-18 and no googlevideo/proxy round-trip on playback.
    # A local-file serve holds no upstream connection, so release the stream
    # semaphore. (See _local_audio_response for the Range/206 detail that fixed
    # the PlaybackError-1-no-audio bug, root-caused on device 2026-06-08.)
    rpath = _remuxed_ready(video_id)
    if rpath:
        resp = _local_audio_response(rpath, request.headers.get("range"))
        if resp is not None:
            _release()
            logger.info("[media_proxy] audio_stream remux HIT video_id=%s", video_id)
            return resp
        # cached file vanished / 0 bytes (pruned mid-flight) → fall through.

    # MISS: build the audio-only remux NOW and serve THAT, instead of proxying
    # the big itag-18 (video+audio, ~11.8MB) all the way to the phone. The server
    # pulls itag-18 over its fast link + ffmpeg-strips the video, then sends the
    # phone a ~4x-smaller audio-only file with a correct Content-Length — which
    # both starts playback far sooner AND stops AVFoundation re-fetching the whole
    # file in a loop (the cold-path thrash seen on device). This removes the
    # 2-replica cache split + warm-timing dependency: every miss self-heals into
    # a fast, consistent audio-only serve. The build is deduped, and we release
    # the per-tenant semaphore first — a build holds no stream slot and the cap
    # is only 5, which AVFoundation's parallel connections would otherwise trip.
    # On build failure / budget timeout → fall open to the itag-18 proxy below.
    if _ffmpeg_available():
        _release()
        built = await _remux_now(video_id, budget=_REMUX_SYNC_BUDGET_SECS)
        if built:
            resp = _local_audio_response(built, request.headers.get("range"))
            if resp is not None:
                logger.info("[media_proxy] audio_stream remux BUILT video_id=%s", video_id)
                return resp
    # ffmpeg unavailable / build failed or timed out — serve the itag-18 proxy
    # this once (fail-open); the background build (still running) HITs next time.
    asyncio.create_task(_ensure_remux_bg(video_id))

    try:
        result = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(None, _cached_extract, video_id),
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
