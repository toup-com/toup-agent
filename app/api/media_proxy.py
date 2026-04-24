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

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from app.api.auth import get_current_user

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


def _extract_audio(video_id: str) -> dict:
    """Blocking extract; callers must run in a thread / executor."""
    import yt_dlp
    opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        # Prefer m4a/aac — iOS AVFoundation decodes it natively. Fall back
        # to any bestaudio (usually webm/opus) only if m4a is unavailable.
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "socket_timeout": 10,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(
            f"https://www.youtube.com/watch?v={video_id}",
            download=False,
        )
    if not info or not info.get("url"):
        return {"error": "no_stream_url"}
    url = info["url"]
    ext = info.get("ext") or "m4a"
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
