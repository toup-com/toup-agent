"""Media proxy — fetches direct video stream URLs for cast-to-TV.

Uses yt-dlp (Python library) to extract stream URLs from YouTube.
Piped APIs are unreliable (frequent downtime), so we use yt-dlp directly.
"""

import logging
from fastapi import APIRouter
from fastapi.responses import JSONResponse

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
