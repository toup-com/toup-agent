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
        try:
            import yt_dlp
        except ImportError:
            return {"error": "yt-dlp not installed"}

        opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            # Get best combined audio+video stream (for casting)
            "format": "best[ext=mp4]/best",
        }
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(f"https://www.youtube.com/watch?v={vid}", download=False)
                if not info:
                    return {"error": "No info found"}
                return {
                    "url": info.get("url", ""),
                    "title": info.get("title", ""),
                    "thumbnail": info.get("thumbnail", ""),
                    "duration": info.get("duration", 0),
                    "ext": info.get("ext", "mp4"),
                }
        except Exception as e:
            logger.warning("[media_proxy] yt-dlp error: %s", e)
            return {"error": str(e)}

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _extract, video_id)

    if "error" in result:
        return JSONResponse(status_code=502, content=result)
    return result
