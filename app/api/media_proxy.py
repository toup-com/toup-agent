"""Media proxy — fetches Piped stream URLs server-side to avoid CORS."""

import logging
from fastapi import APIRouter
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/media", tags=["Media"])

PIPED_APIS = [
    "https://pipedapi.kavin.rocks",
    "https://pipedapi.adminforge.de",
    "https://api.piped.yt",
]


@router.get("/stream/{video_id}")
async def get_stream(video_id: str):
    """Fetch HLS stream URL from Piped (server-side, no CORS issues)."""
    import httpx

    for api in PIPED_APIS:
        try:
            async with httpx.AsyncClient(timeout=8, follow_redirects=True) as client:
                resp = await client.get(f"{api}/streams/{video_id}")
                if resp.status_code != 200:
                    continue
                data = resp.json()
                hls = data.get("hls")
                title = data.get("title", "")
                thumbnail = data.get("thumbnailUrl", "")
                if hls:
                    return {"hls": hls, "title": title, "thumbnail": thumbnail}
        except Exception as e:
            logger.warning("[media_proxy] %s failed: %s", api, e)

    return JSONResponse(status_code=404, content={"error": "No stream found"})
