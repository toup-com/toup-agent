"""
Toup Video Search — YouTube search and metadata via yt-dlp.

No API key needed. Uses yt-dlp's built-in YouTube search capability
to find videos, extract metadata, and return URLs.

For playback: returns the YouTube URL to the user — doesn't try to
play video in the VPS browser.
"""

import asyncio
import json
import logging
import shutil
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class VideoResult:
    title: str
    url: str
    duration: str = ""
    channel: str = ""
    view_count: str = ""
    thumbnail: str = ""
    description: str = ""


async def _run_ytdlp(args: list, timeout: int = 30) -> Optional[str]:
    """Run yt-dlp command and return stdout."""
    ytdlp = shutil.which("yt-dlp")
    if not ytdlp:
        logger.warning("[ToupVideo] yt-dlp not installed")
        return None

    try:
        proc = await asyncio.create_subprocess_exec(
            ytdlp, *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        if proc.returncode != 0:
            logger.warning("[ToupVideo] yt-dlp error: %s", stderr.decode(errors="replace")[:200])
            return None
        return stdout.decode(errors="replace")
    except asyncio.TimeoutError:
        logger.warning("[ToupVideo] yt-dlp timed out after %ds", timeout)
        proc.kill()
        return None
    except Exception as e:
        logger.warning("[ToupVideo] yt-dlp failed: %s", e)
        return None


def _format_duration(seconds: Optional[float]) -> str:
    """Format seconds into MM:SS or HH:MM:SS."""
    if not seconds:
        return ""
    s = int(seconds)
    if s < 3600:
        return f"{s // 60}:{s % 60:02d}"
    return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _format_count(count: Optional[int]) -> str:
    """Format view count with K/M suffix."""
    if not count:
        return ""
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M views"
    if count >= 1_000:
        return f"{count / 1_000:.1f}K views"
    return f"{count} views"


async def toup_video_search(query: str, count: int = 5) -> str:
    """
    Search YouTube for videos matching the query.

    Returns formatted results with title, URL, duration, channel, views.
    Uses yt-dlp's ytsearch feature — no API key needed.
    """
    # yt-dlp search: ytsearchN:query returns N results
    output = await _run_ytdlp([
        f"ytsearch{count}:{query}",
        "--dump-json",
        "--flat-playlist",
        "--no-download",
        "--no-warnings",
        "--quiet",
    ], timeout=30)

    if not output:
        # Fallback: scrape YouTube search results page
        return await _youtube_scrape_fallback(query, count)

    results: List[VideoResult] = []
    for line in output.strip().split("\n"):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            results.append(VideoResult(
                title=data.get("title", "Untitled"),
                url=data.get("webpage_url") or data.get("url") or f"https://www.youtube.com/watch?v={data.get('id', '')}",
                duration=_format_duration(data.get("duration")),
                channel=data.get("channel") or data.get("uploader") or "",
                view_count=_format_count(data.get("view_count")),
                thumbnail=data.get("thumbnail") or "",
                description=(data.get("description") or "")[:200],
            ))
        except (json.JSONDecodeError, KeyError):
            continue

    if not results:
        return "No videos found."

    lines = []
    for i, v in enumerate(results[:count], 1):
        lines.append(f"{i}. {v.title}")
        lines.append(f"   {v.url}")
        meta_parts = []
        if v.duration:
            meta_parts.append(v.duration)
        if v.channel:
            meta_parts.append(v.channel)
        if v.view_count:
            meta_parts.append(v.view_count)
        if meta_parts:
            lines.append(f"   {' · '.join(meta_parts)}")
        lines.append("")

    return "\n".join(lines)


async def toup_video_info(url: str) -> str:
    """Get detailed metadata for a specific video URL."""
    output = await _run_ytdlp([
        url,
        "--dump-json",
        "--no-download",
        "--no-warnings",
        "--quiet",
    ], timeout=30)

    if not output:
        return f"Could not get info for {url}"

    try:
        data = json.loads(output.strip().split("\n")[0])
    except (json.JSONDecodeError, IndexError):
        return f"Could not parse info for {url}"

    parts = [
        f"Title: {data.get('title', 'Unknown')}",
        f"URL: {data.get('webpage_url', url)}",
        f"Channel: {data.get('channel') or data.get('uploader', 'Unknown')}",
        f"Duration: {_format_duration(data.get('duration'))}",
        f"Views: {_format_count(data.get('view_count'))}",
    ]
    desc = data.get("description", "")
    if desc:
        parts.append(f"Description: {desc[:500]}")

    return "\n".join(parts)


async def _youtube_scrape_fallback(query: str, count: int = 5) -> str:
    """Fallback: scrape YouTube search via httpx (if yt-dlp unavailable)."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(
                "https://www.youtube.com/results",
                params={"search_query": query},
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/137.0.0.0 Safari/537.36"
                    ),
                    "Accept-Language": "en-US,en;q=0.9",
                },
            )
            resp.raise_for_status()

        # YouTube embeds search results as JSON in the page
        import re
        match = re.search(r"var ytInitialData = ({.+?});</script>", resp.text)
        if not match:
            return "No videos found."

        data = json.loads(match.group(1))
        contents = (
            data.get("contents", {})
            .get("twoColumnSearchResultsRenderer", {})
            .get("primaryContents", {})
            .get("sectionListRenderer", {})
            .get("contents", [{}])[0]
            .get("itemSectionRenderer", {})
            .get("contents", [])
        )

        lines = []
        idx = 0
        for item in contents:
            video = item.get("videoRenderer")
            if not video:
                continue
            idx += 1
            if idx > count:
                break
            vid_id = video.get("videoId", "")
            title = video.get("title", {}).get("runs", [{}])[0].get("text", "Untitled")
            channel = video.get("ownerText", {}).get("runs", [{}])[0].get("text", "")
            duration = video.get("lengthText", {}).get("simpleText", "")
            views = video.get("viewCountText", {}).get("simpleText", "")

            lines.append(f"{idx}. {title}")
            lines.append(f"   https://www.youtube.com/watch?v={vid_id}")
            meta = [p for p in [duration, channel, views] if p]
            if meta:
                lines.append(f"   {' · '.join(meta)}")
            lines.append("")

        return "\n".join(lines) if lines else "No videos found."

    except Exception as e:
        logger.warning("[ToupVideo] YouTube scrape fallback failed: %s", e)
        return f"Video search failed: {e}"
