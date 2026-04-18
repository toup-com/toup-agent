"""YouTube Music Song Radio — station builder.

Replaces the old picker-per-track approach (scrape YT search → filter → hope)
with the API that powers YT Music's own "Start Radio" button. Same algorithm,
same queue, no filter maintenance.

`ytmusicapi.YTMusic().get_watch_playlist(videoId=seed, limit=N)` returns a
Mix/Station playlist rooted at the seed video. The queue is music-only by
construction — no Shorts, no reaction videos, no "YouTube Video" metadata
failures, proper artist + duration per track. We skip the first entry (it's
the seed itself) and iterate through the rest on each track-end event.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class StationTrack:
    video_id: str
    title: str
    artist: str = ""
    length: str = ""  # e.g. "4:18"

    def display_title(self) -> str:
        # "Artist — Title" for logs / debug. Frontend uses title directly.
        if self.artist and self.title:
            return f"{self.artist} — {self.title}"
        return self.title or self.video_id


def _parse_track(raw: dict) -> Optional[StationTrack]:
    vid = (raw.get("videoId") or "").strip()
    title = (raw.get("title") or "").strip()
    if not vid or not title:
        return None
    artists = raw.get("artists") or []
    artist_name = ""
    if isinstance(artists, list) and artists:
        first = artists[0] or {}
        artist_name = (first.get("name") or "").strip()
    length = (raw.get("length") or "").strip()
    return StationTrack(video_id=vid, title=title, artist=artist_name, length=length)


async def build_station(seed_video_id: str, limit: int = 50) -> List[StationTrack]:
    """Return the YT Music Song Radio queue for `seed_video_id`.

    - Excludes the seed track itself (always index 0 of the returned queue).
    - Filters entries missing a `videoId` or `title` (rare for YT Music output,
      but keeps `StationTrack` invariants clean for downstream code).
    - Returns an empty list if YT Music rejects the seed (non-music, region-
      blocked, age-restricted, etc.) — caller decides the fallback.

    Runs `get_watch_playlist` in a thread since ytmusicapi uses blocking
    requests and we're in asyncio land.
    """
    seed = (seed_video_id or "").strip()
    if not seed:
        return []

    try:
        from ytmusicapi import YTMusic
    except ImportError:
        logger.warning("[radio/playlist] ytmusicapi not installed")
        return []

    def _fetch() -> dict:
        ytm = YTMusic()
        return ytm.get_watch_playlist(videoId=seed, limit=limit)

    try:
        raw = await asyncio.to_thread(_fetch)
    except Exception as e:
        print(
            f"[radio/playlist] build_station failed seed={seed} "
            f"err={type(e).__name__}: {e}",
            flush=True,
        )
        return []

    tracks_raw = raw.get("tracks") or []
    station: List[StationTrack] = []
    skipped = 0
    for i, t in enumerate(tracks_raw):
        # Skip the seed (always index 0 of YT Music's watch-playlist response).
        if i == 0 and (t or {}).get("videoId") == seed:
            continue
        parsed = _parse_track(t or {})
        if parsed is None:
            skipped += 1
            continue
        station.append(parsed)

    print(
        f"[radio/playlist] built seed={seed} tracks={len(station)} "
        f"skipped_invalid={skipped} playlistId={raw.get('playlistId')!r}",
        flush=True,
    )
    return station
