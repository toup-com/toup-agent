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
    length: str = ""                # e.g. "4:18"
    # YT Music's videoType — differentiates the "Song" (ATV = audio/Topic
    # upload) from the "Music Video" (OMV) for the same song. Song-mode
    # display prefers ATV; Video-mode prefers OMV. Empty string = unknown.
    video_type: str = ""
    # Biggest thumbnail URL (album art for ATV tracks, frame cap for OMV).
    # Displayed in the frontend Song-mode overlay.
    thumbnail_url: str = ""

    def display_title(self) -> str:
        # "Artist — Title" for logs / debug. Frontend uses title directly.
        if self.artist and self.title:
            return f"{self.artist} — {self.title}"
        return self.title or self.video_id


def _biggest_thumbnail_url(raw: dict) -> str:
    """Return the URL of the largest thumbnail in a track payload, or ''."""
    thumbs = raw.get("thumbnail") or raw.get("thumbnails") or []
    if not isinstance(thumbs, list) or not thumbs:
        return ""
    # thumbnail entries are {url, width, height}; pick max by width.
    best = max(thumbs, key=lambda t: (t or {}).get("width", 0) or 0, default=None)
    if not best:
        return ""
    return (best.get("url") or "").strip()


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
    video_type = (raw.get("videoType") or "").strip()
    thumbnail_url = _biggest_thumbnail_url(raw)
    return StationTrack(
        video_id=vid,
        title=title,
        artist=artist_name,
        length=length,
        video_type=video_type,
        thumbnail_url=thumbnail_url,
    )


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


# "MUSIC_VIDEO_TYPE_ATV" is YT Music's enum for audio-only tracks (the
# auto-generated "- Topic" channel uploads + proper catalog releases).
# OMV = Official Music Video. UGC = user-generated clip.
_ATV_VIDEO_TYPE = "MUSIC_VIDEO_TYPE_ATV"


async def find_topic_version(
    track: StationTrack,
    timeout: float = 6.0,
) -> Optional[StationTrack]:
    """Try to find an ATV ("Song"/Topic) version of `track` via YT Music search.

    Returns a StationTrack with the ATV videoId (title/artist preserved from
    the caller when the search result's metadata is thinner), or None on
    failure. Best-effort: caller is expected to fall back to the original
    track if this returns None.
    """
    if track.video_type == _ATV_VIDEO_TYPE:
        return track  # already the Song version
    if not track.title:
        return None

    try:
        from ytmusicapi import YTMusic
    except ImportError:
        return None

    q = f"{track.title} {track.artist}".strip()
    if not q:
        return None

    def _search() -> list:
        ytm = YTMusic()
        return ytm.search(q, filter="songs", limit=5) or []

    try:
        results = await asyncio.wait_for(asyncio.to_thread(_search), timeout=timeout)
    except Exception as e:
        print(
            f"[radio/playlist] topic_lookup_failed track={track.video_id} "
            f"q={q!r} err={type(e).__name__}: {e}",
            flush=True,
        )
        return None

    # Prefer the first hit that is ATV and shares the artist (case-insensitive).
    want_artist = track.artist.strip().lower()
    for r in results:
        r_type = (r or {}).get("videoType") or (r or {}).get("resultType")
        if r_type != _ATV_VIDEO_TYPE and (r or {}).get("resultType") != "song":
            continue
        vid = (r.get("videoId") or "").strip()
        if not vid or vid == track.video_id:
            continue
        r_artists = r.get("artists") or []
        r_artist_name = ""
        if r_artists:
            r_artist_name = (r_artists[0] or {}).get("name", "").strip()
        if want_artist and r_artist_name.lower() and want_artist != r_artist_name.lower():
            continue
        # Build a new StationTrack carrying the search result's richer metadata
        # where available, but keep the caller's title/artist if the search is thin.
        return StationTrack(
            video_id=vid,
            title=(r.get("title") or track.title).strip(),
            artist=r_artist_name or track.artist,
            length=(r.get("duration") or track.length or "").strip(),
            video_type=_ATV_VIDEO_TYPE,
            thumbnail_url=_biggest_thumbnail_url(r) or track.thumbnail_url,
        )

    print(
        f"[radio] topic_lookup_failed fallback={track.video_id} q={q!r} "
        f"results_checked={len(results)}",
        flush=True,
    )
    return None
