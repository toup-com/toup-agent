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
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def _ytmusic():
    """YTMusic client routed through the same egress proxy as yt-dlp.

    YT Music's unauthenticated API anti-bot-blocks / rate-limits the agent's
    datacenter IP, so `get_watch_playlist` / `search` return empty FROM THE VPS
    (→ the radio toggle is rejected with `station_unavailable`) even when the
    seed is a perfectly valid Song id — the exact same call from a residential
    IP returns a full 50-track station. Route these requests through
    `settings.yt_dlp_proxy` (the residential Tailscale exit node already used
    for audio extraction in media_proxy.py) so YT Music sees a non-datacenter
    IP. Falls back to a direct client when no proxy is configured (dev/local),
    so behaviour off-VPS — and the web's existing flow — is unchanged.
    """
    from ytmusicapi import YTMusic
    proxy = None
    try:
        from app.config import settings
        proxy = settings.yt_dlp_proxy
    except Exception:
        proxy = None
    if proxy:
        return YTMusic(proxies={"http": proxy, "https": proxy})
    return YTMusic()


def _yt_remote(
    op: str,
    *,
    video_id: str = "",
    query: str = "",
    filter: str = "",
    limit: int = 0,
):
    """Run a ytmusicapi call on the PLATFORM (which has a working residential
    proxy) instead of locally, and return the raw result.

    Why: tenant agents run on Contabo VPS IPs that YT Music anti-bot-blocks,
    and agent containers have NO egress proxy (YT_DLP_PROXY isn't in
    AgentEnvContract), so a local get_watch_playlist/search comes back empty →
    `station_unavailable`. The platform does all other YouTube egress through
    the proxy already; this routes the radio calls the same way.

    Returns the raw result on success, or None to signal "fall back to the
    local call" — i.e. when we're NOT an agent (platform/monolith run their
    own proxied client directly), when the agent isn't wired to the platform,
    or on any RPC error. Fail-open: the worst case is exactly today's
    behaviour, so this can never regress web or a working agent.
    """
    try:
        from app.config import settings
    except Exception:
        return None
    if getattr(settings, "run_mode", "") != "agent":
        return None  # platform/monolith: caller runs _ytmusic() directly (proxied here)
    base = (getattr(settings, "platform_api_url", "") or "").rstrip("/")
    key = getattr(settings, "agent_api_key", "") or ""
    uid = getattr(settings, "user_id", "") or ""
    if not (base and key and uid):
        return None
    body = {"user_id": uid, "op": op}
    if video_id:
        body["video_id"] = video_id
    if query:
        body["query"] = query
    if filter:
        body["filter"] = filter
    if limit:
        body["limit"] = limit
    try:
        import httpx
        resp = httpx.post(
            f"{base}/internal/radio/yt",
            headers={"X-Agent-Key": key},
            json=body,
            timeout=20.0,
        )
    except Exception as e:
        print(f"[radio/playlist] yt_remote op={op} err={type(e).__name__}: {e} — local fallback", flush=True)
        return None
    if resp.status_code != 200:
        print(f"[radio/playlist] yt_remote op={op} http={resp.status_code} — local fallback", flush=True)
        return None
    try:
        return resp.json().get("result")
    except Exception:
        return None


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
    # Which display_mode this track has already been variant-resolved for
    # ("song"/"video"), or "" if not yet resolved. Set by the upcoming-window
    # pre-resolver in ws_chat so the `upcoming` hint the mobile client prebuffers
    # carries the SAME video_id the pop will play (no ATV/OMV swap divergence).
    # "" means "resolve me"; a value == the current mode means "already correct".
    variant_resolved_mode: str = ""

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


async def build_station(
    seed_video_id: str,
    limit: int = 50,
    *,
    seed_title: str = "",
    seed_artist: str = "",
) -> Tuple[Optional[StationTrack], List[StationTrack]]:
    """Return (seed_meta, station) for `seed_video_id` from YT Music Song Radio.

    - `seed_meta` is the StationTrack for the seed itself (index 0 of YT Music's
      watch-playlist response), which carries the real artist / thumbnail /
      video_type the frontend needs for the toggle-on authoritative play
      broadcast. `None` if the response doesn't include the seed.
    - `station` excludes the seed and filters entries missing `videoId` / `title`.
    - If YT Music has no Song-radio for the seed (regional / non-catalog tracks
      return a 200 with empty `tracks`), falls back to a search-based station
      built from `seed_title`/`seed_artist` — see `_build_station_fallback`. Only
      `(None, [])` when even the fallback can't find anything; caller rejects then.

    Runs `get_watch_playlist` in a thread since ytmusicapi uses blocking
    requests and we're in asyncio land.
    """
    seed = (seed_video_id or "").strip()
    if not seed:
        return None, []

    try:
        from ytmusicapi import YTMusic
    except ImportError:
        logger.warning("[radio/playlist] ytmusicapi not installed")
        return None, []

    def _fetch() -> dict:
        remote = _yt_remote("get_watch_playlist", video_id=seed, limit=limit)
        if remote is not None:
            return remote
        ytm = _ytmusic()
        return ytm.get_watch_playlist(videoId=seed, limit=limit)

    try:
        raw = await asyncio.to_thread(_fetch)
    except Exception as e:
        print(
            f"[radio/playlist] build_station failed seed={seed} "
            f"err={type(e).__name__}: {e}",
            flush=True,
        )
        # Don't reject yet — a search-based station may still recover (the seed
        # often carries a usable title even when get_watch_playlist throws).
        raw = {}

    tracks_raw = (raw or {}).get("tracks") or []
    seed_meta: Optional[StationTrack] = None
    station: List[StationTrack] = []
    skipped = 0
    for i, t in enumerate(tracks_raw):
        # Capture the seed's metadata (index 0, matching videoId) and skip it
        # from the station — it's what's already playing, not the queue.
        if i == 0 and (t or {}).get("videoId") == seed:
            seed_meta = _parse_track(t or {})
            continue
        parsed = _parse_track(t or {})
        if parsed is None:
            skipped += 1
            continue
        station.append(parsed)

    print(
        f"[radio/playlist] built seed={seed} tracks={len(station)} "
        f"seed_meta={'yes' if seed_meta else 'no'} "
        f"skipped_invalid={skipped} playlistId={(raw or {}).get('playlistId')!r}",
        flush=True,
    )
    if station:
        return seed_meta, station
    # YT Music had no Song-radio for this seed (regional / non-catalog track →
    # 200 with empty tracks, or the fetch threw above). Recover with a search-
    # based station rather than rejecting the toggle. Reached ONLY on today's
    # empty path, so it can never regress a station that already builds.
    return await _build_station_fallback(seed, seed_title, seed_artist, limit)


async def _build_station_fallback(
    seed: str,
    title: str,
    artist: str,
    limit: int,
) -> Tuple[Optional[StationTrack], List[StationTrack]]:
    """Recover a queue when get_watch_playlist has no Song-radio for `seed`.

    YT Music returns an empty `tracks` list for seeds outside its catalog —
    regional music (e.g. Persian tracks) and plain YouTube uploads — which today
    hard-rejects the radio toggle. Instead, search YT Music for the song by its
    title/artist, re-seed a real station from the top catalog hit (a catalog id
    almost always HAS a watch-playlist even when the seed doesn't), and fall back
    to the search results themselves as the queue.

    Reuses `_yt_remote('search')` / `_ytmusic()` — the SAME proxied egress as
    get_watch_playlist (no new anti-bot surface, no new dependency). Returns
    (None, station): seed_meta is intentionally None so the caller keeps the
    user's real now-playing title/art (the seed isn't in YT Music's catalog, so
    we have no authoritative metadata for it, and the toggle broadcast plays
    `seed_video_id` regardless — see ws_chat radio toggle). `(None, [])` only
    when search is also empty, preserving today's `station_unavailable` reject.
    """
    q = f"{title} {artist}".strip()
    if not q:
        return None, []

    def _search() -> list:
        remote = _yt_remote("search", query=q, filter="songs", limit=8)
        if remote is not None:
            return remote
        return _ytmusic().search(q, filter="songs", limit=8) or []

    try:
        results = await asyncio.wait_for(asyncio.to_thread(_search), timeout=8.0)
    except Exception as e:
        print(
            f"[radio/playlist] fallback search failed q={q!r} "
            f"err={type(e).__name__}: {e}",
            flush=True,
        )
        return None, []

    parsed = [
        p for p in (_parse_track(r or {}) for r in (results or []))
        if p and p.video_id != seed
    ]
    if not parsed:
        print(f"[radio/playlist] fallback no search hits q={q!r}", flush=True)
        return None, []

    # Fallback A: re-seed a genuine station from the top catalog hit.
    top = parsed[0]

    def _reseed() -> dict:
        remote = _yt_remote("get_watch_playlist", video_id=top.video_id, limit=limit)
        if remote is not None:
            return remote
        return _ytmusic().get_watch_playlist(videoId=top.video_id, limit=limit)

    try:
        raw = await asyncio.wait_for(asyncio.to_thread(_reseed), timeout=8.0)
    except Exception:
        raw = {}
    reseeded = [
        p for p in (_parse_track(t or {}) for t in ((raw or {}).get("tracks") or []))
        if p and p.video_id != seed
    ]
    if reseeded:
        print(
            f"[radio/playlist] fallback reseed ok q={q!r} via={top.video_id} "
            f"tracks={len(reseeded)}",
            flush=True,
        )
        return None, reseeded

    # Fallback B: use the search results themselves as the queue.
    print(
        f"[radio/playlist] fallback search-as-queue q={q!r} tracks={len(parsed)}",
        flush=True,
    )
    return None, parsed


# "MUSIC_VIDEO_TYPE_ATV" is YT Music's enum for audio-only tracks (the
# auto-generated "- Topic" channel uploads + proper catalog releases).
# OMV = Official Music Video. UGC = user-generated clip.
_ATV_VIDEO_TYPE = "MUSIC_VIDEO_TYPE_ATV"
_OMV_VIDEO_TYPE = "MUSIC_VIDEO_TYPE_OMV"


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
        remote = _yt_remote("search", query=q, filter="songs", limit=5)
        if remote is not None:
            return remote
        ytm = _ytmusic()
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


async def find_music_video(
    track: StationTrack,
    timeout: float = 6.0,
) -> Optional[StationTrack]:
    """Mirror of `find_topic_version` for Video mode.

    Given an ATV (Topic/audio) track, try to find the Official Music Video
    (OMV) variant via YT Music search with filter='videos'. Returns the OMV
    StationTrack on success, None on failure.

    UGC/reaction/lyric-video uploads are skipped — we accept only OMV hits
    where the artist matches (case-insensitive), same discipline as the
    Topic path. Empty metadata or timeout → fall through.
    """
    if track.video_type == _OMV_VIDEO_TYPE:
        return track  # already the Music Video variant
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
        remote = _yt_remote("search", query=q, filter="videos", limit=10)
        if remote is not None:
            return remote
        ytm = _ytmusic()
        return ytm.search(q, filter="videos", limit=10) or []

    try:
        results = await asyncio.wait_for(asyncio.to_thread(_search), timeout=timeout)
    except Exception as e:
        print(
            f"[radio/playlist] mv_lookup_failed track={track.video_id} "
            f"q={q!r} err={type(e).__name__}: {e}",
            flush=True,
        )
        return None

    want_artist = track.artist.strip().lower()
    for r in results:
        r_type = (r or {}).get("videoType") or ""
        if r_type != _OMV_VIDEO_TYPE:
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
        return StationTrack(
            video_id=vid,
            title=(r.get("title") or track.title).strip(),
            artist=r_artist_name or track.artist,
            length=(r.get("duration") or track.length or "").strip(),
            video_type=_OMV_VIDEO_TYPE,
            thumbnail_url=_biggest_thumbnail_url(r) or track.thumbnail_url,
        )

    print(
        f"[radio] mv_lookup_failed fallback={track.video_id} q={q!r} "
        f"results_checked={len(results)}",
        flush=True,
    )
    return None
