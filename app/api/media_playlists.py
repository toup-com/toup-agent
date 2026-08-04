"""Toup Media — saved, replayable playlists over the radio system.

Every music request already builds a radio station in the user's agent
process; this API is where a station the user liked becomes durable and
replayable: exact tracks, exact order. Music only today; `media_type` exists
so movies can join later without a schema change.

TOPOLOGY (the memories pattern, deliberately):
  * `media_playlists` is an AGENT_ONLY table — the live store is the tenant
    container's DB, where the RadioSessionManager (the thing being captured
    and re-seeded) also lives.
  * The platform proxies every call to the user's agent with X-Agent-Key.
    READS fall back to the platform's mirror table when the agent is
    unreachable; WRITES and PLAY never fall back — a station can only be
    captured or started where the radio manager actually runs, so those
    surface 502 instead of silently succeeding against the wrong database.
  * On the agent, `get_current_user`'s agent-mode branch resolves the proxied
    call to the tenant's own user id, and the station endpoints execute
    locally (the chat WS terminates on the agent, so `broadcast_to_user`
    reaches the phone from here).

Registered on BOTH apps (platform_main + agent_main), same as memories.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import get_current_user
from app.config import settings
from app.db.database import get_db
from app.db.models import AgentConfig, MediaPlaylist, User
from app.schemas import (
    LiveStationResponse,
    PlaylistCreate,
    PlaylistPlayRequest,
    PlaylistPlayResponse,
    PlaylistResponse,
    PlaylistSummaryResponse,
    PlaylistTrack,
    PlaylistUpdate,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/media/playlists", tags=["Toup Media"])

# Stations run ~50 tracks; a capture never needs more, and an unbounded
# tracks_json is how a document column becomes a problem row.
_MAX_TRACKS = 60


# ── Agent proxy helpers (memories.py pattern) ─────────────────────────────

async def _agent_proxy_info(user_id: str, db: AsyncSession) -> Optional[Tuple[str, str]]:
    # Only the PLATFORM proxies. agent_configs is a SHARED table, so a tenant
    # DB carrying a stray row must never make the agent proxy to itself.
    if _serving_locally():
        return None
    try:
        async with db.begin_nested():
            row = (await db.execute(
                select(AgentConfig.agent_url, AgentConfig.agent_api_key).where(
                    AgentConfig.user_id == user_id,
                    AgentConfig.deploy_status == "active",
                )
            )).first()
            if row and row.agent_url and row.agent_api_key:
                return (row.agent_url, row.agent_api_key)
    except Exception:
        pass  # agent_configs may not exist in a tenant DB
    return None


def _serving_locally() -> bool:
    """True where the radio manager + playlist store live in-process."""
    return (settings.run_mode or "").strip().lower() in ("agent", "monolith")


async def _proxy_read(agent_url: str, key: str, path: str):
    """Proxy a read; None on any failure → caller falls back to local DB."""
    from app.services.agent_http import get_agent_http_client

    url = f"{agent_url}/api/media/playlists{path}"
    try:
        client = get_agent_http_client()
        resp = await client.get(url, headers={"X-Agent-Key": key}, timeout=10.0)
        if resp.status_code == 200:
            return resp.json()
        logger.warning("Agent playlists proxy %s returned %s", url, resp.status_code)
    except Exception as e:
        logger.warning("Agent playlists proxy %s failed: %s", url, e)
    return None


async def _proxy_write(agent_url: str, key: str, path: str, method: str, body: Optional[dict] = None):
    """Proxy a write/play. NEVER falls back — 502 when the agent is
    unreachable (a capture or a play only means anything where the radio
    manager runs)."""
    from app.services.agent_http import get_agent_http_client

    url = f"{agent_url}/api/media/playlists{path}"
    try:
        client = get_agent_http_client()
        resp = await client.request(
            method, url, headers={"X-Agent-Key": key}, json=body, timeout=25.0,
        )
    except Exception as e:
        logger.warning("Agent playlists write proxy %s %s failed: %s", method, url, e)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not reach your agent. Please try again.",
        )
    if resp.status_code in (200, 201, 204):
        if resp.status_code == 204 or not resp.content:
            return None
        try:
            return resp.json()
        except Exception:
            return None
    if resp.status_code == 404:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Playlist not found")
    if resp.status_code == 400:
        detail = "Bad request"
        try:
            detail = (resp.json() or {}).get("detail") or detail
        except Exception:
            pass
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail)
    logger.warning("Agent playlists write proxy %s %s returned %s", method, url, resp.status_code)
    raise HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail="Your agent rejected this request. Please try again.",
    )


# ── Serialization ─────────────────────────────────────────────────────────

def _parse_tracks(tracks_json: str) -> List[dict]:
    try:
        parsed = json.loads(tracks_json or "[]")
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def _summary(row: MediaPlaylist) -> PlaylistSummaryResponse:
    return PlaylistSummaryResponse(
        id=row.id,
        name=row.name,
        media_type=row.media_type,
        source=row.source,
        seed_intent=row.seed_intent,
        track_count=row.track_count,
        artwork_url=row.artwork_url,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _full(row: MediaPlaylist) -> PlaylistResponse:
    return PlaylistResponse(
        **_summary(row).model_dump(),
        tracks=[PlaylistTrack(**{k: t.get(k) for k in (
            "video_id", "title", "artist", "thumbnail_url", "length", "video_type",
        )}) for t in _parse_tracks(row.tracks_json) if t.get("video_id")],
    )


def _station_track_dict(t) -> dict:
    return {
        "video_id": t.video_id,
        "title": t.title,
        "artist": t.artist,
        "thumbnail_url": t.thumbnail_url,
        "length": t.length,
        "video_type": t.video_type,
    }


# ── Automatic library capture ─────────────────────────────────────────────
# Every station the agent builds becomes a library entry, without the user
# having to think about it. The founder's ask is literal: "Toup Media must
# contain EVERY playlist the user has ever played." Manual capture stays — it
# is how you name and keep one deliberately — but nothing is lost by default
# any more, which is what the empty library after a day of listening looked
# like.
#
# Called fire-and-forget from the radio paths in ws_chat: never awaited on the
# playback path, never allowed to raise into it.

_AUTOSAVE_DEDUPE_HOURS = 6

# ONE autosave at a time per (user, channel). The station's creation hook and
# its first advance can fire within milliseconds of each other, and both would
# find no row yet — two INSERTs, two near-identical "Ebi radio" entries in the
# library. Everything here is read-modify-write on a single row, so serializing
# is both sufficient and cheap.
_autosave_locks: dict = {}


def _autosave_lock(user_id: str, channel: str) -> asyncio.Lock:
    key = f"{user_id}:{channel}"
    lock = _autosave_locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _autosave_locks[key] = lock
    return lock


def _station_display_name(sess) -> str:
    """A name a human would recognise in a list, from what we know about the
    station: the artist if the seed gave us one, else the words the user used,
    else the date. Never the raw video id."""
    from app.agent.media_resolve import split_artist_title

    seed_title = (getattr(sess.seed_track, "title", "") or "").strip()
    artist, _song = split_artist_title(seed_title)
    if not artist:
        cur = getattr(sess, "current_station_track", None)
        artist = (getattr(cur, "artist", "") or "").strip()
    if artist:
        return f"{artist} radio"
    intent = (getattr(sess, "seed_intent", "") or "").strip()
    if intent:
        from app.agent.media_resolve import content_tokens
        words = content_tokens(intent)
        if words:
            return f"{' '.join(w.capitalize() for w in words[:4])} radio"
    from datetime import datetime, timezone
    return f"Station · {datetime.now(timezone.utc).strftime('%b %-d')}"


async def autosave_station(user_id: str, channel: str) -> None:
    """Create or refresh this station's library entry. Safe to call on every
    advance: it is one row, keyed off the session, and it never raises."""
    try:
        if not _serving_locally():
            return  # the platform has no radio sessions to capture
        async with _autosave_lock(user_id, channel):
            await _autosave_locked(user_id, channel)
    except Exception as e:
        # A library entry is a nicety; playback is not. Never let this surface.
        logger.warning("[toup-media] autosave failed user=%s: %s", user_id[:8], e)


async def _autosave_locked(user_id: str, channel: str) -> None:
    """The body of autosave_station, run under its per-session lock."""
    sess, tracks = _capture_live_station(user_id, channel)
    if sess is None or not tracks:
        return
    if getattr(sess, "autosave_suppressed", False):
        return
    from datetime import datetime, timedelta, timezone
    from app.db.database import async_session_maker

    payload = json.dumps(tracks)
    artwork = next((t.get("thumbnail_url") for t in tracks if t.get("thumbnail_url")), None)
    async with async_session_maker() as db:
        row = None
        if sess.autosave_playlist_id:
            row = (await db.execute(
                select(MediaPlaylist).where(
                    MediaPlaylist.id == sess.autosave_playlist_id,
                    MediaPlaylist.user_id == user_id,
                    # 'auto' ONLY. A row that has been curated by hand is
                    # promoted off 'auto' (delete_playlist_track), and this
                    # capture overwrites tracks_json wholesale — without this
                    # filter the next track advance would silently restore a
                    # song the user just deleted.
                    MediaPlaylist.source == "auto",
                )
            )).scalar_one_or_none()
            if row is None:
                # The row we were mirroring into is gone or is no longer ours to
                # write: the user deleted it, or edited it. Either way this
                # station stops recording rather than resurrecting it on the
                # next advance.
                sess.autosave_suppressed = True
                logger.info(
                    "[toup-media] autosave stopped for user=%s — target row is gone or curated",
                    user_id[:8],
                )
                return
        if row is None:
            # Re-playing the same seed a few minutes later is the same
            # listening session to a human — refresh that entry instead of
            # stacking near-identical rows named "Ebi radio".
            seed_vid = getattr(sess.seed_track, "video_id", "") or ""
            cutoff = datetime.utcnow() - timedelta(hours=_AUTOSAVE_DEDUPE_HOURS)
            if seed_vid:
                row = (await db.execute(
                    select(MediaPlaylist)
                    .where(
                        MediaPlaylist.user_id == user_id,
                        MediaPlaylist.source == "auto",
                        MediaPlaylist.seed_video_id == seed_vid,
                        MediaPlaylist.updated_at >= cutoff,
                    )
                    .order_by(MediaPlaylist.updated_at.desc())
                    .limit(1)
                )).scalar_one_or_none()
        if row is None:
            row = MediaPlaylist(
                user_id=user_id,
                name=_station_display_name(sess),
                media_type="music",
                source="auto",
                seed_intent=(getattr(sess, "seed_intent", "") or "")[:500],
                seed_video_id=(getattr(sess.seed_track, "video_id", "") or ""),
                tracks_json=payload,
                track_count=len(tracks),
                artwork_url=artwork,
            )
            db.add(row)
            await db.flush()
            logger.info(
                "[toup-media] autosaved station user=%s id=%s name=%r tracks=%d",
                user_id[:8], row.id, row.name, len(tracks),
            )
        else:
            row.tracks_json = payload
            row.track_count = len(tracks)
            if artwork:
                row.artwork_url = artwork
            row.updated_at = datetime.utcnow()
        await db.commit()
        sess.autosave_playlist_id = row.id


def _capture_live_station(user_id: str, channel: str):
    """Snapshot the live RadioSession's full tape + remaining queue, in play
    order, deduped, capped. Returns (session, tracks) or (None, [])."""
    from app.agent.radio import get_radio_manager

    mgr = get_radio_manager()
    sess = mgr.get(user_id, channel)
    if sess is None or not sess.enabled:
        return None, []
    tracks: List[dict] = []
    seen: set = set()

    def _add(t) -> None:
        if not getattr(t, "video_id", "") or t.video_id in seen:
            return
        seen.add(t.video_id)
        tracks.append(_station_track_dict(t))

    # The tail of the history, never the head. played_history holds up to 200
    # entries and the cap here is 60, so a naive "history + current + queue"
    # then [:60] on a long-running station saves ONLY the oldest tracks — the
    # song playing right now and everything queued behind it get cut, and the
    # user gets back a playlist of what they listened to an hour ago. Budget
    # the current track and the queue FIRST; history fills whatever is left,
    # taking the most recent.
    upcoming: List = list(sess.playlist[sess.playlist_cursor:])
    head: List = ([sess.current_station_track] if sess.current_station_track is not None else [])
    # Leave at least a third of the playlist for where the station is going.
    reserved = len(head) + min(len(upcoming), max(_MAX_TRACKS // 3, 1))
    history_budget = max(_MAX_TRACKS - reserved, 0)
    for t in list(sess.played_history)[-history_budget:] if history_budget else []:
        _add(t)
    for t in head:
        _add(t)
    for t in upcoming:
        _add(t)
    return sess, tracks[:_MAX_TRACKS]


# ── Routes ────────────────────────────────────────────────────────────────
# NOTE: /live is declared BEFORE /{playlist_id} — route-shadowing, same as
# memories' /breakdown comment.

@router.get("", response_model=List[PlaylistSummaryResponse])
async def list_playlists(
    media_type: str = "music",
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    proxy = await _agent_proxy_info(current_user.id, db)
    agent_unreachable = False
    if proxy:
        remote = await _proxy_read(proxy[0], proxy[1], f"?media_type={media_type}")
        if remote is not None:
            return remote
        agent_unreachable = True
        # fall through to the platform mirror (read fallback)
    rows = (await db.execute(
        select(MediaPlaylist)
        .where(MediaPlaylist.user_id == current_user.id, MediaPlaylist.media_type == media_type)
        .order_by(MediaPlaylist.created_at.desc())
        .limit(200)
    )).scalars().all()
    if agent_unreachable and not rows:
        # Playlists are written through to the tenant agent DB, so the platform
        # mirror is empty for most users by construction. Returning [] here
        # would render a full library as "No playlists yet" — data loss is what
        # that looks like to the user, and they'd have no reason to retry. An
        # explicit failure is the honest answer; the mirror is still used when
        # it actually holds something.
        raise HTTPException(status_code=503, detail="agent_unreachable")
    return [_summary(r) for r in rows]


@router.get("/live", response_model=LiveStationResponse)
async def live_station(
    channel: str = "app",
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Snapshot of the live station (what "Save this station" would capture)."""
    proxy = await _agent_proxy_info(current_user.id, db)
    if proxy:
        remote = await _proxy_read(proxy[0], proxy[1], f"/live?channel={channel}")
        if remote is not None:
            return remote
        return LiveStationResponse(active=False)
    if not _serving_locally():
        return LiveStationResponse(active=False)
    sess, tracks = _capture_live_station(current_user.id, channel)
    if sess is None:
        return LiveStationResponse(active=False)
    current = None
    if sess.current_station_track is not None:
        current = PlaylistTrack(**_station_track_dict(sess.current_station_track))
    elif sess.current_track_id:
        current = PlaylistTrack(video_id=sess.current_track_id)
    return LiveStationResponse(
        active=True,
        seed_intent=sess.seed_intent or None,
        current=current,
        track_count=len(tracks),
    )


@router.post("", response_model=PlaylistResponse)
async def create_playlist(
    payload: PlaylistCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    proxy = await _agent_proxy_info(current_user.id, db)
    if proxy:
        remote = await _proxy_write(
            proxy[0], proxy[1], "", "POST", payload.model_dump(exclude_none=True),
        )
        if remote is not None:
            return remote
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Agent returned no playlist")

    seed_intent: Optional[str] = None
    if payload.tracks:
        tracks = [t.model_dump(exclude_none=True) for t in payload.tracks if t.video_id][:_MAX_TRACKS]
        source = "manual"
    elif payload.from_channel:
        if not _serving_locally():
            # A capture only means anything where the radio manager runs.
            raise HTTPException(
                status.HTTP_502_BAD_GATEWAY,
                "Could not reach your agent to capture the station.",
            )
        sess, tracks = _capture_live_station(current_user.id, payload.from_channel)
        if sess is None or not tracks:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                "Nothing is playing — start some music first.",
            )
        seed_intent = sess.seed_intent or None
    else:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "Provide tracks or from_channel.",
        )
    if not tracks:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Playlist has no tracks.")

    name = (payload.name or "").strip() or (seed_intent or "").strip() or "My station"
    row = MediaPlaylist(
        user_id=current_user.id,
        name=name[:200],
        media_type=(payload.media_type or "music")[:20],
        source="manual" if payload.tracks else "radio",
        seed_intent=seed_intent,
        tracks_json=json.dumps(tracks, ensure_ascii=False),
        track_count=len(tracks),
        artwork_url=(tracks[0].get("thumbnail_url") or None) if tracks else None,
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    logger.info(
        "[toup-media] playlist saved user=%s id=%s tracks=%d source=%s",
        current_user.id[:8], row.id, row.track_count, row.source,
    )
    return _full(row)


@router.get("/{playlist_id}", response_model=PlaylistResponse)
async def get_playlist(
    playlist_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    proxy = await _agent_proxy_info(current_user.id, db)
    agent_unreachable = False
    if proxy:
        remote = await _proxy_read(proxy[0], proxy[1], f"/{playlist_id}")
        if remote is not None:
            return remote
        agent_unreachable = True
    row = (await db.execute(
        select(MediaPlaylist).where(
            MediaPlaylist.id == playlist_id, MediaPlaylist.user_id == current_user.id,
        )
    )).scalar_one_or_none()
    if row is None:
        # Same honesty rule as the list route: the mirror is empty by
        # construction, so "not in the mirror" after a failed proxy read means
        # we could not ask, not that the playlist is gone. Telling the user
        # their playlist no longer exists would be a lie with a delete's weight.
        if agent_unreachable:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "agent_unreachable")
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Playlist not found")
    return _full(row)


@router.patch("/{playlist_id}", response_model=PlaylistSummaryResponse)
async def rename_playlist(
    playlist_id: str,
    payload: PlaylistUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    proxy = await _agent_proxy_info(current_user.id, db)
    if proxy:
        remote = await _proxy_write(
            proxy[0], proxy[1], f"/{playlist_id}", "PATCH", payload.model_dump(),
        )
        if remote is not None:
            return remote
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Agent returned no playlist")
    row = (await db.execute(
        select(MediaPlaylist).where(
            MediaPlaylist.id == playlist_id, MediaPlaylist.user_id == current_user.id,
        )
    )).scalar_one_or_none()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Playlist not found")
    row.name = payload.name.strip()[:200]
    await db.commit()
    await db.refresh(row)
    return _summary(row)


@router.delete("/{playlist_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_playlist(
    playlist_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    proxy = await _agent_proxy_info(current_user.id, db)
    if proxy:
        await _proxy_write(proxy[0], proxy[1], f"/{playlist_id}", "DELETE")
        return None
    row = (await db.execute(
        select(MediaPlaylist).where(
            MediaPlaylist.id == playlist_id, MediaPlaylist.user_id == current_user.id,
        )
    )).scalar_one_or_none()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Playlist not found")
    await db.delete(row)
    await db.commit()
    return None


@router.delete("/{playlist_id}/tracks/{video_id}", response_model=PlaylistResponse)
async def delete_playlist_track(
    playlist_id: str,
    video_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Drop one song from a saved playlist — the curation half of the library.

    Returns the FULL playlist so the client renders the authoritative result
    rather than its own optimistic guess about what is left.
    """
    proxy = await _agent_proxy_info(current_user.id, db)
    if proxy:
        remote = await _proxy_write(
            proxy[0], proxy[1], f"/{playlist_id}/tracks/{video_id}", "DELETE",
        )
        if remote is not None:
            return remote
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Agent returned no playlist")
    if not _serving_locally():
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY, "Could not reach your agent to edit this playlist.",
        )
    row = (await db.execute(
        select(MediaPlaylist).where(
            MediaPlaylist.id == playlist_id, MediaPlaylist.user_id == current_user.id,
        )
    )).scalar_one_or_none()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Playlist not found")
    tracks = _parse_tracks(row.tracks_json)
    kept = [t for t in tracks if t.get("video_id") != video_id]
    if len(kept) == len(tracks):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Track not in this playlist")
    row.tracks_json = json.dumps(kept)
    row.track_count = len(kept)
    # The list artwork came from the first track; if that is what was removed,
    # re-derive it rather than leaving a thumbnail of a song no longer here.
    row.artwork_url = next((t.get("thumbnail_url") for t in kept if t.get("thumbnail_url")), None)
    # A hand-curated playlist is no longer a passive recording of a station:
    # promote it so a later auto-capture of the same seed cannot overwrite the
    # user's edits.
    if row.source == "auto":
        row.source = "radio"
    await db.commit()
    await db.refresh(row)
    logger.info(
        "[toup-media] track removed user=%s playlist=%s video=%s left=%d",
        current_user.id[:8], row.id, video_id, len(kept),
    )
    return _full(row)


@router.post("/{playlist_id}/play", response_model=PlaylistPlayResponse)
async def play_playlist(
    playlist_id: str,
    payload: PlaylistPlayRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Start the saved playlist: seed a radio session with the EXACT stored
    tracks in order and broadcast the first one over the chat WS. When the
    stored list runs out the station extends in the same style (the standard
    refill) — a saved playlist ends by drifting, never by stopping.

    The response carries the first track + upcoming window so a screen with no
    chat mounted can drive the native player directly (the voice-bridge
    pattern) instead of hoping a frame subscriber exists.
    """
    proxy = await _agent_proxy_info(current_user.id, db)
    if proxy:
        remote = await _proxy_write(
            proxy[0], proxy[1], f"/{playlist_id}/play", "POST", payload.model_dump(),
        )
        if remote is not None:
            return remote
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Agent returned no play state")
    if not _serving_locally():
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            "Could not reach your agent to start playback.",
        )

    channel = (payload.channel or "app").strip().lower()
    from app.agent.radio import get_radio_manager
    from app.agent.radio.session import RADIO_ALLOWED_CHANNELS, SeedTrack
    from app.agent.radio.playlist import StationTrack

    if channel not in RADIO_ALLOWED_CHANNELS:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"channel {channel!r} not supported")

    row = (await db.execute(
        select(MediaPlaylist).where(
            MediaPlaylist.id == playlist_id, MediaPlaylist.user_id == current_user.id,
        )
    )).scalar_one_or_none()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Playlist not found")
    tracks = [t for t in _parse_tracks(row.tracks_json) if t.get("video_id")]
    if not tracks:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Playlist is empty.")

    # Start where the user tapped, keeping the rest of the playlist behind it.
    start_at = 0
    if payload.start_video_id:
        for i, t in enumerate(tracks):
            if t.get("video_id") == payload.start_video_id:
                start_at = i
                break
    tracks = tracks[start_at:] + tracks[:start_at]
    first = tracks[0]
    station = [
        StationTrack(
            video_id=t["video_id"],
            title=t.get("title") or "",
            artist=t.get("artist") or "",
            length=t.get("length") or "",
            video_type=t.get("video_type") or "",
            thumbnail_url=t.get("thumbnail_url") or "",
        )
        for t in tracks[1:]
    ]
    mgr = get_radio_manager()
    sess = mgr.enable(
        user_id=current_user.id,
        channel=channel,
        seed_intent=row.name,
        seed_track=SeedTrack(video_id=first["video_id"], title=first.get("title") or row.name),
        station=station,
        source="playlist_play",
    )
    # A saved playlist is an audio library object — pin the surface so the
    # variant machinery and the phone agree (same rule as the voice bridge).
    if channel == "app":
        mgr.set_display_mode(sess, "song", user_initiated=True, source="playlist_play")
    # This station IS a library entry already. Without this, the first advance
    # would auto-capture it a second time under a generated name, and every
    # replay would add another near-duplicate of the same playlist.
    sess.autosave_suppressed = True

    from app.api.ws_chat import (
        _upcoming_tracks, _resolve_upcoming_variants, broadcast_to_user,
    )
    from app.agent.radio.player import broadcast_radio_track

    # Resolve the window's variants BEFORE shipping it — every other producer
    # of an upcoming window does (the toggle, the mode flip, the auto-advance).
    # Skipping it means the phone prefetches the ids stored in the playlist
    # while the pop-time swap hands out different ones for the same songs: the
    # prefetched files are wasted, every advance is a cold round-trip, and the
    # track that plays isn't the id the client was told to expect.
    try:
        await _resolve_upcoming_variants(sess)
    except Exception as e:  # a window is an accelerator, never a blocker
        logger.warning("[toup-media] variant resolve failed for %s: %s", row.id, e)
    upcoming = _upcoming_tracks(sess)
    await broadcast_radio_track(
        user_id=current_user.id,
        video_id=first["video_id"],
        title=first.get("title") or row.name,
        channel=channel,
        artist=first.get("artist") or "",
        thumbnail_url=first.get("thumbnail_url") or "",
        video_type=first.get("video_type") or "",
        reason="playlist_play",
        upcoming=upcoming,
    )
    await broadcast_to_user(current_user.id, sess.to_broadcast_dict())
    logger.info(
        "[toup-media] playlist play user=%s id=%s name=%r tracks=%d",
        current_user.id[:8], row.id, row.name, row.track_count,
    )
    return PlaylistPlayResponse(
        ok=True,
        first=PlaylistTrack(**{k: first.get(k) for k in (
            "video_id", "title", "artist", "thumbnail_url", "length", "video_type",
        )}),
        upcoming=[PlaylistTrack(**{k: u.get(k) for k in (
            "video_id", "title", "artist", "thumbnail_url",
        )}) for u in upcoming],
    )
