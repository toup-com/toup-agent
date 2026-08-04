"""MediaPlaylist — a saved, replayable media queue ("Toup Media" library).

Every music request already builds a radio station server-side; this table is
where a station the user liked becomes durable: exact tracks, exact order,
replayable forever. AGENT_ONLY — it lives in each user's tenant container DB
next to conversations/memories, and the platform reaches it through the same
proxy pattern memories uses (reads fall back to the platform DB, writes
proxy-or-502).

`media_type` is 'music' for everything created today; the column exists so
movies/shows can join the library later without a schema change.

Tracks are an ordered JSON list rather than a join table: a playlist is a
small, atomic, always-read-whole document (≤ ~50 entries of five short
strings), and the radio session consumes it as exactly that shape.
"""

from datetime import datetime
from typing import Optional
import uuid

from sqlalchemy import String, Text, DateTime, Integer, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class MediaPlaylist(Base):
    __tablename__ = "media_playlists"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True)

    name: Mapped[str] = mapped_column(String(200), nullable=False)
    # 'music' today; 'movies' etc. later. Scopes list queries per surface.
    media_type: Mapped[str] = mapped_column(String(20), default="music")
    # 'auto'   = captured automatically when the agent built this station —
    #            the default, so the library holds everything ever played;
    # 'radio'  = the user explicitly saved the live station;
    # 'manual' = composed from an explicit track list.
    source: Mapped[str] = mapped_column(String(20), default="radio")
    # The user's original phrasing that seeded the captured station
    # ("play me some Drake") — the default name and useful provenance.
    seed_intent: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # The video the station was built from. Lets an auto-capture recognise a
    # station it has already recorded instead of stacking near-identical rows
    # each time the user replays the same seed.
    seed_video_id: Mapped[Optional[str]] = mapped_column(String(32), nullable=True, index=True)

    # Ordered JSON list of {video_id, title, artist, thumbnail_url, length,
    # video_type}. Serialized/parsed in app/api/media_playlists.py.
    tracks_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    track_count: Mapped[int] = mapped_column(Integer, default=0)
    # Denormalized first-track artwork for list rendering.
    artwork_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
