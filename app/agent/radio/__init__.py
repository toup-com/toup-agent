"""Radio Mode — continuous vibe-matched playback after a seed track.

Source: YT Music's Song Radio (`ytmusicapi.get_watch_playlist`). Built once
per toggle-on, iterated through on each `media_ended`, extended when low.
No YouTube-search scraping, no LLM picker, no Shorts filtering — the source
is already clean.

Scope: per-channel in-memory sessions (web, telegram, discord, slack, app).
Voice is excluded for v1.
"""
from app.agent.radio.playlist import StationTrack, build_station
from app.agent.radio.session import (
    RadioSession,
    RadioSessionManager,
    SeedTrack,
    get_radio_manager,
    PLAYLIST_REFILL_THRESHOLD,
    RADIO_ALLOWED_CHANNELS,
)

__all__ = [
    "RadioSession",
    "RadioSessionManager",
    "SeedTrack",
    "StationTrack",
    "build_station",
    "get_radio_manager",
    "PLAYLIST_REFILL_THRESHOLD",
    "RADIO_ALLOWED_CHANNELS",
]
