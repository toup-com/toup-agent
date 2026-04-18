"""Radio Mode session store — in-memory, per-(user, channel).

Lives in the agent process memory. Survives request boundaries but not
process restarts. Acceptable for v1 — the user can re-toggle after a
restart. Promote to Redis if restart-survival becomes a requirement.

Channels: web, telegram, discord, slack, app. Voice is explicitly excluded.

The session holds a pre-built **station** (YT Music Song Radio queue) and
a cursor into it. Each `media_ended` pops the next unplayed track. When the
cursor approaches the end, the station is extended from the currently
playing track's id as a new seed — that's how a long listen drifts.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from app.agent.radio.playlist import StationTrack

logger = logging.getLogger(__name__)

RADIO_ALLOWED_CHANNELS = frozenset({"web", "telegram", "discord", "slack", "app"})

MAX_CONSECUTIVE_FAILURES = 3
MAX_PLAYED_HISTORY = 200           # defensive dedupe window
PLAYLIST_REFILL_THRESHOLD = 3      # rebuild from current track when this many tracks remain


@dataclass
class SeedTrack:
    video_id: str
    title: str


@dataclass
class RadioSession:
    user_id: str
    channel: str
    enabled: bool = False
    seed_intent: str = ""                           # user's original phrasing
    seed_track: Optional[SeedTrack] = None          # the first video that seeded the session
    current_track_id: Optional[str] = None
    # The YT Music station — populated on enable, extended when running low.
    playlist: List[StationTrack] = field(default_factory=list)
    playlist_cursor: int = 0                        # next index to try on media_ended
    # Defensive dedupe only — YT Music rarely repeats within one queue, but
    # extension can append tracks already played earlier in the session.
    played_track_ids: set = field(default_factory=set)
    consecutive_failures: int = 0
    last_activity_ts: float = field(default_factory=time.time)

    def to_broadcast_dict(self) -> dict:
        return {
            "type": "radio_state",
            "channel": self.channel,
            "enabled": self.enabled,
            "seed_intent": self.seed_intent,
            "current_track_id": self.current_track_id,
        }


class RadioSessionManager:
    def __init__(self) -> None:
        self._sessions: Dict[Tuple[str, str], RadioSession] = {}
        self._lock = asyncio.Lock()

    @staticmethod
    def is_channel_allowed(channel: Optional[str]) -> bool:
        if not channel:
            return False
        return channel.lower() in RADIO_ALLOWED_CHANNELS

    def get(self, user_id: str, channel: str) -> Optional[RadioSession]:
        return self._sessions.get((user_id, channel))

    def get_or_create(self, user_id: str, channel: str) -> RadioSession:
        key = (user_id, channel)
        sess = self._sessions.get(key)
        if sess is None:
            sess = RadioSession(user_id=user_id, channel=channel)
            self._sessions[key] = sess
        return sess

    def disable(self, user_id: str, channel: str) -> Optional[RadioSession]:
        sess = self._sessions.get((user_id, channel))
        if sess is None:
            return None
        sess.enabled = False
        sess.last_activity_ts = time.time()
        logger.info("[radio] disabled user=%s channel=%s", user_id[:8], channel)
        return sess

    def enable(
        self,
        user_id: str,
        channel: str,
        seed_intent: str,
        seed_track: SeedTrack,
        station: List[StationTrack],
    ) -> RadioSession:
        """Turn radio on with a freshly-built station."""
        sess = self.get_or_create(user_id, channel)
        sess.enabled = True
        sess.seed_intent = seed_intent or sess.seed_intent
        sess.seed_track = seed_track
        sess.current_track_id = seed_track.video_id
        sess.consecutive_failures = 0
        sess.playlist = list(station)
        sess.playlist_cursor = 0
        sess.played_track_ids = {seed_track.video_id}
        sess.last_activity_ts = time.time()
        logger.info(
            "[radio] enabled user=%s channel=%s seed_intent=%r seed_title=%r "
            "station_size=%d",
            user_id[:8], channel, seed_intent, seed_track.title, len(station),
        )
        return sess

    def record_user_seed(
        self,
        user_id: str,
        channel: str,
        seed_intent: str,
        seed_track: SeedTrack,
    ) -> RadioSession:
        """User directly played a track (not via radio auto-pick).

        Radio toggle flips OFF if the intent changed meaningfully — the new
        user intent is treated as "end of this station, start fresh." The
        actual new station is built on the next explicit radio_toggle=true.
        """
        sess = self.get_or_create(user_id, channel)
        was_enabled = sess.enabled
        prior_intent = (sess.seed_intent or "").strip().lower()
        new_intent = (seed_intent or "").strip().lower()
        unrelated = was_enabled and prior_intent and new_intent and prior_intent != new_intent
        if unrelated:
            sess.enabled = False
            logger.info(
                "[radio] new unrelated intent; disabling user=%s channel=%s old=%r new=%r",
                user_id[:8], channel, prior_intent, new_intent,
            )
        sess.seed_intent = seed_intent
        sess.seed_track = seed_track
        sess.current_track_id = seed_track.video_id
        sess.consecutive_failures = 0
        # Reset station state — it gets rebuilt on next enable.
        sess.playlist = []
        sess.playlist_cursor = 0
        sess.played_track_ids = {seed_track.video_id}
        sess.last_activity_ts = time.time()
        return sess

    def record_auto_play(self, sess: RadioSession, track: StationTrack) -> None:
        """Mark `track` as the now-playing track and bump the cursor."""
        sess.current_track_id = track.video_id
        sess.played_track_ids.add(track.video_id)
        sess.consecutive_failures = 0
        sess.last_activity_ts = time.time()
        # Cap memory usage of the dedupe set over a very long session.
        if len(sess.played_track_ids) > MAX_PLAYED_HISTORY:
            # Drop arbitrary oldest entries — set ordering is insertion-order
            # in CPython 3.7+, which is close enough for this purpose.
            to_drop = len(sess.played_track_ids) - MAX_PLAYED_HISTORY
            for vid in list(sess.played_track_ids)[:to_drop]:
                sess.played_track_ids.discard(vid)

    def pop_next_from_playlist(self, sess: RadioSession) -> Optional[StationTrack]:
        """Return the next unplayed track from the station, or None if exhausted.

        Advances the cursor past anything already played (defensive — extension
        can append tracks the user heard earlier this session).
        """
        while sess.playlist_cursor < len(sess.playlist):
            track = sess.playlist[sess.playlist_cursor]
            sess.playlist_cursor += 1
            if track.video_id in sess.played_track_ids:
                continue
            return track
        return None

    def extend_playlist(self, sess: RadioSession, new_tracks: List[StationTrack]) -> int:
        """Append `new_tracks` that haven't been played yet. Returns count added."""
        before = len(sess.playlist)
        added = 0
        for t in new_tracks:
            if t.video_id in sess.played_track_ids:
                continue
            # Also skip anything already queued further down the list.
            if any(existing.video_id == t.video_id for existing in sess.playlist[sess.playlist_cursor:]):
                continue
            sess.playlist.append(t)
            added += 1
        logger.info(
            "[radio] playlist extended user=%s channel=%s before=%d added=%d total=%d",
            sess.user_id[:8], sess.channel, before, added, len(sess.playlist),
        )
        return added

    def record_failure(self, sess: RadioSession) -> bool:
        """Bump the failure counter. Returns True if the session was disabled as fail-safe."""
        sess.consecutive_failures += 1
        sess.last_activity_ts = time.time()
        if sess.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            sess.enabled = False
            logger.warning(
                "[radio] fail-safe disabled user=%s channel=%s after %d failures",
                sess.user_id[:8], sess.channel, sess.consecutive_failures,
            )
            return True
        return False


_singleton: Optional[RadioSessionManager] = None


def get_radio_manager() -> RadioSessionManager:
    global _singleton
    if _singleton is None:
        _singleton = RadioSessionManager()
    return _singleton
