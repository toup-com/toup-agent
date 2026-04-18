"""Radio Mode session store — in-memory, per-(user, channel).

Lives in the agent process memory. Survives request boundaries but not
process restarts. Acceptable for v1 — the user can re-toggle after a
restart. Promote to Redis if restart-survival becomes a requirement.

Channels: web, telegram, discord, slack, app. Voice is explicitly excluded.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

RADIO_ALLOWED_CHANNELS = frozenset({"web", "telegram", "discord", "slack", "app"})

MAX_CONSECUTIVE_FAILURES = 3
MAX_PLAYED_HISTORY = 40


@dataclass
class SeedTrack:
    video_id: str
    title: str


@dataclass
class RadioSession:
    user_id: str
    channel: str
    enabled: bool = False
    seed_intent: str = ""
    seed_track: Optional[SeedTrack] = None
    current_track_id: Optional[str] = None
    played_track_ids: List[str] = field(default_factory=list)
    played_titles: List[str] = field(default_factory=list)
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
    ) -> RadioSession:
        sess = self.get_or_create(user_id, channel)
        sess.enabled = True
        sess.seed_intent = seed_intent or sess.seed_intent
        sess.seed_track = seed_track
        sess.current_track_id = seed_track.video_id
        sess.consecutive_failures = 0
        # Track the seed in played history
        self._record_played(sess, seed_track.video_id, seed_track.title)
        sess.last_activity_ts = time.time()
        logger.info(
            "[radio] enabled user=%s channel=%s seed_intent=%r seed_title=%r",
            user_id[:8], channel, seed_intent, seed_track.title,
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

        Always resets `played_track_ids` so a new seed starts a clean history.
        Keeps `enabled` flag untouched (if toggle was ON, it stays ON with the
        new seed; if it was OFF, it stays OFF). But if the new intent clearly
        differs from the old seed_intent we flip `enabled` OFF per spec:
        "new unrelated media request cancels radio mode".
        """
        sess = self.get_or_create(user_id, channel)
        was_enabled = sess.enabled
        prior_intent = (sess.seed_intent or "").strip().lower()
        new_intent = (seed_intent or "").strip().lower()
        # Coarse "unrelated" check: any change in intent phrasing counts as new.
        # User will see the toggle turn OFF and can re-enable.
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
        sess.played_track_ids = []
        sess.played_titles = []
        self._record_played(sess, seed_track.video_id, seed_track.title)
        sess.last_activity_ts = time.time()
        return sess

    def record_auto_play(self, sess: RadioSession, video_id: str, title: str) -> None:
        sess.current_track_id = video_id
        sess.consecutive_failures = 0
        self._record_played(sess, video_id, title)
        sess.last_activity_ts = time.time()

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

    @staticmethod
    def _record_played(sess: RadioSession, video_id: str, title: str) -> None:
        if video_id and video_id not in sess.played_track_ids:
            sess.played_track_ids.append(video_id)
            sess.played_titles.append(title or "")
            if len(sess.played_track_ids) > MAX_PLAYED_HISTORY:
                sess.played_track_ids = sess.played_track_ids[-MAX_PLAYED_HISTORY:]
                sess.played_titles = sess.played_titles[-MAX_PLAYED_HISTORY:]


_singleton: Optional[RadioSessionManager] = None


def get_radio_manager() -> RadioSessionManager:
    global _singleton
    if _singleton is None:
        _singleton = RadioSessionManager()
    return _singleton
