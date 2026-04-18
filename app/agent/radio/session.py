"""Radio Mode session store — in-memory, per-(user, channel).

Lives in the agent process memory. Survives request boundaries but not
process restarts. Acceptable for v1 — the user can re-toggle after a
restart. Promote to Redis if restart-survival becomes a requirement.

Channels: web, telegram, discord, slack, app. Voice is explicitly excluded.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

RADIO_ALLOWED_CHANNELS = frozenset({"web", "telegram", "discord", "slack", "app"})

MAX_CONSECUTIVE_FAILURES = 3
MAX_PLAYED_HISTORY = 40


# ── Title normalization ────────────────────────────────────────────────
# Used for fuzzy dedupe: two tracks with the same normalized title are
# treated as the same song even if they have different video IDs.
#
# Strips the cosmetic suffixes/prefixes YouTube uploaders routinely add —
# "(Official Video)", "(Lyrics)", " | NCS Release", "VEVO", etc. Result
# is lowercase, whitespace-collapsed, and safe to compare by equality.

_TITLE_STRIP_PATTERNS = [
    # Parentheticals containing cosmetic tags
    re.compile(
        r'\s*\([^)]*(?:official|video|audio|music|lyric|lyrics|hd|hq|4k|'
        r'remaster|remastered|live|cover|acoustic|explicit|extended|'
        r'visualizer|performance|instrumental|radio\s*edit|clean)\b[^)]*\)',
        re.IGNORECASE,
    ),
    re.compile(
        r'\s*\[[^\]]*(?:official|video|audio|music|lyric|lyrics|hd|hq|4k|'
        r'remaster|remastered|live|cover|acoustic|explicit|extended|'
        r'visualizer|performance|instrumental|radio\s*edit|clean)\b[^\]]*\]',
        re.IGNORECASE,
    ),
    # Trailing uploader / channel marks like " | NoCopyrightSounds", "VEVO"
    re.compile(r'\s*[|\-–—]\s*(?:[A-Z0-9]+VEVO|NoCopyrightSounds|NCS|Topic)\s*$', re.IGNORECASE),
    # Featured artist annotations — keep the base title ("Song ft Artist" → "Song")
    re.compile(r'\s*(?:ft\.?|feat\.?|featuring)\s+[^()[\]{}]*$', re.IGNORECASE),
    # Dangling tags
    re.compile(r'\s*\b(?:HD|HQ|4K|Lyrics|Lyric|Official|Audio)\b\s*$', re.IGNORECASE),
]


def normalize_title(title: Optional[str]) -> str:
    """Return a dedupe-stable key for a track title.

    Empty string (equal to empty-string only) if input is missing/garbage —
    callers should treat empty as "no usable title, don't dedupe by title".
    """
    if not title:
        return ""
    t = title.strip()
    if not t or t.lower() == "youtube video":
        return ""
    for pat in _TITLE_STRIP_PATTERNS:
        t = pat.sub(" ", t)
    # Collapse whitespace & punctuation-only runs
    t = re.sub(r"[\s\-–—_·•|]+", " ", t).strip().lower()
    return t


# ── Query rotation strategies ──────────────────────────────────────────
# As the session picks more tracks, successive search queries get broader
# so the player walks outward from the seed instead of collapsing back.
# iteration is 0-indexed: 0 = first auto-pick after the seed.

def rotate_query(seed_intent: str, iteration: int) -> str:
    """Return the next query string for the given iteration."""
    seed = (seed_intent or "").strip()
    if not seed:
        return ""
    # First dash-separated chunk is usually the artist: "Eminem - Lose Yourself" → "Eminem".
    # If there's no dash, treat the whole thing as the artist/vibe.
    artist = seed.split("-", 1)[0].strip() if "-" in seed else seed
    strategies = [
        seed,                          # 0: verbatim seed — already the picker's happy path
        f"{artist} mix",               # 1: artist's mix playlist — YouTube aggregates these
        f"{artist} playlist",          # 2: another shape of catalog expansion
        f"songs like {seed}",          # 3: pivots to genre/vibe peers
        f"{artist} best songs",        # 4: top catalog for the artist
        f"{artist} radio",             # 5: YouTube's own "radio:" concept
    ]
    return strategies[iteration % len(strategies)]


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
    # Normalized titles for fuzzy dedupe — keep alongside ids so different
    # uploads of the same song (e.g. "Lose Yourself (Official Video)" and
    # "Lose Yourself (Lyrics)") collapse to one entry and get skipped.
    played_titles_normalized: set = field(default_factory=set)
    # Monotonic counter: how many auto-picks have been attempted this
    # session. Used by rotate_query() so successive picks use broader
    # search angles instead of resurrecting the seed track.
    pick_iteration: int = 0
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
        sess.pick_iteration = 0
        # Track the seed in played history so the first auto-pick doesn't
        # re-broadcast the same video (either by id or by normalized title).
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
        sess.played_titles_normalized = set()
        sess.pick_iteration = 0
        self._record_played(sess, seed_track.video_id, seed_track.title)
        sess.last_activity_ts = time.time()
        return sess

    def record_auto_play(self, sess: RadioSession, video_id: str, title: str) -> None:
        sess.current_track_id = video_id
        sess.consecutive_failures = 0
        sess.pick_iteration += 1
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
            norm = normalize_title(title)
            if norm:
                sess.played_titles_normalized.add(norm)
            if len(sess.played_track_ids) > MAX_PLAYED_HISTORY:
                sess.played_track_ids = sess.played_track_ids[-MAX_PLAYED_HISTORY:]
                sess.played_titles = sess.played_titles[-MAX_PLAYED_HISTORY:]


_singleton: Optional[RadioSessionManager] = None


def get_radio_manager() -> RadioSessionManager:
    global _singleton
    if _singleton is None:
        _singleton = RadioSessionManager()
    return _singleton
