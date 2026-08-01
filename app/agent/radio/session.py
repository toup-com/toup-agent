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
    # Ordered tape of what's actually been played this session. Used by
    # skip_prev to walk back, and by skip_next to re-enter the tape before
    # advancing the playlist. Distinct from `played_track_ids` (which is
    # an unordered dedupe set).
    played_history: List[StationTrack] = field(default_factory=list)
    history_cursor: int = -1                        # position in played_history; -1 = empty
    # Defensive dedupe only — YT Music rarely repeats within one queue, but
    # extension can append tracks already played earlier in the session.
    played_track_ids: set = field(default_factory=set)
    # "video" (default) or "song". Decides whether the frontend renders the
    # full YouTube iframe (Video) or the audio-centric overlay (Song).
    # Default is Video because Toup plays in the content area — visual-first
    # is the common expected behavior. Users who want Song mode toggle into
    # it and the override persists for the rest of the session.
    display_mode: str = "video"
    # True once the user has explicitly clicked the Song/Video pill at least
    # once in this session. Mid-track auto-advance swaps (Song+OMV → ATV /
    # Video+ATV → OMV) only fire when this is True — default-mode advances
    # play whatever the queue gives. Resets to False on enable() /
    # record_user_seed (new seed, new override context).
    display_mode_user_override: bool = False
    # Full StationTrack for the currently-playing entry (kept alongside
    # `current_track_id` so handlers can access artist / video_type /
    # thumbnail without another YT Music roundtrip — specifically for the
    # Topic-swap path in _handle_radio_display_mode).
    current_station_track: Optional[StationTrack] = None
    consecutive_failures: int = 0
    last_activity_ts: float = field(default_factory=time.time)
    # When we last told this session's client the station ran dry. Both clients
    # render `radio_notice` as a modal alert and neither de-duplicates repeats,
    # so an exhausted station being skipped repeatedly would stack one dialog
    # per tap. Throttled in ws_chat._advance_and_broadcast_next.
    last_exhaustion_notice_ts: float = 0.0
    # ── Track pinning (2026-07-31) ───────────────────────────────────
    # When the current track became current. A `media_ended` arriving a few
    # seconds after that cannot be a track ENDING — no song is 15s long — so it
    # is a client-side artefact (a stalled stream, a replaced player, a dying
    # WebView) and must not advance the station. Without this the user's
    # explicitly requested song was always one spurious frame away from being
    # replaced: "Love The Way You Lie" was preempted ~18s in, and a minute of
    # failed loads walked a Kendrick request all the way to a stranger's track.
    current_track_started_ts: float = 0.0
    # Length of the current track in seconds when known (from YT Music
    # metadata), so the guard can be proportional as well as absolute.
    current_track_length_sec: float = 0.0
    # Last honored end, for idempotency. The client debounces per videoId, but
    # duplicates still arrive from a second connected client, a reconnect
    # replay, or two dispatch paths racing — and each duplicate used to pop
    # another track.
    last_ended_video_id: str = ""
    last_ended_ts: float = 0.0

    @staticmethod
    def parse_length_sec(length: str) -> float:
        """YT Music gives durations as "4:18" / "1:02:33". Return seconds, or
        0.0 when absent or unparseable (the guard then falls back to its
        absolute floor)."""
        parts = (length or "").strip().split(":")
        if not parts or not all(p.isdigit() for p in parts) or len(parts) > 3:
            return 0.0
        total = 0.0
        for p in parts:
            total = total * 60 + int(p)
        return total

    def mark_current_track(self, video_id: Optional[str], length_sec: float = 0.0) -> None:
        """Set the current track AND start its play clock.

        Every writer of `current_track_id` must go through here — the clock is
        what makes the pinning guard in `_handle_media_ended` meaningful, and a
        writer that sets the id without it leaves a track whose apparent start
        is whatever the previous one was (or 0, i.e. "playing since 1970",
        which passes every elapsed check and pins nothing).
        """
        self.current_track_id = video_id
        self.current_track_started_ts = time.time()
        self.current_track_length_sec = max(0.0, float(length_sec or 0.0))

    def to_broadcast_dict(self) -> dict:
        return {
            "type": "radio_state",
            "channel": self.channel,
            "enabled": self.enabled,
            "seed_intent": self.seed_intent,
            # The videoId the station was seeded from. Stable across playlist
            # advances; changes only on explicit re-seed (toggle_on from a
            # different card OR tool_play). Frontend uses this for per-card
            # pill binding — only the card whose videoId matches lights up.
            "seed_video_id": self.seed_track.video_id if self.seed_track else None,
            "current_track_id": self.current_track_id,
            "display_mode": self.display_mode,
            "display_mode_user_override": self.display_mode_user_override,
            # True when the user has nothing earlier to walk back to.
            "can_prev": self.history_cursor > 0,
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

    def disable(self, user_id: str, channel: str, source: str = "toggle_off") -> Optional[RadioSession]:
        sess = self._sessions.get((user_id, channel))
        if sess is None:
            return None
        prev_enabled = sess.enabled
        sess.enabled = False
        sess.last_activity_ts = time.time()
        logger.info(
            "[radio] enabled_mutation source=%s user=%s channel=%s from=%s to=False",
            source, user_id[:8], channel, prev_enabled,
        )
        return sess

    def enable(
        self,
        user_id: str,
        channel: str,
        seed_intent: str,
        seed_track: SeedTrack,
        station: List[StationTrack],
        source: str = "toggle_on",
    ) -> RadioSession:
        """Turn radio on with a freshly-built station."""
        sess = self.get_or_create(user_id, channel)
        prior_seed_vid = sess.seed_track.video_id if sess.seed_track else None
        sess.enabled = True
        sess.seed_intent = seed_intent or sess.seed_intent
        sess.seed_track = seed_track
        sess.mark_current_track(seed_track.video_id)
        sess.consecutive_failures = 0
        sess.playlist = list(station)
        sess.playlist_cursor = 0
        sess.played_track_ids = {seed_track.video_id}
        # Seed the tape with the seed track itself so skip_prev from track 1
        # lands back on the seed (and from the seed is disabled).
        seed_station_track = StationTrack(
            video_id=seed_track.video_id,
            title=seed_track.title,
            artist="",
        )
        sess.played_history = [seed_station_track]
        sess.history_cursor = 0
        sess.display_mode = "video"
        sess.display_mode_user_override = False
        sess.current_station_track = seed_station_track
        sess.last_activity_ts = time.time()
        logger.info(
            "[radio] seed_bind source=%s user=%s channel=%s videoId=%s "
            "intent=%r old_seed=%s station=%d",
            source, user_id[:8], channel, seed_track.video_id,
            seed_intent, prior_seed_vid, len(station),
        )
        return sess

    def record_user_seed(
        self,
        user_id: str,
        channel: str,
        seed_intent: str,
        seed_track: SeedTrack,
        source: str = "tool_play",
    ) -> RadioSession:
        """User directly played a track (not via radio auto-pick).

        Radio toggle flips OFF if the intent changed meaningfully — the new
        user intent is treated as "end of this station, start fresh." The
        actual new station is built on the next explicit radio_toggle=true.
        """
        sess = self.get_or_create(user_id, channel)
        was_enabled = sess.enabled
        prior_seed_vid = sess.seed_track.video_id if sess.seed_track else None
        prior_intent = (sess.seed_intent or "").strip().lower()
        new_intent = (seed_intent or "").strip().lower()
        unrelated = was_enabled and prior_intent and new_intent and prior_intent != new_intent
        logger.info(
            "[radio] station_reseeded by=%s user=%s channel=%s "
            "old_seed=%s new_seed=%s old_intent=%r new_intent=%r "
            "was_enabled=%s unrelated=%s",
            source, user_id[:8], channel,
            prior_seed_vid, seed_track.video_id,
            prior_intent, new_intent, was_enabled, unrelated,
        )
        if unrelated:
            sess.enabled = False
            logger.info(
                "[radio] enabled_mutation source=%s user=%s channel=%s from=True to=False "
                "reason=unrelated_intent",
                source, user_id[:8], channel,
            )
        sess.seed_intent = seed_intent
        sess.seed_track = seed_track
        sess.mark_current_track(seed_track.video_id)
        sess.consecutive_failures = 0
        # Reset station state — it gets rebuilt on next enable.
        sess.playlist = []
        sess.playlist_cursor = 0
        sess.played_track_ids = {seed_track.video_id}
        sess.played_history = []
        sess.history_cursor = -1
        sess.display_mode = "video"
        sess.display_mode_user_override = False
        sess.current_station_track = None
        sess.last_activity_ts = time.time()
        return sess

    def record_auto_play(self, sess: RadioSession, track: StationTrack, source: str = "advance") -> None:
        """Mark `track` as now-playing. Appends to played_history and moves
        history_cursor to the new end. Called after playlist advances — NOT
        called on skip_prev (which re-plays a prior entry)."""
        prev = sess.current_track_id
        sess.mark_current_track(track.video_id, sess.parse_length_sec(track.length))
        sess.current_station_track = track
        sess.played_track_ids.add(track.video_id)
        sess.played_history.append(track)
        sess.history_cursor = len(sess.played_history) - 1
        sess.consecutive_failures = 0
        sess.last_activity_ts = time.time()
        logger.info(
            "[radio] current_track_mutation source=%s user=%s channel=%s "
            "from=%s to=%s history_cursor=%d video_type=%s",
            source, sess.user_id[:8], sess.channel,
            prev, track.video_id, sess.history_cursor, track.video_type,
        )
        # Cap memory usage.
        if len(sess.played_track_ids) > MAX_PLAYED_HISTORY:
            to_drop = len(sess.played_track_ids) - MAX_PLAYED_HISTORY
            for vid in list(sess.played_track_ids)[:to_drop]:
                sess.played_track_ids.discard(vid)
        if len(sess.played_history) > MAX_PLAYED_HISTORY:
            drop = len(sess.played_history) - MAX_PLAYED_HISTORY
            sess.played_history = sess.played_history[drop:]
            sess.history_cursor = len(sess.played_history) - 1

    def skip_prev(self, sess: RadioSession) -> Optional[StationTrack]:
        """Walk one step back in played_history. Returns the track to re-broadcast,
        or None if already at the start. Does NOT append to played_history — this
        is a re-play of something already played."""
        if not sess.played_history or sess.history_cursor <= 0:
            return None
        prev = sess.current_track_id
        sess.history_cursor -= 1
        track = sess.played_history[sess.history_cursor]
        sess.mark_current_track(track.video_id, sess.parse_length_sec(track.length))
        sess.current_station_track = track
        sess.last_activity_ts = time.time()
        logger.info(
            "[radio] current_track_mutation source=skip_prev user=%s channel=%s "
            "from=%s to=%s history_cursor=%d",
            sess.user_id[:8], sess.channel, prev, track.video_id, sess.history_cursor,
        )
        return track

    def step_forward_in_history(self, sess: RadioSession) -> Optional[StationTrack]:
        """If the cursor is behind the tape's end (user hit Prev earlier), step
        forward without touching the playlist. Returns the track or None if
        we're already at the end of history (caller should advance the playlist
        instead)."""
        if sess.history_cursor < len(sess.played_history) - 1:
            prev = sess.current_track_id
            sess.history_cursor += 1
            track = sess.played_history[sess.history_cursor]
            sess.mark_current_track(track.video_id, sess.parse_length_sec(track.length))
            sess.current_station_track = track
            sess.last_activity_ts = time.time()
            logger.info(
                "[radio] current_track_mutation source=skip_next_step_forward "
                "user=%s channel=%s from=%s to=%s history_cursor=%d",
                sess.user_id[:8], sess.channel, prev, track.video_id, sess.history_cursor,
            )
            return track
        return None

    def set_display_mode(
        self,
        sess: RadioSession,
        mode: str,
        user_initiated: bool = False,
        source: str = "user_mode_toggle",
    ) -> None:
        """Set 'song' or 'video'. Other values coerce to 'song'.

        When user_initiated=True, also flips display_mode_user_override so auto-
        detect stops overwriting the user's pick on subsequent track loads.
        """
        prev = sess.display_mode
        sess.display_mode = "video" if mode == "video" else "song"
        if user_initiated:
            sess.display_mode_user_override = True
        sess.last_activity_ts = time.time()
        if prev != sess.display_mode or user_initiated:
            logger.info(
                "[radio] display_mode_mutation source=%s user=%s channel=%s "
                "from=%s to=%s override=%s",
                source, sess.user_id[:8], sess.channel,
                prev, sess.display_mode, sess.display_mode_user_override,
            )

    def maybe_auto_set_mode(self, sess: RadioSession, track: StationTrack) -> str:
        """No-op retained for call-site compatibility.

        Previously auto-detected display mode from StationTrack.video_type
        (ATV → song, OMV → video). Auto-detect was removed on 2026-04-20:
        Video is now the session default, and mode only changes when the
        user explicitly clicks the Song/Video pill. See Rule 9 / 10 in
        docs/skills/radio-mode/SKILL.md.

        Returns the current display_mode unchanged. Callers may keep using
        the return value as "the effective mode to broadcast" — it's just
        always equal to sess.display_mode now.
        """
        return sess.display_mode

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
                "[radio] enabled_mutation source=fail_safe user=%s channel=%s "
                "from=True to=False failures=%d",
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
