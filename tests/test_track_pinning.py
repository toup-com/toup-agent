"""An explicitly requested track must not be taken away mid-play.

On 2026-07-31 a founder recording caught the opposite twice in one evening:

  * "Love The Way You Lie", asked for by voice, was replaced ~18 seconds in by
    a station track nobody requested, while the agent's own transcript still
    said the requested song was playing; and
  * "Play me Kendrick Lamar" spent a minute walking through a station — the
    Dynamic Island flipping artwork every ~10 seconds — and settled on a track
    the user had never heard of.

Both had the same shape on the server: `media_ended` was accepted as gospel.
Any frame naming a video with session provenance advanced the cursor, no matter
how long the track had actually been playing, how many identical frames had
just arrived, or how many copies of the handler were running at once. A phone
whose audio stream had stalled emitted exactly such frames, so a failure to
play became an instruction to play something else.

These tests assert the rule the fix encodes: **only a plausible, un-duplicated,
serialized end may advance the station.** They are hermetic — no DB, no
network, no WebSocket — so they run on every change.
"""
from __future__ import annotations

import asyncio
import os
import time

import pytest

os.environ.setdefault("ENVIRONMENT", "test")

from app.agent.radio.playlist import StationTrack  # noqa: E402
from app.agent.radio.session import RadioSession, SeedTrack  # noqa: E402


# ── The play clock ──────────────────────────────────────────────────────
# Everything below depends on knowing when the current track started. If a
# writer sets current_track_id without starting the clock, the guard reads
# "playing since 1970", every elapsed check passes, and nothing is pinned.

def test_mark_current_track_starts_the_clock():
    sess = RadioSession(user_id="u", channel="app")
    assert sess.current_track_started_ts == 0.0
    before = time.time()
    sess.mark_current_track("abc12345678", length_sec=258.0)
    assert sess.current_track_id == "abc12345678"
    assert sess.current_track_started_ts >= before
    assert sess.current_track_length_sec == 258.0


def test_every_current_track_writer_goes_through_the_clock():
    """Regression guard for the whole mechanism.

    Five methods set the current track. A sixth added later that assigns
    `current_track_id` directly would silently disable pinning for its path,
    and no other test would fail. Read the source and insist.
    """
    import inspect
    from app.agent.radio import session as session_mod

    src = inspect.getsource(session_mod)
    offenders = [
        line.strip()
        for line in src.splitlines()
        # the definition inside mark_current_track is the one legitimate writer
        if "current_track_id = " in line and "self.current_track_id = video_id" not in line
    ]
    assert offenders == [], (
        "current_track_id assigned outside mark_current_track — the play clock "
        f"will not be set on that path: {offenders}"
    )


@pytest.mark.parametrize(
    "length,expected",
    [("4:18", 258.0), ("0:45", 45.0), ("1:02:33", 3753.0), ("", 0.0), ("abc", 0.0), (None, 0.0)],
)
def test_parse_length_sec(length, expected):
    assert RadioSession.parse_length_sec(length) == expected


# ── The pinning guard ───────────────────────────────────────────────────

def _session_playing(video_id: str, *, length_sec: float, started_ago: float) -> RadioSession:
    """A session with radio on, one track current, started `started_ago` ago."""
    sess = RadioSession(user_id="u", channel="app", enabled=True)
    sess.seed_track = SeedTrack(video_id=video_id, title="seed")
    sess.playlist = [
        StationTrack(video_id="nextnext001", title="Next Up", artist="Someone", length="3:30"),
    ]
    sess.mark_current_track(video_id, length_sec)
    sess.current_track_started_ts = time.time() - started_ago
    return sess


async def _media_ended(monkeypatch, sess, video_id: str):
    """Drive the real handler against `sess`, capturing whether it advanced."""
    from app.api import ws_chat

    advanced = []

    async def _fake_advance(user_id, channel, s, trigger):
        advanced.append(trigger)
        return True

    monkeypatch.setattr(ws_chat, "_advance_and_broadcast_next", _fake_advance)
    monkeypatch.setattr(ws_chat, "broadcast_to_user", _noop_broadcast)
    monkeypatch.setattr(
        ws_chat, "get_radio_manager", lambda: _StubManager(sess), raising=False
    )

    import app.agent.radio as radio_mod
    monkeypatch.setattr(radio_mod, "get_radio_manager", lambda: _StubManager(sess))

    await ws_chat._handle_media_ended(
        "user-1", {"channel": "app", "video_id": video_id}
    )
    return advanced


async def _noop_broadcast(user_id, payload):
    return 1


class _StubManager:
    def __init__(self, sess):
        self._sess = sess

    def get(self, user_id, channel):
        return self._sess


@pytest.mark.asyncio
async def test_explicit_track_is_pinned_against_an_early_end(monkeypatch):
    """The "Love The Way You Lie" case: an end 18s into a 4:26 track.

    The phone's stream stalled; AVQueuePlayer moved on and the client reported
    an end. Honoring it replaced the song the user had just asked for.
    """
    sess = _session_playing("uelHwf8o7_U", length_sec=266.0, started_ago=18.0)
    advanced = await _media_ended(monkeypatch, sess, "uelHwf8o7_U")
    assert advanced == [], "an 18s 'end' on a 4:26 track must not advance the station"


@pytest.mark.asyncio
async def test_short_end_is_rejected_even_when_the_length_is_unknown(monkeypatch):
    """Fast-path plays carry no duration, so the absolute floor has to hold."""
    sess = _session_playing("tvTRZJ-4EyI", length_sec=0.0, started_ago=9.0)
    advanced = await _media_ended(monkeypatch, sess, "tvTRZJ-4EyI")
    assert advanced == [], "a 9s 'end' with no known duration must not advance"


@pytest.mark.asyncio
async def test_a_real_end_still_advances(monkeypatch):
    """The guard must not break the feature it protects."""
    sess = _session_playing("uelHwf8o7_U", length_sec=266.0, started_ago=264.0)
    advanced = await _media_ended(monkeypatch, sess, "uelHwf8o7_U")
    assert advanced == ["media_ended"], "a track that actually finished must advance"


@pytest.mark.asyncio
async def test_a_long_track_with_unknown_length_advances(monkeypatch):
    sess = _session_playing("someid00001", length_sec=0.0, started_ago=200.0)
    advanced = await _media_ended(monkeypatch, sess, "someid00001")
    assert advanced == ["media_ended"]


# ── Idempotency ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_duplicate_end_for_the_same_track_advances_once(monkeypatch):
    """One physical end, two frames — from a second client, a reconnect replay,
    or the two dispatch paths racing. It used to pop two tracks, which is how
    the phone's card ended up two songs ahead of its own audio."""
    sess = _session_playing("finished0001", length_sec=200.0, started_ago=199.0)
    first = await _media_ended(monkeypatch, sess, "finished0001")
    assert first == ["media_ended"]
    second = await _media_ended(monkeypatch, sess, "finished0001")
    assert second == [], "the same end arriving twice must advance only once"


@pytest.mark.asyncio
async def test_dedup_does_not_block_a_different_track(monkeypatch):
    sess = _session_playing("trackone0001", length_sec=200.0, started_ago=199.0)
    assert await _media_ended(monkeypatch, sess, "trackone0001") == ["media_ended"]
    # A genuinely different track ends right after (the client had it queued).
    sess.mark_current_track("tracktwo0002", 200.0)
    sess.current_track_started_ts = time.time() - 199.0
    sess.seed_track = SeedTrack(video_id="tracktwo0002", title="two")
    assert await _media_ended(monkeypatch, sess, "tracktwo0002") == ["media_ended"]


# ── Single-flight ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_concurrent_ends_for_different_tracks_advance_once(monkeypatch):
    """`_handle_media_ended` is dispatched with create_task from TWO places —
    the main receive loop and the mid-turn stop-watcher.

    Dedup alone cannot cover this: the two frames name DIFFERENT videos (the
    live track, plus a stale echo for one still in `played_history`), so both
    clear the duplicate check. What stops the double-pop is serialization —
    with the handler serialized, the second task runs its guards AFTER the
    first advance has started a new track, and the pinning floor rejects it
    because that track is one second old. Run them concurrently instead and
    both read the pre-advance state and both pop, moving the cursor two tracks
    for one physical end. That desync is what left the founder's phone showing
    one song on its card and playing another on the lock screen.
    """
    from app.api import ws_chat

    sess = _session_playing("finished0001", length_sec=200.0, started_ago=199.0)
    # A recently-played track, so a stale echo for it passes provenance.
    sess.played_history.append(
        StationTrack(video_id="previous0002", title="Previous", artist="X", length="3:20")
    )
    advanced = []

    async def _real_ish_advance(user_id, channel, s, trigger):
        # What a genuine advance does: yield (it awaits network), then make a
        # NEW track current — which restarts the play clock.
        await asyncio.sleep(0.05)
        advanced.append(trigger)
        s.mark_current_track("advanced0003", 200.0)
        return True

    monkeypatch.setattr(ws_chat, "_advance_and_broadcast_next", _real_ish_advance)
    monkeypatch.setattr(ws_chat, "broadcast_to_user", _noop_broadcast)
    import app.agent.radio as radio_mod
    monkeypatch.setattr(radio_mod, "get_radio_manager", lambda: _StubManager(sess))

    await asyncio.gather(
        ws_chat._handle_media_ended("user-1", {"channel": "app", "video_id": "finished0001"}),
        ws_chat._handle_media_ended("user-1", {"channel": "app", "video_id": "previous0002"}),
    )
    assert len(advanced) == 1, (
        f"one physical end must move the cursor once — got {len(advanced)} advances"
    )


@pytest.mark.asyncio
async def test_skip_is_never_gated_by_the_pinning_guard():
    """The guard is scoped to media_ended, the only PASSIVE trigger.

    A user pressing ⏭ one second into a track is authoritative and must work
    immediately — pinning protects the user's choice, it does not overrule it.
    """
    import inspect
    from app.api import ws_chat

    src = inspect.getsource(ws_chat._handle_media_ended_locked)
    assert "track pinned" in src, "the guard should live in the media_ended path"
    skip_src = inspect.getsource(ws_chat._handle_radio_skip_next)
    assert "track pinned" not in skip_src and "_MIN_TRACK_PLAY_SEC" not in skip_src, (
        "skip_next must not inherit the pinning guard — an explicit skip is "
        "authoritative at any position"
    )
