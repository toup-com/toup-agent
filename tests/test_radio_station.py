"""Radio Mode station engine — the first tests this subsystem has ever had.

Radio Mode has shipped for months with zero automated coverage (`ls tests |
grep -i radio` was empty before this file), and it has been quietly losing
listeners the whole time. Two of the three bugs covered here were invisible
in logs, invisible in `git log`, and only found by reading the code against
the spec:

  * a station that ran dry told the user NOTHING the first time, and could
    never reach the third-failure counter that would have told them
  * the search-based recovery for non-catalog music — the branch whose own
    docstring names Persian tracks — was unreachable on every refill because
    the title it keys on was not threaded through
  * `media_ended`, the only frame that advances a playlist, was silently
    dropped whenever it arrived while the agent happened to be talking

All three are behaviours, not implementation details, so that is what these
assert. They are deliberately hermetic — no DB, no network, no WebSocket —
because the value is in running on every change, not in fidelity to the
transport.
"""
from __future__ import annotations

import asyncio
import os

import pytest

os.environ.setdefault("ENVIRONMENT", "test")

from app.agent.radio.playlist import StationTrack  # noqa: E402
from app.agent.radio.session import (  # noqa: E402
    MAX_CONSECUTIVE_FAILURES,
    RADIO_ALLOWED_CHANNELS,
    RadioSession,
    SeedTrack,
)


# ── Channel contract ────────────────────────────────────────────────────
# The mobile client sends every radio frame on channel 'app' (api.ts
# MY_CHANNEL). Dropping it from this set would disable Radio Mode on the
# phone entirely, with no error anywhere — the handlers just `return`.
def test_app_channel_is_allowed():
    assert "app" in RADIO_ALLOWED_CHANNELS


def test_voice_channel_is_not_allowed():
    # Documents a real limitation rather than asserting a preference: a
    # voice turn runs the agent with channel="voice", so `_tool_play_media`
    # skips its radio seeding. The mobile app compensates by re-seeding on
    # channel 'app' itself (useVoiceMediaBridge / ChatScreen). If this ever
    # changes, that client-side compensation needs revisiting.
    assert "voice" not in RADIO_ALLOWED_CHANNELS


# ── Exhaustion notice ───────────────────────────────────────────────────
def _session() -> RadioSession:
    s = RadioSession(user_id="u" * 32, channel="app")
    s.enabled = True
    s.seed_track = SeedTrack(video_id="seed12345ab", title="Dokhtare Ahvazi")
    s.current_track_id = "seed12345ab"
    s.current_station_track = StationTrack(
        video_id="seed12345ab", title="Dokhtare Ahvazi", artist="Sandy",
    )
    return s


@pytest.mark.asyncio
async def test_first_exhaustion_notifies_the_user(monkeypatch):
    """The FIRST time a station runs dry the user must hear about it.

    Regression: the notice used to be gated on `record_failure` returning
    True, i.e. the third CONSECUTIVE failure — a count unreachable from
    natural playback, because advancing the counter needs a `media_ended`,
    which needs a track to play, which needs a successful advance, which
    resets the counter to zero. So the first exhaustion broadcast nothing at
    all and the session sat enabled forever over a station that would never
    move again.
    """
    import app.api.ws_chat as ws

    sent: list[dict] = []

    async def _fake_broadcast(uid, frame):
        sent.append(frame)
        return 1

    async def _empty_station(*a, **k):
        return None, []

    monkeypatch.setattr(ws, "broadcast_to_user", _fake_broadcast)
    monkeypatch.setattr("app.agent.radio.build_station", _empty_station)

    sess = _session()
    ok = await ws._advance_and_broadcast_next(sess.user_id, "app", sess, "media_ended")

    assert ok is False
    assert sess.consecutive_failures == 1, "should be the FIRST failure, not the third"
    notices = [f for f in sent if f.get("type") == "radio_notice"]
    assert len(notices) == 1, f"expected exactly one notice, got {sent}"
    assert "Ran out of tracks" in notices[0]["message"]


@pytest.mark.asyncio
async def test_exhaustion_notice_is_throttled(monkeypatch):
    """Holding ⏭ on a dead station must not stack a dialog per tap.

    Both clients render `radio_notice` as a modal alert and neither
    de-duplicates, and the mobile skip queue deliberately does not drop
    repeats — so an unthrottled notice becomes a pile of dialogs the user
    has to dismiss one by one.
    """
    import app.api.ws_chat as ws

    sent: list[dict] = []

    async def _fake_broadcast(uid, frame):
        sent.append(frame)
        return 1

    async def _empty_station(*a, **k):
        return None, []

    monkeypatch.setattr(ws, "broadcast_to_user", _fake_broadcast)
    monkeypatch.setattr("app.agent.radio.build_station", _empty_station)

    sess = _session()
    for _ in range(5):
        await ws._advance_and_broadcast_next(sess.user_id, "app", sess, "skip_next")

    notices = [f for f in sent if f.get("type") == "radio_notice"]
    assert len(notices) == 1, f"{len(notices)} notices for 5 rapid skips"


@pytest.mark.asyncio
async def test_disabling_exhaustion_also_sends_radio_state(monkeypatch):
    """On the Nth failure the session really does turn off, and says so."""
    import app.api.ws_chat as ws

    sent: list[dict] = []

    async def _fake_broadcast(uid, frame):
        sent.append(frame)
        return 1

    async def _empty_station(*a, **k):
        return None, []

    monkeypatch.setattr(ws, "broadcast_to_user", _fake_broadcast)
    monkeypatch.setattr("app.agent.radio.build_station", _empty_station)

    sess = _session()
    for _ in range(MAX_CONSECUTIVE_FAILURES):
        await ws._advance_and_broadcast_next(sess.user_id, "app", sess, "skip_next")

    assert sess.enabled is False
    states = [f for f in sent if f.get("type") == "radio_state"]
    assert states and states[-1]["error"] == "no_more_tracks"


# ── Station extension keys ──────────────────────────────────────────────
@pytest.mark.asyncio
async def test_extend_threads_title_and_artist(monkeypatch):
    """The refill must carry title+artist, or non-catalog music dies.

    `_build_station_fallback` — the recovery path for seeds YT Music has no
    Song-radio for, which its docstring names Persian tracks as the case for
    — computes `f"{title} {artist}"` and returns nothing at all when that is
    empty. The extend used to pass only the video id, so the fallback was
    unreachable on EVERY refill and a regional-music station died at its
    first threshold crossing.
    """
    import app.api.ws_chat as ws

    calls: list[dict] = []

    async def _capture(seed, limit=50, *, seed_title="", seed_artist="", variety=False):
        calls.append({"seed": seed, "title": seed_title, "artist": seed_artist})
        return None, []

    async def _fake_broadcast(uid, frame):
        return 1

    monkeypatch.setattr(ws, "broadcast_to_user", _fake_broadcast)
    monkeypatch.setattr("app.agent.radio.build_station", _capture)

    sess = _session()
    await ws._advance_and_broadcast_next(sess.user_id, "app", sess, "media_ended")

    assert calls, "extend never ran"
    assert calls[0]["title"] == "Dokhtare Ahvazi"
    assert calls[0]["artist"] == "Sandy"


@pytest.mark.asyncio
async def test_extend_drops_placeholder_title(monkeypatch):
    """A placeholder title is worse than none.

    `_tool_play_media` initialises `video_title` to "YouTube Video" and
    leaves it there whenever its title regex misses. Searching YT Music for
    that string returns real songs — arbitrary ones — so threading it would
    extend the station with tracks unrelated to anything the user asked for.
    """
    import app.api.ws_chat as ws

    calls: list[dict] = []

    async def _capture(seed, limit=50, *, seed_title="", seed_artist="", variety=False):
        calls.append({"title": seed_title, "artist": seed_artist})
        return None, []

    async def _fake_broadcast(uid, frame):
        return 1

    monkeypatch.setattr(ws, "broadcast_to_user", _fake_broadcast)
    monkeypatch.setattr("app.agent.radio.build_station", _capture)

    sess = _session()
    sess.current_station_track = StationTrack(
        video_id="abc12345678", title="YouTube Video", artist="",
    )
    sess.seed_track = SeedTrack(video_id="abc12345678", title="YouTube Video")
    await ws._advance_and_broadcast_next(sess.user_id, "app", sess, "media_ended")

    assert calls, "extend never ran"
    assert calls[0]["title"] == "", f"placeholder leaked: {calls[0]}"


# ── Mid-turn passthrough ────────────────────────────────────────────────
def test_media_ended_survives_a_streaming_turn():
    """`media_ended` must not be eaten by the mid-turn stop-watcher.

    That task owns the socket's single receive stream for the whole turn, so
    anything it does not explicitly forward is read and dropped. For
    `media_ended` that ends the station outright: it is the only frame that
    advances a playlist, so one swallowed frame means nothing queued, no
    further `media_ended` possible, and a radio pill still lit over a dead
    station. This is "the music just stopped after one song".
    """
    import app.api.ws_chat as ws

    assert "media_ended" in ws._MID_TURN_PASSTHROUGH
    for t in ("radio_skip_next", "radio_skip_prev", "radio_display_mode"):
        assert t in ws._MID_TURN_PASSTHROUGH


def test_radio_toggle_is_deliberately_not_passed_through():
    """…but `radio_toggle` must NOT be, and that is not an oversight.

    A toggle rebuilds the station from a new seed, and mid-turn is exactly
    when the agent is about to broadcast its own `media_play` for the song it
    just found — so forwarding it races two stations for one request. Both
    clients already defer their toggle to the end of the turn for this reason
    (mobile: `pendingReseedRef`).
    """
    import app.api.ws_chat as ws

    assert "radio_toggle" not in ws._MID_TURN_PASSTHROUGH


def test_dispatch_radio_frame_swallows_handler_errors(monkeypatch):
    """A failing radio frame must never take down the turn carrying it."""
    import app.api.ws_chat as ws

    async def _boom(uid, msg):
        raise RuntimeError("yt music exploded")

    monkeypatch.setattr(ws, "_handle_media_ended", _boom)
    asyncio.get_event_loop_policy()
    asyncio.run(ws._dispatch_radio_frame("u" * 32, {"type": "media_ended"}))


# ── Seed metadata (2026-08-11) ──────────────────────────────────────────
# The seed used to be the ONE track that bypassed every metadata path:
# artist="", the raw combined fast-path label as title, no length. The
# autosaved playlist's row 1 read 'Playboi Carti - Magnolia' over an artist
# line of "YouTube" while every window track was clean, a seed re-anchor
# broadcast duration=0, and the seed's proportional end-guard never armed.


def _mgr():
    from app.agent.radio.session import RadioSessionManager
    return RadioSessionManager()


def test_enable_backfills_seed_from_seed_meta():
    meta = StationTrack(
        video_id="atvSeedX", title="Magnolia", artist="Playboi Carti",
        length="3:24", video_type="MUSIC_VIDEO_TYPE_ATV",
        thumbnail_url="https://i.ytimg.com/vi/atvSeedX/hqdefault.jpg",
    )
    sess = _mgr().enable(
        user_id="u" * 32, channel="app", seed_intent="carti",
        seed_track=SeedTrack(video_id="omvSeed", title="Playboi Carti - Magnolia"),
        station=[StationTrack(video_id="w1", title="Next", artist="A")],
        seed_meta=meta,
    )
    seed = sess.played_history[0]
    # The id that is PLAYING, never seed_meta's — that may be the ATV/OMV
    # counterpart of the playing video.
    assert seed.video_id == "omvSeed"
    assert seed.title == "Magnolia"
    assert seed.artist == "Playboi Carti"
    assert seed.length == "3:24"
    assert sess.current_station_track is seed
    # The play clock knows the length, so the proportional end-guard arms.
    assert sess.current_track_length_sec == 204.0


def test_enable_without_meta_derives_artist_from_the_title():
    sess = _mgr().enable(
        user_id="u" * 32, channel="app", seed_intent="carti",
        seed_track=SeedTrack(video_id="vSeed", title="Playboi Carti - Magnolia"),
        station=[],
    )
    seed = sess.played_history[0]
    assert seed.title == "Magnolia"
    assert seed.artist == "Playboi Carti"


def test_enable_split_never_eats_a_format_marker():
    sess = _mgr().enable(
        user_id="u" * 32, channel="app", seed_intent="x",
        seed_track=SeedTrack(video_id="vSeed", title="Magnolia - Official Video"),
        station=[],
    )
    seed = sess.played_history[0]
    # "Official Video" is a format marker, not a title — the split must not run.
    assert seed.title == "Magnolia - Official Video"
    assert seed.artist == ""


def test_parse_track_reads_search_style_duration_keys():
    from app.agent.radio.playlist import _parse_track

    t = _parse_track({"videoId": "v1", "title": "X",
                      "artists": [{"name": "A"}], "duration": "3:35"})
    assert t.length == "3:35"
    t2 = _parse_track({"videoId": "v2", "title": "Y",
                       "artists": [{"name": "A"}], "duration_seconds": 215})
    assert t2.length == "3:35"
