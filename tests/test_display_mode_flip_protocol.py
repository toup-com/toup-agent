"""The display-mode flip protocol: ack first, warm the current track, name the
swap, carry the duration, never block the WS loop.

BACKGROUND (2026-08-09, founder recording): a Video→Song flip produced ~30s of
silence — the handler ran a 9s window resolve BEFORE sending any frame, inline
in the WS receive loop, issued zero warms for the track whose audio the phone
was about to cold-stream, and the Song→Video swap frame said reason
"auto_advance" with no duration, which the client can only read as "new track,
start from zero" (the P3 reset-to-0:00 card with a stale 13:32 length).

Hermetic: the radio manager, variant lookups, broadcaster and warms are all
stubbed; no network, no DB.
"""
import asyncio

import pytest

import app.api.ws_chat as ws_chat


class _Sess:
    def __init__(self, mode="video", track=None):
        self.channel = "app"
        self.enabled = True
        self.display_mode = mode
        self.display_mode_user_override = False
        self.current_station_track = track
        self.current_track_id = track.video_id if track else None
        self.played_track_ids = set()
        self.played_history = [track] if track else []
        self.history_cursor = 0

    def to_broadcast_dict(self):
        return {"type": "radio_state", "display_mode": self.display_mode}


class _Mgr:
    def __init__(self, sess):
        self._sess = sess

    def get(self, user_id, channel):
        return self._sess

    def get_or_create(self, user_id, channel):
        return self._sess

    def set_display_mode(self, sess, mode, user_initiated=False, source=""):
        sess.display_mode = mode
        sess.display_mode_user_override = user_initiated


class _Track:
    def __init__(self, video_id, video_type, length="4:18"):
        self.video_id = video_id
        self.video_type = video_type
        self.title = "T"
        self.artist = "A"
        self.thumbnail_url = ""
        self.length = length

    def display_title(self):
        return self.title


@pytest.fixture
def rig(monkeypatch):
    """Stub every collaborator; record every observable side effect in order."""
    events = []

    async def fake_broadcast(user_id, frame):
        events.append(("frame", frame.get("type"), dict(frame)))
        return 1

    async def fake_resolve(sess, budget=None):
        events.append(("resolve", budget))

    def fake_upcoming(sess):
        events.append(("upcoming_read",))
        return [{"video_id": "up1", "title": "U", "artist": "A", "thumbnail_url": ""}]

    monkeypatch.setattr(ws_chat, "broadcast_to_user", fake_broadcast)
    monkeypatch.setattr(ws_chat, "_resolve_upcoming_variants", fake_resolve)
    monkeypatch.setattr(ws_chat, "_upcoming_tracks", fake_upcoming)

    import app.agent.radio as radio_mod
    import app.agent.radio.player as player_mod
    import app.agent.radio.playlist as playlist_mod

    def fake_warm(ids, mode="build", now_playing=False):
        events.append(("warm", tuple(ids), mode, now_playing))

    async def fake_broadcast_track(**kwargs):
        events.append(("media_play", kwargs))
        return True

    monkeypatch.setattr(player_mod, "warm_audio_cache", fake_warm)
    monkeypatch.setattr(player_mod, "broadcast_radio_track", fake_broadcast_track)
    return events, radio_mod, playlist_mod, monkeypatch


def _install_mgr(monkeypatch, radio_mod, sess):
    monkeypatch.setattr(radio_mod, "get_radio_manager", lambda: _Mgr(sess))


# ── Video→Song ────────────────────────────────────────────────────────────

def test_video_to_song_acks_before_resolving(rig):
    events, radio_mod, _, monkeypatch = rig
    sess = _Sess(mode="video", track=_Track("omv1", "MUSIC_VIDEO_TYPE_OMV"))
    _install_mgr(monkeypatch, radio_mod, sess)

    asyncio.run(ws_chat._handle_radio_display_mode(
        "u1", {"channel": "app", "mode": "song"}
    ))

    kinds = [e[0] for e in events]
    first_frame = kinds.index("frame")
    resolve = kinds.index("resolve")
    # The ack must NOT be held hostage to the (up to 9s) window resolve.
    assert first_frame < resolve, events
    assert events[first_frame][1] == "radio_state"
    assert events[first_frame][2]["display_mode"] == "song"


def test_video_to_song_build_warms_current_track(rig):
    """The phone is about to stream native audio for an id that has never
    been through the proxy (it played inside the iframe). The flip must start
    the platform's spool/remux for it immediately."""
    events, radio_mod, _, monkeypatch = rig
    sess = _Sess(mode="video", track=_Track("omv1", "MUSIC_VIDEO_TYPE_OMV"))
    _install_mgr(monkeypatch, radio_mod, sess)

    asyncio.run(ws_chat._handle_radio_display_mode(
        "u1", {"channel": "app", "mode": "song"}
    ))

    warms = [e for e in events if e[0] == "warm"]
    # now_playing=True is load-bearing: it is what makes a pre-spool platform
    # downgrade this to extract instead of racing the stream the flip starts.
    assert ("warm", ("omv1",), "build", True) in warms, events
    # And the warm precedes the window resolve — it must not queue behind 9s.
    assert events.index(("warm", ("omv1",), "build", True)) < [
        i for i, e in enumerate(events) if e[0] == "resolve"
    ][0]


def test_video_to_song_no_mid_track_swap(rig):
    """App-channel song direction keeps the same video_id — a swap would be a
    cold reload on a native player (the recorded reversal in CLAUDE.md)."""
    events, radio_mod, _, monkeypatch = rig
    sess = _Sess(mode="video", track=_Track("omv1", "MUSIC_VIDEO_TYPE_OMV"))
    _install_mgr(monkeypatch, radio_mod, sess)

    asyncio.run(ws_chat._handle_radio_display_mode(
        "u1", {"channel": "app", "mode": "song"}
    ))
    assert not [e for e in events if e[0] == "media_play"]
    assert sess.current_track_id == "omv1"


# ── Song→Video ────────────────────────────────────────────────────────────

def test_song_to_video_swap_carries_reason_and_duration(rig):
    events, radio_mod, playlist_mod, monkeypatch = rig
    atv = _Track("atv1", "MUSIC_VIDEO_TYPE_ATV")
    omv = _Track("omv9", "MUSIC_VIDEO_TYPE_OMV", length="13:32")
    sess = _Sess(mode="song", track=atv)
    _install_mgr(monkeypatch, radio_mod, sess)

    async def fake_find_mv(track):
        return omv

    monkeypatch.setattr(playlist_mod, "find_music_video", fake_find_mv)

    asyncio.run(ws_chat._handle_radio_display_mode(
        "u1", {"channel": "app", "mode": "video"}
    ))

    plays = [e for e in events if e[0] == "media_play"]
    assert len(plays) == 1
    kw = plays[0][1]
    assert kw["video_id"] == "omv9"
    # The frame must say "same song, new surface" — not "new track from zero".
    assert kw["reason"] == "mv_swap"
    # 13:32 → 812 seconds, so the card renders a real length immediately.
    assert kw["duration"] == 812


def test_song_to_video_acks_before_mv_lookup(rig):
    events, radio_mod, playlist_mod, monkeypatch = rig
    atv = _Track("atv1", "MUSIC_VIDEO_TYPE_ATV")
    sess = _Sess(mode="song", track=atv)
    _install_mgr(monkeypatch, radio_mod, sess)

    order = []

    async def fake_find_mv(track):
        order.append("lookup")
        return None

    monkeypatch.setattr(playlist_mod, "find_music_video", fake_find_mv)

    async def run():
        await ws_chat._handle_radio_display_mode("u1", {"channel": "app", "mode": "video"})

    asyncio.run(run())
    first_frame_idx = [i for i, e in enumerate(events) if e[0] == "frame"][0]
    assert events[first_frame_idx][2]["display_mode"] == "video"
    # the ack frame exists even though the lookup failed
    assert order == ["lookup"]


# ── protocol plumbing ─────────────────────────────────────────────────────

def test_length_to_seconds():
    f = ws_chat._length_to_seconds
    assert f("4:18") == 258
    assert f("13:32") == 812
    assert f("1:02:07") == 3727
    assert f("0:09") == 9
    assert f("") == 0
    assert f(None) == 0
    assert f("live") == 0
    assert f(90) == 90
    assert f(-5) == 0


def test_concurrent_flips_serialize(rig):
    """Two rapid pill taps must not interleave through the handler's awaits —
    the per-user lock queues them (which is what makes create_task dispatch
    order-safe)."""
    events, radio_mod, _, monkeypatch = rig
    sess = _Sess(mode="video", track=_Track("omv1", "MUSIC_VIDEO_TYPE_OMV"))
    _install_mgr(monkeypatch, radio_mod, sess)

    active = {"n": 0, "max": 0}

    async def slow_resolve(s, budget=None):
        active["n"] += 1
        active["max"] = max(active["max"], active["n"])
        await asyncio.sleep(0.05)
        active["n"] -= 1

    monkeypatch.setattr(ws_chat, "_resolve_upcoming_variants", slow_resolve)

    async def go():
        await asyncio.gather(
            ws_chat._handle_radio_display_mode("u1", {"channel": "app", "mode": "song"}),
            ws_chat._handle_radio_display_mode("u1", {"channel": "app", "mode": "song"}),
        )

    asyncio.run(go())
    assert active["max"] == 1


def test_dispatch_site_is_a_task():
    """Source tripwire: the WS receive loop must dispatch radio_display_mode
    as a task, like the skip handlers — awaiting it inline blocks every
    subsequent client frame behind a ~15s worst-case flip."""
    import inspect
    src = inspect.getsource(ws_chat)
    # Find the receive-loop dispatch block for radio_display_mode.
    idx = src.index('msg_type == "radio_display_mode"')
    window = src[idx: idx + 900]
    assert "create_task(_handle_radio_display_mode" in window, window
    # And the awaited-inline form must be gone from the dispatch site.
    assert "await _handle_radio_display_mode(" not in window, window
