"""Only ONE thing may move the station: the user, or a single plausible end.

The 2026-08-10 founder recordings show what every hole in that rule looks like
on a phone: four track identities in under three seconds, a card mixing one
song's title with another's video, the current track restarting at 0:00, a
requested song abandoned seven seconds in, and a "next" that walks onto a
track the station never played. Each had a distinct server-side enabler:

  * `media_ended` for a track that is NOT current ADVANCED the station
    ("typically a user replay" — it never was; it was dying players).
  * The end dedupe remembered only the LAST honored id, so alternating stale
    ends each popped a track.
  * `radio_skip_next` was one lane for two speakers: the user's ⏭ (rightly
    ungated) and the client's auto-skip-on-error — a machine walking the
    station with the user's authority, unpaced and uncapped.
  * The phone optimistically hops to `upcoming[0]`, but the pop could
    re-resolve the slot to a DIFFERENT id, "correcting" the card a second
    later (measured 3-of-9 on 2026-08-06).
  * A typed play fires a server-side `_auto` toggle AND a client reseed; with
    no lock the two built two `variety` stations for one request.

These tests pin the closed holes. Hermetic — no DB, no network.
"""
from __future__ import annotations

import asyncio
import json
import os
import time

import pytest

os.environ.setdefault("ENVIRONMENT", "test")

from app.agent.radio.playlist import StationTrack, _parse_track  # noqa: E402
from app.agent.radio.session import RadioSession, SeedTrack  # noqa: E402


def _tracks(n: int, prefix: str = "t") -> list:
    return [
        StationTrack(
            video_id=f"{prefix}{i:07d}", title=f"Track {i}", artist=f"Artist {i}",
            length="3:20",
        )
        for i in range(n)
    ]


def _session(video_id: str = "current0001", *, started_ago: float = 200.0,
             length_sec: float = 200.0, playlist_n: int = 8) -> RadioSession:
    sess = RadioSession(user_id="u" * 32, channel="app", enabled=True)
    sess.seed_track = SeedTrack(video_id=video_id, title="seed")
    sess.playlist = _tracks(playlist_n)
    cur = StationTrack(video_id=video_id, title="seed", length="3:20")
    sess.played_history = [cur]
    sess.history_cursor = 0
    sess.current_station_track = cur
    sess.mark_current_track(video_id, length_sec)
    sess.current_track_started_ts = time.time() - started_ago
    return sess


from app.agent.radio.session import RadioSessionManager  # noqa: E402


class _StubManager(RadioSessionManager):
    """Real manager methods (they only touch the session), pinned to one sess."""

    def __init__(self, sess):
        super().__init__()
        self._sess = sess
        self._sessions = {(sess.user_id, sess.channel): sess}

    def get(self, user_id, channel):
        return self._sess


def _wire(monkeypatch, sess):
    """Point ws_chat's radio plumbing at `sess`; capture broadcasts/advances."""
    from app.api import ws_chat
    import app.agent.radio as radio_mod

    sent: list = []
    advanced: list = []
    reanchored: list = []

    async def _bcast(user_id, payload):
        sent.append(payload)
        return 1

    async def _advance(user_id, channel, s, trigger, target_video_id=""):
        advanced.append((trigger, target_video_id))
        return True

    async def _reanchor(user_id, channel, s, track, trigger, record):
        reanchored.append((track.video_id, trigger))

    monkeypatch.setattr(ws_chat, "broadcast_to_user", _bcast)
    monkeypatch.setattr(ws_chat, "_advance_and_broadcast_next", _advance)
    monkeypatch.setattr(ws_chat, "_broadcast_track_for_mode", _reanchor)
    monkeypatch.setattr(ws_chat, "_resolve_upcoming_variants", _noop_resolve)
    monkeypatch.setattr(radio_mod, "get_radio_manager", lambda: _StubManager(sess))
    return sent, advanced, reanchored


async def _noop_resolve(sess, *a, **k):
    return None


# ── media_ended: a stray end moves nothing ──────────────────────────────

@pytest.mark.asyncio
async def test_end_for_non_current_session_track_does_not_advance(monkeypatch):
    """The reversal of the old 'in_session=true — advancing' branch. An end
    naming a HISTORY track (a torn-down WebView, a dead queue item, a second
    device) used to advance the CURRENT track — the recorded uncommanded jump."""
    from app.api import ws_chat

    sess = _session("current0001")
    sess.played_history.append(StationTrack(video_id="previous001", title="old"))
    sent, advanced, reanchored = _wire(monkeypatch, sess)

    await ws_chat._handle_media_ended(
        sess.user_id, {"channel": "app", "video_id": "previous001"}
    )
    assert advanced == [], "an end for a non-current track must not advance"
    # The sender is re-anchored with a real media_play for the CURRENT track —
    # web replays history by swapping its iframe locally, and only a play
    # command resumes the station there (a bare radio_state left it dead).
    assert reanchored and reanchored[0][0] == "current0001"


@pytest.mark.asyncio
async def test_every_honored_end_is_remembered_not_only_the_last(monkeypatch):
    """The single last_ended pair was blind to alternation. The dict keeps
    every recently-honored end."""
    from app.api import ws_chat

    sess = _session("aaaaaaaaaa1")
    _wire(monkeypatch, sess)
    await ws_chat._handle_media_ended(sess.user_id, {"channel": "app", "video_id": "aaaaaaaaaa1"})

    sess.mark_current_track("bbbbbbbbbb2", 200.0)
    sess.current_track_started_ts = time.time() - 199.0
    await ws_chat._handle_media_ended(sess.user_id, {"channel": "app", "video_id": "bbbbbbbbbb2"})

    assert "aaaaaaaaaa1" in sess.recent_ended_ids
    assert "bbbbbbbbbb2" in sess.recent_ended_ids


@pytest.mark.asyncio
async def test_replayed_end_for_current_track_is_deduped_via_the_dict(monkeypatch):
    """Walk back to a track that ended <20s ago (skip_prev), then a stale
    duplicate of its end arrives. current matches, the pin floor has passed
    (the track genuinely played out earlier) — only the dict blocks it."""
    from app.api import ws_chat

    sess = _session("replayed001")
    _, advanced, _re = _wire(monkeypatch, sess)
    await ws_chat._handle_media_ended(sess.user_id, {"channel": "app", "video_id": "replayed001"})
    assert len(advanced) == 1

    # skip_prev puts the same track back as current; its old clock survives the
    # test setup (started long ago → the pin cannot help).
    sess.mark_current_track("replayed001", 200.0)
    sess.current_track_started_ts = time.time() - 199.0
    await ws_chat._handle_media_ended(sess.user_id, {"channel": "app", "video_id": "replayed001"})
    assert len(advanced) == 1, "a re-delivered end must not advance twice"


# ── radio_skip_next: two speakers, two rules ────────────────────────────

@pytest.mark.asyncio
async def test_user_skips_are_never_paced(monkeypatch):
    from app.api import ws_chat

    sess = _session()
    _, advanced, _re = _wire(monkeypatch, sess)
    await ws_chat._handle_radio_skip_next(sess.user_id, {"channel": "app"})
    await ws_chat._handle_radio_skip_next(sess.user_id, {"channel": "app"})
    assert len(advanced) == 2, "back-to-back human ⏭ taps are both authoritative"


@pytest.mark.asyncio
async def test_auto_skip_for_a_stale_track_is_a_no_op(monkeypatch):
    """An auto_error skip naming a track the station already advanced past is
    the same death reported twice (ended+skip pair, or two error lanes racing).
    Honoring it moved the cursor twice for one failure."""
    from app.api import ws_chat

    sess = _session("current0001")
    sent, advanced, reanchored = _wire(monkeypatch, sess)
    await ws_chat._handle_radio_skip_next(
        sess.user_id,
        {"channel": "app", "reason": "auto_error", "video_id": "already_gone"},
    )
    assert advanced == []
    assert any(p.get("type") == "radio_state" for p in sent), "sender is re-anchored"


@pytest.mark.asyncio
async def test_auto_skips_are_paced(monkeypatch):
    from app.api import ws_chat

    sess = _session("current0001")
    _, advanced, _re = _wire(monkeypatch, sess)
    await ws_chat._handle_radio_skip_next(
        sess.user_id, {"channel": "app", "reason": "auto_error", "video_id": "current0001"},
    )
    assert len(advanced) == 1
    # The fake advance does not move current_track_id, so the second machine
    # skip names the current track correctly — only the pacing can block it.
    await ws_chat._handle_radio_skip_next(
        sess.user_id, {"channel": "app", "reason": "auto_error", "video_id": "current0001"},
    )
    assert len(advanced) == 1, "a second machine advance within the pacing gap must wait"


@pytest.mark.asyncio
async def test_auto_skips_are_capped_and_the_user_is_told(monkeypatch):
    from app.api import ws_chat

    sess = _session("current0001")
    sent, advanced, reanchored = _wire(monkeypatch, sess)
    now = time.time()
    # Three machine advances already happened inside the window.
    sess.auto_advance_ts = [now - 60.0, now - 40.0, now - 20.0]
    await ws_chat._handle_radio_skip_next(
        sess.user_id, {"channel": "app", "reason": "auto_error", "video_id": "current0001"},
    )
    assert advanced == [], "past the cap the station holds instead of walking"
    assert any(p.get("type") == "radio_notice" for p in sent), "the user is told"


@pytest.mark.asyncio
async def test_user_skip_still_works_past_the_machine_cap(monkeypatch):
    from app.api import ws_chat

    sess = _session("current0001")
    now = time.time()
    sess.auto_advance_ts = [now - 60.0, now - 40.0, now - 20.0]
    _, advanced, _re = _wire(monkeypatch, sess)
    await ws_chat._handle_radio_skip_next(sess.user_id, {"channel": "app"})
    assert len(advanced) == 1, "the cap binds machines, never the user"


# ── The pop honors the phone's optimistic hop ───────────────────────────

def _advance_env(monkeypatch, sess):
    """Run the REAL _advance_and_broadcast_next with broadcast captured."""
    from app.api import ws_chat
    import app.agent.radio as radio_mod

    played: list = []

    async def _capture_broadcast(user_id, channel, s, track, trigger, record):
        played.append(track)

    async def _bcast(user_id, payload):
        return 1

    monkeypatch.setattr(ws_chat, "_broadcast_track_for_mode", _capture_broadcast)
    monkeypatch.setattr(ws_chat, "broadcast_to_user", _bcast)
    monkeypatch.setattr(radio_mod, "get_radio_manager", lambda: _StubManager(sess))
    return played


@pytest.mark.asyncio
async def test_skip_target_is_honored_verbatim(monkeypatch):
    """The phone already advanced its card to upcoming[0]; the pop must play
    that exact id — and pin its variant so nothing downstream re-resolves it
    into a different one (the 3-of-9 'card heals to a stranger' divergence)."""
    from app.api import ws_chat

    sess = _session(playlist_n=8)
    sess.display_mode_user_override = True   # the case where the pop-time swap bites
    sess.display_mode = "song"
    target = sess.playlist[0].video_id
    played = _advance_env(monkeypatch, sess)

    ok = await ws_chat._advance_and_broadcast_next(
        sess.user_id, "app", sess, trigger="skip_next", target_video_id=target,
    )
    assert ok is True
    assert [t.video_id for t in played] == [target]
    assert sess.playlist[0].variant_resolved_mode == "song", (
        "the honored slot must be pinned against a pop-time re-resolve"
    )


@pytest.mark.asyncio
async def test_skip_target_deeper_in_window_jumps_to_it(monkeypatch):
    """A racing advance moved the cursor before the skip landed; the phone's
    card shows a deeper slot. Play what the user is looking at."""
    from app.api import ws_chat

    sess = _session(playlist_n=8)
    target = sess.playlist[2].video_id
    skipped_over = [sess.playlist[0].video_id, sess.playlist[1].video_id]
    played = _advance_env(monkeypatch, sess)

    await ws_chat._advance_and_broadcast_next(
        sess.user_id, "app", sess, trigger="skip_next", target_video_id=target,
    )
    assert [t.video_id for t in played] == [target]
    for vid in skipped_over:
        assert vid in sess.played_track_ids, "jumped-over slots must not replay later"


@pytest.mark.asyncio
async def test_unknown_target_falls_back_to_a_normal_pop(monkeypatch):
    from app.api import ws_chat

    sess = _session(playlist_n=8)
    first = sess.playlist[0].video_id
    played = _advance_env(monkeypatch, sess)
    await ws_chat._advance_and_broadcast_next(
        sess.user_id, "app", sess, trigger="skip_next", target_video_id="notinwindow",
    )
    assert [t.video_id for t in played] == [first]


# ── One request, one station ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_concurrent_toggles_build_one_station(monkeypatch):
    """A typed play fires the server-side _auto toggle as a task while the
    client's reseed rides the socket. Un-serialized, both passed the dedupe
    (it needs a COMMITTED station) and built two variety stations — one
    audible, the other painting the card. Under the lock the second waits,
    then lands in the dedupe."""
    from app.api import ws_chat
    from app.agent.radio import get_radio_manager
    import app.agent.radio.playlist as playlist_mod

    uid = "f" * 32
    mgr = get_radio_manager()
    mgr._sessions.pop((uid, "app"), None)

    builds: list = []

    async def _slow_build(seed, limit=50, *, seed_title="", seed_artist="", variety=False):
        builds.append(seed)
        await asyncio.sleep(0.05)
        return None, _tracks(10, prefix="built")

    async def _no_atv(track, timeout=6.0):
        return None

    async def _bcast(user_id, payload):
        return 1

    import app.agent.radio as radio_mod
    import app.agent.radio.player as player_mod
    monkeypatch.setattr(radio_mod, "build_station", _slow_build)
    monkeypatch.setattr(playlist_mod, "find_topic_version", _no_atv)
    monkeypatch.setattr(ws_chat, "broadcast_to_user", _bcast)
    monkeypatch.setattr(player_mod, "warm_audio_cache", lambda *a, **k: None)

    msg = {
        "channel": "app", "enabled": True, "video_id": "sameseed001",
        "title": "Same Seed", "seed_intent": "play same seed", "_auto": True,
    }
    await asyncio.gather(
        ws_chat._handle_radio_toggle(uid, msg),
        ws_chat._handle_radio_toggle(uid, dict(msg)),
    )
    assert len(builds) == 1, (
        f"two toggles for one request built {len(builds)} stations — the lock "
        "must let the second land in the committed-station dedupe"
    )
    mgr._sessions.pop((uid, "app"), None)


# ── Metadata honesty ────────────────────────────────────────────────────

def test_ugc_artist_is_derived_from_the_title_not_the_channel():
    """'Radio Javan' is the uploader, not the artist."""
    t = _parse_track({
        "videoId": "ugcvideo001",
        "title": "Alishmas & Kimia - Havase Asheghi",
        "artists": [{"name": "Radio Javan"}],
        "videoType": "MUSIC_VIDEO_TYPE_UGC",
        "length": "3:34",
    })
    assert t.artist == "Alishmas & Kimia"
    assert t.title == "Havase Asheghi"


def test_format_marker_right_side_blocks_the_split():
    t = _parse_track({
        "videoId": "ugcvideo002",
        "title": "Havase Asheghi - Official Video",
        "artists": [{"name": "Radio Javan"}],
        "videoType": "MUSIC_VIDEO_TYPE_UGC",
    })
    assert t.title == "Havase Asheghi - Official Video", "a format marker is not a title"
    assert t.artist == "Radio Javan", "no safe derivation → keep what YT Music gave"


def test_catalog_artist_is_never_touched():
    t = _parse_track({
        "videoId": "atvvideo001",
        "title": "Halo - Live",
        "artists": [{"name": "Beyoncé"}],
        "videoType": "MUSIC_VIDEO_TYPE_ATV",
    })
    assert t.artist == "Beyoncé"
    assert t.title == "Halo - Live"


def test_missing_artist_is_derived_when_the_title_carries_one():
    t = _parse_track({
        "videoId": "noartist001",
        "title": "Shadmehr Aghili - Bi Ehsas",
        "artists": [],
        "videoType": "MUSIC_VIDEO_TYPE_OMV",
    })
    assert t.artist == "Shadmehr Aghili"
    assert t.title == "Bi Ehsas"


def test_upcoming_window_carries_durations():
    from app.api.ws_chat import _upcoming_tracks

    sess = _session(playlist_n=6)
    win = _upcoming_tracks(sess)
    assert win, "window should not be empty"
    assert all(w.get("duration") == 200 for w in win), (
        f"every slot ships its length in seconds: {win}"
    )


def test_a_bad_length_costs_one_duration_never_the_window():
    """Regression: the duration read used to reach through `sess` and blow up
    inside the window builder's blanket except — one hostile slot returned an
    EMPTY window (CI caught it via test_variant_counterpart's hermetic stub).
    A bad length must degrade to duration=0 with every slot still shipped."""
    from app.api.ws_chat import _length_sec, _upcoming_tracks

    assert _length_sec("4:18") == 258
    assert _length_sec("1:02:33") == 3753
    assert _length_sec("") == 0
    assert _length_sec(None) == 0
    assert _length_sec(object()) == 0  # never raises, whatever it is handed

    class _BareSess:  # no parse_length_sec, like any hermetic stub
        channel = "app"
        display_mode = "song"
        display_mode_user_override = False

    sess = _BareSess()
    sess.playlist = _session(playlist_n=4).playlist
    sess.playlist[1].length = object()  # hostile slot
    sess.playlist_cursor = 0
    win = _upcoming_tracks(sess, n=4)
    assert len(win) == 4, f"one bad slot emptied the window: {win}"
    assert win[1]["duration"] == 0
    assert win[0]["duration"] == 200


@pytest.mark.asyncio
async def test_every_advance_broadcast_ships_a_duration(monkeypatch):
    from app.api import ws_chat
    import app.agent.radio.player as player_mod

    sess = _session(playlist_n=6)
    got: dict = {}

    async def _capture(**kwargs):
        got.update(kwargs)
        return True

    async def _bcast(user_id, payload):
        return 1

    monkeypatch.setattr(player_mod, "broadcast_radio_track", _capture)
    monkeypatch.setattr(ws_chat, "broadcast_to_user", _bcast)
    await ws_chat._broadcast_track_for_mode(
        sess.user_id, "app", sess, sess.playlist[0], "media_ended", record=False,
    )
    assert got.get("duration") == 200, f"duration missing from the advance broadcast: {got}"


@pytest.mark.asyncio
async def test_reanchor_says_reanchor_and_is_paced(monkeypatch):
    """Live finding 2026-08-11: `trigger` died at _broadcast_track_for_mode's
    boundary, so a duplicate-end re-anchor went out as reason="auto_advance" —
    one advance, two identical 'advance' frames on the wire. A correction must
    name itself, and repeated corrections inside the window must degrade to
    radio_state alone (an end-looping client otherwise gets a media_play per
    bogus report, each of which can re-trigger the report on web)."""
    from app.api import ws_chat
    import app.agent.radio.player as player_mod

    sess = _session(playlist_n=6)
    plays: list = []
    states: list = []

    async def _capture(**kwargs):
        plays.append(kwargs)
        return True

    async def _bcast(user_id, payload):
        states.append(payload)
        return 1

    monkeypatch.setattr(player_mod, "broadcast_radio_track", _capture)
    monkeypatch.setattr(ws_chat, "broadcast_to_user", _bcast)

    await ws_chat._broadcast_track_for_mode(
        sess.user_id, "app", sess, sess.playlist[0], "reanchor", record=False,
    )
    assert plays and plays[0].get("reason") == "reanchor", (
        f"a re-anchor must not masquerade as an advance: {plays}"
    )

    # A genuine advance still says auto_advance.
    await ws_chat._broadcast_track_for_mode(
        sess.user_id, "app", sess, sess.playlist[1], "media_ended", record=False,
    )
    assert plays[1].get("reason") == "auto_advance"

    # A second re-anchor inside the window ships state only, no media_play.
    n_plays = len(plays)
    await ws_chat._broadcast_track_for_mode(
        sess.user_id, "app", sess, sess.playlist[0], "reanchor", record=False,
    )
    assert len(plays) == n_plays, "paced re-anchor must not emit a media_play"
    assert states, "…but it must still re-anchor with radio_state"

    # Outside the window a re-anchor may carry a media_play again.
    sess.last_reanchor_ts = time.time() - ws_chat._REANCHOR_MIN_INTERVAL_SEC - 1
    await ws_chat._broadcast_track_for_mode(
        sess.user_id, "app", sess, sess.playlist[0], "reanchor", record=False,
    )
    assert len(plays) == n_plays + 1 and plays[-1].get("reason") == "reanchor"


# ── A late embed swap must match the card it lands on ───────────────────

class _FakeProc:
    returncode = 0

    async def communicate(self):
        return json.dumps({"age_limit": 18}).encode(), b""


@pytest.mark.asyncio
async def test_stale_embed_swap_is_suppressed(monkeypatch):
    """The age probe answers up to ~10s late; in a skip burst the station has
    moved on. A swap frame for a non-current track is channel-less and would
    repaint whatever card still matches, on every device."""
    from app.api import ws_chat
    import app.agent.radio as radio_mod

    sess = _session("nowplaying1")
    sent: list = []

    async def _bcast(user_id, payload):
        sent.append(payload)
        return 1

    async def _fake_exec(*args, **kwargs):
        return _FakeProc()

    monkeypatch.setattr(ws_chat, "broadcast_to_user", _bcast)
    monkeypatch.setattr(radio_mod, "get_radio_manager", lambda: _StubManager(sess))
    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)

    # A track the station played and MOVED PAST — the one provably-stale case.
    sess.played_history.append(StationTrack(video_id="staletrack1", title="old"))
    await ws_chat._check_age_and_swap("staletrack1", sess.user_id)
    assert sent == [], "an age swap for a track the station moved past must be dropped"

    await ws_chat._check_age_and_swap("nowplaying1", sess.user_id)
    assert [p.get("type") for p in sent] == ["media_embed_swap"], (
        "a swap for the CURRENT track must still go out"
    )


# ── Round 2 (adversarial review): reconcile vs request ──────────────────

@pytest.mark.asyncio
async def test_reconcile_with_target_bypasses_pacing(monkeypatch):
    """The phone's queue ALREADY advanced when a track died — refusing the
    report (pacing/caps) cannot un-advance the phone, it can only freeze the
    server cursor behind reality and starve the station when the local window
    drains (the review's converged critical finding). A machine report WITH a
    target is a reconciliation and always lands."""
    from app.api import ws_chat

    sess = _session("current0001")
    now = time.time()
    sess.auto_advance_ts = [now - 1.0]           # pacing window is HOT
    target = sess.playlist[0].video_id
    sent, advanced, _re = _wire(monkeypatch, sess)

    await ws_chat._handle_radio_skip_next(
        sess.user_id,
        {"channel": "app", "reason": "auto_error",
         "video_id": "current0001", "target_video_id": target},
    )
    assert advanced == [], "a reconcile pops nothing — the phone already moved"
    assert sess.current_track_id == target, "the cursor must follow the phone"
    assert any(p.get("type") == "radio_upcoming" for p in sent), (
        "the phone needs a fresh window for the position it is actually at"
    )


@pytest.mark.asyncio
async def test_reconcile_unknown_target_falls_back_to_request_rules(monkeypatch):
    from app.api import ws_chat

    sess = _session("current0001")
    now = time.time()
    sess.auto_advance_ts = [now - 1.0]
    _, advanced, _re = _wire(monkeypatch, sess)
    await ws_chat._handle_radio_skip_next(
        sess.user_id,
        {"channel": "app", "reason": "auto_error",
         "video_id": "current0001", "target_video_id": "notinwindow1"},
    )
    assert advanced == [], "unknown target inside the pacing gap stays paced"
    assert sess.current_track_id == "current0001"


@pytest.mark.asyncio
async def test_paced_request_reanchors_instead_of_silence(monkeypatch):
    from app.api import ws_chat

    sess = _session("current0001")
    sess.auto_advance_ts = [time.time() - 1.0]
    sent, advanced, _re = _wire(monkeypatch, sess)
    await ws_chat._handle_radio_skip_next(
        sess.user_id,
        {"channel": "app", "reason": "auto_error", "video_id": "current0001"},
    )
    assert advanced == []
    assert any(p.get("type") == "radio_state" for p in sent), (
        "a paced refusal must answer with state, never silence — the client "
        "defers-and-retries off it"
    )


def test_new_station_resets_the_advance_ledger():
    """The user's escape hatch — asking for different music — must not arrive
    pre-capped by the dead station's history."""
    from app.agent.radio import get_radio_manager

    mgr = get_radio_manager()
    uid = "g" * 32
    mgr._sessions.pop((uid, "app"), None)
    sess = mgr.enable(
        user_id=uid, channel="app", seed_intent="x",
        seed_track=SeedTrack(video_id="seedreset01", title="Seed"),
        station=_tracks(5, prefix="r"),
    )
    sess.auto_advance_ts = [time.time()] * 3
    sess.recent_ended_ids = {"old": time.time()}
    mgr.enable(
        user_id=uid, channel="app", seed_intent="y",
        seed_track=SeedTrack(video_id="seedreset02", title="Seed2"),
        station=_tracks(5, prefix="r2"),
    )
    assert sess.auto_advance_ts == []
    assert sess.recent_ended_ids == {}
    sess.auto_advance_ts = [time.time()] * 3
    mgr.record_user_seed(
        user_id=uid, channel="app", seed_intent="z",
        seed_track=SeedTrack(video_id="seedreset03", title="Seed3"),
    )
    assert sess.auto_advance_ts == []
    mgr._sessions.pop((uid, "app"), None)


@pytest.mark.asyncio
async def test_completion_evidence_waives_the_wall_clock_pin(monkeypatch):
    """A user seeks to the last seconds of a track and lets it finish: a
    genuine end the wall-clock floor cannot see. The client's reported
    playhead-at-duration is proof, and refusing it dead-ended the station
    (every client latch spent, review finding)."""
    from app.api import ws_chat

    sess = _session("seekend0001", started_ago=8.0, length_sec=214.0)
    _, advanced, _re = _wire(monkeypatch, sess)
    await ws_chat._handle_media_ended(
        sess.user_id,
        {"channel": "app", "video_id": "seekend0001",
         "position": 213.2, "duration": 214.0},
    )
    assert advanced == [("media_ended", "")] or len(advanced) == 1, (
        "an evidence-backed end must advance despite the wall-clock pin"
    )


@pytest.mark.asyncio
async def test_unevidenced_early_end_is_still_pinned(monkeypatch):
    from app.api import ws_chat

    sess = _session("pinned00001", started_ago=8.0, length_sec=214.0)
    _, advanced, _re = _wire(monkeypatch, sess)
    await ws_chat._handle_media_ended(
        sess.user_id, {"channel": "app", "video_id": "pinned00001"},
    )
    assert advanced == [], "no evidence → the pin stands"


@pytest.mark.asyncio
async def test_partial_evidence_does_not_waive_the_pin(monkeypatch):
    from app.api import ws_chat

    sess = _session("partial0001", started_ago=8.0, length_sec=214.0)
    _, advanced, _re = _wire(monkeypatch, sess)
    await ws_chat._handle_media_ended(
        sess.user_id,
        {"channel": "app", "video_id": "partial0001",
         "position": 30.0, "duration": 214.0},
    )
    assert advanced == [], "a mid-track playhead is not completion evidence"


@pytest.mark.asyncio
async def test_stale_swap_still_sent_for_one_off_plays(monkeypatch):
    """Sessions never self-expire, so 'some session's current isn't this
    video' is NORMAL for every one-off play. Only a track a station provably
    moved past is suppressed (review finding: yesterday's enabled app session
    suppressed every age swap on web)."""
    from app.api import ws_chat
    import app.agent.radio as radio_mod

    sess = _session("yesterday01")
    sent: list = []

    async def _bcast(user_id, payload):
        sent.append(payload)
        return 1

    async def _fake_exec(*args, **kwargs):
        return _FakeProc()

    monkeypatch.setattr(ws_chat, "broadcast_to_user", _bcast)
    monkeypatch.setattr(radio_mod, "get_radio_manager", lambda: _StubManager(sess))
    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)

    # A one-off play of a video no session has ever touched → swap goes out.
    await ws_chat._check_age_and_swap("oneoffvideo", sess.user_id)
    assert [p.get("type") for p in sent] == ["media_embed_swap"]

    # A track the station played and moved past → suppressed.
    sent.clear()
    sess.played_history.append(StationTrack(video_id="movedpast01", title="old"))
    await ws_chat._check_age_and_swap("movedpast01", sess.user_id)
    assert sent == []


@pytest.mark.asyncio
async def test_refill_abandons_when_the_station_was_replaced(monkeypatch):
    """The refill build runs seconds under the advance lock while a reseed
    (toggle lock) replaces the station — broadcasting the DEAD station's pick
    over the just-requested song is an uncommanded jump (review finding)."""
    from app.api import ws_chat
    import app.agent.radio as radio_mod

    sess = _session("oldseed0001", playlist_n=2)
    # Drain the playlist so the advance path must extend.
    for t in sess.playlist:
        sess.played_track_ids.add(t.video_id)

    played: list = []

    async def _capture_broadcast(user_id, channel, s, track, trigger, record):
        played.append(track.video_id)

    async def _bcast(user_id, payload):
        return 1

    async def _build_that_races_a_reseed(seed, limit=50, *, seed_title="",
                                         seed_artist="", variety=False):
        # Mid-build, a reseed replaces the station.
        sess.seed_track = SeedTrack(video_id="newseed0001", title="New")
        return None, _tracks(5, prefix="dead")

    monkeypatch.setattr(ws_chat, "_broadcast_track_for_mode", _capture_broadcast)
    monkeypatch.setattr(ws_chat, "broadcast_to_user", _bcast)
    monkeypatch.setattr(radio_mod, "build_station", _build_that_races_a_reseed)
    monkeypatch.setattr(radio_mod, "get_radio_manager", lambda: _StubManager(sess))

    ok = await ws_chat._advance_and_broadcast_next(
        sess.user_id, "app", sess, trigger="media_ended",
    )
    assert ok is False and played == [], (
        "an advance whose refill outlived its station must broadcast nothing"
    )


# ── Source tripwires ────────────────────────────────────────────────────
# These regions are nested generators / endpoint bodies that a unit test
# cannot reach without a live socket or spool. A tripwire is weaker than a
# behavior test, but it is what turns a silent revert into a red build.

def test_reconnect_reanchors_radio_state():
    import inspect
    from app.api import ws_chat

    src = inspect.getsource(ws_chat)
    assert "re-anchor radio state" in src.lower() or "Resume: re-anchor radio state" in src, (
        "the connect path must announce enabled radio sessions to a "
        "reconnecting client — without it a returning phone keeps a dead card"
    )


def test_mid_stream_stall_is_cut_not_held():
    import inspect
    from app.api import media_proxy

    src = inspect.getsource(media_proxy)
    assert "_MID_STREAM_STALL_SECS" in src
    assert src.count("STALLED") >= 2, (
        "both stream bodies (spool + legacy) must cut a silent mid-body stall"
    )
    assert "TRUNCATED" in src, "a short body must log a distinct truncation line"


def test_stalled_break_lives_in_the_waiting_branch():
    """The spool watchdog must bound WAITING for upstream bytes, never client
    backpressure (a paused player blocks on `yield`, not on the poll loop)."""
    import inspect
    from app.api import media_proxy

    src = inspect.getsource(media_proxy)
    poll = src.split("elif spool.done:")[1].split("finally:")[0]
    assert "_MID_STREAM_STALL_SECS" in poll and "break" in poll, (
        "the stall watchdog belongs in the else-branch that sleeps on "
        "spool.received"
    )


@pytest.mark.asyncio
async def test_mid_stream_stall_cuts_the_spool_response(tmp_path, monkeypatch):
    """Bytes flow, the upstream goes quiet, and the response must END — held
    open, the phone renders a playing UI over dead audio for the downloader's
    full 60s read timeout (the recorded freeze at 0:02 of 3:52)."""
    from app.api import media_proxy

    f = tmp_path / "spool.bin"
    f.write_bytes(b"y" * 100)
    sp = media_proxy._Spool("stalltest01", str(f))
    sp.total = 1000
    sp.received = 100          # bytes flowed…
    sp.done = False            # …then the upstream went quiet forever
    monkeypatch.setattr(media_proxy, "_MID_STREAM_STALL_SECS", 0.2)

    resp = media_proxy._spool_response(
        sp, None, prefetch=True, mime="audio/mp4", video_id="stalltest01",
        on_release=lambda: None, t0=time.monotonic(), cache_ms=0, extract_ms=0,
    )
    assert resp is not None

    got = 0
    t_start = time.monotonic()
    async for chunk in resp.body_iterator:
        got += len(chunk)
    elapsed = time.monotonic() - t_start
    assert got == 100, "the bytes that exist must be delivered"
    assert elapsed < 5.0, (
        f"a silent mid-body stall must cut the response fast, not hold it "
        f"open ({elapsed:.1f}s)"
    )


@pytest.mark.asyncio
async def test_stall_cut_applies_from_byte_zero_of_a_resume(tmp_path, monkeypatch):
    """After a stall cut, AVPlayer re-ranges at its buffered edge — which
    starts a NEW response with sent==0 waiting on the same dead spool. A
    sent>0 exemption on the watchdog degraded the 8s bound straight back to
    the 60s downloader timeout on exactly that retry (review finding)."""
    from app.api import media_proxy

    f = tmp_path / "spool2.bin"
    f.write_bytes(b"z" * 100)
    sp = media_proxy._Spool("stallresume", str(f))
    sp.total = 1000
    sp.received = 100
    sp.done = False
    monkeypatch.setattr(media_proxy, "_MID_STREAM_STALL_SECS", 0.2)

    resp = media_proxy._spool_response(
        sp, "bytes=100-", prefetch=True, mime="audio/mp4", video_id="stallresume",
        on_release=lambda: None, t0=time.monotonic(), cache_ms=0, extract_ms=0,
    )
    assert resp is not None
    got = 0
    t_start = time.monotonic()
    async for chunk in resp.body_iterator:
        got += len(chunk)
    assert got == 0 and time.monotonic() - t_start < 5.0, (
        "a resume landing on a dead spool must be cut fast even at sent==0"
    )


@pytest.mark.asyncio
async def test_client_disconnect_is_not_logged_as_truncated(tmp_path, monkeypatch, caplog):
    """An ordinary partial-read disconnect ends short of Content-Length too;
    logging it as TRUNCATED made the new signature noise on day one."""
    import logging
    from app.api import media_proxy

    f = tmp_path / "spool3.bin"
    f.write_bytes(b"w" * 200)
    sp = media_proxy._Spool("disconnect1", str(f))
    sp.total = 1000
    sp.received = 200
    sp.done = False

    resp = media_proxy._spool_response(
        sp, None, prefetch=True, mime="audio/mp4", video_id="disconnect1",
        on_release=lambda: None, t0=time.monotonic(), cache_ms=0, extract_ms=0,
    )
    assert resp is not None
    with caplog.at_level(logging.WARNING):
        agen = resp.body_iterator
        await agen.__anext__()          # client reads one chunk…
        await agen.aclose()             # …then disconnects
    assert "TRUNCATED" not in caplog.text


def test_if_range_mismatch_serves_the_full_current_entity(tmp_path):
    """The same URL serves different entities over time (itag-18 → m4a). A
    client validating with If-Range must get a full 200 of the CURRENT entity
    instead of a byte slice of the wrong container."""
    from app.api.media_proxy import _local_audio_response

    f = tmp_path / "b.m4a"
    f.write_bytes(b"m" * 1000)

    match = _local_audio_response(str(f), "bytes=100-199")
    assert match is not None and match.status_code == 206
    etag = match.headers.get("ETag")
    assert etag, "responses must carry an entity tag"

    same = _local_audio_response(str(f), "bytes=100-199", etag)
    assert same is not None and same.status_code == 206

    other = _local_audio_response(str(f), "bytes=100-199", '"stale-entity"')
    assert other is not None and other.status_code == 200, (
        "an If-Range mismatch must ignore the Range and serve the full entity"
    )


def test_etag_is_stable_across_replicas(tmp_path):
    """Regression: an mtime-flavoured ETag differed between the two replicas'
    local copies of the SAME R2 artifact, so ~half of all If-Range resumes
    landed on the other replica, mismatched, and restarted the full entity
    (measured live 2026-08-11). The tag must name the entity — deterministic
    basename + size — not the moment a replica happened to download it."""
    from app.api.media_proxy import _local_audio_response

    a_dir, b_dir = tmp_path / "replica_a", tmp_path / "replica_b"
    a_dir.mkdir(), b_dir.mkdir()
    (a_dir / "vid123.m4a").write_bytes(b"m" * 1000)
    (b_dir / "vid123.m4a").write_bytes(b"m" * 1000)
    # The live pair was 20s apart (each replica pulled R2 at its own moment).
    # Set it explicitly — a sleep can land inside the same integer second and
    # let an mtime-flavoured tag slip through (it did, in mutation testing).
    now = os.path.getmtime(a_dir / "vid123.m4a")
    os.utime(b_dir / "vid123.m4a", (now + 20, now + 20))

    ra = _local_audio_response(str(a_dir / "vid123.m4a"), "bytes=0-99")
    rb = _local_audio_response(str(b_dir / "vid123.m4a"), "bytes=0-99")
    ea, eb = ra.headers.get("ETag"), rb.headers.get("ETag")
    assert ea and ea == eb, f"same artifact, different tags: {ea} vs {eb}"

    # …and A's tag validates on B: the resume keeps its 206.
    cross = _local_audio_response(str(b_dir / "vid123.m4a"), "bytes=100-199", ea)
    assert cross is not None and cross.status_code == 206

    # A different-size file at the same name is a different entity.
    (b_dir / "vid123.m4a").write_bytes(b"m" * 4000)
    swapped = _local_audio_response(str(b_dir / "vid123.m4a"), "bytes=100-199", ea)
    assert swapped is not None and swapped.status_code == 200, (
        "a size change is the container swap — the stale tag must force a full 200"
    )


def test_range_beyond_entity_is_416(tmp_path):
    from app.api.media_proxy import _local_audio_response

    f = tmp_path / "a.m4a"
    f.write_bytes(b"x" * 1000)

    resp = _local_audio_response(str(f), "bytes=5000-")
    assert resp is not None and resp.status_code == 416, (
        "a range into a larger entity (the retired spool's itag-18) must be "
        "refused, not answered with a silent 200 of a different container"
    )
    assert resp.headers.get("Content-Range") == "bytes */1000"

    ok = _local_audio_response(str(f), "bytes=0-99")
    assert ok is not None and ok.status_code == 206


def test_wire_titles_are_bare_and_artist_rides_its_own_field():
    """2026-08-11 founder screenshots: cards titled 'void — Wok' over a
    subtitle of 'void'. display_title() composes 'Artist — Title' for LOGS
    (its own contract), but three wire sites shipped it as the frame title
    while `artist` rode the same payload — both clients drew the artist twice."""
    from app.api.ws_chat import _upcoming_tracks

    sess = _session(playlist_n=2)
    sess.playlist[0].title = "Wok"
    sess.playlist[0].artist = "void"
    win = _upcoming_tracks(sess, n=1)
    assert win[0]["title"] == "Wok"
    assert win[0]["artist"] == "void"


@pytest.mark.asyncio
async def test_advance_broadcast_title_is_bare(monkeypatch):
    from app.api import ws_chat
    import app.agent.radio.player as player_mod

    sess = _session(playlist_n=6)
    sess.playlist[0].title = "ORANGE SODA"
    sess.playlist[0].artist = "Baby Keem"
    got: dict = {}

    async def _capture(**kwargs):
        got.update(kwargs)
        return True

    async def _bcast(user_id, payload):
        return 1

    monkeypatch.setattr(player_mod, "broadcast_radio_track", _capture)
    monkeypatch.setattr(ws_chat, "broadcast_to_user", _bcast)
    await ws_chat._broadcast_track_for_mode(
        sess.user_id, "app", sess, sess.playlist[0], "media_ended", record=False,
    )
    assert got.get("title") == "ORANGE SODA", f"composed title on the wire: {got.get('title')!r}"
    assert got.get("artist") == "Baby Keem"


@pytest.mark.asyncio
async def test_skip_target_already_played_reconciles_without_a_pop(monkeypatch):
    """2026-08-11 founder clip: during a 4-in-11s skip burst the server's own
    earlier pops consumed the id the phone then named as its target. 'Normal
    pop' popped one slot FURTHER and each divergent broadcast repainted the
    card — the ORANGE SODA ↔ NOSTYLIST fight. A target the server has already
    played means the PHONE IS THERE: move the cursor to it, ship state, and
    never navigate the phone."""
    from app.api import ws_chat

    sess = _session(playlist_n=8)
    # The burst, as the server experienced it: it popped ghost, then later —
    # its cursor now sits PAST the track the phone is actually playing.
    ghost, later = sess.playlist[0], sess.playlist[1]
    sess.played_track_ids.update({ghost.video_id, later.video_id})
    sess.played_history.extend([ghost, later])
    sess.history_cursor = len(sess.played_history) - 1
    sess.current_station_track = later
    sess.mark_current_track(later.video_id, 200.0)
    sess.playlist_cursor = 2
    played = _advance_env(monkeypatch, sess)

    states: list = []

    async def _bcast(user_id, payload):
        states.append(payload)
        return 1

    monkeypatch.setattr(ws_chat, "broadcast_to_user", _bcast)

    ok = await ws_chat._advance_and_broadcast_next(
        sess.user_id, "app", sess, trigger="skip_next",
        target_video_id=ghost.video_id,
    )
    assert ok is True
    assert played == [], f"a played target must never pop further: {played}"
    assert sess.current_track_id == ghost.video_id
    assert sess.current_station_track is ghost
    assert states and states[-1].get("current_track_id") == ghost.video_id


@pytest.mark.asyncio
async def test_skip_target_equal_to_current_is_a_state_reship(monkeypatch):
    """A late duplicate of a skip this cursor already honored (echoes drain
    behind the advance lock during a burst) must confirm, not advance."""
    from app.api import ws_chat

    sess = _session(playlist_n=8)
    played = _advance_env(monkeypatch, sess)
    states: list = []

    async def _bcast(user_id, payload):
        states.append(payload)
        return 1

    monkeypatch.setattr(ws_chat, "broadcast_to_user", _bcast)

    ok = await ws_chat._advance_and_broadcast_next(
        sess.user_id, "app", sess, trigger="skip_next",
        target_video_id=sess.current_track_id,
    )
    assert ok is True
    assert played == [], f"a duplicate skip must not advance: {played}"
    assert states and states[-1].get("current_track_id") == sess.current_track_id
