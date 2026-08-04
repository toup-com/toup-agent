"""Audio-first defaults + station variety + the Toup Media playlist library.

Covers the 2026-08-03 media overhaul:

  * the 'app' channel is AUDIO-FIRST — a fresh session starts in 'song', web
    stays 'video' (the surface decision that removes the video→audio
    background handoff gap entirely for default plays)
  * `radio_toggle` accepts an initial `mode` and applies it AFTER enable()'s
    reset — collapsing the two-frame race that used to resolve the first
    upcoming window for the wrong surface
  * the `_auto` toggle (fast-path "play me X") now ships the prebuffer window
    via a dedicated `radio_upcoming` frame — the first lock-screen ⏭ after a
    fresh play used to be a guaranteed cold round-trip because the window was
    never sent
  * an app-channel display-mode flip performs NO mid-track variant swap (a
    different video_id is a cold reload + mid-file seek on a native player)
    and re-ships the window instead
  * `_light_shuffle` varies order without teleporting the deep tail forward
  * the playlist library: capturing a live station preserves play order and
    dedupes; the REST surface saves/lists/plays it against the real handlers

Hermetic like test_radio_station.py — no network; DB tests run on the
conftest's shared in-memory sqlite.
"""
from __future__ import annotations

import asyncio
import json
import os

import pytest

os.environ.setdefault("ENVIRONMENT", "test")

from app.agent.radio.playlist import StationTrack, _light_shuffle  # noqa: E402
from app.agent.radio.session import (  # noqa: E402
    RadioSessionManager,
    SeedTrack,
    default_display_mode,
)


def _tracks(n: int, prefix: str = "vid") -> list[StationTrack]:
    return [
        StationTrack(video_id=f"{prefix}{i:03d}xxxx", title=f"Track {i}", artist=f"Artist {i}")
        for i in range(n)
    ]


# ── Audio-first channel defaults ────────────────────────────────────────

def test_default_display_mode_is_song_on_app_video_on_web():
    assert default_display_mode("app") == "song"
    assert default_display_mode("web") == "video"
    assert default_display_mode("telegram") == "video"


def test_enable_applies_channel_default_mode():
    mgr = RadioSessionManager()
    app_sess = mgr.enable(
        user_id="u" * 32, channel="app", seed_intent="drake",
        seed_track=SeedTrack(video_id="seedvid0001", title="Seed"),
        station=_tracks(5),
    )
    assert app_sess.display_mode == "song"
    web_sess = mgr.enable(
        user_id="u" * 32, channel="web", seed_intent="drake",
        seed_track=SeedTrack(video_id="seedvid0001", title="Seed"),
        station=_tracks(5),
    )
    assert web_sess.display_mode == "video"


def test_record_user_seed_applies_channel_default_mode():
    mgr = RadioSessionManager()
    sess = mgr.record_user_seed(
        user_id="u" * 32, channel="app", seed_intent="drake",
        seed_track=SeedTrack(video_id="seedvid0001", title="Seed"),
    )
    assert sess.display_mode == "song"


# ── radio_toggle: initial mode + _auto window ship ──────────────────────

def _patch_toggle_collab(monkeypatch, sent: list):
    """Stub every network/broadcast edge the toggle handler touches."""
    import app.api.ws_chat as ws
    import app.agent.radio.player as player

    async def _fake_broadcast(uid, frame):
        sent.append(frame)
        return 1

    async def _fake_station(seed, limit=50, **k):
        meta = StationTrack(video_id=seed, title="Seed Song", artist="Seed Artist")
        return meta, _tracks(8)

    async def _no_atv(track):
        return None

    monkeypatch.setattr(ws, "broadcast_to_user", _fake_broadcast)
    monkeypatch.setattr(player, "broadcast_to_user", _fake_broadcast, raising=False)
    monkeypatch.setattr("app.agent.radio.build_station", _fake_station)
    monkeypatch.setattr("app.agent.radio.playlist.find_topic_version", _no_atv)
    monkeypatch.setattr(player, "warm_audio_cache", lambda ids: None)


@pytest.mark.asyncio
async def test_toggle_applies_initial_mode_after_enable_reset(monkeypatch):
    import app.api.ws_chat as ws
    from app.agent.radio import get_radio_manager

    sent: list = []
    _patch_toggle_collab(monkeypatch, sent)

    await ws._handle_radio_toggle("u" * 32, {
        "channel": "app", "enabled": True,
        "video_id": "seedvid0001", "title": "Seed Song",
        "mode": "video",
    })
    sess = get_radio_manager().get("u" * 32, "app")
    assert sess is not None and sess.enabled
    # enable() resets to the channel default ('song' on app); the toggle's own
    # mode must win AFTER that reset, with the user-override latch set.
    assert sess.display_mode == "video"
    assert sess.display_mode_user_override is True


@pytest.mark.asyncio
async def test_auto_toggle_ships_radio_upcoming_and_no_media_play(monkeypatch):
    import app.api.ws_chat as ws
    from app.agent.radio import get_radio_manager

    sent: list = []
    _patch_toggle_collab(monkeypatch, sent)

    await ws._handle_radio_toggle("u" * 32, {
        "channel": "app", "enabled": True,
        "video_id": "seedvid0001", "title": "Seed Song",
        "seed_intent": "play me something",
        "_auto": True,
    })
    types = [f.get("type") for f in sent]
    # _auto must NOT re-broadcast the seed (it is already playing)…
    assert "media_play" not in types, f"_auto toggle cold-reloaded the seed: {types}"
    # …but MUST ship the prebuffer window — the missing window was why the
    # first lock-screen ⏭ after a fresh play was always a cold round-trip.
    ups = [f for f in sent if f.get("type") == "radio_upcoming"]
    assert len(ups) == 1, f"expected one radio_upcoming, got {types}"
    assert ups[0]["channel"] == "app"
    assert len(ups[0]["upcoming"]) >= 1
    assert ups[0]["upcoming"][0]["video_id"]
    sess = get_radio_manager().get("u" * 32, "app")
    assert sess is not None and sess.enabled


# ── app-channel display-mode flip: no mid-track swap ────────────────────

@pytest.mark.asyncio
async def test_app_mode_flip_swaps_nothing_and_reships_window(monkeypatch):
    import app.api.ws_chat as ws
    from app.agent.radio import get_radio_manager

    sent: list = []

    async def _fake_broadcast(uid, frame):
        sent.append(frame)
        return 1

    async def _must_not_lookup(track):
        raise AssertionError("variant lookup must not run for an app-channel mode flip")

    monkeypatch.setattr(ws, "broadcast_to_user", _fake_broadcast)
    monkeypatch.setattr("app.agent.radio.playlist.find_topic_version", _must_not_lookup)
    monkeypatch.setattr("app.agent.radio.playlist.find_music_video", _must_not_lookup)

    mgr = get_radio_manager()
    sess = mgr.enable(
        user_id="m" * 32, channel="app", seed_intent="x",
        seed_track=SeedTrack(video_id="seedvid0002", title="Seed"),
        station=_tracks(6),
    )
    # Current track is an OMV — the exact shape that used to trigger a
    # mid-track ATV swap (a different video_id = cold reload on the phone).
    sess.current_station_track = StationTrack(
        video_id="omvvid00001", title="Song", artist="A",
        video_type="MUSIC_VIDEO_TYPE_OMV",
    )
    sess.current_track_id = "omvvid00001"

    await ws._handle_radio_display_mode("m" * 32, {"channel": "app", "mode": "song"})

    types = [f.get("type") for f in sent]
    assert "media_play" not in types, f"app flip re-broadcast a track: {types}"
    assert "radio_upcoming" in types
    assert sess.display_mode == "song"
    assert sess.display_mode_user_override is True


# ── _light_shuffle invariants ───────────────────────────────────────────

def test_light_shuffle_keeps_head_and_multiset():
    station = _tracks(30)
    shuffled = _light_shuffle(station)
    assert [t.video_id for t in shuffled[:2]] == [t.video_id for t in station[:2]]
    assert sorted(t.video_id for t in shuffled) == sorted(t.video_id for t in station)
    # Windowed: a track can move at most within its 8-chunk, so the deep tail
    # can never jump to the front.
    assert shuffled[2].video_id in {t.video_id for t in station[2:10]}


def test_light_shuffle_small_station_untouched():
    station = _tracks(3)
    assert _light_shuffle(station) == station


# ── Playlist library: capture order, save, play ─────────────────────────

def test_capture_live_station_preserves_order_and_dedupes():
    from app.api.media_playlists import _capture_live_station
    from app.agent.radio import get_radio_manager

    mgr = get_radio_manager()
    uid = "c" * 32
    station = _tracks(6, prefix="cap")
    sess = mgr.enable(
        user_id=uid, channel="app", seed_intent="capture me",
        seed_track=SeedTrack(video_id="capseed0001", title="Cap Seed"),
        station=station,
    )
    # Play two tracks so the tape has entries beyond the seed.
    t1 = mgr.pop_next_from_playlist(sess)
    mgr.record_auto_play(sess, t1)
    t2 = mgr.pop_next_from_playlist(sess)
    mgr.record_auto_play(sess, t2)

    got_sess, tracks = _capture_live_station(uid, "app")
    assert got_sess is sess
    ids = [t["video_id"] for t in tracks]
    # Seed first (history[0]), then the two played, then the remaining queue —
    # exact play order, no duplicates.
    assert ids[0] == "capseed0001"
    assert ids[1] == t1.video_id and ids[2] == t2.video_id
    assert len(ids) == len(set(ids))
    assert set(t.video_id for t in station) <= set(ids)


@pytest.fixture()
async def _playlists_app(monkeypatch):
    """Minimal app mounting the real playlists router over the conftest
    sqlite DB, with auth stubbed to a fixed user."""
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from app.api import media_playlists as mp
    from app.api.auth import get_current_user
    from app.db.database import init_db

    await init_db()

    class _U:
        id = "p" * 32

    app = FastAPI()
    app.include_router(mp.router, prefix="/api")
    app.dependency_overrides[get_current_user] = lambda: _U()

    # The play endpoint executes locally in monolith/agent mode.
    monkeypatch.setattr(mp, "_serving_locally", lambda: True)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.mark.asyncio
async def test_playlist_save_list_play_roundtrip(_playlists_app, monkeypatch):
    from app.agent.radio import get_radio_manager
    import app.agent.radio.player as player
    import app.api.ws_chat as ws

    client = _playlists_app
    uid = "p" * 32
    mgr = get_radio_manager()

    # A live station to capture.
    sess = mgr.enable(
        user_id=uid, channel="app", seed_intent="play me test artist",
        seed_track=SeedTrack(video_id="rtseed00001", title="RT Seed"),
        station=_tracks(5, prefix="rt"),
    )
    t1 = mgr.pop_next_from_playlist(sess)
    mgr.record_auto_play(sess, t1)

    # live snapshot
    r = await client.get("/api/media/playlists/live?channel=app")
    assert r.status_code == 200
    live = r.json()
    assert live["active"] is True
    assert live["track_count"] >= 6

    # save it
    r = await client.post("/api/media/playlists", json={"from_channel": "app", "name": "My Test Mix"})
    assert r.status_code == 200, r.text
    saved = r.json()
    assert saved["name"] == "My Test Mix"
    assert saved["track_count"] >= 6
    assert saved["tracks"][0]["video_id"] == "rtseed00001"
    pid = saved["id"]

    # list shows it
    r = await client.get("/api/media/playlists")
    assert r.status_code == 200
    assert any(p["id"] == pid for p in r.json())

    # play it — broadcast edges stubbed, session re-seeded from stored tracks
    frames: list = []

    async def _fake_broadcast(u, frame):
        frames.append(frame)
        return 1

    async def _fake_radio_track(**kwargs):
        frames.append({"type": "media_play", **kwargs})
        return True

    # The play handler imports these at call time, so module-attr patches land.
    monkeypatch.setattr(ws, "broadcast_to_user", _fake_broadcast)
    monkeypatch.setattr(player, "broadcast_radio_track", _fake_radio_track)

    r = await client.post(f"/api/media/playlists/{pid}/play", json={"channel": "app"})
    assert r.status_code == 200, r.text
    play = r.json()
    assert play["ok"] is True
    assert play["first"]["video_id"] == "rtseed00001"
    assert len(play["upcoming"]) >= 1

    new_sess = mgr.get(uid, "app")
    assert new_sess is not None and new_sess.enabled
    assert new_sess.seed_track.video_id == "rtseed00001"
    assert new_sess.display_mode == "song"
    # Exact stored order: the first queued track is the first non-seed entry.
    assert new_sess.playlist[0].video_id == saved["tracks"][1]["video_id"]

    # rename + delete
    r = await client.patch(f"/api/media/playlists/{pid}", json={"name": "Renamed Mix"})
    assert r.status_code == 200 and r.json()["name"] == "Renamed Mix"
    r = await client.delete(f"/api/media/playlists/{pid}")
    assert r.status_code == 204
    r = await client.get(f"/api/media/playlists/{pid}")
    assert r.status_code == 404


# ── Review-driven regressions (2026-08-03) ────────────────────────────────
# Each of these encodes a defect an adversarial review found in the first cut
# of the audio-first work. They are cheap; the failures they prevent are not.

def test_infer_requested_mode_reads_watch_intent_not_song_titles():
    from app.agent.radio.player import infer_requested_mode

    # Explicit watch intent, and non-music content that has nothing to hear.
    for q in (
        "play the music video for HUMBLE",
        "play the video of Talagh",
        "show me the Kendrick video",
        "watch the new Dune trailer",
        "play a good documentary",
        "play the Joe Rogan podcast with Elon",
    ):
        assert infer_requested_mode(q) == "video", q

    # Plain music asks, and the trap: "watch" inside a TITLE is not intent.
    for q in (
        "play talagh googoosh",
        "play me some Drake",
        "play Watch Me by Silento",
        "put on some 80s rock",
        "",
    ):
        assert infer_requested_mode(q) is None, q


def test_capture_keeps_current_and_queue_when_history_is_long():
    """A long station must not save as sixty tracks of ancient history with the
    song that is playing right now cut off the end."""
    from app.api.media_playlists import _capture_live_station, _MAX_TRACKS
    from app.agent.radio import get_radio_manager

    mgr = get_radio_manager()
    uid = "d" * 32
    station = _tracks(90, prefix="long")
    sess = mgr.enable(
        user_id=uid, channel="app", seed_intent="long station",
        seed_track=SeedTrack(video_id="longseed001", title="Long Seed"),
        station=station,
    )
    # Burn through 70 tracks so played_history is far longer than the cap.
    for _ in range(70):
        t = mgr.pop_next_from_playlist(sess)
        assert t is not None
        mgr.record_auto_play(sess, t)

    _, tracks = _capture_live_station(uid, "app")
    ids = [t["video_id"] for t in tracks]
    assert len(ids) <= _MAX_TRACKS
    assert len(ids) == len(set(ids))
    # The audible track is in there…
    assert sess.current_station_track.video_id in ids
    # …and so is what comes next, which is the half of a station that matters.
    remaining = [t.video_id for t in sess.playlist[sess.playlist_cursor:]]
    assert remaining, "test setup: station should still have queued tracks"
    assert remaining[0] in ids
    # And the oldest history is what got dropped, not the present.
    assert "longseed001" not in ids


@pytest.mark.asyncio
async def test_skip_next_abandons_if_radio_turned_off_while_queued(monkeypatch):
    """The advance lock can be held for tens of seconds by a refill. A user who
    turns radio OFF while a skip waits on it must not have music start again."""
    import app.api.ws_chat as wc
    from app.agent.radio import get_radio_manager

    mgr = get_radio_manager()
    uid = "e" * 32
    sess = mgr.enable(
        user_id=uid, channel="app", seed_intent="skip off",
        seed_track=SeedTrack(video_id="skipseed001", title="Skip Seed"),
        station=_tracks(5, prefix="skip"),
    )
    advanced: list = []

    async def _fake_advance(user_id, channel, s, trigger="?"):
        advanced.append(trigger)

    sent: list = []

    async def _fake_bcast(user_id, payload):
        sent.append(payload)
        return 1

    monkeypatch.setattr(wc, "_advance_and_broadcast_next", _fake_advance)
    monkeypatch.setattr(wc, "broadcast_to_user", _fake_bcast)

    # Hold the lock, fire the skip, disable radio, then release — exactly the
    # ordering the review described.
    async with wc._media_ended_lock(uid, "app"):
        task = asyncio.create_task(
            wc._handle_radio_skip_next(uid, {"channel": "app"})
        )
        await asyncio.sleep(0)          # let it reach the lock and park
        mgr.disable(uid, "app", source="test_toggle_off")
    await task

    # No frame at all proves the skip got PAST the entry check and parked on the
    # lock (an early return broadcasts radio_state error=not_enabled) — without
    # that, this test would pass for the wrong reason.
    assert sent == [], f"skip returned early instead of parking on the lock: {sent}"
    assert advanced == [], "a skip queued behind the lock advanced a disabled station"


@pytest.mark.asyncio
async def test_play_playlist_resolves_variants_before_shipping_window(
    _playlists_app, monkeypatch,
):
    """Every other producer of an upcoming window pre-resolves its variants;
    this one skipped it, so the phone prefetched ids the backend would never
    pop."""
    import app.api.ws_chat as wc
    from app.agent.radio import get_radio_manager

    client = _playlists_app
    uid = "p" * 32
    order: list = []

    async def _spy_resolve(sess, n=8):
        order.append("resolve")

    def _spy_upcoming(sess, n=5):
        order.append("upcoming")
        return []

    async def _noop_broadcast(**kwargs):
        return True

    async def _noop_bcast(user_id, payload):
        return 1

    monkeypatch.setattr(wc, "_resolve_upcoming_variants", _spy_resolve)
    monkeypatch.setattr(wc, "_upcoming_tracks", _spy_upcoming)
    monkeypatch.setattr("app.agent.radio.player.broadcast_radio_track", _noop_broadcast)
    monkeypatch.setattr(wc, "broadcast_to_user", _noop_bcast)

    created = await client.post("/api/media/playlists", json={
        "name": "Variant order",
        "tracks": [
            {"video_id": "varone00001", "title": "One"},
            {"video_id": "vartwo00002", "title": "Two"},
        ],
    })
    assert created.status_code == 200, created.text
    pid = created.json()["id"]

    played = await client.post(f"/api/media/playlists/{pid}/play", json={"channel": "app"})
    assert played.status_code == 200, played.text
    assert order[:2] == ["resolve", "upcoming"], f"order was {order}"
    assert get_radio_manager().get(uid, "app") is not None


# ── Round 2: the founder's 2026-08-03 recordings ──────────────────────────

def test_resolver_never_randomizes_a_named_track():
    """The screenshot failure: "Playing Daman Zardoo now" over ONEDAM - Del
    Tanha. variety=true must not randomize when the request names a song."""
    from app.agent import media_resolve as mr

    pool = [
        ("v_wrong1", "Del Tanha", "ONEDAM"),
        ("v_right", "Daman Zardoo", "Mammrez"),
        ("v_wrong2", "Something Else", "Someone"),
    ]
    # Specific ask → the match, every time, even with variety on.
    for _ in range(20):
        assert mr.pick_best("daman zardoo", pool, variety=True)[0] == "v_right"
    # Open-ended ask → free to vary among the artist's tracks.
    ebi = [("a", "Gheseh Eshgh", "Ebi"), ("b", "Delbar", "Ebi"), ("c", "Sarab", "Ebi")]
    picked = {mr.pick_best("ebi", ebi, variety=True)[0] for _ in range(40)}
    assert len(picked) > 1, "an open-ended ask should not always return the same track"


def test_resolver_reports_a_mismatch_to_the_model():
    from app.agent import media_resolve as mr

    warn = mr.describe_mismatch("daman zardoo", "Del Tanha", "ONEDAM")
    assert warn and "could not find" in warn.lower()
    assert "daman zardoo" in warn
    # A correct resolution says nothing, and an open-ended ask cannot mismatch.
    assert mr.describe_mismatch("daman zardoo", "Daman Zardoo", "Mammrez") is None
    assert mr.describe_mismatch("ebi", "Del Tanha", "ONEDAM") is None


def test_scrape_never_borrows_a_neighbours_title():
    """id and title must come from the SAME result block."""
    from app.agent import media_resolve as mr

    html = (
        '{"videoId":"aaaaaaaaaaa","other":1}'          # no title of its own
        '{"videoId":"bbbbbbbbbbb","title":{"runs":[{"text":"Real Title B"}]}}'
    )
    got = mr.scrape_results(html, limit=5)
    ids = [g[0] for g in got]
    assert "aaaaaaaaaaa" not in ids, "a result with no title must be dropped, not back-filled"
    assert ("bbbbbbbbbbb", "Real Title B", "") in got


@pytest.mark.asyncio
async def test_duplicate_toggle_reships_instead_of_rebuilding(monkeypatch):
    """A typed play fires the backend's _auto toggle AND the client's reseed.
    Building twice re-announces the playing seed with different artwork and
    leaves the phone prefetching a station that was thrown away."""
    import app.api.ws_chat as wc
    from app.agent.radio import get_radio_manager

    mgr = get_radio_manager()
    uid = "f" * 32
    sess = mgr.enable(
        user_id=uid, channel="app", seed_intent="play me ebi",
        seed_track=SeedTrack(video_id="dupseed0001", title="Ebi - Gheseh Eshgh"),
        station=_tracks(6, prefix="dup"),
    )
    builds: list = []
    sent: list = []

    async def _no_build(*a, **k):
        builds.append(a)
        return None, _tracks(6, prefix="rebuilt")

    async def _bcast(user_id, payload):
        sent.append(payload)
        return 1

    async def _resolve(_s, n=8):
        return None

    monkeypatch.setattr("app.agent.radio.build_station", _no_build, raising=False)
    monkeypatch.setattr(wc, "broadcast_to_user", _bcast)
    monkeypatch.setattr(wc, "_resolve_upcoming_variants", _resolve)

    await wc._handle_radio_toggle(uid, {
        "channel": "app", "enabled": True,
        "video_id": "dupseed0001", "title": "Ebi - Gheseh Eshgh",
    })

    assert builds == [], "the duplicate toggle rebuilt the station"
    kinds = [p.get("type") for p in sent]
    assert "radio_state" in kinds, "the live station's state must still be re-shipped"
    assert "media_play" not in kinds, "the already-playing seed must not be re-announced"
    # The original station survived intact.
    assert mgr.get(uid, "app").playlist[0].video_id.startswith("dup")


@pytest.mark.asyncio
async def test_station_is_autosaved_to_the_library(monkeypatch, _playlists_app):
    """Every station the agent builds becomes a library entry — the founder's
    'Toup Media must contain EVERY playlist the user has ever played'."""
    from app.api.media_playlists import autosave_station
    from app.agent.radio import get_radio_manager

    client = _playlists_app
    uid = "p" * 32
    mgr = get_radio_manager()
    sess = mgr.enable(
        user_id=uid, channel="app", seed_intent="play me ebi",
        seed_track=SeedTrack(video_id="autoseed001", title="Ebi - Gheseh Eshgh"),
        station=_tracks(5, prefix="auto"),
    )

    await autosave_station(uid, "app")
    listed = (await client.get("/api/media/playlists")).json()
    mine = [p for p in listed if p["source"] == "auto"]
    assert mine, "the station was not captured"
    assert mine[0]["name"] == "Ebi radio", mine[0]["name"]
    assert mine[0]["track_count"] >= 6
    first_id = mine[0]["id"]
    assert sess.autosave_playlist_id == first_id

    # Playing on updates the SAME entry rather than stacking duplicates.
    t = mgr.pop_next_from_playlist(sess)
    mgr.record_auto_play(sess, t)
    await autosave_station(uid, "app")
    again = [p for p in (await client.get("/api/media/playlists")).json() if p["source"] == "auto"]
    assert len(again) == 1, "autosave created a second row for the same station"
    assert again[0]["id"] == first_id


@pytest.mark.asyncio
async def test_removing_a_track_curates_the_playlist(_playlists_app):
    client = _playlists_app
    created = (await client.post("/api/media/playlists", json={
        "name": "Curate me",
        "tracks": [
            {"video_id": "keepone0001", "title": "Keep"},
            {"video_id": "dropone0001", "title": "Drop"},
        ],
    })).json()
    pid = created["id"]

    r = await client.delete(f"/api/media/playlists/{pid}/tracks/dropone0001")
    assert r.status_code == 200, r.text
    body = r.json()
    assert [t["video_id"] for t in body["tracks"]] == ["keepone0001"]
    assert body["track_count"] == 1
    # Curated by hand → no longer a passive recording an autosave may overwrite.
    assert body["source"] != "auto"
    # Removing something that isn't there is a 404, not a silent success.
    assert (await client.delete(f"/api/media/playlists/{pid}/tracks/nothere0001")).status_code == 404


@pytest.mark.asyncio
async def test_concurrent_autosaves_create_one_row(_playlists_app):
    """The station's creation hook and its first advance fire within
    milliseconds; both would find no row yet and INSERT."""
    from app.api.media_playlists import autosave_station
    from app.agent.radio import get_radio_manager

    client = _playlists_app
    uid = "p" * 32
    mgr = get_radio_manager()
    mgr.enable(
        user_id=uid, channel="app", seed_intent="race me",
        seed_track=SeedTrack(video_id="raceseed001", title="Racer - Race Song"),
        station=_tracks(4, prefix="race"),
    )
    await asyncio.gather(*[autosave_station(uid, "app") for _ in range(5)])
    rows = [p for p in (await client.get("/api/media/playlists")).json()
            if p["seed_intent"] == "race me"]
    assert len(rows) == 1, f"concurrent autosaves created {len(rows)} rows"


@pytest.mark.asyncio
async def test_autosave_never_undoes_a_user_edit_or_deletion(_playlists_app):
    """The library mirrors a station, but the user's hand always wins: a
    removed song must not come back, and a deleted playlist must not return,
    on the next track advance."""
    from app.api.media_playlists import autosave_station
    from app.agent.radio import get_radio_manager

    client = _playlists_app
    uid = "p" * 32
    mgr = get_radio_manager()
    sess = mgr.enable(
        user_id=uid, channel="app", seed_intent="edit me",
        seed_track=SeedTrack(video_id="editseed001", title="Editor - Edit Song"),
        station=_tracks(5, prefix="edit"),
    )
    await autosave_station(uid, "app")
    row = [p for p in (await client.get("/api/media/playlists")).json()
           if p["seed_intent"] == "edit me"][0]
    full = (await client.get(f"/api/media/playlists/{row['id']}")).json()
    victim = full["tracks"][1]["video_id"]

    # The user removes a song…
    r = await client.delete(f"/api/media/playlists/{row['id']}/tracks/{victim}")
    assert r.status_code == 200
    # …and the station plays on.
    await autosave_station(uid, "app")
    after = (await client.get(f"/api/media/playlists/{row['id']}")).json()
    assert victim not in [t["video_id"] for t in after["tracks"]], "autosave restored a deleted song"

    # The user deletes the playlist outright; the station plays on.
    assert (await client.delete(f"/api/media/playlists/{row['id']}")).status_code in (200, 204)
    await autosave_station(uid, "app")
    remaining = [p for p in (await client.get("/api/media/playlists")).json()
                 if p["seed_intent"] == "edit me"]
    assert remaining == [], "autosave resurrected a deleted playlist"


@pytest.mark.asyncio
async def test_replaying_a_saved_playlist_does_not_re_record_it(_playlists_app, monkeypatch):
    import app.api.ws_chat as wc
    from app.api.media_playlists import autosave_station

    client = _playlists_app

    async def _noop_broadcast(**kwargs):
        return True

    async def _noop_bcast(user_id, payload):
        return 1

    async def _noop_resolve(_s, n=8):
        return None

    monkeypatch.setattr("app.agent.radio.player.broadcast_radio_track", _noop_broadcast)
    monkeypatch.setattr(wc, "broadcast_to_user", _noop_bcast)
    monkeypatch.setattr(wc, "_resolve_upcoming_variants", _noop_resolve)

    created = (await client.post("/api/media/playlists", json={
        "name": "Saved list",
        "tracks": [{"video_id": "savedone001", "title": "One"},
                   {"video_id": "savedtwo002", "title": "Two"}],
    })).json()
    before = len((await client.get("/api/media/playlists")).json())

    assert (await client.post(f"/api/media/playlists/{created['id']}/play",
                              json={"channel": "app"})).status_code == 200
    await autosave_station("p" * 32, "app")

    after = (await client.get("/api/media/playlists")).json()
    assert len(after) == before, "replaying a saved playlist recorded a duplicate"


def test_resolver_folds_diacritics_so_ascii_queries_match_the_real_catalogue():
    """People type "beyonce halo"; the catalogue says "Beyoncé - Halo". Without
    folding, every accented word scores a MISS, a random ASCII-spelled
    re-upload outranks the official recording, and describe_mismatch fires on a
    correct result — so the agent apologises for the song it is playing.
    Reproduced against live YouTube results before this fix."""
    from app.agent import media_resolve as mr

    cases = [
        ("beyonce halo", "Beyoncé - Halo",
         "Beyonce - Halo - Acoustic: LIVE! Hospital SINGAPORE 2009"),
        ("titi me pregunto", "Bad Bunny - Tití Me Preguntó (Official Video)",
         "Titi Me Pregunto - Bad Bunny (Audio/Estudio) 2022"),
        ("celine dion my heart will go on", "Céline Dion - My Heart Will Go On",
         "Celine Dion - My Heart Will Go On (Lyrics)"),
    ]
    for query, official, reupload in cases:
        assert mr.relevance(query, official) == 1.0, query
        # Rank order is (official first) — it must win, not the ASCII re-upload.
        pick = mr.pick_best(query, [("off", official, ""), ("re", reupload, "")], variety=False)
        assert pick[0] == "off", f"{query} picked the re-upload"
        # …and no false apology for a correct result.
        assert mr.describe_mismatch(query, official) is None, query

    # Letters NFKD cannot decompose, which people still type as ASCII.
    assert mr.relevance("bjork joga", "Björk - Jóga") == 1.0
    assert mr.relevance("motorhead ace of spades", "Motörhead - Ace of Spades") == 1.0
    # A genuine mismatch is still a mismatch.
    assert mr.describe_mismatch("daman zardoo", "Del Tanha", "ONEDAM")


@pytest.mark.asyncio
async def test_fast_path_tells_the_model_the_track_is_starting_not_already_playing(monkeypatch):
    """The [SYSTEM] line must not assert audible playback.

    Broadcasting `media_play` is all that has happened when this line is
    written; on the audio-first phone path the device still has to resolve,
    fetch and buffer, which was 12.5 seconds in the 2026-08-03 recording. The
    old wording said the track "is ALREADY playing", and the model repeated it:
    a fresh "Play me moein" was answered with "Moein's already playing" over
    silence, nine seconds before the first sound — and "already" additionally
    denies the request it is fulfilling.
    """
    import httpx
    from app.api import ws_chat
    from app.agent import media_resolve as _mr

    class _Resp:
        text = "<html>irrelevant — the resolver is stubbed</html>"

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, *a, **k):
            return _Resp()

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    monkeypatch.setattr(_mr, "scrape_results", lambda html, limit=6: [("vid12345678", "Gasam Be Ashgh - Moein", None)])
    monkeypatch.setattr(_mr, "pick_best", lambda q, c, variety=False: ("vid12345678", "Gasam Be Ashgh - Moein", 1.0))
    monkeypatch.setattr(ws_chat, "_check_age_and_swap", lambda *a, **k: asyncio.sleep(0))

    queue: asyncio.Queue = asyncio.Queue()
    result = await ws_chat._fast_media_check("Play me moein", "user-1", queue)

    assert result is not None, "a titled result must take the fast path"
    modified_text, _meta = result
    lowered = modified_text.lower()

    # The regression itself.
    assert "already playing" not in lowered, (
        "the [SYSTEM] line claims audible playback the server cannot verify — "
        f"got: {modified_text!r}"
    )
    # And it must still say something, or the model invents its own framing.
    assert "start" in lowered, "the line must tell the model the track is starting"
    # The resolved title still has to be bound, or the model announces the query.
    assert "Gasam Be Ashgh - Moein" in modified_text


def test_resolver_demotes_download_farm_rips_but_still_answers():
    """A download-site rip must lose to the real upload — without being banned.

    2026-08-03: a Lori request resolved to "دانلود آهنگ لری ار نبینمش خر…", a
    download-site rip with a dead thumbnail. Pure word-overlap cannot see the
    problem, because stuffing the user's exact words into the title is the
    whole SEO strategy — the rip scored at or above the legitimate upload.
    """
    from app.agent import media_resolve as mr

    assert mr.junk_score("Ar Nabinamesh - Lori Song") == 0.0
    assert mr.junk_score("دانلود آهنگ لری ار نبینمش") > 0.0
    # The markers hide inside brackets, which normalize() would have stripped.
    assert mr.junk_score("Ar Nabinamesh [دانلود]") > 0.0

    # The rip is FIRST in search rank and matches every word; it must still lose.
    candidates = [
        ("spam1234567", "دانلود آهنگ لری ار نبینمش | www.example.ir", ""),
        ("real1234567", "ار نبینمش - آهنگ لری", ""),
    ]
    vid, _title, _artist = mr.pick_best("ار نبینمش لری", candidates)
    assert vid == "real1234567"

    # But when the farms are all that exists, we still play something.
    only_spam = [("spam1234567", "دانلود آهنگ لری ار نبینمش", "")]
    assert mr.pick_best("ار نبینمش لری", only_spam)[0] == "spam1234567"

    # Quality never outranks relevance: a clean title that answers a DIFFERENT
    # request must not beat the junky one that actually answers this one.
    mixed = [
        ("clean1234567", "Some Entirely Other Song", ""),
        ("junky1234567", "دانلود ار نبینمش لری", ""),
    ]
    assert mr.pick_best("ار نبینمش لری", mixed)[0] == "junky1234567"
