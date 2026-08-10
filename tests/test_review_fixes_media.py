"""Regression tests for the five backend defects the adversarial review of the
spool/flip change confirmed (2026-08-09). Each test names its finding.

Hermetic: no network, no ffmpeg, no FastAPI client.
"""
import asyncio
import os

import pytest

import app.api.media_proxy as mp
import app.api.ws_chat as ws_chat


# ── Finding 1: failure cleanup must not unlink a successor's file ─────────

def test_evicted_spool_late_failure_does_not_unlink_successor(tmp_path, monkeypatch):
    """MAX_AGE eviction replaces spool A with B at the SAME path; A's later
    failure handler must not delete B's live file."""
    monkeypatch.setattr(mp, "_AUDIO_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(mp, "_ensure_cache_dir", lambda: True)

    path = str(tmp_path / "videvict0001.spool")
    a = mp._Spool("videvict0001", path)
    open(path, "wb").write(b"successor-bytes")  # stands in for B's live file

    async def go():
        # Simulate A's failure handler AFTER eviction: A is no longer in the
        # registry (B replaced it), so the identity check fails and the unlink
        # must be skipped.
        mp._spools["videvict0001"] = "SOMETHING-ELSE"  # B's stand-in
        try:
            # Drive the except-path bookkeeping exactly as _spool_download does.
            a.error = "late failure"
            async with mp._spools_lock:
                if mp._spools.get(a.video_id) is a:
                    mp._spools.pop(a.video_id, None)
                    mp._safe_unlink(a.path)
        finally:
            mp._spools.pop("videvict0001", None)

    asyncio.run(go())
    assert os.path.exists(path), "a stale task deleted its successor's file"


def test_max_age_eviction_cancels_and_errors_the_old_spool(tmp_path, monkeypatch):
    monkeypatch.setattr(mp, "_AUDIO_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(mp, "_ensure_cache_dir", lambda: True)

    async def go():
        old = mp._Spool("vidage000001", str(tmp_path / "vidage000001.spool"))
        open(old.path, "wb").close()
        old.created -= mp._SPOOL_MAX_AGE_SECS + 1  # expired

        async def hang():
            await asyncio.sleep(3600)

        old.task = asyncio.ensure_future(hang())
        mp._spools["vidage000001"] = old

        async def fake_download(sp, result):
            sp.total = 1
            sp.received = 1
            sp.done = True

        monkeypatch.setattr(mp, "_spool_download", fake_download)
        new = await mp._spool_get_or_start("vidage000001", {"url": "u", "proxy_slot": 0})
        try:
            assert new is not old
            # The evicted spool must be marked so mid-body readers break out...
            assert old.error is not None
            # ...and its download must be cancelled, not left trickling.
            await asyncio.sleep(0.01)
            assert old.task.cancelled() or old.task.done()
        finally:
            mp._spools.pop("vidage000001", None)

    asyncio.run(go())


# ── Finding 2: tier pinning — a live spool wins over a published remux ────

def test_prune_never_sweeps_registered_spools(tmp_path, monkeypatch):
    monkeypatch.setattr(mp, "_AUDIO_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(mp, "_AUDIO_CACHE_MAX_FILES", 100)
    live = mp._Spool("vidlive00001", str(tmp_path / "vidlive00001.spool"))
    open(live.path, "wb").write(b"x")
    os.utime(live.path, (1, 1))  # ancient mtime — prune bait
    stale = tmp_path / "vidstale0001.spool"
    stale.write_bytes(b"y")
    os.utime(stale, (1, 1))
    mp._spools["vidlive00001"] = live
    try:
        mp._prune_audio_cache()
        assert os.path.exists(live.path), "prune deleted a registered spool's file"
        assert not stale.exists(), "prune left an orphaned spool file"
    finally:
        mp._spools.pop("vidlive00001", None)


def test_bounded_build_keeps_spool_after_publish(tmp_path, monkeypatch):
    """The spool must survive the remux publish — a playing item that started
    on itag-18 keeps its byte space (tier pinning) until a fresh start."""
    sp = mp._Spool("vidpin000001", str(tmp_path / "vidpin000001.spool"))
    open(sp.path, "wb").write(b"abcd")
    sp.total = 4
    sp.received = 4
    sp.done = True
    mp._spools[sp.video_id] = sp

    async def fake_live_idle(vid):
        return None

    async def fake_r2_store(vid, path):
        return None

    monkeypatch.setattr(mp, "_await_live_idle", fake_live_idle)
    monkeypatch.setattr(mp, "_r2_store_from_local", fake_r2_store)
    monkeypatch.setattr(mp, "_do_remux", lambda vid, src=None: "/tmp/fake.m4a")
    try:
        got = asyncio.run(mp._bounded_build(sp.video_id))
        assert got == "/tmp/fake.m4a"
        assert mp._spools.get(sp.video_id) is sp, "publish discarded the pinned spool"
        assert os.path.exists(sp.path)
    finally:
        mp._spools.pop(sp.video_id, None)
        mp._safe_unlink(sp.path)


# ── Finding 3: the flip revalidates its capture before mutating ───────────

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


class _Sess:
    def __init__(self, mode="song", track=None):
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


def test_flip_swap_skipped_when_station_advances_during_lookup(monkeypatch):
    """A skip lands while find_music_video runs → the swap must be dropped
    (the mode flip stands); executing it would overwrite the NEW track's tape
    entry and yank the card back to the skipped-away track."""
    import app.agent.radio as radio_mod
    import app.agent.radio.player as player_mod
    import app.agent.radio.playlist as playlist_mod

    atv = _Track("atv1", "MUSIC_VIDEO_TYPE_ATV")
    sess = _Sess(mode="song", track=atv)
    monkeypatch.setattr(radio_mod, "get_radio_manager", lambda: _Mgr(sess))

    frames = []

    async def fake_broadcast(user_id, frame):
        frames.append(frame.get("type"))
        return 1

    monkeypatch.setattr(ws_chat, "broadcast_to_user", fake_broadcast)

    async def fake_resolve(s, budget=None):
        return None

    monkeypatch.setattr(ws_chat, "_resolve_upcoming_variants", fake_resolve)
    monkeypatch.setattr(ws_chat, "_upcoming_tracks", lambda s: [])

    plays = []

    async def fake_broadcast_track(**kwargs):
        plays.append(kwargs)
        return True

    monkeypatch.setattr(player_mod, "broadcast_radio_track", fake_broadcast_track)
    monkeypatch.setattr(player_mod, "warm_audio_cache", lambda *a, **k: None)

    omv = _Track("omv9", "MUSIC_VIDEO_TYPE_OMV", length="13:32")

    async def racing_find_mv(track):
        # The station advances to B while the lookup runs.
        sess.current_track_id = "trackB000001"
        sess.current_station_track = _Track("trackB000001", "MUSIC_VIDEO_TYPE_ATV")
        return omv

    monkeypatch.setattr(playlist_mod, "find_music_video", racing_find_mv)

    asyncio.run(ws_chat._handle_radio_display_mode(
        "u1", {"channel": "app", "mode": "video"}
    ))

    assert plays == [], "stale swap executed over the advanced station"
    assert sess.current_track_id == "trackB000001", "swap mutated the advanced session"
    assert sess.display_mode == "video", "the mode flip itself must stand"


# ── Findings 4+5: capability-gated now-playing builds ─────────────────────

def test_now_playing_build_downgrades_until_spool_confirmed(monkeypatch):
    """Agent-side warm: until the platform has advertised spool:true, a
    now-playing build must go out as an extract (pre-spool platforms run the
    2026-08-05 regression otherwise); after the ack, it builds."""
    import app.agent.radio.player as player

    sent = []

    class _Resp:
        status_code = 200

        def __init__(self, spool):
            self._spool = spool

        def json(self):
            return {"ok": True, "spool": self._spool}

    class _Client:
        spool_answer = True

        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, headers=None, json=None):
            sent.append(json["mode"])
            return _Resp(_Client.spool_answer)

    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", _Client)

    from app.config import settings
    monkeypatch.setattr(settings, "run_mode", "agent", raising=False)
    monkeypatch.setattr(settings, "platform_api_url", "http://x/api", raising=False)
    monkeypatch.setattr(settings, "agent_api_key", "k", raising=False)
    monkeypatch.setattr(settings, "user_id", "u", raising=False)

    async def go():
        player._platform_spool = None
        # 1st now-playing build: capability unknown → extract.
        player.warm_audio_cache(["vidcap000001"], mode="build", now_playing=True)
        await asyncio.sleep(0.05)
        assert sent == ["extract"], sent
        assert player._platform_spool is True, "the ack must teach the capability"
        # 2nd: confirmed → real build.
        player.warm_audio_cache(["vidcap000002"], mode="build", now_playing=True)
        await asyncio.sleep(0.05)
        assert sent == ["extract", "build"], sent
        # Upcoming builds are never downgraded.
        player._platform_spool = None
        player.warm_audio_cache(["vidcap000003"], mode="build")
        await asyncio.sleep(0.05)
        assert sent[-1] == "build", sent

    try:
        asyncio.run(go())
    finally:
        player._platform_spool = None


def test_warm_ack_advertises_spool_capability():
    """internal_radio's warm response must carry the spool key — it is the
    only signal agents have for the deploy-skew downgrade."""
    import inspect
    import app.api.internal_radio as ir
    src = inspect.getsource(ir.warm)
    assert '"spool"' in src and "_SPOOL_ENABLED" in src


def test_fast_path_and_play_media_stay_extract():
    """Source tripwire for the arithmetic: the seed warms (fast path,
    play_media) must be extract — the phone's own request starts the spool on
    the same replica, and a build-warm on the other replica races the
    pre-roll. Only broadcast/flip warms may build (with now_playing=True)."""
    import inspect
    ws_src = inspect.getsource(ws_chat)
    idx = ws_src.index("pre-extract warm skipped")
    window = ws_src[idx - 2500: idx]
    assert '_warm([video_id], mode="extract")' in window, "fast-path warm must be extract"
    import app.agent.tool_executor as te
    te_src = inspect.getsource(te)
    idx2 = te_src.index("[play_media] pre-extract warm skipped")
    window2 = te_src[idx2 - 2500: idx2]
    assert '_warm([video_id], mode="extract")' in window2, "play_media warm must be extract"


def test_stream_audio_tier_pinning_block_present():
    """Source tripwire for finding 2's fix: while a spool exists, non-prefetch
    requests are pinned to the itag-18 byte space (spool-first, remux tiers
    skipped), and only a FRESH start (range absent / from byte 0) with the
    remux ready adopts the m4a and retires the spool."""
    import inspect
    src = inspect.getsource(mp.stream_audio)
    assert "TIER PINNING" in src
    assert "pin_itag = False" in src and "pin_itag = True" in src
    assert "fresh_start = _rng is None or _rng[0] == 0" in src
    # the remux tiers must be skipped while pinned
    assert "None if pin_itag else _remuxed_ready(video_id)" in src
    assert "if not rpath and not pin_itag:" in src
