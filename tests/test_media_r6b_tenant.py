"""R6-4b skew-safe scope order + R6-9 tenant findings (addendum 6; contract
v0.3 §7 "Amendment R6").

Owner requirement 7 / pattern B: the caller's newest media request wins; a
stop halts exactly the plays requested before it; the spoken outcome matches
the device.

R6-4b replaces the single total key (scope rank, order) and its tenant-clock
rank with a PAIRWISE rule between a halt (or the item on the device) and a
play:
  * same scope          -> orders (no stamp needed);
  * different scopes, both stamped -> stamps (`media_scope_started_ms`, the
    relay's app-floored hybrid logical clock); equal stamps break on the order
    in which the tenant first saw the scopes;
  * different scopes, either unstamped -> the arrival rule for that pair.
No tenant clock is read for ordering.

R6-9: T1 an ordered play is fenced by the item already on the device when
that item beats it; T2 one normalized modal form; T3 the tail strip never
eats a title; T4 one broadcast chokepoint; T5 structural Persian negation;
T6 the English trailing tag; residual 4 `paused: true`; residual 5 Arabic
punctuation inside a token.

Everything drives the REAL endpoints, play tool, control, radio player and
ws_chat fast path through the rig of `tests/test_media_order_scope_r6.py`;
only the YouTube search and the socket fan-out are stubbed.
"""
# The shared `rig` fixture is imported and then requested by name, which ruff
# reads as a redefinition.
# ruff: noqa: F811
from __future__ import annotations

import ast
import asyncio
import hashlib
import inspect
import itertools
import pathlib
import time

import pytest

from app.agent import media_intent
from app.agent.media_intent import media_request
from app.agent.radio import control
from app.api import api_v1, ws_chat
from app.api.api_v1 import MediaControlRequest
from test_media_order_scope_r6 import (  # noqa: F401  (rig is a fixture)
    SCOPE,
    USER,
    _collect,
    _halt,
    _play,
    _played,
    _Req,
    _Runner,
    _turn,
    _types,
    rig,
)

A, B = "psid-r6b-scope-A", "psid-r6b-scope-B"
# Scope B follows scope A in one call (the app reconnected). B's relay replica
# had a wall clock BEHIND A's (say 9_500 ms), but the app echoed A's stamp as
# its floor, so B's stamp is A + 1 (R6-4b).
T_A, T_B = 10_000, 10_001
T_EQ = 20_000


def _media_frames(frames):
    return [(f["type"], f.get("video_id", "")) for f in frames if f["type"].startswith("media_")]


def _last_media(frames):
    media = _media_frames(frames)
    return media[-1] if media else None


async def _settle(task):
    return await asyncio.wait_for(task, timeout=3.0)


# ══ 1. Cross-replica skew (the app floor makes B = A + 1) ══════════════════

async def test_skew_new_scope_stop_fences_the_old_scope_pending_play(rig):
    gate = rig.hold("Halo")
    play = _play("Halo", order=7, scope=A, started=T_A)
    await asyncio.wait_for(rig.searching["Halo"].wait(), timeout=2.0)
    stop = await _halt(rig, before=1, scope=B, started=T_B)
    assert (stop["ok"], stop["reason"]) == (True, "stopped"), stop
    gate.set()
    out = await _settle(play)
    assert (out["ok"], out["reason"]) == (False, "superseded"), out
    assert "media_play" not in _types(rig.frames)


async def test_skew_late_old_scope_stop_never_halts_the_new_scope_play(rig):
    gate = rig.hold("some jazz")
    play = _play("some jazz", order=1, scope=B, started=T_B)
    await asyncio.wait_for(rig.searching["some jazz"].wait(), timeout=2.0)
    # Nothing is on the device yet: the late stop is applied to it, but it
    # never supersedes the newer scope's play.
    late = await _halt(rig, before=9, scope=A, started=T_A)
    assert late["ok"] is True, late
    gate.set()
    out = await _settle(play)
    assert out["ok"] is True and out["video_id"] == "JAZZ0000001", out
    # Once B's item is on the device, a late A stop answers newer_playing and
    # sends nothing.
    before = list(rig.frames)
    again = await _halt(rig, before=10, scope=A, started=T_A)
    assert (again["ok"], again["reason"]) == (False, "newer_playing"), again
    assert (again["video_id"], again["title"]) == ("JAZZ0000001", "Jazz Mix")
    assert rig.frames == before
    assert _last_media(rig.frames) == ("media_play", "JAZZ0000001")


# ══ 2. Late FIRST arrival: the old scope is first seen after the new one ═══

async def test_late_first_arrival_old_scope_play_after_the_new_scope_stop(rig):
    """The tenant never saw A before its late play: B's stop came first."""
    assert (await _halt(rig, before=1, scope=B, started=T_B))["ok"] is True
    out = await _settle(_play("Halo", order=7, scope=A, started=T_A))
    assert (out["ok"], out["reason"]) == (False, "superseded"), out
    # The new scope's own later play is untouched.
    assert (await _settle(_play("some jazz", order=2, scope=B, started=T_B)))["ok"] is True
    assert _played(rig.frames) == ["JAZZ0000001"]


async def test_late_first_arrival_old_scope_stop_after_the_new_scope_play(rig):
    """The tenant never saw A before its late stop: B's play came first."""
    assert (await _settle(_play("some jazz", order=1, scope=B, started=T_B)))["ok"] is True
    before = list(rig.frames)
    late = await _halt(rig, before=9, scope=A, started=T_A)
    assert (late["ok"], late["reason"]) == (False, "newer_playing"), late
    assert rig.frames == before
    # A pending B play is never halted by another late A stop either.
    gate = rig.hold("Halo")
    play = _play("Halo", order=2, scope=B, started=T_B)
    await asyncio.wait_for(rig.searching["Halo"].wait(), timeout=2.0)
    late2 = await _halt(rig, before=10, scope=A, started=T_A)
    assert late2["reason"] == "newer_playing", late2
    gate.set()
    out = await _settle(play)
    assert out["ok"] is True and out["video_id"] == "HALO0000001", out
    assert _last_media(rig.frames) == ("media_play", "HALO0000001")


# ══ 3. Equal stamps: first-seen breaks the tie, deterministically ══════════

async def test_equal_stamps_scope_seen_first_is_older(rig):
    """A is seen first, then B (same stamp: e.g. two devices): B is newer."""
    gate = rig.hold("Halo")
    play = _play("Halo", order=7, scope=A, started=T_EQ)
    await asyncio.wait_for(rig.searching["Halo"].wait(), timeout=2.0)
    assert (await _halt(rig, before=1, scope=B, started=T_EQ))["ok"] is True
    gate.set()
    assert (await _settle(play))["reason"] == "superseded"
    # No cycle: B's play is not superseded by A's halt, and a later A stop
    # leaves B's item alone.
    assert (await _settle(_play("some jazz", order=2, scope=B, started=T_EQ)))["ok"] is True
    late = await _halt(rig, before=8, scope=A, started=T_EQ)
    assert (late["ok"], late["reason"]) == (False, "newer_playing"), late
    assert _last_media(rig.frames) == ("media_play", "JAZZ0000001")


async def test_equal_stamps_the_other_arrival_order_flips_the_roles_only(rig):
    """B is seen first (its stop), then A's play: A is the newer scope now."""
    assert (await _halt(rig, before=1, scope=B, started=T_EQ))["ok"] is True
    out = await _settle(_play("Halo", order=7, scope=A, started=T_EQ))
    assert out["ok"] is True, out
    late = await _halt(rig, before=2, scope=B, started=T_EQ)
    assert (late["ok"], late["reason"]) == (False, "newer_playing"), late
    assert late["video_id"] == "HALO0000001"
    # …and B's next play IS fenced by A's item (A beats B on first-seen).
    out = await _settle(_play("some jazz", order=3, scope=B, started=T_EQ))
    assert (out["ok"], out["reason"]) == (False, "superseded"), out
    assert _last_media(rig.frames) == ("media_play", "HALO0000001")


def test_the_pairwise_relation_is_antisymmetric_and_has_no_cycle():
    """Pure: over every stamped event of a small domain the caller order is a
    strict total order (antisymmetric, transitive); across scopes it is never
    'same turn'; an unstamped cross-scope pair is incomparable."""
    Ev = control._OrderedEvent
    # One stamp per scope; s1 and s2 share a stamp (the first-seen tie).
    scopes = {"s1": (5, 1), "s2": (5, 2), "s3": (6, 3)}
    events = [
        Ev(scope=s, order=o, stamp=t, first_seen=fs, seq=0)
        for s, (t, fs) in scopes.items() for o in (1, 2, 3)
    ]
    cmp = control._caller_order
    for a, b in itertools.product(events, repeat=2):
        assert cmp(a, b) == -cmp(b, a), (a, b)
        if a.scope != b.scope:
            assert cmp(a, b) != 0
    for a, b, c in itertools.product(events, repeat=3):
        if cmp(a, b) > 0 and cmp(b, c) > 0:
            assert cmp(a, c) > 0, (a, b, c)
    unstamped = Ev(scope="s9", order=1, stamp=None, first_seen=9, seq=0)
    assert all(cmp(unstamped, e) is None and cmp(e, unstamped) is None for e in events)


# ══ 4. Missing stamp across scopes: arrival, both directions, no clock ═════

class _BackwardsClock:
    """A wall clock that goes BACKWARDS on every read."""

    def __init__(self, start: float = 1_900_000_000.0):
        self.now = start
        self.reads = 0

    def time(self) -> float:
        self.reads += 1
        self.now -= 1.0
        return self.now

    def time_ns(self) -> int:
        return int(self.time() * 1e9)


@pytest.mark.parametrize(
    ("a_stamp", "b_stamp"),
    [(None, None), (T_A, None), (None, T_B)],
    ids=["neither-stamped", "only-A-stamped", "only-B-stamped"],
)
async def test_missing_stamp_cross_scope_is_arrival_in_both_directions(rig, monkeypatch, a_stamp, b_stamp):
    clock = _BackwardsClock()
    monkeypatch.setattr(time, "time", clock.time)
    monkeypatch.setattr(time, "time_ns", clock.time_ns)
    # (1) B's stop lands BEFORE A's play arrives: by arrival it is older, so
    #     it does not supersede the play (whatever the clocks say).
    assert (await _halt(rig, before=5, scope=B, started=b_stamp))["ok"] is True
    out = await _settle(_play("Halo", order=7, scope=A, started=a_stamp))
    assert out["ok"] is True, out
    # (2) The device gate: A's item cannot be ordered against a B stop, and
    #     that stop arrived after the broadcast, so it is applied.
    stop = await _halt(rig, before=6, scope=B, started=b_stamp)
    assert (stop["ok"], stop["reason"]) == (True, "stopped"), stop
    # (3) A's next play is in flight when a B stop lands: arrival supersedes.
    gate = rig.hold("some jazz")
    play = _play("some jazz", order=8, scope=A, started=a_stamp)
    await asyncio.wait_for(rig.searching["some jazz"].wait(), timeout=2.0)
    assert (await _halt(rig, before=7, scope=B, started=b_stamp))["ok"] is True
    gate.set()
    assert (await _settle(play))["reason"] == "superseded"
    assert _played(rig.frames) == ["HALO0000001"]


def test_no_clock_is_read_for_ordering(monkeypatch):
    """Pure: every ordering decision runs with every clock unavailable."""

    def no_clock(*a, **kw):
        raise AssertionError("a clock was read for media ordering")

    monkeypatch.setattr(control, "_latest_halts", type(control._pending_acks)())
    monkeypatch.setattr(ws_chat, "_radio_off_generations", {})
    for name in ("time", "time_ns", "monotonic", "monotonic_ns", "perf_counter", "perf_counter_ns"):
        monkeypatch.setattr(time, name, no_clock)
    play = control.media_halt_mark(USER, media_order=7, media_scope=A)
    halt = control._ordered_event(USER, 1, B, None, seq=0)
    assert control._newer_playing(USER, halt) is None
    control._note_media_halt(USER, "stop", halt=halt)
    assert control.media_play_superseded(play) == "stop", "arrival: the stop came after"
    later = control.media_halt_mark(USER, media_order=8, media_scope=A)
    assert control.media_play_superseded(later) is None
    control.note_media_broadcast(USER, later, video_id="X", title="x")
    assert control.recorded_media_item(USER)["video_id"] == "X"


def test_the_control_module_imports_no_clock():
    tree = ast.parse(pathlib.Path(control.__file__).read_text(encoding="utf-8"))
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    } | {
        (node.module or "").split(".")[0]
        for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    assert not imported & {"time", "datetime"}, imported


# ══ 5. Same scope without a stamp: orders ══════════════════════════════════

async def test_same_scope_orders_without_any_stamp(rig):
    gate = rig.hold("some jazz")
    play = _play("some jazz", order=5, scope=SCOPE)
    await asyncio.wait_for(rig.searching["some jazz"].wait(), timeout=2.0)
    assert (await _halt(rig, before=4, scope=SCOPE))["ok"] is True
    gate.set()
    assert (await _settle(play))["ok"] is True, "an older stop never drops a newer play"
    older = await _halt(rig, before=4, scope=SCOPE)
    assert older["reason"] == "newer_playing", older
    newer = await _halt(rig, before=6, scope=SCOPE)
    assert (newer["ok"], newer["reason"]) == (True, "stopped"), newer
    assert _last_media(rig.frames) == ("media_stop", "")


# ══ 6. Unordered controls: legacy relay, phone X, typed stop ═══════════════

async def test_legacy_bodies_keep_the_arrival_rule_exactly(rig):
    gate = rig.hold("some jazz")
    play = _play("some jazz")
    await asyncio.wait_for(rig.searching["some jazz"].wait(), timeout=2.0)
    assert (await _halt(rig))["ok"] is True
    gate.set()
    assert (await _settle(play))["reason"] == "superseded"
    assert (await _settle(_play("Halo")))["ok"] is True
    # A legacy item is always stopped by a legacy stop, and a legacy play is
    # never fenced by the item on the device.
    assert (await _settle(_play("some jazz")))["ok"] is True
    assert (await _halt(rig))["reason"] == "stopped"
    assert _played(rig.frames) == ["HALO0000001", "JAZZ0000001"]


@pytest.mark.parametrize("halt", ["phone_x", "typed_stop"])
async def test_unordered_halts_keep_the_arrival_rule(rig, halt):
    async def unordered_halt():
        channel = "app" if halt == "phone_x" else "web"
        await ws_chat._handle_radio_toggle(
            USER, {"type": "radio_toggle", "channel": channel, "enabled": False},
        )

    gate = rig.hold("Halo")
    play = _play("Halo", order=3, scope=SCOPE, started=T_A)
    await asyncio.wait_for(rig.searching["Halo"].wait(), timeout=2.0)
    await unordered_halt()
    gate.set()
    assert (await _settle(play))["reason"] == "superseded"
    # An ordered item the X stopped is not "newer": a late older ordered
    # stop is applied (and the phone answers for itself).
    assert (await _settle(_play("some jazz", order=5, scope=SCOPE, started=T_A)))["ok"] is True
    await unordered_halt()
    late = await _halt(rig, before=4, scope=SCOPE, started=T_A)
    assert late["reason"] != "newer_playing", late


# ══ 7. R6-9 T1: the item on the device fences an older ordered play ═══════

async def test_t1_same_turn_is_not_fenced(rig):
    assert (await _settle(_play("some jazz", order=3, scope=SCOPE)))["ok"] is True
    out = await _settle(_play("Halo", order=3, scope=SCOPE))
    assert out["ok"] is True, "an equal key is the same turn"


async def test_t1_across_stamped_scopes(rig):
    assert (await _settle(_play("some jazz", order=1, scope=B, started=T_B)))["ok"] is True
    out = await _settle(_play("Halo", order=9, scope=A, started=T_A))
    assert (out["ok"], out["reason"]) == (False, "superseded"), out
    assert _played(rig.frames) == ["JAZZ0000001"]


@pytest.mark.parametrize("case", ["unordered_play", "unordered_item", "unstamped_other_scope"])
async def test_t1_incomparable_pairs_keep_the_arrival_rule(rig, case):
    """Pre-R6: a play never fenced another play by arrival."""
    if case == "unordered_play":
        first, second = dict(order=5, scope=SCOPE), dict()
    elif case == "unordered_item":
        first, second = dict(), dict(order=1, scope=SCOPE)
    else:
        first, second = dict(order=1, scope=B), dict(order=9, scope=A, started=T_A)
    assert (await _settle(_play("some jazz", **first)))["ok"] is True
    out = await _settle(_play("Halo", **second))
    assert out["ok"] is True, (case, out)
    assert _played(rig.frames) == ["JAZZ0000001", "HALO0000001"]


async def test_t1_inside_a_delegated_run_says_what_happened(rig, monkeypatch):
    runner = _Runner(rig.tools)
    monkeypatch.setattr(api_v1, "_agent_runner", runner)
    stream = await api_v1.internal_agent_turn_stream(_turn("Halo", 2, started=T_A), _Req())
    frames = asyncio.create_task(_collect(stream))
    await asyncio.wait_for(runner.thinking["Halo"].wait(), timeout=2.0)
    assert (await _settle(_play("some jazz", order=4, scope=SCOPE, started=T_A)))["ok"] is True
    runner.release["Halo"].set()
    stream_frames = await asyncio.wait_for(frames, timeout=5.0)
    result = runner.results["Halo"]
    assert result.startswith(control.PLAY_SUPERSEDED_PREFIX), result
    assert "newer play" in result and "Jazz Mix" in result and "stopped" not in result
    end = [f for f in stream_frames if f.get("type") == "tool.end"][0]
    assert (end["ok"], end.get("outcome")) == (False, "cancelled")
    assert _played(rig.frames) == ["JAZZ0000001"]


# ══ 8. R6-9 T4: one chokepoint records what actually plays ═════════════════

async def test_t4_the_typed_fast_path_item_is_what_the_gate_reads(rig):
    assert (await _settle(_play("some jazz", order=3, scope=SCOPE)))["ok"] is True
    q: asyncio.Queue = asyncio.Queue()
    typed = await asyncio.wait_for(ws_chat._fast_media_check("play Halo", USER, q), timeout=2.0)
    assert typed is not None and typed[1]["video_id"] == "HALO0000001"
    assert q.get_nowait()["type"] == "media_play"
    assert control.recorded_media_item(USER)["video_id"] == "HALO0000001"
    stop = await _halt(rig, before=2, scope=SCOPE)
    assert (stop["ok"], stop["reason"]) == (True, "stopped"), stop


async def test_t4_a_station_broadcast_is_what_the_gate_reads(rig):
    from app.agent.radio import player

    assert (await _settle(_play("some jazz", order=3, scope=SCOPE)))["ok"] is True
    delivered = await player.broadcast_radio_track(
        user_id=USER, video_id="STAT0000001", title="Station Next", channel="app",
    )
    assert delivered is True
    assert control.recorded_media_item(USER)["video_id"] == "STAT0000001"
    stop = await _halt(rig, before=2, scope=SCOPE)
    assert (stop["ok"], stop["reason"]) == (True, "stopped"), stop
    assert _last_media(rig.frames) == ("media_stop", "")


def test_t4_every_media_play_producer_goes_through_the_chokepoint():
    from app.agent.radio import player
    from app.agent.tool_executor import ToolExecutor

    for fn in (ToolExecutor._tool_play_media, ws_chat._fast_media_check, player.broadcast_radio_track):
        assert "send_media_play(" in inspect.getsource(fn), fn.__qualname__
    app_dir = pathlib.Path(control.__file__).resolve().parents[2]
    callers = sorted(
        str(p.relative_to(app_dir)) for p in app_dir.rglob("*.py")
        if "note_media_broadcast(" in p.read_text(encoding="utf-8")
    )
    assert callers == ["agent/radio/control.py"], callers
    source = inspect.getsource(control)
    assert source.count("note_media_broadcast(") == 2  # its def + the chokepoint
    producers = sorted(
        str(p.relative_to(app_dir)) for p in app_dir.rglob("*.py")
        if '"type": "media_play"' in p.read_text(encoding="utf-8")
    )
    # ws_browser's frame goes only to its own /ws/browser socket (the web
    # hub's browser-agent pane), never to the user's chat sockets the stop
    # reaches — not a device this gate speaks for (R6-10 TA7; the evidence is
    # pinned in tests/test_media_r6c_tenant.py).
    assert producers == [
        "agent/radio/player.py", "agent/tool_executor.py", "api/ws_browser.py", "api/ws_chat.py",
    ], producers


# ══ 9. newer_playing: the wire shape, and paused (residual 4) ══════════════

async def test_newer_playing_shape_and_paused(rig):
    assert (await _settle(_play("some jazz", order=5, scope=SCOPE)))["ok"] is True
    plain = await _halt(rig, before=4, scope=SCOPE)
    assert {k: plain[k] for k in (
        "ok", "changed", "reason", "video_id", "title", "command_id",
        "acked_devices", "silent_devices",
    )} == {
        "ok": False, "changed": False, "reason": "newer_playing",
        "video_id": "JAZZ0000001", "title": "Jazz Mix", "command_id": "",
        "acked_devices": 0, "silent_devices": 0,
    }
    assert "paused" not in plain, "additive only when the item is paused"
    pause = await _halt(rig, "pause", before=6, scope=SCOPE)
    assert (pause["ok"], pause["reason"]) == (True, "paused"), pause
    assert "paused" not in pause
    late = await _halt(rig, before=4, scope=SCOPE)
    assert (late["reason"], late.get("paused")) == ("newer_playing", True), late
    assert late["video_id"] == "JAZZ0000001"
    # A new item is not paused.
    assert (await _settle(_play("Halo", order=7, scope=SCOPE)))["ok"] is True
    again = await _halt(rig, before=4, scope=SCOPE)
    assert again["reason"] == "newer_playing" and "paused" not in again, again


def test_halts_are_kept_for_the_newest_scopes_only(monkeypatch):
    monkeypatch.setattr(control, "_latest_halts", type(control._pending_acks)())
    for i in range(control._MAX_SCOPES + 8):
        control._note_media_halt(
            USER, "stop", halt=control._ordered_event(USER, 1, f"s{i}", 5 + i, seq=0),
        )
    state = control._user_state(USER)
    assert len(state.scopes) == control._MAX_SCOPES
    assert list(state.scopes)[-1] == f"s{control._MAX_SCOPES + 7}" and "s0" not in state.scopes


def test_the_endpoint_keeps_legacy_bodies_legacy():
    req = MediaControlRequest(user_id="u", action="stop", before_order=2, media_scope="s",
                              media_scope_started_ms=True)
    assert (req.before_order, req.media_scope, req.media_scope_started_ms) == (2, "s", None)
    # A zero or out-of-range stamp is unstamped (the pair falls to arrival).
    assert control.media_scope_stamp(0) is None and control.media_scope_stamp(2**53) is None
    assert control.media_scope_stamp(True) is None and control.media_scope_stamp("12") == 12


# ══ 10. media_intent: T2, T3, T5, T6, residual 5 ═══════════════════════════

@pytest.mark.parametrize("modal", [
    "میشه", "می‌شه", "می شه", "مي‌شه",
    "میتونی", "می‌تونی", "می تونی",
    "میخوام", "می‌خوام",
])
def test_t2_every_spelling_of_a_modal_is_the_same_entry(modal):
    ask = media_request(f"{modal} یه آهنگ از ابی پخش کنی؟")
    assert ask is not None and ask.query == "ابی", (modal, ask)


def test_t2_the_modal_set_has_one_entry_per_modal():
    folded = [media_intent._class_words(m) for m in media_intent.MODAL_FRAMING_TEXT]
    assert len(folded) == len(set(folded)), folded


@pytest.mark.parametrize(("text", "query"), [
    # a named title after the media noun / a title marker is never dropped
    ("پخش کن آهنگ سلام", "سلام"),
    ("بذار آهنگ hello", "hello"),
    ("یه آهنگ از ادل پخش کن به اسم hello", "ادل اسم hello"),
    # PIN CHANGED (addendum 6 R6-17, writer:r7-tenant; was «بذار آهنگ someone
    # like you» → 'someone like you'): 'someone' (indefinite) and 'you'
    # (pronoun) are words of the closed trouble classes, so that title takes
    # the agent path (tests/test_media_r7_tenant.py). The same T3 fact — a
    # Latin title after «بذار آهنگ» is never eaten — is pinned on a title
    # with no trouble word:
    ("بذار آهنگ bohemian rhapsody", "bohemian rhapsody"),
    ("بذار آهنگ «hello»", "hello"),
    # CONTROL: a class word inside the title
    ("آهنگ سلام آخر رو بذار", "سلام آخر"),
    # framing after the play verb / after a clause boundary still goes
    ("یه آهنگ از ابی بذار مرسی", "ابی"),
    ("یه آهنگ از ابی بذار، مرسی", "ابی"),
    ("باشه یه آهنگ بذار مرسی", "آهنگ"),
])
def test_t3_the_tail_strip_never_eats_a_title(text, query):
    ask = media_request(text)
    assert ask is not None and ask.query == query, (text, ask)


@pytest.mark.parametrize("text", [
    "دیگه آهنگ پخش نشه", "آهنگ نباید پخش بشه",
    "آهنگ پخش نکن", "نمی‌خوام آهنگ پخش کنی", "نه یه پادکست بذار",
    "آهنگ پخش نشود", "دیگه آهنگ نذارید", "به این آهنگ گوش نده",
])
def test_t5_a_negated_verb_anywhere_declines(text):
    assert media_request(text) is None, text


@pytest.mark.parametrize(("text", "query"), [
    # Words that merely start with «ن» are not negations.
    ("یه آهنگ از نامجو پخش کن", "نامجو"),
    ("یه نماهنگ از ابی پخش کن", "ابی"),
])
def test_t5_controls(text, query):
    ask = media_request(text)
    assert ask is not None and ask.query == query, (text, ask)


@pytest.mark.parametrize(("text", "query"), [
    ("play some jazz, would you?", "some jazz"),
    ("play Halo by Beyonce, could you?", "Halo by Beyonce"),
    ("play some jazz, okay?", "some jazz"),
    ("play some jazz, please", "some jazz"),
    # PIN CHANGED (addendum 6 R6-11 TB1, writer:r6d-tenant; was "some jazz,
    # please" per R6-10 TA2, originally "some jazz" per R6-9 T6): the grammar
    # is [closed tail tags] and a title span ends at the caller's framing
    # ('please'), so every closed tail tag at the end leaves the query. What
    # keeps a title whole is structure, not a count: a tag is a closed
    # assent/modal/politeness phrase (never 'well' / 'hey'), and a strip never
    # leaves a head made only of class words ('Okay, Okay', 'Oh, Yeah' stay
    # whole — tests/test_media_r6c_tenant.py TA2, test_media_r6d_tenant.py).
    ("play some jazz, please, would you?", "some jazz"),
    # a comma inside the title is not a tag. PIN CHANGED (addendum 6 R6-17,
    # writer:r7-tenant; was 'play Thank U, Next'): 'next' is a word of the
    # closed temporal/sequence class, so that title takes the agent path; the
    # comma-in-a-title fact is pinned on titles with no trouble word.
    ("play Well, Well, Well", "Well, Well, Well"),
    ("play Okay, Okay", "Okay, Okay"),
    ("play Halo", "Halo"),
])
def test_t6_the_english_trailing_tag_is_not_searched(text, query):
    ask = media_request(text)
    assert ask is not None and ask.query == query, (text, ask)


@pytest.mark.parametrize("text", [
    "can you play Halo by Beyonce", "I want to listen to jazz",
    "can you play some jazz, please",
])
def test_t6_leading_framing_still_declines(text):
    assert media_request(text) is None, text


def test_residual5_arabic_punctuation_inside_a_token_is_a_boundary():
    for text, query in (
        ("آره،یه آهنگ از ابی پخش کن", "ابی"),
        ("آره،آهنگ ابی رو بذار", "ابی"),
        ("یه آهنگ از ابی بذار،مرسی", "ابی"),
    ):
        ask = media_request(text)
        assert ask is not None and ask.query == query, (text, ask)


async def test_both_fast_paths_read_the_same_residue(rig):
    """The typed half (ws_chat) searches it; the relay's `_media_request` IS
    `media_intent.media_request`."""
    from app.services import live_voice_protocol as lvp

    for text, query in (
        ("آره،یه آهنگ از ابی پخش کن", "ابی"),
        ("می‌شه یه پادکست پخش کنی؟", "پادکست"),
        ("play some jazz, would you?", "some jazz"),
    ):
        assert lvp._media_request(text) == media_request(text), text
        q: asyncio.Queue = asyncio.Queue()
        before = len(rig.searches)
        result = await asyncio.wait_for(ws_chat._fast_media_check(text, USER, q), timeout=2.0)
        assert result is not None and rig.searches[before:] == [query], (text, rig.searches)


def test_the_three_shared_lexicon_sets_are_unchanged():
    """The relay's closed class, spelled in media_intent, is not touched."""
    digest = hashlib.sha256("\x00".join((
        media_intent.ASSENT_WORDS_TEXT,
        "\x01".join(media_intent.ASSENT_PHRASES_TEXT),
        media_intent.CONTENT_FREE_WORDS_TEXT,
    )).encode("utf-8")).hexdigest()
    assert digest == _SNAPF_LEXICON_SHA256, digest


# sha256 of the three texts as shipped in the snapF media_intent (3bacb9b1…).
_SNAPF_LEXICON_SHA256 = "340573129ee54fad6149d719cfb2a6325ecd9d58667a619d1a1a3d7d84ec7c97"
