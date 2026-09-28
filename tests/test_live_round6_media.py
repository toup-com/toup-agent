"""R2 round 6 — the relay's media half (addendum 6 R6-4, R6-4b, R6-5, R6-6 +
the `newer_playing` verdict wording of contract v0.3 §7).

Binding spec: /private/tmp/toup-voice-r2-spec-addendum-6.md (R6-4, R6-4b,
R6-5, R6-6, R6-9 residual 4) and /private/tmp/toup-voice-protocol-contract-
v0.3.md §7.  Findings: r5-round-results.txt / r5-verify-results.txt rows
media-1 … media-6, media-2b, media-x1.

  R6-4  ONE order-keyed media-intent ledger: a stop/pause halts exactly the
        plays asked for BEFORE it, on every path; the newest request wins.
        Every stop is sent at fire time, scoped (`before_order`), never
        withheld or deferred; a newer request's tenant call never overtakes an
        older halt.
  R6-4b the scope stamp: max(wall ms, app floor + 1, this process's last stamp
        for the user + 1); `ready.media_scope_started_ms` only for a client that
        lists `media_scope_floor`; every ordered tenant request carries
        media_order|before_order + media_scope + media_scope_started_ms, and
        never an order without a stamp.
  R6-5  which question is open: an assent to the agent's question that
        RESTATES a relay-owned command continues it (fired: absorbed; armed:
        joins the unit and the delegation binds to it — one stop, no card, a
        function result with the verdict); a refusal is never absorbed and runs
        with the truthful context; a consent card is titled from its proposal.
  R6-6  the relay never quotes the caller's sentence as a track title.
  §7    `newer_playing` (+ `paused`) is worded as "the newer request is what
        plays" — never "stopped", never media_control_failed (en + fa).

WIRE LEVEL: the real relay, the real `ws_realtime._play_media_direct` /
`_control_media_direct` / `_think` building their request bodies, and a fake
TENANT behind `_vps_api` / `_vps_api_stream` that records every body and
implements contract v0.3 §7 (R6-4b pairwise order keys, the device gate and
`newer_playing`, the T1 broadcast fence).  Agent tool starts use
`api_v1._vs_args`, exactly what production sends.  Every socket has its own
FakeProvider session id.

Old-fail is proven by running THIS file against a copy of the frozen merged
snapshot fx4-relay-snapF (lvp 052f1459…): the behaviour tests fail there by
assertion; the controls pass on both trees.
"""

from __future__ import annotations

import asyncio
import itertools
import json
from typing import Optional

import pytest

import test_live_harness as H
import test_live_round4 as R4
from app.api import api_v1 as A
from app.api import ws_realtime as rt
from app.config import settings
from app.services import live_voice_protocol as live


V03 = R4.V03
V03_FLOOR = V03 + ["media_scope_floor"]
TF132 = R4.TF132
Phone = R4.Phone
_send = R4._send
_wait = R4._wait
_phases = R4._phases
_frames_for = R4._frames_for
_commentary = R4._commentary
_thinking = R4._thinking

#: The real tenant-facing helpers, captured before any test patches them.
REAL_THINK = rt._think
REAL_PLAY = rt._play_media_direct
REAL_CONTROL = rt._control_media_direct

_ids = itertools.count(1)
ORDER_KEYS = ("media_order", "before_order", "media_scope", "media_scope_started_ms")


def _tag(name: str) -> str:
    return f"{name}-{next(_ids)}"


def _said(provider) -> list[str]:
    return _commentary(provider) + [
        str(e.get("content") or "") for e in provider.of("session.instructions.append")
    ]


# ══════════════════════════════════════════════════════════════════════
# The fake TENANT (contract v0.3 §7, R6-4b pairwise order)
# ══════════════════════════════════════════════════════════════════════

class Tenant:
    """Records every request body and answers like the §7 tenant.

    A play (fast path, or a `play_media` inside a delegated run) takes its
    arrival mark and its order key (scope, stamp, order) when its REQUEST
    arrives; it is superseded iff some recorded halt beats it — pairwise: same
    scope → orders; different scopes both stamped → stamps; either side
    unordered → arrival.  An ordered play is also fenced by the recorded
    broadcast item when that item beats it (T1).  A stop/pause device-stops
    only an item that is older than it (or unordered); a NEWER item answers
    `newer_playing` (+ `paused: true` when paused).
    """

    def __init__(self, *, searches: Optional[dict] = None, agents: Optional[dict] = None,
                 device: Optional[dict] = None, guard: bool = True):
        #: `guard=False`: a tenant whose stale-play guard missed the halt (it
        #: broadcasts anyway) — the relay's own F29 re-assert must then repair
        #: it, scoped like the halt it repeats.
        self.guard = guard
        self.bodies: list[tuple[str, dict]] = []
        self.events: list[tuple[str, str]] = []
        self.seq = 0
        self.halts: list[dict] = []
        self.first_seen: dict[str, int] = {}
        self.device = device            # {"title","video_id","key","paused"}
        self.searches = searches or {}  # query substring → (delay_s, found)
        self.agents = agents or {}      # display_request substring → script dict

    # ── order keys ────────────────────────────────────────────────────
    def key(self, body: dict, field: str):
        order, scope, stamp = body.get(field), body.get("media_scope"), body.get("media_scope_started_ms")
        if isinstance(order, int) and isinstance(scope, str) and scope and isinstance(stamp, int):
            self.first_seen.setdefault(scope, len(self.first_seen))
            return (scope, stamp, order)
        return None

    def beats(self, a, b) -> bool:
        if a[0] == b[0]:
            return a[2] > b[2]
        if a[1] != b[1]:
            return a[1] > b[1]
        return self.first_seen[a[0]] > self.first_seen[b[0]]

    def superseded(self, key, mark) -> bool:
        if not self.guard:
            return False
        for halt in self.halts:
            if key is not None and halt["key"] is not None:
                if self.beats(halt["key"], key):
                    return True
            elif halt["seq"] > mark:
                return True
        item = self.device
        return bool(
            key is not None and item is not None and item.get("key") is not None
            and self.beats(item["key"], key)
        )

    def search(self, query: str) -> tuple[float, bool]:
        for needle, spec in self.searches.items():
            if needle in query.lower():
                return spec
        return (0.2, True)

    def broadcast(self, title: str, key) -> dict:
        video_id = (title.upper().replace(" ", "") + "0000000000")[:11]
        self.device = {"title": title, "video_id": video_id, "key": key, "paused": False}
        self.events.append(("media_play", title))
        return {"type": "youtube", "video_id": video_id, "title": title}

    # ── endpoints ─────────────────────────────────────────────────────
    async def play_media(self, body: dict) -> dict:
        self.seq += 1
        mark, key = self.seq, self.key(body, "media_order")
        query = str(body.get("query") or "")
        self.events.append(("play_requested", query))
        delay, found = self.search(query)
        await asyncio.sleep(delay)
        if self.superseded(key, mark):
            self.events.append(("play_superseded", query))
            return {"ok": False, "reason": "superseded"}
        if not found:
            self.events.append(("play_not_found", query))
            return {"ok": False, "reason": "not_found"}
        media = self.broadcast(query, key)
        return {"ok": True, "title": media["title"], "video_id": media["video_id"]}

    async def media_control(self, body: dict) -> dict:
        self.seq += 1
        action = str(body.get("action") or "")
        self.events.append(("control", action))
        if action not in {"stop", "pause"}:
            return {"ok": True, "reason": "executed", "title": "Next Song"}
        key = self.key(body, "before_order")
        self.halts.append({"seq": self.seq, "key": key})
        item = self.device
        if (
            key is not None and item is not None and item.get("key") is not None
            and not self.beats(key, item["key"])
        ):
            verdict = {
                "ok": False, "changed": False, "reason": "newer_playing",
                "video_id": item["video_id"], "title": item["title"],
                "command_id": "", "acked_devices": 0, "silent_devices": 0,
            }
            if item.get("paused"):
                verdict["paused"] = True
            self.events.append(("newer_playing", item["title"]))
            return verdict
        if item is None:
            return {"ok": False, "reason": "nothing_playing", "acked_devices": 1, "silent_devices": 0}
        if action == "stop":
            self.device = None
            self.events.append(("media_stop", item["title"]))
            return {"ok": True, "reason": "stopped", "changed": True, "acked_devices": 1}
        item["paused"] = True
        self.events.append(("media_pause", item["title"]))
        return {"ok": True, "reason": "paused", "changed": True, "acked_devices": 1}

    async def agent_turn(self, body: dict, relay) -> tuple:
        self.seq += 1
        mark, key = self.seq, self.key(body, "media_order")
        request = str(body.get("display_request") or "")
        self.events.append(("agent_run", request))
        script = next(
            (spec for needle, spec in self.agents.items() if needle in request.lower()), {},
        )
        await asyncio.sleep(float(script.get("think_s", 0.1)))
        query = script.get("play")
        if not query or relay is None:
            return "stream", {"text": script.get("text", "Here is what I found.")}, 1
        args = script.get("args")
        await relay.on_event({
            "type": "tool.start", "call_id": "c-play", "name": "play_media",
            "args": A._vs_args("play_media", {"query": query}) if args is None else args,
        })
        await asyncio.sleep(float(script.get("search_s", 0.3)))
        if self.superseded(key, mark):
            self.events.append(("play_superseded", query))
            await relay.on_event({"type": "tool.end", "call_id": "c-play", "name": "play_media",
                                  "ok": False, "outcome": "cancelled", "elapsed_ms": 10})
            return "stream", {"text": "I won't start it."}, 2
        media = self.broadcast(query, key)
        await relay.on_event({"type": "tool.end", "call_id": "c-play", "name": "play_media",
                              "ok": True, "elapsed_ms": 10})
        return "stream", {"text": f"Starting {query}.", "media": media}, 2

    # ── the transport seams ────────────────────────────────────────────
    def install(self, monkeypatch) -> None:
        async def vps(_user_id):
            return ("http://tenant.invalid", "key")

        async def vps_api(_url, _key, method, path, params=None, json_body=None, **_kw):
            body = dict(json_body or {})
            self.bodies.append((path, body))
            if path == "/api/v1/internal/play-media":
                return await self.play_media(body)
            if path == "/api/v1/internal/media-control":
                return await self.media_control(body)
            if path == "/api/v1/internal/agent-turn":
                outcome, payload, _n = await self.agent_turn(body, None)
                return payload
            return None

        async def vps_api_stream(_url, _key, path, json_body, relay):
            body = dict(json_body or {})
            self.bodies.append((path, body))
            return await self.agent_turn(body, relay)

        monkeypatch.setattr(rt, "_think", REAL_THINK)
        monkeypatch.setattr(rt, "_play_media_direct", REAL_PLAY)
        monkeypatch.setattr(rt, "_control_media_direct", REAL_CONTROL)
        monkeypatch.setattr(rt, "_get_vps_info", vps)
        monkeypatch.setattr(rt, "_vps_api", vps_api)
        monkeypatch.setattr(rt, "_vps_api_stream", vps_api_stream)
        monkeypatch.setattr(rt, "_agent_runner", None)
        monkeypatch.setattr(settings, "voice_realtime_v2", True)
        monkeypatch.setattr(settings, "voice_realtime_tool_events", True)

    # ── readers ────────────────────────────────────────────────────────
    def of(self, suffix: str) -> list[dict]:
        return [body for path, body in self.bodies if path.endswith(suffix)]

    def controls(self, action: str = "stop") -> list[dict]:
        return [b for b in self.of("/internal/media-control") if b.get("action") == action]

    def plays(self) -> list[dict]:
        return self.of("/internal/play-media")

    def runs(self) -> list[dict]:
        return self.of("/internal/agent-turn/stream") + self.of("/internal/agent-turn")

    def kinds(self) -> list[str]:
        return [kind for kind, _what in self.events]


OLD_SONG = {"title": "Fadat Sham", "video_id": "OLDSONG0001", "key": None, "paused": False}


def _old_song() -> dict:
    return dict(OLD_SONG)


async def _call(monkeypatch, tenant: Tenant, steps, *, grace_ms=1500, features=V03_FLOOR,
                frames=(), config_extra=None, user="", research=False, timeout=24):
    """One socket: `steps(p, client)` scripts the caller and the model."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=3.0)
    monkeypatch.setattr(settings, "voice_live_media_backstop_grace_ms", grace_ms)
    H.patch_relay(monkeypatch)
    tenant.install(monkeypatch)
    box: dict = {}
    tag = _tag("r6m")

    def on_send(p, e):
        box["p"] = p
        if e["type"] == "session.close":
            p.push(H.closed())

    async def script():
        await _wait(lambda: "p" in box and client.of("ready"), 4.0)
        await asyncio.sleep(0.1)
        await steps(box["p"], client)

    client = Phone([
        H.config(features=list(features), **(config_extra or {})), *frames, script, {"type": "stop"},
    ])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True, session_id=f"live-psid-{tag}")
    await H.run_relay(
        client, provider, timeout=timeout, db_session_id=f"db-{tag}", user_id=user or f"user-{tag}",
    )
    return client, provider


async def _acknowledged(client, count: int = 1) -> None:
    """The model's direct acknowledgement reached the phone — so it fenced the
    command turn as answered, and the next request is a turn of its own."""

    await _wait(lambda: len({f.get("response_id") for f in client.of("audio_delta")}) >= count, 3.0)
    await asyncio.sleep(0.15)


def _ready(client) -> dict:
    frames = client.of("ready")
    return frames[0] if frames else {}


def _stamp(client, provider) -> int:
    return int(_ready(client).get("media_scope_started_ms") or 0)


def _psid(provider) -> str:
    return provider._session_id


# ══════════════════════════════════════════════════════════════════════
# PURE — the structural rules
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize(("question", "action"), [
    ("قطعش کنم؟", "stop"),
    ("آهنگ رو قطع کنم؟", "stop"),
    ("Should I stop the music?", "stop"),
    ("Do you want me to stop it?", "stop"),
    ("Would you like me to pause the music?", "pause"),
    ("می‌خوای آهنگ رو قطع کنم؟", "stop"),
])
def test_r6_5_a_question_that_restates_the_command(question, action):
    assert live.question_restates_command(question, action), question


@pytest.mark.parametrize(("question", "action"), [
    ("می‌خوای به جاش یه پادکست برات پخش کنم؟", "stop"),
    ("Do you want me to find a podcast instead?", "stop"),
    ("Should I stop the search too?", "stop"),
    ("Anything else?", "stop"),
    ("قطعش نکنم؟", "stop"),
    ("Should I stop the music?", "pause"),        # the SAME action only
    ("Should I play the next song?", "stop"),
])
def test_r6_5_control_other_questions_restate_nothing(question, action):
    assert not live.question_restates_command(question, action), question


@pytest.mark.parametrize("answer", ["آره", "بله", "yes please", "کن", "کن لطفا", "sure, go ahead", "okay"])
def test_r6_5_pure_assents_continue(answer):
    assert live.media_answer_continues(answer), answer


@pytest.mark.parametrize("answer", ["نه", "no, don't", "yes but wait", "یه پادکست بذار", "اِ", "stop the search"])
def test_r6_5_control_refusals_hedges_and_content_never_continue(answer):
    assert not live.media_answer_continues(answer), answer


def test_r6_5_the_open_question_subject_is_the_proposal():
    assert live.question_subject("Do you want me to find a podcast instead?") == "find a podcast instead"
    assert live.question_subject("قطعش کنم؟") == "قطعش"
    assert "پادکست" in live.question_subject("می‌خوای به جاش یه پادکست برات پخش کنم؟")
    assert "می‌خوای" not in live.question_subject("می‌خوای به جاش یه پادکست برات پخش کنم؟")
    assert live.open_question_sentence("Stopped. Anything else?") == "Anything else?"
    assert live.open_question_sentence("I stopped it.") == ""


@pytest.mark.parametrize("raw", [True, False, "1790000000000", -5, 0, 2 ** 53, 2 ** 60, 1.5e12, None, [1]])
def test_r6_4b_malformed_floors_are_ignored(raw):
    assert live.valid_scope_floor(raw) == 0


def test_r6_4b_stamp_arithmetic(monkeypatch):
    wall = {"ms": 1_800_000_000_000}
    monkeypatch.setattr(live, "_scope_wall_ms", lambda: wall["ms"])
    user = _tag("u-stamp")
    # No floor: the wall clock.
    assert live.next_scope_stamp(user + "a") == wall["ms"]
    # A floor AHEAD of this replica's wall (another replica's clock is ahead):
    # floor + 1, never the wall.
    assert live.next_scope_stamp(user + "b", wall["ms"] + 5000) == wall["ms"] + 5001
    # floor == wall: strictly above it (no tie).
    assert live.next_scope_stamp(user + "c", wall["ms"]) == wall["ms"] + 1
    # Same process, same user, the wall stepped BACKWARDS: last + 1.
    first = live.next_scope_stamp(user + "d")
    wall["ms"] -= 60_000
    assert live.next_scope_stamp(user + "d") == first + 1
    # Another user is independent of that user's history.
    assert live.next_scope_stamp(user + "e") == wall["ms"]
    # A malformed floor is ignored (bool is not an int here).
    assert live.next_scope_stamp(user + "f", True) == wall["ms"]


def test_r6_4b_order_fields_are_all_or_nothing():
    cases = [
        ({"order": 3, "scope": "psid", "started_ms": 1_800_000_000_000}, True),
        ({"order": 3, "scope": "psid", "started_ms": 0}, False),          # no stamp → no order
        ({"order": 3, "scope": "", "started_ms": 1_800_000_000_000}, False),
        ({"order": 0, "scope": "psid", "started_ms": 1_800_000_000_000}, False),
        ({"order": True, "scope": "psid", "started_ms": 1_800_000_000_000}, False),
        ({"order": 3, "scope": "psid", "started_ms": 2 ** 53}, False),
        ({"order": 3, "scope": "psid", "started_ms": "1800000000000"}, False),
    ]
    for value, ordered in cases:
        token = rt.MEDIA_ORDER.set(value)
        try:
            play, halt = rt._media_order_fields(), rt._media_order_fields(halt=True)
        finally:
            rt.MEDIA_ORDER.reset(token)
        if ordered:
            assert play == {"media_order": 3, "media_scope": "psid", "media_scope_started_ms": value["started_ms"]}
            assert halt == {"before_order": 3, "media_scope": "psid", "media_scope_started_ms": value["started_ms"]}
        else:
            assert play == {} and halt == {}, value
    assert rt._media_order_fields() == {}


@pytest.mark.parametrize("lang", ["en", "fa"])
@pytest.mark.parametrize("paused", [False, True])
@pytest.mark.parametrize("title", ["Jazz Mix", ""])
def test_newer_playing_wording(lang, paused, title):
    for action in ("stop", "pause"):
        key, line = live.media_control_line(action, False, "newer_playing", lang, title, paused=paused)
        assert key.startswith("media_newer_paused" if paused else "media_newer_playing"), key
        assert key != "media_control_failed" and line
        assert line != live.relay_line("media_control_failed", lang)
        lowered = line.lower()
        assert "stopped" not in lowered and "stop" not in lowered
        assert "قطع" not in line and "نگه داشتم" not in line
        if title:
            assert f"«{title}»" in line
        if paused:
            assert "playing now" not in lowered and "داره پخش" not in line
        else:
            assert ("playing" in lowered) if lang == "en" else ("پخش" in line)


def test_the_neutral_wont_start_line():
    assert live.relay_line("media_wont_start_untitled", "en") == "OK — I won't start that."
    assert live.relay_line("media_wont_start_untitled", "fa") == "باشه، پخشش نمی‌کنم."


# ══════════════════════════════════════════════════════════════════════
# R6-4 — play Halo → stop → play jazz ends on jazz (fast + agent path)
# ══════════════════════════════════════════════════════════════════════

HALO_FAST = "play halo by beyonce"
HALO_AGENT = "can you play Halo by Beyonce"


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["fast", "agent"])
@pytest.mark.parametrize("jazz_at", ["inside_grace", "after_fire"])
async def test_r6_4_play_halo_stop_play_jazz_ends_on_jazz(monkeypatch, path, jazz_at):
    """media-1/-3: 'play Halo' still in flight → 'stop the music' (only
    'Okay.') → 'play some jazz'.  The stop reaches the tenant scoped to the
    turns before it: Halo (asked before it) never starts, jazz (asked after)
    plays — whenever the stop fires relative to the jazz request."""

    tenant = Tenant(
        searches={"halo": (2.2, True), "jazz": (0.2, True)},
        agents={"halo": {"think_s": 0.1, "play": "Halo Beyonce", "search_s": 2.2}},
        device=_old_song(),
    )

    async def steps(p, client):
        p.push(H.user_delta(HALO_FAST if path == "fast" else HALO_AGENT, 1000, 1600))
        p.push(H.delegation("dHalo", 1620))
        await _wait(lambda: any(k in ("play_requested", "agent_run") for k in tenant.kinds()), 3.0)
        await asyncio.sleep(0.3)
        p.push(H.user_delta("stop the music", 3000, 3500))
        p.push(H.out_text("Okay.", 3600, 3800))
        p.push(H.out_audio("OK"))
        await _acknowledged(client)
        await asyncio.sleep(0.1 if jazz_at == "inside_grace" else 1.8)
        p.push(H.user_delta("play some jazz", 5000, 5600))
        p.push(H.delegation("dJazz", 5620))
        await _wait(lambda: tenant.kinds().count("play_superseded") >= 1
                    and ("media_play", "some jazz") in tenant.events, 8.0)
        await asyncio.sleep(0.5)

    client, provider = await _call(
        monkeypatch, tenant, steps, frames=({"type": "now_playing", "title": "Fadat Sham"},),
    )
    assert ("media_play", "some jazz") in tenant.events, tenant.events
    assert not any(kind == "media_play" and "halo" in what.lower() for kind, what in tenant.events), tenant.events
    assert tenant.device is not None and tenant.device["title"] == "some jazz", tenant.events
    stops = tenant.controls("stop")
    assert len(stops) == 1, tenant.bodies
    # The stop is scoped: it halts the turns before it, never the jazz turn.
    halo_body = (
        next(b for b in tenant.plays() if "halo" in b["query"].lower()) if path == "fast"
        else tenant.runs()[0]
    )
    halo_order = halo_body["media_order"]
    jazz_order = [b for b in tenant.plays() if "jazz" in b["query"]][0]["media_order"]
    assert halo_order < stops[0]["before_order"] < jazz_order, tenant.bodies
    assert "completed" in _phases(client, "dJazz")


@pytest.mark.asyncio
async def test_r6_4_a_newer_agent_play_still_thinking_when_the_older_stop_fires_plays(monkeypatch):
    """media-4: the newer agent run reached the tenant BEFORE the older stop
    did (the stop's evidence arrived late, so it fired at the end of its
    grace).  An arrival-keyed guard would drop the newer play; the stop's
    `before_order` is older than the run's `media_order`, so it plays."""

    tenant = Tenant(
        agents={"halo": {"think_s": 2.4, "play": "Halo Beyonce", "search_s": 0.2}},
        device=_old_song(),
    )

    async def steps(p, client):
        p.push(H.user_delta("stop the music", 1000, 1500))
        p.push(H.out_text("Okay.", 1600, 1800))
        p.push(H.out_audio("OK"))
        await _acknowledged(client)
        p.push(H.user_delta(HALO_AGENT, 2200, 2900))
        p.push(H.delegation("dHalo", 2920))
        await _wait(lambda: "agent_run" in tenant.kinds(), 3.0)
        # Only now does the phone report the song — the stop's evidence.
        _send(client, {"type": "now_playing", "title": "Fadat Sham"})
        await _wait(lambda: ("media_play", "Halo Beyonce") in tenant.events, 6.0)
        await asyncio.sleep(0.4)

    client, provider = await _call(monkeypatch, tenant, steps)
    kinds = tenant.kinds()
    assert "control" in kinds and kinds.index("agent_run") < kinds.index("control"), tenant.events
    run, stop = tenant.runs()[0], tenant.controls("stop")[0]
    assert stop["before_order"] < run["media_order"], (stop, run)
    assert ("media_stop", "Fadat Sham") in tenant.events, tenant.events
    assert ("media_play", "Halo Beyonce") in tenant.events, tenant.events
    assert tenant.device and tenant.device["title"] == "Halo Beyonce"


@pytest.mark.asyncio
@pytest.mark.parametrize("play_text", [HALO_FAST, HALO_AGENT])
async def test_r6_4_a_play_asked_after_the_stop_is_never_halted(monkeypatch, play_text):
    """The deferred-then-newer shape: stop (acknowledged only) → a newer play
    inside the grace.  The stop goes out first (never withheld), the newer
    play is never superseded and no stop follows its broadcast."""

    tenant = Tenant(
        searches={"halo": (0.3, True)},
        agents={"halo": {"think_s": 0.5, "play": "Halo Beyonce", "search_s": 0.2}},
        device=_old_song(),
    )

    async def steps(p, client):
        p.push(H.user_delta("stop the music", 1000, 1500))
        p.push(H.out_text("Okay.", 1600, 1800))
        p.push(H.out_audio("OK"))
        await _acknowledged(client)
        p.push(H.user_delta(play_text, 2200, 2900))
        p.push(H.delegation("dHalo", 2920))
        await _wait(lambda: "media_play" in tenant.kinds(), 5.0)
        await asyncio.sleep(2.0)       # past the grace: nothing fires again

    client, provider = await _call(
        monkeypatch, tenant, steps, frames=({"type": "now_playing", "title": "Fadat Sham"},),
    )
    kinds = tenant.kinds()
    assert "media_play" in kinds, tenant.events
    assert "play_superseded" not in kinds, tenant.events
    assert kinds.count("control") == 1, tenant.events
    assert kinds.index("control") < kinds.index("media_play"), tenant.events
    assert "control" not in kinds[kinds.index("media_play"):], tenant.events


@pytest.mark.asyncio
async def test_r6_4_a_queued_then_cancelled_newer_play_leaves_the_old_song_stopped(monkeypatch):
    """media-5: the newer play waits at capacity and is cancelled there.  The
    stop was sent when it fired — nothing waited for the queued play — so the
    old song is stopped and nothing plays."""

    tenant = Tenant(
        agents={"dentist": {"think_s": 6.0}, "paris": {"think_s": 6.0}},
        device=_old_song(),
    )

    async def steps(p, client):
        p.push(H.user_delta("find me a dentist near union station", 500, 1100))
        p.push(H.delegation("dR1", 1120))
        await _wait(lambda: len(tenant.runs()) >= 1, 3.0)
        p.push(H.user_delta("what's the weather in paris this weekend", 1500, 2100))
        p.push(H.delegation("dR2", 2120))
        await _wait(lambda: len(tenant.runs()) >= 2, 3.0)
        p.push(H.user_delta("stop the music", 3000, 3500))
        p.push(H.out_text("Okay.", 3600, 3800))
        p.push(H.out_audio("OK"))
        await _acknowledged(client)
        p.push(H.user_delta(HALO_FAST, 4200, 4900))
        p.push(H.delegation("dPlay", 4920))
        await _wait(lambda: "pending" in _phases(client, "dPlay"), 3.0)
        await _wait(lambda: tenant.controls("stop"), 4.0)
        _send(client, {"type": "cancel_task", "task_id": "dPlay"})
        await _wait(lambda: "cancelled" in _phases(client, "dPlay"), 3.0)
        await asyncio.sleep(0.5)

    client, provider = await _call(
        monkeypatch, tenant, steps, frames=({"type": "now_playing", "title": "Fadat Sham"},),
        timeout=30,
    )
    assert "pending" in _phases(client, "dPlay") and "cancelled" in _phases(client, "dPlay")
    assert ("media_stop", "Fadat Sham") in tenant.events, tenant.events
    assert tenant.device is None, tenant.events
    assert not tenant.plays(), tenant.bodies
    assert len(tenant.controls("stop")) == 1


# ══════════════════════════════════════════════════════════════════════
# R6-4 controls — plain stop, no evidence, no order fields for legacy callers
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
@pytest.mark.parametrize("delegated", [True, False])
async def test_r6_4_control_a_plain_stop_of_a_playing_song(monkeypatch, delegated):
    tenant = Tenant(device=_old_song())

    async def steps(p, client):
        p.push(H.user_delta("stop the music", 1000, 1500))
        if delegated:
            p.push(H.delegation("dStop", 1520))
        else:
            p.push(H.out_text("Okay.", 1600, 1800))
            p.push(H.out_audio("OK"))
        await _wait(lambda: tenant.controls("stop"), 4.0)
        await asyncio.sleep(0.8)

    client, provider = await _call(
        monkeypatch, tenant, steps, grace_ms=300,
        frames=({"type": "now_playing", "title": "Fadat Sham"},),
    )
    assert ("media_stop", "Fadat Sham") in tenant.events and tenant.device is None
    (stop,) = tenant.controls("stop")
    assert stop["before_order"] == 1 and stop["media_scope"] == _psid(provider)
    assert stop["media_scope_started_ms"] == _stamp(client, provider)
    assert any("Stopped the music." in s for s in _said(provider)), _said(provider)


@pytest.mark.asyncio
async def test_r6_4_the_quiet_reassert_carries_the_halts_scope(monkeypatch):
    """F29's quiet re-assert (an older play the tenant broadcast after the
    stop) is scoped like the halt it repeats: `before_order` = that stop's
    order, the scope and the stamp — so it can never silence a newer item."""

    tenant = Tenant(searches={"jazz": (1.5, True)}, guard=False)

    async def steps(p, client):
        p.push(H.user_delta("play some jazz", 1000, 1500))
        p.push(H.delegation("dPlay", 1520))
        await _wait(lambda: tenant.plays(), 3.0)
        p.push(H.user_delta("stop the music", 2500, 3000))
        p.push(H.delegation("dStop", 3020))
        await _wait(lambda: len(tenant.controls("stop")) >= 2, 5.0)
        await asyncio.sleep(0.3)

    client, provider = await _call(monkeypatch, tenant, steps)
    stops = tenant.controls("stop")
    assert len(stops) == 2, tenant.bodies                  # the stop + one quiet re-assert
    (play,) = tenant.plays()
    for stop in stops:
        assert stop["before_order"] == 2 > play["media_order"] == 1, stops
        assert stop["media_scope"] == _psid(provider)
        assert stop["media_scope_started_ms"] == _stamp(client, provider) > 0
    assert ("media_stop", "some jazz") in tenant.events and tenant.device is None, tenant.events
    assert "cancelled" in _phases(client, "dPlay")


@pytest.mark.asyncio
async def test_r6_4_control_a_stop_with_no_evidence_never_calls_the_tenant(monkeypatch):
    tenant = Tenant(device=_old_song())

    async def steps(p, client):
        p.push(H.user_delta("stop the music", 1000, 1500))
        p.push(H.out_text("Okay.", 1600, 1800))
        p.push(H.out_audio("OK"))
        await asyncio.sleep(1.2)

    client, provider = await _call(monkeypatch, tenant, steps, grace_ms=300)
    assert tenant.bodies == []


@pytest.mark.asyncio
async def test_r6_4b_every_ordered_request_carries_its_order_scope_and_stamp(monkeypatch):
    """Every play-media, agent-turn and stop/pause body of the scope carries
    the three fields with this socket's provider session id and the stamp
    `ready` announced; next/previous carry none."""

    tenant = Tenant(
        searches={"jazz": (0.2, True)},
        agents={"weather": {"think_s": 0.1}},
        device=_old_song(),
    )

    async def steps(p, client):
        p.push(H.user_delta("what's the weather in paris", 1000, 1500))
        p.push(H.delegation("d1", 1520))
        await _wait(lambda: tenant.runs(), 3.0)
        p.push(H.user_delta("play some jazz", 2500, 3000))
        p.push(H.delegation("d2", 3020))
        await _wait(lambda: tenant.plays(), 3.0)
        await asyncio.sleep(0.5)
        p.push(H.user_delta("next song", 4000, 4400))
        p.push(H.delegation("d3", 4420))
        await _wait(lambda: tenant.controls("next"), 3.0)
        p.push(H.user_delta("pause the music", 5500, 6000))
        p.push(H.delegation("d4", 6020))
        await _wait(lambda: tenant.controls("pause"), 3.0)
        await asyncio.sleep(0.4)

    client, provider = await _call(
        monkeypatch, tenant, steps, frames=({"type": "now_playing", "title": "Fadat Sham"},),
    )
    stamp, psid = _stamp(client, provider), _psid(provider)
    assert stamp > 0
    (run,), (play,) = tenant.runs(), tenant.plays()
    (pause,), (nxt,) = tenant.controls("pause"), tenant.controls("next")
    assert (run["media_order"], run["media_scope"], run["media_scope_started_ms"]) == (1, psid, stamp)
    assert (play["media_order"], play["media_scope"], play["media_scope_started_ms"]) == (2, psid, stamp)
    assert (pause["before_order"], pause["media_scope"], pause["media_scope_started_ms"]) == (4, psid, stamp)
    assert "media_order" not in pause and "before_order" not in run and "before_order" not in play
    assert not any(k in nxt for k in ORDER_KEYS), nxt
    for _path, body in tenant.bodies:
        if "media_order" in body or "before_order" in body:
            assert body.get("media_scope") == psid and body.get("media_scope_started_ms") == stamp, body


@pytest.mark.asyncio
async def test_r6_4b_no_order_is_ever_sent_without_a_stamp(monkeypatch):
    monkeypatch.setattr(live, "next_scope_stamp", lambda _user, _floor=0: 0)
    tenant = Tenant(searches={"jazz": (0.1, True)}, agents={"weather": {"think_s": 0.1}}, device=_old_song())

    async def steps(p, client):
        p.push(H.user_delta("what's the weather in paris", 1000, 1500))
        p.push(H.delegation("d1", 1520))
        await _wait(lambda: tenant.runs(), 3.0)
        p.push(H.user_delta("play some jazz", 2500, 3000))
        p.push(H.delegation("d2", 3020))
        await _wait(lambda: tenant.plays(), 3.0)
        p.push(H.user_delta("stop the music", 4000, 4500))
        p.push(H.delegation("d3", 4520))
        await _wait(lambda: tenant.controls("stop"), 3.0)
        await asyncio.sleep(0.3)

    client, provider = await _call(monkeypatch, tenant, steps)
    assert "media_scope_started_ms" not in _ready(client)
    assert tenant.runs() and tenant.plays() and tenant.controls("stop")
    for _path, body in tenant.bodies:
        assert not any(k in body for k in ORDER_KEYS), body


@pytest.mark.asyncio
async def test_r6_4b_control_legacy_callers_send_the_old_bodies(monkeypatch):
    """Outside the Live relay (no order scope set) every body is exactly what
    it was: the realtime path, the typed path and every existing caller."""

    tenant = Tenant(searches={"halo": (0.0, True)})
    H.patch_relay(monkeypatch)
    tenant.install(monkeypatch)
    await rt._play_media_direct("user-1", "halo")
    await rt._control_media_direct("user-1", "stop")
    await rt._control_media_direct("user-1", "next")

    class _Relay:
        async def on_event(self, _ev):
            return True

        async def close_open(self, outcome="error"):
            return None

    await rt._think("user-1", "what's the weather", "db-x", relay=_Relay(), out={})
    paths = [path for path, _b in tenant.bodies]
    assert "/api/v1/internal/play-media" in paths and "/api/v1/internal/agent-turn/stream" in paths
    assert tenant.of("/internal/play-media") == [{"query": "halo", "variety": False}]
    assert tenant.controls("stop") == [{"user_id": "user-1", "action": "stop", "channel": "app"}]
    for _path, body in tenant.bodies:
        assert not any(k in body for k in ORDER_KEYS), body


# ══════════════════════════════════════════════════════════════════════
# R6-4b — the stamp on the wire: ready (negotiated) and the floor
# ══════════════════════════════════════════════════════════════════════

async def _stop_scope(monkeypatch, *, features=V03_FLOOR, config_extra=None, user=""):
    tenant = Tenant(device=_old_song())

    async def steps(p, client):
        p.push(H.user_delta("stop the music", 1000, 1500))
        p.push(H.delegation("dStop", 1520))
        await _wait(lambda: tenant.controls("stop"), 4.0)
        await asyncio.sleep(0.2)

    client, provider = await _call(
        monkeypatch, tenant, steps, features=features, config_extra=config_extra, user=user,
    )
    (stop,) = tenant.controls("stop")
    return client, provider, stop


@pytest.mark.asyncio
@pytest.mark.parametrize("listed", [True, False])
async def test_r6_4b_ready_carries_the_stamp_only_for_the_feature(monkeypatch, listed):
    features = V03_FLOOR if listed else V03
    client, provider, stop = await _stop_scope(monkeypatch, features=features)
    ready = _ready(client)
    if listed:
        assert ready["media_scope_started_ms"] == stop["media_scope_started_ms"] > 0
    else:
        assert "media_scope_started_ms" not in ready
        assert stop["media_scope_started_ms"] > 0          # the tenant still gets it
    assert "media_scope_floor" not in ready["capabilities"]


@pytest.mark.asyncio
async def test_r6_4b_tf132_ready_and_frames_are_byte_identical(monkeypatch):
    client, provider, stop = await _stop_scope(monkeypatch, features=TF132)
    ready = _ready(client)
    assert ready == {
        "type": "ready",
        "session_id": ready["session_id"],
        "capabilities": {
            "voice_tasks": False, "voice_provider": "live", "playback_ack": True,
            "audio": {"type": "pcm16le", "rate": 24000}, "live_turns": True,
            "delegation_frames": True, "media_fast_path": True, "task_lifecycle": True,
            "playback_frames": True, "reask_turns": True, "language_pref": True,
            "turn_timing": True, "media_control": True, "media_transport": True,
            "heard_text": True,
        },
    }
    dumped = json.dumps(client.frames)
    assert "media_scope" not in dumped and "before_order" not in dumped and "media_order" not in dumped
    assert "newer_playing" not in dumped


@pytest.mark.asyncio
async def test_r6_4b_a_floor_ahead_of_the_wall_orders_the_new_scope_after_it(monkeypatch):
    """Cross-replica skew: the app saw a stamp from a replica whose clock is
    ahead; this replica's stamp is floor + 1, never its own (older) wall."""

    wall = 1_800_000_000_000
    monkeypatch.setattr(live, "_scope_wall_ms", lambda: wall)
    floor = wall + 90_000
    client, provider, stop = await _stop_scope(monkeypatch, config_extra={"media_scope_floor_ms": floor})
    assert _ready(client)["media_scope_started_ms"] == floor + 1
    assert stop["media_scope_started_ms"] == floor + 1


@pytest.mark.asyncio
async def test_r6_4b_a_floor_equal_to_the_wall_never_ties(monkeypatch):
    wall = 1_800_000_100_000
    monkeypatch.setattr(live, "_scope_wall_ms", lambda: wall)
    client, provider, stop = await _stop_scope(monkeypatch, config_extra={"media_scope_floor_ms": wall})
    assert stop["media_scope_started_ms"] == wall + 1 == _ready(client)["media_scope_started_ms"]


@pytest.mark.asyncio
async def test_r6_4b_no_floor_is_the_wall(monkeypatch):
    wall = 1_800_000_200_000
    monkeypatch.setattr(live, "_scope_wall_ms", lambda: wall)
    client, provider, stop = await _stop_scope(monkeypatch)
    assert stop["media_scope_started_ms"] == wall == _ready(client)["media_scope_started_ms"]


@pytest.mark.asyncio
async def test_r6_4b_a_same_process_reconnect_with_a_backwards_wall_step(monkeypatch):
    wall = {"ms": 1_800_000_300_000}
    monkeypatch.setattr(live, "_scope_wall_ms", lambda: wall["ms"])
    user = _tag("reconnect")
    _c1, p1, first = await _stop_scope(monkeypatch, user=user)
    wall["ms"] -= 120_000                       # the host clock stepped back
    _c2, p2, second = await _stop_scope(monkeypatch, user=user)
    assert _psid(p1) != _psid(p2)
    assert first["media_scope_started_ms"] == 1_800_000_300_000
    assert second["media_scope_started_ms"] == first["media_scope_started_ms"] + 1


@pytest.mark.asyncio
@pytest.mark.parametrize("floor", [True, "1800000000000", -5, 0, 2 ** 53])
async def test_r6_4b_malformed_floors_are_ignored_on_the_wire(monkeypatch, floor):
    wall = 1_800_000_400_000
    monkeypatch.setattr(live, "_scope_wall_ms", lambda: wall)
    client, provider, stop = await _stop_scope(monkeypatch, config_extra={"media_scope_floor_ms": floor})
    assert stop["media_scope_started_ms"] == wall


@pytest.mark.asyncio
async def test_r6_4b_a_valid_floor_is_honoured_without_the_feature(monkeypatch):
    wall = 1_800_000_500_000
    monkeypatch.setattr(live, "_scope_wall_ms", lambda: wall)
    client, provider, stop = await _stop_scope(
        monkeypatch, features=V03, config_extra={"media_scope_floor_ms": wall + 7},
    )
    assert stop["media_scope_started_ms"] == wall + 8
    assert "media_scope_started_ms" not in _ready(client)


# ══════════════════════════════════════════════════════════════════════
# R6-5 — a question that restates the relay-owned command (media-2, -2b)
# ══════════════════════════════════════════════════════════════════════

RESTATING = [
    ("آهنگ رو قطع", "قطعش کنم؟", "آره", "fa"),
    ("آهنگ رو قطع", "آهنگ رو قطع کنم؟", "کن لطفا", "fa"),
    ("stop the music", "Should I stop the music?", "yes please", "en"),
]


async def _command_then_answer(monkeypatch, command, question, answer, *, wait_s, grace_ms, tenant=None):
    tenant = tenant or Tenant(device=_old_song(), agents={"": {"think_s": 0.1}})

    async def steps(p, client):
        p.push(H.user_delta(command, 1000, 1600))
        p.push(H.out_text(question, 1700, 2300))
        p.push(H.out_audio("Q"))
        await asyncio.sleep(wait_s)
        p.push(H.user_delta(answer, 3000, 3300))
        p.push(H.out_text("چشم." if answer in ("آره", "کن لطفا", "نه") else "Okay.", 3400, 3600))
        p.push(H.out_audio("OK"))
        await asyncio.sleep(0.6)
        p.push(H.delegation("d-late", 3700))
        await asyncio.sleep(2.2)

    client, provider = await _call(
        monkeypatch, tenant, steps, grace_ms=grace_ms,
        frames=({"type": "now_playing", "title": "Fadat Sham"},),
    )
    return client, provider, tenant


def _results(provider, delegation_id):
    return [str(e.get("content") or "") for e in _thinking(provider) if e.get("delegation_id") == delegation_id]


@pytest.mark.asyncio
@pytest.mark.parametrize(("command", "question", "answer", "lang"), RESTATING)
async def test_r6_5_an_assent_after_the_fire_continues_the_command(monkeypatch, command, question, answer, lang):
    """media-2: the command was already fired; the assent to the agent's
    restating question is its continuation — no research run, no card, one
    stop, and the late delegation's function result names the verdict."""

    client, provider, tenant = await _command_then_answer(
        monkeypatch, command, question, answer, wait_s=0.9, grace_ms=300,
    )
    assert len(tenant.controls("stop")) == 1, tenant.bodies
    assert tenant.runs() == [], tenant.runs()
    assert _frames_for(client, "d-late") == []
    results = _results(provider, "d-late")
    stopped = live.relay_line("media_stopped", "fa")      # the command is Persian or English
    assert results and "already carried out" in results[0], results
    assert any(line in results[0] for line in (stopped, live.relay_line("media_stopped", "en"))), results


@pytest.mark.asyncio
@pytest.mark.parametrize(("command", "question", "answer", "lang"), RESTATING + [
    ("آهنگ رو قطع", "چشم.", "کن", "fa"),        # the statement-ack shape (B1 tail), armed
])
async def test_r6_5_an_answer_inside_the_grace_joins_the_armed_command(monkeypatch, command, question, answer, lang):
    """media-2b: the answer and the model's late delegation arrive while the
    stop is still ARMED.  The armed unit owns its command: the delegation of
    the continuation binds to it — one stop, no card, no run of the answer."""

    client, provider, tenant = await _command_then_answer(
        monkeypatch, command, question, answer, wait_s=0.3, grace_ms=1500,
    )
    assert len(tenant.controls("stop")) == 1, tenant.bodies
    assert tenant.runs() == [], tenant.runs()
    assert _frames_for(client, "d-late") == []
    results = _results(provider, "d-late")
    assert results and "already carried out" in results[0], results


@pytest.mark.asyncio
@pytest.mark.parametrize(("command", "question", "answer"), [
    ("آهنگ رو قطع", "قطعش کنم؟", "نه"),
    ("stop the music", "Should I stop the music?", "no, don't"),
])
async def test_r6_5_a_refusal_is_never_absorbed_and_learns_the_stop_ran(monkeypatch, command, question, answer):
    """A refusal after the relay already stopped the music is the caller's
    own request: it runs (never absorbed, never a second stop), and the agent
    is told truthfully that the stop already happened."""

    client, provider, tenant = await _command_then_answer(
        monkeypatch, command, question, answer, wait_s=0.9, grace_ms=300,
    )
    assert len(tenant.controls("stop")) == 1, tenant.bodies
    runs = tenant.runs()
    assert len(runs) == 1, tenant.bodies
    assert "already carried out" in runs[0]["message"] and command in runs[0]["message"], runs[0]["message"]
    assert "play it again" in runs[0]["message"]
    assert not any("belong to their playback command" in r for r in _results(provider, "d-late"))


@pytest.mark.asyncio
async def test_r6_5_control_a_consent_to_a_different_question_is_a_request(monkeypatch):
    """B1 kept: after a fired stop, 'Should I stop the search too?' + 'yes' is
    not the stop's continuation — it is never absorbed as the media command."""

    tenant = Tenant(device=_old_song(), agents={"": {"think_s": 0.1}})
    client, provider, tenant = await _command_then_answer(
        monkeypatch, "stop the music", "Okay. Should I stop the search too?", "yes",
        wait_s=0.9, grace_ms=300, tenant=tenant,
    )
    assert len(tenant.controls("stop")) == 1, tenant.bodies
    assert not any("playback command" in r for r in _results(provider, "d-late"))


@pytest.mark.asyncio
@pytest.mark.parametrize(("question", "answer", "subject_word"), [
    ("می‌خوای یه پادکست برات پخش کنم؟", "آره", "پادکست"),
    ("Do you want me to find a podcast instead?", "yes please", "podcast"),
])
async def test_r6_5_a_consent_card_is_titled_from_its_proposal(monkeypatch, question, answer, subject_word):
    tenant = Tenant(agents={"": {"think_s": 0.1}})

    async def steps(p, client):
        p.push(H.user_delta("I'm bored", 1000, 1500))
        p.push(H.out_text(question, 1600, 2400))
        p.push(H.out_audio("Q"))
        await asyncio.sleep(0.6)
        p.push(H.user_delta(answer, 3000, 3300))
        p.push(H.delegation("d-consent", 3350))
        await _wait(lambda: tenant.runs(), 4.0)
        await asyncio.sleep(0.5)

    client, provider = await _call(monkeypatch, tenant, steps)
    created = [f for f in _frames_for(client, "d-consent") if f.get("phase") == "created"]
    assert created, client.of("delegation")
    title = created[0]["title"]
    assert subject_word in title and title.strip() != answer, title
    (run,) = tenant.runs()
    assert subject_word in run["display_request"] and run["display_request"].strip() != answer
    assert question.rstrip("?؟") in run["message"] or question in run["message"], run["message"]


# ══════════════════════════════════════════════════════════════════════
# R6-6 — never the caller's sentence as a track title
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
@pytest.mark.parametrize(("args", "stop", "ack", "expected"), [
    ({}, "stop the music", "Okay.", "OK — I won't start that."),
    ({}, "آهنگ رو قطع کن", "چشم.", "باشه، پخشش نمی‌کنم."),
    (None, "stop the music", "Okay.", "OK — I won't start «Halo Beyonce»."),
])
async def test_r6_6_the_pending_agent_play_line(monkeypatch, args, stop, ack, expected):
    """An agent play still searching when the acknowledged stop fires: the
    line names the track the tool asked for (`_vs_args` → {'query': …}), or
    — with no title at all — the neutral line; never the caller's request."""

    tenant = Tenant(agents={"halo": {"think_s": 0.1, "play": "Halo Beyonce", "search_s": 2.5, "args": args}})

    async def steps(p, client):
        p.push(H.user_delta(HALO_AGENT, 1000, 1600))
        p.push(H.delegation("dHalo", 1620))
        await _wait(lambda: "agent_run" in tenant.kinds(), 3.0)
        await asyncio.sleep(0.4)
        p.push(H.user_delta(stop, 3000, 3600))
        p.push(H.out_text(ack, 3700, 3900))
        p.push(H.out_audio("OK"))
        await asyncio.sleep(3.0)

    client, provider = await _call(monkeypatch, tenant, steps, grace_ms=300)
    said = _said(provider)
    assert ("play_superseded", "Halo Beyonce") in tenant.events, tenant.events
    assert any(expected in s for s in said), said
    assert not any(HALO_AGENT.lower() in s.lower() for s in said), said


# ══════════════════════════════════════════════════════════════════════
# §7 — the `newer_playing` verdict worded honestly (en + fa)
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
@pytest.mark.parametrize("paused", [False, True])
@pytest.mark.parametrize(("stop", "ack", "lang"), [
    ("stop the music", "", "en"),               # delegated
    ("آهنگ رو قطع کن", "", "fa"),                # delegated
    ("stop the music", "Okay.", "en"),          # backstop
    ("آهنگ رو قطع کن", "چشم.", "fa"),            # backstop
])
async def test_newer_playing_is_worded_as_the_newer_request_playing(monkeypatch, stop, ack, lang, paused):
    """A LATE stop of an older scope reaches a tenant whose broadcast item is
    from a NEWER scope: the tenant leaves it alone (`newer_playing`), and the
    relay says the newer request is what plays (or is paused) — never
    "stopped", never the generic failure."""

    tenant = Tenant()

    async def steps(p, client):
        stamp = _stamp(client, None)
        tenant.first_seen.setdefault("live-psid-newer-scope", 99)
        tenant.device = {
            "title": "Jazz Mix", "video_id": "JAZZMIX0001",
            "key": ("live-psid-newer-scope", stamp + 5000, 1), "paused": paused,
        }
        p.push(H.user_delta(stop, 1000, 1600))
        if ack:
            p.push(H.out_text(ack, 1700, 1900))
            p.push(H.out_audio("OK"))
        else:
            p.push(H.delegation("dStop", 1620))
        await _wait(lambda: tenant.controls("stop"), 4.0)
        await asyncio.sleep(1.2)

    client, provider = await _call(
        monkeypatch, tenant, steps, grace_ms=300,
        frames=({"type": "now_playing", "title": "Jazz Mix"},),
    )
    assert ("newer_playing", "Jazz Mix") in tenant.events, tenant.events
    assert tenant.device is not None and tenant.device["title"] == "Jazz Mix"
    key = "media_newer_paused" if paused else "media_newer_playing"
    expected = live.relay_line(key, lang, title="Jazz Mix")
    said = _said(provider)
    assert any(expected in s for s in said), said
    for s in said:
        assert live.relay_line("media_stopped", lang) not in s
        assert live.relay_line("media_control_failed", lang) not in s
    controls = client.of("media_control")
    assert controls and not any(f.get("status") == "executed" for f in controls), controls
    if not ack:
        assert "completed" in _phases(client, "dStop"), client.of("delegation")
        assert not any(e.get("code") == "delegation_failed" for e in client.of("error"))
