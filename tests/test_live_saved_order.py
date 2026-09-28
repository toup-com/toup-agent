"""The saved call reads the way the live call read, and nothing the tenant could
have taken is lost or mis-reported (R2 fix round, F31-F35 + addendum 9).

The phone lays a call out CAUSALLY (liveProtocol.placeCausal): a user turn is
appended when it appears, and an assistant reply sits under the turn it
answers, after that turn's earlier replies, however late its row is written.
The day chat sorts by `occurred_at` alone. These tests drive the real relay
with provider-shaped events and compare the saved `occurred_at` order with the
live order the relay's own frames produce (`live_order` below mirrors the app
reducer's placement rule). Stamps are also required to be distinct at
MILLISECOND precision, because that is what the app's day-chat sort compares.
"""

import asyncio
import logging
import time
import types
from datetime import datetime, timezone

import httpx
import pytest

import test_live_harness as H


# What TF132 negotiates plus `heard_text` (the shipped build advertises it).
TF132 = [
    "live_turns", "state_seq", "outcomes", "delegation_frames",
    "task_lifecycle", "playback_frames", "turn_timing", "media_control",
    "heard_text",
]
# Exactly what the new app advertises (liveProtocol LIVE_FEATURES).
APP = [
    "live_turns", "delegation_frames", "task_lifecycle", "playback_frames",
    "turn_timing", "media_control", "heard_text", "reask_turns",
    "language_pref", "media_transport",
]


# ── the two orders ────────────────────────────────────────────────────

def saved_rows(saves):
    """ref → {"at": final occurred_at, "voice": final voice}. The tenant keeps
    `occurred_at` when a revision omits it, and replaces it when one sends it."""

    rows: dict = {}
    for save in saves:
        for side in ("user", "assistant"):
            ref = save.get(f"{side}_ref")
            if not ref:
                continue
            if not (save.get(f"{side}_text") or save.get("media") or save.get("tool_events")):
                continue
            row = rows.setdefault(ref, {"at": None, "voice": {}})
            at = save.get(f"{side}_occurred_at")
            if at is not None:
                row["at"] = at
            row["voice"] = save.get(f"{side}_voice") or row["voice"]
    return rows


def saved_order(rows):
    assert all(r["at"] is not None for r in rows.values()), rows
    return [ref for ref, _ in sorted(rows.items(), key=lambda kv: (kv[1]["at"], kv[0]))]


def identity(ref):
    """A saved row's live identity: user turn id, or the epoch's output id."""

    kind, psid, tail = ref.split(":", 2)
    if kind in ("live-output", "live-transcript"):
        return f"live:{psid}:{tail}"
    return ref


def live_order(frames):
    """The app's call transcript order for these relay frames.

    User bubbles append on their first `transcript` frame; an assistant entry
    goes right after its parent and after that parent's lower-epoch replies,
    waiting for a parent that has not appeared yet. A PARENTLESS entry goes to
    the head (the app's rule for a greeting). `assert_saved_matches_live`
    compares only user turns and PARENTED replies: a mid-call reply the relay
    could not attribute is shown at the head live, and that is not a placement
    the saved thread should copy (a test below pins where it IS saved).
    """

    entries: list[dict] = []
    pending: list[dict] = []

    def user_at(turn_id):
        return next((i for i, e in enumerate(entries)
                     if e["role"] == "user" and e["id"] == turn_id), -1)

    def place(e):
        at = user_at(e["parent"]) + 1
        while at < len(entries):
            sib = entries[at]
            if sib["role"] != "assistant" or sib["parent"] != e["parent"] or sib["epoch"] > e["epoch"]:
                break
            at += 1
        entries.insert(at, e)

    for f in frames:
        kind = f.get("type")
        if kind == "transcript" and f.get("turn_id"):
            if user_at(f["turn_id"]) < 0:
                entries.append({"id": f["turn_id"], "role": "user"})
                for e in sorted([p for p in pending if p["parent"] == f["turn_id"]],
                                key=lambda p: p["epoch"]):
                    pending.remove(e)
                    place(e)
        elif kind == "response_text" and f.get("assistant_turn_id"):
            aid = f["assistant_turn_id"]
            if any(e["id"] == aid for e in entries + pending):
                continue
            e = {"id": aid, "role": "assistant",
                 "parent": f.get("parent_user_turn_id") or "", "epoch": int(f.get("epoch") or 0)}
            if not e["parent"]:
                at = 0
                while (at < len(entries) and entries[at]["role"] == "assistant"
                       and not entries[at]["parent"] and entries[at]["epoch"] <= e["epoch"]):
                    at += 1
                entries.insert(at, e)
            elif user_at(e["parent"]) >= 0:
                place(e)
            else:
                pending.append(e)
    return entries


def assert_saved_matches_live(recorded, client):
    rows = saved_rows(recorded["saves"])
    order = saved_order(rows)
    saved_ids = []
    for ref in order:
        ident = identity(ref)
        if ident not in saved_ids:
            saved_ids.append(ident)
    live = [e["id"] for e in live_order(client.frames) if e["role"] == "user" or e["parent"]]
    common = [x for x in live if x in saved_ids]
    assert len(common) >= 3, (common, live, saved_ids)
    assert [x for x in saved_ids if x in common] == common, (
        f"saved order {[x for x in saved_ids if x in common]} != live order {common}"
    )
    # The app sorts the saved thread at MILLISECOND precision (then by a random
    # id): two rows in one millisecond may render in either order.
    ms = [int(rows[ref]["at"].timestamp() * 1000) for ref in order]
    assert ms == sorted(ms) and len(set(ms)) == len(ms), list(zip(order, ms))
    return order, rows


def _provider(on_send=None, **kw):
    def default(p, e):
        if e["type"] == "session.close":
            p.push(H.closed())
        elif on_send is not None:
            on_send(p, e)
    return H.FakeProvider(on_send=default, **kw)


def _idle(epoch, heard, played=600):
    oid = f"live:{H.PSID}:{epoch}"
    return {"type": "playback_idle", "response_id": oid, "item_id": oid,
            "played_ms": played, "heard_text": heard}


def _interrupt(epoch, heard, played=900):
    oid = f"live:{H.PSID}:{epoch}"
    return {"type": "interrupt", "response_id": oid, "item_id": oid,
            "played_ms": played, "heard_text": heard}


# ── F31: a barge-in turn is saved BELOW the answer it interrupted ─────

@pytest.mark.asyncio
async def test_a_transcript_first_barge_in_is_saved_below_the_answer_it_interrupted(monkeypatch):
    """Every relay-synthesized barge-in is transcript-first: the caller's words
    over the reply reach the relay (and settle, and are written) before the
    phone's `interrupt` writes the reply, which the global clamp then saved
    BELOW them — for good, once P5 froze the first stamp."""

    H.fast_clocks(
        monkeypatch, voice_live_utterance_gap_ms=250,
        voice_live_utterance_hard_gap_ms=500, voice_live_output_epoch_gap_ms=3000,
        voice_live_bargein_min_ms=300, voice_live_bargein_min_tokens=3,
    )
    recorded = H.patch_relay(monkeypatch)
    provider = _provider()
    client = H.FakeClient([
        H.config(features=TF132), 0.1,
        lambda: provider.push(H.user_delta("what are the best ramen places downtown", 0, 1500)),
        0.4,
        lambda: (provider.push(H.out_text("Here is a long list of ramen places. First, ", 2000, 9000)),
                 provider.push(H.out_audio())),
        0.3,
        lambda: provider.push(H.user_delta("no wait I meant sushi near me", 4000, 5200)),
        0.3, _interrupt(1, "Here is a long list"), 0.4, {"type": "stop"},
    ])
    await H.run_relay(client, provider, timeout=10)
    order, _ = assert_saved_matches_live(recorded, client)
    assert order.index(f"live-output:{H.PSID}:1") < order.index(f"live-utt:{H.PSID}:2"), order


@pytest.mark.asyncio
async def test_a_two_fragment_barge_in_is_saved_below_the_answer(monkeypatch):
    H.fast_clocks(
        monkeypatch, voice_live_utterance_gap_ms=250,
        voice_live_utterance_hard_gap_ms=500, voice_live_output_epoch_gap_ms=3000,
        voice_live_bargein_min_ms=300, voice_live_bargein_min_tokens=3,
    )
    recorded = H.patch_relay(monkeypatch)
    provider = _provider()
    client = H.FakeClient([
        H.config(features=TF132), 0.1,
        lambda: provider.push(H.user_delta("what are the best ramen places downtown", 0, 1500)),
        0.4,
        lambda: (provider.push(H.out_text("Here is a long list of ramen places. First, ", 2000, 9000)),
                 provider.push(H.out_audio())),
        0.3, lambda: provider.push(H.user_delta("no wait", 4000, 4300)),
        0.15, lambda: provider.push(H.user_delta("I meant sushi near me", 4300, 5200)),
        0.05, _interrupt(1, "Here is a long list"), 0.6, {"type": "stop"},
    ])
    await H.run_relay(client, provider, timeout=10)
    assert_saved_matches_live(recorded, client)


@pytest.mark.asyncio
async def test_an_interrupt_that_lands_after_the_next_turn_settled_keeps_the_answer_above(monkeypatch):
    H.fast_clocks(
        monkeypatch, voice_live_transcript_settle_ms=100,
        voice_live_utterance_gap_ms=400, voice_live_utterance_hard_gap_ms=800,
        voice_live_output_epoch_gap_ms=3000,
        voice_live_bargein_min_ms=600, voice_live_bargein_min_tokens=3,
    )
    recorded = H.patch_relay(monkeypatch)
    provider = _provider()
    client = H.FakeClient([
        H.config(features=TF132), 0.1,
        lambda: provider.push(H.user_delta("what are the best ramen places downtown", 0, 1500)),
        0.6,
        lambda: (provider.push(H.out_text("Here is a long list of ramen places. First, ", 2000, 9000)),
                 provider.push(H.out_audio())),
        0.3, lambda: provider.push(H.user_delta("no wait I ", 4000, 4400)),
        0.05, lambda: provider.push(H.user_delta("meant sushi near me", 4400, 5200)),
        0.2, _interrupt(1, "Here is a long list"), 0.9, {"type": "stop"},
    ])
    await H.run_relay(client, provider, timeout=10)
    assert_saved_matches_live(recorded, client)


@pytest.mark.asyncio
async def test_a_remark_during_a_reply_is_saved_below_that_reply(monkeypatch):
    """No barge-in at all: "mm okay" while the reply plays to the end."""

    H.fast_clocks(monkeypatch, voice_live_utterance_gap_ms=200,
                  voice_live_output_epoch_gap_ms=3000)
    recorded = H.patch_relay(monkeypatch)
    provider = _provider()
    client = H.FakeClient([
        H.config(features=TF132), 0.1,
        lambda: provider.push(H.user_delta("is it going to rain tomorrow", 0, 1200)),
        0.4,
        lambda: (provider.push(H.out_text("Yes, light rain after noon, then clearing.", 1500, 5500)),
                 provider.push(H.out_audio())),
        0.2, lambda: provider.push(H.user_delta("mm okay", 3000, 3300)),
        0.5, _idle(1, "Yes, light rain after noon, then clearing.", 4000),
        0.5, {"type": "stop"},
    ])
    await H.run_relay(client, provider, timeout=10)
    assert_saved_matches_live(recorded, client)


@pytest.mark.asyncio
async def test_a_parentless_reply_barged_into_is_saved_where_it_was_seen(monkeypatch):
    """V1 03:37 shape with no earlier user turn: the reply has no parent, so
    only the moment the phone saw it can place it. Written at the interrupt,
    after the barge-in row, it used to be clamped below that row for good."""

    H.fast_clocks(monkeypatch, voice_live_utterance_gap_ms=5000,
                  voice_live_output_epoch_gap_ms=3000)
    recorded = H.patch_relay(monkeypatch)
    provider = _provider()
    client = H.FakeClient([
        H.config(), 0.1,
        lambda: (provider.push(H.out_text("Here is a long list of professors ", 100, 900)),
                 provider.push(H.out_audio())),
        0.2, lambda: provider.push(H.user_delta("is their ", 1000, 1300)),
        0.1, {"type": "interrupt", "response_id": f"live:{H.PSID}:1",
              "item_id": f"live:{H.PSID}:1", "played_ms": 700},
        0.2, lambda: provider.push(H.user_delta("work on LLMs", 1400, 1800)),
        0.4, {"type": "stop"},
    ])
    await H.run_relay(client, provider, timeout=8)
    rows = saved_rows(recorded["saves"])
    order = saved_order(rows)
    live = [e["id"] for e in live_order(client.frames)]
    assert live.index(f"live:{H.PSID}:1") < live.index(f"live-utt:{H.PSID}:1"), live
    assert order.index(f"live-output:{H.PSID}:1") < order.index(f"live-utt:{H.PSID}:1"), order


# ── F32: parked replies and delegated results sit under their request ─

@pytest.mark.asyncio
async def test_a_parked_reply_whose_receipt_lands_after_the_next_turn_stays_above_it(monkeypatch):
    """heard_text parks a gap-retired reply for its receipt; the caller answers
    over the buffered tail before the receipt arrives."""

    H.fast_clocks(monkeypatch, voice_live_utterance_gap_ms=200,
                  voice_live_output_epoch_gap_ms=80)
    recorded = H.patch_relay(monkeypatch)
    provider = _provider()
    client = H.FakeClient([
        H.config(features=APP), 0.1,
        lambda: provider.push(H.user_delta("is it going to rain tomorrow", 0, 1200)),
        0.5,
        lambda: (provider.push(H.out_text("Yes, light rain after noon. Anything else?", 1500, 4500)),
                 provider.push(H.out_audio())),
        0.4, lambda: provider.push(H.user_delta("ok and on friday", 5000, 6000)),
        0.7, _idle(1, "Yes, light rain after noon. Anything else?", 3000),
        0.4, {"type": "stop"},
    ])
    await H.run_relay(client, provider, timeout=10)
    order, _ = assert_saved_matches_live(recorded, client)
    assert order.index(f"live-output:{H.PSID}:1") < order.index(f"live-utt:{H.PSID}:2"), order


@pytest.mark.asyncio
async def test_a_parked_reply_cut_by_an_interrupt_after_the_next_turn_stays_above_it(monkeypatch):
    H.fast_clocks(monkeypatch, voice_live_utterance_gap_ms=250,
                  voice_live_utterance_hard_gap_ms=500, voice_live_output_epoch_gap_ms=80,
                  voice_live_bargein_min_ms=300, voice_live_bargein_min_tokens=3)
    recorded = H.patch_relay(monkeypatch)
    provider = _provider()
    client = H.FakeClient([
        H.config(features=TF132), 0.1,
        lambda: provider.push(H.user_delta("what are the best ramen places downtown", 0, 1500)),
        0.4,
        lambda: (provider.push(H.out_text(
            "Here is a long list of ramen places. First, Kinton on Baldwin.", 2000, 9000)),
            provider.push(H.out_audio())),
        0.4, lambda: provider.push(H.user_delta("no wait I meant sushi near me", 4000, 5200)),
        0.3, _interrupt(1, "Here is a long list", 1500), 0.5, {"type": "stop"},
    ])
    await H.run_relay(client, provider, timeout=10)
    assert_saved_matches_live(recorded, client)


async def _delegated_with_small_talk(monkeypatch, *, think_s=1.6):
    """U1 asks for research; the model says a holding line; while the agent
    works the caller says something else (U2) and gets a direct answer (A2);
    then the result is paraphrased — later on the provider timeline than all
    of it, and parented on U1."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=3.0,
                  voice_live_utterance_gap_ms=200, voice_live_output_epoch_gap_ms=150)

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        await asyncio.sleep(think_s)
        return "Robarts Library closes at 11 pm tonight.", "m"

    recorded = H.patch_relay(monkeypatch, think=think)

    def on_send(p, e):
        if e["type"] == "session.commentary.append":
            p.push(H.out_text("Robarts closes at eleven tonight.", 9000, 10500))
            p.push(H.out_audio("RESULT"))

    provider = _provider(on_send, auto_ack=True)

    def u1():
        provider.push(H.user_delta("when does robarts library close tonight", 0, 1800))
        provider.push(H.delegation("d1", 1850))

    client = H.FakeClient([
        H.config(features=APP), 0.1, u1, 0.35,
        lambda: (provider.push(H.out_text("Let me check that for you.", 2000, 2600)),
                 provider.push(H.out_audio())),
        0.25, _idle(1, "Let me check that for you."), 0.15,
        lambda: provider.push(H.user_delta("I need to return two books", 4000, 5200)),
        0.4,
        lambda: (provider.push(H.out_text("Okay, I will keep that in mind.", 5600, 6400)),
                 provider.push(H.out_audio())),
        0.25, _idle(2, "Okay, I will keep that in mind."), think_s,
        _idle(3, "Robarts closes at eleven tonight."), 0.5, {"type": "stop"},
    ])
    await H.run_relay(client, provider, timeout=12)
    return recorded, client


@pytest.mark.asyncio
async def test_a_delegated_result_is_saved_under_its_request_not_below_later_turns(monkeypatch):
    recorded, client = await _delegated_with_small_talk(monkeypatch)
    order, rows = assert_saved_matches_live(recorded, client)
    result = f"live-delegation:{H.PSID}:d1"
    u1, u2 = f"live-utt:{H.PSID}:1", f"live-utt:{H.PSID}:2"
    assert rows[result]["voice"].get("parent_user_turn_id") == u1
    assert order.index(u1) < order.index(result) < order.index(u2), order
    # The paraphrase the caller heard follows its result, still above U2 and
    # the reply U2 got.
    heard = [r for r in order if r.startswith("live-transcript:")]
    assert heard, order
    assert order.index(result) < order.index(heard[0]) < order.index(u2), order
    # The holding line the caller heard first stays above the result.
    hold = f"live-output:{H.PSID}:1"
    assert order.index(hold) < order.index(result), order


@pytest.mark.asyncio
async def test_rows_fit_between_turns_at_millisecond_precision_despite_timeline_skew(monkeypatch):
    """Production: output positions run seconds ahead of the input timeline,
    so the next user row is clamped right below the reply. A late result for
    the EARLIER turn must still get its own millisecond between them."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=3.0,
                  voice_live_utterance_gap_ms=200, voice_live_output_epoch_gap_ms=150)

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        await asyncio.sleep(1.2)
        return "Robarts Library closes at 11 pm tonight.", "m"

    recorded = H.patch_relay(monkeypatch, think=think)

    def on_send(p, e):
        if e["type"] == "session.commentary.append":
            p.push(H.out_text("Robarts closes at eleven tonight.", 40000, 41000))
            p.push(H.out_audio("RESULT"))

    provider = _provider(on_send, auto_ack=True)

    def u1():
        provider.push(H.user_delta("when does robarts library close tonight", 0, 1800))
        provider.push(H.delegation("d1", 1850))

    client = H.FakeClient([
        H.config(features=APP), 0.1, u1, 0.35,
        # 28 s ahead of the caller's input timeline (RUNTIME_EVIDENCE §1).
        lambda: (provider.push(H.out_text("Let me check that for you.", 30000, 30600)),
                 provider.push(H.out_audio())),
        0.25, _idle(1, "Let me check that for you."), 0.15,
        lambda: provider.push(H.user_delta("I need to return two books", 4000, 5200)),
        1.6, _idle(2, "Robarts closes at eleven tonight."), 0.5, {"type": "stop"},
    ])
    await H.run_relay(client, provider, timeout=12)
    order, rows = assert_saved_matches_live(recorded, client)
    result = f"live-delegation:{H.PSID}:d1"
    assert order.index(f"live-utt:{H.PSID}:1") < order.index(result) < order.index(
        f"live-utt:{H.PSID}:2"), order


@pytest.mark.asyncio
async def test_a_media_card_that_starts_after_the_next_turn_is_saved_under_its_request(monkeypatch):
    """The fast path's card is the delegated result of a "play" request, and a
    slow tenant play can finish after the caller has said something else."""

    import app.services.live_voice_protocol as live

    H.fast_clocks(monkeypatch, voice_live_utterance_gap_ms=200)

    class _Ask:
        query = "beyonce halo"
        variety = False
        mode = None

    monkeypatch.setattr(live, "_media_request", lambda text: _Ask() if "halo" in text else None)

    async def play(_user_id, query, variety=False):
        await asyncio.sleep(0.8)
        return "Now playing: Halo.", {"type": "youtube", "video_id": "v1", "title": "Halo"}

    recorded = H.patch_relay(monkeypatch, play=play)
    provider = _provider()

    def u1():
        provider.push(H.user_delta("play beyonce halo", 0, 900))
        provider.push(H.delegation("d1", 950))

    client = H.FakeClient([
        H.config(features=APP), 0.1, u1, 0.4,
        lambda: provider.push(H.user_delta("thanks that is all for now", 4000, 5200)),
        1.2, {"type": "stop"},
    ])
    await H.run_relay(client, provider, timeout=8)
    rows = saved_rows(recorded["saves"])
    order = saved_order(rows)
    card = f"live-media:{H.PSID}:d1"
    assert card in order, order
    assert order.index(f"live-utt:{H.PSID}:1") < order.index(card) < order.index(
        f"live-utt:{H.PSID}:2"), order


@pytest.mark.asyncio
async def test_a_request_that_is_still_the_newest_turn_keeps_its_old_stamps(monkeypatch):
    """No later turn: the delegated row is `utterance.end_ms + 1` after
    everything so far (unchanged), and every stamp stays in write order."""

    H.fast_clocks(monkeypatch, voice_live_utterance_gap_ms=400)

    async def think(*args, **kwargs):
        return "The answer.", "m"

    recorded = H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("question one", 0, 400))
            provider.push(H.user_delta("question two", 3000, 3400))
            provider.push(H.delegation("d1", 3500))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([H.config(), 1.0, {"type": "stop"}])
    await H.run_relay(client, H.FakeProvider(on_send=on_send), timeout=6)
    rows = saved_rows(recorded["saves"])
    u2 = rows[f"live-utt:{H.PSID}:2"]["at"]
    result = rows[f"live-delegation:{H.PSID}:d1"]["at"]
    assert (result - u2).total_seconds() == pytest.approx(0.401, abs=0.0005)
    stamps = [
        s.get("user_occurred_at") or s.get("assistant_occurred_at")
        for s in recorded["saves"]
        if s.get("user_occurred_at") or s.get("assistant_occurred_at")
    ]
    assert stamps == sorted(stamps)


# ── F33: an outage the tenant recovers from loses nothing ─────────────

class _Resp:
    def __init__(self, status, body=b'{"id": "m1"}'):
        self.status_code = status
        self.content = body

    def json(self):
        import json
        return json.loads(self.content)


def _outage(monkeypatch, *, down_s, mode):
    """A tenant that is away for `down_s` of WALL time, then back: refusing
    connections, answering 503 from the edge, or with no address at all."""

    from app.api import ws_realtime as rt
    import app.services.agent_http as ah

    t0 = time.monotonic()
    posts: list[tuple[float, str]] = []

    async def vps(_uid):
        if mode == "vps_none" and time.monotonic() - t0 < down_s:
            posts.append((round(time.monotonic() - t0, 3), "vps_none"))
            return None
        return ("http://agent.test", "k")

    class _Client:
        async def post(self, url, **kw):
            now = time.monotonic() - t0
            down = now < down_s
            posts.append((round(now, 3), "down" if down else "up"))
            if down and mode == "connect":
                raise httpx.ConnectError("[Errno 61] Connection refused")
            if down and mode == "503":
                return _Resp(503, b"")
            if down and mode == "500":
                return _Resp(500, b"")
            return _Resp(201)

    monkeypatch.setattr(rt, "_get_vps_info", vps)
    monkeypatch.setattr(ah, "get_agent_http_client", lambda: _Client())
    return posts


def _row(i, *, key=None):
    import app.services.live_voice_protocol as live

    key = key or f"live-utt:probe:{i}"
    return live.PersistenceRecord(kind="message", key=key, payload={
        "user_text": f"sentence {i}", "assistant_text": "", "user_ref": key,
        "user_revision": 1,
    })


def _scaled_ladder(monkeypatch):
    """The production ladder at a tenth of the time, so a test is fast."""

    import app.services.live_voice_protocol as live

    monkeypatch.setattr(live.PersistenceQueue, "BACKOFF_S", (0.05, 0.1, 0.2, 0.4, 0.8))


@pytest.mark.asyncio
async def test_a_two_second_tenant_restart_does_not_lose_the_row(monkeypatch):
    """F33, the finding's own numbers with the REAL ladder: refused for 2 s,
    back at 2 s. The old ladder tried at 0, 0.25 and 1.25 s and gave up."""

    import app.services.live_voice_protocol as live

    counters = H.capture_counters(monkeypatch)
    posts = _outage(monkeypatch, down_s=2.0, mode="connect")
    queue = live.PersistenceQueue("user-1", "db-session")
    queue.start()
    queue.submit(_row(0))
    assert await queue.drain(10.0)
    await queue.stop()
    assert queue.written == 1 and queue.lost == 0, posts
    assert posts[-1][1] == "up" and posts[-1][0] >= 2.0, posts
    names = [n for n, _ in counters]
    assert "voice_transcript_lost" not in names and "live_persist_lost" not in names


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["connect", "503", "vps_none"])
async def test_an_outage_longer_than_the_short_ladder_is_retried_until_the_tenant_is_back(
    monkeypatch, mode,
):
    import app.services.live_voice_protocol as live

    _scaled_ladder(monkeypatch)
    posts = _outage(monkeypatch, down_s=0.4, mode=mode)
    queue = live.PersistenceQueue("user-1", "db-session")
    queue.retry_window_s = 3.0
    queue.start()
    queue.submit(_row(0))
    assert await queue.drain(5.0)
    await queue.stop()
    assert queue.written == 1 and queue.lost == 0, posts


@pytest.mark.asyncio
async def test_an_outage_that_never_ends_is_lost_once_after_the_window(monkeypatch, caplog):
    import app.services.live_voice_protocol as live

    _scaled_ladder(monkeypatch)
    counters = H.capture_counters(monkeypatch)
    caplog.set_level(logging.INFO)
    posts = _outage(monkeypatch, down_s=3600.0, mode="connect")
    queue = live.PersistenceQueue("user-1", "db-session")
    queue.retry_window_s = 0.6
    queue.start()
    queue.submit(_row(0))
    assert await queue.drain(5.0)
    await queue.stop()
    assert queue.written == 0 and queue.lost == 1
    # Bounded by the window, not by a count: the last attempt is at its end.
    assert len(posts) > 3 and 0.5 <= posts[-1][0] <= 1.2, posts
    names = [n for n, _ in counters]
    assert names.count("live_persist_retry") == len(posts) - 1, names
    assert names.count("voice_transcript_lost") == 1 and names.count("live_persist_lost") == 1
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert len([m for m in errors if "LOST" in m]) == 1, errors


@pytest.mark.asyncio
async def test_a_row_specific_tenant_500_still_gets_three_attempts(monkeypatch):
    """Not an outage: a poison row must not hold every later row of the call
    behind it for the whole window."""

    import app.services.live_voice_protocol as live

    monkeypatch.setattr(live.PersistenceQueue, "BACKOFF_S", (0.0,))
    posts = _outage(monkeypatch, down_s=3600.0, mode="500")
    queue = live.PersistenceQueue("user-1", "db-session")
    queue.start()
    queue.submit(_row(0))
    assert await queue.drain(3.0)
    await queue.stop()
    assert len(posts) == 3 and queue.lost == 1


@pytest.mark.asyncio
async def test_the_finisher_writes_a_backlog_through_a_restart_longer_than_the_live_window(
    monkeypatch,
):
    """Hang-up during a tenant restart: the finisher's budget, not the live
    window, bounds the retry — the backlog lands when the tenant is back."""

    import app.services.live_voice_protocol as live

    _scaled_ladder(monkeypatch)
    posts = _outage(monkeypatch, down_s=1.0, mode="connect")
    queue = live.PersistenceQueue("user-1", "db-session")
    queue.retry_window_s = 0.3          # the live window: shorter than the outage
    queue.start()
    for i in range(3):
        queue.submit(_row(i))
    await live._finish_detached(queue, [], 0.1, 5.0)
    assert queue.written == 3 and queue.lost == 0, posts


def test_the_retry_window_is_a_real_setting():
    from app.config import settings

    assert float(settings.voice_live_persist_retry_window_s) >= 30.0


# ── F34: one verdict per ROW, and a row keeps its place ───────────────

def _upsert_tenant(monkeypatch, statuses, *, latency=0.02):
    """Scripted statuses per POST; a 2xx upserts by client_msg_id and sets
    occurred_at only when the body carries it (sessions.py)."""

    from app.api import ws_realtime as rt
    import app.services.agent_http as ah

    posts: list[dict] = []
    rows: dict[str, dict] = {}
    script = list(statuses)

    async def vps(_uid):
        return ("http://agent.test", "k")

    class _Client:
        async def post(self, url, **kw):
            body = dict(kw.get("json") or {})
            await asyncio.sleep(latency)
            status = script.pop(0) if script else 201
            posts.append({"status": status, **body})
            if 200 <= status < 300:
                row = rows.setdefault(body.get("client_msg_id"), {})
                row.update({k: v for k, v in body.items() if k != "occurred_at"})
                if "occurred_at" in body:
                    row["occurred_at"] = body["occurred_at"]
            return _Resp(status)

    monkeypatch.setattr(rt, "_get_vps_info", vps)
    monkeypatch.setattr(ah, "get_agent_http_client", lambda: _Client())
    return posts, rows


def _short_window_queue(monkeypatch):
    """Zero backoff and no outage window: a failing revision is final after
    the three short-ladder attempts, which is the shape F34 is about."""

    import app.services.live_voice_protocol as live

    monkeypatch.setattr(live.PersistenceQueue, "BACKOFF_S", (0.0,))
    queue = live.PersistenceQueue("user-1", "db-session")
    queue.retry_window_s = 0.0
    queue.start()
    return queue


def _user(key, rev, text):
    import app.services.live_voice_protocol as live

    return live.PersistenceRecord(kind="message", key=key, payload={
        "user_text": text, "assistant_text": "", "user_ref": key,
        "user_occurred_at": datetime(2026, 9, 23, 3, 0, 0, tzinfo=timezone.utc),
        "user_revision": rev,
    })


@pytest.mark.asyncio
async def test_a_row_that_lands_on_its_newer_revision_is_not_reported_lost(monkeypatch, caplog):
    counters = H.capture_counters(monkeypatch)
    caplog.set_level(logging.INFO)
    posts, rows = _upsert_tenant(monkeypatch, [503, 503, 503, 201])
    queue = _short_window_queue(monkeypatch)
    queue.submit(_user("live-utt:p:7", 1, "book a table"))
    await asyncio.sleep(0.01)                     # revision 1 is on the wire
    queue.submit(_user("live-utt:p:7", 2, "book a table for two at eight"))
    assert await queue.drain(3.0)
    await queue.stop()

    (row,) = rows.values()
    assert row["content"] == "book a table for two at eight" and row["revision"] == 2
    assert queue.lost == 0 and queue.written == 1
    names = [n for n, _ in counters]
    assert "voice_transcript_lost" not in names and "live_persist_lost" not in names, names
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR and "LOST" in r.getMessage()]


@pytest.mark.asyncio
async def test_a_row_whose_every_revision_fails_is_one_lost_row(monkeypatch):
    counters = H.capture_counters(monkeypatch)
    posts, rows = _upsert_tenant(monkeypatch, [503] * 6)
    queue = _short_window_queue(monkeypatch)
    queue.submit(_user("live-utt:p:8", 1, "hello"))
    await asyncio.sleep(0.01)
    queue.submit(_user("live-utt:p:8", 2, "hello there"))
    assert await queue.drain(3.0)
    await queue.stop()
    assert rows == {}
    names = [n for n, _ in counters]
    assert names.count("voice_transcript_lost") == 1, names
    assert queue.lost == 1


def _delegated_first_write(ref, stamp):
    return {
        "user_text": "", "assistant_text": "Two tables free at eight.",
        "model": "m", "assistant_ref": ref, "assistant_occurred_at": stamp,
        "assistant_voice": {"source": "delegated", "delegation_id": "d1"},
    }


@pytest.mark.asyncio
async def test_a_spoken_flag_revision_that_creates_the_row_carries_its_place(monkeypatch):
    """`write_spoken_flag` omits `occurred_at` because the row normally exists.
    When the first write fails behind it, the flag revision CREATES the row —
    at the tenant's arrival time, unless it inherits the first write's stamp."""

    import app.services.live_voice_protocol as live

    posts, rows = _upsert_tenant(monkeypatch, [503, 503, 503, 201])
    queue = _short_window_queue(monkeypatch)
    ref = "live-delegation:p:d1"
    stamp = datetime(2026, 9, 23, 3, 0, 1, tzinfo=timezone.utc)
    first = _delegated_first_write(ref, stamp)
    queue.submit(live.PersistenceRecord(kind="message", key=ref, payload=first))
    await asyncio.sleep(0.01)                     # the first write is on the wire
    session = types.SimpleNamespace(persistence=queue)
    delivery = types.SimpleNamespace(claim={"ref": ref, "revision": 1, "payload": dict(first)})
    await live._LiveSession.write_spoken_flag(session, delivery, True)
    assert await queue.drain(3.0)
    await queue.stop()

    (row,) = rows.values()
    assert row["voice"]["spoken"] is True
    assert row.get("occurred_at") == stamp.isoformat(), row
    assert queue.lost == 0


@pytest.mark.asyncio
async def test_a_revision_submitted_after_the_row_was_lost_still_creates_it_in_place(monkeypatch):
    import app.services.live_voice_protocol as live

    counters = H.capture_counters(monkeypatch)
    posts, rows = _upsert_tenant(monkeypatch, [503, 503, 503, 201])
    queue = _short_window_queue(monkeypatch)
    ref = "live-delegation:p:d2"
    stamp = datetime(2026, 9, 23, 3, 0, 2, tzinfo=timezone.utc)
    first = _delegated_first_write(ref, stamp)
    queue.submit(live.PersistenceRecord(kind="message", key=ref, payload=first))
    assert await queue.drain(3.0)                 # the first write is lost…
    assert queue.lost == 1
    session = types.SimpleNamespace(persistence=queue)
    delivery = types.SimpleNamespace(claim={"ref": ref, "revision": 1, "payload": dict(first)})
    await live._LiveSession.write_spoken_flag(session, delivery, True)
    assert await queue.drain(3.0)                 # …and the flag revision creates it
    await queue.stop()

    (row,) = rows.values()
    assert row.get("occurred_at") == stamp.isoformat(), row
    assert "live_persist_recovered" in [n for n, _ in counters]


# ── F35: no relay-written row sends a null the tenant has to drop ─────

class _Dropped(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.lines: list[str] = []

    def emit(self, record):
        message = record.getMessage()
        if "voice provenance dropped" in message:
            self.lines.append(message)


def _through_tenant_allowlist(saves):
    from app.api import sessions

    capture = _Dropped()
    sessions.logger.addHandler(capture)
    try:
        voices = []
        for save in saves:
            for side in ("user", "assistant"):
                voice = save.get(f"{side}_voice")
                if voice:
                    voices.append(dict(voice))
                    sessions._clean_voice(dict(voice))
    finally:
        sessions.logger.removeHandler(capture)
    return voices, capture.lines


REPLY = "Hi, how can I help?"


def _greeting_provider():
    return _provider(lambda p, e: (
        p.push(H.user_delta("hello there", 0, 400)) if e["type"] == "session.start" else None
    ))


def _speaks(provider):
    def push():
        provider.push(H.out_text(REPLY, 1500, 2300))
        provider.push(H.out_audio())
    return push


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ["v02_late_ack", "heard_text_deadline", "teardown", "legacy"])
async def test_every_relay_written_voice_passes_the_tenant_allowlist(monkeypatch, shape):
    import app.services.live_voice_protocol as live

    H.fast_clocks(monkeypatch, voice_live_output_epoch_gap_ms=80)
    if shape == "heard_text_deadline":
        monkeypatch.setattr(live, "_SPOKEN_SETTLE_MS", 150)
    recorded = H.patch_relay(monkeypatch)
    provider = _greeting_provider()
    oid = f"live:{H.PSID}:1"
    ack = {"type": "playback_idle", "response_id": oid, "item_id": oid, "played_ms": 800}
    script = {
        # v0.2: the gap writes the row before the ack; the ack revises it.
        "v02_late_ack": [H.config(), 0.3, _speaks(provider), 0.5, ack, 0.3],
        # heard_text: the receipt misses the settle window; the deadline writes.
        "heard_text_deadline": [H.config(features=TF132), 0.3, _speaks(provider), 0.8,
                                {**ack, "heard_text": REPLY}, 0.3],
        # heard_text: the caller hangs up before any receipt.
        "teardown": [H.config(features=TF132), 0.3, _speaks(provider), 0.3],
        # build 129 / web: never acks.
        "legacy": [H.legacy_config(), 0.3, _speaks(provider), 0.5],
    }[shape]
    client = H.FakeClient([*script, {"type": "stop"}])
    await H.run_relay(client, provider, timeout=8)

    spoken = [s for s in recorded["saves"]
              if (s.get("assistant_voice") or {}).get("source") == "live_spoken"]
    assert spoken, shape
    voices, dropped = _through_tenant_allowlist(recorded["saves"])
    assert not dropped, dropped
    for voice in voices:
        assert all(value is not None for value in voice.values()), voice
        assert "played_ms" not in voice or isinstance(voice["played_ms"], int), voice


@pytest.mark.asyncio
async def test_a_heard_paraphrase_row_sends_no_null_either(monkeypatch):
    """The `live_transcript` sibling of a delegated result, written when its
    epoch is gap-retired — before any ack."""

    H.fast_clocks(monkeypatch)

    async def think(*args, **kwargs):
        return "The library closes at eleven tonight, with late hours on Friday.", "m"

    recorded = H.patch_relay(monkeypatch, think=think)
    provider = _provider(lambda p, e: (
        (p.push(H.user_delta("when does it close", 0, 600)), p.push(H.delegation("d1", 620)))
        if e["type"] == "session.start" else None
    ), auto_ack=True, speak_on_commentary=True)
    client = H.FakeClient([H.config(features=TF132), 1.2, {"type": "stop"}])
    await H.run_relay(client, provider, timeout=8)

    heard = [s for s in recorded["saves"]
             if (s.get("assistant_voice") or {}).get("source") == "live_transcript"]
    assert heard, [s.get("assistant_voice") for s in recorded["saves"]]
    voices, dropped = _through_tenant_allowlist(recorded["saves"])
    assert not dropped, dropped
    assert all(v is not None for voice in voices for v in voice.values()), voices


@pytest.mark.asyncio
async def test_a_delegated_row_whose_model_is_unknown_sends_no_null(monkeypatch):
    """`_think` returns `model_override`, which can be None on its fallback
    paths; the delegated row copied it into `assistant_voice.model`."""

    H.fast_clocks(monkeypatch)

    async def think(*args, **kwargs):
        return "The library closes at eleven tonight.", None

    recorded = H.patch_relay(monkeypatch, think=think)
    provider = _provider(lambda p, e: (
        (p.push(H.user_delta("when does it close", 0, 600)), p.push(H.delegation("d1", 620)))
        if e["type"] == "session.start" else None
    ))
    client = H.FakeClient([H.config(features=TF132), 0.8, {"type": "stop"}])
    await H.run_relay(client, provider, timeout=8)

    delegated = [s for s in recorded["saves"]
                 if (s.get("assistant_voice") or {}).get("source") == "delegated"]
    assert delegated, [s.get("assistant_voice") for s in recorded["saves"]]
    voices, dropped = _through_tenant_allowlist(recorded["saves"])
    assert not dropped, dropped
    assert all(v is not None for voice in voices for v in voice.values()), voices


# ── addendum 9: one counts-only latency line per user turn ────────────

def _timeline_lines(caplog):
    return [r.getMessage() for r in caplog.records if "[LIVE] turn timeline" in r.getMessage()]


def _fields(line):
    return dict(part.split("=", 1) for part in line.split() if "=" in part)


def _values(line):
    """Only what the line SAYS — its field values — never its field names."""

    return " ".join(_fields(line).values()).lower()


@pytest.mark.asyncio
async def test_a_directly_answered_turn_logs_one_timeline_line_at_its_first_output(
    monkeypatch, caplog,
):
    caplog.set_level(logging.INFO)
    H.fast_clocks(monkeypatch, voice_live_utterance_gap_ms=200,
                  voice_live_output_epoch_gap_ms=150)
    H.patch_relay(monkeypatch)
    provider = _provider()
    client = H.FakeClient([
        H.config(features=APP), 0.1,
        lambda: provider.push(H.user_delta("what time is it in tokyo", 0, 1200)),
        0.5,
        lambda: (provider.push(H.out_text("It is nine in the morning there.", 1600, 2600)),
                 provider.push(H.out_audio())),
        0.4, {"type": "stop"},
    ])
    await H.run_relay(client, provider, timeout=8)

    lines = _timeline_lines(caplog)
    turn = f"live-utt:{H.PSID}:1"
    mine = [line for line in lines if f"turn_id={turn} " in line]
    assert len(mine) == 1, lines
    fields = _fields(mine[0])
    assert fields["reason"] == "first_output"
    assert fields["close_provider_ms"] == "1200"
    assert fields["first_output_provider_ms"] == "1600"
    assert int(fields["first_output_ms"]) >= 0
    assert set(_fields(mine[0])) <= {
        "turn_id", "reason", "close_provider_ms", "first_output_ms", "first_output_provider_ms",
    }, mine[0]
    for word in ("what", "time", "tokyo", "morning"):
        assert word not in _values(mine[0]), mine[0]


@pytest.mark.asyncio
async def test_a_delegated_turn_logs_one_timeline_line_at_its_task_terminal(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    H.fast_clocks(monkeypatch, voice_live_utterance_gap_ms=200,
                  voice_live_output_epoch_gap_ms=150)

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        await asyncio.sleep(0.3)
        return "Robarts Library closes at 11 pm tonight.", "m"

    H.patch_relay(monkeypatch, think=think)
    provider = _provider(auto_ack=True, speak_on_commentary=True)

    def u1():
        provider.push(H.user_delta("when does robarts library close tonight", 0, 1800))
        provider.push(H.delegation("d1", 1850))

    client = H.FakeClient([
        H.config(features=APP), 0.1, u1, 0.35,
        lambda: (provider.push(H.out_text("Let me check that for you.", 2000, 2600)),
                 provider.push(H.out_audio())),
        1.0, {"type": "stop"},
    ])
    await H.run_relay(client, provider, timeout=8)

    turn = f"live-utt:{H.PSID}:1"
    mine = [line for line in _timeline_lines(caplog) if f"turn_id={turn} " in line]
    assert len(mine) == 1, _timeline_lines(caplog)
    fields = _fields(mine[0])
    assert fields["reason"] == "task_terminal" and fields["outcome"] == "ok"
    for key in ("close_provider_ms", "created_ms", "dispatched_ms", "finished_ms"):
        assert key in fields, fields
    assert int(fields["finished_ms"]) >= int(fields["dispatched_ms"]) >= 0
    for word in ("robarts", "library", "close", "tonight", "check"):
        assert word not in _values(mine[0]), mine[0]
