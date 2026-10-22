"""The saved call reads the way the live call read, and nothing the tenant could
have taken is lost or mis-reported (R2 fix round, F31-F35 + addendum 9).

The phone lays a call out in TIME order (D-P7 R3, 2026-10-05; it was causal,
`placeCausal`, until then): a user turn is appended when it appears, and an
assistant reply is appended when it first appears — never above a user turn
that was final before it began, however late it arrives.  The day chat sorts
by `occurred_at` alone. These tests drive the real relay with provider-shaped
events and compare the saved `occurred_at` order with the live order the
relay's own frames produce (`live_order` below mirrors the app reducer's
placement rule). Stamps are also required to be distinct at MILLISECOND
precision, because that is what the app's day-chat sort compares.
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
    """The app's call transcript order for these relay frames (D-P7 R3).

    User bubbles append on their first `transcript` frame; an assistant entry
    appends on its first `response_text`, moving up only past TRAILING
    assistant entries of a higher epoch — never above a user entry.  Its
    parent no longer places it.
    """

    entries: list[dict] = []
    for f in frames:
        kind = f.get("type")
        if kind == "transcript" and f.get("turn_id"):
            if not any(e["role"] == "user" and e["id"] == f["turn_id"] for e in entries):
                entries.append({"id": f["turn_id"], "role": "user"})
        elif kind == "response_text" and f.get("assistant_turn_id"):
            aid = f["assistant_turn_id"]
            if any(e["id"] == aid for e in entries):
                continue
            e = {"id": aid, "role": "assistant",
                 "parent": f.get("parent_user_turn_id") or "", "epoch": int(f.get("epoch") or 0)}
            at = len(entries)
            while at > 0 and entries[at - 1]["role"] == "assistant" and entries[at - 1]["epoch"] > e["epoch"]:
                at -= 1
            entries.insert(at, e)
    return entries


def assert_saved_matches_live(recorded, client):
    rows = saved_rows(recorded["saves"])
    order = saved_order(rows)
    saved_ids = []
    for ref in order:
        ident = identity(ref)
        if ident not in saved_ids:
            saved_ids.append(ident)
    live = [e["id"] for e in live_order(client.frames)]
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


# ── F32 → D-P7 R2: parked replies stay where the phone saw them; late
#    results are saved where they arrive — after the turns said meanwhile ─

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
async def test_a_delegated_result_is_saved_under_its_question_after_its_own_ack(monkeypatch):
    """PIN RE-BASELINED — D-P7 R2 reversed (DESIGN-R2 D-R8 b).  It was
    "saved where it arrived, after later turns": tail stamps put the 2026-10-06
    «استاپش کن» card 45 s away from its question.  A task result now sits
    under the question that asked for it — after the task's OWN
    acknowledgement line, above the small talk said while it ran — whenever it
    is written.  The paraphrase folds onto it (D-R8 a): no sibling row."""

    recorded, client = await _delegated_with_small_talk(monkeypatch)
    order, rows = assert_saved_matches_live(recorded, client)
    result = f"live-delegation:{H.PSID}:d1"
    u1, u2 = f"live-utt:{H.PSID}:1", f"live-utt:{H.PSID}:2"
    assert rows[result]["voice"].get("parent_user_turn_id") == u1
    hold = f"live-output:{H.PSID}:1"
    assert order.index(u1) < order.index(hold) < order.index(result) < order.index(u2), order
    assert not [r for r in order if r.startswith("live-transcript:")], order


@pytest.mark.asyncio
async def test_rows_fit_between_turns_at_millisecond_precision_despite_timeline_skew(monkeypatch):
    """Production: output positions run seconds ahead of the input timeline,
    so the next user row is clamped right below the reply. Every row still
    gets its own millisecond, in the order the call showed them (D-P7 R2:
    the late result for the EARLIER turn after the later one)."""

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
    # PIN RE-BASELINED — D-P7 R2 reversed (DESIGN-R2 D-R8 b): the late result
    # for the EARLIER turn sits under that turn now, not after the later one.
    assert order.index(f"live-utt:{H.PSID}:1") < order.index(result) < order.index(
        f"live-utt:{H.PSID}:2"), order
    # …and every stamp is the WALL clock, not `anchor + provider_ms`: the
    # provider timeline here runs 28 s ahead and no row may follow it.
    now = datetime.now(timezone.utc)
    for ref, row in rows.items():
        assert abs((now - row["at"]).total_seconds()) < 5.0, (ref, row["at"], now)


@pytest.mark.asyncio
async def test_a_media_card_that_starts_after_the_next_turn_is_saved_under_its_ask(monkeypatch):
    """The fast path's card is the delegated result of a "play" request, and a
    slow tenant play can finish after the caller has said something else.

    PIN RE-BASELINED — D-P7 R2 reversed (DESIGN-R2 D-R8 b): it was saved
    after that; it sits under the ask that started it now."""

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
    """No later turn: the delegated row sits right under its question, and
    every stamp stays in write order.

    PIN RE-BASELINED — D-P7 R2 reversed (DESIGN-R2 D-R8 b): it was
    `utterance.end_ms + 1` on the provider timeline (0.401 s after the
    question's START, i.e. the question's length); it is the question's own
    real stamp + 1 ms now (no acknowledgement epoch was voiced)."""

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
    assert (result - u2).total_seconds() == pytest.approx(0.001, abs=0.0005)
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
    """The paraphrase of a delegated result, folded when its epoch is
    gap-retired — before any ack.

    PIN FLIPPED (voice round 2026-10-06, DESIGN-R2 D-R8 a): it was a
    `live_transcript` sibling row; it folds onto the task row now, which
    carries `spoken_interrupted` — through the tenant allowlist, no null."""

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
    assert heard == [], [s.get("assistant_voice") for s in recorded["saves"]]
    folded = [s for s in recorded["saves"]
              if "spoken_interrupted" in (s.get("assistant_voice") or {})]
    assert folded, [s.get("assistant_voice") for s in recorded["saves"]]
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
    # Pin widened, voice round 2026-10-06 (D-R1 (h)): every line now says
    # whether the turn was a request, how stale the phone's and the
    # provider's input were when it was written, and how long the relay HELD
    # the reply's first output (`held_ms`, present when it was held) — the
    # relay's own share of direct-reply latency.  Still counts only.
    assert set(_fields(mine[0])) <= {
        "turn_id", "reason", "close_provider_ms", "first_output_ms", "first_output_provider_ms",
        "request", "since_client_audio_ms", "since_input_audio_ms", "held_ms",
    }, mine[0]
    assert fields["request"] == "1"
    assert int(fields["since_input_audio_ms"]) >= -1
    assert "since_client_audio_ms" in fields
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


@pytest.mark.asyncio
async def test_a_never_answered_request_still_logs_its_timeline_line(monkeypatch, caplog):
    """D-P1 (2026-10-05 call): the one per-turn latency record was missing
    exactly for the turns that FAILED.  A request superseded by a newer one
    before anything answered it logs `reason=unanswered_superseded`; one the
    call ended on logs `reason=unanswered` at teardown — with the extra
    fields only on those lines, never the caller's words."""

    caplog.set_level(logging.INFO)
    H.fast_clocks(monkeypatch, voice_live_utterance_gap_ms=200,
                  voice_live_output_epoch_gap_ms=150)
    H.patch_relay(monkeypatch)
    provider = _provider()
    client = H.FakeClient([
        H.config(features=APP), 0.1,
        lambda: provider.push(H.user_delta("what time is it in tokyo", 0, 1200)),
        0.5,
        lambda: provider.push(H.user_delta("and how about paris", 4000, 5200)),
        0.5, {"type": "stop"},
    ])
    await H.run_relay(client, provider, timeout=8)

    lines = _timeline_lines(caplog)
    first, second = f"live-utt:{H.PSID}:1", f"live-utt:{H.PSID}:2"
    mine = [line for line in lines if f"turn_id={first} " in line]
    assert len(mine) == 1, lines
    assert _fields(mine[0])["reason"] == "unanswered_superseded", mine[0]
    assert _fields(mine[0])["close_provider_ms"] == "1200"
    theirs = [line for line in lines if f"turn_id={second} " in line]
    assert len(theirs) == 1, lines
    assert _fields(theirs[0])["reason"] == "unanswered", theirs[0]
    for line in mine + theirs:
        # Pin widened (D-R1 (h)): + `request`, the two input clocks, and the
        # ladder's fields (absent here: nothing re-asked, the ladder is off).
        assert set(_fields(line)) <= {
            "turn_id", "reason", "close_provider_ms", "reask_ms", "delayed_ms",
            "client_audio_end_ms", "request", "since_client_audio_ms",
            "since_input_audio_ms", "reask_n", "reask_ack_ms", "backstop_ms", "held_ms",
        }, line
        assert _fields(line)["request"] == "1", line
        for word in ("tokyo", "paris", "time"):
            assert word not in _values(line), line


# ── voice round 2026-10-06 (D-R1 (h)): every caller turn, every rung ──


@pytest.mark.asyncio
async def test_an_expired_owed_reply_logs_unanswered_timeout(monkeypatch, caplog):
    """`_REPLY_PENDING_MAX_S` with nothing — the line used to wait for the
    turn's eviction or the call's end, and said `unanswered` then."""

    from app.services import live_voice_protocol as live

    caplog.set_level(logging.INFO)
    H.fast_clocks(monkeypatch, voice_live_utterance_gap_ms=200)
    monkeypatch.setattr(live, "_REPLY_PENDING_MAX_S", 0.8)
    H.patch_relay(monkeypatch)
    provider = _provider()
    client = H.FakeClient([
        H.config(features=APP), 0.1,
        lambda: provider.push(H.user_delta("what is the capital of peru", 0, 1200)),
        1.8, {"type": "stop"},
    ])
    await H.run_relay(client, provider, timeout=8)

    turn = f"live-utt:{H.PSID}:1"
    mine = [line for line in _timeline_lines(caplog) if f"turn_id={turn} " in line]
    assert len(mine) == 1, _timeline_lines(caplog)
    fields = _fields(mine[0])
    assert fields["reason"] == "unanswered_timeout" and fields["request"] == "1", mine[0]
    assert "since_client_audio_ms" in fields and "since_input_audio_ms" in fields
    assert "peru" not in _values(mine[0])


@pytest.mark.asyncio
async def test_a_backstopped_turn_logs_its_reask_and_backstop_at_its_task_terminal(
    monkeypatch, caplog,
):
    caplog.set_level(logging.INFO)
    H.fast_clocks(
        monkeypatch, voice_live_utterance_gap_ms=200, voice_live_uplink_keeper_enabled=True,
        voice_live_reply_regen_after_s=0.3, voice_live_reply_backstop_enabled=True,
    )

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        await asyncio.sleep(0.2)
        return "Lima is the capital of Peru.", "m"

    H.patch_relay(monkeypatch, think=think)
    provider = _provider(auto_ack=True)
    client = H.FakeClient([
        H.config(features=APP), 0.1,
        lambda: provider.push(H.user_delta("what is the capital of peru", 0, 1200)),
        2.2, {"type": "stop"},
    ])
    await H.run_relay(client, provider, timeout=10)

    turn = f"live-utt:{H.PSID}:1"
    mine = [line for line in _timeline_lines(caplog) if f"turn_id={turn} " in line]
    assert len(mine) == 1, _timeline_lines(caplog)
    fields = _fields(mine[0])
    assert fields["reason"] == "task_terminal", mine[0]
    assert int(fields["backstop_ms"]) >= int(fields["reask_ms"]) >= 300, mine[0]
    assert fields["reask_n"] == "1" and int(fields["reask_ack_ms"]) >= 0, mine[0]
    assert int(fields["created_ms"]) >= int(fields["backstop_ms"]) - 50, mine[0]
    assert "peru" not in _values(mine[0]) and "lima" not in _values(mine[0])


@pytest.mark.asyncio
async def test_a_turn_that_asked_for_nothing_and_got_nothing_still_logs_request_0(
    monkeypatch, caplog,
):
    caplog.set_level(logging.INFO)
    H.fast_clocks(monkeypatch, voice_live_utterance_gap_ms=200)
    H.patch_relay(monkeypatch)
    provider = _provider()
    client = H.FakeClient([
        H.config(features=APP), 0.1,
        lambda: provider.push(H.user_delta("okay thanks", 0, 800)),
        0.6, {"type": "stop"},
    ])
    await H.run_relay(client, provider, timeout=8)

    turn = f"live-utt:{H.PSID}:1"
    mine = [line for line in _timeline_lines(caplog) if f"turn_id={turn} " in line]
    assert len(mine) == 1, _timeline_lines(caplog)
    fields = _fields(mine[0])
    assert fields["reason"] == "unanswered" and fields["request"] == "0", mine[0]
    assert "since_client_audio_ms" in fields, mine[0]
    assert "thanks" not in _values(mine[0])


# ── Voice round 2026-10-06 (DESIGN-R2 D-R8 b): REAL stamps ────────────
# D-P7 R2 stamped every relay row at the TAIL: `anchor_wall + provider_ms`,
# clamped above everything already stamped.  The provider timeline ran 48 s
# ahead of the wall by 3 min into the 2026-10-06 call, so «استاپش کن» —
# spoken at 12:41:07 — was saved at 12:41:55 and its job marker (which uses
# `created_at`) sorted 45 s ABOVE it.

def _stamp_session():
    from app.config import settings as app_settings
    from app.services.live_voice_protocol import (
        LiveClientAdapter, UtteranceAssembler, _LiveSession,
    )

    session = _LiveSession.__new__(_LiveSession)
    session.settings = app_settings
    session.provider_session_id = H.PSID
    session.adapter = LiveClientAdapter(session_id=H.PSID)
    session.utterances = UtteranceAssembler(provider_session_id=H.PSID)
    session.anchor_wall = datetime.now(timezone.utc)
    return session


def test_a_tie_moves_one_millisecond_and_nothing_else_moves_a_stamp():
    from datetime import timedelta

    session = _stamp_session()
    when = datetime(2026, 10, 6, 12, 41, 7, 123000, tzinfo=timezone.utc)
    a = session.stamp_at("row-a", when)
    b = session.stamp_at("row-b", when)
    later = session.stamp_at("row-c", when - timedelta(seconds=30))
    assert a == when
    assert b == when + timedelta(milliseconds=1), "a tie moves +1 ms"
    assert later == when - timedelta(seconds=30), (
        "no `last_occurred` clamp: a row stamped earlier than one already "
        "issued keeps its own time"
    )
    assert session.stamp_at("row-a", when + timedelta(hours=1)) == a, "issued once per row"


def test_a_row_written_late_keeps_the_moment_it_happened():
    """The caller turn was SEEN 3 s ago and is written now; the task it asked
    for finishes now: both keep the real order of events."""

    from datetime import timedelta

    session = _stamp_session()
    _closed, turn = session.utterances.add("play something by drake", 0, 900)
    seen = datetime.now(timezone.utc) - timedelta(seconds=3)
    turn.seen_wall = seen
    from app.services.live_voice_protocol import DelegatedTask

    task = DelegatedTask(delegation_id="d1", offset_ms=0, transcript=turn.text, turn_id=turn.turn_id)
    card = session.stamp_task_row(f"live-media:{H.PSID}:d1", task)
    user = session.stamp_user_row(turn)
    expect = datetime.fromtimestamp(int(seen.timestamp() * 1000) / 1000.0, tz=timezone.utc)
    assert user == expect, "the caller row is stamped when it was seen, written late or not"
    assert card == user + timedelta(milliseconds=1), "the card sits right under its ask"
    body = session.delegated_turn_fields(task)
    assert body["parent_user_turn_id"] == turn.turn_id
    assert body["anchor_occurred_at"] == user.replace(tzinfo=None).isoformat(timespec="milliseconds")
    assert body["task_title"] == "play something by drake"


def test_a_reply_never_sorts_above_the_turn_it_answers():
    """Critic R7: a reply's first words can reach the relay before a late
    input transcript makes its turn's row.  The reply's stamp is clamped to
    ≥ its parent's + 1 ms whenever the parent is known — at reservation, and
    again when the row is written (a parent settled after the reservation)."""

    from datetime import timedelta

    from app.services.live_voice_protocol import ClosedEpoch

    session = _stamp_session()
    _closed, turn = session.utterances.add("what is the next ufc event", 0, 900)
    turn.seen_wall = datetime.now(timezone.utc) + timedelta(seconds=2)   # seen LATE
    session.adapter.provider_event(H.out_text("UFC 300 is on Saturday.", 1000, 2000), now=0.0)
    session.reserve_epoch_stamp()
    reserved = session.epoch_stamps[session.adapter.epoch]
    # Parentless at reservation, so it is the wall clock (before the turn)…
    assert reserved < turn.seen_wall
    closed = ClosedEpoch(
        epoch=session.adapter.epoch, output_id=session.adapter.output_id,
        text="UFC 300 is on Saturday.", interrupted=False,
        parent_user_turn_id=turn.turn_id,          # …settled to the turn later
    )
    row = session.stamp_epoch_row(f"live-output:{H.PSID}:{closed.epoch}", closed)
    assert row == session.stamp_user_row(turn) + timedelta(milliseconds=1)
