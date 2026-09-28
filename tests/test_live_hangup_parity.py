"""The RELAY half of live/saved parity when the caller HANGS UP mid-reply.

The barge-in path was repaired first and is pinned in `test_live_voice_records`
and `test_live_bargein_echo`: the caller talks over the reply, the client sends
an identity-bound `interrupt` carrying `heard_text`, and the thread records the
prefix that was actually on screen.  Explicit hangup is the SAME parity
requirement reached by a different path, and it was open a round longer.

The defect, as reproduced: the app's teardown reset the caption identity, the
played-ms clock and the reveal state BEFORE it sent anything, so the frame it
finally put on the socket was a BARE `stop` and the valid final visible prefix
had already been discarded.  The relay's teardown then force-flushes the parked
epoch (`flush_parked_spoken(force=True)`) with the full GENERATED text, because
a receipt that has not arrived by teardown is never arriving — the right answer
for a dead socket and the wrong answer for a socket that was open and simply
never spoke.  `/private/tmp/toup-supervisor-hangup-repro.py` replays that
bare-stop wire and fails on exactly that.

WHAT THIS FILE PROVES, and what it does not.  It proves that when the frozen
end-call sequence DOES arrive, the relay honours it end to end — through the
real `on_interrupt` -> `note_heard_text` -> `persist_spoken` /
`release_parked_spoken` path and past the teardown force-flush — and that every
degraded and hostile shape around it stays honestly distinguished.  It proves
NOTHING about the React hook that has to send those frames: that is the app
repo's half, it is repaired there, and asserting it from Python would be
fabricated coverage.  The reproducer above is a wire replay plus source
evidence and stays failing while it models a bare stop; making it pass by
teaching the replay to send a receipt would prove only that this file's
assumptions agree with themselves.

THE FROZEN WIRE CONTRACT this builds against (see `end_call_frames`, which is
the single authority every test below replays):

  1. epoch live AND the caption identity matches the playback identity ->
     `{"type":"interrupt", response_id, item_id, played_ms,
       "reason":"client_request", "heard_text": <exact displayed prefix>}`
     then `{"type":"stop"}`, in that order, on the still-open socket.
  2. epoch live but the identities DISAGREE -> the same interrupt WITHOUT the
     `heard_text` key, then stop.  The client genuinely cannot say.
  3. no epoch live -> only `{"type":"stop"}`.

`"reason": "client_request"` is an already-declared value in contract §4.10, so
the end-call path puts no new enum on the wire.  `heard_text: ""` is a real
receipt meaning "nothing was displayed"; an ABSENT key means "this client
cannot say".  They are different facts with different remedies and every test
here keeps them apart.

Assertions are on the PUBLIC PROJECTION — `message_cards.public_text` composed
with `schemas.public_heard_text`, the pair every read surface composes
(`api/day_chats.py` both arms, `api/sessions.py`, `api/messages_recover.py` and
the live `api/message_frames.py`) — and not only on the raw payload.
The persistence API cannot carry an empty body, so for the nothing-heard cases
the raw row and the readable row deliberately disagree and only the readable
one answers "what would a caller see".  `test_nothing_heard_end_to_end`
established that boundary; this file reuses it.
"""

import asyncio

import pytest

import test_live_harness as H


#: A build that can report what it displayed.  `heard_text` gates BOTH halves
#: of the repair — accepting the field and the record split it enables — so a
#: test that forgets it is testing the legacy client and will pass while
#: proving nothing.
FEATURES = [
    "live_turns", "state_seq", "outcomes", "delegation_frames",
    "task_lifecycle", "playback_frames", "turn_timing", "media_control",
    "heard_text",
]

#: Distinguishes "this client cannot say" from `heard_text=""`, which is a real
#: receipt.  A plain `None` default would work, but naming the sentinel is what
#: makes a call site that means ABSENT unmistakable at the call site.
ABSENT = object()


def end_call_frames(output_id, played_ms=0, heard_text=ABSENT):
    """The frames a client must send when the user ends a call, in order.

    Pure and tiny on purpose: it is the ONE place the frozen sequence is
    written down on this side, so a test cannot quietly drift into asserting a
    wire the app was never asked to send — which is the exact failure mode the
    supervisor called out ("changing only the repro to send a made-up receipt
    would not fix the app").  The app repo owns the TypeScript twin —
    `src/shared/voice/liveProtocol.ts::closeCallFrames`, read against this file
    and agreeing shape for shape — and the two are held together by the
    contract rather than by shared code, because nothing can be shared across
    those two runtimes.  The twin differs in one place that is deliberately not
    modelled here: it reports `played_ms` RELATIVE to the epoch's own anchor,
    which is a playback number and never touches the text this file asserts on.

    `output_id=None` means no epoch is live: the stop frame goes out alone, and
    no interrupt is invented for an epoch that does not exist.  `heard_text`
    defaults to ABSENT — "this client cannot say" — because that is the shape
    for a caption identity that does not match the playback identity, and
    because a helper that defaulted to `""` would silently fabricate a receipt
    of silence for every caller that forgot the argument.
    """

    if output_id is None:
        return [{"type": "stop"}]
    interrupt = {
        "type": "interrupt",
        "response_id": output_id,
        "item_id": output_id,
        "played_ms": int(played_ms),
        # Already declared in §4.10, so the end-call path adds no enum to the
        # wire.  It is NOT `user_barge_in`: the caller did not talk over the
        # reply, they ended the call, and the relay echoes this back on
        # `playback_interrupted` for the client's own bookkeeping.
        "reason": "client_request",
    }
    if heard_text is not ABSENT:
        interrupt["heard_text"] = heard_text
    return [interrupt, {"type": "stop"}]


def _oid(epoch: int) -> str:
    return f"live:{H.PSID}:{epoch}"


def _config(**extra):
    return H.config(features=list(FEATURES), **extra)


def _rows(saves, prefix):
    return [s for s in saves if str(s.get("assistant_ref") or "").startswith(prefix)]


def _voice(row) -> dict:
    return row.get("assistant_voice") or {}


def _public(row) -> str:
    """What a caller re-opening this chat would actually be served.

    Both serializers, in the order `schemas.public_heard_text` documents:
    `public_text` answers for the row's ROLE, and the projection then answers
    for its voice provenance.  Reading only `assistant_text` is what let a
    retracted row look saved-but-invisible in one direction and
    saved-and-visible in the other.
    """

    from app.api.message_cards import public_text
    from app.schemas import public_heard_text

    return public_heard_text(public_text("assistant", row.get("assistant_text")), _voice(row))


def _reasons(counters, name="live_heard_text_rejected"):
    return [fields.get("reason") for got, fields in counters if got == name]


def _await_frame(client, kind, timeout=3.0):
    """A script step that blocks until the relay has SENT `kind`.

    A fixed sleep here would not merely be flaky, it would test a different
    scenario: provider output that arrives while the user turn is still open is
    DEFERRED, and an end-call interrupt that overtook the reply is "the caller
    hung up on output that never left the relay", not the hangup-mid-speech
    case every test in this file is about.
    """

    async def wait():
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while not client.of(kind):
            if loop.time() >= deadline:
                raise AssertionError(f"timed out waiting for a {kind} frame")
            await asyncio.sleep(0.01)

    return wait


#: Past `voice_live_output_epoch_gap_ms` under `fast_clocks` (80 ms), so the
#: tick has rotated the epoch and PARKED its row before the hangup lands.  That
#: ordering is the one a real phone produces — the relay's idle timer fires
#: while seconds of buffered audio are still playing — and it is the ordering
#: in which the teardown force-flush is armed and could undo the receipt.
AFTER_GAP = 0.25

FULL = "First sentence. Second sentence."
HEARD = "First sentence."


def _reply_on_send(provider, event):
    if event["type"] == "session.start":
        provider.push(H.user_delta("what's the deadline", 0, 700))
        provider.push(H.out_text(FULL, 900, 1500))
        provider.push(H.out_audio())
    elif event["type"] == "session.close":
        provider.push(H.closed())


async def _run(client, on_send=_reply_on_send, **kwargs):
    await H.run_relay(client, H.FakeProvider(on_send=on_send, **kwargs), timeout=8)


# ── the wire contract itself ──────────────────────────────────────────


def test_the_close_frame_helper_keeps_absent_and_empty_apart():
    """Three shapes, and the one distinction the whole repair rests on.

    `heard_text: ""` and no `heard_text` key at all must never be built by the
    same call: the first suppresses the row, the second falls back to the
    generated text.  A helper that collapsed them would make every test below
    agree with itself while the relay was handed the wrong frame.
    """

    matched = end_call_frames(_oid(1), played_ms=640, heard_text=HEARD)
    assert [f["type"] for f in matched] == ["interrupt", "stop"], (
        "the receipt must precede the stop on the still-open socket"
    )
    assert matched[0]["response_id"] == matched[0]["item_id"] == _oid(1)
    assert matched[0]["reason"] == "client_request"
    assert matched[0]["heard_text"] == HEARD

    nothing_displayed = end_call_frames(_oid(1), heard_text="")
    assert nothing_displayed[0]["heard_text"] == "", "empty is a real receipt"

    cannot_say = end_call_frames(_oid(1), played_ms=640)
    assert "heard_text" not in cannot_say[0], (
        "a mismatched caption identity must omit the key, not send an empty one"
    )

    assert end_call_frames(None) == [{"type": "stop"}], (
        "no live epoch invents no interrupt"
    )


# ── the headline: the prefix survives teardown ────────────────────────


@pytest.mark.asyncio
async def test_hangup_after_the_gap_rotation_saves_exactly_the_displayed_prefix(
    monkeypatch,
):
    """The production ordering, and the one the blocker is about.

    The tick has already retired the epoch on the output gap and PARKED its
    row, so the teardown force-flush is armed with the full generated text.
    The receipt arrives on the last frame before `stop`, on the same socket:
    `on_interrupt` re-retires the epoch out of the adapter's history, validates
    the claim and persists it, which releases the park — so the flush that runs
    moments later finds nothing to write.  One row, one sentence.
    """

    H.fast_clocks(monkeypatch)
    recorded = H.patch_relay(monkeypatch)
    client = H.FakeClient()
    client._script = [
        _config(), _await_frame(client, "audio_delta"), AFTER_GAP,
        *end_call_frames(_oid(1), played_ms=640, heard_text=HEARD),
    ]
    await _run(client)

    rows = _rows(recorded["saves"], "live-output:")
    assert len(rows) == 1, (
        "the receipt released the park, so teardown must not write a second, "
        f"fuller row: {[r.get('assistant_text') for r in rows]}"
    )
    assert _public(rows[-1]) == HEARD, (
        "the caller hung up after one sentence and the thread must say one "
        "sentence"
    )
    voice = _voice(rows[-1])
    assert voice["heard_chars"] == len(HEARD)
    assert voice["generated_chars"] == len(FULL), (
        "what the model produced is still recorded; it is simply not what was "
        "heard"
    )
    assert voice["interrupted"] is True
    assert voice["played_ms"] == 640
    assert voice.get("transcript_retracted") is None, (
        "a partly-heard row is not a retracted one"
    )


@pytest.mark.asyncio
async def test_hangup_before_the_gap_rotation_saves_exactly_the_displayed_prefix(
    monkeypatch,
):
    """The other ordering: the epoch is still OPEN when the call ends.

    A short reply, a fast hangup, or a long output gap all reach here, and it
    takes a different branch of `LiveClientAdapter.interrupt` — the epoch is
    retired BY the interrupt rather than found in history — so it is proved
    separately rather than assumed to be the same code.
    """

    H.fast_clocks(monkeypatch, voice_live_output_epoch_gap_ms=4000)
    recorded = H.patch_relay(monkeypatch)
    client = H.FakeClient()
    client._script = [
        _config(), _await_frame(client, "audio_delta"),
        *end_call_frames(_oid(1), played_ms=640, heard_text=HEARD),
    ]
    await _run(client)

    rows = _rows(recorded["saves"], "live-output:")
    assert len(rows) == 1
    assert _public(rows[-1]) == HEARD
    assert _voice(rows[-1])["heard_chars"] == len(HEARD)


@pytest.mark.asyncio
async def test_hangup_with_an_empty_receipt_saves_no_unheard_words(monkeypatch):
    """"Nothing reached the screen" ends the call with nothing in the thread.

    The strongest form of the nothing-heard rule: the words are not projected
    away later, they are never written.  The park is what makes that reachable
    — without it the gap timer would have published the full answer seconds
    before the caller hung up, and the only remedy left would be retraction.
    """

    H.fast_clocks(monkeypatch)
    recorded = H.patch_relay(monkeypatch)
    client = H.FakeClient()
    client._script = [
        _config(), _await_frame(client, "audio_delta"), AFTER_GAP,
        *end_call_frames(_oid(1), played_ms=0, heard_text=""),
    ]
    await _run(client)

    assert _rows(recorded["saves"], "live-output:") == [], (
        "an empty receipt is a receipt: nothing was displayed, so nothing may "
        "be saved"
    )


@pytest.mark.asyncio
async def test_hangup_with_no_receipt_falls_back_to_the_generated_text(monkeypatch):
    """THE DEGRADED PATH, not the target — and it must stay exactly as it is.

    A socket that is already dead cannot deliver a receipt, and no amount of
    relay cleverness can conjure one: `played_ms` is a playback clock, not a
    display claim, and estimating a prefix from it would be inventing a fact
    about what the caller read.  So a bare `stop` keeps today's honest
    fallback, the canonical generated text, and this test exists to stop a
    later repair from "fixing" it into silence.

    This is also the exact wire `/private/tmp/toup-supervisor-hangup-repro.py`
    replays, which is why that reproducer stays failing: a bare stop has no
    receipt to honour, and the relay refuses to invent one.
    """

    H.fast_clocks(monkeypatch)
    recorded = H.patch_relay(monkeypatch)
    client = H.FakeClient()
    client._script = [
        _config(), _await_frame(client, "audio_delta"), AFTER_GAP,
        *end_call_frames(None),
    ]
    await _run(client)

    rows = _rows(recorded["saves"], "live-output:")
    assert len(rows) == 1
    assert _public(rows[-1]) == FULL, (
        "with no receipt the relay must say what it generated rather than "
        "guess at a prefix"
    )
    assert "heard_chars" not in _voice(rows[-1]), (
        "no receipt arrived, so no receipt may be recorded"
    )


@pytest.mark.asyncio
async def test_an_end_call_interrupt_without_a_receipt_key_is_not_read_as_silence(
    monkeypatch,
):
    """Caption identity did not match the playback identity.

    The client still has to stop the speech, so frame 1 goes out — but it
    carries no `heard_text`, because this build genuinely cannot say what was
    on screen for THIS epoch.  ABSENT must land on the same honest fallback as
    a dead socket, never on the empty-receipt suppression: collapsing the two
    would delete an answer the caller may well have heard in full.
    """

    H.fast_clocks(monkeypatch)
    recorded = H.patch_relay(monkeypatch)
    counters = H.capture_counters(monkeypatch)
    client = H.FakeClient()
    client._script = [
        _config(), _await_frame(client, "audio_delta"), AFTER_GAP,
        *end_call_frames(_oid(1), played_ms=640),
    ]
    await _run(client)

    rows = _rows(recorded["saves"], "live-output:")
    assert len(rows) == 1
    assert _public(rows[-1]) == FULL
    voice = _voice(rows[-1])
    assert "heard_chars" not in voice
    assert voice.get("transcript_retracted") is None
    assert voice["interrupted"] is True, "the epoch was still cut short"
    assert _reasons(counters) == [], (
        "an absent claim is the ordinary shape of a frame, never a refusal"
    )


@pytest.mark.asyncio
async def test_a_repeated_end_call_receipt_is_idempotent(monkeypatch):
    """Teardown can run twice — unmount after an explicit close, or a retry.

    The second copy names an epoch the first one already retired and persisted.
    It must re-state the same fact, not restore the rest of the answer: the
    epoch is found in history, the same prefix validates again, and the row is
    rewritten under the same key with a monotonic revision.
    """

    H.fast_clocks(monkeypatch)
    recorded = H.patch_relay(monkeypatch)
    first, _stop = end_call_frames(_oid(1), played_ms=640, heard_text=HEARD)
    client = H.FakeClient()
    client._script = [
        _config(), _await_frame(client, "audio_delta"), AFTER_GAP,
        dict(first), dict(first), {"type": "stop"},
    ]
    await _run(client)

    rows = _rows(recorded["saves"], "live-output:")
    assert len(rows) == 2, (
        "one write per receipt, and the retry is a receipt: "
        f"{[r.get('assistant_text') for r in rows]}"
    )
    assert {r["assistant_ref"] for r in rows} == {f"live-output:{H.PSID}:1"}, (
        "a retry must revise one row, never open a second key"
    )
    assert [r["assistant_revision"] for r in rows] == sorted(
        r["assistant_revision"] for r in rows
    ), "revisions must stay monotonic across the retry"
    assert _public(rows[-1]) == HEARD, (
        "the retry restored the rest of the answer"
    )


# ── the fences that must survive the end-call path ────────────────────


@pytest.mark.asyncio
async def test_hangup_while_a_delegated_task_runs_never_cancels_it(monkeypatch):
    """Ending the call stops SPEECH.  It has never stopped the work (A6-2/D4).

    The end-call sequence puts an `interrupt` on the wire where there used to
    be only a `stop`, and an interrupt that leaked into the delegation path
    would throw away research the caller asked for — and paid for — seconds
    before it landed.  So this pins both halves at once: the runner is detached
    rather than cancelled, and its answer still reaches the same conversation
    in FULL, while the paraphrase the caller actually heard is recorded
    separately as the prefix.
    """

    H.fast_clocks(monkeypatch)
    started = asyncio.Event()
    release = asyncio.Event()
    cancelled = asyncio.Event()

    async def think(user_id, task, session_id, relay=None, out=None, **kwargs):
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return "The deadline is 15 January.", "m"

    recorded = H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("look up the deadline", 0, 400))
            provider.push(H.delegation("d1", 420))
            provider.push(H.out_text("Let me look that up.", 500, 900))
            provider.push(H.out_audio())
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient()
    client._script = [
        _config(), 0.6,
        *end_call_frames(_oid(1), played_ms=300, heard_text="Let me look"),
    ]
    await _run(client, on_send=on_send)

    assert started.is_set(), "the delegation must have been dispatched at all"
    assert not cancelled.is_set(), (
        "the end-call interrupt cancelled the delegated work"
    )
    assert "cancelled" not in client.phases(), (
        "an interrupt must never be reported to the client as a cancel"
    )

    release.set()
    for _ in range(80):
        await asyncio.sleep(0.05)
        if _rows(recorded["saves"], "live-delegation:"):
            break
    result = _rows(recorded["saves"], "live-delegation:")
    assert result, "detached work must still persist its answer"
    assert _public(result[-1]) == "The deadline is 15 January.", (
        "the task result is not a playback claim and is never clipped by a "
        "receipt"
    )

    spoken = _rows(recorded["saves"], "live-output:")
    assert spoken and _public(spoken[-1]) == "Let me look", (
        "the paraphrase the caller heard is its own, shorter fact"
    )


@pytest.mark.asyncio
async def test_a_forged_receipt_at_hangup_cannot_rewrite_the_parked_row(monkeypatch):
    """The provisional stamp must never outlive the validation that follows it.

    `LiveClientAdapter.interrupt` writes what the frame CLAIMED onto the epoch
    at the same moment it writes `played_ms`, before anything has checked it;
    `note_heard_text` is the only thing that decides whether it stands, and
    `on_interrupt` runs the two with no await between them for exactly this
    reason.  On the end-call path a parked row is sitting behind a teardown
    flush, so an unvalidated claim that survived would be persisted by that
    flush — the receipt fence bypassed by the very frame it exists to police.

    A claim that is not a prefix of what this epoch said is therefore refused
    by reason, and the row falls back to the generated text: being wrong must
    never be more powerful than being right.
    """

    H.fast_clocks(monkeypatch)
    recorded = H.patch_relay(monkeypatch)
    counters = H.capture_counters(monkeypatch)
    client = H.FakeClient()
    client._script = [
        _config(), _await_frame(client, "audio_delta"), AFTER_GAP,
        *end_call_frames(
            _oid(1), played_ms=640, heard_text="Something else entirely.",
        ),
    ]
    await _run(client)

    assert _reasons(counters) == ["not_prefix"]
    rows = _rows(recorded["saves"], "live-output:")
    assert len(rows) == 1
    assert _public(rows[-1]) == FULL, (
        "an unbelievable receipt was persisted as though it had been checked"
    )
    assert "heard_chars" not in _voice(rows[-1])


@pytest.mark.asyncio
async def test_a_stale_receipt_naming_an_older_epoch_cannot_reach_the_new_one(
    monkeypatch,
):
    """A late or forged receipt buys nothing, in either direction.

    Two epochs are parked when the call ends and the receipt names the OLDER
    one while quoting the NEWER one's words.  Step 5 of the validation order
    refuses it — the claim is not a prefix of the epoch it names — and both
    rows then fall back to their own generated text.  What must NOT happen is
    the claim leaking onto epoch 2 because its text happens to match, or epoch
    1 being rewritten with words it never said.
    """

    H.fast_clocks(monkeypatch)
    recorded = H.patch_relay(monkeypatch)
    counters = H.capture_counters(monkeypatch)
    box = {}

    def on_send(provider, event):
        box["provider"] = provider
        if event["type"] == "session.start":
            provider.push(H.out_text("Older answer.", 0, 300))
            provider.push(H.out_audio())
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def second_epoch():
        await asyncio.sleep(AFTER_GAP)          # epoch 1 retires and parks
        box["provider"].push(H.out_text("Newer answer entirely.", 900, 1300))
        box["provider"].push(H.out_audio())
        await asyncio.sleep(AFTER_GAP)          # epoch 2 retires and parks

    client = H.FakeClient()
    client._script = [
        _config(), _await_frame(client, "audio_delta"), second_epoch,
        *end_call_frames(_oid(1), played_ms=200, heard_text="Newer answer"),
    ]
    await _run(client, on_send=on_send)

    assert _reasons(counters) == ["not_prefix"], (
        "a receipt quoting another epoch must be refused, by that reason"
    )
    by_ref = {r["assistant_ref"]: _public(r) for r in _rows(recorded["saves"], "live-output:")}
    assert by_ref == {
        f"live-output:{H.PSID}:1": "Older answer.",
        f"live-output:{H.PSID}:2": "Newer answer entirely.",
    }, "a refused claim must leave both epochs exactly as they were"
    assert all(
        "heard_chars" not in _voice(r)
        for r in _rows(recorded["saves"], "live-output:")
    ), "a refused claim must not be recorded as a receipt on any row"


@pytest.mark.asyncio
async def test_a_persian_prefix_survives_the_end_call_path(monkeypatch):
    """The same path, in the language most of these calls are actually in.

    Persian is not a smoke test here.  The receipt is validated by a PREFIX
    check against the canonical epoch text, and canonicalization touches
    zero-width joiners and Arabic-Indic forms — so a normalization that
    disagreed between the two sides would refuse every real Persian receipt and
    silently fall back to the full answer, which reads as the bug being unfixed
    for exactly the callers who hit it most.
    """

    H.fast_clocks(monkeypatch)
    recorded = H.patch_relay(monkeypatch)
    counters = H.capture_counters(monkeypatch)
    full = "سلام. حالت چطوره؟ امروز چه کاری داری؟"
    heard = "سلام. حالت چطوره؟"

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.out_text(full, 0, 900))
            provider.push(H.out_audio())
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient()
    client._script = [
        _config(), _await_frame(client, "audio_delta"), AFTER_GAP,
        *end_call_frames(_oid(1), played_ms=700, heard_text=heard),
    ]
    await _run(client, on_send=on_send)

    rows = _rows(recorded["saves"], "live-output:")
    assert _reasons(counters) == []
    assert len(rows) == 1
    assert _public(rows[-1]) == heard
    assert _voice(rows[-1])["heard_chars"] == len(heard)


# ── the slow phone: the row was already published ─────────────────────


def _settle_window(monkeypatch, seconds: float) -> None:
    """Shorten the receipt window for a relay run.

    Patched on the METHOD rather than through `H.fast_clocks`, because the
    window is not a `Settings` field yet: `voice_live_spoken_settle_ms` belongs
    in `app/config.py`, and `Settings` refuses `setattr` for a name it does not
    declare.  `spoken_settle_s` is the one seam that field will swap into, so
    this is the same override the setting will be.
    """

    import app.services.live_voice_protocol as live

    monkeypatch.setattr(live._LiveSession, "spoken_settle_s", lambda self: seconds)


@pytest.mark.asyncio
async def test_a_receipt_that_arrives_after_the_row_was_written_corrects_it_down(
    monkeypatch,
):
    """The park is a window, not a guarantee, and a long call can outlast it.

    The row is then already in the thread and cannot be un-published, so the
    end-call receipt has to reach it as a REVISION under the same key.  The
    text corrects downward to the prefix; a second, shorter bubble alongside
    the first would be the same reply twice.
    """

    H.fast_clocks(monkeypatch)
    _settle_window(monkeypatch, 0.05)
    recorded = H.patch_relay(monkeypatch)
    client = H.FakeClient()
    client._script = [
        _config(), _await_frame(client, "audio_delta"), 0.5,
        *end_call_frames(_oid(1), played_ms=640, heard_text=HEARD),
    ]
    await _run(client)

    rows = _rows(recorded["saves"], "live-output:")
    assert len(rows) == 2, "the window expired first, so there are two writes"
    assert {r["assistant_ref"] for r in rows} == {f"live-output:{H.PSID}:1"}
    assert rows[0]["assistant_revision"] < rows[-1]["assistant_revision"]
    assert _public(rows[0]) == FULL, "the provisional row said everything"
    assert _public(rows[-1]) == HEARD, "the receipt must correct it downward"


@pytest.mark.asyncio
async def test_an_empty_receipt_after_the_row_was_written_retracts_it(monkeypatch):
    """Same ordering, nothing displayed: the row cannot be emptied, only retracted.

    `_save_voice_messages` refuses an empty row and the receiving route 400s on
    empty content, so "write no text" is unavailable once a row exists.  The
    words stay in the database and stop reaching a client, which is why the
    assertion that matters is the projection and not the payload.
    """

    H.fast_clocks(monkeypatch)
    _settle_window(monkeypatch, 0.05)
    recorded = H.patch_relay(monkeypatch)
    client = H.FakeClient()
    client._script = [
        _config(), _await_frame(client, "audio_delta"), 0.5,
        *end_call_frames(_oid(1), played_ms=0, heard_text=""),
    ]
    await _run(client)

    rows = _rows(recorded["saves"], "live-output:")
    assert len(rows) == 2
    latest = _voice(rows[-1])
    assert latest["transcript_retracted"] is True
    assert latest["heard_chars"] == 0
    assert latest["interrupted"] is True, (
        "the audio never reached this caller, whatever the retirement said"
    )
    assert rows[-1]["assistant_text"], (
        "the body is kept — the seam cannot carry an empty one"
    )
    assert _public(rows[-1]) == "", (
        "a caller re-opening this chat would be shown words they never heard"
    )
