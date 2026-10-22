"""What the day chat records about a call, and what the caller actually HEARD.

Three defects, one repair (Blocker B):

* ``ClosedEpoch.text`` is everything the model GENERATED for an epoch, and
  ``persist_spoken`` wrote that string — so a reply the caller cut off after
  four words was saved to the thread in full, words they never heard.
* ``fold_onto_delegated_row`` stamped an epoch's playback numbers onto the
  delegated row while leaving its ``assistant_text`` the full backend answer:
  one row carrying playback facts about a different, shorter string — and the
  spoken paraphrase persisted nowhere at all.
* The two sides normalized provider annotations differently, so any prefix
  comparison between them was meaningless before it began.

The fix is a client RECEIPT, ``heard_text``: the exact text this client
displayed for that epoch.  It is not an acoustic claim, it is never estimated
from ``played_ms``, and it is believed only after five checks in a fixed order.
Everything here pins one of those checks, one of the two record kinds, or the
degradation a client that never negotiates the feature must keep getting.

THREE MORE, found by independent review once the receipt existed.  Each one is
the same live/saved divergence arriving by a door the first pass left open, so
the sections that pin them say explicitly what they supersede:

* A LATE frame destroyed an accepted receipt.  The adapter was right — it
  refuses to overwrite a receipt with an absent claim — but
  ``_LiveSession.note_heard_text`` cleared ``heard_text`` for all three
  verdicts, so the ``playback_idle`` that ordinarily follows a barge-in put the
  whole generated answer back on a row that had correctly saved one sentence.
  A receipt is now held apart from the claim slot the adapter stamps, and only
  a newly VALIDATED receipt replaces it: being wrong buys a late frame nothing.
* THE NOTHING-HEARD RULE on the save side.  A validated receipt of ``""`` means
  the caller heard none of that epoch, and the relay used to keep the full text
  on the row with ``heard_chars: 0`` beside it — the live surface showed
  nothing while the saved chat showed everything.  The preferred fix is the
  provisional row: a gap-retired epoch waits out a bounded receipt window
  before it writes anything, so the common ordering never saves unheard words
  at all.  A row that already exists cannot be un-published or rewritten to
  empty, so it is RETRACTED, and every read surface stops serving its text.
* ``spoken_paraphrase_differs`` compared nothing.  A paraphrase that played in
  full but said something other than the task result was represented nowhere,
  so the thread showed the backend's wording as if it were the spoken turn.

A ``task_result`` row is never projected, clipped or retracted: the backend
answer is not a playback claim, and the app renders it in full.
"""

import asyncio
import inspect
import json
import pathlib
import time

import pytest

import test_live_harness as H
from app.services.live_voice_protocol import (
    FEATURE_HEARD_TEXT,
    ClosedEpoch,
    LiveClientAdapter,
    _played_ms_value,
    canonical_spoken_text,
    utf16_units,
)


# VENDORED into the repo on purpose.  The first cut read an absolute
# /private/tmp path, which made this suite depend on a scratch directory
# surviving a reboot or a tmp sweep.  The app vendors a byte-identical copy at
# scripts/fixtures/spoken-canon-fixtures.json and both sides pin the SAME
# sha256, so editing one copy without the other fails on both sides rather than
# letting the two canonicalizers drift apart quietly.
CANON_FIXTURES = str(
    pathlib.Path(__file__).parent / "fixtures" / "spoken-canon-fixtures.json"
)
CANON_SHA256 = "dcffcf919388549047a0c7ae5a8b2a83432fcf397ea5de5ad3013d6b94303237"


def test_the_canonicalization_corpus_is_the_one_the_app_pins():
    """Drift in either vendored copy fails on both sides, not silently."""

    import hashlib

    digest = hashlib.sha256(
        pathlib.Path(CANON_FIXTURES).read_bytes()
    ).hexdigest()
    assert digest == CANON_SHA256, (
        "the vendored corpus changed. Regenerate BOTH copies from the real JS "
        "and update CANON_SHA256 here and in "
        "scripts/check-live-voice-protocol.js, or the two canonicalizers are no "
        "longer pinned to one another."
    )

#: Everything a v0.2 phone negotiates, plus the new receipt.
FEATURES = [
    "live_turns", "state_seq", "outcomes", "delegation_frames",
    "task_lifecycle", "playback_frames", "turn_timing", "media_control",
    FEATURE_HEARD_TEXT,
]


def _config(**extra):
    return H.config(features=list(FEATURES), **extra)


def _oid(epoch: int) -> str:
    return f"live:{H.PSID}:{epoch}"


def _rows(saves, prefix):
    return [s for s in saves if str(s.get("assistant_ref") or "").startswith(prefix)]


def _reasons(counters, name="live_heard_text_rejected"):
    return [fields.get("reason") for got, fields in counters if got == name]


class _Persistence:
    """Everything the persistence seam needs of the queue: where a record went."""

    def __init__(self):
        self.records = []

    def submit(self, record):
        self.records.append(record)

    def keys(self) -> list[str]:
        return [r.key for r in self.records]

    def payload(self, prefix: str) -> dict:
        """The LAST payload written under a key starting with `prefix`."""

        for record in reversed(self.records):
            if record.key.startswith(prefix):
                return record.payload
        raise AssertionError(f"no record written under {prefix!r}: {self.keys()}")

    def has(self, prefix: str) -> bool:
        return any(record.key.startswith(prefix) for record in self.records)


def _session(*, features=tuple(FEATURES), park=False):
    """A `_LiveSession` carrying only what the persistence seam reads.

    Built with `__new__` for the same reason `test_live_result_delivery` does:
    the record split is a decision about ONE closed epoch and one claim, and
    constructing that state directly is the only way each branch can be failed
    on its own rather than through whichever relay scenario happens to reach it.

    `park` opts into the provisional-row window.  It is OFF by default and that
    is not a convenience: `_LiveSession.parked_spoken` is `None` on a session
    built by `__new__` precisely because such a session has no `tick_loop` and
    no teardown, so a parked row would be held by nothing and never written.
    A test that wants the window on says so and then drives it itself, which is
    the same bargain the relay makes.
    """

    from datetime import datetime, timezone
    import time as _time

    from app.config import settings as app_settings
    from app.services.live_voice_protocol import _LiveSession

    session = _LiveSession.__new__(_LiveSession)
    if park:
        session.parked_spoken = {}
    session.settings = app_settings
    session.user_id = "user-1"
    session.features = set(features)
    session.adapter = LiveClientAdapter(session_id=H.PSID)
    session.persistence = _Persistence()
    session.spoken_rows = {}
    session.provider_session_id = H.PSID
    session.deliveries = {}
    session.active_delivery = None
    session.anchor_wall = datetime.now(timezone.utc)
    session.last_occurred = None
    session.provider_now_ms = 0
    session.provider_now_at = _time.monotonic()
    return session


def _epoch(
    *, epoch=1, text="First sentence. Second sentence.", interrupted=True,
    played_ms=900, heard=None, retire_reason="interrupt",
) -> ClosedEpoch:
    closed = ClosedEpoch(
        epoch=epoch, output_id=_oid(epoch), text=text, interrupted=interrupted,
        played_ms=played_ms, parent_user_turn_id=f"live-utt:{H.PSID}:1",
        start_ms=900, end_ms=1500, retire_reason=retire_reason,
    )
    if heard is not None:
        closed.heard_text = heard
        closed.heard_source = retire_reason
    return closed


def _frame(epoch=1, **extra):
    frame = {
        "type": "interrupt",
        "response_id": _oid(epoch),
        "item_id": _oid(epoch),
        "played_ms": 900,
    }
    frame.update(extra)
    return frame


def _client_session(*, features=tuple(FEATURES), park=False):
    """`_session`, plus the few attributes the CLIENT FRAME handlers read.

    `on_interrupt`, `on_playback_idle` and `on_playback_failed` are where a
    receipt actually reaches an epoch, and the defect these tests pin lived in
    the gap between the two layers: `LiveClientAdapter._stamp_heard` preserves a
    receipt correctly when a later frame carries none, and
    `_LiveSession.note_heard_text` then cleared it anyway.  An adapter-level
    test passes over that seam without touching it, so everything below drives
    the real handlers instead.
    """

    import types

    session = _session(features=features, park=park)
    session.tracker = types.SimpleNamespace(fragments=[])
    session.supervisor = types.SimpleNamespace(running={})
    session.deferred_output = []
    session.deferred_output_bytes = 0
    session.deferred_output_overflowed = False
    session.stop_instruction_epoch = -1
    session.last_stop_instruction = 0.0
    session.tick_wake = asyncio.Event()
    session.sent: list[dict] = []
    session.provider_sent: list[dict] = []
    session.states: list[str] = []

    async def emit_frames(frames):
        session.sent.extend(frames)

    async def send(frame):
        session.sent.append(frame)
        return True

    async def provider_send(event):
        session.provider_sent.append(event)

    async def send_state(state, *, force=False):
        session.states.append(state)

    session.emit_frames = emit_frames
    session.send = send
    session.provider_send = provider_send
    session.send_state = send_state
    return session


def _speaking(session, text="First sentence. Second sentence.", *, now=0.0):
    """Open one output epoch on the session's real adapter and return its id."""

    session.adapter.provider_event(H.out_text(text, 0, 400), now=now)
    return session.adapter.output_id


# ── D4: one normalization, ported rather than approximated ────────────

def test_canonical_spoken_text_agrees_with_the_real_js_on_every_ground_truth_case():
    """The corpus was generated by RUNNING `stripSpeechAnnotations`
    (`src/shared/voice/liveProtocol.ts:1429`), so this is agreement with the
    shipped app, not agreement with a second reading of its source.

    A true-prefix check between the relay's epoch text and the client's receipt
    is meaningless unless both sides normalize identically; a divergence here
    does not fail loudly, it silently demotes ordinary sentences to the absent
    policy and the thread quietly goes back to saving words nobody heard.
    """

    with open(CANON_FIXTURES, encoding="utf-8") as handle:
        cases = json.load(handle)

    assert len(cases) >= 51, (
        "the corpus SHRANK — it may grow, but a case may never be removed: "
        "each one pins a divergence the two implementations actually had"
    )
    wrong = [
        (case["raw"], case["expected"], canonical_spoken_text(case["raw"]))
        for case in cases
        if canonical_spoken_text(case["raw"]) != case["expected"]
    ]
    assert wrong == []
    # Idempotent over the same corpus, and the app's guard
    # (`scripts/check-live-voice-protocol.js`) asserts exactly this too.  It has
    # to hold because the two sides do not apply it the same number of times:
    # the relay canonicalizes ONCE over the concatenated epoch, the client
    # canonicalizes per delta and again over the joined epoch.  A second pass
    # that changed the string would make those two arrive at different text and
    # turn every cut reply into `not_prefix`.
    not_stable = [
        case["expected"] for case in cases
        if canonical_spoken_text(case["expected"]) != case["expected"]
    ]
    assert not_stable == []


def test_the_three_places_python_re_and_js_regexp_disagree():
    """Each of these returned a DIFFERENT string before it was handled, and
    each one is silent: the port still runs, still returns a plausible
    sentence, and the prefix check just stops matching.

    * JS `\\s` contains U+FEFF and not U+0085/U+001C; Python's is the opposite.
    * A JS regex without `/u` bounds `{1,40}` in UTF-16 CODE UNITS, so 21-40
      astral characters inside brackets are prose to the app and a tag to a
      naive Python port.
    * `/gi` — every occurrence, case-insensitively.
    """

    # JS trims U+FEFF; `str.strip()` does not.
    assert canonical_spoken_text("﻿Hello﻿") == "Hello"
    # JS does NOT trim U+0085 or U+001C; `str.strip()` does.
    assert canonical_spoken_text("Hello") == "Hello"
    # 25 astral characters: 25 code points, 50 UTF-16 units — past the bound,
    # so the app leaves it and so must the relay.
    long_tag = "[" + "\U0001f600" * 25 + "] tail"
    assert canonical_spoken_text(long_tag) == long_tag
    # 15 astral characters: 30 units, inside the bound, and a tag on both sides.
    assert canonical_spoken_text("[" + "\U0001f600" * 15 + "] tail") == "tail"
    # Every occurrence, either case.
    assert canonical_spoken_text("a (LAUGHS) b (Noise) c") == "a b c"


def test_an_epoch_is_canonical_from_birth():
    """`ClosedEpoch.text` is what every later consumer compares against — the
    segment frame, the persisted row and the receipt's prefix check — so the
    normalization happens once, at retirement, instead of each consumer
    re-deciding what an annotation is."""

    adapter = LiveClientAdapter("canon-at-retire")
    adapter.provider_event(H.out_text("[clears throat] … Hello there", 0, 400), now=0.0)
    _frames, closed = adapter.close_for_new_output("test")

    assert closed is not None
    assert closed.text == "Hello there"


def test_an_annotation_only_epoch_records_nothing():
    """"[inaudible]" is not a sentence the caller heard; it is the provider
    saying it heard nothing.  The old `.strip()` persisted it verbatim."""

    adapter = LiveClientAdapter("canon-annotation-only")
    adapter.provider_event(H.out_text("[inaudible]", 0, 400), now=0.0)
    _frames, closed = adapter.close_for_new_output("test")

    assert closed is not None
    assert closed.text == ""


# ── played_ms: a position, not whatever JSON happened to carry ────────

def test_played_ms_rejects_bool_and_non_finite_and_clamps():
    """`isinstance(v, (int, float))` accepted three things it should not.

    `True` IS an int in Python, so a client bug that sent `played_ms: true`
    recorded one millisecond of playback — a plausible-looking number, which is
    the worst kind of wrong on a row that exists to say what was heard.  NaN
    and Infinity are worse than wrong: `json.loads` accepts both by default and
    `int()` RAISES on them, so one malformed frame took down the whole client
    loop instead of one field.
    """

    assert _played_ms_value(True) is None
    assert _played_ms_value(False) is None
    assert _played_ms_value(float("nan")) is None
    assert _played_ms_value(float("inf")) is None
    assert _played_ms_value(float("-inf")) is None
    assert _played_ms_value("900") is None
    assert _played_ms_value(None) is None
    assert _played_ms_value(-5) == 0
    assert _played_ms_value(1840) == 1840
    assert _played_ms_value(1840.7) == 1840
    assert _played_ms_value(10 ** 12) == 3_600_000


# ── the five-step validation order ────────────────────────────────────

def test_an_absent_receipt_is_todays_behaviour_and_is_not_an_error(monkeypatch):
    """ABSENT is a legitimate answer, not a fault: this client cannot say what
    it displayed, so the canonical epoch text stands exactly as it does now."""

    counters = H.capture_counters(monkeypatch)
    session = _session()
    closed = _epoch()

    assert session.accept_heard_text(_frame(), closed) is None
    assert _reasons(counters) == []


def test_an_empty_receipt_is_a_real_answer_and_not_an_absent_one():
    """`''` and absent are different facts and the record split depends on
    telling them apart: `''` is "the caller heard nothing", absent is "this
    client did not say"."""

    session = _session()
    assert session.accept_heard_text(_frame(heard_text=""), _epoch()) == ""
    assert session.accept_heard_text(_frame(heard_text="   "), _epoch()) == ""


def test_a_lone_surrogate_is_refused_before_it_can_reach_the_queue(monkeypatch):
    """Load-bearing, and the reason the check exists at all: `json.loads`
    happily produces a lone surrogate from "\\ud83d", and the first thing that
    touches it afterwards — `.encode('utf-8')` on the way to the tenant API —
    raises.  Refused here, where the reason is still knowable, instead of three
    awaits later inside the persistence worker."""

    counters = H.capture_counters(monkeypatch)
    session = _session()

    assert session.accept_heard_text(
        _frame(heard_text="First \ud83d sentence."), _epoch(),
    ) is None
    assert session.accept_heard_text(
        _frame(heard_text="First\x00sentence."), _epoch(),
    ) is None
    assert session.accept_heard_text(_frame(heard_text=1840), _epoch()) is None
    assert _reasons(counters) == ["malformed", "malformed", "malformed"]

    # And the refusal is total: the row keeps the full canonical text.
    closed = _epoch()
    session.note_heard_text(_frame(heard_text="First \ud83d sentence."), closed)
    assert closed.heard_text is None


@pytest.mark.asyncio
async def test_the_receipt_is_bounded_in_utf16_units_on_a_code_point_boundary():
    """Ten thousand UTF-16 units — the unit the SENDER counts in — and the cut
    lands on a code point boundary.

    Counting with `len(s.encode('utf-16-le')) // 2` would raise on exactly the
    lone-surrogate input the step above exists to catch, and slicing on UTF-16
    units would manufacture one: half of an emoji or a Persian character is the
    malformation, not a shorter sentence.
    """

    session = _session()
    # 6000 astral characters = 12000 UTF-16 units, all of them one epoch's text.
    huge = "\U0001f600" * 6000
    closed = _epoch(text=huge, interrupted=True)
    accepted = session.accept_heard_text(_frame(heard_text=huge), closed)

    assert accepted is not None
    assert utf16_units(accepted) == 10_000
    assert len(accepted) == 5_000, "a whole number of emoji, never half of one"
    assert accepted == huge[:5000]
    # It is still a true prefix of the epoch, so it is still accepted.
    assert huge.startswith(accepted)


def test_a_forged_or_stale_identity_buys_nothing(monkeypatch):
    """Step 4.  A receipt that names another epoch is either a replay or a
    forgery, and either way it is evidence about a string this epoch never
    said.  It is a strict no-op on the text — the row keeps the full answer —
    not a reason to write something shorter."""

    counters = H.capture_counters(monkeypatch)
    session = _session()
    closed = _epoch(epoch=2)

    # A receipt for the PREVIOUS epoch, arriving late.
    assert session.accept_heard_text(
        _frame(epoch=1, heard_text="First"), closed,
    ) is None
    # A pair that disagrees names no epoch at all (§4.10).
    assert session.accept_heard_text(
        {"type": "interrupt", "response_id": _oid(2), "item_id": _oid(1),
         "heard_text": "First"},
        closed,
    ) is None
    # The no-identity interrupt frame: nothing to check the receipt against.
    assert session.accept_heard_text(
        {"type": "interrupt", "heard_text": "First"}, closed,
    ) is None
    # An outright forgery.
    assert session.accept_heard_text(
        _frame(epoch=2, heard_text="First", assistant_turn_id="live:other:9"),
        closed,
    ) is None
    assert _reasons(counters) == ["identity_mismatch"] * 4


def test_a_receipt_that_is_not_a_prefix_is_refused(monkeypatch):
    """Step 5.  The receipt is what this client DISPLAYED, and the relay knows
    what it sent; a string that is not a prefix of it is not a shorter version
    of this answer, so the full canonical text stands."""

    counters = H.capture_counters(monkeypatch)
    session = _session()
    closed = _epoch(text="First sentence. Second sentence.")

    assert session.accept_heard_text(
        _frame(heard_text="Second sentence."), closed,
    ) is None
    assert session.accept_heard_text(
        _frame(heard_text="First sentence. Second sentence. Third."), closed,
    ) is None
    assert _reasons(counters) == ["not_prefix", "not_prefix"]
    # The identical text is a prefix of itself and is accepted.
    assert session.accept_heard_text(
        _frame(heard_text="First sentence. Second sentence."), closed,
    ) == "First sentence. Second sentence."


def test_a_late_receipt_for_an_unresolvable_epoch_is_not_a_client_fault(monkeypatch):
    """LATE: the epoch is gone, the row it wrote stands, and nothing is
    counted against the client — it did everything right and arrived after the
    relay stopped being able to check."""

    counters = H.capture_counters(monkeypatch)
    session = _session()

    assert session.accept_heard_text(_frame(heard_text="First"), None) is None
    assert _reasons(counters) == []


def test_persian_and_astral_text_survives_the_round_trip():
    """The receipt is bytes-exact, and Persian is where every naive bound
    (UTF-16 slicing, `\\w`, casefolding) breaks first."""

    session = _session()
    full = "سلام. حالت چطوره؟ من خوبم 😀"
    closed = _epoch(text=full)

    assert session.accept_heard_text(_frame(heard_text="سلام."), closed) == "سلام."
    assert session.accept_heard_text(
        _frame(heard_text="سلام. حالت چطوره؟ من خوبم 😀"), closed,
    ) == full
    # ZWNJ is part of the word, not whitespace: a receipt that dropped it is
    # not a prefix of what was said.
    joined = "می‌گردم دنبالش"
    assert session.accept_heard_text(
        _frame(heard_text="می‌گردم"), _epoch(text=joined),
    ) == "می‌گردم"
    assert session.accept_heard_text(
        _frame(heard_text="میگردم"), _epoch(text=joined),
    ) is None


# ── D1: the heard prefix goes in the row that already exists ──────────

@pytest.mark.asyncio
async def test_a_cut_off_reply_is_saved_as_what_was_heard():
    """The headline defect.  One row, revised in place, holding the words the
    caller actually got — not a second bubble, because that row is the only
    holder of this text and nothing is lost by shortening it."""

    session = _session()
    closed = _epoch(heard="First sentence.")
    await session.persist_spoken(closed)

    payload = session.persistence.payload("live-output:")
    assert payload["assistant_text"] == "First sentence."
    voice = payload["assistant_voice"]
    assert voice["record_kind"] == "assistant_transcript"
    assert voice["version"] == 1
    assert voice["heard_chars"] == len("First sentence.")
    assert voice["interrupted"] is True
    assert voice["played_ms"] == 900
    # The full epoch is still countable, so "how much was cut" stays answerable.
    assert voice["generated_chars"] == len("First sentence. Second sentence.")


@pytest.mark.asyncio
async def test_hearing_nothing_writes_no_row_rather_than_an_empty_one():
    """`_save_voice_messages` declines an empty row and the receiving route
    400s on empty content, so an empty prefix means "write no transcript text".
    A record that asks for no row is counted LOST by the queue — writing one
    would be a logged persistence failure on every barge-in."""

    session = _session()
    await session.persist_spoken(_epoch(heard="", played_ms=0))

    assert session.persistence.records == []


@pytest.mark.asyncio
async def test_an_existing_row_that_nobody_heard_is_retracted_not_reworded():
    """A row already in the thread, contradicted by a validated empty receipt.

    RE-AIMED, and the re-aim is the whole point.  This test used to stop at
    "the text is not corrected downward", and that half-truth WAS the defect:
    the row went on serving a full answer the caller never heard, with
    `heard_chars: 0` sitting quietly beside it as the only hint, so the live
    surface showed nothing while the saved chat showed everything.

    The text still cannot be corrected downward — `_save_voice_messages`
    declines an empty row and the receiving route 400s on empty content, and a
    sentence someone may already have read is not unsaid by deleting it. So the
    row is RETRACTED instead: it keeps its bytes and stops being served as
    something that was heard. `schemas.heard_nothing` is the one predicate every
    read surface funnels through, and the assertion below is on that function
    rather than on the flag, because the flag only matters if the projection
    fires.

    `park_spoken` is what keeps this the RARE path: with the receipt window
    open, the ordinary gap-then-receipt ordering never writes the row at all.
    """

    from app.schemas import heard_nothing, public_heard_text

    session = _session()
    await session.persist_spoken(_epoch(interrupted=False, played_ms=4200))
    first = dict(session.persistence.payload("live-output:"))
    assert first["assistant_text"] == "First sentence. Second sentence."
    assert heard_nothing(first["assistant_voice"]) is False

    await session.persist_spoken(_epoch(heard="", played_ms=0))
    second = session.persistence.payload("live-output:")

    assert second["assistant_text"] == first["assistant_text"]
    assert second["assistant_revision"] == first["assistant_revision"] + 1
    voice = second["assistant_voice"]
    assert voice["heard_chars"] == 0
    assert voice["interrupted"] is True
    assert voice["transcript_retracted"] is True
    assert voice["record_kind"] == "assistant_transcript"
    # READ: every serializer composes `public_heard_text` over the row body, so
    # this is what a client is handed for that row.
    assert heard_nothing(voice) is True
    assert public_heard_text(second["assistant_text"], voice) == ""


@pytest.mark.asyncio
async def test_a_retraction_never_reaches_a_task_result_row():
    """The backend answer is not a playback claim.  A delegated turn the caller
    heard none of still has its result in the thread, in full — the epoch's
    playback numbers were moved OFF that row for exactly this reason, and a
    projection that fired on it would delete answers the user asked for."""

    from app.schemas import heard_nothing, public_heard_text

    session = _session()
    _delegated(session)
    await session.persist_spoken(
        _epoch(text="Here is what I found.", heard="", played_ms=0),
    )

    result = session.persistence.payload("live-delegation:")
    voice = result["assistant_voice"]
    assert voice["record_kind"] == "task_result"
    assert voice["heard_chars"] == 0
    assert heard_nothing(voice) is False, (
        "`heard_nothing` keys on `record_kind`, and a task result is never "
        "projected however little of its paraphrase was heard"
    )
    assert public_heard_text(result["assistant_text"], voice) == (
        "The deadline is the first of March."
    )
    # …and no transcript row was invented for words nobody heard.
    assert not session.persistence.has("live-transcript:")


def test_the_tenant_allowlist_stores_the_retraction_flag():
    """`sessions._clean_voice` drops every key it does not know. A retraction
    dropped there is a row that goes on serving unheard words to every client
    for ever — the flag is the only thing standing between the saved text and
    the reader, so it has to survive the allowlist."""

    from app.api.sessions import _VOICE_KEYS, _clean_voice

    assert "transcript_retracted" in _VOICE_KEYS
    sent = {
        "source": "live_spoken", "version": 1,
        "record_kind": "assistant_transcript", "heard_chars": 0,
        "interrupted": True, "transcript_retracted": True,
    }
    assert _clean_voice(dict(sent)) == sent


# ── the provisional row: the row that is never written at all ─────────
#
# Retraction above is the REPAIR for a row that already exists.  This section
# is the reason that path stays rare.  The output-gap timer is the relay's own
# idle clock — it fires ~700 ms after the provider stops emitting, while
# seconds of buffered audio are still playing — so it was always going to
# decide "what did the caller hear" before the caller's phone had said a word
# about it.  Holding the row for the length of the receipt window turns the
# common ordering into "nothing unheard was ever saved" instead of "something
# unheard was saved and then corrected".


def _gap(**extra):
    return _epoch(interrupted=False, heard=None, retire_reason="gap", **extra)


@pytest.mark.asyncio
async def test_a_gap_retired_epoch_holds_its_row_until_its_receipt_window_closes():
    """The relay's own idle timer is evidence about the RELAY. It says the
    provider stopped emitting; it says nothing about what reached the caller,
    and an ack that would say is still in flight."""

    session = _session(park=True)
    closed = _gap()
    await session.persist_spoken(closed)

    assert session.persistence.records == []
    assert list(session.parked_spoken) == [closed.epoch]


@pytest.mark.asyncio
async def test_the_window_closing_with_a_receipt_of_nothing_writes_no_row_at_all():
    """The headline of the nothing-heard rule, in its preferred form.

    The caller heard none of this epoch, so no row is written, so there is
    nothing for a reader to project and nothing for the app to hide. Contrast
    `test_an_existing_row_that_nobody_heard_is_retracted_not_reworded`: same
    fact, arriving one step too late, and a strictly worse outcome — bytes in
    the thread that every layer then has to agree to stop serving.
    """

    session = _session(park=True)
    closed = _gap()
    await session.persist_spoken(closed)

    session.note_heard_text(_frame(heard_text=""), closed)
    closed.interrupted = True
    await session.persist_spoken(closed)

    assert session.persistence.records == [], (
        "the epoch nobody heard reached the thread anyway"
    )
    assert session.parked_spoken == {}
    assert session.spoken_rows == {}


@pytest.mark.asyncio
async def test_the_window_closing_with_a_partial_receipt_writes_only_what_was_heard():
    """The same window, the ordinary outcome: a receipt arrives, and the row
    the thread gets is the one that was right the first time — no revision, no
    intermediate version of the answer that was never true."""

    session = _session(park=True)
    closed = _gap()
    await session.persist_spoken(closed)

    session.note_heard_text(_frame(heard_text="First sentence."), closed)
    await session.persist_spoken(closed)

    payload = session.persistence.payload("live-output:")
    assert payload["assistant_text"] == "First sentence."
    assert payload["assistant_revision"] == 1, (
        "one write, not a correction: nothing unheard was ever published"
    )
    assert "transcript_retracted" not in payload["assistant_voice"]


@pytest.mark.asyncio
async def test_the_window_ends_on_its_deadline_when_no_receipt_ever_comes():
    """A client that negotiated the receipt and then said nothing — it crashed,
    the socket stalled, the build is lying about what it supports. The wait is
    bounded: the row is written with the canonical epoch text, which is exactly
    what the relay would have written before any of this existed."""

    session = _session(park=True)
    session.spoken_settle_s = lambda: 2.5
    closed = _gap()
    await session.persist_spoken(closed)

    # Not yet — the window is still open.
    await session.flush_parked_spoken(now=time.monotonic() + 1.0)
    assert session.persistence.records == []

    await session.flush_parked_spoken(now=time.monotonic() + 3.0)
    payload = session.persistence.payload("live-output:")
    assert payload["assistant_text"] == "First sentence. Second sentence."
    assert closed.spoken_park_released is True
    assert session.parked_spoken == {}


@pytest.mark.asyncio
async def test_teardown_closes_a_window_that_is_still_open():
    """The caller hung up mid-window. A receipt that has not arrived is never
    arriving, and leaving the row parked would lose the assistant side of the
    last reply entirely — a strictly worse failure than saving text whose
    playback the relay cannot confirm."""

    session = _session(park=True)
    closed = _gap()
    await session.persist_spoken(closed)
    assert session.persistence.records == []

    await session.flush_parked_spoken(force=True)

    assert session.persistence.payload("live-output:")["assistant_text"] == (
        "First sentence. Second sentence."
    )


@pytest.mark.asyncio
async def test_a_deadline_cannot_resurrect_a_row_a_receipt_already_settled():
    """The ordering guard, and it is the one that would have hurt. The window
    closes on the FIRST of three things, so the other two must become no-ops —
    a deadline still standing in `tick` after an empty receipt had correctly
    suppressed the row would put the whole generated answer back, which is the
    defect this section exists to remove, arriving by a different door."""

    session = _session(park=True)
    closed = _gap()
    await session.persist_spoken(closed)
    session.note_heard_text(_frame(heard_text=""), closed)
    closed.interrupted = True
    await session.persist_spoken(closed)
    assert session.persistence.records == []

    await session.flush_parked_spoken(now=time.monotonic() + 30.0)
    await session.flush_parked_spoken(force=True)

    assert session.persistence.records == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason", ["interrupt", "playback_idle", "playback_failed", "new_output"],
)
async def test_only_the_relays_own_idle_timer_waits(reason):
    """`gap` is the only retirement that is a guess. Every other one is either a
    client frame that carries (or pointedly omits) a receipt of its own, or the
    relay closing an epoch because a new reply is starting — after which no
    receipt for it is owed and waiting would delay the thread for nothing."""

    session = _session(park=True)
    await session.persist_spoken(_epoch(retire_reason=reason, heard=None))

    assert session.parked_spoken == {}
    assert session.persistence.payload("live-output:")["assistant_text"] == (
        "First sentence. Second sentence."
    )


@pytest.mark.asyncio
async def test_a_client_that_cannot_send_a_receipt_never_waits():
    """Degradation, same rule as everywhere else in this file. A build that
    does not negotiate `heard_text` is never going to close this window, so
    holding its rows for the settle would be a straight latency regression paid
    for a fact it cannot produce."""

    session = _session(features=("live_turns", "playback_frames"), park=True)
    await session.persist_spoken(_gap())

    assert session.parked_spoken == {}
    assert session.persistence.payload("live-output:")["assistant_text"] == (
        "First sentence. Second sentence."
    )


@pytest.mark.asyncio
async def test_a_session_with_nothing_to_flush_it_writes_immediately():
    """`_LiveSession.parked_spoken` is `None` by class default, and that is the
    honest answer for a session built by `__new__`: no `tick_loop`, no
    teardown, nothing that could ever close the window. Parking there would not
    delay a row, it would delete it."""

    from app.services.live_voice_protocol import _LiveSession

    assert _LiveSession.parked_spoken is None
    session = _session()          # park=False — the class default stands
    await session.persist_spoken(_gap())

    assert session.persistence.payload("live-output:")["assistant_text"] == (
        "First sentence. Second sentence."
    )


def test_the_receipt_window_fits_inside_the_adapters_epoch_history():
    """Bounded on both sides, and the upper bound is not a preference.

    `LiveClientAdapter._retire` keeps `self._history[-4:]` — four epochs, which
    its own comment puts at roughly ten seconds of replies at conversational
    pace. A window longer than that would wait for a receipt naming an epoch
    the adapter can no longer resolve, so it would be buying a fact it could
    never act on while holding the row hostage for it.
    """

    session = _session()
    assert 0 < session.spoken_settle_s() <= 4.0
    # The window is a module constant, NOT `getattr(settings, …, 2500)`, and
    # that is deliberate: `voice_live_spoken_settle_ms` is not a field on
    # `app/config.py`'s `Settings` (that file belongs to another owner this
    # round), and
    # `test_live_voice_protocol.test_every_settings_name_the_relay_reads_actually_exists`
    # exists precisely because a settings name the relay reads and `Settings`
    # never declares is silently the literal default for ever. Pinned here so
    # the value is a decision rather than an accident, and so the eventual swap
    # to the real setting has to come past this assertion.
    assert session.spoken_settle_s() == 2.5


@pytest.mark.asyncio
async def test_a_receipt_this_relay_refused_leaves_the_full_answer_on_the_row():
    """Every INVALID verdict ends in the same place, which is the point of
    having one: the row keeps the canonical epoch text, exactly as today."""

    session = _session()
    closed = _epoch()
    session.note_heard_text(_frame(epoch=7, heard_text="First"), closed)
    await session.persist_spoken(closed)

    payload = session.persistence.payload("live-output:")
    assert payload["assistant_text"] == "First sentence. Second sentence."
    assert "heard_chars" not in payload["assistant_voice"]


# ── D2: the task result is never truncated ────────────────────────────

def _delegated(session, *, answer="The deadline is the first of March.", task_id="d1"):
    ref = f"live-delegation:{H.PSID}:{task_id}"
    payload = {
        "user_text": "",
        "assistant_text": answer,
        "assistant_ref": ref,
        "assistant_voice": {
            "source": "delegated",
            "delegation_id": task_id,
            "task_id": task_id,
            "parent_user_turn_id": f"live-utt:{H.PSID}:1",
            "clock": "provider",
        },
    }
    claim = session.build_commentary_claim(ref, payload, delegation_id=task_id)
    session.note_commentary_epoch_claim(claim)
    session.adapter.provider_event(H.out_text("Here is what I found.", 900, 1500))
    return claim


@pytest.mark.asyncio
async def test_the_delegated_row_keeps_the_whole_answer_and_loses_the_playback_numbers():
    """D2.  The `live-delegation:` row is the TASK RESULT — the complete
    backend answer, verbatim.  The playback numbers describe the spoken
    paraphrase, which is a different and shorter string, so a row carrying both
    was a self-contradicting record.  They move to the paraphrase's own row."""

    session = _session()
    _delegated(session)
    closed = _epoch(text="Here is what I found.", heard="Here is", played_ms=400)
    await session.persist_spoken(closed)

    result = session.persistence.payload("live-delegation:")
    assert result["assistant_text"] == "The deadline is the first of March."
    voice = result["assistant_voice"]
    assert voice["record_kind"] == "task_result"
    assert voice["version"] == 1
    # PIN FLIPPED (voice round 2026-10-06, DESIGN-R2 D-R8 (a)): `epoch` is
    # back on the task row — the paraphrase now FOLDS onto it, so the row
    # names the epoch that voiced it.  The four playback numbers stay off.
    for playback_key in ("played_ms", "interrupted", "start_ms", "end_ms"):
        assert playback_key not in voice, playback_key
    assert voice["epoch"] == closed.epoch
    # `heard_chars` deliberately stays, and it is the one field on this row
    # that is about the PARAPHRASE: the frozen spec adds it to every voice
    # payload and its removal list names the five playback numbers and not
    # this. Pinned so the choice is visible rather than incidental.
    assert voice["heard_chars"] == len("Here is")
    # A barge-in cut (no hangup reason): the caller cut it.
    assert voice["spoken_interrupted"] is True

    # PIN FLIPPED (D-R8 (a), "fold always"): no `live-transcript:` sibling —
    # one row per answer.
    assert not session.persistence.has("live-transcript:")


@pytest.mark.asyncio
async def test_a_delivery_cut_before_its_first_text_still_records_the_cut_on_the_task_row():
    """Phase D (correctness-relay-8): `persist_spoken` returned on an empty
    epoch BEFORE looking up the claim, so a delivered result the caller cut
    within its first audio — no output transcript yet — never reached
    `fold_onto_delegated_row`: the task row kept `spoken` with no
    `spoken_interrupted` and no `epoch`, and the app drew no «قطع شد»."""

    session = _session()
    _delegated(session)
    await session.persist_spoken(_epoch(text="", heard="", played_ms=120))

    result = session.persistence.payload("live-delegation:")
    assert result["assistant_text"] == "The deadline is the first of March."
    assert result["assistant_voice"]["spoken_interrupted"] is True
    assert result["assistant_voice"]["epoch"] == 1
    assert not session.persistence.has("live-output:")
    assert not session.persistence.has("live-transcript:")


@pytest.mark.asyncio
async def test_an_empty_unclaimed_epoch_still_writes_no_row():
    session = _session()
    await session.persist_spoken(_epoch(text="", heard=""))
    assert not session.persistence.has("live-output:")
    assert not session.persistence.has("live-delegation:")


@pytest.mark.asyncio
async def test_a_paraphrase_that_repeats_the_task_result_gets_no_second_row():
    """No extra bubble unless there is something extra to say.  When the words
    the caller heard ARE the answer the task returned, the result row already
    tells the whole truth and a sibling would be the same answer twice.

    RE-AIMED.  This test used to assert the same thing for a paraphrase whose
    content was completely different from the task result — it spoke "Here is
    what I found." for an answer about a deadline — and passed only because
    `spoken_paraphrase_differs` never compared the two strings at all.  That
    was the defect, not the rule: the rule is about DUPLICATION, so the case
    that pins it has to actually be a duplicate.  The reworded case is the test
    directly below.
    """

    answer = "The deadline is the first of March."
    session = _session()
    _delegated(session, answer=answer)
    closed = _epoch(
        text=answer, heard=answer,
        interrupted=False, played_ms=1500, retire_reason="playback_idle",
    )
    await session.persist_spoken(closed)

    assert session.persistence.has("live-delegation:")
    assert not session.persistence.has("live-transcript:")


@pytest.mark.asyncio
async def test_a_fully_played_paraphrase_that_reworded_the_answer_folds_onto_the_result():
    """The fourth way a paraphrase differs, and the one nothing checked.

    GPT-Live is handed the backend's answer and says it back in its OWN words.
    Nothing was interrupted, the receipt covers the whole epoch and playback
    succeeded — so all three of the old conditions are false, and the sentence
    the caller actually heard was represented nowhere in the thread.  The row
    the reader saw was the backend's wording, presented as the spoken turn.
    """

    session = _session()
    _delegated(session, answer="The deadline is the first of March.")
    spoken = "It's due on March the first."
    closed = _epoch(
        text=spoken, heard=spoken,
        interrupted=False, played_ms=1500, retire_reason="playback_idle",
    )
    await session.persist_spoken(closed)

    result = session.persistence.payload("live-delegation:")
    assert result["assistant_text"] == "The deadline is the first of March.", (
        "the task result is never projected, clipped or reworded"
    )
    assert result["assistant_voice"]["record_kind"] == "task_result"
    # PIN FLIPPED (voice round 2026-10-06, DESIGN-R2 D-R8 (a), a recorded
    # deviation from the brief's "unless I cut it off"): the reworded
    # paraphrase used to keep its own `live-transcript:` row — the UFC double
    # of the 2026-10-06 call.  It folds now: one row per answer.
    assert not session.persistence.has("live-transcript:")
    assert result["assistant_voice"]["spoken_interrupted"] is False
    assert result["assistant_voice"]["heard_chars"] == len(spoken)


@pytest.mark.asyncio
async def test_an_annotation_is_not_a_difference_the_caller_could_hear():
    """The two strings come from opposite sides of the annotation rule — the
    epoch text is canonical from birth, the task result is the backend's raw
    answer — so the comparison is made on canonical text.  A provider tag the
    app strips before it draws anything is not a second version of the answer,
    and writing a sibling row for it would put the same sentence in the thread
    twice over a bracket run nobody heard."""

    session = _session()
    _delegated(session, answer="  Ready on   Friday.  ")
    closed = _epoch(
        text="[clears throat] Ready on Friday.", heard=None,
        interrupted=False, played_ms=1500, retire_reason="playback_idle",
    )
    # Canonical from birth is the relay's invariant, so state what this epoch
    # would really have closed with rather than asserting on a tag `_retire`
    # would already have removed.
    closed.text = canonical_spoken_text(closed.text)
    assert closed.text == "Ready on Friday."
    await session.persist_spoken(closed)

    assert session.persistence.has("live-delegation:")
    assert not session.persistence.has("live-transcript:")


@pytest.mark.asyncio
async def test_a_place_holder_claim_has_no_answer_to_be_a_duplicate_of():
    """`speak()` files a claim with no `ref` and an empty payload for a
    progress, cancel or failure line, purely so the FIFO keeps its order.  The
    content comparison must not read an absent task result as "everything
    differs" and start writing a sibling row for every progress line — there is
    no second row to duplicate into and nothing was delegated."""

    from app.services.live_voice_protocol import ClosedEpoch

    session = _session()
    closed = ClosedEpoch(
        epoch=1, output_id=_oid(1), text="Still working on it.",
        interrupted=False, played_ms=900, retire_reason="playback_idle",
    )
    place_holder = session.build_commentary_claim("", {}, delegation_id="d1")

    assert session.spoken_paraphrase_differs(closed, place_holder) is False
    assert session.spoken_paraphrase_differs(closed, None) is False


@pytest.mark.asyncio
async def test_a_paraphrase_nobody_heard_at_all_gets_no_row_either():
    """Materially different, yes — but there is no text to write, and a record
    that names no row is counted lost.  The result row still holds the
    answer."""

    session = _session()
    _delegated(session)
    await session.persist_spoken(
        _epoch(text="Here is what I found.", heard="", played_ms=0),
    )

    assert session.persistence.has("live-delegation:")
    assert not session.persistence.has("live-transcript:")


@pytest.mark.asyncio
async def test_a_paraphrase_the_phone_could_not_play_folds_and_is_not_a_cut():
    """Playback failing is the third way a paraphrase materially differs from
    a complete delivery, and the only one with no receipt to prove it.

    PIN FLIPPED (voice round 2026-10-06, DESIGN-R2 D-R8 (a)): it used to get
    its own `live-transcript:` row.  It folds now, and a playback failure is
    NOT the caller cutting the answer off (`spoken_interrupted` is false)."""

    session = _session()
    _delegated(session)
    await session.persist_spoken(_epoch(
        text="Here is what I found.", played_ms=0, retire_reason="playback_failed",
    ))

    assert not session.persistence.has("live-transcript:")
    voice = session.persistence.payload("live-delegation:")["assistant_voice"]
    assert voice["spoken_interrupted"] is False
    assert "heard_chars" not in voice


@pytest.mark.asyncio
async def test_a_hangup_that_cut_the_answer_is_not_the_callers_cut():
    """R8 (D-R8 (a)): the hangup sends `[interrupt, stop]` and its interrupt
    carries `reason: client_request` — the 2026-10-06 Miami answer was saved
    `interrupted: true` because the caller pressed End.  Only a barge-in is
    `spoken_interrupted`."""

    session = _session()
    _delegated(session)
    closed = _epoch(text="Here is what I found.", heard="Here is", played_ms=400)
    closed.cut_by = "session_end"
    await session.persist_spoken(closed)

    voice = session.persistence.payload("live-delegation:")["assistant_voice"]
    assert voice["spoken_interrupted"] is False
    assert voice["heard_chars"] == len("Here is")
    assert not session.persistence.has("live-transcript:")


# ── negotiation: an old client gets exactly today's rows ──────────────

@pytest.mark.asyncio
async def test_a_client_that_never_negotiates_heard_text_gets_todays_behaviour(monkeypatch):
    """The gate covers BOTH halves deliberately.  A build that cannot tell the
    relay what it displayed must not have a shorter answer written into its day
    chat on the word of a field it does not implement — and its delegated row
    must keep the playback numbers it has always carried."""

    counters = H.capture_counters(monkeypatch)
    session = _session(features=("live_turns", "playback_frames"))

    closed = _epoch()
    session.note_heard_text(_frame(heard_text="First sentence."), closed)
    assert closed.heard_text is None
    await session.persist_spoken(closed)

    payload = session.persistence.payload("live-output:")
    assert payload["assistant_text"] == "First sentence. Second sentence."
    voice = payload["assistant_voice"]
    assert voice["played_ms"] == 900
    assert voice["interrupted"] is True
    for added in ("version", "record_kind", "heard_chars"):
        assert added not in voice, added
    # An unnegotiated field is noise, not an error.
    assert _reasons(counters) == []

    folded = _session(features=("live_turns",))
    _delegated(folded)
    await folded.persist_spoken(_epoch(text="Here is what I found.", played_ms=400))
    result = folded.persistence.payload("live-delegation:")
    assert result["assistant_text"] == "The deadline is the first of March."
    assert result["assistant_voice"]["played_ms"] == 400
    assert result["assistant_voice"]["interrupted"] is True
    assert result["assistant_voice"]["epoch"] == 1
    assert result["assistant_voice"]["spoken_interrupted"] is True
    assert "record_kind" not in result["assistant_voice"]
    assert not folded.persistence.has("live-transcript:")


# ── the shape of what is written ──────────────────────────────────────

@pytest.mark.asyncio
async def test_no_new_top_level_payload_key_is_ever_introduced():
    """`PersistenceQueue._write` splats the payload into
    `_save_voice_messages`, which takes named parameters and NO `**kwargs`.  A
    new top-level key is therefore a `TypeError` — caught by the queue, retried
    twice, and lost as a row nobody notices.  Everything this repair adds lives
    inside `assistant_voice`."""

    from app.api import ws_realtime as rt

    allowed = set(inspect.signature(rt._save_voice_messages).parameters) - {
        "user_id", "session_id",
    }

    session = _session()
    _delegated(session)
    await session.persist_spoken(
        _epoch(text="Here is what I found.", heard="Here is", played_ms=400),
    )
    await session.persist_spoken(_epoch(epoch=2, heard="First sentence."))

    # PIN FLIPPED (voice round 2026-10-06, DESIGN-R2 D-R8 (a)): the folded
    # paraphrase writes no `live-transcript:` sibling, so 2 records, not 3
    # (the task row, then epoch 2's own row).
    assert len(session.persistence.records) == 2
    for record in session.persistence.records:
        assert set(record.payload) <= allowed, (record.key, set(record.payload) - allowed)


def test_the_tenant_allowlist_stores_the_record_identity():
    """`sessions._clean_voice` drops every key it does not know, so a relay
    sending `record_kind` without it there would believe it wrote a field
    nobody stored — and a reader could not tell the two record kinds apart."""

    from app.api.sessions import _VOICE_KEYS, _clean_voice

    for key in ("version", "record_kind", "heard_chars"):
        assert key in _VOICE_KEYS, key
    cleaned = _clean_voice({
        "source": "live_spoken", "version": 1,
        "record_kind": "assistant_transcript", "heard_chars": 15,
    })
    assert cleaned == {
        "source": "live_spoken", "version": 1,
        "record_kind": "assistant_transcript", "heard_chars": 15,
    }


# ── the adapter carries the receipt for both current and retired epochs ──

def test_a_receipt_reaches_an_epoch_the_gap_timer_already_retired():
    """DELAYED OUTPUT, and it is the normal case on a phone: the 700 ms output
    gap retires an epoch while seconds of buffered audio are still playing, so
    by the time the caller cuts it the epoch lives in `_history`.  The receipt
    has to land there too or it only ever works for short replies."""

    adapter = LiveClientAdapter("late-receipt", gap_ms=10)
    adapter.provider_event(H.out_text("First sentence. Second sentence.", 0, 400), now=0.0)
    _frames, retired = adapter.rotate_on_gap(now=5.0)
    assert retired is not None and adapter.output_id is None

    ok, closed = adapter.interrupt(
        retired.output_id, retired.output_id, 900,
        heard_text="First sentence.", now=6.0,
    )

    assert ok and closed is retired
    assert closed.heard_text == "First sentence."
    assert closed.heard_source == "interrupt"
    assert closed.interrupted is True


def test_a_later_frame_without_a_receipt_does_not_erase_the_one_we_have():
    """Absent means "this client did not say", never "nothing was heard".  A
    `playback_idle` arriving behind the interrupt that already reported the
    prefix corrects `played_ms` and leaves the receipt alone."""

    adapter = LiveClientAdapter("receipt-not-erased", gap_ms=10)
    adapter.provider_event(H.out_text("First sentence. Second sentence.", 0, 400), now=0.0)
    oid = adapter.output_id
    _ok, closed = adapter.interrupt(
        oid, oid, 900, heard_text="First sentence.", now=1.0,
    )
    assert closed is not None

    adapter.playback_drained(oid, oid, 1200)

    assert closed.heard_text == "First sentence."
    assert closed.played_ms == 1200


# ── a LATE frame can never undo a receipt the relay already validated ──
#
# Everything in this section drives `_LiveSession`'s own frame handlers.  The
# adapter was already right: `_stamp_heard` refuses to overwrite a receipt with
# an absent claim.  `note_heard_text` was not — it wrote `heard_text = None` for
# all three verdicts — so the SECOND frame of the commonest barge-in ordering
# put the whole generated answer back on the row.  An adapter-level test cannot
# see that, which is why the tests above it passed while the relay was wrong.


@pytest.mark.asyncio
async def test_a_late_idle_with_no_receipt_leaves_the_accepted_prefix_alone():
    """The reproduction, through the real handlers.

    The phone reports "I showed one sentence" on the interrupt, then sends the
    `playback_idle` that corrects `played_ms`.  That second frame carries no
    receipt BY DESIGN — absent is "this client did not say", never "nothing was
    heard" — and the row must still hold one sentence afterwards.
    """

    session = _client_session()
    oid = _speaking(session)
    _ok, closed = session.adapter.interrupt(
        oid, oid, 900, heard_text="First sentence.", now=1.0,
    )
    session.note_heard_text(_frame(heard_text="First sentence."), closed)
    await session.persist_spoken(closed)
    assert session.persistence.payload("live-output:")["assistant_text"] == (
        "First sentence."
    )

    await session.on_playback_idle({
        "type": "playback_idle", "response_id": oid, "item_id": oid,
        "played_ms": 1200,
    })

    payload = session.persistence.payload("live-output:")
    assert payload["assistant_text"] == "First sentence."
    assert payload["assistant_voice"]["heard_chars"] == len("First sentence.")
    # The late frame was not wasted: it is why `played_ms` is now the truth.
    assert payload["assistant_voice"]["played_ms"] == 1200
    assert closed.heard_absent_frames == 1
    assert closed.heard_refused == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "late,reason",
    [
        ({"heard_text": "First \ud83d sentence."}, "malformed"),
        ({"heard_text": 1840}, "malformed"),
        ({"heard_text": "Second sentence."}, "not_prefix"),
        ({"heard_text": "First sentence. Second sentence. And more."}, "not_prefix"),
    ],
)
async def test_a_refused_late_claim_cannot_restore_the_full_answer(
    monkeypatch, late, reason,
):
    """Being WRONG must never be more powerful than being right.

    A malformed or non-prefix claim is refused, and the claim the adapter
    stamped next to `played_ms` is discarded with it — nothing downstream may
    persist a string this relay declined to believe.  But the receipt an
    earlier frame already PROVED is not disproved by a later frame being
    garbage, so that is what the epoch falls back to.  Clearing to `None` here
    handed a forged frame the one thing it was trying to buy: the full
    generated text back on the row.
    """

    counters = H.capture_counters(monkeypatch)
    session = _client_session()
    oid = _speaking(session)
    _ok, closed = session.adapter.interrupt(
        oid, oid, 900, heard_text="First sentence.", now=1.0,
    )
    session.note_heard_text(_frame(heard_text="First sentence."), closed)
    await session.persist_spoken(closed)

    await session.on_playback_idle({
        "type": "playback_idle", "response_id": oid, "item_id": oid,
        "played_ms": 1200, **late,
    })

    assert closed.heard_text == "First sentence."
    assert closed.heard_refused == 1
    assert closed.heard_absent_frames == 0
    assert _reasons(counters) == [reason]
    assert _reasons(counters, "live_heard_text_retained") == ["refused"]
    payload = session.persistence.payload("live-output:")
    assert payload["assistant_text"] == "First sentence."


@pytest.mark.asyncio
async def test_a_late_claim_naming_another_epoch_cannot_restore_the_full_answer(
    monkeypatch,
):
    """Step 4 on a LATE frame.  A receipt that names a different epoch is a
    replay or a forgery; it is evidence about a string this epoch never said,
    and the epoch keeps the receipt it has."""

    counters = H.capture_counters(monkeypatch)
    session = _client_session()
    oid = _speaking(session)
    _ok, closed = session.adapter.interrupt(
        oid, oid, 900, heard_text="First sentence.", now=1.0,
    )
    session.note_heard_text(_frame(heard_text="First sentence."), closed)
    await session.persist_spoken(closed)

    # A well-formed frame for THIS epoch whose `assistant_turn_id` names another.
    await session.on_playback_idle({
        "type": "playback_idle", "response_id": oid, "item_id": oid,
        "assistant_turn_id": _oid(9), "played_ms": 1200,
        "heard_text": "First",
    })

    assert closed.heard_text == "First sentence."
    assert closed.heard_refused == 1
    assert _reasons(counters) == ["identity_mismatch"]
    assert session.persistence.payload("live-output:")["assistant_text"] == (
        "First sentence."
    )


@pytest.mark.asyncio
async def test_only_a_valid_receipt_replaces_the_one_before_it():
    """The other half of the rule: retention is not stickiness.  A phone that
    revises its own display — it showed more of the epoch before the audio
    stopped — is believed, because that claim passed the same five steps the
    first one did."""

    session = _client_session()
    oid = _speaking(session)
    _ok, closed = session.adapter.interrupt(
        oid, oid, 640, heard_text="First", now=1.0,
    )
    session.note_heard_text(_frame(heard_text="First"), closed)
    await session.persist_spoken(closed)
    assert session.persistence.payload("live-output:")["assistant_text"] == "First"

    await session.on_playback_idle({
        "type": "playback_idle", "response_id": oid, "item_id": oid,
        "played_ms": 1200, "heard_text": "First sentence.",
    })

    assert closed.heard_text == "First sentence."
    assert closed.heard_verified == "First sentence."
    assert closed.heard_refused == 0
    assert session.persistence.payload("live-output:")["assistant_text"] == (
        "First sentence."
    )


@pytest.mark.asyncio
async def test_a_late_playback_failure_keeps_what_was_heard_before_it():
    """`on_playback_failed` takes the same path and needs the same rule: the
    speaker dying after one sentence does not unsay that sentence, and a
    failure frame that carries no receipt is not a claim that nothing was
    heard."""

    session = _client_session()
    oid = _speaking(session)
    _ok, closed = session.adapter.interrupt(
        oid, oid, 640, heard_text="First sentence.", now=1.0,
    )
    session.note_heard_text(_frame(heard_text="First sentence."), closed)
    await session.persist_spoken(closed)

    await session.on_playback_failed({
        "type": "playback_failed", "response_id": oid, "item_id": oid,
        "played_ms": 640, "reason": "audio_output_failed",
    })

    assert closed.heard_text == "First sentence."
    assert closed.retire_reason == "playback_failed"
    assert closed.heard_absent_frames == 1
    assert session.persistence.payload("live-output:")["assistant_text"] == (
        "First sentence."
    )


@pytest.mark.asyncio
async def test_no_claim_and_a_refused_claim_are_counted_apart(monkeypatch):
    """Two different facts about the client, two different remedies. An absent
    late frame is the ordinary shape of a second ack and needs nothing done
    about it; a refused one is a client or transport defect. Counted together,
    every refusal disappeared into the normal case — and the retention counter
    fires only when a receipt is actually being DEFENDED, or it would report on
    every frame of every call and mean nothing."""

    counters = H.capture_counters(monkeypatch)
    session = _client_session()
    oid = _speaking(session)
    _ok, closed = session.adapter.interrupt(
        oid, oid, 900, heard_text="First sentence.", now=1.0,
    )
    session.note_heard_text(_frame(heard_text="First sentence."), closed)

    session.note_heard_text({"type": "playback_idle", "response_id": oid,
                             "item_id": oid, "played_ms": 1200}, closed)
    session.note_heard_text(_frame(heard_text="Second sentence."), closed)
    session.note_heard_text({"type": "playback_idle", "response_id": oid,
                             "item_id": oid, "played_ms": 1300}, closed)

    assert closed.heard_absent_frames == 2
    assert closed.heard_refused == 1
    assert closed.heard_text == "First sentence."
    assert _reasons(counters) == ["not_prefix"]
    assert _reasons(counters, "live_heard_text_retained") == [
        "absent", "refused", "absent",
    ]

    # Nothing to defend yet: an epoch with no accepted receipt stays silent.
    fresh = _epoch(heard=None)
    fresh.heard_text = None
    session.note_heard_text(_frame(epoch=1, heard_text=None), fresh)
    assert _reasons(counters, "live_heard_text_retained") == [
        "absent", "refused", "absent",
    ]


def test_the_verdict_and_the_text_are_two_different_answers():
    """`accept_heard_text` returns `None` for both "no claim" and "a claim I
    refused", which is fine for a caller that only needs the text and wrong for
    every caller that writes to the epoch. The authority reports both."""

    from app.services.live_voice_protocol import (
        HEARD_ABSENT, HEARD_ACCEPTED, HEARD_REFUSED,
    )

    session = _session()
    closed = _epoch()

    assert session.judge_heard_text(_frame(), closed) == (HEARD_ABSENT, None)
    assert session.judge_heard_text(_frame(heard_text="First"), closed) == (
        HEARD_ACCEPTED, "First",
    )
    assert session.judge_heard_text(_frame(heard_text=""), closed) == (
        HEARD_ACCEPTED, "",
    )
    assert session.judge_heard_text(_frame(heard_text="Nope."), closed) == (
        HEARD_REFUSED, None,
    )
    # LATE — the epoch is unresolvable, so nothing was checked and nothing is
    # held against the client.
    assert session.judge_heard_text(_frame(heard_text="First"), None) == (
        HEARD_ABSENT, None,
    )
    # …but a frame that could not be parsed at all is a refusal whether or not
    # the epoch is still around.
    assert session.judge_heard_text(_frame(heard_text=1840), None) == (
        HEARD_REFUSED, None,
    )


def test_playback_failed_retires_the_epoch_without_hardening_the_causal_fence():
    """A phone that cannot play an epoch is evidence about the PHONE.  Dropping
    the caller's request because their speaker failed would be the same bug §6
    exists to prevent, so the retirement stays revocable exactly as an
    interrupt's does."""

    from app.services.live_voice_protocol import _RETIREMENT_REASONS

    # Retirement causes are diagnostics now, not a revocability discriminator:
    # keying the fence on HOW playback ended is exactly what dropped real
    # requests (see revocable_parent_turn_id).  The behavioural assertion at the
    # end of this test is the one that matters and is unchanged.
    assert "playback_failed" in _RETIREMENT_REASONS

    adapter = LiveClientAdapter("playback-failed-retire")
    adapter.set_next_parent(f"live-utt:{H.PSID}:1")
    adapter.provider_event(H.out_text("Let me look that up.", 0, 400), now=0.0)
    oid = adapter.output_id
    ok, closed = adapter.interrupt(
        oid, oid, 0, reason="playback_failed", now=1.0,
    )

    assert ok and closed is not None
    assert closed.retire_reason == "playback_failed"
    assert closed.interrupted is True
    assert adapter.revocable_parent_turn_id(grace_s=5.0, now=1.5) == f"live-utt:{H.PSID}:1"




# ── end to end, through the real relay ────────────────────────────────

def _await_frame(client, kind, timeout=3.0):
    """A script step that blocks until the relay has SENT `kind`.

    Sleeping a fixed number of milliseconds instead is not merely flaky here,
    it tests the wrong thing: provider output that arrives while the user turn
    is still open is DEFERRED, and `on_interrupt` clears deferred output by
    design.  An interrupt raced ahead of the reply is a different scenario —
    "the caller spoke over output that never left the relay" — and it is not
    the one these tests are about.
    """

    async def wait():
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while not client.of(kind):
            if loop.time() >= deadline:
                raise AssertionError(f"timed out waiting for a {kind} frame")
            await asyncio.sleep(0.01)

    return wait


def _await_save(recorded, prefix, timeout=3.0):
    """A script step that blocks until a row under `prefix` has been PERSISTED.

    The same reasoning as `_await_frame`, one layer further in: a test about a
    deadline that fires on its own cannot sleep a fixed interval and call that
    evidence the deadline fired — it has to wait for the write and fail if it
    never comes.
    """

    async def wait():
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while not _rows(recorded["saves"], prefix):
            if loop.time() >= deadline:
                raise AssertionError(f"timed out waiting for a {prefix!r} row")
            await asyncio.sleep(0.01)

    return wait


@pytest.mark.asyncio
async def test_the_relay_advertises_the_capability(monkeypatch):
    H.fast_clocks(monkeypatch)
    H.patch_relay(monkeypatch)
    client = H.FakeClient([_config(), 0.05, {"type": "stop"}])
    provider = H.FakeProvider(
        on_send=lambda p, e: p.push(H.closed()) if e["type"] == "session.close" else None,
    )
    await H.run_relay(client, provider, timeout=6)

    ready = client.of("ready")
    assert ready and ready[0]["capabilities"]["heard_text"] is True


def _reply_on_send(provider, event):
    if event["type"] == "session.start":
        provider.push(H.user_delta("what's the deadline", 0, 700))
        provider.push(H.out_text("First sentence. Second sentence.", 900, 1500))
        provider.push(H.out_audio())
    elif event["type"] == "session.close":
        provider.push(H.closed())


def _cut(**extra):
    frame = {
        "type": "interrupt", "response_id": _oid(1), "item_id": _oid(1),
        "played_ms": 640, "heard_text": "First sentence.",
    }
    frame.update(extra)
    return frame


@pytest.mark.asyncio
async def test_end_to_end_a_barge_in_saves_only_what_was_heard(monkeypatch):
    """The owner's symptom, through the real relay: the reply is cut after one
    sentence and the day chat shows one sentence."""

    H.fast_clocks(monkeypatch, voice_live_output_epoch_gap_ms=4000)
    recorded = H.patch_relay(monkeypatch)
    client = H.FakeClient()
    client._script = [
        _config(), _await_frame(client, "audio_delta"), _cut(), 0.2, {"type": "stop"},
    ]
    await H.run_relay(client, H.FakeProvider(on_send=_reply_on_send), timeout=8)

    rows = _rows(recorded["saves"], "live-output:")
    assert rows, "the epoch has to reach the thread"
    assert rows[-1]["assistant_text"] == "First sentence."
    voice = rows[-1]["assistant_voice"]
    assert voice["interrupted"] is True
    assert voice["played_ms"] == 640
    assert voice["heard_chars"] == len("First sentence.")
    assert voice["record_kind"] == "assistant_transcript"


@pytest.mark.asyncio
async def test_end_to_end_a_replayed_receipt_revises_one_row(monkeypatch):
    """RECONNECT / REPLAY.  A phone that resends the same frame — a socket
    retry, a duplicated queue flush — must not produce a second bubble, and the
    second copy must not corrupt the first: same key, same text, one revision
    higher."""

    H.fast_clocks(monkeypatch, voice_live_output_epoch_gap_ms=4000)
    recorded = H.patch_relay(monkeypatch)
    client = H.FakeClient()
    client._script = [
        _config(), _await_frame(client, "audio_delta"),
        _cut(), 0.05, _cut(), 0.2, {"type": "stop"},
    ]
    await H.run_relay(client, H.FakeProvider(on_send=_reply_on_send), timeout=8)

    rows = _rows(recorded["saves"], "live-output:")
    assert rows
    assert len({row["assistant_ref"] for row in rows}) == 1, "one key, revised in place"
    assert all(row["assistant_text"] == "First sentence." for row in rows)
    revisions = [row["assistant_revision"] for row in rows]
    assert revisions == sorted(revisions)


@pytest.mark.asyncio
async def test_end_to_end_the_next_reply_is_not_shortened_by_the_last_receipt(monkeypatch):
    """A SUBSEQUENT reply after an interrupted one.  The receipt belongs to one
    epoch; the epoch after it was heard in full and must be saved in full, or
    one barge-in would truncate the rest of the call."""

    H.fast_clocks(
        monkeypatch,
        voice_live_output_epoch_gap_ms=4000,
        voice_live_interrupt_suppress_ms=50,
    )
    recorded = H.patch_relay(monkeypatch)
    box = {}

    def on_send(provider, event):
        box["provider"] = provider
        _reply_on_send(provider, event)

    def second_reply():
        provider = box["provider"]
        provider.push(H.out_text("A whole second answer.", 3000, 3600))
        provider.push(H.out_audio())

    client = H.FakeClient()
    client._script = [
        _config(), _await_frame(client, "audio_delta"), _cut(), 0.2,
        second_reply, 0.3,
        {"type": "playback_idle", "response_id": _oid(2), "item_id": _oid(2),
         "played_ms": 1200, "heard_text": "A whole second answer."},
        0.2, {"type": "stop"},
    ]
    await H.run_relay(client, H.FakeProvider(on_send=on_send), timeout=8)

    by_ref = {
        row["assistant_ref"]: row for row in _rows(recorded["saves"], "live-output:")
    }
    assert by_ref[f"live-output:{H.PSID}:1"]["assistant_text"] == "First sentence."
    second = by_ref[f"live-output:{H.PSID}:2"]
    assert second["assistant_text"] == "A whole second answer."
    assert second["assistant_voice"]["interrupted"] is False


@pytest.mark.asyncio
async def test_end_to_end_a_failed_output_epoch_is_not_a_failed_track(monkeypatch):
    """One frame type, two unrelated meanings.  A `playback_failed` carrying a
    PLAYBACK IDENTITY says an assistant reply never reached the speaker: retire
    the epoch, arm the same fence, record what was heard — and say nothing to
    the model about a track, because there is no track."""

    H.fast_clocks(monkeypatch, voice_live_output_epoch_gap_ms=4000)
    recorded = H.patch_relay(monkeypatch)
    client = H.FakeClient()
    client._script = [
        _config(), _await_frame(client, "audio_delta"),
        {"type": "playback_failed", "response_id": _oid(1), "item_id": _oid(1),
         "assistant_turn_id": _oid(1), "played_ms": 0,
         "reason": "audio_output_failed", "heard_text": ""},
        0.2, {"type": "stop"},
    ]
    provider = H.FakeProvider(on_send=_reply_on_send)
    await H.run_relay(client, provider, timeout=8)

    appends = [str(e.get("content") or "") for e in provider.of("session.thinking.append")]
    assert not any("Playback did not start" in text for text in appends), (
        "there is no track: the media instruction is about a different failure"
    )
    # Nothing was heard, so no transcript text is written — and the epoch is
    # over, so the relay told the client it is listening again.
    assert _rows(recorded["saves"], "live-output:") == []
    assert "listening" in client.states()


@pytest.mark.asyncio
async def test_end_to_end_a_media_playback_failure_still_warns_the_model(monkeypatch):
    """The historical shape — `video_id`, no playback identity — keeps its
    behaviour exactly.  The discriminator is the PRESENCE of an identity, so a
    media frame that happens to omit its id cannot fall through and silently
    retire a live output epoch."""

    H.fast_clocks(monkeypatch, voice_live_output_epoch_gap_ms=4000)
    H.patch_relay(monkeypatch)
    client = H.FakeClient()
    client._script = [
        _config(), _await_frame(client, "audio_delta"),
        {"type": "playback_failed", "video_id": "abc123", "reason": "no_device"},
        0.2, {"type": "stop"},
    ]
    provider = H.FakeProvider(on_send=_reply_on_send)
    await H.run_relay(client, provider, timeout=8)

    appends = [str(e.get("content") or "") for e in provider.of("session.thinking.append")]
    assert any("Playback did not start" in text for text in appends)


@pytest.mark.asyncio
async def test_end_to_end_a_malformed_played_ms_does_not_take_down_the_call(monkeypatch):
    """`json.loads` accepts NaN, and `int(nan)` raises.  Before the parse was
    hardened this one field killed the client loop — every later frame of the
    call, not just this one."""

    H.fast_clocks(monkeypatch, voice_live_output_epoch_gap_ms=4000)
    recorded = H.patch_relay(monkeypatch)

    class _NanClient(H.FakeClient):
        async def receive_text(self):
            raw = await super().receive_text()
            return raw.replace('"played_ms": "NAN"', '"played_ms": NaN')

    client = _NanClient()
    client._script = [
        _config(), _await_frame(client, "audio_delta"),
        _cut(played_ms="NAN"), 0.2, {"type": "stop"},
    ]
    await H.run_relay(client, H.FakeProvider(on_send=_reply_on_send), timeout=8)

    rows = _rows(recorded["saves"], "live-output:")
    assert rows and rows[-1]["assistant_text"] == "First sentence."
    # Unknown, and therefore OMITTED (F35): this pinned `played_ms is None`,
    # i.e. the null the tenant allowlist dropped and logged as "oversize".
    assert "played_ms" not in rows[-1]["assistant_voice"]
    assert client.of("playback_interrupted"), "the call kept running"


@pytest.mark.asyncio
async def test_end_to_end_persian_speech_is_saved_as_the_caller_heard_it(monkeypatch):
    """Persian is where a naive bound breaks first, and it is the owner's own
    language: the cut lands between words, not inside a character."""

    H.fast_clocks(monkeypatch, voice_live_output_epoch_gap_ms=4000)
    recorded = H.patch_relay(monkeypatch)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("چه خبر", 0, 700))
            provider.push(H.out_text("[خنده] سلام. حالت چطوره؟ من خوبم.", 900, 1600))
            provider.push(H.out_audio())
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient()
    client._script = [
        _config(), _await_frame(client, "audio_delta"),
        _cut(played_ms=480, heard_text="سلام."), 0.2, {"type": "stop"},
    ]
    await H.run_relay(client, H.FakeProvider(on_send=on_send), timeout=8)

    rows = _rows(recorded["saves"], "live-output:")
    assert rows and rows[-1]["assistant_text"] == "سلام."
    # The annotation never reached the thread OR the prefix comparison: the
    # relay canonicalized "[خنده] سلام…" the same way the app did before it
    # could report having displayed "سلام.".
    assert rows[-1]["assistant_voice"]["generated_chars"] == len(
        "سلام. حالت چطوره؟ من خوبم."
    )


@pytest.mark.asyncio
async def test_end_to_end_a_malformed_receipt_is_counted_once_and_changes_nothing(monkeypatch):
    """The frame is validated TWICE per interrupt — once to hand the adapter
    what the client claimed, once by the authority that decides whether it
    stands — so the refusal counter is the one thing that can quietly double.
    A fleet counter reading 2x is a fleet counter nobody can act on.

    And the refusal is total: the row keeps the full canonical epoch text, the
    words are never logged, and the call carries on.
    """

    H.fast_clocks(monkeypatch, voice_live_output_epoch_gap_ms=4000)
    counters = H.capture_counters(monkeypatch)
    recorded = H.patch_relay(monkeypatch)
    client = H.FakeClient()
    client._script = [
        _config(), _await_frame(client, "audio_delta"),
        _cut(heard_text="First \ud83d sentence."), 0.2, {"type": "stop"},
    ]
    await H.run_relay(client, H.FakeProvider(on_send=_reply_on_send), timeout=8)

    assert _reasons(counters) == ["malformed"]
    rows = _rows(recorded["saves"], "live-output:")
    assert rows and rows[-1]["assistant_text"] == "First sentence. Second sentence."
    assert "heard_chars" not in rows[-1]["assistant_voice"]


# ── end to end: the nothing-heard rule across SAVE and READ ───────────


def _settle_window(monkeypatch, seconds: float) -> None:
    """Set the receipt window for a relay run.

    Patched on the METHOD rather than through `H.fast_clocks`, because the
    window is not a `Settings` field: `voice_live_spoken_settle_ms` belongs in
    `app/config.py`, which another owner holds this round, and `Settings`
    refuses `setattr` for a name it does not declare. `spoken_settle_s` is the
    one seam that setting will swap into, so patching it here is the same
    override the setting will be — and these tests do not have to wait on the
    handoff to run.
    """

    import app.services.live_voice_protocol as live

    monkeypatch.setattr(
        live._LiveSession, "spoken_settle_s", lambda self: seconds,
    )


def _no_unheard_text_survives(saves) -> None:
    """READ: run every saved row through the projection a client is served by.

    `schemas.public_heard_text` is the one funnel — `test_voice_record_metadata`
    pins that every message serializer goes through it — so this is what the
    app is actually handed for these rows, not a restatement of the save rule.
    """

    from app.schemas import heard_nothing, public_heard_text

    for row in saves:
        voice = row.get("assistant_voice") or {}
        body = public_heard_text(row.get("assistant_text"), voice)
        if heard_nothing(voice):
            assert body == "", (
                "a row the caller heard nothing of is still serving text: "
                f"{row.get('assistant_ref')}"
            )
        else:
            assert body == (row.get("assistant_text") or ""), (
                "the projection clipped a row that WAS heard: "
                f"{row.get('assistant_ref')}"
            )


@pytest.mark.asyncio
async def test_end_to_end_a_late_idle_does_not_put_the_rest_of_the_answer_back(
    monkeypatch,
):
    """The owner's symptom plus the ack that follows it, through the real relay.

    The phone cuts the reply, reports one sentence, and then — because this is
    what a phone does — sends the `playback_idle` that corrects `played_ms`.
    That frame carries no receipt, and the day chat must still show one
    sentence afterwards. Before this repair the second frame silently restored
    the whole answer, so the fix looked right in every test that stopped at the
    first frame.
    """

    H.fast_clocks(monkeypatch, voice_live_output_epoch_gap_ms=4000)
    recorded = H.patch_relay(monkeypatch)
    client = H.FakeClient()
    client._script = [
        _config(), _await_frame(client, "audio_delta"), _cut(), 0.1,
        {"type": "playback_idle", "response_id": _oid(1), "item_id": _oid(1),
         "played_ms": 1800},
        0.2, {"type": "stop"},
    ]
    await H.run_relay(client, H.FakeProvider(on_send=_reply_on_send), timeout=8)

    rows = _rows(recorded["saves"], "live-output:")
    assert rows, "the epoch has to reach the thread"
    assert rows[-1]["assistant_text"] == "First sentence."
    assert rows[-1]["assistant_voice"]["heard_chars"] == len("First sentence.")
    # The late frame still did its job.
    assert rows[-1]["assistant_voice"]["played_ms"] == 1800
    _no_unheard_text_survives(recorded["saves"])


@pytest.mark.asyncio
async def test_end_to_end_a_gap_retired_epoch_nobody_heard_never_reaches_the_thread(
    monkeypatch,
):
    """SAVE, in its preferred form, through the whole relay.

    The output-gap timer retires the epoch while the phone is still holding
    buffered audio; the phone then reports that it displayed nothing of it. The
    thread must end up with no assistant row for that epoch at all — not a row
    that is written and then retracted, and certainly not the full answer with
    `heard_chars: 0` beside it, which is what shipped.
    """

    H.fast_clocks(monkeypatch, voice_live_output_epoch_gap_ms=80)
    _settle_window(monkeypatch, 5.0)
    recorded = H.patch_relay(monkeypatch)
    client = H.FakeClient()
    client._script = [
        _config(),
        # The GAP has to retire the epoch first: that ordering is the defect.
        _await_frame(client, "speech_segment_complete"),
        {"type": "playback_idle", "response_id": _oid(1), "item_id": _oid(1),
         "played_ms": 0, "heard_text": ""},
        0.2, {"type": "stop"},
    ]
    await H.run_relay(client, H.FakeProvider(on_send=_reply_on_send), timeout=8)

    assert _rows(recorded["saves"], "live-output:") == [], (
        "an epoch the caller heard nothing of was saved anyway"
    )
    assert _rows(recorded["saves"], "live-transcript:") == []
    _no_unheard_text_survives(recorded["saves"])
    # The caller's own sentence is untouched: this rule is about the assistant
    # side, and a call that loses the user's words is a different bug. (The
    # user half of a save is keyed by `user_ref`, so `_rows` cannot see it.)
    assert [
        s for s in recorded["saves"]
        if str(s.get("user_ref") or "").startswith("live-utt:")
    ], "the caller's own turn went missing with the reply"


@pytest.mark.asyncio
async def test_end_to_end_a_gap_retired_epoch_with_no_receipt_still_reaches_the_thread(
    monkeypatch,
):
    """The bound on the same window. A client that never acks must not cost the
    call its assistant side — the wait ends at teardown and the row is written
    with the canonical epoch text, which is what the relay wrote before any of
    this existed."""

    H.fast_clocks(monkeypatch, voice_live_output_epoch_gap_ms=80)
    _settle_window(monkeypatch, 30.0)     # far longer than this test runs
    recorded = H.patch_relay(monkeypatch)
    client = H.FakeClient()
    client._script = [
        _config(), _await_frame(client, "speech_segment_complete"),
        0.1, {"type": "stop"},
    ]
    await H.run_relay(client, H.FakeProvider(on_send=_reply_on_send), timeout=8)

    rows = _rows(recorded["saves"], "live-output:")
    assert rows, "a call that ended in silence lost its assistant side"
    assert rows[-1]["assistant_text"] == "First sentence. Second sentence."
    assert "transcript_retracted" not in rows[-1]["assistant_voice"]
    _no_unheard_text_survives(recorded["saves"])


@pytest.mark.asyncio
async def test_end_to_end_the_window_closes_on_its_own_deadline(monkeypatch):
    """…and it closes on the tick deadline too, not only at teardown: a call
    that goes on for minutes after an unacked epoch must not hold that row
    until hang-up. `next_deadline` has to know about the park, or the tick loop
    sleeps straight past it."""

    H.fast_clocks(monkeypatch, voice_live_output_epoch_gap_ms=80)
    _settle_window(monkeypatch, 0.15)
    recorded = H.patch_relay(monkeypatch)
    client = H.FakeClient()
    client._script = [
        _config(), _await_frame(client, "speech_segment_complete"),
        _await_save(recorded, "live-output:"), {"type": "stop"},
    ]
    await H.run_relay(client, H.FakeProvider(on_send=_reply_on_send), timeout=8)

    rows = _rows(recorded["saves"], "live-output:")
    assert rows and rows[-1]["assistant_text"] == "First sentence. Second sentence."


@pytest.mark.asyncio
@pytest.mark.parametrize(("extra", "who"), [({}, "caller"), ({"reason": "client_request"}, "session_end")])
async def test_end_to_end_the_hangups_interrupt_is_not_the_callers_cut(monkeypatch, extra, who):
    """Voice round 2026-10-06 (DESIGN-R2 D-R8 (a)).  The app ends a call
    with `[interrupt, stop]`, the interrupt carrying `reason: client_request`
    (`liveProtocol.closeCallFrames`).  The relay records WHO cut the epoch, so
    only a barge-in reads as `spoken_interrupted` on the task row."""

    import app.services.live_voice_protocol as live

    H.fast_clocks(monkeypatch, voice_live_output_epoch_gap_ms=4000)
    H.patch_relay(monkeypatch)
    seen = []
    original = live._LiveSession.persist_spoken

    async def spy(self, closed):
        seen.append((closed.epoch, closed.cut_by, live.caller_cut(closed)))
        return await original(self, closed)

    monkeypatch.setattr(live._LiveSession, "persist_spoken", spy)
    client = H.FakeClient()
    client._script = [
        _config(), _await_frame(client, "audio_delta"), _cut(**extra), 0.2, {"type": "stop"},
    ]
    await H.run_relay(client, H.FakeProvider(on_send=_reply_on_send), timeout=8)

    cut = [row for row in seen if row[0] == 1]
    assert cut and cut[0][1] == who, seen
    assert cut[0][2] is (who == "caller"), seen


# ── voice round 2026-10-07 (D-R1): a WORDS-bound telling folds the same way ──

@pytest.mark.asyncio
@pytest.mark.parametrize("cut", [True, False])
async def test_a_words_bound_telling_folds_voice_fields_only(cut):
    """D-R1: a result claim no longer enters the epoch FIFO — the epoch whose
    WORDS prove the answer is bound at retirement and folded onto the task
    row.  The fold is the D-R8 fold: voice fields only (the row keeps the
    verified answer, never the paraphrase), `spoken_interrupted` only when the
    CALLER cut a telling that had begun (the partial-telling floor)."""

    import time as _time

    from app.services.live_voice_protocol import DelegatedTask, ResultDelivery

    session = _session()
    answer = "The application deadline is the first of March, at noon Toronto time."
    ref = f"live-delegation:{H.PSID}:d1"
    payload = {
        "user_text": "", "assistant_text": answer, "assistant_ref": ref,
        "assistant_voice": {
            "source": "delegated", "delegation_id": "d1", "task_id": "d1",
            "parent_user_turn_id": f"live-utt:{H.PSID}:1", "clock": "provider",
        },
    }
    claim = session.build_commentary_claim(ref, payload, delegation_id="d1")
    delivery = ResultDelivery(
        task=DelegatedTask(delegation_id="d1", offset_ms=0, transcript="when is the deadline"),
        answer=answer, claim=claim,
    )
    session.deliveries["d1"] = delivery
    session.active_delivery = delivery
    delivery.sent_monotonic = delivery.started_monotonic = _time.monotonic()
    told = (
        "It's the first of March at noon" if cut
        else "The deadline is the first of March, noon, Toronto time."
    )
    closed = _epoch(
        text=told, interrupted=cut, retire_reason="interrupt" if cut else "gap",
        heard=told if cut else None,
    )
    closed.born_monotonic = _time.monotonic()
    await session.persist_spoken(closed)

    assert delivery.resolved and delivery.spoken
    assert delivery.words_reason == ("caller_cut" if cut else "words")
    row = session.persistence.payload("live-delegation:")
    assert row["assistant_text"] == answer, "the fold wrote the paraphrase onto the result"
    voice = row["assistant_voice"]
    assert voice["epoch"] == 1 and voice["record_kind"] == "task_result"
    assert voice["spoken_interrupted"] is cut
    assert not session.persistence.has("live-output:")
    assert not session.persistence.has("live-transcript:")


# ── Voice round 2026-10-07 (D-R3(5)): a cut PIECE of a sentence is never a row ─
# The 12:04 call wrote «باشه» (epoch 8, no audio) and «، حتماً، دارم تو وب
# می‌گردم.» (epoch 9) as two rows — one sentence the relay rotated on a new
# causal parent — and «از این نیاز به» (a reply cut by a stale stop) as a row of
# its own.  Layer B (`carry_spoken_fragment`, called by `persist_spoken` for an
# unbound epoch) holds such a piece for the next epoch or drops it; a CALLER cut
# that carried audio keeps its row (`spoken_interrupted`, D-R8).

import logging as _logging  # noqa: E402

from app.config import settings as _settings  # noqa: E402
import app.services.live_voice_protocol as _live  # noqa: E402


def _carry_session(monkeypatch, carry_s=3.0):
    monkeypatch.setattr(_settings, "voice_live_fragment_carry_s", carry_s)
    session = _live._LiveSession.__new__(_live._LiveSession)
    session.settings = _settings
    session.spoken_carry = None
    session.last_stop_instruction = 0.0
    return session


def _cut_epoch(epoch, text, *, reason="gap", audio=12, parent="U", retired=None):
    return _live.ClosedEpoch(
        epoch=epoch, output_id=f"o{epoch}", text=text, interrupted=reason == "interrupt",
        parent_user_turn_id=parent, retire_reason=reason, audio_chunks=audio,
        retired_monotonic=time.monotonic() if retired is None else retired,
    )


def test_a_cut_head_is_carried_into_the_epoch_that_continues_it(monkeypatch, caplog):
    caplog.set_level(_logging.INFO, logger="app.services.live_voice_protocol")
    session = _carry_session(monkeypatch)
    head = _cut_epoch(8, "باشه", reason="new_output", audio=0, parent="U11")
    assert session.carry_spoken_fragment(head) is True, "the head is held, not written"
    tail = _cut_epoch(9, "، حتماً، دارم تو وب می‌گردم.", parent="U12")
    assert session.carry_spoken_fragment(tail) is False
    assert tail.text == "باشه، حتماً، دارم تو وب می‌گردم."
    assert session.spoken_carry is None
    lines = [r.getMessage() for r in caplog.records if "spoken fragment" in r.getMessage()]
    assert lines == [
        "[LIVE] spoken fragment held epoch=8 parent=U11 reason=new_output chars=4 audio_chunks=0",
        "[LIVE] spoken fragment merged epoch=8 into=9 chars=4",
    ]


def test_a_held_fragment_nothing_continues_is_dropped_never_written(monkeypatch, caplog):
    caplog.set_level(_logging.INFO, logger="app.services.live_voice_protocol")
    session = _carry_session(monkeypatch)
    assert session.carry_spoken_fragment(_cut_epoch(14, "از این نیاز به", reason="new_output", audio=28))
    held_at = session.spoken_carry["at"]
    session.expire_spoken_carry(now=held_at + 2.9)
    assert session.spoken_carry is not None, "inside the carry window it waits"
    session.expire_spoken_carry(now=held_at + 3.1)
    assert session.spoken_carry is None
    assert session.carry_spoken_fragment(_cut_epoch(15, "باشه", reason="new_output", audio=0))
    session.expire_spoken_carry(force=True)
    dropped = [r.getMessage() for r in caplog.records if "spoken fragment dropped" in r.getMessage()]
    assert dropped == [
        "[LIVE] spoken fragment dropped epoch=14 reason=expired voiced=1",
        "[LIVE] spoken fragment dropped epoch=15 reason=teardown voiced=0",
    ]
    # Words only in the held text, never in a log line.
    assert not any("نیاز" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize(
    ("closed", "carried"),
    [
        # A finished sentence keeps its row, however short.
        (dict(epoch=1, text="باشه.", reason="new_output", audio=0), False),
        # A long unfinished VOICED epoch the relay rotated keeps its row.
        (dict(epoch=1, text="این یک جملهٔ طولانی است که هنوز تمام نشده و", reason="new_output",
              audio=30), False),
        # C4: a CALLER cut that carried audio is what they heard — its row stays.
        (dict(epoch=1, text="از این نیاز به", reason="interrupt", audio=28), False),
        (dict(epoch=1, text="از این نیاز به", reason="spoken_over", audio=28), False),
        # …a caller cut before a single chunk was voiced is a fragment.
        (dict(epoch=1, text="از این نیاز به", reason="interrupt", audio=0), True),
        # A short epoch the gap ended right after the relay's own stop note.
        (dict(epoch=1, text="از این نیاز به", reason="gap", audio=28, stop=True), True),
        # …and without that stop it is an ordinary short reply.
        (dict(epoch=1, text="از این نیاز به", reason="gap", audio=28), False),
    ],
)
def test_only_a_cut_piece_is_a_fragment(monkeypatch, closed, carried):
    session = _carry_session(monkeypatch)
    stop = closed.pop("stop", False)
    now = time.monotonic()
    if stop:
        session.last_stop_instruction = now - 0.5
    epoch = _cut_epoch(closed.pop("epoch"), closed.pop("text"), retired=now, **closed)
    assert session.carry_spoken_fragment(epoch) is carried


def test_a_row_never_starts_with_cut_punctuation_and_the_switch_turns_it_all_off(monkeypatch, caplog):
    caplog.set_level(_logging.INFO, logger="app.services.live_voice_protocol")
    session = _carry_session(monkeypatch)
    lead = _cut_epoch(9, "، حتماً، دارم تو وب می‌گردم.")
    assert session.carry_spoken_fragment(lead) is False
    assert lead.text == "حتماً، دارم تو وب می‌گردم."
    assert any(r.getMessage() == "[LIVE] spoken text lead trimmed epoch=9" for r in caplog.records)
    off = _carry_session(monkeypatch, carry_s=0)
    head = _cut_epoch(8, "باشه", reason="new_output", audio=0)
    assert off.carry_spoken_fragment(head) is False and off.spoken_carry is None
    untouched = _cut_epoch(9, "، حتماً.")
    assert off.carry_spoken_fragment(untouched) is False and untouched.text == "، حتماً."


@pytest.mark.parametrize(
    ("text", "reason", "audio", "flag"),
    [
        ("باشه", "new_output", 0, True),
        ("باشه", "interrupt", 0, True),
        ("باشه", "new_output", 3, False),     # voiced: the phone played it
        ("باشه.", "new_output", 0, False),    # a whole sentence
        ("باشه", "gap", 0, False),            # nothing cut it
    ],
)
def test_the_live_wire_marks_a_structurally_cut_epoch(text, reason, audio, flag):
    adapter = _live.LiveClientAdapter("psid-fragment")
    frames = adapter._segment_frames(_cut_epoch(3, text, reason=reason, audio=audio))
    segment = next(f for f in frames if f["type"] == "speech_segment_complete")
    assert segment.get("fragment", False) is flag
    assert ("fragment" in segment) is flag, "additive: absent unless true"


def _parked_rows(session):
    return {
        r.key.rsplit(":", 1)[1]: r.payload["assistant_text"]
        for r in session.persistence.records if r.key.startswith("live-output:")
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("second_pass", ["flush", "late_receipt"])
async def test_a_parked_epochs_second_pass_never_takes_a_later_fragment(monkeypatch, second_pass):
    """Phase D turns-startup-3: on a heard_text client a gap-retired epoch is
    PARKED and `persist_spoken` runs for it again — the park's flush, or a
    late playback receipt.  A fragment held in between (a relay-cut «باشه»)
    was merged into that OLDER epoch on its second pass: its row became
    «باشه The library opens at nine.» and the real continuation lost its
    head.  Layer B judges an epoch once, and never merges into an epoch born
    before the fragment."""

    from app.config import settings as app_settings

    monkeypatch.setattr(app_settings, "voice_live_fragment_carry_s", 3.0)
    session = _session(park=True)
    session.spoken_carry = None
    session.last_stop_instruction = 0.0
    session.spoken_settle_s = lambda: 2.5
    reply = _cut_epoch(7, "The library opens at nine.", audio=40, parent="U6")
    await session.persist_spoken(reply)                       # parked
    await session.persist_spoken(_cut_epoch(8, "باشه", reason="new_output", audio=0, parent="U7"))
    if second_pass == "flush":
        await session.flush_parked_spoken(now=time.monotonic() + 3.0)
    else:
        reply.retire_reason, reply.played_ms = "playback_idle", 2000
        await session.persist_spoken(reply)
    tail = _cut_epoch(9, "، حتماً، دارم تو وب می‌گردم.", audio=30, parent="U7")
    tail.retire_reason = "playback_idle"
    await session.persist_spoken(tail)
    rows = _parked_rows(session)
    assert rows.get("7") == "The library opens at nine.", rows
    assert rows.get("9") == "باشه، حتماً، دارم تو وب می‌گردم.", rows


_KEEP_HEAD, _KEEP_TAIL, _KEEP_HEARD = "باشه", "، حتماً، دارم تو وب می‌گردم.", "، حتماً،"


def _receipt_session(monkeypatch, *, park):
    from app.config import settings as app_settings

    monkeypatch.setattr(app_settings, "voice_live_fragment_carry_s", 3.0)
    session = _client_session(park=park)
    session.spoken_carry = None
    session.last_stop_instruction = 0.0
    return session


def _receipt_head():
    return _live.ClosedEpoch(
        epoch=0, output_id=_oid(0), text=_KEEP_HEAD, interrupted=False, played_ms=None,
        parent_user_turn_id="U1", start_ms=100, end_ms=200, retire_reason="new_output",
        audio_chunks=0, retired_monotonic=time.monotonic(),
    )


def _receipt_rows(session):
    return [
        (r.payload["assistant_text"], r.payload["assistant_voice"].get("heard_chars"))
        for r in session.persistence.records if r.key.startswith("live-output:")
    ]


@pytest.mark.asyncio
async def test_a_receipt_row_keeps_the_carried_head_and_no_leading_comma(monkeypatch):
    """Phase D turns-startup-4 (a): «باشه» is held; the caller barges into the
    continuation «، حتماً…» and the phone sends its receipt «، حتماً،».  The row
    was written from the receipt — validated against the epoch BEFORE Layer B
    ran — so the head was lost and the row began with «،» (D-R3(5): leading
    punctuation never starts a row).  The receipt now gets the same prepend
    and trim as the epoch's text."""

    counters = H.capture_counters(monkeypatch)
    session = _receipt_session(monkeypatch, park=False)
    await session.persist_spoken(_receipt_head())
    session.adapter.provider_event(H.out_text(_KEEP_TAIL, 300, 900), now=1.0)
    session.adapter.provider_event(H.out_audio("AQID"), now=1.0)
    oid = session.adapter.output_id
    _ok, closed = session.adapter.interrupt(oid, oid, 900, heard_text=_KEEP_HEARD, now=2.0)
    closed.cut_by = "caller"
    session.note_heard_text({
        "type": "interrupt", "response_id": oid, "item_id": oid, "played_ms": 900,
        "heard_text": _KEEP_HEARD,
    }, closed)
    await session.persist_spoken(closed)
    texts = [text for text, _chars in _receipt_rows(session)]
    assert texts == ["باشه، حتماً،"], texts
    assert not [c for c in counters if c[0] == "live_heard_text_rejected"], counters


@pytest.mark.asyncio
async def test_a_parked_merged_epoch_accepts_its_own_receipt(monkeypatch):
    """Phase D turns-startup-4 (b): the gap-retired continuation was merged
    («باشه، حتماً…») and parked; the caller's barge-in then sends the receipt
    of what the phone displayed for THAT epoch («، حتماً،»).  It was refused
    `not_prefix` (a client defect on the counter) and the full generated text
    went on a row the caller cut.  The receipt is judged against the epoch's
    own words, and the row says what was heard, head included."""

    counters = H.capture_counters(monkeypatch)
    session = _receipt_session(monkeypatch, park=True)
    await session.persist_spoken(_receipt_head())
    session.adapter.provider_event(H.out_text(_KEEP_TAIL, 300, 900), now=1.0)
    session.adapter.provider_event(H.out_audio("AQID"), now=1.0)
    oid = session.adapter.output_id
    _frames, closed = session.adapter.rotate_on_gap(now=100.0)
    await session.persist_spoken(closed)                      # merged, parked
    _ok, past = session.adapter.interrupt(oid, oid, 900, heard_text=_KEEP_HEARD, now=101.0)
    assert past is closed
    session.note_heard_text({
        "type": "interrupt", "response_id": oid, "item_id": oid, "played_ms": 900,
        "heard_text": _KEEP_HEARD,
    }, past)
    await session.persist_spoken(past)
    assert not [c for c in counters if c[0] == "live_heard_text_rejected"], counters
    assert past.heard_text == "باشه، حتماً،", past.heard_text
    assert [text for text, _chars in _receipt_rows(session)] == ["باشه، حتماً،"]
