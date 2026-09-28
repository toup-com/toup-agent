"""The nothing-heard rule, proved SAVE -> READ in one place.

Three separate workers implemented the three layers of this rule and each
pinned its own half.  That is not the same as proving the rule holds, because a
layer can be individually correct and still not compose: the save layer can
write a marker the read layer does not look for, or the read layer can project
on a key the save layer never emits.  The supervisor asked for behavioural
save/read evidence that no unheard transcript remains while the task result
stays intact, so this file walks one epoch through both layers and asserts the
user-visible outcome rather than either layer's internals.

The RENDER layer is proved in the app repo (scripts/check-history-honesty.js and
scripts/check-voice-chat-unity.js execute the app's own predicate); it cannot be
reached from Python, and pretending otherwise here would be the fake-coverage
this round exists to remove.
"""
import asyncio

import pytest

from app import schemas
from test_live_voice_records import _session, _epoch, _frame


def _read_every_surface(text, voice):
    """What each client-facing serializer would show for this stored row."""

    from app.api.message_cards import public_text

    body = public_text("assistant", text)
    return schemas.public_heard_text(body, voice)


@pytest.mark.asyncio
async def test_production_ordering_never_writes_a_word_the_caller_did_not_hear():
    """gap retirement + a receipt of nothing => no row at all.

    This is the ordering a real phone produces, and it is the strongest form of
    the rule: the unheard words are not projected away later, they are never
    written.
    """

    session = _session(park=True)
    session.tick_wake = asyncio.Event()

    async def _emit(_frames):
        return None

    session.emit_frames = _emit
    closed = _epoch(interrupted=False, heard=None, retire_reason="gap")

    await session.persist_spoken(closed)
    assert session.persistence.records == [], (
        "the gap timer wrote the epoch before the phone had a chance to report "
        "what it played"
    )

    session.note_heard_text(_frame(heard_text=""), closed)
    closed.interrupted = True
    await session.persist_spoken(closed)

    assert session.persistence.records == [], (
        "nothing was heard, so nothing may be saved"
    )


@pytest.mark.asyncio
async def test_a_row_written_before_the_receipt_is_retracted_and_reads_empty():
    """The settle window can expire before a slow phone reports.

    The row then exists and cannot be deleted — the persistence API declines an
    empty message and the tenant route 400s on empty content — so the rule is
    carried by the retraction marker plus the read projection.  What matters is
    the user-visible outcome, which is what this asserts.
    """

    session = _session(park=False)
    session.tick_wake = asyncio.Event()

    async def _emit(_frames):
        return None

    session.emit_frames = _emit
    closed = _epoch(interrupted=False, heard=None, retire_reason="gap")

    await session.persist_spoken(closed)
    first = session.persistence.payload("live-output:")
    assert first["assistant_text"], "the provisional row should hold the text"
    assert _read_every_surface(first["assistant_text"], first["assistant_voice"]), (
        "before any receipt the row reads normally"
    )

    session.note_heard_text(_frame(heard_text=""), closed)
    closed.interrupted = True
    await session.persist_spoken(closed)

    latest = session.persistence.payload("live-output:")
    voice = latest["assistant_voice"]
    assert voice["heard_chars"] == 0
    assert voice["transcript_retracted"] is True
    assert voice["record_kind"] == "assistant_transcript"

    # THE ASSERTION THAT MATTERS: no surface a client can read serves the words.
    assert _read_every_surface(latest["assistant_text"], voice) == "", (
        "a caller re-opening this chat would be shown words they never heard"
    )
    assert schemas.heard_nothing(voice) is True


def test_a_task_result_is_never_projected_away_even_with_a_zero_receipt():
    """The backend answer is not spoken words and is never clipped or hidden.

    A `task_result` row carrying the same zero-heard provenance must still read
    in full: the caller not hearing the paraphrase says nothing about whether
    the work was done or what it found.
    """

    answer = "The professor is Dr. Azadeh Sharafi, University of Toronto."
    voice = {
        "record_kind": "task_result",
        "heard_chars": 0,
        "transcript_retracted": True,
        "task_id": "d1",
    }
    assert schemas.heard_nothing(voice) is False
    assert _read_every_surface(answer, voice) == answer


def test_a_partly_heard_transcript_keeps_exactly_what_was_heard():
    """Anti-vacuity: the projection must not blank every voice row."""

    voice = {"record_kind": "assistant_transcript", "heard_chars": 15}
    assert schemas.heard_nothing(voice) is False
    assert _read_every_surface("First sentence.", voice) == "First sentence."


def test_a_legacy_row_with_no_voice_metadata_is_untouched():
    """The tenant-rollout window: no record_kind means legacy behaviour."""

    assert _read_every_surface("Hello.", None) == "Hello."
    assert _read_every_surface("Hello.", {"source": "live_spoken"}) == "Hello."
    assert schemas.heard_nothing({"source": "live_spoken"}) is False
