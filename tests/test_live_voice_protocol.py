"""Offline contract tests for the GPT-Live transport.

Pure units and wire shapes live here.  The lifecycle (delegation, persistence,
instructions) is driven end-to-end in the sibling `test_live_*` modules.
"""

import pytest

from app.config import settings
from app.services.live_voice_protocol import (
    LIVE_STATES,
    LIVE_WS_URL,
    LiveClientAdapter,
    LiveDelegationTracker,
    LiveUsageTracker,
    Utterance,
    UtteranceAssembler,
    audio_append,
    build_session_start,
    classify_followup,
    close_session,
    commentary_append,
    continuation_repeats,
    delegated_agent_input,
    detect_media_control,
    detect_reply_language,
    estimate_tokens,
    instructions_append,
    looks_unfinished,
    relay_line,
    resolve_live_voice,
    sanitize_user_visible_answer,
    select_continuation_source,
    split_commentary,
    thinking_append,
    token_overlap,
)

import test_live_harness as H


def test_live_connection_and_start_are_not_realtime_shapes():
    assert LIVE_WS_URL == "wss://api.openai.com/v1/live/sessions"
    event = build_session_start(instructions="Keep speech brief.", voice="quartz")
    assert event == {
        "type": "session.start",
        "event_id": "toup-live-start",
        "session": {
            "model": "gpt-live-1",
            "instructions": "Keep speech brief.",
            "audio": {
                "format": {"type": "audio/pcm", "rate": 24000},
                "output": {"voice": "quartz"},
            },
            "delegation": {"type": "client"},
            "store": False,
        },
    }
    assert "tools" not in event["session"]


def test_live_provider_requires_exact_client_protocol_and_server_flag(monkeypatch):
    from app.api.ws_realtime import _live_protocol_selected

    pilot = "00000000-aaaa-4bbb-8ccc-000000000001"
    other = "00000000-aaaa-4bbb-8ccc-000000000002"
    monkeypatch.setattr(settings, "voice_live_enabled", True)
    monkeypatch.setattr(settings, "voice_live_provider", "openai")
    monkeypatch.setattr(settings, "voice_live_user_ids", pilot)
    monkeypatch.setattr(settings, "voice_live_all_users", False)
    assert _live_protocol_selected(pilot, "live1", onboarding=False)
    assert not _live_protocol_selected(other, "live1", onboarding=False)
    assert not _live_protocol_selected(pilot[:8], "live1", onboarding=False)
    assert not _live_protocol_selected(pilot, None, onboarding=False)
    assert not _live_protocol_selected(pilot, "realtime", onboarding=False)
    assert not _live_protocol_selected(pilot, "live1", onboarding=True)

    monkeypatch.setattr(settings, "voice_live_enabled", False)
    assert not _live_protocol_selected(pilot, "live1", onboarding=False)


def test_live_global_flag_does_not_mean_all_users_without_explicit_switch(monkeypatch):
    from app.api.ws_realtime import _live_protocol_selected

    pilot = "00000000-aaaa-4bbb-8ccc-000000000001"
    monkeypatch.setattr(settings, "voice_live_enabled", True)
    monkeypatch.setattr(settings, "voice_live_provider", "openai")
    monkeypatch.setattr(settings, "voice_live_user_ids", "")
    monkeypatch.setattr(settings, "voice_live_all_users", False)
    assert not _live_protocol_selected(pilot, "live1", onboarding=False)

    monkeypatch.setattr(settings, "voice_live_all_users", True)
    assert _live_protocol_selected(pilot, "live1", onboarding=False)


def test_live_cohort_rejects_noncanonical_or_truncated_ids(monkeypatch, caplog):
    from app.api.ws_realtime import _live_protocol_selected

    pilot = "00000000-aaaa-4bbb-8ccc-000000000001"
    monkeypatch.setattr(settings, "voice_live_enabled", True)
    monkeypatch.setattr(settings, "voice_live_provider", "openai")
    monkeypatch.setattr(settings, "voice_live_all_users", False)
    monkeypatch.setattr(settings, "voice_live_user_ids", pilot[:8])

    assert not _live_protocol_selected(pilot, "live1", onboarding=False)
    assert "canonical full UUIDs are required" in caplog.text


def test_live_cohort_keeps_valid_id_when_another_entry_is_invalid(monkeypatch):
    from app.api.ws_realtime import _live_protocol_selected

    pilot = "00000000-aaaa-4bbb-8ccc-000000000001"
    monkeypatch.setattr(settings, "voice_live_enabled", True)
    monkeypatch.setattr(settings, "voice_live_provider", "openai")
    monkeypatch.setattr(settings, "voice_live_all_users", False)
    monkeypatch.setattr(settings, "voice_live_user_ids", f"broken,{pilot}")

    assert _live_protocol_selected(pilot, "live1", onboarding=False)


def test_live_voice_does_not_forward_realtime_only_voice():
    assert resolve_live_voice("shimmer") == "marin"
    assert resolve_live_voice("bogus", default="willow") == "willow"
    assert resolve_live_voice("Gleam") == "gleam"


def test_live_history_uses_role_specific_content_and_bounds_rows():
    history = [{"role": "user", "content": f"q{i}"} for i in range(130)]
    history += [{"role": "assistant", "content": "answer"}, {"role": "tool", "content": "drop"}]
    session = build_session_start(instructions="x", history=history)["session"]
    assert len(session["input"]) == 128
    assert session["input"][0]["content"][0]["text"] == "q3"
    assert session["input"][-1]["content"] == [{"type": "output_text", "text": "answer"}]


def test_live_audio_close_and_delegated_result_shapes():
    assert audio_append("AAEC") == {"type": "session.input_audio.append", "audio": "AAEC"}
    assert close_session("close-7") == {"type": "session.close", "event_id": "close-7"}
    assert commentary_append("delegate-1", "Verified result", event_id="result-1") == {
        "type": "session.commentary.append",
        "event_id": "result-1",
        "delegation_id": "delegate-1",
        "content": "Verified result",
    }
    assert instructions_append("Stop speaking.", event_id="stop-1") == {
        "type": "session.instructions.append",
        "event_id": "stop-1",
        "delegation_id": None,
        "content": "Stop speaking.",
    }
    assert thinking_append("Step: searching.", event_id="p-1", delegation_id="d1") == {
        "type": "session.thinking.append",
        "event_id": "p-1",
        "delegation_id": "d1",
        "content": "Step: searching.",
    }


def test_only_four_state_values_may_reach_a_shipped_client():
    # Both clients map exactly these; a fifth is dropped on mobile and mapped
    # to null on web, i.e. the orb latches on whatever it was showing.
    assert LIVE_STATES == ("listening", "thinking", "tool_use", "speaking")


def test_delegated_agent_followup_gets_bounded_same_call_context():
    assert delegated_agent_input("First request", []) == "First request"
    prompt = delegated_agent_input(
        "Make that shorter",
        [("Draft a note", "Here is the longer note")],
    )
    assert "Prior caller request: Draft a note" in prompt
    assert "Accepted backend result: Here is the longer note" in prompt
    assert prompt.endswith("Current caller request: Make that shorter")

    bounded = delegated_agent_input(
        "Now",
        [("u" * 100, "a" * 100), ("recent", "kept")],
        max_chars=80,
    )
    assert "recent" in bounded
    assert "u" * 100 not in bounded


def test_a_superseded_attempts_executed_operations_ride_into_its_replacement():
    prompt = delegated_agent_input(
        "Make it Thursday instead",
        [],
        ledger=["calendar__create_event", "gmail__send"],
    )
    assert "do NOT repeat" in prompt
    assert "calendar__create_event" in prompt
    assert prompt.rstrip().endswith("Make it Thursday instead")


def test_the_language_directive_rides_the_message_until_the_agent_image_lands():
    prompt = delegated_agent_input("چه خبر", [], language_directive="Reply in the caller's language.")
    assert prompt.endswith("Reply in the caller's language.")
    assert "چه خبر" in prompt


# ── transcript / delegation correlation ───────────────────────────────

def test_speech_during_a_running_delegation_does_not_discard_its_answer():
    # THE round-48 reversal. `add_transcript` used to bump `revision` on every
    # user fragment past the consumed watermark, so a backchannel, a status
    # question or the tail of the same sentence fenced a 25-second research
    # turn: nothing spoken, nothing persisted, the orb stuck on thinking.
    tracker = LiveDelegationTracker()
    utterance = Utterance(turn_id="t1", ordinal=1, start_ms=0, end_ms=500, text="Find the weather")
    tracker.add_delegation("d1", 500)
    task = tracker.consume("d1", utterance)
    assert task is not None and tracker.accepts_result(task)

    tracker.add_transcript("user", "آره", 700, 900)
    tracker.add_transcript("user", "what are you searching?", 1000, 1800)
    assert tracker.accepts_result(task)

    # Only an EXPLICIT supersession makes a result stale.
    assert tracker.supersede("d1", by="d2", reason="correction")
    assert not tracker.accepts_result(task)


def test_a_delegation_dispatches_on_the_closed_utterance_whatever_its_offset():
    # A6-4: offset_ms is the position of the provider's DECISION, normally
    # after the utterance ended. Treating it as a watermark the user had to
    # cross parked the task until they repeated themselves.
    tracker = LiveDelegationTracker()
    utterance = Utterance(turn_id="t1", ordinal=1, start_ms=0, end_ms=3000, text="play a song")
    tracker.add_delegation("d1", 3400)
    task = tracker.consume("d1", utterance)
    assert task is not None
    assert task.transcript == "play a song"
    assert utterance.consumed_by == "d1"
    assert tracker.consume("d1", utterance) is None


def test_pending_delegations_are_ordered_by_offset_not_by_hash():
    tracker = LiveDelegationTracker()
    assert tracker.add_delegation("late", 900)
    assert tracker.add_delegation("early", 100)
    assert not tracker.add_delegation("early", 100)
    assert [p.delegation_id for p in tracker.pending] == ["early", "late"]


def test_a_pending_delegation_expires_rather_than_capturing_the_next_utterance():
    tracker = LiveDelegationTracker()
    tracker.add_delegation("orphan", 100)
    assert tracker.expired(10.0) == []
    stale = tracker.expired(0.0)
    assert [p.delegation_id for p in stale] == ["orphan"]
    tracker.drop_pending("orphan")
    assert tracker.pending == []


def test_fragment_history_is_bounded():
    tracker = LiveDelegationTracker()
    for i in range(500):
        tracker.add_transcript("user", "x", i, i + 1)
    assert len(tracker.fragments) <= 400


# ── utterance assembly ────────────────────────────────────────────────

def test_fragments_within_the_gap_coalesce_into_one_turn():
    asm = UtteranceAssembler(provider_session_id="p", gap_ms=1200)
    rotated, current = asm.add("که ", 0, 300)
    assert rotated is None
    rotated, current = asm.add("ایونت ", 700, 1100)
    assert rotated is None
    rotated, current = asm.add("معروف", 1400, 1900)
    assert rotated is None
    assert current.text == "که ایونت معروف"
    assert current.turn_id == "live-utt:p:1"
    assert current.revision == 3


def test_a_silence_gap_opens_the_next_turn():
    asm = UtteranceAssembler(provider_session_id="p", gap_ms=1200)
    asm.add("first", 0, 400)
    rotated, current = asm.add("second", 2000, 2400)
    assert rotated is not None
    assert rotated.turn_id == "live-utt:p:1"
    assert rotated.close_reason == "gap"
    assert current.turn_id == "live-utt:p:2"
    assert current.text == "second"


@pytest.mark.parametrize(
    ("head", "tail", "combined"),
    [
        ("یه استاد", " دانشگاه تورنتو پیدا کن", "یه استاد دانشگاه تورنتو پیدا کن"),
        ("find a professor and", " check their publications", "find a professor and check their publications"),
    ],
)
def test_an_unfinished_tail_waits_for_the_hard_gap_and_keeps_its_turn(
    head, tail, combined,
):
    gap_ms = 1200
    hard_gap_ms = 2600

    waiting = UtteranceAssembler(
        provider_session_id="p-wait", gap_ms=gap_ms, hard_gap_ms=hard_gap_ms,
    )
    waiting.add(head, 0, 400)
    assert not waiting.should_close_on_gap(400 + gap_ms)
    assert not waiting.should_close_on_gap(400 + hard_gap_ms - 1)
    assert waiting.should_close_on_gap(400 + hard_gap_ms)

    # A continuation arriving after the ordinary gap but before the hard gap is
    # still part of the same utterance. Provider delegation is not an endpoint.
    merging = UtteranceAssembler(
        provider_session_id="p-merge", gap_ms=gap_ms, hard_gap_ms=hard_gap_ms,
    )
    _, original = merging.add(head, 0, 400)
    rotated, current = merging.add(tail, 400 + gap_ms + 100, 2000)
    assert rotated is None
    assert current is original
    assert current.turn_id == "live-utt:p-merge:1"
    assert current.text == combined


@pytest.mark.parametrize(
    "text",
    [
        "find a",
        "find an",
        "find the",
        "می خوام",
        "می‌خوام",
        "میخوام",
    ],
)
def test_reviewed_unfinished_determiners_and_persian_phrases_use_hard_gap(text):
    assert looks_unfinished(text)


def test_duplicate_and_stale_fragments_are_no_ops():
    asm = UtteranceAssembler(provider_session_id="p", gap_ms=1200)
    _, current = asm.add("find ", 100, 200)
    _, current = asm.add("a professor", 200, 400)
    snapshot = (current.text, current.revision, current.start_ms, current.end_ms, asm.ordinal)

    rotated, duplicate_owner = asm.add("a professor", 200, 400)
    assert rotated is None
    assert duplicate_owner is current
    assert (current.text, current.revision, current.start_ms, current.end_ms, asm.ordinal) == snapshot

    rotated, stale_owner = asm.add("stale hypothesis ", 50, 150)
    assert rotated is None
    assert stale_owner is current
    assert (current.text, current.revision, current.start_ms, current.end_ms, asm.ordinal) == snapshot
    assert asm.closed_queue == []


def test_exact_fragment_replay_stays_a_no_op_after_owner_and_history_eviction():
    asm = UtteranceAssembler(
        provider_session_id="p", gap_ms=1200, max_chars=100_000,
    )
    tracker = LiveDelegationTracker()
    for index in range(401):
        text = f"fragment-{index}|"
        assert tracker.add_transcript("user", text, 0, 1)
        asm.add(text, 0, 1)

    current = asm.open
    assert current is not None
    snapshot = (current.text, current.revision, asm.ordinal)
    assert not tracker.add_transcript("user", "fragment-0|", 0, 1)
    rotated, owner = asm.add("fragment-0|", 0, 1)

    assert rotated is None
    assert owner is current
    assert (current.text, current.revision, asm.ordinal) == snapshot


def test_causal_candidate_uses_offset_and_skips_directly_dispatched_turns():
    asm = UtteranceAssembler(provider_session_id="p")
    first = Utterance(
        turn_id="live-utt:p:1", ordinal=1, start_ms=100, end_ms=200,
        text="first search", closed=True,
    )
    latest = Utterance(
        turn_id="live-utt:p:2", ordinal=2, start_ms=300, end_ms=400,
        text="second search", closed=True,
    )
    direct = Utterance(
        turn_id="live-utt:p:3", ordinal=3, start_ms=500, end_ms=600,
        text="hello", closed=True, dispatched=True,
    )
    future = Utterance(
        turn_id="live-utt:p:4", ordinal=4, start_ms=800, end_ms=900,
        text="future request", closed=True,
    )
    asm.closed_queue = [first, latest, direct, future]

    assert asm.causal_candidate(700) is latest
    latest.consumed_by = "delegation-2"
    assert asm.causal_candidate(700) is first
    assert asm.causal_candidate(0) is None
    assert asm.causal_candidate(50) is None


def test_a_monologue_is_paragraphs_not_one_unbounded_row():
    asm = UtteranceAssembler(provider_session_id="p", gap_ms=1200, max_chars=20)
    asm.add("x" * 15, 0, 100)
    rotated, current = asm.add("y" * 15, 150, 250)
    assert rotated is not None and rotated.close_reason == "length"
    assert current.text == "y" * 15


@pytest.mark.parametrize(
    ("kwargs", "text", "start_ms", "end_ms"),
    [
        ({"max_chars": 20}, "x" * 21, 0, 10),
        ({"max_ms": 100}, "short", 0, 101),
    ],
)
def test_one_oversized_provider_fragment_closes_at_the_safety_ceiling(
    kwargs, text, start_ms, end_ms,
):
    asm = UtteranceAssembler(provider_session_id="p", **kwargs)

    rotated, current = asm.add(text, start_ms, end_ms)

    assert rotated is current
    assert current.closed
    assert current.close_reason == "limit"
    assert asm.open is None
    assert asm.closed_queue == [current]


def test_a_consumed_turn_is_not_offered_to_a_second_delegation():
    asm = UtteranceAssembler(provider_session_id="p", gap_ms=10)
    asm.add("hello", 0, 100)
    closed = asm.close("gap")
    assert closed is not None
    assert [u.turn_id for u in asm.unconsumed_closed()] == ["live-utt:p:1"]
    closed.consumed_by = "d1"
    assert asm.unconsumed_closed() == []


# ── output epochs ─────────────────────────────────────────────────────

def test_live_provider_output_preserves_toup_client_envelope():
    adapter = LiveClientAdapter("session-1")
    assert adapter.provider_event({
        "type": "session.output_transcript.delta", "delta": "Hello ",
    }) == [{"type": "response_text", "text": "Hello ", "partial": True}]
    output_id = adapter.output_id
    assert output_id == "live:session-1:1"

    frames = adapter.provider_event({"type": "session.output_audio.delta", "delta": "AAEC"})
    assert frames == [
        {"type": "state", "state": "speaking"},
        {
            "type": "audio_delta",
            "response_id": output_id,
            "item_id": output_id,
            "item": output_id,
            "content_index": 0,
            "data": "AAEC",
        },
    ]
    adapter.provider_event({"type": "session.output_transcript.delta", "delta": "there"})
    frames, closed = adapter.playback_drained(output_id, output_id, played_ms=800)
    assert closed is not None and closed.played_ms == 800
    assert frames == [
        {
            "type": "speech_segment_complete",
            "response_id": output_id,
            "item_id": output_id,
            "text": "Hello there",
            # Load-bearing on both shipped clients: captions only.
            "provider_complete": False,
            "durable": False,
            "epoch": 1,
            "played_ms": 800,
            "interrupted": False,
        },
        {"type": "state", "state": "listening"},
    ]


def test_negotiated_turn_timing_preserves_exact_response_text_causality():
    adapter = LiveClientAdapter("session-timing", turn_timing=True)
    adapter.set_next_parent("live-utt:provider:7")

    assert adapter.provider_event({
        "type": "session.output_transcript.delta",
        "delta": "سلام ",
        "start_ms": 22400,
        "end_ms": 22760,
    }, now=10.0) == [{
        "type": "response_text",
        "text": "سلام ",
        "partial": True,
        "assistant_turn_id": "live:session-timing:1",
        "epoch": 1,
        "parent_user_turn_id": "live-utt:provider:7",
        "start_ms": 22400,
        "end_ms": 22760,
        "clock": "provider",
    }]

    assert adapter.provider_event({
        "type": "session.output_transcript.delta",
        "delta": "دنیا",
        "start_ms": 22760,
        "end_ms": 23100,
    }, now=10.1) == [{
        "type": "response_text",
        "text": "دنیا",
        "partial": True,
        "assistant_turn_id": "live:session-timing:1",
        "epoch": 1,
        "parent_user_turn_id": "live-utt:provider:7",
        "start_ms": 22760,
        "end_ms": 23100,
        "clock": "provider",
    }]


def test_legacy_response_text_shape_drops_unnegotiated_timing_fields():
    adapter = LiveClientAdapter("session-legacy")
    adapter.set_next_parent("live-utt:provider:7")
    assert adapter.provider_event({
        "type": "session.output_transcript.delta",
        "delta": "Hello",
        "start_ms": 100,
        "end_ms": 220,
    }) == [{"type": "response_text", "text": "Hello", "partial": True}]


def test_output_epoch_keeps_a_real_zero_start_across_later_deltas():
    adapter = LiveClientAdapter("session-zero", turn_timing=True)
    adapter.provider_event({
        "type": "session.output_transcript.delta", "delta": "first ",
        "start_ms": 0, "end_ms": 50,
    }, now=1.0)
    adapter.provider_event({
        "type": "session.output_transcript.delta", "delta": "second",
        "start_ms": 50, "end_ms": 100,
    }, now=1.1)
    output_id = adapter.output_id or ""
    _, closed = adapter.playback_drained(output_id, output_id, 100)
    assert closed is not None
    assert (closed.start_ms, closed.end_ms) == (0, 100)


def test_playback_ack_is_identity_fenced_and_interrupt_never_claims_done():
    adapter = LiveClientAdapter("session-2")
    adapter.provider_event({"type": "session.output_transcript.delta", "delta": "Old answer"})
    current = adapter.output_id
    assert adapter.playback_drained("live:session-2:999", current) == ([], None)
    assert adapter.playback_drained(current, "live:session-2:999") == ([], None)
    assert adapter.interrupt("live:session-2:999", current)[0] is False
    assert adapter.interrupt(current, "live:session-2:999")[0] is False
    ok, closed = adapter.interrupt(current, current, played_ms=1200)
    assert ok and closed is not None
    assert closed.interrupted is True and closed.played_ms == 1200
    # An interrupted segment is its own terminal event: the ack that follows
    # must not announce it a second time.
    assert adapter.playback_drained(current, current)[0] == [
        {"type": "state", "state": "listening"},
    ]


def test_an_epoch_the_client_never_acked_still_closes_and_returns_to_listening():
    # A1-03: the synthetic id used to be released ONLY by the native
    # SoundChunkPlayed(isFinal). One missed event fused an entire call into one
    # output id, with the caption and the half-duplex mic gate latched behind it.
    adapter = LiveClientAdapter("session-3", gap_ms=50)
    adapter.provider_event({"type": "session.output_transcript.delta", "delta": "First"}, now=100.0)
    first = adapter.output_id
    assert adapter.rotate_on_gap(now=100.02) == ([], None)   # inside the gap
    frames, closed = adapter.rotate_on_gap(now=100.5)
    assert closed is not None and closed.output_id == first
    assert [f["type"] for f in frames] == ["speech_segment_complete", "state"]
    assert frames[-1] == {"type": "state", "state": "listening"}

    adapter.provider_event({"type": "session.output_audio.delta", "delta": "SECOND"}, now=101.0)
    assert adapter.output_id != first
    # A late ack for the retired epoch completes the right segment and emits no
    # second caption for it.
    late, closed_late = adapter.playback_drained(first, first, played_ms=400)
    assert closed_late is not None and closed_late.played_ms == 400
    assert late == [{"type": "state", "state": "listening"}]


def test_output_text_never_spans_two_epochs():
    adapter = LiveClientAdapter("session-4", gap_ms=10)
    adapter.provider_event({"type": "session.output_transcript.delta", "delta": "one"}, now=1.0)
    _frames, first = adapter.rotate_on_gap(now=2.0)
    adapter.provider_event({"type": "session.output_transcript.delta", "delta": "two"}, now=2.0)
    _frames, second = adapter.playback_drained(adapter.output_id, adapter.output_id)
    assert first is not None and second is not None
    assert first.text == "one"
    assert second.text == "two"


def test_the_interrupt_fence_lifts_on_a_deadline_even_with_no_further_speech():
    # A1-04 / A8-7: `_resume_after_ms=None` made the resume gate's `or`
    # short-circuit true forever, so one interrupt could mute the assistant for
    # the rest of the session with a healthy socket and a UI reading Listening.
    adapter = LiveClientAdapter("session-5", suppress_ms=500)
    adapter.provider_event({"type": "session.output_transcript.delta", "delta": "Old",
                            "start_ms": 100, "end_ms": 300}, now=10.0)
    current = adapter.output_id
    ok, _closed = adapter.interrupt(current, current, resume_after_ms=900, now=10.0)
    assert ok and adapter.suppressed
    assert adapter.provider_event({"type": "session.output_audio.delta", "delta": "LATE"},
                                  now=10.1) == []
    # Full duplex: the next assistant transcript can legitimately be timestamped
    # BELOW the user's newest end_ms. That must delay the resume, never wedge it.
    assert adapter.provider_event({"type": "session.output_transcript.delta", "delta": "still old",
                                   "start_ms": 500, "end_ms": 600}, now=10.2) == []
    frames = adapter.provider_event({"type": "session.output_audio.delta", "delta": "NEW"},
                                    now=10.6)
    assert not adapter.suppressed
    assert frames[-1]["data"] == "NEW"


def test_the_interrupt_fence_lifts_on_new_output_past_the_user_watermark():
    adapter = LiveClientAdapter("session-6", suppress_ms=5000)
    adapter.provider_event({"type": "session.output_transcript.delta", "delta": "Old",
                            "start_ms": 100, "end_ms": 300}, now=1.0)
    current = adapter.output_id
    adapter.interrupt(current, current, resume_after_ms=700, now=1.0)
    assert adapter.provider_event({"type": "session.output_transcript.delta", "delta": "tail",
                                   "start_ms": 650, "end_ms": 690}, now=1.1) == []
    frames = adapter.provider_event({"type": "session.output_transcript.delta", "delta": "New",
                                     "start_ms": 701, "end_ms": 800}, now=1.2)
    assert frames == [{"type": "response_text", "text": "New", "partial": True}]
    assert adapter.output_id != current


def test_barge_in_fires_once_per_output_epoch_and_only_with_real_speech():
    adapter = LiveClientAdapter("session-7")
    assert adapter.bargein_due(500) is False          # nothing speaking
    adapter.provider_event({"type": "session.output_audio.delta", "delta": "A"})
    adapter.note_user_input(100, speech_ms=120, tokens=1)
    assert adapter.bargein_due(500, min_tokens=3) is False
    # R13.3.2: long enough is not enough on its own any more — the rule is
    # conjunctive, so time without words (or words without time) never fires.
    adapter.note_user_input(600, speech_ms=500, tokens=0)
    assert adapter.bargein_due(500, min_tokens=3) is False
    adapter.note_user_input(900, speech_ms=200, tokens=2)
    assert adapter.bargein_due(500, min_tokens=3) is True
    assert adapter.bargein_due(500, min_tokens=3) is False   # once per epoch


# ── usage ─────────────────────────────────────────────────────────────

def test_usage_is_cumulative_and_close_finalizes_without_double_counting():
    usage = LiveUsageTracker()
    assert usage.observe({"type": "session.usage.updated", "usage": {"seconds": 12}}) == 12
    assert usage.observe({"type": "session.usage.updated", "usage": {"seconds": 15}}) == 3
    assert usage.observe({"type": "session.usage.updated", "usage": {"seconds": 14}}) == 0
    assert usage.observe({"type": "session.closed", "usage": {"seconds": 16}}) == 1
    assert usage.seconds == 16
    assert usage.finalized


def test_the_provider_context_window_is_observed():
    usage = LiveUsageTracker()
    usage.observe({
        "type": "session.usage.updated",
        "usage": {"seconds": 5},
        "context_window": {"usage_ratio": 0.93},
    })
    assert usage.context_ratio == pytest.approx(0.93)


# ── chunking + language ───────────────────────────────────────────────

def test_an_ordinary_answer_is_one_commentary_event():
    # A8-10: the 407-character answer in recording 2 was split into two events
    # and paraphrased independently against a 500-TOKEN limit.
    answer = "A" * 407
    assert len(split_commentary(answer)) == 1
    assert estimate_tokens(answer) < 450


def test_a_long_answer_splits_on_sentence_boundaries_and_keeps_every_word():
    answer = " ".join(f"Sentence number {i} says something useful." for i in range(120))
    chunks = split_commentary(answer)
    assert len(chunks) > 1
    assert " ".join(chunks).split() == answer.split()
    assert all(estimate_tokens(c) <= 450 for c in chunks)


def test_persian_text_stays_under_the_token_ceiling():
    persian = "سلام دنیا. " * 200
    chunks = split_commentary(persian)
    assert all(estimate_tokens(c) <= 450 for c in chunks)
    assert "".join(c.replace(" ", "") for c in chunks) == persian.replace(" ", "")


def test_the_relay_speaks_the_callers_language_from_a_fixed_table():
    assert detect_reply_language("play some music") == "en"
    assert detect_reply_language("یه آهنگ بذار") == "fa"
    assert relay_line("delegation_failed", "fa") != relay_line("delegation_failed", "en")
    assert relay_line("media_playing", "fa", title="Halo").startswith("«Halo»")
    assert relay_line("nonexistent", "fa") == ""


def test_classifying_what_the_caller_said_during_running_work():
    running = "search the university of toronto computer science program"
    assert classify_followup("cancel that", running) == "cancel"
    assert classify_followup("cancel the status report", running) == "cancel"
    assert classify_followup("cancel another professor", running) == "cancel"
    assert classify_followup("بی‌خیال", running) == "cancel"
    assert classify_followup("ولش کن", running) == "cancel"
    # A bare "stop" is an ordinary word; only the order form cancels.
    assert classify_followup("stop that", running) == "cancel"
    assert classify_followup("stop the search", running) == "cancel"
    assert classify_followup("جستجو رو متوقف کن", running) == "cancel"
    assert classify_followup("actually make it masters", running) == "correction"
    assert classify_followup("نه، به جای آن", running) == "correction"
    assert classify_followup(
        "search the university of toronto computer science admissions", running,
    ) == "refinement"
    assert classify_followup("what is the weather in paris", running) == "unrelated"
    assert classify_followup("", running) == "new"


@pytest.mark.parametrize(
    "text",
    ["پیدا نکردی؟", "استاد برای یادگیری ماشین پیدا نکردی؟", "نتونستی پیدا کنی؟"],
)
def test_negative_persian_status_is_observation_not_task_mutation(text):
    running = "یه استاد برای یادگیری ماشین در دانشگاه تورنتو پیدا کن"
    assert classify_followup(text, running) == "status"


def test_followup_taxonomy_separates_continuation_refinement_and_unrelated():
    running = "find a machine learning professor at the university of toronto"
    assert classify_followup("another professor", running) == "continuation"
    assert classify_followup("دیگه چه؟", running) == "continuation"
    assert classify_followup("also only show associate professors", running) == "refinement"
    assert classify_followup(
        "find a machine learning professor at Toronto with recent publications", running,
    ) == "refinement"
    assert classify_followup("what is the weather in Paris", running) == "unrelated"


@pytest.mark.parametrize(
    ("text", "action"),
    [
        ("آهنگو عوض کن", "next"),
        ("آهنگ رو عوض کن", "next"),
        ("بعدی", "next"),
        ("بعدی رو بزن", "next"),
        ("لطفاً، بعدی رو بزن.", "next"),
        ("یکی دیگه پخش کن", "next"),
        ("قبلی", "previous"),
        ("آهنگ قبلی رو بزن", "previous"),
    ],
)
def test_required_persian_media_controls_take_the_deterministic_path(text, action):
    assert detect_media_control(text) == action


@pytest.mark.parametrize(
    "text", ["مرحله بعدی رو بزن", "هفته بعدی", "the next song is better"],
)
def test_media_control_does_not_capture_unrelated_mentions(text):
    assert detect_media_control(text) is None


def test_continuation_repeat_fence_detects_the_same_entity_not_just_same_prose():
    prior = "Professor Geoffrey Hinton is professor emeritus at the University of Toronto."
    assert continuation_repeats(prior, prior)
    assert continuation_repeats(
        "Another option is Geoffrey Hinton, a leading researcher in deep learning.",
        prior,
    )
    assert continuation_repeats(
        "Professor Hinton is another strong option.",
        prior,
    )
    assert continuation_repeats(
        "Hinton is another strong option.",
        prior,
    )
    assert not continuation_repeats(
        "Professor Jimmy Ba is an associate professor at the University of Toronto.",
        prior,
    )
    assert continuation_repeats(
        "استاد هینتون، گزینه دیگری است.",
        "استاد جفری هینتون، استاد بازنشسته دانشگاه تورنتو است.",
    )


def test_continuation_source_uses_subject_not_global_completion_order():
    professor = (
        "Find a machine learning professor at the University of Toronto.",
        "Professor Geoffrey Hinton is professor emeritus at U of T.",
    )
    weather = (
        "What is the weather in Paris?",
        "Paris is sunny and 22 degrees.",
    )
    rows = [professor, weather]
    assert select_continuation_source("another professor", rows) == professor
    assert select_continuation_source("another one", rows) == weather
    assert select_continuation_source(
        "another professor",
        [professor, ("Find a professor in Waterloo.", "Professor Alice Gao.")],
    ) is None


@pytest.mark.parametrize(
    "echo",
    [
        "Preserve every constraint from this prior request: find a professor",
        "Preserve every constraint from this prior",
        "Preserve every constraint from this",
        "Return a different primary entity/result.",
        "Prior caller",
        "Prior caller request",
        "Accepted backend",
        "Continuation contract",
        "Current caller",
        "Live-session context from earlier accepted",
        "Live-session context from earlier accepted delegations",
        "Your attempted result repeated the prior entity.",
        "Search again once and return a different primary entity.",
        "Search again once and return a different primary",
        "Search again once and return a different",
    ],
)
def test_partial_internal_scaffold_echoes_fail_closed(echo):
    assert sanitize_user_visible_answer(echo) == ""


@pytest.mark.parametrize(
    "answer",
    [
        "The current caller is Alice.",
        "The accepted backend is PostgreSQL.",
        "Accepted backend results are cached.",
        "A continuation contract was signed.",
        "The prior caller disconnected.",
        "Prior caller requested a refund.",
        "Current caller requests a callback.",
    ],
)
def test_scaffold_sanitizer_keeps_ordinary_prose(answer):
    assert sanitize_user_visible_answer(answer) == answer


def test_token_overlap_is_symmetric_on_the_shorter_side():
    assert token_overlap("alpha beta gamma", "alpha beta gamma") == pytest.approx(1.0)
    assert token_overlap("alpha beta", "") == 0.0


# ── end-to-end shape ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_full_live_relay_flow_with_fake_provider(monkeypatch):
    # The output-gap rotation is deliberately far away here: this test is about
    # the ACK path closing the epoch. The gap path has its own test.
    H.fast_clocks(monkeypatch, voice_live_output_epoch_gap_ms=3000)
    calls = {}

    async def think(user_id, task, session_id, relay=None, out=None, **kwargs):
        calls["task"] = task
        calls["display_request"] = kwargs.get("display_request")
        calls["reply_language"] = kwargs.get("reply_language")
        if out is not None:
            out["attachments"] = [{"id": "file-1"}]
        return "It is sunny.", "gpt-5.6-terra"

    recorded = H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("Find the weather", 100, 500))
            provider.push(H.delegation("delegation-1", 500))
        elif event["type"] == "session.commentary.append":
            provider.push(H.out_text("It is sunny.", 600, 900))
            provider.push(H.out_audio())
        elif event["type"] == "session.close":
            provider.push({"type": "session.usage.updated", "usage": {"seconds": 12}})
            provider.push(H.closed(seconds=12))

    client = H.FakeClient([
        H.config(voice="shimmer", tz="America/Toronto"),
        {"type": "audio_ready"},
        {"type": "audio", "data": "AAEC"},
        0.4,
        {"type": "playback_idle", "response_id": f"live:{H.PSID}:1",
         "item_id": f"live:{H.PSID}:1", "played_ms": 800},
        {"type": "screen_share_start"},
        0.05,
        {"type": "stop"},
    ])
    provider = H.FakeProvider(on_send=on_send)
    meters = []

    import app.services.live_voice_protocol as live

    async def fake_meter(*args, **kwargs):
        meters.append((args, kwargs))

    monkeypatch.setattr(live, "meter_live_session", fake_meter)

    await H.run_relay(client, provider, requested_voice="shimmer")

    start = provider.of("session.start")[0]
    assert start["session"]["model"] == "gpt-live-1"
    assert start["session"]["audio"]["output"]["voice"] == "marin"
    assert start["session"]["delegation"] == {"type": "client"}
    assert any(e["type"] == "session.input_audio.append" for e in provider.sent)
    assert provider.of("session.commentary.append")
    assert provider.of("session.close") == [
        {"type": "session.close", "event_id": "toup-live-stop"},
    ]

    ready = client.of("ready")[0]
    caps = ready["capabilities"]
    # The four values build 129 validates, unchanged.
    assert caps["voice_tasks"] is False
    assert caps["voice_provider"] == "live"
    assert caps["playback_ack"] is True
    assert caps["audio"] == {"type": "pcm16le", "rate": 24000}
    # Additive announcements.
    assert caps["live_turns"] is True
    assert caps["delegation_frames"] is True
    assert caps["media_fast_path"] is True

    assert client.of("audio_delta")
    complete = client.of("speech_segment_complete")[0]
    assert complete["provider_complete"] is False
    assert complete["durable"] is False
    assert complete["played_ms"] == 800
    screen_error = next(f for f in client.frames if f.get("code") == "live_screen_share_unavailable")
    assert screen_error["recoverable"] is True

    # The agent saw the caller's own words as the display request, with the
    # language named, and the message carries the directive as the fallback.
    assert calls["display_request"] == "Find the weather"
    assert calls["reply_language"] == "en"
    assert calls["task"].endswith("Reply in the caller's language.")

    user_rows = [s for s in recorded["saves"] if s.get("user_text")]
    answer_rows = [s for s in recorded["saves"] if s.get("assistant_text")]
    assert {r["user_text"] for r in user_rows} == {"Find the weather"}
    assert any(r["assistant_text"] == "It is sunny." for r in answer_rows)
    assert meters and meters[0][0][1] == H.PSID

    assert all(s in LIVE_STATES for s in client.states())
    seqs = [f["seq"] for f in client.of("state")]
    assert seqs == sorted(seqs)


def test_every_settings_name_the_relay_reads_actually_exists():
    """`getattr(settings, "voice_live_typo", 700)` is silent: the relay would
    run on the literal forever and nothing in this repo would notice."""

    import pathlib
    import re

    src = pathlib.Path("app/services/live_voice_protocol.py").read_text()
    names = set(re.findall(r'getattr\(\s*(?:self\.)?settings\s*,\s*"([a-z_0-9]+)"', src))
    names |= set(re.findall(r'(?:self\.)?settings\.([a-z_0-9]+)', src))
    assert names, "the probe itself must find something"
    missing = sorted(n for n in names if not hasattr(settings, n))
    assert missing == []


# ── Round 48 fix pass: the interrupt shapes that used to be dropped ───

def test_an_interrupt_with_no_identity_still_stops_the_current_epoch():
    """L1R-1. The shipped client sends `{type:'interrupt', played_ms}` with NO
    ids at all (the A2-05 fix: it must be sent even with no active identity).
    The adapter used to return (False, None) for it, and `on_interrupt` then
    armed no fence and sent no stop instruction — i.e. the model kept talking
    through the barge-in it was told about."""

    adapter = LiveClientAdapter("s-idless", suppress_ms=500)
    adapter.provider_event({"type": "session.output_transcript.delta", "delta": "Long answer"},
                           now=1.0)
    open_id = adapter.output_id
    ok, closed = adapter.interrupt(None, None, played_ms=400, now=1.0)
    assert ok is True
    assert closed is not None and closed.output_id == open_id
    assert closed.interrupted is True and closed.played_ms == 400
    assert adapter.suppressed


def test_an_interrupt_after_the_gap_rotation_still_stops_and_marks_that_epoch():
    """L1R-1. The provider streams audio faster than real time, so a 700 ms
    output gap fires while the phone still has seconds queued — the id the user
    barges in with is usually already retired. `playback_drained` searches
    `_history`; `interrupt` did not, and that asymmetry WAS the bug."""

    adapter = LiveClientAdapter("s-rotated", gap_ms=50, suppress_ms=500)
    adapter.provider_event({"type": "session.output_transcript.delta", "delta": "Answer"},
                           now=10.0)
    played = adapter.output_id
    _frames, retired = adapter.rotate_on_gap(now=10.5)
    assert retired is not None and retired.interrupted is False

    ok, closed = adapter.interrupt(played, played, played_ms=900, now=10.6)
    assert ok is True
    assert closed is retired
    assert closed.interrupted is True and closed.played_ms == 900
    assert adapter.suppressed


def test_a_disagreeing_interrupt_pair_names_no_epoch_and_arms_nothing():
    """The one shape that stays rejected: a response_id/item_id pair that
    disagrees is evidence of nothing, so it must not arm a 2.5 s output fence."""

    adapter = LiveClientAdapter("s-mismatch", suppress_ms=500)
    adapter.provider_event({"type": "session.output_transcript.delta", "delta": "Answer"},
                           now=1.0)
    current = adapter.output_id
    assert adapter.interrupt(current, "live:s-mismatch:999", now=1.0) == (False, None)
    assert not adapter.suppressed
    assert adapter.output_id == current


def test_the_relay_declares_only_features_it_consults():
    """C5.7. `FEATURE_STATE_SEQ` and `FEATURE_OUTCOMES` were defined and read by
    nothing, which reads as a behaviour switch that does not exist."""

    import pathlib
    import re

    src = pathlib.Path("app/services/live_voice_protocol.py").read_text()
    declared = set(re.findall(r"^(FEATURE_[A-Z_]+)\s*=", src, re.M))
    assert declared, "the probe itself must find something"
    for name in declared:
        # One definition plus at least one read.
        assert len(re.findall(rf"\b{name}\b", src)) >= 2, f"{name} is declared and never consulted"


def test_a_delegated_task_carries_no_revision_counter():
    """L1R-7. The shipped fence keyed on a revision that `add_transcript`
    bumped, so one backchannel discarded the answer. Keeping the field around
    'for bookkeeping' invites the next reader to believe the fence still reads
    it — `consumed` + `superseded` is the whole fence now."""

    from dataclasses import fields

    from app.services.live_voice_protocol import DelegatedTask

    assert "revision" not in {f.name for f in fields(DelegatedTask)}
    assert not hasattr(LiveDelegationTracker(), "revision")


def test_an_interrupt_naming_nothing_this_session_knows_still_arms_the_fence():
    """The client interrupts with an id the relay no longer holds anywhere —
    four epochs of history is about ten seconds. It still means "stop
    speaking", and the fence is the only thing that stops late audio from the
    interrupted utterance acquiring a fresh id and playing again."""

    adapter = LiveClientAdapter("s-unknown", suppress_ms=500)
    ok, closed = adapter.interrupt("live:elsewhere:7", "live:elsewhere:7", now=1.0)
    assert ok is True and closed is None
    assert adapter.suppressed
