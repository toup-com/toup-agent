"""High-value end-to-end contracts for the Live relay repair.

These tests deliberately use the same fake phone and provider as the rest of
the Live suite.  The unit tests cover the individual classifiers and frame
builders; this file pins the causal ordering that only exists in a full relay.
"""

import asyncio
import json
import time

import pytest

import test_live_harness as H


async def _wait_until(predicate, *, timeout=3.0, label="condition"):
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError(f"timed out waiting for {label}")
        await asyncio.sleep(0.01)


def _delegations(client, delegation_id):
    return [
        frame for frame in client.of("delegation")
        if frame.get("delegation_id") == delegation_id
    ]


def _terminal(client, delegation_id, phase="completed"):
    return any(
        frame.get("phase") == phase
        for frame in _delegations(client, delegation_id)
    )


@pytest.mark.asyncio
async def test_early_delegation_waits_for_the_complete_unfinished_persian_turn(monkeypatch):
    normal_gap_ms = 60
    hard_gap_ms = 260
    head_end_ms = 80
    tail_start_ms = 210
    assert normal_gap_ms < tail_start_ms - head_end_ms < hard_gap_ms
    H.fast_clocks(
        monkeypatch,
        voice_live_utterance_gap_ms=normal_gap_ms,
        voice_live_utterance_hard_gap_ms=hard_gap_ms,
        voice_live_delegation_ttl_s=2.0,
    )

    calls = []

    async def think(_user_id, task, _session_id, **kwargs):
        finals = [frame for frame in client.of("transcript") if frame["final"]]
        calls.append({
            "task": task,
            "display": kwargs.get("display_request"),
            "finals_at_dispatch": list(finals),
        })
        return "Professor B is a match.", "test-model"

    H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("یه استاد", 0, head_end_ms))
            # The provider delegates before the caller has supplied the tail.
            provider.push(H.delegation("d1", 100))
            provider.push(H.user_delta(
                " در دانشگاه تورنتو پیدا کن.", tail_start_ms, 520,
            ))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def wait_for_answer():
        await _wait_until(
            lambda: _terminal(client, "d1"), label="completed delegation d1",
        )

    client = H.FakeClient([H.config(), wait_for_answer, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(
        client, provider, timeout=7, db_session_id="db-repair-unfinished-fa",
    )

    finals = [frame for frame in client.of("transcript") if frame["final"]]
    assert len(finals) == 1
    assert finals[0]["text"] == "یه استاد در دانشگاه تورنتو پیدا کن."
    assert len(calls) == 1
    assert calls[0]["display"] == "یه استاد در دانشگاه تورنتو پیدا کن."
    assert calls[0]["finals_at_dispatch"] == [finals[0]]
    assert "یه استاد در دانشگاه تورنتو پیدا کن." in calls[0]["task"]


@pytest.mark.asyncio
async def test_direct_provider_answer_consumes_old_turn_before_later_delegation(monkeypatch):
    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    seen = []
    provider_box = {}

    async def think(_user_id, task, _session_id, **kwargs):
        seen.append((task, kwargs.get("display_request")))
        return "The delegated answer.", "test-model"

    H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        provider_box["provider"] = provider
        if event["type"] == "session.start":
            provider.push(H.user_delta("سلام.", 0, 200))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def direct_then_delegated():
        provider = provider_box["provider"]
        await _wait_until(
            lambda: any(f["final"] and f["text"] == "سلام." for f in client.of("transcript")),
            label="first finalized turn",
        )
        provider.push(H.out_audio())
        provider.push(H.out_text("Direct provider answer.", 500, 700))
        await _wait_until(lambda: bool(client.of("response_text")), label="direct answer")
        provider.push(H.user_delta("Find the Toronto weather forecast.", 1000, 1350))
        provider.push(H.delegation("d2", 1400))
        await _wait_until(
            lambda: _terminal(client, "d2"), label="completed delegation d2",
        )

    client = H.FakeClient([H.config(), direct_then_delegated, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(
        client, provider, timeout=8, db_session_id="db-repair-direct-causality",
    )

    finals = [frame for frame in client.of("transcript") if frame["final"]]
    assert [frame["text"] for frame in finals] == [
        "سلام.", "Find the Toronto weather forecast.",
    ]
    first_turn, second_turn = [frame["turn_id"] for frame in finals]
    response = client.of("response_text")[0]
    assert response["parent_user_turn_id"] == first_turn
    assert client.of("audio_delta")[0]["response_id"] == response["assistant_turn_id"]
    d2 = [frame for frame in _delegations(client, "d2") if "state" in frame]
    assert d2 and all(frame["parent_user_turn_id"] == second_turn for frame in d2)
    assert len(seen) == 1
    assert seen[0][1] == "Find the Toronto weather forecast."
    assert "Find the Toronto weather forecast." in seen[0][0]


@pytest.mark.asyncio
async def test_zero_offset_provider_greeting_does_not_consume_a_later_user_turn(monkeypatch):
    H.fast_clocks(
        monkeypatch,
        voice_live_delegation_ttl_s=2.0,
        voice_live_output_epoch_gap_ms=5000,
    )
    seen = []
    provider_box = {}

    async def think(_user_id, task, _session_id, **kwargs):
        seen.append((task, kwargs.get("display_request")))
        return "The delegated answer.", "test-model"

    H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        provider_box["provider"] = provider
        if event["type"] == "session.start":
            # The greeting is physically delivered before the later request,
            # not merely timestamped before it. Audio deliberately arrives
            # first to exercise late text causality while the old epoch stays
            # open across the new request.
            provider.push(H.out_audio("GREETING"))
            provider.push(H.out_text("Welcome.", 0, 0))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def greeting_then_delegation():
        provider = provider_box["provider"]
        await _wait_until(
            lambda: bool(client.of("response_text")), label="zero-offset greeting",
        )
        assert client.of("response_text")[0]["parent_user_turn_id"] is None
        provider.push(H.user_delta("Find the Toronto forecast.", 100, 200))
        provider.push(H.delegation("d-after-greeting", 250))
        await _wait_until(
            lambda: _terminal(client, "d-after-greeting"),
            label="delegation after greeting",
        )

    client = H.FakeClient([
        H.config(), greeting_then_delegation, {"type": "stop"},
    ])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(
        client,
        provider,
        timeout=7,
        db_session_id="db-repair-zero-offset-causality",
    )

    assert len(seen) == 1
    assert seen[0][0].startswith("Find the Toronto forecast.")
    assert seen[0][1] == "Find the Toronto forecast."


@pytest.mark.asyncio
async def test_single_fragment_ceiling_dispatches_pending_before_raced_output(monkeypatch):
    H.fast_clocks(
        monkeypatch,
        voice_live_utterance_max_chars=5,
        voice_live_delegation_ttl_s=2.0,
    )
    seen = []

    async def think(_user_id, task, _session_id, **kwargs):
        seen.append((task, kwargs.get("display_request")))
        return "The delegated answer.", "test-model"

    H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.delegation("d-ceiling", 20))
            provider.push(H.user_delta("oversized request", 10, 20))
            provider.push(H.out_text("Acknowledged.", 21, 30))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def wait_for_result():
        await _wait_until(
            lambda: _terminal(client, "d-ceiling"),
            label="ceiling-bound delegation",
        )

    client = H.FakeClient([
        H.config(), wait_for_result, {"type": "stop"},
    ])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(
        client,
        provider,
        timeout=7,
        db_session_id="db-repair-single-fragment-ceiling",
    )

    assert len(seen) == 1
    assert seen[0][0].startswith("oversized request")
    assert seen[0][1] == "oversized request"


@pytest.mark.asyncio
async def test_open_newer_turn_blocks_old_closed_turn_from_stealing_delegation(monkeypatch):
    """A delegation created inside turn 2 must wait for turn 2's endpoint."""

    H.fast_clocks(
        monkeypatch,
        voice_live_utterance_gap_ms=250,
        voice_live_utterance_hard_gap_ms=500,
        voice_live_delegation_ttl_s=2.0,
        # Spec v0.3 §B: an UNANSWERED closed turn within the request-span gap
        # is part of the next delegation's request. This test is about the
        # OLD turn never being bound on its own, so it sits outside the span.
        voice_live_request_span_max_gap_ms=500,
    )
    calls = []
    provider_box = {}

    async def think(_user_id, task, _session_id, **kwargs):
        calls.append((task, kwargs.get("display_request")))
        return "Professor A is a match.", "test-model"

    H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        provider_box["provider"] = provider
        if event["type"] == "session.start":
            # Leave this finalized turn unconsumed.  It is the tempting but
            # causally wrong FIFO candidate when the next delegation arrives.
            provider.push(H.user_delta("Hello.", 0, 100))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def send_second_turn():
        await _wait_until(
            lambda: any(frame["final"] for frame in client.of("transcript")),
            label="first finalized turn",
        )
        provider = provider_box["provider"]
        provider.push(H.user_delta("Find a professor at U of T.", 1000, 1100))
        provider.push(H.delegation("d-new", 1150))
        await _wait_until(
            lambda: _terminal(client, "d-new"),
            label="newer-turn delegation completion",
        )

    client = H.FakeClient([H.config(), send_second_turn, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(
        client, provider, timeout=8, db_session_id="db-repair-open-causality",
    )

    finals = [frame for frame in client.of("transcript") if frame["final"]]
    assert [frame["text"] for frame in finals] == [
        "Hello.", "Find a professor at U of T.",
    ]
    assert len(calls) == 1
    assert calls[0][1] == "Find a professor at U of T."
    assert calls[0][0].startswith("Find a professor at U of T.")
    assert "Hello." not in calls[0][0]
    lifecycle = [
        frame for frame in _delegations(client, "d-new") if "state" in frame
    ]
    assert lifecycle
    assert all(
        frame["parent_user_turn_id"] == finals[1]["turn_id"]
        for frame in lifecycle
    )


@pytest.mark.asyncio
async def test_deferred_provider_ack_keeps_parent_after_delegation_consumes_turn(monkeypatch):
    """Output buffered behind an open delegated turn retains that turn's ID.

    Dispatch runs before the deferred-output flush, so resolving causality from
    only unconsumed turns at flush time used to produce ``parent: null`` here.
    """

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    release = asyncio.Event()

    async def think(_user_id, _task, _session_id, **_kwargs):
        await release.wait()
        return "The delegated answer.", "test-model"

    H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("Find a professor at U of T.", 0, 500))
            provider.push(H.delegation("d-parent", 550))
            # This immediate provider acknowledgement races the guarded turn
            # endpoint and is therefore held until the utterance is final.
            # Audio-first delivery has no timestamp; the later text delta must
            # bind the same epoch rather than splitting or losing its parent.
            provider.push(H.out_audio())
            provider.push(H.out_text("I will look.", 560, 700))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def observe_then_finish():
        await _wait_until(
            lambda: bool(client.of("response_text")),
            label="deferred provider acknowledgement",
        )
        release.set()
        await _wait_until(
            lambda: _terminal(client, "d-parent"),
            label="delegated task completion",
        )

    client = H.FakeClient([H.config(), observe_then_finish, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(
        client, provider, timeout=8, db_session_id="db-repair-deferred-parent",
    )

    final = next(frame for frame in client.of("transcript") if frame["final"])
    acknowledgement = client.of("response_text")[0]
    assert acknowledgement["text"] == "I will look."
    assert acknowledgement["parent_user_turn_id"] == final["turn_id"]


@pytest.mark.asyncio
async def test_english_acknowledgement_before_same_turn_delegation_dispatches_once(
    monkeypatch,
):
    H.fast_clocks(
        monkeypatch,
        voice_live_delegation_ttl_s=0.20,
        voice_live_output_epoch_gap_ms=5000,
    )
    calls = []
    provider_box = {}

    async def think(_user_id, task, _session_id, **kwargs):
        calls.append((task, kwargs))
        return "Professor A is a match.", "test-model"

    H.patch_relay(monkeypatch, think=think)

    request = (
        "Find an Iranian mechanical engineering professor at the "
        "University of Toronto."
    )

    def on_send(provider, event):
        provider_box["provider"] = provider
        if event["type"] == "session.start":
            provider.push(H.user_delta(request, 0, 200))
            provider.push(H.out_text("I will look for that professor.", 250, 350))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def delegate_after_acknowledgement():
        await _wait_until(
            lambda: bool(client.of("response_text")),
            label="English provider acknowledgement",
        )
        provider_box["provider"].push(H.delegation("search-after-ack", 360))
        await _wait_until(
            lambda: _terminal(client, "search-after-ack"),
            label="English post-ack delegation",
        )

    client = H.FakeClient([
        H.config(), delegate_after_acknowledgement, {"type": "stop"},
    ])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(
        client, provider, timeout=7, db_session_id="db-repair-ack-before-delegation-en",
    )

    finals = [frame for frame in client.of("transcript") if frame["final"]]
    assert len(finals) == 1
    assert client.of("response_text")[0]["parent_user_turn_id"] == finals[0]["turn_id"]
    assert len(calls) == 1
    assert calls[0][1]["display_request"] == request
    assert request in calls[0][0]
    lifecycle = [
        frame for frame in _delegations(client, "search-after-ack")
        if "state" in frame
    ]
    assert [frame["state"] for frame in lifecycle] == [
        "queued", "running", "completed",
    ]
    assert all(frame["parent_user_turn_id"] == finals[0]["turn_id"] for frame in lifecycle)
    assert not any(frame["phase"] == "expired" for frame in lifecycle)


@pytest.mark.asyncio
async def test_persian_acknowledgement_before_same_turn_delegation_dispatches_once(
    monkeypatch,
):
    H.fast_clocks(
        monkeypatch,
        voice_live_delegation_ttl_s=0.20,
        voice_live_output_epoch_gap_ms=5000,
    )
    calls = []
    provider_box = {}

    async def think(_user_id, task, _session_id, **kwargs):
        calls.append((task, kwargs))
        return "استاد الف مناسب است.", "test-model"

    H.patch_relay(monkeypatch, think=think)
    request = "یک استاد مهندسی مکانیک در دانشگاه تورنتو پیدا کن."

    def on_send(provider, event):
        provider_box["provider"] = provider
        if event["type"] == "session.start":
            provider.push(H.user_delta(request, 0, 200))
            provider.push(H.out_text("حتماً، برات پیدا می‌کنم.", 250, 350))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def delegate_after_acknowledgement():
        await _wait_until(
            lambda: bool(client.of("response_text")),
            label="Persian provider acknowledgement",
        )
        provider_box["provider"].push(H.delegation("search-after-ack-fa", 360))
        await _wait_until(
            lambda: _terminal(client, "search-after-ack-fa"),
            label="Persian post-ack delegation",
        )

    client = H.FakeClient([
        H.config(), delegate_after_acknowledgement, {"type": "stop"},
    ])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(
        client, provider, timeout=7, db_session_id="db-repair-ack-before-delegation-fa",
    )

    finals = [frame for frame in client.of("transcript") if frame["final"]]
    assert len(finals) == 1
    assert client.of("response_text")[0]["parent_user_turn_id"] == finals[0]["turn_id"]
    assert len(calls) == 1
    assert calls[0][1]["display_request"] == request
    assert calls[0][1]["reply_language"] == "fa"
    assert request in calls[0][0]
    lifecycle = [
        frame for frame in _delegations(client, "search-after-ack-fa")
        if "state" in frame
    ]
    assert [frame["state"] for frame in lifecycle] == [
        "queued", "running", "completed",
    ]
    assert all(frame["parent_user_turn_id"] == finals[0]["turn_id"] for frame in lifecycle)


@pytest.mark.asyncio
async def test_acknowledgement_delegation_id_replay_runs_once(monkeypatch):
    H.fast_clocks(
        monkeypatch,
        voice_live_delegation_ttl_s=0.20,
        voice_live_output_epoch_gap_ms=5000,
    )
    calls = []
    provider_box = {}

    async def think(*args, **kwargs):
        calls.append((args, kwargs))
        return "One result.", "test-model"

    H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        provider_box["provider"] = provider
        if event["type"] == "session.start":
            provider.push(H.user_delta("Find one professor.", 0, 200))
            provider.push(H.out_text("I will look.", 250, 350))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def replay_before_and_after_completion():
        await _wait_until(
            lambda: bool(client.of("response_text")),
            label="provider acknowledgement before replay",
        )
        provider = provider_box["provider"]
        provider.push(H.delegation("same-delegation", 360))
        provider.push(H.delegation("same-delegation", 360))
        await _wait_until(
            lambda: _terminal(client, "same-delegation"),
            label="first replayed delegation completion",
        )
        provider.push(H.delegation("same-delegation", 360))
        await asyncio.sleep(0.05)

    client = H.FakeClient([
        H.config(), replay_before_and_after_completion, {"type": "stop"},
    ])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(
        client, provider, timeout=7, db_session_id="db-repair-ack-delegation-replay",
    )

    assert len(calls) == 1
    frames = _delegations(client, "same-delegation")
    lifecycle = [frame for frame in frames if "state" in frame]
    assert [frame["state"] for frame in lifecycle] == [
        "queued", "running", "completed",
    ]
    assert [frame["task_revision"] for frame in lifecycle] == [1, 2, 3]
    assert not any(frame["phase"] in {"expired", "failed"} for frame in frames)


@pytest.mark.asyncio
async def test_a_playback_acked_direct_answer_executes_at_most_once(monkeypatch):
    """RE-AIMED from test_retired_direct_answer_fences_its_turn_from_later_delegation.

    That test asserted ZERO executions for a first-ever delegation arriving
    after the phone acknowledged full playback of a direct answer.  Independent
    review reproduced the same wire sequence carrying a LEGITIMATE search the
    caller had asked for, and showed it being silently discarded: a client's
    `playback_idle` proves the audio finished, never that the request was
    fulfilled.  The two sequences are byte-for-byte identical and the provider
    supplies no response-level provenance, so zero-execution is not an
    enforceable invariant — only single execution is.

    This test therefore pins what the relay CAN guarantee, and the anti-duplicate
    half is asserted harder than before: the turn runs once, a replay of the same
    delegation id runs nothing, and a DIFFERENT later delegation id cannot run it
    again either, because the turn is consumed.

    See /private/tmp/toup-voice-blockerA-decision.md.
    """

    H.fast_clocks(
        monkeypatch,
        voice_live_delegation_ttl_s=0.6,
        voice_live_output_epoch_gap_ms=5000,
    )
    calls = []
    provider_box = {}

    async def think(*args, **kwargs):
        calls.append(kwargs.get("display_request"))
        return "The answer.", "test-model"

    H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        provider_box["provider"] = provider
        if event["type"] == "session.start":
            provider.push(H.user_delta("Find me a professor.", 0, 200))
            provider.push(H.out_text("I will look for that professor.", 250, 350))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def arm_playback_ack():
        await _wait_until(
            lambda: bool(client.of("response_text")),
            label="direct answer reaching the phone",
        )
        client._script.insert(0, {
            "type": "playback_idle",
            "response_id": f"live:{H.PSID}:1",
            "item_id": f"live:{H.PSID}:1",
            "played_ms": 600,
        })

    def delegate_once():
        provider_box["provider"].push(H.delegation("first-delegation", 360))

    async def replay_and_then_a_different_id():
        await _wait_until(
            lambda: _terminal(client, "first-delegation"),
            timeout=4.0,
            label="the one legitimate execution",
        )
        provider = provider_box["provider"]
        provider.push(H.delegation("first-delegation", 360))     # exact replay
        provider.push(H.delegation("second-delegation", 370))    # different id
        await _wait_until(
            lambda: _terminal(client, "second-delegation", "expired"),
            timeout=4.0,
            label="the consumed turn refusing a second delegation",
        )

    client = H.FakeClient([
        H.config(), arm_playback_ack, delegate_once,
        replay_and_then_a_different_id, {"type": "stop"},
    ])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(
        client, provider, timeout=9, db_session_id="db-repair-at-most-once",
    )

    final = next(frame for frame in client.of("transcript") if frame["final"])
    # Executed exactly once, for the turn the caller actually spoke.
    assert calls == ["Find me a professor."], (
        f"the turn executed {len(calls)} times, not once: {calls!r}"
    )
    # The replay produced no second task and no second execution.
    lifecycle = [
        frame for frame in _delegations(client, "first-delegation") if "state" in frame
    ]
    assert [frame["state"] for frame in lifecycle] == ["queued", "running", "completed"]
    assert all(
        frame["parent_user_turn_id"] == final["turn_id"] for frame in lifecycle
    )
    # A different delegation id cannot re-execute a turn a delegation consumed.
    terminal = [
        frame for frame in _delegations(client, "second-delegation")
        if frame.get("phase") == "expired" and "state" in frame
    ]
    assert len(terminal) == 1
    assert terminal[0]["reason"] == "no_causal_turn"
    assert terminal[0]["parent_user_turn_id"] == ""


@pytest.mark.asyncio
async def test_deferred_outputs_for_consecutive_turns_mint_distinct_epochs(monkeypatch):
    H.fast_clocks(
        monkeypatch,
        voice_live_output_epoch_gap_ms=5000,
        voice_live_utterance_gap_ms=80,
        voice_live_utterance_hard_gap_ms=160,
    )
    H.patch_relay(monkeypatch)
    provider_box = {}

    def on_send(provider, event):
        provider_box["provider"] = provider
        if event["type"] == "session.start":
            provider.push(H.user_delta("First question.", 0, 100))
            provider.push(H.out_text("First answer.", 200, 300))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def send_second_turn():
        await _wait_until(
            lambda: len(client.of("response_text")) == 1,
            label="first deferred response",
        )
        provider = provider_box["provider"]
        provider.push(H.user_delta("Second question.", 1000, 1100))
        provider.push(H.out_text("Second answer.", 1200, 1300))
        await _wait_until(
            lambda: len(client.of("response_text")) == 2,
            label="second deferred response",
        )

    client = H.FakeClient([H.config(), send_second_turn, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(
        client, provider, timeout=7, db_session_id="db-repair-two-epochs",
    )

    finals = [frame for frame in client.of("transcript") if frame["final"]]
    responses = client.of("response_text")
    assert [frame["text"] for frame in responses] == [
        "First answer.", "Second answer.",
    ]
    assert responses[0]["assistant_turn_id"] != responses[1]["assistant_turn_id"]
    assert [frame["parent_user_turn_id"] for frame in responses] == [
        finals[0]["turn_id"], finals[1]["turn_id"],
    ]


@pytest.mark.asyncio
async def test_two_buffered_audio_first_responses_rotate_before_second_audio(monkeypatch):
    H.fast_clocks(
        monkeypatch,
        voice_live_output_epoch_gap_ms=5000,
        voice_live_utterance_gap_ms=80,
        voice_live_utterance_hard_gap_ms=160,
    )
    H.patch_relay(monkeypatch)

    def on_send(provider, event):
        if event["type"] == "session.start":
            # Both replies remain buffered until turn two reaches its endpoint.
            # Each audio chunk arrives before the timestamped text that proves
            # its parent; audio B must never leak into response A's epoch.
            provider.push(H.user_delta("First question.", 0, 100))
            provider.push(H.out_audio("AUDIO-A"))
            provider.push(H.out_text("First answer.", 200, 300))
            provider.push(H.user_delta("Second question.", 1000, 1100))
            provider.push(H.out_audio("AUDIO-A-TAIL"))
            provider.push(H.out_text(" Still first.", 1050, 1080))
            provider.push(H.out_audio("AUDIO-B"))
            provider.push(H.out_text("Second answer.", 1200, 1300))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def wait_for_both():
        await _wait_until(
            lambda: len(client.of("response_text")) >= 3,
            label="two audio-first buffered responses",
        )

    client = H.FakeClient([H.config(), wait_for_both, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(
        client,
        provider,
        timeout=7,
        db_session_id="db-repair-two-audio-first-epochs",
    )

    finals = [frame for frame in client.of("transcript") if frame["final"]]
    responses = client.of("response_text")
    audio = client.of("audio_delta")
    assert len(finals) == 2
    assert len(responses) == len(audio) == 3
    assert [frame["parent_user_turn_id"] for frame in responses] == [
        finals[0]["turn_id"], finals[0]["turn_id"], finals[1]["turn_id"],
    ]
    assert responses[0]["assistant_turn_id"] == responses[1]["assistant_turn_id"]
    assert responses[1]["assistant_turn_id"] != responses[2]["assistant_turn_id"]
    assert [frame["response_id"] for frame in audio] == [
        responses[0]["assistant_turn_id"],
        responses[1]["assistant_turn_id"],
        responses[2]["assistant_turn_id"],
    ]


@pytest.mark.asyncio
async def test_mismatched_interrupt_is_a_session_level_no_op():
    from app.services.live_voice_protocol import (
        LiveClientAdapter,
        LiveDelegationTracker,
        _LiveSession,
    )

    adapter = LiveClientAdapter("provider-mismatch", suppress_ms=500)
    adapter.provider_event(H.out_audio("OPEN"), now=0.0)
    adapter.note_user_input(400, speech_ms=400, tokens=2)
    adapter.note_user_input(800, speech_ms=400, tokens=2)
    assert adapter.bargein_due(600, 3, now=1.0, rearm_ms=1500)
    current = adapter.output_id

    session = _LiveSession.__new__(_LiveSession)
    session.adapter = adapter
    session.tracker = LiveDelegationTracker()
    session.provider_now_ms = 1000
    session.provider_now_at = time.monotonic()
    held = {"type": "session.output_audio.delta", "delta": "HELD"}
    session.deferred_output = [held]
    session.deferred_output_bytes = 4
    session.deferred_output_overflowed = False

    await session.on_interrupt({
        "type": "interrupt",
        "response_id": current,
        "item_id": "live:provider-mismatch:999",
        "played_ms": 100,
    })

    assert session.deferred_output == [held]
    assert session.deferred_output_bytes == 4
    assert adapter.output_id == current
    assert not adapter.suppressed
    adapter.note_user_input(1200, speech_ms=400, tokens=2)
    adapter.note_user_input(1600, speech_ms=400, tokens=2)
    assert adapter.bargein_due(600, 3, now=3.0, rearm_ms=1500)


@pytest.mark.asyncio
async def test_queued_commentary_claim_does_not_split_an_open_output_epoch():
    from app.services.live_voice_protocol import (
        LiveClientAdapter,
        LiveDelegationTracker,
        _LiveSession,
    )

    session = _LiveSession.__new__(_LiveSession)
    session.adapter = LiveClientAdapter("provider-claim-fifo", turn_timing=True)
    session.tracker = LiveDelegationTracker()
    session.utterances = None
    session.tick_wake = asyncio.Event()
    session.provider_now_ms = 0
    session.provider_now_at = time.monotonic()
    emitted = []

    async def emit_frames(frames):
        emitted.extend(frames)

    async def noop(*_args, **_kwargs):
        return None

    session.emit_frames = emit_frames
    session.settle_bound_claims = noop
    session.persist_spoken = noop
    session.adapter.claim_next_output({"parent_user_turn_id": "u1"})
    session.adapter.claim_next_output({"parent_user_turn_id": "u2"})

    await session.on_provider_output(H.out_text("first ", 100, 150))
    await session.on_provider_output(H.out_text("answer", 150, 200))

    responses = [frame for frame in emitted if frame.get("type") == "response_text"]
    assert [frame["text"] for frame in responses] == ["first ", "answer"]
    assert len({frame["assistant_turn_id"] for frame in responses}) == 1
    assert {frame["parent_user_turn_id"] for frame in responses} == {"u1"}
    assert not any(
        frame.get("type") == "speech_segment_complete" for frame in emitted
    )
    assert session.adapter.next_claim_parent() == "u2"


@pytest.mark.asyncio
async def test_exact_provider_replays_cannot_hold_the_silence_endpoint_open(monkeypatch):
    H.fast_clocks(
        monkeypatch,
        voice_live_utterance_gap_ms=60,
        voice_live_utterance_hard_gap_ms=120,
    )
    H.patch_relay(monkeypatch)
    provider_box = {}
    replay_finished = asyncio.Event()

    def on_send(provider, event):
        provider_box["provider"] = provider
        if event["type"] == "session.start":
            provider.push(H.user_delta("Hello.", 0, 200))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def replay_while_waiting_for_endpoint():
        provider = provider_box["provider"]

        async def replay():
            deadline = asyncio.get_running_loop().time() + 0.60
            while asyncio.get_running_loop().time() < deadline:
                provider.push(H.user_delta("Hello.", 0, 200))
                await asyncio.sleep(0.015)
            replay_finished.set()

        replay_task = asyncio.create_task(replay())
        await _wait_until(
            lambda: any(frame["final"] for frame in client.of("transcript")),
            timeout=0.45,
            label="endpoint despite exact provider retries",
        )
        assert not replay_finished.is_set(), (
            "the endpoint appeared only after retries stopped; duplicate input "
            "must not refresh the provider clock"
        )
        await replay_task

    client = H.FakeClient([
        H.config(), replay_while_waiting_for_endpoint, {"type": "stop"},
    ])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(
        client, provider, timeout=5, db_session_id="db-repair-replay-clock",
    )

    finals = [frame for frame in client.of("transcript") if frame["final"]]
    assert len(finals) == 1
    assert finals[0]["text"] == "Hello."


@pytest.mark.asyncio
async def test_expired_delegation_id_is_tombstoned_against_provider_replay(monkeypatch):
    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=0.12)
    calls = []
    provider_box = {}

    async def think(*args, **kwargs):
        calls.append((args, kwargs))
        return "This must not run.", "test-model"

    H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        provider_box["provider"] = provider
        if event["type"] == "session.start":
            provider.push(H.delegation("replayed-orphan", 10))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def replay_after_expiry():
        await _wait_until(
            lambda: _terminal(client, "replayed-orphan", "expired"),
            label="orphan expiry",
        )
        provider = provider_box["provider"]
        provider.push(H.user_delta("A later real request.", 1000, 1200))
        provider.push(H.delegation("replayed-orphan", 1250))
        await _wait_until(
            lambda: any(frame["final"] for frame in client.of("transcript")),
            label="later finalized turn",
        )
        await asyncio.sleep(0.2)

    client = H.FakeClient([H.config(), replay_after_expiry, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(
        client, provider, timeout=6, db_session_id="db-repair-expired-replay",
    )

    assert calls == []
    assert [
        frame["phase"] for frame in _delegations(client, "replayed-orphan")
        if frame["phase"] in {"completed", "failed", "cancelled", "expired"}
    ] == ["expired"]


@pytest.mark.asyncio
async def test_assistant_turn_ids_are_connection_scoped_across_reconnects(monkeypatch):
    H.fast_clocks(monkeypatch, voice_live_output_epoch_gap_ms=5000)
    H.patch_relay(monkeypatch)

    async def run_once(provider_session_id):
        def on_send(provider, event):
            if event["type"] == "session.start":
                provider.push(H.out_text("Hello from this connection.", 10, 100))
            elif event["type"] == "session.close":
                provider.push(H.closed())

        async def wait_for_output():
            await _wait_until(
                lambda: bool(client.of("response_text")),
                label=f"output for {provider_session_id}",
            )

        client = H.FakeClient([
            H.config(), wait_for_output, {"type": "stop"},
        ])
        provider = H.FakeProvider(
            on_send=on_send, session_id=provider_session_id,
        )
        await H.run_relay(
            client,
            provider,
            timeout=5,
            db_session_id="same-day-conversation",
        )
        return client.of("response_text")[0]["assistant_turn_id"]

    first = await run_once("provider-connection-a")
    second = await run_once("provider-connection-b")

    assert first == "live:provider-connection-a:1"
    assert second == "live:provider-connection-b:1"
    assert first != second


@pytest.mark.asyncio
async def test_late_distinct_input_timestamp_cannot_regress_provider_clock():
    from app.config import settings
    from app.services.live_voice_protocol import (
        LiveDelegationTracker,
        UtteranceAssembler,
        _LiveSession,
    )

    session = _LiveSession.__new__(_LiveSession)
    session.settings = settings
    session.tracker = LiveDelegationTracker()
    session.utterances = UtteranceAssembler("provider-clock-test")
    session.adapter = None
    session.settle_deadline = 0.0
    session.tick_wake = asyncio.Event()
    session.provider_now_ms = 100
    session.provider_now_at = time.monotonic() - 2.0

    before = session.provider_clock_ms()
    await session.on_input_transcript(H.user_delta("later", 100, 150))
    after = session.provider_clock_ms()

    assert before >= 2000
    assert after >= before


@pytest.mark.asyncio
async def test_negative_persian_status_does_not_redispatch_or_cancel_running_work(monkeypatch):
    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    started = asyncio.Event()
    release = asyncio.Event()
    calls = []
    provider_box = {}

    async def think(_user_id, task, _session_id, **kwargs):
        calls.append((task, kwargs.get("delegation_id")))
        started.set()
        await release.wait()
        return "Professor A is the result.", "test-model"

    H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        provider_box["provider"] = provider
        if event["type"] == "session.start":
            provider.push(H.user_delta(
                "یه استاد یادگیری ماشین در دانشگاه تورنتو پیدا کن.", 0, 600,
            ))
            provider.push(H.delegation("d1", 650))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def ask_for_status():
        await asyncio.wait_for(started.wait(), timeout=3)
        provider = provider_box["provider"]
        provider.push(H.user_delta("پیدا نکردی؟", 1200, 1450))
        provider.push(H.delegation("d2", 1500))
        # The status answer is the spoken line (on the RUNNING task's id), not
        # a d2 card — see the inverted assertion below.
        await _wait_until(
            lambda: any(
                event.get("delegation_id") == "d1"
                for event in provider.of("session.commentary.append")
            ),
            label="status answer",
        )
        release.set()
        await _wait_until(
            lambda: _terminal(client, "d1"), label="original delegation completion",
        )

    client = H.FakeClient([H.config(), ask_for_status, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(
        client, provider, timeout=8, db_session_id="db-repair-status-fa",
    )

    assert len(calls) == 1
    assert calls[0][1] == "d1"
    d1_phases = [frame["phase"] for frame in _delegations(client, "d1")]
    assert d1_phases[-1] == "completed"
    assert not ({"cancelled", "superseded"} & set(d1_phases))
    # INVERTED in R2 §C. This pinned d2 closing `completed` with
    # `reason: status_answered` — a task card for a status question the relay
    # answered itself, drawn by the app as a Done card with no body
    # (production V1+04:14). The question is answered in speech; it is no job.
    assert _delegations(client, "d2") == []


@pytest.mark.asyncio
async def test_negotiated_task_and_tool_lifecycle_uses_only_actual_tool_events(monkeypatch):
    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    intent_seen = asyncio.Event()
    allow_start = asyncio.Event()
    snapshot = {}

    async def think(_user_id, _task, _session_id, relay=None, **_kwargs):
        await relay.on_event({"type": "tool.intent", "name": "web_search"})
        intent_seen.set()
        await allow_start.wait()
        await relay.on_event({
            "type": "tool.start",
            "call_id": "search-1",
            "name": "web_search",
            "args": {"query": "University of Toronto professor"},
        })
        await relay.on_event({
            "type": "tool.end",
            "call_id": "search-1",
            "name": "web_search",
            "ok": True,
            "preview": "found",
            "elapsed_ms": 40,
        })
        return "Professor A is a match.", "test-model"

    H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("Find a professor at U of T.", 0, 500))
            provider.push(H.delegation("d1", 550))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def inspect_intent_then_run_tool():
        await asyncio.wait_for(intent_seen.wait(), timeout=3)
        await asyncio.sleep(0.05)
        snapshot["delegations"] = list(_delegations(client, "d1"))
        snapshot["tool_started"] = list(client.of("tool_call.started"))
        snapshot["tool_completed"] = list(client.of("tool_call.completed"))
        allow_start.set()
        await _wait_until(
            lambda: _terminal(client, "d1"), label="tool-backed delegation completion",
        )

    client = H.FakeClient([H.config(), inspect_intent_then_run_tool, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(
        client, provider, timeout=8, db_session_id="db-repair-lifecycle",
    )

    assert snapshot["tool_started"] == []
    assert snapshot["tool_completed"] == []
    assert [frame["state"] for frame in snapshot["delegations"]] == [
        "queued", "running",
    ]

    all_delegations = _delegations(client, "d1")
    lifecycle = [frame for frame in all_delegations if "state" in frame]
    assert [frame["state"] for frame in lifecycle] == [
        "queued", "running", "waiting_for_tool", "running", "completed",
    ]
    assert [frame["task_revision"] for frame in lifecycle] == [1, 2, 3, 4, 5]
    assert len({frame["task_id"] for frame in lifecycle}) == 1
    parent_turn = lifecycle[0]["parent_user_turn_id"]
    assert parent_turn and all(
        frame["parent_user_turn_id"] == parent_turn for frame in lifecycle
    )
    # A historical delivery-verdict update may repeat phase=completed for old
    # clients, but it is not a second authoritative v0.2 lifecycle revision.
    auxiliary = [
        frame for frame in all_delegations
        if frame["phase"] == "completed" and "state" not in frame
    ]
    assert all("task_revision" not in frame for frame in auxiliary)

    starts = client.of("tool_call.started")
    completions = client.of("tool_call.completed")
    assert len(starts) == len(completions) == 1
    started, completed = starts[0], completions[0]
    assert started["task_id"] == completed["task_id"] == "d1"
    assert started["parent_user_turn_id"] == completed["parent_user_turn_id"] == parent_turn
    assert started["parent_call_id"] == completed["parent_call_id"] == "live-delegation:d1"
    assert started["call_id"] == completed["call_id"]
    assert started["clock"] == completed["clock"] == "provider"
    assert isinstance(started["started_ms"], int)
    assert isinstance(completed["completed_ms"], int)
    assert completed["completed_ms"] >= started["started_ms"]


@pytest.mark.asyncio
async def test_interrupt_and_targeted_cancel_have_distinct_task_scope(monkeypatch):
    H.fast_clocks(
        monkeypatch,
        voice_live_delegation_ttl_s=2.0,
        voice_live_output_epoch_gap_ms=5000,
    )
    started = {"d1": asyncio.Event(), "d2": asyncio.Event()}
    release = {"d1": asyncio.Event(), "d2": asyncio.Event()}
    cancelled = {"d1": asyncio.Event(), "d2": asyncio.Event()}
    provider_box = {}
    observed = {}

    async def think(_user_id, _task, _session_id, **kwargs):
        delegation_id = kwargs["delegation_id"]
        started[delegation_id].set()
        try:
            await release[delegation_id].wait()
        except asyncio.CancelledError:
            cancelled[delegation_id].set()
            raise
        return f"Result for {delegation_id}.", "test-model"

    H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        provider_box["provider"] = provider
        if event["type"] == "session.start":
            provider.push(H.user_delta("Find a professor in Toronto.", 0, 400))
            provider.push(H.delegation("d1", 450))
            provider.push(H.user_delta("What is the weather in Paris?", 1000, 1400))
            provider.push(H.delegation("d2", 1450))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def prepare_interrupt():
        await asyncio.wait_for(
            asyncio.gather(started["d1"].wait(), started["d2"].wait()), timeout=3,
        )
        provider = provider_box["provider"]
        provider.push(H.out_text("An unrelated spoken update.", 2000, 2300))
        provider.push(H.out_audio("SPEAKING"))
        await _wait_until(lambda: bool(client.of("response_text")), label="open output epoch")

    async def observe_interrupt():
        await _wait_until(
            lambda: bool(client.of("playback_interrupted")),
            label="playback_interrupted frame",
        )
        observed["interrupt"] = dict(client.of("playback_interrupted")[-1])
        observed["cancelled_after_interrupt"] = {
            key: event.is_set() for key, event in cancelled.items()
        }

    async def observe_targeted_cancel():
        await asyncio.wait_for(cancelled["d1"].wait(), timeout=3)
        await _wait_until(
            lambda: _terminal(client, "d1", "cancelled"),
            label="targeted d1 cancellation",
        )
        observed["d2_cancelled_after_target"] = cancelled["d2"].is_set()

    async def observe_missing_id():
        await _wait_until(
            lambda: any(
                frame.get("code") == "cancel_task_id_required"
                for frame in client.of("error")
            ),
            label="missing cancel id error",
        )
        await asyncio.sleep(0.05)
        observed["d2_cancelled_after_missing"] = cancelled["d2"].is_set()
        release["d2"].set()
        await _wait_until(
            lambda: _terminal(client, "d2"), label="uncancelled d2 completion",
        )

    output_id = f"live:{H.PSID}:1"
    client = H.FakeClient([
        H.config(),
        prepare_interrupt,
        {
            "type": "interrupt",
            "response_id": output_id,
            "item_id": output_id,
            "played_ms": 120,
            "reason": "invented-client-reason",
        },
        observe_interrupt,
        {"type": "cancel_task", "task_id": "d1"},
        observe_targeted_cancel,
        {"type": "cancel_task"},
        observe_missing_id,
        {"type": "stop"},
    ])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(
        client, provider, timeout=9, db_session_id="db-repair-interrupt-scope",
    )

    assert observed["interrupt"]["assistant_turn_id"] == output_id
    assert observed["interrupt"]["played_ms"] == 120
    assert observed["interrupt"]["reason"] == "user_barge_in"
    assert observed["interrupt"]["tasks_still_running"] == ["d1", "d2"]
    assert observed["cancelled_after_interrupt"] == {"d1": False, "d2": False}
    assert observed["d2_cancelled_after_target"] is False
    assert observed["d2_cancelled_after_missing"] is False
    assert cancelled["d1"].is_set()
    assert not cancelled["d2"].is_set()
    assert _terminal(client, "d2")


@pytest.mark.asyncio
async def test_natural_cancel_can_target_a_queued_task_without_canceling_running_work(
    monkeypatch,
):
    H.fast_clocks(
        monkeypatch,
        voice_live_max_concurrent_delegations=1,
        voice_live_delegation_ttl_s=2.0,
    )
    running_started = asyncio.Event()
    release_running = asyncio.Event()
    running_cancelled = asyncio.Event()
    calls = []
    provider_box = {}

    async def think(_user_id, _task, _session_id, **kwargs):
        delegation_id = kwargs["delegation_id"]
        calls.append(delegation_id)
        if delegation_id != "d-running":
            return "Unexpected queued execution.", "test-model"
        running_started.set()
        try:
            await release_running.wait()
        except asyncio.CancelledError:
            running_cancelled.set()
            raise
        return "Professor result.", "test-model"

    H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        provider_box["provider"] = provider
        if event["type"] == "session.start":
            # A one-word running task used to score 1.0 under shorter-set
            # overlap and steal the more specifically named queued cancel.
            provider.push(H.user_delta("budget", 0, 300))
            provider.push(H.delegation("d-running", 350))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def queue_then_cancel():
        await asyncio.wait_for(running_started.wait(), timeout=3)
        provider = provider_box["provider"]
        provider.push(H.user_delta(
            "search Toronto finance report details", 1000, 1300,
        ))
        provider.push(H.delegation("d-queued", 1350))
        await _wait_until(
            lambda: any(
                frame.get("phase") == "pending"
                for frame in _delegations(client, "d-queued")
            ),
            label="queued task",
        )
        provider.push(H.user_delta(
            "cancel search Toronto finance budget", 2000, 2300,
        ))
        provider.push(H.delegation("d-cancel", 2350))
        await _wait_until(
            lambda: _terminal(client, "d-queued", "cancelled"),
            label="queued task cancellation",
        )
        # The cancel COMMAND is not a job (R2 §C): it used to be awaited here
        # as its own `cancelled` card. Its acknowledgement is the spoken line.
        await _wait_until(
            lambda: any(
                event.get("delegation_id") == "d-cancel"
                for event in provider.of("session.commentary.append")
            ),
            label="cancel command acknowledgement",
        )
        release_running.set()
        await _wait_until(
            lambda: _terminal(client, "d-running"),
            label="running task completion",
        )

    client = H.FakeClient([H.config(), queue_then_cancel, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(
        client, provider, timeout=9, db_session_id="db-repair-queued-cancel",
    )

    assert calls == ["d-running"]
    assert not running_cancelled.is_set()
    assert _delegations(client, "d-running")[-1]["phase"] == "completed"
    assert _delegations(client, "d-queued")[-1]["phase"] == "cancelled"
    assert _delegations(client, "d-cancel") == [], "the command itself is no card"


@pytest.mark.asyncio
async def test_orphan_pending_entry_does_not_block_a_later_bindable_delegation(
    monkeypatch,
):
    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=0.35)
    calls = []

    async def think(_user_id, task, _session_id, **kwargs):
        calls.append((task, kwargs.get("display_request")))
        return "The valid result.", "test-model"

    H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("Find the valid report.", 1000, 1100))
            provider.push(H.delegation("d-orphan", 100))
            provider.push(H.delegation("d-valid", 1200))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def wait_for_both_outcomes():
        await _wait_until(
            lambda: _terminal(client, "d-valid"),
            label="later bindable delegation",
        )
        await _wait_until(
            lambda: _terminal(client, "d-orphan", "expired"),
            label="orphan expiry",
        )

    client = H.FakeClient([
        H.config(), wait_for_both_outcomes, {"type": "stop"},
    ])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(
        client,
        provider,
        timeout=7,
        db_session_id="db-repair-pending-hol",
    )

    assert len(calls) == 1
    assert calls[0][0].startswith("Find the valid report.")
    assert calls[0][1] == "Find the valid report."


@pytest.mark.asyncio
async def test_idless_interrupt_before_output_stops_provider_without_fake_epoch_ack(monkeypatch):
    H.fast_clocks(monkeypatch)
    H.patch_relay(monkeypatch)

    def on_send(provider, event):
        if event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([
        H.config(),
        {"type": "interrupt", "reason": "anything", "played_ms": 0},
        0.05,
        {"type": "stop"},
    ])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(
        client, provider, timeout=5, db_session_id="db-repair-idless-interrupt",
    )

    assert client.of("playback_interrupted") == []
    stops = [
        event for event in provider.of("session.instructions.append")
        if "Stop speaking now" in str(event.get("content") or "")
    ]
    assert len(stops) == 1


@pytest.mark.asyncio
async def test_continuation_retries_repeated_professor_and_preserves_constraints(monkeypatch):
    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    calls = []
    provider_box = {}
    first_request = "Find a machine learning professor at the University of Toronto."
    professor_a = (
        "Professor Alice Smith is a machine learning professor at the "
        "University of Toronto."
    )
    professor_b = (
        "Consider Bob Chen instead. His lab studies robust neural systems, "
        "and he teaches at U of T."
    )

    async def think(_user_id, task, _session_id, **kwargs):
        calls.append({"task": task, "display": kwargs.get("display_request")})
        if len(calls) == 1:
            return professor_a, "test-model"
        if len(calls) == 2:
            return professor_a, "test-model"
        return professor_b, "test-model"

    recorded = H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        provider_box["provider"] = provider
        if event["type"] == "session.start":
            provider.push(H.user_delta(first_request, 0, 700))
            provider.push(H.delegation("d1", 750))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def ask_for_another():
        await _wait_until(
            lambda: _terminal(client, "d1"), label="first professor completion",
        )
        provider = provider_box["provider"]
        provider.push(H.user_delta("دیگه چه؟", 1800, 2050))
        provider.push(H.delegation("d2", 2100))
        await _wait_until(
            lambda: _terminal(client, "d2"), label="alternate professor completion",
        )

    client = H.FakeClient([H.config(), ask_for_another, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(
        client, provider, timeout=9, db_session_id="db-repair-continuation",
    )

    assert len(calls) == 3
    assert calls[1]["display"] == calls[2]["display"] == "دیگه چه؟"
    continuation_prompt = calls[1]["task"]
    assert "Continuation contract (backend-enforced)" in continuation_prompt
    assert f"Preserve every constraint from this prior request: {first_request}" in continuation_prompt
    assert professor_a in continuation_prompt
    assert "Return a different primary entity/result" in continuation_prompt
    assert "Your attempted result repeated the prior entity" in calls[2]["task"]

    d2_rows = [
        row for row in recorded["saves"]
        if (row.get("assistant_voice") or {}).get("delegation_id") == "d2"
    ]
    assert d2_rows and d2_rows[-1]["assistant_text"] == professor_b
    assert _delegations(client, "d2")[-1]["title"] == "دیگه چه؟"


@pytest.mark.asyncio
async def test_continuation_selects_relevant_result_across_unrelated_completion(monkeypatch):
    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    provider_box = {}
    prompts = []
    professor_request = "Find a machine learning professor at U of T."
    professor_result = "Professor Geoffrey Hinton is professor emeritus at U of T."
    weather_request = "What is the weather in Paris?"
    weather_result = "Paris is sunny and 22 degrees."

    async def think(_user_id, task, _session_id, **_kwargs):
        prompts.append(task)
        if len(prompts) == 1:
            return professor_result, "test-model"
        if len(prompts) == 2:
            return weather_result, "test-model"
        return "Professor Jimmy Ba is another U of T option.", "test-model"

    H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        provider_box["provider"] = provider
        if event["type"] == "session.start":
            provider.push(H.user_delta(professor_request, 0, 500))
            provider.push(H.delegation("d-professor", 550))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def finish_sequence():
        provider = provider_box["provider"]
        await _wait_until(
            lambda: _terminal(client, "d-professor"), label="professor result",
        )
        provider.push(H.user_delta(weather_request, 1000, 1350))
        provider.push(H.delegation("d-weather", 1400))
        await _wait_until(
            lambda: _terminal(client, "d-weather"), label="weather result",
        )
        provider.push(H.user_delta("another professor", 1900, 2150))
        provider.push(H.delegation("d-another", 2200))
        await _wait_until(
            lambda: _terminal(client, "d-another"), label="alternate professor",
        )

    client = H.FakeClient([H.config(), finish_sequence, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(
        client, provider, timeout=10, db_session_id="db-repair-relevant-continuation",
    )

    assert len(prompts) == 3
    continuation = prompts[2]
    assert professor_request in continuation
    assert professor_result in continuation
    assert weather_request not in continuation
    assert weather_result not in continuation


@pytest.mark.asyncio
async def test_partial_scaffold_answer_never_reaches_wire_or_persistence(monkeypatch):
    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)

    async def think(*_args, **_kwargs):
        return "Return a different primary entity/result.", "test-model"

    recorded = H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("Find another professor.", 0, 400))
            provider.push(H.delegation("d-scaffold", 450))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def await_failure():
        await _wait_until(
            lambda: _terminal(client, "d-scaffold", "failed"),
            label="scaffold answer rejected",
        )

    client = H.FakeClient([H.config(), await_failure, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(
        client, provider, timeout=7, db_session_id="db-repair-scaffold-filter",
    )

    leaked = "Return a different primary entity/result."
    assert all(frame.get("text") != leaked for frame in client.frames)
    assert all(row.get("assistant_text") != leaked for row in recorded["saves"])


@pytest.mark.parametrize(
    ("case_id", "phrase"),
    [
        ("compact", "آهنگو عوض کن"),
        ("separated", "آهنگ رو عوض کن"),
        ("next", "بعدی"),
        ("next_imperative", "بعدی رو بزن"),
        ("another", "یکی دیگه پخش کن"),
    ],
)
@pytest.mark.asyncio
async def test_every_required_persian_skip_phrase_uses_correlated_media_control(
    monkeypatch, case_id, phrase,
):
    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    thinks = []
    controls = []

    async def think(*args, **kwargs):
        thinks.append((args, kwargs))
        return "must not run", "test-model"

    async def control(_user_id, action):
        controls.append(action)
        return {"ok": True, "action": action, "title": "آهنگ بعدی"}

    H.patch_relay(monkeypatch, think=think, control=control)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta(phrase, 0, 350))
            provider.push(H.delegation("d1", 400))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def wait_for_control():
        await _wait_until(
            lambda: _terminal(client, "d1"), label=f"media control for {case_id}",
        )

    client = H.FakeClient([H.config(), wait_for_control, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(
        client, provider, timeout=7, db_session_id=f"db-repair-media-{case_id}",
    )

    assert thinks == []
    assert controls == ["next"]
    frames = client.of("media_control")
    assert [frame["status"] for frame in frames] == ["requested", "executed"]
    assert [frame["revision"] for frame in frames] == [1, 2]
    assert frames[0]["control_id"] == frames[1]["control_id"]
    assert frames[0]["task_id"] == frames[1]["task_id"] == "d1"
    assert frames[0]["parent_user_turn_id"] == frames[1]["parent_user_turn_id"]
    assert frames[0]["requested_ms"] == frames[1]["requested_ms"]
    assert frames[1]["ok"] is True and frames[1]["outcome"] == "ok"
    assert frames[1]["tool_call_id"] == client.of("tool_call.started")[0]["call_id"]


@pytest.mark.asyncio
async def test_persian_skip_failure_is_correlated_and_never_calls_agent(monkeypatch):
    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    thinks = []

    async def think(*args, **kwargs):
        thinks.append((args, kwargs))
        return "must not run", "test-model"

    async def control(_user_id, action):
        return {"ok": False, "action": action, "reason": "no_active_session"}

    H.patch_relay(monkeypatch, think=think, control=control)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("بعدی", 0, 250))
            provider.push(H.delegation("d1", 300))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def wait_for_failure():
        await _wait_until(
            lambda: _terminal(client, "d1", "failed"), label="media control failure",
        )

    client = H.FakeClient([H.config(), wait_for_failure, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(
        client, provider, timeout=7, db_session_id="db-repair-media-error",
    )

    assert thinks == []
    frames = client.of("media_control")
    assert [frame["status"] for frame in frames] == ["requested", "error"]
    assert frames[0]["control_id"] == frames[1]["control_id"]
    assert frames[0]["task_id"] == frames[1]["task_id"] == "d1"
    assert frames[0]["parent_user_turn_id"] == frames[1]["parent_user_turn_id"]
    assert frames[1]["ok"] is False
    assert frames[1]["outcome"] == "no_session"
    assert frames[1]["reason"] == "no_active_session"


@pytest.mark.asyncio
async def test_echoed_internal_scaffold_never_reaches_user_visible_surfaces(monkeypatch):
    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    calls = []
    provider_box = {}

    async def think(_user_id, task, _session_id, **kwargs):
        calls.append({"task": task, "display": kwargs.get("display_request")})
        if len(calls) == 1:
            return "Professor Alice Smith is the first result.", "test-model"
        # Simulate the agent echoing the entire private wrapper verbatim.
        return task, "test-model"

    recorded = H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        provider_box["provider"] = provider
        if event["type"] == "session.start":
            provider.push(H.user_delta("Find one professor.", 0, 400))
            provider.push(H.delegation("d1", 450))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def echo_private_prompt():
        await _wait_until(
            lambda: _terminal(client, "d1"), label="seed delegation completion",
        )
        provider = provider_box["provider"]
        provider.push(H.user_delta("Summarize that.", 1200, 1500))
        provider.push(H.delegation("d2", 1550))
        await _wait_until(
            lambda: _terminal(client, "d2", "failed"), label="blocked scaffold echo",
        )

    client = H.FakeClient([H.config(), echo_private_prompt, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(
        client, provider, timeout=9, db_session_id="db-repair-scaffold",
    )

    assert len(calls) == 2
    assert "Live-session context from earlier accepted delegations follows" in calls[1]["task"]
    assert calls[1]["display"] == "Summarize that."

    forbidden = (
        "Live-session context from earlier accepted delegations follows",
        "Prior caller request:",
        "Accepted backend result:",
        "Current caller request:",
    )
    persisted = json.dumps(recorded["saves"], ensure_ascii=False, default=str)
    spoken = "\n".join(
        str(event.get("content") or "")
        for event in provider.of("session.commentary.append")
    )
    titles = "\n".join(
        str(frame.get("title") or "") for frame in client.of("delegation")
    )
    for marker in forbidden:
        assert marker not in persisted
        assert marker not in spoken
        assert marker not in titles
    assert _delegations(client, "d2")[-1]["title"] == "Summarize that."


async def _gap_rotated_acknowledgement(monkeypatch, *, request, ack, lang, tag):
    """The relay's OWN idle timer retired the acknowledgement's epoch.

    This is the production-shaped version of the acknowledgement bug and the
    one the still-open-epoch rule does not reach: the provider pauses longer
    than ``voice_live_output_epoch_gap_ms`` between saying "I'll look" and
    emitting its delegation, ``rotate_on_gap`` retires the epoch, and the
    request used to be lost with ``no_causal_turn``.  A relay timeout is
    evidence about the relay, never about whether the model was finished.
    """

    H.fast_clocks(
        monkeypatch,
        voice_live_delegation_ttl_s=0.6,
        voice_live_output_epoch_gap_ms=80,
    )
    calls = []
    provider_box = {}

    async def think(*args, **kwargs):
        calls.append(kwargs)
        return "Found a professor.", "test-model"

    H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        provider_box["provider"] = provider
        if event["type"] == "session.start":
            provider.push(H.user_delta(request, 0, 200))
            provider.push(H.out_text(ack, 250, 350))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def delegate_after_gap_rotation():
        # Key on the retirement frame itself, never a sleep: these suites run
        # under heavy parallel load and a timing assumption would flake.
        await _wait_until(
            lambda: bool(client.of("speech_segment_complete")),
            label="acknowledgement epoch retired by the output-gap timer",
        )
        provider_box["provider"].push(H.delegation(f"search-after-gap-{tag}", 360))
        await _wait_until(
            lambda: _terminal(client, f"search-after-gap-{tag}"),
            timeout=4.0,
            label="post-gap delegation completion",
        )

    client = H.FakeClient([H.config(), delegate_after_gap_rotation, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(
        client, provider, timeout=9, db_session_id=f"db-repair-gap-ack-{tag}",
    )

    finals = [frame for frame in client.of("transcript") if frame["final"]]
    assert len(finals) == 1
    assert len(calls) == 1, "the gap-retired acknowledgement swallowed the request"
    assert calls[0]["display_request"] == request
    assert calls[0]["reply_language"] == lang
    lifecycle = [
        frame for frame in _delegations(client, f"search-after-gap-{tag}")
        if "state" in frame
    ]
    assert [frame["state"] for frame in lifecycle] == [
        "queued", "running", "completed",
    ]
    assert all(
        frame["parent_user_turn_id"] == finals[0]["turn_id"] for frame in lifecycle
    )
    assert not any(
        frame["phase"] == "expired"
        for frame in _delegations(client, f"search-after-gap-{tag}")
    )


@pytest.mark.asyncio
async def test_gap_rotated_english_acknowledgement_still_dispatches(monkeypatch):
    await _gap_rotated_acknowledgement(
        monkeypatch,
        request="Find an Iranian mechanical engineering professor at the University of Toronto.",
        ack="I will look for that professor.",
        lang="en",
        tag="en",
    )


@pytest.mark.asyncio
async def test_gap_rotated_persian_acknowledgement_still_dispatches(monkeypatch):
    await _gap_rotated_acknowledgement(
        monkeypatch,
        request="یک استاد مهندسی مکانیک ایرانی در دانشگاه تورنتو پیدا کن.",
        ack="دنبالش می‌گردم.",
        lang="fa",
        tag="fa",
    )


@pytest.mark.asyncio
async def test_gap_revocation_window_is_bounded_by_the_delegation_ttl(monkeypatch):
    """The gap grace is a WINDOW, not an open door.

    Past it the direct answer stands as given, which is what stops a genuinely
    late delegation from re-executing an answered turn on a client that never
    acknowledges playback.
    """

    H.fast_clocks(
        monkeypatch,
        voice_live_delegation_ttl_s=0.25,
        voice_live_output_epoch_gap_ms=80,
    )
    calls = []
    provider_box = {}

    async def think(*args, **kwargs):
        calls.append(kwargs)
        return "Duplicate work.", "test-model"

    H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        provider_box["provider"] = provider
        if event["type"] == "session.start":
            provider.push(H.user_delta("What is two plus two?", 0, 200))
            provider.push(H.out_text("Two plus two is four.", 250, 350))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def delegate_well_after_the_window():
        await _wait_until(
            lambda: bool(client.of("speech_segment_complete")),
            label="direct answer epoch retired by the output-gap timer",
        )
        # Comfortably past voice_live_delegation_ttl_s.
        await asyncio.sleep(0.5)
        provider_box["provider"].push(H.delegation("far-too-late", 360))
        await _wait_until(
            lambda: _terminal(client, "far-too-late", "expired"),
            timeout=4.0,
            label="late delegation expiry",
        )

    client = H.FakeClient([
        H.config(), delegate_well_after_the_window, {"type": "stop"},
    ])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(
        client, provider, timeout=9, db_session_id="db-repair-gap-window-bound",
    )

    assert calls == [], "a delegation past the grace re-executed an answered turn"
    terminal = [
        frame for frame in _delegations(client, "far-too-late")
        if frame.get("phase") == "expired" and "state" in frame
    ]
    assert len(terminal) == 1
    assert terminal[0]["state"] == "failed"
    assert terminal[0]["reason"] == "no_causal_turn"
    assert terminal[0]["parent_user_turn_id"] == ""


@pytest.mark.asyncio
async def test_bargein_over_an_acknowledgement_still_dispatches_its_delegation(monkeypatch):
    """Talking over "I'll look" must not lose the request.

    A barge-in stops audio only and never the job (contract §4.10).  Retiring
    the epoch on an interrupt therefore says nothing about whether the model
    had finished, so the causal turn stays revocable for the bounded grace —
    otherwise the caller who interrupts an acknowledgement gets silence instead
    of their search.
    """

    H.fast_clocks(
        monkeypatch,
        voice_live_delegation_ttl_s=0.6,
        voice_live_output_epoch_gap_ms=5000,
    )
    calls = []
    provider_box = {}
    request = "Find an Iranian mechanical engineering professor at the University of Toronto."

    async def think(*args, **kwargs):
        calls.append(kwargs)
        return "Found a professor.", "test-model"

    H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        provider_box["provider"] = provider
        if event["type"] == "session.start":
            provider.push(H.user_delta(request, 0, 200))
            provider.push(H.out_text("I will look for that professor.", 250, 350))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def arm_barge_in():
        await _wait_until(
            lambda: bool(client.of("response_text")),
            label="acknowledgement reaching the phone",
        )
        # Inserted frames are delivered only once this entry returns, so the
        # interrupt and the delegation must be separate script steps.
        client._script.insert(0, {
            "type": "interrupt",
            "response_id": f"live:{H.PSID}:1",
            "item_id": f"live:{H.PSID}:1",
            "played_ms": 400,
        })

    def delegate_after_barge_in():
        provider_box["provider"].push(H.delegation("search-after-bargein", 360))

    async def wait_for_completion():
        await _wait_until(
            lambda: _terminal(client, "search-after-bargein"),
            timeout=4.0,
            label="post-barge-in delegation completion",
        )

    client = H.FakeClient([
        H.config(), arm_barge_in, delegate_after_barge_in, wait_for_completion,
        {"type": "stop"},
    ])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(
        client, provider, timeout=9, db_session_id="db-repair-bargein-ack",
    )

    assert len(calls) == 1, "barge-in over an acknowledgement swallowed the request"
    assert calls[0]["display_request"] == request
    assert not any(
        frame["phase"] == "expired"
        for frame in _delegations(client, "search-after-bargein")
    )


@pytest.mark.asyncio
async def test_playback_acked_acknowledgement_still_dispatches_its_delegation(monkeypatch):
    """The independent supervisor's reproducer, pinned as a regression test.

    /private/tmp/toup-supervisor-playback-ack-repro.py: the phone acknowledges
    full playback of a spoken acknowledgement, and only then does the provider
    emit the delegation for the search the caller actually asked for.  This
    returned zero backend invocations and `no_causal_turn` until the fence
    stopped treating audio drain as proof of fulfilment.
    """

    H.fast_clocks(
        monkeypatch,
        voice_live_delegation_ttl_s=0.6,
        voice_live_output_epoch_gap_ms=5000,
    )
    calls = []
    provider_box = {}
    request = "Find an Iranian mechanical engineering professor at the University of Toronto."

    async def think(*args, **kwargs):
        calls.append(kwargs)
        return "Found a professor.", "test-model"

    H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        provider_box["provider"] = provider
        if event["type"] == "session.start":
            provider.push(H.user_delta(request, 0, 200))
            provider.push(H.out_text("I will look for that professor.", 250, 350))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def finish_ack_playback():
        await _wait_until(
            lambda: bool(client.of("response_text")),
            label="acknowledgement reaching the phone",
        )
        client._script.insert(0, {
            "type": "playback_idle",
            "response_id": f"live:{H.PSID}:1",
            "item_id": f"live:{H.PSID}:1",
            "played_ms": 600,
        })

    def delegate_after_playback():
        provider_box["provider"].push(H.delegation("search-after-playback-ack", 360))

    async def wait_for_completion():
        await _wait_until(
            lambda: _terminal(client, "search-after-playback-ack"),
            timeout=4.0,
            label="post-playback-ack delegation completion",
        )

    client = H.FakeClient([
        H.config(), finish_ack_playback, delegate_after_playback,
        wait_for_completion, {"type": "stop"},
    ])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(
        client, provider, timeout=9, db_session_id="db-repair-playback-ack",
    )

    assert len(calls) == 1, "playback completion swallowed an unfulfilled request"
    assert calls[0]["display_request"] == request
    assert not any(
        frame["phase"] == "expired"
        for frame in _delegations(client, "search-after-playback-ack")
    )


@pytest.mark.asyncio
async def test_persian_playback_acked_acknowledgement_still_dispatches(monkeypatch):
    """The same boundary in Persian — the fix must not be language-shaped."""

    H.fast_clocks(
        monkeypatch,
        voice_live_delegation_ttl_s=0.6,
        voice_live_output_epoch_gap_ms=5000,
    )
    calls = []
    provider_box = {}
    request = "یک استاد مهندسی مکانیک ایرانی در دانشگاه تورنتو پیدا کن."

    async def think(*args, **kwargs):
        calls.append(kwargs)
        return "استاد پیدا شد.", "test-model"

    H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        provider_box["provider"] = provider
        if event["type"] == "session.start":
            provider.push(H.user_delta(request, 0, 200))
            provider.push(H.out_text("دنبالش می‌گردم.", 250, 350))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def finish_ack_playback():
        await _wait_until(
            lambda: bool(client.of("response_text")),
            label="Persian acknowledgement reaching the phone",
        )
        client._script.insert(0, {
            "type": "playback_idle",
            "response_id": f"live:{H.PSID}:1",
            "item_id": f"live:{H.PSID}:1",
            "played_ms": 600,
        })

    def delegate_after_playback():
        provider_box["provider"].push(H.delegation("search-after-playback-fa", 360))

    async def wait_for_completion():
        await _wait_until(
            lambda: _terminal(client, "search-after-playback-fa"),
            timeout=4.0,
            label="Persian post-playback-ack delegation completion",
        )

    client = H.FakeClient([
        H.config(), finish_ack_playback, delegate_after_playback,
        wait_for_completion, {"type": "stop"},
    ])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(
        client, provider, timeout=9, db_session_id="db-repair-playback-ack-fa",
    )

    assert len(calls) == 1, "playback completion swallowed a Persian request"
    assert calls[0]["display_request"] == request
    assert calls[0]["reply_language"] == "fa"
