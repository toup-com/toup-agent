"""Deterministic socket lifecycle tests with real asyncio scheduling, no services."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
import pytest_asyncio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.api.realtime_lifecycle import (
    FunctionCallSupervisor,
    LifecycleCapacityError,
    ResponseCoordinator,
)


@pytest_asyncio.fixture(autouse=True)
async def _reset_database():
    # These tests do not import the database or execute any application services.
    yield


async def settle():
    # Exercise all scheduling boundaries without real network/clock delays.
    for _ in range(20):
        await asyncio.sleep(0)


class Socket:
    def __init__(self):
        self.events = []
        self.writers = 0
        self.max_writers = 0

    async def send(self, event):
        self.writers += 1
        self.max_writers = max(self.max_writers, self.writers)
        try:
            await asyncio.sleep(0)
            self.events.append(event)
        finally:
            self.writers -= 1

    @property
    def outputs(self):
        return [e["item"] for e in self.events if e["type"] == "conversation.item.create"]

    @property
    def creates(self):
        return [e for e in self.events if e["type"] == "response.create"]


@pytest_asyncio.fixture
async def rig():
    socket = Socket()
    responses = ResponseCoordinator(socket.send)
    functions = FunctionCallSupervisor(responses)
    yield socket, responses, functions
    await functions.close()
    await responses.close()


async def answer():
    return "Research complete, with sources."


async def test_execution_does_not_wait_on_or_block_provider_lifecycle(rig):
    socket, responses, functions = rig
    entered, release = asyncio.Event(), asyncio.Event()

    async def research():
        entered.set()
        await release.wait()
        return "Completed research"

    await responses.response_created("original")
    assert functions.submit(call_id="call", response_id="original", run=research)
    await entered.wait()
    # The provider can finish its response and receive a new utterance while
    # the main agent's multi-step research remains pending indefinitely.
    await responses.response_done("original", ["call"])
    await responses.speech_started()
    await responses.speech_stopped()
    await responses.response_created("correction", metadata={})
    assert responses.accepts_response("correction")
    assert functions.pending_count == 1
    assert not socket.outputs
    release.set()
    await settle()
    assert [o["output"] for o in socket.outputs] == ["Completed research"]
    assert not socket.creates  # old research may not speak over the correction


async def test_fast_tool_waits_for_originating_response_done(rig):
    socket, responses, functions = rig
    await responses.response_created("r")
    functions.submit(call_id="a", response_id="r", run=answer)
    await settle()
    assert not socket.events
    await responses.response_done("r", ["a"])
    assert [e["type"] for e in socket.events] == [
        "conversation.item.create", "response.create",
    ]


async def test_done_before_item_done_fences_the_entire_function_batch(rig):
    socket, responses, functions = rig
    await responses.response_created("r")
    await responses.response_done("r", ["a", "b"])
    functions.submit(call_id="a", response_id="r", run=answer)
    await settle()
    assert not socket.events
    functions.submit(call_id="b", response_id="r", run=answer)
    await settle()
    assert [o["call_id"] for o in socket.outputs] == ["a", "b"]
    assert len(socket.creates) == 1
    await responses.response_done("r", ["b", "a"])
    assert len(socket.creates) == 1


async def test_call_id_deduplicates_before_and_after_completion(rig):
    socket, responses, functions = rig
    effects = []

    async def mutate():
        effects.append("sent")
        return "Sent"

    assert functions.submit(call_id="same", response_id="r", run=mutate)
    assert not functions.submit(call_id="same", response_id="r", run=mutate)
    await responses.response_done("r", ["same"])
    await settle()
    assert not functions.submit(call_id="same", response_id="other", run=mutate)
    await settle()
    assert effects == ["sent"]
    assert len(socket.outputs) == len(socket.creates) == 1


async def test_native_mutations_execute_serially(rig):
    _, _, functions = rig
    entered, release = asyncio.Event(), asyncio.Event()
    effects = []

    async def first():
        effects.append("first started")
        entered.set()
        await release.wait()
        effects.append("first finished")
        return "one"

    async def second():
        effects.append("second")
        return "two"

    functions.submit(call_id="a", response_id="r", run=first)
    functions.submit(call_id="b", response_id="r", run=second)
    await entered.wait()
    assert functions.pending_count == 2
    assert effects == ["first started"]
    release.set()
    await settle()
    assert effects == ["first started", "first finished", "second"]
    assert functions.pending_count == 0


async def test_queue_overflow_never_runs_rejected_work(rig):
    socket, responses, _ = rig
    functions = FunctionCallSupervisor(responses, max_pending=1)
    release = asyncio.Event()
    effects = []

    async def first():
        await release.wait()
        return "one"

    async def rejected():
        effects.append("BAD")
        return "two"

    try:
        assert functions.submit(call_id="a", response_id="r", run=first)
        assert not functions.submit(call_id="b", response_id="r", run=rejected)
        assert not functions.submit(call_id="b", response_id="r", run=rejected)
        await responses.response_done("r", ["a", "b"])
        release.set()
        await settle()
        assert effects == []
        assert len(socket.outputs) == 2
        assert "not executed" in socket.outputs[1]["output"]
        assert len(socket.creates) == 1
    finally:
        await functions.close()


async def test_barge_in_preserves_results_but_suppresses_stale_audio(rig):
    socket, responses, functions = rig
    await responses.response_created("old")
    assert responses.accepts_response("old")
    functions.submit(call_id="a", response_id="old", run=answer)
    await responses.speech_started()
    first_epoch = responses.epoch
    await responses.speech_started()  # duplicate VAD start
    assert responses.epoch == first_epoch
    assert not responses.accepts_response("old")
    await responses.response_done("old", ["a"], status="cancelled")
    await responses.speech_stopped()
    await settle()
    assert len(socket.outputs) == 1
    assert not socket.creates
    await responses.response_created("new", metadata={})
    assert responses.accepts_response("new")
    assert not responses.accepts_response("old")


async def test_result_followup_waits_for_user_automatic_response(rig):
    socket, responses, _ = rig
    await responses.speech_started()
    await responses.request_response()  # a durable task has just completed
    await responses.speech_stopped()
    assert not socket.creates
    await responses.response_created("user-response", metadata={})
    assert not socket.creates
    await responses.response_done("user-response")
    assert len(socket.creates) == 1


async def test_manual_vad_can_release_automatic_response_wait(rig):
    socket, responses, _ = rig
    await responses.speech_started()
    await responses.request_response()
    await responses.speech_stopped(expect_user_response=False)
    assert len(socket.creates) == 1


async def test_stop_speaking_pauses_future_automatic_announcements(rig):
    socket, responses, functions = rig
    await responses.response_created("old")
    functions.submit(call_id="a", response_id="old", run=answer)
    await responses.interrupt_speech()
    await responses.response_done("old", ["a"], status="cancelled")
    await settle()
    assert len(socket.outputs) == 1
    await responses.request_response()
    assert not socket.creates
    assert not responses.accepts_response("old")
    await responses.request_response(automatic=False)  # typed user turn
    assert len(socket.creates) == 1


async def test_pending_create_is_reserved_until_matching_ack(rig):
    socket, responses, _ = rig
    await responses.request_response()
    create = socket.creates[0]
    await responses.request_response()
    assert len(socket.creates) == 1
    # An overlapping VAD response is not the ACK for our create request.
    await responses.response_created("automatic", metadata={})
    await responses.response_done("automatic")
    assert len(socket.creates) == 1
    await responses.response_created("ours", metadata=create["response"]["metadata"])
    assert len(socket.creates) == 1
    await responses.response_done("ours")
    assert len(socket.creates) == 1  # duplicate requests coalesced, no extra speech


async def test_response_metadata_requires_bounded_string_pairs(rig):
    socket, responses, _ = rig
    invalid = [
        {"metadata": {"bad": []}},
        {"metadata": {"x" * 65: "value"}},
        {"metadata": {"key": "x" * 513}},
        {"metadata": {f"key_{index}": "value" for index in range(16)}},
        {"metadata": []},
    ]
    for payload in invalid:
        with pytest.raises(ValueError):
            await responses.request_response(payload)
    assert socket.creates == []

    assert await responses.request_response({
        "metadata": {f"key_{index}": "value" for index in range(15)},
    })
    assert len(socket.creates[0]["response"]["metadata"]) == 16
    assert all(
        isinstance(key, str) and isinstance(value, str)
        and len(key) <= 64 and len(value) <= 512
        for key, value in socket.creates[0]["response"]["metadata"].items()
    )


async def test_explicit_response_preserves_displaced_automatic_intent(rig):
    socket, responses, _ = rig
    await responses.response_created("active", metadata={})
    assert await responses.request_response({"instructions": "automatic result"})
    assert await responses.request_response(
        {"instructions": "explicit user reply"}, automatic=False,
    )
    await responses.response_done("active")
    assert socket.creates[-1]["response"]["instructions"] == "explicit user reply"

    explicit = socket.creates[-1]
    await responses.response_created(
        "explicit", metadata=explicit["response"]["metadata"],
    )
    await responses.response_done("explicit")
    assert socket.creates[-1]["response"]["instructions"] == "automatic result"


async def test_late_response_created_cannot_resurrect_done_response(rig):
    socket, responses, functions = rig
    functions.submit(call_id="a", response_id="old", run=answer)
    await responses.response_done("old", ["a"])
    await settle()
    assert len(socket.creates) == 1
    await responses.response_created("old")  # out-of-order duplicate
    await responses.response_done("old", ["a"])
    await responses.request_response()
    assert len(socket.creates) == 1  # original create is still awaiting its ACK


async def test_late_ack_retains_interrupted_epoch(rig):
    socket, responses, _ = rig
    await responses.request_response()
    create = socket.creates[0]
    await responses.speech_started()
    await responses.speech_stopped()
    await responses.response_created("late", metadata=create["response"]["metadata"])
    assert not responses.accepts_response("late")
    await responses.response_done("late")
    assert len(socket.creates) == 1


async def test_provider_writes_are_serialized_and_safe_during_new_events(rig):
    socket, responses, functions = rig
    await responses.response_created("old")
    functions.submit(call_id="a", response_id="old", run=answer)
    await settle()
    await asyncio.gather(
        responses.response_done("old", ["a"]),
        responses.response_created("new", metadata={}),
        responses.send_event({"type": "input_audio_buffer.append", "audio": "x"}),
    )
    assert socket.max_writers == 1
    assert len(socket.outputs) == 1
    assert not socket.creates  # new response is already active


async def test_function_failure_does_not_replay_or_stop_next_function(rig):
    socket, responses, functions = rig
    effects = []

    async def fails_after_mutation():
        effects.append("mutated once")
        raise RuntimeError("sensitive provider details")

    functions.submit(call_id="a", response_id="r", run=fails_after_mutation)
    functions.submit(call_id="b", response_id="r", run=answer)
    await responses.response_done("r", ["a", "b"])
    await settle()
    assert effects == ["mutated once"]
    assert "effects are unknown" in socket.outputs[0]["output"]
    assert "sensitive" not in socket.outputs[0]["output"]
    assert socket.outputs[1]["output"] == await answer()


async def test_cancel_is_explicit_and_distinguishes_queued_from_running(rig):
    socket, responses, functions = rig
    entered = asyncio.Event()

    async def running():
        entered.set()
        await asyncio.Event().wait()
        return "unreachable"

    functions.submit(call_id="running", response_id="r", run=running)
    functions.submit(call_id="queued", response_id="r", run=answer)
    await entered.wait()
    assert functions.cancel("queued")
    assert functions.cancel("running")
    await responses.response_done("r", ["running", "queued"])
    await settle()
    outputs = {o["call_id"]: o["output"] for o in socket.outputs}
    assert "before execution" in outputs["queued"]
    assert "may already have started" in outputs["running"]
    assert not functions.cancel("missing")


async def test_close_prevents_queued_side_effects_and_preserves_detached_job(rig):
    _, _, functions = rig
    entered, finish = asyncio.Event(), asyncio.Event()
    effects = []

    async def durable_job():
        entered.set()
        await finish.wait()
        effects.append("durable finished")
        return "done"

    job = asyncio.create_task(durable_job())

    async def wait_for_job():
        return await asyncio.shield(job)

    async def queued():
        effects.append("BAD queued mutation")
        return "queued"

    functions.submit(call_id="a", response_id="r", run=wait_for_job)
    functions.submit(call_id="b", response_id="r", run=queued)
    await entered.wait()
    await settle()
    await functions.close()
    assert not job.cancelled()
    assert not functions.submit(call_id="c", response_id="r", run=queued)
    finish.set()
    await job
    assert effects == ["durable finished"]


async def test_ambiguous_provider_write_failure_never_retries_effect_or_output():
    attempts, effects = [], []

    async def broken_socket(event):
        attempts.append(event)
        raise ConnectionError("closed after write")

    async def mutation():
        effects.append("once")
        return "done"

    responses = ResponseCoordinator(broken_socket)
    functions = FunctionCallSupervisor(responses)
    try:
        await responses.response_done("r", ["a"])
        functions.submit(call_id="a", response_id="r", run=mutation)
        await settle()
        assert responses.closed
        assert not functions.submit(call_id="a", response_id="r", run=mutation)
        await responses.request_response()
        assert len(attempts) == 1
        assert effects == ["once"]
    finally:
        await functions.close()
        await responses.close()


async def test_create_error_releases_only_matching_request_without_retry(rig):
    socket, responses, _ = rig
    await responses.request_response()
    assert not await responses.response_error("unrelated event")
    assert await responses.response_error(socket.creates[0]["event_id"])
    assert len(socket.creates) == 1
    await responses.request_response(automatic=False)
    assert len(socket.creates) == 2


async def test_active_response_conflict_replays_exact_intent_after_active_done(rig):
    socket, responses, _ = rig
    await responses.request_response({"instructions": "announce result"})
    rejected = socket.creates[0]
    assert await responses.response_error(
        rejected["event_id"], retry_on_active=True,
    )
    assert len(socket.creates) == 1

    # The provider-created response can be observed after its conflict error.
    await responses.response_created("automatic", metadata={})
    assert len(socket.creates) == 1
    await responses.response_done("automatic")

    assert len(socket.creates) == 2
    assert socket.creates[1]["response"]["instructions"] == "announce result"
    assert socket.creates[1]["event_id"] != rejected["event_id"]


async def test_active_response_conflict_retries_are_bounded(rig):
    socket, responses, _ = rig
    rejected_payloads = []

    async def rejected(response):
        rejected_payloads.append(response)

    responses.set_response_rejected_handler(rejected)
    await responses.request_response({"instructions": "bounded announcement"})

    # Initial create plus three proven-active retries. The fourth conflict is
    # returned to the owner rather than cycling forever.
    for index in range(4):
        create = socket.creates[-1]
        assert await responses.response_error(
            create["event_id"], retry_on_active=True,
        )
        await responses.response_created(f"automatic-{index}", metadata={})
        await responses.response_done(f"automatic-{index}")

    assert len(socket.creates) == 4
    assert rejected_payloads == [{"instructions": "bounded announcement"}]


async def test_result_arriving_during_active_response_requests_fresh_generation(rig):
    socket, responses, _ = rig
    await responses.response_created("speaking", metadata={})
    await responses.request_response()
    await responses.send_event({
        "type": "conversation.item.create",
        "item": {"type": "message", "role": "user", "content": []},
    })
    await responses.request_response()
    assert not socket.creates
    await responses.response_done("speaking")
    assert len(socket.creates) == 1


async def test_late_delta_after_done_is_not_audible(rig):
    _, responses, _ = rig
    await responses.response_created("r")
    assert responses.accepts_response("r")
    await responses.response_done("r")
    assert not responses.accepts_response("r")


async def test_history_limit_keeps_deduplication_instead_of_evicting():
    socket = Socket()
    responses = ResponseCoordinator(socket.send, max_calls=1, max_responses=2)
    try:
        assert responses.register_call("original", "r")
        assert not responses.register_call("original", "different")
        with pytest.raises(LifecycleCapacityError):
            responses.register_call("new", "r")
        await responses.response_created("r2")
        with pytest.raises(LifecycleCapacityError):
            await responses.response_created("r3")
    finally:
        await responses.close()
