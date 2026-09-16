import asyncio
import copy
import json

import pytest

from app.api.voice_task_relay import VoiceTaskRelay
from app.api.realtime_lifecycle import ResponseCoordinator


class Coordinator:
    def __init__(self):
        self.events = []
        self.responses = 0
        self.response_payloads = []
        self.rejected = None

    def set_response_rejected_handler(self, handler):
        self.rejected = handler

    async def send_event(self, event):
        self.events.append(event)

    async def request_response(self, response=None):
        self.responses += 1
        self.response_payloads.append(response)
        return True


def task(status="running", revision=1, **kw):
    value = dict(task_id="work", job_id="work", session_id="conversation",
                 revision=revision, status=status, title="Research")
    value.update(kw)
    return value


@pytest.fixture
def harness():
    state = {"tasks": [], "calls": [], "events": []}

    async def request(method, path, **kwargs):
        state["calls"].append((method, path, kwargs))
        kind = (kwargs.get("body") or {}).get("kind")
        failures = state.get("ack_failures", {})
        if path.endswith("/ack") and failures.get(kind, 0):
            failures[kind] -= 1
            raise RuntimeError("temporary acknowledgement outage")
        if path.endswith("voice-tasks") and method == "GET":
            return copy.deepcopy({"tasks": state["tasks"]})
        return copy.deepcopy(state.get("result"))

    async def emit(event):
        state["events"].append(event)

    async def session():
        return "conversation"

    coordinator = Coordinator()
    relay = VoiceTaskRelay(request, emit, coordinator, session, interval=100)
    return relay, coordinator, state


async def test_uncertain_accept_reuses_exact_id_never_falls_back(harness):
    relay, _, state = harness
    result = await relay.submit(request_id="call1", message="Send approved report")
    assert result.startswith("ERROR") and "may already" in result
    assert len(state["calls"]) == 2
    assert state["calls"][0] == state["calls"][1]


async def test_accept_returns_while_work_runs_and_close_never_cancels(harness):
    relay, _, state = harness
    state["result"] = task()
    result = await relay.submit(request_id="call1", message="Research", user_message="پژوهش کن")
    assert json.loads(result)["status"] == "running"
    await relay.close()
    assert not any(path.endswith("/cancel") for _, path, _ in state["calls"])
    assert state["events"][0]["task"]["job_id"] == "work"


async def test_completion_once_with_full_artifact_after_long_work(harness):
    relay, coord, state = harness
    state["tasks"] = [task()]
    await relay.sync()
    # Many unchanged polls do not announce or duplicate the result.
    for _ in range(65):
        await relay.sync(announce=True)
    assert coord.responses == 0
    state["tasks"] = [task("completed", 3, text="Full report " * 1000,
                           artifacts=[{"url": "/api/files/report.pdf", "filename": "report.pdf"}],
                           sources=[{"url": "https://example.com", "title": "Source"}])]
    await relay.sync(announce=True)
    await relay.sync(announce=True)
    assert coord.responses == 1
    metadata = coord.response_payloads[0]["metadata"]
    assert set(metadata) == {"voice_task_speech_token"}
    assert isinstance(metadata["voice_task_speech_token"], str)
    assert len(metadata["voice_task_speech_token"]) <= 512
    assert await relay.speech_started_frame(
        response_id="response-1", metadata=metadata,
    ) == {
        "type": "voice_task.speech_started",
        "response_id": "response-1",
        "tasks": [{"task_id": "work", "revision": 3}],
    }
    delivered = [
        event["task"] for event in state["events"]
        if event.get("type") == "voice_task.updated"
    ][-1]
    assert delivered["artifacts"][0]["filename"] == "report.pdf"
    assert len(delivered["text"]) > 10000


async def test_reconnect_rehydrates_results_without_reexecution_or_reannouncing(harness):
    relay, coord, state = harness
    state["tasks"] = [task("completed", 8, text="Saved answer")]
    await relay.sync()
    await relay.sync(announce=True)
    assert coord.responses == 0
    # Reconnect may advance the delivery cursor, but it never submits work or
    # repeats a control action.
    assert not any(
        method == "POST" and path.rstrip("/").endswith("voice-tasks")
        for method, path, _ in state["calls"]
    )
    assert any(e["type"] == "voice_task.snapshot" and e["tasks"][0]["text"] == "Saved answer"
               for e in state["events"])


async def test_stale_revision_and_late_running_cannot_regress_terminal(harness):
    relay, _, state = harness
    assert await relay._merge(task("completed", 10))
    assert not await relay._merge(task("running", 9))
    assert not await relay._merge(task("running", 11))
    assert len(state["events"]) == 1


async def test_continuation_generation_reopens_terminal_and_rejects_late_frames(harness):
    relay, _, _ = harness
    assert await relay._merge(task("completed", 3, generation=0, text="Old result"))
    assert await relay._merge(task("queued", 4, generation=1, text="Old result"))
    assert await relay._merge(task("running", 5, generation=1))
    assert await relay._merge(task("completed", 6, generation=1, text="Corrected"))
    assert relay.tasks["work"]["text"] == "Corrected"
    assert not await relay._merge(task("running", 99, generation=0))
    assert not await relay._merge(task("queued", 5, generation=1))


async def test_malformed_wire_revisions_are_rejected_without_mutating_state(harness):
    relay, _, state = harness
    assert not await relay._merge(task(revision=True))
    assert not await relay._merge(task(generation=True))
    assert not await relay._merge(task(generation=-1))
    assert not await relay._merge(task(revision="1"))
    assert not await relay._merge(task(receipt_revision="1"))
    assert not await relay._merge(task(spoken_revision=-1))
    assert not await relay._merge(task(delivery_revision=2))
    assert relay.tasks == {}
    assert state["events"] == []

    assert await relay._merge(task(delivery_revision=None))
    assert relay.tasks["work"]["delivery_revision"] == 0
    assert not await relay._merge(task())
    assert not await relay.acknowledge(
        task_id="work", revision=True, kind="received",
    )
    assert not await relay.acknowledge(
        task_id="work", revision="1", kind="spoken",
    )
    assert relay._client_acks == {}


async def test_cross_session_control_cannot_mutate(harness):
    relay, _, state = harness
    state["result"] = {**task(), "session_id": "somebody-elses-conversation"}
    result = await relay.control(action="cancel", task_id="work", request_id="control1")
    assert result.startswith("ERROR")
    assert all(method == "GET" for method, _, _ in state["calls"])


async def test_explicit_steer_has_its_own_id_and_does_not_submit_new_work(harness):
    relay, _, state = harness
    state["result"] = task()
    await relay.control(action="steer", task_id="work", request_id="correction1", message="Only Toronto")
    method, path, data = next(
        call for call in state["calls"] if call[1].endswith("/work/steer")
    )
    assert method == "POST" and path.endswith("/work/steer")
    assert data["body"] == {"request_id": "correction1", "message": "Only Toronto"}


async def test_failed_poll_does_not_erase_known_work(harness):
    relay, coord, state = harness
    await relay._merge(task())

    async def outage(*a, **kw):
        return None

    relay.request = outage
    await relay.sync(announce=True)
    assert relay.tasks["work"]["status"] == "running"
    assert coord.responses == 0


async def test_unknown_outcome_is_not_reported_as_success(harness):
    relay, coord, state = harness
    await relay._merge(task())
    state["tasks"] = [task("unknown", 2, error="Worker interrupted; actions may have run")]
    await relay.sync(announce=True)
    assert coord.responses == 1
    assert '"status": "unknown"' in coord.events[-1]["item"]["content"][0]["text"]


async def test_spoken_ack_is_bound_to_exact_provider_response_and_known_revision(harness):
    relay, coord, _ = harness
    current = task("completed", 5)
    await relay._merge(current)
    await relay._remember_announcement(relay.tasks["work"])
    await relay._request_pending_announcement()
    metadata = coord.response_payloads[-1]["metadata"]
    assert await relay.speech_started_frame(response_id="response-1", metadata=metadata) == {
        "type": "voice_task.speech_started",
        "response_id": "response-1",
        "tasks": [{"task_id": "work", "revision": 5}],
    }
    assert await relay.speech_started_frame(response_id="", metadata=metadata) is None
    assert await relay.speech_started_frame(response_id="response-2", metadata={}) is None
    assert await relay.speech_started_frame(
        response_id="response-3", metadata={"voice_task_speech_token": []},
    ) is None


async def test_late_submit_and_control_responses_return_canonical_newest_state(harness):
    relay, _, state = harness
    await relay._merge(task("completed", 3, text="Canonical result"))
    state["result"] = task("queued", 1)

    submitted = json.loads(await relay.submit(
        request_id="late-submit", message="Research",
    ))
    status = json.loads(await relay.control(
        action="status", task_id="work", request_id="status",
    ))

    assert submitted["status"] == status["status"] == "completed"
    assert submitted["revision"] == status["revision"] == 3
    assert submitted["text"] == status["text"] == "Canonical result"


async def test_failed_client_ack_is_retained_and_retried_without_task_error(harness):
    relay, _, state = harness
    current = task("completed", 5, delivery_revision=5, receipt_revision=0,
                   spoken_revision=0)
    state["result"] = current
    await relay._merge(current)
    state["ack_failures"] = {"received": 1}

    assert not await relay.acknowledge(task_id="work", revision=5, kind="received")
    assert relay._client_acks == {("work", "received"): 5}

    state["tasks"] = [{**current, "receipt_revision": 5}]
    state["result"] = state["tasks"][0]
    await relay.sync(announce=True)

    assert relay._client_acks == {}
    received_calls = [
        call for call in state["calls"]
        if (call[2].get("body") or {}).get("kind") == "received"
    ]
    assert len(received_calls) == 2
    assert relay.tasks["work"]["receipt_revision"] == 5


def real_harness():
    state = {"tasks": [], "events": [], "provider": []}

    async def request(method, path, **kwargs):
        if method == "GET" and path.endswith("voice-tasks"):
            return copy.deepcopy({"tasks": state["tasks"]})
        if path.endswith("/ack"):
            task_id = path.rsplit("/", 2)[-2]
            return copy.deepcopy(next(
                (item for item in state["tasks"] if item["task_id"] == task_id),
                None,
            ))
        return None

    async def emit(event):
        state["events"].append(event)

    async def send_provider(event):
        state["provider"].append(event)

    async def session():
        return "conversation"

    coordinator = ResponseCoordinator(send_provider)
    relay = VoiceTaskRelay(request, emit, coordinator, session, interval=100)
    return relay, coordinator, state


async def test_actual_provider_frame_uses_bounded_string_metadata_for_multiple_tasks():
    relay, coordinator, state = real_harness()
    try:
        state["tasks"] = [task(task_id="first"), task(task_id="second")]
        await relay.sync()
        state["tasks"] = [
            task("completed", 2, task_id="first", text="First"),
            task("completed", 2, task_id="second", text="Second"),
        ]
        await relay.sync(announce=True)

        creates = [event for event in state["provider"] if event["type"] == "response.create"]
        assert len(creates) == 1
        metadata = creates[0]["response"]["metadata"]
        assert metadata
        assert all(
            isinstance(key, str) and isinstance(value, str)
            and len(key) <= 64 and len(value) <= 512
            for key, value in metadata.items()
        )

        await coordinator.response_created("announcement", metadata=metadata)
        frame = await relay.speech_started_frame(
            response_id="announcement", metadata=metadata,
        )
        assert frame["tasks"] == [
            {"task_id": "first", "revision": 2},
            {"task_id": "second", "revision": 2},
        ]
    finally:
        await relay.close()
        await coordinator.close()


async def test_confirmed_announcement_rejection_restores_same_terminal_intent():
    relay, coordinator, state = real_harness()
    try:
        state["tasks"] = [task()]
        await relay.sync()
        state["tasks"] = [task("completed", 2, text="Finished")]
        await relay.sync(announce=True)
        creates = [event for event in state["provider"] if event["type"] == "response.create"]
        first = creates[0]

        assert await coordinator.response_error(first["event_id"])

        creates = [event for event in state["provider"] if event["type"] == "response.create"]
        assert len(creates) == 2
        await relay.sync(announce=True)  # unchanged revision cannot duplicate the retry
        creates = [event for event in state["provider"] if event["type"] == "response.create"]
        assert len(creates) == 2
        assert (
            creates[0]["response"]["metadata"]["voice_task_speech_token"]
            != creates[1]["response"]["metadata"]["voice_task_speech_token"]
        )
        metadata = creates[1]["response"]["metadata"]
        await coordinator.response_created("recovered", metadata=metadata)
        assert await relay.speech_started_frame(
            response_id="recovered", metadata=metadata,
        )
    finally:
        await relay.close()
        await coordinator.close()


async def test_two_completions_queue_separate_markers_behind_active_response():
    relay, coordinator, state = real_harness()
    try:
        state["tasks"] = [task(task_id="first"), task(task_id="second")]
        await relay.sync()
        await coordinator.response_created("user-response", metadata={})

        state["tasks"][0] = task("completed", 2, task_id="first", text="First")
        await relay.sync(announce=True)
        assert not [e for e in state["provider"] if e["type"] == "response.create"]

        state["tasks"][1] = task("completed", 2, task_id="second", text="Second")
        await relay.sync(announce=True)
        await coordinator.response_done("user-response")
        creates = [event for event in state["provider"] if event["type"] == "response.create"]
        assert len(creates) == 1

        first_metadata = creates[0]["response"]["metadata"]
        await coordinator.response_created("first-announcement", metadata=first_metadata)
        first_frame = await relay.speech_started_frame(
            response_id="first-announcement", metadata=first_metadata,
        )
        assert first_frame["tasks"] == [{"task_id": "first", "revision": 2}]

        await coordinator.response_done("first-announcement")
        creates = [event for event in state["provider"] if event["type"] == "response.create"]
        assert len(creates) == 2
        second_metadata = creates[1]["response"]["metadata"]
        await coordinator.response_created("second-announcement", metadata=second_metadata)
        second_frame = await relay.speech_started_frame(
            response_id="second-announcement", metadata=second_metadata,
        )
        assert second_frame["tasks"] == [{"task_id": "second", "revision": 2}]
    finally:
        await relay.close()
        await coordinator.close()


async def test_active_conflict_with_explicit_queue_preserves_token_and_later_completion():
    relay, coordinator, state = real_harness()
    try:
        state["tasks"] = [task(task_id="first"), task(task_id="second")]
        await relay.sync()
        state["tasks"][0] = task(
            "completed", 2, task_id="first", text="First complete",
        )
        await relay.sync(announce=True)
        creates = [e for e in state["provider"] if e["type"] == "response.create"]
        assert len(creates) == 1
        rejected = creates[0]
        rejected_token = rejected["response"]["metadata"]["voice_task_speech_token"]

        # A user reply takes the desired slot while the task announcement is
        # awaiting its provider ACK. The provider then proves that an unrelated
        # automatic response was already active.
        assert await coordinator.request_response(
            {"instructions": "Answer the user first"}, automatic=False,
        )
        assert await coordinator.response_error(
            rejected["event_id"], retry_on_active=True,
        )
        await coordinator.response_created("provider-automatic", metadata={})
        await coordinator.response_done("provider-automatic")

        creates = [e for e in state["provider"] if e["type"] == "response.create"]
        assert len(creates) == 2
        explicit = creates[1]
        assert explicit["response"]["instructions"] == "Answer the user first"
        await coordinator.response_created(
            "explicit", metadata=explicit["response"]["metadata"],
        )
        await coordinator.response_done("explicit")

        creates = [e for e in state["provider"] if e["type"] == "response.create"]
        assert len(creates) == 3
        retried = creates[2]
        assert (
            retried["response"]["metadata"]["voice_task_speech_token"]
            == rejected_token
        )
        await coordinator.response_created(
            "first-announcement", metadata=retried["response"]["metadata"],
        )
        first_frame = await relay.speech_started_frame(
            response_id="first-announcement",
            metadata=retried["response"]["metadata"],
        )
        assert first_frame["tasks"] == [{"task_id": "first", "revision": 2}]
        await coordinator.response_done("first-announcement")

        state["tasks"][1] = task(
            "completed", 2, task_id="second", text="Second complete",
        )
        await relay.sync(announce=True)
        creates = [e for e in state["provider"] if e["type"] == "response.create"]
        assert len(creates) == 4
        later = creates[3]
        await coordinator.response_created(
            "second-announcement", metadata=later["response"]["metadata"],
        )
        second_frame = await relay.speech_started_frame(
            response_id="second-announcement",
            metadata=later["response"]["metadata"],
        )
        assert second_frame["tasks"] == [{"task_id": "second", "revision": 2}]
        assert relay._announcement_tokens == {}
        assert relay._pending_announcements == {}
    finally:
        await relay.close()
        await coordinator.close()


async def test_late_accepted_interrupted_announcement_does_not_block_next_result():
    relay, coordinator, state = real_harness()
    try:
        state["tasks"] = [task(task_id="first"), task(task_id="second")]
        await relay.sync()
        state["tasks"][0] = task(
            "completed", 2, task_id="first", text="First",
        )
        await relay.sync(announce=True)
        creates = [event for event in state["provider"] if event["type"] == "response.create"]
        first_metadata = creates[0]["response"]["metadata"]

        await coordinator.speech_started()
        await coordinator.speech_stopped(expect_user_response=False)
        await coordinator.response_created("stale-announcement", metadata=first_metadata)
        assert not coordinator.accepts_response("stale-announcement")
        assert await relay.speech_started_frame(
            response_id="stale-announcement", metadata=first_metadata,
        )

        state["tasks"][1] = task(
            "completed", 2, task_id="second", text="Second",
        )
        await relay.sync(announce=True)
        await coordinator.response_done("stale-announcement")

        creates = [event for event in state["provider"] if event["type"] == "response.create"]
        assert len(creates) == 2
        assert (
            creates[1]["response"]["metadata"]["voice_task_speech_token"]
            != first_metadata["voice_task_speech_token"]
        )
    finally:
        await relay.close()
        await coordinator.close()


# ── The day scope has a PRODUCER, and it is sync() ─────────────────────────
# `_in_scope` was covered by hand-setting `relay._scope`, so the two lines in
# `sync()` that actually populate it were executed by nothing: deleting them
# left every test in this file and in test_voice_task_session_rebind.py green
# while `control()` went back to answering "This task is not available in this
# conversation" for work the user started two minutes ago (Incident 3).

async def test_sync_ingests_the_scope_key_and_widens_to_the_day(harness):
    relay, _, _state = harness
    foreign = task(session_id="a-previous-socket", scope_key="day-1")

    async def _request(method, path, **kwargs):
        if path.endswith("voice-tasks") and method == "GET":
            return {"tasks": [foreign], "scope_key": "day-1"}
        return None

    relay.request = _request
    assert relay._scope is None
    await relay.sync()

    assert relay._scope == "day-1", "nothing executed the ingestion"
    assert "work" in relay.tasks, (
        "a task started on an earlier socket of the same day was filtered out"
    )


async def test_without_a_scope_key_the_session_check_is_still_the_floor(harness):
    """An agent image that sends no `scope_key` must behave exactly as today —
    which is also what proves the widening above is doing the work."""
    relay, _, _state = harness
    foreign = task(session_id="a-previous-socket")

    async def _request(method, path, **kwargs):
        if path.endswith("voice-tasks") and method == "GET":
            return {"tasks": [foreign]}
        return None

    relay.request = _request
    await relay.sync()

    assert relay._scope is None
    assert relay.tasks == {}, "a foreign session's task must not leak in"


async def test_control_reaches_a_task_from_an_earlier_socket_of_the_same_day(harness):
    """The user-visible half of Incident 3."""
    relay, _, _state = harness
    foreign = task(session_id="a-previous-socket", scope_key="day-1")

    async def _request(method, path, **kwargs):
        if path.endswith("voice-tasks") and method == "GET":
            return {"tasks": [foreign], "scope_key": "day-1"}
        return copy.deepcopy(foreign)

    relay.request = _request
    assert "not available in this conversation" in await relay.control(
        action="status", task_id="work", request_id="c1",
    ), "before the first sync the relay knows no day — the floor holds"

    await relay.sync()
    out = await relay.control(action="status", task_id="work", request_id="c1")
    assert "not available" not in out
    assert json.loads(out)["task_id"] == "work"
