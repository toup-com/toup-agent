"""Durability, fencing, correction, and approval tests for managed voice work."""
from __future__ import annotations

import asyncio
import os
import time
import uuid
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import select, update

from app.agent.operation_identity import (
    tool_operation_checkpoint,
    tool_operation_kind,
    tool_operation_key,
)
from app.agent.voice_tasks import (
    SOURCE_KIND,
    VoiceTaskError,
    VoiceTaskService,
    task_id_for,
)
from app.db import async_session_maker
from app.db.models import BuildJob, Conversation, Message, User


# These tests intentionally exercise competing transactions, row locks, and
# lease fencing. The default in-memory SQLite fixture has one StaticPool
# connection and cannot represent those semantics.
pytestmark = pytest.mark.skipif(
    "postgres" not in os.environ.get("DATABASE_URL", ""),
    reason="durable voice ledger concurrency requires PostgreSQL",
)


def result(text="Finished", *, metadata=None, model="test-model"):
    return SimpleNamespace(
        text=text,
        model=model,
        tokens_input=11,
        tokens_output=7,
        tokens_total=18,
        stopped_reason="",
        metadata=metadata or {},
    )


class GateRunner:
    def __init__(self, answer=None):
        self.answer = answer or result()
        self.calls = []
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def run(self, **kwargs):
        self.calls.append(kwargs)
        self.entered.set()
        await self.release.wait()
        return self.answer


class SequenceRunner:
    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    async def run(self, **kwargs):
        self.calls.append(kwargs)
        return self.answers.pop(0)


class CorrectionRunner:
    def __init__(self):
        self.calls = []
        self.entered = asyncio.Event()
        self.check = asyncio.Event()
        self.corrections = []

    async def run(self, **kwargs):
        self.calls.append(kwargs)
        self.entered.set()
        await self.check.wait()
        self.corrections = await kwargs["steering_check"]()
        return result("Used the corrected scope")


class RequeueRunner:
    def __init__(self):
        self.calls = []
        self.current = 0
        self.max_current = 0
        self.corrections = []

    async def run(self, **kwargs):
        self.calls.append(kwargs)
        self.current += 1
        self.max_current = max(self.max_current, self.current)
        try:
            if len(self.calls) == 1:
                return result("Result from the old scope")
            self.corrections = await kwargs["steering_check"]()
            return result("Result from the corrected scope")
        finally:
            self.current -= 1


class TerminalCorrectionRunner:
    def __init__(self):
        self.calls = []
        self.second_entered = asyncio.Event()
        self.release_second = asyncio.Event()
        self.corrections = []

    async def run(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            return result("Original completed result")
        self.second_entered.set()
        await self.release_second.wait()
        self.corrections = await kwargs["steering_check"]()
        return result("Corrected completed result")


async def seed_conversation():
    user_id, conversation_id = str(uuid.uuid4()), str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(
            id=user_id,
            email=f"voice-{user_id[:8]}@example.test",
            hashed_password="x",
            name="Voice Test",
        ))
        db.add(Conversation(
            id=conversation_id,
            user_id=user_id,
            title="Voice",
            channel="voice",
            message_count=0,
            total_tokens=0,
        ))
        await db.commit()
    return user_id, conversation_id


async def seed_job(user_id, conversation_id, *, status="queued", **values):
    task_id = values.pop("id", str(uuid.uuid4()))
    state = values.pop("state", {
        "request_id": task_id,
        "fingerprint": "seed",
        "original_prompt": "Do the work",
        "execution_prompt": "Do the work",
        "run_number": 0,
        "generation": 0,
        "steering": [],
        "progress": [],
        "sources": [],
        "artifacts": [],
        "pending_actions": [],
        "resolved_operations": [],
    })
    async with async_session_maker() as db:
        db.add(BuildJob(
            id=task_id,
            user_id=user_id,
            title="Do the work",
            prompt="Do the work",
            job_type="agent_task",
            source_kind=SOURCE_KIND,
            source_id=conversation_id,
            conversation_id=conversation_id,
            status=status,
            config_json={"voice": state},
            state_revision=1,
            created_at=datetime.utcnow(),
            **values,
        ))
        await db.commit()
    return task_id


async def wait_status(service, user_id, task_id, wanted, timeout=2.0):
    deadline = time.monotonic() + timeout
    wanted = set(wanted if isinstance(wanted, tuple) else (wanted,))
    snapshot = None
    while time.monotonic() < deadline:
        snapshot = await service.get(user_id=user_id, task_id=task_id)
        if snapshot["status"] in wanted:
            return snapshot
        await asyncio.sleep(0.01)
    handle = service.handles.get(task_id)
    detail = None
    if handle and handle.task:
        detail = {
            "done": handle.task.done(),
            "cancelled": handle.task.cancelled(),
            "stack": [f"{f.f_code.co_name}:{f.f_lineno}" for f in handle.task.get_stack()],
        }
        if handle.task.done() and not handle.task.cancelled():
            detail["exception"] = repr(handle.task.exception())
    raise AssertionError(f"task did not reach {wanted}: {snapshot}; handle={detail}")


async def close_services(*services):
    await asyncio.gather(*(service.close() for service in services), return_exceptions=True)


async def test_exact_request_id_is_idempotent_but_similar_text_is_new_work():
    user_id, conversation_id = await seed_conversation()
    runner = GateRunner()
    service = VoiceTaskService(runner, poll_seconds=30)
    try:
        first = await service.submit(
            user_id=user_id, session_id=conversation_id,
            request_id="provider-call-1", message="Research Toronto",
        )
        await runner.entered.wait()
        repeated = await service.submit(
            user_id=user_id, session_id=conversation_id,
            request_id="provider-call-1", message="Research Toronto",
        )
        assert repeated["task_id"] == first["task_id"]
        assert len(runner.calls) == 1

        with pytest.raises(VoiceTaskError, match="different input") as conflict:
            await service.submit(
                user_id=user_id, session_id=conversation_id,
                request_id="provider-call-1", message="Research Montreal",
            )
        assert conflict.value.status_code == 409

        second = await service.submit(
            user_id=user_id, session_id=conversation_id,
            request_id="provider-call-2", message="Research Toronto ",
        )
        assert second["task_id"] != first["task_id"]
        assert task_id_for(user_id, conversation_id, "provider-call-1") == first["task_id"]
        runner.release.set()
        await wait_status(service, user_id, first["task_id"], "completed")
        await wait_status(service, user_id, second["task_id"], "completed")
        assert len(runner.calls) == 2
    finally:
        runner.release.set()
        await close_services(service)


async def test_two_owners_atomically_claim_queued_work_once():
    user_id, conversation_id = await seed_conversation()
    task_id = await seed_job(user_id, conversation_id)
    first_runner, second_runner = GateRunner(), GateRunner()
    first = VoiceTaskService(first_runner, poll_seconds=30)
    second = VoiceTaskService(second_runner, poll_seconds=30)
    try:
        claims = await asyncio.gather(first._claim_queued(), second._claim_queued())
        assert sorted(claims) == [0, 1]
        deadline = time.monotonic() + 1
        while not (first_runner.calls or second_runner.calls):
            assert time.monotonic() < deadline
            await asyncio.sleep(0.001)
        assert len(first_runner.calls) + len(second_runner.calls) == 1
        winner = first_runner if first_runner.calls else second_runner
        winner.release.set()
        service = first if first_runner.calls else second
        completed = await wait_status(service, user_id, task_id, "completed")
        assert completed["text"] == "Finished"
        async with async_session_maker() as db:
            messages = (await db.execute(select(Message).where(
                Message.conversation_id == conversation_id,
                Message.source == "voice_task",
            ))).scalars().all()
        assert len(messages) == 1
    finally:
        first_runner.release.set()
        second_runner.release.set()
        await close_services(first, second)


async def test_expired_running_claim_becomes_unknown_and_is_never_replayed():
    user_id, conversation_id = await seed_conversation()
    task_id = await seed_job(
        user_id,
        conversation_id,
        status="running",
        claim_owner="dead-replica",
        claim_token="dead-token",
        claim_expires_at=datetime.utcnow() - timedelta(seconds=1),
    )
    runner = GateRunner()
    service = VoiceTaskService(runner, poll_seconds=30)
    try:
        assert await service.recover_expired(now=datetime.utcnow()) == 1
        snapshot = await service.get(user_id=user_id, task_id=task_id)
        assert snapshot["status"] == "unknown"
        assert snapshot["error_code"] == "lease_expired"
        assert not snapshot["retry_safe"]
        assert await service._claim_queued() == 0
        assert runner.calls == []
    finally:
        runner.release.set()
        await close_services(service)


async def test_stale_worker_cannot_overwrite_unknown_after_lease_loss():
    user_id, conversation_id = await seed_conversation()
    task_id = await seed_job(user_id, conversation_id)
    runner = GateRunner(result("stale success", model="stale-model"))
    stale = VoiceTaskService(runner, poll_seconds=30, lease_seconds=30)
    reaper = VoiceTaskService(GateRunner(), poll_seconds=30)
    try:
        assert await stale._claim_queued() == 1
        await runner.entered.wait()
        async with async_session_maker() as db:
            await db.execute(update(BuildJob).where(BuildJob.id == task_id).values(
                claim_expires_at=datetime.utcnow() - timedelta(seconds=1),
            ))
            await db.commit()
        assert await reaper._expire_running() == 1
        runner.release.set()
        await asyncio.sleep(0.05)
        snapshot = await reaper.get(user_id=user_id, task_id=task_id)
        assert snapshot["status"] == "unknown"
        assert snapshot["model"] != "stale-model"
        assert "stale success" not in snapshot["text"]
    finally:
        runner.release.set()
        await close_services(stale, reaper)


async def test_delivery_receipt_and_spoken_cursors_do_not_change_semantic_revision():
    user_id, conversation_id = await seed_conversation()
    runner = GateRunner()
    service = VoiceTaskService(runner, poll_seconds=30)
    try:
        snapshot = await service.submit(
            user_id=user_id, session_id=conversation_id,
            request_id="receipt-test", message="Hold",
        )
        revision = snapshot["revision"]
        delivered = await service.ack(
            user_id=user_id, task_id=snapshot["task_id"],
            revision=revision + 100, kind="delivery",
        )
        received = await service.ack(
            user_id=user_id, task_id=snapshot["task_id"],
            revision=revision, kind="received",
        )
        spoken = await service.ack(
            user_id=user_id, task_id=snapshot["task_id"],
            revision=revision, kind="spoken",
        )
        assert delivered["revision"] == received["revision"] == spoken["revision"] == revision
        assert delivered["delivery_revision"] == revision
        assert received["receipt_revision"] == revision
        assert spoken["spoken_revision"] == revision
    finally:
        runner.release.set()
        await close_services(service)


async def test_running_correction_is_deduplicated_and_reaches_safe_barrier():
    user_id, conversation_id = await seed_conversation()
    runner = CorrectionRunner()
    service = VoiceTaskService(runner, poll_seconds=30)
    try:
        snapshot = await service.submit(
            user_id=user_id, session_id=conversation_id,
            request_id="correction-task", message="Research all of Canada",
        )
        await runner.entered.wait()
        first = await service.steer(
            user_id=user_id, task_id=snapshot["task_id"],
            request_id="correction-1", message="Only Toronto, please",
        )
        repeated = await service.steer(
            user_id=user_id, task_id=snapshot["task_id"],
            request_id="correction-1", message="Only Toronto, please",
        )
        assert repeated["revision"] == first["revision"]
        with pytest.raises(VoiceTaskError) as conflict:
            await service.steer(
                user_id=user_id, task_id=snapshot["task_id"],
                request_id="correction-1", message="Only Ottawa",
            )
        assert conflict.value.status_code == 409
        runner.check.set()
        done = await wait_status(service, user_id, snapshot["task_id"], "completed")
        assert runner.corrections == ["Only Toronto, please"]
        assert done["steering"] == [{"request_id": "correction-1", "status": "applied"}]
    finally:
        runner.check.set()
        await close_services(service)


async def test_finish_requeue_waits_for_old_handle_cleanup_before_reclaim():
    user_id, conversation_id = await seed_conversation()
    runner = RequeueRunner()
    service = VoiceTaskService(runner, poll_seconds=30)
    original_barrier = service._assert_final_barrier
    injected = False

    async def barrier(task_id, barrier_user_id, handle):
        nonlocal injected
        await original_barrier(task_id, barrier_user_id, handle)
        if not injected:
            injected = True
            await service.steer(
                user_id=user_id,
                task_id=task_id,
                request_id="late-correction",
                message="Use only the corrected scope",
            )

    service._assert_final_barrier = barrier
    try:
        submitted = await service.submit(
            user_id=user_id,
            session_id=conversation_id,
            request_id="finish-race",
            message="Use the old scope",
        )
        done = await wait_status(service, user_id, submitted["task_id"], "completed")
        assert done["text"] == "Result from the corrected scope"
        assert runner.corrections == ["Use only the corrected scope"]
        assert len(runner.calls) == 2
        assert runner.max_current == 1
        assert done["steering"] == [
            {"request_id": "late-correction", "status": "applied"}
        ]
    finally:
        await close_services(service)


async def test_completed_task_correction_advances_generation_through_full_lifecycle():
    user_id, conversation_id = await seed_conversation()
    runner = TerminalCorrectionRunner()
    service = VoiceTaskService(runner, poll_seconds=30)
    try:
        submitted = await service.submit(
            user_id=user_id, session_id=conversation_id,
            request_id="terminal-correction", message="Prepare the report",
        )
        first = await wait_status(service, user_id, submitted["task_id"], "completed")
        assert first["generation"] == 0

        queued = await service.steer(
            user_id=user_id, task_id=submitted["task_id"],
            request_id="terminal-correction-1", message="Use the corrected scope",
        )
        assert queued["status"] == "queued"
        assert queued["generation"] == 1
        assert queued["revision"] > first["revision"]

        await runner.second_entered.wait()
        running = await service.get(user_id=user_id, task_id=submitted["task_id"])
        assert running["status"] == "running"
        assert running["generation"] == 1
        runner.release_second.set()

        completed = await wait_status(
            service, user_id, submitted["task_id"], "completed",
        )
        assert completed["generation"] == 1
        assert completed["revision"] > running["revision"]
        assert completed["text"] == "Corrected completed result"
        assert runner.corrections == ["Use the corrected scope"]
    finally:
        runner.release_second.set()
        await close_services(service)


def action(action_id, tool_name, status="pending", **extra):
    return {
        "action_id": action_id,
        "tool_name": tool_name,
        "connector_id": "gmail",
        "payload": {"operation": action_id},
        "status": status,
        **extra,
    }


async def test_lost_callback_and_multiple_approvals_resume_once_without_repeating_action():
    user_id, conversation_id = await seed_conversation()
    first_result = result(
        "Two actions need review",
        metadata={"pending_actions": [
            action("mail-1", "gmail_send_email"),
            action("event-1", "google_calendar_create_event"),
        ]},
    )
    runner = SequenceRunner([first_result, result("All decisions reconciled")])
    remote = {
        "mail-1": action("mail-1", "gmail_send_email", "executed", result={"id": "m1"}),
        "event-1": action("event-1", "google_calendar_create_event", "pending"),
    }

    async def read_action(_, action_id):
        return remote[action_id]

    service = VoiceTaskService(runner, poll_seconds=30, action_reader=read_action)
    try:
        submitted = await service.submit(
            user_id=user_id, session_id=conversation_id,
            request_id="approvals", message="Send mail and create event",
        )
        waiting = await wait_status(
            service, user_id, submitted["task_id"], "waiting_on_user",
        )
        assert len(waiting["metadata"]["pending_actions"]) == 2

        await service.tick_once()
        assert (await service.get(
            user_id=user_id, task_id=submitted["task_id"],
        ))["status"] == "waiting_on_user"
        assert len(runner.calls) == 1

        remote["event-1"] = action(
            "event-1", "google_calendar_create_event", "rejected",
        )
        await service.tick_once()
        done = await wait_status(service, user_id, submitted["task_id"], "completed")
        assert done["text"] == "All decisions reconciled"
        assert len(runner.calls) == 2
        operations = runner.calls[1]["managed_resolved_operations"]
        assert {(item["tool_name"], item["status"]) for item in operations} == {
            ("gmail_send_email", "executed"),
            ("google_calendar_create_event", "rejected"),
        }
        # A duplicate/lost callback after completion cannot queue a third run.
        assert await service.resolve_pending_action(
            action_id="mail-1", outcome="executed",
        ) == 0
        await service.tick_once()
        assert len(runner.calls) == 2
    finally:
        await close_services(service)


@pytest.mark.parametrize("callback_first", [True, False])
async def test_edited_approval_payload_is_authoritative_in_callback_and_poll_order(
    callback_first,
):
    user_id, conversation_id = await seed_conversation()
    original_payload = {
        "to": "draft@example.test", "subject": "Draft", "body": "Draft body",
    }
    approved_payload = {
        "to": "approved@example.test", "subject": "Approved",
        "body": "Approved body",
    }
    runner = SequenceRunner([
        result("Needs review", metadata={"pending_actions": [action(
            "mail-edited", "gmail_send_email", payload=original_payload,
        )]}),
        result("Approved edit completed"),
    ])
    remote = action(
        "mail-edited", "gmail_send_email", "executed",
        payload=approved_payload, result={"kind": "ok", "id": "sent-approved"},
    )

    async def read_action(_, __):
        return remote

    service = VoiceTaskService(runner, poll_seconds=30, action_reader=read_action)
    try:
        submitted = await service.submit(
            user_id=user_id, session_id=conversation_id,
            request_id=f"edited-approval-{callback_first}", message="Send the draft",
        )
        await wait_status(service, user_id, submitted["task_id"], "waiting_on_user")
        if callback_first:
            assert await service.resolve_pending_action(
                action_id="mail-edited", outcome="executed",
            ) == 1
        else:
            await service.tick_once()

        done = await wait_status(service, user_id, submitted["task_id"], "completed")
        if not callback_first:
            assert await service.resolve_pending_action(
                action_id="mail-edited", outcome="executed",
            ) == 0

        card = done["metadata"]["pending_actions"][0]
        assert card["action_id"] == "mail-edited"
        assert card["payload"] == approved_payload
        assert card["staged_payload"] == original_payload

        operations = runner.calls[1]["managed_resolved_operations"]
        assert len(operations) == 1
        operation = operations[0]
        approved_key = tool_operation_key("gmail_send_email", approved_payload)
        staged_key = tool_operation_key("gmail_send_email", original_payload)
        assert operation["operation_id"] == "mail-edited"
        assert operation["operation_key"] == approved_key
        assert operation["staged_operation_key"] == staged_key
        assert operation["operation_aliases"] == [staged_key]
        assert "sent-approved" in operation["result"]
    finally:
        await close_services(service)


async def test_successful_write_invalidates_only_affected_persisted_read_checkpoint():
    user_id, conversation_id = await seed_conversation()
    service = VoiceTaskService(SequenceRunner([]), poll_seconds=30)
    token = "claim-token"
    report_read = {
        "operation_key": tool_operation_key("read_file", {"path": "report.txt"}),
        "tool_name": "read_file", "status": "executed", "ok": True,
        "result": "old report",
        **tool_operation_checkpoint("read_file", {"path": "report.txt"}),
    }
    notes_read = {
        "operation_key": tool_operation_key("read_file", {"path": "notes.txt"}),
        "tool_name": "read_file", "status": "executed", "ok": True,
        "result": "stable notes",
        **tool_operation_checkpoint("read_file", {"path": "notes.txt"}),
    }
    state = {
        "request_id": "read-freshness", "fingerprint": "read-freshness",
        "original_prompt": "Update report", "execution_prompt": "Update report",
        "run_number": 1, "generation": 0, "steering": [], "progress": [],
        "sources": [], "artifacts": [], "pending_actions": [],
        "resolved_operations": [report_read, notes_read],
    }
    task_id = await seed_job(
        user_id, conversation_id, status="running", state=state,
        claim_owner=service.owner, claim_token=token,
        claim_expires_at=datetime.utcnow() + timedelta(minutes=1),
    )
    write_input = {"path": "report.txt", "content": "new report"}
    write_operation = {
        "operation_key": tool_operation_key("write_file", write_input),
        "operation_id": "write-report", "tool_name": "write_file",
        "status": "executed", "ok": True, "result": "Write succeeded",
        "tool_input": write_input,
        **tool_operation_checkpoint("write_file", write_input),
    }
    try:
        await service._progress(
            task_id, user_id, token,
            {"type": "tool.end", "name": "write_file", "ok": True},
            resolved_operation=write_operation,
        )
        async with async_session_maker() as db:
            job = await db.get(BuildJob, task_id)
            operations = job.config_json["voice"]["resolved_operations"]
        keys = {item["operation_key"] for item in operations}
        assert report_read["operation_key"] not in keys
        assert notes_read["operation_key"] in keys
        assert write_operation["operation_key"] in keys
        assert all("tool_input" not in item for item in operations)
    finally:
        await close_services(service)


async def test_volatile_native_reads_do_not_enter_durable_operation_ledger():
    """The service reclassifies mixed-version events before persisting them."""
    user_id, conversation_id = await seed_conversation()
    service = VoiceTaskService(SequenceRunner([]), poll_seconds=30)
    token = "claim-token"
    state = {
        "request_id": "volatile-native", "fingerprint": "volatile-native",
        "original_prompt": "Poll native work", "execution_prompt": "Poll native work",
        "run_number": 1, "generation": 0, "steering": [], "progress": [],
        "sources": [], "artifacts": [], "pending_actions": [],
        "resolved_operations": [],
    }
    task_id = await seed_job(
        user_id, conversation_id, status="running", state=state,
        claim_owner=service.owner, claim_token=token,
        claim_expires_at=datetime.utcnow() + timedelta(minutes=1),
    )
    observations = [
        ("process", {"action": "status", "process_id": "controlled"}),
        ("process", {"action": "output", "process_id": "controlled"}),
        ("process", {"action": "list"}),
        ("sessions_list", {}),
        ("session_status", {}),
        ("lanes_status", {}),
    ]
    start_input = {"action": "start", "command": "controlled-server"}
    try:
        for index, (name, payload) in enumerate(observations):
            # Simulate an older producer that mislabeled the observation as a
            # mutation.  The service still has the input needed to reject it.
            await service._progress(
                task_id, user_id, token,
                {"type": "tool.end", "name": name, "ok": True},
                resolved_operation={
                    "operation_key": tool_operation_key(name, payload),
                    "operation_id": f"observe-{index}", "tool_name": name,
                    "status": "executed", "ok": True,
                    "operation_kind": "mutation", "result": "stale",
                    "tool_input": payload,
                },
            )
        await service._progress(
            task_id, user_id, token,
            {"type": "tool.end", "name": "process", "ok": True},
            resolved_operation={
                "operation_key": tool_operation_key("process", start_input),
                "operation_id": "process-start", "tool_name": "process",
                "status": "executed", "ok": True,
                "operation_kind": "read", "result": "started",
                "tool_input": start_input,
            },
        )
        async with async_session_maker() as db:
            job = await db.get(BuildJob, task_id)
            operations = job.config_json["voice"]["resolved_operations"]
        assert len(operations) == 1
        assert operations[0]["operation_key"] == tool_operation_key(
            "process", start_input,
        )
        assert operations[0]["operation_kind"] == "mutation"
        assert tool_operation_kind("process", start_input) == "mutation"
        assert "tool_input" not in operations[0]
    finally:
        await close_services(service)


@pytest.mark.parametrize(
    "direction", ["relative-read", "relative-write", "document-rewrite"],
)
async def test_persisted_file_alias_write_invalidates_only_same_native_target(
    tmp_path, direction,
):
    from app.agent.tool_executor import ToolExecutor

    user_id, conversation_id = await seed_conversation()
    runner = SequenceRunner([])
    runner.tools = ToolExecutor(workspace=str(tmp_path))
    runner.tools.set_user_id(user_id)
    service = VoiceTaskService(runner, poll_seconds=30)
    token = "claim-token"
    target = (
        tmp_path / "generated" / "report.pdf"
        if direction == "document-rewrite"
        else tmp_path / "state.json"
    )
    if direction == "relative-read":
        read_input = {"path": "./state.json"}
        write_input = {"path": str(target), "content": "new value"}
    elif direction == "relative-write":
        read_input = {"path": str(target)}
        write_input = {"path": "nested/../state.json", "content": "new value"}
    else:
        read_input = {"path": "generated/report.pdf"}
        write_input = {"path": "report.pdf", "content": "new value"}
    notes_input = {"path": "notes.txt"}

    def old_lexical_read(payload, result):
        return {
            "operation_key": tool_operation_key("read_file", payload),
            "tool_name": "read_file", "status": "executed", "ok": True,
            "result": result,
            **tool_operation_checkpoint("read_file", payload),
        }

    stale_read = old_lexical_read(read_input, "old value")
    unrelated_read = old_lexical_read(notes_input, "stable notes")
    state = {
        "request_id": f"file-alias-{direction}",
        "fingerprint": f"file-alias-{direction}",
        "original_prompt": "Update state", "execution_prompt": "Update state",
        "run_number": 1, "generation": 0, "steering": [], "progress": [],
        "sources": [], "artifacts": [], "pending_actions": [],
        "resolved_operations": [stale_read, unrelated_read],
    }
    task_id = await seed_job(
        user_id, conversation_id, status="running", state=state,
        claim_owner=service.owner, claim_token=token,
        claim_expires_at=datetime.utcnow() + timedelta(minutes=1),
    )
    write_operation = {
        "operation_key": tool_operation_key("write_file", write_input),
        "operation_id": f"write-{direction}", "tool_name": "write_file",
        "status": "executed", "ok": True, "result": "Write succeeded",
        "tool_input": write_input,
        **tool_operation_checkpoint(
            "write_file",
            write_input,
            filesystem_path_resolver=runner.tools.resolve_operation_path,
        ),
    }
    try:
        await service._progress(
            task_id, user_id, token,
            {"type": "tool.end", "name": "write_file", "ok": True},
            resolved_operation=write_operation,
        )
        async with async_session_maker() as db:
            job = await db.get(BuildJob, task_id)
            operations = job.config_json["voice"]["resolved_operations"]
        keys = {item["operation_key"] for item in operations}
        assert stale_read["operation_key"] not in keys
        assert unrelated_read["operation_key"] in keys
        assert write_operation["operation_key"] in keys
    finally:
        await close_services(service)


async def test_explicit_correction_refreshes_reads_but_preserves_mutation_guards():
    user_id, conversation_id = await seed_conversation()
    service = VoiceTaskService(SequenceRunner([]), poll_seconds=30)
    read = {
        "operation_key": tool_operation_key("read_file", {"path": "report.txt"}),
        "tool_name": "read_file", "status": "executed", "ok": True,
        "result": "old report",
        **tool_operation_checkpoint("read_file", {"path": "report.txt"}),
    }
    write_input = {"path": "report.txt", "content": "old write"}
    mutation = {
        "operation_key": tool_operation_key("write_file", write_input),
        "tool_name": "write_file", "status": "executed", "ok": True,
        "result": "write complete",
        **tool_operation_checkpoint("write_file", write_input),
    }
    state = {
        "request_id": "refresh-correction", "fingerprint": "refresh-correction",
        "original_prompt": "Build report", "execution_prompt": "Build report",
        "run_number": 1, "generation": 0, "steering": [], "progress": [],
        "sources": [], "artifacts": [], "pending_actions": [],
        "resolved_operations": [read, mutation],
    }
    task_id = await seed_job(
        user_id, conversation_id, status="completed", state=state,
        completed_at=datetime.utcnow(),
    )
    try:
        queued = await service.steer(
            user_id=user_id, task_id=task_id, request_id="refresh-1",
            message="Refresh the report before continuing",
        )
        assert queued["status"] == "queued"
        async with async_session_maker() as db:
            job = await db.get(BuildJob, task_id)
            operations = job.config_json["voice"]["resolved_operations"]
        assert [item["operation_key"] for item in operations] == [
            mutation["operation_key"],
        ]
    finally:
        await close_services(service)


async def test_approved_operation_without_terminal_record_becomes_unknown():
    user_id, conversation_id = await seed_conversation()
    runner = SequenceRunner([result(
        "Needs review",
        metadata={"pending_actions": [action("mail-stuck", "gmail_send_email")]},
    )])
    old = (datetime.utcnow() - timedelta(minutes=10)).isoformat()

    approval_visible = False

    async def read_action(_, action_id):
        if not approval_visible:
            return action(action_id, "gmail_send_email", "pending")
        return action(action_id, "gmail_send_email", "approved", decided_at=old)

    service = VoiceTaskService(runner, poll_seconds=30, action_reader=read_action)
    try:
        submitted = await service.submit(
            user_id=user_id, session_id=conversation_id,
            request_id="stuck-approval", message="Send it",
        )
        await wait_status(service, user_id, submitted["task_id"], "waiting_on_user")
        approval_visible = True
        await service.tick_once()
        unknown = await wait_status(service, user_id, submitted["task_id"], "unknown")
        assert unknown["error_code"] == "approval_outcome_unknown"
        assert len(runner.calls) == 1
    finally:
        await close_services(service)


async def test_cancel_approval_race_retains_authoritative_execution_without_rerun():
    user_id, conversation_id = await seed_conversation()
    runner = SequenceRunner([result(
        "Needs review",
        metadata={"pending_actions": [action("mail-race", "gmail_send_email")]},
    )])

    approval_won = False

    async def cancel_action(_, action_id):
        nonlocal approval_won
        # Approval won the platform's guarded UPDATE before cancellation.
        approval_won = True
        return action(action_id, "gmail_send_email", "approved")

    async def read_action(_, action_id):
        if not approval_won:
            return action(action_id, "gmail_send_email", "pending")
        return action(action_id, "gmail_send_email", "executed", result={"id": "sent"})

    service = VoiceTaskService(
        runner, poll_seconds=30,
        action_reader=read_action, action_canceller=cancel_action,
    )
    try:
        submitted = await service.submit(
            user_id=user_id, session_id=conversation_id,
            request_id="cancel-race", message="Send it",
        )
        await wait_status(service, user_id, submitted["task_id"], "waiting_on_user")
        cancelled = await service.cancel(
            user_id=user_id, task_id=submitted["task_id"],
        )
        assert cancelled["status"] == "cancelled"
        cards = cancelled["metadata"]["pending_actions"]
        assert cards[0]["status"] == "executed"
        assert cards[0]["result"] == {"id": "sent"}
        assert len(runner.calls) == 1
    finally:
        await close_services(service)


async def test_persisted_cancel_retries_one_time_platform_failure():
    user_id, conversation_id = await seed_conversation()
    runner = SequenceRunner([result(
        "Needs review",
        metadata={"pending_actions": [action("mail-cancel", "gmail_send_email")]},
    )])
    canceller_calls = 0

    async def cancel_action(_, action_id):
        nonlocal canceller_calls
        canceller_calls += 1
        if canceller_calls == 1:
            raise ConnectionError("temporary platform outage")
        return action(action_id, "gmail_send_email", "rejected")

    async def read_action(_, action_id):
        return action(action_id, "gmail_send_email", "pending")

    service = VoiceTaskService(
        runner, poll_seconds=30,
        action_reader=read_action, action_canceller=cancel_action,
    )
    try:
        submitted = await service.submit(
            user_id=user_id, session_id=conversation_id,
            request_id="cancel-retry", message="Send it",
        )
        await wait_status(service, user_id, submitted["task_id"], "waiting_on_user")
        cancelled = await service.cancel(
            user_id=user_id, task_id=submitted["task_id"],
        )
        assert cancelled["status"] == "cancelled"
        assert canceller_calls == 2
        assert cancelled["metadata"]["pending_actions"][0]["status"] == "rejected"
    finally:
        await close_services(service)


async def test_persisted_cancel_is_reconciled_after_service_restart():
    user_id, conversation_id = await seed_conversation()
    runner = SequenceRunner([result(
        "Needs review",
        metadata={"pending_actions": [action("mail-restart", "gmail_send_email")]},
    )])

    async def unavailable_cancel(_, __):
        raise ConnectionError("platform unavailable")

    async def pending(_, action_id):
        return action(action_id, "gmail_send_email", "pending")

    first = VoiceTaskService(
        runner, poll_seconds=30,
        action_reader=pending, action_canceller=unavailable_cancel,
    )
    second = None
    try:
        submitted = await first.submit(
            user_id=user_id, session_id=conversation_id,
            request_id="cancel-restart", message="Send it",
        )
        await wait_status(first, user_id, submitted["task_id"], "waiting_on_user")
        cancelling = await first.cancel(
            user_id=user_id, task_id=submitted["task_id"],
        )
        assert cancelling["status"] == "cancelling"
        await first.close()

        canceller_calls = 0

        async def recovered_cancel(_, action_id):
            nonlocal canceller_calls
            canceller_calls += 1
            return action(action_id, "gmail_send_email", "rejected")

        second = VoiceTaskService(
            SequenceRunner([]), poll_seconds=30,
            action_reader=pending, action_canceller=recovered_cancel,
        )
        assert await second._reconcile_waiting() == 1
        cancelled = await second.get(
            user_id=user_id, task_id=submitted["task_id"],
        )
        assert cancelled["status"] == "cancelled"
        assert canceller_calls == 1
    finally:
        await close_services(first, *(tuple([second]) if second else ()))
