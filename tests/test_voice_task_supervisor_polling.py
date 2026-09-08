"""Load and lifecycle regressions for the durable voice-task supervisor."""
from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import event, select, update

from app.agent.voice_tasks import (
    SOURCE_KIND,
    VoiceTaskService,
    voice_task_supervisor_boot_allowed,
)
from app.db import async_session_maker
from app.db.database import get_engine
from app.db.models import BuildJob, Conversation, User


@pytest.fixture(autouse=True)
async def _drain_voice_task_bg():
    """Cancel any voice-task background task that outlived the test.

    `VoiceTaskService.close()` stops the SUPERVISOR, but a submitted task's own
    tasks (`handle.task`, `handle.deadline_task` in voice_tasks.py) are left
    running on purpose — accepted work is not cancelled by a socket close in
    production. Under CI's shared in-memory DB that is a cross-test hazard: a
    leaked task writes a status row to `build_jobs` after the next test's
    conftest fixture has dropped the schema and before it recreates it, and the
    write fails "no such table: build_jobs" — a rare, timing-only flake that
    passed on the PR run and failed on the merge run. Cancelling here makes the
    suite deterministic; production keeps the accepted-work behaviour. Same
    idea as test_voice_turn_survives draining its LA-end task.
    """
    yield
    import asyncio as _asyncio
    leaked = [
        t for t in _asyncio.all_tasks()
        if not t.done()
        and "voice_tasks" in (
            (t.get_coro() is not None
             and getattr(t.get_coro(), "cr_code", None) is not None
             and t.get_coro().cr_code.co_filename) or ""
        )
    ]
    for t in leaked:
        t.cancel()
    if leaked:
        await _asyncio.gather(*leaked, return_exceptions=True)


def _result(text: str = "Finished") -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        model="test-model",
        tokens_input=1,
        tokens_output=1,
        tokens_total=2,
        stopped_reason="",
        metadata={},
    )


class _GateRunner:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.calls: list[dict] = []

    async def run(self, **kwargs):
        self.calls.append(kwargs)
        self.entered.set()
        await self.release.wait()
        return _result()


class _NeverRunner:
    async def run(self, **_kwargs):  # pragma: no cover - a claim is a failure
        raise AssertionError("idle supervisor claimed unexpected work")


class _CountingSessionMaker:
    def __init__(self, delegate) -> None:
        self.delegate = delegate
        self.acquisitions = 0

    def __call__(self):
        self.acquisitions += 1
        return self.delegate()


async def _seed_conversation() -> tuple[str, str]:
    user_id, conversation_id = str(uuid.uuid4()), str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(
            id=user_id,
            email=f"voice-poll-{user_id[:8]}@example.test",
            hashed_password="x",
            name="Voice Poll Test",
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


async def _seed_job(
    user_id: str,
    conversation_id: str,
    *,
    status: str = "queued",
    state: dict | None = None,
    **values,
) -> str:
    task_id = str(uuid.uuid4())
    voice = state or {
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
    }
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
            config_json={"voice": voice},
            state_revision=1,
            created_at=datetime.utcnow(),
            **values,
        ))
        await db.commit()
    return task_id


async def _wait_status(
    service: VoiceTaskService,
    user_id: str,
    task_id: str,
    wanted: str,
    *,
    timeout: float = 1.0,
) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = await service.get(user_id=user_id, task_id=task_id)
        if snapshot["status"] == wanted:
            return snapshot
        await asyncio.sleep(0.01)
    raise AssertionError(f"voice task {task_id} did not reach {wanted}")


async def test_idle_tick_uses_one_session_and_one_open_work_census():
    sessions = _CountingSessionMaker(async_session_maker)
    statements: list[str] = []

    def record_statement(_conn, _cursor, statement, _parameters, _context, _many):
        if "build_jobs" in statement.lower():
            statements.append(statement)

    engine = get_engine()
    event.listen(engine.sync_engine, "before_cursor_execute", record_statement)
    service = VoiceTaskService(
        _NeverRunner(), session_maker=sessions, claim_allowed=lambda: True,
    )
    try:
        assert await service.tick_once() == {
            "expired": 0, "reconciled": 0, "claimed": 0,
        }
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", record_statement)
        await service.close()

    assert sessions.acquisitions == 1
    assert len(statements) == 1
    assert "status" in statements[0].lower()


async def test_empty_supervisor_backs_off_acquisitions_over_real_time():
    sessions = _CountingSessionMaker(async_session_maker)
    service = VoiceTaskService(
        _NeverRunner(),
        session_maker=sessions,
        poll_seconds=0.05,
        idle_poll_seconds=0.4,
        claim_allowed=lambda: True,
    )
    try:
        await service.start()
        await asyncio.sleep(0.36)
        # Startup + nominal 100 ms + nominal 200 ms sweeps. The old fixed
        # 50 ms loop would acquire at least seven sessions in this window.
        assert sessions.acquisitions <= 3
    finally:
        await service.close()


@pytest.mark.parametrize(
    ("enabled", "bound", "passive", "expected"),
    [
        (False, True, False, False),
        (True, False, False, False),  # generic pool lobby / unbound
        (True, True, True, False),   # blue-green passive slot
        (True, True, False, True),   # serving tenant
    ],
)
def test_boot_discovery_requires_a_serving_bound_tenant(
    enabled: bool, bound: bool, passive: bool, expected: bool,
):
    assert voice_task_supervisor_boot_allowed(
        enabled=enabled, bound=bound, passive=passive,
    ) is expected


async def test_submit_from_idle_starts_immediately_and_wakes_first_heartbeat():
    user_id, conversation_id = await _seed_conversation()
    runner = _GateRunner()
    service = VoiceTaskService(
        runner,
        poll_seconds=30,
        idle_poll_seconds=30,
        lease_seconds=1,
        claim_allowed=lambda: True,
    )
    heartbeat = asyncio.Event()
    original_heartbeat = service._heartbeat_owned_in_session

    async def observe_heartbeat(db, **kwargs):
        if service._has_live_handles():
            heartbeat.set()
        await original_heartbeat(db, **kwargs)

    service._heartbeat_owned_in_session = observe_heartbeat
    try:
        submitted = await asyncio.wait_for(service.submit(
            user_id=user_id,
            session_id=conversation_id,
            request_id="wake-from-idle",
            message="Hold this task",
        ), timeout=1.0)
        await asyncio.wait_for(runner.entered.wait(), timeout=0.25)
        await asyncio.wait_for(heartbeat.wait(), timeout=0.25)
        assert service._supervisor is not None
        assert service.poll_seconds <= service.lease_seconds / 3

        cancelling = await service.cancel(
            user_id=user_id, task_id=submitted["task_id"],
        )
        assert cancelling["status"] == "cancelling"
        runner.release.set()
        cancelled = await _wait_status(
            service, user_id, submitted["task_id"], "cancelled",
        )
        assert cancelled["error_code"] == "user_cancelled"
    finally:
        runner.release.set()
        await service.close()


async def test_pending_waiting_work_uses_bounded_fallback_cadence():
    user_id, conversation_id = await _seed_conversation()
    action_id = str(uuid.uuid4())
    await _seed_job(
        user_id,
        conversation_id,
        status="waiting_on_user",
        state={
            "request_id": "waiting",
            "fingerprint": "waiting",
            "original_prompt": "Send it",
            "execution_prompt": "Send it",
            "run_number": 1,
            "generation": 0,
            "steering": [],
            "progress": [],
            "sources": [],
            "artifacts": [],
            "resolved_operations": [],
            "pending_actions": [{"action_id": action_id, "status": "pending"}],
        },
    )
    reads: list[float] = []

    async def read_pending(_user_id: str, requested_id: str) -> dict:
        reads.append(time.monotonic())
        return {"action_id": requested_id, "status": "pending"}

    service = VoiceTaskService(
        _NeverRunner(),
        poll_seconds=0.05,
        waiting_poll_seconds=0.06,
        idle_poll_seconds=1.0,
        action_reader=read_pending,
        claim_allowed=lambda: True,
    )
    try:
        await service.start()
        await asyncio.sleep(0.24)
        # No callback arrived, yet durable fallback reads stay on the bounded
        # waiting cadence instead of drifting to the one-second idle ceiling.
        assert len(reads) >= 4
        assert service._waiting_work
        assert service._next_poll_delay(1.0, activity=False) == pytest.approx(0.06)
        assert service._poll_timeout(0.06) <= 0.0661
    finally:
        await service.close()


async def test_poison_expiry_cannot_roll_back_a_live_heartbeat():
    user_id, conversation_id = await _seed_conversation()
    runner = _GateRunner()
    service = VoiceTaskService(
        runner, lease_seconds=90, poll_seconds=30, claim_allowed=lambda: True,
    )
    live_id = await _seed_job(user_id, conversation_id)
    try:
        assert await service._claim_queued() == 1
        await asyncio.wait_for(runner.entered.wait(), timeout=0.5)
        short_expiry = datetime.utcnow() + timedelta(seconds=10)
        async with async_session_maker() as db:
            await db.execute(update(BuildJob).where(
                BuildJob.id == live_id,
            ).values(claim_expires_at=short_expiry))
            await db.commit()
        poison_id = await _seed_job(
            user_id,
            conversation_id,
            status="running",
            claim_owner="dead-owner",
            claim_token="dead-token",
            claim_expires_at=datetime.utcnow() - timedelta(seconds=1),
        )

        original_persist = service._persist_message

        async def poison_one(db, job, state, result=None):
            if job.id == poison_id:
                raise RuntimeError("poison expiry fixture")
            await original_persist(db, job, state, result=result)

        service._persist_message = poison_one
        tick = await service.tick_once()
        assert tick["expired"] == 0

        async with async_session_maker() as db:
            live = await db.get(BuildJob, live_id)
            poison = await db.get(BuildJob, poison_id)
        assert live.claim_expires_at > datetime.utcnow() + timedelta(seconds=80)
        assert poison.status == "running"
        assert poison.claim_expires_at < datetime.utcnow()
    finally:
        runner.release.set()
        await service.close()


async def test_database_tick_failure_cannot_roll_back_live_heartbeat():
    user_id, conversation_id = await _seed_conversation()
    runner = _GateRunner()
    service = VoiceTaskService(
        runner, lease_seconds=90, poll_seconds=30, claim_allowed=lambda: True,
    )
    live_id = await _seed_job(user_id, conversation_id)
    try:
        assert await service._claim_queued() == 1
        await asyncio.wait_for(runner.entered.wait(), timeout=0.5)
        short_expiry = datetime.utcnow() + timedelta(seconds=10)
        async with async_session_maker() as db:
            await db.execute(update(BuildJob).where(
                BuildJob.id == live_id,
            ).values(claim_expires_at=short_expiry))
            await db.commit()

        async def fail_after_heartbeat(*_args, **_kwargs):
            raise RuntimeError("census fixture")

        service._claim_queued_in_session = fail_after_heartbeat
        with pytest.raises(RuntimeError, match="census fixture"):
            await service.tick_once()

        async with async_session_maker() as db:
            live = await db.get(BuildJob, live_id)
        assert live.claim_expires_at > datetime.utcnow() + timedelta(seconds=80)
    finally:
        runner.release.set()
        await service.close()


async def test_claim_gate_stops_new_work_but_keeps_owned_lease_alive():
    user_id, conversation_id = await _seed_conversation()
    runner = _GateRunner()
    allowed = True
    service = VoiceTaskService(
        runner,
        lease_seconds=90,
        poll_seconds=30,
        claim_allowed=lambda: allowed,
    )
    live_id = await _seed_job(user_id, conversation_id)
    try:
        assert await service._claim_queued() == 1
        await asyncio.wait_for(runner.entered.wait(), timeout=0.5)
        queued_id = await _seed_job(user_id, conversation_id)
        short_expiry = datetime.utcnow() + timedelta(seconds=10)
        async with async_session_maker() as db:
            await db.execute(update(BuildJob).where(
                BuildJob.id == live_id,
            ).values(claim_expires_at=short_expiry))
            await db.commit()

        allowed = False
        tick = await service.tick_once()
        assert tick["claimed"] == 0
        async with async_session_maker() as db:
            live = await db.get(BuildJob, live_id)
            queued = await db.get(BuildJob, queued_id)
        assert live.claim_expires_at > datetime.utcnow() + timedelta(seconds=80)
        assert queued.status == "queued"
    finally:
        runner.release.set()
        await service.close()
