"""Durable Live task recovery after the audio socket changes replicas."""

import asyncio
import contextlib
import json
import time
import uuid
from datetime import date, datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import test_live_harness as H


@pytest.mark.asyncio
async def test_first_live_instructions_include_only_own_current_day(monkeypatch):
    from app.agent import voice_context as vc
    from app.config import settings
    from app.db import async_session_maker
    from app.db.models import Conversation, Message, User
    from app.db.models.day_chat import DayChat

    if settings.run_mode != "agent":
        pytest.skip("agent-only tables require RUN_MODE=agent")
    owner, stranger = str(uuid.uuid4()), str(uuid.uuid4())
    owner_session, stranger_session = str(uuid.uuid4()), str(uuid.uuid4())
    owner_day, stranger_day = str(uuid.uuid4()), str(uuid.uuid4())
    owner_words = "Continue the campus research from this chat."
    stranger_words = "PRIVATE OTHER ACCOUNT PHRASE"
    today = date(2026, 9, 24)
    now = datetime(2026, 9, 24, 19, 0, tzinfo=timezone.utc)
    async with async_session_maker() as db:
        db.add_all([
            User(id=owner, email=f"{owner[:8]}@test.local", hashed_password="x" * 60,
                 timezone="America/Toronto"),
            User(id=stranger, email=f"{stranger[:8]}@test.local", hashed_password="x" * 60,
                 timezone="America/Toronto"),
            DayChat(id=owner_day, user_id=owner, local_date=today,
                    timezone="America/Toronto"),
            DayChat(id=stranger_day, user_id=stranger, local_date=today,
                    timezone="America/Toronto"),
            Conversation(id=owner_session, user_id=owner, channel="voice",
                         day_chat_id=owner_day),
            Conversation(id=stranger_session, user_id=stranger, channel="voice",
                         day_chat_id=stranger_day),
            Message(id=str(uuid.uuid4()), conversation_id=owner_session,
                    day_chat_id=owner_day, role="user", content=owner_words),
            Message(id=str(uuid.uuid4()), conversation_id=stranger_session,
                    day_chat_id=stranger_day, role="user", content=stranger_words),
        ])
        await db.commit()
        context = await vc.build_voice_context(
            db, owner, tz_name="America/Toronto", now_utc=now, live=True,
        )
    assert owner_words in context.instructions
    assert stranger_words not in context.instructions

    H.fast_clocks(monkeypatch)
    H.patch_relay(monkeypatch)

    def on_send(provider, event):
        if event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([H.config(), 0.05, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send)
    await H.run_relay(
        client, provider, timeout=5, instructions=context.instructions,
        user_id=owner, db_session_id=owner_session,
    )
    served = provider.of("session.start")[0]["session"]["instructions"]
    assert owner_words in served and stranger_words not in served
    assert "recent conversation already supplied above" in served

    async def broken_day(*_args, **_kwargs):
        raise RuntimeError("test day-read outage")

    monkeypatch.setattr(vc, "_load_newest_day", broken_day)
    async with async_session_maker() as db:
        degraded = await vc.build_voice_context(
            db, owner, tz_name="America/Toronto", now_utc=now, live=True,
        )
    assert "day" in degraded.degraded
    assert degraded.instructions.strip()
    assert owner_words not in degraded.instructions
    assert stranger_words not in degraded.instructions


def test_resumed_opening_uses_conversation_context():
    from app.services.live_voice_protocol import adapt_instructions_for_live

    rendered = adapt_instructions_for_live(
        "# Voice Conversation Mode\n# Today's Full Conversation History\n"
        "User: Please continue the search."
    )
    assert "recent conversation already supplied above" in rendered
    assert "respond to that context directly" in rendered
    assert "do not repeatedly introduce yourself" in rendered


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_answers", [False, True])
async def test_accepted_reask_watch_reports_only_actual_silence(
    monkeypatch, provider_answers,
):
    import app.services.live_voice_protocol as live

    H.fast_clocks(monkeypatch)
    H.patch_relay(monkeypatch)
    monkeypatch.setattr(live, "_REASK_RESPONSE_WAIT_S", 0.08)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("Please answer this question", 0, 300))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([])

    async def reask_once():
        for _ in range(50):
            finals = [f for f in client.of("transcript") if f.get("final")]
            if finals:
                client._script.insert(0, {
                    "type": "inject_text", "text": "Please answer this question",
                    "reask_of_user_turn_id": finals[0]["turn_id"],
                })
                return
            await asyncio.sleep(0.01)
        raise AssertionError("user turn did not finalize")

    config = H.config()
    config["features"].append("reask_turns")
    client._script = [config, reask_once, 0.2, {"type": "stop"}]
    provider = H.FakeProvider(
        on_send=on_send, auto_ack=True,
        speak_on_instructions=provider_answers,
    )
    await H.run_relay(client, provider, timeout=5)

    results = client.of("reask_result")
    assert [f["outcome"] for f in results] == (
        ["accepted"] if provider_answers else ["accepted", "response_delayed"]
    )
    if not provider_answers:
        assert results[1]["user_turn_id"] == results[0]["user_turn_id"]
        assert results[1]["recoverable"] is True
    assert provider.of("session.commentary.append") == []


@pytest.mark.asyncio
async def test_late_reask_output_keeps_exact_parent_after_timeout(monkeypatch):
    import app.services.live_voice_protocol as live

    H.fast_clocks(monkeypatch)
    H.patch_relay(monkeypatch)
    monkeypatch.setattr(live, "_REASK_RESPONSE_WAIT_S", 0.06)
    box = {}

    def on_send(provider, event):
        box["provider"] = provider
        if event["type"] == "session.start":
            provider.push(H.user_delta("Please answer this request", 0, 300))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([])

    async def ask_again():
        for _ in range(50):
            finals = [f for f in client.of("transcript") if f.get("final")]
            if finals:
                client._script.insert(0, {
                    "type": "inject_text", "text": finals[0]["text"],
                    "reask_of_user_turn_id": finals[0]["turn_id"],
                })
                return
            await asyncio.sleep(0.01)
        raise AssertionError("turn did not finalize")

    async def late_answer():
        assert [f["outcome"] for f in client.of("reask_result")] == [
            "accepted", "response_delayed",
        ]
        box["provider"].push(H.out_text("The answer arrived later.", 1000, 1400))
        box["provider"].push(H.out_audio())
        await asyncio.sleep(0.1)

    config = H.config()
    config["features"].append("reask_turns")
    client._script = [config, ask_again, 0.16, late_answer, {"type": "stop"}]
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=5)
    turn_id = [f for f in client.of("transcript") if f.get("final")][0]["turn_id"]
    replies = [f for f in client.of("response_text")
               if "arrived later" in f.get("text", "")]
    assert replies and replies[0]["parent_user_turn_id"] == turn_id


@pytest.mark.asyncio
async def test_the_response_watch_reads_the_latest_reask_append(monkeypatch, caplog):
    """Voice round 2026-10-06 (D-R1 (b)): a turn may now carry TWO reasks
    (the app's, then the relay's — `_REASK_MAX_PER_TURN` is total and real),
    each with its own event id.  The `response delayed` line's `acked=` /
    `ack_ms=` must describe the LATEST note — here the relay's `-2`, which
    GPT-Live took — not the first, which it never acked."""

    import logging

    import app.services.live_voice_protocol as live

    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")
    H.fast_clocks(
        monkeypatch, voice_live_uplink_keeper_enabled=True,
        voice_live_reply_regen_after_s=0.3,
    )
    H.patch_relay(monkeypatch)
    monkeypatch.setattr(live, "_REASK_RESPONSE_WAIT_S", 0.5)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("Please answer this question", 0, 300))
        elif (
            event["type"] == "session.instructions.append"
            and str(event.get("event_id") or "").endswith("-2")
        ):
            provider.push({
                "type": "session.instructions.appended",
                "client_event_id": event["event_id"], "start_ms": 0, "end_ms": 0,
            })
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([])

    async def reask_once():
        for _ in range(50):
            finals = [f for f in client.of("transcript") if f.get("final")]
            if finals:
                client._script.insert(0, {
                    "type": "inject_text", "text": "Please answer this question",
                    "reask_of_user_turn_id": finals[0]["turn_id"],
                })
                return
            await asyncio.sleep(0.01)
        raise AssertionError("user turn did not finalize")

    config = H.config()
    config["features"].append("reask_turns")
    client._script = [config, reask_once, 1.4, {"type": "stop"}]
    provider = H.FakeProvider(on_send=on_send)
    await H.run_relay(client, provider, timeout=6)

    turn = [f for f in client.of("transcript") if f.get("final")][0]["turn_id"]
    reasks = [
        e["event_id"] for e in provider.of("session.instructions.append")
        if str(e.get("event_id") or "").startswith("toup-live-reask-")
    ]
    assert reasks == [f"toup-live-reask-{turn}", f"toup-live-reask-{turn}-2"], reasks
    delayed = [
        r.getMessage() for r in caplog.records
        if r.getMessage().startswith("[LIVE] accepted reask response delayed")
    ]
    assert len(delayed) == 1, delayed
    assert "acked=True" in delayed[0] and delayed[0].rsplit(" ack_ms=", 1)[1] != "-1", delayed


@pytest.mark.asyncio
@pytest.mark.parametrize("send_error", [TypeError, OSError, AssertionError])
async def test_first_delegation_send_failure_has_honest_lifecycle(
    monkeypatch, send_error,
):
    H.fast_clocks(monkeypatch)
    H.patch_relay(monkeypatch)

    class FaultyClient(H.FakeClient):
        failed = False

        async def send_json(self, frame):
            if (not self.failed and frame.get("type") == "delegation"
                    and frame.get("phase") == "created"):
                self.failed = True
                raise send_error("simulated first delegation send")
            await super().send_json(frame)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("Search for campus data", 0, 400))
            provider.push(H.delegation("task-first-send", 450))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = FaultyClient([H.config(), 0.4, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=5)
    assert client.failed
    if send_error is TypeError:
        # A bad single JSON frame is skipped; later lifecycle frames still
        # reach the live phone instead of killing the whole socket.
        assert any(f.get("phase") == "completed" for f in client.of("delegation"))
    else:
        # A true transport failure tears down promptly; the durable answer
        # still lands and a repaired socket can reconcile it.
        assert not client.of("delegation")


def test_status_row_requires_exact_delegation_identity():
    from app.api.api_v1 import _voice_delegation_row

    row = SimpleNamespace(
        content="Verified answer",
        metadata_json=json.dumps({"voice": {
            "source": "delegated", "delegation_id": "task-1", "spoken": False,
        }}),
    )
    assert _voice_delegation_row(row, "task-2") is None
    assert _voice_delegation_row(row, "task-1") == {
        "record_found": True,
        "status": "completed", "answer": "Verified answer", "spoken": False,
        "recovery_speech_claimed": False, "parent_user_turn_id": "",
        "title": "", "tool_events": [],
    }
    row.metadata_json = json.dumps({"voice": {
        "source": "delegated_record", "delegation_id": "task-1",
    }})
    assert _voice_delegation_row(row, "task-1")["status"] == "failed"
    row.metadata_json = json.dumps({"voice": {
        "source": "delegated_record", "delegation_id": "task-1",
        "cancelled": True,
    }})
    assert _voice_delegation_row(row, "task-1")["status"] == "cancelled"


@pytest.mark.asyncio
async def test_tenant_status_is_keyed_to_owner_and_session(monkeypatch):
    from app.api.api_v1 import (
        VoiceDelegationCancelRequest, VoiceDelegationSpeechClaimRequest,
        VoiceDelegationStatusRequest, internal_voice_delegation_cancel,
        internal_voice_delegation_claim_speech, internal_voice_delegations_status,
    )
    from app.config import settings
    from app.db import async_session_maker
    from app.db.models import BuildJob, Conversation, Message, User

    if settings.run_mode != "agent":
        pytest.skip("agent-only tables require RUN_MODE=agent")
    owner = str(uuid.uuid4())
    other = str(uuid.uuid4())
    owner_session = str(uuid.uuid4())
    other_session = str(uuid.uuid4())
    owner_job = str(uuid.uuid4())
    stopped_job = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add_all([
            User(id=owner, email=f"{owner[:8]}@test.local", hashed_password="x" * 60),
            User(id=other, email=f"{other[:8]}@test.local", hashed_password="x" * 60),
            Conversation(id=owner_session, user_id=owner, channel="voice"),
            Conversation(id=other_session, user_id=other, channel="voice"),
            BuildJob(id=owner_job, user_id=owner, conversation_id=owner_session,
                     job_type="agent_task", status="running", title="Research",
                     prompt="Research", config_json={
                         "voice_turn": True, "voice_delegation_id": "same-task",
                     }),
            BuildJob(id=stopped_job, user_id=owner, conversation_id=owner_session,
                     job_type="agent_task", status="cancelled", title="Stopped",
                     prompt="Stopped", config_json={
                         "voice_turn": True, "voice_delegation_id": "stopped-task",
                     }),
            Message(id=str(uuid.uuid4()), conversation_id=owner_session,
                    role="assistant", content="The request was stopped.",
                    metadata_json=json.dumps({"voice": {
                        "source": "delegated_record", "delegation_id": "stopped-task",
                        "cancelled": True,
                    }})),
            Message(id=str(uuid.uuid4()), conversation_id=other_session,
                    role="assistant", content="Private answer",
                    metadata_json=json.dumps({"voice": {
                        "source": "delegated", "delegation_id": "same-task",
                    }})),
        ])
        await db.commit()

    monkeypatch.setattr(settings, "run_mode", "agent")
    monkeypatch.setattr(settings, "agent_api_key", "test-agent-key")
    monkeypatch.setattr(settings, "user_id", owner)
    request = SimpleNamespace(headers={"X-Agent-Key": "test-agent-key"})
    with pytest.raises(HTTPException) as denied:
        await internal_voice_delegations_status(
            VoiceDelegationStatusRequest(session_id=other_session,
                                         task_ids=["same-task"]), request,
        )
    assert denied.value.status_code == 404
    own = await internal_voice_delegations_status(
        VoiceDelegationStatusRequest(session_id=owner_session,
                                     task_ids=["same-task"]), request,
    )
    assert own == {"tasks": {"same-task": {
        "status": "running", "job_id": owner_job, "cancel_requested": False,
    }}}
    stopped = await internal_voice_delegations_status(
        VoiceDelegationStatusRequest(session_id=owner_session,
                                     task_ids=["stopped-task"]), request,
    )
    assert stopped["tasks"]["stopped-task"]["status"] == "cancelled"
    cancel_req = VoiceDelegationCancelRequest(
        session_id=owner_session, task_id="same-task", job_id=owner_job,
    )
    assert (await internal_voice_delegation_cancel(cancel_req, request))["requested"] is True
    assert (await internal_voice_delegation_cancel(cancel_req, request))["requested"] is True
    own = await internal_voice_delegations_status(
        VoiceDelegationStatusRequest(session_id=owner_session,
                                     task_ids=["same-task"]), request,
    )
    assert own["tasks"]["same-task"]["cancel_requested"] is True
    with pytest.raises(HTTPException) as wrong_task:
        await internal_voice_delegation_cancel(
            VoiceDelegationCancelRequest(
                session_id=owner_session, task_id="other-task", job_id=owner_job,
            ), request,
        )
    assert wrong_task.value.status_code == 404
    with pytest.raises(HTTPException) as wrong_session:
        await internal_voice_delegation_cancel(
            VoiceDelegationCancelRequest(
                session_id=other_session, task_id="same-task", job_id=owner_job,
            ), request,
        )
    assert wrong_session.value.status_code == 404
    async with async_session_maker() as db:
        job = await db.get(BuildJob, owner_job)
        job.status = "completed"
        await db.commit()
    assert await internal_voice_delegation_cancel(cancel_req, request) == {
        "requested": False, "status": "completed",
    }

    async with async_session_maker() as db:
        db.add(Message(
            id=str(uuid.uuid4()), conversation_id=owner_session,
            role="assistant", content="Owner's answer",
            metadata_json=json.dumps({
                "voice": {"source": "delegated", "delegation_id": "same-task",
                          "spoken": False},
                "tool_events": [{
                    "tool": "web_search", "started_at_ms": 1,
                    "completed_at_ms": 2,
                    "sources": [{"domain": "example.com", "url": "https://example.com"}],
                }],
            }),
        ))
        await db.commit()
    own = await internal_voice_delegations_status(
        VoiceDelegationStatusRequest(session_id=owner_session,
                                     task_ids=["same-task"]), request,
    )
    assert own["tasks"]["same-task"]["answer"] == "Owner's answer"
    assert own["tasks"]["same-task"]["status"] == "completed"
    assert own["tasks"]["same-task"]["job_id"] == owner_job
    assert own["tasks"]["same-task"]["tool_events"][0]["tool"] == "web_search"

    claim_req = VoiceDelegationSpeechClaimRequest(
        session_id=owner_session, task_id="same-task",
    )
    assert await internal_voice_delegation_claim_speech(claim_req, request) == {
        "claimed": True,
    }
    assert await internal_voice_delegation_claim_speech(claim_req, request) == {
        "claimed": False,
    }
    own = await internal_voice_delegations_status(
        VoiceDelegationStatusRequest(session_id=owner_session,
                                     task_ids=["same-task"]), request,
    )
    assert own["tasks"]["same-task"]["recovery_speech_claimed"] is True
    with pytest.raises(HTTPException) as cross_session:
        await internal_voice_delegation_claim_speech(
            VoiceDelegationSpeechClaimRequest(
                session_id=other_session, task_id="same-task",
            ), request,
        )
    assert cross_session.value.status_code == 404

    with pytest.raises(HTTPException) as bad_key:
        await internal_voice_delegations_status(
            VoiceDelegationStatusRequest(session_id=owner_session,
                                         task_ids=["same-task"]),
            SimpleNamespace(headers={"X-Agent-Key": "wrong"}),
        )
    assert bad_key.value.status_code == 401


@pytest.mark.asyncio
async def test_repaired_socket_terminalizes_card_and_speaks_durable_answer(monkeypatch):
    from app.api import ws_realtime as rt

    H.fast_clocks(monkeypatch)
    agent_runs = []
    claims = []

    async def no_rerun(*args, **kwargs):
        agent_runs.append((args, kwargs))
        raise AssertionError("recovery must never re-dispatch the agent")

    async def vps(_user_id):
        return "https://tenant.example", "test-key"

    async def api(_url, _key, method, path, **kwargs):
        if path == "/api/v1/internal/voice-delegations/claim-speech":
            claims.append(kwargs["json_body"])
            return {"claimed": True}
        if path == "/api/v1/internal/voice-delegations/status":
            assert method == "POST"
            assert kwargs["json_body"]["task_ids"] == ["old-task"]
            return {"tasks": {"old-task": {
                "status": "completed", "job_id": "job-1",
                "answer": "The research found three options.", "spoken": False,
                "tool_events": [{
                    "tool": "web_search", "label": "Searching the web",
                    "started_at_ms": 100, "completed_at_ms": 200,
                    "job_id": "job-1", "step_index": 0,
                    "step_name": "Search", "steps_total": 2,
                    "sources": [{"title": "Result", "url": "https://example.com/a",
                                 "domain": "example.com"}],
                }],
            }}}
        return []  # optional session-history seed

    H.patch_relay(monkeypatch, think=no_rerun, vps=vps)
    monkeypatch.setattr(rt, "_vps_api", api)

    def on_send(provider, event):
        if event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([
        H.config(),
        {"type": "recover_delegations", "tasks": [{
            "task_id": "old-task", "parent_user_turn_id": "old-turn",
            "title": "Search the web", "task_revision": 5,
        }]},
        0.1,
        {"type": "recover_delegations", "tasks": [{
            "task_id": "old-task", "parent_user_turn_id": "old-turn",
            "title": "Search the web", "task_revision": 5,
        }]},
        0.25, {"type": "stop"},
    ])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=5)

    recovered = [f for f in client.of("delegation") if f.get("task_id") == "old-task"]
    assert len(recovered) == 1
    assert recovered[0]["phase"] == recovered[0]["state"] == "completed"
    assert recovered[0]["task_revision"] == 6
    assert recovered[0]["parent_user_turn_id"] == "old-turn"
    assert recovered[0]["job_id"] == "job-1"
    assert recovered[0]["result_text"] == "The research found three options."
    assert recovered[0]["spoken"] is False
    assert recovered[0]["speech_status"] == "unconfirmed"
    replay = [f for f in client.frames if f.get("recovery") is True]
    assert [f["type"] for f in replay] == ["tool_call.started", "tool_call.completed"]
    assert replay[1]["task_id"] == "old-task"
    assert replay[1]["parent_call_id"] == "live-delegation:old-task"
    assert replay[1]["step_index"] == 0
    assert replay[1]["sources"][0]["domain"] == "example.com"
    assert client.frames.index(replay[1]) < client.frames.index(recovered[0])
    assert "The research found three options." in " ".join(
        e["content"] for e in provider.of("session.commentary.append")
    )
    assert len(provider.of("session.commentary.append")) == 1
    assert claims == [{"session_id": "db-session", "task_id": "old-task"}]
    assert not agent_runs


@pytest.mark.asyncio
@pytest.mark.parametrize("spoken,claimed", [(True, False), (False, True)])
async def test_recovery_does_not_respeak_confirmed_or_claimed_result(
    monkeypatch, spoken, claimed,
):
    from app.api import ws_realtime as rt

    H.fast_clocks(monkeypatch)

    async def vps(_user_id):
        return "https://tenant.example", "test-key"

    async def api(_url, _key, _method, path, **_kwargs):
        if path == "/api/v1/internal/voice-delegations/status":
            return {"tasks": {"old-task": {
                "status": "completed", "answer": "Durable result.",
                "spoken": spoken, "recovery_speech_claimed": claimed,
            }}}
        return []

    H.patch_relay(monkeypatch, vps=vps)
    monkeypatch.setattr(rt, "_vps_api", api)

    def on_send(provider, event):
        if event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([
        H.config(),
        {"type": "recover_delegations", "tasks": [{
            "task_id": "old-task", "parent_user_turn_id": "old-turn",
            "task_revision": 2,
        }]},
        0.2, {"type": "stop"},
    ])
    provider = H.FakeProvider(on_send=on_send)
    await H.run_relay(client, provider, timeout=5)
    assert client.of("delegation")[-1]["task_revision"] == 3
    assert client.of("delegation")[-1]["speech_status"] == (
        "previously_attempted" if claimed else "unconfirmed"
    )
    assert provider.of("session.commentary.append") == []


def test_late_original_revision_preserves_recovery_speech_claim():
    from app.api.sessions import _merge_metadata

    original = json.dumps({"voice": {
        "source": "delegated", "delegation_id": "old-task",
        "spoken": False, "recovery_speech_claimed": True,
    }, "tool_events": [{"tool": "web_search"}]})
    late = json.dumps({"voice": {
        "source": "delegated", "delegation_id": "old-task",
        "spoken": False, "record_kind": "task_result",
    }})
    merged = json.loads(_merge_metadata(original, late))
    assert merged["voice"]["recovery_speech_claimed"] is True
    assert merged["voice"]["record_kind"] == "task_result"
    assert merged["tool_events"] == [{"tool": "web_search"}]
    changed_task = json.loads(_merge_metadata(original, json.dumps({"voice": {
        "source": "delegated", "delegation_id": "different-task",
    }})))
    assert "recovery_speech_claimed" not in changed_task["voice"]


@pytest.mark.asyncio
async def test_repaired_stop_requests_exact_running_job_and_waits_for_terminal(monkeypatch):
    from app.api import ws_realtime as rt
    import app.services.live_voice_protocol as live

    H.fast_clocks(monkeypatch)
    monkeypatch.setattr(live, "_DELEGATION_RECOVERY_POLL_S", 0.02)
    dispatched = []
    cancel_calls = []
    state = {"cancelled": False, "cancelled_reads": 0}

    async def no_dispatch(*args, **kwargs):
        dispatched.append((args, kwargs))
        raise AssertionError("a repaired Stop must not dispatch the agent")

    async def vps(_user_id):
        return "https://tenant.example", "agent-key"

    async def api(_url, _key, _method, path, **kwargs):
        if path.endswith("/status"):
            row = {
                "status": "cancelled" if state["cancelled"] else "running",
                "job_id": "exact-job", "cancel_requested": state["cancelled"],
            }
            if state["cancelled"]:
                state["cancelled_reads"] += 1
                if state["cancelled_reads"] >= 2:
                    row["record_found"] = True
                    row["tool_events"] = [{
                        "tool": "web_search", "job_id": "exact-job",
                        "sources": [{"url": "https://example.com/result",
                                     "domain": "example.com"}],
                    }]
            return {"tasks": {"old-task": row}}
        if path.endswith("/cancel"):
            cancel_calls.append(kwargs["json_body"])
            state["cancelled"] = True
            return {"requested": True, "status": "running", "job_id": "exact-job"}
        return []

    H.patch_relay(monkeypatch, think=no_dispatch, vps=vps)
    monkeypatch.setattr(rt, "_vps_api", api)

    def on_send(provider, event):
        if event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([])

    async def await_cancellable():
        for _ in range(100):
            if any(f.get("type") == "delegation_recovery"
                   and f.get("cancellable") is True for f in client.frames):
                return
            await asyncio.sleep(0.01)
        raise AssertionError("exact running job did not offer Stop")

    client._script = [
        H.config(),
        {"type": "recover_delegations", "tasks": [{
            "task_id": "old-task", "parent_user_turn_id": "old-turn",
            "task_revision": 3,
        }]},
        await_cancellable,
        {"type": "cancel_task", "task_id": "old-task"},
        0.15, {"type": "stop"},
    ]
    await H.run_relay(client, H.FakeProvider(on_send=on_send), timeout=5)

    assert cancel_calls == [{
        "session_id": "db-session", "task_id": "old-task", "job_id": "exact-job",
    }]
    recovery = client.of("delegation_recovery")
    assert recovery[0]["status"] == "checking" and recovery[0]["cancellable"] is True
    assert any(f["status"] == "cancel_requested" and f["cancellable"] is False
               for f in recovery)
    terminal = [f for f in client.of("delegation") if f.get("task_id") == "old-task"]
    assert len(terminal) == 1 and terminal[0]["phase"] == "cancelled"
    assert terminal[0]["task_revision"] == 4
    replay = [f for f in client.frames if f.get("recovery") is True]
    assert [f["type"] for f in replay] == ["tool_call.started", "tool_call.completed"]
    assert replay[1]["sources"][0]["domain"] == "example.com"
    assert client.frames.index(replay[1]) < client.frames.index(terminal[0])
    assert not dispatched


@pytest.mark.asyncio
async def test_rejected_recovered_stop_clears_pending_then_reports_status_outage(monkeypatch):
    from app.api import ws_realtime as rt
    import app.services.live_voice_protocol as live

    H.fast_clocks(monkeypatch)
    monkeypatch.setattr(live, "_DELEGATION_RECOVERY_POLL_S", 0.02)
    monkeypatch.setattr(live, "_DELEGATION_STATUS_OUTAGE_S", 0.05)
    state = {"outage": False, "status_reads_after_cancel": 0, "vps_reads": 0}

    async def vps(_user_id):
        state["vps_reads"] += 1
        return "https://tenant.example", "agent-key"

    async def api(_url, _key, _method, path, **_kwargs):
        if path.endswith("/cancel"):
            state["outage"] = True
            return None
        if path.endswith("/status"):
            if state["outage"]:
                state["status_reads_after_cancel"] += 1
                if state["status_reads_after_cancel"] <= 4:
                    return None
                return {"tasks": {"old-task": {
                    "status": "cancelled", "job_id": "exact-job",
                    "record_found": True, "cancel_requested": True,
                }}}
            return {"tasks": {"old-task": {
                "status": "running", "job_id": "exact-job",
                "cancel_requested": False,
            }}}
        return []

    H.patch_relay(monkeypatch, vps=vps)
    monkeypatch.setattr(rt, "_vps_api", api)

    def on_send(provider, event):
        if event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([])

    async def await_cancellable():
        for _ in range(100):
            if any(f.get("type") == "delegation_recovery"
                   and f.get("cancellable") is True for f in client.frames):
                return
            await asyncio.sleep(0.01)
        raise AssertionError("exact running job did not offer Stop")

    client._script = [
        H.config(),
        {"type": "recover_delegations", "tasks": [{
            "task_id": "old-task", "parent_user_turn_id": "old-turn",
        }]},
        await_cancellable,
        {"type": "cancel_task", "task_id": "old-task"},
        0.18, {"type": "stop"},
    ]
    await H.run_relay(client, H.FakeProvider(on_send=on_send), timeout=5)
    recovery = client.of("delegation_recovery")
    assert [(f["status"], f["cancellable"]) for f in recovery] == [
        ("checking", True), ("checking", False), ("unavailable", False),
    ]
    assert state["vps_reads"] >= 2  # transient tenant address is retried
    terminal = [f for f in client.of("delegation") if f.get("task_id") == "old-task"]
    assert len(terminal) == 1 and terminal[0]["phase"] == "cancelled"


@pytest.mark.asyncio
async def test_old_runner_observes_durable_stop_after_socket_detaches(monkeypatch):
    from app.api import ws_realtime as rt
    import app.services.live_voice_protocol as live

    H.fast_clocks(monkeypatch)
    monkeypatch.setattr(live, "_DURABLE_CANCEL_POLL_S", 0.02)
    # This agent never starts a tool, and the watch now waits for one (or
    # `_DURABLE_CANCEL_NO_TOOL_WAIT_S`) before it reads at all — pinned by
    # test_cancel_watch_waits_for_a_tool_and_backs_off. What this test is
    # about is the detached runner observing the Stop, so no wait here.
    monkeypatch.setattr(live, "_DURABLE_CANCEL_NO_TOOL_WAIT_S", 0.0)
    started = asyncio.Event()
    stopped = asyncio.Event()
    reads = {"n": 0}

    async def think(*_args, **_kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            stopped.set()
            raise

    async def vps(_user_id):
        return "https://tenant.example", "agent-key"

    async def api(_url, _key, _method, path, **_kwargs):
        if path.endswith("/status"):
            reads["n"] += 1
            # No job row exists at the beginning. The watch must neither
            # guess a target nor cancel before an exact durable marker exists.
            if reads["n"] < 3:
                return {"tasks": {"d1": {"status": "unknown"}}}
            return {"tasks": {"d1": {
                "status": "running", "job_id": "exact-job",
                "cancel_requested": True,
            }}}
        return []

    recorded = H.patch_relay(monkeypatch, think=think, vps=vps)
    monkeypatch.setattr(rt, "_vps_api", api)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("research this", 0, 400))
            provider.push(H.delegation("d1", 420))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([H.config(), 0.25, {"type": "stop"}])
    await H.run_relay(client, H.FakeProvider(on_send=on_send), timeout=5)
    assert started.is_set() and stopped.is_set()
    assert reads["n"] >= 3
    assert "completed" not in client.phases()
    assert "cancelled" in client.phases()
    assert not any(row.get("assistant_voice", {}).get("source") == "delegated"
                   and row.get("assistant_text") == "research this"
                   for row in recorded["saves"])


@pytest.mark.asyncio
async def test_tenant_stream_emits_cancelled_after_durable_stop(monkeypatch):
    from app.api import api_v1
    from app.config import settings
    from app.db import async_session_maker
    from app.db.models import BuildJob, Conversation, User

    if settings.run_mode != "agent":
        pytest.skip("agent-only tables require RUN_MODE=agent")
    owner, session_id, job_id = (str(uuid.uuid4()) for _ in range(3))
    async with async_session_maker() as db:
        db.add_all([
            User(id=owner, email=f"{owner[:8]}@test.local", hashed_password="x" * 60),
            Conversation(id=session_id, user_id=owner, channel="voice"),
            BuildJob(id=job_id, user_id=owner, conversation_id=session_id,
                     job_type="agent_task", status="running", title="Research",
                     prompt="Research", stop_requested_at=datetime.utcnow(),
                     config_json={"voice_delegation_id": "task-to-stop"}),
        ])
        await db.commit()

    monkeypatch.setattr(settings, "run_mode", "agent")
    monkeypatch.setattr(settings, "agent_api_key", "test-agent-key")
    monkeypatch.setattr(settings, "user_id", owner)
    monkeypatch.setattr(api_v1, "_VS_CANCEL_POLL_S", 0.02)
    monkeypatch.setattr(api_v1, "_VS_HEARTBEAT_S", 0.02)
    was_cancelled = asyncio.Event()

    class Runner:
        async def run(self, **kwargs):
            await kwargs["on_tool_event"]({
                "phase": "start", "name": "web_search", "call_id": "tool-one",
                "input": {"query": "sample"},
            })
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                was_cancelled.set()
                raise

    monkeypatch.setattr(api_v1, "_agent_runner", Runner())
    response = await api_v1.internal_agent_turn_stream(
        api_v1.ChatRequest(message="Research", session_id=session_id,
                           delegation_id="task-to-stop", save=False),
        SimpleNamespace(headers={"X-Agent-Key": "test-agent-key"}),
    )

    async def collect():
        frames = []
        async for chunk in response.body_iterator:
            if chunk.startswith("data:"):
                frames.append(json.loads(chunk[5:].strip()))
        return frames

    frames = await asyncio.wait_for(collect(), timeout=3)
    assert was_cancelled.is_set()
    assert [f["type"] for f in frames if f["type"] in {"tool.start", "cancelled", "done"}] == [
        "tool.start", "cancelled",
    ]


# ── PR-2 (2026-10-02 call, P2.6): the cancel watch stops hammering a stalled tenant ──


def _scripted_watch(monkeypatch, responses, *, tool_after_sleeps=None):
    """Drive `watch_durable_delegation_cancel` on a fake clock.

    `asyncio.sleep` and `time.monotonic` are replaced INSIDE the relay module
    only (everything else forwards to the real modules), so the waits the
    watch asks for are recorded exactly and cost no wall time.
    """

    from app.api import ws_realtime as rt
    import app.services.live_voice_protocol as live

    clock = {"now": 1000.0}
    sleeps: list[float] = []
    reads: list[float] = []
    real_sleep = asyncio.sleep
    task = live.DelegatedTask(
        delegation_id="d1", offset_ms=0, transcript="research this",
        turn_id="live-utt:x:1", created_ms=0,
    )
    entry = live.RunningDelegation(task=task, started_monotonic=clock["now"])

    class _Asyncio:
        def __getattr__(self, name):
            return getattr(asyncio, name)

        async def sleep(self, delay, *args, **kwargs):
            sleeps.append(round(float(delay), 3))
            clock["now"] += float(delay)
            if tool_after_sleeps is not None and len(sleeps) == tool_after_sleeps:
                entry.tool_started = True       # the agent's first tool round
            await real_sleep(0)

    class _Time:
        def __getattr__(self, name):
            return getattr(time, name)

        def monotonic(self):
            return clock["now"]

    async def vps(_user_id):
        return "https://tenant.example", "agent-key"

    async def api(_url, _key, _method, path, **_kwargs):
        assert path == "/api/v1/internal/voice-delegations/status"
        reads.append(clock["now"] - 1000.0)
        return responses[len(reads) - 1]

    monkeypatch.setattr(live, "asyncio", _Asyncio())
    monkeypatch.setattr(live, "time", _Time())
    monkeypatch.setattr(rt, "_get_vps_info", vps)
    monkeypatch.setattr(rt, "_vps_api", api)

    session = live._LiveSession.__new__(live._LiveSession)
    session.user_id = "user-1"
    session.db_session_id = "db-watch"
    return session, task, entry, sleeps, reads


async def _watch_until_cancelled(session, task, entry):
    runner = asyncio.ensure_future(asyncio.Event().wait())
    try:
        await asyncio.wait_for(
            session.watch_durable_delegation_cancel(task, entry, runner), timeout=5,
        )
    finally:
        runner.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await runner


_STOP = {"tasks": {"d1": {"status": "running", "job_id": "exact-job", "cancel_requested": True}}}
_NO_JOB = {"tasks": {"d1": {"status": "unknown"}}}


@pytest.mark.asyncio
async def test_cancel_watch_waits_for_a_tool_and_backs_off(monkeypatch):
    """The voice job — the only place a durable Stop can be marked — opens at
    the agent's first planned tool round, so a status read before it can never
    find one; during the 2026-10-02 stall every such read just timed out, 1.5 s
    apart. Reads now wait for the first tool, and each failed read doubles the
    next wait (1.5 → 3 → 6 → 10 s cap) until a good read resets it."""

    session, task, entry, sleeps, reads = _scripted_watch(
        monkeypatch, [None, None, None, _NO_JOB, _STOP], tool_after_sleeps=2,
    )
    await _watch_until_cancelled(session, task, entry)

    # Two ticks with no tool and no read; then read, fail ×3 (backing off to
    # the cap), a good read (reset), and the Stop.
    assert sleeps == [1.5, 1.5, 3.0, 6.0, 10.0, 1.5]
    assert reads == [3.0, 6.0, 12.0, 22.0, 23.5]
    assert entry.cancelled and entry.superseded_by == "explicit_cancel"


@pytest.mark.asyncio
async def test_cancel_watch_without_tool_frames_still_reads_after_the_grace(monkeypatch):
    """The blocking agent-turn path relays no tool frames at all, so the watch
    cannot wait for one forever: with none, reads begin after 10 s."""

    session, task, entry, sleeps, reads = _scripted_watch(monkeypatch, [_NO_JOB, _STOP])
    await _watch_until_cancelled(session, task, entry)

    assert reads and reads[0] >= 10.0, reads
    assert reads == [10.5, 12.0]
    assert sleeps == [1.5] * 8
    assert entry.cancelled


@pytest.mark.asyncio
async def test_a_task_that_ended_before_the_repair_keeps_its_retry(monkeypatch):
    """D-P12: `last_outcomes` is per SOCKET, so after any repair (either
    replica) a carried failed card's Retry answered `unknown_task`.  The app
    now carries the tasks that ENDED in the last 10 min on
    `recover_delegations`; the relay rebuilds their outcome, and the card's
    Retry re-runs the failed task instead of being ignored."""

    from app.services import live_voice_protocol as live

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    calls = []

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        calls.append((kw.get("delegation_id"), kw.get("display_request")))
        return "Professor A works on LLMs.", "m"

    H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        if event["type"] == "session.close":
            provider.push(H.closed())

    async def retried():
        for _ in range(300):
            if any(
                (f.get("relation") or {}).get("task_id") == "old-failed"
                and f.get("phase") == "completed"
                for f in client.of("delegation")
            ):
                return
            await asyncio.sleep(0.01)

    features = list(H.config()["features"]) + ["retry_relation"]
    client = H.FakeClient([
        H.config(features=features),
        {"type": "recover_delegations", "tasks": [{
            "task_id": "old-failed", "state": "failed",
            "title": "find the best professor working on LLMs",
            "turn_id": "old-turn", "request_turn_ids": ["old-turn"],
            "reason": "delegation_failed", "reply_language": "en",
        }]},
        0.1,
        {"type": "client_event", "name": "retry_task", "detail": "old-failed"},
        retried, 0.2, {"type": "stop"},
    ])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=6, db_session_id="db-recover-terminal")

    assert len(calls) == 1, calls
    assert calls[0][0].startswith(live.RELAY_RETRY_PREFIX), calls
    frames = [
        f for f in client.of("delegation")
        if (f.get("relation") or {}).get("task_id") == "old-failed"
    ]
    assert frames and frames[0]["relation"]["kind"] == "retries", client.frames
    assert {f["turn_id"] for f in frames} == {"old-turn"}
    assert not [f for f in client.of("delegation") if f.get("task_id") == "old-failed"], (
        "a terminal task is history to rebuild, never a card to recover"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("state, retried, outcome", [
    # The app's TaskState spelling — the relay's own `send_delegation` maps
    # cancelled/superseded to it — and the legacy phase spelling.
    ("canceled", False, "cancelled"),
    ("cancelled", False, "cancelled"),
    # The phase a delegation expires in is a failure to the user: Retry works.
    ("expired", True, "failed"),
    ("failed", True, "failed"),
])
async def test_a_recovered_terminal_task_is_read_in_the_wire_vocabulary(
        monkeypatch, caplog, state, retried, outcome):
    """W1 (relay half). The app builds each ended task from
    `liveProtocol.recoverDelegationsFrame` — EXACTLY `{task_id, state, title,
    turn_id, request_turn_ids?, result_text?, reason?, reply_language?}`, with
    `state` in its TaskState vocabulary (`completed`/`failed`/`canceled`).
    The relay keyed its rebuild on `cancelled`, so a canceled task was never
    rebuilt: its card's Retry and "what happened?" answered `unknown_task`."""

    import logging
    from app.services import live_voice_protocol as live

    caplog.set_level(logging.INFO, logger=live.logger.name)
    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    calls = []

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        calls.append(kw.get("delegation_id"))
        return "Professor A works on LLMs.", "m"

    H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        if event["type"] == "session.close":
            provider.push(H.closed())

    async def settled():
        for _ in range(300):
            if not retried and "retry_task ignored task=old-x" in caplog.text:
                return
            if retried and any(
                (f.get("relation") or {}).get("task_id") == "old-x"
                and f.get("phase") == "completed"
                for f in client.of("delegation")
            ):
                return
            await asyncio.sleep(0.01)

    features = list(H.config()["features"]) + ["retry_relation"]
    ended = {"task_id": "old-x", "state": state,
             "title": "find the best professor working on LLMs",
             "turn_id": "old-turn", "request_turn_ids": ["old-turn"],
             "reason": "delegation_failed", "reply_language": "en"}
    client = H.FakeClient([
        H.config(features=features),
        {"type": "recover_delegations", "tasks": [ended]},
        0.1,
        {"type": "client_event", "name": "retry_task", "detail": "old-x"},
        settled, 0.2, {"type": "stop"},
    ])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=6, db_session_id=f"db-recover-{state}")

    assert f"recovered outcome id=old-x outcome={outcome}" in caplog.text, caplog.text
    assert "reason=unknown_task" not in caplog.text
    assert len(calls) == (1 if retried else 0), calls
    if not retried:
        assert "retry_task ignored task=old-x reason=not_failed" in caplog.text
    assert not client.of("delegation_recovery"), "an ended task is never recovered"
