"""Real overlapping AgentRunner calls keep managed-voice transient state isolated."""
from __future__ import annotations

import asyncio
import json
import uuid

import pytest


class _ConcurrentLLM:
    def __init__(self) -> None:
        self.rounds = {"alpha": 0, "beta": 0}
        self.entered: set[str] = set()
        self.both_entered = asyncio.Event()

    async def create_message_stream(self, **kwargs):
        from app.services.openai_agent_service import StreamEvent

        wire = json.dumps(kwargs["messages"], default=str).lower()
        tag = "alpha" if "alpha" in wire else "beta"
        self.rounds[tag] += 1
        if self.rounds[tag] == 1:
            self.entered.add(tag)
            if self.entered == {"alpha", "beta"}:
                self.both_entered.set()
            await asyncio.wait_for(self.both_entered.wait(), timeout=1)
            yield StreamEvent(
                type="tool_use_start", tool_name="web_search", tool_id=f"search-{tag}"
            )
            yield StreamEvent(
                type="tool_use_end",
                tool_name="web_search",
                tool_id=f"search-{tag}",
                tool_input={"query": tag},
            )
            yield StreamEvent(
                type="message_end",
                stop_reason="tool_use",
                usage={"input_tokens": 5, "output_tokens": 2},
            )
            return
        yield StreamEvent(type="text", text=f"done {tag}")
        yield StreamEvent(
            type="message_end",
            stop_reason="end_turn",
            usage={"input_tokens": 5, "output_tokens": 2},
        )


class _ResolvedOperationLLM:
    def __init__(self) -> None:
        self.round = 0
        self.second_messages = []

    async def create_message_stream(self, **kwargs):
        from app.services.openai_agent_service import StreamEvent

        self.round += 1
        if self.round == 1:
            calls = [
                ("mail-a", "gmail_send_email", {"to": "a@example.test"}),
                ("mail-b", "gmail_send_email", {"to": "b@example.test"}),
                ("read-a", "web_search", {"query": "old evidence"}),
                ("read-b", "web_search", {"query": "new evidence"}),
            ]
            for call_id, name, payload in calls:
                yield StreamEvent(
                    type="tool_use_start", tool_name=name, tool_id=call_id,
                )
                yield StreamEvent(
                    type="tool_use_end", tool_name=name, tool_id=call_id,
                    tool_input=payload,
                )
            yield StreamEvent(
                type="message_end", stop_reason="tool_use",
                usage={"input_tokens": 5, "output_tokens": 2},
            )
            return
        self.second_messages = kwargs["messages"]
        yield StreamEvent(type="text", text="continuation complete")
        yield StreamEvent(
            type="message_end", stop_reason="end_turn",
            usage={"input_tokens": 5, "output_tokens": 2},
        )


class _ScriptedOperationLLM:
    def __init__(self, rounds) -> None:
        self.rounds = list(rounds)
        self.calls = 0
        self.messages = []

    async def create_message_stream(self, **kwargs):
        from app.services.openai_agent_service import StreamEvent

        self.messages.append(kwargs["messages"])
        if self.calls < len(self.rounds):
            calls = self.rounds[self.calls]
            self.calls += 1
            for call_id, name, payload in calls:
                yield StreamEvent(
                    type="tool_use_start", tool_name=name, tool_id=call_id,
                )
                yield StreamEvent(
                    type="tool_use_end", tool_name=name, tool_id=call_id,
                    tool_input=payload,
                )
            yield StreamEvent(
                type="message_end", stop_reason="tool_use",
                usage={"input_tokens": 5, "output_tokens": 2},
            )
            return
        yield StreamEvent(type="text", text="script complete")
        yield StreamEvent(
            type="message_end", stop_reason="end_turn",
            usage={"input_tokens": 5, "output_tokens": 2},
        )


async def _make_operation_runner(monkeypatch, tmp_path, llm, execute=None):
    import app.agent.agent_runner as ar
    from app.agent.tool_executor import ToolExecutor

    async def no_history(self, db, session_id, max_messages=50, client_tz=None):
        return []

    async def fixed_prompt(self, *args, **kwargs):
        return "You are Toup."

    monkeypatch.setattr(ar.AgentRunner, "_load_history", no_history)
    monkeypatch.setattr(ar.AgentRunner, "_build_system_prompt", fixed_prompt)
    monkeypatch.setattr(ar, "_spawn_background", lambda coro: coro.close())
    monkeypatch.setattr(ar, "_spawn_bg", lambda coro, **kwargs: coro.close())
    monkeypatch.setattr(ar.settings, "citation_gate_enabled", False, raising=False)
    tools = ToolExecutor(workspace=str(tmp_path))
    if execute is not None:
        monkeypatch.setattr(tools, "execute", execute)
    return ar.AgentRunner(llm_service=llm, tool_executor=tools)


async def _seed_users() -> dict[str, str]:
    from app.db import User, async_session_maker

    users = {tag: str(uuid.uuid4()) for tag in ("alpha", "beta")}
    async with async_session_maker() as db:
        for tag, user_id in users.items():
            db.add(User(
                id=user_id,
                email=f"voice-overlap-{tag}-{user_id[:8]}@example.test",
                hashed_password="x" * 60,
                name=tag.title(),
            ))
        await db.commit()
    return users


async def test_real_overlapping_managed_voice_runs_isolate_executor_state(
    monkeypatch, tmp_path,
):
    import app.agent.agent_runner as ar
    from app.agent.prompt_profile import PromptProfile
    from app.agent.tool_executor import ToolExecutor

    async def no_history(self, db, session_id, max_messages=50, client_tz=None):
        return []

    async def fixed_prompt(self, *args, **kwargs):
        return "You are Toup."

    monkeypatch.setattr(ar.AgentRunner, "_load_history", no_history)
    monkeypatch.setattr(ar.AgentRunner, "_build_system_prompt", fixed_prompt)
    monkeypatch.setattr(ar, "_spawn_background", lambda coro: coro.close())
    monkeypatch.setattr(ar, "_spawn_bg", lambda coro, **kwargs: coro.close())
    monkeypatch.setattr(ar.settings, "citation_gate_enabled", False, raising=False)

    llm = _ConcurrentLLM()
    tools = ToolExecutor(workspace=str(tmp_path))
    users = await _seed_users()
    seen_context: dict[str, dict] = {}

    async def fake_execute(name, payload):
        tag = payload["query"]
        seen_context[tag] = {
            "user_id": tools._current_user_id,
            "job_id": tools._current_job_id,
            "channel": tools._current_channel,
        }
        tools.pending_attachments.append({
            "id": f"file-{tag}",
            "filename": f"{tag}.txt",
            "mime_type": "text/plain",
            "size_bytes": len(tag),
        })
        tools.google_docs_created_this_run.add(f"doc-{tag}")
        card = {
            "action_id": f"action-{tag}",
            "tool_name": "gmail_send_email",
            "status": "pending",
        }
        tools.staged_pending_action_ids.append(card["action_id"])
        tools.staged_pending_actions.append(card)
        tools._last_pending_action = card
        tools._last_media = {"type": "youtube", "video_id": f"video-{tag}"}
        if tools._on_tool_progress:
            await tools._on_tool_progress(name, tag)
        await asyncio.sleep(0)
        return f"result for {tag} https://{tag}.example.test/source"

    monkeypatch.setattr(tools, "execute", fake_execute)
    runner = ar.AgentRunner(llm_service=llm, tool_executor=tools)
    attachments = {"alpha": [], "beta": []}
    progress = {"alpha": [], "beta": []}

    async def run_one(tag: str):
        async def on_attachment(_message_id, item):
            attachments[tag].append(dict(item))

        async def on_progress(name, chunk):
            progress[tag].append((name, chunk))

        return await runner.run(
            user_message=f"research {tag}",
            user_id=users[tag],
            session_id=str(uuid.uuid4()),
            channel="voice",
            prompt_profile=PromptProfile.FULL,
            model_override="gpt-5.5-mini",
            current_job_id=f"job-{tag}",
            managed_voice_task=True,
            save_user_message=False,
            save_assistant_message=False,
            disable_post_processing=True,
            on_attachment=on_attachment,
            on_tool_progress=on_progress,
        )

    alpha, beta = await asyncio.gather(run_one("alpha"), run_one("beta"))

    for tag, response in (("alpha", alpha), ("beta", beta)):
        assert response.text == f"done {tag}"
        assert seen_context[tag] == {
            "user_id": users[tag], "job_id": f"job-{tag}", "channel": "voice",
        }
        assert attachments[tag] == [{
            "id": f"file-{tag}",
            "filename": f"{tag}.txt",
            "mime_type": "text/plain",
            "size_bytes": len(tag),
        }]
        assert progress[tag] == [("web_search", tag)]
        assert response.metadata["pending_action_ids"] == [f"action-{tag}"]
        assert response.metadata["pending_actions"] == [{
            "action_id": f"action-{tag}",
            "tool_name": "gmail_send_email",
            "status": "pending",
        }]
        assert response.metadata["pending_action"]["action_id"] == f"action-{tag}"
        assert response.metadata["media"]["video_id"] == f"video-{tag}"


async def test_resolved_operation_identity_reuses_exact_calls_and_allows_distinct_ones(
    monkeypatch, tmp_path,
):
    import app.agent.agent_runner as ar
    from app.agent.operation_identity import tool_operation_key
    from app.agent.prompt_profile import PromptProfile
    from app.agent.tool_executor import ToolExecutor

    async def no_history(self, db, session_id, max_messages=50, client_tz=None):
        return []

    async def fixed_prompt(self, *args, **kwargs):
        return "You are Toup."

    monkeypatch.setattr(ar.AgentRunner, "_load_history", no_history)
    monkeypatch.setattr(ar.AgentRunner, "_build_system_prompt", fixed_prompt)
    monkeypatch.setattr(ar, "_spawn_background", lambda coro: coro.close())
    monkeypatch.setattr(ar, "_spawn_bg", lambda coro, **kwargs: coro.close())
    monkeypatch.setattr(ar.settings, "citation_gate_enabled", False, raising=False)

    llm = _ResolvedOperationLLM()
    tools = ToolExecutor(workspace=str(tmp_path))
    executed = []

    async def fake_execute(name, payload):
        executed.append((name, dict(payload)))
        return f"fresh result for {name}: {json.dumps(payload, sort_keys=True)}"

    monkeypatch.setattr(tools, "execute", fake_execute)
    runner = ar.AgentRunner(llm_service=llm, tool_executor=tools)
    user_id = (await _seed_users())["alpha"]
    resolved = [
        {
            "operation_key": tool_operation_key(
                "gmail_send_email", {"to": "a@example.test"},
            ),
            "tool_name": "gmail_send_email", "status": "executed", "ok": True,
            "result": '{"id":"sent-a"}',
        },
        {
            "operation_key": tool_operation_key(
                "web_search", {"query": "old evidence"},
            ),
            "tool_name": "web_search", "status": "executed", "ok": True,
            "result": "cached old evidence",
        },
    ]

    response = await runner.run(
        user_message="Continue from the checkpoint",
        user_id=user_id,
        session_id=str(uuid.uuid4()),
        channel="voice",
        prompt_profile=PromptProfile.FULL,
        model_override="gpt-5.5-mini",
        current_job_id="voice-continuation",
        managed_voice_task=True,
        managed_resolved_operations=resolved,
        save_user_message=False,
        save_assistant_message=False,
        disable_post_processing=True,
    )

    assert response.text == "continuation complete"
    assert executed == [
        ("gmail_send_email", {"to": "b@example.test"}),
        ("web_search", {"query": "new evidence"}),
    ]
    messages = json.dumps(llm.second_messages, default=str)
    assert "sent-a" in messages
    assert "cached old evidence" in messages


async def test_edited_action_alias_reuses_actual_result_and_allows_distinct_send(
    monkeypatch, tmp_path,
):
    from app.agent.operation_identity import (
        tool_operation_checkpoint,
        tool_operation_key,
    )
    from app.agent.prompt_profile import PromptProfile

    draft = {
        "to": "draft@example.test", "subject": "Draft", "body": "Draft body",
    }
    approved = {
        "to": "approved@example.test", "subject": "Approved",
        "body": "Approved body",
    }
    distinct = {
        "to": "second@example.test", "subject": "Second", "body": "Second body",
    }
    llm = _ScriptedOperationLLM([[
        ("draft-call", "gmail_send_email", draft),
        ("approved-call", "gmail_send_email", approved),
        ("distinct-call", "gmail_send_email", distinct),
    ]])
    executed = []

    async def execute(name, payload):
        executed.append((name, dict(payload)))
        return "fresh distinct send"

    runner = await _make_operation_runner(monkeypatch, tmp_path, llm, execute)
    user_id = (await _seed_users())["alpha"]
    approved_key = tool_operation_key("gmail_send_email", approved)
    draft_key = tool_operation_key("gmail_send_email", draft)
    resolved = [{
        "operation_key": approved_key,
        "operation_aliases": [draft_key],
        "staged_operation_key": draft_key,
        "operation_id": "stable-action-id",
        "tool_name": "gmail_send_email", "status": "executed", "ok": True,
        "result": '{"kind":"ok","id":"sent-approved"}',
        **tool_operation_checkpoint("gmail_send_email", approved),
    }]

    response = await runner.run(
        user_message="Continue after approval",
        user_id=user_id, session_id=str(uuid.uuid4()), channel="voice",
        prompt_profile=PromptProfile.FULL, model_override="gpt-5.5-mini",
        current_job_id="edited-action", managed_voice_task=True,
        managed_resolved_operations=resolved, save_user_message=False,
        save_assistant_message=False, disable_post_processing=True,
    )

    assert response.text == "script complete"
    assert executed == [("gmail_send_email", distinct)]
    model_messages = json.dumps(llm.messages[-1], default=str)
    assert model_messages.count("sent-approved") == 2
    assert "user's approved edits" in model_messages


async def test_write_then_read_refreshes_affected_file_and_reuses_unrelated_read(
    monkeypatch, tmp_path,
):
    from app.agent.operation_identity import (
        tool_operation_checkpoint,
        tool_operation_key,
    )
    from app.agent.prompt_profile import PromptProfile

    write_input = {"path": "report.txt", "content": "new value"}
    report_read = {"path": "report.txt"}
    notes_read = {"path": "notes.txt"}
    llm = _ScriptedOperationLLM([
        [("write", "write_file", write_input)],
        [
            ("verify-report", "read_file", report_read),
            ("verify-notes", "read_file", notes_read),
        ],
    ])
    executed = []
    files = {"report.txt": "old value", "notes.txt": "stable notes"}

    async def execute(name, payload):
        executed.append((name, dict(payload)))
        if name == "write_file":
            files[payload["path"]] = payload["content"]
            return "Write succeeded"
        return files[payload["path"]]

    runner = await _make_operation_runner(monkeypatch, tmp_path, llm, execute)
    user_id = (await _seed_users())["alpha"]

    def read_checkpoint(payload, result):
        return {
            "operation_key": tool_operation_key("read_file", payload),
            "tool_name": "read_file", "status": "executed", "ok": True,
            "result": result,
            **tool_operation_checkpoint("read_file", payload),
        }

    response = await runner.run(
        user_message="Update and verify the report",
        user_id=user_id, session_id=str(uuid.uuid4()), channel="voice",
        prompt_profile=PromptProfile.FULL, model_override="gpt-5.5-mini",
        current_job_id="write-read", managed_voice_task=True,
        managed_resolved_operations=[
            read_checkpoint(report_read, "old value"),
            read_checkpoint(notes_read, "stable notes"),
        ],
        save_user_message=False, save_assistant_message=False,
        disable_post_processing=True,
    )

    assert response.text == "script complete"
    assert executed == [
        ("write_file", write_input),
        ("read_file", report_read),
    ]
    model_messages = json.dumps(llm.messages[-1], default=str)
    assert "new value" in model_messages
    assert "Prior result: stable notes" in model_messages
    assert "Prior result: old value" not in model_messages


@pytest.mark.parametrize(
    "direction", ["relative-read", "relative-write", "document-rewrite"],
)
async def test_native_file_alias_write_invalidates_read_and_preserves_unrelated(
    monkeypatch, tmp_path, direction,
):
    from app.agent.operation_identity import (
        tool_operation_checkpoint,
        tool_operation_key,
    )
    from app.agent.prompt_profile import PromptProfile

    target = (
        tmp_path / "generated" / "report.pdf"
        if direction == "document-rewrite"
        else tmp_path / "state.json"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("old value", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("stable notes", encoding="utf-8")
    (tmp_path / "nested").mkdir()
    if direction == "relative-read":
        state_read = {"path": "./state.json"}
        state_write = {"path": str(target), "content": "new value"}
    elif direction == "relative-write":
        state_read = {"path": str(target)}
        state_write = {"path": "nested/../state.json", "content": "new value"}
    else:
        state_read = {"path": "generated/report.pdf"}
        state_write = {"path": "report.pdf", "content": "new value"}
    notes_read = {"path": "notes.txt"}
    llm = _ScriptedOperationLLM([
        [("write-state", "write_file", state_write)],
        [
            ("verify-state", "read_file", state_read),
            ("verify-notes", "read_file", notes_read),
        ],
    ])
    runner = await _make_operation_runner(monkeypatch, tmp_path, llm)
    native_execute = runner.tools.execute
    executed = []

    async def tracked_native_execute(name, payload):
        executed.append((name, dict(payload)))
        return await native_execute(name, payload)

    monkeypatch.setattr(runner.tools, "execute", tracked_native_execute)
    user_id = (await _seed_users())["alpha"]

    def old_lexical_read(payload, result):
        # Deliberately model a checkpoint written before native path identity
        # was available. The runner must migrate its resource key at compare
        # time rather than reuse stale evidence.
        return {
            "operation_key": tool_operation_key("read_file", payload),
            "tool_name": "read_file", "status": "executed", "ok": True,
            "result": result,
            **tool_operation_checkpoint("read_file", payload),
        }

    response = await runner.run(
        user_message="Update and verify the state file",
        user_id=user_id, session_id=str(uuid.uuid4()), channel="voice",
        prompt_profile=PromptProfile.FULL, model_override="gpt-5.5-mini",
        current_job_id=f"native-alias-{direction}", managed_voice_task=True,
        managed_resolved_operations=[
            old_lexical_read(state_read, "old value"),
            old_lexical_read(notes_read, "stable notes"),
        ],
        save_user_message=False, save_assistant_message=False,
        disable_post_processing=True,
    )

    assert response.text == "script complete"
    assert executed == [
        ("write_file", state_write),
        ("read_file", state_read),
    ]
    assert target.read_text(encoding="utf-8") == "new value"
    model_messages = json.dumps(llm.messages[-1], default=str)
    assert "new value" in model_messages
    assert "Prior result: stable notes" in model_messages
    assert "Prior result: old value" not in model_messages


async def test_parallel_duplicate_safe_read_executes_once_per_canonical_input(
    monkeypatch, tmp_path,
):
    from app.agent.prompt_profile import PromptProfile

    duplicate = {"query": "latest AI news", "count": 5}
    distinct = {"count": 5, "query": "latest robotics news"}
    llm = _ScriptedOperationLLM([[
        ("search-a", "web_search", duplicate),
        ("search-a-copy", "web_search", {"count": 5, "query": "latest AI news"}),
        ("search-b", "web_search", distinct),
    ]])
    executed = []
    events = []

    async def execute(name, payload):
        executed.append((name, dict(payload)))
        await asyncio.sleep(0.01)
        return f"evidence for {payload['query']}"

    async def on_tool_event(event):
        events.append(dict(event))

    runner = await _make_operation_runner(monkeypatch, tmp_path, llm, execute)
    user_id = (await _seed_users())["alpha"]
    response = await runner.run(
        user_message="Search the latest AI and robotics news",
        user_id=user_id, session_id=str(uuid.uuid4()), channel="mobile",
        prompt_profile=PromptProfile.FULL, model_override="gpt-5.5-mini",
        save_user_message=False, save_assistant_message=False,
        disable_post_processing=True, on_tool_event=on_tool_event,
    )

    assert response.text == "script complete"
    assert executed == [
        ("web_search", duplicate),
        ("web_search", distinct),
    ]
    ended = [event for event in events if event.get("phase") == "end"]
    assert [event["call_id"] for event in ended] == [
        "search-a", "search-a-copy", "search-b",
    ]
    assert ended[1]["coalesced_operation_id"] == "search-a"
    model_messages = json.dumps(llm.messages[-1], default=str)
    assert model_messages.count("evidence for latest AI news") == 2
    assert "evidence for latest robotics news" in model_messages


async def test_live_status_read_is_polled_again_in_a_later_model_round(
    monkeypatch, tmp_path,
):
    """Read-only status is an observation, not a reusable result."""
    from app.agent.prompt_profile import PromptProfile

    payload = {"job_id": "build-live"}
    llm = _ScriptedOperationLLM([
        [("status-running", "app_builder__get_status", payload)],
        [("status-completed", "app_builder__get_status", payload)],
    ])
    executed = []

    async def execute(name, tool_input):
        executed.append((name, dict(tool_input)))
        return "Status: running" if len(executed) == 1 else "Status: completed"

    runner = await _make_operation_runner(monkeypatch, tmp_path, llm, execute)
    user_id = (await _seed_users())["alpha"]
    response = await runner.run(
        user_message="Build the app and keep checking until it completes",
        user_id=user_id, session_id=str(uuid.uuid4()), channel="mobile",
        prompt_profile=PromptProfile.FULL, model_override="gpt-5.5-mini",
        save_user_message=False, save_assistant_message=False,
        disable_post_processing=True,
    )

    assert response.text == "script complete"
    assert executed == [
        ("app_builder__get_status", payload),
        ("app_builder__get_status", payload),
    ]
    model_messages = json.dumps(llm.messages[-1], default=str)
    assert "Status: running" in model_messages
    assert "Status: completed" in model_messages


async def test_managed_native_live_reads_poll_again_in_later_model_round(
    monkeypatch, tmp_path,
):
    """Native status/list/output tools are observations, not mutations."""
    from app.agent.operation_identity import (
        tool_operation_cacheable_read,
        tool_operation_kind,
    )
    from app.agent.prompt_profile import PromptProfile

    polls = [
        ("process", {"action": "status", "process_id": "controlled"}),
        ("process", {"action": "output", "process_id": "controlled"}),
        ("process", {"action": "list"}),
        ("sessions_list", {}),
        ("session_status", {}),
        ("lanes_status", {}),
    ]
    for name, payload in polls:
        assert tool_operation_kind(name, payload) == "read"
        assert not tool_operation_cacheable_read(name, payload)
    assert tool_operation_kind("process", {"action": "start"}) == "mutation"
    assert tool_operation_kind("process", {"action": "stop"}) == "mutation"
    for name, action in [
        ("config_reload", "list"),
        ("config_reload", "get"),
        ("thread", "list"),
        ("tts_prefs", "get"),
        ("skill_marketplace", "search"),
        ("skill_marketplace", "list_installed"),
        ("talk_mode", "status"),
    ]:
        assert tool_operation_kind(name, {"action": action}) == "read"
        assert not tool_operation_cacheable_read(name, {"action": action})
    for name, action in [
        ("config_reload", "set"),
        ("thread", "create"),
        ("tts_prefs", "set"),
        ("skill_marketplace", "install"),
        ("talk_mode", "start"),
    ]:
        assert tool_operation_kind(name, {"action": action}) == "mutation"

    llm = _ScriptedOperationLLM([
        [(f"round-1-{index}", name, payload) for index, (name, payload) in enumerate(polls)],
        [(f"round-2-{index}", name, payload) for index, (name, payload) in enumerate(polls)],
    ])
    counts = {}
    executed = []

    async def execute(name, tool_input):
        key = (name, json.dumps(tool_input, sort_keys=True))
        counts[key] = counts.get(key, 0) + 1
        executed.append((name, dict(tool_input)))
        return f"{name}/{tool_input.get('action', 'read')} observation {counts[key]}"

    runner = await _make_operation_runner(monkeypatch, tmp_path, llm, execute)
    user_id = (await _seed_users())["alpha"]
    response = await runner.run(
        user_message="Keep polling the native work until its state changes",
        user_id=user_id, session_id=str(uuid.uuid4()), channel="voice",
        prompt_profile=PromptProfile.FULL, model_override="gpt-5.5-mini",
        current_job_id="native-poll", managed_voice_task=True,
        save_user_message=False, save_assistant_message=False,
        disable_post_processing=True,
    )

    assert response.text == "script complete"
    assert executed == polls + polls
    assert set(counts.values()) == {2}
    model_messages = json.dumps(llm.messages[-1], default=str)
    for name, payload in polls:
        label = f"{name}/{payload.get('action', 'read')}"
        assert f"{label} observation 1" in model_messages
        assert f"{label} observation 2" in model_messages


async def test_managed_continuation_ignores_old_misclassified_poll_records(
    monkeypatch, tmp_path,
):
    """Old mutation labels cannot freeze polls; real mutations stay guarded."""
    from app.agent.operation_identity import tool_operation_key
    from app.agent.prompt_profile import PromptProfile

    status_input = {"action": "status", "process_id": "controlled"}
    sessions_input = {}
    start_input = {"action": "start", "command": "controlled-server"}
    stop_input = {"action": "stop", "process_id": "controlled"}
    calls = [
        ("status", "process", status_input),
        ("sessions", "sessions_list", sessions_input),
        ("start", "process", start_input),
        ("stop", "process", stop_input),
    ]
    llm = _ScriptedOperationLLM([calls])
    executed = []

    async def execute(name, tool_input):
        executed.append((name, dict(tool_input)))
        return f"fresh {name}/{tool_input.get('action', 'read')} observation"

    runner = await _make_operation_runner(monkeypatch, tmp_path, llm, execute)
    user_id = (await _seed_users())["alpha"]

    def old_record(name, payload, result):
        return {
            "operation_key": tool_operation_key(name, payload),
            "tool_name": name, "status": "executed", "ok": True,
            "operation_kind": "mutation", "result": result,
        }

    response = await runner.run(
        user_message="Continue the managed native task",
        user_id=user_id, session_id=str(uuid.uuid4()), channel="voice",
        prompt_profile=PromptProfile.FULL, model_override="gpt-5.5-mini",
        current_job_id="native-continuation", managed_voice_task=True,
        managed_resolved_operations=[
            old_record("process", status_input, "stale process running"),
            old_record("sessions_list", sessions_input, "stale session list"),
            old_record("process", start_input, "guarded process start"),
            old_record("process", stop_input, "guarded process stop"),
        ],
        save_user_message=False, save_assistant_message=False,
        disable_post_processing=True,
    )

    assert response.text == "script complete"
    assert executed == [
        ("process", status_input),
        ("sessions_list", sessions_input),
    ]
    model_messages = json.dumps(llm.messages[-1], default=str)
    assert "fresh process/status observation" in model_messages
    assert "fresh sessions_list/read observation" in model_messages
    assert "stale process running" not in model_messages
    assert "stale session list" not in model_messages
    assert "guarded process start" in model_messages
    assert "guarded process stop" in model_messages


async def test_explicit_fresh_read_bypasses_parallel_coalescing_and_run_cache(
    monkeypatch, tmp_path,
):
    from app.agent.operation_identity import tool_operation_cacheable_read
    from app.agent.prompt_profile import PromptProfile

    payload = {"query": "latest AI news", "fresh": True}
    assert not tool_operation_cacheable_read("web_search", payload)
    assert not tool_operation_cacheable_read(
        "web_search", {"query": "latest AI news", "cache": False},
    )
    assert tool_operation_cacheable_read(
        "web_search", {"query": "latest AI news"},
    )
    llm = _ScriptedOperationLLM([[
        ("fresh-a", "web_search", payload),
        ("fresh-b", "web_search", dict(payload)),
    ]])
    executed = []
    events = []

    async def execute(name, tool_input):
        executed.append((name, dict(tool_input)))
        await asyncio.sleep(0)
        return f"fresh observation {len(executed)}"

    async def on_tool_event(event):
        events.append(dict(event))

    runner = await _make_operation_runner(monkeypatch, tmp_path, llm, execute)
    user_id = (await _seed_users())["alpha"]
    response = await runner.run(
        user_message="Fetch two independent fresh observations",
        user_id=user_id, session_id=str(uuid.uuid4()), channel="mobile",
        prompt_profile=PromptProfile.FULL, model_override="gpt-5.5-mini",
        save_user_message=False, save_assistant_message=False,
        disable_post_processing=True, on_tool_event=on_tool_event,
    )

    assert response.text == "script complete"
    assert len(executed) == 2
    assert all(item == ("web_search", payload) for item in executed)
    ended = [event for event in events if event.get("phase") == "end"]
    assert [event["call_id"] for event in ended] == ["fresh-a", "fresh-b"]
    assert all("coalesced_operation_id" not in event for event in ended)


async def test_mobile_chat_reuses_exact_read_until_same_domain_mutation(
    monkeypatch, tmp_path,
):
    from app.agent.prompt_profile import PromptProfile

    list_input = {"calendar": "today"}
    create_input = {"title": "New item"}
    llm = _ScriptedOperationLLM([[
        ("list-before", "routines__list", list_input),
        ("list-duplicate", "routines__list", {"calendar": "today"}),
        ("create", "routines__create", create_input),
        ("list-after", "routines__list", list_input),
    ]])
    executed = []
    list_count = 0

    async def execute(name, payload):
        nonlocal list_count
        executed.append((name, dict(payload)))
        if name == "routines__list":
            list_count += 1
            return f"list version {list_count}"
        return "created"

    runner = await _make_operation_runner(monkeypatch, tmp_path, llm, execute)
    user_id = (await _seed_users())["alpha"]
    response = await runner.run(
        user_message="List today's routines, add an item, and verify",
        user_id=user_id, session_id=str(uuid.uuid4()), channel="mobile",
        prompt_profile=PromptProfile.FULL, model_override="gpt-5.5-mini",
        save_user_message=False, save_assistant_message=False,
        disable_post_processing=True,
    )

    assert response.text == "script complete"
    assert executed == [
        ("routines__list", list_input),
        ("routines__create", create_input),
        ("routines__list", list_input),
    ]
    model_messages = json.dumps(llm.messages[-1], default=str)
    assert model_messages.count("list version 1") == 2
    assert model_messages.count("list version 2") == 1


async def test_unchanged_read_checkpoint_is_reused_by_actual_runner(
    monkeypatch, tmp_path,
):
    from app.agent.operation_identity import (
        tool_operation_checkpoint,
        tool_operation_key,
    )
    from app.agent.prompt_profile import PromptProfile

    payload = {"path": "report.txt"}
    llm = _ScriptedOperationLLM([[("read", "read_file", payload)]])
    executed = []

    async def execute(name, tool_input):
        executed.append((name, dict(tool_input)))
        return "unexpected fresh read"

    runner = await _make_operation_runner(monkeypatch, tmp_path, llm, execute)
    user_id = (await _seed_users())["alpha"]
    resolved = [{
        "operation_key": tool_operation_key("read_file", payload),
        "tool_name": "read_file", "status": "executed", "ok": True,
        "result": "unchanged value",
        **tool_operation_checkpoint("read_file", payload),
    }]
    response = await runner.run(
        user_message="Use the existing report evidence",
        user_id=user_id, session_id=str(uuid.uuid4()), channel="voice",
        prompt_profile=PromptProfile.FULL, model_override="gpt-5.5-mini",
        current_job_id="unchanged-read", managed_voice_task=True,
        managed_resolved_operations=resolved, save_user_message=False,
        save_assistant_message=False, disable_post_processing=True,
    )

    assert response.text == "script complete"
    assert executed == []
    assert "Prior result: unchanged value" in json.dumps(
        llm.messages[-1], default=str,
    )
