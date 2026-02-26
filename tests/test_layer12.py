"""
Layer 12 Tests — RPC Protocol, Activation Prompt, Token Tracker,
Channel Bindings, Sandbox Manager, TTS Auto-Mode.
"""

import asyncio
import json
import pytest


# ── RPC Protocol ──────────────────────────────────────

class TestRPCProtocol:
    def test_import(self):
        from app.agent.rpc_protocol import RPCServer, RPCRequest, RPCResponse
        assert RPCServer is not None

    def test_register_method(self):
        from app.agent.rpc_protocol import RPCServer
        server = RPCServer()

        async def echo(**kwargs):
            return kwargs

        server.register("echo", echo, description="Echo back params")
        methods = server.list_methods()
        assert len(methods) == 1
        assert methods[0]["name"] == "echo"

    def test_handle_simple_request(self):
        from app.agent.rpc_protocol import RPCServer
        server = RPCServer()

        async def add(a=0, b=0):
            return a + b

        server.register("add", add)
        loop = asyncio.get_event_loop()
        raw = json.dumps({"jsonrpc": "2.0", "method": "add", "params": {"a": 3, "b": 5}, "id": 1})
        response = loop.run_until_complete(server.handle_request(raw))
        data = json.loads(response)
        assert data["result"] == 8
        assert data["id"] == 1

    def test_handle_method_not_found(self):
        from app.agent.rpc_protocol import RPCServer
        server = RPCServer()
        loop = asyncio.get_event_loop()
        raw = json.dumps({"jsonrpc": "2.0", "method": "nonexistent", "id": 1})
        response = loop.run_until_complete(server.handle_request(raw))
        data = json.loads(response)
        assert data["error"]["code"] == -32601

    def test_handle_parse_error(self):
        from app.agent.rpc_protocol import RPCServer
        server = RPCServer()
        loop = asyncio.get_event_loop()
        response = loop.run_until_complete(server.handle_request("invalid json{{{"))
        data = json.loads(response)
        assert data["error"]["code"] == -32700

    def test_handle_invalid_request(self):
        from app.agent.rpc_protocol import RPCServer
        server = RPCServer()
        loop = asyncio.get_event_loop()
        raw = json.dumps({"method": "test", "id": 1})  # Missing jsonrpc
        response = loop.run_until_complete(server.handle_request(raw))
        data = json.loads(response)
        assert data["error"]["code"] == -32600

    def test_notification(self):
        from app.agent.rpc_protocol import RPCServer
        server = RPCServer()
        called = []

        async def notify(**kwargs):
            called.append(kwargs)

        server.register("notify", notify)
        loop = asyncio.get_event_loop()
        raw = json.dumps({"jsonrpc": "2.0", "method": "notify", "params": {"msg": "hi"}})
        response = loop.run_until_complete(server.handle_request(raw))
        assert response is None  # No response for notifications
        assert len(called) == 1

    def test_batch_request(self):
        from app.agent.rpc_protocol import RPCServer
        server = RPCServer()

        async def double(x=0):
            return x * 2

        server.register("double", double)
        loop = asyncio.get_event_loop()
        batch = json.dumps([
            {"jsonrpc": "2.0", "method": "double", "params": {"x": 3}, "id": 1},
            {"jsonrpc": "2.0", "method": "double", "params": {"x": 5}, "id": 2},
        ])
        response = loop.run_until_complete(server.handle_request(batch))
        data = json.loads(response)
        assert len(data) == 2
        assert data[0]["result"] == 6
        assert data[1]["result"] == 10

    def test_internal_error(self):
        from app.agent.rpc_protocol import RPCServer
        server = RPCServer()

        async def fail():
            raise RuntimeError("boom")

        server.register("fail", fail)
        loop = asyncio.get_event_loop()
        raw = json.dumps({"jsonrpc": "2.0", "method": "fail", "id": 1})
        response = loop.run_until_complete(server.handle_request(raw))
        data = json.loads(response)
        assert data["error"]["code"] == -32603

    def test_unregister(self):
        from app.agent.rpc_protocol import RPCServer
        server = RPCServer()

        async def noop():
            pass

        server.register("test", noop)
        assert server.unregister("test") == True
        assert server.unregister("test") == False

    def test_stats(self):
        from app.agent.rpc_protocol import RPCServer
        server = RPCServer()

        async def noop():
            return "ok"

        server.register("test", noop)
        loop = asyncio.get_event_loop()
        raw = json.dumps({"jsonrpc": "2.0", "method": "test", "id": 1})
        loop.run_until_complete(server.handle_request(raw))
        s = server.stats()
        assert s["total_requests"] == 1

    def test_request_to_dict(self):
        from app.agent.rpc_protocol import RPCRequest
        req = RPCRequest(method="test", params={"a": 1}, id=42)
        d = req.to_dict()
        assert d["method"] == "test"
        assert d["id"] == 42

    def test_response_success(self):
        from app.agent.rpc_protocol import RPCResponse
        resp = RPCResponse.success(1, {"data": "ok"})
        d = resp.to_dict()
        assert d["result"]["data"] == "ok"

    def test_singleton(self):
        from app.agent.rpc_protocol import get_rpc_server
        s1 = get_rpc_server()
        s2 = get_rpc_server()
        assert s1 is s2


# ── Activation Prompt ─────────────────────────────────

class TestActivation:
    def test_import(self):
        from app.agent.activation import ActivationManager, get_activation_manager
        assert ActivationManager is not None

    def test_set_get_prompt(self):
        from app.agent.activation import ActivationManager
        mgr = ActivationManager()
        mgr.set_prompt("agent-1", "Hello agent!")
        assert mgr.get_prompt("agent-1") == "Hello agent!"

    def test_get_prompt_not_set(self):
        from app.agent.activation import ActivationManager
        mgr = ActivationManager()
        assert mgr.get_prompt("unknown") is None

    def test_global_prompt_fallback(self):
        from app.agent.activation import ActivationManager
        mgr = ActivationManager()
        mgr.set_global_prompt("Global init message")
        assert mgr.get_prompt("unknown") == "Global init message"

    def test_agent_overrides_global(self):
        from app.agent.activation import ActivationManager
        mgr = ActivationManager()
        mgr.set_global_prompt("Global")
        mgr.set_prompt("agent-1", "Agent-specific")
        assert mgr.get_prompt("agent-1") == "Agent-specific"

    def test_remove_prompt(self):
        from app.agent.activation import ActivationManager
        mgr = ActivationManager()
        mgr.set_prompt("a1", "Hi")
        assert mgr.remove_prompt("a1") == True
        assert mgr.get_prompt("a1") is None

    def test_trigger_session_start(self):
        from app.agent.activation import ActivationManager
        mgr = ActivationManager()
        mgr.set_prompt("a1", "Boot message")
        result = mgr.trigger("a1", event="session_start")
        assert result == "Boot message"

    def test_trigger_increments_count(self):
        from app.agent.activation import ActivationManager
        mgr = ActivationManager()
        mgr.set_prompt("a1", "Boot")
        mgr.trigger("a1")
        mgr.trigger("a1")
        config = mgr.get_config("a1")
        assert config.trigger_count == 2

    def test_trigger_disabled(self):
        from app.agent.activation import ActivationManager
        mgr = ActivationManager()
        mgr.set_prompt("a1", "Boot")
        mgr.disable("a1")
        assert mgr.trigger("a1") is None

    def test_enable_disable(self):
        from app.agent.activation import ActivationManager
        mgr = ActivationManager()
        mgr.set_prompt("a1", "Boot")
        mgr.disable("a1")
        assert mgr.get_prompt("a1") is None
        mgr.enable("a1")
        assert mgr.get_prompt("a1") == "Boot"

    def test_list_configs(self):
        from app.agent.activation import ActivationManager
        mgr = ActivationManager()
        mgr.set_prompt("a1", "Boot1")
        mgr.set_prompt("a2", "Boot2")
        configs = mgr.list_configs()
        assert len(configs) == 2

    def test_history(self):
        from app.agent.activation import ActivationManager
        mgr = ActivationManager()
        mgr.set_prompt("a1", "Boot")
        mgr.trigger("a1")
        history = mgr.get_history()
        assert len(history) == 1

    def test_stats(self):
        from app.agent.activation import ActivationManager
        mgr = ActivationManager()
        mgr.set_prompt("a1", "Boot")
        s = mgr.stats()
        assert s["total_configs"] == 1
        assert s["enabled"] == 1

    def test_singleton(self):
        from app.agent.activation import get_activation_manager
        m1 = get_activation_manager()
        m2 = get_activation_manager()
        assert m1 is m2


# ── Token Tracker ─────────────────────────────────────

class TestTokenTracker:
    def test_import(self):
        from app.agent.token_tracker import TokenTracker, get_token_tracker
        assert TokenTracker is not None

    def test_record_usage(self):
        from app.agent.token_tracker import TokenTracker
        t = TokenTracker()
        record = t.record_usage("s1", "gpt-4o", 1000, 500)
        assert record.total_tokens == 1500
        assert record.cost_usd > 0

    def test_session_usage(self):
        from app.agent.token_tracker import TokenTracker
        t = TokenTracker()
        t.record_usage("s1", "gpt-4o", 1000, 500)
        t.record_usage("s1", "gpt-4o", 2000, 1000)
        usage = t.get_session_usage("s1")
        assert usage["total_input_tokens"] == 3000
        assert usage["total_output_tokens"] == 1500
        assert usage["total_requests"] == 2

    def test_session_not_found(self):
        from app.agent.token_tracker import TokenTracker
        t = TokenTracker()
        assert t.get_session_usage("nonexistent") is None

    def test_total_usage(self):
        from app.agent.token_tracker import TokenTracker
        t = TokenTracker()
        t.record_usage("s1", "gpt-4o", 1000, 500)
        t.record_usage("s2", "gpt-4o-mini", 2000, 1000)
        total = t.get_total_usage()
        assert total["total_sessions"] == 2
        assert total["total_tokens"] == 4500

    def test_usage_by_model(self):
        from app.agent.token_tracker import TokenTracker
        t = TokenTracker()
        t.record_usage("s1", "gpt-4o", 1000, 500)
        t.record_usage("s1", "gpt-4o-mini", 2000, 1000)
        by_model = t.get_usage_by_model()
        assert "gpt-4o" in by_model
        assert "gpt-4o-mini" in by_model

    def test_cost_calculation(self):
        from app.agent.token_tracker import TokenTracker
        t = TokenTracker()
        # gpt-4o: $2.50/1M input, $10.00/1M output
        record = t.record_usage("s1", "gpt-4o", 1_000_000, 1_000_000)
        assert abs(record.cost_usd - 12.50) < 0.01

    def test_format_status(self):
        from app.agent.token_tracker import TokenTracker
        t = TokenTracker()
        t.record_usage("s1", "gpt-4o", 1000, 500, tool_calls=3)
        status = t.get_session_status("s1")
        assert "Session Usage" in status
        assert "Tool calls: 3" in status

    def test_format_usage_report(self):
        from app.agent.token_tracker import TokenTracker
        t = TokenTracker()
        t.record_usage("s1", "gpt-4o", 1000, 500)
        report = t.format_usage_report()
        assert "Usage Report" in report

    def test_clear_session(self):
        from app.agent.token_tracker import TokenTracker
        t = TokenTracker()
        t.record_usage("s1", "gpt-4o", 1000, 500)
        assert t.clear_session("s1") == True
        assert t.get_session_usage("s1") is None

    def test_recent_records(self):
        from app.agent.token_tracker import TokenTracker
        t = TokenTracker()
        t.record_usage("s1", "gpt-4o", 1000, 500)
        records = t.get_recent_records()
        assert len(records) == 1

    def test_models_used_tracking(self):
        from app.agent.token_tracker import TokenTracker
        t = TokenTracker()
        t.record_usage("s1", "gpt-4o", 1000, 500)
        t.record_usage("s1", "gpt-4o-mini", 500, 200)
        usage = t.get_session_usage("s1")
        assert "gpt-4o" in usage["models_used"]
        assert "gpt-4o-mini" in usage["models_used"]

    def test_stats(self):
        from app.agent.token_tracker import TokenTracker
        t = TokenTracker()
        t.record_usage("s1", "gpt-4o", 1000, 500)
        s = t.stats()
        assert s["total_records"] == 1

    def test_singleton(self):
        from app.agent.token_tracker import get_token_tracker
        t1 = get_token_tracker()
        t2 = get_token_tracker()
        assert t1 is t2


# ── Channel Bindings ──────────────────────────────────

class TestChannelBindings:
    def test_import(self):
        from app.agent.channel_bindings import ChannelBindingManager, get_binding_manager
        assert ChannelBindingManager is not None

    def test_bind_and_resolve(self):
        from app.agent.channel_bindings import ChannelBindingManager
        mgr = ChannelBindingManager()
        mgr.bind("telegram", "-100123", "agent-coder")
        assert mgr.resolve("telegram", "-100123") == "agent-coder"

    def test_resolve_default(self):
        from app.agent.channel_bindings import ChannelBindingManager
        mgr = ChannelBindingManager(default_agent="main")
        assert mgr.resolve("telegram", "-999") == "main"

    def test_unbind(self):
        from app.agent.channel_bindings import ChannelBindingManager
        mgr = ChannelBindingManager()
        mgr.bind("telegram", "-100123", "agent-1")
        assert mgr.unbind("telegram", "-100123") == True
        assert mgr.resolve("telegram", "-100123") == "default"

    def test_wildcard_binding(self):
        from app.agent.channel_bindings import ChannelBindingManager
        mgr = ChannelBindingManager()
        mgr.bind("discord", "*", "discord-agent")
        assert mgr.resolve("discord", "any-guild-id") == "discord-agent"

    def test_specific_overrides_wildcard(self):
        from app.agent.channel_bindings import ChannelBindingManager
        mgr = ChannelBindingManager()
        mgr.bind("discord", "*", "general-agent")
        mgr.bind("discord", "guild_123", "special-agent")
        assert mgr.resolve("discord", "guild_123") == "special-agent"
        assert mgr.resolve("discord", "guild_other") == "general-agent"

    def test_list_bindings(self):
        from app.agent.channel_bindings import ChannelBindingManager
        mgr = ChannelBindingManager()
        mgr.bind("telegram", "-1", "a1")
        mgr.bind("discord", "g1", "a2")
        all_bindings = mgr.list_bindings()
        assert len(all_bindings) == 2

        tg = mgr.list_bindings(channel_type="telegram")
        assert len(tg) == 1

    def test_list_agents(self):
        from app.agent.channel_bindings import ChannelBindingManager
        mgr = ChannelBindingManager()
        mgr.bind("telegram", "-1", "agent-1")
        mgr.bind("discord", "g1", "agent-2")
        agents = mgr.list_agents()
        assert "agent-1" in agents
        assert "agent-2" in agents
        assert "default" in agents

    def test_get_groups_for_agent(self):
        from app.agent.channel_bindings import ChannelBindingManager
        mgr = ChannelBindingManager()
        mgr.bind("telegram", "-1", "agent-1")
        mgr.bind("telegram", "-2", "agent-1")
        mgr.bind("discord", "g1", "agent-2")
        groups = mgr.get_groups_for_agent("agent-1")
        assert len(groups) == 2

    def test_disable_enable(self):
        from app.agent.channel_bindings import ChannelBindingManager
        mgr = ChannelBindingManager()
        mgr.bind("telegram", "-1", "agent-1")
        mgr.disable_binding("telegram", "-1")
        assert mgr.resolve("telegram", "-1") == "default"  # Falls to default
        mgr.enable_binding("telegram", "-1")
        assert mgr.resolve("telegram", "-1") == "agent-1"

    def test_bulk_bind(self):
        from app.agent.channel_bindings import ChannelBindingManager
        mgr = ChannelBindingManager()
        count = mgr.bulk_bind([
            {"channel_type": "telegram", "group_id": "-1", "agent_id": "a1"},
            {"channel_type": "telegram", "group_id": "-2", "agent_id": "a2"},
        ])
        assert count == 2

    def test_stats(self):
        from app.agent.channel_bindings import ChannelBindingManager
        mgr = ChannelBindingManager()
        mgr.bind("telegram", "-1", "a1")
        s = mgr.stats()
        assert s["total_bindings"] == 1
        assert s["enabled"] == 1

    def test_singleton(self):
        from app.agent.channel_bindings import get_binding_manager
        m1 = get_binding_manager()
        m2 = get_binding_manager()
        assert m1 is m2


# ── Sandbox Manager ───────────────────────────────────

class TestSandboxManager:
    def test_import(self):
        from app.agent.sandbox_manager import SandboxManager, get_sandbox_manager
        assert SandboxManager is not None

    def test_create_sandbox(self):
        from app.agent.sandbox_manager import SandboxManager, SandboxState
        mgr = SandboxManager()
        sb = mgr.create_sandbox("s1")
        assert sb.state == SandboxState.RUNNING
        assert sb.session_id == "s1"

    def test_get_sandbox(self):
        from app.agent.sandbox_manager import SandboxManager
        mgr = SandboxManager()
        sb = mgr.create_sandbox("s1")
        assert mgr.get_sandbox(sb.sandbox_id) is sb

    def test_get_sandbox_for_session(self):
        from app.agent.sandbox_manager import SandboxManager
        mgr = SandboxManager()
        sb = mgr.create_sandbox("s1")
        assert mgr.get_sandbox_for_session("s1") is sb

    def test_exec_in_sandbox(self):
        from app.agent.sandbox_manager import SandboxManager
        mgr = SandboxManager()
        sb = mgr.create_sandbox("s1")
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(
            mgr.exec_in_sandbox(sb.sandbox_id, "echo hello")
        )
        assert result.exit_code == 0
        assert sb.exec_count == 1

    def test_exec_not_found(self):
        from app.agent.sandbox_manager import SandboxManager
        mgr = SandboxManager()
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(
            mgr.exec_in_sandbox("nonexistent", "echo hi")
        )
        assert result.exit_code == -1

    def test_pause_resume(self):
        from app.agent.sandbox_manager import SandboxManager, SandboxState
        mgr = SandboxManager()
        sb = mgr.create_sandbox("s1")
        assert mgr.pause_sandbox(sb.sandbox_id) == True
        assert sb.state == SandboxState.PAUSED
        assert mgr.resume_sandbox(sb.sandbox_id) == True
        assert sb.state == SandboxState.RUNNING

    def test_stop_sandbox(self):
        from app.agent.sandbox_manager import SandboxManager, SandboxState
        mgr = SandboxManager()
        sb = mgr.create_sandbox("s1")
        assert mgr.stop_sandbox(sb.sandbox_id) == True
        assert sb.state == SandboxState.STOPPED

    def test_destroy_sandbox(self):
        from app.agent.sandbox_manager import SandboxManager
        mgr = SandboxManager()
        sb = mgr.create_sandbox("s1")
        assert mgr.destroy_sandbox(sb.sandbox_id) == True
        assert mgr.get_sandbox(sb.sandbox_id) is None

    def test_idempotent_create(self):
        from app.agent.sandbox_manager import SandboxManager
        mgr = SandboxManager()
        sb1 = mgr.create_sandbox("s1")
        sb2 = mgr.create_sandbox("s1")  # Same session
        assert sb1 is sb2

    def test_list_sandboxes(self):
        from app.agent.sandbox_manager import SandboxManager
        mgr = SandboxManager()
        mgr.create_sandbox("s1")
        mgr.create_sandbox("s2")
        assert len(mgr.list_sandboxes()) == 2

    def test_resource_limits(self):
        from app.agent.sandbox_manager import SandboxManager, ResourceLimits
        mgr = SandboxManager()
        limits = ResourceLimits(cpu_cores=2.0, memory_mb=1024)
        sb = mgr.create_sandbox("s1", limits=limits)
        assert sb.limits.cpu_cores == 2.0
        assert sb.limits.memory_mb == 1024

    def test_stats(self):
        from app.agent.sandbox_manager import SandboxManager
        mgr = SandboxManager()
        mgr.create_sandbox("s1")
        s = mgr.stats()
        assert s["total_sandboxes"] == 1

    def test_singleton(self):
        from app.agent.sandbox_manager import get_sandbox_manager
        m1 = get_sandbox_manager()
        m2 = get_sandbox_manager()
        assert m1 is m2


# ── TTS Auto-Mode ─────────────────────────────────────

class TestTTSAutoMode:
    def test_import(self):
        from app.agent.tts_auto_mode import TTSModeManager, TTSAutoMode, get_tts_mode_manager
        assert TTSModeManager is not None

    def test_default_off(self):
        from app.agent.tts_auto_mode import TTSModeManager, TTSAutoMode
        mgr = TTSModeManager()
        assert mgr.get_mode("any") == TTSAutoMode.OFF

    def test_set_mode(self):
        from app.agent.tts_auto_mode import TTSModeManager, TTSAutoMode
        mgr = TTSModeManager()
        mgr.set_mode("s1", TTSAutoMode.ALWAYS)
        assert mgr.get_mode("s1") == TTSAutoMode.ALWAYS

    def test_should_tts_off(self):
        from app.agent.tts_auto_mode import TTSModeManager, TTSAutoMode
        mgr = TTSModeManager()
        mgr.set_mode("s1", TTSAutoMode.OFF)
        assert mgr.should_tts("s1") == False

    def test_should_tts_always(self):
        from app.agent.tts_auto_mode import TTSModeManager, TTSAutoMode
        mgr = TTSModeManager()
        mgr.set_mode("s1", TTSAutoMode.ALWAYS)
        assert mgr.should_tts("s1") == True

    def test_should_tts_inbound_voice(self):
        from app.agent.tts_auto_mode import TTSModeManager, TTSAutoMode
        mgr = TTSModeManager()
        mgr.set_mode("s1", TTSAutoMode.INBOUND)
        assert mgr.should_tts("s1", inbound_is_voice=True) == True
        assert mgr.should_tts("s1", inbound_is_voice=False) == False

    def test_should_tts_tagged(self):
        from app.agent.tts_auto_mode import TTSModeManager, TTSAutoMode
        mgr = TTSModeManager()
        mgr.set_mode("s1", TTSAutoMode.TAGGED)
        assert mgr.should_tts("s1", has_voice_tag=True) == True
        assert mgr.should_tts("s1", has_voice_tag=False) == False

    def test_channel_mode(self):
        from app.agent.tts_auto_mode import TTSModeManager, TTSAutoMode
        mgr = TTSModeManager()
        mgr.set_channel_mode("telegram", TTSAutoMode.INBOUND)
        assert mgr.should_tts("unknown", inbound_is_voice=True, channel_type="telegram") == True

    def test_cycle_mode(self):
        from app.agent.tts_auto_mode import TTSModeManager, TTSAutoMode
        mgr = TTSModeManager()
        mgr.set_mode("s1", TTSAutoMode.OFF)
        next_mode = mgr.cycle_mode("s1")
        assert next_mode == TTSAutoMode.INBOUND
        next_mode = mgr.cycle_mode("s1")
        assert next_mode == TTSAutoMode.ALWAYS

    def test_remove_config(self):
        from app.agent.tts_auto_mode import TTSModeManager, TTSAutoMode
        mgr = TTSModeManager()
        mgr.set_mode("s1", TTSAutoMode.ALWAYS)
        assert mgr.remove_config("s1") == True
        assert mgr.get_mode("s1") == TTSAutoMode.OFF

    def test_tts_count_tracking(self):
        from app.agent.tts_auto_mode import TTSModeManager, TTSAutoMode
        mgr = TTSModeManager()
        mgr.set_mode("s1", TTSAutoMode.ALWAYS)
        mgr.should_tts("s1")
        mgr.should_tts("s1")
        config = mgr.get_config("s1")
        assert config.tts_count == 2

    def test_list_configs(self):
        from app.agent.tts_auto_mode import TTSModeManager, TTSAutoMode
        mgr = TTSModeManager()
        mgr.set_mode("s1", TTSAutoMode.ALWAYS)
        mgr.set_mode("s2", TTSAutoMode.INBOUND)
        configs = mgr.list_configs()
        assert len(configs) == 2

    def test_stats(self):
        from app.agent.tts_auto_mode import TTSModeManager, TTSAutoMode
        mgr = TTSModeManager()
        mgr.set_mode("s1", TTSAutoMode.ALWAYS)
        s = mgr.stats()
        assert s["total_configs"] == 1

    def test_singleton(self):
        from app.agent.tts_auto_mode import get_tts_mode_manager
        m1 = get_tts_mode_manager()
        m2 = get_tts_mode_manager()
        assert m1 is m2
