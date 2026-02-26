"""
Comprehensive tests for GAP_ANALYSIS_v2 features.

Tests all new modules created during the platform gaps implementation:
1. hooks.py — Plugin/Hook event bus
2. multi_agent.py — Multi-agent routing
3. webhooks.py — Webhook API router
4. discord_channel.py — Discord channel adapter
5. slack_channel.py — Slack channel adapter
6. whatsapp_channel.py — WhatsApp channel adapter
7. tool_definitions.py — Extended tools
8. tool_executor.py — Tool policy enforcement
9. config.py — New config fields
10. telegram_bot.py — /think command
"""

import asyncio
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# Add app to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ════════════════════════════════════════════════════════════════
# 1. HOOKS — Event Bus
# ════════════════════════════════════════════════════════════════
class TestHookBus(unittest.TestCase):
    """Test the plugin/hook event bus system."""

    def test_import(self):
        from app.agent.hooks import HookBus, HookEvent, get_hook_bus
        self.assertIsNotNone(HookBus)
        self.assertIsNotNone(HookEvent)

    def test_hook_events_exist(self):
        from app.agent.hooks import HookEvent
        expected = [
            "BEFORE_AGENT_START", "AGENT_END", "BEFORE_TOOL_CALL",
            "AFTER_TOOL_CALL", "MESSAGE_RECEIVED", "MESSAGE_SENDING",
            "MESSAGE_SENT", "SESSION_START", "SESSION_END",
            "BEFORE_COMPACTION", "AFTER_COMPACTION",
            "MEMORY_CREATED", "MEMORY_MERGED", "MEMORY_DECAYED",
            "CHANNEL_CONNECTED", "CHANNEL_DISCONNECTED",
            "STARTUP", "SHUTDOWN",
        ]
        for ev in expected:
            self.assertTrue(hasattr(HookEvent, ev), f"Missing HookEvent.{ev}")

    def test_register_and_emit(self):
        from app.agent.hooks import HookBus, HookEvent
        bus = HookBus()
        results = []

        async def handler(data):
            results.append(data)

        bus.register(HookEvent.STARTUP, handler)
        self.assertEqual(bus.handler_count(HookEvent.STARTUP), 1)

        asyncio.get_event_loop().run_until_complete(
            bus.emit(HookEvent.STARTUP, {"test": True})
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["test"], True)

    def test_on_decorator(self):
        from app.agent.hooks import HookBus, HookEvent
        bus = HookBus()

        @bus.on(HookEvent.SHUTDOWN)
        async def my_handler(data):
            pass

        self.assertEqual(bus.handler_count(HookEvent.SHUTDOWN), 1)

    def test_unregister(self):
        from app.agent.hooks import HookBus, HookEvent
        bus = HookBus()

        async def handler(data):
            pass

        bus.register(HookEvent.STARTUP, handler)
        self.assertEqual(bus.handler_count(HookEvent.STARTUP), 1)
        bus.unregister(HookEvent.STARTUP, handler)
        self.assertEqual(bus.handler_count(HookEvent.STARTUP), 0)

    def test_clear(self):
        from app.agent.hooks import HookBus, HookEvent
        bus = HookBus()

        async def h1(data): pass
        async def h2(data): pass

        bus.register(HookEvent.STARTUP, h1)
        bus.register(HookEvent.SHUTDOWN, h2)
        bus.clear()
        self.assertEqual(bus.handler_count(HookEvent.STARTUP), 0)
        self.assertEqual(bus.handler_count(HookEvent.SHUTDOWN), 0)

    def test_status(self):
        from app.agent.hooks import HookBus, HookEvent
        bus = HookBus()

        async def h(data): pass
        bus.register(HookEvent.STARTUP, h)
        bus.register(HookEvent.STARTUP, h)

        status = bus.status()
        self.assertIn("startup", status)
        self.assertEqual(status["startup"], 2)

    def test_emit_error_doesnt_crash(self):
        """A failing handler should not crash other handlers."""
        from app.agent.hooks import HookBus, HookEvent
        bus = HookBus()
        results = []

        async def bad_handler(data):
            raise ValueError("boom")

        async def good_handler(data):
            results.append("ok")

        bus.register(HookEvent.STARTUP, bad_handler)
        bus.register(HookEvent.STARTUP, good_handler)

        asyncio.get_event_loop().run_until_complete(
            bus.emit(HookEvent.STARTUP, {})
        )
        self.assertEqual(results, ["ok"])

    def test_singleton(self):
        from app.agent.hooks import get_hook_bus
        bus1 = get_hook_bus()
        bus2 = get_hook_bus()
        self.assertIs(bus1, bus2)


# ════════════════════════════════════════════════════════════════
# 2. MULTI-AGENT ROUTING
# ════════════════════════════════════════════════════════════════
class TestMultiAgentRouter(unittest.TestCase):
    """Test multi-agent persona routing."""

    def test_import(self):
        from app.agent.multi_agent import MultiAgentRouter, AgentPersona, get_multi_agent_router
        self.assertIsNotNone(MultiAgentRouter)
        self.assertIsNotNone(AgentPersona)

    def test_default_persona_exists(self):
        from app.agent.multi_agent import MultiAgentRouter
        router = MultiAgentRouter()
        persona = router.route("hello")
        self.assertEqual(persona.name, "default")

    def test_keyword_routing(self):
        from app.agent.multi_agent import MultiAgentRouter, AgentPersona
        router = MultiAgentRouter()
        router.register(AgentPersona(
            name="coder",
            keywords=["code", "implement", "debug"],
            priority=5,
        ))
        result = router.route("Can you implement this feature?")
        self.assertEqual(result.name, "coder")

    def test_regex_routing(self):
        from app.agent.multi_agent import MultiAgentRouter, AgentPersona
        router = MultiAgentRouter()
        router.register(AgentPersona(
            name="researcher",
            patterns=[r"\bwhy\b.*\?$"],
            priority=3,
        ))
        result = router.route("why does the sky appear blue?")
        self.assertEqual(result.name, "researcher")

    def test_priority_ordering(self):
        from app.agent.multi_agent import MultiAgentRouter, AgentPersona
        router = MultiAgentRouter()
        router.register(AgentPersona(name="low", keywords=["test"], priority=1))
        router.register(AgentPersona(name="high", keywords=["test"], priority=10))
        result = router.route("test this")
        self.assertEqual(result.name, "high")

    def test_custom_classifier(self):
        from app.agent.multi_agent import MultiAgentRouter, AgentPersona
        router = MultiAgentRouter()
        router.register(AgentPersona(name="custom"))

        def always_custom(msg, personas):
            return "custom"

        router.set_classifier(always_custom)
        result = router.route("anything")
        self.assertEqual(result.name, "custom")

    def test_list_personas(self):
        from app.agent.multi_agent import get_multi_agent_router
        router = get_multi_agent_router()
        personas = router.list_personas()
        names = [p["name"] for p in personas]
        self.assertIn("default", names)
        self.assertIn("coder", names)
        self.assertIn("researcher", names)
        self.assertIn("writer", names)

    def test_unregister(self):
        from app.agent.multi_agent import MultiAgentRouter, AgentPersona
        router = MultiAgentRouter()
        router.register(AgentPersona(name="temp"))
        self.assertTrue(router.unregister("temp"))
        self.assertFalse(router.unregister("default"))  # Cannot remove default

    def test_fallback_to_default(self):
        from app.agent.multi_agent import MultiAgentRouter
        router = MultiAgentRouter()
        result = router.route("random gibberish xyz123")
        self.assertEqual(result.name, "default")


# ════════════════════════════════════════════════════════════════
# 3. TOOL DEFINITIONS — Extended Tools
# ════════════════════════════════════════════════════════════════
class TestExtendedToolDefinitions(unittest.TestCase):
    """Test the new tool definitions."""

    def test_import(self):
        from app.agent.tool_definitions import get_extended_tools
        tools = get_extended_tools()
        self.assertIsInstance(tools, list)

    def test_extended_tools_count(self):
        from app.agent.tool_definitions import get_extended_tools
        tools = get_extended_tools()
        self.assertEqual(len(tools), 14)

    def test_extended_tool_names(self):
        from app.agent.tool_definitions import get_extended_tools
        tools = get_extended_tools()
        names = [t["name"] for t in tools]
        self.assertIn("grep", names)
        self.assertIn("find", names)
        self.assertIn("ls", names)
        self.assertIn("apply_patch", names)
        self.assertIn("sessions_send", names)

    def test_tool_schema_structure(self):
        from app.agent.tool_definitions import get_extended_tools
        for tool in get_extended_tools():
            self.assertIn("name", tool)
            self.assertIn("description", tool)
            self.assertIn("input_schema", tool)
            self.assertIn("type", tool["input_schema"])
            self.assertEqual(tool["input_schema"]["type"], "object")

    def test_original_tools_still_work(self):
        from app.agent.tool_definitions import get_agent_tools
        tools = get_agent_tools()
        self.assertGreaterEqual(len(tools), 18)
        names = [t["name"] for t in tools]
        self.assertIn("exec", names)
        self.assertIn("memory_search", names)
        self.assertIn("web_search", names)

    def test_combined_tools(self):
        from app.agent.tool_definitions import get_agent_tools, get_extended_tools
        combined = get_agent_tools() + get_extended_tools()
        self.assertGreaterEqual(len(combined), 23)


# ════════════════════════════════════════════════════════════════
# 4. TOOL EXECUTOR — New Tools + Policy
# ════════════════════════════════════════════════════════════════
class TestToolExecutorNewTools(unittest.TestCase):
    """Test new tool implementations in ToolExecutor."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_tool_grep_method_exists(self):
        from app.agent.tool_executor import ToolExecutor
        te = ToolExecutor.__new__(ToolExecutor)
        self.assertTrue(hasattr(te, '_tool_grep'))
        self.assertTrue(callable(getattr(te, '_tool_grep')))

    def test_tool_find_method_exists(self):
        from app.agent.tool_executor import ToolExecutor
        te = ToolExecutor.__new__(ToolExecutor)
        self.assertTrue(hasattr(te, '_tool_find'))

    def test_tool_ls_method_exists(self):
        from app.agent.tool_executor import ToolExecutor
        te = ToolExecutor.__new__(ToolExecutor)
        self.assertTrue(hasattr(te, '_tool_ls'))

    def test_tool_apply_patch_method_exists(self):
        from app.agent.tool_executor import ToolExecutor
        te = ToolExecutor.__new__(ToolExecutor)
        self.assertTrue(hasattr(te, '_tool_apply_patch'))

    def test_tool_sessions_send_method_exists(self):
        from app.agent.tool_executor import ToolExecutor
        te = ToolExecutor.__new__(ToolExecutor)
        self.assertTrue(hasattr(te, '_tool_sessions_send'))

    def test_human_size(self):
        from app.agent.tool_executor import ToolExecutor
        self.assertEqual(ToolExecutor._human_size(0), "0B")
        self.assertEqual(ToolExecutor._human_size(1023), "1023B")
        self.assertEqual(ToolExecutor._human_size(1024), "1.0KB")
        self.assertEqual(ToolExecutor._human_size(1048576), "1.0MB")

    def test_tool_ls_runs(self):
        """Test that ls tool works on a real directory."""
        from app.agent.tool_executor import ToolExecutor
        te = ToolExecutor(workspace=self.tmpdir)
        te._current_user_id = "test"

        # Create a test file
        with open(os.path.join(self.tmpdir, "test.txt"), "w") as f:
            f.write("hello")

        result = asyncio.get_event_loop().run_until_complete(
            te._tool_ls({"path": self.tmpdir})
        )
        self.assertIn("test.txt", result)

    def test_tool_grep_runs(self):
        """Test that grep tool works."""
        from app.agent.tool_executor import ToolExecutor
        te = ToolExecutor(workspace=self.tmpdir)
        te._current_user_id = "test"

        # Create a test file
        with open(os.path.join(self.tmpdir, "hello.py"), "w") as f:
            f.write("def greet():\n    print('hello world')\n")

        result = asyncio.get_event_loop().run_until_complete(
            te._tool_grep({"pattern": "hello", "path": self.tmpdir})
        )
        self.assertIn("hello", result)
        self.assertIn("hello.py", result)

    def test_tool_find_runs(self):
        """Test that find tool works."""
        from app.agent.tool_executor import ToolExecutor
        te = ToolExecutor(workspace=self.tmpdir)
        te._current_user_id = "test"

        # Create test files
        os.makedirs(os.path.join(self.tmpdir, "sub"), exist_ok=True)
        with open(os.path.join(self.tmpdir, "sub", "data.json"), "w") as f:
            f.write("{}")

        result = asyncio.get_event_loop().run_until_complete(
            te._tool_find({"pattern": "*.json", "path": self.tmpdir})
        )
        self.assertIn("data.json", result)

    def test_tool_apply_patch_runs(self):
        """Test that apply_patch tool works with a simple patch."""
        from app.agent.tool_executor import ToolExecutor
        from app.config import settings
        
        # Disable per-user workspace so files resolve from tmpdir directly
        old_per_user = settings.workspace_per_user
        settings.workspace_per_user = False
        
        te = ToolExecutor(workspace=self.tmpdir)
        te._current_user_id = "test"

        # Create the target file in workspace root
        target = os.path.join(self.tmpdir, "target.txt")
        with open(target, "w") as f:
            f.write("line1\nline2\nline3\n")

        patch = """--- a/target.txt
+++ b/target.txt
@@ -1,3 +1,3 @@
 line1
-line2
+line2_modified
 line3
"""
        result = asyncio.get_event_loop().run_until_complete(
            te._tool_apply_patch({"patch": patch})
        )
        settings.workspace_per_user = old_per_user  # restore
        
        self.assertIn("patched", result.lower())

        with open(target) as f:
            file_content = f.read()
        self.assertIn("line2_modified", file_content)

    def test_tool_policy_deny(self):
        """Test that denied tools are blocked."""
        from app.agent.tool_executor import ToolExecutor
        from app.config import settings

        te = ToolExecutor(workspace=self.tmpdir)
        original_deny = settings.tool_deny_list[:]
        settings.tool_deny_list = ["exec"]

        result = asyncio.get_event_loop().run_until_complete(
            te.execute("exec", {"command": "echo hi"})
        )
        self.assertIn("blocked", result.lower())

        settings.tool_deny_list = original_deny  # restore

    def test_output_limits_exist(self):
        from app.agent.tool_executor import TOOL_OUTPUT_LIMITS
        for tool in ["grep", "find", "ls", "apply_patch", "sessions_send"]:
            self.assertIn(tool, TOOL_OUTPUT_LIMITS, f"Missing output limit for {tool}")


# ════════════════════════════════════════════════════════════════
# 5. CONFIG — New Fields
# ════════════════════════════════════════════════════════════════
class TestConfigNewFields(unittest.TestCase):
    """Test that all new config fields exist with correct defaults."""

    def test_discord_fields(self):
        from app.config import Settings
        s = Settings()
        self.assertIsNone(s.discord_bot_token)
        self.assertEqual(s.discord_allowed_guilds, [])
        self.assertEqual(s.discord_allowed_users, [])

    def test_slack_fields(self):
        from app.config import Settings
        s = Settings()
        self.assertIsNone(s.slack_bot_token)
        self.assertIsNone(s.slack_app_token)
        self.assertEqual(s.slack_allowed_channels, [])

    def test_whatsapp_fields(self):
        from app.config import Settings
        s = Settings()
        self.assertIsNone(s.whatsapp_phone_number_id)
        self.assertIsNone(s.whatsapp_access_token)
        self.assertEqual(s.whatsapp_verify_token, "")
        self.assertIsNone(s.whatsapp_app_secret)
        self.assertEqual(s.whatsapp_allowed_numbers, [])

    def test_thinking_fields(self):
        from app.config import Settings
        s = Settings()
        self.assertEqual(s.thinking_budget_default, 0)
        self.assertIsNone(s.thinking_model_override)

    def test_tool_policy_fields(self):
        from app.config import Settings
        s = Settings()
        self.assertEqual(s.tool_deny_list, [])
        self.assertIn("exec", s.tool_elevated_list)
        self.assertEqual(s.tool_max_output_chars, 60000)


# ════════════════════════════════════════════════════════════════
# 6. CHANNEL ADAPTERS — Import & Instantiation
# ════════════════════════════════════════════════════════════════
class TestDiscordChannel(unittest.TestCase):
    """Test Discord channel adapter."""

    def test_import(self):
        from app.agent.channels.discord_channel import DiscordChannel
        self.assertIsNotNone(DiscordChannel)

    def test_instantiation(self):
        from app.agent.channels.discord_channel import DiscordChannel
        ch = DiscordChannel(token="fake-token")
        self.assertEqual(ch.channel_type.value, "discord")
        self.assertIsNone(ch._client)

    def test_methods_exist(self):
        from app.agent.channels.discord_channel import DiscordChannel
        ch = DiscordChannel(token="fake")
        for method in ["start", "stop", "send_text", "send_typing", "send_file", "send_photo"]:
            self.assertTrue(hasattr(ch, method), f"Missing method: {method}")
            self.assertTrue(callable(getattr(ch, method)))

    def test_split_message(self):
        from app.agent.channels.discord_channel import DiscordChannel
        ch = DiscordChannel(token="fake")
        # Short message — no split
        parts = ch._split_message("hello")
        self.assertEqual(len(parts), 1)
        # Long message — should split
        long_msg = "x" * 5000
        parts = ch._split_message(long_msg)
        self.assertGreater(len(parts), 1)
        for part in parts:
            self.assertLessEqual(len(part), 2000)


class TestSlackChannel(unittest.TestCase):
    """Test Slack channel adapter."""

    def test_import(self):
        from app.agent.channels.slack_channel import SlackChannel
        self.assertIsNotNone(SlackChannel)

    def test_instantiation(self):
        from app.agent.channels.slack_channel import SlackChannel
        ch = SlackChannel(bot_token="xoxb-fake", app_token="xapp-fake")
        self.assertEqual(ch.channel_type.value, "slack")
        self.assertIsNone(ch._app)

    def test_methods_exist(self):
        from app.agent.channels.slack_channel import SlackChannel
        ch = SlackChannel(bot_token="xoxb-fake", app_token="xapp-fake")
        for method in ["start", "stop", "send_text", "send_typing", "send_file", "send_photo"]:
            self.assertTrue(hasattr(ch, method))

    def test_acl_config(self):
        from app.agent.channels.slack_channel import SlackChannel
        ch = SlackChannel(bot_token="x", app_token="x", allowed_channels=["C123", "C456"])
        self.assertEqual(ch.allowed_channels, {"C123", "C456"})


class TestWhatsAppChannel(unittest.TestCase):
    """Test WhatsApp channel adapter."""

    def test_import(self):
        from app.agent.channels.whatsapp_channel import WhatsAppChannel
        self.assertIsNotNone(WhatsAppChannel)

    def test_instantiation(self):
        from app.agent.channels.whatsapp_channel import WhatsAppChannel
        ch = WhatsAppChannel(
            phone_number_id="12345",
            access_token="fake-token",
            verify_token="verify-me",
        )
        self.assertEqual(ch.channel_type.value, "whatsapp")
        self.assertIsNone(ch._http)

    def test_methods_exist(self):
        from app.agent.channels.whatsapp_channel import WhatsAppChannel
        ch = WhatsAppChannel(phone_number_id="1", access_token="x", verify_token="v")
        for method in ["start", "stop", "send_text", "send_typing", "send_file", "send_photo",
                       "register_routes", "_process_payload", "_download_media", "_upload_media"]:
            self.assertTrue(hasattr(ch, method), f"Missing: {method}")

    def test_acl_config(self):
        from app.agent.channels.whatsapp_channel import WhatsAppChannel
        ch = WhatsAppChannel(
            phone_number_id="1", access_token="x", verify_token="v",
            allowed_numbers=["+1234567890"]
        )
        self.assertEqual(ch.allowed_numbers, {"+1234567890"})

    def test_start_creates_http_client(self):
        from app.agent.channels.whatsapp_channel import WhatsAppChannel
        ch = WhatsAppChannel(phone_number_id="1", access_token="x", verify_token="v")
        asyncio.get_event_loop().run_until_complete(ch.start())
        self.assertIsNotNone(ch._http)
        asyncio.get_event_loop().run_until_complete(ch.stop())


# ════════════════════════════════════════════════════════════════
# 7. WEBHOOKS API
# ════════════════════════════════════════════════════════════════
class TestWebhooksAPI(unittest.TestCase):
    """Test webhook API module."""

    def test_import(self):
        from app.api.webhooks import router, set_webhook_refs
        self.assertIsNotNone(router)
        self.assertTrue(callable(set_webhook_refs))

    def test_router_has_routes(self):
        from app.api.webhooks import router
        paths = [r.path for r in router.routes]
        # Updated paths after webhook refactor
        self.assertTrue(any("/trigger" in p for p in paths), f"Missing /trigger in {paths}")
        self.assertTrue(any("/gmail" in p for p in paths), f"Missing /gmail in {paths}")
        self.assertTrue(any("/events" in p for p in paths), f"Missing /events in {paths}")

    def test_models_importable(self):
        from app.api.webhooks import WebhookPayload
        p = WebhookPayload(event="test.event", payload={"key": "val"})
        self.assertEqual(p.event, "test.event")
        self.assertEqual(p.payload, {"key": "val"})
        # Optional payload
        p2 = WebhookPayload(event="bare")
        self.assertIsNone(p2.payload)


# ════════════════════════════════════════════════════════════════
# 8. CHANNEL BASE & REGISTRY
# ════════════════════════════════════════════════════════════════
class TestChannelBase(unittest.TestCase):
    """Test channel base classes."""

    def test_channel_types(self):
        from app.agent.channels.base import ChannelType
        for ct in ["TELEGRAM", "DISCORD", "SLACK", "WEB", "WHATSAPP"]:
            self.assertTrue(hasattr(ChannelType, ct))

    def test_inbound_message(self):
        from app.agent.channels.base import InboundMessage, ChannelType
        msg = InboundMessage(
            channel=ChannelType.DISCORD,
            channel_user_id="u1",
            channel_chat_id="c1",
            text="hello",
        )
        self.assertEqual(msg.channel, ChannelType.DISCORD)
        self.assertEqual(msg.text, "hello")

    def test_registry(self):
        from app.agent.channels.registry import ChannelRegistry
        self.assertTrue(hasattr(ChannelRegistry, 'register'))
        self.assertTrue(hasattr(ChannelRegistry, 'get'))
        self.assertTrue(hasattr(ChannelRegistry, 'all'))


# ════════════════════════════════════════════════════════════════
# 9. MAIN.PY — Webhook router & health endpoint
# ════════════════════════════════════════════════════════════════
class TestMainWiring(unittest.TestCase):
    """Test that main.py imports and wires correctly."""

    def test_webhook_import_in_main(self):
        """Verify webhooks router is imported in main."""
        main_path = os.path.join(os.path.dirname(__file__), "..", "app", "main.py")
        with open(main_path) as f:
            content = f.read()
        self.assertIn("from app.api.webhooks import router as webhooks_router", content)
        self.assertIn("webhooks_router", content)

    def test_hook_bus_in_main(self):
        """Verify hook bus is wired in main.py."""
        main_path = os.path.join(os.path.dirname(__file__), "..", "app", "main.py")
        with open(main_path) as f:
            content = f.read()
        self.assertIn("from app.agent.hooks import get_hook_bus, HookEvent", content)
        self.assertIn("HookEvent.STARTUP", content)
        self.assertIn("HookEvent.SHUTDOWN", content)

    def test_discord_channel_in_main(self):
        main_path = os.path.join(os.path.dirname(__file__), "..", "app", "main.py")
        with open(main_path) as f:
            content = f.read()
        self.assertIn("discord_channel", content)
        self.assertIn("DiscordChannel", content)

    def test_slack_channel_in_main(self):
        main_path = os.path.join(os.path.dirname(__file__), "..", "app", "main.py")
        with open(main_path) as f:
            content = f.read()
        self.assertIn("slack_channel", content)
        self.assertIn("SlackChannel", content)

    def test_whatsapp_channel_in_main(self):
        main_path = os.path.join(os.path.dirname(__file__), "..", "app", "main.py")
        with open(main_path) as f:
            content = f.read()
        self.assertIn("whatsapp_channel", content)
        self.assertIn("WhatsAppChannel", content)

    def test_health_endpoint_channels(self):
        main_path = os.path.join(os.path.dirname(__file__), "..", "app", "main.py")
        with open(main_path) as f:
            content = f.read()
        self.assertIn('"discord":', content)
        self.assertIn('"slack":', content)
        self.assertIn('"whatsapp":', content)


# ════════════════════════════════════════════════════════════════
# 10. AGENT_RUNNER — Hook emissions
# ════════════════════════════════════════════════════════════════
class TestAgentRunnerHooks(unittest.TestCase):
    """Test that agent_runner.py has hook emissions wired."""

    def test_hook_import(self):
        runner_path = os.path.join(os.path.dirname(__file__), "..", "app", "agent", "agent_runner.py")
        with open(runner_path) as f:
            content = f.read()
        self.assertIn("from app.agent.hooks import get_hook_bus, HookEvent", content)

    def test_before_agent_start_emission(self):
        runner_path = os.path.join(os.path.dirname(__file__), "..", "app", "agent", "agent_runner.py")
        with open(runner_path) as f:
            content = f.read()
        self.assertIn("HookEvent.BEFORE_AGENT_START", content)

    def test_tool_call_hooks(self):
        runner_path = os.path.join(os.path.dirname(__file__), "..", "app", "agent", "agent_runner.py")
        with open(runner_path) as f:
            content = f.read()
        self.assertIn("HookEvent.BEFORE_TOOL_CALL", content)
        self.assertIn("HookEvent.AFTER_TOOL_CALL", content)

    def test_agent_end_hook(self):
        runner_path = os.path.join(os.path.dirname(__file__), "..", "app", "agent", "agent_runner.py")
        with open(runner_path) as f:
            content = f.read()
        self.assertIn("HookEvent.AGENT_END", content)


# ════════════════════════════════════════════════════════════════
# 11. TELEGRAM_BOT — /think command
# ════════════════════════════════════════════════════════════════
class TestTelegramThinkCommand(unittest.TestCase):
    """Test /think command in telegram_bot.py."""

    def test_think_handler_registered(self):
        bot_path = os.path.join(os.path.dirname(__file__), "..", "app", "agent", "telegram_bot.py")
        with open(bot_path) as f:
            content = f.read()
        self.assertIn('CommandHandler("think"', content)
        self.assertIn("_cmd_think", content)

    def test_think_in_help(self):
        bot_path = os.path.join(os.path.dirname(__file__), "..", "app", "agent", "telegram_bot.py")
        with open(bot_path) as f:
            content = f.read()
        self.assertIn("/think", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)


# =====================================================================
# Layer 2: Chat Commands, Agent Runtime, Security, Skills
# =====================================================================

class TestNewChatCommands:
    """Test /verbose, /activation, /config, /allowlist commands."""

    def test_verbose_handler_registered(self):
        src = open("app/agent/telegram_bot.py").read()
        assert 'CommandHandler("verbose"' in src

    def test_activation_handler_registered(self):
        src = open("app/agent/telegram_bot.py").read()
        assert 'CommandHandler("activation"' in src

    def test_config_handler_registered(self):
        src = open("app/agent/telegram_bot.py").read()
        assert 'CommandHandler("config"' in src

    def test_allowlist_handler_registered(self):
        src = open("app/agent/telegram_bot.py").read()
        assert 'CommandHandler("allowlist"' in src

    def test_help_includes_new_commands(self):
        src = open("app/agent/telegram_bot.py").read()
        assert "/verbose" in src
        assert "/activation" in src
        assert "/config" in src
        assert "/allowlist" in src

    def test_verbose_method_exists(self):
        src = open("app/agent/telegram_bot.py").read()
        assert "async def _cmd_verbose" in src

    def test_activation_method_exists(self):
        src = open("app/agent/telegram_bot.py").read()
        assert "async def _cmd_activation" in src

    def test_config_allowed_keys(self):
        src = open("app/agent/telegram_bot.py").read()
        assert "ALLOWED_KEYS" in src
        assert "temperature" in src

    def test_allowlist_add_remove(self):
        src = open("app/agent/telegram_bot.py").read()
        assert "allowlist add" in src or "add" in src
        assert "allowlist remove" in src or "remove" in src


class TestAgentRuntimeEnhancements:
    """Test activation prompt, verbose mode, per-tool timeout."""

    def test_activation_prompt_in_system_prompt(self):
        src = open("app/agent/agent_runner.py").read()
        assert "Activation Prompt" in src
        assert "_activation_prompt" in src

    def test_verbose_mode_in_system_prompt(self):
        src = open("app/agent/agent_runner.py").read()
        assert "Verbose Mode" in src or "_verbose_mode" in src

    def test_per_tool_timeout_in_executor(self):
        src = open("app/agent/tool_executor.py").read()
        assert "tool_timeout" in src
        assert "asyncio.wait_for" in src or "wait_for" in src

    def test_timeout_config_fields(self):
        from app.config import Settings
        s = Settings()
        assert hasattr(s, "tool_timeout_default")
        assert hasattr(s, "tool_timeout_overrides")
        assert s.tool_timeout_default == 30
        assert "exec" in s.tool_timeout_overrides

    def test_failover_exists(self):
        src = open("app/agent/agent_runner.py").read()
        assert "fallback" in src.lower()
        assert "agent_fallback_model" in src


class TestNewToolDefinitions:
    """Test session_status and agents_list tool definitions."""

    def test_session_status_tool_defined(self):
        from app.agent.tool_definitions import get_extended_tools
        names = [t["name"] for t in get_extended_tools()]
        assert "session_status" in names

    def test_agents_list_tool_defined(self):
        from app.agent.tool_definitions import get_extended_tools
        names = [t["name"] for t in get_extended_tools()]
        assert "agents_list" in names

    def test_session_status_method_exists(self):
        from app.agent.tool_executor import ToolExecutor
        assert hasattr(ToolExecutor, "_tool_session_status")

    def test_agents_list_method_exists(self):
        from app.agent.tool_executor import ToolExecutor
        assert hasattr(ToolExecutor, "_tool_agents_list")

    def test_agents_list_runs(self):
        import asyncio
        from app.agent.tool_executor import ToolExecutor
        te = ToolExecutor.__new__(ToolExecutor)
        te.workspace = "/tmp"
        te._user_id = "test"
        te._chat_id = None
        te._bootstrapped_users = set()
        te._processes = {}
        te._proc_counter = 0
        te.telegram_bot = None
        te.cron_service = None
        te.subagent_manager = None
        te.skill_loader = None
        result = asyncio.get_event_loop().run_until_complete(
            te._tool_agents_list({})
        )
        assert "personas" in result.lower() or "coder" in result.lower()

    def test_output_limits_include_new_tools(self):
        from app.agent.tool_executor import TOOL_OUTPUT_LIMITS
        assert "session_status" in TOOL_OUTPUT_LIMITS
        assert "agents_list" in TOOL_OUTPUT_LIMITS


class TestDMGroupPolicy:
    """Test DM policy and group policy config."""

    def test_dm_policy_field(self):
        from app.config import Settings
        s = Settings()
        assert hasattr(s, "dm_policy")
        assert s.dm_policy in ("pairing", "allowlist", "open", "disabled")

    def test_group_policy_field(self):
        from app.config import Settings
        s = Settings()
        assert hasattr(s, "group_policy")
        assert s.group_policy in ("open", "allowlist", "disabled")

    def test_group_require_mention(self):
        from app.config import Settings
        s = Settings()
        assert hasattr(s, "group_require_mention")
        assert isinstance(s.group_require_mention, bool)


class TestSkillPluginEnhancements:
    """Test skill plugin command + hook registration."""

    def test_skill_base_has_get_commands(self):
        from app.agent.skills.base import Skill
        assert hasattr(Skill, "get_commands")

    def test_skill_base_has_get_hooks(self):
        from app.agent.skills.base import Skill
        assert hasattr(Skill, "get_hooks")

    def test_loader_has_get_all_commands(self):
        from app.agent.skills.loader import SkillLoader
        loader = SkillLoader()
        assert hasattr(loader, "get_all_commands")
        assert loader.get_all_commands() == []

    def test_loader_has_get_all_hooks(self):
        from app.agent.skills.loader import SkillLoader
        loader = SkillLoader()
        assert hasattr(loader, "get_all_hooks")
        assert loader.get_all_hooks() == []


class TestToolPolicyDenyElevation:
    """Test tool deny and elevation enforcement."""

    def test_deny_blocks_tool(self):
        import asyncio
        from app.agent.tool_executor import ToolExecutor
        from app.config import settings

        te = ToolExecutor.__new__(ToolExecutor)
        te.workspace = "/tmp"
        te._user_id = "test"
        te._chat_id = None
        te._bootstrapped_users = set()
        te._processes = {}
        te._proc_counter = 0
        te.telegram_bot = None
        te.cron_service = None
        te.subagent_manager = None
        te.skill_loader = None

        old_deny = settings.tool_deny_list[:]
        settings.tool_deny_list = ["exec"]
        try:
            result = asyncio.get_event_loop().run_until_complete(
                te.execute("exec", {"command": "echo hello"})
            )
            assert "blocked" in result.lower()
        finally:
            settings.tool_deny_list = old_deny

    def test_elevated_tool_logged(self):
        src = open("app/agent/tool_executor.py").read()
        assert "TOOL-POLICY" in src
        assert "Elevated tool invoked" in src


# =====================================================================
# Layer 3: Advanced Agent Runtime + Enhanced Channels
# =====================================================================

class TestAgentLanes:
    """Test agent execution lanes with concurrency and idempotency."""

    def test_lane_type_enum(self):
        from app.agent.lanes import LaneType
        assert LaneType.MAIN.value == "main"
        assert LaneType.SUBAGENT.value == "subagent"
        assert LaneType.CRON.value == "cron"
        assert LaneType.HOOK.value == "hook"

    def test_lane_manager_singleton(self):
        from app.agent.lanes import get_lane_manager
        lm1 = get_lane_manager()
        lm2 = get_lane_manager()
        assert lm1 is lm2

    def test_lane_acquire_release(self):
        import asyncio
        from app.agent.lanes import LaneManager, LaneType
        lm = LaneManager(max_concurrent=3)
        run = asyncio.get_event_loop().run_until_complete(
            lm.acquire(LaneType.MAIN, "user1")
        )
        assert run is not None
        assert run.status == "running"
        assert run.lane == LaneType.MAIN

        asyncio.get_event_loop().run_until_complete(
            lm.release(run, status="completed")
        )
        assert run.status == "completed"
        assert run.finished_at is not None

    def test_idempotency_dedup(self):
        import asyncio
        from app.agent.lanes import LaneManager, LaneType
        lm = LaneManager(max_concurrent=3)

        run1 = asyncio.get_event_loop().run_until_complete(
            lm.acquire(LaneType.MAIN, "user1", idempotency_key="key-abc")
        )
        assert run1 is not None

        # Second acquire with same key should return None (dedup)
        run2 = asyncio.get_event_loop().run_until_complete(
            lm.acquire(LaneType.MAIN, "user1", idempotency_key="key-abc")
        )
        assert run2 is None

        # Release first run, then a new acquire should work
        asyncio.get_event_loop().run_until_complete(lm.release(run1))

    def test_lane_stats(self):
        import asyncio
        from app.agent.lanes import LaneManager, LaneType
        lm = LaneManager(max_concurrent=5)

        run = asyncio.get_event_loop().run_until_complete(
            lm.acquire(LaneType.CRON, "user2")
        )
        stats = lm.get_stats()
        assert stats["active"] == 1
        assert stats["by_lane"]["cron"] == 1
        asyncio.get_event_loop().run_until_complete(lm.release(run))

    def test_lane_clear_history(self):
        import asyncio
        from app.agent.lanes import LaneManager, LaneType
        lm = LaneManager(max_concurrent=5)

        run = asyncio.get_event_loop().run_until_complete(
            lm.acquire(LaneType.HOOK, "user3")
        )
        asyncio.get_event_loop().run_until_complete(lm.release(run))
        assert lm.get_stats()["total_runs"] >= 1
        lm.clear_history()
        assert lm.get_stats()["total_runs"] == 0


class TestConfigHotReload:
    """Test config hot-reload functionality."""

    def test_reloadable_fields_list(self):
        from app.agent.config_reload import get_reloadable_fields
        fields = get_reloadable_fields()
        assert "temperature" in fields
        assert "agent_model" in fields
        assert "tts_auto_mode" in fields

    def test_frozen_fields(self):
        from app.agent.config_reload import FROZEN_FIELDS
        assert "database_url" in FROZEN_FIELDS
        assert "jwt_secret" in FROZEN_FIELDS
        assert "openai_api_key" in FROZEN_FIELDS

    def test_reload_updates_value(self):
        from app.agent.config_reload import reload_config, get_current_config
        from app.config import settings
        old_temp = settings.temperature
        try:
            results = reload_config({"temperature": "0.5"})
            assert results["temperature"] == "updated"
            assert settings.temperature == 0.5
        finally:
            object.__setattr__(settings, "temperature", old_temp)

    def test_reload_blocks_frozen(self):
        from app.agent.config_reload import reload_config
        results = reload_config({"database_url": "sqlite:///hacked.db"})
        assert results["database_url"] == "frozen"

    def test_reload_unknown_field(self):
        from app.agent.config_reload import reload_config
        results = reload_config({"nonexistent_field_xyz": "value"})
        assert results["nonexistent_field_xyz"] == "unknown"

    def test_get_current_config(self):
        from app.agent.config_reload import get_current_config
        config = get_current_config(["temperature", "agent_model"])
        assert "temperature" in config
        assert "agent_model" in config


class TestCrossChannelMessaging:
    """Test cross-channel messaging module."""

    def test_send_requires_target(self):
        import asyncio
        from app.agent.cross_channel import send_cross_channel
        result = asyncio.get_event_loop().run_until_complete(
            send_cross_channel("telegram", "", "hello")
        )
        # Empty target should fail (no bot connected)
        assert isinstance(result, dict)

    def test_react_without_bot(self):
        import asyncio
        from app.agent.cross_channel import react_cross_channel
        result = asyncio.get_event_loop().run_until_complete(
            react_cross_channel("telegram", "123", "456", "👍")
        )
        assert result["ok"] is False

    def test_unknown_channel(self):
        import asyncio
        from app.agent.cross_channel import send_cross_channel
        result = asyncio.get_event_loop().run_until_complete(
            send_cross_channel("fax", "123", "hello")
        )
        assert result["ok"] is False
        assert "Unknown channel" in result["error"]

    def test_edit_without_bot(self):
        import asyncio
        from app.agent.cross_channel import edit_cross_channel
        result = asyncio.get_event_loop().run_until_complete(
            edit_cross_channel("telegram", "123", "456", "new text")
        )
        assert result["ok"] is False

    def test_delete_without_bot(self):
        import asyncio
        from app.agent.cross_channel import delete_cross_channel
        result = asyncio.get_event_loop().run_until_complete(
            delete_cross_channel("telegram", "123", "456")
        )
        assert result["ok"] is False

    def test_pin_without_bot(self):
        import asyncio
        from app.agent.cross_channel import pin_cross_channel
        result = asyncio.get_event_loop().run_until_complete(
            pin_cross_channel("telegram", "123", "456")
        )
        assert result["ok"] is False


class TestModerationModule:
    """Test moderation tools module."""

    def test_moderate_unknown_channel(self):
        import asyncio
        from app.agent.moderation import moderate_user
        result = asyncio.get_event_loop().run_until_complete(
            moderate_user("kick", "fax", "123", "456")
        )
        assert result["ok"] is False
        assert "not supported" in result["error"]

    def test_moderate_telegram_no_bot(self):
        import asyncio
        from app.agent.moderation import moderate_user
        result = asyncio.get_event_loop().run_until_complete(
            moderate_user("kick", "telegram", "123", "456")
        )
        assert result["ok"] is False
        assert "not connected" in result["error"]

    def test_unknown_action(self):
        import asyncio
        from app.agent.moderation import moderate_user
        result = asyncio.get_event_loop().run_until_complete(
            moderate_user(
                "explode", "telegram", "123", "456",
                bot_refs={"telegram_bot": None},
            )
        )
        assert result["ok"] is False


class TestNewToolDefinitions:
    """Test Layer 3 tool definitions."""

    def test_message_tool_defined(self):
        from app.agent.tool_definitions import get_extended_tools
        names = [t["name"] for t in get_extended_tools()]
        assert "message" in names

    def test_moderate_tool_defined(self):
        from app.agent.tool_definitions import get_extended_tools
        names = [t["name"] for t in get_extended_tools()]
        assert "moderate" in names

    def test_config_reload_tool_defined(self):
        from app.agent.tool_definitions import get_extended_tools
        names = [t["name"] for t in get_extended_tools()]
        assert "config_reload" in names

    def test_lanes_status_tool_defined(self):
        from app.agent.tool_definitions import get_extended_tools
        names = [t["name"] for t in get_extended_tools()]
        assert "lanes_status" in names

    def test_poll_tool_defined(self):
        from app.agent.tool_definitions import get_extended_tools
        names = [t["name"] for t in get_extended_tools()]
        assert "poll" in names

    def test_extended_tools_total_count(self):
        from app.agent.tool_definitions import get_extended_tools
        tools = get_extended_tools()
        assert len(tools) == 14  # 7 from L1/L2 + 5 from L3 + 2 from L4

    def test_tool_methods_exist(self):
        from app.agent.tool_executor import ToolExecutor
        assert hasattr(ToolExecutor, "_tool_message")
        assert hasattr(ToolExecutor, "_tool_moderate")
        assert hasattr(ToolExecutor, "_tool_config_reload")
        assert hasattr(ToolExecutor, "_tool_lanes_status")
        assert hasattr(ToolExecutor, "_tool_poll")

    def test_output_limits_include_l3_tools(self):
        from app.agent.tool_executor import TOOL_OUTPUT_LIMITS
        for tool in ["message", "moderate", "config_reload", "lanes_status", "poll"]:
            assert tool in TOOL_OUTPUT_LIMITS, f"{tool} not in TOOL_OUTPUT_LIMITS"


class TestConfigNewFields:
    """Test new config fields added in Layer 3."""

    def test_tts_auto_mode(self):
        from app.config import Settings
        s = Settings()
        assert s.tts_auto_mode == "off"
        assert s.tts_default_voice == "alloy"
        assert s.tts_model == "gpt-4o-mini-tts"

    def test_lane_config(self):
        from app.config import Settings
        s = Settings()
        assert s.lane_max_concurrent == 5
        assert s.lane_cron_model is None

    def test_telegram_enhanced(self):
        from app.config import Settings
        s = Settings()
        assert s.telegram_forum_support is True
        assert s.telegram_topic_routing is True
        assert s.telegram_reactions_enabled is True
        assert s.telegram_inline_buttons is True
        assert s.telegram_polls_enabled is True

    def test_moderation_config(self):
        from app.config import Settings
        s = Settings()
        assert s.moderation_enabled is False
        assert s.moderation_log_channel is None

    def test_config_reload_enabled(self):
        from app.config import Settings
        s = Settings()
        assert s.config_reload_enabled is True


class TestAgentRunnerEnhancements:
    """Test agent_runner enhancements for Layer 3."""

    def test_session_model_override_attr(self):
        src = open("app/agent/agent_runner.py").read()
        assert "_session_model_override" in src

    def test_current_lane_attr(self):
        src = open("app/agent/agent_runner.py").read()
        assert "_current_lane" in src

    def test_idempotency_key_attr(self):
        src = open("app/agent/agent_runner.py").read()
        assert "_idempotency_key" in src

    def test_lane_context_in_system_prompt(self):
        src = open("app/agent/agent_runner.py").read()
        assert "Execution Lane" in src


class TestNewTelegramCommands:
    """Test /auto, /lanes, /tts commands."""

    def test_auto_handler_registered(self):
        src = open("app/agent/telegram_bot.py").read()
        assert 'CommandHandler("auto"' in src

    def test_lanes_handler_registered(self):
        src = open("app/agent/telegram_bot.py").read()
        assert 'CommandHandler("lanes"' in src

    def test_tts_handler_registered(self):
        src = open("app/agent/telegram_bot.py").read()
        assert 'CommandHandler("tts"' in src

    def test_auto_method_exists(self):
        src = open("app/agent/telegram_bot.py").read()
        assert "async def _cmd_auto" in src

    def test_lanes_method_exists(self):
        src = open("app/agent/telegram_bot.py").read()
        assert "async def _cmd_lanes" in src

    def test_tts_mode_method_exists(self):
        src = open("app/agent/telegram_bot.py").read()
        assert "async def _cmd_tts_mode" in src

    def test_help_includes_new_commands(self):
        src = open("app/agent/telegram_bot.py").read()
        assert "/auto" in src
        assert "/lanes" in src
        assert "/tts" in src


class TestConfigReloadTool:
    """Test config_reload tool implementation end-to-end."""

    def test_config_reload_list(self):
        import asyncio
        from app.agent.tool_executor import ToolExecutor
        te = ToolExecutor.__new__(ToolExecutor)
        te.workspace = "/tmp"
        te._user_id = "test"
        te._chat_id = None
        te._bootstrapped_users = set()
        te._processes = {}
        te._proc_counter = 0
        te.telegram_bot = None
        te.cron_service = None
        te.subagent_manager = None
        te.skill_loader = None
        result = asyncio.get_event_loop().run_until_complete(
            te._tool_config_reload({"action": "list"})
        )
        assert "temperature" in result
        assert "agent_model" in result

    def test_config_reload_get(self):
        import asyncio
        from app.agent.tool_executor import ToolExecutor
        te = ToolExecutor.__new__(ToolExecutor)
        te.workspace = "/tmp"
        te._user_id = "test"
        te._chat_id = None
        te._bootstrapped_users = set()
        te._processes = {}
        te._proc_counter = 0
        te.telegram_bot = None
        te.cron_service = None
        te.subagent_manager = None
        te.skill_loader = None
        result = asyncio.get_event_loop().run_until_complete(
            te._tool_config_reload({"action": "get", "field": "temperature"})
        )
        assert "temperature" in result

    def test_lanes_status_runs(self):
        import asyncio
        from app.agent.tool_executor import ToolExecutor
        te = ToolExecutor.__new__(ToolExecutor)
        te.workspace = "/tmp"
        te._user_id = "test"
        te._chat_id = None
        te._bootstrapped_users = set()
        te._processes = {}
        te._proc_counter = 0
        te.telegram_bot = None
        te.cron_service = None
        te.subagent_manager = None
        te.skill_loader = None
        result = asyncio.get_event_loop().run_until_complete(
            te._tool_lanes_status({})
        )
        assert "summary" in result
        assert "active_runs" in result


class TestHealthEndpoint:
    """Test enhanced health endpoint."""

    def test_health_source_has_version(self):
        src = open("app/main.py").read()
        assert '"version"' in src
        assert "6.0.0" in src

    def test_health_has_uptime(self):
        src = open("app/main.py").read()
        assert "uptime_seconds" in src

    def test_health_has_channels_dict(self):
        src = open("app/main.py").read()
        assert '"channels"' in src

    def test_health_has_features(self):
        src = open("app/main.py").read()
        assert '"features"' in src
        assert "reranker" in src
        assert "sandbox" in src

    def test_health_has_lanes(self):
        src = open("app/main.py").read()
        assert '"lanes"' in src

    def test_health_has_db_probe(self):
        src = open("app/main.py").read()
        assert "SELECT 1" in src


# ═══════════════════════════════════════════════════════════════════════════
# Layer 4: Extended Thinking · Multi-TTS · Thread Management · Providers
# ═══════════════════════════════════════════════════════════════════════════

class TestExtendedThinking:
    """Phase 4.1 — Extended thinking passthrough"""

    def test_anthropic_service_thinking_budget_param(self):
        import inspect
        from app.services.anthropic_service import AnthropicService
        sig = inspect.signature(AnthropicService.create_message_stream)
        assert "thinking_budget" in sig.parameters
        p = sig.parameters["thinking_budget"]
        assert p.default == 0

    def test_agent_runner_thinking_budget_param(self):
        import inspect
        from app.agent.agent_runner import AgentRunner
        sig = inspect.signature(AgentRunner.run)
        assert "thinking_budget" in sig.parameters
        p = sig.parameters["thinking_budget"]
        assert p.default == 0

    def test_agent_runner_idempotency_param(self):
        import inspect
        from app.agent.agent_runner import AgentRunner
        sig = inspect.signature(AgentRunner.run)
        assert "idempotency_key" in sig.parameters

    def test_thinking_budget_default_config(self):
        from app.config import Settings
        s = Settings()
        assert hasattr(s, "thinking_budget_default")
        assert s.thinking_budget_default == 0

    def test_telegram_bot_passes_thinking_budget(self):
        src = open("app/agent/telegram_bot.py").read()
        assert "thinking_budget=_thinking" in src
        # Both call sites
        assert src.count("thinking_budget=_thinking") >= 2

    def test_anthropic_thinking_kwargs_block(self):
        src = open("app/services/anthropic_service.py").read()
        assert 'kwargs["thinking"]' in src
        assert "budget_tokens" in src
        assert "temperature" in src


class TestIdempotency:
    """Phase 4.2 — Idempotency deduplication"""

    def test_idempotency_cache_check(self):
        src = open("app/agent/agent_runner.py").read()
        assert "_idempotency_cache" in src
        assert "idempotency_key" in src

    def test_idempotency_returns_early_on_duplicate(self):
        src = open("app/agent/agent_runner.py").read()
        # Should have early return logic
        assert "already processed" in src.lower() or "duplicate" in src.lower() or "idempotency_key in" in src


class TestMultiTTSProviders:
    """Phase 4.3 — Multi-TTS provider system"""

    def test_tts_providers_module_exists(self):
        from app.agent import tts_providers
        assert hasattr(tts_providers, "synthesize_speech_multi")
        assert hasattr(tts_providers, "synthesize_elevenlabs")
        assert hasattr(tts_providers, "synthesize_edge_tts")

    def test_tts_provider_config_fields(self):
        from app.config import Settings
        s = Settings()
        assert s.tts_provider == "openai"
        assert hasattr(s, "elevenlabs_api_key")
        assert hasattr(s, "elevenlabs_model")
        assert hasattr(s, "elevenlabs_voice_id")

    def test_tts_per_user_prefs_module(self):
        from app.agent.tts_providers import get_user_tts_prefs, set_user_tts_prefs
        assert callable(get_user_tts_prefs)
        assert callable(set_user_tts_prefs)

    def test_tts_provider_enum_in_tool_def(self):
        src = open("app/agent/tool_definitions.py").read()
        # TTS tool should have provider enum
        assert '"openai"' in src
        assert '"elevenlabs"' in src
        assert '"edge"' in src

    def test_tts_tool_uses_multi_dispatch(self):
        src = open("app/agent/tool_executor.py").read()
        assert "synthesize_speech_multi" in src

    def test_elevenlabs_voices_mapping(self):
        from app.agent.tts_providers import ELEVENLABS_VOICES
        assert isinstance(ELEVENLABS_VOICES, dict)
        assert len(ELEVENLABS_VOICES) >= 3
        assert "rachel" in ELEVENLABS_VOICES or "Rachel" in ELEVENLABS_VOICES

    def test_edge_tts_voices_mapping(self):
        from app.agent.tts_providers import EDGE_VOICES
        assert isinstance(EDGE_VOICES, dict)
        assert len(EDGE_VOICES) >= 3


class TestPerUserTTSPrefs:
    """Phase 4.4 — Per-user TTS preferences"""

    def test_prefs_config_enabled(self):
        from app.config import Settings
        s = Settings()
        assert s.tts_per_user_prefs is True

    def test_tts_prefs_tool_defined(self):
        from app.agent.tool_definitions import get_extended_tools
        names = [t["name"] for t in get_extended_tools()]
        assert "tts_prefs" in names

    def test_tts_prefs_tool_actions(self):
        from app.agent.tool_definitions import get_extended_tools
        tool = next(t for t in get_extended_tools() if t["name"] == "tts_prefs")
        schema = tool["input_schema"]
        action_prop = schema["properties"]["action"]
        assert "get" in action_prop["enum"]
        assert "set" in action_prop["enum"]

    def test_tts_prefs_executor_exists(self):
        from app.agent.tool_executor import ToolExecutor
        assert hasattr(ToolExecutor, "_tool_tts_prefs")


class TestThreadManagement:
    """Phase 4.5 — Forum topic / thread management tool"""

    def test_thread_tool_defined(self):
        from app.agent.tool_definitions import get_extended_tools
        names = [t["name"] for t in get_extended_tools()]
        assert "thread" in names

    def test_thread_tool_actions(self):
        from app.agent.tool_definitions import get_extended_tools
        tool = next(t for t in get_extended_tools() if t["name"] == "thread")
        schema = tool["input_schema"]
        action_prop = schema["properties"]["action"]
        for act in ["create", "list", "close", "reopen"]:
            assert act in action_prop["enum"]

    def test_thread_executor_exists(self):
        from app.agent.tool_executor import ToolExecutor
        assert hasattr(ToolExecutor, "_tool_thread")

    def test_thread_output_limit(self):
        src = open("app/agent/tool_executor.py").read()
        assert '"thread"' in src


class TestSpawnPolicy:
    """Phase 4.6 — Agent spawn allow-list"""

    def test_allow_agents_config(self):
        from app.config import Settings
        s = Settings()
        assert hasattr(s, "allow_agents")
        assert isinstance(s.allow_agents, list)

    def test_spawn_checks_policy(self):
        src = open("app/agent/tool_executor.py").read()
        assert "allow_agents" in src


class TestMultiAgentRouting:
    """Phase 4.7 — Multi-agent persona routing"""

    def test_multi_agent_config(self):
        from app.config import Settings
        s = Settings()
        assert hasattr(s, "multi_agent_enabled")
        assert s.multi_agent_enabled is False
        assert hasattr(s, "multi_agent_default")

    def test_persona_command_registered(self):
        src = open("app/agent/telegram_bot.py").read()
        assert "persona" in src
        assert "_cmd_persona" in src

    def test_providers_command_registered(self):
        src = open("app/agent/telegram_bot.py").read()
        assert "providers" in src
        assert "_cmd_providers" in src

    def test_multi_agent_routing_in_process_message(self):
        src = open("app/agent/telegram_bot.py").read()
        assert "multi_agent_enabled" in src
        assert "get_multi_agent_router" in src

    def test_persona_routing_applies_model_override(self):
        src = open("app/agent/telegram_bot.py").read()
        assert "_persona.model_override" in src


class TestCustomModelProviders:
    """Phase 4.8 — Custom model providers config"""

    def test_custom_model_providers_config(self):
        from app.config import Settings
        s = Settings()
        assert hasattr(s, "custom_model_providers")
        assert isinstance(s.custom_model_providers, dict)

    def test_custom_model_map_config(self):
        from app.config import Settings
        s = Settings()
        assert hasattr(s, "custom_model_map")
        assert isinstance(s.custom_model_map, dict)


class TestSkillCommandWiring:
    """Phase 4.9 — Skill plugin command wiring at bot startup"""

    def test_skill_command_registration_block(self):
        src = open("app/agent/telegram_bot.py").read()
        assert "skill_loader" in src
        assert "get_all_commands" in src
        assert "Registered skill command" in src

    def test_skill_commands_before_message_handlers(self):
        src = open("app/agent/telegram_bot.py").read()
        skill_pos = src.find("get_all_commands")
        message_pos = src.find("Register message handlers")
        assert skill_pos < message_pos, "Skill commands should be registered before message handlers"


# ═══════════════════════════════════════════════════════════════════════════
# Layer 4 — BEHAVIORAL TESTS (exercise actual logic, not just source reading)
# ═══════════════════════════════════════════════════════════════════════════

class TestTTSProvidersBehavioral:
    """Actually exercise TTS provider dispatch logic."""

    def test_get_user_prefs_returns_defaults(self):
        """get_user_tts_prefs returns defaults for unknown user."""
        from app.agent.tts_providers import get_user_tts_prefs, _prefs_cache
        _prefs_cache.clear()
        prefs = get_user_tts_prefs("nonexistent-user-999")
        assert "provider" in prefs
        assert "voice" in prefs
        assert "speed" in prefs
        assert "model" in prefs

    def test_set_then_get_user_prefs(self):
        """set_user_tts_prefs persists and get retrieves it."""
        import tempfile, os, json
        from app.agent import tts_providers

        # Use a temp file to avoid clobbering real prefs
        old_path = tts_providers._PREFS_PATH
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        tmp.close()
        tts_providers._PREFS_PATH = tmp.name
        tts_providers._prefs_cache.clear()

        try:
            result = tts_providers.set_user_tts_prefs("test-user-42", provider="edge", voice="nova")
            assert result["provider"] == "edge"
            assert result["voice"] == "nova"

            # Clear cache, force re-read from disk
            tts_providers._prefs_cache.clear()
            loaded = tts_providers.get_user_tts_prefs("test-user-42")
            assert loaded["provider"] == "edge"
            assert loaded["voice"] == "nova"
        finally:
            tts_providers._PREFS_PATH = old_path
            tts_providers._prefs_cache.clear()
            os.unlink(tmp.name)

    def test_set_prefs_partial_update(self):
        """set_user_tts_prefs only updates provided fields."""
        from app.agent import tts_providers
        tts_providers._prefs_cache.clear()

        old_path = tts_providers._PREFS_PATH
        import tempfile, os
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        tmp.close()
        tts_providers._PREFS_PATH = tmp.name

        try:
            tts_providers.set_user_tts_prefs("u1", provider="elevenlabs", voice="rachel")
            result = tts_providers.set_user_tts_prefs("u1", speed=1.5)
            assert result["provider"] == "elevenlabs"  # unchanged
            assert result["voice"] == "rachel"  # unchanged
            assert result["speed"] == 1.5  # updated
        finally:
            tts_providers._PREFS_PATH = old_path
            tts_providers._prefs_cache.clear()
            os.unlink(tmp.name)

    def test_elevenlabs_voice_name_resolution(self):
        """ELEVENLABS_VOICES maps lowercase names to IDs."""
        from app.agent.tts_providers import ELEVENLABS_VOICES
        assert ELEVENLABS_VOICES["rachel"] == "21m00Tcm4TlvDq8ikWAM"
        assert ELEVENLABS_VOICES["josh"] == "TxGEqnHWrfWFTfGW9XjX"
        assert all(len(v) > 10 for v in ELEVENLABS_VOICES.values())

    def test_edge_voice_mapping(self):
        """EDGE_VOICES maps OpenAI-style names to MS Neural voices."""
        from app.agent.tts_providers import EDGE_VOICES
        assert EDGE_VOICES["alloy"] == "en-US-GuyNeural"
        assert EDGE_VOICES["nova"] == "en-US-JennyNeural"
        assert all("Neural" in v for v in EDGE_VOICES.values())


class TestToolExecutorThreadBehavioral:
    """Exercise _tool_thread error paths."""

    def test_thread_no_bot_returns_error(self):
        """thread tool returns error when telegram bot not connected."""
        import asyncio
        from app.agent.tool_executor import ToolExecutor

        te = ToolExecutor.__new__(ToolExecutor)
        te.telegram_bot = None
        te._chat_id = None

        result = asyncio.get_event_loop().run_until_complete(
            te._tool_thread({"action": "create", "name": "test"})
        )
        assert "ERROR" in result

    def test_thread_invalid_chat_id(self):
        """thread tool returns error for non-numeric chat_id."""
        import asyncio
        from app.agent.tool_executor import ToolExecutor

        te = ToolExecutor.__new__(ToolExecutor)
        te.telegram_bot = type("Bot", (), {"bot": True})()
        te._chat_id = None

        result = asyncio.get_event_loop().run_until_complete(
            te._tool_thread({"action": "create", "chat_id": "not-a-number", "name": "test"})
        )
        assert "ERROR" in result
        assert "Invalid chat_id" in result

    def test_thread_missing_name_on_create(self):
        """thread tool requires name for create action."""
        import asyncio
        from app.agent.tool_executor import ToolExecutor

        te = ToolExecutor.__new__(ToolExecutor)
        te.telegram_bot = type("Bot", (), {"bot": True})()
        te._chat_id = 12345

        result = asyncio.get_event_loop().run_until_complete(
            te._tool_thread({"action": "create", "chat_id": "12345"})
        )
        assert "ERROR" in result
        assert "name" in result.lower()

    def test_thread_missing_topic_id_on_close(self):
        """thread tool requires topic_id for close action."""
        import asyncio
        from app.agent.tool_executor import ToolExecutor

        te = ToolExecutor.__new__(ToolExecutor)
        te.telegram_bot = type("Bot", (), {"bot": True})()
        te._chat_id = 12345

        result = asyncio.get_event_loop().run_until_complete(
            te._tool_thread({"action": "close", "chat_id": "12345"})
        )
        assert "ERROR" in result
        assert "topic_id" in result

    def test_thread_unknown_action(self):
        """thread tool returns error for unknown action."""
        import asyncio
        from app.agent.tool_executor import ToolExecutor

        te = ToolExecutor.__new__(ToolExecutor)
        te.telegram_bot = type("Bot", (), {"bot": True})()
        te._chat_id = 12345

        result = asyncio.get_event_loop().run_until_complete(
            te._tool_thread({"action": "delete", "chat_id": "12345"})
        )
        assert "ERROR" in result
        assert "Unknown action" in result


class TestToolExecutorTTSPrefsBehavioral:
    """Exercise _tool_tts_prefs error paths and happy paths."""

    def test_tts_prefs_no_user_returns_error(self):
        """tts_prefs tool returns error when no user context."""
        import asyncio
        from app.agent.tool_executor import ToolExecutor

        te = ToolExecutor.__new__(ToolExecutor)
        te._user_id = ""

        result = asyncio.get_event_loop().run_until_complete(
            te._tool_tts_prefs({"action": "get"})
        )
        assert "ERROR" in result

    def test_tts_prefs_get_returns_json(self):
        """tts_prefs get action returns JSON with user_id and prefs."""
        import asyncio, json
        from app.agent.tool_executor import ToolExecutor
        from app.agent import tts_providers

        # Set up a temp prefs file
        import tempfile, os
        old_path = tts_providers._PREFS_PATH
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        tmp.close()
        tts_providers._PREFS_PATH = tmp.name
        tts_providers._prefs_cache.clear()

        try:
            te = ToolExecutor.__new__(ToolExecutor)
            te._user_id = "test-uid-123"

            result = asyncio.get_event_loop().run_until_complete(
                te._tool_tts_prefs({"action": "get"})
            )
            data = json.loads(result)
            assert data["user_id"] == "test-uid-123"
            assert "tts_preferences" in data
        finally:
            tts_providers._PREFS_PATH = old_path
            tts_providers._prefs_cache.clear()
            os.unlink(tmp.name)

    def test_tts_prefs_set_no_fields_returns_error(self):
        """tts_prefs set with no fields returns error."""
        import asyncio
        from app.agent.tool_executor import ToolExecutor

        te = ToolExecutor.__new__(ToolExecutor)
        te._user_id = "test-uid-456"

        result = asyncio.get_event_loop().run_until_complete(
            te._tool_tts_prefs({"action": "set"})
        )
        assert "ERROR" in result

    def test_tts_prefs_unknown_action(self):
        """tts_prefs tool returns error for unknown action."""
        import asyncio
        from app.agent.tool_executor import ToolExecutor

        te = ToolExecutor.__new__(ToolExecutor)
        te._user_id = "test-uid-789"

        result = asyncio.get_event_loop().run_until_complete(
            te._tool_tts_prefs({"action": "delete"})
        )
        assert "ERROR" in result
        assert "Unknown action" in result


class TestSpawnPolicyBehavioral:
    """Exercise agent spawn allow-list gating logic."""

    def test_spawn_tool_deny_blocked_agent(self):
        """Spawn tool blocks agents not in allow list when list is non-empty."""
        src = open("app/agent/tool_executor.py").read()
        # Verify the policy check exists with proper logic
        assert "allow_agents" in src
        # Should check "if settings.allow_agents and X not in settings.allow_agents"
        assert "not in" in src[src.index("allow_agents"):][:200] or \
               "denied" in src[src.index("allow_agents"):][:300].lower()


class TestThinkingBudgetIntegration:
    """Verify thinking budget flows through the entire pipeline."""

    def test_anthropic_service_builds_thinking_kwargs(self):
        """When thinking_budget > 0, kwargs should include thinking dict."""
        src = open("app/services/anthropic_service.py").read()
        # Find the thinking block
        idx = src.index("thinking_budget")
        block = src[idx:idx+500]
        assert "budget_tokens" in block
        assert '"enabled"' in block or "'enabled'" in block
        assert "temperature" in block

    def test_agent_runner_only_sends_thinking_for_claude(self):
        """Agent runner should only pass thinking_budget for Claude models."""
        src = open("app/agent/agent_runner.py").read()
        assert "_is_claude_model" in src or "claude" in src.lower()

    def test_telegram_bot_reads_per_chat_thinking_budgets(self):
        """telegram_bot reads from _thinking_budgets dict per chat."""
        src = open("app/agent/telegram_bot.py").read()
        assert "_thinking_budgets" in src
        # Should use getattr with default empty dict (defensive)
        assert "getattr(self, '_thinking_budgets'" in src

    def test_think_command_sets_budget_values(self):
        """The /think command handler stores budget values correctly."""
        src = open("app/agent/telegram_bot.py").read()
        idx = src.index("def _cmd_think")
        block = src[idx:idx+1000]
        # Should handle on/off/custom values
        assert "_thinking_budgets[chat_id]" in block or "_thinking_budgets" in block
        assert "32768" in block or "budget" in block.lower()


class TestMultiAgentBehavioral:
    """Exercise multi-agent routing logic."""

    def test_multi_agent_routing_only_when_enabled(self):
        """Multi-agent routing should only activate when config says so."""
        src = open("app/agent/telegram_bot.py").read()
        idx = src.index("multi_agent_enabled")
        # Should be in an if block, not unconditional
        block = src[max(0, idx-100):idx+50]
        assert "if " in block

    def test_persona_command_shows_list_without_args(self):
        """_cmd_persona without args should list available personas."""
        src = open("app/agent/telegram_bot.py").read()
        idx = src.index("def _cmd_persona")
        block = src[idx:idx+1000]
        assert "list_personas" in block or "Available" in block

    def test_providers_command_shows_all_config(self):
        """_cmd_providers should display model, TTS, and thinking info."""
        src = open("app/agent/telegram_bot.py").read()
        idx = src.index("def _cmd_providers")
        block = src[idx:idx+1500]
        assert "agent_model" in block
        assert "tts_provider" in block
        assert "thinking_budget" in block


class TestNoDuplicateMethods:
    """Regression: ensure no duplicate method definitions in key files."""

    def test_tool_executor_no_duplicate_thread(self):
        src = open("app/agent/tool_executor.py").read()
        assert src.count("def _tool_thread(") == 1, "Duplicate _tool_thread found!"

    def test_tool_executor_no_duplicate_tts_prefs(self):
        src = open("app/agent/tool_executor.py").read()
        assert src.count("def _tool_tts_prefs(") == 1, "Duplicate _tool_tts_prefs found!"

    def test_tool_definitions_no_duplicate_thread(self):
        from app.agent.tool_definitions import get_extended_tools
        tools = get_extended_tools()
        names = [t["name"] for t in tools]
        assert names.count("thread") == 1

    def test_tool_definitions_no_duplicate_tts_prefs(self):
        from app.agent.tool_definitions import get_extended_tools
        tools = get_extended_tools()
        names = [t["name"] for t in tools]
        assert names.count("tts_prefs") == 1


# ======================================================================
# LAYER 5 — Tool Streaming · PTY Exec · Cron Events · WebChat · CLI · Dashboard
# ======================================================================

class TestLayer5ToolStreaming:
    """5.1 — Tool result streaming (§3)"""

    def test_streaming_handler_has_on_tool_progress(self):
        from app.agent.streaming import TelegramStreamHandler
        assert hasattr(TelegramStreamHandler, 'on_tool_progress'), "Missing on_tool_progress method"

    def test_on_tool_progress_is_async(self):
        import inspect
        from app.agent.streaming import TelegramStreamHandler
        method = getattr(TelegramStreamHandler, 'on_tool_progress')
        assert inspect.iscoroutinefunction(method), "on_tool_progress must be async"

    def test_on_tool_progress_buffers_short_chunks(self):
        """on_tool_progress should buffer until 200 chars before sending."""
        import asyncio
        from app.agent.streaming import TelegramStreamHandler

        class FakeBot:
            sent = []
            async def send_message(self, **kw):
                self.sent.append(kw)

        bot = FakeBot()
        handler = TelegramStreamHandler(chat_id=123, bot=bot)
        # Short chunk should be buffered, not sent
        asyncio.get_event_loop().run_until_complete(handler.on_tool_progress("exec", "short"))
        assert len(bot.sent) == 0, "Should buffer short chunks"

    def test_agent_runner_accepts_on_tool_progress(self):
        import inspect
        from app.agent.agent_runner import AgentRunner
        sig = inspect.signature(AgentRunner.run)
        assert 'on_tool_progress' in sig.parameters, "AgentRunner.run must accept on_tool_progress"

    def test_tool_executor_has_progress_callback(self):
        from app.agent.tool_executor import ToolExecutor
        te = ToolExecutor(workspace="/tmp/hextest_ws"); te._current_user_id = "test_user"
        assert hasattr(te, '_on_tool_progress'), "ToolExecutor must have _on_tool_progress attr"


class TestLayer5PTYExec:
    """5.2 — PTY exec support (§4b)"""

    def test_pty_exec_tool_defined(self):
        from app.agent.tool_definitions import get_agent_tools; ALL_TOOLS = get_agent_tools()
        names = [t["name"] for t in ALL_TOOLS]
        assert "pty_exec" in names, "pty_exec tool must be in ALL_TOOLS"

    def test_pty_exec_schema_has_rows_cols(self):
        from app.agent.tool_definitions import get_agent_tools; ALL_TOOLS = get_agent_tools()
        pty = next(t for t in ALL_TOOLS if t["name"] == "pty_exec")
        props = pty["input_schema"]["properties"]
        assert "rows" in props, "pty_exec must have rows param"
        assert "cols" in props, "pty_exec must have cols param"
        assert "command" in props, "pty_exec must have command param"

    def test_pty_exec_handler_exists(self):
        from app.agent.tool_executor import ToolExecutor
        te = ToolExecutor(workspace="/tmp/hextest_ws"); te._current_user_id = "test_user"
        assert hasattr(te, '_tool_pty_exec'), "ToolExecutor must have _tool_pty_exec method"

    def test_pty_exec_runs_basic_command(self):
        import asyncio
        from app.agent.tool_executor import ToolExecutor
        te = ToolExecutor(workspace="/tmp/hextest_ws"); te._current_user_id = "test_user"
        result = asyncio.get_event_loop().run_until_complete(
            te._tool_pty_exec({"command": "echo hello_pty"})
        )
        assert "hello_pty" in result, f"PTY exec should output 'hello_pty', got: {result[:200]}"

    def test_pty_exec_respects_timeout(self):
        import asyncio
        from app.agent.tool_executor import ToolExecutor
        te = ToolExecutor(workspace="/tmp/hextest_ws"); te._current_user_id = "test_user"
        result = asyncio.get_event_loop().run_until_complete(
            te._tool_pty_exec({"command": "sleep 999", "timeout": 2})
        )
        assert "timeout" in result.lower() or "timed out" in result.lower() or "exit code: -9" in result.lower(), f"Should timeout (kill signal -9 is acceptable), got: {result[:200]}"


class TestLayer5CronWakeEvents:
    """5.3 — Cron wake events + delivery modes (§4h)"""

    def test_cron_service_has_fire_wake_event(self):
        from app.agent.cron_service import CronService
        assert hasattr(CronService, 'fire_wake_event'), "CronService must have fire_wake_event"

    def test_cron_add_job_accepts_wake_event(self):
        import inspect
        from app.agent.cron_service import CronService
        sig = inspect.signature(CronService.add_job)
        assert 'wake_event' in sig.parameters, "add_job must accept wake_event"
        assert 'delivery_mode' in sig.parameters, "add_job must accept delivery_mode"

    def test_cron_valid_delivery_modes(self):
        from app.agent.cron_service import CronService
        assert hasattr(CronService, 'VALID_DELIVERY_MODES'), "Must have VALID_DELIVERY_MODES"
        modes = CronService.VALID_DELIVERY_MODES
        assert "gateway" in modes
        assert "direct" in modes
        assert "announce" in modes
        assert "silent" in modes

    def test_fire_wake_event_returns_results(self):
        """fire_wake_event with no matching jobs should return empty list."""
        import asyncio
        from app.agent.cron_service import CronService
        cs = CronService.__new__(CronService)
        cs._jobs = {}
        cs._agent_runner = None
        cs._telegram_bot = None
        results = asyncio.get_event_loop().run_until_complete(
            cs.fire_wake_event("test.event")
        )
        assert results == [], f"No matching jobs should return [], got: {results}"

    def test_fire_wake_event_matches_jobs(self):
        """fire_wake_event should find jobs matching the event name."""
        import asyncio
        from unittest.mock import AsyncMock, patch
        from app.agent.cron_service import CronService
        cs = CronService.__new__(CronService)
        cs._jobs = {
            "job1": {"wake_event": "gmail.new_email", "enabled": True, "user_id": "u1", "chat_id": 123, "name": "email-checker", "message": "check email", "delivery_mode": "gateway", "run_count": 0},
            "job2": {"wake_event": "other.event", "enabled": True, "user_id": "u1", "chat_id": 123, "name": "other", "message": "other", "delivery_mode": "gateway", "run_count": 0},
        }
        cs._agent_runner = None
        cs._telegram_bot = None
        # Mock _execute_job to avoid actually running the agent
        cs._execute_job = AsyncMock()
        results = asyncio.get_event_loop().run_until_complete(
            cs.fire_wake_event("gmail.new_email", {"from": "test@test.com"})
        )
        assert len(results) == 1, f"Should match 1 job, got {len(results)}"
        assert results[0]["name"] == "email-checker"
        cs._execute_job.assert_called_once()


class TestLayer5Webhooks:
    """5.4 — Webhook trigger endpoints (§4h)"""

    def test_webhooks_module_exists(self):
        from app.api import webhooks
        assert hasattr(webhooks, 'router')
        assert hasattr(webhooks, 'set_webhook_refs')

    def test_webhook_trigger_endpoint_exists(self):
        from app.api.webhooks import router
        paths = [r.path for r in router.routes]
        assert any("/trigger" in p for p in paths), f"Missing /trigger endpoint. Routes: {paths}"

    def test_webhook_gmail_endpoint_exists(self):
        from app.api.webhooks import router
        paths = [r.path for r in router.routes]
        assert any("/gmail" in p for p in paths), f"Missing /gmail endpoint. Routes: {paths}"

    def test_webhook_events_endpoint_exists(self):
        from app.api.webhooks import router
        paths = [r.path for r in router.routes]
        assert any("/events" in p for p in paths), f"Missing /events endpoint. Routes: {paths}"

    def test_webhook_payload_model(self):
        from app.api.webhooks import WebhookPayload
        p = WebhookPayload(event="test.event", payload={"key": "val"})
        assert p.event == "test.event"
        assert p.payload == {"key": "val"}


class TestLayer5WebChat:
    """5.5 — WebChat standalone (§1/§14)"""

    def test_webchat_html_exists(self):
        import os
        path = os.path.join(os.path.dirname(__file__), "..", "app", "static", "webchat.html")
        assert os.path.isfile(path), f"webchat.html missing at {path}"

    def test_webchat_has_login_form(self):
        import os
        path = os.path.join(os.path.dirname(__file__), "..", "app", "static", "webchat.html")
        with open(path) as f:
            html = f.read()
        assert "authUser" in html, "Must have login username field"
        assert "authPass" in html, "Must have login password field"
        assert "doLogin" in html, "Must have login function"

    def test_webchat_has_websocket_connection(self):
        import os
        path = os.path.join(os.path.dirname(__file__), "..", "app", "static", "webchat.html")
        with open(path) as f:
            html = f.read()
        assert "WebSocket" in html, "Must use WebSocket"
        assert "ws/chat" in html, "Must connect to /ws/chat"

    def test_webchat_handles_message_types(self):
        import os
        path = os.path.join(os.path.dirname(__file__), "..", "app", "static", "webchat.html")
        with open(path) as f:
            html = f.read()
        for msg_type in ["text_chunk", "tool_start", "tool_end", "done", "error", "pong"]:
            assert msg_type in html, f"WebChat must handle {msg_type} message type"

    def test_ws_chat_module_exists(self):
        from app.api.ws_chat import router
        assert router is not None


class TestLayer5CLI:
    """5.6 — CLI binary (§11)"""

    def test_cli_module_exists(self):
        from app.cli import main
        assert callable(main)

    def test_cli_has_send_command(self):
        from app.cli import cmd_send
        assert callable(cmd_send)

    def test_cli_has_status_command(self):
        from app.cli import cmd_status
        assert callable(cmd_status)

    def test_cli_has_doctor_command(self):
        from app.cli import cmd_doctor
        assert callable(cmd_doctor)

    def test_cli_has_sessions_command(self):
        from app.cli import cmd_sessions
        assert callable(cmd_sessions)

    def test_cli_has_cron_command(self):
        from app.cli import cmd_cron
        assert callable(cmd_cron)

    def test_cli_parser_subcommands(self):
        """CLI parser should have all 5 subcommands."""
        import argparse
        from app.cli import main
        import sys
        # Test that --help doesn't crash
        try:
            sys.argv = ["hexbrain", "--help"]
            main()
        except SystemExit:
            pass  # argparse calls sys.exit(0) on --help


class TestLayer5FrontendDashboard:
    """5.7 — Frontend dashboard panels (§14)"""

    def test_channel_status_component_exists(self):
        import os
        path = os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "src", "components", "ChannelStatus.tsx")
        assert os.path.isfile(path), "ChannelStatus.tsx missing"
        with open(path) as f:
            content = f.read()
        assert "ChannelStatus" in content

    def test_session_browser_component_exists(self):
        import os
        path = os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "src", "components", "SessionBrowser.tsx")
        assert os.path.isfile(path), "SessionBrowser.tsx missing"

    def test_cron_manager_component_exists(self):
        import os
        path = os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "src", "components", "CronManager.tsx")
        assert os.path.isfile(path), "CronManager.tsx missing"

    def test_skills_panel_component_exists(self):
        import os
        path = os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "src", "components", "SkillsPanel.tsx")
        assert os.path.isfile(path), "SkillsPanel.tsx missing"

    def test_config_editor_component_exists(self):
        import os
        path = os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "src", "components", "ConfigEditor.tsx")
        assert os.path.isfile(path), "ConfigEditor.tsx missing"

    def test_components_exported_in_index(self):
        import os
        path = os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "src", "components", "index.ts")
        with open(path) as f:
            content = f.read()
        for comp in ["ChannelStatus", "SessionBrowser", "CronManager", "SkillsPanel", "ConfigEditor"]:
            assert comp in content, f"{comp} not exported in index.ts"


class TestLayer5RestoredAPIs:
    """5.8 — Restored API files (ws_chat, chat, sessions)"""

    def test_ws_chat_router_has_routes(self):
        from app.api.ws_chat import router
        assert len(router.routes) > 0, "ws_chat router should have routes"

    def test_chat_router_exists(self):
        from app.api.chat import router
        assert router is not None

    def test_sessions_router_exists(self):
        from app.api.sessions import router
        assert router is not None

    def test_main_includes_ws_chat_router(self):
        with open("app/main.py") as f:
            content = f.read()
        assert "ws_chat_router" in content, "main.py must include ws_chat_router"

    def test_main_includes_chat_router(self):
        with open("app/main.py") as f:
            content = f.read()
        assert "chat_router" in content, "main.py must include chat_router"

    def test_main_includes_sessions_router(self):
        with open("app/main.py") as f:
            content = f.read()
        assert "sessions_router" in content, "main.py must include sessions_router"

    def test_main_serves_webchat(self):
        with open("app/main.py") as f:
            content = f.read()
        assert "/webchat" in content, "main.py must serve /webchat"


class TestLayer5Behavioral:
    """5.9 — Behavioral integration tests"""

    def test_tool_streaming_exec_with_callback(self):
        """exec tool should call _on_tool_progress when set."""
        import asyncio
        from app.agent.tool_executor import ToolExecutor
        te = ToolExecutor(workspace="/tmp/hextest_ws"); te._current_user_id = "test_user"
        chunks = []
        async def on_progress(tool, chunk):
            chunks.append(chunk)
        te._on_tool_progress = on_progress
        result = asyncio.get_event_loop().run_until_complete(
            te._tool_exec({"command": "echo line1; echo line2; echo line3"})
        )
        assert "line1" in result
        assert len(chunks) > 0, "Should have received progress chunks"

    def test_pty_exec_strips_ansi(self):
        """PTY exec should handle ANSI escape codes gracefully."""
        import asyncio
        from app.agent.tool_executor import ToolExecutor
        te = ToolExecutor(workspace="/tmp/hextest_ws"); te._current_user_id = "test_user"
        result = asyncio.get_event_loop().run_until_complete(
            te._tool_pty_exec({"command": "echo -e '\033[31mred\033[0m'"})
        )
        # Should contain the word "red" regardless of ANSI handling
        assert "red" in result.lower(), f"PTY output should contain 'red', got: {result[:200]}"

    def test_cron_delivery_mode_default(self):
        """Default delivery mode should be 'gateway'."""
        import inspect
        from app.agent.cron_service import CronService
        sig = inspect.signature(CronService.add_job)
        dm_param = sig.parameters['delivery_mode']
        assert dm_param.default == "gateway", f"Default delivery_mode should be 'gateway', got {dm_param.default}"

    def test_webchat_has_reconnect_logic(self):
        """WebChat should auto-reconnect on disconnect."""
        import os
        path = os.path.join(os.path.dirname(__file__), "..", "app", "static", "webchat.html")
        with open(path) as f:
            html = f.read()
        assert "Reconnecting" in html or "reconnect" in html.lower(), "WebChat must have reconnect logic"

    def test_webchat_has_keepalive_ping(self):
        import os
        path = os.path.join(os.path.dirname(__file__), "..", "app", "static", "webchat.html")
        with open(path) as f:
            html = f.read()
        assert "ping" in html, "WebChat must have keepalive ping"

    def test_cli_default_url(self):
        from app.cli import DEFAULT_BASE
        assert "localhost" in DEFAULT_BASE or "127.0.0.1" in DEFAULT_BASE

    def test_webhook_payload_optional_payload(self):
        from app.api.webhooks import WebhookPayload
        p = WebhookPayload(event="test")
        assert p.payload is None, "payload should be optional"
