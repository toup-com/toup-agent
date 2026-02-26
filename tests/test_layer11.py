"""
Layer 11 Tests — Thread Manager, Polls, Tool Elevation,
Plugin Registry, Agent Workspace, Model Providers.
"""

import asyncio
import os
import shutil
import tempfile
import pytest

# ── Thread Manager ────────────────────────────────────

class TestThreadManager:
    def test_import(self):
        from app.agent.thread_manager import ThreadManager, get_thread_manager
        assert ThreadManager is not None

    def test_create_thread(self):
        from app.agent.thread_manager import ThreadManager
        mgr = ThreadManager()
        thread = asyncio.get_event_loop().run_until_complete(
            mgr.create_thread("discord", "ch1", "Bug Report")
        )
        assert thread.title == "Bug Report"
        assert thread.channel_type == "discord"

    def test_reply_to_thread(self):
        from app.agent.thread_manager import ThreadManager
        mgr = ThreadManager()
        loop = asyncio.get_event_loop()
        thread = loop.run_until_complete(
            mgr.create_thread("slack", "ch1", "Discussion")
        )
        msg = loop.run_until_complete(
            mgr.reply_to_thread(thread.thread_id, "Hello!", sender_id="user1")
        )
        assert msg.content == "Hello!"
        assert thread.message_count == 1

    def test_reply_unknown_thread(self):
        from app.agent.thread_manager import ThreadManager
        mgr = ThreadManager()
        with pytest.raises(ValueError, match="not found"):
            asyncio.get_event_loop().run_until_complete(
                mgr.reply_to_thread("nonexistent", "Hello!")
            )

    def test_list_threads(self):
        from app.agent.thread_manager import ThreadManager
        mgr = ThreadManager()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(mgr.create_thread("discord", "ch1", "T1"))
        loop.run_until_complete(mgr.create_thread("slack", "ch2", "T2"))
        loop.run_until_complete(mgr.create_thread("discord", "ch1", "T3"))

        all_threads = loop.run_until_complete(mgr.list_threads())
        assert len(all_threads) == 3

        discord_threads = loop.run_until_complete(mgr.list_threads("discord"))
        assert len(discord_threads) == 2

    def test_get_thread_messages(self):
        from app.agent.thread_manager import ThreadManager
        mgr = ThreadManager()
        loop = asyncio.get_event_loop()
        thread = loop.run_until_complete(mgr.create_thread("telegram", "ch1", "Topic"))
        loop.run_until_complete(mgr.reply_to_thread(thread.thread_id, "msg1"))
        loop.run_until_complete(mgr.reply_to_thread(thread.thread_id, "msg2"))

        msgs = loop.run_until_complete(mgr.get_thread_messages(thread.thread_id))
        assert len(msgs) == 2

    def test_archive_thread(self):
        from app.agent.thread_manager import ThreadManager, ThreadState
        mgr = ThreadManager()
        loop = asyncio.get_event_loop()
        thread = loop.run_until_complete(mgr.create_thread("discord", "ch1", "Old"))
        result = loop.run_until_complete(mgr.archive_thread(thread.thread_id))
        assert result == True
        assert thread.state == ThreadState.ARCHIVED

    def test_lock_thread(self):
        from app.agent.thread_manager import ThreadManager, ThreadState
        mgr = ThreadManager()
        loop = asyncio.get_event_loop()
        thread = loop.run_until_complete(mgr.create_thread("discord", "ch1", "Locked"))
        loop.run_until_complete(mgr.lock_thread(thread.thread_id))
        assert thread.state == ThreadState.LOCKED

        # Cannot reply to locked thread
        with pytest.raises(ValueError, match="locked"):
            loop.run_until_complete(
                mgr.reply_to_thread(thread.thread_id, "can't post")
            )

    def test_delete_thread(self):
        from app.agent.thread_manager import ThreadManager
        mgr = ThreadManager()
        loop = asyncio.get_event_loop()
        thread = loop.run_until_complete(mgr.create_thread("discord", "ch1", "Delete Me"))
        result = loop.run_until_complete(mgr.delete_thread(thread.thread_id))
        assert result == True
        assert mgr.get_thread(thread.thread_id) is None

    def test_register_handler(self):
        from app.agent.thread_manager import ThreadManager
        mgr = ThreadManager()
        called = []

        async def mock_create(**kwargs):
            called.append(kwargs)
            return {"thread_id": "custom_123"}

        mgr.register_handler("discord", "create", mock_create)
        loop = asyncio.get_event_loop()
        thread = loop.run_until_complete(mgr.create_thread("discord", "ch1", "With Handler"))
        assert len(called) == 1
        assert thread.thread_id == "custom_123"

    def test_stats(self):
        from app.agent.thread_manager import ThreadManager
        mgr = ThreadManager()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(mgr.create_thread("discord", "ch1", "T1"))
        s = mgr.stats()
        assert s["total_threads"] == 1

    def test_thread_to_dict(self):
        from app.agent.thread_manager import ThreadManager
        mgr = ThreadManager()
        loop = asyncio.get_event_loop()
        thread = loop.run_until_complete(mgr.create_thread("slack", "ch1", "Test"))
        d = thread.to_dict()
        assert d["title"] == "Test"

    def test_singleton(self):
        from app.agent.thread_manager import get_thread_manager
        m1 = get_thread_manager()
        m2 = get_thread_manager()
        assert m1 is m2


# ── Polls ─────────────────────────────────────────────

class TestPolls:
    def test_import(self):
        from app.agent.polls import PollManager, get_poll_manager
        assert PollManager is not None

    def test_create_poll(self):
        from app.agent.polls import PollManager
        mgr = PollManager()
        poll = mgr.create_poll(
            question="Favorite color?",
            options=["Red", "Blue", "Green"],
            channel_type="telegram",
            channel_id="123",
        )
        assert poll.question == "Favorite color?"
        assert len(poll.options) == 3

    def test_create_poll_too_few_options(self):
        from app.agent.polls import PollManager
        mgr = PollManager()
        with pytest.raises(ValueError, match="at least 2"):
            mgr.create_poll("Q?", ["Only one"], "telegram", "123")

    def test_create_poll_too_many_options(self):
        from app.agent.polls import PollManager
        mgr = PollManager()
        with pytest.raises(ValueError, match="at most 10"):
            mgr.create_poll("Q?", [f"opt{i}" for i in range(11)], "telegram", "123")

    def test_cast_vote(self):
        from app.agent.polls import PollManager
        mgr = PollManager()
        poll = mgr.create_poll("Q?", ["A", "B"], "telegram", "123")
        assert mgr.cast_vote(poll.poll_id, "user1", 0) == True
        assert poll.options[0].vote_count == 1

    def test_single_choice_replaces_vote(self):
        from app.agent.polls import PollManager
        mgr = PollManager()
        poll = mgr.create_poll("Q?", ["A", "B", "C"], "telegram", "123")
        mgr.cast_vote(poll.poll_id, "user1", 0)
        mgr.cast_vote(poll.poll_id, "user1", 2)  # Changes vote
        assert poll.options[0].vote_count == 0
        assert poll.options[2].vote_count == 1

    def test_invalid_option(self):
        from app.agent.polls import PollManager
        mgr = PollManager()
        poll = mgr.create_poll("Q?", ["A", "B"], "telegram", "123")
        with pytest.raises(ValueError, match="Invalid option"):
            mgr.cast_vote(poll.poll_id, "user1", 5)

    def test_vote_on_closed_poll(self):
        from app.agent.polls import PollManager
        mgr = PollManager()
        poll = mgr.create_poll("Q?", ["A", "B"], "telegram", "123")
        mgr.close_poll(poll.poll_id)
        with pytest.raises(ValueError, match="closed"):
            mgr.cast_vote(poll.poll_id, "user1", 0)

    def test_close_poll_results(self):
        from app.agent.polls import PollManager
        mgr = PollManager()
        poll = mgr.create_poll("Q?", ["A", "B"], "telegram", "123")
        mgr.cast_vote(poll.poll_id, "u1", 0)
        mgr.cast_vote(poll.poll_id, "u2", 0)
        mgr.cast_vote(poll.poll_id, "u3", 1)
        results = mgr.close_poll(poll.poll_id)
        assert results["total_votes"] == 3

    def test_retract_vote(self):
        from app.agent.polls import PollManager
        mgr = PollManager()
        poll = mgr.create_poll("Q?", ["A", "B"], "telegram", "123")
        mgr.cast_vote(poll.poll_id, "u1", 0)
        mgr.retract_vote(poll.poll_id, "u1", 0)
        assert poll.options[0].vote_count == 0

    def test_delete_poll(self):
        from app.agent.polls import PollManager
        mgr = PollManager()
        poll = mgr.create_poll("Q?", ["A", "B"], "telegram", "123")
        assert mgr.delete_poll(poll.poll_id) == True
        assert mgr.get_poll(poll.poll_id) is None

    def test_list_polls(self):
        from app.agent.polls import PollManager
        mgr = PollManager()
        mgr.create_poll("Q1?", ["A", "B"], "telegram", "123")
        mgr.create_poll("Q2?", ["X", "Y"], "discord", "456")
        all_polls = mgr.list_polls()
        assert len(all_polls) == 2

        tg_polls = mgr.list_polls(channel_type="telegram")
        assert len(tg_polls) == 1

    def test_poll_results(self):
        from app.agent.polls import PollManager
        mgr = PollManager()
        poll = mgr.create_poll("Q?", ["A", "B"], "telegram", "123")
        mgr.cast_vote(poll.poll_id, "u1", 0)
        results = mgr.get_results(poll.poll_id)
        assert results["total_votes"] == 1

    def test_stats(self):
        from app.agent.polls import PollManager
        mgr = PollManager()
        mgr.create_poll("Q?", ["A", "B"], "telegram", "123")
        s = mgr.stats()
        assert s["total_polls"] == 1
        assert s["active"] == 1

    def test_singleton(self):
        from app.agent.polls import get_poll_manager
        m1 = get_poll_manager()
        m2 = get_poll_manager()
        assert m1 is m2


# ── Tool Elevation ────────────────────────────────────

class TestToolElevation:
    def test_import(self):
        from app.agent.tool_elevation import ToolElevationManager, get_elevation_manager
        assert ToolElevationManager is not None

    def test_default_elevated(self):
        from app.agent.tool_elevation import ToolElevationManager
        mgr = ToolElevationManager()
        assert mgr.requires_confirmation("exec")
        assert mgr.requires_confirmation("apply_patch")
        assert not mgr.requires_confirmation("web_search")

    def test_mark_elevated(self):
        from app.agent.tool_elevation import ToolElevationManager
        mgr = ToolElevationManager()
        mgr.mark_elevated("custom_tool", reason="Dangerous")
        assert mgr.requires_confirmation("custom_tool")

    def test_mark_normal(self):
        from app.agent.tool_elevation import ToolElevationManager
        mgr = ToolElevationManager()
        mgr.mark_elevated("tool1")
        assert mgr.mark_normal("tool1") == True
        assert not mgr.requires_confirmation("tool1")

    def test_blocked(self):
        from app.agent.tool_elevation import ToolElevationManager, ElevationLevel
        mgr = ToolElevationManager()
        mgr.mark_elevated("danger", level=ElevationLevel.BLOCKED)
        assert mgr.is_blocked("danger") == True

    def test_approval_flow(self):
        from app.agent.tool_elevation import ToolElevationManager
        mgr = ToolElevationManager()
        mgr.mark_elevated("tool1")
        req = mgr.request_approval("tool1", "session1", arguments_summary="rm -rf")
        assert req.status.value == "pending"
        assert mgr.approve(req.request_id) == True

    def test_session_approval(self):
        from app.agent.tool_elevation import ToolElevationManager
        mgr = ToolElevationManager()
        mgr.mark_elevated("tool1")
        req = mgr.request_approval("tool1", "s1")
        mgr.approve(req.request_id, grant_session=True)

        # After approval, session has standing permission
        assert not mgr.requires_confirmation("tool1", session_id="s1")
        # Other sessions still need approval
        assert mgr.requires_confirmation("tool1", session_id="s2")

    def test_deny(self):
        from app.agent.tool_elevation import ToolElevationManager, ApprovalStatus
        mgr = ToolElevationManager()
        mgr.mark_elevated("tool1")
        req = mgr.request_approval("tool1", "s1")
        assert mgr.deny(req.request_id) == True
        assert req.status == ApprovalStatus.DENIED

    def test_revoke_session(self):
        from app.agent.tool_elevation import ToolElevationManager
        mgr = ToolElevationManager()
        mgr.mark_elevated("tool1")
        req = mgr.request_approval("tool1", "s1")
        mgr.approve(req.request_id)
        assert not mgr.requires_confirmation("tool1", session_id="s1")

        mgr.revoke_session_approval("s1")
        assert mgr.requires_confirmation("tool1", session_id="s1")

    def test_list_elevated(self):
        from app.agent.tool_elevation import ToolElevationManager
        mgr = ToolElevationManager()
        items = mgr.list_elevated()
        assert len(items) >= 3  # 3 defaults

    def test_list_pending(self):
        from app.agent.tool_elevation import ToolElevationManager
        mgr = ToolElevationManager()
        mgr.mark_elevated("t1")
        mgr.request_approval("t1", "s1")
        pending = mgr.list_pending()
        assert len(pending) == 1

    def test_stats(self):
        from app.agent.tool_elevation import ToolElevationManager
        mgr = ToolElevationManager()
        s = mgr.stats()
        assert s["elevated_tools"] >= 3

    def test_singleton(self):
        from app.agent.tool_elevation import get_elevation_manager
        m1 = get_elevation_manager()
        m2 = get_elevation_manager()
        assert m1 is m2


# ── Plugin Registry ───────────────────────────────────

class TestPluginRegistry:
    def test_import(self):
        from app.agent.plugin_registry import PluginRegistry, get_plugin_registry
        assert PluginRegistry is not None

    def test_register_plugin(self):
        from app.agent.plugin_registry import PluginRegistry
        reg = PluginRegistry()
        info = reg.register_plugin("my_plugin", version="2.0.0")
        assert info.name == "my_plugin"
        assert info.version == "2.0.0"

    def test_register_tool(self):
        from app.agent.plugin_registry import PluginRegistry, ToolDefinition
        reg = PluginRegistry()
        reg.register_plugin("p1")
        tool = reg.register_tool("p1", ToolDefinition(name="my_tool", description="A tool"))
        assert tool.plugin_name == "p1"
        assert reg.get_tool("my_tool") is not None

    def test_register_command(self):
        from app.agent.plugin_registry import PluginRegistry, CommandDefinition
        reg = PluginRegistry()
        cmd = reg.register_command("p1", CommandDefinition(name="hello", description="Say hi"))
        assert cmd.plugin_name == "p1"
        assert reg.get_command("hello") is not None

    def test_register_channel(self):
        from app.agent.plugin_registry import PluginRegistry, ChannelDefinition
        reg = PluginRegistry()
        ch = reg.register_channel("p1", ChannelDefinition(name="custom_chan", description="Custom"))
        assert reg.get_channel("custom_chan") is not None

    def test_unregister_plugin_removes_all(self):
        from app.agent.plugin_registry import PluginRegistry, ToolDefinition, CommandDefinition
        reg = PluginRegistry()
        reg.register_plugin("p1")
        reg.register_tool("p1", ToolDefinition(name="t1", description="T"))
        reg.register_command("p1", CommandDefinition(name="c1", description="C"))
        assert reg.unregister_plugin("p1") == True
        assert reg.get_tool("t1") is None
        assert reg.get_command("c1") is None

    def test_list_tools(self):
        from app.agent.plugin_registry import PluginRegistry, ToolDefinition
        reg = PluginRegistry()
        reg.register_tool("p1", ToolDefinition(name="t1", description="T1"))
        reg.register_tool("p2", ToolDefinition(name="t2", description="T2"))
        all_tools = reg.list_tools()
        assert len(all_tools) == 2

        p1_tools = reg.list_tools(plugin_name="p1")
        assert len(p1_tools) == 1

    def test_list_commands(self):
        from app.agent.plugin_registry import PluginRegistry, CommandDefinition
        reg = PluginRegistry()
        reg.register_command("p1", CommandDefinition(name="c1", description="C"))
        cmds = reg.list_commands()
        assert len(cmds) == 1

    def test_list_plugins(self):
        from app.agent.plugin_registry import PluginRegistry
        reg = PluginRegistry()
        reg.register_plugin("p1")
        reg.register_plugin("p2")
        assert len(reg.list_plugins()) == 2

    def test_disable_enable(self):
        from app.agent.plugin_registry import PluginRegistry, PluginState
        reg = PluginRegistry()
        reg.register_plugin("p1")
        assert reg.disable_plugin("p1") == True
        assert reg.get_plugin("p1").state == PluginState.DISABLED
        assert reg.enable_plugin("p1") == True
        assert reg.get_plugin("p1").state == PluginState.ACTIVE

    def test_auto_register_plugin(self):
        from app.agent.plugin_registry import PluginRegistry, ToolDefinition
        reg = PluginRegistry()
        # Register tool without registering plugin first
        reg.register_tool("auto_plugin", ToolDefinition(name="t1", description="T"))
        assert reg.get_plugin("auto_plugin") is not None

    def test_stats(self):
        from app.agent.plugin_registry import PluginRegistry, ToolDefinition
        reg = PluginRegistry()
        reg.register_plugin("p1")
        reg.register_tool("p1", ToolDefinition(name="t1", description="T"))
        s = reg.stats()
        assert s["total_plugins"] == 1
        assert s["total_tools"] == 1

    def test_singleton(self):
        from app.agent.plugin_registry import get_plugin_registry
        r1 = get_plugin_registry()
        r2 = get_plugin_registry()
        assert r1 is r2


# ── Agent Workspace ───────────────────────────────────

class TestAgentWorkspace:
    def setup_method(self):
        self.tmp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_import(self):
        from app.agent.agent_workspace import WorkspaceManager, AgentWorkspace
        assert WorkspaceManager is not None

    def test_create_workspace(self):
        from app.agent.agent_workspace import WorkspaceManager
        mgr = WorkspaceManager(base_path=self.tmp_dir)
        ws = mgr.create_workspace("agent-1")
        assert os.path.isdir(ws.root_path)

    def test_write_and_read_file(self):
        from app.agent.agent_workspace import WorkspaceManager
        mgr = WorkspaceManager(base_path=self.tmp_dir)
        ws = mgr.create_workspace("agent-1")
        ws.write_file("notes.md", "# Hello\n")
        content = ws.read_file("notes.md")
        assert content == "# Hello\n"

    def test_file_not_found(self):
        from app.agent.agent_workspace import WorkspaceManager
        mgr = WorkspaceManager(base_path=self.tmp_dir)
        ws = mgr.create_workspace("agent-1")
        with pytest.raises(FileNotFoundError):
            ws.read_file("nonexistent.txt")

    def test_delete_file(self):
        from app.agent.agent_workspace import WorkspaceManager
        mgr = WorkspaceManager(base_path=self.tmp_dir)
        ws = mgr.create_workspace("agent-1")
        ws.write_file("temp.txt", "data")
        assert ws.delete_file("temp.txt") == True
        assert ws.file_exists("temp.txt") == False

    def test_list_files(self):
        from app.agent.agent_workspace import WorkspaceManager
        mgr = WorkspaceManager(base_path=self.tmp_dir)
        ws = mgr.create_workspace("agent-1")
        ws.write_file("a.txt", "a")
        ws.write_file("b.txt", "b")
        ws.write_file("sub/c.txt", "c")
        files = ws.list_files()
        assert len(files) == 3

    def test_path_escape_prevention(self):
        from app.agent.agent_workspace import WorkspaceManager
        mgr = WorkspaceManager(base_path=self.tmp_dir)
        ws = mgr.create_workspace("agent-1")
        with pytest.raises(ValueError, match="escapes"):
            ws.write_file("../../etc/passwd", "hack")

    def test_workspace_isolation(self):
        from app.agent.agent_workspace import WorkspaceManager
        mgr = WorkspaceManager(base_path=self.tmp_dir)
        ws1 = mgr.create_workspace("agent-1")
        ws2 = mgr.create_workspace("agent-2")
        ws1.write_file("private.txt", "agent1 data")
        assert ws2.file_exists("private.txt") == False

    def test_get_workspace(self):
        from app.agent.agent_workspace import WorkspaceManager
        mgr = WorkspaceManager(base_path=self.tmp_dir)
        mgr.create_workspace("agent-1")
        assert mgr.get_workspace("agent-1") is not None
        assert mgr.get_workspace("nonexistent") is None

    def test_destroy_workspace(self):
        from app.agent.agent_workspace import WorkspaceManager
        mgr = WorkspaceManager(base_path=self.tmp_dir)
        ws = mgr.create_workspace("agent-1")
        root = ws.root_path
        assert mgr.destroy_workspace("agent-1") == True
        assert not os.path.exists(root)

    def test_list_workspaces(self):
        from app.agent.agent_workspace import WorkspaceManager
        mgr = WorkspaceManager(base_path=self.tmp_dir)
        mgr.create_workspace("a1")
        mgr.create_workspace("a2")
        ws_list = mgr.list_workspaces()
        assert len(ws_list) == 2

    def test_quota_check(self):
        from app.agent.agent_workspace import WorkspaceManager
        mgr = WorkspaceManager(base_path=self.tmp_dir)
        ws = mgr.create_workspace("agent-1")
        ws.write_file("data.txt", "x" * 100)
        quota = mgr.check_quota("agent-1")
        assert quota["exists"] == True
        assert quota["within_quota"] == True

    def test_workspace_info(self):
        from app.agent.agent_workspace import WorkspaceManager
        mgr = WorkspaceManager(base_path=self.tmp_dir)
        ws = mgr.create_workspace("agent-1")
        ws.write_file("test.txt", "hello")
        info = ws.get_info()
        assert info.agent_id == "agent-1"
        assert info.file_count == 1

    def test_stats(self):
        from app.agent.agent_workspace import WorkspaceManager
        mgr = WorkspaceManager(base_path=self.tmp_dir)
        mgr.create_workspace("a1")
        s = mgr.stats()
        assert s["total_workspaces"] == 1

    def test_idempotent_create(self):
        from app.agent.agent_workspace import WorkspaceManager
        mgr = WorkspaceManager(base_path=self.tmp_dir)
        ws1 = mgr.create_workspace("agent-1")
        ws2 = mgr.create_workspace("agent-1")
        assert ws1 is ws2


# ── Model Providers ───────────────────────────────────

class TestModelProviders:
    def test_import(self):
        from app.agent.model_providers import ModelProviderRegistry, get_provider_registry
        assert ModelProviderRegistry is not None

    def test_builtin_providers(self):
        from app.agent.model_providers import ModelProviderRegistry
        reg = ModelProviderRegistry()
        providers = reg.list_providers()
        names = [p["name"] for p in providers]
        assert "openai" in names
        assert "anthropic" in names

    def test_get_provider_for_model(self):
        from app.agent.model_providers import ModelProviderRegistry
        reg = ModelProviderRegistry()
        p = reg.get_provider_for_model("gpt-4o")
        assert p is not None
        assert p.name == "openai"

    def test_get_provider_for_claude(self):
        from app.agent.model_providers import ModelProviderRegistry
        reg = ModelProviderRegistry()
        p = reg.get_provider_for_model("claude-sonnet-4-20250514")
        assert p is not None
        assert p.name == "anthropic"

    def test_register_custom_provider(self):
        from app.agent.model_providers import ModelProviderRegistry, ProviderConfig, ProviderType
        reg = ModelProviderRegistry()
        reg.register_provider(ProviderConfig(
            name="groq",
            provider_type=ProviderType.OPENAI_COMPATIBLE,
            base_url="https://api.groq.com/openai/v1",
            api_key_env="GROQ_API_KEY",
            models=["llama-3.3-70b"],
        ))
        p = reg.get_provider_for_model("llama-3.3-70b")
        assert p is not None
        assert p.name == "groq"

    def test_resolve_model(self):
        from app.agent.model_providers import ModelProviderRegistry
        reg = ModelProviderRegistry()
        result = reg.resolve_model("gpt-4o")
        assert result["provider"] == "openai"
        assert "base_url" in result

    def test_resolve_unknown_model(self):
        from app.agent.model_providers import ModelProviderRegistry
        reg = ModelProviderRegistry()
        result = reg.resolve_model("unknown-model-xyz")
        assert "error" in result

    def test_list_models(self):
        from app.agent.model_providers import ModelProviderRegistry
        reg = ModelProviderRegistry()
        models = reg.list_models()
        assert "gpt-4o" in models
        assert "claude-sonnet-4-20250514" in models

    def test_list_models_by_provider(self):
        from app.agent.model_providers import ModelProviderRegistry
        reg = ModelProviderRegistry()
        openai_models = reg.list_models("openai")
        assert "gpt-4o" in openai_models
        assert "claude-sonnet-4-20250514" not in openai_models

    def test_add_model(self):
        from app.agent.model_providers import ModelProviderRegistry
        reg = ModelProviderRegistry()
        assert reg.add_model("openai", "gpt-5", max_tokens=16384) == True
        assert "gpt-5" in reg.list_models("openai")

    def test_disable_enable_provider(self):
        from app.agent.model_providers import ModelProviderRegistry
        reg = ModelProviderRegistry()
        assert reg.disable_provider("openai") == True
        result = reg.resolve_model("gpt-4o")
        assert "error" in result  # Provider disabled

        assert reg.enable_provider("openai") == True
        result = reg.resolve_model("gpt-4o")
        assert "error" not in result

    def test_unregister_provider(self):
        from app.agent.model_providers import ModelProviderRegistry
        reg = ModelProviderRegistry()
        assert reg.unregister_provider("anthropic") == True
        assert reg.get_provider_for_model("claude-sonnet-4-20250514") is None

    def test_provider_priority(self):
        from app.agent.model_providers import ModelProviderRegistry
        reg = ModelProviderRegistry()
        providers = reg.list_providers()
        assert providers[0]["name"] == "openai"  # priority 10
        assert providers[1]["name"] == "anthropic"  # priority 20

    def test_stats(self):
        from app.agent.model_providers import ModelProviderRegistry
        reg = ModelProviderRegistry()
        s = reg.stats()
        assert s["total_providers"] >= 2
        assert s["total_models"] >= 5

    def test_singleton(self):
        from app.agent.model_providers import get_provider_registry
        r1 = get_provider_registry()
        r2 = get_provider_registry()
        assert r1 is r2
