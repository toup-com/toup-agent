"""
Layer 14 Tests — Grep Tool, Find Tool, LS Tool, Chrome Relay,
Device Nodes, Voice Call Plugin.
"""

import asyncio
import os
import tempfile
import pytest


# ── Grep Tool ─────────────────────────────────────────

class TestGrepTool:
    def test_import(self):
        from app.agent.grep_tool import GrepTool, GrepResult
        assert GrepTool is not None

    def test_search_plain(self):
        from app.agent.grep_tool import GrepTool
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "test.py"), "w") as f:
                f.write("def hello():\n    print('world')\n")
            tool = GrepTool(workspace_root=tmpdir)
            result = tool.search("hello")
            assert result.total_matches == 1
            assert result.matches[0].line_number == 1

    def test_search_regex(self):
        from app.agent.grep_tool import GrepTool
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "test.py"), "w") as f:
                f.write("TODO: fix this\nFIXME: and this\nOK line\n")
            tool = GrepTool(workspace_root=tmpdir)
            result = tool.search(r"TODO|FIXME", is_regex=True)
            assert result.total_matches == 2

    def test_search_case_insensitive(self):
        from app.agent.grep_tool import GrepTool
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "test.txt"), "w") as f:
                f.write("Hello World\nhello world\n")
            tool = GrepTool(workspace_root=tmpdir)
            result = tool.search("hello", case_sensitive=False)
            assert result.total_matches == 2

    def test_search_file_pattern(self):
        from app.agent.grep_tool import GrepTool
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "test.py"), "w") as f:
                f.write("target\n")
            with open(os.path.join(tmpdir, "test.txt"), "w") as f:
                f.write("target\n")
            tool = GrepTool(workspace_root=tmpdir)
            result = tool.search("target", file_pattern="*.py")
            assert result.total_matches == 1

    def test_search_with_context(self):
        from app.agent.grep_tool import GrepTool
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "test.py"), "w") as f:
                f.write("line1\nline2\ntarget\nline4\nline5\n")
            tool = GrepTool(workspace_root=tmpdir)
            result = tool.search("target", context_lines=1)
            assert len(result.matches[0].context_before) == 1
            assert len(result.matches[0].context_after) == 1

    def test_count(self):
        from app.agent.grep_tool import GrepTool
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "test.py"), "w") as f:
                f.write("a\na\nb\n")
            tool = GrepTool(workspace_root=tmpdir)
            assert tool.count("a") == 2

    def test_result_format(self):
        from app.agent.grep_tool import GrepTool
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "test.py"), "w") as f:
                f.write("target line\n")
            tool = GrepTool(workspace_root=tmpdir)
            result = tool.search("target")
            text = result.format()
            assert "1 matches" in text

    def test_no_matches(self):
        from app.agent.grep_tool import GrepTool
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "test.py"), "w") as f:
                f.write("nothing here\n")
            tool = GrepTool(workspace_root=tmpdir)
            result = tool.search("nonexistent")
            assert result.total_matches == 0

    def test_max_results(self):
        from app.agent.grep_tool import GrepTool
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "test.py"), "w") as f:
                f.write("\n".join(["match"] * 50))
            tool = GrepTool(workspace_root=tmpdir)
            result = tool.search("match", max_results=5)
            assert len(result.matches) == 5
            assert result.truncated == True

    def test_to_dict(self):
        from app.agent.grep_tool import GrepResult
        r = GrepResult(query="test", total_matches=5)
        d = r.to_dict()
        assert d["query"] == "test"


# ── Find Tool ─────────────────────────────────────────

class TestFindTool:
    def test_import(self):
        from app.agent.find_tool import FindTool, FileType
        assert FindTool is not None

    def test_find_files(self):
        from app.agent.find_tool import FindTool
        with tempfile.TemporaryDirectory() as tmpdir:
            open(os.path.join(tmpdir, "a.py"), "w").close()
            open(os.path.join(tmpdir, "b.py"), "w").close()
            open(os.path.join(tmpdir, "c.txt"), "w").close()
            tool = FindTool(workspace_root=tmpdir)
            result = tool.find("*.py")
            assert result.total_found == 2

    def test_find_directories(self):
        from app.agent.find_tool import FindTool, FileType
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "src"))
            os.makedirs(os.path.join(tmpdir, "tests"))
            tool = FindTool(workspace_root=tmpdir)
            result = tool.find("*", file_type=FileType.DIRECTORY)
            assert result.total_found == 2

    def test_find_with_extension(self):
        from app.agent.find_tool import FindTool
        with tempfile.TemporaryDirectory() as tmpdir:
            open(os.path.join(tmpdir, "a.py"), "w").close()
            open(os.path.join(tmpdir, "b.js"), "w").close()
            tool = FindTool(workspace_root=tmpdir)
            result = tool.find("*", extension=".py")
            assert result.total_found == 1

    def test_find_sort_by_name(self):
        from app.agent.find_tool import FindTool, SortBy
        with tempfile.TemporaryDirectory() as tmpdir:
            open(os.path.join(tmpdir, "b.py"), "w").close()
            open(os.path.join(tmpdir, "a.py"), "w").close()
            tool = FindTool(workspace_root=tmpdir)
            result = tool.find("*.py", sort_by=SortBy.NAME)
            assert result.entries[0].name == "a.py"

    def test_find_max_depth(self):
        from app.agent.find_tool import FindTool
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "a", "b", "c"), exist_ok=True)
            open(os.path.join(tmpdir, "a", "top.py"), "w").close()
            open(os.path.join(tmpdir, "a", "b", "c", "deep.py"), "w").close()
            tool = FindTool(workspace_root=tmpdir)
            result = tool.find("*.py", max_depth=2)
            names = [e.name for e in result.entries]
            assert "top.py" in names

    def test_tree(self):
        from app.agent.find_tool import FindTool
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "src"))
            open(os.path.join(tmpdir, "src", "main.py"), "w").close()
            tool = FindTool(workspace_root=tmpdir)
            tree_str = tool.tree(max_depth=2)
            assert "src/" in tree_str
            assert "main.py" in tree_str

    def test_find_by_content(self):
        from app.agent.find_tool import FindTool
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "a.py"), "w") as f:
                f.write("import os\n")
            with open(os.path.join(tmpdir, "b.py"), "w") as f:
                f.write("import sys\n")
            tool = FindTool(workspace_root=tmpdir)
            result = tool.find_by_content("import os")
            assert result.total_found == 1

    def test_result_format(self):
        from app.agent.find_tool import FindResult
        r = FindResult(pattern="*.py", total_found=3)
        text = r.format()
        assert "3 items" in text


# ── LS Tool ───────────────────────────────────────────

class TestLsTool:
    def test_import(self):
        from app.agent.ls_tool import LsTool, LsResult
        assert LsTool is not None

    def test_ls_directory(self):
        from app.agent.ls_tool import LsTool
        with tempfile.TemporaryDirectory() as tmpdir:
            open(os.path.join(tmpdir, "a.py"), "w").close()
            os.makedirs(os.path.join(tmpdir, "src"))
            tool = LsTool(workspace_root=tmpdir)
            result = tool.ls(".")
            assert result.total_files >= 1
            assert result.total_dirs >= 1

    def test_ls_hidden_files(self):
        from app.agent.ls_tool import LsTool
        with tempfile.TemporaryDirectory() as tmpdir:
            open(os.path.join(tmpdir, ".hidden"), "w").close()
            open(os.path.join(tmpdir, "visible"), "w").close()
            tool = LsTool(workspace_root=tmpdir)
            result_no_hidden = tool.ls(".")
            assert all(not e.name.startswith(".") for e in result_no_hidden.entries)
            result_hidden = tool.ls(".", show_hidden=True)
            hidden = [e for e in result_hidden.entries if e.name == ".hidden"]
            assert len(hidden) == 1

    def test_ls_sort_by_size(self):
        from app.agent.ls_tool import LsTool
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "small.txt"), "w") as f:
                f.write("x")
            with open(os.path.join(tmpdir, "large.txt"), "w") as f:
                f.write("x" * 1000)
            tool = LsTool(workspace_root=tmpdir)
            result = tool.ls(".", sort_by="size", reverse=True, dirs_first=False)
            assert result.entries[0].name == "large.txt"

    def test_ls_filter_ext(self):
        from app.agent.ls_tool import LsTool
        with tempfile.TemporaryDirectory() as tmpdir:
            open(os.path.join(tmpdir, "a.py"), "w").close()
            open(os.path.join(tmpdir, "b.txt"), "w").close()
            tool = LsTool(workspace_root=tmpdir)
            result = tool.ls(".", filter_ext=".py")
            file_names = [e.name for e in result.entries if not e.is_dir]
            assert "a.py" in file_names
            assert "b.txt" not in file_names

    def test_exists(self):
        from app.agent.ls_tool import LsTool
        with tempfile.TemporaryDirectory() as tmpdir:
            open(os.path.join(tmpdir, "test.py"), "w").close()
            tool = LsTool(workspace_root=tmpdir)
            assert tool.exists("test.py") == True
            assert tool.exists("nope.py") == False

    def test_is_dir(self):
        from app.agent.ls_tool import LsTool
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "src"))
            tool = LsTool(workspace_root=tmpdir)
            assert tool.is_dir("src") == True

    def test_file_info(self):
        from app.agent.ls_tool import LsTool
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "test.py"), "w") as f:
                f.write("hello")
            tool = LsTool(workspace_root=tmpdir)
            info = tool.file_info("test.py")
            assert info["size"] == 5
            assert info["extension"] == ".py"

    def test_format_output(self):
        from app.agent.ls_tool import LsTool
        with tempfile.TemporaryDirectory() as tmpdir:
            open(os.path.join(tmpdir, "test.py"), "w").close()
            tool = LsTool(workspace_root=tmpdir)
            result = tool.ls(".")
            text = result.format()
            assert "test.py" in text

    def test_nonexistent_dir(self):
        from app.agent.ls_tool import LsTool
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = LsTool(workspace_root=tmpdir)
            result = tool.ls("nonexistent")
            assert len(result.entries) == 0


# ── Chrome Relay ──────────────────────────────────────

class TestChromeRelay:
    def test_import(self):
        from app.agent.chrome_relay import ChromeRelay, get_chrome_relay
        assert ChromeRelay is not None

    def test_configure(self):
        from app.agent.chrome_relay import ChromeRelay
        relay = ChromeRelay()
        config = relay.configure(host="localhost", port=9222)
        assert config.host == "localhost"
        assert config.port == 9222

    def test_connect(self):
        from app.agent.chrome_relay import ChromeRelay
        relay = ChromeRelay()
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(relay.connect())
        assert result == True
        assert relay.is_connected == True

    def test_disconnect(self):
        from app.agent.chrome_relay import ChromeRelay
        relay = ChromeRelay()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(relay.connect())
        loop.run_until_complete(relay.disconnect())
        assert relay.is_connected == False

    def test_add_and_list_tabs(self):
        from app.agent.chrome_relay import ChromeRelay
        relay = ChromeRelay()
        relay.add_tab("t1", "Google", "https://google.com", active=True)
        relay.add_tab("t2", "GitHub", "https://github.com")
        loop = asyncio.get_event_loop()
        tabs = loop.run_until_complete(relay.list_tabs())
        assert len(tabs) == 2

    def test_evaluate(self):
        from app.agent.chrome_relay import ChromeRelay
        relay = ChromeRelay()
        relay.add_tab("t1", "Test", "https://test.com")
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(relay.evaluate("t1", "document.title"))
        assert result.success == True

    def test_evaluate_tab_not_found(self):
        from app.agent.chrome_relay import ChromeRelay
        relay = ChromeRelay()
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(relay.evaluate("nonexistent", "1+1"))
        assert result.success == False

    def test_navigate(self):
        from app.agent.chrome_relay import ChromeRelay
        relay = ChromeRelay()
        relay.add_tab("t1", "Test", "https://old.com")
        loop = asyncio.get_event_loop()
        loop.run_until_complete(relay.navigate("t1", "https://new.com"))
        # Verify tab URL updated
        tabs = loop.run_until_complete(relay.list_tabs())
        assert any(t["url"] == "https://new.com" for t in tabs)

    def test_screenshot(self):
        from app.agent.chrome_relay import ChromeRelay
        relay = ChromeRelay()
        relay.add_tab("t1", "Test", "https://test.com")
        loop = asyncio.get_event_loop()
        data = loop.run_until_complete(relay.screenshot("t1"))
        assert data is not None

    def test_profiles(self):
        from app.agent.chrome_relay import ChromeRelay, RelayConfig
        relay = ChromeRelay()
        relay.add_profile("work", RelayConfig(port=9223))
        assert "work" in relay.list_profiles()

    def test_stats(self):
        from app.agent.chrome_relay import ChromeRelay
        relay = ChromeRelay()
        s = relay.stats()
        assert s["state"] == "disconnected"

    def test_singleton(self):
        from app.agent.chrome_relay import get_chrome_relay
        r1 = get_chrome_relay()
        r2 = get_chrome_relay()
        assert r1 is r2


# ── Device Nodes ──────────────────────────────────────

class TestDeviceNodes:
    def test_import(self):
        from app.agent.device_nodes import DeviceNodeManager, get_node_manager
        assert DeviceNodeManager is not None

    def test_register_node(self):
        from app.agent.device_nodes import DeviceNodeManager, NodePlatform
        mgr = DeviceNodeManager()
        node = mgr.register_node("iphone1", platform=NodePlatform.IOS, capabilities=["camera", "location"])
        assert node.node_id == "iphone1"
        assert node.has_capability("camera") == True

    def test_unregister(self):
        from app.agent.device_nodes import DeviceNodeManager
        mgr = DeviceNodeManager()
        mgr.register_node("n1")
        assert mgr.unregister_node("n1") == True
        assert mgr.get_node("n1") is None

    def test_capture_camera(self):
        from app.agent.device_nodes import DeviceNodeManager
        mgr = DeviceNodeManager()
        mgr.register_node("n1", capabilities=["camera"])
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(mgr.capture_camera("n1"))
        assert result.success == True

    def test_capture_no_capability(self):
        from app.agent.device_nodes import DeviceNodeManager
        mgr = DeviceNodeManager()
        mgr.register_node("n1", capabilities=["location"])
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(mgr.capture_camera("n1"))
        assert result.success == False

    def test_get_location(self):
        from app.agent.device_nodes import DeviceNodeManager
        mgr = DeviceNodeManager()
        mgr.register_node("n1", capabilities=["location"])
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(mgr.get_location("n1"))
        assert result.success == True

    def test_send_notification(self):
        from app.agent.device_nodes import DeviceNodeManager
        mgr = DeviceNodeManager()
        mgr.register_node("n1", capabilities=["notifications"])
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(mgr.send_notification("n1", title="Hi", body="Hello"))
        assert result.success == True
        assert result.data["title"] == "Hi"

    def test_node_not_found(self):
        from app.agent.device_nodes import DeviceNodeManager
        mgr = DeviceNodeManager()
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(mgr.capture_camera("nonexistent"))
        assert result.success == False

    def test_set_state(self):
        from app.agent.device_nodes import DeviceNodeManager, NodeState
        mgr = DeviceNodeManager()
        mgr.register_node("n1")
        mgr.set_state("n1", NodeState.OFFLINE)
        node = mgr.get_node("n1")
        assert node.state == NodeState.OFFLINE

    def test_offline_node_blocked(self):
        from app.agent.device_nodes import DeviceNodeManager, NodeState
        mgr = DeviceNodeManager()
        mgr.register_node("n1", capabilities=["camera"])
        mgr.set_state("n1", NodeState.OFFLINE)
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(mgr.capture_camera("n1"))
        assert result.success == False

    def test_find_nodes_with_capability(self):
        from app.agent.device_nodes import DeviceNodeManager
        mgr = DeviceNodeManager()
        mgr.register_node("n1", capabilities=["camera", "location"])
        mgr.register_node("n2", capabilities=["location"])
        nodes = mgr.find_nodes_with_capability("camera")
        assert len(nodes) == 1

    def test_heartbeat(self):
        from app.agent.device_nodes import DeviceNodeManager
        mgr = DeviceNodeManager()
        mgr.register_node("n1")
        assert mgr.heartbeat("n1") == True

    def test_list_nodes(self):
        from app.agent.device_nodes import DeviceNodeManager
        mgr = DeviceNodeManager()
        mgr.register_node("n1")
        mgr.register_node("n2")
        assert len(mgr.list_nodes()) == 2

    def test_stats(self):
        from app.agent.device_nodes import DeviceNodeManager
        mgr = DeviceNodeManager()
        mgr.register_node("n1")
        s = mgr.stats()
        assert s["total_nodes"] == 1

    def test_singleton(self):
        from app.agent.device_nodes import get_node_manager
        m1 = get_node_manager()
        m2 = get_node_manager()
        assert m1 is m2


# ── Voice Call Plugin ─────────────────────────────────

class TestVoiceCall:
    def test_import(self):
        from app.agent.voice_call import VoiceCallManager, get_voice_call_manager
        assert VoiceCallManager is not None

    def test_configure_provider(self):
        from app.agent.voice_call import VoiceCallManager
        mgr = VoiceCallManager()
        config = mgr.configure_provider("twilio", credentials={"sid": "xxx"}, from_number="+1555000")
        assert config.provider.value == "twilio"

    def test_initiate_call(self):
        from app.agent.voice_call import VoiceCallManager, CallState
        mgr = VoiceCallManager()
        mgr.configure_provider("twilio", from_number="+1555000")
        loop = asyncio.get_event_loop()
        call = loop.run_until_complete(mgr.initiate_call("+1555999"))
        assert call.state == CallState.RINGING
        assert call.to_number == "+1555999"

    def test_initiate_no_provider(self):
        from app.agent.voice_call import VoiceCallManager, CallState
        mgr = VoiceCallManager()
        loop = asyncio.get_event_loop()
        call = loop.run_until_complete(mgr.initiate_call("+1555999"))
        assert call.state == CallState.FAILED

    def test_answer_call(self):
        from app.agent.voice_call import VoiceCallManager, CallState
        mgr = VoiceCallManager()
        mgr.configure_provider("twilio")
        loop = asyncio.get_event_loop()
        call = loop.run_until_complete(mgr.initiate_call("+1555999"))
        result = loop.run_until_complete(mgr.answer(call.call_id))
        assert result == True
        assert mgr.get_call(call.call_id).state == CallState.IN_PROGRESS

    def test_hangup(self):
        from app.agent.voice_call import VoiceCallManager, CallState
        mgr = VoiceCallManager()
        mgr.configure_provider("twilio")
        loop = asyncio.get_event_loop()
        call = loop.run_until_complete(mgr.initiate_call("+1555999"))
        loop.run_until_complete(mgr.answer(call.call_id))
        loop.run_until_complete(mgr.hangup(call.call_id))
        assert mgr.get_call(call.call_id).state == CallState.ENDED

    def test_hold_resume(self):
        from app.agent.voice_call import VoiceCallManager, CallState
        mgr = VoiceCallManager()
        mgr.configure_provider("twilio")
        loop = asyncio.get_event_loop()
        call = loop.run_until_complete(mgr.initiate_call("+1555999"))
        loop.run_until_complete(mgr.answer(call.call_id))
        loop.run_until_complete(mgr.hold(call.call_id))
        assert mgr.get_call(call.call_id).state == CallState.ON_HOLD
        loop.run_until_complete(mgr.resume(call.call_id))
        assert mgr.get_call(call.call_id).state == CallState.IN_PROGRESS

    def test_inbound_call(self):
        from app.agent.voice_call import VoiceCallManager, CallDirection
        mgr = VoiceCallManager()
        call = mgr.handle_inbound("ext-123", "+1555111", "+1555000", "twilio")
        assert call.direction == CallDirection.INBOUND

    def test_add_transcript(self):
        from app.agent.voice_call import VoiceCallManager
        mgr = VoiceCallManager()
        mgr.configure_provider("twilio")
        loop = asyncio.get_event_loop()
        call = loop.run_until_complete(mgr.initiate_call("+1555999"))
        mgr.add_transcript(call.call_id, "Hello!")
        assert len(mgr.get_call(call.call_id).transcript) == 1

    def test_list_calls(self):
        from app.agent.voice_call import VoiceCallManager
        mgr = VoiceCallManager()
        mgr.configure_provider("twilio")
        loop = asyncio.get_event_loop()
        loop.run_until_complete(mgr.initiate_call("+1555111"))
        loop.run_until_complete(mgr.initiate_call("+1555222"))
        assert len(mgr.list_calls()) == 2

    def test_list_providers(self):
        from app.agent.voice_call import VoiceCallManager
        mgr = VoiceCallManager()
        mgr.configure_provider("twilio")
        mgr.configure_provider("telnyx")
        assert len(mgr.list_providers()) == 2

    def test_invalid_provider(self):
        from app.agent.voice_call import VoiceCallManager
        mgr = VoiceCallManager()
        with pytest.raises(ValueError):
            mgr.configure_provider("invalid_provider")

    def test_stats(self):
        from app.agent.voice_call import VoiceCallManager
        mgr = VoiceCallManager()
        mgr.configure_provider("twilio")
        s = mgr.stats()
        assert s["providers"] == 1

    def test_singleton(self):
        from app.agent.voice_call import get_voice_call_manager
        m1 = get_voice_call_manager()
        m2 = get_voice_call_manager()
        assert m1 is m2
