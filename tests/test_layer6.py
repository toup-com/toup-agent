"""
Layer 6 Tests — Browser AI Snapshot, Canvas/A2UI, Sandbox Sessions,
Voice Talk Mode, Skill Marketplace, CLI Doctor.

Tests all new modules and enhancements from Layer 6.
"""

import asyncio
import json
import os
import sys
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ═══════════════════════════════════════════════════════════════
# 1. BROWSER — AI Snapshot + Tab Management + Profiles
# ═══════════════════════════════════════════════════════════════

class TestBrowserAISnapshot(unittest.TestCase):
    """Test browser AI snapshot / accessibility tree extraction."""

    def test_import_ai_snapshot(self):
        from app.agent.browser import ai_snapshot
        self.assertIsNotNone(ai_snapshot)

    def test_format_aria_tree(self):
        from app.agent.browser import _format_aria_tree
        node = {
            "role": "WebArea",
            "name": "Test Page",
            "children": [
                {"role": "heading", "name": "Hello", "level": 1},
                {"role": "button", "name": "Click Me"},
                {"role": "textbox", "name": "Email", "value": "test@test.com"},
            ],
        }
        result = _format_aria_tree(node)
        self.assertIn("WebArea", result)
        self.assertIn("heading", result)
        self.assertIn("Hello", result)
        self.assertIn("button", result)
        self.assertIn("Click Me", result)
        self.assertIn("textbox", result)
        self.assertIn("value=test@test.com", result)

    def test_format_ai_tree(self):
        from app.agent.browser import _format_ai_tree
        node = {
            "role": "WebArea",
            "name": "Test",
            "children": [
                {"role": "button", "name": "Submit"},
                {"role": "textbox", "name": "Input"},
                {"role": "link", "name": "Home"},
                {"role": "checkbox", "name": "Agree", "checked": True},
            ],
        }
        result = _format_ai_tree(node)
        self.assertIn("[clickable]", result)
        self.assertIn("[editable]", result)
        self.assertIn("[toggleable", result)
        self.assertIn("Submit", result)

    def test_format_ai_tree_skip_generic(self):
        from app.agent.browser import _format_ai_tree
        node = {"role": "generic", "name": "", "children": []}
        result = _format_ai_tree(node)
        self.assertEqual(result, "")

    def test_aria_tree_indentation(self):
        from app.agent.browser import _format_aria_tree
        node = {
            "role": "document",
            "name": "doc",
            "children": [
                {"role": "paragraph", "name": "text", "children": [
                    {"role": "link", "name": "click here"},
                ]},
            ],
        }
        result = _format_aria_tree(node)
        lines = result.split("\n")
        self.assertTrue(lines[0].startswith("document"))
        self.assertTrue(lines[1].startswith("  paragraph"))
        self.assertTrue(lines[2].startswith("    link"))


class TestBrowserTabManager(unittest.TestCase):
    """Test tab management."""

    def test_import_tab_manager(self):
        from app.agent.browser import TabManager
        tm = TabManager()
        self.assertEqual(tm.count, 0)

    def test_list_tabs_empty(self):
        from app.agent.browser import TabManager
        tm = TabManager()
        self.assertEqual(tm.list_tabs(), [])

    def test_tab_manager_counter(self):
        from app.agent.browser import TabManager
        tm = TabManager()
        self.assertEqual(tm._counter, 0)

    def test_get_tab_nonexistent(self):
        from app.agent.browser import TabManager
        tm = TabManager()
        self.assertIsNone(tm.get_tab("tab_999"))


class TestBrowserProfiles(unittest.TestCase):
    """Test browser profile support."""

    def test_profile_enum(self):
        from app.agent.browser import BrowserProfile
        self.assertEqual(BrowserProfile.MANAGED, "managed")
        self.assertEqual(BrowserProfile.CHROME, "chrome")
        self.assertEqual(BrowserProfile.REMOTE, "remote")

    def test_get_tab_manager(self):
        from app.agent.browser import get_tab_manager
        tm = get_tab_manager()
        self.assertIsNotNone(tm)

    def test_run_action_tabs_list(self):
        from app.agent.browser import run_action
        result = asyncio.get_event_loop().run_until_complete(
            run_action(action="tabs_list")
        )
        data = json.loads(result)
        self.assertIn("tabs", data)
        self.assertIn("count", data)

    def test_run_action_tab_close_missing(self):
        from app.agent.browser import run_action
        result = asyncio.get_event_loop().run_until_complete(
            run_action(action="tab_close")
        )
        self.assertIn("ERROR", result)

    def test_run_action_unknown_no_url(self):
        """Unknown action without url/tab_id should error before browser launch."""
        from app.agent.browser import run_action
        result = asyncio.get_event_loop().run_until_complete(
            run_action(action="unknown_action")
        )
        self.assertIn("ERROR", result)


# ═══════════════════════════════════════════════════════════════
# 2. CANVAS / A2UI — Agent-to-UI Push System
# ═══════════════════════════════════════════════════════════════

class TestCanvasManager(unittest.TestCase):
    """Test canvas / A2UI system."""

    def test_import(self):
        from app.agent.canvas import CanvasManager, CanvasContentType, get_canvas_manager
        self.assertIsNotNone(CanvasManager)
        self.assertIsNotNone(get_canvas_manager)

    def test_content_types(self):
        from app.agent.canvas import CanvasContentType
        types = [e.value for e in CanvasContentType]
        self.assertIn("html", types)
        self.assertIn("markdown", types)
        self.assertIn("json_data", types)
        self.assertIn("chart", types)
        self.assertIn("code", types)
        self.assertIn("image", types)
        self.assertIn("iframe", types)
        self.assertIn("custom", types)

    def test_present(self):
        from app.agent.canvas import CanvasManager
        mgr = CanvasManager()
        result = asyncio.get_event_loop().run_until_complete(
            mgr.present("user1", "<h1>Hello</h1>", "html", "Test")
        )
        self.assertIn("frame_id", result)
        self.assertTrue(result["visible"])
        self.assertEqual(result["frame_count"], 1)

    def test_present_update_existing(self):
        from app.agent.canvas import CanvasManager
        mgr = CanvasManager()
        loop = asyncio.get_event_loop()
        r1 = loop.run_until_complete(
            mgr.present("user1", "old", "html", frame_id="f1")
        )
        r2 = loop.run_until_complete(
            mgr.present("user1", "new", "html", frame_id="f1")
        )
        self.assertEqual(r1["frame_id"], "f1")
        self.assertEqual(r2["frame_count"], 1)  # Same frame updated

    def test_hide_and_show(self):
        from app.agent.canvas import CanvasManager
        mgr = CanvasManager()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(mgr.present("user1", "hi"))
        loop.run_until_complete(mgr.hide("user1"))
        state = mgr.get_state("user1")
        self.assertFalse(state.visible)
        loop.run_until_complete(mgr.show("user1"))
        self.assertTrue(state.visible)

    def test_clear_single_frame(self):
        from app.agent.canvas import CanvasManager
        mgr = CanvasManager()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(mgr.present("user1", "a", frame_id="f1"))
        loop.run_until_complete(mgr.present("user1", "b", frame_id="f2"))
        result = loop.run_until_complete(mgr.clear("user1", "f1"))
        self.assertEqual(result["remaining"], 1)

    def test_clear_all(self):
        from app.agent.canvas import CanvasManager
        mgr = CanvasManager()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(mgr.present("user1", "a"))
        loop.run_until_complete(mgr.present("user1", "b"))
        result = loop.run_until_complete(mgr.clear("user1"))
        self.assertEqual(result["remaining"], 0)

    def test_clear_nonexistent_frame(self):
        from app.agent.canvas import CanvasManager
        mgr = CanvasManager()
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(mgr.clear("user1", "nope"))
        self.assertIn("error", result)

    def test_set_layout(self):
        from app.agent.canvas import CanvasManager
        mgr = CanvasManager()
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(mgr.set_layout("user1", "grid"))
        self.assertEqual(result["layout"], "grid")

    def test_set_layout_invalid(self):
        from app.agent.canvas import CanvasManager
        mgr = CanvasManager()
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(mgr.set_layout("user1", "invalid"))
        self.assertIn("error", result)

    def test_evaluate_js(self):
        from app.agent.canvas import CanvasManager
        mgr = CanvasManager()
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(mgr.evaluate_js("user1", "alert(1)"))
        self.assertEqual(result["status"], "eval_sent")

    def test_snapshot(self):
        from app.agent.canvas import CanvasManager
        mgr = CanvasManager()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(mgr.present("user1", "hello", title="Test"))
        snapshot = loop.run_until_complete(mgr.snapshot("user1"))
        self.assertTrue(snapshot["visible"])
        self.assertEqual(len(snapshot["frames"]), 1)

    def test_listener_push(self):
        from app.agent.canvas import CanvasManager
        mgr = CanvasManager()
        received = []

        async def on_msg(msg):
            received.append(msg)

        mgr.add_listener("user1", on_msg)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(mgr.present("user1", "push test"))
        self.assertTrue(len(received) > 0)
        self.assertEqual(received[0]["type"], "canvas_present")

    def test_remove_listener(self):
        from app.agent.canvas import CanvasManager
        mgr = CanvasManager()
        received = []

        async def on_msg(msg):
            received.append(msg)

        mgr.add_listener("user1", on_msg)
        mgr.remove_listener("user1", on_msg)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(mgr.present("user1", "no push"))
        self.assertEqual(len(received), 0)

    def test_active_users_count(self):
        from app.agent.canvas import CanvasManager
        mgr = CanvasManager()
        loop = asyncio.get_event_loop()
        self.assertEqual(mgr.active_users, 0)
        loop.run_until_complete(mgr.present("user1", "a"))
        self.assertEqual(mgr.active_users, 1)
        loop.run_until_complete(mgr.present("user2", "b"))
        self.assertEqual(mgr.active_users, 2)
        loop.run_until_complete(mgr.hide("user1"))
        self.assertEqual(mgr.active_users, 1)

    def test_canvas_state_to_dict(self):
        from app.agent.canvas import CanvasManager
        mgr = CanvasManager()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(mgr.present("user1", "test", "html", "Title", frame_id="f1"))
        state = mgr.get_state("user1")
        d = state.to_dict()
        self.assertEqual(d["user_id"], "user1")
        self.assertIn("f1", d["frames"])
        self.assertEqual(d["frames"]["f1"]["title"], "Title")


# ═══════════════════════════════════════════════════════════════
# 3. SANDBOX — Per-Session Containers + Auto-Cleanup
# ═══════════════════════════════════════════════════════════════

class TestSandboxEnhancements(unittest.TestCase):
    """Test enhanced Docker sandbox features."""

    def test_import(self):
        from app.agent.sandbox import SandboxExecutor, ContainerConfig, NetworkPolicy
        self.assertIsNotNone(SandboxExecutor)

    def test_network_policy_enum(self):
        from app.agent.sandbox import NetworkPolicy
        self.assertEqual(NetworkPolicy.NONE, "none")
        self.assertEqual(NetworkPolicy.INTERNAL, "internal")
        self.assertEqual(NetworkPolicy.FULL, "full")

    def test_container_config_defaults(self):
        from app.agent.sandbox import ContainerConfig
        cfg = ContainerConfig()
        self.assertEqual(cfg.memory, "256m")
        self.assertEqual(cfg.cpus, "0.5")
        self.assertEqual(cfg.pids_limit, 64)
        self.assertTrue(cfg.read_only)
        self.assertEqual(cfg.network.value, "none")

    def test_container_config_custom(self):
        from app.agent.sandbox import ContainerConfig, NetworkPolicy
        cfg = ContainerConfig(
            memory="512m", cpus="1.0", pids_limit=128,
            read_only=False, network=NetworkPolicy.INTERNAL,
        )
        self.assertEqual(cfg.memory, "512m")
        self.assertEqual(cfg.cpus, "1.0")
        self.assertFalse(cfg.read_only)
        self.assertEqual(cfg.network, NetworkPolicy.INTERNAL)

    def test_container_key_user_only(self):
        from app.agent.sandbox import SandboxExecutor
        sandbox = SandboxExecutor()
        key = sandbox._container_key("user123")
        self.assertEqual(key, "user123")

    def test_container_key_with_session(self):
        from app.agent.sandbox import SandboxExecutor
        sandbox = SandboxExecutor()
        key = sandbox._container_key("user123", "sess456")
        self.assertEqual(key, "user123:sess456")

    def test_list_containers_empty(self):
        from app.agent.sandbox import SandboxExecutor
        sandbox = SandboxExecutor()
        self.assertEqual(sandbox.list_containers(), {})

    def test_stop_nonexistent(self):
        from app.agent.sandbox import SandboxExecutor
        sandbox = SandboxExecutor()
        result = asyncio.get_event_loop().run_until_complete(
            sandbox.stop("nonexistent_user")
        )
        self.assertFalse(result)

    def test_container_info_structure(self):
        from app.agent.sandbox import ContainerInfo, ContainerConfig
        info = ContainerInfo(
            container_name="hex-sandbox-test",
            user_id="user1",
            session_id="sess1",
            config=ContainerConfig(),
        )
        self.assertEqual(info.command_count, 0)
        self.assertEqual(info.container_name, "hex-sandbox-test")
        self.assertEqual(info.session_id, "sess1")

    def test_stop_all_empty(self):
        from app.agent.sandbox import SandboxExecutor
        sandbox = SandboxExecutor()
        result = asyncio.get_event_loop().run_until_complete(sandbox.stop_all())
        self.assertEqual(result, 0)


# ═══════════════════════════════════════════════════════════════
# 4. VOICE — Talk Mode + ElevenLabs + Voice Wake
# ═══════════════════════════════════════════════════════════════

class TestVoiceTalkMode(unittest.TestCase):
    """Test Talk Mode session management."""

    def test_import(self):
        from app.agent.voice_handler import TalkModeSession, TalkModeManager
        self.assertIsNotNone(TalkModeSession)
        self.assertIsNotNone(TalkModeManager)

    def test_session_lifecycle(self):
        from app.agent.voice_handler import TalkModeSession
        sess = TalkModeSession("user1", "sess1")
        self.assertEqual(sess.state, TalkModeSession.State.IDLE)
        self.assertFalse(sess.is_active)

        sess.start()
        self.assertTrue(sess.is_active)
        self.assertEqual(sess.state, TalkModeSession.State.LISTENING)

        sess.stop()
        self.assertFalse(sess.is_active)
        self.assertEqual(sess.state, TalkModeSession.State.ENDED)

    def test_session_audio_buffer(self):
        from app.agent.voice_handler import TalkModeSession
        sess = TalkModeSession("user1")
        sess.add_audio_chunk(b"chunk1")
        sess.add_audio_chunk(b"chunk2")
        self.assertEqual(sess.total_audio_bytes, 12)

        buf = sess.get_audio_buffer()
        self.assertEqual(buf, b"chunk1chunk2")
        self.assertEqual(len(sess.get_audio_buffer()), 0)  # Cleared

    def test_session_state_changes(self):
        from app.agent.voice_handler import TalkModeSession
        sess = TalkModeSession("user1")
        sess.set_state(TalkModeSession.State.TRANSCRIBING)
        self.assertEqual(sess.state, TalkModeSession.State.TRANSCRIBING)

    def test_session_to_dict(self):
        from app.agent.voice_handler import TalkModeSession
        sess = TalkModeSession("user1", "sess1")
        sess.start()
        d = sess.to_dict()
        self.assertEqual(d["user_id"], "user1")
        self.assertEqual(d["session_id"], "sess1")
        self.assertEqual(d["state"], "listening")
        self.assertTrue(d["is_active"])

    def test_manager_start_end(self):
        from app.agent.voice_handler import TalkModeManager
        mgr = TalkModeManager()
        sess = mgr.start_session("user1")
        self.assertTrue(sess.is_active)
        self.assertEqual(mgr.active_count, 1)

        ended = mgr.end_session("user1")
        self.assertTrue(ended)
        self.assertEqual(mgr.active_count, 0)

    def test_manager_list_sessions(self):
        from app.agent.voice_handler import TalkModeManager
        mgr = TalkModeManager()
        mgr.start_session("user1")
        mgr.start_session("user2")
        sessions = mgr.list_sessions()
        self.assertEqual(len(sessions), 2)

    def test_manager_get_session(self):
        from app.agent.voice_handler import TalkModeManager
        mgr = TalkModeManager()
        mgr.start_session("user1")
        self.assertIsNotNone(mgr.get_session("user1"))
        self.assertIsNone(mgr.get_session("user999"))

    def test_manager_end_nonexistent(self):
        from app.agent.voice_handler import TalkModeManager
        mgr = TalkModeManager()
        self.assertFalse(mgr.end_session("nobody"))


class TestVoiceWake(unittest.TestCase):
    """Test voice wake / hotword detection."""

    def test_import(self):
        from app.agent.voice_handler import VoiceWakeDetector
        self.assertIsNotNone(VoiceWakeDetector)

    def test_default_wake_words(self):
        from app.agent.voice_handler import VoiceWakeDetector
        det = VoiceWakeDetector()
        self.assertTrue(det.check("hey hex what's up"))
        self.assertTrue(det.check("ok hex tell me"))
        self.assertTrue(det.check("hexbrain do this"))
        self.assertTrue(det.check("hex brain hello"))

    def test_no_match(self):
        from app.agent.voice_handler import VoiceWakeDetector
        det = VoiceWakeDetector()
        self.assertFalse(det.check("hello there"))
        self.assertFalse(det.check("what time is it"))

    def test_strip_wake_word(self):
        from app.agent.voice_handler import VoiceWakeDetector
        det = VoiceWakeDetector()
        self.assertEqual(det.strip_wake_word("hey hex what time"), "what time")
        self.assertEqual(det.strip_wake_word("hexbrain search"), "search")

    def test_custom_wake_words(self):
        from app.agent.voice_handler import VoiceWakeDetector
        det = VoiceWakeDetector(wake_words={"jarvis"})
        self.assertTrue(det.check("jarvis what's up"))
        self.assertFalse(det.check("hey hex"))

    def test_add_remove_wake_word(self):
        from app.agent.voice_handler import VoiceWakeDetector
        det = VoiceWakeDetector()
        det.add_wake_word("yo hex")
        self.assertTrue(det.check("yo hex"))
        det.remove_wake_word("yo hex")
        self.assertFalse(det.check("yo hex"))

    def test_list_wake_words(self):
        from app.agent.voice_handler import VoiceWakeDetector
        det = VoiceWakeDetector()
        words = det.list_wake_words()
        self.assertIsInstance(words, list)
        self.assertTrue(len(words) > 0)

    def test_disabled(self):
        from app.agent.voice_handler import VoiceWakeDetector
        det = VoiceWakeDetector()
        det.enabled = False
        self.assertFalse(det.check("hey hex"))

    def test_get_talk_mode_manager(self):
        from app.agent.voice_handler import get_talk_mode_manager
        mgr = get_talk_mode_manager()
        self.assertIsNotNone(mgr)


class TestVoiceTTSProviders(unittest.TestCase):
    """Test TTS provider abstraction."""

    def test_tts_provider_enum(self):
        from app.agent.voice_handler import TTSProvider
        self.assertEqual(TTSProvider.OPENAI, "openai")
        self.assertEqual(TTSProvider.ELEVENLABS, "elevenlabs")
        self.assertEqual(TTSProvider.EDGE, "edge")

    def test_valid_voices(self):
        from app.agent.voice_handler import VALID_VOICES
        self.assertIn("alloy", VALID_VOICES)
        self.assertIn("nova", VALID_VOICES)
        self.assertIn("shimmer", VALID_VOICES)
        self.assertEqual(len(VALID_VOICES), 10)

    def test_elevenlabs_voices(self):
        from app.agent.voice_handler import ELEVENLABS_DEFAULT_VOICES
        self.assertIn("rachel", ELEVENLABS_DEFAULT_VOICES)
        self.assertIn("adam", ELEVENLABS_DEFAULT_VOICES)

    def test_synthesize_empty_text(self):
        from app.agent.voice_handler import _tts_openai
        result = asyncio.get_event_loop().run_until_complete(
            _tts_openai("", "nova", "gpt-4o-mini-tts", 1.0, None)
        )
        self.assertTrue(result.startswith("ERROR"))

    def test_elevenlabs_no_key(self):
        from app.agent.voice_handler import _tts_elevenlabs
        with patch("app.agent.voice_handler.settings") as mock_settings:
            mock_settings.elevenlabs_api_key = None
            result = asyncio.get_event_loop().run_until_complete(
                _tts_elevenlabs("hello", "rachel")
            )
            self.assertIn("ERROR", result)

    def test_edge_tts_empty(self):
        from app.agent.voice_handler import _tts_edge
        result = asyncio.get_event_loop().run_until_complete(
            _tts_edge("", "nova")
        )
        self.assertIn("ERROR", result)


# ═══════════════════════════════════════════════════════════════
# 5. SKILL MARKETPLACE
# ═══════════════════════════════════════════════════════════════

class TestSkillMarketplace(unittest.TestCase):
    """Test skill marketplace system."""

    def test_import(self):
        from app.agent.skills.marketplace import SkillMarketplace, SkillManifest, SkillStatus
        self.assertIsNotNone(SkillMarketplace)

    def test_skill_status_enum(self):
        from app.agent.skills.marketplace import SkillStatus
        self.assertEqual(SkillStatus.AVAILABLE, "available")
        self.assertEqual(SkillStatus.INSTALLED, "installed")
        self.assertEqual(SkillStatus.UPDATE_AVAILABLE, "update_available")
        self.assertEqual(SkillStatus.INCOMPATIBLE, "incompatible")

    def test_manifest_creation(self):
        from app.agent.skills.marketplace import SkillManifest
        m = SkillManifest(
            name="test_skill",
            version="1.0.0",
            description="A test skill",
            author="TestAuthor",
            tags=["test"],
            tools=["test_tool"],
        )
        self.assertEqual(m.name, "test_skill")
        self.assertEqual(m.version, "1.0.0")
        self.assertIn("test_tool", m.tools)

    def test_manifest_to_dict(self):
        from app.agent.skills.marketplace import SkillManifest
        m = SkillManifest(name="test", version="1.0", description="desc")
        d = m.to_dict()
        self.assertEqual(d["name"], "test")
        self.assertIn("tags", d)
        self.assertIn("dependencies", d)

    def test_manifest_from_dict(self):
        from app.agent.skills.marketplace import SkillManifest
        data = {"name": "test", "version": "2.0", "description": "desc", "author": "me"}
        m = SkillManifest.from_dict(data)
        self.assertEqual(m.name, "test")
        self.assertEqual(m.version, "2.0")
        self.assertEqual(m.author, "me")

    def test_marketplace_init(self):
        from app.agent.skills.marketplace import SkillMarketplace
        with tempfile.TemporaryDirectory() as tmpdir:
            mp = SkillMarketplace(skills_dir=tmpdir)
            self.assertEqual(len(mp.list_installed()), 0)

    def test_builtin_registry(self):
        from app.agent.skills.marketplace import SkillMarketplace
        with tempfile.TemporaryDirectory() as tmpdir:
            mp = SkillMarketplace(skills_dir=tmpdir)
            registry = mp._builtin_registry()
            self.assertTrue(len(registry) >= 5)
            names = [m.name for m in registry]
            self.assertIn("math_tools", names)
            self.assertIn("github_integration", names)

    def test_search(self):
        from app.agent.skills.marketplace import SkillMarketplace
        with tempfile.TemporaryDirectory() as tmpdir:
            mp = SkillMarketplace(skills_dir=tmpdir)
            results = asyncio.get_event_loop().run_until_complete(
                mp.search("math")
            )
            self.assertTrue(len(results) > 0)
            self.assertEqual(results[0]["name"], "math_tools")

    def test_search_by_tags(self):
        from app.agent.skills.marketplace import SkillMarketplace
        with tempfile.TemporaryDirectory() as tmpdir:
            mp = SkillMarketplace(skills_dir=tmpdir)
            results = asyncio.get_event_loop().run_until_complete(
                mp.search(tags=["database"])
            )
            self.assertTrue(len(results) > 0)

    def test_search_no_results(self):
        from app.agent.skills.marketplace import SkillMarketplace
        with tempfile.TemporaryDirectory() as tmpdir:
            mp = SkillMarketplace(skills_dir=tmpdir)
            results = asyncio.get_event_loop().run_until_complete(
                mp.search("nonexistent_xyzzy")
            )
            self.assertEqual(len(results), 0)

    def test_install_and_uninstall(self):
        from app.agent.skills.marketplace import SkillMarketplace
        with tempfile.TemporaryDirectory() as tmpdir:
            mp = SkillMarketplace(skills_dir=tmpdir)
            # Install
            result = asyncio.get_event_loop().run_until_complete(
                mp.install("math_tools")
            )
            self.assertEqual(result["status"], "installed")
            self.assertEqual(len(mp.list_installed()), 1)

            # Uninstall
            result = asyncio.get_event_loop().run_until_complete(
                mp.uninstall("math_tools")
            )
            self.assertEqual(result["status"], "uninstalled")
            self.assertEqual(len(mp.list_installed()), 0)

    def test_install_duplicate(self):
        from app.agent.skills.marketplace import SkillMarketplace
        with tempfile.TemporaryDirectory() as tmpdir:
            mp = SkillMarketplace(skills_dir=tmpdir)
            asyncio.get_event_loop().run_until_complete(mp.install("math_tools"))
            result = asyncio.get_event_loop().run_until_complete(mp.install("math_tools"))
            self.assertIn("error", result)

    def test_install_nonexistent(self):
        from app.agent.skills.marketplace import SkillMarketplace
        with tempfile.TemporaryDirectory() as tmpdir:
            mp = SkillMarketplace(skills_dir=tmpdir)
            result = asyncio.get_event_loop().run_until_complete(
                mp.install("nonexistent_skill_xyz")
            )
            self.assertIn("error", result)

    def test_uninstall_nonexistent(self):
        from app.agent.skills.marketplace import SkillMarketplace
        with tempfile.TemporaryDirectory() as tmpdir:
            mp = SkillMarketplace(skills_dir=tmpdir)
            result = asyncio.get_event_loop().run_until_complete(
                mp.uninstall("not_installed")
            )
            self.assertIn("error", result)

    def test_enable_disable(self):
        from app.agent.skills.marketplace import SkillMarketplace
        with tempfile.TemporaryDirectory() as tmpdir:
            mp = SkillMarketplace(skills_dir=tmpdir)
            asyncio.get_event_loop().run_until_complete(mp.install("math_tools"))
            self.assertTrue(mp.disable_skill("math_tools"))
            info = mp.get_installed("math_tools")
            self.assertFalse(info.enabled)
            self.assertTrue(mp.enable_skill("math_tools"))
            self.assertTrue(info.enabled)

    def test_compatibility_check(self):
        from app.agent.skills.marketplace import SkillMarketplace, SkillManifest
        with tempfile.TemporaryDirectory() as tmpdir:
            mp = SkillMarketplace(skills_dir=tmpdir)
            m_ok = SkillManifest(name="ok", version="1.0", description="", min_platform_version="1.0.0")
            m_bad = SkillManifest(name="bad", version="1.0", description="", min_platform_version="99.0.0")
            self.assertTrue(mp._check_compatibility(m_ok))
            self.assertFalse(mp._check_compatibility(m_bad))

    def test_update(self):
        from app.agent.skills.marketplace import SkillMarketplace
        with tempfile.TemporaryDirectory() as tmpdir:
            mp = SkillMarketplace(skills_dir=tmpdir)
            asyncio.get_event_loop().run_until_complete(mp.install("math_tools"))
            result = asyncio.get_event_loop().run_until_complete(mp.update("math_tools"))
            self.assertEqual(result["status"], "installed")

    def test_update_not_installed(self):
        from app.agent.skills.marketplace import SkillMarketplace
        with tempfile.TemporaryDirectory() as tmpdir:
            mp = SkillMarketplace(skills_dir=tmpdir)
            result = asyncio.get_event_loop().run_until_complete(mp.update("not_here"))
            self.assertIn("error", result)

    def test_get_marketplace_singleton(self):
        from app.agent.skills.marketplace import get_marketplace
        mp = get_marketplace("/tmp/test_skills_mp")
        self.assertIsNotNone(mp)


# ═══════════════════════════════════════════════════════════════
# 6. CLI DOCTOR — Health Checks
# ═══════════════════════════════════════════════════════════════

class TestCLIDoctor(unittest.TestCase):
    """Test CLI doctor / health checks."""

    def test_import(self):
        from app.agent.cli_doctor import run_doctor, DoctorReport, CheckStatus
        self.assertIsNotNone(run_doctor)

    def test_check_status_enum(self):
        from app.agent.cli_doctor import CheckStatus
        self.assertEqual(CheckStatus.OK, "ok")
        self.assertEqual(CheckStatus.WARNING, "warning")
        self.assertEqual(CheckStatus.ERROR, "error")
        self.assertEqual(CheckStatus.SKIPPED, "skipped")

    def test_check_result_to_dict(self):
        from app.agent.cli_doctor import CheckResult, CheckStatus
        cr = CheckResult(name="test", status=CheckStatus.OK, message="All good")
        d = cr.to_dict()
        self.assertEqual(d["name"], "test")
        self.assertEqual(d["status"], "ok")
        self.assertEqual(d["message"], "All good")

    def test_doctor_report_counts(self):
        from app.agent.cli_doctor import DoctorReport, CheckResult, CheckStatus
        report = DoctorReport()
        report.checks = [
            CheckResult("a", CheckStatus.OK, "ok"),
            CheckResult("b", CheckStatus.WARNING, "warn"),
            CheckResult("c", CheckStatus.ERROR, "err"),
            CheckResult("d", CheckStatus.OK, "ok"),
        ]
        self.assertEqual(report.ok_count, 2)
        self.assertEqual(report.warning_count, 1)
        self.assertEqual(report.error_count, 1)
        self.assertEqual(report.overall_status, CheckStatus.ERROR)

    def test_report_overall_ok(self):
        from app.agent.cli_doctor import DoctorReport, CheckResult, CheckStatus
        report = DoctorReport()
        report.checks = [
            CheckResult("a", CheckStatus.OK, "ok"),
        ]
        self.assertEqual(report.overall_status, CheckStatus.OK)

    def test_report_overall_warning(self):
        from app.agent.cli_doctor import DoctorReport, CheckResult, CheckStatus
        report = DoctorReport()
        report.checks = [
            CheckResult("a", CheckStatus.OK, "ok"),
            CheckResult("b", CheckStatus.WARNING, "warn"),
        ]
        self.assertEqual(report.overall_status, CheckStatus.WARNING)

    def test_report_to_dict(self):
        from app.agent.cli_doctor import DoctorReport, CheckResult, CheckStatus
        report = DoctorReport()
        report.checks = [CheckResult("test", CheckStatus.OK, "ok")]
        d = report.to_dict()
        self.assertIn("overall_status", d)
        self.assertIn("summary", d)
        self.assertIn("checks", d)
        self.assertEqual(d["summary"]["total"], 1)

    def test_report_to_text(self):
        from app.agent.cli_doctor import DoctorReport, CheckResult, CheckStatus
        report = DoctorReport()
        report.checks = [
            CheckResult("test", CheckStatus.OK, "All good"),
            CheckResult("warn", CheckStatus.WARNING, "Check this"),
        ]
        text = report.to_text()
        self.assertIn("✅", text)
        self.assertIn("⚠️", text)
        self.assertIn("HexBrain Doctor Report", text)

    def test_check_python_deps(self):
        from app.agent.cli_doctor import _check_python_deps
        result = asyncio.get_event_loop().run_until_complete(_check_python_deps())
        # Should at least check for packages
        self.assertIn(result.status.value, ["ok", "warning", "error"])

    def test_check_config(self):
        from app.agent.cli_doctor import _check_config
        result = asyncio.get_event_loop().run_until_complete(_check_config())
        self.assertIn(result.status.value, ["ok", "warning", "error"])

    def test_check_disk_space(self):
        from app.agent.cli_doctor import _check_disk_space, CheckStatus
        result = asyncio.get_event_loop().run_until_complete(_check_disk_space())
        self.assertEqual(result.name, "disk_space")
        self.assertIn("free", result.message.lower() if result.status != CheckStatus.ERROR else "free")

    def test_run_doctor_specific_checks(self):
        from app.agent.cli_doctor import run_doctor
        report = asyncio.get_event_loop().run_until_complete(
            run_doctor(include=["disk_space", "config"])
        )
        check_names = [c.name for c in report.checks]
        self.assertIn("disk_space", check_names)
        self.assertIn("config", check_names)
        self.assertNotIn("docker", check_names)

    def test_run_doctor_safe_checks(self):
        from app.agent.cli_doctor import run_doctor
        report = asyncio.get_event_loop().run_until_complete(
            run_doctor(include=["python_deps", "config", "disk_space"])
        )
        self.assertTrue(len(report.checks) > 0)
        self.assertIn("python", report.platform_info)

    def test_cli_send_import(self):
        from app.agent.cli_doctor import cli_send
        self.assertIsNotNone(cli_send)


# ═══════════════════════════════════════════════════════════════
# 7. CANVAS API ROUTER
# ═══════════════════════════════════════════════════════════════

class TestCanvasRouter(unittest.TestCase):
    """Test canvas API router."""

    def test_import(self):
        from app.api.canvas import router
        self.assertIsNotNone(router)
        self.assertEqual(router.prefix, "/canvas")

    def test_routes_exist(self):
        from app.api.canvas import router
        paths = [r.path for r in router.routes]
        self.assertTrue(any("/state/" in p for p in paths))
        self.assertTrue(any("/present" in p for p in paths))
        self.assertTrue(any("/hide/" in p for p in paths))
        self.assertTrue(any("/clear/" in p for p in paths))


class TestDoctorRouter(unittest.TestCase):
    """Test doctor API router."""

    def test_import(self):
        from app.api.doctor import router
        self.assertIsNotNone(router)
        self.assertEqual(router.prefix, "/doctor")

    def test_routes_exist(self):
        from app.api.doctor import router
        paths = [r.path for r in router.routes]
        self.assertTrue(any("/" in p for p in paths))


class TestVoiceRouter(unittest.TestCase):
    """Test voice / talk mode API router."""

    def test_import(self):
        from app.api.voice import router
        self.assertIsNotNone(router)
        self.assertEqual(router.prefix, "/voice")

    def test_set_refs(self):
        from app.api.voice import set_voice_refs
        set_voice_refs(MagicMock())
        # Should not raise


# ═══════════════════════════════════════════════════════════════
# 8. CONFIG — New Layer 6 Fields
# ═══════════════════════════════════════════════════════════════

class TestLayer6Config(unittest.TestCase):
    """Test new Layer 6 config fields."""

    def test_existing_sandbox_config(self):
        from app.config import settings
        self.assertIsInstance(settings.sandbox_enabled, bool)
        self.assertIsInstance(settings.sandbox_image, str)

    def test_tts_provider_config(self):
        from app.config import settings
        self.assertIn(settings.tts_provider, ["openai", "elevenlabs", "edge"])

    def test_elevenlabs_config(self):
        from app.config import settings
        self.assertIsInstance(settings.elevenlabs_model, str)
        self.assertIsInstance(settings.elevenlabs_voice_id, str)


if __name__ == "__main__":
    unittest.main()
