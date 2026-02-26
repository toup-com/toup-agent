"""
Layer 13 Tests — Thinking Levels, Verbose Mode, Idempotency,
Apply Patch, TTS Preferences, DM Pairing.
"""

import asyncio
import json
import os
import tempfile
import pytest


# ── Thinking Levels ───────────────────────────────────

class TestThinkingLevels:
    def test_import(self):
        from app.agent.thinking_levels import ThinkingManager, ThinkingLevel
        assert ThinkingManager is not None

    def test_default_medium(self):
        from app.agent.thinking_levels import ThinkingManager, ThinkingLevel
        mgr = ThinkingManager()
        assert mgr.get_level("any") == ThinkingLevel.MEDIUM

    def test_set_level(self):
        from app.agent.thinking_levels import ThinkingManager, ThinkingLevel
        mgr = ThinkingManager()
        mgr.set_level("s1", ThinkingLevel.HIGH)
        assert mgr.get_level("s1") == ThinkingLevel.HIGH

    def test_get_model_params(self):
        from app.agent.thinking_levels import ThinkingManager, ThinkingLevel
        mgr = ThinkingManager()
        mgr.set_level("s1", ThinkingLevel.HIGH)
        params = mgr.get_model_params("s1")
        assert params["thinking_budget"] == 4096
        assert params["max_tokens"] == 8192

    def test_custom_budget(self):
        from app.agent.thinking_levels import ThinkingManager, ThinkingLevel
        mgr = ThinkingManager()
        config = mgr.set_level("s1", ThinkingLevel.HIGH, custom_budget=8000)
        assert config.effective_budget == 8000
        params = mgr.get_model_params("s1")
        assert params["thinking_budget"] == 8000

    def test_cycle_level(self):
        from app.agent.thinking_levels import ThinkingManager, ThinkingLevel
        mgr = ThinkingManager()
        mgr.set_level("s1", ThinkingLevel.LOW)
        assert mgr.cycle_level("s1") == ThinkingLevel.MEDIUM
        assert mgr.cycle_level("s1") == ThinkingLevel.HIGH
        assert mgr.cycle_level("s1") == ThinkingLevel.XHIGH
        assert mgr.cycle_level("s1") == ThinkingLevel.LOW

    def test_format_status(self):
        from app.agent.thinking_levels import ThinkingManager, ThinkingLevel
        mgr = ThinkingManager()
        mgr.set_level("s1", ThinkingLevel.XHIGH)
        status = mgr.format_status("s1")
        assert "xhigh" in status
        assert "Maximum reasoning" in status

    def test_remove_config(self):
        from app.agent.thinking_levels import ThinkingManager, ThinkingLevel
        mgr = ThinkingManager()
        mgr.set_level("s1", ThinkingLevel.HIGH)
        assert mgr.remove_config("s1") == True
        assert mgr.get_level("s1") == ThinkingLevel.MEDIUM

    def test_list_configs(self):
        from app.agent.thinking_levels import ThinkingManager, ThinkingLevel
        mgr = ThinkingManager()
        mgr.set_level("s1", ThinkingLevel.HIGH)
        mgr.set_level("s2", ThinkingLevel.LOW)
        configs = mgr.list_configs()
        assert len(configs) == 2

    def test_list_levels(self):
        from app.agent.thinking_levels import ThinkingManager
        mgr = ThinkingManager()
        levels = mgr.list_levels()
        assert len(levels) == 4
        assert any(l["level"] == "xhigh" for l in levels)

    def test_change_count(self):
        from app.agent.thinking_levels import ThinkingManager, ThinkingLevel
        mgr = ThinkingManager()
        mgr.set_level("s1", ThinkingLevel.LOW)
        mgr.set_level("s1", ThinkingLevel.HIGH)
        config = mgr.get_config("s1")
        assert config.change_count == 1

    def test_stats(self):
        from app.agent.thinking_levels import ThinkingManager, ThinkingLevel
        mgr = ThinkingManager()
        mgr.set_level("s1", ThinkingLevel.HIGH)
        s = mgr.stats()
        assert s["total_configs"] == 1

    def test_default_params(self):
        from app.agent.thinking_levels import ThinkingManager, ThinkingLevel
        mgr = ThinkingManager()
        params = mgr.get_model_params("unset")
        assert params["thinking_budget"] == 1024  # medium default

    def test_singleton(self):
        from app.agent.thinking_levels import get_thinking_manager
        m1 = get_thinking_manager()
        m2 = get_thinking_manager()
        assert m1 is m2


# ── Verbose Mode ──────────────────────────────────────

class TestVerboseMode:
    def test_import(self):
        from app.agent.verbose_mode import VerboseManager, VerboseLevel
        assert VerboseManager is not None

    def test_default_off(self):
        from app.agent.verbose_mode import VerboseManager
        mgr = VerboseManager()
        assert mgr.is_verbose("any") == False

    def test_set_verbose(self):
        from app.agent.verbose_mode import VerboseManager
        mgr = VerboseManager()
        mgr.set_verbose("s1", True)
        assert mgr.is_verbose("s1") == True

    def test_toggle(self):
        from app.agent.verbose_mode import VerboseManager
        mgr = VerboseManager()
        assert mgr.toggle("s1") == True
        assert mgr.toggle("s1") == False

    def test_format_tool_call_basic(self):
        from app.agent.verbose_mode import VerboseManager, VerboseLevel
        mgr = VerboseManager()
        mgr.set_verbose("s1", True, level=VerboseLevel.BASIC)
        text = mgr.format_tool_call("web_search", {"query": "test"}, session_id="s1")
        assert "web_search" in text

    def test_format_tool_call_detailed(self):
        from app.agent.verbose_mode import VerboseManager, VerboseLevel
        mgr = VerboseManager()
        mgr.set_verbose("s1", True, level=VerboseLevel.DETAILED)
        text = mgr.format_tool_call("web_search", {"query": "test"}, session_id="s1")
        assert "query" in text

    def test_format_tool_result(self):
        from app.agent.verbose_mode import VerboseManager, VerboseLevel
        mgr = VerboseManager()
        mgr.set_verbose("s1", True, level=VerboseLevel.FULL)
        text = mgr.format_tool_result("exec", "output data", 150.0, session_id="s1")
        assert "exec" in text
        assert "150ms" in text

    def test_narration_count(self):
        from app.agent.verbose_mode import VerboseManager
        mgr = VerboseManager()
        mgr.set_verbose("s1", True)
        mgr.format_tool_call("t1", session_id="s1")
        mgr.format_tool_call("t2", session_id="s1")
        config = mgr.get_config("s1")
        assert config.narration_count == 2

    def test_format_status(self):
        from app.agent.verbose_mode import VerboseManager
        mgr = VerboseManager()
        mgr.set_verbose("s1", True)
        status = mgr.format_status("s1")
        assert "ON" in status

    def test_remove_config(self):
        from app.agent.verbose_mode import VerboseManager
        mgr = VerboseManager()
        mgr.set_verbose("s1", True)
        assert mgr.remove_config("s1") == True
        assert mgr.is_verbose("s1") == False

    def test_get_level(self):
        from app.agent.verbose_mode import VerboseManager, VerboseLevel
        mgr = VerboseManager()
        mgr.set_verbose("s1", True, level=VerboseLevel.DETAILED)
        assert mgr.get_level("s1") == VerboseLevel.DETAILED

    def test_stats(self):
        from app.agent.verbose_mode import VerboseManager
        mgr = VerboseManager()
        mgr.set_verbose("s1", True)
        s = mgr.stats()
        assert s["enabled"] == 1

    def test_singleton(self):
        from app.agent.verbose_mode import get_verbose_manager
        m1 = get_verbose_manager()
        m2 = get_verbose_manager()
        assert m1 is m2


# ── Idempotency ───────────────────────────────────────

class TestIdempotency:
    def test_import(self):
        from app.agent.idempotency import IdempotencyStore, get_idempotency_store
        assert IdempotencyStore is not None

    def test_acquire_new_key(self):
        from app.agent.idempotency import IdempotencyStore
        store = IdempotencyStore()
        assert store.acquire("key-1") == True  # New key

    def test_acquire_duplicate_key(self):
        from app.agent.idempotency import IdempotencyStore
        store = IdempotencyStore()
        store.acquire("key-1")
        assert store.acquire("key-1") == False  # Already exists

    def test_store_and_get_result(self):
        from app.agent.idempotency import IdempotencyStore
        store = IdempotencyStore()
        store.acquire("key-1")
        store.store_result("key-1", {"response": "ok"})
        result = store.get_result("key-1")
        assert result["response"] == "ok"

    def test_get_result_not_completed(self):
        from app.agent.idempotency import IdempotencyStore
        store = IdempotencyStore()
        store.acquire("key-1")
        assert store.get_result("key-1") is None  # Still pending

    def test_has_key(self):
        from app.agent.idempotency import IdempotencyStore
        store = IdempotencyStore()
        assert store.has_key("key-1") == False
        store.acquire("key-1")
        assert store.has_key("key-1") == True

    def test_mark_failed(self):
        from app.agent.idempotency import IdempotencyStore
        store = IdempotencyStore()
        store.acquire("key-1")
        store.mark_failed("key-1", "timeout")
        entry = store.get_entry("key-1")
        assert entry.status == "failed"

    def test_remove(self):
        from app.agent.idempotency import IdempotencyStore
        store = IdempotencyStore()
        store.acquire("key-1")
        assert store.remove("key-1") == True
        assert store.has_key("key-1") == False

    def test_generate_key(self):
        from app.agent.idempotency import IdempotencyStore
        store = IdempotencyStore()
        key1 = store.generate_key("session_1", "msg_1")
        key2 = store.generate_key("session_1", "msg_1")
        key3 = store.generate_key("session_1", "msg_2")
        assert key1 == key2  # Deterministic
        assert key1 != key3  # Different inputs

    def test_hit_count(self):
        from app.agent.idempotency import IdempotencyStore
        store = IdempotencyStore()
        store.acquire("key-1")
        store.acquire("key-1")
        store.acquire("key-1")
        entry = store.get_entry("key-1")
        assert entry.hit_count == 2  # 2 duplicates

    def test_list_entries(self):
        from app.agent.idempotency import IdempotencyStore
        store = IdempotencyStore()
        store.acquire("key-1")
        store.acquire("key-2")
        entries = store.list_entries()
        assert len(entries) == 2

    def test_cleanup(self):
        from app.agent.idempotency import IdempotencyStore
        store = IdempotencyStore()
        store.acquire("key-1")
        # Manually expire the entry
        entry = store.get_entry("key-1")
        entry.created_at -= 7200  # 2 hours ago
        entry.ttl_seconds = 3600  # 1 hour TTL → expired
        removed = store.cleanup()
        assert removed == 1

    def test_stats(self):
        from app.agent.idempotency import IdempotencyStore
        store = IdempotencyStore()
        store.acquire("key-1")
        s = store.stats()
        assert s["total_entries"] == 1

    def test_singleton(self):
        from app.agent.idempotency import get_idempotency_store
        s1 = get_idempotency_store()
        s2 = get_idempotency_store()
        assert s1 is s2


# ── Apply Patch ───────────────────────────────────────

class TestApplyPatch:
    def test_import(self):
        from app.agent.apply_patch import PatchTool, PatchParser
        assert PatchTool is not None

    def test_parse_simple_patch(self):
        from app.agent.apply_patch import PatchParser
        parser = PatchParser()
        patch = """--- a/file.txt
+++ b/file.txt
@@ -1,3 +1,3 @@
 line1
-line2
+line2_modified
 line3
"""
        files = parser.parse(patch)
        assert len(files) == 1
        assert files[0].old_path == "file.txt"
        assert len(files[0].hunks) == 1

    def test_parse_new_file(self):
        from app.agent.apply_patch import PatchParser
        parser = PatchParser()
        patch = """--- /dev/null
+++ b/new_file.txt
@@ -0,0 +1,2 @@
+hello
+world
"""
        files = parser.parse(patch)
        assert len(files) == 1
        assert files[0].is_new == True

    def test_parse_deleted_file(self):
        from app.agent.apply_patch import PatchParser
        parser = PatchParser()
        patch = """--- a/old_file.txt
+++ /dev/null
@@ -1,2 +0,0 @@
-goodbye
-world
"""
        files = parser.parse(patch)
        assert len(files) == 1
        assert files[0].is_deleted == True

    def test_dry_run(self):
        from app.agent.apply_patch import PatchTool
        tool = PatchTool()
        patch = """--- a/file.txt
+++ b/file.txt
@@ -1,3 +1,4 @@
 line1
-line2
+line2a
+line2b
 line3
"""
        result = tool.dry_run(patch)
        assert result.success == True
        assert result.files_modified == 1
        assert result.total_additions == 2
        assert result.total_deletions == 1

    def test_apply_new_file(self):
        from app.agent.apply_patch import PatchTool
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = PatchTool(workspace_root=tmpdir)
            patch = """--- /dev/null
+++ b/hello.txt
@@ -0,0 +1,2 @@
+hello
+world
"""
            result = tool.apply(patch)
            assert result.success == True
            assert result.files_created == 1
            assert os.path.exists(os.path.join(tmpdir, "hello.txt"))

    def test_apply_modify_file(self):
        from app.agent.apply_patch import PatchTool
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create initial file
            path = os.path.join(tmpdir, "test.txt")
            with open(path, 'w') as f:
                f.write("line1\nline2\nline3\n")

            tool = PatchTool(workspace_root=tmpdir)
            patch = """--- a/test.txt
+++ b/test.txt
@@ -1,3 +1,3 @@
 line1
-line2
+modified
 line3
"""
            result = tool.apply(patch)
            assert result.success == True
            assert result.files_modified == 1

    def test_apply_delete_file(self):
        from app.agent.apply_patch import PatchTool
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "delete_me.txt")
            with open(path, 'w') as f:
                f.write("bye\n")

            tool = PatchTool(workspace_root=tmpdir)
            patch = """--- a/delete_me.txt
+++ /dev/null
@@ -1 +0,0 @@
-bye
"""
            result = tool.apply(patch)
            assert result.success == True
            assert result.files_deleted == 1
            assert not os.path.exists(path)

    def test_patch_file_not_found(self):
        from app.agent.apply_patch import PatchTool
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = PatchTool(workspace_root=tmpdir)
            patch = """--- a/nonexistent.txt
+++ b/nonexistent.txt
@@ -1,2 +1,2 @@
 x
-y
+z
"""
            result = tool.apply(patch)
            assert result.success == False
            assert len(result.errors) > 0

    def test_parse_multi_hunk(self):
        from app.agent.apply_patch import PatchParser
        parser = PatchParser()
        patch = """--- a/file.txt
+++ b/file.txt
@@ -1,2 +1,2 @@
-old1
+new1
 mid
@@ -10,2 +10,2 @@
-old10
+new10
 end
"""
        files = parser.parse(patch)
        assert len(files[0].hunks) == 2

    def test_result_to_dict(self):
        from app.agent.apply_patch import PatchResult
        r = PatchResult(success=True, files_modified=2, total_additions=5)
        d = r.to_dict()
        assert d["success"] == True
        assert d["total_additions"] == 5


# ── TTS Preferences ──────────────────────────────────

class TestTTSPreferences:
    def test_import(self):
        from app.agent.tts_preferences import TTSPreferencesManager
        assert TTSPreferencesManager is not None

    def test_set_preferences(self):
        from app.agent.tts_preferences import TTSPreferencesManager
        mgr = TTSPreferencesManager()
        prefs = mgr.set_preferences("user1", voice="nova", speed=1.5)
        assert prefs.voice == "nova"
        assert prefs.speed == 1.5

    def test_get_defaults(self):
        from app.agent.tts_preferences import TTSPreferencesManager
        mgr = TTSPreferencesManager()
        prefs = mgr.get_preferences("unknown_user")
        assert prefs.voice == "alloy"
        assert prefs.speed == 1.0

    def test_has_preferences(self):
        from app.agent.tts_preferences import TTSPreferencesManager
        mgr = TTSPreferencesManager()
        assert mgr.has_preferences("u1") == False
        mgr.set_preferences("u1", voice="echo")
        assert mgr.has_preferences("u1") == True

    def test_update_partial(self):
        from app.agent.tts_preferences import TTSPreferencesManager
        mgr = TTSPreferencesManager()
        mgr.set_preferences("u1", voice="nova", speed=1.0)
        mgr.set_preferences("u1", speed=1.5)  # Only update speed
        prefs = mgr.get_preferences("u1")
        assert prefs.voice == "nova"  # Unchanged
        assert prefs.speed == 1.5

    def test_speed_clamping(self):
        from app.agent.tts_preferences import TTSPreferencesManager
        mgr = TTSPreferencesManager()
        mgr.set_preferences("u1", speed=10.0)  # Over max
        assert mgr.get_preferences("u1").speed == 4.0

    def test_remove_preferences(self):
        from app.agent.tts_preferences import TTSPreferencesManager
        mgr = TTSPreferencesManager()
        mgr.set_preferences("u1", voice="echo")
        assert mgr.remove_preferences("u1") == True
        assert mgr.has_preferences("u1") == False

    def test_set_defaults(self):
        from app.agent.tts_preferences import TTSPreferencesManager
        mgr = TTSPreferencesManager()
        mgr.set_defaults(voice="shimmer", speed=0.8)
        prefs = mgr.get_preferences("any")
        assert prefs.voice == "shimmer"

    def test_available_voices(self):
        from app.agent.tts_preferences import TTSPreferencesManager
        mgr = TTSPreferencesManager()
        voices = mgr.get_available_voices()
        assert "openai" in voices
        assert "alloy" in voices["openai"]

    def test_validate_voice(self):
        from app.agent.tts_preferences import TTSPreferencesManager
        mgr = TTSPreferencesManager()
        assert mgr.validate_voice("openai", "alloy") == True
        assert mgr.validate_voice("openai", "nonexistent") == False

    def test_format_preferences(self):
        from app.agent.tts_preferences import TTSPreferencesManager
        mgr = TTSPreferencesManager()
        mgr.set_preferences("u1", voice="nova")
        text = mgr.format_preferences("u1")
        assert "nova" in text
        assert "custom" in text

    def test_list_users(self):
        from app.agent.tts_preferences import TTSPreferencesManager
        mgr = TTSPreferencesManager()
        mgr.set_preferences("u1", voice="nova")
        mgr.set_preferences("u2", voice="echo")
        users = mgr.list_users()
        assert len(users) == 2

    def test_stats(self):
        from app.agent.tts_preferences import TTSPreferencesManager
        mgr = TTSPreferencesManager()
        mgr.set_preferences("u1", voice="nova", provider="openai")
        s = mgr.stats()
        assert s["total_users"] == 1

    def test_singleton(self):
        from app.agent.tts_preferences import get_tts_prefs_manager
        m1 = get_tts_prefs_manager()
        m2 = get_tts_prefs_manager()
        assert m1 is m2


# ── DM Pairing ────────────────────────────────────────

class TestDMPairing:
    def test_import(self):
        from app.agent.dm_pairing import PairingManager, DMPolicy
        assert PairingManager is not None

    def test_create_pairing(self):
        from app.agent.dm_pairing import PairingManager
        mgr = PairingManager()
        code = mgr.create_pairing("telegram", "user_123")
        assert len(code.code) == 8
        assert code.status == "pending"

    def test_approve_pairing(self):
        from app.agent.dm_pairing import PairingManager
        mgr = PairingManager()
        code = mgr.create_pairing("telegram", "user_123")
        assert mgr.approve_pairing(code.code) == True
        assert mgr.is_paired("telegram", "user_123") == True

    def test_reject_pairing(self):
        from app.agent.dm_pairing import PairingManager
        mgr = PairingManager()
        code = mgr.create_pairing("telegram", "user_123")
        assert mgr.reject_pairing(code.code) == True
        assert mgr.is_paired("telegram", "user_123") == False

    def test_check_access_open(self):
        from app.agent.dm_pairing import PairingManager, DMPolicy
        mgr = PairingManager()
        mgr.set_policy("telegram", DMPolicy.OPEN)
        assert mgr.check_access("telegram", "anyone") == True

    def test_check_access_disabled(self):
        from app.agent.dm_pairing import PairingManager, DMPolicy
        mgr = PairingManager()
        mgr.set_policy("telegram", DMPolicy.DISABLED)
        assert mgr.check_access("telegram", "anyone") == False

    def test_check_access_pairing(self):
        from app.agent.dm_pairing import PairingManager, DMPolicy
        mgr = PairingManager()
        mgr.set_policy("telegram", DMPolicy.PAIRING)
        assert mgr.check_access("telegram", "user_1") == False  # Not paired
        mgr.add_to_allowlist("telegram", "user_1")
        assert mgr.check_access("telegram", "user_1") == True

    def test_add_remove_allowlist(self):
        from app.agent.dm_pairing import PairingManager
        mgr = PairingManager()
        mgr.add_to_allowlist("telegram", "user_1")
        assert mgr.is_paired("telegram", "user_1") == True
        assert mgr.remove_from_allowlist("telegram", "user_1") == True
        assert mgr.is_paired("telegram", "user_1") == False

    def test_list_pending(self):
        from app.agent.dm_pairing import PairingManager
        mgr = PairingManager()
        mgr.create_pairing("telegram", "u1")
        mgr.create_pairing("discord", "u2")
        pending = mgr.list_pending()
        assert len(pending) == 2

    def test_list_paired_users(self):
        from app.agent.dm_pairing import PairingManager
        mgr = PairingManager()
        mgr.add_to_allowlist("telegram", "u1")
        mgr.add_to_allowlist("telegram", "u2")
        users = mgr.list_paired_users("telegram")
        assert len(users) == 2

    def test_approve_invalid_code(self):
        from app.agent.dm_pairing import PairingManager
        mgr = PairingManager()
        assert mgr.approve_pairing("INVALID") == False

    def test_policy_per_channel(self):
        from app.agent.dm_pairing import PairingManager, DMPolicy
        mgr = PairingManager()
        mgr.set_policy("telegram", DMPolicy.OPEN)
        mgr.set_policy("discord", DMPolicy.DISABLED)
        assert mgr.get_policy("telegram") == DMPolicy.OPEN
        assert mgr.get_policy("discord") == DMPolicy.DISABLED

    def test_stats(self):
        from app.agent.dm_pairing import PairingManager
        mgr = PairingManager()
        mgr.create_pairing("telegram", "u1")
        s = mgr.stats()
        assert s["pending"] == 1

    def test_singleton(self):
        from app.agent.dm_pairing import get_pairing_manager
        m1 = get_pairing_manager()
        m2 = get_pairing_manager()
        assert m1 is m2
