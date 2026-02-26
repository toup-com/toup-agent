"""
Layer 10 Tests — Tool Timeout, Block Streaming, Onboarding Wizard,
Config Migration, Updater, Voice Directives.
"""

import asyncio
import json
import time
import pytest

# ── Tool Timeout ──────────────────────────────────────

class TestToolTimeout:
    def test_import(self):
        from app.agent.tool_timeout import ToolTimeoutManager, get_timeout_manager
        assert ToolTimeoutManager is not None

    def test_default_timeout(self):
        from app.agent.tool_timeout import ToolTimeoutManager
        mgr = ToolTimeoutManager(default_timeout=30.0)
        assert mgr.default_timeout == 30.0

    def test_set_timeout(self):
        from app.agent.tool_timeout import ToolTimeoutManager
        mgr = ToolTimeoutManager()
        mgr.set_timeout("web_search", 15.0)
        assert mgr.get_timeout("web_search") == 15.0

    def test_get_timeout_default(self):
        from app.agent.tool_timeout import ToolTimeoutManager
        mgr = ToolTimeoutManager(default_timeout=60.0)
        assert mgr.get_timeout("unknown_tool") == 60.0

    def test_remove_timeout(self):
        from app.agent.tool_timeout import ToolTimeoutManager
        mgr = ToolTimeoutManager()
        mgr.set_timeout("exec", 120.0)
        assert mgr.remove_timeout("exec") == True
        assert mgr.remove_timeout("exec") == False

    def test_execute_with_timeout_success(self):
        from app.agent.tool_timeout import ToolTimeoutManager
        mgr = ToolTimeoutManager()
        mgr.set_timeout("fast_tool", 5.0)

        async def fast_op():
            return "done"

        result = asyncio.get_event_loop().run_until_complete(
            mgr.execute_with_timeout("fast_tool", fast_op())
        )
        assert result == "done"

    def test_execute_with_timeout_exceeds(self):
        from app.agent.tool_timeout import ToolTimeoutManager
        mgr = ToolTimeoutManager()
        mgr.set_timeout("slow_tool", 0.1)

        async def slow_op():
            await asyncio.sleep(5)
            return "done"

        with pytest.raises(asyncio.TimeoutError):
            asyncio.get_event_loop().run_until_complete(
                mgr.execute_with_timeout("slow_tool", slow_op())
            )

    def test_timeout_history(self):
        from app.agent.tool_timeout import ToolTimeoutManager
        mgr = ToolTimeoutManager()
        mgr.set_timeout("tool", 0.05)

        async def slow():
            await asyncio.sleep(5)

        try:
            asyncio.get_event_loop().run_until_complete(
                mgr.execute_with_timeout("tool", slow())
            )
        except asyncio.TimeoutError:
            pass

        history = mgr.get_history()
        assert len(history) == 1
        assert history[0]["timed_out"] == True

    def test_timeout_stats(self):
        from app.agent.tool_timeout import ToolTimeoutManager
        mgr = ToolTimeoutManager()
        mgr.set_timeout("t1", 0.05)

        async def slow():
            await asyncio.sleep(5)

        try:
            asyncio.get_event_loop().run_until_complete(
                mgr.execute_with_timeout("t1", slow())
            )
        except asyncio.TimeoutError:
            pass

        stats = mgr.get_timeout_stats()
        assert stats["total_timeouts"] >= 1

    def test_list_configured(self):
        from app.agent.tool_timeout import ToolTimeoutManager
        mgr = ToolTimeoutManager()
        mgr.set_timeout("a", 10.0)
        mgr.set_timeout("b", 20.0)
        items = mgr.list_configured()
        assert len(items) == 2

    def test_clear_history(self):
        from app.agent.tool_timeout import ToolTimeoutManager
        mgr = ToolTimeoutManager()
        mgr._history.append(None)  # Dummy
        count = mgr.clear_history()
        assert count == 1

    def test_set_default_invalid(self):
        from app.agent.tool_timeout import ToolTimeoutManager
        mgr = ToolTimeoutManager()
        with pytest.raises(ValueError):
            mgr.default_timeout = -1

    def test_singleton(self):
        from app.agent.tool_timeout import get_timeout_manager
        m1 = get_timeout_manager()
        m2 = get_timeout_manager()
        assert m1 is m2


# ── Block Streaming ──────────────────────────────────

class TestBlockStreaming:
    def test_import(self):
        from app.agent.block_streaming import BlockStream, BlockType, BlockStreamRegistry
        assert BlockStream is not None

    def test_block_type_enum(self):
        from app.agent.block_streaming import BlockType
        assert BlockType.TEXT.value == "text"
        assert BlockType.TOOL_CALL.value == "tool_call"
        assert BlockType.THINKING.value == "thinking"

    def test_add_text(self):
        from app.agent.block_streaming import BlockStream
        stream = BlockStream(session_id="s1")
        stream.add_text("Hello ")
        stream.add_text("world!")
        assert stream.get_full_text() == "Hello world!"
        assert stream.block_count == 2

    def test_add_tool_call(self):
        from app.agent.block_streaming import BlockStream, BlockType
        stream = BlockStream()
        block = stream.add_tool_call("web_search", {"query": "test"})
        assert block.type == BlockType.TOOL_CALL
        assert block.content["tool"] == "web_search"

    def test_add_tool_result(self):
        from app.agent.block_streaming import BlockStream
        stream = BlockStream()
        block = stream.add_tool_result("web_search", {"results": []}, success=True)
        assert block.content["success"] == True

    def test_add_thinking(self):
        from app.agent.block_streaming import BlockStream, BlockType
        stream = BlockStream()
        block = stream.add_thinking("Let me consider...")
        assert block.type == BlockType.THINKING

    def test_add_error(self):
        from app.agent.block_streaming import BlockStream
        stream = BlockStream()
        stream.add_error("Something failed", tool_name="exec")
        errors = stream.get_errors()
        assert len(errors) == 1

    def test_add_status(self):
        from app.agent.block_streaming import BlockStream, BlockType
        stream = BlockStream()
        block = stream.add_status("Processing...")
        assert block.type == BlockType.STATUS

    def test_add_media(self):
        from app.agent.block_streaming import BlockStream, BlockType
        stream = BlockStream()
        block = stream.add_media("image", "https://example.com/img.png")
        assert block.type == BlockType.MEDIA
        assert block.content["url"] == "https://example.com/img.png"

    def test_finalize(self):
        from app.agent.block_streaming import BlockStream
        stream = BlockStream()
        stream.add_text("done")
        stream.finalize()
        assert stream.is_finalized == True
        with pytest.raises(RuntimeError):
            stream.add_text("more")

    def test_get_tool_calls(self):
        from app.agent.block_streaming import BlockStream
        stream = BlockStream()
        stream.add_text("thinking...")
        stream.add_tool_call("search", {"q": "a"})
        stream.add_tool_call("read", {"f": "b"})
        calls = stream.get_tool_calls()
        assert len(calls) == 2

    def test_to_dict(self):
        from app.agent.block_streaming import BlockStream
        stream = BlockStream(session_id="test")
        stream.add_text("hi")
        d = stream.to_dict()
        assert d["session_id"] == "test"
        assert d["block_count"] == 1

    def test_summary(self):
        from app.agent.block_streaming import BlockStream
        stream = BlockStream()
        stream.add_text("a")
        stream.add_tool_call("t", {})
        s = stream.summary()
        assert s["total_blocks"] == 2
        assert "text" in s["by_type"]

    def test_listener(self):
        from app.agent.block_streaming import BlockStream
        stream = BlockStream()
        received = []
        stream.on_block(lambda b: received.append(b))
        stream.add_text("hello")
        assert len(received) == 1

    def test_registry(self):
        from app.agent.block_streaming import BlockStreamRegistry
        reg = BlockStreamRegistry()
        s = reg.create("s1")
        assert reg.get("s1") is s
        assert reg.count == 1
        assert reg.list_active() == ["s1"]
        s.finalize()
        assert reg.list_active() == []
        assert reg.remove("s1") == True

    def test_singleton(self):
        from app.agent.block_streaming import get_block_stream_registry
        r1 = get_block_stream_registry()
        r2 = get_block_stream_registry()
        assert r1 is r2


# ── Onboarding Wizard ────────────────────────────────

class TestOnboardingWizard:
    def test_import(self):
        from app.agent.onboarding import WizardStep, WizardResult, get_wizard_steps
        assert WizardStep is not None

    def test_get_wizard_steps(self):
        from app.agent.onboarding import get_wizard_steps
        steps = get_wizard_steps()
        assert len(steps) >= 5
        keys = [s.key for s in steps]
        assert "database_url" in keys
        assert "openai_api_key" in keys

    def test_validate_url(self):
        from app.agent.onboarding import validate_url
        ok, _ = validate_url("postgresql+asyncpg://localhost/db")
        assert ok == True
        ok, _ = validate_url("http://localhost")
        assert ok == False

    def test_validate_api_key(self):
        from app.agent.onboarding import validate_api_key
        ok, _ = validate_api_key("sk-1234567890abcdef")
        assert ok == True
        ok, _ = validate_api_key("short")
        assert ok == False

    def test_validate_bot_token(self):
        from app.agent.onboarding import validate_bot_token
        ok, _ = validate_bot_token("123456:ABCDEF")
        assert ok == True
        ok, _ = validate_bot_token("invalid")
        assert ok == False

    def test_validate_port(self):
        from app.agent.onboarding import validate_port
        ok, _ = validate_port("8000")
        assert ok == True
        ok, _ = validate_port("99999")
        assert ok == False
        ok, _ = validate_port("abc")
        assert ok == False

    def test_generate_jwt_secret(self):
        from app.agent.onboarding import generate_jwt_secret
        s = generate_jwt_secret()
        assert len(s) == 64
        s2 = generate_jwt_secret(32)
        assert len(s2) == 32

    def test_run_wizard_noninteractive(self):
        from app.agent.onboarding import run_wizard_noninteractive
        result = run_wizard_noninteractive({
            "database_url": "postgresql+asyncpg://localhost/db",
            "admin_password": "secret123",
            "openai_api_key": "sk-test1234567890",
        })
        assert result.completed == True
        assert "DATABASE_URL" in result.env_vars
        assert "JWT_SECRET" in result.env_vars  # Auto-generated

    def test_run_wizard_missing_required(self):
        from app.agent.onboarding import run_wizard_noninteractive
        result = run_wizard_noninteractive({})
        # admin_password is required with no default
        assert "admin_password" in [e.split(":")[0] for e in result.errors]

    def test_wizard_result_to_env(self):
        from app.agent.onboarding import run_wizard_noninteractive
        result = run_wizard_noninteractive({
            "database_url": "postgresql+asyncpg://localhost/db",
            "admin_password": "pass123",
            "openai_api_key": "sk-test1234567890",
        })
        env_content = result.to_env_content()
        assert "DATABASE_URL=" in env_content
        assert "OPENAI_API_KEY=" in env_content

    def test_check_existing_config(self):
        from app.agent.onboarding import check_existing_config
        # Non-existent file
        result = check_existing_config("/tmp/nonexistent_env_file")
        assert all(v == False for v in result.values())

    def test_wizard_skips_optional(self):
        from app.agent.onboarding import run_wizard_noninteractive
        result = run_wizard_noninteractive({
            "database_url": "postgresql+asyncpg://localhost/db",
            "admin_password": "pass123",
            "openai_api_key": "sk-test1234567890",
        })
        assert "elevenlabs_api_key" in result.skipped


# ── Config Migration ─────────────────────────────────

class TestConfigMigration:
    def test_import(self):
        from app.agent.config_migration import ConfigMigrationManager, get_migration_manager
        assert ConfigMigrationManager is not None

    def test_builtin_migrations(self):
        from app.agent.config_migration import ConfigMigrationManager
        mgr = ConfigMigrationManager()
        assert mgr.migration_count >= 2

    def test_migrate_v1_to_v2(self):
        from app.agent.config_migration import ConfigMigrationManager
        mgr = ConfigMigrationManager()
        old = {"OPENAI_KEY": "sk-123", "TG_TOKEN": "123:ABC", "MODEL": "gpt-4"}
        result = mgr.migrate(old, from_version="1.0", to_version="2.0")
        assert result.success == True
        assert "OPENAI_API_KEY" in result.config
        assert "TELEGRAM_BOT_TOKEN" in result.config
        assert "AGENT_MODEL" in result.config

    def test_migrate_v1_to_v3(self):
        from app.agent.config_migration import ConfigMigrationManager
        mgr = ConfigMigrationManager()
        old = {"OPENAI_KEY": "sk-123"}
        result = mgr.migrate(old, from_version="1.0")
        assert result.final_version == "3.0"
        assert "THINKING_BUDGET_DEFAULT" in result.config
        assert len(result.applied_migrations) == 2

    def test_migrate_v2_to_v3(self):
        from app.agent.config_migration import ConfigMigrationManager
        mgr = ConfigMigrationManager()
        old = {"OPENAI_API_KEY": "sk-123"}
        result = mgr.migrate(old, from_version="2.0")
        assert result.final_version == "3.0"
        assert "DM_POLICY" in result.config

    def test_detect_version_v1(self):
        from app.agent.config_migration import ConfigMigrationManager
        mgr = ConfigMigrationManager()
        assert mgr.detect_version({"OPENAI_KEY": "x"}) == "1.0"

    def test_detect_version_v3(self):
        from app.agent.config_migration import ConfigMigrationManager
        mgr = ConfigMigrationManager()
        assert mgr.detect_version({"DM_POLICY": "open"}) == "3.0"

    def test_detect_version_v2(self):
        from app.agent.config_migration import ConfigMigrationManager
        mgr = ConfigMigrationManager()
        assert mgr.detect_version({"OPENAI_API_KEY": "x"}) == "2.0"

    def test_no_path(self):
        from app.agent.config_migration import ConfigMigrationManager
        mgr = ConfigMigrationManager()
        result = mgr.migrate({}, from_version="3.0")
        assert result.final_version == "3.0"
        assert len(result.applied_migrations) == 0

    def test_list_migrations(self):
        from app.agent.config_migration import ConfigMigrationManager
        mgr = ConfigMigrationManager()
        migs = mgr.list_migrations()
        assert len(migs) >= 2
        assert migs[0]["from"] == "1.0"

    def test_migration_result_to_dict(self):
        from app.agent.config_migration import ConfigMigrationManager
        mgr = ConfigMigrationManager()
        result = mgr.migrate({"OPENAI_KEY": "x"}, from_version="1.0")
        d = result.to_dict()
        assert d["success"] == True
        assert d["original_version"] == "1.0"

    def test_custom_migration(self):
        from app.agent.config_migration import ConfigMigrationManager
        mgr = ConfigMigrationManager()

        def custom_mig(config):
            config["CUSTOM"] = "added"
            return config

        mgr.register_migration("3.0", "4.0", custom_mig, "Add CUSTOM key")
        result = mgr.migrate({"OPENAI_KEY": "x"}, from_version="1.0", to_version="4.0")
        assert "CUSTOM" in result.config
        assert result.final_version == "4.0"

    def test_singleton(self):
        from app.agent.config_migration import get_migration_manager
        m1 = get_migration_manager()
        m2 = get_migration_manager()
        assert m1 is m2


# ── Updater ──────────────────────────────────────────

class TestUpdater:
    def test_import(self):
        from app.agent.updater import Updater, UpdateStrategy, UpdateState, get_updater
        assert Updater is not None

    def test_update_strategy_enum(self):
        from app.agent.updater import UpdateStrategy
        assert UpdateStrategy.GIT.value == "git"
        assert UpdateStrategy.DOCKER.value == "docker"

    def test_update_state_enum(self):
        from app.agent.updater import UpdateState
        assert UpdateState.IDLE.value == "idle"
        assert UpdateState.AVAILABLE.value == "available"

    def test_current_version(self):
        from app.agent.updater import Updater
        u = Updater()
        assert u.current_version == "1.0.0"

    def test_status(self):
        from app.agent.updater import Updater
        u = Updater()
        s = u.status()
        assert s["current_version"] == "1.0.0"
        assert s["state"] == "idle"
        assert s["strategy"] == "git"

    def test_set_strategy(self):
        from app.agent.updater import Updater, UpdateStrategy
        u = Updater()
        u.strategy = UpdateStrategy.DOCKER
        assert u.strategy == UpdateStrategy.DOCKER

    def test_version_info(self):
        from app.agent.updater import VersionInfo
        v = VersionInfo(version="2.0.0", release_date="2026-02-17")
        d = v.to_dict()
        assert d["version"] == "2.0.0"

    def test_update_result(self):
        from app.agent.updater import UpdateResult
        r = UpdateResult(
            success=True,
            from_version="1.0.0",
            to_version="2.0.0",
            strategy="git",
        )
        d = r.to_dict()
        assert d["success"] == True

    def test_get_update_history_empty(self):
        from app.agent.updater import Updater
        u = Updater()
        assert u.get_update_history() == []

    def test_check_for_updates(self):
        from app.agent.updater import Updater, UpdateStrategy
        u = Updater(strategy=UpdateStrategy.MANUAL)
        result = asyncio.get_event_loop().run_until_complete(u.check_for_updates())
        assert "current_version" in result

    def test_apply_update_docker(self):
        from app.agent.updater import Updater, UpdateStrategy
        u = Updater(strategy=UpdateStrategy.DOCKER)
        result = asyncio.get_event_loop().run_until_complete(u.apply_update("2.0.0"))
        assert result.success == False  # Docker requires manual
        assert "docker compose" in result.message

    def test_apply_update_manual(self):
        from app.agent.updater import Updater, UpdateStrategy
        u = Updater(strategy=UpdateStrategy.MANUAL)
        result = asyncio.get_event_loop().run_until_complete(u.apply_update("2.0.0"))
        assert result.success == False
        assert "Manual" in result.message

    def test_singleton(self):
        from app.agent.updater import get_updater
        u1 = get_updater()
        u2 = get_updater()
        assert u1 is u2


# ── Voice Directives ─────────────────────────────────

class TestVoiceDirectives:
    def test_import(self):
        from app.agent.voice_directives import parse_voice_directives, VoiceDirective
        assert parse_voice_directives is not None

    def test_parse_no_directives(self):
        from app.agent.voice_directives import parse_voice_directives
        result = parse_voice_directives("Hello world!")
        assert result.clean_text == "Hello world!"
        assert result.has_directives == False

    def test_parse_voice(self):
        from app.agent.voice_directives import parse_voice_directives
        result = parse_voice_directives("Hello [voice:nova] world!")
        assert result.voice == "nova"
        assert "nova" not in result.clean_text

    def test_parse_speed(self):
        from app.agent.voice_directives import parse_voice_directives
        result = parse_voice_directives("[speed:1.5] Fast talking")
        assert result.speed == 1.5

    def test_parse_speed_clamped(self):
        from app.agent.voice_directives import parse_voice_directives
        result = parse_voice_directives("[speed:10.0] Too fast")
        assert result.speed == 4.0  # Clamped to max

    def test_parse_pause(self):
        from app.agent.voice_directives import parse_voice_directives
        result = parse_voice_directives("Wait [pause:500] here")
        assert 500 in result.pauses

    def test_parse_emotion(self):
        from app.agent.voice_directives import parse_voice_directives
        result = parse_voice_directives("[emotion:excited] Great news!")
        assert result.emotion == "excited"

    def test_parse_language(self):
        from app.agent.voice_directives import parse_voice_directives
        result = parse_voice_directives("[language:es] Hola mundo")
        assert result.language == "es"

    def test_parse_whisper(self):
        from app.agent.voice_directives import parse_voice_directives
        result = parse_voice_directives("[whisper] Secret message")
        assert result.whisper == True

    def test_parse_sing(self):
        from app.agent.voice_directives import parse_voice_directives
        result = parse_voice_directives("[sing] La la la")
        assert result.sing == True

    def test_parse_multiple(self):
        from app.agent.voice_directives import parse_voice_directives
        result = parse_voice_directives("[voice:alloy] [speed:0.8] Slow and steady")
        assert result.voice == "alloy"
        assert result.speed == 0.8
        assert len(result.directives) == 2

    def test_clean_text(self):
        from app.agent.voice_directives import parse_voice_directives
        result = parse_voice_directives("Hello [voice:nova] beautiful [speed:1.2] world")
        assert result.clean_text == "Hello beautiful world"

    def test_to_dict(self):
        from app.agent.voice_directives import parse_voice_directives
        result = parse_voice_directives("[voice:echo] Test")
        d = result.to_dict()
        assert d["voice"] == "echo"
        assert len(d["directives"]) == 1

    def test_build_tts_config(self):
        from app.agent.voice_directives import parse_voice_directives, build_tts_config
        result = parse_voice_directives("[voice:nova] [speed:1.3] Text")
        config = build_tts_config(result, defaults={"model": "tts-1"})
        assert config["voice"] == "nova"
        assert config["speed"] == 1.3
        assert config["model"] == "tts-1"

    def test_validate_directive_valid(self):
        from app.agent.voice_directives import validate_directive
        ok, _ = validate_directive("voice", "alloy")
        assert ok == True

    def test_validate_directive_invalid_voice(self):
        from app.agent.voice_directives import validate_directive
        ok, err = validate_directive("voice", "nonexistent_voice")
        assert ok == False
        assert "Unknown voice" in err

    def test_validate_directive_speed_range(self):
        from app.agent.voice_directives import validate_directive
        ok, _ = validate_directive("speed", "1.5")
        assert ok == True
        ok, err = validate_directive("speed", "10.0")
        assert ok == False

    def test_validate_directive_unknown_type(self):
        from app.agent.voice_directives import validate_directive
        ok, err = validate_directive("unknown", "value")
        assert ok == False

    def test_case_insensitive(self):
        from app.agent.voice_directives import parse_voice_directives
        result = parse_voice_directives("[Voice:Nova] hello")
        assert result.voice == "nova"
