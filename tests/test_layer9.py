"""
Layer 9 Tests — Model Failover, Presence Tracking, Channel Config,
Transcript Export, Health Probes, Webhook Triggers.
"""

import asyncio
import time
import json
import pytest

# ── Model Failover ──────────────────────────────────────

class TestModelFailover:
    def test_import(self):
        from app.agent.model_failover import FailoverChain, ProviderStatus, get_failover_chain
        assert FailoverChain is not None

    def test_provider_status_enum(self):
        from app.agent.model_failover import ProviderStatus
        assert ProviderStatus.HEALTHY.value == "healthy"
        assert ProviderStatus.DOWN.value == "down"
        assert ProviderStatus.COOLDOWN.value == "cooldown"

    def test_configure_chain(self):
        from app.agent.model_failover import FailoverChain
        chain = FailoverChain()
        chain.configure([
            {"model": "claude-sonnet-4-20250514", "provider": "anthropic"},
            {"model": "gpt-4o", "provider": "openai"},
        ])
        assert chain.chain_length == 2

    def test_next_available(self):
        from app.agent.model_failover import FailoverChain
        chain = FailoverChain()
        chain.configure([
            {"model": "model-a", "provider": "prov-a", "priority": 0},
            {"model": "model-b", "provider": "prov-b", "priority": 1},
        ])
        result = asyncio.get_event_loop().run_until_complete(chain.next_available())
        assert result == ("model-a", "prov-a")

    def test_next_available_skip(self):
        from app.agent.model_failover import FailoverChain
        chain = FailoverChain()
        chain.configure([
            {"model": "model-a", "provider": "prov-a"},
            {"model": "model-b", "provider": "prov-b"},
        ])
        result = asyncio.get_event_loop().run_until_complete(
            chain.next_available(skip=["prov-a/model-a"])
        )
        assert result == ("model-b", "prov-b")

    def test_record_success(self):
        from app.agent.model_failover import FailoverChain, ProviderStatus
        chain = FailoverChain()
        chain.configure([{"model": "m1", "provider": "p1"}])
        chain.record_success("m1", "p1", latency_ms=50.0)
        health = chain.get_health()
        assert len(health) == 1
        assert health[0]["status"] == "healthy"
        assert health[0]["avg_latency_ms"] == 50.0

    def test_record_failure_degraded(self):
        from app.agent.model_failover import FailoverChain, ProviderStatus
        chain = FailoverChain()
        chain.configure([{"model": "m1", "provider": "p1"}])
        chain.record_failure("m1", "p1", "timeout")
        chain.record_failure("m1", "p1", "timeout")
        health = chain.get_health()
        assert health[0]["status"] == "degraded"

    def test_record_failure_down(self):
        from app.agent.model_failover import FailoverChain
        chain = FailoverChain()
        chain.configure([{"model": "m1", "provider": "p1"}])
        for i in range(5):
            chain.record_failure("m1", "p1", "error")
        health = chain.get_health()
        assert health[0]["status"] == "down"
        assert health[0]["available"] == False

    def test_failover_skips_down(self):
        from app.agent.model_failover import FailoverChain
        chain = FailoverChain()
        chain.configure([
            {"model": "m1", "provider": "p1"},
            {"model": "m2", "provider": "p2"},
        ])
        for i in range(5):
            chain.record_failure("m1", "p1", "down")
        result = asyncio.get_event_loop().run_until_complete(chain.next_available())
        assert result == ("m2", "p2")

    def test_reset_health(self):
        from app.agent.model_failover import FailoverChain
        chain = FailoverChain()
        chain.configure([{"model": "m1", "provider": "p1"}])
        chain.record_failure("m1", "p1", "err")
        count = chain.reset_health()
        assert count == 1
        health = chain.get_health()
        assert health[0]["status"] == "healthy"

    def test_get_chain(self):
        from app.agent.model_failover import FailoverChain
        chain = FailoverChain()
        chain.configure([
            {"model": "m1", "provider": "p1", "timeout_seconds": 60},
        ])
        info = chain.get_chain()
        assert len(info) == 1
        assert info[0]["model"] == "m1"
        assert info[0]["timeout_seconds"] == 60.0

    def test_singleton(self):
        from app.agent.model_failover import get_failover_chain
        c1 = get_failover_chain()
        c2 = get_failover_chain()
        assert c1 is c2


# ── Presence Tracking ──────────────────────────────────

class TestPresenceTracker:
    def test_import(self):
        from app.agent.presence import PresenceTracker, ConnectionState, get_presence_tracker
        assert PresenceTracker is not None

    def test_connection_state_enum(self):
        from app.agent.presence import ConnectionState
        assert ConnectionState.CONNECTED.value == "connected"
        assert ConnectionState.ERROR.value == "error"

    def test_update_connect(self):
        from app.agent.presence import PresenceTracker
        tracker = PresenceTracker()
        tracker.update("telegram", connected=True)
        status = tracker.get_status("telegram")
        assert status["state"] == "connected"

    def test_update_disconnect(self):
        from app.agent.presence import PresenceTracker
        tracker = PresenceTracker()
        tracker.update("telegram", connected=True)
        tracker.update("telegram", connected=False)
        status = tracker.get_status("telegram")
        assert status["state"] == "disconnected"

    def test_update_error(self):
        from app.agent.presence import PresenceTracker
        tracker = PresenceTracker()
        tracker.update("telegram", error="Connection refused")
        status = tracker.get_status("telegram")
        assert status["state"] == "error"
        assert status["error"] == "Connection refused"

    def test_message_count(self):
        from app.agent.presence import PresenceTracker
        tracker = PresenceTracker()
        tracker.update("telegram", connected=True)
        tracker.update("telegram", message=True)
        tracker.update("telegram", message=True)
        status = tracker.get_status("telegram")
        assert status["message_count"] == 2

    def test_reconnect_count(self):
        from app.agent.presence import PresenceTracker
        tracker = PresenceTracker()
        tracker.update("telegram", connected=True, message=True)
        tracker.update("telegram", connected=False)
        tracker.update("telegram", connected=True)
        status = tracker.get_status("telegram")
        assert status["reconnect_count"] == 1

    def test_get_all(self):
        from app.agent.presence import PresenceTracker
        tracker = PresenceTracker()
        tracker.update("telegram", connected=True)
        tracker.update("discord", connected=False)
        all_status = tracker.get_all()
        assert len(all_status) == 2

    def test_connected_channels(self):
        from app.agent.presence import PresenceTracker
        tracker = PresenceTracker()
        tracker.update("telegram", connected=True)
        tracker.update("discord", connected=False)
        tracker.update("slack", connected=True)
        assert sorted(tracker.connected_channels()) == ["slack", "telegram"]

    def test_is_connected(self):
        from app.agent.presence import PresenceTracker
        tracker = PresenceTracker()
        tracker.update("telegram", connected=True)
        assert tracker.is_connected("telegram") == True
        assert tracker.is_connected("discord") == False

    def test_summary(self):
        from app.agent.presence import PresenceTracker
        tracker = PresenceTracker()
        tracker.update("telegram", connected=True)
        tracker.update("discord", error="fail")
        tracker.update("slack", connected=True)
        s = tracker.summary()
        assert s["total_channels"] == 3
        assert s["connected"] == 2
        assert s["errored"] == 1

    def test_probe(self):
        from app.agent.presence import PresenceTracker
        tracker = PresenceTracker()
        tracker.update("telegram", connected=True)

        async def tg_probe(ch):
            return True

        tracker.register_probe("telegram", tg_probe)
        result = asyncio.get_event_loop().run_until_complete(tracker.run_probe("telegram"))
        assert result["ok"] == True
        assert "latency_ms" in result

    def test_probe_failure(self):
        from app.agent.presence import PresenceTracker
        tracker = PresenceTracker()
        tracker.update("discord", connected=True)

        async def bad_probe(ch):
            raise ConnectionError("Disconnected")

        tracker.register_probe("discord", bad_probe)
        result = asyncio.get_event_loop().run_until_complete(tracker.run_probe("discord"))
        assert result["ok"] == False
        assert "Disconnected" in result["error"]

    def test_clear(self):
        from app.agent.presence import PresenceTracker
        tracker = PresenceTracker()
        tracker.update("telegram", connected=True)
        tracker.clear("telegram")
        assert tracker.get_status("telegram") is None

    def test_singleton(self):
        from app.agent.presence import get_presence_tracker
        t1 = get_presence_tracker()
        t2 = get_presence_tracker()
        assert t1 is t2

    def test_details(self):
        from app.agent.presence import PresenceTracker
        tracker = PresenceTracker()
        tracker.update("telegram", connected=True, details={"bot_id": "123"})
        status = tracker.get_status("telegram")
        assert status["details"]["bot_id"] == "123"


# ── Channel Config ─────────────────────────────────────

class TestChannelConfig:
    def test_import(self):
        from app.agent.channel_config import ChannelConfigManager, get_channel_config_manager
        assert ChannelConfigManager is not None

    def test_get_defaults(self):
        from app.agent.channel_config import ChannelConfigManager
        mgr = ChannelConfigManager()
        cfg = mgr.get("telegram")
        assert cfg["enabled"] == True
        assert cfg["dm_policy"] == "open"
        assert cfg["delivery_mode"] == "gateway"

    def test_set_override(self):
        from app.agent.channel_config import ChannelConfigManager
        mgr = ChannelConfigManager()
        results = mgr.set("telegram", {"model": "gpt-4o", "dm_policy": "pairing"})
        assert results["model"] == "updated"
        cfg = mgr.get("telegram")
        assert cfg["model"] == "gpt-4o"
        assert cfg["dm_policy"] == "pairing"

    def test_set_unchanged(self):
        from app.agent.channel_config import ChannelConfigManager
        mgr = ChannelConfigManager()
        mgr.set("telegram", {"enabled": True})
        results = mgr.set("telegram", {"enabled": True})
        assert results["enabled"] == "unchanged"

    def test_unknown_key_warning(self):
        from app.agent.channel_config import ChannelConfigManager
        mgr = ChannelConfigManager()
        results = mgr.set("telegram", {"custom_field": "value"})
        assert results["custom_field"] == "warning:unknown_key"

    def test_reset(self):
        from app.agent.channel_config import ChannelConfigManager
        mgr = ChannelConfigManager()
        mgr.set("telegram", {"model": "gpt-4o"})
        assert mgr.reset("telegram") == True
        cfg = mgr.get("telegram")
        assert cfg["model"] is None

    def test_remove(self):
        from app.agent.channel_config import ChannelConfigManager
        mgr = ChannelConfigManager()
        mgr.get("telegram")
        assert mgr.remove("telegram") == True
        assert mgr.remove("telegram") == False

    def test_list_channels(self):
        from app.agent.channel_config import ChannelConfigManager
        mgr = ChannelConfigManager()
        mgr.get("telegram")
        mgr.get("discord")
        assert sorted(mgr.list_channels()) == ["discord", "telegram"]

    def test_get_value(self):
        from app.agent.channel_config import ChannelConfigManager
        mgr = ChannelConfigManager()
        assert mgr.get_value("telegram", "rate_limit_per_minute") == 30
        assert mgr.get_value("telegram", "nonexistent", "default") == "default"

    def test_get_channels_by_agent(self):
        from app.agent.channel_config import ChannelConfigManager
        mgr = ChannelConfigManager()
        mgr.set("telegram", {"agent": "coder"})
        mgr.set("discord", {"agent": "researcher"})
        mgr.set("slack", {"agent": "coder"})
        assert sorted(mgr.get_channels_by_agent("coder")) == ["slack", "telegram"]

    def test_diff_from_defaults(self):
        from app.agent.channel_config import ChannelConfigManager
        mgr = ChannelConfigManager()
        mgr.set("telegram", {"model": "gpt-4o", "enabled": True})
        diff = mgr.diff_from_defaults("telegram")
        assert "model" in diff
        assert "enabled" not in diff  # Same as default

    def test_validate_ok(self):
        from app.agent.channel_config import ChannelConfigManager
        mgr = ChannelConfigManager()
        issues = mgr.validate("telegram")
        assert len(issues) == 0

    def test_validate_bad(self):
        from app.agent.channel_config import ChannelConfigManager
        mgr = ChannelConfigManager()
        mgr.set("telegram", {"dm_policy": "invalid", "auto_tts": "bad"})
        issues = mgr.validate("telegram")
        assert len(issues) == 2

    def test_on_change_listener(self):
        from app.agent.channel_config import ChannelConfigManager
        mgr = ChannelConfigManager()
        changes = []
        mgr.on_change(lambda ch, overrides: changes.append((ch, overrides)))
        mgr.set("telegram", {"model": "gpt-4o"})
        assert len(changes) == 1
        assert changes[0][0] == "telegram"

    def test_singleton(self):
        from app.agent.channel_config import get_channel_config_manager
        m1 = get_channel_config_manager()
        m2 = get_channel_config_manager()
        assert m1 is m2


# ── Transcript Export ──────────────────────────────────

class TestTranscriptManager:
    def test_import(self):
        from app.agent.transcript import TranscriptManager, ExportFormat, get_transcript_manager
        assert TranscriptManager is not None

    def test_append(self):
        from app.agent.transcript import TranscriptManager
        mgr = TranscriptManager()
        idx = mgr.append("s1", "user", "Hello!")
        assert idx == 0
        idx2 = mgr.append("s1", "assistant", "Hi there!")
        assert idx2 == 1

    def test_get(self):
        from app.agent.transcript import TranscriptManager
        mgr = TranscriptManager()
        mgr.append("s1", "user", "test")
        t = mgr.get("s1")
        assert t is not None
        assert t.message_count == 1

    def test_get_nonexistent(self):
        from app.agent.transcript import TranscriptManager
        mgr = TranscriptManager()
        assert mgr.get("missing") is None

    def test_export_json(self):
        from app.agent.transcript import TranscriptManager
        mgr = TranscriptManager()
        mgr.append("s1", "user", "Hello")
        mgr.append("s1", "assistant", "World")
        out = mgr.export("s1", fmt="json")
        data = json.loads(out)
        assert data["message_count"] == 2
        assert len(data["messages"]) == 2

    def test_export_markdown(self):
        from app.agent.transcript import TranscriptManager
        mgr = TranscriptManager()
        mgr.append("s1", "user", "Hello")
        mgr.append("s1", "assistant", "World")
        out = mgr.export("s1", fmt="markdown")
        assert "# Session Transcript" in out
        assert "Hello" in out
        assert "World" in out

    def test_export_text(self):
        from app.agent.transcript import TranscriptManager
        mgr = TranscriptManager()
        mgr.append("s1", "user", "Hello")
        out = mgr.export("s1", fmt="text")
        assert "[user]" in out

    def test_export_nonexistent(self):
        from app.agent.transcript import TranscriptManager
        mgr = TranscriptManager()
        assert mgr.export("missing") is None

    def test_word_count(self):
        from app.agent.transcript import TranscriptManager
        mgr = TranscriptManager()
        mgr.append("s1", "user", "one two three")
        t = mgr.get("s1")
        assert t.word_count == 3

    def test_list_sessions(self):
        from app.agent.transcript import TranscriptManager
        mgr = TranscriptManager()
        mgr.append("s1", "user", "a")
        mgr.append("s2", "user", "b")
        sessions = mgr.list_sessions()
        assert len(sessions) == 2

    def test_delete(self):
        from app.agent.transcript import TranscriptManager
        mgr = TranscriptManager()
        mgr.append("s1", "user", "a")
        assert mgr.delete("s1") == True
        assert mgr.delete("s1") == False

    def test_clear(self):
        from app.agent.transcript import TranscriptManager
        mgr = TranscriptManager()
        mgr.append("s1", "user", "a")
        mgr.append("s2", "user", "b")
        assert mgr.clear() == 2
        assert mgr.session_count == 0

    def test_eviction(self):
        from app.agent.transcript import TranscriptManager
        mgr = TranscriptManager(max_sessions=2)
        mgr.append("s1", "user", "a")
        mgr.append("s2", "user", "b")
        mgr.append("s3", "user", "c")  # Should evict s1
        assert mgr.session_count == 2
        assert mgr.get("s1") is None

    def test_metadata(self):
        from app.agent.transcript import TranscriptManager
        mgr = TranscriptManager()
        mgr.append("s1", "user", "hello", metadata={"channel": "telegram"})
        t = mgr.get("s1")
        assert t.messages[0].metadata["channel"] == "telegram"

    def test_singleton(self):
        from app.agent.transcript import get_transcript_manager
        t1 = get_transcript_manager()
        t2 = get_transcript_manager()
        assert t1 is t2


# ── Health Probes ──────────────────────────────────────

class TestHealthRegistry:
    def test_import(self):
        from app.agent.health import HealthRegistry, HealthStatus, get_health_registry
        assert HealthRegistry is not None

    def test_health_status_enum(self):
        from app.agent.health import HealthStatus
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"

    def test_register_probe(self):
        from app.agent.health import HealthRegistry
        registry = HealthRegistry()

        async def check_db():
            return True

        registry.register("database", check_db)
        assert registry.probe_count == 1

    def test_register_decorator(self):
        from app.agent.health import HealthRegistry
        registry = HealthRegistry()

        @registry.probe("cache")
        async def check_cache():
            return True

        assert "cache" in registry.list_components()

    def test_check_healthy(self):
        from app.agent.health import HealthRegistry, HealthStatus
        registry = HealthRegistry()

        async def check_db():
            return True, {"version": "15.0"}

        registry.register("database", check_db)
        result = asyncio.get_event_loop().run_until_complete(registry.check("database"))
        assert result.status == HealthStatus.HEALTHY
        assert result.details.get("version") == "15.0"

    def test_check_unhealthy(self):
        from app.agent.health import HealthRegistry, HealthStatus
        registry = HealthRegistry()

        async def check_db():
            return False

        registry.register("database", check_db)
        result = asyncio.get_event_loop().run_until_complete(registry.check("database"))
        assert result.status == HealthStatus.UNHEALTHY

    def test_check_exception(self):
        from app.agent.health import HealthRegistry, HealthStatus
        registry = HealthRegistry()

        async def check_db():
            raise ConnectionError("refused")

        registry.register("database", check_db)
        result = asyncio.get_event_loop().run_until_complete(registry.check("database"))
        assert result.status == HealthStatus.UNHEALTHY
        assert "refused" in result.message

    def test_check_unknown(self):
        from app.agent.health import HealthRegistry, HealthStatus
        registry = HealthRegistry()
        result = asyncio.get_event_loop().run_until_complete(registry.check("nonexistent"))
        assert result.status == HealthStatus.UNKNOWN

    def test_run_all(self):
        from app.agent.health import HealthRegistry, HealthStatus
        registry = HealthRegistry()

        async def check_db():
            return True

        async def check_cache():
            return True

        registry.register("database", check_db)
        registry.register("cache", check_cache)
        report = asyncio.get_event_loop().run_until_complete(registry.run_all())
        assert report.overall_status == HealthStatus.HEALTHY
        assert report.healthy_count == 2
        assert report.total_count == 2

    def test_run_all_degraded(self):
        from app.agent.health import HealthRegistry, HealthStatus
        registry = HealthRegistry()

        async def check_db():
            return True

        async def check_cache():
            return False

        registry.register("database", check_db)
        registry.register("cache", check_cache)
        report = asyncio.get_event_loop().run_until_complete(registry.run_all())
        assert report.overall_status == HealthStatus.UNHEALTHY

    def test_report_to_dict(self):
        from app.agent.health import HealthRegistry
        registry = HealthRegistry()

        async def check_db():
            return True

        registry.register("database", check_db)
        report = asyncio.get_event_loop().run_until_complete(registry.run_all())
        d = report.to_dict()
        assert d["status"] == "healthy"
        assert "database" in d["components"]

    def test_unregister(self):
        from app.agent.health import HealthRegistry
        registry = HealthRegistry()

        async def check_db():
            return True

        registry.register("database", check_db)
        assert registry.unregister("database") == True
        assert registry.probe_count == 0

    def test_singleton(self):
        from app.agent.health import get_health_registry
        r1 = get_health_registry()
        r2 = get_health_registry()
        assert r1 is r2


# ── Webhook Triggers ──────────────────────────────────

class TestWebhookTriggers:
    def test_import(self):
        from app.agent.webhook_triggers import WebhookManager, WebhookRule, get_webhook_manager
        assert WebhookManager is not None

    def test_register_rule(self):
        from app.agent.webhook_triggers import WebhookManager
        mgr = WebhookManager()
        rule = mgr.register_rule("github_push", match={"event": "push"})
        assert rule.name == "github_push"
        assert mgr.rule_count == 1

    def test_match_simple(self):
        from app.agent.webhook_triggers import WebhookRule
        rule = WebhookRule(name="test", match={"event": "push"})
        assert rule.matches({"event": "push", "repo": "test"}) == True
        assert rule.matches({"event": "pull_request"}) == False

    def test_match_regex(self):
        from app.agent.webhook_triggers import WebhookRule
        rule = WebhookRule(name="test", match={"source": "regex:github|gitlab"})
        assert rule.matches({"source": "github"}) == True
        assert rule.matches({"source": "gitlab"}) == True
        assert rule.matches({"source": "bitbucket"}) == False

    def test_match_missing_key(self):
        from app.agent.webhook_triggers import WebhookRule
        rule = WebhookRule(name="test", match={"event": "push"})
        assert rule.matches({}) == False

    def test_build_prompt(self):
        from app.agent.webhook_triggers import WebhookRule
        rule = WebhookRule(
            name="test",
            match={},
            agent_prompt_template="New {event} from {repo}",
        )
        prompt = rule.build_prompt({"event": "push", "repo": "my-repo"})
        assert prompt == "New push from my-repo"

    def test_build_prompt_default(self):
        from app.agent.webhook_triggers import WebhookRule
        rule = WebhookRule(name="test", match={})
        prompt = rule.build_prompt({"key": "val"})
        assert "Webhook received" in prompt

    def test_process_match(self):
        from app.agent.webhook_triggers import WebhookManager
        mgr = WebhookManager()
        mgr.register_rule("test", match={"event": "push"}, action="agent")
        results = asyncio.get_event_loop().run_until_complete(
            mgr.process({"event": "push"})
        )
        assert len(results) == 1
        assert results[0].success == True
        assert results[0].action == "agent"

    def test_process_no_match(self):
        from app.agent.webhook_triggers import WebhookManager
        mgr = WebhookManager()
        mgr.register_rule("test", match={"event": "push"})
        results = asyncio.get_event_loop().run_until_complete(
            mgr.process({"event": "deploy"})
        )
        assert len(results) == 0

    def test_process_event_action(self):
        from app.agent.webhook_triggers import WebhookManager
        mgr = WebhookManager()
        mgr.register_rule("test", match={"type": "email"}, action="event", target_event="gmail.new_email")
        results = asyncio.get_event_loop().run_until_complete(
            mgr.process({"type": "email"})
        )
        assert len(results) == 1
        assert results[0].data["event"] == "gmail.new_email"

    def test_process_custom_handler(self):
        from app.agent.webhook_triggers import WebhookManager
        mgr = WebhookManager()

        async def my_handler(payload, rule):
            return {"processed": True}

        mgr.register_custom_handler("custom", my_handler)
        mgr.register_rule("test", match={"x": 1}, action="custom")
        results = asyncio.get_event_loop().run_until_complete(
            mgr.process({"x": 1})
        )
        assert len(results) == 1
        assert results[0].data["processed"] == True

    def test_unregister_rule(self):
        from app.agent.webhook_triggers import WebhookManager
        mgr = WebhookManager()
        mgr.register_rule("test", match={"a": 1})
        assert mgr.unregister_rule("test") == True
        assert mgr.rule_count == 0

    def test_verify_signature_no_secret(self):
        from app.agent.webhook_triggers import WebhookManager
        mgr = WebhookManager()
        assert mgr.verify_signature("test", b"data", "sig") == True

    def test_verify_signature_valid(self):
        import hashlib, hmac
        from app.agent.webhook_triggers import WebhookManager
        mgr = WebhookManager()
        mgr.set_secret("github", "my_secret")
        payload = b'{"event": "push"}'
        sig = hmac.new(b"my_secret", payload, hashlib.sha256).hexdigest()
        assert mgr.verify_signature("github", payload, sig, algorithm="sha256") == True

    def test_verify_signature_invalid(self):
        from app.agent.webhook_triggers import WebhookManager
        mgr = WebhookManager()
        mgr.set_secret("github", "my_secret")
        assert mgr.verify_signature("github", b"data", "bad_sig", algorithm="sha256") == False

    def test_list_rules(self):
        from app.agent.webhook_triggers import WebhookManager
        mgr = WebhookManager()
        mgr.register_rule("r1", match={"a": 1})
        mgr.register_rule("r2", match={"b": 2})
        rules = mgr.list_rules()
        assert len(rules) == 2

    def test_history(self):
        from app.agent.webhook_triggers import WebhookManager
        mgr = WebhookManager()
        mgr.register_rule("test", match={"x": 1})
        asyncio.get_event_loop().run_until_complete(mgr.process({"x": 1}))
        history = mgr.get_history()
        assert len(history) == 1
        assert "test" in history[0]["matched_rules"]

    def test_disabled_rule(self):
        from app.agent.webhook_triggers import WebhookManager
        mgr = WebhookManager()
        rule = mgr.register_rule("test", match={"x": 1})
        rule.enabled = False
        results = asyncio.get_event_loop().run_until_complete(mgr.process({"x": 1}))
        assert len(results) == 0

    def test_singleton(self):
        from app.agent.webhook_triggers import get_webhook_manager
        m1 = get_webhook_manager()
        m2 = get_webhook_manager()
        assert m1 is m2
