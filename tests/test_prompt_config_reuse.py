"""Owner mobile prompt reuse keeps the old prompt and saves two config reads."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

from app import config
from app.agent import agent_runner
from app.agent.prompt_profile import PromptProfile


OWNER = "871bac24-1111-4111-8111-111111111111"
OTHER = "871bac24-2222-4222-8222-222222222222"


class _Result:
    def __init__(self, value=None, rows=()):
        self.value = value
        self.rows = rows

    def scalars(self):
        return self

    def all(self):
        return list(self.rows)

    def scalar_one_or_none(self):
        return self.value


class _PromptDB:
    def __init__(self):
        self.config_reads = 0
        self.savepoints = 0

    @asynccontextmanager
    async def begin_nested(self):
        self.savepoints += 1
        yield

    async def execute(self, query):
        sql = str(query).lower()
        if "agent_configs" in sql:
            self.config_reads += 1
            if self.config_reads == 1:
                return _Result("Owner Agent")
            return _Result(SimpleNamespace(onboarding_completed=False))
        if "identities" in sql:
            return _Result(rows=[])
        return _Result()


def _runner():
    runner = agent_runner.AgentRunner.__new__(agent_runner.AgentRunner)
    runner._memory_health = {}
    runner.skill_loader = None
    runner.tools = SimpleNamespace(mcp_tool_defs=[])
    runner.max_iterations = 3
    return runner


def test_exact_uuid_and_owner_mobile_full_only(monkeypatch):
    monkeypatch.setattr(config.settings, "prompt_config_reuse_canary_user_ids", "")
    assert not config.prompt_config_reuse_enabled(OWNER)
    monkeypatch.setattr(
        config.settings,
        "prompt_config_reuse_canary_user_ids",
        f"{OWNER},871bac24,*,{OTHER.upper()}",
    )
    assert config.prompt_config_reuse_enabled(OWNER)
    assert not config.prompt_config_reuse_enabled(OTHER)
    assert not config.prompt_config_reuse_enabled(OWNER[:8])
    assert not config.prompt_config_reuse_enabled(None)

    row = SimpleNamespace(agent_name="Owner Agent", onboarding_completed=False)
    snapshot = agent_runner._prompt_config_from_phase1(
        OWNER, "mobile", PromptProfile.FULL, row,
    )
    assert snapshot == agent_runner._PromptConfigSnapshot("Owner Agent", False)
    assert agent_runner._prompt_config_from_phase1(
        OWNER, "voice", PromptProfile.FULL, row,
    ) is None
    assert agent_runner._prompt_config_from_phase1(
        OWNER, "mobile", PromptProfile.SUBAGENT, row,
    ) is None
    assert agent_runner._prompt_config_from_phase1(
        OTHER, "mobile", PromptProfile.FULL, row,
    ) is None
    assert agent_runner._prompt_config_from_phase1(
        OWNER, "mobile", PromptProfile.FULL, None,
    ) is None


def test_prompt_reuses_both_config_fields_and_keeps_the_same_sections():
    async def run():
        runner = _runner()
        legacy_db = _PromptDB()
        legacy = await runner._build_system_prompt(
            legacy_db, OWNER, "Hi", channel="mobile",
            prompt_profile=PromptProfile.FULL,
        )
        fast_db = _PromptDB()
        fast = await runner._build_system_prompt(
            fast_db, OWNER, "Hi", channel="mobile",
            prompt_profile=PromptProfile.FULL,
            prompt_config_snapshot=agent_runner._PromptConfigSnapshot(
                "Owner Agent", False,
            ),
        )
        assert legacy_db.config_reads == legacy_db.savepoints == 2
        assert fast_db.config_reads == fast_db.savepoints == 0
        assert "Your name is **Owner Agent**" in legacy
        assert "Your name is **Owner Agent**" in fast
        assert "# Onboarding Mode (ACTIVE)" in legacy
        assert "# Onboarding Mode (ACTIVE)" in fast

    asyncio.run(run())


def test_failed_phase1_read_uses_both_legacy_prompt_reads():
    async def run():
        runner = _runner()
        db = _PromptDB()
        failed_snapshot = agent_runner._prompt_config_from_phase1(
            OWNER, "mobile", PromptProfile.FULL, None,
        )
        prompt = await runner._build_system_prompt(
            db, OWNER, "Hi", channel="mobile",
            prompt_profile=PromptProfile.FULL,
            prompt_config_snapshot=failed_snapshot,
        )
        assert db.config_reads == db.savepoints == 2
        assert "# Onboarding Mode (ACTIVE)" in prompt

    asyncio.run(run())
