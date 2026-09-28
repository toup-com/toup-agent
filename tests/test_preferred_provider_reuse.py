"""The mobile latency pilot removes a duplicate pre-model provider read."""

from __future__ import annotations

import asyncio

from app import config
from app.agent import agent_runner
from app.db import database


OWNER = "871bac24-1111-4111-8111-111111111111"
OTHER = "871bac24-2222-4222-8222-222222222222"


class _Result:
    def __init__(self, value):
        self.value = value

    def scalar_one_or_none(self):
        return self.value


class _Session:
    def __init__(self, reads, value):
        self.reads = reads
        self.value = value

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return None

    async def execute(self, _query):
        self.reads.append(1)
        await asyncio.sleep(0)
        return _Result(self.value)


def test_exact_owner_uuid_is_required(monkeypatch):
    monkeypatch.setattr(
        config.settings,
        "preferred_provider_reuse_canary_user_ids",
        f"{OWNER},871bac24,*,{OTHER.upper()}",
    )
    assert config.preferred_provider_reuse_enabled(OWNER)
    assert not config.preferred_provider_reuse_enabled(OTHER)
    assert not config.preferred_provider_reuse_enabled(OWNER[:8])
    assert not config.preferred_provider_reuse_enabled(None)


def test_mobile_owner_skips_one_db_session_and_others_keep_legacy_read(monkeypatch):
    reads = []
    monkeypatch.setattr(agent_runner, "preferred_provider_reuse_enabled", lambda u: u == OWNER)
    monkeypatch.setattr(database, "async_session_maker", lambda: _Session(reads, "anthropic"))

    async def run():
        assert await agent_runner._preferred_provider_for_auto_route(
            OWNER, "mobile", "openai", True,
        ) == "openai"
        assert len(reads) == 0
        assert await agent_runner._preferred_provider_for_auto_route(
            OWNER, "voice", "openai", True,
        ) == "anthropic"
        assert await agent_runner._preferred_provider_for_auto_route(
            OTHER, "mobile", "openai", True,
        ) == "anthropic"
        assert len(reads) == 2

    asyncio.run(run())


def test_failed_first_read_retries_and_missing_row_reuses_none(monkeypatch):
    reads = []
    monkeypatch.setattr(agent_runner, "preferred_provider_reuse_enabled", lambda u: u == OWNER)
    monkeypatch.setattr(database, "async_session_maker", lambda: _Session(reads, "openai"))

    async def run():
        assert await agent_runner._preferred_provider_for_auto_route(
            OWNER, "mobile", None, False,
        ) == "openai"
        assert len(reads) == 1
        assert await agent_runner._preferred_provider_for_auto_route(
            OWNER, "mobile", None, True,
        ) is None
        assert len(reads) == 1

    asyncio.run(run())


def test_concurrent_users_keep_their_own_phase1_values(monkeypatch):
    reads = []
    monkeypatch.setattr(agent_runner, "preferred_provider_reuse_enabled", lambda u: u == OWNER)
    monkeypatch.setattr(database, "async_session_maker", lambda: _Session(reads, "anthropic"))

    async def run():
        return await asyncio.gather(
            agent_runner._preferred_provider_for_auto_route(OWNER, "mobile", "openai", True),
            agent_runner._preferred_provider_for_auto_route(OTHER, "mobile", "openai", True),
        )

    assert asyncio.run(run()) == ["openai", "anthropic"]
    assert len(reads) == 1
