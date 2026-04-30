"""
Tests for GET /api/models — the canonical model registry endpoint that
the frontend consumes instead of hardcoding model strings.

Verifies:
  - Endpoint shape (default, fallback, auto_builder, anthropic, openai).
  - `default` and `fallback` come from the resolver (env override visible
    through the wire).
  - `auto_builder.{planner,builder}` reflect the split resolver functions.
  - Provider lists pull from `model_providers.get_provider_registry()`,
    not a separate hardcoded list.
  - Response carries the documented Cache-Control header.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api.models import router as models_router
from app.services import model_resolver as mr


# Override the suite-wide DB autouse fixture; this test doesn't touch DB.
@pytest_asyncio.fixture(autouse=True)
async def _reset_database():
    yield


@asynccontextmanager
async def _client():
    app = FastAPI()
    app.include_router(models_router, prefix="/api")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def _settings_with(**overrides):
    return patch.object(mr, "settings", SimpleNamespace(**overrides))


# ── Shape ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_endpoint_returns_expected_top_level_keys():
    async with _client() as c:
        resp = await c.get("/api/models")
    assert resp.status_code == 200
    body = resp.json()
    for key in ("default", "fallback", "auto_builder", "anthropic", "openai"):
        assert key in body, f"missing top-level key: {key}"


@pytest.mark.asyncio
async def test_auto_builder_has_planner_and_builder():
    async with _client() as c:
        resp = await c.get("/api/models")
    body = resp.json()
    assert "planner" in body["auto_builder"]
    assert "builder" in body["auto_builder"]


@pytest.mark.asyncio
async def test_provider_lists_are_id_label_objects():
    async with _client() as c:
        resp = await c.get("/api/models")
    body = resp.json()
    for entry in body["anthropic"] + body["openai"]:
        assert "id" in entry
        assert "label" in entry
        assert isinstance(entry["id"], str)
        assert isinstance(entry["label"], str)


# ── Resolver integration — flipping settings is visible on the wire ──


@pytest.mark.asyncio
async def test_default_tracks_settings_agent_model():
    with _settings_with(
        agent_model="claude-opus-4-99",
        agent_fallback_model=None,
        app_builder_planner_model=None,
        app_builder_builder_model=None,
    ):
        async with _client() as c:
            resp = await c.get("/api/models")
    assert resp.json()["default"] == "claude-opus-4-99"


@pytest.mark.asyncio
async def test_fallback_tracks_settings():
    with _settings_with(
        agent_model="claude-opus-4-7",
        agent_fallback_model="gpt-7.0",
        app_builder_planner_model=None,
        app_builder_builder_model=None,
    ):
        async with _client() as c:
            resp = await c.get("/api/models")
    assert resp.json()["fallback"] == "gpt-7.0"


@pytest.mark.asyncio
async def test_auto_builder_split_visible_when_set():
    with _settings_with(
        agent_model="agent-default",
        agent_fallback_model=None,
        app_builder_planner_model="opus-strong",
        app_builder_builder_model="sonnet-cheap",
    ):
        async with _client() as c:
            resp = await c.get("/api/models")
    body = resp.json()
    assert body["auto_builder"]["planner"] == "opus-strong"
    assert body["auto_builder"]["builder"] == "sonnet-cheap"


@pytest.mark.asyncio
async def test_auto_builder_split_inherits_when_unset():
    with _settings_with(
        agent_model="claude-opus-4-7",
        agent_fallback_model=None,
        app_builder_planner_model=None,
        app_builder_builder_model=None,
    ):
        async with _client() as c:
            resp = await c.get("/api/models")
    body = resp.json()
    assert body["auto_builder"]["planner"] == "claude-opus-4-7"
    assert body["auto_builder"]["builder"] == "claude-opus-4-7"


# ── Caching ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cache_control_header_present():
    async with _client() as c:
        resp = await c.get("/api/models")
    cache = resp.headers.get("cache-control", "")
    assert "public" in cache
    assert "max-age" in cache
