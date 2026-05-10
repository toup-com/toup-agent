"""
T2b — Catalog endpoint smoke.

Unit-level via direct call (no HTTP / FastAPI lifespan dependency).
The endpoint returns:

  - one entry per loaded manifest
  - status overlaid from the user's ConnectorIdentity rows
  - scope descriptions resolved through manifest override → built-in
    fallback → raw scope string
  - `coming_soon` flag wired
  - experimental connectors filtered in production env
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import ClassVar

import pytest
import pytest_asyncio
from cryptography.fernet import Fernet

from app.config import settings
from app.connectors.base import (
    BaseConnectorProvider, ConnectorOk, HealthResult, RefreshResult,
)
from app.db.database import async_session_maker
from app.db.models import User
from app.services import connector_vault as vault
from app.services.connector_registry import (
    ConnectorEntry, ConnectorManifest, ConnectorTool, ChannelPolicy,
    HealthSpec, OAuthSpec, get_registry, reset_registry_for_tests,
)
from app.services.credential_crypto import _multi_fernet

# FastAPI/Starlette pin mismatch on local skips the entire file rather
# than letting every test fail at module import. CI has the right pins.
try:
    from app.api.connectors_catalog import get_connector_catalog
except TypeError as e:
    pytest.skip(
        f"FastAPI pin mismatch (CI has correct pins): {e}",
        allow_module_level=True,
    )


class _StubProvider(BaseConnectorProvider):
    manifest_id: ClassVar[str] = "cstub"

    async def execute(self, n, i, c): return ConnectorOk(content="")
    async def revoke(self, *a, **k): pass
    async def refresh(self, r):
        return RefreshResult(
            access_token="a", refresh_token=r,
            expires_at=datetime.utcnow() + timedelta(hours=1),
        )
    async def health_probe(self, ctx): return HealthResult(ok=True)


def _install(*, scope_descriptions: dict | None = None,
             scopes: list[str] | None = None,
             scopes_optional: list[str] | None = None,
             experimental: bool = False) -> _StubProvider:
    reset_registry_for_tests()
    manifest = ConnectorManifest(
        manifest_version=1,
        id="cstub",
        name="Catalog Stub",
        short_description="catalog smoke",
        status="experimental" if experimental else "stable",
        category="test",
        oauth=OAuthSpec(
            provider_app="stub_provider_app",
            scopes=scopes or [],
            scopes_optional=scopes_optional or [],
            pkce=True, refresh=True,
        ),
        health=HealthSpec(probe="cstub__do"),
        tools=[
            ConnectorTool(
                name="cstub__do",
                description="x",
                input_schema={"type": "object", "properties": {}},
                mutates=False,
                elevation=False,
                output_redaction=[],
                channel_policy=ChannelPolicy(default="allow", deny=[]),
            )
        ],
        scope_descriptions=scope_descriptions or {},
    )
    provider = _StubProvider()
    reg = get_registry()
    reg._entries["cstub"] = ConnectorEntry(manifest=manifest, provider=provider)
    for tool in manifest.tools:
        reg._tool_index[tool.name] = manifest.id
    return provider


@pytest.fixture(autouse=True)
def _crypto():
    settings.platform_encryption_key = Fernet.generate_key().decode()
    settings.platform_encryption_key_previous = ""
    _multi_fernet.cache_clear()
    yield


@pytest_asyncio.fixture
async def alice_id() -> str:
    async with async_session_maker() as db:
        uid = str(uuid.uuid4())
        db.add(User(id=uid, email=f"{uid[:8]}@example.com",
                    hashed_password="x", name="Alice"))
        await db.commit()
    return uid


# Mock current_user
class _FakeUser:
    def __init__(self, uid):
        self.id = uid


@pytest.mark.asyncio
async def test_catalog_lists_loaded_connectors(alice_id):
    from app.api.connectors_catalog import get_connector_catalog
    _install()

    async with async_session_maker() as db:
        resp = await get_connector_catalog(
            current_user=_FakeUser(alice_id), db=db,
        )
    ids = [c.id for c in resp.connectors]
    assert "cstub" in ids


@pytest.mark.asyncio
async def test_catalog_status_null_when_not_connected(alice_id):
    from app.api.connectors_catalog import get_connector_catalog
    _install()

    async with async_session_maker() as db:
        resp = await get_connector_catalog(
            current_user=_FakeUser(alice_id), db=db,
        )
    [entry] = [c for c in resp.connectors if c.id == "cstub"]
    assert entry.status is None
    assert entry.last_used_at is None


@pytest.mark.asyncio
async def test_catalog_status_overlay_after_connect(alice_id):
    from app.api.connectors_catalog import get_connector_catalog
    _install()

    async with async_session_maker() as db:
        await vault.put(
            db, alice_id, "cstub",
            access_token="seed",
            refresh_token="rt",
            access_expires_at=datetime.utcnow() + timedelta(hours=1),
        )
        resp = await get_connector_catalog(
            current_user=_FakeUser(alice_id), db=db,
        )
    [entry] = [c for c in resp.connectors if c.id == "cstub"]
    assert entry.status == "active"


@pytest.mark.asyncio
async def test_catalog_scope_descriptions_use_manifest_override_first(alice_id):
    """Manifest-declared description wins over the built-in dictionary."""
    from app.api.connectors_catalog import get_connector_catalog
    _install(
        scopes=["custom_scope_xyz"],
        scope_descriptions={"custom_scope_xyz": "MANIFEST OVERRIDE"},
    )

    async with async_session_maker() as db:
        resp = await get_connector_catalog(
            current_user=_FakeUser(alice_id), db=db,
        )
    [entry] = [c for c in resp.connectors if c.id == "cstub"]
    [scope] = [s for s in entry.scopes if s.scope == "custom_scope_xyz"]
    assert scope.description == "MANIFEST OVERRIDE"
    assert scope.required is True


@pytest.mark.asyncio
async def test_catalog_scope_descriptions_fall_back_to_builtin(alice_id):
    """Real Gmail scope hits the built-in DEFAULT_SCOPE_DESCRIPTIONS."""
    from app.api.connectors_catalog import get_connector_catalog
    _install(
        scopes=["https://www.googleapis.com/auth/gmail.readonly"],
    )

    async with async_session_maker() as db:
        resp = await get_connector_catalog(
            current_user=_FakeUser(alice_id), db=db,
        )
    [entry] = [c for c in resp.connectors if c.id == "cstub"]
    [scope] = [s for s in entry.scopes if "gmail.readonly" in s.scope]
    assert "Gmail" in scope.description or "mail" in scope.description.lower()


@pytest.mark.asyncio
async def test_catalog_optional_scopes_marked_required_false(alice_id):
    from app.api.connectors_catalog import get_connector_catalog
    _install(
        scopes=["req_scope"],
        scopes_optional=["opt_scope"],
        scope_descriptions={"req_scope": "Req", "opt_scope": "Opt"},
    )

    async with async_session_maker() as db:
        resp = await get_connector_catalog(
            current_user=_FakeUser(alice_id), db=db,
        )
    [entry] = [c for c in resp.connectors if c.id == "cstub"]
    by_scope = {s.scope: s for s in entry.scopes}
    assert by_scope["req_scope"].required is True
    assert by_scope["opt_scope"].required is False


@pytest.mark.asyncio
async def test_catalog_filters_experimental_in_production(alice_id, monkeypatch):
    """Experimental connectors hidden when ENVIRONMENT=production."""
    from app.api.connectors_catalog import get_connector_catalog
    _install(experimental=True)

    monkeypatch.setattr(settings, "environment", "production")
    async with async_session_maker() as db:
        resp = await get_connector_catalog(
            current_user=_FakeUser(alice_id), db=db,
        )
    ids = [c.id for c in resp.connectors]
    assert "cstub" not in ids


@pytest.mark.asyncio
async def test_catalog_includes_tool_listings(alice_id):
    from app.api.connectors_catalog import get_connector_catalog
    _install()

    async with async_session_maker() as db:
        resp = await get_connector_catalog(
            current_user=_FakeUser(alice_id), db=db,
        )
    [entry] = [c for c in resp.connectors if c.id == "cstub"]
    assert len(entry.tools) == 1
    assert entry.tools[0].name == "cstub__do"
    assert entry.tools[0].mutates is False
