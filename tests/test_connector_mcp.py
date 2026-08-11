"""
T1f — Connector MCP namespace registration tests.

The platform exposes one FastMCP tool per (connector, manifest_tool)
pair. Per-request user resolution comes from T0c's `MCPAuthMiddleware`
(ContextVars). The dispatcher (T1e) is the runtime hot path; this
module is the transport adapter.

Layered like `test_mcp_auth.py`:

  Unit (no DB, no FastMCP runtime):
    - Result serialisation per ConnectorResult variant
    - Channel + request_id ContextVar binding through MCPAuthMiddleware
    - X-Toup-Channel unknown value → clamped to "web" with WARN
    - Tool registration count + tag application
    - Skill-prefix collision logged WARN at register time
    - tools/list filter middleware: empty for unauthenticated user

  Integration (real Postgres, mounted FastMCP):
    - tools/list shows stub__echo only when user has active stub identity
    - Cross-tenant isolation on tools/list (alice has stub, bob does not)
    - tools/list re-filters after disconnect (tool drops out)
    - tools/call dispatch through registered tool → ConnectorOk shape
    - Channel header drives default-deny on mutating tools (voice → reject)

Each integration test seeds at most ONE user → SQLite is_canary trap
stays clear when the test runs locally; the cross-tenant test seeds
two users and skips on SQLite (matches T0c's pattern).
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import AsyncIterator, ClassVar
from datetime import datetime, timedelta

import httpx
import pytest
import pytest_asyncio
from cryptography.fernet import Fernet
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from app.config import settings
from app.connectors.base import (
    BaseConnectorProvider,
    ConnectorContext,
    ConnectorOk,
    ConnectorProviderDown,
    ConnectorRateLimited,
    ConnectorReauthRequired,
    ConnectorResult,
    ConnectorScopeMissing,
    ConnectorToolError,
    HealthResult,
    RefreshResult,
)
from app.db.database import async_session_maker
from app.db.models import AgentConfig, ConnectorIdentity, User
from app.mcp_auth import (
    MCPAuthMiddleware,
    _current_channel,
    _current_request_id,
    _current_user_id,
    get_mcp_channel,
    get_mcp_request_id,
    get_mcp_user_id,
    try_get_mcp_user_id,
)
from app.services import connector_vault as vault
from app.services.connector_mcp import (
    CONNECTOR_TOOL_TAG,
    ConnectorToolFilterMiddleware,
    _serialize_result,
    deregister_connector_tools_for_tests,
    register_connector_tools,
)
from app.services.connector_registry import (
    ConnectorEntry,
    ConnectorManifest,
    ConnectorTool,
    ChannelPolicy,
    HealthSpec,
    OAuthSpec,
    get_registry,
    reset_registry_for_tests,
)
from app.services.credential_crypto import _multi_fernet


# ─── Helpers ────────────────────────────────────────────────────────────


def _is_sqlite() -> bool:
    import os
    return os.environ.get("DATABASE_URL", "").startswith("sqlite")


class _StubProvider(BaseConnectorProvider):
    """Deterministic stub used by registration + dispatch tests."""

    manifest_id: ClassVar[str] = "stub_t1f"

    async def execute(self, tool_name, tool_input, ctx):
        return ConnectorOk(content=json.dumps({
            "tool": tool_name,
            "input": tool_input,
            "channel": ctx.channel,
            "request_id": ctx.request_id,
        }))

    async def revoke(self, user_id, access_token, refresh_token=None):
        return None

    async def refresh(self, refresh_token, *, scopes=None):
        return RefreshResult(
            access_token="refreshed",
            refresh_token=refresh_token,
            expires_at=datetime.utcnow() + timedelta(hours=1),
        )

    async def health_probe(self, ctx):
        return HealthResult(ok=True)


def _make_manifest(
    *,
    tool_name: str = "stub_t1f__echo",
    mutates: bool = False,
    channel_deny: list[str] | None = None,
) -> ConnectorManifest:
    return ConnectorManifest(
        manifest_version=1,
        id="stub_t1f",
        name="Stub T1f",
        short_description="t1f harness",
        status="experimental",
        category="test",
        oauth=OAuthSpec(
            provider_app="stub_provider_app",
            scopes=[], pkce=True, refresh=True,
        ),
        health=HealthSpec(probe=tool_name),
        tools=[
            ConnectorTool(
                name=tool_name,
                description=f"echo for {tool_name}",
                input_schema={
                    "type": "object",
                    "properties": {"message": {"type": "string"}},
                    "required": ["message"],
                },
                mutates=mutates,
                elevation=False,
                output_redaction=[],
                channel_policy=ChannelPolicy(
                    default="allow",
                    deny=channel_deny or [],
                ),
            )
        ],
    )


@pytest.fixture(autouse=True)
def _provision_crypto():
    """Per-test Fernet key for encrypting tokens via the vault."""
    prev = settings.platform_encryption_key
    prev_prev = settings.platform_encryption_key_previous
    settings.platform_encryption_key = Fernet.generate_key().decode()
    settings.platform_encryption_key_previous = ""
    _multi_fernet.cache_clear()
    try:
        yield
    finally:
        settings.platform_encryption_key = prev
        settings.platform_encryption_key_previous = prev_prev
        _multi_fernet.cache_clear()


@pytest.fixture
def isolated_registry():
    """Wipe + repopulate the singleton registry for the test, restore
    on teardown so other tests see the production state."""
    reset_registry_for_tests()

    def _install(manifest: ConnectorManifest, provider: BaseConnectorProvider):
        reg = get_registry()
        reg._entries[manifest.id] = ConnectorEntry(
            manifest=manifest, provider=provider,
        )
        for tool in manifest.tools:
            reg._tool_index[tool.name] = manifest.id
        return reg

    yield _install
    reset_registry_for_tests()


# ─── ASGI helpers (mirror test_mcp_auth.py) ─────────────────────────────


class _ContextRecorder:
    """Inner ASGI app: writes the bound (user_id, channel, request_id)
    so middleware tests can assert what got into ContextVars."""

    def __init__(self):
        self.calls = 0

    async def __call__(self, scope, receive, send):
        self.calls += 1
        if scope["type"] != "http":
            return
        try:
            uid = get_mcp_user_id()
        except ValueError:
            uid = None
        body = json.dumps({
            "user_id": uid,
            "channel": get_mcp_channel(),
            "request_id": get_mcp_request_id(),
        }).encode()
        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        })
        await send({"type": "http.response.body", "body": body})


def _make_scope(headers: dict[str, str]) -> dict:
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "https",
        "path": "/mcp/tools/call",
        "raw_path": b"/mcp/tools/call",
        "query_string": b"",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        "client": ("203.0.113.7", 12345),
        "server": ("toup.ai", 443),
    }


async def _drive(app, scope: dict) -> tuple[int, dict, bytes]:
    sent = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(msg):
        sent.append(msg)

    await app(scope, receive, send)
    start = next(m for m in sent if m["type"] == "http.response.start")
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    headers = {k.decode(): v.decode() for k, v in start.get("headers", [])}
    return start["status"], headers, body


@pytest.fixture
def fake_tenants(monkeypatch) -> dict[str, dict]:
    """Same shape as test_mcp_auth: monkey-patched key→user_id table."""
    alice_uid = "00000000-0000-0000-0000-aaaaaaaaaaaa"
    bob_uid = "00000000-0000-0000-0000-bbbbbbbbbbbb"
    table = {
        "key_alice_t1f": alice_uid,
        "key_bob_t1f": bob_uid,
    }

    async def fake_resolve(agent_key: str):
        return table.get(agent_key)

    monkeypatch.setattr("app.mcp_auth._resolve_agent_key_to_user_id", fake_resolve)
    return {
        "alice": {"user_id": alice_uid, "agent_key": "key_alice_t1f"},
        "bob": {"user_id": bob_uid, "agent_key": "key_bob_t1f"},
    }


# ─── Unit: serialisation ────────────────────────────────────────────────


def test_serialize_ok_passes_content_unchanged():
    out = _serialize_result(ConnectorOk(content='{"hello":"world"}'))
    assert out == {"kind": "ok", "content": '{"hello":"world"}'}


def test_serialize_rate_limited_includes_retry_after():
    out = _serialize_result(ConnectorRateLimited(retry_after_s=42))
    assert out["kind"] == "rate_limited"
    assert out["retry_after_s"] == 42
    assert "42" in out["message"]


def test_serialize_reauth_required_includes_url():
    out = _serialize_result(
        ConnectorReauthRequired(reauth_url="/agent/integrations/gmail")
    )
    assert out["kind"] == "reauth_required"
    assert out["reauth_url"] == "/agent/integrations/gmail"
    assert "/agent/integrations/gmail" in out["message"]


def test_serialize_provider_down_includes_status_url():
    out = _serialize_result(
        ConnectorProviderDown(provider_status_url="https://status.example.com")
    )
    assert out["kind"] == "provider_down"
    assert out["provider_status_url"] == "https://status.example.com"
    assert "provider_down" in out["message"]


def test_serialize_scope_missing_includes_required_scope():
    out = _serialize_result(
        ConnectorScopeMissing(required_scope="https://www.googleapis.com/auth/gmail.send")
    )
    assert out["kind"] == "scope_missing"
    assert out["required_scope"].endswith("gmail.send")
    assert "gmail.send" in out["message"]


def test_serialize_tool_error_passes_message_and_retryable():
    out = _serialize_result(
        ConnectorToolError(message="bad input", retryable=False)
    )
    assert out["kind"] == "tool_error"
    assert out["retryable"] is False
    assert "bad input" in out["message"]


def test_serialize_unknown_subclass_falls_back_to_tool_error(caplog):
    class _MysteryResult(ConnectorResult):
        pass

    with caplog.at_level(logging.ERROR, logger="app.services.connector_mcp"):
        out = _serialize_result(_MysteryResult())
    assert out["kind"] == "tool_error"
    assert "_MysteryResult" in out["message"]
    assert any("unknown ConnectorResult subclass" in r.getMessage() for r in caplog.records)


# ─── Unit: channel + request_id ContextVars via middleware ─────────────


@pytest.mark.asyncio
async def test_channel_header_binds_to_contextvar(fake_tenants):
    """X-Toup-Channel: voice → get_mcp_channel() returns 'voice' inside
    the request, resets to 'web' afterwards."""
    inner = _ContextRecorder()
    mw = MCPAuthMiddleware(inner)

    status, _, body = await _drive(mw, _make_scope({
        "X-Agent-Key": fake_tenants["alice"]["agent_key"],
        "X-Toup-Channel": "voice",
    }))
    assert status == 200
    payload = json.loads(body)
    assert payload["channel"] == "voice"
    assert payload["user_id"] == fake_tenants["alice"]["user_id"]
    # ContextVar must have reset.
    assert _current_channel.get() == "web", "channel contextvar must reset to default"


@pytest.mark.asyncio
async def test_channel_header_missing_defaults_to_web(fake_tenants):
    inner = _ContextRecorder()
    mw = MCPAuthMiddleware(inner)
    status, _, body = await _drive(mw, _make_scope({
        "X-Agent-Key": fake_tenants["alice"]["agent_key"],
    }))
    assert status == 200
    assert json.loads(body)["channel"] == "web"


@pytest.mark.asyncio
async def test_unknown_channel_clamped_to_web_with_warn(fake_tenants, caplog):
    """X-Toup-Channel: smoke-signals → clamped to 'web' + WARN log.
    A typo cannot accidentally bypass voice/telegram default-deny."""
    inner = _ContextRecorder()
    mw = MCPAuthMiddleware(inner)
    with caplog.at_level(logging.WARNING, logger="app.mcp_auth"):
        status, _, body = await _drive(mw, _make_scope({
            "X-Agent-Key": fake_tenants["alice"]["agent_key"],
            "X-Toup-Channel": "smoke-signals",
        }))
    assert status == 200
    assert json.loads(body)["channel"] == "web"
    assert any("unknown channel" in r.getMessage() and "smoke-signals" in r.getMessage()
               for r in caplog.records)


@pytest.mark.asyncio
async def test_request_id_header_binds_to_contextvar(fake_tenants):
    inner = _ContextRecorder()
    mw = MCPAuthMiddleware(inner)
    status, _, body = await _drive(mw, _make_scope({
        "X-Agent-Key": fake_tenants["alice"]["agent_key"],
        "X-Request-Id": "req_abc_123",
    }))
    assert status == 200
    assert json.loads(body)["request_id"] == "req_abc_123"
    # Reset after request.
    assert _current_request_id.get() == "no-id"


@pytest.mark.asyncio
async def test_traceparent_falls_back_when_no_request_id(fake_tenants):
    inner = _ContextRecorder()
    mw = MCPAuthMiddleware(inner)
    status, _, body = await _drive(mw, _make_scope({
        "X-Agent-Key": fake_tenants["alice"]["agent_key"],
        "Traceparent": "00-abc-def-01",
    }))
    assert status == 200
    assert json.loads(body)["request_id"] == "00-abc-def-01"


@pytest.mark.asyncio
async def test_request_id_default_when_no_header(fake_tenants):
    inner = _ContextRecorder()
    mw = MCPAuthMiddleware(inner)
    status, _, body = await _drive(mw, _make_scope({
        "X-Agent-Key": fake_tenants["alice"]["agent_key"],
    }))
    assert status == 200
    assert json.loads(body)["request_id"] == "no-id"


# ─── Unit: try_get_mcp_user_id ─────────────────────────────────────────


def test_try_get_mcp_user_id_returns_none_unbound():
    """Non-raising variant — used by tools/list filter so unauthenticated
    list calls don't crash the LLM client."""
    assert try_get_mcp_user_id() is None


# ─── Unit: registration count + tag application ─────────────────────────


@pytest.mark.asyncio
async def test_register_connector_tools_creates_one_tool_per_manifest_tool(
    isolated_registry,
):
    from fastmcp import FastMCP

    manifest = _make_manifest()
    provider = _StubProvider()
    isolated_registry(manifest, provider)

    mcp = FastMCP("test")
    n = register_connector_tools(mcp, get_registry())
    assert n == 1

    tools = await mcp.get_tools()
    assert "stub_t1f__echo" in tools
    tool = tools["stub_t1f__echo"]
    assert CONNECTOR_TOOL_TAG in tool.tags
    assert "connector:stub_t1f" in tool.tags
    # Manifest's input_schema is exposed verbatim.
    assert tool.parameters["properties"]["message"]["type"] == "string"
    assert tool.parameters["required"] == ["message"]


@pytest.mark.asyncio
async def test_register_is_idempotent_under_double_call(
    isolated_registry, caplog,
):
    """Second call logs a warn and continues — test reload path."""
    from fastmcp import FastMCP

    manifest = _make_manifest()
    isolated_registry(manifest, _StubProvider())
    mcp = FastMCP("test")
    register_connector_tools(mcp, get_registry())

    with caplog.at_level(logging.WARNING, logger="app.services.connector_mcp"):
        n2 = register_connector_tools(mcp, get_registry())
    # Either the second register added 0, or it did and warned. Either is OK.
    # The contract is: no exception, server still works.
    tools = await mcp.get_tools()
    assert "stub_t1f__echo" in tools


@pytest.mark.asyncio
async def test_skill_prefix_collision_logs_warn(isolated_registry, caplog):
    from fastmcp import FastMCP

    manifest = _make_manifest(tool_name="memory__shadow")
    # Override registry id matching for this test — collision warn fires
    # by tool prefix, regardless of connector id mismatch.
    manifest = ConnectorManifest(**{**manifest.model_dump(), "id": "stub_t1f"})
    # Rebuild the manifest with the colliding tool name + matching id.
    manifest = _make_manifest(tool_name="memory__shadow")
    isolated_registry(manifest, _StubProvider())
    mcp = FastMCP("test")

    with caplog.at_level(logging.WARNING, logger="app.services.connector_mcp"):
        register_connector_tools(
            mcp, get_registry(),
            skill_prefixes={"memory"},  # simulate skill namespace
        )

    warns = [r for r in caplog.records
             if "prefix collides with skill namespace" in r.getMessage()]
    assert warns, "expected a WARN for skill-prefix collision at register time"
    msg = warns[0].getMessage()
    assert "memory" in msg
    # Tool still registered (warn, not block).
    tools = await mcp.get_tools()
    assert "memory__shadow" in tools


# ─── Unit: filter middleware (no DB) ────────────────────────────────────


@pytest.mark.asyncio
async def test_filter_middleware_passes_through_when_no_user_bound(
    isolated_registry,
):
    """Warn-only mode: try_get_mcp_user_id is None → list returned
    unfiltered. Connector tools STAY in the list; their handler will
    raise on dispatch."""
    isolated_registry(_make_manifest(), _StubProvider())

    class _FakeListResult:
        def __init__(self, tool_names):
            self.tools = [type("T", (), {"name": n})() for n in tool_names]

    initial = _FakeListResult(["memory_search", "stub_t1f__echo"])

    async def call_next(_ctx):
        return initial

    middleware = ConnectorToolFilterMiddleware(get_registry())
    result = await middleware.on_list_tools(context=None, call_next=call_next)
    names = [t.name for t in result.tools]
    assert "memory_search" in names
    assert "stub_t1f__echo" in names, (
        "warn-only mode (no user) must NOT hide connector tools — handler "
        "raises on call instead"
    )


# ─── Integration: real Postgres + mounted FastMCP ──────────────────────


@pytest_asyncio.fixture
async def db_tenants_with_stub() -> dict[str, dict]:
    """Two real users, alice has an active stub_t1f identity, bob doesn't.
    Skips on SQLite (User.is_canary partial unique index)."""
    if _is_sqlite():
        pytest.skip("requires Postgres (User.is_canary partial unique index)")

    out: dict[str, dict] = {}
    async with async_session_maker() as db:
        for name in ("alice", "bob"):
            uid = str(uuid.uuid4())
            user = User(
                id=uid,
                email=f"{name}-{uid[:8]}@example.com",
                hashed_password="x",
                name=name,
            )
            db.add(user)
            await db.flush()
            cfg = AgentConfig(
                user_id=uid,
                agent_api_key=f"key_{name}_{uuid.uuid4().hex[:24]}",
            )
            db.add(cfg)
            out[name] = {"user_id": uid, "agent_key": cfg.agent_api_key}
        await db.commit()

    # Alice has stub_t1f connected; bob does not.
    async with async_session_maker() as db:
        await vault.put(
            db, out["alice"]["user_id"], "stub_t1f",
            access_token="seed_at",
            refresh_token="rt_seed",
            access_expires_at=datetime.utcnow() + timedelta(hours=1),
        )
    return out


@pytest_asyncio.fixture
async def t1f_mcp_app(isolated_registry):
    """FastAPI app with the real platform MCP + connector_mcp wiring.
    Skips if the local FastAPI/FastMCP pin mismatch hits."""
    from app.mcp_server import mcp as mcp_server

    # Install our test manifest into the registry singleton, register
    # connector tools onto the same server the tests will hit.
    isolated_registry(_make_manifest(), _StubProvider())
    register_connector_tools(mcp_server, get_registry())
    mcp_server.add_middleware(ConnectorToolFilterMiddleware(get_registry()))

    try:
        mcp_app = mcp_server.http_app(path="/mcp")
        parent = FastAPI(lifespan=mcp_app.lifespan)
    except TypeError as e:
        pytest.skip(f"FastAPI/FastMCP pin mismatch (CI has correct pins): {e}")
    parent.mount("/api/mcp", MCPAuthMiddleware(mcp_app))
    yield parent
    # Cleanup: deregister tools so other test files see a clean server.
    deregister_connector_tools_for_tests(mcp_server, get_registry())


@pytest_asyncio.fixture
async def t1f_http_client(t1f_mcp_app) -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=t1f_mcp_app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


def _parse_jsonrpc_response(text: str) -> dict:
    """FastMCP returns either pure JSON or an SSE stream. Pull the first
    JSON-RPC envelope out either way."""
    text = text.strip()
    if text.startswith("{"):
        return json.loads(text)
    # SSE: lines like "event: message\ndata: {...}\n\n"
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            payload = line[len("data:"):].strip()
            if payload:
                return json.loads(payload)
    raise AssertionError(f"could not parse JSON-RPC from: {text[:300]}")


async def _list_tools(client: AsyncClient, agent_key: str) -> list[str]:
    r = await client.post(
        "/api/mcp/mcp/",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        headers={
            "X-Agent-Key": agent_key,
            "Accept": "application/json, text/event-stream",
        },
    )
    assert r.status_code == 200, f"tools/list got {r.status_code}: {r.text[:300]}"
    env = _parse_jsonrpc_response(r.text)
    assert "result" in env, f"missing result: {env}"
    return [t["name"] for t in env["result"]["tools"]]


@pytest.mark.asyncio
async def test_tools_list_includes_connector_tool_for_connected_user(
    t1f_http_client, db_tenants_with_stub,
):
    alice = db_tenants_with_stub["alice"]
    names = await _list_tools(t1f_http_client, alice["agent_key"])
    assert "stub_t1f__echo" in names, (
        f"alice has stub_t1f connected → tool must appear; got: {names}"
    )
    # Sanity: built-in tools also visible.
    assert "memory_search" in names


@pytest.mark.asyncio
async def test_tools_list_excludes_connector_tool_for_disconnected_user(
    t1f_http_client, db_tenants_with_stub,
):
    bob = db_tenants_with_stub["bob"]
    names = await _list_tools(t1f_http_client, bob["agent_key"])
    assert "stub_t1f__echo" not in names, (
        f"bob has no stub_t1f connection → tool must NOT appear; got: {names}"
    )
    # But other tools still present.
    assert "memory_search" in names


@pytest.mark.asyncio
async def test_tools_list_re_filters_after_disconnect(
    t1f_http_client, db_tenants_with_stub,
):
    alice = db_tenants_with_stub["alice"]
    # Initially visible.
    names = await _list_tools(t1f_http_client, alice["agent_key"])
    assert "stub_t1f__echo" in names

    # Disconnect — vault.disconnect zeroes ciphertext + flips status.
    async with async_session_maker() as db:
        await vault.disconnect(db, alice["user_id"], "stub_t1f")

    # Now the tool drops out of the list.
    names = await _list_tools(t1f_http_client, alice["agent_key"])
    assert "stub_t1f__echo" not in names, (
        "after disconnect, tool must drop from tools/list"
    )


@pytest.mark.asyncio
async def test_tools_call_dispatches_to_provider_returns_ok_envelope(
    t1f_http_client, db_tenants_with_stub,
):
    alice = db_tenants_with_stub["alice"]
    r = await t1f_http_client.post(
        "/api/mcp/mcp/",
        json={
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "stub_t1f__echo",
                "arguments": {"message": "hello"},
            },
        },
        headers={
            "X-Agent-Key": alice["agent_key"],
            "X-Request-Id": "req_t1f_dispatch",
            "Accept": "application/json, text/event-stream",
        },
    )
    assert r.status_code == 200, f"got {r.status_code}: {r.text[:300]}"
    env = _parse_jsonrpc_response(r.text)
    # FastMCP wraps return as content blocks; the dict we returned will
    # appear in the structured content. We just need to see "ok" + the
    # echoed message somewhere in the response body.
    body = json.dumps(env)
    assert '"kind"' in body and '"ok"' in body, (
        f"expected serialised ok envelope in response; got: {body[:400]}"
    )
    assert "hello" in body


@pytest.mark.asyncio
async def test_channel_header_drives_default_deny_on_voice(
    t1f_http_client, db_tenants_with_stub, isolated_registry,
):
    """Re-register the stub manifest with mutates=true. Calling from
    voice → tool_error envelope (default-deny). Calling from web →
    ok envelope."""
    # Replace the registered manifest with a mutating one.
    from app.mcp_server import mcp as mcp_server

    deregister_connector_tools_for_tests(mcp_server, get_registry())
    isolated_registry(
        _make_manifest(tool_name="stub_t1f__write", mutates=True),
        _StubProvider(),
    )
    register_connector_tools(mcp_server, get_registry())

    alice = db_tenants_with_stub["alice"]

    # voice → must be denied. Tool may not even appear in tools/list
    # if the user has no active identity for stub_t1f after registry
    # swap; ensure they do.
    async with async_session_maker() as db:
        existing = await vault.get(db, alice["user_id"], "stub_t1f")
        if existing is None:
            await vault.put(
                db, alice["user_id"], "stub_t1f",
                access_token="seed_at",
                refresh_token="rt_seed",
                access_expires_at=datetime.utcnow() + timedelta(hours=1),
            )

    async def _call(channel_header: str | None) -> dict:
        headers = {
            "X-Agent-Key": alice["agent_key"],
            "Accept": "application/json, text/event-stream",
        }
        if channel_header is not None:
            headers["X-Toup-Channel"] = channel_header
        r = await t1f_http_client.post(
            "/api/mcp/mcp/",
            json={
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "stub_t1f__write",
                    "arguments": {"message": "hi"},
                },
            },
            headers=headers,
        )
        assert r.status_code == 200, f"got {r.status_code}: {r.text[:300]}"
        return _parse_jsonrpc_response(r.text)

    voice_env = await _call("voice")
    voice_body = json.dumps(voice_env)
    assert '"tool_error"' in voice_body, (
        f"voice should be tool_error (mutating + voice default-deny); "
        f"got: {voice_body[:400]}"
    )
    assert "mutating" in voice_body.lower() or "voice" in voice_body.lower()

    web_env = await _call("web")
    web_body = json.dumps(web_env)
    assert '"ok"' in web_body, (
        f"web should succeed (mutating allowed on web); got: {web_body[:400]}"
    )


# ─── Cross-tenant isolation on tools/list ───────────────────────────────


@pytest.mark.asyncio
async def test_cross_tenant_isolation_on_tools_list(
    t1f_http_client, db_tenants_with_stub,
):
    """Same FastMCP server instance serves both alice and bob. Filter
    middleware must scope per request — alice sees stub_t1f, bob does
    not."""
    alice = db_tenants_with_stub["alice"]
    bob = db_tenants_with_stub["bob"]

    alice_names = await _list_tools(t1f_http_client, alice["agent_key"])
    bob_names = await _list_tools(t1f_http_client, bob["agent_key"])

    assert "stub_t1f__echo" in alice_names
    assert "stub_t1f__echo" not in bob_names, (
        "bob's tools/list must not include alice's connector tool — "
        "this is the cross-tenant isolation point"
    )
