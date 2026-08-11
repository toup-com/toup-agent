"""B-1 pins — the app-MCP lifespan must be composed into the agent's.

The GA audit found every authenticated `POST /api/app-mcp/mcp` returning
500: `agent_main.py` mounted `app_mcp.http_app()` WITHOUT passing its
lifespan to the parent `FastAPI(...)`, so FastMCP's
`StreamableHTTPSessionManager` task group never started. Mounting a
sub-app never runs that sub-app's lifespan — the composition has to be
explicit, which forces a boot-order change: the MCP app must exist
BEFORE the `FastAPI(...)` constructor.

Two layers of pins:

  1. Source-level (the repo's convention for boot-order invariants —
     see test_agent_defer_boot_init.py: importing agent_main spins up
     the full app, too heavyweight for a unit test):
       * the MCP app is built before the FastAPI constructor,
       * the constructor takes the composed lifespan,
       * a build failure falls back to the plain lifespan (fail-open:
         app-MCP degrades, the tenant does NOT lose boot).
  2. Functional (the failure class itself, on a minimal app): mounting
     a FastMCP http_app without composing its lifespan 500s exactly
     like production did; composing it serves. This half is the
     red-proof twin — it fails the broken shape forever, so the class
     cannot silently come back.

The live acceptance stays what the ledger says: canary boots clean,
valid-key POST returns JSON-RPC not 500, no-key control still 401s.
"""

from __future__ import annotations

import re
from pathlib import Path


def _agent_main_src() -> str:
    return (Path(__file__).resolve().parents[1] / "agent_main.py").read_text()


# ── 1. Source-level boot-order pins ─────────────────────────────────


def test_mcp_app_is_built_before_the_fastapi_constructor():
    src = _agent_main_src()
    built_at = src.find(".http_app(path=\"/mcp\")")
    constructed_at = src.find("app = FastAPI(")
    assert built_at != -1, "app_mcp.http_app(path=\"/mcp\") is no longer built"
    assert constructed_at != -1
    assert built_at < constructed_at, (
        "the MCP app is built AFTER the FastAPI constructor — its "
        "lifespan cannot be composed into the agent's, so the "
        "StreamableHTTPSessionManager task group never starts and every "
        "authenticated /api/app-mcp/mcp request 500s (GA audit, B-1)"
    )


def test_agent_lifespan_composes_the_mcp_lifespan():
    src = _agent_main_src()
    m = re.search(
        r"async def _combined_lifespan\(.*?\):\s*\n"
        r"(?:\s*#.*\n)*"
        r"\s*async with lifespan\(.*?\):\s*\n"
        r"(?:\s*#.*\n)*"
        r"\s*async with mcp_app\.lifespan\(.*?\):\s*\n"
        r"(?:\s*#.*\n)*"
        r"\s*yield",
        src,
    )
    assert m, (
        "_combined_lifespan must nest mcp_app.lifespan INSIDE the "
        "agent's own lifespan (agent services up first, MCP session "
        "manager second, torn down in reverse)"
    )
    assert "lifespan=_boot_lifespan" in src, (
        "the FastAPI constructor no longer takes the composed lifespan"
    )


def test_mcp_build_failure_stays_fail_open():
    """The old mount sat in try/except so a broken MCP import degraded
    one surface instead of stopping boot. The boot-order fix must keep
    that property — agent_main is the highest-blast-radius file in the
    system."""
    src = _agent_main_src()
    assert re.search(r"except Exception as \w+:\s*\n\s*mcp_app = None", src), (
        "building the MCP app before the constructor must not introduce "
        "a new way for a tenant container to fail boot — on build "
        "failure, mcp_app must become None"
    )
    assert re.search(r"else:\s*\n\s*_boot_lifespan = lifespan", src), (
        "when the MCP app could not be built the agent must boot on its "
        "plain lifespan (fail-open)"
    )
    assert re.search(r"if mcp_app is not None:\s*\n\s*app\.mount\(\"/api/app-mcp\"", src), (
        "the mount must reuse the pre-built app and skip cleanly when "
        "the build failed"
    )


# ── 2. Functional twin — the failure class on a minimal app ─────────


_INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-03-26",
        "capabilities": {},
        "clientInfo": {"name": "pin", "version": "0"},
    },
}

_MCP_HEADERS = {
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json",
}


def _mini_mcp_http_app():
    from fastmcp import FastMCP

    server = FastMCP("b1-pin")

    @server.tool()
    def ping() -> str:
        return "pong"

    return server.http_app(path="/mcp")


async def _post_initialize(parent):
    """POST the MCP initialize handshake through the parent app's
    ASGI stack (no real server; app exceptions surface as 500s, the
    same shape uvicorn gives a live client)."""
    import httpx

    transport = httpx.ASGITransport(app=parent, raise_app_exceptions=False)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://pin", follow_redirects=True
    ) as client:
        return await client.post(
            "/api/app-mcp/mcp", json=_INITIALIZE, headers=_MCP_HEADERS
        )


async def test_mounting_without_composed_lifespan_fails_exactly_like_prod():
    """The broken shape — mount only, parent keeps its own lifespan —
    must 500 on the first authenticated request. If a FastMCP upgrade
    ever makes this pass, the boot-order constraint is gone and the
    source pins above can be retired with it."""
    from fastapi import FastAPI

    mcp_http = _mini_mcp_http_app()
    parent = FastAPI()  # NOTE: no lifespan composition — the bug
    parent.mount("/api/app-mcp", mcp_http)

    async with parent.router.lifespan_context(parent):
        r = await _post_initialize(parent)
    assert r.status_code == 500, (
        f"expected the un-composed mount to 500 (task group never "
        f"started); got {r.status_code} — the failure class this pin "
        f"guards has changed shape, re-verify B-1"
    )


async def test_mounting_with_composed_lifespan_serves():
    """The fixed shape — sub-app lifespan composed into the parent's —
    must answer the same request with a real JSON-RPC response."""
    from contextlib import asynccontextmanager

    from fastapi import FastAPI

    mcp_http = _mini_mcp_http_app()

    @asynccontextmanager
    async def _composed(app):
        async with mcp_http.lifespan(app):
            yield

    parent = FastAPI(lifespan=_composed)
    parent.mount("/api/app-mcp", mcp_http)

    async with parent.router.lifespan_context(parent):
        r = await _post_initialize(parent)
    assert r.status_code != 500, (
        "composed lifespan still 500s — the pre-staged fix shape does "
        "not actually start the session manager"
    )
    assert r.status_code == 200, f"expected JSON-RPC 200, got {r.status_code}"
    assert "serverInfo" in r.text, "initialize did not return a server description"
