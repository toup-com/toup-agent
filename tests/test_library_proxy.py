"""The platform side of the file library (app/api/workspace_proxy.py).

The platform's whole job here is: authenticate the caller, find THEIR
agent, forward the request verbatim with X-Agent-Key, stream the answer
back — and fail as a 503 (never an empty listing) when the agent is not
there. These tests pin exactly what crosses the wire, using a mock
transport in place of the tenant agent.
"""

from __future__ import annotations

import json
import uuid

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

AGENT_URL = "http://10.0.0.7:9123"
AGENT_KEY = "agent-key-xyz"


@pytest_asyncio.fixture
async def platform(monkeypatch):
    from fastapi import FastAPI
    from app.api.workspace_proxy import router
    from app.config import settings
    monkeypatch.setattr(settings, "run_mode", "platform")
    app = FastAPI()
    app.include_router(router, prefix=settings.api_prefix)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://platform") as ac:
        yield ac


@pytest_asyncio.fixture
async def agent_config(test_user_id):
    from app.db import async_session_maker
    from app.db.models import AgentConfig
    async with async_session_maker() as db:
        db.add(AgentConfig(user_id=test_user_id, agent_url=AGENT_URL, agent_api_key=AGENT_KEY,
                           deploy_status="active"))
        await db.commit()


@pytest.fixture
def agent(monkeypatch):
    """A fake tenant agent: records every request, answers from `respond`."""
    from app.services import agent_http
    seen: list[httpx.Request] = []
    state = {"respond": lambda req: httpx.Response(200, json={"ok": True})}

    async def handler(request: httpx.Request) -> httpx.Response:
        # materialise a streamed body so tests can inspect it
        body = b""
        async for chunk in request.stream:  # type: ignore[union-attr]
            body += chunk
        request._body = body  # noqa: SLF001
        seen.append(request)
        return state["respond"](request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(agent_http, "_client", client)
    state["seen"] = seen
    return state


# ── Forwarding ───────────────────────────────────────────────────────

async def test_listing_is_forwarded_verbatim_with_agent_key(platform, auth_headers, agent_config, agent):
    agent["respond"] = lambda req: httpx.Response(
        200, json={"path": "", "files": [{"name": "Documents", "type": "dir"}], "base": "/", "curated": True})
    r = await platform.get("/api/workspace/files", params={"path": "Documents/Q3", "refresh": "1"},
                           headers=auth_headers)
    assert r.status_code == 200 and r.json()["base"] == "/"
    req = agent["seen"][0]
    assert req.method == "GET"
    assert str(req.url) == f"{AGENT_URL}/api/workspace/files?path=Documents%2FQ3&refresh=1"
    assert req.headers["x-agent-key"] == AGENT_KEY
    assert "authorization" not in req.headers          # the user's JWT never reaches the agent


async def test_library_routes_forward_method_body_and_content_type(platform, auth_headers, agent_config, agent):
    agent["respond"] = lambda req: httpx.Response(201, json={"id": "f1", "name": "Reports"})
    r = await platform.post("/api/library/folders", json={"name": "Reports"}, headers=auth_headers)
    assert r.status_code == 201 and r.json()["id"] == "f1"
    req = agent["seen"][-1]
    assert req.method == "POST" and str(req.url) == f"{AGENT_URL}/api/library/folders"
    assert req.headers["content-type"].startswith("application/json")
    assert json.loads(req._body) == {"name": "Reports"}  # noqa: SLF001

    for method, url in (("patch", "/api/library/files/abc"), ("delete", "/api/library/files/abc"),
                        ("put", "/api/library/files/abc/content")):
        agent["respond"] = lambda req: httpx.Response(200, json={"ok": True})
        r = await getattr(platform, method)(url, headers=auth_headers, **({} if method == "delete" else {"json": {"x": 1}}))
        assert r.status_code == 200, (method, r.text)
        assert agent["seen"][-1].method == method.upper()
        assert str(agent["seen"][-1].url) == f"{AGENT_URL}{url}"


async def test_multipart_upload_streams_through_untouched(platform, auth_headers, agent_config, agent):
    agent["respond"] = lambda req: httpx.Response(201, json={"success": True, "path": "Uploads/a.txt"})
    r = await platform.post("/api/workspace/file-upload", params={"path": ""},
                            files={"file": ("a.txt", b"hello", "text/plain")}, headers=auth_headers)
    assert r.status_code == 201
    req = agent["seen"][-1]
    assert req.headers["content-type"].startswith("multipart/form-data; boundary=")
    assert b'filename="a.txt"' in req._body and b"hello" in req._body  # noqa: SLF001
    assert req.headers.get("content-length") == str(len(req._body))  # noqa: SLF001


async def test_binary_download_streams_back_with_headers(platform, auth_headers, agent_config, agent, _test_user):
    payload = b"%PDF-1.4 " + b"x" * 5000
    agent["respond"] = lambda req: httpx.Response(
        200, content=payload,
        headers={"content-type": "application/pdf",
                 "content-disposition": 'inline; filename="Q3.pdf"',
                 "cache-control": "private, max-age=3600",
                 "x-content-type-options": "nosniff",
                 "set-cookie": "leak=1", "server": "uvicorn"})
    r = await platform.get("/api/library/files/f1/download", params={"inline": "1", "token": _test_user["token"]})
    assert r.status_code == 200 and r.content == payload
    assert r.headers["content-type"] == "application/pdf"
    assert r.headers["content-disposition"] == 'inline; filename="Q3.pdf"'
    assert r.headers["x-content-type-options"] == "nosniff"
    assert "set-cookie" not in r.headers and r.headers.get("server") != "uvicorn"
    req = agent["seen"][-1]
    assert "token" not in dict(req.url.params)          # ?token= is the caller's credential, not the agent's
    assert req.url.params["inline"] == "1"


async def test_agent_4xx_is_relayed_as_is(platform, auth_headers, agent_config, agent):
    agent["respond"] = lambda req: httpx.Response(
        409, json={"detail": "Folder is not empty", "code": "folder_not_empty"})
    r = await platform.delete("/api/library/folders/f1", headers=auth_headers)
    assert r.status_code == 409 and r.json() == {"detail": "Folder is not empty", "code": "folder_not_empty"}
    agent["respond"] = lambda req: httpx.Response(404, json={"detail": "File not found"})
    r = await platform.get("/api/library/files/nope", headers=auth_headers)
    assert r.status_code == 404 and r.json()["detail"] == "File not found"


# ── Failure semantics ────────────────────────────────────────────────

async def test_no_agent_is_503_not_an_empty_listing(platform, auth_headers, agent):
    r = await platform.get("/api/workspace/files", headers=auth_headers)
    assert r.status_code == 503
    assert "files" not in r.json() and "agent" in r.json()["detail"].lower()
    assert agent["seen"] == []


async def test_unreachable_agent_is_503(platform, auth_headers, agent_config, agent):
    def boom(req):
        raise httpx.ConnectError("refused", request=req)
    agent["respond"] = boom
    r = await platform.get("/api/library/files", headers=auth_headers)
    assert r.status_code == 503


async def test_pre_library_agent_image_reads_as_updating(platform, auth_headers, agent_config, agent):
    # FastAPI's default 404 for an unmounted route — an old agent image
    agent["respond"] = lambda req: httpx.Response(404, json={"detail": "Not Found"})
    r = await platform.get("/api/library/files", headers=auth_headers)
    assert r.status_code == 503 and "updated" in r.json()["detail"]


# ── Auth ─────────────────────────────────────────────────────────────

async def test_unauthenticated_is_401_and_never_reaches_the_agent(platform, agent_config, agent):
    for url in ("/api/workspace/files", "/api/library/files", "/api/library/files/f1/download"):
        r = await platform.get(url)
        assert r.status_code == 401, url
    r = await platform.post("/api/library/folders", json={"name": "x"})
    assert r.status_code == 401
    assert agent["seen"] == []


async def test_token_query_param_only_works_on_embed_routes(platform, agent_config, agent, _test_user):
    agent["respond"] = lambda req: httpx.Response(200, content=b"bytes", headers={"content-type": "image/png"})
    r = await platform.get("/api/workspace/file-download", params={"path": "Images/a.png", "token": _test_user["token"]})
    assert r.status_code == 200 and r.content == b"bytes"
    r = await platform.get("/api/library/files/f1/preview", params={"token": _test_user["token"]})
    assert r.status_code == 200
    # a JSON route with only ?token= is not authenticated (session auth required)
    r = await platform.get("/api/library/files", params={"token": _test_user["token"]})
    assert r.status_code == 401
    r = await platform.delete("/api/library/files/f1", params={"token": _test_user["token"]})
    assert r.status_code == 401


async def test_only_the_callers_own_agent_is_used(platform, auth_headers, agent):
    """Another user's AgentConfig row must never be picked up."""
    from app.db import async_session_maker
    from app.db.models import AgentConfig, User
    other = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(id=other, email=f"o-{other[:8]}@example.com", hashed_password="x", name="Other"))
        db.add(AgentConfig(user_id=other, agent_url="http://10.0.0.9:9999", agent_api_key="theirs",
                           deploy_status="active"))
        await db.commit()
    r = await platform.get("/api/workspace/files", headers=auth_headers)
    assert r.status_code == 503 and agent["seen"] == []


async def test_removed_preview_route_is_gone_at_the_platform_too(platform, auth_headers, agent_config, agent):
    # The platform no longer implements /workspace/preview itself; it forwards
    # to the agent, whose answer for it is 410.
    agent["respond"] = lambda req: httpx.Response(410, json={"detail": "removed"})
    r = await platform.get("/api/workspace/preview/vibecoding/todo/index.html", headers=auth_headers)
    assert r.status_code == 410


def test_module_has_no_shell_or_host_paths():
    """The old module ran ssh/docker exec against host paths and stripped
    `..` from client input. None of that may come back."""
    import inspect
    from app.api import workspace_proxy
    src = inspect.getsource(workspace_proxy)
    body = src.split('"""', 2)[2]  # skip the module docstring, which describes the past
    for forbidden in ("_ssh_cmd", "docker exec", "/data/agents", "/app/workspace",
                      'replace("..", "")', "shlex", "subprocess", "create_subprocess_exec", "base64"):
        assert forbidden not in body, forbidden
