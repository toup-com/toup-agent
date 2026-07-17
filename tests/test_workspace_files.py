"""Agent-side workspace file surfacing (app/api/workspace_files.py).

Pool containers keep /app/workspace entirely inside the container (no
host bind), so these agent routes are the only live view of files that
write_file produces — the platform proxy and mobile Workspace screen
depend on them. Security posture: client paths resolve() against the
workspace root and must stay is_relative_to() it — traversal, absolute
paths, and symlink escapes all 404 (never 403; no existence leak).
"""

from __future__ import annotations

import os

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "agent_workspace_dir", str(tmp_path))
    return tmp_path


@pytest.fixture
async def client(workspace):
    from fastapi import FastAPI
    from app.api.workspace_files import router

    app = FastAPI()
    app.include_router(router, prefix="/api")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://agent") as ac:
        yield ac


# ── Listing ───────────────────────────────────────────────────────────

async def test_listing_returns_files_dirs_first_and_skips_dotfiles(client, workspace):
    (workspace / "report.md").write_text("# Report")
    (workspace / "notes.txt").write_text("notes")
    (workspace / "data").mkdir()
    (workspace / ".dashboard").mkdir()          # agent internals — hidden
    (workspace / ".hidden.md").write_text("x")  # dotfile — hidden

    resp = await client.get("/api/workspace/files")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["path"] == "" and body["base"] == "/workspace"

    names = [f["name"] for f in body["files"]]
    assert names == ["data", "notes.txt", "report.md"]  # dirs first, then sorted
    by_name = {f["name"]: f for f in body["files"]}
    assert by_name["data"]["type"] == "dir"
    assert by_name["report.md"]["type"] == "file"
    assert by_name["report.md"]["size"] == len("# Report")
    assert by_name["report.md"]["modified"]  # ISO timestamp present


async def test_listing_subdirectory(client, workspace):
    sub = workspace / "reports"
    sub.mkdir()
    (sub / "q3.md").write_text("q3")

    resp = await client.get("/api/workspace/files", params={"path": "reports"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["path"] == "reports"
    assert [f["name"] for f in body["files"]] == ["q3.md"]


async def test_listing_missing_dir_404(client):
    resp = await client.get("/api/workspace/files", params={"path": "nope"})
    assert resp.status_code == 404


# ── Traversal / escape attempts ──────────────────────────────────────

async def test_listing_traversal_and_absolute_404(client):
    for attempt in ("../..", "..", "/etc", "/"):
        resp = await client.get("/api/workspace/files", params={"path": attempt})
        assert resp.status_code == 404, f"{attempt!r} must 404, got {resp.status_code}"


async def test_file_traversal_and_absolute_404(client):
    # /etc/passwd exists on the test host — the guard must still 404 it.
    for attempt in ("../../etc/passwd", "/etc/passwd", "reports/../../etc/passwd"):
        resp = await client.get("/api/workspace/file", params={"path": attempt})
        assert resp.status_code == 404, f"{attempt!r} must 404, got {resp.status_code}"


async def test_symlink_escape_404(client, workspace):
    # resolve() follows the link BEFORE the containment check, so a
    # symlink pointing outside the workspace fails is_relative_to().
    secret = workspace.parent / "secret.txt"
    secret.write_text("out of bounds")
    os.symlink(secret, workspace / "link.md")

    resp = await client.get("/api/workspace/file", params={"path": "link.md"})
    assert resp.status_code == 404


# ── File read ─────────────────────────────────────────────────────────

async def test_file_read_roundtrip(client, workspace):
    content = "# UI/UX Report\n\nFindings…\n"
    (workspace / "report.md").write_text(content, encoding="utf-8")

    resp = await client.get("/api/workspace/file", params={"path": "report.md"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["path"] == "report.md"
    assert body["content"] == content
    assert body["size"] == len(content.encode())
    assert body["mime"].startswith("text/")
    assert body["modified"]


async def test_file_over_1mb_413(client, workspace):
    (workspace / "big.md").write_bytes(b"x" * 1_000_001)
    resp = await client.get("/api/workspace/file", params={"path": "big.md"})
    assert resp.status_code == 413


async def test_file_missing_404(client):
    resp = await client.get("/api/workspace/file", params={"path": "missing.md"})
    assert resp.status_code == 404


async def test_binary_file_415(client, workspace):
    (workspace / "chart.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    resp = await client.get("/api/workspace/file", params={"path": "chart.png"})
    assert resp.status_code == 415


# ── Dashboard memory routes share the same guard ─────────────────────
# Function-level (not HTTP) because /memory/{directory:path} is
# registered before /memory/file/{filepath:path} and shadows it —
# pre-existing routing, not this feature's to change.

async def test_memory_dir_listing_cannot_escape_workspace(workspace):
    """A ../ escape must return [] (same as a missing dir) instead of
    listing .md files outside the workspace root."""
    from app.api.dashboard import list_memory_dir

    outside = workspace.parent / f"{workspace.name}_outside"
    outside.mkdir()
    (outside / "leak.md").write_text("- leaked")
    assert await list_memory_dir(f"../{outside.name}") == []

    # Nested legit paths keep working (web InsightsPanel depends on this)
    legit = workspace / "memory"
    legit.mkdir()
    (legit / "notes.md").write_text("- note")
    assert await list_memory_dir("memory") == [{"name": "notes.md", "entries": 1}]


async def test_memory_file_read_cannot_escape_workspace(workspace):
    from fastapi import HTTPException
    from app.api.dashboard import get_memory_file

    (workspace / "memory").mkdir()
    (workspace / "memory" / "notes.md").write_text("- note")
    assert (await get_memory_file("memory/notes.md"))["content"] == "- note"

    with pytest.raises(HTTPException) as exc:
        await get_memory_file("../../etc/passwd")
    assert exc.value.status_code == 404


# ── write_file → tappable link pin ───────────────────────────────────

def test_write_file_result_carries_report_link():
    """Pin that tool_executor's write_file success message hands the
    model the toup://report deep link for in-workspace writes — the
    prompt (agent_runner) tells the model to surface it, so silently
    dropping the link would kill the tappable-report feature."""
    import inspect
    from app.agent import tool_executor

    src = inspect.getsource(tool_executor.ToolExecutor._tool_write_file)
    assert "toup://report?path=" in src
    assert "_get_user_workspace" in src  # link only for in-workspace paths
