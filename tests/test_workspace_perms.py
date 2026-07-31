"""Workspace permissions — the agent must keep its own exec sandbox able to
write what the agent wrote.

Regression target: the post-rollout manual
`chmod -R a+rwX /data/agents/<prefix>/workspace/generated`, which had to be
re-run by a human on every hardened tenant after every recreate.
"""

import os
import stat

import pytest

from app.services import workspace_perms


def _mode(path: str) -> int:
    return stat.S_IMODE(os.lstat(path).st_mode)


@pytest.fixture
def ws(tmp_path, monkeypatch):
    """Point the workspace root at a temp dir."""
    root = tmp_path / "workspace"
    root.mkdir()
    monkeypatch.setattr(
        workspace_perms, "workspace_root", lambda: str(root), raising=True
    )
    return root


def test_sweep_widens_root_written_files(ws, monkeypatch):
    """The exact ritual case: a file the root agent wrote in a previous
    container life is unwritable to the dropped uid until the sweep runs."""
    gen = ws / "generated"
    gen.mkdir()
    doc = gen / "report.pdf"
    doc.write_bytes(b"x")
    os.chmod(doc, 0o644)
    os.chmod(gen, 0o755)

    summary = workspace_perms.sweep_workspace_perms()

    assert _mode(doc) == 0o666, "generated file must be writable by the sandbox uid"
    assert _mode(gen) == 0o777, "generated dir must be traversable+writable"
    assert summary["changed"] >= 2


def test_sweep_creates_missing_subdirs(ws):
    workspace_perms.sweep_workspace_perms()
    for sub in workspace_perms.SHARED_SUBDIRS:
        p = ws / sub
        assert p.is_dir(), f"{sub} should be created by the sweep"
        assert _mode(str(p)) == 0o777


def test_sweep_is_idempotent(ws):
    workspace_perms.sweep_workspace_perms()
    second = workspace_perms.sweep_workspace_perms()
    assert second["changed"] == 0, "a second sweep must be a no-op"


def test_sweep_respects_kill_switch(ws, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "workspace_shared_perms_enabled", False, raising=False)
    gen = ws / "generated"
    gen.mkdir()
    doc = gen / "report.pdf"
    doc.write_bytes(b"x")
    os.chmod(doc, 0o600)

    summary = workspace_perms.sweep_workspace_perms()

    assert summary == {"skipped": "disabled"}
    assert _mode(doc) == 0o600, "kill switch must leave permissions untouched"


def test_shared_makedirs_widens_every_level(ws):
    target = ws / "generated" / "nested" / "deep"
    workspace_perms.shared_makedirs(str(target))

    assert _mode(str(target)) == 0o777
    assert _mode(str(ws / "generated" / "nested")) == 0o777
    assert _mode(str(ws / "generated")) == 0o777


def test_shared_makedirs_stops_at_workspace_root(ws):
    """Must not climb above the workspace (i.e. never widen /app)."""
    before = _mode(str(ws.parent))
    workspace_perms.shared_makedirs(str(ws / "generated"))
    assert _mode(str(ws.parent)) == before


def test_share_path_missing_file_is_silent(ws):
    workspace_perms.share_path(str(ws / "generated" / "nope.txt"))  # must not raise


@pytest.mark.asyncio
async def test_local_disk_backend_put_is_sandbox_writable(ws, monkeypatch):
    """file_storage is the document-attachment write path."""
    from app.services.file_storage import LocalDiskBackend

    backend = LocalDiskBackend(root=str(ws))
    await backend.put("abc123_report.pdf", b"data")

    full = backend.path("abc123_report.pdf")
    assert _mode(full) == 0o666
    assert _mode(os.path.dirname(full)) == 0o777
