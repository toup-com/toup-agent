"""`alembic upgrade head` fails on every tenant container and nothing knew.

Measured read-only on the production host on 2026-09-07. Three containers on
three different paths — the dedicated canary `toup-agent-533354ce`, the bound
pool slot `toup-agent-pool-47` and a generic slot spawned that day,
`toup-agent-pool-85` — all log

    sqlalchemy.exc.ProgrammingError: (psycopg2.errors.DuplicateTable)
    relation "users" already exists

from inside the FIRST migration, and all three tenant databases answer NULL to
`SELECT to_regclass('alembic_version')`: the version table does not exist. The
columns those containers use come from `app/db/database.py`'s
`ADD COLUMN IF NOT EXISTS` self-heal, not from the migration chain.

`Dockerfile.agent`'s `alembic upgrade head || echo '…'` swallowed the failure
into one sentence mid-boot, with no marker any instrument could read. So the
state was invisible to `/agent/health`, to the bridge's health tick and to the
rollout canary — all of which reported these containers as healthy, which they
are.

This file pins the OBSERVABILITY, not the mechanism. The mechanism (stamp the
fleet vs. make the self-heal canonical) is a separate decision; see the PR.
"""
from __future__ import annotations

import json
import os
import pathlib
import re

import pytest

os.environ.setdefault("ENVIRONMENT", "test")

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


# ── The boot marker ───────────────────────────────────────────────────────

def test_the_boot_cmd_records_the_outcome_where_something_can_read_it():
    src = (_BACKEND / "Dockerfile.agent").read_text()
    cmd = [l for l in src.splitlines() if l.startswith("CMD [")]
    assert len(cmd) == 1, "Dockerfile.agent has no single CMD line"
    argv = json.loads(cmd[0][len("CMD "):])
    script = argv[2]
    # The failure must be captured, not merely echoed into the void.
    assert "TOUP_ALEMBIC_BOOT_MARKER" in script, \
        "the boot outcome is still unreadable to anything but a human"
    assert "ALEMBIC_BOOT=" in script, "no distinctive log line to grep the fleet for"
    # Both outcomes recorded — a marker only written on success is a marker
    # that says 'unknown' for exactly the case that matters.
    assert re.search(r"if alembic upgrade head; then\s+\w+=ok; else\s+\w+=failed; fi", script), \
        "the CMD does not branch on the upgrade's exit status"
    # …and the container still serves either way (the swallow is deliberate).
    assert script.rstrip().endswith("--port 8001"), "boot no longer execs uvicorn"
    assert "exec uvicorn" in script
    assert "TOUP_ALEMBIC_BOOT_MARKER" in src.split("CMD [")[0], \
        "the marker path has no ENV default, so a bare `docker run` writes nowhere"


def test_the_marker_reader_answers_ok_failed_or_unknown(tmp_path, monkeypatch):
    from agent_main import read_alembic_boot_marker, _ALEMBIC_BOOT_MARKER_ENV

    marker = tmp_path / "m"
    monkeypatch.setenv(_ALEMBIC_BOOT_MARKER_ENV, str(marker))

    # Absent file — an image built before the marker existed, or a read-only
    # /tmp. This must NOT read as healthy.
    assert read_alembic_boot_marker() == "unknown"

    marker.write_text("ok\n")
    assert read_alembic_boot_marker() == "ok"
    marker.write_text("failed\n")
    assert read_alembic_boot_marker() == "failed"
    marker.write_text("something else")
    assert read_alembic_boot_marker() == "unknown"


# ── The health field ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_alembic_ok_needs_BOTH_a_clean_boot_and_a_stamped_table(monkeypatch):
    """The production shape is `boot=failed, version=None`. But `boot=ok` with
    no version table is equally dead — `alembic upgrade head` exits 0 when it
    believes there is nothing to do — and a stamped table proves nothing if
    THIS boot's upgrade blew up. Both halves, or the field lies."""
    import agent_main

    async def _version(value):
        return value

    for boot, version, expected in (
        ("ok", "101", True),
        ("ok", None, False),
        ("failed", "101", False),
        ("failed", None, False),
        ("unknown", "101", False),
    ):
        monkeypatch.setattr(agent_main, "read_alembic_boot_marker", lambda b=boot: b)
        monkeypatch.setattr(
            agent_main, "read_alembic_version",
            lambda v=version: _version(v),
        )
        monkeypatch.setattr(agent_main, "_schema_status_cache", None)
        got = await agent_main.agent_schema_status(refresh=True)
        assert got["alembic_boot"] == boot
        assert got["alembic_version"] == version
        assert got["alembic_ok"] is expected, (boot, version)


@pytest.mark.asyncio
async def test_the_version_read_answers_none_rather_than_raising(monkeypatch):
    """Every container in production is in exactly this state, and a health
    field that raises takes the endpoint the bridge polls with it."""
    import agent_main

    class _Boom:
        def __call__(self, *a, **k):
            raise RuntimeError("no database")

    monkeypatch.setattr("app.db.database.async_session_maker", _Boom())
    assert await agent_main.read_alembic_version() is None


def test_health_reports_the_schema_object():
    import inspect
    import agent_main

    src = inspect.getsource(agent_main.agent_health)
    assert "agent_schema_status()" in src, "/agent/health never asks"
    assert '"schema": _schema' in src, "/agent/health never reports it"
    # It must not be able to take the endpoint down.
    at = src.find("agent_schema_status()")
    assert "except Exception" in src[at:at + 400], \
        "the schema probe is not fenced; a DB blip would 500 the health route"


def test_the_probe_is_cached_so_health_polls_do_not_query_per_call(monkeypatch):
    """The bridge polls this route every few seconds across ~95 containers, on
    a host whose Postgres has repeatedly sat at 297+ of 300 backends."""
    import asyncio
    import agent_main

    calls = {"n": 0}

    async def _counted():
        calls["n"] += 1
        return "101"

    monkeypatch.setattr(agent_main, "read_alembic_boot_marker", lambda: "ok")
    monkeypatch.setattr(agent_main, "read_alembic_version", _counted)
    monkeypatch.setattr(agent_main, "_schema_status_cache", None)

    async def _drive():
        for _ in range(5):
            await agent_main.agent_schema_status()

    asyncio.run(_drive())
    assert calls["n"] == 1, f"queried {calls['n']} times for 5 health polls"
