"""The agent must SERVE the identity documents it stores.

`identities` is AGENT_ONLY (app/db/models/base.py), and `PUT /api/soul`
compiles into it on the agent itself. But `app/api/identity.py` — the only
module in the repo with `prefix="/identity"` — was mounted by
`platform_main.py` alone. Nothing on the agent served it.

Meanwhile `app/api/ws_realtime.py` opens every voice session by prefetching
four reads from the user's container, and the FIRST of them is

    _vps_api(agent_url, agent_api_key, "GET", "/api/identity",
             params={"active_only": "true"})

Measured against the live canary container on 2026-08-04:

    /api/identity -> 404
    /api/sessions -> 200      (control: X-Agent-Key auth works)
    /api/soul     -> 200      (control: another agent route answers)

`_vps_api` returns None rather than raising, and the consumer is guarded by
`if identities_data:` — so the 404 was silent. The cost is not an error
message, it is four missing sections of the voice agent's system prompt:
`# Core Identity`, `# Behavioral Guidelines`, `# About the User` and
`# Tool Guidelines`. Every voice session ran without its persona,
its behavioural rules, the user profile, and its tool guidance.

Same shape as #431: a missing route plus a defensive guard equals silent
quality loss, with nothing failing loudly enough for anyone to notice. Both
`git log -S` trails start at the fc5c056b repo restructure, so this had been
true since the platform/agent split.

Run:
    cd backend && RUN_MODE=agent PYTHONPATH=. pytest tests/test_agent_serves_identity.py
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import pytest


BACKEND = pathlib.Path(__file__).resolve().parents[1]


def _routes_of(entrypoint: str, run_mode: str) -> list[str]:
    """Import an entrypoint in a SUBPROCESS and return its route paths.

    A subprocess because importing `agent_main` runs real module-level wiring;
    doing that inside the pytest process would leak it into every later test in
    the file. Hermetic env for the same reason `test_sqlite_test_infra` uses
    one: an ambient value must not be able to decide the answer.

    This asserts on the app object FastAPI actually builds, not on the text of
    the entrypoint. A grep for `include_router(identity_router` would pass
    against a line that had been commented out below it, or against an import
    that never got mounted — which is precisely the bug being fixed here.
    """
    env = {
        "PATH": os.environ["PATH"],
        "HOME": os.environ.get("HOME", "/tmp"),
        "LANG": "C",
        "ENVIRONMENT": "test",
        "RUN_MODE": run_mode,
        "PYTHONPATH": str(BACKEND),
        "JWT_SECRET": "test-jwt-secret-for-ci",
        "ENCRYPTION_KEY": "test-32-byte-encryption-key--x12",
        "DATABASE_URL": "sqlite+aiosqlite:///file::memory:?cache=shared&uri=true",
    }
    code = (
        f"import {entrypoint} as m, json;"
        f"print(json.dumps(sorted({{r.path for r in m.app.routes}})))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=str(BACKEND), env=env,
        capture_output=True, text=True, timeout=300,
    )
    assert out.returncode == 0, (
        f"importing {entrypoint} under RUN_MODE={run_mode} failed:\n"
        f"{out.stderr[-2000:]}"
    )
    import json
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def agent_routes() -> list[str]:
    return _routes_of("agent_main", "agent")


def test_agent_mounts_the_identity_router(agent_routes):
    """GET /api/identity must exist on the agent.

    MUTATION: remove `app.include_router(identity_router, ...)` from
    agent_main.py -> red. Verified by running this against the pre-fix tree,
    where the identity route list is exactly [].
    """
    assert "/api/identity" in agent_routes, (
        "the agent does not serve /api/identity, so ws_realtime's voice "
        "bootstrap prefetch 404s and every voice session loses its "
        "# Core Identity / # Behavioral Guidelines / # About the User / "
        "# Tool Guidelines sections — silently, because _vps_api returns "
        "None and the consumer is guarded by `if identities_data:`"
    )


def test_the_identity_read_path_ws_realtime_uses_is_present(agent_routes):
    """Specifically the shape ws_realtime calls: a collection GET plus the
    per-document routes the platform uses to supersede an old identity."""
    for path in ("/api/identity", "/api/identity/{identity_id}"):
        assert path in agent_routes, f"agent is missing {path}"


def test_the_route_list_is_not_vacuously_empty(agent_routes):
    """Anti-vacuity control.

    If the subprocess import silently produced an empty list, every assertion
    above would fail for the wrong reason — or, had they been written as
    `not in`, would have PASSED for the wrong reason. Pin a route that has
    been on the agent all along, so "the import worked" is itself asserted.
    """
    assert len(agent_routes) > 50, f"only {len(agent_routes)} routes — import likely degraded"
    assert "/api/soul" in agent_routes, (
        "/api/soul is missing too — this is not an identity-mounting problem, "
        "the agent app did not build correctly"
    )


def test_identity_and_soul_are_two_views_of_the_same_agent_data(agent_routes):
    """Why this belongs on the agent at all, pinned so it is not 'fixed' by
    moving the router back to the platform.

    `PUT /api/soul` compiles the soul config INTO an Identity row, and
    `identities` is AGENT_ONLY — so the write side already lives here. A read
    route on the platform can only work by reading the platform's own
    `identities` table, which exists solely as a monolith leftover: it is
    absent from any freshly-partitioned platform DB.
    """
    from app.db.models.base import AGENT_ONLY_TABLES

    assert "identities" in AGENT_ONLY_TABLES, (
        "identities is no longer AGENT_ONLY — if it moved to SHARED, the "
        "platform would get an EMPTY copy and the identity endpoints would "
        "start answering with nothing instead of failing, which is worse"
    )
    assert "/api/soul" in agent_routes and "/api/identity" in agent_routes, (
        "the write view (/api/soul) and the read view (/api/identity) must be "
        "served by the same process — they are backed by the same table"
    )
