"""The `hop=registered` line: t0 of every `[signup-trace]` trail.

Split out of test_signup_trace_wiring.py for ROUTING, not preference. This one
test drives the real `POST /api/auth/register`, and `create_user` seeds the
new user's default `identities` rows — an AGENT_ONLY table that `init_db()`
does not build under the platform sweep's RUN_MODE=platform (CI run
34151220144: "no such table: identities" → PendingRollbackError → 500). The
rest of that suite proves a reclaim is not a signup over `managed_containers`,
which is PLATFORM_ONLY; no single lane holds both tables, so the register hop
lives here and runs in the agent-mode step (see tests/COVERAGE_DEBT.txt), the
same lane as test_signup_hardening.py, the other suite that registers users.

Why the hop matters is in test_signup_trace_wiring.py's docstring: without it,
`elapsed_ms` is measured from whichever hop this replica happened to see first.
"""
from __future__ import annotations

import logging
import os
import uuid

os.environ.setdefault("ENVIRONMENT", "test")

import pytest


@pytest.fixture(autouse=True)
def _no_leaked_discovery_loops():
    """`ensure_discovery` outlives its caller on purpose; in a test that is a
    task still holding a sqlite session when conftest drops the schema."""
    import asyncio
    yield
    from app.services import pool_service as ps
    try:
        asyncio.get_running_loop()
        running = True
    except RuntimeError:
        running = False
    for t in list(asyncio.all_tasks()) if running else []:
        if not t.done() and (t.get_name() or "").startswith(("discover:", "adopt-once:")):
            t.cancel()
    ps._DISCOVERY_INFLIGHT.clear()
    ps._invalidate_pool_list_cache()


def _trace_lines(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if "[signup-trace]" in r.getMessage()]


def _field(line: str, key: str) -> str:
    for tok in line.split():
        if tok.startswith(key + "="):
            return tok[len(key) + 1:]
    return ""


@pytest.mark.asyncio
async def test_register_seeds_the_trace_and_records_the_registered_hop(client, caplog):
    """FALSIFIER: on the unchanged tree `auth.register` calls neither
    `seed_signup_trace` nor `signup_trace`, so no `hop=registered` line
    exists and every later elapsed_ms is measured from the wrong instant."""
    email = f"trace-{uuid.uuid4().hex[:10]}@example.com"
    with caplog.at_level(logging.INFO, logger="app.services.pool_service"):
        r = await client.post(
            "/api/auth/register",
            json={"email": email, "password": "password1234", "name": "TraceUser"},
        )
    assert r.status_code == 201, r.text
    uid = r.json()["id"]

    lines = [ln for ln in _trace_lines(caplog) if f"user={uid[:8]} " in ln]
    registered = [ln for ln in lines if "hop=registered" in ln]
    assert registered, (
        f"no hop=registered line for the user that just registered; got "
        f"{lines!r}. Without it nothing stamps t0 and every elapsed_ms in the "
        f"trail is measured from whichever hop this replica saw first"
    )
    assert _field(registered[0], "origin") == "signup", (
        f"the registration hop must be tagged as a signup; got "
        f"{registered[0]!r}"
    )
    assert _field(registered[0], "detail") == "via=password", (
        f"the hop must come from `register` itself, not from the background "
        f"finalizer it spawns — the finalizer runs AFTER the response and "
        f"cannot be t0; got {registered[0]!r}"
    )
    assert uid not in registered[0], "only the 8-hex prefix may appear"
    assert len(registered) == 1, (
        f"one signup, one `registered` hop; got {len(registered)}. "
        f"`_bg_finalize_signup` runs in the SAME process right after register "
        f"— an unconditional re-seed there moves t0 past the work it is "
        f"supposed to measure and prints the hop twice: {registered!r}"
    )
