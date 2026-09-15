"""An unbound generic pool spare must not warm a headless Brave.

`warm_browser()` keeps the browser resident for the life of the process — the
round-24 fix, and correct for a container that will build apps. But agent boot
fired it on EVERY container, bound or not, and the 2026-09-12 host was carrying
672 Brave processes across 96 containers at load 21 on 16 vCPU, ~11 of them in
lobby spares nobody had claimed (~104 MiB PSS each) for a build that cannot be
requested until someone is bound.

So boot gates on `is_bound()` and `/admin/bind` warms the container the moment
it HAS a user — still earlier than the app-build that needs it.
"""
from __future__ import annotations

import asyncio
import os
import pathlib
from typing import Any
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.agent.skills.builtins.app_html import verify
from app.config import settings

_TOKEN = "pool-secret-warm"


@pytest.fixture(autouse=True)
def _clean_warm_state(monkeypatch):
    monkeypatch.setenv("POOL_ADMIN_TOKEN", _TOKEN)
    monkeypatch.setattr(verify, "_browser", None, raising=False)
    monkeypatch.setattr(verify, "_warm_task", None, raising=False)
    yield
    task = getattr(verify, "_warm_task", None)
    if task is not None and not task.done():
        task.cancel()
    verify._warm_task = None


# ── The predicate ────────────────────────────────────────────────────────

def test_an_unbound_container_is_not_allowed_to_warm():
    assert verify.browser_warm_boot_allowed(bound=False) is False


def test_a_bound_container_is_allowed_to_warm():
    assert verify.browser_warm_boot_allowed(bound=True) is True


# ── The boot call site actually uses it ──────────────────────────────────

def test_agent_main_gates_the_boot_warm_up_on_being_bound():
    """A predicate nothing calls is not a gate. Read the real boot block.

    This is the same source-probe shape `test_tenant_cron_spread.py` uses for
    the cron registrations — the lifespan is not unit-constructible, and an
    ungated `create_task` would be invisible to every other check here.
    """
    src = pathlib.Path(__file__).resolve().parents[1].joinpath("agent_main.py").read_text()
    at = src.find("# Warm the verify browser in the background")
    assert at > 0, "the browser warm-up block moved"
    block = src[at:at + 1400]

    assert "browser_warm_boot_allowed" in block, "boot warms unconditionally again"
    assert "runtime_identity.is_bound()" in block, "the gate reads something other than boundness"
    # The old unconditional task must not have come back alongside the gate.
    assert "asyncio.create_task" not in block, (
        "boot schedules its own task again — it must go through "
        "schedule_warm_browser so the bind path shares the idempotency"
    )


# ── Scheduling is idempotent, and the bind path triggers it ──────────────

@pytest.mark.asyncio
async def test_schedule_warm_browser_starts_exactly_one_task():
    calls: list[int] = []

    async def _fake_warm() -> bool:
        calls.append(1)
        await asyncio.sleep(0)
        return True

    with patch.object(verify, "warm_browser", _fake_warm):
        assert verify.schedule_warm_browser("boot") is True
        # A second trigger while the first is in flight must be free.
        assert verify.schedule_warm_browser("bind") is False
        await verify._warm_task
    assert calls == [1]


@pytest.mark.asyncio
async def test_schedule_warm_browser_is_a_no_op_once_a_browser_exists():
    """`refresh-config` is routine and both Railway replicas push one."""
    verify._browser = object()
    with patch.object(verify, "warm_browser") as spy:
        assert verify.schedule_warm_browser("bind") is False
    spy.assert_not_called()


async def _bind(payload: dict[str, Any]) -> int:
    from app.api.admin_pool import router as admin_router

    app = FastAPI()
    app.include_router(admin_router)
    saved = {"user_id": settings.user_id, "agent_api_key": settings.agent_api_key}
    saved_env = {k: os.environ.get(k) for k in ("USER_ID", "AGENT_API_KEY")}
    try:
        with patch("app.services.runtime_identity.write_runtime"), patch(
            "app.agent.mcp_bootstrap.ensure_mcp_initialized", return_value="skipped"
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://testserver") as http:
                r = await http.post(
                    "/admin/bind", json=payload, headers={"X-Pool-Admin-Token": _TOKEN}
                )
                return r.status_code
    finally:
        for k, v in saved.items():
            object.__setattr__(settings, k, v)
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@pytest.mark.asyncio
async def test_bind_warms_the_browser_once_across_repeat_binds():
    """The other half of the gate: a claimed container still gets a browser,
    and the second replica's refresh-config does not start a second one."""
    calls: list[str] = []

    async def _fake_warm() -> bool:
        calls.append("warm")
        await asyncio.sleep(0)
        return True

    with patch.object(verify, "warm_browser", _fake_warm):
        assert await _bind({"user_id": "u-warm", "agent_api_key": "k1"}) == 200
        assert verify._warm_task is not None, "bind did not schedule the warm-up"
        await verify._warm_task
        assert calls == ["warm"]

        # Second bind, browser now up: must not schedule again.
        verify._browser = object()
        assert await _bind({"user_id": "u-warm", "agent_api_key": "k1"}) == 200
        assert calls == ["warm"]
