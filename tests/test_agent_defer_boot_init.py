"""Regression test for TKT-LAT-017 — defer non-essential agent boot init.

When `agent_defer_boot_init` is True (default), the lifespan should:
  * Not block on mcp_tools_cache.refresh() — schedule as background task.
  * Not block on tunnel_client.start() — schedule as background task.
  * Emit `[PERF] boot_deferred=...` log lines so deferral is observable.

We verify the source-level invariants directly (importing agent_main
spins up the full FastAPI app + lifespan, which is too heavyweight for
a unit test).
"""

from __future__ import annotations

from pathlib import Path


def _agent_main_src() -> str:
    return Path(
        Path(__file__).resolve().parents[1] / "agent_main.py"
    ).read_text()


def test_flag_defaults_to_on():
    """The defer-boot-init flag must default to True — that's the
    whole point of the ticket. If a future refactor flips this to
    False, the win silently disappears.

    Read the source to avoid pydantic-settings env-file validation
    side effects (the prod .env may include keys outside this
    Settings class's declared fields)."""
    src = Path(
        Path(__file__).resolve().parents[1] / "app" / "config.py"
    ).read_text()
    assert "agent_defer_boot_init: bool = True" in src


def test_mcp_refresh_is_deferred_behind_flag():
    """The MCP cache refresh must be guarded by the flag and schedule
    a named asyncio task when on."""
    src = _agent_main_src()
    # The refresh helper is wrapped in an inner async fn called
    # _boot_refresh_mcp; verify both the flag-gated dispatch and the
    # named task exist.
    assert "_boot_refresh_mcp" in src
    assert 'asyncio.create_task(\n                    _boot_refresh_mcp()' in src \
        or "create_task(_boot_refresh_mcp" in src
    assert "lat017-mcp-refresh" in src
    assert "boot_deferred=mcp_refresh" in src
    assert "settings.agent_defer_boot_init" in src


def test_tunnel_start_is_deferred_behind_flag():
    """Same contract for the platform tunnel start."""
    src = _agent_main_src()
    assert "_boot_start_tunnel" in src
    assert "lat017-tunnel-start" in src
    assert "boot_deferred=tunnel_start" in src


def test_blocking_fallback_still_exists_when_flag_off():
    """If the operator flips the flag off, the original blocking
    semantics must still be reachable. Source must contain an `else:
    await _boot_refresh_mcp()` and `else: await _boot_start_tunnel()`
    branch — not just the deferral path."""
    src = _agent_main_src()
    assert "else:\n                await _boot_refresh_mcp()" in src
    assert "else:\n                await _boot_start_tunnel()" in src


def test_perf_log_records_actual_wall_time():
    """Each deferred task must record its own wall-time so we can
    correlate `boot_deferred=mcp_refresh` with the actual cost when it
    finally completes (e.g. `boot_mcp_refresh_ms=750`). Without this,
    we can't tell whether the deferral is buying us a real win or
    just moving the cost to a background slot."""
    src = _agent_main_src()
    assert "boot_mcp_refresh_ms" in src
    assert "boot_tunnel_start_ms" in src


def test_deferral_pattern_preserves_error_swallowing():
    """The original blocking code wrapped failures in try/except. The
    deferred wrapper must still catch + log so a failed task doesn't
    surface as an unhandled-exception warning."""
    src = _agent_main_src()
    # Find the _boot_refresh_mcp body and verify it contains
    # try/except + traceback printing (the original behavior).
    idx = src.find("async def _boot_refresh_mcp")
    assert idx > 0
    body = src[idx:idx + 2000]
    assert "try:" in body
    assert "except Exception" in body
    assert "traceback" in body

    idx2 = src.find("async def _boot_start_tunnel")
    assert idx2 > 0
    body2 = src[idx2:idx2 + 1200]
    assert "try:" in body2
    assert "except Exception" in body2
