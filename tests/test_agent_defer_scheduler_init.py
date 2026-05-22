"""Regression test for TKT-LAT-017 (wave 3) — defer cron/routine schedulers.

Wave 1 deferred MCP cache refresh + tunnel client start. Wave 3 adds
cron_service.start() and routine_runner.start() to the deferred set.

Both schedulers fire on a time basis (cron daily/hourly, routines at
user tz-local wake) — a 200–800 ms startup delay never misses a tick.

trigger_runner.start() is NOT deferred — it registers webhook
handlers + a restart sweep, and an inbound Gmail push arriving during
start() would land in undefined state.
"""

from __future__ import annotations

from pathlib import Path


def _config_src() -> str:
    return Path(
        Path(__file__).resolve().parents[1] / "app" / "config.py"
    ).read_text()


def _agent_main_src() -> str:
    return Path(
        Path(__file__).resolve().parents[1] / "agent_main.py"
    ).read_text()


def test_scheduler_defer_flag_defaults_to_on():
    """Default ON for the same reason as agent_defer_boot_init."""
    assert "agent_defer_scheduler_init: bool = True" in _config_src()


def test_cron_start_deferred_behind_flag():
    """cron_service.start() runs in an asyncio.create_task when the
    flag is on, with a named task tag for observability."""
    src = _agent_main_src()
    assert "_boot_start_cron" in src
    assert "lat017-cron-start" in src
    assert "boot_deferred=cron_start" in src


def test_routine_start_deferred_behind_flag():
    """Same contract for routine_runner.start()."""
    src = _agent_main_src()
    assert "_boot_start_routine" in src
    assert "lat017-routine-start" in src
    assert "boot_deferred=routine_start" in src


def test_trigger_runner_NOT_deferred():
    """trigger_runner.start() must still block boot — inbound webhooks
    rely on the restart sweep + handler registration completing before
    the agent serves requests. Source must NOT contain a deferred
    wrapper for trigger_runner."""
    src = _agent_main_src()
    assert "lat017-trigger-start" not in src
    assert "boot_deferred=trigger_start" not in src
    # The original blocking await must still be present.
    assert "await trigger_runner.start()" in src


def test_blocking_fallback_preserved_when_flag_off():
    """If the operator disables agent_defer_scheduler_init, cron + routine
    must still run synchronously — same observable behavior as today."""
    src = _agent_main_src()
    # Cron blocking branch
    assert "else:\n            try:\n                await cron_service.start()" in src
    # Routine blocking branch
    assert "else:\n                await routine_runner.start()" in src


def test_perf_log_records_actual_wall_time():
    """When the deferred starts eventually complete, they must record
    their own wall-time so we can correlate `boot_deferred=cron_start`
    with the real cost when it finishes."""
    src = _agent_main_src()
    assert "boot_cron_start_ms" in src
    assert "boot_routine_start_ms" in src
