"""Regression test for TKT-LAT-003 — non-blocking agent-ready proxy.

The WS chat proxy used to retry agent lookup 6×5s before giving up,
producing a visible 30-second "Connecting…" spinner for any user
whose container was still in 'provisioning' or 'starting' state.

This ticket introduces:
  * `[PERF] ws_proxy_agent_wait_ms=…` observability (fires regardless
    of the flag — so we can see real-world distributions).
  * `agent_ws_proxy_fast_fail` flag (default OFF). When ON, returns
    WS close code 4503 "agent_starting" + a JSON
    `{type: agent_starting, retry_after_ms: 2000}` frame so the
    frontend can render "Waking…" and retry.
  * Default behavior preserved (6×5s retry) until the frontend ships
    the 4503 handler — flipping the flag without that work would just
    turn a 30-s spinner into a hard error.
"""

from __future__ import annotations

from pathlib import Path


def _ws_proxy_src() -> str:
    return Path(
        Path(__file__).resolve().parents[1] / "app" / "api" / "ws_chat_proxy.py"
    ).read_text()


def _config_src() -> str:
    return Path(
        Path(__file__).resolve().parents[1] / "app" / "config.py"
    ).read_text()


def test_flag_defaults_to_off():
    """Default OFF preserves original 30-second retry behavior until
    the frontend ships the 4503 handler."""
    assert "agent_ws_proxy_fast_fail: bool = False" in _config_src()


def test_perf_log_fires_unconditionally():
    """The wait-time PERF log must fire on every WS connect, not just
    when fast-fail is on. Without baseline observability we can't
    decide when to flip the flag."""
    src = _ws_proxy_src()
    assert "[PERF] ws_proxy_agent_wait_ms=" in src
    # And the log line must record both the attempts count and the
    # ready/not_ready outcome — those are the two scoring axes.
    assert "attempts=%d" in src
    assert "outcome=%s" in src


def test_blocking_path_preserved_behind_flag_off():
    """When the flag is OFF, the original 6-attempt loop must still
    execute. Source-level invariant: `for attempt in range(6)` and
    the 5-second sleep both still appear."""
    src = _ws_proxy_src()
    assert "for attempt in range(6)" in src
    assert "await asyncio.sleep(5)" in src


def test_fast_fail_path_returns_4503_with_retry_hint():
    """When the flag is ON, the proxy must close with 4503 + a
    machine-readable JSON frame so the frontend knows to retry."""
    src = _ws_proxy_src()
    assert "code=4503" in src
    assert "Agent starting" in src
    assert "agent_starting" in src
    # retry_after_ms gives the frontend a concrete delay rather than
    # leaving it to guess the right interval.
    assert "retry_after_ms" in src


def test_legacy_4404_preserved_when_flag_off():
    """The original 4404 "No agent" close must still fire when the
    flag is off so any client that depends on it doesn't regress."""
    src = _ws_proxy_src()
    assert "code=4404" in src
    assert "No active agent found. Deploy your agent first." in src


def test_no_blocking_sleep_in_fast_fail_path():
    """The fast-fail branch must NOT call asyncio.sleep — the whole
    point is to return immediately. Source-level invariant: the
    fast-fail branch is one lookup, no retry."""
    src = _ws_proxy_src()
    # Locate the fast-fail branch and verify it does exactly one
    # lookup before falling through to the result check.
    idx = src.find("if _ws_settings.agent_ws_proxy_fast_fail:")
    assert idx > 0
    # The branch body is between the if and the matching `else:`.
    branch = src[idx:src.find("else:", idx)]
    assert "_attempts_used = 1" in branch
    # And specifically: no asyncio.sleep inside the fast-fail body.
    assert "asyncio.sleep" not in branch
