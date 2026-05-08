"""Aggregate prewarm telemetry from platform stdout logs.

Reads platform log lines from stdin (or a file passed as argv[1]) and
joins three markers on user prefix:

  [PREWARM-START] user=<8hex> ts=<iso>
  [BOOT-READY]    user=<8hex> ts=<iso> phase=<...>
  [WAKE-CLICK]    user=<8hex> ts=<iso>

Emits p10 / p50 / p90 for:
  prewarm_to_boot_s   — how long the prewarm took to reach a usable agent
  boot_to_wake_s      — slack the prewarm bought us at Wake-button click
                        (positive = boot finished before click; the goal)
  prewarm_to_wake_s   — total user-visible elapsed time

Usage:
  railway logs --service platform-api --json | python -m scripts.prewarm_aggregator
  python scripts/prewarm_aggregator.py path/to/saved.log

The Railway "logs" stream may include a JSON wrapper per line; we
detect both the wrapped form ({"message": "..."}) and the plain form.

Designed to run from a daily cron — see docs/onboarding/prewarm-coverage-telemetry.md
for the rollout plan.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable


_PATTERN = re.compile(
    r"\[(PREWARM-START|BOOT-READY|BOOT-READY-TIMEOUT|WAKE-CLICK)\]"
    r"\s+user=(?P<user>[0-9a-f]{1,8})"
    r"\s+ts=(?P<ts>[0-9T:.\-+Z]+)"
)


@dataclass
class UserSession:
    user: str
    prewarm_start: datetime | None = None
    boot_ready: datetime | None = None
    boot_timeout: datetime | None = None
    wake_click: datetime | None = None
    extra: dict = field(default_factory=dict)


def _extract_message(line: str) -> str:
    """Strip Railway's JSON envelope if present, else return line as-is."""
    line = line.rstrip("\n")
    if not line:
        return ""
    if line.startswith("{") and line.endswith("}"):
        try:
            obj = json.loads(line)
        except ValueError:
            return line
        for key in ("message", "msg", "text", "log"):
            if key in obj and isinstance(obj[key], str):
                return obj[key]
    return line


def parse_lines(lines: Iterable[str]) -> dict[str, UserSession]:
    sessions: dict[str, UserSession] = defaultdict(lambda: UserSession(user=""))
    for raw in lines:
        msg = _extract_message(raw)
        m = _PATTERN.search(msg)
        if not m:
            continue
        marker = m.group(1)
        user = m.group("user")
        ts = _parse_ts(m.group("ts"))
        if ts is None:
            continue
        s = sessions[user]
        s.user = user
        if marker == "PREWARM-START":
            # First wins: a user reprovisioning later (e.g., bundle webhook
            # forced recreate) creates a second PREWARM-START; we want the
            # original onboarding pair.
            if s.prewarm_start is None or ts < s.prewarm_start:
                s.prewarm_start = ts
        elif marker == "BOOT-READY":
            if s.boot_ready is None or ts < s.boot_ready:
                s.boot_ready = ts
        elif marker == "BOOT-READY-TIMEOUT":
            s.boot_timeout = ts
        elif marker == "WAKE-CLICK":
            if s.wake_click is None or ts < s.wake_click:
                s.wake_click = ts
    return sessions


def _parse_ts(s: str) -> datetime | None:
    s = s.strip()
    # Python's fromisoformat handles "+00:00" but pre-3.11 chokes on "Z".
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    k = (len(values) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def _fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:+.1f}s"


def aggregate(sessions: dict[str, UserSession]) -> dict:
    prewarm_to_boot: list[float] = []
    boot_to_wake: list[float] = []
    prewarm_to_wake: list[float] = []
    timeout_users: list[str] = []
    incomplete: list[str] = []

    for s in sessions.values():
        if s.prewarm_start and s.boot_ready:
            prewarm_to_boot.append((s.boot_ready - s.prewarm_start).total_seconds())
        if s.boot_ready and s.wake_click:
            boot_to_wake.append((s.wake_click - s.boot_ready).total_seconds())
        if s.prewarm_start and s.wake_click:
            prewarm_to_wake.append((s.wake_click - s.prewarm_start).total_seconds())
        if s.boot_timeout and not s.boot_ready:
            timeout_users.append(s.user)
        if s.prewarm_start and not s.wake_click:
            incomplete.append(s.user)

    return {
        "users_seen": len(sessions),
        "users_with_prewarm": sum(1 for s in sessions.values() if s.prewarm_start),
        "users_reached_boot_ready": sum(1 for s in sessions.values() if s.boot_ready),
        "users_reached_wake_click": sum(1 for s in sessions.values() if s.wake_click),
        "users_boot_timeout": len(timeout_users),
        "users_incomplete_no_wake": len(incomplete),
        "prewarm_to_boot_p10": _percentile(prewarm_to_boot, 0.10),
        "prewarm_to_boot_p50": _percentile(prewarm_to_boot, 0.50),
        "prewarm_to_boot_p90": _percentile(prewarm_to_boot, 0.90),
        "boot_to_wake_p10": _percentile(boot_to_wake, 0.10),
        "boot_to_wake_p50": _percentile(boot_to_wake, 0.50),
        "boot_to_wake_p90": _percentile(boot_to_wake, 0.90),
        "prewarm_to_wake_p10": _percentile(prewarm_to_wake, 0.10),
        "prewarm_to_wake_p50": _percentile(prewarm_to_wake, 0.50),
        "prewarm_to_wake_p90": _percentile(prewarm_to_wake, 0.90),
        "timeout_users": timeout_users,
    }


def render_report(stats: dict) -> str:
    lines = [
        "Prewarm telemetry — daily aggregate",
        "─" * 50,
        f"Users seen:                 {stats['users_seen']}",
        f"Users with [PREWARM-START]: {stats['users_with_prewarm']}",
        f"Users reached [BOOT-READY]: {stats['users_reached_boot_ready']}",
        f"Users reached [WAKE-CLICK]: {stats['users_reached_wake_click']}",
        f"Users with boot timeout:    {stats['users_boot_timeout']}",
        f"Users prewarm w/o wake:     {stats['users_incomplete_no_wake']}",
        "",
        "Distributions (negative = wake fired before boot — bad)",
        f"  prewarm → boot   p10/p50/p90: "
        f"{_fmt(stats['prewarm_to_boot_p10'])} / "
        f"{_fmt(stats['prewarm_to_boot_p50'])} / "
        f"{_fmt(stats['prewarm_to_boot_p90'])}",
        f"  boot → wake      p10/p50/p90: "
        f"{_fmt(stats['boot_to_wake_p10'])} / "
        f"{_fmt(stats['boot_to_wake_p50'])} / "
        f"{_fmt(stats['boot_to_wake_p90'])}",
        f"  prewarm → wake   p10/p50/p90: "
        f"{_fmt(stats['prewarm_to_wake_p10'])} / "
        f"{_fmt(stats['prewarm_to_wake_p50'])} / "
        f"{_fmt(stats['prewarm_to_wake_p90'])}",
    ]
    if stats["timeout_users"]:
        lines.append("")
        lines.append(f"Boot timeout users: {', '.join(stats['timeout_users'])}")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    if len(argv) > 1:
        with open(argv[1], "r") as f:
            sessions = parse_lines(f)
    else:
        sessions = parse_lines(sys.stdin)
    stats = aggregate(sessions)
    print(render_report(stats))
    if argv[1:2] == ["--json"]:
        # Drop the timeout_users list from JSON to keep it small;
        # the human report already names them.
        compact = {k: v for k, v in stats.items() if k != "timeout_users"}
        print(json.dumps(compact))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
