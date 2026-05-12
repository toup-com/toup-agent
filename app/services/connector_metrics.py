"""
T5a — Connector observability primitives.

Lightweight in-process counters + a `/metrics` HTTP exposition (Prom
text format). No third-party dep; we already control the `/api/...`
surface and Prometheus scrapes a plain-text endpoint without any
client library on our side.

Counters tracked (per `01-architecture.md` §8):

  - `connector_tool_calls_total{connector,tool,channel,outcome}`
  - `connector_oauth_starts_total{connector}`
  - `connector_oauth_callbacks_total{connector,outcome}`
  - `connector_refresh_total{connector,outcome}`
  - `connector_health_probes_total{connector,outcome}`
  - `connector_dispatch_latency_ms_bucket{connector,tool,channel,le}`
    (histogram with default Prometheus buckets)

Tracked from:
  - `connector_dispatcher.execute()` → tool_calls + dispatch_latency
  - `oauth.connect/callback` → oauth_starts + oauth_callbacks
  - `connector_dispatcher._refresh_with_coalescing` → refresh_total
  - `health_probe_scheduler` (T5c) → health_probes_total

Memory: one `dict` per metric. No bound — labels are derived from
user input (connector, tool, channel) which are all small enumerated
sets. If somehow a runaway label appears, T5b's force-quarantine is
the operator's release valve.

Why not the real prometheus_client library: extra dep + multiproc
contortions on Railway/Vercel multi-worker setups. The text-format
exposition is 30 lines and bullet-proof.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Iterable

# Default Prometheus histogram buckets (seconds). For ms we multiply.
_BUCKETS_MS = [
    50, 100, 250, 500, 1000, 2500, 5000, 10_000, 30_000, 60_000,
]

_lock = threading.Lock()
_counters: dict[tuple[str, tuple[tuple[str, str], ...]], int] = defaultdict(int)
_histograms: dict[tuple[str, tuple[tuple[str, str], ...]], list[int]] = defaultdict(
    lambda: [0] * (len(_BUCKETS_MS) + 1)  # last slot is +Inf
)
_histogram_sum: dict[tuple[str, tuple[tuple[str, str], ...]], float] = defaultdict(
    float
)
_histogram_count: dict[tuple[str, tuple[tuple[str, str], ...]], int] = defaultdict(int)


def _key(name: str, labels: dict[str, str]) -> tuple:
    return (name, tuple(sorted(labels.items())))


def inc(name: str, *, labels: dict[str, str] | None = None, by: int = 1) -> None:
    with _lock:
        _counters[_key(name, labels or {})] += by


def observe_ms(name: str, value_ms: float, *, labels: dict[str, str] | None = None) -> None:
    """Record a timing into the histogram for `name` with given labels."""
    k = _key(name, labels or {})
    with _lock:
        _histogram_sum[k] += value_ms
        _histogram_count[k] += 1
        bucket_counts = _histograms[k]
        placed = False
        for i, b in enumerate(_BUCKETS_MS):
            if value_ms <= b:
                bucket_counts[i] += 1
                placed = True
                break
        if not placed:
            bucket_counts[-1] += 1


# ─── Exposition ──────────────────────────────────────────────────────


def _label_str(labels: tuple[tuple[str, str], ...]) -> str:
    if not labels:
        return ""
    parts = [f'{k}="{_escape(v)}"' for k, v in labels]
    return "{" + ",".join(parts) + "}"


def _escape(v: str) -> str:
    return (
        v.replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace('"', '\\"')
    )


def render() -> str:
    """Render Prometheus text format. Called by the /metrics endpoint."""
    lines: list[str] = []
    seen_metric_help: set[str] = set()
    with _lock:
        # Counters
        for (name, labels), value in sorted(_counters.items()):
            if name not in seen_metric_help:
                lines.append(f"# TYPE {name} counter")
                seen_metric_help.add(name)
            lines.append(f"{name}{_label_str(labels)} {value}")
        # Histograms
        for (name, labels), buckets in sorted(_histograms.items()):
            if name not in seen_metric_help:
                lines.append(f"# TYPE {name} histogram")
                seen_metric_help.add(name)
            cumulative = 0
            for i, b in enumerate(_BUCKETS_MS):
                cumulative += buckets[i]
                bucket_labels = labels + (("le", str(b)),)
                lines.append(
                    f"{name}_bucket{_label_str(bucket_labels)} {cumulative}"
                )
            cumulative += buckets[-1]
            bucket_labels = labels + (("le", "+Inf"),)
            lines.append(
                f"{name}_bucket{_label_str(bucket_labels)} {cumulative}"
            )
            lines.append(
                f"{name}_count{_label_str(labels)} {_histogram_count[(name, labels)]}"
            )
            lines.append(
                f"{name}_sum{_label_str(labels)} {_histogram_sum[(name, labels)]}"
            )
    lines.append("")
    return "\n".join(lines)


def reset_for_tests() -> None:
    with _lock:
        _counters.clear()
        _histograms.clear()
        _histogram_sum.clear()
        _histogram_count.clear()


# ─── Standard metric names (named constants for grep-ability) ────────


M_TOOL_CALLS = "connector_tool_calls_total"
M_OAUTH_STARTS = "connector_oauth_starts_total"
M_OAUTH_CALLBACKS = "connector_oauth_callbacks_total"
M_REFRESH = "connector_refresh_total"
M_HEALTH_PROBES = "connector_health_probes_total"
M_DISPATCH_LATENCY = "connector_dispatch_latency_ms"
# Incremented when the MCP filter hides all tools for a connector the
# user has an ACTIVE identity for. Production smell — same shape as
# the 2026-05-12 "Gmail tool isn't exposed" incident. Operator alert
# on any non-zero rate (this should be 0 in steady state).
M_MCP_FILTER_EMPTY = "connector_mcp_filter_empty_total"
