"""
Health Probe System — Component-level health checks with probes.

Provides a central registry for health checks across all subsystems
(database, telegram, memory, channels, etc.). Each component registers
a probe function that returns a HealthResult.

Usage:
    from app.agent.health import get_health_registry

    registry = get_health_registry()

    @registry.probe("database")
    async def check_db():
        # ... check DB connection
        return True, {"latency_ms": 5}

    report = await registry.run_all()
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthResult:
    """Result of a single health check."""
    component: str
    status: HealthStatus
    latency_ms: float = 0.0
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    checked_at: float = 0.0

    def __post_init__(self):
        if self.checked_at == 0.0:
            self.checked_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component": self.component,
            "status": self.status.value,
            "latency_ms": round(self.latency_ms, 1),
            "message": self.message,
            "details": self.details,
        }


@dataclass
class HealthReport:
    """Aggregate health report across all components."""
    results: List[HealthResult] = field(default_factory=list)
    checked_at: float = 0.0
    total_latency_ms: float = 0.0

    def __post_init__(self):
        if self.checked_at == 0.0:
            self.checked_at = time.time()

    @property
    def overall_status(self) -> HealthStatus:
        if not self.results:
            return HealthStatus.UNKNOWN
        statuses = [r.status for r in self.results]
        if all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY
        if any(s == HealthStatus.UNHEALTHY for s in statuses):
            return HealthStatus.UNHEALTHY
        if any(s == HealthStatus.DEGRADED for s in statuses):
            return HealthStatus.DEGRADED
        return HealthStatus.UNKNOWN

    @property
    def healthy_count(self) -> int:
        return sum(1 for r in self.results if r.status == HealthStatus.HEALTHY)

    @property
    def total_count(self) -> int:
        return len(self.results)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.overall_status.value,
            "healthy": self.healthy_count,
            "total": self.total_count,
            "total_latency_ms": round(self.total_latency_ms, 1),
            "components": {r.component: r.to_dict() for r in self.results},
        }


# Probe function type: returns (ok, details_dict) or just bool
ProbeFunction = Callable[[], Coroutine[Any, Any, Union[bool, Tuple[bool, Dict[str, Any]]]]]


class HealthRegistry:
    """
    Central registry for component health probes.

    Components register async probe functions that return health status.
    The registry can run all probes and produce an aggregate report.
    """

    def __init__(self, timeout: float = 10.0):
        self._probes: Dict[str, ProbeFunction] = {}
        self._timeout = timeout
        self._last_report: Optional[HealthReport] = None
        self._cache_ttl: float = 5.0  # seconds

    def register(self, component: str, probe_fn: ProbeFunction) -> None:
        """Register a health probe for a component."""
        self._probes[component] = probe_fn
        logger.debug(f"[HEALTH] Registered probe: {component}")

    def probe(self, component: str):
        """Decorator to register a health probe."""
        def decorator(fn: ProbeFunction):
            self.register(component, fn)
            return fn
        return decorator

    def unregister(self, component: str) -> bool:
        """Remove a health probe."""
        return self._probes.pop(component, None) is not None

    async def check(self, component: str) -> HealthResult:
        """Run a single component health check."""
        probe_fn = self._probes.get(component)
        if not probe_fn:
            return HealthResult(
                component=component,
                status=HealthStatus.UNKNOWN,
                message="No probe registered",
            )

        t0 = time.time()
        try:
            result = await asyncio.wait_for(probe_fn(), timeout=self._timeout)
            latency = (time.time() - t0) * 1000

            if isinstance(result, tuple):
                ok, details = result
            else:
                ok, details = result, {}

            return HealthResult(
                component=component,
                status=HealthStatus.HEALTHY if ok else HealthStatus.UNHEALTHY,
                latency_ms=latency,
                details=details,
            )
        except asyncio.TimeoutError:
            latency = (time.time() - t0) * 1000
            return HealthResult(
                component=component,
                status=HealthStatus.UNHEALTHY,
                latency_ms=latency,
                message=f"Probe timed out after {self._timeout}s",
            )
        except Exception as e:
            latency = (time.time() - t0) * 1000
            return HealthResult(
                component=component,
                status=HealthStatus.UNHEALTHY,
                latency_ms=latency,
                message=str(e),
            )

    async def run_all(self, use_cache: bool = False) -> HealthReport:
        """
        Run all registered health probes.

        Args:
            use_cache: If True, return cached report if within TTL.

        Returns:
            Aggregate HealthReport.
        """
        if use_cache and self._last_report:
            age = time.time() - self._last_report.checked_at
            if age < self._cache_ttl:
                return self._last_report

        t0 = time.time()
        results = []
        for component in sorted(self._probes.keys()):
            result = await self.check(component)
            results.append(result)

        report = HealthReport(
            results=results,
            total_latency_ms=(time.time() - t0) * 1000,
        )
        self._last_report = report
        return report

    def list_components(self) -> List[str]:
        """List all registered components."""
        return sorted(self._probes.keys())

    @property
    def probe_count(self) -> int:
        return len(self._probes)


# ── Singleton ────────────────────────────────────────────
_registry: Optional[HealthRegistry] = None


def get_health_registry() -> HealthRegistry:
    """Get the global health registry."""
    global _registry
    if _registry is None:
        _registry = HealthRegistry()
    return _registry
