"""
Model Failover Chain — Automatic provider failover with health tracking.

When a model provider fails (timeout, rate-limit, 5xx), automatically
retry with the next provider in the chain. Track provider health and
avoid unhealthy providers for a cooldown period.

Usage:
    from app.agent.model_failover import get_failover_chain

    chain = get_failover_chain()
    chain.configure([
        {"model": "claude-sonnet-4-20250514", "provider": "anthropic"},
        {"model": "gpt-4o", "provider": "openai"},
        {"model": "gpt-4o-mini", "provider": "openai"},
    ])

    model, provider = await chain.next_available()
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ProviderStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"
    COOLDOWN = "cooldown"


@dataclass
class ProviderHealth:
    """Health state for a single provider."""
    provider: str
    model: str
    status: ProviderStatus = ProviderStatus.HEALTHY
    consecutive_failures: int = 0
    total_requests: int = 0
    total_failures: int = 0
    last_failure_at: float = 0.0
    last_success_at: float = 0.0
    cooldown_until: float = 0.0
    avg_latency_ms: float = 0.0
    _latencies: List[float] = field(default_factory=list, repr=False)

    def record_success(self, latency_ms: float = 0.0):
        self.total_requests += 1
        self.consecutive_failures = 0
        self.last_success_at = time.time()
        self.status = ProviderStatus.HEALTHY
        self.cooldown_until = 0.0
        if latency_ms > 0:
            self._latencies.append(latency_ms)
            if len(self._latencies) > 50:
                self._latencies = self._latencies[-50:]
            self.avg_latency_ms = sum(self._latencies) / len(self._latencies)

    def record_failure(self, error: str = ""):
        self.total_requests += 1
        self.total_failures += 1
        self.consecutive_failures += 1
        self.last_failure_at = time.time()

        if self.consecutive_failures >= 5:
            self.status = ProviderStatus.DOWN
            self.cooldown_until = time.time() + 300  # 5 min cooldown
        elif self.consecutive_failures >= 2:
            self.status = ProviderStatus.DEGRADED
            self.cooldown_until = time.time() + 30  # 30s cooldown
        else:
            self.status = ProviderStatus.DEGRADED

        logger.warning(
            "[FAILOVER] %s/%s failure #%d: %s → %s (cooldown until %.0f)",
            self.provider, self.model, self.consecutive_failures,
            error[:100], self.status.value, self.cooldown_until,
        )

    def is_available(self) -> bool:
        if self.cooldown_until > 0 and time.time() < self.cooldown_until:
            return False
        return self.status != ProviderStatus.DOWN or time.time() >= self.cooldown_until

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "status": self.status.value,
            "consecutive_failures": self.consecutive_failures,
            "total_requests": self.total_requests,
            "total_failures": self.total_failures,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "available": self.is_available(),
        }


@dataclass
class FailoverEntry:
    """A single entry in the failover chain."""
    model: str
    provider: str
    priority: int = 0  # Lower = preferred
    max_retries: int = 2
    timeout_seconds: float = 120.0


class FailoverChain:
    """
    Manages an ordered chain of model providers with health tracking.

    When a provider fails, the chain moves to the next available provider.
    Providers in cooldown are skipped. Health is tracked per-provider.
    """

    def __init__(self):
        self._chain: List[FailoverEntry] = []
        self._health: Dict[str, ProviderHealth] = {}
        self._lock = asyncio.Lock()

    def configure(self, entries: List[Dict[str, Any]]) -> None:
        """Configure the failover chain from a list of dicts."""
        self._chain = []
        for i, entry in enumerate(entries):
            fe = FailoverEntry(
                model=entry["model"],
                provider=entry.get("provider", "openai"),
                priority=entry.get("priority", i),
                max_retries=entry.get("max_retries", 2),
                timeout_seconds=entry.get("timeout_seconds", 120.0),
            )
            self._chain.append(fe)
            key = f"{fe.provider}/{fe.model}"
            if key not in self._health:
                self._health[key] = ProviderHealth(
                    provider=fe.provider, model=fe.model
                )
        self._chain.sort(key=lambda e: e.priority)
        logger.info("[FAILOVER] Configured chain: %s", [f"{e.provider}/{e.model}" for e in self._chain])

    async def next_available(self, skip: Optional[List[str]] = None) -> Optional[Tuple[str, str]]:
        """
        Get the next available (model, provider) tuple.

        Args:
            skip: List of "provider/model" keys to skip (already tried).

        Returns:
            (model, provider) or None if all providers are down.
        """
        skip = set(skip or [])
        for entry in self._chain:
            key = f"{entry.provider}/{entry.model}"
            if key in skip:
                continue
            health = self._health.get(key)
            if health and not health.is_available():
                continue
            return (entry.model, entry.provider)
        return None

    def record_success(self, model: str, provider: str, latency_ms: float = 0.0) -> None:
        """Record a successful request."""
        key = f"{provider}/{model}"
        health = self._health.get(key)
        if health:
            health.record_success(latency_ms)

    def record_failure(self, model: str, provider: str, error: str = "") -> None:
        """Record a failed request."""
        key = f"{provider}/{model}"
        health = self._health.get(key)
        if health:
            health.record_failure(error)

    def get_health(self, model: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get health status for all providers or a specific model."""
        results = []
        for key, health in self._health.items():
            if model and health.model != model:
                continue
            results.append(health.to_dict())
        return results

    def reset_health(self, provider: Optional[str] = None) -> int:
        """Reset health counters. Returns number of providers reset."""
        count = 0
        for key, health in self._health.items():
            if provider and health.provider != provider:
                continue
            health.consecutive_failures = 0
            health.status = ProviderStatus.HEALTHY
            health.cooldown_until = 0.0
            count += 1
        return count

    @property
    def chain_length(self) -> int:
        return len(self._chain)

    def get_chain(self) -> List[Dict[str, Any]]:
        """Return the current failover chain configuration."""
        return [
            {
                "model": e.model,
                "provider": e.provider,
                "priority": e.priority,
                "max_retries": e.max_retries,
                "timeout_seconds": e.timeout_seconds,
                "health": self._health.get(f"{e.provider}/{e.model}", ProviderHealth(e.provider, e.model)).to_dict(),
            }
            for e in self._chain
        ]


# ── Singleton ────────────────────────────────────────────
_failover_chain: Optional[FailoverChain] = None


def get_failover_chain() -> FailoverChain:
    """Get or create the global failover chain."""
    global _failover_chain
    if _failover_chain is None:
        _failover_chain = FailoverChain()
        # Default chain
        _failover_chain.configure([
            {"model": "claude-sonnet-4-20250514", "provider": "anthropic", "priority": 0},
            {"model": "gpt-4o", "provider": "openai", "priority": 1},
            {"model": "gpt-4o-mini", "provider": "openai", "priority": 2},
        ])
    return _failover_chain
