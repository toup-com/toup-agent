"""
Model Per-Session Override, Idempotency Keys, Token/Usage Tracking.
Provides per-session model switching and usage analytics.
"""

import hashlib
import logging
import time
import uuid
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


AVAILABLE_MODELS = {
    "claude-opus-4-6": {"provider": "anthropic", "cost_in": 15.0, "cost_out": 75.0, "context": 200000},
    "claude-sonnet-4-20250514": {"provider": "anthropic", "cost_in": 3.0, "cost_out": 15.0, "context": 200000},
    "gpt-4o": {"provider": "openai", "cost_in": 2.5, "cost_out": 10.0, "context": 128000},
    "gpt-4o-mini": {"provider": "openai", "cost_in": 0.15, "cost_out": 0.60, "context": 128000},
    "gpt-4.1": {"provider": "openai", "cost_in": 2.0, "cost_out": 8.0, "context": 1000000},
    "gpt-4.1-mini": {"provider": "openai", "cost_in": 0.40, "cost_out": 1.60, "context": 1000000},
}


@dataclass
class SessionModelConfig:
    """Per-session model configuration."""
    session_id: str
    model: str = "claude-opus-4-6"
    thinking_level: str = "off"
    thinking_budget: int = 0
    verbose: bool = False
    temperature: float = 0.7
    max_tokens: int = 4096

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "model": self.model,
            "thinking_level": self.thinking_level,
            "thinking_budget": self.thinking_budget,
            "verbose": self.verbose,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }


@dataclass
class UsageRecord:
    """A single usage record."""
    record_id: str
    session_id: str
    model: str
    tokens_in: int
    tokens_out: int
    cost: float
    timestamp: float = field(default_factory=time.time)
    tool_calls: int = 0
    duration_ms: float = 0

    def to_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "session_id": self.session_id,
            "model": self.model,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cost": self.cost,
            "timestamp": self.timestamp,
            "tool_calls": self.tool_calls,
            "duration_ms": self.duration_ms,
        }


class ModelSessionManager:
    """Manage per-session model overrides and usage tracking."""

    def __init__(self):
        self._configs: Dict[str, SessionModelConfig] = {}
        self._usage: List[UsageRecord] = []
        self._idempotency_keys: Dict[str, dict] = {}
        self._idempotency_ttl: float = 3600

    def get_config(self, session_id: str) -> SessionModelConfig:
        """Get or create model config for a session."""
        if session_id not in self._configs:
            self._configs[session_id] = SessionModelConfig(session_id=session_id)
        return self._configs[session_id]

    def set_model(self, session_id: str, model: str) -> dict:
        """Set model for a session."""
        if model not in AVAILABLE_MODELS:
            return {"success": False, "error": f"Unknown model: {model}", "available": list(AVAILABLE_MODELS.keys())}
        config = self.get_config(session_id)
        old_model = config.model
        config.model = model
        return {"success": True, "old_model": old_model, "new_model": model}

    def set_thinking(self, session_id: str, level: str, budget: int = 0) -> dict:
        """Set thinking level for a session."""
        levels = {"off": 0, "low": 1024, "medium": 4096, "high": 10000, "xhigh": 32000}
        if level not in levels:
            return {"success": False, "error": f"Invalid level. Choose: {', '.join(levels.keys())}"}
        config = self.get_config(session_id)
        config.thinking_level = level
        config.thinking_budget = budget or levels[level]
        return {"success": True, "level": level, "budget": config.thinking_budget}

    def set_verbose(self, session_id: str, verbose: bool) -> dict:
        config = self.get_config(session_id)
        config.verbose = verbose
        return {"success": True, "verbose": verbose}

    def track_usage(
        self,
        session_id: str,
        model: str,
        tokens_in: int,
        tokens_out: int,
        tool_calls: int = 0,
        duration_ms: float = 0,
    ) -> UsageRecord:
        """Track token usage for a request."""
        model_info = AVAILABLE_MODELS.get(model, {"cost_in": 0, "cost_out": 0})
        cost = (tokens_in / 1_000_000 * model_info["cost_in"]) + (tokens_out / 1_000_000 * model_info["cost_out"])

        record = UsageRecord(
            record_id=str(uuid.uuid4())[:12],
            session_id=session_id,
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost=cost,
            tool_calls=tool_calls,
            duration_ms=duration_ms,
        )
        self._usage.append(record)
        return record

    def get_usage_summary(self, session_id: Optional[str] = None) -> dict:
        """Get usage summary, optionally filtered by session."""
        records = self._usage
        if session_id:
            records = [r for r in records if r.session_id == session_id]

        total_in = sum(r.tokens_in for r in records)
        total_out = sum(r.tokens_out for r in records)
        total_cost = sum(r.cost for r in records)
        total_tools = sum(r.tool_calls for r in records)

        by_model: Dict[str, dict] = {}
        for r in records:
            if r.model not in by_model:
                by_model[r.model] = {"tokens_in": 0, "tokens_out": 0, "cost": 0, "requests": 0}
            by_model[r.model]["tokens_in"] += r.tokens_in
            by_model[r.model]["tokens_out"] += r.tokens_out
            by_model[r.model]["cost"] += r.cost
            by_model[r.model]["requests"] += 1

        return {
            "total_tokens_in": total_in,
            "total_tokens_out": total_out,
            "total_cost": round(total_cost, 6),
            "total_requests": len(records),
            "total_tool_calls": total_tools,
            "by_model": by_model,
        }

    def check_idempotency(self, key: str) -> Optional[dict]:
        """Check if an idempotency key was already used. Returns cached result or None."""
        entry = self._idempotency_keys.get(key)
        if not entry:
            return None
        if time.time() - entry["timestamp"] > self._idempotency_ttl:
            del self._idempotency_keys[key]
            return None
        return entry.get("result")

    def set_idempotency(self, key: str, result: dict) -> None:
        """Cache a result for an idempotency key."""
        self._idempotency_keys[key] = {"result": result, "timestamp": time.time()}

    def generate_idempotency_key(self, session_id: str, message: str) -> str:
        """Generate an idempotency key from session + message content."""
        content = f"{session_id}:{message}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def cleanup_idempotency(self) -> int:
        """Remove expired idempotency keys."""
        now = time.time()
        expired = [k for k, v in self._idempotency_keys.items() if now - v["timestamp"] > self._idempotency_ttl]
        for k in expired:
            del self._idempotency_keys[k]
        return len(expired)

    def list_models(self) -> List[dict]:
        """List available models with pricing info."""
        return [
            {"model": name, **info}
            for name, info in AVAILABLE_MODELS.items()
        ]


_manager: Optional[ModelSessionManager] = None


def get_model_session_manager() -> ModelSessionManager:
    """Get the global model session manager singleton."""
    global _manager
    if _manager is None:
        _manager = ModelSessionManager()
    return _manager
