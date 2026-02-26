"""
Token Tracker — User-facing token usage and cost tracking.

Tracks input/output tokens, costs per model, per session, and
per user. Provides /usage and /status data for chat commands.

Usage:
    from app.agent.token_tracker import get_token_tracker

    tracker = get_token_tracker()
    tracker.record_usage(
        session_id="s1",
        model="gpt-4o",
        input_tokens=1500,
        output_tokens=500,
    )

    usage = tracker.get_session_usage("s1")
    total = tracker.get_total_usage()
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Cost per 1M tokens (USD)
MODEL_PRICING: Dict[str, Dict[str, float]] = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "o1": {"input": 15.00, "output": 60.00},
    "o1-mini": {"input": 3.00, "output": 12.00},
    "o3-mini": {"input": 1.10, "output": 4.40},
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
}


@dataclass
class UsageRecord:
    """A single usage record."""
    session_id: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float = 0.0
    timestamp: float = 0.0
    tool_calls: int = 0
    user_id: str = ""

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
        if self.cost_usd == 0.0:
            self.cost_usd = self._calculate_cost()

    def _calculate_cost(self) -> float:
        pricing = MODEL_PRICING.get(self.model, {"input": 0.0, "output": 0.0})
        input_cost = self.input_tokens / 1_000_000 * pricing["input"]
        output_cost = self.output_tokens / 1_000_000 * pricing["output"]
        return round(input_cost + output_cost, 6)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "tool_calls": self.tool_calls,
        }


@dataclass
class SessionUsage:
    """Aggregated usage for a session."""
    session_id: str
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    total_requests: int = 0
    total_tool_calls: int = 0
    models_used: Dict[str, int] = field(default_factory=dict)
    started_at: float = 0.0
    last_activity: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 4),
            "total_requests": self.total_requests,
            "total_tool_calls": self.total_tool_calls,
            "models_used": self.models_used,
            "duration_seconds": round(self.last_activity - self.started_at, 1) if self.started_at else 0,
        }

    def format_status(self) -> str:
        """Format as a human-readable status string."""
        lines = [
            f"📊 **Session Usage**",
            f"  Tokens: {self.total_tokens:,} ({self.total_input_tokens:,} in / {self.total_output_tokens:,} out)",
            f"  Cost: ${self.total_cost_usd:.4f}",
            f"  Requests: {self.total_requests}",
            f"  Tool calls: {self.total_tool_calls}",
            f"  Models: {', '.join(self.models_used.keys())}",
        ]
        return "\n".join(lines)


class TokenTracker:
    """
    Tracks token usage and costs across sessions.

    Records every API call's token usage and computes costs
    based on model pricing. Provides aggregation by session,
    user, model, and time period.
    """

    def __init__(self):
        self._records: List[UsageRecord] = []
        self._sessions: Dict[str, SessionUsage] = {}
        self._max_records: int = 10000

    def record_usage(
        self,
        session_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        *,
        tool_calls: int = 0,
        user_id: str = "",
    ) -> UsageRecord:
        """Record a usage event."""
        record = UsageRecord(
            session_id=session_id,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_calls=tool_calls,
            user_id=user_id,
        )

        self._records.append(record)
        if len(self._records) > self._max_records:
            self._records = self._records[-self._max_records:]

        # Update session aggregation
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionUsage(
                session_id=session_id,
                started_at=record.timestamp,
            )

        session = self._sessions[session_id]
        session.total_input_tokens += input_tokens
        session.total_output_tokens += output_tokens
        session.total_cost_usd += record.cost_usd
        session.total_requests += 1
        session.total_tool_calls += tool_calls
        session.last_activity = record.timestamp
        session.models_used[model] = session.models_used.get(model, 0) + 1

        return record

    def get_session_usage(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get aggregated usage for a session."""
        session = self._sessions.get(session_id)
        if not session:
            return None
        return session.to_dict()

    def get_session_status(self, session_id: str) -> str:
        """Get formatted status string for /status command."""
        session = self._sessions.get(session_id)
        if not session:
            return "📊 No usage data for this session."
        return session.format_status()

    def get_total_usage(self) -> Dict[str, Any]:
        """Get total usage across all sessions."""
        total_input = sum(s.total_input_tokens for s in self._sessions.values())
        total_output = sum(s.total_output_tokens for s in self._sessions.values())
        total_cost = sum(s.total_cost_usd for s in self._sessions.values())
        total_requests = sum(s.total_requests for s in self._sessions.values())

        return {
            "total_sessions": len(self._sessions),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "total_cost_usd": round(total_cost, 4),
            "total_requests": total_requests,
        }

    def get_usage_by_model(self) -> Dict[str, Dict[str, Any]]:
        """Get usage breakdown by model."""
        by_model: Dict[str, Dict[str, Any]] = {}
        for r in self._records:
            if r.model not in by_model:
                by_model[r.model] = {
                    "input_tokens": 0, "output_tokens": 0,
                    "cost_usd": 0.0, "requests": 0,
                }
            by_model[r.model]["input_tokens"] += r.input_tokens
            by_model[r.model]["output_tokens"] += r.output_tokens
            by_model[r.model]["cost_usd"] = round(
                by_model[r.model]["cost_usd"] + r.cost_usd, 6
            )
            by_model[r.model]["requests"] += 1
        return by_model

    def get_recent_records(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent usage records."""
        return [r.to_dict() for r in self._records[-limit:]]

    def format_usage_report(self) -> str:
        """Format a usage report for /usage command."""
        total = self.get_total_usage()
        by_model = self.get_usage_by_model()

        lines = [
            "📊 **Usage Report**",
            f"  Sessions: {total['total_sessions']}",
            f"  Total tokens: {total['total_tokens']:,}",
            f"  Total cost: ${total['total_cost_usd']:.4f}",
            f"  Requests: {total['total_requests']}",
            "",
            "📈 **By Model:**",
        ]

        for model, data in sorted(by_model.items()):
            lines.append(
                f"  {model}: {data['requests']} reqs, "
                f"{data['input_tokens'] + data['output_tokens']:,} tokens, "
                f"${data['cost_usd']:.4f}"
            )

        return "\n".join(lines)

    def clear_session(self, session_id: str) -> bool:
        """Clear usage data for a session."""
        return self._sessions.pop(session_id, None) is not None

    def stats(self) -> Dict[str, Any]:
        """Get tracker statistics."""
        return {
            "total_records": len(self._records),
            "total_sessions": len(self._sessions),
            **self.get_total_usage(),
        }


# ── Singleton ────────────────────────────────────────────
_tracker: Optional[TokenTracker] = None


def get_token_tracker() -> TokenTracker:
    """Get the global token tracker."""
    global _tracker
    if _tracker is None:
        _tracker = TokenTracker()
    return _tracker
