"""
Retrieval Feedback Service — Self-Improvement Loop (Phase 5)

Tracks retrieval quality signals and feeds them back to improve extraction.

Signals collected:
1. Which memories were retrieved but NOT used in the response (low relevance)
2. Which memories were retrieved AND referenced (high relevance)
3. When the user corrects the agent ("No, I said X not Y")
4. When memory search returns 0 results (extraction gap)

Provides:
- log_retrieval_feedback(): Record every retrieval event
- analyze_extraction_quality(): Weekly quality metrics
- suggest_extraction_improvements(): LLM-powered improvement suggestions
- detect_correction(): Identify user corrections in messages
"""

import json
import logging
import re
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import and_, func, select, case, text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# Correction detection patterns
CORRECTION_PATTERNS = [
    r"(?:no|nope),?\s+(?:i\s+(?:said|meant|told you)|it'?s?\s+(?:actually|not))",
    r"that'?s?\s+(?:not\s+(?:right|correct|true|what)|wrong)",
    r"(?:actually|correction),?\s+(?:i|it|my|the)",
    r"(?:i\s+)?(?:didn'?t\s+(?:say|mean)|never\s+said)",
    r"you(?:'re|\s+are)\s+(?:wrong|mistaken|confused)",
    r"(?:let\s+me\s+)?correct\s+(?:that|you|myself)",
    r"(?:i\s+)?meant\s+to\s+say",
    r"(?:to\s+be\s+)?(?:clear|precise|exact),?\s+(?:i|it|my|the)",
]

# Compile patterns once
_CORRECTION_RE = [re.compile(p, re.IGNORECASE) for p in CORRECTION_PATTERNS]


def detect_correction(user_message: str) -> bool:
    """
    Detect if the user is correcting the agent.
    Returns True if correction patterns are found.
    """
    for pattern in _CORRECTION_RE:
        if pattern.search(user_message):
            return True
    return False


class RetrievalFeedback:
    """
    Tracks retrieval quality signals and feeds them back to improve extraction.
    """

    def __init__(self, db: AsyncSession):
        self.db = db

    # ------------------------------------------------------------------
    # 1. Log retrieval feedback
    # ------------------------------------------------------------------
    async def log_retrieval_feedback(
        self,
        user_id: str,
        query: str,
        retrieved_memories: List[Dict[str, Any]],
        response: str,
        conversation_id: Optional[str] = None,
        strategies_used: Optional[List[str]] = None,
        retrieval_time_ms: Optional[int] = None,
    ) -> str:
        """
        Log a retrieval event for quality analysis.

        Performs lightweight content matching to determine which retrieved
        memories were actually referenced in the response.

        Returns the retrieval event ID.
        """
        from app.db.models import RetrievalEvent

        retrieved_ids = [m.get("id", "") for m in retrieved_memories]
        used_ids = []
        irrelevant_ids = []

        # Lightweight content matching: check if memory content appears
        # (even partially) in the response
        response_lower = response.lower()
        for mem in retrieved_memories:
            content = mem.get("content", "")
            if not content:
                continue
            # Check if significant words from the memory appear in the response
            words = [w for w in content.lower().split() if len(w) > 3]
            if not words:
                continue
            match_count = sum(1 for w in words if w in response_lower)
            match_ratio = match_count / len(words) if words else 0

            if match_ratio >= 0.3:  # At least 30% of significant words match
                used_ids.append(mem.get("id", ""))
            else:
                irrelevant_ids.append(mem.get("id", ""))

        # Determine quality signal
        if not retrieved_ids:
            quality_signal = "empty"  # Nothing retrieved
        elif len(used_ids) >= len(retrieved_ids) * 0.5:
            quality_signal = "good"  # Most retrieved memories were useful
        elif used_ids:
            quality_signal = "partial"  # Some were useful
        else:
            quality_signal = "miss"  # None were useful

        # Detect extraction gap: response uses info not in any retrieved memory
        has_gap = quality_signal in ("empty", "miss")

        # Detect if user is correcting the agent
        is_correction = detect_correction(query)

        event_id = str(uuid.uuid4())
        event = RetrievalEvent(
            id=event_id,
            user_id=user_id,
            conversation_id=conversation_id,
            query=query[:2000],
            strategies_used_json=json.dumps(strategies_used) if strategies_used else None,
            retrieved_memory_ids_json=json.dumps(retrieved_ids),
            used_memory_ids_json=json.dumps(used_ids),
            irrelevant_memory_ids_json=json.dumps(irrelevant_ids),
            result_count=len(retrieved_ids),
            has_extraction_gap=has_gap,
            is_correction=is_correction,
            quality_signal=quality_signal,
            retrieval_time_ms=retrieval_time_ms,
        )
        self.db.add(event)
        # Don't commit here — let the caller handle the transaction
        await self.db.flush()

        logger.info(
            f"[FEEDBACK] Logged retrieval event {event_id}: "
            f"retrieved={len(retrieved_ids)}, used={len(used_ids)}, "
            f"irrelevant={len(irrelevant_ids)}, quality={quality_signal}, "
            f"correction={is_correction}"
        )
        return event_id

    # ------------------------------------------------------------------
    # 2. Analyze extraction quality
    # ------------------------------------------------------------------
    async def analyze_extraction_quality(
        self,
        user_id: str,
        time_window_days: int = 7,
    ) -> Dict[str, Any]:
        """
        Analyze extraction quality over a time window.

        Returns:
        - total_events: Number of retrieval events in the window
        - recall_estimate: How often relevant info is found (% non-empty)
        - precision_estimate: How often retrieved info is actually relevant (% used/retrieved)
        - gap_rate: How often extraction gaps are detected
        - correction_rate: How often user corrects the agent
        - quality_distribution: {"good": N, "partial": N, "miss": N, "empty": N}
        - redundancy_estimate: avg irrelevant memories per retrieval
        - avg_retrieval_time_ms: average retrieval latency
        """
        from app.db.models import RetrievalEvent

        cutoff = datetime.utcnow() - timedelta(days=time_window_days)

        # Get all events in the window
        result = await self.db.execute(
            select(RetrievalEvent).where(
                and_(
                    RetrievalEvent.user_id == user_id,
                    RetrievalEvent.created_at >= cutoff,
                )
            ).order_by(RetrievalEvent.created_at.desc())
        )
        events = result.scalars().all()

        if not events:
            return {
                "total_events": 0,
                "recall_estimate": 0.0,
                "precision_estimate": 0.0,
                "gap_rate": 0.0,
                "correction_rate": 0.0,
                "quality_distribution": {"good": 0, "partial": 0, "miss": 0, "empty": 0},
                "redundancy_estimate": 0.0,
                "avg_retrieval_time_ms": 0,
                "time_window_days": time_window_days,
            }

        total = len(events)
        quality_dist = {"good": 0, "partial": 0, "miss": 0, "empty": 0}
        total_retrieved = 0
        total_used = 0
        total_irrelevant = 0
        gap_count = 0
        correction_count = 0
        total_retrieval_time = 0
        time_count = 0

        for ev in events:
            signal = ev.quality_signal or "empty"
            quality_dist[signal] = quality_dist.get(signal, 0) + 1

            retrieved = json.loads(ev.retrieved_memory_ids_json) if ev.retrieved_memory_ids_json else []
            used = json.loads(ev.used_memory_ids_json) if ev.used_memory_ids_json else []
            irrelevant = json.loads(ev.irrelevant_memory_ids_json) if ev.irrelevant_memory_ids_json else []

            total_retrieved += len(retrieved)
            total_used += len(used)
            total_irrelevant += len(irrelevant)

            if ev.has_extraction_gap:
                gap_count += 1
            if ev.is_correction:
                correction_count += 1
            if ev.retrieval_time_ms:
                total_retrieval_time += ev.retrieval_time_ms
                time_count += 1

        recall_estimate = (total - quality_dist.get("empty", 0)) / total if total else 0
        precision_estimate = total_used / total_retrieved if total_retrieved else 0

        return {
            "total_events": total,
            "recall_estimate": round(recall_estimate, 3),
            "precision_estimate": round(precision_estimate, 3),
            "gap_rate": round(gap_count / total, 3) if total else 0,
            "correction_rate": round(correction_count / total, 3) if total else 0,
            "quality_distribution": quality_dist,
            "redundancy_estimate": round(total_irrelevant / total, 2) if total else 0,
            "avg_retrieval_time_ms": int(total_retrieval_time / time_count) if time_count else 0,
            "time_window_days": time_window_days,
        }

    # ------------------------------------------------------------------
    # 3. Suggest extraction improvements (LLM-powered)
    # ------------------------------------------------------------------
    async def suggest_extraction_improvements(
        self,
        user_id: str,
        time_window_days: int = 7,
    ) -> List[Dict[str, Any]]:
        """
        Use LLM to analyze retrieval failures and suggest improvements.

        Looks at:
        - Queries that returned no results (extraction gaps)
        - Queries where no retrieved memories were used (misses)
        - User corrections
        - Patterns in failed retrievals

        Returns a list of improvement suggestions.
        """
        from app.db.models import RetrievalEvent

        cutoff = datetime.utcnow() - timedelta(days=time_window_days)

        # Get problematic events
        result = await self.db.execute(
            select(RetrievalEvent).where(
                and_(
                    RetrievalEvent.user_id == user_id,
                    RetrievalEvent.created_at >= cutoff,
                    # Only look at failures
                    RetrievalEvent.quality_signal.in_(["empty", "miss"]),
                )
            ).order_by(RetrievalEvent.created_at.desc()).limit(50)
        )
        failed_events = result.scalars().all()

        if not failed_events:
            return [{"type": "info", "message": "No retrieval failures detected in the last week."}]

        # Build analysis prompt
        failed_queries = [
            {
                "query": ev.query[:200],
                "signal": ev.quality_signal,
                "result_count": ev.result_count,
                "is_correction": ev.is_correction,
            }
            for ev in failed_events[:30]  # Limit to 30 for prompt size
        ]

        prompt = f"""You are analyzing retrieval quality for a personal AI assistant's memory system.

Below are queries from the last {time_window_days} days where the memory system FAILED to retrieve useful information:

{json.dumps(failed_queries, indent=2)}

Quality signals:
- "empty" = No memories were retrieved at all
- "miss" = Memories were retrieved but none were relevant to the query

Analyze these failures and provide 3-5 actionable improvement suggestions. Consider:
1. Are there topic patterns where extraction is consistently failing?
2. Are there categories of information the system isn't capturing?
3. Are users asking about things in ways the extraction doesn't anticipate?
4. Are there entity types or relationship types being missed?

Return ONLY valid JSON:
{{
  "suggestions": [
    {{
      "type": "extraction_gap",
      "topic": "the topic area that needs better coverage",
      "description": "What to improve and why",
      "priority": "high|medium|low"
    }}
  ]
}}"""

        try:
            from app.services.llm_service import get_llm_service
            llm = get_llm_service()
            response = await llm.complete_with_json(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1500,
            )
            result = json.loads(response.content)
            return result.get("suggestions", [])
        except Exception as e:
            logger.warning(f"LLM analysis failed: {e}")
            # Fall back to rule-based analysis
            return self._rule_based_suggestions(failed_queries)

    def _rule_based_suggestions(
        self, failed_queries: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Fallback: generate suggestions without LLM."""
        suggestions = []

        empty_count = sum(1 for q in failed_queries if q["signal"] == "empty")
        miss_count = sum(1 for q in failed_queries if q["signal"] == "miss")
        correction_count = sum(1 for q in failed_queries if q.get("is_correction"))

        if empty_count > len(failed_queries) * 0.5:
            suggestions.append({
                "type": "extraction_gap",
                "topic": "general",
                "description": f"{empty_count} queries returned no results. The extraction pipeline may be too selective or missing important conversation topics.",
                "priority": "high",
            })

        if miss_count > len(failed_queries) * 0.3:
            suggestions.append({
                "type": "relevance",
                "topic": "retrieval",
                "description": f"{miss_count} queries retrieved memories that weren't relevant. Consider improving embedding quality or adding re-ranking.",
                "priority": "medium",
            })

        if correction_count > 3:
            suggestions.append({
                "type": "accuracy",
                "topic": "corrections",
                "description": f"User corrected the agent {correction_count} times. Review extraction accuracy and memory update mechanisms.",
                "priority": "high",
            })

        if not suggestions:
            suggestions.append({
                "type": "info",
                "topic": "general",
                "description": "Some retrieval failures detected but no clear pattern. Monitor for trends.",
                "priority": "low",
            })

        return suggestions

    # ------------------------------------------------------------------
    # 4. Weekly quality report
    # ------------------------------------------------------------------
    async def generate_weekly_report(
        self,
        user_id: str,
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive weekly quality report combining
        metrics analysis and improvement suggestions.
        """
        metrics = await self.analyze_extraction_quality(user_id, time_window_days=7)
        suggestions = await self.suggest_extraction_improvements(user_id, time_window_days=7)

        report = {
            "report_date": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "metrics": metrics,
            "suggestions": suggestions,
            "health_grade": self._compute_health_grade(metrics),
        }

        logger.info(
            f"[FEEDBACK] Weekly report for {user_id}: "
            f"grade={report['health_grade']}, "
            f"events={metrics['total_events']}, "
            f"recall={metrics['recall_estimate']:.2%}, "
            f"precision={metrics['precision_estimate']:.2%}"
        )
        return report

    @staticmethod
    def _compute_health_grade(metrics: Dict[str, Any]) -> str:
        """Compute an A-F grade based on quality metrics."""
        if metrics["total_events"] < 5:
            return "N/A"  # Not enough data

        # Weighted score: recall 40%, precision 30%, low gap rate 20%, low correction rate 10%
        recall = metrics["recall_estimate"]
        precision = metrics["precision_estimate"]
        gap_score = 1.0 - metrics["gap_rate"]
        correction_score = 1.0 - metrics["correction_rate"]

        score = recall * 0.4 + precision * 0.3 + gap_score * 0.2 + correction_score * 0.1

        if score >= 0.9:
            return "A"
        elif score >= 0.8:
            return "B"
        elif score >= 0.65:
            return "C"
        elif score >= 0.5:
            return "D"
        else:
            return "F"


def get_retrieval_feedback(db: AsyncSession) -> RetrievalFeedback:
    """Factory function for RetrievalFeedback service."""
    return RetrievalFeedback(db)
