"""
Feedback API — Dashboard endpoints for retrieval quality metrics (Phase 5).

Endpoints:
  GET  /api/feedback/quality         — Get quality metrics for a time window
  GET  /api/feedback/suggestions     — Get LLM-powered improvement suggestions
  GET  /api/feedback/report          — Get full weekly quality report
  GET  /api/feedback/events          — List recent retrieval events
"""

import json
import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.api.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/feedback", tags=["Retrieval Feedback"])


class QualityMetrics(BaseModel):
    total_events: int = 0
    recall_estimate: float = 0.0
    precision_estimate: float = 0.0
    gap_rate: float = 0.0
    correction_rate: float = 0.0
    quality_distribution: dict = {}
    redundancy_estimate: float = 0.0
    avg_retrieval_time_ms: int = 0
    time_window_days: int = 7


class Suggestion(BaseModel):
    type: str
    topic: Optional[str] = None
    description: Optional[str] = None
    message: Optional[str] = None
    priority: Optional[str] = None


class QualityReport(BaseModel):
    report_date: str
    user_id: str
    metrics: QualityMetrics
    suggestions: list = []
    health_grade: str = "N/A"


class RetrievalEventSummary(BaseModel):
    id: str
    query: str
    result_count: int
    quality_signal: Optional[str]
    is_correction: bool
    has_extraction_gap: bool
    retrieval_time_ms: Optional[int]
    created_at: str


@router.get("/quality", response_model=QualityMetrics)
async def get_quality_metrics(
    days: int = Query(default=7, ge=1, le=90, description="Time window in days"),
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user),
):
    """Get retrieval quality metrics for a time window."""
    from app.services.retrieval_feedback import get_retrieval_feedback
    feedback = get_retrieval_feedback(db)
    metrics = await feedback.analyze_extraction_quality(current_user.id, time_window_days=days)
    return QualityMetrics(**metrics)


@router.get("/suggestions", response_model=list)
async def get_suggestions(
    days: int = Query(default=7, ge=1, le=90, description="Time window in days"),
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user),
):
    """Get LLM-powered improvement suggestions based on retrieval failures."""
    from app.services.retrieval_feedback import get_retrieval_feedback
    feedback = get_retrieval_feedback(db)
    suggestions = await feedback.suggest_extraction_improvements(current_user.id, time_window_days=days)
    return suggestions


@router.get("/report", response_model=QualityReport)
async def get_quality_report(
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user),
):
    """Get a comprehensive weekly quality report."""
    from app.services.retrieval_feedback import get_retrieval_feedback
    feedback = get_retrieval_feedback(db)
    report = await feedback.generate_weekly_report(current_user.id)
    return QualityReport(**report)


@router.get("/events", response_model=list)
async def list_retrieval_events(
    limit: int = Query(default=20, ge=1, le=100),
    quality: Optional[str] = Query(default=None, description="Filter by quality: good, partial, miss, empty"),
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user),
):
    """List recent retrieval events with optional quality filter."""
    from app.db.models import RetrievalEvent

    conditions = [RetrievalEvent.user_id == current_user.id]
    if quality:
        conditions.append(RetrievalEvent.quality_signal == quality)

    result = await db.execute(
        select(RetrievalEvent)
        .where(and_(*conditions))
        .order_by(RetrievalEvent.created_at.desc())
        .limit(limit)
    )
    events = result.scalars().all()

    return [
        RetrievalEventSummary(
            id=ev.id,
            query=ev.query[:200],
            result_count=ev.result_count,
            quality_signal=ev.quality_signal,
            is_correction=ev.is_correction,
            has_extraction_gap=ev.has_extraction_gap,
            retrieval_time_ms=ev.retrieval_time_ms,
            created_at=ev.created_at.isoformat() if ev.created_at else "",
        )
        for ev in events
    ]
