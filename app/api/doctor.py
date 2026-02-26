"""
Doctor / Health Check API Router.
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, Query

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/doctor", tags=["Health Checks"])


@router.get("/")
async def doctor_report(checks: Optional[str] = Query(None)):
    """
    Run health checks and return a report.

    Optional: ?checks=database,docker,config to run specific checks only.
    """
    from app.agent.cli_doctor import run_doctor

    include = [c.strip() for c in checks.split(",")] if checks else None
    report = await run_doctor(include=include)
    return report.to_dict()


@router.get("/text")
async def doctor_text(checks: Optional[str] = Query(None)):
    """Get health report as human-readable text."""
    from app.agent.cli_doctor import run_doctor

    include = [c.strip() for c in checks.split(",")] if checks else None
    report = await run_doctor(include=include)
    return {"text": report.to_text()}
