"""
T5a — Prometheus `/metrics` exposition for the connector subsystem.

Public endpoint (no auth) — Prometheus scrapers don't carry JWTs.
The metric data exposed is operational signal only (counters of
events by connector/tool/channel/outcome); no PII.

Mount path: `GET /api/connectors/metrics`. Per-environment IP
allowlist enforcement (Cloudflare / Railway origin lockdown) is
the right access-control layer; the endpoint itself doesn't try
to authenticate Prometheus.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from app.services.connector_metrics import render

router = APIRouter(prefix="/connectors", tags=["Metrics"])


@router.get("/metrics", response_class=PlainTextResponse)
async def connector_metrics() -> str:
    """Prometheus text-format exposition. Content-Type defaults to
    `text/plain` via PlainTextResponse — Prometheus accepts either
    `text/plain` or its versioned variant; the basic media type is
    always safe.
    """
    return render()
