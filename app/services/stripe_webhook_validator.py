"""Stripe webhook URL drift detection.

Detects the failure mode that broke us in May 2026: the Stripe dashboard
had a webhook configured at ``/api/billing/stripe/webhook`` while the
FastAPI app only served ``/api/vps/webhook/stripe``. Every event
returned 404 silently for weeks — no payment confirmations, no bundle
activations, no subscription cancellations actually reached the DB.

Strategy: at startup, fetch the live webhook list from Stripe and
compare each endpoint's path against the routes the FastAPI app
actually exposes. Anything that won't be served lands in the warning
output so an operator notices on the next deploy.

This is **detection only** — it never auto-edits Stripe state. If an
operator wants to repair drift they read the warning and use the
Stripe dashboard or the helper script in ``backend/scripts/``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WebhookCheck:
    """Per-endpoint result. Stable shape for tests + log formatting."""

    webhook_id: str
    url: str
    path: str
    matches_route: bool


def _extract_app_paths(app) -> set[str]:
    """Return the set of POST routes the FastAPI app exposes.

    We only consider POST because every Stripe webhook is a POST. A
    malformed config that points at a GET handler would still 405 —
    that's also drift, but the route's existence is what matters here.
    """
    paths: set[str] = set()
    for route in getattr(app, "routes", []):
        path = getattr(route, "path", None)
        if not path:
            continue
        methods = getattr(route, "methods", None) or set()
        if "POST" in methods:
            paths.add(path)
    return paths


def check_webhooks(
    app,
    *,
    secret_key: Optional[str],
    expected_host: Optional[str] = None,
) -> list[WebhookCheck]:
    """Compare Stripe-registered webhooks against this app's routes.

    Returns one ``WebhookCheck`` per registered Stripe webhook. Empty
    list means either Stripe is unreachable or no webhooks exist —
    both surface in the caller's warning logic.

    ``expected_host`` (e.g. ``"toup.ai"``) lets the caller restrict the
    check to webhooks pointing at this deployment. A staging webhook
    hitting ``staging.toup.ai`` shouldn't be flagged on the prod boot.
    """
    if not secret_key:
        return []

    try:
        import stripe
    except ImportError:
        logger.warning("stripe SDK unavailable — skipping webhook URL drift check")
        return []

    try:
        stripe.api_key = secret_key
        endpoints = stripe.WebhookEndpoint.list(limit=100)
    except Exception as exc:
        # Network error, expired key, etc. We never want a Stripe
        # outage to block app startup, so log + bail.
        logger.warning("Could not list Stripe webhooks: %s", exc)
        return []

    app_paths = _extract_app_paths(app)
    checks: list[WebhookCheck] = []
    for ep in endpoints.auto_paging_iter() if hasattr(endpoints, "auto_paging_iter") else endpoints.get("data", []):
        url = ep.get("url") if isinstance(ep, dict) else getattr(ep, "url", "")
        wid = ep.get("id") if isinstance(ep, dict) else getattr(ep, "id", "")
        if not url:
            continue
        parsed = urlparse(url)
        if expected_host and parsed.hostname and parsed.hostname != expected_host:
            # Different deployment — not our problem on this boot.
            continue
        checks.append(
            WebhookCheck(
                webhook_id=wid,
                url=url,
                path=parsed.path or "/",
                matches_route=parsed.path in app_paths,
            )
        )
    return checks


def format_warnings(checks: Iterable[WebhookCheck]) -> list[str]:
    """Human-readable warning lines for any drifting endpoints."""
    return [
        f"{c.webhook_id}: {c.url} → no matching route"
        for c in checks
        if not c.matches_route
    ]
