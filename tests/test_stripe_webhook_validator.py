"""Unit tests for the Stripe webhook URL drift detector.

We don't hit Stripe in these tests — the validator's purpose is to
catch a misconfigured dashboard, so we mock the Stripe SDK to return
known-shaped endpoints and verify the drift detection itself.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.services.stripe_webhook_validator import (
    WebhookCheck,
    check_webhooks,
    format_warnings,
)


class _FakeApp:
    """FastAPI app stand-in — we only need `routes` with .path/.methods."""

    def __init__(self, post_paths: list[str]):
        self.routes = [
            SimpleNamespace(path=p, methods={"POST"}) for p in post_paths
        ]


def _stripe_list(endpoints: list[dict]):
    """Return an object that mimics what stripe.WebhookEndpoint.list returns."""
    return SimpleNamespace(get=lambda key, default=None: endpoints if key == "data" else default)


def test_returns_empty_when_no_secret_key():
    app = _FakeApp(["/api/vps/webhook/stripe"])
    assert check_webhooks(app, secret_key=None) == []
    assert check_webhooks(app, secret_key="") == []


def test_matching_route_is_not_flagged():
    app = _FakeApp(["/api/vps/webhook/stripe"])
    fake = SimpleNamespace(
        WebhookEndpoint=SimpleNamespace(list=lambda limit: _stripe_list([
            {"id": "we_ok", "url": "https://toup.ai/api/vps/webhook/stripe"},
        ])),
        api_key=None,
    )
    with patch.dict("sys.modules", {"stripe": fake}):
        checks = check_webhooks(app, secret_key="sk_test_xxx", expected_host="toup.ai")
    assert len(checks) == 1
    assert checks[0].matches_route is True
    assert format_warnings(checks) == []


def test_drifting_route_is_flagged():
    """The actual May 2026 incident: dashboard pointed at billing path,
    code only served vps path. Should be flagged."""
    app = _FakeApp(["/api/vps/webhook/stripe"])
    fake = SimpleNamespace(
        WebhookEndpoint=SimpleNamespace(list=lambda limit: _stripe_list([
            {"id": "we_drift", "url": "https://toup.ai/api/billing/stripe/webhook"},
        ])),
        api_key=None,
    )
    with patch.dict("sys.modules", {"stripe": fake}):
        checks = check_webhooks(app, secret_key="sk_test_xxx", expected_host="toup.ai")
    assert len(checks) == 1
    assert checks[0].matches_route is False
    warnings = format_warnings(checks)
    assert len(warnings) == 1
    assert "we_drift" in warnings[0]
    assert "/api/billing/stripe/webhook" in warnings[0]


def test_other_host_is_skipped():
    """A staging webhook hitting staging.toup.ai shouldn't pollute prod boot logs."""
    app = _FakeApp(["/api/vps/webhook/stripe"])
    fake = SimpleNamespace(
        WebhookEndpoint=SimpleNamespace(list=lambda limit: _stripe_list([
            {"id": "we_staging", "url": "https://staging.toup.ai/api/foo"},
        ])),
        api_key=None,
    )
    with patch.dict("sys.modules", {"stripe": fake}):
        checks = check_webhooks(app, secret_key="sk_test_xxx", expected_host="toup.ai")
    assert checks == []  # filtered out by host


def test_stripe_outage_does_not_raise():
    """A network blip on Stripe must NOT block app startup."""
    def _boom(limit):
        raise RuntimeError("connection refused")
    app = _FakeApp(["/api/vps/webhook/stripe"])
    fake = SimpleNamespace(
        WebhookEndpoint=SimpleNamespace(list=_boom),
        api_key=None,
    )
    with patch.dict("sys.modules", {"stripe": fake}):
        checks = check_webhooks(app, secret_key="sk_test_xxx")
    assert checks == []


def test_format_warnings_only_includes_misses():
    checks = [
        WebhookCheck(webhook_id="ok1", url="https://x/y", path="/y", matches_route=True),
        WebhookCheck(webhook_id="bad1", url="https://x/z", path="/z", matches_route=False),
    ]
    out = format_warnings(checks)
    assert len(out) == 1
    assert "bad1" in out[0]
