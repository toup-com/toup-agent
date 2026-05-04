"""Boot-time guard against Stripe-key / environment mismatch.

The guard lives in `Settings._validate_stripe_environment_match` and runs on
construction — meaning misconfiguration aborts at import, not at first
request. We test by constructing `Settings` directly with kwargs that
override env-derived defaults.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import Settings


def _build(**overrides) -> Settings:
    # Pass through pydantic-settings; kwargs win over env / .env.
    return Settings(**overrides)


def test_production_with_live_key_boots():
    s = _build(environment="production", stripe_secret_key="sk_live_example123")
    assert s.environment == "production"
    assert s.stripe_secret_key == "sk_live_example123"


def test_production_with_test_key_refuses():
    with pytest.raises(ValidationError) as ei:
        _build(environment="production", stripe_secret_key="sk_test_oops")
    msg = str(ei.value)
    assert "STRIPE_SECRET_KEY" in msg
    assert "sk_live_" in msg
    assert "production" in msg


def test_production_without_stripe_key_boots():
    # Stripe is optional — None / "" must not trip the guard.
    s = _build(environment="production", stripe_secret_key=None)
    assert s.environment == "production"
    s2 = _build(environment="production", stripe_secret_key="")
    assert s2.stripe_secret_key == ""


def test_staging_with_live_key_refuses():
    with pytest.raises(ValidationError) as ei:
        _build(environment="staging", stripe_secret_key="sk_live_dangerous")
    msg = str(ei.value)
    assert "STRIPE_SECRET_KEY" in msg
    assert "sk_live_" in msg
    assert "staging" in msg


def test_staging_with_test_key_boots():
    s = _build(environment="staging", stripe_secret_key="sk_test_safe")
    assert s.environment == "staging"
    assert s.stripe_secret_key == "sk_test_safe"


def test_development_with_live_key_refuses():
    with pytest.raises(ValidationError):
        _build(environment="development", stripe_secret_key="sk_live_x")


def test_whitespace_around_live_key_in_production_boots():
    # The strip-validator runs first; trailing whitespace must not bypass
    # the prefix check on either side.
    s = _build(environment="production", stripe_secret_key="  sk_live_padded  ")
    assert s.stripe_secret_key == "sk_live_padded"


def test_whitespace_around_live_key_in_staging_refuses():
    with pytest.raises(ValidationError):
        _build(environment="staging", stripe_secret_key="  sk_live_padded  ")
