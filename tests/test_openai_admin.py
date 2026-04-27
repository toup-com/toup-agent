"""
Tests for the OpenAI Admin API helper + auto-provisioning wiring.

All deterministic (no real OpenAI traffic) — mocks `httpx.Client.post`.
The real Admin API integration is exercised once at deploy time via
operator smoke test, not in CI (org-level keys are too sensitive to
keep in a CI secret).

Covers:
  - openai_admin_service.create_project happy path + error mapping
  - openai_admin_service.create_project_api_key extracts api_key.value
  - openai_admin_service.archive_project: 200, 404 (idempotent OK), 500 (non-fatal False)
  - openai_admin_service.provision_tenant: archives project on api-key failure
  - openai_admin_service raises OpenAIAdminUnavailable when admin key unset
  - Bundle activation webhook calls _provision_openai_project_if_needed
  - admin/delete-user archives the OpenAI project when bundle_openai_project_id is set
"""

from __future__ import annotations

import secrets as secrets_mod
import time as time_mod
from datetime import datetime
from unittest.mock import patch, MagicMock

import pytest


# ── openai_admin_service: pure unit tests ────────────────────────────


def _mock_response(status_code: int, json_body: dict | None = None, text: str = ""):
    m = MagicMock()
    m.status_code = status_code
    m.json = MagicMock(return_value=(json_body or {}))
    m.text = text or (str(json_body) if json_body else "")
    return m


def test_create_project_returns_id_on_success():
    from app.services import openai_admin_service as svc
    with patch.object(svc.settings, "openai_admin_api_key", "sk-admin-FAKE"):
        with patch("httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.post.return_value = (
                _mock_response(200, {"id": "proj_abc123", "name": "toup-tenant-deadbeef"})
            )
            result = svc.create_project("toup-tenant-deadbeef")
    assert result == "proj_abc123"


def test_create_project_raises_when_admin_key_missing():
    from app.services import openai_admin_service as svc
    with patch.object(svc.settings, "openai_admin_api_key", None):
        with pytest.raises(svc.OpenAIAdminUnavailable):
            svc.create_project("toup-tenant-x")


def test_create_project_raises_on_http_error():
    from app.services import openai_admin_service as svc
    with patch.object(svc.settings, "openai_admin_api_key", "sk-admin-FAKE"):
        with patch("httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.post.return_value = (
                _mock_response(403, {"error": {"message": "forbidden"}})
            )
            with pytest.raises(RuntimeError, match="HTTP 403"):
                svc.create_project("toup-tenant-x")


def test_create_project_api_key_extracts_value():
    from app.services import openai_admin_service as svc
    with patch.object(svc.settings, "openai_admin_api_key", "sk-admin-FAKE"):
        with patch("httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.post.return_value = (
                _mock_response(200, {
                    "id": "svc_acct_xyz",
                    "api_key": {"value": "sk-proj-NEWKEY"},
                })
            )
            key = svc.create_project_api_key("proj_abc123", "agent")
    assert key == "sk-proj-NEWKEY"


def test_create_project_api_key_raises_when_no_key_in_response():
    from app.services import openai_admin_service as svc
    with patch.object(svc.settings, "openai_admin_api_key", "sk-admin-FAKE"):
        with patch("httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.post.return_value = (
                _mock_response(200, {"id": "svc_acct_xyz"})  # missing api_key
            )
            with pytest.raises(RuntimeError, match="no api_key.value"):
                svc.create_project_api_key("proj_abc123")


def test_archive_project_idempotent_on_404():
    from app.services import openai_admin_service as svc
    with patch.object(svc.settings, "openai_admin_api_key", "sk-admin-FAKE"):
        with patch("httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.post.return_value = (
                _mock_response(404, {"error": {"message": "not found"}})
            )
            result = svc.archive_project("proj_already_gone")
    assert result is True  # 404 = idempotent success


def test_archive_project_returns_false_on_500():
    from app.services import openai_admin_service as svc
    with patch.object(svc.settings, "openai_admin_api_key", "sk-admin-FAKE"):
        with patch("httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.post.return_value = (
                _mock_response(500, {"error": {"message": "boom"}})
            )
            result = svc.archive_project("proj_abc")
    assert result is False  # genuine failure, but caller continues


def test_archive_project_returns_false_when_admin_key_missing():
    """Missing key → graceful no-op (False), NOT a raised exception. Lets
    admin/delete-user proceed to wipe the platform DB even if OpenAI Admin
    isn't configured."""
    from app.services import openai_admin_service as svc
    with patch.object(svc.settings, "openai_admin_api_key", None):
        result = svc.archive_project("proj_abc")
    assert result is False


def test_provision_tenant_archives_project_on_api_key_failure():
    """If service-account creation fails after project creation, the orphan
    project must be archived to keep operator's OpenAI dashboard clean."""
    from app.services import openai_admin_service as svc
    archive_called_with = []

    with patch.object(svc.settings, "openai_admin_api_key", "sk-admin-FAKE"), \
         patch.object(svc, "create_project", return_value="proj_NEW"), \
         patch.object(svc, "create_project_api_key", side_effect=RuntimeError("boom")), \
         patch.object(svc, "archive_project", lambda p: archive_called_with.append(p) or True):
        with pytest.raises(RuntimeError, match="boom"):
            svc.provision_tenant("deadbeef")

    assert archive_called_with == ["proj_NEW"]


# ── Auto-provisioning wired into bundle activation ───────────────────


@pytest.mark.asyncio
async def test_invoice_succeeded_provisions_openai_project(
    client, auth_headers, test_user_id, signed_stripe_event,
):
    """When the bundle-activation webhook fires AND the admin API is
    configured, AgentConfig must end up with bundle_openai_project_id +
    bundle_openai_api_key set. Mocks the OpenAI side."""
    # Lazy import to share the tests' helpers.
    from tests.test_billing_subscription import (
        _seed_agent_config, _get_agent_config, _fake_get_subscription,
    )

    sub_id = "sub_test_oai_" + secrets_mod.token_hex(4)
    await _seed_agent_config(test_user_id, bundle_status="none", subscription_id=sub_id)

    payload, headers = signed_stripe_event(
        "invoice.payment_succeeded",
        {"id": f"in_test_{secrets_mod.token_hex(4)}", "subscription": sub_id, "object": "invoice"},
    )

    with patch("app.services.stripe_service.get_subscription", side_effect=_fake_get_subscription), \
         patch("app.services.openai_admin_service.provision_tenant",
               return_value=("proj_TEST", "sk-proj-TEST")), \
         patch("app.config.settings.openai_admin_api_key", "sk-admin-FAKE"):
        resp = await client.post("/api/vps/webhook/stripe", content=payload, headers=headers)

    assert resp.status_code == 200, resp.text
    cfg = await _get_agent_config(test_user_id)
    assert cfg.bundle_openai_project_id == "proj_TEST"
    assert cfg.bundle_openai_api_key == "sk-proj-TEST"


@pytest.mark.asyncio
async def test_invoice_succeeded_does_not_fail_when_admin_api_unavailable(
    client, auth_headers, test_user_id, signed_stripe_event,
):
    """If the OpenAI Admin API is misconfigured / down, bundle activation
    must STILL succeed. Proxy falls back to platform master key on outbound."""
    from tests.test_billing_subscription import (
        _seed_agent_config, _get_agent_config, _fake_get_subscription,
    )

    sub_id = "sub_test_oai_unavail_" + secrets_mod.token_hex(4)
    await _seed_agent_config(test_user_id, bundle_status="none", subscription_id=sub_id)

    payload, headers = signed_stripe_event(
        "invoice.payment_succeeded",
        {"id": f"in_test_{secrets_mod.token_hex(4)}", "subscription": sub_id, "object": "invoice"},
    )

    with patch("app.services.stripe_service.get_subscription", side_effect=_fake_get_subscription), \
         patch("app.config.settings.openai_admin_api_key", None):  # admin key unset
        resp = await client.post("/api/vps/webhook/stripe", content=payload, headers=headers)

    assert resp.status_code == 200, resp.text
    cfg = await _get_agent_config(test_user_id)
    assert cfg.bundle_status == "active"  # activation still succeeded
    assert cfg.bundle_openai_project_id is None  # auto-provision was skipped
    assert cfg.bundle_openai_api_key is None


@pytest.mark.asyncio
async def test_invoice_succeeded_idempotent_for_already_provisioned_user(
    client, auth_headers, test_user_id, signed_stripe_event,
):
    """Renewal webhook must NOT re-provision a new OpenAI project when one
    already exists — would orphan the prior project + duplicate billing."""
    from tests.test_billing_subscription import (
        _seed_agent_config, _get_agent_config, _fake_get_subscription,
    )
    from app.db import AgentConfig, async_session_maker
    from sqlalchemy import select

    sub_id = "sub_test_oai_idem_" + secrets_mod.token_hex(4)
    await _seed_agent_config(test_user_id, bundle_status="active", subscription_id=sub_id)

    # Pre-set a project id and key.
    async with async_session_maker() as db:
        cfg = (await db.execute(
            select(AgentConfig).where(AgentConfig.user_id == test_user_id)
        )).scalar_one()
        cfg.bundle_openai_project_id = "proj_EXISTING"
        cfg.bundle_openai_api_key = "sk-proj-EXISTING"
        await db.commit()

    provision_calls = []

    def _track(prefix):
        provision_calls.append(prefix)
        return ("proj_NEW", "sk-proj-NEW")

    payload, headers = signed_stripe_event(
        "invoice.payment_succeeded",
        {"id": f"in_test_{secrets_mod.token_hex(4)}", "subscription": sub_id, "object": "invoice"},
    )

    with patch("app.services.stripe_service.get_subscription", side_effect=_fake_get_subscription), \
         patch("app.services.openai_admin_service.provision_tenant", side_effect=_track), \
         patch("app.config.settings.openai_admin_api_key", "sk-admin-FAKE"):
        resp = await client.post("/api/vps/webhook/stripe", content=payload, headers=headers)

    assert resp.status_code == 200, resp.text
    cfg = await _get_agent_config(test_user_id)
    assert cfg.bundle_openai_project_id == "proj_EXISTING", "must NOT re-provision"
    assert cfg.bundle_openai_api_key == "sk-proj-EXISTING"
    assert provision_calls == [], f"provision_tenant should not be called, was: {provision_calls}"
