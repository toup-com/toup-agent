"""End-to-end integration tests for POST /api/auth/reauth and
POST /api/auth/delete-account.

These exercise the real FastAPI app via the `client` fixture from
conftest.py. External services (Stripe, docker_host bridge, OpenAI
admin) are monkey-patched at the module level so the cascade runs
to completion in-process.

The mobile + web clients consume:
  - /api/auth/reauth          → returns a 5-min sensitive-action token
  - /api/auth/delete-account  → requires the SAT in
                                 X-Sensitive-Action-Token header
"""

from __future__ import annotations

import pytest
from sqlalchemy import select

from app.db import async_session_maker
from app.db.models import User


# ── Fixtures: silence the external-service calls ───────────────────


@pytest.fixture(autouse=True)
def patch_externals(monkeypatch):
    """Auto-applied to every test in this file. Stripe, container
    teardown, and OpenAI archive all become no-ops that succeed."""
    def fake_delete_customer(cid):
        return True

    def fake_find_customers_by_email(email):
        return []

    async def fake_destroy_container(db, user_id):
        return None

    def fake_archive_project(project_id):
        return None

    monkeypatch.setattr(
        "app.services.stripe_service.delete_customer", fake_delete_customer,
    )
    monkeypatch.setattr(
        "app.services.stripe_service.find_customers_by_email",
        fake_find_customers_by_email,
    )
    monkeypatch.setattr(
        "app.services.docker_host_service.destroy_container",
        fake_destroy_container,
    )
    monkeypatch.setattr(
        "app.services.openai_admin_service.archive_project",
        fake_archive_project,
    )


# ── /auth/reauth ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reauth_requires_authentication(client) -> None:
    """No Bearer token → 401."""
    res = await client.post("/api/auth/reauth")
    assert res.status_code == 401


@pytest.mark.asyncio
async def test_reauth_returns_sensitive_action_token(client, auth_headers) -> None:
    res = await client.post("/api/auth/reauth", headers=auth_headers)
    assert res.status_code == 200
    body = res.json()
    assert "sensitive_action_token" in body
    assert isinstance(body["sensitive_action_token"], str)
    assert len(body["sensitive_action_token"]) > 50
    assert body["expires_in"] == 300


# ── /auth/delete-account ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_delete_account_requires_sensitive_action_token(
    client, auth_headers,
) -> None:
    """Authenticated but no X-Sensitive-Action-Token → 401."""
    res = await client.post("/api/auth/delete-account", headers=auth_headers)
    assert res.status_code == 401


@pytest.mark.asyncio
async def test_delete_account_rejects_session_jwt_as_sat(
    client, auth_headers,
) -> None:
    """Passing the regular session JWT in X-Sensitive-Action-Token
    must be rejected — that token has no `kind: sensitive_action`
    claim so the SAT verifier rejects it."""
    # Lift the bearer JWT out of auth_headers and pass it in the
    # SAT header to force the wrong-kind path.
    bearer = auth_headers["Authorization"].split(" ", 1)[1]
    res = await client.post(
        "/api/auth/delete-account",
        headers={
            **auth_headers,
            "X-Sensitive-Action-Token": bearer,
        },
    )
    assert res.status_code == 401


@pytest.mark.asyncio
async def test_delete_account_full_roundtrip(
    client, auth_headers, test_user_id,
) -> None:
    """The mainline user flow:
      1. POST /api/auth/reauth → SAT
      2. POST /api/auth/delete-account with SAT → 200
      3. User row is gone from the DB.
    """
    reauth = await client.post("/api/auth/reauth", headers=auth_headers)
    assert reauth.status_code == 200
    sat = reauth.json()["sensitive_action_token"]

    delete = await client.post(
        "/api/auth/delete-account",
        headers={
            **auth_headers,
            "X-Sensitive-Action-Token": sat,
        },
    )
    assert delete.status_code == 200
    body = delete.json()
    assert body["success"] is True
    receipt = body["receipt"]
    assert receipt["user_id"] == test_user_id
    assert receipt["user_row_deleted"] is True
    assert receipt["actor"] == "self"

    # User row should be gone.
    async with async_session_maker() as db:
        result = await db.execute(
            select(User).where(User.id == test_user_id)
        )
        assert result.scalar_one_or_none() is None


@pytest.mark.asyncio
async def test_delete_account_rejects_replay(
    client, auth_headers, test_user_id,
) -> None:
    """Replaying the same SAT against /delete-account fails on the
    second call. (First call succeeds and deletes the user; second
    call's SAT verify hits the redemption row and 401s. Even if the
    user weren't deleted, the second call should still 401 — that's
    the contract.)

    To exercise this without the user-already-gone confound, we
    DON'T let the first call succeed: we mock destroy_container to
    fail, so the cascade aborts at CONTAINER, the user row stays,
    but the SAT redemption row is committed pre-cascade and
    therefore burned. Replay then fails on the same SAT with a
    fresh token-already-redeemed 401.
    """
    import pytest as _pytest_module  # local alias

    # Override container teardown to fail for this test.
    from unittest.mock import patch

    async def fail_container(db, user_id):
        raise RuntimeError("boom")

    reauth = await client.post("/api/auth/reauth", headers=auth_headers)
    sat = reauth.json()["sensitive_action_token"]

    with patch(
        "app.services.docker_host_service.destroy_container",
        side_effect=fail_container,
    ):
        first = await client.post(
            "/api/auth/delete-account",
            headers={**auth_headers, "X-Sensitive-Action-Token": sat},
        )
        # Cascade aborted at CONTAINER → 502.
        assert first.status_code == 502
        body = first.json()
        # FastAPI wraps detail in a "detail" key.
        assert body["detail"]["step"] == "container"

    # Replay attempt with the same SAT.
    second = await client.post(
        "/api/auth/delete-account",
        headers={**auth_headers, "X-Sensitive-Action-Token": sat},
    )
    assert second.status_code == 401


@pytest.mark.asyncio
async def test_delete_account_502_on_cascade_abort(
    client, auth_headers, test_user_id,
) -> None:
    """Cascade abort returns 502 with step + message body, NOT 200.
    This was the behavior change vs the previous implementation
    which swallowed the exception and returned 200 with the cookie
    cleared while Stripe billing kept running."""
    from unittest.mock import patch

    def fail_stripe(cid):
        raise RuntimeError("simulated stripe outage")

    reauth = await client.post("/api/auth/reauth", headers=auth_headers)
    sat = reauth.json()["sensitive_action_token"]

    # Need to pre-populate stripe_customer_id so the Stripe step actually
    # tries to call delete_customer — otherwise it skips (no customer).
    async with async_session_maker() as db:
        u = (await db.execute(
            select(User).where(User.id == test_user_id)
        )).scalar_one()
        u.stripe_customer_id = "cus_will_fail"
        await db.commit()

    with patch(
        "app.services.stripe_service.delete_customer",
        side_effect=fail_stripe,
    ):
        res = await client.post(
            "/api/auth/delete-account",
            headers={**auth_headers, "X-Sensitive-Action-Token": sat},
        )
    assert res.status_code == 502
    body = res.json()
    assert body["detail"]["step"] == "stripe"
    assert "simulated stripe outage" in body["detail"]["message"]

    # User row still exists — abort means data preserved.
    async with async_session_maker() as db:
        result = await db.execute(
            select(User).where(User.id == test_user_id)
        )
        assert result.scalar_one_or_none() is not None
