"""Tests for the silent-timezone-capture path on PATCH /api/auth/profile.

The frontend reads `Intl.DateTimeFormat().resolvedOptions().timeZone`
on app boot and PATCHes here so scheduled actions (morning briefing,
day-as-chat boundaries) fire on the user's actual local clock. No
permission popup, no UI — these tests pin the contract that lets the
backend trust that drip.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select

from app.db import async_session_maker
from app.db.models import User


@pytest.mark.asyncio
async def test_profile_patch_persists_valid_iana_timezone(
    client, auth_headers, test_user_id
) -> None:
    res = await client.patch(
        "/api/auth/profile",
        json={"timezone": "America/Toronto"},
        headers=auth_headers,
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["timezone"] == "America/Toronto"

    async with async_session_maker() as db:
        row = (
            await db.execute(select(User).where(User.id == test_user_id))
        ).scalar_one()
        assert row.timezone == "America/Toronto"


@pytest.mark.asyncio
async def test_profile_patch_rejects_invalid_timezone(
    client, auth_headers
) -> None:
    """Spoofed or typo'd values must 400, not silently land in the DB.
    zoneinfo treats unknown names as ZoneInfoNotFoundError."""
    res = await client.patch(
        "/api/auth/profile",
        json={"timezone": "Mars/Olympus_Mons"},
        headers=auth_headers,
    )
    assert res.status_code == 400, res.text


@pytest.mark.asyncio
async def test_profile_patch_idempotent_when_timezone_unchanged(
    client, auth_headers, test_user_id
) -> None:
    """Frontend re-fires capture on every page load. The endpoint must
    no-op the write when the value matches — otherwise we'd thrash
    updated_at and write rows for hundreds of users per page navigation."""
    # First call: set it.
    await client.patch(
        "/api/auth/profile",
        json={"timezone": "Europe/Berlin"},
        headers=auth_headers,
    )
    async with async_session_maker() as db:
        before = (
            await db.execute(select(User).where(User.id == test_user_id))
        ).scalar_one()
        ts_before = before.updated_at

    # Second call: same value. updated_at should NOT advance.
    res = await client.patch(
        "/api/auth/profile",
        json={"timezone": "Europe/Berlin"},
        headers=auth_headers,
    )
    assert res.status_code == 200
    async with async_session_maker() as db:
        after = (
            await db.execute(select(User).where(User.id == test_user_id))
        ).scalar_one()
        assert after.updated_at == ts_before


@pytest.mark.asyncio
async def test_profile_patch_applies_name_and_timezone_together(
    client, auth_headers, test_user_id
) -> None:
    """The frontend can (and on the Profile section, does) send both
    fields in one PATCH. Both must land."""
    res = await client.patch(
        "/api/auth/profile",
        json={"name": "Renamed User", "timezone": "Asia/Tokyo"},
        headers=auth_headers,
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["name"] == "Renamed User"
    assert body["timezone"] == "Asia/Tokyo"

    async with async_session_maker() as db:
        row = (
            await db.execute(select(User).where(User.id == test_user_id))
        ).scalar_one()
        assert row.name == "Renamed User"
        assert row.timezone == "Asia/Tokyo"


@pytest.mark.asyncio
async def test_profile_patch_requires_authentication(client) -> None:
    res = await client.patch(
        "/api/auth/profile", json={"timezone": "America/Toronto"}
    )
    assert res.status_code == 401
