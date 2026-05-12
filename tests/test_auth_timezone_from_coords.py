"""Tests for POST /api/auth/timezone-from-coords.

Account page's "Share precise location" flow: after the user grants
geolocation permission, we POST (lat, lng) here and the server derives
the IANA name via timezonefinder's bundled shapefile (no external API).
These tests pin the contract that:
  - real city coords resolve to the right IANA name
  - persists on User.timezone
  - validates coordinate bounds
  - requires authentication
"""

from __future__ import annotations

import pytest
from sqlalchemy import select

from app.db import async_session_maker
from app.db.models import User


@pytest.mark.asyncio
async def test_resolves_toronto_coords_to_toronto_zone(
    client, auth_headers, test_user_id
) -> None:
    res = await client.post(
        "/api/auth/timezone-from-coords",
        json={"lat": 43.6532, "lng": -79.3832},  # Toronto City Hall
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
async def test_resolves_tokyo_coords_to_tokyo_zone(client, auth_headers) -> None:
    res = await client.post(
        "/api/auth/timezone-from-coords",
        json={"lat": 35.6895, "lng": 139.6917},  # Tokyo
        headers=auth_headers,
    )
    assert res.status_code == 200
    assert res.json()["timezone"] == "Asia/Tokyo"


@pytest.mark.asyncio
async def test_rejects_out_of_range_latitude(client, auth_headers) -> None:
    res = await client.post(
        "/api/auth/timezone-from-coords",
        json={"lat": 91.0, "lng": 0.0},
        headers=auth_headers,
    )
    assert res.status_code == 400


@pytest.mark.asyncio
async def test_rejects_out_of_range_longitude(client, auth_headers) -> None:
    res = await client.post(
        "/api/auth/timezone-from-coords",
        json={"lat": 0.0, "lng": -200.0},
        headers=auth_headers,
    )
    assert res.status_code == 400


@pytest.mark.asyncio
async def test_requires_authentication(client) -> None:
    res = await client.post(
        "/api/auth/timezone-from-coords",
        json={"lat": 43.6532, "lng": -79.3832},
    )
    assert res.status_code == 401


@pytest.mark.asyncio
async def test_idempotent_when_zone_unchanged(
    client, auth_headers, test_user_id
) -> None:
    """Repeated calls from the same location shouldn't bump updated_at —
    matches the silent /auth/profile path's idempotency contract."""
    await client.post(
        "/api/auth/timezone-from-coords",
        json={"lat": 52.52, "lng": 13.40},  # Berlin
        headers=auth_headers,
    )
    async with async_session_maker() as db:
        before = (
            await db.execute(select(User).where(User.id == test_user_id))
        ).scalar_one()
        ts_before = before.updated_at

    res = await client.post(
        "/api/auth/timezone-from-coords",
        json={"lat": 52.52, "lng": 13.40},
        headers=auth_headers,
    )
    assert res.status_code == 200
    async with async_session_maker() as db:
        after = (
            await db.execute(select(User).where(User.id == test_user_id))
        ).scalar_one()
        assert after.updated_at == ts_before
