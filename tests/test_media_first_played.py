"""The first-media-played latch behind the progressively-disclosed Media nav entry.

The mobile sidebar hides "Media" until the account has actually heard something
play, then shows it forever. That decision is only as good as three properties,
and each test below pins one of them:

  * the flag is ALWAYS on the /auth/me wire, `null` included. A UI element gated
    on a field the server sometimes omits is not degraded, it is absent — the
    exact defect that left the Usage screen's daily meter dead for every user on
    every build. `test_me_always_carries_the_field` is the tripwire for that.
  * the write is monotonic, enforced server-side. The client retries this call
    freely (it never blocks the UI on it) and two devices can report at once, so
    write-once cannot be a client promise.
  * it is genuinely per-account, since it rides a shared /auth/me payload.
"""

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy import select

from app.db import async_session_maker
from app.db.models import User


@pytest.mark.asyncio
async def test_me_always_carries_the_field(client, auth_headers) -> None:
    """A fresh account: the key is PRESENT and null — never missing."""
    res = await client.get("/api/auth/me", headers=auth_headers)
    assert res.status_code == 200, res.text
    body = res.json()
    assert "first_media_played_at" in body, (
        "the client gates a nav row on this key; a server that omits it hides "
        "the row for everyone, on every build, silently"
    )
    assert body["first_media_played_at"] is None


@pytest.mark.asyncio
async def test_report_stamps_and_me_surfaces_it(
    client, auth_headers, test_user_id
) -> None:
    res = await client.post("/api/account/media-played", headers=auth_headers)
    assert res.status_code == 200, res.text
    stamped = res.json()["first_media_played_at"]
    assert stamped is not None

    async with async_session_maker() as db:
        row = (
            await db.execute(select(User).where(User.id == test_user_id))
        ).scalar_one()
        assert row.first_media_played_at is not None

    me = await client.get("/api/auth/me", headers=auth_headers)
    assert me.status_code == 200
    assert me.json()["first_media_played_at"] == stamped


@pytest.mark.asyncio
async def test_write_is_once_and_the_answer_is_the_first_instant(
    client, auth_headers, test_user_id
) -> None:
    """A replayed report must not move the instant, and must not lie about it.

    The endpoint returns the value it READ back, not the `now` it just tried to
    write — on every call after the first the UPDATE matches no row, and the
    honest answer is the original instant.
    """
    first = (await client.post("/api/account/media-played", headers=auth_headers)).json()
    await asyncio.sleep(0.01)
    second = (await client.post("/api/account/media-played", headers=auth_headers)).json()

    assert second["first_media_played_at"] == first["first_media_played_at"]

    async with async_session_maker() as db:
        row = (
            await db.execute(select(User).where(User.id == test_user_id))
        ).scalar_one()
        assert row.first_media_played_at.isoformat() == first["first_media_played_at"]


@pytest.mark.asyncio
async def test_concurrent_reports_agree(client, auth_headers) -> None:
    """Two devices reporting at once still yield ONE instant.

    The guard is the `WHERE first_media_played_at IS NULL` predicate, not a
    read-then-write in Python, which is exactly the shape that races.
    """
    results = await asyncio.gather(
        *[client.post("/api/account/media-played", headers=auth_headers) for _ in range(4)]
    )
    assert {r.status_code for r in results} == {200}
    instants = {r.json()["first_media_played_at"] for r in results}
    assert len(instants) == 1, f"racing reports produced {instants}"


@pytest.mark.asyncio
async def test_report_requires_auth(client) -> None:
    res = await client.post("/api/account/media-played")
    assert res.status_code in (401, 403), res.text
