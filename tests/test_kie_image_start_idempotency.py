"""`/llm/kie/image/start` — many sources in, and a retry that does not re-render.

Two claims, both about spending the user's money exactly once:

  1. `images_b64` decodes to N sources in order, the scalar `image_b64` still
     works for an agent that predates references, and a bad or oversized list
     is a 400 BEFORE any Kie job exists.
  2. A retry of this route under the same `idempotency_key` returns the job
     that already exists. A started Kie job is a billed job — `/start` takes a
     credit hold for exactly that reason — so creating a second one is charging
     twice for one request.

Driven against the real handler with the conftest's in-memory database, so the
marker row, the dedupe branch and the response shape are all executed. The
free-tier image cap is switched off in these (limit 0), because its TOCTOU
reservation takes a Postgres advisory lock — that path has its own coverage in
tests/test_kie_image.py.

Run: cd backend && env ENVIRONMENT=test STRIPE_SECRET_KEY=sk_test_x \
        PYTHONPATH=$(pwd) pytest tests/test_kie_image_start_idempotency.py -q
"""

from __future__ import annotations

import base64
import uuid

import pytest
from fastapi import HTTPException

pytestmark = pytest.mark.asyncio


class _Req:
    """The two things the handler asks a Request for."""

    def __init__(self, body: dict, token: str):
        self._body = body
        self.headers = {"authorization": f"Bearer {token}"}

    async def json(self):
        return self._body


@pytest.fixture
async def agent_token(test_user_id):
    """An AgentConfig whose llm token authenticates the proxy routes."""
    from app.api.llm_proxy import _hash_token
    from app.db import async_session_maker
    from app.db.models import AgentConfig

    token = f"tok-{uuid.uuid4().hex}"
    async with async_session_maker() as db:
        db.add(AgentConfig(user_id=test_user_id, llm_token_hash=_hash_token(token),
                           bundle_status="active"))
        await db.commit()
    return token


@pytest.fixture
def no_free_cap(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "free_tier_monthly_image_limit", 0, raising=False)


@pytest.fixture
def kie(monkeypatch):
    """Records every start_task call and mints a fresh task id per call."""
    from app.services import kie_client
    calls: list[dict] = []

    async def _start(mode, prompt, *, size=None, image_bytes=None,
                     mime="image/png", sources=None):
        calls.append({"mode": mode, "prompt": prompt,
                      "sources": list(sources or []),
                      "image_bytes": image_bytes})
        return f"task-{len(calls)}"
    monkeypatch.setattr(kie_client, "start_task", _start)
    return calls


async def _start(body, token, db):
    from app.api.llm_proxy import proxy_kie_image_start
    return await proxy_kie_image_start(_Req(body, token), db)


async def test_images_b64_reaches_the_provider_in_order(agent_token, no_free_cap, kie):
    from app.db import async_session_maker

    a, b, c = b"AAAA", b"BBBB", b"CCCC"
    body = {
        "mode": "edit", "prompt": "compose",
        "images_b64": [
            {"b64": base64.b64encode(a).decode(), "mime": "image/png"},
            {"b64": base64.b64encode(b).decode(), "mime": "image/jpeg"},
            {"b64": base64.b64encode(c).decode(), "mime": "image/webp"},
        ],
    }
    async with async_session_maker() as db:
        out = await _start(body, agent_token, db)

    assert out["task_id"] == "task-1"
    assert [s[0] for s in kie[0]["sources"]] == [a, b, c]
    assert [s[1] for s in kie[0]["sources"]] == ["image/png", "image/jpeg", "image/webp"]


async def test_the_scalar_form_still_starts_a_single_source_edit(agent_token, no_free_cap, kie):
    from app.db import async_session_maker

    body = {"mode": "edit", "prompt": "night",
            "image_b64": base64.b64encode(b"ZZZZ").decode(),
            "image_mime": "image/webp"}
    async with async_session_maker() as db:
        await _start(body, agent_token, db)

    assert kie[0]["sources"] == [(b"ZZZZ", "image/webp")]


async def test_over_the_cap_is_a_400_and_no_job(agent_token, no_free_cap, kie):
    from app.db import async_session_maker
    from app.services.kie_client import KIE_MAX_IMAGE_INPUT

    body = {"mode": "edit", "prompt": "compose", "images_b64": [
        {"b64": base64.b64encode(b"x").decode()}
        for _ in range(KIE_MAX_IMAGE_INPUT + 1)
    ]}
    async with async_session_maker() as db:
        with pytest.raises(HTTPException) as e:
            await _start(body, agent_token, db)
    assert e.value.status_code == 400
    assert kie == [], "an overflow must not start a billable render"


async def test_bad_base64_is_a_400_and_no_job(agent_token, no_free_cap, kie):
    from app.db import async_session_maker

    body = {"mode": "edit", "prompt": "compose",
            "images_b64": [{"b64": "not-base64!!!"}]}
    async with async_session_maker() as db:
        with pytest.raises(HTTPException) as e:
            await _start(body, agent_token, db)
    assert e.value.status_code == 400
    assert kie == []


async def test_a_retry_under_the_same_key_returns_the_same_task(agent_token, no_free_cap, kie):
    """The response was lost, not the render. Do not pay for a second one."""
    from app.db import async_session_maker

    key = uuid.uuid4().hex
    body = {"mode": "edit", "prompt": "compose", "idempotency_key": key,
            "image_b64": base64.b64encode(b"AAAA").decode()}

    async with async_session_maker() as db:
        first = await _start(body, agent_token, db)
    async with async_session_maker() as db:
        second = await _start(dict(body), agent_token, db)

    assert first["task_id"] == "task-1"
    assert second["task_id"] == "task-1", "the retry must not start a second render"
    assert second.get("deduped") is True
    assert len(kie) == 1, "exactly one createTask for one logical request"


async def test_a_different_key_is_a_different_render(agent_token, no_free_cap, kie):
    from app.db import async_session_maker

    body = {"mode": "edit", "prompt": "compose",
            "image_b64": base64.b64encode(b"AAAA").decode()}
    async with async_session_maker() as db:
        one = await _start(dict(body, idempotency_key=uuid.uuid4().hex), agent_token, db)
    async with async_session_maker() as db:
        two = await _start(dict(body, idempotency_key=uuid.uuid4().hex), agent_token, db)

    assert one["task_id"] != two["task_id"]
    assert len(kie) == 2


async def test_no_key_at_all_behaves_exactly_as_before(agent_token, no_free_cap, kie):
    """Absence of the field must never become a refusal or a dedupe."""
    from app.db import async_session_maker

    body = {"mode": "edit", "prompt": "compose",
            "image_b64": base64.b64encode(b"AAAA").decode()}
    async with async_session_maker() as db:
        one = await _start(dict(body), agent_token, db)
    async with async_session_maker() as db:
        two = await _start(dict(body), agent_token, db)

    assert one["task_id"] == "task-1" and two["task_id"] == "task-2"
    assert "deduped" not in one


async def test_the_marker_is_written_only_after_the_job_exists(agent_token, no_free_cap, monkeypatch):
    """A marker naming a task that was never created would suppress the retry
    that is the whole point — so a failed createTask must leave nothing."""
    from app.db import async_session_maker
    from app.services import kie_client
    from sqlalchemy import select
    from app.db.models import CreditReservation

    async def _boom(*a, **kw):
        raise kie_client.KieError("provider down")
    monkeypatch.setattr(kie_client, "start_task", _boom)

    key = uuid.uuid4().hex
    body = {"mode": "edit", "prompt": "compose", "idempotency_key": key,
            "image_b64": base64.b64encode(b"AAAA").decode()}
    async with async_session_maker() as db:
        with pytest.raises(HTTPException) as e:
            await _start(body, agent_token, db)
        assert e.value.status_code == 502
        rows = (await db.execute(select(CreditReservation).where(
            CreditReservation.idempotency_key == f"kie_idem:{key}"))).scalars().all()
    assert rows == [], "no job, no marker — the next attempt must be free to start one"


async def test_the_marker_is_invisible_to_the_free_image_cap(agent_token, no_free_cap, kie):
    """It lives in credit_reservations for its UNIQUE index, not as a hold.

    `reserve_free_image_slot` counts OPEN reservations whose event_type is the
    image ledger type; a marker that shared that type would consume one of a
    free user's monthly images per render, silently halving the allowance.
    """
    from app.db import async_session_maker
    from sqlalchemy import select
    from app.db.models import CreditReservation, LEDGER_IMAGE_GEN

    key = uuid.uuid4().hex
    body = {"mode": "edit", "prompt": "compose", "idempotency_key": key,
            "image_b64": base64.b64encode(b"AAAA").decode()}
    async with async_session_maker() as db:
        await _start(body, agent_token, db)
        row = (await db.execute(select(CreditReservation).where(
            CreditReservation.idempotency_key == f"kie_idem:{key}"))).scalar_one()

    assert row.event_type != LEDGER_IMAGE_GEN
    assert row.event_type == "kie_idem"
    assert float(row.estimated_amount) == 0.0
    assert (row.metadata_json or {}).get("task_id") == "task-1"


def test_decode_image_sources_prefers_the_list_but_keeps_the_scalar():
    from app.api.llm_proxy import _decode_image_sources

    assert _decode_image_sources({}) == []
    assert _decode_image_sources({
        "image_b64": base64.b64encode(b"a").decode(), "image_mime": "image/gif",
    }) == [(b"a", "image/gif")]
    # A list wins, and a bare string entry defaults to png.
    assert _decode_image_sources({
        "image_b64": base64.b64encode(b"ignored").decode(),
        "images_b64": [base64.b64encode(b"a").decode(),
                       {"b64": base64.b64encode(b"b").decode(), "mime": "image/webp"}],
    }) == [(b"a", "image/png"), (b"b", "image/webp")]
