"""Cache telemetry (F-7 / A9-1) — cached_tokens captured, persisted, exposed.

The token-efficiency audit (audit_A9 / verify_F-F) found that
``usage.prompt_tokens_details.cached_tokens`` was captured on every chat
call but never persisted anywhere — the only cache-hit observable was
grepping ``[PERF] cache_read=`` in tenant container logs. These tests pin
the four legs of the fix:

  1. Proxy extraction reads cached tokens from BOTH shapes (streamed SSE
     usage chunks and non-stream JSON bodies) — real unit tests, pure fns.
  2. llm_proxy_events grows an additive-only nullable cached_tokens column
     (migration 075 + init_db mirror) threaded through _log_event.
  3. The BYOK/direct path forwards cached_tokens in the agent-deduct
     payload; the platform stores it in the ledger row's metadata_json.
  4. The admin rollup endpoint exists behind the require_admin guard
     (401 unauthenticated / 422 bad params — never 405).

Billing math is UNTOUCHED by design: A9-2 (cost_cents prices cached input
at the full rate) and A9-3 (per-session idempotency key under-metering)
are documented out of scope; source-grep tests below pin that the charge
paths did not change.
"""

from __future__ import annotations

import inspect
import types
from pathlib import Path

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api import llm_proxy as lp


# ── 1. Streamed SSE shape ────────────────────────────────────────────


def _sse(*objs: str) -> bytes:
    return ("".join(f"data: {o}\n\n" for o in objs) + "data: [DONE]\n\n").encode()


def test_openai_stream_usage_extracts_cached_tokens():
    raw = _sse(
        '{"choices": [{"delta": {"content": "hi"}}]}',
        '{"usage": {"prompt_tokens": 1200, "completion_tokens": 40,'
        ' "prompt_tokens_details": {"cached_tokens": 1024}}}',
    )
    assert lp._extract_openai_usage(raw) == (1200, 40, 1024)


def test_openai_stream_usage_without_details_is_zero_cached():
    """Older API versions omit prompt_tokens_details entirely."""
    raw = _sse('{"usage": {"prompt_tokens": 300, "completion_tokens": 7}}')
    assert lp._extract_openai_usage(raw) == (300, 7, 0)


def test_openai_stream_usage_null_details_is_zero_cached():
    raw = _sse(
        '{"usage": {"prompt_tokens": 300, "completion_tokens": 7,'
        ' "prompt_tokens_details": null}}'
    )
    assert lp._extract_openai_usage(raw) == (300, 7, 0)


def test_anthropic_stream_usage_extracts_cache_read():
    raw = _sse(
        '{"type": "message_start", "message": {"usage": {"input_tokens": 900,'
        ' "cache_read_input_tokens": 850}}}',
        '{"type": "message_delta", "usage": {"output_tokens": 33}}',
    )
    assert lp._extract_anthropic_usage(raw) == (900, 33, 850)


# ── 1b. Non-stream JSON shape (shared helper) ────────────────────────


def test_nonstream_usage_dict_with_cached_details():
    usage = {
        "prompt_tokens": 500,
        "completion_tokens": 20,
        "prompt_tokens_details": {"cached_tokens": 448},
    }
    assert lp._extract_openai_cached_tokens(usage) == 448


@pytest.mark.parametrize("usage", [
    {"prompt_tokens": 500, "completion_tokens": 20},
    {"prompt_tokens_details": None},
    {"prompt_tokens_details": {}},
    {"prompt_tokens_details": {"cached_tokens": None}},
    {"prompt_tokens_details": "bogus"},
])
def test_nonstream_usage_dict_defaults_to_zero(usage):
    assert lp._extract_openai_cached_tokens(usage) == 0


def test_nonstream_path_uses_shared_cached_helper():
    """The non-stream JSON branch of proxy_chat must read cached tokens
    through the same helper as the SSE extractor so the two shapes can't
    drift apart."""
    src = inspect.getsource(lp.proxy_chat)
    assert "_extract_openai_cached_tokens(usage)" in src
    assert 'usage.get("cache_read_input_tokens", 0)' in src  # anthropic twin


# ── 2. Persistence: column + migration + _log_event threading ────────


def test_llm_proxy_event_has_cached_tokens_column():
    from app.db.models.platform import LLMProxyEvent
    col = LLMProxyEvent.__table__.columns["cached_tokens"]
    assert col.nullable, "cached_tokens must be nullable (additive-only)"


def test_log_event_threads_cached_tokens():
    src = inspect.getsource(lp._log_event)
    assert "cached_tokens=cached_tokens" in src


def test_stream_and_nonstream_call_sites_pass_cached():
    src = inspect.getsource(lp.proxy_chat)
    assert src.count("cached_tokens=cached") >= 2, (
        "both the stream_and_log finally-block and the non-stream branch "
        "must pass cached_tokens to _log_event"
    )


def test_migration_075_is_additive_only():
    mig = Path(__file__).resolve().parents[1] / "alembic" / "versions"
    src = next(mig.glob("*075*")).read_text()
    assert "add_column" in src
    assert "cached_tokens" in src
    # Additive-only: upgrade() must not drop or alter anything.
    upgrade_src = src.split("def upgrade")[1].split("def downgrade")[0]
    assert "drop_column" not in upgrade_src
    assert "drop_table" not in upgrade_src
    assert "alter_column" not in upgrade_src


def test_init_db_mirrors_cached_tokens_alter():
    """platform-api boots via init_db (alembic is best-effort) — the 073
    precedent requires the ADD COLUMN mirror or _log_event 500s on a
    pre-075 platform DB."""
    from app.db import database as dbm
    src = inspect.getsource(dbm)
    assert (
        "ALTER TABLE llm_proxy_events ADD COLUMN IF NOT EXISTS cached_tokens INTEGER"
        in src
    )


# ── 3. BYOK/direct path: reporter payload + deduct metadata ──────────


def test_report_llm_usage_accepts_cached_tokens():
    from app.services.credit_reporter import report_llm_usage
    sig = inspect.signature(report_llm_usage)
    param = sig.parameters.get("cached_tokens")
    assert param is not None
    assert param.default is None, "must default to None (old agents omit it)"
    src = inspect.getsource(report_llm_usage)
    assert 'payload["cached_tokens"]' in src


def test_openai_service_forwards_cached_tokens_at_report_time():
    from app.services import openai_agent_service as oas
    src = inspect.getsource(oas.OpenAIAgentService.create_message_stream)
    assert 'cached_tokens=int(usage_data.get("cache_read_input_tokens", 0) or 0)' in src


def test_agent_deduct_request_has_optional_cached_tokens():
    from app.api.credits import AgentDeductRequest
    field = AgentDeductRequest.model_fields["cached_tokens"]
    assert field.default is None, "old agents omit the field — must be optional"


def test_agent_deduct_stores_cached_in_metadata_not_amount():
    """cached_tokens goes into metadata_json ONLY. The charge amount must
    still be computed from input/output alone (billing math untouched —
    A9-2/A9-3 out of scope)."""
    from app.api import credits as cr
    src = inspect.getsource(cr.agent_deduct)
    assert '_deduct_meta["cached_tokens"]' in src
    # The charge line is byte-identical to pre-change: tokens only.
    assert "credits = _tokens_to_credits(body.model, body.input_tokens, body.output_tokens)" in src


def test_calc_cost_cents_legacy_models_ignore_cached_tokens():
    """A9-2 (cost overstatement on cache hits) stays out of scope for the
    LIVE fleet: G1 prep added optional cached/cache-write kwargs to
    _calc_cost_cents, but they may only act on models whose pricing entry
    carries the cached_input/cache_write columns (gpt-5.6 family — dark).
    For every current model the math must stay byte-identical, cached or
    not."""
    sig = inspect.signature(lp._calc_cost_cents)
    assert list(sig.parameters) == [
        "model", "input_tokens", "output_tokens",
        "cached_tokens", "cache_write_tokens",
    ]
    assert sig.parameters["cached_tokens"].default == 0
    assert sig.parameters["cache_write_tokens"].default == 0
    for model in ("gpt-5.5", "gpt-4o", "gpt-4o-mini", "claude-opus-4-6",
                  "totally-unknown-model"):
        base = lp._calc_cost_cents(model, 27_200, 500)
        assert lp._calc_cost_cents(
            model, 27_200, 500, cached_tokens=22_144, cache_write_tokens=5_000
        ) == base, f"{model}: cache kwargs must be inert without cache pricing columns"


# ── 4. Admin rollup endpoint ─────────────────────────────────────────


def test_cache_daily_uses_require_admin_guard():
    """Guard copied from the admin/infrastructure endpoints: Depends(require_admin)
    → 401 unauthenticated (via get_current_user), 403 non-admin."""
    src = inspect.getsource(lp.get_admin_cache_daily)
    assert "Depends(require_admin)" in src
    from app.api.admin.deps import require_admin as _ra
    assert lp.require_admin is _ra


_CURRENT = {"user": None}


def _user(uid: str, role: str = "user"):
    return types.SimpleNamespace(id=uid, role=role, email="u@x.com")


@pytest_asyncio.fixture
async def admin_app_client():
    """Minimal app mounting only the llm admin router (mirrors
    test_support_grading's pattern). get_current_user is NOT overridden so
    the unauthenticated case exercises the real 401."""
    from app.config import settings
    from app.db import get_db, async_session_maker

    app = FastAPI()
    app.include_router(lp.admin_router, prefix=settings.api_prefix)

    async def _override_db():
        async with async_session_maker() as db:
            yield db

    app.dependency_overrides[get_db] = _override_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://t") as c:
        yield c, app


@pytest.mark.asyncio
async def test_cache_daily_unauthenticated_is_401_never_405(admin_app_client):
    client, _app = admin_app_client
    res = await client.get("/api/admin/llm/cache-daily")
    assert res.status_code == 401, res.text
    assert res.status_code != 405  # repo rule: route must be registered


@pytest.mark.asyncio
async def test_cache_daily_bad_days_is_422(admin_app_client):
    from app.api.auth import get_current_user
    client, app = admin_app_client
    app.dependency_overrides[get_current_user] = lambda: _user("admin-1", role="admin")
    res = await client.get("/api/admin/llm/cache-daily", params={"days": 0})
    assert res.status_code == 422, res.text


@pytest.mark.asyncio
async def test_cache_daily_admin_gets_rollup(admin_app_client):
    from app.api.auth import get_current_user
    from app.db import async_session_maker
    from app.db.models.platform import LLMProxyEvent

    client, app = admin_app_client
    app.dependency_overrides[get_current_user] = lambda: _user("admin-1", role="admin")

    async with async_session_maker() as db:
        db.add(LLMProxyEvent(
            user_id="cache-user-1", provider="openai", model="gpt-5.5",
            endpoint="chat", input_tokens=1000, output_tokens=50,
            cost_cents=3, cached_tokens=750,
        ))
        db.add(LLMProxyEvent(
            user_id="cache-user-1", provider="openai", model="gpt-5.5",
            endpoint="chat", input_tokens=1000, output_tokens=50,
            cost_cents=3, cached_tokens=None,  # pre-075 row → counts as 0
        ))
        await db.commit()

    res = await client.get(
        "/api/admin/llm/cache-daily",
        params={"days": 7, "user_id": "cache-user-1"},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["days"] == 7
    assert body["user_id"] == "cache-user-1"
    assert len(body["rows"]) == 1
    row = body["rows"][0]
    assert row["prompt_tokens"] == 2000
    assert row["cached_tokens"] == 750
    assert row["calls"] == 2
    assert row["cache_hit_ratio"] == pytest.approx(750 / 2000, abs=1e-4)


@pytest.mark.asyncio
async def test_cache_daily_non_admin_is_403(admin_app_client):
    from app.api.auth import get_current_user
    client, app = admin_app_client
    app.dependency_overrides[get_current_user] = lambda: _user("pleb-1", role="user")
    res = await client.get("/api/admin/llm/cache-daily")
    assert res.status_code == 403, res.text


def test_cached_tokens_coercion_never_raises():
    """Review pr5-#2: a truthy non-numeric provider value must coerce to
    0, not raise out of the stream path's narrow except and abort the
    _log_event write in stream_and_log's finally."""
    from app.api.llm_proxy import _extract_openai_cached_tokens
    assert _extract_openai_cached_tokens(
        {"prompt_tokens_details": {"cached_tokens": "not-a-number"}}
    ) == 0
    assert _extract_openai_cached_tokens(
        {"prompt_tokens_details": {"cached_tokens": ["weird"]}}
    ) == 0
    assert _extract_openai_cached_tokens(
        {"prompt_tokens_details": {"cached_tokens": "1536"}}
    ) == 1536  # numeric strings still count


def test_cache_daily_filters_chat_ok_by_default():
    """Review pr5-#1: voice/embeddings/error rows would dilute the
    chat-path hit ratio — the rollup defaults to endpoint=chat and
    status=ok, with endpoint=all as the escape hatch."""
    import inspect
    from app.api import llm_proxy as lp
    src = inspect.getsource(lp.get_admin_cache_daily)
    assert 'LLMProxyEvent.status == "ok"' in src
    assert 'endpoint != "all"' in src
    assert 'LLMProxyEvent.endpoint == endpoint' in src
