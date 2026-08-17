"""
FF-B.2 — bounded-retry + auth-fallback + structured-stub tests for the
day summarizer.

Covers:
- M1: Anthropic 401 → OpenAI fallback fires (was the binding bug).
  W0.4b moved the fallback into internal_llm.call_system_llm — the
  tests now exercise it through that layer (BYOK direct path).
- M2: bounded retry with backoff via should_summarize() eligibility
- M3: failure reason persisted to DayChat.summary_last_failure_reason
- _classify_failure taxonomy: status-code & exception mapping
- W0.4b: bundle tenants (no raw keys) summarize via the bundle client;
  failure reasons surface the real cause, never a bogus no_keys.

Pattern matches backend/tests/test_active_task.py — sqlite + raw CREATE
TABLE + ORM inserts, bypassing the heavy app.services init and the
broken conftest. Direct invocation works:
    cd backend && ENVIRONMENT=development python tests/test_day_summarizer_ffb2.py
"""
import asyncio
import os
import sys
import uuid
from contextlib import ExitStack, contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("ENVIRONMENT", "development")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from app.db.models.user import User
from app.db.models.day_chat import DayChat


# ── Helpers: import the summarizer module without pulling app.services init ──

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "_summarizer_under_test",
    str(Path(__file__).resolve().parent.parent / "app" / "services" / "day_summarizer.py"),
)
_sumr = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_sumr)


@contextmanager
def _patched_settings(**overrides):
    """Patch attributes on the REAL app.config.settings instance.

    internal_llm holds a module-level reference to that instance, so
    swapping app.config.settings for a stub (the pre-W0.4b pattern)
    would not reach it — attribute patching does.
    """
    from app.config import settings as real_settings

    with ExitStack() as stack:
        for key, value in overrides.items():
            stack.enter_context(patch.object(real_settings, key, value))
        # Never let a test write llm_proxy_events to a real DB.
        stack.enter_context(
            patch("app.services.internal_llm._log_system_event", new=AsyncMock())
        )
        yield


_BYOK = dict(
    llm_mode="manual",
    toup_token="",
    platform_anthropic_api_key=None,
    platform_openai_api_key=None,
)


def _no_network(**kw):
    raise AssertionError("httpx.AsyncClient must not be constructed in this test")


# ── Test 1 — _classify_failure: status code + exception taxonomy ──

def test_classify_failure_taxonomy():
    """The classifier maps real-world error inputs to the M3 reason set."""
    f = _sumr._classify_failure
    # Auth — the actual prod failure mode for the founder's tenant
    assert f(status_code=401) == "auth_error"
    assert f(status_code=403) == "auth_error"
    # Rate limit
    assert f(status_code=429) == "rate_limit"
    # Server-side
    assert f(status_code=500) == "server_error"
    assert f(status_code=502) == "server_error"
    assert f(status_code=503) == "server_error"
    assert f(status_code=599) == "server_error"
    # Other 4xx → other (we don't fan these out further)
    assert f(status_code=400) == "other"
    assert f(status_code=404) == "other"
    # Exceptions
    class _FakeTimeout(Exception):
        pass
    _FakeTimeout.__name__ = "ReadTimeout"
    assert f(exception=_FakeTimeout("x")) == "timeout"
    assert f(exception=RuntimeError("connection reset")) == "other"
    # Defensive: nothing supplied
    assert f() == "other"
    print("OK test_classify_failure_taxonomy")


# ── Test 2 — M1: Anthropic 401 falls back to OpenAI (the binding bug) ──

async def test_m1_anthropic_401_falls_back_to_openai():
    """
    The pre-FF-B.2 code returned None when Anthropic returned 401, even
    if the OpenAI key was set. M1 routes 401 → OpenAI. W0.4b: the
    fallback now happens inside call_system_llm's BYOK direct path —
    same observable behaviour through _try_summarize.
    """
    # Mock Anthropic 401 then OpenAI 200
    fake_anthropic_resp = AsyncMock()
    fake_anthropic_resp.status_code = 401
    fake_anthropic_resp.text = '{"error":{"type":"authentication_error"}}'
    fake_anthropic_resp.headers = {"x-request-id": "req_test_anthropic"}

    fake_openai_resp = AsyncMock()
    fake_openai_resp.status_code = 200
    fake_openai_resp.json = lambda: {"choices": [{"message": {"content": "fallback summary text"}}]}

    class _ClientCtx:
        def __init__(self, **kw):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            return None
        async def post(self, url, **kwargs):
            if "anthropic.com" in url:
                return fake_anthropic_resp
            if "openai.com" in url:
                return fake_openai_resp
            raise AssertionError(f"unexpected url: {url}")

    with _patched_settings(
        **_BYOK,
        anthropic_api_key="sk-ant-test",
        openai_api_key="sk-openai-test",
    ), patch("httpx.AsyncClient", _ClientCtx):
        text, reason = await _sumr._try_summarize("user-1", "dummy prompt")
    assert text == "fallback summary text", f"expected fallback summary, got {text!r}"
    assert reason == "", f"expected empty reason on success, got {reason!r}"
    print("OK test_m1_anthropic_401_falls_back_to_openai")


# ── Test 3 — M1: both providers fail → reason carries the proximate cause ──

async def test_m1_both_providers_fail_returns_anthropic_reason():
    """When both providers fail, the more-diagnostic reason wins.
    Anthropic 401 (auth_error) trumps OpenAI server_error generic 'other'.
    """
    fake_anthropic = AsyncMock()
    fake_anthropic.status_code = 401
    fake_anthropic.text = "auth"
    fake_anthropic.headers = {"x-request-id": "req_a"}

    fake_openai = AsyncMock()
    fake_openai.status_code = 500
    fake_openai.text = "internal"

    class _ClientCtx:
        def __init__(self, **kw):
            pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return None
        async def post(self, url, **kw):
            return fake_anthropic if "anthropic.com" in url else fake_openai

    with _patched_settings(
        **_BYOK,
        anthropic_api_key="sk-ant-test",
        openai_api_key="sk-openai-test",
    ), patch("httpx.AsyncClient", _ClientCtx):
        text, reason = await _sumr._try_summarize("user-1", "dummy")
    assert text is None
    # Anthropic's auth_error is more diagnostic than openai's server_error;
    # implementation prefers the non-generic reason from anthropic.
    assert reason == "auth_error", f"expected auth_error, got {reason!r}"
    print("OK test_m1_both_providers_fail_returns_anthropic_reason")


# ── Test 4 — M1: no keys configured (BYOK) → no_keys reason ──

async def test_m1_no_keys_returns_no_keys_reason():
    with _patched_settings(
        **_BYOK,
        anthropic_api_key=None,
        openai_api_key=None,
    ), patch("httpx.AsyncClient", _no_network):
        text, reason = await _sumr._try_summarize("user-1", "dummy")
    assert text is None
    assert reason == "no_keys"
    print("OK test_m1_no_keys_returns_no_keys_reason")


# ── Test 4b — W0.4b: bundle tenant without raw keys gets a summary ──

async def test_w04b_bundle_mode_summarizes_without_raw_keys():
    """The headline W0.4b fix: a bundle tenant has NO raw provider keys
    — pre-fix that meant summary=failed reason=no_keys forever. Now the
    call routes through bundle_client (platform proxy) and succeeds,
    never touching httpx directly."""
    block = MagicMock()
    block.text = "bundle summary"
    resp = MagicMock()
    resp.content = [block]
    resp.usage = MagicMock(input_tokens=100, output_tokens=50)
    fake_client = MagicMock()
    fake_client.messages.create = AsyncMock(return_value=resp)

    with _patched_settings(
        llm_mode="bundle",
        toup_token="toup_ct_test",
        anthropic_api_key=None,
        openai_api_key=None,
        platform_anthropic_api_key=None,
        platform_openai_api_key=None,
    ), patch("app.services.bundle_client.make_anthropic_client", return_value=fake_client), \
         patch("httpx.AsyncClient", _no_network):
        text, reason = await _sumr._try_summarize("user-1", "dummy prompt")

    assert text == "bundle summary", f"expected bundle summary, got {text!r}"
    assert reason == "", f"expected empty reason on success, got {reason!r}"
    kwargs = fake_client.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-haiku-4-5-20251001"  # haiku pin preserved
    assert kwargs["max_tokens"] == 1200  # cap preserved
    assert kwargs["system"] == _sumr.SUMMARIZER_SYSTEM_PROMPT
    print("OK test_w04b_bundle_mode_summarizes_without_raw_keys")


# ── Test 4c — W0.4b: bundle failure surfaces the REAL reason, not no_keys ──

async def test_w04b_bundle_failure_surfaces_real_reason():
    """When the bundle call fails (e.g. 429 from the proxy) and the
    OpenAI fallback can't run either, the persisted reason must be the
    diagnostic one (rate_limit) — not a misleading no_keys."""
    class _RateLimited(Exception):
        status_code = 429

    fake_client = MagicMock()
    fake_client.messages.create = AsyncMock(side_effect=_RateLimited("429 from proxy"))

    with _patched_settings(
        llm_mode="bundle",
        toup_token="toup_ct_test",
        anthropic_api_key=None,
        openai_api_key=None,
        platform_anthropic_api_key=None,
        platform_openai_api_key=None,
    ), patch("app.services.bundle_client.make_anthropic_client", return_value=fake_client), \
         patch("app.services.bundle_client.make_openai_client", return_value=None), \
         patch("httpx.AsyncClient", _no_network):
        text, reason = await _sumr._try_summarize("user-1", "dummy")

    assert text is None
    assert reason == "rate_limit", f"expected rate_limit, got {reason!r}"
    print("OK test_w04b_bundle_failure_surfaces_real_reason")


# ── Test 5 — M2: should_summarize backoff window enforcement ──

async def _make_engine():
    """sqlite test engine matching the columns the summarizer touches."""
    engine = create_async_engine(
        "sqlite+aiosqlite://", connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        # `users` comes FROM THE ORM MODEL, never from a copy of it — the copy
        # that used to live here had already drifted (it was missing
        # `notification_preferences`) and went red the moment the model gained
        # `first_media_played_at` (migration 086). A hand-written schema is a
        # second source of truth nothing keeps in sync. Same fix as
        # test_reply_history.py and the others that hit this before.
        await conn.run_sync(User.__table__.create, checkfirst=True)
        for stmt in [
            """CREATE TABLE IF NOT EXISTS day_chats (
                id VARCHAR(36) PRIMARY KEY, user_id VARCHAR(36),
                local_date DATE NOT NULL, timezone VARCHAR(50) DEFAULT 'UTC',
                started_at TIMESTAMP, last_message_at TIMESTAMP,
                message_count INTEGER DEFAULT 0, total_tokens INTEGER DEFAULT 0,
                rolling_summary TEXT, summary_up_to_message_id VARCHAR(50),
                summary_updated_at TIMESTAMP,
                summary_status VARCHAR(20) DEFAULT 'up_to_date',
                archival_summary TEXT,
                archival_summary_generated_at TIMESTAMP,
                archival_summary_status VARCHAR(20) NOT NULL DEFAULT 'not_needed',
                summary_failure_count INTEGER NOT NULL DEFAULT 0,
                summary_last_failure_at TIMESTAMP,
                summary_last_failure_reason VARCHAR(50)
            )""",
            """CREATE TABLE IF NOT EXISTS conversations (
                id VARCHAR(36) PRIMARY KEY, user_id VARCHAR(36),
                day_chat_id VARCHAR(36), channel VARCHAR(50),
                started_at TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS messages (
                id VARCHAR(50) PRIMARY KEY, conversation_id VARCHAR(36),
                day_chat_id VARCHAR(36), role VARCHAR(20), content TEXT,
                created_at TIMESTAMP
            , origin VARCHAR(16))""",
        ]:
            await conn.run_sync(lambda c, s=stmt: c.execute(text(s)))
    return engine


async def test_m2_backoff_window_blocks_premature_retry():
    """A failed day inside its backoff window is NOT eligible for retry."""
    engine = await _make_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    user_id = str(uuid.uuid4())
    dc_id = str(uuid.uuid4())

    async with sm() as db:
        db.add(User(
            id=user_id, email="t@t.local", hashed_password="x",
            name="Test", role="beta_user", is_active=True, is_canary=False,
            created_at=datetime.utcnow(), updated_at=datetime.utcnow(),
        ))
        from datetime import date as _date
        # Insert 11 messages so the message-count gate passes
        from sqlalchemy import text as _text
        await db.execute(_text("""
            INSERT INTO day_chats (id, user_id, local_date, summary_status,
                summary_failure_count, summary_last_failure_at,
                summary_last_failure_reason)
            VALUES (:id, :uid, :d, 'failed', 1, :recent, 'auth_error')
        """), {
            "id": dc_id, "uid": user_id, "d": _date.today(),
            "recent": datetime.utcnow() - timedelta(seconds=60),  # well inside 1h backoff
        })
        await db.execute(_text("""
            INSERT INTO conversations (id, user_id, day_chat_id, channel, started_at)
            VALUES (:cid, :uid, :did, 'web', :now)
        """), {"cid": str(uuid.uuid4()), "uid": user_id, "did": dc_id, "now": datetime.utcnow()})
        for i in range(11):
            await db.execute(_text("""
                INSERT INTO messages (id, conversation_id, day_chat_id, role, content, created_at)
                VALUES (:id, (SELECT id FROM conversations WHERE day_chat_id=:did LIMIT 1),
                        :did, 'user', 'hi', :now)
            """), {"id": f"m-{i}", "did": dc_id, "now": datetime.utcnow()})
        await db.commit()

    async with sm() as db:
        eligible = await _sumr.should_summarize(db, dc_id)
    assert eligible is False, "inside backoff window should NOT be eligible"
    await engine.dispose()
    print("OK test_m2_backoff_window_blocks_premature_retry")


async def test_m2_past_backoff_is_eligible():
    """Same setup, but failure_at moved beyond the 1h backoff → eligible."""
    engine = await _make_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    user_id = str(uuid.uuid4())
    dc_id = str(uuid.uuid4())

    async with sm() as db:
        db.add(User(
            id=user_id, email="t@t.local", hashed_password="x",
            name="Test", role="beta_user", is_active=True, is_canary=False,
            created_at=datetime.utcnow(), updated_at=datetime.utcnow(),
        ))
        from datetime import date as _date
        from sqlalchemy import text as _text
        await db.execute(_text("""
            INSERT INTO day_chats (id, user_id, local_date, summary_status,
                summary_failure_count, summary_last_failure_at,
                summary_last_failure_reason)
            VALUES (:id, :uid, :d, 'failed', 1, :old, 'auth_error')
        """), {
            "id": dc_id, "uid": user_id, "d": _date.today(),
            "old": datetime.utcnow() - timedelta(hours=2),  # past 1h backoff for attempt 1
        })
        await db.execute(_text("""
            INSERT INTO conversations (id, user_id, day_chat_id, channel, started_at)
            VALUES (:cid, :uid, :did, 'web', :now)
        """), {"cid": str(uuid.uuid4()), "uid": user_id, "did": dc_id, "now": datetime.utcnow()})
        for i in range(11):
            await db.execute(_text("""
                INSERT INTO messages (id, conversation_id, day_chat_id, role, content, created_at)
                VALUES (:id, (SELECT id FROM conversations WHERE day_chat_id=:did LIMIT 1),
                        :did, 'user', 'hi', :now)
            """), {"id": f"m-{i}", "did": dc_id, "now": datetime.utcnow()})
        await db.commit()

    async with sm() as db:
        eligible = await _sumr.should_summarize(db, dc_id)
    assert eligible is True, "past backoff window should be eligible"
    await engine.dispose()
    print("OK test_m2_past_backoff_is_eligible")


async def test_m2_max_retries_permanent_fail():
    """failure_count >= MAX_RETRIES (default 3) → permanently ineligible."""
    engine = await _make_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    user_id = str(uuid.uuid4())
    dc_id = str(uuid.uuid4())

    async with sm() as db:
        db.add(User(
            id=user_id, email="t@t.local", hashed_password="x",
            name="Test", role="beta_user", is_active=True, is_canary=False,
            created_at=datetime.utcnow(), updated_at=datetime.utcnow(),
        ))
        from datetime import date as _date
        from sqlalchemy import text as _text
        await db.execute(_text("""
            INSERT INTO day_chats (id, user_id, local_date, summary_status,
                summary_failure_count, summary_last_failure_at,
                summary_last_failure_reason)
            VALUES (:id, :uid, :d, 'failed', :n, :ago, 'auth_error')
        """), {
            "id": dc_id, "uid": user_id, "d": _date.today(),
            "n": _sumr.MAX_RETRIES,
            "ago": datetime.utcnow() - timedelta(days=10),  # well past any backoff
        })
        await db.commit()

    async with sm() as db:
        eligible = await _sumr.should_summarize(db, dc_id)
    assert eligible is False, "MAX_RETRIES exceeded should be permanently ineligible"
    await engine.dispose()
    print("OK test_m2_max_retries_permanent_fail")


# ── W0.4b — both LLM entry points route through call_system_llm ──
#
# ORM-created sqlite tables (full column set) because generate_summary /
# generate_archival_summary select the whole Message entity, not just
# the columns the raw CREATE TABLE helper above covers.

async def _make_orm_engine():
    from app.db.models.user import User
    from app.db.models.day_chat import DayChat
    from app.db.models import Message, Conversation

    engine = create_async_engine(
        "sqlite+aiosqlite://", connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(
            lambda c: Message.metadata.create_all(
                c,
                tables=[
                    User.__table__, DayChat.__table__,
                    Conversation.__table__, Message.__table__,
                ],
            )
        )
    return engine


async def _seed_day(sm, n_messages: int = 3):
    """Insert user + day_chat + conversation + n messages. Returns (user_id, dc_id)."""
    from datetime import date as _date
    from app.db.models import Message, Conversation

    user_id = str(uuid.uuid4())
    dc_id = str(uuid.uuid4())
    async with sm() as db:
        db.add(User(
            id=user_id, email=f"{user_id[:8]}@t.local", hashed_password="x",
            name="Test", role="beta_user", is_active=True,
            created_at=datetime.utcnow(), updated_at=datetime.utcnow(),
        ))
        db.add(DayChat(id=dc_id, user_id=user_id, local_date=_date.today()))
        conv_id = str(uuid.uuid4())
        db.add(Conversation(id=conv_id, user_id=user_id, day_chat_id=dc_id, channel="web"))
        for i in range(n_messages):
            db.add(Message(
                id=f"m-{dc_id[:8]}-{i}", conversation_id=conv_id, day_chat_id=dc_id,
                role="user" if i % 2 == 0 else "assistant",
                content=f"message number {i}",
                created_at=datetime.utcnow() + timedelta(seconds=i),
            ))
        await db.commit()
    return user_id, dc_id


async def test_w04b_generate_summary_uses_metered_path():
    """generate_summary hands the prompt to call_system_llm with the
    day-summary operation tag + haiku pin, and passes the day owner's
    user_id so metering lands on the right tenant."""
    engine = await _make_orm_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    user_id, dc_id = await _seed_day(sm)

    mock_llm = AsyncMock(return_value="rolling summary text")
    with patch("app.services.internal_llm.call_system_llm", mock_llm):
        async with sm() as db:
            text, reason = await _sumr.generate_summary(db, dc_id)

    assert text == "rolling summary text"
    assert reason == ""
    mock_llm.assert_called_once()
    kwargs = mock_llm.call_args.kwargs
    assert kwargs["user_id"] == user_id
    assert kwargs["operation_type"] == "system.day_summary"
    assert kwargs["model"] == "claude-haiku-4-5-20251001"
    assert kwargs["max_tokens"] == 1200
    assert "message number 0" in kwargs["messages"][0]["content"]
    await engine.dispose()
    print("OK test_w04b_generate_summary_uses_metered_path")


async def test_w04b_generate_summary_failure_reason_propagates():
    """When the metered call fails, generate_summary surfaces the reason
    from failure_out — run_summarizer_if_needed persists it verbatim."""
    engine = await _make_orm_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    _, dc_id = await _seed_day(sm)

    async def _failing_llm(*, failure_out=None, **kwargs):
        if failure_out is not None:
            failure_out["reason"] = "server_error"
        return None

    with patch("app.services.internal_llm.call_system_llm", _failing_llm):
        async with sm() as db:
            text, reason = await _sumr.generate_summary(db, dc_id)

    assert text is None
    assert reason == "server_error", f"expected server_error, got {reason!r}"
    await engine.dispose()
    print("OK test_w04b_generate_summary_failure_reason_propagates")


async def test_w04b_archival_routes_through_call_system_llm():
    """The archival path gets the same treatment — unified metered call,
    no legacy per-provider helpers."""
    engine = await _make_orm_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    user_id, dc_id = await _seed_day(sm)

    mock_llm = AsyncMock(return_value="archival text")
    with patch("app.services.internal_llm.call_system_llm", mock_llm):
        async with sm() as db:
            text = await _sumr.generate_archival_summary(db, dc_id)

    assert text == "archival text"
    mock_llm.assert_called_once()
    kwargs = mock_llm.call_args.kwargs
    assert kwargs["user_id"] == user_id
    assert kwargs["operation_type"] == "system.day_archival"
    assert kwargs["model"] == "claude-haiku-4-5-20251001"
    assert kwargs["max_tokens"] == 1800
    await engine.dispose()
    print("OK test_w04b_archival_routes_through_call_system_llm")



# ── Connection lifetime: the pooled session must NOT be held across the LLM ──

async def _assert_released_before_llm(fn_name: str, patched: str):
    """Shared body: the session must have no open transaction when the LLM
    call is made.

    Holding a pooled connection across a multi-second LLM round-trip pins it
    idle-in-transaction; and because run_summarizer_if_needed is invoked
    fire-and-forget (asyncio.create_task) on a path that cancels routinely
    (voice turns), a cancellation mid-call leaked the connection outright.
    The GC then terminated it ("non-checked-in connection ... will be
    terminated"), the pool degraded, and later turns died on
    PendingRollbackError -> HTTP 500 from /internal/agent-turn. Measured on
    the canary 2026-08-01: turn latency 14s -> 148s, then every turn 500'd.
    Same defect class as 6d173563 (support agent).
    """
    engine = await _make_orm_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    _, dc_id = await _seed_day(sm)

    holder = {}
    seen = {}

    async def _probe(**kwargs):
        # in_transaction() is sync on AsyncSession and reports whether a
        # connection is currently checked out for this session.
        seen["in_transaction"] = holder["db"].in_transaction()
        return "summary text"

    with patch(patched, _probe):
        async with sm() as db:
            holder["db"] = db
            await getattr(_sumr, fn_name)(db, dc_id)

    assert "in_transaction" in seen, f"{fn_name}: LLM was never called"
    assert seen["in_transaction"] is False, (
        f"{fn_name} held the pooled connection across the LLM call "
        "(idle-in-transaction; leaks outright if the task is cancelled)"
    )
    await engine.dispose()
    print(f"OK connection released before LLM in {fn_name}")


async def test_generate_summary_releases_connection_before_llm():
    await _assert_released_before_llm(
        "generate_summary", "app.services.internal_llm.call_system_llm")


async def test_archival_releases_connection_before_llm():
    await _assert_released_before_llm(
        "generate_archival_summary", "app.services.internal_llm.call_system_llm")

# ── Run all ──

if __name__ == "__main__":
    test_classify_failure_taxonomy()
    asyncio.run(test_m1_anthropic_401_falls_back_to_openai())
    asyncio.run(test_m1_both_providers_fail_returns_anthropic_reason())
    asyncio.run(test_m1_no_keys_returns_no_keys_reason())
    asyncio.run(test_w04b_bundle_mode_summarizes_without_raw_keys())
    asyncio.run(test_w04b_bundle_failure_surfaces_real_reason())
    asyncio.run(test_m2_backoff_window_blocks_premature_retry())
    asyncio.run(test_m2_past_backoff_is_eligible())
    asyncio.run(test_m2_max_retries_permanent_fail())
    asyncio.run(test_w04b_generate_summary_uses_metered_path())
    asyncio.run(test_w04b_generate_summary_failure_reason_propagates())
    asyncio.run(test_w04b_archival_routes_through_call_system_llm())
    asyncio.run(test_generate_summary_releases_connection_before_llm())
    asyncio.run(test_archival_releases_connection_before_llm())
    print("\nALL FF-B.2 TESTS PASSED")
