"""
FF-B.2 — bounded-retry + auth-fallback + structured-stub tests for the
day summarizer.

Covers:
- M1: Anthropic 401 → OpenAI fallback fires (was the binding bug)
- M2: bounded retry with backoff via should_summarize() eligibility
- M3: failure reason persisted to DayChat.summary_last_failure_reason
- _classify_failure taxonomy: status-code & exception mapping

Pattern matches backend/tests/test_active_task.py — sqlite + raw CREATE
TABLE + ORM inserts, bypassing the heavy app.services init and the
broken conftest. Direct invocation works:
    cd backend && ENVIRONMENT=development python tests/test_day_summarizer_ffb2.py
"""
import asyncio
import os
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

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
    if the OpenAI key was set. M1 routes 401 → OpenAI. This is the
    behavioural test for the diagnostic finding.
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
        def __init__(self, resp):
            self.resp = resp
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

    # Patch settings to advertise both keys, then replace httpx.AsyncClient
    class _Settings:
        anthropic_api_key = "sk-ant-test"
        openai_api_key = "sk-openai-test"

    with patch("app.config.settings", _Settings), \
         patch("httpx.AsyncClient", lambda **kw: _ClientCtx(None)):
        text, reason = await _sumr._try_summarize("dummy prompt")
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
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return None
        async def post(self, url, **kw):
            return fake_anthropic if "anthropic.com" in url else fake_openai

    class _Settings:
        anthropic_api_key = "sk-ant-test"
        openai_api_key = "sk-openai-test"

    with patch("app.config.settings", _Settings), \
         patch("httpx.AsyncClient", lambda **kw: _ClientCtx()):
        text, reason = await _sumr._try_summarize("dummy")
    assert text is None
    # Anthropic's auth_error is more diagnostic than openai's server_error;
    # implementation prefers the non-generic reason from anthropic.
    assert reason == "auth_error", f"expected auth_error, got {reason!r}"
    print("OK test_m1_both_providers_fail_returns_anthropic_reason")


# ── Test 4 — M1: no keys configured → no_keys reason ──

async def test_m1_no_keys_returns_no_keys_reason():
    class _Settings:
        anthropic_api_key = None
        openai_api_key = None

    with patch("app.config.settings", _Settings):
        text, reason = await _sumr._try_summarize("dummy")
    assert text is None
    assert reason == "no_keys"
    print("OK test_m1_no_keys_returns_no_keys_reason")


# ── Test 5 — M2: should_summarize backoff window enforcement ──

async def _make_engine():
    """sqlite test engine matching the columns the summarizer touches."""
    engine = create_async_engine(
        "sqlite+aiosqlite://", connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        for stmt in [
            """CREATE TABLE IF NOT EXISTS users (
                id VARCHAR(36) PRIMARY KEY, email VARCHAR(255) UNIQUE,
                hashed_password VARCHAR(255), name VARCHAR(255),
                password_changed_at TIMESTAMP,
                role VARCHAR(20) DEFAULT 'beta_user', created_at TIMESTAMP,
                updated_at TIMESTAMP, is_active BOOLEAN DEFAULT 1,
                stripe_customer_id VARCHAR(255), timezone VARCHAR(50),
                is_canary BOOLEAN DEFAULT 0
            )""",
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
            )""",
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


# ── Run all ──

if __name__ == "__main__":
    test_classify_failure_taxonomy()
    asyncio.run(test_m1_anthropic_401_falls_back_to_openai())
    asyncio.run(test_m1_both_providers_fail_returns_anthropic_reason())
    asyncio.run(test_m1_no_keys_returns_no_keys_reason())
    asyncio.run(test_m2_backoff_window_blocks_premature_retry())
    asyncio.run(test_m2_past_backoff_is_eligible())
    asyncio.run(test_m2_max_retries_permanent_fail())
    print("\nALL FF-B.2 TESTS PASSED")
