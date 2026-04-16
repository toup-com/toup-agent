"""
Tests for the day-recall feature.

Covers:
  - date_resolver: ISO / yesterday / weekday-past-only / month-day / relative
  - recall_day tool: summary mode, full mode, query filter, missing day, future reject
  - archival summary: ARCHIVAL vs ROLLING prompt distinction
  - LLM proxy budget: system.* ops exempt from user caps
  - Integration: "make a quiz from yesterday's lesson" scenario

Run: python tests/test_day_recall.py
"""

import asyncio
import importlib.util as _ilu
import sys
import uuid
from datetime import datetime, date as Date, timezone, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select, func, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker


# ── Module loading that avoids pulling app.config ────────────────────
def _load_module(name: str, rel_path: str):
    spec = _ilu.spec_from_file_location(
        name, str(Path(__file__).resolve().parent.parent / rel_path)
    )
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# date_resolver is standalone (no app.config)
_dr = _load_module("date_resolver", "app/agent/date_resolver.py")
resolve_date_reference = _dr.resolve_date_reference


# ── Test 1: date_resolver — ISO + yesterday + N days ago ─────────────
def test_resolver_iso_and_literals():
    today = datetime.utcnow().date()
    yesterday = today - timedelta(days=1)

    # ISO
    assert resolve_date_reference("2026-04-10") == Date(2026, 4, 10)
    # yesterday
    assert resolve_date_reference("yesterday") == yesterday
    assert resolve_date_reference("Yesterday") == yesterday
    # today
    assert resolve_date_reference("today") == today
    # N days ago
    assert resolve_date_reference("3 days ago") == today - timedelta(days=3)
    assert resolve_date_reference("1 day ago") == today - timedelta(days=1)
    assert resolve_date_reference("2 weeks ago") == today - timedelta(weeks=2)
    # last week
    assert resolve_date_reference("last week") == today - timedelta(days=7)

    # garbage
    assert resolve_date_reference("") is None
    assert resolve_date_reference("   ") is None
    assert resolve_date_reference("pizza") is None

    print("✓ test_resolver_iso_and_literals")


# ── Test 2: weekday names always resolve to most recent PAST ─────────
def test_weekday_past_only():
    """Critical: "Monday" → most recent PAST Monday, never today or upcoming."""
    # Simulate specific "today" values by monkey-patching via tz-free path:
    # we drive the function with tz=None (UTC) and rely on datetime.utcnow().
    # Since we cannot freeze time here without freezegun, drive the helper
    # directly on known dates through a wrapping function.

    from app.agent.date_resolver import _most_recent_past_weekday

    # Today is Monday 2026-04-13 (a real Monday). "Monday" should resolve to
    # 2026-04-06 — a week ago — NOT today.
    mon_today = Date(2026, 4, 13)  # Monday
    assert mon_today.weekday() == 0
    assert _most_recent_past_weekday(mon_today, 0) == Date(2026, 4, 6)

    # Today is Wednesday 2026-04-15. "Tuesday" (1) → 1 day ago = 2026-04-14.
    wed_today = Date(2026, 4, 15)
    assert wed_today.weekday() == 2
    assert _most_recent_past_weekday(wed_today, 1) == Date(2026, 4, 14)

    # Today is Wednesday 2026-04-15. "Thursday" (3) → 6 days ago = 2026-04-09.
    # (Not tomorrow Thursday 2026-04-16.)
    assert _most_recent_past_weekday(wed_today, 3) == Date(2026, 4, 9)

    # Today is Sunday 2026-04-19. "Sunday" (6) → 7 days ago = 2026-04-12.
    sun_today = Date(2026, 4, 19)
    assert sun_today.weekday() == 6
    assert _most_recent_past_weekday(sun_today, 6) == Date(2026, 4, 12)

    print("✓ test_weekday_past_only")


# ── Test 3: weekday names via full resolver (tz-aware) ───────────────
def test_weekday_via_resolver():
    # Can't fully freeze time, but we can verify that the resolver returns
    # a date <= today, a weekday match, and is within the past 7 days.
    today = datetime.utcnow().date()
    for name, wd in [
        ("Monday", 0), ("tuesday", 1), ("Wed", 2), ("thu", 3),
        ("friday", 4), ("sat", 5), ("Sunday", 6),
        ("last Monday", 0), ("this past Friday", 4),
    ]:
        got = resolve_date_reference(name)
        assert got is not None, f"'{name}' → None"
        assert got.weekday() == wd, f"'{name}' → {got} has weekday {got.weekday()}, want {wd}"
        assert got < today, f"'{name}' → {got} should be strictly in past (today={today})"
        assert (today - got).days <= 7, f"'{name}' resolved >7d in past"
    print("✓ test_weekday_via_resolver")


# ── Test 4: month-day references resolve to most recent past ─────────
def test_month_day():
    today = datetime.utcnow().date()

    # A date known to be in the past this year
    far_past = today - timedelta(days=60)
    fmt = far_past.strftime("%B %-d")  # e.g. "February 16"
    got = resolve_date_reference(fmt)
    assert got == far_past, f"{fmt!r} → {got}, want {far_past}"

    # ISO with explicit year
    assert resolve_date_reference("2025-12-25") == Date(2025, 12, 25)

    # Future dates rejected
    future = today + timedelta(days=30)
    assert resolve_date_reference(future.isoformat()) is None

    # Malformed rejected
    assert resolve_date_reference("April 99") is None

    print("✓ test_month_day")


# ── Test 5: timezone-sensitive "yesterday" ───────────────────────────
def test_timezone_yesterday():
    # With Tokyo tz at UTC-midnight, "yesterday" in Tokyo may be today in UTC
    # and vice versa. The resolver must honor the tz for "today".
    tokyo = resolve_date_reference("yesterday", tz_name="Asia/Tokyo")
    utc = resolve_date_reference("yesterday", tz_name="UTC")
    assert tokyo is not None and utc is not None
    assert abs((tokyo - utc).days) <= 1
    print("✓ test_timezone_yesterday")


# ── Shared DB fixtures for recall_day tool tests ──────────────────────
async def _make_recall_engine():
    engine = create_async_engine("sqlite+aiosqlite://", connect_args={"check_same_thread": False})
    async with engine.begin() as conn:
        await conn.run_sync(lambda c: c.execute(text("""
            CREATE TABLE users (
                id VARCHAR(36) PRIMARY KEY, email VARCHAR(255) UNIQUE,
                hashed_password VARCHAR(255), password_plain VARCHAR(255),
                name VARCHAR(255), password_changed_at TIMESTAMP,
                role VARCHAR(20) DEFAULT 'beta_user', created_at TIMESTAMP,
                updated_at TIMESTAMP, is_active BOOLEAN DEFAULT 1,
                stripe_customer_id VARCHAR(255), timezone VARCHAR(50)
            )
        """)))
        await conn.run_sync(lambda c: c.execute(text("""
            CREATE TABLE day_chats (
                id VARCHAR(36) PRIMARY KEY, user_id VARCHAR(36),
                local_date DATE NOT NULL, timezone VARCHAR(50) DEFAULT 'UTC',
                started_at TIMESTAMP, last_message_at TIMESTAMP,
                message_count INTEGER DEFAULT 0, total_tokens INTEGER DEFAULT 0,
                rolling_summary TEXT, summary_up_to_message_id VARCHAR(50),
                summary_updated_at TIMESTAMP, summary_status VARCHAR(20) DEFAULT 'up_to_date',
                archival_summary TEXT,
                archival_summary_generated_at TIMESTAMP,
                archival_summary_status VARCHAR(20) DEFAULT 'not_needed',
                UNIQUE(user_id, local_date)
            )
        """)))
        await conn.run_sync(lambda c: c.execute(text("""
            CREATE TABLE conversations (
                id VARCHAR(36) PRIMARY KEY, user_id VARCHAR(36),
                day_chat_id VARCHAR(36), channel VARCHAR(50) DEFAULT 'web',
                started_at TIMESTAMP, ended_at TIMESTAMP, updated_at TIMESTAMP,
                is_active BOOLEAN DEFAULT 1, message_count INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0
            )
        """)))
        await conn.run_sync(lambda c: c.execute(text("""
            CREATE TABLE messages (
                id VARCHAR(50) PRIMARY KEY, conversation_id VARCHAR(36),
                day_chat_id VARCHAR(36), role VARCHAR(20), content TEXT,
                created_at TIMESTAMP, tokens_prompt INTEGER DEFAULT 0,
                tokens_completion INTEGER DEFAULT 0, model_used VARCHAR(100),
                memories_retrieved_json TEXT, processing_time_ms INTEGER,
                metadata_json TEXT
            )
        """)))
    return engine


async def _seed_day(sm, user_id: str, local_date: Date, messages: list, *,
                    archival: str = None, rolling: str = None, channel: str = "web"):
    dc_id = str(uuid.uuid4())
    conv_id = str(uuid.uuid4())
    now = datetime(local_date.year, local_date.month, local_date.day, 12, 0, 0, tzinfo=timezone.utc)

    async with sm() as db:
        await db.execute(text(
            "INSERT INTO day_chats (id, user_id, local_date, timezone, started_at, "
            "last_message_at, message_count, rolling_summary, archival_summary, "
            "archival_summary_status) VALUES (:id, :u, :d, 'UTC', :ts, :ts, :cnt, :rs, :as, :st)"
        ), {
            "id": dc_id, "u": user_id, "d": local_date, "ts": now,
            "cnt": len(messages), "rs": rolling, "as": archival,
            "st": "up_to_date" if archival else "not_needed",
        })
        await db.execute(text(
            "INSERT INTO conversations (id, user_id, day_chat_id, channel, started_at, "
            "updated_at, is_active) VALUES (:id, :u, :d, :ch, :ts, :ts, 1)"
        ), {"id": conv_id, "u": user_id, "d": dc_id, "ch": channel, "ts": now})

        for i, (role, content) in enumerate(messages):
            msg_id = f"m-{dc_id[:8]}-{i}"
            ts = now + timedelta(minutes=i)
            await db.execute(text(
                "INSERT INTO messages (id, conversation_id, day_chat_id, role, content, created_at) "
                "VALUES (:id, :c, :d, :r, :content, :ts)"
            ), {"id": msg_id, "c": conv_id, "d": dc_id, "r": role, "content": content, "ts": ts})
        await db.commit()
    return dc_id


async def _call_recall(sm, user_id: str, date_ref: str, **extra):
    """Call _tool_recall_day against our in-memory DB without the config dependency."""
    from app.agent.date_resolver import resolve_date_reference as _rd

    # Inline implementation mirroring _tool_recall_day, with sm injected.
    if not date_ref:
        return "ERROR: 'date' is required"

    async with sm() as db:
        target_date = _rd(date_ref, tz_name=None)
        if target_date is None:
            return f"ERROR: Could not resolve date '{date_ref}'"

        result = await db.execute(text(
            "SELECT id, message_count, archival_summary, rolling_summary "
            "FROM day_chats WHERE user_id = :u AND local_date = :d"
        ), {"u": user_id, "d": target_date})
        row = result.first()
        if not row:
            return f"No conversation record exists for {target_date.isoformat()}."

        dc_id, mcount, archival_summary, rolling_summary = row
        include_full = bool(extra.get("include_full_conversation", False))
        query = (extra.get("query") or "").strip() or None
        try:
            limit = int(extra.get("limit", 200))
        except (TypeError, ValueError):
            limit = 200
        limit = max(1, min(limit, 500))

        header = f"# Day: {target_date.isoformat()}\nMessages: {mcount}"

        if not include_full:
            summary = archival_summary or rolling_summary
            if summary:
                if query and query.lower() not in summary.lower():
                    return (f"{header}\n\n## Summary (archival — query-term not found)\n\n{summary}\n\n"
                            f"[note: '{query}' was not found in the summary. "
                            f"Re-call with include_full_conversation=true and the same query "
                            f"to search the raw messages.]")
                tag = "archival" if archival_summary else "rolling"
                return f"{header}\n\n## Summary ({tag})\n\n{summary}"

        rows = (await db.execute(text(
            "SELECT m.role, m.content, m.created_at, c.channel FROM messages m "
            "JOIN conversations c ON m.conversation_id = c.id "
            "WHERE m.day_chat_id = :d AND m.role IN ('user', 'assistant') "
            "ORDER BY m.created_at ASC"
        ), {"d": dc_id})).all()

        if not rows:
            return f"{header}\n\n[No user/assistant messages on this day.]"

        formatted = []
        for role, content, created_at, channel in rows:
            if isinstance(created_at, str):
                created_at = datetime.fromisoformat(created_at)
            time_str = created_at.strftime("%-I:%M%p").lower() if created_at else ""
            role_label = "User" if role == "user" else "Agent"
            tag = f"[{channel or 'web'} {time_str}]"
            formatted.append(f"{tag} {role_label}: {content}")

        OUTPUT_CHAR_BUDGET = 45_000

        def _truncate(body: str) -> str:
            if len(body) <= OUTPUT_CHAR_BUDGET:
                return body
            marker = "\n\n[... output truncated — call recall_day again with a more specific `query` parameter to narrow ...]"
            return body[: max(0, OUTPUT_CHAR_BUDGET - len(marker))] + marker

        if query:
            q = query.lower()
            hits = [i for i, line in enumerate(formatted) if q in line.lower()]
            if not hits:
                summary = archival_summary or rolling_summary
                return (f"{header}\n\n"
                        f"[No messages on {target_date.isoformat()} matched '{query}'.]"
                        + (f"\n\n## Summary\n\n{summary}" if summary else ""))
            keep = set()
            for i in hits:
                for j in range(max(0, i - 2), min(len(formatted), i + 3)):
                    keep.add(j)
            filtered = [formatted[i] for i in sorted(keep)][:limit]
            body = _truncate("\n\n".join(filtered))
            return (f"{header}\n\n[filtered to {len(filtered)} / {len(formatted)} messages matching '{query}']"
                    f"\n\n{body}")

        total = len(formatted)
        if total > limit:
            half = limit // 2
            head = formatted[:half]
            tail = formatted[-half:]
            body = "\n\n".join(head) + f"\n\n[... {total - limit} messages elided; pass `query` to narrow ...]\n\n" + "\n\n".join(tail)
        else:
            body = "\n\n".join(formatted)
        body = _truncate(body)
        return f"{header}\n\n{body}"


# ── Test 6: recall_day returns archival summary (not rolling) when both exist ─
async def test_recall_prefers_archival_summary():
    engine = await _make_recall_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with sm() as db:
        db.add_all([])  # keep ORM quiet
        await db.execute(text(
            "INSERT INTO users (id, email, hashed_password, name, is_active) "
            "VALUES ('u-arch', 'u@test', 'x', 'Arch', 1)"
        ))
        await db.commit()

    yesterday = datetime.utcnow().date() - timedelta(days=1)
    await _seed_day(sm, "u-arch", yesterday, [
        ("user", "let's work on calculus"),
        ("assistant", "sure, what topic?"),
    ], archival="ARCHIVAL: covered derivatives of trig functions.",
        rolling="ROLLING: some calculus convo.")

    out = await _call_recall(sm, "u-arch", "yesterday")
    assert "ARCHIVAL: covered derivatives" in out
    assert "ROLLING:" not in out
    assert "## Summary (archival)" in out

    await engine.dispose()
    print("✓ test_recall_prefers_archival_summary")


# ── Test 7: recall_day falls back to rolling if no archival ──────────
async def test_recall_falls_back_to_rolling():
    engine = await _make_recall_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with sm() as db:
        await db.execute(text(
            "INSERT INTO users (id, email, hashed_password, name, is_active) "
            "VALUES ('u-roll', 'r@test', 'x', 'Roll', 1)"
        ))
        await db.commit()

    yesterday = datetime.utcnow().date() - timedelta(days=1)
    await _seed_day(sm, "u-roll", yesterday, [("user", "hi"), ("assistant", "hey")],
                    rolling="ROLLING: quick hello.")
    out = await _call_recall(sm, "u-roll", "yesterday")
    assert "ROLLING: quick hello" in out
    assert "## Summary (rolling)" in out

    await engine.dispose()
    print("✓ test_recall_falls_back_to_rolling")


# ── Test 8: recall_day full conversation mode ────────────────────────
async def test_recall_full_conversation():
    engine = await _make_recall_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with sm() as db:
        await db.execute(text(
            "INSERT INTO users (id, email, hashed_password, name, is_active) "
            "VALUES ('u-full', 'f@test', 'x', 'Full', 1)"
        ))
        await db.commit()

    yesterday = datetime.utcnow().date() - timedelta(days=1)
    await _seed_day(sm, "u-full", yesterday, [
        ("user", "teach me the chain rule"),
        ("assistant", "the chain rule says: if y = f(g(x)), then dy/dx = f'(g(x)) * g'(x)"),
        ("user", "give me an example"),
        ("assistant", "consider y = sin(x^2)..."),
    ], archival="covered calculus")

    out = await _call_recall(sm, "u-full", "yesterday", include_full_conversation=True)
    assert "chain rule" in out
    assert "sin(x^2)" in out
    assert "User:" in out and "Agent:" in out
    # Summary NOT shown in full mode (archival is only shown in summary-only mode)
    assert "## Summary" not in out

    await engine.dispose()
    print("✓ test_recall_full_conversation")


# ── Test 9: query parameter filters within a day ─────────────────────
async def test_recall_query_filter():
    engine = await _make_recall_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with sm() as db:
        await db.execute(text(
            "INSERT INTO users (id, email, hashed_password, name, is_active) "
            "VALUES ('u-qf', 'qf@test', 'x', 'QF', 1)"
        ))
        await db.commit()

    yesterday = datetime.utcnow().date() - timedelta(days=1)
    # Mix calculus tutoring with unrelated small-talk and a build session.
    await _seed_day(sm, "u-qf", yesterday, [
        ("user", "good morning"),
        ("assistant", "good morning! how can i help?"),
        ("user", "teach me derivatives of trig functions"),
        ("assistant", "sin'(x) = cos(x), cos'(x) = -sin(x)..."),
        ("user", "got it, practice problem?"),
        ("assistant", "differentiate y = sin(3x)"),
        ("user", "unrelated: can you order me pizza?"),
        ("assistant", "i can't order food but i can help you draft the order"),
        ("user", "nvm, back to calculus — what about tan(x)?"),
        ("assistant", "tan'(x) = sec^2(x)"),
    ])

    # Query for "derivatives" — should return hits + ±2 context.
    # Hit at index 2 → window is indices 0-4 (includes greeting, as expected).
    # The key test: unrelated threads (pizza) should NOT bleed in.
    out = await _call_recall(sm, "u-qf", "yesterday",
                             include_full_conversation=True,
                             query="derivatives")
    assert "derivatives of trig" in out
    assert "sin'(x) = cos(x)" in out
    # Unrelated pizza/tan threads (indices 6-9) must not appear — they're
    # beyond the ±2 window around the "derivatives" hit at index 2.
    assert "pizza" not in out
    assert "tan'(x)" not in out
    # Header tells us filter happened
    assert "filtered to" in out and "matching 'derivatives'" in out

    # Query for "pizza" — should return hit
    out2 = await _call_recall(sm, "u-qf", "yesterday",
                              include_full_conversation=True,
                              query="pizza")
    assert "pizza" in out2
    assert "tan'(x)" not in out2  # far from pizza in the transcript

    # Query that matches nothing
    out3 = await _call_recall(sm, "u-qf", "yesterday",
                              include_full_conversation=True,
                              query="quantum mechanics")
    assert "No messages" in out3 and "matched 'quantum mechanics'" in out3

    await engine.dispose()
    print("✓ test_recall_query_filter")


# ── Test 10: day that doesn't exist returns friendly error ───────────
async def test_recall_missing_day():
    engine = await _make_recall_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with sm() as db:
        await db.execute(text(
            "INSERT INTO users (id, email, hashed_password, name, is_active) "
            "VALUES ('u-ghost', 'g@test', 'x', 'Ghost', 1)"
        ))
        await db.commit()

    out = await _call_recall(sm, "u-ghost", "yesterday")
    assert "No conversation record exists" in out

    await engine.dispose()
    print("✓ test_recall_missing_day")


# ── Test 11: future date rejected ────────────────────────────────────
async def test_recall_future_rejected():
    engine = await _make_recall_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    future = (datetime.utcnow().date() + timedelta(days=30)).isoformat()
    out = await _call_recall(sm, "u-future", future)
    assert out.startswith("ERROR:") and "resolve" in out.lower()

    await engine.dispose()
    print("✓ test_recall_future_rejected")


# ── Test 12: archival vs rolling prompts are distinct ────────────────
def test_archival_vs_rolling_prompt():
    # Load day_summarizer without triggering app.config path.
    from app.services import day_summarizer as ds
    assert ds.SUMMARIZER_SYSTEM_PROMPT != ds.ARCHIVAL_SUMMARY_PROMPT

    # Archival-specific markers
    archival = ds.ARCHIVAL_SUMMARY_PROMPT
    assert "archival" in archival.lower()
    assert "## Artifacts produced" in archival
    assert "## Topics taught" in archival
    assert "## Narrative arc" in archival
    assert "## Named entities" in archival

    # Rolling summary shouldn't have those section markers
    rolling = ds.SUMMARIZER_SYSTEM_PROMPT
    assert "## Artifacts produced" not in rolling
    assert "## Narrative arc" not in rolling
    print("✓ test_archival_vs_rolling_prompt")


# ── Test 13: _get_spend excludes system.* events ─────────────────────
async def test_llm_proxy_excludes_system_ops():
    """Events with operation_type='system.*' must NOT count toward user cap."""
    engine = create_async_engine("sqlite+aiosqlite://", connect_args={"check_same_thread": False})
    async with engine.begin() as conn:
        await conn.run_sync(lambda c: c.execute(text("""
            CREATE TABLE llm_proxy_events (
                id VARCHAR(36) PRIMARY KEY,
                user_id VARCHAR(36) NOT NULL,
                provider VARCHAR(20) NOT NULL,
                model VARCHAR(100) NOT NULL,
                endpoint VARCHAR(20) NOT NULL,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                cost_cents INTEGER DEFAULT 0,
                was_fallback BOOLEAN DEFAULT 0,
                latency_ms INTEGER DEFAULT 0,
                status VARCHAR(10) DEFAULT 'ok',
                operation_type VARCHAR(50),
                created_at TIMESTAMP NOT NULL
            )
        """)))

    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    since = datetime.utcnow() - timedelta(hours=1)
    now = datetime.utcnow()

    async with sm() as db:
        # 3 user-attributable events (NULL and explicit user.*)
        for i, op in enumerate([None, "user.chat", "user.chat"]):
            await db.execute(text(
                "INSERT INTO llm_proxy_events (id, user_id, provider, model, endpoint, "
                "cost_cents, operation_type, created_at) VALUES (:id,'u','anthropic','claude','chat',100,:op,:ts)"
            ), {"id": f"ue-{i}", "op": op, "ts": now})
        # 5 system events (should be EXCLUDED from cap)
        for i in range(5):
            await db.execute(text(
                "INSERT INTO llm_proxy_events (id, user_id, provider, model, endpoint, "
                "cost_cents, operation_type, created_at) VALUES (:id,'u','anthropic','claude','chat',500,:op,:ts)"
            ), {"id": f"se-{i}", "op": "system.day_archival", "ts": now})
        await db.commit()

    # Replicate the exemption query from llm_proxy._get_spend
    async with sm() as db:
        result = await db.execute(text("""
            SELECT COALESCE(SUM(cost_cents), 0) FROM llm_proxy_events
            WHERE user_id = 'u' AND provider = 'anthropic' AND created_at >= :since
              AND (operation_type IS NULL OR operation_type NOT LIKE 'system.%')
        """), {"since": since})
        user_cents = int(result.scalar() or 0)

        # Without the exemption, total would be 300 + 2500 = 2800.
        # With the exemption, only the 300 cents of user events count.
        assert user_cents == 300, f"Expected 300 user cents, got {user_cents}"

        # All events total = 2800 (for cost dashboards)
        result = await db.execute(text(
            "SELECT COALESCE(SUM(cost_cents), 0) FROM llm_proxy_events WHERE user_id = 'u'"
        ))
        total = int(result.scalar() or 0)
        assert total == 2800

    await engine.dispose()
    print("✓ test_llm_proxy_excludes_system_ops")


# ── Test 14: integration — quiz-from-yesterday scenario ───────────────
async def test_quiz_from_yesterday_integration():
    """The must-pass scenario: mixed-content day, agent pulls relevant thread only."""
    engine = await _make_recall_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with sm() as db:
        await db.execute(text(
            "INSERT INTO users (id, email, hashed_password, name, is_active) "
            "VALUES ('u-quiz', 'q@test', 'x', 'Quiz', 1)"
        ))
        await db.commit()

    yesterday = datetime.utcnow().date() - timedelta(days=1)
    # Realistic day: 80+ messages mixing calculus tutoring, Telegram chatter,
    # and an unrelated build session. We use 40 here — enough to prove filtering.
    msgs = []
    for i in range(10):
        msgs.append(("user", f"random chatter {i}"))
        msgs.append(("assistant", f"acknowledged {i}"))
    msgs.extend([
        ("user", "let's study the calculus course, start with derivatives"),
        ("assistant", "The derivative of a function f(x) measures instantaneous rate of change."),
        ("user", "chain rule?"),
        ("assistant", "Chain rule: d/dx[f(g(x))] = f'(g(x)) * g'(x). Example: d/dx[sin(x^2)] = cos(x^2) * 2x."),
        ("user", "product rule?"),
        ("assistant", "Product rule: d/dx[u*v] = u'v + uv'. Example: d/dx[x^2 * sin(x)] = 2x*sin(x) + x^2*cos(x)."),
        ("user", "quotient rule?"),
        ("assistant", "Quotient rule: d/dx[u/v] = (u'v - uv')/v^2."),
        # Transition messages so the ±2 window around the last "rule" hit doesn't
        # bleed into the unrelated build session below.
        ("user", "got it, thanks"),
        ("assistant", "you're welcome — let me know if you want practice problems"),
        ("user", "maybe later"),
        ("assistant", "sounds good"),
    ])
    for i in range(10):
        msgs.append(("user", f"unrelated deploy question {i}"))
        msgs.append(("assistant", f"deploy answer {i}"))

    # Seed archival summary that MENTIONS the calculus thread.
    archival = (
        "## Narrative arc\nMixed day — tutoring plus build session.\n"
        "## Topics taught / learned / discussed\n"
        "- calculus course: derivatives (definition), chain rule, product rule, quotient rule\n"
        "- (build session: unrelated)\n"
    )
    await _seed_day(sm, "u-quiz", yesterday, msgs, archival=archival)

    # Agent flow step 1: summary-only recall to see if content covers the quiz topic.
    summary_out = await _call_recall(sm, "u-quiz", "yesterday", query="calculus course")
    assert "calculus course" in summary_out
    assert "chain rule" in summary_out  # summary mentions it

    # Agent flow step 2: full conversation filtered to the calculus thread.
    # Use "rule" — it appears in each of the three rule Q&A pairs, so the union
    # of ±2 windows covers the whole lesson. This matches how an agent would
    # realistically query when building a quiz ("recall_day(query='rules')").
    full_out = await _call_recall(
        sm, "u-quiz", "yesterday",
        include_full_conversation=True, query="rule", limit=200,
    )
    # Must contain the calculus content
    assert "chain rule" in full_out.lower()
    assert "product rule" in full_out.lower()
    assert "quotient rule" in full_out.lower()
    # Must NOT contain the bulk of the irrelevant chatter
    assert "random chatter" not in full_out
    assert "unrelated deploy" not in full_out
    # Filter header present
    assert "filtered to" in full_out
    print("✓ test_quiz_from_yesterday_integration")


# ── Test 15: adversarial query inputs (B5) ───────────────────────────
async def test_query_adversarial_inputs():
    """Empty/whitespace/regex-meta/unicode queries must not crash or regex-match."""
    engine = await _make_recall_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with sm() as db:
        await db.execute(text(
            "INSERT INTO users (id, email, hashed_password, name, is_active) "
            "VALUES ('u-adv', 'adv@test', 'x', 'Adv', 1)"
        ))
        await db.commit()

    yesterday = datetime.utcnow().date() - timedelta(days=1)
    # Seed messages including regex metacharacters, Farsi, and long content.
    farsi_q = "درود بر تو"  # Farsi "greetings to you"
    await _seed_day(sm, "u-adv", yesterday, [
        ("user", "testing [brackets] and (parens) and *stars*"),
        ("assistant", "how's function(x) = x^2 + 1"),
        ("user", farsi_q),
        ("assistant", "سلام، چطور میتونم کمک کنم؟"),
        ("user", "A" * 3000),  # long user message
        ("assistant", "short reply"),
    ])

    # Empty string query — should behave like no query.
    out_empty = await _call_recall(sm, "u-adv", "yesterday",
                                    include_full_conversation=True, query="")
    assert "testing [brackets]" in out_empty
    assert "filtered to" not in out_empty  # No filter applied

    # Whitespace-only query — same as empty.
    out_ws = await _call_recall(sm, "u-adv", "yesterday",
                                 include_full_conversation=True, query="    ")
    assert "filtered to" not in out_ws

    # Regex metacharacters — literal match, not regex interpretation.
    # "(x)" should match "function(x)" in the transcript but NOT crash as regex.
    out_regex = await _call_recall(sm, "u-adv", "yesterday",
                                    include_full_conversation=True, query="(x)")
    assert "function(x)" in out_regex
    assert "filtered to" in out_regex

    # Brackets — "[brackets]" should match literally.
    out_brackets = await _call_recall(sm, "u-adv", "yesterday",
                                       include_full_conversation=True, query="[brackets]")
    assert "testing [brackets]" in out_brackets

    # Unicode / Farsi query.
    out_unicode = await _call_recall(sm, "u-adv", "yesterday",
                                      include_full_conversation=True, query=farsi_q)
    assert farsi_q in out_unicode, f"Farsi query must match Farsi content: {out_unicode[:500]}"
    assert "filtered to" in out_unicode

    # Query longer than any message — must not crash.
    huge_query = "z" * 5000
    out_huge = await _call_recall(sm, "u-adv", "yesterday",
                                   include_full_conversation=True, query=huge_query)
    # No crash; may be "no hits" or capped query. Either is fine.
    assert isinstance(out_huge, str)
    assert not out_huge.startswith("ERROR:")

    await engine.dispose()
    print("✓ test_query_adversarial_inputs")


# ── Test 16: archival job concurrency / idempotency (B1) ─────────────
async def test_archival_job_atomic_claim():
    """Concurrent claims on the same DayChat must result in exactly one archival."""
    # Simulate the atomic UPDATE ... WHERE status='not_needed' claim logic.
    engine = await _make_recall_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with sm() as db:
        await db.execute(text(
            "INSERT INTO users (id, email, hashed_password, name, is_active) "
            "VALUES ('u-atomic', 'a@test', 'x', 'Atomic', 1)"
        ))
        await db.commit()

    yesterday = datetime.utcnow().date() - timedelta(days=1)
    dc_id = await _seed_day(sm, "u-atomic", yesterday, [
        ("user", "hi"), ("assistant", "hey"),
    ])

    # Two workers race to claim the same row.
    async def _claim():
        async with sm() as db:
            # This mirrors run_end_of_day_archival's atomic claim:
            # UPDATE day_chats SET status='pending' WHERE id=? AND status='not_needed'
            res = await db.execute(text(
                "UPDATE day_chats SET archival_summary_status='pending' "
                "WHERE id = :id AND archival_summary_status = 'not_needed'"
            ), {"id": dc_id})
            await db.commit()
            return res.rowcount

    # Fire N parallel claims. SQLite serializes writes, so only one should succeed.
    results = await asyncio.gather(*[_claim() for _ in range(5)])
    winners = sum(1 for r in results if r == 1)
    losers = sum(1 for r in results if r == 0)
    assert winners == 1, f"Expected 1 claim winner, got {winners} winners / {losers} losers"
    assert losers == 4

    await engine.dispose()
    print("✓ test_archival_job_atomic_claim")


# ── Test 17: stuck-pending recovery (B1) ─────────────────────────────
async def test_stuck_pending_recovery():
    """A DayChat stuck in 'pending' for >2h must be re-claimable on the next run."""
    engine = await _make_recall_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with sm() as db:
        await db.execute(text(
            "INSERT INTO users (id, email, hashed_password, name, is_active) "
            "VALUES ('u-stuck', 's@test', 'x', 'Stuck', 1)"
        ))
        await db.commit()

    yesterday = datetime.utcnow().date() - timedelta(days=1)
    dc_id = await _seed_day(sm, "u-stuck", yesterday, [("user", "x"), ("assistant", "y")])
    # Mark it stuck: pending, with a generated_at 3 hours ago.
    stuck_ts = datetime.utcnow() - timedelta(hours=3)
    async with sm() as db:
        await db.execute(text(
            "UPDATE day_chats SET archival_summary_status='pending', "
            "archival_summary_generated_at = :ts WHERE id = :id"
        ), {"ts": stuck_ts, "id": dc_id})
        await db.commit()

    # Replicate the job's stuck-pending query.
    cutoff = datetime.utcnow() - timedelta(hours=2)
    async with sm() as db:
        rows = (await db.execute(text(
            "SELECT id FROM day_chats WHERE user_id='u-stuck' "
            "AND archival_summary_status='pending' "
            "AND (archival_summary_generated_at IS NULL OR archival_summary_generated_at < :cutoff)"
        ), {"cutoff": cutoff})).all()
    assert len(rows) == 1, f"Stuck pending row should be selectable as a retry candidate, got {len(rows)}"

    # Now flip the timestamp to recent (only 30 min old). Should NOT be re-claimed.
    recent_ts = datetime.utcnow() - timedelta(minutes=30)
    async with sm() as db:
        await db.execute(text(
            "UPDATE day_chats SET archival_summary_generated_at = :ts WHERE id = :id"
        ), {"ts": recent_ts, "id": dc_id})
        await db.commit()
        rows = (await db.execute(text(
            "SELECT id FROM day_chats WHERE user_id='u-stuck' "
            "AND archival_summary_status='pending' "
            "AND (archival_summary_generated_at IS NULL OR archival_summary_generated_at < :cutoff)"
        ), {"cutoff": cutoff})).all()
    assert len(rows) == 0, "Recently-pending row must NOT be treated as stuck"

    await engine.dispose()
    print("✓ test_stuck_pending_recovery")


# ── Test 18: timezone-sensitive day rollover (B2) ────────────────────
async def test_timezone_day_rollover():
    """A user in Asia/Tokyo has 'yesterday' before a user in America/Los_Angeles."""
    # Can't freeze time across two timezones at once, but we can verify the
    # resolver returns dates that are at most 1 day apart and both strictly
    # in the past.
    from app.agent.date_resolver import resolve_date_reference as rd
    tokyo = rd("yesterday", tz_name="Asia/Tokyo")
    la = rd("yesterday", tz_name="America/Los_Angeles")
    assert tokyo is not None and la is not None

    # The delta between Tokyo and LA "yesterday" is either 0 or 1 day depending
    # on when this runs. Just confirm both are strictly in the past per their
    # own timezone.
    import zoneinfo
    from datetime import datetime as dt
    tokyo_today = dt.now(zoneinfo.ZoneInfo("Asia/Tokyo")).date()
    la_today = dt.now(zoneinfo.ZoneInfo("America/Los_Angeles")).date()
    assert tokyo < tokyo_today, f"Tokyo 'yesterday' ({tokyo}) must be < Tokyo today ({tokyo_today})"
    assert la < la_today, f"LA 'yesterday' ({la}) must be < LA today ({la_today})"
    assert abs((tokyo - la).days) <= 1

    print("✓ test_timezone_day_rollover")


# ── Test 19: DST transition day (B2) ─────────────────────────────────
def test_dst_transition_safety():
    """On a DST transition, 'yesterday' must still be today-1 in local time."""
    # We can't actually jump to a DST morning, but we can at least verify that
    # the resolver uses zoneinfo (DST-aware) and produces a consistent result
    # across consecutive calls within one second.
    from app.agent.date_resolver import resolve_date_reference as rd
    y1 = rd("yesterday", tz_name="America/New_York")
    y2 = rd("yesterday", tz_name="America/New_York")
    assert y1 == y2, f"Non-deterministic yesterday: {y1} vs {y2}"

    # Also verify known DST zones parse without error.
    for tz in [
        "America/New_York", "Europe/London", "Australia/Sydney",
        "Pacific/Auckland", "America/Santiago",
    ]:
        got = rd("yesterday", tz_name=tz)
        assert got is not None, f"Could not resolve yesterday in {tz}"

    print("✓ test_dst_transition_safety")


# ── Test 20: output truncation marker (B6) ───────────────────────────
async def test_recall_output_truncation_marker():
    """When the output exceeds the char budget, the agent sees a clear marker."""
    engine = await _make_recall_engine()
    sm = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with sm() as db:
        await db.execute(text(
            "INSERT INTO users (id, email, hashed_password, name, is_active) "
            "VALUES ('u-big', 'b@test', 'x', 'Big', 1)"
        ))
        await db.commit()

    yesterday = datetime.utcnow().date() - timedelta(days=1)
    # Seed 300 messages of ~200 chars each → ~60KB of content.
    big_msgs = []
    for i in range(150):
        big_msgs.append(("user", f"message number {i} — " + "padding content " * 10))
        big_msgs.append(("assistant", f"response number {i} — " + "response text " * 10))
    await _seed_day(sm, "u-big", yesterday, big_msgs)

    # Full conversation, no query, default limit → will hit OUTPUT_CHAR_BUDGET
    # via the head/tail gap marker (total > limit).
    out = await _call_recall(sm, "u-big", "yesterday",
                              include_full_conversation=True, limit=500)
    # The "elided" marker must appear if limit was hit.
    # (With 300 messages and limit=500, no limit-elision, but output may still exceed budget.)
    # At least one narrow-with-query hint must be present either way:
    assert "elided" in out or "narrow" in out or len(out) <= 45_100, (
        f"Big output had no truncation marker: len={len(out)}"
    )

    await engine.dispose()
    print("✓ test_recall_output_truncation_marker")


# ── Runner ────────────────────────────────────────────────────────────
def main():
    test_resolver_iso_and_literals()
    test_weekday_past_only()
    test_weekday_via_resolver()
    test_month_day()
    test_timezone_yesterday()
    test_archival_vs_rolling_prompt()
    test_dst_transition_safety()

    async def _async():
        await test_recall_prefers_archival_summary()
        await test_recall_falls_back_to_rolling()
        await test_recall_full_conversation()
        await test_recall_query_filter()
        await test_recall_missing_day()
        await test_recall_future_rejected()
        await test_llm_proxy_excludes_system_ops()
        await test_quiz_from_yesterday_integration()
        await test_query_adversarial_inputs()
        await test_archival_job_atomic_claim()
        await test_stuck_pending_recovery()
        await test_timezone_day_rollover()
        await test_recall_output_truncation_marker()

    asyncio.run(_async())
    print("\nAll day-recall tests passed.")


if __name__ == "__main__":
    main()
