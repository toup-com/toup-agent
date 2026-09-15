"""The tz seed is correct code reading a column nobody populated (E1).

2026-09-14 01:40:21Z. A WhatsApp turn landed on a tenant whose `users.timezone`
was NULL — the platform knew `America/Toronto`, the tenant row had never been
told — so `_resolve_effective_tz` returned None, the runner fell to
`tz_name='UTC'`, and the day chat was minted for 2026-09-14 while the owner's
local date was still 2026-09-13. Every later mobile turn resolved 09-13, so the
WhatsApp half of the conversation was invisible in the app.

This file pins the four sources of D4 in priority order and the isolation
between users. It runs in the DEFAULT platform sweep: `_resolve_effective_tz`
only SELECTs `users` (PLATFORM+AGENT), and the `day_tables` fixture builds the
AGENT_ONLY tables the seeding half needs.

Local run (one process, from backend/):
    RUN_MODE=platform PYTHONPATH=. ENVIRONMENT=test JWT_SECRET=x ENCRYPTION_KEY=y \
      pytest tests/test_channel_tz_seed.py -q --tb=short -p no:cacheprovider
"""
from __future__ import annotations

import uuid

import pytest
import pytest_asyncio


@pytest_asyncio.fixture
async def day_tables():
    """The AGENT_ONLY tables, created directly so this file stays in the
    platform sweep (recipe: tests/test_shared_day_context_invariants.py:48)."""
    from sqlalchemy import inspect as sa_inspect
    from app.db.database import engine
    from app.db.models.base import Base

    names = ["day_chats", "conversations", "messages", "context_budget_logs"]
    async with engine.begin() as conn:
        for n in names:
            table = Base.metadata.tables.get(n)
            if table is None:
                continue
            try:
                await conn.run_sync(table.create, checkfirst=True)
            except Exception:
                pass

    async with engine.connect() as conn:
        existing = await conn.run_sync(lambda c: set(sa_inspect(c).get_table_names()))
    missing = [n for n in names if n not in existing]
    if missing:
        pytest.skip(f"cannot create {missing} on this backend")
    yield


@pytest.fixture(autouse=True)
def _clear_tz_cache():
    """A leaked TTL entry makes every case after the first vacuous — the
    resolver answers from the cache and never reaches the branch under test."""
    from app.agent._user_tz_cache import _USER_TZ_CACHE

    _USER_TZ_CACHE.clear()
    yield
    _USER_TZ_CACHE.clear()


def _runner():
    """A bare runner: `_resolve_effective_tz` touches no instance state."""
    from app.agent.agent_runner import AgentRunner

    return AgentRunner.__new__(AgentRunner)


def _require_seed_path():
    """Skip until lane B1 lands D4's runtime/phone seeds on the RESOLVER.

    `_resolve_effective_tz` is a METHOD, so the probe must read the class —
    a module-level hasattr answers False forever and would make every case
    below a permanent skip that reads green.
    """
    from app.agent.agent_runner import AgentRunner

    if not hasattr(AgentRunner, "_nullfill_user_tz"):
        pytest.skip("D4 runtime user_timezone seed not landed yet — lane B1")


def _require_phone_source():
    """`_tz_from_phone` is a METHOD on AgentRunner (the resolver calls
    `self._tz_from_phone(channel)`), so the probe reads the class only — a
    module-level fallback here once let the case below self-skip forever."""
    from app.agent.agent_runner import AgentRunner

    if not hasattr(AgentRunner, "_tz_from_phone"):
        pytest.skip("D4 source 4 (phone-derived tz) not landed yet — lane B1")


async def _seed_user(tz):
    from app.db.database import async_session_maker
    from app.db.models import User

    uid = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(
            id=uid, email=f"tzseed-{uuid.uuid4().hex[:10]}@example.com",
            hashed_password="x", name="TZ Seed", timezone=tz,
        ))
        await db.commit()
    return uid


async def _resolve(user_id, client_tz, channel):
    from app.db.database import async_session_maker
    from app.agent.agent_runner import AgentRunner

    async with async_session_maker() as db:
        return await AgentRunner._resolve_effective_tz(
            _runner(), db, user_id, client_tz, channel,
        )


async def _read_tz(user_id):
    from sqlalchemy import select
    from app.db.database import async_session_maker
    from app.db.models import User

    async with async_session_maker() as db:
        row = (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
        return getattr(row, "timezone", None)


# ── source 2: the tenant users row ────────────────────────────────────


@pytest.mark.asyncio
async def test_whatsapp_turn_uses_the_tenant_profile_timezone():
    """THE E1 case with the column populated: no client_tz, channel whatsapp."""
    uid = await _seed_user("America/Toronto")
    assert await _resolve(uid, None, "whatsapp") == "America/Toronto"


@pytest.mark.asyncio
async def test_null_profile_timezone_resolves_to_none():
    """The state the incident started in. None is the contract — the caller
    is what turns it into 'UTC', and D4 makes that path log `tz_unknown`."""
    uid = await _seed_user(None)
    assert await _resolve(uid, None, "whatsapp") is None


@pytest.mark.asyncio
async def test_explicit_client_tz_wins_over_a_conflicting_profile():
    uid = await _seed_user("America/Toronto")
    assert await _resolve(uid, "Europe/Berlin", "mobile") == "Europe/Berlin"


@pytest.mark.asyncio
async def test_one_users_timezone_is_never_returned_for_another():
    """Cache-key isolation. The TTL cache is a module-level dict keyed by
    user id; a missing key would make one tenant's tz fleet-wide."""
    a = await _seed_user("America/Toronto")
    b = await _seed_user(None)
    assert await _resolve(a, None, "whatsapp") == "America/Toronto"
    assert await _resolve(b, None, "whatsapp") is None


@pytest.mark.asyncio
async def test_the_profile_lookup_is_cached_after_the_first_hit():
    """Anti-vacuity for the isolation test above: prove the cache is real,
    so `test_one_users_timezone_is_never_returned_for_another` is checking a
    live path rather than an unused one."""
    from app.agent._user_tz_cache import _USER_TZ_CACHE

    uid = await _seed_user("America/Toronto")
    assert await _resolve(uid, None, "whatsapp") == "America/Toronto"
    assert uid in _USER_TZ_CACHE


# ── source 3: runtime_identity user_timezone (D4, lane B1) ────────────


@pytest.mark.asyncio
async def test_runtime_user_timezone_fills_a_null_tenant_row(monkeypatch, day_tables):
    """D4 source 3. The bind carries the platform's tz into runtime.json;
    the resolver reads it, NULL-FILLS the tenant column and caches it, so
    the very first tz-less turn on a fresh tenant lands on the right day."""
    _require_seed_path()

    uid = await _seed_user(None)
    monkeypatch.setattr(
        "app.services.runtime_identity.get_runtime_field",
        lambda k, default=None: "America/Toronto" if k == "user_timezone" else default,
        raising=False,
    )
    assert await _resolve(uid, None, "whatsapp") == "America/Toronto"
    assert await _read_tz(uid) == "America/Toronto", "the NULL-fill did not happen"


@pytest.mark.asyncio
async def test_a_non_null_tenant_timezone_is_never_overwritten(monkeypatch, day_tables):
    """The tenant row is authority once it holds a local value — the bind and
    the runtime seed are NULL-FILLS, never writes."""
    _require_seed_path()

    uid = await _seed_user("Europe/London")
    monkeypatch.setattr(
        "app.services.runtime_identity.get_runtime_field",
        lambda k, default=None: "America/Toronto" if k == "user_timezone" else default,
        raising=False,
    )
    assert await _resolve(uid, None, "whatsapp") == "Europe/London"
    assert await _read_tz(uid) == "Europe/London"


@pytest.mark.asyncio
async def test_utc_is_never_persisted_as_a_learned_timezone(monkeypatch, day_tables):
    """D4's guard. 'UTC' is the DEFAULT the resolver falls to when it knows
    nothing — persisting it makes the unknown state indistinguishable from a
    user who really lives in UTC, and permanently unrepairable."""
    _require_seed_path()

    uid = await _seed_user(None)
    monkeypatch.setattr(
        "app.services.runtime_identity.get_runtime_field",
        lambda k, default=None: "UTC" if k == "user_timezone" else default,
        raising=False,
    )
    await _resolve(uid, None, "whatsapp")
    assert await _read_tz(uid) is None, "'UTC' was written into users.timezone"


@pytest.mark.asyncio
async def test_an_invalid_timezone_string_degrades_and_never_raises(monkeypatch, day_tables):
    """`ZoneInfo('Mars/Olympus')` raises; the resolver must not."""
    _require_seed_path()

    uid = await _seed_user(None)
    monkeypatch.setattr(
        "app.services.runtime_identity.get_runtime_field",
        lambda k, default=None: "Mars/Olympus" if k == "user_timezone" else default,
        raising=False,
    )
    assert await _resolve(uid, None, "whatsapp") is None
    assert await _read_tz(uid) is None


# ── source 4: phone-derived, whatsapp/telegram only, NEVER persisted ──


@pytest.mark.asyncio
async def test_the_phone_derived_timezone_is_ephemeral(monkeypatch, day_tables):
    """D4 source 4 is a last resort and is explicitly not authority: it is
    never written to the tenant row and never cached past the turn."""
    _require_phone_source()
    import app.agent.agent_runner as ar
    from app.agent._user_tz_cache import _USER_TZ_CACHE

    uid = await _seed_user(None)
    # The CLASS attribute: patching the module name never reached the
    # resolver, the value came back None and the two assertions that matter
    # were skipped on every run.
    monkeypatch.setattr(
        ar.AgentRunner, "_tz_from_phone", lambda self, channel=None: "America/Toronto",
    )
    got = await _resolve(uid, None, "whatsapp")
    assert got == "America/Toronto", "the phone source was not consulted for a tz-less WhatsApp turn"
    assert await _read_tz(uid) is None, "a phone-derived tz must never be persisted"
    assert uid not in _USER_TZ_CACHE, "a phone-derived tz must never be cached"


@pytest.mark.asyncio
async def test_the_phone_source_is_not_consulted_for_a_web_turn(monkeypatch, day_tables):
    """A web/mobile turn has a real client_tz when it has one at all; guessing
    from a WhatsApp number for a browser session is a wrong answer, not a
    fallback."""
    _require_phone_source()
    import app.agent.agent_runner as ar

    uid = await _seed_user(None)
    calls = []

    def _spy(*a, **k):
        calls.append(a)
        return "America/Toronto"

    monkeypatch.setattr(ar, "_tz_from_phone", _spy, raising=False)
    assert await _resolve(uid, None, "web") is None
    assert calls == [], "the phone source was consulted for channel='web'"


@pytest.mark.asyncio
async def test_a_bind_payload_saying_utc_is_not_a_zone(monkeypatch, day_tables):
    """Review R1 P3: source 3 accepted and CACHED 'UTC' where the nullfill it
    calls refuses it. Cached, 'UTC' is indistinguishable from a deliberate
    choice and it disables the tz-change repair path (ws_chat's gate reads
    `_old_tz != "UTC"`). The resolver must fall through as if the platform
    knew nothing."""
    _require_seed_path()
    from app.agent._user_tz_cache import _USER_TZ_CACHE
    from app.services import runtime_identity as ri

    uid = await _seed_user(None)
    monkeypatch.setattr(
        ri, "get_runtime_field",
        lambda name, default=None: "UTC" if name == "user_timezone" else default,
    )
    got = await _resolve(uid, None, "web")  # web: no phone source to fall to
    assert got is None, f"a bind 'UTC' was accepted as the effective zone: {got!r}"
    assert await _read_tz(uid) is None
    assert uid not in _USER_TZ_CACHE, "'UTC' was cached from the bind payload"
