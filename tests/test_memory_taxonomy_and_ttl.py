"""Regression tests for the 2026-07-29 memory taxonomy + hygiene work.

Each test locks one property whose absence caused a symptom visible in the
production Memory screen. The audit numbers quoted are from the founder's
tenant (toup-agent-871bac24, 119 active rows) on 2026-07-29.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine


async def _memory_session():
    """Local sqlite session with just the tables these tests touch.

    `memories` is an AGENT_ONLY table, so conftest's platform-profile
    `init_db()` does not create it — same reason test_active_task.py builds
    its own engine. Schema comes from the live ORM so it cannot drift.
    """
    from app.db.models.base import Base
    from app.db.models.memory import Memory as _M, MemoryEvent as _ME
    from app.db.models.user import User as _U

    engine = create_async_engine(
        "sqlite+aiosqlite://", connect_args={"check_same_thread": False}
    )
    async with engine.begin() as conn:
        await conn.run_sync(
            Base.metadata.create_all,
            tables=[_U.__table__, _M.__table__, _ME.__table__],
        )
    return engine, async_sessionmaker(engine, expire_on_commit=False)


# ── 1. Taxonomy: ONE definition, no drift ─────────────────────────────

def test_schemas_and_models_share_one_taxonomy():
    """schemas.py and db/models/enums.py must be the SAME enum objects.

    They used to be two declarations with only 12 of ~21 values in common,
    which is why the extractor wrote categories the app had no label for.
    """
    from app import memory_taxonomy
    from app.db.models import enums as model_enums
    from app import schemas

    assert schemas.MemoryCategory is memory_taxonomy.MemoryCategory
    assert model_enums.MemoryCategory is memory_taxonomy.MemoryCategory
    assert schemas.MemoryType is model_enums.MemoryType
    assert schemas.AgentCategory is model_enums.AgentCategory
    assert schemas.BrainType is model_enums.BrainType


def test_every_category_has_a_prompt_definition():
    """The extraction prompt is GENERATED from the enum.

    A category with no definition would be emitted to the LLM as a bare word,
    which is the failure mode that collapsed production to three buckets.
    `_assert_guides_complete()` enforces this at import; this test pins it.
    """
    from app.memory_taxonomy import (
        CATEGORY_GUIDE, MemoryCategory, build_category_prompt_block,
    )

    for cat in MemoryCategory:
        assert cat in CATEGORY_GUIDE, f"{cat.value} missing a definition"

    block = build_category_prompt_block()
    for cat in MemoryCategory:
        assert cat.value in block, f"{cat.value} absent from generated prompt"


@pytest.mark.parametrize(
    "legacy,expected",
    [
        ("schedule", "active_task"),
        ("projects", "work"),
        ("learning", "skills"),
        ("tools", "possessions"),
        ("family", "people"),
        ("places", "locations"),
        ("food", "preferences"),
        ("travel", "experiences"),
        ("context", "other"),
    ],
)
def test_legacy_categories_map_to_canonical(legacy, expected):
    """Every value the old extractor wrote must land on a labelled category."""
    from app.memory_taxonomy import normalize_category

    assert normalize_category(legacy) == expected


def test_extracted_memory_category_keeps_dot_value_access():
    """Several consumers do an unguarded `mem.category.value`.

    api/chat.py:211 and modules/chat/router.py:210 both run on agent
    containers — a bare `str` there is an AttributeError that would kill
    memory extraction on those routes. The extractor must hand back enum
    members (which are also `str`, so `==` comparisons still work).
    """
    from app.memory_taxonomy import MemoryCategory, MemoryType
    from app.services.memory_extractor import ExtractedMemory

    mem = ExtractedMemory(
        content="The user lives in Toronto", summary=None,
        category=MemoryCategory("locations"), memory_type=MemoryType("fact"),
        importance=0.7, confidence=0.9, entities=[], tags=[], metadata={},
    )
    assert mem.category.value == "locations"
    assert mem.memory_type.value == "fact"
    # str-subclass: plain comparisons keep working for everyone else.
    assert mem.category == "locations"


def test_unknown_category_falls_back_not_raises():
    """An unrecognised category must never cost us the memory itself."""
    from app.memory_taxonomy import normalize_category

    assert normalize_category("wharrgarbl") == "other"
    assert normalize_category(None) == "other"
    assert normalize_category("") == "other"
    assert normalize_category("nonsense", brain_type="agent") == "domain_knowledge"


def test_no_alias_shadows_a_canonical_value():
    """An alias that shadows a real value would silently rewrite it."""
    from app.memory_taxonomy import MEMORY_CATEGORY_ALIASES, MemoryCategory

    canonical = {c.value for c in MemoryCategory}
    assert not (canonical & set(MEMORY_CATEGORY_ALIASES))


def test_mobile_label_map_covers_every_canonical_category():
    """The app's CATEGORY_LABELS must label every value the backend can write.

    A missing entry renders as "Other" — that gap was 28% of production rows.
    Parsed from the TSX so the two can never drift silently.
    """
    import pathlib
    import re

    screen = pathlib.Path(
        "/Users/nariman/toup-platform-app/src/variants/c-tabs/screens/MemoryScreen.tsx"
    )
    if not screen.exists():
        pytest.skip("mobile app repo not present")

    src = screen.read_text()
    block = re.search(
        r"const CATEGORY_LABELS: Record<string, string> = \{(.*?)\n\};", src, re.S
    )
    assert block, "CATEGORY_LABELS not found in MemoryScreen.tsx"
    labelled = set(re.findall(r"(\w+):\s*'", block.group(1)))

    from app.memory_taxonomy import AgentCategory, MemoryCategory

    for cat in list(MemoryCategory) + list(AgentCategory):
        assert cat.value in labelled, f"{cat.value} has no label in the mobile app"


# ── 2. Transience / TTL ───────────────────────────────────────────────

def test_durable_categories_never_expire_even_if_flagged_transient():
    """A mislabel must not be able to expire a real fact about the user."""
    from app.memory_taxonomy import NEVER_EXPIRE_CATEGORIES, resolve_ttl_days

    for cat in NEVER_EXPIRE_CATEGORIES:
        assert resolve_ttl_days(cat.value, 1) is None, cat.value
        assert resolve_ttl_days(cat.value) is None, cat.value


def test_transient_categories_get_a_horizon():
    from app.memory_taxonomy import resolve_ttl_days

    assert resolve_ttl_days("active_task") == 7
    # Sub-day horizons round up to a full day of grace.
    assert resolve_ttl_days("active_task", 0.001) == 1
    # Absurd horizons are clamped.
    assert resolve_ttl_days("active_task", 10_000) == 365


def test_recurring_arrangements_are_not_treated_as_transient():
    """A standing routine is phrased like a schedule but is a durable preference.

    Both land in ACTIVE_TASK, so category alone cannot separate them. Strings
    below are taken verbatim from the founder's live tenant — 9 of its 28
    legacy `schedule` rows describe recurring arrangements, and expiring those
    would stop the agent knowing about routines the user still relies on.
    """
    from app.memory_taxonomy import describes_recurring_arrangement

    recurring = [
        "The user wants a recurring daily Gmail briefing every day at 11:49 AM",
        "The user wants to receive a motivational quote every day at 5:06 PM.",
        "The user wants a daily routine at 1:18 to receive the latest 5 Gmail emails.",
        "Send a summary each morning",
        "Remind me every Monday to file the report",
    ]
    one_off = [
        "The user wants to be reminded to eat tea 2 minutes after the request.",
        "The user requested a reminder to wake up two minutes later.",
        "The user urgently asked the assistant to research the 3 best project-management tools",
        "The user requested that an existing scheduled item be changed to 1:21 PM.",
    ]
    for c in recurring:
        assert describes_recurring_arrangement(c) is True, c
    for c in one_off:
        assert describes_recurring_arrangement(c) is False, c


@pytest.mark.asyncio
async def test_expiry_sweep_archives_but_never_deletes():
    """The sweep sets is_active=False. The row and its content survive."""
    from app.db.models import Memory, User
    from app.services.memory_expiry import expire_stale_memories

    user_id = str(uuid.uuid4())
    stale_id, fresh_id, permanent_id = (str(uuid.uuid4()) for _ in range(3))

    engine, Session = await _memory_session()
    async with Session() as db:
        db.add(User(id=user_id, email=f"ttl-{user_id[:8]}@example.com", hashed_password="x"))
        db.add(Memory(
            id=stale_id, user_id=user_id, content="remind me to eat tea in 2 minutes",
            category="active_task", memory_type="task", brain_type="user",
            expires_at=datetime.utcnow() - timedelta(days=1),
        ))
        db.add(Memory(
            id=fresh_id, user_id=user_id, content="the deploy is still running",
            category="active_task", memory_type="task", brain_type="user",
            expires_at=datetime.utcnow() + timedelta(days=3),
        ))
        db.add(Memory(
            id=permanent_id, user_id=user_id, content="the user's daughter is called Mira",
            category="people", memory_type="fact", brain_type="user",
            expires_at=None,
        ))
        await db.commit()

        archived = await expire_stale_memories(db, user_id)
        await db.commit()

        assert [m.id for m in archived] == [stale_id]

        stale = await db.get(Memory, stale_id)
        fresh = await db.get(Memory, fresh_id)
        permanent = await db.get(Memory, permanent_id)

        # Archived, NOT deleted — content intact and still queryable.
        assert stale.is_active is False
        assert stale.is_deleted is False
        assert stale.content == "remind me to eat tea in 2 minutes"

        assert fresh.is_active is True
        assert permanent.is_active is True

    await engine.dispose()


@pytest.mark.asyncio
async def test_expiry_sweep_is_tenant_scoped():
    """The sweep must never touch another user's memories."""
    from app.db.models import Memory, User
    from app.services.memory_expiry import expire_stale_memories

    mine, theirs = str(uuid.uuid4()), str(uuid.uuid4())
    their_memory_id = str(uuid.uuid4())

    engine, Session = await _memory_session()
    async with Session() as db:
        for uid in (mine, theirs):
            db.add(User(id=uid, email=f"iso-{uid[:8]}@example.com", hashed_password="x"))
        db.add(Memory(
            id=their_memory_id, user_id=theirs, content="another tenant's expired memory",
            category="active_task", memory_type="task", brain_type="user",
            expires_at=datetime.utcnow() - timedelta(days=5),
        ))
        await db.commit()

        archived = await expire_stale_memories(db, mine)
        await db.commit()

        assert archived == []
        their_memory = await db.get(Memory, their_memory_id)
        assert their_memory.is_active is True, "cross-tenant write!"

    await engine.dispose()


# ── 3. Rendering: no raw triples ──────────────────────────────────────

def test_relationship_memories_do_not_write_arrow_summaries():
    """`summary` must not carry the machine triple — the card renders it.

    Asserted against the source because building the row needs a live DB and
    an embedding provider; the property is a single literal in one place.
    """
    import pathlib

    src = pathlib.Path("app/services/memory_service.py").read_text()
    assert 'summary=f"{source_name} → {relationship} → {target_name}"' not in src, (
        "store_entity_relationship is writing raw triples into summary again"
    )


def test_relationship_category_is_not_a_hardcoded_binary():
    import pathlib

    src = pathlib.Path("app/services/memory_service.py").read_text()
    assert 'category="people" if source_type == "person" else "knowledge"' not in src, (
        "the people|knowledge binary is back"
    )


@pytest.mark.parametrize(
    "source_type,target_type,expected",
    [
        ("person", "organization", "people"),
        ("person", "person", "people"),
        ("project", "tool", "work"),
        ("show", "organization", "work"),
        ("topic", "topic", "knowledge"),
        ("book", "person", "people"),
        (None, None, "knowledge"),
    ],
)
def test_relationship_category_resolution(source_type, target_type, expected):
    from app.memory_taxonomy import category_for_relationship

    assert category_for_relationship(source_type, target_type) == expected


def test_mobile_card_renders_content_first():
    """The card must prefer `content`; `summary` held the machine format."""
    import pathlib

    screen = pathlib.Path(
        "/Users/nariman/toup-platform-app/src/variants/c-tabs/screens/MemoryScreen.tsx"
    )
    if not screen.exists():
        pytest.skip("mobile app repo not present")

    src = screen.read_text()
    assert "{memory.summary || memory.content}" not in src
    assert "{memory.content || memory.summary}" in src


# ── 4. Reinforcement must not neutralise decay ────────────────────────

def test_reinforcement_does_not_inflate_consolidation_count():
    """consolidation_count is a decay-resistance multiplier worth up to 2x.

    Incrementing it on every RESTATEMENT (rather than on real consolidation)
    made frequently-mentioned throwaways the most decay-resistant rows in the
    brain — production had counts as high as 94.

    Scoped to the two reinforce functions on purpose: `merge_memory` and
    `consolidation_service` increment it legitimately, because a merge really
    is a consolidation.
    """
    import inspect
    import re

    from app.services.memory_dedup_service import MemoryDedupService
    from app.services.memory_service import MemoryService

    # Match an actual MUTATION of the field, not a mention of it — the reinforce
    # paths legitimately READ consolidation_count when building audit events.
    mutation = re.compile(
        r"\.consolidation_count\s*(?:\+=|=(?!=))", re.MULTILINE
    )

    for fn in (
        MemoryDedupService._reinforce_existing_memory,
        MemoryService._reinforce_memory,
    ):
        src = inspect.getsource(fn)
        assert not mutation.search(src), (
            f"{fn.__qualname__} mutates consolidation_count — this is a "
            f"decay-resistance multiplier and must only change on real "
            f"consolidation, not on every restatement"
        )


def test_importance_is_not_a_one_way_ratchet():
    import pathlib

    for path in (
        "app/services/memory_dedup_service.py",
        "app/services/memory_service.py",
    ):
        src = pathlib.Path(path).read_text()
        assert "0.7 * _old_importance + 0.3 * new_data.importance" in src, path


# ── 5. Agent brain ────────────────────────────────────────────────────

def test_reflection_gate_fires_only_on_corrections_and_rules():
    from app.services.agent_reflection import should_reflect

    for msg in [
        "No, that's wrong — reminders should go to the current chat",
        "I meant the other project",
        "Always keep your answers short",
        "stop sending me telegram messages",
        "from now on use metric units",
    ]:
        assert should_reflect(msg) is True, msg

    # False positives found by adversarial review — an over-greedy leading-"no"
    # pattern fired on all of these, spending an LLM call on ordinary chat and
    # risking junk agent-brain rows.
    for msg in [
        "what's the weather today",
        "thanks!",
        "can you summarise this article",
        "hi",
        "I have no idea what to cook tonight",
        "there is no rush on this",
        "no worries, can you check my calendar",
        "there's no problem with the deploy",
        "I always enjoy these conversations",
    ]:
        assert should_reflect(msg) is False, msg


@pytest.mark.asyncio
async def test_agent_brain_writes_are_not_absorbed_into_user_brain():
    """Dedup must be brain-scoped.

    Without the brain filter a new agent-brain note could match a similar
    user-brain row and reinforce that instead of being stored — one reason
    the agent brain never accumulated anything.
    """
    from app.db.models import Memory, User
    from app.services.memory_service import MemoryService

    import json

    user_id = str(uuid.uuid4())
    identical_text = "Send reminders to the current chat, not Telegram"
    vec = [0.1] * 1536

    engine, Session = await _memory_session()
    async with Session() as db:
        db.add(User(id=user_id, email=f"brain-{user_id[:8]}@example.com", hashed_password="x"))
        db.add(Memory(
            id=str(uuid.uuid4()), user_id=user_id, content=identical_text,
            category="preferences", memory_type="preference", brain_type="user",
            embedding_json=json.dumps(vec),
        ))
        await db.commit()

        svc = MemoryService(db)
        # Stub the embedder: this test is about the brain_type predicate, not
        # about the vector provider, and must not make a network call.
        svc.embedding_service.embed = lambda text, api_key=None: vec

        # An IDENTICAL string in the user brain must not be visible to an
        # agent-brain search — otherwise the agent-brain write is absorbed.
        agent_hits = await svc.find_similar_memories(
            user_id=user_id, content=identical_text,
            min_similarity=0.0, brain_type="agent",
        )
        assert agent_hits == [], "cross-brain match — agent writes would be absorbed"

        # Control: the same search scoped to the user brain DOES find it, so
        # the test is proving the filter and not merely a broken query.
        user_hits = await svc.find_similar_memories(
            user_id=user_id, content=identical_text,
            min_similarity=0.0, brain_type="user",
        )
        assert len(user_hits) == 1

    await engine.dispose()


def test_agent_reflection_rejects_self_praise():
    """The reflection prompt sees the assistant's own text; guard the output."""
    from app.services.agent_reflection import _SELF_PRAISE

    for bad in [
        "You did a great job explaining the deploy process",
        "The user was happy with the answer",
        "You successfully completed the task",
    ]:
        assert _SELF_PRAISE.search(bad), bad

    for good in [
        "Send reminders to the current chat, not Telegram",
        "Keep answers under three sentences unless asked for detail",
    ]:
        assert not _SELF_PRAISE.search(good), good


# ── 5b. Every consumer of the taxonomy speaks canonical ───────────────

def test_query_classifier_emits_only_canonical_categories():
    """The classifier was a FIFTH copy of the taxonomy.

    hybrid_search ANDs `Memory.category.in_(categories)` onto every strategy,
    so one retired value ("places", "learning") returns ZERO memories for a
    whole class of question instead of degrading. Adversarial review caught
    this as a critical regression.
    """
    from app.memory_taxonomy import MemoryCategory
    from app.services.query_classifier import classify_query

    canonical = {c.value for c in MemoryCategory}

    probes = [
        "where do i live",
        "who am i",
        "what am i studying at the moment",
        "what are my goals",
        "what do i like to eat",
        "what am i working on",
        "who is sarah",
        "how is my health",
        "remind me what i decided",
    ]
    for probe in probes:
        cats = classify_query(probe).get("categories") or []
        stale = [c for c in cats if c not in canonical]
        assert not stale, f"{probe!r} -> retired categories {stale}"


def test_portrait_categories_are_canonical():
    """PORTRAIT_CATEGORIES was a SIXTH copy — it listed retired `projects`."""
    from app.memory_taxonomy import MemoryCategory
    from app.services.user_portrait_service import PORTRAIT_CATEGORIES

    canonical = {c.value for c in MemoryCategory}
    stale = [c for c in PORTRAIT_CATEGORIES if c not in canonical]
    assert not stale, f"portrait references retired categories: {stale}"


def test_agent_runner_normalizes_classifier_categories_defensively():
    """Belt and braces: even a stale list must cost recall, not correctness."""
    import inspect

    from app.agent.agent_runner import AgentRunner

    src = inspect.getsource(AgentRunner._build_system_prompt)
    assert "normalize_category(c) for c in _raw_categories" in src


# ── 5c. Expiry leases respect explicit user intent ────────────────────

def test_user_keep_clears_the_expiry_lease():
    """Tapping Keep must actually save the memory.

    Reinforcement moved strength and last_reinforced_at but never expires_at,
    so a memory the user explicitly kept was still archived by the sweep —
    and there is no restore route.
    """
    import inspect

    from app.services.decay_service import DecayService

    src = inspect.getsource(DecayService.reinforce_memory)
    assert 'access_context == "user_reinforce"' in src
    assert "memory.expires_at = None" in src
    # Must run BEFORE the 1-hour cooldown early-return, or Keep silently
    # no-ops for a recently-reinforced memory.
    assert src.index("memory.expires_at = None") < src.index("Cooldown")


def test_durable_restatement_promotes_a_transient_memory():
    """Transient -> durable must clear the lease.

    Dedup returns the incumbent WITHOUT creating a new row, so the reinforce
    path is the only place the promotion can happen.
    """
    import inspect

    from app.services.memory_dedup_service import MemoryDedupService
    from app.services.memory_service import MemoryService

    for fn in (
        MemoryDedupService._reinforce_existing_memory,
        MemoryService._reinforce_memory,
    ):
        src = inspect.getsource(fn)
        assert "if new_expiry is None:" in src or "if _new_expiry is None:" in src, (
            f"{fn.__qualname__} keeps a stale lease on a durable restatement"
        )


def test_active_task_reinforcement_renews_the_expiry_lease():
    """Two TTL mechanisms cover these rows; both clocks must move together."""
    import inspect

    from app.services import active_task_service

    src = inspect.getsource(active_task_service.store_active_task)
    assert "mem.expires_at = datetime.utcnow() + timedelta(days=ACTIVE_TASK_TTL_DAYS)" in src


# ── 6. Write path reaches the tenant ──────────────────────────────────

def test_memory_writes_never_silently_fall_back_to_platform_db():
    """A write that cannot reach the tenant must error, not report success.

    `memories` is AGENT_ONLY; a "successful" write against the platform DB is
    how a user comes to believe they deleted something that is still there.
    """
    import inspect

    from app.api import memories as memories_api

    src = inspect.getsource(memories_api._proxy_memories_write)
    assert "HTTPException" in src
    assert "502" in src or "BAD_GATEWAY" in src
    # The read helper keeps its fallback; the write helper must not have one.
    assert "return None" not in src.split("if resp.status_code == 404")[-1]


def test_write_routes_proxy_to_the_tenant():
    import inspect

    from app.api import memories as memories_api

    for fn in (
        memories_api.update_memory,
        memories_api.delete_memory,
        memories_api.create_memory,
        memories_api.reinforce_memory,
    ):
        src = inspect.getsource(fn)
        assert "_proxy_memories_write" in src, f"{fn.__name__} does not proxy writes"
