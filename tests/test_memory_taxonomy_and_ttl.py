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
        # These three CHANGED. The original table asserted PEOPLE-wins, which
        # is the behaviour the founder reported as a bug ("User -> owns -> Toup
        # categorized People"). See
        # test_relationship_category_describes_the_object_not_the_person.
        ("person", "organization", "work"),   # was "people"
        ("show", "organization", "media"),    # was "work"
        ("book", "person", "media"),          # was "people"
        ("person", "person", "people"),
        ("project", "tool", "work"),
        ("topic", "topic", "knowledge"),
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


# ── 5b. Relationship rows read as sentences, not predicate-ese ────────

def test_relationship_rendering_never_leaks_a_raw_predicate():
    """"never show raw `play_on` predicates to a user" — the founder's brief.

    #375 removed the arrow form from `summary`, but `content` — which is what
    the card actually renders now that summary is None — was still
    `predicate.replace("_", " ")` spliced between two entity names. That is
    grammatical only by accident: `performed_by` reads fine, `play_on` gave
    "Better Call Saul play on Netflix". These are real predicates from a live
    tenant (40 distinct ones on toup-agent-871bac24 alone).
    """
    from app.memory_taxonomy import humanize_relationship

    assert (
        humanize_relationship("Better Call Saul", "play_on", "Netflix")
        == "Better Call Saul is available on Netflix"
    )
    assert (
        humanize_relationship("Bunker", "performed_by", "Baltazar")
        == "Baltazar performs Bunker"
    )
    assert (
        humanize_relationship("Shops at Don Mills", "located_in", "Toronto")
        == "Shops at Don Mills is in Toronto"
    )
    # Aliases collapse the vocabulary drift the extractor produces.
    assert humanize_relationship("Nariman", "owner_of", "Toup") == \
        humanize_relationship("Nariman", "owns", "Toup")
    assert humanize_relationship("User", "connected_to", "Gmail") == \
        humanize_relationship("User", "connects_to", "Gmail")

    # An unknown predicate must still produce a sentence, not a bare stem.
    assert humanize_relationship("User", "frobnicate", "Widget") == \
        "User frobnicates Widget"
    assert humanize_relationship("User", "carry", "Bag") == "User carries Bag"
    # Already-inflected verbs are left alone rather than double-suffixed.
    assert humanize_relationship("User", "watches", "Netflix") == \
        "User watches Netflix"


def test_forgotten_relationships_stay_forgotten_across_the_phrasing_change():
    """The soft-delete guard matches content EXACTLY, and the template moved.

    Rows forgotten before humanize_relationship carry the raw predicate form.
    If the guard only checked the new prose form, every one of them would be
    resurrected on the next mention — and there is no restore route, so the
    user cannot undo it a second time.
    """
    import inspect

    from app.services.memory_service import MemoryService

    src = inspect.getsource(MemoryService.store_entity_relationship)
    assert "legacy_content" in src, (
        "the pre-humanize phrasing is no longer checked against soft-deletes"
    )
    guard = inspect.getsource(MemoryService._relationship_was_forgotten)
    assert "in_(candidates)" in guard


# ── 5c. Expiry leases respect explicit user intent ────────────────────

async def test_user_keep_clears_the_expiry_lease():
    """Tapping Keep must actually save the memory.

    Reinforcement moved strength and last_reinforced_at but never expires_at,
    so a memory the user explicitly kept was still archived by the sweep —
    and there is no restore route.

    This test EXECUTES the path. The previous version only asserted the source
    text contained the right lines, which is why it stayed green while the
    branch raised `NameError: logger` on every real Keep: decay_service.py
    called logger.info() in a module that never imported logging. A source
    assertion cannot catch an undefined name — only running the code can.
    """
    from app.db.models.memory import Memory
    from app.services.decay_service import DecayService

    engine, Session = await _memory_session()
    try:
        async with Session() as db:
            mem = Memory(
                id="keep-1",
                user_id="u-keep",
                content="The user wants a daily Gmail briefing at 11:49.",
                category="active_task",
                brain_type="user",
                memory_type="fact",
                memory_level="episodic",
                importance=0.6,
                confidence=0.8,
                strength=1.0,
                # Recently reinforced ON PURPOSE: Keep must win over the
                # 1-hour cooldown early-return, not be skipped by it.
                last_reinforced_at=datetime.utcnow() - timedelta(minutes=5),
                expires_at=datetime.utcnow() + timedelta(days=3),
            )
            db.add(mem)
            await db.flush()

            svc = DecayService(db)
            returned = await svc.reinforce_memory(
                "keep-1", "u-keep", access_context="user_reinforce"
            )

            assert returned is not None, "Keep could not load the memory"
            assert mem.expires_at is None, (
                "Keep must clear the expiry lease; the memory is still on a "
                "countdown and the sweep will archive it anyway."
            )
    finally:
        await engine.dispose()


async def test_recurring_arrangements_survive_the_active_task_archiver():
    """decay_expired_tasks archives on category alone — it must not eat routines.

    The taxonomy migration remapped legacy `schedule` onto `active_task` and
    deliberately left recurring arrangements without an expires_at. That
    exemption is worthless unless THIS archiver honours it too: a standing
    "daily Gmail briefing" has no reinforcement of its own (the routine fires,
    not the memory), so the plain age rule archived four of the founder's real
    routines on the day of the rollout.
    """
    from app.db.models.memory import Memory
    from app.services.active_task_service import decay_expired_tasks

    old = datetime.utcnow() - timedelta(days=30)
    engine, Session = await _memory_session()
    try:
        async with Session() as db:
            recurring = Memory(
                id="rec-1", user_id="u-rec",
                content="The user wants a daily email routine at 1:43 PM to "
                        "summarize the last five Gmail messages.",
                category="active_task", brain_type="user", memory_type="fact",
                memory_level="episodic", importance=0.6, confidence=0.8,
                strength=1.0, created_at=old, last_reinforced_at=old,
                expires_at=None,
            )
            one_off = Memory(
                id="task-1", user_id="u-rec",
                content="The user asked to be reminded to call their brother.",
                category="active_task", brain_type="user", memory_type="fact",
                memory_level="episodic", importance=0.4, confidence=0.8,
                strength=1.0, created_at=old, last_reinforced_at=old,
                expires_at=None,
            )
            leased = Memory(
                id="task-2", user_id="u-rec",
                content="The user is comparing project-management tools.",
                category="active_task", brain_type="user", memory_type="fact",
                memory_level="episodic", importance=0.4, confidence=0.8,
                strength=1.0, created_at=old, last_reinforced_at=old,
                # Old enough for the age rule, but the lease has NOT run out.
                expires_at=datetime.utcnow() + timedelta(days=2),
            )
            db.add_all([recurring, one_off, leased])
            await db.flush()

            archived = await decay_expired_tasks(db, "u-rec")

            assert recurring.is_active is True, (
                "a standing daily routine was archived by the active-task TTL"
            )
            assert leased.is_active is True, (
                "expires_at is authoritative; archiving ahead of it overrules "
                "the sweep, the reinforce path and the Keep button"
            )
            assert one_off.is_active is False, (
                "a genuinely stale one-off task should still be archived"
            )
            assert archived == 1
    finally:
        await engine.dispose()


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


# ── 7. D-mem-B: MCP memory writes must reach the tenant ───────────────

def test_mcp_memory_writes_proxy_to_the_tenant():
    """D-mem-B (2026-07-29 remediation tracker).

    `memory_update` on an id the agent had just been handed returned
    "Memory not found". Root cause: the MCP tools run on the PLATFORM and were
    bound to the platform session, but `memories` is AGENT_ONLY — the row is
    real, it is simply in the tenant's database. #375 fixed this class for the
    REST routes and missed this surface.

    Parsed from source: importing mcp_server needs fastmcp, which is not
    installed in every environment.
    """
    import ast
    import pathlib

    src = pathlib.Path("app/mcp_server.py").read_text()
    tree = ast.parse(src)

    names = {
        n.name: n
        for n in ast.walk(tree)
        if isinstance(n, (ast.AsyncFunctionDef, ast.FunctionDef))
    }
    assert "_proxy_memory_write_to_tenant" in names

    # memory_create was missing from this loop, and so was the proxy: the
    # agent's own memory_create tool wrote tenant-private content into the
    # SHARED platform DB, where its owner can never read it back. Worse after
    # #377 — update and delete proxy to a tenant that has no such row, so the
    # stranded row is also un-editable and un-deletable.
    for tool in ("memory_create", "memory_update", "memory_delete"):
        body = ast.unparse(names[tool])
        assert "_proxy_memory_write_to_tenant" in body, (
            f"{tool} does not route to the tenant — it will 404 on every real id"
        )
        # The tenant must be consulted BEFORE the local session, or the
        # platform's empty table answers first and we are back to the bug.
        assert body.index("_proxy_memory_write_to_tenant") < body.index(
            "MemoryService(db)"
        ), f"{tool} consults the platform session before the tenant"

    # memory_list is a READ, so it may fall back to the platform session — but
    # it has to ASK the tenant first. Listing the platform DB does not error,
    # it returns an empty list, so an agent checking what it already knows
    # about someone is told "nothing" while their memories sit in the tenant.
    assert "_proxy_memory_list_from_tenant" in names
    list_body = ast.unparse(names["memory_list"])
    assert "_proxy_memory_list_from_tenant" in list_body, (
        "memory_list reads the platform DB and will report zero memories"
    )
    assert list_body.index("_proxy_memory_list_from_tenant") < list_body.index(
        "MemoryService(db)"
    ), "memory_list consults the platform session before the tenant"


def test_remaining_mcp_tools_on_agent_only_tables_are_known():
    """Guard rail: this surface is not fully proxied yet, and that is tracked.

    memory_create/update/delete/list now route to the tenant. The rest of the
    MCP tools still run against the platform session while their tables are
    AGENT_ONLY — `entities`, `entity_relationships`, `conversations`. Fixing
    those needs agent-side REST routes that do not exist yet (there is no
    app/api/entities.py to proxy to), so it is a separate change.

    This test fails if that set GROWS or SHRINKS, so the gap cannot quietly
    widen and cannot be silently forgotten once the routes land.
    """
    import ast
    import pathlib

    src = pathlib.Path("app/mcp_server.py").read_text()
    tree = ast.parse(src)

    unproxied = set()
    for n in ast.walk(tree):
        if not isinstance(n, (ast.AsyncFunctionDef, ast.FunctionDef)):
            continue
        if not any("mcp.tool" in ast.unparse(d) for d in n.decorator_list):
            continue
        body = ast.unparse(n)
        if "async_session_maker()" in body and "_from_tenant" not in body \
                and "_to_tenant" not in body:
            unproxied.add(n.name)

    assert unproxied == {
        "memory_search",          # has a builtin in tool_executor; never hits MCP
        "session_create", "session_list",
        "entity_search", "graph_traverse", "entity_relationship_create",
        "identity_get", "identity_update",
    }, f"the unproxied MCP surface changed: {sorted(unproxied)}"


# ── 8. Corrective backfill (scripts/repair_memory_regressions.py) ──────

def test_dead_reminder_regex_matches_the_extractor_s_actual_phrasing():
    """The repair script's sub-hour detector must match REAL content.

    A prefix-only pattern ("in N minutes") looks right and catches nothing:
    the extractor overwhelmingly writes the SUFFIX form ("... 2 minutes after
    the request"). Every string below is verbatim from a live tenant.
    """
    import importlib

    mod = importlib.import_module("scripts.repair_memory_regressions")
    rx = mod._SUBHOUR_HORIZON

    dead = [
        "The user wants to be reminded to eat tea 2 minutes after the request.",
        "The user wants to be reminded two minutes after the request to call "
        "their brother.",
        "The user asked to be reminded here in chat in 2 minutes to drink "
        "water, rather than through Telegram",
        "The user requested a reminder to wake up two minutes later.",
    ]
    durable = [
        "The user is researching UofT future-student events",
        "The user wants a daily Gmail briefing every day at 10:58 AM "
        "America/Toronto time",
        "The user wants a recurring daily 2:27 PM briefing with their latest "
        "5 Gmail messages",
        "The user wants to be reminded on 2026-05-21 when they wake up to "
        "work on the mobile app",
    ]
    for c in dead:
        assert rx.search(c), f"dead reminder not detected: {c!r}"
    for c in durable:
        assert not rx.search(c), f"durable memory wrongly flagged: {c!r}"


def test_repair_never_resurrects_a_forgotten_or_superseded_row():
    """Restoring routines must not undo a deliberate user action.

    The signal is the COLUMN, not the event log. Verified on a live tenant:
    the only trigger_source values that exist are `api`, `conversation`,
    `taxonomy_migration`, `memory_expiry` and `conversation_backfill` — there
    is no user-archive event at all, and `decay_expired_tasks` writes none
    either. An earlier version keyed off trigger_source and rejected all 11
    real candidates, because `api/reinforced` (906 of them) is automated
    reinforcement during retrieval, not a user action.

    A user forgetting a memory sets `is_deleted=True`; archival sets
    `is_active=False`. Filtering on both already excludes user deletions. The
    remaining exclusion is dedup supersession, whose loser is also deactivated
    but whose content is still live under `superseded_by`.
    """
    import inspect
    import importlib

    mod = importlib.import_module("scripts.repair_memory_regressions")
    src = inspect.getsource(mod.restore_archived_routines)

    assert "Memory.is_deleted == False" in src, (
        "a memory the user forgot must be excluded by column, not by heuristic"
    )
    assert "mem.superseded_by" in src, (
        "restoring a dedup loser resurrects a duplicate of a live row"
    )
    assert "describes_recurring_arrangement" in src
    # The retired trigger_source heuristic must not come back.
    assert "_AUTOMATED_TRIGGERS" not in inspect.getsource(mod)

    # And the whole script must be dry-run-capable with no default mutation.
    main_src = inspect.getsource(mod.main)
    assert "required=True" in main_src, "apply must never be the default"
    assert "await session.rollback()" in main_src


def test_humanize_step_keeps_richer_prose():
    """Only rows whose content IS the old machine format may be rewritten.

    Many entity_extraction rows carry real prose from another write path —
    "When handling Gmail, including during the user's daily briefing, ..." for
    a reconnect_using edge. Re-rendering those from the bare triple would
    DESTROY content to fix formatting that was never broken.
    """
    import inspect
    import importlib

    mod = importlib.import_module("scripts.repair_memory_regressions")
    src = inspect.getsource(mod.humanize_relationship_rows)

    assert "richer_content_kept" in src
    assert 'legacy = f"{source} {str(predicate).replace(\'_\', \' \')} {target}"' in src
    # The real metadata keys, read off live rows. Guessing source_name made
    # this whole step a silent no-op across 75 rows.
    assert "source_entity" in src and "target_entity" in src
    assert "relationship_type" in src


def test_humanizer_never_inflects_a_modal_verb():
    """The verb-repair fallback must not mangle finite verbs.

    `might_cause_problems_in` is a real predicate from a live tenant. Blindly
    appending "-s" to the head word turned "Rampage might cause problems in
    Canada" into "Rampage mights cause problems in Canada" — worse than the raw
    form this function exists to improve. Found by dry-running the backfill
    against production before applying it.
    """
    from app.memory_taxonomy import humanize_relationship as h

    assert h("Rampage", "might_cause_problems_in", "Canada") == \
        "Rampage might cause problems in Canada"
    assert h("X", "can_use", "Y") == "X can use Y"
    assert h("X", "should_be_sent_via", "Y") == "X should be sent via Y"
    # US spelling, matching the rest of the codebase.
    assert h("email routine", "summarizes", "Gmail") == \
        "email routine summarizes Gmail"
    # And the genuinely-broken case still gets repaired.
    assert h("User", "frobnicate", "Widget") == "User frobnicates Widget"


def test_relationship_category_describes_the_object_not_the_person():
    """"User -> owns -> Toup categorized People" — the founder's Symptom 2.

    PEOPLE must win only when BOTH ends are people. The first version of this
    table put PEOPLE first, reasoning that "person -> works_at -> organization
    is about the person". That is backwards: when a person is one end, the
    OTHER end is the subject matter. 24 of 72 relationship rows on the
    founder's tenant had collapsed into PEOPLE for that reason.
    """
    from app.memory_taxonomy import category_for_relationship as c

    assert c("person", "organization") == "work"      # User owns Toup
    assert c("person", "person") == "people"          # Nariman chatted with majid
    assert c("person", "skill") == "skills"           # the user's skill, not the user
    assert c("person", "place") == "locations"
    assert c("show", "organization") == "media"       # the show, not the streamer
    assert c("place", "location") == "locations"
    # Unknown on both ends stays honest rather than guessing.
    assert c(None, None) == "knowledge"
    assert c("wharrgarbl", "flimflam") == "knowledge"


def test_recategorize_is_not_in_the_default_repair_run():
    """Its logic is right; bulk-applying it to legacy rows makes data worse.

    It depends on entity TYPES, and those are unreliable upstream — "Better
    Call Saul" is typed `topic`, songs are typed `project`. Measured on the
    founder's tenant: ~5 rows improved, ~8 degraded. And no subset of type
    pairs separates them, because the same person/project pair produces both
    "User owns Toup" (correctly -> work) and "Drake artist of 0-100"
    (wrongly -> work, should be media).
    """
    import importlib

    mod = importlib.import_module("scripts.repair_memory_regressions")

    assert "recategorize" in mod.STEPS, "keep it reachable via --only"
    assert "recategorize" not in mod.DEFAULT_STEPS, (
        "recategorize must not run by default — entity typing is the blocker"
    )
    assert set(mod.DEFAULT_STEPS) == {"restore", "humanize", "reminders"}


# ── 9. Category breakdown (the filter bar's data source) ───────────────

def test_breakdown_route_is_declared_before_the_id_route():
    """`/memories/breakdown` must not be swallowed by `/memories/{memory_id}`.

    FastAPI matches in declaration order, so a path param declared first would
    turn every breakdown request into a lookup for a memory whose id is
    literally "breakdown" — a 404, or worse a 500 from the uuid parse. The
    file already carries this hazard for `/search`; the note there is why.
    """
    import inspect
    import re

    from app.api import memories as memories_api

    src = inspect.getsource(memories_api)
    order = re.findall(r'@router\.get\("(/[^"]*)"', src)

    assert "/breakdown" in order, "the breakdown route is not registered"
    assert order.index("/breakdown") < order.index("/{memory_id}"), (
        "/breakdown is declared after /{memory_id} and will be shadowed"
    )


def test_breakdown_normalizes_legacy_categories_before_counting():
    """Counts must fold legacy values, or the filter bar shows both.

    A tenant mid-migration can hold `schedule` and `active_task` at once. Two
    chips for one category — with the counts split between them — is worse than
    no chip at all, because tapping either shows a partial list.
    """
    import inspect

    from app.api import memories as memories_api

    src = inspect.getsource(memories_api.memory_breakdown)
    assert "normalize_category" in src, (
        "breakdown counts raw category values and will double-count legacy rows"
    )
    # Counting must exclude what the user cannot see, or the chips will not add
    # up to the "N remembered" header.
    assert "Memory.is_deleted == False" in src
    assert "Memory.is_active == True" in src
    # And it must reach the tenant: `memories` is AGENT_ONLY.
    assert "_proxy_memories" in src
