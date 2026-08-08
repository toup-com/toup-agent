"""
Retrieval must not be a write, and it must not commit the caller's session.

Two defects, one site — the unconditional top-5 loop that used to sit at the
end of ``MemoryService.hybrid_search``:

  1. It reinforced the ranker's own top results on EVERY search. Strength is an
     input to that same ranker, and decay never runs against tenant memories
     (``agent_memory_maintenance_enabled`` is False), so a mis-retrieval got
     permanently stronger and more likely to be retrieved again — a one-way
     ratchet to MAX_STRENGTH.

  2. ``DecayService.reinforce_memory`` commits the session it is handed. That
     session is the caller's. On the chat path that is the turn-scoped session
     from ``AgentRunner._run_inner``, and ``hybrid_search`` is called from the
     middle of ``_build_system_prompt`` — twice per turn (user leg and
     agent-brain leg), so up to ten commits fired inside prompt assembly.

Reinforcement now happens in ``RetrievalFeedback.log_retrieval_feedback``, off
the request path, only for memories the finished response actually cited. On a
miss turn ``used_ids`` is empty by construction, so "do not reinforce on a
miss" is causal rather than bolted on.

The ANTI-VACUITY controls in this file exist so that "fixed" can never quietly
mean "reinforcement is dead": TestReinforcementStillWorks must stay green.
"""

import json
from datetime import datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import init_db, drop_db, async_session_maker
from app.db.models import Memory, MemoryEvent, MemoryEventType, MemoryLevel, RetrievalEvent
from app.services import create_user
from app.services.decay_service import DecayService
from app.services.memory_service import MemoryService
from app.services.retrieval_feedback import RetrievalFeedback


# The content is deliberately word-rich and distinctive: the cite detector in
# log_retrieval_feedback needs >=30% of the words longer than 3 chars to appear
# in the response, and _keyword_search needs literal words to ILIKE against.
SEEDED_CONTENT = (
    "Nariman prefers dense espresso roasted beans from the Barbari roastery"
)
SEEDED_QUERY = "espresso roastery beans"


@pytest_asyncio.fixture(autouse=True)
async def setup_database():
    await init_db()
    yield
    await drop_db()


@pytest_asyncio.fixture
async def db_session():
    async with async_session_maker() as session:
        yield session


@pytest_asyncio.fixture
async def test_user(db_session: AsyncSession):
    return await create_user(
        db_session,
        email="reinforce-probe@example.com",
        password="not-a-real-password",
        name="Reinforce Probe",
    )


def _make_memory(user_id: str, content: str = SEEDED_CONTENT) -> Memory:
    return Memory(
        user_id=user_id,
        content=content,
        summary=content,
        category="preferences",
        memory_type="preference",
        brain_type="user",
        importance=0.5,
        confidence=0.9,
        strength=0.4,
        memory_level=MemoryLevel.EPISODIC,
        emotional_salience=0.2,
        decay_rate=1.0,
        # Reinforced 7 days ago: comfortably outside the 1-hour cooldown, and
        # far enough back that the log-based boost is clearly measurable.
        last_reinforced_at=datetime.utcnow() - timedelta(days=7),
    )


@pytest_asyncio.fixture
async def seeded_memory(db_session: AsyncSession, test_user) -> Memory:
    mem = _make_memory(str(test_user.id))
    db_session.add(mem)
    await db_session.commit()
    await db_session.refresh(mem)
    return mem


async def _reinforced_events(db: AsyncSession, memory_id: str):
    result = await db.execute(
        select(MemoryEvent)
        .where(MemoryEvent.memory_id == memory_id)
        .where(MemoryEvent.event_type == MemoryEventType.REINFORCED)
    )
    return list(result.scalars().all())


class _CommitCounter:
    """Counts commit() calls on ONE session instance, passing them through."""

    def __init__(self, session: AsyncSession):
        self._session = session
        self._orig = session.commit
        self.count = 0
        session.commit = self  # type: ignore[method-assign]

    async def __call__(self, *args, **kwargs):
        self.count += 1
        return await self._orig(*args, **kwargs)

    def release(self):
        self._session.commit = self._orig  # type: ignore[method-assign]


# ----------------------------------------------------------------------
# 1. Retrieval alone does not reinforce — the ratchet is broken
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_hybrid_search_does_not_reinforce(
    db_session: AsyncSession, test_user, seeded_memory: Memory
):
    """A search that FINDS the memory must not strengthen it."""
    before_strength = seeded_memory.strength
    before_reinforced_at = seeded_memory.last_reinforced_at
    before_access_count = seeded_memory.access_count

    svc = MemoryService(db_session)
    results = await svc.hybrid_search(
        user_id=str(test_user.id),
        query=SEEDED_QUERY,
        limit=5,
        strategies=["keyword"],
    )

    # Anti-vacuity guard for THIS test: if the search found nothing, the
    # "no reinforcement" assertions below would pass for the wrong reason.
    assert [r["id"] for r in results] == [str(seeded_memory.id)], (
        "precondition: hybrid_search must actually retrieve the seeded memory, "
        f"otherwise this test is vacuous. got={results}"
    )

    await db_session.refresh(seeded_memory)
    assert seeded_memory.strength == pytest.approx(before_strength), (
        "retrieval must not strengthen a memory — strength feeds the ranker, "
        "and with decay disabled fleet-wide that is a one-way ratchet"
    )
    assert seeded_memory.last_reinforced_at == before_reinforced_at
    assert seeded_memory.access_count == before_access_count
    assert await _reinforced_events(db_session, str(seeded_memory.id)) == []


@pytest.mark.asyncio
async def test_repeated_searches_do_not_ratchet_strength(
    db_session: AsyncSession, test_user, seeded_memory: Memory
):
    """The production symptom: rows pinned at strength 1.0 by recall alone."""
    svc = MemoryService(db_session)
    for _ in range(6):
        await svc.hybrid_search(
            user_id=str(test_user.id),
            query=SEEDED_QUERY,
            limit=5,
            strategies=["keyword"],
        )

    await db_session.refresh(seeded_memory)
    assert seeded_memory.strength == pytest.approx(0.4)


# ----------------------------------------------------------------------
# 2. hybrid_search must not commit the caller's session
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_hybrid_search_does_not_commit_callers_session(
    db_session: AsyncSession, test_user, seeded_memory: Memory
):
    """
    The production-risk half, asserted two ways.

    (a) semantically: uncommitted work pending on the caller's session must
        still be invisible to an independent session after hybrid_search
        returns. This is the assertion that matters — it does not care how
        the commit was issued.
    (b) mechanically: commit() is never called on that session.
    """
    # Pending, deliberately NOT committed by us.
    sentinel = _make_memory(str(test_user.id), content="sentinel uncommitted row")
    db_session.add(sentinel)
    await db_session.flush()
    sentinel_id = str(sentinel.id)

    # READABILITY CONTROL for (a). Before trusting a negative, prove this
    # backend can see a leak at all. The sqlite harness
    # (file::memory:?cache=shared) hands every session the SAME cache, so an
    # uncommitted row is already cross-session visible there and the check
    # below would report "no commit" no matter what hybrid_search did. On
    # Postgres — the backend this actually runs on — it is a real assertion.
    async with async_session_maker() as probe:
        pre = (await probe.execute(
            select(Memory.id).where(Memory.id == sentinel_id)
        )).scalar_one_or_none()
    isolation_is_observable = pre is None

    counter = _CommitCounter(db_session)
    try:
        svc = MemoryService(db_session)
        results = await svc.hybrid_search(
            user_id=str(test_user.id),
            query=SEEDED_QUERY,
            limit=5,
            strategies=["keyword"],
        )
        # Precondition: the loop under test only ran when there were results.
        assert results, "precondition: hybrid_search must return results"
    finally:
        counter.release()

    # (a) first, because it is the assertion with production meaning.
    if isolation_is_observable:
        async with async_session_maker() as other:
            leaked = await other.execute(
                select(Memory.id).where(Memory.id == sentinel_id)
            )
            assert leaked.scalar_one_or_none() is None, (
                "hybrid_search committed the caller's pending work — the "
                "sentinel row became visible to an independent session"
            )

    # (b) mechanical backstop. Runs on every backend, including the one where
    # (a) is unreadable.
    assert counter.count == 0, (
        f"hybrid_search committed the caller's session {counter.count}x — "
        "a turn calls it twice, so this is up to 10 commits fired from inside "
        "system-prompt assembly"
    )

    await db_session.rollback()


# ----------------------------------------------------------------------
# 3. Cited memories DO get reinforced, from the background path
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cited_memory_is_reinforced_from_feedback_path(
    db_session: AsyncSession, test_user, seeded_memory: Memory
):
    before_strength = seeded_memory.strength

    retrieved = [{
        "id": str(seeded_memory.id),
        "content": SEEDED_CONTENT,
        "similarity_score": 0.8,
    }]
    fb = RetrievalFeedback(db_session)
    event_id = await fb.log_retrieval_feedback(
        user_id=str(test_user.id),
        query="what coffee do I like?",
        retrieved_memories=retrieved,
        # The response repeats the memory almost verbatim → cited.
        response=(
            "You prefer dense espresso roasted beans from the Barbari "
            "roastery, Nariman."
        ),
    )
    await db_session.commit()

    assert event_id
    ev = (await db_session.execute(
        select(RetrievalEvent).where(RetrievalEvent.id == event_id)
    )).scalar_one()
    assert json.loads(ev.used_memory_ids_json) == [str(seeded_memory.id)]

    await db_session.refresh(seeded_memory)
    assert seeded_memory.strength > before_strength, (
        "a memory the response actually used must be reinforced"
    )

    events = await _reinforced_events(db_session, str(seeded_memory.id))
    assert len(events) == 1
    payload = json.loads(events[0].event_data_json)
    assert payload["access_context"] == "cited"


@pytest.mark.asyncio
async def test_miss_turn_reinforces_nothing(
    db_session: AsyncSession, test_user, seeded_memory: Memory
):
    """
    used_ids is empty by construction on a miss, so nothing can be reinforced.
    """
    before_strength = seeded_memory.strength

    fb = RetrievalFeedback(db_session)
    event_id = await fb.log_retrieval_feedback(
        user_id=str(test_user.id),
        query="what coffee do I like?",
        retrieved_memories=[{
            "id": str(seeded_memory.id),
            "content": SEEDED_CONTENT,
            "similarity_score": 0.8,
        }],
        # Nothing from the memory appears here.
        response="Sure, I've scheduled the meeting for tomorrow at nine.",
    )
    await db_session.commit()

    ev = (await db_session.execute(
        select(RetrievalEvent).where(RetrievalEvent.id == event_id)
    )).scalar_one()
    assert ev.quality_signal == "miss"
    assert json.loads(ev.used_memory_ids_json) == []

    await db_session.refresh(seeded_memory)
    assert seeded_memory.strength == pytest.approx(before_strength)
    assert await _reinforced_events(db_session, str(seeded_memory.id)) == []


@pytest.mark.asyncio
async def test_empty_turn_reinforces_nothing(
    db_session: AsyncSession, test_user, seeded_memory: Memory
):
    """Nothing retrieved → nothing cited → nothing reinforced."""
    fb = RetrievalFeedback(db_session)
    event_id = await fb.log_retrieval_feedback(
        user_id=str(test_user.id),
        query="anything?",
        retrieved_memories=[],
        response="You prefer dense espresso roasted beans from Barbari.",
    )
    await db_session.commit()

    ev = (await db_session.execute(
        select(RetrievalEvent).where(RetrievalEvent.id == event_id)
    )).scalar_one()
    assert ev.quality_signal == "empty"
    assert await _reinforced_events(db_session, str(seeded_memory.id)) == []


@pytest.mark.asyncio
async def test_log_retrieval_feedback_does_not_commit_its_session(
    db_session: AsyncSession, test_user, seeded_memory: Memory
):
    """
    The method's documented contract ("let the caller handle the transaction")
    survives the new reinforcement — reinforce_memory is called with
    commit=False.
    """
    counter = _CommitCounter(db_session)
    try:
        fb = RetrievalFeedback(db_session)
        await fb.log_retrieval_feedback(
            user_id=str(test_user.id),
            query="what coffee do I like?",
            retrieved_memories=[{
                "id": str(seeded_memory.id),
                "content": SEEDED_CONTENT,
                "similarity_score": 0.8,
            }],
            response=(
                "You prefer dense espresso roasted beans from the Barbari "
                "roastery, Nariman."
            ),
        )
    finally:
        counter.release()

    assert counter.count == 0
    await db_session.rollback()


@pytest.mark.asyncio
async def test_cite_reinforcement_respects_the_one_hour_cooldown(
    db_session: AsyncSession, test_user, seeded_memory: Memory
):
    """The pre-existing per-memory cooldown must not be bypassed."""
    seeded_memory.last_reinforced_at = datetime.utcnow() - timedelta(minutes=5)
    await db_session.commit()
    before_strength = seeded_memory.strength

    fb = RetrievalFeedback(db_session)
    await fb.log_retrieval_feedback(
        user_id=str(test_user.id),
        query="what coffee do I like?",
        retrieved_memories=[{
            "id": str(seeded_memory.id),
            "content": SEEDED_CONTENT,
            "similarity_score": 0.8,
        }],
        response=(
            "You prefer dense espresso roasted beans from the Barbari "
            "roastery, Nariman."
        ),
    )
    await db_session.commit()

    await db_session.refresh(seeded_memory)
    assert seeded_memory.strength == pytest.approx(before_strength)


@pytest.mark.asyncio
async def test_cite_reinforcement_is_capped(
    db_session: AsyncSession, test_user
):
    """At most REINFORCE_CITED_LIMIT rows per turn — never more than the
    old retrieval-time top-5."""
    mems = []
    for i in range(7):
        m = _make_memory(
            str(test_user.id),
            content=f"Nariman prefers dense espresso roasted beans batch{i}",
        )
        db_session.add(m)
        mems.append(m)
    await db_session.commit()

    fb = RetrievalFeedback(db_session)
    reinforced = await fb._reinforce_used(
        str(test_user.id),
        [{"id": str(m.id), "content": m.content, "similarity_score": 0.8}
         for m in mems],
    )
    await db_session.commit()

    assert len(reinforced) == RetrievalFeedback.REINFORCE_CITED_LIMIT == 5


# ----------------------------------------------------------------------
# 4. ANTI-VACUITY — reinforcement itself is still alive
#    These must stay GREEN when the production change is reverted.
# ----------------------------------------------------------------------

class TestReinforcementStillWorks:

    @pytest.mark.asyncio
    async def test_decay_service_still_reinforces(
        self, db_session: AsyncSession, test_user, seeded_memory: Memory
    ):
        before = seeded_memory.strength
        out = await DecayService(db_session).reinforce_memory(
            str(seeded_memory.id), str(test_user.id), similarity_score=0.8,
        )
        assert out is not None
        assert out.strength > before
        assert len(await _reinforced_events(db_session, str(seeded_memory.id))) == 1

    @pytest.mark.asyncio
    async def test_reinforce_memory_still_commits_by_default(
        self, db_session: AsyncSession, test_user, seeded_memory: Memory
    ):
        """Existing callers (the /memories reinforce route, admin) are
        unchanged: commit defaults to True."""
        await DecayService(db_session).reinforce_memory(
            str(seeded_memory.id), str(test_user.id),
        )
        async with async_session_maker() as other:
            fresh = (await other.execute(
                select(Memory).where(Memory.id == str(seeded_memory.id))
            )).scalar_one()
            assert fresh.strength > 0.4, (
                "reinforce_memory(commit=True) must still persist"
            )

    @pytest.mark.asyncio
    async def test_strength_rises_for_a_genuinely_used_memory(
        self, db_session: AsyncSession, test_user, seeded_memory: Memory
    ):
        """End-to-end control: retrieve, then answer using the memory.
        Strength must be unchanged after retrieval and higher after the
        response cites it."""
        svc = MemoryService(db_session)
        results = await svc.hybrid_search(
            user_id=str(test_user.id),
            query=SEEDED_QUERY,
            limit=5,
            strategies=["keyword"],
        )
        assert results, "precondition: retrieval must return the memory"

        fb = RetrievalFeedback(db_session)
        await fb.log_retrieval_feedback(
            user_id=str(test_user.id),
            query=SEEDED_QUERY,
            retrieved_memories=results,
            response=(
                "You prefer dense espresso roasted beans from the Barbari "
                "roastery, Nariman."
            ),
        )
        await db_session.commit()

        await db_session.refresh(seeded_memory)
        assert seeded_memory.strength > 0.4
