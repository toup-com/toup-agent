"""Drivers that exercise the REAL production memory pipeline.

Nothing here re-implements capture. `drive_turn` calls
`AgentRunner._extract_memories` — the exact bound method the agent invokes in
turn post-processing (agent_runner.py:4637) — with a stub `self` that supplies
the only attribute it touches (`_last_extraction_ok`). That means the LLM
extraction call, the write-time gate, the batched dedup, the entity upsert and
the relationship mirror all run exactly as they do in production, against a real
Postgres with pgvector.

`recall` drives the read side through `MemoryService.hybrid_search` with the same
parameters `agent_runner.py:3454` uses.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy import and_, func, select


# ── Write path ───────────────────────────────────────────────────────────

class _RunnerStub:
    """Minimal stand-in for AgentRunner.

    `_extract_memories` reads nothing off `self` and writes only
    `_last_extraction_ok`; asserting that here means a future refactor that
    starts depending on more runner state fails loudly instead of silently
    testing a different code path.
    """

    _last_extraction_ok: str = "-"


async def drive_turn(
    db,
    user_id: str,
    user_message: str,
    assistant_response: str = "",
    *,
    trivial: bool = False,
) -> int:
    """Run one conversational turn through production capture. Returns count."""
    from app.agent.agent_runner import AgentRunner

    stub = _RunnerStub()
    n = await AgentRunner._extract_memories(
        stub, db, user_id, user_message, assistant_response, trivial
    )
    await db.commit()
    return n


async def drive_conversation(db, user_id: str, turns: Sequence["Turn"]) -> int:
    total = 0
    for t in turns:
        total += await drive_turn(
            db, user_id, t.user, t.assistant, trivial=t.trivial
        )
    return total


async def store_direct(
    db,
    user_id: str,
    content: str,
    *,
    category: str = "identity",
    memory_type: str = "fact",
    importance: float = 0.8,
    expires_at=None,
) -> Any:
    """Write one memory through the dedup service (the memory_store tool path)."""
    from app.schemas import MemoryCreate, BrainType, MemoryType, MemoryLevel
    from app.services.memory_dedup_service import MemoryDedupService

    dedup = MemoryDedupService(db)
    memory, action = await dedup.smart_create_memory(
        new_memory=MemoryCreate(
            content=content,
            brain_type=BrainType.USER,
            category=category,
            memory_type=MemoryType(memory_type),
            importance=importance,
            confidence=0.9,
            memory_level=MemoryLevel.EPISODIC,
            emotional_salience=0.5,
            source_type="agent_tool",
            expires_at=expires_at,
        ),
        user_id=user_id,
    )
    await db.commit()
    return memory, action


async def seed_bulk(db, user_id: str, contents: Sequence[str], *, chunk: int = 64) -> int:
    """Seed many memories fast, bypassing dedup but NOT bypassing storage.

    Rows are written through `MemoryService.create_memory` with real embeddings
    (batched concurrently), so vector search, tsvector search and the graph all
    behave exactly as they would on organically-captured rows. Only the dedup
    adjudication is skipped — deduplication is measured in category F, and
    running it here would make a 1000-row seed cost 1000 LLM calls.
    """
    import asyncio

    from app.schemas import MemoryCreate, BrainType, MemoryType, MemoryLevel
    from app.services.embedding_service import get_embedding_service
    from app.services.memory_service import MemoryService

    svc = MemoryService(db)
    embedder = get_embedding_service()
    written = 0

    for start in range(0, len(contents), chunk):
        batch = list(contents[start : start + chunk])
        vectors = await asyncio.gather(
            *(embedder.embed_async(c) for c in batch), return_exceptions=True
        )
        for content, vec in zip(batch, vectors):
            await svc.create_memory(
                user_id=user_id,
                memory_data=MemoryCreate(
                    content=content,
                    brain_type=BrainType.USER,
                    category="other",
                    memory_type=MemoryType("fact"),
                    importance=0.6,
                    confidence=0.9,
                    memory_level=MemoryLevel.EPISODIC,
                    emotional_salience=0.5,
                    source_type="seed",
                ),
                deduplicate=False,
                embedding=None if isinstance(vec, BaseException) else vec,
                commit=False,
            )
            written += 1
        await db.commit()
    return written


# ── Read path ────────────────────────────────────────────────────────────

async def recall(
    db,
    user_id: str,
    query: str,
    *,
    limit: Optional[int] = None,
    min_similarity: Optional[float] = None,
    strategies: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Retrieve as agent_runner.py:3454 does."""
    from app.config import settings
    from app.services.memory_service import MemoryService

    return await MemoryService(db).hybrid_search(
        user_id=user_id,
        query=query,
        limit=limit if limit is not None else settings.memory_retrieval_limit,
        min_similarity=(
            min_similarity
            if min_similarity is not None
            else settings.memory_retrieval_min_similarity
        ),
        strategies=strategies or ["vector", "keyword", "graph"],
    )


async def injected_context(db, user_id: str, query: str) -> List[Dict[str, Any]]:
    """The memory rows that actually reach the model on one turn.

    Mirrors agent_runner.py's assembly: the query-independent core facts
    (`get_core_facts`, capped at CORE_FACTS_LIMIT) merged ahead of the
    query-conditioned `hybrid_search` results. Retrieval tests assert on THIS,
    because "did the agent know?" is a question about the composed context, not
    about one leg of it.
    """
    from app.agent.agent_runner import CORE_FACTS_LIMIT
    from app.services.memory_service import MemoryService

    svc = MemoryService(db)
    core = await svc.get_core_facts(user_id=user_id, limit=CORE_FACTS_LIMIT)
    retrieved = await recall(db, user_id, query)

    seen = {m["id"] for m in core}
    return core + [m for m in retrieved if m["id"] not in seen]


async def injected_contents(db, user_id: str, query: str) -> List[str]:
    return [m["content"] for m in await injected_context(db, user_id, query)]


# ── Store inspection ─────────────────────────────────────────────────────

async def active_memories(db, user_id: str) -> List[Any]:
    """Every memory a retrieval path could serve for this user."""
    from app.db.models.memory import Memory

    rows = await db.execute(
        select(Memory)
        .where(
            and_(
                Memory.user_id == user_id,
                Memory.is_deleted == False,  # noqa: E712
                Memory.is_active == True,  # noqa: E712
            )
        )
        .order_by(Memory.created_at)
    )
    return list(rows.scalars())


async def all_memory_rows(db, user_id: str) -> List[Any]:
    """Every row, including soft-deleted and superseded."""
    from app.db.models.memory import Memory

    rows = await db.execute(
        select(Memory).where(Memory.user_id == user_id).order_by(Memory.created_at)
    )
    return list(rows.scalars())


async def active_contents(db, user_id: str) -> List[str]:
    return [m.content for m in await active_memories(db, user_id)]


async def count_all_memories(db) -> int:
    from app.db.models.memory import Memory

    return int((await db.execute(select(func.count(Memory.id)))).scalar() or 0)


# ── Ground-truth matching ────────────────────────────────────────────────

def norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return re.sub(r"\s+", " ", text)


@dataclass(frozen=True)
class Marker:
    """A machine-checkable claim about the store.

    `all_of` substrings must ALL appear in a SINGLE memory for the marker to be
    considered present. Markers use distinctive tokens (proper nouns, numbers,
    rare words) so matching cannot be satisfied by an unrelated row.
    """

    id: str
    all_of: Sequence[str]

    def found_in(self, contents: Sequence[str]) -> Optional[str]:
        for c in contents:
            n = norm(c)
            if all(norm(tok) in n for tok in self.all_of):
                return c
        return None


@dataclass
class Turn:
    user: str
    assistant: str = ""
    trivial: bool = False


@dataclass
class Scenario:
    id: str
    turns: List[Turn]
    must_store: List[Marker] = field(default_factory=list)
    must_not_store: List[Marker] = field(default_factory=list)
    lang: str = "en"
    note: str = ""


@dataclass
class ScenarioResult:
    scenario_id: str
    stored: List[str]
    missed: List[str]
    junk: List[str]

    @property
    def recall(self) -> float:
        total = len(self.stored) + len(self.missed)
        return 1.0 if total == 0 else len(self.stored) / total

    def describe(self) -> str:
        lines = [f"scenario={self.scenario_id} recall={self.recall:.0%} junk={len(self.junk)}"]
        if self.missed:
            lines.append("  MISSED (must be stored, was not):")
            lines += [f"    - {m}" for m in self.missed]
        if self.junk:
            lines.append("  JUNK (must NOT be stored, was):")
            lines += [f"    - {j}" for j in self.junk]
        return "\n".join(lines)


async def run_scenario(db, user_id: str, sc: Scenario) -> ScenarioResult:
    await drive_conversation(db, user_id, sc.turns)
    contents = await active_contents(db, user_id)

    stored, missed, junk = [], [], []
    for m in sc.must_store:
        (stored if m.found_in(contents) else missed).append(m.id)
    for m in sc.must_not_store:
        hit = m.found_in(contents)
        if hit:
            junk.append(f"{m.id} :: {hit[:160]}")
    return ScenarioResult(sc.id, stored, missed, junk)
