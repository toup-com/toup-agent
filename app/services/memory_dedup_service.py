"""
Memory Deduplication and Evolution Service

This service handles:
1. Detecting duplicate/similar memories before insertion
2. Merging related memories into evolving records
3. Tracking history of memory changes
4. Generating change summaries using LLM
"""

import asyncio
import json
import logging
import time
from typing import Optional, List, Tuple, Dict, Any
from datetime import datetime

from sqlalchemy import and_, update as sa_update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.services.memory_service import (
    MemoryService,
    conflicts_on_value,
    mentions_different_subjects,
    value_tokens,
)
from app.services.embedding_service import get_embedding_service
from app.services.memory_gate import memory_gate_reason
from app.services.llm_service import get_llm_service
from app.schemas import MemoryCreate, MemoryResponse, BrainType
from app.db.models import Memory

from app.services.memory_log import describe_memory

logger = logging.getLogger(__name__)

# Similarity thresholds - used as initial filter, LLM makes final decision
# These are deliberately LOW to catch potential matches, LLM decides what to do
CANDIDATE_THRESHOLD = 0.40  # Consider as potential match for LLM analysis
MIN_THRESHOLD = 0.25        # Below this, definitely not related

# W1.4d: threshold shortcut around the LLM adjudication — near-identical
# vectors are duplicates, weak matches are new memories; only the ambiguous
# middle band is worth an LLM call.
AUTO_DUPLICATE_THRESHOLD = 0.90  # >= : reinforce existing without asking the LLM
AUTO_NEW_THRESHOLD = 0.50        # best candidate below this: create without asking

# Hard ceiling on how many candidates one write adjudicates against, and the
# `limit` handed to the candidate search that produces them. Adjudication is
# batched into ONE LLM call regardless of how many pairs it covers, so this
# bounds PROMPT SIZE, not call count — but it has to stay explicit: without it,
# raising the search limit would silently widen every write's adjudication
# prompt.
MAX_ADJUDICATED_CANDIDATES = 5

# Precedence when candidates disagree — higher wins. See _select_decision.
_ACTION_PRECEDENCE = {
    "new": 0,
    "duplicate": 1,
    "merge": 2,
    "contradiction_update": 3,
}

# Verdicts that make a fact stop being retrievable, and must therefore be
# confirmed by a second, independent adjudication before they are applied.
#
#   duplicate            -> the INCOMING text is discarded (existing row is
#                           reinforced instead)
#   contradiction_update -> the MATCHED row is retired (is_active=False)
#
# `merge` is deliberately absent: it composes both texts into one row rather
# than dropping either, so a wrong merge is a tidiness problem, not a
# disappearance. See _confirm_destructive_verdict.
_DESTRUCTIVE_ACTIONS = frozenset({"duplicate", "contradiction_update"})

# Verdict origins that are already reproducible, so a second opinion could only
# add noise. Every decision dict carries an explicit `source`; anything NOT
# listed here — including a dict that omits the key — gets confirmed, so the
# default for a future verdict producer is the safe one.
_DETERMINISTIC_SOURCES = frozenset({"threshold", "heuristic", "default"})

# How long a same-user write will queue for the dedup advisory lock before
# giving up and proceeding unlocked. Must stay comfortably under the engine's
# 30s `command_timeout` / `statement_timeout`: past that the DATABASE kills the
# waiter, asyncpg invalidates the connection, and the write is lost rather than
# merely duplicated. See `_lock_user_writes` for the full account.
_LOCK_WAIT_BUDGET_S = 8.0
_LOCK_POLL_INTERVAL_S = 0.05


class SupersedeRaceError(RuntimeError):
    """Another writer superseded the same row first. The loser's replacement
    is rolled back; _apply_decision retries it as a plain create."""


class MemoryDedupService:
    """
    Service for intelligent memory deduplication and evolution.
    
    This service provides smart memory management by:
    - Detecting duplicates before insertion
    - Merging related information into existing memories
    - Tracking how memories evolve over time
    - Using LLM to intelligently combine information
    """
    
    def __init__(self, db: AsyncSession, api_key: Optional[str] = None):
        self.db = db
        self.api_key = api_key
        self.memory_service = MemoryService(db, api_key=api_key)
        self.embedding_service = get_embedding_service()
        if api_key:
            from app.services.llm_service import LLMService
            self.llm_service = LLMService(api_key=api_key)
        else:
            self.llm_service = get_llm_service()
    
    async def smart_create_memory(
        self,
        new_memory: MemoryCreate,
        user_id: str,
        *,
        explicit_save: bool = False,
    ) -> Tuple[Optional[Memory], str]:
        """
        Intelligently create or merge a memory.
        
        This is the main entry point for creating memories with deduplication.
        It checks for similar existing memories and decides whether to:
        - Create a new memory
        - Reinforce an existing one (exact duplicate)
        - Merge with an existing one (similar but with new info)
        
        Args:
            new_memory: The memory data to create
            user_id: The user's ID
            
        Returns:
            Tuple of (memory, action) where action is one of:
            - "created": New memory created
            - "merged": Merged into existing memory
            - "skipped": Exact duplicate, no action taken (returns existing)
            - "reinforced": Same info, just strengthened existing
        """
        # Step 0: the write gate.
        #
        # The gate used to live in exactly one producer (the LLM extractor), so
        # every OTHER way into memory was unscreened — the agent's own
        # memory_store tool, the MCP create_memory tool, and the chat paths all
        # arrive here having passed nothing. Those are the paths where the MODEL
        # chooses the content, which is precisely where a screen is worth having:
        # the same omission on the active-task path was storing secrets and
        # prompt-injection payloads.
        #
        # Placed at the storage boundary rather than at each caller, so a new
        # caller is gated by construction instead of by remembering to ask.
        gate_reason = memory_gate_reason(new_memory.content, explicit_save=explicit_save)
        if gate_reason:
            logger.info(
                "[memory_gate] refused at write (%s): %s",
                gate_reason, describe_memory(new_memory.content),
            )
            return None, f"rejected:{gate_reason}"

        # Step 1: Generate embedding for new content (async — the sync embed
        # blocks the event loop on OpenAI HTTP; W1.4e).
        # W1.5 mirror (write path): an embedding failure must degrade to an
        # unembedded write, not error the memory_store tool — agent images
        # deliberately ship WITHOUT sentence-transformers, so a provider that
        # resolves "local" raises ImportError there.
        new_embedding = await self._embed_or_none(new_memory.content)

        # Taken AFTER embedding (a network call we do not want to hold a lock
        # across) and BEFORE the read-then-write window that actually races.
        await self._lock_user_writes(user_id)

        # Step 2+3: Search for similar memories and filter candidates
        # (no vector → nothing to compare against; go straight to create)
        if new_embedding is None:
            candidates = []
        else:
            candidates = await self._find_candidates(new_memory, new_embedding, user_id)

        if not candidates:
            # Nothing similar enough to consider
            memory = await self.memory_service.create_memory(
                user_id=user_id,
                memory_data=new_memory,
                deduplicate=False,  # We already checked
                embedding=new_embedding
            )
            logger.info(f"Created new memory: {memory.id}")
            return memory, "created"

        # Step 4: Adjudicate against EVERY candidate above the threshold
        # (thresholds shortcut the LLM for the obvious cases; W1.4d).
        # Rank by SIMILARITY, not by the caller's blended relevance score.
        # find_similar_memories sorts by `final_score`, in which similarity is
        # only 40% of the weight against strength (0.25), importance (0.2),
        # emotional_salience (0.15) and a recency boost. Since create_memory
        # always sets strength=1.0, a hot, frequently-retrieved row could
        # outrank a genuinely nearer duplicate by a wide margin and take the
        # top slot — so dedup read `similarity_score` off the WRONG row, landed
        # in the ambiguous band, and asked the LLM to compare a pair that was
        # never the closest one. The real duplicate was never adjudicated.
        #
        # That is self-reinforcing, and it is exactly the profile of the seven
        # near-duplicate Persian-music rows on the founder's tenant
        # (access_count 20/23/12): the more a junk memory is retrieved, the
        # more reliably it hijacks the comparison and shields the true
        # duplicate from ever being seen.
        #
        # Ranking is necessary but NOT sufficient: the ranked list was then
        # truncated to `candidates[0]` and everything else discarded, so the
        # adjudicator only ever saw the single nearest row. Dedup and
        # contradiction detection are the SAME adjudication — _decide_action
        # returns duplicate|merge|contradiction_update|new for one pair — so a
        # stale memory that contradicts the incoming fact but ranks 2nd was
        # never compared against it at all, and the new fact was simply stored
        # alongside it. Both then retrieve, forever, and the agent has two
        # answers to the same question. Now every candidate above
        # CANDIDATE_THRESHOLD is adjudicated and the verdicts are resolved by
        # precedence (see _select_decision).
        candidates = sorted(
            candidates, key=lambda c: c.get('similarity_score', 0), reverse=True
        )[:MAX_ADJUDICATED_CANDIDATES]

        # Deterministic verdicts first; only the ambiguous band reaches the LLM,
        # and all of it goes in ONE call (_llm_decide_actions_batch), so
        # widening the comparison from 1 candidate to N does not multiply calls.
        decisions, pending_positions = await self._plan_adjudication(
            candidates, new_memory.content
        )
        if pending_positions:
            llm_decisions = await self._llm_decide_actions_batch([
                (candidates[pos].get('content', ''), new_memory.content)
                for pos in pending_positions
            ])
            for pos, llm_decision in zip(pending_positions, llm_decisions):
                decisions[pos] = llm_decision

        decision, top_match = self._select_decision(candidates, decisions)

        logger.info(
            "Adjudicated %d candidate(s); winner similarity %.3f: %s",
            len(candidates), top_match.get('similarity_score', 0),
            describe_memory(top_match.get('content', '')),
        )
        logger.info("Dedup decision: %s - %s", decision['action'],
                    describe_memory(decision.get('reason', '')))

        return await self._apply_decision(
            decision=decision,
            top_match=top_match,
            new_memory=new_memory,
            user_id=user_id,
            new_embedding=new_embedding,
        )

    async def smart_create_memories(
        self,
        new_memories: List[MemoryCreate],
        user_id: str,
        *,
        explicit_save: bool = False,
    ) -> List[Tuple[Optional[Memory], str]]:
        """
        Batch variant of smart_create_memory for one turn's extraction (W1.4d).

        Auto-resolvable memories (no candidate, or best similarity >=
        AUTO_DUPLICATE_THRESHOLD / < AUTO_NEW_THRESHOLD) are applied inline
        so later memories in the batch can dedup against them; the ambiguous
        middle band is adjudicated in ONE LLM call instead of one per memory.

        Returns one (memory, action) tuple per input, in input order.
        """
        results: List[Optional[Tuple[Memory, str]]] = [None] * len(new_memories)
        pending: List[Dict[str, Any]] = []

        for i, new_memory in enumerate(new_memories):
            # Same write gate as the singular path. Kept here as well as there
            # because these two functions are reached by different callers and
            # the last time a fix landed on only one of them (candidate sorting)
            # it sat on the path nobody takes for as long as it existed.
            gate_reason = memory_gate_reason(new_memory.content, explicit_save=explicit_save)
            if gate_reason:
                logger.info(
                    "[memory_gate] refused at write (%s): %s",
                    gate_reason, describe_memory(new_memory.content),
                )
                results[i] = (None, f"rejected:{gate_reason}")
                continue

            # W1.5 mirror (write path) — see smart_create_memory
            new_embedding = await self._embed_or_none(new_memory.content)
            # Same read-then-write race as the singular path.
            await self._lock_user_writes(user_id)
            if new_embedding is None:
                candidates = []
            else:
                candidates = await self._find_candidates(new_memory, new_embedding, user_id)

            if not candidates:
                memory = await self.memory_service.create_memory(
                    user_id=user_id,
                    memory_data=new_memory,
                    deduplicate=False,
                    embedding=new_embedding
                )
                logger.info(f"Created new memory: {memory.id}")
                results[i] = (memory, "created")
                continue

            # Rank by SIMILARITY, exactly as smart_create_memory does — see the
            # long comment there. This line was MISSING here, and this is the
            # path production actually uses: agent_runner.py:4731 calls the
            # batch variant on every turn, while the singular one is only
            # reached from the memory_store tool. So the documented fix was
            # live on the path nobody takes and absent from the path everybody
            # takes, and `candidates[0]` was the top BLENDED-score row
            # (similarity is 40% of it, against strength/importance/salience/
            # recency) rather than the nearest duplicate.
            candidates = sorted(
                candidates, key=lambda c: c.get('similarity_score', 0), reverse=True
            )[:MAX_ADJUDICATED_CANDIDATES]

            # …and adjudicate against ALL of them, not just candidates[0] — see
            # the long comment in smart_create_memory. Same reasoning, and this
            # is the path production actually takes.
            decisions, pending_positions = await self._plan_adjudication(
                candidates, new_memory.content
            )

            if not pending_positions:
                # Every candidate resolved without an LLM — apply now so later
                # batch members can dedup against the result.
                decision, top_match = self._select_decision(candidates, decisions)
                results[i] = await self._apply_decision(
                    decision=decision,
                    top_match=top_match,
                    new_memory=new_memory,
                    user_id=user_id,
                    new_embedding=new_embedding,
                )
            else:
                pending.append({
                    "index": i,
                    "candidates": candidates,
                    "decisions": decisions,
                    "pending_positions": pending_positions,
                    "new_memory": new_memory,
                    "embedding": new_embedding,
                })

        if pending:
            # Still exactly ONE adjudication call per turn: every unresolved
            # (candidate, new memory) pair from every pending memory rides in
            # the same request.
            pairs = [
                (p["candidates"][pos].get('content', ''), p["new_memory"].content)
                for p in pending
                for pos in p["pending_positions"]
            ]
            llm_decisions = await self._llm_decide_actions_batch(pairs)

            cursor = 0
            for p in pending:
                for pos in p["pending_positions"]:
                    p["decisions"][pos] = llm_decisions[cursor]
                    cursor += 1
                decision, top_match = self._select_decision(
                    p["candidates"], p["decisions"]
                )
                logger.info("Dedup decision: %s - %s", decision['action'],
                    describe_memory(decision.get('reason', '')))
                results[p["index"]] = await self._apply_decision(
                    decision=decision,
                    top_match=top_match,
                    new_memory=p["new_memory"],
                    user_id=user_id,
                    new_embedding=p["embedding"],
                )

        return results  # type: ignore[return-value]

    async def _lock_user_writes(self, user_id: str) -> None:
        """Serialise this user's memory writes for the rest of the transaction.

        Dedup is read-then-write: find candidates, decide, insert. Two turns for
        the same user running concurrently — the web + mobile case, and rapid
        successive messages — both read BEFORE either inserts, so both see no
        duplicate and both create one. Measured: 4 concurrent writes of "The
        user is a vegetarian" produced 2 rows.

        A transaction-scoped Postgres advisory lock keyed on the user closes it.
        The cost is bounded: only writes for the SAME user serialise, and
        capture already runs off the user-visible reply path, so the wait is
        invisible. Other tenants and other users are unaffected.

        No-op on non-Postgres backends (CI runs SQLite, which has a single
        writer and therefore cannot exhibit the race).

        The wait is BOUNDED, and that is the whole point
        ------------------------------------------------
        This used to be a bare `pg_advisory_xact_lock`, which blocks for as
        long as it takes. It is held across `_find_candidates` (a vector
        search) and `_decide_action` (an LLM call, in the ambiguous
        similarity band), so a queue of same-user writes advances at LLM
        latency — and the engine sets `command_timeout: 30` on the client and
        `statement_timeout: 30000` on the server. Past thirty seconds the
        WAITER ITSELF is killed, asyncpg invalidates the connection, and the
        caller gets

            PendingRollbackError: Can't reconnect until invalid transaction
            is rolled back

        on a session that had done nothing wrong. **The fact is then lost** —
        this path is `memory_store`/the tool path, which the capture outbox
        does not cover.

        That is report §5.8, recorded on 2026-08-03 as a concurrency failure
        seen once and neither reproduced nor explained. It reproduced on
        2026-08-05: 2 of 20 rapid-fire writes lost, both raising exactly that
        error, in a test that ran for 30.55 seconds against a 30-second
        timeout. Connection exhaustion was correctly ruled out at the time
        (peak 18 of max_connections=100) — the pool was never the problem, the
        lock queue was.

        So the wait now degrades instead of dying: poll `pg_try_advisory_xact_lock`
        (which returns immediately and therefore can never be the statement
        that times out) for a budget well under the 30s ceiling, and if the
        lock cannot be had in that time, go ahead WITHOUT it.

        That trade is deliberate. Proceeding unlocked risks a duplicate row,
        which dedup detects on the next write and merges, and which the
        adjudication path exists to resolve. Blocking to the timeout loses the
        user's fact outright, with no retry and no trace beyond a warning. A
        recoverable duplicate beats an unrecoverable loss.
        """
        bind = self.db.bind
        if bind is None or bind.dialect.name != "postgresql":
            return
        try:
            from sqlalchemy import text as _text

            key = f"memwrite:{user_id}"
            deadline = time.monotonic() + _LOCK_WAIT_BUDGET_S
            while True:
                got = (
                    await self.db.execute(
                        _text("SELECT pg_try_advisory_xact_lock(hashtext(:k))"),
                        {"k": key},
                    )
                ).scalar()
                if got:
                    return
                if time.monotonic() >= deadline:
                    logger.warning(
                        "[MEMORY] advisory lock for user=%s not acquired in %.1fs; "
                        "proceeding UNLOCKED — a duplicate is possible and will be "
                        "merged, whereas waiting to the %ss statement timeout would "
                        "lose the fact",
                        str(user_id)[:8], _LOCK_WAIT_BUDGET_S, 30,
                    )
                    return
                await asyncio.sleep(_LOCK_POLL_INTERVAL_S)
        except Exception as exc:  # pragma: no cover - lock is best-effort
            logger.warning("[MEMORY] advisory lock unavailable (%s); "
                           "concurrent writes may duplicate", exc)

    async def _embed_or_none(self, content: str) -> Optional[List[float]]:
        """Embed content, degrading to None when the embedding backend is
        unavailable (missing sentence-transformers on agent images, transient
        proxy/OpenAI outage). Broad except on purpose — same reasoning as the
        W1.5 read-path degrade in MemoryService.hybrid_search: a memory WRITE
        must never fail because the vector could not be computed."""
        try:
            return await self.embedding_service.embed_async(content, api_key=self.api_key)
        except Exception as e:
            from app.services.embedding_service import record_embed_degrade
            record_embed_degrade(e, site="dedup")
            logger.warning(f"[MEMORY] Embedding failed, skipping dedup and storing without vector: {e}")
            return None

    async def _find_candidates(
        self,
        new_memory: MemoryCreate,
        new_embedding: List[float],
        user_id: str
    ) -> List[Dict[str, Any]]:
        """Search similar memories and filter to candidates above CANDIDATE_THRESHOLD."""
        # Handle brain_type - can be enum or string
        brain_type_value = (
            new_memory.brain_type.value
            if hasattr(new_memory.brain_type, 'value')
            else new_memory.brain_type
        ) if new_memory.brain_type else 'user'

        similar_memories = await self.memory_service.search_memories_by_embedding(
            user_id=user_id,
            embedding=new_embedding,
            limit=MAX_ADJUDICATED_CANDIDATES,
            min_similarity=MIN_THRESHOLD,
            brain_types=[brain_type_value] if brain_type_value else None,
            categories=None  # Search across ALL categories to find related memories
        )
        if not similar_memories:
            return []
        return [m for m in similar_memories if m.get('similarity_score', 0) >= CANDIDATE_THRESHOLD]

    async def _plan_adjudication(
        self,
        candidates: List[Dict[str, Any]],
        new_content: str,
    ) -> Tuple[List[Optional[Dict[str, str]]], List[int]]:
        """Resolve every candidate that the thresholds can settle on their own.

        Returns (decisions, pending_positions) where `decisions` is aligned to
        `candidates` and holds None at each position the LLM still has to
        answer. Splitting the deterministic half out is what lets BOTH entry
        points collect all of a write's (and, in the batch path, all of a
        turn's) unresolved pairs into a single `_llm_decide_actions_batch`
        request instead of one call per candidate.
        """
        decisions: List[Optional[Dict[str, str]]] = []
        pending_positions: List[int] = []
        for pos, candidate in enumerate(candidates):
            decision = await self._decide_action(
                similarity=candidate.get('similarity_score', 0),
                existing_content=candidate.get('content', ''),
                new_content=new_content,
                allow_llm=False,
            )
            decisions.append(decision)
            if decision is None:
                pending_positions.append(pos)
        return decisions, pending_positions

    @staticmethod
    def _select_decision(
        candidates: List[Dict[str, Any]],
        decisions: List[Optional[Dict[str, str]]],
    ) -> Tuple[Dict[str, str], Dict[str, Any]]:
        """Resolve N per-candidate verdicts into the ONE action to apply.

        Precedence: contradiction_update > merge > duplicate > new, ties broken
        by similarity (the caller passes candidates already sorted descending,
        and `max` keeps the first maximum, so a tie resolves to the nearer row).

        Why contradiction wins over duplicate — the interesting disagreement.
        Both verdicts can be returned for the same incoming fact only when two
        STORED rows already disagree with each other, which is rare. In that
        state the two outcomes are:
          * duplicate wins  -> reinforce the near-identical row, drop the
            incoming fact, and leave the contradicted row ACTIVE. The agent
            keeps two conflicting answers to the same question and retrieves
            whichever ranks higher that day. Nothing later fixes it: the next
            restatement adjudicates into the same tie.
          * contradiction wins -> the stale row is retired (is_active=False,
            superseded_by set, content preserved and reversible) and the
            incoming fact is stored.
        The cost of choosing contradiction is a redundant near-duplicate pair
        (the reinforced-instead row plus the new one) — a tidiness problem that
        `find_and_merge_duplicates` exists to sweep, and that the next write on
        the topic re-adjudicates. The cost of choosing duplicate is a wrong
        fact that stays retrievable forever. Correctness outranks tidiness, so
        contradiction_update wins.
        merge > duplicate for the same reason in miniature: merge keeps the
        added detail, duplicate discards it. duplicate > new because a verdict
        of "we already hold this" is exactly what dedup exists to act on.
        `contradiction_update` remains kill-switchable via
        settings.memory_supersede_on_conflict, which still denies the >=0.90
        escalation that produces most of them.
        """
        best = max(
            range(len(candidates)),
            key=lambda pos: (
                _ACTION_PRECEDENCE.get((decisions[pos] or {}).get('action', 'new'), 0),
                candidates[pos].get('similarity_score', 0),
            ),
        )
        decision = decisions[best] or {
            "action": "new", "reason": "no verdict returned for this candidate",
            "source": "default",
        }
        return decision, candidates[best]

    async def _decide_action(
        self,
        similarity: float,
        existing_content: str,
        new_content: str,
        *,
        allow_llm: bool = True,
    ) -> Optional[Dict[str, str]]:
        """Threshold shortcut around the LLM adjudication (W1.4d).

        With allow_llm=False the LLM is never consulted and None is returned
        for pairs that would have needed it, so the caller can batch every such
        pair into one request. That is the only difference: the thresholds, the
        value-conflict guard and the different-subject guard are identical on
        both settings, which is what keeps the batched path and the single-pair
        path from drifting apart.
        """
        if similarity >= AUTO_DUPLICATE_THRESHOLD:
            # D-mem-A (2026-07-29): near-identical vectors are NOT always the
            # same fact. "…passphrase is kestrel-dbf7" restated as
            # "…kestrel-13b4" embeds >= 0.90, and the unconditional
            # "duplicate" verdict here reinforced the OLD row and threw the
            # conflicting new value away — the user could not change a stored
            # fact. When the two texts share their shape but disagree on a
            # value token, fall through to the LLM adjudicator, whose
            # contradiction_update verdict routes to the existing supersede
            # path (_supersede_with_new: new row created, old row gets
            # superseded_by + is_active=False, id trail preserved). True
            # paraphrase duplicates keep the cheap shortcut — a value
            # conflict requires HIGH token overlap plus a differing token,
            # which a mere rewording fails (see _conflicts_on_value).
            # Kill switch: settings.memory_supersede_on_conflict (changes
            # write semantics for restated facts).
            #
            # 2026-07-31: the same shortcut also swallowed facts about
            # DIFFERENT PEOPLE that merely share a sentence shape —
            # "colleague Priya's desk door code is …" vs "Marco … uses door
            # code … for his desk" embed >= 0.90 alike, so Marco's code was
            # reinforced into Priya's row and never stored. The value guard
            # does not catch it (those two are phrased differently enough
            # that token overlap is ~0.28, under its 0.5 floor), so a
            # distinct-subject check runs alongside it. Neither guard
            # decides the outcome — both merely deny the SILENT shortcut and
            # let the adjudicator answer.
            if getattr(settings, "memory_supersede_on_conflict", True) and (
                self._conflicts_on_value(existing_content, new_content)
                or mentions_different_subjects(existing_content, new_content)
            ):
                if not allow_llm:
                    return None
                return await self._llm_decide_action(
                    existing_content=existing_content,
                    new_content=new_content,
                )
            return {
                "action": "duplicate",
                "reason": f"auto: similarity {similarity:.2f} >= {AUTO_DUPLICATE_THRESHOLD}",
                # source=threshold: deterministic and reproducible, and it only
                # gets here with both guards above satisfied. Re-asking an LLM
                # would replace a guarded measurement with a sampled opinion —
                # and an LLM "new" on a true paraphrase creates exactly the
                # near-duplicate rows dedup exists to prevent. Not confirmed.
                "source": "threshold",
            }
        if similarity < AUTO_NEW_THRESHOLD:
            return {
                "action": "new",
                "reason": f"auto: similarity {similarity:.2f} < {AUTO_NEW_THRESHOLD}",
                "source": "threshold",
            }
        if not allow_llm:
            return None
        return await self._llm_decide_action(
            existing_content=existing_content,
            new_content=new_content
        )

    async def _apply_decision(
        self,
        decision: Dict[str, str],
        top_match: Dict[str, Any],
        new_memory: MemoryCreate,
        user_id: str,
        new_embedding: Optional[List[float]] = None
    ) -> Tuple[Memory, str]:
        """Apply an adjudication verdict.

        Every verdict that would make a fact stop being retrievable is
        re-asked on its own first (_confirm_destructive_verdict). This is the
        one choke point all three entry points reach, so the confirmation
        cannot be bypassed by adding a fourth.
        """
        decision = await self._confirm_destructive_verdict(
            decision, top_match, new_memory.content
        )

        if decision['action'] == 'duplicate':
            # Same information, just reinforce
            existing = await self._get_memory_by_id(top_match['id'])
            if existing:
                reinforced = await self._reinforce_existing_memory(existing, new_memory)
                return reinforced, "reinforced"

        elif decision['action'] == 'merge':
            # Related with new info, merge them
            try:
                merged = await self._merge_memories(
                    existing_memory_id=top_match['id'],
                    new_content=new_memory.content,
                    new_memory_data=new_memory,
                    user_id=user_id
                )
                return merged, "merged"
            except Exception as e:
                logger.error(f"Merge failed: {e}, creating new memory instead")

        elif decision['action'] == 'contradiction_update':
            # New info contradicts/supersedes old — create new, mark old as superseded
            try:
                superseded = await self._supersede_with_new(
                    old_memory_id=top_match['id'],
                    new_memory_data=new_memory,
                    user_id=user_id,
                    reason=decision.get('reason', 'Updated information'),
                    new_embedding=new_embedding,
                )
                return superseded, "contradiction_updated"
            except Exception as e:
                logger.error(f"Supersede failed: {e}, creating new memory instead")

        # decision['action'] == 'new' or merge/supersede failed - create new memory
        memory = await self.memory_service.create_memory(
            user_id=user_id,
            memory_data=new_memory,
            deduplicate=False,
            embedding=new_embedding
        )
        logger.info(f"Created new memory: {memory.id}")
        return memory, "created"

    async def _confirm_destructive_verdict(
        self,
        decision: Dict[str, str],
        top_match: Dict[str, Any],
        new_content: str,
    ) -> Dict[str, str]:
        """Re-ask a fact-destroying verdict on its own before acting on it.

        WHY (measured, 2026-08-08 — this is the mechanism behind the silent
        loss in test_i_concurrency.py::test_rapid_fire_writes_lose_nothing).

        Adjudication is batched: every candidate above CANDIDATE_THRESHOLD for
        one write goes into a SINGLE `_llm_decide_actions_batch` request, and
        `_select_decision` then applies the highest-precedence verdict among
        them. Both halves are deliberate — widening from 1 candidate to N
        (2026-07-29) fixed a real bug where a contradicting row that ranked
        2nd was never compared at all. But together they mean ONE verdict out
        of N decides a destructive action, so a per-pair error rate of p
        becomes roughly 1-(1-p)^N per write.

        The error rate is not small, and what drives it is not the facts but
        WHERE the incoming one sits among the candidates it is shown next to.
        Memories that share a sentence template and differ only in a two-digit
        number ("project number 04 … vireo-04"), embedding at 0.91–0.92,
        gpt-4o-mini, 40 trials per cell, "destructive" = the applied verdict
        removed a fact:

            incoming 04 vs {05, 03}   straddled    │ 15/40 and 28/40 (two runs)
            incoming 04 vs {03, 02}   both below   │  0/40
            incoming 02 vs {00, 01}   sequential   │  0/40
            incoming 03 vs {00,01,02} sequential   │  0/40

        Straddled, the model reads the incoming fact as project 05's value
        having CHANGED and answers contradiction_update, which retires the
        existing row. Widening the batch dilutes it — the same straddled pair
        inside 3 candidates measured 4/40, inside 5 candidates 1/40, and asked
        on its own 0/40.

        So the misfire needs TWO things at once: a straddle, and few enough
        candidates that it is not diluted. Sequential writes never straddle;
        a large store never has few candidates. Concurrency produces both,
        because arrival order at the `_lock_user_writes` advisory lock decides
        which neighbours have committed — which is why the same twenty facts
        produce a different store on different runs, and why this needed a
        race to surface at all.

        Prompt drift was NOT the cause, though it existed: re-running the
        straddled two-candidate cell with the batched prompt brought to parity
        with the single-pair prompt's rubric moved it 28% -> 22%, i.e. nothing
        outside sampling noise. What is reliable is asking about ONE pair at a
        time. So the batched call stays the cheap pre-filter it was designed
        to be, and the verdicts that destroy data are confirmed by the
        accurate question.

        Measured end to end on the real write path, three writes where the
        third lands between the first two: 5 of 10 rounds lost a fact before
        this gate, 0 of 10 after. Full protocol against the live model across
        the three batch sizes: 0/120.

        Cost: one extra `memory_extraction_model` call, and only when a
        destructive verdict wins — `new` and `merge` never pay it.

        A refusal is not a downgrade to silence: returning `new` stores the
        incoming fact and leaves the matched row alone, so the outcome of a
        disagreement is that BOTH facts remain retrievable. That is the
        recoverable direction; the unconfirmed verdicts were not.
        """
        action = decision.get("action")
        if action not in _DESTRUCTIVE_ACTIONS:
            return decision

        # Only verdicts the MODEL produced are re-asked. A "threshold" verdict
        # is a similarity measurement past both guards and a "heuristic" one is
        # a string comparison — both deterministic, so a second opinion would
        # substitute noise for a reproducible decision rather than corroborate
        # it. `source` is set explicitly at every return site instead of being
        # sniffed out of `reason`, so a new verdict producer that forgets it
        # gets confirmed (the safe default) rather than silently exempted.
        if decision.get("source") in _DETERMINISTIC_SOURCES:
            return decision

        existing_content = (top_match or {}).get("content") or ""
        if not existing_content:
            return decision

        try:
            confirmation = await self._llm_decide_action(
                existing_content=existing_content,
                new_content=new_content,
            )
        except Exception as e:
            # An unavailable adjudicator must not be able to authorise data
            # loss by failing. Degrade to the outcome that keeps both facts.
            logger.warning(
                "[dedup] could not confirm %s verdict (%s); keeping both facts",
                action, e,
            )
            return {
                "action": "new",
                "reason": f"unconfirmed {action}: adjudicator unavailable ({e})",
            }

        confirmed_action = (confirmation or {}).get("action")
        if confirmed_action == action:
            return decision

        logger.info(
            "[dedup] REFUSED %s: re-asked on its own the adjudicator said %s "
            "— keeping both facts. existing=%s incoming=%s",
            action, confirmed_action,
            describe_memory(existing_content), describe_memory(new_content),
        )
        return {
            "action": "new",
            "reason": (
                f"unconfirmed {action}: re-adjudicated alone as "
                f"{confirmed_action}"
            ),
        }

    async def _get_memory_by_id(self, memory_id: str) -> Optional[Memory]:
        """Get a memory by ID without user check (internal use)."""
        from sqlalchemy import select
        result = await self.db.execute(
            select(Memory).where(Memory.id == memory_id)
        )
        return result.scalar_one_or_none()
    
    async def _reinforce_existing_memory(
        self,
        memory: Memory,
        new_data: MemoryCreate
    ) -> Memory:
        """Reinforce an existing memory with new occurrence.

        This path used to be a one-way ratchet that quietly made decay
        impossible (2026-07-29 audit):
          * `importance` only ever increased, so an item restated a few times
            drifted to the top and stayed there;
          * `consolidation_count` was incremented on every RESTATEMENT even
            though no consolidation had occurred — and DecayService reads that
            field as a stability multiplier worth up to 2x
            (`stability *= 1 + 0.2 * min(consolidation_count, 5)`).
        Combined, a frequently-mentioned throwaway became the single most
        decay-resistant thing in the brain. Both are corrected below.
        """
        # Increase strength (capped at 1.0) — this is the legitimate
        # reinforcement signal and stays as-is.
        memory.strength = min(1.0, (memory.strength or 0.5) + 0.1)

        # Importance now MOVES TOWARD the new observation instead of taking the
        # max, so a memory repeatedly restated as unimportant can drift back
        # down. Weighted to the incumbent so a single odd reading can't
        # discard an established judgement.
        _old_importance = memory.importance if memory.importance is not None else 0.5
        memory.importance = round(
            min(1.0, max(0.0, 0.7 * _old_importance + 0.3 * new_data.importance)), 4
        )

        # Update confidence if the new one is higher
        if new_data.confidence > (memory.confidence or 0.5):
            memory.confidence = new_data.confidence

        # Expiry lease on a restatement:
        #   - restated as DURABLE (no expires_at) -> promote to permanent.
        #     "I'm working on the billing rewrite" (transient, 30d) followed by
        #     "the billing rewrite is my main project" (durable) must not keep
        #     the original lease — dedup returns the incumbent WITHOUT creating
        #     a new row, so this is the only place the promotion can happen.
        #     A later transient restatement can always re-stamp a horizon.
        #   - restated as transient with a LATER horizon -> extend.
        #   - restated as transient with an earlier horizon -> keep the longer.
        if memory.expires_at is not None:
            new_expiry = getattr(new_data, "expires_at", None)
            if new_expiry is None:
                memory.expires_at = None
            elif new_expiry > memory.expires_at:
                memory.expires_at = new_expiry

        # Track reinforcement. NOTE: consolidation_count is deliberately NOT
        # touched here — it means "times the consolidation service merged this
        # into a semantic memory", and decay depends on that meaning. The
        # reinforcement record lives in access_count, last_reinforced_at and
        # history_json below.
        memory.last_reinforced_at = datetime.utcnow()
        memory.access_count = (memory.access_count or 0) + 1
        memory.updated_at = datetime.utcnow()
        
        # Add to history.
        #
        # `content` is the SURVIVING row's text — that is what the rest of the
        # history schema means by it, and _supersede_with_new writes the same
        # field the same way. But a reinforcement is the one action that throws
        # text away: the incoming wording is discarded and the existing row
        # kept. Recording only the survivor made that discard untraceable —
        # after the fact there was no way to tell whether a fact the user
        # stated had been correctly recognised as a restatement or wrongly
        # absorbed, which is precisely the question asked when a memory goes
        # missing. `discarded_content` closes that: the text is preserved
        # whenever it differs from what was kept.
        history = json.loads(memory.history_json) if memory.history_json else []
        entry = {
            "date": datetime.utcnow().isoformat(),
            "content": memory.canonical_content or memory.content,
            "source": new_data.source_type or "conversation",
            "action": "reinforced",
            "change_summary": "Memory reinforced (duplicate occurrence)"
        }
        incoming = (new_data.content or "").strip()
        if incoming and incoming != (memory.content or "").strip():
            entry["discarded_content"] = incoming
        history.append(entry)
        memory.history_json = json.dumps(history)
        
        await self.db.commit()
        await self.db.refresh(memory)
        
        return memory
    
    async def _supersede_with_new(
        self,
        old_memory_id: str,
        new_memory_data: MemoryCreate,
        user_id: str,
        reason: str = "Updated information",
        new_embedding: Optional[List[float]] = None,
    ) -> Memory:
        """
        Mark old memory as superseded and create the replacement.

        Used when new information contradicts the old (e.g., user changed jobs).
        The old memory is kept for history but excluded from active search.

        ATOMIC (2026-07-31). This used to be TWO transactions: create_memory
        committed the replacement, then the old row was deactivated and
        committed separately. A crash, a lost connection or a pgbouncer reset
        in between left BOTH conflicting values active and retrievable — the
        exact state supersede exists to prevent — and two concurrent writers
        each left a stray active duplicate. Now the replacement is only
        FLUSHED (commit=False) and the deactivation rides the same
        transaction, so the pair is all-or-nothing. The deactivation itself is
        an optimistic `UPDATE … WHERE is_active = true`: a concurrent
        superseder wins, and the loser rolls its replacement back and is
        retried by the caller (_apply_decision falls through to a plain
        create) instead of leaving a duplicate behind.
        """
        # Read the old row FIRST — its pre-supersede content is what goes into
        # the history entry, and there is no point writing a replacement for a
        # row that has already gone.
        old_memory = await self._get_memory_by_id(old_memory_id)

        # Create the new memory (flush only — this transaction stays open)
        new_memory = await self.memory_service.create_memory(
            user_id=user_id,
            memory_data=new_memory_data,
            deduplicate=False,
            embedding=new_embedding,
            commit=False,
        )

        try:
            if old_memory is not None:
                now = datetime.utcnow()
                history = json.loads(old_memory.history_json) if old_memory.history_json else []
                history.append({
                    "date": now.isoformat(),
                    "content": old_memory.canonical_content or old_memory.content,
                    "source": "contradiction_update",
                    "action": "superseded",
                    "change_summary": reason,
                    "superseded_by": new_memory.id,
                })

                # Optimistic guard: only the writer that finds the row still
                # active gets to supersede it.
                result = await self.db.execute(
                    sa_update(Memory)
                    .where(and_(Memory.id == old_memory_id, Memory.is_active == True))
                    .values(
                        superseded_by=new_memory.id,
                        is_active=False,
                        strength=max((old_memory.strength or 0.5) * 0.3, 0.1),
                        updated_at=now,
                        history_json=json.dumps(history),
                    )
                    .execution_options(synchronize_session=False)
                )
                if (result.rowcount or 0) == 0:
                    raise SupersedeRaceError(
                        f"memory {old_memory_id} was already superseded by another writer"
                    )

                # Keep the in-session copy consistent with the row we just
                # wrote (synchronize_session=False leaves it stale, and
                # sessions built with expire_on_commit=False would keep
                # serving the pre-update values to this request).
                old_memory.superseded_by = new_memory.id
                old_memory.is_active = False
                old_memory.strength = max((old_memory.strength or 0.5) * 0.3, 0.1)
                old_memory.updated_at = now
                old_memory.history_json = json.dumps(history)

                # Track lineage on the new memory
                merged_from = json.loads(new_memory.merged_from_json) if new_memory.merged_from_json else []
                if old_memory_id not in merged_from:
                    merged_from.append(old_memory_id)
                    new_memory.merged_from_json = json.dumps(merged_from)

            await self.db.commit()
        except Exception:
            # Never leave the replacement behind without the deactivation.
            await self.db.rollback()
            raise

        await self.db.refresh(new_memory)
        # create_memory's own post-commit follow-up, which commit=False skipped
        await self.memory_service._update_brain_stats(user_id)

        if old_memory is not None:
            logger.info(
                f"Superseded memory {old_memory_id} → {new_memory.id}: {reason}"
            )
        return new_memory

    async def _merge_memories(
        self,
        existing_memory_id: str,
        new_content: str,
        new_memory_data: MemoryCreate,
        user_id: str
    ) -> Memory:
        """
        Merge new information into an existing memory.
        
        1. Get the existing memory
        2. Use LLM to generate merged content
        3. Generate change summary
        4. Update memory with new canonical_content and history entry
        5. Re-generate embedding for the merged content
        """
        # Get existing memory
        existing = await self._get_memory_by_id(existing_memory_id)
        if not existing:
            raise ValueError(f"Memory {existing_memory_id} not found")
        
        existing_content = existing.canonical_content or existing.content
        
        # Check if the new content actually adds information
        # If it's essentially the same, just reinforce
        if self._is_same_information(existing_content, new_content):
            return await self._reinforce_existing_memory(existing, new_memory_data)
        
        # Use LLM to merge the contents
        merged_content, change_summary = await self._llm_merge_contents(
            existing_content=existing_content,
            new_content=new_content
        )

        # If merge didn't produce new content, just reinforce
        if merged_content.strip() == existing_content.strip():
            return await self._reinforce_existing_memory(existing, new_memory_data)

        # W1.4e: embed the merged content here (async) instead of letting
        # merge_memory re-embed it with the loop-blocking sync path.
        merged_embedding = await self.embedding_service.embed_async(
            merged_content, api_key=self.api_key
        )

        # Update the memory using memory_service method
        updated = await self.memory_service.merge_memory(
            user_id=user_id,
            memory_id=existing_memory_id,
            new_content=merged_content,
            change_summary=change_summary,
            source_type=new_memory_data.source_type or "merge",
            new_embedding=merged_embedding
        )
        
        # Reconcile the expiry lease, with exactly the rule
        # _reinforce_existing_memory applies. MERGE and REINFORCE are the two
        # outcomes of the same dedup decision, and only one of them honoured the
        # lifetime model: a transient memory ("I'm working on the billing
        # rewrite", 30d) restated as a durable fact ("the billing rewrite is my
        # main project") kept its 30-day lease if the adjudicator happened to
        # choose merge instead of reinforce, and was archived on schedule.
        #   - restated as DURABLE (no expires_at) -> promote to permanent
        #   - restated as transient with a LATER horizon -> extend
        #   - restated as transient with an earlier horizon -> keep the longer
        _dirty = False
        if updated.expires_at is not None:
            _new_expiry = getattr(new_memory_data, "expires_at", None)
            if _new_expiry is None:
                updated.expires_at = None
                _dirty = True
            elif _new_expiry > updated.expires_at:
                updated.expires_at = _new_expiry
                _dirty = True

        # Update importance if new info is significant
        if new_memory_data.importance and new_memory_data.importance > (updated.importance or 0.5):
            updated.importance = new_memory_data.importance
            _dirty = True

        if _dirty:
            await self.db.commit()
            await self.db.refresh(updated)

        return updated
    
    def _is_same_information(self, content1: str, content2: str) -> bool:
        """
        Quick check if two pieces of content have the same information.
        Uses simple normalization and comparison.
        """
        def normalize(s: str) -> str:
            return ' '.join(s.lower().split())
        
        n1, n2 = normalize(content1), normalize(content2)

        # If one contains the other, they might have the same info
        if n1 in n2 or n2 in n1:
            # If lengths are similar, probably same info
            if abs(len(n1) - len(n2)) < 20:
                return True

        return n1 == n2

    @staticmethod
    def _value_tokens(content: str) -> set:
        """Thin alias — the implementation is shared with
        MemoryService.create_memory's own dedup shortcut (memory_service.py)."""
        return value_tokens(content)

    @staticmethod
    def _conflicts_on_value(existing_content: str, new_content: str) -> bool:
        """D-mem-A value-conflict guard. Shared with the REST/MCP create
        surface — see memory_service.conflicts_on_value for the rules and for
        why the containment shortcut no longer gates it."""
        return conflicts_on_value(existing_content, new_content)

    async def _llm_decide_action(
        self,
        existing_content: str,
        new_content: str
    ) -> Dict[str, str]:
        """
        Use LLM to decide what action to take with a potential duplicate/related memory.
        
        This is the "smart" part - instead of relying on similarity thresholds,
        we ask the LLM to understand the semantic relationship.
        
        Returns:
            Dict with 'action' (duplicate|merge|new) and 'reason'
        """
        prompt = f"""You are a memory management system. Compare these two pieces of information and decide what to do.

EXISTING MEMORY:
"{existing_content}"

NEW INFORMATION:
"{new_content}"

Decide ONE of these actions:

1. "duplicate" - The new information says the EXACT SAME FACT, just worded differently.
   Example: "I love pizza" vs "Pizza is my favorite food" → duplicate
   Example: "My name is John" vs "I'm John" → duplicate

2. "merge" - The new information ADDS DETAILS to the SAME SPECIFIC TOPIC.
   Example: "I love pizza" vs "I love pepperoni pizza from Dominos" → merge (both about pizza preference)
   Example: "I play basketball" vs "I play basketball as point guard on Saturdays" → merge (both about basketball)

3. "contradiction_update" - The SAME THING's value has changed, so the old statement is now WRONG.
   The subject must be the SAME specific entity in both statements.
   Example: "I work at Google" vs "I just joined Apple" → contradiction_update
   Example: "I live in NYC" vs "I moved to London" → contradiction_update
   Example: "My goal is to learn Python" vs "I've mastered Python, now learning Rust" → contradiction_update
   Example: "My front door code is 1234" vs "I changed the front door code to 9876" → contradiction_update (same door)

4. "new" - The information is about a DIFFERENT topic OR a DIFFERENT THING, even if worded almost identically.
   Example: "My name is John" vs "I love pizza" → new (name vs food = different topics)
   Example: "I love pizza" vs "I have a dog named Max" → new (food vs pet = different topics)
   Example: "I play basketball" vs "My favorite movie is Inception" → new (sport vs movie = different topics)
   Example: "My birthday is Feb 19" vs "I work as an engineer" → new (birthday vs job = different topics)
   Example: "My front door code is 1234" vs "My garage door code is 9876" → new (TWO DIFFERENT DOORS)
   Example: "My work laptop is a ThinkPad" vs "My personal laptop is a MacBook" → new (two different laptops)

CRITICAL: Just because two facts mention the same person does NOT mean they should merge.
Only merge if they are about the SAME SPECIFIC TOPIC (e.g., both about food, both about sports, both about work).

CRITICAL: DIFFERENT INSTANCES OF THE SAME KIND OF THING ARE "new", NEVER contradiction_update.
Front door vs garage door, work laptop vs personal laptop, home wifi vs office wifi, storage locker
vs gym locker — these are separate facts that must BOTH be kept. Two sentences can share almost every
word and still be about different things; check WHICH THING each one is about before deciding.
Choose contradiction_update ONLY when the SAME named thing's value has changed. When unsure between
"new" and "contradiction_update", choose "new" — contradiction_update RETIRES the existing memory and
the user cannot get it back.

Respond in JSON:
{{
    "action": "duplicate|merge|contradiction_update|new",
    "reason": "Brief explanation"
}}"""

        try:
            response = await self.llm_service.complete_with_json(
                messages=[{"role": "user", "content": prompt}],
                # W1.4a: explicit pin — the service default rides the premium
                # chat model whenever an Anthropic key is present.
                model=settings.memory_extraction_model,
            )

            # Parse response — strip markdown fences if present
            if hasattr(response, 'content'):
                import json as json_module
                import re as re_module
                raw = response.content.strip()
                if raw.startswith("```"):
                    raw = re_module.sub(r"^```(?:json)?\s*", "", raw)
                    raw = re_module.sub(r"\s*```$", "", raw)
                parsed = json_module.loads(raw)
            else:
                parsed = response
            
            action = parsed.get("action", "new")
            reason = parsed.get("reason", "")
            
            # Validate action
            if action not in ["duplicate", "merge", "contradiction_update", "new"]:
                action = "new"
            
            return {"action": action, "reason": reason, "source": "llm"}

        except Exception as e:
            logger.error(f"LLM decision failed: {e}")
            # Fallback: use simple heuristic
            return self._heuristic_decision(existing_content, new_content, error=e)

    def _heuristic_decision(
        self,
        existing_content: str,
        new_content: str,
        *,
        error: Optional[BaseException] = None,
    ) -> Dict[str, str]:
        """Non-LLM fallback verdict — shared by _llm_decide_action's
        except-branch and by _llm_decide_actions_batch's missing-entry filler,
        so the two cannot drift.

        source="heuristic": `_is_same_information` is a normalised text
        comparison (one string contains the other, within 20 characters), not
        a judgement. It is deterministic, so a second opinion adds nothing —
        and it must not be sent to an adjudicator that has just been shown to
        be unreachable.
        """
        if self._is_same_information(existing_content, new_content):
            return {
                "action": "duplicate",
                "reason": "Fallback: text similarity",
                "source": "heuristic",
            }
        return {
            "action": "new",
            "reason": (
                f"Fallback due to error: {error}" if error is not None
                else "Fallback: heuristic"
            ),
            "source": "heuristic",
        }

    async def _llm_decide_actions_batch(
        self,
        pairs: List[Tuple[str, str]]
    ) -> List[Dict[str, str]]:
        """
        Adjudicate ALL ambiguous (existing, new) pairs of a turn in ONE LLM
        call (W1.4d) — the per-pair template is ~483 tokens, so N separate
        calls resend it N times.

        Args:
            pairs: list of (existing_content, new_content) tuples

        Returns:
            One decision dict ({"action", "reason"}) per pair, aligned by index.
        """
        if not pairs:
            return []
        if len(pairs) == 1:
            # Single pair — the battle-tested single-pair prompt wins
            return [await self._llm_decide_action(
                existing_content=pairs[0][0], new_content=pairs[0][1]
            )]

        pair_blocks = []
        for i, (existing_content, new_content) in enumerate(pairs):
            pair_blocks.append(
                f'PAIR {i}:\n'
                f'EXISTING MEMORY:\n"{existing_content}"\n'
                f'NEW INFORMATION:\n"{new_content}"'
            )
        pairs_doc = "\n\n".join(pair_blocks)

        prompt = f"""You are a memory management system. For EACH numbered pair below, compare the two pieces of information and decide what to do.

{pairs_doc}

For each pair, decide ONE of these actions:

1. "duplicate" - The new information says the EXACT SAME FACT, just worded differently.
   Example: "I love pizza" vs "Pizza is my favorite food" → duplicate

2. "merge" - The new information ADDS DETAILS to the SAME SPECIFIC TOPIC.
   Example: "I love pizza" vs "I love pepperoni pizza from Dominos" → merge

3. "contradiction_update" - The SAME THING's value has changed, so the old statement is now WRONG.
   The subject must be the SAME specific entity in both statements.
   Example: "I work at Google" vs "I just joined Apple" → contradiction_update
   Example: "My front door code is 1234" vs "I changed the front door code to 9876" → contradiction_update (same door)

4. "new" - The information is about a DIFFERENT topic OR a DIFFERENT THING, even if worded almost identically.
   Example: "My name is John" vs "I love pizza" → new (name vs food = different topics)
   Example: "My front door code is 1234" vs "My garage door code is 9876" → new (TWO DIFFERENT DOORS)

CRITICAL: Just because two facts mention the same person does NOT mean they should merge.
Only merge if they are about the SAME SPECIFIC TOPIC (e.g., both about food, both about sports, both about work).

CRITICAL: DIFFERENT INSTANCES OF THE SAME KIND OF THING ARE "new", NEVER contradiction_update.
Front door vs garage door, work laptop vs personal laptop, storage locker vs gym locker — separate facts
that must BOTH be kept. Two sentences can share almost every word and still be about different things.
Choose contradiction_update ONLY when the SAME named thing's value has changed; when unsure, choose "new"
— contradiction_update RETIRES the existing memory and the user cannot get it back.

Respond in JSON with EXACTLY one decision per pair, in order:
{{
    "decisions": [
        {{"index": 0, "action": "duplicate|merge|contradiction_update|new", "reason": "Brief explanation"}}
    ]
}}"""

        try:
            response = await self.llm_service.complete_with_json(
                messages=[{"role": "user", "content": prompt}],
                # W1.4a: explicit pin (see _llm_decide_action)
                model=settings.memory_extraction_model,
            )

            if hasattr(response, 'content'):
                import json as json_module
                import re as re_module
                raw = response.content.strip()
                if raw.startswith("```"):
                    raw = re_module.sub(r"^```(?:json)?\s*", "", raw)
                    raw = re_module.sub(r"\s*```$", "", raw)
                parsed = json_module.loads(raw)
            else:
                parsed = response

            # Missing/garbled entries degrade to the heuristic, never crash
            decisions = [
                self._heuristic_decision(existing, new) for existing, new in pairs
            ]
            for item in parsed.get("decisions", []):
                try:
                    idx = int(item.get("index", -1))
                except (TypeError, ValueError):
                    continue
                if not (0 <= idx < len(pairs)):
                    continue
                action = item.get("action", "new")
                if action not in ["duplicate", "merge", "contradiction_update", "new"]:
                    action = "new"
                decisions[idx] = {
                    "action": action,
                    "reason": item.get("reason", ""),
                    # source=llm AND batched: this is the population measured
                    # at up to 70% wrong when the incoming fact straddles its
                    # candidates. A destructive verdict from here is re-asked
                    # on its own before it is applied
                    # (_confirm_destructive_verdict).
                    "source": "llm",
                }
            return decisions

        except Exception as e:
            logger.error(f"Batch LLM adjudication failed: {e}")
            return [
                self._heuristic_decision(existing, new) for existing, new in pairs
            ]

    async def _llm_merge_contents(
        self,
        existing_content: str,
        new_content: str
    ) -> Tuple[str, str]:
        """
        Use LLM to intelligently merge two pieces of information.
        
        Returns:
            Tuple of (merged_content, change_summary)
        """
        prompt = f"""You are merging two pieces of information about the same topic into one coherent memory.

EXISTING MEMORY:
{existing_content}

NEW INFORMATION:
{new_content}

Tasks:
1. Combine these into a single, coherent statement that includes all information from both.
2. Summarize what new information was added.

Respond in this exact JSON format:
{{
    "merged_content": "The combined memory statement",
    "change_summary": "Brief summary of what was added/changed"
}}

Rules:
- Keep it concise but complete
- Don't lose any information from either source
- If there's a contradiction, prefer the new information but note it
- The merged content should read naturally as a single fact/memory
- If the new information doesn't add anything new, just return the existing content"""

        try:
            response = await self.llm_service.complete_with_json(
                messages=[{"role": "user", "content": prompt}],
                # W1.4a: explicit pin (see _llm_decide_action)
                model=settings.memory_extraction_model,
            )

            # Parse JSON response — strip markdown fences if present
            if hasattr(response, 'content'):
                import json as json_module
                import re as re_module
                raw = response.content.strip()
                if raw.startswith("```"):
                    raw = re_module.sub(r"^```(?:json)?\s*", "", raw)
                    raw = re_module.sub(r"\s*```$", "", raw)
                parsed = json_module.loads(raw)
            else:
                parsed = response
            
            merged = parsed.get("merged_content", new_content)
            summary = parsed.get("change_summary", "Updated with new information")
            
            return merged, summary
            
        except Exception as e:
            logger.error(f"LLM merge failed: {e}")
            # Fallback: simple concatenation
            if new_content not in existing_content:
                merged = f"{existing_content} Additionally, {new_content}"
                return merged, f"Added: {new_content[:50]}..."
            return existing_content, "No changes (duplicate)"
    
    async def find_and_merge_duplicates(
        self,
        user_id: str,
        category: Optional[str] = None,
        brain_type: Optional[str] = None,
        dry_run: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Scan existing memories and find/merge duplicates.
        
        Useful for cleaning up existing data.
        
        Args:
            user_id: User whose memories to scan
            category: Optional category filter
            brain_type: Optional brain type filter
            dry_run: If True, just report what would be merged without doing it
            
        Returns:
            List of merge operations (performed or proposed)
        """
        # Get all active memories for user
        memories = await self.memory_service.get_user_memories(
            user_id=user_id,
            category=category,
            brain_type=brain_type,
            is_active=True
        )
        
        merge_operations = []
        processed_ids = set()
        
        for memory in memories:
            if memory.id in processed_ids:
                continue
            
            # Get embedding
            if not memory.embedding_json:
                continue
            embedding = json.loads(memory.embedding_json)
            
            # Find similar memories
            similar = await self.memory_service.search_memories_by_embedding(
                user_id=user_id,
                embedding=embedding,
                limit=10,
                min_similarity=CANDIDATE_THRESHOLD
            )
            
            # Filter out self and already processed
            duplicates = [
                s for s in similar 
                if s['id'] != memory.id and s['id'] not in processed_ids
            ]
            
            if duplicates:
                operation = {
                    "primary_memory": {
                        "id": str(memory.id),
                        "content": memory.canonical_content or memory.content
                    },
                    "duplicates": [
                        {
                            "id": str(d['id']),
                            "content": d.get('content', ''),
                            "similarity": d.get('similarity_score', 0)
                        }
                        for d in duplicates
                    ],
                    "action": "proposed" if dry_run else "merged"
                }
                
                if not dry_run:
                    # Actually merge them
                    for dup in duplicates:
                        try:
                            # Merge duplicate into primary
                            await self._merge_memories(
                                existing_memory_id=memory.id,
                                new_content=dup.get('content', ''),
                                new_memory_data=MemoryCreate(
                                    content=dup.get('content', ''),
                                    category=memory.category,
                                    memory_type=memory.memory_type,
                                    brain_type=BrainType(memory.brain_type) if memory.brain_type else BrainType.USER,
                                    source_type="dedup_merge"
                                ),
                                user_id=user_id
                            )
                            
                            # Mark duplicate as superseded
                            await self.memory_service.supersede_memory(
                                user_id=user_id,
                                old_memory_id=dup['id'],
                                new_memory_id=memory.id
                            )
                            
                            processed_ids.add(dup['id'])
                            logger.info(f"Merged memory {dup['id']} into {memory.id}")
                            
                        except Exception as e:
                            logger.error(f"Failed to merge {dup['id']} into {memory.id}: {e}")
                            operation["errors"] = operation.get("errors", []) + [str(e)]
                
                merge_operations.append(operation)
                processed_ids.add(memory.id)
        
        return merge_operations
    
    async def get_duplicate_report(
        self,
        user_id: str,
        category: Optional[str] = None,
        threshold: float = CANDIDATE_THRESHOLD
    ) -> Dict[str, Any]:
        """
        Generate a report of potential duplicates without making changes.
        
        Args:
            user_id: User whose memories to scan
            category: Optional category filter
            threshold: Similarity threshold for considering duplicates
            
        Returns:
            Report with statistics and duplicate groups
        """
        operations = await self.find_and_merge_duplicates(
            user_id=user_id,
            category=category,
            dry_run=True
        )
        
        total_memories = len(await self.memory_service.get_user_memories(
            user_id=user_id,
            category=category,
            is_active=True
        ))
        
        total_duplicates = sum(len(op.get("duplicates", [])) for op in operations)
        
        return {
            "total_memories": total_memories,
            "duplicate_groups": len(operations),
            "total_duplicates": total_duplicates,
            "potential_reduction": total_duplicates,
            "groups": operations
        }


# Singleton instance
_dedup_service: Optional[MemoryDedupService] = None


def get_memory_dedup_service(db: AsyncSession) -> MemoryDedupService:
    """Get or create a MemoryDedupService instance."""
    return MemoryDedupService(db)
