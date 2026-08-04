"""Memory service - CRUD operations and search for memories"""

import asyncio
import json
import logging
import re
from collections import defaultdict
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
import uuid
import time

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, func, and_, or_, text
from sqlalchemy.orm import aliased, selectinload

from app.config import settings
from app.services.memory_gate import MemoryRejected, sensitive_content_reason
from app.db.models import (
    Memory, Entity, EntityLink, EntityRelationship, BrainStats,
    MemoryEvent, MemoryEventType, memory_relationships,
)
from app.schemas import (
    MemoryCreate, MemoryUpdate, MemoryResponse, MemoryWithScore,
    MemorySearchRequest, MemoryCategory, MemoryType, MemoryLevel, BrainType
)
from app.memory_taxonomy import (
    category_for_relationship,
    humanize_relationship,
    normalize_category,
    normalize_entity_type,
)

# Types that mean "we could not place this". An entity carrying one of these is
# a candidate for upgrade the moment a concrete type shows up.
_VAGUE_ENTITY_TYPES = frozenset({"unknown", "topic", "note", "conversation", ""})

# How the user may be named as one end of a graph edge, before tenant-specific
# display names are added. Mirrors memory_gate._SELF_REFERENTS; kept here as a
# plain list because it goes into a SQL ANY(:selves) bind.
_USER_SELF_NAMES = ["user", "the user", "me", "myself", "self", "owner"]
from app.services.embedding_service import get_embedding_service

# Module-level, deliberately. This file previously had only two FUNCTION-LOCAL
# `logger = logging.getLogger(__name__)` bindings, so any `logger.` call added
# outside those two functions was a NameError waiting for the first user who
# hit that branch. That is not hypothetical: PR #375 shipped exactly this bug
# in decay_service.py and every tap of the Keep button raised NameError in
# production before the commit. One module-level binding removes the trap.
from app.services.memory_log import describe_memory

logger = logging.getLogger(__name__)


def value_tokens(content: str) -> set:
    """Lowercased tokens with EDGE punctuation stripped. Inner punctuation
    survives on purpose — "kestrel-13b4" must stay ONE token, or a value swap
    would read as a shared prefix plus two differing suffixes."""
    return {
        t.strip(".,;:!?\"'()[]{}")
        for t in (content or "").lower().split()
    } - {""}


_NAME_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")

# Capitalised words that are never a subject identity. Kept deliberately
# small (a large stoplist becomes a fifth vocabulary to drift — see
# memory_taxonomy's four-vocabulary problem); everything here appears
# sentence-initially or as a pronoun in real extracted memories.
_NAME_STOPWORDS = frozenset({
    "The", "A", "An", "My", "Our", "I", "He", "She", "They", "We", "You",
    "It", "This", "That", "These", "Those", "There", "Here", "User", "Users",
    "When", "Where", "What", "Who", "Why", "How", "His", "Her", "Their",
    "Its", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
    "Saturday", "Sunday", "January", "February", "March", "April", "May",
    "June", "July", "August", "September", "October", "November", "December",
    "Also", "And", "But", "For", "From", "With", "Every", "Each", "Both",
})


def named_subjects(content: str) -> set:
    """Capitalised words that plausibly name a distinct subject."""
    return {m.group(0) for m in _NAME_RE.finditer(content or "")} - _NAME_STOPWORDS


def mentions_different_subjects(existing_content: str, new_content: str) -> bool:
    """True when two texts each name someone/something the other does not.

    The ≥0.90 "auto-duplicate" shortcut assumes near-identical embeddings
    mean the same fact restated. For a family of facts that share a
    sentence shape but describe DIFFERENT people, that assumption silently
    destroys data:

        "My colleague Priya's desk door code is 1111"
        "Marco from the design team uses door code 7a26 for his desk"

    embed ≥0.90 alike, so the second was reinforced into the first and
    Marco's code was never stored — measured on the memory-quality harness
    (distractor_resistance 0/1, four colleagues seeded, the second one
    unrecallable). `conflicts_on_value` does not catch it either: the two
    sentences are phrased differently enough that token overlap lands
    around 0.28, well under its 0.5 floor.

    Requiring a unique name on BOTH sides is what keeps added-detail
    merging intact: "I love pizza" vs "I love pepperoni pizza from
    Dominos" has a unique name on one side only, so it still merges.

    This does not decide the outcome — it only denies the silent shortcut,
    handing the pair to the adjudicator, which may still answer duplicate,
    merge, contradiction_update ("works at Google" → "joined Apple") or
    new. Code, not prompt prose, because prose does not bind (see
    services/memory_gate).
    """
    subjects_existing = named_subjects(existing_content)
    subjects_new = named_subjects(new_content)
    if not subjects_existing or not subjects_new:
        return False
    return bool(subjects_existing - subjects_new) and bool(subjects_new - subjects_existing)


def conflicts_on_value(existing_content: str, new_content: str) -> bool:
    """True when two near-identical texts look like the SAME fact carrying a
    DIFFERENT value ("…passphrase is kestrel-dbf7" vs "…kestrel-13b4").

    Lives here (not on MemoryDedupService) because BOTH write surfaces need
    it: the dedup service's AUTO_DUPLICATE shortcut *and*
    MemoryService.create_memory's own `_find_similar_memory → _reinforce`
    shortcut, which is what the REST POST /api/memories handler and #379's
    MCP memory_create actually run. memory_dedup_service imports this module,
    so the helper cannot live the other way round.

    Deliberately conservative — the shortcut must keep skipping the LLM for
    plain rewordings:
      - each side must carry at least one token the other lacks; a strict
        subset is ADDED DETAIL ("I play basketball on Saturdays"), never a
        conflicting value, and identical texts trivially have neither;
      - the two token sets must mostly overlap (same sentence shape).
        "I love pizza" vs "pizza is my favorite food" shares 1 of 7 tokens
        and stays on the duplicate shortcut.

    NOTE (2026-07-31): the token check runs INDEPENDENTLY of the
    `_is_same_information` containment shortcut it used to sit behind. That
    shortcut calls two texts identical whenever one normalized string
    CONTAINS the other with a length delta < 20 chars — which is exactly the
    shape of a typo fix or a truncation ("gate code is 991" → "…9911",
    "hunter2" → "hunter22"). Abstaining there sent every one-character
    correction straight back to the reinforce path that swallowed it.
    """
    tokens_existing = value_tokens(existing_content)
    tokens_new = value_tokens(new_content)
    if not tokens_existing or not tokens_new:
        return False
    only_existing = tokens_existing - tokens_new
    only_new = tokens_new - tokens_existing
    if not only_existing or not only_new:
        return False
    union = tokens_existing | tokens_new
    overlap = len(tokens_existing & tokens_new) / len(union)
    return overlap >= 0.5


def extract_temporal_filters(query: str) -> dict:
    """
    Detect temporal references in a query and return date filters.
    
    Parses natural language time expressions and converts them to
    created_after / created_before datetime bounds.
    
    Examples:
        "last week"        → created_after = now - 7 days
        "yesterday"        → created_after = start of yesterday, created_before = start of today
        "in January"       → created_after = Jan 1, created_before = Jan 31
        "3 days ago"       → created_after = now - 3 days
        "since December"   → created_after = Dec 1
        "before I moved"   → {} (can't resolve, returns empty)
    
    Returns:
        {
            "created_after": datetime | None,
            "created_before": datetime | None,
            "temporal_keywords": list[str],  # matched temporal phrases
            "has_temporal": bool,
        }
    """
    now = datetime.utcnow()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    result = {"created_after": None, "created_before": None, "temporal_keywords": [], "has_temporal": False}
    query_lower = query.lower()
    
    # --- Relative time patterns (most specific first) ---
    
    # "N days/weeks/months ago"
    m = re.search(r'(\d+)\s+days?\s+ago', query_lower)
    if m:
        days = int(m.group(1))
        result["created_after"] = now - timedelta(days=days)
        result["created_before"] = now - timedelta(days=max(0, days - 1))
        result["temporal_keywords"].append(m.group(0))
        result["has_temporal"] = True
        return result
    
    m = re.search(r'(\d+)\s+weeks?\s+ago', query_lower)
    if m:
        weeks = int(m.group(1))
        result["created_after"] = now - timedelta(weeks=weeks)
        result["created_before"] = now - timedelta(weeks=max(0, weeks - 1))
        result["temporal_keywords"].append(m.group(0))
        result["has_temporal"] = True
        return result
    
    m = re.search(r'(\d+)\s+months?\s+ago', query_lower)
    if m:
        months = int(m.group(1))
        year = now.year
        month = now.month - months
        while month <= 0:
            month += 12
            year -= 1
        result["created_after"] = now.replace(year=year, month=month, day=1, hour=0, minute=0, second=0, microsecond=0)
        result["temporal_keywords"].append(m.group(0))
        result["has_temporal"] = True
        return result
    
    # "today"
    if re.search(r'\btoday\b', query_lower):
        result["created_after"] = today_start
        result["temporal_keywords"].append("today")
        result["has_temporal"] = True
        return result
    
    # "yesterday"
    if re.search(r'\byesterday\b', query_lower):
        result["created_after"] = today_start - timedelta(days=1)
        result["created_before"] = today_start
        result["temporal_keywords"].append("yesterday")
        result["has_temporal"] = True
        return result
    
    # "last week"
    if re.search(r'\blast\s+week\b', query_lower):
        # Last Monday through last Sunday
        days_since_monday = now.weekday()
        this_monday = today_start - timedelta(days=days_since_monday)
        last_monday = this_monday - timedelta(weeks=1)
        result["created_after"] = last_monday
        result["created_before"] = this_monday
        result["temporal_keywords"].append("last week")
        result["has_temporal"] = True
        return result
    
    # "this week"
    if re.search(r'\bthis\s+week\b', query_lower):
        days_since_monday = now.weekday()
        this_monday = today_start - timedelta(days=days_since_monday)
        result["created_after"] = this_monday
        result["temporal_keywords"].append("this week")
        result["has_temporal"] = True
        return result
    
    # "last month"
    if re.search(r'\blast\s+month\b', query_lower):
        first_of_this_month = today_start.replace(day=1)
        if now.month == 1:
            first_of_last_month = first_of_this_month.replace(year=now.year - 1, month=12)
        else:
            first_of_last_month = first_of_this_month.replace(month=now.month - 1)
        result["created_after"] = first_of_last_month
        result["created_before"] = first_of_this_month
        result["temporal_keywords"].append("last month")
        result["has_temporal"] = True
        return result
    
    # "this month"
    if re.search(r'\bthis\s+month\b', query_lower):
        result["created_after"] = today_start.replace(day=1)
        result["temporal_keywords"].append("this month")
        result["has_temporal"] = True
        return result
    
    # "last N days/weeks"
    m = re.search(r'(?:in\s+the\s+)?last\s+(\d+)\s+days?', query_lower)
    if m:
        result["created_after"] = now - timedelta(days=int(m.group(1)))
        result["temporal_keywords"].append(m.group(0))
        result["has_temporal"] = True
        return result
    
    m = re.search(r'(?:in\s+the\s+)?last\s+(\d+)\s+weeks?', query_lower)
    if m:
        result["created_after"] = now - timedelta(weeks=int(m.group(1)))
        result["temporal_keywords"].append(m.group(0))
        result["has_temporal"] = True
        return result
    
    # "recently" / "recent"
    if re.search(r'\brecent(?:ly)?\b', query_lower):
        result["created_after"] = now - timedelta(days=7)
        result["temporal_keywords"].append("recently")
        result["has_temporal"] = True
        return result
    
    # "since <month>" / "after <month>"
    months_map = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
        "jan": 1, "feb": 2, "mar": 3, "apr": 4,
        "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9,
        "oct": 10, "nov": 11, "dec": 12,
    }
    
    # "since January" / "after February" / "from March"
    m = re.search(r'(?:since|after|from)\s+(' + '|'.join(months_map.keys()) + r')(?:\s+(\d{4}))?', query_lower)
    if m:
        month_num = months_map[m.group(1)]
        year = int(m.group(2)) if m.group(2) else now.year
        # If month is in the future this year, assume last year
        if year == now.year and month_num > now.month:
            year -= 1
        result["created_after"] = datetime(year, month_num, 1)
        result["temporal_keywords"].append(m.group(0))
        result["has_temporal"] = True
        return result
    
    # "in January" / "in February 2025" / "during March"
    m = re.search(r'(?:in|during)\s+(' + '|'.join(months_map.keys()) + r')(?:\s+(\d{4}))?', query_lower)
    if m:
        month_num = months_map[m.group(1)]
        year = int(m.group(2)) if m.group(2) else now.year
        # If month is in the future this year, assume last year
        if year == now.year and month_num > now.month:
            year -= 1
        result["created_after"] = datetime(year, month_num, 1)
        # End of that month
        if month_num == 12:
            result["created_before"] = datetime(year + 1, 1, 1)
        else:
            result["created_before"] = datetime(year, month_num + 1, 1)
        result["temporal_keywords"].append(m.group(0))
        result["has_temporal"] = True
        return result
    
    # "before <month>"
    m = re.search(r'before\s+(' + '|'.join(months_map.keys()) + r')(?:\s+(\d{4}))?', query_lower)
    if m:
        month_num = months_map[m.group(1)]
        year = int(m.group(2)) if m.group(2) else now.year
        result["created_before"] = datetime(year, month_num, 1)
        result["temporal_keywords"].append(m.group(0))
        result["has_temporal"] = True
        return result
    
    # "last year"
    if re.search(r'\blast\s+year\b', query_lower):
        result["created_after"] = datetime(now.year - 1, 1, 1)
        result["created_before"] = datetime(now.year, 1, 1)
        result["temporal_keywords"].append("last year")
        result["has_temporal"] = True
        return result
    
    # "this year"
    if re.search(r'\bthis\s+year\b', query_lower):
        result["created_after"] = datetime(now.year, 1, 1)
        result["temporal_keywords"].append("this year")
        result["has_temporal"] = True
        return result
    
    return result


class MemoryService:
    """Service for memory CRUD operations and search"""
    
    def __init__(self, db: AsyncSession, api_key: Optional[str] = None):
        self.db = db
        self.api_key = api_key
        self.embedding_service = get_embedding_service()
    
    async def create_memory(
        self,
        user_id: str,
        memory_data: MemoryCreate,
        source_message_id: Optional[str] = None,
        deduplicate: bool = True,
        similarity_threshold: float = 0.9,
        embedding: Optional[List[float]] = None,
        commit: bool = True,
    ) -> Memory:
        """Create a new memory with embedding, with deduplication support.

        Args:
            user_id: The user ID
            memory_data: Memory creation data
            source_message_id: Optional source message ID
            deduplicate: If True, check for similar memories first
            similarity_threshold: Minimum similarity to consider as duplicate (default 0.9)
            embedding: Pre-computed embedding for memory_data.content (optional —
                W1.4e: the dedup path already embedded the same string; passing
                it here avoids a second identical embedding call)
            commit: When False, the row is FLUSHED (so it has an id) but the
                transaction is left open for the caller to commit — the caller
                also owns _link_memories/_update_brain_stats, both of which
                commit. Used by MemoryDedupService._supersede_with_new so that
                "new row created" and "old row deactivated" land in ONE
                transaction instead of two (a crash between the two commits
                left both conflicting values active and retrievable).

        Returns:
            Memory: The created or reinforced memory

        Raises:
            MemoryRejected: the content carries a never-store secret.
        """
        # ── Storage backstop ─────────────────────────────────────────────
        #
        # `smart_create_memory` screens with the full gate, but EIGHT callers
        # reach this function directly and bypass it entirely — including
        # `mcp_server.create_memory`, which is model-controlled, and the REST
        # endpoint. This is the last function before the row exists, so the
        # policy that must hold everywhere is enforced here.
        #
        # Deliberately ONLY the never-store tier: payment cards, CVVs,
        # government identity numbers, API keys and bearer tokens. Those are
        # never legitimate in any memory, on any path, however the write was
        # requested — so applying them here cannot contradict a decision made
        # upstream.
        #
        # The rest of the gate is NOT applied here on purpose. Length and
        # scaffolding would break document ingestion, which legitimately stores
        # long RAG chunks; quoted-content and echo need the conversation turn,
        # which this layer does not have; and the passphrase tier depends on
        # whether the user explicitly asked, which only the caller knows. Those
        # rules stay at the boundary that has the context to apply them.
        _content = (memory_data.content or "")
        _secret = sensitive_content_reason(_content, explicit_save=True)
        if _secret:
            import logging
            logging.getLogger(__name__).warning(
                "[memory_gate] storage backstop refused (%s) for user=%s",
                _secret, user_id,
            )
            raise MemoryRejected(_secret)

        # Generate embedding (unless the caller already computed it).
        # W1.5 mirror (write path): an embedding failure must never error the
        # memory write — agent images deliberately ship WITHOUT
        # sentence-transformers, so a provider that resolves "local" raises
        # ImportError, and the proxy/OpenAI path can fail transiently. Store
        # the memory unembedded instead; keyword/graph retrieval still finds
        # it, and dedup (which needs the vector) is skipped below.
        if embedding is None:
            try:
                embedding = await self.embedding_service.embed_async(memory_data.content, api_key=self.api_key)
            except Exception as e:
                import logging
                from app.services.embedding_service import record_embed_degrade
                record_embed_degrade(e)
                logging.getLogger(__name__).warning(
                    f"[MEMORY] Embedding failed, storing memory without vector: {e}"
                )
                embedding = None

        # Deduplication: Check for similar existing memories
        # Resolve brain_type BEFORE the dedup check — the check must be scoped
        # to the same brain, or an agent-brain write can be absorbed into a
        # similar user-brain row.
        _brain_type_for_dedup = (
            memory_data.brain_type.value
            if hasattr(memory_data.brain_type, 'value')
            else memory_data.brain_type
        ) if memory_data.brain_type else 'user'

        # (needs the vector — skipped when embedding degraded to None)
        if deduplicate and embedding is not None:
            similar_memory = await self._find_similar_memory(
                user_id=user_id,
                embedding=embedding,
                threshold=similarity_threshold,
                brain_type=_brain_type_for_dedup,
            )
            if similar_memory:
                # D-mem-A on the REST/MCP surface (2026-07-31). This shortcut
                # is a SECOND, independent copy of the dedup service's
                # AUTO_DUPLICATE rule: POST /api/memories (and #379's MCP
                # memory_create, which lands on that handler through the
                # platform proxy) calls create_memory with the defaults, so it
                # never sees MemoryDedupService._decide_action's guard.
                # _reinforce_memory never reads new_data.content, so a >=0.90
                # similar row whose VALUE disagrees ("…kestrel-dbf7" restated
                # as "…kestrel-13b4") had the correction silently discarded.
                #
                # On a value conflict, fall THROUGH to the normal insert: the
                # correction is stored and retrievable. Deliberately NOT an
                # automatic supersede — this path has no adjudicator, and
                # conflicts_on_value alone cannot tell a corrected value from
                # two different instances of the same kind of fact ("front
                # door code" vs "garage door code"), where deactivating the
                # old row would destroy an unrelated, unrecoverable fact.
                # Kill switch: settings.memory_supersede_on_conflict.
                _existing_text = similar_memory.canonical_content or similar_memory.content or ""
                # Same two guards as the dedup service's shortcut: a
                # differing value, or two different named subjects sharing a
                # sentence shape (colleague Priya's code vs Marco's).
                if getattr(settings, "memory_supersede_on_conflict", True) and (
                    conflicts_on_value(_existing_text, memory_data.content)
                    or mentions_different_subjects(_existing_text, memory_data.content)
                ):
                    logger.info(
                        "[MEMORY] value conflict with %s — storing correction as a "
                        "new row instead of reinforcing",
                        similar_memory.id,
                    )
                else:
                    # Reinforce existing memory instead of creating duplicate
                    return await self._reinforce_memory(similar_memory, memory_data)

        # Handle memory_level - can be MemoryLevel enum or string
        memory_level_value = (
            memory_data.memory_level.value
            if hasattr(memory_data.memory_level, 'value')
            else memory_data.memory_level
        )

        # Handle brain_type - can be BrainType enum or string
        brain_type_value = (
            memory_data.brain_type.value 
            if hasattr(memory_data.brain_type, 'value') 
            else memory_data.brain_type
        ) if memory_data.brain_type else 'user'
        
        # Handle category - can be string or enum. Normalised against the
        # canonical taxonomy so a legacy value ("schedule", "projects") coming
        # from an older client or a stale prompt lands on a category the app
        # can actually label, instead of silently rendering as "Other".
        category_value = normalize_category(
            memory_data.category.value
            if hasattr(memory_data.category, 'value')
            else memory_data.category,
            brain_type=brain_type_value,
        )

        # Handle memory_type - can be MemoryType enum or string
        memory_type_value = (
            memory_data.memory_type.value 
            if hasattr(memory_data.memory_type, 'value') 
            else memory_data.memory_type
        )
        
        memory = Memory(
            id=str(uuid.uuid4()),
            user_id=user_id,
            content=memory_data.content,
            summary=memory_data.summary,
            brain_type=brain_type_value,
            category=category_value,
            memory_type=memory_type_value,
            importance=memory_data.importance,
            confidence=memory_data.confidence,
            # NEW: Memory enhancement fields
            strength=1.0,  # Start at full strength
            memory_level=memory_level_value,
            emotional_salience=memory_data.emotional_salience,
            last_reinforced_at=datetime.utcnow(),
            consolidation_count=0,
            decay_rate=0.1,  # Default decay rate
            # Embedding and metadata
            embedding=embedding,  # Native pgvector column (nullable — see degrade above)
            # Backward compat (DEPRECATED). None (not "null") when unembedded so
            # `if memory.embedding_json:` guards in the fallback searches skip it.
            embedding_json=json.dumps(embedding) if embedding is not None else None,
            tags_json=json.dumps(memory_data.tags) if memory_data.tags else None,
            metadata_json=json.dumps(memory_data.metadata) if memory_data.metadata else None,
            source_message_id=source_message_id,
            # NULL unless the extractor flagged this memory transient — see
            # app.memory_taxonomy.resolve_ttl_days, which refuses to expire
            # durable-fact categories even on a mislabel.
            expires_at=getattr(memory_data, "expires_at", None),
            source_type=memory_data.source_type or ("conversation" if source_message_id else "manual"),
            # Memory Evolution fields
            canonical_content=memory_data.content,  # Initially same as content
            history_json=json.dumps([{
                "date": datetime.utcnow().isoformat(),
                "content": memory_data.content,
                "source": memory_data.source_type or ("conversation" if source_message_id else "manual"),
                "action": "created"
            }]),
            merged_from_json=json.dumps([]),
            is_active=True,
        )
        
        self.db.add(memory)
        await self.db.flush()  # Get the ID before committing
        
        # Log CREATED event to audit trail
        await self._log_memory_event(
            memory_id=memory.id,
            user_id=user_id,
            event_type=MemoryEventType.CREATED,
            event_data={
                "category": memory.category,
                "memory_type": memory.memory_type,
                "memory_level": memory.memory_level,
                "importance": memory.importance,
                "source_type": memory.source_type,
            },
            trigger_source="api"
        )

        if not commit:
            # Caller owns the transaction (see the `commit` arg): the row is
            # flushed and carries its id, but nothing is durable yet. Skip the
            # two follow-ups as well — both commit.
            return memory

        await self.db.commit()
        await self.db.refresh(memory)

        # Link related memories
        if memory_data.related_memory_ids:
            await self._link_memories(memory.id, memory_data.related_memory_ids)

        # Update brain stats
        await self._update_brain_stats(user_id)

        return memory
    
    async def _relationship_was_forgotten(
        self,
        user_id: str,
        *relationship_contents: str,
    ) -> bool:
        """True if this relationship memory was soft-deleted by the user.

        Deletion is soft and there is no restore route, so re-creating a
        forgotten memory is effectively irreversible from the user's side.
        Exact content match is deliberate: relationship memories are generated
        from a fixed template, so the same edge always produces the same string.

        Takes SEVERAL candidate strings because that template changed. Rows
        written before `humanize_relationship` carry the raw predicate form
        ("Better Call Saul play on Netflix"); rows written after carry prose
        ("Better Call Saul is available on Netflix"). Matching only the current
        form would silently resurrect every relationship a user forgot before
        the change — the exact irreversibility this guard exists to prevent.
        """
        candidates = [c for c in relationship_contents if c]
        if not candidates:
            return False
        result = await self.db.execute(
            select(Memory.id).where(
                and_(
                    Memory.user_id == user_id,
                    Memory.brain_type == "user",
                    Memory.source_type == "entity_extraction",
                    Memory.content.in_(candidates),
                    Memory.is_deleted == True,
                )
            ).limit(1)
        )
        return result.scalar_one_or_none() is not None

    async def _find_similar_memory(
        self,
        user_id: str,
        embedding: List[float],
        threshold: float = 0.9,
        brain_type: Optional[str] = None,
    ) -> Optional[Memory]:
        """
        Find an existing memory that is very similar to the new one.
        Uses pgvector for fast similarity search.

        Args:
            user_id: User ID
            embedding: Embedding of the new memory content
            threshold: Minimum similarity to consider as duplicate
            brain_type: Scope the search to one brain. REQUIRED in practice —
                without it a new agent-brain memory can match a semantically
                similar user-brain row and reinforce that instead of being
                stored, silently keeping the agent brain empty.

        Returns:
            The most similar existing memory if above threshold, else None
        """
        brain_filter = [Memory.brain_type == brain_type] if brain_type else []

        # Use pgvector if the Memory model has a native embedding column
        if hasattr(Memory, 'embedding') and Memory.embedding is not None:
            try:
                # pgvector accepts Python list directly (no string conversion needed)
                embedding_vec = embedding  # raw list for pgvector
                query = (
                    select(
                        Memory,
                        (1 - Memory.embedding.cosine_distance(embedding_vec)).label("similarity")
                    )
                    .where(
                        and_(
                            Memory.user_id == user_id,
                            Memory.is_deleted == False,
                            # D-mem-A companion: superseded/expired rows are
                            # is_active=False. Without this filter a freshly
                            # superseded row could be matched here and
                            # "reinforced" — swallowing the new value into a
                            # row no retrieval path will ever surface again.
                            Memory.is_active == True,
                            Memory.embedding.isnot(None),
                            *brain_filter,
                        )
                    )
                    .order_by(Memory.embedding.cosine_distance(embedding_vec))
                    .limit(1)
                )
                result = await self.db.execute(query)
                row = result.first()
                if row and row.similarity >= threshold:
                    return row.Memory
                return None
            except Exception:
                pass  # Fall back to Python-side search
        
        # Fallback: Python-side cosine similarity (for SQLite or if pgvector fails)
        result = await self.db.execute(
            select(Memory).where(
                and_(
                    Memory.user_id == user_id,
                    Memory.is_deleted == False,
                    # D-mem-A companion — same reasoning as the pgvector
                    # branch above: never dedup against a superseded row.
                    Memory.is_active == True,
                    *brain_filter,
                )
            )
        )
        memories = result.scalars().all()

        best_match = None
        best_similarity = threshold
        
        for memory in memories:
            if memory.embedding_json:
                memory_embedding = json.loads(memory.embedding_json)
                similarity = self.embedding_service.cosine_similarity(
                    embedding, memory_embedding
                )
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = memory
        
        return best_match
    
    async def _reinforce_memory(
        self,
        memory: Memory,
        new_data: MemoryCreate
    ) -> Memory:
        """
        Reinforce an existing memory instead of creating a duplicate.
        
        Updates the memory's strength, importance, and last_reinforced_at.
        
        Args:
            memory: The existing memory to reinforce
            new_data: The new memory data (used to potentially update importance)
            
        Returns:
            The reinforced memory
        """
        # Increase strength (capped at 1.0)
        memory.strength = min(1.0, memory.strength + 0.1)

        # Importance moves TOWARD the new observation rather than only ever
        # climbing. See the matching note in
        # MemoryDedupService._reinforce_existing_memory: the old max() rule
        # combined with consolidation_count inflation made frequently-restated
        # throwaways the most decay-resistant rows in the brain.
        _old_importance = memory.importance if memory.importance is not None else 0.5
        memory.importance = round(
            min(1.0, max(0.0, 0.7 * _old_importance + 0.3 * new_data.importance)), 4
        )

        # Update confidence if the new one is higher
        if new_data.confidence > memory.confidence:
            memory.confidence = new_data.confidence

        # Expiry lease on a restatement — mirrors
        # MemoryDedupService._reinforce_existing_memory: a restatement with no
        # horizon promotes the memory to permanent; one with a later horizon
        # extends it.
        if memory.expires_at is not None:
            _new_expiry = getattr(new_data, "expires_at", None)
            if _new_expiry is None:
                memory.expires_at = None
            elif _new_expiry > memory.expires_at:
                memory.expires_at = _new_expiry

        # Track reinforcement. consolidation_count is NOT incremented here —
        # it is read by DecayService as a stability multiplier and must mean
        # "actually consolidated", not "mentioned again".
        memory.last_reinforced_at = datetime.utcnow()
        memory.access_count += 1
        memory.updated_at = datetime.utcnow()
        
        # Log REINFORCED event
        await self._log_memory_event(
            memory_id=memory.id,
            user_id=memory.user_id,
            event_type=MemoryEventType.REINFORCED,
            event_data={
                "strength": memory.strength,
                "consolidation_count": memory.consolidation_count,
                "importance": memory.importance,
            },
            trigger_source="deduplication"
        )
        
        await self.db.commit()
        await self.db.refresh(memory)
        
        return memory

    async def merge_memory(
        self,
        memory_id: str,
        new_content: str,
        change_summary: str,
        source_type: str = "merge",
        new_embedding: Optional[List[float]] = None,
        *,
        user_id: str,
    ) -> Memory:
        """
        Merge new information into an existing memory, updating history.
        
        Args:
            memory_id: ID of the memory to update
            new_content: The new merged content
            change_summary: Summary of what changed
            source_type: Source of this update (merge, conversation, manual)
            new_embedding: Pre-computed embedding for new content (optional)
            
        Returns:
            The updated memory
        """
        # `user_id` is a REQUIRED keyword argument, not an optional filter.
        # This primitive took an id and nothing else, so it would happily
        # rewrite any row in the table given its uuid — every caller today
        # establishes ownership first, but that is a convention, and a
        # convention is one careless caller away from a cross-tenant write.
        # Making it required means a caller that forgets fails loudly at the
        # call site instead of silently succeeding against someone else's row.
        result = await self.db.execute(
            select(Memory).where(
                and_(Memory.id == memory_id, Memory.user_id == user_id)
            )
        )
        memory = result.scalar_one_or_none()
        if not memory:
            raise ValueError(f"Memory {memory_id} not found")
        
        # Get existing history
        existing_history = json.loads(memory.history_json) if memory.history_json else []
        
        # Create history entry
        history_entry = {
            "date": datetime.utcnow().isoformat(),
            "content": new_content,
            "previous_content": memory.canonical_content or memory.content,
            "source": source_type,
            "action": "updated",
            "change_summary": change_summary
        }
        existing_history.append(history_entry)
        
        # Generate new embedding if not provided
        if new_embedding is None:
            new_embedding = await self.embedding_service.embed_async(new_content, api_key=self.api_key)
        
        # Update memory
        memory.content = new_content
        memory.canonical_content = new_content
        memory.history_json = json.dumps(existing_history)
        memory.embedding_json = json.dumps(new_embedding)  # Backward compat
        if hasattr(memory, 'embedding'):
            memory.embedding = new_embedding  # Native pgvector
        memory.updated_at = datetime.utcnow()
        # A merge IS a real consolidation (two memories became one), so unlike
        # the reinforce path this increment is legitimate — DecayService reads
        # it as genuine stability.
        memory.consolidation_count += 1
        # Drop any stale summary. `summary` was write-once: merge rewrote
        # content while leaving summary untouched, and the card renders summary
        # first — so a merged memory kept displaying its pre-merge text forever.
        memory.summary = None

        # Log UPDATED event
        await self._log_memory_event(
            memory_id=memory.id,
            user_id=memory.user_id,
            event_type=MemoryEventType.UPDATED,
            event_data={
                "action": "merged",
                "change_summary": change_summary,
                "version_count": len(existing_history),
            },
            trigger_source=source_type
        )
        
        await self.db.commit()
        await self.db.refresh(memory)
        
        return memory
    
    async def supersede_memory(
        self,
        old_memory_id: str,
        new_memory_id: str,
        *,
        user_id: str,
    ) -> Memory:
        """
        Mark a memory as superseded by another (after merging).
        
        Args:
            old_memory_id: ID of the memory being superseded
            new_memory_id: ID of the memory that supersedes it
            
        Returns:
            The updated old memory
        """
        # Required keyword, for the reason given on merge_memory. BOTH rows are
        # scoped: superseding across tenants would need only one of them to be
        # a stranger's, and the target is where merged_from is recorded.
        result = await self.db.execute(
            select(Memory).where(
                and_(Memory.id == old_memory_id, Memory.user_id == user_id)
            )
        )
        memory = result.scalar_one_or_none()
        if not memory:
            raise ValueError(f"Memory {old_memory_id} not found")
        
        memory.superseded_by = new_memory_id
        memory.is_active = False
        memory.updated_at = datetime.utcnow()
        
        # Update the target memory's merged_from
        target_result = await self.db.execute(
            select(Memory).where(
                and_(Memory.id == new_memory_id, Memory.user_id == user_id)
            )
        )
        target_memory = target_result.scalar_one_or_none()
        if target_memory:
            merged_from = json.loads(target_memory.merged_from_json) if target_memory.merged_from_json else []
            if old_memory_id not in merged_from:
                merged_from.append(old_memory_id)
                target_memory.merged_from_json = json.dumps(merged_from)
        
        await self.db.commit()
        await self.db.refresh(memory)
        
        return memory
    
    async def get_memory_history(self, memory_id: str, user_id: str) -> Optional[Dict]:
        """
        Get the evolution history of a memory.
        
        Returns:
            Dict with memory details and history, or None if not found
        """
        result = await self.db.execute(
            select(Memory).where(
                and_(
                    Memory.id == memory_id,
                    Memory.user_id == user_id
                )
            )
        )
        memory = result.scalar_one_or_none()
        if not memory:
            return None
        
        history = json.loads(memory.history_json) if memory.history_json else []
        merged_from = json.loads(memory.merged_from_json) if memory.merged_from_json else []
        
        return {
            "memory_id": memory.id,
            "current_content": memory.canonical_content or memory.content,
            "original_content": memory.content,
            "history": history,
            "version_count": len(history) if history else 1,
            "merged_from": merged_from,
            "superseded_by": memory.superseded_by,
            "is_active": memory.is_active,
            "created_at": memory.created_at.isoformat() if memory.created_at else None,
            "last_updated": memory.updated_at.isoformat() if memory.updated_at else None
        }
    
    async def get_user_memories(
        self,
        user_id: str,
        category: Optional[str] = None,
        brain_type: Optional[str] = None,
        is_active: bool = True,
        include_superseded: bool = False
    ) -> List[Memory]:
        """
        Get all memories for a user with optional filters.
        
        Args:
            user_id: User ID
            category: Optional category filter
            brain_type: Optional brain type filter
            is_active: If True, only return active memories
            include_superseded: If True, include superseded memories
            
        Returns:
            List of memories
        """
        conditions = [
            Memory.user_id == user_id,
            Memory.is_deleted == False
        ]
        
        if category:
            conditions.append(Memory.category == category)
        if brain_type:
            conditions.append(Memory.brain_type == brain_type)
        if is_active and not include_superseded:
            conditions.append(Memory.is_active == True)
        
        result = await self.db.execute(
            select(Memory).where(and_(*conditions)).order_by(Memory.created_at.desc())
        )
        return list(result.scalars().all())

    async def get_memory(self, memory_id: str, user_id: str) -> Optional[Memory]:
        """Get a memory by ID"""
        result = await self.db.execute(
            select(Memory).where(
                and_(
                    Memory.id == memory_id,
                    Memory.user_id == user_id,
                    Memory.is_deleted == False
                )
            )
        )
        memory = result.scalar_one_or_none()
        
        if memory:
            # Update access stats
            memory.last_accessed_at = datetime.utcnow()
            memory.access_count += 1
            
            # Log ACCESSED event to audit trail
            await self._log_memory_event(
                memory_id=memory.id,
                user_id=user_id,
                event_type=MemoryEventType.ACCESSED,
                event_data={
                    "access_count": memory.access_count,
                    "strength": memory.strength,
                },
                trigger_source="api"
            )
            
            await self.db.commit()
        
        return memory
    
    async def list_memories(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0,
        brain_type: Optional[str] = None,
        category: Optional[str] = None,
        memory_type: Optional[str] = None,
        min_importance: Optional[float] = None,
        include_inactive: bool = False
    ) -> Tuple[List[Memory], int]:
        """
        List memories with optional filters, ordered by creation date (newest first).
        
        Args:
            user_id: User ID
            limit: Maximum number of results
            offset: Offset for pagination
            brain_type: Optional brain type filter
            category: Optional category filter
            memory_type: Optional memory type filter
            min_importance: Optional minimum importance filter
            include_inactive: Include superseded/merged memories
            
        Returns:
            Tuple of (memories list, total count)
        """
        # Build conditions - only show active memories by default
        conditions = [
            Memory.user_id == user_id,
            Memory.is_deleted == False
        ]
        
        if not include_inactive:
            conditions.append(Memory.is_active == True)
        
        if brain_type:
            conditions.append(Memory.brain_type == brain_type)
        if category:
            conditions.append(Memory.category == category)
        if memory_type:
            conditions.append(Memory.memory_type == memory_type)
        if min_importance is not None:
            conditions.append(Memory.importance >= min_importance)
        
        # Count total
        count_query = select(func.count(Memory.id)).where(and_(*conditions))
        count_result = await self.db.execute(count_query)
        total_count = count_result.scalar() or 0
        
        # Fetch memories
        query = (
            select(Memory)
            .where(and_(*conditions))
            .order_by(Memory.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self.db.execute(query)
        memories = list(result.scalars().all())
        
        return memories, total_count

    async def _resolve_active_head(self, memory: Memory, user_id: str) -> Memory:
        """Walk `superseded_by` from an archived row to the live one.

        Dedup archives a superseded row (is_active=False + superseded_by) and
        the reinforce path already returns the SUCCESSOR's id, so an id that
        was handed out before a supersede must keep resolving to the same
        logical memory afterwards — otherwise an edit lands on a tombstone no
        retrieval path reads.

        Bounded and cycle-safe: superseded_by is a plain column with no FK
        guarantee of acyclicity, and a self-reference or an A→B→A pair would
        otherwise spin forever. Any dead end (missing row, deleted successor,
        loop) returns the best row reached so far — strictly no worse than
        today's behaviour, never an exception on the write path.
        """
        seen = {memory.id}
        current = memory
        for _ in range(10):
            if current.is_active or not current.superseded_by:
                return current
            successor_id = current.superseded_by
            if successor_id in seen:
                logger.warning(
                    "[MEMORY] supersede cycle at %s — updating in place", current.id
                )
                return current
            seen.add(successor_id)
            result = await self.db.execute(
                select(Memory).where(
                    and_(
                        Memory.id == successor_id,
                        Memory.user_id == user_id,
                        Memory.is_deleted == False,
                    )
                )
            )
            successor = result.scalar_one_or_none()
            if successor is None:
                logger.warning(
                    "[MEMORY] superseded_by %s missing for %s — updating in place",
                    successor_id, current.id,
                )
                return current
            logger.info(
                "[MEMORY] update on superseded %s redirected to active head %s",
                current.id, successor.id,
            )
            current = successor
        logger.warning(
            "[MEMORY] supersede chain too long from %s — updating %s in place",
            memory.id, current.id,
        )
        return current

    async def update_memory(
        self,
        memory_id: str,
        user_id: str,
        update_data: MemoryUpdate
    ) -> Optional[Memory]:
        """Update a memory"""
        memory = await self.get_memory(memory_id, user_id)
        if not memory:
            return None

        # Follow the supersede chain to the row retrieval can actually see.
        # get_memory filters is_deleted ONLY, so an update against a
        # superseded id (the model routinely holds an id from an earlier turn
        # or an earlier search) used to succeed against a tombstone: the write
        # returned 200, and retrieval — which filters is_active — kept serving
        # the SUCCESSOR's old value forever, with no surface that could show
        # the user their edit had gone nowhere.
        memory = await self._resolve_active_head(memory, user_id)

        update_dict = update_data.model_dump(exclude_unset=True)
        old_values = {}  # Track changes for audit

        # Update content and regenerate embedding if content changed
        if "content" in update_dict:
            old_values["content"] = memory.content[:100]  # Truncate for audit
            # W1.5 mirror (write path) + W1.4e: the old sync
            # `embedding_service.embed(...)` blocked the event loop and raised
            # on agent images without sentence-transformers — the ONE write
            # path that could still fail on a missing vector. Degrade to a
            # content-only update instead of 500ing.
            embedding = None
            try:
                embedding = await self.embedding_service.embed_async(
                    update_dict["content"], api_key=self.api_key
                )
            except Exception as e:
                from app.services.embedding_service import record_embed_degrade
                record_embed_degrade(e, site="update")
                logger.warning(
                    f"[MEMORY] Embedding failed on update, clearing stale vector: {e}"
                )
            if embedding is not None:
                memory.embedding_json = json.dumps(embedding)  # Backward compat
                if hasattr(memory, 'embedding'):
                    memory.embedding = embedding  # Native pgvector
            else:
                # 2026-07-31: the degrade used to KEEP the old vector. A vector
                # that describes text the row no longer contains is worse than
                # no vector: search surfaces the row for its pre-update meaning
                # and misses it for the new one, and the repair job
                # (scripts/backfill_embeddings) only re-embeds rows whose
                # embedding IS NULL, so it could never notice. Clearing matches
                # the create-path degrade — keyword/graph retrieval still finds
                # the row, and the backfill can fix it.
                memory.embedding_json = None
                if hasattr(memory, 'embedding'):
                    memory.embedding = None
            memory.content = update_dict["content"]
            # D-mem-A companion: dedup adjudicates against `canonical_content
            # or content` and the mobile card renders `summary || content`, so
            # an update that left canonical_content/summary stale kept BOTH
            # future dedup decisions and the UI anchored to the pre-update
            # text. Same consistency contract as merge_memory.
            memory.canonical_content = update_dict["content"]
            if "summary" not in update_dict:
                memory.summary = None  # stale abbreviation must not outrank new content

        if "summary" in update_dict:
            old_values["summary"] = memory.summary
            memory.summary = update_dict["summary"]
        if "category" in update_dict:
            old_values["category"] = memory.category
            # MemoryUpdate.category is a plain Optional[str] (schemas.py) —
            # the old `.value` access raised AttributeError on EVERY
            # category-carrying update. Normalize through the single taxonomy
            # (brain-scoped) so aliases/retired values degrade instead of 500.
            _raw_category = update_dict["category"]
            memory.category = normalize_category(
                _raw_category.value if hasattr(_raw_category, "value") else _raw_category,
                brain_type=memory.brain_type or "user",
            )
        if "memory_type" in update_dict:
            old_values["memory_type"] = memory.memory_type
            memory.memory_type = update_dict["memory_type"].value
        if "importance" in update_dict:
            old_values["importance"] = memory.importance
            memory.importance = update_dict["importance"]
        if "tags" in update_dict:
            memory.tags_json = json.dumps(update_dict["tags"]) if update_dict["tags"] else None
        if "metadata" in update_dict:
            memory.metadata_json = json.dumps(update_dict["metadata"]) if update_dict["metadata"] else None
        
        # NEW: Handle memory enhancement fields
        if "memory_level" in update_dict:
            old_values["memory_level"] = memory.memory_level
            memory.memory_level = update_dict["memory_level"].value if hasattr(update_dict["memory_level"], 'value') else update_dict["memory_level"]
        if "emotional_salience" in update_dict:
            old_values["emotional_salience"] = memory.emotional_salience
            memory.emotional_salience = update_dict["emotional_salience"]
        
        memory.updated_at = datetime.utcnow()
        
        # Log UPDATED event to audit trail
        await self._log_memory_event(
            memory_id=memory.id,
            user_id=user_id,
            event_type=MemoryEventType.UPDATED,
            event_data={
                "old_values": old_values,
                "fields_updated": list(update_dict.keys()),
            },
            trigger_source="api"
        )
        
        await self.db.commit()
        await self.db.refresh(memory)
        
        return memory
    
    async def delete_memory(self, memory_id: str, user_id: str) -> bool:
        """Soft delete a memory"""
        # First get the memory to log it
        result = await self.db.execute(
            select(Memory).where(
                and_(Memory.id == memory_id, Memory.user_id == user_id)
            )
        )
        memory = result.scalar_one_or_none()
        
        if not memory:
            return False
        
        # Log DELETED event to audit trail BEFORE deleting
        await self._log_memory_event(
            memory_id=memory.id,
            user_id=user_id,
            event_type=MemoryEventType.DELETED,
            event_data={
                "category": memory.category,
                "memory_type": memory.memory_type,
                "content_preview": memory.content[:100],
                "strength_at_deletion": memory.strength,
            },
            trigger_source="api"
        )
        
        # Now perform soft delete
        memory.is_deleted = True
        memory.deleted_at = datetime.utcnow()
        await self.db.commit()
        
        await self._update_brain_stats(user_id)
        return True
    
    async def get_core_facts(
        self,
        user_id: str,
        limit: int = 5,
        *,
        min_importance: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """The user's standing facts, retrieved WITHOUT reference to the query.

        Why this exists
        ---------------
        Everything the model learns about the user used to arrive through
        `hybrid_search`, which is similarity-gated. Measured on
        text-embedding-3-small, that silently loses the facts that matter most:

            "suggest somewhere to eat dinner tonight"
                vs "The user is severely allergic to peanuts."      sim 0.072
            "where do I work?"
                vs "The user is the founder of Toup."               sim 0.125
            "do I eat meat?"
                vs "The user has been vegetarian for eight years."  sim 0.346

        against a production floor of 0.35. The dinner question is the case
        that matters: the agent recommends a restaurant knowing nothing about a
        severe allergy. No threshold fixes it — "convert 400 kelvin to celsius"
        scores 0.055 against the same row, so any floor low enough to catch the
        allergy also drags in unrelated memories on every turn.

        The second channel, the user portrait, could not cover it either:
        PORTRAIT_CATEGORIES had no `health` entry at all, so an allergy reached
        the model through NO path on that turn. (Fixed alongside this, in
        user_portrait_service.)

        A standing constraint is not a search result. It is retrieved by
        standing, not by resemblance — which is exactly what the prompt section
        it feeds ("Core facts about this user") already claims to be.

        Contract, so this cannot become a token regression:
          * bounded — `limit` rows, default 5, hard-capped by the caller
          * JIT — one indexed SELECT per turn, no LLM, no cache to go stale
          * OUTSIDE the cached system-prompt prefix, in the same query-time
            memory block as everything else (STABLE_PREFIX_LAYOUT untouched)
        """
        rows = (
            await self.db.execute(
                select(Memory)
                .where(
                    and_(
                        Memory.user_id == user_id,
                        Memory.brain_type == "user",
                        Memory.is_deleted == False,  # noqa: E712
                        Memory.is_active == True,  # noqa: E712
                        Memory.importance >= min_importance,
                        Memory.memory_type.notin_(["event", "conversation", "task"]),
                        or_(
                            Memory.expires_at.is_(None),
                            Memory.expires_at > datetime.utcnow(),
                        ),
                    )
                )
                .order_by(
                    Memory.importance.desc(),
                    Memory.strength.desc(),
                    Memory.created_at.desc(),
                )
                .limit(limit)
            )
        ).scalars().all()

        return [
            {
                "id": m.id,
                "content": m.content,
                "category": m.category,
                "memory_type": m.memory_type,
                "importance": m.importance,
                "strength": m.strength,
                "brain_type": m.brain_type,
                "created_at": m.created_at.isoformat() if m.created_at else None,
                "similarity_score": 1.0,  # standing, not scored
                "retrieval_reason": "core_fact",
            }
            for m in rows
        ]

    async def forget_all_memories(
        self,
        user_id: str,
        *,
        trigger_source: str = "forget_all",
    ) -> int:
        """Forget everything about one user. Returns how many rows were removed.

        "Forget everything about me" had no executable path anywhere in the
        system: no service method, no agent tool, no endpoint. Account deletion
        cascades by foreign key, but that deletes the account — a user who wants
        their agent to forget them while keeping their account could not be
        served at all.

        Semantics match single-fact deletion so there is one deletion model, not
        two: rows are SOFT-deleted (is_deleted=True), which makes them
        unreachable from every retrieval path and every list endpoint, and each
        one gets a DELETED audit event before it is touched. The derived graph
        is cleared too — an entity map rebuilt from forgotten memories would put
        the forgotten content straight back in front of the model.

        Scoping is by user_id on every statement, so one user's wipe can never
        reach another's rows. That is asserted, not assumed
        (tests/memverify/test_e_updates_forget.py).
        """
        rows = (
            await self.db.execute(
                select(Memory).where(
                    and_(
                        Memory.user_id == user_id,
                        Memory.is_deleted == False,  # noqa: E712
                    )
                )
            )
        ).scalars().all()

        now = datetime.utcnow()
        for memory in rows:
            await self._log_memory_event(
                memory_id=memory.id,
                user_id=user_id,
                event_type=MemoryEventType.DELETED,
                event_data={
                    "category": memory.category,
                    "memory_type": memory.memory_type,
                    "content_preview": memory.content[:100],
                    "strength_at_deletion": memory.strength,
                    "bulk": True,
                },
                trigger_source=trigger_source,
            )
            memory.is_deleted = True
            memory.deleted_at = now
            memory.is_active = False

        # Derived structures. The graph is never gated on the write path, so if
        # it survived a wipe the next entity/graph retrieval would re-surface
        # exactly what the user asked to be forgotten.
        entity_ids = (
            await self.db.execute(
                select(Entity.id).where(Entity.user_id == user_id)
            )
        ).scalars().all()
        if entity_ids:
            await self.db.execute(
                delete(EntityLink).where(EntityLink.entity_id.in_(entity_ids))
            )
        await self.db.execute(
            delete(EntityRelationship).where(EntityRelationship.user_id == user_id)
        )
        await self.db.execute(delete(Entity).where(Entity.user_id == user_id))

        await self.db.commit()
        await self._update_brain_stats(user_id)

        logger.info(
            "[MEMORY] forget_all: removed %d memories for user=%s (source=%s)",
            len(rows), user_id[:8], trigger_source,
        )
        return len(rows)

    async def search_memories(
        self,
        user_id: str,
        request: MemorySearchRequest
    ) -> Tuple[List[MemoryWithScore], int, float]:
        """
        Semantic search for memories with optional filters.
        Returns (results, total_count, search_time_ms)
        """
        start_time = time.time()
        
        # Generate query embedding (async to avoid blocking event loop)
        query_embedding = await self.embedding_service.embed_async(request.query, api_key=self.api_key)

        # Build base query with filters - only search active, non-deleted memories
        query = select(Memory).where(
            and_(
                Memory.user_id == user_id,
                Memory.is_deleted == False,
                Memory.is_active == True  # Only search active memories
            )
        )
        
        # Apply filters
        # Filter by brain_type
        if request.brain_type:
            brain_type_value = (
                request.brain_type.value 
                if hasattr(request.brain_type, 'value') 
                else request.brain_type
            )
            query = query.where(Memory.brain_type == brain_type_value)
        
        if request.categories:
            categories = [c.value for c in request.categories]
            query = query.where(Memory.category.in_(categories))
        
        if request.memory_types:
            types = [t.value for t in request.memory_types]
            query = query.where(Memory.memory_type.in_(types))
        
        # NEW: Filter by memory levels
        if request.memory_levels:
            levels = [l.value for l in request.memory_levels]
            query = query.where(Memory.memory_level.in_(levels))
        
        if request.min_importance is not None:
            query = query.where(Memory.importance >= request.min_importance)
        
        # NEW: Filter by minimum strength (exclude decayed memories)
        if request.min_strength is not None:
            query = query.where(Memory.strength >= request.min_strength)
        
        if request.start_date:
            query = query.where(Memory.created_at >= request.start_date)
        
        if request.end_date:
            query = query.where(Memory.created_at <= request.end_date)
        
        if request.tags:
            # Check if any tag matches (JSON array search)
            for tag in request.tags:
                query = query.where(Memory.tags_json.contains(f'"{tag}"'))
        
        # Try pgvector-accelerated search first
        use_pgvector = hasattr(Memory, 'embedding') and Memory.embedding is not None
        use_weighted = getattr(request, 'use_weighted_scoring', True)
        scored_memories = []
        total_count = 0

        if use_pgvector:
            try:
                # Add pgvector filter
                query = query.where(Memory.embedding.isnot(None))
                # pgvector accepts Python list directly (no string conversion needed)
                embedding_vec = query_embedding  # raw list for pgvector
                
                # Query with pgvector cosine similarity
                pgvector_query = (
                    select(
                        Memory,
                        (1 - Memory.embedding.cosine_distance(embedding_vec)).label("similarity")
                    )
                    .where(query.whereclause)
                    .order_by(Memory.embedding.cosine_distance(embedding_vec))
                    .limit(request.limit * 2)  # Over-fetch for weighted re-ranking
                )
                
                result = await self.db.execute(pgvector_query)
                rows = result.all()
                total_count = len(rows)
                
                for memory, similarity in rows:
                    if similarity < request.min_similarity:
                        continue
                    
                    if use_weighted:
                        final_score = (
                            similarity * 0.4 +
                            memory.strength * 0.25 +
                            memory.importance * 0.2 +
                            memory.emotional_salience * 0.15
                        )
                        if getattr(request, 'boost_recent_access', True) and memory.last_accessed_at:
                            days_since_access = (datetime.utcnow() - memory.last_accessed_at).days
                            if days_since_access < 7:
                                recency_boost = (7 - days_since_access) / 7 * 0.1
                                final_score += recency_boost
                    else:
                        final_score = similarity
                    
                    explanation = None
                    if request.include_explanation:
                        explanation = self._generate_explanation(memory, similarity, final_score if use_weighted else None)
                    
                    scored_memories.append((memory, final_score, similarity, explanation))
                
                scored_memories.sort(key=lambda x: x[1], reverse=True)
                scored_memories = scored_memories[:request.limit]
            except Exception:
                scored_memories = []  # Fall through to Python-side search

        # Fallback: Python-side cosine similarity (SQLite or pgvector failure)
        if not scored_memories and not use_pgvector:
            result = await self.db.execute(query)
            memories = result.scalars().all()
            total_count = len(memories)
            
            for memory in memories:
                if memory.embedding_json:
                    memory_embedding = json.loads(memory.embedding_json)
                    similarity = self.embedding_service.cosine_similarity(
                        query_embedding, memory_embedding
                    )
                    
                    if use_weighted:
                        final_score = (
                            similarity * 0.4 +
                            memory.strength * 0.25 +
                            memory.importance * 0.2 +
                            memory.emotional_salience * 0.15
                        )
                        if getattr(request, 'boost_recent_access', True) and memory.last_accessed_at:
                            days_since_access = (datetime.utcnow() - memory.last_accessed_at).days
                            if days_since_access < 7:
                                recency_boost = (7 - days_since_access) / 7 * 0.1
                                final_score += recency_boost
                    else:
                        final_score = similarity
                    
                    if similarity >= request.min_similarity:
                        explanation = None
                        if request.include_explanation:
                            explanation = self._generate_explanation(memory, similarity, final_score if use_weighted else None)
                        
                        scored_memories.append((memory, final_score, similarity, explanation))
            
            scored_memories.sort(key=lambda x: x[1], reverse=True)
            scored_memories = scored_memories[:request.limit]
        
        # Convert to response objects with new fields
        results = []
        for m, final_score, raw_similarity, explanation in scored_memories:
            results.append(MemoryWithScore(
                id=m.id,
                content=m.content,
                summary=m.summary,
                category=m.category,
                memory_type=m.memory_type,
                importance=m.importance,
                confidence=m.confidence,
                # NEW: Memory enhancement fields in response
                strength=m.strength,
                memory_level=m.memory_level,
                emotional_salience=m.emotional_salience,
                last_reinforced_at=m.last_reinforced_at,
                consolidation_count=m.consolidation_count,
                decay_rate=m.decay_rate,
                # Timestamps
                created_at=m.created_at,
                updated_at=m.updated_at,
                last_accessed_at=m.last_accessed_at,
                access_count=m.access_count,
                source_type=m.source_type,
                tags=json.loads(m.tags_json) if m.tags_json else None,
                metadata=json.loads(m.metadata_json) if m.metadata_json else None,
                similarity_score=raw_similarity,  # Use raw similarity for compatibility
                explanation=explanation
            ))
        
        search_time_ms = (time.time() - start_time) * 1000
        return results, total_count, search_time_ms
    
    async def get_memories_by_category(
        self,
        user_id: str,
        category,
        limit: int = 50,
        offset: int = 0
    ) -> List[Memory]:
        """Get memories by category (only active, non-deleted)"""
        # Handle both enum and string categories
        cat_value = category.value if hasattr(category, 'value') else str(category)
        result = await self.db.execute(
            select(Memory)
            .where(
                and_(
                    Memory.user_id == user_id,
                    Memory.category == cat_value,
                    Memory.is_deleted == False,
                    Memory.is_active == True  # Only active memories
                )
            )
            .order_by(Memory.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        return result.scalars().all()
    
    async def get_memories_by_brain_type(
        self,
        user_id: str,
        brain_type: str = "agent",
        limit: int = 50,
    ) -> List[dict]:
        """Get all active memories for a brain type, returned as dicts."""
        result = await self.db.execute(
            select(Memory)
            .where(
                and_(
                    Memory.user_id == user_id,
                    Memory.brain_type == brain_type,
                    Memory.is_deleted == False,
                    Memory.is_active == True,
                )
            )
            .order_by(Memory.strength.desc(), Memory.created_at.desc())
            .limit(limit)
        )
        memories = result.scalars().all()
        return [
            {
                "id": m.id,
                "content": m.content,
                "category": m.category,
                "brain_type": m.brain_type,
                "strength": m.strength,
                "importance": m.importance,
            }
            for m in memories
        ]

    # Alias for backward compatibility
    async def get_memories_by_region(
        self,
        user_id: str,
        region: MemoryCategory,
        limit: int = 50,
        offset: int = 0
    ) -> List[Memory]:
        """Alias for get_memories_by_category"""
        return await self.get_memories_by_category(user_id, region, limit, offset)
    
    async def get_related_memories(
        self,
        memory_id: str,
        user_id: str,
        depth: int = 1
    ) -> List[Memory]:
        """Get memories related to a given memory (graph traversal)"""
        visited = set()
        to_visit = [memory_id]
        related = []
        
        for _ in range(depth):
            next_level = []
            for mid in to_visit:
                if mid in visited:
                    continue
                visited.add(mid)
                
                # Get directly related memories
                result = await self.db.execute(
                    select(memory_relationships.c.target_id)
                    .where(memory_relationships.c.source_id == mid)
                )
                targets = [row[0] for row in result.fetchall()]
                
                result = await self.db.execute(
                    select(memory_relationships.c.source_id)
                    .where(memory_relationships.c.target_id == mid)
                )
                sources = [row[0] for row in result.fetchall()]
                
                next_level.extend(targets + sources)
            
            to_visit = [m for m in next_level if m not in visited]
        
        # Fetch the actual memory objects
        if visited:
            visited.discard(memory_id)  # Don't include the original
            result = await self.db.execute(
                select(Memory).where(
                    and_(
                        Memory.id.in_(visited),
                        Memory.user_id == user_id,
                        Memory.is_deleted == False
                    )
                )
            )
            related = result.scalars().all()
        
        return related
    
    async def search_by_entity_graph(
        self,
        user_id: str,
        entity_name: str,
        depth: int = 2,
        limit: int = 20,
    ) -> List[dict]:
        """
        Search memories by traversing the entity knowledge graph.
        
        1. Find the entity by name
        2. Find all memories linked to that entity
        3. Find related entities (via shared memories)
        4. Find memories linked to those related entities (depth traversal)
        
        Returns memory dicts sorted by relevance (direct links first).
        """
        # Step 1: Find matching entities
        result = await self.db.execute(
            select(Entity).where(
                and_(
                    Entity.user_id == user_id,
                    func.lower(Entity.name).contains(entity_name.lower()),
                )
            )
        )
        entities = list(result.scalars().all())
        
        if not entities:
            return []
        
        entity_ids = {e.id for e in entities}
        all_memory_ids = set()
        memory_depth = {}  # memory_id -> depth (lower = more relevant)
        
        for current_depth in range(depth):
            # Find memories linked to current entities
            result = await self.db.execute(
                select(EntityLink.memory_id, EntityLink.entity_id).where(
                    EntityLink.entity_id.in_(entity_ids)
                )
            )
            links = result.fetchall()
            
            new_memory_ids = set()
            for memory_id, entity_id in links:
                if memory_id not in all_memory_ids:
                    new_memory_ids.add(memory_id)
                    memory_depth[memory_id] = current_depth
            
            all_memory_ids.update(new_memory_ids)
            
            if current_depth < depth - 1 and new_memory_ids:
                # Find new entities connected through these memories
                result = await self.db.execute(
                    select(EntityLink.entity_id).where(
                        and_(
                            EntityLink.memory_id.in_(new_memory_ids),
                            ~EntityLink.entity_id.in_(entity_ids),
                        )
                    )
                )
                new_entity_ids = {row[0] for row in result.fetchall()}
                entity_ids.update(new_entity_ids)
        
        if not all_memory_ids:
            return []
        
        # Fetch actual memories
        result = await self.db.execute(
            select(Memory).where(
                and_(
                    Memory.id.in_(all_memory_ids),
                    Memory.user_id == user_id,
                    Memory.is_deleted == False,
                    Memory.is_active == True,
                )
            )
        )
        memories = list(result.scalars().all())
        
        # Score and sort: direct links first, then by importance
        scored = []
        for m in memories:
            d = memory_depth.get(m.id, depth)
            score = (1.0 / (1 + d)) * 0.5 + m.importance * 0.3 + m.strength * 0.2
            scored.append({
                "id": m.id,
                "content": m.content,
                "summary": m.summary,
                "brain_type": m.brain_type,
                "category": m.category,
                "importance": m.importance,
                "strength": m.strength,
                "graph_depth": d,
                "graph_score": round(score, 3),
                "created_at": m.created_at.isoformat() if m.created_at else None,
            })
        
        scored.sort(key=lambda x: x["graph_score"], reverse=True)
        return scored[:limit]
    
    async def find_similar_memories(
        self,
        user_id: str,
        content: str,
        limit: int = 5,
        min_similarity: float = 0.7,
        brain_type: Optional[str] = None,
    ) -> List[Tuple[Memory, float]]:
        """Find memories similar to given content. Uses pgvector when available.

        `brain_type` scopes the search to one brain. It matters: this method
        backs `create_memory`'s dedup check, and without the filter a new
        agent-brain memory ("the user corrected me about X") could match and
        silently reinforce a semantically similar USER-brain row instead of
        being stored — which is one reason the agent brain never accumulated
        anything. Callers that genuinely want a cross-brain search pass None.
        """
        embedding = self.embedding_service.embed(content, api_key=self.api_key)

        brain_filter = [Memory.brain_type == brain_type] if brain_type else []

        # Try pgvector-accelerated search
        if hasattr(Memory, 'embedding') and Memory.embedding is not None:
            try:
                # pgvector accepts Python list directly (no string conversion needed)
                embedding_vec = embedding  # raw list for pgvector
                query = (
                    select(
                        Memory,
                        (1 - Memory.embedding.cosine_distance(embedding_vec)).label("similarity")
                    )
                    .where(
                        and_(
                            Memory.user_id == user_id,
                            Memory.is_deleted == False,
                            Memory.embedding.isnot(None),
                            *brain_filter,
                        )
                    )
                    .order_by(Memory.embedding.cosine_distance(embedding_vec))
                    .limit(limit)
                )
                result = await self.db.execute(query)
                rows = result.all()

                similar = []
                for memory, similarity in rows:
                    if similarity >= min_similarity:
                        similar.append((memory, similarity))
                return similar
            except Exception:
                pass  # Fall through to Python-side search

        # Fallback: Python-side cosine similarity
        result = await self.db.execute(
            select(Memory).where(
                and_(
                    Memory.user_id == user_id,
                    Memory.is_deleted == False,
                    *brain_filter,
                )
            )
        )
        memories = result.scalars().all()
        
        similar = []
        for memory in memories:
            if memory.embedding_json:
                memory_embedding = json.loads(memory.embedding_json)
                similarity = self.embedding_service.cosine_similarity(
                    embedding, memory_embedding
                )
                if similarity >= min_similarity:
                    similar.append((memory, similarity))
        
        similar.sort(key=lambda x: x[1], reverse=True)
        return similar[:limit]
    
    async def _user_aliases(self, user_id: str) -> List[str]:
        """Names this user may appear under as one end of a graph edge.

        The extractor writes edges under whatever name the conversation used —
        "User", "user", "Nariman", "Nariman HOSSEINI" — so the gate needs the
        real display name to recognise `Nariman --owns--> Toup` as being about
        the user. Generic self-referents ("user", "me") are handled inside the
        gate; this supplies only the tenant-specific part.

        Cached per service instance and fail-soft: a lookup failure must never
        block a memory write, it just makes the gate slightly stricter.
        """
        cached = getattr(self, "_alias_cache", None)
        if cached is not None:
            return cached

        aliases: List[str] = []
        try:
            from app.db.models.user import User
            row = await self.db.execute(
                select(User.name, User.email).where(User.id == user_id)
            )
            found = row.first()
            if found:
                name, email = found
                if name:
                    aliases.append(name)
                    # "Nariman HOSSEINI" also appears as bare "Nariman".
                    first = name.strip().split()[0] if name.strip() else ""
                    if first and first.lower() != name.strip().lower():
                        aliases.append(first)
                if email:
                    aliases.append(email.split("@")[0])
        except Exception as exc:
            logger.debug("[memory_gate] alias lookup failed (non-fatal): %s", exc)

        self._alias_cache = aliases
        return aliases

    async def _endpoint_in_user_world(
        self, user_id: str, source_name: str, target_name: str
    ) -> bool:
        """True when either endpoint is already ONE HOP from the user.

        This is the graph half of the extraction prompt's rule: an edge counts
        when the user, *or someone/something in the user's life*, is one end.
        A pure string gate can only see the user themselves, so it was
        discarding facts about the people and projects the user had already
        told the agent about.

        MEASURED on the founder's live graph before shipping — 16 entities sit
        within one hop of the user, and against that set this rescue:
          - readmits ZERO of the 15 labelled world-knowledge rows. Better Call
            Saul, Netflix, Drake, Rampage, Shops at Don Mills and Codex are all
            entity-to-entity edges the user was never linked to.
          - rescues ALL of Toup, Majid Tajik, Brother, Sarah Rostami and
            Mohammad — every third-party subject the user actually has a
            relationship with.

        KNOWN LEAK, accepted deliberately: once a user-to-X edge exists, X is
        in the user's world forever, so a later "X is available on Netflix"
        would pass. That is the correct semantics of "something in the user's
        life", and the tautology / scaffolding / agent-self rules still apply
        to those edges. Revisit only with a fresh measurement, never by
        intuition — this rule cost two calibration mistakes already.
        """
        names = [n.strip().lower() for n in (source_name, target_name) if n and n.strip()]
        if not names:
            return False
        selves = _USER_SELF_NAMES + [
            a.lower() for a in await self._user_aliases(user_id)
        ]
        try:
            # ORM rather than raw SQL: the first draft used Postgres-only
            # `= ANY(:names)`, which would have passed in production and failed
            # on the SQLite test path — a prod-only construct is a test that
            # cannot fail for the reason that matters.
            src = aliased(Entity)
            tgt = aliased(Entity)
            q = (
                select(EntityRelationship.id)
                .join(src, src.id == EntityRelationship.source_entity_id)
                .join(tgt, tgt.id == EntityRelationship.target_entity_id)
                .where(
                    and_(
                        EntityRelationship.user_id == user_id,
                        or_(
                            and_(func.lower(src.name).in_(names),
                                 func.lower(tgt.name).in_(selves)),
                            and_(func.lower(tgt.name).in_(names),
                                 func.lower(src.name).in_(selves)),
                        ),
                    )
                )
                .limit(1)
            )
            return (await self.db.execute(q)).first() is not None
        except Exception as exc:
            # Fail CLOSED here: the pure gate already said "no user endpoint",
            # and a broken lookup must not silently readmit world knowledge.
            logger.debug("[memory_gate] user-world lookup failed: %s", exc)
            return False

    async def store_entity_relationship(
        self,
        user_id: str,
        source_name: str,
        source_type: str,
        target_name: str,
        target_type: str,
        relationship: str,
        confidence: float = 0.7,
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Store an entity-entity relationship in the knowledge graph.
        
        Uses the dedicated entity_relationships table for efficient graph traversal.
        Also creates/updates a Memory record for backward compatibility and
        vector search (so the relationship is discoverable via semantic search too).
        """
        # Upsert source and target entities
        source_entity = await self._upsert_entity(user_id, source_name, source_type)
        target_entity = await self._upsert_entity(user_id, target_name, target_type)
        
        # Upsert into entity_relationships table (knowledge graph edge)
        result = await self.db.execute(
            select(EntityRelationship).where(
                and_(
                    EntityRelationship.source_entity_id == source_entity.id,
                    EntityRelationship.target_entity_id == target_entity.id,
                    EntityRelationship.relationship_type == relationship,
                )
            )
        )
        existing_rel = result.scalar_one_or_none()
        
        if existing_rel:
            existing_rel.mention_count += 1
            existing_rel.confidence = max(existing_rel.confidence, confidence)
            existing_rel.last_seen_at = datetime.utcnow()
            existing_rel.updated_at = datetime.utcnow()
        else:
            new_rel = EntityRelationship(
                id=str(uuid.uuid4()),
                user_id=user_id,
                source_entity_id=source_entity.id,
                target_entity_id=target_entity.id,
                relationship_type=relationship,
                relationship_label=humanize_relationship(
                    source_name, relationship, target_name
                ),
                confidence=confidence,
                mention_count=1,
                first_seen_at=datetime.utcnow(),
                last_seen_at=datetime.utcnow(),
            )
            self.db.add(new_rel)
        
        # Also maintain backward-compatible relationship memory for vector search
        #
        # `content` is what the user READS on the Memory screen (the card shows
        # content, and summary is None for these rows) and what gets embedded.
        # It used to be `predicate.replace("_", " ")` spliced between the two
        # entity names, which is grammatical only by accident — `performed_by`
        # reads fine, `play_on` produced "Better Call Saul play on Netflix".
        # The triple itself is untouched: it stays structured in metadata_json
        # below and in the entity_relationships edge.
        relationship_content = humanize_relationship(
            source_name, relationship, target_name
        )
        legacy_content = f"{source_name} {relationship.replace('_', ' ')} {target_name}"

        # ── Write-time gate on the Memory MIRROR only ────────────────────
        # The graph edge above is already committed and is NOT gated: every
        # traversal, entity map and graph query keeps full fidelity. What is
        # gated is whether this edge also becomes a row the user reads on the
        # Memory screen and that competes for space in the retrieval corpus.
        #
        # 59 of the 73 junk memories on the founder's tenant came through
        # here, because this path had no quality check of any kind — it
        # rendered whatever triple the LLM emitted and INSERTed it.
        from app.services.memory_gate import relationship_gate_reason

        gate_reason = relationship_gate_reason(
            source_name,
            relationship,
            target_name,
            user_aliases=await self._user_aliases(user_id),
            rendered=relationship_content,
        )
        # "no_user_endpoint" is the one verdict that needs the GRAPH to decide.
        # The extraction prompt's actual rule is "the USER, *or someone/
        # something in the USER's life*, must be one end of the edge" — the
        # pure gate can only implement the first half, so on its own it was
        # rejecting "Majid Tajik works at Microsoft" (the user's tutor),
        # "Brother lives in Tehran", and even "Toup uses React Native" — the
        # user's OWN product, which the prompt explicitly lists as valid
        # (Project → Technology).
        if gate_reason == "no_user_endpoint" and await self._endpoint_in_user_world(
            user_id, source_name, target_name
        ):
            gate_reason = None

        if gate_reason:
            logger.info(
                "[memory_gate] relationship not mirrored (%s): %s",
                gate_reason, describe_memory(relationship_content),
            )
            return
        existing = await self.find_similar_memories(
            user_id=user_id,
            content=relationship_content,
            limit=1,
            min_similarity=0.85,
            brain_type="user",
        )

        # A relationship the user has explicitly forgotten must stay forgotten.
        # find_similar_memories excludes soft-deleted rows, so without this
        # check the next mention of the same edge would resurrect it as a
        # brand-new memory — and there is no restore route to undo that.
        # Both phrasings are checked: see _relationship_was_forgotten.
        if not existing and await self._relationship_was_forgotten(
            user_id, relationship_content, legacy_content
        ):
            return

        if existing:
            mem, _ = existing[0]
            mem.access_count += 1
            mem.strength = min(1.0, mem.strength + 0.1)
            mem.last_accessed_at = datetime.utcnow()
            mem.confidence = max(mem.confidence, confidence)
        else:
            # Supersede the previous value of this same relationship.
            #
            # The mirror bypasses MemoryDedupService entirely — its only check
            # is the 0.85 similarity probe above, which asks "have I seen this
            # exact sentence?" and never "does this contradict something I
            # already believe?". So when a user moved city, the brain ended up
            # holding "USER lives in Toronto" AND "USER lives in Vancouver",
            # both active, with nothing to tell the model which is current.
            #
            # The triple is stored structurally in metadata_json, so the stale
            # row can be found exactly — same user, same source entity, same
            # predicate, different target — with no LLM call and no guessing.
            # Superseding (rather than deleting) matches the extracted-memory
            # contradiction path: the old row keeps its id and its trail.
            stale_rows = (
                await self.db.execute(
                    select(Memory).where(
                        and_(
                            Memory.user_id == user_id,
                            Memory.source_type == "entity_extraction",
                            Memory.is_active == True,  # noqa: E712
                            Memory.is_deleted == False,  # noqa: E712
                            Memory.metadata_json.isnot(None),
                        )
                    )
                )
            ).scalars().all()
            for row in stale_rows:
                try:
                    meta = json.loads(row.metadata_json or "{}")
                except (ValueError, TypeError):
                    continue
                if (
                    meta.get("relationship_type") == relationship
                    and (meta.get("source_entity") or "").strip().lower()
                    == (source_name or "").strip().lower()
                    and (meta.get("target_entity") or "").strip().lower()
                    != (target_name or "").strip().lower()
                ):
                    row.is_active = False
                    row.updated_at = datetime.utcnow()
                    logger.info(
                        "[MEMORY] relationship mirror superseded: %s",
                        describe_memory(row.content),
                    )

            embedding = self.embedding_service.embed(relationship_content, api_key=self.api_key)
            
            rel_memory = Memory(
                id=str(uuid.uuid4()),
                user_id=user_id,
                content=relationship_content,
                # summary was `f"{source} → {rel} → {target}"`. The mobile card
                # renders `summary || content`, so that machine format was what
                # users actually saw ("Bunker → performed_by → Baltazar") even
                # though the readable sentence sat on the same row. The triple
                # is still fully preserved, structured, in metadata_json below.
                summary=None,
                brain_type="user",
                # was: "people" if source_type == "person" else "knowledge" —
                # a hardcoded binary that produced two of the three categories
                # ever seen in production.
                category=category_for_relationship(source_type, target_type),
                memory_type="fact",
                memory_level="episodic",
                embedding_json=json.dumps(embedding),
                embedding=embedding,
                importance=0.6,
                confidence=confidence,
                strength=1.0,
                emotional_salience=0.3,
                source_type="entity_extraction",
                tags_json=json.dumps([source_type, target_type, "relationship"]),
                metadata_json=json.dumps({
                    "relationship_type": relationship,
                    "source_entity": source_name,
                    "target_entity": target_name,
                    "extracted_by": "llm",
                }),
            )
            self.db.add(rel_memory)
            await self.db.flush()
            
            # Link memory to both entities
            source_link = EntityLink(
                id=str(uuid.uuid4()),
                memory_id=rel_memory.id,
                entity_id=source_entity.id,
                role="subject",
            )
            target_link = EntityLink(
                id=str(uuid.uuid4()),
                memory_id=rel_memory.id,
                entity_id=target_entity.id,
                role="object",
            )
            self.db.add(source_link)
            self.db.add(target_link)
        
        await self.db.commit()
    
    async def _upsert_entity(
        self,
        user_id: str,
        name: str,
        entity_type: str,
        schema_type: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Entity:
        """Find or create an entity by name and type.

        Phase 4: Now accepts schema_type (e.g. 'PersonEntity') and
        structured attributes dict to deep-merge into attributes_json.
        """
        # The type decides the CATEGORY of every relationship memory this entity
        # takes part in (see category_for_relationship), so an unfolded synonym
        # is not cosmetic — "song" or "tv show" would fall through to `topic`
        # and file the row under Knowledge.
        entity_type = normalize_entity_type(entity_type)

        result = await self.db.execute(
            select(Entity).where(
                and_(
                    Entity.user_id == user_id,
                    func.lower(Entity.name) == name.lower(),
                )
            )
        )
        entity = result.scalar_one_or_none()

        if entity:
            entity.mention_count += 1
            entity.last_seen_at = datetime.utcnow()
            # Upgrade to a more specific type when one arrives.
            #
            # This used to require the stored type to be literally "unknown",
            # which made a mistyped entity permanent: "Better Call Saul" typed
            # `topic` on first mention stayed `topic` forever, even once the
            # extractor learned to say `show`. `topic` is the vocabulary's
            # fallback — "we could not place this" — so it has to be as
            # replaceable as "unknown", or the first guess wins for good.
            if entity_type not in _VAGUE_ENTITY_TYPES and (
                entity.entity_type in _VAGUE_ENTITY_TYPES or not entity.entity_type
            ):
                entity.entity_type = entity_type
            # Update schema_type if provided and not set yet
            if schema_type and not entity.schema_type:
                entity.schema_type = schema_type
            # Deep-merge attributes
            if attributes:
                existing = {}
                if entity.attributes_json:
                    try:
                        existing = json.loads(entity.attributes_json)
                    except (json.JSONDecodeError, TypeError):
                        existing = {}
                # Merge: new values overwrite, but don't erase existing keys
                merged = {**existing, **{k: v for k, v in attributes.items() if v is not None}}
                entity.attributes_json = json.dumps(merged)
        else:
            entity = Entity(
                id=str(uuid.uuid4()),
                user_id=user_id,
                name=name,
                entity_type=entity_type,
                schema_type=schema_type,
                attributes_json=json.dumps(attributes) if attributes else None,
            )
            self.db.add(entity)
            await self.db.flush()
        
        return entity
    
    async def _link_memories(
        self,
        source_id: str,
        target_ids: List[str],
        relationship_type: str = "related_to"
    ):
        """Link memories together"""
        for target_id in target_ids:
            await self.db.execute(
                memory_relationships.insert().values(
                    source_id=source_id,
                    target_id=target_id,
                    relationship_type=relationship_type,
                    strength=1.0,
                    created_at=datetime.utcnow()
                )
            )
        await self.db.commit()
    
    async def _update_brain_stats(self, user_id: str):
        """Update cached brain statistics"""
        # Count memories by category
        result = await self.db.execute(
            select(Memory.category, func.count(Memory.id))
            .where(
                and_(
                    Memory.user_id == user_id,
                    Memory.is_deleted == False
                )
            )
            .group_by(Memory.category)
        )
        category_counts = {row[0]: row[1] for row in result.fetchall()}
        
        # Count total
        total_memories = sum(category_counts.values())
        
        # Count entities
        result = await self.db.execute(
            select(func.count(Entity.id)).where(Entity.user_id == user_id)
        )
        total_entities = result.scalar() or 0
        
        # Count connections
        result = await self.db.execute(
            select(func.count())
            .select_from(memory_relationships)
            .join(Memory, Memory.id == memory_relationships.c.source_id)
            .where(Memory.user_id == user_id)
        )
        total_connections = result.scalar() or 0
        
        # Normalize sizes for visualization
        max_count = max(category_counts.values()) if category_counts else 1
        category_sizes = {
            cat: count / max_count
            for cat, count in category_counts.items()
        }
        
        # Upsert brain stats.
        #
        # This used to be SELECT-then-INSERT, which is a lost-update race with
        # a UNIQUE index on brain_stats.user_id: two turns for the same user
        # arriving together (web + mobile, or rapid-fire messages) both saw no
        # row and both INSERTed. The second raised UniqueViolationError, and
        # because this runs INSIDE create_memory's transaction it took the
        # whole memory write down with it — the user stated a fact and it was
        # silently dropped. Measured: 4 of 20 concurrent writes lost.
        #
        # A single INSERT ... ON CONFLICT DO UPDATE is atomic, so concurrent
        # turns serialise on the row instead of colliding.
        payload = {
            "region_counts_json": json.dumps(category_counts),
            "region_sizes_json": json.dumps(category_sizes),
            "total_memories": total_memories,
            "total_entities": total_entities,
            "total_connections": total_connections,
            "updated_at": datetime.utcnow(),
        }
        dialect = self.db.bind.dialect.name if self.db.bind is not None else ""
        if dialect == "postgresql":
            from sqlalchemy.dialects.postgresql import insert as _pg_insert

            stmt = _pg_insert(BrainStats).values(
                id=str(uuid.uuid4()), user_id=user_id, **payload
            )
            await self.db.execute(
                stmt.on_conflict_do_update(
                    index_elements=[BrainStats.user_id], set_=payload
                )
            )
        else:
            # SQLite (tests//CI) has no matching unique-index upsert here;
            # the race it guards against cannot happen on a single-writer
            # in-memory database anyway.
            existing = (
                await self.db.execute(
                    select(BrainStats).where(BrainStats.user_id == user_id)
                )
            ).scalar_one_or_none()
            if existing:
                for key, value in payload.items():
                    setattr(existing, key, value)
            else:
                self.db.add(BrainStats(user_id=user_id, **payload))

        await self.db.commit()
    
    async def _log_memory_event(
        self,
        memory_id: str,
        user_id: str,
        event_type: MemoryEventType,
        event_data: Dict[str, Any],
        trigger_source: str
    ) -> MemoryEvent:
        """
        Log an event to the immutable memory audit trail.
        This is called automatically by create/update/delete operations.
        """
        event = MemoryEvent(
            id=str(uuid.uuid4()),
            memory_id=memory_id,
            user_id=user_id,
            event_type=event_type.value,
            timestamp=datetime.utcnow(),
            event_data_json=json.dumps(event_data) if event_data else None,
            trigger_source=trigger_source,
        )
        self.db.add(event)
        # Don't commit here - let the calling method handle the transaction
        return event
    
    async def get_memory_events(
        self,
        memory_id: str,
        user_id: str,
        limit: int = 100
    ) -> List[MemoryEvent]:
        """Get the event audit trail for a memory."""
        result = await self.db.execute(
            select(MemoryEvent).where(
                and_(
                    MemoryEvent.memory_id == memory_id,
                    MemoryEvent.user_id == user_id,
                )
            ).order_by(MemoryEvent.timestamp.desc()).limit(limit)
        )
        return list(result.scalars().all())
    
    def _generate_explanation(self, memory: Memory, similarity: float, weighted_score: Optional[float] = None) -> str:
        """Generate explanation for why a memory was retrieved"""
        parts = []
        
        # Similarity explanation
        if similarity >= 0.9:
            parts.append("Very high semantic match")
        elif similarity >= 0.7:
            parts.append("Good semantic match")
        else:
            parts.append("Partial semantic match")
        
        parts.append(f"({similarity:.0%} similar)")
        
        # NEW: Add strength/decay info if using weighted scoring
        if weighted_score is not None:
            if memory.strength >= 0.8:
                parts.append("• Strong memory")
            elif memory.strength < 0.4:
                parts.append("• Fading memory")
            
            if memory.memory_level == "semantic":
                parts.append("• Consolidated knowledge")
            elif memory.memory_level == "procedural":
                parts.append("• Procedural skill")
        
        # Add category context
        category_names = {
            "identity": "Identity",
            "preferences": "Preferences",
            "beliefs": "Beliefs",
            "emotions": "Emotions",
            "people": "People",
            "places": "Places",
            "family": "Family",
            "experiences": "Experiences",
            "projects": "Projects",
            "schedule": "Schedule",
            "work": "Work",
            "learning": "Learning",
            "knowledge": "Knowledge",
            "tools": "Tools",
            "media": "Media",
            "health": "Health",
            "habits": "Habits",
            "food": "Food",
            "travel": "Travel",
            "goals": "Goals",
            "context": "Context",
        }
        category_name = category_names.get(memory.category, memory.category)
        parts.append(f"• From {category_name}")
        
        # Add recency
        age_days = (datetime.utcnow() - memory.created_at).days
        if age_days == 0:
            parts.append("• Created today")
        elif age_days == 1:
            parts.append("• Created yesterday")
        elif age_days < 7:
            parts.append(f"• Created {age_days} days ago")
        
        return " ".join(parts)

    # ==================================================================
    # HYBRID RETRIEVAL ENGINE — Multi-strategy search with RRF fusion
    # ==================================================================

    @staticmethod
    def reciprocal_rank_fusion(
        ranked_lists: List[List[Tuple[str, float]]],
        k: int = 60,
    ) -> List[Tuple[str, float]]:
        """
        Fuse multiple ranked result lists using Reciprocal Rank Fusion.
        
        For each document d, its RRF score is:
        RRF(d) = Σ 1/(k + rank_i(d))  for each ranked list i
        
        k=60 is the standard constant from the RRF paper (Cormack et al., 2009).
        """
        scores: Dict[str, float] = defaultdict(float)
        for ranked_list in ranked_lists:
            for rank, (doc_id, _score) in enumerate(ranked_list, start=1):
                scores[doc_id] += 1.0 / (k + rank)
        fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return fused

    async def hybrid_search(
        self,
        user_id: str,
        query: str,
        limit: int = 15,
        min_similarity: float = 0.3,
        min_strength: float = 0.1,
        brain_types: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        created_after: Optional[datetime] = None,
        created_before: Optional[datetime] = None,
        strategies: Optional[List[str]] = None,
    ) -> List[dict]:
        """
        Multi-strategy memory retrieval with Reciprocal Rank Fusion.

        Strategies:
          - "vector": pgvector cosine similarity (semantic search)
          - "keyword": tsvector full-text search (BM25-style keyword matching)
          - "graph": entity graph traversal (knowledge graph walk)
          - "temporal": recency-weighted search (auto-activated on time queries)

        Temporal reasoning: Automatically detects time references in queries
        (e.g., "last week", "yesterday", "in January") and applies date filters
        to all strategies + adds a temporal strategy for recency-weighted results.

        Results from all strategies are fused with RRF, then weighted
        by Brain's proprietary scoring (strength, importance, emotion, recency).

        F5 (2026-05-08): adds a real min_strength floor (NEW, default 0.1).
        Pre-F5 there was no strength filter at all — strength only weighted
        final scoring at 20%, so faded memories could still surface if RRF
        ranked them via keyword/graph match. With this floor, decay actually
        means "fades out of recall."
          - Active_task memories are unaffected: their reinforcement path
            (active_task_service.py) resets strength to 1.0 on every match,
            so they never approach the floor while live.
          - High-importance memories that DO fade below the floor were
            either created with strength=1.0 and never revisited (genuine
            forgetting) or seeded as low-strength on purpose.
          - Pass min_strength=0.0 to opt out (keep old behaviour for
            specific call sites that want everything).

        F-final (2026-05-10): min_similarity is now LIVE — applied to
        the _vector_search path only (NOT to the merged RRF set).
        Rationale (the conservative product call):
          - A memory matched by KEYWORD or GRAPH still surfaces at any
            vector similarity. Lexical/graph evidence is independent of
            embedding proximity; floor-gating the merged set would drop
            valid matches that happen to have low cosine distance.
          - A memory matched ONLY by the vector path with low similarity
            is the noise case (semantic-but-tangential). That's what we
            want filtered.
        Default 0.3 from the function signature; callers can override
        per-query (chat path passes 0.1 for max recall, prompt assembly
        leans on the default).
        """
        import logging
        logger = logging.getLogger(__name__)

        if strategies is None:
            strategies = ["vector", "keyword", "graph"]

        # Build common filter conditions
        conditions = [
            Memory.user_id == user_id,
            Memory.is_deleted == False,
            Memory.is_active == True,
        ]
        # F5: real min_strength floor. The decay model is meaningless if
        # forgotten memories can still be retrieved. Active_task memories
        # with importance=0.9 are unaffected — strength resets to 1.0 on
        # every reinforcement. Long-faded memories (strength < 0.1) drop.
        if min_strength is not None and min_strength > 0:
            conditions.append(Memory.strength >= min_strength)
        if brain_types:
            conditions.append(Memory.brain_type.in_(brain_types))
        if categories:
            conditions.append(Memory.category.in_(categories))
        if created_after:
            conditions.append(Memory.created_at >= created_after)
        if created_before:
            conditions.append(Memory.created_at <= created_before)
        
        # Phase 7: Auto-detect temporal intent from query
        temporal = extract_temporal_filters(query)
        temporal_after = created_after  # Preserve explicit params
        temporal_before = created_before
        if temporal["has_temporal"]:
            logger.info(f"[HYBRID] Temporal detected: {temporal['temporal_keywords']} → after={temporal.get('created_after')}, before={temporal.get('created_before')}")
            if temporal["created_after"] and not created_after:
                conditions.append(Memory.created_at >= temporal["created_after"])
                temporal_after = temporal["created_after"]
            if temporal["created_before"] and not created_before:
                conditions.append(Memory.created_at <= temporal["created_before"])
                temporal_before = temporal["created_before"]
        
        # Generate query embedding once (shared by vector strategy)
        # Use async version to avoid blocking the event loop during chat
        # W1.5: an embedding failure must not kill the whole search —
        # keyword/graph/temporal don't need the embedding, so degrade
        # to them instead of raising (per-strategy isolation below only
        # covers the gather, not this shared call).
        try:
            query_embedding = await self.embedding_service.embed_async(query, api_key=self.api_key)
        except Exception as e:
            query_embedding = None
            logger.warning(f"[HYBRID] Query embedding failed, skipping vector strategy: {e}")

        # Run enabled strategies in parallel
        tasks = {}
        fetch_limit = limit * 4  # Over-fetch per strategy for re-ranker input
        
        if "vector" in strategies and query_embedding is not None:
            # F-final (2026-05-10): pass min_similarity down so the
            # vector-only path applies the floor BEFORE results enter
            # RRF. Memories matched by keyword/graph still surface
            # without a similarity check — that's the conservative
            # product call documented in the hybrid_search docstring.
            tasks["vector"] = self._vector_search(
                user_id, query_embedding, fetch_limit, conditions,
                min_similarity=min_similarity,
            )
        if "keyword" in strategies:
            tasks["keyword"] = self._keyword_search(
                user_id, query, fetch_limit, conditions
            )
        if "graph" in strategies:
            tasks["graph"] = self._graph_search(
                user_id, query, fetch_limit
            )
        if "temporal" in strategies or temporal.get("has_temporal"):
            tasks["temporal"] = self._temporal_search(
                user_id, query, fetch_limit, temporal_after, temporal_before
            )
        
        # Execute all strategies concurrently
        task_names = list(tasks.keys())
        task_coros = list(tasks.values())
        
        try:
            results = await asyncio.gather(*task_coros, return_exceptions=True)
        except Exception as e:
            logger.warning(f"Hybrid search gather failed: {e}")
            results = []
        
        # Collect valid ranked lists
        ranked_lists = []
        for name, result in zip(task_names, results):
            if isinstance(result, Exception):
                logger.warning(f"Strategy '{name}' failed: {result}")
                continue
            if result:
                ranked_lists.append(result)
                logger.info(f"[HYBRID] Strategy '{name}' returned {len(result)} results")
        
        if not ranked_lists:
            return []
        
        # Fuse with RRF
        fused = self.reciprocal_rank_fusion(ranked_lists)
        fused_ids = [doc_id for doc_id, _ in fused[:limit * 3]]  # Keep 3x for re-ranker
        
        if not fused_ids:
            return []
        
        # Fetch the actual Memory objects for the top results
        result = await self.db.execute(
            select(Memory).where(
                and_(
                    Memory.id.in_(fused_ids),
                    Memory.user_id == user_id,
                    Memory.is_deleted == False,
                    Memory.is_active == True,
                )
            )
        )
        memories_map = {m.id: m for m in result.scalars().all()}
        
        # Compute per-memory vector similarity (for scoring + response)
        similarity_map: Dict[str, float] = {}
        if hasattr(Memory, 'embedding') and Memory.embedding is not None and fused_ids and query_embedding is not None:
            try:
                # pgvector accepts Python list directly (no string conversion needed)
                embedding_vec = query_embedding  # raw list for pgvector
                sim_query = (
                    select(
                        Memory.id,
                        (1 - Memory.embedding.cosine_distance(embedding_vec)).label("sim"),
                    )
                    .where(
                        and_(
                            Memory.id.in_(fused_ids),
                            Memory.embedding.isnot(None),
                        )
                    )
                )
                sim_result = await self.db.execute(sim_query)
                for row in sim_result.all():
                    similarity_map[row.id] = row.sim
            except Exception:
                pass  # Fall through — similarity will be 0
        
        # Apply Brain's weighted scoring on top of RRF
        rrf_scores = dict(fused)
        scored_memories = []
        
        for memory_id in fused_ids:
            memory = memories_map.get(memory_id)
            if not memory:
                continue
            
            similarity = similarity_map.get(memory_id, 0.0)
            rrf_score = rrf_scores.get(memory_id, 0.0)
            
            # Weighted score: combines semantic similarity + memory quality
            final_score = (
                similarity * 0.30 +
                rrf_score * 10.0 * 0.25 +  # Normalize RRF (typically 0.01-0.05 range)
                memory.strength * 0.20 +
                memory.importance * 0.15 +
                memory.emotional_salience * 0.10
            )
            
            # Recency boost
            if memory.last_accessed_at:
                days_since_access = (datetime.utcnow() - memory.last_accessed_at).days
                if days_since_access < 7:
                    final_score += (7 - days_since_access) / 7 * 0.1
            
            scored_memories.append({
                "id": memory.id,
                "content": memory.content,
                "summary": memory.summary,
                "brain_type": memory.brain_type,
                "category": memory.category,
                "memory_type": memory.memory_type,
                "importance": memory.importance,
                "confidence": memory.confidence,
                "strength": memory.strength,
                "created_at": memory.created_at.isoformat() if memory.created_at else None,
                "updated_at": memory.updated_at.isoformat() if memory.updated_at else None,
                "last_accessed_at": memory.last_accessed_at.isoformat() if memory.last_accessed_at else None,
                "access_count": memory.access_count,
                "source_type": memory.source_type,
                "similarity_score": similarity,
                "final_score": final_score,
                "retrieval_strategies": [n for n, r in zip(task_names, results) if not isinstance(r, Exception)],
            })
        
        scored_memories.sort(key=lambda x: x["final_score"], reverse=True)

        # Phase 6: Cross-encoder re-ranking for improved precision
        try:
            from app.config import settings as _settings
            if _settings.enable_reranker and len(scored_memories) > limit:
                from app.services.reranker_service import get_reranker_service
                reranker = get_reranker_service()
                reranked = await reranker.rerank(
                    query=query,
                    candidates=scored_memories,
                    top_k=limit,
                )
                logger.info(
                    f"[HYBRID] Re-ranked {len(scored_memories)}→{len(reranked)} candidates"
                )
                scored_memories = reranked
        except Exception as e:
            logger.warning(f"[HYBRID] Re-ranker failed, using score-based ranking: {e}")
            scored_memories = scored_memories[:limit]

        # Auto-reinforce top retrieved memories (spaced repetition)
        try:
            from app.services.decay_service import DecayService
            decay_svc = DecayService(self.db)
            for mem in scored_memories[:5]:
                await decay_svc.reinforce_memory(
                    memory_id=mem["id"],
                    user_id=user_id,
                    access_context="retrieval",
                    similarity_score=mem.get("similarity_score", 0.5),
                )
        except Exception as e:
            logger.warning(f"[HYBRID] Auto-reinforcement failed (non-fatal): {e}")

        return scored_memories

    async def _vector_search(
        self,
        user_id: str,
        query_embedding: List[float],
        limit: int,
        conditions: list,
        min_similarity: float = 0.05,
    ) -> List[Tuple[str, float]]:
        """
        pgvector cosine similarity search.
        Returns [(memory_id, similarity_score)] sorted by similarity desc.

        F-final (2026-05-10): min_similarity is now a real parameter
        (was previously a hard-coded 0.05 noise floor). hybrid_search
        passes its own min_similarity through (default 0.3) — that's
        the active, vector-only floor. Direct callers can still pass
        0.05 to keep the lenient behaviour.
        """
        if not (hasattr(Memory, 'embedding') and Memory.embedding is not None):
            return []

        try:
            extra_conditions = list(conditions) + [Memory.embedding.isnot(None)]
            # pgvector accepts Python list directly (no string conversion needed)
            embedding_vec = query_embedding  # raw list for pgvector

            query = (
                select(
                    Memory.id,
                    (1 - Memory.embedding.cosine_distance(embedding_vec)).label("similarity"),
                )
                .where(and_(*extra_conditions))
                .order_by(Memory.embedding.cosine_distance(embedding_vec))
                .limit(limit)
            )

            result = await self.db.execute(query)
            # Apply the floor as an explicit filter so callers can tune it.
            floor = max(min_similarity, 0.0)
            return [(row.id, row.similarity) for row in result.all() if row.similarity > floor]
        except Exception:
            return []

    async def _keyword_search(
        self,
        user_id: str,
        query: str,
        limit: int,
        conditions: list,
    ) -> List[Tuple[str, float]]:
        """
        PostgreSQL tsvector full-text search with ts_rank scoring.
        Falls back to ILIKE if tsvector column doesn't exist.
        Returns [(memory_id, rank_score)] sorted by rank desc.
        """
        # Clean query for tsquery — remove special chars
        clean_query = re.sub(r'[^\w\s]', ' ', query).strip()
        if not clean_query:
            return []
        
        try:
            # Try tsvector path first
            ts_query = func.plainto_tsquery('english', clean_query)
            
            stmt = (
                select(
                    Memory.id,
                    func.ts_rank(Memory.search_vector, ts_query).label("rank"),
                )
                .where(
                    and_(
                        *conditions,
                        Memory.search_vector.op('@@')(ts_query),
                    )
                )
                .order_by(func.ts_rank(Memory.search_vector, ts_query).desc())
                .limit(limit)
            )
            
            result = await self.db.execute(stmt)
            rows = result.all()
            if rows:
                return [(row.id, float(row.rank)) for row in rows]
        except Exception:
            pass  # Fall through to ILIKE
        
        # Fallback: simple ILIKE search
        try:
            keywords = clean_query.split()[:5]  # Max 5 keywords
            ilike_conditions = list(conditions)
            for kw in keywords:
                if len(kw) >= 3:
                    ilike_conditions.append(
                        or_(
                            Memory.content.ilike(f"%{kw}%"),
                            Memory.summary.ilike(f"%{kw}%"),
                        )
                    )
            
            if len(ilike_conditions) == len(conditions):
                return []  # No valid keywords
            
            stmt = (
                select(Memory.id)
                .where(and_(*ilike_conditions))
                .order_by(Memory.importance.desc())
                .limit(limit)
            )
            
            result = await self.db.execute(stmt)
            # Give ILIKE results a uniform rank (no ts_rank available)
            return [(row.id, 0.5) for row in result.all()]
        except Exception:
            return []

    async def _graph_search(
        self,
        user_id: str,
        query: str,
        limit: int,
    ) -> List[Tuple[str, float]]:
        """
        Entity-graph search using recursive CTE for multi-hop traversal.
        
        1. Extract entity names from query
        2. Find seed entities in DB
        3. Traverse EntityRelationship edges (up to 2 hops) via recursive CTE
        4. Find memories linked to all discovered entities
        5. Score: direct matches > 1-hop > 2-hop
        
        Returns [(memory_id, graph_score)] sorted by graph_score desc.
        """
        # Step 1: Extract candidate entity names from the query
        entity_names = self._extract_entity_names(query)
        if not entity_names:
            return []
        
        # Step 2: Find matching entities in the database
        seed_entity_ids: set = set()
        for name in entity_names:
            result = await self.db.execute(
                select(Entity.id).where(
                    and_(
                        Entity.user_id == user_id,
                        func.lower(Entity.name).contains(name.lower()),
                    )
                )
            )
            for row in result.all():
                seed_entity_ids.add(row.id)
        
        if not seed_entity_ids:
            return []
        
        # Step 3: Recursive CTE traversal through entity_relationships (up to 2 hops)
        traversal_results = await self.traverse_entity_graph(
            user_id=user_id,
            seed_entity_ids=list(seed_entity_ids),
            max_depth=2,
            limit=limit * 3,
        )
        
        # Build entity→depth map (depth 0 = seed, 1 = 1-hop, 2 = 2-hop)
        entity_depth: Dict[str, int] = {}
        for eid in seed_entity_ids:
            entity_depth[eid] = 0
        for item in traversal_results:
            eid = item["entity_id"]
            depth = item["depth"]
            if eid not in entity_depth or depth < entity_depth[eid]:
                entity_depth[eid] = depth
        
        all_entity_ids = set(entity_depth.keys())
        
        # Step 4: Find memories linked to these entities via EntityLink
        if not all_entity_ids:
            return []
        
        result = await self.db.execute(
            select(EntityLink.memory_id, EntityLink.entity_id).where(
                EntityLink.entity_id.in_(all_entity_ids)
            )
        )
        links = result.all()
        
        memory_scores: Dict[str, float] = defaultdict(float)
        for memory_id, entity_id in links:
            depth = entity_depth.get(entity_id, 3)
            # Decay score by depth: direct=1.0, 1-hop=0.5, 2-hop=0.25
            memory_scores[memory_id] += 1.0 / (1 + depth)
        
        # Filter to active, non-deleted memories owned by this user
        if memory_scores:
            result = await self.db.execute(
                select(Memory.id).where(
                    and_(
                        Memory.id.in_(list(memory_scores.keys())),
                        Memory.user_id == user_id,
                        Memory.is_deleted == False,
                        Memory.is_active == True,
                    )
                )
            )
            valid_ids = {row.id for row in result.all()}
            memory_scores = {k: v for k, v in memory_scores.items() if k in valid_ids}
        
        # Sort by score descending
        scored = sorted(memory_scores.items(), key=lambda x: x[1], reverse=True)
        return scored[:limit]


    async def _temporal_search(
        self,
        user_id: str,
        query: str,
        limit: int,
        created_after: Optional[datetime] = None,
        created_before: Optional[datetime] = None,
    ) -> List[Tuple[str, float]]:
        """
        Temporal retrieval strategy — prioritizes recency within a time window.
        
        When a user asks "what happened recently?" or "what did I discuss last week?",
        this strategy surfaces memories sorted by recency + importance, independent
        of semantic similarity. This catches memories that vector search might miss
        because the query terms don't semantically match the content.
        
        Score formula:
            score = recency_weight * (1 / (1 + days_ago)) + importance * 0.3 + strength * 0.2
        
        Returns [(memory_id, temporal_score)] sorted by score desc.
        """
        try:
            conditions = [
                Memory.user_id == user_id,
                Memory.is_deleted == False,
                Memory.is_active == True,
            ]
            
            if created_after:
                conditions.append(Memory.created_at >= created_after)
            if created_before:
                conditions.append(Memory.created_at <= created_before)
            
            stmt = (
                select(
                    Memory.id,
                    Memory.created_at,
                    Memory.importance,
                    Memory.strength,
                    Memory.emotional_salience,
                )
                .where(and_(*conditions))
                .order_by(Memory.created_at.desc())
                .limit(limit)
            )
            
            result = await self.db.execute(stmt)
            rows = result.all()
            
            if not rows:
                return []
            
            now = datetime.utcnow()
            scored = []
            for row in rows:
                days_ago = max(0.01, (now - row.created_at).total_seconds() / 86400.0)
                recency_score = 1.0 / (1.0 + days_ago)
                importance = row.importance if row.importance else 0.5
                strength = row.strength if row.strength else 0.5
                emotion = row.emotional_salience if row.emotional_salience else 0.0
                
                temporal_score = (
                    recency_score * 0.50 +
                    importance * 0.25 +
                    strength * 0.15 +
                    emotion * 0.10
                )
                scored.append((row.id, temporal_score))
            
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:limit]
        
        except Exception:
            return []

    async def traverse_entity_graph(
        self,
        user_id: str,
        seed_entity_ids: Optional[List[str]] = None,
        entity_names: Optional[List[str]] = None,
        max_depth: int = 2,
        limit: int = 50,
    ) -> List[dict]:
        """
        Multi-hop BFS traversal through EntityRelationship edges using
        a PostgreSQL recursive CTE for maximum efficiency.
        
        Accepts either seed_entity_ids (pre-resolved) or entity_names (to resolve).
        Returns a list of dicts: {entity_id, entity_name, entity_type, depth,
            relationship_type, relationship_label, from_entity_id, from_entity_name}
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Resolve entity names to IDs if needed
        if seed_entity_ids is None and entity_names:
            seed_entity_ids = []
            for name in entity_names:
                result = await self.db.execute(
                    select(Entity.id).where(
                        and_(
                            Entity.user_id == user_id,
                            func.lower(Entity.name).contains(name.lower()),
                        )
                    )
                )
                for row in result.all():
                    seed_entity_ids.append(row.id)
        
        if not seed_entity_ids:
            return []
        
        # Build the recursive CTE query
        # This traverses entity_relationships in BOTH directions (bidirectional graph)
        #
        # SQL-INJECTION guard (audit-2026 re-audit round 10 — CRITICAL): the
        # seed ids are interpolated into the CTE text below, and they can arrive
        # RAW + unvalidated from POST /api/graph/traverse's `entity_ids`
        # (List[str], no coercion) — which is mounted on the SHARED central
        # Postgres too. Entity.id is always a uuid4 string, so coerce each id to
        # a canonical UUID: a well-formed UUID contains only hex + hyphens (no
        # quotes/UNION/-- possible), and any injection payload fails uuid.UUID()
        # and is dropped. Bound params would also work; coercion additionally
        # rejects malformed ids outright.
        import uuid as _uuid
        _safe_ids = []
        for _eid in seed_entity_ids:
            try:
                _safe_ids.append(str(_uuid.UUID(str(_eid))))
            except (ValueError, TypeError, AttributeError):
                continue
        if not _safe_ids:
            return []
        seed_ids_str = ",".join(f"'{_e}'" for _e in _safe_ids)
        
        cte_sql = text(f"""
            WITH RECURSIVE graph_walk AS (
                -- Base case: seed entities at depth 0
                SELECT
                    e.id AS entity_id,
                    e.name AS entity_name,
                    e.entity_type AS entity_type,
                    0 AS depth,
                    NULL::varchar AS relationship_type,
                    NULL::varchar AS relationship_label,
                    NULL::varchar AS from_entity_id,
                    NULL::varchar AS from_entity_name
                FROM entities e
                WHERE e.id IN ({seed_ids_str})
                  AND e.user_id = :user_id

                UNION ALL

                -- Bidirectional edges: traverse both directions in one recursive term
                SELECT
                    e2.id AS entity_id,
                    e2.name AS entity_name,
                    e2.entity_type AS entity_type,
                    gw.depth + 1 AS depth,
                    er.relationship_type,
                    er.relationship_label,
                    gw.entity_id AS from_entity_id,
                    gw.entity_name AS from_entity_name
                FROM graph_walk gw
                JOIN entity_relationships er
                    ON er.source_entity_id = gw.entity_id
                    OR er.target_entity_id = gw.entity_id
                JOIN entities e2
                    ON e2.id = CASE
                        WHEN er.source_entity_id = gw.entity_id THEN er.target_entity_id
                        ELSE er.source_entity_id
                    END
                WHERE gw.depth < :max_depth
                  AND er.user_id = :user_id
            )
            SELECT DISTINCT ON (entity_id)
                entity_id, entity_name, entity_type, depth,
                relationship_type, relationship_label,
                from_entity_id, from_entity_name
            FROM graph_walk
            ORDER BY entity_id, depth ASC
            LIMIT :limit
        """)
        
        try:
            result = await self.db.execute(
                cte_sql,
                {"user_id": user_id, "max_depth": max_depth, "limit": limit},
            )
            rows = result.fetchall()
            
            return [
                {
                    "entity_id": row.entity_id,
                    "entity_name": row.entity_name,
                    "entity_type": row.entity_type,
                    "depth": row.depth,
                    "relationship_type": row.relationship_type,
                    "relationship_label": row.relationship_label,
                    "from_entity_id": row.from_entity_id,
                    "from_entity_name": row.from_entity_name,
                }
                for row in rows
            ]
        except Exception as e:
            logger.warning(f"Recursive CTE graph traversal failed: {e}")
            # Fallback to simple 1-hop if CTE fails
            expanded = set(seed_entity_ids)
            try:
                result = await self.db.execute(
                    select(
                        EntityRelationship.target_entity_id,
                        EntityRelationship.relationship_type,
                    ).where(EntityRelationship.source_entity_id.in_(seed_entity_ids))
                )
                for row in result.all():
                    expanded.add(row.target_entity_id)
                
                result = await self.db.execute(
                    select(
                        EntityRelationship.source_entity_id,
                        EntityRelationship.relationship_type,
                    ).where(EntityRelationship.target_entity_id.in_(seed_entity_ids))
                )
                for row in result.all():
                    expanded.add(row.source_entity_id)
            except Exception:
                pass
            
            # Fetch entity details
            result = await self.db.execute(
                select(Entity.id, Entity.name, Entity.entity_type).where(
                    Entity.id.in_(expanded)
                )
            )
            return [
                {
                    "entity_id": row.id,
                    "entity_name": row.name,
                    "entity_type": row.entity_type,
                    "depth": 0 if row.id in seed_entity_ids else 1,
                    "relationship_type": None,
                    "relationship_label": None,
                    "from_entity_id": None,
                    "from_entity_name": None,
                }
                for row in result.all()
            ]

    async def get_entity_relationships(
        self,
        user_id: str,
        entity_id: Optional[str] = None,
        relationship_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[dict]:
        """
        Get entity relationships, optionally filtered by entity or type.
        Returns relationship details with source/target entity info.
        """
        conditions = [EntityRelationship.user_id == user_id]
        if entity_id:
            conditions.append(
                or_(
                    EntityRelationship.source_entity_id == entity_id,
                    EntityRelationship.target_entity_id == entity_id,
                )
            )
        if relationship_type:
            conditions.append(EntityRelationship.relationship_type == relationship_type)
        
        result = await self.db.execute(
            select(EntityRelationship)
            .where(and_(*conditions))
            .order_by(EntityRelationship.mention_count.desc())
            .limit(limit)
        )
        rels = result.scalars().all()
        
        # Batch-load referenced entities
        entity_ids = set()
        for r in rels:
            entity_ids.add(r.source_entity_id)
            entity_ids.add(r.target_entity_id)
        
        entities_map: Dict[str, Entity] = {}
        if entity_ids:
            result = await self.db.execute(
                select(Entity).where(Entity.id.in_(entity_ids))
            )
            for e in result.scalars().all():
                entities_map[e.id] = e
        
        output = []
        for r in rels:
            src = entities_map.get(r.source_entity_id)
            tgt = entities_map.get(r.target_entity_id)
            output.append({
                "id": r.id,
                "relationship_type": r.relationship_type,
                "relationship_label": r.relationship_label,
                "confidence": r.confidence,
                "mention_count": r.mention_count,
                "first_seen_at": r.first_seen_at.isoformat() if r.first_seen_at else None,
                "last_seen_at": r.last_seen_at.isoformat() if r.last_seen_at else None,
                "properties": json.loads(r.properties_json) if r.properties_json else None,
                "source_entity": {
                    "id": src.id,
                    "name": src.name,
                    "entity_type": src.entity_type,
                } if src else None,
                "target_entity": {
                    "id": tgt.id,
                    "name": tgt.name,
                    "entity_type": tgt.entity_type,
                } if tgt else None,
            })
        
        return output

    async def get_entities(
        self,
        user_id: str,
        entity_type: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 100,
    ) -> List[dict]:
        """
        List entities for a user, optionally filtered by type or name search.
        """
        conditions = [Entity.user_id == user_id]
        if entity_type:
            conditions.append(Entity.entity_type == entity_type)
        if search:
            conditions.append(
                func.lower(Entity.name).contains(search.lower())
            )
        
        result = await self.db.execute(
            select(Entity)
            .where(and_(*conditions))
            .order_by(Entity.mention_count.desc())
            .limit(limit)
        )
        entities = result.scalars().all()
        
        output = []
        for e in entities:
            output.append({
                "id": e.id,
                "name": e.name,
                "entity_type": e.entity_type,
                "description": e.description,
                "mention_count": e.mention_count,
                "first_seen_at": e.first_seen_at.isoformat() if e.first_seen_at else None,
                "last_seen_at": e.last_seen_at.isoformat() if e.last_seen_at else None,
                "attributes": json.loads(e.attributes_json) if e.attributes_json else None,
            })
        
        return output

    @staticmethod
    def _extract_entity_names(query: str) -> List[str]:
        """
        Extract candidate entity names from a query string.
        
        Uses heuristics:
        1. Capitalized words/phrases (proper nouns)
        2. Quoted strings
        3. Words after entity-indicating prepositions
        
        Fast and local — no LLM call needed.
        """
        names = set()
        
        # 1. Extract quoted strings
        quoted = re.findall(r'"([^"]+)"', query)
        names.update(quoted)
        quoted_single = re.findall(r"'([^']+)'", query)
        names.update(quoted_single)
        
        # 2. Extract capitalized words/phrases (consecutive capitalized words)
        # Skip first word (may be capitalized because it starts a sentence)
        words = query.split()
        if len(words) > 1:
            cap_phrase = []
            for word in words[1:]:
                clean = re.sub(r'[^\w]', '', word)
                if clean and clean[0].isupper() and len(clean) > 1:
                    cap_phrase.append(clean)
                else:
                    if cap_phrase:
                        names.add(' '.join(cap_phrase))
                        cap_phrase = []
            if cap_phrase:
                names.add(' '.join(cap_phrase))
        
        # 3. Also check the first word if it looks like a name (>= 2 chars, capitalized)
        if words:
            first = re.sub(r'[^\w]', '', words[0])
            if first and first[0].isupper() and len(first) > 1 and first.lower() not in {
                'what', 'when', 'where', 'who', 'how', 'why', 'the', 'this', 'that',
                'which', 'tell', 'show', 'find', 'get', 'do', 'does', 'did', 'is',
                'are', 'was', 'were', 'can', 'could', 'will', 'would', 'should',
                'have', 'has', 'had', 'my', 'your', 'their', 'our', 'its',
            }:
                names.add(first)
        
        return list(names)

    # ==================================================================
    # END HYBRID RETRIEVAL ENGINE
    # ==================================================================

    async def search_memories_by_embedding(
        self,
        user_id: str,
        embedding: List[float],
        limit: int = 15,
        min_similarity: float = 0.5,
        brain_types: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        created_after: Optional[datetime] = None,
        created_before: Optional[datetime] = None,
    ) -> List[dict]:
        """
        Search memories using a pre-computed embedding.
        Uses pgvector for fast similarity search when available.
        
        Args:
            user_id: User ID to search memories for
            embedding: Pre-computed query embedding
            limit: Maximum number of results
            min_similarity: Minimum similarity threshold
            brain_types: Optional list of brain types to filter by
            categories: Optional list of categories to filter by
            created_after: Optional temporal filter — only memories created after this datetime
            created_before: Optional temporal filter — only memories created before this datetime
        
        Returns:
            List of memory dicts with similarity scores
        """
        # Build base conditions
        conditions = [
            Memory.user_id == user_id,
            Memory.is_deleted == False,
            Memory.is_active == True,
        ]
        
        if brain_types:
            conditions.append(Memory.brain_type.in_(brain_types))
        if categories:
            conditions.append(Memory.category.in_(categories))
        if created_after:
            conditions.append(Memory.created_at >= created_after)
        if created_before:
            conditions.append(Memory.created_at <= created_before)
        
        scored_memories = []
        
        # Try pgvector-accelerated search
        if hasattr(Memory, 'embedding') and Memory.embedding is not None:
            try:
                conditions.append(Memory.embedding.isnot(None))
                # pgvector accepts Python list directly (no string conversion needed)
                embedding_vec = embedding  # raw list for pgvector
                
                query = (
                    select(
                        Memory,
                        (1 - Memory.embedding.cosine_distance(embedding_vec)).label("similarity")
                    )
                    .where(and_(*conditions))
                    .order_by(Memory.embedding.cosine_distance(embedding_vec))
                    .limit(limit * 2)  # Over-fetch for weighted re-ranking
                )
                
                result = await self.db.execute(query)
                rows = result.all()
                
                for memory, similarity in rows:
                    if similarity < min_similarity:
                        continue
                    
                    final_score = (
                        similarity * 0.4 +
                        memory.strength * 0.25 +
                        memory.importance * 0.2 +
                        memory.emotional_salience * 0.15
                    )
                    
                    if memory.last_accessed_at:
                        days_since_access = (datetime.utcnow() - memory.last_accessed_at).days
                        if days_since_access < 7:
                            recency_boost = (7 - days_since_access) / 7 * 0.1
                            final_score += recency_boost
                    
                    scored_memories.append({
                        "id": memory.id,
                        "content": memory.content,
                        "summary": memory.summary,
                        "brain_type": memory.brain_type,
                        "category": memory.category,
                        "memory_type": memory.memory_type,
                        "importance": memory.importance,
                        "confidence": memory.confidence,
                        "strength": memory.strength,
                        "created_at": memory.created_at.isoformat() if memory.created_at else None,
                        "updated_at": memory.updated_at.isoformat() if memory.updated_at else None,
                        "last_accessed_at": memory.last_accessed_at.isoformat() if memory.last_accessed_at else None,
                        "access_count": memory.access_count,
                        "source_type": memory.source_type,
                        "similarity_score": similarity,
                        "final_score": final_score,
                    })
                
                scored_memories.sort(key=lambda x: x["final_score"], reverse=True)
                return scored_memories[:limit]
            except Exception:
                scored_memories = []  # Fall through to Python-side search
        
        # Fallback: Python-side cosine similarity
        # Remove the pgvector-specific condition if it was added
        conditions = [
            Memory.user_id == user_id,
            Memory.is_deleted == False,
            Memory.is_active == True,
        ]
        if brain_types:
            conditions.append(Memory.brain_type.in_(brain_types))
        if categories:
            conditions.append(Memory.category.in_(categories))
        if created_after:
            conditions.append(Memory.created_at >= created_after)
        if created_before:
            conditions.append(Memory.created_at <= created_before)
        
        query = select(Memory).where(and_(*conditions))
        result = await self.db.execute(query)
        memories = result.scalars().all()
        
        for memory in memories:
            if memory.embedding_json:
                memory_embedding = json.loads(memory.embedding_json)
                similarity = self.embedding_service.cosine_similarity(
                    embedding, memory_embedding
                )
                
                if similarity >= min_similarity:
                    final_score = (
                        similarity * 0.4 +
                        memory.strength * 0.25 +
                        memory.importance * 0.2 +
                        memory.emotional_salience * 0.15
                    )
                    
                    if memory.last_accessed_at:
                        days_since_access = (datetime.utcnow() - memory.last_accessed_at).days
                        if days_since_access < 7:
                            recency_boost = (7 - days_since_access) / 7 * 0.1
                            final_score += recency_boost
                    
                    scored_memories.append({
                        "id": memory.id,
                        "content": memory.content,
                        "summary": memory.summary,
                        "brain_type": memory.brain_type,
                        "category": memory.category,
                        "memory_type": memory.memory_type,
                        "importance": memory.importance,
                        "confidence": memory.confidence,
                        "strength": memory.strength,
                        "created_at": memory.created_at.isoformat() if memory.created_at else None,
                        "updated_at": memory.updated_at.isoformat() if memory.updated_at else None,
                        "last_accessed_at": memory.last_accessed_at.isoformat() if memory.last_accessed_at else None,
                        "access_count": memory.access_count,
                        "source_type": memory.source_type,
                        "similarity_score": similarity,
                        "final_score": final_score,
                    })
        
        scored_memories.sort(key=lambda x: x["final_score"], reverse=True)
        return scored_memories[:limit]
