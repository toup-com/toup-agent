"""Canonical memory taxonomy — the single source of truth.

Historically these enums were declared TWICE with divergent values: once in
`app/schemas.py` (used by the extractor and the API) and once in
`app/db/models/enums.py` (used by the DB model, brain stats and the mobile
label map). Only 12 of ~21 category values overlapped, so the extractor wrote
`schedule`/`projects`/`learning`/`tools`/`family`/`places`/`food`/`travel`
rows that the app had no label for and rendered as "Other" — 28% of
production rows on the founder's tenant (2026-07-29 audit).

A third copy of the taxonomy lived as prose inside the extraction prompt, so
even fixing the two enums would have left the LLM emitting a stale list.
`build_category_prompt_block()` below now GENERATES that prompt section, which
makes prose drift structurally impossible.

This module is a leaf: it imports nothing but `enum`. Both `app/schemas.py`
and `app/db/models/enums.py` re-export from here, which keeps every legacy
import path working while there is exactly one definition. It must stay
import-free — `app.db.models.enums` cannot be imported from `app.schemas`
directly, because that executes `app/db/models/__init__.py` and pulls in every
SQLAlchemy model.
"""

import re
from enum import Enum
from typing import Dict, Optional, Set


class BrainType(str, Enum):
    """Which brain a memory belongs to."""
    USER = "user"       # Facts about the user
    AGENT = "agent"     # What the agent has learned about working with them
    WORK = "work"       # Operational/process knowledge


class MemoryCategory(str, Enum):
    """USER-brain categories. Canonical set — the app's labels derive from this.

    Every value carries a human-readable label and a definition (see
    CATEGORY_GUIDE) so the extraction prompt can be generated rather than
    hand-maintained.
    """
    # Personal profile
    IDENTITY = "identity"
    PREFERENCES = "preferences"
    BELIEFS = "beliefs"
    EMOTIONS = "emotions"

    # Social
    PEOPLE = "people"
    RELATIONSHIPS = "relationships"

    # Life & work
    EXPERIENCES = "experiences"
    GOALS = "goals"
    HABITS = "habits"
    SKILLS = "skills"
    WORK = "work"
    FINANCE = "finance"
    HEALTH = "health"

    # World & environment
    KNOWLEDGE = "knowledge"
    LOCATIONS = "locations"
    POSSESSIONS = "possessions"
    MEDIA = "media"

    # System
    INTERACTION = "interaction"
    ACTIVE_TASK = "active_task"
    OTHER = "other"


# Backwards-compatible alias (pre-2026 name).
BrainRegion = MemoryCategory


class AgentCategory(str, Enum):
    """AGENT-brain categories — what the agent learns about working with a user.

    This is what the app's "Learned" tab shows. Values are unchanged from
    `db/models/enums.py`; `schemas.py`'s divergent `agent_*` set is mapped in
    via AGENT_CATEGORY_ALIASES.
    """
    CORRECTIONS = "corrections"                 # The user corrected the agent
    USER_PATTERNS = "user_patterns"             # How this user tends to work
    CONVERSATION_STYLE = "conversation_style"   # How they like to be talked to
    TOOL_USAGE = "tool_usage"                   # What worked / failed with tools
    DOMAIN_KNOWLEDGE = "domain_knowledge"       # Their domain, learned over time
    SKILLS_LEARNED = "skills_learned"           # Procedures the agent picked up
    PREFERENCES = "preferences"                 # Working preferences


class WorkCategory(str, Enum):
    """WORK-brain categories — operational knowledge."""
    PROCESS = "process"


class MemoryType(str, Enum):
    """What kind of information a memory captures.

    Canonical set is the 12-value superset that production data already uses
    (`db/models/enums.py` previously declared only 5, but every write path used
    the 12-value copy in `schemas.py`, so narrowing would orphan live rows).
    """
    FACT = "fact"
    PREFERENCE = "preference"
    TASK = "task"
    EVENT = "event"
    PERSON = "person"
    PLACE = "place"
    PROJECT = "project"
    DECISION = "decision"
    SKILL = "skill"
    FILE = "file"
    NOTE = "note"
    CONVERSATION = "conversation"


class MemoryLevel(str, Enum):
    """Cognitive hierarchy: episodic experiences consolidate into semantic facts."""
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    META = "meta"


# ── Legacy value mapping ──────────────────────────────────────────────
# Values written by the pre-unification extractor (the `schemas.py` copy),
# plus a few near-miss strings LLMs emit. Applied on every read and write via
# normalize_category(), and by the one-shot migration in
# scripts/migrate_memory_taxonomy.py.
MEMORY_CATEGORY_ALIASES: Dict[str, MemoryCategory] = {
    # schemas.py values with no db/models/enums.py equivalent
    "places": MemoryCategory.LOCATIONS,
    "family": MemoryCategory.PEOPLE,
    "projects": MemoryCategory.WORK,
    "schedule": MemoryCategory.ACTIVE_TASK,
    "learning": MemoryCategory.SKILLS,
    "tools": MemoryCategory.POSSESSIONS,
    "food": MemoryCategory.PREFERENCES,
    "travel": MemoryCategory.EXPERIENCES,
    "context": MemoryCategory.OTHER,
    # Common LLM near-misses
    "person": MemoryCategory.PEOPLE,
    "place": MemoryCategory.LOCATIONS,
    "location": MemoryCategory.LOCATIONS,
    "project": MemoryCategory.WORK,
    "goal": MemoryCategory.GOALS,
    "habit": MemoryCategory.HABITS,
    "skill": MemoryCategory.SKILLS,
    "task": MemoryCategory.ACTIVE_TASK,
    "tasks": MemoryCategory.ACTIVE_TASK,
    "reminder": MemoryCategory.ACTIVE_TASK,
    "reminders": MemoryCategory.ACTIVE_TASK,
    "appointment": MemoryCategory.ACTIVE_TASK,
    "calendar": MemoryCategory.ACTIVE_TASK,
    "possession": MemoryCategory.POSSESSIONS,
    "belonging": MemoryCategory.POSSESSIONS,
    "belongings": MemoryCategory.POSSESSIONS,
    "relationship": MemoryCategory.RELATIONSHIPS,
    "emotion": MemoryCategory.EMOTIONS,
    "feeling": MemoryCategory.EMOTIONS,
    "feelings": MemoryCategory.EMOTIONS,
    "belief": MemoryCategory.BELIEFS,
    "experience": MemoryCategory.EXPERIENCES,
    "money": MemoryCategory.FINANCE,
    "financial": MemoryCategory.FINANCE,
    "job": MemoryCategory.WORK,
    "career": MemoryCategory.WORK,
    "uncategorized": MemoryCategory.OTHER,
    "general": MemoryCategory.OTHER,
    "misc": MemoryCategory.OTHER,
}

AGENT_CATEGORY_ALIASES: Dict[str, AgentCategory] = {
    # schemas.py `agent_*` values
    "agent_tools": AgentCategory.TOOL_USAGE,
    "agent_skills": AgentCategory.SKILLS_LEARNED,
    "agent_procedures": AgentCategory.SKILLS_LEARNED,
    "agent_patterns": AgentCategory.USER_PATTERNS,
    "agent_decisions": AgentCategory.CORRECTIONS,
    # agent_soul rows are Soul-owned and were retired by soul_compiler; the
    # closest surviving concept is how the agent talks to this user.
    "agent_soul": AgentCategory.CONVERSATION_STYLE,
    "correction": AgentCategory.CORRECTIONS,
    "pattern": AgentCategory.USER_PATTERNS,
    "patterns": AgentCategory.USER_PATTERNS,
    "style": AgentCategory.CONVERSATION_STYLE,
}

MEMORY_TYPE_ALIASES: Dict[str, MemoryType] = {
    "entity": MemoryType.FACT,
    "opinion": MemoryType.PREFERENCE,
    "todo": MemoryType.TASK,
    "reminder": MemoryType.TASK,
    "meeting": MemoryType.EVENT,
    "document": MemoryType.FILE,
}


# ── Transience ────────────────────────────────────────────────────────
# Default lifetime for memories the extractor flags as time-bounded. A
# reminder to "eat tea in 2 minutes" has no business outliving the day; a
# project the user is actively working on should survive a couple of weeks
# without being restated. These are FALLBACKS — the extractor may return an
# explicit horizon, which always wins.
DEFAULT_TTL_DAYS: Dict[MemoryCategory, int] = {
    MemoryCategory.ACTIVE_TASK: 7,
    MemoryCategory.EMOTIONS: 14,
    MemoryCategory.WORK: 30,
}

# Applied when the extractor says "transient" but names no category-specific
# horizon and the category has no DEFAULT_TTL_DAYS entry.
FALLBACK_TTL_DAYS = 14

# Categories that describe durable facts about a person. A memory in one of
# these is never auto-expired even if the extractor mislabels it transient —
# losing "user's daughter is called Mira" to a TTL bug is unacceptable in a
# way that losing a stale reminder is not.
NEVER_EXPIRE_CATEGORIES: Set[MemoryCategory] = {
    MemoryCategory.IDENTITY,
    MemoryCategory.PEOPLE,
    MemoryCategory.RELATIONSHIPS,
    MemoryCategory.BELIEFS,
    MemoryCategory.HEALTH,
    MemoryCategory.FINANCE,
    MemoryCategory.SKILLS,
    MemoryCategory.PREFERENCES,
}


# ── Prompt generation ─────────────────────────────────────────────────
# One-line definitions. These are what the extraction LLM actually reads, and
# the reason the taxonomy can no longer drift: the prompt is built from this
# dict, so adding an enum value without a definition raises at import time.
CATEGORY_GUIDE: Dict[MemoryCategory, str] = {
    MemoryCategory.IDENTITY: "who the user is — name, age, background, where they're from",
    MemoryCategory.PREFERENCES: "likes, dislikes, favourites, tastes, how they take their coffee",
    MemoryCategory.BELIEFS: "values, opinions, worldview, principles they hold",
    MemoryCategory.EMOTIONS: "how they felt about something — moods and reactions",
    MemoryCategory.PEOPLE: "a specific person in their life and who that person is to them",
    MemoryCategory.RELATIONSHIPS: "how two people they know are connected to each other",
    MemoryCategory.EXPERIENCES: "something that happened to them — trips, events, stories",
    MemoryCategory.GOALS: "what they are trying to achieve, long-term ambitions",
    MemoryCategory.HABITS: "recurring routines and patterns in their life",
    MemoryCategory.SKILLS: "what they can do, are learning, or are studying",
    MemoryCategory.WORK: "their job, career, company, and the projects they work on",
    MemoryCategory.FINANCE: "money, budgets, subscriptions, accounts",
    MemoryCategory.HEALTH: "medical facts, fitness, diet, wellbeing",
    MemoryCategory.KNOWLEDGE: "a fact about the world that is personally useful TO THIS USER",
    MemoryCategory.LOCATIONS: "places that matter to them — home, office, cities, venues",
    MemoryCategory.POSSESSIONS: "things they own or use — devices, software, tools, accounts",
    MemoryCategory.MEDIA: "books, films, music, shows, articles they consume",
    MemoryCategory.INTERACTION: "how they want the assistant to behave and communicate",
    MemoryCategory.ACTIVE_TASK: "something they are working on or asked for RIGHT NOW — "
                                "reminders, appointments, one-off requests (time-bounded)",
    MemoryCategory.OTHER: "genuinely does not fit any category above",
}

AGENT_CATEGORY_GUIDE: Dict[AgentCategory, str] = {
    AgentCategory.CORRECTIONS: "the user corrected you — a mistake you made and the right answer",
    AgentCategory.USER_PATTERNS: "how this user tends to work, decide, or phrase things",
    AgentCategory.CONVERSATION_STYLE: "how they want you to talk — length, tone, formatting",
    AgentCategory.TOOL_USAGE: "a tool or approach that worked well, or failed, for this user",
    AgentCategory.DOMAIN_KNOWLEDGE: "their domain/company specifics you had to learn",
    AgentCategory.SKILLS_LEARNED: "a repeatable procedure you worked out with them",
    AgentCategory.PREFERENCES: "their working preferences for how you do tasks",
}


def _assert_guides_complete() -> None:
    """Fail loudly at import if an enum value has no definition.

    This is the mechanism that keeps the prompt honest: you cannot add a
    category without telling the extractor what it means.
    """
    missing = [c.value for c in MemoryCategory if c not in CATEGORY_GUIDE]
    if missing:
        raise RuntimeError(
            f"memory_taxonomy: MemoryCategory values missing from CATEGORY_GUIDE: {missing}"
        )
    missing_agent = [c.value for c in AgentCategory if c not in AGENT_CATEGORY_GUIDE]
    if missing_agent:
        raise RuntimeError(
            f"memory_taxonomy: AgentCategory values missing from AGENT_CATEGORY_GUIDE: "
            f"{missing_agent}"
        )
    # An alias must never shadow a real value, or normalize() would silently
    # rewrite a valid category into a different one.
    canonical = {c.value for c in MemoryCategory}
    shadowed = sorted(canonical & set(MEMORY_CATEGORY_ALIASES))
    if shadowed:
        raise RuntimeError(
            f"memory_taxonomy: aliases shadow canonical categories: {shadowed}"
        )


_assert_guides_complete()


def build_category_prompt_block() -> str:
    """Render the category list for the extraction prompt.

    Replaces the hand-written bare list that used to sit at
    memory_extractor.py:642 with no definitions — the direct cause of the
    taxonomy collapsing to three visible buckets in production.
    """
    lines = [f"   - {cat.value}: {desc}" for cat, desc in CATEGORY_GUIDE.items()]
    return "\n".join(lines)


def build_agent_category_prompt_block() -> str:
    """Render the agent-brain category list for the reflection prompt."""
    lines = [f"   - {cat.value}: {desc}" for cat, desc in AGENT_CATEGORY_GUIDE.items()]
    return "\n".join(lines)


# ── Normalisation ─────────────────────────────────────────────────────

def normalize_category(
    value: Optional[str],
    brain_type: str = BrainType.USER.value,
) -> str:
    """Coerce any category string to a canonical value for the given brain.

    Unknown values fall back to `other` (user/work) or `domain_knowledge`
    (agent) rather than raising — an unrecognised category must never cost us
    the memory itself. Returns a plain `str` so callers can assign it straight
    onto the SQLAlchemy column.
    """
    raw = (value or "").strip().lower().replace(" ", "_").replace("-", "_")

    if brain_type == BrainType.AGENT.value:
        try:
            return AgentCategory(raw).value
        except ValueError:
            pass
        mapped_agent = AGENT_CATEGORY_ALIASES.get(raw)
        if mapped_agent is not None:
            return mapped_agent.value
        return AgentCategory.DOMAIN_KNOWLEDGE.value

    if brain_type == BrainType.WORK.value:
        try:
            return WorkCategory(raw).value
        except ValueError:
            return WorkCategory.PROCESS.value

    try:
        return MemoryCategory(raw).value
    except ValueError:
        pass
    mapped = MEMORY_CATEGORY_ALIASES.get(raw)
    if mapped is not None:
        return mapped.value
    return MemoryCategory.OTHER.value


def normalize_memory_type(value: Optional[str]) -> str:
    """Coerce any memory_type string to a canonical value; unknown → fact."""
    raw = (value or "").strip().lower().replace(" ", "_").replace("-", "_")
    try:
        return MemoryType(raw).value
    except ValueError:
        pass
    mapped = MEMORY_TYPE_ALIASES.get(raw)
    if mapped is not None:
        return mapped.value
    return MemoryType.FACT.value


# ── Entity-relationship categorisation ────────────────────────────────
# `store_entity_relationship` used to write
#     category = "people" if source_type == "person" else "knowledge"
# — a hardcoded binary that single-handedly manufactured two of the three
# categories visible in production. These map an entity's type to the category
# a memory ABOUT that entity belongs in.
ENTITY_TYPE_TO_CATEGORY: Dict[str, MemoryCategory] = {
    "person": MemoryCategory.PEOPLE,
    "place": MemoryCategory.LOCATIONS,
    "location": MemoryCategory.LOCATIONS,
    "organization": MemoryCategory.WORK,
    "company": MemoryCategory.WORK,
    "project": MemoryCategory.WORK,
    "skill": MemoryCategory.SKILLS,
    "tool": MemoryCategory.POSSESSIONS,
    "software": MemoryCategory.POSSESSIONS,
    "file": MemoryCategory.POSSESSIONS,
    "event": MemoryCategory.EXPERIENCES,
    "media": MemoryCategory.MEDIA,
    "book": MemoryCategory.MEDIA,
    "movie": MemoryCategory.MEDIA,
    "show": MemoryCategory.MEDIA,
    "music": MemoryCategory.MEDIA,
    "topic": MemoryCategory.KNOWLEDGE,
    "decision": MemoryCategory.BELIEFS,
    "note": MemoryCategory.OTHER,
    "conversation": MemoryCategory.OTHER,
}

# A relationship's category should say what the edge is ABOUT.
#
# PEOPLE is LAST on purpose, so it wins only when it is the sole candidate —
# i.e. when BOTH ends are people. An earlier version put it first, reasoning
# that "person -> works_at -> organization is about the person". That is
# backwards: when a person is one end, the OTHER end is the subject matter.
# The founder pointed at the symptom directly — "User owns Toup" filed under
# People — and 24 of 72 relationship rows on that tenant had collapsed into
# PEOPLE for the same reason.
#
#   person/organization -> work        ("User owns Toup")
#   person/person       -> people      ("Nariman chatted with majid")
#   person/skill        -> skills      (the user's skill, not the user)
#   show/organization   -> media       ("Better Call Saul is available on Netflix")
#
# MEDIA and LOCATIONS sit ahead of WORK because `organization` is the broadest
# bucket in ENTITY_TYPE_TO_CATEGORY — a streaming service or a mall would
# otherwise swallow the more specific side of the edge.
_RELATIONSHIP_CATEGORY_PRECEDENCE = (
    MemoryCategory.MEDIA,
    MemoryCategory.LOCATIONS,
    MemoryCategory.SKILLS,
    MemoryCategory.WORK,
    MemoryCategory.POSSESSIONS,
    MemoryCategory.EXPERIENCES,
    MemoryCategory.BELIEFS,
    MemoryCategory.KNOWLEDGE,
    MemoryCategory.PEOPLE,
)


def category_for_relationship(
    source_type: Optional[str],
    target_type: Optional[str],
) -> str:
    """Pick the category for a memory describing an entity-to-entity edge."""
    candidates = []
    for raw in (source_type, target_type):
        cat = ENTITY_TYPE_TO_CATEGORY.get((raw or "").strip().lower())
        if cat is not None:
            candidates.append(cat)
    if not candidates:
        return MemoryCategory.KNOWLEDGE.value
    for cat in _RELATIONSHIP_CATEGORY_PRECEDENCE:
        if cat in candidates:
            return cat.value
    return candidates[0].value


# ── Recurrence guard ──────────────────────────────────────────────────
# A one-off reminder ("remind me to eat tea in 2 minutes") should expire. A
# STANDING arrangement ("send me a Gmail briefing every day at 11:49") is a
# durable preference that happens to be phrased like a schedule, and losing it
# would stop the agent knowing about a routine the user still relies on.
#
# Both land in ACTIVE_TASK, so category alone cannot separate them. Checked
# against the founder's live tenant: of 28 legacy `schedule` rows, 9 describe
# recurring arrangements (daily briefings, daily quotes) and 19 are genuinely
# one-off. `ref_kind`/`ref_id` would be the principled link to the routines
# table, but it is unset on 28/28 of those rows, so content is the only
# available signal.
_RECURRING = re.compile(
    r"\b("
    r"every\s+(?:day|morning|afternoon|evening|night|week|month|hour|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    r"|everyday|daily|weekly|monthly|hourly"
    r"|each\s+(?:day|morning|afternoon|evening|night|week|month)"
    r"|recurring|routine|on\s+weekdays|every\s+other\s+day"
    r")\b",
    re.IGNORECASE,
)


def describes_recurring_arrangement(content: Optional[str]) -> bool:
    """True if this memory describes a standing/recurring arrangement.

    Used to exempt such memories from TTL expiry even when their category is
    otherwise transient.
    """
    return bool(content and _RECURRING.search(content))


# ── Relationship phrasing ─────────────────────────────────────────────
#
# Knowledge-graph edges are stored as (source, predicate, target) triples and
# the triple is preserved verbatim in metadata_json. What a PERSON reads must
# be a sentence.
#
# The predicate is invented by the LLM per extraction, so the vocabulary is
# unbounded and inconsistent — one live tenant carried 40 distinct predicates
# including connects_to/connected_to, chats_with/has_chat_with/chatted_with,
# sends_to/sent_to and scheduled_in_timezone/scheduled_in_time_zone. Rendering
# `predicate.replace("_", " ")` is only accidentally grammatical: `performed_by`
# reads fine, `play_on` produces "Better Call Saul play on Netflix".
#
# So: normalise the predicate to a canonical key, then render from a template.
# Unknown predicates fall back to a repaired verb phrase rather than raw text.

_PREDICATE_ALIASES: Dict[str, str] = {
    "connected_to": "connects_to",
    "connects_to": "connects_to",
    "chatted_with": "chats_with",
    "has_chat_with": "chats_with",
    "chats_with": "chats_with",
    "sent_to": "sends_to",
    "sends_to": "sends_to",
    "is_recipient_of": "receives",
    "receives_daily": "receives",
    "receives": "receives",
    "fetches_messages_from": "fetches_from",
    "fetches_from": "fetches_from",
    "are_from": "is_from",
    "is_from": "is_from",
    "scheduled_in_time_zone": "scheduled_in_timezone",
    "scheduled_in_timezone": "scheduled_in_timezone",
    "avoids_repeating_quotes_from": "avoid_repeating_quotes_from",
    "avoid_repeating_quotes_from": "avoid_repeating_quotes_from",
    "play_on": "available_on",
    "available_on": "available_on",
    "owner_of": "owns",
    "owns": "owns",
    "reconnect_using": "reconnects_using",
    "reconnects_using": "reconnects_using",
    "summarizes_new_messages_from": "summarizes",
    "summarizes": "summarizes",
}

# "{s}" = source, "{t}" = target.
_PREDICATE_TEMPLATES: Dict[str, str] = {
    "available_on": "{s} is available on {t}",
    "located_in": "{s} is in {t}",
    "happens_in": "{s} takes place in {t}",
    "is_included_in": "{s} is included in {t}",
    "is_about": "{s} is about {t}",
    "is_from": "{s} comes from {t}",
    "owns": "{s} owns {t}",
    "develops": "{s} develops {t}",
    "created": "{s} created {t}",
    "created_content_about": "{s} created content about {t}",
    "works_on": "{s} works on {t}",
    "researches": "{s} is researching {t}",
    "reviews": "{s} reviews {t}",
    "uses": "{s} uses {t}",
    "can_use": "{s} can use {t}",
    "has_account_on": "{s} has an account on {t}",
    "affiliated_with": "{s} is affiliated with {t}",
    "connects_to": "{s} is connected to {t}",
    "reconnects_using": "{s} reconnects to {t}",
    "chats_with": "{s} chats with {t}",
    "sends_to": "{s} sends {t}",
    "receives": "{s} receives {t}",
    "fetches_from": "{s} fetches messages from {t}",
    "summarizes": "{s} summarizes {t}",
    "performed_by": "{t} performs {s}",
    "is_agent_of": "{s} is {t}'s assistant",
    "scheduled_in_timezone": "{s} is scheduled in {t}",
    "should_be_sent_via": "{s} should be sent via {t}",
    "should_not_be_sent_via": "{s} should not be sent via {t}",
    "avoid_repeating_quotes_from": "{s} should not repeat quotes from {t}",
    "has_attraction_toward": "{s} is attracted to {t}",
    "has_childhood_connection_with": "{s} knew {t} in childhood",
    "is_drawn_to": "{s} is drawn to {t}",
    "has_to_call": "{s} needs to call {t}",
    "reconnects": "{s} reconnects to {t}",
}

# Predicates already in third-person singular ("develops", "owns") need no
# repair; bare stems ("play", "connect") read as imperatives after a subject.
_THIRD_PERSON = re.compile(r"(?:s|ed|ing)$", re.IGNORECASE)

# Words that must NEVER be given a third-person "-s". Modals and auxiliaries
# are already finite, and the extractor does emit them inside predicates —
# `might_cause_problems_in` on a live tenant became "Rampage mights cause
# problems in Canada", which is worse than the raw form this function exists
# to improve.
_NEVER_INFLECT = frozenset({
    "might", "may", "can", "could", "shall", "should", "will", "would",
    "must", "ought", "is", "are", "was", "were", "be", "been", "being",
    "has", "have", "had", "does", "do", "did", "cannot", "wont",
})


def humanize_relationship(
    source_name: str,
    relationship: str,
    target_name: str,
) -> str:
    """Render a knowledge-graph edge as a sentence a person can read.

    The triple itself is never lost — callers keep it in `metadata_json`. This
    governs only what a human sees on the Memory screen and what gets embedded
    for search, both of which are better served by prose than by predicate-ese.
    """
    source = (source_name or "").strip()
    target = (target_name or "").strip()
    raw = (relationship or "").strip()
    if not raw:
        return f"{source} — {target}".strip(" —")

    key = raw.lower().replace(" ", "_")
    key = _PREDICATE_ALIASES.get(key, key)

    template = _PREDICATE_TEMPLATES.get(key)
    if template:
        return template.format(s=source, t=target).strip()

    # Unknown predicate. Repair the verb so the result is at least a sentence:
    # a bare stem gets "-s", anything already inflected is left alone.
    words = key.split("_")
    head = words[0]
    if head and head.lower() not in _NEVER_INFLECT and not _THIRD_PERSON.search(head):
        if head.endswith(("s", "x", "z", "ch", "sh")):
            head += "es"
        elif head.endswith("y") and len(head) > 1 and head[-2] not in "aeiou":
            head = head[:-1] + "ies"
        else:
            head += "s"
    phrase = " ".join([head] + words[1:])
    return f"{source} {phrase} {target}".strip()


def resolve_ttl_days(
    category: str,
    explicit_days: Optional[float] = None,
) -> Optional[int]:
    """Decide how long a transient memory should live.

    Returns None when the memory must NOT expire — either because its category
    holds durable personal facts, or because nothing suggests a horizon.
    `explicit_days` (from the extractor) wins when the category permits expiry.
    """
    canonical = normalize_category(category)
    try:
        cat = MemoryCategory(canonical)
    except ValueError:
        return None

    if cat in NEVER_EXPIRE_CATEGORIES:
        return None

    if explicit_days is not None:
        try:
            days = int(round(float(explicit_days)))
        except (TypeError, ValueError):
            days = 0
        # Clamp: sub-day horizons still get a full day of grace so a reminder
        # stays visible for the rest of the day it was created.
        if days >= 1:
            return min(days, 365)
        return 1

    if cat in DEFAULT_TTL_DAYS:
        return DEFAULT_TTL_DAYS[cat]
    return FALLBACK_TTL_DAYS


__all__ = [
    "BrainType",
    "MemoryCategory",
    "BrainRegion",
    "AgentCategory",
    "WorkCategory",
    "MemoryType",
    "MemoryLevel",
    "MEMORY_CATEGORY_ALIASES",
    "AGENT_CATEGORY_ALIASES",
    "MEMORY_TYPE_ALIASES",
    "DEFAULT_TTL_DAYS",
    "FALLBACK_TTL_DAYS",
    "NEVER_EXPIRE_CATEGORIES",
    "CATEGORY_GUIDE",
    "AGENT_CATEGORY_GUIDE",
    "ENTITY_TYPE_TO_CATEGORY",
    "category_for_relationship",
    "build_category_prompt_block",
    "build_agent_category_prompt_block",
    "normalize_category",
    "normalize_memory_type",
    "resolve_ttl_days",
]
