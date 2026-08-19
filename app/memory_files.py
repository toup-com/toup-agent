"""Memory files — the organizing layer above individual memory rows.

Canon for the Claude-style file model (docs/memory/rebuild-2026-08.md §3).
A user's memory is a small set of curated files; every memory row belongs to
exactly one file via ``memories.file_slug``. Files are the unit of curation,
rendering and navigation; rows stay the unit of provenance, retrieval and
decay — nothing about embeddings, events or the calendar log changes.

Leaf module on purpose: stdlib only, importable from models, services,
prompts and scripts alike (mirrors app.memory_taxonomy, which it reads).

The section map below MUST stay in lockstep with the clients' shared
``memoryModel.ts`` (mobile canon, vendored to web): a category that maps to a
different section on the two sides makes an entry appear in one file in the
app and another in the prompt. ``tests/test_memory_files.py`` pins the map
against the taxonomy; the clients' guard pins theirs.
"""

from __future__ import annotations

import re
import unicodedata
from enum import Enum
from typing import Dict, Iterable, List, Optional

from app.memory_taxonomy import MemoryCategory, normalize_category


class FileSection(str, Enum):
    """Where a file sits on the Memory page. Order = display order."""

    PROFILE = "profile"
    PEOPLE = "people"
    AREAS = "areas"
    PREFERENCES = "preferences"
    KNOWLEDGE = "knowledge"
    LEARNED = "learned"
    WORKING = "working"


SECTION_ORDER: List[FileSection] = [
    FileSection.PROFILE,
    FileSection.PEOPLE,
    FileSection.AREAS,
    FileSection.PREFERENCES,
    FileSection.KNOWLEDGE,
    FileSection.LEARNED,
    FileSection.WORKING,
]

SECTION_LABEL: Dict[FileSection, str] = {
    FileSection.PROFILE: "Profile",
    FileSection.PEOPLE: "People",
    FileSection.AREAS: "Areas",
    FileSection.PREFERENCES: "Preferences",
    FileSection.KNOWLEDGE: "Knowledge",
    FileSection.LEARNED: "Learned",
    FileSection.WORKING: "Working on",
}


# ── System files ──────────────────────────────────────────────────────
# The fixed singletons every user has. People and area files are created on
# demand (one per person / topic) and are NOT system files — they can be
# deleted outright, while deleting a system file only empties it.
#
# `purpose` is the one-line header both readers see: the user on the file
# page, the model in the memory-file index. Second person is deliberate —
# storage speaks to the user ("you"), and the model is told whose memory it
# is reading by the block that injects it.

SYSTEM_FILES: Dict[str, Dict[str, str]] = {
    "profile": {
        "section": FileSection.PROFILE.value,
        "title": "Profile",
        "purpose": (
            "Who you are — identity, health, habits, and how your life is "
            "set up; read when a reply depends on personal context."
        ),
    },
    "people": {
        "section": FileSection.PEOPLE.value,
        "title": "People",
        "purpose": (
            "People in your life who don't have their own file yet; read "
            "when someone comes up."
        ),
    },
    "areas/work": {
        "section": FileSection.AREAS.value,
        "title": "Work & goals",
        "purpose": (
            "Your work, projects and goals that don't have their own file "
            "yet; read when planning or acting on them."
        ),
    },
    "preferences": {
        "section": FileSection.PREFERENCES.value,
        "title": "Preferences",
        "purpose": (
            "What you like and how you want things done — taste, tools, and "
            "how you like to be talked to; read before making choices for you."
        ),
    },
    "knowledge": {
        "section": FileSection.KNOWLEDGE.value,
        "title": "Knowledge",
        "purpose": (
            "Facts about your world — places, things, media and past "
            "experiences; read when a reply leans on them."
        ),
    },
    "learned": {
        "section": FileSection.LEARNED.value,
        "title": "Learned",
        "purpose": (
            "Corrections and lessons from working together; read before "
            "acting on your behalf."
        ),
    },
    "working": {
        "section": FileSection.WORKING.value,
        "title": "Working on",
        "purpose": (
            "What's in motion right now — tasks with a shelf life and "
            "standing arrangements; read at the start of every reply."
        ),
    },
}


# ── Category → section ────────────────────────────────────────────────
# Mirrors USER_CATEGORY_GROUP in the clients' memoryModel.ts, collapsed to
# the file sections (the clients' 'now' is WORKING; 'personal' is PROFILE;
# 'work' is AREAS; 'other'/'process' fold into KNOWLEDGE/LEARNED — files
# have no junk drawer). Keys are canonical taxonomy values; legacy values
# reach this map through normalize_category().

USER_CATEGORY_SECTION: Dict[str, FileSection] = {
    MemoryCategory.IDENTITY.value: FileSection.PROFILE,
    MemoryCategory.BELIEFS.value: FileSection.PROFILE,
    MemoryCategory.EMOTIONS.value: FileSection.PROFILE,
    MemoryCategory.HEALTH.value: FileSection.PROFILE,
    MemoryCategory.HABITS.value: FileSection.PROFILE,
    MemoryCategory.PEOPLE.value: FileSection.PEOPLE,
    MemoryCategory.RELATIONSHIPS.value: FileSection.PEOPLE,
    MemoryCategory.WORK.value: FileSection.AREAS,
    MemoryCategory.GOALS.value: FileSection.AREAS,
    MemoryCategory.FINANCE.value: FileSection.AREAS,
    MemoryCategory.SKILLS.value: FileSection.AREAS,
    MemoryCategory.PREFERENCES.value: FileSection.PREFERENCES,
    MemoryCategory.INTERACTION.value: FileSection.PREFERENCES,
    MemoryCategory.KNOWLEDGE.value: FileSection.KNOWLEDGE,
    MemoryCategory.MEDIA.value: FileSection.KNOWLEDGE,
    MemoryCategory.EXPERIENCES.value: FileSection.KNOWLEDGE,
    MemoryCategory.LOCATIONS.value: FileSection.KNOWLEDGE,
    MemoryCategory.POSSESSIONS.value: FileSection.KNOWLEDGE,
    MemoryCategory.ACTIVE_TASK.value: FileSection.WORKING,
    MemoryCategory.OTHER.value: FileSection.KNOWLEDGE,
}

# The category a NEW entry gets when only its file is known (the instruct
# path's `add` op). LEARNED is the agent brain; everything else is user brain.
SECTION_DEFAULT_CATEGORY: Dict[FileSection, str] = {
    FileSection.PROFILE: MemoryCategory.IDENTITY.value,
    FileSection.PEOPLE: MemoryCategory.PEOPLE.value,
    FileSection.AREAS: MemoryCategory.WORK.value,
    FileSection.PREFERENCES: MemoryCategory.PREFERENCES.value,
    FileSection.KNOWLEDGE: MemoryCategory.KNOWLEDGE.value,
    FileSection.LEARNED: "user_patterns",
    FileSection.WORKING: MemoryCategory.ACTIVE_TASK.value,
}

# Section a row lands in when nothing finer is known. People/areas rows that
# can be attributed to a specific person or topic get their own file instead.
DEFAULT_SLUG_FOR_SECTION: Dict[FileSection, str] = {
    FileSection.PROFILE: "profile",
    FileSection.PEOPLE: "people",
    FileSection.AREAS: "areas/work",
    FileSection.PREFERENCES: "preferences",
    FileSection.KNOWLEDGE: "knowledge",
    FileSection.LEARNED: "learned",
    FileSection.WORKING: "working",
}


def section_for(category: Optional[str], brain_type: Optional[str] = "user") -> FileSection:
    """The section a (category, brain_type) pair belongs to.

    Same precedence as the clients' groupOf(): brain decides before category.
    Unknown user categories fall to KNOWLEDGE (files have no "Other").
    """
    if brain_type == "agent":
        return FileSection.LEARNED
    raw = (category or "").strip().lower()
    if brain_type == "work":
        return FileSection.WORKING if raw == MemoryCategory.ACTIVE_TASK.value else FileSection.LEARNED
    canonical = normalize_category(raw, brain_type="user") if raw else MemoryCategory.OTHER.value
    return USER_CATEGORY_SECTION.get(canonical, FileSection.KNOWLEDGE)


def default_slug_for(category: Optional[str], brain_type: Optional[str] = "user") -> str:
    """The catch-all file slug for a (category, brain_type) pair."""
    return DEFAULT_SLUG_FOR_SECTION[section_for(category, brain_type)]


# ── Slugs ─────────────────────────────────────────────────────────────
# A slug is `name` or `section/name` where name is lowercase word characters
# and hyphens. Unicode letters are kept (a Persian name makes a Persian
# slug); everything else collapses to a hyphen. Person/topic slugs are
# namespaced under their section so `people/maya` and a hypothetical area
# `maya` can coexist.

_SLUG_MAX = 120
_SLUG_RE = re.compile(r"^[\w-]+(?:/[\w-]+)?$", re.UNICODE)
_SLUG_STRIP = re.compile(r"[^\w-]+", re.UNICODE)


def slugify(name: str) -> str:
    """A stable slug segment from a display name. '' if nothing survives."""
    normalized = unicodedata.normalize("NFKC", name or "").strip().lower()
    collapsed = _SLUG_STRIP.sub("-", normalized).replace("_", "-").strip("-")
    collapsed = re.sub(r"-{2,}", "-", collapsed)
    return collapsed[:_SLUG_MAX]


def is_valid_slug(slug: str) -> bool:
    return (
        bool(slug)
        and len(slug) <= _SLUG_MAX
        and "_" not in slug
        and bool(_SLUG_RE.match(slug))
    )


def person_slug(name: str) -> Optional[str]:
    seg = slugify(name)
    return f"people/{seg}" if seg else None


def area_slug(topic: str) -> Optional[str]:
    seg = slugify(topic)
    if not seg or seg == "work":
        return "areas/work"
    return f"areas/{seg}"


def section_of_slug(slug: str) -> FileSection:
    """The section a slug belongs to, from its shape."""
    if slug in SYSTEM_FILES:
        return FileSection(SYSTEM_FILES[slug]["section"])
    head = slug.split("/", 1)[0]
    if head == "people":
        return FileSection.PEOPLE
    if head == "areas":
        return FileSection.AREAS
    # An unnamespaced non-system slug shouldn't exist; treat as knowledge.
    return FileSection.KNOWLEDGE


def default_purpose(section: FileSection, title: str) -> str:
    """Fallback purpose line for a created person/topic file until the
    consolidation pass writes a real one."""
    if section == FileSection.PEOPLE:
        return f"{title} — someone in your life; read when they come up."
    if section == FileSection.AREAS:
        return f"{title} — an ongoing area of your life; read when it comes up."
    return SYSTEM_FILES.get(DEFAULT_SLUG_FOR_SECTION[section], {}).get("purpose", "")


# ── Rendering ─────────────────────────────────────────────────────────

def render_file_markdown(
    title: str,
    purpose: Optional[str],
    entries: Iterable[str],
    related: Optional[Iterable[str]] = None,
) -> str:
    """One file as markdown — the shape both the export and the prompt use."""
    lines: List[str] = [f"# {title}"]
    if purpose:
        lines.append(f"*{purpose}*")
    lines.append("")
    for text in entries:
        text = (text or "").strip()
        if text:
            lines.append(f"- {text}")
    related_list = [s for s in (related or []) if s]
    if related_list:
        lines.append("")
        lines.append("See also: " + ", ".join(related_list))
    return "\n".join(lines).rstrip() + "\n"


def file_index_line(slug: str, title: str, purpose: Optional[str], entry_count: int) -> str:
    """One line of the memory-file index injected into the prompt."""
    head = f"- {slug} — {title}"
    if purpose:
        head += f": {purpose}"
    return f"{head} ({entry_count} entr{'y' if entry_count == 1 else 'ies'})"
