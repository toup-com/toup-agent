"""Memory file service — list, read, organize and delete curated memory files.

The file model's canon lives in app.memory_files; this service is the DB
half. Two modes, decided by what exists rather than by configuration:

- REAL (agent container): `memory_files` rows exist (or are created by
  `organize_user`), every active memory carries a `file_slug`, and the file
  view is authoritative and curated.
- VIRTUAL (platform read-fallback, or an agent whose organize pass hasn't
  run yet): the same view is synthesized from memories rows alone via the
  category→section map. Organized-but-uncurated, never a lie — and the
  platform DB deliberately has no memory_files table, so this path never
  touches it.

Entry counts and freshness are always computed live from memories rows.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    Entity, EntityLink, Memory, MemoryEvent, MemoryEventType, MemoryFile,
)
from app.memory_files import (
    SYSTEM_FILES,
    FileSection,
    SECTION_ORDER,
    default_purpose,
    default_slug_for,
    is_valid_slug,
    person_slug,
    section_of_slug,
)

logger = logging.getLogger(__name__)

# Entity types that name a person. The extractor's schema types are open
# strings; these are the values seen in production plus obvious variants.
# Public: agent_runner uses it to pull person names out of a turn's extracted
# entities before handing the batch to dedup for file routing.
PERSON_ENTITY_TYPES = ("person", "people", "family", "friend", "contact")
_PERSON_ENTITY_TYPES = PERSON_ENTITY_TYPES

# Categories whose rows are about a person and may earn a person file.
_PERSON_CATEGORIES = ("people", "relationships", "family")


def _title_from_slug(slug: str) -> str:
    """Display title for a file we only know by slug (virtual fallback)."""
    if slug in SYSTEM_FILES:
        return SYSTEM_FILES[slug]["title"]
    seg = slug.split("/", 1)[-1].replace("-", " ").strip()
    return seg[:1].upper() + seg[1:] if seg else slug


def _freshest(m: Memory) -> datetime:
    return m.updated_at or m.created_at


class MemoryFileService:
    def __init__(self, db: AsyncSession):
        self.db = db

    # ── Reads ─────────────────────────────────────────────────────────

    async def _active_rows_light(self, user_id: str) -> List[Memory]:
        """Every active, non-deleted memory row — the fold source for
        counts/freshness. Column-light on purpose; a user's corpus is
        hundreds of rows, not millions."""
        result = await self.db.execute(
            select(Memory).where(
                and_(
                    Memory.user_id == user_id,
                    Memory.is_deleted == False,  # noqa: E712
                    Memory.is_active == True,  # noqa: E712
                )
            )
        )
        return list(result.scalars().all())

    def _slug_of(self, m: Memory) -> str:
        return m.file_slug or default_slug_for(m.category, m.brain_type)

    async def list_files(self, user_id: str, *, virtual_only: bool = False) -> Dict:
        """The sectioned file listing.

        Returns {"sections": [{section, label, files: [...]}], "organized"}.
        Every system file appears even when empty; person/topic files appear
        once they exist. `organized` is False on the virtual path so clients
        can label freshness honestly.
        """
        rows = await self._active_rows_light(user_id)

        counts: Dict[str, int] = {}
        latest: Dict[str, datetime] = {}
        for m in rows:
            slug = self._slug_of(m)
            counts[slug] = counts.get(slug, 0) + 1
            when = _freshest(m)
            if slug not in latest or when > latest[slug]:
                latest[slug] = when

        file_rows: List[MemoryFile] = []
        if not virtual_only:
            result = await self.db.execute(
                select(MemoryFile).where(MemoryFile.user_id == user_id)
                .order_by(MemoryFile.position, MemoryFile.created_at)
            )
            file_rows = list(result.scalars().all())

        organized = bool(file_rows)

        files: Dict[str, Dict] = {}
        for f in file_rows:
            files[f.slug] = {
                "slug": f.slug,
                "section": f.section,
                "title": f.title,
                "purpose": f.purpose,
                "related": json.loads(f.related_json) if f.related_json else [],
                "is_system": f.is_system,
                "entry_count": counts.get(f.slug, 0),
                "updated_at": (latest.get(f.slug) or f.updated_at or f.created_at).isoformat(),
                "consolidated_at": f.consolidated_at.isoformat() if f.consolidated_at else None,
            }
        # System files always exist in the view, real rows or not.
        for slug, spec in SYSTEM_FILES.items():
            if slug not in files:
                files[slug] = {
                    "slug": slug,
                    "section": spec["section"],
                    "title": spec["title"],
                    "purpose": spec["purpose"],
                    "related": [],
                    "is_system": True,
                    "entry_count": counts.get(slug, 0),
                    "updated_at": latest.get(slug).isoformat() if latest.get(slug) else None,
                    "consolidated_at": None,
                }
        # Rows pointing at a file with no row and no system spec (virtual
        # view of an organized corpus read through the fallback).
        for slug, n in counts.items():
            if slug not in files:
                files[slug] = {
                    "slug": slug,
                    "section": section_of_slug(slug).value,
                    "title": _title_from_slug(slug),
                    "purpose": None,
                    "related": [],
                    "is_system": False,
                    "entry_count": n,
                    "updated_at": latest[slug].isoformat(),
                    "consolidated_at": None,
                }

        sections = []
        for section in SECTION_ORDER:
            in_section = [f for f in files.values() if f["section"] == section.value]
            in_section.sort(key=lambda f: (not f["is_system"], f["title"].lower()))
            sections.append({
                "section": section.value,
                "files": in_section,
            })

        return {"sections": sections, "organized": organized}

    async def get_file(
        self, user_id: str, slug: str, *, virtual_only: bool = False
    ) -> Optional[Tuple[Dict, List[Memory]]]:
        """One file's metadata + its active entries in curated order.

        None when the slug is unknown AND holds no entries — a system slug
        always resolves (empty file). `virtual_only` skips the memory_files
        table entirely (the platform DB doesn't have it)."""
        if not is_valid_slug(slug):
            return None

        file_row = None
        if not virtual_only:
            result = await self.db.execute(
                select(MemoryFile).where(
                    and_(MemoryFile.user_id == user_id, MemoryFile.slug == slug)
                )
            )
            file_row = result.scalar_one_or_none()

        rows = await self._active_rows_light(user_id)
        entries = [m for m in rows if self._slug_of(m) == slug]
        entries.sort(key=lambda m: (
            m.file_position if m.file_position is not None else 10**9,
            m.created_at or datetime.min,
        ))

        if file_row is None and slug not in SYSTEM_FILES and not entries:
            return None

        spec = SYSTEM_FILES.get(slug)
        info = {
            "slug": slug,
            "section": (file_row.section if file_row else (spec["section"] if spec else section_of_slug(slug).value)),
            "title": (file_row.title if file_row else (spec["title"] if spec else _title_from_slug(slug))),
            "purpose": (file_row.purpose if file_row else (spec["purpose"] if spec else None)),
            "related": (json.loads(file_row.related_json) if file_row and file_row.related_json else []),
            "is_system": (file_row.is_system if file_row else slug in SYSTEM_FILES),
            "entry_count": len(entries),
            "updated_at": (
                max((_freshest(m) for m in entries), default=None)
                or (file_row.updated_at if file_row else None)
            ),
            "consolidated_at": file_row.consolidated_at.isoformat() if file_row and file_row.consolidated_at else None,
        }
        if isinstance(info["updated_at"], datetime):
            info["updated_at"] = info["updated_at"].isoformat()
        return info, entries

    # ── Writes ────────────────────────────────────────────────────────

    async def ensure_file(
        self,
        user_id: str,
        slug: str,
        *,
        title: Optional[str] = None,
        purpose: Optional[str] = None,
        commit: bool = False,
    ) -> Optional[MemoryFile]:
        """Get-or-create a file row. Flushes; commits only when asked."""
        if not is_valid_slug(slug):
            return None
        result = await self.db.execute(
            select(MemoryFile).where(
                and_(MemoryFile.user_id == user_id, MemoryFile.slug == slug)
            )
        )
        existing = result.scalar_one_or_none()
        if existing:
            return existing

        spec = SYSTEM_FILES.get(slug)
        section = FileSection(spec["section"]) if spec else section_of_slug(slug)
        resolved_title = title or (spec["title"] if spec else _title_from_slug(slug))
        row = MemoryFile(
            user_id=user_id,
            slug=slug,
            section=section.value,
            title=resolved_title,
            purpose=purpose or (spec["purpose"] if spec else default_purpose(section, resolved_title)),
            is_system=slug in SYSTEM_FILES,
            position=list(SYSTEM_FILES).index(slug) if slug in SYSTEM_FILES else 100,
        )
        self.db.add(row)
        await self.db.flush()
        if commit:
            await self.db.commit()
        return row

    async def _next_position(self, user_id: str, slug: str) -> int:
        result = await self.db.execute(
            select(func.max(Memory.file_position)).where(
                and_(Memory.user_id == user_id, Memory.file_slug == slug)
            )
        )
        current = result.scalar()
        return (current + 1) if current is not None else 0

    async def assign_to_file(
        self, memory: Memory, slug: str, *, position: Optional[int] = None
    ) -> None:
        """Point a row at a file and stamp its order. Caller commits."""
        memory.file_slug = slug
        memory.file_position = (
            position if position is not None
            else await self._next_position(memory.user_id, slug)
        )

    async def route_slug(
        self,
        user_id: str,
        category: Optional[str],
        brain_type: Optional[str],
        *,
        person_names: Optional[List[str]] = None,
    ) -> str:
        """The file a new memory belongs in. Creates the person file when a
        people-category row names exactly one known person."""
        if (brain_type or "user") == "user" and (category or "").lower() in _PERSON_CATEGORIES:
            names = [n for n in (person_names or []) if n and n.strip()]
            if len(names) == 1:
                slug = person_slug(names[0])
                if slug:
                    await self.ensure_file(user_id, slug, title=names[0].strip())
                    return slug
        slug = default_slug_for(category, brain_type)
        await self.ensure_file(user_id, slug)
        return slug

    async def _person_names_by_memory(self, user_id: str) -> Dict[str, List[str]]:
        """memory_id → linked person-entity names, strongest mention first."""
        result = await self.db.execute(
            select(EntityLink.memory_id, Entity.name, Entity.mention_count)
            .join(Entity, Entity.id == EntityLink.entity_id)
            .where(
                and_(
                    Entity.user_id == user_id,
                    func.lower(Entity.entity_type).in_(_PERSON_ENTITY_TYPES),
                )
            )
        )
        by_memory: Dict[str, List[Tuple[str, int]]] = {}
        for memory_id, name, mentions in result.all():
            by_memory.setdefault(memory_id, []).append((name, mentions or 0))
        return {
            mid: [name for name, _ in sorted(pairs, key=lambda p: (-p[1], p[0]))]
            for mid, pairs in by_memory.items()
        }

    async def organize_user(self, user_id: str) -> Dict:
        """Phase A of the migration: deterministic, idempotent, LLM-free.

        Ensures the system files exist and assigns every active memory row a
        file_slug. People rows linked to exactly one person entity get that
        person's file; everything else lands in its section's default. Safe
        to run repeatedly — rows that already carry a slug are untouched.
        """
        for slug in SYSTEM_FILES:
            await self.ensure_file(user_id, slug)

        rows = await self._active_rows_light(user_id)
        pending = [m for m in rows if not m.file_slug]
        people = await self._person_names_by_memory(user_id) if pending else {}

        next_pos: Dict[str, int] = {}
        for m in rows:
            if m.file_slug and m.file_position is not None:
                next_pos[m.file_slug] = max(next_pos.get(m.file_slug, 0), m.file_position + 1)

        assigned: Dict[str, int] = {}
        # Stable order: oldest first, so positions read as the file grew.
        pending.sort(key=lambda m: (m.created_at or datetime.min, m.id))
        for m in pending:
            names = people.get(m.id, [])
            slug = None
            # Exactly one linked person — a row naming two people belongs to
            # neither one's file (same rule as route_slug).
            if (m.brain_type or "user") == "user" and (m.category or "").lower() in _PERSON_CATEGORIES and len(names) == 1:
                slug = person_slug(names[0])
                if slug:
                    await self.ensure_file(user_id, slug, title=names[0].strip())
            if not slug:
                slug = default_slug_for(m.category, m.brain_type)
            m.file_slug = slug
            m.file_position = next_pos.get(slug, 0)
            next_pos[slug] = m.file_position + 1
            assigned[slug] = assigned.get(slug, 0) + 1

        await self.db.commit()
        report = {
            "user_id": user_id,
            "assigned": sum(assigned.values()),
            "already_organized": len(rows) - len(pending),
            "by_file": assigned,
        }
        if assigned:
            logger.info("[memory_files] organized user=%s %s", user_id, report)
        return report

    async def delete_file(
        self, user_id: str, slug: str, *, virtual_only: bool = False
    ) -> Optional[int]:
        """Archive a file: soft-delete its entries (with DELETED events, same
        semantics as deleting one memory) and drop the file row unless it is
        a system file — those stay and read as empty. Returns the number of
        archived entries, or None when the slug resolves to nothing."""
        resolved = await self.get_file(user_id, slug, virtual_only=virtual_only)
        if resolved is None:
            return None
        _, entries = resolved

        now = datetime.utcnow()
        for m in entries:
            self.db.add(MemoryEvent(
                memory_id=m.id,
                user_id=user_id,
                event_type=MemoryEventType.DELETED.value,
                event_data_json=json.dumps({
                    "category": m.category,
                    "memory_type": m.memory_type,
                    "content_preview": (m.canonical_content or m.content or "")[:100],
                    "strength_at_deletion": m.strength,
                    "via": f"file:{slug}",
                }),
                trigger_source="api",
            ))
            m.is_deleted = True
            m.deleted_at = now

        if not virtual_only and slug not in SYSTEM_FILES:
            result = await self.db.execute(
                select(MemoryFile).where(
                    and_(MemoryFile.user_id == user_id, MemoryFile.slug == slug)
                )
            )
            row = result.scalar_one_or_none()
            if row is not None:
                await self.db.delete(row)

        await self.db.commit()
        return len(entries)
