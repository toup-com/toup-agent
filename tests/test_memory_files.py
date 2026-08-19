"""Memory files — the organizing layer (docs/memory/rebuild-2026-08.md §3).

Pins: the category→section map covers the whole taxonomy, slugs are safe,
virtual synthesis groups rows exactly like the organized view will, the
organize pass is idempotent and person-aware, file reads order by curation,
and file deletion archives (soft) with audit events — never hard-deletes.
"""

import uuid
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.models.base import Base
from app.db.models.user import User
from app.db.models.memory import Memory, MemoryEvent, MemoryFile
from app.db.models.entity import Entity, EntityLink
from app.memory_files import (
    SYSTEM_FILES,
    SECTION_ORDER,
    FileSection,
    default_slug_for,
    is_valid_slug,
    person_slug,
    render_file_markdown,
    section_for,
    slugify,
)
from app.memory_taxonomy import MEMORY_CATEGORY_ALIASES, MemoryCategory
from app.services.memory_file_service import MemoryFileService


async def _make_session() -> tuple[AsyncSession, str]:
    engine = create_async_engine(
        "sqlite+aiosqlite://", connect_args={"check_same_thread": False}
    )
    async with engine.begin() as conn:
        await conn.run_sync(
            Base.metadata.create_all,
            tables=[
                User.__table__, Memory.__table__, MemoryFile.__table__,
                MemoryEvent.__table__, Entity.__table__, EntityLink.__table__,
            ],
        )
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    db = maker()
    user_id = str(uuid.uuid4())
    db.add(User(id=user_id, email=f"{user_id[:8]}@test.local", hashed_password="x"))
    await db.commit()
    return db, user_id


def _memory(user_id: str, content: str, *, category: str, brain_type: str = "user",
            created_days_ago: int = 0, **kw) -> Memory:
    when = datetime.utcnow() - timedelta(days=created_days_ago)
    return Memory(
        user_id=user_id, content=content, category=category,
        memory_type=kw.pop("memory_type", "fact"), brain_type=brain_type,
        created_at=when, updated_at=when, **kw,
    )


# ── Canon: map + slugs ────────────────────────────────────────────────

async def test_section_map_covers_taxonomy():
    # Every canonical user category resolves to a section — none may fall
    # through to a KeyError or to a section outside the seven.
    for cat in MemoryCategory:
        section = section_for(cat.value, "user")
        assert section in SECTION_ORDER, f"{cat.value} → {section}"
    # Every legacy alias resolves through normalize_category to the same
    # section its canonical value gets.
    for alias, canonical in MEMORY_CATEGORY_ALIASES.items():
        assert section_for(alias, "user") == section_for(canonical.value, "user"), alias
    # Brain precedence: agent brain is always Learned (even for the
    # colliding 'preferences'), work brain splits on active_task.
    assert section_for("preferences", "agent") == FileSection.LEARNED
    assert section_for("anything", "agent") == FileSection.LEARNED
    assert section_for("active_task", "work") == FileSection.WORKING
    assert section_for("process", "work") == FileSection.LEARNED
    # Unknown user categories land in Knowledge — files have no junk drawer.
    assert section_for("completely-novel", "user") == FileSection.KNOWLEDGE
    assert section_for(None, "user") == FileSection.KNOWLEDGE
    # Every section has a default slug and every default slug is a system file.
    for section in SECTION_ORDER:
        slug = default_slug_for(
            "active_task" if section == FileSection.WORKING else None, "user"
        )
        assert is_valid_slug(slug)
    assert default_slug_for("goals", "user") == "areas/work"
    assert default_slug_for("corrections", "agent") == "learned"


async def test_slugs_are_safe_and_unicode_capable():
    assert slugify("Majid Tajik") == "majid-tajik"
    assert slugify("  IELTS — Prep!  ") == "ielts-prep"
    assert slugify("محمد السالم") == "محمد-السالم"  # Persian survives
    assert person_slug("Majid Tajik") == "people/majid-tajik"
    assert person_slug("!!!") is None
    assert is_valid_slug("people/majid-tajik")
    assert is_valid_slug("profile")
    assert not is_valid_slug("../../etc/passwd")
    assert not is_valid_slug("a/b/c")
    assert not is_valid_slug("has_underscore")
    assert not is_valid_slug("")
    # System slugs must themselves validate — they ride URLs.
    for slug in SYSTEM_FILES:
        assert is_valid_slug(slug), slug


async def test_render_markdown_shape():
    md = render_file_markdown(
        "IELTS", "Your IELTS prep; read when it comes up.",
        ["You are targeting band 7.5.", "", "Your tutor is Majid."],
        related=["people/majid-tajik"],
    )
    lines = md.splitlines()
    assert lines[0] == "# IELTS"
    assert lines[1].startswith("*") and lines[1].endswith("*")
    assert "- You are targeting band 7.5." in lines
    assert "- Your tutor is Majid." in lines
    assert lines[-1] == "See also: people/majid-tajik"
    assert "- \n" not in md  # empty entries dropped


# ── Virtual synthesis ─────────────────────────────────────────────────

async def test_virtual_listing_groups_rows():
    db, user_id = await _make_session()
    db.add_all([
        _memory(user_id, "Your name is Nariman.", category="identity"),
        _memory(user_id, "You prefer dark mode.", category="preferences"),
        _memory(user_id, "You are preparing for IELTS.", category="goals"),
        _memory(user_id, "Maya is your daughter.", category="people"),
        _memory(user_id, "Prefers short answers.", category="corrections", brain_type="agent"),
        _memory(user_id, "Daily briefing at 11:49.", category="active_task", memory_type="task"),
        _memory(user_id, "An inactive row must not count.", category="identity", is_active=False),
    ])
    await db.commit()

    service = MemoryFileService(db)
    listing = await service.list_files(user_id, virtual_only=True)
    assert listing["organized"] is False
    by_slug = {
        f["slug"]: f for s in listing["sections"] for f in s["files"]
    }
    # Sections arrive in canonical order.
    assert [s["section"] for s in listing["sections"]] == [s.value for s in SECTION_ORDER]
    # Every system file is present even when empty.
    for slug in SYSTEM_FILES:
        assert slug in by_slug, slug
    assert by_slug["profile"]["entry_count"] == 1  # inactive row skipped
    assert by_slug["preferences"]["entry_count"] == 1
    assert by_slug["areas/work"]["entry_count"] == 1
    assert by_slug["people"]["entry_count"] == 1
    assert by_slug["learned"]["entry_count"] == 1
    assert by_slug["working"]["entry_count"] == 1
    assert by_slug["knowledge"]["entry_count"] == 0
    assert by_slug["knowledge"]["updated_at"] is None
    assert by_slug["profile"]["updated_at"] is not None


# ── Organize (Phase A) ────────────────────────────────────────────────

async def test_organize_assigns_slugs_and_person_files_idempotently():
    db, user_id = await _make_session()
    m_person = _memory(user_id, "Majid Tajik is your IELTS tutor.", category="people")
    m_group = _memory(user_id, "Your cousins visit in the summer.", category="people")
    m_goal = _memory(user_id, "You are applying to UofT MScAC.", category="goals", created_days_ago=3)
    db.add_all([m_person, m_group, m_goal])
    entity = Entity(user_id=user_id, name="Majid Tajik", entity_type="person", mention_count=4)
    db.add(entity)
    await db.flush()
    db.add(EntityLink(memory_id=m_person.id, entity_id=entity.id))
    # A two-person row must NOT be attributed to either person's file.
    entity2 = Entity(user_id=user_id, name="Sara", entity_type="person", mention_count=1)
    db.add(entity2)
    await db.flush()
    db.add(EntityLink(memory_id=m_group.id, entity_id=entity.id))
    db.add(EntityLink(memory_id=m_group.id, entity_id=entity2.id))
    await db.commit()

    service = MemoryFileService(db)
    report = await service.organize_user(user_id)
    assert report["assigned"] == 3
    await db.refresh(m_person); await db.refresh(m_goal); await db.refresh(m_group)
    assert m_person.file_slug == "people/majid-tajik"
    assert m_group.file_slug == "people"  # ambiguous → catch-all
    assert m_goal.file_slug == "areas/work"
    assert m_goal.file_position is not None

    # The person file row exists, carries the person's name, is not system.
    row = (await db.execute(
        select(MemoryFile).where(MemoryFile.user_id == user_id, MemoryFile.slug == "people/majid-tajik")
    )).scalar_one()
    assert row.title == "Majid Tajik"
    assert row.is_system is False
    assert row.purpose  # fallback purpose until consolidation writes one

    # Idempotent: a second run assigns nothing and creates nothing new.
    report2 = await service.organize_user(user_id)
    assert report2["assigned"] == 0
    files = (await db.execute(select(MemoryFile).where(MemoryFile.user_id == user_id))).scalars().all()
    assert len(files) == len(SYSTEM_FILES) + 1

    listing = await service.list_files(user_id)
    assert listing["organized"] is True
    by_slug = {f["slug"]: f for s in listing["sections"] for f in s["files"]}
    assert by_slug["people/majid-tajik"]["entry_count"] == 1
    assert by_slug["people/majid-tajik"]["section"] == "people"


# ── File reads ────────────────────────────────────────────────────────

async def test_get_file_orders_by_curation_and_resolves_system_empties():
    db, user_id = await _make_session()
    a = _memory(user_id, "Older but curated later.", category="identity", created_days_ago=9)
    b = _memory(user_id, "Newer but curated first.", category="identity", created_days_ago=1)
    a.file_slug = "profile"; a.file_position = 2
    b.file_slug = "profile"; b.file_position = 1
    c = _memory(user_id, "No position yet.", category="identity", created_days_ago=5)
    c.file_slug = "profile"
    db.add_all([a, b, c])
    await db.commit()

    service = MemoryFileService(db)
    resolved = await service.get_file(user_id, "profile")
    assert resolved is not None
    info, entries = resolved
    assert info["title"] == "Profile"
    assert [m.id for m in entries] == [b.id, a.id, c.id]  # position asc, unpositioned last
    assert info["entry_count"] == 3

    # A system slug with no entries still resolves (empty file)…
    empty = await service.get_file(user_id, "knowledge")
    assert empty is not None and empty[0]["entry_count"] == 0
    # …an unknown slug with no entries does not.
    assert await service.get_file(user_id, "areas/nonexistent") is None
    # …and traversal shapes never resolve.
    assert await service.get_file(user_id, "../etc") is None


# ── Deletion ──────────────────────────────────────────────────────────

async def test_delete_file_archives_softly_with_events():
    db, user_id = await _make_session()
    service = MemoryFileService(db)
    m1 = _memory(user_id, "Majid teaches on Tuesdays.", category="people")
    m2 = _memory(user_id, "Majid prefers WhatsApp.", category="people")
    db.add_all([m1, m2])
    await db.flush()
    await service.ensure_file(user_id, "people/majid", title="Majid")
    m1.file_slug = "people/majid"; m1.file_position = 0
    m2.file_slug = "people/majid"; m2.file_position = 1
    await db.commit()

    archived = await service.delete_file(user_id, "people/majid")
    assert archived == 2
    await db.refresh(m1); await db.refresh(m2)
    assert m1.is_deleted and m2.is_deleted
    assert m1.deleted_at is not None
    # Audit events written, one per entry.
    events = (await db.execute(
        select(MemoryEvent).where(MemoryEvent.user_id == user_id)
    )).scalars().all()
    assert len(events) == 2 and all(e.event_type == "deleted" for e in events)
    # The person file row is gone; a system file would have survived.
    gone = (await db.execute(
        select(MemoryFile).where(MemoryFile.user_id == user_id, MemoryFile.slug == "people/majid")
    )).scalar_one_or_none()
    assert gone is None

    # Deleting a SYSTEM file keeps the row, archives entries.
    m3 = _memory(user_id, "You prefer tea.", category="preferences")
    m3.file_slug = "preferences"
    db.add(m3)
    await service.ensure_file(user_id, "preferences")
    await db.commit()
    assert await service.delete_file(user_id, "preferences") == 1
    kept = (await db.execute(
        select(MemoryFile).where(MemoryFile.user_id == user_id, MemoryFile.slug == "preferences")
    )).scalar_one_or_none()
    assert kept is not None
    # Unknown slug → None (route turns this into a 404).
    assert await service.delete_file(user_id, "areas/never-existed") is None
