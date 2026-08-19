"""Curation ops engine (docs/memory/rebuild-2026-08.md §3.6): the strict
contract between the LLM's proposals and what actually touches rows.

Pins: validation rejects unknown ids, double-touch, third-person voice and
over-deletion; application rides the existing evolution primitives (merge →
supersede with lineage + consolidated events, delete → soft, add → filed);
nothing hard-deletes.
"""

import json
import uuid
from datetime import datetime, timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.models.base import Base
from app.db.models.user import User
from app.db.models.memory import BrainStats, Memory, MemoryEvent, MemoryFile, memory_relationships
from app.db.models.entity import Entity, EntityLink
from app.services.memory_file_ops import (
    apply_ops,
    build_curation_prompt,
    render_entries_for_prompt,
    validate_ops,
)
from app.services.memory_file_service import MemoryFileService


class _StubEmbedding:
    """Offline embedding: behaves like the degraded (no-provider) path."""

    async def embed_async(self, text, api_key=None):
        return None

    def embed(self, text, api_key=None):
        return None


@pytest.fixture(autouse=True)
def _offline_embeddings(monkeypatch):
    monkeypatch.setattr(
        "app.services.memory_service.get_embedding_service", lambda: _StubEmbedding()
    )
    monkeypatch.setattr(
        "app.services.memory_dedup_service.get_embedding_service", lambda: _StubEmbedding()
    )


async def _make_session():
    engine = create_async_engine(
        "sqlite+aiosqlite://", connect_args={"check_same_thread": False}
    )
    async with engine.begin() as conn:
        await conn.run_sync(
            Base.metadata.create_all,
            tables=[
                User.__table__, Memory.__table__, MemoryFile.__table__,
                MemoryEvent.__table__, BrainStats.__table__,
                Entity.__table__, EntityLink.__table__, memory_relationships,
            ],
        )
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    db = maker()
    user_id = str(uuid.uuid4())
    db.add(User(id=user_id, email=f"{user_id[:8]}@test.local", hashed_password="x"))
    await db.commit()
    return db, user_id


def _mem(user_id, content, *, category="active_task", slug="working", pos=0, **kw):
    return Memory(
        user_id=user_id, content=content, category=category,
        memory_type=kw.pop("memory_type", "task"), brain_type="user",
        file_slug=slug, file_position=pos, **kw,
    )


# ── Validation ────────────────────────────────────────────────────────

async def test_validate_rejects_bad_ops_and_keeps_good_ones():
    entries = [_mem("u", f"Entry {i}") for i in range(4)]
    ops, problems = validate_ops(
        [
            {"op": "merge", "ids": ["e1", "e2"], "text": "You get a daily Gmail briefing at 11:49 AM."},
            {"op": "update", "id": "e9", "text": "Unknown id."},          # unknown
            {"op": "update", "id": "e1", "text": "Already merged above."},  # double touch
            {"op": "update", "id": "e3", "text": "The user prefers tea."},  # third person
            {"op": "update", "id": "e3", "text": ""},                       # empty
            {"op": "merge", "ids": ["e4"], "text": "One id is not a merge."},
            {"op": "set_related", "slugs": ["people/majid", "../etc", "areas/ielts"]},
            {"op": "nonsense"},
        ],
        entries, instruction_mode=False,
    )
    assert [o["op"] for o in ops] == ["merge", "set_related"]
    assert ops[1]["slugs"] == ["people/majid", "areas/ielts"]  # traversal dropped
    assert len(problems) == 6  # incl. the unknown-op complaint
    # The whole batch is never rejected for one bad op.
    assert any("third-person" in p for p in problems)


async def test_validate_delete_cap_differs_by_mode():
    entries = [_mem("u", f"Entry {i}") for i in range(10)]
    deletes = [{"op": "delete", "id": f"e{i + 1}", "reason": "noise"} for i in range(4)]
    # Consolidation may delete at most 1/5 — four deletes on ten entries all drop.
    ops, problems = validate_ops(deletes, entries, instruction_mode=False)
    assert ops == [] and any("delete cap" in p for p in problems)
    # An explicit user instruction gets max(3, half) — four of ten is fine.
    ops, problems = validate_ops(deletes, entries, instruction_mode=True)
    assert len(ops) == 4 and not problems


async def test_validate_ops_shape_guards():
    assert validate_ops("not-a-list", [], instruction_mode=False) == ([], ["ops must be a list"])
    ops, problems = validate_ops(
        [{"op": "update", "id": "e1", "text": "x" * 700}],
        [_mem("u", "hi")], instruction_mode=False,
    )
    assert ops == [] and any("over" in p for p in problems)


# ── Prompt shape ──────────────────────────────────────────────────────

async def test_prompt_renders_entries_and_voice_contract():
    m1 = _mem("u", "You get a motivational quote at 5:06 PM.",
              tags_json=json.dumps(["standing"]))
    m1.created_at = datetime(2026, 5, 18)
    m2 = _mem("u", "Daily Gmail briefing at 11:49.", pos=1)
    m2.created_at = datetime(2026, 8, 10)
    m2.expires_at = datetime(2026, 8, 25)
    block = render_entries_for_prompt([m1, m2])
    assert block.splitlines()[0].startswith("[e1] (saved 2026-05-18, standing arrangement)")
    assert "fades 2026-08-25" in block.splitlines()[1]

    prompt = build_curation_prompt("Working on", None, block, instruction=None)
    assert "SECOND PERSON" in prompt
    assert 'NEVER "The user"' in prompt
    assert '{"ops": []}' in prompt

    asked = build_curation_prompt("Working on", "p", block, instruction="merge the briefing entries")
    assert "merge the briefing entries" in asked


# ── Application ───────────────────────────────────────────────────────

async def test_apply_merge_keeps_survivor_and_archives_sources_softly():
    db, user_id = await _make_session()
    service = MemoryFileService(db)
    await service.ensure_file(user_id, "working")
    a = _mem(user_id, "Send a summary of my Gmail inbox every day at 11:49 AM.", pos=0)
    b = _mem(user_id, "You want a daily briefing of your Gmail at 11:49.", pos=1)
    c = _mem(user_id, "Daily Gmail briefing at 11:49 AM Toronto time.", pos=2)
    db.add_all([a, b, c])
    await db.commit()

    result = await apply_ops(
        db, user_id, "working",
        [{"op": "merge", "ids": ["e1", "e2", "e3"],
          "text": "You get a daily Gmail briefing at 11:49 AM (America/Toronto)."}],
        [a, b, c],
    )
    assert result["applied"] == [{"op": "merge", "survivor": a.id, "folded": 2}]

    await db.refresh(a); await db.refresh(b); await db.refresh(c)
    assert a.canonical_content == "You get a daily Gmail briefing at 11:49 AM (America/Toronto)."
    assert a.is_active and not a.is_deleted
    assert json.loads(a.merged_from_json) == [b.id, c.id]
    assert (b.is_active, c.is_active) == (False, False)
    assert b.superseded_by == a.id and c.superseded_by == a.id
    assert not b.is_deleted and not c.is_deleted  # soft, reversible
    # Survivor keeps its place in the file.
    assert (a.file_slug, a.file_position) == ("working", 0)

    events = (await db.execute(select(MemoryEvent).where(MemoryEvent.user_id == user_id))).scalars().all()
    kinds = sorted(e.event_type for e in events)
    assert kinds.count("consolidated") == 2  # one per folded source
    assert "updated" in kinds  # the survivor's rewrite


async def test_apply_update_delete_add_and_file_metadata():
    db, user_id = await _make_session()
    service = MemoryFileService(db)
    file_row = await service.ensure_file(user_id, "preferences", commit=True)
    a = _mem(user_id, "The user prefers dark mode.", category="preferences",
             slug="preferences", pos=0, memory_type="preference")
    b = _mem(user_id, "You could not find the track 'qolb qolnb'.",
             category="preferences", slug="preferences", pos=1)
    db.add_all([a, b])
    await db.commit()

    result = await apply_ops(
        db, user_id, "preferences",
        [
            {"op": "update", "id": "e1", "text": "You prefer dark mode."},
            {"op": "delete", "id": "e2", "reason": "one-off playback outcome"},
            {"op": "add", "text": "You like your coffee black.", "category": "preferences"},
            {"op": "set_purpose", "text": "What you like; read before choosing for you."},
            {"op": "set_related", "slugs": ["profile"]},
        ],
        [a, b],
    )
    ops_applied = [o["op"] for o in result["applied"]]
    assert ops_applied == ["update", "delete", "add", "set_purpose", "set_related"]

    await db.refresh(a); await db.refresh(b); await db.refresh(file_row)
    assert a.canonical_content == "You prefer dark mode."
    assert b.is_deleted and b.deleted_at is not None and b.superseded_by is None
    assert file_row.purpose == "What you like; read before choosing for you."
    assert json.loads(file_row.related_json) == ["profile"]

    added_id = result["applied"][2]["id"]
    added = (await db.execute(select(Memory).where(Memory.id == added_id))).scalar_one()
    assert added.file_slug == "preferences"
    assert added.file_position == 2  # appended after existing positions
    assert added.category == "preferences"

    # A no-op update (same text) applies nothing.
    result2 = await apply_ops(
        db, user_id, "preferences",
        [{"op": "update", "id": "e1", "text": "You prefer dark mode."}],
        [a],
    )
    assert result2["applied"] == []
