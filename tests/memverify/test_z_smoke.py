"""Harness self-check: proves the suite is running against the real stack.

If any of these fail, every other assertion in this package is meaningless.
"""
from sqlalchemy import text

from .pipeline import bodies_by_slug, drive_turn


async def test_smoke_running_on_postgres_with_the_v3_tables(db):
    ver = (await db.execute(text("SELECT version()"))).scalar()
    assert "PostgreSQL" in ver, ver

    from app.config import settings
    assert settings.run_mode == "agent", settings.run_mode

    # `memory_files` is AGENT_ONLY and self-healed by init_db, never by
    # alembic — a platform-mode database does not have it at all.
    for table, column in (
        ("memory_files", "body_md"),
        ("memory_file_changes", "day_key"),
    ):
        got = (await db.execute(text(
            "SELECT 1 FROM information_schema.columns "
            f"WHERE table_name='{table}' AND column_name='{column}'"
        ))).scalar()
        assert got, f"{table}.{column} is missing — the v3 schema is not here"

    # pgvector still matters: the document/media leg of memory_search is the
    # one surviving reader of `memories` (v3 §3.4).
    ext = (await db.execute(
        text("SELECT extversion FROM pg_extension WHERE extname='vector'")
    )).scalar()
    assert ext, "pgvector extension is not installed"


async def test_smoke_the_real_writer_writes_a_real_file(db, user_a):
    """A real key, a real model call, a real file body. No mocks anywhere:
    a mocked writer cannot measure writing quality, which is the whole point
    of categories A and B."""
    result = await drive_turn(
        db, user_a,
        "I'm allergic to peanuts — it's a serious allergy, not a preference.",
        "Noted, I'll keep that in mind for anything food-related.",
    )
    assert result.get("skipped") is None, result
    bodies = await bodies_by_slug(db, user_a)
    joined = " ".join(bodies.values()).lower()
    assert "peanut" in joined, bodies
