"""Harness self-check: proves the suite is running against the real stack.

If any of these fail, every other assertion in this package is meaningless.
"""
import os
import pytest
from sqlalchemy import text

from .pipeline import drive_turn, active_contents, recall


async def test_smoke_running_on_postgres_with_pgvector(db):
    ver = (await db.execute(text("SELECT version()"))).scalar()
    assert "PostgreSQL" in ver, ver
    ext = (await db.execute(text("SELECT extversion FROM pg_extension WHERE extname='vector'"))).scalar()
    assert ext, "pgvector extension is not installed"

    from app.config import settings
    assert settings.run_mode == "agent", settings.run_mode

    # The table CI cannot create.
    cols = (await db.execute(text(
        "SELECT data_type FROM information_schema.columns "
        "WHERE table_name='memories' AND column_name='embedding'"
    ))).scalar()
    assert cols == "USER-DEFINED", f"embedding column is {cols}, not a pgvector column"


async def test_smoke_real_extraction_writes_a_real_memory(db, user_a):
    n = await drive_turn(
        db, user_a,
        "My name is Nariman and I am allergic to peanuts.",
        "Noted — I'll keep that in mind.",
    )
    contents = await active_contents(db, user_a)
    assert n > 0, "extraction produced nothing"
    assert any("peanut" in c.lower() for c in contents), contents

    # And the read path finds it.
    hits = await recall(db, user_a, "what am I allergic to?")
    assert any("peanut" in h["content"].lower() for h in hits), hits
