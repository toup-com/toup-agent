"""
W0.1b — tenant DB schema-drift self-heal (init_db).

PROVEN in prod (tenant 871bac24): an old tenant DB was missing the
memory_events + memory_relationships tables AND several memories columns
(live errors: UndefinedColumnError on memories.source_*; ProgrammingError
on memory_events) while fresh tenant DBs have them. Agents boot via
init_db() create_all, NOT alembic, so init_db must self-heal:

  1. an ancient memories/entities table gains every missing model column
     via the explicit guarded ALTER mirrors — WITH model defaults, so
     filter-critical columns (is_deleted, is_active, …) read FALSE/TRUE
     on legacy rows instead of NULL (`WHERE is_deleted = FALSE` must
     keep matching them);
  2. missing tables (memory_events, memory_relationships) are re-created
     by create_all / the missing-table backstop — even when the bulk
     create_all pass fails (the no-pgvector fallback + backstop create
     each table in its OWN transaction so one failure can't poison the
     rest);
  3. fresh sqlite runs now get memories + entities at all: a bare
     TSVECTOR on entities.name_search used to abort the whole bulk
     create_all on sqlite, and the fallback then skipped both tables.

Run: python3 -m pytest tests/test_tenant_schema_drift_selfheal.py -q
"""

import os

import pytest
from sqlalchemy import text, inspect as sa_inspect


async def _table_names(engine) -> set[str]:
    async with engine.connect() as conn:
        return set(await conn.run_sync(lambda sc: sa_inspect(sc).get_table_names()))


async def _column_names(engine, table: str) -> set[str]:
    async with engine.connect() as conn:
        return set(await conn.run_sync(
            lambda sc, t=table: {c["name"] for c in sa_inspect(sc).get_columns(t)}
        ))


# A memories table as an ancient agent image would have created it —
# core columns only, none of the User Brain upgrade fields and no
# source tracking (the proven 871bac24 vintage).
LEGACY_MEMORIES_DDL = (
    "CREATE TABLE memories ("
    "id VARCHAR(36) PRIMARY KEY, "
    "user_id VARCHAR(36), "
    "content TEXT, "
    "category VARCHAR(20), "
    "memory_type VARCHAR(20), "
    "importance FLOAT, "
    "created_at TIMESTAMP)"
)


@pytest.mark.asyncio
async def test_fresh_db_has_memory_tables():
    """Regression for the sqlite TSVECTOR compile abort: after a plain
    init_db (conftest runs it), memories/entities and every memory-adjacent
    table must exist — the bulk create_all must not die on entities."""
    from app.db.database import engine

    present = await _table_names(engine)
    for t in ("memories", "entities", "memory_events", "memory_relationships",
              "brain_stats", "retrieval_events", "entity_links"):
        assert t in present, f"fresh init_db left {t} missing"


@pytest.mark.asyncio
async def test_drifted_memories_table_gains_missing_columns():
    """An ancient memories table self-heals to the full model column set,
    and legacy rows read model defaults on the filter-critical columns."""
    from app.db.database import engine, init_db

    async with engine.begin() as conn:
        # Drop dependents first so memories can be dropped.
        #
        # `document_chunks` and `media` were added to this list on 2026-08-06:
        # both carry a FK to memories (document_chunks_memory_id_fkey,
        # media_memory_id_fkey), so once a run creates the full table set the
        # DROP below fails with
        #   DependentObjectsStillExistError: cannot drop table memories
        #   because other objects depend on it
        # and the whole self-heal assertion never gets to run. init_db()
        # recreates them, which is the same contract the other three rely on.
        for dep in ("memory_events", "memory_relationships", "entity_links",
                    "document_chunks", "media"):
            await conn.execute(text(f"DROP TABLE IF EXISTS {dep}"))
        await conn.execute(text("DROP TABLE IF EXISTS memories"))
        await conn.execute(text(LEGACY_MEMORIES_DDL))
        await conn.execute(text(
            "INSERT INTO memories (id, user_id, content, category, memory_type) "
            "VALUES ('m-legacy', 'u1', 'legacy row', 'fact', 'fact')"
        ))

    await init_db()

    cols = await _column_names(engine, "memories")
    expected = {
        "brain_type", "summary", "embedding_json", "search_vector",
        "confidence", "strength", "memory_level", "emotional_salience",
        "last_reinforced_at", "consolidation_count", "decay_rate",
        # The decay clock (alembic 080). Tenant DBs have no alembic_version,
        # so init_db is the only thing that can put it there — and without it
        # every decay pass re-charges the whole elapsed interval.
        "last_decayed_at",
        "updated_at", "last_accessed_at", "access_count",
        "source_message_id", "source_type", "ref_kind", "ref_id",
        "metadata_json", "tags_json", "canonical_content", "history_json",
        "merged_from_json", "superseded_by", "is_active", "is_deleted",
        "deleted_at",
    }
    missing = expected - cols
    assert not missing, f"init_db did not heal memories columns: {sorted(missing)}"

    # Legacy rows must read model defaults, not NULL — otherwise every
    # `WHERE is_deleted = FALSE` / `is_active = TRUE` filter silently
    # hides the user's entire pre-drift Brain.
    async with engine.connect() as conn:
        row = (await conn.execute(text(
            "SELECT is_deleted, is_active, source_type, strength "
            "FROM memories WHERE id = 'm-legacy'"
        ))).one()
    assert row[0] in (False, 0), "legacy row is_deleted should default FALSE"
    assert row[1] in (True, 1), "legacy row is_active should default TRUE"
    assert row[2] == "conversation", "legacy row source_type should default 'conversation'"
    assert row[3] == 1.0, "legacy row strength should default 1.0"

    # The dropped dependent tables must have been re-created too.
    present = await _table_names(engine)
    for t in ("memory_events", "memory_relationships", "entity_links"):
        assert t in present, f"init_db did not re-create {t}"


@pytest.mark.asyncio
async def test_drifted_entities_table_gains_missing_columns():
    """Same heal for an ancient entities table."""
    from app.db.database import engine, init_db

    async with engine.begin() as conn:
        for dep in ("entity_links", "entity_relationships"):
            await conn.execute(text(f"DROP TABLE IF EXISTS {dep}"))
        await conn.execute(text("DROP TABLE IF EXISTS entities"))
        await conn.execute(text(
            "CREATE TABLE entities ("
            "id VARCHAR(36) PRIMARY KEY, "
            "user_id VARCHAR(36), "
            "name VARCHAR(255), "
            "entity_type VARCHAR(50))"
        ))

    await init_db()

    cols = await _column_names(engine, "entities")
    expected = {
        "description", "embedding_json", "schema_type", "attributes_json",
        "name_search", "mention_count", "first_seen_at", "last_seen_at",
        "created_at", "updated_at",
    }
    missing = expected - cols
    assert not missing, f"init_db did not heal entities columns: {sorted(missing)}"


@pytest.mark.asyncio
async def test_missing_memory_tables_recreated_by_create_all():
    """memory_events / memory_relationships absent → next boot creates them."""
    from app.db.database import engine, init_db

    async with engine.begin() as conn:
        await conn.execute(text("DROP TABLE IF EXISTS memory_events"))
        await conn.execute(text("DROP TABLE IF EXISTS memory_relationships"))

    present = await _table_names(engine)
    assert "memory_events" not in present and "memory_relationships" not in present

    await init_db()

    present = await _table_names(engine)
    assert "memory_events" in present, "init_db did not create memory_events"
    assert "memory_relationships" in present, "init_db did not create memory_relationships"


@pytest.mark.asyncio
@pytest.mark.skipif(
    os.environ.get("DATABASE_URL", "").startswith(("postgresql", "postgres")),
    reason="sqlite-only: with real pgvector, init_db re-raises the synthetic "
           "'vector' error instead of taking the fallback path this test pins",
)
async def test_missing_tables_healed_even_when_bulk_create_all_fails(monkeypatch):
    """The 871bac24 failure shape: the bulk create_all pass dies with a
    vector-ish error, and the per-table fallback / missing-table backstop
    must still create the absent non-vector tables — one bad table must
    never silently kill the rest of the schema."""
    from app.db import database as db_mod
    from app.db.database import engine, init_db

    async with engine.begin() as conn:
        await conn.execute(text("DROP TABLE IF EXISTS memory_events"))
        await conn.execute(text("DROP TABLE IF EXISTS memory_relationships"))

    def _failing_create_all(*args, **kwargs):
        raise Exception('type "vector" does not exist')

    monkeypatch.setattr(db_mod.Base.metadata, "create_all", _failing_create_all)
    await init_db()

    present = await _table_names(engine)
    assert "memory_events" in present, "fallback path did not create memory_events"
    assert "memory_relationships" in present, (
        "fallback path did not create memory_relationships"
    )
