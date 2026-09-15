"""Migration 007 must not ALTER a table no migration ever creates.

`007_pgvector_embedding` step 7 ran

    ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS embedding vector(1536)

and `document_chunks` appears in exactly one file in `alembic/versions/` —
that one — with no `create_table` anywhere. The table exists only because
`init_db`'s `Base.metadata.create_all` makes it. So `alembic upgrade head`
against an empty database was GUARANTEED to fail, alembic rolled the 001-006
span back in its single transaction, `alembic_version` was never stamped, and
every later boot re-ran 001 against a schema `init_db` had meanwhile populated
→ `DuplicateTable: relation "users" already exists`. Fleet-wide, every boot,
since 2026-02-13.

These tests are static on purpose — they assert the property that made the
statement unrunnable (it names a table nothing in the chain creates, and is
not guarded), rather than re-running Postgres in CI. The live verification is
recorded in the commit message and in `alembic/env.py`'s design note.

They do NOT claim the chain now runs: with 007 guarded it reaches 021 and
dies on `day_chats`. See `alembic/env.py`.
"""
from __future__ import annotations

import pathlib
import re

VERSIONS = pathlib.Path(__file__).resolve().parents[1] / "alembic" / "versions"
M007 = next(VERSIONS.glob("*_007_pgvector_embedding_column.py"))


def _upgrade_body(path: pathlib.Path) -> str:
    src = path.read_text()
    start = src.index("def upgrade()")
    end = src.find("def downgrade()", start)
    return src[start: end if end > 0 else len(src)]


def test_document_chunks_is_created_by_no_migration_in_the_chain():
    """The premise. If some migration ever starts creating the table, this
    test should fail so the guard can be reconsidered rather than cargo-culted.
    """
    creators = [
        p.name
        for p in VERSIONS.glob("*.py")
        if re.search(r"create_table\(\s*['\"]document_chunks['\"]", p.read_text())
        or re.search(r"CREATE TABLE (IF NOT EXISTS )?document_chunks", p.read_text())
    ]
    assert creators == [], f"document_chunks is now created by {creators}"


def test_every_document_chunks_statement_in_the_upgrade_is_guarded():
    """UPGRADE only, and that is deliberate.

    `downgrade()` names the same table and would fail the same way, but it is
    left byte-for-byte as it was: the repo's `forbid-destructive-migrations`
    check greps ADDED lines in `alembic/versions/*.py` for destructive
    patterns, so re-indenting that pre-existing block into a guard turns an
    untouched line into an added one and trips the lint. It is also the branch
    that has never run — the chain has never reached 007 on any tenant in
    production, let alone reversed it. Upgrade is the path the fleet boots.
    """
    body = _upgrade_body(M007)
    assert "document_chunks" in body, "step 7 vanished — was it deleted rather than guarded?"
    assert "to_regclass('document_chunks')" in body, "the guard is gone"

    # No bare statement may name the table outside a guarded block.
    for stmt in re.findall(r"(ALTER TABLE document_chunks[^\n;]*|UPDATE\s+document_chunks)", body):
        at = body.index(stmt)
        block = body[max(0, at - 600): at]
        assert "to_regclass('document_chunks')" in block, (
            f"unguarded statement on a table no migration creates: {stmt.strip()[:80]}"
        )


def test_the_downgrade_is_untouched_so_the_destructive_lint_stays_clean():
    """Locks the trade-off above so nobody 'completes' the guard by hand and
    silently fails CI on a pattern that has nothing to do with this fix."""
    body = M007.read_text()
    down = body[body.index("def downgrade()"):]
    assert "to_regclass" not in down, (
        "the downgrade was guarded — that re-writes a pre-existing destructive "
        "line and trips forbid-destructive-migrations; see the test above"
    )


def test_the_guard_is_a_plpgsql_block_not_a_bare_conditional():
    """A bare `ALTER … WHERE to_regclass(…) IS NOT NULL` is not valid DDL, and
    a guard outside PL/pgSQL still PARSES the table reference. Only a DO block
    defers planning to the branch being taken."""
    body = _upgrade_body(M007)
    at = body.index("to_regclass('document_chunks')")
    block = body[max(0, at - 400): at]
    assert "DO $$" in block and "BEGIN" in block, (
        "the document_chunks guard is not inside a PL/pgSQL DO block"
    )


def test_007_still_does_its_actual_job_on_the_tables_001_creates():
    """The guard must not have turned the migration into a no-op. `memories`,
    `entities` and `messages` ARE created by 001_initial, so their statements
    stay unguarded and must still be there."""
    body = _upgrade_body(M007)
    assert "CREATE EXTENSION IF NOT EXISTS vector" in body
    for table in ("memories", "entities", "messages"):
        assert f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS embedding vector(1536)" in body, (
            f"007 no longer adds the vector column to {table}"
        )
    assert "ix_memories_embedding_hnsw" in body


def test_the_tables_007_touches_unguarded_are_all_created_by_001():
    """The completeness check: if a future edit adds an unguarded statement
    against another `create_all`-only table, this is what catches it."""
    initial = next(VERSIONS.glob("*_001_*.py")).read_text()
    created = set(re.findall(r"create_table\(\s*['\"]([a-z_]+)['\"]", initial))
    assert "memories" in created and "document_chunks" not in created, (
        "001_initial's create_table list could not be parsed"
    )

    body = _upgrade_body(M007)
    # Strip the guarded DO blocks; whatever names a table after that is bare.
    bare = re.sub(r"DO \$\$.*?END \$\$;", "", body, flags=re.DOTALL)
    touched = set(re.findall(r"ALTER TABLE ([a-z_]+)", bare)) | set(
        re.findall(r"UPDATE\s+([a-z_]+)\s*\n?\s*SET", bare)
    )
    missing = sorted(touched - created)
    assert not missing, (
        f"007 ALTERs/UPDATEs {missing} unguarded, but 001_initial does not create "
        f"them — the same defect as document_chunks"
    )
