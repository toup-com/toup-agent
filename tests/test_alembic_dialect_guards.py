"""Phase 4 P2 — regression guard: Postgres-only DDL/DML in alembic
migrations must live inside a dialect skip.

Why pin this: the previous regression mode was "operator runs
`alembic upgrade head` on a sqlite test DB to sanity-check the chain,
gets a `near 'USING' / 'to_char' / 'DO $$': syntax error`, gives up,
the local sanity gate stays useless." This test scans every migration
file for known Postgres-only string patterns and asserts each one is
within a function that ALSO contains a dialect guard (an `if … dialect
== "postgresql"` or `_is_postgres()` callsite).

The check is a lint, not a full proof — it doesn't ensure the guard is
on the RIGHT side of the construct, only that some guard exists in the
same function. Good enough for a regression gate; the human review
catches the placement.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest


_MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "alembic" / "versions"

# Strings that only compile on Postgres. Anything outside this set
# (string-quoted SQL that uses sqlite-portable syntax) doesn't need a
# dialect guard. Update this list when adding new patterns.
_PG_ONLY_PATTERNS = [
    r"\bto_char\s*\(",
    r"\bto_tsvector\s*\(",
    r"\bsetweight\s*\(",
    r"USING\s+hnsw\b",
    r"USING\s+GIN\b",
    r"::\s*vector\b",
    r"::\s*tsvector\b",
    r"::\s*jsonb\b",
    r"DO\s+\$\$",
    r"CREATE\s+OR\s+REPLACE\s+FUNCTION\b",
    r"CREATE\s+TRIGGER\b",
    r"GENERATED\s+ALWAYS\s+AS\b.*\bSTORED\b",
    r"E'\\\\n'",  # E-prefixed escape literals
]

_GUARD_HINTS = [
    "_is_postgres()",
    "is_postgres()",
    'dialect.name == "postgresql"',
    "dialect.name == 'postgresql'",
    'dialect == "postgresql"',
    "dialect == 'postgresql'",
    'dialect.name != "postgresql"',
    "dialect.name != 'postgresql'",
]


def _function_contains_guard(fn_source: str) -> bool:
    return any(hint in fn_source for hint in _GUARD_HINTS)


def _collect_unguarded_pg_only_calls(path: Path) -> list[tuple[str, str]]:
    """Return list of (function_name, matched_pattern) for any PG-only
    string that lives in a function with no dialect guard."""
    src = path.read_text()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        pytest.fail(f"Could not parse {path.name} as Python")

    issues: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in {"upgrade", "downgrade"}:
            continue
        # Get the function's source.
        start = node.lineno - 1
        end = node.end_lineno or len(src.splitlines())
        fn_source = "\n".join(src.splitlines()[start:end])
        for pattern in _PG_ONLY_PATTERNS:
            if re.search(pattern, fn_source):
                if not _function_contains_guard(fn_source):
                    issues.append((node.name, pattern))
    return issues


@pytest.mark.parametrize("migration_path", sorted(_MIGRATIONS_DIR.glob("*.py")), ids=lambda p: p.name)
def test_pg_only_constructs_have_dialect_guards(migration_path: Path):
    if migration_path.name == "__init__.py":
        pytest.skip("init file")
    issues = _collect_unguarded_pg_only_calls(migration_path)
    assert not issues, (
        f"{migration_path.name} has Postgres-only patterns "
        f"without a dialect guard: {issues}. Wrap with "
        f"`if op.get_bind().dialect.name == 'postgresql'` or a "
        f"`_is_postgres()` helper before the offending statement."
    )


def test_migrations_dir_actually_scanned():
    """Defensive — if the migrations dir moves, the parametrize above
    silently produces zero test cases and the suite looks green."""
    files = list(_MIGRATIONS_DIR.glob("*.py"))
    assert len(files) >= 30, f"Expected many migration files, found {len(files)}"


def test_alembic_upgrade_passes_pg_only_construct_migrations_on_sqlite(tmp_path):
    """`alembic upgrade head` must get PAST every migration that carries
    a Postgres-only construct (004 to_char, 007 hnsw/::vector, 008
    tsvector+GIN+trigger, 009 GENERATED tsvector) when run on sqlite.

    This is the real win of P2: before the dialect guards, alembic
    crashed at migration 004 with `near "'\\n'": syntax error`. Now it
    sails through all four.

    KNOWN LIMITATION (documented in the PR parking lot, NOT fixed here):
    full `alembic upgrade head` on a *cold* sqlite DB still fails later
    at migration 021 because `day_chats` (and `telegram_user_mappings`
    at 006) are created by `init_db()`'s `create_all`, not by any
    alembic migration — the early "phantom" CREATE-TABLE migrations were
    deleted from the repo. Restoring them is a separate, larger effort.
    So this test asserts we clear the PG-only-construct gauntlet (through
    rev 020), which is exactly what these dialect guards are responsible
    for.
    """
    import subprocess
    import sys

    db_path = tmp_path / "alembic_sqlite_gate.db"
    env = {
        **_os_environ_for_subprocess(),
        "ENVIRONMENT": "test",
        "PYTHONPATH": ".",
        "DATABASE_URL": f"sqlite+aiosqlite:///{db_path}",
    }
    backend_dir = Path(__file__).resolve().parent.parent
    # Upgrade only as far as rev 020 — the last revision before the
    # phantom-table gap at 021. Every PG-only-construct migration
    # (004/007/008/009) is below this watermark.
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "020"],
        cwd=str(backend_dir),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    combined = result.stdout + result.stderr
    assert result.returncode == 0, (
        f"alembic upgrade 020 failed on sqlite — a PG-only construct is "
        f"unguarded:\n{combined[-3000:]}"
    )
    # Sanity: confirm we actually executed the PG-only-construct revisions.
    for rev in ("004_memory_evolution", "008_hybrid_retrieval", "009_entity_graph_v2"):
        assert rev in combined, f"expected migration {rev} to run; log:\n{combined[-2000:]}"


def _os_environ_for_subprocess() -> dict:
    import os
    return dict(os.environ)
