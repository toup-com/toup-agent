"""
Smoke test — confirm `init_db()` ALTER list applies the migration 021 columns
so existing user DBs self-heal on next agent startup and new DBs get them.

Doesn't actually run init_db (that needs the full app stack). Instead asserts
the ALTER statements are present in the init_db() source, which is the
canonical mechanism this codebase uses for idempotent schema evolution.

Run: python tests/test_init_db_applies_021.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_init_db_has_021_alters():
    path = Path(__file__).resolve().parent.parent / "app/db/database.py"
    src = path.read_text()

    # Every column migration 021 adds must have a matching ADD COLUMN IF NOT
    # EXISTS statement in init_db()'s _alter_statements list.
    required = [
        "llm_proxy_events ADD COLUMN IF NOT EXISTS operation_type",
        "day_chats ADD COLUMN IF NOT EXISTS archival_summary",
        "day_chats ADD COLUMN IF NOT EXISTS archival_summary_generated_at",
        "day_chats ADD COLUMN IF NOT EXISTS archival_summary_status",
        "CREATE INDEX IF NOT EXISTS ix_llm_proxy_operation_type",
    ]
    missing = [r for r in required if r not in src]
    assert not missing, (
        f"init_db() is missing {len(missing)} migration-021 alter statements. "
        f"Without these, existing agent DBs won't self-heal on restart. "
        f"Missing: {missing}"
    )
    print("✓ test_init_db_has_021_alters")


if __name__ == "__main__":
    test_init_db_has_021_alters()
    print("\nmigration 021 self-heal path: CONFIRMED in init_db()")
