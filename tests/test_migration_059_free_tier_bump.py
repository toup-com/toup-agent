"""Alembic migration 059 — Free-tier credit bump (30/120/5 → 100/500/15).

Source-grep guards that pin the migration's intended effect. The full
end-to-end alembic upgrade/downgrade roundtrip isn't tested here — the
migration is two SQL UPDATEs against `subscription_plans` and
`credit_balances`, both well-exercised by the test_credit_service.py
suite once init_db has seeded the post-bump numbers. The grep tests
catch the higher-risk regressions: someone editing the bump numbers,
the revision chain, or the idempotency-flag key without noticing.
"""
from __future__ import annotations

from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parent.parent
_MIG_PATH = (
    BACKEND_DIR
    / "alembic/versions/20260525_0059_059_free_tier_credit_bump.py"
)
_MIG_SRC = _MIG_PATH.read_text(encoding="utf-8") if _MIG_PATH.exists() else ""


def test_migration_file_exists_with_right_revision_chain():
    assert _MIG_PATH.exists(), f"migration 059 missing at {_MIG_PATH}"
    assert 'revision = "059"' in _MIG_SRC
    assert 'down_revision = "058"' in _MIG_SRC, (
        "059 must chain off 058 (grandfather byo llm_mode)"
    )


def test_migration_writes_new_free_tier_numbers():
    """The plan-row UPDATE must set message_credits_monthly=100,
    integration_credits_monthly=500, message_credits_daily_cap=15.
    These three values define the new Free tier — changing them
    silently in a future edit would mis-size the tier without any
    test signal."""
    assert "message_credits_monthly = 100" in _MIG_SRC
    assert "integration_credits_monthly = 500" in _MIG_SRC
    assert "message_credits_daily_cap = 15" in _MIG_SRC
    assert "WHERE id = 'free'" in _MIG_SRC


def test_migration_tops_up_in_flight_balances_additively():
    """In-flight free-tier balances must get +70 msg / +380 int
    (the new-minus-old delta) added to whatever they already had —
    NOT a reset to the new monthly. A reset would over-grant users
    who hadn't spent anything yet, and could clip remaining for
    users on a paid downgrade path. Additive is the safe semantics."""
    assert "message_credits_remaining = message_credits_remaining + 70" in _MIG_SRC
    assert "integration_credits_remaining = integration_credits_remaining + 380" in _MIG_SRC


def test_migration_is_idempotent_via_platform_settings_flag():
    """Re-running 059 must not double-bump balances. The migration
    sets a one-shot flag in platform_settings; subsequent runs check
    the flag and skip the additive UPDATE."""
    assert "credit.free_bump_059_applied" in _MIG_SRC
    assert "platform_settings" in _MIG_SRC


def test_downgrade_reverses_plan_only_not_balances():
    """Balance top-ups are deliberately irreversible — a user who
    consumed bonus credits would land below zero on naive reversal.
    Downgrade restores the plan row to 30/120/5 and leaves balances
    alone. Pin this so a future refactor doesn't introduce a buggy
    clawback."""
    # The downgrade must touch the plan row.
    assert "def downgrade()" in _MIG_SRC
    # Look for the explicit reversal numbers in the downgrade block.
    downgrade_block = _MIG_SRC.split("def downgrade()", 1)[1]
    assert "message_credits_monthly = 30" in downgrade_block
    assert "integration_credits_monthly = 120" in downgrade_block
    assert "message_credits_daily_cap = 5" in downgrade_block
    # Downgrade must NOT subtract from balances (that would clip
    # bonus credits some users have already spent).
    assert "message_credits_remaining = message_credits_remaining -" not in downgrade_block
    assert "integration_credits_remaining = integration_credits_remaining -" not in downgrade_block


def test_database_py_seed_mirrors_post_bump_numbers():
    """init_db's _seed_statements creates the free row on fresh DBs
    via INSERT ... ON CONFLICT DO NOTHING. On production this is a
    no-op (mig 053 already inserted, mig 059 then UPDATEd), but on
    CI test fixtures + brand-new tenant Postgres the INSERT fires and
    must use the post-bump numbers — otherwise tests against fresh
    DBs would assert against 30/120/5."""
    db_py = (BACKEND_DIR / "app/db/database.py").read_text(encoding="utf-8")
    # The free-tier INSERT must carry the post-bump numbers as
    # positional literals in the VALUES clause.
    assert "VALUES ('free', 'Free', 0, 100, 500, 15" in db_py, (
        "init_db seed must mirror mig 059's post-bump numbers "
        "(100 msg / 500 int / 15 day-cap), not the legacy 30/120/5"
    )
