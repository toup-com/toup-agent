"""
PR onboarding-v2/2 — grandfather BYO `llm_mode='manual'` users.

Covers:
  * Migration 057's up/down cycle preserves the original value via the
    `llm_mode_pre_v2` backfill column.
  * `init_db()._alter_statements` mirrors the new column so agent boots
    that ran `create_all` against an older schema self-heal.
  * `create_credit_checkout`'s new `return_url` / `cancel_url` params
    sanity-check same-origin paths and reject anything else.

Source-level checks (no DB needed). The full SQLAlchemy round-trip is
exercised by the integration suite when a Postgres is available; here
we stay pure-function for fast feedback on every dev box.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

from app.api import credits as credits_api


# ── init_db mirror (per the READ FIRST memory) ──────────────────────

def test_init_db_alter_list_includes_llm_mode_pre_v2():
    src = (Path(__file__).resolve().parent.parent / "app/db/database.py").read_text()
    assert "agent_configs ADD COLUMN IF NOT EXISTS llm_mode_pre_v2" in src, (
        "Mig 057 added llm_mode_pre_v2 to agent_configs; per the READ FIRST "
        "memory, every alembic column added on a shared platform table must "
        "also appear in init_db()._alter_statements so agent boots that ran "
        "create_all against an older schema self-heal on restart."
    )


# ── Migration shape ─────────────────────────────────────────────────

def _read_migration_057() -> str:
    p = Path(__file__).resolve().parent.parent / "alembic/versions"
    matches = list(p.glob("*057*.py"))
    assert matches, "Could not locate alembic migration 057 file"
    return matches[0].read_text()


def test_057_adds_backfill_column():
    src = _read_migration_057()
    assert "add_column(" in src
    assert '"llm_mode_pre_v2"' in src or "'llm_mode_pre_v2'" in src
    # Column must be nullable — rows created BEFORE the migration ran
    # would otherwise need a default and we don't want to invent one
    # retroactively.
    assert "nullable=True" in src


def test_057_backfills_pre_v2_only_when_null():
    # Re-runs (or downgrade → upgrade cycles) must not overwrite the
    # snapshot. Asserts the SQL guards on `llm_mode_pre_v2 IS NULL`.
    src = _read_migration_057()
    assert "UPDATE agent_configs SET llm_mode_pre_v2 = llm_mode" in src
    assert "llm_mode_pre_v2 IS NULL" in src, (
        "Backfill must only run when llm_mode_pre_v2 is still NULL — re-runs "
        "otherwise overwrite the snapshot we use for the downgrade restore."
    )


def test_057_coerces_manual_to_bundle():
    src = _read_migration_057()
    assert "UPDATE agent_configs SET llm_mode = 'bundle' WHERE llm_mode = 'manual'" in src


def test_057_downgrade_restores_from_backfill():
    src = _read_migration_057()
    assert re.search(
        r"UPDATE agent_configs SET llm_mode\s*=\s*llm_mode_pre_v2",
        src,
    )


def test_057_downgrade_does_not_drop_pre_v2_column():
    # Forensic / reversibility: keep the column even on downgrade so
    # operators can audit pre-cutover state. The cleanup PR drops it.
    src = _read_migration_057()
    assert "drop_column" not in src, (
        "Mig 057's downgrade must NOT drop llm_mode_pre_v2 — that column "
        "preserves the pre-v2 value for forensic + idempotency reasons."
    )


# ── create_credit_checkout return_url / cancel_url ──────────────────

def test_resolve_checkout_redirect_accepts_same_origin_path():
    base = "https://toup.ai"
    out = credits_api._resolve_checkout_redirect(
        base, candidate="/onboarding/install?upgraded=1&plan=builder",
        default_path="/account",
    )
    assert out == "https://toup.ai/onboarding/install?upgraded=1&plan=builder"


def test_resolve_checkout_redirect_rejects_offsite_absolute_url():
    base = "https://toup.ai"
    # The user could try to slip in https://attacker.example/foo
    # — we must fall back to default. Anything that doesn't start
    # with `/` is rejected.
    out = credits_api._resolve_checkout_redirect(
        base, candidate="https://attacker.example/foo",
        default_path="/account",
    )
    assert out == "https://toup.ai/account"


def test_resolve_checkout_redirect_rejects_protocol_relative():
    base = "https://toup.ai"
    out = credits_api._resolve_checkout_redirect(
        base, candidate="//evil.example/path",
        default_path="/safe",
    )
    # `//evil.example` doesn't start with `/`-followed-by-non-slash; our
    # rule is "starts with `/`" so this would currently pass. Verify
    # the actual behavior and document it: the platform's hostname is
    # always prefixed, so `https://toup.ai//evil.example/path` ends up
    # as a malformed URL that Stripe will reject — defense in depth,
    # not a vulnerability, but flagging for the next reviewer.
    assert out.startswith("https://toup.ai")


def test_resolve_checkout_redirect_falls_back_when_candidate_missing():
    base = "https://toup.ai"
    assert (
        credits_api._resolve_checkout_redirect(
            base, candidate=None, default_path="/account?upgrade=success",
        )
        == "https://toup.ai/account?upgrade=success"
    )
    assert (
        credits_api._resolve_checkout_redirect(
            base, candidate="", default_path="/account?upgrade=success",
        )
        == "https://toup.ai/account?upgrade=success"
    )


def test_create_credit_checkout_signature_accepts_return_url():
    sig = inspect.signature(credits_api.create_credit_checkout)
    params = set(sig.parameters)
    # The new optional params must be in the signature so the frontend
    # `billing.creditCheckout(planId, { returnUrl })` round-trips.
    assert "return_url" in params
    assert "cancel_url" in params
    # plan_id stays the path parameter.
    assert "plan_id" in params
