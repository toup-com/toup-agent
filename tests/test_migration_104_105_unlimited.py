"""Alembic 104 + 105 — the Unlimited plan row, the grandfather columns, and
the ``unlimited_grants`` audit table.

A real round-trip, not a source grep. Both migrations are executed against a
fresh SQLite DB seeded with the pre-104 schema, through alembic's own
``Operations`` context, so ``op.get_bind()``, ``op.add_column``,
``op.create_table`` and ``op.drop_column`` do what they will do on the
platform DB rather than what a shim re-implements.

They earn a round-trip rather than the source-grep treatment 059 gets because
three of their four behaviours are CONDITIONAL — the free daily-cap fix only
reverses when its marker says this upgrade applied it, 104's downgrade refuses
while any balance still holds the plan, and 105's refuses while the table
holds a single row — and a conditional that is only ever read, never executed,
is the class of defect this repo has shipped most often.

The partial unique index is the other reason. ``uq_unlimited_grant_live`` is
what makes a revoked grant re-grantable, and a plain unique index would look
identical in review while making every user un-re-grantable forever after
their first revoke. Only running it can tell the two apart.
"""
from __future__ import annotations

import importlib.util as ilu
import re
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy import create_engine, inspect, text

BACKEND_DIR = Path(__file__).resolve().parent.parent
VERSIONS = BACKEND_DIR / "alembic/versions"
M104_PATH = VERSIONS / "20260914_0104_104_unlimited_plan.py"
M105_PATH = VERSIONS / "20260914_0105_105_unlimited_grants.py"


def _load(name: str, path: Path):
    spec = ilu.spec_from_file_location(name, str(path))
    mod = ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(conn, fn):
    """Execute a migration function with alembic's real op proxy bound."""
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    ctx = MigrationContext.configure(conn)
    with Operations.context(ctx):
        fn()


# The pre-104 schema, cut down to the tables these two migrations touch.
# `free` carries the daily cap of 15 that database.py's seed still writes —
# the drift 104 step 2 exists to correct.
_PRE_104 = [
    """CREATE TABLE users (
           id VARCHAR(36) PRIMARY KEY,
           email VARCHAR(320),
           role VARCHAR(20) DEFAULT 'beta_user'
       )""",
    """CREATE TABLE subscription_plans (
           id VARCHAR(20) PRIMARY KEY,
           display_name VARCHAR(50) NOT NULL,
           price_cents INTEGER NOT NULL,
           stripe_price_id VARCHAR(120),
           message_credits_monthly INTEGER NOT NULL,
           integration_credits_monthly INTEGER NOT NULL,
           message_credits_daily_cap INTEGER,
           rollover_message_credits BOOLEAN NOT NULL DEFAULT 0,
           rollover_integration_credits BOOLEAN NOT NULL DEFAULT 0,
           rollover_max_pct INTEGER NOT NULL DEFAULT 0,
           active BOOLEAN NOT NULL DEFAULT 1,
           sort_order INTEGER NOT NULL DEFAULT 0,
           created_at TIMESTAMP
       )""",
    """CREATE TABLE credit_balances (
           user_id VARCHAR(36) PRIMARY KEY,
           plan_id VARCHAR(20) NOT NULL REFERENCES subscription_plans(id)
       )""",
    """CREATE TABLE apple_subscriptions (
           id VARCHAR(36) PRIMARY KEY,
           user_id VARCHAR(36) NOT NULL,
           original_transaction_id VARCHAR(64) NOT NULL UNIQUE,
           product_id VARCHAR(128) NOT NULL,
           plan_id VARCHAR(20) NOT NULL,
           status VARCHAR(20) NOT NULL,
           expires_date TIMESTAMP,
           environment VARCHAR(16)
       )""",
    """CREATE TABLE platform_settings (
           key VARCHAR(120) PRIMARY KEY,
           value TEXT,
           updated_at TIMESTAMP
       )""",
    "INSERT INTO subscription_plans VALUES "
    "('free','Free',0,NULL,100,500,15,0,0,0,1,0,CURRENT_TIMESTAMP)",
    "INSERT INTO subscription_plans VALUES "
    "('starter','Starter',1600,NULL,130,2500,NULL,0,0,0,1,10,CURRENT_TIMESTAMP)",
    "INSERT INTO subscription_plans VALUES "
    "('builder','Builder',4000,NULL,320,12500,NULL,0,0,0,1,20,CURRENT_TIMESTAMP)",
]


@pytest.fixture
def db(tmp_path):
    """A pre-104 platform DB. File-backed, per-test: the sweep runs several
    pytest processes at once and a fixed path is a live collision (the trap
    test_migration_021_roundtrip documents)."""
    engine = create_engine(f"sqlite:///{tmp_path}/plat.db")
    with engine.begin() as conn:
        for stmt in _PRE_104:
            conn.execute(text(stmt))
    yield engine
    engine.dispose()


@pytest.fixture
def m104():
    return _load("mig104", M104_PATH)


@pytest.fixture
def m105():
    return _load("mig105", M105_PATH)


def _plan(conn, pid):
    return conn.execute(
        text("SELECT id, display_name, price_cents, message_credits_monthly, "
             "integration_credits_monthly, message_credits_daily_cap, active, "
             "sort_order FROM subscription_plans WHERE id = :p"), {"p": pid},
    ).first()


# ── revision identity ────────────────────────────────────────────────


def test_the_chain_is_103_104_105():
    """Two branches can each hold one head and still be lethal together: a
    duplicate revision id merges cleanly and then alembic applies NOTHING on
    every boot. migration-lint guards this in CI; pin it here too, because the
    grandfather command is unrunnable if 104 never lands."""
    s104 = M104_PATH.read_text(encoding="utf-8")
    s105 = M105_PATH.read_text(encoding="utf-8")
    assert 'revision = "104"' in s104 and 'down_revision = "103"' in s104
    assert 'revision = "105"' in s105 and 'down_revision = "104"' in s105


def test_the_price_lives_in_exactly_one_constant():
    """CAD 18.95, decided 2026-09-13. 18.97 was asked for and is not a price
    Apple sells; the Canadian grid runs X.00/X.49/X.90/X.95/X.99 and 18.95 is
    on it. The number must stay a single named constant — a literal copied
    into the migration, the seed or the paywall is a price the App Store will
    not charge, and the fallback below exists only for an app/ refactor."""
    from app.db import plan_catalog

    assert plan_catalog.UNLIMITED_PRICE_CENTS == 1895

    # The migration reads the constant; its literal fallback exists only for
    # an app/ refactor and must not become a second source of truth.
    s104 = M104_PATH.read_text(encoding="utf-8")
    assert "from app.db.plan_catalog import" in s104
    assert '"price": UNLIMITED_PRICE_CENTS' in s104

    # …and the fallback must not silently drift from it. An ImportError here
    # is the one path that reaches the literal, so a stale copy would seed a
    # price nobody chose, in the one code path nobody exercises.
    fallback = re.search(
        r"^    UNLIMITED_PRICE_CENTS = (\d+)$", s104, re.MULTILINE
    )
    assert fallback, "the migration's literal fallback moved or was renamed"
    assert int(fallback.group(1)) == plan_catalog.UNLIMITED_PRICE_CENTS


# ── 104 upgrade ──────────────────────────────────────────────────────


def test_104_adds_the_unlimited_row_from_the_catalogue(db, m104):
    from app.db import plan_catalog

    with db.begin() as conn:
        _run(conn, m104.upgrade)
    with db.connect() as conn:
        row = _plan(conn, "unlimited")
    assert row is not None, "the Unlimited plan row was not created"
    assert row[1] == plan_catalog.UNLIMITED_PLAN_DISPLAY_NAME
    assert row[2] == plan_catalog.UNLIMITED_PRICE_CENTS
    assert row[3] == plan_catalog.UNLIMITED_MESSAGE_CREDITS_MONTHLY
    assert row[4] == plan_catalog.UNLIMITED_INTEGRATION_CREDITS_MONTHLY
    assert row[5] is None, "Unlimited must carry no daily cap"
    assert row[7] == plan_catalog.UNLIMITED_SORT_ORDER
    assert bool(row[6]) is False, (
        "the plan must ship INACTIVE. `active` has exactly one reader in the "
        "backend — GET /api/billing/plans, which is the PUBLIC unauthenticated "
        "pricing catalogue — so an active row publishes a sixth card at a "
        "PLACEHOLDER price, unattended, on the deploy that lands this branch. "
        "Merchandising is an operator step after the app build ships."
    )


def test_the_seed_and_the_migration_agree_that_it_is_inactive(db, m104):
    """Both seeds must move together; they have already drifted once (053
    seeds free at 30/120/5, 059 bumped it to 100/500/15, production is
    100/500/NULL). A seed that writes active=true undoes the migration's care
    on every fresh environment."""
    seed = (BACKEND_DIR / "app/db/database.py").read_text(encoding="utf-8")
    stmt = seed[seed.index("VALUES ('unlimited', 'Unlimited'"):]
    stmt = stmt[:stmt.index("ON CONFLICT")]
    assert "{_PC.UNLIMITED_SORT_ORDER}, false," in stmt, (
        f"the init_db seed must write active=false:\n{stmt}"
    )
    # And the two agree on the numbers, which is the whole point of
    # plan_catalog being a leaf module both read.
    assert "_PC.UNLIMITED_PRICE_CENTS" in stmt
    assert "_PC.UNLIMITED_MESSAGE_CREDITS_MONTHLY" in stmt
    assert "_PC.UNLIMITED_INTEGRATION_CREDITS_MONTHLY" in stmt


def test_104_corrects_the_free_daily_cap_and_records_that_it_did(db, m104):
    """Production is already NULL, so this UPDATE matches zero rows there. It
    exists for a fresh environment, where database.py's seed still writes 15 —
    a cap a single gpt-5.5 turn (26-28 credits quoted) can never satisfy."""
    with db.begin() as conn:
        _run(conn, m104.upgrade)
    with db.connect() as conn:
        assert _plan(conn, "free")[5] is None
        marker = conn.execute(text(
            "SELECT value FROM platform_settings WHERE key = "
            "'credit.free_daily_cap_null_104_applied'")).first()
    assert marker is not None, (
        "without the marker, downgrade() cannot tell a cap it removed from a "
        "cap that was already NULL, and would invent one on the way back"
    )


def test_104_leaves_an_already_null_cap_unmarked(db, m104):
    """The production shape. Nothing changed, so nothing is recorded — and a
    later downgrade must therefore NOT stamp 15 onto the free plan."""
    with db.begin() as conn:
        conn.execute(text(
            "UPDATE subscription_plans SET message_credits_daily_cap = NULL "
            "WHERE id = 'free'"))
        _run(conn, m104.upgrade)
    with db.begin() as conn:
        assert conn.execute(text(
            "SELECT COUNT(*) FROM platform_settings")).scalar() == 0
        _run(conn, m104.downgrade)
    with db.connect() as conn:
        assert _plan(conn, "free")[5] is None, (
            "downgrade invented a daily cap this upgrade never removed"
        )


def test_104_adds_the_grandfather_columns(db, m104):
    with db.begin() as conn:
        _run(conn, m104.upgrade)
    cols = {c["name"] for c in inspect(db).get_columns("apple_subscriptions")}
    assert {"grandfathered_at", "grandfathered_product_id"} <= cols


def test_104_is_idempotent(db, m104):
    """Applied twice must be indistinguishable from applied once — the plan row
    is not duplicated, and the free cap is not re-marked."""
    with db.begin() as conn:
        _run(conn, m104.upgrade)
    with db.begin() as conn:
        _run(conn, m104.upgrade)
    with db.connect() as conn:
        assert conn.execute(text(
            "SELECT COUNT(*) FROM subscription_plans WHERE id='unlimited'"
        )).scalar() == 1
        assert conn.execute(text(
            "SELECT COUNT(*) FROM platform_settings")).scalar() == 1


def test_104_skips_an_agent_db_entirely(tmp_path, m104):
    """Agent (per-tenant) DBs carry no credit state. The migration must no-op
    rather than raise — every tenant runs this chain."""
    engine = create_engine(f"sqlite:///{tmp_path}/tenant.db")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE memories (id VARCHAR(36) PRIMARY KEY)"))
    with engine.begin() as conn:
        _run(conn, m104.upgrade)
        _run(conn, m104.downgrade)
    assert "subscription_plans" not in inspect(engine).get_table_names()
    engine.dispose()


# ── 104 downgrade ────────────────────────────────────────────────────


def test_104_downgrade_refuses_ENTIRELY_while_an_account_holds_the_plan(db, m104):
    """Never delete billing state out from under a live account, and refuse
    BEFORE the first DDL — not halfway down.

    The refusal used to be partial: the only guard sat on step 1, so a
    downgrade dropped `grandfathered_at` / `grandfathered_product_id` for every
    row first and only then declined to remove the plan row those accounts are
    still on. The accounts kept plan_id='unlimited' while the only record of
    WHY was gone and unrecoverable — the receipt stores the state BEFORE the
    stamp, not the stamp. Re-upgrading re-adds the columns as NULL, and the
    next legacy DID_RENEW then resolves to builder/starter and silently
    reverts the grandfather.
    """
    with db.begin() as conn:
        _run(conn, m104.upgrade)
        conn.execute(text(
            "INSERT INTO credit_balances VALUES ('u1','unlimited')"))

    with pytest.raises(RuntimeError) as exc:
        with db.begin() as conn:
            _run(conn, m104.downgrade)
    assert "grandfather_unlimited --revert" in str(exc.value)

    with db.connect() as conn:
        assert _plan(conn, "unlimited") is not None
        assert _plan(conn, "free")[5] is None, "the free cap was rolled back"
    cols = {c["name"] for c in inspect(db).get_columns("apple_subscriptions")}
    assert "grandfathered_at" in cols, "the receipt columns were destroyed"
    assert "grandfathered_product_id" in cols


def test_104_downgrade_also_refuses_on_a_stamped_subscription(db, m104):
    """The second arm, and the one the plan-row check cannot cover: a
    subscription can be stamped while its balance has already moved off
    Unlimited (a lapse, a revoked grant). The stamp is still the only thing
    that keeps the next DID_RENEW from reverting them."""
    with db.begin() as conn:
        _run(conn, m104.upgrade)
        conn.execute(text(
            "INSERT INTO apple_subscriptions "
            "(id, user_id, original_transaction_id, product_id, plan_id, "
            " status, environment, grandfathered_at, grandfathered_product_id) "
            "VALUES ('s1','u1','T1','ai.toup.app.sub.builder','builder',"
            "        'active','Production', :t, 'ai.toup.app.sub.builder')"),
            {"t": datetime.utcnow()})

    with pytest.raises(RuntimeError) as exc:
        with db.begin() as conn:
            _run(conn, m104.downgrade)
    assert "grandfathered apple_subscriptions row" in str(exc.value)

    cols = {c["name"] for c in inspect(db).get_columns("apple_subscriptions")}
    assert "grandfathered_at" in cols


def test_104_downgrade_reverses_everything_once_nobody_holds_it(db, m104):
    with db.begin() as conn:
        _run(conn, m104.upgrade)
    with db.begin() as conn:
        _run(conn, m104.downgrade)
    with db.connect() as conn:
        assert _plan(conn, "unlimited") is None
        assert _plan(conn, "free")[5] == 15, "the pre-104 cap was not restored"
        assert conn.execute(text(
            "SELECT COUNT(*) FROM platform_settings")).scalar() == 0
    cols = {c["name"] for c in inspect(db).get_columns("apple_subscriptions")}
    assert not ({"grandfathered_at", "grandfathered_product_id"} & cols)


# ── 105 ──────────────────────────────────────────────────────────────


def _grant(conn, user, **kw):
    now = datetime.utcnow()
    vals = dict(
        id=str(uuid.uuid4()), granted_to_user_id=user, sponsor_kind="apple",
        sponsor_user_id=None, sponsor_original_txn_id="txn-1",
        sponsor_stripe_sub_id=None, reason="second account", expires_at=None,
        granted_by_user_id="admin-1", granted_by_email="ops@toup.ai",
        granted_at=now, revoked_at=None, revoked_by_user_id=None,
        revoked_by_email=None, revoked_reason=None, lapsed_at=None,
        lapsed_reason=None, created_at=now, updated_at=now,
    )
    vals.update(kw)
    conn.execute(text(
        "INSERT INTO unlimited_grants (id, granted_to_user_id, sponsor_kind, "
        "sponsor_user_id, sponsor_original_txn_id, sponsor_stripe_sub_id, "
        "reason, expires_at, granted_by_user_id, granted_by_email, granted_at, "
        "revoked_at, revoked_by_user_id, revoked_by_email, revoked_reason, "
        "lapsed_at, lapsed_reason, created_at, updated_at) VALUES "
        "(:id, :granted_to_user_id, :sponsor_kind, :sponsor_user_id, "
        ":sponsor_original_txn_id, :sponsor_stripe_sub_id, :reason, :expires_at, "
        ":granted_by_user_id, :granted_by_email, :granted_at, :revoked_at, "
        ":revoked_by_user_id, :revoked_by_email, :revoked_reason, :lapsed_at, "
        ":lapsed_reason, :created_at, :updated_at)"), vals)
    return vals["id"]


@pytest.fixture
def upgraded(db, m104, m105):
    with db.begin() as conn:
        _run(conn, m104.upgrade)
        _run(conn, m105.upgrade)
        conn.execute(text("INSERT INTO users (id, email) VALUES "
                          "('u1','a@example.com'), ('u2','b@example.com')"))
    return db


def test_105_creates_the_table_and_its_indexes(upgraded):
    insp = inspect(upgraded)
    assert "unlimited_grants" in insp.get_table_names()
    names = {ix["name"] for ix in insp.get_indexes("unlimited_grants")}
    assert {"uq_unlimited_grant_live", "ix_unlimited_grant_user",
            "ix_unlimited_grant_sponsor_txn",
            "ix_unlimited_grant_sponsor_usr"} <= names


def test_105_is_idempotent(db, m104, m105):
    with db.begin() as conn:
        _run(conn, m104.upgrade)
        _run(conn, m105.upgrade)
    with db.begin() as conn:
        _run(conn, m105.upgrade)
    assert "unlimited_grants" in inspect(db).get_table_names()


def test_one_live_grant_per_user_is_enforced_by_the_index(upgraded):
    """The service layer's 409 is the message; this is the constraint. Two
    concurrent grants can both pass the SELECT and only the index stops the
    second."""
    with upgraded.begin() as conn:
        _grant(conn, "u1")
    with pytest.raises(sa.exc.IntegrityError):
        with upgraded.begin() as conn:
            _grant(conn, "u1")


def test_a_revoked_grant_leaves_the_user_re_grantable(upgraded):
    """The whole reason the index is PARTIAL. A plain unique index reviews
    identically and makes every user un-re-grantable forever after their first
    revoke — while the row itself must survive, because it is billing history."""
    with upgraded.begin() as conn:
        first = _grant(conn, "u1")
        conn.execute(text("UPDATE unlimited_grants SET revoked_at = :t "
                          "WHERE id = :i"),
                     {"t": datetime.utcnow(), "i": first})
    with upgraded.begin() as conn:
        _grant(conn, "u1")
    with upgraded.connect() as conn:
        assert conn.execute(text(
            "SELECT COUNT(*) FROM unlimited_grants WHERE granted_to_user_id='u1'"
        )).scalar() == 2, "the revoked row must still be there"


def test_a_lapsed_grant_also_frees_the_slot(upgraded):
    with upgraded.begin() as conn:
        first = _grant(conn, "u1")
        conn.execute(text("UPDATE unlimited_grants SET lapsed_at = :t, "
                          "lapsed_reason = 'apple:expired' WHERE id = :i"),
                     {"t": datetime.utcnow(), "i": first})
    with upgraded.begin() as conn:
        _grant(conn, "u1")


def test_two_users_each_hold_their_own_live_grant(upgraded):
    with upgraded.begin() as conn:
        _grant(conn, "u1")
        _grant(conn, "u2")


def test_105_downgrade_refuses_to_destroy_an_audit_trail(upgraded, m105):
    """Even a grant that already lapsed records who comped whom, why, and on
    whose subscription. Dropping the table erases all of it with no trace."""
    with upgraded.begin() as conn:
        gid = _grant(conn, "u1")
        conn.execute(text("UPDATE unlimited_grants SET lapsed_at = :t "
                          "WHERE id = :i"),
                     {"t": datetime.utcnow(), "i": gid})
    with pytest.raises(RuntimeError, match="refusing to drop an audit trail"):
        with upgraded.begin() as conn:
            _run(conn, m105.downgrade)
    assert "unlimited_grants" in inspect(upgraded).get_table_names()


def test_105_downgrade_drops_an_empty_table(upgraded, m105):
    with upgraded.begin() as conn:
        _run(conn, m105.downgrade)
    assert "unlimited_grants" not in inspect(upgraded).get_table_names()


def test_105_skips_an_agent_db(tmp_path, m105):
    engine = create_engine(f"sqlite:///{tmp_path}/tenant.db")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE memories (id VARCHAR(36) PRIMARY KEY)"))
    with engine.begin() as conn:
        _run(conn, m105.upgrade)
        _run(conn, m105.downgrade)
    assert "unlimited_grants" not in inspect(engine).get_table_names()
    engine.dispose()


def test_the_full_chain_round_trips(db, m104, m105):
    """104 → 105 → 105⁻¹ → 104⁻¹ leaves the schema and the catalogue exactly as
    they were. A downgrade nobody runs is a downgrade nobody knows is broken."""
    before_tables = set(inspect(db).get_table_names())
    before_cols = {c["name"] for c in inspect(db).get_columns("apple_subscriptions")}
    with db.connect() as conn:
        before_free = _plan(conn, "free")

    with db.begin() as conn:
        _run(conn, m104.upgrade)
        _run(conn, m105.upgrade)
    with db.begin() as conn:
        _run(conn, m105.downgrade)
        _run(conn, m104.downgrade)

    assert set(inspect(db).get_table_names()) == before_tables
    assert {c["name"] for c in inspect(db).get_columns("apple_subscriptions")} \
        == before_cols
    with db.connect() as conn:
        assert _plan(conn, "free") == before_free
        assert _plan(conn, "unlimited") is None
