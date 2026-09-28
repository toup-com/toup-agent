"""106 — desktop_relay: pairing, devices and local-action consent for Toup for Mac.

Revision ID: 106
Revises: 105
Create Date: 2026-09-20

Schema only. **Nothing here turns the feature on.** The agent's tool family
is gated by ``settings.desktop_relay_enabled``, which defaults to False and
is resolved once per process in ``tool_entitlements.skill_enabled`` — so
these tables can exist, empty, on a fleet whose wire tools array is
byte-identical to today's. Creating them is separable from launching on
purpose: pairing state has to be durable before the first device can pair,
and a migration is not a rollout.

Three tables, all PLATFORM_ONLY (see ``app/db/models/desktop.py`` for the
full argument — in one line, the device row IS the revocation list for a
credential that can read files and run commands, and a second empty tenant
copy would be a revocation check that passes everything while looking
correct).

Two details that are load-bearing rather than stylistic:

  * **``desktop_pairings`` exists because the extension's equivalent does
    not.** ``app/api/extension.py`` keeps pending pairings in a module-level
    dict, and its own docstring says so: "this is sufficient until we scale
    beyond one platform replica". ``railway.json`` sets
    ``numReplicas: 2``. So a ``pair/init`` served by replica A and a
    ``pair/approve`` served by replica B cannot see each other, and pairing
    fails roughly half the time with no error that names the cause. This
    table is that bug not being inherited.

  * **``desktop_pending_actions`` is a SIBLING of
    ``connector_pending_actions``, not a column on it.** Same invariants
    (atomic claim guarded on ``status = 'pending'``, hard ``expires_at``,
    ``tool_name`` as a COLUMN so the approve endpoint cannot be talked into
    running a different tool, rows retained as the audit trail) — but
    ``connector_pending_actions.connector_id`` is NOT NULL and the approve
    path around it resolves that id to an OAuth identity and calls the
    connector dispatcher. A local action has no connector and executes on
    the user's Mac. Adding a discriminator column instead would mean the
    existing ``GET /pending-actions`` serves desktop rows to a client that
    POSTs them to the connector approve route: one route, two executors,
    and the wrong one picked by default.

No ALTER mirror in ``app/db/database.py``. That list exists so tenant DBs,
which never run alembic, can self-heal columns of tables they carry; these
tables have no tenant copy to heal. And no ``SHARED_COLUMN_AUTHORITY``
entries, since that map covers SHARED tables only — both facts are asserted
by ``tests/test_table_partitioning.py``.

Downgrade drops all three. Safe in the sense that matters: nothing else
references them, and a dropped device row means every paired Mac has to
pair again, which is the correct failure for a credential store that has
gone away.
"""

from alembic import op
import sqlalchemy as sa

revision = "106"
down_revision = "105"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "desktop_devices",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "user_id",
            sa.String(36),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("device_name", sa.String(200), nullable=False),
        sa.Column("app_version", sa.String(40), nullable=True),
        sa.Column("os_version", sa.String(40), nullable=True),
        sa.Column("flavor", sa.String(16), nullable=True),
        # The jti of the CURRENTLY-issued token. Authentication requires the
        # presented jti to equal this AND revoked_at to be NULL, which is
        # what makes the row itself the denylist — there is no separate
        # denylist table to forget to consult. `extension.py:841-848`
        # describes exactly this as a "follow-up" and ships without it.
        sa.Column("token_jti", sa.String(64), nullable=True),
        sa.Column("token_expires_at", sa.DateTime(), nullable=True),
        sa.Column("paired_at", sa.DateTime(), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(), nullable=True),
        sa.Column("revoked_at", sa.DateTime(), nullable=True),
    )
    op.create_index(
        "ix_desktop_devices_user_id", "desktop_devices", ["user_id"],
    )
    op.create_index(
        "ix_desktop_devices_token_jti", "desktop_devices", ["token_jti"],
    )
    op.create_index(
        "ix_desktop_devices_revoked_at", "desktop_devices", ["revoked_at"],
    )
    # The connect-time authenticator's own query shape: this user, not
    # revoked.
    op.create_index(
        "ix_desktop_devices_user_revoked",
        "desktop_devices",
        ["user_id", "revoked_at"],
    )

    op.create_table(
        "desktop_pairings",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("device_code", sa.String(128), nullable=False),
        sa.Column("user_code", sa.String(16), nullable=False),
        sa.Column("device_name", sa.String(200), nullable=False),
        sa.Column("app_version", sa.String(40), nullable=True),
        sa.Column("os_version", sa.String(40), nullable=True),
        sa.Column("flavor", sa.String(16), nullable=True),
        sa.Column(
            "status", sa.String(16), nullable=False, server_default="pending",
        ),
        sa.Column(
            "user_id",
            sa.String(36),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("approved_device_id", sa.String(36), nullable=True),
        sa.Column("approved_token", sa.Text(), nullable=True),
        sa.Column("approved_expires_in", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.Column("last_polled_at", sa.DateTime(), nullable=True),
        sa.Column("decided_at", sa.DateTime(), nullable=True),
        # `expired` is its own value, not a flavour of `denied`: the lazy
        # sweep rewrites a timed-out code from somebody else's request, and
        # folding the two together made the device tell its user "you
        # declined this" about a page they never opened (§1.2 gives the two
        # different answers).
        sa.CheckConstraint(
            "status IN ('pending', 'approved', 'denied', 'expired', "
            "'consumed')",
            name="ck_desktop_pairings_status",
        ),
    )
    # UNIQUE, not merely indexed: `pair/init` retries on collision, and a
    # duplicate user_code would otherwise let one typed code approve a
    # different machine than the one the page named.
    op.create_index(
        "ix_desktop_pairings_device_code",
        "desktop_pairings", ["device_code"], unique=True,
    )
    op.create_index(
        "ix_desktop_pairings_user_code",
        "desktop_pairings", ["user_code"], unique=True,
    )
    op.create_index(
        "ix_desktop_pairings_user_id", "desktop_pairings", ["user_id"],
    )
    op.create_index(
        "ix_desktop_pairings_status", "desktop_pairings", ["status"],
    )
    op.create_index(
        "ix_desktop_pairings_status_expires",
        "desktop_pairings", ["status", "expires_at"],
    )

    op.create_table(
        "desktop_pending_actions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "user_id",
            sa.String(36),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("device_id", sa.String(36), nullable=False),
        sa.Column("tool_name", sa.String(128), nullable=False),
        sa.Column("payload_json", sa.Text(), nullable=False),
        sa.Column("reason", sa.String(240), nullable=True),
        sa.Column(
            "status", sa.String(16), nullable=False, server_default="pending",
        ),
        sa.Column(
            "channel", sa.String(32), nullable=False, server_default="desktop",
        ),
        sa.Column("conversation_id", sa.String(36), nullable=True),
        sa.Column("message_id", sa.String(36), nullable=True),
        # The relay idempotency key, minted once at staging and reused
        # verbatim on every dispatch of this row. RELAY_PROTOCOL.md §5.3: a
        # replayed id is answered from the device's result cache rather than
        # executed twice, so a retried delivery cannot run `rm` again. A
        # fresh id on retry would defeat that completely.
        sa.Column("task_id", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.Column("decided_at", sa.DateTime(), nullable=True),
        sa.Column("decided_via", sa.String(32), nullable=True),
        sa.Column("dispatched_at", sa.DateTime(), nullable=True),
        sa.Column("result_json", sa.Text(), nullable=True),
        sa.CheckConstraint(
            "status IN ('pending', 'approved', 'dispatched', 'executed', "
            "'failed', 'rejected', 'expired')",
            name="ck_desktop_pending_actions_status",
        ),
    )
    op.create_index(
        "ix_desktop_pending_actions_user_id",
        "desktop_pending_actions", ["user_id"],
    )
    op.create_index(
        "ix_desktop_pending_actions_status",
        "desktop_pending_actions", ["status"],
    )
    op.create_index(
        "ix_desktop_pending_actions_conversation_id",
        "desktop_pending_actions", ["conversation_id"],
    )
    op.create_index(
        "ix_desktop_pending_actions_task_id",
        "desktop_pending_actions", ["task_id"],
    )
    op.create_index(
        "ix_desktop_pending_actions_user_status",
        "desktop_pending_actions", ["user_id", "status"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_desktop_pending_actions_user_status",
        table_name="desktop_pending_actions",
    )
    op.drop_index(
        "ix_desktop_pending_actions_task_id",
        table_name="desktop_pending_actions",
    )
    op.drop_index(
        "ix_desktop_pending_actions_conversation_id",
        table_name="desktop_pending_actions",
    )
    op.drop_index(
        "ix_desktop_pending_actions_status",
        table_name="desktop_pending_actions",
    )
    op.drop_index(
        "ix_desktop_pending_actions_user_id",
        table_name="desktop_pending_actions",
    )
    op.drop_table("desktop_pending_actions")

    op.drop_index(
        "ix_desktop_pairings_status_expires", table_name="desktop_pairings",
    )
    op.drop_index("ix_desktop_pairings_status", table_name="desktop_pairings")
    op.drop_index("ix_desktop_pairings_user_id", table_name="desktop_pairings")
    op.drop_index("ix_desktop_pairings_user_code", table_name="desktop_pairings")
    op.drop_index(
        "ix_desktop_pairings_device_code", table_name="desktop_pairings",
    )
    op.drop_table("desktop_pairings")

    op.drop_index(
        "ix_desktop_devices_user_revoked", table_name="desktop_devices",
    )
    op.drop_index("ix_desktop_devices_revoked_at", table_name="desktop_devices")
    op.drop_index("ix_desktop_devices_token_jti", table_name="desktop_devices")
    op.drop_index("ix_desktop_devices_user_id", table_name="desktop_devices")
    op.drop_table("desktop_devices")
