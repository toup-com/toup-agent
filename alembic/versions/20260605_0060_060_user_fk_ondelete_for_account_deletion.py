"""user-referencing FKs: ON DELETE CASCADE / SET NULL so account deletion works.

Account deletion (`delete_user_completely`) wipes the user's child rows in
one transaction, commits, then runs `DELETE FROM users`. Production failed at
that final step with:

    ForeignKeyViolationError: update or delete on "users" violates
    "agent_configs_user_id_fkey" on table "agent_configs"

Root cause: the wipe DID delete `agent_configs`, but several get-or-create-
config endpoints (toup_code `/code/status`, soul, agent_setup) RE-CREATE the
row from the user's still-open session DURING the delete window, so a fresh
`agent_configs` row existed again by the time `DELETE FROM users` ran — and the
FK had no ON DELETE action. Same latent risk for `managed_containers` (container
reconciler / provision re-create) and `extension_devices` (extension /pair,
/suggest polls). `platform_settings.updated_by_user_id` and `rollouts.triggered_by`
are audit references that blocked deletion of admins.

Fix: make the user-row delete atomic + race-proof at the DB level —
  - user-owned data  -> ON DELETE CASCADE  (agent_configs, managed_containers,
    extension_devices)
  - audit references  -> ON DELETE SET NULL (platform_settings.updated_by_user_id,
    rollouts.triggered_by)

These are all small tables, so the DROP+ADD CONSTRAINT is effectively instant.
Already applied manually to production 2026-06-05; this migration makes it
durable + reproducible on fresh DBs. Idempotent (DROP CONSTRAINT IF EXISTS).
Postgres-only — SQLite (tests) gets the same behavior from the ORM `ondelete`
on create_all and does not support ALTER ... CONSTRAINT.

Revision ID: 060
Revises: 059
Create Date: 2026-06-05
"""
from __future__ import annotations

import logging

from alembic import op


revision = "060"
down_revision = "059"
branch_labels = None
depends_on = None

logger = logging.getLogger("alembic.060")

# (table, constraint_name, column, on_delete)
_FKS = [
    ("agent_configs", "agent_configs_user_id_fkey", "user_id", "CASCADE"),
    ("managed_containers", "managed_containers_user_id_fkey", "user_id", "CASCADE"),
    ("extension_devices", "extension_devices_user_id_fkey", "user_id", "CASCADE"),
    ("platform_settings", "platform_settings_updated_by_user_id_fkey", "updated_by_user_id", "SET NULL"),
    ("rollouts", "rollouts_triggered_by_fkey", "triggered_by", "SET NULL"),
]


def _is_pg() -> bool:
    return op.get_bind().dialect.name == "postgresql"


def upgrade() -> None:
    if not _is_pg():
        logger.info("[alembic.060] non-postgres dialect; ORM ondelete covers create_all, skipping")
        return
    for table, name, col, ondelete in _FKS:
        op.execute(f'ALTER TABLE {table} DROP CONSTRAINT IF EXISTS {name}')
        op.execute(
            f'ALTER TABLE {table} ADD CONSTRAINT {name} '
            f'FOREIGN KEY ({col}) REFERENCES users(id) ON DELETE {ondelete}'
        )
        logger.info("[alembic.060] %s.%s -> ON DELETE %s", table, col, ondelete)


def downgrade() -> None:
    if not _is_pg():
        return
    # Revert to NO ACTION (the prior default).
    for table, name, col, _ondelete in _FKS:
        op.execute(f'ALTER TABLE {table} DROP CONSTRAINT IF EXISTS {name}')
        op.execute(
            f'ALTER TABLE {table} ADD CONSTRAINT {name} '
            f'FOREIGN KEY ({col}) REFERENCES users(id)'
        )
