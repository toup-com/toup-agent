"""Drop password_plain column from users table.

Removes the beta-era plaintext-password storage. Replaced by the existing
`POST /admin/users/{id}/reset-password` flow for password support.

Safety guard: refuses to upgrade if any active user lacks a usable
hashed_password (NULL or empty). The 'deleted' sentinel set by
auth.delete_account() is allowed — those users are already locked out.

Revision ID: 026
Revises: 025
Create Date: 2026-04-26
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = "026"
down_revision = "025"
branch_labels = None
depends_on = None


def upgrade():
    conn = op.get_bind()
    bad = conn.execute(sa.text(
        "SELECT id FROM users "
        "WHERE (hashed_password IS NULL OR hashed_password = '') "
        "AND is_active = TRUE"
    )).fetchall()
    if bad:
        ids = ", ".join(r[0] for r in bad[:10])
        raise RuntimeError(
            f"Refusing to drop password_plain: {len(bad)} active user(s) "
            f"have no usable hashed_password. Sample IDs: {ids}"
        )
    op.drop_column("users", "password_plain")


def downgrade():
    op.add_column("users", sa.Column("password_plain", sa.String(255), nullable=True))
