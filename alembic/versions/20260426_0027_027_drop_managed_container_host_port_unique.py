"""Drop unique constraint on managed_containers.host_port.

The `ix_managed_containers_host_port` UNIQUE index is a Phase 2 leftover.
Back then, `host_port` was the actual published port on the Docker host
(e.g., `-p <host_port>:8001`) and naturally had to be unique. Phase 3
moved tenant containers behind Caddy mTLS reverse-proxy on per-tenant
docker bridge networks — host ports are no longer published, and the
provisioning bridge's response field `host_port` is now an internal
bookkeeping counter that may collide across tenants without breaking
anything on the host.

The collision surfaced 2026-04-26 when the first first-time bundle
subscriber (Alireza) was provisioned: bridge returned `host_port: 9002`
which was already taken by parmida's row, the `INSERT` failed with
UniqueViolationError, the exception bubbled past `provision_container`'s
narrow `httpx.*` exception handler, and the user landed at the install
step with a half-created bridge container + no platform DB row.

Dropping the unique index (not the column) keeps the bookkeeping field
for any code that still reads it while removing the wrong-after-Phase-3
constraint. The column itself stays nullable=False to match the model.

Revision ID: 027
Revises: 026
Create Date: 2026-04-26
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = "027"
down_revision = "026"
branch_labels = None
depends_on = None


def upgrade():
    # Drop the UNIQUE index, keep the column. A non-unique index could be
    # added if any query needs to filter by host_port, but right now nothing
    # does — the bridge owns port-allocation decisions, not the platform.
    op.drop_index("ix_managed_containers_host_port", table_name="managed_containers")


def downgrade():
    # Restoring the unique index would fail if any duplicates exist
    # (which is precisely the situation that motivated the drop). Recreate
    # as non-unique on downgrade so the migration is reversible without
    # data surgery; manual re-uniqueness can be applied separately if Phase
    # 3 ever reverts to host-port publishing.
    op.create_index(
        "ix_managed_containers_host_port",
        "managed_containers",
        ["host_port"],
        unique=False,
    )
