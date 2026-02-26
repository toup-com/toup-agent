"""VPS plans and instances tables

Revision ID: 20260218_0014
Revises: 20260218_0013
Create Date: 2026-02-18

Adds:
- vps_plans: Static plan catalog (Starter / Standard / Pro)
- vps_instances: Provisioned EC2 instances per user
"""
from alembic import op
import sqlalchemy as sa

revision = "014_vps_instances"
down_revision = "013_password_plain"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── vps_plans ──────────────────────────────────────────────────
    op.create_table(
        "vps_plans",
        sa.Column("id", sa.String(20), primary_key=True),
        sa.Column("name", sa.String(50), nullable=False),
        sa.Column("instance_type", sa.String(20), nullable=False),
        sa.Column("vcpu", sa.Integer, nullable=False),
        sa.Column("ram_gb", sa.Integer, nullable=False),
        sa.Column("storage_gb", sa.Integer, nullable=False),
        sa.Column("price_cents", sa.Integer, nullable=False),
        sa.Column("stripe_price_id", sa.String(100), nullable=False, server_default=""),
    )

    # Seed plans
    op.execute("""
        INSERT INTO vps_plans (id, name, instance_type, vcpu, ram_gb, storage_gb, price_cents, stripe_price_id)
        VALUES
            ('starter',  'Starter',  't3.small',  2, 2,  20, 1000, ''),
            ('standard', 'Standard', 't3.medium', 2, 4,  40, 2000, ''),
            ('pro',      'Pro',      't3.large',  2, 8,  80, 4000, '')
    """)

    # ── vps_instances ──────────────────────────────────────────────
    op.create_table(
        "vps_instances",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("plan_id", sa.String(20), sa.ForeignKey("vps_plans.id"), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("aws_instance_id", sa.String(50), nullable=True),
        sa.Column("aws_region", sa.String(20), nullable=False, server_default="us-east-1"),
        sa.Column("public_ip", sa.String(45), nullable=True),
        sa.Column("public_dns", sa.String(255), nullable=True),
        sa.Column("ami_id", sa.String(50), nullable=False, server_default=""),
        sa.Column("ssh_password", sa.String(64), nullable=True),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("stripe_session_id", sa.String(100), nullable=True, unique=True),
        sa.Column("stripe_subscription_id", sa.String(100), nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("provisioned_at", sa.DateTime, nullable=True),
        sa.Column("terminated_at", sa.DateTime, nullable=True),
    )

    op.create_index("ix_vps_instances_user_id", "vps_instances", ["user_id"])
    op.create_index("ix_vps_instances_status", "vps_instances", ["status"])
    op.create_index("ix_vps_instances_stripe_session", "vps_instances", ["stripe_session_id"])


def downgrade() -> None:
    op.drop_index("ix_vps_instances_stripe_session", "vps_instances")
    op.drop_index("ix_vps_instances_status", "vps_instances")
    op.drop_index("ix_vps_instances_user_id", "vps_instances")
    op.drop_table("vps_instances")
    op.drop_table("vps_plans")
