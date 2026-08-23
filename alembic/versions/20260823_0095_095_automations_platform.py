"""095 — automations: platform-side grants + templates (Round 26).

Two PLATFORM_ONLY tables for the chat-built automations engine:

  - ``automation_grants``    — standing write permissions (action +
                               pinned target + cadence). Enforced by the
                               connector dispatcher at call time; lives
                               next to the tokens it gates.
  - ``automation_templates`` — product-curated "Suggested" templates,
                               seeded here with the Jira→Slack demo.

The engine's per-user tables (automations, bindings, events, outbox,
auth sessions) are AGENT_ONLY and arrive via init_db create_all on
agent boot — new tables need no alembic mirror.

Gated on ``settings.run_mode`` like revisions 078-095's predecessors:
``alembic upgrade head`` runs on boot in BOTH images, and a
``CREATE TABLE … REFERENCES users`` takes a lock on ``users`` that a
tenant DB mid blue/green must never wait on. Tenants never need these
tables at all.

Everything is dark behind the ``automations`` feature flag; creating
the tables changes no behavior.
"""
from __future__ import annotations

import logging

import sqlalchemy as sa
from alembic import op

revision = "095"
down_revision = "094"
branch_labels = None
depends_on = None

logger = logging.getLogger("alembic.095")

_GRANTS = "automation_grants"
_TEMPLATES = "automation_templates"

# The Jira→Slack demo template. Spec shape matches
# app/agent/automations/validator.py; params_template values use
# {{event.*}} placeholders resolved by the executor's prepare step.
_JIRA_TO_SLACK_SPEC = """\
{
  "name": "Jira \\u2192 Slack",
  "trigger": {
    "mode": "poll",
    "connector_id": "jira",
    "event": "issue_created",
    "poll_interval_s": 300,
    "filter": {}
  },
  "action": {
    "connector_id": "slack",
    "tool": "slack__send_message",
    "params_template": {
      "channel": "{{grant.target.id}}",
      "text": "New Jira issue {{event.key}}: {{event.summary}} ({{event.url}})"
    }
  },
  "dedupe_key": "event.key",
  "mode": "auto"
}
"""


def _is_platform_db() -> bool:
    try:
        from app.config import settings
        return (settings.run_mode or "").strip().lower() in ("platform", "monolith")
    except Exception:
        logger.warning("[alembic.095] run_mode unreadable; skipping to stay safe")
        return False


def upgrade() -> None:
    if not _is_platform_db():
        logger.info(
            "[alembic.095] not a platform DB; skipping (%s/%s are PLATFORM_ONLY)",
            _GRANTS, _TEMPLATES,
        )
        return

    conn = op.get_bind()
    insp = sa.inspect(conn)
    insp.clear_cache()
    tables = set(insp.get_table_names())
    if "users" not in tables:
        logger.info("[alembic.095] users absent; skipping")
        return

    if _GRANTS not in tables:
        op.create_table(
            _GRANTS,
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column(
                "user_id",
                sa.String(36),
                sa.ForeignKey("users.id", ondelete="CASCADE"),
                nullable=False,
            ),
            # Agent-side automations.id — cross-database soft pointer.
            sa.Column("automation_id", sa.String(36), nullable=True),
            sa.Column("connector_id", sa.String(64), nullable=False),
            sa.Column("tool_name", sa.String(128), nullable=False),
            sa.Column("target_json", sa.Text(), nullable=False),
            sa.Column("cadence_json", sa.Text(), nullable=True),
            sa.Column(
                "mode", sa.String(16), nullable=False, server_default="confirm",
            ),
            sa.Column("summary", sa.String(300), nullable=False),
            sa.Column("preview_json", sa.Text(), nullable=True),
            sa.Column(
                "status", sa.String(16), nullable=False, server_default="pending",
            ),
            sa.Column("uses_day_key", sa.String(10), nullable=True),
            sa.Column(
                "uses_today", sa.Integer(), nullable=False, server_default="0",
            ),
            sa.Column("uses_hour_key", sa.String(13), nullable=True),
            sa.Column(
                "uses_this_hour", sa.Integer(), nullable=False, server_default="0",
            ),
            sa.Column("last_used_at", sa.DateTime(), nullable=True),
            sa.Column(
                "created_at", sa.DateTime(), nullable=False,
                server_default=sa.func.now(),
            ),
            sa.Column("expires_at", sa.DateTime(), nullable=False),
            sa.Column("decided_at", sa.DateTime(), nullable=True),
            sa.Column("decided_via", sa.String(32), nullable=True),
            sa.Column("revoked_at", sa.DateTime(), nullable=True),
            sa.CheckConstraint(
                "status IN ('pending', 'approved', 'rejected', 'expired', "
                "'revoked')",
                name="ck_automation_grants_status",
            ),
        )
        op.create_index(
            "ix_automation_grants_user_id", _GRANTS, ["user_id"],
        )
        op.create_index(
            "ix_automation_grants_automation_id", _GRANTS, ["automation_id"],
        )
        op.create_index(
            "ix_automation_grants_status", _GRANTS, ["status"],
        )
        op.create_index(
            "ix_automation_grants_user_status", _GRANTS, ["user_id", "status"],
        )
    else:
        logger.info("[alembic.095] %s already present", _GRANTS)

    if _TEMPLATES not in tables:
        op.create_table(
            _TEMPLATES,
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column("slug", sa.String(64), nullable=False),
            sa.Column("name", sa.String(120), nullable=False),
            sa.Column("description", sa.Text(), nullable=True),
            sa.Column("icon", sa.String(64), nullable=True),
            sa.Column(
                "connectors_json", sa.Text(), nullable=False,
                server_default="[]",
            ),
            sa.Column("spec_json", sa.Text(), nullable=False),
            sa.Column(
                "enabled", sa.Boolean(), nullable=False, server_default="true",
            ),
            sa.Column(
                "sort_order", sa.Integer(), nullable=False, server_default="0",
            ),
            sa.Column(
                "created_at", sa.DateTime(), nullable=False,
                server_default=sa.func.now(),
            ),
            sa.Column(
                "updated_at", sa.DateTime(), nullable=False,
                server_default=sa.func.now(),
            ),
            sa.UniqueConstraint("slug", name="uq_automation_templates_slug"),
        )
    else:
        logger.info("[alembic.095] %s already present", _TEMPLATES)

    # Seed the Jira→Slack demo template. Idempotent on the slug UNIQUE:
    # a re-run (or a boot after create_all already made the table) skips.
    insp.clear_cache()
    if _TEMPLATES in set(insp.get_table_names()):
        existing = conn.execute(
            sa.text(f"SELECT 1 FROM {_TEMPLATES} WHERE slug = :slug"),
            {"slug": "jira-to-slack"},
        ).first()
        if existing is None:
            import uuid as _uuid
            conn.execute(
                sa.text(
                    f"INSERT INTO {_TEMPLATES} "
                    "(id, slug, name, description, icon, connectors_json, "
                    " spec_json, enabled, sort_order) "
                    "VALUES (:id, :slug, :name, :description, :icon, "
                    " :connectors, :spec, :enabled, :sort_order)"
                ),
                {
                    "id": str(_uuid.uuid4()),
                    "slug": "jira-to-slack",
                    "name": "Jira → Slack",
                    "description": (
                        "Post a Slack message to a channel you pick "
                        "whenever a new Jira issue appears."
                    ),
                    "icon": "jira",
                    "connectors": '["jira", "slack"]',
                    "spec": _JIRA_TO_SLACK_SPEC,
                    "enabled": True,
                    "sort_order": 0,
                },
            )
            logger.info("[alembic.095] seeded template jira-to-slack")


def downgrade() -> None:
    if not _is_platform_db():
        return
    conn = op.get_bind()
    insp = sa.inspect(conn)
    tables = set(insp.get_table_names())
    if _TEMPLATES in tables:
        op.drop_table(_TEMPLATES)
    if _GRANTS in tables:
        op.drop_table(_GRANTS)
