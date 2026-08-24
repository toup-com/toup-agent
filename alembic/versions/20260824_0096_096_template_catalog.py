"""096 — automation_templates: category + variables (Round 28).

Two columns for the server-driven template catalog:

  - ``category``       — work/email/code/calendar/school/personal
                         (code enum ``TEMPLATE_CATEGORIES``); default
                         'work' keeps the 095-seeded row valid.
  - ``variables_json`` — declared template variables the setup agent
                         asks the user for.

No row seeding here: the catalog itself is code
(``app/services/automation_template_catalog.py``) and upserts by slug
on platform boot — a data migration would fossilize ~28 templates in
an alembic file that can never be edited.

Platform DB only, same guard as 095 (the two automation platform
tables are PLATFORM_ONLY; tenant DBs must not grow them).

Revision ID: 096
Revises: 095
"""

from __future__ import annotations

import logging

import sqlalchemy as sa
from alembic import op

revision = "096"
down_revision = "095"
branch_labels = None
depends_on = None

logger = logging.getLogger("alembic.runtime.migration")

_TEMPLATES = "automation_templates"


def _is_platform_db() -> bool:
    try:
        from app.config import settings
        return (settings.run_mode or "").strip().lower() in ("platform", "monolith")
    except Exception:
        logger.warning("[alembic.096] run_mode unreadable; skipping to stay safe")
        return False


def upgrade() -> None:
    if not _is_platform_db():
        logger.info("[alembic.096] not a platform DB; skipping "
                    "(%s is PLATFORM_ONLY)", _TEMPLATES)
        return
    conn = op.get_bind()
    insp = sa.inspect(conn)
    if _TEMPLATES not in set(insp.get_table_names()):
        # create_all made (or will make) the table with both columns
        # already on the model — nothing to alter.
        logger.info("[alembic.096] %s absent; model create_all covers it",
                    _TEMPLATES)
        return
    cols = {c["name"] for c in insp.get_columns(_TEMPLATES)}
    if "category" not in cols:
        op.add_column(
            _TEMPLATES,
            sa.Column("category", sa.String(32), nullable=False,
                      server_default="work"),
        )
    if "variables_json" not in cols:
        op.add_column(
            _TEMPLATES,
            sa.Column("variables_json", sa.Text(), nullable=False,
                      server_default="[]"),
        )


def downgrade() -> None:
    if not _is_platform_db():
        return
    conn = op.get_bind()
    insp = sa.inspect(conn)
    if _TEMPLATES not in set(insp.get_table_names()):
        return
    cols = {c["name"] for c in insp.get_columns(_TEMPLATES)}
    if "variables_json" in cols:
        op.drop_column(_TEMPLATES, "variables_json")
    if "category" in cols:
        op.drop_column(_TEMPLATES, "category")
