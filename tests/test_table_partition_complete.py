"""Every declared table must have a home — and FKs must not cross lanes.

Pinned incident (2026-08-25, R29-D): `automation_facts` (the v2 engine's
facts ledger, FK → automations) was declared in models but listed in NO
partition set. An unlisted table is excluded from neither lane, so the
PLATFORM create_all tried to build it — and its FK target `automations`
is AGENT_ONLY, absent from the platform DB. Postgres refused
(UndefinedTableError), platform boot degraded, and BOTH Railway deploys
of that evening FAILED while CI stayed green: sqlite happily creates an
FK to a missing table, so the platform-lane sweep could not see it.
The cage was kinder than production; these checks equalize it — they
are pure metadata introspection and fail on sqlite CI exactly where
postgres boot would have failed in prod.
"""
from __future__ import annotations

import os

os.environ.setdefault("AGENT_API_KEY", "test-key-partition")
os.environ.setdefault("USER_ID", "00000000-0000-0000-0000-0000000000aa")


def _lists():
    from app.db.models.base import (
        AGENT_ONLY_TABLES, PLATFORM_ONLY_TABLES, SHARED_TABLES,
    )
    return AGENT_ONLY_TABLES, PLATFORM_ONLY_TABLES, SHARED_TABLES


def _declared():
    import app.db.models  # noqa: F401 — force every model module to import
    from app.db.models.base import Base
    return Base.metadata.tables


def test_every_declared_table_is_in_exactly_one_partition_list():
    agent, platform, shared = _lists()
    declared = set(_declared().keys())
    listed = agent | platform | shared

    unlisted = declared - listed
    assert not unlisted, (
        f"Tables declared in models but in NO partition list: "
        f"{sorted(unlisted)}. An unlisted table is created on BOTH "
        "lanes — if it references a lane-only table, the other lane's "
        "postgres boot dies (the automation_facts deploy failure, "
        "2026-08-25). Add each to exactly one of AGENT_ONLY_TABLES / "
        "PLATFORM_ONLY_TABLES / SHARED_TABLES in app/db/models/base.py."
    )

    ghost = listed - declared
    assert not ghost, (
        f"Partition lists name tables no model declares: {sorted(ghost)} "
        "— a rename or deletion left the list stale."
    )

    overlaps = (agent & platform) | (agent & shared) | (platform & shared)
    assert not overlaps, (
        f"Tables in MORE than one partition list: {sorted(overlaps)}"
    )


def test_no_foreign_key_crosses_out_of_its_lane():
    """A table each lane creates must be able to satisfy its FKs there.

    - a PLATFORM-created table (platform-only or shared) must not
      reference an AGENT_ONLY table;
    - an AGENT-created table (agent-only or shared) must not reference
      a PLATFORM_ONLY table.
    Either way one lane's create_all emits an FK whose target does not
    exist on that lane's database — sqlite shrugs, postgres refuses.
    """
    agent, platform, shared = _lists()
    tables = _declared()

    violations: list[str] = []
    for tname, table in tables.items():
        targets = {
            fk.column.table.name
            for fk in table.foreign_keys
        }
        # An UNLISTED table is excluded from neither lane — create_all
        # builds it on BOTH. Treat it that way here so this check fails
        # on the incident even in isolation from the completeness test.
        unlisted = (
            tname not in agent and tname not in platform
            and tname not in shared
        )
        created_on_platform = tname in platform or tname in shared or unlisted
        created_on_agent = tname in agent or tname in shared or unlisted
        if created_on_platform:
            bad = targets & agent
            if bad:
                violations.append(
                    f"{tname} (platform-created) → agent-only {sorted(bad)}"
                )
        if created_on_agent:
            bad = targets & platform
            if bad:
                violations.append(
                    f"{tname} (agent-created) → platform-only {sorted(bad)}"
                )
    assert not violations, (
        "Cross-lane foreign keys (each kills the other lane's postgres "
        "create_all): " + "; ".join(violations)
    )
