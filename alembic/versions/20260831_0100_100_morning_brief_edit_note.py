"""100 — tell the thread what 099 changed (R43).

099 rewrote the installed "Morning work brief" into the seven-step
shape and dropped GitHub, Teams and Outlook with it. Nobody was told.
The next weekday morning the brief simply stopped mentioning three
accounts the user had connected on purpose, and the only record of the
change was a `workflow_rev` the app uses to invalidate drafts.

So this appends the sentence 099 owed, into the automation's own
thread, where every other edit to that automation is already written
down.

TENANT DBs — same guard as 099, for the same reason: `automations`,
`automation_threads` and `automation_turns` are AGENT_ONLY, so this is
a no-op on the platform DB and on a tenant that has not run `init_db`
yet. `alembic upgrade head` runs on boot in both images, which is what
carries it to every account.

TWO ROWS, not one, and that is the ledger's grammar rather than a
choice. A `note` turn is the centred caps stamp and carries no prose —
`ledger.validate_turn_payload` keeps `stamp`/`at`/`writes_count` and
nothing else — so the divider the app draws as "EDITED · <date>" and
the sentence under it are a note turn and an `agent` turn, in that
order. That is exactly what `workflow._edited_note` writes for the
divider, and what every writer in `workflow.py` writes for the
sentence; anything else would persist a row the app cannot render.

WHO GETS IT. The population 099 considered (the template slug, not
deleted), narrowed to rows that are:

  - the seven-step shape, by step id and tool — the same identity test
    099 used against the old shape, so an automation somebody has since
    edited (a re-added account is an eighth step) is left alone;
  - already edited at least once. 099's UPDATE bumped `workflow_rev` on
    every row it touched and creation leaves it at 0, so a brief
    ADOPTED from the R42 catalog — already seven steps, and which never
    read GitHub, Teams or Outlook — is not told it stopped reading
    them. The sentence has to be true for whoever reads it;
  - carrying a thread. There is no thread to write into otherwise, and
    this migration does not create one: a thread minted here would be
    an empty screen whose only content is a note about a change the
    user cannot see the other half of.

IDEMPOTENT by the sentence itself. The note grammar has no field for a
marker — an unknown key is dropped by the validator — so the marker is
a fragment of the copy, matched with LIKE before anything is written.

`workflow_rev` is deliberately NOT bumped: nothing about the workflow
changed here, and a bump invalidates a draft a device is holding.

Revision ID: 100
Revises: 099
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime

import sqlalchemy as sa
from alembic import op

revision = "100"
down_revision = "099"
branch_labels = None
depends_on = None

logger = logging.getLogger("alembic.runtime.migration")

_AUTOMATIONS = "automations"
_THREADS = "automation_threads"
_TURNS = "automation_turns"
_SLUG = "morning-work-brief"

#: 099's new shape, by step id and tool — `rank` is the agent step and
#: calls no connector. Identity, not a heuristic: a spec that does not
#: match this is one somebody has changed since, and what it reads is no
#: longer a thing this file can describe.
_RESPEC_STEPS = (
    ("cal", "calendar__list_events"),
    ("mail", "gmail__list_messages"),
    ("waiting", "gmail__list_messages"),
    ("rooms", "slack__search_messages"),
    ("board", "jira__search_issues"),
    ("rank", None),
    ("post", "slack__send_message"),
)

#: Passes `copy_guard.scan` — which is why the screen is named as the
#: proper noun the canvas header uses ("workflow" in lower case is a
#: banned word, and "Workflow" is the sanctioned exact string).
_NOTE_TEXT = (
    "I now read your calendar, your mail, what has been waiting on you, "
    "Slack when someone names you, and your Jira board — and I "
    "stopped reading GitHub, Teams and Outlook. Add any of them back "
    "from the Workflow."
)

#: The idempotency marker: ASCII, so it survives `json.dumps`'s escaping
#: byte for byte inside `payload_json` and can be matched with LIKE.
_MARKER = "I stopped reading GitHub, Teams and Outlook"


def _is_respec_shape(spec: object) -> bool:
    """True when this spec is exactly what 099 writes.

    Pure, so the whole decision is readable in one place and testable
    without a database.
    """
    if not isinstance(spec, dict) or spec.get("version") != 2:
        return False
    steps = [s for s in spec.get("steps") or [] if isinstance(s, dict)]
    shape = tuple(
        (s.get("id"), None if s.get("kind") == "agent" else s.get("tool"))
        for s in steps
    )
    return shape == _RESPEC_STEPS


def _insert_turn(conn, *, turn_id: str, thread_id: str, seq: int,
                 kind: str, payload: dict, now: datetime) -> None:
    stmt = sa.text(
        f"INSERT INTO {_TURNS} "
        "(id, thread_id, run_id, seq, kind, payload_json, created_at) "
        "VALUES (:id, :thread_id, NULL, :seq, :kind, :payload, :created_at)"
    ).bindparams(sa.bindparam("created_at", type_=sa.DateTime()))
    conn.execute(stmt, {
        "id": turn_id,
        "thread_id": thread_id,
        "seq": seq,
        "kind": kind,
        # `ledger.append_turn`'s own encoding, so a turn written here and
        # a turn written by the app are the same bytes.
        "payload": json.dumps(payload, default=str),
        "created_at": now,
    })


def upgrade() -> None:
    conn = op.get_bind()
    have = set(sa.inspect(conn).get_table_names())
    if not {_AUTOMATIONS, _THREADS, _TURNS} <= have:
        logger.info("[alembic.100] no automation ledger here (platform DB, "
                    "or a tenant before its first init_db) — nothing to do")
        return

    rows = conn.execute(sa.text(
        f"SELECT a.id AS automation_id, a.spec_json AS spec_json, "
        f"t.id AS thread_id "
        f"FROM {_AUTOMATIONS} a JOIN {_THREADS} t "
        f"ON t.automation_id = a.id "
        "WHERE a.template_slug = :slug AND a.deleted_at IS NULL "
        "AND a.workflow_rev > 0"
    ), {"slug": _SLUG}).fetchall()

    noted = skipped = 0
    for row in rows:
        try:
            spec = json.loads(row.spec_json or "{}")
        except (ValueError, TypeError):
            spec = None
        if not _is_respec_shape(spec):
            logger.info("[alembic.100] %s: not the shape 099 writes — "
                        "left alone", row.automation_id)
            skipped += 1
            continue
        already = conn.execute(sa.text(
            f"SELECT 1 FROM {_TURNS} WHERE thread_id = :tid "
            "AND kind = 'agent' AND payload_json LIKE :marker LIMIT 1"
        ), {"tid": row.thread_id, "marker": f"%{_MARKER}%"}).first()
        if already is not None:
            skipped += 1
            continue

        seq = conn.execute(sa.text(
            f"SELECT COALESCE(MAX(seq), 0) FROM {_TURNS} "
            "WHERE thread_id = :tid"
        ), {"tid": row.thread_id}).scalar() or 0
        now = datetime.utcnow()
        _insert_turn(
            conn, turn_id=str(uuid.uuid4()), thread_id=row.thread_id,
            seq=seq + 1, kind="note", now=now,
            payload={"stamp": "edited", "at": now.isoformat() + "Z",
                     "writes_count": 0},
        )
        _insert_turn(
            conn, turn_id=str(uuid.uuid4()), thread_id=row.thread_id,
            seq=seq + 2, kind="agent", now=now,
            payload={"text": _NOTE_TEXT},
        )
        noted += 1

    logger.info("[alembic.100] morning brief: %d told, %d left alone",
                noted, skipped)


def downgrade() -> None:
    """One-way.

    Deleting the note would take back something the user has read, and
    the change it describes is not being undone.
    """
    logger.info("[alembic.100] no downgrade: the note describes a change "
                "099 already made")
