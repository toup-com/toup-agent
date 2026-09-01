"""R43 — migration 100's note, and putting back what 099 took out.

099 re-spec'd the installed "Morning work brief" and dropped GitHub,
Teams and Outlook without telling anyone, and `add_connector` could
only re-add a connector the automation's TEMPLATE still declared — so
those three were gone for good. This file pins both halves of the
repair.

PLATFORM lane, and pure by construction. The migration runs against a
throwaway sqlite database it builds itself (its SQL is portable), and
`add_connector` is driven with its two seams — the row loader and the
persist — replaced, so the whole spec-level decision is exercised
against the REAL manifest-built registry with no automations table
anywhere. That is deliberate: `automations` is AGENT_ONLY, and a test
that needed it would have to move to the agent-mode lane to prove
something neither the database nor the lane has an opinion about.

Run: python3 -m pytest tests/test_r43_respec_note.py -q
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

from app.agent.automations import service
from app.agent.automations.copy_guard import scan
from app.agent.automations.ledger import validate_turn_payload
from app.agent.automations.spec import SpecError, validate_spec
from app.agent.automations.spec_v2 import MAX_STEPS
from app.db.models import Automation
from app.services.automation_template_catalog import CATALOG
from app.services.connector_registry import ConnectorRegistry

_BACKEND = Path(__file__).resolve().parents[1]
_MIGRATION = (
    _BACKEND / "alembic" / "versions"
    / "20260831_0100_100_morning_brief_edit_note.py"
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mig():
    return _load("mig100", _MIGRATION)


@pytest.fixture(scope="module")
def registry():
    r = ConnectorRegistry()
    r.load_all(include_experimental=True)
    return {e["connector_id"]: e for e in r.automation_registry()}


@pytest.fixture(scope="module")
def brief_spec() -> dict:
    entry = next(e for e in CATALOG if e["slug"] == "morning-work-brief")
    return json.loads(json.dumps(entry["spec"]))


# ── the note itself ───────────────────────────────────────────────────

def test_note_passes_the_copy_contract(mig):
    """It is authored copy in a user's thread, so the guard binds it —
    and "workflow" in lower case is a banned word, which is why the
    sentence names the screen the way the canvas header does."""
    assert scan(mig._NOTE_TEXT) == []


def test_note_is_a_turn_the_app_can_render(mig):
    """The migration writes rows by hand. If the ledger's own validator
    would change either payload, the app is being handed something no
    writer in the platform produces."""
    now = "2026-08-31T08:00:00Z"
    assert validate_turn_payload("agent", {"text": mig._NOTE_TEXT}) == {
        "text": mig._NOTE_TEXT,
    }
    assert validate_turn_payload("note", {
        "stamp": "edited", "at": now, "writes_count": 0,
    }) == {"stamp": "edited", "at": now, "writes_count": 0}


def test_marker_survives_the_encoding_it_is_matched_through(mig):
    """The idempotency check is a LIKE over `payload_json`, so the
    marker has to be there after `json.dumps` has escaped everything it
    escapes."""
    assert mig._MARKER in mig._NOTE_TEXT
    assert mig._MARKER in json.dumps({"text": mig._NOTE_TEXT}, default=str)


# ── the identity gate ─────────────────────────────────────────────────

def test_shape_gate_accepts_what_099_writes(mig, brief_spec):
    """The catalog is the live copy of the same spec; 099 froze a
    point-in-time one. A gate that does not recognise both is a gate
    that misses either the migrated rows or the adopted ones."""
    assert mig._is_respec_shape(brief_spec) is True

    frozen = _load("mig099", _BACKEND / "alembic" / "versions"
                   / "20260831_0099_099_morning_brief_respec.py")
    assert mig._is_respec_shape(frozen._NEW_SPEC) is True


def test_shape_gate_refuses_anything_else(mig, brief_spec):
    assert mig._is_respec_shape(None) is False
    assert mig._is_respec_shape({"version": 1, "steps": []}) is False

    # The pre-099 shape — the rows 099 itself was looking for.
    old = {"version": 2, "steps": [
        {"id": sid, "tool": tool} for sid, tool in (
            ("issues", "jira__search_issues"),
            ("repo", "github__list_issues"),
            ("chat", "teams__read_chat_messages"),
            ("mail", "gmail__list_messages"),
            ("outlook", "outlook__list_messages"),
            ("post", "slack__send_message"),
        )
    ]}
    assert mig._is_respec_shape(old) is False

    # An account put back is an eighth step — and this file can no
    # longer describe what that automation reads.
    edited = json.loads(json.dumps(brief_spec))
    edited["steps"].insert(5, {"id": "github", "connector_id": "github",
                               "tool": "github__list_issues", "params": {}})
    assert mig._is_respec_shape(edited) is False

    # The agent step keeps its id and stops being an agent step.
    swapped = json.loads(json.dumps(brief_spec))
    swapped["steps"][5] = {"id": "rank", "connector_id": "slack",
                           "tool": "slack__search_messages", "params": {}}
    assert mig._is_respec_shape(swapped) is False


# ── the migration, on a database ──────────────────────────────────────

_DDL = """
CREATE TABLE automations (
  id VARCHAR(36) PRIMARY KEY,
  spec_json TEXT NOT NULL,
  template_slug VARCHAR(64),
  workflow_rev INTEGER NOT NULL DEFAULT 0,
  deleted_at TIMESTAMP
);
CREATE TABLE automation_threads (
  id VARCHAR(36) PRIMARY KEY,
  automation_id VARCHAR(36) NOT NULL,
  archived_at TIMESTAMP,
  created_at TIMESTAMP
);
CREATE TABLE automation_turns (
  id VARCHAR(36) PRIMARY KEY,
  thread_id VARCHAR(36) NOT NULL,
  run_id VARCHAR(36),
  seq INTEGER NOT NULL,
  kind VARCHAR(16) NOT NULL,
  payload_json TEXT NOT NULL,
  created_at TIMESTAMP NOT NULL,
  UNIQUE (thread_id, seq)
);
"""


def _run(conn, fn) -> None:
    """Execute the migration exactly as `alembic upgrade` does — with
    the global `op` proxy bound to this connection."""
    with Operations.context(MigrationContext.configure(conn)):
        fn()


def _seed(conn, aid: str, spec: dict, *, rev: int = 1,
          slug: str = "morning-work-brief", thread: bool = True) -> None:
    conn.execute(sa.text(
        "INSERT INTO automations (id, spec_json, template_slug, "
        "workflow_rev, deleted_at) VALUES (:id, :spec, :slug, :rev, NULL)"
    ), {"id": aid, "spec": json.dumps(spec), "slug": slug, "rev": rev})
    if thread:
        conn.execute(sa.text(
            "INSERT INTO automation_threads (id, automation_id) "
            "VALUES (:tid, :aid)"
        ), {"tid": f"t-{aid}", "aid": aid})


def _turns(conn, aid: str) -> list[tuple[str, str]]:
    rows = conn.execute(sa.text(
        "SELECT kind, payload_json FROM automation_turns "
        "WHERE thread_id = :tid ORDER BY seq"
    ), {"tid": f"t-{aid}"}).fetchall()
    return [(r[0], r[1]) for r in rows]


@pytest.fixture()
def db():
    engine = sa.create_engine("sqlite://")
    with engine.begin() as conn:
        for stmt in _DDL.strip().split(";"):
            if stmt.strip():
                conn.execute(sa.text(stmt))
        yield conn
    engine.dispose()


def test_no_ledger_here_is_a_no_op(mig):
    """The platform DB, and a tenant before its first init_db."""
    engine = sa.create_engine("sqlite://")
    with engine.begin() as conn:
        _run(conn, mig.upgrade)
    engine.dispose()


def test_a_matching_row_is_told_what_changed(mig, db, brief_spec):
    _seed(db, "a1", brief_spec)
    _run(db, mig.upgrade)

    turns = _turns(db, "a1")
    assert [k for k, _ in turns] == ["note", "agent"], (
        "the divider and the sentence are two rows because the note "
        "grammar has no field for prose"
    )
    note = json.loads(turns[0][1])
    assert note["stamp"] == "edited" and note["at"]
    assert json.loads(turns[1][1])["text"] == mig._NOTE_TEXT


def test_an_edited_row_is_left_alone(mig, db, brief_spec):
    edited = json.loads(json.dumps(brief_spec))
    edited["steps"].insert(5, {"id": "gh", "connector_id": "github",
                               "tool": "github__list_issues", "params": {}})
    _seed(db, "a2", edited)
    _run(db, mig.upgrade)
    assert _turns(db, "a2") == []


def test_a_freshly_adopted_brief_is_not_told_it_lost_accounts(
    mig, db, brief_spec,
):
    """`workflow_rev` is 0 at creation and 099 bumped every row it
    touched, so an untouched adoption of the R42 catalog — which never
    read GitHub, Teams or Outlook — must not be told it stopped."""
    _seed(db, "a3", brief_spec, rev=0)
    _run(db, mig.upgrade)
    assert _turns(db, "a3") == []


def test_a_row_with_no_thread_gets_no_thread(mig, db, brief_spec):
    _seed(db, "a4", brief_spec, thread=False)
    _run(db, mig.upgrade)
    assert db.execute(sa.text(
        "SELECT COUNT(*) FROM automation_threads"
    )).scalar() == 0


def test_another_automation_is_not_touched(mig, db, brief_spec):
    _seed(db, "a5", brief_spec, slug="jira-to-slack")
    _run(db, mig.upgrade)
    assert _turns(db, "a5") == []


def test_a_second_run_appends_nothing(mig, db, brief_spec):
    _seed(db, "a6", brief_spec)
    _run(db, mig.upgrade)
    first = _turns(db, "a6")
    _run(db, mig.upgrade)
    assert _turns(db, "a6") == first


def test_the_note_lands_after_the_thread_it_joins(mig, db, brief_spec):
    """`seq` is unique per thread and the app renders in that order, so
    a note minted at 1 would both collide and read as the beginning of
    the conversation."""
    _seed(db, "a7", brief_spec)
    for seq in (1, 2):
        db.execute(sa.text(
            "INSERT INTO automation_turns (id, thread_id, seq, kind, "
            "payload_json, created_at) VALUES (:id, 't-a7', :seq, 'agent', "
            "'{\"text\": \"earlier\"}', '2026-08-01 08:00:00')"
        ), {"id": f"old{seq}", "seq": seq})
    _run(db, mig.upgrade)
    seqs = db.execute(sa.text(
        "SELECT seq FROM automation_turns WHERE thread_id = 't-a7' "
        "ORDER BY seq"
    )).scalars().all()
    assert seqs == [1, 2, 3, 4]


# ── putting the accounts back ─────────────────────────────────────────

class _Persisted(Exception):
    """Carries the spec `add_connector` handed to the persist seam."""

    def __init__(self, spec: dict):
        self.spec = spec
        super().__init__("persisted")


@pytest.fixture()
def added(monkeypatch, registry, brief_spec):
    """`add_connector(connector_id) -> spec`, with the row loader and
    the persist replaced and everything between them real."""
    async def _templates(_user_id):
        # The R42 catalog: the flagship declares nothing for the three
        # connectors 099 dropped, which is the whole difficulty.
        return [{"slug": e["slug"], "spec": e["spec"]} for e in CATALOG]

    async def _registry(_user_id, **_kw):
        return registry

    async def _update(_db, *, automation_id, user_id, spec):
        raise _Persisted(spec)

    monkeypatch.setattr(service.reg, "fetch_templates", _templates)
    monkeypatch.setattr(service.reg, "fetch_registry", _registry)
    monkeypatch.setattr(service, "update_automation", _update)

    def _add(connector_id: str, spec: dict | None = None) -> dict:
        row = Automation(
            id="a1", user_id="u1", name="Morning work brief",
            spec_json=json.dumps(spec if spec is not None else brief_spec),
            trigger_mode="schedule", template_slug="morning-work-brief",
        )

        async def _load(_db, _aid, _uid, **_kw):
            return row

        monkeypatch.setattr(service, "_load_owned", _load)
        import asyncio
        try:
            asyncio.run(service.add_connector(
                db=None, automation_id="a1", user_id="u1",
                connector_id=connector_id,
            ))
        except _Persisted as persisted:
            return persisted.spec
        raise AssertionError("add_connector persisted nothing")

    return _add


@pytest.mark.parametrize("connector_id,tool", [
    ("github", "github__list_issues"),
    ("teams", "teams__read_chat_messages"),
    ("outlook", "outlook__list_messages"),
])
def test_the_three_accounts_099_dropped_go_back_on(
    added, registry, connector_id, tool,
):
    """None of the three is in the template any more, so every one of
    them used to raise `no_template_step` — the picker offered an
    account it could not write."""
    spec = added(connector_id)
    step = next(s for s in spec["steps"]
                if s.get("connector_id") == connector_id)
    assert step["tool"] == tool
    assert step["on_error"] == "continue"
    assert not step.get("grant_id"), "membership adds READS only"
    validate_spec(spec, registry, template_mode=True)


def test_the_read_lands_ahead_of_the_write(added, brief_spec):
    """Both ways of knowing a step writes: the grant an installed
    automation carries, and — for a draft that has none yet — the
    registry the validator itself asks."""
    ids = [s["id"] for s in added("outlook")["steps"]]
    assert ids.index("outlook") < ids.index("post")

    installed = json.loads(json.dumps(brief_spec))
    installed["steps"][-1]["grant_id"] = "g1"
    ids = [s["id"] for s in added("outlook", installed)["steps"]]
    assert ids.index("outlook") < ids.index("post")


def test_the_target_a_pin_fills_is_left_empty(added):
    """`params_required` names a place only the user can choose, and
    `executor_v2._apply_focus_scope` fills exactly those from a pin.
    A default here would read a repository nobody asked for."""
    step = next(s for s in added("github")["steps"]
                if s.get("connector_id") == "github")
    assert "owner" not in step["params"] and "repo" not in step["params"]
    assert step["params"] == {"state": "open", "per_page": 50}


def test_the_collect_line_carries_no_raw_identifier(added):
    """An event's fields exist to identify a row for dedupe; the line
    this builds is read by a person and by the ranking step."""
    for connector_id in ("teams", "outlook"):
        step = next(s for s in added(connector_id)["steps"]
                    if s.get("connector_id") == connector_id)
        fields = step["collect"]["fields"]
        assert "id" not in fields
        assert fields, "a collect with no fields collects nothing"
        assert step["collect"]["items_path"]


def test_an_account_already_on_it_is_refused(added):
    with pytest.raises(service.MembershipError) as e:
        added("gmail")
    assert e.value.code == "already_member"


def test_a_full_automation_refuses_in_words(added, brief_spec):
    """The validator would refuse too — as a `SpecError` the membership
    routes do not catch, which is a 500 for a user whose automation is
    simply full."""
    full = json.loads(json.dumps(brief_spec))
    filler = dict(full["steps"][0])
    while len(full["steps"]) < MAX_STEPS:
        filler = dict(filler)
        filler["id"] = f"cal{len(full['steps'])}"
        full["steps"].insert(1, filler)
    with pytest.raises(service.MembershipError) as e:
        added("github", full)
    assert e.value.code == "too_many_steps"


def test_a_connector_with_nothing_to_read_is_refused(added, monkeypatch):
    """An unreachable platform answers `{}`, and "nothing is
    automatable right now" must not be written into a spec as "this
    account has no read"."""
    async def _empty(_user_id, **_kw):
        return {}

    monkeypatch.setattr(service.reg, "fetch_registry", _empty)
    with pytest.raises(service.MembershipError) as e:
        added("github")
    assert e.value.code == "no_template_step"


def test_every_derived_step_validates_against_its_own_manifest(registry):
    """The derivation is manifest-driven, so a manifest that grows an
    event must not be able to produce a step the validator refuses."""
    base = {"version": 2, "name": "probe", "mode": "auto",
            "trigger": {"sources": [{"id": "sched", "mode": "schedule",
                                     "schedule": {"cron_local": "0 8 * * 1-5"}}]},
            "steps": []}
    for connector_id, cap in registry.items():
        step = service._default_read_step(connector_id, cap, set())
        if step is None:
            continue
        spec = json.loads(json.dumps(base))
        spec["steps"] = [step]
        try:
            validate_spec(spec, registry, template_mode=True)
        except SpecError as exc:  # pragma: no cover — the assertion is it
            raise AssertionError(f"{connector_id}: {exc.errors}") from exc
