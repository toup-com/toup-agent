# agent-mode: automations/automation_threads/_turns/_writes/build_jobs are
# AGENT_ONLY — the three run-close tests drive the REAL executor and outbox
# against them (it borrows test_run_ledger_v3's fixtures, which are agent-mode
# for the same reason). The three container_of tests are pure and mode-agnostic;
# they ride along rather than earning the file a second entry.
"""R42 review — the run's END, now that the terminal goes before the prose.

R42 moved the outbox flush ahead of the narrator so the write, and the
run's terminal with it, stop waiting on two LLM calls. Two things in the
engine were still written against the old ordering:

  1. `ledger.close_ledger` reads "the run is terminal and has no result
     turn" as "narration failed outright" and appends a mechanical one.
     With the flush first that test fires on every HEALTHY write run —
     the narrator's real result then lands behind the fabricated one and
     the thread carries two. The close is deferred to after phase 2
     instead (`narration_pending`), which also restores what R42 had
     silently taken with it: the missing-item reconciliation, the
     vocabulary tripwire and the result-row episodes all read turns that
     do not exist at flush time.

  2. A preview ROW pin's id is `<container id>#<row id>` and its kind is
     the one a CONTAINER pin also uses, so `_apply_focus_scope` aimed a
     read at the channel `C0ALL#1712345.678`, which exists nowhere.
     `contents.container_of` is the one place that format is taken
     apart, and every target fill goes through it.
"""

from __future__ import annotations

import pytest

from app.agent.automations import ledger
from app.agent.automations.spec import validate_spec
from app.db.models.automation_ledger import RESULT_VOCABULARIES

from tests.test_run_ledger_v3 import (  # noqa: F401 — shared fixtures
    REGISTRY_V2, _fire, _mk_automation_v2, _mk_user, _one_run,
)


def _write_spec():
    """One read that collects ITEMS (so the completeness invariant
    applies) and one write (so the terminal lands in the flush)."""
    return validate_spec({
        "version": 2, "name": "Brief", "mode": "auto",
        "trigger": {"sources": [
            {"id": "sched", "mode": "schedule",
             "schedule": {"cron_local": "0 8 * * 1-5"}}]},
        "steps": [
            {"id": "issues", "connector_id": "jira",
             "tool": "jira__search_issues", "params": {"jql": "x"},
             "collect": {"items_path": "issues",
                         "fields": {"key": "key", "summary": "summary"},
                         "format": "{{item.key}} {{item.summary}}",
                         "empty_text": "none"},
             "on_error": "skip"},
            {"id": "post", "connector_id": "slack",
             "tool": "slack__send_message",
             "params": {"channel": "{{grant.target.id}}",
                        "text": "{{steps.issues.text}}"},
             "grant_id": "g-1",
             "grant_target": {"kind": "channel", "id": "C-PIN",
                              "label": "#platform"}},
        ],
    }, REGISTRY_V2)


async def _run_turns(a_id: str) -> list[dict]:
    from app.db.database import async_session_maker
    job = await _one_run(a_id)
    async with async_session_maker() as db:
        return await ledger.run_turns(db, run_id=job.id)


def _narrator_returning_a_result(monkeypatch):
    async def _narrate(record, *, complete=None):
        ids = [it["id"] for st in record["steps"] for it in st["items"]]
        groups = [{"rank": i + 1, "label": label, "tone": tone, "rows": []}
                  for i, (label, tone)
                  in enumerate(RESULT_VOCABULARIES["brief"])]
        groups[0]["rows"].append({"text": "both of them", "sub": "",
                                  "tag": str(len(ids)), "item_refs": ids})
        return {"turns": [{"kind": "result", "title": "Your morning",
                           "vocabulary": "brief", "groups": groups}],
                "problems": [], "attempts": 1}
    monkeypatch.setattr(
        "app.agent.automations.narrator.narrate_run", _narrate)


# ── one result turn, whichever way the narration goes ────────────────


@pytest.mark.asyncio
async def test_a_narrated_write_run_ends_with_the_narrators_result_only(
        monkeypatch, caplog):
    _narrator_returning_a_result(monkeypatch)

    uid = await _mk_user()
    vspec = _write_spec()
    a = await _mk_automation_v2(uid, vspec)
    with caplog.at_level("WARNING"):
        assert await _fire(monkeypatch, uid, a, vspec) == "run"

    turns = await _run_turns(a.id)
    results = [t for t in turns if t["kind"] == "result"]
    assert len(results) == 1, [t["kind"] for t in turns]
    assert results[0]["title"] == "Your morning"
    # The fabricated stand-in announces itself in the log before it is
    # appended — so this is the same assertion twice, on purpose.
    assert "unaccounted" not in caplog.text

    # The close still RAN, just later: the accounts stamp is its work,
    # and the flag it was deferred by is cleared behind it.
    job = await _one_run(a.id)
    cfg = job.config_json or {}
    assert cfg.get("accounts_read_ok") == ["jira"]
    assert cfg.get("narration_pending") is False


@pytest.mark.asyncio
async def test_the_mechanical_result_still_lands_when_narration_dies(
        monkeypatch):
    """Deferring the close must not disarm it. A run whose narration
    never produced prose still owes the thread ONE result turn."""
    async def _narrate(record, *, complete=None):
        raise RuntimeError("model down")
    monkeypatch.setattr(
        "app.agent.automations.narrator.narrate_run", _narrate)

    uid = await _mk_user()
    vspec = _write_spec()
    a = await _mk_automation_v2(uid, vspec)
    assert await _fire(monkeypatch, uid, a, vspec) == "run"

    turns = await _run_turns(a.id)
    results = [t for t in turns if t["kind"] == "result"]
    assert len(results) == 1, [t["kind"] for t in turns]
    rows = [r for g in results[0]["groups"] for r in g["rows"]]
    assert any("could not rank" in r["text"] for r in rows), rows


@pytest.mark.asyncio
async def test_a_finished_write_run_reaches_its_own_total(monkeypatch):
    """R38 made narration a visible step and extended the run's total by
    one. R42 stopped EMITTING a running frame for it on a terminalized
    run — correct, a `running` frame after `Done` walks the card
    backwards — and stopped stamping the columns with it, which left
    every write run one short of its own total forever."""
    _narrator_returning_a_result(monkeypatch)

    uid = await _mk_user()
    vspec = _write_spec()
    a = await _mk_automation_v2(uid, vspec)
    assert await _fire(monkeypatch, uid, a, vspec) == "run"

    job = await _one_run(a.id)
    assert job.progress_total == len(vspec.steps) + 1
    assert job.progress_step == job.progress_total


# ── a pin id is not a target id ──────────────────────────────────────


def test_a_row_pin_aims_a_read_at_its_CONTAINER_not_at_the_row():
    """`<container id>#<row id>` is a pin, never a channel id. Both
    halves of the hazard are here: a Slack ROW pin wears the same
    `thread` kind the table accepts, and Teams mints `thread` for its
    container AND its rows."""
    from app.agent.automations.executor_v2 import _apply_focus_scope

    slack = _apply_focus_scope(
        "slack", "slack__read_messages", {"channel": "", "limit": 10},
        [{"kind": "thread", "id": "C0ALL#1712345.678", "label": "sam: hi"}],
    )
    assert slack["channel"] == "C0ALL"

    teams = _apply_focus_scope(
        "teams", "teams__read_chat_messages", {},
        [{"kind": "thread", "id": "19:abc@thread.v2#msg-4"}],
    )
    assert teams["chat_id"] == "19:abc@thread.v2"

    # A container pin still fills with itself.
    assert _apply_focus_scope(
        "slack", "slack__read_messages", {},
        [{"kind": "channel", "id": "C0ALL"}],
    )["channel"] == "C0ALL"


def test_a_github_ticket_pin_names_its_repository_and_never_splits_a_row():
    """`acme/api#42` split into owner `acme` and repo `api#42` before —
    a repository that does not exist, on a read the pin was only ever
    meant to rank."""
    from app.agent.automations.executor_v2 import _apply_focus_scope

    out = _apply_focus_scope(
        "github", "github__list_issues", {"state": "open"},
        [{"kind": "ticket", "id": "acme/api#42", "label": "#42 flaky test"}],
    )
    assert (out["owner"], out["repo"]) == ("acme", "api")


def test_a_pin_whose_container_cannot_be_resolved_is_skipped():
    """Never guessed at, and never the reason a later pin is dropped."""
    from app.agent.automations.executor_v2 import _apply_focus_scope

    # `person` is not a place: Slack's table accepts channel|thread only.
    assert _apply_focus_scope(
        "slack", "slack__read_messages", {},
        [{"kind": "person", "id": "U-1"}],
    ) == {}
    # A jira ticket pin with no project in its key names nothing — and
    # github's own table is asked for a repo it cannot form.
    assert _apply_focus_scope(
        "github", "github__list_issues", {},
        [{"kind": "ticket", "id": "no-slash-here"}],
    ) == {}
