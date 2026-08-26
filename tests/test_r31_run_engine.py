# agent-mode: automation_threads/_turns/account_health are AGENT_ONLY.
"""R31 §4.2 / §4.2a — continue-on-failure, partial runs, the state table.

Every test here is a pin for a defect the 26 August recordings show:

  R31-12  a failing connector stopped the whole run. Jira and Gmail both
          answered; GitHub and Outlook did not; the run said "Stopped
          before it finished" and Slack was never posted, so the two
          accounts that DID answer bought the user nothing.
  R31-07  the failure was reported as `Could not reach an account` —
          no name, no reason, no fix.
  R31-31  a reads-only run never terminalized, so a thread ending "Your
          inbox is clear for now." sat under a card reading
          `Tried 1:20 · it did not finish`.
  R31-04  only a user Stop may write `stopped_by_user`; a cap, a drain
          or a crash is `failed`, with its reason.
  R31-30  `Run it now` is refused only for a RUNNING run.
"""

import json
import uuid

import pytest
from sqlalchemy import select

from app.db.database import async_session_maker
from app.db.models import (
    Automation, AutomationTurn, AccountHealth, BuildJob, User,
)
from app.agent.automations import compiler
from app.agent.automations.spec import validate_spec


REGISTRY = {
    "jira": {
        "connector_id": "jira", "push": False, "poll": True, "floor_s": 300,
        "rate_budget": {}, "scopes_read": [],
        "scopes_write_by_action": {}, "target_param_by_action": {},
        "events": [{
            "key": "issue_created", "description": "",
            "source_tool": "jira__search_issues", "poll_args": {},
            "items_path": "issues", "dedupe_field": "key",
            "fields": {"key": "key", "summary": "summary"},
        }],
    },
    "github": {
        "connector_id": "github", "push": False, "poll": True,
        "floor_s": 300, "rate_budget": {}, "scopes_read": [],
        "scopes_write_by_action": {}, "target_param_by_action": {},
        "events": [],
    },
    "gmail": {
        "connector_id": "gmail", "push": False, "poll": True,
        "floor_s": 300, "rate_budget": {}, "scopes_read": [],
        "scopes_write_by_action": {}, "target_param_by_action": {},
        "events": [],
    },
    "outlook": {
        "connector_id": "outlook", "push": False, "poll": True,
        "floor_s": 300, "rate_budget": {}, "scopes_read": [],
        "scopes_write_by_action": {}, "target_param_by_action": {},
        "events": [],
    },
    "slack": {
        "connector_id": "slack", "push": False, "poll": False,
        "floor_s": 300, "rate_budget": {}, "scopes_read": [],
        "scopes_write_by_action": {"slack__send_message": ["w"]},
        "target_param_by_action": {"slack__send_message": "channel"},
        "events": [],
    },
}

_OK_EMPTY = {"kind": "ok", "content": "{}"}


def _read_step(sid: str, connector: str, tool: str) -> dict:
    return {
        "id": sid, "connector_id": connector, "tool": tool,
        "params": {"q": "x"},
        "collect": {"items_path": "items",
                    "fields": {"t": "t"},
                    "format": "{{item.t}}",
                    "empty_text": "none"},
    }


def _post_step() -> dict:
    return {
        "id": "post", "connector_id": "slack",
        "tool": "slack__send_message",
        "params": {"channel": "{{grant.target.id}}", "text": "brief"},
        "grant_id": "g-1",
        "grant_target": {"kind": "channel", "id": "C-ALL",
                         "label": "#all-toup"},
    }


def _spec(steps: list[dict], name: str = "Morning work brief"):
    return validate_spec({
        "version": 2, "name": name, "mode": "auto",
        "trigger": {"sources": [
            {"id": "sched", "mode": "schedule",
             "schedule": {"cron_local": "0 8 * * 1-5"}},
        ]},
        "steps": steps,
    }, REGISTRY)


async def _mk_user() -> str:
    uid = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(id=uid, email=f"{uid[:8]}@example.com",
                    hashed_password="x", name="R31"))
        await db.commit()
    return uid


async def _mk_automation(uid: str, vspec):
    async with async_session_maker() as db:
        a = Automation(
            user_id=uid, name=vspec.name, status="armed",
            spec_json=json.dumps(vspec.raw, sort_keys=True),
            trigger_mode=vspec.trigger_mode,
            connector_id=vspec.trigger_connector_id,
        )
        db.add(a)
        await db.flush()
        await compiler.compile_bindings(db, a, vspec)
        await db.commit()
        return a


def _dispatch(outcomes: dict):
    """`outcomes[tool]` is an envelope, or an Exception to raise."""
    calls: list[str] = []

    async def _fn(user_id, *, connector_id, tool_name, tool_input,
                  grant_id=None, automation_id=None, request_id=None,
                  timeout_s=60.0):
        calls.append(tool_name)
        resp = outcomes.get(tool_name, _OK_EMPTY)
        if isinstance(resp, Exception):
            raise resp
        return resp

    _fn.calls = calls
    return _fn


async def _fire(monkeypatch, a, vspec, outcomes):
    from app.agent.automations import executor_v2
    fn = _dispatch(outcomes)
    monkeypatch.setattr(
        "app.agent.automations.registry.dispatch_via_platform", fn)
    monkeypatch.setattr(
        "app.agent.automations.executor_v2.AUTOMATION_OUTBOX_UNDO_WINDOW_S",
        0)
    async with async_session_maker() as db:
        a2 = await db.get(Automation, a.id)
        rc = await executor_v2.run_schedule_fire_v2(
            db, a2, vspec, vspec.schedule_source(),
            fire_key=f"t:{uuid.uuid4()}",
        )
    return rc, fn


async def _latest_run(a_id: str) -> BuildJob:
    async with async_session_maker() as db:
        return (await db.execute(
            select(BuildJob).where(BuildJob.source_id == a_id)
            .order_by(BuildJob.created_at.desc()).limit(1)
        )).scalar_one()


async def _turns(a_id: str) -> list[dict]:
    from app.agent.automations import ledger
    async with async_session_maker() as db:
        thread = await ledger.thread_for(db, a_id)
        if thread is None:
            return []
        rows = (await db.execute(
            select(AutomationTurn)
            .where(AutomationTurn.thread_id == thread.id)
            .order_by(AutomationTurn.seq.asc())
        )).scalars().all()
        return [
            {"kind": r.kind, **json.loads(r.payload_json)} for r in rows
        ]


# ── §4.2a continue-on-failure ────────────────────────────────────────


@pytest.mark.asyncio
async def test_partial_run_posts_and_names_every_failed_source(monkeypatch):
    """The founder's run, with the fix.

    Jira and Gmail answer; GitHub and Outlook do not. The brief must
    still be posted to Slack, `accounts_failed` must hold both broken
    accounts, the thread must carry one `needs_you` card per broken
    account with a real reason and a working fix, and the honest line
    must NAME them.
    """
    uid = await _mk_user()
    vspec = _spec([
        _read_step("jira", "jira", "jira__search_issues"),
        _read_step("gh", "github", "github__search_issues"),
        _read_step("mail", "gmail", "gmail__search_messages"),
        _read_step("out", "outlook", "outlook__search_messages"),
        _post_step(),
    ])
    a = await _mk_automation(uid, vspec)

    rc, fn = await _fire(monkeypatch, a, vspec, {
        "github__search_issues": RuntimeError(
            "step 'gh' failed: tool_error: the organization has not "
            "approved this OAuth app"),
        "outlook__search_messages": RuntimeError(
            "step 'out' failed: reauth_required: token expired"),
    })
    assert rc == "run"

    # 1. The brief WENT OUT. Two accounts answered; the run is worth
    #    something to the user only if their work is delivered.
    assert "slack__send_message" in fn.calls, (
        "a partial run must still post — the whole point of "
        "continue-on-failure is that two working accounts are not "
        "wasted by two broken ones"
    )

    # 2. Both broken accounts are on the run, by name.
    job = await _latest_run(a.id)
    cfg = job.config_json or {}
    assert set(cfg.get("accounts_failed") or []) == {"github", "outlook"}

    # 3. One needs_you card each, with a real reason and a real fix.
    turns = await _turns(a.id)
    cards = {t["account_id"]: t for t in turns if t["kind"] == "needs_you"}
    assert set(cards) == {"github", "outlook"}
    assert cards["github"]["reason_code"] == "org_approval_needed"
    assert cards["github"]["fix"] == "approve"
    assert cards["outlook"]["reason_code"] == "token_expired"
    assert cards["outlook"]["fix"] == "reconnect"
    for card in cards.values():
        assert card["sentence"] and "{" not in card["sentence"]
        assert card["fix_label"]

    # 4. The brief says what is missing from it, by name (§4.2a).
    #    The words are C's (`missing_from_this_*`, falling back to
    #    `could_not_reach_*`); what is pinned here is that the sentence
    #    EXISTS and NAMES both accounts. A brief that silently omits two
    #    of four sources is the failure the partial-run design exists to
    #    prevent, and silence is what it looked like on 26 August.
    honest = [t["text"] for t in turns if t["kind"] == "agent"
              and "GitHub" in t.get("text", "")
              and "Outlook" in t.get("text", "")]
    assert honest, (
        "a partial brief must say what it could not read, by name"
    )

    # 5. `an account` never appears (R31-07).
    for t in turns:
        for value in t.values():
            if isinstance(value, str):
                assert "an account" not in value


@pytest.mark.asyncio
async def test_a_run_whose_every_source_fails_posts_nothing(monkeypatch):
    """`failed`, not `partial` — and still no nameless failure.

    A brief assembled from nothing is a lie with a nice layout, so the
    write is never staged. The thread still carries one named card per
    account, which is the whole difference from before.
    """
    uid = await _mk_user()
    vspec = _spec([
        _read_step("gh", "github", "github__search_issues"),
        _read_step("out", "outlook", "outlook__search_messages"),
        _post_step(),
    ])
    a = await _mk_automation(uid, vspec)

    rc, fn = await _fire(monkeypatch, a, vspec, {
        "github__search_issues": RuntimeError("boom: timeout"),
        "outlook__search_messages": RuntimeError("boom: reauth_required"),
    })
    assert rc == "failed"
    assert "slack__send_message" not in fn.calls

    job = await _latest_run(a.id)
    assert job.status == "failed"
    assert job.outcome == "all_sources_failed"
    # The job row's own message names both accounts.
    assert "GitHub" in (job.user_message or "")
    assert "Outlook" in (job.user_message or "")
    assert "an account" not in (job.user_message or "")

    cards = [t for t in await _turns(a.id) if t["kind"] == "needs_you"]
    assert {c["account_id"] for c in cards} == {"github", "outlook"}


@pytest.mark.asyncio
async def test_failed_read_never_wears_a_count(monkeypatch):
    """R31-12/F4: `0 new threads` on a read that never happened.

    Gmail's run row said `0 new threads` while the Gmail app showed 7
    unread. A read that FAILED must wear a failure phrasing, never a
    count — a zero the user can act on and a zero that means "I could
    not look" are different facts.
    """
    uid = await _mk_user()
    vspec = _spec([
        _read_step("mail", "gmail", "gmail__search_messages"),
        _post_step(),
    ])
    a = await _mk_automation(uid, vspec)
    await _fire(monkeypatch, a, vspec, {
        "gmail__search_messages": RuntimeError("boom: reauth_required"),
    })

    tools = [t for t in await _turns(a.id) if t["kind"] == "tool"]
    assert tools, "a failed read still owes the thread its tool turn"
    failed = [t for t in tools if not t["ok"]]
    assert failed
    for t in failed:
        assert "0 " not in t["detail"], (
            f"a failed read reported a count: {t['detail']!r}"
        )


@pytest.mark.asyncio
async def test_a_silent_skip_writes_no_card(monkeypatch):
    """`on_error: skip` stays silent — the Teams provider_down
    precedent. Only `continue` owes the user a card."""
    uid = await _mk_user()
    steps = [_read_step("gh", "github", "github__search_issues"),
             _read_step("jira", "jira", "jira__search_issues"),
             _post_step()]
    steps[0]["on_error"] = "skip"
    vspec = _spec(steps)
    a = await _mk_automation(uid, vspec)

    await _fire(monkeypatch, a, vspec, {
        "github__search_issues": RuntimeError("boom: provider_down"),
    })
    cards = [t for t in await _turns(a.id) if t["kind"] == "needs_you"]
    assert cards == []


@pytest.mark.asyncio
async def test_health_is_written_from_the_run(monkeypatch):
    """R31-13. The ledger is the health source.

    Before R31 a tool call that failed auth was recorded as audit
    metadata and the identity kept saying `active` — which is how the
    Connectors page read `Connected · 10` while Outlook's own sheet read
    `Could not connect · access expired`.
    """
    from app.agent.automations import account_health

    uid = await _mk_user()
    vspec = _spec([
        _read_step("out", "outlook", "outlook__search_messages"),
        _read_step("jira", "jira", "jira__search_issues"),
        _post_step(),
    ])
    a = await _mk_automation(uid, vspec)
    await _fire(monkeypatch, a, vspec, {
        "outlook__search_messages": RuntimeError("boom: reauth_required"),
    })

    async with async_session_maker() as db:
        row = (await db.execute(
            select(AccountHealth).where(
                AccountHealth.user_id == uid,
                AccountHealth.account_id == "outlook",
            )
        )).scalar_one()
        assert row.state == "expired"
        assert row.reason_code == "token_expired"
        assert row.fix == "reconnect"
        assert row.source == "use"

        # And every surface reads THAT, not the identity's opinion.
        state = await account_health.state_for(
            db, user_id=uid, account_id="outlook",
            identity_status="active",          # the vault still says fine
        )
        assert state["account_state"] == "expired", (
            "a live token that fails every call is not `Connected`"
        )


@pytest.mark.asyncio
async def test_a_transient_failure_keeps_connected(monkeypatch):
    """The other direction of the same rule.

    A timeout says nothing about the credential. Moving an account to
    `Needs reconnecting` for one bad minute teaches the user to ignore
    the words, and sends them through an OAuth round trip that fixes
    nothing.
    """
    from app.agent.automations import account_health

    uid = await _mk_user()
    vspec = _spec([
        _read_step("jira", "jira", "jira__search_issues"),
        _post_step(),
    ])
    a = await _mk_automation(uid, vspec)
    await _fire(monkeypatch, a, vspec, {
        "jira__search_issues": RuntimeError("boom: timeout"),
    })

    async with async_session_maker() as db:
        state = await account_health.state_for(
            db, user_id=uid, account_id="jira", identity_status="active",
        )
    assert state["account_state"] == "connected"
    assert state["fix"] == "retry"

    cards = [t for t in await _turns(a.id) if t["kind"] == "needs_you"]
    assert [c["fix"] for c in cards] == ["retry"]


# ── R31-31: the run that never ended ─────────────────────────────────


@pytest.mark.asyncio
async def test_reads_only_run_terminalizes(monkeypatch):
    """A reads-only spec is legal (R30 §4.11a) and never terminalized.

    `_run_steps` reached `_finalize_job` only through the outbox flush,
    and a reads-only run has no outbox row — so the job sat `running`
    for 360 s and was reaped as `failed/lost` with a "Fix this" chip.
    That is the founder's `Morning new-email briefing`: a thread ending
    "Your inbox is clear for now." under a card reading `Tried 1:20 ·
    it did not finish`.
    """
    uid = await _mk_user()
    vspec = _spec([_read_step("mail", "gmail", "gmail__search_messages")],
                  name="Morning new-email briefing")
    a = await _mk_automation(uid, vspec)

    rc, _ = await _fire(monkeypatch, a, vspec, {})
    assert rc == "run"

    job = await _latest_run(a.id)
    assert job.status == "completed", (
        f"a reads-only run ended {job.status!r}/{job.outcome!r} — it must "
        "terminalize itself, or the stuck-run reaper will call it failed"
    )
    assert job.outcome == "sent"
