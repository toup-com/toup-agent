# agent-mode
"""Routine → automation migration (CONTRACTS-R30.md §4.11a/§4.11b).

RUN_MODE=agent (routines/automations/automation_bindings are
AGENT_ONLY). Listed in COVERAGE_DEBT.txt with `# agent-mode` so the CI
agent sweep runs it.

Pins (§7): email_briefing → automation ONCE; idempotent; the promised
time is the routine's schedule VERBATIM (the D-12 pin — never
re-derived from a creation instant); reminders/agent_tasks untouched;
arm failure leaves a draft and reports instead of raising. The founder
path itself runs in D's live pass.
"""

import json
import uuid

import pytest
from sqlalchemy import select

from app.agent.automations import compiler, registry as reg
from app.agent.automations import routine_migration as mig
from app.db.database import async_session_maker
from app.db.models import (
    Automation, AutomationBinding, BuildJob, Routine, User,
)

REGISTRY = {
    "gmail": {
        "connector_id": "gmail", "push": True, "poll": False,
        "floor_s": 300, "rate_budget": {}, "scopes_read": [],
        "scopes_write_by_action": {}, "target_param_by_action": {},
        "events": [{
            "key": "email_received", "description": "",
            "dedupe_field": "gmail_message_id",
            "fields": {"message_id": "gmail_message_id"},
        }],
    },
}


@pytest.fixture(autouse=True)
def _registry(monkeypatch):
    async def _fake_fetch(user_id, *, force=False):
        return REGISTRY
    monkeypatch.setattr(reg, "fetch_registry", _fake_fetch)


async def _mk_user() -> str:
    uid = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(id=uid, email=f"{uid[:8]}@example.com",
                    hashed_password="x", name="Migration"))
        await db.commit()
    return uid


async def _mk_routine(uid: str, **over) -> str:
    row = Routine(
        user_id=uid,
        kind=over.pop("kind", "email_briefing"),
        enabled=over.pop("enabled", True),
        name=over.pop("name", "Morning briefing"),
        schedule_cron_local=over.pop("schedule_cron_local", "0 8 * * *"),
        config_json=over.pop("config_json",
                             {"connector_identity_id": "ci-1"}),
        **over,
    )
    async with async_session_maker() as db:
        db.add(row)
        await db.commit()
        return row.id


@pytest.mark.asyncio
async def test_enabled_briefing_migrates_once_to_an_armed_automation():
    uid = await _mk_user()
    rid = await _mk_routine(uid)

    async with async_session_maker() as db:
        result = await mig.migrate_email_briefings(db, user_id=uid)

    assert [m["routine_id"] for m in result["migrated"]] == [rid]
    assert result["errors"] == [] and result["skipped"] == []
    aid = result["migrated"][0]["automation_id"]
    assert result["migrated"][0]["armed"] is True

    async with async_session_maker() as db:
        a = await db.get(Automation, aid)
        assert a.status == "armed"
        assert a.name == "Morning briefing"
        raw = json.loads(a.spec_json)
        assert raw["version"] == 2
        assert raw["mode"] == "auto"
        # §4.11b / D-12: the routine's cron IS the promised time —
        # copied verbatim, never re-derived.
        [src] = raw["trigger"]["sources"]
        assert src["mode"] == "schedule"
        assert src["schedule"] == {"cron_local": "0 8 * * *"}
        # Reads only: one gmail read step, NO write steps (delivery is
        # the notification pipeline's, per the §4.11a deviation).
        assert [s["tool"] for s in raw["steps"]] == ["gmail__list_messages"]
        assert all(not s.get("grant_id") for s in raw["steps"])

        # The routine is retired and stamped; the stamp MERGES.
        r = await db.get(Routine, rid)
        assert r.enabled is False
        assert r.config_json["migrated_to"] == aid
        assert r.config_json["connector_identity_id"] == "ci-1"

        # The armed schedule the compiler wrote carries the same cron.
        binding = (await db.execute(
            select(AutomationBinding)
            .where(AutomationBinding.automation_id == aid)
        )).scalar_one()
        armed_routine = await db.get(Routine, binding.target_id)
        assert armed_routine.schedule_cron_local == "0 8 * * *"
        assert armed_routine.enabled is True


@pytest.mark.asyncio
async def test_disabled_briefing_is_not_resurrected_unless_selected():
    """ND-12: a routine the user SWITCHED OFF must not come back as a
    new object (the founder's list went 3 -> 6 that way). Explicitly
    selecting it still migrates it, to a paused draft."""
    uid = await _mk_user()
    rid = await _mk_routine(uid, enabled=False)

    async with async_session_maker() as db:
        untouched = await mig.migrate_email_briefings(db, user_id=uid)
    assert untouched["migrated"] == []
    assert [e["routine_id"] for e in untouched["needs_review"]] == [rid]
    assert "disabled" in untouched["needs_review"][0]["reason"]

    async with async_session_maker() as db:
        result = await mig.migrate_email_briefings(
            db, user_id=uid, routine_ids=[rid],
        )

    aid = result["migrated"][0]["automation_id"]
    assert result["migrated"][0]["armed"] is False
    async with async_session_maker() as db:
        a = await db.get(Automation, aid)
        assert a.status == "draft"
        r = await db.get(Routine, rid)
        assert r.enabled is False
        assert r.config_json["migrated_to"] == aid


@pytest.mark.asyncio
async def test_second_call_migrates_nothing():
    uid = await _mk_user()
    rid = await _mk_routine(uid)

    async with async_session_maker() as db:
        first = await mig.migrate_email_briefings(db, user_id=uid)
    async with async_session_maker() as db:
        second = await mig.migrate_email_briefings(db, user_id=uid)

    assert len(first["migrated"]) == 1
    assert second["migrated"] == [] and second["errors"] == []
    assert [s["routine_id"] for s in second["skipped"]] == [rid]
    async with async_session_maker() as db:
        autos = (await db.execute(
            select(Automation).where(Automation.user_id == uid)
        )).scalars().all()
        assert len(autos) == 1, "a second call must not mint a twin"


@pytest.mark.asyncio
async def test_reminders_and_engine_owned_kinds_are_never_touched():
    """§4.11a: pure reminders keep the main-chat path. The engine-owned
    kinds are excluded for a different reason — they ARE an automation's
    own compiled bindings, so "migrating" one would duplicate the
    automation that owns it."""
    uid = await _mk_user()
    untouchable = [
        await _mk_routine(uid, kind="reminder", reminder_text="stand up",
                          config_json=None),
        await _mk_routine(uid, kind="automation_schedule",
                          config_json=None),
        await _mk_routine(uid, kind="automation_poll", config_json=None),
    ]

    async with async_session_maker() as db:
        result = await mig.migrate_email_briefings(db, user_id=uid)

    assert result["migrated"] == []
    assert result["superseded"] == []
    async with async_session_maker() as db:
        for rid in untouchable:
            r = await db.get(Routine, rid)
            assert r.enabled is True
            assert not (r.config_json or {}).get("migrated_to")
            assert not (r.config_json or {}).get("superseded_by")
        autos = (await db.execute(
            select(Automation).where(Automation.user_id == uid)
        )).scalars().all()
        assert autos == []


@pytest.mark.asyncio
async def test_nd9_the_founders_real_shape_migrates():
    """ND-9 (live, 2026-08-25): the founder's "Morning new-email
    briefing" is kind `agent_task` with cron `0 8 * * *` — NOT
    `email_briefing`. Selecting on the kind vocabulary matched nothing
    on the real account and returned a clean 200 that migrated nothing,
    which would have been recorded as "migration done". This drives the
    ACTUAL production shape, not the kind the name suggests."""
    uid = await _mk_user()
    rid = await _mk_routine(
        uid, kind="agent_task", name="Morning new-email briefing",
        prompt_text="summarise my new email",
        schedule_cron_local="0 8 * * *", schedule_kind="cron",
    )
    # Unprompted it is only REPORTED — an agent_task is not a briefing
    # by kind, and its intent must be selected, never inferred (ND-12).
    async with async_session_maker() as db:
        unprompted = await mig.migrate_email_briefings(db, user_id=uid)
    assert unprompted["migrated"] == []
    assert [e["routine_id"] for e in unprompted["needs_review"]] == [rid]

    # Selected by id — the way the live pass drives it.
    async with async_session_maker() as db:
        result = await mig.migrate_email_briefings(
            db, user_id=uid, routine_ids=[rid],
        )

    assert len(result["migrated"]) == 1, result
    entry = result["migrated"][0]
    assert entry["routine_id"] == rid
    async with async_session_maker() as db:
        a = await db.get(Automation, entry["automation_id"])
        raw = json.loads(a.spec_json)
        crons = [src.get("schedule", {}).get("cron_local")
                 for src in raw["trigger"]["sources"] if src.get("schedule")]
        # §4.11b: the routine's cron IS the promised time, verbatim.
        assert crons == ["0 8 * * *"], crons
        r = await db.get(Routine, rid)
        assert r.enabled is False
        assert (r.config_json or {}).get("migrated_to") == a.id


@pytest.mark.asyncio
async def test_nd9_a_duplicate_intent_is_retired_not_cloned():
    """D-12 collapse: the founder's per-minute "Jira → Slack new-issue
    alerts" routine duplicates automation "Jira → Slack". Migrating it
    would make a THIRD object for one intent; it is retired against the
    automation that already owns the intent."""
    uid = await _mk_user()
    async with async_session_maker() as db:
        a = Automation(
            user_id=uid, name="Jira → Slack new-issue alerts",
            status="armed", spec_json=json.dumps({"version": 2}),
            trigger_mode="poll", connector_id="jira",
        )
        db.add(a)
        await db.commit()
        twin_id = a.id

    rid = await _mk_routine(
        uid, kind="agent_task", name="Jira → Slack new-issue alerts",
        prompt_text="post new jira issues to slack",
        schedule_cron_local="* * * * *", schedule_kind="cron",
    )
    async with async_session_maker() as db:
        result = await mig.migrate_email_briefings(db, user_id=uid)

    assert result["migrated"] == [], result
    assert len(result["superseded"]) == 1, result
    assert result["superseded"][0]["superseded_by"] == twin_id
    async with async_session_maker() as db:
        r = await db.get(Routine, rid)
        assert r.enabled is False, "the per-minute duplicate must stop"
        assert (r.config_json or {}).get("superseded_by") == twin_id
        # No third object was created.
        autos = (await db.execute(
            select(Automation).where(Automation.user_id == uid)
        )).scalars().all()
        assert len(autos) == 1


@pytest.mark.asyncio
async def test_nd9_a_non_mail_task_is_flagged_not_mis_migrated():
    """The migrated spec READS GMAIL. Converting a Jira alerter into a
    mail brief would misstate what the user set up, so it is reported
    for review instead — and a one-shot is a reminder wearing another
    kind."""
    uid = await _mk_user()
    jira = await _mk_routine(
        uid, kind="agent_task", name="Weekly work recap",
        prompt_text="summarise closed jira tickets",
        schedule_cron_local="0 17 * * 5", schedule_kind="cron",
    )
    oneshot = await _mk_routine(
        uid, kind="agent_task", name="Remind me about the mail once",
        prompt_text="mail me", schedule_kind="at",
        schedule_cron_local="@at", auto_disable_after_fire=True,
    )
    async with async_session_maker() as db:
        result = await mig.migrate_email_briefings(db, user_id=uid)

    assert result["migrated"] == [], result
    flagged = {e["routine_id"] for e in result["needs_review"]}
    assert flagged == {jira, oneshot}, result["needs_review"]
    async with async_session_maker() as db:
        for rid in (jira, oneshot):
            r = await db.get(Routine, rid)
            # Untouched: a routine we refuse to migrate keeps working.
            assert r.enabled is True


@pytest.mark.asyncio
async def test_every_schedule_and_default_name_survive():
    uid = await _mk_user()
    rid = await _mk_routine(
        uid, name=None, schedule_kind="every",
        schedule_interval_seconds=3600, schedule_cron_local="@every",
    )
    async with async_session_maker() as db:
        result = await mig.migrate_email_briefings(db, user_id=uid)
    aid = result["migrated"][0]["automation_id"]
    assert result["migrated"][0]["routine_id"] == rid
    async with async_session_maker() as db:
        a = await db.get(Automation, aid)
        assert a.name == mig.DEFAULT_NAME
        [src] = json.loads(a.spec_json)["trigger"]["sources"]
        assert src["schedule"] == {"every_s": 3600}


@pytest.mark.asyncio
async def test_arm_failure_leaves_a_draft_and_reports(monkeypatch):
    from app.agent.automations import service

    uid = await _mk_user()
    rid = await _mk_routine(uid)

    async def _boom(db, *, automation_id, user_id):
        raise compiler.CompileError("grant_unverifiable", "platform away")

    monkeypatch.setattr(service, "arm_automation", _boom)
    async with async_session_maker() as db:
        result = await mig.migrate_email_briefings(db, user_id=uid)

    entry = result["migrated"][0]
    assert entry["armed"] is False
    assert "platform away" in entry["arm_error"]
    async with async_session_maker() as db:
        a = await db.get(Automation, entry["automation_id"])
        assert a.status == "draft"
        r = await db.get(Routine, rid)
        assert r.enabled is False, "the automation owns the intent now"
        assert r.config_json["migrated_to"] == a.id


@pytest.mark.asyncio
async def test_migration_report_lists_every_briefing():
    uid = await _mk_user()
    rid = await _mk_routine(uid)
    async with async_session_maker() as db:
        await mig.migrate_email_briefings(db, user_id=uid)
        report = await mig.migration_report(db, user_id=uid)
    [row] = report["routines"]
    assert row["routine_id"] == rid
    assert row["name"] == "Morning briefing"
    assert row["migrated_to"]
    assert row["enabled"] is False


def test_promised_time_cron_renders_the_stated_time():
    # §4.11b: "8:00" said in chat must arm as 8:00 — the cron is the
    # promise rendered, not an offset from when the spec was compiled.
    assert mig.promised_time_cron("8:00") == "0 8 * * *"
    assert mig.promised_time_cron("08:00") == "0 8 * * *"
    assert mig.promised_time_cron("22:52") == "52 22 * * *"
    with pytest.raises(ValueError):
        mig.promised_time_cron("25:00")


@pytest.mark.asyncio
async def test_nd6_the_route_actually_triggers_the_migration(monkeypatch):
    """ND-6: the migration had NO production caller — pinned here: the
    agent route drives it end to end and stays idempotent."""
    import uuid as _uuid
    from app.api.automations import migrate_routines
    from app.config import settings
    from app.db.models import Routine

    uid = await _mk_user()
    monkeypatch.setattr(settings, "automations_enabled", True)
    monkeypatch.setattr(settings, "user_id", uid)
    async with async_session_maker() as db:
        db.add(Routine(
            id=str(_uuid.uuid4()), user_id=uid, kind="email_briefing",
            enabled=True, name="Morning new-email briefing",
            prompt_text="brief me", schedule_cron_local="0 8 * * *",
            schedule_kind="cron",
        ))
        await db.commit()
    out = await migrate_routines()
    assert len(out.get("migrated") or []) == 1, out
    out2 = await migrate_routines()
    assert not out2.get("migrated"), out2


@pytest.mark.asyncio
async def test_nd11_the_report_sees_its_subjects_and_predicts_the_outcome():
    """ND-11 (live on the founder, tenant already on 98d03bab): the
    migration query was widened and its report was left on the old
    singular constant, so the audit view answered {"routines": []} for
    an account whose routines the migration was about to act on — and
    an empty report reads as "nothing to migrate". The two now share ONE
    selector, and this pins the stronger property: the report is a DRY
    RUN whose prediction matches what running it actually does.
    """
    uid = await _mk_user()
    # The founder's three real shapes, plus an untouchable.
    async with async_session_maker() as db:
        twin = Automation(
            user_id=uid, name="Jira → Slack new-issue alerts",
            status="armed", spec_json=json.dumps({"version": 2}),
            trigger_mode="poll", connector_id="jira",
        )
        db.add(twin)
        await db.commit()

    brief = await _mk_routine(
        uid, kind="agent_task", name="Morning new-email briefing",
        prompt_text="summarise my new email",
        schedule_cron_local="0 8 * * *", schedule_kind="cron")
    recap = await _mk_routine(
        uid, kind="agent_task", name="Weekly work recap",
        prompt_text="summarise closed jira tickets",
        schedule_cron_local="0 17 * * 5", schedule_kind="cron")
    dupe = await _mk_routine(
        uid, kind="agent_task", name="Jira → Slack new-issue alerts",
        prompt_text="post new jira issues to slack",
        schedule_cron_local="* * * * *", schedule_kind="cron")
    await _mk_routine(uid, kind="reminder", reminder_text="stand up",
                      config_json=None)

    async with async_session_maker() as db:
        report = await mig.migration_report(db, user_id=uid)
    seen = {e["routine_id"]: e for e in report["routines"]}

    # It can SEE its subjects — the ND-11 regression.
    assert {brief, recap, dupe} <= set(seen), report
    # And it predicts each outcome, with the schedule for the capture.
    assert seen[brief]["outcome"] == "would_need_review"
    assert seen[brief]["likely_mail"] is True   # the hint, not a gate
    assert seen[brief]["schedule"] == {"cron_local": "0 8 * * *"}
    assert seen[recap]["outcome"] == "would_need_review"
    assert seen[recap].get("likely_mail") is False
    assert seen[dupe]["outcome"] == "would_supersede"
    # A reminder is never a candidate at all.
    assert all(e["kind"] != "reminder" for e in report["routines"])

    # THE PROPERTY: running it does exactly what the report predicted.
    async with async_session_maker() as db:
        result = await mig.migrate_email_briefings(
            db, user_id=uid, routine_ids=[brief],
        )
    assert [e["routine_id"] for e in result["migrated"]] == [brief]
    assert [e["routine_id"] for e in result["superseded"]] == [dupe]
    assert [e["routine_id"] for e in result["needs_review"]] == [recap]

    # After the run the report reflects the new state, not the old plan.
    async with async_session_maker() as db:
        after = await mig.migration_report(db, user_id=uid)
    seen_after = {e["routine_id"]: e for e in after["routines"]}
    assert seen_after[brief]["outcome"] == "already_migrated"
    assert seen_after[brief]["migrated_to"]
    assert seen_after[dupe]["outcome"] == "already_superseded"


@pytest.mark.asyncio
async def test_nd12_a_quote_routine_is_never_rewritten_into_a_gmail_brief():
    """ND-12 (live on the founder): "Daily motivational quote" — "Send
    Nariman one short motivational quote every day" — was converted
    into an automation whose rule read "Every day at 16:39, check
    Gmail." The intent was not misstated, it was REPLACED. A keyword
    scan over prose cannot make that call, so intent is now SELECTED:
    an agent_task is never migrated unprompted, no matter what its
    prose contains."""
    uid = await _mk_user()
    quote = await _mk_routine(
        uid, kind="agent_task", name="Daily motivational quote",
        prompt_text="Send Nariman one short motivational quote every "
                    "day. Keep it concise, uplifting, and not cheesy — "
                    "no need to email anything else.",
        schedule_cron_local="39 16 * * *", schedule_kind="cron")

    async with async_session_maker() as db:
        result = await mig.migrate_email_briefings(db, user_id=uid)

    assert result["migrated"] == [], result
    assert [e["routine_id"] for e in result["needs_review"]] == [quote]
    async with async_session_maker() as db:
        autos = (await db.execute(
            select(Automation).where(Automation.user_id == uid)
        )).scalars().all()
        assert autos == [], "no automation may be invented from prose"
        r = await db.get(Routine, quote)
        assert r.enabled is True and not (r.config_json or {}).get(
            "migrated_to")


@pytest.mark.asyncio
async def test_nd12_repair_undoes_a_mis_migration_and_restores_the_routine():
    """ND-12's trap: a mis-migrated routine is STAMPED, so a corrected
    selector skips it and the bad automation persists — the fix cannot
    self-heal. Repair reverses the pair and puts the routine back as it
    was, and refuses any automation that has already RUN."""
    uid = await _mk_user()
    quote = await _mk_routine(
        uid, kind="agent_task", name="Daily motivational quote",
        prompt_text="one uplifting quote a day",
        schedule_cron_local="39 16 * * *", schedule_kind="cron",
        enabled=False)

    # Simulate the bad pair the old rules produced (selected => migrates).
    async with async_session_maker() as db:
        bad = await mig.migrate_email_briefings(
            db, user_id=uid, routine_ids=[quote],
        )
    aid = bad["migrated"][0]["automation_id"]

    # Unprompted it is a DRY RUN: it plans, and changes nothing.
    async with async_session_maker() as db:
        dry = await mig.repair_mismigrations(db, user_id=uid)
    assert dry["repaired"] == [], dry
    assert [e["routine_id"] for e in dry["plan"]] == [quote], dry
    async with async_session_maker() as db:
        assert await db.get(Automation, aid) is not None

    async with async_session_maker() as db:
        out = await mig.repair_mismigrations(
            db, user_id=uid, routine_ids=[quote],
        )

    assert [e["routine_id"] for e in out["repaired"]] == [quote], out
    async with async_session_maker() as db:
        assert await db.get(Automation, aid) is None, "the draft is gone"
        r = await db.get(Routine, quote)
        # Restored to the state recorded at migration time — the user
        # had switched it off, so it stays off.
        assert r.enabled is False
        assert not (r.config_json or {}).get("migrated_to")

    # A pair whose automation has RUN is kept, never deleted.
    brief = await _mk_routine(uid, name="Morning briefing")
    async with async_session_maker() as db:
        ok = await mig.migrate_email_briefings(db, user_id=uid)
    ok_aid = ok["migrated"][0]["automation_id"]
    async with async_session_maker() as db:
        r = await db.get(Routine, brief)
        r.kind = "agent_task"          # force the repair to consider it
        db.add(BuildJob(
            id=str(uuid.uuid4()), user_id=uid, title="run", prompt="",
            job_type="automation_run", status="completed",
            source_kind="automation", source_id=ok_aid,
        ))
        await db.commit()
    async with async_session_maker() as db:
        out2 = await mig.repair_mismigrations(
            db, user_id=uid, routine_ids=[brief],
        )
    assert [e["automation_id"] for e in out2["kept"]] == [ok_aid], out2
    assert "record of work" in out2["kept"][0]["reason"]
    async with async_session_maker() as db:
        assert await db.get(Automation, ok_aid) is not None


@pytest.mark.asyncio
async def test_nd13_a_longer_title_still_matches_the_same_intent():
    """ND-13 (live): exact normalised equality never fired — the
    founder's routine "Jira → Slack new-issue alerts" and the
    automation "Jira → Slack" are one intent under two titles, so the
    per-minute duplicate kept running. Token-subset with >= 2 shared
    tokens is the honest test, and a single shared word must NOT
    collapse two unrelated things."""
    assert mig._same_intent("Jira → Slack new-issue alerts", "Jira → Slack")
    assert mig._same_intent("Morning brief", "Morning brief extended")
    # One shared word is not an intent match.
    assert not mig._same_intent("Daily motivational quote", "Daily recap")
    assert not mig._same_intent("Morning work brief",
                                "Morning new-email briefing")

    uid = await _mk_user()
    async with async_session_maker() as db:
        db.add(Automation(
            user_id=uid, name="Jira → Slack", status="armed",
            spec_json=json.dumps({"version": 2}), trigger_mode="poll",
            connector_id="jira",
        ))
        await db.commit()
    dupe = await _mk_routine(
        uid, kind="agent_task", name="Jira → Slack new-issue alerts",
        prompt_text="post new jira issues to slack",
        schedule_cron_local="* * * * *", schedule_kind="cron")

    async with async_session_maker() as db:
        result = await mig.migrate_email_briefings(db, user_id=uid)

    assert [e["routine_id"] for e in result["superseded"]] == [dupe], result
    async with async_session_maker() as db:
        r = await db.get(Routine, dupe)
        assert r.enabled is False, "the per-minute duplicate must stop"


@pytest.mark.asyncio
async def test_nd12_repair_never_touches_an_armed_migration():
    """D's catch, pre-empted: "would today's rules produce this pair?"
    is NOT a usable repair criterion — under selection semantics no
    agent_task migrates unprompted, so that test would undo every
    correct, explicitly-selected migration, including the one §7
    requires. An armed automation is live and doing its job: repair
    refuses it even when named."""
    uid = await _mk_user()
    brief = await _mk_routine(
        uid, kind="agent_task", name="Morning new-email briefing",
        prompt_text="summarise my new email",
        schedule_cron_local="0 8 * * *", schedule_kind="cron")
    async with async_session_maker() as db:
        res = await mig.migrate_email_briefings(
            db, user_id=uid, routine_ids=[brief],
        )
    aid = res["migrated"][0]["automation_id"]
    async with async_session_maker() as db:
        a = await db.get(Automation, aid)
        a.status = "armed"
        await db.commit()

    async with async_session_maker() as db:
        out = await mig.repair_mismigrations(
            db, user_id=uid, routine_ids=[brief],
        )
    assert out["repaired"] == [], out
    assert [e["automation_id"] for e in out["kept"]] == [aid], out
    assert "armed" in out["kept"][0]["reason"]
    async with async_session_maker() as db:
        assert await db.get(Automation, aid) is not None
        r = await db.get(Routine, brief)
        assert (r.config_json or {}).get("migrated_to") == aid


@pytest.mark.asyncio
async def test_engine_routines_are_not_listed_as_the_users_routines(
    monkeypatch,
):
    """ND-18: one automation must not read as three objects.

    D measured it on the founder tenant: the agent, asked how many
    automations exist, answered NINE against a ground truth of four. It
    was counting the engine's own `[automation] …` schedule binding
    alongside the automation it belongs to, plus the source routines the
    migration had already replaced. It degrades with use — every
    migration mints a binding AND retires a source, so each one added
    two phantoms.

    R26 listed `automation_schedule` deliberately ("it IS a schedule the
    user asked for in words"), which was true when the automation had no
    surface of its own. R30 gives it one.
    """
    from app.api.routines import list_routines
    from app.db.models import Routine

    uid = await _mk_user()
    async with async_session_maker() as db:
        rows = [
            # The user's own routine — always listed.
            Routine(id=str(uuid.uuid4()), user_id=uid, kind="briefing",
                    name="Morning brief", enabled=True,
                    schedule_kind="cron", schedule_cron_local="0 8 * * *"),
            # Engine plumbing for an automation the user sees elsewhere.
            Routine(id=str(uuid.uuid4()), user_id=uid,
                    kind="automation_schedule",
                    name="[automation] Morning new-email briefing",
                    enabled=True, schedule_kind="cron",
                    schedule_cron_local="0 8 * * *",
                    config_json={"automation_id": "a-1"}),
            Routine(id=str(uuid.uuid4()), user_id=uid,
                    kind="automation_poll",
                    name="[automation] Jira watch", enabled=True,
                    schedule_kind="every", schedule_cron_local="",
                    schedule_interval_seconds=900,
                    config_json={"automation_id": "a-2"}),
            # A routine an automation has REPLACED.
            Routine(id=str(uuid.uuid4()), user_id=uid, kind="briefing",
                    name="Old mail brief", enabled=False,
                    schedule_kind="cron", schedule_cron_local="0 7 * * *",
                    config_json={"migrated_to": "a-1"}),
            Routine(id=str(uuid.uuid4()), user_id=uid, kind="briefing",
                    name="Duplicate brief", enabled=False,
                    schedule_kind="cron", schedule_cron_local="0 7 * * *",
                    config_json={"superseded_by": "a-1"}),
        ]
        for r in rows:
            db.add(r)
        await db.commit()

    from app.config import settings
    monkeypatch.setattr(settings, "user_id", uid)
    listed = await list_routines()

    names = [r.name for r in listed]
    assert names == ["Morning brief"], (
        f"the engine's own rows reached the user: {names}")
    assert not any(n.startswith("[automation]") for n in names)
