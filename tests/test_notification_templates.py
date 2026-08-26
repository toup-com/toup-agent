"""The notification body is one exact string per situation (R30 §5.7).

The same string reaches the in-chat card, the push banner and the
live-activity end state — so these tests pin the exact bytes, not
shapes. The n=1 scheduled-brief line is the canvas's own notification
line (atlas 04) verbatim; a wording drift there is a parity break, not
a style choice.

Pure functions, no DB — runs in the platform sweep.
"""

from __future__ import annotations

from app.agent.automations.notification_templates import (
    auto_pause_body,
    draft_staged_body,
    notification_body,
    setup_card,
)


def _run(vocabulary="brief", status="completed", run_kind="scheduled", **kw):
    return {"run_kind": run_kind, "status": status, "vocabulary": vocabulary, **kw}


def test_the_canvas_notification_line_is_verbatim():
    body = notification_body("automation_run", _run(needs_count=1))
    assert body == (
        "It ran on time. One thing needs you today — open the run and "
        "I will walk you through it there."
    )


def test_zero_and_plural_needs_counts():
    assert notification_body("automation_run", _run(needs_count=0)) == (
        "It ran on time. Nothing needs you today — it is all there when you want it."
    )
    assert notification_body("automation_run", _run(needs_count=2)) == (
        "It ran on time. Two things need you today — open the run and "
        "I will walk you through them there."
    )
    # Number words for one–nine, digits above (§5.7).
    assert "Nine things" in notification_body("automation_run", _run(needs_count=9))
    assert "12 things" in notification_body("automation_run", _run(needs_count=12))


def test_run_now_swaps_the_first_clause_only():
    body = notification_body("automation_run", _run(run_kind="run_now", needs_count=1))
    assert body.startswith("Done — it ran just now.")
    assert body.endswith("open the run and I will walk you through it there.")


def test_changes_vocabulary_counts_writes():
    one = notification_body("automation_run", _run("changes", writes_count=1))
    assert one == (
        "It ran on time and made one change — open the run to see it, "
        "and what it left alone on purpose."
    )
    three = notification_body("automation_run", _run("changes", writes_count=3))
    assert three == (
        "It ran on time and made three changes — open the run to see "
        "each one, ranked by what you may want to undo."
    )


def test_failed_names_the_connector_and_offers_the_fix():
    body = notification_body(
        "automation_run", _run(status="failed", failed_connector_name="GitHub")
    )
    assert body == (
        # R31-07: "refused" is true of exactly one failure and wrong
        # about the rest — expired access did not refuse, and neither
        # did an organisation that has not approved Toup. run_summary
        # carries the name but not the reason, so the body names the
        # account and sends the user where the reason is written.
        "It could not finish — GitHub needs you. Nothing was missed. "
        "Open the run and I will show you the fix."
    )
    # R31-07. This used to assert "an account refused" and call it
    # honest. It was not: a run that fails for a reason no account owns
    # — a drain, the run cap, a crash — has no refusing account, so the
    # sentence blamed one that did not exist and named nobody. With no
    # connector there is also no fix to offer, so the invitation says
    # what it can actually deliver.
    anon = notification_body("automation_run", _run(status="failed"))
    assert anon == (
        "It could not finish. Nothing was missed. Open the run and "
        "I will show you what happened."
    )
    assert "an account" not in anon
    assert "refused" not in anon


def test_needs_you_expired_access_and_waiting():
    expired = notification_body(
        "automation_needs_you", {"failed_connector_name": "GitHub"}
    )
    assert expired == (
        "GitHub access ran out, so it stopped where it was. "
        "Reconnect and it picks up from there."
    )
    waiting = notification_body("automation_needs_you", {"status": "waiting_on_user"})
    assert waiting == (
        "It prepared a change and is waiting on you — "
        "nothing happens until you approve."
    )
    # The waiting body is identical whichever kind carries it.
    assert waiting == notification_body(
        "automation_run", _run(status="waiting_on_user")
    )


def test_draft_and_setup_surfaces():
    assert draft_staged_body() == (
        "A draft is waiting — nothing has been sent. Open the run to read it."
    )
    card = setup_card("Jira → Slack alerts")
    assert card == {
        "title": "Setting up: Jira → Slack alerts",
        "body": "Continue setting it up ›",
    }


def test_bodies_never_carry_findings():
    # A body is a count and an invitation (§5.7): no fixture name, no
    # item title, no vendor content can appear, because the inputs
    # simply do not include them — pin the input surface.
    body = notification_body("automation_run", _run(needs_count=3))
    for finding in ("TP-482", "Marcus", "SOC 2", "#platform"):
        assert finding not in body


def test_the_auto_pause_notice_is_plain_and_honest():
    body = auto_pause_body()
    assert body == (
        "It failed three times in a row, so I paused it. "
        "Open the run and I will show you what went wrong."
    )
    # The live string it replaces wore an emoji and markdown bold.
    assert "⚠" not in body and "**" not in body
