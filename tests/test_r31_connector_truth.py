# agent-mode: account_health/automation_turns are AGENT_ONLY tables.
"""R31 §4.4 / §4.7 / §4.10 — connector truth, copy, and the two writers.

Pins:
  R31-13  one account_state, from the last REAL use; a transient
          failure keeps `Connected`.
  R31-07  every failing account is named, at any count; a run with an
          empty `accounts_failed` can never wear a failure label.
  R31-25  no `{placeholder}` survives a rendered string, and the
          serializer will not serve one.
  R31-26  a server-rendered stamp is LOCAL, from the DB when the cache
          is cold.
  R31-32  the card's description is derived from the workflow, so it
          cannot name a connector that is not on the canvas.
  R31-33  a reminder whose due time is long past can never ring.
  R31-34  a non-ASCII Gmail subject round-trips.
"""

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from app.agent.automations import account_health as health
from app.services import automation_verbs as verbs


# ── R31-13: the state vocabulary ─────────────────────────────────────


@pytest.mark.parametrize("reason,expect_state,expect_fix", [
    ("token_expired", "expired", "reconnect"),
    ("token_revoked", "revoked", "reconnect"),
    ("scope_missing:Mail.Read", "scope_missing", "grant"),
    ("org_approval_needed", "org_approval_needed", "approve"),
    ("not_connected", "not_connected", "connect"),
    # The three that describe the MOMENT, not the credential.
    ("rate_limited", "connected", "retry"),
    ("vendor_down", "connected", "retry"),
    ("timeout", "connected", "retry"),
])
def test_only_a_credential_problem_moves_an_account_off_connected(
    reason, expect_state, expect_fix,
):
    """R31-13, both directions.

    A dead token must not read `Connected` anywhere — that is the
    Connectors page saying `Connected · 10` while the same Outlook
    account's sheet said `access expired`. And a timeout must not read
    `Needs reconnecting` — sending someone through an OAuth round trip
    for one bad minute fixes nothing and teaches them to ignore the
    word the next time it is true.
    """
    state, fix = health.state_for_reason(reason)
    assert state == expect_state
    assert fix == expect_fix
    assert health.is_transient(reason) is (expect_state == "connected")


def test_every_reason_has_a_sentence_and_a_button():
    """A card with no sentence is `Could not reach an account` again."""
    for reason in ("token_expired", "token_revoked", "scope_missing",
                   "org_approval_needed", "not_connected",
                   "rate_limited", "vendor_down", "timeout"):
        payload = health.needs_you_payload(
            account_id="outlook", connector_id="outlook",
            name="Outlook", reason_code=reason,
        )
        assert payload["sentence"], reason
        assert payload["fix_label"], reason
        assert payload["fix"] in health.FIXES, reason
        # R31-25: nothing rendered may carry a brace.
        assert "{" not in payload["sentence"], payload["sentence"]
        assert "{" not in payload["fix_label"], payload["fix_label"]
        # §4.3(7): the string this round exists to delete.
        assert "an account" not in payload["sentence"].lower()


def test_a_per_connector_sentence_wins_over_the_generic_one():
    """GitHub's org-approval wording is written once, in GitHub's block,
    and every surface that names it reads the same sentence."""
    s = health.sentence_for(
        account_state="org_approval_needed",
        reason_code="org_approval_needed",
        connector_id="github", name="GitHub",
    )
    assert "organisation" in s
    assert "GitHub" in s


def test_a_missing_scope_is_never_spoken_as_an_expired_token():
    """The alias that sent users through the wrong repair.

    `_C_FAILURE_ALIASES` mapped `scope_missing` onto `access_expired`,
    so an Outlook connection that had never been granted mail-read
    scope was reported as "access expired" — and reconnecting an
    unexpired token fixes nothing.
    """
    out = verbs.failure_action("outlook", "scope_missing")
    assert "expired" not in out["detail"].lower(), out
    assert "access" in out["detail"].lower() or "more" in out["detail"].lower()


# ── R31-07: every name, at any count ─────────────────────────────────


def test_a_failure_label_names_every_account():
    def turns(*ids):
        return [{"tool_kind": "read", "ok": False, "account_id": i}
                for i in ids]

    one = verbs.job_card_label(turns("github"))
    two = verbs.job_card_label(turns("github", "outlook"))
    three = verbs.job_card_label(turns("github", "outlook", "slack"))
    assert "GitHub" in one
    assert "GitHub" in two and "Outlook" in two
    for name in ("GitHub", "Outlook", "Slack"):
        assert name in three, three
    for label in (one, two, three):
        assert "an account" not in label


def test_a_run_with_no_failed_account_wears_no_failure_label():
    """ND-16's rule, applied to the second reader.

    The home card was fixed to stop naming a connector for a run that
    died with none recorded; `job_card_label` kept its own predicate and
    was never fixed. A run that died mid-"Wrapping up" must not accuse
    an account that never refused.
    """
    label = verbs.job_card_label(
        [{"tool_kind": "read", "ok": False, "account_id": "github"}],
        failed_accounts=[],
    )
    assert "GitHub" not in label
    assert "an account" not in label


# ── R31-25: no placeholder survives ──────────────────────────────────


def test_the_renderer_drops_a_clause_it_cannot_fill():
    """The `{need_count}` mechanism itself, driven directly.

    This must NOT be written as "render the shipped entries and look
    for braces": C's entry tables are slot-free today (deliberately, so
    a merge cannot land slot-bearing strings before this renderer), so
    such a test passes whether the renderer drops clauses or not. It
    was written that way first and a mutation that removed the drop
    left it green — the kinder-cage failure this repo has a name for.

    So: hand the renderer the exact shape that shipped.
    """
    # The literal template from the entry that reached a user's screen.
    out = verbs._n("{count} issues · {need_count} needs you", 3)
    assert "{" not in out and "}" not in out, out
    assert "3 issues" in out
    assert "need_count" not in out

    # A slot the caller CAN fill is filled, not dropped.
    filled = verbs._n("Posted in {target}", None, target="#all-toup")
    assert filled == "Posted in #all-toup"

    # A whole-sentence template with an unfillable slot says nothing
    # rather than showing a brace.
    assert verbs._n("Held {when}", None) == ""


def test_no_placeholder_survives_the_shipped_entries():
    """And the entries as they actually ship, as a second net."""
    for connector, tool, kind in (
        ("jira", "jira__search_issues", "read"),
        ("jira", "jira__add_comment", "write"),
        ("jira", "jira__create_issue", "write"),
        ("calendar", "calendar__create_event", "write"),
        ("notion", "notion__create_page", "write"),
        ("slack", "slack__send_message", "write"),
        ("gmail", "gmail__create_draft", "write"),
    ):
        for count in (None, 0, 1, 7):
            out = verbs.turn_action(connector, tool, kind=kind, count=count)
            assert "{" not in out["action"], (connector, tool, out)
            assert "}" not in out["action"], (connector, tool, out)
            assert "{" not in out["detail"], (connector, tool, out)
            assert "}" not in out["detail"], (connector, tool, out)


def test_the_serializer_will_not_serve_a_slot():
    """`is_served_action` accepted `Held {when}` because C's templates
    compile to regexes with `{slot}` → `.+?` — so the one predicate
    meant to stop raw strings was an identity function for this class."""
    assert verbs.is_served_action("Held {when}") is False
    assert verbs.is_served_action("Commented on {target}") is False
    assert verbs.is_served_action("Checked your board") is True


def test_a_read_verb_reads_like_a_read():
    """R31-07: `0 issues moved` on a step that changed nothing.

    The count is real — issues that changed since the last look — and
    the verb is not. "Moved" is something the automation DID.
    """
    for count in (0, 1, 3):
        out = verbs.turn_action("jira", "jira__search_issues",
                                kind="read", count=count)
        assert "moved" not in out["detail"].lower(), out


# ── R31-34: the em dash ──────────────────────────────────────────────


def test_a_non_ascii_gmail_subject_round_trips():
    """R31-34, and the register's wording is wrong in a way that matters.

    The founder's inbox showed `R29-D live loop test Ã¢Â€Â" Gmail push`,
    which reads as a double encode. Measured, it is the opposite: the
    subject was written with a bare f-string, so U+2014 went onto the
    wire as three raw UTF-8 bytes inside a header. RFC 5322 headers are
    7-bit; the receiver reads them as Latin-1 and renders `â€"`, which
    something downstream mangles again. ZERO encodings, not two.
    """
    from email.header import decode_header, make_header
    from app.connectors.gmail.provider import _build_rfc822

    subject = "R29-D live loop test — Gmail push"
    raw = _build_rfc822(to="a@b.com", subject=subject, body="hi")
    header_line = next(
        ln for ln in raw.split("\r\n") if ln.startswith("Subject: ")
    )
    value = header_line[len("Subject: "):]

    # It IS encoded — the whole defect was that it was not.
    assert value.startswith("=?"), value
    # And it comes back the way it went in.
    assert str(make_header(decode_header(value))) == subject
    # The header block is 7-bit, which is the actual rule.
    header_block = raw.split("\r\n\r\n", 1)[0]
    header_block.encode("ascii")


def test_an_ascii_subject_is_left_alone():
    """Encoding a plain subject is correct and unreadable in every
    client's raw view, and this is the header users quote back."""
    from app.connectors.gmail.provider import _build_rfc822
    raw = _build_rfc822(to="a@b.com", subject="Weekly recap", body="x")
    assert "Subject: Weekly recap" in raw


# ── R31-33: the reminder that rang three months late ─────────────────


def test_a_long_past_reminder_never_rings():
    """"Time to call your brother — May 18 at 4:30 PM", ringing on 26
    August, with the alarm sound and quiet hours bypassed.

    The catch-up branch exists for a real case — an agent that rolled
    between a reminder's due time and now — and it had no age bound of
    any kind, was re-evaluated every ten minutes for the life of the
    process, and deduped on the LOCAL DATE, so a catch-up today minted
    a key the original day's claim could not block.
    """
    from app.agent.routines.runner import REMINDER_CATCH_UP_MAX

    now = datetime.now(timezone.utc)
    # The case the branch was written for: minutes late.
    assert (now - (now - timedelta(minutes=7))) <= REMINDER_CATCH_UP_MAX
    # The case that shipped: ninety-nine days late.
    assert (now - (now - timedelta(days=99))) > REMINDER_CATCH_UP_MAX
    # And the boundary is hours, not days — a reminder is a promise
    # about a moment.
    assert REMINDER_CATCH_UP_MAX < timedelta(days=1)


def test_the_catch_up_branch_reads_the_bound():
    """Drives the source, not the constant: a bound nothing consults is
    a comment."""
    import inspect
    from app.agent.routines import runner

    src = inspect.getsource(runner)
    assert "REMINDER_CATCH_UP_MAX" in src
    # The refusal must be BEFORE the reschedule, or it is decoration.
    guard = src.index("REMINDER_CATCH_UP_MAX", src.index("_too_old"))
    fire = src.index("trigger_tag = \"catch_up\"")
    assert guard < fire, (
        "the staleness check must gate the reschedule, not follow it"
    )


# ── R31-22: the setup card that said the run went fine ───────────────


def test_the_setup_card_does_not_announce_a_completed_run():
    """`notification_templates.notification_body` has no
    `automation_setup` branch, so a setup card fell through its status
    ladder to the completed-run line.

    Measured: `notification_body("automation_setup", {})` returns "It
    ran on time. Nothing needs you today — it is all there when you
    want it." — on a card that exists precisely because setup has NOT
    started. And because that string is non-empty, the A-owned fallback
    was unreachable: the wrong answer won by being confident rather
    than by being right.

    A seam that cannot say "I have nothing for this kind" needs its
    exceptions checked BEFORE it.
    """
    from app.agent.automations.run_v3 import _notification_body

    body = _notification_body("automation_setup", {})
    assert "ran on time" not in body, body
    assert "Nothing needs you today" not in body, body
    assert body.strip(), "a setup card with no body at all"

    # And the ordinary run body is untouched — the guard must be a
    # branch, not a replacement.
    run = _notification_body("automation_run", {
        "status": "completed", "vocabulary": "brief",
        "needs_count": 0, "run_kind": "scheduled",
    })
    assert "ran on time" in run, run


def test_a_stored_placeholder_cannot_reach_the_connector_card():
    """R31-25 at the READ boundary, from a live reading rather than a
    hypothesis.

    `_n` was made total so the engine can no longer MINT a slot-bearing
    string. But `workflow._last_use` serves the `detail` of a stored tool
    turn verbatim, and rows written by earlier builds are still there —
    so on 2026-08-26, against a platform whose renderer could not have
    produced it, `GET /api/accounts/jira/card` still answered:

        Checked your board · 0 issues moved · {need_count} needs you

    A fix at the write boundary does not reach data that was already
    written. The same rule now runs where the string is read.
    """
    from app.services.automation_verbs import drop_unfilled

    live = "Checked your board · 0 issues moved · {need_count} needs you"
    assert drop_unfilled(live) == "Checked your board · 0 issues moved"

    # A clean sentence is untouched — the guard must not eat real copy.
    clean = "Read your unread mail · 0 new threads"
    assert drop_unfilled(clean) == clean

    # A single unfillable clause says nothing rather than showing a brace.
    assert drop_unfilled("{need_count} needs you") == ""

    # And `_last_use` is the call site that actually put it on the card,
    # so it must be the one that applies the rule.
    # Anchored on the CALL, not on the name. A first version asserted
    # `"drop_unfilled" in src` and survived deleting both calls, because
    # the import line inside the function still carried the word — a
    # probe that reads its own explanation and calls it evidence.
    import inspect
    from app.agent.automations import workflow
    src = inspect.getsource(workflow._last_use)
    assert 'drop_unfilled(body.get("detail")' in src, (
        "_last_use serves the stored detail verbatim again"
    )
    assert 'drop_unfilled(body.get("action")' in src, (
        "_last_use serves the stored action verbatim again"
    )

    # BOTH read sites. `account_last_use` is a second, independent copy of
    # the same loop and it is the one that served the founder's card — the
    # first pass at this fix guarded the workflow's twin and left the
    # observed defect in place.
    from app.api import automations as _api
    api_src = inspect.getsource(_api.account_last_use)
    assert 'drop_unfilled(body.get("detail")' in api_src, (
        "account_last_use serves the stored detail verbatim again"
    )
    assert 'drop_unfilled(body.get("action")' in api_src, (
        "account_last_use serves the stored action verbatim again"
    )


def test_a_clean_use_writes_no_remedy():
    """`record_use` is the WRITER, and a wrong value written once outlives
    every read-side fix: `state_for` prefers the stored `row.fix`.

    A successful tool call clears the account; a cleared account has
    nothing to retry.
    """
    import inspect
    from app.agent.automations import account_health

    assert account_health.fix_for("connected", "") == ""
    assert account_health.fix_for("connected", "timeout") == "retry"
    assert account_health.fix_for("expired", "token_expired") == "reconnect"

    src = inspect.getsource(account_health.record_use)
    assert '("connected", "retry")' not in src, (
        "record_use stamps a blanket retry on a healthy account again"
    )
