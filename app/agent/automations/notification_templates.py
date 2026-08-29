"""Notification bodies — one string, three surfaces (R30 §4.10, §5.7).

The dispatcher calls `notification_body(kind, run_summary)` at run
completion; the SAME string lands on the in-chat notification card, the
push banner and the live-activity end state — byte-identical, so the
banner and the card never disagree. A body is a count and an
invitation, never a finding: what the run learned stays in the thread.

`run_summary` (engine-served, CONTRACTS-R30 §5.7):
    run_kind               "scheduled" | "run_now"
    status                 v3 run status
    vocabulary             "brief" | "changes"
    needs_count            rows in tiers 1–2 (brief) /
                           pending waiting turns (changes)
    writes_count           rows in the honest write ledger
    failed_connector_name  display name, failed runs only
    failed_reason_body     the classified reason's notification_body
                           string (R36-6), resolved by run_v3 from the
                           same table the thread's needs-you card reads

Every string here passes the copy guard; `test_notification_templates`
pins that, the canvas line, and the count grammar.
"""

from __future__ import annotations

_NUMBER_WORDS = {
    1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five",
    6: "Six", 7: "Seven", 8: "Eight", 9: "Nine",
}


def _count_word(n: int) -> str:
    """Capitalised number word for one–nine, digits above (§5.7)."""
    return _NUMBER_WORDS.get(n, str(n))


def _opening(run_kind: str, status: str = "") -> str:
    # A run that got part of the way must not open by claiming a whole
    # one — the count sentences that follow are still true of it.
    if status == "partial":
        return "It got part of the way."
    return "Done — it ran just now." if run_kind == "run_now" else "It ran on time."


def _brief_body(run_kind: str, needs_count: int, status: str = "") -> str:
    opening = _opening(run_kind, status)
    if needs_count <= 0:
        return f"{opening} Nothing needs you today — it is all there when you want it."
    if needs_count == 1:
        return (
            f"{opening} One thing needs you today — open the run and "
            "I will walk you through it there."
        )
    return (
        f"{opening} {_count_word(needs_count)} things need you today — "
        "open the run and I will walk you through them there."
    )


def _changes_body(run_kind: str, writes_count: int,
                  status: str = "") -> str:
    opening = _opening(run_kind, status)
    if writes_count == 1:
        return (
            f"{opening[:-1]} and made one change — open the run to see it, "
            "and what it left alone on purpose."
        )
    return (
        f"{opening[:-1]} and made {_count_word(writes_count).lower()} changes — "
        "open the run to see each one, ranked by what you may want to undo."
    )


def notification_body(kind: str, run_summary: dict) -> str:
    """The §4.10 `body`. `kind` is the notification kind
    (`automation_run` | `automation_needs_you`); the run's situation is
    read from `run_summary`."""
    run_kind = str(run_summary.get("run_kind") or "scheduled")
    status = str(run_summary.get("status") or "")
    vocabulary = str(run_summary.get("vocabulary") or "brief")
    connector = str(run_summary.get("failed_connector_name") or "").strip()

    if kind == "automation_needs_you":
        if status == "waiting_on_user":
            return (
                "It prepared a change and is waiting on you — "
                "nothing happens until you approve."
            )
        # R36-6: the run's own classified reason, resolved by run_v3
        # from the same (state, reason) table the thread reads. This
        # branch used to assert "access ran out" for EVERY named
        # connector — a specific mechanism it could not know — so one
        # Gmail failure read "it did not tell me why" in the thread and
        # "access ran out, reconnect" on the card. When no reason was
        # recorded, say where the real one is written, never invent one.
        body = str(run_summary.get("failed_reason_body") or "").strip()
        if body:
            return body
        if connector:
            return (
                f"It stopped at {connector} — the reason is written in "
                "the thread. Open it and I will show you the fix."
            )
        return "Something needs you — open the run and I will show you."

    if status == "failed":
        # R31-07. This was `who = connector or "an account"` — so a run
        # that failed for a reason no account owns (a drain, the run
        # cap, a crash) told the user "an account refused", naming
        # nobody and blaming something that had not happened. There is
        # no fix to offer for that, either, so the invitation changes
        # with it. When a connector IS named, R30 §5.7's line stands
        # verbatim.
        if connector:
            # R31-07. This said "{connector} refused", which is true of
            # exactly one failure and wrong about the rest: access that
            # ran out did not refuse, an organisation that has not
            # approved Toup did not refuse, and a service that timed out
            # did not refuse. `run_summary` carries the NAME but not the
            # reason, so the honest body names the account and sends the
            # user to where the real reason is written, rather than
            # inventing a mechanism to fill the gap. (The canvas does
            # not mandate this line — it was R30 §5.7's own wording.)
            return (
                f"It could not finish — {connector} needs you. Nothing "
                "was missed. Open the run and I will show you the fix."
            )
        return (
            "It could not finish. Nothing was missed. Open the run and "
            "I will show you what happened."
        )

    if status == "waiting_on_user":
        return (
            "It prepared a change and is waiting on you — "
            "nothing happens until you approve."
        )

    # `run_v3_status` returns eight statuses. This table knew three of
    # them and let the rest fall through to the count sentences, whose
    # opening is "It ran on time." So a run the user STOPPED, a run a
    # newer one SUPERSEDED, and a run that found nothing to do each
    # pushed a banner claiming a clean on-time run — on the one surface
    # the user sees without opening anything. `partial` keeps its counts
    # (they are true of it) but loses the whole-run opening.
    if status == "stopped_by_user":
        return (
            "You stopped it, so nothing else went out. "
            "Open the run to see how far it got."
        )
    if status == "superseded":
        return (
            "A newer run replaced this one. "
            "Open the newer run to see what happened."
        )
    if status == "skipped":
        # `skipped` covers BOTH "the user declined/let it expire"
        # (confirm.py maps rejected AND expired to cancelled+skipped) and
        # "there was nothing to do". Claiming the second at a user who
        # just tapped "Not now" contradicts the run's own note, so this
        # stays neutral; the engine prefers the run's recorded sentence
        # when it has one.
        return (
            "It ended without making any changes. "
            "Open the run to see why."
        )

    if vocabulary == "changes":
        return _changes_body(
            run_kind, int(run_summary.get("writes_count") or 0), status)
    return _brief_body(
        run_kind, int(run_summary.get("needs_count") or 0), status)


def draft_staged_body() -> str:
    """The proactive-flow draft notification (§5.7)."""
    return "A draft is waiting — nothing has been sent. Open the run to read it."


def auto_pause_body() -> str:
    """The three-strikes auto-pause notice — replaces the live string
    D's rig caught wearing an emoji and markdown bold ("⚠️ **Live
    schedule** was paused after 3 failed runs in a row.")."""
    return (
        "It failed three times in a row, so I paused it. "
        "Open the run and I will show you what went wrong."
    )


def setup_card(title: str) -> dict:
    """The `automation_setup` day-chat card (§3.3, §4.10)."""
    return {"title": f"Setting up: {title}", "body": "Continue setting it up ›"}
