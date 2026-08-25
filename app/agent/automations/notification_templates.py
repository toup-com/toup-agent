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


def _opening(run_kind: str) -> str:
    return "Done — it ran just now." if run_kind == "run_now" else "It ran on time."


def _brief_body(run_kind: str, needs_count: int) -> str:
    opening = _opening(run_kind)
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


def _changes_body(run_kind: str, writes_count: int) -> str:
    opening = _opening(run_kind)
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
        if connector:
            return (
                f"{connector} access ran out, so it stopped where it was. "
                "Reconnect and it picks up from there."
            )
        return "Something needs you — open the run and I will show you."

    if status == "failed":
        who = connector or "an account"
        return (
            f"It could not finish — {who} refused. Nothing was missed. "
            "Open the run and I will show you the fix."
        )

    if status == "waiting_on_user":
        return (
            "It prepared a change and is waiting on you — "
            "nothing happens until you approve."
        )

    if vocabulary == "changes":
        return _changes_body(run_kind, int(run_summary.get("writes_count") or 0))
    return _brief_body(run_kind, int(run_summary.get("needs_count") or 0))


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
