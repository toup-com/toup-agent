"""`plan_uploaded_attachments` — the turn's attachment decision, driven.

Round 46 review. The 223-line inbound block in `ws_chat.py` decided the count
cap, the per-turn byte budget, the rejection list and the ingest report, and
its only test asserted that three reason STRINGS appeared somewhere in the
source. Every inversion inside it passed: `+` for `-` on `_turn_bytes`, `<` for
`>` on the budget, a rejection stamped at the wrong index. The decision is now
one pure function (resolution injected) and this file executes it.

The defect the index assertions pin: accepted items were numbered by their
position in the ACCEPTED list and rejections went out at `index: -1`, so a
single rejected file painted every later file's status onto the wrong row and
the rejection itself reached the client as `sent[-1]` — undefined, i.e. no
status at all. Platform sweep.
"""

from __future__ import annotations

import pytest

from app.agent.attachment_limits import (
    MAX_ATTACHMENTS_PER_TURN,
    MAX_TOTAL_BYTES_PER_TURN,
    REASON_TOTAL_TOO_LARGE,
)
from app.api.chat_attachments import plan_uploaded_attachments


def _rec(name: str, size: int, status: str = "ok", reason=None):
    return {
        "name": name,
        "status": status,
        "reason": reason,
        "attachment": {"id": name, "size_bytes": size},
    }


def _store(table):
    """`resolve(attachment_id) -> record | None`, from a dict."""
    return lambda aid: table.get(aid)


def test_everything_resolvable_is_accepted_in_order():
    table = {f"a{i}": _rec(f"f{i}.pdf", 10) for i in range(3)}
    recs, idx, rej = plan_uploaded_attachments(["a0", "a1", "a2"], _store(table))
    assert [r["name"] for r in recs] == ["f0.pdf", "f1.pdf", "f2.pdf"]
    assert idx == [0, 1, 2]
    assert rej == []


def test_the_count_cap_refuses_the_overflow_at_its_own_position():
    n = MAX_ATTACHMENTS_PER_TURN + 1
    table = {f"a{i}": _rec(f"f{i}.pdf", 1) for i in range(n)}
    recs, idx, rej = plan_uploaded_attachments([f"a{i}" for i in range(n)], _store(table))
    assert len(recs) == MAX_ATTACHMENTS_PER_TURN
    assert idx == list(range(MAX_ATTACHMENTS_PER_TURN))
    assert [r["reason"] for r in rej] == ["attachment_count_exceeded"]
    assert rej[0]["index"] == MAX_ATTACHMENTS_PER_TURN


def test_the_per_turn_byte_budget_is_enforced_here_and_never_exceeded():
    """Eight already-uploaded 25 MB documents entered one turn on a 768 MiB
    container through this branch: the budget was declared in the table and
    enforced only by the two CLIENTS."""
    half = MAX_TOTAL_BYTES_PER_TURN // 2
    table = {
        "a": _rec("one.pdf", half),
        "b": _rec("two.pdf", half),
        "c": _rec("three.pdf", half),
    }
    recs, idx, rej = plan_uploaded_attachments(["a", "b", "c"], _store(table))
    assert [r["name"] for r in recs] == ["one.pdf", "two.pdf"]
    assert sum(r["attachment"]["size_bytes"] for r in recs) <= MAX_TOTAL_BYTES_PER_TURN
    assert [r["reason"] for r in rej] == [REASON_TOTAL_TOO_LARGE]
    assert rej[0]["index"] == 2
    assert rej[0]["name"] == "three.pdf"


def test_a_file_under_the_remaining_budget_after_a_refusal_still_gets_in():
    """The budget is cumulative over what was ACCEPTED, not a latch: refusing
    one oversized file must not refuse everything behind it."""
    table = {
        "big": _rec("big.pdf", MAX_TOTAL_BYTES_PER_TURN // 2),
        "huge": _rec("huge.pdf", MAX_TOTAL_BYTES_PER_TURN),
        "small": _rec("small.pdf", 100),
    }
    recs, idx, rej = plan_uploaded_attachments(["big", "huge", "small"], _store(table))
    assert [r["name"] for r in recs] == ["big.pdf", "small.pdf"]
    assert idx == [0, 2], "the accepted file keeps the index the client gave it"
    assert [r["index"] for r in rej] == [1]


def test_an_unresolvable_id_is_refused_at_its_own_index():
    table = {"a": _rec("one.pdf", 1), "c": _rec("three.pdf", 1)}
    recs, idx, rej = plan_uploaded_attachments(["a", "missing", "c"], _store(table))
    assert idx == [0, 2]
    assert rej == [{"index": 1, "name": "", "reason": "attachment_not_found"}]


def test_a_rejection_never_shifts_a_later_accepted_row():
    """The defect, stated as the test: with one rejection in the middle, the
    THIRD file's status must be reported against index 2."""
    table = {"a": _rec("one.pdf", 1), "c": _rec("three.pdf", 1)}
    recs, idx, rej = plan_uploaded_attachments(["a", "gone", "c"], _store(table))
    report = [
        {"index": idx[i], "name": r["name"], "status": r["status"]}
        for i, r in enumerate(recs)
    ] + [{"index": r["index"], "name": r["name"], "status": "rejected"} for r in rej]
    assert {e["index"]: e["name"] for e in report} == {
        0: "one.pdf", 1: "", 2: "three.pdf",
    }


def test_a_durable_ingest_status_survives_the_plan():
    """A password-protected PDF is ACCEPTED (the turn still runs) and carries
    its status, so the user is told why it could not be read."""
    table = {"a": _rec("locked.pdf", 10, status="password_protected",
                       reason="password_protected")}
    recs, idx, rej = plan_uploaded_attachments(["a"], _store(table))
    assert rej == []
    assert recs[0]["status"] == "password_protected"


@pytest.mark.parametrize("junk", [None, 7, "", {}, []])
def test_a_non_string_id_is_skipped_without_consuming_a_slot(junk):
    table = {"a": _rec("one.pdf", 1)}
    recs, idx, rej = plan_uploaded_attachments([junk, "a"], _store(table))
    assert [r["name"] for r in recs] == ["one.pdf"]
    assert idx == [1], "the surviving file keeps the client's own position"
    assert rej == []


def test_a_missing_size_does_not_make_the_budget_unusable():
    """`size_bytes` is absent or a string on a record whose blob vanished; the
    plan must not raise and must not treat it as the whole budget."""
    table = {"a": {"name": "x.pdf", "status": "ok", "attachment": {"size_bytes": "nope"}},
             "b": {"name": "y.pdf", "status": "ok"}}
    recs, idx, rej = plan_uploaded_attachments(["a", "b"], _store(table))
    assert len(recs) == 2 and rej == []
