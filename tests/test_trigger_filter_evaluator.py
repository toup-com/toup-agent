"""Unit tests for the email_received trigger filter evaluator.

Pure-function tests — no DB, no async. Validates the AND-across-fields,
OR-within-list semantics plus the Gmail category alias normalisation.

The filter evaluator is the production decision point for which
emails the user actually wants to know about. A regression here =
floods of unwanted notifications OR silent drops of wanted ones,
both bad. Tests pin the load-bearing cases:
  - empty / None filter → match-all
  - from / subject substring case-insensitive
  - labels intersection
  - category aliases ("promotions" → "CATEGORY_PROMOTIONS")
  - AND across fields, OR within lists
"""

from __future__ import annotations

from app.agent.triggers.filter_evaluator import matches_filter


def _msg(
    *,
    from_: str = "alice@example.com",
    subject: str = "Hello",
    labels: list[str] | None = None,
) -> dict:
    """Helper to build a Gmail message dict with the bits the
    evaluator reads."""
    return {
        "payload": {
            "headers": [
                {"name": "From", "value": from_},
                {"name": "Subject", "value": subject},
            ]
        },
        "labelIds": labels or ["INBOX"],
        "snippet": "",
    }


# ── Match-all paths ──────────────────────────────────────────────────


def test_empty_filter_matches():
    ok, reason = matches_filter(_msg(), {})
    assert ok and reason == "ok"


def test_none_filter_matches():
    ok, reason = matches_filter(_msg(), None)
    assert ok and reason == "ok"


def test_garbage_filter_fails_open():
    """Corrupted filter_json (someone wrote a string) → match-all with
    a warning. Caller logs; we don't drop the user's emails because
    we couldn't parse our own config."""
    ok, _ = matches_filter(_msg(), "not a dict")  # type: ignore[arg-type]
    assert ok is True


# ── From contains ───────────────────────────────────────────────────


def test_from_contains_match_substring():
    ok, reason = matches_filter(
        _msg(from_="Noreply <noreply@stripe.com>"),
        {"from_contains": ["stripe.com"]},
    )
    assert ok is True
    assert reason == "ok"


def test_from_contains_case_insensitive():
    ok, _ = matches_filter(
        _msg(from_="ALICE@EXAMPLE.com"),
        {"from_contains": ["alice"]},
    )
    assert ok is True


def test_from_contains_or_within_list():
    """List items are OR'd — match if ANY substring hits."""
    ok, _ = matches_filter(
        _msg(from_="boss@company.com"),
        {"from_contains": ["nope", "company"]},
    )
    assert ok is True


def test_from_contains_no_match_returns_reason():
    ok, reason = matches_filter(
        _msg(from_="random@elsewhere.io"),
        {"from_contains": ["company.com", "stripe"]},
    )
    assert ok is False
    assert reason == "no_from_match"


# ── Subject contains ────────────────────────────────────────────────


def test_subject_contains_match():
    ok, _ = matches_filter(
        _msg(subject="Your invoice for May"),
        {"subject_contains": ["invoice"]},
    )
    assert ok is True


def test_subject_contains_no_match():
    ok, reason = matches_filter(
        _msg(subject="Newsletter: weekly digest"),
        {"subject_contains": ["invoice", "receipt"]},
    )
    assert ok is False
    assert reason == "no_subject_match"


# ── Labels intersection ─────────────────────────────────────────────


def test_labels_match_any():
    ok, _ = matches_filter(
        _msg(labels=["INBOX", "Label_42"]),
        {"labels": ["IMPORTANT", "Label_42"]},
    )
    assert ok is True


def test_labels_no_intersection():
    ok, reason = matches_filter(
        _msg(labels=["INBOX"]),
        {"labels": ["IMPORTANT"]},
    )
    assert ok is False
    assert reason == "no_label_match"


# ── Category exclude ────────────────────────────────────────────────


def test_excludes_promotions_via_alias():
    """User specifies `"promotions"` (the short form); the evaluator
    translates to `CATEGORY_PROMOTIONS` and matches."""
    ok, reason = matches_filter(
        _msg(labels=["INBOX", "CATEGORY_PROMOTIONS"]),
        {"exclude_categories": ["promotions"]},
    )
    assert ok is False
    assert reason == "excluded_category"


def test_excludes_multiple_categories():
    ok, reason = matches_filter(
        _msg(labels=["INBOX", "CATEGORY_SOCIAL"]),
        {"exclude_categories": ["promotions", "social"]},
    )
    assert ok is False
    assert reason == "excluded_category"


def test_does_not_exclude_when_no_category_label():
    ok, _ = matches_filter(
        _msg(labels=["INBOX"]),
        {"exclude_categories": ["promotions"]},
    )
    assert ok is True


def test_raw_category_label_id_also_works():
    """Power users can paste the raw label id; should still match."""
    ok, _ = matches_filter(
        _msg(labels=["INBOX", "CATEGORY_UPDATES"]),
        {"exclude_categories": ["CATEGORY_UPDATES"]},
    )
    assert ok is False


# ── AND across fields ───────────────────────────────────────────────


def test_all_fields_must_match():
    """All present fields are AND'd."""
    filt = {
        "from_contains": ["stripe"],
        "subject_contains": ["invoice"],
        "labels": ["IMPORTANT"],
    }
    # from + subject match but labels don't → fail
    ok, reason = matches_filter(
        _msg(
            from_="stripe@stripe.com",
            subject="Your invoice is ready",
            labels=["INBOX"],   # no IMPORTANT
        ),
        filt,
    )
    assert ok is False
    assert reason == "no_label_match"

    # All three match
    ok, reason = matches_filter(
        _msg(
            from_="stripe@stripe.com",
            subject="Your invoice is ready",
            labels=["INBOX", "IMPORTANT"],
        ),
        filt,
    )
    assert ok is True and reason == "ok"


def test_blank_strings_in_list_ignored():
    """User passes [""] — should be normalised away, not match
    everything by accident."""
    ok, reason = matches_filter(
        _msg(from_="alice@example.com"),
        {"from_contains": ["", "  "]},  # all blank
    )
    # All entries got stripped → empty list → no constraint → match
    assert ok is True


def test_single_string_value_treated_as_one_element():
    """Forgiving on input — accept a bare string where a list was
    expected, treat as [s]."""
    ok, _ = matches_filter(
        _msg(from_="boss@company.com"),
        {"from_contains": "company"},  # type: ignore[dict-item]
    )
    assert ok is True
