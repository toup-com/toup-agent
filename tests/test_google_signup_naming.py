"""
PR onboarding-v2/1 — Google signup display-name extraction.

`_pick_google_display_name` is the helper that takes a Google userinfo
payload + the user's email and returns the string we store in
``users.name``. It must:

  * prefer the structured ``given_name`` + ``family_name`` pair so
    locales that order surname-first keep Google's ordering;
  * fall back to the unstructured ``name`` field if neither half is
    present;
  * fall back to the email local-part if Google returned no name fields
    at all (rare but possible — older OAuth scopes, deleted profile);
  * NEVER transliterate. Persian / Arabic / CJK names must round-trip
    UTF-8 to the admin panel and chat sidebar exactly as Google sent
    them. Only sanitisation is whitespace trimming and a length cap.

These are pure-function tests — no DB, no HTTP, no fixtures.
"""

from __future__ import annotations

from app.api.auth import _pick_google_display_name


# ── Happy paths: structured given/family pair ────────────────────────

def test_picks_given_plus_family_when_both_present():
    info = {"given_name": "Ada", "family_name": "Lovelace", "name": "should-not-win"}
    assert _pick_google_display_name(info, "ada@example.com") == "Ada Lovelace"


def test_picks_given_only_when_family_missing():
    info = {"given_name": "Cher"}
    assert _pick_google_display_name(info, "cher@example.com") == "Cher"


def test_picks_family_only_when_given_missing():
    # Hungarian/Japanese-style single surname.
    info = {"family_name": "Kovács"}
    assert _pick_google_display_name(info, "k@example.com") == "Kovács"


# ── Fallback ladder ─────────────────────────────────────────────────

def test_falls_back_to_name_when_structured_pair_missing():
    info = {"name": "  Grace Hopper  "}
    assert _pick_google_display_name(info, "gh@example.com") == "Grace Hopper"


def test_falls_back_to_email_local_part_when_no_name_fields():
    info: dict = {}
    assert _pick_google_display_name(info, "alan.turing@example.com") == "alan.turing"


def test_falls_back_to_email_local_part_when_all_fields_blank():
    info = {"given_name": "  ", "family_name": "", "name": "   "}
    assert _pick_google_display_name(info, "ceo@ete.bar") == "ceo"


def test_returns_user_when_email_also_missing():
    # Defence-in-depth: the callback rejects empty emails before reaching
    # this helper, but we still want a non-empty result so `users.name`
    # never lands NULL.
    assert _pick_google_display_name({}, "") == "User"
    assert _pick_google_display_name({}, "@no-local-part.example") == "User"


# ── UTF-8 preservation (no transliteration) ──────────────────────────

def test_persian_script_preserved_end_to_end():
    info = {"given_name": "نریمان", "family_name": "هیئت‌مقدم"}
    out = _pick_google_display_name(info, "nariman@toup.ai")
    assert out == "نریمان هیئت‌مقدم"
    # Explicit byte-level check that we did not ASCII-fold.
    assert "ن" in out
    assert all(ord(c) > 127 for c in out.replace(" ", ""))


def test_mixed_script_preserved():
    info = {"given_name": "علی", "family_name": "Smith"}
    assert _pick_google_display_name(info, "x@y.z") == "علی Smith"


def test_cjk_preserved():
    info = {"given_name": "明", "family_name": "陳"}
    assert _pick_google_display_name(info, "ming@example.com") == "明 陳"


# ── Sanitisation: whitespace, length ─────────────────────────────────

def test_strips_surrounding_whitespace_in_each_half():
    info = {"given_name": "  Marie  ", "family_name": "\tCurie\n"}
    assert _pick_google_display_name(info, "mc@example.com") == "Marie Curie"


def test_whitespace_only_fields_treated_as_empty():
    # Counts as "given absent" → falls through to `name`.
    info = {"given_name": "   ", "family_name": "\t", "name": "Real Name"}
    assert _pick_google_display_name(info, "x@y.z") == "Real Name"


def test_email_local_part_capped_to_64_chars():
    # Gmail caps locals at 64, but other providers can be longer.
    long_local = "a" * 100
    out = _pick_google_display_name({}, f"{long_local}@example.com")
    assert len(out) == 64
    assert out == "a" * 64
