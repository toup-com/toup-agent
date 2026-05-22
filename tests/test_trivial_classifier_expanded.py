"""Regression tests for the TKT-LAT-019 trivial-query classifier expansion.

Verifies the broadened pattern set still rejects non-trivial messages.
Strictly additive vs the original test — these patterns weren't covered
in the initial PR.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _qc():
    p = Path(__file__).resolve().parents[1] / "app" / "services" / "query_classifier.py"
    spec = importlib.util.spec_from_file_location("qc_ext", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_new_acknowledgments():
    qc = _qc()
    for w in ["gotcha", "noted", "understood", "copy", "copy that", "roger", "roger that"]:
        assert qc.is_trivial_query(w), f"{w!r} should be trivial"


def test_extended_thanks():
    qc = _qc()
    for w in ["tysm", "much appreciated", "appreciate it", "cheers"]:
        assert qc.is_trivial_query(w), f"{w!r} should be trivial"


def test_affirmations():
    qc = _qc()
    for w in [
        "lgtm", "wfm", "sounds good", "looks good", "no worries", "np",
        "no problem", "all good", "of course", "ofc", "exactly", "fair enough",
        "love it", "love that", "amazing", "brilliant",
    ]:
        assert qc.is_trivial_query(w), f"{w!r} should be trivial"


def test_extended_greetings_and_farewells():
    qc = _qc()
    for w in ["howdy", "ttyl", "catch you later", "take care", "peace", "see ya"]:
        assert qc.is_trivial_query(w), f"{w!r} should be trivial"


def test_reactions():
    qc = _qc()
    for w in ["hahaha", "lmfao", "yikes", "oof", "huh", "hmm"]:
        assert qc.is_trivial_query(w), f"{w!r} should be trivial"


def test_punctuation_variants_on_new_patterns():
    qc = _qc()
    assert qc.is_trivial_query("lgtm!")
    assert qc.is_trivial_query("no worries.")
    assert qc.is_trivial_query("  tysm  ")
    assert qc.is_trivial_query("OK.")


def test_nontrivial_when_substring_present_in_longer_message():
    """Critical: phrases that contain trivial words but carry real
    intent must NOT be classified as trivial. The 6-word cap + exact
    match are what prevent this."""
    qc = _qc()
    for q in [
        "tysm for the help with the migration",
        "no worries about the deadline tho",
        "love it but can you make it bigger?",
        "noted that the build is failing",
        "lgtm — can we ship today?",
        "of course but only after we test",
    ]:
        assert not qc.is_trivial_query(q), (
            f"{q!r} should NOT be trivial — has real intent past the trivial word"
        )


def test_case_insensitivity_on_new_patterns():
    qc = _qc()
    for variant in ["LGTM", "Tysm", "NoTeD", "Hahaha"]:
        assert qc.is_trivial_query(variant), f"{variant!r} should be trivial (case-insensitive)"
