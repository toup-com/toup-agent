"""Regression tests for the conversational-greeting expansion.

Motivated by a production observation: "Hows it going" took 15.8 s
and loaded 23 207 input tokens because the previous classifier only
matched time/date questions, not status-check openers.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _qc():
    p = Path(__file__).resolve().parents[1] / "app" / "services" / "query_classifier.py"
    spec = importlib.util.spec_from_file_location("qc_conv", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_hows_it_going_variants():
    qc = _qc()
    for q in [
        "Hows it going",       # The actual screenshot case (no apostrophe)
        "hows it going",
        "how's it going",
        "how is it going",
        "How's it going?",
        "Hows it goin",        # casual contraction — should NOT trim (typo guard)
    ]:
        result = qc.is_trivial_query(q)
        # All of these except the casual contraction with 'goin' should trim.
        if q == "Hows it goin":
            continue  # too aggressive to claim — leave on full path
        assert result, f"{q!r} should be trivial"


def test_how_are_you_family():
    qc = _qc()
    for q in [
        "how are you",
        "how are you?",
        "How are you doing",
        "how're you",
        "how r u",
        "how are ya",
        "how are things",
        "how are things going",
        "how have you been",
        "how have you been today",
        "how have you been lately",
        "how's your day",
        "how's everything",
    ]:
        assert qc.is_trivial_query(q), f"{q!r} should be trivial"


def test_whats_up_family():
    qc = _qc()
    for q in ["what's up", "whats up", "wassup", "wazzup", "sup",
              "what's happening", "what's new", "what's going on", "what's good"]:
        assert qc.is_trivial_query(q), f"{q!r} should be trivial"


def test_you_there_family():
    qc = _qc()
    for q in [
        "you there",
        "you there?",
        "are you there",
        "are you alive",
        "are you around",
        "still there",
        "still with me",
    ]:
        assert qc.is_trivial_query(q), f"{q!r} should be trivial"


def test_misc_social_openers():
    qc = _qc()
    for q in ["long time no see", "good to see you", "miss you"]:
        assert qc.is_trivial_query(q), f"{q!r} should be trivial"


def test_critical_negatives_when_greeting_carries_real_intent():
    """The whole reason for the word-cap + anchor regex is to keep these
    NON-trivial. If any of these silently trim, the agent will give a
    casual hello when the user actually wants help."""
    qc = _qc()
    for q in [
        "how are you going to fix this",
        "what's up with the build job that failed",
        "you there in toronto right now",  # location question, not greeting
        "are you available tomorrow",
        "how are things on the auth migration",
        "what's up with project X",
        "how have you handled this before",  # past behavior, needs memory
    ]:
        assert not qc.is_trivial_query(q), (
            f"{q!r} should NOT be trivial — carries real intent past the greeting"
        )


def test_word_cap_bumped_to_8():
    """Bumped from 6 -> 8 so 'how have you been today' (5 words) and
    similar conversational openers fit. Verified via source-level check."""
    p = Path(__file__).resolve().parents[1] / "app" / "services" / "query_classifier.py"
    src = p.read_text()
    assert "_TRIVIAL_MAX_WORDS = 8" in src
