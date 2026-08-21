"""Round 13 — one title shape, whichever producer wrote it.

`job_titles` is what stops a runner-derived voice title and a model-written
chat title from reading as two different products on the same screen. Pure
functions, no DB — this file runs in the platform sweep.
"""
from __future__ import annotations

import pytest

from app.agent.job_titles import (
    JOB_TITLE_MAX,
    derive_job_title,
    is_rtl_text,
    normalize_job_title,
)


# ── normalize: the shape both paths land in ─────────────────────────────

@pytest.mark.parametrize("raw,want", [
    ("Find the strongest image-generation model.", "Find the strongest image-generation model"),
    ("**Research the newest model**", "Research the newest model"),
    ('"Find UofT LLM professors"', "Find UofT LLM professors"),
    ("«بررسی بهترین مدل»", "بررسی بهترین مدل"),
    ("Compare  the   benchmarks\n\nagain", "Compare the benchmarks again"),
    ("Ship it!", "Ship it"),
])
def test_the_card_shape(raw, want):
    assert normalize_job_title(raw) == want


def test_a_question_keeps_its_question_mark():
    """"?" is meaning, not decoration — a job that IS a question reads as one."""
    assert normalize_job_title("Which model is strongest?") == "Which model is strongest?"


def test_normalize_is_idempotent():
    once = normalize_job_title("  **Find the model.**  ")
    assert normalize_job_title(once) == once == "Find the model"


def test_normalize_rejects_what_is_not_a_title():
    for junk in (None, "", "   ", "ok", 42, ["a"]):
        assert normalize_job_title(junk) == ""


def test_a_long_title_is_clipped_on_a_word_boundary():
    long = ("Find the strongest image generation model for product photography "
            "and short-form video work across every provider")
    assert len(long) > JOB_TITLE_MAX
    out = normalize_job_title(long)
    assert len(out) <= JOB_TITLE_MAX + 1  # +1 for the ellipsis
    assert out.endswith("…")
    # Clipped between words, never mid-word.
    assert long.startswith(out[:-1])
    assert out[-2] != " "


# ── derive: an utterance becomes a title ────────────────────────────────

@pytest.mark.parametrize("said,want", [
    ("hey, can you please find the strongest image-generation model thanks",
     "Find the strongest image-generation model"),
    ("ok so I want you to compare GPT-5.6 and Claude on coding.",
     "Compare GPT-5.6 and Claude on coding"),
    ("could you look up the UofT LLM professors",
     "Look up the UofT LLM professors"),
    ("let's check my email", "Check my email"),
    ("please summarize today's AI news", "Summarize today's AI news"),
])
def test_the_spoken_scaffolding_comes_off(said, want):
    assert derive_job_title(said) == want


def test_product_names_survive_sentence_casing():
    """`.title()` would ship "Gpt-5.6" to a lock screen. Only the first
    character is ever touched."""
    assert derive_job_title("check if GPT-5.6 beats Claude") == "Check if GPT-5.6 beats Claude"


def test_a_persian_request_keeps_its_verb():
    """The founder speaks Persian to this surface. "سرچ کن" (search) is the
    VERB — only the politeness opener may be removed."""
    out = derive_job_title("لطفا سرچ کن ببین بهترین مدل تولید تصویر کدومه")
    assert out.startswith("سرچ کن")
    assert "لطفا" not in out


def test_persian_is_never_sentence_cased_or_reordered():
    said = "بهترین مدل تولید تصویر کدومه"
    assert derive_job_title(said) == said


def test_rtl_detection():
    assert is_rtl_text("سلام") and is_rtl_text("mixed سلام text")
    assert not is_rtl_text("hello") and not is_rtl_text("")


def test_an_acknowledgement_is_not_a_job():
    """The card must not be minted for "ok" — noise on this surface is the
    failure the voice path already had once."""
    for said in ("ok", "yes", "yeah!", "sure.", "thanks", "باشه", "", None, "  "):
        assert derive_job_title(said) == ""
        assert derive_job_title(said, fallback="Voice request") == "Voice request"


def test_an_acknowledgement_check_does_not_reach_the_tool_path():
    """`create_job`'s contract is "any non-empty title is legal" — the
    ack list is a DERIVE-only rule and must not start rejecting a model's
    deliberately terse title."""
    assert normalize_job_title("Done") == "Done"
    assert normalize_job_title("Yes") == "Yes"


def test_the_fallback_is_normalised_too():
    assert derive_job_title("", fallback="  **Voice request.**  ") == "Voice request"


def test_stacked_openers_all_come_off():
    assert derive_job_title(
        "um ok so could you please go ahead and book the flight, thanks"
    ) == "Book the flight"


def test_derive_never_raises_on_anything():
    for junk in (None, 0, [], {}, b"bytes", "\x00\x00"):
        derive_job_title(junk)  # must not raise
