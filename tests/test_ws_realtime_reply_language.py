"""The reply-language contract, both directions.

A Farsi turn must not come back in English (round nine's rule, still pinned
below). And — round ten's P0 — an ENGLISH turn must not come back in Farsi.

The second failure is the interesting one, because the mechanism ran entirely
through code that looked correct. The account belongs to a Persian speaker, so
the session's transcription prompt asserted Persian; English audio under that
prompt came back rendered in Persian script; the reinforcement read the
script off that transcript and issued a STANDING "reply in Persian"
instruction; and the model obeyed the instruction over its own ears for the
rest of the call.

Every assertion here exists to keep one of those steps from coming back:

  * the decision consults BOTH halves of a turn, never the transcript alone;
  * it never carries a previous turn's verdict into the current one;
  * a language earns the benefit of the doubt only after this call has
    corroborated it once;
  * a user's explicit pin outranks all of it;
  * and the inferred transcription prompt describes a bilingual speaker
    instead of asserting one language.
"""
import inspect

from app.api import ws_realtime
from app.api.ws_realtime import (
    _LATIN,
    next_reply_directive,
    script_evidence,
    transcription_prompt,
)

FA = "سلام حالت چطوره امروز چه خبر"
FA_SHORT = "سلام"
EN = "search me what is the strongest model for video generating?"
EN_SHORT = "what time is it?"


# ── script_evidence: "no claim" is a real third answer ────────────────────

def test_script_evidence_reads_a_full_sentence():
    assert script_evidence(FA) == "fa"
    assert script_evidence(EN) == _LATIN
    assert script_evidence(EN_SHORT) == _LATIN


def test_script_evidence_declines_on_thin_input():
    # Too short to be decisive either way — and "no evidence" must not be
    # spelled the same as "English", or every grunt clears a real directive.
    assert script_evidence("") is None
    assert script_evidence(FA_SHORT) is None
    assert script_evidence("ok") is None


def test_an_english_sentence_carrying_a_persian_name_is_still_english():
    # The mirror of the code-switch case: one foreign token must not flip the
    # sentence. This is what min_share is for.
    assert script_evidence("Play something by ابی and then tell me the news") == _LATIN


# ── The reported bug ──────────────────────────────────────────────────────

def test_english_audio_mistranscribed_as_farsi_does_NOT_flip_the_session():
    """THE round-ten regression test.

    The transcriber rendered English speech in Persian script. The model heard
    the audio and answered in English. That disagreement is the signature of a
    mis-transcription, and it must produce no directive at all.
    """
    corroborated: set = set()
    want = next_reply_directive(
        user_text="سرچ کن قوی‌ترین مدل برای ساخت ویدیو چیه",   # what the transcriber wrote
        reply_text="The strongest model right now is Seedance 2.0, with Veo 3.1 "
                   "the safer pick for polished production work.",  # what the model SAID
        corroborated=corroborated,
        pinned=None,
    )
    assert want is None, "a transcript the model contradicted must not pin a language"
    assert corroborated == set(), "a disagreement must not corroborate anything"


def test_a_genuine_farsi_turn_still_gets_the_directive():
    corroborated: set = set()
    want = next_reply_directive(
        user_text=FA,
        reply_text="سلام! خوبم ممنون، امروز چه کمکی از دستم برمیاد؟",
        corroborated=corroborated,
        pinned=None,
    )
    assert want == "fa"
    assert corroborated == {"fa"}


def test_english_turn_clears_a_directive_a_previous_turn_set():
    corroborated = {"fa"}
    assert next_reply_directive(FA, "سلام خوبم ممنون از تو", corroborated) == "fa"
    # ...and now they switch to English. The old rule kept Persian through any
    # turn under eight Latin letters; this one must not.
    assert next_reply_directive(EN, "Sure — here is what I found.", corroborated) is None
    assert next_reply_directive(EN_SHORT, "It is 4:22 PM.", corroborated) is None


def test_there_is_no_carry_over_between_turns():
    """A turn with no evidence must not inherit the last turn's verdict.

    The old rule's `else applied_ctx["directive"]` branch is exactly how a
    stale Persian directive survived a session of English.
    """
    corroborated = {"fa"}
    # Neither half says anything decisive.
    assert next_reply_directive("ok", "Mm-hm.", corroborated) is None
    assert next_reply_directive("", "", corroborated) is None


def test_a_corroborated_language_survives_the_model_slipping_once():
    """The 2026-08-17 defect the directive was built for.

    Once this call has PROVEN Persian, a Persian transcript that the model
    answered in English is read as the model slipping, not as a bad
    transcript — and the reinforcement fires. Before corroboration the same
    input is read the other way, which is the round-ten bug.
    """
    fresh: set = set()
    assert next_reply_directive(FA, "Sure, I can help with that.", fresh) is None

    proven = {"fa"}
    assert next_reply_directive(FA, "Sure, I can help with that.", proven) == "fa"


def test_the_pin_outranks_every_inference():
    corroborated: set = set()
    # Pinned English: no directive, whatever the transcript claims.
    assert next_reply_directive(FA, "سلام خوبم ممنون از تو", corroborated, pinned="en") is None
    # Pinned Farsi: the directive stands even on an English-looking turn.
    assert next_reply_directive(EN, "Here you go.", corroborated, pinned="fa") == "fa"


def test_a_farsi_then_english_then_farsi_session_tracks_every_switch():
    corroborated: set = set()
    seq = [
        (FA, "سلام! امروز چه کمکی از دستم برمیاد؟", "fa"),
        (EN, "The strongest one is Seedance 2.0 right now.", None),
        (EN_SHORT, "It is 4:22 in the afternoon.", None),
        (FA, "بله حتماً، الان برایت پیدا می‌کنم.", "fa"),
        (EN, "Done — that is queued up for you.", None),
    ]
    for user_text, reply_text, expected in seq:
        got = next_reply_directive(user_text, reply_text, corroborated)
        assert got == expected, f"{user_text[:24]!r} -> {got!r}, wanted {expected!r}"


# ── The transcription prompt ──────────────────────────────────────────────

def test_inferred_farsi_prompt_describes_a_bilingual_speaker():
    """An INFERRED language may not assert what the utterance is.

    Round nine's prompt said the audio IS colloquial Persian. That is a claim
    about the speaker's repertoire being applied to a single utterance, and it
    is what pushed English audio into Persian script.
    """
    p = transcription_prompt("fa", pinned=False)
    assert "bilingual" in p.lower()
    assert "Never translate." in p
    # Both scripts present, so the model has both in context.
    assert "English" in p and "فارسی" in p


def test_a_PINNED_language_may_be_asserted():
    p = transcription_prompt("fa", pinned=True)
    assert "bilingual" not in p.lower()
    assert "فارسی محاوره‌ای" in p


def test_no_language_still_means_the_plain_english_prompt():
    assert transcription_prompt(None) == transcription_prompt(None, pinned=True)
    assert "bilingual" not in transcription_prompt(None).lower()


def test_pin_vocabulary_is_closed_and_has_no_auto_member():
    # "auto" is spelled by sending nothing; two spellings of the default is
    # how one of them rots.
    assert ws_realtime._VOICE_LANG_PINS == {"en", "fa"}


# ── Layer one: the static rule, unchanged from round nine ─────────────────

def test_reply_lang_names_cover_detectable_scripts():
    from app.api.voice import _SCRIPT_LANGUAGES
    detectable = {code for code, _ in _SCRIPT_LANGUAGES}
    assert detectable <= set(ws_realtime._REPLY_LANG_NAMES), (
        "a script detect_script_language can return has no directive name — "
        "that turn silently gets no reinforcement"
    )
    assert ws_realtime._REPLY_LANG_NAMES["fa"] == "Persian (Farsi)"


def test_static_rule_leads_both_instruction_blocks():
    src = inspect.getsource(ws_realtime)
    assert src.count('"- REPLY LANGUAGE') == 2, "the hard rule must sit in BOTH blocks"
    assert src.count("reason to reply in English") == 2


# ── Where the decision runs, and what it is allowed to read ───────────────

def test_the_directive_is_decided_where_BOTH_halves_are_known():
    """Placement is the fix, so placement is pinned.

    At the transcript handler only one half of the turn exists, and reading it
    alone is the bug. The decision must sit in the response.done branch, after
    the reply text has been resolved.
    """
    src = inspect.getsource(ws_realtime)
    transcript_handler = src.split("input_audio_transcription.completed")[1][:4000]
    assert "next_reply_directive(" not in transcript_handler, (
        "the directive must not be decided from the transcript alone"
    )
    assert "response.done" in src
    call = src.index("want = next_reply_directive(")
    persist = src.index('"[REALTIME] Failed to save assistant message"')
    assert call > persist, "the decision must run in the assistant-reply branch"


def test_the_directive_is_re_derived_and_never_carried():
    src = inspect.getsource(inspect.getmodule(next_reply_directive))
    fn = src[src.index("def next_reply_directive"):]
    fn = fn[: fn.index("\ndef ", 1)] if "\ndef " in fn[1:] else fn
    assert 'applied_ctx' not in fn, "the decision must not read the previous verdict"


def test_reinforcement_still_fires_only_on_change_and_reuses_applied_context():
    src = inspect.getsource(ws_realtime)
    assert 'want != applied_ctx["directive"]' in src
    assert 'applied_ctx["tools"], applied_ctx["language"]' in src


def test_the_directive_text_puts_the_audio_above_itself():
    """The wording is load-bearing.

    "The user is speaking Persian right now" is a standing fact that outlives
    its turn and beats the model's own ears. The replacement has to defer to
    what the model actually heard, in the same breath.
    """
    src = inspect.getsource(ws_realtime)
    # Probe the CONSTRUCTION, not the module: the diagnosis comment above
    # `_REPLY_LANG_NAMES` quotes the old sentence on purpose, and a module-wide
    # substring test would forbid describing the bug that was fixed.
    built = src[src.index("# Current speech language"):]
    built = built[: built.index("await openai_ws.send")]
    assert "The user is speaking" not in built, "the standing-fact wording is back"
    assert "what you actually heard always wins" in built
    assert "expected reply language" in built
