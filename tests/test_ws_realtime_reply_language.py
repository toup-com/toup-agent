"""The reply-language contract: a Farsi turn must not come back in English.

Two layers, both asserted here. The static REPLY LANGUAGE rule must lead both
instruction blocks (a model whose whole context is English needs the rule to
outrank that context), and the per-turn reinforcement must re-issue
session.update from the applied context when the transcript's script flips —
firing only on a state change, clearing on a clearly-Latin turn.
"""
import inspect

from app.api import ws_realtime


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
    # The rule must name the failure it exists to prevent: English context is
    # not a reason to answer in English.
    assert src.count("reason to reply in English") == 2


def test_reinforcement_fires_only_on_change_and_reuses_applied_context():
    src = inspect.getsource(ws_realtime)
    # gated on a state CHANGE, not per-turn spam
    assert 'want != applied_ctx["directive"]' in src
    # rebuilt from the exact applied session config, not a re-derivation
    assert 'applied_ctx["tools"], applied_ctx["language"]' in src
    # a clearly-Latin turn CLEARS the directive rather than pinning stale Farsi
    assert "latin_letters >= 8" in src
    # the reinforcement must sit AFTER the transcript persist, INSIDE the
    # transcript-completed handler
    handler = src.split('input_audio_transcription.completed')[1][:4000]
    assert "Reply-language reinforcement" in handler
