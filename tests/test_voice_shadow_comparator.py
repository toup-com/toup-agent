"""The W-6 shadow comparator must not hide what it cannot see.

The shadow (`ws_realtime._instructions_step`) is the ONLY gate on the
voice-context flip: when it reports `match=True` often enough, the agent
assembler replaces the legacy builder as the thing that writes a real
user's Realtime instructions. So whatever the comparator is blind to is,
by construction, invisible at the moment the decision is made.

Two blindnesses were found live on 2026-08-12 by running the harness
across all 45 bound tenants:

  * DUPLICATES — `_section_fingerprints` keyed a plain dict on the
    section header, so a section emitted twice collapsed into its twin.
    A duplicated section is never intentional, so this is fatal by
    design.

    Precision matters here, because the first version of this note got
    it wrong. Tenant 03cbc72f carried two identical active `soul` rows
    and the assembler emitted `# Core Identity` twice — and the shadow
    DID report that one, as a 2-char content diff (601 vs 603). Not by
    design: `split("\n\n# ")` consumes the `# ` marker on every block
    except the first, so section 0's count and digest are computed over
    two more characters than every other section's. A duplicated FIRST
    section trips that asymmetry by accident. A duplicated LATER section
    trips nothing at all — `[A,B,B]` vs `[A,B]` compares equal across a
    46-character gap. Both the asymmetry and the collapse are fixed
    below.

  * ORDER — a dict has no order, so the same sections in a different
    sequence compared equal. Tenant 07ccb7c2 reported match=True with
    `Who you are (identity)` and `Voice Conversation Mode` swapped.

Order is deliberately NOT fatal: the assembler puts `identity_anchor` at
index 1, where the TEXT channel's runner puts it, instead of after the
whole day transcript (`voice_context.VOICE_SECTION_ORDER`, "Drift D2" —
a white-label guard the model reads 20k characters after the persona is
a guard the model has already contradicted). That reorder is an intended
improvement, so a comparator that failed on it would block the flip on a
fix. It must be VISIBLE, not fatal — an unintended reorder would
otherwise read exactly like no change at all.

Red-first: every test here was run against the pre-fix tree first.
`compare_voice_contexts` did not exist, and the duplicate case passed
under the old dict algorithm (reproduced inline below as
`_old_fingerprints`, which still demonstrates the defect).
"""

from __future__ import annotations

import hashlib


def _old_fingerprints(text: str) -> dict:
    """The pre-fix comparator, verbatim — kept so the defect it had stays
    demonstrable rather than merely described."""
    out = {}
    for block in (text or "").split("\n\n# "):
        block = block.strip()
        if not block:
            continue
        head = block.split("\n", 1)[0].lstrip("# ").strip()[:40]
        out[head] = (len(block), hashlib.sha256(block.encode("utf-8")).hexdigest()[:8])
    return out


def _old_match(agent: str, legacy: str) -> bool:
    a, l = _old_fingerprints(agent), _old_fingerprints(legacy)
    diff = sorted(set(a) ^ set(l)) + sorted(
        k for k in a if k in l and a[k][1] != l[k][1]
    )
    return not diff


# Three sections whose bodies are fixed template text, never user data.
_ALPHA = "# Core Identity\nYou are an intelligent AI assistant."
_BETA = "# Behavioral Guidelines\nBe concise and kind."
_GAMMA = "# Voice Conversation Mode\nYou are in a LIVE VOICE conversation."


def _doc(*sections: str) -> str:
    return "\n\n".join(sections)


# ── the two blindnesses ──────────────────────────────────────────────


def test_a_duplicated_later_section_is_fatal_not_invisible():
    """The real blindness: a section duplicated anywhere but position 0
    collapses into its twin and the comparator reports a match across a
    whole missing section."""
    from app.api.ws_realtime import compare_voice_contexts

    agent = _doc(_ALPHA, _BETA, _BETA)
    legacy = _doc(_ALPHA, _BETA)
    assert len(agent) - len(legacy) == 46, "the gap the old comparator missed"

    # The defect, still demonstrable rather than merely described.
    assert _old_match(agent, legacy) is True, (
        "the old comparator is supposed to be blind here — if this now "
        "fails, the inline copy has drifted from what shipped"
    )

    result = compare_voice_contexts(agent, legacy)
    assert result["match"] is False, "a duplicated section must never match"
    assert any("Behavioral Guidelines" in d for d in result["differs"]), result["differs"]


def test_a_duplicated_first_section_is_fatal_for_the_right_reason():
    """03cbc72f, live. The old comparator caught this one by accident —
    the leading-`# ` asymmetry made section 0 two characters longer than
    an identical copy of itself. It must still fail, but as a duplicate,
    not as a phantom content difference."""
    from app.api.ws_realtime import compare_voice_contexts, voice_section_fingerprints

    agent = _doc(_ALPHA, _ALPHA, _BETA)
    result = compare_voice_contexts(agent, _doc(_ALPHA, _BETA))
    assert result["match"] is False

    # The asymmetry itself is gone: two identical sections now fingerprint
    # identically, so the only thing distinguishing them is the dup marker.
    fp = voice_section_fingerprints(agent)
    assert fp["Core Identity"][0] == fp["Core Identity#2"][0], (
        "section 0 must not be measured over two more characters than its "
        f"own duplicate: {fp}"
    )
    assert fp["Core Identity"][1] == fp["Core Identity#2"][1]


def test_reordered_sections_are_reported_but_not_fatal():
    """07ccb7c2, live: same sections, swapped, reported match=True with no
    hint that anything moved. Order stays non-fatal (Drift D2 is an
    intended reorder) but must be visible."""
    from app.api.ws_realtime import compare_voice_contexts

    agent = _doc(_ALPHA, _GAMMA, _BETA)
    legacy = _doc(_ALPHA, _BETA, _GAMMA)

    result = compare_voice_contexts(agent, legacy)
    assert result["match"] is True, "content is identical — order is not fatal"
    assert result["order_match"] is False, "…but the reorder must be visible"
    assert result["agent_order"] == [
        "Core Identity", "Voice Conversation Mode", "Behavioral Guidelines",
    ]
    assert result["legacy_order"] == [
        "Core Identity", "Behavioral Guidelines", "Voice Conversation Mode",
    ]


# ── the comparator must still do its original job ────────────────────


def test_identical_documents_match_in_content_and_order():
    from app.api.ws_realtime import compare_voice_contexts

    doc = _doc(_ALPHA, _BETA, _GAMMA)
    result = compare_voice_contexts(doc, doc)
    assert result["match"] is True
    assert result["order_match"] is True
    assert result["differs"] == []
    assert sorted(result["same"]) == sorted(
        ["Core Identity", "Behavioral Guidelines", "Voice Conversation Mode"]
    )


def test_a_changed_section_body_still_fails():
    from app.api.ws_realtime import compare_voice_contexts

    result = compare_voice_contexts(
        _doc(_ALPHA, "# Behavioral Guidelines\nDIFFERENT."),
        _doc(_ALPHA, _BETA),
    )
    assert result["match"] is False
    assert result["differs"] == ["Behavioral Guidelines"]


def test_a_missing_section_still_fails():
    from app.api.ws_realtime import compare_voice_contexts

    result = compare_voice_contexts(_doc(_ALPHA), _doc(_ALPHA, _BETA))
    assert result["match"] is False
    assert result["differs"] == ["Behavioral Guidelines"]


def test_the_comparison_never_carries_section_bodies():
    """Everything this returns goes into a log line, and the sections are
    the user's persona, brains and day transcript."""
    from app.api.ws_realtime import compare_voice_contexts

    secret = "the user's dog is named Kesh"
    result = compare_voice_contexts(
        _doc(_ALPHA, f"# User Brain (What You Know About the User)\n{secret}"),
        _doc(_ALPHA),
    )
    assert secret not in repr(result), "section bodies must never leave the process"


def test_empty_or_missing_input_is_not_a_match():
    from app.api.ws_realtime import compare_voice_contexts

    for agent, legacy in (("", _doc(_ALPHA)), (_doc(_ALPHA), ""), ("", "")):
        assert compare_voice_contexts(agent, legacy)["match"] is False
