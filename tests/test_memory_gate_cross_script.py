"""The gate's cross-script blind spot, and the sampling that exposed it.

memverify's B06-world-knowledge-farsi flaked because gpt-4o-mini (temperature
0.3, sampled once per CI run) sometimes renders an ENGLISH memory from a Farsi
question-only turn — and at that script boundary every deterministic rule
abstained at once:

    assistant_echo_reason        abstains cross-lingually BY DESIGN (no
                                 lexical overlap to measure)
    unsupported_claim_reason     abstains EXPLICITLY on a script mismatch
    inferred_interest_reason     an English regex blocklist — free phrasings
                                 sail past it

`cross_script_question_only_reason` closes the corridor structurally: a
question-only turn in one script cannot source a memory written in another.
These tests run on the platform lane (pure functions, no DB) so the class is
caught on push, not on the next memverify run that happens to sample badly.

Proven red on pre-fix code before landing: with the rule absent, case (a)
below was ADMITTED by `memory_gate_reason` (returned None).
"""

from __future__ import annotations

from pathlib import Path

from app.services import memory_gate as gate

# The exact B06-world-knowledge-farsi turn (backend/tests/memverify/corpus.py).
B06_USER = "آپشن سهام چطور کار می‌کنه؟"
B06_ASSISTANT = (
    "آپشن سهام به شما این حق را می‌دهد که سهام را با قیمت مشخصی بخرید. "
    "ارزیابی 409A ارزش منصفانه سهام عادی شرکت‌های خصوصی را تعیین می‌کند "
    "و معمولاً وستینگ چهار ساله با کلیف یک ساله است."
)

# A free phrasing the inferred_interest regex does NOT match — no "is
# interested in", no "has knowledge about". Pure encyclopedia, in English,
# from a turn the user wrote entirely in Farsi.
B06_FREE_PHRASING = (
    "409A valuation determines fair market value of common stock "
    "for private companies."
)


# ── (a) the flake class itself: refused, deterministically ────────────────

def test_english_encyclopedia_memory_from_farsi_question_is_refused():
    """Pre-fix this returned None — the memory was admitted whenever the
    sampled extractor phrased it past the interest regex. That admission is
    what B06 caught intermittently on #529/#530.

    The full-gate assert runs FIRST so that on pre-fix code this test fails
    on `assert None == 'cross_script_question_only'` — proof of admission —
    rather than on an AttributeError for a function that doesn't exist yet."""
    assert gate.memory_gate_reason(
        B06_FREE_PHRASING,
        user_message=B06_USER,
        assistant_response=B06_ASSISTANT,
    ) == "cross_script_question_only"
    assert gate.cross_script_question_only_reason(B06_FREE_PHRASING, B06_USER) == (
        "cross_script_question_only"
    )


def test_every_prior_rule_abstains_on_the_flake_class():
    """The reason the new rule exists: each older control, asked directly,
    stays silent on this exact input. If one of them ever starts firing here,
    the new rule may be narrowable — revisit, don't just fix the assert."""
    assert gate.inferred_interest_reason(B06_FREE_PHRASING, B06_USER) is None
    assert gate.assistant_echo_reason(B06_FREE_PHRASING, B06_USER, B06_ASSISTANT) is None
    assert gate.unsupported_claim_reason(B06_FREE_PHRASING, B06_USER, B06_ASSISTANT) is None


# ── (b) A08-style Farsi declarative turn: not question-only, passes ───────

A08_USER = "من در تورنتو زندگی می‌کنم و برادرم اسمش بهراد است."
A08_ASSISTANT = "متوجه شدم."


def test_farsi_declarative_turn_with_farsi_memory_passes():
    """A08-farsi-only is a must-STORE control in the corpus. The turn states
    facts (and is not question-only), so the rule must abstain."""
    memory = "برادر کاربر بهراد نام دارد."
    assert gate.cross_script_question_only_reason(memory, A08_USER) is None
    assert gate.memory_gate_reason(
        memory, user_message=A08_USER, assistant_response=A08_ASSISTANT
    ) is None


def test_farsi_declarative_turn_with_english_memory_passes():
    """The production shape: the founder speaks Persian to a brain that
    stores English. Cross-script alone must NEVER refuse — only cross-script
    from a turn where the user asserted nothing. A08 notes the memory 'may be
    written in English or Farsi'; both must survive."""
    memory = "The user lives in Toronto and has a brother named Behrad."
    assert gate.cross_script_question_only_reason(memory, A08_USER) is None
    assert gate.memory_gate_reason(
        memory, user_message=A08_USER, assistant_response=A08_ASSISTANT
    ) is None


# ── (c) A25-style same-script question with an embedded fact: passes ──────

A25_USER = (
    "Since I'm colorblind — deuteranopia — can you suggest a chart palette "
    "that works for me?"
)
A25_ASSISTANT = "Use blue/orange rather than red/green; here are three palettes."


def test_same_script_question_with_embedded_fact_passes():
    """A25-fact-inside-a-question is a must-STORE control. Two independent
    outs, both of which must keep holding: the first-person declarative makes
    the turn not question-only, and the scripts match anyway."""
    memory = "The user is colorblind (deuteranopia)."
    assert gate.cross_script_question_only_reason(memory, A25_USER) is None
    assert gate.memory_gate_reason(
        memory, user_message=A25_USER, assistant_response=A25_ASSISTANT
    ) is None


# ── (d) explicit save always wins ─────────────────────────────────────────

def test_explicit_save_bypasses_the_cross_script_rule():
    """'Remember this' is the user stating it, whatever script the note lands
    in — the provenance rules all yield to an explicit ask and this one is no
    exception. Same content and turn as case (a); only the flag differs."""
    assert gate.memory_gate_reason(
        B06_FREE_PHRASING,
        user_message=B06_USER,
        assistant_response=B06_ASSISTANT,
        explicit_save=True,
    ) is None


# ── source pin: adjudication no longer samples at the chat default ────────

def _call_arg_blocks(src: str, needle: str) -> list:
    """The argument text of every `needle(...)` call, by paren matching."""
    blocks = []
    i = 0
    while True:
        j = src.find(needle, i)
        if j == -1:
            return blocks
        k = j + len(needle)
        depth, start = 1, k
        while depth:
            if src[k] == "(":
                depth += 1
            elif src[k] == ")":
                depth -= 1
            k += 1
        blocks.append(src[start : k - 1])
        i = k


def test_dedup_adjudication_calls_are_pinned_to_temperature_zero():
    """The three adjudication call sites in memory_dedup_service inherited
    the CHAT default temperature of 0.7 (config.py) — an accident for a
    classifier whose verdicts retire rows (#517). Pin all of them at 0.0, and
    pin the COUNT so a fourth call site cannot appear unpinned."""
    src = (
        Path(__file__).resolve().parent.parent
        / "app" / "services" / "memory_dedup_service.py"
    ).read_text()
    blocks = _call_arg_blocks(src, "complete_with_json(")
    assert len(blocks) == 3, (
        f"expected exactly 3 complete_with_json call sites, found {len(blocks)} "
        "— if you added one, pin its temperature and update this count"
    )
    unpinned = [b for b in blocks if "temperature=0.0" not in b]
    assert not unpinned, (
        f"{len(unpinned)} adjudication call site(s) missing temperature=0.0: "
        f"{unpinned}"
    )


# ── B12: one shared token is not the user gesturing at a claim ───────────
#
# Separate corridor from the cross-script one above, and a nastier kind of
# bug because it hid behind a ratio. `unsupported_claim_reason` refuses a
# claim on a question-only turn when the user's own words support
# "essentially none" of it — expressed as `overlap <= UNSUPPORTED_USER_MAX`
# (0.10). But a ratio has granularity 1/len(tokens): a nine-token memory
# sharing ONE word with the question scores 0.111 and clears the ceiling, so
# for any memory under ten distinctive tokens the ceiling was unreachable
# except at exactly zero overlap.
#
# memverify's B12-speculation-no-user-statement walked through that gap on
# three unrelated diffs (PRs #542 twice, #551 once) before it was read as a
# defect rather than a flake.


B12_USER = "What's a good framework for a side project?"
B12_ASSISTANT = (
    "Given that most side projects die from scope creep, you would probably "
    "want something with batteries included — Django or Rails."
)
B12_INFERRED = (
    "The user is considering side projects and is aware that many of them "
    "fail due to scope creep."
)


def test_single_shared_token_does_not_buy_immunity():
    """The exact row memverify stored. Shares one token with the question
    ('projects' — the subject the user used to ASK), draws 0.44 of itself
    from the assistant's answer, and asserts a trait the user never stated."""
    assert gate.memory_gate_reason(
        B12_INFERRED,
        user_message=B12_USER,
        assistant_response=B12_ASSISTANT,
    ) == "unsupported_on_question_turn"


def test_a_genuinely_referenced_fact_still_survives():
    """The rule's own documented keep-case, unchanged: two shared tokens
    ('flight', 'berlin') is the user actually gesturing at the claim."""
    assert gate.memory_gate_reason(
        "The user has a flight to Berlin.",
        user_message="Do you know if my flight to Berlin is on time?",
        assistant_response="Your flight to Berlin looks on time, departing at 4pm from gate 22.",
    ) is None


def test_the_storm_case_that_motivated_the_rule_still_fails_closed():
    """BUG-26 (measured in production 2026-08-05) must stay refused."""
    assert gate.memory_gate_reason(
        "The user charges their phones and keeps a flashlight handy during storms.",
        user_message="There's a big storm forecast tonight — anything I should do?",
        assistant_response=(
            "Charge your phones, keep a flashlight handy, and bring in any "
            "balcony items before the wind picks up."
        ),
    ) == "unsupported_on_question_turn"
