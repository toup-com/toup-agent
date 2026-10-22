"""PR-13 (P11, D4): the call of 2026-10-02 answered the literal mishearing.

The caller asked for "the most experienced professor working on LLMs at U of
T". Speech recognition wrote "Can you search Who is most experienced officer in
LLM in U of T", and nothing between the microphone and the agent questioned it:

* the Live prompt's only clarify rule covered words that "came through
  garbled" — this one was heard cleanly and simply did not fit;
* the relay appended its own '# Backend' section after the tenant had already
  rendered one, so the model read two near-identical routing documents;
* the agent was handed the raw transcript and told to search immediately, with
  nothing saying the text was speech and could hold a misheard word;
* had the model asked a restating question, an answer like "No, professor"
  would have reached the agent with no question and no original request.

These tests pin each fix. The agent learns the request is a transcript from
its VOICE CONTRACT (system side), never from text in the delegated message:
the tenant classifies tool intent from the whole message
(`classify_query_intent(user_message)`, agent_runner.py), so any scaffold
sentence there re-gates the turn — a relay-side note flipped "Play some jazz"
from media to code and took `play_media` away.  The section "Tool gating"
pins that the relay hands the agent the same text, and so the same tools, as
before.

Old-fail: the prompt and backend-section tests fail against f49c49a6 by
assertion; the tool-gating tests pass on f49c49a6 and failed on the reviewed
cb1a2838 (which put a transcript note in the message); CONTROL tests pass
before and after.
"""

from __future__ import annotations

import asyncio
import re
import time
from datetime import datetime, timezone

import pytest

import test_live_harness as H
import app.services.live_voice_protocol as live
from app.agent.voice_context import render_live_delegation, render_onboarding, render_voice_mode
from app.agent.query_intent import classify_query_intent
from app.agent.source_conflict import wants_source_conflict_rules
from app.services.live_voice_protocol import (
    LIVE_BACKEND_SECTION,
    LIVE_CLARIFY_RULES,
    adapt_instructions_for_live,
    delegated_agent_input,
    open_question_sentence,
    question_subject,
)
from test_live_f5_echo_head import CLOCKS, Phone, _agent, _playback_idle


#: The exact sentences the clarify rule must say (FINDINGS §2 P11 "Fix").
CLARIFY_SENTENCES = (
    "- Before you delegate, check the key words of the request: the person, role, "
    "thing, place or name the work depends on.",
    "ask ONE short yes-or-no question that restates the whole request with your "
    "best reading, and wait for the answer before delegating:",
    "- Ask at most once per request. If the answer still leaves it unclear, "
    "delegate with your best reading.",
    "- If every key word fits, do not ask: interpret the request in the context of "
    "this conversation and what you know about the user",
    "- When two readings fit equally (an acronym with two meanings, a name you "
    "cannot place), go with the likelier one for this user and name the other in "
    "a few words.",
    "- When you delegate, your one short line names what you will look up or do",
)

OLD_GARBLED_RELAY = (
    "- If a name or word came through garbled, ask one short question instead "
    "of acting on a guess."
)
OLD_GARBLED_TENANT = "- If a name or a word came through garbled, ask one short question"

KEEP_AS_SAID = "- A word that makes sense as said is not a mis-hearing: keep it."
DIRECTIVE = "Reply in the caller's language."
FROZEN = datetime(2026, 10, 2, 2, 28, tzinfo=timezone.utc)


def _example_question(document: str) -> str:
    found = re.findall(r'"(Should I look up [^"]+\?)"', document)
    assert len(found) == 1, found
    return found[0]


# ══════════════════════════════════════════════════════════════════════
# (1) The Live clarify rule
# ══════════════════════════════════════════════════════════════════════

def test_the_live_clarify_rule_restates_the_whole_request_once():
    for sentence in CLARIFY_SENTENCES:
        assert sentence in LIVE_BACKEND_SECTION, sentence
        assert sentence in LIVE_CLARIFY_RULES, sentence
    assert OLD_GARBLED_RELAY not in LIVE_BACKEND_SECTION
    # A real "officer" ("Chief AI Officer") must not read as a mis-hearing:
    # the examples are generic, not the call's own words, and a word that
    # makes sense as said is kept.
    assert KEEP_AS_SAID in LIVE_CLARIFY_RULES
    for incident_word in ("officer", "professor", "U of T", "LLM"):
        assert incident_word not in LIVE_CLARIFY_RULES, incident_word
    # …and is served to GPT-Live.
    live_prompt = adapt_instructions_for_live("# Voice Conversation Mode\n- Be brief.")
    for sentence in CLARIFY_SENTENCES:
        assert sentence in live_prompt, sentence


def test_the_prompts_example_question_parses_to_a_look_up_subject():
    """The shape is load-bearing: `question_subject` is what titles a consent
    card and carries the question to the agent. 'Did you mean professor?' and
    either/or questions parse to nothing."""

    question = _example_question(LIVE_BACKEND_SECTION)
    assert open_question_sentence("Hmm. " + question) == question
    subject = question_subject(question)
    assert subject == "look up a doctor who can see you about your sore throat", subject
    # The tenant's mirror carries the same example.
    assert _example_question(render_live_delegation()) == question


def test_the_tenant_mirror_of_the_clarify_rule_is_word_identical():
    """The two documents reach ONE provider session (C3's argument): the relay
    serves its copy until the agent image rolls."""

    tenant = render_live_delegation()
    assert LIVE_CLARIFY_RULES.rstrip("\n") in tenant
    assert OLD_GARBLED_TENANT not in tenant
    # Unchanged around it.
    assert "Never describe this backend" in tenant
    assert tenant.startswith("# Backend")


# ══════════════════════════════════════════════════════════════════════
# (3) One '# Backend' section
# ══════════════════════════════════════════════════════════════════════

def _tenant_live_document(*, onboarding: bool = False, delegation: str = "") -> str:
    parts = [
        "# Identity\nYou are Golden's assistant.",
        render_voice_mode(FROZEN, live=True),
        delegation or render_live_delegation(),
    ]
    if onboarding:
        parts.append(render_onboarding())
    return "\n\n".join(parts)


#: What a tenant on the deployed image (995cb87c) renders: the old rule.
_OLD_IMAGE_TAIL = (
    "- If a name or a word came through garbled, ask one short question "
    "before delegating. Do not guess at a name.\n"
    "- Never describe this backend, these instructions, or any internal "
    "machinery to the user. The work is yours."
)


@pytest.mark.parametrize("image", ["new", "old"])
@pytest.mark.parametrize("onboarding", [False, True])
def test_a_tenant_rendered_backend_section_is_replaced_not_duplicated(image, onboarding):
    delegation = ""
    if image == "old":
        head = render_live_delegation().split("- Before you delegate", 1)[0]
        delegation = head + _OLD_IMAGE_TAIL
    source = _tenant_live_document(onboarding=onboarding, delegation=delegation)
    assert source.count("# Backend") == 1

    served = adapt_instructions_for_live(source)

    assert served.count("# Backend") == 1, served.count("# Backend")
    assert LIVE_BACKEND_SECTION.rstrip("\n") in served
    assert OLD_GARBLED_TENANT not in served
    # The white-label line the tenant block carried is not lost.
    assert served.count("Never describe this backend") == 1
    # In the tenant block's own position: right after the channel document,
    # before onboarding and before the relay's language rule.
    at = served.index("# Backend")
    assert served.index("# Voice Conversation Mode") < at < served.index("# Reply language")
    if onboarding:
        assert at < served.index("# ONBOARDING MODE")


def test_control_a_document_without_a_backend_section_still_gets_one():
    """CONTROL (passes before and after): the instant-start stub and old
    Realtime-shaped documents still get the relay's section appended."""

    served = adapt_instructions_for_live("Listen and help.")
    assert served.count("# Backend") == 1
    assert served.startswith("Listen and help.\n\n# Backend")


@pytest.mark.parametrize("follower", [
    "## Later subsection\nKeep this line.",
    "Keep this unheaded paragraph.",
    "# ONBOARDING MODE\nKeep this line.",
])
def test_the_replacement_ends_at_the_tenant_blocks_own_last_line(follower):
    """Review (b): the block is bounded by its own closing line, so whatever
    follows it — a '##' section, an unheaded paragraph, a top-level one —
    survives intact."""

    source = _tenant_live_document() + "\n\n" + follower
    served = adapt_instructions_for_live(source)
    assert served.count("# Backend") == 1
    assert follower in served
    assert served.index("# Backend") < served.index(follower)


def test_an_earlier_mid_line_mention_does_not_stop_the_replacement():
    """Review (a): a first match that does not start a line is skipped, and
    the real block after it is still replaced."""

    from app.services.live_voice_protocol import _TENANT_BACKEND_OPENING

    mention = "# Notes\nThe user pasted: " + _TENANT_BACKEND_OPENING + ", quoted."
    source = mention + "\n\n" + _tenant_live_document()
    assert "and it answers you." in source  # the tenant block's own wording
    served = adapt_instructions_for_live(source)
    assert "and it answers you." not in served
    assert served.count(LIVE_BACKEND_SECTION.rstrip("\n")) == 1
    assert served.count("Never describe this backend") == 1
    assert served.startswith(mention)


def test_control_a_block_without_its_closing_line_is_left_and_the_section_appended():
    """CONTROL: a block this relay does not recognise end to end (truncated,
    or a future reshaped one) is never cut; the relay appends its own section
    as it did before PR-13."""

    head = render_live_delegation().split("- Never describe this backend", 1)[0].rstrip()
    source = "# Voice Conversation Mode\n- Be brief.\n\n" + head
    served = adapt_instructions_for_live(source)
    assert served.startswith(source)
    assert served.count("# Backend") == 2


def test_the_tenant_block_is_followed_only_by_onboarding():
    """The replacement assumes the tenant block is one contiguous run; pin the
    order of what may follow it (identical on f49c49a6, 16ee4958, 995cb87c)."""

    from app.agent.voice_context import VOICE_SECTION_ORDER

    order = list(VOICE_SECTION_ORDER)
    assert order[order.index("voice_mode") + 1] == "live_delegation"
    assert order[order.index("live_delegation") + 1:] == ["onboarding"]


# ══════════════════════════════════════════════════════════════════════
# (2) Tool gating: the delegated message is the request, as before
# ══════════════════════════════════════════════════════════════════════

def _gating(message: str):
    intent = classify_query_intent(message)
    return (
        intent.category, intent.tool_names, intent.include_media_section,
        wants_source_conflict_rules(message),
    )


def test_control_a_plain_delegation_is_byte_identical():
    """CONTROL (passes on f49c49a6 and now): no transcript note, no scaffold."""

    assert delegated_agent_input("First request", []) == "First request"
    assert delegated_agent_input("First request", [], language_directive=DIRECTIVE) == (
        "First request\n\n" + DIRECTIVE
    )

# ══════════════════════════════════════════════════════════════════════
# (4) The agent's voice contract
# ══════════════════════════════════════════════════════════════════════

def test_the_voice_contract_says_the_request_is_a_speech_transcript():
    from app.agent.agent_runner import CHANNEL_GUIDANCE

    rule = CHANNEL_GUIDANCE["voice"].split("  2. ")[0]
    sentence = (
        "The request is a speech transcript, so resolve an obvious mis-hearing from "
        "context before you search"
    )
    assert sentence in rule
    # Generic, and guarded: a real "officer" is kept as said.
    assert "A word that makes sense as said is kept as said." in rule
    assert "officer" not in rule and "professor" not in rule
    # Before the instruction to search immediately, which it qualifies.
    assert rule.index(sentence) < rule.index("call `web_search` immediately")


# ══════════════════════════════════════════════════════════════════════
# Harness: the relay end to end
# ══════════════════════════════════════════════════════════════════════

async def _call(monkeypatch, script, *, receipts=(), instructions="Listen and help."):
    """One relay call on the F5 clocks; returns (phone, provider, calls)."""

    H.fast_clocks(monkeypatch, **CLOCKS)
    monkeypatch.setattr(live, "_ECHO_RECEIPT_TAIL_MS", 100, raising=False)
    calls: list[dict] = []

    async def think(_user_id, task, _session_id, **kwargs):
        calls.append({"task": task, "display": kwargs.get("display_request")})
        return "An answer.", "test-model"

    H.patch_relay(monkeypatch, think=think)
    box: dict = {}

    def on_send(provider, event):
        if event["type"] == "session.start":
            box["p"] = provider
            box["t0"] = asyncio.get_running_loop().time()
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def at(wall_s):
        delay = box["t0"] + wall_s - asyncio.get_running_loop().time()
        if delay > 0:
            await asyncio.sleep(delay)

    def start():
        box["run"] = asyncio.ensure_future(script(box["p"], at))

    items: list = [H.config(), start]
    for wall_s, make in receipts:
        items.append(lambda wall_s=wall_s: at(wall_s))
        items.append(("frame", lambda make=make: make(box["client"])))
    items.append(lambda: box["run"])
    items.append({"type": "stop"})
    client = Phone(items)
    box["client"] = client
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(
        client, provider, timeout=30.0, instructions=instructions,
        db_session_id=f"db-pr13-{time.monotonic_ns()}",
    )
    return client, provider, calls


def _says(pieces, *, wall_s, step_s=0.12, span_ms=120, delegation=""):
    async def run(p, at):
        position = int(wall_s * 1000)
        for k, piece in enumerate(pieces):
            await at(wall_s + k * step_s)
            p.push(H.user_delta(piece, position, position + span_ms))
            position += span_ms
        if delegation:
            p.push(H.delegation(delegation, position + 40))
    return run


def _segments(segments, *, wall_s, pause_s, delegation):
    """Several caller turns separated by pauses longer than the gap."""

    async def run(p, at):
        wall = wall_s
        for k, text in enumerate(segments):
            await at(wall)
            position = int(wall * 1000)
            p.push(H.user_delta(text, position, position + 300))
            if k == len(segments) - 1:
                p.push(H.delegation(delegation, position + 340))
            wall += 0.3 + pause_s
    return run


def _echo_at(chunks, *, wall_s, step_s=0.18, lag_ms=150):
    async def run(p, at):
        for k, chunk in enumerate(chunks):
            wall = wall_s + k * step_s
            await at(wall)
            position = int(wall * 1000) - lag_ms
            p.push(H.user_delta(chunk, position, position + lag_ms))
    return run


def _script(*parts, until_s):
    async def script(p, at):
        await asyncio.gather(*(part(p, at) for part in parts))
        await at(until_s)
    return script


def _finals(client):
    return [(f["text"], f.get("echo_only")) for f in client.of("transcript") if f.get("final")]


REQUEST = ["Can you search", " who is most experienced", " officer in LLM in U of T"]
REQUEST_TEXT = "".join(REQUEST)
LITERAL_QUESTION = "Should I look up the most experienced officer in the LLM program at U of T?"


@pytest.mark.asyncio
async def test_replay_of_the_2026_10_02_call_serves_one_backend_and_keeps_the_request(monkeypatch):
    """Offline replay of the call: the greeting, the model's 'What's up with
    you?', the request in three recognition segments, GPT-Live's delegation.
    GPT-Live is started with ONE '# Backend' section carrying the new clarify
    rule; the agent gets the caller's request and nothing else (so the same
    tool gating as before); the greeting question (which proposes nothing)
    adds no answer context; the card shows only the caller's words."""

    greeting = (
        "Hey! I'm doing good, thanks. Just hanging out, ready to help. "
        "What's up with you?"
    ).split()
    segments = [
        "Everything is good. Can you search Who is", " most experienced",
        " officer in LLM in U of T",
    ]
    client, provider, calls = await _call(
        monkeypatch,
        _script(
            _says(["Hey,", " how's it going"], wall_s=0.2),
            _agent(greeting, wall_s=0.9, step_s=0.05, position_ms=900),
            _segments(segments, wall_s=2.2, pause_s=0.5, delegation="item_EUjmcVJ"),
            until_s=5.0,
        ),
        receipts=[(1.8, _playback_idle)],
        instructions=_tenant_live_document(),
    )

    served = provider.of("session.start")[0]["session"]["instructions"]
    assert served.count("# Backend") == 1
    for sentence in CLARIFY_SENTENCES:
        assert sentence in served, sentence

    assert len(calls) == 1, (calls, _finals(client))
    request = "".join(segments)
    # The card is titled from the caller's own words only — since PR-10
    # (light P7) from the first sentence that asks something, so the leading
    # wellbeing answer "Everything is good." is not in the title. Still a
    # suffix of what the caller said, never the proposal or any context.
    title = "Can you search Who is most experienced officer in LLM in U of T"
    assert live.display_request_text(request) == title
    assert request.endswith(title)
    assert calls[0]["display"] == title, (calls, _finals(client))
    # The agent still gets every word the caller said, and nothing else.
    message = calls[0]["task"]
    assert message == request + "\n\n" + DIRECTIVE, message
    assert _gating(message)[0] == "web"


#: (request, the category it has always had as a Live delegation).  The
#: categories are the deployed tenant's: query_intent.py is byte-identical on
#: f49c49a6 and 995cb87c.
GATED_REQUESTS = [
    ("Play some jazz", "media"),
    ("Can you search who is most experienced officer in LLM in U of T", "web"),
    ("Send an email to Sara saying I'll be late", "full"),
    ("Remind me at 5pm to call mom", "scheduling"),
    ("Tell a joke", "question"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "request_text,category", GATED_REQUESTS, ids=[c for _r, c in GATED_REQUESTS],
)
async def test_a_live_delegation_keeps_the_requests_tool_gating(
    monkeypatch, request_text, category,
):
    """Review blocking item: the tenant classifies the WHOLE delegated message.
    Through the real relay, a first request reaches the agent as the request
    plus the language line — exactly the f49c49a6 bytes — so its category,
    tools, media section and source-conflict rules are unchanged.  All five
    failed on cb1a2838, whose transcript note changed the message (the review
    measured media, web, full and scheduling requests re-gated to 'code')."""

    words = request_text.split(" ")
    pieces = [words[0]] + [" " + w for w in words[1:]]
    client, provider, calls = await _call(
        monkeypatch,
        _script(_says(pieces, wall_s=0.2, delegation="d-gate"), until_s=2.5),
    )

    assert len(calls) == 1, (calls, _finals(client))
    message = calls[0]["task"]
    assert message == request_text + "\n\n" + DIRECTIVE, message
    gating = _gating(message)
    assert gating == _gating(request_text + "\n\n" + DIRECTIVE)
    assert gating[0] == category, gating[0]
    if category == "media":
        assert "play_media" in gating[1] and gating[2] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer", [["No,", " professor"], ["Yeah", " exactly"]],
    ids=["correction", "agreement-answer_consents-does-not-parse"],
)
async def test_an_answer_that_is_not_a_parsed_consent_carries_the_request(monkeypatch, answer):
    """D4-gap: the model asks before delegating (as the new rule says) and the
    caller answers. The request turn was answered by the question, so
    `request_span` stops before it; the agent used to receive only the answer.
    It now gets the question AND the original request.  'Yeah exactly' is an
    agreement in English, but `answer_consents` does not read it as one, so it
    takes this branch (correction wording, the caller's words as title) rather
    than the R6-5 consent branch — better than base, which carried nothing."""

    client, provider, calls = await _call(
        monkeypatch,
        _script(
            _says(REQUEST, wall_s=0.2),
            _agent(LITERAL_QUESTION.split(), wall_s=1.3, step_s=0.05, position_ms=1300),
            _says(answer, wall_s=2.8, delegation="d-correct"),
            until_s=4.5,
        ),
        receipts=[(2.4, _playback_idle)],
    )

    said = "".join(answer)
    assert [c["display"] for c in calls] == [said], (calls, _finals(client))
    message = calls[0]["task"]
    head, tail = message.split("Current caller request: ")
    assert LITERAL_QUESTION in head, message
    assert REQUEST_TEXT in head, message
    assert "do only what they ask" in head
    assert tail == said + "\n\n" + DIRECTIVE
    # Gated like the R6-5 consent to the same question: research, with
    # web_search — never a tool-less question turn and never 'code'.
    gating = _gating(message)
    assert gating[0] == "web" and "web_search" in gating[1], gating[0]
    # The card is the caller's words, never the request or the question.
    created = [f for f in client.of("delegation") if f.get("phase") == "created"]
    assert [f.get("title") for f in created] == [said]


@pytest.mark.asyncio
async def test_control_a_consent_keeps_the_r6_5_wording_and_title(monkeypatch):
    """CONTROL for the new branch: 'yes' to the proposal is still the R6-5
    consent — titled from the proposal, worded exactly as before."""

    client, provider, calls = await _call(
        monkeypatch,
        _script(
            _says(REQUEST, wall_s=0.2),
            _agent(LITERAL_QUESTION.split(), wall_s=1.3, step_s=0.05, position_ms=1300),
            _says(["yes", " please"], wall_s=2.8, delegation="d-yes"),
            until_s=4.5,
        ),
        receipts=[(2.4, _playback_idle)],
    )

    assert len(calls) == 1, (calls, _finals(client))
    message = calls[0]["task"]
    assert (
        f"The caller's words answer the assistant's own question «{LITERAL_QUESTION}» — "
        "they agree to what it proposed."
    ) in message
    assert "which it asked about their request" not in message
    assert "officer in the LLM program" in calls[0]["display"]


@pytest.mark.asyncio
async def test_a_correction_that_repeats_the_restating_question_is_the_callers_turn(monkeypatch):
    """The echo test. The question's own echo comes back through the hot mic
    while it plays; after the phone reports playback over, the caller corrects
    it with the question's own words ('the most experienced'). Those words are
    the caller's — a certain turn, never echo, never an UNCERTAIN head — and
    the agent gets the question and the original request with them.

    Old-fail scope: on f49c49a6 only the question/request-context assertions
    fail; the echo assertions (final not echo_only, no UNCERTAIN head) are
    regression pins for the new answer path and were not shown to fail
    before."""

    echo = [" should I look up", " the most experienced", " officer in the",
            " LLM program at", " U of T"]
    correction = ["No,", " the most experienced", " professor"]
    client, provider, calls = await _call(
        monkeypatch,
        _script(
            _says(REQUEST, wall_s=0.2),
            _agent(LITERAL_QUESTION.split(), wall_s=1.3, step_s=0.05, position_ms=1300),
            _echo_at(echo, wall_s=1.6),
            _says(correction, wall_s=3.2, delegation="d-echo"),
            until_s=5.0,
        ),
        receipts=[(2.7, _playback_idle)],
    )

    said = "".join(correction)
    finals = _finals(client)
    assert finals[-1] == (said, None), finals
    assert any(flag is True for _text, flag in finals[:-1]), finals
    assert [c["display"] for c in calls] == [said], (calls, finals)
    message = calls[0]["task"]
    assert "UNCERTAIN" not in message
    head, tail = message.split("Current caller request: ")
    assert LITERAL_QUESTION in head and REQUEST_TEXT in head, message
    assert tail == said + "\n\n" + DIRECTIVE
