"""Did the user name a format, and did we deliver it? — round 46, C9.

Nothing in this codebase ever compared the two. There was no final-format
check, no truthful fallback for a format with no producer, and no telemetry
naming either side — which is why a fleet-wide search over 42 h returns zero
rows for any document activity at all: a format failure was unobservable by
construction, and the incident this round fixes can only be reasoned to.

Pure regex + tables. Platform sweep.
"""

from __future__ import annotations

import pytest

from app.agent.artifact_kinds import ArtifactKind
from app.agent.format_intent import (
    UNSUPPORTED_KINDS,
    delivered_kinds,
    format_check,
    label_for_kind,
    requested_format_section,
    requested_formats,
)
from app.agent.query_intent import TOOLS_DOCGEN, has_document_intent

MIME_PDF = "application/pdf"
MIME_DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


@pytest.mark.parametrize(
    "text,kind",
    [
        ("my resume as an editable Word doc", ArtifactKind.DOCX),
        ("PDF please", ArtifactKind.PDF),
        ("export this as CSV", ArtifactKind.DATA_CSV),
        ("turn this into a spreadsheet", ArtifactKind.XLSX),
        ("make me a deck for Monday", ArtifactKind.PPTX),
        ("give me the JSON", ArtifactKind.DATA_JSON),
        ("write it as markdown", ArtifactKind.MARKDOWN),
        ("save it as a txt file", ArtifactKind.TEXT),
        ("write me a .py file that does this", ArtifactKind.CODE),
        ("read it to me", ArtifactKind.AUDIO),
        ("record a voice note of the summary", ArtifactKind.AUDIO),
        ("can you make an mp3 of that", ArtifactKind.AUDIO),
        ("make it a video", ArtifactKind.VIDEO),
        ("draw me a picture of a cat", ArtifactKind.IMAGE),
    ],
)
def test_positive_corpus(text, kind):
    assert kind in requested_formats(text), requested_formats(text)


@pytest.mark.parametrize(
    "text",
    [
        "search the latest AI news and give me the 5 most important",
        "what did the report say?",
        "what's the weather tomorrow",
        "remind me to call mom at 5",
        "hello",
        "",
        None,
        # Names an artifact but NOT a format. That opens the tool gate (which
        # is query_intent's job); comparing it against a delivered kind would
        # fabricate a mismatch out of a request that named nothing.
        "write me a document about the migration",
    ],
)
def test_negative_corpus(text):
    assert requested_formats(text) == ()


@pytest.mark.parametrize(
    "text",
    [
        # The product's own core media flow. `_VIDEO_RE` used to be matched with
        # no verb gate at all, so every one of these injected "You cannot make
        # one — say so plainly", including over a request the product SERVES.
        "play the music video for HUMBLE",
        "what happened in the movie",
        "summarize this video for me",
        "watch the trailer for Dune",
        # Delivery, not creation: the file already exists.
        "send me the video of the meeting",
        "download the podcast",
        "summarize this podcast",
    ],
)
def test_consuming_media_is_never_a_request_to_produce_it(text):
    assert ArtifactKind.VIDEO not in requested_formats(text), requested_formats(text)
    assert ArtifactKind.AUDIO not in requested_formats(text), requested_formats(text)
    assert "cannot make one" not in requested_format_section(text)


@pytest.mark.parametrize(
    "text",
    [
        "show me a photo of my dog",
        "can you find a picture of a cat",
    ],
)
def test_asking_to_be_shown_a_picture_is_not_asking_for_one_to_be_drawn(text):
    """`show\\s+me` and `can\\s+you\\s+\\S+` were in the production-verb list, so a
    request to FIND a picture was turned into an order to GENERATE one."""
    assert ArtifactKind.IMAGE not in requested_formats(text)
    assert "generate_image" not in requested_format_section(text)


def test_an_ordinary_noun_needs_a_production_verb():
    """'picture', 'script', 'text' are ordinary English. Without a verb that
    asks for one to be MADE, they are subject matter, not a format."""
    assert ArtifactKind.IMAGE not in requested_formats(
        "the picture you sent yesterday was lovely")
    assert ArtifactKind.CODE not in requested_formats(
        "the script of the film was better than the book")


def test_the_gate_and_the_intent_cannot_disagree():
    """Round 46's own defect, in reverse: `csv` was in the turn-1 gate that
    unlocks the export tools and there was no CSV generator behind it. Now
    that there is one, every kind with a generator must ALSO open the gate —
    otherwise the tool is invisible on the turn that asks for it."""
    for text, kind in [
        ("export this as CSV", ArtifactKind.DATA_CSV),
        ("give me the JSON", ArtifactKind.DATA_JSON),
        ("save it as a txt file", ArtifactKind.TEXT),
        ("write me a .py file", ArtifactKind.CODE),
        ("read it to me", ArtifactKind.AUDIO),
        ("PDF please", ArtifactKind.PDF),
        ("as a Word doc", ArtifactKind.DOCX),
    ]:
        assert kind in requested_formats(text), text
        assert has_document_intent(text), f"{text!r} names {kind} and does not open the gate"


def test_the_new_generators_are_in_the_turn_1_tool_set():
    assert "generate_data_file" in TOOLS_DOCGEN
    assert "generate_audio" in TOOLS_DOCGEN


def test_delivered_kinds_reads_mimes_and_names():
    assert delivered_kinds([MIME_PDF], ["a.pdf"]) == (ArtifactKind.PDF,)
    assert delivered_kinds(["text/plain"], ["rows.csv"]) == (ArtifactKind.DATA_CSV,)
    # De-duplicated, order preserved, and a missing filename is tolerated.
    assert delivered_kinds([MIME_PDF, MIME_PDF, "image/png"]) == (
        ArtifactKind.PDF, ArtifactKind.IMAGE)
    assert delivered_kinds([]) == ()


def test_verdicts():
    # The headline case: an editable Word document answered with a PDF.
    v = format_check("my resume as an editable word doc", [MIME_PDF], ["r.pdf"])
    assert v.verdict == "mismatch"
    assert v.requested_str == "docx" and v.delivered_str == "pdf"
    assert not v.ok

    assert format_check("pdf please", [MIME_PDF], ["a.pdf"]).verdict == "match"
    assert format_check("hello", [], []).verdict == "none_asked"
    assert format_check("hello", [], []).ok
    # Asked and got nothing at all — worth a line precisely because there is
    # no attachment to hang it on.
    assert format_check("export as csv", [], []).verdict == "missing"


def test_unsupported_outranks_everything():
    """A14: there is no video producer anywhere in the product. Handing back a
    PDF is not a match and not a near miss — it is a request we cannot serve,
    and the only correct answer names that."""
    v = format_check("make me a video of it", [MIME_PDF], ["a.pdf"])
    assert v.verdict == "unsupported"
    assert v.unsupported == (ArtifactKind.VIDEO,)
    assert ArtifactKind.VIDEO in UNSUPPORTED_KINDS


def test_format_check_never_raises():
    for args in [(None, None, None), ("x", [None], [None]), ("x", ["a"], [])]:
        v = format_check(*args)   # type: ignore[arg-type]
        assert v.verdict in ("none_asked", "match", "mismatch", "unsupported", "missing")


def test_the_log_line_carries_kinds_and_never_a_filename():
    """The telemetry contract: `[FORMAT] requested= delivered= match=` is built
    from these two strings, so a filename cannot reach the trail by accident."""
    v = format_check("pdf please", [MIME_PDF], ["quarterly-board-pack.pdf"])
    assert "quarterly" not in v.requested_str + v.delivered_str
    assert v.delivered_str == "pdf"
    empty = format_check("hello", [], [])
    assert empty.requested_str == "-" and empty.delivered_str == "-"


def test_the_prompt_section_is_conditional_and_short():
    assert requested_format_section("what's the weather") == ""
    assert requested_format_section("") == ""
    csv = requested_format_section("export this as CSV")
    assert csv.startswith("# Requested format")
    assert "generate_data_file" in csv
    # ~30 tokens. It is injected per turn, so it must not be a paragraph.
    assert len(csv) // 4 <= 45, csv


def test_the_prompt_section_refuses_video_instead_of_promising_it():
    sec = requested_format_section("can you make a video of this")
    assert "cannot make one" in sec
    assert "generate_" not in sec, "it must not name a tool that does not exist"


def test_the_FORMAT_line_is_actually_wired_into_save_messages():
    """The round's own measurement of whether a requested format was honoured
    had no test at all, and its call site is wrapped in a bare
    `except Exception: logger.debug(...)` — so losing it produced neither a red
    test nor a log line. Order matters as much as presence: it reads the drained
    `pending_attachments`, so it has to sit AFTER the drain."""
    import inspect

    from app.agent.agent_runner import AgentRunner

    src = inspect.getsource(AgentRunner._save_messages)
    assert "format_check(" in src, "the [FORMAT] hook is not wired into _save_messages"
    assert "[FORMAT] requested=" in src
    assert src.index("pending_attachments = []") < src.index("format_check("), (
        "the check reads the drained list; it must run after the drain"
    )
    # …and nothing that is not a kind, a verdict or a count reaches the line.
    line = src[src.index('"[FORMAT] requested='):]
    line = line[: line.index(")")]
    for banned in ("filename", "mime_type", "user_message", "text"):
        assert banned not in line, f"[FORMAT] must not carry {banned}"


def test_labels_read_like_the_request_not_like_the_wire():
    assert label_for_kind(ArtifactKind.DATA_CSV) == "a CSV file"
    assert label_for_kind(ArtifactKind.DOCX) == "an editable Word document"
    assert label_for_kind("unknown-kind") == "unknown-kind"


# ── ordinary nouns that happen to name a file format ─────────────────────

@pytest.mark.parametrize("text", [
    "can you check my slides?",
    "why does this json fail to parse?",
    "forward me the spreadsheet Ana sent",
    "the deck from Monday was too long",
    "her presentation is at 3",
    "which worksheet has the totals?",
])
def test_merely_MENTIONING_a_format_noun_is_not_a_request_to_produce_one(text):
    """`_UNAMBIGUOUS` bypasses the production-verb gate, so anything listed
    there is an ORDER: "The user asked for a slide deck. Deliver exactly that,
    with generate_pptx. Another format is a wrong answer, not a near one."
    `slides`, `deck`, `presentation`, `spreadsheet`, `worksheet` and bare
    `json` are ordinary English and sat there. The same review had already
    pulled `can you \\S+` and `show me` out of the production verbs for exactly
    this reason."""
    assert requested_formats(text) == (), text
    assert requested_format_section(text) == ""


@pytest.mark.parametrize("text,kind", [
    ("make me a slide deck about Q3", ArtifactKind.PPTX),
    ("put together a presentation for Monday", ArtifactKind.PPTX),
    ("export the numbers as a spreadsheet", ArtifactKind.XLSX),
    ("give me the json", ArtifactKind.DATA_JSON),
    ("save it as json", ArtifactKind.DATA_JSON),
])
def test_the_same_nouns_inside_a_request_to_PRODUCE_still_count(text, kind):
    """The gate must not cost the feature."""
    assert kind in requested_formats(text), text


@pytest.mark.parametrize("text,kind", [
    ("pptx please", ArtifactKind.PPTX),
    ("as an xlsx", ArtifactKind.XLSX),
    ("powerpoint", ArtifactKind.PPTX),
    ("excel", ArtifactKind.XLSX),
    ("pdf", ArtifactKind.PDF),
])
def test_a_true_format_word_is_still_the_request_on_its_own(text, kind):
    """`pptx` and `xlsx` are not English words; naming one IS the ask."""
    assert kind in requested_formats(text), text
