"""Did the user name a format, and did we deliver it?

Round 46, incident 7. Nothing in this codebase ever compared the format the
user asked for with the format that was delivered: there was no final-format
check, no truthful fallback for a format we cannot produce, and no telemetry
naming either side — which is why a fleet-wide search over 42 h returns zero
rows for any document activity at all. A format failure was unobservable by
construction.

Two functions:

* :func:`requested_formats` — which :class:`~app.agent.artifact_kinds.ArtifactKind`
  the user named, built on the SAME lexicon as
  ``query_intent._DOCUMENT_INTENT_RE`` (the turn-1 gate that unlocks the export
  tools), so the gate and the intent cannot disagree about what counts. That
  parity is asserted by ``tests/test_format_intent_contract.py``.
* :func:`format_check` — a verdict over (requested, delivered, channel), used
  for the ``[FORMAT]`` telemetry line and for the prompt's requested-format
  section. Kinds and counts only: it is handed MIMEs and filenames and returns
  kinds, so nothing downstream can log a filename by accident.

Pure: regex + the kind tables, no I/O, no settings, no model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from app.agent.artifact_kinds import ArtifactKind, kind_for_mime

# ── Per-kind lexicon ─────────────────────────────────────────────────────
#: Unambiguous format nouns: naming one IS the request ("pdf please"). Only
#: file-format words belong here — a word that is also an ordinary English
#: noun goes in `_NEEDS_VERB`, however strongly it suggests a file. Round 46
#: review: `slides`, `deck`, `presentation`, `spreadsheet`, `worksheet` and
#: bare `json` sat here, so "can you check my slides?", "why does this json
#: fail to parse?" and "forward me the spreadsheet Ana sent" each injected an
#: imperative ordering the model to PRODUCE that file. The same review had
#: already pulled `can you \S+` and `show me` out of the production verbs for
#: exactly this reason; this list was not given the same treatment.
_UNAMBIGUOUS = (
    (ArtifactKind.PDF, r"\bpdfs?\b"),
    (ArtifactKind.DOCX, r"\bdocx?\b|\bword\s+(?:doc(?:ument)?|file)s?\b|\bms\s*word\b"),
    (ArtifactKind.XLSX, r"\bxlsx?\b|\bexcel\b"),
    (ArtifactKind.PPTX, r"\bpptx?\b|\bpowerpoint\b"),
    (ArtifactKind.DATA_CSV, r"\bcsvs?\b|\bcomma[\s-]separated\b"),
    (ArtifactKind.MARKDOWN, r"\bmarkdown\b|\.md\b"),
)

# Kinds whose words are ordinary English ("picture", "script", "text") and
# therefore only count inside a request to PRODUCE something.
_NEEDS_VERB = (
    (ArtifactKind.IMAGE, r"\bimages?\b|\bpictures?\b|\bphotos?\b|\billustrations?\b"
                         r"|\bdrawings?\b|\blogos?\b|\bposters?\b|\bartwork\b|\brenders?\b"),
    (ArtifactKind.CODE, r"\bscripts?\b|\bsource\s+code\b|\bcode\s+file\b"
                        r"|\.(?:py|js|ts|sh|rb|go|rs|java|sql)\b"
                        r"|\bpython\s+file\b|\bshell\s+script\b"),
    (ArtifactKind.TEXT, r"\btxt\b|\bplain[\s-]text\s+file\b|\btext\s+file\b"),
    (ArtifactKind.ARCHIVE, r"\bzip(?:\s*file|\s*archive)?\b|\barchives?\b"),
    (ArtifactKind.DATA_JSON, r"\bjson\b"),
    (ArtifactKind.XLSX, r"\bspreadsheets?\b|\bworkbooks?\b|\bworksheets?\b"),
    (ArtifactKind.PPTX, r"\bslide\s*decks?\b|\bslides?\b|\bpresentations?\b|\bdecks?\b"),
)

# Audio and video carry their own verbs ("read it to me" is the request), so
# they are matched on phrases rather than on nouns plus a verb.
_AUDIO_RE = re.compile(
    r"\bmp3\b|\baudio\s+(?:file|version|clip)\b|\bvoice\s*(?:note|memo|message)\b"
    r"|\bvoice[\s-]?over\b"
    r"|\bread\s+(?:it|this|that|them|the\s+\S+)\s+(?:out\s+loud|aloud|to\s+me)\b"
    r"|\bsay\s+(?:it|this|that)\s+out\s+loud\b|\bspeak\s+(?:it|this|that)\b"
    r"|\btext[\s-]to[\s-]speech\b|\btts\b|\bas\s+audio\b|\bin\s+audio\b",
    re.IGNORECASE,
)

#: Audio words that are ordinary subject matter on their own — "summarize this
#: podcast", "the narration was good" — and so need the production verb the
#: other ordinary nouns need.
_AUDIO_NOUN_RE = re.compile(r"\bnarrat(?:e|ion)\b|\bpodcasts?\b", re.IGNORECASE)

_VIDEO_RE = re.compile(
    r"\bvideos?\b|\bmp4\b|\bmovies?\b|\banimations?\b|\banimated\s+clip\b"
    r"|\bvideo\s+clip\b|\breels?\b|\bscreen\s*recording\b",
    re.IGNORECASE,
)

#: Asking to WATCH, FIND or PLAY something is not asking us to PRODUCE it. Same
#: rule as `radio.player.infer_requested_mode`'s: a consumption verb counts at
#: the START of the request, because mid-sentence it is far more likely to be a
#: title. (That function itself is deliberately NOT reused here: its
#: "video of|for|version" pattern also matches "make a video of this", so it
#: cannot answer THIS question — it answers "which surface should this play on".)
_CONSUMPTION_RE = re.compile(
    r"^\s*(?:play|watch|listen\s+to|find|search\s+for|look\s+up|put\s+on)\b",
    re.IGNORECASE,
)

#: Verbs that ask for something to be MADE, as opposed to delivered or found.
#: Video and the ordinary audio nouns need one of these, not merely a production
#: verb: "send me the video of the meeting" and "download the podcast" are about
#: files that already exist, and answering them with "I cannot make one" is the
#: false refusal this gate removes.
_CREATION_VERB_RE = re.compile(
    r"\b(?:make|makes?\s+me|create|generate|produce|build|record|draw|design"
    r"|render|animate|compose|turn\s+(?:this|that|it)\s+into)\b",
    re.IGNORECASE,
)

# The production/delivery context that turns an ordinary noun into a request.
# Same shape as query_intent's second alternation, kept separate because this
# one is a PRESENCE test over the whole message, not an ordered match.
#: `can\s+you\s+\S+` and `show\s+me` were removed after round 46's review:
#: "can you find a picture of X" and "show me a photo of my dog" are requests to
#: FIND one, and both read as a production verb, so the prompt ordered the model
#: to GENERATE an image instead. Every real production phrasing they covered
#: ("can you make…", "can you write…") is already covered by the verb itself.
_PRODUCTION_VERB_RE = re.compile(
    r"\b(?:make|makes?\s+me|write|create|generate|draft|prepare|produce|build"
    r"|compile|put\s+together|turn\s+(?:this|that|it)\s+into|convert|export"
    r"|download|save|attach|give\s+me|send\s+me|draw|design|render"
    r"|i\s+(?:need|want|would\s+like)|as\s+an?|in\s+an?)\b",
    re.IGNORECASE,
)

_COMPILED_UNAMBIGUOUS = tuple((k, re.compile(p, re.IGNORECASE)) for k, p in _UNAMBIGUOUS)
_COMPILED_NEEDS_VERB = tuple((k, re.compile(p, re.IGNORECASE)) for k, p in _NEEDS_VERB)

#: Kinds no producer exists for, anywhere in the product (A14, round 46).
#: The agent must answer a request for one of these truthfully and offer the
#: closest thing it CAN make — never promise it, never silently substitute.
UNSUPPORTED_KINDS = frozenset({ArtifactKind.VIDEO})

#: Kinds a given channel cannot PRODUCE. Empty, and deliberately not the place
#: a DELIVERY hole is recorded: the live Baileys WhatsApp adapter implements
#: neither ``send_photo`` nor ``send_file``, but the file is still made, still
#: on the thread and still in the user's Files — and this table's prompt copy
#: is "You cannot make one", which would be a fresh lie. An adapter that cannot
#: carry a file is handled where it is known to be true, at delivery time
#: (`channels/shared/message_handler._deliver_attachments`), and the user is
#: told in the same breath where the file actually is.
CHANNEL_UNSUPPORTED: dict = {}

#: What we offer instead, per unsupported kind.
_ALTERNATIVE = {
    ArtifactKind.VIDEO: (
        "still images, a slide deck, or a narrated audio file"
    ),
}


def _asks_to_make(s: str) -> bool:
    """A request to MAKE one, not to be handed or shown an existing one."""
    return bool(_CREATION_VERB_RE.search(s)) and not _CONSUMPTION_RE.search(s)


def requested_formats(text: Optional[str]) -> Tuple[str, ...]:
    """The artifact kinds the user named, in a stable order. Empty when none."""
    s = (text or "").strip()
    if not s:
        return ()
    found: list = []

    def _add(kind: str) -> None:
        if kind not in found:
            found.append(kind)

    for kind, rx in _COMPILED_UNAMBIGUOUS:
        if rx.search(s):
            _add(kind)
    if _AUDIO_RE.search(s):
        _add(ArtifactKind.AUDIO)
    # Video and the ordinary audio nouns need a CREATION verb. Ungated,
    # `_VIDEO_RE` turned the product's own core media flow ("play the music
    # video for X") and every "summarize this video" into a prompt section
    # ordering the model to say it cannot make a video.
    if _asks_to_make(s):
        if _AUDIO_NOUN_RE.search(s):
            _add(ArtifactKind.AUDIO)
        if _VIDEO_RE.search(s):
            _add(ArtifactKind.VIDEO)
    if _PRODUCTION_VERB_RE.search(s):
        for kind, rx in _COMPILED_NEEDS_VERB:
            if rx.search(s):
                _add(kind)
    # "a document"/"a file" with no format word is a request for SOMETHING, not
    # for a kind. It opens the tool gate (query_intent's job) and is not a
    # format intent — comparing it against a delivered kind would fabricate a
    # mismatch out of a request that named nothing.
    return tuple(found)


@dataclass(frozen=True)
class FormatVerdict:
    requested: Tuple[str, ...]
    delivered: Tuple[str, ...]
    verdict: str          # none_asked | match | mismatch | unsupported | missing
    channel: str
    unsupported: Tuple[str, ...] = ()

    @property
    def requested_str(self) -> str:
        return ",".join(self.requested) if self.requested else "-"

    @property
    def delivered_str(self) -> str:
        return ",".join(self.delivered) if self.delivered else "-"

    @property
    def ok(self) -> bool:
        return self.verdict in ("none_asked", "match")


def delivered_kinds(mimes: Sequence[str],
                    filenames: Sequence[str] = ()) -> Tuple[str, ...]:
    """Kinds for a list of delivered attachments, de-duplicated, order kept."""
    out: list = []
    names = list(filenames) + [""] * max(0, len(mimes) - len(filenames))
    for mime, name in zip(mimes, names):
        k = kind_for_mime(mime, name)
        if k not in out:
            out.append(k)
    return tuple(out)


def format_check(user_text: Optional[str],
                 delivered_mimes: Sequence[str],
                 delivered_filenames: Sequence[str] = (),
                 *, channel: str = "app") -> FormatVerdict:
    """Compare what was asked for with what was produced. Never raises.

    ``unsupported`` outranks everything: if the user asked for a video we have
    no producer for, delivering a PDF is not a match and not a mismatch — it is
    a request the product cannot serve, and the only correct answer names that.
    """
    ch = (channel or "app").strip().lower() or "app"
    try:
        requested = requested_formats(user_text)
    except Exception:
        requested = ()
    try:
        delivered = delivered_kinds(list(delivered_mimes or ()), list(delivered_filenames or ()))
    except Exception:
        delivered = ()

    blocked = tuple(k for k in requested
                    if k in UNSUPPORTED_KINDS or k in CHANNEL_UNSUPPORTED.get(ch, ()))
    if not requested:
        return FormatVerdict((), delivered, "none_asked", ch)
    if blocked:
        return FormatVerdict(requested, delivered, "unsupported", ch, blocked)
    if not delivered:
        return FormatVerdict(requested, (), "missing", ch)
    if set(requested) & set(delivered):
        return FormatVerdict(requested, delivered, "match", ch)
    return FormatVerdict(requested, delivered, "mismatch", ch)


# ── Prompt text ──────────────────────────────────────────────────────────

#: The tool that produces each kind. Named in the prompt so the model does not
#: have to infer it from a tool list of 120.
_TOOL_FOR_KIND = {
    ArtifactKind.PDF: "generate_pdf",
    ArtifactKind.DOCX: "generate_docx",
    ArtifactKind.XLSX: "generate_xlsx",
    ArtifactKind.PPTX: "generate_pptx",
    ArtifactKind.MARKDOWN: "generate_markdown",
    ArtifactKind.DATA_CSV: "generate_data_file(format='csv')",
    ArtifactKind.DATA_JSON: "generate_data_file(format='json')",
    ArtifactKind.TEXT: "generate_data_file(format='txt')",
    ArtifactKind.CODE: "generate_data_file(format='code')",
    ArtifactKind.AUDIO: "generate_audio",
    ArtifactKind.IMAGE: "generate_image",
}


#: How to NAME a kind to the model. The wire vocabulary is for telemetry; the
#: prompt has to read like the request the user actually made.
_LABEL = {
    ArtifactKind.PDF: "a PDF",
    ArtifactKind.DOCX: "an editable Word document",
    ArtifactKind.XLSX: "an Excel workbook",
    ArtifactKind.PPTX: "a slide deck",
    ArtifactKind.MARKDOWN: "a Markdown file",
    ArtifactKind.DATA_CSV: "a CSV file",
    ArtifactKind.DATA_JSON: "a JSON file",
    ArtifactKind.TEXT: "a plain-text file",
    ArtifactKind.CODE: "a code file",
    ArtifactKind.AUDIO: "an audio file",
    ArtifactKind.IMAGE: "an image",
    ArtifactKind.VIDEO: "a video",
    ArtifactKind.ARCHIVE: "a zip archive",
}


def label_for_kind(kind: str) -> str:
    return _LABEL.get(kind, kind)


def requested_format_section(text: Optional[str], *, channel: str = "app") -> str:
    """The ``# Requested format`` prompt section, or ``""`` when no format was
    named. Budgeted at roughly 30 tokens: it is injected on the turns that name
    a format and on no others, so it must not move the prefix for everyone."""
    kinds = requested_formats(text)
    if not kinds:
        return ""
    ch = (channel or "app").strip().lower()
    blocked = [k for k in kinds if k in UNSUPPORTED_KINDS or k in CHANNEL_UNSUPPORTED.get(ch, ())]
    if blocked:
        alt = _ALTERNATIVE.get(blocked[0], "a document, an image, or an audio file")
        return (
            "# Requested format\n"
            f"The user asked for {_LABEL.get(blocked[0], blocked[0])}. You cannot make one — "
            f"say so plainly, never promise it, and offer {alt} instead."
        )
    tools = [
        _TOOL_FOR_KIND[k] for k in kinds if k in _TOOL_FOR_KIND
    ]
    if not tools:
        return ""
    named = ", ".join(_LABEL.get(k, k) for k in kinds if k in _TOOL_FOR_KIND)
    return (
        "# Requested format\n"
        f"The user asked for {named}. Deliver exactly that, with "
        f"{', '.join(tools)}. Another format is a wrong answer, not a near one."
    )
