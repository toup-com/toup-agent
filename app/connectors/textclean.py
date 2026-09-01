"""Reading a provider's text ONCE, correctly.

Three separate defects met on the same string and produced the same
symptom — a subject that reads ``R29-D live loop test Ã¢Â\x80Â\x94 Gmail
push`` where the sender wrote an em dash:

1. **No RFC 2047 decode on the read path.** A mail header is 7-bit, so
   anything outside ASCII arrives as an encoded word
   (``=?UTF-8?Q?...?=``). Gmail's ``users.messages.get`` hands those
   through verbatim and this repo took them verbatim
   (``{h["name"]: h["value"] …}``), so a subject with an accent, a
   curly quote or an em dash reached the brief as its own encoding.

2. **``html.unescape`` applied three times** to one string, at three
   layers (the collect line, the minted item, the narrator's row). It
   is not idempotent: ``&amp;amp;`` → ``&amp;`` → ``&``, and
   ``&amp;#8212;`` becomes an em dash the sender never wrote. Unescape
   at ingestion, once, and nowhere else.

3. **Mojibake already in the store.** UTF-8 bytes that an earlier hop
   read as Latin-1 are now correctly-encoded nonsense: no decoder can
   fix them, because nothing is broken any more — the text really does
   say ``Ã¢Â\x80Â\x94``. The only repair is to undo the wrong decode,
   and the only safe way is to require that undoing it succeeds AND
   measurably improves the string.

   That last one is why the repair ITERATES. The founder's subject went
   through the wrong decode twice (once when it was sent by the old
   header path, once on the way back in), so one round trip leaves
   ``â\x80\x94`` — still wrong, and a single pass would have called it
   done.

All three live here so a fourth caller cannot invent a fourth answer.
"""

from __future__ import annotations

import html
import re
from email.header import decode_header, make_header

__all__ = [
    "decode_mail_header",
    "unescape_once",
    "repair_mojibake",
    "clean_provider_text",
]


#: Encoded-word marker. Cheap enough to run on every header, and it keeps
#: `decode_header` — tolerant to the point of returning junk for malformed
#: input — off strings that plainly do not need it.
_ENCODED_WORD = re.compile(r"=\?[^?]+\?[BbQq]\?[^?]*\?=")

#: What a failed decode leaves behind.
_REPLACEMENT = "�"

#: The characters a wrong decode PRODUCES: the Latin-1 lead bytes of a
#: UTF-8 sequence (Ã Â â), the cp1252 glyphs for the
#: continuation bytes (€ ‘-” š œ Ÿ ƒ),
#: and the C1 controls Latin-1 leaves in their place. Counting these is
#: what makes the round trip self-validating: a repair is accepted only
#: when it strictly reduces the count, so ordinary accented prose — whose
#: count is already low and cannot fall — is never rewritten.
_ODD = re.compile(
    "[ÂÃâ-€‘-”šœŸƒ]"
)

#: One cheap gate before the scoring work.
_SUSPECT = re.compile("[ÂÃâ-]")

#: Wrong decodes come in two flavours, and they need different inverses:
#: a strict Latin-1 read (which turns the UTF-8 continuation bytes into
#: C1 controls) and a cp1252 read (which turns them into glyphs). Try the
#: strict one first — cp1252 has five undefined bytes, so it is the
#: lossier hypothesis and must not win a tie.
_INVERSES = ("latin-1", "cp1252")

#: The founder's subject needed two. A third pass has never been observed
#: and the bound keeps a pathological string from looping.
_MAX_PASSES = 3


def decode_mail_header(value: str | None) -> str:
    """An RFC 2047 header as the sender meant it.

    Returns the input unchanged when it holds no encoded word, and when
    decoding raises — a malformed header is still a header, and showing
    its raw form beats showing nothing.
    """
    if not value:
        return ""
    text = str(value)
    if not _ENCODED_WORD.search(text):
        return text
    try:
        return str(make_header(decode_header(text)))
    except (UnicodeDecodeError, LookupError, ValueError):
        return text


def unescape_once(value: str | None) -> str:
    """``html.unescape``, at the ONE place text enters the system."""
    return html.unescape(str(value)) if value else ""


def _odd_count(text: str) -> int:
    return len(_ODD.findall(text))


def repair_mojibake(value: str | None) -> str:
    """Undo a wrong decode of UTF-8 bytes, but only where it demonstrably works.

    Each pass tries both inverse decodes and keeps a result only if it
    round-trips without raising, introduces no replacement character, and
    lowers the odd-character count. When neither inverse improves the
    string the loop stops and the best-so-far is returned — so text that
    was never mojibake comes back byte-identical.
    """
    if not value:
        return ""
    text = str(value)
    if not _SUSPECT.search(text):
        return text
    best = text
    for _ in range(_MAX_PASSES):
        score = _odd_count(best)
        improved = None
        for enc in _INVERSES:
            try:
                candidate = best.encode(enc).decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
            if _REPLACEMENT in candidate and _REPLACEMENT not in best:
                continue
            if _odd_count(candidate) < score:
                improved = candidate
                break
        if improved is None:
            break
        best = improved
    return best


def clean_provider_text(value: str | None, *, header: bool = False) -> str:
    """The one door for provider text on the way in.

    ``header=True`` adds the RFC 2047 pass, which is meaningless — and
    occasionally destructive — on a body or a snippet: a message that
    happens to quote ``=?utf-8?q?…?=`` in its text is quoting it.
    """
    text = decode_mail_header(value) if header else (str(value) if value else "")
    return repair_mojibake(unescape_once(text))
