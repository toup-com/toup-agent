"""One contract for what an artifact IS.

Round 46, incident 7: "format" was never a field. The set of things the agent
could deliver was defined by which functions happened to call
``doc_generators._persist``; the MIME was guessed at each call site (every
image was stamped ``image/png`` regardless of the bytes); and what a client may
DO with a file was decided independently in four places — ``ws_chat``'s
``on_attachment`` closure, ``day_chats._attachment_urls``,
``voice_tasks._wire_artifact`` and a fifth, private mime table inside the
mobile ``AttachmentCard``. All three server copies had drifted and all three
omitted PPTX, which ``files.py`` has been able to render the whole time and
``doc_generators._prewarm_preview`` pre-converts on every generated deck.

This module is the single source. It is PURE: no I/O, no settings, no DB, no
imports outside the standard library, so both the agent and the platform can
import it from anywhere (including inside a request handler) with no cost.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

# ── Kinds ────────────────────────────────────────────────────────────────
# A closed vocabulary. Deliberately coarser than MIME (the client wants to
# know "can I read this" / "can I play this", not which Office schema it is)
# and deliberately finer than "file" (a CSV and a PDF are not the same offer).


class ArtifactKind:
    TEXT = "text"
    MARKDOWN = "markdown"
    CODE = "code"
    DATA_CSV = "data_csv"
    DATA_JSON = "data_json"
    IMAGE = "image"
    PDF = "pdf"
    DOCX = "docx"
    XLSX = "xlsx"
    PPTX = "pptx"
    AUDIO = "audio"
    VIDEO = "video"
    ARCHIVE = "archive"
    OTHER = "other"


ALL_KINDS: Tuple[str, ...] = (
    ArtifactKind.TEXT, ArtifactKind.MARKDOWN, ArtifactKind.CODE,
    ArtifactKind.DATA_CSV, ArtifactKind.DATA_JSON, ArtifactKind.IMAGE,
    ArtifactKind.PDF, ArtifactKind.DOCX, ArtifactKind.XLSX,
    ArtifactKind.PPTX, ArtifactKind.AUDIO, ArtifactKind.VIDEO,
    ArtifactKind.ARCHIVE, ArtifactKind.OTHER,
)


class ArtifactRole:
    """What an attachment is FOR, which is not the same as what it is.

    ``final`` is the deliverable. ``source`` is evidence the model looked at
    (a browser screenshot, a user upload) — it belongs in the thread but it is
    not the answer. ``preview``/``thumbnail`` are derivatives a client must
    never draw a second card for. ``progress`` is an intermediate render.
    """

    FINAL = "final"
    PREVIEW = "preview"
    THUMBNAIL = "thumbnail"
    SOURCE = "source"
    PROGRESS = "progress"


#: Roles that must never produce their own card in a conversation — they are
#: attached to, or derived from, something that already has one.
NON_CARD_ROLES = frozenset({ArtifactRole.PREVIEW, ArtifactRole.THUMBNAIL})

#: Roles a channel adapter must never PUSH off-platform. "Do not draw a second
#: card" and "do not ship this to WhatsApp" are different questions, and reading
#: them off one set sent every browser screenshot — taken by a browser logged
#: into the user's own accounts — into the user's messenger thread on every
#: browsing turn started from a channel. Only the turn's ANSWER is delivered;
#: evidence stays visible in-app, which is what `source` was introduced for.
NON_DELIVERED_ROLES = frozenset(
    {ArtifactRole.PREVIEW, ArtifactRole.THUMBNAIL, ArtifactRole.SOURCE, ArtifactRole.PROGRESS}
)


# ── MIME tables ──────────────────────────────────────────────────────────

MIME_PDF = "application/pdf"
MIME_DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
MIME_XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
MIME_PPTX = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
MIME_MD = "text/markdown"
MIME_CSV = "text/csv"
MIME_JSON = "application/json"
MIME_TXT = "text/plain"
MIME_MP3 = "audio/mpeg"

_EXACT_MIME_KIND: Dict[str, str] = {
    MIME_PDF: ArtifactKind.PDF,
    MIME_DOCX: ArtifactKind.DOCX,
    MIME_XLSX: ArtifactKind.XLSX,
    MIME_PPTX: ArtifactKind.PPTX,
    "application/msword": ArtifactKind.DOCX,
    "application/vnd.ms-excel": ArtifactKind.XLSX,
    "application/vnd.ms-powerpoint": ArtifactKind.PPTX,
    MIME_MD: ArtifactKind.MARKDOWN,
    "text/x-markdown": ArtifactKind.MARKDOWN,
    MIME_CSV: ArtifactKind.DATA_CSV,
    "application/csv": ArtifactKind.DATA_CSV,
    MIME_JSON: ArtifactKind.DATA_JSON,
    "text/json": ArtifactKind.DATA_JSON,
    "application/zip": ArtifactKind.ARCHIVE,
    "application/x-zip-compressed": ArtifactKind.ARCHIVE,
    "application/gzip": ArtifactKind.ARCHIVE,
    "application/x-tar": ArtifactKind.ARCHIVE,
    MIME_TXT: ArtifactKind.TEXT,
    "text/html": ArtifactKind.CODE,
}

#: Extensions that mean "source code" whatever the server guessed the MIME to
#: be. A .py served as text/plain is still code to a reader that can colour it.
_CODE_EXTS = frozenset({
    ".py", ".js", ".jsx", ".ts", ".tsx", ".json5", ".sh", ".bash", ".zsh",
    ".rb", ".go", ".rs", ".java", ".kt", ".swift", ".c", ".h", ".cc", ".cpp",
    ".hpp", ".cs", ".php", ".pl", ".lua", ".sql", ".r", ".m", ".mm",
    ".yaml", ".yml", ".toml", ".ini", ".xml", ".html", ".htm", ".css",
    ".scss", ".dockerfile", ".gradle", ".tf",
})

_EXT_MIME: Dict[str, str] = {
    ".pdf": MIME_PDF, ".docx": MIME_DOCX, ".xlsx": MIME_XLSX, ".pptx": MIME_PPTX,
    ".md": MIME_MD, ".markdown": MIME_MD, ".csv": MIME_CSV, ".json": MIME_JSON,
    ".txt": MIME_TXT, ".log": MIME_TXT,
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".webp": "image/webp", ".gif": "image/gif", ".heic": "image/heic",
    ".mp3": MIME_MP3, ".m4a": "audio/mp4", ".aac": "audio/aac",
    ".wav": "audio/wav", ".ogg": "audio/ogg", ".opus": "audio/opus",
    ".mp4": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm",
    ".zip": "application/zip", ".html": "text/html", ".css": "text/css",
    ".py": "text/x-python", ".js": "text/javascript", ".ts": "text/x-typescript",
    ".sh": "text/x-shellscript", ".yaml": "text/yaml", ".yml": "text/yaml",
    ".xml": "application/xml", ".sql": "text/x-sql",
}

#: Canonical extension per MIME. Only the MIMEs we PRODUCE need an entry; the
#: fallback is the filename's own extension, then "".
_MIME_EXT: Dict[str, str] = {
    MIME_PDF: ".pdf", MIME_DOCX: ".docx", MIME_XLSX: ".xlsx", MIME_PPTX: ".pptx",
    MIME_MD: ".md", MIME_CSV: ".csv", MIME_JSON: ".json", MIME_TXT: ".txt",
    "text/html": ".html",
    "image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp",
    "image/gif": ".gif", "image/heic": ".heic",
    MIME_MP3: ".mp3", "audio/mp4": ".m4a", "audio/wav": ".wav",
    "audio/ogg": ".ogg", "audio/opus": ".opus", "audio/aac": ".aac",
    "video/mp4": ".mp4", "video/quicktime": ".mov", "video/webm": ".webm",
    "application/zip": ".zip",
}

#: Kind → the MIME we would PRODUCE for it. The inverse of the table above is
#: many-to-one, so this is written out rather than derived.
_KIND_MIME: Dict[str, str] = {
    ArtifactKind.TEXT: MIME_TXT,
    ArtifactKind.MARKDOWN: MIME_MD,
    ArtifactKind.CODE: MIME_TXT,
    ArtifactKind.DATA_CSV: MIME_CSV,
    ArtifactKind.DATA_JSON: MIME_JSON,
    ArtifactKind.IMAGE: "image/png",
    ArtifactKind.PDF: MIME_PDF,
    ArtifactKind.DOCX: MIME_DOCX,
    ArtifactKind.XLSX: MIME_XLSX,
    ArtifactKind.PPTX: MIME_PPTX,
    ArtifactKind.AUDIO: MIME_MP3,
    ArtifactKind.VIDEO: "video/mp4",
    ArtifactKind.ARCHIVE: "application/zip",
}

#: Pillow's ``Image.format`` → the MIME of those bytes. The only honest source
#: for an image's type is the bytes: the Kie/OpenAI call sites hardcoded
#: ``image/png`` at four places and a JPEG shipped as ``.png``.
_PIL_FORMAT_MIME: Dict[str, str] = {
    "PNG": "image/png", "JPEG": "image/jpeg", "JPEG2000": "image/jp2",
    "WEBP": "image/webp", "GIF": "image/gif", "BMP": "image/bmp",
    "TIFF": "image/tiff", "HEIF": "image/heic",
}


def kind_for_mime(mime: Optional[str], filename: Optional[str] = None) -> str:
    """The :class:`ArtifactKind` for a file. Never raises, never returns None.

    The filename is consulted only where MIME genuinely under-describes the
    file: source code is served as ``text/plain`` by every storage backend
    there is, and a reader that can syntax-colour it needs to know.
    """
    m = (mime or "").split(";")[0].strip().lower()
    ext = os.path.splitext((filename or "").strip())[1].lower()
    # A generic MIME is not evidence. Every storage backend serves .py, .csv
    # and .md as text/plain or octet-stream, so where the extension is more
    # specific than the MIME the extension wins.
    if m in ("", "text/plain", "application/octet-stream", "binary/octet-stream"):
        refined = _EXT_MIME.get(ext)
        if refined and refined != m:
            m = refined
    if ext in _CODE_EXTS and (not m or m.startswith("text/") or m in ("application/xml", "application/javascript")):
        return ArtifactKind.CODE
    if m in _EXACT_MIME_KIND:
        return _EXACT_MIME_KIND[m]
    if m.startswith("image/"):
        return ArtifactKind.IMAGE
    if m.startswith("audio/"):
        return ArtifactKind.AUDIO
    if m.startswith("video/"):
        return ArtifactKind.VIDEO
    if m.startswith("text/"):
        return ArtifactKind.TEXT
    # No usable MIME: fall back to the extension, which is all a legacy row has.
    if ext:
        guessed = _EXT_MIME.get(ext)
        if guessed and guessed != m:
            return kind_for_mime(guessed, None if ext in _CODE_EXTS else filename)
    return ArtifactKind.OTHER


def mime_for_kind(kind: Optional[str]) -> str:
    """The MIME we would produce for a kind. ``application/octet-stream`` for
    a kind with no producer (``other``, and ``video`` has no producer either —
    it is in the table so a truthful refusal can still name a type)."""
    return _KIND_MIME.get((kind or "").strip().lower(), "application/octet-stream")


def ext_for_mime(mime: Optional[str], filename: Optional[str] = None) -> str:
    """Canonical dotted extension for a MIME, or the filename's own, or ``""``."""
    m = (mime or "").split(";")[0].strip().lower()
    if m in _MIME_EXT:
        return _MIME_EXT[m]
    return os.path.splitext((filename or "").strip())[1].lower()


def mime_for_filename(name: Optional[str]) -> str:
    """Best MIME for a filename, from our own table first and the stdlib
    second. ``application/octet-stream`` when neither knows."""
    ext = os.path.splitext((name or "").strip())[1].lower()
    if ext in _EXT_MIME:
        return _EXT_MIME[ext]
    import mimetypes
    guessed, _ = mimetypes.guess_type(name or "")
    return guessed or "application/octet-stream"


#: Per-channel upload ceilings in bytes, by kind class. A generated file that
#: does not fit is NOT dropped silently — the handler says so in the reply and
#: the file is still in the thread and in the user's Files. Conservative on
#: purpose: an upload refused by the platform costs the user their answer's
#: credibility, and every one of these numbers is the smaller of the
#: documented cap and what the adapter has been seen to accept.
_CHANNEL_MAX_BYTES = {
    "telegram": {"image": 10 * 1024 * 1024, "other": 50 * 1024 * 1024},
    "whatsapp": {"image": 5 * 1024 * 1024, "other": 64 * 1024 * 1024},
    "discord": {"image": 8 * 1024 * 1024, "other": 8 * 1024 * 1024},
    "slack": {"image": 50 * 1024 * 1024, "other": 50 * 1024 * 1024},
}
_DEFAULT_MAX_BYTES = {"image": 15 * 1024 * 1024, "other": 25 * 1024 * 1024}


def channel_size_limit(channel: Optional[str], mime: Optional[str],
                       filename: Optional[str] = None) -> int:
    """Largest file this channel will accept for this kind, in bytes."""
    table = _CHANNEL_MAX_BYTES.get((channel or "").strip().lower(), _DEFAULT_MAX_BYTES)
    slot = "image" if kind_for_mime(mime, filename) == ArtifactKind.IMAGE else "other"
    return int(table.get(slot, table.get("other", 0)))


def mime_for_pil_format(pil_format: Optional[str]) -> Optional[str]:
    """MIME for a Pillow ``Image.format`` string, or None if unrecognised."""
    return _PIL_FORMAT_MIME.get((pil_format or "").strip().upper())


def canonical_filename(name: Optional[str], mime: Optional[str]) -> str:
    """``name`` with the extension that matches ``mime``.

    The STEM is never touched — a model-chosen name is the user's name for the
    thing. The extension is replaced only when the one it has does not already
    denote the same MIME, so ``portrait.jpg`` + ``image/jpeg`` is left exactly
    as it is while ``portrait.jpg`` + ``image/png`` becomes ``portrait.png``
    and ``sunset`` + ``image/png`` becomes ``sunset.png``.
    """
    base = os.path.basename((name or "").replace("\\", "/").strip())
    stem, cur = os.path.splitext(base)
    if not stem:
        stem = base or "file"
        cur = ""
    want = ext_for_mime(mime, base)
    if not want:
        return base or stem
    if cur and _EXT_MIME.get(cur.lower()) == (mime or "").split(";")[0].strip().lower():
        return base
    return f"{stem}{want}"


# ── The ONE preview policy ───────────────────────────────────────────────

_PREVIEW_NATIVE = frozenset({MIME_PDF})
_PREVIEW_CONVERTED = frozenset({MIME_DOCX, MIME_XLSX, MIME_PPTX})


def preview_policy(mime: Optional[str]) -> str:
    """``'native'`` | ``'converted'`` | ``'none'`` — may this file be previewed,
    and does previewing it cost a conversion?

    The ONE predicate. ``!= 'none'`` is what every serializer asks before it
    advertises a ``preview_url``. PPTX is in ``converted`` because
    ``files.py::render_preview`` has always rendered it through the same
    LibreOffice path as DOCX (files.py:669-695) and ``_prewarm_preview``
    already pays for the conversion on every generated deck — the three
    drifted server-side copies of this set simply never listed it, so the work
    was done and never offered.
    """
    m = (mime or "").split(";")[0].strip().lower()
    if m in _PREVIEW_NATIVE or m.startswith("image/"):
        return "native"
    if m in _PREVIEW_CONVERTED:
        return "converted"
    return "none"


def has_preview(mime: Optional[str]) -> bool:
    return preview_policy(mime) != "none"
