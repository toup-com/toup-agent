"""The attachment limit table — the server half, and the backstop.

The same numbers live in the app (`src/shared/attachmentLimits.ts`) and the web
(`frontend/src/modules/chat/attachmentLimits.ts`). The clients enforce them so
the user gets a refusal that NAMES THE FILE before anything is uploaded; this
module enforces them again because an app build in the field carries an older
copy of the table and a WS frame is not a trusted input.

Round 46, incident 4: before this table there was no size gate anywhere on the
attachment path — not in the app, not in the platform proxy, not in the agent.
The only ceiling was uvicorn's 16 MiB websocket frame, and breaching it closed
the socket, which the app rendered as the user's internet being at fault.

Every refusal carries a STABLE REASON CODE. The code is what crosses the wire;
the sentence is composed on the client, next to the file's name.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

MAX_ATTACHMENTS_PER_TURN = 8
MAX_BYTES_PER_IMAGE = 15 * 1024 * 1024
MAX_BYTES_PER_DOCUMENT = 25 * 1024 * 1024
MAX_TOTAL_BYTES_PER_TURN = 60 * 1024 * 1024
#: base64 total for the LEGACY in-frame ``payload.media`` path only. Kept
#: forever: app builds 123–126 and the web client send bytes inside the chat
#: frame, and a container has 768 MiB with 1.00 CPU.
LEGACY_MAX_FRAME_MEDIA_BYTES = 8 * 1024 * 1024
#: The long edge of the derivative actually shown to the model. Matches
#: ``doc_generators._THUMB_LONG_EDGE`` on purpose — the derivative _persist
#: already writes IS the model's copy, so an uploaded image is never resized
#: twice.
MODEL_IMAGE_LONG_EDGE = 1280
MAX_EXTRACTED_CHARS_PER_DOCUMENT = 200_000
MAX_EXTRACTED_CHARS_PER_TURN = 400_000
MAX_PDF_PAGES_RASTERIZED = 20
#: Pixels one untrusted image may be DECODED to. A file's declared dimensions
#: cost ~4 bytes each once decoded, so a small PNG can ask for hundreds of MB
#: inside a 768 MiB container; Pillow's own default only warns. Eight times the
#: model's long edge, squared — far above any real photo.
MAX_DECODED_IMAGE_PIXELS = (MODEL_IMAGE_LONG_EDGE * 8) ** 2
#: Model-visible IMAGE blocks per turn. `MAX_ATTACHMENTS_PER_TURN` counts FILES,
#: and one scanned PDF legitimately becomes `MAX_PDF_PAGES_RASTERIZED` images —
#: eight of them is 160 image blocks in a single request.
MAX_MODEL_IMAGES_PER_TURN = 24
#: Below this many extracted characters a PDF is treated as SCANNED and
#: rasterized for the vision model instead of being reported as empty.
SCAN_MIN_CHARS = 200

KIND_IMAGE = "image"
KIND_DOCUMENT = "document"

#: Reason codes. Shared verbatim with both clients.
REASON_TOO_LARGE = "attachment_too_large"
REASON_TOTAL_TOO_LARGE = "attachment_total_too_large"
REASON_UNSUPPORTED = "attachment_unsupported"
REASON_COUNT_EXCEEDED = "attachment_count_exceeded"

IMAGE_MIMES = {
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/gif",
    "image/heic",
    "image/heif",
}

DOCUMENT_MIMES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "text/plain",
    "text/markdown",
    "text/csv",
    "application/json",
    "application/zip",
}

_EXT_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".heic": "image/heic",
    ".heif": "image/heif",
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".csv": "text/csv",
    ".json": "application/json",
    ".yaml": "text/plain",
    ".yml": "text/plain",
    ".zip": "application/zip",
    ".x-zip-compressed": "application/zip",
}

_ALIASES = {
    "image/jpg": "image/jpeg",
    "image/pjpeg": "image/jpeg",
    "application/x-zip-compressed": "application/zip",
    "application/x-javascript": "text/plain",
    "text/x-markdown": "text/markdown",
}


def normalize_mime(declared: Optional[str], filename: str = "") -> str:
    """The MIME we will act on.

    A picker's declared type is frequently absent or ``octet-stream``; the
    extension is the only other evidence before the bytes are sniffed.
    """
    d = (declared or "").split(";")[0].strip().lower()
    d = _ALIASES.get(d, d)
    if d and d != "application/octet-stream" and (d in IMAGE_MIMES or d in DOCUMENT_MIMES):
        return d
    ext = os.path.splitext((filename or "").lower())[1]
    by_ext = _EXT_MIME.get(ext)
    if by_ext:
        return by_ext
    return d or "application/octet-stream"


def kind_for(mime: str, filename: str = "") -> Optional[str]:
    """``"image"`` / ``"document"`` / ``None`` when unsupported."""
    m = normalize_mime(mime, filename)
    if m in IMAGE_MIMES:
        return KIND_IMAGE
    if m in DOCUMENT_MIMES:
        return KIND_DOCUMENT
    return None


def max_bytes_for(kind: str) -> int:
    return MAX_BYTES_PER_IMAGE if kind == KIND_IMAGE else MAX_BYTES_PER_DOCUMENT


def check_one(size_bytes: int, mime: str, filename: str = "") -> Tuple[Optional[str], Optional[str]]:
    """``(kind, reason)`` — exactly one of the two is ``None``."""
    kind = kind_for(mime, filename)
    if kind is None:
        return None, REASON_UNSUPPORTED
    if size_bytes > max_bytes_for(kind):
        return None, REASON_TOO_LARGE
    return kind, None
