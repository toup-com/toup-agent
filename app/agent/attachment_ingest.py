"""One classifier for every inbound attachment.

Round 46, incident 4. What this replaces, and why each branch exists:

* ``agent_runner._extract_document_text`` decided what a file was from its
  EXTENSION, ran on the event loop of a 1.00-CPU container, capped nothing, and
  had exactly two outcomes — text, or a line of prose. A password-protected PDF
  put the raw pypdf exception string into the model's context; a scanned PDF
  reached the model as "could not extract text" even though every image on the
  same turn was already being read by a vision model.
* Zero ``[AGENT] Document extracted`` lines exist fleet-wide across the whole
  Loki retention window, so nothing about the old path was observable either.

Every branch now ends in a STATUS. The statuses are the vocabulary the user,
the model and the durable ``Message.attachments[i].ingest`` all share:

    ok | truncated | password_protected | unreadable | unsupported | empty

Nothing in here raises, and no exception text ever reaches the model — a
stack string in the prompt is both a leak and a lie about what the file is.

Everything here is SYNCHRONOUS and CPU-bound on purpose: callers run it with
``asyncio.to_thread``. It is the half of the attachment path that used to block
the event loop, which is the same starvation signature as the incident-2 WS
accept delay.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import zipfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.agent.attachment_limits import (
    KIND_DOCUMENT,
    KIND_IMAGE,
    MAX_DECODED_IMAGE_PIXELS,
    MAX_EXTRACTED_CHARS_PER_DOCUMENT,
    MAX_PDF_PAGES_RASTERIZED,
    MODEL_IMAGE_LONG_EDGE,
    SCAN_MIN_CHARS,
    kind_for,
    normalize_mime,
)

logger = logging.getLogger(__name__)

STATUS_OK = "ok"
STATUS_TRUNCATED = "truncated"
STATUS_PASSWORD_PROTECTED = "password_protected"
STATUS_UNREADABLE = "unreadable"
STATUS_UNSUPPORTED = "unsupported"
STATUS_EMPTY = "empty"

#: What the model is told to say about a file it could not be given. Keyed by
#: status so the wording cannot drift between the prompt and the app.
MODEL_GUIDANCE = {
    STATUS_PASSWORD_PROTECTED: (
        "This file is password-protected. Tell the user it needs a password; "
        "do not claim to have read it."
    ),
    STATUS_UNREADABLE: (
        "This file could not be read. Tell the user it did not open; "
        "do not guess at its contents."
    ),
    STATUS_UNSUPPORTED: (
        "This kind of file cannot be read yet. Tell the user which formats work "
        "(images, PDF, DOCX, PPTX, XLSX, text, CSV, JSON, ZIP)."
    ),
    STATUS_EMPTY: (
        "This file opened but held no readable text. Tell the user that; "
        "do not invent contents."
    ),
}

#: Magic-byte signatures, used when python-magic is unavailable. The extension
#: is the user's claim; the first bytes are the file's own.
_SIGNATURES = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"%PDF-", "application/pdf"),
)


#: libmagic verdicts that do NOT settle the question. ``application/zip`` is
#: here because every OOXML container IS a zip: a libmagic build that answers
#: ``application/zip`` for a 4 KiB DOCX prefix (this happens — the verdict for a
#: truncated OOXML prefix is build-dependent) would otherwise route a Word
#: document into the archive branch and stamp it ``kind=archive``. Both fall
#: through to the ``PK\x03\x04`` disambiguation below.
_INDECISIVE_MIMES = frozenset({
    "application/octet-stream",
    "application/zip",
    "application/x-zip-compressed",
})


def _magic_mime(data: bytes) -> Optional[str]:
    """libmagic's verdict, or ``None`` when it is unavailable or raises.

    Split out of :func:`sniff_mime` so the branch that actually RUNS IN
    PRODUCTION can be tested: ``requirements.agent.txt`` ships python-magic and
    Dockerfile.agent ships libmagic1, while CI installs neither — so every
    sniff test was certifying the fallback and none of them touched this.
    """
    try:
        import magic  # type: ignore
    except Exception:
        return None
    try:
        detected = magic.from_buffer((data or b"")[:4096], mime=True)
    except Exception:
        return None
    return ((detected or "").split(";")[0].strip().lower()) or None


def sniff_mime(data: bytes, filename: str = "", declared: Optional[str] = None) -> str:
    """The file's real type.

    python-magic when it is importable (libmagic is in Dockerfile.agent), then
    signatures, then the declared type / extension. A picker's declared type is
    routinely wrong — expo's image picker hands base64 JPEG bytes under a
    ``.heic`` name — and the model, the vision API and the storage key all read
    this value.
    """
    detected = _magic_mime(data)
    if detected == "text/plain":
        # libmagic cannot tell markdown/csv/json apart from plain text; the
        # extension is better evidence for those three.
        return normalize_mime(declared, filename) or "text/plain"
    if detected and detected not in _INDECISIVE_MIMES:
        return normalize_mime(detected, filename)
    head = data[:16]
    for sig, mime in _SIGNATURES:
        if head.startswith(sig):
            return mime
    if data[:4] == b"PK\x03\x04":
        # Every OOXML container is a zip; the extension names which one.
        by_name = normalize_mime(declared, filename)
        if by_name in (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ):
            return by_name
        return "application/zip"
    if data[4:12] == b"ftypheic" or data[4:12] == b"ftypheix" or data[4:12] == b"ftypmif1":
        return "image/heic"
    if data[:12].startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    return normalize_mime(declared, filename)


@dataclass
class IngestedAttachment:
    """One inbound file, classified. `to_record()` is what the prompt builder
    and the durable ``Message.attachments[i].ingest`` both read."""

    name: str
    mime: str
    kind: Optional[str]
    size_bytes: int
    status: str
    reason: Optional[str] = None
    #: Extracted text for a document, already capped.
    text: Optional[str] = None
    #: The bytes actually shown to the model for an image (the derivative).
    image_bytes: Optional[bytes] = None
    image_mime: Optional[str] = None
    #: Rasterized pages of a scanned PDF, in page order.
    page_images: List[bytes] = field(default_factory=list)
    page_count: Optional[int] = None
    truncated: bool = False
    sha256: Optional[str] = None

    def to_record(self) -> Dict[str, Any]:
        """The durable half — no bytes, no contents, safe to persist and log."""
        rec: Dict[str, Any] = {"status": self.status}
        if self.reason:
            rec["reason"] = self.reason
        if self.kind:
            rec["kind"] = self.kind
        if self.page_count is not None:
            rec["page_count"] = self.page_count
        if self.truncated:
            rec["truncated"] = True
        if self.text is not None:
            rec["chars"] = len(self.text)
        if self.page_images:
            rec["rasterized_pages"] = len(self.page_images)
        return rec


def _cap_text(text: str) -> tuple[str, bool]:
    if len(text) <= MAX_EXTRACTED_CHARS_PER_DOCUMENT:
        return text, False
    return text[:MAX_EXTRACTED_CHARS_PER_DOCUMENT], True


def downscale_image(data: bytes, mime: str, long_edge: int = MODEL_IMAGE_LONG_EDGE) -> tuple[bytes, str]:
    """The copy the model sees. Returns the original unchanged when it is
    already small enough or when Pillow cannot read it — a picture is worth
    delivering even when its header will not parse.

    A modern iPhone photo is ~4000 px; sent whole it costs several thousand
    tokens and four full copies inside a 768 MiB container, for detail no
    vision model uses.
    """
    try:
        from PIL import Image as _PILImage

        # A DELIBERATE number rather than Pillow's default, which only WARNS
        # between MAX_IMAGE_PIXELS and 2x it: a small PNG/WebP declaring
        # 9000x9000 decodes to ~324 MB of pixels without any error, inside a
        # 768 MiB container. The pre-`load()` size test below is the real gate;
        # this makes the library's own ceiling explicit for every other caller.
        _PILImage.MAX_IMAGE_PIXELS = MAX_DECODED_IMAGE_PIXELS

        try:  # HEIC/HEIF decode — same opener the image tools register.
            import pillow_heif  # type: ignore

            pillow_heif.register_heif_opener()
        except Exception:
            pass
        with _PILImage.open(io.BytesIO(data)) as im:
            # `open()` only reads the header; `load()` allocates the pixels. A
            # declared size is untrusted input, so it is tested BEFORE the
            # allocation, not caught after it (an OOM kill catches nothing).
            _w, _h = im.size
            if _w * _h > MAX_DECODED_IMAGE_PIXELS:
                logger.warning(
                    "attachment: image refused for decode px=%d", int(_w) * int(_h))
                return data, mime
            im.load()
            fits = max(im.size) <= long_edge
            if fits and mime in ("image/png", "image/jpeg", "image/webp", "image/gif"):
                return data, mime
            if im.mode not in ("RGB", "RGBA"):
                im = im.convert("RGBA" if "A" in im.getbands() else "RGB")
            if not fits:
                im.thumbnail((long_edge, long_edge), _PILImage.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format="WEBP", quality=82, method=4)
            out = buf.getvalue()
        if out and (len(out) < len(data) or mime not in ("image/png", "image/jpeg", "image/webp", "image/gif")):
            return out, "image/webp"
        return data, mime
    except Exception:
        logger.debug("attachment: could not build model derivative", exc_info=True)
        return data, mime


def _extract_pdf(data: bytes) -> tuple[str, Optional[int], Optional[str]]:
    """``(text, page_count, status_override)``."""
    try:
        from pypdf import PdfReader
    except Exception:
        return "", None, None
    try:
        reader = PdfReader(io.BytesIO(data))
        if getattr(reader, "is_encrypted", False):
            # An empty-password PDF is encrypted but readable; try that once
            # before telling the user their file needs a password.
            try:
                if reader.decrypt("") == 0:
                    return "", None, STATUS_PASSWORD_PROTECTED
            except Exception:
                return "", None, STATUS_PASSWORD_PROTECTED
        pages = list(reader.pages)
        text = "\n\n".join((p.extract_text() or "") for p in pages).strip()
        return text, len(pages), None
    except Exception:
        logger.debug("attachment: pdf extraction failed", exc_info=True)
        return "", None, STATUS_UNREADABLE


def raster_scale(page_w: float, page_h: float) -> float:
    """The render scale for one PDF page: 144 dpi, CLAMPED by the page's size.

    A 386-byte PDF may declare the maximum legal MediaBox (14400 pt), and at a
    fixed scale of 2.0 that asks pypdfium2 for a 28800x28800 bitmap — 3.3 GB,
    per page, up to `MAX_PDF_PAGES_RASTERIZED` times, inside a 768 MiB
    container. The page COUNT was bounded and the page SIZE was not.
    `pil.thumbnail(...)` afterwards is no defence: the full bitmap has to exist
    first, and an OOM kill is not an exception anything can catch.
    """
    longest = max(float(page_w or 0.0), float(page_h or 0.0))
    if longest <= 0:
        return 2.0
    return min(2.0, (2.0 * MODEL_IMAGE_LONG_EDGE) / longest)


def rasterize_pdf(data: bytes, max_pages: int = MAX_PDF_PAGES_RASTERIZED) -> List[bytes]:
    """A scanned PDF's pages as images, for the vision model.

    There is no OCR in this image and none is wanted: the same turn already
    hands every photo to a model that reads text out of pictures. Silence on a
    scanned PDF was a capability we already had and never connected.
    """
    try:
        import pypdfium2 as pdfium  # type: ignore
    except Exception:
        return []
    out: List[bytes] = []
    try:
        doc = pdfium.PdfDocument(io.BytesIO(data))
        try:
            for i in range(min(len(doc), max_pages)):
                page = doc[i]
                # 2.0 ≈ 144 dpi — enough for a vision model to read body text.
                # CLAMPED by the page's own size: a 386-byte PDF may declare the
                # maximum legal MediaBox (14400 pt), which at a fixed scale of
                # 2.0 asks pypdfium2 for a 28800x28800 bitmap — 3.3 GB per page,
                # up to 20 pages, inside a 768 MiB container. `MAX_PDF_PAGES_…`
                # bounds the page COUNT and nothing bounded the page SIZE.
                try:
                    _pw, _ph = page.get_size()
                except Exception:
                    _pw = _ph = 0.0
                _scale = raster_scale(_pw, _ph)
                if _scale <= 0:
                    continue
                pil = page.render(scale=_scale).to_pil()
                pil.thumbnail((MODEL_IMAGE_LONG_EDGE, MODEL_IMAGE_LONG_EDGE))
                buf = io.BytesIO()
                pil.convert("RGB").save(buf, format="WEBP", quality=80, method=4)
                out.append(buf.getvalue())
        finally:
            try:
                doc.close()
            except Exception:
                pass
    except Exception:
        logger.debug("attachment: pdf rasterize failed", exc_info=True)
        return out
    return out


#: Declared-uncompressed ceiling for the MARKUP parts of an OOXML container.
#: Media parts are excluded: an embedded photo is already compressed, so its
#: uncompressed size is bounded by the upload's own size. The XML is where the
#: amplification lives — measured, a 200 KB .docx whose `word/document.xml`
#: deflates 1023:1 drove `ingest_one` to 433 MB peak RSS.
MAX_OOXML_XML_BYTES = 32 * 1024 * 1024

_OOXML_MARKUP_SUFFIXES = (".xml", ".rels", ".txt", ".vml", ".json")


def _ooxml_budget_ok(data: bytes) -> bool:
    """Refuse an OOXML bomb BEFORE python-docx/pptx/openpyxl sees it.

    `_extract_zip` streams every read under two numbers and its own docstring
    explains why; these three extractors handed the whole archive to a library
    that materialises every part in full, with no ratio test and no budget —
    the fourth decompression-bomb class, and the one left open. Same two
    numbers, read before a single part is inflated.
    """
    try:
        if not zipfile.is_zipfile(io.BytesIO(data)):
            return False
        xml_bytes = 0
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                declared = int(getattr(info, "file_size", 0) or 0)
                packed = max(1, int(getattr(info, "compress_size", 0) or 0))
                if declared // packed > MAX_ZIP_ENTRY_RATIO:
                    logger.warning(
                        "attachment: ooxml entry refused ratio=%d", declared // packed)
                    return False
                if info.filename.lower().endswith(_OOXML_MARKUP_SUFFIXES):
                    xml_bytes += declared
        if xml_bytes > MAX_OOXML_XML_BYTES:
            logger.warning("attachment: ooxml markup budget exceeded")
            return False
        return True
    except Exception:
        return False


class _OoxmlRefused(Exception):
    """Raised past the budget gate: `ingest_one` turns it into `unreadable`."""


def _extract_docx(data: bytes) -> str:
    if not _ooxml_budget_ok(data):
        raise _OoxmlRefused("docx")
    from docx import Document as DocxDocument

    doc = DocxDocument(io.BytesIO(data))
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())


def _extract_pptx(data: bytes) -> str:
    if not _ooxml_budget_ok(data):
        raise _OoxmlRefused("pptx")
    from pptx import Presentation as PptxPresentation

    prs = PptxPresentation(io.BytesIO(data))
    parts: List[str] = []
    for i, slide in enumerate(prs.slides, 1):
        slide_parts = [f"--- Slide {i} ---"]
        for shape in slide.shapes:
            if shape.has_text_frame:
                t = shape.text_frame.text.strip()
                if t:
                    slide_parts.append(t)
            if getattr(shape, "has_table", False):
                for row in shape.table.rows:
                    row_text = " | ".join(cell.text.strip() for cell in row.cells)
                    if row_text.strip(" |"):
                        slide_parts.append(row_text)
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                slide_parts.append(f"[Speaker Notes] {notes}")
        parts.append("\n".join(slide_parts))
    return "\n\n".join(parts)


def _extract_xlsx(data: bytes) -> str:
    if not _ooxml_budget_ok(data):
        raise _OoxmlRefused("xlsx")
    from openpyxl import load_workbook

    wb = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    parts: List[str] = []
    try:
        for ws in wb.worksheets:
            parts.append(f"--- Sheet: {ws.title} ---")
            for row in ws.iter_rows(values_only=True):
                cells = ["" if c is None else str(c) for c in row]
                if any(c.strip() for c in cells):
                    parts.append(" | ".join(cells))
                if len("\n".join(parts)) > MAX_EXTRACTED_CHARS_PER_DOCUMENT:
                    break
    finally:
        try:
            wb.close()
        except Exception:
            pass
    return "\n".join(parts)


#: Total UNCOMPRESSED bytes one archive may yield. Four bytes per extracted
#: character is generous for UTF-8 and keeps a whole archive inside the
#: per-document text cap it feeds.
MAX_ZIP_UNCOMPRESSED_BYTES = MAX_EXTRACTED_CHARS_PER_DOCUMENT * 4
#: Entries considered, per archive.
MAX_ZIP_ENTRIES = 50
#: A ratio no real document reaches. Measured: 8 MiB of zeros compresses to
#: 8265 bytes — 1015:1.
MAX_ZIP_ENTRY_RATIO = 200

_ZIP_TEXT_EXTS = {".txt", ".md", ".json", ".csv", ".yaml", ".yml", ".py", ".js", ".ts"}
_ZIP_DOC_EXTS = {".pdf", ".docx", ".pptx"}


def _extract_zip(data: bytes) -> str:
    """Text out of an archive, under a hard uncompressed budget.

    Round 46 review: this was an unguarded decompression bomb. ``zf.read(info)``
    materialises a whole entry before any cap runs, and ``application/zip`` is a
    DOCUMENT_MIME with a 25 MB ceiling — 25 MB in at a measured 1015:1 ratio is
    ~25 GB read into a 768 MiB container, which the OOM killer reaches before
    any ``except`` does. It also recursed: ``sniff_mime`` classifies by BYTES, so
    a zip NAMED ``a.pdf`` sniffs back to ``application/zip`` and re-entered here,
    50 entries per level, i.e. 50**depth full decompressions from one ~1 MB
    upload. Both are structural now — every read is streamed and truncated to
    the remaining budget, and a nested archive is never re-entered.
    """
    if not zipfile.is_zipfile(io.BytesIO(data)):
        return ""
    parts: List[str] = []
    budget = MAX_ZIP_UNCOMPRESSED_BYTES
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for i, info in enumerate(zf.infolist()):
            if info.is_dir() or i >= MAX_ZIP_ENTRIES:
                continue
            if budget <= 0:
                break
            inner_ext = os.path.splitext(info.filename.lower())[1]
            if inner_ext not in _ZIP_TEXT_EXTS and inner_ext not in _ZIP_DOC_EXTS:
                continue
            declared = int(getattr(info, "file_size", 0) or 0)
            packed = max(1, int(getattr(info, "compress_size", 0) or 0))
            if declared // packed > MAX_ZIP_ENTRY_RATIO:
                logger.warning("attachment: zip entry refused ratio=%d", declared // packed)
                continue
            try:
                # NEVER `zf.read(info)`: that decompresses the entry in full
                # before anything can cap it.
                with zf.open(info) as fh:
                    raw = fh.read(budget)
            except Exception:
                continue
            budget -= len(raw)
            if inner_ext in _ZIP_TEXT_EXTS:
                parts.append(
                    f"=== {info.filename} ===\n{raw.decode('utf-8', errors='replace')}"
                )
                continue
            if sniff_mime(raw, info.filename) == "application/zip":
                # A nested archive, whatever it is named. Recursion here is the
                # zip-quine amplifier, and no real attachment needs it.
                continue
            try:
                inner = ingest_one(raw, info.filename, None)
            except Exception:
                continue
            if inner.text:
                parts.append(f"=== {info.filename} ===\n{inner.text}")
    return "\n\n".join(parts)


def ingest_one(data: bytes, filename: str, declared_mime: Optional[str] = None) -> IngestedAttachment:
    """Classify and extract ONE attachment. Synchronous; never raises."""
    name = os.path.basename(filename or "file") or "file"
    size = len(data or b"")
    mime = sniff_mime(data or b"", name, declared_mime)
    kind = kind_for(mime, name)
    sha = hashlib.sha256(data or b"").hexdigest()

    if size == 0:
        return IngestedAttachment(
            name=name, mime=mime, kind=kind, size_bytes=0, status=STATUS_EMPTY, sha256=sha
        )
    if kind is None:
        return IngestedAttachment(
            name=name,
            mime=mime,
            kind=None,
            size_bytes=size,
            status=STATUS_UNSUPPORTED,
            reason="attachment_unsupported",
            sha256=sha,
        )

    if kind == KIND_IMAGE:
        model_bytes, model_mime = downscale_image(data, mime)
        return IngestedAttachment(
            name=name,
            mime=mime,
            kind=KIND_IMAGE,
            size_bytes=size,
            status=STATUS_OK,
            image_bytes=model_bytes,
            image_mime=model_mime,
            sha256=sha,
        )

    text = ""
    page_count: Optional[int] = None
    status: Optional[str] = None
    try:
        if mime == "application/pdf":
            text, page_count, status = _extract_pdf(data)
            if status is None and len(text.strip()) < SCAN_MIN_CHARS:
                pages = rasterize_pdf(data)
                if pages:
                    return IngestedAttachment(
                        name=name,
                        mime=mime,
                        kind=KIND_DOCUMENT,
                        size_bytes=size,
                        status=STATUS_OK,
                        text=text.strip() or None,
                        page_images=pages,
                        page_count=page_count,
                        truncated=bool(page_count and page_count > len(pages)),
                        sha256=sha,
                    )
        elif mime.endswith("wordprocessingml.document"):
            text = _extract_docx(data)
        elif mime.endswith("presentationml.presentation"):
            text = _extract_pptx(data)
        elif mime.endswith("spreadsheetml.sheet"):
            text = _extract_xlsx(data)
        elif mime == "application/zip":
            text = _extract_zip(data)
        else:
            text = (data.decode("utf-8", errors="replace")).strip()
    except Exception:
        # Deliberately no exception text anywhere near the prompt: it leaks
        # library internals into the model's context and tells the user nothing.
        logger.debug("attachment: extraction failed kind=%s", kind, exc_info=True)
        status = STATUS_UNREADABLE

    if status == STATUS_PASSWORD_PROTECTED:
        return IngestedAttachment(
            name=name,
            mime=mime,
            kind=KIND_DOCUMENT,
            size_bytes=size,
            status=STATUS_PASSWORD_PROTECTED,
            reason=STATUS_PASSWORD_PROTECTED,
            page_count=page_count,
            sha256=sha,
        )
    if status == STATUS_UNREADABLE:
        return IngestedAttachment(
            name=name,
            mime=mime,
            kind=KIND_DOCUMENT,
            size_bytes=size,
            status=STATUS_UNREADABLE,
            reason=STATUS_UNREADABLE,
            page_count=page_count,
            sha256=sha,
        )

    text = (text or "").strip()
    if not text:
        return IngestedAttachment(
            name=name,
            mime=mime,
            kind=KIND_DOCUMENT,
            size_bytes=size,
            status=STATUS_EMPTY,
            reason=STATUS_EMPTY,
            page_count=page_count,
            sha256=sha,
        )
    text, truncated = _cap_text(text)
    return IngestedAttachment(
        name=name,
        mime=mime,
        kind=KIND_DOCUMENT,
        size_bytes=size,
        status=STATUS_TRUNCATED if truncated else STATUS_OK,
        text=text,
        page_count=page_count,
        truncated=truncated,
        sha256=sha,
    )


def record_from_ingested(ing: IngestedAttachment) -> Dict[str, Any]:
    """One ingest result → the record `_build_attachment_blocks` reads.

    The bytes are base64'd HERE rather than at render time so the blocks
    builder stays pure and the encode happens once, on the thread that already
    holds the image.
    """
    import base64 as _b64

    rec: Dict[str, Any] = {
        "name": ing.name,
        "mime": ing.mime,
        "kind": ing.kind,
        "status": ing.status,
        "reason": ing.reason,
        "text": ing.text,
        "page_count": ing.page_count,
        "truncated": ing.truncated,
        "attachment_id": None,
    }
    if ing.image_bytes:
        rec["image_b64"] = _b64.standard_b64encode(ing.image_bytes).decode("ascii")
        rec["image_mime"] = ing.image_mime or ing.mime
    if ing.page_images:
        rec["page_images_b64"] = [
            _b64.standard_b64encode(p).decode("ascii") for p in ing.page_images
        ]
    return rec


def ingest_path(path: str, declared_mime: Optional[str] = None) -> IngestedAttachment:
    """Ingest a file already on disk (the legacy/channel path)."""
    name = os.path.basename(path)
    try:
        with open(path, "rb") as f:
            data = f.read()
    except Exception:
        logger.debug("attachment: could not read %s", "<path>", exc_info=True)
        return IngestedAttachment(
            name=name,
            mime=normalize_mime(declared_mime, name),
            kind=None,
            size_bytes=0,
            status=STATUS_UNREADABLE,
            reason=STATUS_UNREADABLE,
        )
    return ingest_one(data, name, declared_mime)
