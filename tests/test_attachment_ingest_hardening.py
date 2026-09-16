"""attachment_ingest — the untrusted-parser hardening (round 46 review).

Every branch in this module parses a file an authenticated user chose, inside a
768 MiB / 1.00-CPU container, and (before the platform fallback was closed) also
inside the shared multi-tenant platform process. Three of them were unbounded:

* `_extract_zip` decompressed each entry in full with `zf.read(info)` and
  recursed into nested archives 50 entries wide;
* `rasterize_pdf` rendered at a fixed scale, so a declared MediaBox decided the
  bitmap size;
* `downscale_image` called `im.load()` on whatever the header claimed.

These are behavioural tests over the real functions. Platform sweep.
"""

from __future__ import annotations

import io
import os
import struct
import sys
import zipfile
import zlib

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "fixtures"))
import attachments as fx  # noqa: E402

from app.agent.attachment_ingest import (  # noqa: E402
    MAX_ZIP_UNCOMPRESSED_BYTES,
    _extract_zip,
    downscale_image,
    ingest_one,
    raster_scale,
    sniff_mime,
)
from app.agent.attachment_limits import (  # noqa: E402
    MAX_DECODED_IMAGE_PIXELS,
    MODEL_IMAGE_LONG_EDGE,
)


# ── zip: a bomb is refused, not merely capped afterwards ─────────────────

def _zip_with(entries) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, payload in entries:
            zf.writestr(name, payload)
    return buf.getvalue()


def test_a_high_ratio_entry_is_never_decompressed():
    """8 MiB of zeros compresses to ~8 KB — a measured 1015:1. `application/zip`
    is a DOCUMENT_MIME with a 25 MB ceiling, so the old `zf.read(info)` turned
    a 25 MB upload into ~25 GB of RAM before `_cap_text` could ever run."""
    bomb = _zip_with([("payload.txt", b"\x00" * (8 * 1024 * 1024))])
    assert len(bomb) < 64 * 1024, "fixture is not actually compressed"
    # Read through `_extract_zip` directly: `ingest_one` caps the TEXT at
    # MAX_EXTRACTED_CHARS_PER_DOCUMENT afterwards either way, so the final
    # string cannot tell "refused" from "decompressed 8 MiB and then trimmed".
    assert _extract_zip(bomb) == ""
    ing = ingest_one(bomb, "bomb.zip", "application/zip")
    assert ing.status == "empty"


def test_the_archive_has_a_total_uncompressed_budget():
    """Many ORDINARY-ratio entries must not add up past the budget either, so
    this fixture is deliberately incompressible (hex of random bytes)."""
    entries = [(f"f{i}.txt", os.urandom(90_000).hex().encode()) for i in range(30)]
    raw = _extract_zip(_zip_with(entries))
    assert len(raw) <= MAX_ZIP_UNCOMPRESSED_BYTES + 4096  # + the entry headers


def test_a_nested_archive_is_never_re_entered():
    """`sniff_mime` classifies by BYTES, so a zip NAMED `a.pdf` sniffs back to
    `application/zip` and used to re-enter `_extract_zip` — 50 entries per
    level, i.e. 50**depth full decompressions from one ~1 MB upload."""
    inner = _zip_with([("deep.txt", b"DEEPMARKER")])
    outer = _zip_with([("a.pdf", inner)])
    assert sniff_mime(inner, "a.pdf") == "application/zip"
    ing = ingest_one(outer, "outer.zip", "application/zip")
    assert "DEEPMARKER" not in (ing.text or "")


def test_a_plain_archive_of_text_still_reads():
    """The hardening must not cost the feature."""
    ing = ingest_one(_zip_with([("notes.txt", b"ZIPTOKEN-OK")]), "d.zip", "application/zip")
    assert "ZIPTOKEN-OK" in (ing.text or "")


# ── pdf: the render scale is clamped by the page, not by hope ────────────

def test_the_render_scale_is_clamped_by_the_page_size():
    letter_w, letter_h = 612.0, 792.0
    assert raster_scale(letter_w, letter_h) == 2.0
    # The maximum legal MediaBox. 2.0 here is a 28800x28800 bitmap (3.3 GB).
    huge = raster_scale(14400.0, 14400.0)
    assert huge < 2.0
    assert huge * 14400.0 <= 2.0 * MODEL_IMAGE_LONG_EDGE + 1
    # A page we could not measure still renders at the normal scale.
    assert raster_scale(0.0, 0.0) == 2.0


@pytest.mark.skipif(not fx.have("pypdfium2"), reason="pypdfium2 not installed here")
def test_a_scanned_pdf_still_rasterizes_under_the_clamp():
    from app.agent.attachment_ingest import rasterize_pdf

    pages = rasterize_pdf(fx.scanned_pdf("CLAMPED"))
    assert pages, "the clamp must not cost the rasterizer"
    from PIL import Image

    with Image.open(io.BytesIO(pages[0])) as im:
        assert max(im.size) <= MODEL_IMAGE_LONG_EDGE


# ── image: the declared size is untrusted input ──────────────────────────

def _png_claiming(width: int, height: int) -> bytes:
    """A real 1x1 PNG whose IHDR claims `width x height`. Pillow reads the
    header on `open()` and only allocates on `load()` — which is the whole
    point: the size test has to happen between the two."""
    base = fx.png_image(1, 1)
    ihdr_at = base.index(b"IHDR") + 4
    body = struct.pack(">II", width, height) + base[ihdr_at + 8:ihdr_at + 13]
    crc = zlib.crc32(b"IHDR" + body) & 0xFFFFFFFF
    return base[:ihdr_at] + body + struct.pack(">I", crc) + base[ihdr_at + 17:]


@pytest.fixture
def load_spy(monkeypatch):
    """Counts `Image.load()` calls — the allocation itself.

    The property under test is that the pixels are NEVER allocated for an
    oversized declaration, and asserting on the return value alone cannot show
    that: Pillow's own bomb error (only raised above 2x MAX_IMAGE_PIXELS, and
    only AFTER it tries) is swallowed by the same `except` and produces the same
    answer. Between 1x and 2x it merely warns and decodes.
    """
    from PIL import Image as PILImage

    calls = {"load": 0}
    real_open = PILImage.open

    class _Spy:
        def __init__(self, im):
            self._im = im

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return self._im.__exit__(*a)

        def load(self):
            calls["load"] += 1
            return self._im.load()

        def __getattr__(self, k):
            return getattr(self._im, k)

    monkeypatch.setattr(PILImage, "open", lambda fh, *a, **k: _Spy(real_open(fh, *a, **k)))
    return calls


def test_a_decompression_bomb_image_is_never_decoded(load_spy):
    """Pillow only WARNS between MAX_IMAGE_PIXELS and 2x it, so a small PNG
    declaring 9000x9000 (~324 MB of pixels) decodes with no error at all."""
    side = int((MAX_DECODED_IMAGE_PIXELS * 1.5) ** 0.5)
    data = _png_claiming(side, side)
    out, mime = downscale_image(data, "image/png")
    assert load_spy["load"] == 0, "the pixels were allocated for a declared size we refuse"
    # Returned untouched: no derivative was built, because none was decoded.
    assert out == data and mime == "image/png"


def test_an_image_inside_the_bound_is_still_decoded(load_spy):
    downscale_image(fx.jpeg_image(2000, 1500), "image/jpeg")
    assert load_spy["load"] == 1


def test_an_ordinary_photo_is_still_downscaled():
    big = fx.jpeg_image(2000, 1500)
    out, mime = downscale_image(big, "image/jpeg")
    assert out != big and mime == "image/webp"
    from PIL import Image

    with Image.open(io.BytesIO(out)) as im:
        assert max(im.size) <= MODEL_IMAGE_LONG_EDGE


# ── sniff: the branch that RUNS IN PRODUCTION ────────────────────────────

class _StubMagic:
    def __init__(self, verdict):
        self._verdict = verdict

    def from_buffer(self, data, mime=False):
        return self._verdict


@pytest.fixture
def with_magic(monkeypatch):
    """Inject a `magic` module. Production installs python-magic + libmagic1
    (requirements.agent.txt / Dockerfile.agent); CI installs neither, so every
    existing sniff test certifies the FALLBACK and none of them ever ran this."""
    def _install(verdict):
        monkeypatch.setitem(sys.modules, "magic", _StubMagic(verdict))
    return _install


OOXML_DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


@pytest.mark.parametrize(
    "verdict",
    ["application/octet-stream", "text/plain", "application/zip", OOXML_DOCX],
)
def test_a_docx_resolves_to_docx_whatever_libmagic_answers(with_magic, verdict):
    """libmagic's verdict for a 4 KiB OOXML prefix is BUILD-DEPENDENT: `file`
    answers `application/octet-stream` for a .docx here and the correct OOXML
    type for an .xlsx. `application/zip` must therefore be treated as
    indecisive — otherwise a Word document is stamped `kind=archive` and routed
    into the zip branch."""
    with_magic(verdict)
    doc = fx.docx_doc("MAGICTOKEN")
    assert sniff_mime(doc, "a.docx", OOXML_DOCX) == OOXML_DOCX


def test_libmagic_still_overrules_a_lying_extension(with_magic):
    with_magic("image/png")
    assert sniff_mime(fx.png_image(), "photo.heic", "image/heic") == "image/png"


def test_a_libmagic_crash_falls_through_to_the_signatures(monkeypatch):
    class _Boom:
        def from_buffer(self, data, mime=False):
            raise RuntimeError("libmagic is unhappy")

    monkeypatch.setitem(sys.modules, "magic", _Boom())
    assert sniff_mime(fx.png_image(), "photo.heic", "image/heic") == "image/png"


def test_the_inline_derivative_writer_has_the_same_pixel_gate(load_spy):
    """`doc_generators._render_thumbnail` opens the same untrusted bytes for
    every stored image; one guarded decoder and one unguarded one is no gate."""
    from app.agent.doc_generators import _render_thumbnail

    side = int((MAX_DECODED_IMAGE_PIXELS * 1.5) ** 0.5)
    assert _render_thumbnail(_png_claiming(side, side), "image/png") is None
    assert load_spy["load"] == 0, "the pixels were allocated for a declared size we refuse"
    # …and an ordinary large photo still gets its derivative.
    out = _render_thumbnail(fx.jpeg_image(2000, 1500), "image/jpeg")
    assert out and len(out) < len(fx.jpeg_image(2000, 1500))
    assert load_spy["load"] == 1


# ── ooxml: the fourth bomb class, and the one left open ──────────────────

def _ooxml_bomb(part: str) -> bytes:
    """A container whose MARKUP part deflates ~1000:1.

    `zipfile.is_zipfile` passes, the OOXML part names are real, and every
    library below (`python-docx`, `python-pptx`, `openpyxl`) materialises the
    part in full before it can be capped. Measured on this checkout with the
    pre-fix code: a 200 KB .docx drove `ingest_one` to 433 MB peak RSS.
    """
    return _zip_with([
        ("[Content_Types].xml", b"<Types/>"),
        ("_rels/.rels", b"<Relationships/>"),
        (part, b" " * (64 * 1024 * 1024)),
    ])


@pytest.mark.parametrize("part,name,mime", [
    ("word/document.xml", "bomb.docx",
     "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
    ("ppt/presentation.xml", "bomb.pptx",
     "application/vnd.openxmlformats-officedocument.presentationml.presentation"),
    ("xl/workbook.xml", "bomb.xlsx",
     "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
])
def test_an_ooxml_bomb_is_refused_before_the_parser_sees_it(part, name, mime):
    from app.agent.attachment_ingest import _ooxml_budget_ok

    bomb = _ooxml_bomb(part)
    assert len(bomb) < 1024 * 1024, "fixture is not actually compressed"
    assert _ooxml_budget_ok(bomb) is False
    ing = ingest_one(bomb, name, mime)
    # Refused, and named as such: the user is told the file could not be read
    # rather than the container being OOM-killed mid-turn.
    assert ing.status == "unreadable"


def test_the_ooxml_gate_runs_before_python_docx(monkeypatch):
    """The property is the ORDER: a gate that runs after the parse is the
    433 MB it was written to prevent."""
    import docx as _docx

    seen = {"n": 0}

    def _spy(*a, **kw):
        seen["n"] += 1
        raise AssertionError("python-docx must never see a refused container")

    monkeypatch.setattr(_docx, "Document", _spy)
    ingest_one(_ooxml_bomb("word/document.xml"), "bomb.docx",
               "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    assert seen["n"] == 0


@pytest.mark.skipif(not fx.have("docx"), reason="python-docx not installed")
def test_an_ordinary_docx_still_reads():
    """The hardening must not cost the feature."""
    ing = ingest_one(fx.docx_doc("OOXML-OK"), "ok.docx",
                     "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    assert ing.status == "ok"
    assert "OOXML-OK" in (ing.text or "")
