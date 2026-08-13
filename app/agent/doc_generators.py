"""
Document generators — produce PDF, DOCX, XLSX, PPTX, Markdown, and HTML→PDF
files from a structured content schema, and register them as attachments
via the file_storage backend.

Each generator returns an Attachment dataclass. The tool executor wraps
these into the per-call pending_attachments list; agent_runner drains that
list when persisting the assistant message so the frontend receives an
`attachment` WS event and the DocumentSplit pane can render the file.

Heavy deps (reportlab, openpyxl, python-docx, python-pptx, weasyprint)
are imported lazily at call-site so the platform image (which doesn't
need weasyprint) can skip installing Pango/Cairo.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.services.file_storage import get_storage_backend


logger = logging.getLogger(__name__)

# Strong refs to in-flight preview warms — see _prewarm_preview.
_PREWARM_TASKS: set[asyncio.Task] = set()

MIME_PDF = "application/pdf"
MIME_DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
MIME_XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
MIME_PPTX = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
MIME_MD = "text/markdown"
MIME_HTML = "text/html"


@dataclass
class Attachment:
    id: str
    filename: str
    mime_type: str
    size_bytes: int
    storage_path: str
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _safe_filename(name: str, default_ext: str) -> str:
    """Strip directory traversal and ensure the requested extension."""
    # Normalize Windows-style separators BEFORE basename — on POSIX,
    # os.path.basename("..\\..\\etc\\x") returns the whole string, which
    # would otherwise survive into the storage key as a literal filename.
    name = (name or "").replace("\\", "/")
    name = os.path.basename(name).strip() or f"document{default_ext}"
    if not name.lower().endswith(default_ext):
        name = f"{os.path.splitext(name)[0] or 'document'}{default_ext}"
    return name


def _prewarm_preview(storage_key: str, mime_type: str) -> None:
    """Start the DOCX/PPTX -> PDF conversion now, not on first view.

    The preview pane converts through LibreOffice, and that conversion was
    measured at 10.3 s cold (3.3 s warm, 0.2 s cached) on a real user file
    on 2026-08-13. Every generated document is cold exactly once — on the
    view that immediately follows generation, which is the view the user
    always takes. So the one time it matters is the one time it is slow.

    Doing it here costs nothing the user waits for: the agent has just
    written the bytes and is still composing its reply, so the conversion
    overlaps with the rest of the turn and the pane opens from cache.

    Fire-and-forget by design. A failure here must never fail the tool
    call that produced the document — the preview route still converts on
    demand, and the user gets a file either way.
    """
    if mime_type not in (MIME_DOCX, MIME_PPTX):
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return

    async def _run() -> None:
        try:
            from app.services.doc_preview import render_to_pdf
            await render_to_pdf(storage_key)
        except Exception as e:  # never surface — this is pure latency work
            logger.debug("preview prewarm skipped for %s: %s", storage_key, e)

    task = loop.create_task(_run())
    # Hold a reference; a bare create_task can be garbage-collected
    # mid-flight, which would silently drop the warm we just scheduled.
    _PREWARM_TASKS.add(task)
    task.add_done_callback(_PREWARM_TASKS.discard)


async def _persist(data: bytes, filename: str, mime_type: str, user_scope: str) -> Attachment:
    """Write bytes to storage under {user_scope}/{uuid}_{filename} and return an Attachment."""
    att_id = uuid.uuid4().hex
    key = f"{user_scope}/{att_id}_{filename}" if user_scope else f"{att_id}_{filename}"
    backend = get_storage_backend()
    await backend.put(key, data)
    _prewarm_preview(key, mime_type)
    return Attachment(
        id=att_id,
        filename=filename,
        mime_type=mime_type,
        size_bytes=len(data),
        storage_path=key,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


# ── Structured content iterators shared by PDF/DOCX/PPTX ──────────

def _iter_blocks(content: Any) -> List[Dict[str, Any]]:
    """Normalize content to a list of block dicts.

    Accepts either a list of blocks, or a string (treated as a single
    paragraph). Blocks have a `type` key: heading|paragraph|table|image|page_break.
    """
    if isinstance(content, str):
        return [{"type": "paragraph", "text": content}]
    if isinstance(content, list):
        return content
    return []


# ── PDF via reportlab ─────────────────────────────────────────────

async def gen_pdf(
    content: Any,
    filename: str,
    *,
    user_scope: str,
    title: Optional[str] = None,
    cover_page: bool = False,
) -> Attachment:
    from reportlab.lib.pagesizes import LETTER  # type: ignore
    from reportlab.lib.styles import getSampleStyleSheet  # type: ignore
    from reportlab.platypus import (  # type: ignore
        SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image,
    )
    from reportlab.lib import colors  # type: ignore
    from reportlab.lib.units import inch  # type: ignore

    filename = _safe_filename(filename, ".pdf")
    buf = io.BytesIO()

    def _on_page(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 9)
        canvas.setFillGray(0.4)
        canvas.drawRightString(LETTER[0] - 0.5 * inch, 0.4 * inch, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(buf, pagesize=LETTER, title=title or filename)
    styles = getSampleStyleSheet()
    story: list = []

    if cover_page and title:
        story.extend([
            Spacer(1, 3 * inch),
            Paragraph(f"<para align='center'><font size='24'><b>{title}</b></font></para>", styles["Title"]),
            Spacer(1, 0.3 * inch),
            Paragraph(
                f"<para align='center'>{datetime.now().strftime('%B %d, %Y')}</para>",
                styles["Normal"],
            ),
            PageBreak(),
        ])

    for block in _iter_blocks(content):
        btype = block.get("type", "paragraph")
        if btype == "heading":
            level = int(block.get("level", 1))
            style_name = {1: "Heading1", 2: "Heading2", 3: "Heading3", 4: "Heading4"}.get(level, "Heading2")
            story.append(Paragraph(block.get("text", ""), styles[style_name]))
        elif btype == "paragraph":
            story.append(Paragraph(block.get("text", ""), styles["BodyText"]))
            story.append(Spacer(1, 0.1 * inch))
        elif btype == "table":
            rows = [block.get("headers", [])] + list(block.get("rows", []))
            if rows and rows[0]:
                t = Table(rows, repeatRows=1)
                t.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]))
                story.append(t)
                story.append(Spacer(1, 0.15 * inch))
        elif btype == "image":
            path = block.get("path", "")
            if path and os.path.isfile(path):
                try:
                    story.append(Image(path, width=5 * inch, height=3 * inch, kind="proportional"))
                    if block.get("caption"):
                        story.append(Paragraph(
                            f"<para align='center'><i>{block['caption']}</i></para>",
                            styles["Italic"],
                        ))
                    story.append(Spacer(1, 0.15 * inch))
                except Exception:
                    pass
        elif btype == "page_break":
            story.append(PageBreak())

    doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)
    return await _persist(buf.getvalue(), filename, MIME_PDF, user_scope)


# ── DOCX via python-docx ──────────────────────────────────────────

async def gen_docx(
    content: Any,
    filename: str,
    *,
    user_scope: str,
    title: Optional[str] = None,
) -> Attachment:
    from docx import Document  # type: ignore
    from docx.shared import Inches  # type: ignore

    filename = _safe_filename(filename, ".docx")
    doc = Document()

    if title:
        doc.add_heading(title, level=0)

    for block in _iter_blocks(content):
        btype = block.get("type", "paragraph")
        if btype == "heading":
            level = max(1, min(4, int(block.get("level", 1))))
            doc.add_heading(block.get("text", ""), level=level)
        elif btype == "paragraph":
            doc.add_paragraph(block.get("text", ""))
        elif btype == "bullet_list":
            for item in block.get("items", []):
                doc.add_paragraph(str(item), style="List Bullet")
        elif btype == "numbered_list":
            for item in block.get("items", []):
                doc.add_paragraph(str(item), style="List Number")
        elif btype == "table":
            headers = block.get("headers", [])
            rows = block.get("rows", [])
            if headers:
                tbl = doc.add_table(rows=1 + len(rows), cols=len(headers))
                tbl.style = "Light Grid Accent 1"
                for i, h in enumerate(headers):
                    tbl.rows[0].cells[i].text = str(h)
                for r, row in enumerate(rows, start=1):
                    for c, cell in enumerate(row[: len(headers)]):
                        tbl.rows[r].cells[c].text = "" if cell is None else str(cell)
        elif btype == "image":
            path = block.get("path", "")
            if path and os.path.isfile(path):
                try:
                    doc.add_picture(path, width=Inches(5))
                except Exception:
                    pass
        elif btype == "page_break":
            doc.add_page_break()

    buf = io.BytesIO()
    doc.save(buf)
    return await _persist(buf.getvalue(), filename, MIME_DOCX, user_scope)


# ── XLSX via openpyxl ─────────────────────────────────────────────

async def gen_xlsx(
    sheets: List[Dict[str, Any]],
    filename: str,
    *,
    user_scope: str,
) -> Attachment:
    from openpyxl import Workbook  # type: ignore
    from openpyxl.styles import Font, PatternFill  # type: ignore
    from openpyxl.utils import get_column_letter  # type: ignore

    filename = _safe_filename(filename, ".xlsx")
    wb = Workbook()
    wb.remove(wb.active)  # Drop the default sheet

    if not sheets:
        sheets = [{"name": "Sheet1", "headers": [], "rows": []}]

    for sheet_def in sheets:
        name = str(sheet_def.get("name", "Sheet"))[:31] or "Sheet"
        ws = wb.create_sheet(title=name)
        headers = sheet_def.get("headers", [])
        rows = sheet_def.get("rows", [])

        if headers:
            ws.append(headers)
            bold = Font(bold=True)
            fill = PatternFill("solid", fgColor="F0F0F0")
            for col_idx in range(1, len(headers) + 1):
                c = ws.cell(row=1, column=col_idx)
                c.font = bold
                c.fill = fill

        for row in rows:
            ws.append(list(row))

        # Auto-size columns (rough estimate — openpyxl doesn't support true auto-fit).
        for col_idx in range(1, max(1, len(headers)) + 1):
            letter = get_column_letter(col_idx)
            max_len = len(str(headers[col_idx - 1])) if col_idx - 1 < len(headers) else 8
            for row in rows:
                if col_idx - 1 < len(row):
                    v = row[col_idx - 1]
                    if v is not None:
                        max_len = max(max_len, len(str(v)))
            ws.column_dimensions[letter].width = min(60, max_len + 2)

    buf = io.BytesIO()
    wb.save(buf)
    return await _persist(buf.getvalue(), filename, MIME_XLSX, user_scope)


# ── PPTX via python-pptx ──────────────────────────────────────────

async def gen_pptx(
    slides: List[Dict[str, Any]],
    filename: str,
    *,
    user_scope: str,
) -> Attachment:
    from pptx import Presentation  # type: ignore
    from pptx.util import Inches, Pt  # type: ignore

    filename = _safe_filename(filename, ".pptx")
    prs = Presentation()

    for slide_def in slides or []:
        stype = slide_def.get("type", "content")
        if stype == "title":
            layout = prs.slide_layouts[0]
            slide = prs.slides.add_slide(layout)
            slide.shapes.title.text = slide_def.get("title", "")
            if len(slide.placeholders) > 1:
                slide.placeholders[1].text = slide_def.get("subtitle", "")
        elif stype == "section":
            layout = prs.slide_layouts[2] if len(prs.slide_layouts) > 2 else prs.slide_layouts[0]
            slide = prs.slides.add_slide(layout)
            slide.shapes.title.text = slide_def.get("title", "")
        elif stype == "image":
            layout = prs.slide_layouts[5] if len(prs.slide_layouts) > 5 else prs.slide_layouts[1]
            slide = prs.slides.add_slide(layout)
            slide.shapes.title.text = slide_def.get("title", "")
            path = slide_def.get("path", "")
            if path and os.path.isfile(path):
                try:
                    slide.shapes.add_picture(path, Inches(1), Inches(1.5), width=Inches(8))
                except Exception:
                    pass
        else:  # "content" default
            layout = prs.slide_layouts[1]
            slide = prs.slides.add_slide(layout)
            slide.shapes.title.text = slide_def.get("title", "")
            body = slide.placeholders[1].text_frame if len(slide.placeholders) > 1 else None
            bullets = slide_def.get("bullets", [])
            if body and bullets:
                body.text = str(bullets[0])
                for b in bullets[1:]:
                    p = body.add_paragraph()
                    p.text = str(b)
                    p.level = 0

    buf = io.BytesIO()
    prs.save(buf)
    return await _persist(buf.getvalue(), filename, MIME_PPTX, user_scope)


# ── Markdown (plain write) ────────────────────────────────────────

async def gen_markdown(content: str, filename: str, *, user_scope: str) -> Attachment:
    filename = _safe_filename(filename, ".md")
    data = (content or "").encode("utf-8")
    return await _persist(data, filename, MIME_MD, user_scope)


# ── HTML → PDF via weasyprint (agent-side only) ───────────────────

async def gen_html_to_pdf(html: str, filename: str, *, user_scope: str) -> Attachment:
    try:
        from weasyprint import HTML  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "weasyprint is not installed in this runtime. "
            "HTML→PDF is agent-side only (requires Pango/Cairo). "
            "Use generate_pdf with structured content instead."
        ) from e

    filename = _safe_filename(filename, ".pdf")
    buf = io.BytesIO()
    # SSRF/LFI guard (audit-2026 re-audit round 9): WeasyPrint's DEFAULT
    # url_fetcher resolves file:// and http(s) URLs embedded in the (possibly
    # injection-controlled) HTML. `<img src="file:///proc/self/environ">` would
    # embed the agent's env secrets into the PDF; `<img
    # src="http://169.254.169.254/…">` or an internal host would SSRF. Restrict
    # to inline data: URIs + public https only.
    HTML(string=html or "", url_fetcher=_safe_pdf_url_fetcher).write_pdf(buf)
    return await _persist(buf.getvalue(), filename, MIME_PDF, user_scope)


def _safe_pdf_url_fetcher(url: str):
    """Allow only inline data: URIs and public https for WeasyPrint PDF
    rendering (audit-2026 re-audit round 9). Blocks file:// (LFI) and any
    private/loopback/link-local/metadata host (SSRF)."""
    from weasyprint.urls import default_url_fetcher  # type: ignore
    u = (url or "").strip()
    low = u.lower()
    if low.startswith("data:"):
        return default_url_fetcher(u)
    if low.startswith("https://"):
        from app.agent.smart_fetch.reader import _assert_public_url
        _assert_public_url(u)  # raises ValueError on private/internal/metadata
        return default_url_fetcher(u)
    raise ValueError(f"blocked non-public URL in PDF generation: {u[:80]!r}")
