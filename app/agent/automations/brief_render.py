"""ONE composition of a run's brief, in each of the catalogue's formats.

CONTRACT-R43 §9, last rule: "The Slack post text and the card are
rendered from ONE composition (`brief_render(groups, format)`), so they
cannot disagree."

They did disagree, and by construction. The card is the narrator's
`result` turn — five ranked tiers, a consequence sentence per row, a tag.
The Slack post is whatever the automation's write STEP interpolates,
which for every shipped template is one `{{steps.<id>.text}}` — a single
read's rendered collect lines, before any ranking existed. So the phone
said "DO FIRST · BLOCKS OTHERS" about three things and the channel
carried a bulleted dump of a different set, in a different order, with
none of the whys. Neither was wrong; they were answers to different
questions, and the user had to reconcile them.

This module is the answer to one question. Every delivery channel and
the card render the SAME groups through here; only the shape changes.

Everything is PURE: no session, no clock, no network. `groups` is the
result turn's own list — `{rank, label, tone, rows[], items[],
empty_reason}` — and a group with no `items` (a turn minted before R43,
or a digest that never resolved its refs) falls back to its rows, so an
old thread renders rather than blanking.
"""

from __future__ import annotations

import csv
import io
import logging
from dataclasses import dataclass
from typing import Any, Optional

from . import catalog

logger = logging.getLogger(__name__)

#: §1.3 "Five short lines". A hard five: the format's whole promise is
#: that it is read without opening anything, and a sixth line breaks it.
MAX_LINES = 5

#: What a one-page PDF can hold at 11pt on LETTER with the tier
#: headings. Past this the "one-page" in the format's name is false, so
#: the tail is counted rather than printed.
PDF_MAX_ROWS = 26


class PdfUnavailable(RuntimeError):
    """reportlab could not be loaded, so no PDF exists.

    Raised rather than returned as markdown-under-another-name: a
    delivery that answers "One-page PDF" with a text blob is the same
    class of lie as a chip that narrows nothing.
    """


@dataclass(frozen=True)
class Brief:
    """One rendered brief.

    `text` is what a channel that can only carry TEXT posts, and it is
    the format's own shape in every case — a CSV's text is the CSV, a
    markdown file's text is the markdown. `document` is the same content
    as a file, for a channel that can carry one: `(filename, mime,
    bytes)`, or None for the two chat formats.

    The two are never different content. `document` exists because a
    file has a name and a type, not because a file says something else.
    """

    format: str
    title: str
    text: str
    document: Optional[tuple[str, str, bytes]] = None

    @property
    def filename(self) -> Optional[str]:
        return self.document[0] if self.document else None

    @property
    def mime(self) -> str:
        return self.document[1] if self.document else "text/plain"


# ── the material ─────────────────────────────────────────────────────


def _rows_of(group: dict) -> list[dict]:
    return [r for r in (group.get("rows") or []) if isinstance(r, dict)]


def _items_of(group: dict) -> list[dict]:
    return [i for i in (group.get("items") or []) if isinstance(i, dict)]


def _live_groups(groups: Any) -> list[dict]:
    """The groups that said something, in rank order.

    An EMPTY group is dropped from every rendered format — it is the
    card that owes the reader `empty_reason`, because the card has a
    heading to put it under. A Slack post with five headings and nothing
    beneath four of them is the "0 items" defect wearing a different
    surface.
    """
    out = []
    for g in groups or []:
        if isinstance(g, dict) and (_rows_of(g) or _items_of(g)):
            out.append(g)
    return out


def _tag(row: dict) -> str:
    return str(row.get("tag") or "").strip()


def _one_line(row: dict) -> str:
    """A row as one line: the statement, then its tag."""
    text = " ".join(str(row.get("text") or "").split())
    tag = _tag(row)
    return f"{text} · {tag}" if tag and tag not in text else text


def _item_line(it: dict) -> str:
    """An ITEM as one line: who, what, where — the three slots §4 shows
    on a card, in the order a person reads them."""
    bits = [b for b in (str(it.get("who") or "").strip(),
                        str(it.get("title") or "").strip()) if b]
    line = " · ".join(bits)
    where = str(it.get("where") or "").strip()
    return f"{line} ({where})" if where and line else (line or where)


# ── the five formats ─────────────────────────────────────────────────


def _render_lines(title: str, groups: list[dict]) -> str:
    """Five short lines, top tier first (§1.3 `lines`).

    Rows before items: a row is the ranking's own one-line statement of
    consequence, which is exactly what this format is for. Items fill in
    only when a group has none — a digest whose rows never resolved.
    """
    lines: list[str] = []
    for g in groups:
        for r in _rows_of(g):
            lines.append(_one_line(r))
            if len(lines) >= MAX_LINES:
                return "\n".join(lines)
        if not _rows_of(g):
            for it in _items_of(g):
                line = _item_line(it)
                if line:
                    lines.append(line)
                if len(lines) >= MAX_LINES:
                    return "\n".join(lines)
    return "\n".join(lines)


def _render_ranked(title: str, groups: list[dict]) -> str:
    """The tiers, in order, with their rows under them (§1.3 `ranked`).

    The count beside a label is `len(items)` and never `len(rows)` — §9,
    and the reason this round exists: a tier holding one row that stands
    for 128 ignored notifications reads "128", not "1".
    """
    out: list[str] = [title] if title else []
    for g in groups:
        n = len(_items_of(g)) or len(_rows_of(g))
        out.append("")
        out.append(f"{g.get('label') or ''} · {n}".strip(" ·"))
        for r in _rows_of(g):
            out.append(f"- {_one_line(r)}")
            sub = " ".join(str(r.get("sub") or "").split())
            if sub:
                out.append(f"  {sub}")
        if not _rows_of(g):
            for it in _items_of(g):
                line = _item_line(it)
                if line:
                    out.append(f"- {line}")
    return "\n".join(out).strip()


def _render_markdown(title: str, groups: list[dict]) -> str:
    out: list[str] = [f"# {title}"] if title else []
    for g in groups:
        n = len(_items_of(g)) or len(_rows_of(g))
        out.append("")
        out.append(f"## {g.get('label') or ''} ({n})")
        for r in _rows_of(g):
            tag = _tag(r)
            head = " ".join(str(r.get("text") or "").split())
            sub = " ".join(str(r.get("sub") or "").split())
            line = f"- **{head}**"
            if tag:
                line += f" — `{tag}`"
            out.append(line)
            if sub:
                out.append(f"  {sub}")
        for it in _items_of(g):
            why = " ".join(str(it.get("why") or "").split())
            line = _item_line(it)
            if not line:
                continue
            out.append(f"  - {line}" if _rows_of(g) else f"- {line}")
            if why:
                out.append(f"    _{why}_")
    return "\n".join(out).strip() + "\n"


#: §1.3 `csv` — "one row per thing it found". The header is the §9 item
#: shape plus the tier it landed in, so a spreadsheet can sort by what
#: breaks first without the reader knowing the vocabulary.
CSV_COLUMNS = ("tier", "rank", "who", "title", "sub", "why", "when",
               "source", "where", "live")


def _render_csv(title: str, groups: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(CSV_COLUMNS)
    for g in groups:
        label = str(g.get("label") or "")
        rank = g.get("rank") or ""
        items = _items_of(g)
        if items:
            for it in items:
                w.writerow([
                    label, rank,
                    it.get("who") or "", it.get("title") or "",
                    it.get("sub") or "", it.get("why") or "",
                    it.get("at") or "", it.get("source") or "",
                    it.get("where") or "",
                    "yes" if it.get("hot") else "",
                ])
            continue
        # A tier whose rows never resolved to items still owes the sheet
        # a row: an empty CSV under a brief that had content is a file
        # the user opens once and never again.
        for r in _rows_of(g):
            w.writerow([label, rank, "", str(r.get("text") or ""),
                        str(r.get("sub") or ""), "", "", "", "",
                        ""])
    return buf.getvalue()


def _render_pdf(title: str, groups: list[dict]) -> bytes:
    """One page, via reportlab — the same writer `doc_generators.gen_pdf`
    uses, imported lazily because this module is on the run path and a
    document library is not.

    `PdfUnavailable` rather than a fallback: see the class docstring.
    """
    try:
        from reportlab.lib.pagesizes import LETTER  # type: ignore
        from reportlab.lib.styles import getSampleStyleSheet  # type: ignore
        from reportlab.lib.units import inch  # type: ignore
        from reportlab.platypus import (  # type: ignore
            SimpleDocTemplate, Paragraph, Spacer,
        )
    except Exception as e:  # noqa: BLE001
        raise PdfUnavailable(str(e)) from e

    from xml.sax.saxutils import escape

    styles = getSampleStyleSheet()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=LETTER, title=title or "Brief",
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
        leftMargin=0.7 * inch, rightMargin=0.7 * inch,
    )
    story: list = [Paragraph(escape(title or "Your brief"), styles["Title"]),
                   Spacer(1, 0.15 * inch)]
    printed = 0
    dropped = 0
    for g in groups:
        rows = _rows_of(g)
        n = len(_items_of(g)) or len(rows)
        if printed >= PDF_MAX_ROWS:
            dropped += n
            continue
        story.append(Paragraph(
            escape(f"{g.get('label') or ''} · {n}".strip(" ·")),
            styles["Heading3"]))
        for r in rows:
            if printed >= PDF_MAX_ROWS:
                dropped += 1
                continue
            head = escape(" ".join(str(r.get("text") or "").split()))
            sub = escape(" ".join(str(r.get("sub") or "").split()))
            tag = escape(_tag(r))
            body = f"<b>{head}</b>"
            if tag:
                body += f" — {tag}"
            if sub:
                body += f"<br/>{sub}"
            story.append(Paragraph(body, styles["BodyText"]))
            printed += 1
        story.append(Spacer(1, 0.08 * inch))
    if dropped:
        story.append(Paragraph(
            escape(f"{dropped} more, in the app."), styles["BodyText"]))
    doc.build(story)
    return buf.getvalue()


# ── the entry point ──────────────────────────────────────────────────


def brief_render(
    groups: Any, format_id: str, *, title: str = "", slug: str = "brief",
) -> Brief:
    """Render one run's result groups in one catalogue format.

    An unknown format falls back to the catalogue default rather than
    raising: this is on the delivery path, and a brief that does not go
    out because a stored id drifted is worse than a brief in the shape
    the automation started with. The fallback is logged.
    """
    fmt = catalog.format_(format_id)
    if fmt is None:
        logger.warning("[automations] unknown brief format %r — rendering %r",
                       format_id, catalog.DEFAULT_DELIVERY["format"])
        fmt = catalog.format_(catalog.DEFAULT_DELIVERY["format"])
    fid = fmt["id"]
    live = _live_groups(groups)
    title = " ".join(str(title or "").split())
    stem = "".join(
        c if c.isalnum() or c in "-_" else "-" for c in (slug or "brief")
    ).strip("-").lower() or "brief"

    if fid == "lines":
        return Brief(fid, title, _render_lines(title, live))
    if fid == "ranked":
        return Brief(fid, title, _render_ranked(title, live))
    if fid == "markdown":
        text = _render_markdown(title, live)
        return Brief(fid, title, text,
                     (f"{stem}.md", "text/markdown", text.encode("utf-8")))
    if fid == "csv":
        text = _render_csv(title, live)
        return Brief(fid, title, text,
                     (f"{stem}.csv", "text/csv", text.encode("utf-8")))
    # pdf. The TEXT beside it is the ranked shape, not a second brief:
    # a channel that cannot carry a file still has to say the same
    # words, and `_render_ranked` is the one-page document read aloud.
    return Brief(fid, title, _render_ranked(title, live),
                 (f"{stem}.pdf", "application/pdf", _render_pdf(title, live)))
