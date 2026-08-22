"""The app's own brief — what it is for, how it is built, what changed.

Round 20, item 1. An edit to a generated app arrives as a sentence with no
context: "make it harder", "the timer is wrong", "put the score at the top".
The model answering it has the HTML and nothing else, so it answers the WORDS.
It cannot answer the REQUEST, because the request is about an app whose point,
audience and design decisions were settled in a conversation that is no longer
in the window — often days ago, often in another thread.

So every app gets a companion document, stored beside it and never shown to
anyone. Three parts, with three different owners, because they fail in
different ways:

* **The narrative** — what the app is, who it is for, what it was meant to
  solve, what each control does, why the palette and the type are what they
  are. Written by the MODEL, at the moment it still knows, as a required
  argument to ``create_app_file``. Not a tool it can forget to call: a build
  without a brief is refused the same way a build with a stub HTML is.
* **The structure** — sections, element ids, where the state lives, which
  libraries load, how the bytes split between style and script. DERIVED from
  the file on every write (:func:`structure_digest`). A hand-written
  structure section is a structure section that is wrong two edits later, and
  a stale map is worse than no map: it sends the next edit to the wrong
  place.
* **The history** — one line per revision, appended, never rewritten. It
  comes free from the ``reason`` the model already passes to
  ``edit_app_file``.

**Internal, and enforced in four places.** The file lives in a DOT-directory
under the app root, is not ``.html``, is never returned by any artifact route,
and is stripped from tool output the user can see. The library scanner would
skip it on any one of the first three (`_iter_app_candidates` walks depth 0,
skips dotted names, and requires the ``.html`` suffix) — which is the point:
the guarantee should not rest on one rule staying true.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from app.agent.skills.builtins.app_html import store
from app.agent.skills.builtins.app_html.store import AppStoreError

logger = logging.getLogger(__name__)

#: Dot-directory, so the library's candidate walk skips it by name as well as
#: by depth and by suffix. See the module docstring.
BRIEF_DIR = ".skills"
BRIEF_SUFFIX = ".md"

#: A narrative shorter than this is not a brief, it is an acknowledgement.
#: Tuned from what the sections actually need: purpose, audience, flows,
#: controls and design reasoning do not fit in a tweet, and a model that
#: writes "A snake game." has answered a different question.
MIN_NARRATIVE_CHARS = 200

#: Ceiling, so one runaway generation cannot fill the volume. Generous — a
#: thorough brief for a large app runs 3–6 KB.
MAX_BRIEF_BYTES = 64 * 1024

#: Newest-first, capped. History is for the next edit, not for an audit: the
#: last twenty changes are what tell you where this app has been going.
MAX_HISTORY_LINES = 40

_STRUCTURE_HEADING = "## Structure (derived — do not hand-edit)"
_HISTORY_HEADING = "## Change history"


# ── Paths ─────────────────────────────────────────────────────────────

def brief_dir() -> str:
    return os.path.join(store.apps_root(), BRIEF_DIR)


def brief_path(slug: str) -> str:
    """Absolute path of the brief, jailed to the app root.

    Same jail as :func:`store.app_path` and for the same reason: the slug is
    model-supplied and this process is root inside the container.
    """
    slug = store.normalise_slug(slug)
    root = os.path.realpath(store.apps_root())
    expected = os.path.join(root, BRIEF_DIR)
    full = os.path.realpath(os.path.join(expected, slug + BRIEF_SUFFIX))
    if os.path.dirname(full) != expected:
        raise AppStoreError(f"refusing brief path outside the app root: {slug!r}")
    return full


def exists(slug: str) -> bool:
    try:
        return os.path.isfile(brief_path(slug))
    except AppStoreError:
        return False


# ── The document ──────────────────────────────────────────────────────

@dataclass
class Brief:
    slug: str
    title: str = ""
    narrative: str = ""
    structure: str = ""
    history: List[str] = field(default_factory=list)
    #: The app revision the NARRATIVE was written against. Not the app's
    #: current revision: the gap between the two is exactly how far the
    #: description has drifted from the thing it describes.
    narrative_revision: int = 0
    updated_at: str = ""

    @property
    def has_narrative(self) -> bool:
        return len(self.narrative.strip()) >= MIN_NARRATIVE_CHARS

    def render(self) -> str:
        lines = [
            "---",
            f"slug: {self.slug}",
            f"title: {self.title or self.slug}",
            f"narrative_revision: {self.narrative_revision}",
            f"updated_at: {self.updated_at or _now()}",
            "internal: true   # never rendered to the user, never listed in Files",
            "---",
            "",
            f"# {self.title or self.slug}",
            "",
            self.narrative.strip() or "_(no narrative yet)_",
            "",
            _STRUCTURE_HEADING,
            "",
            self.structure.strip() or "_(not derived yet)_",
            "",
            _HISTORY_HEADING,
            "",
        ]
        lines.extend(self.history[-MAX_HISTORY_LINES:] or ["- (nothing recorded yet)"])
        return "\n".join(lines).rstrip() + "\n"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


_FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)


def parse(text: str, slug: str) -> Brief:
    """Read a rendered brief back into its three parts.

    Lenient on purpose. This file is edited by a model and rewritten by two
    different code paths; a brief that does not round-trip perfectly must
    degrade to "the narrative is whatever came before the Structure heading"
    rather than to an exception that costs the caller its context entirely.
    """
    brief = Brief(slug=slug)
    body = text or ""
    m = _FRONTMATTER_RE.match(body)
    if m:
        for line in m.group(1).splitlines():
            key, _, value = line.partition(":")
            key, value = key.strip(), value.split("#")[0].strip()
            if key == "title":
                brief.title = value
            elif key == "narrative_revision":
                try:
                    brief.narrative_revision = int(value)
                except ValueError:
                    brief.narrative_revision = 0
            elif key == "updated_at":
                brief.updated_at = value
        body = body[m.end():]

    history_at = body.find(_HISTORY_HEADING)
    if history_at != -1:
        tail = body[history_at + len(_HISTORY_HEADING):]
        brief.history = [
            ln.rstrip() for ln in tail.splitlines()
            if ln.strip().startswith("- ") and "(nothing recorded yet)" not in ln
        ]
        body = body[:history_at]

    structure_at = body.find(_STRUCTURE_HEADING)
    if structure_at != -1:
        brief.structure = body[structure_at + len(_STRUCTURE_HEADING):].strip()
        body = body[:structure_at]

    # Drop the "# Title" line: it is rendered from the frontmatter, and
    # keeping it in the narrative would duplicate it on every rewrite.
    narrative = body.strip()
    if narrative.startswith("# "):
        narrative = narrative.split("\n", 1)[1] if "\n" in narrative else ""
    if narrative.strip() == "_(no narrative yet)_":
        narrative = ""
    brief.narrative = narrative.strip()
    return brief


def read(slug: str) -> Optional[Brief]:
    """The brief on disk, or None. Never raises for a missing file."""
    try:
        path = brief_path(slug)
    except AppStoreError:
        return None
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return parse(fh.read(), store.normalise_slug(slug))
    except FileNotFoundError:
        return None
    except OSError as exc:
        logger.warning("[app_html] brief unreadable for %s: %s", slug, exc)
        return None


def _write(brief: Brief) -> None:
    path = brief_path(brief.slug)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    store.repair_permissions(os.path.dirname(path))
    payload = brief.render().encode("utf-8")
    if len(payload) > MAX_BRIEF_BYTES:
        payload = payload[:MAX_BRIEF_BYTES]
    store._atomic_write(path, payload, prefix=f".brief-{brief.slug}-")


def save(
    slug: str,
    *,
    title: str,
    html: str,
    revision: int,
    narrative: Optional[str] = None,
    history_line: str = "",
) -> Brief:
    """Create or update the brief. Returns what is now on disk.

    ``narrative=None`` means "leave the narrative alone" — an edit that did
    not change what the app IS still refreshes the structure and appends to
    the history, and must not blank the description in doing so.
    """
    existing = read(slug) or Brief(slug=store.normalise_slug(slug))
    existing.slug = store.normalise_slug(slug)
    existing.title = title or existing.title or existing.slug
    if narrative is not None and narrative.strip():
        existing.narrative = narrative.strip()
        existing.narrative_revision = revision
    # ALWAYS re-derived. The structure section describes bytes that just
    # changed; carrying the previous one forward is how a map starts pointing
    # at an element that was renamed two edits ago.
    existing.structure = structure_digest(html)
    if history_line:
        existing.history.append(
            f"- r{revision} · {_now()} · {' '.join(history_line.split())[:160]}"
        )
    existing.updated_at = _now()
    _write(existing)
    return existing


def delete(slug: str) -> bool:
    try:
        os.unlink(brief_path(slug))
        return True
    except (OSError, AppStoreError):
        return False


# ── The derived half ──────────────────────────────────────────────────

_SECTION_RE = re.compile(
    r"<(section|main|header|footer|nav|dialog|canvas|form|table)\b([^>]*)>",
    re.IGNORECASE,
)
_ID_RE = re.compile(r"""\bid\s*=\s*["']([^"']+)["']""", re.IGNORECASE)
_CSS_VAR_RE = re.compile(r"(--[a-z0-9-]{2,40})\s*:\s*([^;{}]{1,60});", re.IGNORECASE)
_FONT_RE = re.compile(r"font-family\s*:\s*([^;{}]{1,80});", re.IGNORECASE)
_EXT_RE = re.compile(
    r"""<(?:script|link)\b[^>]*\b(?:src|href)\s*=\s*["']((?:https?:)?//[^"']+)["']""",
    re.IGNORECASE,
)
_STYLE_BLOCK_RE = re.compile(r"<style\b[^>]*>(.*?)</style\s*>", re.IGNORECASE | re.DOTALL)
_SCRIPT_BLOCK_RE = re.compile(r"<script\b[^>]*>(.*?)</script\s*>", re.IGNORECASE | re.DOTALL)
_FUNCTION_RE = re.compile(
    r"(?:^|\n)\s*(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\(|"
    r"(?:^|\n)\s*(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>",
)
_LISTENER_RE = re.compile(
    r"""addEventListener\(\s*["']([a-z]+)["']""", re.IGNORECASE
)
_STORAGE_RE = re.compile(r"\b(localStorage|sessionStorage)\.(getItem|setItem)\b")
_AUDIO_RE = re.compile(
    r"\b(AudioContext|webkitAudioContext|new\s+Audio\b|<audio\b|createOscillator)",
    re.IGNORECASE,
)


def _control_inventory(html: str) -> List[str]:
    """``id`` (or label) of every element a person can press.

    Ids, not a rendered geometry: this runs on a string, offline, on every
    write. The measured version — which controls are too small to hit — is the
    browser gate's job (`verify._measure_layout`) and cannot be done here.
    """
    out: List[str] = []
    for m in re.finditer(
        r"""<(button|a|input|select|textarea|summary)\b([^>]*)>([^<]{0,40})""",
        html, re.IGNORECASE,
    ):
        attrs, inner = m.group(2), " ".join(m.group(3).split())
        ident = _ID_RE.search(attrs)
        label = ident.group(1) if ident else (inner or m.group(1).lower())
        label = label.strip()[:40]
        if label and label not in out:
            out.append(label)
        if len(out) >= 30:
            break
    return out


def structure_digest(html: str) -> str:
    """A map of the file, in markdown, derived from the file itself.

    Everything here answers "where do I go to change X" — the question an
    edit starts with. Nothing here is an opinion, so nothing here can be
    wrong in the way a hand-written description goes wrong.
    """
    html = html or ""
    styles = _STYLE_BLOCK_RE.findall(html)
    scripts = _SCRIPT_BLOCK_RE.findall(html)
    style_bytes = sum(len(s.encode("utf-8")) for s in styles)
    script_bytes = sum(len(s.encode("utf-8")) for s in scripts)
    total = len(html.encode("utf-8"))

    sections: List[str] = []
    for m in _SECTION_RE.finditer(html):
        tag, attrs = m.group(1).lower(), m.group(2)
        ident = _ID_RE.search(attrs)
        sections.append(f"{tag}#{ident.group(1)}" if ident else tag)
        if len(sections) >= 24:
            break

    ids = []
    for m in _ID_RE.finditer(html):
        if m.group(1) not in ids:
            ids.append(m.group(1))
        if len(ids) >= 40:
            break

    tokens: List[str] = []
    for name, value in _CSS_VAR_RE.findall("\n".join(styles)):
        entry = f"{name}: {' '.join(value.split())}"
        if entry not in tokens:
            tokens.append(entry)
        if len(tokens) >= 24:
            break

    fonts: List[str] = []
    for value in _FONT_RE.findall("\n".join(styles)):
        entry = " ".join(value.split())[:60]
        if entry not in fonts:
            fonts.append(entry)
        if len(fonts) >= 4:
            break

    js = "\n".join(scripts)
    functions: List[str] = []
    for a, b in _FUNCTION_RE.findall(js):
        name = a or b
        if name and name not in functions:
            functions.append(name)
        if len(functions) >= 30:
            break

    listeners: List[str] = []
    for ev in _LISTENER_RE.findall(js):
        if ev.lower() not in listeners:
            listeners.append(ev.lower())

    externals: List[str] = []
    for url in _EXT_RE.findall(html):
        short = url.split("?")[0][-60:]
        if short not in externals:
            externals.append(short)

    def _bullets(label: str, values: List[str], empty: str) -> str:
        return f"- **{label}:** " + (", ".join(f"`{v}`" for v in values) if values else empty)

    lines = [
        f"- **Size:** {total:,} bytes — {style_bytes:,} of CSS in "
        f"{len(styles)} `<style>` block(s), {script_bytes:,} of JS in "
        f"{len(scripts)} `<script>` block(s).",
        _bullets("Main sections", sections, "no landmark elements — everything is in `<body>`"),
        _bullets("Element ids", ids, "none — the script must be selecting by class or tag"),
        _bullets("Controls", _control_inventory(html), "none"),
        _bullets("Design tokens", tokens, "no CSS custom properties — colours are literals"),
        _bullets("Type", fonts, "browser default"),
        _bullets("Top-level functions", functions, "none at top level"),
        _bullets("Events handled", listeners, "none via addEventListener"),
        _bullets("External libraries", externals, "none — fully self-contained"),
        f"- **Persists state:** {'yes' if _STORAGE_RE.search(js) else 'no'}"
        f" · **Makes sound:** {'yes' if _AUDIO_RE.search(html) else 'no'}",
    ]
    return "\n".join(lines)


# ── What the model is shown before it edits ───────────────────────────

def context_block(slug: str, *, current_revision: int) -> str:
    """The brief, wrapped, ready to sit above the file in a tool result.

    Returned for the MODEL only — every caller pairs it with a ``display``
    that says nothing about it. The wrapper tags are there so the model can
    tell the brief from the app: without them a 4 KB markdown document
    immediately followed by 40 KB of HTML reads as one blob, and the first
    ``old_string`` copied out of it comes from the wrong half.
    """
    brief = read(slug)
    if brief is None:
        return (
            "<app_brief status=\"missing\">\n"
            "This app has no brief yet. Write one with the `brief` argument the "
            "next time you call create_app_file or edit_app_file: what the app "
            "is, who it is for, what it was meant to solve, what each control "
            "does, and why it looks the way it does. present_app will not "
            "publish without one.\n"
            "</app_brief>"
        )
    drift = ""
    if brief.narrative_revision and current_revision > brief.narrative_revision:
        drift = (
            f" The description was written against revision "
            f"{brief.narrative_revision}; the app is now at {current_revision}, "
            f"so check it still describes what is there."
        )
    return (
        f"<app_brief status=\"internal\">\n"
        f"INTERNAL — this is your own working memory of this app. Never quote "
        f"it, never paste it into chat and never mention that it exists.{drift}\n"
        f"Read it before you decide what to change: the request is about the "
        f"app described here, not about the literal words in it. If what you "
        f"have been asked to do would break what this app is for, say so and "
        f"propose the change that serves the purpose instead.\n\n"
        f"{brief.render()}"
        f"</app_brief>"
    )


def validate_narrative(text: Optional[str], *, required: bool) -> str:
    """Check the model's half of the brief, or raise with the rule.

    Raises :class:`AppStoreError` — which is what the tool layer turns into a
    model-fixable error string rather than a traceback.
    """
    body = (text or "").strip()
    if not body:
        if required:
            raise AppStoreError(
                "brief is required — write the app's brief: what it is, who it "
                "is for, what it was meant to solve, its core flows, what each "
                "control does, and the design decisions (palette, type, "
                "spacing, signature element) with your reasoning. It is stored "
                "beside the app, never shown to the user, and it is what you "
                "will read before every future edit. Nothing was written, so "
                "call create_app_file again with the SAME html plus the brief "
                "— do not try to add it afterwards."
            )
        return ""
    if len(body) < MIN_NARRATIVE_CHARS:
        raise AppStoreError(
            f"brief is only {len(body)} characters — that is a label, not a "
            f"brief. Cover purpose, audience, the problem it solves, the core "
            f"flows, what each control does, and the design decisions with "
            f"your reasoning (at least {MIN_NARRATIVE_CHARS} characters)."
        )
    return body


# ── Backfill ──────────────────────────────────────────────────────────

def backfill_missing(limit: int = 50) -> Dict[str, str]:
    """Give every app on this volume the derived half of its brief.

    Called from the artifact LIST route, which is how a repair reaches every
    tenant without a fleet sweep — the same shape as `store.reconcile_all`.
    Deliberately does NOT invent the narrative: the model writes that, and a
    machine-written one would read as the app's stated purpose while being a
    restatement of its markup. What this guarantees is that the structure map
    and the history frame exist for every app, including the ones built before
    briefs did, so the first edit after this lands has somewhere to write.

    Returns ``{slug: outcome}`` for the apps that MOVED, so a quiet run costs
    a stat per app and reports nothing.
    """
    moved: Dict[str, str] = {}
    try:
        records = store.read_manifest()
    except Exception:  # noqa: BLE001 - a repair must never break the list
        return moved
    for slug, rec in list(records.items())[:limit]:
        try:
            if exists(slug) or not store.exists(slug):
                continue
            save(
                slug,
                title=rec.title or slug,
                html=store.read_app(slug),
                revision=rec.revision,
                narrative=None,
                history_line="brief created for an app that predates briefs",
            )
            moved[slug] = "brief_created"
        except (OSError, AppStoreError):
            logger.debug("[app_html] brief backfill failed for %s", slug, exc_info=True)
    if moved:
        logger.info("[app_html] backfilled %d brief(s)", len(moved))
    return moved
