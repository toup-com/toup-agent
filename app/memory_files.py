"""Memory files — the ONLY unit the memory product knows (v3).

Canon for `docs/memory/rebuild-2026-08-v3.md` §1. A user's memory is a small
set of curated files; each file's `body_md` — a markdown bullet list — IS the
memory. Rows in `memories` are retired from the product (the document/media
leg of `memory_search` is the one surviving reader, §3.4).

Leaf module on purpose: stdlib only, importable from models, services,
prompts and scripts alike (mirrors app.memory_taxonomy, which the LEGACY
half at the bottom of this file still reads).

Everything above the LEGACY banner is v3. Everything below it is round-8
canon kept alive for exactly one consumer — `memory_v3_migration.py`, which
reads a row's round-8 file assignment (or derives it, read-only, for a row
written after the last organize pass) to know what the writer should be
told about where a memory used to live. Nothing v3 may import from it.

The section map below MUST stay in lockstep with the clients' shared
``memoryModel.ts`` (mobile canon, vendored to web).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# ══ v3 ════════════════════════════════════════════════════════════════

class FileSection(str, Enum):
    """Where a file sits on the Memory page. Order = display order."""

    YOU = "you"
    PEOPLE = "people"
    TOPICS = "topics"
    AREAS = "areas"
    LEARNED = "learned"


SECTION_ORDER: List[FileSection] = [
    FileSection.YOU,
    FileSection.PEOPLE,
    FileSection.TOPICS,
    FileSection.AREAS,
    FileSection.LEARNED,
]

SECTION_LABEL: Dict[FileSection, str] = {
    FileSection.YOU: "You",
    FileSection.PEOPLE: "People",
    FileSection.TOPICS: "Topics",
    FileSection.AREAS: "Areas",
    FileSection.LEARNED: "Learned",
}


# ── System files ──────────────────────────────────────────────────────
# The three fixed files every user has. Everything else is created on
# demand by the curator and can be deleted outright; deleting a system
# file only empties its body.
#
# `description` follows the same regex every generated description must
# (`DESCRIPTION_RE`) — these are the worked examples the writer copies.

PROFILE_SLUG = "you/profile"
CURRENT_CONTEXT_SLUG = "you/current-context"
LEARNED_SLUG = "learned"

SYSTEM_FILES: Dict[str, Dict[str, str]] = {
    PROFILE_SLUG: {
        "section": FileSection.YOU.value,
        "title": "Profile",
        "description": (
            "Who this person is — identity, setup, health and standing "
            "commitments; read when a reply depends on personal context."
        ),
    },
    CURRENT_CONTEXT_SLUG: {
        "section": FileSection.YOU.value,
        "title": "Current context",
        "description": (
            "What is going on right now — today, this week, this year; "
            "read when a reply depends on what is currently happening."
        ),
    },
    LEARNED_SLUG: {
        "section": FileSection.LEARNED.value,
        "title": "Learned",
        "description": (
            "Corrections and working rules from this person — how they want "
            "things done; read when you are about to act on their behalf."
        ),
    },
}

#: Injected on every non-trivial reply, in this order (§3.1).
ALWAYS_INJECTED_SLUGS: Tuple[str, ...] = (
    PROFILE_SLUG, CURRENT_CONTEXT_SLUG, LEARNED_SLUG,
)


# ── Slugs ─────────────────────────────────────────────────────────────
# `name` or `namespace/name`, lowercase word characters and hyphens.
# Unicode letters are kept (a Persian name makes a Persian slug).
# Namespaces: you/, people/, topics/, areas/; plus the bare `learned`.

_SLUG_MAX = 120
_SLUG_RE = re.compile(r"^[\w-]+(?:/[\w-]+)?$", re.UNICODE)
_SLUG_STRIP = re.compile(r"[^\w-]+", re.UNICODE)

SLUG_NAMESPACES: Dict[str, FileSection] = {
    "you": FileSection.YOU,
    "people": FileSection.PEOPLE,
    "topics": FileSection.TOPICS,
    "areas": FileSection.AREAS,
}


def slugify(name: str) -> str:
    """A stable slug segment from a display name. '' if nothing survives."""
    normalized = unicodedata.normalize("NFKC", name or "").strip().lower()
    collapsed = _SLUG_STRIP.sub("-", normalized).replace("_", "-").strip("-")
    collapsed = re.sub(r"-{2,}", "-", collapsed)
    return collapsed[:_SLUG_MAX]


def is_valid_slug(slug: str) -> bool:
    return (
        bool(slug)
        and len(slug) <= _SLUG_MAX
        and "_" not in slug
        and bool(_SLUG_RE.match(slug))
    )


def person_slug(name: str) -> Optional[str]:
    seg = slugify(name)
    return f"people/{seg}" if seg else None


def topic_slug(name: str) -> Optional[str]:
    seg = slugify(name)
    return f"topics/{seg}" if seg else None


def area_slug(name: str) -> Optional[str]:
    seg = slugify(name)
    return f"areas/{seg}" if seg else None


def section_of_slug(slug: str) -> Optional[FileSection]:
    """The section a slug belongs to, from its shape. None when the shape
    names no v3 namespace — the caller rejects rather than guessing, which
    is what stops a round-8 row (`knowledge`, `working`) from appearing in
    a v3 section it was never written for."""
    if slug in SYSTEM_FILES:
        return FileSection(SYSTEM_FILES[slug]["section"])
    head, _, rest = slug.partition("/")
    if rest:
        return SLUG_NAMESPACES.get(head)
    return None


def title_from_slug(slug: str) -> str:
    """Display title for a file known only by slug (a dangling link)."""
    if slug in SYSTEM_FILES:
        return SYSTEM_FILES[slug]["title"]
    seg = slug.split("/", 1)[-1].replace("-", " ").strip()
    return seg[:1].upper() + seg[1:] if seg else slug


# ── Descriptions (§1.4) ───────────────────────────────────────────────
# `<what this is> — <scope>; read when <trigger>.` Generated by the
# writer, regenerated when the body changes materially. There is NO
# template fallback: a file is born with a real description or the create
# op is rejected.

DESCRIPTION_RE = re.compile(r"^.{3,80} — .{3,120}; read when .{3,120}\.$")
DESCRIPTION_MAX = 400


def description_problem(text: Optional[str]) -> Optional[str]:
    """None when the description is well formed, else why it is not."""
    text = (text or "").strip()
    if not text:
        return "description is empty"
    if "\n" in text:
        return "description must be one line"
    if len(text) > DESCRIPTION_MAX:
        return f"description over {DESCRIPTION_MAX} chars"
    if not DESCRIPTION_RE.match(text):
        return (
            "description must read '<what this is> — <scope>; read when "
            "<trigger>.' (em dash, semicolon, trailing period)"
        )
    return None


# ── Bullet voice (§1.3) ───────────────────────────────────────────────
# Subjectless telegraphic third person. The file's subject is implied and
# never restated. Deterministic lint only — the rest lives in the prompt
# and in WS-2's eval set.

# Two, not three. "likes Googoosh" is a complete fact about a life and the
# design doc's own worked example; a three-word floor rejected it while the
# only shape the floor exists to catch — a dangling half-predicate ("wants
# to", "allergic to") — is already one or two words. Raising it back costs
# real bullets and catches nothing extra.
BULLET_MIN_WORDS = 2
BULLET_MAX_CHARS = 400

_LEADING_SUBJECT_RE = re.compile(r"^(?:you\b|your\b|the\s+user\b)", re.IGNORECASE)
_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
# A bare hex run this long is an internal id, never a word or a number a
# person would say. 16 keeps ordinary decimals (a phone number, a year)
# and hex-looking words ("deadbeef" is 8) out of the net.
_HEX_ID_RE = re.compile(r"\b[0-9a-fA-F]{16,}\b")
# `max_results=1`, `limit=20` — a tool parameter, not a fact about a life.
_PARAM_RE = re.compile(r"\b[\w.]+=[\w.\"'-]+")


def normalize_bullet(text: Optional[str]) -> str:
    """Drop the full stop a single-sentence bullet must not carry.

    The contract has always said "no trailing period unless the bullet holds
    more than one sentence", and nothing enforced it — so the rule was a
    preference the model followed about half the time. The founder's migrated
    Profile came back as four capitalised sentences each ending in a period
    while the same writer's turn path produces "listens to Googoosh and Ebi
    constantly", and one corpus in two voices is what acceptance criterion
    "one consistent voice" forbids.

    NORMALISED, not rejected. A rejection costs the retry and can lose the
    fact outright — the lesson of CI 32430971208, where four turns were
    answered with a description and nothing was stored. Punctuation is not
    worth a fact.

    CASE IS DELIBERATELY LEFT ALONE. Lower-casing the first character cannot
    be done safely: "IELTS exam booked …" and "Nariman's sister …" both open
    on a word that must keep its capital, and no rule separates those from an
    ordinary verb without guessing. The prompt asks; this does not enforce.
    """
    t = (text or "").rstrip()
    if not t.endswith("."):
        return t
    body = t[:-1]
    # Any earlier terminator means the bullet really does hold more than one
    # sentence and the final stop belongs there. This also covers a trailing
    # "..." — the second dot is an earlier terminator — so an ellipsis needs
    # no clause of its own. One was written, and a mutation proved it dead.
    if any(ch in body for ch in ".!?"):
        return t
    return body


def bullet_problem(text: Optional[str]) -> Optional[str]:
    """None when the bullet passes the deterministic voice lint."""
    text = (text or "").strip()
    if not text:
        return "bullet is empty"
    if "\n" in text:
        return "a bullet is one line — split it or join with a semicolon"
    if len(text) > BULLET_MAX_CHARS:
        return f"bullet over {BULLET_MAX_CHARS} chars"
    if len(text.split()) < BULLET_MIN_WORDS:
        return (
            f"a bullet of fewer than {BULLET_MIN_WORDS} words is a fragment, "
            "not a fact — write the whole predicate"
        )
    if _LEADING_SUBJECT_RE.match(text):
        return (
            "the file's subject is implied — drop the leading 'You'/'Your'/"
            "'The user' and start with the predicate"
        )
    if _UUID_RE.search(text) or _HEX_ID_RE.search(text):
        return "internal ids are never stored in a memory file"
    if _PARAM_RE.search(text):
        return "tool parameters are never stored in a memory file"
    return None


# ── Bodies ────────────────────────────────────────────────────────────
# `body_md` is a markdown bullet list. Current context additionally uses
# `##` layer headings with prose beneath (§6) — the bullet helpers below
# leave any non-bullet line untouched, so a heading survives a round trip.

MAX_BODY_CHARS = 8 * 1024

_BULLET_LINE_RE = re.compile(r"^\s*[-*]\s+(.*)$")
LINK_RE = re.compile(r"\[\[([^\[\]]+)\]\]")


def is_bullet_list(body_md: Optional[str]) -> bool:
    """True when every non-blank line of a body is a bullet.

    False for Current context, whose body is `##` layer headings with prose
    beneath (§6). `parse_bullets` cannot see that prose, so re-rendering
    such a body from its bullets would DELETE the layers — callers that
    rewrite a body must ask this first.
    """
    for line in (body_md or "").splitlines():
        if line.strip() and not _BULLET_LINE_RE.match(line):
            return False
    return True


def parse_bullets(body_md: Optional[str]) -> List[str]:
    """The bullet texts of a body, in order. Non-bullet lines are ignored."""
    out: List[str] = []
    for line in (body_md or "").splitlines():
        m = _BULLET_LINE_RE.match(line)
        if m and m.group(1).strip():
            out.append(m.group(1).strip())
    return out


def render_bullets(bullets: Iterable[str]) -> str:
    """A body from bullet texts. Empty in, empty out (never a stray '- ')."""
    lines = [f"- {b.strip()}" for b in bullets if (b or "").strip()]
    return "\n".join(lines)


def body_is_empty(body_md: Optional[str]) -> bool:
    return not (body_md or "").strip()


def extract_links(body_md: Optional[str]) -> List[str]:
    """`[[slug]]` targets in a body, de-duplicated, in first-seen order."""
    seen: List[str] = []
    for raw in LINK_RE.findall(body_md or ""):
        slug = raw.strip()
        if slug and slug not in seen:
            seen.append(slug)
    return seen


TRUNCATION_NOTE = "…(truncated — open with memory_read_file)"


def truncate_body(body_md: str, cap_chars: int) -> str:
    """Trim a body to a budget on BULLET boundaries.

    Half a bullet is a false fact, not a short one: "allergic to" reads as
    a complete predicate. So the cut lands between lines and says so.
    """
    body_md = body_md or ""
    if cap_chars <= 0 or len(body_md) <= cap_chars:
        return body_md
    kept: List[str] = []
    used = 0
    budget = max(0, cap_chars - len(TRUNCATION_NOTE) - 1)
    for line in body_md.splitlines():
        if used + len(line) + 1 > budget:
            break
        kept.append(line)
        used += len(line) + 1
    kept.append(TRUNCATION_NOTE)
    return "\n".join(kept)


# ── Injection render (§3.1) ───────────────────────────────────────────
# ONE renderer, two assemblers: agent_runner's `# User Brain` block and
# voice_context's. Forgetting the second assembler is a documented
# prod-incident class, so there is only one of these.

CAP_PROFILE = 2400
CAP_CURRENT_CONTEXT = 3200
CAP_LEARNED = 1600
CAP_RELEVANT_FILE = 2400
MAX_INDEX_LINES = 40
MAX_RELEVANT_FILES = 2

HEADING_PROFILE = "## Profile"
HEADING_CURRENT_CONTEXT = "## Current context"
HEADING_LEARNED = "## Learned (how to work with this user)"
HEADING_INDEX = "## Memory files"


def index_line(title: str, description: Optional[str]) -> str:
    """One line of the file index. The description IS the signal — round
    8's index dropped it, so the model had a list of filenames and no
    reason to open any."""
    return f"- {title} — {description}" if description else f"- {title}"


def render_user_brain(
    *,
    profile_body: str = "",
    current_context_body: str = "",
    learned_body: str = "",
    index: Sequence[Tuple[str, Optional[str]]] = (),
    relevant: Sequence[Tuple[str, str]] = (),
    cap_profile: int = CAP_PROFILE,
    cap_current_context: int = CAP_CURRENT_CONTEXT,
    cap_learned: int = CAP_LEARNED,
    cap_relevant: int = CAP_RELEVANT_FILE,
    max_index: int = MAX_INDEX_LINES,
    max_relevant: int = MAX_RELEVANT_FILES,
) -> str:
    """The body of the memory block, WITHOUT its `# User Brain` heading.

    Each caller prepends its own heading — agent_runner's is the literal
    the injection fence binds to.
    """
    parts: List[str] = []
    if (profile_body or "").strip():
        parts.append(f"{HEADING_PROFILE}\n{truncate_body(profile_body.strip(), cap_profile)}")
    if (current_context_body or "").strip():
        # NOT `truncate_body`: that cuts between bullet LINES and appends the
        # "open with memory_read_file" note. Current context is layers of
        # prose, there is no longer version to open, and a line-boundary cut
        # can leave `## This month` or `### Aug 2026` heading nothing at all.
        parts.append(
            f"{HEADING_CURRENT_CONTEXT}\n"
            f"{trim_current_context(current_context_body.strip(), cap_current_context)}"
        )
    if (learned_body or "").strip():
        parts.append(f"{HEADING_LEARNED}\n{truncate_body(learned_body.strip(), cap_learned)}")
    lines = [index_line(t, d) for t, d in list(index)[:max_index]]
    if lines:
        parts.append(HEADING_INDEX + "\n" + "\n".join(lines))
    for title, body in list(relevant)[:max_relevant]:
        if (body or "").strip():
            parts.append(f"## {title}\n{truncate_body(body.strip(), cap_relevant)}")
    return "\n\n".join(parts)


# ── Current context layers (§6 — WS-3 owns the content) ───────────────
#
# The one file whose body is NOT a bullet list. Six `##` layers of connected
# prose, oldest last, and the last of them carries `### <Mon YYYY>` month
# paragraphs. It is a situation report, not a memory file: nothing in here is
# a durable fact, and nothing in here is written by the curator
# (`FileState.has_prose` refuses every bullet op on it).
#
# This serialisation is the contract the CLIENTS parse — mobile
# `memoryModel.ts::parseContextLayers` (vendored byte-exact to web) reads
# `^\s*(#{1,6})\s+(.*)$` for a heading and splits a trailing `(…)` off the
# text as the layer's note. Two consequences, both load-bearing: a note may
# never contain parentheses, and a `###` line is a SUB-heading of the layer
# above it rather than a layer of its own.

LAYER_TODAY = "Today"
LAYER_YESTERDAY = "Yesterday"
LAYER_LAST_2_DAYS = "Last 2 days"
LAYER_THIS_WEEK = "This week"
LAYER_THIS_MONTH = "This month"
LAYER_PAST_12_MONTHS = "Past 12 months"

CURRENT_CONTEXT_LAYERS: Tuple[str, ...] = (
    LAYER_TODAY,
    LAYER_YESTERDAY,
    LAYER_LAST_2_DAYS,
    LAYER_THIS_WEEK,
    LAYER_THIS_MONTH,
    LAYER_PAST_12_MONTHS,
)

#: The five prose layers, newest first. `Past 12 months` is not one of them —
#: it holds `### <Mon YYYY>` paragraphs, not prose of its own.
PROSE_LAYERS: Tuple[str, ...] = CURRENT_CONTEXT_LAYERS[:5]

#: Per-layer character budgets (§6), enforced in CODE. A prompt rule alone
#: is a suggestion; these are the numbers the injected render is sized on.
LAYER_BUDGETS: Dict[str, int] = {
    LAYER_TODAY: 600,
    LAYER_YESTERDAY: 400,
    LAYER_LAST_2_DAYS: 300,
    LAYER_THIS_WEEK: 400,
    LAYER_THIS_MONTH: 500,
}
MONTH_PARAGRAPH_MAX = 300
MAX_MONTH_PARAGRAPHS = 12

_MONTH_NAMES = (
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
)
_MONTH_KEY_RE = re.compile(
    r"^(" + "|".join(_MONTH_NAMES) + r")\s+(\d{4})$", re.IGNORECASE
)
_HEADING_RE = re.compile(r"^\s*(#{1,6})\s+(.*)$")
#: The client splits a trailing `(…)` off a heading as the layer's note.
_HEADING_NOTE_RE = re.compile(r"^(.*?)\s*\(([^()]*)\)\s*$")
# `.` `!` `?` plus the Persian question mark and the CJK full stop — the
# scripts this product actually stores.
_SENTENCE_END_RE = re.compile(r"[.!?؟。]['\"”’)\]]*\s")


def month_key(when: date) -> str:
    """`Aug 2026` — the label of a `### ` month paragraph."""
    return f"{_MONTH_NAMES[when.month - 1]} {when.year}"


def parse_month_key(label: str) -> Optional[Tuple[int, int]]:
    """`(year, month)` for a month label, or None when it is not one."""
    m = _MONTH_KEY_RE.match((label or "").strip())
    if not m:
        return None
    return int(m.group(2)), _MONTH_NAMES.index(m.group(1)[:3].title()) + 1


def clamp_prose(text: Optional[str], cap: int) -> str:
    """Trim prose to a budget WITHOUT ending mid-sentence.

    `truncate_body` cuts between bullet lines because half a bullet reads as
    a complete predicate; a layer is one paragraph, so the same reasoning
    lands on the sentence. Cut at the last sentence end inside the budget;
    with no sentence end to find, cut at a word boundary and say so with an
    ellipsis — a layer that stops mid-word is visibly truncated, which is
    the honest failure. A layer never carries the `memory_read_file` note:
    there is no longer version of it to open.
    """
    text = (text or "").strip()
    if cap <= 0:
        return ""
    if len(text) <= cap:
        return text
    window = text[: cap + 1]
    last_end = 0
    for m in _SENTENCE_END_RE.finditer(window):
        if m.end() <= cap:
            last_end = m.end()
    # Also accept a terminator that lands exactly on the budget's last char.
    if len(window) > cap and window[cap - 1] in ".!?؟。":
        last_end = max(last_end, cap)
    if last_end:
        return window[:last_end].strip()
    cut = text[: max(0, cap - 1)]
    space = cut.rfind(" ")
    if space > cap * 0.5:
        cut = cut[:space]
    return cut.rstrip(" ,;:—-") + "…"


@dataclass
class CurrentContext:
    """The parsed body of `you/current-context`."""

    #: layer title → prose. Only layers that have something to say.
    layers: Dict[str, str] = field(default_factory=dict)
    #: `[(label, paragraph)]`, most recent month first.
    months: List[Tuple[str, str]] = field(default_factory=list)
    #: The `## Today (…)` parenthetical, e.g. "Wed, Aug 20 — America/Toronto".
    today_note: Optional[str] = None

    def get(self, layer: str) -> str:
        return self.layers.get(layer, "")

    def set(self, layer: str, text: Optional[str]) -> None:
        """Write a layer, clamped to its budget. Empty removes it."""
        clamped = clamp_prose(text, LAYER_BUDGETS.get(layer, MONTH_PARAGRAPH_MAX))
        if clamped:
            self.layers[layer] = clamped
        else:
            self.layers.pop(layer, None)

    def is_empty(self) -> bool:
        return not self.layers and not self.months


def parse_current_context(body_md: Optional[str]) -> CurrentContext:
    """Read a Current-context body back into layers and months.

    Tolerant by construction: an unknown `##` heading, prose before the
    first heading, and a `### <Mon YYYY>` that is not under `Past 12 months`
    are all kept rather than dropped. A body the writer got wrong must still
    round-trip — losing a user's context to a parse mismatch is the one
    failure this file cannot afford.
    """
    ctx = CurrentContext()
    current_layer: Optional[str] = None
    current_month: Optional[str] = None
    buf: List[str] = []

    def flush() -> None:
        text = " ".join(p.strip() for p in buf if p.strip()).strip()
        buf.clear()
        if not text:
            return
        if current_month is not None:
            ctx.months.append((current_month, text))
        else:
            # `""` is the leading, untitled layer — text that arrived before
            # any heading. The clients keep it for the same reason: a body
            # the writer got wrong must still be readable.
            key = current_layer or ""
            prior = ctx.layers.get(key)
            ctx.layers[key] = f"{prior} {text}".strip() if prior else text

    for raw in (body_md or "").splitlines():
        line = raw.rstrip()
        heading = _HEADING_RE.match(line)
        if heading and len(heading.group(1)) <= 2:
            flush()
            title, note = _split_heading(heading.group(2))
            current_month = None
            current_layer = _canonical_layer(title)
            if current_layer == LAYER_TODAY and note:
                ctx.today_note = note
            continue
        if heading:
            flush()
            current_month = _split_heading(heading.group(2))[0]
            continue
        if not line.strip():
            continue
        # A stray bullet marker is stripped: the clients render this file as
        # prose, so a `- ` that survived into the body would be shown as a
        # literal dash by every surface that reads a layer's text.
        bullet = _BULLET_LINE_RE.match(line)
        buf.append(bullet.group(1) if bullet else line.strip())
    flush()
    return ctx


def _split_heading(text: str) -> Tuple[str, Optional[str]]:
    m = _HEADING_NOTE_RE.match((text or "").strip())
    if m and m.group(1).strip():
        return m.group(1).strip(), m.group(2).strip()
    return (text or "").strip(), None


def _canonical_layer(title: str) -> str:
    """Fold a heading onto its canonical layer name, or keep it as written."""
    low = (title or "").strip().lower()
    for layer in CURRENT_CONTEXT_LAYERS:
        if low == layer.lower():
            return layer
    return (title or "").strip()


def render_current_context(ctx: CurrentContext) -> str:
    """Serialise back to the body both clients parse.

    An EMPTY layer is omitted, heading and all. A body of bare headings is
    not empty to `body_is_empty`, so a brand-new user would read as "has a
    Current context" in the prompt health line, in the injected block and in
    the client's empty state — three lies for zero information.
    """
    parts: List[str] = []
    lead = (ctx.layers.get("") or "").strip()
    if lead:
        parts.append(lead)
    for layer in PROSE_LAYERS:
        text = (ctx.layers.get(layer) or "").strip()
        if not text:
            continue
        heading = f"## {layer}"
        if layer == LAYER_TODAY and ctx.today_note:
            heading = f"## {layer} ({ctx.today_note})"
        parts.append(f"{heading}\n{text}")
    # Any layer the writer invented, after the canonical five, so a body that
    # round-trips through here never silently loses text.
    for layer, text in ctx.layers.items():
        if (layer and layer not in PROSE_LAYERS
                and layer != LAYER_PAST_12_MONTHS and text.strip()):
            parts.append(f"## {layer}\n{text.strip()}")
    if ctx.months:
        block = [f"## {LAYER_PAST_12_MONTHS}"]
        for label, text in ctx.months[:MAX_MONTH_PARAGRAPHS]:
            if text.strip():
                block.append(f"### {label}\n{text.strip()}")
        if len(block) > 1:
            parts.append("\n\n".join(block))
    return "\n\n".join(parts)


def trim_current_context(body_md: str, cap_chars: int) -> str:
    """The INJECTED render of Current context, sized to fit.

    Six layers plus twelve month paragraphs sum to ~5,800 characters, which
    is more than `CAP_CURRENT_CONTEXT`. "Injected in full every reply" means
    the file is always present and never fetched on demand — it does not
    mean every month survives the render.

    Drop order: month paragraphs from the OLDEST end, then whole layers from
    the oldest end. `Today` is never dropped; if it alone is over budget the
    cut lands on a sentence boundary. Nothing here ever ends a layer
    mid-sentence, and nothing ever leaves a heading with no body.
    """
    body_md = (body_md or "").strip()
    if cap_chars <= 0 or len(body_md) <= cap_chars:
        return body_md
    ctx = parse_current_context(body_md)
    if not ctx.months and not any(k for k in ctx.layers):
        # No layer heading anywhere — a hand-written or migrated body that is
        # not in this shape at all. Re-rendering it here would flatten a
        # bullet list into one paragraph, so fall back to the bullet-boundary
        # cut rather than reformatting a body we do not understand.
        return truncate_body(body_md, cap_chars)
    while ctx.months and len(render_current_context(ctx)) > cap_chars:
        ctx.months.pop()
    for layer in reversed(PROSE_LAYERS[1:]):
        if len(render_current_context(ctx)) <= cap_chars:
            break
        ctx.layers.pop(layer, None)
    rendered = render_current_context(ctx)
    if len(rendered) <= cap_chars:
        return rendered
    # Today alone is over budget — only reachable from a hand-written or
    # migrated body, since `CurrentContext.set` clamps every write.
    overhead = len(rendered) - len(ctx.get(LAYER_TODAY))
    ctx.layers[LAYER_TODAY] = clamp_prose(
        ctx.get(LAYER_TODAY), max(0, cap_chars - overhead)
    )
    return render_current_context(ctx)


# ══ LEGACY (round 8) ══════════════════════════════════════════════════
# Retired by WS-2 together with `memory_file_service.py` (row routing) and
# read by WS-5's migration, which must interpret round-8 file assignments
# to move them into v3 files. Nothing above this banner may use it, and no
# NEW code may either: categories are no longer the router.

from app.memory_taxonomy import MemoryCategory, normalize_category  # noqa: E402


class LegacyFileSection(str, Enum):
    PROFILE = "profile"
    PEOPLE = "people"
    AREAS = "areas"
    PREFERENCES = "preferences"
    KNOWLEDGE = "knowledge"
    LEARNED = "learned"
    WORKING = "working"


LEGACY_SECTION_ORDER: List[LegacyFileSection] = [
    LegacyFileSection.PROFILE,
    LegacyFileSection.PEOPLE,
    LegacyFileSection.AREAS,
    LegacyFileSection.PREFERENCES,
    LegacyFileSection.KNOWLEDGE,
    LegacyFileSection.LEARNED,
    LegacyFileSection.WORKING,
]

LEGACY_SYSTEM_FILES: Dict[str, Dict[str, str]] = {
    "profile": {"section": "profile", "title": "Profile", "purpose": "Who you are."},
    "people": {"section": "people", "title": "People", "purpose": "People in your life."},
    "areas/work": {"section": "areas", "title": "Work & goals", "purpose": "Your work and goals."},
    "preferences": {"section": "preferences", "title": "Preferences", "purpose": "What you like."},
    "knowledge": {"section": "knowledge", "title": "Knowledge", "purpose": "Facts about your world."},
    "learned": {"section": "learned", "title": "Learned", "purpose": "Corrections and lessons."},
    "working": {"section": "working", "title": "Working on", "purpose": "What's in motion now."},
}

LEGACY_USER_CATEGORY_SECTION: Dict[str, LegacyFileSection] = {
    MemoryCategory.IDENTITY.value: LegacyFileSection.PROFILE,
    MemoryCategory.BELIEFS.value: LegacyFileSection.PROFILE,
    MemoryCategory.EMOTIONS.value: LegacyFileSection.PROFILE,
    MemoryCategory.HEALTH.value: LegacyFileSection.PROFILE,
    MemoryCategory.HABITS.value: LegacyFileSection.PROFILE,
    MemoryCategory.PEOPLE.value: LegacyFileSection.PEOPLE,
    MemoryCategory.RELATIONSHIPS.value: LegacyFileSection.PEOPLE,
    MemoryCategory.WORK.value: LegacyFileSection.AREAS,
    MemoryCategory.GOALS.value: LegacyFileSection.AREAS,
    MemoryCategory.FINANCE.value: LegacyFileSection.AREAS,
    MemoryCategory.SKILLS.value: LegacyFileSection.AREAS,
    MemoryCategory.PREFERENCES.value: LegacyFileSection.PREFERENCES,
    MemoryCategory.INTERACTION.value: LegacyFileSection.PREFERENCES,
    MemoryCategory.KNOWLEDGE.value: LegacyFileSection.KNOWLEDGE,
    MemoryCategory.MEDIA.value: LegacyFileSection.KNOWLEDGE,
    MemoryCategory.EXPERIENCES.value: LegacyFileSection.KNOWLEDGE,
    MemoryCategory.LOCATIONS.value: LegacyFileSection.KNOWLEDGE,
    MemoryCategory.POSSESSIONS.value: LegacyFileSection.KNOWLEDGE,
    MemoryCategory.ACTIVE_TASK.value: LegacyFileSection.WORKING,
    MemoryCategory.OTHER.value: LegacyFileSection.KNOWLEDGE,
}

LEGACY_SECTION_DEFAULT_CATEGORY: Dict[LegacyFileSection, str] = {
    LegacyFileSection.PROFILE: MemoryCategory.IDENTITY.value,
    LegacyFileSection.PEOPLE: MemoryCategory.PEOPLE.value,
    LegacyFileSection.AREAS: MemoryCategory.WORK.value,
    LegacyFileSection.PREFERENCES: MemoryCategory.PREFERENCES.value,
    LegacyFileSection.KNOWLEDGE: MemoryCategory.KNOWLEDGE.value,
    LegacyFileSection.LEARNED: "user_patterns",
    LegacyFileSection.WORKING: MemoryCategory.ACTIVE_TASK.value,
}

LEGACY_DEFAULT_SLUG_FOR_SECTION: Dict[LegacyFileSection, str] = {
    LegacyFileSection.PROFILE: "profile",
    LegacyFileSection.PEOPLE: "people",
    LegacyFileSection.AREAS: "areas/work",
    LegacyFileSection.PREFERENCES: "preferences",
    LegacyFileSection.KNOWLEDGE: "knowledge",
    LegacyFileSection.LEARNED: "learned",
    LegacyFileSection.WORKING: "working",
}


def legacy_section_for(
    category: Optional[str], brain_type: Optional[str] = "user"
) -> LegacyFileSection:
    if brain_type == "agent":
        return LegacyFileSection.LEARNED
    raw = (category or "").strip().lower()
    if brain_type == "work":
        return (
            LegacyFileSection.WORKING
            if raw == MemoryCategory.ACTIVE_TASK.value
            else LegacyFileSection.LEARNED
        )
    canonical = normalize_category(raw, brain_type="user") if raw else MemoryCategory.OTHER.value
    return LEGACY_USER_CATEGORY_SECTION.get(canonical, LegacyFileSection.KNOWLEDGE)


def legacy_default_slug_for(
    category: Optional[str], brain_type: Optional[str] = "user"
) -> str:
    return LEGACY_DEFAULT_SLUG_FOR_SECTION[legacy_section_for(category, brain_type)]


def legacy_section_of_slug(slug: str) -> LegacyFileSection:
    if slug in LEGACY_SYSTEM_FILES:
        return LegacyFileSection(LEGACY_SYSTEM_FILES[slug]["section"])
    head = slug.split("/", 1)[0]
    if head == "people":
        return LegacyFileSection.PEOPLE
    if head == "areas":
        return LegacyFileSection.AREAS
    return LegacyFileSection.KNOWLEDGE
